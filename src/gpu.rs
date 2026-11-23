//! The GPU video path on D3D11: decode, luma pyramid, motion search, interpolation (compute
//! shaders in gpu.hlsl) and the final fit-to-window draw into our own swap chain.
//!
//! The swap chain is a DirectComposition visual on the window's *non-topmost* target, so it
//! sits underneath gpui's layer (which gpui clears to transparent) and the controls draw on top.

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::{HMODULE, HWND};
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
    ID3DBlob,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::DirectComposition::{
    DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget, IDCompositionVisual,
};
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::{Interface, PCSTR};

use crate::convert::{self, Matrix};
use crate::quality::Quality;
use crate::video::Layout;

const SOURCE: &str = include_str!("gpu.hlsl");

// Must match gpu.hlsl.
const F_HDR: u32 = 1;
const F_GAMUT: u32 = 2;
const F_TEMPORAL: u32 = 4;
const F_BLACK: u32 = 8;
const F_SEED: u32 = 16;
const F_COUNT: u32 = 32;
const F_MARK: u32 = 64;
const F_PROP: u32 = 128;

/// The constant buffer every pass reads (gpu.hlsl's `Params`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    sz: [i32; 4],
    gr: [i32; 4],
    kc: [f32; 4],
    misc: [f32; 4],
    fl: [u32; 4],
    rect: [f32; 4],
}

/// How to turn the source's YCbCr into display RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoding {
    pub layout: Layout,
    /// Treat as HDR10 (PQ) and tone map.
    pub hdr: bool,
    pub matrix: Matrix,
    /// Convert BT.2020 primaries to BT.709 (HDR only).
    pub gamut_2020: bool,
    pub full_range: bool,
    /// HDR brightness shown as SDR white.
    pub white_nits: f32,
}

/// A compute shader and its entry point's name (for profiling).
struct Kernel {
    cs: ID3D11ComputeShader,
    name: &'static str,
}

struct Kernels {
    decode: Kernel,
    luma0: Kernel,
    halve: Kernel,
    flow_full: Kernel,
    flow_patch: Kernel,
    flow_dense: Kernel,
    flow_smooth: Kernel,
    flow_snap: Kernel,
    flow_check: Kernel,
    flow_inplace: Kernel,
    flow_edges: Kernel,
    flow_held: Kernel,
    flow_fill: Kernel,
    flow_final: Kernel,
    project: Kernel,
    /// By `Quality`.
    compose: [Kernel; 3],
    luma8: Kernel,
    hw_dense: Kernel,
    hw_seed: Kernel,
}

pub struct Gpu {
    pub device: ID3D11Device,
    pub ctx: ID3D11DeviceContext,
    cb: ID3D11Buffer,
    lin: ID3D11SamplerState,
    k: Kernels,
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    /// While profiling: timestamps around every dispatch.
    prof: std::cell::RefCell<Option<Profile>>,
}

/// How long pieces of work took the GPU, by its own clock, picked up without ever waiting.
pub struct Timer<T> {
    sets: Vec<Stamps>,
    /// Ended and not read back yet, oldest first.
    running: std::collections::VecDeque<(usize, T)>,
    free: Vec<usize>,
}

struct Stamps {
    disjoint: ID3D11Query,
    begin: ID3D11Query,
    end: ID3D11Query,
}

/// Pieces timed at once at most; any more just go untimed.
const TIMED: usize = 32;

impl<T> Timer<T> {
    /// Without timestamps (a driver that has none) nothing is ever timed.
    pub fn new(gpu: &Gpu) -> Timer<T> {
        let make = || -> Option<Stamps> {
            let desc = D3D11_QUERY_DESC { Query: D3D11_QUERY_TIMESTAMP_DISJOINT, MiscFlags: 0 };
            let mut q = None;
            unsafe { gpu.device.CreateQuery(&desc, Some(&mut q)).ok()? };
            Some(Stamps { disjoint: q?, begin: gpu.timestamp()?, end: gpu.timestamp()? })
        };
        let sets: Vec<Stamps> = (0..TIMED).map_while(|_| make()).collect();
        Timer { free: (0..sets.len()).collect(), sets, running: Default::default() }
    }

    /// Runs `work`, timed as `tag`. Timed pieces don't nest.
    pub fn time<R>(&mut self, gpu: &Gpu, tag: T, work: impl FnOnce() -> R) -> R {
        let Some(i) = self.free.pop() else { return work() };
        unsafe {
            gpu.ctx.Begin(&self.sets[i].disjoint);
            gpu.ctx.End(&self.sets[i].begin);
        }
        let out = work();
        unsafe {
            gpu.ctx.End(&self.sets[i].end);
            gpu.ctx.End(&self.sets[i].disjoint);
        }
        self.running.push_back((i, tag));
        out
    }

    /// The oldest piece the GPU has finished with a time to go by.
    pub fn poll(&mut self, gpu: &Gpu) -> Option<(T, std::time::Duration)> {
        loop {
            let &(i, _) = self.running.front()?;
            let s = &self.sets[i];
            let mut dj = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
            // "Not yet" reads as success and leaves the data alone (no frequency then).
            if !gpu.peek(&s.disjoint, &mut dj) || dj.Frequency == 0 {
                return None;
            }
            let (mut from, mut to) = (0u64, 0u64);
            let read = gpu.peek(&s.begin, &mut from) && gpu.peek(&s.end, &mut to);
            let (i, tag) = self.running.pop_front()?;
            self.free.push(i);
            if read && !dj.Disjoint.as_bool() && from != 0 && to > from {
                let secs = (to - from) as f64 / dj.Frequency as f64;
                return Some((tag, std::time::Duration::from_secs_f64(secs)));
            }
        }
    }
}

struct Profile {
    disjoint: ID3D11Query,
    marks: Vec<(&'static str, ID3D11Query, ID3D11Query)>,
}

/// A 2D texture with the views it was made with.
pub struct Tex {
    pub(crate) tex: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
    uav: Option<ID3D11UnorderedAccessView>,
    pub w: u32,
    pub h: u32,
}

fn compile(entry: &str, target: &str) -> Result<Vec<u8>> {
    let entry_c = format!("{entry}\0");
    let target_c = format!("{target}\0");
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // Development: PASSTHRU_HLSL=FILE compiles that file instead of the built-in shaders
    // (tuning against recorded clips without rebuilding).
    let file = std::env::var("PASSTHRU_HLSL").ok().and_then(|p| std::fs::read_to_string(p).ok());
    let source = file.as_deref().unwrap_or(SOURCE);
    let r = unsafe {
        D3DCompile(
            source.as_ptr() as _,
            source.len(),
            PCSTR(c"gpu.hlsl".as_ptr() as _),
            None,
            None,
            PCSTR(entry_c.as_ptr()),
            PCSTR(target_c.as_ptr()),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut blob,
            Some(&mut errors),
        )
    };
    let bytes = |b: &ID3DBlob| unsafe {
        std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()
    };
    if let Err(e) = r {
        let msg = errors.as_ref().map(|b| String::from_utf8_lossy(&bytes(b)).into_owned()).unwrap_or_default();
        bail!("compiling {entry}: {e} {msg}");
    }
    Ok(bytes(&blob.context("no shader blob")?))
}

fn groups(n: u32, size: u32) -> u32 {
    n.div_ceil(size)
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let mut device = None;
        let mut ctx = None;
        // Video support is for the upscaler (the video processor); fall back to a device without it.
        let create = |flags, device: &mut Option<ID3D11Device>, ctx: &mut Option<ID3D11DeviceContext>| unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                flags,
                Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(device),
                None,
                Some(ctx),
            )
        };
        if create(D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT, &mut device, &mut ctx).is_err() {
            create(D3D11_CREATE_DEVICE_BGRA_SUPPORT, &mut device, &mut ctx).context("D3D11CreateDevice")?;
        }
        let device: ID3D11Device = device.context("no device")?;
        let ctx = ctx.context("no context")?;

        // Compile everything in parallel: the motion kernels take a while each.
        let cs_names = [
            "decode",
            "luma0",
            "halve",
            "flow_full",
            "flow_patch",
            "flow_dense",
            "flow_smooth",
            "flow_snap",
            "flow_check",
            "flow_inplace",
            "flow_edges",
            "flow_held",
            "flow_fill",
            "flow_final",
            "project",
            "compose",
            "compose_balanced",
            "compose_high",
            "luma8",
            "hw_dense",
            "hw_seed",
        ];
        let compiled: Vec<Result<Vec<u8>>> = std::thread::scope(|s| {
            let jobs: Vec<_> = cs_names
                .iter()
                .map(|n| (n, "cs_5_0"))
                .chain([(&"present_vs", "vs_5_0"), (&"present_ps", "ps_5_0")])
                .map(|(n, t)| s.spawn(move || compile(n, t)))
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap_or_else(|_| bail!("compiler thread panicked"))).collect()
        });
        let mut code = compiled.into_iter().collect::<Result<Vec<_>>>()?.into_iter();
        let mut names = cs_names.iter();
        let mut cs = || -> Result<Kernel> {
            let bytes = code.next().context("missing shader")?;
            let mut s = None;
            unsafe { device.CreateComputeShader(&bytes, None, Some(&mut s))? };
            Ok(Kernel { cs: s.context("no compute shader")?, name: names.next().context("missing name")? })
        };
        let k = Kernels {
            decode: cs()?,
            luma0: cs()?,
            halve: cs()?,
            flow_full: cs()?,
            flow_patch: cs()?,
            flow_dense: cs()?,
            flow_smooth: cs()?,
            flow_snap: cs()?,
            flow_check: cs()?,
            flow_inplace: cs()?,
            flow_edges: cs()?,
            flow_held: cs()?,
            flow_fill: cs()?,
            flow_final: cs()?,
            project: cs()?,
            compose: [cs()?, cs()?, cs()?],
            luma8: cs()?,
            hw_dense: cs()?,
            hw_seed: cs()?,
        };
        let (vs_code, ps_code) = (code.next().context("vs")?, code.next().context("ps")?);
        let (mut vs, mut ps) = (None, None);
        unsafe {
            device.CreateVertexShader(&vs_code, None, Some(&mut vs))?;
            device.CreatePixelShader(&ps_code, None, Some(&mut ps))?;
        }

        let desc = D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<Params>() as u32,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        let mut cb = None;
        unsafe { device.CreateBuffer(&desc, None, Some(&mut cb))? };
        let sampler = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_NEVER,
            BorderColor: [0.0; 4],
            MinLOD: 0.0,
            MaxLOD: D3D11_FLOAT32_MAX,
        };
        let mut lin = None;
        unsafe { device.CreateSamplerState(&sampler, Some(&mut lin))? };
        Ok(Gpu {
            device,
            ctx,
            cb: cb.context("no constant buffer")?,
            lin: lin.context("no sampler")?,
            k,
            vs: vs.context("no vertex shader")?,
            ps: ps.context("no pixel shader")?,
            prof: std::cell::RefCell::new(None),
        })
    }

    pub fn tex(&self, w: u32, h: u32, format: DXGI_FORMAT, uav: bool) -> Result<Tex> {
        self.tex_bound(w, h, format, uav, false)
    }

    /// `tex`, also bindable as a render target (what the optical flow API asks for).
    fn tex_bound(&self, w: u32, h: u32, format: DXGI_FORMAT, uav: bool, rt: bool) -> Result<Tex> {
        let mut bind = D3D11_BIND_SHADER_RESOURCE.0 as u32;
        if uav {
            bind |= D3D11_BIND_UNORDERED_ACCESS.0 as u32;
        }
        if rt {
            bind |= D3D11_BIND_RENDER_TARGET.0 as u32;
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w.max(1),
            Height: h.max(1),
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut tex = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut tex))? };
        let tex: ID3D11Texture2D = tex.context("no texture")?;
        let mut srv = None;
        unsafe { self.device.CreateShaderResourceView(&tex, None, Some(&mut srv))? };
        let mut u = None;
        if uav {
            unsafe { self.device.CreateUnorderedAccessView(&tex, None, Some(&mut u))? };
        }
        Ok(Tex { tex, srv, uav: u, w: w.max(1), h: h.max(1) })
    }

    /// A texture that can be drawn into (render target) and read by shaders.
    pub fn target(&self, w: u32, h: u32, format: DXGI_FORMAT) -> Result<(Tex, ID3D11RenderTargetView)> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w.max(1),
            Height: h.max(1),
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut tex = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut tex))? };
        let tex: ID3D11Texture2D = tex.context("no texture")?;
        let (mut srv, mut rtv) = (None, None);
        unsafe {
            self.device.CreateShaderResourceView(&tex, None, Some(&mut srv))?;
            self.device.CreateRenderTargetView(&tex, None, Some(&mut rtv))?;
        }
        Ok((Tex { tex, srv, uav: None, w: w.max(1), h: h.max(1) }, rtv.context("no render target view")?))
    }

    /// Draws `src` (or black) fitted into `rect` (x, y, w, h in pixels) on a `size` render target.
    /// Draws `src` (or black) fitted into `rect`; `mark` puts that number's barcode in the
    /// top-left corner (test sessions recording the screen).
    pub fn draw(
        &self,
        rtv: &ID3D11RenderTargetView,
        size: (u32, u32),
        src: Option<&Tex>,
        rect: [f32; 4],
        mark: Option<u32>,
    ) {
        let mut flags = if src.is_none() { F_BLACK } else { 0 };
        if mark.is_some() {
            flags |= F_MARK;
        }
        let p = Params {
            sz: [0, 0, src.map_or(1, |t| t.w as i32), src.map_or(1, |t| t.h as i32)],
            fl: [flags, 0, 0, mark.unwrap_or(0)],
            rect,
            ..Default::default()
        };
        self.set_params(&p);
        let ctx = &self.ctx;
        unsafe {
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size.0 as f32,
                Height: size.1 as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.PSSetShader(&self.ps, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.PSSetSamplers(0, Some(&[Some(self.lin.clone())]));
            ctx.PSSetShaderResources(19, Some(&[src.and_then(|t| t.srv.clone())]));
            ctx.Draw(3, 0);
            ctx.PSSetShaderResources(19, Some(&[None]));
            ctx.OMSetRenderTargets(None, None);
        }
    }

    fn set_params(&self, p: &Params) {
        unsafe {
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            if self.ctx.Map(&self.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut m)).is_ok() {
                std::ptr::copy_nonoverlapping(p as *const Params, m.pData as *mut Params, 1);
                self.ctx.Unmap(&self.cb, 0);
            }
        }
    }

    /// Binds inputs (SRV slot, texture) and outputs (UAV slot, texture), dispatches, unbinds.
    fn run(&self, cs: &Kernel, p: &Params, srvs: &[(u32, &Tex)], uavs: &[(u32, &Tex)], g: (u32, u32)) {
        self.run_z(cs, p, srvs, uavs, (g.0, g.1, 1));
    }

    /// `run` with a third dispatch dimension.
    fn run_z(&self, k: &Kernel, p: &Params, srvs: &[(u32, &Tex)], uavs: &[(u32, &Tex)], g: (u32, u32, u32)) {
        self.set_params(p);
        let ctx = &self.ctx;
        unsafe {
            ctx.CSSetShader(&k.cs, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.CSSetSamplers(0, Some(&[Some(self.lin.clone())]));
            for (slot, t) in srvs {
                ctx.CSSetShaderResources(*slot, Some(std::slice::from_ref(&t.srv)));
            }
            for (slot, t) in uavs {
                ctx.CSSetUnorderedAccessViews(*slot, 1, Some(&t.uav.clone()), None);
            }
            let marks = self.prof.borrow().as_ref().map(|_| (self.timestamp(), self.timestamp()));
            if let Some((Some(t0), _)) = &marks {
                ctx.End(t0);
            }
            ctx.Dispatch(g.0.max(1), g.1.max(1), g.2.max(1));
            if let Some((Some(t0), Some(t1))) = marks {
                ctx.End(&t1);
                if let Some(p) = self.prof.borrow_mut().as_mut() {
                    p.marks.push((k.name, t0, t1));
                }
            }
            for (slot, _) in srvs {
                ctx.CSSetShaderResources(*slot, Some(&[None]));
            }
            for (slot, _) in uavs {
                ctx.CSSetUnorderedAccessViews(*slot, 1, Some(&None), None);
            }
        }
    }

    /// Copies a texture back to the CPU, rows packed (`bpp` bytes per pixel).
    /// A CPU-readable texture the size and format of `t`.
    fn staging_like(&self, t: &Tex) -> Result<ID3D11Texture2D> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { t.tex.GetDesc(&mut desc) };
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        desc.MiscFlags = 0;
        let mut staging = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut staging))? };
        staging.context("no staging texture")
    }

    pub fn read(&self, t: &Tex, bpp: usize) -> Result<Vec<u8>> {
        let staging = self.staging_like(t)?;
        let row = t.w as usize * bpp;
        let mut out = vec![0u8; row * t.h as usize];
        unsafe {
            self.ctx.CopyResource(&staging, &t.tex);
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            self.ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m))?;
            for y in 0..t.h as usize {
                let src = std::slice::from_raw_parts((m.pData as *const u8).add(y * m.RowPitch as usize), row);
                out[y * row..(y + 1) * row].copy_from_slice(src);
            }
            self.ctx.Unmap(&staging, 0);
        }
        Ok(out)
    }

    /// A query's data as it stands, without flushing or waiting.
    fn peek<T>(&self, q: &ID3D11Query, into: &mut T) -> bool {
        unsafe {
            self.ctx.GetData(
                q,
                Some(into as *mut T as _),
                std::mem::size_of::<T>() as u32,
                D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
            )
        }
        .is_ok()
    }

    fn timestamp(&self) -> Option<ID3D11Query> {
        let desc = D3D11_QUERY_DESC { Query: D3D11_QUERY_TIMESTAMP, MiscFlags: 0 };
        let mut q = None;
        unsafe { self.device.CreateQuery(&desc, Some(&mut q)).ok()? };
        q
    }

    /// Starts timing every dispatch (tests only).
    pub fn profile_begin(&self) -> Result<()> {
        let desc = D3D11_QUERY_DESC { Query: D3D11_QUERY_TIMESTAMP_DISJOINT, MiscFlags: 0 };
        let mut q = None;
        unsafe { self.device.CreateQuery(&desc, Some(&mut q))? };
        let disjoint: ID3D11Query = q.context("no query")?;
        unsafe { self.ctx.Begin(&disjoint) };
        *self.prof.borrow_mut() = Some(Profile { disjoint, marks: Vec::new() });
        Ok(())
    }

    /// Stops timing and returns GPU milliseconds per entry point, in first-use order.
    pub fn profile_end(&self) -> Result<Vec<(&'static str, f64)>> {
        let p = self.prof.borrow_mut().take().context("not profiling")?;
        unsafe { self.ctx.End(&p.disjoint) };
        let get = |q: &ID3D11Query, out: *mut std::ffi::c_void, size: usize| -> Result<()> {
            unsafe { self.ctx.GetData(q, Some(out), size as u32, 0)? };
            Ok(())
        };
        // Not ready reads as success too (S_FALSE) but leaves the data alone: wait for the
        // frequency to appear. Once the outer query is done, every timestamp inside it is.
        let mut dj = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
        while dj.Frequency == 0 {
            get(&p.disjoint, &mut dj as *mut _ as _, std::mem::size_of_val(&dj))?;
            std::thread::yield_now();
        }
        if dj.Disjoint.as_bool() {
            bail!("timestamps unreliable (GPU clock changed)");
        }
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for (name, t0, t1) in &p.marks {
            let (mut a, mut b) = (0u64, 0u64);
            get(t0, &mut a as *mut _ as _, 8)?;
            get(t1, &mut b as *mut _ as _, 8)?;
            let ms = b.saturating_sub(a) as f64 * 1000.0 / dj.Frequency as f64;
            match out.iter_mut().find(|(n, _)| n == name) {
                Some(e) => e.1 += ms,
                None => out.push((name, ms)),
            }
        }
        Ok(out)
    }

    /// A marker after everything submitted so far; `passed` says when the GPU got there.
    pub fn fence(&self) -> Result<ID3D11Query> {
        let desc = D3D11_QUERY_DESC { Query: D3D11_QUERY_EVENT, MiscFlags: 0 };
        let mut q = None;
        unsafe {
            self.device.CreateQuery(&desc, Some(&mut q))?;
            let q: ID3D11Query = q.context("no query")?;
            self.ctx.End(&q);
            self.ctx.Flush();
            Ok(q)
        }
    }

    /// Whether the GPU has reached `fence` (doesn't wait or flush).
    pub fn passed(&self, fence: &ID3D11Query) -> bool {
        let mut done = windows::core::BOOL(0);
        let r = unsafe {
            self.ctx.GetData(
                fence,
                Some(&mut done as *mut _ as _),
                std::mem::size_of_val(&done) as u32,
                D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
            )
        };
        r.is_ok() && done.as_bool()
    }

    /// Blocks until the GPU has finished everything submitted so far.
    pub fn wait_idle(&self) -> Result<()> {
        let desc = D3D11_QUERY_DESC { Query: D3D11_QUERY_EVENT, MiscFlags: 0 };
        let mut q = None;
        unsafe { self.device.CreateQuery(&desc, Some(&mut q))? };
        let q: ID3D11Query = q.context("no query")?;
        unsafe { self.ctx.End(&q) };
        loop {
            let mut done = windows::core::BOOL(0);
            unsafe {
                self.ctx.GetData(&q, Some(&mut done as *mut _ as _), std::mem::size_of_val(&done) as u32, 0)?;
            }
            if done.as_bool() {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}

/// A decoded picture and its luma pyramid (level 0 is 1/`Frames::scale0` size).
pub struct Slot {
    pub frame: Tex,
    luma: Vec<Tex>,
}

/// Everything sized to one video format: upload textures, two picture slots and, while frame
/// generation is on, what it needs (`Gen`) and the generated frames. Without frame generation
/// only the first three are made: at 4K the rest is several hundred megabytes.
pub struct Frames {
    pub w: u32,
    pub h: u32,
    pub layout: Layout,
    /// Full-size pixels per level-0 pixel: 2, or 4 for 4K.
    scale0: u32,
    input: Vec<Tex>,
    pub slots: [Slot; 2],
    /// Size of each pyramid level, and of its patch grid (empty without frame generation).
    planes: Vec<(u32, u32)>,
    patches: Vec<(u32, u32)>,
    fg: Option<Gen>,
    /// Generated frames, used round-robin: a pair's frames are all made when it starts, and
    /// some may still be waiting their turn when the next pair starts. Made as they're first
    /// needed (`ensure_outs`), up to `OUTS`.
    pub outs: Vec<Tex>,
}

/// The flow both ways (last pair's kept as candidates), its checks and the landing maps.
struct Gen {
    seed: Tex,
    patch: Vec<Tex>,
    /// The patches after propagation (see `flow_patch`).
    spread: Vec<Tex>,
    rough: Vec<Tex>,
    /// Dense flow per [set][direction][level]; the sets alternate so last pair's survives.
    flow: [[Vec<Tex>; 2]; 2],
    set: usize,
    temporal: bool,
    /// Level-0 flow after snapping to edges, per direction.
    snapped: [Tex; 2],
    info: [Tex; 2],
    /// What stayed as it was, and for how long (per set), and from that the overlays.
    edges: [Tex; 2],
    held: Tex,
    fin: [Tex; 2],
    keys: [Tex; 2],
    counts: Tex,
    /// Copies of `counts` on their way back to the CPU (test sessions), and whose they are.
    readback: Vec<(ID3D11Texture2D, Option<u64>)>,
    /// Hardware optical flow, when it's the method and available.
    hw: Option<Hw>,
}

/// Counters of a pair's motion (see gpu.hlsl's `CountsIn`).
pub const COUNTS: usize = 11;

/// Most generated frames kept at once (each one frame in size): two pairs' worth at 30 -> 240.
pub const OUTS: usize = 14;

/// Where frame generation's motion comes from.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FlowMethod {
    /// The shader pyramid in gpu.hlsl (any GPU).
    Shader,
    /// NVIDIA's optical flow hardware (nvof.rs), used as it comes.
    Hardware,
    /// The hardware's vectors refined by the shader's last level.
    HardwareRefined,
}

/// The method to use: the hardware if `hardware` (the setting), else the shader;
/// `PASSTHRU_FLOW=shader|hw|hw-refined` overrides it (for comparing). Hardware falls back
/// to the shader where it isn't available.
///
/// An experiment, off unless asked for. Measured on real 4K capture (10 clips of play, a
/// picture regenerated from its neighbours): the hardware at its medium preset scores 25.71 dB
/// against the shader's 25.51 and its worst spots are 6% better (163 against 174), but it takes
/// 8.7 ms a pair against 2.2. D3D11 runs it in the same queue as the presents, so live at
/// 30 -> 120 on 240 Hz only 75% of frames kept the steady two refreshes (shader: 99.5%) and
/// latency rose 10 ms. Worth it only with its own queue and cross-queue synchronisation.
pub fn flow_method(hardware: bool) -> FlowMethod {
    match std::env::var("PASSTHRU_FLOW").as_deref() {
        Ok("hw") => FlowMethod::Hardware,
        Ok("hw-refined") => FlowMethod::HardwareRefined,
        Ok("shader") => FlowMethod::Shader,
        _ if hardware => FlowMethod::Hardware,
        _ => FlowMethod::Shader,
    }
}

/// The hardware optical flow session and its textures: 8-bit luma per slot (its input), and
/// its vectors both ways.
struct Hw {
    flow: crate::nvof::Flow,
    luma: [Tex; 2],
    input: [crate::nvof::Buffer; 2],
    out: [Tex; 2],
    output: [crate::nvof::Buffer; 2],
    /// Level 0's patch starting vectors made from them.
    seed0: Tex,
    refine: bool,
    /// The slot last measured to (the next pair continues from it).
    last_to: Option<usize>,
    /// Its input is 1/div of the picture's size.
    div: u32,
    size: (u32, u32),
}

impl Hw {
    fn new(gpu: &Gpu, w: u32, h: u32, scale0: u32, patches0: (u32, u32), refine: bool) -> Result<Hw> {
        // Experiments: PASSTHRU_NVOF_PERF (5 slow/best, 10, 20 fast) and _DIV (input at
        // 1/DIV size).
        let env = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (perf, div) = (env("PASSTHRU_NVOF_PERF", 10), env("PASSTHRU_NVOF_DIV", 1).max(1));
        let (w, h) = (w.div_ceil(div), h.div_ceil(div));
        // A vector per level-0 pixel at most (finer would only be averaged away).
        let mut flow = crate::nvof::Flow::new(&gpu.device, &gpu.ctx, (w, h), (scale0 / div).max(1), perf)?;
        let luma = [gpu.tex_bound(w, h, DXGI_FORMAT_R8_UNORM, true, true)?, gpu.tex_bound(w, h, DXGI_FORMAT_R8_UNORM, true, true)?];
        let (gw, gh) = (w.div_ceil(flow.grid), h.div_ceil(flow.grid));
        let out = [gpu.tex_bound(gw, gh, DXGI_FORMAT_R16G16_SINT, true, true)?, gpu.tex_bound(gw, gh, DXGI_FORMAT_R16G16_SINT, true, true)?];
        let res = |t: &Tex| -> Result<ID3D11Resource> { Ok(t.tex.cast()?) };
        let input = [flow.register(&res(&luma[0])?)?, flow.register(&res(&luma[1])?)?];
        let output = [flow.register(&res(&out[0])?)?, flow.register(&res(&out[1])?)?];
        let seed0 = gpu.tex(patches0.0, patches0.1, DXGI_FORMAT_R32G32B32A32_FLOAT, true)?;
        Ok(Hw { flow, luma, input, out, output, seed0, refine, last_to: None, div, size: (w, h) })
    }
}

/// Lucas-Kanade steps per patch at level `l`. The finest level starts close (and is snapped to
/// edges afterwards): on the test clips 4 steps there score the same as 10, 0 a little worse.
fn lk_steps(l: usize) -> u32 {
    if l == 0 { 4 } else { 10 }
}

/// Stop the pyramid once a level is this narrow.
const COARSEST_WIDTH: u32 = 100;

/// Above this many pixels (more than 1440p), motion is measured from quarter size instead of
/// half: a 4K picture's half size is 1080p, more detail than motion needs, and measuring it
/// there costs about three times as much, which adds to the delay with smoothing on.
const QUARTER_ABOVE: u32 = 2560 * 1440;

fn level0_scale(w: u32, h: u32) -> u32 {
    // Experiments: PASSTHRU_SCALE0=2|4.
    if let Some(s) = std::env::var("PASSTHRU_SCALE0").ok().and_then(|v| v.parse().ok()).filter(|&s| s == 2 || s == 4) {
        return s;
    }
    if w * h > QUARTER_ABOVE { 4 } else { 2 }
}

impl Frames {
    /// With everything for frame generation, and all `OUTS` generated frames (the tests).
    pub fn new(gpu: &Gpu, w: u32, h: u32, layout: Layout) -> Result<Frames> {
        let mut f = Frames::with(gpu, w, h, layout, true, flow_method(false))?;
        f.ensure_outs(gpu, OUTS)?;
        Ok(f)
    }

    /// `fg`: with what frame generation needs (pyramids, flow), but no generated frames yet.
    pub fn with(gpu: &Gpu, w: u32, h: u32, layout: Layout, fg: bool, method: FlowMethod) -> Result<Frames> {
        let input = match layout {
            Layout::Nv12 => {
                vec![gpu.tex(w, h, DXGI_FORMAT_R8_UNORM, false)?, gpu.tex(w / 2, h / 2, DXGI_FORMAT_R8G8_UNORM, false)?]
            }
            Layout::Yuy2 => vec![gpu.tex(w / 2, h, DXGI_FORMAT_R8G8B8A8_UNORM, false)?],
            Layout::Rgb32 => vec![gpu.tex(w, h, DXGI_FORMAT_B8G8R8A8_UNORM, false)?],
        };
        let scale0 = level0_scale(w, h);
        let mut planes = vec![(w.div_ceil(scale0), h.div_ceil(scale0))];
        if !fg {
            planes.clear();
        }
        while let Some(&(pw, ph)) = planes.last()
            && pw > COARSEST_WIDTH
            && ph >= 32
        {
            planes.push((pw.div_ceil(2), ph.div_ceil(2)));
        }
        let patches: Vec<(u32, u32)> = planes.iter().map(|&(pw, ph)| (pw.div_ceil(4), ph.div_ceil(4))).collect();
        let slot = || -> Result<Slot> {
            Ok(Slot {
                frame: gpu.tex(w, h, DXGI_FORMAT_R8G8B8A8_UNORM, true)?,
                luma: per_level(gpu, &planes, DXGI_FORMAT_R32_FLOAT)?,
            })
        };
        let slots = [slot()?, slot()?];
        let mut fg = if fg { Some(Gen::new(gpu, w, h, &planes, &patches)?) } else { None };
        if let Some(g) = fg.as_mut() {
            if method != FlowMethod::Shader {
                match Hw::new(gpu, w, h, scale0, patches[0], method == FlowMethod::HardwareRefined) {
                    Ok(hw) => {
                        crate::log::line(&format!(
                            "gpu: hardware optical flow ({method:?}), a vector per {} pixels",
                            hw.flow.grid
                        ));
                        g.hw = Some(hw);
                    }
                    Err(e) => crate::log::line(&format!("gpu: no hardware optical flow ({e:#}); using the shader")),
                }
            }
        }
        Ok(Frames { w, h, layout, scale0, input, slots, planes, patches, fg, outs: Vec::new() })
    }

    /// Whether this was made with frame generation.
    pub fn has_gen(&self) -> bool {
        self.fg.is_some()
    }

    /// Whether motion comes from the optical flow hardware.
    pub fn hardware_flow(&self) -> bool {
        self.fg.as_ref().is_some_and(|g| g.hw.is_some())
    }

    /// Makes sure there are at least `n` generated frames (at most `OUTS`).
    pub fn ensure_outs(&mut self, gpu: &Gpu, n: usize) -> Result<()> {
        while self.outs.len() < n.min(OUTS) {
            self.outs.push(gpu.tex(self.w, self.h, DXGI_FORMAT_R8G8B8A8_UNORM, true)?);
        }
        Ok(())
    }

    fn fg(&self) -> &Gen {
        self.fg.as_ref().expect("made without frame generation")
    }
}

fn per_level(gpu: &Gpu, sizes: &[(u32, u32)], fmt: DXGI_FORMAT) -> Result<Vec<Tex>> {
    sizes.iter().map(|&(pw, ph)| gpu.tex(pw, ph, fmt, true)).collect()
}

impl Gen {
    fn new(gpu: &Gpu, w: u32, h: u32, planes: &[(u32, u32)], patches: &[(u32, u32)]) -> Result<Gen> {
        let (p0w, p0h) = planes[0];
        let coarsest = *patches.last().expect("at least one level");
        let level0 = || gpu.tex(p0w, p0h, DXGI_FORMAT_R32G32B32A32_FLOAT, true);
        let flows = || -> Result<[Vec<Tex>; 2]> {
            Ok([per_level(gpu, planes, DXGI_FORMAT_R32G32_FLOAT)?, per_level(gpu, planes, DXGI_FORMAT_R32G32_FLOAT)?])
        };
        Ok(Gen {
            seed: gpu.tex(coarsest.0, coarsest.1, DXGI_FORMAT_R32G32B32A32_FLOAT, true)?,
            patch: per_level(gpu, patches, DXGI_FORMAT_R32G32B32A32_FLOAT)?,
            spread: per_level(gpu, patches, DXGI_FORMAT_R32G32B32A32_FLOAT)?,
            rough: per_level(gpu, planes, DXGI_FORMAT_R32G32_FLOAT)?,
            flow: [flows()?, flows()?],
            set: 0,
            temporal: false,
            snapped: [
                gpu.tex(p0w, p0h, DXGI_FORMAT_R32G32_FLOAT, true)?,
                gpu.tex(p0w, p0h, DXGI_FORMAT_R32G32_FLOAT, true)?,
            ],
            info: [level0()?, level0()?],
            edges: [gpu.tex(p0w, p0h, DXGI_FORMAT_R32_FLOAT, true)?, gpu.tex(p0w, p0h, DXGI_FORMAT_R32_FLOAT, true)?],
            held: gpu.tex(p0w, p0h, DXGI_FORMAT_R32_FLOAT, true)?,
            fin: [level0()?, level0()?],
            keys: [gpu.tex(w, h, DXGI_FORMAT_R32_UINT, true)?, gpu.tex(w, h, DXGI_FORMAT_R32_UINT, true)?],
            counts: gpu.tex(COUNTS as u32, 1, DXGI_FORMAT_R32_UINT, true)?,
            readback: Vec::new(),
            hw: None,
        })
    }
}

impl Frames {

    /// Uploads one packed frame: NV12 (Y then UV rows), YUY2 or top-down BGRA rows.
    pub fn upload(&self, gpu: &Gpu, data: &[u8]) -> Result<()> {
        let (w, h) = (self.w as usize, self.h as usize);
        let need = match self.layout {
            Layout::Nv12 => w * h + w * (h / 2),
            Layout::Yuy2 => w * 2 * h,
            Layout::Rgb32 => w * 4 * h,
        };
        if data.len() < need {
            bail!("frame is {} bytes, expected {need}", data.len());
        }
        unsafe {
            match self.layout {
                Layout::Nv12 => {
                    gpu.ctx.UpdateSubresource(&self.input[0].tex, 0, None, data.as_ptr() as _, w as u32, 0);
                    gpu.ctx.UpdateSubresource(&self.input[1].tex, 0, None, data[w * h..].as_ptr() as _, w as u32, 0);
                }
                Layout::Yuy2 => {
                    gpu.ctx.UpdateSubresource(&self.input[0].tex, 0, None, data.as_ptr() as _, (w * 2) as u32, 0)
                }
                Layout::Rgb32 => {
                    gpu.ctx.UpdateSubresource(&self.input[0].tex, 0, None, data.as_ptr() as _, (w * 4) as u32, 0)
                }
            }
        }
        Ok(())
    }

    /// Decodes the uploaded frame into `slot`, and builds its pyramid if `pyramid`.
    pub fn decode(&self, gpu: &Gpu, slot: usize, d: &Decoding, pyramid: bool) {
        let (kr, kb) = convert::kr_kb(d.matrix);
        let kg = 1.0 - kr - kb;
        let (y_off, y_range, c_range) = if d.full_range { (0.0, 255.0, 255) } else { (16.0, 219.0, 224) };
        let mut flags = 0;
        if d.hdr {
            flags |= F_HDR;
            if d.gamut_2020 {
                flags |= F_GAMUT;
            }
        }
        let layout = match d.layout {
            Layout::Nv12 => 0,
            Layout::Yuy2 => 1,
            Layout::Rgb32 => 2,
        };
        let p = Params {
            sz: [self.w as i32, self.h as i32, 0, 0],
            kc: [
                (2.0 * (1.0 - kr)) as f32,
                (-2.0 * (1.0 - kb) * kb / kg) as f32,
                (-2.0 * (1.0 - kr) * kr / kg) as f32,
                (2.0 * (1.0 - kb)) as f32,
            ],
            misc: [0.0, d.white_nits, y_off, y_range],
            fl: [flags, layout, c_range, 0],
            ..Default::default()
        };
        let srvs: Vec<(u32, &Tex)> = match self.layout {
            Layout::Nv12 => vec![(0, &self.input[0]), (1, &self.input[1])],
            _ => vec![(2, &self.input[0])],
        };
        let s = &self.slots[slot];
        gpu.run(&gpu.k.decode, &p, &srvs, &[(0, &s.frame)], (groups(self.w, 8), groups(self.h, 8)));
        if !pyramid {
            return;
        }
        let mut above = (self.w, self.h);
        for (l, &(pw, ph)) in self.planes.iter().enumerate() {
            let p = Params {
                sz: [pw as i32, ph as i32, above.0 as i32, above.1 as i32],
                misc: [0.0, self.scale0 as f32, 0.0, 0.0],
                ..Default::default()
            };
            let g = (groups(pw, 8), groups(ph, 8));
            if l == 0 {
                gpu.run(&gpu.k.luma0, &p, &[(3, &s.frame)], &[(1, &s.luma[0])], g);
            } else {
                gpu.run(&gpu.k.halve, &p, &[(4, &s.luma[l - 1])], &[(1, &s.luma[l])], g);
            }
            above = (pw, ph);
        }
        if let Some(hw) = self.fg.as_ref().and_then(|g| g.hw.as_ref()) {
            let (lw, lh) = hw.size;
            let p = Params {
                sz: [lw as i32, lh as i32, self.w as i32, self.h as i32],
                misc: [0.0, hw.div as f32, 0.0, 0.0],
                ..Default::default()
            };
            gpu.run(&gpu.k.luma8, &p, &[(3, &s.frame)], &[(8, &hw.luma[slot])], (groups(lw, 8), groups(lh, 8)));
        }
    }

    /// Measures motion between slots `a` and `b` (both decoded with pyramids), both ways.
    pub fn estimate(&mut self, gpu: &Gpu, a: usize, b: usize) {
        let fg = self.fg.as_mut().expect("made without frame generation");
        unsafe { gpu.ctx.ClearUnorderedAccessViewUint(fg.counts.uav.as_ref().expect("uav"), &[0; 4]) };
        fg.set = 1 - fg.set;
        let (cur, prev) = (fg.set, 1 - fg.set);
        let n = self.planes.len();
        let pair = |dir: usize| {
            if dir == 0 { (&self.slots[a], &self.slots[b]) } else { (&self.slots[b], &self.slots[a]) }
        };
        let hw_ok = match fg.hw.as_mut() {
            Some(hw) => {
                let continuing = hw.last_to == Some(a);
                hw.last_to = Some(b);
                match hw.flow.run(hw.input[a], hw.input[b], hw.output[0], hw.output[1], continuing) {
                    Ok(()) => true,
                    Err(e) => {
                        crate::log::line(&format!("gpu: hardware optical flow failed ({e:#}); using the shader"));
                        fg.hw = None;
                        false
                    }
                }
            }
            None => false,
        };
        if hw_ok && let Some(hw) = fg.hw.as_ref() {
            let (pw, ph) = self.planes[0];
            let (gw, gh) = self.patches[0];
            let (fw, fh) = (hw.size.0.div_ceil(hw.flow.grid), hw.size.1.div_ceil(hw.flow.grid));
            for dir in 0..2 {
                let (sx, sy) = pair(dir);
                let p = Params {
                    sz: [pw as i32, ph as i32, 0, 0],
                    gr: [fw as i32, fh as i32, gw as i32, gh as i32],
                    misc: [0.0, self.scale0 as f32, hw.flow.grid as f32, hw.div as f32],
                    ..Default::default()
                };
                let g = (groups(pw, 8), groups(ph, 8));
                let flow = [(22, &hw.out[dir])];
                gpu.run(&gpu.k.hw_seed, &p, &flow, &[(2, &hw.seed0)], (groups(gw, 8), groups(gh, 8)));
                // The patches from those starting vectors: refined (or not), and measured for
                // the cut test either way.
                let mut pp = Params {
                    sz: [pw as i32, ph as i32, 0, 0],
                    gr: [gw as i32, gh as i32, 0, 0],
                    fl: [F_SEED, 0, if hw.refine { lk_steps(0) } else { 0 }, 0],
                    ..Default::default()
                };
                if dir == 0 {
                    pp.fl[0] |= F_COUNT;
                }
                let luma = [(5, &sx.luma[0]), (6, &sy.luma[0])];
                if hw.refine || dir == 0 {
                    let srvs = [luma[0], luma[1], (9, &hw.seed0)];
                    gpu.run(&gpu.k.flow_patch, &pp, &srvs, &[(2, &fg.patch[0]), (4, &fg.counts)], (gw, gh));
                }
                if hw.refine {
                    gpu.run(&gpu.k.flow_dense, &pp, &[luma[0], luma[1], (9, &fg.patch[0])], &[(3, &fg.rough[0])], g);
                    gpu.run(&gpu.k.flow_smooth, &pp, &[(10, &fg.rough[0])], &[(3, &fg.flow[cur][dir][0])], g);
                } else {
                    gpu.run(&gpu.k.hw_dense, &p, &flow, &[(3, &fg.flow[cur][dir][0])], g);
                }
            }
        }
        // Coarse to fine, both directions per level: each patch also tries the opposite
        // direction's motion (B->A at the coarser level for A->B, A->B at this level for B->A).
        for l in (0..n).rev() {
            if hw_ok {
                break;
            }
            for dir in 0..2 {
                let (sx, sy) = pair(dir);
                let (pw, ph) = self.planes[l];
                let (gw, gh) = self.patches[l];
                let coarser = self.planes.get(l + 1).copied().unwrap_or((1, 1));
                let mut p = Params {
                    sz: [pw as i32, ph as i32, 0, 0],
                    gr: [gw as i32, gh as i32, coarser.0 as i32, coarser.1 as i32],
                    fl: [0, 0, lk_steps(l), 0],
                    ..Default::default()
                };
                let luma = [(5, &sx.luma[l]), (6, &sy.luma[l])];
                let pg = (gw, gh);
                if l == n - 1 {
                    gpu.run(&gpu.k.flow_full, &p, &luma, &[(2, &fg.seed)], (gw, gh));
                    p.fl[0] = F_SEED;
                    gpu.run(&gpu.k.flow_patch, &p, &[luma[0], luma[1], (9, &fg.seed)], &[(2, &fg.patch[l])], pg);
                } else {
                    if fg.temporal {
                        p.fl[0] |= F_TEMPORAL;
                    }
                    if l == 0 && dir == 0 {
                        p.fl[0] |= F_COUNT;
                    }
                    let other = if dir == 1 { &fg.flow[cur][0][l] } else { &fg.flow[cur][1][l + 1] };
                    p.fl[3] = if dir == 1 { 1 } else { 2 };
                    let srvs = [
                        luma[0],
                        luma[1],
                        (7, &fg.flow[cur][dir][l + 1]),
                        (8, &fg.flow[prev][dir][l]),
                        (11, other),
                    ];
                    gpu.run(&gpu.k.flow_patch, &p, &srvs, &[(2, &fg.patch[l]), (4, &fg.counts)], pg);
                }
                // Propagation: each patch tries its neighbours' vectors (no Lucas-Kanade steps:
                // those vectors are refined already). Between the coarsest level (searched in
                // full) and level 0 (where it costs most and gains least), so a good match
                // spreads before the next level starts from it.
                let spread = l != 0 && l != n - 1;
                let pp = Params { fl: [(p.fl[0] & !F_SEED) | F_PROP, 0, 0, 0], ..p };
                let patches = if spread {
                    let srvs = [luma[0], luma[1], (9, &fg.patch[l])];
                    gpu.run(&gpu.k.flow_patch, &pp, &srvs, &[(2, &fg.spread[l]), (4, &fg.counts)], pg);
                    &fg.spread[l]
                } else {
                    &fg.patch[l]
                };
                let g = (groups(pw, 8), groups(ph, 8));
                gpu.run(&gpu.k.flow_dense, &p, &[luma[0], luma[1], (9, patches)], &[(3, &fg.rough[l])], g);
                gpu.run(&gpu.k.flow_smooth, &p, &[(10, &fg.rough[l])], &[(3, &fg.flow[cur][dir][l])], g);
            }
        }
        let (pw, ph) = self.planes[0];
        let p = Params { sz: [pw as i32, ph as i32, 0, 0], ..Default::default() };
        let g = (groups(pw, 8), groups(ph, 8));
        for dir in 0..2 {
            let (sx, sy) = pair(dir);
            let srvs = [(5, &sx.luma[0]), (6, &sy.luma[0]), (10, &fg.flow[cur][dir][0])];
            gpu.run(&gpu.k.flow_snap, &p, &srvs, &[(3, &fg.snapped[dir])], g);
        }
        for dir in 0..2 {
            let (sx, sy) = pair(dir);
            let srvs = [(5, &sx.luma[0]), (6, &sy.luma[0]), (10, &fg.snapped[dir]), (11, &fg.snapped[1 - dir])];
            gpu.run(&gpu.k.flow_check, &p, &srvs, &[(2, &fg.info[dir])], g);
        }
        // Whether text inverted where it is (a menu's highlight jumped: see the shader).
        let luma = [(5, &pair(0).0.luma[0]), (6, &pair(0).1.luma[0])];
        gpu.run(&gpu.k.flow_inplace, &p, &luma, &[(4, &fg.counts)], (pw / 8, ph / 8));
        // Filled motion goes through rough[0], free once the levels are done.
        let pf = Params {
            sz: [pw as i32, ph as i32, self.w as i32, self.h as i32],
            misc: [0.0, self.scale0 as f32, 0.0, 0.0],
            ..Default::default()
        };
        // Overlays standing still over moving scenery, from both pictures at full size.
        let srvs = [(16, &self.slots[a].frame), (17, &self.slots[b].frame), (25, &fg.edges[prev])];
        let pe = Params { fl: [if fg.temporal { F_TEMPORAL } else { 0 }, 0, 0, 0], ..pf };
        gpu.run(&gpu.k.flow_edges, &pe, &srvs, &[(1, &fg.edges[cur])], g);
        let srvs = [(10, &fg.snapped[0]), (23, &fg.edges[cur])];
        gpu.run(&gpu.k.flow_held, &p, &srvs, &[(1, &fg.held), (4, &fg.counts)], g);
        for dir in 0..2 {
            let (sx, _) = pair(dir);
            let srvs = [(10, &fg.snapped[dir]), (20, &fg.info[dir]), (21, &sx.frame)];
            gpu.run(&gpu.k.flow_fill, &pf, &srvs, &[(3, &fg.rough[0])], g);
            // A->B counts how much of what changed it can follow (the UI-change test).
            let pd = Params { fl: [if dir == 0 { F_COUNT } else { 0 }, 0, 0, 0], ..p };
            let uavs = [(2, &fg.fin[dir]), (4, &fg.counts)];
            gpu.run(&gpu.k.flow_final, &pd, &[(10, &fg.rough[0]), (20, &fg.info[dir])], &uavs, g);
        }
        fg.temporal = true;
    }

    /// Forgets last pair's motion (after a jump in the picture).
    pub fn forget_motion(&mut self) {
        if let Some(fg) = self.fg.as_mut() {
            fg.temporal = false;
            if let Some(hw) = fg.hw.as_mut() {
                hw.last_to = None;
            }
        }
    }

    /// Generates the frame at time `t` between slots `a` and `b` into generated frame `out`.
    pub fn interpolate(&self, gpu: &Gpu, a: usize, b: usize, t: f32, out: usize, quality: Quality) {
        let fg = self.fg();
        for k in &fg.keys {
            unsafe { gpu.ctx.ClearUnorderedAccessViewUint(k.uav.as_ref().expect("uav"), &[u32::MAX; 4]) };
        }
        let p = Params {
            sz: [self.w as i32, self.h as i32, 0, 0],
            misc: [t, self.scale0 as f32, 0.0, 0.0],
            ..Default::default()
        };
        let fin = [(12, &fg.fin[0]), (13, &fg.fin[1])];
        let g = (groups(self.w, 16), groups(self.h, 16));
        let held = (24, &fg.held);
        let (fa, fb) = ((16, &self.slots[a].frame), (17, &self.slots[b].frame));
        gpu.run_z(&gpu.k.project, &p, &[fin[0], fin[1], fa, fb, (18, &fg.counts), held], &[(5, &fg.keys[0]), (6, &fg.keys[1])], (g.0, g.1, 2));
        let srvs = [
            fin[0],
            fin[1],
            (14, &fg.keys[0]),
            (15, &fg.keys[1]),
            fa,
            fb,
            (18, &fg.counts),
            held,
        ];
        gpu.run(&gpu.k.compose[quality as usize], &p, &srvs, &[(7, &self.outs[out])], g);
    }

    /// How many level-0 pixels the last pair found to be overlay, whether or not the camera
    /// moved enough for that to count.
    #[cfg(test)]
    pub fn held_count(&self, gpu: &Gpu) -> Result<usize> {
        let bytes = gpu.read(&self.fg().held, 4)?;
        Ok(bytes.as_chunks::<4>().0.iter().filter(|b| f32::from_le_bytes(**b) > 0.0).count())
    }

    /// The finished level-0 fields of direction `dir` (0: A -> B), per pixel: vector x, y,
    /// mismatch with the other direction, unreliability (tests only).
    pub fn fields(&self, gpu: &Gpu, dir: usize) -> Result<(Vec<[f32; 4]>, usize, usize)> {
        let t = &self.fg().fin[dir];
        let b = gpu.read(t, 16)?;
        let px = b
            .chunks_exact(16)
            .map(|c| [0, 1, 2, 3].map(|i| f32::from_le_bytes([c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]])))
            .collect();
        Ok((px, t.w as usize, t.h as usize))
    }

    /// Copies this pair's counters for reading back later, without waiting (test sessions).
    pub fn request_counts(&mut self, gpu: &Gpu, tag: u64) {
        let Some(fg) = self.fg.as_mut() else { return };
        let i = match fg.readback.iter().position(|r| r.1.is_none()) {
            Some(i) => i,
            None if fg.readback.len() < 8 => match gpu.staging_like(&fg.counts) {
                Ok(t) => {
                    fg.readback.push((t, None));
                    fg.readback.len() - 1
                }
                Err(_) => return,
            },
            None => return,
        };
        unsafe { gpu.ctx.CopyResource(&fg.readback[i].0, &fg.counts.tex) };
        fg.readback[i].1 = Some(tag);
    }

    /// Counters copied by `request_counts` that the GPU has finished, with their tags.
    pub fn poll_counts(&mut self, gpu: &Gpu) -> Vec<(u64, [u32; COUNTS])> {
        let mut out = Vec::new();
        let Some(fg) = self.fg.as_mut() else { return out };
        for (tex, tag) in &mut fg.readback {
            let Some(t) = *tag else { continue };
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            let mapped = unsafe {
                gpu.ctx.Map(&*tex, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32, Some(&mut m))
            };
            if mapped.is_err() {
                continue;
            }
            let v: [u32; COUNTS] = std::array::from_fn(|i| unsafe { *(m.pData as *const u32).add(i) });
            unsafe { gpu.ctx.Unmap(&*tex, 0) };
            *tag = None;
            out.push((t, v));
        }
        out
    }

    /// The pair's counters (see gpu.hlsl's `CountsIn`), read back synchronously (tests only).
    pub fn counts(&self, gpu: &Gpu) -> Result<[u32; COUNTS]> {
        let b = gpu.read(&self.fg().counts, 4)?;
        Ok(std::array::from_fn(|i| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])))
    }
}

/// Our swap chain under gpui's layer.
pub struct Swap {
    chain: IDXGISwapChain1,
    rtv: Option<ID3D11RenderTargetView>,
    pub w: u32,
    pub h: u32,
    _dcomp: IDCompositionDevice,
    _target: IDCompositionTarget,
    _visual: IDCompositionVisual,
}

impl Swap {
    pub fn new(gpu: &Gpu, hwnd: HWND, w: u32, h: u32) -> Result<Swap> {
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))? };
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: w.max(1),
            Height: h.max(1),
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: 0,
        };
        let chain = unsafe { factory.CreateSwapChainForComposition(&gpu.device, &desc, None) }
            .context("CreateSwapChainForComposition")?;
        let dxgi: IDXGIDevice = gpu.device.cast()?;
        let dcomp: IDCompositionDevice =
            unsafe { DCompositionCreateDevice(&dxgi) }.context("DCompositionCreateDevice")?;
        // Non-topmost: underneath gpui's own (topmost) target on the same window.
        let target = unsafe { dcomp.CreateTargetForHwnd(hwnd, false) }.context("CreateTargetForHwnd")?;
        let visual = unsafe { dcomp.CreateVisual()? };
        unsafe {
            visual.SetContent(&chain)?;
            target.SetRoot(&visual)?;
            dcomp.Commit()?;
        }
        let mut s =
            Swap { chain, rtv: None, w: desc.Width, h: desc.Height, _dcomp: dcomp, _target: target, _visual: visual };
        s.make_rtv(gpu)?;
        Ok(s)
    }

    fn make_rtv(&mut self, gpu: &Gpu) -> Result<()> {
        let buffer: ID3D11Texture2D = unsafe { self.chain.GetBuffer(0)? };
        let mut rtv = None;
        unsafe { gpu.device.CreateRenderTargetView(&buffer, None, Some(&mut rtv))? };
        self.rtv = rtv;
        Ok(())
    }

    /// Grows the buffers to at least w x h (never shrinks: the window is clipped anyway, and
    /// staying big means resizing never uncovers the desktop behind the window).
    pub fn ensure(&mut self, gpu: &Gpu, w: u32, h: u32) -> Result<()> {
        if w <= self.w && h <= self.h {
            return Ok(());
        }
        let (nw, nh) = (w.max(self.w), h.max(self.h));
        self.rtv = None;
        unsafe {
            gpu.ctx.OMSetRenderTargets(None, None);
            gpu.ctx.Flush();
            self.chain.ResizeBuffers(0, nw, nh, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG(0))?;
        }
        self.w = nw;
        self.h = nh;
        self.make_rtv(gpu)
    }

    /// Draws `src` (or black) fitted into `rect` (x, y, w, h in pixels) and presents.
    pub fn show(&self, gpu: &Gpu, src: Option<&Tex>, rect: [f32; 4], mark: Option<u32>) -> Result<()> {
        let rtv = self.rtv.as_ref().context("no back buffer")?;
        gpu.draw(rtv, (self.w, self.h), src, rect, mark);
        unsafe { self.chain.Present(0, DXGI_PRESENT(0)).ok().context("Present")? };
        Ok(())
    }

    /// The swap chain's present count so far.
    pub fn present_count(&self) -> Option<u32> {
        unsafe { self.chain.GetLastPresentCount().ok() }
    }

    /// The latest present that reached the screen: its present count, the refresh it was
    /// queued at, the vblank it was shown at and that vblank's QPC time.
    pub fn frame_stats(&self) -> Option<(u32, u32, u32, i64)> {
        let mut s = DXGI_FRAME_STATISTICS::default();
        unsafe { self.chain.GetFrameStatistics(&mut s).ok()? };
        Some((s.PresentCount, s.PresentRefreshCount, s.SyncRefreshCount, s.SyncQPCTime))
    }
}
