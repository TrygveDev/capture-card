//! Upscaling with NVIDIA RTX Video Super Resolution, for when the video is drawn bigger than it
//! arrives (a 1080p signal in a 1440p fullscreen, say).
//!
//! DLSS can't be used here: it needs the game's motion vectors and depth, and a capture card
//! only delivers finished pixels. RTX Video Super Resolution is NVIDIA's upscaler for finished
//! video. It runs inside the D3D11 video processor once NVIDIA's stream extension switches it
//! on (the same switch browsers use). It needs an RTX card and the driver's "RTX Video
//! Enhancement → Super resolution" setting, and the driver accepts the switch even when that
//! setting is off. So the first frames are scaled both ways and compared: only if they differ
//! is super resolution really running. Otherwise the present shader's Lanczos scaling does the
//! job (it beats the video processor's plain scaling).

use std::mem::ManuallyDrop;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{GUID, Interface};

use crate::gpu::{Gpu, Tex};

/// NVIDIA's video processor extension, and its "super resolution, version 1" call.
const NVIDIA_PPE: GUID = GUID::from_u128(0xd43ce1b3_1f4b_48ac_baee_c3c25375e6f7);
const NV_VERSION: u32 = 1;
const NV_SUPER_RESOLUTION: u32 = 2;

/// While super resolution isn't running, check again this often (the setting may be turned on).
const RECHECK: Duration = Duration::from_secs(30);

/// Scales with RTX Video Super Resolution when it's really running.
pub struct Upscaler {
    vsr: Processor,
    plain: Processor,
    /// Whether super resolution changed the picture, for which input size, and when checked.
    verdict: Option<(bool, (u32, u32), Instant)>,
}

impl Upscaler {
    pub fn new(gpu: &Gpu) -> Result<Upscaler> {
        Ok(Upscaler { vsr: Processor::new(gpu, true)?, plain: Processor::new(gpu, false)?, verdict: None })
    }

    /// The last picture `run` made with super resolution.
    pub fn output(&self) -> Option<&Tex> {
        self.vsr.output()
    }

    /// Scales `src` to `w` x `h` with super resolution, or None if it isn't running (draw it
    /// with the shader then). Errors mean the video processor can't be used at all.
    pub fn run(&mut self, gpu: &Gpu, src: &Tex, w: u32, h: u32) -> Result<Option<&Tex>> {
        let size = (src.w, src.h);
        let due = match self.verdict {
            None => true,
            Some((on, checked, at)) => checked != size || (!on && at.elapsed() >= RECHECK),
        };
        if due {
            self.vsr.run(gpu, src, w, h)?;
            self.plain.run(gpu, src, w, h)?;
            let on = differs(gpu, self.vsr.output().context("no output")?, self.plain.output().context("no output")?)?;
            if self.verdict.is_none_or(|(was, _, _)| was != on) {
                crate::log::line(&format!(
                    "upscale: RTX Video Super Resolution {}",
                    if on { "is running" } else { "isn't running (off in the NVIDIA settings?); using Lanczos" }
                ));
            }
            self.verdict = Some((on, size, Instant::now()));
            // The plain one is only for the check.
            self.plain.setup = None;
            return Ok(if on { self.vsr.output() } else { None });
        }
        if self.verdict.is_some_and(|(on, _, _)| on) { self.vsr.run(gpu, src, w, h).map(Some) } else { Ok(None) }
    }
}

/// Whether two same-sized outputs differ: their centres compared, a little noise allowed.
fn differs(gpu: &Gpu, a: &Tex, b: &Tex) -> Result<bool> {
    let (w, h) = (a.w.min(256), a.h.min(256));
    let (x0, y0) = ((a.w - w) / 2, (a.h - h) / 2);
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { a.tex.GetDesc(&mut desc) };
    desc.Width = w;
    desc.Height = h;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
    let region = D3D11_BOX { left: x0, top: y0, front: 0, right: x0 + w, bottom: y0 + h, back: 1 };
    let mut rows = Vec::new();
    for t in [a, b] {
        let mut staging = None;
        unsafe { gpu.device.CreateTexture2D(&desc, None, Some(&mut staging))? };
        let staging: ID3D11Texture2D = staging.context("no staging texture")?;
        let mut px = vec![0u8; (w * h * 4) as usize];
        unsafe {
            gpu.ctx.CopySubresourceRegion(&staging, 0, 0, 0, 0, &t.tex, 0, Some(&region));
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            gpu.ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m))?;
            for y in 0..h as usize {
                let row = std::slice::from_raw_parts((m.pData as *const u8).add(y * m.RowPitch as usize), w as usize * 4);
                px[y * w as usize * 4..][..w as usize * 4].copy_from_slice(row);
            }
            gpu.ctx.Unmap(&staging, 0);
        }
        rows.push(px);
    }
    let changed = rows[0].chunks_exact(4).zip(rows[1].chunks_exact(4)).filter(|(p, q)| {
        p.iter().zip(q.iter()).any(|(x, y)| x.abs_diff(*y) > 3)
    });
    Ok(changed.count() * 200 > (w * h) as usize)
}

/// The D3D11 video processor scaling one picture to another size, optionally with super
/// resolution switched on.
pub struct Processor {
    vdev: ID3D11VideoDevice,
    vctx: ID3D11VideoContext,
    setup: Option<Setup>,
    vsr: bool,
}

/// A video processor for one input format and size, and output size.
struct Setup {
    key: (DXGI_FORMAT, u32, u32, u32, u32),
    en: ID3D11VideoProcessorEnumerator,
    vp: ID3D11VideoProcessor,
    out: Tex,
    view: ID3D11VideoProcessorOutputView,
}

impl Processor {
    /// Fails without a video processor.
    pub fn new(gpu: &Gpu, vsr: bool) -> Result<Processor> {
        let vdev: ID3D11VideoDevice = gpu.device.cast().context("no D3D11 video device")?;
        let vctx: ID3D11VideoContext = gpu.ctx.cast().context("no D3D11 video context")?;
        Ok(Processor { vdev, vctx, setup: None, vsr })
    }

    fn output(&self) -> Option<&Tex> {
        self.setup.as_ref().map(|s| &s.out)
    }

    /// Scales display-ready RGBA `src` to `w` x `h`. The result stays valid until the next call.
    pub fn run(&mut self, gpu: &Gpu, src: &Tex, w: u32, h: u32) -> Result<&Tex> {
        self.run_texture(gpu, &src.tex, (src.w, src.h), DXGI_FORMAT_R8G8B8A8_UNORM, (w, h))
    }

    /// `run` for any texture the video processor takes: RGBA, or NV12 (BT.709, limited range;
    /// made without the shader resource bind, which the video processor refuses for NV12).
    pub fn run_texture(
        &mut self,
        gpu: &Gpu,
        src: &ID3D11Texture2D,
        (iw, ih): (u32, u32),
        format: DXGI_FORMAT,
        (w, h): (u32, u32),
    ) -> Result<&Tex> {
        let key = (format, iw, ih, w, h);
        if self.setup.as_ref().is_none_or(|s| s.key != key) {
            self.setup = None;
            self.setup = Some(self.make(gpu, key)?);
        }
        let s = self.setup.as_ref().expect("made above");
        let desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV { MipSlice: 0, ArraySlice: 0 },
            },
        };
        let mut input = None;
        unsafe { self.vdev.CreateVideoProcessorInputView(src, &s.en, &desc, Some(&mut input)) }
            .context("video processor input view")?;
        let stream = D3D11_VIDEO_PROCESSOR_STREAM {
            Enable: true.into(),
            pInputSurface: ManuallyDrop::new(Some(input.context("no input view")?)),
            ..Default::default()
        };
        let r = unsafe { self.vctx.VideoProcessorBlt(&s.vp, &s.view, 0, std::slice::from_ref(&stream)) };
        drop(ManuallyDrop::into_inner(stream.pInputSurface));
        r.context("VideoProcessorBlt")?;
        Ok(&s.out)
    }

    fn make(&self, gpu: &Gpu, key: (DXGI_FORMAT, u32, u32, u32, u32)) -> Result<Setup> {
        let (format, iw, ih, ow, oh) = key;
        let rate = DXGI_RATIONAL { Numerator: 60, Denominator: 1 };
        let desc = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            InputFrameRate: rate,
            InputWidth: iw,
            InputHeight: ih,
            OutputFrameRate: rate,
            OutputWidth: ow,
            OutputHeight: oh,
            Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
        };
        let en = unsafe { self.vdev.CreateVideoProcessorEnumerator(&desc) }.context("video processor enumerator")?;
        let input = D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT.0 as u32;
        let output = D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT.0 as u32;
        if unsafe { en.CheckVideoProcessorFormat(format) }.unwrap_or(0) & input == 0 {
            bail!("the video processor doesn't take {format:?} input");
        }
        let rgba = unsafe { en.CheckVideoProcessorFormat(DXGI_FORMAT_R8G8B8A8_UNORM) }.unwrap_or(0);
        let out_format = if rgba & output != 0 { DXGI_FORMAT_R8G8B8A8_UNORM } else { DXGI_FORMAT_B8G8R8A8_UNORM };
        let vp = unsafe { self.vdev.CreateVideoProcessor(&en, 0) }.context("CreateVideoProcessor")?;
        let c = &self.vctx;
        let whole = |w: u32, h: u32| RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 };
        let in_space = if format == DXGI_FORMAT_NV12 {
            DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709
        } else {
            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
        };
        unsafe {
            c.VideoProcessorSetStreamFrameFormat(&vp, 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
            c.VideoProcessorSetStreamAutoProcessingMode(&vp, 0, false);
            c.VideoProcessorSetStreamSourceRect(&vp, 0, true, Some(&whole(iw, ih)));
            c.VideoProcessorSetStreamDestRect(&vp, 0, true, Some(&whole(ow, oh)));
            c.VideoProcessorSetOutputTargetRect(&vp, true, Some(&whole(ow, oh)));
            // Display-ready RGB out: scale (and convert NV12), nothing else.
            if let Ok(c1) = c.cast::<ID3D11VideoContext1>() {
                c1.VideoProcessorSetStreamColorSpace1(&vp, 0, in_space);
                c1.VideoProcessorSetOutputColorSpace1(&vp, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
            }
        }
        if self.vsr {
            let on = [NV_VERSION, NV_SUPER_RESOLUTION, 1u32];
            let hr = unsafe {
                c.VideoProcessorSetStreamExtension(&vp, 0, &NVIDIA_PPE, size_of_val(&on) as u32, on.as_ptr() as _)
            };
            if hr < 0 {
                bail!("no RTX Video Super Resolution (0x{:08X})", hr as u32);
            }
        }
        let (out, _rtv) = gpu.target(ow, oh, out_format)?;
        let vdesc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
            ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 { Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 } },
        };
        let mut view = None;
        unsafe { self.vdev.CreateVideoProcessorOutputView(&out.tex, &en, &vdesc, Some(&mut view)) }
            .context("video processor output view")?;
        Ok(Setup { key, en, vp, out, view: view.context("no output view")? })
    }
}
