//! NVIDIA's hardware optical flow (the Optical Flow Accelerator on Turing and newer GPUs, the
//! engine DLSS Frame Generation uses), through the D3D11 interface of `nvofapi64.dll`, which
//! comes with the driver. Loaded at run time: without it (another GPU, an old driver) the
//! shader flow in gpu.hlsl is used.
//!
//! Declarations follow NVIDIA's Optical Flow SDK 5.0 headers (nvOpticalFlowCommon.h,
//! nvOpticalFlowD3D11.h; MIT licence).

use std::ffi::c_void;

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11DeviceContext, ID3D11Resource};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{Interface, s, w};

type Handle = *mut c_void;
type Status = i32;

const API_VERSION: u32 = (5 << 4) | 0;
const SUCCESS: Status = 0;

// Enums (C ints).
const MODE_OPTICALFLOW: i32 = 1;
const PERF_SLOW: i32 = 5;
const PRED_BOTH: i32 = 2;
const FORMAT_GRAYSCALE8: i32 = 1;
const CAPS_OUTPUT_GRID_SIZES: i32 = 0;

#[repr(C)]
struct InitParams {
    width: u32,
    height: u32,
    out_grid_size: i32,
    hint_grid_size: i32,
    mode: i32,
    perf_level: i32,
    enable_external_hints: i32,
    enable_output_cost: i32,
    priv_data: Handle,
    disparity_range: i32,
    enable_roi: i32,
    pred_direction: i32,
    enable_global_flow: i32,
    input_buffer_format: i32,
}

#[repr(C)]
struct ExecuteInput {
    input_frame: Handle,
    reference_frame: Handle,
    external_hints: Handle,
    disable_temporal_hints: i32,
    padding: u32,
    priv_data: Handle,
    padding2: u32,
    num_rois: u32,
    roi_data: *mut c_void,
}

#[repr(C)]
struct ExecuteOutput {
    output: Handle,
    output_cost: Handle,
    priv_data: Handle,
    bwd_output: Handle,
    bwd_output_cost: Handle,
    global_flow: Handle,
}

type Create = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut Handle) -> Status;
type Init = unsafe extern "system" fn(Handle, *const InitParams) -> Status;
type FormatCount = unsafe extern "system" fn(Handle, i32, i32, *mut u32) -> Status;
type Formats = unsafe extern "system" fn(Handle, i32, i32, *mut i32) -> Status;
type Register = unsafe extern "system" fn(Handle, *mut c_void, *mut Handle) -> Status;
type Unregister = unsafe extern "system" fn(Handle) -> Status;
type Execute = unsafe extern "system" fn(Handle, *const ExecuteInput, *mut ExecuteOutput) -> Status;
type Destroy = unsafe extern "system" fn(Handle) -> Status;
type LastError = unsafe extern "system" fn(Handle, *mut u8, *mut u32) -> Status;
type Caps = unsafe extern "system" fn(Handle, i32, *mut u32, *mut u32) -> Status;

#[repr(C)]
#[derive(Default)]
struct Functions {
    create: Option<Create>,
    init: Option<Init>,
    format_count: Option<FormatCount>,
    formats: Option<Formats>,
    register: Option<Register>,
    unregister: Option<Unregister>,
    execute: Option<Execute>,
    destroy: Option<Destroy>,
    last_error: Option<LastError>,
    caps: Option<Caps>,
}

type CreateInstance = unsafe extern "system" fn(u32, *mut Functions) -> Status;

/// One optical-flow session: a fixed input size, both directions per call.
pub struct Flow {
    f: Functions,
    handle: Handle,
    _lib: HMODULE,
    /// Full-size pixels per output vector (each way).
    pub grid: u32,
    registered: Vec<Handle>,
}

/// A D3D11 texture registered with a session.
#[derive(Clone, Copy)]
pub struct Buffer(Handle);

impl Flow {
    /// A session for `w` x `h` grayscale pictures on the given device, with the finest output
    /// grid the hardware offers that's no finer than `want_grid`.
    pub fn new(
        device: &ID3D11Device,
        ctx: &ID3D11DeviceContext,
        (w, h): (u32, u32),
        want_grid: u32,
        perf: u32,
    ) -> Result<Flow> {
        unsafe {
            let lib = LoadLibraryW(w!("nvofapi64.dll")).context("no nvofapi64.dll (not an NVIDIA driver)")?;
            let create_instance = GetProcAddress(lib, s!("NvOFAPICreateInstanceD3D11"))
                .context("nvofapi64.dll has no NvOFAPICreateInstanceD3D11")?;
            let create_instance: CreateInstance = std::mem::transmute(create_instance);
            let mut f = Functions::default();
            check(create_instance(API_VERSION, &mut f), "NvOFAPICreateInstanceD3D11")?;
            let create = f.create.context("no create function")?;
            let mut handle: Handle = std::ptr::null_mut();
            check(create(device.as_raw(), ctx.as_raw(), &mut handle), "creating the optical flow session")?;
            let mut flow = Flow { f, handle, _lib: lib, grid: 0, registered: Vec::new() };
            let grids = flow.caps(CAPS_OUTPUT_GRID_SIZES)?;
            let grid = grids.iter().copied().filter(|&g| g <= want_grid).max().or(grids.iter().copied().min());
            flow.grid = grid.context("no output grid sizes")?;
            let params = InitParams {
                width: w,
                height: h,
                out_grid_size: flow.grid as i32,
                hint_grid_size: 0,
                mode: MODE_OPTICALFLOW,
                perf_level: if perf == 0 { PERF_SLOW } else { perf as i32 },
                enable_external_hints: 0,
                enable_output_cost: 0,
                priv_data: std::ptr::null_mut(),
                disparity_range: 0,
                enable_roi: 0,
                pred_direction: PRED_BOTH,
                enable_global_flow: 0,
                input_buffer_format: FORMAT_GRAYSCALE8,
            };
            let init = flow.f.init.context("no init function")?;
            flow.check(init(flow.handle, &params), "initialising the optical flow session")?;
            Ok(flow)
        }
    }

    fn caps(&self, which: i32) -> Result<Vec<u32>> {
        let caps = self.f.caps.context("no caps function")?;
        let mut n = 0u32;
        unsafe { self.check(caps(self.handle, which, std::ptr::null_mut(), &mut n), "reading capabilities")? };
        let mut v = vec![0u32; n as usize];
        unsafe { self.check(caps(self.handle, which, v.as_mut_ptr(), &mut n), "reading capabilities")? };
        v.truncate(n as usize);
        Ok(v)
    }

    /// Registers a texture (8-bit grayscale input, or R16G16_SINT output in grid units).
    pub fn register(&mut self, resource: &ID3D11Resource) -> Result<Buffer> {
        let reg = self.f.register.context("no register function")?;
        let mut h: Handle = std::ptr::null_mut();
        unsafe { self.check(reg(self.handle, resource.as_raw(), &mut h), "registering a texture")? };
        self.registered.push(h);
        Ok(Buffer(h))
    }

    /// Queues the flow from `from` to `to` into `forward`, and back into `backward`. Vectors
    /// are in full-size pixels times 32 (S10.5). Runs in order with the device context's work.
    /// `continuing`: `from` is the picture `to` was last time (the last flow helps).
    pub fn run(&self, from: Buffer, to: Buffer, forward: Buffer, backward: Buffer, continuing: bool) -> Result<()> {
        let exec = self.f.execute.context("no execute function")?;
        let input = ExecuteInput {
            input_frame: from.0,
            reference_frame: to.0,
            external_hints: std::ptr::null_mut(),
            disable_temporal_hints: if continuing { 0 } else { 1 },
            padding: 0,
            priv_data: std::ptr::null_mut(),
            padding2: 0,
            num_rois: 0,
            roi_data: std::ptr::null_mut(),
        };
        let mut output = ExecuteOutput {
            output: forward.0,
            output_cost: std::ptr::null_mut(),
            priv_data: std::ptr::null_mut(),
            bwd_output: backward.0,
            bwd_output_cost: std::ptr::null_mut(),
            global_flow: std::ptr::null_mut(),
        };
        unsafe { self.check(exec(self.handle, &input, &mut output), "computing optical flow") }
    }

    fn check(&self, status: Status, what: &str) -> Result<()> {
        if status == SUCCESS {
            return Ok(());
        }
        let mut detail = String::new();
        if let Some(last) = self.f.last_error {
            let mut buf = [0u8; 256];
            let mut n = buf.len() as u32;
            if unsafe { last(self.handle, buf.as_mut_ptr(), &mut n) } == SUCCESS {
                detail = String::from_utf8_lossy(&buf[..(n as usize).min(buf.len())]).trim_end_matches('\0').to_string();
            }
        }
        bail!("{what}: NvOF error {status} {detail}")
    }
}

fn check(status: Status, what: &str) -> Result<()> {
    if status == SUCCESS { Ok(()) } else { bail!("{what}: NvOF error {status}") }
}

impl Drop for Flow {
    fn drop(&mut self) {
        unsafe {
            if let Some(unreg) = self.f.unregister {
                for h in self.registered.drain(..) {
                    let _ = unreg(h);
                }
            }
            if let Some(destroy) = self.f.destroy {
                let _ = destroy(self.handle);
            }
        }
    }
}
