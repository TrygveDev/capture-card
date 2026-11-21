//! Audio passthrough: WASAPI capture from the card's endpoint into a small queue, WASAPI render
//! from it to the default output. Both sides run event-driven at the smallest period the engine
//! allows, and the render side trims or re-buffers the queue so it never builds up latency.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::*;
use windows::Win32::Media::KernelStreaming::{
    KSDATAFORMAT_SUBTYPE_PCM, SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT, WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::StructuredStorage::PropVariantToStringAlloc;
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, STGM_READ,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

pub struct AudioShared {
    /// Interleaved stereo f32 at `rate`.
    queue: Mutex<VecDeque<f32>>,
    rate: AtomicU32,
    /// Largest capture packet seen, in frames: the queue has to cover one of these.
    packet: AtomicUsize,
    /// Bumped whenever capture (re)opens, so render can resync.
    generation: AtomicU32,
    /// 0..=1, as f32 bits.
    pub volume: AtomicU32,
    pub muted: AtomicBool,
    pub status: Mutex<String>,
}

impl AudioShared {
    pub fn new(volume: f32, muted: bool) -> Arc<Self> {
        Arc::new(AudioShared {
            queue: Mutex::new(VecDeque::with_capacity(48_000)),
            rate: AtomicU32::new(0),
            packet: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
            volume: AtomicU32::new(volume.to_bits()),
            muted: AtomicBool::new(muted),
            status: Mutex::new(String::new()),
        })
    }

    pub fn volume(&self) -> f32 {
        f32::from_bits(self.volume.load(Ordering::Relaxed))
    }

    pub fn set_volume(&self, v: f32) {
        self.volume.store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
}

pub fn spawn(shared: Arc<AudioShared>, device: Option<String>) {
    let s = shared.clone();
    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            com_init();
            crate::raise_thread_priority("Pro Audio");
            loop {
                if let Err(e) = run_capture(&s, device.as_deref()) {
                    eprintln!("audio capture: {e:#}");
                    *s.status.lock().unwrap() = format!("{e:#}");
                }
                s.rate.store(0, Ordering::Release);
                std::thread::sleep(Duration::from_secs(1));
            }
        })
        .expect("spawn audio capture");
    std::thread::Builder::new()
        .name("audio-render".into())
        .spawn(move || {
            com_init();
            crate::raise_thread_priority("Pro Audio");
            loop {
                if let Err(e) = run_render(&shared) {
                    eprintln!("audio render: {e:#}");
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
        })
        .expect("spawn audio render");
}

fn com_init() {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }
}

fn enumerator() -> Result<IMMDeviceEnumerator> {
    Ok(unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? })
}

fn friendly_name(device: &IMMDevice) -> String {
    unsafe {
        let Ok(store) = device.OpenPropertyStore(STGM_READ) else {
            return String::new();
        };
        let Ok(value) = store.GetValue(&PKEY_Device_FriendlyName) else {
            return String::new();
        };
        match PropVariantToStringAlloc(&value) {
            Ok(p) => {
                let s = p.to_string().unwrap_or_default();
                CoTaskMemFree(Some(p.0 as _));
                s
            }
            Err(_) => String::new(),
        }
    }
}

fn device_id(device: &IMMDevice) -> String {
    unsafe {
        match device.GetId() {
            Ok(p) => {
                let s = p.to_string().unwrap_or_default();
                CoTaskMemFree(Some(p.0 as _));
                s
            }
            Err(_) => String::new(),
        }
    }
}

/// Every capture endpoint, for `--list`.
pub fn list() -> Result<()> {
    com_init();
    let e = enumerator()?;
    for (flow, label) in [(eCapture, "audio input"), (eRender, "audio output")] {
        let col = unsafe { e.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE)? };
        for i in 0..unsafe { col.GetCount()? } {
            let d = unsafe { col.Item(i)? };
            println!("{label}: {}", friendly_name(&d));
        }
    }
    Ok(())
}

fn find_capture(e: &IMMDeviceEnumerator, wanted: Option<&str>) -> Result<IMMDevice> {
    let col = unsafe { e.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)? };
    let mut best: Option<(u8, IMMDevice)> = None;
    for i in 0..unsafe { col.GetCount()? } {
        let d = unsafe { col.Item(i)? };
        let name = friendly_name(&d).to_lowercase();
        let score = match wanted {
            Some(w) if name.contains(&w.to_lowercase()) => 3,
            Some(_) => 0,
            None if name.contains("hd60") => 3,
            None if name.contains("elgato") && !name.contains("virtual") => 2,
            None => 0,
        };
        if score > 0 && best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((score, d));
        }
    }
    best.map(|(_, d)| d)
        .context("Capture card audio input not found")
}

/// An initialised, event-driven shared-mode stream.
struct Stream {
    client: IAudioClient,
    event: HANDLE,
    rate: u32,
    channels: usize,
    float: bool,

    buffer_frames: u32,
    period_frames: u32,
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = self.client.Stop();
            let _ = CloseHandle(self.event);
        }
    }
}

fn stereo_float(rate: u32) -> WAVEFORMATEXTENSIBLE {
    WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_EXTENSIBLE as u16,
            nChannels: 2,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * 8,
            nBlockAlign: 8,
            wBitsPerSample: 32,
            cbSize: 22,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 { wValidBitsPerSample: 32 },
        dwChannelMask: SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT,
        SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    }
}

/// Reads channel count, rate and sample type out of a WAVEFORMATEX(TENSIBLE).
unsafe fn parse_format(f: *const WAVEFORMATEX) -> (u32, usize, bool, u16) {
    unsafe {
        let fx = &*f;
        let float = if fx.wFormatTag as u32 == WAVE_FORMAT_EXTENSIBLE {
            let ext = &*(f as *const WAVEFORMATEXTENSIBLE);
            let sub = ext.SubFormat;
            if sub != KSDATAFORMAT_SUBTYPE_IEEE_FLOAT && sub != KSDATAFORMAT_SUBTYPE_PCM {
                return (fx.nSamplesPerSec, fx.nChannels as usize, false, 0);
            }
            sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        } else {
            fx.wFormatTag == 3 // WAVE_FORMAT_IEEE_FLOAT
        };
        (fx.nSamplesPerSec, fx.nChannels as usize, float, fx.wBitsPerSample)
    }
}

/// Opens `device` in shared mode. With `want_rate` set the stream must run at that rate (the
/// engine resamples if the device doesn't); otherwise it runs at the device's own mix format.
/// The device's mix format gets the low-latency IAudioClient3 path when the driver offers one.
fn open_stream(device: &IMMDevice, want_rate: Option<u32>) -> Result<Stream> {
    unsafe {
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let mix = client.GetMixFormat()?;
        let (mix_rate, mix_ch, mix_float, mix_bits) = parse_format(mix);
        let usable_mix = (mix_float && mix_bits == 32) || (!mix_float && mix_bits == 16);
        let event = CreateEventW(None, false, false, None)?;

        let native = usable_mix && want_rate.is_none_or(|r| r == mix_rate);
        let result = (|| -> Result<Stream> {
            if native {
                // Smallest period the shared engine supports for this format.
                if let Ok(c3) = client.cast::<IAudioClient3>() {
                    let (mut def, mut fund, mut min, mut max) = (0, 0, 0, 0);
                    if c3
                        .GetSharedModeEnginePeriod(mix, &mut def, &mut fund, &mut min, &mut max)
                        .is_ok()
                        && c3
                            .InitializeSharedAudioStream(
                                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                min,
                                mix,
                                None,
                            )
                            .is_ok()
                    {
                        return finish(client.clone(), event, mix_rate, mix_ch, mix_float, min);
                    }
                }
                client.Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                    0,
                    0,
                    mix,
                    None,
                )?;
                let mut def = 0i64;
                client.GetDevicePeriod(Some(&mut def), None)?;
                let period = (def as u64 * mix_rate as u64 / 10_000_000) as u32;
                return finish(client.clone(), event, mix_rate, mix_ch, mix_float, period);
            }
            let rate = want_rate.unwrap_or(48_000);
            let fmt = stereo_float(rate);
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                    | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
                    | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                0,
                0,
                &fmt.Format,
                None,
            )?;
            let mut def = 0i64;
            client.GetDevicePeriod(Some(&mut def), None)?;
            let period = (def as u64 * rate as u64 / 10_000_000) as u32;
            finish(client.clone(), event, rate, 2, true, period)
        })();
        CoTaskMemFree(Some(mix as _));
        if result.is_err() {
            let _ = CloseHandle(event);
        }
        result
    }
}

unsafe fn finish(
    client: IAudioClient,
    event: HANDLE,
    rate: u32,
    channels: usize,
    float: bool,

    period_frames: u32,
) -> Result<Stream> {
    unsafe {
        client.SetEventHandle(event)?;
        let buffer_frames = client.GetBufferSize()?;
        Ok(Stream { client, event, rate, channels, float, buffer_frames, period_frames })
    }
}

/// Converts one interleaved device frame to stereo.
#[inline]
fn read_frame(data: *const u8, i: usize, s: &Stream) -> (f32, f32) {
    unsafe {
        let ch = s.channels;
        let get = |c: usize| -> f32 {
            if s.float {
                *(data as *const f32).add(i * ch + c)
            } else {
                *(data as *const i16).add(i * ch + c) as f32 / 32768.0
            }
        };
        if ch == 1 {
            let v = get(0);
            (v, v)
        } else {
            (get(0), get(1))
        }
    }
}

fn run_capture(shared: &AudioShared, wanted: Option<&str>) -> Result<()> {
    let e = enumerator()?;
    let device = find_capture(&e, wanted)?;
    let name = friendly_name(&device);
    let stream = open_stream(&device, None)?;
    let capture: IAudioCaptureClient = unsafe { stream.client.GetService()? };
    eprintln!(
        "audio in: {name}  {} Hz {}ch, period {} frames ({:.2} ms)",
        stream.rate,
        stream.channels,
        stream.period_frames,
        stream.period_frames as f64 * 1000.0 / stream.rate as f64
    );
    shared.queue.lock().unwrap().clear();
    shared.packet.store(stream.period_frames as usize, Ordering::Relaxed);
    shared.rate.store(stream.rate, Ordering::Release);
    shared.generation.fetch_add(1, Ordering::AcqRel);
    shared.status.lock().unwrap().clear();
    unsafe { stream.client.Start()? };

    // Keep at most this much queued if the output side stalls.
    let cap = stream.rate as usize / 5 * 2;
    let mut scratch: Vec<f32> = Vec::with_capacity(4096);
    loop {
        if unsafe { WaitForSingleObject(stream.event, 500) } != WAIT_OBJECT_0 {
            continue;
        }
        loop {
            let n = unsafe { capture.GetNextPacketSize()? };
            if n == 0 {
                break;
            }
            let mut data = std::ptr::null_mut();
            let mut frames = 0u32;
            let mut flags = 0u32;
            unsafe { capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None)? };
            scratch.clear();
            if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                scratch.resize(frames as usize * 2, 0.0);
            } else {
                for i in 0..frames as usize {
                    let (l, r) = read_frame(data, i, &stream);
                    scratch.push(l);
                    scratch.push(r);
                }
            }
            unsafe { capture.ReleaseBuffer(frames)? };
            shared.packet.fetch_max(frames as usize, Ordering::Relaxed);
            let mut q = shared.queue.lock().unwrap();
            q.extend(scratch.iter().copied());
            if q.len() > cap {
                let excess = q.len() - cap / 4;
                q.drain(..excess & !1);
            }
        }
    }
}

/// Keeps the queue as short as the capture side's packet timing allows.
struct Jitter {
    /// Safety margin above one capture packet, in frames. Grows on underrun.
    margin: usize,
    trough: usize,
    window: Instant,
    /// Frames to skip to shed accumulated latency (clock drift, startup bursts).
    shed: usize,
    prefilling: bool,
}

fn run_render(shared: &AudioShared) -> Result<()> {
    // Wait for capture to tell us its rate.
    let rate = loop {
        let r = shared.rate.load(Ordering::Acquire);
        if r != 0 {
            break r;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let generation = shared.generation.load(Ordering::Acquire);
    let e = enumerator()?;
    let device = unsafe { e.GetDefaultAudioEndpoint(eRender, eConsole)? };
    let id = device_id(&device);
    let stream = open_stream(&device, Some(rate))?;
    if stream.rate != rate {
        bail!("output opened at {} Hz, wanted {rate}", stream.rate);
    }
    let render: IAudioRenderClient = unsafe { stream.client.GetService()? };
    eprintln!(
        "audio out: {}  {} Hz {}ch, period {} frames ({:.2} ms), buffer {} frames",
        friendly_name(&device),
        stream.rate,
        stream.channels,
        stream.period_frames,
        stream.period_frames as f64 * 1000.0 / stream.rate as f64,
        stream.buffer_frames
    );
    let ms = |m: usize| rate as usize * m / 1000;
    let mut jit = Jitter {
        margin: ms(2),
        trough: usize::MAX,
        window: Instant::now(),
        shed: 0,
        prefilling: true,
    };
    let mut gain = 0.0f32;
    let mut last_check = Instant::now();
    unsafe { stream.client.Start()? };

    loop {
        unsafe { WaitForSingleObject(stream.event, 100) };

        if last_check.elapsed() >= Duration::from_millis(500) {
            last_check = Instant::now();
            // Follow default-device and capture changes by reopening.
            if shared.generation.load(Ordering::Acquire) != generation {
                return Ok(());
            }
            if let Ok(d) = unsafe { e.GetDefaultAudioEndpoint(eRender, eConsole) }
                && device_id(&d) != id
            {
                return Ok(());
            }
        }

        let padding = unsafe { stream.client.GetCurrentPadding()? };
        let avail = stream.buffer_frames.saturating_sub(padding) as usize;
        if avail == 0 {
            continue;
        }
        let data = unsafe { render.GetBuffer(avail as u32)? };

        let vol = shared.volume();
        let target = if shared.muted.load(Ordering::Relaxed) { 0.0 } else { vol * vol };
        let step = (target - gain) / avail as f32;

        let packet = shared.packet.load(Ordering::Relaxed);
        let mut q = shared.queue.lock().unwrap();
        let frames_queued = q.len() / 2;
        if jit.prefilling && frames_queued >= packet + jit.margin {
            jit.prefilling = false;
            jit.trough = usize::MAX;
            jit.window = Instant::now();
        }
        let mut underrun = false;
        for i in 0..avail {
            let (mut l, mut r) = (0.0f32, 0.0f32);
            if !jit.prefilling {
                // Skip one frame in 16 while shedding: a ~6% speed-up for a few ms, inaudible.
                if jit.shed > 0 && i % 16 == 0 && q.len() >= 4 {
                    q.drain(..2);
                    jit.shed -= 1;
                }
                match (q.pop_front(), q.pop_front()) {
                    (Some(a), Some(b)) => (l, r) = (a, b),
                    _ => underrun = true,
                }
            }
            gain += step;
            l *= gain;
            r *= gain;
            write_frame(data, i, &stream, l, r);
        }
        gain = target;
        let level = q.len() / 2;
        drop(q);
        unsafe { render.ReleaseBuffer(avail as u32, 0)? };

        if underrun {
            jit.prefilling = true;
            jit.margin = (jit.margin + ms(1)).min(ms(30));
            continue;
        }
        if !jit.prefilling {
            jit.trough = jit.trough.min(level);
            if jit.window.elapsed() >= Duration::from_millis(500) {
                // The trough is what's left just before the next capture packet lands; anything
                // above the margin there is pure latency, so shed it gradually.
                if jit.trough != usize::MAX && jit.trough > jit.margin + ms(1) {
                    jit.shed = jit.trough - jit.margin;
                }
                jit.trough = usize::MAX;
                jit.window = Instant::now();
            }
        }
    }
}

#[inline]
fn write_frame(data: *mut u8, i: usize, s: &Stream, l: f32, r: f32) {
    unsafe {
        let ch = s.channels;
        for c in 0..ch {
            let v = match (ch, c) {
                (1, _) => (l + r) * 0.5,
                (_, 0) => l,
                (_, 1) => r,
                _ => 0.0,
            };
            if s.float {
                *(data as *mut f32).add(i * ch + c) = v;
            } else {
                *(data as *mut i16).add(i * ch + c) = (v.clamp(-1.0, 1.0) * 32767.0) as i16;
            }
        }
    }
}
