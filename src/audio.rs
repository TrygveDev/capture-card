//! Audio passthrough: WASAPI capture from the card's endpoint into a small queue, WASAPI render
//! from it to the default output. Both sides run event-driven at the smallest period the engine
//! allows. The render side is woken by both: sound goes into the device's buffer the moment it
//! arrives, so one margin covers the jitter of both sides, and it's trimmed (or grown, if it ran
//! out) so latency never builds up.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
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
use windows::Win32::System::Threading::{CreateEventW, SetEvent, WaitForMultipleObjects, WaitForSingleObject};
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
    /// When the input last carried sound (not silence).
    pub last_sound: Mutex<Option<Instant>>,
    /// Set as sound arrives in the queue (an event handle).
    arrived: isize,
    /// Sound devices picked in the menu (endpoint IDs; None picks automatically), bumped
    /// `picks` when either changes, and what's in use: input and output names.
    input: Mutex<Option<String>>,
    output: Mutex<Option<String>>,
    picks: AtomicU32,
    pub using: Mutex<(String, String)>,
}

/// Quieter than this (-60 dB) is silence.
const SILENCE: f32 = 0.001;

impl AudioShared {
    pub fn new(volume: f32, muted: bool, input: Option<String>, output: Option<String>) -> Arc<Self> {
        Arc::new(AudioShared {
            queue: Mutex::new(VecDeque::with_capacity(48_000)),
            rate: AtomicU32::new(0),
            packet: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
            volume: AtomicU32::new(volume.to_bits()),
            muted: AtomicBool::new(muted),
            last_sound: Mutex::new(None),
            arrived: unsafe { CreateEventW(None, false, false, None) }.expect("create an event").0 as isize,
            status: Mutex::new(String::new()),
            input: Mutex::new(input),
            output: Mutex::new(output),
            picks: AtomicU32::new(0),
            using: Mutex::new((String::new(), String::new())),
        })
    }

    /// The card's sound from this input (None: found automatically); taken up at once.
    pub fn set_input(&self, id: Option<String>) {
        *self.input.lock().unwrap() = id;
        self.picks.fetch_add(1, Ordering::AcqRel);
    }

    /// Played on this output (None: Windows' default); taken up at once.
    pub fn set_output(&self, id: Option<String>) {
        *self.output.lock().unwrap() = id;
        self.picks.fetch_add(1, Ordering::AcqRel);
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
            // The video says which device the card is: give it a moment to open one first.
            let started = Instant::now();
            while device.is_none() && CARD.lock().unwrap().is_none() && started.elapsed() < Duration::from_secs(3) {
                std::thread::sleep(Duration::from_millis(50));
            }
            // Logged once per new error: a card that's unplugged fails every second.
            let mut last = String::new();
            loop {
                if let Err(e) = run_capture(&s, device.as_deref()) {
                    let e = format!("{e:#}");
                    if e != last {
                        crate::log::line(&format!("audio capture: {e}"));
                        last = e.clone();
                    }
                    *s.status.lock().unwrap() = e;
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
            let mut last = String::new();
            loop {
                if let Err(e) = run_render(&shared) {
                    let e = format!("{e:#}");
                    if e != last {
                        crate::log::line(&format!("audio render: {e}"));
                        last = e;
                    }
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

/// A sound device for the menu: its endpoint ID and name.
pub struct Endpoint {
    pub id: String,
    pub name: String,
}

/// The connected sound inputs (or outputs), for the menu.
pub fn endpoints(inputs: bool) -> Vec<Endpoint> {
    com_init();
    let Ok(e) = enumerator() else { return Vec::new() };
    let flow = if inputs { eCapture } else { eRender };
    let Ok(col) = (unsafe { e.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE) }) else { return Vec::new() };
    let count = unsafe { col.GetCount() }.unwrap_or(0);
    let mut out: Vec<Endpoint> = (0..count)
        .filter_map(|i| unsafe { col.Item(i) }.ok())
        .map(|d| Endpoint { id: device_id(&d), name: friendly_name(&d) })
        .collect();
    out.sort_by_key(|d| d.name.to_lowercase());
    out
}

/// The active device of `flow` with this endpoint ID.
fn device_by_id(e: &IMMDeviceEnumerator, flow: EDataFlow, id: &str) -> Option<IMMDevice> {
    let col = unsafe { e.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE) }.ok()?;
    (0..unsafe { col.GetCount() }.ok()?).filter_map(|i| unsafe { col.Item(i) }.ok()).find(|d| device_id(d) == id)
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

/// The video device Passthru opened: its name, and which physical device it is (devices.rs).
static CARD: Mutex<Option<(String, Option<windows::core::GUID>)>> = Mutex::new(None);

/// Called when the video opens a device: its sound is looked for on the same physical device.
pub fn set_card(name: &str, container: Option<windows::core::GUID>) {
    *CARD.lock().unwrap() = Some((name.to_string(), container));
}

/// The capture card's sound input, and how it was recognised (for the log). Best is the input
/// on the same physical device as the video (any card, whatever its sound is called), then one
/// named like the video device, then known card names.
fn find_capture(e: &IMMDeviceEnumerator, wanted: Option<&str>) -> Result<(IMMDevice, &'static str)> {
    let card = CARD.lock().unwrap().clone();
    let col = unsafe { e.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)? };
    let mut best: Option<(u8, IMMDevice)> = None;
    let mut names = Vec::new();
    for i in 0..unsafe { col.GetCount()? } {
        let d = unsafe { col.Item(i)? };
        let name = friendly_name(&d);
        let lower = name.to_lowercase();
        names.push(name);
        let score = match wanted {
            Some(w) if lower.contains(&w.to_lowercase()) => 5,
            Some(_) => 0,
            None => {
                let same_device = card.as_ref().and_then(|(_, c)| *c).is_some_and(|c| {
                    crate::devices::container_of_instance(&format!(r"SWD\MMDEVAPI\{}", device_id(&d))) == Some(c)
                });
                let video_name = card.as_ref().map(|(n, _)| n.to_lowercase()).filter(|n| n.len() >= 3);
                if same_device {
                    4
                } else if video_name.is_some_and(|n| lower.contains(&n)) {
                    3
                } else if lower.contains("hd60") {
                    2
                } else if lower.contains("elgato") && !lower.contains("virtual") {
                    1
                } else {
                    0
                }
            }
        };
        if score > 0 && best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((score, d));
        }
    }
    let Some((score, device)) = best else {
        let video = card.map_or_else(|| "none yet".to_string(), |(n, c)| format!("{n}{}", if c.is_some() { "" } else { ", device unknown" }));
        bail!("Capture card audio input not found (video: {video}; inputs: {})", names.join(", "));
    };
    let how = match score {
        5 => "as asked",
        4 => "same device as the video",
        3 => "named like the video",
        _ => "by name",
    };
    Ok((device, how))
}

/// An initialised, event-driven shared-mode stream.
struct Stream {
    /// How it was opened (for the log): low-latency, standard, or converted by Windows.
    how: &'static str,
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
/// The device's mix format gets the low-latency IAudioClient3 path when the driver offers one
/// and `low_latency` allows it.
fn open_stream(device: &IMMDevice, want_rate: Option<u32>, low_latency: bool) -> Result<Stream> {
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
                if let Some(c3) = low_latency.then(|| client.cast::<IAudioClient3>().ok()).flatten() {
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
                        return finish(client.clone(), event, mix_rate, mix_ch, mix_float, min, "low-latency");
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
                return finish(client.clone(), event, mix_rate, mix_ch, mix_float, period, "standard");
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
            finish(client.clone(), event, rate, 2, true, period, "converted")
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
    how: &'static str,
) -> Result<Stream> {
    unsafe {
        client.SetEventHandle(event)?;
        let buffer_frames = client.GetBufferSize()?;
        Ok(Stream { how, client, event, rate, channels, float, buffer_frames, period_frames })
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
    let picks = shared.picks.load(Ordering::Acquire);
    let picked = shared.input.lock().unwrap().clone();
    let found = picked.as_deref().and_then(|id| device_by_id(&e, eCapture, id));
    // A picked input that isn't connected is looked for again now and then.
    let missing = picked.is_some() && found.is_none();
    let (device, how) = match found {
        Some(d) => (d, "picked in the menu"),
        None => {
            let (d, how) = find_capture(&e, wanted)?;
            (d, if picked.is_some() { "the one picked isn't connected" } else { how })
        }
    };
    let name = friendly_name(&device);
    let stream = open_stream(&device, None, true)?;
    let capture: IAudioCaptureClient = unsafe { stream.client.GetService()? };
    crate::log::line(&format!(
        "audio in: {name} ({how})  {} Hz {}ch, {}, period {} frames ({:.2} ms)",
        stream.rate,
        stream.channels,
        stream.how,
        stream.period_frames,
        stream.period_frames as f64 * 1000.0 / stream.rate as f64
    ));
    shared.queue.lock().unwrap().clear();
    shared.packet.store(stream.period_frames as usize, Ordering::Relaxed);
    shared.rate.store(stream.rate, Ordering::Release);
    shared.generation.fetch_add(1, Ordering::AcqRel);
    shared.status.lock().unwrap().clear();
    shared.using.lock().unwrap().0 = name.clone();
    unsafe { stream.client.Start()? };

    // Keep at most this much queued if the output side stalls.
    let cap = stream.rate as usize / 5 * 2;
    let mut scratch: Vec<f32> = Vec::with_capacity(4096);
    let mut looked = Instant::now();
    loop {
        // Another input picked in the menu, or the picked one is back: reopen.
        if shared.picks.load(Ordering::Acquire) != picks {
            return Ok(());
        }
        if missing && looked.elapsed() >= Duration::from_secs(3) {
            looked = Instant::now();
            if picked.as_deref().and_then(|id| device_by_id(&e, eCapture, id)).is_some() {
                return Ok(());
            }
        }
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
            if scratch.iter().any(|v| v.abs() > SILENCE) {
                *shared.last_sound.lock().unwrap() = Some(Instant::now());
            }
            let mut q = shared.queue.lock().unwrap();
            q.extend(scratch.iter().copied());
            if q.len() > cap {
                let excess = q.len() - cap / 4;
                q.drain(..excess & !1);
            }
            drop(q);
            unsafe {
                let _ = SetEvent(HANDLE(shared.arrived as _));
            }
        }
    }
}

/// The margin (how late a packet from the card may come without a gap): at first, what running
/// dry adds to it, and the most it grows to.
const MARGIN_MS: usize = 6;
const MARGIN_STEP_MS: usize = 1;
const MARGIN_MOST_MS: usize = 30;
/// A margin that grew comes back a step after this long without running dry (one busy moment
/// mustn't mean more delay for the rest of the session).
const MARGIN_BACK: Duration = Duration::from_secs(600);
/// Too little is put right after this long, too much only after this long of it.
const ADD_AFTER: Duration = Duration::from_millis(500);
const SHED_AFTER: Duration = Duration::from_secs(5);

/// Adjusting changes the speed by this fraction at most: 2%, half a second for a 10 ms period,
/// as the two clocks drift past each other now and then.
const ADJUST_ONE_IN: usize = 48;

/// How much to have buffered as a packet lands, not counting it, for the next to be allowed
/// `margin` late: `ahead` is until the engine next takes a `period` (all in frames).
///
/// The engine takes whole periods, so between those more buys nothing but delay: the least is
/// whole periods and `guard`, for the clocks' drift.
fn needed(ahead: usize, margin: usize, period: usize, guard: usize) -> usize {
    margin.saturating_sub(ahead).div_ceil(period) * period + guard
}

/// Keeps what's buffered as short as both sides' timing allows.
struct Jitter {
    /// How late a packet may come, in frames. Grows when the output runs dry.
    margin: usize,
    /// The least more than needed that was buffered as a packet landed, lately (negative: too
    /// little), and since the last time too much was checked.
    least: isize,
    least_long: isize,
    window: Instant,
    long_window: Instant,
    /// Frames still to skip (too much buffered: clock drift, start-up bursts) or to add (too
    /// little).
    adjust: isize,
    prefilling: bool,
}

impl Jitter {
    fn new(margin: usize, now: Instant) -> Self {
        Jitter {
            margin,
            least: isize::MAX,
            least_long: isize::MAX,
            window: now,
            long_window: now,
            adjust: 0,
            prefilling: true,
        }
    }

    fn restart_windows(&mut self, now: Instant) {
        self.least = isize::MAX;
        self.least_long = isize::MAX;
        self.window = now;
        self.long_window = now;
        self.adjust = 0;
    }

    /// A packet landed with `over` frames more than needed buffered (negative: too few) and
    /// `spare` more than shedding keeps: says what to add or skip.
    fn landed(&mut self, over: isize, spare: isize, guard: usize, now: Instant) {
        // Part-way through an adjustment the buffer still shows what it's putting right:
        // counted, that would be put right twice.
        if self.adjust != 0 {
            return;
        }
        let guard = guard as isize;
        self.least = self.least.min(over);
        self.least_long = self.least_long.min(spare);
        if now.saturating_duration_since(self.window) < ADD_AFTER {
            return;
        }
        if self.least < -guard / 2 {
            let short = self.least;
            self.restart_windows(now);
            self.adjust = -short;
        } else {
            self.least = isize::MAX;
            self.window = now;
            if now.saturating_duration_since(self.long_window) >= SHED_AFTER {
                let extra = self.least_long;
                self.restart_windows(now);
                if extra > guard {
                    self.adjust = -extra;
                }
            }
        }
    }

    /// `changed` frames of the adjustment were just made.
    fn adjusted(&mut self, changed: isize, now: Instant) {
        self.adjust -= changed;
        if changed != 0 && self.adjust == 0 {
            // Only what's buffered from here on counts.
            self.restart_windows(now);
        }
    }
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
    let picks = shared.picks.load(Ordering::Acquire);
    let e = enumerator()?;
    // The output picked in the menu, else (or while it isn't connected) Windows' default.
    let picked_id = shared.output.lock().unwrap().clone();
    let picked = picked_id.as_deref().and_then(|id| device_by_id(&e, eRender, id));
    let follows_default = picked.is_none();
    let mut looked = Instant::now();
    let device = match picked {
        Some(d) => d,
        None => unsafe { e.GetDefaultAudioEndpoint(eRender, eConsole)? },
    };
    let id = device_id(&device);
    let stream = open_stream(&device, Some(rate), true)?;
    if stream.rate != rate {
        bail!("output opened at {} Hz, wanted {rate}", stream.rate);
    }
    let render: IAudioRenderClient = unsafe { stream.client.GetService()? };
    let out_name = friendly_name(&device);
    shared.using.lock().unwrap().1 = out_name.clone();
    crate::log::line(&format!(
        "audio out: {} ({})  {} Hz {}ch, {}, period {} frames ({:.2} ms), buffer {} frames",
        out_name,
        if follows_default { "Windows' default" } else { "picked in the menu" },
        stream.rate,
        stream.channels,
        stream.how,
        stream.period_frames,
        stream.period_frames as f64 * 1000.0 / stream.rate as f64,
        stream.buffer_frames
    ));
    let ms = |m: usize| rate as usize * m / 1000;
    let period = (stream.period_frames as usize).max(1);
    let guard = ms(1);
    let mut jit = Jitter::new(ms(MARGIN_MS), Instant::now());
    let mut gain = 0.0f32;
    // Queued still after the last write (the device's buffer was full).
    let mut carried = 0usize;
    let mut engine_at = Instant::now();
    // In the device's buffer after the last look and write.
    let mut held = 0usize;
    let mut last_check = Instant::now();
    let mut grown_at = Instant::now();
    let mut dry = 0u32;
    let mut dry_logged = Instant::now();
    let wakes = [stream.event, HANDLE(shared.arrived as _)];
    unsafe { stream.client.Start()? };

    loop {
        let woke = unsafe { WaitForMultipleObjects(&wakes, false, 100) };
        let now = Instant::now();
        if woke == WAIT_OBJECT_0 {
            // The engine has just taken a period.
            engine_at = now;
        }

        if last_check.elapsed() >= Duration::from_millis(500) {
            last_check = now;
            // Follow default-device, capture and menu changes by reopening.
            if shared.generation.load(Ordering::Acquire) != generation || shared.picks.load(Ordering::Acquire) != picks {
                return Ok(());
            }
            if follows_default
                && let Ok(d) = unsafe { e.GetDefaultAudioEndpoint(eRender, eConsole) }
                && device_id(&d) != id
            {
                return Ok(());
            }
            // The picked output is connected again.
            if follows_default && picked_id.is_some() && looked.elapsed() >= Duration::from_secs(3) {
                looked = now;
                if picked_id.as_deref().and_then(|i| device_by_id(&e, eRender, i)).is_some() {
                    return Ok(());
                }
            }
        }
        if dry > 0 && dry_logged.elapsed() >= Duration::from_secs(60) {
            crate::log::line(&format!(
                "audio: the output ran dry {dry} times in the last minute (margin now {} ms)",
                jit.margin * 1000 / rate as usize
            ));
            dry = 0;
            dry_logged = now;
        }

        let padding = unsafe { stream.client.GetCurrentPadding()? } as usize;
        // Also when its event isn't here yet: the card's packets can land in the same instant
        // (both follow the audio engine's clock).
        if padding + period / 2 <= held {
            engine_at = now;
        }
        held = padding;
        let mut q = shared.queue.lock().unwrap();
        let mut queued = q.len() / 2;
        if jit.prefilling {
            let start = period + jit.margin;
            if padding + queued < start {
                carried = queued;
                continue;
            }
            // Whatever piled up while starting is old sound: only the newest is played.
            let old = (padding + queued - start).min(queued);
            q.drain(..old * 2);
            queued -= old;
            jit.prefilling = false;
            jit.restart_windows(now);
        } else if padding + carried == 0 {
            // Ran out (the device plays silence meanwhile): a little more margin from here.
            dry += 1;
            jit.prefilling = true;
            jit.margin = (jit.margin + ms(MARGIN_STEP_MS)).min(ms(MARGIN_MOST_MS));
            grown_at = now;
            if padding + queued < period + jit.margin {
                carried = queued;
                continue;
            }
            jit.prefilling = false;
            jit.restart_windows(now);
        } else if queued > carried {
            // A packet landed: how much there was without it, against what that has to be.
            let since = (now.saturating_duration_since(engine_at).as_secs_f64() * rate as f64) as usize;
            let ahead = period.saturating_sub(since);
            let have = (padding + carried) as isize;
            let over = have - needed(ahead, jit.margin, period, guard) as isize;
            // Shedding wants a little over the margin, not to be added straight back.
            let spare = have - needed(ahead, jit.margin + guard, period, guard) as isize;
            jit.landed(over, spare, guard, now);
            if jit.margin > ms(MARGIN_MS) && grown_at.elapsed() >= MARGIN_BACK {
                jit.margin = jit.margin.saturating_sub(ms(MARGIN_STEP_MS)).max(ms(MARGIN_MS));
                grown_at = now;
            }
        }

        // While adjusting one frame in `ADJUST_ONE_IN` is played twice or skipped.
        let space = (stream.buffer_frames as usize).saturating_sub(padding);
        let some = (queued / ADJUST_ONE_IN) as isize;
        let change = jit.adjust.clamp(-some, some);
        let write = ((queued as isize + change) as usize).min(space);
        if write == 0 {
            carried = queued;
            continue;
        }
        let data = unsafe { render.GetBuffer(write as u32)? };
        let vol = shared.volume();
        let target = if shared.muted.load(Ordering::Relaxed) { 0.0 } else { vol * vol };
        let step = (target - gain) / write as f32;
        let mut changed = 0isize;
        let mut sample = (0.0f32, 0.0f32);
        for i in 0..write {
            let due = changed != change && i % (ADJUST_ONE_IN - 1) == ADJUST_ONE_IN - 2;
            if due && change < 0 && q.len() >= 4 {
                q.drain(..2);
                changed -= 1;
            }
            if due && change > 0 {
                // The last one again.
                changed += 1;
            } else {
                sample = (q.pop_front().unwrap_or(0.0), q.pop_front().unwrap_or(0.0));
            }
            gain += step;
            write_frame(data, i, &stream, sample.0 * gain, sample.1 * gain);
        }
        gain = target;
        jit.adjusted(changed, now);
        carried = q.len() / 2;
        held += write;
        drop(q);
        unsafe { render.ReleaseBuffer(write as u32, 0)? };
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

#[cfg(test)]
mod tests {
    use super::{ADJUST_ONE_IN, Jitter, needed};
    use std::time::{Duration, Instant};

    /// The card's clock drifting past the engine's leaves it a period short, or with one spare.
    #[test]
    fn makes_up_a_period_once() {
        let (period, guard) = (480isize, 48);
        let a_write = period / ADJUST_ONE_IN as isize;
        for start in [-period, period] {
            let t0 = Instant::now();
            let mut jit = Jitter::new(288, t0);
            let mut over = start;
            let mut furthest = 0;
            // A packet every 10 ms for 20 s.
            for n in 0..2000 {
                let now = t0 + Duration::from_millis(10 * n);
                jit.landed(over, over, guard, now);
                let changed = jit.adjust.clamp(-a_write, a_write);
                jit.adjusted(changed, now);
                over += changed;
                furthest = if start < 0 { furthest.max(over) } else { furthest.min(over) };
            }
            assert_eq!((over, furthest), (0, 0), "starting {start} out");
        }
    }

    #[test]
    fn buffers_whole_periods_for_the_margin() {
        let (period, margin, guard) = (480, 288, 48);
        // The packet lands 8 ms before the engine takes the next period: that's margin enough.
        assert_eq!(needed(384, margin, period, guard), guard);
        // Just before it does: a late one would miss that, so a period more.
        assert_eq!(needed(10, margin, period, guard), period + guard);
        // A margin longer than a period.
        assert_eq!(needed(0, 960, period, guard), 2 * period + guard);
        // A 3 ms engine under a 6 ms margin.
        assert_eq!(needed(0, margin, 144, guard), 2 * 144 + guard);
    }
}
