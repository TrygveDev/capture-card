//! Video capture through Media Foundation's synchronous source reader in low-latency mode.
//! Raw NV12/YUY2 frames are converted here; anything else (MJPG, P010, ...) is converted by
//! Media Foundation's video processor to RGB32 first.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures::channel::mpsc::UnboundedSender;
use gpui::RenderImage;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree};
use windows::core::{GUID, Interface, PWSTR};

use crate::convert::{self, Decode, Hdr, Matrix, Sdr};
use crate::gpu::Decoding;
use crate::present::{self, Presenter, RawFrame};

pub struct VideoShared {
    /// Newest converted frame and when it was ready, waiting for the UI to take it.
    pub latest: Mutex<Option<(Arc<RenderImage>, Instant)>>,
    /// Frames converted so far, and the total time spent converting them (for the stats).
    pub produced: std::sync::atomic::AtomicU64,
    pub convert_ns: std::sync::atomic::AtomicU64,
    /// Set when a frame is waiting for the UI, so we wake it at most once per UI frame.
    pub pending: AtomicBool,
    pub waker: UnboundedSender<()>,
    pub status: Mutex<String>,
    /// Short description of the current mode for the controls, like "1440p60".
    pub mode: Mutex<String>,
    pub last_frame: Mutex<Option<Instant>>,
    /// When a frame last had a picture in it. Frames alone don't mean a signal: the Elgato
    /// 4K S keeps sending them with nothing on its input (black, then a slate: no_signal.rs).
    pub last_picture: Mutex<Option<Instant>>,
    /// Width / height of the incoming video, as f32 bits.
    pub aspect: std::sync::atomic::AtomicU32,
    /// Decode the signal as HDR10 (PQ, BT.2020) and tone map it.
    pub hdr: AtomicBool,
    /// Width in device pixels the UI draws the video at (0 = unknown), for downscaling.
    pub display_w: std::sync::atomic::AtomicU32,
    /// What the FPS counter shows, measured by whichever path draws.
    pub meter: crate::meter::Meter,
    /// The GPU path (drawing, smoothing) and its state.
    pub gpu: present::Shared,
    /// Which capture mode, and what the device offers.
    pub control: Control,
}

/// A capture mode as the menu lists it: width, height, frames per second.
pub type ModeKey = (u32, u32, u32);

/// The capture mode, shared with the UI.
#[derive(Default)]
pub struct Control {
    /// Modes the open device offers, best first.
    pub modes: Mutex<Vec<ModeKey>>,
    /// The mode picked in the menu; None takes the best one offered.
    pub wanted: Mutex<Option<ModeKey>>,
    /// Width and height of the frames arriving now, packed (0 before the first).
    size: std::sync::atomic::AtomicU64,
    /// Set to reopen the device (a mode was picked).
    reopen: AtomicBool,
    /// The range picked in the menu.
    range: Mutex<ColourRange>,
    /// What the picture says: 0 not sure yet, 1 limited, 2 full.
    detected: std::sync::atomic::AtomicU8,
    /// Decoding as full range now.
    pub full_now: AtomicBool,
}

impl Control {
    pub fn set_range(&self, range: ColourRange) {
        *self.range.lock().unwrap() = range;
    }

    /// The range the picture looks like (None until it's clear).
    pub fn detected_full(&self) -> Option<bool> {
        match self.detected.load(Ordering::Relaxed) {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }

    /// Switches to `mode` (None: the best offered), reopening the device.
    pub fn pick(&self, mode: Option<ModeKey>) {
        *self.wanted.lock().unwrap() = mode;
        self.reopen.store(true, Ordering::Release);
    }

    /// What the menu shows for Auto: the detected range, or the one the device claims.
    pub fn auto_range_text(&self) -> String {
        match self.detected_full() {
            Some(true) => "Detected: full range (0–255)".into(),
            Some(false) => "Detected: limited range (16–235)".into(),
            None => format!(
                "Using what the card reports ({}) until the picture shows",
                if self.full_now.load(Ordering::Relaxed) { "full" } else { "limited" }
            ),
        }
    }

    /// Width and height of the frames arriving now.
    pub fn size(&self) -> (u32, u32) {
        let v = self.size.load(Ordering::Relaxed);
        ((v >> 32) as u32, v as u32)
    }
}

/// No frames for this long while the device is open: reopen it. The card stops sending when
/// its input changes; reopening picks up the new mode. Doubles while nothing comes, up to 4x.
const STALL: Duration = Duration::from_secs(3);

/// How often a read waiting for a frame looks for a picked mode.
const POLL: Duration = Duration::from_millis(100);

/// A finished read, from the source reader's callback.
struct Sampled {
    status: windows::core::HRESULT,
    flags: u32,
    sample: Option<IMFSample>,
}

// SAFETY: Media Foundation samples are free-threaded, and every thread using one is in the MTA.
unsafe impl Send for Sampled {}

/// Reads are asynchronous so that waiting for a frame can give up: a synchronous ReadSample
/// waits forever when the card stops sending, and shutting the source down doesn't end it.
#[windows::core::implement(IMFSourceReaderCallback)]
struct ReadCallback {
    tx: Mutex<std::sync::mpsc::Sender<Sampled>>,
}

impl IMFSourceReaderCallback_Impl for ReadCallback_Impl {
    fn OnReadSample(
        &self,
        status: windows::core::HRESULT,
        _stream: u32,
        flags: u32,
        _time: i64,
        sample: windows::core::Ref<'_, IMFSample>,
    ) -> windows::core::Result<()> {
        // Nobody listening any more is fine: that reader was given up on.
        let _ = self.tx.lock().unwrap().send(Sampled { status, flags, sample: sample.cloned() });
        Ok(())
    }

    fn OnFlush(&self, _stream: u32) -> windows::core::Result<()> {
        Ok(())
    }

    fn OnEvent(&self, _stream: u32, _event: windows::core::Ref<'_, IMFMediaEvent>) -> windows::core::Result<()> {
        Ok(())
    }
}

/// "1440p60", or "2560x1080 60" for sizes that aren't 16:9.
pub fn mode_label(w: u32, h: u32, fps: u32) -> String {
    if w * 9 == h * 16 { format!("{h}p{fps}") } else { format!("{w}×{h} {fps}") }
}

/// A resolution's name: 1440p, or 3440×1440 for sizes that aren't 16:9.
pub fn resolution_label(w: u32, h: u32) -> String {
    if w * 9 == h * 16 { format!("{h}p") } else { format!("{w}×{h}") }
}

/// The modes on offer grouped by resolution, most pixels first, each with its frame rates
/// fastest first (one row each in the menu).
pub fn by_resolution(modes: &[ModeKey]) -> Vec<((u32, u32), Vec<u32>)> {
    let mut rows: Vec<((u32, u32), Vec<u32>)> = Vec::new();
    for &(w, h, f) in modes {
        match rows.iter_mut().find(|r| r.0 == (w, h)) {
            Some(r) if !r.1.contains(&f) => r.1.push(f),
            Some(_) => {}
            None => rows.push(((w, h), vec![f])),
        }
    }
    rows.sort_by_key(|((w, h), _)| std::cmp::Reverse((w * h, *w)));
    for r in &mut rows {
        r.1.sort_unstable_by(|a, b| b.cmp(a));
    }
    rows
}

#[derive(Clone, Debug)]
pub struct VideoOptions {
    pub device: Option<String>,
    /// (width, height, fps)
    pub mode: Option<(u32, u32, u32)>,
    /// Feed a synthetic moving pattern instead of the capture card, to test the
    /// convert-and-display path on its own: (frame rate, picture changes per second).
    pub test: Option<(u32, u32)>,
    /// The test pattern's width and height (2560x1440 if not given).
    pub test_size: Option<(u32, u32)>,
    /// The test pattern's pictures come unevenly now and then, like a game missing a frame.
    pub test_hitch: bool,
}

#[derive(Clone, Debug)]
struct Mode {
    index: u32,
    width: u32,
    height: u32,
    fps: f64,
    subtype: GUID,
}

impl Mode {
    fn subtype_name(&self) -> String {
        subtype_name(&self.subtype)
    }
}

fn subtype_name(g: &GUID) -> String {
    // Video subtypes are FOURCCs stored in Data1.
    let b = g.data1.to_le_bytes();
    if b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        String::from_utf8_lossy(&b).trim().to_string()
    } else if *g == MFVideoFormat_RGB32 {
        "RGB32".into()
    } else {
        format!("{:?}", g)
    }
}

fn subtype_rank(g: &GUID) -> u8 {
    if *g == MFVideoFormat_NV12 {
        3
    } else if *g == MFVideoFormat_YUY2 {
        2
    } else {
        1
    }
}

pub fn spawn(shared: Arc<VideoShared>, opts: VideoOptions) {
    std::thread::Builder::new()
        .name("video".into())
        .spawn(move || {
            unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                if let Err(e) = MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) {
                    *shared.status.lock().unwrap() = format!("Media Foundation failed: {e}");
                    return;
                }
            }
            crate::raise_thread_priority("Capture");
            let presenter = Presenter::spawn(shared.clone());
            if let Some((fps, content)) = opts.test {
                test_pattern(&shared, &presenter, (fps, content, opts.test_hitch), opts.test_size);
                return;
            }
            // Debug: a short limit makes reads give up during start-up, to test reopening.
            let stall = std::env::var("CAPCARD_STALL_MS").ok().and_then(|v| v.parse().ok()).map_or(STALL, Duration::from_millis);
            // Stalls in a row without a frame: each waits twice as long, up to 4x.
            let mut stalls = 0u32;
            // Kept across reopens: the signal's range rarely changes with them.
            let mut range = RangeGuess::new();
            loop {
                match run(&shared, &presenter, &opts, stall * (1 << stalls.min(2)), &mut range) {
                    Ok(Ended::Reopen) => stalls = 0,
                    Ok(Ended::Switched) => {
                        stalls = 0;
                        range = RangeGuess::new();
                    }
                    Ok(Ended::Stalled { had_frames }) => {
                        stalls = if had_frames { 0 } else { stalls + 1 };
                        range = RangeGuess::new();
                    }
                    Err(e) => {
                        crate::log::line(&format!("video: {e:#}"));
                        *shared.status.lock().unwrap() = e.to_string();
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
            }
        })
        .expect("spawn video thread");
}

/// Why `run` closed the device without an error.
enum Ended {
    /// A mode was picked.
    Reopen,
    /// The device switched mode by itself (its input changed).
    Switched,
    /// No frames came for a while.
    Stalled { had_frames: bool },
}

struct Device {
    name: String,
    activate: IMFActivate,
}

fn enumerate_devices() -> Result<Vec<Device>> {
    unsafe {
        let mut attrs: Option<IMFAttributes> = None;
        MFCreateAttributes(&mut attrs, 1).context("MFCreateAttributes")?;
        let attrs = attrs.context("no attributes")?;
        attrs.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;
        let mut list: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        MFEnumDeviceSources(&attrs, &mut list, &mut count).context("MFEnumDeviceSources")?;
        let mut out = Vec::new();
        for i in 0..count as usize {
            if let Some(activate) = (*list.add(i)).take() {
                let mut name = PWSTR::null();
                let mut len = 0u32;
                let label = if activate
                    .GetAllocatedString(&MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &mut name, &mut len)
                    .is_ok()
                {
                    let s = name.to_string().unwrap_or_default();
                    CoTaskMemFree(Some(name.0 as _));
                    s
                } else {
                    String::from("?")
                };
                out.push(Device { name: label, activate });
            }
        }
        CoTaskMemFree(Some(list as _));
        Ok(out)
    }
}

fn pick_device(devices: Vec<Device>, wanted: Option<&str>) -> Option<Device> {
    let lower = |d: &Device| d.name.to_lowercase();
    if let Some(w) = wanted {
        let w = w.to_lowercase();
        return devices.into_iter().find(|d| lower(d).contains(&w));
    }
    let score = |d: &Device| {
        let n = lower(d);
        if n.contains("virtual") {
            0
        } else if n.contains("hd60") {
            3
        } else if n.contains("elgato") || n.contains("capture") {
            2
        } else {
            1
        }
    };
    devices
        .into_iter()
        .filter(|d| score(d) > 0)
        .max_by_key(|d| score(d))
}

fn native_modes(reader: &IMFSourceReader) -> Vec<Mode> {
    let mut modes = Vec::new();
    for index in 0.. {
        let Ok(t) = (unsafe { reader.GetNativeMediaType(VIDEO_STREAM, index) }) else {
            break;
        };
        if let Some((width, height, fps, subtype)) = describe(&t) {
            modes.push(Mode { index, width, height, fps, subtype });
        }
    }
    modes
}

fn describe(t: &IMFMediaType) -> Option<(u32, u32, f64, GUID)> {
    unsafe {
        let size = t.GetUINT64(&MF_MT_FRAME_SIZE).ok()?;
        let rate = t.GetUINT64(&MF_MT_FRAME_RATE).unwrap_or(0);
        let subtype = t.GetGUID(&MF_MT_SUBTYPE).ok()?;
        let (num, den) = ((rate >> 32) as u32, rate as u32);
        let fps = if den == 0 { 0.0 } else { num as f64 / den as f64 };
        Some(((size >> 32) as u32, size as u32, fps, subtype))
    }
}

const VIDEO_STREAM: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

/// A frame is checked for a picture this often.
const PICTURE_EVERY: u64 = 8;
/// Luma differing by more than this anywhere in a frame is a picture.
const FLAT: u8 = 2;

/// Whether any of `rows` has anything but one flat level, looking at every `step`th byte of
/// each (early out: a real picture answers within a few pixels).
fn varies<'a>(rows: impl Iterator<Item = &'a [u8]>, step: usize) -> bool {
    let (mut lo, mut hi) = (u8::MAX, u8::MIN);
    for row in rows {
        for &v in row.iter().step_by(step) {
            (lo, hi) = (lo.min(v), hi.max(v));
            if hi - lo > FLAT {
                return true;
            }
        }
    }
    false
}

/// Whether `sample` has a picture in it: not flat (looking at some 240 rows of about 320
/// pixels) and not the card's no-signal slate.
fn has_picture(sample: &IMFSample, fmt: &Format) -> Result<bool> {
    let buffer = unsafe {
        if sample.GetBufferCount()? == 1 { sample.GetBufferByIndex(0)? } else { sample.ConvertToContiguousBuffer()? }
    };
    let locked = Locked::new(buffer)?;
    let (bytes, offset) = match fmt.layout {
        Layout::Nv12 => (1, 0),
        Layout::Yuy2 => (2, 0),
        // Green stands in for luma.
        Layout::Rgb32 => (4, 1),
    };
    let (w, h) = (fmt.width, fmt.height);
    let pitch = if locked.pitch != 0 { locked.pitch.unsigned_abs() } else { w * bytes };
    // With rows bottom-up the rows still lie back to back, from the buffer's start.
    let first = if locked.pitch < 0 { 0 } else { locked.scan0 as usize - locked.base as usize };
    let data = locked.bytes();
    let rows = (0..h).step_by((h / 240).max(1)).filter_map(|r| {
        let at = first + r * pitch;
        data.get(at + offset..at + w * bytes)
    });
    if !varies(rows, (w / 320).max(1) * bytes) {
        return Ok(false);
    }
    // The last row's last byte has to be there for the slate's look at it.
    if data.len() < first + (h - 1) * pitch + w * bytes {
        return Ok(true);
    }
    Ok(!crate::no_signal::is_slate(w, h, |x, y| data[first + y * pitch + x * bytes + offset]))
}

/// The cheapest mode to glance at the input in: least data a second, in a format we read.
fn cheapest_mode(modes: &[Mode]) -> Option<Mode> {
    let readable = modes.iter().any(|m| subtype_rank(&m.subtype) > 1);
    modes
        .iter()
        .filter(|m| m.fps > 0.0 && (!readable || subtype_rank(&m.subtype) > 1))
        .min_by_key(|m| ((m.width * m.height) as f64 * m.fps) as u64)
        .cloned()
}

/// From the tray's glances at the input: when a picture has arrived.
struct Arrival {
    /// The input has been without a picture since the watch began (or didn't need to be).
    armed: bool,
    seen: u32,
    missing: u32,
}

impl Arrival {
    /// Glances in a row it takes to believe a picture.
    const SURE: u32 = 2;
    /// And that it's gone, when it was there at closing: longer than a black loading screen, or
    /// the black while the card switches to the watching mode.
    const GONE: u32 = 15;

    fn new(wait_for_change: bool) -> Self {
        Arrival { armed: !wait_for_change, seen: 0, missing: 0 }
    }

    /// Takes one glance; true once a picture has arrived.
    fn glance(&mut self, picture: bool) -> bool {
        if picture {
            self.seen += 1;
            self.missing = 0;
        } else {
            self.missing += 1;
            self.seen = 0;
            self.armed |= self.missing >= Self::GONE;
        }
        self.armed && self.seen >= Self::SURE
    }
}

/// The tray looks at the input this often.
const GLANCE_EVERY: Duration = Duration::from_secs(1);
/// And tries a card that wouldn't open (unplugged, or another program has it) this often.
const RETRY_EVERY: Duration = Duration::from_secs(15);

/// For the tray: watches the card as cheaply as it can be until a picture arrives (true) or
/// `stop` says to leave (false). Closed with a picture on (`wait_for_change`), only one that
/// comes after that picture has gone counts.
pub fn watch_picture(device: Option<&str>, wait_for_change: bool, stop: &std::sync::mpsc::Receiver<()>) -> bool {
    use std::sync::mpsc::RecvTimeoutError::Timeout;
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        if let Err(e) = MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) {
            crate::log::line(&format!("tray: Media Foundation failed: {e}"));
            return false;
        }
    }
    let mut arrival = Arrival::new(wait_for_change);
    let mut complaint = String::new();
    loop {
        let watched = (|| -> Result<Option<bool>> {
            let mut o = open_mode(device, cheapest_mode)?;
            let m = &o.mode;
            crate::log::line(&format!("tray: watching {} at {}", o.name, mode_label(m.width, m.height, m.fps.round() as u32)));
            // The first frames after opening can be left over from before.
            for _ in 0..3 {
                read(&mut o, None, Duration::from_secs(1))?;
            }
            loop {
                if stop.recv_timeout(GLANCE_EVERY) != Err(Timeout) {
                    return Ok(Some(false));
                }
                let picture = match read(&mut o, None, Duration::from_millis(500))? {
                    Read::Frame(s) => has_picture(&s, &o.fmt)?,
                    Read::Nothing | Read::Stalled => false,
                    // The input changed: look again in what it offers now.
                    Read::ModeChanged(_) | Read::Reopen => return Ok(None),
                };
                if arrival.glance(picture) {
                    crate::log::line("tray: a picture arrived");
                    return Ok(Some(true));
                }
            }
        })();
        let pause = match watched {
            Ok(Some(arrived)) => return arrived,
            Ok(None) => GLANCE_EVERY,
            Err(e) => {
                let text = format!("tray: can't watch the card: {e:#}");
                if text != complaint {
                    crate::log::line(&text);
                    complaint = text;
                }
                RETRY_EVERY
            }
        };
        if stop.recv_timeout(pause) != Err(Timeout) {
            return false;
        }
    }
}

/// Prints every capture device and its modes (for `--list`).
pub fn list() -> Result<()> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET)?;
    }
    for d in enumerate_devices()? {
        println!("video device: {}", d.name);
        if let Ok((reader, _source)) = open_reader(&d, None) {
            for m in native_modes(&reader) {
                println!(
                    "    {}x{} @ {:.2} fps  {}",
                    m.width,
                    m.height,
                    m.fps,
                    m.subtype_name()
                );
                if let Ok(t) = unsafe { reader.GetNativeMediaType(VIDEO_STREAM, m.index) } {
                    let get = |k: &GUID| unsafe { t.GetUINT32(k) }.ok();
                    println!(
                        "        transfer {:?} primaries {:?} matrix {:?} range {:?}",
                        get(&MF_MT_TRANSFER_FUNCTION),
                        get(&MF_MT_VIDEO_PRIMARIES),
                        get(&MF_MT_YUV_MATRIX),
                        get(&MF_MT_VIDEO_NOMINAL_RANGE),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Opens a reader on `device`, asynchronous if given a callback.
fn open_reader(
    device: &Device,
    callback: Option<&IMFSourceReaderCallback>,
) -> Result<(IMFSourceReader, IMFMediaSource)> {
    unsafe {
        let source: IMFMediaSource = device
            .activate
            .ActivateObject()
            .with_context(|| format!("could not open {}", device.name))?;
        let mut attrs: Option<IMFAttributes> = None;
        MFCreateAttributes(&mut attrs, 4)?;
        let attrs = attrs.context("no attributes")?;
        if let Some(cb) = callback {
            attrs.SetUnknown(&MF_SOURCE_READER_ASYNC_CALLBACK, cb)?;
        }
        attrs.SetUINT32(&MF_LOW_LATENCY, 1)?;
        attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1)?;
        attrs.SetUINT32(&MF_READWRITE_DISABLE_CONVERTERS, 0)?;
        let reader = MFCreateSourceReaderFromMediaSource(&source, &attrs)
            .context("MFCreateSourceReaderFromMediaSource")?;
        Ok((reader, source))
    }
}

impl Mode {
    fn key(&self) -> ModeKey {
        (self.width, self.height, self.fps.round() as u32)
    }

    /// Best first: smooth (50 fps or more), then the most pixels (1440p60 over 1080p120),
    /// then frame rate, then a format we convert ourselves.
    fn rank(&self) -> (bool, u32, u32, u8) {
        let fps = self.fps.round() as u32;
        (fps >= 50, self.width * self.height, fps, subtype_rank(&self.subtype))
    }
}

/// The modes on offer for the menu: each size and rate once, best first.
fn offered(modes: &[Mode]) -> Vec<ModeKey> {
    let mut sorted: Vec<&Mode> = modes.iter().collect();
    sorted.sort_by_key(|m| std::cmp::Reverse(m.rank()));
    let mut keys: Vec<ModeKey> = Vec::new();
    for m in sorted {
        if !keys.contains(&m.key()) {
            keys.push(m.key());
        }
    }
    keys
}

/// The wanted mode if it's offered (in the format we handle best), else the best one.
fn choose_mode(modes: &[Mode], wanted: Option<ModeKey>) -> Option<Mode> {
    if let Some(w) = wanted
        && let Some(m) = modes.iter().filter(|m| m.key() == w).max_by_key(|m| subtype_rank(&m.subtype))
    {
        return Some(m.clone());
    }
    modes.iter().max_by_key(|m| m.rank()).cloned()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    Nv12,
    Yuy2,
    Rgb32,
}

struct Format {
    layout: Layout,
    width: usize,
    height: usize,
    matrix: Matrix,
    hdr_matrix: Matrix,
    primaries_2020: bool,
    /// The range the device claims (the HD60 X claims limited for the PS5's full-range SDR).
    reported_full: bool,
    /// The range decoded with.
    full_range: bool,
    sdr: Sdr,
    hdr: Hdr,
    /// The same decode for the GPU path, SDR and HDR.
    gpu_sdr: Decoding,
    gpu_hdr: Decoding,
}

impl Format {
    fn new(layout: Layout, (w, h): (u32, u32), matrix: Matrix, hdr_matrix: Matrix, primaries_2020: bool, reported_full: bool) -> Format {
        let gpu_sdr = Decoding {
            layout,
            hdr: false,
            matrix,
            gamut_2020: false,
            full_range: reported_full,
            white_nits: HDR_WHITE_NITS as f32,
        };
        Format {
            layout,
            width: w as usize,
            height: h as usize,
            matrix,
            hdr_matrix,
            primaries_2020,
            reported_full,
            full_range: reported_full,
            sdr: Sdr::new(matrix, reported_full),
            hdr: Hdr::new(hdr_matrix, primaries_2020, reported_full, HDR_WHITE_NITS),
            gpu_sdr,
            gpu_hdr: Decoding { hdr: true, matrix: hdr_matrix, gamut_2020: primaries_2020, ..gpu_sdr },
        }
    }

    /// Decodes as full range (0-255) or limited (16-235) from now on.
    fn set_full_range(&mut self, full: bool) {
        if full != self.full_range {
            let reported = self.reported_full;
            *self = Format::new(
                self.layout,
                (self.width as u32, self.height as u32),
                self.matrix,
                self.hdr_matrix,
                self.primaries_2020,
                full,
            );
            self.reported_full = reported;
        }
    }
}

/// What a picture's luma says about its range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RangeVote {
    Full,
    Limited,
    Unsure,
}

/// Looks at a sample of the luma (`step` bytes apart in a row: 1 for NV12, 2 for YUY2).
/// Limited-range video keeps black at 16 and white at 235 and barely strays past them;
/// full-range video labelled limited (the HD60 X does that with the PS5's SDR output) has its
/// darks well below 16 and its whites well above 235. So: more than a sliver under 8 or over
/// 245 is full range; nothing under 15 with a good share sitting at 16 is limited; anything
/// else (mid-tones only) says nothing. Returns the vote and the shares outside 8-245 and at
/// 15-17.
pub fn range_vote(y: &[u8], pitch: usize, step: usize, w: usize, h: usize) -> (RangeVote, f64, f64) {
    let (mut n, mut deep, mut low, mut black) = (0u64, 0u64, 0u64, 0u64);
    for r in (0..h).step_by(7) {
        let row = &y[r * pitch..];
        for x in (0..w).step_by(13) {
            let v = row[x * step];
            n += 1;
            deep += !(8..=245).contains(&v) as u64;
            low += (v < 15) as u64;
            black += (15..=17).contains(&v) as u64;
        }
    }
    let share = |c: u64| c as f64 / n.max(1) as f64;
    let vote = if share(deep) > 0.002 {
        RangeVote::Full
    } else if low == 0 && share(black) > 0.01 {
        RangeVote::Limited
    } else {
        RangeVote::Unsure
    };
    (vote, share(deep), share(black))
}

/// The range decided from the votes of recent pictures: full after 2 in a row, limited after
/// 6 (crushed blacks are worse than lifted ones, so it leaves limited quicker). Starts over
/// when the signal pauses, as it does when the console switches between HDR (limited) and
/// SDR (full on the PS5).
struct RangeGuess {
    full: Option<bool>,
    run: (RangeVote, u32),
}

impl RangeGuess {
    fn new() -> Self {
        RangeGuess { full: None, run: (RangeVote::Unsure, 0) }
    }

    fn feed(&mut self, v: RangeVote) -> Option<bool> {
        if v != RangeVote::Unsure {
            self.run = if self.run.0 == v { (v, self.run.1 + 1) } else { (v, 1) };
            if self.run.1 >= if v == RangeVote::Full { 2 } else { 6 } {
                self.full = Some(v == RangeVote::Full);
            }
        }
        self.full
    }
}

/// How to treat the signal's range (the menu's choice).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColourRange {
    /// Detected from the picture.
    #[default]
    Auto,
    Limited,
    Full,
}

fn current_format(reader: &IMFSourceReader) -> Result<Format> {
    unsafe {
        let t = reader.GetCurrentMediaType(VIDEO_STREAM)?;
        let (w, h, _, subtype) = describe(&t).context("output type has no frame size")?;
        let layout = if subtype == MFVideoFormat_NV12 {
            Layout::Nv12
        } else if subtype == MFVideoFormat_YUY2 {
            Layout::Yuy2
        } else {
            Layout::Rgb32
        };
        let reported_matrix = match t.GetUINT32(&MF_MT_YUV_MATRIX) {
            Ok(m) if m == MFVideoTransferMatrix_BT601.0 as u32 => Some(Matrix::Bt601),
            Ok(m) if m == MFVideoTransferMatrix_BT709.0 as u32 => Some(Matrix::Bt709),
            Ok(m) if m == MFVideoTransferMatrix_BT2020_10.0 as u32
                || m == MFVideoTransferMatrix_BT2020_12.0 as u32 =>
            {
                Some(Matrix::Bt2020)
            }
            _ => None,
        };
        let matrix = reported_matrix.unwrap_or(if h >= 720 { Matrix::Bt709 } else { Matrix::Bt601 });
        let full_range = matches!(
            t.GetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE),
            Ok(r) if r == MFNominalRange_0_255.0 as u32
        );
        // HDR10 is BT.2020, but trust what the source reports: the HD60 X converts the gamut
        // to BT.709 itself (verified against the official sRGB artwork it was showing).
        let primaries_2020 = match t.GetUINT32(&MF_MT_VIDEO_PRIMARIES) {
            Ok(p) => p == MFVideoPrimaries_BT2020.0 as u32,
            Err(_) => true,
        };
        let hdr_matrix = reported_matrix.unwrap_or(Matrix::Bt2020);
        Ok(Format::new(layout, (w, h), matrix, hdr_matrix, primaries_2020, full_range))
    }
}

/// The range vote of a sample's picture (None for RGB, which has no luma range to guess).
fn sample_range(sample: &IMFSample, fmt: &Format) -> Result<Option<(RangeVote, f64, f64)>> {
    if fmt.layout == Layout::Rgb32 {
        return Ok(None);
    }
    let buffer = unsafe {
        if sample.GetBufferCount()? == 1 { sample.GetBufferByIndex(0)? } else { sample.ConvertToContiguousBuffer()? }
    };
    let locked = Locked::new(buffer)?;
    let (w, h) = (fmt.width, fmt.height);
    let step = if fmt.layout == Layout::Yuy2 { 2 } else { 1 };
    let pitch = if locked.pitch > 0 { locked.pitch as usize } else { w * step };
    let off = locked.scan0 as usize - locked.base as usize;
    let bytes = locked.bytes();
    if bytes.len() < off + pitch * h {
        bail!("frame buffer too small");
    }
    Ok(Some(range_vote(&bytes[off..], pitch, step, w, h)))
}

/// HDR brightness that maps to SDR white. Measured on this PS5 + HD60 X: game tiles on the
/// home screen, matched cell by cell against their official sRGB art, fit sRGB at ~370 nits
/// with no bias across darks, mids and brights. (BT.2408's 203 came out ~20 codes too bright;
/// the PS5 draws its white UI text brighter still, ~500 nits, which just clips to white.)
pub const HDR_WHITE_NITS: f64 = 370.0;

struct Opened {
    name: String,
    // Dropped in this order: the reader before its source is shut down.
    reader: IMFSourceReader,
    _shutdown: Shutdown,
    /// Finished reads, from the reader's callback.
    samples: std::sync::mpsc::Receiver<Sampled>,
    mode: Mode,
    /// Everything the device offers.
    modes: Vec<Mode>,
    fmt: Format,
}

fn open_configured(opts: &VideoOptions, wanted: Option<ModeKey>) -> Result<Opened> {
    open_mode(opts.device.as_deref(), |modes| choose_mode(modes, wanted))
}

fn open_mode(name: Option<&str>, choose: impl FnOnce(&[Mode]) -> Option<Mode>) -> Result<Opened> {
    let devices = enumerate_devices()?;
    let Some(device) = pick_device(devices, name) else {
        bail!("No capture device found");
    };
    let (tx, samples) = std::sync::mpsc::channel();
    let callback: IMFSourceReaderCallback = ReadCallback { tx: Mutex::new(tx) }.into();
    let (reader, source) = open_reader(&device, Some(&callback))?;
    let shutdown = Shutdown(source);
    let modes = native_modes(&reader);
    let Some(mode) = choose(&modes) else {
        bail!("{}: no matching video mode", device.name);
    };
    unsafe {
        let native = reader.GetNativeMediaType(VIDEO_STREAM, mode.index)?;
        reader.SetCurrentMediaType(VIDEO_STREAM, None, &native)?;
        if subtype_rank(&mode.subtype) == 1 {
            let out = MFCreateMediaType()?;
            out.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            out.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
            reader
                .SetCurrentMediaType(VIDEO_STREAM, None, &out)
                .context("no converter to RGB32")?;
        }
        // Only the video stream: an audio stream on the same source would stall the reader.
        let _ = reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false);
        reader.SetStreamSelection(VIDEO_STREAM, true)?;
    }
    let fmt = current_format(&reader)?;
    Ok(Opened { name: device.name, reader, _shutdown: shutdown, samples, mode, modes, fmt })
}

/// What a read brought.
enum Read {
    Frame(IMFSample),
    /// A stream tick without a frame.
    Nothing,
    /// The device switched to another mode by itself (its input changed): reopen to pick
    /// from what it offers now.
    ModeChanged(String),
    /// A mode was picked while waiting.
    Reopen,
    /// No frame came within the limit.
    Stalled,
}

/// Reads the next frame, waiting up to `limit` for it, and giving up early if `control` asks
/// for a reopen.
fn read(o: &mut Opened, control: Option<&Control>, limit: Duration) -> Result<Read> {
    unsafe { o.reader.ReadSample(VIDEO_STREAM, 0, None, None, None, None)? };
    let asked = Instant::now();
    let done = loop {
        match o.samples.recv_timeout(POLL.min(limit)) {
            Ok(s) => break s,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!("{}: the reader went away", o.name),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if control.is_some_and(|c| c.reopen.load(Ordering::Acquire)) {
                    return Ok(Read::Reopen);
                }
                if asked.elapsed() >= limit {
                    return Ok(Read::Stalled);
                }
            }
        }
    };
    done.status.ok().with_context(|| format!("{}: read failed", o.name))?;
    let (flags, sample) = (done.flags, done.sample);
    if flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0 {
        bail!("{}: stream error", o.name);
    }
    if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
        bail!("{}: stream ended", o.name);
    }
    if flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED.0 as u32 != 0 {
        let now = unsafe { o.reader.GetNativeMediaType(VIDEO_STREAM, MF_SOURCE_READER_CURRENT_TYPE_INDEX.0 as u32) }
            .ok()
            .and_then(|t| describe(&t));
        let m = &o.mode;
        if let Some((w, h, fps, subtype)) = now
            && (w, h, fps.round() as u32, subtype) != (m.width, m.height, m.fps.round() as u32, m.subtype)
        {
            return Ok(Read::ModeChanged(format!("{} {:.0} fps {}", mode_label(w, h, fps.round() as u32), fps, subtype_name(&subtype))));
        }
    }
    if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
        o.fmt = current_format(&o.reader)?;
        crate::log::line(&format!("video: frames are now {}x{} {:?}", o.fmt.width, o.fmt.height, o.fmt.layout));
    }
    Ok(sample.map_or(Read::Nothing, Read::Frame))
}

/// Shows what's open: the status line, the mode label and the frame size.
fn publish(shared: &VideoShared, o: &Opened) {
    let (w, h) = (o.fmt.width as u32, o.fmt.height as u32);
    let fps = o.mode.fps.round() as u32;
    *shared.status.lock().unwrap() = format!("{}  {w}x{h} @ {fps} fps {}", o.name, o.mode.subtype_name());
    *shared.mode.lock().unwrap() = mode_label(w, h, fps);
    shared.control.size.store(((w as u64) << 32) | h as u64, Ordering::Relaxed);
    set_aspect(shared, &o.fmt);
}

/// Opens the device and passes its frames on until it should be reopened.
fn run(
    shared: &VideoShared,
    presenter: &Presenter,
    opts: &VideoOptions,
    stall: Duration,
    range: &mut RangeGuess,
) -> Result<Ended> {
    let c = &shared.control;
    c.reopen.store(false, Ordering::Release);
    let wanted = *c.wanted.lock().unwrap();
    let mut o = open_configured(opts, wanted)?;
    let offer = offered(&o.modes);
    let list: Vec<String> = offer.iter().map(|&(w, h, f)| mode_label(w, h, f)).collect();
    let picked = o.mode.key();
    crate::log::line(&format!(
        "video: opened {} at {} {}{} (offers {})",
        o.name,
        mode_label(picked.0, picked.1, picked.2),
        o.mode.subtype_name(),
        match wanted {
            None => " (auto)".to_string(),
            Some(w) if w == picked => " (picked)".to_string(),
            Some((w, h, f)) => format!(" (picked {} isn't offered now)", mode_label(w, h, f)),
        },
        list.join(", ")
    ));
    *c.modes.lock().unwrap() = offer;
    publish(shared, &o);

    let mut had_frames = false;
    let mut frames = 0u64;
    let mut last_at: Option<Instant> = None;
    c.detected.store(range.full.map_or(0, |f| if f { 2 } else { 1 }), Ordering::Relaxed);
    loop {
        if c.reopen.load(Ordering::Acquire) {
            return Ok(Ended::Reopen);
        }
        let sample = match read(&mut o, Some(c), stall)? {
            Read::Frame(s) => s,
            Read::Nothing => continue,
            Read::ModeChanged(now) => {
                crate::log::line(&format!("video: the device switched to {now}, reopening"));
                return Ok(Ended::Switched);
            }
            Read::Reopen => return Ok(Ended::Reopen),
            Read::Stalled => {
                crate::log::line(&format!("video: no frames for {:.1} s, reopening the device", stall.as_secs_f64()));
                return Ok(Ended::Stalled { had_frames });
            }
        };
        if c.reopen.load(Ordering::Acquire) {
            return Ok(Ended::Reopen);
        }
        if (o.fmt.width as u32, o.fmt.height as u32) != c.size() {
            publish(shared, &o);
        }
        // A pause in the frames is the input re-syncing (HDR switched on or off, say): its
        // range may have changed, so start the guess over.
        let now = Instant::now();
        if last_at.is_some_and(|t| now - t > Duration::from_millis(250)) && range.full.is_some() {
            crate::log::line("video: the signal paused; checking its range again");
            *range = RangeGuess::new();
            c.detected.store(0, Ordering::Relaxed);
        }
        last_at = Some(now);
        // The range: look at the picture every 8th frame (and the first).
        if frames % 8 == 0
            && let Some((vote, _, _)) = sample_range(&sample, &o.fmt)?
        {
            let before = range.full;
            let guess = range.feed(vote);
            c.detected.store(guess.map_or(0, |f| if f { 2 } else { 1 }), Ordering::Relaxed);
            if guess != before && let Some(full) = guess {
                crate::log::line(&format!("video: the picture looks {} range", if full { "full" } else { "limited" }));
            }
        }
        if frames % PICTURE_EVERY == 0 && has_picture(&sample, &o.fmt)? {
            *shared.last_picture.lock().unwrap() = Some(Instant::now());
        }
        frames += 1;
        had_frames = true;
        let full = match *c.range.lock().unwrap() {
            ColourRange::Auto => range.full.unwrap_or(o.fmt.reported_full),
            ColourRange::Limited => false,
            ColourRange::Full => true,
        };
        if full != o.fmt.full_range {
            crate::log::line(&format!("video: decoding as {} range", if full { "full (0-255)" } else { "limited (16-235)" }));
            o.fmt.set_full_range(full);
        }
        c.full_now.store(full, Ordering::Relaxed);
        let t0 = Instant::now();
        let hdr = shared.hdr.load(Ordering::Relaxed);
        if shared.gpu.ready.load(Ordering::Acquire) {
            let decode = if hdr { o.fmt.gpu_hdr } else { o.fmt.gpu_sdr };
            let data = raw_frame(&sample, &o.fmt, presenter.buffer())?;
            let (w, h) = (o.fmt.width as u32, o.fmt.height as u32);
            let spent = t0.elapsed();
            counted(shared, spent);
            let (seq, copy_us, fps) = (next_seq(), spent.as_micros() as u32, o.mode.fps as f32);
            presenter.push(RawFrame { data, w, h, decode, at: t0, seq, copy_us, fps });
        } else {
            let display_w = shared.display_w.load(Ordering::Relaxed) as usize;
            let image = convert_sample(&sample, &o.fmt, hdr, display_w)?;
            counted(shared, t0.elapsed());
            show(shared, Arc::new(image));
        }
    }
}

/// Card frames are numbered across the whole run (test sessions follow each one through).
fn next_seq() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

/// Counts a captured frame and the time spent preparing it.
fn counted(shared: &VideoShared, spent: Duration) {
    *shared.last_frame.lock().unwrap() = Some(Instant::now());
    shared.produced.fetch_add(1, Ordering::Relaxed);
    shared.convert_ns.fetch_add(spent.as_nanos() as u64, Ordering::Relaxed);
}

/// Copies a sample's pixels out packed for the GPU: NV12 as Y rows then UV rows, YUY2 rows,
/// or BGRA rows top-down.
/// `out` is an empty buffer to fill (one used before has room already).
fn raw_frame(sample: &IMFSample, fmt: &Format, mut out: Vec<u8>) -> Result<Vec<u8>> {
    let buffer = unsafe {
        if sample.GetBufferCount()? == 1 {
            sample.GetBufferByIndex(0)?
        } else {
            sample.ConvertToContiguousBuffer()?
        }
    };
    let locked = Locked::new(buffer)?;
    let (w, h) = (fmt.width, fmt.height);
    let bytes = locked.bytes();
    match fmt.layout {
        Layout::Nv12 => {
            let pitch = if locked.pitch > 0 { locked.pitch as usize } else { w };
            let y_off = locked.scan0 as usize - locked.base as usize;
            let avail = bytes.len() - y_off;
            // Some drivers pad the Y plane's height (1080 -> 1088); the UV plane follows the padding.
            let plane_rows = (avail / pitch * 2 / 3).max(h);
            let uv_off = y_off + plane_rows * pitch;
            if bytes.len() < uv_off + (h / 2) * pitch {
                bail!("NV12 buffer too small");
            }
            out.reserve_exact(w * h * 3 / 2);
            for r in 0..h {
                out.extend_from_slice(&bytes[y_off + r * pitch..][..w]);
            }
            for r in 0..h / 2 {
                out.extend_from_slice(&bytes[uv_off + r * pitch..][..w]);
            }
        }
        Layout::Yuy2 => {
            let pitch = if locked.pitch > 0 { locked.pitch as usize } else { w * 2 };
            let off = locked.scan0 as usize - locked.base as usize;
            if bytes.len() < off + pitch * h {
                bail!("YUY2 buffer too small");
            }
            out.reserve_exact(w * 2 * h);
            for r in 0..h {
                out.extend_from_slice(&bytes[off + r * pitch..][..w * 2]);
            }
        }
        Layout::Rgb32 => {
            let (pitch, first) = if locked.pitch != 0 {
                (locked.pitch, locked.scan0 as usize - locked.base as usize)
            } else {
                // Plain buffers of RGB32 from MF are bottom-up.
                let p = (w * 4) as isize;
                (-p, (h - 1) * w * 4)
            };
            out.reserve_exact(w * 4 * h);
            for r in 0..h as isize {
                let start = (first as isize + r * pitch) as usize;
                out.extend_from_slice(bytes.get(start..start + w * 4).context("RGB32 buffer too small")?);
            }
        }
    }
    Ok(out)
}

/// Hands a frame to the UI and wakes it (at most once until it has drawn).
fn show(shared: &VideoShared, image: Arc<RenderImage>) {
    let now = Instant::now();
    // The CPU path doesn't look for repeated pictures.
    shared.meter.card_frame(now, None);
    *shared.latest.lock().unwrap() = Some((image, now));
    if !shared.pending.swap(true, Ordering::AcqRel) {
        let _ = shared.waker.unbounded_send(());
    }
}

/// A synthetic NV12 source (2560x1440 unless `size` says) at `fps`: a bar sweeping across a gradient, moving
/// `content` times a second, paced on absolute deadlines. Goes through the same converter
/// and hand-off as real frames.
/// The test pattern's picture at card frame `n`: evenly paced, or with `hitch` every fifth
/// picture held a card frame longer and the next one a frame shorter, as a game that misses a
/// frame delivers them (at 30 fps on 60 Hz: held 2, 2, 2, 3, 1).
fn test_picture(n: u64, fps: u32, content: u32, hitch: bool) -> usize {
    let h = (fps / content.max(1)) as u64;
    if !hitch || h < 2 {
        return (n * content as u64 / fps as u64) as usize;
    }
    let (cycle, r) = (n / (5 * h), n % (5 * h));
    let starts = [0, h, 2 * h, 3 * h, 4 * h + 1];
    cycle as usize * 5 + starts.iter().rposition(|&s| r >= s).unwrap_or(0)
}

fn test_pattern(shared: &VideoShared, presenter: &Presenter, rates: (u32, u32, bool), size: Option<(u32, u32)>) {
    let (fps, content, hitch) = rates;
    #[allow(non_snake_case)] // constants until the size became an option
    let (W, H) = size.map_or((2560, 1440), |(w, h)| (w as usize & !1, h as usize & !1));
    *shared.status.lock().unwrap() = format!("Test pattern {W}x{H} @ {fps}");
    *shared.mode.lock().unwrap() = format!("test {}", mode_label(W as u32, H as u32, fps));
    shared.control.size.store(((W as u64) << 32) | H as u64, Ordering::Relaxed);
    *shared.control.modes.lock().unwrap() = vec![(W as u32, H as u32, fps)];
    shared.aspect.store((W as f32 / H as f32).to_bits(), Ordering::Relaxed);
    let sdr = Sdr::new(Matrix::Bt709, false);
    // Gradient background, built once.
    let mut base = vec![0u8; W * H * 3 / 2];
    for y in 0..H {
        for x in 0..W {
            base[y * W + x] = (16 + (x * 200 / W)) as u8;
        }
    }
    for i in 0..W * H / 2 {
        base[W * H + i] = if i % 2 == 0 { 150 } else { 110 };
    }
    // Static detail for checking the picture reaches the screen pixel for pixel: 1 px
    // black/white columns, then a 1 px checkerboard, both with neutral chroma.
    for y in 100..400 {
        for x in 100..900 {
            base[y * W + x] = if x % 2 == 0 { 16 } else { 235 };
        }
    }
    for y in 500..800 {
        for x in 100..900 {
            base[y * W + x] = if (x + y) % 2 == 0 { 16 } else { 235 };
        }
    }
    for y in (100 / 2)..(800 / 2) {
        for x in (100 / 2)..(900 / 2) {
            base[W * H + y * W + x * 2] = 128;
            base[W * H + y * W + x * 2 + 1] = 128;
        }
    }
    let period = Duration::from_secs_f64(1.0 / fps as f64);
    let start = Instant::now();
    let mut buf = base.clone();
    for n in 0u64.. {
        let deadline = start + period * n as u32;
        if let Some(wait) = deadline.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        buf.copy_from_slice(&base);
        *shared.last_picture.lock().unwrap() = Some(Instant::now());
        // A white bar moving one bar-width per picture, so dropped frames show as jumps.
        let bar = 64;
        let picture = test_picture(n, fps, content, hitch);
        let x0 = (picture * bar) % W;
        for y in 0..H {
            buf[y * W + x0..y * W + (x0 + bar).min(W)].fill(235);
        }
        let t0 = Instant::now();
        if shared.gpu.ready.load(Ordering::Acquire) {
            let decode = Decoding {
                layout: Layout::Nv12,
                hdr: false,
                matrix: Matrix::Bt709,
                gamut_2020: false,
                full_range: false,
                white_nits: HDR_WHITE_NITS as f32,
            };
            let mut data = presenter.buffer();
            data.extend_from_slice(&buf);
            let spent = t0.elapsed();
            counted(shared, spent);
            let (seq, copy_us) = (next_seq(), spent.as_micros() as u32);
            presenter.push(RawFrame { data, w: W as u32, h: H as u32, decode, at: t0, seq, copy_us, fps: fps as f32 });
            continue;
        }
        let mut out = vec![0u8; W * H * 4];
        convert::nv12_to_bgra(&sdr, &buf[..W * H], &buf[W * H..], W, W, H, &mut out);
        let display_w = shared.display_w.load(Ordering::Relaxed) as usize;
        let Some(img) = to_render_image(out, W, H, display_w) else { return };
        counted(shared, t0.elapsed());
        show(shared, Arc::new(img));
    }
}

fn set_aspect(shared: &VideoShared, fmt: &Format) {
    let aspect = fmt.width as f32 / fmt.height.max(1) as f32;
    shared.aspect.store(aspect.to_bits(), Ordering::Relaxed);
}

/// Shuts the media source down when the reader goes away so the device is released for a retry.
struct Shutdown(IMFMediaSource);
impl Drop for Shutdown {
    fn drop(&mut self) {
        unsafe {
            let _ = self.0.Shutdown();
        }
    }
}

/// A locked view of a sample's pixels; unlocks on drop.
struct Locked {
    buffer: IMFMediaBuffer,
    two_d: Option<IMF2DBuffer>,
    /// Pointer to the first row.
    scan0: *const u8,
    pitch: isize,
    /// Bytes readable from `scan0` in memory order (for negative pitch, from the buffer start).
    base: *const u8,
    len: usize,
}

impl Locked {
    fn new(buffer: IMFMediaBuffer) -> Result<Self> {
        unsafe {
            if let Ok(b2) = buffer.cast::<IMF2DBuffer2>() {
                let mut scan0 = std::ptr::null_mut();
                let mut pitch = 0i32;
                let mut start = std::ptr::null_mut();
                let mut len = 0u32;
                b2.Lock2DSize(
                    MF2DBuffer_LockFlags_Read,
                    &mut scan0,
                    &mut pitch,
                    &mut start,
                    &mut len,
                )?;
                return Ok(Locked {
                    two_d: Some(b2.cast()?),
                    buffer,
                    scan0,
                    pitch: pitch as isize,
                    base: start,
                    len: len as usize,
                });
            }
            let mut ptr = std::ptr::null_mut();
            let mut len = 0u32;
            buffer.Lock(&mut ptr, None, Some(&mut len))?;
            Ok(Locked {
                buffer,
                two_d: None,
                scan0: ptr,
                pitch: 0,
                base: ptr,
                len: len as usize,
            })
        }
    }

    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        unsafe {
            match &self.two_d {
                Some(b) => {
                    let _ = b.Unlock2D();
                }
                None => {
                    let _ = self.buffer.Unlock();
                }
            }
        }
    }
}

fn convert_sample(sample: &IMFSample, fmt: &Format, hdr: bool, display_w: usize) -> Result<RenderImage> {
    let (w, h) = (fmt.width, fmt.height);
    let mut out = vec![0u8; w * h * 4];
    if hdr {
        convert_into(sample, fmt, &fmt.hdr, &mut out)?;
    } else {
        convert_into(sample, fmt, &fmt.sdr, &mut out)?;
    }
    to_render_image(out, w, h, display_w).context("frame size")
}

/// Wraps a BGRA frame for gpui, first halving it (2x2 average) while it's still at least
/// twice the size it will be drawn at. gpui samples images bilinearly without mipmaps, which
/// aliases below half size; this keeps the GPU's share of the scaling between 0.5x and 1x.
fn to_render_image(mut px: Vec<u8>, mut w: usize, mut h: usize, display_w: usize) -> Option<RenderImage> {
    while display_w > 0 && w / 2 >= display_w && h >= 4 {
        px = convert::halve_bgra(&px, w, h);
        w /= 2;
        h /= 2;
    }
    let img = image::RgbaImage::from_raw(w as u32, h as u32, px)?;
    Some(RenderImage::new(vec![image::Frame::new(img)]))
}

/// Converts a sample's pixels into BGRA `out` (width * height * 4 bytes).
fn convert_into<D: Decode>(sample: &IMFSample, fmt: &Format, d: &D, out: &mut [u8]) -> Result<()> {
    let buffer = unsafe {
        if sample.GetBufferCount()? == 1 {
            sample.GetBufferByIndex(0)?
        } else {
            sample.ConvertToContiguousBuffer()?
        }
    };
    let locked = Locked::new(buffer)?;
    let (w, h) = (fmt.width, fmt.height);
    let bytes = locked.bytes();
    match fmt.layout {
        Layout::Nv12 => {
            let pitch = if locked.pitch > 0 { locked.pitch as usize } else { w };
            let y_off = locked.scan0 as usize - locked.base as usize;
            let avail = bytes.len() - y_off;
            // Some drivers pad the Y plane's height (1080 -> 1088); the UV plane follows the padding.
            let plane_rows = (avail / pitch * 2 / 3).max(h);
            let uv_off = y_off + plane_rows * pitch;
            if bytes.len() < uv_off + (h / 2) * pitch {
                bail!("NV12 buffer too small");
            }
            convert::nv12_to_bgra(d, &bytes[y_off..], &bytes[uv_off..], pitch, w, h, out);
        }
        Layout::Yuy2 => {
            let pitch = if locked.pitch > 0 { locked.pitch as usize } else { w * 2 };
            let off = locked.scan0 as usize - locked.base as usize;
            if bytes.len() < off + pitch * h {
                bail!("YUY2 buffer too small");
            }
            convert::yuy2_to_bgra(d, &bytes[off..], pitch, w, h, out);
        }
        Layout::Rgb32 => {
            let (pitch, first) = if locked.pitch != 0 {
                (locked.pitch, locked.scan0 as usize - locked.base as usize)
            } else {
                // Plain buffers of RGB32 from MF are bottom-up.
                let p = (w * 4) as isize;
                (-p, (h - 1) * w * 4)
            };
            convert::bgrx_to_bgra(bytes, first, pitch, w, h, out);
        }
    }
    Ok(())
}

/// Decodes a packed NV12 dump from `--snapshot` with the HDR path (for `--decode-raw`), so
/// colour changes can be checked against the same frame without the capture card.
pub fn decode_raw(raw: &std::path::Path, w: usize, h: usize, how: &str, out: &std::path::Path) -> Result<()> {
    let data = std::fs::read(raw)?;
    if data.len() < w * h * 3 / 2 {
        bail!("{} is smaller than a {w}x{h} NV12 frame", raw.display());
    }
    let (vote, deep, black) = range_vote(&data, w, 1, w, h);
    println!("range: {vote:?} ({:.2}% under 8 or over 245, {:.2}% at 15-17)", deep * 100.0, black * 100.0);
    let gamut_2020 = how == "2020";
    let matrix = if gamut_2020 { Matrix::Bt2020 } else { Matrix::Bt709 };
    let white = std::env::var("CAPCARD_WHITE_NITS").ok().and_then(|v| v.parse().ok()).unwrap_or(HDR_WHITE_NITS);
    let mut px = vec![0u8; w * h * 4];
    let (y, uv) = (&data[..w * h], &data[w * h..]);
    match how {
        "sdr" => convert::nv12_to_bgra(&Sdr::new(matrix, false), y, uv, w, w, h, &mut px),
        "sdr-full" => convert::nv12_to_bgra(&Sdr::new(matrix, true), y, uv, w, w, h, &mut px),
        _ => convert::nv12_to_bgra(&Hdr::new(matrix, gamut_2020, false, white), y, uv, w, w, h, &mut px),
    }
    for p in px.chunks_exact_mut(4) {
        p.swap(0, 2);
    }
    image::save_buffer(out, &px, w as u32, h as u32, image::ExtendedColorType::Rgba8)?;
    Ok(())
}

/// Grabs one frame and writes it decoded both ways (for `--snapshot DIR`), with luma stats,
/// to check whether the source is SDR or HDR.
pub fn snapshot(opts: &VideoOptions, dir: &std::path::Path) -> Result<()> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET)?;
    }
    let mut o = open_configured(opts, opts.mode)?;
    // Let auto exposure / the first partial frames settle.
    let mut sample = None;
    let mut got = 0;
    while got < 30 {
        match read(&mut o, None, Duration::from_secs(5))? {
            Read::Frame(s) => {
                sample = Some(s);
                got += 1;
            }
            Read::Nothing | Read::Reopen => {}
            Read::ModeChanged(now) => bail!("the device switched to {now}"),
            Read::Stalled => bail!("no frames for 5 s"),
        }
    }
    let sample = sample.context("no frame")?;
    let (w, h) = (o.fmt.width, o.fmt.height);
    std::fs::create_dir_all(dir)?;
    for (name, hdr) in [("sdr", false), ("hdr", true)] {
        let mut out = vec![0u8; w * h * 4];
        if hdr {
            convert_into(&sample, &o.fmt, &o.fmt.hdr, &mut out)?;
        } else {
            convert_into(&sample, &o.fmt, &o.fmt.sdr, &mut out)?;
        }
        for px in out.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let path = dir.join(format!("{name}.png"));
        image::save_buffer(&path, &out, w as u32, h as u32, image::ExtendedColorType::Rgba8)?;
        println!("wrote {}", path.display());
    }
    // Luma histogram of the raw Y plane (NV12 only).
    if o.fmt.layout == Layout::Nv12 {
        let buffer = unsafe { sample.ConvertToContiguousBuffer()? };
        let locked = Locked::new(buffer)?;
        let pitch = if locked.pitch > 0 { locked.pitch as usize } else { w };
        let off = locked.scan0 as usize - locked.base as usize;
        let mut hist = [0u64; 256];
        for r in 0..h {
            for &y in &locked.bytes()[off + r * pitch..off + r * pitch + w] {
                hist[y as usize] += 1;
            }
        }
        // Tightly packed NV12, for comparing against reference decoders such as ffmpeg.
        let avail = locked.bytes().len() - off;
        let uv_off = off + (avail / pitch * 2 / 3).max(h) * pitch;
        let mut raw = Vec::with_capacity(w * h * 3 / 2);
        for r in 0..h {
            raw.extend_from_slice(&locked.bytes()[off + r * pitch..off + r * pitch + w]);
        }
        for r in 0..h / 2 {
            raw.extend_from_slice(&locked.bytes()[uv_off + r * pitch..uv_off + r * pitch + w]);
        }
        let raw_path = dir.join(format!("raw-{w}x{h}.nv12"));
        std::fs::write(&raw_path, raw)?;
        println!("wrote {}", raw_path.display());
        let total: u64 = hist.iter().sum();
        let pct = |p: f64| {
            let mut acc = 0;
            for (i, n) in hist.iter().enumerate() {
                acc += n;
                if acc as f64 >= p * total as f64 {
                    return i;
                }
            }
            255
        };
        println!(
            "Y min {} p1 {} p50 {} p99 {} p99.9 {} max {}",
            pct(0.0),
            pct(0.01),
            pct(0.5),
            pct(0.99),
            pct(0.999),
            pct(1.0)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_frames_are_not_a_picture() {
        let no_input = [17u8; 320];
        let mut logo = no_input;
        logo[160] = 200;
        assert!(!varies([&no_input[..]; 4].into_iter(), 1));
        assert!(!varies([&[16u8, 17, 18, 17][..]].into_iter(), 1), "within the card's noise");
        assert!(varies([&no_input[..], &logo[..]].into_iter(), 1));
    }

    #[test]
    fn the_tray_opens_when_a_picture_arrives() {
        // Started with Windows: a console that's already on opens it.
        let mut a = Arrival::new(false);
        assert!(!a.glance(true));
        assert!(a.glance(true));
        // A single odd frame doesn't.
        let mut a = Arrival::new(false);
        assert!(!a.glance(true) && !a.glance(false) && !a.glance(true));
        // Closed while playing: stays closed until the console has been off and comes back.
        let mut a = Arrival::new(true);
        assert!((0..10).all(|_| !a.glance(true)));
        assert!((1..Arrival::GONE).all(|_| !a.glance(false)), "not gone yet");
        assert!(!a.glance(true) && !a.glance(true), "a black screen in the game isn't off");
        assert!((0..Arrival::GONE).all(|_| !a.glance(false)));
        assert!(!a.glance(true));
        assert!(a.glance(true));
    }

    fn mode(width: u32, height: u32, fps: f64, subtype: GUID) -> Mode {
        Mode { index: 0, width, height, fps, subtype }
    }

    #[test]
    fn auto_prefers_resolution_at_a_smooth_rate() {
        let modes = [
            mode(1920, 1080, 120.0, MFVideoFormat_NV12),
            mode(3840, 2160, 30.0, MFVideoFormat_NV12),
            mode(2560, 1440, 59.94, MFVideoFormat_YUY2),
            mode(2560, 1440, 60.0, MFVideoFormat_NV12),
            mode(1280, 720, 60.0, MFVideoFormat_NV12),
        ];
        let best = choose_mode(&modes, None).unwrap();
        assert_eq!((best.key(), best.subtype), ((2560, 1440, 60), MFVideoFormat_NV12));
        assert_eq!(offered(&modes), vec![(2560, 1440, 60), (1920, 1080, 120), (1280, 720, 60), (3840, 2160, 30)]);
    }

    #[test]
    fn a_picked_mode_is_used_while_offered() {
        let modes = [mode(2560, 1440, 60.0, MFVideoFormat_NV12), mode(1920, 1080, 60.0, MFVideoFormat_NV12)];
        assert_eq!(choose_mode(&modes, Some((1920, 1080, 60))).unwrap().key(), (1920, 1080, 60));
        // Not offered (the card follows its input): the best one instead.
        assert_eq!(choose_mode(&modes, Some((3840, 2160, 30))).unwrap().key(), (2560, 1440, 60));
    }

    #[test]
    fn modes_group_into_one_row_per_resolution() {
        let modes = [(2560, 1440, 60), (1920, 1080, 120), (2560, 1440, 120), (3440, 1440, 60), (1920, 1080, 60), (2560, 1440, 30)];
        assert_eq!(
            by_resolution(&modes),
            [((3440, 1440), vec![60]), ((2560, 1440), vec![120, 60, 30]), ((1920, 1080), vec![120, 60])]
        );
        assert_eq!(resolution_label(2560, 1440), "1440p");
        assert_eq!(resolution_label(3440, 1440), "3440×1440");
    }

    #[test]
    fn labels() {
        assert_eq!(mode_label(2560, 1440, 60), "1440p60");
        assert_eq!(mode_label(2560, 1080, 60), "2560×1080 60");
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;

    /// A w x h luma plane: `dark` of the rows at `low`, the rest at 120.
    fn plane(w: usize, h: usize, dark: f64, low: u8) -> Vec<u8> {
        (0..h).flat_map(|r| std::iter::repeat_n(if (r as f64) < h as f64 * dark { low } else { 120 }, w)).collect()
    }

    #[test]
    fn votes() {
        let (w, h) = (640, 360);
        assert_eq!(range_vote(&plane(w, h, 0.3, 0), w, 1, w, h).0, RangeVote::Full);
        // Whites past 235 are full range too.
        assert_eq!(range_vote(&plane(w, h, 0.3, 255), w, 1, w, h).0, RangeVote::Full);
        assert_eq!(range_vote(&plane(w, h, 0.3, 16), w, 1, w, h).0, RangeVote::Limited);
        // Bright everywhere: no telling.
        assert_eq!(range_vote(&plane(w, h, 0.0, 0), w, 1, w, h).0, RangeVote::Unsure);
    }

    #[test]
    fn guess_needs_agreeing_votes() {
        let mut g = RangeGuess::new();
        assert_eq!(g.feed(RangeVote::Full), None);
        assert_eq!(g.feed(RangeVote::Unsure), None);
        assert_eq!(g.feed(RangeVote::Full), Some(true));
        // Back to limited takes six in a row.
        for _ in 0..5 {
            assert_eq!(g.feed(RangeVote::Limited), Some(true));
        }
        assert_eq!(g.feed(RangeVote::Limited), Some(false));
    }
}

#[cfg(test)]
mod test_pattern_tests {
    use super::test_picture;

    #[test]
    fn hitches_hold_one_picture_longer_and_the_next_shorter() {
        let holds = |hitch| {
            let p: Vec<usize> = (0..20).map(|n| test_picture(n, 60, 30, hitch)).collect();
            let mut runs = vec![1];
            for w in p.windows(2) {
                if w[0] == w[1] { *runs.last_mut().unwrap() += 1 } else { runs.push(1) }
            }
            runs
        };
        assert_eq!(holds(false), [2; 10]);
        assert_eq!(holds(true), [2, 2, 2, 3, 1, 2, 2, 2, 3, 1]);
    }
}
