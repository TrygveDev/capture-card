//! The presenter thread: takes raw captured frames, decodes them on the GPU and shows them in
//! our own layer under gpui's, with optional frame generation ("smooth" in the controls).
//!
//! With smoothing off, every frame is uploaded, decoded and presented at once.
//!
//! With smoothing on: the card always delivers 60 frames a second, so a 30 fps game arrives as
//! every picture twice. Repeats are dropped, the time each picture was on screen is tracked,
//! and that time is filled with generated frames at the target rate, presented on whole
//! refreshes of the display so their spacing is even. A generated frame needs the next real
//! picture first, so the picture runs one step behind: about +17 ms for 30 -> 60, +25 ms for
//! 30 -> 120, +8 ms for 60 -> 120 (plus up to one refresh to line up with the display).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rayon::prelude::*;
use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Dwm::{DWM_TIMING_INFO, DwmGetCompositionTimingInfo};
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

use crate::gpu::{Decoding, Frames, Gpu, Swap};
use crate::telemetry::{self, Ev, FrameKind};
use crate::upscale::Upscaler;
use crate::video::VideoShared;

/// Most frames made per picture (8 covers 30 fps -> 240, 15 fps -> 120).
const MAX_STEPS: u32 = 8;
/// Pictures held longer than this many card frames are a pause, not the game's frame rate.
const MAX_CADENCE: u32 = 4;
/// No frame for this long: show black (the UI says "No signal").
const SIGNAL_TIMEOUT: Duration = Duration::from_millis(1000);
/// Frames are presented this far into a refresh (a share of it) after a vblank, measured on
/// the real vblanks (`Vsync`). Test sessions at 240 Hz: presents 1-2.75 ms after a vblank
/// reach the screen at the next one 97-98% of the time; from 3 ms on DWM has already composed
/// it and they slip a refresh (21% at 3.25 ms, 50% at 3.5). Early in the refresh leaves the
/// most room for the scheduler's lateness and the draw.
const PRESENT_PHASE: f64 = 0.3;
/// A new picture waits while a present is due within its GPU work plus this (see `run`).
const DEFER_MARGIN: Duration = Duration::from_millis(2);
/// The delay ratchet. Smoothing shows a picture after its in-between frames, so the last pair's
/// picture is on screen shortly before the next one arrives. If it's still waiting when the
/// next arrives (a late picture, an early one, a slow pair), the newcomer waits for it, starts
/// later, and its picture shows later still: every pair after it inherits the delay, and
/// nothing ever took it back (36.5 ms from arrival to screen in a 4K 30 -> 120 session where
/// 31.5 is possible). So a picture that had to wait this long has its pair's last interval
/// shortened by a refresh, which brings the schedule back; at most once per `SHORTEN_EVERY`, so
/// a GPU too slow for the shorter delay doesn't get uneven pacing out of it. Test pattern 30 on
/// 60 -> 120 at 240 Hz: 38.3 ms -> 33.6 ms to the screen, one interval in ~500 a refresh short.
const WAITED: Duration = Duration::from_micros(1500);
const SHORTEN_EVERY: Duration = Duration::from_secs(3);
/// How long a pair's GPU work (motion and its generated frames) is assumed to take before
/// it has been measured.
const GPU_GUESS: Duration = Duration::from_millis(8);

/// A captured frame, packed: NV12 (Y rows then UV rows), YUY2, or top-down BGRA.
pub struct RawFrame {
    pub data: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub decode: Decoding,
    pub at: Instant,
    /// The card frame's number (test sessions match it to `capture.csv`).
    pub seq: u64,
    /// Time spent copying it out of the capture buffer.
    pub copy_us: u32,
    /// The capture mode's frame rate.
    pub fps: f32,
}

/// A card frame without its pixels, kept with them until the presenter is done (test
/// sessions record it then).
#[derive(Clone, Copy)]
struct RawInfo {
    seq: u64,
    at: Instant,
    copy_us: u32,
    w: u32,
    h: u32,
    layout: crate::video::Layout,
    fps: f32,
}

impl RawFrame {
    fn info(&self) -> RawInfo {
        RawInfo {
            seq: self.seq,
            at: self.at,
            copy_us: self.copy_us,
            w: self.w,
            h: self.h,
            layout: self.decode.layout,
            fps: self.fps,
        }
    }
}

/// State shared with the capture thread and the UI.
pub struct Shared {
    /// Smoothing target in frames per second (0 = off).
    pub target: AtomicU32,
    /// Detected picture rate of the source while smoothing, as f32 bits (0 = unknown).
    pub content_fps: AtomicU32,
    /// The GPU layer is up and drawing the video (the UI then draws only the controls).
    pub ready: AtomicBool,
    /// The window to draw under, once it exists.
    pub hwnd: OnceLock<isize>,
    /// Print rate and pacing every few seconds.
    pub stats: AtomicBool,
    /// Whole-run summary after warm-up, for `--exit-after`.
    pub summary: Mutex<String>,
    /// Upscale with RTX Video Super Resolution when it's running.
    pub rtx: AtomicBool,
    /// How the last frame was scaled (a `Scaling`).
    pub scaling: AtomicU8,
    /// Measure motion with the optical flow hardware where available (an experiment, see
    /// `gpu::flow_method`), and whether it's in use: 0 not smoothing yet, 1 shader, 2 hardware.
    pub hw_flow: AtomicBool,
    pub flow_in_use: AtomicU8,
    /// The display's refresh rate in whole Hz (0 = not known yet).
    pub refresh_hz: AtomicU32,
    /// The window is hidden (Passthru waits in the tray): frames are dropped as they come,
    /// with no GPU work at all.
    pub paused: AtomicBool,
}

/// How the video is being scaled to the window.
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum Scaling {
    /// Shown 1:1 or shrunk.
    NotUp = 0,
    /// Enlarged with the Lanczos shader.
    Lanczos = 1,
    /// Enlarged with RTX Video Super Resolution.
    Rtx = 2,
}

impl Shared {
    pub fn new(target: u32, stats: bool, rtx: bool, hw_flow: bool) -> Self {
        Shared {
            target: AtomicU32::new(target),
            content_fps: AtomicU32::new(0),
            ready: AtomicBool::new(false),
            hwnd: OnceLock::new(),
            stats: AtomicBool::new(stats),
            summary: Mutex::new(String::new()),
            rtx: AtomicBool::new(rtx),
            scaling: AtomicU8::new(Scaling::NotUp as u8),
            hw_flow: AtomicBool::new(hw_flow),
            flow_in_use: AtomicU8::new(0),
            refresh_hz: AtomicU32::new(0),
            paused: AtomicBool::new(false),
        }
    }

    pub fn scaling(&self) -> Scaling {
        match self.scaling.load(Ordering::Relaxed) {
            1 => Scaling::Lanczos,
            2 => Scaling::Rtx,
            _ => Scaling::NotUp,
        }
    }

    pub fn content_fps(&self) -> f32 {
        f32::from_bits(self.content_fps.load(Ordering::Relaxed))
    }
}

pub struct Presenter {
    tx: Sender<RawFrame>,
    spare: Spare,
}

/// Frame buffers the presenter is done with, for the capture side to fill again. A 4K frame
/// is 12 MB: a fresh allocation that size comes straight from the OS, zeroed page by page,
/// which costs about a millisecond every frame before the copy even starts.
#[derive(Clone, Default)]
pub struct Spare(Arc<Mutex<Vec<Vec<u8>>>>);

impl Spare {
    /// Enough for the frame in flight, the one kept for comparison and one to swap, and for
    /// the frames queued for a test session's input video. Only as many as were ever in use
    /// at once are kept, so without recording it stays at three or four.
    const KEEP: usize = 12;

    pub fn take(&self) -> Vec<u8> {
        self.0.lock().unwrap().pop().unwrap_or_default()
    }

    pub fn give(&self, buf: Vec<u8>) {
        let mut spare = self.0.lock().unwrap();
        if spare.len() < Self::KEEP {
            spare.push(buf);
        }
    }
}

/// The shared state, for test sessions' system samples.
static SHARED: OnceLock<Arc<VideoShared>> = OnceLock::new();

/// The display's refresh rate, once the presenter knows it.
pub fn display_hz() -> Option<u32> {
    let hz = SHARED.get()?.gpu.refresh_hz.load(Ordering::Relaxed);
    (hz > 0).then_some(hz)
}

/// Display rate, smoothing target and detected game rate, for test sessions.
pub fn shared_numbers() -> (u32, u32, f32) {
    SHARED.get().map_or((0, 0, 0.0), |s| {
        (s.gpu.refresh_hz.load(Ordering::Relaxed), s.gpu.target.load(Ordering::Relaxed), s.gpu.content_fps())
    })
}

impl Presenter {
    pub fn spawn(shared: Arc<VideoShared>) -> Presenter {
        let _ = SHARED.set(shared.clone());
        let (tx, rx) = mpsc::channel();
        let spare = Spare::default();
        let theirs = spare.clone();
        std::thread::Builder::new()
            .name("present".into())
            .spawn(move || {
                crate::raise_thread_priority("Playback");
                if let Err(e) = run(&shared, rx, theirs) {
                    crate::log::line(&format!("gpu: {e:#}; falling back to the CPU path"));
                }
                shared.gpu.ready.store(false, Ordering::Release);
            })
            .expect("spawn present thread");
        Presenter { tx, spare }
    }

    pub fn push(&self, frame: RawFrame) {
        let _ = self.tx.send(frame);
    }

    /// An empty buffer to copy the next frame into (with room for one, after the first few).
    pub fn buffer(&self) -> Vec<u8> {
        let mut b = self.spare.take();
        b.clear();
        b
    }
}

/// True if two raw frames show the same picture: identical, or within 2 code values
/// everywhere (a little capture noise). Any real change, even a blinking cursor, differs.
pub fn same_picture(a: &[u8], b: &[u8]) -> bool {
    const CHUNK: usize = 1 << 16;
    a.len() == b.len()
        && a.par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .all(|(x, y)| x == y || x.iter().zip(y).all(|(p, q)| p.abs_diff(*q) <= 2))
}

/// Card frame period and how long recent pictures were held.
struct Cadence {
    /// Card frame period in seconds, smoothed.
    period: f64,
    /// Card frames each recent picture lasted.
    history: VecDeque<u32>,
}

impl Cadence {
    fn new() -> Self {
        Cadence { period: 1.0 / 60.0, history: VecDeque::new() }
    }

    /// A card frame arrived `dt` seconds after the previous one; false if that's a gap.
    fn frame(&mut self, dt: f64) -> bool {
        if dt > self.period * 3.0 {
            return false;
        }
        self.period += (dt.clamp(0.004, 0.05) - self.period) * 0.05;
        true
    }

    fn usual(&self) -> Option<u32> {
        let mut s: Vec<u32> = self.history.iter().copied().collect();
        s.sort_unstable();
        s.get(s.len() / 2).copied()
    }

    /// A new picture after `n` card frames: how many card frames the previous one was really
    /// held. After a pause (a still scene), the move to this picture took one normal picture's
    /// time, not the whole pause.
    fn picture(&mut self, n: u32) -> u32 {
        let usual = self.usual().unwrap_or(n.min(MAX_CADENCE)).max(1);
        if n <= MAX_CADENCE {
            self.history.push_back(n);
            if self.history.len() > 15 {
                self.history.pop_front();
            }
        }
        if n <= MAX_CADENCE && n <= 2 * usual { n } else { usual.min(MAX_CADENCE) }
    }

    fn content_fps(&self) -> Option<f32> {
        self.usual().map(|m| (1.0 / (m as f64 * self.period)) as f32)
    }
}

/// The display's next vblank from now and its refresh period, from DWM. DWM's vblank time
/// may be the last one or the coming one, so only its phase is used. An estimate: `Vsync` has
/// the real ones once frames have been shown.
fn vblank() -> Option<(Instant, Duration)> {
    let (now_qpc, freq) = qpc_now()?;
    let mut info = DWM_TIMING_INFO { cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32, ..Default::default() };
    unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut info).ok()? };
    let now = Instant::now();
    let period_qpc = info.qpcRefreshPeriod as i64;
    if period_qpc <= 0 {
        return None;
    }
    let phase = (now_qpc - info.qpcVBlank as i64).rem_euclid(period_qpc);
    let until = Duration::from_secs_f64((period_qpc - phase) as f64 / freq as f64);
    Some((now + until, Duration::from_secs_f64(period_qpc as f64 / freq as f64)))
}

fn qpc_now() -> Option<(i64, i64)> {
    let (mut now, mut freq) = (0i64, 0i64);
    unsafe {
        QueryPerformanceCounter(&mut now).ok()?;
        QueryPerformanceFrequency(&mut freq).ok()?;
    }
    (freq > 0).then_some((now, freq))
}

/// The display's vblanks as DXGI's frame statistics report them: when our frames were really
/// scanned out. DWM's estimate (`vblank`) can sit a couple of milliseconds off and drift, too
/// coarse to keep presents clear of the moment DWM picks up the next frame.
#[derive(Default)]
struct Vsync {
    /// (vblank count, its QPC time): where the measurement starts, and the latest.
    anchor: Option<(u32, i64)>,
    latest: Option<(u32, i64)>,
}

impl Vsync {
    /// Vblanks to measure over before the period is trusted (half a second at 240 Hz).
    const SPAN: u32 = 120;

    fn feed(&mut self, refresh: u32, qpc: i64) {
        if refresh == 0 || qpc <= 0 {
            return;
        }
        if let Some((r, q)) = self.latest {
            if refresh < r || qpc < q {
                // The count went back (a new swap chain or display): start over.
                *self = Vsync::default();
            } else if refresh == r {
                return;
            }
        }
        self.anchor.get_or_insert((refresh, qpc));
        self.latest = Some((refresh, qpc));
    }

    /// The next vblank after now and the refresh period: measured once there's enough to go
    /// on and it agrees with DWM's period (the display may have changed rate), else DWM's.
    fn next(&mut self) -> Option<(Instant, Duration)> {
        let dwm = vblank();
        let (Some((r0, q0)), Some((r1, q1))) = (self.anchor, self.latest) else { return dwm };
        if r1 - r0 < Self::SPAN {
            return dwm;
        }
        let (now_qpc, freq) = qpc_now()?;
        let now = Instant::now();
        let period = (q1 - q0) as f64 / (r1 - r0) as f64;
        if let Some((_, p)) = dwm
            && (period / freq as f64 / p.as_secs_f64() - 1.0).abs() > 0.02
        {
            self.anchor = self.latest;
            return dwm;
        }
        let k = ((now_qpc - q1) as f64 / period).floor() + 1.0;
        let until = (q1 as f64 + k * period - now_qpc as f64) / freq as f64;
        Some((now + Duration::from_secs_f64(until.max(0.0)), Duration::from_secs_f64(period / freq as f64)))
    }
}

/// Fits the frames still waiting from the last pair (ending with its real picture) in before
/// `first`, when the next pair wants to start, so the delay doesn't build up. They move closer
/// together, by whole refreshes, down to one refresh apart: motion speeds up for a moment. Only
/// if they don't fit even then are generated frames dropped (spread out; never the next one
/// due or the real picture). Returns when the next pair can start and how many were dropped.
///
/// Games that miss a frame deliver the next one early (held 3 card frames, then 1): test
/// sessions showed the whole queue thrown away each time before (a visible jump, and real
/// pictures skipped), 66 times in 100 s of play.
fn catch_up(queue: &mut VecDeque<Entry>, first: Instant, every: Duration, refresh: Duration) -> (Instant, u32) {
    let (Some(front), Some(back)) = (queue.front(), queue.back()) else {
        return (first, 0);
    };
    if back.at + every <= first {
        return (first, 0);
    }
    let front_at = front.at;
    let r = refresh.as_secs_f64();
    // Whole refreshes from the next frame due to `first`: each waiting frame needs one.
    let room = (first.saturating_duration_since(front_at).as_secs_f64() / r + 1e-6).floor() as usize;
    let mut dropped = 0;
    while queue.len() > room.max(1) {
        let generated: Vec<usize> = (1..queue.len()).filter(|&i| !matches!(queue[i].what, Showing::Slot(_))).collect();
        let Some(&i) = generated.get(generated.len() / 2) else { break };
        queue.remove(i);
        dropped += 1;
    }
    let n = queue.len() as u32;
    let planned = ((every.as_secs_f64() / r).round() as u32).max(1);
    let gap = (room as u32 / n.max(1)).clamp(1, planned);
    for (i, e) in queue.iter_mut().enumerate() {
        e.at = front_at + refresh * (gap * i as u32);
    }
    let last = queue.back().expect("not empty").at;
    (first.max(last + refresh * gap), dropped)
}

#[derive(Clone, Copy, PartialEq)]
enum Showing {
    Black,
    /// A real picture (decoded slot).
    Slot(usize),
    /// A generated frame (index into the ring).
    Generated(usize),
    /// A frame still to be generated into the ring at index .0, at time .1 of the current pair.
    /// Made a step before it's due (the first one right away), so the GPU's work for a pair
    /// before its first frame is the motion and that one frame.
    Pending(usize, f32),
}

/// A frame waiting for its time.
struct Entry {
    at: Instant,
    what: Showing,
    /// For real pictures, when they arrived (for latency).
    arrived: Option<Instant>,
}

/// What the next present is, for test sessions (`Video::draw` records it).
#[derive(Clone, Copy, Default)]
struct PresentMeta {
    due: Option<Instant>,
    arrived: Option<Instant>,
}

/// Rate and pacing counters for one stretch of time.
#[derive(Clone)]
struct Window {
    since: Instant,
    frames: u64,
    pictures: u64,
    presented: u64,
    generated: u64,
    lat_sum: Duration,
    lat_n: u64,
    lat_max: Duration,
    gap_max: Duration,
    last: Option<Instant>,
}

impl Window {
    fn new() -> Self {
        Window {
            since: Instant::now(),
            frames: 0,
            pictures: 0,
            presented: 0,
            generated: 0,
            lat_sum: Duration::ZERO,
            lat_n: 0,
            lat_max: Duration::ZERO,
            gap_max: Duration::ZERO,
            last: None,
        }
    }

    fn present(&mut self, now: Instant, generated: bool, arrived: Option<Instant>) {
        self.presented += 1;
        if generated {
            self.generated += 1;
        }
        if let Some(a) = arrived {
            let l = now.saturating_duration_since(a);
            self.lat_sum += l;
            self.lat_n += 1;
            self.lat_max = self.lat_max.max(l);
        }
        if let Some(prev) = self.last {
            self.gap_max = self.gap_max.max(now - prev);
        }
        self.last = Some(now);
    }

    fn report(&self, label: &str, smoothing: bool) -> String {
        let secs = self.since.elapsed().as_secs_f64().max(1e-6);
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let smooth = if smoothing {
            format!(", {:.1} pictures/s, {:.1} generated/s", self.pictures as f64 / secs, self.generated as f64 / secs)
        } else {
            String::new()
        };
        format!(
            "{label}: source {:.2} fps, presented {:.2} fps{smooth}, arrival->present avg {:.2} ms max {:.2} ms, \
             longest gap {:.1} ms",
            self.frames as f64 / secs,
            self.presented as f64 / secs,
            ms(self.lat_sum) / self.lat_n.max(1) as f64,
            ms(self.lat_max),
            ms(self.gap_max),
        )
    }
}

struct Video<'a> {
    shared: &'a VideoShared,
    hwnd: HWND,
    gpu: Gpu,
    swap: Swap,
    frames: Option<Frames>,
    /// The hardware flow setting `frames` was made with.
    made_hw_flow: bool,
    showing: Showing,
    /// Slot the next decode goes into.
    next_slot: usize,
    /// Slot of the newest picture, if it has a pyramid (smoothing on).
    key: Option<usize>,
    /// Next generated-frame texture to use.
    next_out: usize,
    queue: VecDeque<Entry>,
    last_raw: Option<(Vec<u8>, RawInfo)>,
    /// A new picture held back until the present due first has gone (see `run`).
    deferred: Option<RawFrame>,
    spare: Spare,
    last_decode: Option<Decoding>,
    last_at: Option<Instant>,
    since_new: u32,
    cadence: Cadence,
    client: (u32, u32),
    aspect: f32,
    window: Window,
    total: Option<Window>,
    started: Instant,
    /// The last pair's GPU work: a fence after it, when it was submitted, and its picture.
    pending: Option<(windows::Win32::Graphics::Direct3D11::ID3D11Query, Instant, u64)>,
    /// Test sessions: presents so far, the card frame in each slot, the picture and time of
    /// each generated frame, and what the next present is.
    presents: u64,
    /// The display's real vblanks, from what DXGI says about our presents.
    vsync: Vsync,
    slot_seq: [u64; 2],
    out_meta: [(u64, f32); crate::gpu::OUTS],
    meta: PresentMeta,
    /// How long a pair's GPU work takes (smoothed), and whether it's been measured yet.
    gpu_time: Duration,
    gpu_measured: bool,
    /// When a pair was last shortened against the delay ratchet (see `WAITED`).
    shortened: Option<Instant>,
    /// Slots of the pair the motion field is for, while it has frames still to generate.
    pair: Option<(usize, usize)>,
    /// A recent vblank (or the coming one), the refresh period, and when they were read: for
    /// when frames reach the screen, for the FPS counter.
    refresh: Option<(Instant, Duration, Instant)>,
    /// RTX Video Super Resolution, while the video processor works.
    upscaler: Option<Upscaler>,
}

fn client_size(hwnd: HWND) -> (u32, u32) {
    let mut r = RECT::default();
    unsafe {
        let _ = GetClientRect(hwnd, &mut r);
    }
    ((r.right - r.left).max(1) as u32, (r.bottom - r.top).max(1) as u32)
}

fn monitor_size(hwnd: HWND) -> (u32, u32) {
    let mut info = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
    unsafe {
        let m = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        if !GetMonitorInfoW(m, &mut info).as_bool() {
            return (1, 1);
        }
    }
    let r = info.rcMonitor;
    ((r.right - r.left).max(1) as u32, (r.bottom - r.top).max(1) as u32)
}

/// The video's rectangle in the client area: fitted to the aspect, on whole pixels (at 1440p
/// in fullscreen it lands exactly 1:1 on the screen).
fn fit(client: (u32, u32), aspect: f32) -> [f32; 4] {
    let (vw, vh) = (client.0 as f32, client.1 as f32);
    let (w, h) =
        if vw / vh > aspect { ((vh * aspect).round(), vh.round()) } else { (vw.round(), (vw / aspect).round()) };
    [((vw - w) / 2.0).floor(), ((vh - h) / 2.0).floor(), w, h]
}

fn run(shared: &VideoShared, rx: Receiver<RawFrame>, spare: Spare) -> Result<()> {
    // Wait for the window.
    let hwnd = loop {
        if let Some(&h) = shared.gpu.hwnd.get() {
            break HWND(h as *mut _);
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let gpu = Gpu::new().context("starting Direct3D")?;
    let client = client_size(hwnd);
    let monitor = monitor_size(hwnd);
    let swap = Swap::new(&gpu, hwnd, client.0.max(monitor.0), client.1.max(monitor.1))?;
    let upscaler = Upscaler::new(&gpu)
        .inspect_err(|e| crate::log::line(&format!("upscale: no video processor ({e:#}); using Lanczos")))
        .ok();
    let mut v = Video {
        shared,
        hwnd,
        gpu,
        swap,
        frames: None,
        made_hw_flow: false,
        showing: Showing::Black,
        next_slot: 0,
        key: None,
        next_out: 0,
        queue: VecDeque::new(),
        last_raw: None,
        deferred: None,
        spare,
        last_decode: None,
        last_at: None,
        since_new: 0,
        cadence: Cadence::new(),
        client,
        aspect: f32::from_bits(shared.aspect.load(Ordering::Relaxed)),
        window: Window::new(),
        total: None,
        started: Instant::now(),
        pending: None,
        presents: 0,
        vsync: Vsync::default(),
        slot_seq: [0; 2],
        out_meta: [(0, 0.0); crate::gpu::OUTS],
        meta: PresentMeta::default(),
        gpu_time: GPU_GUESS,
        gpu_measured: false,
        shortened: None,
        pair: None,
        refresh: None,
        upscaler,
    };
    v.draw(None)?;
    shared.gpu.ready.store(true, Ordering::Release);
    let timing = vblank();
    v.refresh = timing.map(|(at, p)| (at, p, Instant::now()));
    v.note_refresh();
    match timing {
        Some((_, p)) => crate::log::line(&format!("gpu: drawing video on the GPU, display at {:.1} Hz", 1.0 / p.as_secs_f64())),
        None => crate::log::line("gpu: drawing video on the GPU, display timing unknown"),
    }
    // Frames that queued up while starting are stale.
    while rx.try_recv().is_ok() {}

    loop {
        if shared.gpu.paused.load(Ordering::Relaxed) {
            v.queue.clear();
            if let Some(raw) = v.deferred.take() {
                v.spare.give(raw.data);
            }
            match rx.recv_timeout(Duration::from_millis(250)) {
                Ok(raw) => v.spare.give(raw.data),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
            // Coming back, the next picture starts afresh (no stale cadence).
            v.last_at = None;
            continue;
        }
        let msg = if v.queue.is_empty() {
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(m) => Some(m),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        } else {
            match rx.try_recv() {
                Ok(m) => Some(m),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => return Ok(()),
            }
        };
        if let Some(raw) = msg {
            // A new picture's GPU work (upload, decode, motion: ~3 ms at 4K) goes in the queue
            // ahead of any present after it. A present due before that's done would miss its
            // vblank: at 30 -> 120 the pair's last frame is due just as the next picture
            // arrives, and 15% of them slipped a refresh (a steady stutter). So the picture
            // waits until that present has gone; its frames come after it anyway.
            if v.would_delay_present() {
                if let Some(older) = v.deferred.replace(raw) {
                    v.arrive(older)?;
                }
            } else {
                if let Some(older) = v.deferred.take() {
                    v.arrive(older)?;
                }
                v.arrive(raw)?;
            }
            continue;
        }
        if v.deferred.is_some() && !v.would_delay_present() {
            let raw = v.deferred.take().expect("checked");
            v.arrive(raw)?;
            continue;
        }
        v.watch()?;
        v.measure();
        if shared.gpu.target.load(Ordering::Relaxed) == 0 {
            v.queue.clear();
        }
        let Some(front) = v.queue.front() else {
            continue;
        };
        let now = Instant::now();
        if now >= front.at {
            v.generate(1);
            let e = v.queue.pop_front().expect("non-empty");
            v.meta.due = Some(e.at);
            v.show(e.what, e.arrived)?;
            // The next one now, a step ahead of its time.
            v.generate(1);
            if let Some(raw) = v.deferred.take() {
                v.arrive(raw)?;
            }
            continue;
        }
        std::thread::sleep((front.at - now).min(Duration::from_millis(1)));
    }
}

impl Video<'_> {
    fn draw(&mut self, what: Option<Showing>) -> Result<()> {
        let what = what.unwrap_or(self.showing);
        let src = match (what, &self.frames) {
            (Showing::Slot(i), Some(f)) => Some(&f.slots[i].frame),
            (Showing::Generated(i) | Showing::Pending(i, _), Some(f)) => Some(&f.outs[i]),
            _ => None,
        };
        let rect = fit(self.client, self.aspect);
        let (rw, rh) = (rect[2] as u32, rect[3] as u32);
        // Bigger on screen than it arrives: RTX Video Super Resolution if it's running, else
        // the present shader's Lanczos.
        let mut scaling = Scaling::NotUp;
        if let Some(t) = src
            && (rw > t.w || rh > t.h)
        {
            scaling = Scaling::Lanczos;
            if self.shared.gpu.rtx.load(Ordering::Relaxed)
                && let Some(up) = self.upscaler.as_mut()
            {
                match up.run(&self.gpu, t, rw, rh) {
                    Ok(Some(_)) => scaling = Scaling::Rtx,
                    Ok(None) => {}
                    Err(e) => {
                        crate::log::line(&format!("upscale: the video processor failed ({e:#}); using Lanczos"));
                        self.upscaler = None;
                    }
                }
            }
        }
        self.shared.gpu.scaling.store(scaling as u8, Ordering::Relaxed);
        let shown = if scaling == Scaling::Rtx { self.upscaler.as_ref().and_then(|u| u.output()) } else { src };
        let has_src = src.is_some();
        self.presents += 1;
        let mark = telemetry::marks().then_some((self.presents & 0xFFFF) as u32);
        let call = Instant::now();
        self.swap.show(&self.gpu, shown, rect, mark)?;
        let meta = std::mem::take(&mut self.meta);
        let stats = self.swap.frame_stats();
        if let Some((_, _, sync_refresh, sync_qpc)) = stats {
            self.vsync.feed(sync_refresh, sync_qpc);
        }
        if telemetry::on() {
            self.record_present(what, has_src, meta, call, stats);
        }
        self.showing = if has_src { what } else { Showing::Black };
        Ok(())
    }

    /// Test sessions: the present just made, and the latest present DXGI says reached the
    /// screen (and at which vblank).
    fn record_present(
        &self,
        what: Showing,
        has_src: bool,
        meta: PresentMeta,
        call: Instant,
        stats: Option<(u32, u32, u32, i64)>,
    ) {
        let done = Instant::now();
        let (kind, seq, t) = match what {
            _ if !has_src => (0, 0, 0.0),
            Showing::Slot(i) => (1, self.slot_seq[i], 1.0),
            Showing::Generated(i) | Showing::Pending(i, _) => (2, self.out_meta[i].0, self.out_meta[i].1),
            Showing::Black => (0, 0, 0.0),
        };
        telemetry::emit(Ev::Present(telemetry::PresentRec {
            n: self.presents,
            swap_count: self.swap.present_count(),
            kind,
            seq,
            t,
            due: meta.due,
            call,
            done,
            arrived: meta.arrived,
            queued: self.queue.len() as u32,
        }));
        if let Some((present_count, refresh, sync_refresh, sync_qpc)) = stats {
            telemetry::emit(Ev::Displayed { present_count, refresh, sync_refresh, sync_qpc });
        }
    }

    /// Presents and counts it.
    fn show(&mut self, what: Showing, arrived: Option<Instant>) -> Result<()> {
        self.meta.arrived = arrived;
        self.draw(Some(what))?;
        let now = Instant::now();
        let generated = matches!(what, Showing::Generated(_) | Showing::Pending(..));
        self.window.present(now, generated, arrived);
        if let Some(t) = &mut self.total {
            t.present(now, generated, arrived);
        }
        if self.shared.meter.is_on() {
            let seen = self.on_screen(now);
            self.shared.meter.shown(seen, generated, arrived);
        }
        Ok(())
    }

    /// When a frame presented at `now` reaches the screen: the vblank at or after it. Frames
    /// are composed on vblanks, so this leaves out the scheduling noise of the present itself
    /// and keeps what the eye gets (a frame that misses a vblank shows a refresh late).
    fn on_screen(&mut self, now: Instant) -> Instant {
        if self.refresh.is_none_or(|(_, _, read)| now.saturating_duration_since(read) > Duration::from_millis(500)) {
            self.refresh = self.vsync.next().map(|(v, p)| (v, p, now));
        }
        self.note_refresh();
        let Some((v, p, _)) = self.refresh else {
            return now;
        };
        let p = p.as_secs_f64();
        if now <= v {
            v - Duration::from_secs_f64(((v - now).as_secs_f64() / p).floor() * p)
        } else {
            v + Duration::from_secs_f64(((now - v).as_secs_f64() / p).ceil() * p)
        }
    }

    /// Follows the window's size and the video's aspect, and goes black without a signal.
    fn watch(&mut self) -> Result<()> {
        let client = client_size(self.hwnd);
        let aspect = f32::from_bits(self.shared.aspect.load(Ordering::Relaxed));
        let lost = self.last_at.is_some_and(|t| t.elapsed() > SIGNAL_TIMEOUT) && self.showing != Showing::Black;
        if client != self.client || aspect != self.aspect || lost {
            self.client = client;
            self.aspect = aspect;
            self.swap.ensure(&self.gpu, client.0, client.1)?;
            if lost {
                telemetry::note("signal_lost", || "no frame for a second: showing black".into());
                self.queue.clear();
                self.key = None;
                self.draw(Some(Showing::Black))?;
            } else {
                self.draw(None)?;
            }
        }
        self.tick_stats();
        Ok(())
    }

    /// Publishes the display's refresh rate for the UI (the window may move to another display).
    fn note_refresh(&self) {
        if let Some((_, p, _)) = self.refresh {
            let hz = (1.0 / p.as_secs_f64()).round() as u32;
            self.shared.gpu.refresh_hz.store(hz, Ordering::Relaxed);
        }
    }

    fn tick_stats(&mut self) {
        let smoothing = self.shared.gpu.target.load(Ordering::Relaxed) > 0;
        if self.total.is_none() && self.started.elapsed() >= Duration::from_secs(2) {
            self.total = Some(Window::new());
        }
        if let Some(t) = &self.total
            && t.presented % 30 == 0
        {
            *self.shared.gpu.summary.lock().unwrap() = t.report("TOTAL", smoothing);
        }
        self.shared.meter.set_gpu(smoothing.then_some(self.gpu_time));
        if self.window.since.elapsed() >= Duration::from_secs(5) {
            if let Some((at, p)) = self.vsync.next() {
                self.refresh = Some((at, p, Instant::now()));
                self.note_refresh();
            }
            if self.shared.gpu.stats.load(Ordering::Relaxed) {
                eprintln!(
                    "{}, GPU {:.1} ms a pair",
                    self.window.report("5s", smoothing),
                    self.gpu_time.as_secs_f64() * 1000.0
                );
            }
            self.window = Window::new();
        }
    }

    /// A card frame that arrived at `at`: a new picture, a repeat, or not checked (None).
    fn count_frame(&mut self, at: Instant, checked: Option<bool>) {
        self.shared.meter.card_frame(at, checked);
        let picture = checked.unwrap_or(true);
        self.window.frames += 1;
        if picture {
            self.window.pictures += 1;
        }
        if let Some(t) = &mut self.total {
            t.frames += 1;
            if picture {
                t.pictures += 1;
            }
        }
    }

    /// Makes the first `n` frames still to be generated, in order.
    fn generate(&mut self, n: usize) {
        let (Some((a, b)), Some(f)) = (self.pair, &self.frames) else {
            return;
        };
        for e in self.queue.iter_mut().filter(|e| matches!(e.what, Showing::Pending(..))).take(n) {
            if let Showing::Pending(out, t) = e.what {
                f.interpolate(&self.gpu, a, b, t, out);
                e.what = Showing::Generated(out);
            }
        }
    }

    /// Picks up how long the last pair's GPU work took.
    fn measure(&mut self) {
        if telemetry::on()
            && let Some(f) = self.frames.as_mut()
        {
            for (seq, counts) in f.poll_counts(&self.gpu) {
                telemetry::emit(Ev::PairCounts { seq, counts });
            }
        }
        let Some((fence, at, seq)) = &self.pending else {
            return;
        };
        if !self.gpu.passed(fence) {
            return;
        }
        let took = at.elapsed();
        telemetry::emit(Ev::PairGpu { seq: *seq, gpu_us: took.as_micros() as u32 });
        // Quick to grow (a frame shown late costs more than a little delay), slow to shrink;
        // the first measurement replaces the guess (else the first seconds run late).
        let w = if !self.gpu_measured {
            1.0
        } else if took > self.gpu_time {
            0.5
        } else {
            0.1
        };
        self.gpu_measured = true;
        self.gpu_time = self.gpu_time.mul_f64(1.0 - w) + took.mul_f64(w);
        self.pending = None;
    }

    /// Keeps `data` as the last frame, for spotting repeats; the one it replaces is released.
    fn keep_raw(&mut self, data: Option<(Vec<u8>, RawInfo)>) {
        if let Some((old, info)) = std::mem::replace(&mut self.last_raw, data) {
            self.release(old, info);
        }
    }

    /// Done with a card frame: in a test session it's recorded (the input video takes the
    /// buffer and gives it back when written), else it goes straight back for reuse.
    fn release(&mut self, data: Vec<u8>, info: RawInfo) {
        if !telemetry::on() {
            self.spare.give(data);
            return;
        }
        let spare = self.spare.clone();
        let format = (info.w, info.h, info.layout);
        let input = match telemetry::input_frame(data, format, info.fps, || Arc::new(move |b| spare.give(b))) {
            Ok(at) => Some(at),
            Err(data) => {
                self.spare.give(data);
                None
            }
        };
        telemetry::emit(Ev::Capture { seq: info.seq, at: info.at, copy_us: info.copy_us, w: info.w, h: info.h, input });
    }

    /// Whether a present is due before a new picture's GPU work would be done.
    fn would_delay_present(&self) -> bool {
        self.shared.gpu.target.load(Ordering::Relaxed) > 0
            && self.queue.front().is_some_and(|e| e.at < Instant::now() + self.gpu_time + DEFER_MARGIN)
    }

    /// Test sessions: what this card frame turned out to be.
    fn frame_kind(&self, seq: u64, at: Instant, kind: FrameKind, held: u32) {
        if telemetry::on() {
            telemetry::emit(Ev::Frame { seq, at, recv: Instant::now(), kind, held });
        }
    }

    fn take_slot(&mut self) -> usize {
        let s = self.next_slot;
        self.next_slot = 1 - s;
        s
    }

    fn arrive(&mut self, raw: RawFrame) -> Result<()> {
        let target = self.shared.gpu.target.load(Ordering::Relaxed);
        let gap = match self.last_at {
            Some(t) => !self.cadence.frame(raw.at.saturating_duration_since(t).as_secs_f64()),
            None => true,
        };
        self.last_at = Some(raw.at);
        // Frame generation's textures exist only while it's on (several hundred megabytes at
        // 4K), so turning it on or off remakes them like a new format.
        let smoothing = target > 0;
        // So does switching how motion is measured.
        let hw_flow = self.shared.gpu.hw_flow.load(Ordering::Relaxed);
        let new_format = self.frames.as_ref().is_none_or(|f| {
            (f.w, f.h, f.layout, f.has_gen()) != (raw.w, raw.h, raw.decode.layout, smoothing)
        }) || (smoothing && self.made_hw_flow != hw_flow);
        if new_format {
            telemetry::note("frames", || {
                let fg = if smoothing { "on" } else { "off" };
                format!("{}x{} {:?}, frame generation {fg}", raw.w, raw.h, raw.decode.layout)
            });
            self.queue.clear();
            self.pair = None;
            self.key = None;
            self.showing = Showing::Black;
            self.next_out = 0;
            self.frames = None;
            let method = crate::gpu::flow_method(hw_flow);
            let frames = Frames::with(&self.gpu, raw.w, raw.h, raw.decode.layout, smoothing, method)?;
            let in_use = if !smoothing { 0 } else if frames.hardware_flow() { 2 } else { 1 };
            self.shared.gpu.flow_in_use.store(in_use, Ordering::Relaxed);
            self.made_hw_flow = hw_flow;
            self.frames = Some(frames);
            let spare = self.spare.clone();
            telemetry::input_format((raw.w, raw.h, raw.decode.layout), raw.fps, || Arc::new(move |b| spare.give(b)));
        }
        let new_decode = self.last_decode != Some(raw.decode);
        self.last_decode = Some(raw.decode);
        // `client` may be stale if the window just changed; keep up before drawing.
        self.watch()?;

        if target == 0 {
            // Straight through: decode and present now.
            self.queue.clear();
            self.key = None;
            let slot = self.take_slot();
            self.slot_seq[slot] = raw.seq;
            let f = self.frames.as_ref().expect("made above");
            f.upload(&self.gpu, &raw.data)?;
            f.decode(&self.gpu, slot, &raw.decode, false);
            self.frame_kind(raw.seq, raw.at, FrameKind::Passthrough, 0);
            self.show(Showing::Slot(slot), Some(raw.at))?;
            // For the FPS counter, whether the game's picture changed: checked after
            // presenting, so it adds no delay.
            let checked = if self.shared.meter.is_on() {
                let new = self.last_raw.as_ref().is_none_or(|prev| !same_picture(&prev.0, &raw.data));
                let info = raw.info();
                self.keep_raw(Some((raw.data, info)));
                Some(new)
            } else {
                self.keep_raw(None);
                let info = raw.info();
                self.release(raw.data, info);
                None
            };
            self.count_frame(raw.at, checked);
            return Ok(());
        }

        let restart = gap || new_format || new_decode || self.key.is_none();
        if !restart {
            self.since_new += 1;
            let repeat = self.last_raw.as_ref().is_some_and(|prev| same_picture(&prev.0, &raw.data));
            if repeat {
                self.frame_kind(raw.seq, raw.at, FrameKind::Repeat, self.since_new);
                self.count_frame(raw.at, Some(false));
                // Nothing waiting to be shown (a still picture): the screen holds this one,
                // as it does with smoothing off, so the counter counts it as a frame.
                if self.queue.is_empty() && self.shared.meter.is_on() {
                    let seen = self.on_screen(Instant::now());
                    self.shared.meter.shown(seen, false, None);
                }
                let info = raw.info();
                self.keep_raw(Some((raw.data, info)));
                return Ok(());
            }
        }
        self.count_frame(raw.at, Some(true));
        // The last pair's frames still to be generated need its slots and motion, which this
        // picture is about to replace: make them now.
        self.generate(usize::MAX);
        self.pair = None;
        let slot = self.take_slot();
        self.slot_seq[slot] = raw.seq;
        {
            let f = self.frames.as_ref().expect("made above");
            f.upload(&self.gpu, &raw.data)?;
            f.decode(&self.gpu, slot, &raw.decode, true);
        }
        let info = raw.info();
        self.keep_raw(Some((raw.data, info)));
        let prev_key = self.key.replace(slot);
        let mut kind = FrameKind::Restart;
        let mut held_n = 0;
        let held = if restart {
            self.since_new = 0;
            None
        } else {
            let n = std::mem::take(&mut self.since_new);
            held_n = n;
            if n > MAX_CADENCE {
                kind = FrameKind::AfterStill;
            }
            let held = self.cadence.picture(n);
            if let Some(fps) = self.cadence.content_fps() {
                self.shared.gpu.content_fps.store(fps.to_bits(), Ordering::Relaxed);
            }
            // The first change after a still picture is usually a reaction to input (moving
            // through a menu, opening the map): show it now instead of a frame's time later,
            // and don't invent motion into it. Smoothing picks up again from the next one.
            if n > MAX_CADENCE { None } else { Some(held) }
        };
        let (Some(a), Some(held)) = (prev_key, held) else {
            self.frame_kind(raw.seq, raw.at, kind, held_n);
            if let Some(f) = &mut self.frames {
                f.forget_motion();
            }
            self.queue.clear();
            return self.show(Showing::Slot(slot), Some(raw.at));
        };

        // How many steps the picture's time splits into, no shorter than a refresh each.
        let span = held as f64 * self.cadence.period;
        let timing = self.vsync.next();
        if let Some((v, p)) = timing {
            self.refresh = Some((v, p, Instant::now()));
            self.note_refresh();
        }
        let refresh = timing.map_or(1.0 / 240.0, |(_, p)| p.as_secs_f64());
        // A little slack: 33.33 ms over 4.1667 ms refreshes measures as 7.99, which is 8.
        let by_refresh = (span / refresh + 0.25).floor().max(1.0) as u32;
        let steps = ((span * target as f64).round() as u32).clamp(1, MAX_STEPS).min(by_refresh);
        if steps <= 1 {
            self.frame_kind(raw.seq, raw.at, FrameKind::Single, held_n);
            self.queue.clear();
            return self.show(Showing::Slot(slot), Some(raw.at));
        }
        self.frame_kind(raw.seq, raw.at, FrameKind::Pair, held_n);
        let submitted = Instant::now();
        let f = self.frames.as_mut().expect("made above");
        // Two pairs' worth of generated frames (see `Frames::outs`), made the first time a
        // pair needs this many: 6 at 30 -> 120, 14 at 30 -> 240.
        f.ensure_outs(&self.gpu, 2 * (steps as usize - 1))?;
        let ring = f.outs.len();
        f.estimate(&self.gpu, a, slot);
        if telemetry::on() {
            f.request_counts(&self.gpu, raw.seq);
        }
        self.pair = Some((a, slot));
        // The first frame now; the rest a step before each is due (see `Pending`).
        let mut made = Vec::with_capacity(steps as usize);
        for j in 1..steps {
            let out = self.next_out;
            self.next_out = (self.next_out + 1) % ring;
            let t = j as f32 / steps as f32;
            self.out_meta[out] = (raw.seq, t);
            if j == 1 {
                f.interpolate(&self.gpu, a, slot, t, out);
                made.push(Showing::Generated(out));
            } else {
                made.push(Showing::Pending(out, t));
            }
        }
        if self.pending.is_none()
            && let Ok(fence) = self.gpu.fence()
        {
            self.pending = Some((fence, submitted, raw.seq));
        }

        // Line the frames up on refreshes, `every` apart: the first right after the next
        // vblank the GPU can make, or straight after what's still waiting from the previous
        // pair, so spacing stays even and nothing is dropped when a picture arrives a little
        // early.
        let now = Instant::now();
        let step = span / steps as f64;
        let ready = submitted + self.gpu_time;
        let (mut first, every) = match timing {
            Some((next, period)) => {
                let k = (step / period.as_secs_f64()).round().max(1.0) as u32;
                // The first present: `PRESENT_PHASE` into a refresh, once the GPU has made the
                // frame (`timing` was read before the work above).
                let mut slot = next - period + period.mul_f64(PRESENT_PHASE);
                while slot < now.max(ready) {
                    slot += period;
                }
                (slot, period * k)
            }
            None => (now, Duration::from_secs_f64(step)),
        };
        // Still busy with the last pair (the game's last picture came late and this one
        // early, or a slow pair pushed things later): fit what's waiting in before this pair.
        let refresh = timing.map_or(every, |(_, p)| p);
        let (start, dropped) = catch_up(&mut self.queue, first, every, refresh);
        first = start;
        let (skipped, backlog_cleared) = (dropped, false);
        if telemetry::on() {
            telemetry::emit(Ev::Pair(telemetry::PairPlan {
                seq: raw.seq,
                at: now,
                held,
                steps,
                target,
                content_fps: self.shared.gpu.content_fps(),
                gpu_guess: self.gpu_time,
                first,
                every,
                skipped,
                backlog_cleared,
                refresh: timing.map(|(_, p)| p),
            }));
        }
        let last = made.len() as u32;
        let mut real_at = first + every * last;
        // Waited behind the last pair's picture: show this one a refresh early (see `WAITED`).
        // Only where an interval is two refreshes or more, so none gets shorter than one.
        if let Some((_, period)) = timing
            && now.saturating_duration_since(raw.at) > WAITED
            && every >= period * 2
            && self.shortened.is_none_or(|t| now.saturating_duration_since(t) >= SHORTEN_EVERY)
        {
            real_at -= period;
            self.shortened = Some(now);
            telemetry::note("pacing", || format!("picture {} shown a refresh early (it waited)", raw.seq));
        }
        for (j, what) in made.into_iter().enumerate() {
            self.queue.push_back(Entry { at: first + every * j as u32, what, arrived: None });
        }
        self.queue.push_back(Entry { at: real_at, what: Showing::Slot(slot), arrived: Some(raw.at) });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cadence_of_a_30fps_game_on_a_60hz_card() {
        let mut c = Cadence::new();
        for _ in 0..10 {
            assert!(c.frame(1.0 / 60.0));
            assert_eq!(c.picture(2), 2);
        }
        assert!((c.content_fps().unwrap() - 30.0).abs() < 0.5);
    }

    #[test]
    fn a_pause_counts_as_one_picture() {
        let mut c = Cadence::new();
        for _ in 0..10 {
            c.picture(2);
        }
        // Still for 20 card frames, then moving again: the move took one picture's time.
        assert_eq!(c.picture(20), 2);
        // A dropped frame in a 30 fps game: held for 3.
        assert_eq!(c.picture(3), 3);
    }

    #[test]
    fn gaps_are_noticed() {
        let mut c = Cadence::new();
        assert!(c.frame(0.017));
        assert!(!c.frame(0.5));
    }

    #[test]
    fn fit_is_whole_pixels_and_centred() {
        assert_eq!(fit((2560, 1440), 16.0 / 9.0), [0.0, 0.0, 2560.0, 1440.0]);
        assert_eq!(fit((5120, 1440), 16.0 / 9.0), [1280.0, 0.0, 2560.0, 1440.0]);
        let r = fit((1001, 700), 16.0 / 9.0);
        assert_eq!(r[2], 1001.0);
        assert_eq!(r[3], 563.0);
    }

    fn waiting(start: Instant, spacing: Duration, kinds: &[bool]) -> VecDeque<Entry> {
        kinds
            .iter()
            .enumerate()
            .map(|(i, &real)| Entry {
                at: start + spacing * i as u32,
                what: if real { Showing::Slot(0) } else { Showing::Generated(i) },
                arrived: None,
            })
            .collect()
    }

    #[test]
    fn on_schedule_nothing_moves() {
        let t = Instant::now();
        let (r, every) = (Duration::from_micros(4167), Duration::from_micros(8333));
        let mut q = waiting(t, every, &[false, false, true]);
        let before: Vec<Instant> = q.iter().map(|e| e.at).collect();
        let (first, dropped) = catch_up(&mut q, t + every * 3, every, r);
        assert_eq!((first, dropped), (t + every * 3, 0));
        assert_eq!(q.iter().map(|e| e.at).collect::<Vec<_>>(), before);
    }

    #[test]
    fn an_early_picture_squeezes_the_wait_instead_of_dropping_it() {
        // 120 on 240 Hz: three frames (the last one real) 2 refreshes apart, but the next pair
        // wants to start 4 refreshes after the first of them.
        let t = Instant::now();
        let (r, every) = (Duration::from_micros(4167), Duration::from_micros(8334));
        let mut q = waiting(t, every, &[false, false, true]);
        let (first, dropped) = catch_up(&mut q, t + r * 4, every, r);
        assert_eq!(dropped, 0, "all three still shown");
        assert_eq!(q.iter().map(|e| e.at).collect::<Vec<_>>(), [t, t + r, t + r * 2]);
        assert_eq!(first, t + r * 4, "the next pair still starts when it wanted to");
    }

    #[test]
    fn with_no_room_generated_frames_go_but_never_the_real_picture() {
        // 240 on 240 Hz: six waiting, a refresh apart, room for three.
        let t = Instant::now();
        let r = Duration::from_micros(4167);
        let mut q = waiting(t, r, &[false, false, false, false, false, true]);
        let (first, dropped) = catch_up(&mut q, t + r * 3, r, r);
        assert_eq!(dropped, 3);
        assert!(matches!(q.back().unwrap().what, Showing::Slot(_)), "the real picture stays");
        assert!(matches!(q.front().unwrap().what, Showing::Generated(0)), "the next one due stays");
        assert_eq!(first, t + r * 3);
    }

    #[test]
    fn duplicate_detection() {
        let a = vec![100u8; 4096];
        let mut b = a.clone();
        assert!(same_picture(&a, &b));
        b[100] = 102;
        assert!(same_picture(&a, &b), "noise within 2 levels");
        b[200] = 110;
        assert!(!same_picture(&a, &b));
    }
}
