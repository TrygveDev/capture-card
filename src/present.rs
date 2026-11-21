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
use crate::upscale::Upscaler;
use crate::video::VideoShared;

/// Most frames made per picture (8 covers 30 fps -> 240, 15 fps -> 120).
const MAX_STEPS: u32 = 8;
/// Pictures held longer than this many card frames are a pause, not the game's frame rate.
const MAX_CADENCE: u32 = 4;
/// No frame for this long: show black (the UI says "No signal").
const SIGNAL_TIMEOUT: Duration = Duration::from_millis(1000);
/// Generated frames are presented this long after a vblank, so each lands on the next one.
const VBLANK_PHASE: Duration = Duration::from_millis(1);
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
    /// The display's refresh rate in whole Hz (0 = not known yet).
    pub refresh_hz: AtomicU32,
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
    pub fn new(target: u32, stats: bool, rtx: bool) -> Self {
        Shared {
            target: AtomicU32::new(target),
            content_fps: AtomicU32::new(0),
            ready: AtomicBool::new(false),
            hwnd: OnceLock::new(),
            stats: AtomicBool::new(stats),
            summary: Mutex::new(String::new()),
            rtx: AtomicBool::new(rtx),
            scaling: AtomicU8::new(Scaling::NotUp as u8),
            refresh_hz: AtomicU32::new(0),
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
    /// Enough for the frame in flight, the one kept for comparison and one to swap.
    const KEEP: usize = 3;

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

impl Presenter {
    pub fn spawn(shared: Arc<VideoShared>) -> Presenter {
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
/// may be the last one or the coming one, so only its phase is used.
fn vblank() -> Option<(Instant, Duration)> {
    let mut info = DWM_TIMING_INFO { cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32, ..Default::default() };
    let (mut now_qpc, mut freq) = (0i64, 0i64);
    unsafe {
        DwmGetCompositionTimingInfo(HWND::default(), &mut info).ok()?;
        QueryPerformanceCounter(&mut now_qpc).ok()?;
        QueryPerformanceFrequency(&mut freq).ok()?;
    }
    let now = Instant::now();
    let period_qpc = info.qpcRefreshPeriod as i64;
    if freq <= 0 || period_qpc <= 0 {
        return None;
    }
    let phase = (now_qpc - info.qpcVBlank as i64).rem_euclid(period_qpc);
    let until = Duration::from_secs_f64((period_qpc - phase) as f64 / freq as f64);
    Some((now + until, Duration::from_secs_f64(period_qpc as f64 / freq as f64)))
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
    showing: Showing,
    /// Slot the next decode goes into.
    next_slot: usize,
    /// Slot of the newest picture, if it has a pyramid (smoothing on).
    key: Option<usize>,
    /// Next generated-frame texture to use.
    next_out: usize,
    queue: VecDeque<Entry>,
    last_raw: Option<Vec<u8>>,
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
    /// The last pair's GPU work: a fence after it and when it was submitted.
    pending: Option<(windows::Win32::Graphics::Direct3D11::ID3D11Query, Instant)>,
    /// How long a pair's GPU work takes (smoothed).
    gpu_time: Duration,
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
        showing: Showing::Black,
        next_slot: 0,
        key: None,
        next_out: 0,
        queue: VecDeque::new(),
        last_raw: None,
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
        gpu_time: GPU_GUESS,
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
            v.show(e.what, e.arrived)?;
            // The next one now, a step ahead of its time.
            v.generate(1);
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
        self.swap.show(&self.gpu, shown, rect)?;
        self.showing = if src.is_some() { what } else { Showing::Black };
        Ok(())
    }

    /// Presents and counts it.
    fn show(&mut self, what: Showing, arrived: Option<Instant>) -> Result<()> {
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
            self.refresh = vblank().map(|(v, p)| (v, p, now));
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
            if let Some((at, p)) = vblank() {
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
        let Some((fence, at)) = &self.pending else {
            return;
        };
        if !self.gpu.passed(fence) {
            return;
        }
        let took = at.elapsed();
        // Quick to grow (a frame shown late costs more than a little delay), slow to shrink.
        let w = if took > self.gpu_time { 0.5 } else { 0.1 };
        self.gpu_time = self.gpu_time.mul_f64(1.0 - w) + took.mul_f64(w);
        self.pending = None;
    }

    /// Keeps `data` as the last frame, for spotting repeats; the one it replaces is reused.
    fn keep_raw(&mut self, data: Option<Vec<u8>>) {
        if let Some(old) = std::mem::replace(&mut self.last_raw, data) {
            self.spare.give(old);
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
        let new_format = self.frames.as_ref().is_none_or(|f| {
            (f.w, f.h, f.layout, f.has_gen()) != (raw.w, raw.h, raw.decode.layout, smoothing)
        });
        if new_format {
            self.queue.clear();
            self.pair = None;
            self.key = None;
            self.showing = Showing::Black;
            self.next_out = 0;
            self.frames = None;
            self.frames = Some(Frames::with(&self.gpu, raw.w, raw.h, raw.decode.layout, smoothing)?);
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
            let f = self.frames.as_ref().expect("made above");
            f.upload(&self.gpu, &raw.data)?;
            f.decode(&self.gpu, slot, &raw.decode, false);
            self.show(Showing::Slot(slot), Some(raw.at))?;
            // For the FPS counter, whether the game's picture changed: checked after
            // presenting, so it adds no delay.
            let checked = if self.shared.meter.is_on() {
                let new = self.last_raw.as_ref().is_none_or(|prev| !same_picture(prev, &raw.data));
                self.keep_raw(Some(raw.data));
                Some(new)
            } else {
                self.keep_raw(None);
                self.spare.give(raw.data);
                None
            };
            self.count_frame(raw.at, checked);
            return Ok(());
        }

        let restart = gap || new_format || new_decode || self.key.is_none();
        if !restart {
            self.since_new += 1;
            let repeat = self.last_raw.as_ref().is_some_and(|prev| same_picture(prev, &raw.data));
            if repeat {
                self.count_frame(raw.at, Some(false));
                // Nothing waiting to be shown (a still picture): the screen holds this one,
                // as it does with smoothing off, so the counter counts it as a frame.
                if self.queue.is_empty() && self.shared.meter.is_on() {
                    let seen = self.on_screen(Instant::now());
                    self.shared.meter.shown(seen, false, None);
                }
                self.keep_raw(Some(raw.data));
                return Ok(());
            }
        }
        self.count_frame(raw.at, Some(true));
        // The last pair's frames still to be generated need its slots and motion, which this
        // picture is about to replace: make them now.
        self.generate(usize::MAX);
        self.pair = None;
        let slot = self.take_slot();
        {
            let f = self.frames.as_ref().expect("made above");
            f.upload(&self.gpu, &raw.data)?;
            f.decode(&self.gpu, slot, &raw.decode, true);
        }
        self.keep_raw(Some(raw.data));
        let prev_key = self.key.replace(slot);
        let held = if restart {
            self.since_new = 0;
            None
        } else {
            let n = std::mem::take(&mut self.since_new);
            let held = self.cadence.picture(n);
            if let Some(fps) = self.cadence.content_fps() {
                self.shared.gpu.content_fps.store(fps.to_bits(), Ordering::Relaxed);
            }
            Some(held)
        };
        let (Some(a), Some(held)) = (prev_key, held) else {
            if let Some(f) = &mut self.frames {
                f.forget_motion();
            }
            self.queue.clear();
            return self.show(Showing::Slot(slot), Some(raw.at));
        };

        // How many steps the picture's time splits into, no shorter than a refresh each.
        let span = held as f64 * self.cadence.period;
        let timing = vblank();
        if let Some((v, p)) = timing {
            self.refresh = Some((v, p, Instant::now()));
            self.note_refresh();
        }
        let refresh = timing.map_or(1.0 / 240.0, |(_, p)| p.as_secs_f64());
        // A little slack: 33.33 ms over 4.1667 ms refreshes measures as 7.99, which is 8.
        let by_refresh = (span / refresh + 0.25).floor().max(1.0) as u32;
        let steps = ((span * target as f64).round() as u32).clamp(1, MAX_STEPS).min(by_refresh);
        if steps <= 1 {
            self.queue.clear();
            return self.show(Showing::Slot(slot), Some(raw.at));
        }
        let submitted = Instant::now();
        let f = self.frames.as_mut().expect("made above");
        // Two pairs' worth of generated frames (see `Frames::outs`), made the first time a
        // pair needs this many: 6 at 30 -> 120, 14 at 30 -> 240.
        f.ensure_outs(&self.gpu, 2 * (steps as usize - 1))?;
        let ring = f.outs.len();
        f.estimate(&self.gpu, a, slot);
        self.pair = Some((a, slot));
        // The first frame now; the rest a step before each is due (see `Pending`).
        let mut made = Vec::with_capacity(steps as usize);
        for j in 1..steps {
            let out = self.next_out;
            self.next_out = (self.next_out + 1) % ring;
            let t = j as f32 / steps as f32;
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
            self.pending = Some((fence, submitted));
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
                // `timing` was read before the work above; step to the first vblank still ahead
                // whose frame the GPU will have finished (with a millisecond to spare before
                // the one after, when it's composed).
                let mut next = next;
                while next < now || next + period < ready + Duration::from_millis(1) {
                    next += period;
                }
                (next + VBLANK_PHASE, period * k)
            }
            None => (now, Duration::from_secs_f64(step)),
        };
        if let Some(back) = self.queue.back() {
            let after = back.at + every;
            if after > first + Duration::from_secs_f64(span) {
                // More than a picture's worth behind: let the backlog go.
                self.queue.clear();
            } else if after > first {
                // Still busy with the last pair: carry on straight after it. A whole step or
                // more behind (a slow pair pushed everything later), skip that many of this
                // pair's first generated frames, so the delay doesn't stay.
                let behind = ((after - first).as_secs_f64() / every.as_secs_f64()) as usize;
                made.drain(..behind.min(made.len()));
                first = after;
            }
        }
        let last = made.len() as u32;
        for (j, what) in made.into_iter().enumerate() {
            self.queue.push_back(Entry { at: first + every * j as u32, what, arrived: None });
        }
        self.queue.push_back(Entry { at: first + every * last, what: Showing::Slot(slot), arrived: Some(raw.at) });
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
