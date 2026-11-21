//! Test sessions: everything the app does, recorded so it can be analysed afterwards
//! (`tools/analyze.py` turns a session into a report).
//!
//! A session is a folder (`%APPDATA%\Passthru\sessions\<date>_<time>` unless given) with:
//! - `session.json`: the app, machine, display, device, settings, and what was recorded;
//! - `capture.csv`: every card frame (arrival, copy time, its frame in `input_*.mkv`);
//! - `frames.csv`: what the presenter made of each card frame (new picture, repeat, restart...);
//! - `pairs.csv`, `pair_gpu.csv`, `pair_counts.csv`: each frame-generation pair, its plan,
//!   measured GPU time and the motion pass's counters (cut and UI-change tests);
//! - `presents.csv`: every frame presented (real or generated, when it was due, when it went);
//! - `display.csv`: DXGI frame statistics, which present reached the screen at which vblank;
//! - `system.csv`: CPU, memory, GPU load and memory, display rate, four times a second;
//! - `events.csv`: everything else (log lines, settings changes, bookmarks, restarts);
//! - `input_NNN.mkv`: the card's frames as received (one video frame per card frame);
//! - `output.mkv`: the screen under the window, each frame carrying a barcode of the present
//!   that made it, so frames on screen map back to presents exactly.
//!
//! Times are microseconds on one clock for the whole run (`us`); `session.json` says where
//! the session starts on it. Nothing here costs anything while no session is running: every
//! hook checks one atomic flag first.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::video::Layout;

/// No console windows for the ffmpeg processes (the app is a GUI program).
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
/// System samples this often.
const SAMPLE_EVERY: Duration = Duration::from_millis(250);
/// Card frames waiting for the input video's encoder at most; more and they're dropped (and
/// counted), so a slow encoder never holds up capture. ffmpeg reads one frame and then takes
/// a few hundred milliseconds to open the encoder: this covers that at 60 fps and more.
const INPUT_QUEUE: usize = 32;

// ---------------------------------------------------------------------------------------------
// The clock.

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Microseconds since the app's clock started (negative for earlier instants).
pub fn us(t: Instant) -> i64 {
    let e = *EPOCH.get_or_init(Instant::now);
    if t >= e { (t - e).as_micros() as i64 } else { -((e - t).as_micros() as i64) }
}

/// A QueryPerformanceCounter value on the same clock as `us`.
pub fn qpc_us(qpc: i64) -> i64 {
    use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    static BASE: OnceLock<(i64, i64, i64)> = OnceLock::new();
    let &(q0, t0, freq) = BASE.get_or_init(|| {
        let (mut q, mut f) = (0i64, 0i64);
        let t = Instant::now();
        unsafe {
            let _ = QueryPerformanceCounter(&mut q);
            let _ = QueryPerformanceFrequency(&mut f);
        }
        (q, us(t), f.max(1))
    });
    t0 + ((qpc - q0) as i128 * 1_000_000 / freq as i128) as i64
}

// ---------------------------------------------------------------------------------------------
// What gets recorded.

/// What the presenter made of a card frame.
#[derive(Clone, Copy, Debug)]
pub enum FrameKind {
    /// Smoothing off: shown as it came.
    Passthrough,
    /// The same picture as the last frame.
    Repeat,
    /// A new picture, the end of a frame-generation pair.
    Pair,
    /// A new picture shown at once: after a gap in the frames, a new format, smoothing just
    /// turned on, or the start.
    Restart,
    /// A new picture shown at once because the one before it was still for a while.
    AfterStill,
    /// A new picture shown at once: too short a time to split into frames.
    Single,
}

pub enum Ev {
    Capture { seq: u64, at: Instant, copy_us: u32, w: u32, h: u32, input: Option<(u32, u32)> },
    Frame { seq: u64, at: Instant, recv: Instant, kind: FrameKind, held: u32 },
    Pair(PairPlan),
    PairGpu { seq: u64, gpu_us: u32 },
    PairCounts { seq: u64, counts: [u32; crate::gpu::COUNTS] },
    Present(PresentRec),
    Displayed { present_count: u32, refresh: u32, sync_refresh: u32, sync_qpc: i64 },
    System(SystemSample),
    Note { at: Instant, kind: &'static str, detail: String },
}

pub struct PairPlan {
    /// The picture ending the pair (its card frame).
    pub seq: u64,
    pub at: Instant,
    /// Card frames the previous picture was held.
    pub held: u32,
    pub steps: u32,
    pub target: u32,
    pub content_fps: f32,
    pub gpu_guess: Duration,
    /// When its first frame is due and the spacing.
    pub first: Instant,
    pub every: Duration,
    /// Generated frames skipped to catch up, and whether the backlog was dropped.
    pub skipped: u32,
    pub backlog_cleared: bool,
    pub refresh: Option<Duration>,
}

pub struct PresentRec {
    /// The presenter's own count, also in the output video's barcode (its low 16 bits).
    pub n: u64,
    /// The swap chain's count after this present (matches `Displayed::present_count`).
    pub swap_count: Option<u32>,
    /// 0 black, 1 real picture, 2 generated.
    pub kind: u8,
    /// The picture shown, or the pair's end picture for a generated frame.
    pub seq: u64,
    pub t: f32,
    pub due: Option<Instant>,
    pub call: Instant,
    pub done: Instant,
    pub arrived: Option<Instant>,
    pub queued: u32,
}

#[derive(Default)]
pub struct SystemSample {
    pub at: Option<Instant>,
    pub cpu_process: f32,
    pub cpu_system: f32,
    pub working_set_mb: f32,
    pub private_mb: f32,
    pub gpu_3d_process: f32,
    pub gpu_3d_total: f32,
    pub gpu_encode_total: f32,
    pub gpu_mem_process_mb: f32,
    pub gpu_mem_total_mb: f32,
    pub refresh_hz: u32,
    pub target: u32,
    pub content_fps: f32,
}

/// What a session records besides the numbers.
#[derive(Clone, Debug)]
pub struct Options {
    pub dir: Option<PathBuf>,
    pub input_video: bool,
    pub output_video: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { dir: None, input_video: true, output_video: true }
    }
}

/// What's known about the run when a session starts (for `session.json`).
pub struct SessionContext {
    pub hwnd: Option<isize>,
    pub info: Value,
}

// ---------------------------------------------------------------------------------------------
// The hub: one writer thread, started with the first session.

enum Msg {
    Start(Box<Session>),
    Ev(Ev),
    Stop(Sender<()>),
}

struct Hub {
    on: AtomicBool,
    /// Presents carry a barcode (the output video is being recorded).
    marks: AtomicBool,
    tx: Sender<Msg>,
    state: Mutex<State>,
    session: AtomicU64,
}

#[derive(Default)]
struct State {
    dir: Option<PathBuf>,
    started: Option<Instant>,
    input: Option<InputRec>,
    want_input: bool,
    output: Option<Child>,
    input_dropped: u64,
}

static HUB: OnceLock<Hub> = OnceLock::new();

fn hub() -> &'static Hub {
    HUB.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("telemetry".into())
            .spawn(move || writer(rx))
            .expect("spawn telemetry writer");
        Hub {
            on: AtomicBool::new(false),
            marks: AtomicBool::new(false),
            tx,
            state: Mutex::new(State::default()),
            session: AtomicU64::new(0),
        }
    })
}

/// A session is recording.
pub fn on() -> bool {
    HUB.get().is_some_and(|h| h.on.load(Ordering::Relaxed))
}

/// Presents should carry their barcode.
pub fn marks() -> bool {
    HUB.get().is_some_and(|h| h.marks.load(Ordering::Relaxed))
}

pub fn emit(ev: Ev) {
    if let Some(h) = HUB.get()
        && h.on.load(Ordering::Relaxed)
    {
        let _ = h.tx.send(Msg::Ev(ev));
    }
}

/// A discrete event; `detail` is only built while recording.
pub fn note(kind: &'static str, detail: impl FnOnce() -> String) {
    if on() {
        emit(Ev::Note { at: Instant::now(), kind, detail: detail() });
    }
}

/// The folder of the session being recorded, and how long it has run.
pub fn status() -> Option<(PathBuf, Duration)> {
    let h = HUB.get()?;
    if !h.on.load(Ordering::Relaxed) {
        return None;
    }
    let s = h.state.lock().unwrap();
    Some((s.dir.clone()?, s.started?.elapsed()))
}

fn sessions_root() -> PathBuf {
    crate::app::data_dir().unwrap_or_else(std::env::temp_dir).join("sessions")
}

/// Starts a session. Returns its folder.
pub fn start(opts: &Options, ctx: SessionContext) -> Result<PathBuf> {
    let h = hub();
    if h.on.load(Ordering::Acquire) {
        bail!("already recording");
    }
    let started = Instant::now();
    let dir = match &opts.dir {
        Some(d) => d.clone(),
        None => sessions_root().join(local_stamp()),
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let id = h.session.fetch_add(1, Ordering::AcqRel) + 1;

    let ffmpeg = ffmpeg();
    let mut output = None;
    let mut notes = Vec::new();
    if opts.output_video {
        match (&ffmpeg, ctx.hwnd) {
            (Some(ff), Some(hwnd)) => match start_output(ff, hwnd, &dir) {
                Ok((child, desc)) => {
                    output = Some(child);
                    notes.push(("output_video", desc));
                }
                Err(e) => notes.push(("output_video_failed", format!("{e:#}"))),
            },
            (None, _) => notes.push(("output_video_failed", "ffmpeg not found".into())),
            (_, None) => notes.push(("output_video_failed", "no window".into())),
        }
    }
    let want_input = opts.input_video && ffmpeg.is_some();
    if opts.input_video && ffmpeg.is_none() {
        notes.push(("input_video_failed", "ffmpeg not found".into()));
    }

    let mut info = ctx.info;
    info["session"] = json!({
        "started_us": us(started),
        "started_unix": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0),
        "started_local": local_stamp(),
        "input_video": want_input,
        "output_video": output.is_some(),
        "encoder": ffmpeg.as_ref().map(|f| f.encoder.clone()),
        "ffmpeg": ffmpeg.as_ref().map(|f| f.path.display().to_string()),
    });
    info["machine"] = machine_info();
    let recording_output = output.is_some();
    let session = Session::create(&dir, info)?;
    let _ = h.tx.send(Msg::Start(Box::new(session)));
    let hint = INPUT_HINT.lock().unwrap().clone();
    // Recording starts here so the input segment's note (made as it opens) is kept.
    h.on.store(true, Ordering::Release);
    {
        let mut s = h.state.lock().unwrap();
        s.dir = Some(dir.clone());
        s.started = Some(started);
        s.output = output;
        s.want_input = want_input;
        s.input_dropped = 0;
        s.input = None;
        if want_input && let Some((format, fps, recycle)) = hint {
            s.input = start_input(&dir, 0, format, fps, recycle);
        }
    }
    h.marks.store(recording_output, Ordering::Release);
    for (kind, detail) in notes {
        emit(Ev::Note { at: Instant::now(), kind, detail });
    }
    spawn_sampler(id);
    crate::log::line(&format!("telemetry: recording to {}", dir.display()));
    Ok(dir)
}

/// Stops the session (if one is running) and waits until its files are complete.
pub fn stop() {
    let Some(h) = HUB.get() else { return };
    if !h.on.swap(false, Ordering::AcqRel) {
        return;
    }
    h.marks.store(false, Ordering::Release);
    h.session.fetch_add(1, Ordering::AcqRel);
    let (input, output, dropped, dir) = {
        let mut s = h.state.lock().unwrap();
        (s.input.take(), s.output.take(), s.input_dropped, s.dir.take())
    };
    // The writer finishes the files once both videos are closed.
    if let Some(i) = input {
        i.finish();
    }
    if let Some(mut child) = output {
        stop_ffmpeg(&mut child);
    }
    let (done_tx, done_rx) = mpsc::channel();
    let _ = h.tx.send(Msg::Ev(Ev::Note {
        at: Instant::now(),
        kind: "input_frames_dropped",
        detail: dropped.to_string(),
    }));
    let _ = h.tx.send(Msg::Stop(done_tx));
    let _ = done_rx.recv_timeout(Duration::from_secs(10));
    if let Some(d) = dir {
        crate::log::line(&format!("telemetry: session saved in {}", d.display()));
    }
}

/// Starts or stops a session.
pub fn toggle(opts: &Options, ctx: impl FnOnce() -> SessionContext) -> Result<Option<PathBuf>> {
    if on() {
        stop();
        Ok(None)
    } else {
        start(opts, ctx()).map(Some)
    }
}

// ---------------------------------------------------------------------------------------------
// The writer.

struct Session {
    dir: PathBuf,
    info: Value,
    files: HashMap<&'static str, BufWriter<File>>,
    counts: HashMap<&'static str, u64>,
    /// Last DXGI statistics written, to skip repeats.
    last_displayed: Option<(u32, u32)>,
}

const STREAMS: &[(&str, &str)] = &[
    ("capture", "seq,t_us,copy_us,w,h,input_segment,input_frame"),
    ("frames", "seq,t_us,recv_us,kind,held"),
    (
        "pairs",
        "seq,t_us,held,steps,target,content_fps,gpu_guess_us,first_us,every_us,skipped,backlog_cleared,refresh_us",
    ),
    ("pair_gpu", "seq,gpu_us"),
    ("pair_counts", "seq,missed,textured,cost,detail,changed,lost,hidden,unexplained"),
    ("presents", "n,swap_count,kind,seq,t,due_us,call_us,done_us,arrived_us,queued"),
    ("display", "present_count,refresh,sync_refresh,sync_us"),
    (
        "system",
        "t_us,cpu_process,cpu_system,working_set_mb,private_mb,gpu_3d_process,gpu_3d_total,gpu_encode_total,\
         gpu_mem_process_mb,gpu_mem_total_mb,refresh_hz,target,content_fps",
    ),
    ("events", "t_us,kind,detail"),
];

impl Session {
    fn create(dir: &Path, info: Value) -> Result<Session> {
        let mut files = HashMap::new();
        for &(name, header) in STREAMS {
            let mut f = BufWriter::with_capacity(1 << 16, File::create(dir.join(format!("{name}.csv")))?);
            writeln!(f, "{header}")?;
            files.insert(name, f);
        }
        let s = Session { dir: dir.to_path_buf(), info, files, counts: HashMap::new(), last_displayed: None };
        s.write_info(false)?;
        Ok(s)
    }

    fn write_info(&self, finished: bool) -> Result<()> {
        let mut info = self.info.clone();
        if finished {
            info["session"]["ended_us"] = json!(us(Instant::now()));
            info["rows"] = json!(self.counts);
        }
        info["session"]["finished"] = json!(finished);
        let text = serde_json::to_string_pretty(&info)?;
        std::fs::write(self.dir.join("session.json"), text)?;
        Ok(())
    }

    fn row(&mut self, stream: &'static str, line: std::fmt::Arguments) {
        if let Some(f) = self.files.get_mut(stream) {
            let _ = f.write_fmt(line);
            let _ = f.write_all(b"\n");
            *self.counts.entry(stream).or_default() += 1;
        }
    }

    fn write(&mut self, ev: Ev) {
        let o = |t: Option<Instant>| t.map_or(String::new(), |t| us(t).to_string());
        match ev {
            Ev::Capture { seq, at, copy_us, w, h, input } => {
                let (s, f) = input.map_or((String::new(), String::new()), |(s, f)| (s.to_string(), f.to_string()));
                self.row("capture", format_args!("{seq},{},{copy_us},{w},{h},{s},{f}", us(at)));
            }
            Ev::Frame { seq, at, recv, kind, held } => {
                let k = match kind {
                    FrameKind::Passthrough => "passthrough",
                    FrameKind::Repeat => "repeat",
                    FrameKind::Pair => "pair",
                    FrameKind::Restart => "restart",
                    FrameKind::AfterStill => "after_still",
                    FrameKind::Single => "single",
                };
                self.row("frames", format_args!("{seq},{},{},{k},{held}", us(at), us(recv)));
            }
            Ev::Pair(p) => {
                self.row(
                    "pairs",
                    format_args!(
                        "{},{},{},{},{},{:.3},{},{},{},{},{},{}",
                        p.seq,
                        us(p.at),
                        p.held,
                        p.steps,
                        p.target,
                        p.content_fps,
                        p.gpu_guess.as_micros(),
                        us(p.first),
                        p.every.as_micros(),
                        p.skipped,
                        p.backlog_cleared as u8,
                        p.refresh.map_or(String::new(), |r| r.as_micros().to_string()),
                    ),
                );
            }
            Ev::PairGpu { seq, gpu_us } => self.row("pair_gpu", format_args!("{seq},{gpu_us}")),
            Ev::PairCounts { seq, counts: c } => {
                self.row(
                    "pair_counts",
                    format_args!("{seq},{},{},{},{},{},{},{},{}", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]),
                )
            }
            Ev::Present(p) => {
                self.row(
                    "presents",
                    format_args!(
                        "{},{},{},{},{:.4},{},{},{},{},{}",
                        p.n,
                        p.swap_count.map_or(String::new(), |c| c.to_string()),
                        p.kind,
                        p.seq,
                        p.t,
                        o(p.due),
                        us(p.call),
                        us(p.done),
                        o(p.arrived),
                        p.queued,
                    ),
                );
            }
            Ev::Displayed { present_count, refresh, sync_refresh, sync_qpc } => {
                if present_count == 0 || self.last_displayed == Some((present_count, sync_refresh)) {
                    return;
                }
                self.last_displayed = Some((present_count, sync_refresh));
                self.row("display", format_args!("{present_count},{refresh},{sync_refresh},{}", qpc_us(sync_qpc)));
            }
            Ev::System(s) => {
                self.row(
                    "system",
                    format_args!(
                        "{},{:.2},{:.2},{:.1},{:.1},{:.2},{:.2},{:.2},{:.1},{:.1},{},{},{:.2}",
                        o(s.at),
                        s.cpu_process,
                        s.cpu_system,
                        s.working_set_mb,
                        s.private_mb,
                        s.gpu_3d_process,
                        s.gpu_3d_total,
                        s.gpu_encode_total,
                        s.gpu_mem_process_mb,
                        s.gpu_mem_total_mb,
                        s.refresh_hz,
                        s.target,
                        s.content_fps,
                    ),
                );
            }
            Ev::Note { at, kind, detail } => {
                let detail = detail.replace('"', "'").replace(['\n', '\r'], " ");
                self.row("events", format_args!("{},{kind},\"{detail}\"", us(at)));
            }
        }
    }

    fn flush(&mut self) {
        for f in self.files.values_mut() {
            let _ = f.flush();
        }
    }
}

fn writer(rx: Receiver<Msg>) {
    let mut session: Option<Session> = None;
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(Msg::Start(s)) => {
                if let Some(mut old) = session.take() {
                    old.flush();
                    let _ = old.write_info(true);
                }
                session = Some(*s);
            }
            Ok(Msg::Ev(ev)) => {
                if let Some(s) = session.as_mut() {
                    s.write(ev);
                }
            }
            Ok(Msg::Stop(done)) => {
                if let Some(mut s) = session.take() {
                    s.flush();
                    let _ = s.write_info(true);
                }
                let _ = done.send(());
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(s) = session.as_mut() {
                    s.flush();
                }
            }
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ffmpeg.

struct Ffmpeg {
    path: PathBuf,
    /// The best H.265/H.264 encoder it has: hardware first.
    encoder: String,
}

fn ffmpeg() -> Option<&'static Ffmpeg> {
    static FF: OnceLock<Option<Ffmpeg>> = OnceLock::new();
    FF.get_or_init(|| {
        let path = std::env::var_os("PASSTHRU_FFMPEG").map(PathBuf::from).unwrap_or_else(|| "ffmpeg".into());
        let out = Command::new(&path)
            .args(["-hide_banner", "-encoders"])
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        let list = String::from_utf8_lossy(&out.stdout);
        let encoder = ["hevc_nvenc", "hevc_amf", "hevc_qsv", "h264_nvenc", "h264_amf", "h264_qsv", "libx264"]
            .into_iter()
            .find(|e| list.contains(&format!(" {e} ")))?
            .to_string();
        Some(Ffmpeg { path, encoder })
    })
    .as_ref()
}

/// Encoder arguments: near-lossless, keyframes often enough to seek to any frame quickly.
fn encode_args(encoder: &str) -> Vec<&'static str> {
    match encoder {
        e if e.ends_with("_nvenc") => vec!["-preset", "p4", "-rc", "constqp", "-qp", "16", "-g", "30"],
        e if e.ends_with("_amf") => vec!["-rc", "cqp", "-qp_i", "16", "-qp_p", "16", "-g", "30"],
        e if e.ends_with("_qsv") => vec!["-global_quality", "16", "-g", "30"],
        _ => vec!["-preset", "ultrafast", "-crf", "16", "-g", "30"],
    }
}

fn stop_ffmpeg(child: &mut Child) {
    // ffmpeg finishes the file cleanly on "q".
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(b"q");
        let _ = stdin.flush();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
}

/// Records the screen under the window's client area (the output video).
fn start_output(ff: &Ffmpeg, hwnd: isize, dir: &Path) -> Result<(Child, String)> {
    let (idx, x, y, w, h, hz) = output_region(hwnd)?;
    let src = format!("ddagrab=output_idx={idx}:framerate={hz}:offset_x={x}:offset_y={y}:video_size={w}x{h}:draw_mouse=0");
    let mut cmd = Command::new(&ff.path);
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", &src]);
    if !ff.encoder.ends_with("_nvenc") {
        // Other encoders take frames in system memory.
        cmd.args(["-vf", "hwdownload,format=bgra"]);
    }
    cmd.args(["-c:v", &ff.encoder]).args(encode_args(&ff.encoder));
    cmd.arg(dir.join("output.mkv"));
    let child = cmd
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(File::create(dir.join("output_ffmpeg.log")).map(Stdio::from).unwrap_or(Stdio::null()))
        .spawn()
        .context("starting ffmpeg for the output video")?;
    Ok((child, format!("output {idx} at {x},{y} {w}x{h} @ {hz} Hz")))
}

/// The DXGI output (as ffmpeg's ddagrab numbers them: the first adapter's outputs) showing the
/// window, the client area relative to it (clipped, even sizes), and the display's rate.
fn output_region(hwnd: isize) -> Result<(u32, i32, i32, i32, i32, u32)> {
    use windows::Win32::Foundation::{HWND, POINT, RECT};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    use windows::Win32::Graphics::Gdi::{ClientToScreen, MONITOR_DEFAULTTONEAREST, MonitorFromWindow};
    use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
    let hwnd = HWND(hwnd as *mut _);
    let mut r = RECT::default();
    let mut origin = POINT::default();
    let monitor = unsafe {
        GetClientRect(hwnd, &mut r)?;
        let _ = ClientToScreen(hwnd, &mut origin);
        MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    };
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let adapter = unsafe { factory.EnumAdapters1(0)? };
    for i in 0.. {
        let Ok(output) = (unsafe { adapter.EnumOutputs(i) }) else { break };
        let desc = unsafe { output.GetDesc()? };
        if desc.Monitor != monitor {
            continue;
        }
        let d = desc.DesktopCoordinates;
        let x0 = (origin.x).max(d.left);
        let y0 = (origin.y).max(d.top);
        let x1 = (origin.x + r.right).min(d.right);
        let y1 = (origin.y + r.bottom).min(d.bottom);
        let (w, h) = ((x1 - x0) & !1, (y1 - y0) & !1);
        if w < 16 || h < 16 {
            bail!("the window isn't on screen");
        }
        let hz = crate::present::display_hz().unwrap_or(60).max(1);
        return Ok((i, x0 - d.left, y0 - d.top, w, h, hz));
    }
    bail!("the window's display isn't on the first GPU (ffmpeg can't capture it)")
}

/// Gives a frame buffer back once the input video is done with it.
pub type Recycle = Arc<dyn Fn(Vec<u8>) + Send + Sync>;

/// The card's frames, encoded as they come (one video frame per card frame). Takes the
/// presenter's buffers once it's done with them, so recording costs no extra copy.
struct InputRec {
    tx: SyncSender<Vec<u8>>,
    format: (u32, u32, Layout),
    segment: u32,
    frames: u32,
    done: std::thread::JoinHandle<()>,
}

impl InputRec {
    fn start(
        ff: &Ffmpeg,
        dir: &Path,
        segment: u32,
        (w, h, layout): (u32, u32, Layout),
        fps: f32,
        recycle: Recycle,
    ) -> Result<InputRec> {
        let pix = match layout {
            Layout::Nv12 => "nv12",
            Layout::Yuy2 => "yuyv422",
            Layout::Rgb32 => "bgra",
        };
        let name = format!("input_{segment:03}.mkv");
        let rate = format!("{:.3}", fps.max(1.0));
        let size = format!("{w}x{h}");
        let mut cmd = Command::new(&ff.path);
        cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix])
            .args(["-s", &size, "-framerate", &rate, "-i", "-"]);
        // Hardware encoders want 4:2:0.
        if layout != Layout::Nv12 {
            cmd.args(["-pix_fmt", "yuv420p"]);
        }
        cmd.args(["-c:v", &ff.encoder]).args(encode_args(&ff.encoder)).arg(dir.join(&name));
        let log = File::create(dir.join(format!("input_{segment:03}_ffmpeg.log")));
        let mut child = cmd
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(log.map(Stdio::from).unwrap_or(Stdio::null()))
            .spawn()
            .context("starting ffmpeg for the input video")?;
        let stdin: ChildStdin = child.stdin.take().context("no stdin")?;
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(INPUT_QUEUE);
        let done = std::thread::Builder::new()
            .name("input-video".into())
            .spawn(move || {
                let mut stdin = stdin;
                let mut broken = false;
                while let Ok(buf) = rx.recv() {
                    if !broken && stdin.write_all(&buf).is_err() {
                        broken = true;
                    }
                    recycle(buf);
                }
                drop(stdin);
                let _ = child.wait();
            })
            .context("spawn input encoder")?;
        Ok(InputRec { tx, format: (w, h, layout), segment, frames: 0, done })
    }

    /// Closes the segment and waits for the encoder to finish it.
    fn finish(self) {
        drop(self.tx);
        let _ = self.done.join();
    }
}

/// Hands a card frame (that the presenter is done with) to the input video. Returns where it
/// went (segment, frame number in it), or the buffer back if it wasn't taken.
pub fn input_frame(
    data: Vec<u8>,
    (w, h, layout): (u32, u32, Layout),
    fps: f32,
    recycle: impl FnOnce() -> Recycle,
) -> std::result::Result<(u32, u32), Vec<u8>> {
    let Some(hub) = HUB.get() else { return Err(data) };
    if !hub.on.load(Ordering::Relaxed) {
        return Err(data);
    }
    let mut s = hub.state.lock().unwrap();
    if !s.want_input {
        return Err(data);
    }
    let Some(dir) = s.dir.clone() else { return Err(data) };
    let format = (w, h, layout);
    if s.input.as_ref().is_some_and(|i| i.format != format) {
        let old = s.input.take().expect("checked");
        let segment = old.segment + 1;
        std::thread::spawn(move || old.finish());
        s.input = start_input(&dir, segment, format, fps, recycle());
        if s.input.is_none() {
            s.want_input = false;
        }
    } else if s.input.is_none() {
        s.input = start_input(&dir, 0, format, fps, recycle());
        if s.input.is_none() {
            s.want_input = false;
        }
    }
    let Some(rec) = s.input.as_mut() else { return Err(data) };
    match rec.tx.try_send(data) {
        Ok(()) => {
            let at = (rec.segment, rec.frames);
            rec.frames += 1;
            Ok(at)
        }
        Err(TrySendError::Full(data)) | Err(TrySendError::Disconnected(data)) => {
            s.input_dropped += 1;
            Err(data)
        }
    }
}

/// The card's current frame format and how to give buffers back, kept up to date by the
/// presenter so a session's input video can start before its first frame (an encoder takes a
/// few hundred milliseconds to open; frames meanwhile would be dropped).
static INPUT_HINT: Mutex<Option<((u32, u32, Layout), f32, Recycle)>> = Mutex::new(None);

pub fn input_format(format: (u32, u32, Layout), fps: f32, recycle: impl FnOnce() -> Recycle) {
    let mut hint = INPUT_HINT.lock().unwrap();
    if hint.as_ref().is_none_or(|h| (h.0, h.1) != (format, fps)) {
        *hint = Some((format, fps, recycle()));
    }
}

fn start_input(dir: &Path, segment: u32, format: (u32, u32, Layout), fps: f32, recycle: Recycle) -> Option<InputRec> {
    let ff = ffmpeg()?;
    match InputRec::start(ff, dir, segment, format, fps, recycle) {
        Ok(r) => {
            let (w, h, layout) = format;
            note("input_segment", || format!("input_{segment:03}.mkv {w}x{h} {layout:?} @ {fps:.2}"));
            Some(r)
        }
        Err(e) => {
            note("input_video_failed", || format!("{e:#}"));
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------
// System samples.

fn spawn_sampler(id: u64) {
    let _ = std::thread::Builder::new().name("telemetry-sampler".into()).spawn(move || {
        let mut cpu = CpuMeter::new();
        let mut gpu = GpuMeter::new();
        while HUB.get().is_some_and(|h| h.session.load(Ordering::Acquire) == id && h.on.load(Ordering::Acquire)) {
            std::thread::sleep(SAMPLE_EVERY);
            let (cpu_process, cpu_system) = cpu.sample();
            let (working_set_mb, private_mb) = memory_mb();
            let g = gpu.as_mut().map(|g| g.sample()).unwrap_or_default();
            let shared = crate::present::shared_numbers();
            emit(Ev::System(SystemSample {
                at: Some(Instant::now()),
                cpu_process,
                cpu_system,
                working_set_mb,
                private_mb,
                gpu_3d_process: g.0,
                gpu_3d_total: g.1,
                gpu_encode_total: g.2,
                gpu_mem_process_mb: g.3,
                gpu_mem_total_mb: g.4,
                refresh_hz: shared.0,
                target: shared.1,
                content_fps: shared.2,
            }));
        }
    });
}

/// Process and system CPU use, as a share of the whole machine (all cores), in percent.
struct CpuMeter {
    last: Option<(u64, u64, u64)>,
}

impl CpuMeter {
    fn new() -> Self {
        CpuMeter { last: None }
    }

    fn now() -> Option<(u64, u64, u64)> {
        use windows::Win32::Foundation::FILETIME;
        use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes, GetSystemTimes};
        let ft = |f: FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
        let (mut c, mut e, mut k, mut u) = Default::default();
        let (mut idle, mut sk, mut su) = Default::default();
        unsafe {
            GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u).ok()?;
            GetSystemTimes(Some(&mut idle), Some(&mut sk), Some(&mut su)).ok()?;
        }
        // System kernel time includes idle time.
        Some((ft(k) + ft(u), ft(sk) + ft(su), ft(idle)))
    }

    fn sample(&mut self) -> (f32, f32) {
        let Some(now) = Self::now() else { return (0.0, 0.0) };
        let out = match self.last {
            Some(prev) => {
                let total = now.1.saturating_sub(prev.1).max(1) as f64;
                let idle = now.2.saturating_sub(prev.2) as f64;
                let proc_ = now.0.saturating_sub(prev.0) as f64;
                ((proc_ / total * 100.0) as f32, ((total - idle) / total * 100.0) as f32)
            }
            None => (0.0, 0.0),
        };
        self.last = Some(now);
        out
    }
}

fn memory_mb() -> (f32, f32) {
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows::Win32::System::Threading::GetCurrentProcess;
    let mut m = PROCESS_MEMORY_COUNTERS_EX { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32, ..Default::default() };
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut m as *mut _ as *mut _, m.cb).is_ok() };
    if !ok {
        return (0.0, 0.0);
    }
    let mb = |b: usize| b as f32 / (1024.0 * 1024.0);
    (mb(m.WorkingSetSize), mb(m.PrivateUsage))
}

use windows::Win32::System::Performance::{PDH_HCOUNTER, PDH_HQUERY};

/// GPU load and memory from Windows' performance counters (the numbers Task Manager shows).
struct GpuMeter {
    query: PDH_HQUERY,
    engine: PDH_HCOUNTER,
    memory_process: PDH_HCOUNTER,
    memory_total: PDH_HCOUNTER,
    pid_tag: String,
}

impl GpuMeter {
    fn new() -> Option<GpuMeter> {
        use windows::Win32::System::Performance::{PdhAddEnglishCounterW, PdhCollectQueryData, PdhOpenQueryW};
        use windows::core::HSTRING;
        let mut query = PDH_HQUERY::default();
        unsafe {
            if PdhOpenQueryW(None, 0, &mut query) != 0 {
                return None;
            }
        }
        let add = |path: &str| -> Option<PDH_HCOUNTER> {
            let mut c = PDH_HCOUNTER::default();
            let r = unsafe { PdhAddEnglishCounterW(query, &HSTRING::from(path), 0, &mut c) };
            (r == 0).then_some(c)
        };
        let engine = add(r"\GPU Engine(*)\Utilization Percentage")?;
        let memory_process = add(r"\GPU Process Memory(*)\Dedicated Usage")?;
        let memory_total = add(r"\GPU Adapter Memory(*)\Dedicated Usage")?;
        unsafe {
            let _ = PdhCollectQueryData(query);
        }
        Some(GpuMeter {
            query,
            engine,
            memory_process,
            memory_total,
            pid_tag: format!("pid_{}_", std::process::id()),
        })
    }

    /// Every instance of a wildcard counter: (instance name, value).
    fn values(&self, counter: PDH_HCOUNTER) -> Vec<(String, f64)> {
        use windows::Win32::System::Performance::{
            PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_MORE_DATA, PdhGetFormattedCounterArrayW,
        };
        let (mut bytes, mut count) = (0u32, 0u32);
        unsafe {
            let r = PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut bytes, &mut count, None);
            if r != PDH_MORE_DATA || bytes == 0 {
                return Vec::new();
            }
            let mut buf = vec![0u8; bytes as usize];
            let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            if PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut bytes, &mut count, Some(items)) != 0 {
                return Vec::new();
            }
            (0..count as usize)
                .map(|i| {
                    let it = &*items.add(i);
                    let name = it.szName.to_string().unwrap_or_default();
                    (name, it.FmtValue.Anonymous.doubleValue)
                })
                .collect()
        }
    }

    /// 3D load of this process and of everything, video-encode load, this process's and all
    /// dedicated GPU memory (MB).
    fn sample(&mut self) -> (f32, f32, f32, f32, f32) {
        use windows::Win32::System::Performance::PdhCollectQueryData;
        unsafe {
            if PdhCollectQueryData(self.query) != 0 {
                return Default::default();
            }
        }
        let (mut mine3d, mut all3d) = (0.0, 0.0);
        let mut encode: HashMap<String, f64> = HashMap::new();
        let mut per_engine: HashMap<String, f64> = HashMap::new();
        for (name, v) in self.values(self.engine) {
            // Instances: pid_1234_luid_..._phys_0_eng_3_engtype_3D
            let engine = name.split("_luid_").nth(1).unwrap_or(&name).to_string();
            if name.ends_with("engtype_3D") {
                *per_engine.entry(engine.clone()).or_default() += v;
                if name.starts_with(&self.pid_tag) {
                    mine3d += v;
                }
            } else if name.contains("engtype_VideoEncode") {
                *encode.entry(engine).or_default() += v;
            }
        }
        for v in per_engine.values() {
            all3d = f64::max(all3d, *v);
        }
        let enc = encode.values().copied().fold(0.0, f64::max);
        let mb = 1024.0 * 1024.0;
        let mine_mem: f64 =
            self.values(self.memory_process).iter().filter(|(n, _)| n.starts_with(&self.pid_tag)).map(|(_, v)| v).sum();
        let all_mem = self.values(self.memory_total).iter().map(|(_, v)| *v).fold(0.0, f64::max);
        (
            mine3d.min(100.0) as f32,
            all3d.min(100.0) as f32,
            enc.min(100.0) as f32,
            (mine_mem / mb) as f32,
            (all_mem / mb) as f32,
        )
    }
}

impl Drop for GpuMeter {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::System::Performance::PdhCloseQuery(self.query);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Machine description and names.

fn machine_info() -> Value {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    let mut gpus = Vec::new();
    if let Ok(f) = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() } {
        for i in 0.. {
            let Ok(a) = (unsafe { f.EnumAdapters1(i) }) else { break };
            if let Ok(d) = unsafe { a.GetDesc1() } {
                let name = String::from_utf16_lossy(&d.Description).trim_end_matches('\0').to_string();
                gpus.push(json!({ "name": name, "vram_mb": d.DedicatedVideoMemory / (1024 * 1024) }));
            }
        }
    }
    json!({
        "cpu": std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default(),
        "logical_cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "os": os_version(),
        "gpus": gpus,
    })
}

fn os_version() -> String {
    use windows::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW};
    use windows::core::{HSTRING, PCWSTR};
    let read = |name: &str| -> Option<String> {
        let mut buf = [0u16; 128];
        let mut len = (buf.len() * 2) as u32;
        let key = HSTRING::from(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion");
        let r = unsafe {
            RegGetValueW(
                HKEY_LOCAL_MACHINE,
                PCWSTR(key.as_ptr()),
                PCWSTR(HSTRING::from(name).as_ptr()),
                RRF_RT_REG_SZ,
                None,
                Some(buf.as_mut_ptr() as *mut _),
                Some(&mut len),
            )
        };
        r.is_ok().then(|| String::from_utf16_lossy(&buf[..(len as usize / 2).saturating_sub(1)]))
    };
    format!(
        "{} {} (build {})",
        read("ProductName").unwrap_or_default(),
        read("DisplayVersion").unwrap_or_default(),
        read("CurrentBuildNumber").unwrap_or_default()
    )
}

/// Local date and time for folder names: 2026-11-21_17-55-03.
fn local_stamp() -> String {
    use windows::Win32::System::SystemInformation::GetLocalTime;
    let t = unsafe { GetLocalTime() };
    format!("{:04}-{:02}-{:02}_{:02}-{:02}-{:02}", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sessions share the process-wide hub: one test at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn session(dir: &Path, input_video: bool) -> PathBuf {
        let _ = std::fs::remove_dir_all(dir);
        let opts = Options { dir: Some(dir.to_path_buf()), input_video, output_video: false };
        start(&opts, SessionContext { hwnd: None, info: json!({ "test": true }) }).expect("start")
    }

    fn rows(dir: &Path, stream: &str) -> Vec<String> {
        let text = std::fs::read_to_string(dir.join(format!("{stream}.csv"))).unwrap();
        text.lines().skip(1).map(str::to_string).collect()
    }

    // One test runs the whole lifecycle: sessions share the process-wide hub.
    #[test]
    fn a_session_records_every_stream_and_samples_the_system() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join("passthru-telemetry-test");
        session(&dir, false);
        assert!(on());
        let t = Instant::now();
        emit(Ev::Capture { seq: 1, at: t, copy_us: 900, w: 64, h: 32, input: None });
        emit(Ev::Frame { seq: 1, at: t, recv: t, kind: FrameKind::Pair, held: 2 });
        emit(Ev::PairGpu { seq: 1, gpu_us: 2500 });
        emit(Ev::PairCounts { seq: 1, counts: [1, 2, 3, 4, 5, 6, 7, 8] });
        emit(Ev::Displayed { present_count: 7, refresh: 100, sync_refresh: 101, sync_qpc: 0 });
        emit(Ev::Displayed { present_count: 7, refresh: 100, sync_refresh: 101, sync_qpc: 0 });
        note("bookmark", || "1".into());
        note("weird", || "a \"quoted\"\nline".into());
        // Windows' GPU counters take about a second to set up before the first sample.
        std::thread::sleep(Duration::from_millis(2000));
        stop();
        assert!(!on());
        assert_eq!(rows(&dir, "capture"), ["1,".to_string() + &us(t).to_string() + ",900,64,32,,"]);
        assert_eq!(rows(&dir, "frames").len(), 1);
        assert!(rows(&dir, "frames")[0].ends_with(",pair,2"));
        assert_eq!(rows(&dir, "pair_counts"), ["1,1,2,3,4,5,6,7,8"]);
        assert_eq!(rows(&dir, "display").len(), 1, "repeated statistics are written once");
        let events = rows(&dir, "events");
        assert!(events.iter().any(|e| e.contains(",bookmark,\"1\"")));
        assert!(events.iter().any(|e| e.contains("a 'quoted' line")), "{events:?}");
        let samples = rows(&dir, "system");
        assert!(samples.len() >= 2, "sampled every 250 ms: {samples:?}");
        let info: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("session.json")).unwrap()).unwrap();
        assert_eq!(info["session"]["finished"], json!(true));
        assert!(info["session"]["ended_us"].as_i64().unwrap() > info["session"]["started_us"].as_i64().unwrap());
        assert!(info["machine"]["logical_cores"].as_u64().unwrap() > 0);
        // Not recording: nothing is written.
        emit(Ev::PairGpu { seq: 2, gpu_us: 1 });
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(rows(&dir, "pair_gpu"), ["1,2500"]);
    }

    /// Needs ffmpeg on PATH: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn the_input_video_has_one_frame_per_card_frame() {
        let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join("passthru-telemetry-video-test");
        let free: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
        let back = free.clone();
        input_format((256, 144, Layout::Nv12), 60.0, || Arc::new(move |b| back.lock().unwrap().push(b)));
        session(&dir, true);
        let mut taken = 0;
        for i in 0..30u32 {
            // Hardware encoders have a minimum size.
            let frame = vec![(i * 8) as u8; 256 * 144 * 3 / 2];
            let back = free.clone();
            let recycle = move || -> Recycle { Arc::new(move |b| back.lock().unwrap().push(b)) };
            if input_frame(frame, (256, 144, Layout::Nv12), 60.0, recycle).is_ok() {
                taken += 1;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        stop();
        assert_eq!(taken, 30, "the encoder opens with the session, so none are dropped");
        let out = Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"])
            .arg(dir.join("input_000.mkv"))
            .output()
            .unwrap();
        let n: u32 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap();
        assert_eq!(n, taken);
        assert_eq!(free.lock().unwrap().len(), taken as usize, "every buffer comes back");
    }
}
