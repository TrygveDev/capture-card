#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod audio;
mod convert;
mod gpu;
mod gputest;
mod install;
mod instance;
mod log;
mod meter;
mod nvof;
mod overlay;
mod present;
mod settings;
mod spring;
mod telemetry;
mod update;
mod upscale;
mod video;
mod winutil;

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use gpui::{
    App, AppContext, Application, Bounds, Context, FocusHandle, Focusable, FontWeight, Hsla,
    ImageSource, InteractiveElement, IntoElement, KeyDownEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, ObjectFit, ParentElement, Pixels, Render, RenderImage, ScrollWheelEvent,
    SharedString, StatefulInteractiveElement, Styled, StyledImage, TitlebarOptions, Window,
    WindowBackgroundAppearance, WindowBounds, WindowOptions, canvas, div, img,
    prelude::FluentBuilder, point, px, relative, rgb, rgba, size,
};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
use windows::core::HSTRING;

use crate::audio::AudioShared;
use crate::settings::{Overlay, Settings};
use crate::video::{VideoOptions, VideoShared};

/// Controls hide after the pointer has been still this long.
const IDLE_HIDE: Duration = Duration::from_millis(2000);
/// No frame for this long counts as "no signal".
const SIGNAL_TIMEOUT: Duration = Duration::from_millis(1000);
/// The FPS counter's numbers change this often (readable); its graphs move every tick.
const NUMBERS_EVERY: Duration = Duration::from_millis(250);
/// How often the pointer is checked for hover / idle.
const TICK: Duration = Duration::from_millis(30);

/// How far the controls slide to be out of view below the window (their height, their gap to
/// the bottom, and their shadow).
const BAR_TRAVEL: f32 = 76.0;

/// "Close when the signal is lost" waits this long: a console switching resolution or HDR
/// drops the signal for a few seconds.
const CLOSE_AFTER_LOSS: Duration = Duration::from_secs(10);
/// A Passthru left running in the tray looks for updates this often (the check itself is
/// throttled to about daily).
const UPDATE_EVERY: Duration = Duration::from_secs(6 * 3600);

/// Messages top right stay this long.
const TOAST_FOR: Duration = Duration::from_millis(3000);

const ICON_FONT: &str = "Segoe Fluent Icons";
const ICON_VOLUME: &str = "\u{E995}";
const ICON_VOLUME_LOW: &str = "\u{E993}";
const ICON_MUTE: &str = "\u{E74F}";
const ICON_FULLSCREEN: &str = "\u{E740}";
const ICON_EXIT_FULLSCREEN: &str = "\u{E73F}";
const ICON_SETTINGS: &str = "\u{E713}";
const ICON_CLOSE: &str = "\u{E8BB}";
const ICON_CHECK: &str = "\u{E73E}";
const ICON_CHEVRON_UP: &str = "\u{E70E}";

pub(crate) fn raise_thread_priority(task: &str) {
    let mut index = 0u32;
    unsafe {
        let _ = AvSetMmThreadCharacteristicsW(&HSTRING::from(task), &mut index);
    }
}

struct Viewer {
    video: Arc<VideoShared>,
    audio: Arc<AudioShared>,
    settings: Settings,
    focus: FocusHandle,
    hwnd: Option<HWND>,
    shown: Option<Arc<RenderImage>>,
    aspect: f32,
    had_signal: bool,
    /// Whether the GPU layer was drawing the video at the last render.
    gpu_drawing: bool,

    /// Pointer position in client pixels while it's over the window.
    cursor: Option<(i32, i32)>,
    last_move: Instant,
    controls_visible: bool,
    cursor_hidden: bool,
    dragging_volume: bool,
    /// The menu open above the controls, if any.
    menu_open: Option<Menu>,
    /// Debug (`--show-menu`): the controls and menu stay up.
    show_menu: bool,
    /// What the controls last showed: mode, frame size, modes offered, scaling, range.
    #[allow(clippy::type_complexity)]
    controls_state: Option<(String, (u32, u32), Vec<video::ModeKey>, present::Scaling, (Option<bool>, bool))>,
    /// Layout bounds of the controls (the pill, and the menu while it's open) and the volume
    /// track, in logical pixels.
    pill_bounds: Rc<Cell<Option<Bounds<Pixels>>>>,
    slider_bounds: Rc<Cell<Option<Bounds<Pixels>>>>,
    stats: Stats,
    /// The FPS counter: its numbers, when they were read, and its graphs (graph mode).
    numbers: meter::Numbers,
    numbers_at: Option<Instant>,
    graphs: Option<(meter::Graph, Option<meter::Graph>)>,
    /// Test sessions: what they record, one waiting for the video to start (`--record`),
    /// bookmarks dropped, the REC badge's last shown second, and a short message.
    record: telemetry::Options,
    record_pending: bool,
    /// The controls' slide: 0 in place, 1 out of view below; whether it's moving, when it was
    /// last stepped, and whether Windows wants animations at all.
    /// Updates from GitHub, and the state last shown (a notice when one is ready).
    updater: std::sync::Arc<update::Updater>,
    update_seen: update::State,
    bar: spring::Spring,
    bar_moving: bool,
    bar_at: Instant,
    animate: bool,
    bookmarks: u32,
    rec_second: Option<u64>,
    toast: Option<(String, Instant)>,
    /// When the update check last started (one left running in the tray checks again).
    updater_at: Instant,
    /// The window is on screen (not waiting in the tray), and has been at least once.
    window_shown: bool,
    placed: bool,
    /// The main window is fullscreen (the controls' window can't tell).
    fullscreen: bool,
    /// The controls' window has something on it (else it needn't redraw).
    controls_drawn: bool,
    /// To the main window: requests from the controls (see `request`).
    requests: futures::channel::mpsc::UnboundedSender<winutil::Event>,
    /// The signal as `tick` last saw it, and when it went.
    signal_on: bool,
    signal_lost_at: Option<Instant>,
}

/// The controls' window: draws what `Viewer::render_controls` gives.
struct ControlsView {
    viewer: gpui::Entity<Viewer>,
}

impl ControlsView {
    fn new(viewer: gpui::Entity<Viewer>, cx: &mut Context<Self>) -> Self {
        // Redraw with the viewer, while there's something to draw (or to take away).
        cx.observe(&viewer, |_, viewer, cx| {
            let v = viewer.read(cx);
            if v.controls_visible || v.controls_drawn || v.toast.is_some() || telemetry::status().is_some() {
                cx.notify();
            }
        })
        .detach();
        ControlsView { viewer }
    }
}

impl Render for ControlsView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.viewer.update(cx, |v, cx| v.render_controls(window, cx))
    }
}

/// The menus above the controls.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Menu {
    /// Capture mode, colour range, upscaling.
    Mode,
    /// The window, the background, updates.
    Settings,
}

/// Frame-rate and latency counters, printed every few seconds with `--stats` (always in dev
/// builds), and as a summary on exit with `--exit-after`.
struct Stats {
    enabled: bool,
    exit_after: Option<Duration>,
    /// Debug: save the window's on-screen pixels to this file after this long, once.
    capture: Option<(Duration, std::path::PathBuf)>,
    started: Instant,
    window: StatsWindow,
    /// Whole run, minus the first two seconds of warm-up.
    total: Option<StatsWindow>,
}

#[derive(Clone, Copy)]
struct StatsWindow {
    since: Instant,
    produced_at_start: u64,
    convert_ns_at_start: u64,
    shown: u64,
    latency_sum: Duration,
    latency_max: Duration,
    /// Longest gap between two shown frames.
    gap_max: Duration,
    last_shown: Option<Instant>,
}

impl StatsWindow {
    fn new(video: &VideoShared) -> Self {
        StatsWindow {
            since: Instant::now(),
            produced_at_start: video.produced.load(Ordering::Relaxed),
            convert_ns_at_start: video.convert_ns.load(Ordering::Relaxed),
            shown: 0,
            latency_sum: Duration::ZERO,
            latency_max: Duration::ZERO,
            gap_max: Duration::ZERO,
            last_shown: None,
        }
    }

    fn record(&mut self, ready_at: Instant, now: Instant) {
        let latency = now.saturating_duration_since(ready_at);
        self.shown += 1;
        self.latency_sum += latency;
        self.latency_max = self.latency_max.max(latency);
        if let Some(prev) = self.last_shown {
            self.gap_max = self.gap_max.max(now - prev);
        }
        self.last_shown = Some(now);
    }

    fn report(&self, label: &str, video: &VideoShared) -> String {
        let secs = self.since.elapsed().as_secs_f64().max(1e-6);
        let produced = video.produced.load(Ordering::Relaxed) - self.produced_at_start;
        let convert_ms = (video.convert_ns.load(Ordering::Relaxed) - self.convert_ns_at_start) as f64
            / 1e6
            / produced.max(1) as f64;
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        format!(
            "{label}: source {:.2} fps, shown {:.2} fps ({} of {} frames, {} replaced before shown), \
             convert {:.2} ms, ready->drawn avg {:.2} ms max {:.2} ms, longest gap {:.1} ms",
            produced as f64 / secs,
            self.shown as f64 / secs,
            self.shown,
            produced,
            produced.saturating_sub(self.shown),
            convert_ms,
            ms(self.latency_sum) / self.shown.max(1) as f64,
            ms(self.latency_max),
            ms(self.gap_max),
        )
    }
}

impl Focusable for Viewer {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Viewer {
    fn has_signal(&self) -> bool {
        self.video
            .last_frame
            .lock()
            .unwrap()
            .is_some_and(|t| t.elapsed() < SIGNAL_TIMEOUT)
    }

    /// The pill in client pixels, if it's on screen.
    fn pill_rect(&self, scale: f32) -> Option<(i32, i32, i32, i32)> {
        let b = self.pill_bounds.get()?;
        let s = |p: Pixels| (f32::from(p) * scale).round() as i32;
        Some((s(b.left()), s(b.top()), s(b.right()), s(b.bottom())))
    }

    /// Polls the pointer: shows the controls while it moves over the window, hides them (and
    /// the cursor, in fullscreen) once it's been still a while.
    /// More than a few seconds since start (the GPU path may never come: `--cpu`).
    fn started_long_ago(&self) -> bool {
        self.stats.started.elapsed() > Duration::from_secs(5)
    }

    fn tick_stats(&mut self, cx: &mut Context<Self>) {
        // On the GPU path the presenter keeps (and prints) the numbers; these are for the CPU path.
        let gpu = self.video.gpu.ready.load(Ordering::Acquire);
        let s = &mut self.stats;
        if s.total.is_none() && s.started.elapsed() >= Duration::from_secs(2) {
            s.total = Some(StatsWindow::new(&self.video));
        }
        if s.window.since.elapsed() >= Duration::from_secs(5) {
            if s.enabled && !gpu {
                eprintln!("{}", s.window.report("5s", &self.video));
            }
            s.window = StatsWindow::new(&self.video);
        }
        if let Some((after, _)) = &s.capture
            && s.started.elapsed() >= *after
            && let Some((_, path)) = s.capture.take()
            && let Some(h) = self.hwnd
        {
            match winutil::capture(h, &path) {
                Ok(()) => eprintln!("captured the window to {}", path.display()),
                Err(e) => eprintln!("capture: {e:#}"),
            }
        }
        let s = &mut self.stats;
        if let Some(limit) = s.exit_after
            && s.started.elapsed() >= limit
        {
            if gpu {
                eprintln!("{}", self.video.gpu.summary.lock().unwrap());
            } else if let Some(total) = &s.total {
                eprintln!("{}", total.report("TOTAL", &self.video));
            }
            telemetry::stop();
            cx.quit();
        }
    }

    /// Reads the FPS counter: its numbers a few times a second, its graphs every tick.
    fn tick_overlay(&mut self, cx: &mut Context<Self>) {
        let mode = self.settings.overlay;
        if mode == Overlay::Off || !self.had_signal {
            return;
        }
        let now = Instant::now();
        let mut changed = mode == Overlay::Graph;
        if self.numbers_at.is_none_or(|t| now.duration_since(t) >= NUMBERS_EVERY) {
            self.numbers = self.video.meter.numbers(now);
            self.numbers_at = Some(now);
            changed = true;
        }
        if mode == Overlay::Graph {
            self.graphs = Some(self.video.meter.graphs(now));
        }
        if changed {
            cx.notify();
        }
    }

    fn tick(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.tick_stats(cx);
        self.tick_overlay(cx);
        let update = self.updater.state();
        if update != self.update_seen {
            if let update::State::Ready(v) = &update {
                self.toast(format!("Passthru {v} is ready: it starts next time (or restart from Settings)"));
            }
            self.update_seen = update;
            cx.notify();
        }
        if self.settings.auto_update
            && self.updater_at.elapsed() >= UPDATE_EVERY
            && matches!(self.update_seen, update::State::UpToDate | update::State::Failed(_))
        {
            self.updater = update::spawn(true, false);
            self.updater_at = Instant::now();
        }
        // `--record`: once the video is drawing (the session records what it shows).
        if self.record_pending && (self.video.gpu.ready.load(Ordering::Acquire) || self.started_long_ago()) {
            self.record_pending = false;
            self.toggle_recording(window, cx);
        }
        let second = telemetry::status().map(|(_, t)| t.as_secs());
        let toast_live = self.toast.as_ref().is_some_and(|(_, at)| at.elapsed() < TOAST_FOR + TICK);
        if second != self.rec_second || toast_live {
            self.rec_second = second;
            cx.notify();
        }
        self.tick_signal(window, cx);
        if !self.window_shown {
            return;
        }
        if window.is_fullscreen() != self.fullscreen {
            self.fullscreen = !self.fullscreen;
            cx.notify();
        }
        let Some(hwnd) = self.hwnd else { return };
        let scale = window.scale_factor();
        let cursor = winutil::cursor_in_client(hwnd);
        if cursor != self.cursor {
            if cursor.is_some() {
                self.last_move = Instant::now();
            }
            self.cursor = cursor;
        }

        if self.dragging_volume {
            if winutil::left_button_down() {
                if let Some((x, _)) = cursor {
                    self.volume_from_x(px(x as f32 / scale), cx);
                }
            } else {
                self.dragging_volume = false;
                self.settings.save();
            }
        }

        let over_pill = match (cursor, self.controls_visible.then(|| self.pill_rect(scale)).flatten()) {
            (Some((x, y)), Some((l, t, r, b))) => x >= l && x < r && y >= t && y < b,
            _ => false,
        };
        let visible = self.dragging_volume
            || self.show_menu
            || (cursor.is_some() && (over_pill || self.menu_open.is_some() || self.last_move.elapsed() < IDLE_HIDE));
        if visible != self.controls_visible {
            self.controls_visible = visible;
            if !visible {
                self.pill_bounds.set(None);
                self.menu_open = None;
            }
            cx.notify();
        }
        winutil::set_controls_rect(if visible { self.pill_rect(scale) } else { None });

        let hide_cursor = window.is_fullscreen() && cursor.is_some() && !visible;
        if hide_cursor != self.cursor_hidden {
            self.cursor_hidden = hide_cursor;
            winutil::set_cursor_hidden(hide_cursor);
        }

        if self.had_signal != self.has_signal() || self.gpu_drawing != self.video.gpu.ready.load(Ordering::Acquire) {
            cx.notify();
        }
        // The mode, its size, what's offered and how it's scaled can change with no frame
        // reaching this layer (the GPU layer draws them): redraw the controls when they do.
        if self.controls_visible {
            let state = (
                self.video.mode.lock().unwrap().clone(),
                self.video.control.size(),
                self.video.control.modes.lock().unwrap().clone(),
                self.video.gpu.scaling(),
                (self.video.control.detected_full(), self.video.control.full_now.load(Ordering::Relaxed)),
            );
            if self.controls_state.as_ref() != Some(&state) {
                self.controls_state = Some(state);
                cx.notify();
            }
        }
    }

    /// Opens the window when a signal arrives and closes it once the signal has been gone a
    /// while, as the settings menu says.
    fn tick_signal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let signal = self.has_signal();
        if signal != self.signal_on {
            self.signal_on = signal;
            self.signal_lost_at = (!signal).then(Instant::now);
            if signal && !self.window_shown && self.settings.open_on_signal {
                log::line("signal: arrived, opening the window");
                self.show_window(window, cx);
            }
        }
        if self.window_shown
            && self.settings.close_on_signal_loss
            && self.signal_lost_at.is_some_and(|t| t.elapsed() >= CLOSE_AFTER_LOSS)
        {
            self.signal_lost_at = None;
            log::line("signal: gone, closing the window");
            self.close(window, cx);
        }
    }

    /// Shows the tray icon while Passthru can wait there (or is waiting there).
    fn sync_tray(&self) {
        if let Some(h) = self.hwnd {
            winutil::set_tray(h, self.settings.stays_in_tray() || !self.window_shown);
        }
    }

    /// Brings the window up (from the tray, or to the front).
    fn show_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(h) = self.hwnd else { return };
        let was_hidden = !self.window_shown;
        self.window_shown = true;
        self.video.gpu.paused.store(false, Ordering::Relaxed);
        self.audio.muted.store(self.settings.muted, Ordering::Relaxed);
        // Started hidden, the window's first showing is gpui's (activate_window puts it where
        // it was last); after that it's shown here.
        if std::mem::replace(&mut self.placed, true) {
            winutil::set_shown(h, true);
        }
        if was_hidden {
            self.signal_lost_at = None;
            if self.settings.start_fullscreen && !window.is_fullscreen() {
                window.toggle_fullscreen();
            }
        }
        // To the front, and the keyboard (Windows is reluctant for a background app).
        window.activate_window();
        window.focus(&self.focus);
        self.sync_tray();
        self.last_move = Instant::now();
        cx.notify();
    }

    /// Hides the window: Passthru waits in the tray, drawing and playing nothing.
    fn hide_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(h) = self.hwnd else { return };
        self.save_window(window);
        self.window_shown = false;
        self.menu_open = None;
        self.controls_visible = false;
        self.pill_bounds.set(None);
        winutil::set_controls_rect(None);
        self.video.gpu.paused.store(true, Ordering::Relaxed);
        self.audio.muted.store(true, Ordering::Relaxed);
        winutil::set_shown(h, false);
        self.sync_tray();
        cx.notify();
    }

    /// The close button, Alt+F4, a lost signal: to the tray while Passthru waits there, else
    /// quit.
    fn close(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.settings.stays_in_tray() {
            self.hide_window(window, cx);
        } else {
            self.quit(window, cx);
        }
    }

    fn quit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.save_window(window);
        if let Some(h) = self.hwnd {
            winutil::set_tray(h, false);
        }
        telemetry::stop();
        cx.quit();
    }

    fn toggle_menu(&mut self, menu: Menu, cx: &mut Context<Self>) {
        self.menu_open = if self.menu_open == Some(menu) { None } else { Some(menu) };
        cx.notify();
    }

    fn toggle_start_fullscreen(&mut self, cx: &mut Context<Self>) {
        self.settings.start_fullscreen = !self.settings.start_fullscreen;
        self.settings.save();
        cx.notify();
    }

    fn toggle_open_on_signal(&mut self, cx: &mut Context<Self>) {
        self.settings.open_on_signal = !self.settings.open_on_signal;
        self.settings.save();
        self.sync_tray();
        cx.notify();
    }

    fn toggle_close_on_loss(&mut self, cx: &mut Context<Self>) {
        self.settings.close_on_signal_loss = !self.settings.close_on_signal_loss;
        // Counted from now, not from a loss before it was turned on.
        self.signal_lost_at = self.signal_lost_at.map(|_| Instant::now());
        self.settings.save();
        cx.notify();
    }

    fn toggle_start_with_windows(&mut self, cx: &mut Context<Self>) {
        let on = !self.settings.start_with_windows;
        match install::set_autostart(on) {
            Ok(()) => {
                self.settings.start_with_windows = on;
                self.settings.save();
                self.sync_tray();
            }
            Err(e) => self.toast(format!("Can't change Start with Windows: {e:#}")),
        }
        cx.notify();
    }

    fn toggle_hide_from_capture(&mut self, cx: &mut Context<Self>) {
        self.settings.hide_from_capture = !self.settings.hide_from_capture;
        winutil::set_overlay_hidden_from_capture(self.settings.hide_from_capture);
        self.settings.save();
        cx.notify();
    }

    fn check_for_updates(&mut self, cx: &mut Context<Self>) {
        self.updater = update::spawn(true, true);
        self.updater_at = Instant::now();
        cx.notify();
    }

    fn set_volume(&mut self, v: f32, cx: &mut Context<Self>) {
        let v = (v.clamp(0.0, 1.0) * 100.0).round() / 100.0;
        self.audio.set_volume(v);
        self.settings.volume = v;
        if v > 0.0 && self.settings.muted {
            self.set_muted(false);
        }
        cx.notify();
    }

    fn set_muted(&mut self, muted: bool) {
        self.settings.muted = muted;
        self.audio.muted.store(muted, Ordering::Relaxed);
    }

    fn toggle_mute(&mut self, cx: &mut Context<Self>) {
        self.set_muted(!self.settings.muted);
        self.settings.save();
        cx.notify();
    }

    fn toggle_hdr(&mut self, cx: &mut Context<Self>) {
        self.settings.hdr = !self.settings.hdr;
        telemetry::note("setting", || format!("hdr {}", self.settings.hdr));
        self.video.hdr.store(self.settings.hdr, Ordering::Relaxed);
        self.settings.save();
        cx.notify();
    }

    /// Cycles frame generation: off, then up to 60, 120 or 240 fps, skipping rates past the
    /// display's own refresh (it can't show them; 144 Hz still gets 120).
    fn cycle_smooth(&mut self, cx: &mut Context<Self>) {
        let hz = self.video.gpu.refresh_hz.load(Ordering::Relaxed);
        self.settings.smooth = next_smooth(self.settings.smooth, hz);
        telemetry::note("setting", || format!("smooth {}", self.settings.smooth));
        self.video.gpu.target.store(self.settings.smooth, Ordering::Relaxed);
        self.settings.save();
        cx.notify();
    }

    /// Cycles the FPS counter: off, the counter, the counter with graphs.
    fn cycle_overlay(&mut self, cx: &mut Context<Self>) {
        self.settings.overlay = self.settings.overlay.next();
        telemetry::note("setting", || format!("overlay {:?}", self.settings.overlay));
        self.video.meter.set_on(self.settings.overlay != Overlay::Off);
        self.numbers = meter::Numbers::default();
        self.numbers_at = None;
        self.graphs = None;
        self.settings.save();
        cx.notify();
    }

    fn toggle_pin(&mut self, cx: &mut Context<Self>) {
        self.settings.on_top = !self.settings.on_top;
        if let Some(h) = self.hwnd {
            winutil::set_topmost(h, self.settings.on_top);
        }
        self.settings.save();
        cx.notify();
    }

    /// Captures in `mode` from now on (None: the best the device offers).
    fn pick_mode(&mut self, mode: Option<video::ModeKey>, cx: &mut Context<Self>) {
        telemetry::note("setting", || format!("mode {mode:?}"));
        self.video.control.pick(mode);
        self.settings.mode = mode.map(|(w, h, f)| [w, h, f]);
        self.settings.save();
        self.menu_open = None;
        cx.notify();
    }

    fn set_range(&mut self, range: video::ColourRange, cx: &mut Context<Self>) {
        self.settings.range = range;
        telemetry::note("setting", || format!("range {range:?}"));
        self.video.control.set_range(range);
        self.settings.save();
        cx.notify();
    }

    fn toggle_auto_update(&mut self, cx: &mut Context<Self>) {
        self.settings.auto_update = !self.settings.auto_update;
        self.settings.save();
        // Turned on: check now.
        self.updater = update::spawn(self.settings.auto_update, self.settings.auto_update);
        cx.notify();
    }

    fn toggle_rtx(&mut self, cx: &mut Context<Self>) {
        self.settings.rtx = !self.settings.rtx;
        telemetry::note("setting", || format!("rtx {}", self.settings.rtx));
        self.video.gpu.rtx.store(self.settings.rtx, Ordering::Relaxed);
        self.settings.save();
        cx.notify();
    }

    /// What's known about this run, for a session's `session.json`.
    fn session_context(&self, window: &Window) -> telemetry::SessionContext {
        let b = window.window_bounds().get_bounds();
        let info = serde_json::json!({
            "app": {
                "version": env!("CARGO_PKG_VERSION"),
                "commit": env!("PASSTHRU_COMMIT"),
                "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                "args": std::env::args().skip(1).collect::<Vec<_>>(),
            },
            "settings": &self.settings,
            "video": {
                "status": *self.video.status.lock().unwrap(),
                "mode": *self.video.mode.lock().unwrap(),
                "gpu_path": self.video.gpu.ready.load(Ordering::Acquire),
                "scaling": format!("{:?}", self.video.gpu.scaling()),
            },
            "display": {
                "refresh_hz": self.video.gpu.refresh_hz.load(Ordering::Relaxed),
                "scale_factor": window.scale_factor(),
                "fullscreen": window.is_fullscreen(),
                "window": [f32::from(b.origin.x), f32::from(b.origin.y), f32::from(b.size.width), f32::from(b.size.height)],
            },
        });
        telemetry::SessionContext { hwnd: self.hwnd.map(|h| h.0 as isize), info }
    }

    /// Starts or stops a test session.
    fn toggle_recording(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let ctx = self.session_context(window);
        match telemetry::toggle(&self.record, || ctx) {
            Ok(Some(dir)) => {
                self.bookmarks = 0;
                self.toast(format!("Recording to {}", dir.display()));
            }
            Ok(None) => self.toast("Recording saved".to_string()),
            Err(e) => self.toast(format!("Can't record: {e:#}")),
        }
        cx.notify();
    }

    /// Marks this moment in the session ("something looked wrong here").
    fn bookmark(&mut self, cx: &mut Context<Self>) {
        if !telemetry::on() {
            self.toast("Not recording: R starts a session".to_string());
        } else {
            self.bookmarks += 1;
            let n = self.bookmarks;
            telemetry::note("bookmark", || n.to_string());
            self.toast(format!("Bookmark {n}"));
        }
        cx.notify();
    }

    fn toast(&mut self, text: String) {
        crate::log::line(&format!("ui: {text}"));
        self.toast = Some((text, Instant::now()));
    }

    /// The REC badge and messages, top right.
    fn badges(&self) -> Option<gpui::Div> {
        let rec = telemetry::status().map(|(_, t)| {
            let s = t.as_secs();
            let marks = if self.bookmarks > 0 { format!("  \u{2691} {}", self.bookmarks) } else { String::new() };
            format!("\u{25CF} REC {:02}:{:02}{marks}", s / 60, s % 60)
        });
        let toast = self.toast.as_ref().filter(|(_, at)| at.elapsed() < TOAST_FOR).map(|(t, _)| t.clone());
        if rec.is_none() && toast.is_none() {
            return None;
        }
        let pill = |text: String, color: u32| {
            div()
                .px(px(10.))
                .py(px(4.))
                .rounded(px(12.))
                .bg(rgba(0x000000b0))
                .text_color(rgb(color))
                .text_size(px(12.))
                .font_weight(FontWeight::SEMIBOLD)
                .child(text)
        };
        Some(
            div()
                .absolute()
                .top(px(12.))
                .right(px(12.))
                .flex()
                .flex_col()
                .items_end()
                .gap(px(6.))
                .children(rec.map(|t| pill(t, 0xff5555)))
                .children(toast.map(|t| pill(t, 0xffffff))),
        )
    }

    /// The capture mode menu: what the device offers, the colour range and upscaling.
    fn mode_menu(&self, max_height: Pixels, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        let control = &self.video.control;
        let offered = control.modes.lock().unwrap().clone();
        let wanted = *control.wanted.lock().unwrap();
        let mut menu = menu_panel("mode-menu", 372., max_height).child(menu_header("CAPTURE MODE")).child(
            menu_item(
                "mode-auto".into(),
                "Auto".into(),
                offered.first().map(|&(w, h, f)| format!("Best offered: {}", video::mode_label(w, h, f))),
                wanted.is_none(),
            )
            .on_click(cx.listener(|this, _, _, cx| this.pick_mode(None, cx))),
        );
        // One row per resolution, its frame rates as chips: 40-odd modes in about ten rows.
        for (row, ((w, h), rates)) in video::by_resolution(&offered).into_iter().enumerate() {
            let picked_here = wanted.is_some_and(|(pw, ph, _)| (pw, ph) == (w, h));
            let chips = rates.into_iter().map(|f| {
                let on = wanted == Some((w, h, f));
                div()
                    .id(("rate", row * 1000 + f as usize))
                    .px(px(7.))
                    .py(px(2.))
                    .rounded(px(9.))
                    .text_size(px(11.))
                    .cursor_pointer()
                    .when(on, |d| d.bg(rgb(0xffffff)).text_color(rgb(0x000000)).font_weight(FontWeight::SEMIBOLD))
                    .when(!on, |d| d.bg(rgba(0xffffff14)).text_color(rgba(0xffffffcc)).hover(|s| s.bg(rgba(0xffffff2e))))
                    .child(f.to_string())
                    .on_click(cx.listener(move |this, _, _, cx| this.pick_mode(Some((w, h, f)), cx)))
            });
            menu = menu.child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .px(px(10.))
                    .py(px(4.))
                    .child(check_mark(picked_here))
                    .child(
                        div()
                            .w(px(72.))
                            .flex_none()
                            .text_size(px(13.))
                            .text_color(rgb(0xffffff))
                            .child(video::resolution_label(w, h)),
                    )
                    .child(div().flex().flex_wrap().gap(px(4.)).flex_1().children(chips)),
            );
        }
        if let Some(m) = wanted.filter(|m| !offered.contains(m)) {
            menu = menu.child(menu_note(format!(
                "{} is picked but not offered right now, so Auto is used until it is.",
                video::mode_label(m.0, m.1, m.2)
            )));
        }
        if offered.len() == 1 {
            menu = menu.child(menu_note(
                "The card offers only the resolution its input sends. For another one, change the console's video output.",
            ));
        }
        let rtx_state = match (self.settings.rtx, self.video.gpu.scaling()) {
            (false, _) => "Off: enlarging uses Lanczos".to_string(),
            (true, present::Scaling::Rtx) => "Running".to_string(),
            (true, present::Scaling::Lanczos) => {
                "Not running, so Lanczos is used. Turn on RTX Video › Super Resolution in the NVIDIA App.".to_string()
            }
            (true, present::Scaling::NotUp) => "Used when the video is enlarged to fit".to_string(),
        };
        let range = self.settings.range;
        let ranges = [
            (video::ColourRange::Auto, "Auto", control.auto_range_text()),
            (video::ColourRange::Limited, "Limited", "16–235, the video standard".to_string()),
            (video::ColourRange::Full, "Full", "0–255, the PS5's SDR output".to_string()),
        ];
        menu = menu.child(menu_rule()).child(menu_header("COLOUR RANGE"));
        for (choice, label, detail) in ranges {
            menu = menu.child(
                menu_item(("range", choice as usize).into(), label.into(), Some(detail), range == choice)
                    .on_click(cx.listener(move |this, _, _, cx| this.set_range(choice, cx))),
            );
        }
        menu.child(menu_rule()).child(menu_header("UPSCALING")).child(
            menu_item("rtx".into(), "RTX Video Super Resolution".into(), Some(rtx_state), self.settings.rtx)
                .on_click(cx.listener(|this, _, _, cx| this.toggle_rtx(cx))),
        )
    }

    /// The settings menu: how the window behaves, waiting in the background, updates.
    fn settings_menu(&self, max_height: Pixels, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        let s = &self.settings;
        let installed = install::running_installed();
        let update_state = self.updater.state();
        let ready = match &update_state {
            update::State::Ready(v) => Some(v.clone()),
            _ => None,
        };
        let checking = update_state == update::State::Checking;
        let update_text = match update_state {
            update::State::Off("not installed") => "Only the installed app updates itself".to_string(),
            update::State::Off(_) => "Off".to_string(),
            update::State::Checking => "Checking for a new version…".to_string(),
            update::State::UpToDate => "Up to date".to_string(),
            update::State::Ready(v) => format!("{v} downloaded: it starts next time"),
            update::State::Failed(e) => format!("Couldn't check: {}", e.chars().take(80).collect::<String>()),
        };
        let secs = CLOSE_AFTER_LOSS.as_secs();
        let loss_text = if s.stays_in_tray() {
            format!("After {secs} s without one, back to the tray")
        } else {
            format!("After {secs} s without one, Passthru quits")
        };
        menu_panel("settings-menu", 320., max_height)
            .child(menu_header("WINDOW"))
            .child(
                menu_switch("on-top", "Always on top", Some("Above other windows (T)".into()), s.on_top, true)
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_pin(cx))),
            )
            .child(
                menu_switch("start-fullscreen", "Open in fullscreen", None, s.start_fullscreen, true)
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_start_fullscreen(cx))),
            )
            .child(
                menu_switch(
                    "hide-from-capture",
                    "Hide controls from recordings",
                    Some("Screenshots, OBS and Discord see only the video".into()),
                    s.hide_from_capture,
                    true,
                )
                .on_click(cx.listener(|this, _, _, cx| this.toggle_hide_from_capture(cx))),
            )
            .child(menu_rule())
            .child(menu_header("BACKGROUND"))
            .child(
                menu_switch(
                    "open-on-signal",
                    "Open when a signal arrives",
                    Some("Closing the window keeps Passthru in the tray".into()),
                    s.open_on_signal,
                    true,
                )
                .on_click(cx.listener(|this, _, _, cx| this.toggle_open_on_signal(cx))),
            )
            .child(
                menu_switch("close-on-loss", "Close when the signal is lost", Some(loss_text), s.close_on_signal_loss, true)
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_close_on_loss(cx))),
            )
            .child({
                let detail = if installed { "In the tray, ready for a signal" } else { "Only for the installed app" };
                let row = menu_switch("start-with-windows", "Start with Windows", Some(detail.into()), s.start_with_windows, installed);
                if installed { row.on_click(cx.listener(|this, _, _, cx| this.toggle_start_with_windows(cx))) } else { row }
            })
            .child(menu_rule())
            .child(menu_header("UPDATES"))
            .child({
                let row = menu_switch("autoupdate", "Update automatically", Some(update_text), s.auto_update, installed);
                if installed { row.on_click(cx.listener(|this, _, _, cx| this.toggle_auto_update(cx))) } else { row }
            })
            .when(installed && ready.is_none(), |d| {
                d.child(
                    menu_action("check-updates", if checking { "Checking…" } else { "Check for updates" }, !checking)
                        .on_click(cx.listener(|this, _, _, cx| {
                            if this.updater.state() != update::State::Checking {
                                this.check_for_updates(cx);
                            }
                        })),
                )
            })
            .children(ready.map(|v| {
                menu_action("restart", &format!("Restart now to use {v}"), true).on_click(cx.listener(|this, _, _, _| {
                    if update::restart().is_ok() {
                        this.request(winutil::Event::Quit);
                    }
                }))
            }))
            .child(
                div()
                    .px(px(10.))
                    .pt(px(6.))
                    .pb(px(4.))
                    .text_size(px(10.))
                    .text_color(rgba(0xffffff4d))
                    .child(format!("{} {}", app::NAME, env!("CARGO_PKG_VERSION"))),
            )
    }

    fn scroll_volume(&mut self, e: &ScrollWheelEvent, cx: &mut Context<Self>) {
        let dy = e.delta.pixel_delta(px(20.)).y;
        if dy != px(0.) {
            let step = if dy > px(0.) { 0.05 } else { -0.05 };
            self.set_volume(self.settings.volume + step, cx);
            self.settings.save();
            self.last_move = Instant::now();
        }
    }

    /// Asks for something done to the main window (the controls are in another one).
    fn request(&self, e: winutil::Event) {
        let _ = self.requests.unbounded_send(e);
    }

    fn volume_from_x(&mut self, x: Pixels, cx: &mut Context<Self>) {
        if let Some(b) = self.slider_bounds.get() {
            self.set_volume((x - b.origin.x) / b.size.width, cx);
        }
    }

    /// Remembers where the window is (its restored bounds, if maximized or fullscreen).
    fn save_window(&mut self, window: &Window) {
        let b = window.window_bounds().get_bounds();
        self.settings.window = Some([
            f32::from(b.origin.x),
            f32::from(b.origin.y),
            f32::from(b.size.width),
            f32::from(b.size.height),
        ]);
        self.settings.save();
    }
}

/// A menu: dark, rounded, scrolling once it's taller than `max_height`.
fn menu_panel(id: &'static str, width: f32, max_height: Pixels) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .w(px(width))
        .max_h(max_height)
        .overflow_y_scroll()
        .flex()
        .flex_col()
        .p(px(4.))
        .rounded(px(10.))
        .bg(rgba(0x121212f5))
        .border_1()
        .border_color(rgba(0xffffff1a))
}

fn menu_header(text: impl Into<SharedString>) -> gpui::Div {
    div()
        .px(px(10.))
        .pt(px(8.))
        .pb(px(4.))
        .text_size(px(10.))
        .font_weight(FontWeight::BOLD)
        .text_color(rgba(0xffffff66))
        .child(text.into())
}

fn menu_note(text: impl Into<SharedString>) -> gpui::Div {
    div().px(px(10.)).pb(px(6.)).text_size(px(11.)).text_color(rgba(0xffffff80)).child(text.into())
}

fn menu_rule() -> gpui::Div {
    div().h(px(1.)).mx(px(6.)).my(px(4.)).bg(rgba(0xffffff1a))
}

fn check_mark(checked: bool) -> gpui::Div {
    div()
        .w(px(14.))
        .flex_none()
        .font_family(ICON_FONT)
        .text_size(px(12.))
        .text_color(rgb(0xffffff))
        .child(if checked { ICON_CHECK } else { "" })
}

/// A row's label, and a line under it in grey; it takes the row's width, so long details wrap.
fn menu_label(label: impl Into<SharedString>, detail: Option<String>) -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .flex_1()
        .min_w(px(0.))
        .child(div().text_size(px(13.)).text_color(rgb(0xffffff)).child(label.into()))
        .children(detail.map(|d| div().text_size(px(11.)).text_color(rgba(0xffffff80)).child(d)))
}

fn menu_row(id: impl Into<gpui::ElementId>, enabled: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id.into())
        .flex()
        .items_center()
        .gap(px(8.))
        .px(px(10.))
        .py(px(6.))
        .rounded(px(6.))
        .when(enabled, |d| d.cursor_pointer().hover(|s| s.bg(rgba(0xffffff1f))))
        .when(!enabled, |d| d.opacity(0.4))
}

/// One choice of several, ticked when it's the one picked.
fn menu_item(id: gpui::ElementId, label: String, detail: Option<String>, checked: bool) -> gpui::Stateful<gpui::Div> {
    menu_row(id, true).child(check_mark(checked)).child(menu_label(label, detail))
}

/// A setting that's on or off: its label, and a switch at the right.
fn menu_switch(id: &'static str, label: &'static str, detail: Option<String>, on: bool, enabled: bool) -> gpui::Stateful<gpui::Div> {
    let switch = div()
        .relative()
        .w(px(28.))
        .h(px(16.))
        .flex_none()
        .rounded_full()
        .when(on, |d| d.bg(rgb(0xffffff)))
        .when(!on, |d| d.bg(rgba(0xffffff2e)))
        .child(
            div()
                .absolute()
                .top(px(2.))
                .left(px(if on { 14. } else { 2. }))
                .size(px(12.))
                .rounded_full()
                .bg(if on { rgb(0x121212) } else { rgb(0xffffff) }),
        );
    menu_row(id, enabled).gap(px(12.)).child(menu_label(label, detail)).child(switch)
}

/// Something to do (check for updates), lined up with the labels above it.
fn menu_action(id: &'static str, label: &str, enabled: bool) -> gpui::Stateful<gpui::Div> {
    menu_row(id, enabled).child(div().text_size(px(13.)).text_color(rgb(0xffffff)).child(label.to_string()))
}

fn icon_button(id: &'static str, glyph: &'static str) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .size(px(32.))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .rounded(px(6.))
        .cursor_pointer()
        .font_family(ICON_FONT)
        .text_size(px(15.))
        .text_color(rgb(0xffffff))
        .hover(|s| s.bg(rgba(0xffffff26)))
        .child(glyph)
}

/// A small text switch: filled white when on, outlined when off.
/// A toggle in the controls, "NAME | STATE", `width` wide whatever the state says (sized for
/// its longest), so the bar doesn't change width, and buttons don't move, as states change.
fn chip(id: &'static str, name: &'static str, state: String, on: bool, width: f32) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(24.))
        .w(px(width))
        .mx(px(2.))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .rounded(px(5.))
        .cursor_pointer()
        .text_size(px(11.))
        .font_weight(FontWeight::BOLD)
        .when(on, |d| d.bg(rgb(0xffffff)).text_color(rgb(0x000000)))
        .when(!on, |d| {
            d.border_1()
                .border_color(rgba(0xffffff55))
                .text_color(rgba(0xffffffaa))
                .hover(|s| s.bg(rgba(0xffffff26)))
        })
        .gap(px(5.))
        .child(name)
        .child(div().opacity(0.45).child("|"))
        .child(state)
}

fn divider() -> gpui::Div {
    div().w(px(1.)).h(px(18.)).mx(px(4.)).flex_none().bg(rgba(0xffffff33))
}

impl Viewer {
    /// The controls, their menus and the messages top right. They're drawn in a window of their
    /// own over the video's (see winutil's overlay), which recordings can be kept from seeing.
    fn render_controls(&mut self, window: &mut Window, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        let viewport = window.viewport_size();
        let fullscreen = self.fullscreen;
        let muted = self.settings.muted;
        let volume = self.settings.volume;
        let hdr = self.settings.hdr;
        let slider_bounds = self.slider_bounds.clone();
        let pill_bounds = self.pill_bounds.clone();
        let fill: Hsla = if muted { rgba(0xffffff55).into() } else { rgb(0xffffff).into() };
        let slider = div()
            .id("volume")
            .relative()
            .w(px(120.))
            .h(px(32.))
            // Room for half the knob at each end, so it never pokes out of the slider.
            .px(px(7.))
            .flex()
            .flex_none()
            .items_center()
            .cursor_pointer()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, e: &MouseDownEvent, _, cx| {
                    this.dragging_volume = true;
                    this.volume_from_x(e.position.x, cx);
                    cx.stop_propagation();
                }),
            )
            .child(
                div()
                    .relative()
                    .w_full()
                    .h(px(4.))
                    .rounded(px(2.))
                    .bg(rgba(0xffffff33))
                    .child(canvas(move |b, _, _| slider_bounds.set(Some(b)), |_, _, _, _| {}).absolute().size_full())
                    .child(div().h_full().rounded(px(2.)).w(relative(volume)).bg(fill))
                    // Knob, centred on the end of the fill.
                    .child(
                        div()
                            .absolute()
                            .top(px(-4.))
                            .left(relative(volume))
                            .ml(px(-6.))
                            .size(px(12.))
                            .rounded_full()
                            .bg(fill),
                    ),
            );

        let volume_glyph = if muted || volume == 0.0 {
            ICON_MUTE
        } else if volume < 0.5 {
            ICON_VOLUME_LOW
        } else {
            ICON_VOLUME
        };
        // The capture mode, and nothing else (2160p60).
        let mode = self.video.mode.lock().unwrap().clone();
        let smooth = self.settings.smooth;

        let on_off = |on: bool| if on { "ON".to_string() } else { "OFF".to_string() };
        let hdr_button =
            chip("hdr", "HDR", on_off(hdr), hdr, 72.).on_click(cx.listener(|this, _, _, cx| this.toggle_hdr(cx)));
        let smooth_value = if smooth == 0 { "OFF".to_string() } else { smooth.to_string() };
        let smooth_button = chip("smooth", "SMOOTHING", smooth_value, smooth > 0, 118.)
            .on_click(cx.listener(|this, _, _, cx| this.cycle_smooth(cx)));
        let overlay_mode = self.settings.overlay;
        let overlay_value = match overlay_mode {
            Overlay::Off => "OFF",
            Overlay::Counter => "ON",
            Overlay::Graph => "GRAPH",
        };
        let overlay_button = chip("fps", "FPS", overlay_value.to_string(), overlay_mode != Overlay::Off, 88.)
            .on_click(cx.listener(|this, _, _, cx| this.cycle_overlay(cx)));

        // Room above the controls, so the menu never runs off the top (it scrolls instead).
        let menu_room = (viewport.height - px(90.)).max(px(160.));
        let menu = match self.menu_open {
            Some(Menu::Mode) => Some(self.mode_menu(menu_room, cx)),
            Some(Menu::Settings) => Some(self.settings_menu(menu_room, cx)),
            None => None,
        };
        // Slide in from below the window and back out, on a spring; settled, no more frames.
        let target = if self.controls_visible { 0.0 } else { 1.0 };
        let now = Instant::now();
        let dt = if self.bar_moving { now.duration_since(self.bar_at).as_secs_f32() } else { 0.0 };
        self.bar_at = now;
        if self.animate {
            self.bar_moving = self.bar.step(target, dt);
            if self.bar_moving {
                window.request_animation_frame();
            }
        } else {
            self.bar.snap(target);
        }
        let slide = self.bar.pos;
        let controls = div()
            .absolute()
            .bottom(px(14.0 - BAR_TRAVEL * slide))
            .opacity((1.0 - 0.6 * slide).clamp(0.0, 1.0))
            .left_0()
            .right_0()
            .flex()
            .justify_center()
            .child(
                // The pill, and the menu above it while open. A press anywhere else closes the menu.
                div()
                    .id("controls")
                    .relative()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap(px(8.))
                    .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                        if this.menu_open.take().is_some() {
                            cx.notify();
                        }
                    }))
                    .child(canvas(move |b, _, _| pill_bounds.set(Some(b)), |_, _, _, _| {}).absolute().size_full())
                    .children(menu)
                    .child(
                div()
                    .id("pill")
                    .relative()
                    .flex()
                    .items_center()
                    .gap(px(4.))
                    .px(px(8.))
                    .py(px(4.))
                    .rounded(px(10.))
                    .bg(rgba(0x000000b8))
                    .border_1()
                    .border_color(rgba(0xffffff1a))
                    .child(
                        icon_button("mute", volume_glyph)
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_mute(cx))),
                    )
                    .child(slider)
                    .child(
                        div()
                            .w(px(28.))
                            .ml(px(2.))
                            .flex_none()
                            .text_size(px(12.))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgba(0xffffffcc))
                            .child(format!("{:.0}", volume * 100.0)),
                    )
                    .child(divider())
                    .child(hdr_button)
                    .child(smooth_button)
                    .child(overlay_button)
                    // The mode, which opens the capture mode menu.
                    .child(
                        // As wide as the mode's name: it only changes when the mode does, not while
                        // clicking the buttons beside it.
                        div()
                            .id("mode")
                            .h(px(24.))
                            .px(px(6.))
                            .flex()
                            .flex_none()
                            .items_center()
                            .gap(px(5.))
                            .rounded(px(5.))
                            .cursor_pointer()
                            .text_size(px(12.))
                            .text_color(rgba(0xffffffb3))
                            .hover(|s| s.bg(rgba(0xffffff26)))
                            .when(self.menu_open == Some(Menu::Mode), |d| d.bg(rgba(0xffffff26)))
                            .child(if mode.is_empty() { "Mode".to_string() } else { mode })
                            .child(div().flex_none().font_family(ICON_FONT).text_size(px(9.)).child(ICON_CHEVRON_UP))
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_menu(Menu::Mode, cx))),
                    )
                    .child(divider())
                    .child(
                        icon_button("settings", ICON_SETTINGS)
                            .when(self.menu_open == Some(Menu::Settings), |d| d.bg(rgba(0xffffff26)))
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_menu(Menu::Settings, cx))),
                    )
                    .child(
                        icon_button("fullscreen", if fullscreen { ICON_EXIT_FULLSCREEN } else { ICON_FULLSCREEN })
                            .on_click(cx.listener(|this, _, _, _| this.request(winutil::Event::ToggleFullscreen))),
                    )
                    .child(
                        icon_button("close", ICON_CLOSE)
                            .on_click(cx.listener(|this, _, _, _| this.request(winutil::Event::Close))),
                    ),
                    ),
            );


        let badges = self.badges();
        let drawn = self.controls_visible || slide < 0.999;
        self.controls_drawn = drawn || badges.is_some();
        div()
            .id("controls-root")
            .relative()
            .size_full()
            .on_mouse_move(cx.listener(|this, e: &MouseMoveEvent, _, cx| {
                if this.dragging_volume {
                    this.volume_from_x(e.position.x, cx);
                }
            }))
            .on_scroll_wheel(cx.listener(|this, e: &ScrollWheelEvent, _, cx| this.scroll_volume(e, cx)))
            .children(badges)
            .when(drawn, |d| d.child(controls))
    }
}

impl Render for Viewer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Swap in the newest frame and free the previous one's GPU texture.
        if let Some((frame, ready_at)) = self.video.latest.lock().unwrap().take() {
            let now = Instant::now();
            self.video.meter.shown(now, false, Some(ready_at));
            self.stats.window.record(ready_at, now);
            if let Some(total) = &mut self.stats.total {
                total.record(ready_at, now);
            }
            if let Some(old) = self.shown.replace(frame) {
                let _ = window.drop_image(old);
            }
        }
        self.video.pending.store(false, Ordering::Release);
        // Once the GPU layer under this one draws the video, this layer stays transparent
        // except for the controls.
        let gpu = self.video.gpu.ready.load(Ordering::Acquire);
        self.gpu_drawing = gpu;
        if gpu && let Some(old) = self.shown.take() {
            let _ = window.drop_image(old);
        }

        let aspect = f32::from_bits(self.video.aspect.load(Ordering::Relaxed));
        if aspect != self.aspect {
            self.aspect = aspect;
            winutil::set_aspect(aspect);
        }

        // Fit the video ourselves, snapped to whole device pixels: a fractional position would
        // blur every pixel, and at 1440p in fullscreen it lands exactly 1:1 on the screen.
        let scale = window.scale_factor();
        let viewport = window.viewport_size();
        let (vw, vh) = (f32::from(viewport.width) * scale, f32::from(viewport.height) * scale);
        let (fit_w, fit_h) = if vw / vh > aspect {
            ((vh * aspect).round(), vh.round())
        } else {
            (vw.round(), (vw / aspect).round())
        };
        let (fit_x, fit_y) = (((vw - fit_w) / 2.0).floor(), ((vh - fit_h) / 2.0).floor());
        self.video.display_w.store(fit_w as u32, Ordering::Relaxed);
        let logical = |v: f32| px(v / scale);

        let signal = self.has_signal();
        self.had_signal = signal;

        let video = div()
            .id("video")
            .absolute()
            .size_full()
            .when_some(self.shown.clone().filter(|_| signal && !gpu), |d, frame| {
                d.child(
                    img(ImageSource::Render(frame))
                        .absolute()
                        .left(logical(fit_x))
                        .top(logical(fit_y))
                        .w(logical(fit_w))
                        .h(logical(fit_h))
                        .object_fit(ObjectFit::Fill),
                )
            })
            .when(!signal, |d| {
                let status = {
                    let v = self.video.status.lock().unwrap().clone();
                    if self.video.last_frame.lock().unwrap().is_some() || v.is_empty() {
                        "No signal".to_string()
                    } else {
                        v
                    }
                };
                d.flex().items_center().justify_center().child(
                    div()
                        .text_color(rgba(0xffffff66))
                        .text_size(px(14.))
                        .child(SharedString::from(status)),
                )
            });

        let overlay_mode = self.settings.overlay;
        // Inside the video's top-left corner (clear of black bars).
        let overlay = (signal && overlay_mode != Overlay::Off).then(|| {
            overlay::panel(overlay_mode, &self.numbers, self.graphs.as_ref())
                .absolute()
                .left(logical(fit_x) + px(12.))
                .top(logical(fit_y) + px(12.))
        });

        div()
            .id("root")
            .track_focus(&self.focus)
            .relative()
            .size_full()
            .when(!gpu, |d| d.bg(rgb(0x000000)))
            .on_mouse_move(cx.listener(|this, e: &MouseMoveEvent, _, cx| {
                if this.dragging_volume {
                    this.volume_from_x(e.position.x, cx);
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| {
                    if std::mem::take(&mut this.dragging_volume) {
                        this.settings.save();
                    }
                }),
            )
            .on_scroll_wheel(cx.listener(|this, e: &ScrollWheelEvent, _, cx| this.scroll_volume(e, cx)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if this.menu_open.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .on_key_down(cx.listener(|this, e: &KeyDownEvent, window, cx| {
                match e.keystroke.key.as_str() {
                    "f" | "f11" | "enter" => window.toggle_fullscreen(),
                    "escape" if this.menu_open.is_some() => {
                        this.menu_open = None;
                        cx.notify();
                    }
                    "escape" if window.is_fullscreen() => window.toggle_fullscreen(),
                    "m" => this.toggle_mute(cx),
                    "h" => this.toggle_hdr(cx),
                    "s" => this.cycle_smooth(cx),
                    "o" => this.cycle_overlay(cx),
                    "t" | "p" => this.toggle_pin(cx),
                    // Test sessions are a developer tool (`--dev`, see app::dev_mode).
                    "r" if app::dev_mode() => this.toggle_recording(window, cx),
                    "b" if app::dev_mode() => this.bookmark(cx),
                    "up" => {
                        this.set_volume(this.settings.volume + 0.05, cx);
                        this.settings.save();
                    }
                    "down" => {
                        this.set_volume(this.settings.volume - 0.05, cx);
                        this.settings.save();
                    }
                    _ => return,
                }
                this.last_move = Instant::now();
            }))
            .child(video)
            .children(overlay)
    }
}

/// Opens the controls' window: transparent, borderless, never activated, owned by the main
/// window and kept on its client area (winutil does that, and the clicks: only those on the
/// controls stay here, the rest go through to the video).
fn open_controls_window(viewer: gpui::Entity<Viewer>, in_capture: bool, cx: &mut App) {
    let (main, hide) = {
        let v = viewer.read(cx);
        (v.hwnd, v.settings.hide_from_capture && !in_capture)
    };
    let Some(main) = main else { return };
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(Bounds::new(point(px(0.), px(0.)), size(px(64.), px(64.))))),
        titlebar: None,
        focus: false,
        show: false,
        kind: gpui::WindowKind::PopUp,
        is_movable: false,
        is_resizable: false,
        is_minimizable: false,
        window_background: WindowBackgroundAppearance::Transparent,
        ..Default::default()
    };
    // Taken over only once gpui has set the window up: sizing it before then is lost.
    let opened = cx
        .open_window(options, |_, cx| cx.new(|cx| ControlsView::new(viewer, cx)))
        .and_then(|w| w.update(cx, |_, window, _| winutil::hwnd(window)));
    match opened {
        Ok(Some(h)) => winutil::install_overlay(h, main, hide),
        Ok(None) => {}
        Err(e) => log::line(&format!("controls window: {e:#}")),
    }
}

/// The frame generation target after `now` (0 = off), offering only rates up to `hz` (0 =
/// display rate not known: all of them).
fn next_smooth(now: u32, hz: u32) -> u32 {
    let fits = |f: u32| hz == 0 || f <= hz + hz / 20;
    [60, 120, 240].into_iter().filter(|&f| f > now && fits(f)).min().unwrap_or(0)
}

#[cfg(test)]
mod smooth_tests {
    use super::next_smooth;

    #[test]
    fn cycles_only_through_rates_the_display_shows() {
        let cycle = |hz| {
            let mut seen = vec![];
            let mut s = 0;
            loop {
                s = next_smooth(s, hz);
                if s == 0 {
                    return seen;
                }
                seen.push(s);
            }
        };
        assert_eq!(cycle(240), [60, 120, 240]);
        assert_eq!(cycle(144), [60, 120]);
        assert_eq!(cycle(60), [60]);
        assert_eq!(cycle(59), [60]);
        assert_eq!(cycle(0), [60, 120, 240]);
    }
}

struct Args {
    list: bool,
    snapshot: Option<std::path::PathBuf>,
    video: VideoOptions,
    audio: Option<String>,
    stats: bool,
    exit_after: Option<Duration>,
    fullscreen: bool,
    on_top: bool,
    /// Frame generation target for this run, without saving it.
    smooth: Option<u32>,
    /// FPS counter mode for this run, without saving it.
    overlay: Option<Overlay>,
    /// Draw through gpui on the CPU path only.
    cpu: bool,
    /// Debug: after SECS, save what the window shows on screen to PATH.
    capture: Option<(Duration, std::path::PathBuf)>,
    /// Debug: keep the controls up with the mode menu open (for screenshots).
    show_menu: bool,
    /// Debug: the same with the settings menu open.
    show_settings: bool,
    /// Debug: screenshots and recordings see the controls, this run (for screenshots of them).
    controls_in_capture: bool,
    /// Start hidden, in the tray (how Start with Windows starts it).
    background: bool,
    /// Started by a Passthru that's exiting (an update): wait for it rather than defer to it.
    restarted: bool,
    /// Start a test session as soon as the video is up.
    record: bool,
    record_opts: telemetry::Options,
}

fn parse_args() -> Args {
    let mut a = Args {
        list: false,
        snapshot: None,
        video: VideoOptions { device: None, mode: None, test: None, test_size: None, test_hitch: false },
        audio: None,
        stats: cfg!(debug_assertions),
        exit_after: None,
        fullscreen: false,
        on_top: false,
        smooth: None,
        overlay: None,
        cpu: false,
        capture: None,
        show_menu: false,
        show_settings: false,
        controls_in_capture: false,
        background: false,
        restarted: false,
        record: false,
        record_opts: telemetry::Options::default(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list" => a.list = true,
            "--stats" => a.stats = true,
            "--fullscreen" => a.fullscreen = true,
            "--on-top" => a.on_top = true,
            "--smooth" => a.smooth = args.next().and_then(|f| f.parse().ok()),
            "--fps" => a.overlay = args.next().and_then(|m| serde_json::from_value(serde_json::Value::String(m)).ok()),
            "--cpu" => a.cpu = true,
            "--show-menu" => a.show_menu = true,
            "--show-settings" => a.show_settings = true,
            "--controls-in-capture" => a.controls_in_capture = true,
            "--background" => a.background = true,
            "--restarted" => a.restarted = true,
            // Test sessions (see telemetry.rs); R toggles one in the app too.
            // Developer mode (test sessions, R and B); `--record` implies it.
            "--dev" => {}
            "--record" => a.record = true,
            "--record-dir" => {
                a.record = true;
                a.record_opts.dir = args.next().map(Into::into);
            }
            "--record-no-input" => a.record_opts.input_video = false,
            "--record-no-output" => a.record_opts.output_video = false,
            "--capture-after" => {
                let secs = args.next().and_then(|s| s.parse().ok()).map(Duration::from_secs_f64);
                a.capture = secs.zip(args.next().map(Into::into));
            }
            // FPS, or FPS/PICTURES to repeat frames like a slower game: 60/30.
            "--test-pattern" => {
                a.video.test = args.next().and_then(|f| {
                    let (fps, content) = f.split_once('/').unwrap_or((&f, &f));
                    let fps: u32 = fps.parse().ok()?;
                    Some((fps, content.parse::<u32>().ok()?.clamp(1, fps)))
                })
            }
            // The test pattern's pictures come unevenly now and then (held 2, 2, 2, 3, 1 at 30 on 60).
            "--test-hitch" => a.video.test_hitch = true,
            // WIDTHxHEIGHT of the test pattern (2560x1440 if not given).
            "--test-size" => {
                a.video.test_size = args.next().and_then(|s| {
                    let (w, h) = s.split_once('x')?;
                    Some((w.parse().ok()?, h.parse().ok()?))
                })
            }
            "--exit-after" => {
                a.exit_after = args.next().and_then(|s| s.parse().ok()).map(Duration::from_secs_f64)
            }
            "--snapshot" => a.snapshot = args.next().map(Into::into),
            "--video" => a.video.device = args.next(),
            "--audio" => a.audio = args.next(),
            "--mode" => {
                // WIDTHxHEIGHT@FPS
                a.video.mode = args.next().and_then(|m| {
                    let (res, fps) = m.split_once('@')?;
                    let (w, h) = res.split_once('x')?;
                    Some((w.parse().ok()?, h.parse().ok()?, fps.parse().ok()?))
                });
            }
            other => eprintln!("unknown argument: {other}"),
        }
    }
    a
}

fn main() {
    // Debug: --decode-raw RAW.nv12 WIDTH HEIGHT OUT.png [2020 | sdr | sdr-full] (HDR by default)
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("--decode-raw") && argv.len() >= 6 {
        let (w, h) = (argv[3].parse().unwrap_or(0), argv[4].parse().unwrap_or(0));
        let how = argv.get(6).map_or("hdr", String::as_str);
        if let Err(e) = video::decode_raw(argv[2].as_ref(), w, h, how, argv[5].as_ref()) {
            eprintln!("decode-raw: {e:#}");
            std::process::exit(1);
        }
        return;
    }
    match argv.get(1).map(String::as_str) {
        // Debug: --gpu-test IMAGE OUTDIR [OTHER]
        Some("--gpu-test") if argv.len() >= 4 => {
            let other = argv.get(4).map(std::path::Path::new);
            if let Err(e) = gputest::interp_test(argv[2].as_ref(), argv[3].as_ref(), other) {
                eprintln!("gpu-test: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        // Debug: --gpu-eval CLIP.y4m OUTDIR [SKIP] [PAIRS TO SAVE, comma separated]
        Some("--gpu-eval") if argv.len() >= 4 => {
            let skip = argv.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
            let save: Vec<usize> = argv.get(5).map(|s| s.split(',').filter_map(|p| p.parse().ok()).collect()).unwrap_or_default();
            if let Err(e) = gputest::eval(argv[2].as_ref(), argv[3].as_ref(), skip, &save) {
                eprintln!("gpu-eval: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        // Debug: --mark-test IMAGE OUTDIR (test sessions' barcode, for tools/analyze.py selftest)
        Some("--mark-test") if argv.len() >= 4 => {
            if let Err(e) = gputest::mark_test(argv[2].as_ref(), argv[3].as_ref()) {
                eprintln!("mark-test: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        // Debug: --upscale-test IMAGE OUTDIR [SCALE]
        Some("--upscale-test") if argv.len() >= 4 => {
            let scale = argv.get(4).and_then(|s| s.parse().ok()).unwrap_or(4.0 / 3.0);
            if let Err(e) = gputest::upscale_test(argv[2].as_ref(), argv[3].as_ref(), scale) {
                eprintln!("upscale-test: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        // Debug: --gpu-decode-test RAW.nv12 WIDTH HEIGHT
        Some("--gpu-decode-test") if argv.len() >= 5 => {
            let (w, h) = (argv[3].parse().unwrap_or(0), argv[4].parse().unwrap_or(0));
            if let Err(e) = gputest::decode_test(argv[2].as_ref(), w, h) {
                eprintln!("gpu-decode-test: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("--install") => {
            if let Err(e) = install::install() {
                eprintln!("install: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        Some("--uninstall") => {
            let quiet = argv.iter().any(|a| a == "--quiet");
            if let Err(e) = install::uninstall(quiet) {
                eprintln!("uninstall: {e:#}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }
    install::set_process_app_id();
    let args = parse_args();
    if args.list {
        if let Err(e) = video::list() {
            eprintln!("video: {e:#}");
        }
        if let Err(e) = audio::list() {
            eprintln!("audio: {e:#}");
        }
        return;
    }
    if let Some(dir) = &args.snapshot {
        if let Err(e) = video::snapshot(&args.video, dir) {
            eprintln!("snapshot: {e:#}");
            std::process::exit(1);
        }
        return;
    }

    // Started from Downloads or similar with a normal double-click: offer to install (not for
    // dev builds, or runs with options, which are tests and tools).
    let dev_build = std::env::current_exe().is_ok_and(|p| p.components().any(|c| c.as_os_str() == "target"));
    if argv.len() == 1 && !cfg!(debug_assertions) && !dev_build && install::offer() {
        return;
    }
    // One at a time: if one's already running (in the tray, say), bring that one up instead.
    if !cfg!(debug_assertions) && !dev_build {
        let wait = if args.restarted { Duration::from_secs(10) } else { Duration::ZERO };
        if !instance::claim(wait) && (args.background || instance::show_running() || !instance::claim(Duration::from_secs(3))) {
            return;
        }
    }
    log::start();
    if install::running_installed() {
        update::clean_up();
        install::refresh_registration();
    }
    let mut settings = Settings::load();
    if install::running_installed() && settings.start_with_windows {
        let _ = install::set_autostart(true);
    }
    if let Some(s) = args.smooth {
        settings.smooth_for_run(s);
    }
    if let Some(o) = args.overlay {
        settings.overlay_for_run(o);
    }
    let (waker, mut woken) = futures::channel::mpsc::unbounded::<()>();
    let video = Arc::new(VideoShared {
        latest: Mutex::new(None),
        produced: AtomicU64::new(0),
        convert_ns: AtomicU64::new(0),
        pending: AtomicBool::new(false),
        waker,
        status: Mutex::new(String::new()),
        mode: Mutex::new(String::new()),
        last_frame: Mutex::new(None),
        aspect: AtomicU32::new((16.0f32 / 9.0).to_bits()),
        hdr: AtomicBool::new(settings.hdr),
        display_w: AtomicU32::new(0),
        meter: meter::Meter::new(),
        gpu: present::Shared::new(settings.smooth, args.stats, settings.rtx, false),
        control: video::Control::default(),
    });
    // `--mode` for this run, else the one picked in the menu.
    *video.control.wanted.lock().unwrap() = args.video.mode.or(settings.mode.map(|[w, h, f]| (w, h, f)));
    video.control.set_range(settings.range);
    video.meter.set_on(settings.overlay != Overlay::Off);
    let audio = AudioShared::new(settings.volume, settings.muted);
    let updater = update::spawn(settings.auto_update, false);
    video::spawn(video.clone(), args.video);
    audio::spawn(audio.clone(), args.audio);
    let (stats_enabled, exit_after, start_fullscreen) = (args.stats, args.exit_after, args.fullscreen);
    let (force_on_top, cpu_only, capture) = (args.on_top, args.cpu, args.capture.clone());
    let (show_menu, show_settings) = (args.show_menu, args.show_settings);
    let start_hidden = args.background;
    let controls_in_capture = args.controls_in_capture;
    let start_fullscreen = (start_fullscreen || settings.start_fullscreen) && !start_hidden;
    let (record_now, record_opts) = (args.record, args.record_opts.clone());

    Application::new().run(move |cx: &mut App| {
        // Last position if it's still on a connected display, else centred.
        let saved = settings.window.map(|[x, y, w, h]| Bounds {
            origin: point(px(x), px(y)),
            size: size(px(w), px(h)),
        });
        let bounds = saved
            .filter(|b| cx.displays().iter().any(|d| d.bounds().intersects(b)))
            .unwrap_or_else(|| Bounds::centered(None, size(px(1280.), px(720.)), cx));
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions {
                title: Some(app::NAME.into()),
                appears_transparent: true,
                traffic_light_position: None,
            }),
            window_min_size: Some(size(px(320.), px(180.))),
            window_background: WindowBackgroundAppearance::Opaque,
            app_id: Some("passthru".into()),
            show: !start_hidden,
            focus: !start_hidden,
            ..Default::default()
        };
        let main_window = cx.open_window(options, |window, cx| {
            let hwnd = winutil::hwnd(window);
            let (events_tx, mut events) = futures::channel::mpsc::unbounded();
            let requests = events_tx.clone();
            if let Some(h) = hwnd {
                winutil::install(h, events_tx);
                if settings.on_top || force_on_top {
                    winutil::set_topmost(h, true);
                }
                // The GPU layer starts once it has the window (never, with --cpu).
                if !cpu_only {
                    let _ = video.gpu.hwnd.set(h.0 as isize);
                }
            }
            let view = cx.new(|cx| {
                // Repaint whenever the capture thread has a new frame.
                cx.spawn(async move |this, cx| {
                    while woken.next().await.is_some() {
                        if this.update(cx, |_: &mut Viewer, cx| cx.notify()).is_err() {
                            break;
                        }
                    }
                })
                .detach();
                // Requests from the window procedure wrapper.
                cx.spawn_in(window, async move |this, cx| {
                    while let Some(event) = events.next().await {
                        let ok = this.update_in(cx, |this: &mut Viewer, window, cx| match event {
                            winutil::Event::PressOutside => {
                                if this.menu_open.take().is_some() {
                                    cx.notify();
                                }
                            }
                            winutil::Event::Open => this.show_window(window, cx),
                            winutil::Event::Quit => this.quit(window, cx),
                            winutil::Event::Close => this.close(window, cx),
                            winutil::Event::ToggleFullscreen => window.toggle_fullscreen(),
                        });
                        if ok.is_err() {
                            break;
                        }
                    }
                })
                .detach();
                // Pointer polling for the auto-hiding controls.
                cx.spawn_in(window, async move |this, cx| {
                    loop {
                        cx.background_executor().timer(TICK).await;
                        if this.update_in(cx, |this, window, cx| this.tick(window, cx)).is_err() {
                            break;
                        }
                    }
                })
                .detach();
                let focus = cx.focus_handle();
                window.focus(&focus);
                if start_fullscreen {
                    window.toggle_fullscreen();
                }
                let stats = Stats {
                    enabled: stats_enabled,
                    exit_after,
                    capture: capture.clone(),
                    started: Instant::now(),
                    window: StatsWindow::new(&video),
                    total: None,
                };
                Viewer {
                    stats,
                    numbers: meter::Numbers::default(),
                    numbers_at: None,
                    graphs: None,
                    record: record_opts.clone(),
                    record_pending: record_now,
                    updater,
                    update_seen: update::State::Off(""),
                    bar: spring::Spring::at(1.0),
                    bar_moving: false,
                    bar_at: Instant::now(),
                    animate: winutil::animations_enabled(),
                    bookmarks: 0,
                    rec_second: None,
                    toast: None,
                    video,
                    audio,
                    settings,
                    focus,
                    hwnd,
                    shown: None,
                    aspect: 0.0,
                    had_signal: false,
                    gpu_drawing: false,
                    cursor: None,
                    last_move: Instant::now(),
                    controls_visible: false,
                    cursor_hidden: false,
                    dragging_volume: false,
                    menu_open: if show_settings {
                        Some(Menu::Settings)
                    } else {
                        show_menu.then_some(Menu::Mode)
                    },
                    show_menu: show_menu || show_settings,
                    updater_at: Instant::now(),
                    window_shown: !start_hidden,
                    placed: !start_hidden,
                    signal_on: false,
                    signal_lost_at: None,
                    fullscreen: false,
                    controls_drawn: false,
                    requests,
                    controls_state: None,
                    pill_bounds: Rc::new(Cell::new(None)),
                    slider_bounds: Rc::new(Cell::new(None)),
                }
            });
            view.update(cx, |v, _| {
                if start_hidden {
                    v.video.gpu.paused.store(true, Ordering::Relaxed);
                    v.audio.muted.store(true, Ordering::Relaxed);
                }
                v.sync_tray();
            });
            // Alt+F4 and friends: to the tray while Passthru waits there, else remember where
            // the window was and quit.
            let weak = view.downgrade();
            window.on_window_should_close(cx, move |window, cx| {
                weak.update(cx, |v, cx| {
                    if v.settings.stays_in_tray() {
                        v.hide_window(window, cx);
                        return false;
                    }
                    v.save_window(window);
                    if let Some(h) = v.hwnd {
                        winutil::set_tray(h, false);
                    }
                    telemetry::stop();
                    true
                })
                .unwrap_or(true)
            });
            view
        })
        .expect("open window");
        if let Ok(viewer) = main_window.update(cx, |_, _, cx| cx.entity()) {
            open_controls_window(viewer, controls_in_capture, cx);
        }
        cx.on_window_closed(|cx| cx.quit()).detach();
        if !start_hidden {
            cx.activate(true);
        }
    });
    telemetry::stop();
}
