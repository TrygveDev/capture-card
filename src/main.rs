#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod convert;
mod gpu;
mod gputest;
mod install;
mod log;
mod meter;
mod overlay;
mod present;
mod settings;
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

const ICON_FONT: &str = "Segoe Fluent Icons";
const ICON_VOLUME: &str = "\u{E995}";
const ICON_VOLUME_LOW: &str = "\u{E993}";
const ICON_MUTE: &str = "\u{E74F}";
const ICON_FULLSCREEN: &str = "\u{E740}";
const ICON_EXIT_FULLSCREEN: &str = "\u{E73F}";
const ICON_PIN: &str = "\u{E718}";
/// Segoe Fluent "PinFill": the pin's solid head, drawn under `ICON_PIN` when pinned.
const ICON_PINNED: &str = "\u{E841}";
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
    /// The capture mode menu above the controls is open.
    menu_open: bool,
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
            || (cursor.is_some() && (over_pill || self.menu_open || self.last_move.elapsed() < IDLE_HIDE));
        if visible != self.controls_visible {
            self.controls_visible = visible;
            if !visible {
                self.pill_bounds.set(None);
                self.menu_open = false;
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
        self.video.hdr.store(self.settings.hdr, Ordering::Relaxed);
        self.settings.save();
        cx.notify();
    }

    /// Cycles frame generation: off, then up to 60, 120 or 240 fps (never past the display's
    /// own refresh rate).
    fn cycle_smooth(&mut self, cx: &mut Context<Self>) {
        self.settings.smooth = match self.settings.smooth {
            0 => 60,
            60 => 120,
            120 => 240,
            _ => 0,
        };
        self.video.gpu.target.store(self.settings.smooth, Ordering::Relaxed);
        self.settings.save();
        cx.notify();
    }

    /// Cycles the FPS counter: off, the counter, the counter with graphs.
    fn cycle_overlay(&mut self, cx: &mut Context<Self>) {
        self.settings.overlay = self.settings.overlay.next();
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
        self.video.control.pick(mode);
        self.settings.mode = mode.map(|(w, h, f)| [w, h, f]);
        self.settings.save();
        self.menu_open = false;
        cx.notify();
    }

    fn set_range(&mut self, range: video::ColourRange, cx: &mut Context<Self>) {
        self.settings.range = range;
        self.video.control.set_range(range);
        self.settings.save();
        cx.notify();
    }

    fn toggle_rtx(&mut self, cx: &mut Context<Self>) {
        self.settings.rtx = !self.settings.rtx;
        self.video.gpu.rtx.store(self.settings.rtx, Ordering::Relaxed);
        self.settings.save();
        cx.notify();
    }

    /// The capture mode menu: what the device offers, and the upscaling switch.
    fn menu(&self, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        let control = &self.video.control;
        let offered = control.modes.lock().unwrap().clone();
        let wanted = *control.wanted.lock().unwrap();
        let header = |text: &'static str| {
            div()
                .px(px(10.))
                .pt(px(8.))
                .pb(px(4.))
                .text_size(px(10.))
                .font_weight(FontWeight::BOLD)
                .text_color(rgba(0xffffff66))
                .child(text)
        };
        let note = |text: String| {
            div().px(px(10.)).pb(px(6.)).text_size(px(11.)).text_color(rgba(0xffffff80)).child(text)
        };
        let item = |id: gpui::ElementId, label: String, detail: Option<String>, checked: bool| {
            div()
                .id(id)
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(10.))
                .py(px(6.))
                .rounded(px(6.))
                .cursor_pointer()
                .hover(|s| s.bg(rgba(0xffffff1f)))
                .child(
                    div()
                        .w(px(14.))
                        .flex_none()
                        .font_family(ICON_FONT)
                        .text_size(px(12.))
                        .text_color(rgb(0xffffff))
                        .child(if checked { ICON_CHECK } else { "" }),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        // Take the row's width, so long details wrap inside the menu.
                        .flex_1()
                        .min_w(px(0.))
                        .child(div().text_size(px(13.)).text_color(rgb(0xffffff)).child(label))
                        .children(detail.map(|d| div().text_size(px(11.)).text_color(rgba(0xffffff80)).child(d))),
                )
        };
        let size = |(w, h, f): video::ModeKey| format!("{w} × {h} at {f} fps");
        let mut menu = div()
            .id("menu")
            .w(px(300.))
            .flex()
            .flex_col()
            .p(px(4.))
            .rounded(px(10.))
            .bg(rgba(0x121212f5))
            .border_1()
            .border_color(rgba(0xffffff1a))
            .child(header("CAPTURE MODE"))
            .child(
                item("mode-auto".into(), "Auto".into(), offered.first().map(|&m| format!("Best offered: {}", size(m))), wanted.is_none())
                    .on_click(cx.listener(|this, _, _, cx| this.pick_mode(None, cx))),
            );
        for (i, &m) in offered.iter().enumerate() {
            menu = menu.child(
                item(("mode", i).into(), video::mode_label(m.0, m.1, m.2), Some(size(m)), wanted == Some(m))
                    .on_click(cx.listener(move |this, _, _, cx| this.pick_mode(Some(m), cx))),
            );
        }
        if let Some(m) = wanted.filter(|m| !offered.contains(m)) {
            menu = menu.child(note(format!(
                "{} is picked but not offered right now, so Auto is used until it is.",
                video::mode_label(m.0, m.1, m.2)
            )));
        }
        if offered.len() == 1 {
            menu = menu.child(note(
                "The card offers only the resolution its input sends. For another one, change the console's video output.".into(),
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
        menu = menu.child(div().h(px(1.)).mx(px(6.)).my(px(4.)).bg(rgba(0xffffff1a))).child(header("COLOUR RANGE"));
        for (choice, label, detail) in ranges {
            menu = menu.child(
                item(("range", choice as usize).into(), label.into(), Some(detail), range == choice)
                    .on_click(cx.listener(move |this, _, _, cx| this.set_range(choice, cx))),
            );
        }
        menu.child(div().h(px(1.)).mx(px(6.)).my(px(4.)).bg(rgba(0xffffff1a)))
            .child(header("UPSCALING"))
            .child(
                item("rtx".into(), "RTX Video Super Resolution".into(), Some(rtx_state), self.settings.rtx)
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_rtx(cx))),
            )
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
fn chip(id: &'static str, label: impl Into<SharedString>, on: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(24.))
        .px(px(7.))
        .mx(px(2.))
        .flex()
        .flex_none()
        .items_center()
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
        .child(label.into())
}

fn divider() -> gpui::Div {
    div().w(px(1.)).h(px(18.)).mx(px(4.)).flex_none().bg(rgba(0xffffff33))
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
        let fullscreen = window.is_fullscreen();
        let muted = self.settings.muted;
        let volume = self.settings.volume;
        let hdr = self.settings.hdr;

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
        let mut mode = self.video.mode.lock().unwrap().clone();
        // How big it's drawn, when that isn't 1:1: "85%" in a smaller window, "133% RTX" when
        // enlarged with super resolution.
        let (src_w, _) = self.video.control.size();
        if !mode.is_empty() && signal && src_w > 0 {
            let pct = (fit_w / src_w as f32 * 100.0).round();
            if pct != 100.0 {
                mode = format!("{mode} · {pct:.0}%");
                if gpu && self.video.gpu.scaling() == present::Scaling::Rtx {
                    mode.push_str(" RTX");
                }
            }
        }
        // While smoothing, show what it's doing: the game's detected rate -> the target.
        let smooth = self.settings.smooth;
        let content = self.video.gpu.content_fps();
        if smooth > 0 && content > 1.0 && content.round() < smooth as f32 && !mode.is_empty() {
            mode = format!("{mode} · {:.0}→{smooth}", content);
        }

        let hdr_button = chip("hdr", "HDR", hdr).on_click(cx.listener(|this, _, _, cx| this.toggle_hdr(cx)));
        let smooth_label = if smooth == 0 { "SMOOTH".to_string() } else { format!("SMOOTH {smooth}") };
        let smooth_button =
            chip("smooth", smooth_label, smooth > 0).on_click(cx.listener(|this, _, _, cx| this.cycle_smooth(cx)));
        let overlay_mode = self.settings.overlay;
        let overlay_label = if overlay_mode == Overlay::Graph { "FPS GRAPH" } else { "FPS" };
        let overlay_button = chip("fps", overlay_label, overlay_mode != Overlay::Off)
            .on_click(cx.listener(|this, _, _, cx| this.cycle_overlay(cx)));
        // Inside the video's top-left corner (clear of black bars).
        let overlay = (signal && overlay_mode != Overlay::Off).then(|| {
            overlay::panel(overlay_mode, &self.numbers, self.graphs.as_ref())
                .absolute()
                .left(logical(fit_x) + px(12.))
                .top(logical(fit_y) + px(12.))
        });

        let menu = self.menu_open.then(|| self.menu(cx));
        let controls = div()
            .absolute()
            .bottom(px(14.))
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
                        if this.menu_open {
                            this.menu_open = false;
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
                            .when(self.menu_open, |d| d.bg(rgba(0xffffff26)))
                            .child(if mode.is_empty() { "Mode".to_string() } else { mode })
                            .child(div().font_family(ICON_FONT).text_size(px(9.)).child(ICON_CHEVRON_UP))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.menu_open = !this.menu_open;
                                cx.notify();
                            })),
                    )
                    .child(divider())
                    .child(
                        icon_button("pin", ICON_PIN)
                            // Pinned: the solid head under the outline, which keeps the needle.
                            .when(self.settings.on_top, |d| {
                                d.relative().bg(rgba(0xffffff26)).child(
                                    div()
                                        .absolute()
                                        .inset_0()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .child(ICON_PINNED),
                                )
                            })
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_pin(cx))),
                    )
                    .child(
                        icon_button("fullscreen", if fullscreen { ICON_EXIT_FULLSCREEN } else { ICON_FULLSCREEN })
                            .on_click(|_, window, _| window.toggle_fullscreen()),
                    )
                    .child(icon_button("close", ICON_CLOSE).on_click(cx.listener(|this, _, window, cx| {
                        this.save_window(window);
                        cx.quit();
                    }))),
                    ),
            );

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
            .on_scroll_wheel(cx.listener(|this, e: &ScrollWheelEvent, _, cx| {
                let dy = e.delta.pixel_delta(px(20.)).y;
                if dy != px(0.) {
                    let step = if dy > px(0.) { 0.05 } else { -0.05 };
                    this.set_volume(this.settings.volume + step, cx);
                    this.settings.save();
                    this.last_move = Instant::now();
                }
            }))
            .on_key_down(cx.listener(|this, e: &KeyDownEvent, window, cx| {
                match e.keystroke.key.as_str() {
                    "f" | "f11" | "enter" => window.toggle_fullscreen(),
                    "escape" if this.menu_open => {
                        this.menu_open = false;
                        cx.notify();
                    }
                    "escape" if window.is_fullscreen() => window.toggle_fullscreen(),
                    "m" => this.toggle_mute(cx),
                    "h" => this.toggle_hdr(cx),
                    "s" => this.cycle_smooth(cx),
                    "o" => this.cycle_overlay(cx),
                    "t" | "p" => this.toggle_pin(cx),
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
            .when(self.controls_visible, |d| d.child(controls))
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
}

fn parse_args() -> Args {
    let mut a = Args {
        list: false,
        snapshot: None,
        video: VideoOptions { device: None, mode: None, test: None, test_size: None },
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

    log::start();
    let mut settings = Settings::load();
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
        gpu: present::Shared::new(settings.smooth, args.stats, settings.rtx),
        control: video::Control::default(),
    });
    // `--mode` for this run, else the one picked in the menu.
    *video.control.wanted.lock().unwrap() = args.video.mode.or(settings.mode.map(|[w, h, f]| (w, h, f)));
    video.control.set_range(settings.range);
    video.meter.set_on(settings.overlay != Overlay::Off);
    let audio = AudioShared::new(settings.volume, settings.muted);
    video::spawn(video.clone(), args.video);
    audio::spawn(audio.clone(), args.audio);
    let (stats_enabled, exit_after, start_fullscreen) = (args.stats, args.exit_after, args.fullscreen);
    let (force_on_top, cpu_only, capture) = (args.on_top, args.cpu, args.capture.clone());
    let show_menu = args.show_menu;

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
                title: Some("Capture Card".into()),
                appears_transparent: true,
                traffic_light_position: None,
            }),
            window_min_size: Some(size(px(320.), px(180.))),
            window_background: WindowBackgroundAppearance::Opaque,
            app_id: Some("capture-card".into()),
            ..Default::default()
        };
        cx.open_window(options, |window, cx| {
            let hwnd = winutil::hwnd(window);
            let (events_tx, mut events) = futures::channel::mpsc::unbounded();
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
                        let ok = this.update_in(cx, |this: &mut Viewer, _, cx| match event {
                            winutil::Event::PressOutside => {
                                if this.menu_open {
                                    this.menu_open = false;
                                    cx.notify();
                                }
                            }
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
                    menu_open: show_menu,
                    show_menu,
                    controls_state: None,
                    pill_bounds: Rc::new(Cell::new(None)),
                    slider_bounds: Rc::new(Cell::new(None)),
                }
            });
            // Alt+F4 and friends: remember where the window was.
            let weak = view.downgrade();
            window.on_window_should_close(cx, move |window, cx| {
                let _ = weak.update(cx, |v, _| v.save_window(window));
                true
            });
            view
        })
        .expect("open window");
        cx.on_window_closed(|cx| cx.quit()).detach();
        cx.activate(true);
    });
}
