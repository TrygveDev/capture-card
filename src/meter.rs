//! Measurements for the FPS counter: when frames reached the screen, when the card's frames
//! arrived, and whether each of those carried a new picture from the game. Only kept while
//! the counter is on.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::quality::Quality;

/// Graph columns, each covering `COLUMN`: 4.8 seconds in all.
pub const COLUMNS: usize = 240;
const COLUMN: Duration = Duration::from_millis(20);
/// Rates are over the last second.
const RATE_OVER: Duration = Duration::from_secs(1);
/// History kept: the graphs' span and a little more.
const KEEP: Duration = Duration::from_secs(6);
/// Graph heights in milliseconds. Each graph takes the smallest that's at least twice its
/// usual frame time, so that sits at or under half height and a frame twice as long still fits.
const SCALES: [f32; 7] = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0];

pub struct Meter {
    on: AtomicBool,
    /// GPU time for a pair of pictures while smoothing, f32 milliseconds (0: not smoothing).
    gpu_ms: AtomicU32,
    /// Smoothing's quality level, its index plus 1 (0: not known).
    quality: AtomicU32,
    /// Graph columns are whole `COLUMN`s from here, so they keep their contents as they scroll.
    epoch: Instant,
    history: Mutex<History>,
}

#[derive(Default)]
struct History {
    /// Frames on screen: when they got there (estimated) and whether they were generated.
    shown: VecDeque<(Instant, bool)>,
    /// Card frames: when they arrived and whether the game's picture had changed.
    card: VecDeque<(Instant, Option<bool>)>,
    /// The newest card frame wasn't checked for a new picture: the CPU path doesn't check
    /// (and draws the first frames, before the GPU path is up).
    unchecked: bool,
    /// Real pictures: when they reached the screen and how long after they arrived.
    delay: VecDeque<(Instant, Duration)>,
}

impl History {
    fn trim(&mut self, now: Instant) {
        let old = |t: Instant| now.saturating_duration_since(t) > KEEP;
        while self.shown.front().is_some_and(|e| old(e.0)) {
            self.shown.pop_front();
        }
        while self.card.front().is_some_and(|e| old(e.0)) {
            self.card.pop_front();
        }
        while self.delay.front().is_some_and(|e| old(e.0)) {
            self.delay.pop_front();
        }
    }

    fn pictures(&self) -> impl Iterator<Item = Instant> + '_ {
        self.card.iter().filter(|e| e.1 == Some(true)).map(|e| e.0)
    }
}

/// The counter's numbers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Numbers {
    /// Frames on screen per second.
    pub fps: f32,
    /// Average time between frames on screen, milliseconds.
    pub frame_ms: f32,
    /// 1% low: frames per second at the slowest 1% of frame times over the graphs' span.
    pub low: Option<f32>,
    /// New pictures from the game per second, where that's checked.
    pub game: Option<f32>,
    /// Frames per second from the card.
    pub card: f32,
    /// Share of the frames on screen that were generated.
    pub generated: f32,
    /// Average time from a real picture's arrival to the screen, milliseconds.
    pub delay_ms: Option<f32>,
    /// GPU time for a pair of pictures while smoothing, milliseconds.
    pub gpu_ms: Option<f32>,
    /// The quality level smoothing is running at.
    pub quality: Option<Quality>,
}

/// A frame-time graph, oldest column first: the shortest and longest frame time in each.
#[derive(Clone, Debug)]
pub struct Graph {
    pub columns: Vec<Option<(f32, f32)>>,
    /// Frame time at the top, milliseconds.
    pub top: f32,
}

impl Meter {
    pub fn new() -> Self {
        Meter {
            on: AtomicBool::new(false),
            gpu_ms: AtomicU32::new(0),
            quality: AtomicU32::new(0),
            epoch: Instant::now(),
            history: Mutex::new(History::default()),
        }
    }

    pub fn is_on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// Starts or stops measuring. Starting forgets what was measured before.
    pub fn set_on(&self, on: bool) {
        if on && !self.is_on() {
            *self.history.lock().unwrap() = History::default();
        }
        self.on.store(on, Ordering::Relaxed);
    }

    /// A frame reached the screen at `at`; real pictures say when they arrived.
    pub fn shown(&self, at: Instant, generated: bool, arrived: Option<Instant>) {
        if !self.is_on() {
            return;
        }
        let mut h = self.history.lock().unwrap();
        h.shown.push_back((at, generated));
        if let Some(a) = arrived {
            h.delay.push_back((at, at.saturating_duration_since(a)));
        }
        h.trim(at);
    }

    /// A frame came from the card at `at`, with a new picture or a repeat (None: not checked).
    pub fn card_frame(&self, at: Instant, new_picture: Option<bool>) {
        if !self.is_on() {
            return;
        }
        let mut h = self.history.lock().unwrap();
        h.card.push_back((at, new_picture));
        h.unchecked = new_picture.is_none();
        h.trim(at);
    }

    pub fn set_gpu(&self, t: Option<Duration>) {
        let ms = t.map_or(0.0, |t| t.as_secs_f32() * 1000.0);
        self.gpu_ms.store(ms.to_bits(), Ordering::Relaxed);
    }

    pub fn set_quality(&self, q: Option<Quality>) {
        self.quality.store(q.map_or(0, |q| q as u32 + 1), Ordering::Relaxed);
    }

    pub fn numbers(&self, now: Instant) -> Numbers {
        let h = self.history.lock().unwrap();
        let recent = |t: Instant| now.saturating_duration_since(t) <= RATE_OVER;
        let fps = rate(h.shown.iter().map(|e| e.0), now);
        let (n, made) = h.shown.iter().filter(|e| recent(e.0)).fold((0u32, 0u32), |(n, g), e| (n + 1, g + e.1 as u32));
        let delays: Vec<f32> = h.delay.iter().filter(|e| recent(e.0)).map(|e| e.1.as_secs_f32() * 1000.0).collect();
        let span = COLUMN * COLUMNS as u32;
        let mut times: Vec<f32> = intervals(h.shown.iter().map(|e| e.0))
            .into_iter()
            .filter(|e| now.saturating_duration_since(e.0) <= span)
            .map(|e| e.1)
            .collect();
        let low = (times.len() >= 20).then(|| {
            times.sort_by(f32::total_cmp);
            let i = ((times.len() as f32 * 0.99).ceil() as usize).clamp(1, times.len()) - 1;
            1000.0 / times[i].max(0.001)
        });
        let gpu = f32::from_bits(self.gpu_ms.load(Ordering::Relaxed));
        Numbers {
            fps,
            frame_ms: if fps > 0.0 { 1000.0 / fps } else { 0.0 },
            low,
            game: (!h.unchecked && !h.card.is_empty()).then(|| rate(h.pictures(), now)),
            card: rate(h.card.iter().map(|e| e.0), now),
            generated: if n > 0 { made as f32 / n as f32 } else { 0.0 },
            delay_ms: (!delays.is_empty()).then(|| delays.iter().sum::<f32>() / delays.len() as f32),
            gpu_ms: (gpu > 0.0).then_some(gpu),
            quality: (self.quality.load(Ordering::Relaxed) as usize).checked_sub(1).and_then(|i| Quality::ALL.get(i).copied()),
        }
    }

    /// Frame times on screen, and of the game's pictures where those are checked.
    pub fn graphs(&self, now: Instant) -> (Graph, Option<Graph>) {
        let h = self.history.lock().unwrap();
        // The right edge, on a whole column.
        let n = now.saturating_duration_since(self.epoch).as_nanos() / COLUMN.as_nanos();
        let end = self.epoch + Duration::from_nanos((n * COLUMN.as_nanos()) as u64);
        let screen = graph(&intervals(h.shown.iter().map(|e| e.0)), end);
        let game = (!h.unchecked).then(|| graph(&intervals(h.pictures()), end));
        (screen, game)
    }
}

/// Events a second over the last second. A steady stream goes by its spacing, so 30 a second
/// reads 30 and not 30 or 31 depending on where the second's edge falls. A stream that has
/// stopped goes by its count, so it falls to 0.
fn rate(times: impl Iterator<Item = Instant>, now: Instant) -> f32 {
    let (mut first, mut last, mut n) = (None, None, 0u32);
    for t in times.filter(|&t| now.saturating_duration_since(t) <= RATE_OVER) {
        first.get_or_insert(t);
        last = Some(t);
        n += 1;
    }
    let (Some(first), Some(last)) = (first, last) else {
        return 0.0;
    };
    if n >= 2 {
        let span = last.saturating_duration_since(first).as_secs_f64();
        let each = span / (n - 1) as f64;
        if span > 0.0 && now.saturating_duration_since(last).as_secs_f64() <= 2.0 * each + 0.05 {
            return ((n - 1) as f64 / span) as f32;
        }
    }
    n as f32 / RATE_OVER.as_secs_f32()
}

/// Times between consecutive events: when the later one happened, and milliseconds.
fn intervals(times: impl Iterator<Item = Instant>) -> Vec<(Instant, f32)> {
    let mut prev: Option<Instant> = None;
    let mut out = Vec::new();
    for t in times {
        if let Some(p) = prev {
            out.push((t, t.saturating_duration_since(p).as_secs_f32() * 1000.0));
        }
        prev = Some(t);
    }
    out
}

/// `later - earlier` in seconds, negative if `later` is earlier.
fn secs_between(later: Instant, earlier: Instant) -> f64 {
    match later.checked_duration_since(earlier) {
        Some(d) => d.as_secs_f64(),
        None => -(earlier - later).as_secs_f64(),
    }
}

/// Spreads frame times over the columns ending at `end`. Each frame time covers the span
/// from the frame before to its own, so a long one fills every column it spans.
fn graph(samples: &[(Instant, f32)], end: Instant) -> Graph {
    let mut columns = vec![None::<(f32, f32)>; COLUMNS];
    let col = COLUMN.as_secs_f64();
    let mut seen = Vec::new();
    for &(t, ms) in samples {
        // Not on screen yet: it joins once its column is complete.
        if t > end {
            continue;
        }
        let right = COLUMNS as f64 - secs_between(end, t) / col;
        let left = right - ms as f64 / 1000.0 / col;
        if right <= 0.0 {
            continue;
        }
        seen.push(ms);
        let (a, b) = (left.max(0.0).floor() as usize, (right.ceil() as usize).min(COLUMNS));
        for c in &mut columns[a..b] {
            *c = Some(c.map_or((ms, ms), |(lo, hi)| (lo.min(ms), hi.max(ms))));
        }
    }
    let top = if seen.is_empty() {
        40.0
    } else {
        seen.sort_by(f32::total_cmp);
        let usual = seen[seen.len() / 2];
        SCALES.iter().copied().find(|&s| s >= 2.0 * usual - 0.01).unwrap_or(SCALES[SCALES.len() - 1])
    };
    Graph { columns, top }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A meter that has seen `secs` seconds of a 30 fps game on a 60 Hz card, smoothed to 120
    /// (three of every four frames on screen generated), and that moment.
    fn smoothed(secs: u32) -> (Meter, Instant) {
        let m = Meter::new();
        m.set_on(true);
        let start = Instant::now();
        let screen = Duration::from_secs_f64(1.0 / 120.0);
        for i in 0..secs * 120 {
            let t = start + screen * i;
            // Card frames every other step, a new picture every other card frame.
            if i % 2 == 0 {
                m.card_frame(t, Some(i % 4 == 0));
            }
            m.shown(t + Duration::from_millis(30), i % 4 != 0, (i % 4 == 0).then_some(t));
        }
        (m, start + screen * (secs * 120 - 1) + Duration::from_millis(30))
    }

    #[test]
    fn rates_of_a_smoothed_30fps_game() {
        let (m, now) = smoothed(3);
        let n = m.numbers(now);
        assert!((n.fps - 120.0).abs() < 0.5, "{n:?}");
        assert!((n.frame_ms - 8.33).abs() < 0.05, "{n:?}");
        assert!((n.card - 60.0).abs() < 0.5, "{n:?}");
        assert!((n.game.unwrap() - 30.0).abs() < 0.5, "{n:?}");
        assert!((n.generated - 0.75).abs() < 0.02, "{n:?}");
        assert!((n.delay_ms.unwrap() - 30.0).abs() < 0.5, "{n:?}");
        assert!((n.low.unwrap() - 120.0).abs() < 1.0, "{n:?}");
    }

    #[test]
    fn a_stopped_stream_falls_off() {
        let (m, now) = smoothed(2);
        assert!(m.numbers(now + Duration::from_millis(500)).fps < 70.0);
        assert_eq!(m.numbers(now + Duration::from_millis(1500)).fps, 0.0);
    }

    #[test]
    fn one_percent_low_catches_hitches() {
        let m = Meter::new();
        m.set_on(true);
        let start = Instant::now();
        let mut t = start;
        for i in 0..480 {
            t += Duration::from_secs_f64(if i % 100 == 50 { 0.025 } else { 1.0 / 120.0 });
            m.shown(t, false, None);
        }
        let low = m.numbers(t).low.unwrap();
        assert!((low - 40.0).abs() < 1.0, "{low}");
    }

    #[test]
    fn graphs_fill_and_scale() {
        let (m, now) = smoothed(6);
        let (screen, game) = m.graphs(now);
        let game = game.unwrap();
        assert_eq!(screen.top, 20.0);
        assert_eq!(game.top, 80.0);
        // Every column but the newest has frames; on screen they're all 1/120 s.
        assert!(screen.columns[..COLUMNS - 2].iter().all(|c| c.is_some_and(|(lo, hi)| (lo - 8.33).abs() < 0.05 && (hi - 8.33).abs() < 0.05)));
        assert!(game.columns[..COLUMNS - 3].iter().all(|c| c.is_some_and(|(_, hi)| (hi - 33.3).abs() < 0.1)));
    }

    #[test]
    fn unchecked_frames_have_no_game_rate() {
        let m = Meter::new();
        m.set_on(true);
        let t = Instant::now();
        m.card_frame(t, None);
        m.card_frame(t + Duration::from_millis(16), None);
        assert!(m.numbers(t + Duration::from_millis(16)).game.is_none());
        assert!(m.graphs(t).1.is_none());
        // The GPU path takes over (it checks): the game's rate is back.
        let mut at = t + Duration::from_millis(16);
        for i in 0..60 {
            at += Duration::from_secs_f64(1.0 / 60.0);
            m.card_frame(at, Some(i % 2 == 0));
        }
        assert!((m.numbers(at).game.unwrap() - 30.0).abs() < 0.5);
        assert!(m.graphs(at).1.is_some());
    }

    #[test]
    fn nothing_kept_while_off() {
        let m = Meter::new();
        let t = Instant::now();
        m.shown(t, false, None);
        m.set_on(true);
        assert_eq!(m.numbers(t).fps, 0.0);
    }
}
