//! How much GPU smoothing spends on clean edges: as much as there's room for, found out live.
//!
//! The levels differ in how the frame's last pass clears up grain where motion couldn't be
//! matched (gpu.hlsl, `clean`): not at all, over one ring of neighbours, or over two. The GPU
//! times its own work (`gpu::Timer`), so the resolution, the target rate, the card and whatever
//! else keeps it busy all count, whoever made it.
//!
//! Only a pair's motion and first made frame hold up a present, so that frame never gets more
//! than `FIRST_EXTRA` over the cheapest level, and what clean edges add there comes out of the
//! present's slack rather than moving the schedule (`Governor::slack_used`): a better level
//! costs GPU, not delay.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Quality {
    Light,
    Balanced,
    #[default]
    High,
}

impl Quality {
    pub const ALL: [Quality; 3] = [Quality::Light, Quality::Balanced, Quality::High];

    pub fn name(self) -> &'static str {
        ["Light", "Balanced", "High"][self as usize]
    }

    fn up(self) -> Option<Quality> {
        Self::ALL.get(self as usize + 1).copied()
    }

    fn down(self) -> Option<Quality> {
        (self as usize).checked_sub(1).map(|i| Self::ALL[i])
    }
}

/// Development: `PASSTHRU_QUALITY=light|balanced|high` holds one level.
pub fn forced() -> Option<Quality> {
    let v = std::env::var("PASSTHRU_QUALITY").ok()?;
    Quality::ALL.into_iter().find(|q| q.name().eq_ignore_ascii_case(&v))
}

/// What a made frame costs at each level against Light (measured at 4K and 1080p on easy and
/// hard clips, a little on the high side).
const RELATIVE: [f64; 3] = [1.0, 1.3, 1.9];
/// Up only while the next level would take under this share of the time a picture lasts, and
/// each frame under this share of its own step.
const ROOM: f64 = 0.65;
/// Down past this share.
const FULL: f64 = 0.8;
/// A new picture's motion and first frame land in one step along with a frame being made
/// ahead, and all of it has to be done before that step's vblank: up only while that's under
/// this share of a step, down past the second. (At 4K 30 -> 240 High took 0.97 of the 4.2 ms
/// step, and 7% of the frames came a refresh late.)
const BURST_ROOM: f64 = 0.8;
const BURST_FULL: f64 = 0.9;
/// The pair's work, as the presenter sees it (waiting for the GPU included), taking this share
/// of the picture's time means the GPU is busy elsewhere.
const BUSY: f64 = 0.7;
/// The most the frame a present waits for may cost over Light.
const FIRST_EXTRA: f64 = 0.0005;
/// The share of the motion's time that filling what leaves the picture and finding overlays
/// take (13% at 4K, the more the faster things move).
const EDGES: f64 = 0.13;
/// The most that may come out of a present's slack: it goes out 0.3 into a refresh, so it has
/// 2.9 ms until the vblank at 240 Hz, more on slower displays, and 2 ms of margin before that.
const MOST_SLACK: f64 = 0.0007;
/// Room this long before going up.
const PATIENCE: Duration = Duration::from_secs(1);
const MOST_PATIENCE: Duration = Duration::from_secs(300);
/// No room this long before going down: a single slow pair is nothing.
const OVER_FOR: Duration = Duration::from_millis(250);
/// Going down this soon after going up was that move's fault: wait longer before the next.
const BLAME: Duration = Duration::from_secs(10);
/// A level held this long has proved itself.
const PROVED: Duration = Duration::from_secs(60);

/// Work the GPU timed.
#[derive(Clone, Copy, Debug)]
pub enum Work {
    Motion,
    Frame(Quality),
}

/// The levels of one pair's frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The first made frame, which the present waits for.
    pub first: Quality,
    pub rest: Quality,
}

/// A pair about to be made.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    /// Frames to make.
    pub frames: u32,
    /// How long the picture lasts, seconds.
    pub span: f64,
    /// The last pairs' time from submitting to done, seconds (motion and the first frame).
    pub waited: f64,
}

pub struct Governor {
    forced: Option<Quality>,
    level: Quality,
    /// Seconds, smoothed.
    motion: Option<f64>,
    /// A made frame's seconds at Light (those at other levels scaled to it), smoothed.
    frame: Option<f64>,
    room_since: Option<Instant>,
    over_since: Option<Instant>,
    went_up: Option<Instant>,
    patience: Duration,
}

/// Quick to grow, slow to shrink, like the presenter's GPU time.
fn smooth(old: &mut Option<f64>, new: f64) {
    *old = Some(match *old {
        None => new,
        Some(o) if new > o => o * 0.7 + new * 0.3,
        Some(o) => o * 0.95 + new * 0.05,
    });
}

impl Governor {
    pub fn new(forced: Option<Quality>) -> Self {
        Governor {
            forced,
            level: forced.unwrap_or(Quality::Light),
            motion: None,
            frame: None,
            room_since: None,
            over_since: None,
            went_up: None,
            patience: PATIENCE,
        }
    }

    pub fn level(&self) -> Quality {
        self.level
    }

    /// Starts over: another size or format costs something else.
    pub fn reset(&mut self) {
        *self = Governor::new(self.forced);
    }

    pub fn measured(&mut self, work: Work, took: Duration) {
        let s = took.as_secs_f64();
        match work {
            Work::Motion => smooth(&mut self.motion, s),
            Work::Frame(q) => smooth(&mut self.frame, s / RELATIVE[q as usize]),
        }
    }

    fn first(&self, level: Quality, light: f64) -> Quality {
        let fits = |q: &Quality| *q <= level && light * (RELATIVE[*q as usize] - 1.0) <= FIRST_EXTRA;
        Quality::ALL.into_iter().rev().find(fits).unwrap_or(Quality::Light)
    }

    /// How much of the work a present waits for is clean edges: the presenter plans as if it
    /// weren't there. Its checks are all or nothing, so a quarter of a millisecond more
    /// otherwise had nine pictures in ten wait for the present before them (3.5 ms of delay).
    pub fn slack_used(&self) -> Duration {
        let edges = self.motion.map_or(0.0, |m| m * EDGES);
        let first = self.frame.map_or(0.0, |f| f * (RELATIVE[self.first(self.level, f) as usize] - 1.0));
        Duration::from_secs_f64((edges + first).min(MOST_SLACK))
    }

    /// Whether `level` stays within the limits for this pair: of the picture's time and of one
    /// step (`share`), and what lands together in one step (`burst`).
    fn fits(&self, level: Quality, pair: &Pair, motion: f64, light: f64, (share, burst): (f64, f64)) -> bool {
        let frame = |q: Quality| light * RELATIVE[q as usize];
        let rest = pair.frames.saturating_sub(1);
        let first = frame(self.first(level, light));
        let work = motion + first + rest as f64 * frame(level);
        let step = pair.span / (pair.frames + 1) as f64;
        let together = motion + first + if rest > 0 { frame(level) } else { 0.0 };
        work <= share * pair.span && frame(level) <= share * step && together <= burst * step
    }

    /// The levels for `pair`, moving up or down as the measurements say.
    pub fn plan(&mut self, pair: &Pair, now: Instant) -> Plan {
        if let Some(q) = self.forced {
            return Plan { first: q, rest: q };
        }
        let (Some(motion), Some(light)) = (self.motion, self.frame) else {
            // Nothing measured (yet, or the GPU has no timestamps).
            return Plan { first: self.level, rest: self.level };
        };
        let busy = pair.waited > BUSY * pair.span;
        if busy || !self.fits(self.level, pair, motion, light, (FULL, BURST_FULL)) {
            self.room_since = None;
            let since = *self.over_since.get_or_insert(now);
            if let Some(down) = self.level.down()
                && now.saturating_duration_since(since) >= OVER_FOR
            {
                self.patience = match self.went_up.take() {
                    Some(t) if now.saturating_duration_since(t) < BLAME => (self.patience * 2).min(MOST_PATIENCE),
                    _ => self.patience,
                };
                self.level = down;
                self.over_since = None;
            }
        } else {
            self.over_since = None;
            if self.went_up.is_some_and(|t| now.saturating_duration_since(t) >= PROVED) {
                self.went_up = None;
                self.patience = PATIENCE;
            }
            match self.level.up() {
                Some(up) if self.fits(up, pair, motion, light, (ROOM, BURST_ROOM)) => {
                    let since = *self.room_since.get_or_insert(now);
                    if now.saturating_duration_since(since) >= self.patience {
                        self.level = up;
                        self.went_up = Some(now);
                        self.room_since = None;
                    }
                }
                _ => self.room_since = None,
            }
        }
        Plan { first: self.first(self.level, light), rest: self.level }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU: what motion and a frame at Light take it, in milliseconds.
    #[derive(Clone, Copy)]
    struct Card {
        motion: f64,
        light: f64,
    }

    /// Runs `secs` of pairs, returning the last plan and how often the level moved.
    fn run(g: &mut Governor, card: Card, span_ms: f64, frames: u32, from: &mut Instant, secs: f64) -> (Plan, u32) {
        let (mut plan, mut moves) = (Plan { first: g.level, rest: g.level }, 0);
        for _ in 0..(secs * 1000.0 / span_ms) as u32 {
            let cost = |q: Quality| card.light * RELATIVE[q as usize];
            g.measured(Work::Motion, Duration::from_secs_f64(card.motion / 1000.0));
            g.measured(Work::Frame(plan.first), Duration::from_secs_f64(cost(plan.first) / 1000.0));
            if frames > 1 {
                g.measured(Work::Frame(plan.rest), Duration::from_secs_f64(cost(plan.rest) / 1000.0));
            }
            let waited = (card.motion + cost(plan.first)) / 1000.0;
            let was = g.level;
            plan = g.plan(&Pair { frames, span: span_ms / 1000.0, waited }, *from);
            moves += (g.level != was) as u32;
            *from += Duration::from_secs_f64(span_ms / 1000.0);
        }
        (plan, moves)
    }

    const PICTURE_30: f64 = 1000.0 / 30.0;

    #[test]
    fn a_strong_card_gets_high_without_the_present_waiting_for_it() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        let fast_4k = Card { motion: 2.3, light: 1.2 };
        let (plan, moves) = run(&mut g, fast_4k, PICTURE_30, 3, &mut t, 5.0);
        assert_eq!((plan.first, plan.rest, moves), (Quality::Balanced, Quality::High, 2));
        // At 1080p High costs the first frame under half a millisecond too.
        g.reset();
        let (plan, _) = run(&mut g, Card { motion: 2.2, light: 0.3 }, PICTURE_30, 3, &mut t, 5.0);
        assert_eq!((plan.first, plan.rest), (Quality::High, Quality::High));
    }

    #[test]
    fn a_weak_card_stays_light() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        let (plan, moves) = run(&mut g, Card { motion: 14.0, light: 5.0 }, PICTURE_30, 3, &mut t, 30.0);
        assert_eq!((plan.first, plan.rest, moves), (Quality::Light, Quality::Light, 0));
    }

    #[test]
    fn what_lands_in_one_step_must_fit_it() {
        // An RTX 5080 at 4K, as measured. At 240 fps a step is 4.2 ms: room in the picture as a
        // whole, but High there had 7% of the frames come a refresh late.
        let measured = Card { motion: 1.6, light: 0.86 };
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        let (plan, moves) = run(&mut g, measured, PICTURE_30, 7, &mut t, 10.0);
        assert_eq!((plan.rest, moves), (Quality::Light, 0));
        // At 120 the same card has the room, and at 1080p it has it at 240 too.
        g.reset();
        assert_eq!(run(&mut g, measured, PICTURE_30, 3, &mut t, 5.0).0.rest, Quality::High);
        g.reset();
        assert_eq!(run(&mut g, Card { motion: 1.5, light: 0.22 }, PICTURE_30, 7, &mut t, 5.0).0.rest, Quality::High);
    }

    #[test]
    fn switching_to_240_backs_off_from_high() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        let measured = Card { motion: 1.6, light: 0.86 };
        run(&mut g, measured, PICTURE_30, 3, &mut t, 5.0);
        assert_eq!(g.level(), Quality::High);
        assert!(run(&mut g, measured, PICTURE_30, 7, &mut t, 1.5).0.rest < Quality::High);
    }

    #[test]
    fn a_gpu_getting_busy_backs_off_at_once() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        run(&mut g, Card { motion: 2.3, light: 1.2 }, PICTURE_30, 3, &mut t, 5.0);
        assert_eq!(g.level(), Quality::High);
        let (plan, _) = run(&mut g, Card { motion: 10.0, light: 5.5 }, PICTURE_30, 3, &mut t, 1.5);
        assert_eq!(plan.rest, Quality::Light);
        let (plan, _) = run(&mut g, Card { motion: 2.3, light: 1.2 }, PICTURE_30, 3, &mut t, 10.0);
        assert_eq!(plan.rest, Quality::High);
    }

    #[test]
    fn waiting_for_the_gpu_counts_as_busy() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        run(&mut g, Card { motion: 2.3, light: 1.2 }, PICTURE_30, 3, &mut t, 5.0);
        for _ in 0..30 {
            g.plan(&Pair { frames: 3, span: PICTURE_30 / 1000.0, waited: 0.028 }, t);
            t += Duration::from_secs_f64(PICTURE_30 / 1000.0);
        }
        assert_eq!(g.level(), Quality::Light);
    }

    #[test]
    fn it_does_not_flip_flop() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        // Room for High only while nothing else runs, which comes and goes every 2 s.
        let mut moves = 0;
        for i in 0..300 {
            let card = if i % 2 == 0 { Card { motion: 2.3, light: 1.2 } } else { Card { motion: 8.0, light: 4.2 } };
            moves += run(&mut g, card, PICTURE_30, 3, &mut t, 2.0).1;
        }
        assert!(moves < 120, "{moves} moves in ten minutes");
    }

    #[test]
    fn what_comes_out_of_the_slack_is_small() {
        let (mut g, mut t) = (Governor::new(None), Instant::now());
        assert_eq!(g.slack_used(), Duration::ZERO, "nothing measured, nothing assumed");
        run(&mut g, Card { motion: 2.3, light: 1.2 }, PICTURE_30, 3, &mut t, 5.0);
        let used = g.slack_used().as_secs_f64() * 1000.0;
        assert!((0.3..=0.7).contains(&used), "{used} ms");
        g.reset();
        run(&mut g, Card { motion: 25.0, light: 9.0 }, PICTURE_30, 3, &mut t, 2.0);
        assert!(g.slack_used().as_secs_f64() <= MOST_SLACK, "a slow GPU's slack isn't eaten");
    }

    #[test]
    fn forced_holds() {
        let (mut g, mut t) = (Governor::new(Some(Quality::Balanced)), Instant::now());
        let (plan, moves) = run(&mut g, Card { motion: 30.0, light: 9.0 }, PICTURE_30, 3, &mut t, 3.0);
        assert_eq!((plan.first, plan.rest, moves), (Quality::Balanced, Quality::Balanced, 0));
    }
}
