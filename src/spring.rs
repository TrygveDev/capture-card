//! A damped spring for UI motion: position and velocity, stepped every frame towards a target.
//! Retargeting mid-flight keeps the velocity, so a bar sliding out turns around smoothly when
//! the pointer comes back instead of jumping.

/// Stiffness and damping ratio: a little under critical damping, so it settles with a small
/// overshoot (about 350 ms).
const STIFFNESS: f32 = 380.0;
const DAMPING_RATIO: f32 = 0.72;
/// Integration steps no longer than this (frames can be late).
const MAX_STEP: f32 = 0.004;

#[derive(Clone, Copy, Debug)]
pub struct Spring {
    pub pos: f32,
    vel: f32,
}

impl Spring {
    pub fn at(pos: f32) -> Spring {
        Spring { pos, vel: 0.0 }
    }

    /// Moves `dt` seconds towards `target`; returns whether it's still moving.
    pub fn step(&mut self, target: f32, dt: f32) -> bool {
        let damping = 2.0 * DAMPING_RATIO * STIFFNESS.sqrt();
        let mut left = dt.clamp(0.0, 0.1);
        while left > 0.0 {
            let h = left.min(MAX_STEP);
            // Semi-implicit Euler: stable at these step sizes.
            self.vel += (-STIFFNESS * (self.pos - target) - damping * self.vel) * h;
            self.pos += self.vel * h;
            left -= h;
        }
        if (self.pos - target).abs() < 0.001 && self.vel.abs() < 0.01 {
            self.pos = target;
            self.vel = 0.0;
            return false;
        }
        true
    }

    /// Jumps to `target` (animations turned off in Windows).
    pub fn snap(&mut self, target: f32) {
        self.pos = target;
        self.vel = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(s: &mut Spring, target: f32, secs: f32) -> (f32, f32) {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let mut t = 0.0;
        while t < secs {
            s.step(target, 1.0 / 240.0);
            lo = lo.min(s.pos);
            hi = hi.max(s.pos);
            t += 1.0 / 240.0;
        }
        (lo, hi)
    }

    #[test]
    fn settles_on_its_target_with_a_small_overshoot() {
        let mut s = Spring::at(1.0);
        let (lo, _) = run(&mut s, 0.0, 0.6);
        assert_eq!(s.pos, 0.0, "settled within 0.6 s");
        assert!(lo < 0.0 && lo > -0.08, "a small overshoot past the target: {lo}");
    }

    #[test]
    fn turning_back_mid_flight_is_smooth() {
        let mut s = Spring::at(0.0);
        run(&mut s, 1.0, 0.08);
        let (mid, before) = (s.pos, s);
        assert!(mid > 0.1 && mid < 0.9);
        // The first step after retargeting moves on from where it was, with its speed.
        s.step(0.0, 1.0 / 240.0);
        assert!((s.pos - before.pos).abs() < 0.05);
        run(&mut s, 0.0, 0.8);
        assert_eq!(s.pos, 0.0);
    }

    #[test]
    fn long_frames_stay_stable() {
        let mut s = Spring::at(1.0);
        for _ in 0..20 {
            s.step(0.0, 0.1);
        }
        assert_eq!(s.pos, 0.0);
    }
}
