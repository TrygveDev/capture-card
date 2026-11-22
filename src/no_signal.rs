//! The card's own "no signal" screen, so it isn't taken for a picture.
//!
//! With nothing on its input the Elgato 4K S sends black for about 7 seconds and then a still
//! slate (NO SIGNAL over a dark gradient), in whatever mode it's open in. "Is it moving?" can't
//! tell that from a paused game, so the slate itself is recognised: its brightness on a 32x18
//! grid (measured in six modes, 4:3 to 21:9 and up to 4K, within 0.4 levels of each other), and
//! the cells its words are in on their own (so a dark scene doesn't pass for it). The closest of
//! 1994 frames of recorded play was 5 levels off, and 37 in the words.

const COLS: usize = 32;
const ROWS: usize = 18;
/// Points looked at per cell, each way.
const POINTS: usize = 8;
/// How far off a match may be: the whole grid's level, each cell after that, and the words.
const MAX_OFFSET: f32 = 10.0;
const MAX_MEAN: f32 = 2.0;
const MAX_WORDS: f32 = 12.0;
/// Cells at least this bright are the words'.
const WORDS: u8 = 64;

#[rustfmt::skip]
const ELGATO_4K_S: [[u8; COLS]; ROWS] = [
    [20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35],
    [20, 21, 21, 22, 22, 23, 23, 24, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35],
    [20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 29, 30, 30, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35],
    [21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35],
    [21, 22, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 29, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36],
    [21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 26, 27, 28, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 36],
    [22, 22, 23, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36],
    [22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 34, 35, 35, 36, 36],
    [22, 23, 23, 24, 24, 25, 25, 25, 26, 26, 27, 27, 56, 66, 60, 65, 65, 72, 56, 44, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37],
    [22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 50, 62, 51, 65, 65, 65, 69, 55, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 37],
    [23, 23, 24, 24, 25, 25, 25, 26, 26, 27, 27, 28, 28, 29, 42, 38, 34, 36, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37],
    [23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 69, 72, 70, 64, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37],
    [23, 24, 24, 25, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 34, 34, 34, 35, 35, 36, 36, 37, 37, 38],
    [23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 29, 30, 30, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38],
    [24, 24, 25, 25, 26, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38],
    [24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 38],
    [24, 25, 25, 26, 26, 27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 35, 36, 36, 37, 37, 38, 38, 39],
    [24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39],
];

/// Whether a `w` x `h` picture is the slate; `luma(x, y)` reads its brightness.
pub fn is_slate(w: usize, h: usize, luma: impl Fn(usize, usize) -> u8) -> bool {
    let mut diff = [0.0f32; COLS * ROWS];
    for (i, d) in diff.iter_mut().enumerate() {
        let (col, row) = (i % COLS, i / COLS);
        let mut sum = 0u32;
        for j in 0..POINTS {
            let y = (2 * (row * POINTS + j) + 1) * h / (2 * ROWS * POINTS);
            for k in 0..POINTS {
                let x = (2 * (col * POINTS + k) + 1) * w / (2 * COLS * POINTS);
                sum += luma(x, y) as u32;
            }
        }
        *d = sum as f32 / (POINTS * POINTS) as f32 - ELGATO_4K_S[row][col] as f32;
    }
    // The level as a whole moves a little with the mode and colour range.
    let mut sorted = diff;
    sorted.sort_unstable_by(f32::total_cmp);
    let offset = sorted[COLS * ROWS / 2];
    let (mut all, mut words, mut n) = (0.0, 0.0, 0);
    for (d, &r) in diff.iter().zip(ELGATO_4K_S.as_flattened()) {
        let e = (d - offset).abs();
        all += e;
        if r >= WORDS {
            words += e;
            n += 1;
        }
    }
    offset.abs() <= MAX_OFFSET && all / (COLS * ROWS) as f32 <= MAX_MEAN && words / n as f32 <= MAX_WORDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knows_the_slate_at_any_size_and_nothing_like_it() {
        let slate = |w: usize, h: usize| move |x: usize, y: usize| ELGATO_4K_S[y * ROWS / h][x * COLS / w];
        for (w, h) in [(640, 480), (1280, 720), (2560, 1080), (3840, 2160)] {
            assert!(is_slate(w, h, slate(w, h)), "{w}x{h}");
            assert!(is_slate(w, h, |x, y| slate(w, h)(x, y) + 2), "{w}x{h}, a little brighter");
            // The same dark gradient without the words: a dark scene.
            assert!(!is_slate(w, h, |x, y| slate(w, h)(x, y).min(34)), "{w}x{h}, no words");
            assert!(!is_slate(w, h, |_, _| 17), "{w}x{h}, black");
            assert!(!is_slate(w, h, |x, _| (x * 255 / w) as u8), "{w}x{h}, a bright picture");
        }
    }
}
