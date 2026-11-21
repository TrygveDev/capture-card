//! YUV -> BGRA conversion, done here instead of in Media Foundation so it runs on every
//! core at once and writes straight into the buffer gpui uploads (one pass, no extra copy).
//!
//! Two decoders share the row loops: plain SDR (BT.601/709), and HDR10 (BT.2020 + PQ) tone
//! mapped down to SDR, since gpui presents an 8-bit sRGB swapchain. Without the HDR path an HDR
//! source looks washed out and grey, because PQ code values shown as SDR gamma are far too flat.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
}

pub trait Decode: Sync {
    fn pixel(&self, y: u8, u: u8, v: u8) -> u32;
}

/// SDR lookup tables for one colour matrix and range, in 16.16 fixed point.
pub struct Sdr {
    y: [i32; 256],
    rv: [i32; 256],
    gu: [i32; 256],
    gv: [i32; 256],
    bu: [i32; 256],
}

impl Sdr {
    pub fn new(matrix: Matrix, full_range: bool) -> Self {
        let (kr, kb) = kr_kb(matrix);
        let kg = 1.0 - kr - kb;
        let (y_off, y_scale, c_scale) = if full_range {
            (0.0, 1.0, 1.0)
        } else {
            (16.0, 255.0 / 219.0, 255.0 / 224.0)
        };
        let fix = |v: f64| (v * 65536.0).round() as i32;
        let mut t = Sdr { y: [0; 256], rv: [0; 256], gu: [0; 256], gv: [0; 256], bu: [0; 256] };
        for i in 0..256 {
            let yv = (i as f64 - y_off) * y_scale;
            let c = (i as f64 - 128.0) * c_scale;
            t.y[i] = fix(yv) + 32768; // + 0.5 for rounding
            t.rv[i] = fix(2.0 * (1.0 - kr) * c);
            t.bu[i] = fix(2.0 * (1.0 - kb) * c);
            t.gu[i] = fix(-2.0 * (1.0 - kb) * kb / kg * c);
            t.gv[i] = fix(-2.0 * (1.0 - kr) * kr / kg * c);
        }
        t
    }
}

impl Decode for Sdr {
    #[inline(always)]
    fn pixel(&self, y: u8, u: u8, v: u8) -> u32 {
        let yy = self.y[y as usize];
        let r = ((yy + self.rv[v as usize]) >> 16).clamp(0, 255) as u32;
        let g = ((yy + self.gu[u as usize] + self.gv[v as usize]) >> 16).clamp(0, 255) as u32;
        let b = ((yy + self.bu[u as usize]) >> 16).clamp(0, 255) as u32;
        0xFF00_0000 | (r << 16) | (g << 8) | b
    }
}

/// Entries in the PQ -> linear table (over PQ code values 0..1).
const PQ_LUT: usize = 1024;
/// Entries in the linear -> sRGB table (over 0..1 linear).
const SRGB_LUT: usize = 4096;

/// HDR: YCbCr carrying PQ (SMPTE ST 2084), tone mapped to BT.709 / sRGB with SDR reference
/// white at `white_nits`. HDR10 proper is BT.2020, but the Elgato HD60 X converts the gamut to
/// BT.709 itself and says so in the media type, leaving only the PQ curve, so the matrix and
/// primaries come from the source rather than being assumed.
pub struct Hdr {
    y: [f32; 256],
    c: [f32; 256],
    /// Cr -> R, Cb -> G, Cr -> G, Cb -> B for the matrix in use.
    k: [f32; 4],
    /// Linear-light gamut conversion to BT.709, when the source primaries are wider.
    gamut: Option<[[f32; 3]; 3]>,
    /// PQ code value -> linear light, scaled so 1.0 is reference white.
    pq: Box<[f32; PQ_LUT + 1]>,
    srgb: Box<[u8; SRGB_LUT + 1]>,
}

/// SMPTE ST 2084 EOTF: code value 0..1 -> nits.
fn pq_to_nits(e: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    let p = e.max(0.0).powf(1.0 / M2);
    let num = (p - C1).max(0.0);
    let den = C2 - C3 * p;
    10000.0 * (num / den).powf(1.0 / M1)
}

fn srgb_encode(l: f64) -> f64 {
    if l <= 0.003_130_8 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
}

/// Kr and Kb for a YCbCr matrix.
pub fn kr_kb(matrix: Matrix) -> (f64, f64) {
    match matrix {
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Bt709 => (0.2126, 0.0722),
        Matrix::Bt2020 => (0.2627, 0.0593),
    }
}

/// BT.2020 -> BT.709 primaries in linear light (ITU-R BT.2087).
const BT2020_TO_709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641, -0.072_850],
    [-0.124_550, 1.132_900, -0.008_349],
    [-0.018_151, -0.100_579, 1.118_730],
];

impl Hdr {
    /// `white_nits`: the HDR brightness that lands on SDR white.
    pub fn new(matrix: Matrix, bt2020_primaries: bool, full_range: bool, white_nits: f64) -> Self {
        let (y_off, y_scale, c_scale) = if full_range {
            (0.0, 1.0 / 255.0, 1.0 / 255.0)
        } else {
            (16.0, 1.0 / 219.0, 1.0 / 224.0)
        };
        let mut y = [0f32; 256];
        let mut c = [0f32; 256];
        for i in 0..256 {
            y[i] = ((i as f64 - y_off) * y_scale) as f32;
            c[i] = ((i as f64 - 128.0) * c_scale) as f32;
        }
        let (kr, kb) = kr_kb(matrix);
        let kg = 1.0 - kr - kb;
        let k = [
            2.0 * (1.0 - kr),
            -2.0 * (1.0 - kb) * kb / kg,
            -2.0 * (1.0 - kr) * kr / kg,
            2.0 * (1.0 - kb),
        ]
        .map(|v| v as f32);
        let mut pq = Box::new([0f32; PQ_LUT + 1]);
        for (i, v) in pq.iter_mut().enumerate() {
            *v = (pq_to_nits(i as f64 / PQ_LUT as f64) / white_nits) as f32;
        }
        let mut srgb = Box::new([0u8; SRGB_LUT + 1]);
        for (i, v) in srgb.iter_mut().enumerate() {
            *v = (srgb_encode(i as f64 / SRGB_LUT as f64) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        Hdr { y, c, k, gamut: bt2020_primaries.then_some(BT2020_TO_709), pq, srgb }
    }

    #[inline(always)]
    fn linear(&self, e: f32) -> f32 {
        let i = (e.clamp(0.0, 1.0) * PQ_LUT as f32 + 0.5) as usize;
        self.pq[i]
    }

    #[inline(always)]
    fn encode(&self, l: f32) -> u32 {
        self.srgb[(l.clamp(0.0, 1.0) * SRGB_LUT as f32 + 0.5) as usize] as u32
    }
}

/// Saturation gain for the HDR path (1.0 = none). Tuned on the PS5 home tiles against their
/// official art.
const SATURATION: f32 = 1.05;

/// Where highlight roll-off starts, as a fraction of SDR white. Everything below is shown
/// as-is, so SDR content inside an HDR signal (console UI, store art) comes out unchanged.
const KNEE: f32 = 0.9;

/// Highlight roll-off: linear up to the knee, then a rational curve that approaches 1 with a
/// matching slope at the knee, so midtones stay untouched and highlights don't clip flat.
#[inline(always)]
fn tonemap(x: f32) -> f32 {
    if x <= KNEE {
        x
    } else {
        let t = (x - KNEE) / (1.0 - KNEE);
        KNEE + (1.0 - KNEE) * t / (1.0 + t)
    }
}

impl Decode for Hdr {
    #[inline(always)]
    fn pixel(&self, y: u8, u: u8, v: u8) -> u32 {
        let yy = self.y[y as usize];
        let cb = self.c[u as usize];
        let cr = self.c[v as usize];
        let [rv, gu, gv, bu] = self.k;
        let mut r = self.linear(yy + rv * cr);
        let mut g = self.linear(yy + gu * cb + gv * cr);
        let mut b = self.linear(yy + bu * cb);
        if let Some(m) = &self.gamut {
            (r, g, b) = (
                (m[0][0] * r + m[0][1] * g + m[0][2] * b).max(0.0),
                (m[1][0] * r + m[1][1] * g + m[1][2] * b).max(0.0),
                (m[2][0] * r + m[2][1] * g + m[2][2] * b).max(0.0),
            );
        }
        // A touch more colour: measured against official art the decode sat a few percent
        // under it. Scaled around luminance in linear light, so hue and brightness hold.
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let (r, g, b) = (
            (l + (r - l) * SATURATION).max(0.0),
            (l + (g - l) * SATURATION).max(0.0),
            (l + (b - l) * SATURATION).max(0.0),
        );
        // Tone map on the brightest channel so hue is kept as highlights compress.
        let m = r.max(g).max(b);
        let s = if m > KNEE { tonemap(m) / m } else { 1.0 };
        let (r, g, b) = (self.encode(r * s), self.encode(g * s), self.encode(b * s));
        0xFF00_0000 | (r << 16) | (g << 8) | b
    }
}

/// Rows per rayon job: big enough to amortise scheduling, small enough to spread a frame.
const ROWS_PER_JOB: usize = 16;

/// NV12: full-res Y plane, then interleaved UV at half res.
pub fn nv12_to_bgra<D: Decode>(
    d: &D,
    y_plane: &[u8],
    uv_plane: &[u8],
    pitch: usize,
    width: usize,
    height: usize,
    out: &mut [u8],
) {
    let out_row = width * 4;
    out[..out_row * height]
        .par_chunks_mut(out_row * ROWS_PER_JOB)
        .enumerate()
        .for_each(|(job, chunk)| {
            for (i, row) in chunk.chunks_exact_mut(out_row).enumerate() {
                let r = job * ROWS_PER_JOB + i;
                let ys = &y_plane[r * pitch..r * pitch + width];
                let uvs = &uv_plane[(r / 2) * pitch..(r / 2) * pitch + (width & !1)];
                let px = row.as_chunks_mut::<8>().0;
                for ((o, yy), uv) in px.iter_mut().zip(ys.chunks_exact(2)).zip(uvs.chunks_exact(2)) {
                    o[..4].copy_from_slice(&d.pixel(yy[0], uv[0], uv[1]).to_le_bytes());
                    o[4..].copy_from_slice(&d.pixel(yy[1], uv[0], uv[1]).to_le_bytes());
                }
            }
        });
}

/// YUY2: packed Y0 U Y1 V per pixel pair.
pub fn yuy2_to_bgra<D: Decode>(d: &D, src: &[u8], pitch: usize, width: usize, height: usize, out: &mut [u8]) {
    let out_row = width * 4;
    out[..out_row * height]
        .par_chunks_mut(out_row * ROWS_PER_JOB)
        .enumerate()
        .for_each(|(job, chunk)| {
            for (i, row) in chunk.chunks_exact_mut(out_row).enumerate() {
                let r = job * ROWS_PER_JOB + i;
                let s = &src[r * pitch..r * pitch + (width & !1) * 2];
                let px = row.as_chunks_mut::<8>().0;
                for (o, q) in px.iter_mut().zip(s.chunks_exact(4)) {
                    o[..4].copy_from_slice(&d.pixel(q[0], q[1], q[3]).to_le_bytes());
                    o[4..].copy_from_slice(&d.pixel(q[2], q[1], q[3]).to_le_bytes());
                }
            }
        });
}

/// Halves a BGRA image (2x2 box average, rounded). Odd edges are dropped.
pub fn halve_bgra(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (ow, oh) = (w / 2, h / 2);
    let mut out = vec![0u8; ow * oh * 4];
    out.par_chunks_mut(ow * 4).enumerate().for_each(|(y, row)| {
        let a = &src[(2 * y) * w * 4..(2 * y + 1) * w * 4];
        let b = &src[(2 * y + 1) * w * 4..(2 * y + 2) * w * 4];
        for x in 0..ow {
            for c in 0..4 {
                let i = x * 8 + c;
                let sum = a[i] as u16 + a[i + 4] as u16 + b[i] as u16 + b[i + 4] as u16;
                row[x * 4 + c] = ((sum + 2) / 4) as u8;
            }
        }
    });
    out
}

/// RGB32 from Media Foundation (BGRX in memory): copy rows and force alpha opaque.
/// `first_row` is the byte offset of the top row; a negative pitch walks a bottom-up image.
pub fn bgrx_to_bgra(src: &[u8], first_row: usize, pitch: isize, width: usize, height: usize, out: &mut [u8]) {
    let out_row = width * 4;
    out[..out_row * height]
        .par_chunks_mut(out_row * ROWS_PER_JOB)
        .enumerate()
        .for_each(|(job, chunk)| {
            for (i, row) in chunk.chunks_exact_mut(out_row).enumerate() {
                let r = (job * ROWS_PER_JOB + i) as isize;
                let start = (first_row as isize + r * pitch) as usize;
                let s = &src[start..start + out_row];
                for (o, p) in row.chunks_exact_mut(4).zip(s.chunks_exact(4)) {
                    o[0] = p[0];
                    o[1] = p[1];
                    o[2] = p[2];
                    o[3] = 0xFF;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(p: u32) -> (u32, u32, u32) {
        ((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF)
    }

    #[test]
    fn limited_range_black_and_white() {
        let t = Sdr::new(Matrix::Bt709, false);
        assert_eq!(t.pixel(16, 128, 128), 0xFF00_0000);
        assert_eq!(t.pixel(235, 128, 128), 0xFFFF_FFFF);
    }

    #[test]
    fn bt709_red() {
        let t = Sdr::new(Matrix::Bt709, false);
        let (r, g, b) = rgb(t.pixel(63, 102, 240));
        assert!(r >= 253 && g <= 2 && b <= 2, "{r} {g} {b}");
    }

    #[test]
    fn pq_reference_points() {
        // PQ 0.5807 is ~203 nits, 0.7518 ~1000 nits.
        assert!((pq_to_nits(0.5807) - 203.0).abs() < 2.0);
        assert!((pq_to_nits(0.7518) - 1000.0).abs() < 10.0);
        assert_eq!(pq_to_nits(0.0), 0.0);
    }

    #[test]
    fn hdr_reference_white_is_near_white_and_black_is_black() {
        let t = Hdr::new(Matrix::Bt2020, true, false, 203.0);
        assert_eq!(rgb(t.pixel(16, 128, 128)), (0, 0, 0));
        // Limited-range Y for PQ 0.5807 is 16 + 0.5807 * 219 = 143.
        let (r, g, b) = rgb(t.pixel(143, 128, 128));
        assert!(r > 225 && g > 225 && b > 225, "{r} {g} {b}");
        // Peak brightness rolls off to white without overflowing.
        assert_eq!(rgb(t.pixel(235, 128, 128)), (255, 255, 255));
    }

    #[test]
    fn halving_averages_each_2x2_block() {
        // 4x2 BGRA: left block averages to 100, right block to 200 (alpha stays 255).
        let px: Vec<u8> = [
            [80u8, 80, 80, 255], [120, 120, 120, 255], [200, 200, 200, 255], [200, 200, 200, 255],
            [100, 100, 100, 255], [100, 100, 100, 255], [190, 190, 190, 255], [210, 210, 210, 255],
        ]
        .concat();
        assert_eq!(halve_bgra(&px, 4, 2), vec![100, 100, 100, 255, 200, 200, 200, 255]);
    }

    /// Inverse of `pq_to_nits`, for building test signals.
    fn nits_to_pq(n: f64) -> f64 {
        const M1: f64 = 2610.0 / 16384.0;
        const M2: f64 = 2523.0 / 4096.0 * 128.0;
        const C1: f64 = 3424.0 / 4096.0;
        const C2: f64 = 2413.0 / 4096.0 * 32.0;
        const C3: f64 = 2392.0 / 4096.0 * 32.0;
        let y = (n / 10000.0).powf(M1);
        ((C1 + C2 * y) / (1.0 + C3 * y)).powf(M2)
    }

    #[test]
    fn sdr_grey_inside_pq_comes_back_unchanged() {
        // A console putting SDR grey levels into PQ at `white` nits, as the PS5 does for its UI
        // and store art: decoding at the same white must give the original sRGB codes back.
        let white = 370.0;
        let t = Hdr::new(Matrix::Bt709, false, false, white);
        for code in [40u8, 90, 128, 180, 220] {
            let lin = {
                let c = code as f64 / 255.0;
                if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
            };
            let y = (16.0 + 219.0 * nits_to_pq(lin * white)).round() as u8;
            let (r, g, b) = rgb(t.pixel(y, 128, 128));
            assert!(r == g && g == b, "grey stays neutral: {r} {g} {b}");
            assert!((r as i32 - code as i32).abs() <= 3, "code {code} came back as {r}");
        }
    }
}
