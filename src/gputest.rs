//! Headless checks of the GPU path, without a window or capture card.
//!
//! `--gpu-test IMAGE OUTDIR [OTHER]` builds a synthetic scene from IMAGE (camera pan and zoom,
//! a patch moving the other way, static HUD), generates its midpoint on the GPU, and compares
//! it with the true midpoint and a plain blend (PSNR), with timings. OTHER, an unrelated image
//! of the same size, checks the cut test.
//!
//! `--gpu-decode-test RAW.nv12 W H` decodes a `--snapshot` dump on the GPU and the CPU, SDR
//! and HDR, and reports how far apart they are.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::convert::{self, Hdr, Matrix, Sdr};
use crate::gpu::{Decoding, Frames, Gpu};
use crate::video::Layout;

fn load(p: &Path) -> Result<(Vec<[f32; 3]>, usize, usize)> {
    let img = image::open(p).with_context(|| format!("open {}", p.display()))?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.pixels().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect(), w, h))
}

/// RGB -> BGRA bytes.
fn bgra(px: &[[f32; 3]]) -> Vec<u8> {
    let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    px.iter().flat_map(|c| [q(c[2]), q(c[1]), q(c[0]), 255]).collect()
}

fn psnr(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let se: f64 = a.iter().zip(b).map(|(p, q)| (0..3).map(|c| ((p[c] - q[c]) as f64).powi(2)).sum::<f64>()).sum();
    let mse = se / (a.len() * 3) as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

fn save(path: &Path, px: &[[f32; 3]], w: usize, region: (usize, usize, usize, usize)) -> Result<()> {
    let (x0, y0, cw, ch) = region;
    let mut rgb = Vec::with_capacity(cw * ch * 3);
    for y in y0..y0 + ch {
        for c in &px[y * w + x0..y * w + x0 + cw] {
            rgb.extend(c.iter().map(|v| v.round().clamp(0.0, 255.0) as u8));
        }
    }
    image::save_buffer(path, &rgb, cw as u32, ch as u32, image::ExtendedColorType::Rgb8)?;
    Ok(())
}

/// Scene at time s (0..1): background pans and zooms in, a patch slides the other way, HUD
/// stays put.
fn scene(src: &[[f32; 3]], w: usize, h: usize, s: f32) -> Vec<[f32; 3]> {
    let bilinear = |x: f32, y: f32| -> [f32; 3] {
        let x = x.clamp(0.0, w as f32 - 1.001);
        let y = y.clamp(0.0, h as f32 - 1.001);
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |xx: usize, yy: usize| src[yy * w + xx];
        let mut o = [0.0; 3];
        for (c, v) in o.iter_mut().enumerate() {
            *v = p(x0, y0)[c] * (1.0 - fx) * (1.0 - fy)
                + p(x0 + 1, y0)[c] * fx * (1.0 - fy)
                + p(x0, y0 + 1)[c] * (1.0 - fx) * fy
                + p(x0 + 1, y0 + 1)[c] * fx * fy;
        }
        o
    };
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let zoom = 1.0 + 0.05 * s;
    let (pan_x, pan_y) = (48.0 * s, 12.0 * s);
    let (pw, ph) = (w / 8, h / 7);
    let (fx0, fy0) = (w as f32 * 0.25 - 64.0 * s, h as f32 * 0.55 + 20.0 * s);
    let mut out = vec![[0.0f32; 3]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (xf, yf) = (x as f32, y as f32);
            let in_patch = xf >= fx0 && xf < fx0 + pw as f32 && yf >= fy0 && yf < fy0 + ph as f32;
            let hud = x >= w / 20 && x < w / 20 + w / 5 && y >= h * 7 / 8 && y < h * 7 / 8 + h / 24;
            let px = if hud {
                // Text-like: thin bars.
                if (x / 3) % 3 == 0 || (y / 4) % 3 == 0 { [240.0; 3] } else { [32.0; 3] }
            } else if in_patch {
                bilinear(w as f32 * 0.7 + (xf - fx0), h as f32 * 0.2 + (yf - fy0))
            } else {
                bilinear(cx + (xf - cx) / zoom + pan_x, cy + (yf - cy) / zoom + pan_y)
            };
            // Quantise like a real frame.
            *o = px.map(|v: f32| v.round().clamp(0.0, 255.0));
        }
    });
    out
}

const RGB: Decoding = Decoding {
    layout: Layout::Rgb32,
    hdr: false,
    matrix: Matrix::Bt709,
    gamut_2020: false,
    full_range: true,
    white_nits: 370.0,
};

/// RGBA bytes read back from the GPU -> RGB.
fn rgb_of(bytes: &[u8]) -> Vec<[f32; 3]> {
    bytes.chunks_exact(4).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect()
}

/// `--upscale-test IMAGE OUTDIR [SCALE]`: shrinks IMAGE to 1/SCALE (default 4/3, so 1440p
/// becomes 1080p), scales it back up on the GPU three ways (the present shader's Lanczos, the
/// video processor alone, and with RTX Video Super Resolution), and scores each against the
/// original. Saves the centre of each for a look.
pub fn upscale_test(src: &Path, out_dir: &Path, scale: f64) -> Result<()> {
    use windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM};

    let full = image::open(src).with_context(|| format!("open {}", src.display()))?.to_rgba8();
    let (w, h) = (full.width(), full.height());
    let (sw, sh) = ((w as f64 / scale).round() as u32, (h as f64 / scale).round() as u32);
    let small = image::imageops::resize(&full, sw, sh, image::imageops::FilterType::Triangle);
    let truth = rgb_of(full.as_raw());
    std::fs::create_dir_all(out_dir)?;

    let gpu = Gpu::new()?;
    let f = Frames::new(&gpu, sw, sh, Layout::Rgb32)?;
    let bgra: Vec<u8> = small.pixels().flat_map(|p| [p[2], p[1], p[0], 255]).collect();
    f.upload(&gpu, &bgra)?;
    f.decode(&gpu, 0, &RGB, false);
    let input = &f.slots[0].frame;
    println!("{w}x{h} shrunk to {sw}x{sh} and scaled back up:");

    // Reads a result back as RGB, whichever byte order it's in.
    let read = |t: &crate::gpu::Tex| -> Result<Vec<[f32; 3]>> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { t.tex.GetDesc(&mut desc) };
        let mut px = rgb_of(&gpu.read(t, 4)?);
        if desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM {
            px.iter_mut().for_each(|p| p.swap(0, 2));
        }
        Ok(px)
    };
    let crop = (w as usize / 2 - 480, h as usize / 2 - 270, 960, 540);
    let report = |name: &str, px: &[[f32; 3]], ms: f64| -> Result<()> {
        println!("  {name:<28} {:.2} dB, {ms:.2} ms a frame", psnr(px, &truth));
        save(&out_dir.join(format!("{name}.png")), px, w as usize, crop)
    };
    save(&out_dir.join("original.png"), &truth, w as usize, crop)?;

    let iters = 60;
    let (target, rtv) = gpu.target(w, h, DXGI_FORMAT_R8G8B8A8_UNORM)?;
    let rect = [0.0, 0.0, w as f32, h as f32];
    gpu.draw(&rtv, (w, h), Some(input), rect, None);
    gpu.wait_idle()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.draw(&rtv, (w, h), Some(input), rect, None);
    }
    gpu.wait_idle()?;
    report("lanczos", &read(&target)?, t.elapsed().as_secs_f64() * 1000.0 / iters as f64)?;

    // The same picture as NV12 (BT.709, limited range), as the card sends it.
    let nv12 = {
        let (sw, sh) = (sw as usize, sh as usize);
        let (kr, kb) = (0.2126f32, 0.0722f32);
        let rgb = |x: usize, y: usize| small.get_pixel(x as u32, y as u32).0.map(|v| v as f32);
        let mut data = vec![0u8; sw * sh * 3 / 2];
        for y in 0..sh {
            for x in 0..sw {
                let [r, g, b, _] = rgb(x, y);
                data[y * sw + x] = (16.0 + 219.0 * (kr * r + (1.0 - kr - kb) * g + kb * b) / 255.0).round() as u8;
            }
        }
        for y in 0..sh / 2 {
            for x in 0..sw / 2 {
                let (mut cb, mut cr) = (0.0, 0.0);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let [r, g, b, _] = rgb(x * 2 + dx, y * 2 + dy);
                    let l = kr * r + (1.0 - kr - kb) * g + kb * b;
                    cb += 0.25 * (b - l) / (2.0 * (1.0 - kb));
                    cr += 0.25 * (r - l) / (2.0 * (1.0 - kr));
                }
                let uv = sw * sh + y * sw + x * 2;
                data[uv] = (128.0 + 224.0 * cb / 255.0).round() as u8;
                data[uv + 1] = (128.0 + 224.0 * cr / 255.0).round() as u8;
            }
        }
        use windows::Win32::Graphics::Direct3D11::*;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: sw as u32,
            Height: sh as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: 0, // the video processor refuses NV12 made with the shader resource bind
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init = D3D11_SUBRESOURCE_DATA { pSysMem: data.as_ptr() as _, SysMemPitch: sw as u32, SysMemSlicePitch: 0 };
        let mut t = None;
        unsafe { gpu.device.CreateTexture2D(&desc, Some(&init), Some(&mut t)) }.context("NV12 texture")?;
        t.context("no NV12 texture")?
    };

    let mut outputs = Vec::new();
    for (name, vsr, yuv) in [
        ("video-processor", false, false),
        ("rtx-super-resolution", true, false),
        ("video-processor-nv12", false, true),
        ("rtx-super-resolution-nv12", true, true),
    ] {
        let mut up = crate::upscale::Processor::new(&gpu, vsr)?;
        let go = |up: &mut crate::upscale::Processor| -> Result<Vec<[f32; 3]>> {
            let out = if yuv {
                up.run_texture(&gpu, &nv12, (sw, sh), windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12, (w, h))?
            } else {
                up.run(&gpu, input, w, h)?
            };
            read(out)
        };
        if let Err(e) = go(&mut up) {
            println!("  {name:<28} unavailable: {e:#}");
            continue;
        }
        gpu.wait_idle()?;
        let t = Instant::now();
        for _ in 0..iters {
            if yuv {
                up.run_texture(&gpu, &nv12, (sw, sh), windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12, (w, h))?;
            } else {
                up.run(&gpu, input, w, h)?;
            }
        }
        gpu.wait_idle()?;
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let px = go(&mut up)?;
        report(name, &px, ms)?;
        outputs.push((name, px));
    }
    for pair in outputs.chunks(2) {
        if let [(plain_name, plain), (vsr_name, vsr)] = pair {
            let apart = psnr(plain, vsr);
            let verdict = if apart > 60.0 { "the same: super resolution is NOT running" } else { "different: it's running" };
            println!("  {vsr_name} vs {plain_name}: {apart:.2} dB apart ({verdict})");
        }
    }
    println!("crops of the centre in {}", out_dir.display());
    Ok(())
}

pub fn interp_test(src: &Path, out_dir: &Path, other: Option<&Path>) -> Result<()> {
    let (src_px, w, h) = load(src)?;
    let (a, b, truth) = (scene(&src_px, w, h, 0.0), scene(&src_px, w, h, 1.0), scene(&src_px, w, h, 0.5));
    let t0 = Instant::now();
    let gpu = Gpu::new()?;
    println!("Direct3D up and shaders compiled in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
    let mut f = Frames::new(&gpu, w as u32, h as u32, Layout::Rgb32)?;
    let (ab, bb) = (bgra(&a), bgra(&b));

    f.upload(&gpu, &ab)?;
    f.decode(&gpu, 0, &RGB, true);
    f.upload(&gpu, &bb)?;
    f.decode(&gpu, 1, &RGB, true);
    f.estimate(&gpu, 0, 1);
    f.interpolate(&gpu, 0, 1, 0.5, 0, crate::quality::forced().unwrap_or_default());
    let out = rgb_of(&gpu.read(&f.outs[0], 4)?);
    let [_, _, cost, detail, ..] = f.counts(&gpu)?;

    // Timings: decode + pyramid of one frame (already uploaded), and motion + one generated frame.
    let iters = 50;
    gpu.wait_idle()?;
    let t = Instant::now();
    for _ in 0..iters {
        f.decode(&gpu, 1, &RGB, true);
    }
    gpu.wait_idle()?;
    let decode_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let t = Instant::now();
    for _ in 0..iters {
        f.estimate(&gpu, 0, 1);
    }
    gpu.wait_idle()?;
    let estimate_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let t = Instant::now();
    for _ in 0..iters {
        f.interpolate(&gpu, 0, 1, 0.5, 0, crate::quality::forced().unwrap_or_default());
    }
    gpu.wait_idle()?;
    let interp_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let t = Instant::now();
    for _ in 0..iters {
        f.upload(&gpu, &bb)?;
    }
    gpu.wait_idle()?;
    let upload_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Where the time goes: GPU time per pass over 10 pairs (motion + one frame each), clocks
    // already up from the loops above.
    gpu.profile_begin()?;
    for _ in 0..10 {
        f.estimate(&gpu, 0, 1);
        f.interpolate(&gpu, 0, 1, 0.5, 0, crate::quality::forced().unwrap_or_default());
    }
    let passes = gpu.profile_end()?;
    let total: f64 = passes.iter().map(|p| p.1).sum::<f64>() / 10.0;
    let list: Vec<String> = passes.iter().map(|(n, ms)| format!("{n} {:.2}", ms / 10.0)).collect();
    println!("GPU per pair {total:.2} ms: {}", list.join(", "));

    let blend: Vec<[f32; 3]> =
        a.iter().zip(&b).map(|(p, q)| [0, 1, 2].map(|c| ((p[c] + q[c]) / 2.0).round())).collect();
    println!(
        "{w}x{h} on the GPU: upload {upload_ms:.2} ms (BGRA), decode + pyramid {decode_ms:.2} ms, \
         motion {estimate_ms:.2} ms, generate {interp_ms:.2} ms"
    );
    println!(
        "cut test: cost {:.3} of detail (a cut above 0.9)",
        cost as f64 / detail.max(1) as f64
    );
    println!(
        "PSNR vs true midpoint: generated {:.2} dB, plain blend {:.2} dB, repeat A {:.2} dB",
        psnr(&out, &truth),
        psnr(&blend, &truth),
        psnr(&a, &truth)
    );
    std::fs::create_dir_all(out_dir)?;
    // Full frames, and 1:1 crops around the moving patch and the HUD.
    let (cw, ch) = (w / 3, h / 3);
    let cx = ((w as f32 * 0.25) as usize).saturating_sub(w / 10).min(w - cw);
    let cy = ((h as f32 * 0.55) as usize).saturating_sub(h / 10).min(h - ch);
    let hud =
        ((w / 20).saturating_sub(16), (h * 7 / 8).saturating_sub(32), (w / 5 + 32).min(w), (h / 24 + 64).min(h / 8));
    // Where it goes wrong: absolute error against the truth, 4x.
    let error: Vec<[f32; 3]> =
        out.iter().zip(&truth).map(|(p, q)| [0, 1, 2].map(|c| (p[c] - q[c]).abs() * 4.0)).collect();
    let text = (0, 0, (w * 3 / 8).min(w), (h * 2 / 7).min(h));
    for (name, px) in
        [("generated", &out), ("truth", &truth), ("blend", &blend), ("a", &a), ("b", &b), ("error", &error)]
    {
        save(&out_dir.join(format!("{name}.png")), px, w, (0, 0, w, h))?;
        save(&out_dir.join(format!("{name}-patch.png")), px, w, (cx, cy, cw, ch))?;
        save(&out_dir.join(format!("{name}-hud.png")), px, w, hud)?;
        save(&out_dir.join(format!("{name}-text.png")), px, w, text)?;
    }
    // The fields are from the last pair the profile ran, which is this same pair.
    for dir in 0..2 {
        save_fields(&f.fields(&gpu, dir)?, &out_dir.join(["ab", "ba"][dir]))?;
    }
    println!("wrote {}", out_dir.display());

    if let Some(other) = other {
        let (o, ow, oh) = load(other)?;
        if (ow, oh) != (w, h) {
            bail!("{} is {ow}x{oh}, not {w}x{h}", other.display());
        }
        f.upload(&gpu, &bgra(&o))?;
        f.decode(&gpu, 1, &RGB, true);
        f.forget_motion();
        f.estimate(&gpu, 0, 1);
        let [_, _, cost, detail, ..] = f.counts(&gpu)?;
        let share = cost as f64 / detail.max(1) as f64;
        println!(
            "unrelated picture: cost {share:.3} of detail -> {}",
            if share > 0.9 { "cut" } else { "NOT a cut" }
        );
    }
    Ok(())
}

/// A YUV4MPEG2 (4:2:0, 8-bit) clip as packed NV12 frames.
struct Clip {
    w: usize,
    h: usize,
    frames: Vec<Vec<u8>>,
}

fn load_y4m(p: &Path) -> Result<Clip> {
    let data = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    let line_end = |from: usize| data[from..].iter().position(|&b| b == b'\n').map(|i| from + i);
    let head_end = line_end(0).context("no y4m header")?;
    let header = std::str::from_utf8(&data[..head_end])?;
    if !header.starts_with("YUV4MPEG2") {
        bail!("{} is not a y4m file", p.display());
    }
    let field = |c: char| header.split(' ').find_map(|t| t.strip_prefix(c)).and_then(|v| v.parse::<usize>().ok());
    let (w, h) = (field('W').context("no width")?, field('H').context("no height")?);
    if let Some(c) = header.split(' ').find_map(|t| t.strip_prefix('C'))
        && !c.starts_with("420")
    {
        bail!("only 4:2:0 clips, this one is C{c}");
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut frames = Vec::new();
    let mut at = head_end + 1;
    while let Some(e) = line_end(at) {
        let start = e + 1;
        let end = start + w * h + 2 * cw * ch;
        if end > data.len() {
            break;
        }
        let (y, u, v) = (
            &data[start..start + w * h],
            &data[start + w * h..start + w * h + cw * ch],
            &data[start + w * h + cw * ch..end],
        );
        let mut nv12 = Vec::with_capacity(w * h + w * ch);
        nv12.extend_from_slice(y);
        for r in 0..ch {
            for c in 0..cw {
                nv12.push(u[r * cw + c]);
                nv12.push(v[r * cw + c]);
            }
        }
        frames.push(nv12);
        at = end;
    }
    Ok(Clip { w, h, frames })
}

/// Mean absolute error per 32x32 block (all channels): the worst block and the 99th
/// percentile. Broken-up patches show here even when the frame's PSNR barely moves.
fn block_errors(a: &[[f32; 3]], b: &[[f32; 3]], w: usize, h: usize) -> (f64, f64) {
    let (bw, bh) = (w / 32, h / 32);
    let mut errs: Vec<f64> = (0..bw * bh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % bw * 32, i / bw * 32);
            let mut s = 0.0f64;
            for y in by..by + 32 {
                for x in bx..bx + 32 {
                    let (p, q) = (a[y * w + x], b[y * w + x]);
                    s += (0..3).map(|c| (p[c] - q[c]).abs() as f64).sum::<f64>();
                }
            }
            s / (32.0 * 32.0 * 3.0)
        })
        .collect();
    errs.sort_by(|x, y| x.total_cmp(y));
    (errs.last().copied().unwrap_or(0.0), errs.get(errs.len() * 99 / 100).copied().unwrap_or(0.0))
}

/// Where generating did most harm compared with a plain blend: the 32x32 block whose mean error
/// grew the most, as (growth, x, y).
fn worst_excess(made: &[[f32; 3]], blend: &[[f32; 3]], truth: &[[f32; 3]], w: usize, h: usize) -> (f64, usize, usize) {
    let (bw, bh) = (w / 32, h / 32);
    (0..bw * bh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % bw * 32, i / bw * 32);
            let mut s = 0.0f64;
            for y in by..by + 32 {
                for x in bx..bx + 32 {
                    let (g, b, t) = (made[y * w + x], blend[y * w + x], truth[y * w + x]);
                    s += (0..3).map(|c| ((g[c] - t[c]).abs() - (b[c] - t[c]).abs()) as f64).sum::<f64>();
                }
            }
            (s / (32.0 * 32.0 * 3.0), bx, by)
        })
        .reduce(|| (f64::MIN, 0, 0), |a, b| if b.0 > a.0 { b } else { a })
}

/// `--gpu-eval CLIP.y4m OUTDIR [SKIP] [SAVE]`: plays a clip as if only every SKIP-th frame
/// arrived (2: 60 -> 30), generates the frames in between and scores them against the real
/// ones, next to a plain blend. SAVE lists pair numbers (comma separated) whose frames are
/// written to OUTDIR.
pub fn eval(clip: &Path, out_dir: &Path, skip: usize, keep: &[usize]) -> Result<()> {
    let c = load_y4m(clip)?;
    let (w, h) = (c.w, c.h);
    if c.frames.len() <= skip || skip < 2 {
        bail!("{} has {} frames, need more than {skip} (and SKIP >= 2)", clip.display(), c.frames.len());
    }
    let gpu = Gpu::new()?;
    let mut f = Frames::new(&gpu, w as u32, h as u32, Layout::Nv12)?;
    let dec = Decoding {
        layout: Layout::Nv12,
        hdr: false,
        matrix: Matrix::Bt709,
        gamut_2020: false,
        full_range: false,
        white_nits: 370.0,
    };
    std::fs::create_dir_all(out_dir)?;
    let name = clip.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    // How much each frame differs from the one before (mean absolute luma difference): a game
    // that missed 60 fps shows repeats (near 0) and uneven steps, and then the "true" middle
    // frame isn't halfway.
    let steps: Vec<String> = c
        .frames
        .windows(2)
        .map(|p| {
            let s: u64 = p[0][..w * h].iter().zip(&p[1][..w * h]).map(|(a, b)| a.abs_diff(*b) as u64).sum();
            format!("{:.1}", s as f64 / (w * h) as f64)
        })
        .collect();
    println!("{name} frame-to-frame change: {}", steps.join(" "));
    let read = |f: &Frames, slot: usize| -> Result<Vec<[f32; 3]>> { Ok(rgb_of(&gpu.read(&f.slots[slot].frame, 4)?)) };
    f.upload(&gpu, &c.frames[0])?;
    f.decode(&gpu, 0, &dec, true);
    let (mut gen_sum, mut blend_sum, mut n) = (0.0, 0.0, 0usize);
    let (mut gen_worst, mut blend_worst) = (0.0f64, 0.0f64);
    let (mut est_ms, mut gen_ms) = (0.0, 0.0);
    let mut pair = 0;
    while (pair + 1) * skip < c.frames.len() {
        let (ia, ib) = (pair * skip, (pair + 1) * skip);
        let (sa, sb) = (pair % 2, 1 - pair % 2);
        f.upload(&gpu, &c.frames[ib])?;
        f.decode(&gpu, sb, &dec, true);
        gpu.wait_idle()?;
        let t = Instant::now();
        f.estimate(&gpu, sa, sb);
        gpu.wait_idle()?;
        est_ms += t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        for j in 1..skip {
            f.interpolate(&gpu, sa, sb, j as f32 / skip as f32, j - 1, crate::quality::forced().unwrap_or_default());
        }
        gpu.wait_idle()?;
        gen_ms += t.elapsed().as_secs_f64() * 1000.0 / (skip - 1) as f64;
        let (a, b) = (read(&f, sa)?, read(&f, sb)?);
        let [_, _, cost, detail, changed, lost, hidden, unexplained, inverted, strong, _] = f.counts(&gpu)?;
        let ch = changed.max(1) as f64;
        let mut line = format!(
            "pair {pair:2} ({ia}->{ib}) cost/detail {:.2} lost/changed {:.2} hidden {:.2} unexplained {:.2} inverted {inverted}/{strong} ({changed}):",
            cost as f64 / detail.max(1) as f64,
            lost as f64 / ch,
            hidden as f64 / ch,
            unexplained as f64 / ch,
        );
        for j in 1..skip {
            let ts = j as f32 / skip as f32;
            let out = rgb_of(&gpu.read(&f.outs[j - 1], 4)?);
            // The real in-between frame, decoded into A's slot (A isn't needed any more).
            f.upload(&gpu, &c.frames[ia + j])?;
            f.decode(&gpu, sa, &dec, false);
            let truth = read(&f, sa)?;
            let blend: Vec<[f32; 3]> =
                a.iter().zip(&b).map(|(p, q)| [0, 1, 2].map(|k| (p[k] + (q[k] - p[k]) * ts).round())).collect();
            let (pg, pb) = (psnr(&out, &truth), psnr(&blend, &truth));
            let ((gw, g99), (bw, _)) = (block_errors(&out, &truth, w, h), block_errors(&blend, &truth, w, h));
            let (ex, exx, exy) = worst_excess(&out, &blend, &truth, w, h);
            line += &format!(
                "  t={ts:.2} gen {pg:.2} dB (worst block {gw:.1}, p99 {g99:.1}) blend {pb:.2} dB ({bw:.1}) worse than blend by {ex:.1} at {exx},{exy}"
            );
            gen_sum += pg;
            blend_sum += pb;
            gen_worst = gen_worst.max(gw);
            blend_worst = blend_worst.max(bw);
            n += 1;
            if keep.contains(&pair) {
                let err: Vec<[f32; 3]> =
                    out.iter().zip(&truth).map(|(p, q)| [0, 1, 2].map(|k| (p[k] - q[k]).abs() * 4.0)).collect();
                for (tag, px) in [("gen", &out), ("truth", &truth), ("blend", &blend), ("err", &err)] {
                    save(&out_dir.join(format!("{name}-p{pair}-{j}-{tag}.png")), px, w, (0, 0, w, h))?;
                }
            }
        }
        println!("{line}");
        if keep.contains(&pair) {
            for dir in 0..2 {
                save_fields(&f.fields(&gpu, dir)?, &out_dir.join(format!("{name}-p{pair}-{}", ["ab", "ba"][dir])))?;
            }
        }
        pair += 1;
    }
    println!(
        "{name}: {n} frames, generated {:.2} dB (worst block {gen_worst:.1}), plain blend {:.2} dB ({blend_worst:.1}); \
         motion {:.2} ms, a frame {:.2} ms",
        gen_sum / n as f64,
        blend_sum / n as f64,
        est_ms / pair as f64,
        gen_ms / pair as f64
    );
    Ok(())
}

/// Pictures of a direction's fields: `-flow` (hue = direction, brightness = speed, full at 32
/// level-0 pixels), `-occ` (white = hidden in the other picture, grey = partly) and `-unrel` (white =
/// falls back to a blend).
fn save_fields(fields: &(Vec<[f32; 4]>, usize, usize), base: &Path) -> Result<()> {
    let (px, w, h) = fields;
    let flow: Vec<[f32; 3]> = px
        .iter()
        .map(|p| {
            let (dx, dy) = (p[0], p[1]);
            let mag = ((dx * dx + dy * dy).sqrt() / 32.0).min(1.0);
            let hue = (dy.atan2(dx) / std::f32::consts::TAU + 1.0).fract() * 6.0;
            let c = |k: f32| {
                let x = ((hue + k) % 6.0 - 3.0).abs() - 1.0;
                x.clamp(0.0, 1.0)
            };
            [c(0.0), c(4.0), c(2.0)].map(|v| (0.25 + 0.75 * v) * mag * 255.0)
        })
        .collect();
    let occ: Vec<[f32; 3]> = px
        .iter()
        .map(|p| {
            [if p[2] < 0.0 {
                64.0
            } else if p[2] > 1.0 {
                255.0
            } else {
                p[2] * 128.0
            }; 3]
        })
        .collect();
    let unrel: Vec<[f32; 3]> = px.iter().map(|p| [p[3] * 255.0; 3]).collect();
    for (tag, img) in [("flow", &flow), ("occ", &occ), ("unrel", &unrel)] {
        let mut path = base.as_os_str().to_owned();
        path.push(format!("-{tag}.png"));
        save(Path::new(&path), img, *w, (0, 0, *w, *h))?;
    }
    Ok(())
}

pub fn decode_test(raw: &Path, w: usize, h: usize) -> Result<()> {
    let data = std::fs::read(raw)?;
    if data.len() < w * h * 3 / 2 {
        bail!("{} is smaller than a {w}x{h} NV12 frame", raw.display());
    }
    let gpu = Gpu::new()?;
    let f = Frames::new(&gpu, w as u32, h as u32, Layout::Nv12)?;
    f.upload(&gpu, &data)?;
    for hdr in [false, true] {
        let d = Decoding {
            layout: Layout::Nv12,
            hdr,
            matrix: Matrix::Bt709,
            gamut_2020: false,
            full_range: false,
            white_nits: crate::video::HDR_WHITE_NITS as f32,
        };
        f.decode(&gpu, 0, &d, false);
        let g = gpu.read(&f.slots[0].frame, 4)?;
        let mut cpu = vec![0u8; w * h * 4];
        if hdr {
            let t = Hdr::new(Matrix::Bt709, false, false, crate::video::HDR_WHITE_NITS);
            convert::nv12_to_bgra(&t, &data[..w * h], &data[w * h..], w, w, h, &mut cpu);
        } else {
            convert::nv12_to_bgra(&Sdr::new(Matrix::Bt709, false), &data[..w * h], &data[w * h..], w, w, h, &mut cpu);
        }
        // GPU is RGBA, CPU is BGRA.
        let (mut sum, mut max, mut over) = (0u64, 0u8, 0usize);
        for (gp, cp) in g.chunks_exact(4).zip(cpu.chunks_exact(4)) {
            for (gc, cc) in [(gp[0], cp[2]), (gp[1], cp[1]), (gp[2], cp[0])] {
                let d = gc.abs_diff(cc);
                sum += d as u64;
                max = max.max(d);
                if d > 2 {
                    over += 1;
                }
            }
        }
        println!(
            "{}: GPU vs CPU mean diff {:.3}, max {max}, {over} channel values off by more than 2",
            if hdr { "HDR" } else { "SDR" },
            sum as f64 / (w * h * 3) as f64,
        );
    }
    Ok(())
}

/// `--mark-test IMAGE OUTDIR`: draws IMAGE through the present shader with test sessions'
/// barcode for a set of present numbers, and saves each frame (`mark_NNNN.png`) and the
/// numbers (`marks.txt`), for checking the analyzer reads them back (also after encoding).
pub fn mark_test(src: &Path, out_dir: &Path) -> Result<()> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
    let img = image::open(src).with_context(|| format!("open {}", src.display()))?.to_rgba8();
    let (w, h) = (img.width() & !1, img.height() & !1);
    std::fs::create_dir_all(out_dir)?;
    let gpu = Gpu::new()?;
    let f = Frames::new(&gpu, w, h, Layout::Rgb32)?;
    let bgra: Vec<u8> = img.pixels().flat_map(|p| [p[2], p[1], p[0], 255]).collect();
    let bgra: Vec<u8> = bgra.chunks_exact(img.width() as usize * 4).take(h as usize).flat_map(|r| &r[..w as usize * 4]).copied().collect();
    f.upload(&gpu, &bgra)?;
    f.decode(&gpu, 0, &RGB, false);
    let (target, rtv) = gpu.target(w, h, DXGI_FORMAT_R8G8B8A8_UNORM)?;
    let rect = [0.0, 0.0, w as f32, h as f32];
    // Edge cases, then a run of consecutive numbers wrapping past 16 bits.
    let mut codes: Vec<u32> = vec![0, 1, 2, 0x5555, 0xAAAA, 0xFFFF, 0x8000, 12345];
    codes.extend((0..40).map(|i| (65_520 + i) & 0xFFFF));
    let mut list = String::new();
    for (i, &code) in codes.iter().enumerate() {
        gpu.draw(&rtv, (w, h), Some(&f.slots[0].frame), rect, Some(code));
        let px = rgb_of(&gpu.read(&target, 4)?);
        save(&out_dir.join(format!("mark_{i:04}.png")), &px, w as usize, (0, 0, w as usize, h as usize))?;
        list += &format!("{code}\n");
    }
    std::fs::write(out_dir.join("marks.txt"), list)?;
    println!("{} marked frames of {w}x{h} in {}", codes.len(), out_dir.display());
    Ok(())
}

#[cfg(test)]
mod menu_tests {
    use super::*;

    const W: usize = 1920;
    const H: usize = 1080;

    /// Whether (x, y) is ink of a row of made-up letters starting at (left, top): 8 cells of
    /// 30 x 40, each some strokes both ways.
    fn ink(x: usize, y: usize, left: usize, top: usize, row: usize) -> bool {
        if !(left..left + 240).contains(&x) || !(top..top + 40).contains(&y) {
            return false;
        }
        let (cx, cy) = ((x - left) % 30, y - top);
        if cx >= 24 {
            return false;
        }
        let bits = ((x - left) / 30 + 8 * row + 3).wrapping_mul(0x9E37_79B1);
        (bits & 1 << 4 != 0 && cx < 6)
            || (bits & 1 << 7 != 0 && cx >= 18)
            || (bits & 1 << 9 != 0 && cy < 6)
            || (bits & 1 << 12 != 0 && (17..23).contains(&cy))
            || (bits & 1 << 14 != 0 && cy >= 34)
            || (9..15).contains(&cx) && cy % 16 < 8
    }

    /// Six rows of text with row `on` highlighted, or the same rows moved down by `down`.
    fn menu(on: usize, down: usize) -> Vec<u8> {
        let mut f = vec![128u8; W * H * 3 / 2];
        for y in 0..H {
            for x in 0..W {
                let mut v = 38;
                for row in 0..6 {
                    let top = 200 + 96 * row + down;
                    let bar = row == on && (top..top + 72).contains(&y) && (40..560).contains(&x);
                    if bar {
                        v = 236;
                    }
                    if ink(x, y, 120, top + 16, row) {
                        v = if row == on { 25 } else { 220 };
                    }
                }
                f[y * W + x] = v;
            }
        }
        f
    }

    /// The frame generated halfway from `a` to `b`, and how far it is from `want` in levels (the
    /// worst 16x16 block).
    fn halfway(gpu: &Gpu, a: &[u8], b: &[u8], want: &[u8]) -> Result<f64> {
        Ok(worst_block(&made(gpu, a, b, crate::quality::forced().unwrap_or_default())?, want, 0))
    }

    /// The luma-like green of the frame made halfway from `a` to `b` at `level`.
    fn made(gpu: &Gpu, a: &[u8], b: &[u8], level: crate::quality::Quality) -> Result<Vec<f64>> {
        let mut f = Frames::new(gpu, W as u32, H as u32, Layout::Nv12)?;
        let dec = Decoding {
            layout: Layout::Nv12,
            hdr: false,
            matrix: Matrix::Bt709,
            gamut_2020: false,
            full_range: true,
            white_nits: 370.0,
        };
        for (slot, data) in [(0, a), (1, b)] {
            f.upload(gpu, data)?;
            f.decode(gpu, slot, &dec, true);
        }
        f.estimate(gpu, 0, 1);
        f.interpolate(gpu, 0, 1, 0.5, 0, level);
        gpu.wait_idle()?;
        Ok(rgb_of(&gpu.read(&f.outs[0], 4)?).iter().map(|p| p[1] as f64).collect())
    }

    /// How far the worst 16x16 block is from `want` in levels, `margin` in from top and bottom.
    fn worst_block(out: &[f64], want: &[u8], margin: usize) -> f64 {
        let mut worst = 0.0f64;
        for by in margin / 16..(H - margin) / 16 {
            for bx in 0..W / 16 {
                let mut sum = 0.0;
                for y in by * 16..by * 16 + 16 {
                    for x in bx * 16..bx * 16 + 16 {
                        sum += (out[y * W + x] - want[y * W + x] as f64).abs();
                    }
                }
                worst = worst.max(sum / 256.0);
            }
        }
        worst
    }

    /// The outer 64 pixels left and right, away from the corners: how far off they are on
    /// average, and how many of their 16x16 blocks are over 14 levels off.
    fn sides(out: &[f64], want: &[u8]) -> (f64, usize) {
        let (mut total, mut bad, mut n) = (0.0, 0, 0);
        for by in 2..H / 16 - 2 {
            for bx in (0..4).chain(W / 16 - 4..W / 16) {
                let mut sum = 0.0;
                for y in by * 16..by * 16 + 16 {
                    for x in bx * 16..bx * 16 + 16 {
                        sum += (out[y * W + x] - want[y * W + x] as f64).abs();
                    }
                }
                total += sum / 256.0;
                bad += (sum / 256.0 > 14.0) as usize;
                n += 1;
            }
        }
        (total / n as f64, bad)
    }

    /// A smooth random texture of 24-pixel cells, `right` pixels along.
    fn scene(right: usize) -> Vec<u8> {
        pan(right, 0)
    }

    /// The same, `down` pixels down as well.
    fn pan(right: usize, down: usize) -> Vec<u8> {
        let at = |x: usize, y: usize| {
            let h = ((x as u64) << 32 | y as u64).wrapping_add(7).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 56) as f64
        };
        let mut f = vec![128u8; W * H * 3 / 2];
        for y in 0..H {
            for x in 0..W {
                let (u, v) = (x + right, y + down);
                let (cx, cy, fx, fy) = (u / 24, v / 24, (u % 24) as f64 / 24.0, (v % 24) as f64 / 24.0);
                let top = at(cx, cy) * (1.0 - fx) + at(cx + 1, cy) * fx;
                let below = at(cx, cy + 1) * (1.0 - fx) + at(cx + 1, cy + 1) * fx;
                f[y * W + x] = (30.0 + 0.75 * (top * (1.0 - fy) + below * fy)) as u8;
            }
        }
        f
    }

    /// `pictures` one after another, as in play: the Frames after the last pair, with the
    /// frame halfway through it made.
    fn played(gpu: &Gpu, pictures: &[Vec<u8>], level: crate::quality::Quality) -> Result<Frames> {
        let mut f = Frames::new(gpu, W as u32, H as u32, Layout::Nv12)?;
        let dec = Decoding {
            layout: Layout::Nv12,
            hdr: false,
            matrix: Matrix::Bt709,
            gamut_2020: false,
            full_range: true,
            white_nits: 370.0,
        };
        f.forget_motion();
        for (i, data) in pictures.iter().enumerate() {
            f.upload(gpu, data)?;
            f.decode(gpu, i % 2, &dec, true);
            if i > 0 {
                f.estimate(gpu, (i + 1) % 2, i % 2);
            }
        }
        let last = pictures.len() - 1;
        f.interpolate(gpu, (last + 1) % 2, last % 2, 0.5, 0, level);
        gpu.wait_idle()?;
        Ok(f)
    }

    #[test]
    fn an_overlay_stays_whole_while_the_camera_pans_under_it() {
        let Ok(gpu) = Gpu::new() else { return };
        // Three plain bars with a thin outline, as a game draws health and stamina.
        let bars = [(64..256usize, 27..42usize, 190u8), (64..266, 57..72, 175), (64..224, 87..102, 150)];
        let picture = |k: usize| {
            let mut f = pan(48 * k, 32 * (7 - k));
            for (xs, ys, fill) in &bars {
                for y in ys.clone() {
                    for x in xs.clone() {
                        let rim = x < xs.start + 2 || x >= xs.end - 2 || y < ys.start + 2 || y >= ys.end - 2;
                        f[y * W + x] = if rim { 20 } else { *fill };
                    }
                }
            }
            f
        };
        let pictures: Vec<Vec<u8>> = (0..7).map(picture).collect();
        for level in crate::quality::Quality::ALL {
            let f = played(&gpu, &pictures, level).unwrap();
            let out = rgb_of(&gpu.read(&f.outs[0], 4).unwrap());
            let (mut off, mut n) = (0usize, 0usize);
            for (xs, ys, _) in &bars {
                for y in ys.clone() {
                    for x in xs.clone() {
                        off += ((out[y * W + x][1] as f64 - pictures[6][y * W + x] as f64).abs() > 20.0) as usize;
                        n += 1;
                    }
                }
            }
            assert!(off * 100 <= n, "{level:?}: the overlay tore, {off} of its {n} pixels");
        }
    }

    #[test]
    fn railings_lining_up_once_are_no_overlay() {
        let Ok(gpu) = Gpu::new() else { return };
        // Posts every 40 pixels sliding by unevenly, the last time by exactly their spacing.
        let picture = |at: usize| {
            let mut f = vec![128u8; W * H * 3 / 2];
            for y in 0..H {
                for x in 0..W {
                    f[y * W + x] = if (x + at) % 40 < 8 { 40 } else { 210 };
                }
            }
            f
        };
        let pictures: Vec<Vec<u8>> = [0, 23, 52, 69, 100, 140].map(picture).into();
        let f = played(&gpu, &pictures, crate::quality::Quality::High).unwrap();
        let held = f.held_count(&gpu).unwrap();
        assert_eq!(held, 0, "railings were taken for an overlay");
    }

    #[test]
    fn nothing_is_held_while_the_camera_stands_still() {
        let Ok(gpu) = Gpu::new() else { return };
        // A big thing crosses a still, hard-edged background: that all stayed, but it isn't an
        // overlay, and what crosses it mustn't be rubbed out.
        let picture = |k: usize| {
            let (mut f, tex) = (menu(0, 0), scene(0));
            for y in 0..H {
                for x in 700 + 80 * k..1300 + 80 * k {
                    f[y * W + x] = tex[y * W + x - 80 * k];
                }
            }
            f
        };
        let pictures: Vec<Vec<u8>> = (0..7).map(picture).collect();
        let f = played(&gpu, &pictures, crate::quality::Quality::High).unwrap();
        let moving = f.counts(&gpu).unwrap()[10] as usize;
        assert!(f.held_count(&gpu).unwrap() > 0, "the test needs edges that stayed");
        // Level 0 is half size here; HELD_MOVING is 0.6 of it.
        assert!(5 * moving < 3 * (W * H / 4), "a still camera counts as moving: {moving} pixels");
    }

    #[test]
    fn what_leaves_the_picture_moves_with_the_rest() {
        let Ok(gpu) = Gpu::new() else { return };
        // A pan, both ways: a band at each side is in one picture only.
        for (a, b) in [(0, 96), (96, 0)] {
            for level in crate::quality::Quality::ALL {
                let out = made(&gpu, &scene(a), &scene(b), level).unwrap();
                let (off, bad) = sides(&out, &scene(48));
                // (0.2.5: some 10 levels and over a quarter of them. One lucky false match is allowed.)
                assert!(off < 1.5 && bad <= 5, "{level:?}: a pan from {a} to {b}: edges {off:.2} levels off, {bad} blocks broken");
            }
        }
    }

    #[test]
    fn a_value_changing_its_text_does_not_tear_and_small_motion_still_moves() {
        let Ok(gpu) = Gpu::new() else { return };
        // One row's text changes where it is.
        let write = |row: usize| {
            let mut f = menu(0, 0);
            for y in 680..720 {
                for x in 900..1140 {
                    f[y * W + x] = if ink(x, y, 900, 680, row) { 220 } else { 38 };
                }
            }
            f
        };
        let (a, b) = (write(7), write(11));
        let fade: Vec<u8> = a.iter().zip(&b).map(|(&p, &q)| (p as u16 + q as u16).div_ceil(2) as u8).collect();
        for level in crate::quality::Quality::ALL {
            let out = made(&gpu, &a, &b, level).unwrap();
            let off = [&a, &b, &fade].map(|w| worst_block(&out, w, 0)).into_iter().fold(f64::MAX, f64::min);
            assert!(off < 10.0, "{level:?}: the text tore, {off:.1} levels off");
        }
        // A little soft thing moving in a still picture, uncovering some (someone talking):
        // that's motion, however little of the picture it is.
        let thing = |dx: usize| {
            let (mut f, tex, ground) = (vec![128u8; W * H * 3 / 2], scene(0), scene(700));
            for y in 0..H {
                for x in 0..W {
                    let inside = (800 + dx..900 + dx).contains(&x) && (400..480).contains(&y);
                    f[y * W + x] = if inside { 54 + tex[y * W + x - dx] / 7 } else { 50 + ground[y * W + x] / 12 };
                }
            }
            f
        };
        let (a, b, want) = (thing(0), thing(56), thing(28));
        let fade: Vec<u8> = a.iter().zip(&b).map(|(&p, &q)| (p as u16 + q as u16).div_ceil(2) as u8).collect();
        let out = made(&gpu, &a, &b, crate::quality::Quality::High).unwrap();
        // The middle of the thing, away from its border.
        let off = |w: &[u8]| {
            let mut sum = 0.0;
            for y in 412..468 {
                for x in 862..894 {
                    sum += (out[y * W + x] - w[y * W + x] as f64).abs();
                }
            }
            sum / (56.0 * 32.0)
        };
        assert!(off(&want) < off(&fade), "small motion cross-fades: {:.1} from moved, {:.1} from faded", off(&want), off(&fade));
    }

    #[test]
    fn a_highlight_jumping_rows_shows_at_once_and_a_scrolling_list_still_moves() {
        // CI's machines have no GPU.
        let Ok(gpu) = Gpu::new() else { return };
        for (from, to) in [(3, 4), (4, 3), (1, 2), (0, 5)] {
            let (a, b) = (menu(from, 0), menu(to, 0));
            let off = halfway(&gpu, &a, &b, &b).unwrap();
            assert!(off < 4.0, "row {from} to {to}: torn, {off:.1} levels from the new picture");
        }
        let off = halfway(&gpu, &menu(2, 0), &menu(2, 16), &menu(2, 8)).unwrap();
        assert!(off < 40.0, "a scrolling list stopped moving: {off:.1} levels off");
    }
}
