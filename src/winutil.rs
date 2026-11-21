//! The window behaviour gpui doesn't expose, done by wrapping its window procedure:
//! - drag anywhere to move (everything outside the controls is caption), and double-clicks
//!   there do nothing (Windows would maximize),
//! - edge drags keep the video's aspect ratio,
//! - hiding the cursor, and always-on-top.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, AtomicU32, Ordering};

use futures::channel::mpsc::UnboundedSender;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_LBUTTON};
use windows::Win32::UI::WindowsAndMessaging::*;

/// Width / height to hold the client area at, as f32 bits.
static ASPECT: AtomicU32 = AtomicU32::new(0x3FE3_8E39); // 16/9
static PREV_PROC: AtomicIsize = AtomicIsize::new(0);
/// Controls rectangle in client pixels; empty (right <= left) when hidden.
static CONTROLS: [AtomicI32; 4] = [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)];
static HIDE_CURSOR: AtomicBool = AtomicBool::new(false);
static EVENTS: OnceLock<UnboundedSender<Event>> = OnceLock::new();

#[derive(Debug)]
pub enum Event {
    /// A press outside the controls (it starts a window drag, so gpui never sees it).
    PressOutside,
}

pub fn hwnd(window: &gpui::Window) -> Option<HWND> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    match HasWindowHandle::window_handle(window).ok()?.as_raw() {
        RawWindowHandle::Win32(h) => Some(HWND(h.hwnd.get() as _)),
        _ => None,
    }
}

pub fn set_aspect(aspect: f32) {
    if aspect.is_finite() && aspect > 0.1 {
        ASPECT.store(aspect.to_bits(), Ordering::Relaxed);
    }
}

/// Client-pixel rectangle the controls cover, so clicks there reach them instead of dragging.
pub fn set_controls_rect(r: Option<(i32, i32, i32, i32)>) {
    let (l, t, rt, b) = r.unwrap_or((0, 0, 0, 0));
    for (slot, v) in CONTROLS.iter().zip([l, t, rt, b]) {
        slot.store(v, Ordering::Relaxed);
    }
}

fn in_controls(x: i32, y: i32) -> bool {
    let [l, t, r, b] = CONTROLS.each_ref().map(|a| a.load(Ordering::Relaxed));
    r > l && x >= l && x < r && y >= t && y < b
}

pub fn set_cursor_hidden(hidden: bool) {
    if HIDE_CURSOR.swap(hidden, Ordering::Relaxed) != hidden && hidden {
        unsafe {
            SetCursor(None);
        }
    }
}

/// Installs the wrapper. `events` receives what the wrapper needs gpui to do.
pub fn install(hwnd: HWND, events: UnboundedSender<Event>) {
    let _ = EVENTS.set(events);
    unsafe {
        let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, wndproc as *const () as usize as isize);
        PREV_PROC.store(prev, Ordering::Relaxed);
    }
}

/// gpui removes the resize frame in fullscreen.
fn is_fullscreen(hwnd: HWND) -> bool {
    unsafe { (GetWindowLongPtrW(hwnd, GWL_STYLE) as u32) & WS_THICKFRAME.0 == 0 }
}

fn point_from_lparam(lparam: LPARAM) -> POINT {
    POINT { x: (lparam.0 & 0xFFFF) as i16 as i32, y: ((lparam.0 >> 16) & 0xFFFF) as i16 as i32 }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe {
        let prev: WNDPROC = std::mem::transmute(PREV_PROC.load(Ordering::Relaxed));
        match msg {
            WM_SIZING => {
                constrain(hwnd, wparam.0 as u32, &mut *(lparam.0 as *mut RECT));
                return LRESULT(1);
            }
            WM_NCHITTEST => {
                let hit = CallWindowProcW(prev, hwnd, msg, wparam, lparam);
                if hit.0 as u32 == HTCLIENT && !is_fullscreen(hwnd) {
                    let mut p = point_from_lparam(lparam);
                    let _ = ScreenToClient(hwnd, &mut p);
                    if !in_controls(p.x, p.y) {
                        return LRESULT(HTCAPTION as isize);
                    }
                }
                return hit;
            }
            // Let Windows run its own move loop (snapping included) without gpui in the way.
            WM_NCLBUTTONDOWN if wparam.0 as u32 == HTCAPTION => {
                if let Some(tx) = EVENTS.get() {
                    let _ = tx.unbounded_send(Event::PressOutside);
                }
                return DefWindowProcW(hwnd, msg, wparam, lparam);
            }
            // No maximize on double-click (fullscreen is the button, F or F11).
            WM_NCLBUTTONDBLCLK if wparam.0 as u32 == HTCAPTION => {
                return LRESULT(0);
            }
            WM_SETCURSOR if HIDE_CURSOR.load(Ordering::Relaxed) => {
                SetCursor(None);
                return LRESULT(1);
            }
            _ => {}
        }
        CallWindowProcW(prev, hwnd, msg, wparam, lparam)
    }
}

unsafe fn constrain(hwnd: HWND, edge: u32, r: &mut RECT) {
    let aspect = f32::from_bits(ASPECT.load(Ordering::Relaxed));
    let (mut wr, mut cr) = (RECT::default(), RECT::default());
    unsafe {
        if GetWindowRect(hwnd, &mut wr).is_err() || GetClientRect(hwnd, &mut cr).is_err() {
            return;
        }
    }
    // Invisible resize borders etc. that sit outside the client area.
    let dx = (wr.right - wr.left) - (cr.right - cr.left);
    let dy = (wr.bottom - wr.top) - (cr.bottom - cr.top);
    let mut cw = (r.right - r.left - dx).max(1) as f32;
    let mut ch = (r.bottom - r.top - dy).max(1) as f32;
    match edge {
        WMSZ_LEFT | WMSZ_RIGHT => ch = cw / aspect,
        WMSZ_TOP | WMSZ_BOTTOM => cw = ch * aspect,
        _ => {
            // Corner: follow whichever axis the user dragged further.
            if cw / aspect > ch {
                ch = cw / aspect;
            } else {
                cw = ch * aspect;
            }
        }
    }
    let w = cw.round() as i32 + dx;
    let h = ch.round() as i32 + dy;
    // Keep the edge opposite the one being dragged where it is.
    match edge {
        WMSZ_LEFT | WMSZ_TOPLEFT | WMSZ_BOTTOMLEFT => r.left = r.right - w,
        _ => r.right = r.left + w,
    }
    match edge {
        WMSZ_TOP | WMSZ_TOPLEFT | WMSZ_TOPRIGHT => r.top = r.bottom - h,
        _ => r.bottom = r.top + h,
    }
}

/// The cursor in client pixels, if it's over this window (and not over another on top of it).
pub fn cursor_in_client(hwnd: HWND) -> Option<(i32, i32)> {
    unsafe {
        let mut p = POINT::default();
        GetCursorPos(&mut p).ok()?;
        let top = WindowFromPoint(p);
        if top != hwnd && GetAncestor(top, GA_ROOT) != hwnd {
            return None;
        }
        let mut cr = RECT::default();
        GetClientRect(hwnd, &mut cr).ok()?;
        let _ = ScreenToClient(hwnd, &mut p);
        (p.x >= 0 && p.y >= 0 && p.x < cr.right && p.y < cr.bottom).then_some((p.x, p.y))
    }
}

pub fn left_button_down() -> bool {
    unsafe { GetAsyncKeyState(VK_LBUTTON.0 as i32) as u16 & 0x8000 != 0 }
}

/// Whether Windows wants interface animations (Settings > Accessibility > Visual effects >
/// Animation effects).
pub fn animations_enabled() -> bool {
    use windows::Win32::UI::WindowsAndMessaging::{
        SPI_GETCLIENTAREAANIMATION, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SystemParametersInfoW,
    };
    let mut on = windows::core::BOOL(1);
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            Some(&mut on as *mut _ as *mut _),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
    };
    ok.is_err() || on.as_bool()
}

pub fn set_topmost(hwnd: HWND, on: bool) {
    unsafe {
        let _ = SetWindowPos(
            hwnd,
            Some(if on { HWND_TOPMOST } else { HWND_NOTOPMOST }),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
}

/// Saves what the window's client area shows on screen to a PNG: the composited result, so
/// it includes every layer (ours and gpui's). Debug only (`--capture-after`).
pub fn capture(hwnd: HWND, path: &std::path::Path) -> anyhow::Result<()> {
    use windows::Win32::Graphics::Gdi::*;
    let mut r = RECT::default();
    let mut origin = POINT { x: 0, y: 0 };
    unsafe {
        GetClientRect(hwnd, &mut r)?;
        let _ = ClientToScreen(hwnd, &mut origin);
    }
    let (w, h) = (r.right - r.left, r.bottom - r.top);
    if w <= 0 || h <= 0 {
        anyhow::bail!("window has no client area");
    }
    let mut px = vec![0u8; (w * h * 4) as usize];
    unsafe {
        let screen = GetDC(None);
        let mem = CreateCompatibleDC(Some(screen));
        let bmp = CreateCompatibleBitmap(screen, w, h);
        let old = SelectObject(mem, bmp.into());
        let blit = BitBlt(mem, 0, 0, w, h, Some(screen), origin.x, origin.y, SRCCOPY);
        let mut info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        GetDIBits(mem, bmp, 0, h as u32, Some(px.as_mut_ptr() as _), &mut info, DIB_RGB_COLORS);
        SelectObject(mem, old);
        let _ = DeleteObject(bmp.into());
        let _ = DeleteDC(mem);
        ReleaseDC(None, screen);
        blit?;
    }
    let rgb: Vec<u8> = px.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]).collect();
    image::save_buffer(path, &rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8)?;
    Ok(())
}
