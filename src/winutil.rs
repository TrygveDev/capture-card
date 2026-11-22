//! The window behaviour gpui doesn't expose, done by wrapping its window procedure:
//! - drag anywhere to move (everything outside the controls is caption), and double-clicks
//!   there do nothing (Windows would maximize),
//! - edge drags keep the video's aspect ratio,
//! - hiding the cursor, and always-on-top,
//! - the tray icon (while Passthru can wait in the background), and hiding/showing the window,
//! - the controls' own window over the video (the overlay): it follows the main window, passes
//!   clicks outside the controls through to it, never takes focus, and can be kept out of
//!   screenshots and recordings.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, AtomicU32, Ordering};

use futures::channel::mpsc::UnboundedSender;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{ClientToScreen, ScreenToClient};
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
    /// Show the window: the tray icon clicked, Open in its menu, another Passthru started.
    Open,
    /// Quit: from the tray icon's menu, an installer, or an update restarting.
    Quit,
    /// The controls' close button: to the tray, or quit (see `Viewer::close`).
    Close,
    /// The controls' fullscreen button (they're in another window: see the overlay).
    ToggleFullscreen,
}

/// The controls' window, and its window procedure before ours.
static OVERLAY: AtomicIsize = AtomicIsize::new(0);
static OVERLAY_PREV_PROC: AtomicIsize = AtomicIsize::new(0);
static MAIN: AtomicIsize = AtomicIsize::new(0);
/// Posted to the overlay to move it onto the main window (wparam: the main window's
/// SET_WINDOW_POS_FLAGS). Posted, not done right away: gpui only takes a new size from the
/// message loop (inside one of its updates the change is lost).
const FOLLOW_MSG: u32 = WM_APP + 2;

/// The tray icon's messages to the window.
pub const TRAY_MSG: u32 = WM_APP + 1;
const TRAY_ID: u32 = 1;
static TRAY_ON: AtomicBool = AtomicBool::new(false);
/// Explorer's "the taskbar was (re)created" message: the icon has to be added again.
static TASKBAR_CREATED: AtomicU32 = AtomicU32::new(0);
/// Another Passthru asking this one to come up, or to quit (see instance.rs).
static SHOW_MSG: AtomicU32 = AtomicU32::new(0);
static QUIT_MSG: AtomicU32 = AtomicU32::new(0);

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
    let (show, quit) = crate::instance::messages();
    SHOW_MSG.store(show, Ordering::Relaxed);
    QUIT_MSG.store(quit, Ordering::Relaxed);
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
            WM_WINDOWPOSCHANGED => {
                let r = CallWindowProcW(prev, hwnd, msg, wparam, lparam);
                let pos = &*(lparam.0 as *const WINDOWPOS);
                follow(hwnd, pos.flags);
                return r;
            }
            TRAY_MSG => {
                match (lparam.0 & 0xFFFF) as u32 {
                    WM_LBUTTONUP => send(Event::Open),
                    WM_RBUTTONUP | WM_CONTEXTMENU => {
                        if let Some(e) = tray_menu(hwnd) {
                            send(e);
                        }
                    }
                    _ => {}
                }
                return LRESULT(0);
            }
            m if m != 0 && m == SHOW_MSG.load(Ordering::Relaxed) => {
                send(Event::Open);
                return LRESULT(0);
            }
            m if m != 0 && m == QUIT_MSG.load(Ordering::Relaxed) => {
                send(Event::Quit);
                return LRESULT(0);
            }
            m if m != 0 && m == TASKBAR_CREATED.load(Ordering::Relaxed) && TRAY_ON.load(Ordering::Relaxed) => {
                TRAY_ON.store(false, Ordering::Relaxed);
                set_tray(hwnd, true);
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

/// Takes `overlay` (a borderless gpui window) as the controls' window over `main`.
pub fn install_overlay(overlay: HWND, main: HWND, hidden_from_capture: bool) {
    unsafe {
        // Owned by the main window: always above it, minimized and restored with it.
        SetWindowLongPtrW(overlay, GWLP_HWNDPARENT, main.0 as isize);
        let ex = GetWindowLongPtrW(overlay, GWL_EXSTYLE) as u32;
        SetWindowLongPtrW(overlay, GWL_EXSTYLE, (ex | WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0) as isize);
        let prev = SetWindowLongPtrW(overlay, GWLP_WNDPROC, overlay_proc as *const () as usize as isize);
        OVERLAY_PREV_PROC.store(prev, Ordering::Relaxed);
        OVERLAY.store(overlay.0 as isize, Ordering::Relaxed);
        MAIN.store(main.0 as isize, Ordering::Relaxed);
        if GetWindowLongPtrW(main, GWL_EXSTYLE) as u32 & WS_EX_TOPMOST.0 != 0 {
            let _ = SetWindowPos(overlay, Some(HWND_TOPMOST), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        }
    }
    set_overlay_hidden_from_capture(hidden_from_capture);
    follow(main, if unsafe { IsWindowVisible(main).as_bool() } { SWP_SHOWWINDOW } else { SWP_HIDEWINDOW });
}

fn overlay() -> Option<HWND> {
    let h = OVERLAY.load(Ordering::Relaxed);
    (h != 0).then_some(HWND(h as _))
}

/// Keeps the controls out of screenshots, recordings and streams (Windows 10 2004 and later;
/// before that they stay in).
pub fn set_overlay_hidden_from_capture(hidden: bool) {
    let Some(o) = overlay() else { return };
    unsafe {
        let _ = SetWindowDisplayAffinity(o, if hidden { WDA_EXCLUDEFROMCAPTURE } else { WDA_NONE });
    }
}

/// Has the overlay put on `main`'s client area, and shown or hidden with it.
fn follow(_main: HWND, flags: SET_WINDOW_POS_FLAGS) {
    if let Some(o) = overlay() {
        unsafe {
            let _ = PostMessageW(Some(o), FOLLOW_MSG, WPARAM(flags.0 as usize), LPARAM(0));
        }
    }
}

fn do_follow(flags: SET_WINDOW_POS_FLAGS) {
    let Some(o) = overlay() else { return };
    let main = HWND(MAIN.load(Ordering::Relaxed) as _);
    unsafe {
        if flags.contains(SWP_HIDEWINDOW) {
            let _ = ShowWindow(o, SW_HIDE);
            return;
        }
        let mut cr = RECT::default();
        let mut origin = POINT::default();
        if GetClientRect(main, &mut cr).is_err() || !ClientToScreen(main, &mut origin).as_bool() {
            return;
        }
        let _ = SetWindowPos(
            o,
            None,
            origin.x,
            origin.y,
            cr.right,
            cr.bottom,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER,
        );
        if flags.contains(SWP_SHOWWINDOW) || (IsWindowVisible(main).as_bool() && !IsWindowVisible(o).as_bool() && !IsIconic(main).as_bool()) {
            let _ = ShowWindow(o, SW_SHOWNOACTIVATE);
        }
    }
}

unsafe extern "system" fn overlay_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe {
        let prev: WNDPROC = std::mem::transmute(OVERLAY_PREV_PROC.load(Ordering::Relaxed));
        match msg {
            // Only the controls are here; everything else is the video's window under it.
            WM_NCHITTEST => {
                let mut p = point_from_lparam(lparam);
                let _ = ScreenToClient(hwnd, &mut p);
                return LRESULT(if in_controls(p.x, p.y) { HTCLIENT as isize } else { HTTRANSPARENT as isize });
            }
            FOLLOW_MSG => {
                do_follow(SET_WINDOW_POS_FLAGS(wparam.0 as u32));
                return LRESULT(0);
            }
            // Clicking the controls leaves the main window active (it has the keyboard).
            WM_MOUSEACTIVATE => return LRESULT(MA_NOACTIVATE as isize),
            // Nothing to size or frame: the client area is the whole window.
            WM_NCCALCSIZE if wparam.0 != 0 => return LRESULT(0),
            _ => {}
        }
        CallWindowProcW(prev, hwnd, msg, wparam, lparam)
    }
}

fn send(e: Event) {
    if let Some(tx) = EVENTS.get() {
        let _ = tx.unbounded_send(e);
    }
}

/// The tray icon's right-click menu: Open, Quit. Returns what was picked.
pub fn tray_menu(hwnd: HWND) -> Option<Event> {
    unsafe {
        let Ok(menu) = CreatePopupMenu() else { return None };
        let _ = AppendMenuW(menu, MF_STRING, 1, &windows::core::HSTRING::from("Open Passthru"));
        let _ = AppendMenuW(menu, MF_SEPARATOR, 0, None);
        let _ = AppendMenuW(menu, MF_STRING, 2, &windows::core::HSTRING::from("Quit"));
        let mut p = POINT::default();
        let _ = GetCursorPos(&mut p);
        // Without this the menu doesn't close when clicking elsewhere.
        let _ = SetForegroundWindow(hwnd);
        let picked = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, p.x, p.y, None, hwnd, None);
        let _ = DestroyMenu(menu);
        match picked.0 {
            1 => Some(Event::Open),
            2 => Some(Event::Quit),
            _ => None,
        }
    }
}

/// Shows or removes the tray icon (the app's own icon, "Passthru" as its tooltip).
pub fn set_tray(hwnd: HWND, on: bool) {
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::Shell::{NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD, NIM_DELETE, NOTIFYICONDATAW, Shell_NotifyIconW};
    if TRAY_ON.swap(on, Ordering::Relaxed) == on {
        return;
    }
    unsafe {
        if TASKBAR_CREATED.load(Ordering::Relaxed) == 0 {
            TASKBAR_CREATED.store(RegisterWindowMessageW(&windows::core::HSTRING::from("TaskbarCreated")), Ordering::Relaxed);
        }
        let mut data = NOTIFYICONDATAW {
            cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: hwnd,
            uID: TRAY_ID,
            uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
            uCallbackMessage: TRAY_MSG,
            ..Default::default()
        };
        if on {
            let module = GetModuleHandleW(None).unwrap_or_default();
            data.hIcon = LoadIconW(Some(module.into()), windows::core::PCWSTR(1 as _)).unwrap_or_default();
            let tip: Vec<u16> = crate::app::NAME.encode_utf16().collect();
            data.szTip[..tip.len()].copy_from_slice(&tip);
            let _ = Shell_NotifyIconW(NIM_ADD, &data);
        } else {
            let _ = Shell_NotifyIconW(NIM_DELETE, &data);
        }
    }
}

/// Hides the window (Passthru waits in the tray) or shows it and brings it to the front.
pub fn set_shown(hwnd: HWND, shown: bool) {
    unsafe {
        if shown {
            let _ = ShowWindow(hwnd, if IsIconic(hwnd).as_bool() { SW_RESTORE } else { SW_SHOW });
            let _ = SetForegroundWindow(hwnd);
        } else {
            let _ = ShowWindow(hwnd, SW_HIDE);
        }
    }
}

/// Minimised to the taskbar.
pub fn minimised(hwnd: HWND) -> bool {
    unsafe { IsIconic(hwnd).as_bool() }
}

/// The cursor in client pixels, if it's over this window (and not over another on top of it).
pub fn cursor_in_client(hwnd: HWND) -> Option<(i32, i32)> {
    unsafe {
        let mut p = POINT::default();
        GetCursorPos(&mut p).ok()?;
        let top = WindowFromPoint(p);
        if top != hwnd && GetAncestor(top, GA_ROOT) != hwnd && Some(top) != overlay() {
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
        // The controls' window too, so it stays above the video in the topmost band.
        for h in [Some(hwnd), overlay()].into_iter().flatten() {
            let _ = SetWindowPos(
                h,
            Some(if on { HWND_TOPMOST } else { HWND_NOTOPMOST }),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            );
        }
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
