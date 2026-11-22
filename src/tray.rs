//! Waiting in the tray, at next to no cost.
//!
//! Closing to the tray ends the app (its window, GPU device, capture and sound all go with the
//! process) and starts it again as `--background`, which is this: a tray icon and a message
//! loop, before anything of the app is created. With "Open when a signal arrives" a thread also
//! glances at the card in its cheapest mode (`video::watch_picture`); without, the card isn't
//! touched at all, so other programs can have it.
//!
//! Opening returns to `main`, which then starts the app as usual in this same process.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::ProcessStatus::K32EmptyWorkingSet;
use windows::Win32::System::Threading::{
    BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, NORMAL_PRIORITY_CLASS, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE, ProcessPowerThrottling, SetPriorityClass,
    SetProcessInformation,
};
use windows::Win32::UI::HiDpi::{DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{HSTRING, w};

use crate::winutil::{self, Event};

/// The tray window's class (another Passthru looks for it: see instance.rs).
pub const CLASS: windows::core::PCWSTR = w!("Passthru.Tray");

/// The watching thread saw a picture arrive.
const PICTURE_MSG: u32 = WM_APP + 3;
/// Housekeeping: memory handed back once start-up has settled, and updates.
const TIMER: usize = 1;
const TIMER_FIRST_MS: u32 = 10_000;
const TIMER_MS: u32 = 10 * 60 * 1000;

/// How the wait ended.
pub enum Wake {
    /// Show the app.
    Open,
    /// Show the app, as the version that was installed while this waited.
    OpenUpdated,
    Quit,
    /// An update was installed meanwhile: carry on waiting as the new version.
    Updated,
}

const WAITING: u8 = 0;
const OPEN: u8 = 1;
const QUIT: u8 = 2;
static WOKE: AtomicU8 = AtomicU8::new(WAITING);
static TIMER_DUE: AtomicBool = AtomicBool::new(false);
static SHOW_MSG: AtomicU32 = AtomicU32::new(0);
static QUIT_MSG: AtomicU32 = AtomicU32::new(0);
static TASKBAR_CREATED: AtomicU32 = AtomicU32::new(0);

/// What to do while waiting.
pub struct Options {
    /// Watch the card and open when a picture arrives.
    pub open_on_signal: bool,
    /// Closed with a picture showing: only one that arrives after it's gone opens the app.
    pub wait_for_change: bool,
    pub auto_update: bool,
    pub device: Option<String>,
    /// Debug (`--exit-after`): stop waiting after this long.
    pub exit_after: Option<Duration>,
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let wake = |how: u8| {
        WOKE.store(how, Ordering::Relaxed);
        // GetMessage only returns for a posted message.
        unsafe {
            let _ = PostMessageW(Some(hwnd), WM_NULL, WPARAM(0), LPARAM(0));
        }
        LRESULT(0)
    };
    let is = |registered: &AtomicU32| msg != 0 && msg == registered.load(Ordering::Relaxed);
    match msg {
        winutil::TRAY_MSG => match (lparam.0 & 0xFFFF) as u32 {
            WM_LBUTTONUP => wake(OPEN),
            WM_RBUTTONUP | WM_CONTEXTMENU => match winutil::tray_menu(hwnd) {
                Some(Event::Open) => wake(OPEN),
                Some(Event::Quit) => wake(QUIT),
                _ => LRESULT(0),
            },
            _ => LRESULT(0),
        },
        PICTURE_MSG => wake(OPEN),
        WM_CLOSE | WM_ENDSESSION => wake(QUIT),
        WM_TIMER => {
            TIMER_DUE.store(true, Ordering::Relaxed);
            unsafe {
                let _ = PostMessageW(Some(hwnd), WM_NULL, WPARAM(0), LPARAM(0));
            }
            LRESULT(0)
        }
        _ if is(&SHOW_MSG) => wake(OPEN),
        _ if is(&QUIT_MSG) => wake(QUIT),
        _ if is(&TASKBAR_CREATED) => {
            winutil::set_tray(hwnd, false);
            winutil::set_tray(hwnd, true);
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// Efficiency mode while waiting (slow, cool cores and background priority); normal for the app.
fn economise(on: bool) {
    let speed = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        // No mask hands the choice back to Windows.
        ControlMask: if on { speed } else { 0 },
        StateMask: if on { speed } else { 0 },
    };
    unsafe {
        let me = GetCurrentProcess();
        let _ = SetProcessInformation(
            me,
            ProcessPowerThrottling,
            &state as *const _ as _,
            std::mem::size_of_val(&state) as u32,
        );
        let _ = SetPriorityClass(me, if on { BELOW_NORMAL_PRIORITY_CLASS } else { NORMAL_PRIORITY_CLASS });
    }
}

/// Hands memory only start-up needed back to Windows.
fn trim() {
    unsafe {
        let _ = K32EmptyWorkingSet(GetCurrentProcess());
    }
}

/// Waits in the tray until the app should open, or quit.
pub fn wait(opts: &Options) -> Result<Wake> {
    WOKE.store(WAITING, Ordering::Relaxed);
    let (show, quit) = crate::instance::messages();
    SHOW_MSG.store(show, Ordering::Relaxed);
    QUIT_MSG.store(quit, Ordering::Relaxed);
    let hwnd = unsafe {
        // gpui asks for the same later, which is too late once the process has had a window.
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        TASKBAR_CREATED.store(RegisterWindowMessageW(w!("TaskbarCreated")), Ordering::Relaxed);
        let module = GetModuleHandleW(None)?;
        let class = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: module.into(),
            lpszClassName: CLASS,
            ..Default::default()
        };
        RegisterClassW(&class);
        // Never shown. Not a message-only window: those don't hear TaskbarCreated.
        CreateWindowExW(
            WS_EX_TOOLWINDOW,
            CLASS,
            &HSTRING::from(crate::app::NAME),
            WS_POPUP,
            0,
            0,
            0,
            0,
            None,
            None,
            Some(module.into()),
            None,
        )
        .context("creating the tray window")?
    };
    winutil::set_tray(hwnd, true);
    economise(true);
    crate::log::line(if opts.open_on_signal {
        "tray: waiting, watching for a picture"
    } else {
        "tray: waiting (not using the card)"
    });

    let watcher = opts.open_on_signal.then(|| {
        let (stop, stopped) = mpsc::channel::<()>();
        let (device, wait_for_change, target) = (opts.device.clone(), opts.wait_for_change, hwnd.0 as isize);
        let thread = std::thread::spawn(move || {
            if crate::video::watch_picture(device.as_deref(), wait_for_change, &stopped) {
                unsafe {
                    let _ = PostMessageW(Some(HWND(target as _)), PICTURE_MSG, WPARAM(0), LPARAM(0));
                }
            }
        });
        (stop, thread)
    });

    let started = Instant::now();
    let first = opts.exit_after.map_or(TIMER_FIRST_MS, |d| (d.as_millis() as u32).min(TIMER_FIRST_MS));
    unsafe {
        SetTimer(Some(hwnd), TIMER, first, None);
    }
    let mut updater: Option<std::sync::Arc<crate::update::Updater>> = None;
    let mut checked: Option<Instant> = None;
    let mut updated = false;
    let mut msg = MSG::default();
    while WOKE.load(Ordering::Relaxed) == WAITING && unsafe { GetMessageW(&mut msg, None, 0, 0) }.0 > 0 {
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if !TIMER_DUE.swap(false, Ordering::Relaxed) {
            continue;
        }
        if opts.exit_after.is_some_and(|d| started.elapsed() >= d) {
            break;
        }
        // The last check's outcome: it's had ten minutes (or the state says it's still going).
        if let Some(u) = &updater
            && matches!(u.state(), crate::update::State::Ready(_))
        {
            updated = true;
            break;
        }
        if opts.auto_update && checked.is_none_or(|t| t.elapsed() >= crate::UPDATE_EVERY) {
            updater = Some(crate::update::spawn(true, false));
            checked = Some(Instant::now());
        }
        trim();
        let next = match opts.exit_after {
            Some(d) => (d.saturating_sub(started.elapsed()).as_millis() as u32).clamp(100, TIMER_MS),
            None => TIMER_MS,
        };
        unsafe {
            SetTimer(Some(hwnd), TIMER, next, None);
        }
    }

    // The card is free again before the app opens it.
    if let Some((stop, thread)) = watcher {
        let _ = stop.send(());
        let _ = thread.join();
    }
    winutil::set_tray(hwnd, false);
    unsafe {
        let _ = KillTimer(Some(hwnd), TIMER);
        let _ = DestroyWindow(hwnd);
    }
    updated |= updater.is_some_and(|u| matches!(u.state(), crate::update::State::Ready(_)));
    let wake = match WOKE.load(Ordering::Relaxed) {
        OPEN if updated => Wake::OpenUpdated,
        OPEN => Wake::Open,
        _ if updated => Wake::Updated,
        _ => Wake::Quit,
    };
    if matches!(wake, Wake::Open | Wake::OpenUpdated) {
        economise(false);
    }
    crate::log::line(match wake {
        Wake::Open => "tray: opening",
        Wake::OpenUpdated => "tray: updated, opening the new version",
        Wake::Quit => "tray: quitting",
        Wake::Updated => "tray: updated, waiting on as the new version",
    });
    Ok(wake)
}

/// Starts a copy that waits in the tray, once this process has exited.
pub fn hand_over(wait_for_change: bool) -> Result<()> {
    let exe = if crate::install::running_installed() { crate::install::installed_exe()? } else { std::env::current_exe()? };
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--background", "--restarted"]);
    if wait_for_change {
        cmd.arg("--wait-for-change");
    }
    cmd.spawn().context("starting the tray copy")?;
    Ok(())
}
