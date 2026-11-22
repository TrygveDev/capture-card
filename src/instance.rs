//! One Passthru at a time. With "Start with Windows" a copy waits in the tray; opening
//! Passthru again brings that one up instead of starting a second, which would fight it for
//! the capture card. Builds run from `target` (development) skip all of this.

use std::time::Duration;

use windows::Win32::Foundation::{CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, HANDLE, LPARAM, WAIT_ABANDONED, WAIT_OBJECT_0, WPARAM};
use windows::Win32::System::Threading::{CreateMutexW, ReleaseMutex, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::{FindWindowExW, PostMessageW, RegisterWindowMessageW};
use windows::core::{HSTRING, w};

const MUTEX: &str = "Local\\Passthru.App";
/// Messages a running Passthru's window takes from another process.
pub const SHOW: &str = "Passthru.Show";
pub const QUIT: &str = "Passthru.Quit";

fn message(name: &str) -> u32 {
    unsafe { RegisterWindowMessageW(&HSTRING::from(name)) }
}

/// Posts `name` to every Passthru window (gpui's window class or the tray's, titled Passthru,
/// hidden ones too): only the running one understands it. Returns whether there was any.
fn post(name: &str) -> bool {
    let msg = message(name);
    let title = HSTRING::from(crate::app::NAME);
    let mut any = false;
    unsafe {
        for class in [w!("Zed::Window"), crate::tray::CLASS] {
            let mut after = None;
            while let Ok(h) = FindWindowExW(None, after, class, &title) {
                if h.is_invalid() {
                    break;
                }
                any |= PostMessageW(Some(h), msg, WPARAM(0), LPARAM(0)).is_ok();
                after = Some(h);
            }
        }
    }
    any
}

/// Takes the one-instance lock for the rest of this run. If another Passthru holds it, waits
/// up to `wait` for it to exit (an update restarting, say); returns false if it still runs.
pub fn claim(wait: Duration) -> bool {
    unsafe {
        let Ok(mutex) = CreateMutexW(None, true, &HSTRING::from(MUTEX)) else { return true };
        if GetLastError() != ERROR_ALREADY_EXISTS {
            // Held until this process exits.
            return true;
        }
        let r = WaitForSingleObject(mutex, wait.as_millis().min(u32::MAX as u128) as u32);
        if r == WAIT_OBJECT_0 || r == WAIT_ABANDONED {
            return true;
        }
        let _ = CloseHandle(mutex);
        false
    }
}

/// Asks the running Passthru to come to the front. Returns false if there's no window yet.
pub fn show_running() -> bool {
    post(SHOW)
}

/// Asks a running Passthru to quit and waits (a few seconds at most) until it has, so its
/// exe can be replaced (installing a newer one over it).
pub fn quit_running() {
    if !post(QUIT) {
        return;
    }
    unsafe {
        let Ok(mutex) = CreateMutexW(None, false, &HSTRING::from(MUTEX)) else { return };
        let r = WaitForSingleObject(mutex, 5000);
        if r == WAIT_OBJECT_0 || r == WAIT_ABANDONED {
            let _ = ReleaseMutex(mutex);
        }
        let _ = CloseHandle(HANDLE(mutex.0));
    }
    // The process lets go of its exe a moment after its window.
    std::thread::sleep(Duration::from_millis(300));
}

/// The registered numbers of the SHOW and QUIT messages, for the window procedure.
pub fn messages() -> (u32, u32) {
    (message(SHOW), message(QUIT))
}
