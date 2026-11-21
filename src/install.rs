//! Per-user install, the way an installer would do it, so Windows treats this as a real app:
//! - the exe in %LOCALAPPDATA%\Programs\Passthru,
//! - a Start Menu shortcut carrying the app's AppUserModelID (search, pinning, grouping),
//! - an entry under Settings > Apps > Installed apps, with a working Uninstall,
//! - an App Paths entry so `passthru` works from Run (Win+R).
//!
//! Run from anywhere else, the app offers to install itself (`offer`); also `--install` and
//! `--uninstall`. No admin rights needed. Installing also removes the app's install from
//! before it was renamed (Capture Card).

use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::ERROR_SUCCESS;
use windows::Win32::Storage::EnhancedStorage::PKEY_AppUserModel_ID;
use windows::Win32::System::Com::StructuredStorage::{
    PROPVARIANT, PROPVARIANT_0, PROPVARIANT_0_0, PROPVARIANT_0_0_0, PropVariantClear,
};
use windows::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemAlloc,
    IPersistFile,
};
use windows::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, KEY_WRITE, REG_DWORD, REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey,
    RegCreateKeyExW, RegDeleteTreeW, RegSetValueExW,
};
use windows::Win32::System::Variant::VT_LPWSTR;
use windows::Win32::UI::Shell::PropertiesSystem::IPropertyStore;
use windows::Win32::UI::Shell::{
    IShellLinkW, SHCNE_ASSOCCHANGED, SHCNE_CREATE, SHCNE_DELETE, SHCNF_IDLIST, SHCNF_PATHW,
    SHChangeNotify, SetCurrentProcessExplicitAppUserModelID, ShellLink,
};
use windows::Win32::UI::WindowsAndMessaging::{MB_ICONINFORMATION, MB_OK, MessageBoxW};
use windows::core::{HSTRING, Interface, PCWSTR, PWSTR};

/// Ties the running window to the Start Menu shortcut, so pinning and taskbar grouping work.
pub const APP_ID: &str = "Passthru.App";
const NAME: &str = crate::app::NAME;
const UNINSTALL_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Passthru";
const APP_PATHS_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\App Paths\passthru.exe";
/// The install from before the rename, removed when installing.
const OLD_UNINSTALL_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\CaptureCard";
const OLD_APP_PATHS_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\App Paths\capture-card.exe";
const OLD_SHORTCUT: &str = "Capture Card.lnk";
const OLD_DIR: &str = "capture-card";
const OLD_EXE: &str = "capture-card.exe";

/// Call before any window is created.
pub fn set_process_app_id() {
    unsafe {
        let _ = SetCurrentProcessExplicitAppUserModelID(&HSTRING::from(APP_ID));
    }
}

fn programs_dir() -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set")?;
    let base = PathBuf::from(base);
    if !base.is_absolute() || base.as_os_str().is_empty() {
        bail!("LOCALAPPDATA is not an absolute path");
    }
    Ok(base.join("Programs"))
}

pub fn install_dir() -> Result<PathBuf> {
    Ok(programs_dir()?.join(crate::app::DIR))
}

/// Where the installed exe is (whether or not it's there).
pub fn installed_exe() -> Result<PathBuf> {
    Ok(install_dir()?.join(crate::app::EXE))
}

/// This process is the installed copy.
pub fn running_installed() -> bool {
    let (Ok(me), Ok(inst)) = (std::env::current_exe().and_then(|p| p.canonicalize()), installed_exe()) else {
        return false;
    };
    inst.canonicalize().is_ok_and(|i| i == me)
}

/// The installed version, as registered (None: not installed).
pub fn installed_version() -> Option<String> {
    use windows::Win32::System::Registry::{RRF_RT_REG_SZ, RegGetValueW};
    if !installed_exe().ok()?.exists() {
        return None;
    }
    let mut buf = [0u16; 64];
    let mut len = (buf.len() * 2) as u32;
    let err = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            &HSTRING::from(UNINSTALL_KEY),
            &HSTRING::from("DisplayVersion"),
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr() as *mut _),
            Some(&mut len),
        )
    };
    (err == ERROR_SUCCESS).then(|| String::from_utf16_lossy(&buf[..(len as usize / 2).saturating_sub(1)]))
}

/// The installed copy after an update: brings Installed apps up to this version.
pub fn refresh_registration() {
    if running_installed()
        && installed_version().as_deref() != Some(env!("CARGO_PKG_VERSION"))
        && let (Ok(dir), Ok(exe)) = (install_dir(), installed_exe())
    {
        let _ = register(&dir, &exe);
    }
}

/// Starts the installed copy (after installing from elsewhere).
pub fn launch_installed() -> Result<()> {
    std::process::Command::new(installed_exe()?).spawn().context("starting the installed app")?;
    Ok(())
}

/// Running from somewhere else (Downloads, say), with it not installed or an older version
/// installed: asks whether to install, and if so installs and starts the installed copy.
/// Returns true when this process should exit.
pub fn offer() -> bool {
    use windows::Win32::UI::WindowsAndMessaging::{IDYES, MB_ICONQUESTION, MB_YESNO};
    let mine = env!("CARGO_PKG_VERSION");
    let installed = installed_version();
    if running_installed() || installed.as_deref().is_some_and(|v| !crate::update::newer(mine, v)) {
        return false;
    }
    let text = match &installed {
        Some(v) => format!("Passthru {v} is installed. Update it to {mine}?\n\nNo runs this copy once without installing."),
        None => "Install Passthru on this PC?\n\nIt goes in your Start menu and Installed apps (no admin rights needed) \
                 and keeps itself up to date.\n\nNo runs this copy once without installing."
            .to_string(),
    };
    let answer = unsafe { MessageBoxW(None, &HSTRING::from(text), &HSTRING::from(NAME), MB_YESNO | MB_ICONQUESTION) };
    if answer != IDYES {
        return false;
    }
    match install().and_then(|()| launch_installed()) {
        Ok(()) => true,
        Err(e) => {
            unsafe {
                MessageBoxW(
                    None,
                    &HSTRING::from(format!("Couldn't install: {e:#}\n\nThis copy runs without installing.")),
                    &HSTRING::from(NAME),
                    MB_OK | MB_ICONINFORMATION,
                );
            }
            false
        }
    }
}

fn shortcut_path() -> Result<PathBuf> {
    let base = std::env::var_os("APPDATA").context("APPDATA is not set")?;
    Ok(PathBuf::from(base).join(r"Microsoft\Windows\Start Menu\Programs").join(format!("{NAME}.lnk")))
}

pub fn install() -> Result<()> {
    let dir = install_dir()?;
    let target = installed_exe()?;
    let current = std::env::current_exe()?.canonicalize()?;
    std::fs::create_dir_all(&dir)?;
    if target.exists() && target.canonicalize()? == current {
        // Already running the installed copy: just (re)register it.
    } else {
        std::fs::copy(&current, &target)
            .with_context(|| format!("copying to {} (is Passthru still open?)", target.display()))?;
    }

    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
    }
    let lnk = shortcut_path()?;
    create_shortcut(&lnk, &target, &dir)?;
    register(&dir, &target)?;
    remove_old_install();
    unsafe {
        let wide = HSTRING::from(lnk.as_os_str());
        SHChangeNotify(SHCNE_CREATE, SHCNF_PATHW, Some(wide.as_ptr() as _), None);
        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None);
    }
    println!("installed {}", target.display());
    println!("start menu: {}", lnk.display());
    Ok(())
}

fn create_shortcut(lnk: &Path, target: &Path, dir: &Path) -> Result<()> {
    unsafe {
        let link: IShellLinkW = CoCreateInstance(&ShellLink, None, CLSCTX_INPROC_SERVER)?;
        link.SetPath(&HSTRING::from(target.as_os_str()))?;
        link.SetWorkingDirectory(&HSTRING::from(dir.as_os_str()))?;
        link.SetIconLocation(&HSTRING::from(target.as_os_str()), 0)?;
        link.SetDescription(&HSTRING::from("Low-latency capture card viewer with frame generation"))?;

        // The AppUserModelID, as a VT_LPWSTR whose buffer is COM-allocated, the way
        // InitPropVariantFromString does it, and released with PropVariantClear.
        let wide: Vec<u16> = APP_ID.encode_utf16().chain([0]).collect();
        let buf = CoTaskMemAlloc(wide.len() * 2) as *mut u16;
        if buf.is_null() {
            bail!("out of memory");
        }
        std::ptr::copy_nonoverlapping(wide.as_ptr(), buf, wide.len());
        let mut value = PROPVARIANT {
            Anonymous: PROPVARIANT_0 {
                Anonymous: std::mem::ManuallyDrop::new(PROPVARIANT_0_0 {
                    vt: VT_LPWSTR,
                    wReserved1: 0,
                    wReserved2: 0,
                    wReserved3: 0,
                    Anonymous: PROPVARIANT_0_0_0 { pwszVal: PWSTR(buf) },
                }),
            },
        };
        let store: IPropertyStore = link.cast()?;
        let set = store.SetValue(&PKEY_AppUserModel_ID, &value);
        let _ = PropVariantClear(&mut value);
        set?;
        store.Commit()?;

        let file: IPersistFile = link.cast()?;
        file.Save(&HSTRING::from(lnk.as_os_str()), true)?;
    }
    Ok(())
}

fn register(dir: &Path, exe: &Path) -> Result<()> {
    let exe_s = exe.display().to_string();
    let size_kb = std::fs::metadata(exe).map(|m| (m.len() / 1024) as u32).unwrap_or(0);
    let key = create_key(UNINSTALL_KEY)?;
    let result = (|| {
        set_string(key, "DisplayName", NAME)?;
        set_string(key, "DisplayIcon", &format!("{exe_s},0"))?;
        set_string(key, "DisplayVersion", env!("CARGO_PKG_VERSION"))?;
        set_string(key, "InstallLocation", &dir.display().to_string())?;
        set_string(key, "UninstallString", &format!("\"{exe_s}\" --uninstall"))?;
        set_string(key, "QuietUninstallString", &format!("\"{exe_s}\" --uninstall --quiet"))?;
        set_dword(key, "EstimatedSize", size_kb)?;
        set_dword(key, "NoModify", 1)?;
        set_dword(key, "NoRepair", 1)
    })();
    unsafe {
        let _ = RegCloseKey(key);
    }
    result?;

    let key = create_key(APP_PATHS_KEY)?;
    let result = (|| {
        set_string(key, "", &exe_s)?;
        set_string(key, "Path", &dir.display().to_string())
    })();
    unsafe {
        let _ = RegCloseKey(key);
    }
    result
}

fn create_key(path: &str) -> Result<HKEY> {
    let mut key = HKEY::default();
    let err = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            &HSTRING::from(path),
            None,
            PCWSTR::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE,
            None,
            &mut key,
            None,
        )
    };
    if err != ERROR_SUCCESS {
        bail!("creating HKCU\\{path}: {err:?}");
    }
    Ok(key)
}

fn set_string(key: HKEY, name: &str, value: &str) -> Result<()> {
    let wide: Vec<u16> = value.encode_utf16().chain([0]).collect();
    let bytes = unsafe { std::slice::from_raw_parts(wide.as_ptr() as *const u8, wide.len() * 2) };
    let err = unsafe { RegSetValueExW(key, &HSTRING::from(name), None, REG_SZ, Some(bytes)) };
    if err != ERROR_SUCCESS {
        bail!("writing {name}: {err:?}");
    }
    Ok(())
}

fn set_dword(key: HKEY, name: &str, value: u32) -> Result<()> {
    let err = unsafe { RegSetValueExW(key, &HSTRING::from(name), None, REG_DWORD, Some(&value.to_le_bytes())) };
    if err != ERROR_SUCCESS {
        bail!("writing {name}: {err:?}");
    }
    Ok(())
}

/// The app's install from before it was renamed: shortcut, registry entries and exe. Its
/// settings were moved over already (`app::data_dir`).
fn remove_old_install() {
    if let Ok(base) = std::env::var("APPDATA") {
        let lnk = PathBuf::from(base).join(r"Microsoft\Windows\Start Menu\Programs").join(OLD_SHORTCUT);
        if std::fs::remove_file(&lnk).is_ok() {
            unsafe {
                let wide = HSTRING::from(lnk.as_os_str());
                SHChangeNotify(SHCNE_DELETE, SHCNF_PATHW, Some(wide.as_ptr() as _), None);
            }
        }
    }
    unsafe {
        let _ = RegDeleteTreeW(HKEY_CURRENT_USER, &HSTRING::from(OLD_UNINSTALL_KEY));
        let _ = RegDeleteTreeW(HKEY_CURRENT_USER, &HSTRING::from(OLD_APP_PATHS_KEY));
    }
    if let Ok(programs) = programs_dir() {
        let dir = programs.join(OLD_DIR);
        // Only the exe, then the folder if that left it empty.
        if std::fs::remove_file(dir.join(OLD_EXE)).is_ok() {
            let _ = std::fs::remove_dir(&dir);
        }
    }
}

/// Removes the shortcut, the registry entries and the installed exe. Settings in
/// %APPDATA%\Passthru are left alone, as uninstallers usually do.
pub fn uninstall(quiet: bool) -> Result<()> {
    let lnk = shortcut_path()?;
    if lnk.exists() {
        std::fs::remove_file(&lnk)?;
        unsafe {
            let wide = HSTRING::from(lnk.as_os_str());
            SHChangeNotify(SHCNE_DELETE, SHCNF_PATHW, Some(wide.as_ptr() as _), None);
        }
    }
    unsafe {
        let _ = RegDeleteTreeW(HKEY_CURRENT_USER, &HSTRING::from(UNINSTALL_KEY));
        let _ = RegDeleteTreeW(HKEY_CURRENT_USER, &HSTRING::from(APP_PATHS_KEY));
        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None);
    }

    let dir = install_dir()?;
    let exe = installed_exe()?;
    let current = std::env::current_exe()?.canonicalize().ok();
    if exe.exists() {
        if current.as_deref() == exe.canonicalize().ok().as_deref() {
            // Can't delete a running exe: a detached cmd deletes this one file, then removes
            // the folder only if it's empty (rmdir without /s). Both paths are absolute and
            // fixed under %LOCALAPPDATA%\Programs\Passthru, checked by programs_dir().
            let (exe_s, dir_s) = (exe.display().to_string(), dir.display().to_string());
            if exe_s.contains('"') || dir_s.contains('"') {
                bail!("unexpected quote in install path");
            }
            std::process::Command::new("cmd.exe")
                .raw_arg(format!(
                    "/d /c \"ping -n 3 127.0.0.1 >nul & del /f /q \"{exe_s}\" & rmdir \"{dir_s}\"\""
                ))
                .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
                .spawn()?;
        } else {
            std::fs::remove_file(&exe)?;
            let _ = std::fs::remove_dir(&dir); // only succeeds if empty
        }
    }
    if !quiet {
        unsafe {
            MessageBoxW(
                None,
                &HSTRING::from("Passthru has been uninstalled."),
                &HSTRING::from(NAME),
                MB_OK | MB_ICONINFORMATION,
            );
        }
    }
    Ok(())
}
