//! The app's name and where its things live, in one place.

use std::path::PathBuf;

/// What people see: window title, Start menu, Installed apps.
pub const NAME: &str = "Passthru";
/// The executable, installed and released under this name.
pub const EXE: &str = "passthru.exe";
/// Folder name under %APPDATA% (settings, log, test sessions) and %LOCALAPPDATA%\Programs.
pub const DIR: &str = "Passthru";
/// Releases come from here (the updater).
pub const REPO: &str = "TrygveDev/passthru";
/// Developer mode: test sessions (R to record, B to bookmark) and other tools for working on
/// the app. Off for everyone else; on with `--dev` or the environment variable PASSTHRU_DEV=1
/// (starting a recording with `--record` turns it on too).
pub fn dev_mode() -> bool {
    use std::sync::OnceLock;
    static DEV: OnceLock<bool> = OnceLock::new();
    *DEV.get_or_init(|| {
        std::env::var("PASSTHRU_DEV").is_ok_and(|v| !v.is_empty() && v != "0")
            || std::env::args().any(|a| a == "--dev" || a.starts_with("--record"))
    })
}

/// The data folder before the rename (it's moved over once).
const OLD_DIR: &str = "capture-card";

/// %APPDATA%\Passthru: settings, the log of the last run, test sessions. The first time, the
/// folder from before the rename (%APPDATA%\capture-card) is moved here.
pub fn data_dir() -> Option<PathBuf> {
    let base = PathBuf::from(std::env::var_os("APPDATA")?);
    let dir = base.join(DIR);
    let old = base.join(OLD_DIR);
    if !dir.exists() && old.is_dir() {
        let _ = std::fs::rename(&old, &dir);
    }
    Some(dir)
}
