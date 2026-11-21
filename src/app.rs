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
