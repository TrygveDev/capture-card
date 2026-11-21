//! A short record of what happened this run (devices opened, modes, format changes, stalls,
//! errors), in %APPDATA%\Passthru\log.txt next to the settings, so odd behaviour can be
//! looked at afterwards: the installed app has no console. Each run starts a fresh file.
//! Everything also goes to stderr.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Stop writing past this, in case something logs in a loop.
const MAX_BYTES: u64 = 2 << 20;

static FILE: Mutex<Option<(std::fs::File, u64)>> = Mutex::new(None);
static START: OnceLock<Instant> = OnceLock::new();

pub fn path() -> Option<PathBuf> {
    Some(crate::app::data_dir()?.join("log.txt"))
}

/// Starts this run's log (the viewer only: tools and tests don't touch it).
pub fn start() {
    START.get_or_init(Instant::now);
    let Some(p) = path() else { return };
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(f) = std::fs::File::create(&p) {
        *FILE.lock().unwrap() = Some((f, 0));
    }
    line(&format!("{} {} started", crate::app::NAME, env!("CARGO_PKG_VERSION")));
}

/// A panic anywhere ends the app (as `panic = "abort"` would), after writing it to the log:
/// a thread carrying on half-broken is worse, and the installed app has no console to show it.
pub fn exit_on_panic() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // try_lock: the panic may have come from inside `line`, holding the lock.
        if let Ok(mut file) = FILE.try_lock()
            && let Some((f, _)) = file.as_mut()
        {
            let secs = START.get_or_init(Instant::now).elapsed().as_secs_f64();
            let _ = writeln!(f, "{secs:9.3}  panic: {info}");
            let _ = f.flush();
        }
        default(info);
        std::process::abort();
    }));
}

pub fn line(msg: &str) {
    eprintln!("{msg}");
    crate::telemetry::note("log", || msg.to_string());
    let mut file = FILE.lock().unwrap();
    if let Some((f, written)) = file.as_mut()
        && *written < MAX_BYTES
    {
        let secs = START.get_or_init(Instant::now).elapsed().as_secs_f64();
        let text = format!("{secs:9.3}  {msg}\n");
        if f.write_all(text.as_bytes()).is_ok() {
            *written += text.len() as u64;
        }
    }
}
