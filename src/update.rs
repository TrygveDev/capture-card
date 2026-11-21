//! Keeps the installed app up to date from GitHub releases.
//!
//! The installed copy (only: a copy run from Downloads or a dev build never changes itself)
//! asks GitHub for the latest release at most once a day. If it's newer, it downloads the
//! release's `passthru.exe` in the background, checks it against the SHA-256 published with
//! it (`passthru.exe.sha256`), and swaps it in: the running exe is renamed aside (Windows
//! allows renaming a running exe, not replacing it) and the new one takes its name. The next
//! start runs the new version, which removes the old file. The UI offers a restart meanwhile.
//!
//! Uses Windows' own HTTPS (WinHTTP: system proxy settings, certificate checks) and SHA-256
//! (CNG), so it adds nothing to the app. Nothing personal is sent: the request carries only
//! a generic user agent.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

/// Checks at most this often.
const EVERY: Duration = Duration::from_secs(20 * 60 * 60);
/// A release exe smaller or larger than this isn't one.
const EXE_SIZE: std::ops::Range<usize> = 1 << 20..200 << 20;
const UPDATE_NAME: &str = "passthru.update.exe";
const OLD_NAME: &str = "passthru.old.exe";

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Not the installed copy, or turned off.
    Off(&'static str),
    Checking,
    UpToDate,
    /// Downloaded and swapped in: the next start runs this version.
    Ready(String),
    Failed(String),
}

pub struct Updater {
    state: Mutex<State>,
}

impl Updater {
    pub fn state(&self) -> State {
        self.state.lock().unwrap().clone()
    }

    fn set(&self, s: State) {
        crate::log::line(&format!("update: {s:?}"));
        *self.state.lock().unwrap() = s;
    }
}

/// Starts the updater. `enabled`: the setting. `force`: check even if checked recently.
pub fn spawn(enabled: bool, force: bool) -> Arc<Updater> {
    let u = Arc::new(Updater { state: Mutex::new(State::Off("")) });
    if !crate::install::running_installed() {
        u.set(State::Off("not installed"));
        return u;
    }
    if !enabled {
        u.set(State::Off("turned off"));
        return u;
    }
    let me = u.clone();
    let _ = std::thread::Builder::new().name("update".into()).spawn(move || {
        if !force && !due() {
            me.set(State::UpToDate);
            return;
        }
        me.set(State::Checking);
        let result = crate::install::install_dir().and_then(|dir| check_and_stage(env!("CARGO_PKG_VERSION"), &dir));
        // Only a check that worked counts (offline, or GitHub down: try again next start).
        if result.is_ok() {
            remember_check();
        }
        me.set(match result {
            Ok(Some(v)) => State::Ready(v),
            Ok(None) => State::UpToDate,
            Err(e) => State::Failed(format!("{e:#}")),
        });
    });
    u
}

/// Removes the exe an update replaced (it was still running then).
pub fn clean_up() {
    if let Ok(dir) = crate::install::install_dir() {
        let _ = std::fs::remove_file(dir.join(OLD_NAME));
        let _ = std::fs::remove_file(dir.join(UPDATE_NAME));
    }
}

/// Starts the new version and returns (the caller quits).
pub fn restart() -> Result<()> {
    crate::install::launch_installed()
}

/// Whether version `a` is newer than `b` ("0.2.10" > "0.2.9"; a leading "v" is ignored).
pub fn newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> {
        v.trim().trim_start_matches('v').split(['.', '-', '+']).take(3).map(|p| p.parse().unwrap_or(0)).collect()
    };
    parse(a) > parse(b)
}

fn stamp_path() -> Option<PathBuf> {
    Some(crate::app::data_dir()?.join("update-check"))
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn due() -> bool {
    let last = stamp_path().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| s.trim().parse::<u64>().ok());
    last.is_none_or(|t| now_secs().saturating_sub(t) >= EVERY.as_secs())
}

fn remember_check() {
    if let Some(p) = stamp_path() {
        let _ = std::fs::write(p, now_secs().to_string());
    }
}

/// Checks GitHub; if there's a release newer than `current`, downloads it, verifies it and
/// swaps it in for the exe in `dir`. Returns the new version if it did.
fn check_and_stage(current: &str, dir: &Path) -> Result<Option<String>> {
    let api = format!("https://api.github.com/repos/{}/releases/latest", crate::app::REPO);
    let release: serde_json::Value =
        serde_json::from_slice(&http_get(&api, 1 << 20)?).context("reading the release information")?;
    let tag = release["tag_name"].as_str().context("no tag in the latest release")?;
    if !newer(tag, current) {
        return Ok(None);
    }
    let asset = |name: &str| -> Option<String> {
        release["assets"].as_array()?.iter().find(|a| a["name"] == name)?["browser_download_url"].as_str().map(String::from)
    };
    let exe_url = asset(crate::app::EXE).context("the release has no passthru.exe")?;
    let sum_url = asset(&format!("{}.sha256", crate::app::EXE)).context("the release has no checksum")?;
    let sum_text = String::from_utf8(http_get(&sum_url, 4096)?).context("reading the checksum")?;
    let want = sum_text.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
    if want.len() != 64 || !want.bytes().all(|c| c.is_ascii_hexdigit()) {
        bail!("the release's checksum isn't a SHA-256");
    }
    let exe = http_get(&exe_url, EXE_SIZE.end)?;
    if !EXE_SIZE.contains(&exe.len()) || !exe.starts_with(b"MZ") {
        bail!("the downloaded file isn't an exe ({} bytes)", exe.len());
    }
    let got = hex(&sha256(&exe)?);
    if got != want {
        bail!("the download doesn't match its checksum (got {got}, want {want})");
    }
    stage(dir, &exe)?;
    Ok(Some(tag.trim_start_matches('v').to_string()))
}

/// Puts the new exe in place of the installed one (which may be running).
fn stage(dir: &Path, exe: &[u8]) -> Result<()> {
    let installed = dir.join(crate::app::EXE);
    let update = dir.join(UPDATE_NAME);
    let old = dir.join(OLD_NAME);
    std::fs::write(&update, exe).context("writing the update")?;
    let _ = std::fs::remove_file(&old);
    std::fs::rename(&installed, &old).context("moving the running version aside")?;
    if let Err(e) = std::fs::rename(&update, &installed) {
        // Put the old one back so the app still starts.
        let _ = std::fs::rename(&old, &installed);
        return Err(e).context("putting the update in place");
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(data: &[u8]) -> Result<[u8; 32]> {
    use windows::Win32::Security::Cryptography::{BCRYPT_SHA256_ALG_HANDLE, BCryptHash};
    let mut out = [0u8; 32];
    let status = unsafe { BCryptHash(BCRYPT_SHA256_ALG_HANDLE, None, data, &mut out) };
    if status.is_err() {
        bail!("SHA-256 failed ({status:?})");
    }
    Ok(out)
}

/// GET over HTTPS with WinHTTP (follows redirects, uses the system proxy), at most `limit`
/// bytes.
fn http_get(url: &str, limit: usize) -> Result<Vec<u8>> {
    use windows::Win32::Networking::WinHttp::*;
    use windows::core::{HSTRING, PCWSTR};

    let rest = url.strip_prefix("https://").context("only HTTPS")?;
    let (host, path) = rest.split_once('/').map_or((rest, "/".to_string()), |(h, p)| (h, format!("/{p}")));

    struct H(*mut core::ffi::c_void);
    impl Drop for H {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = WinHttpCloseHandle(self.0);
                }
            }
        }
    }
    unsafe {
        let agent = HSTRING::from(format!("{}-Updater/{}", crate::app::NAME, env!("CARGO_PKG_VERSION")));
        let session = H(WinHttpOpen(
            &agent,
            WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
            PCWSTR::null(),
            PCWSTR::null(),
            0,
        ));
        if session.0.is_null() {
            bail!("WinHttpOpen failed");
        }
        let _ = WinHttpSetTimeouts(session.0, 10_000, 10_000, 15_000, 60_000);
        let connect = H(WinHttpConnect(session.0, &HSTRING::from(host), INTERNET_DEFAULT_HTTPS_PORT, 0));
        if connect.0.is_null() {
            bail!("connecting to {host} failed");
        }
        let request = H(WinHttpOpenRequest(
            connect.0,
            &HSTRING::from("GET"),
            &HSTRING::from(path.as_str()),
            PCWSTR::null(),
            PCWSTR::null(),
            std::ptr::null(),
            WINHTTP_FLAG_SECURE,
        ));
        if request.0.is_null() {
            bail!("WinHttpOpenRequest failed");
        }
        let headers: Vec<u16> = "Accept: application/vnd.github+json, application/octet-stream\r\n".encode_utf16().collect();
        WinHttpSendRequest(request.0, Some(&headers), None, 0, 0, 0).context("sending the request")?;
        WinHttpReceiveResponse(request.0, std::ptr::null_mut()).context("receiving the response")?;
        let mut status = 0u32;
        let mut len = std::mem::size_of::<u32>() as u32;
        WinHttpQueryHeaders(
            request.0,
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            PCWSTR::null(),
            Some(&mut status as *mut _ as *mut _),
            &mut len,
            std::ptr::null_mut(),
        )
        .context("reading the status")?;
        if status != 200 {
            bail!("{host} answered {status}");
        }
        let mut body = Vec::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let mut read = 0u32;
            WinHttpReadData(request.0, buf.as_mut_ptr() as *mut _, buf.len() as u32, &mut read).context("downloading")?;
            if read == 0 {
                break;
            }
            body.extend_from_slice(&buf[..read as usize]);
            if body.len() > limit {
                bail!("{host} sent more than expected");
            }
        }
        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_by_number() {
        assert!(newer("v0.2.1", "0.2.0"));
        assert!(newer("0.10.0", "0.9.9"));
        assert!(newer("1.0.0", "0.99.0"));
        assert!(!newer("v0.2.0", "0.2.0"));
        assert!(!newer("0.1.9", "0.2.0"));
    }

    #[test]
    fn sha256_matches_the_standard() {
        assert_eq!(hex(&sha256(b"abc").unwrap()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn staging_swaps_the_exe_and_keeps_the_old_one_aside() {
        let dir = std::env::temp_dir().join("passthru-update-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(crate::app::EXE), b"old").unwrap();
        stage(&dir, b"MZnew").unwrap();
        assert_eq!(std::fs::read(dir.join(crate::app::EXE)).unwrap(), b"MZnew");
        assert_eq!(std::fs::read(dir.join(OLD_NAME)).unwrap(), b"old");
        assert!(!dir.join(UPDATE_NAME).exists());
    }

    /// The whole update against the real latest release (talks to GitHub; needs a release):
    /// `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn updates_from_the_latest_release() {
        let dir = std::env::temp_dir().join("passthru-update-live-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(crate::app::EXE), b"old").unwrap();
        let v = check_and_stage("0.0.0", &dir).unwrap().expect("a newer release");
        let exe = std::fs::read(dir.join(crate::app::EXE)).unwrap();
        assert!(exe.starts_with(b"MZ") && exe.len() > 1 << 20, "{v}: {} bytes", exe.len());
        assert_eq!(check_and_stage(&v, &dir).unwrap(), None, "nothing newer than itself");
    }

    /// Talks to GitHub: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn reads_a_release_from_github() {
        let body = http_get("https://api.github.com/repos/rust-lang/rust/releases/latest", 1 << 20).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(v["tag_name"].as_str().is_some());
    }
}
