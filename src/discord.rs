//! The Discord status: "Playing Passthru · On a console", with the Passthru icon and the time
//! played, while there's a picture. It talks to the Discord desktop app over the app's local
//! pipe (Rich Presence), so nothing leaves the PC from here; without Discord running it does
//! nothing, and it picks Discord up again if Discord starts or restarts later.

use std::fs::File;
use std::io::{Read, Write};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::{app, log};

/// How often a lost Discord is looked for again (and a live one checked on).
const CHECK_EVERY: Duration = Duration::from_secs(15);
/// The second line under the app's name.
const DETAILS: &str = "On a console";

/// Frame kinds on the pipe.
const OP_HANDSHAKE: u32 = 0;
const OP_FRAME: u32 = 1;
const OP_CLOSE: u32 = 2;

/// The icon shown with the status: the app's own, from the repository.
fn icon_url() -> String {
    format!("https://raw.githubusercontent.com/{}/main/assets/icon.png", app::REPO)
}

/// Sets the status from any thread; a thread of its own does the talking.
pub struct Presence {
    tx: Sender<Option<u64>>,
    shown: Option<u64>,
}

impl Presence {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel();
        let _ = std::thread::Builder::new().name("discord".into()).spawn(move || run(rx));
        Presence { tx, shown: None }
    }

    /// Shows the status with its timer counting from `since_ms` (Unix time in milliseconds),
    /// or clears it with None. Cheap to call every tick: only a change is passed on.
    pub fn set(&mut self, since_ms: Option<u64>) {
        if since_ms != self.shown {
            self.shown = since_ms;
            let _ = self.tx.send(since_ms);
        }
    }
}

fn run(rx: Receiver<Option<u64>>) {
    let mut want: Option<u64> = None;
    let mut conn: Option<Conn> = None;
    // The last failure logged, so a Discord that stays closed isn't logged every check.
    let mut last_error = String::new();
    loop {
        match rx.recv_timeout(CHECK_EVERY) {
            Ok(w) => want = w,
            Err(RecvTimeoutError::Timeout) => {}
            // Passthru is closing: dropping the pipe clears the status.
            Err(RecvTimeoutError::Disconnected) => return,
        }
        while let Ok(w) = rx.try_recv() {
            want = w;
        }
        if conn.as_mut().is_some_and(|c| !c.alive()) {
            log::line("discord: the app went away");
            conn = None;
        }
        let Some(since) = want else {
            // Closing the pipe clears the status at once.
            conn = None;
            continue;
        };
        let result = (|| -> Result<()> {
            if conn.is_none() {
                conn = Some(Conn::open()?);
            }
            let c = conn.as_mut().unwrap();
            if c.shown != Some(since) {
                c.set_activity(since)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => last_error.clear(),
            Err(e) => {
                conn = None;
                let e = format!("{e:#}");
                if e != last_error {
                    log::line(&format!("discord: {e}"));
                    last_error = e;
                }
            }
        }
    }
}

/// A connection to the Discord app.
struct Conn {
    pipe: File,
    /// The timer start last set on this connection.
    shown: Option<u64>,
    nonce: u64,
}

impl Conn {
    /// Finds Discord's pipe (one of ten, the first that's there) and introduces the app.
    fn open() -> Result<Conn> {
        let pipe = (0..10)
            .find_map(|i| {
                File::options().read(true).write(true).open(format!(r"\\.\pipe\discord-ipc-{i}")).ok()
            })
            .context("Discord isn't running")?;
        let mut conn = Conn { pipe, shown: None, nonce: 0 };
        conn.send(OP_HANDSHAKE, &json!({ "v": 1, "client_id": app::DISCORD_APP_ID }))?;
        let (op, reply) = conn.receive()?;
        if op != OP_FRAME || reply["evt"] != "READY" {
            bail!("Discord refused the connection: {}", describe(&reply));
        }
        log::line("discord: connected");
        Ok(conn)
    }

    fn set_activity(&mut self, since_ms: u64) -> Result<()> {
        self.nonce += 1;
        self.send(OP_FRAME, &activity_command(since_ms, std::process::id(), self.nonce))?;
        let (_, reply) = self.receive()?;
        if reply["evt"] == "ERROR" {
            bail!("Discord didn't take the status: {}", describe(&reply));
        }
        self.shown = Some(since_ms);
        Ok(())
    }

    /// Whether Discord is still at the other end; anything it sent unasked is read and dropped.
    fn alive(&mut self) -> bool {
        use std::os::windows::io::AsRawHandle;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Pipes::PeekNamedPipe;
        let mut waiting = 0u32;
        let handle = HANDLE(self.pipe.as_raw_handle());
        if unsafe { PeekNamedPipe(handle, None, 0, None, Some(&mut waiting), None) }.is_err() {
            return false;
        }
        let mut sink = vec![0u8; waiting as usize];
        waiting == 0 || self.pipe.read_exact(&mut sink).is_ok()
    }

    fn send(&mut self, op: u32, body: &Value) -> Result<()> {
        self.pipe.write_all(&frame(op, body)).context("writing to Discord")
    }

    fn receive(&mut self) -> Result<(u32, Value)> {
        let mut header = [0u8; 8];
        self.pipe.read_exact(&mut header).context("reading from Discord")?;
        let op = u32::from_le_bytes(header[..4].try_into().unwrap());
        let len = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        if len > 1 << 20 {
            bail!("Discord sent {len} bytes");
        }
        let mut body = vec![0u8; len];
        self.pipe.read_exact(&mut body).context("reading from Discord")?;
        let value = serde_json::from_slice(&body).context("reading Discord's reply")?;
        if op == OP_CLOSE {
            bail!("Discord closed the connection: {}", describe(&value));
        }
        Ok((op, value))
    }
}

/// One message on the pipe: its kind and length (little-endian), then the JSON.
fn frame(op: u32, body: &Value) -> Vec<u8> {
    let json = body.to_string();
    let mut out = Vec::with_capacity(8 + json.len());
    out.extend_from_slice(&op.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out
}

fn activity_command(since_ms: u64, pid: u32, nonce: u64) -> Value {
    json!({
        "cmd": "SET_ACTIVITY",
        "args": {
            "pid": pid,
            "activity": {
                "type": 0,
                "details": DETAILS,
                "timestamps": { "start": since_ms },
                "assets": { "large_image": icon_url(), "large_text": app::NAME },
            },
        },
        "nonce": nonce.to_string(),
    })
}

/// An error reply's message (or the whole reply, if it has none).
fn describe(reply: &Value) -> String {
    let message = reply["data"]["message"].as_str().or(reply["message"].as_str());
    message.map_or_else(|| reply.to_string(), str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_carry_their_kind_and_length() {
        let f = frame(OP_HANDSHAKE, &json!({ "v": 1 }));
        assert_eq!(&f[..4], &[0, 0, 0, 0]);
        assert_eq!(u32::from_le_bytes(f[4..8].try_into().unwrap()) as usize, f.len() - 8);
        assert_eq!(&f[8..], br#"{"v":1}"#);
    }

    #[test]
    fn the_status_says_where_and_since_when() {
        let c = activity_command(1_700_000_000_000, 42, 3);
        assert_eq!(c["cmd"], "SET_ACTIVITY");
        assert_eq!(c["nonce"], "3");
        let a = &c["args"]["activity"];
        assert_eq!(c["args"]["pid"], 42);
        assert_eq!(a["details"], "On a console");
        assert_eq!(a["timestamps"]["start"], 1_700_000_000_000u64);
        assert!(a["assets"]["large_image"].as_str().unwrap().starts_with("https://"));
    }

    #[test]
    fn errors_are_described_by_their_message() {
        assert_eq!(describe(&json!({ "evt": "ERROR", "data": { "code": 4000, "message": "bad" } })), "bad");
        assert_eq!(describe(&json!({ "code": 4000, "message": "Invalid Client ID" })), "Invalid Client ID");
    }

    /// Shows the status in the Discord app running on this PC for 30 s (look at your profile).
    #[test]
    #[ignore = "needs the Discord app running"]
    fn shows_in_discord() {
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        let mut conn = Conn::open().expect("connecting");
        conn.set_activity(since).expect("setting the status");
        std::thread::sleep(Duration::from_secs(30));
        assert!(conn.alive());
    }
}
