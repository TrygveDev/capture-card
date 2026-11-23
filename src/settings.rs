//! Volume, mute, HDR, smoothing, the FPS counter, capture mode, upscaling, the settings menu's
//! switches and window placement, kept in %APPDATA%\Passthru\settings.json.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub volume: f32,
    pub muted: bool,
    pub on_top: bool,
    /// Decode the signal as HDR10 and tone map it.
    pub hdr: bool,
    /// Frame generation cap in frames per second: 0 (off), 60, 120 or 240.
    pub smooth: u32,
    /// The FPS counter over the video.
    pub overlay: Overlay,
    /// Capture mode picked in the menu: width, height, fps. None takes the best offered.
    pub mode: Option<[u32; 3]>,
    /// Upscale with RTX Video Super Resolution when it's available (else Lanczos).
    pub rtx: bool,
    /// The installed app checks GitHub for new releases (at most daily) and updates itself.
    pub auto_update: bool,
    /// Go fullscreen whenever the window opens.
    pub start_fullscreen: bool,
    /// Wait in the tray (closing the window hides it) and open when a signal arrives.
    pub open_on_signal: bool,
    /// Close the window once the signal has been gone a while (to the tray, if waiting there).
    pub close_on_signal_loss: bool,
    /// Start with Windows, in the tray (the installed app only).
    pub start_with_windows: bool,
    /// Keep the controls out of screenshots, recordings and streams (OBS, Discord...).
    pub hide_from_capture: bool,
    /// Show "Playing Passthru" in Discord, with the time played, while there's a picture.
    pub discord: bool,
    /// The signal's colour range: detected, or forced limited or full.
    pub range: crate::video::ColourRange,
    /// Last windowed position and size in logical pixels: x, y, width, height.
    pub window: Option<[f32; 4]>,
    /// Values given for this run only on the command line, and what they replaced.
    #[serde(skip)]
    run_only: RunOnly,
}

/// Each is (value for this run, saved value). While a setting still holds its value for the
/// run, saving writes the saved value instead; once it's changed in the app, the change is saved.
#[derive(Clone, Debug, Default)]
struct RunOnly {
    smooth: Option<(u32, u32)>,
    overlay: Option<(Overlay, Overlay)>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            volume: 1.0,
            muted: false,
            on_top: false,
            hdr: true,
            smooth: 0,
            overlay: Overlay::Off,
            mode: None,
            rtx: true,
            auto_update: true,
            start_fullscreen: false,
            open_on_signal: true,
            close_on_signal_loss: true,
            start_with_windows: true,
            hide_from_capture: false,
            discord: true,
            range: crate::video::ColourRange::Auto,
            window: None,
            run_only: RunOnly::default(),
        }
    }
}

fn path() -> Option<PathBuf> {
    Some(crate::app::data_dir()?.join("settings.json"))
}

fn parse(bytes: &[u8]) -> Option<Settings> {
    // Saved by some editors with a byte order mark, which JSON doesn't allow.
    serde_json::from_slice(bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)).ok()
}

impl Settings {
    /// Passthru keeps running in the tray when its window is closed.
    pub fn stays_in_tray(&self) -> bool {
        self.open_on_signal || self.start_with_windows
    }

    pub fn load() -> Self {
        path().and_then(|p| std::fs::read(p).ok()).and_then(|b| parse(&b)).unwrap_or_default()
    }

    /// Smoothing for this run, without saving it (`--smooth`).
    pub fn smooth_for_run(&mut self, smooth: u32) {
        self.run_only.smooth = Some((smooth, self.smooth));
        self.smooth = smooth;
    }

    /// The FPS counter for this run, without saving it (`--fps`).
    pub fn overlay_for_run(&mut self, overlay: Overlay) {
        self.run_only.overlay = Some((overlay, self.overlay));
        self.overlay = overlay;
    }

    /// What goes in the file: run-only values that are still in effect are left out.
    fn to_save(&self) -> Settings {
        let mut s = self.clone();
        if let Some((run, saved)) = self.run_only.smooth
            && s.smooth == run
        {
            s.smooth = saved;
        }
        if let Some((run, saved)) = self.run_only.overlay
            && s.overlay == run
        {
            s.overlay = saved;
        }
        s
    }

    pub fn save(&self) {
        let Some(p) = path() else { return };
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_vec_pretty(&self.to_save()) {
            let _ = std::fs::write(p, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_files_with_a_byte_order_mark() {
        let s = parse(b"\xEF\xBB\xBF{\"volume\": 0.25, \"open_on_signal\": true}").unwrap();
        assert_eq!((s.volume, s.open_on_signal, s.hide_from_capture), (0.25, true, false));
    }

    #[test]
    fn run_only_values_are_not_saved() {
        let mut s = Settings { smooth: 240, ..Settings::default() };
        s.smooth_for_run(120);
        s.overlay_for_run(Overlay::Graph);
        s.volume = 0.5;
        let saved = s.to_save();
        assert_eq!((saved.smooth, saved.overlay, saved.volume), (240, Overlay::Off, 0.5));
        // Changed in the app afterwards: that is saved.
        s.overlay = Overlay::Counter;
        assert_eq!(s.to_save().overlay, Overlay::Counter);
    }
}

/// What the FPS counter shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Overlay {
    #[default]
    Off,
    /// Frames on screen per second and the game's own rate.
    Counter,
    /// That, more numbers, and frame-time graphs.
    Graph,
}

impl Overlay {
    pub fn next(self) -> Self {
        match self {
            Overlay::Off => Overlay::Counter,
            Overlay::Counter => Overlay::Graph,
            Overlay::Graph => Overlay::Off,
        }
    }
}
