//! Volume, mute, HDR, smoothing, the FPS counter, capture mode, upscaling, always-on-top and
//! window placement, kept in %APPDATA%\capture-card\settings.json.

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
            range: crate::video::ColourRange::Auto,
            window: None,
            run_only: RunOnly::default(),
        }
    }
}

fn path() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("APPDATA")?).join("capture-card").join("settings.json"))
}

impl Settings {
    pub fn load() -> Self {
        path()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
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
