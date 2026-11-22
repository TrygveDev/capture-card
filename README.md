# Passthru

A lightweight, low-latency viewer for capture cards on Windows: play a console (PlayStation,
Xbox, Switch…) through a capture card on your PC with as little delay as possible, and
optionally smooth it with frame generation.

## Download and install

1. Download `passthru.exe` from the [latest release](https://github.com/TrygveDev/passthru/releases/latest).
2. Run it. It offers to install itself: it goes in your Start menu and under Settings › Apps ›
   Installed apps (no admin rights needed). Say no to just run it once.
3. That's it. The installed app keeps itself up to date: it checks for a new release once a
   day, downloads it in the background (checked against its published SHA-256), and the new
   version starts the next time you open it (or right away, from Settings). Turn it off in
   Settings under *Update automatically*.

Uninstall it like any other app, from Installed apps. Windows may warn about an unrecognised
app the first time (the exe isn't code-signed): *More info › Run anyway*.

## Features

- **Low latency**: frames go straight from the card to a GPU swap chain (about 1–2 ms from
  arrival to present with frame generation off). With it on, frames go out on an even grid that
  follows the card's uneven timing without passing it on. Sound is passed on as it arrives, with
  just enough buffered for the card's and the PC's timing.
- **Frame generation** ("SMOOTH"): a 30 fps game shown at 60, 120 or 240 fps. Repeated card
  frames are detected, so the game's real frame rate is used, and in-between frames are made
  with bidirectional optical flow on the GPU (D3D11 compute shaders). Any vendor's GPU works.
- **Upscaling**: NVIDIA RTX Video Super Resolution when it's available, otherwise a Lanczos
  shader.
- **HDR**: HDR10 input is tone mapped to SDR.
- **FPS counter** with frame-time graphs, showing the game's own rate next to the displayed rate.
- A borderless window with no clutter: drag anywhere to move, edges keep the aspect ratio, and
  the controls hide when the mouse is still. Always-on-top and fullscreen are one key away.
- **Hidden from recordings**: the controls stay out of screenshots, OBS and Discord (they see
  only the video), on Windows 10 2004 and later.
- **Ready when the console is**: Passthru can wait in the tray, start with Windows, open when a
  signal arrives and close again when it's gone. In the tray it's only an icon: no CPU, under
  1 MB of memory, no GPU, and the capture card is free for other programs. Watching for a
  signal, it glances at the card once a second in its cheapest mode (under 1% of one core).

## Keys

| Key | Action |
| --- | --- |
| `F` / `F11` / `Enter` | Fullscreen |
| `Esc` | Close menu / leave fullscreen |
| `S` | Frame generation: off → 60 → 120 → 240 (only rates your display can show) |
| `M` | Mute |
| `↑` / `↓` / scroll | Volume |
| `H` | HDR input |
| `O` | FPS counter: off → numbers → graphs |
| `T` / `P` | Always on top |

The capture mode (resolution and frame rate), colour range and upscaling are in the mode menu
on the controls; the window, the background and updates are under the gear (Settings):

| Setting | What it does |
| --- | --- |
| Always on top | Keeps the window above others (also `T`) |
| Open in fullscreen | Goes fullscreen whenever the window opens |
| Hide controls from recordings | Screenshots, OBS and Discord see only the video (on by default) |
| Open when a signal arrives | Closing the window leaves Passthru in the tray; it opens again when a picture arrives. Closed while the console is on, it waits for that picture to go away first |
| Close when the signal is lost | After 10 s without frames, or 30 s of blank picture and silence: back to the tray, or quit if it isn't waiting there |
| Start with Windows | Starts in the tray at sign-in (installed app only) |
| Update automatically | Checks GitHub about once a day; *Check for updates* checks now |

Cards differ in what they send with nothing connected. Frames that stop and flat frames count
as no signal, and so does the Elgato 4K S's *NO SIGNAL* screen. Another card's own no-signal
screen counts as a picture, so the two signal settings won't work with one that shows such a screen.

Only one Passthru runs at a time: opening it again brings up the one already running (in the
tray, say). Settings are saved in `%APPDATA%\Passthru\settings.json`, and a log of
the last run is next to it in `log.txt` (the run before in `log.prev.txt`).

## Tips for the lowest delay

- A higher capture frame rate beats frame generation for input delay. If your console and card
  support 120 Hz (for example 1440p120 or 1080p120), use it and pick that mode in the menu.
- Frame generation has to wait for the next real frame, so it adds about one game frame of delay
  (about 17 ms at 30 → 60, 25 ms at 30 → 120, 8 ms at 60 → 120). Turn it off for competitive play.

## Build

Requires Windows 10/11 and a recent stable Rust.

```sh
cargo build --release
```

The executable is at `target\release\passthru.exe`. To install it for the current user
(Start Menu entry, listed under Installed apps, no admin needed):

```sh
target\release\passthru.exe --install
```

`--uninstall` removes it again. Builds run from `target` never offer to install or update.

### Releasing

Bump `version` in `Cargo.toml`, commit, then tag and push:

```sh
git tag v0.2.1
git push origin v0.2.1
```

GitHub Actions tests and builds it and publishes a release with `passthru.exe` and
`passthru.exe.sha256`; installed copies pick it up within a day. Every push to `main` and
every pull request is built and tested too (the exe is kept as a build artifact).

### Useful flags

- `--list`: list capture and audio devices
- `--video NAME`, `--audio NAME`: pick a device
- `--mode 2560x1440@60`: pick a capture mode
- `--smooth 120`: frame generation target for this run
- `--stats`: print frame rate and latency every few seconds
- `--test-pattern 60/30 [--test-size 3840x2160]`: run without a capture card (60 Hz signal
  carrying a 30 fps "game")

## Test sessions

A developer tool, off in normal use: start the app with `--dev` (or set the environment
variable `PASSTHRU_DEV=1`) to enable it. To find out what happened when something looked or
felt wrong, record a session: press `R` in the app (or start it with `--record`), play, press
`B` the moment anything looks off, and press `R` again to stop. A red REC badge shows while recording. Each session is a folder in
`%APPDATA%\Passthru\sessions\` with:

- every card frame, every frame-generation pair (plan, GPU time, scene-cut and UI tests),
  every present and when it really reached the screen (DXGI frame statistics),
- CPU, memory, GPU load and memory, display rate, four times a second,
- events: settings changes, log lines, bookmarks, signal loss,
- `input_*.mkv`: the card's frames exactly as received (one video frame per card frame),
- `output.mkv`: the screen under the window, each frame tagged with a small barcode in the
  top-left corner, so every recorded frame maps back to the present that made it.

Then analyse it (needs Python 3.9+ and ffmpeg on PATH):

```sh
python tools/analyze.py report --open            # the newest session
python tools/analyze.py report SESSION_FOLDER --open
python tools/analyze.py compare BEFORE AFTER      # e.g. before and after a change
python tools/analyze.py extract SESSION --at 12.5 # that moment's pair as a clip for --gpu-eval
python tools/analyze.py list
```

The report has the session's key numbers (frames on time, drops, frame time percentiles,
latency, GPU time per pair, CPU/GPU/memory), zoomable timelines with the bookmarks marked,
and a ranked list of incidents (drops, frames shown longer or shorter than planned, capture
gaps, GPU overruns, UI fallbacks, bookmarks) with thumbnails of the input pair and of what was
on screen frame by frame.

Flags: `--record-dir DIR` to choose the folder, `--record-no-input` / `--record-no-output` to
leave out a video (recording the screen costs some GPU, so leave it out when measuring load).
`python tools/analyze.py selftest` checks the barcode reader against the app's own shader.

Developer checks: `--gpu-test IMAGE OUTDIR` (synthetic scene) and `--gpu-eval CLIP.y4m OUTDIR`
(drops frames from a real clip, regenerates them and scores them against the originals).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md). The source is open: you're free to use, study, modify
and share it for any noncommercial purpose (personal use, hobby projects, education, research).
**Commercial use is not allowed**, including selling it, selling modified versions, or bundling
it into a paid product. Copies must keep the license and its `Required Notice` lines.

Because commercial use is restricted, this is "source-available" rather than open source in the
OSI sense.

## Credits

Created by [Elliot (@ellipog)](https://github.com/ellipog), who wrote the original code and came
up with the idea. Further work by [Trygve (@TrygveDev)](https://github.com/TrygveDev).
