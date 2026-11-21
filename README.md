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
   version starts the next time you open it (or right away, from the menu). Turn it off in the
   menu under *Update automatically*.

Uninstall it like any other app, from Installed apps. Windows may warn about an unrecognised
app the first time (the exe isn't code-signed): *More info › Run anyway*.

## Features

- **Low latency**: frames go straight from the card to a GPU swap chain (about 1–2 ms from
  arrival to present with frame generation off). Audio passthrough keeps its buffer trimmed so
  delay doesn't build up.
- **Frame generation** ("SMOOTH"): a 30 fps game shown at 60, 120 or 240 fps. Repeated card
  frames are detected, so the game's real frame rate is used, and in-between frames are made
  with bidirectional optical flow on the GPU (D3D11 compute shaders). Any vendor's GPU works.
- **Upscaling**: NVIDIA RTX Video Super Resolution when it's available, otherwise a Lanczos
  shader.
- **HDR**: HDR10 input is tone mapped to SDR.
- **FPS counter** with frame-time graphs, showing the game's own rate next to the displayed rate.
- A borderless window with no clutter: drag anywhere to move, edges keep the aspect ratio, and
  the controls hide when the mouse is still. Always-on-top and fullscreen are one key away.

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
| `R` | Start / stop a test session (see below) |
| `B` | Bookmark this moment in the session ("something looked wrong here") |

The capture mode (resolution and frame rate), colour range, upscaling and updates are in the
menu on the controls. Settings are saved in `%APPDATA%\Passthru\settings.json`, and a log of
the last run is next to it in `log.txt`.

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

To find out what happened when something looked or felt wrong, record a session: press `R`
in the app (or start it with `--record`), play, press `B` the moment anything looks off, and
press `R` again to stop. A red REC badge shows while recording. Each session is a folder in
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
