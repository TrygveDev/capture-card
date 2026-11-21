# capture-card

A lightweight, low-latency viewer for capture cards on Windows: play a console (PlayStation,
Xbox, Switch…) through a capture card on your PC with as little delay as possible, and
optionally smooth it with frame generation.

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

The capture mode (resolution and frame rate) and upscaling are in the menu on the controls.
Settings are saved in `%APPDATA%\capture-card\settings.json`, and a log of the last run is next
to it in `log.txt`.

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

The executable is at `target\release\capture-card.exe`. To install it for the current user
(Start Menu entry, listed under Installed apps, no admin needed):

```sh
target\release\capture-card.exe --install
```

`--uninstall` removes it again.

### Useful flags

- `--list`: list capture and audio devices
- `--video NAME`, `--audio NAME`: pick a device
- `--mode 2560x1440@60`: pick a capture mode
- `--smooth 120`: frame generation target for this run
- `--stats`: print frame rate and latency every few seconds
- `--test-pattern 60/30 [--test-size 3840x2160]`: run without a capture card (60 Hz signal
  carrying a 30 fps "game")

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
