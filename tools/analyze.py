#!/usr/bin/env python3
"""Analyses capture-card test sessions (see src/telemetry.rs for what a session records).

    python tools/analyze.py report [SESSION] [--open] [--no-thumbs]
        Writes SESSION/report.html: summary, timelines (frame times on screen, pacing, latency,
        GPU, frame-generation tests, rates, CPU/GPU/memory) and a list of incidents (dropped or
        late frames, hitches, capture gaps, UI fallbacks, bookmarks...) with thumbnails from the
        input and output videos. SESSION defaults to the newest one.
    python tools/analyze.py compare A B
        Key numbers of two sessions side by side (before/after a change).
    python tools/analyze.py extract SESSION --at SECONDS [--out FILE]
        The card frames of the frame-generation pair at SECONDS as a .y4m clip, for
        `capture-card --gpu-eval CLIP OUTDIR HELD`.
    python tools/analyze.py list
    python tools/analyze.py marks VIDEO
        Decodes the present numbers from an output video's barcodes.
    python tools/analyze.py selftest [CAPTURE_CARD_EXE]
        Checks the barcode reader against frames drawn by the app's own shader, encoded the
        way a session encodes them.

Needs Python 3.9+ and ffmpeg/ffprobe on PATH (only for videos and thumbnails).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

# The barcode drawn by the present shader (gpu.hlsl, present_ps): cells of MARK_W x MARK_H
# pixels in the top-left corner: white, black, 16 bits low first, parity of each byte.
MARK_W, MARK_H, MARK_CELLS = 8, 8, 20
STRIP_W = MARK_W * MARK_CELLS
# Matroska keeps times to the millisecond: frame times are matched this loosely (still well
# under half a frame at 240 fps).
EPS = 0.0015

# ---------------------------------------------------------------------------------------------
# Loading


def sessions_root() -> Path:
    base = os.environ.get("APPDATA") or tempfile.gettempdir()
    return Path(base) / "capture-card" / "sessions"


def newest_session() -> Path:
    root = sessions_root()
    dirs = sorted((d for d in root.glob("*") if (d / "session.json").exists()), key=lambda d: d.stat().st_mtime)
    if not dirs:
        sys.exit(f"no sessions in {root} (record one with R in the app, or --record)")
    return dirs[-1]


def number(v: str):
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: number(v) for k, v in row.items()} for row in csv.DictReader(f)]


class Session:
    def __init__(self, path: Path):
        self.dir = path
        self.info = json.loads((path / "session.json").read_text(encoding="utf-8"))
        s = self.info.get("session", {})
        self.t0 = s.get("started_us", 0)
        self.t1 = s.get("ended_us")
        for name in ("capture", "frames", "pairs", "pair_gpu", "pair_counts", "presents", "display", "system", "events"):
            setattr(self, name, read_csv(path / f"{name}.csv"))
        if self.t1 is None:
            # Unfinished (the app was killed): the last row of anything.
            ends = [r["t_us"] for r in self.capture[-1:] + self.system[-1:] + self.events[-1:]]
            ends += [r["call_us"] for r in self.presents[-1:]]
            self.t1 = max(ends) if ends else self.t0

    def s(self, us) -> float:
        return (us - self.t0) / 1e6

    @property
    def duration(self) -> float:
        return max(0.0, self.s(self.t1))


# ---------------------------------------------------------------------------------------------
# Small statistics helpers


def pct(values, p):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def fmt(v, digits=1, unit=""):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}{unit}"
    return f"{v}{unit}"


# ---------------------------------------------------------------------------------------------
# ffmpeg


def tool(name: str) -> str | None:
    return shutil.which(name)


def run(args, **kw):
    return subprocess.run(args, capture_output=True, **kw)


def frame_times(video: Path) -> list[float]:
    """Presentation time of every frame, in order."""
    out = run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
               "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(video)], text=True).stdout
    times = []
    for line in out.split():
        try:
            times.append(float(line.strip().strip(",")))
        except ValueError:
            pass
    return times


def decode_strip(px: bytes, width: int) -> int | None:
    """The present number in one frame's barcode row (grey levels), or None if unreadable."""
    def cell(i):
        x0 = i * MARK_W + 2
        vals = px[x0:x0 + MARK_W - 4]
        return sum(vals) / len(vals)
    if width < STRIP_W:
        return None
    white, black = cell(0), cell(1)
    if white - black < 80:
        return None
    mid = (white + black) / 2
    bits = [1 if cell(i) > mid else 0 for i in range(2, MARK_CELLS)]
    code = sum(b << i for i, b in enumerate(bits[:16]))
    if bits[16] != bin(code & 0xFF).count("1") % 2 or bits[17] != bin(code >> 8).count("1") % 2:
        return None
    return code


def decode_marks(video: Path) -> list[tuple[float, int | None]]:
    """(time, 16-bit present number or None) for every frame of an output video."""
    times = frame_times(video)
    rows = MARK_H
    # Every frame as stored: raw output would otherwise be resampled to a constant rate
    # (frames repeated or dropped), and the n-th frame read wouldn't be the file's n-th.
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(video), "-vf",
           f"crop={STRIP_W}:{rows}:0:0,format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = STRIP_W * rows
    out = []
    i = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        mid_row = buf[(rows // 2) * STRIP_W:(rows // 2 + 1) * STRIP_W]
        out.append((times[i] if i < len(times) else None, decode_strip(mid_row, STRIP_W)))
        i += 1
    proc.wait()
    return out


def unwrap(codes, first_n):
    """16-bit present numbers back to full ones, starting near `first_n`."""
    out, last = [], first_n
    for c in codes:
        if c is None:
            out.append(None)
            continue
        base = last - (last & 0xFFFF)
        best = min((base + c + k * 65536 for k in (-1, 0, 1)), key=lambda n: abs(n - last))
        out.append(best)
        last = best
    return out


def grab(video: Path, t0: float, t1: float, out: Path, width=480, tile=None) -> bool:
    """Frames of `video` with times in [t0, t1] (seconds, the file's own clock), tiled side by
    side into one image."""
    if not video.exists() or not tool("ffmpeg"):
        return False
    n = tile or 1
    vf = f"select='between(t\\,{t0 - EPS:.6f}\\,{t1 + EPS:.6f})',scale={width}:-2"
    if n > 1:
        vf += f",tile={n}x1:padding=4:color=0x202020"
    r = run([tool("ffmpeg"), "-v", "error", "-y", "-ss", f"{max(0.0, t0 - 1.0):.6f}", "-copyts", "-i", str(video),
             "-vf", vf, "-fps_mode", "passthrough", "-frames:v", "1", "-q:v", "4", str(out)])
    return r.returncode == 0 and out.exists()


# ---------------------------------------------------------------------------------------------
# Analysis


def analyse(s: Session, marks: bool = True) -> dict:
    """Everything the report shows, as plain data."""
    a = {"warnings": []}
    S = s.s

    # --- Display timing: the refresh period and which present reached the screen when.
    refresh_us = None
    d = sorted((r for r in s.display if r["sync_us"] is not None), key=lambda r: r["sync_refresh"])
    steps = [(b["sync_us"] - x["sync_us"]) / (b["sync_refresh"] - x["sync_refresh"])
             for x, b in zip(d, d[1:]) if b["sync_refresh"] > x["sync_refresh"]]
    if steps:
        refresh_us = statistics.median(steps)
    elif s.pairs:
        refresh_us = statistics.median(r["refresh_us"] for r in s.pairs if r["refresh_us"]) if any(r["refresh_us"] for r in s.pairs) else None
    if refresh_us is None:
        hz = statistics.median([r["refresh_hz"] for r in s.system if r["refresh_hz"]] or [60])
        refresh_us = 1e6 / hz
        a["warnings"].append("No DXGI frame statistics: on-screen timing is estimated from the display rate.")
    a["refresh_hz"] = 1e6 / refresh_us
    shown = {}
    for r in s.display:
        shown.setdefault(r["present_count"], r)
    vgrid = d[0]["sync_us"] if d else None

    # --- Presents.
    P = sorted(s.presents, key=lambda r: r["n"])
    planned = [p["due_us"] if p["due_us"] is not None else p["call_us"] for p in P]
    present_rows = []
    displayed = []
    for i, p in enumerate(P):
        disp = shown.get(p["swap_count"]) if p["swap_count"] is not None else None
        intended = None
        if i + 1 < len(P):
            intended = round((planned[i + 1] - planned[i]) / refresh_us)
        row = {
            "n": p["n"], "kind": p["kind"], "seq": p["seq"], "t": p["t"], "x": S(p["call_us"]),
            "late_ms": (p["call_us"] - p["due_us"]) / 1000 if p["due_us"] is not None else None,
            "phase_ms": ((p["call_us"] - vgrid) % refresh_us) / 1000 if vgrid is not None else None,
            "intended": intended,
            "sync_refresh": disp["sync_refresh"] if disp else None,
            "sync_us": disp["sync_us"] if disp else None,
            "to_screen": round((disp["sync_us"] - p["call_us"]) / refresh_us) if disp else None,
            "arrived_us": p["arrived_us"],
            "call_us": p["call_us"],
            "redraw": p["due_us"] is None and p["arrived_us"] is None and p["kind"] != 0,
        }
        present_rows.append(row)
        if disp:
            displayed.append(row)
    displayed.sort(key=lambda r: r["sync_refresh"])
    for x, b in zip(displayed, displayed[1:]):
        x["on_screen"] = b["sync_refresh"] - x["sync_refresh"]
    have_stats = len(displayed) > 0.5 * max(1, len(P))
    if P and not have_stats:
        a["warnings"].append(
            f"DXGI frame statistics covered only {len(displayed)} of {len(P)} presents: drops and on-screen "
            "times are incomplete (the output video's barcodes are used where they exist).")
    # Dropped: never shown although a later present was.
    last_shown = max((r["sync_refresh"] is not None and r["n"]) or 0 for r in present_rows) if present_rows else 0
    for r in present_rows:
        r["dropped"] = have_stats and r["sync_refresh"] is None and r["n"] < last_shown
    a["presents"] = present_rows

    # --- Output video: which present each recorded frame shows.
    a["output"] = None
    out_video = s.dir / "output.mkv"
    if marks and out_video.exists() and tool("ffmpeg") and P:
        raw = decode_marks(out_video)
        ns = unwrap([c for _, c in raw], P[0]["n"])
        read = sum(1 for n in ns if n is not None)
        a["output"] = {"frames": len(raw), "read": read, "times": [t for t, _ in raw], "n": ns}
        if raw and read < 0.9 * len(raw):
            a["warnings"].append(f"Only {read} of {len(raw)} output frames had a readable barcode.")
        # Presents never seen in the recording (a second opinion on drops).
        seen = Counter(n for n in ns if n is not None)
        if seen:
            lo, hi = min(seen), max(seen)
            for r in present_rows:
                r["in_video"] = seen.get(r["n"], 0) if lo <= r["n"] <= hi else None

    # --- Card frames.
    C = sorted(s.capture, key=lambda r: r["seq"])
    gaps = [b["t_us"] - x["t_us"] for x, b in zip(C, C[1:])]
    card_period = statistics.median(gaps) if gaps else None
    a["card_period_ms"] = card_period / 1000 if card_period else None
    a["capture_gaps"] = [
        {"x": S(b["t_us"]), "ms": (b["t_us"] - x["t_us"]) / 1000, "seq": b["seq"]}
        for x, b in zip(C, C[1:]) if card_period and b["t_us"] - x["t_us"] > 1.5 * card_period
    ]
    a["copy_ms"] = [r["copy_us"] / 1000 for r in C]
    by_seq = {r["seq"]: r for r in C}
    a["input_frames"] = sum(1 for r in C if r["input_frame"] is not None)

    # --- What each card frame was.
    a["frame_kinds"] = dict(Counter(r["kind"] for r in s.frames))

    # --- Pairs.
    gpu = {r["seq"]: r["gpu_us"] for r in s.pair_gpu}
    counts = {r["seq"]: r for r in s.pair_counts}
    pairs = []
    for p in s.pairs:
        c = counts.get(p["seq"])
        lost = c["lost"] / c["changed"] if c and c["changed"] else None
        cut = c["cost"] / c["detail"] if c and c["detail"] else None
        pairs.append({
            "seq": p["seq"], "x": S(p["t_us"]), "held": p["held"], "steps": p["steps"], "target": p["target"],
            "content_fps": p["content_fps"], "every_ms": p["every_us"] / 1000, "gpu_ms": gpu[p["seq"]] / 1000 if p["seq"] in gpu else None,
            "guess_ms": p["gpu_guess_us"] / 1000, "skipped": p["skipped"], "backlog": p["backlog_cleared"],
            "lost": lost, "cut": cut, "changed": c["changed"] if c else None,
        })
    a["pairs"] = pairs

    # --- Latency of real pictures: arrival -> present call -> screen.
    lat = []
    for r in present_rows:
        if r["kind"] == 1 and r["arrived_us"] is not None:
            lat.append({
                "x": r["x"],
                "present_ms": (r["call_us"] - r["arrived_us"]) / 1000,
                "screen_ms": (r["sync_us"] - r["arrived_us"]) / 1000 if r["sync_us"] is not None else None,
            })
    a["latency"] = lat

    # --- Rates over time (half-second bins).
    bin_s = 0.5
    nb = max(1, int(math.ceil(s.duration / bin_s)))

    def rate(xs):
        b = [0] * nb
        for x in xs:
            i = int(x / bin_s)
            if 0 <= i < nb:
                b[i] += 1
        return [v / bin_s for v in b]

    new_pictures = [S(r["t_us"]) for r in s.frames if r["kind"] not in ("repeat", "passthrough")]
    a["rates"] = {
        "x": [(i + 0.5) * bin_s for i in range(nb)],
        "card": rate(S(r["t_us"]) for r in C),
        "game": rate(new_pictures) if any(r["kind"] != "passthrough" for r in s.frames) else None,
        "presented": rate(r["x"] for r in present_rows if not r["redraw"]),
        "displayed": rate(S(r["sync_us"]) for r in displayed),
    }
    a["system"] = [{**r, "x": S(r["t_us"])} for r in s.system if r["t_us"] is not None]
    a["events"] = [{"x": S(r["t_us"]), "kind": r["kind"], "detail": str(r["detail"] or "")} for r in s.events]

    a["summary"] = summarise(s, a, displayed, present_rows)
    a["incidents"] = incidents(s, a, present_rows)
    return a


def summarise(s: Session, a: dict, displayed: list, present_rows: list) -> dict:
    hz = a["refresh_hz"]
    on = [r.get("on_screen") for r in displayed if r.get("on_screen") is not None and r.get("intended") is not None]
    mism = [r for r in displayed if r.get("on_screen") is not None and r.get("intended") is not None and r["on_screen"] != r["intended"]]
    longer = [r for r in mism if r["on_screen"] > r["intended"]]
    frame_ms = [r["on_screen"] * 1000 / hz for r in displayed if r.get("on_screen") is not None]
    to_screen = Counter(r["to_screen"] for r in displayed if r["to_screen"] is not None)
    sysm = a["system"]
    pairs = a["pairs"]
    gen = sum(1 for r in present_rows if r["kind"] == 2)
    total = sum(1 for r in present_rows if not r["redraw"])
    info = s.info
    return {
        "duration_s": s.duration,
        "display_hz": hz,
        "card_fps": 1000 / a["card_period_ms"] if a["card_period_ms"] else None,
        "game_fps": mean(p["content_fps"] for p in pairs if p["content_fps"]) if pairs else None,
        "presented_fps": total / s.duration if s.duration else None,
        "displayed_fps": len(displayed) / s.duration if s.duration and displayed else None,
        "generated_share": gen / total if total else None,
        "frames_on_time": 1 - len(mism) / len(on) if on else None,
        "frames_longer": len(longer),
        "frames_shorter": len(mism) - len(longer),
        "dropped": sum(1 for r in present_rows if r.get("dropped")),
        "frame_ms_p50": pct(frame_ms, 50),
        "frame_ms_p99": pct(frame_ms, 99),
        "frame_ms_max": max(frame_ms) if frame_ms else None,
        "to_screen_refreshes": dict(sorted(to_screen.items())),
        "late_presents": sum(1 for r in present_rows if r["late_ms"] is not None and r["late_ms"] > 1.5),
        "latency_screen_ms": mean(r["screen_ms"] for r in a["latency"]),
        "latency_screen_p95": pct([r["screen_ms"] for r in a["latency"]], 95),
        "latency_present_ms": mean(r["present_ms"] for r in a["latency"]),
        "gpu_pair_ms": mean(p["gpu_ms"] for p in pairs),
        "gpu_pair_p99": pct([p["gpu_ms"] for p in pairs], 99),
        "ui_fallback_pairs": sum(1 for p in pairs if p["lost"] is not None and p["lost"] > 0.55),
        "cut_pairs": sum(1 for p in pairs if p["cut"] is not None and p["cut"] > 0.9),
        "skipped_frames": sum(p["skipped"] for p in pairs),
        "backlogs": sum(1 for p in pairs if p["backlog"]),
        "capture_gaps": len(a["capture_gaps"]),
        "copy_ms": mean(a["copy_ms"]),
        "cpu_process": mean(r["cpu_process"] for r in sysm),
        "cpu_process_max": max((r["cpu_process"] for r in sysm), default=None),
        "cpu_system": mean(r["cpu_system"] for r in sysm),
        "gpu_3d_process": mean(r["gpu_3d_process"] for r in sysm),
        "gpu_3d_total": mean(r["gpu_3d_total"] for r in sysm),
        "gpu_3d_total_max": max((r["gpu_3d_total"] for r in sysm), default=None),
        "working_set_mb": max((r["working_set_mb"] for r in sysm), default=None),
        "private_mb": max((r["private_mb"] for r in sysm), default=None),
        "gpu_mem_mb": max((r["gpu_mem_process_mb"] for r in sysm), default=None),
        "bookmarks": sum(1 for e in a["events"] if e["kind"] == "bookmark"),
        "input_frames": a["input_frames"],
        "card_frames": len(s.capture),
        "target": mean(r["target"] for r in sysm),
        "mode": info.get("video", {}).get("mode"),
        "commit": info.get("app", {}).get("commit"),
    }


# Incident kinds: how much each one matters when ranking, and what it means.
KINDS = {
    "bookmark": (100, "Bookmarked while recording"),
    "signal_lost": (40, "No signal"),
    "dropped": (8, "A frame never reached the screen"),
    "capture_gap": (6, "The card's frames stopped for a moment"),
    "backlog": (6, "Fell a picture behind and dropped the queue"),
    "long": (4, "A frame stayed on screen longer than planned"),
    "short": (2, "A frame stayed on screen shorter than planned"),
    "skipped": (3, "Generated frames skipped to catch up"),
    "gpu_over": (3, "Frame generation took longer than a step"),
    "late": (2, "Presented late"),
    "ui_fallback": (1, "Whole-frame cross-fade (UI change)"),
    "cut": (1, "Scene cut"),
    "restart": (1, "Frame generation restarted"),
    "after_still": (0.5, "Shown at once after a still picture"),
}


def incidents(s: Session, a: dict, present_rows: list) -> list[dict]:
    raw = []  # (x, kind, detail, seq, n)
    for e in a["events"]:
        if e["kind"] in ("bookmark", "signal_lost"):
            raw.append((e["x"], e["kind"], e["detail"], None, None))
    for r in present_rows:
        if r.get("dropped"):
            raw.append((r["x"], "dropped", f"present {r['n']} ({'generated' if r['kind'] == 2 else 'real'})", r["seq"], r["n"]))
        if r.get("on_screen") is not None and r.get("intended") is not None and r["on_screen"] != r["intended"]:
            k = "long" if r["on_screen"] > r["intended"] else "short"
            raw.append((r["x"], k, f"present {r['n']}: {r['on_screen']} refreshes, planned {r['intended']}", r["seq"], r["n"]))
        if r["late_ms"] is not None and r["late_ms"] > 1.5:
            raw.append((r["x"], "late", f"present {r['n']} {r['late_ms']:.1f} ms after its time", r["seq"], r["n"]))
    for g in a["capture_gaps"]:
        raw.append((g["x"], "capture_gap", f"{g['ms']:.1f} ms without a card frame", g["seq"], None))
    for p in a["pairs"]:
        if p["backlog"]:
            raw.append((p["x"], "backlog", f"pair ending {p['seq']}", p["seq"], None))
        if p["skipped"]:
            raw.append((p["x"], "skipped", f"{p['skipped']} skipped in pair ending {p['seq']}", p["seq"], None))
        if p["gpu_ms"] is not None and p["gpu_ms"] > p["every_ms"]:
            raw.append((p["x"], "gpu_over", f"{p['gpu_ms']:.1f} ms of GPU for {p['every_ms']:.1f} ms steps", p["seq"], None))
        if p["lost"] is not None and p["lost"] > 0.55:
            raw.append((p["x"], "ui_fallback", f"{p['lost']:.0%} of the change unfollowable", p["seq"], None))
        if p["cut"] is not None and p["cut"] > 0.9:
            raw.append((p["x"], "cut", f"cost/detail {p['cut']:.2f}", p["seq"], None))
    for r in s.frames:
        if r["kind"] in ("restart", "after_still"):
            raw.append((s.s(r["t_us"]), r["kind"], f"card frame {r['seq']}", r["seq"], None))
    raw.sort(key=lambda r: r[0])

    # Group what happens within 0.3 s of each other, at most a second per incident (pacing
    # misses can come every few frames for minutes; they'd make one endless incident).
    groups = []
    for item in raw:
        if groups and item[0] - groups[-1]["end"] <= 0.3 and item[0] - groups[-1]["start"] < 1.0:
            g = groups[-1]
        else:
            g = {"start": item[0], "end": item[0], "items": []}
            groups.append(g)
        g["end"] = item[0]
        g["items"].append(item)
    out = []
    for g in groups:
        kinds = Counter(i[1] for i in g["items"])
        score = sum(KINDS[k][0] * min(c, 5) for k, c in kinds.items())
        seqs = [i[3] for i in g["items"] if i[3] is not None]
        ns = [i[4] for i in g["items"] if i[4] is not None]
        out.append({
            "start": g["start"], "end": g["end"], "score": score,
            "kinds": dict(kinds),
            "details": [f"{i[1]}: {i[2]}" for i in g["items"][:12]] + ([f"… {len(g['items']) - 12} more"] if len(g["items"]) > 12 else []),
            "seq": seqs[len(seqs) // 2] if seqs else None,
            "n": ns[len(ns) // 2] if ns else None,
        })
    return out


# ---------------------------------------------------------------------------------------------
# Thumbnails


def input_video_for(s: Session, seq: int):
    """(file, time in it) of a card frame in the input video."""
    r = next((c for c in s.capture if c["seq"] == seq), None)
    if not r or r["input_frame"] is None:
        return None
    video = s.dir / f"input_{r['input_segment']:03d}.mkv"
    fps = None
    for e in s.events:
        if e["kind"] == "input_segment" and str(e["detail"]).startswith(video.name):
            try:
                fps = float(str(e["detail"]).rsplit("@", 1)[1])
            except (IndexError, ValueError):
                pass
    if not fps:
        fps = 1000 / statistics.median(b["t_us"] / 1000 - a_["t_us"] / 1000 for a_, b in zip(s.capture, s.capture[1:]))
    return video, r["input_frame"] / fps, fps


def thumbnails(s: Session, a: dict, inc: list, limit: int) -> None:
    assets = s.dir / "report_assets"
    assets.mkdir(exist_ok=True)
    pairs_by_seq = {p["seq"]: p for p in a["pairs"]}
    frames_by_seq = sorted(r["seq"] for r in s.capture)
    out = a.get("output")
    present_by_n = {p["n"]: p for p in a["presents"]}
    chosen = sorted(inc, key=lambda i: -i["score"])[:limit]
    for k, i in enumerate(chosen):
        i["thumbs"] = []
        # Input: the pair's two pictures (or the card frame and the one before it).
        seq = i["seq"]
        if seq is not None:
            p = pairs_by_seq.get(seq)
            held = p["held"] if p else 1
            j = bisect.bisect_left(frames_by_seq, seq)
            a_seq = frames_by_seq[max(0, j - held)]
            src_a, src_b = input_video_for(s, a_seq), input_video_for(s, seq)
            if src_a and src_b and src_a[0] == src_b[0]:
                f = assets / f"inc{k:03d}_input.jpg"
                if grab(src_a[0], src_a[1], src_b[1], f, width=420, tile=held + 1):
                    i["thumbs"].append({"file": f"report_assets/{f.name}",
                                        "label": f"Input: card frames {a_seq}…{seq} (the pair's two pictures at the ends)"})
        # Output: what was on screen, via the barcodes.
        if out and out["frames"]:
            target = i["n"]
            if target is None:
                # The present nearest the incident.
                xs = [p["x"] for p in a["presents"]]
                j = bisect.bisect_left(xs, i["start"])
                if a["presents"]:
                    target = a["presents"][min(j, len(xs) - 1)]["n"]
            idx = [m for m, n in enumerate(out["n"]) if n is not None and target is not None and abs(n - target) <= 3]
            if idx:
                lo, hi = max(0, idx[0] - 1), min(out["frames"] - 1, idx[-1] + 1)
                hi = min(hi, lo + 7)
                t0, t1 = out["times"][lo], out["times"][hi]
                f = assets / f"inc{k:03d}_output.jpg"
                if t0 is not None and t1 is not None and grab(s.dir / "output.mkv", t0, t1, f, width=420, tile=hi - lo + 1):
                    labels = []
                    for m in range(lo, hi + 1):
                        n = out["n"][m]
                        pr = present_by_n.get(n)
                        tag = "?" if pr is None else ("R" if pr["kind"] == 1 else f"G{pr['t']:.2f}" if pr["kind"] == 2 else "B")
                        labels.append(f"{n if n is not None else '?'} {tag}")
                    i["thumbs"].append({"file": f"report_assets/{f.name}",
                                        "label": "Screen, frame by frame (present number, R real / G generated at t): " + " · ".join(labels)})


# ---------------------------------------------------------------------------------------------
# The report


def series(rows, key, xkey="x", digits=3):
    xs, ys = [], []
    for r in rows:
        y = r.get(key)
        if y is None or r.get(xkey) is None:
            continue
        xs.append(round(r[xkey], 4))
        ys.append(round(y, digits))
    return {"x": xs, "y": ys}


def build_charts(a: dict) -> list[dict]:
    hz = a["refresh_hz"]
    disp = [r for r in a["presents"] if r.get("on_screen") is not None]
    on_time = [r for r in disp if r.get("intended") is None or r["on_screen"] == r["intended"]]
    off = [r for r in disp if r.get("intended") is not None and r["on_screen"] != r["intended"]]
    ms = lambda rows: [{**r, "ms": r["on_screen"] * 1000 / hz, "planned": (r["intended"] or 0) * 1000 / hz} for r in rows]
    charts = [
        {"title": "Frame time on screen", "unit": "ms", "kind": "dots", "series": [
            {"label": "as planned", "color": "--c1", **series(ms(on_time), "ms")},
            {"label": "not as planned", "color": "--bad", **series(ms(off), "ms")},
            {"label": "planned", "color": "--muted", "line": True, **series(ms(disp), "planned")},
        ]},
        {"title": "Present phase after vblank (colour: refreshes until on screen)", "unit": "ms", "kind": "dots",
         "ymax": 1000 / hz, "series": [
            {"label": f"{k} refresh{'es' if k != 1 else ''}", "color": f"--c{(k % 5) + 1}",
             **series([r for r in a["presents"] if r.get("to_screen") == k], "phase_ms")}
            for k in sorted({r["to_screen"] for r in a["presents"] if r.get("to_screen") is not None})
        ]},
        {"title": "Latency of real pictures", "unit": "ms", "kind": "dots", "series": [
            {"label": "arrival → on screen", "color": "--c2", **series(a["latency"], "screen_ms")},
            {"label": "arrival → presented", "color": "--c3", **series(a["latency"], "present_ms")},
        ]},
        {"title": "Frame generation: GPU per pair and step", "unit": "ms", "kind": "dots", "series": [
            {"label": "GPU (motion + first frame)", "color": "--c4", **series(a["pairs"], "gpu_ms")},
            {"label": "step between frames", "color": "--muted", "line": True, **series(a["pairs"], "every_ms")},
        ]},
        {"title": "Frame generation: tests (above the line: whole-frame fallback)", "unit": "", "kind": "dots", "ymax": 1.05, "series": [
            {"label": "unfollowable share (UI test, fallback from 0.55)", "color": "--c5", **series(a["pairs"], "lost")},
            {"label": "cost/detail (cut test, cut at 0.9)", "color": "--c3", **series(a["pairs"], "cut")},
        ], "rules": [0.55, 0.9]},
        {"title": "Rates", "unit": "fps", "kind": "lines", "series": [
            {"label": name, "color": color, "x": a["rates"]["x"], "y": [round(v, 1) for v in a["rates"][key]]}
            for key, name, color in (("card", "card", "--c3"), ("game", "game pictures", "--c2"),
                                     ("presented", "presented", "--c4"), ("displayed", "on screen", "--c1"))
            if a["rates"][key] is not None
        ]},
        {"title": "CPU and GPU load", "unit": "%", "kind": "lines", "series": [
            {"label": "app CPU (whole machine)", "color": "--c1", **series(a["system"], "cpu_process", digits=1)},
            {"label": "system CPU", "color": "--muted", **series(a["system"], "cpu_system", digits=1)},
            {"label": "app GPU 3D", "color": "--c4", **series(a["system"], "gpu_3d_process", digits=1)},
            {"label": "GPU 3D (all)", "color": "--c2", **series(a["system"], "gpu_3d_total", digits=1)},
            {"label": "GPU video encode", "color": "--c5", **series(a["system"], "gpu_encode_total", digits=1)},
        ]},
        {"title": "Memory", "unit": "MB", "kind": "lines", "series": [
            {"label": "working set", "color": "--c1", **series(a["system"], "working_set_mb", digits=0)},
            {"label": "private (incl. driver)", "color": "--c3", **series(a["system"], "private_mb", digits=0)},
            {"label": "GPU memory (app)", "color": "--c4", **series(a["system"], "gpu_mem_process_mb", digits=0)},
        ]},
    ]
    for c in charts:
        c["series"] = [x for x in c["series"] if x["x"]]
    return [c for c in charts if c["series"]]


def card(label, value, hint="", tone=""):
    return f'<div class="card {tone}"><div class="v">{html.escape(value)}</div><div class="l">{html.escape(label)}</div>' \
           f'{f"<div class=h>{html.escape(hint)}</div>" if hint else ""}</div>'


def render(s: Session, a: dict) -> str:
    m = a["summary"]
    info = s.info
    machine = info.get("machine", {})
    gpus = ", ".join(g["name"] for g in machine.get("gpus", []))
    sess = info.get("session", {})
    title = f"Session {sess.get('started_local', s.dir.name)}"
    on_time = m["frames_on_time"]
    tone_time = "" if on_time is None else ("good" if on_time > 0.99 else "warn" if on_time > 0.95 else "bad")
    to_screen = m["to_screen_refreshes"]
    to_screen_txt = " / ".join(f"{k}:{v}" for k, v in to_screen.items()) if to_screen else "–"
    cards = "".join([
        card("Duration", fmt(m["duration_s"], 1, " s")),
        card("Mode", str(m["mode"] or "–"), f"card {fmt(m['card_fps'], 1)} fps"),
        card("Display", fmt(m["display_hz"], 1, " Hz")),
        card("Game rate", fmt(m["game_fps"], 1, " fps"), "detected new pictures"),
        card("Smoothing target", fmt(m["target"], 0, " fps")),
        card("On screen", fmt(m["displayed_fps"], 1, " fps"), f"presented {fmt(m['presented_fps'], 1)}"),
        card("Generated", fmt(None if m["generated_share"] is None else m["generated_share"] * 100, 0, " %")),
        card("Frames on time", fmt(None if on_time is None else on_time * 100, 2, " %"),
             f"{m['frames_longer']} longer, {m['frames_shorter']} shorter than planned", tone_time),
        card("Dropped", str(m["dropped"]), "never reached the screen", "bad" if m["dropped"] else "good"),
        card("Frame time p50 / p99", f"{fmt(m['frame_ms_p50'], 1)} / {fmt(m['frame_ms_p99'], 1)} ms", f"max {fmt(m['frame_ms_max'], 1)} ms"),
        card("Present → screen", to_screen_txt, "refreshes: count (one value = steady pacing)",
             "good" if len(to_screen) <= 1 else "warn"),
        card("Latency (real pictures)", fmt(m["latency_screen_ms"], 1, " ms"),
             f"p95 {fmt(m['latency_screen_p95'], 1)} ms · to present {fmt(m['latency_present_ms'], 1)} ms"),
        card("GPU per pair", fmt(m["gpu_pair_ms"], 2, " ms"), f"p99 {fmt(m['gpu_pair_p99'], 2)} ms"),
        card("UI fallbacks / cuts", f"{m['ui_fallback_pairs']} / {m['cut_pairs']}"),
        card("Skipped / backlogs", f"{m['skipped_frames']} / {m['backlogs']}", "", "warn" if m["backlogs"] else ""),
        card("Capture gaps", str(m["capture_gaps"]), f"copy {fmt(m['copy_ms'], 2)} ms a frame", "warn" if m["capture_gaps"] else ""),
        card("App CPU", fmt(m["cpu_process"], 1, " %"), f"max {fmt(m['cpu_process_max'], 1)} % of all cores"),
        card("GPU 3D", f"{fmt(m['gpu_3d_process'], 0)} % app", f"{fmt(m['gpu_3d_total'], 0)} % all, max {fmt(m['gpu_3d_total_max'], 0)} %"),
        card("Memory", fmt(m["working_set_mb"], 0, " MB"), f"private {fmt(m['private_mb'], 0)} MB · GPU {fmt(m['gpu_mem_mb'], 0)} MB"),
        card("Bookmarks", str(m["bookmarks"])),
    ])
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in a["warnings"])
    incs = sorted(a["incidents"], key=lambda i: (-i["score"], i["start"]))
    rows = []
    for i in incs[:200]:
        kinds = " ".join(f'<span class="tag t-{k}">{k}{f" ×{c}" if c > 1 else ""}</span>' for k, c in sorted(i["kinds"].items(), key=lambda kc: -KINDS[kc[0]][0]))
        thumbs = "".join(f'<figure><img loading="lazy" src="{html.escape(t["file"])}"><figcaption>{html.escape(t["label"])}</figcaption></figure>'
                         for t in i.get("thumbs", []))
        details = "<br>".join(html.escape(d) for d in i["details"])
        rows.append(f'<details class="inc" data-x="{i["start"]:.3f}" data-x1="{i["end"]:.3f}"><summary>'
                    f'<span class="t">{i["start"]:.2f} s</span>{kinds}<span class="score">{i["score"]:.0f}</span></summary>'
                    f'<div class="body"><div class="d">{details}</div>{thumbs}<button class="zoom">Show in timelines</button></div></details>')
    legend_kinds = "".join(f'<li><span class="tag t-{k}">{k}</span> {html.escape(v[1])}</li>' for k, v in KINDS.items())
    data = {
        "duration": s.duration,
        "charts": build_charts(a),
        "events": [e for e in a["events"] if e["kind"] in ("bookmark", "setting", "signal_lost", "frames", "input_segment")],
    }
    meta = [
        ("App", f"{info.get('app', {}).get('version', '?')} ({m['commit']}, {info.get('app', {}).get('profile', '?')})"),
        ("Machine", f"{machine.get('cpu', '')} · {machine.get('logical_cores', '?')} threads · {gpus} · {machine.get('os', '')}"),
        ("Video", f"{info.get('video', {}).get('status', '')} · scaling {info.get('video', {}).get('scaling', '?')}"),
        ("Recorded", f"input video: {'yes' if sess.get('input_video') else 'no'} ({m['input_frames']} of {m['card_frames']} card frames) · "
                     f"output video: {'yes' if sess.get('output_video') else 'no'}"
                     + (f" ({a['output']['read']} of {a['output']['frames']} frames identified)" if a.get("output") else "")
                     + f" · encoder {sess.get('encoder') or '–'}"),
        ("Folder", str(s.dir)),
    ]
    meta_html = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in meta)
    return TEMPLATE.replace("%TITLE%", html.escape(title)) \
        .replace("%CARDS%", cards) \
        .replace("%META%", meta_html) \
        .replace("%WARN%", f'<ul class="warn">{warn}</ul>' if warn else "") \
        .replace("%INCIDENTS%", "".join(rows) or "<p>No incidents.</p>") \
        .replace("%INCCOUNT%", str(len(incs))) \
        .replace("%LEGEND%", legend_kinds) \
        .replace("%DATA%", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>%TITLE%</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#1b1f24;--muted:#8a929c;--line:#e3e6ea;--c1:#2f7de1;--c2:#1aa37a;--c3:#d9822b;--c4:#8b5cf6;--c5:#d6409f;--bad:#e5484d;--good:#1aa37a;--warn:#d9822b}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--panel:#171a20;--text:#e6e8eb;--muted:#7d8590;--line:#2a2f37;--c1:#5aa2ff;--c2:#3ecf9b;--c3:#f0a35c;--c4:#a78bfa;--c5:#f472b6;--bad:#ff6b6f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1280px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums}.card .l{color:var(--muted);font-size:12px}.card .h{color:var(--muted);font-size:11px;margin-top:2px}
.card.good .v{color:var(--good)}.card.warn .v{color:var(--warn)}.card.bad .v{color:var(--bad)}
table.meta{border-collapse:collapse;margin-top:12px;font-size:12px}table.meta th{text-align:left;color:var(--muted);font-weight:500;padding:2px 12px 2px 0;vertical-align:top}
ul.warn{background:color-mix(in srgb,var(--warn) 14%,transparent);border-radius:8px;padding:8px 8px 8px 28px}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:8px;padding:6px 8px}
.chart .head{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;font-size:12px}.chart .head b{font-size:13px}
.chart .key{display:inline-flex;align-items:center;gap:5px;color:var(--muted)}.chart .key i{width:10px;height:10px;border-radius:2px;display:inline-block}
canvas{width:100%;height:150px;display:block;cursor:crosshair}
.hint{color:var(--muted);font-size:12px;margin:-4px 0 8px}
.inc{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:6px}
.inc summary{cursor:pointer;padding:8px 10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.inc .t{font-variant-numeric:tabular-nums;font-weight:600;min-width:70px}.inc .score{margin-left:auto;color:var(--muted);font-size:12px}
.inc .body{padding:0 10px 10px}.inc .d{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--muted);margin-bottom:8px}
.inc figure{margin:0 0 10px}.inc img{max-width:100%;border-radius:6px;border:1px solid var(--line)}.inc figcaption{font-size:12px;color:var(--muted)}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;background:var(--line)}
.t-bookmark{background:var(--bad);color:#fff}.t-dropped,.t-signal_lost{background:color-mix(in srgb,var(--bad) 30%,transparent)}
.t-long,.t-capture_gap,.t-backlog{background:color-mix(in srgb,var(--warn) 30%,transparent)}
button{font:inherit;font-size:12px;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--text);cursor:pointer}
.legend{columns:2 320px;font-size:12px;color:var(--muted);padding-left:0;list-style:none}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;display:none;z-index:9;white-space:pre}
.bar{display:flex;gap:8px;align-items:center;margin-bottom:8px}
</style></head><body><main>
<h1>%TITLE%</h1>
<table class="meta">%META%</table>
%WARN%
<h2>Summary</h2><div class="cards">%CARDS%</div>
<h2>Timelines</h2>
<div class="bar"><button id="reset">Whole session</button><span class="hint" id="range"></span></div>
<p class="hint">Drag across a chart to zoom, scroll to zoom around the pointer, double-click to see everything. Vertical lines: red bookmarks, grey setting changes.</p>
<div id="charts"></div>
<h2>Incidents (%INCCOUNT%)</h2>
<p class="hint">Things that went wrong or are worth a look, grouped when they happen within 0.3 s of each other (at most a second each), most important first. Thumbnails for the top ones.</p>
%INCIDENTS%
<h2>What the tags mean</h2><ul class="legend">%LEGEND%</ul>
</main><div id="tip"></div>
<script>
const D=%DATA%;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let view=[0,Math.max(D.duration,0.001)];
const charts=[];
function lower(a,v){let lo=0,hi=a.length;while(lo<hi){const m=(lo+hi)>>1;if(a[m]<v)lo=m+1;else hi=m}return lo}
function setView(a,b){const d=D.duration||1;a=Math.max(0,a);b=Math.min(d,b);if(b-a<0.002)return;view=[a,b];draw();
 document.getElementById('range').textContent=`${a.toFixed(3)} – ${b.toFixed(3)} s`}
function make(c){const box=document.createElement('div');box.className='chart';
 const head=document.createElement('div');head.className='head';head.innerHTML=`<b>${c.title}</b>`+c.series.map(s=>`<span class="key"><i style="background:${css(s.color)}"></i>${s.label}</span>`).join('');
 const cv=document.createElement('canvas');box.append(head,cv);document.getElementById('charts').append(box);
 const ch={c,cv};charts.push(ch);let drag=null;
 const xAt=e=>{const r=cv.getBoundingClientRect();return view[0]+(e.clientX-r.left)/r.width*(view[1]-view[0])};
 cv.addEventListener('mousedown',e=>{drag={x0:xAt(e),px:e.clientX}});
 window.addEventListener('mouseup',e=>{if(drag&&Math.abs(e.clientX-drag.px)>4){const x1=xAt(e);setView(Math.min(drag.x0,x1),Math.max(drag.x0,x1))}drag=null;draw()});
 cv.addEventListener('mousemove',e=>{ch.hover=xAt(e);ch.drag=drag?[drag.x0,ch.hover]:null;draw();tip(e,ch)});
 cv.addEventListener('mouseleave',()=>{ch.hover=null;document.getElementById('tip').style.display='none';draw()});
 cv.addEventListener('dblclick',()=>setView(0,D.duration));
 cv.addEventListener('wheel',e=>{e.preventDefault();const x=xAt(e),k=e.deltaY>0?1.25:0.8;setView(x-(x-view[0])*k,x+(view[1]-x)*k)},{passive:false})}
function tip(e,ch){const t=document.getElementById('tip');if(ch.hover==null){t.style.display='none';return}
 const lines=[`${ch.hover.toFixed(3)} s`];for(const s of ch.c.series){const i=Math.min(s.x.length-1,lower(s.x,ch.hover));
  const j=(i>0&&Math.abs(s.x[i-1]-ch.hover)<Math.abs(s.x[i]-ch.hover))?i-1:i;if(j>=0&&s.x.length)lines.push(`${s.label}: ${s.y[j]} ${ch.c.unit} (at ${s.x[j].toFixed(3)} s)`)}
 t.textContent=lines.join('\n');t.style.display='block';t.style.left=(e.clientX+14)+'px';t.style.top=(e.clientY+14)+'px'}
function draw(){for(const ch of charts)drawOne(ch)}
function drawOne({c,cv,hover,drag}){const dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
 const g=cv.getContext('2d');g.scale(dpr,dpr);const L=44,R=8,T=6,B=18,w=W-L-R,h=H-T-B;
 let ymax=c.ymax||0;if(!c.ymax){for(const s of c.series){const a=lower(s.x,view[0]),b=lower(s.x,view[1]);for(let i=a;i<b;i++)if(s.y[i]>ymax)ymax=s.y[i]}
  ymax=ymax>0?ymax*1.1:1}
 const X=x=>L+(x-view[0])/(view[1]-view[0])*w,Y=y=>T+h-(y/ymax)*h;
 g.strokeStyle=css('--line');g.fillStyle=css('--muted');g.font='11px system-ui';g.lineWidth=1;
 for(let k=0;k<=4;k++){const y=ymax*k/4,py=Y(y);g.beginPath();g.moveTo(L,py);g.lineTo(L+w,py);g.stroke();g.fillText(y<10?y.toFixed(2):y.toFixed(0),2,py+4)}
 const span=view[1]-view[0],step=[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,30,60,120,300,600].find(s=>span/s<=10)||600;
 for(let x=Math.ceil(view[0]/step)*step;x<=view[1];x+=step){const px=X(x);g.fillText((step<1?x.toFixed(3):x.toFixed(0))+'s',px-12,H-4)}
 for(const r of c.rules||[]){g.strokeStyle=css('--muted');g.setLineDash([4,4]);g.beginPath();g.moveTo(L,Y(r));g.lineTo(L+w,Y(r));g.stroke();g.setLineDash([])}
 for(const e of D.events){if(e.x<view[0]||e.x>view[1])continue;g.strokeStyle=e.kind==='bookmark'?css('--bad'):css('--muted');
  g.globalAlpha=e.kind==='bookmark'?0.9:0.35;g.beginPath();g.moveTo(X(e.x),T);g.lineTo(X(e.x),T+h);g.stroke();g.globalAlpha=1}
 g.save();g.beginPath();g.rect(L,T,w,h);g.clip();
 for(const s of c.series){const col=css(s.color);const a=Math.max(0,lower(s.x,view[0])-1),b=Math.min(s.x.length,lower(s.x,view[1])+1);
  if(c.kind==='lines'||s.line){g.strokeStyle=col;g.lineWidth=1.4;g.beginPath();let started=false;
   const stride=Math.max(1,Math.floor((b-a)/(w*2)));for(let i=a;i<b;i+=stride){const px=X(s.x[i]),py=Y(s.y[i]);if(!started){g.moveTo(px,py);started=true}else g.lineTo(px,py)}g.stroke()}
  else{g.fillStyle=col;const r=(b-a)>4000?1.2:2;for(let i=a;i<b;i++){g.fillRect(X(s.x[i])-r,Y(s.y[i])-r,r*2,r*2)}}}
 g.restore();
 if(drag){g.fillStyle=css('--c1');g.globalAlpha=0.15;g.fillRect(X(Math.min(...drag)),T,Math.abs(X(drag[1])-X(drag[0])),h);g.globalAlpha=1}
 if(hover!=null){g.strokeStyle=css('--text');g.globalAlpha=0.4;g.beginPath();g.moveTo(X(hover),T);g.lineTo(X(hover),T+h);g.stroke();g.globalAlpha=1}}
for(const c of D.charts)make(c);
document.getElementById('reset').onclick=()=>setView(0,D.duration);
for(const d of document.querySelectorAll('details.inc')){const z=()=>{const x=+d.dataset.x,x1=+d.dataset.x1;setView(x-0.5,x1+0.5);document.getElementById('charts').scrollIntoView({behavior:'smooth'})};
 d.querySelector('.zoom').onclick=z}
addEventListener('resize',draw);setView(0,D.duration);
</script></body></html>
"""


# ---------------------------------------------------------------------------------------------
# Commands


def cmd_report(args):
    path = Path(args.session) if args.session else newest_session()
    s = Session(path)
    print(f"analysing {path}")
    a = analyse(s, marks=not args.no_video)
    if not args.no_thumbs and tool("ffmpeg"):
        thumbnails(s, a, a["incidents"], args.thumbs)
    out = path / "report.html"
    out.write_text(render(s, a), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(a["summary"], indent=2), encoding="utf-8")
    m = a["summary"]
    print(f"  {m['duration_s']:.1f} s, {m['displayed_fps'] or 0:.1f} fps on screen, "
          f"{(m['frames_on_time'] or 0) * 100:.2f}% on time, {m['dropped']} dropped, "
          f"present->screen {m['to_screen_refreshes']}, {len(a['incidents'])} incidents")
    for w in a["warnings"]:
        print(f"  note: {w}")
    print(f"  wrote {out}")
    if args.open:
        webbrowser.open(out.as_uri())


COMPARE = [
    ("duration_s", "Duration (s)", 1), ("display_hz", "Display (Hz)", 1), ("target", "Smoothing target", 0),
    ("game_fps", "Game rate (fps)", 1), ("displayed_fps", "On screen (fps)", 1), ("frames_on_time", "Frames on time", 4),
    ("frames_longer", "Frames longer than planned", 0), ("dropped", "Dropped", 0), ("frame_ms_p99", "Frame time p99 (ms)", 2),
    ("latency_screen_ms", "Latency to screen (ms)", 2), ("latency_screen_p95", "Latency p95 (ms)", 2),
    ("gpu_pair_ms", "GPU per pair (ms)", 2), ("ui_fallback_pairs", "UI fallbacks", 0), ("skipped_frames", "Skipped frames", 0),
    ("capture_gaps", "Capture gaps", 0), ("cpu_process", "App CPU (%)", 2), ("gpu_3d_process", "App GPU 3D (%)", 1),
    ("working_set_mb", "Working set (MB)", 0), ("private_mb", "Private (MB)", 0), ("gpu_mem_mb", "GPU memory (MB)", 0),
]


def cmd_compare(args):
    sums = []
    for p in (args.a, args.b):
        s = Session(Path(p))
        sums.append((s, analyse(s, marks=False)["summary"]))
    (sa, a), (sb, b) = sums
    print(f"{'':32} {sa.dir.name:>24} {sb.dir.name:>24} {'change':>10}")
    for key, label, digits in COMPARE:
        va, vb = a.get(key), b.get(key)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = f"{vb - va:+.{digits}f}"
        print(f"{label:32} {fmt(va, digits):>24} {fmt(vb, digits):>24} {delta:>10}")
    print(f"{'Present -> screen (refreshes)':32} {str(a['to_screen_refreshes']):>24} {str(b['to_screen_refreshes']):>24}")


def cmd_extract(args):
    s = Session(Path(args.session))
    t = args.at
    pairs = sorted(s.pairs, key=lambda p: p["t_us"])
    if not pairs:
        sys.exit("no frame-generation pairs in this session")
    p = min(pairs, key=lambda p: abs(s.s(p["t_us"]) - t))
    seqs = sorted(r["seq"] for r in s.capture)
    j = bisect.bisect_left(seqs, p["seq"])
    a_seq = seqs[max(0, j - p["held"])]
    src_a, src_b = input_video_for(s, a_seq), input_video_for(s, p["seq"])
    if not src_a or not src_b or src_a[0] != src_b[0]:
        sys.exit("those card frames aren't in one input video segment")
    out = Path(args.out or (s.dir / f"pair_{p['seq']}.y4m"))
    fps = src_a[2]
    vf = f"select='between(t\\,{src_a[1] - EPS:.6f}\\,{src_b[1] + EPS:.6f})'"
    r = run([tool("ffmpeg"), "-v", "error", "-y", "-ss", f"{max(0.0, src_a[1] - 1.0):.6f}", "-copyts", "-i", str(src_a[0]),
             "-vf", vf, "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", str(out)])
    if r.returncode:
        sys.exit(r.stderr.decode(errors="replace"))
    print(f"pair ending at card frame {p['seq']} ({s.s(p['t_us']):.3f} s): frames {a_seq}..{p['seq']} -> {out}")
    print(f"  capture-card --gpu-eval \"{out}\" OUTDIR {p['held']} 0")


def cmd_list(args):
    root = sessions_root()
    for d in sorted(root.glob("*")):
        f = d / "session.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text(encoding="utf-8"))
        sess = info.get("session", {})
        dur = (sess.get("ended_us", 0) - sess.get("started_us", 0)) / 1e6 if sess.get("ended_us") else None
        report = "report" if (d / "report.html").exists() else ""
        print(f"{d.name}  {fmt(dur, 1, ' s'):>9}  {info.get('video', {}).get('mode', ''):<12} "
              f"smooth {info.get('settings', {}).get('smooth', '?'):<4} {report}")


def cmd_marks(args):
    for t, c in decode_marks(Path(args.video)):
        print(f"{t}\t{c}")


def cmd_selftest(args):
    exe = Path(args.exe or Path(__file__).resolve().parents[1] / "target" / "release" / "capture-card.exe").resolve()
    if not tool("ffmpeg"):
        sys.exit("ffmpeg not found")
    # An older build without --mark-test would start the app itself instead: check first.
    if not exe.exists() or b"marked frames of" not in exe.read_bytes():
        sys.exit(f"{exe} doesn't have --mark-test (build the current source first)")
    exe = str(exe)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = tmp / "bg.png"
        # A busy background right up to the barcode.
        run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720", "-frames:v", "1", str(img)], check=True)
        r = run([exe, "--mark-test", str(img), str(tmp / "m")], text=True)
        if r.returncode:
            sys.exit(f"{exe} --mark-test failed: {r.stderr}")
        want = [int(x) for x in (tmp / "m" / "marks.txt").read_text().split()]
        ok = True
        for enc in ("hevc_nvenc", "libx264"):
            video = tmp / f"m_{enc}.mkv"
            q = ["-rc", "constqp", "-qp", "16"] if enc.endswith("nvenc") else ["-crf", "16"]
            e = run([tool("ffmpeg"), "-v", "error", "-y", "-framerate", "60", "-i", str(tmp / "m" / "mark_%04d.png"),
                     "-pix_fmt", "yuv420p", "-c:v", enc, *q, str(video)])
            if e.returncode:
                print(f"  {enc}: not available, skipped")
                continue
            got = [c for _, c in decode_marks(video)]
            full = unwrap(got, want[0])
            match = sum(1 for g, w in zip(got, want) if g == w)
            print(f"  {enc}: {match}/{len(want)} barcodes read correctly")
            ok &= match == len(want) and len(got) == len(want)
            # Unwrapping across 16 bits: the run 65520..65559 counts on.
            tail = full[-40:]
            ok &= all(b - x == 1 for x, b in zip(tail, tail[1:]) if x is not None and b is not None)
        print("selftest " + ("passed" if ok else "FAILED"))
        sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="analyse a session into report.html")
    r.add_argument("session", nargs="?")
    r.add_argument("--open", action="store_true", help="open the report in the browser")
    r.add_argument("--no-thumbs", action="store_true", help="skip thumbnails (faster)")
    r.add_argument("--no-video", action="store_true", help="don't read the output video's barcodes")
    r.add_argument("--thumbs", type=int, default=30, help="incidents to make thumbnails for (default 30)")
    r.set_defaults(fn=cmd_report)
    c = sub.add_parser("compare", help="two sessions side by side")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)
    e = sub.add_parser("extract", help="a frame-generation pair as a .y4m for --gpu-eval")
    e.add_argument("session")
    e.add_argument("--at", type=float, required=True, help="seconds into the session")
    e.add_argument("--out")
    e.set_defaults(fn=cmd_extract)
    sub.add_parser("list", help="sessions recorded").set_defaults(fn=cmd_list)
    m = sub.add_parser("marks", help="decode an output video's barcodes")
    m.add_argument("video")
    m.set_defaults(fn=cmd_marks)
    t = sub.add_parser("selftest", help="check the barcode reader against the app's shader")
    t.add_argument("exe", nargs="?")
    t.set_defaults(fn=cmd_selftest)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
