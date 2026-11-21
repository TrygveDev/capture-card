#!/usr/bin/env python3
"""Builds the app icon from tools/icon.svg (48 px and up) and tools/icon-small.svg (16-32 px,
bolder for small sizes): renders every size with Microsoft Edge (headless, transparent), then
writes assets/icon.ico (each size as drawn, PNG-compressed), assets/icon.png (1024) and
assets/icon-256.png.

    python tools/make_icon.py
"""
import shutil, struct, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
LARGE, SMALL = (256, 128, 64, 48), (32, 24, 16)


def render(svg: Path, size: int, out: Path, tmp: Path):
    page = tmp / f"r{size}.html"
    page.write_text(f'<html><body style="margin:0;background:transparent"><img src="{svg.as_uri()}" '
                    f'width="{size}" height="{size}" style="display:block"></body></html>')
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    "--default-background-color=00000000", f"--window-size={size},{size}", f"--screenshot={out}",
                    page.as_uri()], check=True, capture_output=True)


def write_ico(pngs: list[tuple[int, bytes]], out: Path):
    head = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, data = b"", b""
    for size, png in pngs:
        s = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", s, s, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    out.write_bytes(head + entries + data)


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        pngs = []
        for size in LARGE + SMALL:
            svg = ROOT / "tools" / ("icon-small.svg" if size in SMALL else "icon.svg")
            out = tmp / f"{size}.png"
            render(svg, size, out, tmp)
            pngs.append((size, out.read_bytes()))
        write_ico(pngs, ROOT / "assets" / "icon.ico")
        render(ROOT / "tools" / "icon.svg", 1024, ROOT / "assets" / "icon.png", tmp)
        shutil.copy(tmp / "256.png", ROOT / "assets" / "icon-256.png")
    print("wrote assets/icon.ico, icon.png, icon-256.png")


if __name__ == "__main__":
    main()
