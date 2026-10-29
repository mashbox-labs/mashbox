#!/usr/bin/env python3
"""Render the Track2Rig hero media from the quickstart clip, drawn from the rig itself: every tracked point is re-posed
by the predicted skin and bone transforms and coloured by its main bone; lines are the predicted skeleton.

Needs the package with the track2rig extra, matplotlib, pillow and the ffmpeg command:
    pip install ".[cpu,track2rig]" matplotlib pillow
    python tools/track2rig_preview.py --out-dir hf/track2rig/assets [--weights track2rig-v2.safetensors]

Writes hero.gif (one pass of the clip) and hero.mp4 (H.264, the clip looped to at least --min-seconds); with
--contact-sheet PATH also a PNG sheet of frames to check by eye (keep it out of the published assets). The caption on
the media names the model and version from models.json, the weights file, the device (CPU) and the input clip.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLIP = "examples/track2rig/tracks/grison_walk.npz"   # the clip the README quickstart runs


def palette(n):
    hue = (np.arange(n) * 0.61803398875) % 1.0
    return np.stack([np.abs(hue * 6 - 3) - 1, 2 - np.abs(hue * 6 - 2), 2 - np.abs(hue * 6 - 4)], 1).clip(0, 1) * 0.8 + 0.15


def project(P, elev, azim):
    """Orthographic camera, Z up: turn about Z by azim, tilt down by elev. Returns screen x, screen y, depth."""
    a, e = np.radians(azim), np.radians(elev)
    x = P[..., 0] * np.cos(a) - P[..., 1] * np.sin(a)
    d = P[..., 0] * np.sin(a) + P[..., 1] * np.cos(a)
    return x, P[..., 2] * np.cos(e) + d * np.sin(e), d * np.cos(e) - P[..., 2] * np.sin(e)


def render_frames(r, caption, out_dir, size=(1280, 840), views=((0, "side"), (20, "three-quarter")), quarter=25):
    """One PNG per clip frame: a side view and a three-quarter view stacked, same rig, Z up, orthographic."""
    Y = r.reconstruct()                                                         # [T,N,3] points re-posed by the rig
    J = np.einsum("tbij,bj->tbi", r.transforms[..., :3], r.joints) + r.transforms[..., 3]   # joints per frame
    col = palette(r.num_bones)[r.weights.argmax(1)]
    ext = np.ptp(Y[0], 0)
    base = -90.0 if ext[1] > ext[0] else 0.0                                     # body's long axis across the screen
    cams = [(elev, base + (quarter if elev else 0.0), label) for elev, label in views]
    lims = []
    for elev, azim, _ in cams:
        x, y, _ = project(np.concatenate([Y.reshape(-1, 3), J.reshape(-1, 3)]), elev, azim)
        px, py = 0.04 * np.ptp(x), 0.04 * np.ptp(x)
        lims.append((x.min() - px, x.max() + px, y.min() - py, y.max() + py))
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    gs = fig.add_gridspec(len(cams), 1, height_ratios=[(y1 - y0) / (x1 - x0) for x0, x1, y0, y1 in lims],
                          left=0.02, right=0.98, bottom=0.07, top=0.98, hspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(len(cams))]
    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, color="#333")
    paths = []
    for t in range(Y.shape[0]):
        for ax, (elev, azim, label), (x0, x1, y0, y1) in zip(axes, cams, lims):
            ax.cla()
            ax.set_aspect("equal", adjustable="box")
            px, py, pd = project(Y[t], elev, azim)
            o = np.argsort(-pd)                                                  # far points first
            ax.scatter(px[o], py[o], c=col[o], s=10, linewidths=0)
            jx, jy, _ = project(J[t], elev, azim)
            for b, pa in enumerate(r.parent):
                if pa >= 0:
                    ax.plot([jx[pa], jx[b]], [jy[pa], jy[b]], color="k", lw=1.3)
            ax.scatter(jx, jy, c="k", s=9, zorder=3)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_axis_off()
            ax.text(0.0, 1.0, f"{label} view", transform=ax.transAxes, fontsize=11, color="#333", va="top")
        axes[0].text(1.0, 1.0, f"frame {t + 1}/{Y.shape[0]}, {r.num_bones} bones", transform=axes[0].transAxes,
                     fontsize=11, color="#333", ha="right", va="top")
        p = Path(out_dir) / f"f{t:04d}.png"
        fig.savefig(p, dpi=100, facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def contact_sheet(paths, out, cols=4, rows=3, width=1600):
    pick = [paths[round(i * (len(paths) - 1) / (cols * rows - 1))] for i in range(cols * rows)]
    ims = [Image.open(p).convert("RGB") for p in pick]
    w = width // cols
    hgt = round(ims[0].height * w / ims[0].width)
    sheet = Image.new("RGB", (w * cols, hgt * rows), "white")
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, hgt), Image.LANCZOS), ((i % cols) * w, (i // cols) * hgt))
    sheet.save(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", default=CLIP, help=f"input clip (default {CLIP}, the quickstart clip)")
    ap.add_argument("--weights", help="local copy of the released weights (default: download at the pinned revision)")
    ap.add_argument("--out-dir", default="hf/track2rig/assets")
    ap.add_argument("--fps", type=float, default=30.0, help="clip frame rate, as in the GLB export (default 30)")
    ap.add_argument("--min-seconds", type=float, default=8.0, help="loop the clip in the MP4 to at least this long")
    ap.add_argument("--gif-width", type=int, default=800)
    ap.add_argument("--gif-fps", type=float, default=15.0, help="GIF frame rate; frames are dropped to keep real speed")
    ap.add_argument("--contact-sheet", metavar="PATH", help="also write a PNG sheet of 12 frames here")
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    from mashbox.models.track2rig import load
    reg = json.loads((ROOT / "models.json").read_text())
    m = next(x for x in reg["models"] if x["id"] == "track2rig")
    wname = next(f["name"] for f in m["files"] if f.get("kind") == "weights" and f.get("default"))
    r = load(a.weights).rig_file(a.tracks)
    T = r.transforms.shape[0]
    loops = max(1, int(np.ceil(a.min_seconds * a.fps / T)))
    base = (f"{m['name']} {m['version']} (weights {Path(wname).stem}), CPU. Input: {Path(a.tracks).name} "
            f"(the quickstart clip), {T} frames at {a.fps:g} fps.")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        gdir, vdir = Path(td, "gif"), Path(td, "mp4"); gdir.mkdir(); vdir.mkdir()
        render_frames(r, base, gdir)
        vcap = base if loops == 1 else f"{base} Looped {loops}x."
        vpaths = render_frames(r, vcap, vdir)
        step = max(1, round(a.fps / a.gif_fps))
        ffmpeg("-framerate", a.fps, "-i", gdir / "f%04d.png", "-vf",
               f"select='not(mod(n\\,{step}))',setpts=N/({a.fps / step:g}*TB),scale={a.gif_width}:-1:flags=lanczos,"
               "split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
               "-r", a.fps / step, "-loop", 0, out / "hero.gif")
        ffmpeg("-framerate", a.fps, "-i", vdir / "f%04d.png", "-vf", f"loop=loop={loops - 1}:size={T}:start=0",
               "-c:v", "libx264", "-preset", "slow", "-crf", 24, "-pix_fmt", "yuv420p", "-an",
               "-movflags", "+faststart", out / "hero.mp4")
        if a.contact_sheet:
            contact_sheet(vpaths, a.contact_sheet)
            print("wrote", a.contact_sheet)
    for f in ("hero.gif", "hero.mp4"):
        print("wrote", out / f, (out / f).stat().st_size, "bytes")
    print(f"clip {a.tracks}: {T} frames, {r.num_bones} bones, mp4 loops {loops}, gif every {step} frame(s)")


if __name__ == "__main__":
    main()
