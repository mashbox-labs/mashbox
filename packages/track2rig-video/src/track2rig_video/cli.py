# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""track2rig-video VIDEO -o tracks.npz        video -> tracks.npz (needs TrackEverything + a CUDA GPU)
track2rig-video align RAW.npz -o tracks.npz   raw tracks from any 3D tracker -> tracks.npz (CPU)
track2rig-video check tracks.npz              check a tracks.npz with the rules `mashbox run track2rig` applies (CPU)"""
import argparse
import json
import math
import sys
import time

import numpy as np


def _unit(s):
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number from 0 to 1, got {s}") from None
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"must be a finite number from 0 to 1, got {s}")
    return v


def _align_args(p):
    p.add_argument("--vis-thr", type=_unit, default=0.5, help="visibility probability above which a point counts as seen (0 to 1, default 0.5)")
    p.add_argument("--keep-travel", action="store_true", help="keep whole-body travel (default: remove it; Track2Rig was trained on in-place motion)")
    p.add_argument("--flip-heading", action="store_true", help="turn the subject 180 degrees about the up axis (use when it ends up facing backwards)")


def _do_align(raw, a, out, extra_info=None):
    from .tracksfile import save_tracks
    from .align import align
    t0 = time.time()
    X, M, info = align(raw, vis_thr=a.vis_thr, keep_travel=a.keep_travel, flip_heading=a.flip_heading)
    info["align_s"] = round(time.time() - t0, 2)
    if extra_info:
        info.update(extra_info)
    save_tracks(out, X, M)
    side = (out[:-4] if out.endswith(".npz") else out) + ".json"
    with open(side, "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    print(f"wrote {out}: T={X.shape[0]} N={X.shape[1]} visible={M[1:].mean():.3f} " + json.dumps(info["align"]))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "align":
        p = argparse.ArgumentParser(prog="track2rig-video align",
                                    description="Raw 3D point tracks (any tracker) -> tracks.npz for Track2Rig. Runs on CPU.")
        p.add_argument("raw", help="raw tracker .npz: coords [T,N,3], visib [T,N]; optional w2c, K, depth0, mask0 (see README.md)")
        p.add_argument("-o", "--out", required=True)
        _align_args(p)
        a = p.parse_args(argv[1:])
        from .raw import load_raw
        _do_align(load_raw(a.raw), a, a.out, dict(source=a.raw.rsplit("/", 1)[-1]))
        return 0
    if argv and argv[0] == "check":
        p = argparse.ArgumentParser(prog="track2rig-video check")
        p.add_argument("tracks")
        a = p.parse_args(argv[1:])
        from .tracksfile import load_tracks
        X, M = load_tracks(a.tracks)
        print(f"OK {a.tracks}: T={X.shape[0]} N={X.shape[1]} visible={M.mean():.3f}")
        return 0
    if argv and argv[0] == "track":
        argv = argv[1:]
    p = argparse.ArgumentParser(prog="track2rig-video",
                                description="Video -> 3D point tracks of the subject -> tracks.npz for Track2Rig. "
                                            "Uses TrackEverything (CC BY-NC 4.0, non-commercial) on a CUDA GPU.",
                                epilog="Other commands (CPU only): `track2rig-video align RAW.npz -o tracks.npz` puts raw tracks "
                                       "from any 3D tracker into Track2Rig's frame; `track2rig-video check tracks.npz` "
                                       "applies Track2Rig's input checks.")
    p.add_argument("video")
    p.add_argument("-o", "--out", required=True, help="tracks.npz to write (a .json with the settings and timings is written next to it)")
    p.add_argument("--points", type=int, default=1024, help="points to track on the subject (default 1024, max 2048; Track2Rig was trained on 1024)")
    p.add_argument("--max-frames", type=int, default=176, help="frames to use from the start (default and max 176)")
    p.add_argument("--frame-step", type=int, default=1, help="use every k-th frame (default 1)")
    p.add_argument("--mask", default=None, help="frame-0 subject mask PNG (white = subject); default: rembg")
    p.add_argument("--raw-out", default=None, help="also save the raw tracker output (.npz) for `align` re-runs")
    p.add_argument("--device", default="cuda")
    _align_args(p)
    a = p.parse_args(argv)
    if not 8 <= a.points <= 2048:
        p.error("--points must be between 8 and 2048")
    if a.frame_step < 1:
        p.error("--frame-step must be 1 or more")
    if not 2 <= a.max_frames <= 176:
        p.error("--max-frames must be between 2 and 176")
    from .trackeverything import track
    t0 = time.time()
    raw, info = track(a.video, n_points=a.points, max_frames=a.max_frames, frame_step=a.frame_step, mask_path=a.mask, device=a.device)
    info["track_total_s"] = round(time.time() - t0, 2)
    info["source"] = a.video.rsplit("/", 1)[-1]
    if a.raw_out:
        np.savez_compressed(a.raw_out, **raw)
    _do_align(raw, a, a.out, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
