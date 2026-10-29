# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""The raw tracker file that `align` reads, and the tracks.npz that `align` writes.

tracks.npz (read by `mashbox run track2rig`, see mashbox.models.track2rig.tracks) holds exactly two arrays:
    tracks      float32 [T, N, 3]   3D position of every point in every frame, Z up, subject facing -Y
    visibility  bool    [T, N]      True where the tracker saw the point; frame 0 must be all True
Frame 0 is the bind pose. Hidden samples carry any finite value (we write 0).
Limits: 2 <= T <= 176 frames, 8 <= N <= 2048 points. Units do not matter: the model
normalises by the frame-0 bounding box.

raw tracker file (input to `align`, written by `track` or by any other 3D point tracker):
    coords  float [T, N, 3]  world positions (any right-handed frame)
    visib   float [T, N]     visibility as probability in [0, 1] or as logits (finite)
  optional, used to find the ground plane and the camera's up direction:
    w2c     float [T, 4, 4]  world-to-camera matrices, OpenCV convention (x right, y down, z forward)
    K       float [T, 3, 3]  intrinsics
    depth0  float [H, W]     frame-0 depth
    mask0   bool  [H, W]     frame-0 subject mask (True on the subject)
    q2d     float [N, 3]     query points as [frame, x_px, y_px]
No other arrays. Each array is at most 256 MiB once decompressed, and the file at most 512 MiB in all.
"""
import numpy as np


MAX_MEMBER_BYTES = 1 << 28  # 256 MiB per array: far above a 176-frame, 2,048-point clip or a 4K depth map
MAX_TOTAL_BYTES = 1 << 29   # 512 MiB for the whole file once decompressed
ARRAYS = ("coords", "visib", "w2c", "K", "depth0", "mask0", "q2d")
SHAPES = {"w2c": (None, 4, 4), "K": (None, 3, 3), "q2d": (None, 3)}


def load_raw(path):
    """Read a raw tracker file into a dict of arrays. Never unpickles. Before any array is read: only the arrays listed
    above are allowed, and each one's uncompressed size and the total are capped. Then coords and visib must be finite,
    and the optional arrays `align` uses are checked for shape and finite values."""
    with np.load(str(path), allow_pickle=False) as d:
        extra = sorted(set(d.files) - set(ARRAYS))
        if extra:
            raise ValueError(f"unexpected arrays in {path}: {extra} (allowed: {', '.join(ARRAYS)})")
        total = 0
        for zi in d.zip.infolist():
            if zi.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"{path}: {zi.filename} is {zi.file_size} bytes uncompressed; limit {MAX_MEMBER_BYTES}")
            total += zi.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"{path}: {total} bytes uncompressed in all; limit {MAX_TOTAL_BYTES}")
        raw = {k: np.asarray(d[k]) for k in d.files}
    for k in ("coords", "visib"):
        if k not in raw:
            raise ValueError(f"raw file has no '{k}' array")
    if raw["coords"].ndim != 3 or raw["coords"].shape[2] != 3:
        raise ValueError(f"coords shape {raw['coords'].shape}; need [T, N, 3]")
    if raw["visib"].shape != raw["coords"].shape[:2]:
        raise ValueError(f"visib shape {raw['visib'].shape}; need {raw['coords'].shape[:2]}")
    if not np.isfinite(raw["coords"]).all():
        raise ValueError("coords contain NaN or Inf")
    if not np.isfinite(raw["visib"]).all():
        raise ValueError("visib contains NaN or Inf")
    T = raw["coords"].shape[0]
    for k, want in SHAPES.items():
        if k in raw:
            a = raw[k]
            if a.ndim != len(want) or any(w is not None and w != g for w, g in zip(want, a.shape)):
                raise ValueError(f"{k} shape {a.shape}; need {['T' if w is None else w for w in want]}")
            if not np.isfinite(a).all():
                raise ValueError(f"{k} contains NaN or Inf")
    if "w2c" in raw and raw["w2c"].shape[0] < 1:
        raise ValueError("w2c is empty")
    if "w2c" in raw and raw["w2c"].shape[0] not in (1, T):
        raise ValueError(f"w2c has {raw['w2c'].shape[0]} frames; need 1 or {T}")
    if "depth0" in raw and raw["depth0"].ndim != 2:
        raise ValueError(f"depth0 shape {raw['depth0'].shape}; need [H, W]")
    if "mask0" in raw and raw["mask0"].ndim != 2:
        raise ValueError(f"mask0 shape {raw['mask0'].shape}; need [H, W]")
    if "K" in raw and raw["K"].shape[0] < 1:
        raise ValueError("K is empty")
    return raw
