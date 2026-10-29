# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""Write and check tracks.npz with the same rules `mashbox run track2rig` applies (mashbox.models.track2rig.tracks).

This package runs in the tracker's Python environment, which cannot also hold track2rig's
pinned versions, so the checks are repeated here. tests/test_video_tracks.py compares them
with mashbox.models.track2rig.tracks when both are installed."""
import numpy as np

TMAX = 176            # frames Track2Rig sees at once
NMAX = 2048           # points per clip
TMIN, NMIN = 2, 8
COORD_MAX = 1e6
CUT_JUMP = 0.5        # median per-frame jump of co-visible points, as a fraction of the frame-0 bbox diagonal


def check_tracks(X, mask=None):
    """Validate arrays and return (X float32 [T,N,3], mask bool [T,N]). Raises ValueError with a plain message."""
    X = np.asarray(X)
    if X.dtype.kind != "f":
        raise ValueError(f"tracks dtype {X.dtype}; need floating point")
    X = X.astype(np.float32)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"tracks shape {X.shape}; need [T, N, 3]")
    T, N = X.shape[:2]
    if T > TMAX:
        raise ValueError(f"tracks has T={T} frames; max {TMAX} (use --max-frames or --frame-step)")
    if N > NMAX:
        raise ValueError(f"tracks has N={N} points; max {NMAX} (use --points)")
    if T < TMIN or N < NMIN:
        raise ValueError(f"tracks shape {X.shape}; need T >= {TMIN} and N >= {NMIN}")
    mask = np.ones((T, N), bool) if mask is None else np.asarray(mask)
    if mask.dtype != bool or mask.shape != (T, N):
        raise ValueError(f"visibility {mask.dtype} {mask.shape}; need bool {(T, N)}")
    if not np.isfinite(X).all():
        raise ValueError("tracks contain NaN or Inf (put any finite value at hidden samples; it is ignored)")
    if not mask[0].all():
        raise ValueError("frame 0 is the bind pose and must be fully visible")
    X = np.where(mask[..., None], X, np.float32(0.0))  # hidden samples are ignored: written as 0, as the rig package does
    if np.abs(X).max() > COORD_MAX:  # after the line above, so only visible samples are range-checked
        raise ValueError(f"coordinates exceed {COORD_MAX:g} in magnitude")
    diag = float(np.linalg.norm(np.ptp(X[0].astype(np.float64), 0)))
    if not diag > 1e-6 * max(1.0, float(np.abs(X[0]).max())):
        raise ValueError("frame 0 point cloud is degenerate (zero extent)")
    co = mask[1:] & mask[:-1]
    jump = np.linalg.norm(X[1:].astype(np.float64) - X[:-1], axis=-1)
    for t in range(T - 1):
        if co[t].sum() >= NMIN and np.median(jump[t][co[t]]) > CUT_JUMP * diag:
            raise ValueError(f"frame {t}->{t + 1}: median point jump {np.median(jump[t][co[t]]) / diag:.2f} of the bbox "
                             "diagonal (camera cut or a different subject? try --max-frames to stop before it)")
    return X, mask


def save_tracks(path, X, mask=None):
    X, mask = check_tracks(X, mask)
    np.savez_compressed(str(path), tracks=X, visibility=mask)


MAX_MEMBER_BYTES = TMAX * NMAX * 3 * 8 + 4096  # largest allowed array (float64 tracks) plus the .npy header


def check_npz_sizes(d, path, limit=MAX_MEMBER_BYTES):
    """Refuse an .npz whose members would decompress past `limit` bytes, before any array is read."""
    for zi in d.zip.infolist():
        if zi.file_size > limit:
            raise ValueError(f"{path}: {zi.filename} is {zi.file_size} bytes uncompressed; the most a tracks file needs is {limit}")


def load_tracks(path):
    """Read a tracks .npz (never unpickles) and validate it. Member sizes are checked before any array is read."""
    with np.load(str(path), allow_pickle=False) as d:
        check_npz_sizes(d, path)
        extra = set(d.files) - {"tracks", "visibility"}
        if extra:
            raise ValueError(f"unexpected arrays in {path}: {sorted(extra)} (allowed: tracks, visibility)")
        if "tracks" not in d.files:
            raise ValueError(f"{path} has no 'tracks' array")
        X = d["tracks"]
        mask = d["visibility"] if "visibility" in d.files else None
    return check_tracks(X, mask)
