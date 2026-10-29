# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""Raw 3D point tracks (any tracker) -> the frame Track2Rig was trained in.

Track2Rig was trained on creatures in a Z-up world, facing -Y, moving in place. Its input
features use raw positions and displacement from frame 0, so the up axis, the facing and any
whole-body travel all matter. `align` puts tracker output into that frame:

1. Up axis. Fit a plane (RANSAC, 3000 tries, seed 0) to frame-0 background pixels (depth0
   unprojected, subject mask excluded) whose normal is within 35 degrees of camera-0 up.
   With more than 10% inliers, the plane normal becomes +Z and the floor z = 0. Otherwise
   camera-0 up is used and the floor is the 1st percentile of frame-0 heights.
2. Outliers. Drop points whose frame-0 position is more than 4 MADs from the median
   (depth bleeding at the silhouette).
3. Travel and heading. Per-frame median horizontal displacement of visible points,
   median-filtered over 5 frames. If the net travel is more than 5% of the body diagonal,
   its direction is the heading; otherwise the frame-0 long axis is used and its sign is a
   guess (logged; use flip_heading). Rotate about Z so the heading points to -Y, then
   subtract the horizontal travel (unless keep_travel).
4. Visibility. Probability (logits are passed through a sigmoid) above vis_thr. Frame 0 is
   forced visible; hidden samples are set to 0. At most 176 frames are kept.
"""
import math

import numpy as np

from .tracksfile import TMAX


def _rot_to(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    s = np.linalg.norm(w)
    c = u @ v
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # opposite vectors: turn 180 degrees about any axis at right angles to u (a rotation, det +1; -I is a reflection)
        a = np.eye(3)[np.argmin(np.abs(u))]
        k = np.cross(u, a)
        k = k / np.linalg.norm(k)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = w / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * Kx @ Kx


def _median_filter_time(tr, size=5):
    """Same as scipy.ndimage.median_filter(tr, size=(size, 1), mode='nearest') for a [T, 2] array."""
    from scipy.ndimage import median_filter
    return median_filter(tr, size=(size, 1), mode="nearest")


def _ground_up(raw, C0, rng, log):
    """Returns (unit up vector n, a point on the floor plane or None)."""
    if "w2c" in raw:
        c2w0 = np.linalg.inv(raw["w2c"][0])
    else:
        c2w0 = np.eye(4)
        log["camera"] = "none given: world = camera-0 frame (OpenCV, y down)"
    up_cam = -c2w0[:3, 1]
    up_cam = up_cam / np.linalg.norm(up_cam)
    if not all(k in raw for k in ("K", "depth0", "mask0")):
        log["up"] = dict(src="camera-0 up (no depth0/K/mask0 for a ground fit)")
        return up_cam, None
    K = raw["K"][0]
    D = raw["depth0"]
    H, W = D.shape
    m0 = raw["mask0"].astype(bool)
    if m0.shape != D.shape:
        yy = (np.arange(H) * m0.shape[0] / H).astype(int)
        xx = (np.arange(W) * m0.shape[1] / W).astype(int)
        m0 = m0[yy][:, xx]
    ys, xs = np.nonzero((~m0) & (D > 0) & np.isfinite(D))
    if len(ys) < 3:
        log["up"] = dict(src="camera-0 up (fewer than 3 background pixels with depth for a ground fit)")
        return up_cam, None
    pick = rng.choice(len(ys), min(40000, len(ys)), replace=False)
    ys, xs = ys[pick], xs[pick]
    z = D[ys, xs]
    pc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    pw = pc @ c2w0[:3, :3].T + c2w0[:3, 3]
    best = (0, None)
    scale = np.linalg.norm(np.ptp(C0, 0))
    for _ in range(3000):
        s = pw[rng.choice(len(pw), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ up_cam < 0:
            n = -n
        if n @ up_cam < np.cos(np.radians(35)):
            continue
        inl = np.abs((pw - s[0]) @ n) < 0.01 * scale
        if inl.sum() > best[0]:
            best = (int(inl.sum()), (n, s[0], inl))
    if best[1] is not None and best[0] > 0.1 * len(pw):
        n, _, inl = best[1]
        P = pw[inl]
        cc = P.mean(0)
        n = np.linalg.svd(P - cc, full_matrices=False)[2][-1]
        n = n if n @ up_cam > 0 else -n
        log["up"] = dict(src="ground RANSAC", inlier_frac=round(best[0] / len(pw), 3),
                         angle_to_cam_up_deg=round(float(np.degrees(np.arccos(np.clip(n @ up_cam, -1, 1)))), 1))
        return n, cc
    log["up"] = dict(src="camera-0 up (no ground plane found)")
    return up_cam, None


def align(raw, vis_thr=0.5, keep_travel=False, flip_heading=False):
    """raw: dict with coords [T,N,3], visib [T,N] (+ optional w2c, K, depth0, mask0).
    Returns (tracks float32 [T,N',3], visibility bool [T,N'], info dict). N' <= N after outlier removal."""
    vis_thr = float(vis_thr)
    if not (math.isfinite(vis_thr) and 0.0 <= vis_thr <= 1.0):
        raise ValueError(f"vis_thr must be a finite number from 0 to 1, got {vis_thr}")
    log = {}
    C = np.asarray(raw["coords"], dtype=np.float64)
    V = np.asarray(raw["visib"], dtype=np.float64)
    if not np.isfinite(C).all():
        raise ValueError("coords contain NaN or Inf")
    if not np.isfinite(V).all():
        raise ValueError("visib contains NaN or Inf")
    if V.min() < 0 or V.max() > 1:
        V = 1 / (1 + np.exp(-V))
        log["visib"] = "logits->sigmoid"
    T = min(len(C), TMAX)
    C, V = C[:T], V[:T]
    M = V > vis_thr
    M[0] = True
    rng = np.random.default_rng(0)

    # 1. up axis
    n, cc = _ground_up(raw, C[0], rng, log)
    R1 = _rot_to(n, np.array([0, 0, 1.0]))
    X = C @ R1.T
    floor = (cc @ R1.T)[2] if cc is not None else np.percentile(X[0][:, 2], 1)
    X[..., 2] -= floor

    # 2. outliers at frame 0
    med = np.median(X[0], 0)
    dist = np.linalg.norm(X[0] - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-9
    keep = dist < np.median(dist) + 4 * 1.4826 * mad
    X, M = X[:, keep], M[:, keep]
    log["outliers_dropped"] = int((~keep).sum())

    # 3. travel + heading
    disp = X[:, :, :2] - X[0][None, :, :2]
    tr = np.array([np.median(disp[t][M[t]], 0) if M[t].sum() > 10 else np.zeros(2) for t in range(T)])
    tr = _median_filter_time(tr, 5)
    diag = float(np.linalg.norm(np.ptp(X[0], 0)))
    net = tr[-1] - tr[0]
    travel = float(np.linalg.norm(net))
    log["travel_frac_diag"] = round(travel / diag, 3)
    if travel > 0.05 * diag:
        h = net / travel
        log["heading"] = "travel direction"
    else:
        xy = X[0][:, :2] - X[0][:, :2].mean(0)
        h = np.linalg.svd(xy, full_matrices=False)[2][0]
        log["heading"] = "PCA long axis (sign ambiguous)"
    if flip_heading:
        h = -h
        log["heading"] += " FLIPPED"
    ang = np.arctan2(h[1], h[0])
    tgt = -np.pi / 2
    c_, s_ = np.cos(tgt - ang), np.sin(tgt - ang)
    R2 = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
    X = X @ R2.T
    tr = tr @ R2[:2, :2].T
    if not keep_travel:
        X[..., :2] -= tr[:, None, :]
        log["travel"] = "removed (in-place)"
    else:
        log["travel"] = "kept"

    # 4. visibility
    ext = np.ptp(X[0], 0)
    log["frame0_extent_xyz"] = ext.round(3).tolist()
    log["long_axis_is_Y"] = bool(ext[1] >= ext[0])
    log["vis_frac"] = round(float(M[1:].mean()), 3) if T > 1 else 1.0
    log["N"] = int(X.shape[1])
    log["T"] = int(T)
    X[~M] = 0.0
    info = dict(align=log, vis_thr=vis_thr, keep_travel=bool(keep_travel), flip_heading=bool(flip_heading),
                R_up=R1.tolist(), R_heading=R2.tolist(), floor=float(floor), travel_xy=tr.tolist(),
                kept_point_index=np.nonzero(keep)[0].tolist())
    return X.astype(np.float32), M, info
