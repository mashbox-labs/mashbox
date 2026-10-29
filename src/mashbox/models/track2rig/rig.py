"""From model output to a rig: bones, skeleton tree with joint positions, top-4 skin, per-frame transforms, quality."""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .model import K, TOPK, predict

# Flag thresholds are heuristics, not calibrated on a measured set. They ship in rig.json so callers can see them.
THRESHOLDS = dict(depth_squash_extent_ratio_lt=0.10, low_motion_nonrigid_pct_lt=1.0, heavy_occlusion_visibility_lt=0.5,
                  few_bones_le=4, high_recon_error_pct_gt=3.0, short_clip_frames_lt=16)


def wkabsch(P, X, A):
    """Weighted rigid fit in numpy (same math as model.kabsch). P [N,3], X [T,N,3], A [T,N,B] -> Rt [T,B,3,4]."""
    A = A + 1e-6
    s = A.sum(1)
    pc = np.einsum("tnk,nc->tkc", A, P) / s[..., None]
    xc = np.einsum("tnk,tnc->tkc", A, X) / s[..., None]
    Hm = np.einsum("tnk,nc,tnd->tkcd", A, P, X, optimize=True) - s[..., None, None] * pc[..., :, None] * xc[..., None, :]
    Hm = Hm + 1e-6 * np.eye(3)
    U, S, Vt = np.linalg.svd(Hm)
    det = np.linalg.det(np.einsum("tkji,tklj->tkil", Vt, U))
    Dm = np.concatenate([np.ones(det.shape + (2,)), det[..., None]], -1)
    R = np.einsum("tkji,tkj,tklj->tkil", Vt, Dm, U)
    t = xc - np.einsum("tkij,tkj->tki", R, pc)
    return np.concatenate([R, t[..., None]], -1)


def lbs(W, Rt, P):
    """Linear blend skinning. W [N,B], Rt [T,B,3,4], P [N,3] -> [T,N,3]."""
    T, B = Rt.shape[:2]
    M = np.matmul(W, Rt.reshape(T, B, 12)).reshape(T, -1, 3, 4)
    return np.matmul(M[..., :3], P[None, :, :, None])[..., 0] + M[..., 3]


def skeleton_from_lbs(W, Rt, P, diag, lam=1e-3):
    """Joint between bones a and b = least-squares point that both bones move the same way over the clip (with a small
    pull to their shared points), after Le and Deng (2014); tree = minimum spanning tree over the joint residual.
    Returns edges [(a, b)], J [B,B,3] joint positions (bind frame), bone centroids [B,3]."""
    B = W.shape[1]
    alive = np.where(W.sum(0) > 1e-3)[0]
    cen = (W.T @ P) / np.maximum(W.sum(0)[:, None], 1e-9)
    T = Rt.shape[0]
    cost = np.full((B, B), np.inf)
    J = np.zeros((B, B, 3))
    for ii, a in enumerate(alive):
        for b in alive[ii + 1:]:
            A = (Rt[:, a, :, :3] - Rt[:, b, :, :3]).reshape(-1, 3)
            y = -(Rt[:, a, :, 3] - Rt[:, b, :, 3]).reshape(-1)
            share = W[:, a] * W[:, b]
            c0 = (share @ P) / share.sum() if share.sum() > 1e-6 else 0.5 * (cen[a] + cen[b])
            lt = lam * T
            p = np.linalg.solve(A.T @ A + lt * np.eye(3), A.T @ y + lt * c0)
            r = np.sqrt(np.mean((A @ p - y) ** 2)) / diag
            cost[a, b] = cost[b, a] = r + 0.1 * np.linalg.norm(cen[a] - cen[b]) / diag
            J[a, b] = J[b, a] = p
    edges = []
    if len(alive) > 1:
        intree = {alive[0]}
        while len(intree) < len(alive):
            best = None
            for a in intree:
                for b in alive:
                    if b in intree:
                        continue
                    if best is None or cost[a, b] < best[0]:
                        best = (cost[a, b], a, b)
            edges.append((best[1], best[2]))
            intree.add(best[2])
    return edges, J, cen


def skin_top4(W):
    """Dense [N,B] -> (index [N,4] int, weight [N,4]); unused slots are index 0 with weight 0."""
    B = W.shape[1]
    idx = np.argsort(-W, 1, kind="stable")[:, :TOPK]
    val = np.take_along_axis(W, idx, 1)
    if B < TOPK:
        idx = np.pad(idx, ((0, 0), (0, TOPK - B)))
        val = np.pad(val, ((0, 0), (0, TOPK - B)))
    idx = np.where(val > 0, idx, 0)
    return idx.astype(np.int64), val


@dataclass
class Rig:
    weights: np.ndarray        # [N,B] skin weights per tracked point (at most 4 non-zero per row, rows sum to 1)
    transforms: np.ndarray     # [T,B,3,4] bind (frame 0) -> frame t, x_t = R x_bind + t, input units
    bind_points: np.ndarray    # [N,3] tracked points at frame 0
    parent: np.ndarray         # [B] parent bone, -1 at the root
    joints: np.ndarray         # [B,3] bone head positions at frame 0 (joint with the parent; centroid for the root)
    centroids: np.ndarray      # [B,3] skin-weighted centroid of each bone's points at frame 0
    root: int
    model_slot: np.ndarray     # [B] which of the model's 32 bone slots each bone came from
    quality: dict
    flags: list
    notes: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    @property
    def num_bones(self):
        return int(self.weights.shape[1])

    def skin_top4(self):
        return skin_top4(self.weights)

    def reconstruct(self):
        """Tracked points re-posed by the rig, [T,N,3]."""
        return lbs(self.weights, self.transforms, self.bind_points)

    def to_json(self, fps=30.0):
        from .export import rig_json
        return rig_json(self, fps)

    def to_glb(self, fps=30.0):
        from .export import rig_glb
        return rig_glb(self, fps)

    def save(self, out_dir, fps=30.0, name="rig"):
        """Write <name>.json and <name>.glb into out_dir; returns the two paths."""
        import os
        if not name or name in (".", "..") or "/" in name or "\\" in name or os.sep in name:
            raise ValueError(f"name must be a file name without a folder, got {name!r}")
        from .export import check_fps
        check_fps(fps)
        os.makedirs(out_dir, exist_ok=True)
        pj, pg = os.path.join(out_dir, name + ".json"), os.path.join(out_dir, name + ".glb")
        with open(pj, "wb") as f:
            f.write(self.to_json(fps))
        with open(pg, "wb") as f:
            f.write(self.to_glb(fps))
        return pj, pg


def solve(params, X, mask, max_bones=K):
    """Run the model and build the rig. X [T,N,3], mask [T,N] bool (validated by tracks.check_tracks)."""
    if not (isinstance(max_bones, (int, np.integer)) and 1 <= max_bones <= K):
        raise ValueError(f"max_bones must be an integer in [1, {K}]")
    st = {}
    t0 = time.perf_counter()
    W, Rt, P = predict(params, X, mask)
    st["model"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    T, N = mask.shape
    notes = []
    mass = W.sum(0)
    alive = np.where(mass > 1e-3)[0]
    slots_model = len(alive)
    capped = False
    if len(alive) > max_bones:
        capped = True
        keep = np.sort(alive[np.argsort(-mass[alive], kind="stable")[:max_bones]])
        drop = np.setdiff1d(np.arange(K), keep)
        Wk = W.copy()
        Wk[:, drop] = 0.0
        orphan = np.where(Wk.sum(1) <= 0)[0]
        if len(orphan):   # points whose top-4 bones were all dropped -> the kept bone that best explains their motion
            Rkeep = Rt[:, keep]
            Y = np.einsum("tbij,nj->tbni", Rkeep[..., :3], P[orphan]) + Rkeep[:, :, None, :, 3]
            e = (np.linalg.norm(Y - X[:, None, orphan], axis=-1) * mask[:, None, orphan]).sum(0)
            Wk[orphan, keep[np.argmin(e, 0)]] = 1.0
        Wk /= Wk.sum(1, keepdims=True)
        A = Wk[None][..., keep] * mask[..., None]
        Rk = wkabsch(P, X.astype(np.float64), A)
        W, Rt, bone_slot = Wk[:, keep], Rk, keep
        notes.append(f"max_bones={max_bones}: kept the {len(keep)} heaviest of {slots_model} model bones, "
                     f"{len(orphan)} points reassigned, transforms re-solved (weighted Kabsch)")
    else:
        W, Rt, bone_slot = W[:, alive], Rt[:, alive], alive
    B = W.shape[1]
    diag = float(np.linalg.norm(np.ptp(P, 0)))
    edges, J, cen = skeleton_from_lbs(W, Rt, P, diag)
    root = int(np.argmax(W.sum(0)))
    adj = {b: [] for b in range(B)}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = -np.ones(B, int)
    order = [root]
    seen = {root}
    i = 0
    while i < len(order):
        a = order[i]
        i += 1
        for b in sorted(adj[a]):
            if b not in seen:
                parent[b] = a
                seen.add(b)
                order.append(b)
    for b in range(B):   # guard: every bone gets a parent even if the tree came out split
        if b not in seen:
            parent[b] = root
            order.append(b)
            seen.add(b)
    joints = np.array([cen[b] if parent[b] < 0 else J[parent[b], b] for b in range(B)])
    depth = np.zeros(B, int)
    for b in order[1:]:
        depth[b] = depth[parent[b]] + 1
    leaves = int(sum(1 for b in range(B) if b != root and not (parent == b).any()))
    st["skeleton"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    Xd = X.astype(np.float64)
    rec = lbs(W, Rt, P)
    err = np.linalg.norm(rec - Xd, axis=-1)[mask] / diag * 100
    disp = np.linalg.norm(Xd - Xd[0:1], axis=-1)[mask] / diag * 100
    R1 = wkabsch(P, Xd, mask[..., None].astype(np.float64))[:, 0]
    res = np.linalg.norm(np.einsum("tij,nj->tni", R1[..., :3], P) + R1[:, None, :, 3] - Xd, axis=-1)
    nonrigid = float((res * mask)[1:].sum() / max(mask[1:].sum(), 1) / diag * 100)
    ev = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov(P.T)), 0, None))
    extent = float(ev[0] / ev[-1]) if ev[-1] > 0 else 0.0
    vis = mask.mean(0)
    q = dict(recon_err_pct_diag=float(err.mean()), recon_err_p95_pct_diag=float(np.percentile(err, 95)),
             recon_err_basis="rig re-posing error vs the input tracks at visible samples (includes the input's own noise), "
                             "% of the frame-0 bounding-box diagonal",
             bones=int(B), bone_slots_model=int(slots_model), sparsity_mean_weights_gt_0p01=float((W > 0.01).sum(1).mean()),
             visibility_coverage=float(mask.mean()), points_visible_lt_50pct=float((vis < 0.5).mean()),
             motion_energy_pct_diag=float(disp.mean()), nonrigid_motion_pct_diag=nonrigid,
             depth_squash_extent_ratio=extent, skeleton_leaves=leaves, skeleton_max_depth=int(depth.max()),
             frames=int(T), points=int(N), bbox_diag=diag)
    th = THRESHOLDS
    flags = []
    if extent < th["depth_squash_extent_ratio_lt"]:
        flags.append("depth_squashed")
    if nonrigid < th["low_motion_nonrigid_pct_lt"]:
        flags.append("low_motion")
    if q["visibility_coverage"] < th["heavy_occlusion_visibility_lt"]:
        flags.append("heavy_occlusion")
    if B <= th["few_bones_le"]:
        flags.append("few_bones")
    if q["recon_err_pct_diag"] > th["high_recon_error_pct_gt"]:
        flags.append("high_recon_error")
    if T < th["short_clip_frames_lt"]:
        flags.append("short_clip")
    if capped:
        flags.append("max_bones_capped")
    st["quality"] = (time.perf_counter() - t0) * 1e3
    return Rig(weights=W, transforms=Rt, bind_points=P, parent=parent, joints=joints, centroids=cen, root=root,
               model_slot=np.asarray(bone_slot), quality=q, flags=flags, notes=notes, timings_ms=st)


def decimals_for(diag):
    """Decimal places for positions so rounding stays near 1e-6 of the creature size."""
    return int(np.clip(6 - math.floor(math.log10(diag)), 0, 12))
