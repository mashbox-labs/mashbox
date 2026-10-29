"""Benchmark protocol: how a clip with a ground-truth rig is turned into noisy, partly hidden tracks, and how a rig is scored.

A clip = an animated mesh with its true skin and bone transforms. We sample 1,024 tracked points and 1,024 held-out
points by surface area, add Gaussian noise of 0.5% of the bounding-box diagonal to the tracked points, and hide each
tracked point for random runs of 5 to 30 frames until 20% of its frames are hidden (frame 0 always visible).
The main score is recon_heldout: the rig re-poses the held-out points (never tracked; each takes the skin of its nearest
tracked point at frame 0) and we measure the mean distance to their true positions, in % of the bbox diagonal.
"""
import hashlib
import json

import numpy as np

from .rig import lbs, skeleton_from_lbs

N_TRACK = 1024
N_EVAL = 1024
NOISE_FRAC = 0.005
OCC_FRAC = 0.20


def clip_seed(cid):
    return int(hashlib.md5(cid.encode()).hexdigest()[:8], 16)


def vertex_area(rest, faces):
    a, b, c = rest[faces[:, 0]], rest[faces[:, 1]], rest[faces[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    va = np.zeros(len(rest))
    np.add.at(va, faces.ravel(), np.repeat(ar / 3, 3))
    return va


def _observe(rng, clean, diag, noisy):
    T = clean.shape[0]
    X = clean.copy()
    mask = np.ones((T, clean.shape[1]), bool)
    if noisy:
        X = X + rng.normal(0, NOISE_FRAC * diag, X.shape)
        for i in range(clean.shape[1]):
            hidden = 0
            while hidden < OCC_FRAC * (T - 1):
                L = int(rng.integers(5, 31))
                s = int(rng.integers(1, max(2, T - L + 1)))
                mask[s:s + L, i] = False
                hidden = (~mask[1:, i]).sum()
        mask[0] = True
    X[~mask] = 0.0
    return X, mask


def make_clip(rest_v, faces, V, M, W_idx, W_val, bone_head, bone_parent, cid, noisy=True):
    """Full protocol from a mesh. rest_v [Nv,3], faces [F,3], V [T,Nv,3] animated vertices, M [T,B,4,4] rest->frame bone
    matrices, W_idx/W_val [Nv,4] true skin, bone_head [B,3], bone_parent [B]. cid = clip id string (seeds the noise)."""
    rng = np.random.default_rng(clip_seed(cid))
    V = V.astype(np.float64)
    rest = rest_v.astype(np.float64)
    va = vertex_area(rest, faces) + 1e-12
    va[va <= 1e-11] = 0
    p = va / va.sum()
    n_take = min(N_TRACK + N_EVAL, int((p > 0).sum()))
    sel = rng.choice(V.shape[1], n_take, replace=False, p=p)
    idx, eidx = sel[:N_TRACK], sel[N_TRACK:]
    diag = float(np.linalg.norm(np.ptp(V[0], 0)))
    clean = V[:, idx]
    X, mask = _observe(rng, clean, diag, noisy)
    return dict(X=X, mask=mask, clean=clean, eval_clean=V[:, eidx], eval_bind=V[0, eidx], diag=diag,
                gt_Wi=W_idx[idx], gt_Wv=W_val[idx], gt_M=M.astype(np.float64), gt_head=bone_head, gt_parent=bone_parent, T=V.shape[0])


def load_example(path, noisy=True):
    """Load a shipped example clip. It stores the sampled points and the random-generator state right after point
    sampling, so the noise and hidden runs come out exactly as the full protocol makes them."""
    with np.load(path, allow_pickle=False) as d:
        Vt, Ve = d["V_track"].astype(np.float64), d["V_eval"].astype(np.float64)
        diag = float(d["diag"])
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(d["rng_state"]))
        X, mask = _observe(rng, Vt, diag, noisy)
        return dict(X=X, mask=mask, clean=Vt, eval_clean=Ve, eval_bind=Ve[0], diag=diag, gt_Wi=d["gt_weight_index"],
                    gt_Wv=d["gt_weight_value"], gt_M=d["gt_transforms"].astype(np.float64), gt_head=d["gt_bone_head"],
                    gt_parent=d["gt_bone_parent"], T=Vt.shape[0])


def gt_joints_bind(c):
    """True joint positions at frame 0 = bone heads under the frame-0 pose."""
    h = np.c_[c["gt_head"], np.ones(len(c["gt_head"]))]
    return np.einsum("bij,bj->bi", c["gt_M"][0], h)[:, :3]


def gt_rig(c):
    """The true rig in our convention (bind = frame 0): Rt_t = M_t M_0^-1, dense W over tracked points."""
    M = c["gt_M"]
    B = M.shape[1]
    Rt = (M @ np.linalg.inv(M[0])[None])[:, :, :3, :]
    W = np.zeros((len(c["gt_Wi"]), B))
    np.add.at(W, (np.arange(len(c["gt_Wi"]))[:, None], c["gt_Wi"].astype(int)), c["gt_Wv"])
    return W, Rt, c["clean"][0]


def chamfer(A, Bp):
    if len(A) == 0 or len(Bp) == 0:
        return float("nan")
    D = np.linalg.norm(A[:, None] - Bp[None], axis=-1)
    return 0.5 * (D.min(1).mean() + D.min(0).mean())


def evaluate(c, W, Rt, P):
    """Score a rig (W [N,B] dense, Rt [T,B,3,4] bind->t, P [N,3] bind positions of the tracked points). All errors in %
    of the bbox diagonal. recon_tracked: tracked points vs their clean positions. recon_heldout: held-out points.
    j2j: symmetric Chamfer distance between the rig's joints and the true joints at frame 0."""
    d = c["diag"]
    e_tr = np.linalg.norm(lbs(W, Rt, P) - c["clean"], axis=-1).mean() / d * 100
    nn = np.argmin(np.linalg.norm(c["eval_bind"][:, None] - c["clean"][0][None], axis=-1), axis=1)
    e_ho = np.linalg.norm(lbs(W[nn], Rt, c["eval_bind"]) - c["eval_clean"], axis=-1).mean() / d * 100
    nb = int((W.sum(0) > 1e-3).sum())
    sp = float((W > 0.01).sum(1).mean())
    edges, J, _ = skeleton_from_lbs(W, Rt, P, d)
    joints = np.array([J[a, b] for a, b in edges]) if edges else np.zeros((0, 3))
    j2j = chamfer(joints, gt_joints_bind(c)) / d * 100
    return dict(recon_tracked=float(e_tr), recon_heldout=float(e_ho), bones=nb, sparsity=sp, j2j=float(j2j),
                gt_bones=int(len(c["gt_head"])))
