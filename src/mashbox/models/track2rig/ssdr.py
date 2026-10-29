"""SSDR baseline: Smooth Skinning Decomposition with Rigid Bones (Le and Deng, ACM TOG 31(6), SIGGRAPH Asia 2012).
Our own implementation of the published method, masked for hidden track samples. Bind pose = frame 0.
Model: x_i(t) = sum_j w_ij (R_j(t) p_i + T_j(t)), w >= 0, sum 1, at most 4 influences per point.
Used as the comparison method at the same bone count as Track2Rig. It optimises each clip from scratch: 30 iterations,
about 90 to 140 s per clip of 1,024 points at 32 bones on an AMD EPYC 7502P CPU."""
import numpy as np
from scipy.optimize import nnls

KMAX = 4


def fill_interp(X, mask):
    """linear interpolation over hidden frames, for initialisation only."""
    T, N, _ = X.shape; Y = X.copy(); t = np.arange(T)
    for i in range(N):
        v = mask[:, i]
        if v.all(): continue
        for k in range(3): Y[~v, i, k] = np.interp(t[~v], t[v], X[v, i, k])
    return Y


def wkabsch(P, Q, a):
    """batched weighted rigid fit: P [N,3] source, Q [T,N,3] targets, a [T,N] weights -> R [T,3,3], t [T,3] (Q ~ R P + t)."""
    s = a.sum(1, keepdims=True) + 1e-12
    pc = (a @ P) / s                          # [T,3]
    qc = np.einsum("tn,tni->ti", a, Q) / s
    Pc = P[None] - pc[:, None]; Qc = Q - qc[:, None]
    H = np.einsum("tn,tni,tnj->tij", a, Pc, Qc)
    U, S, Vt = np.linalg.svd(H)
    D = np.sign(np.linalg.det(np.einsum("tji,tkj->tik", Vt, U)))
    Dm = np.tile(np.eye(3), (len(D), 1, 1)); Dm[:, 2, 2] = D
    R = np.einsum("tji,tjk,tlk->til", Vt, Dm, U)  # V D U^T
    tt = qc - np.einsum("tij,tj->ti", R, pc)
    return R, tt


def kmeans(F, B, rng, iters=30):
    C = F[rng.choice(len(F), B, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((F[:, None] - C[None]) ** 2).sum(-1), 1)
        for j in range(B):
            m = lab == j
            C[j] = F[m].mean(0) if m.any() else F[rng.integers(len(F))]
    return lab


def bone_preds(Rt, P):
    """Rt [T,B,3,4], P [N,3] -> [T,B,N,3]"""
    return np.einsum("tbij,nj->tbni", Rt[..., :3], P) + Rt[:, :, None, :, 3]


def solve_weights(Rt, P, X, mask, kmax=KMAX):
    T, B = Rt.shape[:2]; N = len(P)
    Y = bone_preds(Rt, P)                                     # [T,B,N,3]
    m = mask.astype(float)
    err = (((Y - X[:, None]) ** 2).sum(-1) * m[:, None]).sum(0)  # [B,N]
    W = np.zeros((N, B))
    big = 1e3 * np.sqrt(m.sum(0).mean())
    Yn = np.ascontiguousarray(Y.transpose(2, 0, 1, 3))      # [N,T,B,3]
    cands = np.argsort(err, axis=0)[:kmax].T                  # [N,k]
    for i in range(N):
        cand = cands[i]
        vt = mask[:, i]
        A = Yn[i][vt][:, cand].transpose(0, 2, 1).reshape(-1, len(cand))   # [Tv*3,k]
        b = X[vt, i].reshape(-1)
        A2 = np.vstack([A, big * np.ones((1, len(cand)))]); b2 = np.r_[b, big]
        w, _ = nnls(A2, b2)
        s = w.sum(); W[i, cand] = w / s if s > 0 else np.eye(len(cand))[0]
    return W


def solve_transforms(W, Rt, P, X, mask):
    T, B = Rt.shape[:2]
    Y = bone_preds(Rt, P)
    full = np.einsum("nb,tbni->tni", W, Y)
    m = mask.astype(float)
    for j in range(B):
        wj = W[:, j]
        use = wj > 1e-4
        if use.sum() < 3: continue
        others = full[:, use] - wj[use][None, :, None] * Y[:, j][:, use]
        Q = (X[:, use] - others) / wj[use][None, :, None]
        a = (wj[use] ** 2)[None] * m[:, use]
        R, t = wkabsch(P[use], Q, a)
        newY = np.einsum("tij,nj->tni", R, P) + t[:, None]
        full = full + wj[None, :, None] * (newY - Y[:, j])
        Y[:, j] = newY
        Rt[:, j, :, :3] = R; Rt[:, j, :, 3] = t
    return Rt


def ssdr(X, mask, B, iters=30, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    T, N, _ = X.shape
    P = X[0].copy()  # bind frame = frame 0
    Xf = fill_interp(X, mask)
    F = (Xf - Xf[0:1]).transpose(1, 0, 2).reshape(N, -1)
    lab = kmeans(F, B, rng)
    Rt = np.zeros((T, B, 3, 4))
    for j in range(B):
        mj = lab == j
        if mj.sum() < 3:
            Rt[:, j, :, :3] = np.eye(3); continue
        R, t = wkabsch(P[mj], Xf[:, mj], mask[:, mj].astype(float))
        Rt[:, j, :, :3] = R; Rt[:, j, :, 3] = t
    W = np.eye(B)[lab]
    hist = []
    for it in range(iters):
        W = solve_weights(Rt, P, X, mask)
        # re-seed dead bones on the worst-fit points
        dead = np.where(W.sum(0) < 1e-3)[0]
        if len(dead):
            rec = np.einsum("nb,tbni->tni", W, bone_preds(Rt, P))
            e = (np.linalg.norm(rec - X, axis=-1) * mask).sum(0)
            worst = np.argsort(-e)
            for k, j in enumerate(dead):
                grp = worst[k * 8:(k + 1) * 8]
                R, t = wkabsch(P[grp], X[:, grp], mask[:, grp].astype(float))
                Rt[:, j, :, :3] = R; Rt[:, j, :, 3] = t
        Rt = solve_transforms(W, Rt, P, X, mask)
        if verbose or it == iters - 1:
            rec = np.einsum("nb,tbni->tni", W, bone_preds(Rt, P))
            hist.append(float((np.linalg.norm(rec - X, axis=-1) * mask).sum() / mask.sum()))
    return W, Rt, P, hist
