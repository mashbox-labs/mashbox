"""Track2Rig network: dense 3D point tracks in, per-point skin weights and per-frame rigid bone transforms out.

One feed-forward pass. A transformer encodes one token per tracked point plus K learned bone-slot tokens; skin
weights are a softmax over point-to-slot similarity; each bone's transform per frame is the weighted rigid fit
(Kabsch) of its points from the bind frame (frame 0) to that frame. All model math runs in float32 under JAX.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

# Model configuration (fixed by the released weights; checked against the weights file at load).
TMAX = 176   # frames the model sees at once (shorter clips are zero-padded with a mask)
NB = 32      # time bins in the per-point motion feature
K = 32       # bone slots
D = 192      # width
H = 6        # attention heads
L = 6        # transformer layers
EPS = 1e-6
TOPK = 4     # skin influences kept per point
CONFIG = dict(TMAX=TMAX, NB=NB, K=K, D=D, H=H, L=L)


def featurize(X, mask, T):
    """X [TMAX,N,3] normalised positions, mask [TMAX,N], T valid frames -> [N, 3 + NB*4] per-point features:
    frame-0 position, mean visible displacement from frame 0 in NB equal time bins, visible fraction per bin."""
    Tm = X.shape[0]
    t = jnp.arange(Tm)
    b = jnp.clip((t * NB) // jnp.maximum(T, 1), 0, NB - 1)
    onehot = jax.nn.one_hot(b, NB) * (t < T)[:, None]
    m = mask.astype(jnp.float32)
    disp = (X - X[0:1]) * m[..., None]
    cnt = jnp.einsum("tb,tn->bn", onehot, m)
    tot = jnp.einsum("tb,t->b", onehot, jnp.ones(Tm))[:, None]
    mean = jnp.einsum("tb,tnc->bnc", onehot, disp) / jnp.maximum(cnt, 1)[..., None]
    vis = cnt / jnp.maximum(tot, 1)
    f = jnp.concatenate([mean.transpose(1, 0, 2).reshape(X.shape[1], -1) * 10.0, vis.T], -1)
    return jnp.concatenate([X[0] * 2.0, f], -1)


def _ln(x, p):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(v + 1e-5) * p["g"] + p["b"]


def encoder(p, F):
    """F [N,fin] -> point embeddings [N,D], bone-slot embeddings [K,D]."""
    x = jnp.concatenate([F @ p["inp"]["w"] + p["inp"]["b"], p["slots"]], 0)
    n = x.shape[0]
    for lp in p["layers"]:
        h = _ln(x, lp["ln1"])
        q, k, v = jnp.split(h @ lp["qkv"], 3, -1)
        q = q.reshape(n, H, D // H)
        k = k.reshape(n, H, D // H)
        v = v.reshape(n, H, D // H)
        a = jax.nn.softmax(jnp.einsum("nhd,mhd->hnm", q, k) / math.sqrt(D // H), -1)
        x = x + jnp.einsum("hnm,mhd->nhd", a, v).reshape(n, D) @ lp["o"]
        h = _ln(x, lp["ln2"])
        x = x + jax.nn.gelu(h @ lp["m1"]) @ lp["m2"]
    x = _ln(x, p["lnf"])
    return x[:-K], x[-K:]


def skin(p, F):
    h, s = encoder(p, F)
    logits = (h @ p["q"]) @ (s @ p["k"]).T / math.sqrt(D)
    return jax.nn.softmax(logits, -1)


def kabsch(P, X, A):
    """Weighted rigid fit per bone and frame. P [N,3] bind, X [T,N,3], A [T,N,K] weights -> R [T,K,3,3], t [T,K,3]."""
    A = A + EPS
    s = A.sum(1)
    pc = jnp.einsum("tnk,nc->tkc", A, P) / s[..., None]
    xc = jnp.einsum("tnk,tnc->tkc", A, X) / s[..., None]
    Hm = jnp.einsum("tnk,nc,tnd->tkcd", A, P, X) - s[..., None, None] * pc[..., :, None] * xc[..., None, :]
    Hm = Hm + 1e-6 * jnp.eye(3)
    U, S, Vt = jnp.linalg.svd(Hm)
    det = jnp.linalg.det(jnp.einsum("tkji,tklj->tkil", Vt, U))
    Dm = jnp.concatenate([jnp.ones(det.shape + (2,)), det[..., None]], -1)
    R = jnp.einsum("tkji,tkj,tklj->tkil", Vt, Dm, U)
    t = xc - jnp.einsum("tkij,tkj->tki", R, pc)
    return R, t


def forward(p, X, mask, T, topk=None):
    F = featurize(X, mask, T)
    W = skin(p, F)
    if topk:
        thr = jax.lax.top_k(W, topk)[0][:, -1:]
        W = jnp.where(W >= thr, W, 0.0)
        W = W / W.sum(-1, keepdims=True)
    A = W[None] * mask.astype(jnp.float32)[..., None]
    R, t = kabsch(X[0], X, A)
    return W, R, t


_forward_jit = jax.jit(forward, static_argnums=(4,))


def predict(params, X, mask, topk=TOPK):
    """Run the model on numpy arrays.

    X    [T,N,3] point positions in any unit (T <= TMAX). Frame 0 is the bind pose.
    mask [T,N] bool, True where the point is visible. Values at hidden samples are ignored (set to 0 here).
    Returns W [N,K] skin weights (top-`topk` per point, rows sum to 1), Rt [T,K,3,4] bind->frame transforms in the
    input's units (x_t = R x_bind + t), and P [N,3] bind positions (= X[0])."""
    mask = np.asarray(mask, bool)
    X = np.asarray(X)
    X = np.where(mask[..., None], X, 0).astype(X.dtype)  # values at hidden samples never reach the output
    T = X.shape[0]
    if T > TMAX:
        raise ValueError(f"clip has {T} frames; the model takes at most {TMAX} (resample or split the clip)")
    c0 = X[0].mean(0)
    s = np.linalg.norm(np.ptp(X[0], 0))
    Xn = np.zeros((TMAX, X.shape[1], 3), np.float32)
    Xn[:T] = (X - c0) / s
    m = np.zeros((TMAX, X.shape[1]), bool)
    m[:T] = mask
    # Always on the host CPU, also when a TPU or GPU build of JAX is installed: the released numbers are float32 CPU
    # numbers, and an accelerator's default matmul precision changes the rig. Every JAX op of the model runs here.
    with jax.default_device(jax.devices("cpu")[0]):
        W, R, t = _forward_jit(params, jnp.asarray(Xn), jnp.asarray(m), T, topk)
    W, R, t = np.asarray(W, np.float64), np.asarray(R, np.float64)[:T], np.asarray(t, np.float64)[:T]
    # undo the normalisation: x = s*xn + c0, xn' = R xn + t  ->  x' = R x + (s t + c0 - R c0)
    tt = s * t + c0 - np.einsum("tkij,j->tki", R, c0)
    return W, np.concatenate([R, tt[..., None]], -1), X[0].copy()
