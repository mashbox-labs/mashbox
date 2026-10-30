"""Trilinear sampling of a sparse voxel attribute field in JAX (our own code; replaces flex_gemm grid_sample_3d,
mode='trilinear', as called by o_voxel.postprocess.to_glb).

Semantics (from the FlexGEMM op at the pinned commit, LICENSE-FlexGEMM in the package folder above; CUDA path the official to_glb uses):
  query q in voxel units (q = (p - aabb_min) / voxel_size); voxel (i,j,k) has its centre at (i+0.5, j+0.5, k+0.5).
  corners = floor(q - 0.5) + {0,1}^3; a corner counts only if it is inside [0, W) x [0, H) x [0, D) AND present in the
  sparse set; weight = prod(1 - |q - corner - 0.5|); weights of present corners are renormalised by max(sum, 1e-12);
  no present corner -> output 0.
Lookup: voxel keys x*H*D + y*D + z are sorted once; each corner is found by searchsorted (no hash map), which is a
fixed-shape gather that maps to TPU without a custom kernel.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], np.int32)


def build_index(coords, grid_size):
    """coords [N,3] int -> (sorted keys int32/int64, permutation)."""
    W, H, D = (int(g) for g in grid_size)
    c = np.asarray(coords, np.int64)
    key = (c[:, 0] * H + c[:, 1]) * D + c[:, 2]
    order = np.argsort(key, kind="stable")
    dt = np.int32 if W * H * D < 2**31 else np.int64
    return key[order].astype(dt), order.astype(np.int32)


@partial(jax.jit, static_argnames=("W", "H", "D"))
def _sample(feats_sorted, keys_sorted, q, W, H, D):
    base = jnp.floor(q - 0.5).astype(jnp.int32)  # [L,3]
    cor = base[:, None, :] + jnp.asarray(_CORNERS)[None]  # [L,8,3]
    inb = (cor[..., 0] >= 0) & (cor[..., 0] < W) & (cor[..., 1] >= 0) & (cor[..., 1] < H) & \
          (cor[..., 2] >= 0) & (cor[..., 2] < D)
    k = (cor[..., 0].astype(keys_sorted.dtype) * H + cor[..., 1]) * D + cor[..., 2]
    pos = jnp.searchsorted(keys_sorted, k, side="left")
    pos = jnp.minimum(pos, keys_sorted.shape[0] - 1)
    hit = inb & (keys_sorted[pos] == k)
    d = jnp.abs(q[:, None, :] - cor.astype(jnp.float32) - 0.5)
    w = jnp.prod(1.0 - d, axis=-1)
    w = jnp.where(hit, w, 0.0)
    w = w / jnp.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    f = feats_sorted[pos]  # [L,8,C]
    return jnp.einsum("lk,lkc->lc", w, f)


def grid_sample_trilinear(feats, coords, grid_size, query, chunk=1 << 20):
    """feats [N,C], coords [N,3] (voxel indices), grid_size (W,H,D), query [L,3] in voxel units -> [L,C] numpy."""
    W, H, D = (int(g) for g in grid_size)
    keys, order = build_index(coords, (W, H, D))
    fs = jnp.asarray(np.asarray(feats, np.float32)[order])
    ks = jnp.asarray(keys)
    q = np.asarray(query, np.float32)
    out = []
    for s in range(0, q.shape[0], chunk):
        part = q[s:s + chunk]
        n = part.shape[0]
        B = 1
        while B < n:
            B *= 2
        pad = np.zeros((B, 3), np.float32)
        pad[:n] = part
        out.append(np.asarray(_sample(fs, ks, jnp.asarray(pad), W=W, H=H, D=D))[:n])
    if not out:
        return np.zeros((0, feats.shape[1]), np.float32)
    return np.concatenate(out, axis=0)
