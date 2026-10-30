"""Flexible dual grid (FDG) -> triangle mesh, our own numpy + JAX code (the stage 8 tail).

Reimplements the math of o_voxel.convert.flexible_dual_grid_to_mesh (MIT, TRELLIS.2 @ 75fbf01) without the
CUDA hashmap. The voxel lookup is a sort + binary search over uint32 linear keys (same key range rule as the
official uint32 hashmap: prod(grid) must be < 2**32).

Semantics matched (read from the MIT source):
  * vertices: one per voxel, V[i] = (coords[i] + dual_vertices[i]) * voxel_size + aabb[0], all N voxels kept
    (unreferenced ones included), float32.
  * for voxel i and axis a with intersected[i, a]: quad = voxels at coords[i] + OFF[a, k], k = 0..3, with
      OFF[x] = (0,0,0),(0,0,1),(0,1,1),(0,1,0)
      OFF[y] = (0,0,0),(1,0,0),(1,0,1),(0,0,1)
      OFF[z] = (0,0,0),(0,1,0),(1,1,0),(1,0,0)
    the quad is kept only if all 4 voxels exist (out-of-grid neighbours count as missing).
  * quad order = voxel-major, axis-minor (torch boolean-mask order), two triangles per quad, in order.
  * split with weights (inference path, weights = softplus(h[:, 6])): w02 = w[q0]*w[q2], w13 = w[q1]*w[q3];
    w02 > w13 -> (0,1,2),(0,2,3) else (0,1,3),(3,1,2). Ties go to the second split.
  * split without weights: unnormalised dot of the two triangle normals for each split, larger |dot| wins,
    ties go to the second split.
  * train path: one extra vertex per kept quad (weighted mean of the diagonals), 4 triangles per quad.
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np

EDGE_NEIGHBOR_OFFSET = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],   # x axis
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],   # y axis
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],   # z axis
], dtype=np.int32)                                    # (3, 4, 3)
QUAD_SPLIT_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
QUAD_SPLIT_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)
QUAD_SPLIT_TRAIN = np.array([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=np.int64)
KEY_SENTINEL = np.uint32(0xFFFFFFFF)


def _grid_and_voxel_size(aabb, grid_size=None, voxel_size=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same float32 arithmetic as the official function."""
    aabb = np.asarray(aabb, dtype=np.float32).reshape(2, 3)
    if voxel_size is not None:
        vs = np.broadcast_to(np.asarray(voxel_size, dtype=np.float32), (3,)).astype(np.float32)
        gs = np.round((aabb[1] - aabb[0]) / vs).astype(np.int32)
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        gs = np.broadcast_to(np.asarray(grid_size, dtype=np.int32), (3,)).astype(np.int32)
        vs = ((aabb[1] - aabb[0]) / gs.astype(np.float32)).astype(np.float32)
    vol = int(gs[0]) * int(gs[1]) * int(gs[2])
    assert vol < 2 ** 32, f"grid volume {vol} needs 64-bit keys (official code switches to uint64 too)"
    return aabb, gs, vs


# ----------------------------------------------------------------------------------------------------------
# decoder head (the 7 output channels of FlexiDualGridVaeDecoder, eval path)
# ----------------------------------------------------------------------------------------------------------
def decoder_head_np(h: np.ndarray, voxel_margin: float = 0.5):
    """h [N, 7] float32 -> dual_vertices [N,3] f32, intersected [N,3] bool, split_weight [N,1] f32.
    vertices = (1 + 2m) * sigmoid(h[:, :3]) - m ; intersected = h[:, 3:6] > 0 ; quad_lerp = softplus(h[:, 6:7])
    softplus follows torch defaults (beta 1, threshold 20: identity above 20)."""
    h = np.asarray(h, dtype=np.float32)
    sig = (np.float32(1) / (np.float32(1) + np.exp(-h[:, 0:3]))).astype(np.float32)
    dual = (np.float32(1 + 2 * voxel_margin) * sig - np.float32(voxel_margin)).astype(np.float32)
    inter = h[:, 3:6] > 0
    x = h[:, 6:7]
    sp = np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(np.float32)
    return dual, inter, sp


# ----------------------------------------------------------------------------------------------------------
# numpy reference-path implementation (host)
# ----------------------------------------------------------------------------------------------------------
def _linear_key_np(c: np.ndarray, gs: np.ndarray) -> np.ndarray:
    c = c.astype(np.uint64)
    return (c[..., 0] * np.uint64(gs[1]) * np.uint64(gs[2]) + c[..., 1] * np.uint64(gs[2]) + c[..., 2]).astype(np.uint32)


def lookup_np(coords: np.ndarray, queries: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Index of each query voxel in coords, or -1 (missing or outside the grid). Sort + searchsorted."""
    keys = _linear_key_np(coords, gs)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    inb = np.all((queries >= 0) & (queries < gs.reshape((1,) * (queries.ndim - 1) + (3,))), axis=-1)
    qk = _linear_key_np(np.where(inb[..., None], queries, 0), gs)
    pos = np.searchsorted(sk, qk, side="left")
    pos_c = np.minimum(pos, max(len(sk) - 1, 0))
    if len(sk) == 0:
        return np.full(qk.shape, -1, dtype=np.int64)
    found = inb & (sk[pos_c] == qk)
    return np.where(found, order[pos_c].astype(np.int64), -1)


def _float_dtype(a, prod_dtype):
    """dtype the official code multiplies split weights in: the split_weight tensor's own dtype (fp32 in the
    f32 reference, fp16/bf16 if the decoder emits that). prod_dtype overrides (e.g. "bfloat16")."""
    if prod_dtype is not None:
        if isinstance(prod_dtype, str) and prod_dtype in ("bfloat16", "torch.bfloat16"):
            import ml_dtypes
            return np.dtype(ml_dtypes.bfloat16)
        if isinstance(prod_dtype, str):
            return np.dtype(prod_dtype.replace("torch.", ""))
        return np.dtype(prod_dtype)
    a = np.asarray(a)
    return a.dtype if a.dtype.kind == "f" or a.dtype.name == "bfloat16" else np.dtype(np.float32)


def fdg_to_mesh_np(coords, dual_vertices, intersected, split_weight, aabb, voxel_size=None, grid_size=None,
                   train: bool = False, prod_dtype=None):
    """numpy port. coords [N,3] int, dual_vertices [N,3] float, intersected [N,3] bool, split_weight [N,1] float
    or None. Returns vertices [N(+L),3] float32, faces [F,3] int64 (same order as the official function).
    The split-weight product w[q0]*w[q2] vs w[q1]*w[q3] is done in prod_dtype (default: split_weight's dtype),
    as torch does it in the tensor's dtype."""
    coords = np.asarray(coords).astype(np.int32)
    dual = np.asarray(dual_vertices, dtype=np.float32)
    inter = np.asarray(intersected).astype(bool)
    aabb, gs, vs = _grid_and_voxel_size(aabb, grid_size, voxel_size)
    N = coords.shape[0]

    nb = coords[:, None, None, :] + EDGE_NEIGHBOR_OFFSET[None]          # (N,3,4,3)
    cand = nb[inter]                                                     # (M,4,3) voxel-major, axis-minor
    idx = lookup_np(coords, cand, gs)                                    # (M,4)
    quads = idx[np.all(idx >= 0, axis=1)]                                # (L,4)
    L = quads.shape[0]

    V = ((coords.astype(np.float32) + dual) * vs + aabb[0]).astype(np.float32)
    if not train:
        if split_weight is None:
            t1 = quads[:, QUAD_SPLIT_1]
            t2 = quads[:, QUAD_SPLIT_2]
            a1 = _split_align_np(V, t1)
            a2 = _split_align_np(V, t2)
            faces = np.where((a1 > a2)[:, None], t1, t2).reshape(-1, 3)
        else:
            pd = _float_dtype(split_weight, prod_dtype)
            w = np.asarray(split_weight).astype(pd).reshape(-1)
            ws = w[quads]
            w02 = (ws[:, 0] * ws[:, 2]).astype(pd)
            w13 = (ws[:, 1] * ws[:, 3]).astype(pd)
            faces = np.where((w02 > w13)[:, None], quads[:, QUAD_SPLIT_1], quads[:, QUAD_SPLIT_2]).reshape(-1, 3)
        return V, faces
    assert split_weight is not None, "split_weight must be provided in training mode"
    w = np.asarray(split_weight, dtype=np.float32).reshape(-1, 1)
    qv = V[quads]
    m02 = ((qv[:, 0] + qv[:, 2]) / np.float32(2)).astype(np.float32)
    m13 = ((qv[:, 1] + qv[:, 3]) / np.float32(2)).astype(np.float32)
    ws = w[quads]                                                        # (L,4,1)
    w02 = ws[:, 0] * ws[:, 2]
    w13 = ws[:, 1] * ws[:, 3]
    mid = ((w02 * m02 + w13 * m13) / (w02 + w13)).astype(np.float32)
    V = np.concatenate([V, mid], axis=0)
    q5 = np.concatenate([quads, np.arange(N, N + L, dtype=np.int64)[:, None]], axis=1)
    return V, q5[:, QUAD_SPLIT_TRAIN].reshape(-1, 3)


def _split_align_np(V, tri):
    a, b, c, d = (V[tri[:, k]] for k in (0, 1, 2, 3))
    # as the official code: columns 0..3 of the 6-index split vector,
    # n0 = (v1-v0) x (v2-v0), n1 = (v2-v1) x (v3-v1), score = |n0 . n1| (unnormalised)
    n0 = np.cross(b - a, c - a)
    n1 = np.cross(c - b, d - b)
    return np.abs(np.sum(n0 * n1, axis=1, dtype=np.float32)).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------
# JAX implementation (static shapes; jit-able; this is the device path)
# ----------------------------------------------------------------------------------------------------------
def make_fdg_to_mesh_jax(grid_size: Sequence[int], aabb=((-0.5,) * 3, (0.5,) * 3), split: str = "weight",
                         search_method: str = "scan", prod_dtype: str = "float32"):
    """Return a jitted fn(coords[Np,3] i32, dual[Np,3] f32, inter[Np,3] bool, split_weight[Np,1] f32,
    valid[Np] bool) -> (V[Np,3] f32, faces[6*Np,3] i32, face_valid[6*Np] bool).

    Np is a padded voxel count (bucket); padded rows have valid=False and are never matched. Candidate quad
    slot q = 3*i + a (voxel-major, axis-minor), triangles 2q and 2q+1, so a stable compaction of face_valid
    reproduces the official face order exactly. split in {"weight", "angle"}."""
    import jax
    import jax.numpy as jnp

    aabb_np, gs, vs = _grid_and_voxel_size(aabb, grid_size=grid_size)
    g1, g2 = int(gs[1]), int(gs[2])
    gs_j = jnp.asarray(gs, dtype=jnp.int32)
    off = jnp.asarray(EDGE_NEIGHBOR_OFFSET)
    s1 = jnp.asarray(QUAD_SPLIT_1, dtype=jnp.int32)
    s2 = jnp.asarray(QUAD_SPLIT_2, dtype=jnp.int32)
    vs_j = jnp.asarray(vs)
    a0_j = jnp.asarray(aabb_np[0])

    def key(c):
        c = c.astype(jnp.uint32)
        return c[..., 0] * jnp.uint32(g1 * g2) + c[..., 1] * jnp.uint32(g2) + c[..., 2]

    def fn(coords, dual, inter, split_weight, valid):
        Np = coords.shape[0]
        coords = coords.astype(jnp.int32)
        keys = jnp.where(valid, key(jnp.where(valid[:, None], coords, 0)), jnp.uint32(0xFFFFFFFF))
        order = jnp.argsort(keys, stable=True)
        sk = keys[order]
        nb = coords[:, None, None, :] + off[None]                               # (Np,3,4,3)
        inb = jnp.all((nb >= 0) & (nb < gs_j), axis=-1)
        qk = key(jnp.where(inb[..., None], nb, 0))
        pos = jnp.searchsorted(sk, qk, side="left", method=search_method)
        pos = jnp.minimum(pos, Np - 1)
        found = inb & (sk[pos] == qk) & (sk[pos] != jnp.uint32(0xFFFFFFFF))
        nidx = order[pos].astype(jnp.int32)                                      # (Np,3,4)
        qvalid = inter & valid[:, None] & jnp.all(found, axis=-1)               # (Np,3)
        quads = jnp.where(found, nidx, 0)

        V = (coords.astype(jnp.float32) + dual) * vs_j + a0_j
        t1 = quads[..., s1]                                                      # (Np,3,6)
        t2 = quads[..., s2]
        if split == "weight":
            w = split_weight.reshape(-1).astype(jnp.dtype(prod_dtype.replace("torch.", "")))
            ws = w[quads]
            choose1 = ws[..., 0] * ws[..., 2] > ws[..., 1] * ws[..., 3]
        elif split == "angle":
            def align(t):
                a, b, c, d = (V[t[..., k]] for k in range(4))
                n0 = jnp.cross(b - a, c - a)
                n1 = jnp.cross(c - b, d - b)
                return jnp.abs(jnp.sum(n0 * n1, axis=-1))
            choose1 = align(t1) > align(t2)
        else:
            raise ValueError(split)
        tri = jnp.where(choose1[..., None], t1, t2)                              # (Np,3,6)
        faces = tri.reshape(Np * 3 * 2, 3)
        fvalid = jnp.repeat(qvalid.reshape(-1), 2)
        return V, faces, fvalid

    return jax.jit(fn)


def compact_faces_jax(faces, fvalid, cap: int):
    """On-device stable compaction to a fixed cap (fill rows = -1). Returns (faces[cap,3], count)."""
    import jax.numpy as jnp
    (sel,) = jnp.nonzero(fvalid, size=cap, fill_value=-1)
    out = jnp.where((sel >= 0)[:, None], faces[jnp.maximum(sel, 0)], -1)
    return out, jnp.sum(fvalid.astype(jnp.int32))


def pad_bucket(n: int, buckets: Sequence[int] = (1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 21, 1 << 22, 1 << 23)) -> int:
    for b in buckets:
        if n <= b:
            return b
    return int(2 ** int(np.ceil(np.log2(n))))


def fdg_to_mesh_jax_host(coords, dual_vertices, intersected, split_weight, aabb, grid_size, split="weight",
                         Np: Optional[int] = None, fn=None):
    """Convenience wrapper: pad to a bucket, run the jitted device fn, compact on host. Returns V [N,3], F [F,3]."""
    coords = np.asarray(coords).astype(np.int32)
    N = coords.shape[0]
    Np = Np or pad_bucket(N)
    pad = Np - N
    c = np.concatenate([coords, np.zeros((pad, 3), np.int32)])
    d = np.concatenate([np.asarray(dual_vertices, np.float32), np.zeros((pad, 3), np.float32)])
    it = np.concatenate([np.asarray(intersected, bool), np.zeros((pad, 3), bool)])
    sw = np.ones((N, 1), np.float32) if split_weight is None else np.asarray(split_weight, np.float32).reshape(N, 1)
    sw = np.concatenate([sw, np.ones((pad, 1), np.float32)])
    valid = np.arange(Np) < N
    if fn is None:
        fn = make_fdg_to_mesh_jax(tuple(int(g) for g in np.broadcast_to(grid_size, (3,))), aabb, split=split)
    V, F, m = fn(c, d, it, sw, valid)
    V = np.asarray(V)[:N]
    F = np.asarray(F)[np.asarray(m)]
    return V, F.astype(np.int64)
