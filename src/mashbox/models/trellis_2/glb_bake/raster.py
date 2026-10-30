"""UV-space triangle rasterizer in JAX (our own code, written from scratch; replaces the GPU rasterize +
interpolate calls in o_voxel.postprocess.to_glb).

Output layout is the one the official to_glb reads, so the rest of the bake reads the same as the official code:
  rast[r, c] = (b0, b1, 0, face_id + 1), face_id + 1 == 0 means "no triangle".
  b0, b1 are the barycentric weights of the face's vertex 0 and vertex 1; vertex 2 gets 1 - b0 - b1.
Texel grid: column c covers u in [c/R, (c+1)/R), row r covers v in [r/R, (r+1)/R); a texel is sampled at its centre
((c+0.5)/R, (r+0.5)/R), which is the official uv*2-1 clip mapping with pixel centres at half-integers, row 0 at v=0.
Both windings are rasterized (no culling). Degenerate faces are skipped.
Overlaps: the highest face id wins (the official code overwrites earlier 100k-face chunks with later ones).
Ties on shared edges: each edge is evaluated from a canonical endpoint order, so the two faces sharing an edge get
exactly negated edge values, and the edge is owned by the face that sees it pointing "down" (dy<0, or dy==0 and dx>0).
That makes the raster watertight: a texel centre on a shared interior edge is covered exactly once.

Algorithm (fixed shapes, TPU friendly): faces are bucketed by bbox size S (power of two). For a bucket every face
enumerates its S x S candidate texels, evaluates the 3 edge functions, and scatter-maxes face_id+1 into a flat
[R*R] buffer. A second pass gathers the winning face per texel and computes the barycentrics.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def _edge(ax, ay, bx, by, px, py):
    """Edge function of directed edge a->b at p, evaluated from a canonical endpoint order so that
    edge(a, b, p) == -edge(b, a, p) bit for bit."""
    swap = (ax > bx) | ((ax == bx) & (ay > by))
    sx = jnp.where(swap, bx, ax)
    sy = jnp.where(swap, by, ay)
    ex = jnp.where(swap, ax, bx)
    ey = jnp.where(swap, ay, by)
    e = (ex - sx) * (py - sy) - (ey - sy) * (px - sx)
    return jnp.where(swap, -e, e)


def _owns(ax, ay, bx, by):
    """Tie owner for a texel exactly on edge a->b (after the face is oriented to positive area)."""
    dx = bx - ax
    dy = by - ay
    return (dy < 0) | ((dy == 0) & (dx > 0))


@partial(jax.jit, static_argnames=("S", "R"))
def _raster_bucket(buf, P, fid, x0, y0, S, R):
    """P: [B,3,2] texel-space vertex positions (centre at integer); fid: [B] int32 (-1 = padding);
    x0, y0: [B] int32 bbox origin. Scatter-max face_id+1 into buf [R*R]."""
    oy, ox = jnp.meshgrid(jnp.arange(S, dtype=jnp.int32), jnp.arange(S, dtype=jnp.int32), indexing="ij")
    px = (x0[:, None] + ox.reshape(-1)[None, :])  # [B, S*S]
    py = (y0[:, None] + oy.reshape(-1)[None, :])
    fx = px.astype(jnp.float32)
    fy = py.astype(jnp.float32)
    a = P[:, 0]
    b = P[:, 1]
    c = P[:, 2]
    area = _edge(a[:, 0], a[:, 1], b[:, 0], b[:, 1], c[:, 0], c[:, 1])  # 2x signed area
    sgn = jnp.where(area < 0, -1.0, 1.0)[:, None]
    ok = (area != 0) & (fid >= 0)
    inside = ok[:, None] & (px < R) & (py < R)
    for (u, v) in ((b, c), (c, a), (a, b)):  # w0 opposite vertex 0, w1 opposite 1, w2 opposite 2
        w = _edge(u[:, 0, None], u[:, 1, None], v[:, 0, None], v[:, 1, None], fx, fy) * sgn
        # owner test on the edge as seen with positive orientation
        flip = (sgn[:, 0] < 0)
        ux = jnp.where(flip, v[:, 0], u[:, 0])
        uy = jnp.where(flip, v[:, 1], u[:, 1])
        vx = jnp.where(flip, u[:, 0], v[:, 0])
        vy = jnp.where(flip, u[:, 1], v[:, 1])
        own = _owns(ux, uy, vx, vy)[:, None]
        inside = inside & ((w > 0) | ((w == 0) & own))
    idx = jnp.where(inside, py * R + px, R * R)  # out-of-range index -> dropped
    val = jnp.where(inside, fid[:, None] + 1, 0)
    return buf.at[idx.reshape(-1)].max(val.reshape(-1), mode="drop")


@partial(jax.jit, static_argnames=("R",))
def _barycentrics(buf, P_all, R):
    """buf [R*R] face_id+1 -> rast [R, R, 4] (b0, b1, 0, face_id+1)."""
    f = buf - 1
    valid = f >= 0
    tri = P_all[jnp.maximum(f, 0)]  # [R*R, 3, 2]
    n = buf.shape[0]
    rr = jnp.arange(n, dtype=jnp.int32)
    fx = (rr % R).astype(jnp.float32)
    fy = (rr // R).astype(jnp.float32)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = _edge(a[:, 0], a[:, 1], b[:, 0], b[:, 1], c[:, 0], c[:, 1])
    w0 = _edge(b[:, 0], b[:, 1], c[:, 0], c[:, 1], fx, fy)
    w1 = _edge(c[:, 0], c[:, 1], a[:, 0], a[:, 1], fx, fy)
    safe = jnp.where(valid & (area != 0), area, 1.0)
    b0 = jnp.where(valid, w0 / safe, 0.0)
    b1 = jnp.where(valid, w1 / safe, 0.0)
    out = jnp.stack([b0, b1, jnp.zeros_like(b0), buf.astype(jnp.float32)], axis=-1)
    return out.reshape(R, R, 4)


def _pad_pow2(n, lo=256):
    m = lo
    while m < n:
        m *= 2
    return m


def rasterize_uv(uvs, faces, resolution, max_candidates=1 << 23):
    """uvs [V,2] in [0,1], faces [F,3] -> rast [R,R,4] float32 (numpy), in the layout of the module doc.
    Bucket and padding sizes are powers of two so a long run recompiles only a handful of shapes."""
    R = int(resolution)
    uv = np.asarray(uvs, np.float32)
    fa = np.asarray(faces, np.int64)
    P = uv[fa] * np.float32(R) - np.float32(0.5)  # [F,3,2] texel space, centres at integers
    lo = np.ceil(P.min(axis=1)).astype(np.int64)
    hi = np.floor(P.max(axis=1)).astype(np.int64)
    lo = np.clip(lo, 0, R - 1)
    hi = np.clip(hi, -1, R - 1)
    ext = np.maximum(hi - lo + 1, 0).max(axis=1)  # [F]
    nonempty = (hi >= lo).all(axis=1)
    buf = jnp.zeros((R * R,), jnp.int32)
    S = 1
    remaining = np.nonzero(nonempty)[0]
    while remaining.size:
        sel = remaining[ext[remaining] <= S]
        remaining = remaining[ext[remaining] > S]
        chunk = max(1, max_candidates // (S * S))
        for s0 in range(0, sel.size, chunk):
            part = sel[s0:s0 + chunk]
            B = _pad_pow2(part.size)
            Pp = np.zeros((B, 3, 2), np.float32)
            Pp[:part.size] = P[part]
            fid = np.full((B,), -1, np.int32)
            fid[:part.size] = part
            x0 = np.zeros((B,), np.int32)
            y0 = np.zeros((B,), np.int32)
            x0[:part.size] = lo[part, 0]
            y0[:part.size] = lo[part, 1]
            buf = _raster_bucket(buf, jnp.asarray(Pp), jnp.asarray(fid), jnp.asarray(x0), jnp.asarray(y0), S=S, R=R)
        S *= 2
    rast = _barycentrics(buf, jnp.asarray(P), R=R)
    return np.asarray(rast)


def interpolate(attr, rast, faces):
    """Per-pixel attribute for one image, blended from the covering face's three vertices with the raster's barycentric
    weights (b0, b1, 1 - b0 - b1): attr [V,C], rast [R,R,4], faces [F,3] -> [R,R,C] (0 where empty)."""
    attr = np.asarray(attr, np.float32)
    fid = rast[..., 3].astype(np.int64) - 1
    valid = fid >= 0
    tri = np.asarray(faces, np.int64)[np.maximum(fid, 0)]  # [R,R,3]
    b0 = rast[..., 0:1]
    b1 = rast[..., 1:2]
    out = attr[tri[..., 0]] * b0 + attr[tri[..., 1]] * b1 + attr[tri[..., 2]] * (1.0 - b0 - b1)
    return np.where(valid[..., None], out, 0.0).astype(np.float32)
