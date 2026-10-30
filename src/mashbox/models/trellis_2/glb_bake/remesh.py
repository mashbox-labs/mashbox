"""Narrow-band UDF + simple dual contouring remesh (to_glb remesh=True). Our own code.

Written from the documented semantics of CuMesh 12289e1 (MIT as published):
remesh_narrow_band_dc and its simple dual contouring step (mean of edge crossings, voxel centre
when none; per-axis crossing sign on the voxel's (u=1, v=1) edge) and the bounds-checked 3D hash lookup
(out-of-grid neighbour -> invalid, so the quad is dropped). Distance queries use our closest.py (ClosestPointMesh),
never cubvh: no cubvh code was read or copied for this module.

Arithmetic is float32 like the reference (positions, UDF, eps shift, crossings); the reference's python-float
scalars are cast to float32 the way torch applies them to float32 tensors. Bit-identity with the CUDA kernel is
not claimed (summation and FMA order can differ); the check is geometric.
"""
import numpy as np

from . import closest

F32 = np.float32
# 4 voxels around the voxel's (u=1, v=1) edge, per axis (remeshing.py edge_neighbor_voxel_offset)
_EDGE_NB = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], np.int64)
_Q1N = np.array([0, 1, 2, 0, 2, 3])
_Q1P = np.array([0, 2, 1, 0, 3, 2])
_Q2N = np.array([0, 1, 3, 3, 1, 2])
_Q2P = np.array([0, 3, 1, 3, 2, 1])
_OFF = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], np.int64)


def _udf(cp, P):
    _, _, d2 = cp.query(np.ascontiguousarray(P, F32))
    return np.sqrt(np.asarray(d2, F32))


def _keys(c, R):
    c = np.asarray(c, np.int64)
    return (c[:, 0] * R + c[:, 1]) * R + c[:, 2]


def _lookup(sorted_keys, order, c, R):
    """Index of each coord in the voxel table, -1 if out of the grid or absent (bounds-checked like the reference)."""
    c = np.asarray(c, np.int64)
    inb = ((c >= 0) & (c < R)).all(1)
    k = _keys(np.where(inb[:, None], c, 0), R)
    pos = np.searchsorted(sorted_keys, k)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    hit = inb & (sorted_keys[pos] == k)
    return np.where(hit, order[pos], -1)


def remesh_narrow_band_dc(vertices, faces, center, scale, resolution, band=1.0, project_back=0.0, cp=None,
                          return_stats=False):
    """vertices [N,3] float32, faces [M,3] int -> (V_new [n,3] float32, F_new [m,3] int32).
    center [3], scale float, resolution int (e.g. grid_size.max()); cp = ClosestPointMesh of (vertices, faces)."""
    V = np.ascontiguousarray(vertices, F32)
    Fc = np.ascontiguousarray(faces, np.int64)
    center = np.asarray(center, F32).reshape(3)
    if cp is None:
        cp = closest.ClosestPointMesh(V, Fc)
    st = {}
    eps = band * scale / resolution                         # python float in the reference
    base = resolution
    while base > 32:
        assert base % 2 == 0
        base //= 2
    g = np.arange(base, dtype=np.int64)
    coords = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    levels = []
    while True:
        cell = scale / base
        pts = ((coords.astype(F32) + F32(0.5)) / F32(base) - F32(0.5)) * F32(scale) + center
        d = np.abs(_udf(cp, pts) - F32(eps))
        coords = coords[d < F32(0.87 * cell)]
        levels.append([int(base), int(len(coords))])
        if base >= resolution:
            break
        base *= 2
        coords = ((coords * 2)[:, None, :] + _OFF[None]).reshape(-1, 3)
    st['levels'] = levels
    R = int(resolution)
    nvox = len(coords)
    # corner vertices of the active voxels (dedup) and their shifted UDF
    corners = (coords[:, None, :] + _OFF[None]).reshape(-1, 3)
    ck = _keys(corners, R + 1)
    uk, inv = np.unique(ck, return_inverse=True)
    gv = np.stack([uk // ((R + 1) * (R + 1)), (uk // (R + 1)) % (R + 1), uk % (R + 1)], 1)
    pv = (gv.astype(F32) / F32(R) - F32(0.5)) * F32(scale) + center
    val = _udf(cp, pv) - F32(eps)
    st['n_vox'], st['n_grid_verts'] = int(nvox), int(len(uk))
    cv = val[inv.reshape(nvox, 8)]                          # [nvox, 8] in _OFF order (x fastest)
    C = cv.reshape(nvox, 2, 2, 2).transpose(0, 3, 2, 1)     # C[:, dx, dy, dz]
    cf = coords.astype(F32)
    ssum = np.zeros((nvox, 3), F32)
    cnt = np.zeros((nvox,), np.int32)
    inter = np.zeros((nvox, 3), np.int32)
    for axis in range(3):
        for u in (0, 1):
            for v in (0, 1):
                if axis == 0:
                    a, b, off = C[:, 0, u, v], C[:, 1, u, v], (None, u, v)
                elif axis == 1:
                    a, b, off = C[:, u, 0, v], C[:, u, 1, v], (u, None, v)
                else:
                    a, b, off = C[:, u, v, 0], C[:, u, v, 1], (u, v, None)
                neg_a, neg_b = a < 0, b < 0
                x = neg_a != neg_b
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = np.where(x, -a / (b - a), F32(0)).astype(F32)
                for k in range(3):
                    add = (cf[:, k] + t) if off[k] is None else (cf[:, k] + F32(off[k]))
                    ssum[:, k] = np.where(x, ssum[:, k] + add, ssum[:, k])
                cnt += x
                if u == 1 and v == 1:
                    inter[:, axis] = np.where(neg_a & ~neg_b, 1, np.where(~neg_a & neg_b, -1, 0))
    has = cnt > 0
    dual = np.where(has[:, None], ssum / np.maximum(cnt, 1)[:, None].astype(F32), cf + F32(0.5)).astype(F32)
    # topology: 4 voxels around each crossing edge -> quad -> 2 triangles (split by normal alignment)
    nb = coords[:, None, None, :] + _EDGE_NB[None]          # [nvox, 3, 4, 3]
    m = inter != 0
    conn = nb[m]                                            # [M, 4, 3] (voxel-major, then axis)
    idir = inter[m]
    vk = _keys(coords, R)
    order = np.argsort(vk, kind='stable')
    q = _lookup(vk[order], order, conn.reshape(-1, 3), R).reshape(-1, 4)
    ok = (q >= 0).all(1)
    q, idir = q[ok], idir[ok]
    st['n_quads'] = int(len(q))
    uv_ = np.unique(q.reshape(-1))
    vmap = np.zeros((nvox,), np.int64)
    vmap[uv_] = np.arange(len(uv_))
    q = vmap[q]
    mv = (dual[uv_] / F32(R) - F32(0.5)) * F32(scale) + center
    pos = (idir == 1)[:, None]

    def _align(tri):
        n0 = np.cross(mv[tri[:, 1]] - mv[tri[:, 0]], mv[tri[:, 2]] - mv[tri[:, 0]])
        n1 = np.cross(mv[tri[:, 2]] - mv[tri[:, 1]], mv[tri[:, 3]] - mv[tri[:, 1]])
        return np.abs((n0 * n1).sum(1))

    t0 = np.where(pos, q[:, _Q1P], q[:, _Q1N])
    t1 = np.where(pos, q[:, _Q2P], q[:, _Q2N])
    tris = np.where((_align(t0) > _align(t1))[:, None], t0, t1).reshape(-1, 3)
    if project_back > 0:
        cpts, _, _ = cp.query(mv)
        mv = (mv - F32(project_back) * (mv - np.asarray(cpts, F32))).astype(F32)
    st['out'] = [int(len(mv)), int(len(tris))]
    out = (mv.astype(F32), tris.astype(np.int32))
    return (out + (st,)) if return_stats else out
