"""QEM simplification for stage 10 (replaces CuMesh simplify).

simplify() (used by to_glb) is a port of the simplify call path of CuMesh 12289e1, whose LICENSE ships unchanged
beside this module as LICENSE-CuMesh (CuMesh.simplify, its simplify step, the vertex-face adjacency, edge and boundary helpers and its QEM
type): the same target, the same
stopping criterion (stop after the first parallel step that reaches <= target, so the result may land under it),
the same threshold schedule (thresh 1e-8, x10 whenever a step removes < 1% of the faces) and the same cost
(QEM at the boundary-weighted midpoint + 1e-2 x edge length^2 + 1e-3 x mean skinny term x edge length^2, INF on a
normal flip), with the same one-pass parallel rule (an edge collapses iff cost <= thresh and its packed
(cost bits, edge index) is the minimum on every face around both endpoints). All arithmetic is float32 in the
kernel's operation order. Known non-bitwise points (inferred, not measured): nvcc FMA contraction and rsqrtf vs our IEEE
mul/add and 1/sqrt; CUDA's atomics-order vert2face lists vs our face-id order (summation order only); NaN costs
are given CUDA's canonical NaN bits 0x7FFFFFFF.

"""
import time

import numpy as np

from . import meshops

# Port of the official CuMesh simplify call path (see module docstring). float32 throughout, kernel operation order.
# ---------------------------------------------------------------------------------------------------------------
F32 = np.float32
_U64MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
_CUDA_NAN_BITS = np.uint32(0x7FFFFFFF)


def _vert2face(F, nv):
    """get_vertex_face_adjacency: CSR of incident face ids per vertex (a face with a repeated vertex is listed once
    per occurrence, as the kernel does). Order inside a list: face id ascending (CUDA: atomics order)."""
    flat = F.reshape(-1)
    order = np.argsort(flat, kind="stable")
    v2f = order // 3
    cnt = np.bincount(flat, minlength=nv)
    off = np.zeros(nv + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    return v2f, off


def _edges(F):
    """get_edges: (min << 32) | max per face side, radix-sorted, run-length unique -> (edges, edge2face_cnt)."""
    a, b, c = (F[:, k].astype(np.uint64) for k in range(3))
    s = np.uint64(32)
    e = np.stack([(np.minimum(a, b) << s) | np.maximum(a, b), (np.minimum(b, c) << s) | np.maximum(b, c),
                  (np.minimum(c, a) << s) | np.maximum(c, a)], axis=1).reshape(-1)
    return np.unique(e, return_counts=True)


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _vertex_qems(V, F, v2f, off):
    """get_qem_kernel: per face n = normalize(cross(v1 - v0, v2 - v0)), d = -(n . v0); per vertex the planes' outer
    products (upper triangle, 10 floats) summed in list order."""
    nv = V.shape[0]
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    nx, ny, nz = _cross(e1[:, 0], e1[:, 1], e1[:, 2], e2[:, 0], e2[:, 1], e2[:, 2])
    with np.errstate(all="ignore"):
        inv = F32(1) / np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx * inv, ny * inv, nz * inv
        d = -(nx * v0[:, 0] + ny * v0[:, 1] + nz * v0[:, 2])
        planes = np.stack([nx * nx, nx * ny, nx * nz, nx * d, ny * ny, ny * nz, ny * d, nz * nz, nz * d, d * d], 1)
    q = np.zeros((nv, 10), F32)
    vid = np.repeat(np.arange(nv), np.diff(off))
    with np.errstate(all="ignore"):
        np.add.at(q, vid, planes[v2f])
    return q


def _qem_eval(Q, x, y, z):
    """QEM::evaluate, v = (x, y, z, 1), terms added in the kernel's order."""
    w = F32(1)
    two = F32(2)
    r = F32(0) + Q[:, 0] * x * x
    r = r + two * Q[:, 1] * x * y
    r = r + two * Q[:, 2] * x * z
    r = r + two * Q[:, 3] * x * w
    r = r + Q[:, 4] * y * y
    r = r + two * Q[:, 5] * y * z
    r = r + two * Q[:, 6] * y * w
    r = r + Q[:, 7] * z * z
    r = r + two * Q[:, 8] * z * w
    r = r + Q[:, 9] * w * w
    return r


def _entries(e0, e1, v2f, off):
    """Per edge: the incident faces of e0 (keep e0, other e1), then those of e1 (keep e1, other e0), in list order.
    This is the loop order of get_edge_collapse_cost_kernel, propagate_cost_kernel and collapse_edges_kernel."""
    d0 = off[e0 + 1] - off[e0]
    d1 = off[e1 + 1] - off[e1]
    n = d0 + d1
    E = e0.shape[0]
    edge_of = np.repeat(np.arange(E), n)
    start = np.cumsum(n) - n
    pos = np.arange(edge_of.shape[0]) - start[edge_of]
    first = pos < d0[edge_of]
    keep = np.where(first, e0[edge_of], e1[edge_of])
    other = np.where(first, e1[edge_of], e0[edge_of])
    local = np.where(first, pos, pos - d0[edge_of])
    face = v2f[off[keep] + local]
    return edge_of, first, keep, other, face


def _collapse_costs(V, F, e0, e1, vb, q, ent, lam_e, lam_s, chunk=8_000_000):
    """get_edge_collapse_cost_kernel. Returns (cost float32 (E,), v_new float32 (E, 3))."""
    edge_of, first, keep, other, face = ent
    E = e0.shape[0]
    b0, b1 = vb[e0].astype(bool), vb[e1].astype(bool)
    w0 = np.full(E, F32(0.5), F32)
    w0[b0 & ~b1] = F32(1)
    w0[~b0 & b1] = F32(0)
    p0, p1 = V[e0], V[e1]
    vnew = p0 * w0[:, None] + p1 * (F32(1) - w0)[:, None]
    with np.errstate(all="ignore"):
        cost = F32(0) + _qem_eval(q[e0] + q[e1], vnew[:, 0], vnew[:, 1], vnew[:, 2])
        dd = p1 - p0
        len2 = dd[:, 0] * dd[:, 0] + dd[:, 1] * dd[:, 1] + dd[:, 2] * dd[:, 2]
        cost = cost + F32(lam_e) * len2
    sk = np.zeros(E, F32)
    nt = np.zeros(E, np.int64)
    flip = np.zeros(E, bool)
    eps = F32(1e-12)
    k4s3 = F32(4) * np.sqrt(F32(3))
    for s in range(0, edge_of.shape[0], chunk):
        sl = slice(s, s + chunk)
        eo, kp, ot, fc = edge_of[sl], keep[sl], other[sl], face[sl]
        fv = F[fc]
        skip = (fv == ot[:, None]).any(1)
        live = ~skip
        eo, kp, fv = eo[live], kp[live], fv[live]
        a, b, c = V[fv[:, 0]], V[fv[:, 1]], V[fv[:, 2]]
        vn = vnew[eo]
        na = np.where((fv[:, 0] == kp)[:, None], vn, a)
        nb = np.where((fv[:, 1] == kp)[:, None], vn, b)
        nc = np.where((fv[:, 2] == kp)[:, None], vn, c)
        with np.errstate(all="ignore"):
            o1, o2 = b - a, c - a
            on = _cross(o1[:, 0], o1[:, 1], o1[:, 2], o2[:, 0], o2[:, 1], o2[:, 2])
            n1, n2 = nb - na, nc - na
            nn = _cross(n1[:, 0], n1[:, 1], n1[:, 2], n2[:, 0], n2[:, 1], n2[:, 2])
            new_area = F32(0.5) * np.sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2])
            fl = (on[0] * nn[0] + on[1] * nn[1] + on[2] * nn[2]) < F32(0)
            n0 = nc - nb
            den = (n0[:, 0] * n0[:, 0] + n0[:, 1] * n0[:, 1] + n0[:, 2] * n0[:, 2]) + \
                  (n1[:, 0] * n1[:, 0] + n1[:, 1] * n1[:, 1] + n1[:, 2] * n1[:, 2]) + \
                  (n2[:, 0] * n2[:, 0] + n2[:, 1] * n2[:, 1] + n2[:, 2] * n2[:, 2])
            den = np.where(den < eps, eps, den)
            shape = k4s3 * new_area / den
            term = F32(1) - np.fmin(np.fmax(shape, F32(0)), F32(1))
        flip[eo[fl]] = True
        np.add.at(sk, eo, term.astype(F32))
        np.add.at(nt, eo, 1)
    with np.errstate(all="ignore"):
        sk = np.where(nt > 0, sk / nt.astype(F32), sk)
        cost = cost + F32(lam_s) * sk * len2
    cost[flip] = F32(np.inf)
    return cost.astype(F32), vnew


def _pack(cost):
    bits = np.ascontiguousarray(cost, F32).view(np.uint32).copy()
    bits[np.isnan(cost)] = _CUDA_NAN_BITS
    return (bits.astype(np.uint64) << np.uint64(32)) | np.arange(cost.shape[0], dtype=np.uint64)


def simplify_step(V, F, lam_e, lam_s, thresh):
    """CuMesh::simplify_step: adjacency, edges, boundary, QEM, costs, propagate, one parallel collapse, compress."""
    nv, nf = V.shape[0], F.shape[0]
    v2f, off = _vert2face(F, nv)
    edges, ecnt = _edges(F)
    e0 = (edges >> np.uint64(32)).astype(np.int64)
    e1 = (edges & np.uint64(0xFFFFFFFF)).astype(np.int64)
    vb = np.zeros(nv, np.uint8)
    bnd = ecnt == 1
    vb[e0[bnd]] = 1
    vb[e1[bnd]] = 1
    q = _vertex_qems(V, F, v2f, off)
    ent = _entries(e0, e1, v2f, off)
    cost, vnew = _collapse_costs(V, F, e0, e1, vb, q, ent, lam_e, lam_s)
    pack = _pack(cost)
    edge_of, first, keep, other, face = ent
    prop = np.full(nf, _U64MAX, np.uint64)
    np.minimum.at(prop, face, pack[edge_of])
    with np.errstate(invalid="ignore"):
        cand = ~(cost > F32(thresh))
    lose = np.zeros(e0.shape[0], bool)
    lose[edge_of[prop[face] != pack[edge_of]]] = True
    win = cand & ~lose
    w = np.nonzero(win)[0]
    V2 = V.copy()
    V2[e0[w]] = vnew[w]
    vkeep = np.ones(nv, bool)
    vkeep[e1[w]] = False
    fkeep = np.ones(nf, bool)
    went = win[edge_of]
    # delete shared faces: faces of e0 that contain e1 (read before any update, as in the kernel)
    s0 = went & first
    f0 = face[s0]
    fkeep[f0[(F[f0] == other[s0][:, None]).any(1)]] = False
    # update faces of e1: replace the first slot equal to e1 by e0; a face listed twice gets a second pass
    s1 = went & ~first
    f1, r1, k1 = face[s1], other[s1], keep[s1]  # here keep = e1 (removed), other = e0 (kept)
    F2 = F.copy()
    o = np.argsort(f1, kind="stable")
    f1, r1, k1 = f1[o], r1[o], k1[o]
    rank = np.zeros(f1.shape[0], np.int64)
    if f1.shape[0]:
        same = np.r_[False, f1[1:] == f1[:-1]]
        grp = np.cumsum(~same) - 1
        gstart = np.nonzero(~same)[0]
        rank = np.arange(f1.shape[0]) - gstart[grp]
    for r in range(3):
        sel = rank == r
        if not sel.any():
            break
        ff, kk, rr = f1[sel], k1[sel], r1[sel]
        m = F2[ff] == kk[:, None]
        has = m.any(1)
        j = np.argmax(m, axis=1)
        F2[ff[has], j[has]] = rr[has]
    vmap = np.zeros(nv + 1, np.int64)
    np.cumsum(vkeep, out=vmap[1:])
    return V2[vkeep], vmap[F2[fkeep]], int(w.shape[0])


def simplify(V, F, target_num_faces, return_stats=False, options=None, max_steps=100000):
    """CuMesh.simplify(target_num_faces, options={}) as called by the official to_glb: no-op when F <= target;
    otherwise repeat simplify_step until a step ends with <= target faces; thresh starts at options thresh (1e-8)
    and is multiplied by 10 whenever a step removes less than 1% of the faces. The result can land under the
    target by up to one step's worth of collapses. max_steps only raises (the official loop has no cap)."""
    options = options or {}
    assert isinstance(target_num_faces, (int, np.integer)) and target_num_faces > 0
    V, F = meshops._as(V, F)
    num_face = int(F.shape[0])
    stats = {"rule": "cumesh_simplify_port", "in_faces": num_face, "target": int(target_num_faces)}
    if num_face <= target_num_faces:
        stats.update(out_faces=num_face, steps=0)
        return (V, F, stats) if return_stats else (V, F)
    thresh = float(options.get("thresh", 1e-8))
    lam_e = float(options.get("lambda_edge_length", 1e-2))
    lam_s = float(options.get("lambda_skinny", 1e-3))
    log = []
    t0 = time.time()
    while True:
        V, F, ncol = simplify_step(V, F, lam_e, lam_s, thresh)
        new_num_face = int(F.shape[0])
        log.append([float(F32(thresh)), new_num_face, ncol])
        if new_num_face <= target_num_faces:
            break
        if (num_face - new_num_face) / num_face < 1e-2:
            thresh *= 10
        num_face = new_num_face
        if len(log) >= max_steps:
            raise RuntimeError("simplify: max_steps reached at %d faces" % new_num_face)
    nref = np.zeros(V.shape[0], bool)
    nref[F.reshape(-1)[F.reshape(-1) < V.shape[0]]] = True
    stats.update(out_faces=int(F.shape[0]), out_vertices=int(V.shape[0]), steps=len(log), step_log=log,
                 unreferenced_vertices=int((~nref).sum()), face_index_oob=int((F >= V.shape[0]).sum()),
                 t_s=time.time() - t0)
    return (V, F, stats) if return_stats else (V, F)
