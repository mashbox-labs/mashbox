"""Host-side mesh clean-up for stage 10, our own numpy/scipy code. Replaces the CuMesh calls in o_voxel.postprocess.to_glb:
fill_holes, remove_duplicate_faces, remove_degenerate_faces, repair_non_manifold_edges,
remove_small_connected_components, unify_face_orientations, compute_vertex_normals.

Semantics follow the CuMesh Python docstrings at the pinned commit (MIT, read for behaviour only, no code copied):
- fill_holes(max_hole_perimeter): every boundary loop whose perimeter <= max gets one new vertex at the mean of the loop
  vertices and a triangle fan to it, wound to match the faces adjacent to the loop.
- repair_non_manifold_edges: split vertices so every edge has at most two faces (duplicates share coordinates).
- remove_small_connected_components(min_area): drop edge-connected face components with total area < min_area.
All functions take and return (V float32 [N,3], F int64 [M,3]).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def _as(V, F):
    return np.ascontiguousarray(V, np.float32), np.ascontiguousarray(F, np.int64).reshape(-1, 3)


def face_areas(V, F):
    V64 = V.astype(np.float64)
    a, b, c = V64[F[:, 0]], V64[F[:, 1]], V64[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def remove_unreferenced_vertices(V, F):
    V, F = _as(V, F)
    used = np.zeros(V.shape[0], bool)
    used[F.reshape(-1)] = True
    remap = np.full(V.shape[0], -1, np.int64)
    remap[used] = np.arange(int(used.sum()))
    return V[used], remap[F]


def _half_edges(F):
    """[3M,2] directed edges (a->b) in face order: edge k of face f is row 3f+k, from F[f,k] to F[f,(k+1)%3]."""
    return np.stack([F, np.roll(F, -1, axis=1)], axis=-1).reshape(-1, 2)


def _undirected_ids(he, nv):
    lo = np.minimum(he[:, 0], he[:, 1])
    hi = np.maximum(he[:, 0], he[:, 1])
    key = lo * nv + hi
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return inv, cnt, uniq


def remove_duplicate_faces(V, F):
    V, F = _as(V, F)
    if F.shape[0] == 0:
        return V, F
    s = np.sort(F, axis=1)
    _, first = np.unique(s, axis=0, return_index=True)
    keep = np.zeros(F.shape[0], bool)
    keep[first] = True
    return V, F[keep]


def remove_degenerate_faces(V, F, abs_thresh=1e-24, rel_thresh=1e-12):
    V, F = _as(V, F)
    V64 = V.astype(np.float64)
    a, b, c = V64[F[:, 0]], V64[F[:, 1]], V64[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    le = np.max(np.stack([np.sum((b - a) ** 2, 1), np.sum((c - b) ** 2, 1), np.sum((a - c) ** 2, 1)], 1), 1)
    bad = (area < abs_thresh) | (area < rel_thresh * le) | (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])
    return remove_unreferenced_vertices(V, F[~bad])


def boundary_loops(F, nv):
    """Boundary loops as lists of vertex ids, walked in the hole direction (reverse of the adjacent face edges).
    Vertices with more than one outgoing hole edge (pinched boundary) are not walked through; loops touching them are
    returned in `skipped` count instead."""
    he = _half_edges(F)
    inv, cnt, _ = _undirected_ids(he, nv)
    bmask = cnt[inv] == 1
    hb = he[bmask][:, ::-1]  # hole edges b->a
    if hb.shape[0] == 0:
        return [], 0
    outdeg = np.bincount(hb[:, 0], minlength=nv)
    nxt = np.full(nv, -1, np.int64)
    simple = outdeg[hb[:, 0]] == 1
    nxt[hb[simple, 0]] = hb[simple, 1]
    visited = np.zeros(nv, bool)
    loops, skipped = [], 0
    starts = hb[simple, 0]
    for s in starts:
        if visited[s]:
            continue
        loop = [s]
        visited[s] = True
        cur = nxt[s]
        ok = True
        while cur != s:
            if cur < 0 or visited[cur] or outdeg[cur] != 1:
                ok = False
                break
            visited[cur] = True
            loop.append(cur)
            cur = nxt[cur]
        if ok and len(loop) >= 3:
            loops.append(np.asarray(loop, np.int64))
        else:
            skipped += 1
    return loops, skipped


def fill_holes(V, F, max_hole_perimeter=3e-2, return_stats=False):
    V, F = _as(V, F)
    loops, skipped = boundary_loops(F, V.shape[0])
    newV, newF = [], []
    nv = V.shape[0]
    filled = 0
    V64 = V.astype(np.float64)
    for lp in loops:
        p = V64[lp]
        per = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum()
        if per > max_hole_perimeter:
            continue
        c = nv + len(newV)
        newV.append(p.mean(axis=0))
        newF.append(np.stack([lp, np.roll(lp, -1), np.full(lp.shape[0], c)], axis=1))
        filled += 1
    if newV:
        V = np.concatenate([V, np.asarray(newV, np.float32)], axis=0)
        F = np.concatenate([F] + newF, axis=0)
    if return_stats:
        return V, F, {"loops": len(loops), "filled": filled, "skipped_pinched": skipped}
    return V, F


def repair_non_manifold_edges(V, F):
    """Split vertices along non-manifold edges and at non-manifold vertices: face corners that meet at a vertex stay
    one vertex only if they are connected through manifold edges (edges with exactly two faces) around it."""
    V, F = _as(V, F)
    M = F.shape[0]
    if M == 0:
        return V, F
    he = _half_edges(F)
    inv, cnt, _ = _undirected_ids(he, V.shape[0])
    man = cnt[inv] == 2
    hid = np.nonzero(man)[0]
    order = np.argsort(inv[hid], kind="stable")
    hid = hid[order]
    pa, pb = hid[0::2], hid[1::2]  # half-edge pairs on the same manifold edge
    fa, ka = pa // 3, pa % 3
    fb, kb = pb // 3, pb % 3
    # corner id = 3*f + k (corner k holds vertex F[f,k]); half-edge (f,k) goes F[f,k] -> F[f,(k+1)%3]
    va0, va1 = 3 * fa + ka, 3 * fa + (ka + 1) % 3
    vb0, vb1 = 3 * fb + kb, 3 * fb + (kb + 1) % 3
    Fa = F.reshape(-1)
    # link corners holding the same vertex
    r1 = np.where(Fa[va0] == Fa[vb0], vb0, vb1)
    r2 = np.where(Fa[va1] == Fa[vb0], vb0, vb1)
    rows = np.concatenate([va0, va1])
    cols = np.concatenate([r1, r2])
    n = 3 * M
    G = sp.coo_matrix((np.ones(rows.shape[0], np.int8), (rows, cols)), shape=(n, n)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    # first component that uses a vertex keeps the original id, extra components get copies
    vid = np.full(ncomp, -1, np.int64)
    vid[lab] = Fa  # any corner of the component (all share one vertex)
    nv = V.shape[0]
    # deterministic: components ordered by their smallest corner index; the first one per vertex keeps the id
    first_corner = np.full(ncomp, n, np.int64)
    np.minimum.at(first_corner, lab, np.arange(n))
    order = np.argsort(first_corner, kind="stable")
    v_sorted = vid[order]
    _, first_idx = np.unique(v_sorted, return_index=True)
    is_first = np.zeros(ncomp, bool)
    is_first[first_idx] = True
    newid = np.empty(ncomp, np.int64)
    newid[order[is_first]] = v_sorted[is_first]
    extra = v_sorted[~is_first]
    newid[order[~is_first]] = nv + np.arange(extra.shape[0])
    if extra.shape[0]:
        V = np.concatenate([V, V[extra]], axis=0)
    return V, newid[lab].reshape(M, 3)


def _face_graph(F, nv, manifold_only):
    he = _half_edges(F)
    inv, cnt, _ = _undirected_ids(he, nv)
    sel = (cnt[inv] == 2) if manifold_only else (cnt[inv] >= 2)
    hid = np.nonzero(sel)[0]
    order = np.argsort(inv[hid], kind="stable")
    hid = hid[order]
    e = inv[hid]
    # connect consecutive half-edges on the same edge (a chain is enough for connectivity)
    same = e[1:] == e[:-1]
    a, b = hid[:-1][same], hid[1:][same]
    return a, b, he


def remove_small_connected_components(V, F, min_area):
    V, F = _as(V, F)
    M = F.shape[0]
    if M == 0:
        return V, F
    a, b, _ = _face_graph(F, V.shape[0], manifold_only=False)
    G = sp.coo_matrix((np.ones(a.shape[0], np.int8), (a // 3, b // 3)), shape=(M, M)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    area = np.bincount(lab, weights=face_areas(V, F), minlength=ncomp)
    keep = area[lab] >= min_area
    return remove_unreferenced_vertices(V, F[keep])


def unify_face_orientations(V, F):
    """Make neighbouring faces (through manifold edges) traverse their shared edge in opposite directions. The face
    with the lowest index in each component keeps its winding. Parity to the root by pointer jumping on a BFS tree."""
    from scipy.sparse.csgraph import breadth_first_tree
    V, F = _as(V, F)
    M = F.shape[0]
    if M == 0:
        return V, F
    a, b, he = _face_graph(F, V.shape[0], manifold_only=True)
    fa, fb = a // 3, b // 3
    same_dir = (he[a, 0] == he[b, 0])  # same direction => one of them must flip relative to the other
    data = np.where(same_dir, 2, 1).astype(np.int8)  # 1 = consistent, 2 = flip
    G = sp.coo_matrix((np.concatenate([data, data]), (np.concatenate([fa, fb]), np.concatenate([fb, fa]))),
                      shape=(M, M)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    root = np.full(ncomp, M, np.int64)
    np.minimum.at(root, lab, np.arange(M))
    # BFS tree from a virtual super-root: run per component root via one BFS over a graph with an extra node
    S = M
    rows = np.concatenate([np.full(ncomp, S), G.tocoo().row])
    cols = np.concatenate([root, G.tocoo().col])
    vals = np.concatenate([np.ones(ncomp, np.int8), G.tocoo().data])
    G2 = sp.csr_matrix((vals, (rows, cols)), shape=(M + 1, M + 1))
    T = breadth_first_tree(G2, S, directed=True).tocoo()
    pred = np.full(M + 1, S, np.int64)
    pred[T.col] = T.row
    pred[S] = S
    par = np.zeros(M + 1, np.int8)
    child = T.col[T.row != S]
    parent = T.row[T.row != S]
    rel = np.asarray(G[child, parent]).reshape(-1)
    par[child] = (rel == 2).astype(np.int8)
    anc = pred.copy()
    for _ in range(64):
        if np.all(anc == S):
            break
        par = par ^ par[anc]
        anc = anc[anc]
    flip = par[:M].astype(bool)
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    return V, F


def vertex_normals(V, F):
    """Area-weighted vertex normals (sum of un-normalised face normals, then normalised)."""
    V64 = V.astype(np.float64)
    n = np.cross(V64[F[:, 1]] - V64[F[:, 0]], V64[F[:, 2]] - V64[F[:, 0]])
    vn = np.zeros_like(V64)
    for k in range(3):
        np.add.at(vn, F[:, k], n)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.maximum(ln, 1e-20)).astype(np.float32)


def face_normals(V, F):
    V64 = V.astype(np.float64)
    n = np.cross(V64[F[:, 1]] - V64[F[:, 0]], V64[F[:, 2]] - V64[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def cleanup(V, F, max_hole_perimeter=3e-2, min_component_area=1e-5):
    """The official clean-up block: remove_duplicate_faces, repair_non_manifold_edges,
    remove_small_connected_components(1e-5), fill_holes(3e-2)."""
    V, F = remove_duplicate_faces(V, F)
    V, F = repair_non_manifold_edges(V, F)
    V, F = remove_small_connected_components(V, F, min_component_area)
    V, F = fill_holes(V, F, max_hole_perimeter)
    return V, F
