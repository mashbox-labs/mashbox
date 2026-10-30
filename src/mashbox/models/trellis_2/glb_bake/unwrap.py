"""UV unwrap for stage 10 (replaces CuMesh uv_unwrap). Same two-step structure as the official call:
  1. fast chart clustering on the mesh (ours): each face gets the axis of +-X/+-Y/+-Z closest to its normal, and charts
     are the edge-connected components of equal label. Every chart then lies inside a normal cone of half-angle
     <= 54.8 deg around its axis, tighter than the official threshold_cone_half_angle_rad = 90 deg. CuMesh refines its
     clusters iteratively; UVs are not unique, so they may differ.
  2. xatlas (MIT, python binding xatlas-python MIT): every chart is added as its own mesh, then compute_charts +
     pack_charts with default options (the official code passes empty kwargs), and the results are gathered with
     a vertex map back to the input vertices.
Returns (vertices [N',3], faces [M,3], uvs [N',2] in [0,1], vmap [N'] -> input vertex id).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import meshops

_AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64)


def compute_charts(V, F, min_chart_faces=0):
    """-> chart id per face [M] (int64), number of charts."""
    n = meshops.face_normals(V, F)
    lab = np.argmax(n @ _AXES.T, axis=1)
    a, b, _ = meshops._face_graph(F, V.shape[0], manifold_only=True)
    fa, fb = a // 3, b // 3
    same = lab[fa] == lab[fb]
    M = F.shape[0]
    G = sp.coo_matrix((np.ones(int(same.sum()), np.int8), (fa[same], fb[same])), shape=(M, M)).tocsr()
    nc, cid = connected_components(G, directed=False)
    return cid.astype(np.int64), int(nc)


def uv_unwrap(V, F, return_stats=False, chart_options=None, pack_options=None):
    import xatlas
    V, F = meshops.remove_degenerate_faces(V, F)
    cid, nc = compute_charts(V, F)
    order = np.argsort(cid, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(cid[order]))[0] + 1, order.shape[0]]
    atlas = xatlas.Atlas()
    vmaps = []
    for i in range(nc):
        fs = order[bounds[i]:bounds[i + 1]]
        Fi = F[fs]
        vid, Fl = np.unique(Fi.reshape(-1), return_inverse=True)
        atlas.add_mesh(np.ascontiguousarray(V[vid], np.float32), np.ascontiguousarray(Fl.reshape(-1, 3), np.uint32))
        vmaps.append(vid)
    co = xatlas.ChartOptions()
    po = xatlas.PackOptions()
    for k, v in (chart_options or {}).items():
        setattr(co, k, v)
    for k, v in (pack_options or {}).items():
        setattr(po, k, v)
    atlas.generate(co, po)
    outv, outf, outuv = [], [], []
    cnt = 0
    for i in range(nc):
        vm, fi, uv = atlas[i]
        outv.append(vmaps[i][vm])
        outf.append(fi.astype(np.int64) + cnt)
        outuv.append(uv)
        cnt += vm.shape[0]
    vmap = np.concatenate(outv)
    res = (V[vmap].astype(np.float32), np.concatenate(outf), np.concatenate(outuv).astype(np.float32), vmap)
    if return_stats:
        st = {"clusters": nc}
        for k in ("width", "height", "chart_count", "utilization"):
            v = getattr(atlas, k, None)
            if v is not None and not callable(v):
                st["atlas_" + k] = v if np.isscalar(v) else np.asarray(v).tolist()
        return res + (st,)
    return res
