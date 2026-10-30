"""Exact closest point on a triangle mesh (replaces cuBVH unsigned_distance(..., return_uvw=True) in to_glb, where the
baked texel position is snapped back onto the original high-res mesh). Our own code.

Device part (JAX, fixed shapes): point-to-triangle closest point by Voronoi-region classification (the textbook
method, e.g. Ericson, Real-Time Collision Detection 5.1.5), vectorised over [Q, K] query x candidate pairs.
Host part: candidate faces from a KD-tree on face centroids (scipy cKDTree, BSD-3). The result is EXACT, not
approximate: a face outside the K nearest centroids is at distance >= d_K - r_max (r_max = largest centroid-to-vertex
distance over all faces), so a query whose best candidate distance is <= d_K - r_max is final; the rest are re-run
with every face whose centroid lies within best + r_max (ball query).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree


def _dot(a, b):
    return jnp.sum(a * b, axis=-1)


@jax.jit
def closest_on_triangles(p, A, B, C):
    """p [...,3]; A,B,C [...,3] (broadcast) -> (closest point [...,3], squared distance [...])."""
    ab = B - A
    ac = C - A
    ap = p - A
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    bp = p - B
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    cp = p - C
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def sdiv(n, d):
        return n / jnp.where(d == 0, 1.0, d)

    # interior
    den = va + vb + vc
    v_in = sdiv(vb, den)
    w_in = sdiv(vc, den)
    res = A + ab * v_in[..., None] + ac * w_in[..., None]
    # edge BC
    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = sdiv(d4 - d3, (d4 - d3) + (d5 - d6))
    res = jnp.where(m_bc[..., None], B + (C - B) * w_bc[..., None], res)
    # edge AC
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = sdiv(d2, d2 - d6)
    res = jnp.where(m_ac[..., None], A + ac * w_ac[..., None], res)
    # vertex C, then edge AB (priority below A and B, matching the sequential early-return order A, B, AB, C, AC, BC)
    m_c = (d6 >= 0) & (d5 <= d6)
    res = jnp.where(m_c[..., None], C, res)
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = sdiv(d1, d1 - d3)
    res = jnp.where(m_ab[..., None], A + ab * v_ab[..., None], res)
    m_b = (d3 >= 0) & (d4 <= d3)
    res = jnp.where(m_b[..., None], B, res)
    m_a = (d1 <= 0) & (d2 <= 0)
    res = jnp.where(m_a[..., None], A, res)
    diff = p - res
    return res, _dot(diff, diff)


@partial(jax.jit, static_argnames=())
def _best_of_candidates(p, tris):
    """p [Q,3], tris [Q,K,3,3] -> (best point [Q,3], best d2 [Q], argmin [Q])."""
    pts, d2 = closest_on_triangles(p[:, None, :], tris[:, :, 0], tris[:, :, 1], tris[:, :, 2])
    k = jnp.argmin(d2, axis=1)
    bp = jnp.take_along_axis(pts, k[:, None, None], axis=1)[:, 0]
    bd = jnp.take_along_axis(d2, k[:, None], axis=1)[:, 0]
    return bp, bd, k


def _pad_rows(a, B):
    out = np.zeros((B,) + a.shape[1:], a.dtype)
    out[:a.shape[0]] = a
    return out


class ClosestPointMesh:
    def __init__(self, V, F, K=16, k_ext=0, workers=1):
        """k_ext = 0, workers = 1 is the original path. k_ext > K: query_ext (same results by construction)."""
        self.V = np.ascontiguousarray(V, np.float32)
        self.F = np.ascontiguousarray(F, np.int64)
        self.T = self.V[self.F]  # [M,3,3]
        cen = self.T.mean(axis=1)
        self.cen = cen
        self.rmax = float(np.sqrt(((self.T - cen[:, None]) ** 2).sum(-1)).max()) if len(self.F) else 0.0
        self.tree = cKDTree(cen.astype(np.float64))
        self.K = int(min(K, len(self.F)))
        self.k_ext = int(min(k_ext, len(self.F)))
        self.workers = int(workers)

    def query(self, P, chunk=1 << 16, return_stats=False):
        """P [Q,3] -> (closest points [Q,3] float32, face id [Q], squared distance [Q])."""
        if self.k_ext > self.K:
            return self.query_ext(P, chunk, return_stats)
        P = np.ascontiguousarray(P, np.float32)
        Q = P.shape[0]
        out_p = np.zeros((Q, 3), np.float32)
        out_d = np.zeros((Q,), np.float32)
        out_f = np.zeros((Q,), np.int64)
        K = self.K
        dK_all = np.zeros((Q,), np.float64)
        for s in range(0, Q, chunk):
            part = P[s:s + chunk]
            n = part.shape[0]
            dk, ik = self.tree.query(part.astype(np.float64), k=K)
            if K == 1:
                dk, ik = dk[:, None], ik[:, None]
            Bp = 1
            while Bp < n:
                Bp *= 2
            tris = _pad_rows(self.T[ik], Bp)
            bp, bd, k = _best_of_candidates(jnp.asarray(_pad_rows(part, Bp)), jnp.asarray(tris))
            bp, bd, k = np.asarray(bp)[:n], np.asarray(bd)[:n], np.asarray(k)[:n]
            out_p[s:s + n] = bp
            out_d[s:s + n] = bd
            out_f[s:s + n] = ik[np.arange(n), k]
            dK_all[s:s + n] = dk[:, -1]
        # exactness check (small slack for float32 rounding of the candidate distances)
        best = np.sqrt(out_d.astype(np.float64))
        need = (best > dK_all - self.rmax - 1e-7) & (K < len(self.F))
        nfb = int(need.sum())
        if nfb:
            idx = np.nonzero(need)[0]
            lists = self.tree.query_ball_point(P[idx].astype(np.float64), best[idx] + self.rmax + 1e-6)
            qi = np.concatenate([np.full(len(l), i, np.int64) for i, l in zip(idx, lists)])
            fi = np.concatenate([np.asarray(l, np.int64) for l in lists])
            order = np.lexsort((fi, qi))
            qi, fi = qi[order], fi[order]
            pts, d2 = [], []
            step = 1 << 20
            for s in range(0, qi.shape[0], step):
                q, f = qi[s:s + step], fi[s:s + step]
                n = q.shape[0]
                Bp = 1
                while Bp < n:
                    Bp *= 2
                tri = _pad_rows(self.T[f], Bp)
                pp, dd = closest_on_triangles(jnp.asarray(_pad_rows(P[q], Bp)), jnp.asarray(tri[:, 0]),
                                              jnp.asarray(tri[:, 1]), jnp.asarray(tri[:, 2]))
                pts.append(np.asarray(pp)[:n])
                d2.append(np.asarray(dd)[:n])
            pts = np.concatenate(pts)
            d2 = np.concatenate(d2)
            # segment argmin per query (ties -> lowest face id, the lexsort order)
            starts = np.r_[0, np.nonzero(qi[1:] != qi[:-1])[0] + 1]
            seg_min = np.minimum.reduceat(d2, starts)
            seg_id = np.repeat(np.arange(starts.shape[0]), np.diff(np.r_[starts, qi.shape[0]]))
            is_min = d2 == seg_min[seg_id]
            first = np.full(starts.shape[0], -1, np.int64)
            cand = np.nonzero(is_min)[0]
            first_pos = np.unique(seg_id[cand], return_index=True)[1]
            first[np.unique(seg_id[cand])] = cand[first_pos]
            qs = qi[starts]
            out_p[qs] = pts[first]
            out_d[qs] = d2[first]
            out_f[qs] = fi[first]
        if return_stats:
            return out_p, out_f, out_d, {"queries": Q, "fallback": nfb, "K": K, "rmax": self.rmax}
        return out_p, out_f, out_d

    def query_ext(self, P, chunk=1 << 16, return_stats=False):
        """Same outputs as query() with k_ext KD candidates per query instead of K, so far fewer ball queries.
        A row that passes the K-candidate certificate keeps query()'s fast-path answer (first minimum, in KD order,
        among the first K candidates). Any other row ends in query()'s fallback, which returns the lowest face id
        among the faces at the global minimum d2; if the k_ext certificate passes, all of those faces are among the
        k_ext candidates, so the same answer is taken from them. The rest run the same ball query (its radius only
        has to contain the minimum faces). cKDTree results do not depend on workers."""
        P = np.ascontiguousarray(P, np.float32)
        Q = P.shape[0]
        out_p = np.zeros((Q, 3), np.float32)
        out_d = np.zeros((Q,), np.float32)
        out_f = np.zeros((Q,), np.int64)
        K, KE, w = self.K, self.k_ext, self.workers
        ball = np.zeros((Q,), bool)
        n_ext = 0
        for s in range(0, Q, chunk):
            part = P[s:s + chunk]
            n = part.shape[0]
            dk, ik = self.tree.query(part.astype(np.float64), k=KE, workers=w)
            Bp = 1
            while Bp < n:
                Bp *= 2
            pts, d2 = _all_candidates(jnp.asarray(_pad_rows(part, Bp)), jnp.asarray(_pad_rows(self.T[ik], Bp)))
            pts, d2 = np.asarray(pts)[:n], np.asarray(d2)[:n]
            r = np.arange(n)
            k1 = np.argmin(d2[:, :K], axis=1)
            bd = d2[r, k1]
            sel_p, sel_d, sel_f = pts[r, k1], bd, ik[r, k1]
            need = (np.sqrt(bd.astype(np.float64)) > dk[:, K - 1] - self.rmax - 1e-7) & (K < len(self.F))
            if need.any():
                n_ext += int(need.sum())
                be = d2.min(axis=1)
                ke = np.argmin(np.where(d2 == be[:, None], ik, np.iinfo(np.int64).max), axis=1)
                pe = (np.sqrt(be.astype(np.float64)) <= dk[:, -1] - self.rmax - 1e-7) | (KE >= len(self.F))
                m = need & pe
                sel_p[m], sel_d[m], sel_f[m] = pts[r, ke][m], be[m], ik[r, ke][m]
                mb = need & ~pe
                sel_d[mb] = be[mb]
                ball[s:s + n] = mb
            out_p[s:s + n], out_d[s:s + n], out_f[s:s + n] = sel_p, sel_d, sel_f
        nfb = int(ball.sum())
        if nfb:
            idx = np.nonzero(ball)[0]
            best = np.sqrt(out_d[idx].astype(np.float64))
            lists = self.tree.query_ball_point(P[idx].astype(np.float64), best + self.rmax + 1e-6, workers=w)
            qi = np.concatenate([np.full(len(l), i, np.int64) for i, l in zip(idx, lists)])
            fi = np.concatenate([np.asarray(l, np.int64) for l in lists])
            order = np.lexsort((fi, qi))
            qi, fi = qi[order], fi[order]
            pts, d2 = [], []
            step = 1 << 20
            for s in range(0, qi.shape[0], step):
                q, f = qi[s:s + step], fi[s:s + step]
                n = q.shape[0]
                Bp = 1
                while Bp < n:
                    Bp *= 2
                tri = _pad_rows(self.T[f], Bp)
                pp, dd = closest_on_triangles(jnp.asarray(_pad_rows(P[q], Bp)), jnp.asarray(tri[:, 0]),
                                              jnp.asarray(tri[:, 1]), jnp.asarray(tri[:, 2]))
                pts.append(np.asarray(pp)[:n])
                d2.append(np.asarray(dd)[:n])
            pts = np.concatenate(pts)
            d2 = np.concatenate(d2)
            starts = np.r_[0, np.nonzero(qi[1:] != qi[:-1])[0] + 1]
            seg_min = np.minimum.reduceat(d2, starts)
            seg_id = np.repeat(np.arange(starts.shape[0]), np.diff(np.r_[starts, qi.shape[0]]))
            is_min = d2 == seg_min[seg_id]
            first = np.full(starts.shape[0], -1, np.int64)
            cand = np.nonzero(is_min)[0]
            first_pos = np.unique(seg_id[cand], return_index=True)[1]
            first[np.unique(seg_id[cand])] = cand[first_pos]
            qs = qi[starts]
            out_p[qs] = pts[first]
            out_d[qs] = d2[first]
            out_f[qs] = fi[first]
        if return_stats:
            return out_p, out_f, out_d, {"queries": Q, "fallback": nfb, "fallback_k": n_ext, "K": K, "k_ext": KE,
                                         "rmax": self.rmax}
        return out_p, out_f, out_d


@jax.jit
def _all_candidates(p, tris):
    """p [Q,3], tris [Q,K,3,3] -> (closest point per candidate [Q,K,3], squared distance [Q,K])."""
    return closest_on_triangles(p[:, None, :], tris[:, :, 0], tris[:, :, 1], tris[:, :, 2])
