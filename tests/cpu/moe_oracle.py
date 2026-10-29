"""Test oracles for the routed MoE path (tests only, never imported by the engine).

old_moe: the all-experts graph of qwen35.py on main 7245e88 (make_forward.moe), copied as it was: every expert runs on
every row with its routing weight (0 when not picked). route_llama_np: llama.cpp build_moe_ffn's router (a37cc46,
src/llama-graph.cpp) row by row in NumPy float32, with ggml's CPU soft_max (exp(x - max), the sum accumulated in
double, then times 1 / sum) and ggml_sum_rows (double accumulation) for norm_w.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from mashbox import ops

f32 = jnp.float32
HI = lax.Precision.HIGHEST


def old_moe(h, w, E, K, w_scale=0.0):
    lin_ = ops.qmatmul

    def ffn(h, g, u, d):
        a = (jax.nn.silu(lin_(h, g).astype(f32)) * lin_(h, u).astype(f32)).astype(h.dtype)
        return lin_(a, d)
    lg = lax.dot_general(h.astype(f32), w["gi"], (((h.ndim - 1,), (1,)), ((), ())), precision=HI,
                         preferred_element_type=f32)
    tv, ti = lax.top_k(jax.nn.softmax(lg, -1), K)
    tv = tv / jnp.maximum(jnp.sum(tv, -1, keepdims=True), np.float32(6.103515625e-5))
    if w_scale not in (0.0, 1.0):
        tv = tv * np.float32(w_scale)
    coef = jnp.sum(jax.nn.one_hot(ti, E, dtype=f32) * tv[..., None], -2)

    def one(acc, xs):
        g, u, d, cf = xs
        return acc + cf[..., None] * ffn(h, g, u, d).astype(f32), None
    acc, _ = lax.scan(one, jnp.zeros(h.shape, f32), (w["ge"], w["ue"], w["de"], jnp.moveaxis(coef, -1, 0)))
    if "su" not in w:
        return acc.astype(h.dtype)
    sg = jax.nn.sigmoid(jnp.sum(h.astype(f32) * w["gs"], -1, keepdims=True))
    return (acc + ffn(h, w["sg"], w["su"], w["sd"]).astype(f32) * sg).astype(h.dtype)


def _ggml_soft_max(x):
    x = x.astype(np.float32)
    e = np.exp(x - x.max()).astype(np.float32)
    s = float(np.sum(e.astype(np.float64)))
    return (e * np.float32(1.0 / s)).astype(np.float32)


def _top(x, k):
    """ggml_argsort_top_k: indices of the k largest, best first (ties: lower index first)"""
    return np.argsort(-x.astype(np.float64), kind="stable")[:k]


def route_llama_np(logits, spec, sel_bias=None):
    """logits (N, E) float32 (router bias already added) -> (ids (N, K), wts (N, K) float32)"""
    E, K = spec.n_expert, spec.n_used
    ids, wts = [], []
    for lg in np.asarray(logits, np.float32):
        if spec.gating == "softmax":
            probs = _ggml_soft_max(lg)
        elif spec.gating == "sigmoid":
            probs = (np.float32(1) / (np.float32(1) + np.exp(-lg))).astype(np.float32)
        else:
            probs = lg.copy()
        sel = probs if sel_bias is None else (probs + np.asarray(sel_bias, np.float32)).astype(np.float32)
        G = getattr(spec, "n_group", 1) or 1
        if G > 1:
            grp = sel.reshape(G, E // G)
            score = np.array([np.float32(np.float32(g[_top(g, 2)[0]]) + np.float32(g[_top(g, 2)[1]])) for g in grp])
            keep = _top(score, spec.n_group_used)
            m = np.full_like(grp, -np.inf)
            m[keep] = grp[keep]
            sel = m.reshape(E)
        sel_ids = _top(sel, K)
        w = probs[sel_ids].astype(np.float32)
        if spec.gating == "softmax_weight":
            w = _ggml_soft_max(w)
        if spec.norm_w:
            s = np.float32(max(float(np.sum(w.astype(np.float64))), 6.103515625e-05))
            w = (w / s).astype(np.float32)
        ws = getattr(spec, "w_scale", 1.0) or 1.0
        if ws != 1.0:
            w = (w * np.float32(ws)).astype(np.float32)
        ids.append(sel_ids)
        wts.append(w)
    return np.asarray(ids, np.int32), np.asarray(wts, np.float32)
