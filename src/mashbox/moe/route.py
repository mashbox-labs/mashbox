"""MoE router: which K experts each row uses and with what weight, in llama.cpp's order (src/llama-graph.cpp
build_moe_ffn, a37cc46): router logits (+ router bias) -> gating (softmax | sigmoid | raw logits for softmax_weight)
-> selection scores (+ the selection-only bias exp_probs_b) -> optional group routing (each group scored by the
sum of its top 2, the best n_group_used groups kept, the rest masked to -inf) -> top K by selection score -> weights read
from the UNBIASED gating output -> softmax over the K (softmax_weight) -> renormalise to sum 1 (norm_w; the sum clamped
at 6.103515625e-05, the smallest normal f16, as llama.cpp does) -> times w_scale.

Everything runs in float32 on the float32 router weight. `spec` is any object with the MoESpec fields used here:
n_expert, n_used, gating, norm_w, w_scale, n_group, n_group_used.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

f32 = jnp.float32
HI = lax.Precision.HIGHEST
NORM_CLAMP = np.float32(6.103515625e-05)
GATINGS = ("softmax", "sigmoid", "softmax_weight")


def router_logits(h, w_router, router_bias=None):
    """h (N, D) -> (N, E) float32 logits: h @ w_router.T at HIGHEST precision (+ router bias)"""
    lg = lax.dot_general(h.astype(f32), w_router.astype(f32), (((h.ndim - 1,), (1,)), ((), ())), precision=HI,
                         preferred_element_type=f32)
    return lg if router_bias is None else lg + router_bias.astype(f32)


def route(h_f32, w_router, spec, router_bias=None, sel_bias=None, logits=None):
    """h_f32 (N, D) (the router runs in float32 whatever h's dtype) -> (ids int32 (N, K), wts float32 (N, K)), ids in
    top-K order (best first). logits: precomputed router logits (N, E) instead of h @ w_router."""
    E, K = int(spec.n_expert), int(spec.n_used)
    gating = spec.gating
    if gating not in GATINGS:
        raise ValueError(f"mashbox: expert gating '{gating}' is not one of {GATINGS}")
    lg = router_logits(h_f32, w_router, router_bias) if logits is None else logits.astype(f32)
    if gating == "softmax":
        probs = jax.nn.softmax(lg, -1)
    elif gating == "sigmoid":
        probs = jax.nn.sigmoid(lg)
    else:
        probs = lg
    sel = probs if sel_bias is None else probs + sel_bias.astype(f32)
    G = int(getattr(spec, "n_group", 1) or 1)
    if G > 1:
        Gu = int(spec.n_group_used)
        if E % G:
            raise ValueError(f"mashbox: expert_group_count {G} does not divide expert_count {E}")
        sg = sel.reshape(sel.shape[:-1] + (G, E // G))
        score = jnp.sum(lax.top_k(sg, 2)[0], -1)                                  # (N, G): sum of each group's top 2
        _, gi = lax.top_k(score, Gu)
        keep = jnp.any(jax.nn.one_hot(gi, G, dtype=jnp.bool_), -2)                 # (N, G)
        sel = jnp.where(keep[..., None], sg, -jnp.inf).reshape(sel.shape)
    _, ids = lax.top_k(sel, K)
    wts = jnp.take_along_axis(probs, ids, -1)
    if gating == "softmax_weight":
        wts = jax.nn.softmax(wts, -1)
    if spec.norm_w:
        wts = wts / jnp.maximum(jnp.sum(wts, -1, keepdims=True), NORM_CLAMP)
    ws = float(getattr(spec, "w_scale", 1.0) or 1.0)
    if ws != 1.0:
        wts = wts * np.float32(ws)
    return ids.astype(jnp.int32), wts
