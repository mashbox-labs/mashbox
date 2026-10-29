"""Routed MoE FFN for every row of one forward call (decode and prefill alike, inside the one continuous-batching step).

  route -> (row, expert) pairs -> sorted by expert -> ops.expert_ffn (each picked expert's weights fetched once per
  call; only the picked experts are computed) -> inverse-permutation gather -> sum over k in top-k order (llama.cpp's
  order: e0 + e1 + ... ) -> + the shared expert (optionally sigmoid-gated).

Pairs: P = N * K is static. A row that is not valid (padding, past its length) gets the sentinel expert id E: its pairs
sort to the end, belong to no expert group and are never computed; its output row is 0. No capacity factor and no
dropped token (dropless).

W (one layer): gi router weight (E, D) float32; optional gi_b router bias (E,), sel_b selection-only bias (E,);
experts as ops handles stacked (E, rows, cols): ge / ue / de (or gue = fused gate|up); optional expert biases
ge_b / ue_b / de_b / gue_b (E, rows) float32; shared expert sg / su / sd handles and gs (D,) its sigmoid gate weight.
Called with layer (moe_ffn), the expert entries are the whole stacks over layer groups (groups, E, ...) instead.
"""
import jax
import jax.numpy as jnp

from .. import ops
from .route import route

f32 = jnp.float32


def sort_pairs(ids, wts, valid, E):
    """ids/wts (N, K), valid (N,) bool -> (e_sorted (P,), row_of_pair (P,), w_sorted (P,), inv (P,)): pairs sorted by
    expert (stable: rows ascending inside an expert), invalid rows' pairs carry the sentinel E and weight 0; inv maps
    pair p = n * K + k to its sorted position."""
    N, K = ids.shape
    P = N * K
    e = jnp.where(valid[:, None], ids, E).reshape(P).astype(jnp.int32)
    order = jnp.argsort(e, stable=True).astype(jnp.int32)
    e_s = e[order]
    w_s = jnp.where(e_s < E, wts.reshape(P)[order], 0.0).astype(f32)
    inv = jnp.zeros((P,), jnp.int32).at[order].set(jnp.arange(P, dtype=jnp.int32))
    return e_s, order // K, w_s, inv


def group_sizes(e_sorted, E):
    """rows per expert (E,) int32; sentinel rows are not counted"""
    return jnp.zeros((E,), jnp.int32).at[e_sorted].add(1, mode="drop")


def combine(y_sorted, inv, N, K):
    """weighted expert outputs in sorted order (P, D) -> (N, D): gather back to (row, k) and add k = 0, 1, ... in order"""
    yp = y_sorted[inv].reshape(N, K, y_sorted.shape[-1])
    out = yp[:, 0]
    for k in range(1, K):
        out = out + yp[:, k]
    return out


def shared_ffn(h, W, act_dtype):
    """the shared expert (dense SwiGLU through ops.qmatmul) times its sigmoid gate when the layer has one -> float32"""
    q = ops.qmatmul
    a = (jax.nn.silu(q(h, W["sg"]).astype(f32)) * q(h, W["su"]).astype(f32)).astype(act_dtype)
    y = q(a, W["sd"]).astype(f32)
    if "gs" in W:
        y = y * jax.nn.sigmoid(jnp.sum(h.astype(f32) * W["gs"].astype(f32), -1, keepdims=True))
    return y


EXPERT_KEYS = ("ge", "ue", "de", "gue", "ge_b", "ue_b", "de_b", "gue_b")
OPS_KEYS = dict(ge="gate", ue="up", de="down", gue="gate_up", ge_b="gate_b", ue_b="up_b", de_b="down_b",
                gue_b="gate_up_b")


def moe_ffn(h, spec, W, valid=None, stats=None, layer=None):
    """h (N, D) in the activation dtype, valid (N,) bool or None (all rows) -> y (N, D) float32: routed experts (+ the
    shared expert when W has one). stats: a dict that receives the group sizes (E,) of this call (tests, stats pass).
    layer: None, or a layer-group index (int32 scalar, e.g. the arch's lax.scan group index) when W's expert entries
    (EXPERT_KEYS) are the whole stacks over layer groups (groups, E, ...); router and shared-expert entries stay one
    layer's. The routed op then reads that group's experts in place: a scan slice of the stacks would be copied into
    the expert kernel's call at every layer."""
    N, D = h.shape
    E, K = int(spec.n_expert), int(spec.n_used)
    valid = jnp.ones((N,), jnp.bool_) if valid is None else valid.reshape(N)
    ids, wts = route(h, W["gi"], spec, router_bias=W.get("gi_b"), sel_bias=W.get("sel_b"))
    e_s, row, w_s, inv = sort_pairs(ids, wts, valid, E)
    x_s = jnp.where((e_s < E)[:, None], h[row], jnp.zeros((), h.dtype))
    handles = {OPS_KEYS[k]: W[k] for k in EXPERT_KEYS if k in W}
    y_s = ops.expert_ffn(x_s, e_s, w_s, handles, act=getattr(spec, "act", "silu"), n_expert=E, layer=layer)
    if stats is not None:
        stats["group_sizes"] = group_sizes(e_s, E)
    y = combine(y_s, inv, N, K)
    if "su" in W:
        y = y + shared_ffn(h, W, h.dtype)
    return jnp.where(valid[:, None], y, 0.0)
