"""Per-row sampling for a batch of logits.

Order: repetition penalty (prompt + generated tokens; positive logits divided, negative multiplied) -> frequency penalty
(minus f x count of generated tokens) -> presence penalty -> temperature -> top_k -> top_p -> min_p -> one draw.
temperature 0 = greedy on the processed logits. Ties at a top_k or top_p cut are kept.
Random numbers: threefry, keyed by (seed, token index), so a row's tokens depend only on its own seed, prompt and logits,
never on the other rows of the batch.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

f32, i32 = jnp.float32, jnp.int32
FIELDS = ("temperature", "top_p", "min_p", "rep", "freq", "pres")


def pack(rows, S):
    """rows: list of dicts (None = greedy) -> host arrays for S rows."""
    out = {k: np.zeros(S, np.float32) for k in FIELDS}
    out["top_p"][:] = 1.0
    out["rep"][:] = 1.0
    out["top_k"] = np.zeros(S, np.int32)
    seed = np.zeros((S, 2), np.uint32)
    for i, r in enumerate(rows[:S]):
        r = r or {}
        out["temperature"][i] = float(r.get("temperature") or 0.0)
        out["top_k"][i] = int(r.get("top_k") or 0)
        out["top_p"][i] = float(1.0 if r.get("top_p") is None else r["top_p"])
        out["min_p"][i] = float(r.get("min_p") or 0.0)
        out["rep"][i] = float(1.0 if r.get("repetition_penalty") is None else r["repetition_penalty"])
        out["freq"][i] = float(r.get("frequency_penalty") or 0.0)
        out["pres"][i] = float(r.get("presence_penalty") or 0.0)
        s = int(r.get("seed") or 0) & ((1 << 64) - 1)
        seed[i] = (s >> 32, s & 0xFFFFFFFF)
    out["seed"] = seed
    return out


def to_device(sp):
    return {k: jnp.asarray(v) for k, v in sp.items()}


def no_penalty(sp):
    return (sp["rep"] == 1.0) & (sp["freq"] == 0.0) & (sp["pres"] == 0.0)


def penalise(lg, cnt, seen, sp):
    """lg (B, V) f32 logits, cnt (B, V) f32 counts of generated tokens, seen (B, V) bool prompt + generated tokens."""
    rep = sp["rep"][:, None]
    x = jnp.where(seen, jnp.where(lg > 0, lg / rep, lg * rep), lg)
    x = x - sp["freq"][:, None] * cnt
    return x - sp["pres"][:, None] * (cnt > 0).astype(f32)


def _key(x):
    """f32 -> int32 with the same order as the float values."""
    b = lax.bitcast_convert_type(x, i32)
    return jnp.where(b < 0, b ^ jnp.int32(0x7FFFFFFF), b)


def _mid_lo(lo, hi):
    return (lo >> 1) + (hi >> 1) + (lo & hi & 1)


def _mid_hi(lo, hi):
    return (lo >> 1) + (hi >> 1) + ((lo | hi) & 1)


def _top_k_cut(key, k):
    """Largest t with count(key >= t) >= k, per row (bisection on the ordered keys)."""
    def body(_, c):
        lo, hi = c
        mid = _mid_hi(lo, hi)
        ok = jnp.sum((key >= mid[:, None]).astype(i32), -1) >= k
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid - 1)
    lo, _ = lax.fori_loop(0, 33, body, (jnp.min(key, -1), jnp.max(key, -1)))
    return lo


def _top_p_cut(key, p, top_p):
    """Smallest t with mass(key > t) < top_p, per row. Keeping key >= t keeps the smallest set of top tokens whose
    probability mass reaches top_p."""
    def body(_, c):
        lo, hi = c
        mid = _mid_lo(lo, hi)
        ok = jnp.sum(jnp.where(key > mid[:, None], p, 0.0), -1) < top_p
        return jnp.where(ok, lo, mid + 1), jnp.where(ok, mid, hi)
    _, hi = lax.fori_loop(0, 33, body, (jnp.min(key, -1), jnp.max(key, -1)))
    return hi


def sample(x, sp, step, greedy_tok=None):
    """x (B, V) penalised logits f32, step (B,) int32 token index per row -> (B,) int32 tokens."""
    greedy = sp["temperature"] <= 0.0
    if greedy_tok is None:
        greedy_tok = jnp.argmax(x, -1).astype(i32)
    x = transform(x, sp)

    def draw(row, seed, st):
        key = jax.random.fold_in(jax.random.wrap_key_data(seed, impl="threefry2x32"), st)
        return jnp.argmax(row + jax.random.gumbel(key, row.shape, f32)).astype(i32)
    return jnp.where(greedy, greedy_tok, jax.vmap(draw)(x, sp["seed"], step))


def transform(x, sp):
    """temperature -> top_k -> top_p -> min_p on (B, V) penalised logits: the logits one draw is taken from (cut tokens
    -inf). Speculative decoding builds both its draft and target distributions with it."""
    t = sp["temperature"]
    greedy = t <= 0.0
    x = x / jnp.where(greedy, 1.0, t)[:, None]
    k = sp["top_k"]
    x = lax.cond(jnp.any(k > 0),
                 lambda x: jnp.where((k > 0)[:, None] & (_key(x) < _top_k_cut(_key(x), k)[:, None]), -jnp.inf, x),
                 lambda x: x, x)
    tp = sp["top_p"]

    def cut_p(x):
        key = _key(x)
        return jnp.where((tp < 1.0)[:, None] & (key < _top_p_cut(key, jax.nn.softmax(x, -1), tp)[:, None]), -jnp.inf, x)
    x = lax.cond(jnp.any(tp < 1.0), cut_p, lambda x: x, x)
    mp = sp["min_p"]

    def cut_min(x):
        pm = jax.nn.softmax(x, -1)
        return jnp.where((mp > 0.0)[:, None] & (pm < mp[:, None] * jnp.max(pm, -1, keepdims=True)), -jnp.inf, x)
    return lax.cond(jnp.any(mp > 0.0), cut_min, lambda x: x, x)
