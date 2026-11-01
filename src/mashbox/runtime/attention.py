"""The full-attention core every model the engine runs calls (models/qwen3_8_27b, gguf/archs qwen35, dense, gemma4): scores,
masked softmax, weighted sum, for q (B, T, KV, G, HD) against the call's keys kk / values vv (B, S, KV, HD) under mask
(B, T, S) -> (B, T, KV, G, HD) in `act`.

Two forms, picked from static shapes:
  - dense (every call with S <= BLOCK_FROM keys, and every call with fewer than BLOCK_MIN_T query tokens: decode and
    speculative verify): the whole (T, S) score matrix at once. This is the code the models ran before this module,
    unchanged, so those programs are bit-identical to it.
  - blocked (a prefill chunk against a long window): keys in fixed blocks of KV_BLOCK positions in absolute order with
    an online softmax (running max and sum in float32), so a chunk's scratch is (T, KV_BLOCK) whatever the window.
    Without it the engine had to shrink the prefill chunk as the window grew (engine.prefill: down to 16 tokens past
    64k), and every call then streamed all the weights for 16 tokens: Qwen3.8-27B BQ8_R on one TPU v6e chip prefilled
    5,173 tokens/s at a 4k prompt and 537 tokens/s at 64k, measured.
    Not bit-identical to dense (the sum is taken block by block; the probabilities are rounded to `act` before the
    normalisation instead of after): op-tolerance, checked against dense in tests/cpu/test_attention_blocked.py.

attend_cache (the form every call site uses): the same core against the K/V cache itself plus the call's own keys. A
window of at most BLOCK_FROM cached keys runs exactly as before (read the window, append the call's keys, attend: the
same program, bit for bit). A longer window runs as a loop over key blocks read straight from the cache (DEC_BLOCK keys
per block for decode and verify, KV_BLOCK for a prefill chunk), with the call's own keys as one last block, under the
same online softmax; the loop runs only over the blocks below `hi` (the highest cached position any live row reads,
traced), not over the whole static window, and the window is never concatenated or copied whole. Before it, decode at a
long window read the whole window, copied it into the concatenation with the new key, and attended over it densely:
Qwen3.8-27B BQ8_R on one TPU v6e chip decoded 40.6 tokens/s at a 4k context and 23.1 at 64k, measured.

width(Tc, T) is the score width the engine budgets a prefill call for (Engine attn_bytes), so the chunk it picks and the
form the model runs agree."""
import math

import jax
import jax.numpy as jnp
from jax import lax

f32 = jnp.float32
NEG = -1e30
BLOCK_FROM = 4096        # keys above which a prefill chunk runs blocked
BLOCK_MIN_T = 16         # query tokens from which a call can run blocked (decode and verify stay dense)
KV_BLOCK = 2048          # keys per block
DEC_BLOCK = 8192         # keys per block of a decode / verify call (T < BLOCK_MIN_T) in attend_cache
DEC_FORM = "flat"        # the score / weighted-sum dots of such a call: "flat" (block-diagonal q, below) or "heads"


def blocked(T, S):
    return T >= BLOCK_MIN_T and S > BLOCK_FROM


def width(Tc, T):
    """score columns the engine budgets one row of a prefill call for: Tc cached + T new keys; where the call runs blocked,
    the larger of one block and the widest call that still runs dense (the engine picks a chunk from a group's first
    window estimate, and a call whose rows sit at lower positions runs at a smaller window, which can be dense)"""
    return max(KV_BLOCK, min(Tc + T, BLOCK_FROM)) if blocked(T, Tc + T) else Tc + T


def attend(q, kk, vv, mask, scale, act):
    """scale None = no scaling of the scores (gemma3's q is scaled before the product)"""
    T, S = q.shape[1], kk.shape[1]
    if not blocked(T, S):
        s = jnp.einsum("btkgd,bskd->bkgts", q, kk, preferred_element_type=f32)
        if scale is not None:
            s = s * scale
        p = jax.nn.softmax(jnp.where(mask[:, None, None], s, NEG), -1).astype(act)
        return jnp.einsum("bkgts,bskd->btkgd", p, vv, preferred_element_type=f32).astype(act)
    B, _, KV, G, HD = q.shape
    nb = -(-S // KV_BLOCK)
    pad = nb * KV_BLOCK - S
    if pad:
        kk = jnp.pad(kk, ((0, 0), (0, pad), (0, 0), (0, 0)))
        vv = jnp.pad(vv, ((0, 0), (0, pad), (0, 0), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, 0), (0, pad)))

    def body(i, carry):
        m, l, o = carry
        at = i * KV_BLOCK
        kb = lax.dynamic_slice_in_dim(kk, at, KV_BLOCK, 1)
        vb = lax.dynamic_slice_in_dim(vv, at, KV_BLOCK, 1)
        mb = lax.dynamic_slice_in_dim(mask, at, KV_BLOCK, 2)[:, None, None]       # (B, 1, 1, T, blk)
        s = jnp.einsum("btkgd,bskd->bkgts", q, kb, preferred_element_type=f32)
        if scale is not None:
            s = s * scale
        s = jnp.where(mb, s, NEG)
        m2 = jnp.maximum(m, s.max(-1))
        c = jnp.exp(m - m2)
        p = jnp.where(mb, jnp.exp(s - m2[..., None]), 0.0)
        l2 = l * c + p.sum(-1)
        o2 = o * c[..., None] + jnp.einsum("bkgts,bskd->bkgtd", p.astype(act), vb, preferred_element_type=f32)
        return m2, l2, o2

    m0 = jnp.full((B, KV, G, T), NEG, f32)
    l0 = jnp.zeros((B, KV, G, T), f32)
    o0 = jnp.zeros((B, KV, G, T, HD), f32)
    _, l, o = lax.fori_loop(0, nb, body, (m0, l0, o0))
    o = o / jnp.maximum(l, jnp.finfo(f32).tiny)[..., None]
    return jnp.transpose(o, (0, 3, 1, 2, 4)).astype(act)


def cache_hi(p0, Tc, keep=None):
    """the number of cached positions any row of a call reads (its cache mask is position < p0): max p0 over the rows,
    rows that are not live (keep False: padding rows of a decode bucket sit at the context end) left out, capped at Tc"""
    p = p0.astype(jnp.int32)
    if keep is not None:
        p = jnp.where(keep, p, 0)
    return jnp.minimum(jnp.max(p), Tc)


def attend_cache(q, read, Tc, k, v, mask, scale, act, hi=None):
    """q (B, T, KV, G, HD) against Tc cached positions and the call's own keys k / v (B, T, KV, HD) under mask
    (B, T, Tc + T) -> (B, T, KV, G, HD) in `act`. read(start, n, mult) -> the cached K, V (B, n, KV, HD) at positions
    start .. start + n - 1, start a multiple of the static int mult (runtime/kvcache.reader). hi (traced int or None =
    Tc): cached positions from hi on are masked for every row that matters, so their blocks are not read."""
    if Tc <= BLOCK_FROM:
        if Tc:
            kc, vc = read(0, Tc, Tc)
            if isinstance(kc, tuple):           # int8 cache (runtime/kvcache.pack): a short window is dequantized whole
                kc, vc = [(c.astype(f32) * sc[..., None]).astype(act) for c, sc in (kc, vc)]
            kk, vv = jnp.concatenate([kc, k], 1), jnp.concatenate([vc, v], 1)
        else:
            kk, vv = k, v
        return attend(q, kk, vv, mask, scale, act)
    B, T, KV, G, HD = q.shape
    blk = min(Tc, DEC_BLOCK if T < BLOCK_MIN_T else KV_BLOCK)
    hi = Tc if hi is None else hi
    nb = jnp.minimum((jnp.asarray(hi, jnp.int32) + blk - 1) // blk, -(-Tc // blk))
    jb = jnp.arange(blk, dtype=jnp.int32)

    def per_key(sc):                            # (B, n, KV) scales -> (B, KV, 1, 1, n), against the scores' axes
        return jnp.transpose(sc, (0, 2, 1))[:, :, None, None, :]

    flat = T < BLOCK_MIN_T and DEC_FORM == "flat"
    if flat:        # q as one block-diagonal (KV x HD, KV x G x T) matrix: the block's (n, KV x HD) rows go into one plain
        # dot in the cache's own layout (no per-head batch axis, so no relayout of the block); the zero blocks add exact
        # zeros. Decode only: it does KV times the score FLOPs, nothing at T < 16.
        qbd = jnp.einsum("btkgd,kj->bkdjgt", q, jnp.eye(KV, dtype=q.dtype)).reshape(B, KV * HD, KV * G * T)

    def qk(kb):
        if flat:
            s = jnp.einsum("bnf,bfm->bmn", kb.reshape(B, -1, KV * HD).astype(act), qbd, preferred_element_type=f32)
            return s.reshape(B, KV, G, T, -1)
        return jnp.einsum("btkgd,bskd->bkgts", q, kb.astype(act), preferred_element_type=f32)

    def pvdot(pv, vb):
        if flat:
            full = jnp.einsum("bmn,bnf->bmf", pv.reshape(B, KV * G * T, -1), vb.reshape(B, -1, KV * HD).astype(act),
                              preferred_element_type=f32).reshape(B, KV, G, T, KV, HD)
            return jnp.stack([full[:, j, :, :, j] for j in range(KV)], 1)
        return jnp.einsum("bkgts,bskd->bkgtd", pv, vb.astype(act), preferred_element_type=f32)

    def update(carry, s, mb, vb, dot):
        m, l, o = carry
        if scale is not None:
            s = s * scale
        s = jnp.where(mb, s, NEG)
        m2 = jnp.maximum(m, s.max(-1))
        c = jnp.exp(m - m2)
        p = jnp.where(mb, jnp.exp(s - m2[..., None]), 0.0)
        l2 = l * c + p.sum(-1)
        if isinstance(vb, tuple):               # int8 V: the per-key scale goes on the weights, the codes into the dot
            vb, sv = vb
            pv = (p * per_key(sv)).astype(act)
        else:
            pv = p.astype(act)
        return m2, l2, o * c[..., None] + dot(pv, vb)

    def body(i, carry):
        at = i * blk
        st = jnp.minimum(at, Tc - blk)          # the last block of a window that is not a multiple of blk starts earlier;
        kb, vb = read(st, blk, math.gcd(Tc, blk))   # its positions below `at` were read by the block before and are masked;
        # every start is 0, a multiple of blk or Tc - blk, so a multiple of gcd(Tc, blk) (an int8 cache reads whole tiles then)
        mb = lax.dynamic_slice_in_dim(mask, st, blk, 2) & (st + jb >= at)[None, None]
        if isinstance(kb, tuple):               # int8 K: dot with the codes, then the per-key scale
            s = qk(kb[0]) * per_key(kb[1])
        else:
            s = qk(kb)
        return update(carry, s, mb[:, None, None], vb, pvdot)

    m0 = jnp.full((B, KV, G, T), NEG, f32)
    l0 = jnp.zeros((B, KV, G, T), f32)
    o0 = jnp.zeros((B, KV, G, T, HD), f32)
    carry = lax.fori_loop(0, nb, body, (m0, l0, o0))
    s = jnp.einsum("btkgd,bskd->bkgts", q, k, preferred_element_type=f32)
    _, l, o = update(carry, s, mask[:, None, None, :, Tc:], v,
                     lambda pv, vb: jnp.einsum("bkgts,bskd->bkgtd", pv, vb, preferred_element_type=f32))
    o = o / jnp.maximum(l, jnp.finfo(f32).tiny)[..., None]
    return jnp.transpose(o, (0, 3, 1, 2, 4)).astype(act)
