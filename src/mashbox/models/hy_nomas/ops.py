"""The operations that have a compiled TPU kernel in this model's bundle, each with a plain JAX version.

The model code calls these through an ``Ops`` object. When the kernel bundle matches the chip and JAX version and holds
a program for the exact shape asked for, the compiled kernel runs; otherwise the plain JAX version runs. Both give the
same outputs within bf16 rounding. Which one ran is recorded per op and shape in ``Ops.used``.

Layouts are the ones the main model uses on each chip: queries and keys as [rows, heads * 128], text padded to a fixed
length, padded rows filled with zeros. Attention visibility uses group ids (see buckets.py): key j is visible to query i
when ``gk[j] < 0`` or ``gk[j] == gq[i]``, and keys at or past ``nvalid`` are never visible.
"""
import math

import jax
import jax.numpy as jnp

F32, BF = jnp.float32, jnp.bfloat16
LOG2E = 1.4426950408889634
CHECK_LIMIT = 96.0          # above this check value the bundle's exact long attention program runs instead of the fast one


# ------------------------------------------------------------------ plain JAX versions
def qk_norm_rope(x, w, cos, sin, *, H, scale=None, out_rows=None, eps=1e-6):
    """Per head: RMSNorm over 128 lanes (times the norm weight), interleaved-pair rotary embedding, optional scale.

    x [L, H*D] (bf16 linear output), w [D], cos/sin [L, D] f32 -> [out_rows or L, H*D] bf16. Math in f32, one bf16
    rounding at the end. Rows L..out_rows are zeros (the caller fills them).
    """
    L, HD = x.shape
    D = HD // H
    x32 = x.astype(F32).reshape(L, H, D)
    y = x32 * jax.lax.rsqrt((x32 * x32).mean(-1, keepdims=True) + eps) * w.astype(F32)
    yp = y.reshape(L, H, D // 2, 2)
    rot = jnp.stack([-yp[..., 1], yp[..., 0]], -1).reshape(L, H, D)
    o = y * cos.astype(F32)[:, None, :] + rot * sin.astype(F32)[:, None, :]
    if scale is not None:
        o = o * scale
    o = o.reshape(L, HD).astype(BF)
    if out_rows is not None and out_rows > L:
        o = jnp.concatenate([o, jnp.zeros((out_rows - L, HD), BF)])
    return o


def masked_attention(q, k, v, gq, gk, *, H, nvalid, bq, bkv):
    """Exact softmax attention in base 2 (q is already scaled by log2(e)/sqrt(D)), keys masked by group ids and nvalid.

    q [LQ, H*D], k/v [LK, H*D] bf16, gq [LQ, 1], gk [1, LK] int32 -> [LQ, H*D] bf16. Runs one query block at a time and
    walks the keys in blocks with a running max, so memory stays at one [H, bq, bkv] score block.
    """
    LQ, HD = q.shape
    D = HD // H
    LK = k.shape[0]
    nq, nk = LQ // bq, LK // bkv
    qb = q.reshape(nq, bq, H, D)
    gqb = gq.reshape(nq, bq)
    kb = k.reshape(nk, bkv, H, D)
    vb = v.reshape(nk, bkv, H, D)
    gkb = gk.reshape(nk, bkv)
    col = jnp.arange(LK, dtype=jnp.int32).reshape(nk, bkv)

    def one_block(args):
        qi, gqi = args

        def step(carry, xs):
            m, l, acc = carry
            ki, vi, gki, ci = xs
            s = jnp.einsum('qhd,khd->hqk', qi, ki, preferred_element_type=F32)
            vis = (ci < nvalid)[None, :] & ((gki < 0)[None, :] | (gki[None, :] == gqi[:, None]))
            s = jnp.where(vis[None], s, -jnp.inf)
            m_new = jnp.maximum(m, s.max(-1))
            m_use = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
            p = jnp.exp2(s - m_use[..., None])
            a = jnp.where(jnp.isfinite(m), jnp.exp2(m - m_use), 0.0)
            l = a * l + p.sum(-1)
            acc = a[..., None] * acc + jnp.einsum('hqk,khd->hqd', p.astype(BF), vi, preferred_element_type=F32)
            return (m_new, l, acc), None
        init = (jnp.full((H, bq), -jnp.inf, F32), jnp.zeros((H, bq), F32), jnp.zeros((H, bq, D), F32))
        (m, l, acc), _ = jax.lax.scan(step, init, (kb, vb, gkb, col))
        o = acc / jnp.where(l > 0, l, 1.0)[..., None]
        return o.transpose(1, 0, 2).reshape(bq, HD).astype(BF)
    return jax.lax.map(one_block, (qb, gqb)).reshape(LQ, HD)


def dec_qk_norm_rope(x, wt, wh, cos, sin, *, H, eps=1e-6):
    """Pixel decoder q/k prep: per head, RMSNorm of lanes [0, 64) and [64, 128) with their own weights, then the
    multi-axis rotary embedding (sections 64, 32, 32, each rotated by half). x [N, H*128] f32, cos/sin [N, 128] f32
    -> [N, H*128] bf16 (math in f32, one rounding)."""
    N, HD = x.shape
    x32 = x.astype(F32).reshape(N, H, 128)
    t, s = x32[..., :64], x32[..., 64:]
    t = t * jax.lax.rsqrt((t * t).mean(-1, keepdims=True) + eps) * wt.astype(F32)
    s = s * jax.lax.rsqrt((s * s).mean(-1, keepdims=True) + eps) * wh.astype(F32)
    y = jnp.concatenate([t, s], -1)
    parts, off = [], 0
    for n in (64, 32, 32):
        p = y[..., off:off + n]
        parts.append(jnp.concatenate([-p[..., n // 2:], p[..., :n // 2]], -1))
        off += n
    rot = jnp.concatenate(parts, -1)
    o = y * cos.astype(F32)[:, None, :] + rot * sin.astype(F32)[:, None, :]
    return o.reshape(N, HD).astype(BF)


def dec_local_attention(q, k, v, lay, *, heads, kv_heads, head_dim):
    """Pixel decoder local attention: each tile of queries sees the keys of its 3 x 3 tile neighbourhood (lay from
    decoder.layout). q [N, heads, 128], k/v [N, kv_heads, 128] -> [1, N, heads*128]. f32 math at highest precision."""
    hi = jax.lax.Precision.HIGHEST
    n = q.shape[0]
    rep = heads // kv_heads
    scale = head_dim ** -0.5

    def one(args):
        qg, ki, kv = args
        qs = q[qg].astype(F32).reshape(qg.shape[0], kv_heads, rep, head_dim)
        ks = k[ki].astype(F32)
        vs = v[ki].astype(F32)
        s = jnp.einsum('qhrd,khd->hrqk', qs, ks, precision=hi) * scale
        s = jnp.where(kv[None, None, None, :], s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1)
        o = jnp.einsum('hrqk,khd->qhrd', p, vs, precision=hi)
        return o.reshape(qg.shape[0], heads, head_dim).astype(v.dtype)
    outs = jax.lax.map(one, (lay['qg'], lay['ki'], lay['kv']))
    out = jnp.zeros((n, heads, head_dim), v.dtype)
    out = out.at[lay['qi'].reshape(-1)].set(outs.reshape(-1, heads, head_dim), mode='drop')
    return out.reshape(1, n, heads * head_dim)


# ------------------------------------------------------------------ dispatch
class Ops:
    """Kernel-or-plain dispatch. ``bundle`` is a kernels.Bundle (or None for plain JAX everywhere)."""

    def __init__(self, bundle=None):
        self.bundle = bundle
        self.used = {}

    def _get(self, op, key):
        fn = self.bundle.get(op, key) if self.bundle is not None else None
        self.used[op + ' ' + key] = 'kernel' if fn is not None else 'plain'
        return fn

    def summary(self):
        k = sorted(n for n, how in self.used.items() if how == 'kernel')
        p = sorted(n for n, how in self.used.items() if how == 'plain')
        return dict(kernel=k, plain=p)

    # main model (per chip, inside the tensor-parallel map)
    def dit_qk_norm_rope(self, x, w, cos, sin, *, H, scale=None, out_rows):
        which = 'q' if scale is not None else 'k'
        fn = self._get('dit_qk_norm_rope', '%s_r%d' % (which, out_rows))
        if fn is not None:
            return fn(x, w, cos, sin)
        return qk_norm_rope(x, w, cos, sin, H=H, scale=scale, out_rows=out_rows)

    def dit_attention(self, q, k, v, gq, gk, *, H, n_img, nvalid, bq, bkv):
        LQ, LK = q.shape[0], k.shape[0]
        key = 'q%d_k%d_n%d' % (LQ, LK, nvalid)
        prep = self._get('dit_v_prep', 'r%d' % LK)
        check = self._get('dit_attention_masked_pad', key)
        fast = self._get('dit_attention_masked', key)
        long_ = self._get('dit_attention_masked_long', key)
        if None in (prep, check, fast, long_):
            for op in ('dit_v_prep r%d' % LK, 'dit_attention_masked_pad ' + key, 'dit_attention_masked ' + key,
                       'dit_attention_masked_long ' + key):
                self.used[op] = 'plain'
            return masked_attention(q, k, v, gq, gk, H=H, nvalid=nvalid, bq=bq, bkv=bkv)
        v1 = prep(v)
        kb, g = check(q, k)
        return jax.lax.cond(g > CHECK_LIMIT, lambda: long_(q, k, v1, gq, gk), lambda: fast(q, k, v1, kb, gq, gk))

    # pixel decoder (one chip per call)
    def dec_qk_norm_rope(self, x, wt, wh, cos, sin, *, H, which, frames):
        fn = self._get('dec_qk_norm_rope', '%s_t%d' % (which, frames))
        if fn is not None:
            return fn(x, wt, wh, cos, sin)
        return dec_qk_norm_rope(x, wt, wh, cos, sin, H=H)

    def dec_local_attention(self, q, k, v, lay, *, frames, heads, kv_heads, head_dim):
        fn = self._get('dec_local_attention', 't%d' % frames)
        if fn is not None:
            return fn(q.astype(BF), k.astype(BF), v.astype(BF))
        return dec_local_attention(q, k, v, lay, heads=heads, kv_heads=kv_heads, head_dim=head_dim)


def q_scale(head_dim):
    """Scale folded into the queries: log2(e) / sqrt(head_dim) (attention runs in base 2)."""
    return LOG2E / math.sqrt(head_dim)
