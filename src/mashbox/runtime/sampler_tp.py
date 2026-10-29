"""Sampling on vocabulary shards (tensor parallel): runtime/sampler.py for logits whose vocabulary axis is split over the
chips of one mesh axis, as the lm head leaves them under tensor parallel (models/qwen3_8_27b/shard.py).

Same settings, order and cuts as sampler.py: penalties -> temperature -> top_k -> top_p -> min_p -> one draw; ties at a
cut are kept; temperature 0 = greedy (the first index of the largest logit). A sampled position costs 2 collectives:

  1. One all-gather. Each chip sends its first argmax of the logits, its C largest temperature-scaled logits with their
     global token ids, and its partial exp-sum. From these every chip computes the same cut thresholds per row (candidates
     merged by value descending, then global id ascending). A threshold found this way is exact whenever the cut lands
     above every chip's C-th candidate; the cut itself is then applied to each chip's whole shard.
  2. The draw. Each chip takes the first argmax of (cut logits + noise) over its own shard; one all-gather of (value, id),
     then the first chip holding the largest value wins, which is the first index of the largest value over the whole row.

A row whose cut needs more than the candidates (top_k above C, or a top_p cut or the set a probability sum runs over that
reaches down to a chip's C-th candidate) runs the full-row cut instead: one more all-gather of the rows, then sampler.py's
own transform on them (lax.cond, so it costs nothing when no row needs it): 3 collectives.

Noise comes from the global token id explicitly: g(v) = Gumbel(threefry2x32(fold_in(seed key, token index), (0, v))).
This is the bit pattern jax.random.gumbel draws for a whole (V,) row with threefry's partitionable mode, computed from v
itself, so a draw never depends on that flag, on the chip count or on where a shard starts. The probability sums run
in another order than on one chip, so a top_p or min_p cut can differ from sampler.py's at a near tie (measured per
deployment); greedy, temperature-only and top_k draws are the same tokens as sampler.py for the same logits.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.random import threefry2x32_p
from jax.sharding import PartitionSpec as P

from . import sampler as SP

f32, i32, u32 = jnp.float32, jnp.int32, jnp.uint32
CANDIDATES = 256        # per chip

pack, check, to_device, no_penalty, penalise, transform = SP.pack, SP.check, SP.to_device, SP.no_penalty, SP.penalise, SP.transform

try:                                    # jax >= 0.7
    from jax import shard_map as _shard_map

    def _smap(f, mesh, in_specs, out_specs):
        return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
except ImportError:                     # older jax
    from jax.experimental.shard_map import shard_map as _shard_map

    def _smap(f, mesh, in_specs, out_specs):
        return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)


def noise(seed, step, ids):
    """seed (2,) uint32 key data, step int32 token index, ids int32 global token ids (any shape) -> f32 Gumbel noise per id,
    bit-equal to jax.random.gumbel(fold_in(key, step), (V,))[ids] under threefry's partitionable mode."""
    k = jax.random.key_data(jax.random.fold_in(jax.random.wrap_key_data(seed, impl="threefry2x32"), step))
    v = ids.astype(u32)
    b1, b2 = threefry2x32_p.bind(k[0], k[1], jnp.zeros_like(v), v)
    fl = lax.bitcast_convert_type(lax.shift_right_logical(b1 ^ b2, u32(9)) | u32(0x3F800000), f32) - f32(1.0)
    lo, hi = f32(jnp.finfo(f32).tiny), f32(1.0)
    return -jnp.log(-jnp.log(lax.max(lo, fl * (hi - lo) + lo)))


def _unkey(k):
    """Inverse of sampler._key: ordered int32 key -> the f32 value."""
    return lax.bitcast_convert_type(jnp.where(k < 0, k ^ i32(0x7FFFFFFF), k), f32)


def _bits(x):
    return lax.bitcast_convert_type(x, i32)


class Sampler:
    """sampler.py's penalise / sample for (B, V) logits split on the vocabulary axis over mesh axis `axis`."""

    def __init__(self, mesh, axis="tp", candidates=CANDIDATES):
        self.mesh, self.axis, self.n, self.C = mesh, axis, int(mesh.shape[axis]), int(candidates)
        self.pack, self.check, self.to_device, self.no_penalty, self.penalise = pack, check, to_device, no_penalty, penalise

    def sample(self, x, sp, step, greedy_tok=None):
        """x (B, V) penalised logits f32, step (B,) int32 token index per row -> (B,) int32 tokens, as sampler.sample.
        greedy_tok is accepted for the same signature; greedy rows take the first argmax of x from the merge, which is the
        engine's greedy token (for rows without penalties x is the logits unchanged, so it is the forward's own argmax)."""
        return self.sample_rows(x, sp, step)[0]

    def sample_rows(self, x, sp, step):
        """-> (tokens (B,) int32, (B,) bool: the rows whose cut ran on the whole row)."""
        ax = self.axis
        f = _smap(self._body, self.mesh, (P(None, ax), P(), P(), P(), P(), P(), P()), (P(), P()))
        return f(x.astype(f32), sp["temperature"], sp["top_k"], sp["top_p"], sp["min_p"], sp["seed"], step.astype(i32))

    def _body(self, x, t, k, tp, mp, seed, step):
        ax, n = self.axis, self.n
        B, Vl = x.shape
        C = min(self.C, Vl)
        complete = C >= Vl                       # every token of every shard is a candidate
        me = lax.axis_index(ax)
        ids = me * Vl + jnp.arange(Vl, dtype=i32)
        greedy = t <= 0.0
        ga = jnp.argmax(x, -1).astype(i32)
        gm = jnp.take_along_axis(x, ga[:, None], -1)[:, 0]
        xt = x / jnp.where(greedy, 1.0, t)[:, None]
        key = SP._key(xt)
        ck, ci = lax.top_k(key, C)               # descending; equal keys: lower index first
        m_loc = _unkey(ck[:, 0])
        z_loc = jnp.sum(jnp.exp(xt - m_loc[:, None]), -1)
        pk = jnp.concatenate([ck, ci.astype(i32) + me * Vl, ga[:, None] + me * Vl, _bits(gm)[:, None],
                              _bits(m_loc)[:, None], _bits(z_loc)[:, None]], -1)
        G = jnp.moveaxis(lax.all_gather(pk, ax), 0, 1)                      # collective 1: (B, n, 2C + 4)
        gk, gid = G[:, :, :C], G[:, :, C:2 * C]
        g_id, g_m = G[:, :, 2 * C], lax.bitcast_convert_type(G[:, :, 2 * C + 1], f32)
        mv, zv = lax.bitcast_convert_type(G[:, :, 2 * C + 2], f32), lax.bitcast_convert_type(G[:, :, 2 * C + 3], f32)
        gtok = jnp.take_along_axis(g_id, jnp.argmax(g_m, -1)[:, None], -1)[:, 0]     # first chip with the largest value
        # merged candidates: key descending, then global id ascending (~key ascending == key descending, no overflow)
        nk, sid = lax.sort((~gk.reshape(B, n * C), gid.reshape(B, n * C)), num_keys=2)
        sk = ~nk
        M = _unkey(sk[:, 0])
        low = jnp.max(gk[:, :, C - 1], -1)                # the largest C-th candidate: below it a shard may hold more
        exact = (lambda thr: jnp.ones_like(thr, bool)) if complete else (lambda thr: thr > low)
        e = jnp.exp(_unkey(sk) - M[:, None])             # unnormalised probabilities of the sorted candidates
        z_all = jnp.sum(jnp.exp(mv - M[:, None]) * zv, -1)
        lo_key = jnp.iinfo(jnp.int32).min
        # top_k: the k-th largest key (ties kept), exact for k <= C
        use_k = k > 0
        tk = jnp.where(use_k, jnp.take_along_axis(sk, jnp.clip(k - 1, 0, n * C - 1)[:, None], -1)[:, 0], lo_key)
        need = use_k & (k > C) & (not complete)
        zk = jnp.where(use_k, jnp.sum(jnp.where(sk >= tk[:, None], e, 0.0), -1), z_all)
        # top_p on the top_k cut: the key of the first candidate whose running probability reaches top_p
        use_p = tp < 1.0
        p = jnp.where(sk >= tk[:, None], e, 0.0) / zk[:, None]
        reach = jnp.cumsum(p, -1) >= tp[:, None]
        tpk = jnp.take_along_axis(sk, jnp.argmax(reach, -1)[:, None], -1)[:, 0]
        tpk = jnp.where(use_p, tpk, lo_key)
        sums_k = use_k & (use_p | (mp > 0.0))             # a probability sum over the top_k survivors needs all of them
        need |= sums_k & ~exact(tk)
        need |= use_p & (~jnp.any(reach, -1) | ~exact(tpk))
        thr = jnp.maximum(tk, tpk)
        z2 = jnp.where(use_p, jnp.sum(jnp.where(sk >= thr[:, None], e, 0.0), -1), zk)
        need &= ~greedy
        keep = key >= thr[:, None]
        el = jnp.exp(xt - M[:, None])
        keep &= ~((mp > 0.0)[:, None] & (el / z2[:, None] < mp[:, None] * (1.0 / z2)[:, None]))
        xc = jnp.where(keep, xt, -jnp.inf)

        def full(_):
            xf = jnp.moveaxis(lax.all_gather(x, ax), 0, 1).reshape(B, n * Vl)       # collective (fallback only)
            sp = dict(temperature=t, top_k=k, top_p=tp, min_p=mp)
            xs = lax.dynamic_slice_in_dim(SP.transform(xf, sp), me * Vl, Vl, 1)
            return jnp.where(need[:, None], xs, xc)
        xc = lax.cond(jnp.any(need), full, lambda _: xc, None)
        y = xc + jax.vmap(noise, (0, 0, None))(seed, step, ids)
        da = jnp.argmax(y, -1).astype(i32)
        dv = jnp.take_along_axis(y, da[:, None], -1)[:, 0]
        D = jnp.moveaxis(lax.all_gather(jnp.stack([da + me * Vl, _bits(dv)], -1), ax), 0, 1)     # collective 2: (B, n, 2)
        dtok = jnp.take_along_axis(D[:, :, 0], jnp.argmax(lax.bitcast_convert_type(D[:, :, 1], f32), -1)[:, None], -1)[:, 0]
        return jnp.where(greedy, gtok, dtok), need
