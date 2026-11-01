"""runtime/attention.py: the one full-attention core. Guards: (1) a dense call changing at all (decode, verify and every
call up to BLOCK_FROM keys must be bit-identical to the formula the models ran before the module); (2) the blocked
prefill form (long windows) drifting from dense beyond op tolerance, mishandling a partial last block, masked keys or a
fully masked padding row; (3) the engine's chunk budget and the model's form disagreeing.
Tolerances written before the first run: float32 act, max abs diff <= 2e-6 + 2e-5 x |dense|; bfloat16 act, max abs diff
<= 1.6e-2 and mean abs diff <= 1e-3 (outputs O(1); one bf16 step at 1.0 is 7.8e-3)."""
import numpy as np
import jax
import jax.numpy as jnp

from mashbox.runtime import attention as A

f32, bf16 = jnp.float32, jnp.bfloat16


def old_dense(q, kk, vv, mask, scale, act):
    s = jnp.einsum("btkgd,bskd->bkgts", q, kk, preferred_element_type=f32) * scale
    p = jax.nn.softmax(jnp.where(mask[:, None, None], s, A.NEG), -1).astype(act)
    return jnp.einsum("bkgts,bskd->btkgd", p, vv, preferred_element_type=f32).astype(act)


def case(B, T, S, KV=2, G=2, HD=32, seed=0, act=f32, p0=None):
    r = np.random.default_rng(seed)
    q = jnp.asarray(r.standard_normal((B, T, KV, G, HD)), act)
    kk = jnp.asarray(r.standard_normal((B, S, KV, HD)), act)
    vv = jnp.asarray(r.standard_normal((B, S, KV, HD)), act)
    p0 = np.full(B, S - T) if p0 is None else np.asarray(p0)          # causal: cached positions < p0, then the chunk
    pos = np.arange(S)[None, None, :]
    qpos = (p0[:, None] + np.arange(T)[None, :])[:, :, None]
    cached = pos < p0[:, None, None]
    new = (pos >= S - T) & ((pos - (S - T)) <= (qpos - p0[:, None, None]))
    mask = jnp.asarray(cached | new)
    return q, kk, vv, mask


def test_dense_calls_are_unchanged():
    for (B, T, S) in ((1, 1, 9000), (3, 4, 9000), (2, 15, 20000), (2, 64, 4096), (1, 512, 512)):
        assert not A.blocked(T, S)
        q, kk, vv, mask = case(B, T, S, act=bf16, seed=S + T)
        sc = np.float32(32 ** -0.5)
        assert jnp.array_equal(A.attend(q, kk, vv, mask, sc, bf16), old_dense(q, kk, vv, mask, sc, bf16))


def test_blocked_matches_dense_within_tolerance():
    for (B, T, S, p0) in ((1, 16, 4097, None), (2, 64, 10000, [9936, 5000]), (1, 256, 2 * A.KV_BLOCK * 3 + 7, None)):
        assert A.blocked(T, S)
        sc = np.float32(32 ** -0.5)
        q, kk, vv, mask = case(B, T, S, act=f32, seed=T, p0=p0)
        d, b = np.asarray(old_dense(q, kk, vv, mask, sc, f32)), np.asarray(A.attend(q, kk, vv, mask, sc, f32))
        assert np.all(np.abs(b - d) <= 2e-6 + 2e-5 * np.abs(d)), np.abs(b - d).max()
        q, kk, vv, mask = case(B, T, S, act=bf16, seed=T + 1, p0=p0)
        d = np.asarray(old_dense(q, kk, vv, mask, sc, bf16), np.float32)
        b = np.asarray(A.attend(q, kk, vv, mask, sc, bf16), np.float32)
        assert np.abs(b - d).max() <= 1.6e-2 and np.abs(b - d).mean() <= 1e-3, (np.abs(b - d).max(), np.abs(b - d).mean())


def test_blocked_no_scale_and_masked_rows():
    q, kk, vv, mask = case(2, 32, 5000, act=f32, seed=3)
    d, b = old_dense(q, kk, vv, mask, np.float32(1.0), f32), A.attend(q, kk, vv, mask, None, f32)
    assert np.allclose(np.asarray(b), np.asarray(d), atol=2e-6, rtol=2e-5)
    mask = mask.at[1].set(False)                       # a padding row: every key masked
    out = np.asarray(A.attend(q, kk, vv, mask, None, f32))
    assert np.isfinite(out).all()
    assert np.allclose(out[0], np.asarray(d)[0], atol=2e-6, rtol=2e-5)


def test_engine_budget_covers_both_forms():
    """width() never budgets less than the form that runs: blocked = one block; dense = Tc + T"""
    for Tc in (0, 128, 2048, 4096, 8192, 65536, 131072):
        for T in (1, 4, 16, 64, 256, 512):
            w = A.width(Tc, T)
            if A.blocked(T, Tc + T):
                assert w >= A.KV_BLOCK and w >= min(Tc + T, A.BLOCK_FROM)
            else:
                assert w == Tc + T
