"""runtime/attention.attend_cache: attention against the K/V cache itself (runtime/kvcache.reader) plus the call's own
keys. Guards: (1) every call with at most BLOCK_FROM cached keys is bit-identical to the code the models ran before
(read the window, concatenate the call's keys, attention.attend); (2) the blocked form (decode, verify and prefill against
a longer window) drifting from dense beyond op tolerance, mishandling a window that is not a multiple of the block, rows
with different fill lengths, padding rows or a fully masked row; (3) blocks at or past `hi` being read at all (NaN there
must not reach a live row); (4) cache_hi leaving out rows that are not live.
Tolerances (KV-Q bars, written before the first run; the same as tests/cpu/test_attention_blocked.py): float32 act, max
abs diff <= 2e-6 + 2e-5 x |dense|; bfloat16 act, max abs diff <= 1.6e-2 and mean abs diff <= 1e-3."""
import numpy as np
import jax.numpy as jnp

from mashbox.runtime import attention as A, kvcache

f32, bf16 = jnp.float32, jnp.bfloat16


def case(B, T, Tc, p0, KV=2, G=3, HD=32, seed=0, act=f32, rows=None, contig=True):
    """a cache (1 group, rows slots, Tc positions) with the call's rows at slots 0..B-1 (contig) or reversed (gathered);
    mask = cached positions < p0 per row, then causal over the call's T keys"""
    r = np.random.default_rng(seed)
    R = rows or B + 1
    Kc = jnp.asarray(r.standard_normal((1, R, Tc, KV * HD)), act)
    Vc = jnp.asarray(r.standard_normal((1, R, Tc, KV * HD)), act)
    q = jnp.asarray(r.standard_normal((B, T, KV, G, HD)), act)
    k = jnp.asarray(r.standard_normal((B, T, KV, HD)), act)
    v = jnp.asarray(r.standard_normal((B, T, KV, HD)), act)
    p0 = np.asarray(p0)
    m1 = np.arange(Tc)[None, None, :] < p0[:, None, None]
    m2 = np.tril(np.ones((T, T), bool))[None]
    mask = jnp.asarray(np.concatenate([np.broadcast_to(m1, (B, T, Tc)), np.broadcast_to(m2, (B, T, T))], -1))
    slots = jnp.arange(B, dtype=jnp.int32) if contig else jnp.arange(B, dtype=jnp.int32)[::-1] + (R - B)
    return Kc, Vc, q, k, v, mask, slots


def old(Kc, Vc, q, k, v, mask, slots, Tc, scale, act, contig=True):
    """the code every call site ran before attend_cache"""
    KV, HD = k.shape[2], k.shape[3]
    if Tc:
        kc, vc = kvcache.read_heads(Kc, Vc, 0, slots, Tc, contig, KV, HD)
        kk, vv = jnp.concatenate([kc, k], 1), jnp.concatenate([vc, v], 1)
    else:
        kk, vv = k, v
    return A.attend(q, kk, vv, mask, scale, act)


def new(Kc, Vc, q, k, v, mask, slots, Tc, scale, act, contig=True, hi=None):
    rd = kvcache.reader(Kc, Vc, 0, slots, contig, k.shape[2], k.shape[3])
    return A.attend_cache(q, rd, Tc, k, v, mask, scale, act, hi=hi)


def close(b, d, act):
    b, d = np.asarray(b, np.float32), np.asarray(d, np.float32)
    if act == f32:
        assert np.all(np.abs(b - d) <= 2e-6 + 2e-5 * np.abs(d)), np.abs(b - d).max()
    else:
        assert np.abs(b - d).max() <= 1.6e-2 and np.abs(b - d).mean() <= 1e-3, (np.abs(b - d).max(), np.abs(b - d).mean())


def test_short_windows_are_bit_identical():
    sc = np.float32(32 ** -0.5)
    for (B, T, Tc, contig) in ((1, 1, 0, True), (2, 1, 128, True), (3, 4, 4096, True), (2, 64, 4096, False),
                               (1, 512, 2048, True)):
        p0 = np.minimum(np.arange(B) * 37 + Tc // 2, Tc)
        Kc, Vc, q, k, v, mask, slots = case(B, T, Tc, p0, act=bf16, seed=Tc + T, contig=contig)
        a = old(Kc, Vc, q, k, v, mask, slots, Tc, sc, bf16, contig)
        b = new(Kc, Vc, q, k, v, mask, slots, Tc, sc, bf16, contig, hi=A.cache_hi(jnp.asarray(p0), Tc))
        assert jnp.array_equal(a, b)


def test_long_windows_match_dense():
    sc = np.float32(32 ** -0.5)
    # decode (T 1), verify widths (2, 4), a prefill chunk (32); windows a multiple of the block and not; rows filled to
    # different lengths, one at position 0 of the window, contiguous and gathered slots
    for (B, T, Tc, p0, contig) in ((1, 1, 8192, [8191], True), (2, 1, 2 * A.DEC_BLOCK + 640, [3000, 2 * A.DEC_BLOCK + 600], True),
                                   (3, 2, 12288, [0, 4097, 12288], False), (2, 4, 4096 + 128, [4100, 17], True),
                                   (2, 32, 3 * A.KV_BLOCK + 512, [6000, 6656], True), (1, 16, 5120, [5104], False)):
        for act in (f32, bf16):
            Kc, Vc, q, k, v, mask, slots = case(B, T, Tc, p0, act=act, seed=T + Tc, contig=contig)
            d = old(Kc, Vc, q, k, v, mask, slots, Tc, sc, act, contig)
            close(new(Kc, Vc, q, k, v, mask, slots, Tc, sc, act, contig), d, act)
            close(new(Kc, Vc, q, k, v, mask, slots, Tc, sc, act, contig, hi=A.cache_hi(jnp.asarray(p0), Tc)), d, act)


def test_no_scale_and_fully_masked_row():
    Tc, p0 = 9000, [8000, 50]
    Kc, Vc, q, k, v, mask, slots = case(2, 1, Tc, p0, seed=5)
    close(new(Kc, Vc, q, k, v, mask, slots, Tc, None, f32), old(Kc, Vc, q, k, v, mask, slots, Tc, np.float32(1.0), f32), f32)
    mask = mask.at[1].set(False)
    out = np.asarray(new(Kc, Vc, q, k, v, mask, slots, Tc, None, f32))
    assert np.isfinite(out).all()


def test_blocks_past_hi_are_not_read():
    """NaN in every cached position from the first block boundary at or past hi: a live row's output stays finite and
    equal to the dense result over the clean cache; with hi = Tc (all blocks read) the NaN does reach the output"""
    blk = A.DEC_BLOCK
    Tc, p0 = 4 * blk, np.array([blk + 5, 2 * blk - 1])
    Kc, Vc, q, k, v, mask, slots = case(2, 1, Tc, p0, seed=9)
    d = old(Kc, Vc, q, k, v, mask, slots, Tc, np.float32(0.2), f32)
    Kn, Vn = Kc.at[:, :, 2 * blk:].set(jnp.nan), Vc.at[:, :, 2 * blk:].set(jnp.nan)
    hi = A.cache_hi(jnp.asarray(p0), Tc)
    assert int(hi) == 2 * blk - 1
    close(new(Kn, Vn, q, k, v, mask, slots, Tc, np.float32(0.2), f32, hi=hi), d, f32)
    assert not np.isfinite(np.asarray(new(Kn, Vn, q, k, v, mask, slots, Tc, np.float32(0.2), f32))).all()


def test_cache_hi_leaves_out_rows_that_are_not_live():
    p0 = jnp.asarray([100, 65536, 7], jnp.int32)
    assert int(A.cache_hi(p0, 65536)) == 65536
    assert int(A.cache_hi(p0, 65536, keep=jnp.asarray([True, False, True]))) == 100
    assert int(A.cache_hi(p0, 4096, keep=jnp.asarray([True, True, True]))) == 4096
    assert int(A.cache_hi(p0, 4096, keep=jnp.asarray([False, False, False]))) == 0


def _all_eqns(jaxpr):
    for e in jaxpr.eqns:
        yield e
        for p in e.params.values():
            for j in (p if isinstance(p, (list, tuple)) else [p]):
                inner = getattr(j, "jaxpr", j)
                if hasattr(inner, "eqns"):
                    yield from _all_eqns(inner)


def test_long_window_is_never_materialised_whole():
    """structure: inside attend_cache at a long window (bf16 and int8 cache), no value carries the window axis: every
    read is one block, and nothing concatenates the window with the call's keys"""
    import jax
    Tc, KV, HD = 65536, 2, 32
    for store in ("bf16", "int8"):
        kvcache.set_store(store)
        try:
            Kc, Vc = kvcache.zeros(1, 2, Tc, KV, HD, bf16)
        finally:
            kvcache.set_store("bf16")
        for T in (1, 32):
            q = jnp.zeros((1, T, KV, 3, HD), bf16)
            k = jnp.zeros((1, T, KV, HD), bf16)
            mask = jnp.ones((1, T, Tc + T), bool)

            def f(Kc, Vc, q, k, mask):
                rd = kvcache.reader(Kc, Vc, 0, jnp.arange(1, dtype=jnp.int32), True, KV, HD)
                return A.attend_cache(q, rd, Tc, k, k, mask, None, bf16, hi=jnp.int32(Tc))
            jp = jax.make_jaxpr(f)(Kc, Vc, q, k, mask)
            for e in _all_eqns(jp.jaxpr):
                assert e.primitive.name != "concatenate" or all(Tc not in v.aval.shape for v in e.outvars), e
                for v in e.outvars:
                    assert not any(d >= Tc for d in getattr(v.aval, "shape", ())), (store, T, e.primitive.name, v.aval)
