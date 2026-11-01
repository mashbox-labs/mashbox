"""MASHBOX_KV_CACHE=int8 (runtime/kvcache.py pack/unpack, write, reader; runtime/attention.attend_cache's int8 form). Guards:
(1) the packed row losing more than int8 rounding (per position and head: |x - x'| <= absmax / 127, + float rounding);
(2) int8 attention drifting from bf16 attention on the same values beyond 5e-2 (dense short window and blocked long
window, decode and prefill); (3) a write landing anywhere but its (slot, positions) or not in the packed form; (4) a
model forward on int8 K/V (GGUF fixture qwen3-f32, prefill then decode, dense and forced-blocked attention) moving its
logits beyond the KV-Q bars (KVQ-BARS.txt ADDENDUM 2, written before the first run); (5) the bf16 default changing:
the store must be reset to bf16 after each test, and int8 must be refused where it is not offered."""
import importlib.util
import json
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mashbox.runtime import attention as A, kvcache

f32, bf16 = jnp.float32, jnp.bfloat16
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def int8():
    kvcache.set_store("int8")
    yield
    kvcache.set_store("bf16")


def test_pack_round_trip(int8):
    r = np.random.default_rng(0)
    x = r.standard_normal((2, 3, 5, 4, 64)).astype(np.float32) * np.array([1e-3, 1.0, 30.0, 0.0])[None, None, None, :, None]
    kr, vr = kvcache.pack(jnp.asarray(x[0]), jnp.asarray(x[1] * 7.0), 4, 64)
    assert kr.dtype == vr.dtype == jnp.int8 and kr.shape == (3, 5, 4 * 64 + kvcache.QPAD) and vr.shape == (3, 5, 4 * 64)
    for (c, s), want in zip(kvcache.unpack(kr, vr, 4, 64), (x[0], x[1] * 7.0)):
        y = np.asarray(kvcache.dequant(c, s, f32))
        amax = np.abs(want).max(-1, keepdims=True)
        assert np.all(np.abs(y - want) <= amax / 127 * 0.5 * (1 + 1e-5) + 1e-30)
        assert np.all(y[:, :, 3] == 0)                     # an all-zero head stays zero (scale 0, no NaN)


def test_write_lands_in_place_packed(int8):
    Kc, Vc = kvcache.zeros(2, 3, 64, 2, 32, bf16)
    assert Kc.dtype == Vc.dtype == jnp.int8 and Kc.shape == (2, 3, 64, 64 + kvcache.QPAD) and Vc.shape == (2, 3, 64, 64)
    r = np.random.default_rng(1)
    ks = jnp.asarray(r.standard_normal((2, 2, 4, 2, 32)), bf16)
    vs = jnp.asarray(r.standard_normal((2, 2, 4, 2, 32)), bf16)
    slots, pos = jnp.asarray([2, 0], jnp.int32), jnp.asarray([[10, 11, 12, 13], [62, 63, 64, 65]], jnp.int32)
    K2, V2 = kvcache.write(Kc, Vc, slots, pos, ks, vs)
    wk, wv = (np.asarray(a) for a in kvcache.pack(ks, vs, 2, 32))
    for C, want in ((np.array(K2), wk), (np.array(V2), wv)):
        assert np.array_equal(C[:, 2, 10:14], want[:, 0])
        assert np.array_equal(C[:, 0, 62:64], want[:, 1, :2])  # positions past the cache are dropped
        C[:, 2, 10:14] = 0
        C[:, 0, 62:64] = 0
        assert not C.any()


def test_tile_writes_and_reads_match_positionwise(int8):
    """int8 writes and reads go through whole 32-position tiles (kvcache._tiles): against a position-by-position
    reference (later rows win on a shared slot; positions past the cache are dropped), over windows that start inside a
    tile, cross tiles, end at or run past the cache end, a call as long as the cache, and one group of several"""
    r = np.random.default_rng(3)
    CTX = 96
    for (B, T, p0s, slots, g, n) in ((1, 1, [0], [1], 0, 1), (2, 1, [31, 95], [0, 2], 1, 1), (3, 5, [30, 93, 94], [2, 2, 0], 0, 2),
                                     (2, 40, [5, 70], [1, 0], 0, 1), (1, 96, [0], [2], 1, 1), (2, 33, [64, 63], [0, 1], 0, 2)):
        Kc = jnp.asarray(r.integers(-127, 128, (3, 3, CTX, 64 + kvcache.QPAD)), jnp.int8)
        Vc = jnp.asarray(r.integers(-127, 128, (3, 3, CTX, 64)), jnp.int8)
        ks = jnp.asarray(r.standard_normal((n, B, T, 2, 32)), bf16)
        vs = jnp.asarray(r.standard_normal((n, B, T, 2, 32)), bf16)
        pos = jnp.asarray(np.asarray(p0s)[:, None] + np.arange(T)[None], jnp.int32)
        K2, V2 = kvcache.write(Kc, Vc, jnp.asarray(slots, jnp.int32), pos, ks, vs, g=g)
        wk, wv = (np.asarray(a) for a in kvcache.pack(ks, vs, 2, 32))
        for C0, C, w in ((Kc, K2, wk), (Vc, V2, wv)):
            ref = np.array(C0)
            for b in range(B):
                for t in range(T):
                    if p0s[b] + t < CTX:
                        ref[g:g + n, slots[b], p0s[b] + t] = w[:, b, t]
            assert np.array_equal(np.asarray(C), ref), (B, T, p0s)
        # whole tiles (mult a multiple of 32) and cut windows (any start, any length; up to the cache end)
        for start, m, mult in ((0, 96, 96), (32, 32, 32), (64, 32, 32), (0, 64, 64), (16, 48, 1), (5, 91, 1), (70, 26, 2),
                               (64, 32, 1), (90, 6, 1)):
            for contig, sl in ((True, [0, 1]), (False, [2, 0])):
                a, c = kvcache.read(K2, V2, 1, jnp.asarray(sl, jnp.int32), m, contig, start=jnp.int32(start), mult=mult)
                for C, got in ((K2, a), (V2, c)):
                    want = np.asarray(C)[1, sl if not contig else [0, 1], start:start + m]
                    assert np.array_equal(np.asarray(got), want), (start, m, mult, contig)
    with pytest.raises(ValueError):
        kvcache.zeros(1, 2, 100, 2, 32, bf16)               # an int8 cache holds whole 32-position tiles


def _cache_case(B, T, Tc, p0, KV=2, G=3, HD=32, seed=0):
    r = np.random.default_rng(seed)
    kv = r.standard_normal((2, 1, B + 1, Tc, KV, HD)).astype(np.float32)
    q = jnp.asarray(r.standard_normal((B, T, KV, G, HD)), bf16)
    k = jnp.asarray(r.standard_normal((B, T, KV, HD)), bf16)
    v = jnp.asarray(r.standard_normal((B, T, KV, HD)), bf16)
    m1 = np.arange(Tc)[None, None, :] < np.asarray(p0)[:, None, None]
    mask = jnp.asarray(np.concatenate([np.broadcast_to(m1, (B, T, Tc)), np.broadcast_to(np.tril(np.ones((T, T), bool)), (B, T, T))], -1))
    return kv, q, k, v, mask


@pytest.mark.parametrize("B,T,Tc,p0", [(2, 1, 2048, [2000, 7]), (1, 1, 3 * A.DEC_BLOCK + 300, [3 * A.DEC_BLOCK + 250]),
                                       (2, 4, 9000, [8990, 4500]), (1, 32, 2 * A.KV_BLOCK + 4200, [8000])])
def test_int8_attention_close_to_bf16(B, T, Tc, p0):
    kv, q, k, v, mask = _cache_case(B, T, Tc, p0, seed=Tc + T)
    slots, KV, HD = jnp.arange(B, dtype=jnp.int32), 2, 32
    Kb, Vb = (jnp.asarray(a.reshape(1, B + 1, Tc, KV * HD), bf16) for a in kv)
    ref = A.attend_cache(q, kvcache.reader(Kb, Vb, 0, slots, True, KV, HD), Tc, k, v, mask, np.float32(HD ** -0.5), bf16)
    # the int8 cache holds whole 32-position tiles (kvcache.TILE): positions past Tc are there and never read
    pad = [(0, 0)] * 3 + [(0, -Tc % kvcache.TILE)] + [(0, 0)] * 2
    kv = np.pad(kv, pad, constant_values=np.nan)
    Kq, Vq = kvcache.pack(jnp.asarray(kv[0], bf16), jnp.asarray(kv[1], bf16), KV, HD)
    assert kvcache.quantized(Kq)
    hi = A.cache_hi(jnp.asarray(p0), Tc)
    out = A.attend_cache(q, kvcache.reader(Kq, Vq, 0, slots, True, KV, HD), Tc, k, v, mask, np.float32(HD ** -0.5), bf16, hi=hi)
    d = np.abs(np.asarray(out, np.float32) - np.asarray(ref, np.float32))
    assert d.max() <= 5e-2, d.max()


_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)


def _run(path, toks, split, forced, monkeypatch):
    from mashbox.gguf import open_model
    if forced:
        monkeypatch.setattr(A, "BLOCK_FROM", 8)
        monkeypatch.setattr(A, "KV_BLOCK", 8)
        monkeypatch.setattr(A, "DEC_BLOCK", 8)
    f, M, c = open_model(path, act=f32)
    W, _ = M.load(f, c, use_kernels=False)
    fwd = jax.jit(M.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = M.new_state(c, 1, 32)
    i32, z, T = jnp.int32, jnp.zeros((1,), jnp.int32), len(toks)
    t = jnp.asarray([toks], i32)
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    monkeypatch.undo()
    return np.concatenate(out, 0), ck


def _kl_top1(ref, ours):
    p = np.exp(ref - ref.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    return float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - lq), -1).mean()), int(np.sum(ref.argmax(-1) == ours.argmax(-1)))


def test_model_on_int8_kv_within_bars(tmp_path, monkeypatch):
    meta = json.load(open(os.path.join(ROOT, "tests", "fixtures", "gguf", "qwen3-f32.json")))
    p = os.path.join(ROOT, "tests", "fixtures", "gguf", "qwen3-f32.gguf")
    if meta["made_by"] == "gguf-py":
        p = FXT.build("qwen3-f32", str(tmp_path))
    toks = meta["tokens"]
    assert len(toks) == 16
    try:
        dense, ckb = _run(p, toks, 8, False, monkeypatch)
        blocked, _ = _run(p, toks, 8, True, monkeypatch)
        e = float(np.sum((blocked.astype(np.float64) - dense) ** 2) / np.sum(dense.astype(np.float64) ** 2))
        assert e <= 1e-4, e
        kvcache.set_store("int8")
        for forced, ref in ((False, dense), (True, blocked)):
            q8, ck8 = _run(p, toks, 8, forced, monkeypatch)
            assert ck8[0].dtype == jnp.int8 and ck8[0].shape[-1] == ckb[0].shape[-1] + kvcache.QPAD and ck8[1].shape == ckb[1].shape
            kl, top1 = _kl_top1(ref, q8)
            print(f"int8 forced={forced}: KL {kl:.3e} top-1 {top1}/16")
            assert kl <= 1e-2 and top1 >= 15, (kl, top1)
    finally:
        kvcache.set_store("bf16")


def test_int8_refused_where_not_offered():
    kvcache.set_store("int8")
    try:
        with pytest.raises(NotImplementedError):
            kvcache.ring_write(*kvcache.zeros(1, 2, 32, 1, 32, bf16), jnp.zeros(1, jnp.int32), jnp.zeros((1, 1), jnp.int32),
                               jnp.ones(1, jnp.int32), jnp.zeros((1, 1, 1, 1, 32)), jnp.zeros((1, 1, 1, 1, 32)), 32)
        with pytest.raises(ValueError):
            kvcache.set_store("fp4")
    finally:
        kvcache.set_store("bf16")
    assert kvcache.STORE == "bf16" and kvcache.zeros(1, 1, 4, 1, 8, bf16)[0].dtype == bf16
