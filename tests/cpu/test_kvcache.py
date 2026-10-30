"""runtime/kvcache.py: the K/V cache of every model the engine runs (layout, read, write, the sliding-window ring).

What these tests hold:
  * write() equals the old dropping scatter (Kc.at[:n, slots[:, None], pos].set(ks, mode="drop") on the (KV, HD) view)
    bit for bit: one-token rows at positions 0, 9, 15, 16 (the end), 19 (past it); a batch of rows; multi-token rows
    that start inside, run past and start past the end; a single group written at an offset; calls with more
    positions per row than the cache holds (a 512-token chunk on a 256-position context), starting at 0, inside, and
    at the end.
  * ring_write() keeps, for every slot and every ring entry whose ring_pos() is >= 0, the exact K/V a linear cache of
    the same writes holds at that position, through 300 random calls (rows of 1..7 tokens on a ring of 5, padding
    tokens past each row's length, rows the call may not change, positions up to and past the context end); entries of
    rows that a call may not change, and tokens past the length or the context, change nothing.
  * The write and ring programs have no cond and no scatter (XLA updates a donated cache in place only through
    dynamic_update_slice; the cond and the dropping scatter each gave the decode loop's cache a layout of its own on TPU,
    copied whole at every step: tools/kv_hlo_check.py). The read holds its slice in the cache's own layout.
  * On a TPU compile (deviceless, libtpu on the host; skipped without it): the decode program of a 2-K/V-head x 64 model
    (the layout the old cache padded) has 0 copies of the whole cache and keeps it aliased in place. The old padded
    layout gave 6 whole-cache copies there.
"""
import importlib.util
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox.runtime import kvcache as KC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G, R, CTX, KV, HD = 3, 5, 16, 2, 4


def _kv(seed=0, shape=(G, R, CTX, KV * HD)):
    r = np.random.default_rng(seed)
    return tuple(jnp.asarray(r.standard_normal(shape, dtype=np.float32)) for _ in range(2))


def _prims(fn, *args):
    out = []

    def walk(j):
        for e in j.eqns:
            out.append(e.primitive.name)
            for p in e.params.values():
                for q in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(q, "jaxpr"):
                        walk(q.jaxpr if hasattr(q.jaxpr, "eqns") else q.jaxpr.jaxpr)
    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return out


def _scatter(C, slots, pos, x, g=0):
    n = x.shape[0]
    C5 = C.reshape(G, R, CTX, KV, HD)
    return C5.at[g:g + n, slots[:, None], pos].set(x, mode="drop").reshape(C.shape)


def test_layout_is_heads_flat_in_the_row():
    assert KC.shape(G, R, CTX, KV, HD) == (G, R, CTX, KV * HD)
    K, V = KC.zeros(G, R, CTX, KV, HD, jnp.bfloat16)
    assert K.shape == V.shape == (G, R, CTX, KV * HD) and K.dtype == jnp.bfloat16


CASES = {
    "one-pos0": ([2], [0], 1), "one-pos9": ([2], [9], 1), "one-pos15": ([2], [15], 1), "one-pos16": ([2], [16], 1),
    "one-pos19": ([2], [19], 1), "batch": ([0, 3, 1], [4, 15, 16], 1), "multi-inside": ([1, 2], [0, 6], 4),
    "multi-past-end": ([1, 2], [13, 16], 4), "multi-long": ([4], [3], 9),
    "longer-than-cache": ([1, 2], [0, 5], 20), "longer-past-end": ([3, 0], [16, 2], 20),
}


@pytest.mark.parametrize("case", list(CASES))
def test_write_equals_the_dropping_scatter(case):
    sl, p0, T = CASES[case]
    Kc, Vc = _kv(1)
    r = np.random.default_rng(2)
    ks, vs = (jnp.asarray(r.standard_normal((G, len(sl), T, KV, HD), dtype=np.float32)) for _ in range(2))
    s = jnp.asarray(sl, jnp.int32)
    pos = jnp.asarray(p0, jnp.int32)[:, None] + jnp.arange(T, dtype=jnp.int32)[None]
    got = jax.jit(KC.write)(Kc, Vc, s, pos, ks, vs)
    for g_, C, x in zip(got, (Kc, Vc), (ks, vs)):
        assert np.array_equal(np.asarray(g_), np.asarray(_scatter(C, s, pos, x)))


def test_write_of_one_group_at_an_offset():
    Kc, Vc = _kv(3)
    r = np.random.default_rng(4)
    k = jnp.asarray(r.standard_normal((1, 2, 1, KV, HD), dtype=np.float32))
    s, pos = jnp.asarray([1, 3], jnp.int32), jnp.asarray([[5], [16]], jnp.int32)
    K2, V2 = jax.jit(KC.write, static_argnames=())(Kc, Vc, s, pos, k, k, jnp.int32(2))
    assert np.array_equal(np.asarray(K2), np.asarray(_scatter(Kc, s, pos, k, g=2)))


@pytest.mark.parametrize("B,T", [(1, 1), (3, 1), (2, 4)])
def test_write_program_has_no_cond_and_no_scatter(B, T):
    Kc, Vc = _kv()
    x = jnp.ones((G, B, T, KV, HD))
    pr = _prims(lambda K, V, s, p: KC.write(K, V, s, p, x, x), Kc, Vc, jnp.arange(B, dtype=jnp.int32),
                jnp.zeros((B, T), jnp.int32))
    assert "cond" not in pr and not any("scatter" in q for q in pr) and "dynamic_update_slice" in pr, pr


def test_read_is_one_slice_or_a_slot_gather_held_in_the_cache_layout():
    Kc, Vc = _kv(5)
    slots = jnp.arange(3, dtype=jnp.int32)
    pr = _prims(lambda K, V: KC.read(K, V, 1, slots, 8, True), Kc, Vc)
    assert "gather" not in pr and pr.count("layout_constraint") == 2, pr
    k, v = jax.jit(lambda K, V: KC.read_heads(K, V, 1, slots, 8, True, KV, HD))(Kc, Vc)
    assert np.array_equal(np.asarray(k), np.asarray(Kc[1][:3, :8].reshape(3, 8, KV, HD)))
    other = jnp.asarray([4, 0], jnp.int32)
    k, v = KC.read(Kc, Vc, 2, other, 8, False, width=4)
    assert np.array_equal(np.asarray(v), np.asarray(Vc[2][other, :8, :4]))


def test_ring_holds_what_a_linear_cache_holds_at_each_entrys_position():
    size, ctx, rows, n, w = 5, 24, 4, 2, 3
    rng = np.random.default_rng(11)
    Ks, Vs = KC.zeros(n, rows, size, 1, w, jnp.float32)
    lin = np.zeros((n, rows, ctx + 8, w), np.float32)       # every real write at its position
    cur = np.zeros(rows, np.int64)                          # each slot's next position
    fns = {}
    for it in range(300):
        T = int(rng.integers(1, 8))
        B = int(rng.integers(1, 3))
        sl = rng.choice(rows, B, replace=False)
        if rng.random() < 0.1:                              # a slot starts over (a new conversation)
            cur[sl[0]] = 0
        p0 = cur[sl].copy()
        lens = rng.integers(0, T + 1, B)
        keep = rng.random(B) < 0.85
        x = rng.standard_normal((n, B, T, w)).astype(np.float32)
        pos = p0[:, None] + np.arange(T)[None]
        f = fns.setdefault((B, T), jax.jit(lambda K, V, s, p, l, a, kp: KC.ring_write(K, V, s, p, l, a, a, ctx, keep=kp)))
        before = np.asarray(Ks)
        Ks, Vs = f(Ks, Vs, jnp.asarray(sl, jnp.int32), jnp.asarray(pos, jnp.int32), jnp.asarray(lens, jnp.int32),
                   jnp.asarray(x), jnp.asarray(keep))
        after = np.asarray(Ks)
        assert np.array_equal(after, np.asarray(Vs))
        for b, s_ in enumerate(sl):
            c = int(min(lens[b], ctx - p0[b])) if keep[b] else 0
            c = max(c, 0)
            if c == 0:
                assert np.array_equal(after[:, s_], before[:, s_]), (it, "a row that writes nothing changed its ring")
            lin[:, s_, p0[b]:p0[b] + c] = x[:, b, :c]
            cur[s_] = p0[b] + c
        for s_ in range(rows):
            if s_ in sl or it % 37 == 0:
                rp = np.asarray(KC.ring_pos(jnp.asarray([cur[s_]], jnp.int32), size, size))[0]
                for e in range(size):
                    if rp[e] >= 0:
                        assert np.array_equal(after[:, s_, e], lin[:, s_, rp[e]]), (it, s_, e, int(rp[e]))
    assert cur.max() >= ctx - 1                             # the context end was reached


def test_ring_program_has_no_cond_and_no_scatter():
    Ks, Vs = KC.zeros(2, 3, 5, 1, 4, jnp.float32)
    for B, T in ((1, 1), (2, 3), (1, 8)):
        x = jnp.ones((2, B, T, 4))
        pr = _prims(lambda K, V, s, p, l: KC.ring_write(K, V, s, p, l, x, x, 16), Ks, Vs, jnp.arange(B, dtype=jnp.int32),
                    jnp.zeros((B, T), jnp.int32), jnp.ones((B,), jnp.int32))
        assert "cond" not in pr and not any("scatter" in q for q in pr), pr


def _libtpu():
    try:
        return importlib.util.find_spec("libtpu") is not None
    except Exception:
        return False


@pytest.mark.skipif(not _libtpu(), reason="needs libtpu on the host for a deviceless TPU compile (no chip)")
def test_tpu_decode_updates_the_cache_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_SKIP_MDS_QUERY", "1")
    monkeypatch.setenv("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    s = importlib.util.spec_from_file_location("kv_hlo_check", os.path.join(ROOT, "tools", "kv_hlo_check.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m.MODELS["t-kv2-hd64"] = ("qwen2", dict(layers=2, hp=dict(D=128, F=128, H=4, KV=2, HD=64)))
    out = m.run("t-kv2-hd64", S=4, CTX=512, K=4, programs=("dec1", "decS"), log=lambda *_: None)
    for p, r in out["programs"].items():
        assert "error" not in r and r["cache_copies"] == 0 and r["aliased"], (p, r)
