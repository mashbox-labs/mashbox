"""The record-only top-k hook (runtime/engine.py): with trace=True and MASHBOX_RECORD_TOPK=k the
SAME decode program that commits each token also returns that step's top-k f32 logits (values and ids); the first token
of a row comes from the prefill's last-row logits, as its trace hash does. Off by default (no variable, or trace off).
The hook must not change what the engine decodes: greedy and sampled, 1 row and 3 rows, the committed ids and the f32
logits' hash of every token are equal with the hook on and off. The model is a stand-in whose logits depend only on the
fed token and its position, with every logit in [1, 1.001) (one bf16 value), so a hook that rounds to bf16 shows."""
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mashbox.runtime.engine import Engine

V, TOPK, CTX = 256, 64, 256
SAMP = {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "presence_penalty": 1.0, "seed": 3}
PROMPTS = [[5, 9, 2], [17, 3, 3, 8, 1, 4, 30], [250]]


def logits(h):
    """(B,) f32 -> (B, V) f32 in [1, 1.001): a hash-like spread, the top values far apart in f32, equal in bf16"""
    v = jnp.arange(V, dtype=jnp.float32)
    u = jnp.sin(h[:, None] * 12.9898 + v[None, :] * 78.233) * 43758.5453
    return 1.0 + 1e-3 * (u - jnp.floor(u))


def new_state(c, S, CTX):
    kv = (1, S + 1, CTX, 2)
    return ((jnp.zeros(kv, jnp.bfloat16), jnp.zeros(kv, jnp.bfloat16)),
            (jnp.zeros((1, 1, S + 1, 1, 1, 1), jnp.float32), jnp.zeros((1, 1, S + 1, 1, 1), jnp.bfloat16)))


def make_forward(c, decode_kernels=None):
    def fwd(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, verify=False, hidden=False, keep=None):
        (K_, V_), (R, C) = ck, cv
        B, T = toks.shape
        last = jnp.take_along_axis(toks, jnp.maximum(lens - 1, 0)[:, None], 1)[:, 0]
        h = last.astype(jnp.float32) * W["w"] + 7.0 * (p0 + lens - 1).astype(jnp.float32)
        K_ = K_.at[0, slots, p0, 0].set(h.astype(K_.dtype))
        lg = logits(h)
        out = {"ck": (K_, V_), "cv": (R, C), "nxt": jnp.argmax(lg, -1).astype(jnp.int32), "lg": lg}
        if all_logits:
            out["lg"] = jnp.broadcast_to(lg[:, None], (B, T, c.V))
        return out
    return fwd


MODEL = types.SimpleNamespace(new_state=new_state, make_forward=make_forward)


def engine(monkeypatch, topk=None, trace=True, S=4):
    if topk is None:
        monkeypatch.delenv("MASHBOX_RECORD_TOPK", raising=False)
    else:
        monkeypatch.setenv("MASHBOX_RECORD_TOPK", str(topk))
    c = types.SimpleNamespace(V=V, H=2)
    return Engine(c, {"w": jnp.ones((), jnp.float32)}, MODEL, CTX=CTX, S=S, K=4, chunk=64, tokens_per_call=2048, trace=trace)


def runs(e):
    """1 row per call (greedy, sampled), then 3 rows in one call (greedy, sampled): the dec and decs programs"""
    out = [e.generate([p], max_new=11) for p in PROMPTS]
    out += [e.generate([p], max_new=11, samp=[dict(SAMP, seed=i)]) for i, p in enumerate(PROMPTS)]
    out.append(e.generate(PROMPTS, max_new=11))
    out.append(e.generate(PROMPTS, max_new=11, samp=[dict(SAMP, seed=i) for i in range(3)]))
    return out


def test_the_hook_is_off_by_default_and_needs_trace(monkeypatch):
    assert engine(monkeypatch).topk == 0 and "topk" not in engine(monkeypatch).generate([PROMPTS[0]], max_new=3)
    assert "topk" not in engine(monkeypatch, topk=TOPK, trace=False).generate([PROMPTS[0]], max_new=3)


def test_trace_turned_on_after_start_carries_the_hook(monkeypatch):
    """the record wrapper starts the engine as a user does (trace off), then sets eng.trace = True for its trace pass"""
    e = engine(monkeypatch, topk=TOPK, trace=False)
    plain = e.generate([PROMPTS[1]], max_new=11)
    e.trace = True
    got = e.generate([PROMPTS[1]], max_new=11)
    assert got["ids"] == plain["ids"] and len(got["topk"][0]) == 11
    assert all(ti[0] == t and len(ti) == TOPK for (ti, _), t in zip(got["topk"][0], got["ids"][0]))


def test_decode_is_bit_identical_with_the_hook_on_and_off(monkeypatch):
    off, plain, on = runs(engine(monkeypatch)), runs(engine(monkeypatch, trace=False)), runs(engine(monkeypatch, topk=TOPK))
    for a, b, c in zip(off, plain, on):
        assert a["ids"] == b["ids"] == c["ids"]
        assert a["lh"] == c["lh"]                       # every token's f32 logits hash and top-2 gap: equal
        assert all(len(x) == 11 for x in c["ids"])


def test_each_tokens_topk_is_its_own_steps_f32_logits(monkeypatch):
    for res in runs(engine(monkeypatch, topk=TOPK)):
        for ids, lh, tk in zip(res["ids"], res["lh"], res["topk"]):
            assert len(tk) == len(ids) == len(lh)
            for j, ((ti, tv), (_, gap)) in enumerate(zip(tk, lh)):
                tv32 = np.asarray(tv, np.float32)
                assert len(ti) == len(tv) == TOPK and len(set(ti)) == TOPK and np.all(np.diff(tv32) <= 0)
                assert np.float32(tv32[0] - tv32[1]) == np.float32(gap)          # the trace's f32 top-2 gap, bit for bit
                assert tv32[0] - tv32[-1] > 0                                      # f32 values (bf16 would be all 1.0)
    one = engine(monkeypatch, topk=TOPK).generate([PROMPTS[1]], max_new=11)
    ids, tk = one["ids"][0], one["topk"][0]
    assert all(ti[0] == t for (ti, _), t in zip(tk, ids))                     # greedy: the token is the top-1 id
    seq = PROMPTS[1] + ids
    for j in range(1, len(ids)):                                                # decode steps: the fed token and position
        pos = len(PROMPTS[1]) + j - 1
        ref = np.asarray(logits(jnp.asarray([seq[pos] + 7.0 * pos], jnp.float32)))[0]
        ti, tv = np.asarray(tk[j][0]), np.asarray(tk[j][1], np.float32)
        np.testing.assert_allclose(tv, ref[ti], rtol=0, atol=2e-6)              # this step's logits at those ids
        np.testing.assert_allclose(tv, np.sort(ref)[::-1][:TOPK], rtol=0, atol=2e-6)   # and its 64 largest
