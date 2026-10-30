"""Engine.warm covers every program a request can reach: after the warm-up, requests at log-spaced prompt lengths 1..ctx,
greedy and sampled, 1..S rows, continued turns (prefix cache) and host-cache restores add no compiled program (the jit
caches of the prefill and decode programs keep their size). The model is a stand-in with the real engine's state layout
(K/V per slot and position, a linear-attention state per slot), so the test needs only JAX on CPU.
The rule it checks: no compile at serve time."""
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox.runtime.engine import Engine  # noqa: E402

V = 64


def new_state(c, S, CTX):
    kv = (1, S + 1, CTX, 2)
    return ((jnp.zeros(kv, jnp.bfloat16), jnp.zeros(kv, jnp.bfloat16)),
            (jnp.zeros((1, 1, S + 1, 1, 1, 1), jnp.float32), jnp.zeros((1, 1, S + 1, 1, 1), jnp.bfloat16)))


def make_forward(c, decode_kernels=None):
    def fwd(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, verify=False, hidden=False, keep=None):
        # the model forward's signature (verify, hidden: speculative decoding, which this stand-in has no head for; keep:
        # the rows whose state a call may change, which a stand-in with no linear state has nothing to hold back for)
        (K_, V_), (R, C) = ck, cv
        B, T = toks.shape
        x = jnp.where(jnp.arange(T)[None, :] < lens[:, None], toks.astype(jnp.float32) * W["w"], 0.0)
        h = x.sum(1) + p0.astype(jnp.float32) + R[0, 0, slots, 0, 0, 0]
        if Tc:
            h = h + 0.0 * K_[0, slots, :Tc, 0].astype(jnp.float32).sum(1)
        K_ = K_.at[0, slots, p0, 0].set(h.astype(K_.dtype))
        R = R.at[0, 0, slots, 0, 0, 0].set(h)
        lg = jax.nn.one_hot(h.astype(jnp.int32) % c.V, c.V, dtype=jnp.float32) + 0.01 * jnp.arange(c.V, dtype=jnp.float32)
        out = dict(ck=(K_, V_), cv=(R, C), nxt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg)
        if all_logits:
            out["lg"] = jnp.broadcast_to(lg[:, None], (B, T, c.V))
        return out
    return fwd


MODEL = types.SimpleNamespace(new_state=new_state, make_forward=make_forward)
SAMP = dict(temperature=0.7, top_p=0.9, top_k=20, presence_penalty=1.0, seed=3)


def sizes(e):
    return {k: getattr(e, k)._cache_size() for k in ("fwd", "dec", "decs", "first")}


@pytest.mark.parametrize("CTX,S,chunk,tpc,attn", [
    (512, 1, 64, 2048, 2e9),          # one slot (the --agents server shape)
    (512, 4, 64, 128, 3e5),           # rows per call capped by tokens_per_call; chunk and rows cut by attn_bytes
    (512, 2, 64, 2048, 2e9),          # S <= 2: no scratch slot
])
def test_no_compile_after_warm(CTX, S, chunk, tpc, attn):
    c = types.SimpleNamespace(V=V, H=2)
    e = Engine(c, {"w": jnp.ones((), jnp.float32)}, MODEL, CTX=CTX, S=S, K=4, chunk=chunk, tokens_per_call=tpc,
               prefix_cache=True, attn_bytes=attn, cache_ram=1 << 30)
    hello = [5, 6, 7]
    e.generate([hello], max_new=5)                                     # as cli.warm: the small helper programs
    e.generate([hello], max_new=5, samp=[SAMP])
    w = e.warm(workers=2)
    assert w["prefill"] == len(e.prefill_shapes()) and w["decode"] > 0 and w["compile_failed"] == 0
    s0 = sizes(e)
    rng = np.random.default_rng(1)
    conv = None
    for L in (1, 2, 5, 15, 16, 17, 40, 63, 64, 65, 100, 129, 200, 255, 256, 300, 400, 480, CTX - 9):
        p = rng.integers(1, V, L).tolist()
        e.generate([p], max_new=min(8, CTX - L))
        e.generate([p], max_new=min(8, CTX - L), samp=[dict(SAMP, seed=L)])
    p = rng.integers(1, V, 300).tolist()
    e.generate([p], max_new=CTX - 300, samp=[SAMP])                     # decode through every window up to CTX
    e.generate([p[:20]], max_new=CTX - 20)
    for n in range(1, S + 1):                                          # 1..S rows, mixed lengths, greedy and sampled
        ps = [rng.integers(1, V, int(L)).tolist() for L in rng.integers(1, 300, n)]
        e.generate(ps, max_new=12)
        e.generate(ps, max_new=12, samp=[SAMP if i % 2 else None for i in range(n)])
    conv = rng.integers(1, V, 30).tolist()                             # a chat: each turn resends the whole conversation
    for t in range(4):
        r = e.generate([conv], max_new=40, samp=[SAMP], marks=[[0, len(conv) - 5]])
        conv = conv + r["ids"][0] + rng.integers(1, V, 25).tolist()
    other = rng.integers(1, V, 50).tolist()                             # another conversation takes the slot (host-cache save)
    e.generate([other], max_new=8)
    r = e.generate([conv], max_new=8)                                  # and the chat comes back (restore or re-prefill)
    assert sizes(e) == s0, (s0, sizes(e))


def test_prefill_shapes_cover_prefill():
    """Every program prefill() runs is in prefill_shapes(), for random groups of rows, starts and lengths."""
    c = types.SimpleNamespace(V=V, H=48)
    for CTX, S, tpc, attn in ((2048, 8, 2048, 2e9), (32768, 1, 2048, 5e8), (32768, 8, 2048, 5e8), (512, 2, 2048, 2e9)):
        e = Engine.__new__(Engine)
        e.c, e.CTX, e.S, e.chunk, e.tokens_per_call, e.attn_bytes, e.shapes, e.DK = c, CTX, S, 512, tpc, attn, [], 0
        shapes = set(e.prefill_shapes())
        seen = []

        def fwd(W, ck, cv, toks, p0, lens, sl, Tc, verify=False, hidden=False):
            B = toks.shape[0]
            seen.append((B, toks.shape[1], Tc))
            return dict(ck=ck, cv=cv, nxt=np.zeros(B, np.int32), lg=np.zeros((B, 1), np.float32))
        e.fwd, e.W = fwd, None
        rng = np.random.default_rng(CTX + S)
        for _ in range(300):
            n = int(rng.integers(1, S + 1))
            ends = [int(x) for x in rng.integers(1, CTX, n)]
            starts = [int(rng.integers(0, x)) if rng.random() < 0.7 else 0 for x in ends]
            e.prefill((None, None), [list(range(x)) for x in ends], list(range(n)), starts, ends)
        assert set(seen) <= shapes, sorted(set(seen) - shapes)[:5]


def test_state_committed_on_one_device():
    c = types.SimpleNamespace(V=V, H=2)
    e = Engine(c, {"w": jnp.ones((), jnp.float32)}, MODEL, CTX=256, S=2, K=4)
    assert e.dev is not None and all(getattr(a, "committed", getattr(a, "_committed", None)) for a in jax.tree.leaves(e.new_state()))
