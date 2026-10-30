"""The DFlash 2 drafter (runtime/drafters/dflash.py) inside the engine's batched step, CPU, on the tiny random models of
the Qwen3.8-27B architecture from test_spec_in_step.py and a tiny random DFlash 2 drafter (2 layers, block 8, three taps,
sliding window 32 so the ring wraps):
  taps    forward(taps=...) returns the tapped layers' outputs; the program without taps is unchanged
  G-D1    one user alone, greedy, drafter on (k = 7, verify width 8): every committed token, teacher-forced through the plain
          forward, is its argmax or within one bf16 step of its top logit; the near-tie count is recorded
  exact   on the near-bigram model (no ties): drafter on == off token for token
  ring    after a run, the ring holds exactly the committed positions of the last W (tags), and its K/V equal the ring
          rebuilt from one plain forward over prompt + answer (taps from the plain forward, not the verify pass)
  resume  prefix cache: a conversation moved to a lower slot reads its own ring (it moves with the slot's state); one
          restored from host memory into a slot that held another conversation holds only what it wrote itself
"""
import json

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
jax = pytest.importorskip("jax")
pytest.importorskip("safetensors")
import jax.numpy as jnp  # noqa: E402

from mashbox.models import qwen3_8_27b as Q  # noqa: E402
from mashbox.models.qwen3_8_27b import model as QM  # noqa: E402
from mashbox.runtime.drafters import dflash as DF  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402
from test_spec_in_step import D, V, drive, prompts, tie_aware_replay, tiny  # noqa: E402,F401  (tiny: the fixture)

bf = ml_dtypes.bfloat16
TAPS, DL, DF_, DH, DKV, DHD, SW = (1, 3, 6), 2, 96, 4, 2, 16, 32


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    from safetensors.numpy import save_file
    d = tmp_path_factory.mktemp("dflash_tiny")
    json.dump(dict(num_hidden_layers=DL, hidden_size=D, intermediate_size=DF_, num_attention_heads=DH, num_key_value_heads=DKV,
                   head_dim=DHD, vocab_size=V, rms_norm_eps=1e-6, rope_theta=10000.0, is_causal=False,
                   layer_types=["sliding_attention"] * DL, sliding_window=SW,
                   dflash_config=dict(block_size=8, mask_token_id=V - 1, target_layer_ids=list(TAPS), conv_kernel_size=2,
                                      conv_group_size=16, selector_rank=8, selector_top_k=4)), open(d / "config.json", "w"))
    rng = np.random.default_rng(3)
    r = lambda *s, sd=0.1: rng.normal(0, sd, s).astype(bf)
    T = {"fc.weight": r(D, len(TAPS) * D), "hidden_norm.weight": r(D, sd=1.0), "norm.weight": r(D, sd=1.0),
         "candidate_selector.hidden_projection.weight": r(8, D), "candidate_selector.predecessor_codebook": r(V, 8, sd=1.0),
         "candidate_selector.successor_codebook": r(V, 8, sd=1.0)}
    for i in range(DL):
        p = f"layers.{i}."
        T.update({p + "self_attn.q_proj.weight": r(DH * DHD, D), p + "self_attn.k_proj.weight": r(DKV * DHD, D),
                  p + "self_attn.v_proj.weight": r(DKV * DHD, D), p + "self_attn.o_proj.weight": r(D, DH * DHD),
                  p + "self_attn.q_norm.weight": r(DHD, sd=1.0), p + "self_attn.k_norm.weight": r(DHD, sd=1.0),
                  p + "input_layernorm.weight": r(D, sd=1.0), p + "post_attention_layernorm.weight": r(D, sd=1.0),
                  p + "mlp.gate_proj.weight": r(DF_, D), p + "mlp.up_proj.weight": r(DF_, D), p + "mlp.down_proj.weight": r(D, DF_),
                  p + "attention_conv.base_kernel": r(2, 2, D), p + "attention_conv.kernel_projection.weight": r(2 * 2 * D // 16, D),
                  p + "mlp_conv.base_kernel": r(2, 2, D), p + "mlp_conv.kernel_projection.weight": r(2 * 2 * D // 16, D)})
    save_file(T, str(d / "model.safetensors"))
    return str(d)


def engines(t, ddir, k=7, **kw):
    cfg, W, _ = t
    return Engine(cfg, W, Q, **kw), Engine(cfg, W, Q, drafter=DF.DFlash(Q, cfg, ddir, k), draft_k=k, spec_rounds=3, **kw)


def test_taps_are_the_layer_outputs_and_change_nothing_else(tiny):  # noqa: F811
    cfg, W, _ = tiny["random"]
    fwd = jax.jit(Q.make_forward(cfg), static_argnames=("Tc", "contig", "verify", "hidden", "taps"))
    toks = jnp.asarray(np.random.default_rng(1).integers(1, V, (2, 9)), jnp.int32)
    p0, lens, sl = jnp.zeros(2, jnp.int32), jnp.full(2, 9, jnp.int32), jnp.arange(2, dtype=jnp.int32)
    ck, cv = Q.new_state(cfg, 2, 64)
    a = fwd(W, ck, cv, toks, p0, lens, sl, Tc=0)
    ck, cv = Q.new_state(cfg, 2, 64)
    b = fwd(W, ck, cv, toks, p0, lens, sl, Tc=0, taps=TAPS)
    assert b["taps"].shape == (2, 9, len(TAPS) * D)
    np.testing.assert_array_equal(np.asarray(a["lg"]), np.asarray(b["lg"]))
    ck, cv = Q.new_state(cfg, 2, 64)
    c = fwd(W, ck, cv, toks, p0, lens, sl, Tc=0, taps=(cfg.L - 1,))
    xn = QM.rms_norm(c["taps"], W["norm"], cfg.eps)      # the last layer's output, normed, is the head's input
    np.testing.assert_array_equal(np.asarray(QM.linear(xn[:, -1], W["head"]).astype(jnp.float32)), np.asarray(a["lg"]))


@pytest.mark.parametrize("tag", ["random", "bigram"])
def test_gd1_dflash_tie_aware_replay(tiny, drafter_dir, tag, record_property):  # noqa: F811
    base, spec = engines(tiny[tag], drafter_dir, CTX=256, S=4, K=4)
    n = near = over = 0
    for p in prompts(4):
        got = drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"]
        assert len(got) == 40
        a, b = tie_aware_replay(base, p, got)
        n, near, over = n + len(got), near + a, over + b
    record_property("gd1_near_tie", near)
    print(f"DFlash G-D1 {tag}: {n} tokens, near-tie {near}, beyond one bf16 step {over}, rounds {spec.spec_stats['rounds']}, "
          f"hist {spec.spec_stats['hist']}")
    assert over == 0, (tag, n, near, over)


def test_dflash_bigram_on_equals_off(tiny, drafter_dir):  # noqa: F811
    base, spec = engines(tiny["bigram"], drafter_dir, CTX=256, S=4, K=4)
    for p in prompts(4, seed=9):
        want = base.generate([p], max_new=40)["ids"][0]
        assert drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"] == want


def test_dflash_ring_holds_the_committed_positions(tiny, drafter_dir):  # noqa: F811
    cfg, W, _ = tiny["bigram"]
    base, spec = engines(tiny["bigram"], drafter_dir, CTX=256, S=4, K=4)
    p = prompts(1, seed=21, lo=40, hi=41)[0]
    got = drive(spec, [(0, dict(ids=p, max_new=37))])[0]
    dr, R = spec.drafter, jax.tree.map(np.asarray, spec.dstate)
    ids = p + got["ids"]
    # written so far: every position but the last round's commit, which waits in pt/pp/pn for the next draft
    upto = int(R["pp"][0]) if int(R["pn"][0]) else len(ids) - 1
    tags = sorted(int(t) for t in R["pos"][0] if t >= 0)
    assert tags == list(range(max(0, upto - SW), upto)), (tags[:3], tags[-3:], upto)
    fwd = jax.jit(Q.make_forward(cfg), static_argnames=("Tc", "contig", "verify", "hidden", "taps"))
    ck, cv = Q.new_state(cfg, 1, 256)
    t = fwd(W, ck, cv, jnp.asarray([ids[:upto]], jnp.int32), jnp.zeros(1, jnp.int32), jnp.asarray([upto], jnp.int32),
            jnp.zeros(1, jnp.int32), Tc=0, taps=TAPS)["taps"]
    ref = DF.ring_write(dr.W, dr.dc, jax.tree.map(lambda a: a[:1], DF.new_ring(dr.dc, 1)), jnp.zeros(1, jnp.int32), t,
                        jnp.zeros(1, jnp.int32), jnp.asarray([upto], jnp.int32))
    np.testing.assert_array_equal(np.asarray(ref["pos"][0]), R["pos"][0])
    # the ring was written from verify-width taps; the reference from one prefill-width pass: equal up to bf16 rounding
    np.testing.assert_allclose(np.asarray(ref["k"][0], np.float32), R["k"][0].astype(np.float32), rtol=0.05, atol=0.05)
    np.testing.assert_allclose(np.asarray(ref["v"][0], np.float32), R["v"][0].astype(np.float32), rtol=0.05, atol=0.05)


def test_dflash_warm_then_serve_on_equals_off(tiny, drafter_dir):  # noqa: F811
    """Engine.warm traces the speculation programs in several threads at once: no drafter state may pass from draft to
    after_commit through the object (a value kept on self in one trace reached another, UnexpectedTracerError on chip,
    2026-10-30). Warm, then serve: still equal to plain."""
    base, spec = engines(tiny["bigram"], drafter_dir, CTX=256, S=4, K=4)
    spec.warm()
    p = prompts(1, seed=5)[0]
    assert drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"] == base.generate([p], max_new=40)["ids"][0]


def _ring(eng, s, ids):
    """-> (positions the ring of slot s holds, the end of what is written, the state); the last round's commit waits in
    pt/pp/pn for the next draft"""
    R = jax.tree.map(np.asarray, eng.dstate)
    upto = int(R["pp"][s]) if int(R["pn"][s]) else len(ids) - 1
    return sorted(int(t) for t in R["pos"][s] if t >= 0), upto, R


def test_dflash_ring_after_host_restore(tiny, drafter_dir):  # noqa: F811
    """One slot: A, then B takes the slot (A goes to host memory), then A's prompt plus 5 tokens is restored into the slot
    B left. The host copy has no ring, so the ring holds exactly the positions this request wrote, none of B's."""
    base, spec = engines(tiny["bigram"], drafter_dir, CTX=256, S=1, K=4, prefix_cache=True)
    pa, pb = prompts(2, seed=31, lo=40, hi=41)
    spec.generate([pa], max_new=10)
    spec.generate([pb], max_new=10)
    pc = pa + prompts(1, seed=32, lo=5, hi=6)[0]
    got = spec.generate([pc], max_new=8)
    st = got["restored"][0]
    assert st > 0, got
    tags, upto, _ = _ring(spec, 0, pc + got["ids"][0])
    assert tags == list(range(st, upto)), (st, tags[:3], tags[-3:], upto)
    assert got["ids"][0] == base.generate([pc], max_new=8)["ids"][0]


def test_dflash_ring_follows_a_swapped_slot(tiny, drafter_dir):  # noqa: F811
    """Six conversations fill slots; the one in the highest slot is resumed alone, so the engine moves its state to slot 0
    (the smallest batch bucket). Its ring moves too: slot 0's ring then equals the ring rebuilt from one plain forward over
    its tokens (the other conversation's ring would not)."""
    cfg, W, _ = tiny["bigram"]
    base, spec = engines(tiny["bigram"], drafter_dir, CTX=256, S=8, K=4, prefix_cache=True)
    spec.generate(prompts(6, seed=41, lo=40, hi=41), max_new=1)
    hi_ = max(s for s in range(8) if spec.snaps[s])
    pa = list(spec.snaps[hi_][-1]["ids"])
    assert hi_ > 0 and len(pa) == 40
    pc = pa + prompts(1, seed=42, lo=5, hi=6)[0]
    got = spec.generate([pc], max_new=8)
    assert got["cached"][0] == len(pa) and got["restored"][0] == 0, got
    assert spec.snaps[0] and spec.snaps[0][-1]["ids"][:len(pa)] == tuple(pa)        # it runs in slot 0 now
    ids = pc + got["ids"][0]
    tags, upto, R = _ring(spec, 0, ids)
    assert tags == list(range(max(0, upto - SW), upto)), (tags[:3], tags[-3:], upto)
    dr = spec.drafter
    fwd = jax.jit(Q.make_forward(cfg), static_argnames=("Tc", "contig", "verify", "hidden", "taps"))
    ck, cv = Q.new_state(cfg, 1, 256)
    t = fwd(W, ck, cv, jnp.asarray([ids[:upto]], jnp.int32), jnp.zeros(1, jnp.int32), jnp.asarray([upto], jnp.int32),
            jnp.zeros(1, jnp.int32), Tc=0, taps=TAPS)["taps"]
    ref = DF.ring_write(dr.W, dr.dc, jax.tree.map(lambda a: a[:1], DF.new_ring(dr.dc, 1)), jnp.zeros(1, jnp.int32), t,
                        jnp.zeros(1, jnp.int32), jnp.asarray([upto], jnp.int32))
    np.testing.assert_array_equal(np.asarray(ref["pos"][0]), R["pos"][0])
    np.testing.assert_allclose(np.asarray(ref["k"][0], np.float32), R["k"][0].astype(np.float32), rtol=0.05, atol=0.05)
    np.testing.assert_allclose(np.asarray(ref["v"][0], np.float32), R["v"][0].astype(np.float32), rtol=0.05, atol=0.05)
    assert got["ids"][0] == base.generate([pc], max_new=8)["ids"][0]
