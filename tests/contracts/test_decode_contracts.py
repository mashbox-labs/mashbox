"""The decode contracts: what a `--decode fast` or `--decode long` record on a chip stands on, so a drafter approved once
on a model and chip carries across releases until one of these moves (tools/contracts.py decode_contracts).

  spec/block   the causal block drafter inside the engine's batched step, on the tiny random models of the
               Qwen3.8-27B architecture (tests/cpu/test_spec_in_step.py) and a tiny random block drafter
               (tests/cpu/test_block_drafter.py)
  fit/plan     the context plan with a drafter counted (runtime/context.py Fit, drafters/block.py plan_bytes, cli.plan_context)
  kv/int8      the 8-bit K/V store (runtime/kvcache.py pack/unpack): the record a long-context passkey run names

G-D1 reference rule: the teacher-forced replay that judges a run uses the same prefill chunk as the
run under test, for every run and every chunk; no per-prompt choice. The replay forward's own result moves with the call
shape (measured on the tiny random model: two bf16 steps between chunk 64/512 and chunk 16/1 at one near tie), so a
replay at another chunk would judge the reference's numerics, not the run. tests/cpu/test_block_drafter.py records that
movement across chunks 512, 128, 16 and 1 as a documented numerics property.

Bars, written before the first run:
  spec/block   1. G-D1 tie-aware, greedy, k 7, prompts(4) of seed 5, 40 tokens each, on the random and the near-bigram
                  model: every committed token, teacher-forced through the plain forward, is its argmax or within one
                  bf16 step of its top logit (0 beyond).
               2. On the near-bigram model (no ties) drafter on == off token for token, bf16 and int8 drafter weights.
               3. Four users joining at steps 0..3 on the random model: G-D1 as bar 1 for every row.
               4. Seed 33 (prompts(2, seed=33)), bar 1 with bf16 and int8 drafter weights, run and replay at chunk
                  512: a real near-tie whose result depends on the CPU (an Intel Core i5-13500 misses by 1 at 512
                  and 128, an AMD EPYC 7502P passes at every chunk), kept as a non-strict xfail
                  (tests/cpu/test_block_drafter.py SEED33); a known failure counts as FAIL here, so spec/block is FAIL
                  on every CPU until the defect is resolved.
  fit/plan     5. plan_bytes of the tiny drafter (bf16 and int8 weights) and its state at (1, 1024), (2, 4096) and
                  (4, 8192) are the integers they are; with a stand-in target state of 1,000 bytes per slot-position and
                  a 10 GB limit, the plan with the drafter is shorter than without it, and an explicit context past
                  the longest that fits is refused before load.
  kv/int8      6. pack/unpack on a fixed K/V grid (values k / 64, k = -300 .. 299, rounded to bf16): unpack(pack(x))
                  restores the codes and scales, every value dequantizes within half a code step (s / 2) of x, a head
                  of zeros stays zeros.
Recorded (exact): spec/block: METHOD, IMPL, DRAFTER_ABI, SPEC_STEP_VERSION and the near-bigram committed ids (no
ties there: bar 2 holds them equal to plain decode). fit/plan: every integer of bar 5. kv/int8: the packed K and V rows'
bytes. Float logits are checked by their bars and never recorded.
"""
import argparse
import os
import sys

import numpy as np
import pytest

pytest.importorskip("ml_dtypes")
jax = pytest.importorskip("jax")
pytest.importorskip("safetensors")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpu"))

from mashbox.models import qwen3_8_27b as Q  # noqa: E402
from mashbox.runtime.drafters import DRAFTER_ABI  # noqa: E402
from mashbox.runtime.drafters import block as B  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402
from mashbox.runtime.spec_step import SPEC_STEP_VERSION  # noqa: E402
from test_block_drafter import CTX, SEED33, drafter_dir  # noqa: E402,F401
from test_spec_in_step import drive, prompts, tie_aware_replay, tiny  # noqa: E402,F401


def _pair(t, ddir, wf="bf16"):
    cfg, W, _ = t
    return (Engine(cfg, W, Q, CTX=CTX, S=4, K=4),
            Engine(cfg, W, Q, drafter=B.Block(Q, cfg, ddir, 7, window=CTX, weights=wf), draft_k=7, spec_rounds=3,
                   CTX=CTX, S=4, K=4))


def _ident(c):
    for x in (B.METHOD, B.IMPL, str(DRAFTER_ABI), str(SPEC_STEP_VERSION)):
        c.record(x)


@pytest.mark.contract
@pytest.mark.parametrize("tag", ["random", "bigram"])
def test_spec_block_gd1(contract, tiny, drafter_dir, tag):  # noqa: F811
    c = contract("spec/block")
    _ident(c)
    base, spec = _pair(tiny[tag], drafter_dir)
    over = 0
    for p in prompts(4):
        got = drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"]
        assert len(got) == 40
        over += tie_aware_replay(base, p, got)[1]
    assert over == 0, (tag, over)


@pytest.mark.contract
@pytest.mark.parametrize("wf", ["bf16", "int8"])
def test_spec_block_bigram_on_equals_off(contract, tiny, drafter_dir, wf):  # noqa: F811
    c = contract("spec/block")
    base, spec = _pair(tiny["bigram"], drafter_dir, wf)
    for p in prompts(4):
        got = drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"]
        assert got == base.generate([p], max_new=40)["ids"][0]
        c.record(np.asarray(got, np.int32))
    assert spec.spec_stats["rounds"] > 0


@pytest.mark.contract
def test_spec_block_four_users_gd1(contract, tiny, drafter_dir):  # noqa: F811
    contract("spec/block")
    base, spec = _pair(tiny["random"], drafter_dir)
    ps = prompts(4, seed=17)
    got = drive(spec, [(i, dict(ids=p, max_new=40)) for i, p in enumerate(ps)])
    assert sum(tie_aware_replay(base, p, g["ids"])[1] for p, g in zip(ps, got)) == 0


@pytest.mark.contract
@pytest.mark.xfail(strict=False, reason=SEED33)
@pytest.mark.parametrize("wf", ["bf16", "int8"])
def test_spec_block_seed33_known_miss(contract, tiny, drafter_dir, wf):  # noqa: F811
    contract("spec/block")
    base, spec = _pair(tiny["random"], drafter_dir, wf)
    over = 0
    for p in prompts(2, seed=33):
        over += tie_aware_replay(base, p, drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"])[1]
    assert over == 0


@pytest.mark.contract
def test_fit_plan_counts_the_drafter(contract, drafter_dir, monkeypatch):  # noqa: F811
    c = contract("fit/plan")
    from mashbox import cli
    from mashbox.runtime import context as CX
    for wf in ("bf16", "int8"):
        pb = B.plan_bytes(drafter_dir, wf)
        c.record(np.asarray([int(pb["fixed"])] + [int(pb["state"](s, n)) for s, n in ((1, 1024), (2, 4096), (4, 8192))],
                            np.int64))
    monkeypatch.setattr(CX, "eval_state", lambda ns, s, n: 1000 * s * n)
    ns = lambda **kw: argparse.Namespace(**{**dict(ctx="auto", slots=None, speculation=None, drafter_dir=None, mtp=None,
                                                   draft_model=None), **kw})
    f0 = cli.plan_context(ns(), None, 1e9, 0, limit=10e9)
    f1 = cli.plan_context(ns(speculation="block", drafter_dir=drafter_dir), None, 1e9, 0, limit=10e9)
    assert 0 < f1.longest(1) < f0.longest(1)
    c.record(np.asarray([f0.longest(1), f1.longest(1), f0.longest(4), f1.longest(4)], np.int64))
    with pytest.raises(SystemExit, match="does not fit"):
        cli.plan_context(ns(ctx=f1.longest(1) + 64 * 1024, slots=1, speculation="block", drafter_dir=drafter_dir),
                         None, 1e9, 0, limit=10e9)
    c.record("refused")


@pytest.mark.contract
def test_kv_int8_pack_round_trip(contract):
    import jax.numpy as jnp
    import ml_dtypes
    from mashbox.runtime import kvcache as KV
    c = contract("kv/int8")
    kv, hd = 2, 16
    x = (np.arange(-300, 300, dtype=np.float32) / 64).astype(ml_dtypes.bfloat16)
    k = jnp.asarray(x[:3 * kv * hd].reshape(3, kv, hd))
    v = jnp.asarray(x[-3 * kv * hd:].reshape(3, kv, hd)).at[1, 0].set(0)
    kr, vr = KV.pack(k, v, kv, hd)
    (ck, sk), (cv, sv) = KV.unpack(kr, vr, kv, hd)
    for codes, s, ref in ((ck, sk, k), (cv, sv, v)):
        back = np.asarray(KV.dequant(codes, s, jnp.float32))
        err = np.abs(back - np.asarray(ref, np.float32))
        assert np.all(err <= np.asarray(s)[..., None] / 2 + 1e-7), float(err.max())
    assert not np.any(np.asarray(cv)[1, 0]) and float(np.asarray(sv)[1, 0]) == 0.0
    c.record(np.asarray(kr))
    c.record(np.asarray(vr))
