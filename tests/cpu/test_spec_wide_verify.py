"""Wide verify (block drafters: DFlash 2 drafts 7 tokens, so its verify pass is 8 positions wide) on tiny random models of
the Qwen3.8-27B architecture, CPU. The same engine step as test_spec_in_step.py (Engine._spec_step, runtime/spec_step.py),
at k = 4 and 7 (verify widths 5 and 8) with the MTP drafter, which drafts any k:
  G-D1  one user alone, greedy: every committed token, teacher-forced through the plain forward, is its argmax or within
        one bf16 step of its top logit (the tie-aware replay); the near-tie count is recorded and printed
  exact on the near-bigram model (no ties): speculation on == off token for token, with long commits really accepted
  roll  on the near-bigram model, the slot after speculation equals plain decode of the same tokens (linear-attention
        state, conv tail, full-attention K/V at every committed position): rejected positions leave no trace at width 8
"""
import numpy as np
import pytest

pytest.importorskip("ml_dtypes")
pytest.importorskip("jax")

from test_spec_in_step import drive, engines, prompts, tie_aware_replay, tiny  # noqa: E402,F401  (tiny: the fixture)


@pytest.mark.parametrize("tag", ["random", "bigram"])
@pytest.mark.parametrize("k", [4, 7])
def test_gd1_wide_verify_tie_aware_replay(tiny, tag, k, record_property):  # noqa: F811
    base, spec = engines(tiny[tag], k, CTX=256, S=4, K=4)
    n = near = over = 0
    for p in prompts(4):
        got = drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"]
        assert len(got) == 40
        a, b = tie_aware_replay(base, p, got)
        n, near, over = n + len(got), near + a, over + b
    tpr = spec.spec_stats["tokens"] / max(1, spec.spec_stats["rounds"])
    record_property("gd1_tokens", n)
    record_property("gd1_near_tie", near)
    record_property("gd1_beyond_one_bf16_step", over)
    record_property("tokens_per_round", round(tpr, 3))
    print(f"wide G-D1 {tag} k={k}: {n} tokens, near-tie {near}, beyond one bf16 step {over}, tokens/round {tpr:.3f}")
    assert over == 0, (tag, k, n, near, over)


@pytest.mark.parametrize("k", [4, 7])
def test_wide_verify_bigram_on_equals_off(tiny, k):  # noqa: F811
    base, spec = engines(tiny["bigram"], k, CTX=256, S=4, K=4)
    for p in prompts(4):
        want = base.generate([p], max_new=40)["ids"][0]
        assert drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"] == want
    tpr = spec.spec_stats["tokens"] / spec.spec_stats["rounds"]
    print(f"wide on==off bigram k={k}: tokens/round {tpr:.3f}, hist {spec.spec_stats['hist']}")
    assert tpr > 2.5, tpr                    # long commits ran (widths past 4 really used)
    assert sum(spec.spec_stats["hist"][5:]) > 0, spec.spec_stats["hist"]


def test_wide_verify_rollback_leaves_no_trace(tiny):  # noqa: F811
    cfg = tiny["bigram"][0]
    base, spec = engines(tiny["bigram"], 7, CTX=256, S=4, K=1, prefix_cache=True, cache_ram=0)
    p = prompts(1, seed=21)[0]
    got = drive(spec, [(0, dict(ids=p, max_new=37))])[0]
    want = drive(base, [(0, dict(ids=p, max_new=37))], force=got["ids"])[0]
    assert got["ids"] == want["ids"] and got["slot"] == want["slot"] == 0
    n = len(p) + 37 - 1
    (kb, vb), (rb, cb) = base.state
    (ks, vs), (rs, cs) = spec.state
    G = cfg.NG
    np.testing.assert_array_equal(np.asarray(ks[:G, 0, :n]), np.asarray(kb[:G, 0, :n]))
    np.testing.assert_array_equal(np.asarray(vs[:G, 0, :n]), np.asarray(vb[:G, 0, :n]))
    np.testing.assert_allclose(np.asarray(rs[:, :, 0]), np.asarray(rb[:, :, 0]), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(cs[:, :, 0]), np.asarray(cb[:, :, 0]))
