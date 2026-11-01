"""The context plan (runtime/context.py, cli.plan_context): the engine says what context this chip holds for a model and
runs the longest that fits instead of a fixed 2,048. Guards: a default that runs out of chip memory, a plan past the
model's trained context, a context that ignores the slot count, and a silent change of an explicit --ctx/--slots."""
import json

import pytest

from mashbox import cli
from mashbox.runtime import context as CX

GB = 1e9
LIMIT = 33.546e9                    # TPU v6e-1 bytes_limit (MEASURED, JAX memory_stats)


def state(kv_per_token, fixed_per_row=0.0):
    """slot state as the engine allocates it: S rows at S <= 2, S + 1 rows above (scratch); 2+ slots: swap copy"""
    def f(s, c):
        rows = s if s <= 2 else s + 1
        b = rows * (kv_per_token * c + fixed_per_row)
        return 2 * b if s >= 2 else b
    return f


def test_parse():
    assert CX.parse("auto") == "auto" and CX.parse("MAX") == "max"
    assert CX.parse("32768") == 32768 and CX.parse("32k") == 32768 and CX.parse("131_072") == 131072
    for bad in ("0", "-5", "lots", ""):
        with pytest.raises(ValueError):
            CX.parse(bad)


def test_every_plan_fits_the_chip_and_the_trained_context():
    """Whatever is chosen, its need (weights + state + working memory) is inside the limit and its context inside the
    trained context, for a grid of model sizes, K/V widths and trained contexts."""
    for fixed in (0.5 * GB, 5.7 * GB, 16.5 * GB, 26.94 * GB):
        for kv in (12288, 32768, 65536, 114688, 196608):
            for trained in (8192, 32768, 131072, 262144):
                fit = CX.Fit(state(kv, 1.5e8), fixed, LIMIT, trained)
                for want in ("auto", "max"):
                    try:
                        ctx, slots, _ = fit.choose(want)
                    except ValueError:
                        assert fit.longest(1) == 0
                        continue
                    assert ctx <= trained and ctx % CX.STEP == 0 and 1 <= slots <= 8
                    assert fit.need(slots, ctx) <= LIMIT
                    if want == "max":
                        assert slots == 1 and ctx == fit.longest(1)
                        assert ctx == trained or fit.need(1, ctx + CX.STEP) > LIMIT    # the longest, not just one that fits


def test_auto_prefers_slots_only_above_the_agent_context():
    """auto takes the most slots whose context is at least min(32768, trained), else one slot at its longest"""
    big = CX.Fit(state(65536, 1.539e8), 26.94 * GB, LIMIT, 262144)            # Qwen3.8-27B BQ8_R on one v6e chip
    ctx, slots, _ = big.choose("auto")
    assert slots == 1 and ctx == big.longest(1) and ctx > 32768
    small = CX.Fit(state(12288), 0.5 * GB, LIMIT, 32768)                      # a 0.5B model, trained 32k
    assert small.choose("auto")[:2] == (32768, 8)
    mid = CX.Fit(state(114688), 2.2 * GB, LIMIT, 131072)                      # Llama-3.2-3B-like K/V
    ctx, slots, _ = mid.choose("auto")
    assert ctx >= 32768 and slots > 1 and (slots == 8 or mid.longest(slots * 2) < 32768)


def test_explicit_ctx_takes_the_most_slots_that_fit_and_refuses_what_cannot():
    fit = CX.Fit(state(65536, 1.539e8), 26.94 * GB, LIMIT, 262144)
    ctx, slots, _ = fit.choose(16384)
    assert ctx == 16384 and fit.need(slots, 16384) <= LIMIT and (slots == 8 or fit.need(slots * 2, 16384) > LIMIT)
    with pytest.raises(ValueError, match="--ctx max"):
        fit.choose(262144)


def test_plan_context_sets_numbers_and_logs_what_the_chip_holds(capsys):
    """cli.plan_context: --ctx auto -> numbers on the args, one JSON event and one plain line naming the one-request
    maximum, the trained context and the setting; an explicit --ctx N --slots M is kept as given; with no memory figure
    (CPU) auto is 2,048 x 8, as before; past the trained context it warns"""
    p = cli.build_parser()
    ns = lambda *x: p.parse_args(["run", "m", *x])
    new_state = lambda s, c: __import__("jax").numpy.zeros((2, s + 1, c, 1024), "bfloat16")   # 4 KiB per token per row
    a = ns()
    assert a.ctx == "auto" and a.slots is None
    fit = cli.plan_context(a, new_state, 20 * GB, 65536, "m", limit=LIMIT)
    out = capsys.readouterr().err + capsys.readouterr().out
    assert isinstance(a.ctx, int) and isinstance(a.slots, int) and a.ctx <= 65536
    ev = [json.loads(x) for x in out.splitlines() if x.startswith("{") and '"context"' in x]
    assert ev and ev[-1]["ctx"] == a.ctx and ev[-1]["slots"] == a.slots and ev[-1]["trained"] == 65536
    assert "for one request" in out and "trained context 65,536" in out and "--ctx" in out
    assert fit.need(a.slots, a.ctx) <= LIMIT
    a = ns("--ctx", "4096", "--slots", "3")
    cli.plan_context(a, new_state, 20 * GB, 65536, "m", limit=LIMIT)
    assert (a.ctx, a.slots) == (4096, 3)
    a = ns()
    assert cli.plan_context(a, new_state, 20 * GB, 65536, "m", limit=0) is None and (a.ctx, a.slots) == (2048, 8)
    capsys.readouterr()
    a = ns("--ctx", "100000")
    cli.plan_context(a, new_state, 1 * GB, 65536, "m", limit=LIMIT)
    out = capsys.readouterr()
    assert a.ctx == 100000 and "longer than m's trained context" in out.err + out.out


def test_a_drafter_keeps_the_old_default(capsys):
    """the plan does not count a drafter's memory: with one asked for, auto stays 2,048 unless --ctx N"""
    p = cli.build_parser()
    a = p.parse_args(["run", "m", "--speculation", "ngram"])
    new_state = lambda s, c: __import__("jax").numpy.zeros((1, s + 1, c, 8), "bfloat16")
    assert cli.plan_context(a, new_state, 1 * GB, 65536, "m", limit=LIMIT) is None and a.ctx == 2048
