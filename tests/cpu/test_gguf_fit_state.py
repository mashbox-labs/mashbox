"""GGUF zero-flag slot fit: moving the slot state takes scratch of the state and one more row (jit_swap asked for 5.00 GiB,
10 rows, with a state of 9 rows), so besides weights + state inside the headroom, weights + state + that scratch must stay
inside the chip's limit.

On one v6e chip, Qwen3-32B Q4_K_M at the default 8 slots was inside the headroom (weights 25.71 GB as the fit estimates
them with the kernel package, + state 4.83 GB) and then failed to load jit_swap ("reserve 5.00G ... 2.79G free"); it ran
with --slots 4. On one v5e chip, Qwen3-14B Q4_K_M at 8 slots needed 15.96 GiB in jit_forward. Inputs: each file's own
metadata (layers, K/V heads, head size) for the state, through need_bytes itself; for the weights, the estimate need_bytes
gave on the real Qwen3-32B file with the kernel package, and for the other files the kernel planes their runs loaded (a
lower bound of the estimate, which also counts the blocks kept off the kernels); the chip limits are gguf.CHIPS."""
import importlib.util
import os
import types

import pytest

import gguf  # noqa: E402,F401  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jnp = pytest.importorskip("jax.numpy")

from mashbox import gguf as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

V5E, V6E = (b for _, b in G.CHIPS[:2])

# file: (block_count, head_count_kv, head size), weights in bytes
FILES = {
    "Qwen3-32B-Q4_K_M": ((64, 8, 128), 25_712_930_816),              # need_bytes on the real file, kernel package on
    "Qwen3-14B-Q4_K_M": ((40, 8, 128), 11_158_814_720),              # planes loaded (lower bound)
    "Llama-3.1-8B-Instruct-Q4_K_M": ((32, 8, 128), 6_011_486_208),   # planes loaded (lower bound)
    "Llama-3.2-1B-Instruct-Q4_K_M": ((16, 8, 64), 765_460_480),      # planes loaded (lower bound)
    "Qwen2.5-1.5B-Instruct-q8_0": ((28, 2, 128), 1_748_901_888),     # planes loaded (lower bound)
}


def _as_file(monkeypatch, name):
    """need_bytes for one of FILES: its weights, need_bytes' own K/V state for its shape, no float32 scratch (every
    multiplied weight of these files has a kernel)"""
    (L, KV, HD), weights = FILES[name]
    c = types.SimpleNamespace(act=jnp.bfloat16, NG=L, FI=1, KV=KV, HD=HD, HV=1, DK=1, DV=1, CK=2, CONV=1)
    real = G.need_bytes

    def need_bytes(f, c_, ctx, slots, kernel_types=frozenset(), roles=None):
        return weights, real(None, c, ctx, slots, roles=((), ()))[1], 0
    monkeypatch.setattr(G, "need_bytes", need_bytes)
    return c


def _scratch(st, slots):
    """the scratch moving the state takes: the state (slots + 1 rows) and one more row"""
    return st // (slots + 1) * (slots + 2)


def test_the_scratch_is_what_jit_swap_asked_for(monkeypatch):
    c = _as_file(monkeypatch, "Qwen3-32B-Q4_K_M")
    w, st, big = G.need_bytes(None, c, 2048, 8)
    assert st == 9 * 2**29                                                 # 4.83 GB: 9 rows of 512 MiB
    assert G.slot_need(None, c, 2048, 8)[1] - (w + big + st) == 5 * 2**30  # "Attempting to reserve 5.00G"


@pytest.mark.parametrize("name,lim,want", [
    ("Qwen3-32B-Q4_K_M", V6E, 4),        # 8 slots: jit_swap could not load; --slots 4 ran
    ("Qwen3-14B-Q4_K_M", V5E, 4),        # 8 slots: jit_forward over the chip
])
def test_zero_flags_pick_fewer_slots_when_the_state_and_its_scratch_do_not_fit(monkeypatch, name, lim, want):
    c = _as_file(monkeypatch, name)
    w, st, big = G.need_bytes(None, c, 2048, 8)
    assert w + big + st <= G.HEADROOM * lim < lim < w + big + st + _scratch(st, 8)   # why 8 looked fine, and was not
    assert G.fit_slots(None, c, 2048, 8, lim=lim) == want


@pytest.mark.parametrize("name,lim", [
    ("Qwen3-14B-Q4_K_M", V6E), ("Llama-3.1-8B-Instruct-Q4_K_M", V6E), ("Llama-3.1-8B-Instruct-Q4_K_M", V5E),
    ("Llama-3.2-1B-Instruct-Q4_K_M", V6E), ("Llama-3.2-1B-Instruct-Q4_K_M", V5E),
    ("Qwen2.5-1.5B-Instruct-q8_0", V6E), ("Qwen2.5-1.5B-Instruct-q8_0", V5E),
])
def test_files_that_ran_with_8_slots_keep_8(monkeypatch, name, lim):
    c = _as_file(monkeypatch, name)
    assert G.fit_slots(None, c, 2048, 8, lim=lim) == 8


def test_slot_need_is_the_state_and_the_state_with_its_scratch(tmp_path):
    f, _, c = G.open_model(FXT.build("qwen35-f32", str(tmp_path)))
    for s in (8, 4, 2, 1):
        w, st, big = G.need_bytes(f, c, 64, s)
        assert st > 0 and G.slot_need(f, c, 64, s) == (w + big + st, w + big + st + _scratch(st, s))


def test_one_slot_inside_the_limit_but_not_the_headroom_still_runs(tmp_path):
    f, _, c = G.open_model(FXT.build("qwen35-f32", str(tmp_path)))
    w, st, big = G.need_bytes(f, c, 64, 1)
    lim = w + big + st + _scratch(st, 1)                            # weights, the state and its scratch: just inside
    assert w + big + st > G.HEADROOM * lim                          # one slot is outside the headroom
    assert G.fit_slots(f, c, 64, 8, lim=lim) == 1                   # both inside the limit: the best chance
    assert G.fit_slots(f, c, 64, 8, lim=lim - 1) == 8               # the scratch does not fit: check_fits decides
