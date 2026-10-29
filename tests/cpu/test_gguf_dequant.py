"""Every GGUF block type the pinned gguf package reads: our plain-JAX dequantization == gguf-py's, bit for bit.

Blocks are random bytes (so every scale value appears: normal, subnormal, inf and NaN float16 scales, every code) plus
edge blocks (all zero bits, all one bits). NaN outputs must be NaN on both sides; every other output must have the same
32 bits. No chip, no network, no secrets.
"""
import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox.ops import reference as R  # noqa: E402


ROWS = 6


def _raw(t, rng):
    name, bs, ts = R.TYPES[t]
    cols = bs * 8 if bs > 1 else 64
    rb = R.row_bytes(t, cols)
    raw = rng.integers(0, 256, size=(ROWS, rb), dtype=np.uint8)
    raw[0] = 0                     # all-zero blocks
    raw[1] = 0xFF                  # all-one blocks
    return raw, cols


def _same(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    nan = np.isnan(a) | np.isnan(b)
    both_nan = np.isnan(a) & np.isnan(b)
    bits = a.view(np.uint32) == b.view(np.uint32)
    return int(np.sum(~np.where(nan, both_nan, bits)))


@pytest.mark.parametrize("t", sorted(R.TYPES), ids=lambda t: R.TYPES[t][0])
def test_dequant_bit_exact(t):
    rng = np.random.default_rng(1000 + t)
    raw, cols = _raw(t, rng)
    want = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(t))
    got = jax.jit(lambda r: R.dequant(r, t, cols))(jnp.asarray(raw))
    assert _same(got, want) == 0


@pytest.mark.parametrize("t", sorted(R.TYPES), ids=lambda t: R.TYPES[t][0])
def test_dequant_leading_axes(t):
    """stacked layers (L, rows, row_bytes) dequantize the same as one tensor at a time"""
    rng = np.random.default_rng(2000 + t)
    raw, cols = _raw(t, rng)
    st = np.stack([raw, raw[::-1]])
    got = np.asarray(R.dequant(jnp.asarray(st), t, cols))
    for i in range(2):
        assert _same(got[i], gguf.quants.dequantize(st[i], gguf.GGMLQuantizationType(t))) == 0


def test_covers_every_type_the_pinned_reader_dequantizes():
    """the reference covers exactly the types gguf-py dequantizes (F32, F16 and every quant class)"""
    from gguf import quants as Q
    have = {0, 1} | {int(t) for t in Q._type_traits}
    assert set(R.TYPES) == have, sorted(set(R.TYPES) ^ have)
    for t, (name, bs, ts) in R.TYPES.items():
        assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(t)] == (bs, ts), name
        assert gguf.GGMLQuantizationType(t).name == name
