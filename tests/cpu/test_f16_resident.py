"""F16 weights held as their values on the device (Dense32 with float16 storage): the exact path's product, bit for bit.

Before: an F16 weight (a GGUF tensor of ggml type 1, or a Hugging Face float16 safetensors weight) was a Raw handle, and
every call rebuilt its float32 values from the byte pairs (two uint8 per value) before the float32 dot. Now prepare
holds the float16 values (the same bytes as the file) and qmatmul runs the exact path's own float32 HIGHEST dot on the
same float32 values, in the same row chunks (exact_chunk_rows) and behind the same barrier, so every output has the same
bits as the Raw path. Covered: unstacked, chunked rows, a layer stack through lax.scan, a Mixed-free stack, bfloat16 and
float32 activations, dequant and take_rows bits, the route label, and expert stacks staying Raw. No chip, no network.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from mashbox import ops
from mashbox.ops import reference as R

F16 = 1


def _raw(lead, rows, cols, rng):
    v = rng.standard_normal(tuple(lead) + (rows, cols)).astype(np.float16)
    v.reshape(-1)[:3] = np.array([0.0, -0.0, 6.1e-5], np.float16)      # zero, negative zero, the smallest normal
    return v.view(np.uint8).reshape(tuple(lead) + (rows, cols * 2))


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def _raw_handle(raw, shape):
    return ops.Raw(jnp.asarray(ops._flat(raw)), F16, shape)


@pytest.mark.parametrize("rows,cols,chunk", [(5, 48, None), (12, 40, 4 * 40), (9, 33, 3 * 33)])
@pytest.mark.parametrize("dt", [jnp.float32, jnp.bfloat16])
def test_f16_product_is_the_exact_path_bit_for_bit(rows, cols, chunk, dt, monkeypatch):
    if chunk is not None:
        monkeypatch.setattr(ops, "EXACT_CHUNK", chunk)
    rng = np.random.default_rng(rows * 100 + cols)
    raw = _raw((), rows, cols, rng)
    h = ops.prepare(raw, F16, (rows, cols))
    assert isinstance(h, ops.Dense32) and h.w.dtype == jnp.float16 and ops.route(h) == "resident_f32"
    x = jnp.asarray(rng.standard_normal((3, cols)).astype(np.float32)).astype(dt)
    want = jax.jit(ops.qmatmul)(x, _raw_handle(raw, (rows, cols)))
    got = jax.jit(ops.qmatmul)(x, h)
    assert got.dtype == x.dtype
    assert np.array_equal(_bits(got.astype(jnp.float32)), _bits(want.astype(jnp.float32)))
    assert np.array_equal(_bits(ops.dequant(h)), _bits(R.dequant(jnp.asarray(raw), F16, cols)))
    ids = jnp.asarray([0, rows - 1, 1])
    assert np.array_equal(_bits(ops.take_rows(h, ids)), _bits(ops.take_rows(_raw_handle(raw, (rows, cols)), ids)))


def test_f16_layer_stack_through_scan_is_the_exact_path_bit_for_bit(monkeypatch):
    monkeypatch.setattr(ops, "EXACT_CHUNK", 2 * 64)                  # rows rebuilt 2 at a time inside the scan body
    rng = np.random.default_rng(7)
    L, rows, cols = 3, 6, 64
    raw = _raw((L,), rows, cols, rng)
    h = ops.prepare(raw, F16, (rows, cols))
    assert isinstance(h, ops.Dense32) and h.w.shape == (L, rows, cols)
    x0 = jnp.asarray(rng.standard_normal((2, cols)).astype(np.float32)).astype(jnp.bfloat16)

    def run(w):
        def body(x, wl):
            y = ops.qmatmul(x, wl)
            return x, y
        return lax.scan(body, x0, w)[1]
    want = jax.jit(run)(_raw_handle(raw, (rows, cols)))
    got = jax.jit(run)(h)
    assert np.array_equal(_bits(got.astype(jnp.float32)), _bits(want.astype(jnp.float32)))


def test_f16_device_bytes_are_the_file_bytes():
    rng = np.random.default_rng(3)
    raw = _raw((2,), 4, 32, rng)
    h = ops.prepare(raw, F16, (4, 32))
    assert int(h.w.nbytes) == raw.nbytes
    assert ops.device_nbytes(F16, (4, 32), raw[0].nbytes) == raw[0].nbytes


def test_f16_expert_stacks_and_f32_keep_their_paths():
    rng = np.random.default_rng(5)
    e = _raw((4,), 3, 32, rng)                                        # (experts, rows, cols): the expert op's Raw
    assert isinstance(ops.prepare(e, F16, (4, 3, 32)), ops.Raw)
    f32raw = rng.standard_normal((4, 32)).astype(np.float32).view(np.uint8)
    h = ops.prepare(f32raw, 0, (4, 32))
    assert isinstance(h, ops.Dense32) and h.w.dtype == jnp.float32
    stack = rng.standard_normal((2, 4, 32)).astype(np.float32).view(np.uint8).reshape(2, 4, 128)
    assert isinstance(ops.prepare(stack, 0, (4, 32)), ops.Raw)          # F32 stacks unchanged (still Raw)
