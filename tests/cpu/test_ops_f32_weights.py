"""An F32 weight on the exact path is held as its float32 values, not rebuilt from the file's bytes at every call.

An F32 block IS its float32 value, so holding the values costs the same bytes as holding the blocks. Rebuilding them per
call (four byte planes shifted and or-ed, then bitcast) can dominate a decode step: a hybrid model's small
linear-attention gate weights (ssm_alpha / ssm_beta) are F32 in common quantized files. The product stays the exact
path's own float32 dot, bit for bit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from mashbox import ops
from mashbox.ops import reference as R

f32 = jnp.float32


def _f32_blocks(rows, cols, seed=0):
    v = np.random.default_rng(seed).standard_normal((rows, cols)).astype("<f4")
    v[0, :4] = [0.0, -0.0, np.float32(1e-40), np.float32(3.4e38)]          # zeros, a subnormal, a large value
    return v, v.view(np.uint8).reshape(rows, cols * 4)


def _old_exact(x, raw, cols):
    """the exact path's product as it stands for a Raw F32 weight: rebuild, then the HIGHEST float32 dot"""
    w = R.dequant(jnp.asarray(raw), 0, cols)
    return lax.dot_general(x.astype(f32), w, (((x.ndim - 1,), (1,)), ((), ())), precision=lax.Precision.HIGHEST,
                           preferred_element_type=f32).astype(x.dtype)


def _prims(fn, *a):
    out = set()

    def walk(j):
        for e in j.eqns:
            out.add(e.primitive.name)
            for p in e.params.values():
                for s in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(s, "jaxpr"):
                        walk(s.jaxpr if hasattr(s.jaxpr, "eqns") else s.jaxpr.jaxpr)
    walk(jax.make_jaxpr(fn)(*a).jaxpr)
    return out


@pytest.mark.parametrize("host", [True, False])
def test_an_f32_weight_multiplies_with_no_byte_rebuild_in_the_program(host):
    v, raw = _f32_blocks(32, 256)
    h = ops.prepare(raw if host else jnp.asarray(raw), 0, (32, 256), use_kernel=False)
    x = jnp.ones((1, 256), jnp.bfloat16)
    p = _prims(lambda x, h: ops.qmatmul(x, h), x, h)
    assert not p & {"shift_left", "or", "bitcast_convert_type"}, f"the program rebuilds the weight from bytes: {sorted(p)}"


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("host", [True, False])
def test_the_product_is_the_exact_paths_bit_for_bit(dtype, host):
    v, raw = _f32_blocks(32, 256, seed=1)
    h = ops.prepare(raw if host else jnp.asarray(raw), 0, (32, 256), use_kernel=False)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((3, 256)), dtype)
    got = np.asarray(jax.jit(ops.qmatmul)(x, h))
    want = np.asarray(jax.jit(_old_exact, static_argnums=2)(x, raw, 256))
    assert got.dtype == want.dtype and got.shape == want.shape == (3, 32)
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


def test_dequant_and_take_rows_give_the_stored_values():
    v, raw = _f32_blocks(32, 256, seed=3)
    h = ops.prepare(raw, 0, (32, 256), use_kernel=False)
    assert np.array_equal(np.asarray(ops.dequant(h)).view(np.uint32), v.view(np.uint32))
    ids = jnp.asarray([5, 0, 31, 5])
    assert np.array_equal(np.asarray(ops.take_rows(h, ids)).view(np.uint32), v[[5, 0, 31, 5]].view(np.uint32))


def test_a_weight_the_exact_path_chunks_keeps_its_blocks(monkeypatch):
    """over EXACT_CHUNK values the exact path multiplies a chunk of rows at a time; that path is left as it is"""
    monkeypatch.setattr(ops, "EXACT_CHUNK", 32 * 256 - 1)
    v, raw = _f32_blocks(32, 256, seed=4)
    assert isinstance(ops.prepare(raw, 0, (32, 256), use_kernel=False), ops.Raw)


def test_other_types_and_stacks_keep_their_handles():
    v, raw = _f32_blocks(32, 256, seed=5)
    assert isinstance(ops.prepare(np.stack([raw, raw]), 0, (32, 256), use_kernel=False), ops.Raw)   # a stack: unchanged
    q = np.zeros((32, 256 // 32 * 34), np.uint8)                                                    # Q8_0 blocks
    assert isinstance(ops.prepare(q, 8, (32, 256), use_kernel=False), ops.Raw)
