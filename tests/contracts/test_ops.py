"""op/<TYPE> contracts: every ggml block type the GGUF reader accepts dequantizes and multiplies right on the exact path.

Guards: a weight of some block type that comes out wrong on the CPU path the engine runs without a kernel package: a
wrong scale, a block layout read one byte or one nibble off, a sign or zero point flipped, rows or chunks put back in
the wrong order, a leading axis dropped, an embedding row gathered from the wrong bytes, an expert taken from the wrong
slot. One contract id per type in mashbox.ops.reference.TYPES (op/F32 ... op/NVFP4) plus op/<TYPE>:experts for its
expert-stack case (a MoE row names both, a dense row only op/<TYPE>), reported through the `contract`
fixture (tests/contracts/conftest.py) to contracts.json. Only the bit-exact outputs (layout, dequant, take_rows,
expert_weights) are recorded for the fingerprint; a product under a tolerance is decided by its bar alone.

Reference: llama.cpp's own gguf Python package (gguf==0.19.0: GGML_QUANT_SIZES and gguf.quants.dequantize), never the
engine's code, and float64 NumPy for every product. Like llama.cpp's tests/test-backend-ops.cpp (a37cc46) this proves
types, not files: per type one small grid of shapes (weight rows 1 to 9; K = 1, 2 and 16 blocks; leading axes on the
activations, a stack of layers through lax.scan, a stack of experts; float types take 1, 33 and 512 values per row).
Each case calls the engine the way the GGUF archs do: ops.prepare(raw, t, shape) for weights (so BF16 takes its native
bfloat16 handle, and also its dequantized fallback), ops.prepare(..., use_kernel=False) for the embedding, then
ops.qmatmul / ops.take_rows / ops.expert_weights / ops.dequant.

Bars, written 2026-10-30 before the first run and never moved after a result:
- layout: the type's (values per block, bytes per block) == gguf-py's GGML_QUANT_SIZES.
- dequant, take_rows, expert_weights: the same float32 bits as gguf-py's dequantize; NaN only where gguf-py has NaN.
- qmatmul, float32 activations: NMSE = sum (y - ref)^2 / sum ref^2 (llama.cpp's metric) <= 1e-8, AND every value
  within the float32 dot product bound |y - ref| <= gamma_K * (|x| @ |W|.T), gamma_K = K u / (1 - K u), u = 2^-24,
  K = values per row. Why: the weights are the same float32 values as the reference and the dot runs with float32
  accumulation (HIGHEST precision), so the only error is float32 summation of K products, about K u^2 = 1.5e-11 NMSE at
  K = 4096, and the bound holds for any summation order. llama.cpp's mul_mat ceiling 5e-4 (test-backend-ops.cpp:5343)
  is far above both.
- qmatmul, bfloat16 activations (the forward's dtype): NMSE <= 2e-5: rounding the float32 result to bfloat16 moves
  each value by at most 2^-8 of itself (bfloat16 has 8 significand bits; NMSE <= 2^-16 = 1.53e-5). Correction
  2026-10-30, made after the first run: this bar was first written as 5e-6 from a unit roundoff of 2^-9, but
  bfloat16's is 2^-8. Only the constant changed, not the formula.
- a reference that is exactly zero: the output is exactly zero. Every output is finite.
Weights for the products are drawn so every block is finite with its largest value in [2^-6, 2^8]: random bytes,
blocks outside that range redrawn (float types: normal values). The dequant case also runs raw random bytes with
all-zero and all-one blocks, so every scale value (subnormal, inf, NaN) and every code appears. No chip, no network.
"""
import ml_dtypes
import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
import jax  # noqa: E402  a contract never skips: a missing jax fails the run
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from mashbox import ops  # noqa: E402
from mashbox.ops import dense16, reference as R  # noqa: E402

U = 2.0 ** -24                  # float32 unit roundoff
NMSE_F32 = 1e-8                 # qmatmul, float32 activations
NMSE_BF16 = 2e-5                # qmatmul, bfloat16 activations: 2^-16 = 1.53e-5 rounded up (first 5e-6 from 2^-9: wrong)
LLAMA_CPP_MUL_MAT = 5e-4        # test-backend-ops.cpp:5343 at a37cc46: the ceiling no bar here may pass
assert NMSE_F32 < NMSE_BF16 < LLAMA_CPP_MUL_MAT
LO, HI = 2.0 ** -6, 2.0 ** 8    # every product block's largest value
FLOAT_COLS = {1: 1, 2: 33, 16: 512}   # float types (one value per block): values per row for K = 1, 2, 16

# case: (weight rows, K blocks, activation leading shape, activation dtype, kind)
F32, BF16 = np.float32, ml_dtypes.bfloat16
CASES = {
    "r1-k1-m5": (1, 1, (5,), F32, "matmul"),             # a 1-row weight, many activation rows
    "r9-k2-m3": (9, 2, (3,), F32, "matmul"),
    "r4-k16-m2": (4, 16, (2,), F32, "matmul"),           # K = 16 blocks
    "r7-k1-lead2x3": (7, 1, (2, 3), F32, "matmul"),      # leading axes on the activations
    "r5-k2-m4-bf16": (5, 2, (4,), BF16, "matmul"),       # the forward's activation dtype
    "r6-k2-chunk2": (6, 2, (3,), F32, "chunked"),        # rows rebuilt 2 at a time (lax.map over chunks)
    "L3-r3-k2-scan": (3, 2, (2,), F32, "scan"),          # a stack of 3 layers, one per lax.scan step
    "r9-k1-embed": (9, 1, None, None, "take_rows"),      # embedding rows gathered from the byte run
    "E4-r3-k1-experts": (3, 1, None, None, "experts"),   # 2 of 4 stacked experts gathered and rebuilt
    "r4-k2-randbytes": (4, 2, None, None, "dequant"),    # every scale value and code, NaN and inf included
}
# A known engine bug, kept as a FAILING case (strict: it turns red the day the fix lands, then this mark goes): BF16
# expert stacks from ops.prepare (use_kernel=True, as the archs call it) are dense16 handles, and ops.expert_weights
# reads h.raw, so a BF16 MoE on the exact path stops with AttributeError. The conftest counts it: op/BF16:experts FAIL.
BF16_EXPERTS_XFAIL = pytest.mark.xfail(strict=True, raises=AttributeError,
                                       reason="ops.expert_weights has no Dense16 case (ops/__init__.py:647)")
PATHS = ["default", "dequantized"]   # dequantized: BF16's fallback (MASHBOX_DENSE16=off); other types have one path


def _params():
    for t in sorted(R.TYPES):
        name = R.TYPES[t][0]
        yield pytest.param(t, "default", "layout", id=f"{name}-layout")
        for case in CASES:
            for path in PATHS:
                if path == "dequantized" and t not in dense16.TYPES:
                    continue
                if path == "dequantized" and CASES[case][4] in ("take_rows", "dequant"):
                    continue          # these already run use_kernel=False, the engine's embedding call
                marks = [BF16_EXPERTS_XFAIL] if (t, path, CASES[case][4]) == (30, "default", "experts") else []
                yield pytest.param(t, path, case, marks=marks,
                                   id=f"{name}-{case}" + ("" if path == "default" else "-dequantized"))


def _cols(t, kb):
    bs = R.TYPES[t][1]
    return FLOAT_COLS[kb] if bs == 1 else kb * bs


def _ref(raw, t):
    """llama.cpp's gguf-py dequantization of raw (..., row_bytes) -> (..., cols) float32"""
    return np.asarray(gguf.quants.dequantize(np.asarray(raw), gguf.GGMLQuantizationType(t)), np.float32)


def _weights(t, lead, rows, cols, rng):
    """raw (*lead, rows, row_bytes) uint8 of type t: every block finite, its largest value in [LO, HI]"""
    name, bs, ts = R.TYPES[t]
    n = int(np.prod(lead, dtype=np.int64)) * rows * cols // bs
    if bs == 1:
        v = rng.standard_normal(n)
        raw = {0: v.astype(np.float32), 1: v.astype(np.float16), 30: v.astype(np.float32).astype(BF16)}[t].view(np.uint8)
        return raw.reshape(tuple(lead) + (rows, R.row_bytes(t, cols)))
    kept, have = [], 0
    for _ in range(1000):
        cand = rng.integers(0, 256, size=(max(256, 8 * n), ts), dtype=np.uint8)
        v = _ref(cand, t).reshape(len(cand), bs)
        with np.errstate(invalid="ignore"):
            m = np.max(np.abs(v), axis=1)
            ok = np.all(np.isfinite(v), axis=1) & (m >= LO) & (m <= HI)
        kept.append(cand[ok])
        have += int(ok.sum())
        if have >= n:
            break
    assert have >= n, f"{name}: only {have} of {n} random blocks are finite with largest value in [{LO}, {HI}]"
    return np.concatenate(kept)[:n].reshape(tuple(lead) + (rows, R.row_bytes(t, cols)))


def _acts(lead, cols, dtype, rng):
    return rng.standard_normal(tuple(lead) + (cols,)).astype(np.float32).astype(dtype)


def _prepare(raw, t, shape, path):
    return ops.prepare(raw, t, shape) if path == "default" else ops.prepare(raw, t, shape, use_kernel=False)


def _same_bits(got, want):
    """-> number of values whose float32 bits differ (NaN must be NaN on both sides)"""
    a, b = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    nan = np.isnan(a) | np.isnan(b)
    return int(np.sum(~np.where(nan, np.isnan(a) & np.isnan(b), a.view(np.uint32) == b.view(np.uint32))))


def _check_product(y, x, w, cols, dtype):
    """y (..., rows) from the engine vs float64 x @ w.T; returns the NMSE after asserting every bar"""
    x64, w64 = np.asarray(x).astype(np.float64), w.astype(np.float64)
    ref = x64 @ np.swapaxes(w64, -1, -2)
    y = np.asarray(y).astype(np.float64)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    assert np.all(np.isfinite(y)), "non-finite output"
    den = float(np.sum(ref ** 2))
    if den == 0.0:
        assert np.all(y == 0.0), "reference is exactly zero, output is not"
        return 0.0
    nmse = float(np.sum((y - ref) ** 2)) / den
    if dtype is F32:
        gamma = cols * U / (1 - cols * U)
        bound = gamma * (np.abs(x64) @ np.swapaxes(np.abs(w64), -1, -2))
        over = int(np.sum(np.abs(y - ref) > bound))
        assert over == 0, f"{over} values outside the float32 dot bound (NMSE {nmse:.3e})"
        assert nmse <= NMSE_F32, f"NMSE {nmse:.3e} > {NMSE_F32:.0e}"
    else:
        assert nmse <= NMSE_BF16, f"NMSE {nmse:.3e} > {NMSE_BF16:.0e}"
    return nmse


@pytest.mark.contract
@pytest.mark.parametrize("t,path,case", list(_params()))
def test_op(contract, t, path, case, monkeypatch):
    name, bs, ts = R.TYPES[t]
    kind = "layout" if case == "layout" else CASES[case][4]
    c = contract(f"op/{name}" + (":experts" if kind == "experts" else ""))
    if kind == "layout":
        assert (bs, ts) == tuple(gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(t)])
        c.record(np.array([bs, ts], np.int64))
        return
    rows, kb, lead, dtype, _ = CASES[case]
    cols = _cols(t, kb)
    rng = np.random.default_rng([t, sorted(CASES).index(case), PATHS.index(path)])
    assert ops.weight_path(t, (rows, cols)) == "exact", "a kernel package is loaded: these are exact-path contracts"
    if kind == "dequant":
        raw = rng.integers(0, 256, size=(rows, R.row_bytes(t, cols)), dtype=np.uint8)
        raw[0] = 0
        raw[1] = 0xFF
        got = jax.jit(ops.dequant)(ops.prepare(raw, t, (rows, cols), use_kernel=False))
        bad = _same_bits(got, _ref(raw, t))
        c.record(np.asarray(got), exact=True)
        assert bad == 0, f"{bad} of {got.size} values differ from gguf-py"
        return
    if kind == "take_rows":
        raw = _weights(t, (), rows, cols, rng)
        ids = jnp.asarray([8, 0, 3, 3, 5], jnp.int32)
        got = jax.jit(lambda h, i: ops.take_rows(h, i))(ops.prepare(raw, t, (rows, cols), use_kernel=False), ids)
        bad = _same_bits(got, _ref(raw, t)[np.asarray(ids)])
        c.record(np.asarray(got), exact=True)
        assert bad == 0, f"{bad} of {got.size} values differ from gguf-py"
        return
    if kind == "experts":
        E, pick = 4, jnp.asarray([3, 1], jnp.int32)
        raw = _weights(t, (E,), rows, cols, rng)
        h = _prepare(raw, t, (E, rows, cols), path)
        x = jnp.asarray(_acts((2,), cols, F32, rng))
        got, _ = jax.jit(lambda h, i, x: ops.expert_weights(h, i, x))(h, pick, x)
        bad = _same_bits(got, _ref(raw, t)[np.asarray(pick)])
        c.record(np.asarray(got), exact=True)
        assert bad == 0, f"{bad} of {got.size} values differ from gguf-py"
        return
    x = _acts(lead, cols, dtype, rng)
    if kind == "scan":
        L = 3
        raw = _weights(t, (L,), rows, cols, rng)
        H = _prepare(raw, t, (rows, cols), path)
        held, xs = ops.scan_split(H)

        def run(H, x):
            def body(_, a):
                layer, w = a
                return None, ops.qmatmul(x, ops.scan_join(held, w, layer))
            return lax.scan(body, None, (jnp.arange(L), xs))[1]
        y = jax.jit(run)(H, jnp.asarray(x))
        w = _ref(raw, t)                                                  # (L, rows, cols)
        nmse = max(_check_product(y[i], x, w[i], cols, dtype) for i in range(L))
    else:
        raw = _weights(t, (), rows, cols, rng)
        h = _prepare(raw, t, (rows, cols), path)
        if kind == "chunked":
            monkeypatch.setattr(ops, "EXACT_CHUNK", 2 * cols)
            if isinstance(h, ops.Raw):
                assert ops.exact_chunk_rows(rows, cols) == 2
        y = jax.jit(lambda x, h: ops.qmatmul(x, h))(jnp.asarray(x), h)
        assert y.dtype == jnp.dtype(dtype), (y.dtype, dtype)
        nmse = _check_product(y, x, _ref(raw, t), cols, dtype)
    assert nmse <= LLAMA_CPP_MUL_MAT
