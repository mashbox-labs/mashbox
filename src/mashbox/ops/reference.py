"""Exact plain-JAX dequantization of GGUF block types: the reference path every GGUF weight can run on.

Each function takes a tensor's raw GGUF bytes as stored in the file (uint8, shape (..., bytes per row)) and returns the
weights as float32 (..., values per row). The math follows the NumPy dequantization in llama.cpp's `gguf` Python
package (gguf-py, gguf/quants.py, MIT licence, Copyright (c) 2023 Georgi Gerganov; github.com/ggml-org/llama.cpp,
pinned as the PyPI release gguf==0.19.0), step for step and in the same order, so the result is bit for bit the same
float32 values. tests/cpu/test_gguf_dequant.py checks that for every type against gguf-py itself. The IQ codebooks
and sign tables are read from the installed `gguf` package at run time; none are copied here.

This path is exact and slow: every call rebuilds float32 weights from the blocks. The TPU kernel package, when
installed, runs the common types without it (see mashbox/ops/__init__.py).
"""
import numpy as np
import jax.numpy as jnp
from jax import lax

f32 = jnp.float32
u8, u16, u32, i8 = jnp.uint8, jnp.uint16, jnp.uint32, jnp.int8
QK_K = 256

# ggml type id -> (name, values per block, bytes per block): gguf-py GGML_QUANT_SIZES for the types read here
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
}
NAMES = {v[0]: k for k, v in TYPES.items()}


def _sh(*v):
    return jnp.asarray(np.array(v, np.uint8))


def _u16le(b):
    """(..., 2k) uint8 -> (..., k) uint16, little-endian"""
    b = b.reshape(b.shape[:-1] + (-1, 2)).astype(u16)
    return b[..., 0] | (b[..., 1] << 8)


def _u32le(b):
    """(..., 4k) uint8 -> (..., k) uint32, little-endian"""
    b = b.reshape(b.shape[:-1] + (-1, 4)).astype(u32)
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def _f16(b):
    """(..., 2) uint8 -> (..., 1) float32 from a little-endian float16"""
    return lax.bitcast_convert_type(_u16le(b), jnp.float16).astype(f32)


def _i8(b):
    return lax.bitcast_convert_type(b, i8)


def _nib(qs, n):
    """qs (N, ..., n) bytes -> low nibbles then high nibbles of each group of n bytes: (N, ..., 2, n)"""
    return (qs[..., None, :] >> _sh(0, 4)[:, None]) & u8(0x0F)


# ------------------------------------------------------------------------------------------------- per-type blocks
# Each takes blocks (N, bytes per block) uint8 and returns (N, values per block) float32.

def _q4_0(b):
    N = b.shape[0]
    d, qs = _f16(b[:, :2]), b[:, 2:]
    q = (_nib(qs, 16).reshape(N, -1)).astype(i8) - i8(8)
    return d * q.astype(f32)


def _q4_1(b):
    N = b.shape[0]
    d, m, qs = _f16(b[:, :2]), _f16(b[:, 2:4]), b[:, 4:]
    q = _nib(qs, 16).reshape(N, -1).astype(f32)
    return (d * q) + m


def _q5_hi(qh4, N):
    qh = _u32le(qh4)                                                     # (N, 1)
    return ((qh >> jnp.arange(32, dtype=u32)[None]) & u32(1)).astype(u8)  # (N, 32)


def _q5_0(b):
    N = b.shape[0]
    d, qh, qs = _f16(b[:, :2]), _q5_hi(b[:, 2:6], N), b[:, 6:]
    ql = _nib(qs, 16).reshape(N, -1)
    q = (ql | (qh << u8(4))).astype(i8) - i8(16)
    return d * q.astype(f32)


def _q5_1(b):
    N = b.shape[0]
    d, m, qh, qs = _f16(b[:, :2]), _f16(b[:, 2:4]), _q5_hi(b[:, 4:8], N), b[:, 8:]
    ql = _nib(qs, 16).reshape(N, -1)
    q = (ql | (qh << u8(4))).astype(f32)
    return (d * q) + m


def _q8_0(b):
    d, x = _f16(b[:, :2]), _i8(b[:, 2:]).astype(f32)
    return x * d


def _q2_k(b):
    N = b.shape[0]
    scales, qs = b[:, :16], b[:, 16:80]
    d, dmin = _f16(b[:, 80:82]), _f16(b[:, 82:84])
    dl = (d * (scales & u8(0xF)).astype(f32)).reshape(N, 16, 1)
    ml = (dmin * (scales >> u8(4)).astype(f32)).reshape(N, 16, 1)
    q = (qs.reshape(N, -1, 1, 32) >> _sh(0, 2, 4, 6).reshape(1, 1, 4, 1)) & u8(3)
    q = q.reshape(N, 16, 16).astype(f32)
    return (dl * q - ml).reshape(N, -1)


def _q3_k(b):
    N = b.shape[0]
    hmask, qs, sc, d = b[:, :32], b[:, 32:96], b[:, 96:108], _f16(b[:, 108:110])
    ls = (sc[:, :8].reshape(N, 1, 8) >> _sh(0, 4).reshape(1, 2, 1)).reshape(N, 16)
    hs = (sc[:, 8:].reshape(N, 1, 4) >> _sh(0, 2, 4, 6).reshape(1, 4, 1)).reshape(N, 16)
    s = (ls & u8(0x0F)) | ((hs & u8(0x03)) << u8(4))
    s = (s.astype(i8) - i8(32)).astype(f32)
    dl = (d * s).reshape(N, 16, 1)
    ql = (qs.reshape(N, -1, 1, 32) >> _sh(0, 2, 4, 6).reshape(1, 1, 4, 1)).reshape(N, 16, 16) & u8(3)
    qh = (hmask.reshape(N, -1, 1, 32) >> jnp.arange(8, dtype=u8).reshape(1, 1, 8, 1)).reshape(N, 16, 16) & u8(1)
    qh = qh ^ u8(1)
    q = (ql.astype(i8) - (qh << u8(2)).astype(i8)).astype(f32)
    return (dl * q).reshape(N, QK_K)


def _scale_min_k4(s):
    """12 packed bytes -> 8 six-bit scales and 8 six-bit mins (gguf-py Q4_K.get_scale_min)"""
    N = s.shape[0]
    s = s.reshape(N, 3, 4)
    d, m, md = s[:, 0:1], s[:, 1:2], s[:, 2:3]
    sc = jnp.concatenate([d & u8(0x3F), (md & u8(0x0F)) | ((d >> u8(2)) & u8(0x30))], -1)
    mn = jnp.concatenate([m & u8(0x3F), (md >> u8(4)) | ((m >> u8(2)) & u8(0x30))], -1)
    return sc.reshape(N, 8), mn.reshape(N, 8)


def _q4_k(b):
    N = b.shape[0]
    d, dmin, sc, qs = _f16(b[:, :2]), _f16(b[:, 2:4]), b[:, 4:16], b[:, 16:]
    s, m = _scale_min_k4(sc)
    d = (d * s.astype(f32)).reshape(N, -1, 1)
    dm = (dmin * m.astype(f32)).reshape(N, -1, 1)
    q = (_nib(qs.reshape(N, -1, 32), 32)).reshape(N, -1, 32).astype(f32)
    return (d * q - dm).reshape(N, QK_K)


def _q5_k(b):
    N = b.shape[0]
    d, dmin, sc, qh, qs = _f16(b[:, :2]), _f16(b[:, 2:4]), b[:, 4:16], b[:, 16:48], b[:, 48:]
    s, m = _scale_min_k4(sc)
    d = (d * s.astype(f32)).reshape(N, -1, 1)
    dm = (dmin * m.astype(f32)).reshape(N, -1, 1)
    ql = _nib(qs.reshape(N, -1, 32), 32).reshape(N, -1, 32)
    qh = ((qh.reshape(N, -1, 1, 32) >> jnp.arange(8, dtype=u8).reshape(1, 1, 8, 1)) & u8(1)).reshape(N, -1, 32)
    q = (ql | (qh << u8(4))).astype(f32)
    return (d * q - dm).reshape(N, QK_K)


def _q6_k(b):
    N = b.shape[0]
    ql, qh, sc, d = b[:, :128], b[:, 128:192], b[:, 192:208], _f16(b[:, 208:210])
    sc = _i8(sc).astype(f32)
    d = (d * sc).reshape(N, 16, 1)
    ql = _nib(ql.reshape(N, -1, 64), 64).reshape(N, -1, 32)
    qh = ((qh.reshape(N, -1, 1, 32) >> _sh(0, 2, 4, 6).reshape(1, 1, 4, 1)) & u8(3)).reshape(N, -1, 32)
    q = (ql | (qh << u8(4))).astype(i8) - i8(32)
    q = q.reshape(N, 16, -1).astype(f32)
    return (d * q).reshape(N, QK_K)


def _tq1_0(b):
    N = b.shape[0]
    qs, qh, d = b[:, :48], b[:, 48:52], _f16(b[:, 52:54])
    p5, p4 = _sh(1, 3, 9, 27, 81), _sh(1, 3, 9, 27)
    q0 = (qs[:, :32].reshape(N, -1, 1, 32) * p5.reshape(1, 1, 5, 1)).reshape(N, -1)
    q1 = (qs[:, 32:].reshape(N, -1, 1, 16) * p5.reshape(1, 1, 5, 1)).reshape(N, -1)
    q2 = (qh.reshape(N, -1, 1, 4) * p4.reshape(1, 1, 4, 1)).reshape(N, -1)
    q = jnp.concatenate([q0, q1, q2], -1)
    q = ((q.astype(u16) * u16(3)) >> u16(8)).astype(i8) - i8(1)
    return d * q.astype(f32)


def _tq2_0(b):
    N = b.shape[0]
    qs, d = b[:, :64], _f16(b[:, 64:66])
    q = ((qs.reshape(N, -1, 1, 32) >> _sh(0, 2, 4, 6).reshape(1, 1, 4, 1)) & u8(3)).reshape(N, -1).astype(i8) - i8(1)
    return d * q.astype(f32)


_KV_FP4 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _mxfp4(b):
    """gguf-py computes e8m0_to_fp32_half(e) * kvalue, which is kvalue * 2**(e - 128) exactly (the half scale is a power
    of two, kvalues are small integers). XLA flushes float32 subnormals to zero on CPU and TPU, and e = 0 or 1 gives
    subnormal results, so the float32 bits are built with integer ops instead of a float multiply: same bits, every e."""
    N = b.shape[0]
    e = b[:, :1].astype(jnp.int32)                                            # (N, 1)
    q = _nib(b[:, 1:].reshape(N, 1, 16), 16).reshape(N, 32).astype(jnp.int32)
    # |kvalue| = odd * 2**p with odd in {0, 1, 3}: 0, 1, 2, 3, 4, 6, 8, 12
    odd = jnp.asarray(np.array([0, 1, 1, 3, 1, 3, 1, 3] * 2, np.int32))[q]
    p = jnp.asarray(np.array([0, 0, 1, 0, 2, 1, 3, 2] * 2, np.int32))[q]
    neg = q >= 8
    x = p + e - 128                                     # value = odd * 2**x
    # odd 1: 1.0 * 2**x ; odd 3: 1.5 * 2**(x + 1)
    E = jnp.where(odd == 3, x + 1, x)
    frac = jnp.where(odd == 3, jnp.int32(1 << 22), jnp.int32(0))
    normal = ((E + 127) << 23) | frac
    sub = jnp.where(odd == 3, jnp.int32(3) << jnp.clip(x + 149, 0, 30), jnp.int32(1) << jnp.clip(x + 149, 0, 30))
    bits = jnp.where(E < -126, sub, normal)
    bits = jnp.where(E > 127, jnp.int32(0x7F800000), bits)
    bits = jnp.where(odd == 0, jnp.int32(0), bits)
    bits = bits.astype(u32) | jnp.where(neg & (odd != 0), u32(0x80000000), u32(0))
    return lax.bitcast_convert_type(bits, f32)


def _ue4m3_half(x):
    e = (x >> u8(3)).astype(jnp.int32) & 0xF
    man = (x & u8(0x7)).astype(f32)
    p2 = lax.bitcast_convert_type(((e - 7 + 127).astype(jnp.int32) << 23), f32)    # 2 ** (e - 7), exact for e in 1..15
    raw = jnp.where(e == 0, man * f32(2.0 ** -9), (f32(1.0) + man / f32(8.0)) * p2)
    return jnp.where((x == 0) | (x == 0x7F), f32(0.0), raw * f32(0.5))


def _nvfp4(b):
    N = b.shape[0]
    d = _ue4m3_half(b[:, :4]).reshape(N, 4, 1)
    qs = b[:, 4:].reshape(N, 4, 8)
    v = jnp.concatenate([qs & u8(0x0F), qs >> u8(4)], -1)
    kv = jnp.asarray(np.array(_KV_FP4, np.int8))
    return (d * kv[v].astype(f32)).reshape(N, 64)


# IQ tables come from the installed gguf package (its own codebooks), cached as NumPy arrays
_TABLES = {}


def _table(name):
    if name not in _TABLES:
        from gguf import quants as Q
        if name == "ksigns":
            _TABLES[name] = np.frombuffer(Q.IQ2_XXS.ksigns, dtype=np.uint8).copy()
        else:
            cls = getattr(Q, name)
            cls.init_grid()
            _TABLES[name] = np.asarray(cls.grid, np.float32).reshape(cls.grid_shape)
    return jnp.asarray(_TABLES[name])


def _signs(bits):
    return jnp.where(bits == 0, f32(1), f32(-1))


def _ksigns(idx):
    """7-bit sign indices (...) -> (..., 8) +-1 from gguf-py's ksigns table"""
    s = _table("ksigns")[idx.astype(jnp.int32)]
    return _signs((s[..., None] >> jnp.arange(8, dtype=u8)) & u8(1))


def _iq2_xxs(b):
    N = b.shape[0]
    d, qs = _f16(b[:, :2]), b[:, 2:].reshape(N, 8, 2, 4)
    aux = _u32le(qs[:, :, 1, :]).reshape(N, 8)
    db = d * (f32(0.5) + (aux >> u32(28)).astype(f32)) * f32(0.25)
    db = db.reshape(N, -1, 1, 1)
    si = ((aux.reshape(N, -1, 1) >> jnp.asarray(np.array([0, 7, 14, 21], np.uint32)).reshape(1, 1, 4)) & u32(0x7F))
    sg = _ksigns(si)                                                       # (N, 8, 4, 8)
    grid = _table("IQ2_XXS")[qs[:, :, 0, :].reshape(N, -1).astype(jnp.int32)].reshape(N, -1, 4, 8)
    return (db * grid * sg).reshape(N, -1)


def _iq2_xs(b):
    N = b.shape[0]
    d, qs, sc = _f16(b[:, :2]), _u16le(b[:, 2:66]), b[:, 66:74]
    sc = (sc.reshape(N, -1, 1) >> _sh(0, 4).reshape(1, 1, 2)).reshape(N, -1) & u8(0x0F)
    db = (d * (f32(0.5) + sc.astype(f32)) * f32(0.25)).reshape(N, -1, 1, 1)
    sg = _ksigns(qs >> u16(9)).reshape(N, -1, 2, 8)
    grid = _table("IQ2_XS")[(qs & u16(511)).astype(jnp.int32)].reshape(N, -1, 2, 8)
    return (db * grid * sg).reshape(N, -1)


def _iq2_s(b):
    N = b.shape[0]
    d, qs, sb, qh, sc = _f16(b[:, :2]), b[:, 2:34], b[:, 34:66], b[:, 66:74], b[:, 74:82]
    sc = (sc.reshape(N, -1, 1) >> _sh(0, 4).reshape(1, 1, 2)).reshape(N, -1) & u8(0x0F)
    db = (d * (f32(0.5) + sc.astype(f32)) * f32(0.25)).reshape(N, -1, 1, 1)
    sg = _signs((sb.reshape(N, -1, 1) >> jnp.arange(8, dtype=u8).reshape(1, 1, 8)) & u8(1)).reshape(N, -1, 2, 8)
    qh = qh.reshape(N, -1, 1) >> _sh(0, 2, 4, 6).reshape(1, 1, 4)
    idx = qs.astype(u16) | ((qh & u8(0x03)).astype(u16) << u16(8)).reshape(N, -1)
    grid = _table("IQ2_S")[idx.astype(jnp.int32)].reshape(N, -1, 2, 8)
    return (db * grid * sg).reshape(N, -1)


def _iq3_xxs(b):
    N = b.shape[0]
    d, qs, sc = _f16(b[:, :2]), b[:, 2:66], _u32le(b[:, 66:98])
    db = (d * (f32(0.5) + (sc >> u32(28)).astype(f32)) * f32(0.5)).reshape(N, -1, 1, 1)
    si = (sc.reshape(N, -1, 1) >> jnp.asarray(np.array([0, 7, 14, 21], np.uint32)).reshape(1, 1, 4)) & u32(0x7F)
    sg = _ksigns(si)                                                       # (N, 8, 4, 8)
    grid = _table("IQ3_XXS")[qs.astype(jnp.int32)].reshape(N, -1, 4, 8)
    return (db * grid * sg).reshape(N, -1)


def _iq3_s(b):
    N = b.shape[0]
    d, qs, qh, sb, sc = _f16(b[:, :2]), b[:, 2:66], b[:, 66:74], b[:, 74:106], b[:, 106:110]
    sc = (sc.reshape(N, -1, 1) >> _sh(0, 4).reshape(1, 1, 2)).reshape(N, -1) & u8(0x0F)
    db = (d * (u8(1) + u8(2) * sc).astype(f32)).reshape(N, -1, 1, 1)
    sg = _signs((sb.reshape(N, -1, 1) >> jnp.arange(8, dtype=u8).reshape(1, 1, 8)) & u8(1)).reshape(N, -1, 4, 8)
    qh = ((qh.reshape(N, -1, 1) >> jnp.arange(8, dtype=u8)) & u8(1)).astype(u16).reshape(N, -1)
    idx = qs.astype(u16) | (qh << u16(8))
    grid = _table("IQ3_S")[idx.astype(jnp.int32)].reshape(N, -1, 4, 8)
    return (db * grid * sg).reshape(N, -1)


def _iq1_s(b):
    N = b.shape[0]
    d, qs, qh = _f16(b[:, :2]), b[:, 2:34], _u16le(b[:, 34:50])
    dl = (d * (u16(2) * ((qh >> u16(12)) & u16(7)) + u16(1)).astype(f32)).reshape(N, -1, 1, 1)
    delta = jnp.where((qh & u16(0x8000)) == 0, f32(0.125), f32(-0.125)).reshape(N, -1, 1, 1)
    qh = qh.reshape(N, -1, 1) >> jnp.asarray(np.array([0, 3, 6, 9], np.uint16)).reshape(1, 1, 4)
    idx = qs.astype(u16) | ((qh & u16(7)) << u16(8)).reshape(N, -1)
    grid = _table("IQ1_S")[idx.astype(jnp.int32)].reshape(N, -1, 4, 8)
    return (dl * (grid + delta)).reshape(N, -1)


def _iq1_m(b):
    N = b.shape[0]
    qs, qh, sc = b[:, :32], b[:, 32:48], _u16le(b[:, 48:56])
    dd = (sc.reshape(N, 4) & u16(0xF000)) >> jnp.asarray(np.array([12, 8, 4, 0], np.uint16)).reshape(1, 4)
    dd = dd[:, 0] | dd[:, 1] | dd[:, 2] | dd[:, 3]
    d = lax.bitcast_convert_type(dd, jnp.float16).astype(f32).reshape(N, 1)
    sc = (sc.reshape(N, -1, 1) >> jnp.asarray(np.array([0, 3, 6, 9], np.uint16)).reshape(1, 1, 4))
    sc = (sc & u16(0x07)).reshape(N, -1)
    dl = (d * (u16(2) * sc + u16(1)).astype(f32)).reshape(N, -1, 2, 1, 1)
    qh = qh.reshape(N, -1, 1) >> _sh(0, 4).reshape(1, 1, 2)
    idx = qs.astype(u16) | ((qh & u8(0x07)).astype(u16) << u16(8)).reshape(N, -1)
    delta = jnp.where((qh & u8(0x08)) == 0, f32(0.125), f32(-0.125)).reshape(N, -1, 2, 2, 1)
    grid = _table("IQ1_S")[idx.astype(jnp.int32)].reshape(N, -1, 2, 2, 8)
    return (dl * (grid + delta)).reshape(N, -1)


_KV_IQ4 = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _iq4_nl(b):
    N = b.shape[0]
    d = _f16(b[:, :2])
    q = _nib(b[:, 2:].reshape(N, 1, 16), 16).reshape(N, -1)
    kv = jnp.asarray(np.array(_KV_IQ4, np.int8))
    return d * kv[q].astype(f32)


def _iq4_xs(b):
    N = b.shape[0]
    d, sh, sl, qs = _f16(b[:, :2]), _u16le(b[:, 2:4]), b[:, 4:8], b[:, 8:]
    sl = (sl.reshape(N, -1, 1) >> _sh(0, 4).reshape(1, 1, 2)).reshape(N, -1) & u8(0x0F)
    sh = (sh.reshape(N, 1, -1) >> jnp.asarray(np.arange(0, 16, 2, dtype=np.uint16)).reshape(1, -1, 1)).reshape(N, -1)
    sh = sh.astype(u8) & u8(0x03)
    s = (sl | (sh << u8(4))).astype(i8) - i8(32)
    dl = (d * s.astype(f32)).reshape(N, -1, 1)
    q = (_nib(qs.reshape(N, -1, 16), 16)).reshape(N, -1, 32)
    kv = jnp.asarray(np.array(_KV_IQ4, np.int8))
    return (dl * kv[q].astype(f32)).reshape(N, -1)


def _f32(b):
    return lax.bitcast_convert_type(_u32le(b), f32)


def _f16t(b):
    return lax.bitcast_convert_type(_u16le(b), jnp.float16).astype(f32)


def _bf16(b):
    return lax.bitcast_convert_type(_u16le(b).astype(u32) << u32(16), f32)


BLOCKS = {0: _f32, 1: _f16t, 30: _bf16, 2: _q4_0, 3: _q4_1, 6: _q5_0, 7: _q5_1, 8: _q8_0, 10: _q2_k, 11: _q3_k,
          12: _q4_k, 13: _q5_k, 14: _q6_k, 34: _tq1_0, 35: _tq2_0, 39: _mxfp4, 40: _nvfp4, 16: _iq2_xxs, 17: _iq2_xs,
          22: _iq2_s, 18: _iq3_xxs, 21: _iq3_s, 19: _iq1_s, 29: _iq1_m, 20: _iq4_nl, 23: _iq4_xs}
assert set(BLOCKS) == set(TYPES)


def row_bytes(t, cols):
    """bytes one row of `cols` values takes in ggml type t"""
    _, bs, ts = TYPES[t]
    if cols % bs:
        raise ValueError(f"{TYPES[t][0]}: row of {cols} values is not a multiple of the block size {bs}")
    return cols // bs * ts


def dequant(raw, t, cols):
    """raw (..., row_bytes(t, cols)) uint8 -> (..., cols) float32, exact (gguf-py's own math)"""
    if t not in BLOCKS:
        raise NotImplementedError(f"ggml type {t}")
    _, bs, ts = TYPES[t]
    lead = raw.shape[:-1]
    blocks = raw.reshape(-1, ts)
    if t in (0, 1, 30):
        out = BLOCKS[t](raw.reshape(-1, raw.shape[-1]))
    else:
        out = BLOCKS[t](blocks)
    return out.reshape(lead + (cols,))
