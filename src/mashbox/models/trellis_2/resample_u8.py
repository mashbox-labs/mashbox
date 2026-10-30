"""uint8 LANCZOS resize, written to reproduce the TRELLIS.2 reference env's resize bit for bit (numpy + math only).

Why: the reference env (vendor setup.sh --basic) loads Pillow-SIMD 12.1.1.post0 (SSE4 build, no AVX2/FMA), and its
LANCZOS rounding differs from stock Pillow by 1 uint8 level on a few hundred values per 512 image. This module is our
own implementation of that arithmetic, so the port needs neither Pillow-SIMD nor any particular Pillow build for stage 0.
Pillow-SIMD served only as the oracle when this module was checked.

Arithmetic reproduced (read from the published Pillow-SIMD 12.1.1.post0 sdist, sha256 8e9694ec; its LICENSE ships
unchanged beside this module as LICENSE-Pillow-SIMD):
- filter coefficients in float64 exactly as precompute_coeffs (libm sin via math.sin, same operation order, no FMA);
- per pass, coefficients re-quantised to int16 at a dynamic precision p (largest p with round(max*2^(p+1)) < 2^15,
  p <= 22), out = clip((sum(pix * k) + 2^(p-1)) >> p, 0, 255) with int32 sums (stock Pillow uses a fixed 22-bit int32);
- horizontal pass first (only on the source rows the vertical pass reads), then vertical, each only when needed;
- Image.resize semantics: same size + full box -> copy; RGBA -> premultiplied RGBa -> resize -> RGBA.
- RGBa -> RGBA uses SSE rcpps in Pillow-SIMD for pixel groups of 4 (CPU-dependent approximate reciprocal) and an exact
  integer division for the last width % 4 pixels of each row. RCP_FACTOR below holds the per-alpha 16-bit factor
  low16(cvtps_epi32(65408.0f * rcpps(float(alpha)))) read on an Intel Xeon. The reference
  outputs were also made on Intel, so the same table is assumed there; it was not read on that machine.
"""
from __future__ import annotations

import math

import numpy as np

PRECISION_BITS = 32 - 8 - 2
MAX_COEFS_PRECISION = 16 - 1
LANCZOS_SUPPORT = 3.0

# Filled from rcpps_factor_sse_intel.txt next to this file (see module docstring); while None, the RGBA path raises.
RCP_FACTOR: np.ndarray | None = None


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    x = x * math.pi
    return math.sin(x) / x


def _lanczos(x: float) -> float:
    if -3.0 <= x < 3.0:
        return _sinc(x) * _sinc(x / 3)
    return 0.0


def precompute_coeffs(in_size: int, in0: float, in1: float, out_size: int):
    """-> ksize, bounds int64 [out, 2] (xmin, xmax count), kk float64 [out, ksize] (zero padded)."""
    f0, f1 = np.float32(in0), np.float32(in1)
    scale = float(np.float32(f1 - f0)) / out_size  # C: (double)(in1 - in0) / outSize, in0/in1 are C float
    filterscale = scale
    if filterscale < 1.0:
        filterscale = 1.0
    support = LANCZOS_SUPPORT * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    kk = np.zeros((out_size, ksize), np.float64)
    bounds = np.zeros((out_size, 2), np.int64)
    in0d = float(f0)
    for xx in range(out_size):
        center = in0d + (xx + 0.5) * scale
        ww = 0.0
        ss = 1.0 / filterscale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        row = [0.0] * max(xmax, 0)
        for x in range(xmax):
            w = _lanczos((x + xmin - center + 0.5) * ss)
            row[x] = w
            ww += w
        for x in range(xmax):
            kk[xx, x] = row[x] / ww if ww != 0.0 else row[x]
        bounds[xx, 0], bounds[xx, 1] = xmin, xmax
    return ksize, bounds, kk


def normalize_coeffs_8bpc(kk: np.ndarray):
    """-> (int32 coefficients with int16 wrap semantics, precision)."""
    maxkk = float(kk.max())
    p = 0
    while p < PRECISION_BITS:
        if int(0.5 + maxkk * (1 << (p + 1))) >= (1 << MAX_COEFS_PRECISION):
            break
        p += 1
    s = kk * float(1 << p)
    q = np.where(kk < 0, np.trunc(-0.5 + s), np.trunc(0.5 + s)).astype(np.int64)
    q = ((q + 32768) & 0xFFFF) - 32768  # INT16 store
    return q.astype(np.int32), p


def _pass(a: np.ndarray, axis: int, bounds: np.ndarray, kk: np.ndarray) -> np.ndarray:
    """One convolution pass along axis (1 = horizontal, 0 = vertical) of uint8 [H, W, C]."""
    ki, p = normalize_coeffs_8bpc(kk)
    n_in = a.shape[axis]
    out_n, ksize = ki.shape
    idx = np.minimum(bounds[:, :1] + np.arange(ksize)[None, :], n_in - 1)  # taps past xmax have k == 0
    shp = [out_n if d == axis else a.shape[d] for d in range(3)]
    acc = np.full(shp, 1 << (p - 1), np.int32)
    a32 = a.astype(np.int32)
    for k in range(ksize):
        col = ki[:, k]
        if not col.any():
            continue
        g = np.take(a32, idx[:, k], axis=axis)
        acc += g * (col[None, :, None] if axis == 1 else col[:, None, None])
    return np.clip(acc >> p, 0, 255).astype(np.uint8)


def resample(a: np.ndarray, size: tuple[int, int], box=None) -> np.ndarray:
    """ImagingResample(LANCZOS) on uint8 [H, W, C] (any band count); size = (width, height) like PIL."""
    a = np.ascontiguousarray(a)
    if a.ndim == 2:
        return resample(a[:, :, None], size, box)[:, :, 0]
    h_in, w_in = a.shape[:2]
    xsize, ysize = int(size[0]), int(size[1])
    if box is None:
        box = (0.0, 0.0, float(w_in), float(h_in))
    b = [float(np.float32(v)) for v in box]
    need_h = xsize != w_in or b[0] != 0 or b[2] != xsize
    need_v = ysize != h_in or b[1] != 0 or b[3] != ysize
    _, bh, kh = precompute_coeffs(w_in, b[0], b[2], xsize)
    _, bv, kv = precompute_coeffs(h_in, b[1], b[3], ysize)
    out = None
    if need_h:
        y0 = int(bv[0, 0])
        y1 = int(bv[ysize - 1, 0] + bv[ysize - 1, 1])
        bv = bv.copy()
        bv[:, 0] -= y0
        a = _pass(a[y0:y1], 1, bh, kh)
        out = a
    if need_v:
        out = _pass(a, 0, bv, kv)
    if out is None:
        out = a.copy()
    return out


def premultiply(rgba: np.ndarray) -> np.ndarray:
    """RGBA -> RGBa (MULDIV255; the SSE and scalar paths are the same integer formula)."""
    v = rgba.astype(np.int32)
    t = v[:, :, :3] * v[:, :, 3:4] + 128
    out = rgba.copy()
    out[:, :, :3] = ((t + (t >> 8)) >> 8).astype(np.uint8)
    return out


def unpremultiply(rgba_pm: np.ndarray) -> np.ndarray:
    """RGBa -> RGBA as Pillow-SIMD (SSE build): groups of 4 pixels via RCP_FACTOR, last width % 4 per row exact."""
    if RCP_FACTOR is None:
        raise RuntimeError('RCP_FACTOR not loaded (RGBA path needs the measured rcpps table)')
    h, w = rgba_pm.shape[:2]
    v = rgba_pm[:, :, :3].astype(np.int64)
    al = rgba_pm[:, :, 3].astype(np.int64)
    f = RCP_FACTOR.astype(np.int64)[al][:, :, None]
    r = ((v << 8) * f) >> 16  # _mm_mulhi_epu16 -> uint16
    simd = np.where(r >= 32768, 0, np.minimum(r, 255))  # _mm_packus_epi16 reads int16: >= 2^15 is negative -> 0
    with np.errstate(divide='ignore', invalid='ignore'):
        sc = np.where((al[:, :, None] == 0) | (al[:, :, None] == 255), v,
                      np.minimum((255 * v) // np.maximum(al[:, :, None], 1), 255))
    n4 = (w // 4) * 4
    out = rgba_pm.copy()
    out[:, :n4, :3] = simd[:, :n4].astype(np.uint8)
    out[:, n4:, :3] = sc[:, n4:].astype(np.uint8)
    return out


def resize(a: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    """Image.resize(size, LANCZOS) for mode in RGB / L / RGBA; a is the uint8 array of that image."""
    h, w = a.shape[:2]
    if (w, h) == (int(size[0]), int(size[1])):
        return a.copy()
    if mode == 'RGBA':
        return unpremultiply(resample(premultiply(a), size))
    if mode in ('RGB', 'L'):
        return resample(a, size)
    raise ValueError(f'mode {mode} not covered')


def load_rcp_table(path: str) -> None:
    """Load the measured 256-entry factor table (one int per line, alpha 0..255)."""
    global RCP_FACTOR
    t = np.array([int(x) for x in open(path).read().split()], np.int64)
    if t.shape != (256,):
        raise ValueError('rcpps table must have 256 entries')
    RCP_FACTOR = t
