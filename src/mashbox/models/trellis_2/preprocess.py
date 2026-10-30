"""Host-side input prep for TRELLIS.2 stage 1 (numpy + Pillow, no torch).

Same semantics as the official MIT code at pin 75fbf01:
- Trellis2ImageTo3DPipeline.preprocess_image, alpha path only (RGBA input with non-opaque alpha; rembg is never run here):
  downscale so max side <= 1024 (LANCZOS), square crop around the alpha>0.8 bbox, premultiply RGB by alpha.
- DinoV3FeatureExtractor.__call__ for a PIL list: resize to (S, S) LANCZOS, RGB float32 /255, ImageNet normalise.
Pillow 13.0.0: MIT-CMU as published; numpy 2.5.4: BSD-3-Clause (plus bundled 0BSD/MIT/Zlib/CC0 parts) as published.

Both LANCZOS resizes go through our resample_u8 by default (TRELLIS_S0_RESIZE=ours): it reproduces the reference env's
Pillow-SIMD 12.1.1.post0 output bit for bit, so stage 0 no
longer depends on which Pillow build is installed. TRELLIS_S0_RESIZE=pillow restores the installed Pillow's resize.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from . import resample_u8 as R

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE = os.environ.get('TRELLIS_S0_RESIZE', 'ours')
R.load_rcp_table(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rcpps_factor_sse_intel.txt'))


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Image.resize(size, LANCZOS): ours (default) or the installed Pillow."""
    if RESIZE == 'pillow':
        return img.resize(size, Image.Resampling.LANCZOS)
    if RESIZE != 'ours':
        raise ValueError(f'TRELLIS_S0_RESIZE={RESIZE!r} (ours | pillow)')
    return Image.fromarray(R.resize(np.array(img), size, img.mode))


def preprocess_rgba(img: Image.Image) -> Image.Image:
    """stage 0 for RGBA inputs. Raises if there is no usable alpha (background removal is out of scope for this module)."""
    if img.mode != 'RGBA' or np.all(np.array(img)[:, :, 3] == 255):
        raise ValueError('RGBA input with a real alpha channel required (rembg path not ported here)')
    scale = min(1, 1024 / max(img.size))
    if scale < 1:
        img = _resize(img, (int(img.width * scale), int(img.height * scale)))
    a = np.array(img)
    ys, xs = np.nonzero(a[:, :, 3] > 0.8 * 255)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0) * 1)
    box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    img = img.crop(box)
    f = np.array(img).astype(np.float32) / 255
    f = f[:, :, :3] * f[:, :, 3:4]
    return Image.fromarray((f * 255).astype(np.uint8))


def to_encoder_input(imgs: list[Image.Image], size: int) -> np.ndarray:
    """PIL list -> float32 [B, 3, size, size], ImageNet-normalised (what the encoder sees)."""
    out = []
    for im in imgs:
        im = _resize(im, (size, size))
        x = np.array(im.convert('RGB')).astype(np.float32) / 255
        out.append(x.transpose(2, 0, 1))
    x = np.stack(out)
    return ((x - MEAN[None, :, None, None]) / STD[None, :, None, None]).astype(np.float32)
