"""Host-side pipeline glue for TRELLIS.2 (numpy + Pillow only, no torch, no jax).

stage 0 preprocess, image-cond input prep, BiRefNet pre/post wrapper, SLat (de)normalisation,
cascade token quantisation, sparse padding. Written from Trellis2ImageTo3DPipeline,
DinoV3FeatureExtractor and pipelines/rembg/BiRefNet.py at TRELLIS.2 commit 75fbf018 (MIT).
Pillow licence as published: MIT-CMU (HPND).
"""
import json
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- stage 0 preprocess
def needs_rembg(img: Image.Image) -> bool:
    """Official rule: only an RGBA image whose alpha is not all 255 skips background removal."""
    if img.mode == "RGBA":
        return bool(np.all(np.asarray(img)[:, :, 3] == 255))
    return True


def preprocess_image(img: Image.Image, rembg_fn: Optional[Callable[[Image.Image], Image.Image]] = None) -> Image.Image:
    """Same output as Trellis2ImageTo3DPipeline.preprocess_image.

    1. has_alpha decided on the ORIGINAL image (RGBA and alpha not all 255);
    2. downscale so max side <= 1024 (Pillow LANCZOS, int() truncation of the new size);
    3. no alpha -> rembg_fn(RGB) must return RGBA (official: BiRefNet). rembg_fn=None raises;
    4. bbox of alpha > 204, square of side max(w, h) (pixel-index extents) around the bbox centre,
       corners rounded half-to-even (Pillow crop rounds float boxes), out-of-image area = 0;
    5. RGB premultiplied by alpha in float32, *255, truncated to uint8 -> RGB image.
    """
    has_alpha = not needs_rembg(img)
    max_size = max(img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        out = img
    else:
        if rembg_fn is None:
            raise ValueError("input has no usable alpha; pass rembg_fn (BiRefNet) or an RGBA image")
        out = rembg_fn(img.convert("RGB"))
    arr = np.asarray(out)
    alpha = arr[:, :, 3]
    ys, xs = np.nonzero(alpha > 0.8 * 255)
    if xs.size == 0:
        raise ValueError("alpha mask is empty (no pixel with alpha > 204)")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0)
    box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    bx0, by0, bx1, by1 = (int(round(v)) for v in box)   # Python round = half-to-even, as Pillow
    crop = _crop_zero_pad(arr, bx0, by0, bx1, by1)
    f = crop.astype(np.float32) / 255
    rgb = f[:, :, :3] * f[:, :, 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _crop_zero_pad(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = arr.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + arr.shape[2:], dtype=arr.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


# ---------------------------------------------------------------- stage 1 input prep
def image_cond_input(images, size: int) -> np.ndarray:
    """DinoV3FeatureExtractor.__call__ input prep: LANCZOS to size x size, RGB, /255, ImageNet norm.
    Returns float32 [B, 3, size, size] (NCHW, as the official encoder takes)."""
    out = []
    for im in images:
        im = im.resize((size, size), Image.LANCZOS)
        a = np.asarray(im.convert("RGB")).astype(np.float32) / 255
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        out.append(a.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)


def neg_cond_like(cond: np.ndarray) -> np.ndarray:
    return np.zeros_like(cond)


# ---------------------------------------------------------------- rembg wrapper (BiRefNet)
def birefnet_input(rgb: Image.Image) -> np.ndarray:
    """transforms.Resize((1024,1024)) [Pillow BILINEAR on a PIL input] -> ToTensor -> Normalize.
    float32 [1, 3, 1024, 1024]."""
    im = rgb.resize((1024, 1024), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32) / 255
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def birefnet_apply_mask(rgb: Image.Image, logits: np.ndarray) -> Image.Image:
    """sigmoid(last output) -> ToPILImage (x255, uint8 truncation, mode L) -> resize to the
    image size (Pillow default filter, BICUBIC) -> putalpha. logits: [1024, 1024] float32."""
    pred = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    mask = Image.fromarray((pred.astype(np.float32) * 255).astype(np.uint8), mode="L")
    mask = mask.resize(rgb.size)
    out = rgb.copy()
    out.putalpha(mask)
    return out


# ---------------------------------------------------------------- latents
def load_normalization(path: str) -> dict:
    a = json.load(open(path))["args"]
    return {k: {s: np.asarray(a[f"{k}_normalization"][s], dtype=np.float32) for s in ("mean", "std")}
            for k in ("shape_slat", "tex_slat")}


def slat_denormalize(feats, norm):
    """slat * std + mean (sampler output -> decoder input)."""
    return feats * norm["std"][None] + norm["mean"][None]


def slat_normalize(feats, norm):
    """(slat - mean) / std (shape slat -> tex sampler concat_cond)."""
    return (feats - norm["mean"][None]) / norm["std"][None]


def cascade_quantize(hr_coords: np.ndarray, lr_resolution: int, resolution: int,
                     max_num_tokens: int = 49152) -> Tuple[np.ndarray, int]:
    """sample_shape_slat_cascade token step: quantise upsampled coords to the hr grid, unique
    (lexicographic, as torch.unique(dim=0)), and drop resolution by 128 until N < max tokens
    (or the resolution reaches 1024). float32 math and int truncation as torch .int()."""
    hr_res = resolution
    b = hr_coords[:, :1].astype(np.int32)
    xyz = hr_coords[:, 1:].astype(np.float32)
    while True:
        q = ((xyz + np.float32(0.5)) / np.float32(lr_resolution) * np.float32(hr_res // 16)).astype(np.int32)
        coords = np.unique(np.concatenate([b, q], axis=1), axis=0)
        if coords.shape[0] < max_num_tokens or hr_res == 1024:
            break
        hr_res -= 128
    return coords, hr_res


def pad_tokens(feats_list, bucket: int):
    """List of [N_i, C] -> (x [B, bucket, C], mask [B, bucket]) for the sparse sampler."""
    C = feats_list[0].shape[1]
    x = np.zeros((len(feats_list), bucket, C), dtype=feats_list[0].dtype)
    m = np.zeros((len(feats_list), bucket), dtype=bool)
    for i, f in enumerate(feats_list):
        if f.shape[0] > bucket:
            raise ValueError(f"{f.shape[0]} tokens > bucket {bucket}")
        x[i, :f.shape[0]] = f
        m[i, :f.shape[0]] = True
    return x, m


def unpad_tokens(x, mask):
    return [np.asarray(x[i])[np.asarray(mask[i])] for i in range(x.shape[0])]
