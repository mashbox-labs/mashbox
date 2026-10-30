"""DINOv3 ViT-L/16 image encoder in JAX (TRELLIS.2 stage 1, image conditioning).

Our own JAX code. The math follows the official TRELLIS.2 DinoV3FeatureExtractor.extract_features (MIT, pin 75fbf01):
embeddings -> rope -> 24 x DINOv3ViTLayer -> F.layer_norm WITHOUT affine (the model's own final `norm` is NOT used).
Layer math matches the HF DINOv3ViTModel definition (read, not copied): pre-LN blocks, layer scale, exact GELU MLP,
RoPE on patch tokens only (cls + 4 registers untouched), softmax attention with scale head_dim**-0.5.

Weights: HF facebook/dinov3-vitl16-pretrain-lvd1689m @ ea8dc2863c51be0a264bab82070e3e8836b02d51 (DINOv3 License).
They are read by our own small safetensors reader (numpy only, no extra dependency).

Shapes: [B,3,512,512] -> [B,1029,1024]; [B,3,1024,1024] -> [B,4101,1024] (1 cls + 4 registers + (H/16)*(W/16) patches).
"""
from __future__ import annotations

import json
import math
import struct
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

# Config of the pinned checkpoint (config.json at ea8dc28). Checked against the file in load_params.
CFG = dict(hidden=1024, heads=16, layers=24, mlp=4096, patch=16, registers=4, eps=1e-5, rope_theta=100.0)
N_PREFIX = 1 + CFG['registers']  # cls + registers, never rotated

_ST_DTYPES = {'F32': np.float32, 'F16': np.float16, 'BF16': None, 'F64': np.float64}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Minimal safetensors reader (format: u64 header length, JSON header, raw little-endian data). Memory-mapped."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode='r', offset=8 + n)
    out = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        if v['dtype'] == 'BF16':
            import ml_dtypes  # ships with jax (Apache-2.0)
            dt = ml_dtypes.bfloat16
        else:
            dt = _ST_DTYPES[v['dtype']]
        out[k] = np.frombuffer(mm[a:b], dtype=dt).reshape(v['shape'])
    return out


def load_params(weights_dir: str, dtype=jnp.float32) -> dict:
    """Load the HF checkpoint into a JAX pytree. Per-layer weights are stacked on a leading axis [24, ...] for lax.scan.
    Linear weights are stored transposed to [in, out] so x @ W matches torch's x @ W.T."""
    cfg = json.load(open(f'{weights_dir}/config.json'))
    assert cfg['hidden_size'] == CFG['hidden'] and cfg['num_attention_heads'] == CFG['heads']
    assert cfg['num_hidden_layers'] == CFG['layers'] and cfg['intermediate_size'] == CFG['mlp']
    assert cfg['patch_size'] == CFG['patch'] and cfg['num_register_tokens'] == CFG['registers']
    assert cfg['layer_norm_eps'] == CFG['eps'] and cfg['rope_theta'] == CFG['rope_theta']
    assert cfg['hidden_act'] == 'gelu' and not cfg['use_gated_mlp'] and not cfg['key_bias']
    sd = read_safetensors(f'{weights_dir}/model.safetensors')
    L, D = CFG['layers'], CFG['hidden']
    f32 = lambda a: np.asarray(a, dtype=np.float32)

    def stack(name, transpose=False):
        arrs = [f32(sd[f'layer.{i}.{name}']) for i in range(L)]
        if transpose:
            arrs = [a.T for a in arrs]
        return np.stack(arrs)

    zeros_kb = np.zeros((L, D), np.float32)  # key_bias=False in the checkpoint
    p = {
        'patch_w': f32(sd['embeddings.patch_embeddings.weight']).reshape(D, -1).T,  # [3*16*16, D], (c, kh, kw) order
        'patch_b': f32(sd['embeddings.patch_embeddings.bias']),
        'cls': f32(sd['embeddings.cls_token']).reshape(1, D),
        'reg': f32(sd['embeddings.register_tokens']).reshape(CFG['registers'], D),
        'layers': {
            'ln1_w': stack('norm1.weight'), 'ln1_b': stack('norm1.bias'),
            'ln2_w': stack('norm2.weight'), 'ln2_b': stack('norm2.bias'),
            # fused qkv [L, D, 3D]: one GEMM on the MXU instead of three
            'qkv_w': np.concatenate([stack('attention.q_proj.weight', True), stack('attention.k_proj.weight', True),
                                     stack('attention.v_proj.weight', True)], axis=-1),
            'qkv_b': np.concatenate([stack('attention.q_proj.bias'), zeros_kb, stack('attention.v_proj.bias')], axis=-1),
            'o_w': stack('attention.o_proj.weight', True), 'o_b': stack('attention.o_proj.bias'),
            'ls1': stack('layer_scale1.lambda1'), 'ls2': stack('layer_scale2.lambda1'),
            'up_w': stack('mlp.up_proj.weight', True), 'up_b': stack('mlp.up_proj.bias'),
            'down_w': stack('mlp.down_proj.weight', True), 'down_b': stack('mlp.down_proj.bias'),
        },
    }
    return jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=dtype), p)


def rope_tables(h_patches: int, w_patches: int) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin [N, head_dim] for patch tokens, computed in float32 like the reference (eval mode: no coord augmentation).
    coords = patch centres in [-1, 1] as (y, x); angles = 2*pi*coord*inv_freq, laid out [y freqs | x freqs] then tiled x2."""
    hd = CFG['hidden'] // CFG['heads']
    inv_freq = (1.0 / (np.float32(CFG['rope_theta']) ** np.arange(0, 1, 4 / hd, dtype=np.float32))).astype(np.float32)
    ch = (np.arange(0.5, h_patches, dtype=np.float32) / np.float32(h_patches)).astype(np.float32)
    cw = (np.arange(0.5, w_patches, dtype=np.float32) / np.float32(w_patches)).astype(np.float32)
    yy, xx = np.meshgrid(ch, cw, indexing='ij')
    coords = np.stack([yy, xx], -1).reshape(-1, 2).astype(np.float32)
    coords = (np.float32(2.0) * coords - np.float32(1.0)).astype(np.float32)
    ang = (np.float32(2 * math.pi) * coords[:, :, None] * inv_freq[None, None, :]).astype(np.float32)
    ang = ang.reshape(ang.shape[0], -1)
    ang = np.concatenate([ang, ang], -1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _layer_norm(x, w=None, b=None, eps=CFG['eps']):
    mu = jnp.mean(x, -1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), -1, keepdims=True)
    y = (x - mu) * jax.lax.rsqrt(var + eps)
    if w is not None:
        y = y * w + b
    return y


def _rope(x, cos, sin):
    """x [B, H, T, hd]; rotate only tokens N_PREFIX.. (patch tokens). rotate_half = cat(-x2, x1)."""
    pre, pat = x[:, :, :N_PREFIX], x[:, :, N_PREFIX:N_PREFIX + cos.shape[0]]
    rest = x[:, :, N_PREFIX + cos.shape[0]:]  # padding tokens (if any) stay as they are
    h = pat.shape[-1] // 2
    rot = jnp.concatenate([-pat[..., h:], pat[..., :h]], -1)
    pat = pat * cos + rot * sin
    return jnp.concatenate([pre, pat, rest], axis=2)


def attention_xla(q, k, v, n_valid: int):
    """Plain softmax attention, q/k/v [B, H, T, hd]. Keys >= n_valid (padding) are masked out."""
    scale = q.shape[-1] ** -0.5
    s = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    T = k.shape[2]
    if n_valid < T:
        s = jnp.where((jnp.arange(T) < n_valid)[None, None, None, :], s, -jnp.inf)
    pr = jax.nn.softmax(s, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', pr, v)


ATTN = {'xla': attention_xla}


def _block(x, lp, cos, sin, n_valid, attn):
    B, T, D = x.shape
    H = CFG['heads']
    hd = D // H
    h = _layer_norm(x, lp['ln1_w'], lp['ln1_b'])
    qkv = h @ lp['qkv_w'] + lp['qkv_b']
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = (t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = attn(q, k, v, n_valid).transpose(0, 2, 1, 3).reshape(B, T, D)
    x = x + (o @ lp['o_w'] + lp['o_b']) * lp['ls1']
    h = _layer_norm(x, lp['ln2_w'], lp['ln2_b'])
    h = jax.nn.gelu(h @ lp['up_w'] + lp['up_b'], approximate=False)  # HF "gelu" = exact erf GELU
    x = x + (h @ lp['down_w'] + lp['down_b']) * lp['ls2']
    return x


def embed(params, img):
    """img [B, 3, H, W] (already ImageNet-normalised) -> tokens [B, 5 + N, D]. The stride-16 16x16 conv is a GEMM."""
    B, C, Hh, Ww = img.shape
    P = CFG['patch']
    hp, wp = Hh // P, Ww // P
    x = img[:, :, :hp * P, :wp * P].reshape(B, C, hp, P, wp, P)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(B, hp * wp, C * P * P)  # (c, kh, kw) like the conv weight
    pe = x @ params['patch_w'] + params['patch_b']
    D = pe.shape[-1]
    cls = jnp.broadcast_to(params['cls'][None], (B, 1, D)).astype(pe.dtype)
    reg = jnp.broadcast_to(params['reg'][None], (B, CFG['registers'], D)).astype(pe.dtype)
    return jnp.concatenate([cls, reg, pe], axis=1)


@partial(jax.jit, static_argnames=('attn_impl', 'pad_multiple', 'return_layers'))
def encode(params, img, attn_impl: str = 'xla', pad_multiple: int = 0, return_layers: bool = False):
    """TRELLIS.2 stage 1: img [B,3,H,W] float (normalised) -> cond [B, 5+N, 1024] = layer_norm_noaffine(last hidden).
    pad_multiple > 0 pads the token axis to a multiple of it (padding keys masked) for block-tiled TPU kernels;
    the padded rows are dropped before return."""
    B, _, Hh, Ww = img.shape
    hp, wp = Hh // CFG['patch'], Ww // CFG['patch']
    cos, sin = rope_tables(hp, wp)
    cos, sin = jnp.asarray(cos, img.dtype), jnp.asarray(sin, img.dtype)
    x = embed(params, img)
    T = x.shape[1]
    Tp = T if pad_multiple <= 0 else ((T + pad_multiple - 1) // pad_multiple) * pad_multiple
    if Tp > T:
        x = jnp.pad(x, ((0, 0), (0, Tp - T), (0, 0)))
    attn = ATTN[attn_impl]

    def step(x, lp):
        y = _block(x, lp, cos, sin, T, attn)
        return y, (y[:, :T] if return_layers else None)

    x, per_layer = jax.lax.scan(step, x, params['layers'])
    out = _layer_norm(x[:, :T])
    return (out, per_layer) if return_layers else out
