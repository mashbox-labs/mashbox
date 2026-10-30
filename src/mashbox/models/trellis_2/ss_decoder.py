"""TRELLIS.2 stage 3: sparse-structure decoder in JAX (our own code).

Matches the math of trellis2/models/sparse_structure_vae.py::SparseStructureDecoder (TRELLIS.2 @75fbf01, MIT) and the
post-processing in trellis2/pipelines/trellis2_image_to_3d.py::sample_sparse_structure (threshold > 0, max_pool3d,
argwhere). Layout is channels-last (NDHWC) for the TPU; the public API takes and returns the reference layout
(NCDHW) so stage fixtures can be fed unchanged.

Weights: microsoft/TRELLIS-image-large @25e0d31 ckpts/ss_dec_conv3d_16l8_fp16 (MIT). Loaded with a small numpy
safetensors reader (no extra dependency).

Precision model (mirrors the reference):
  input_layer and out_layer always run in float32 (the reference keeps them fp32 even with use_fp16=True);
  the torso (middle_block + blocks) runs in `torso_dtype` (float32 for parity, bfloat16 on chip, float16 = vendor);
  ChannelLayerNorm32 upcasts to float32, normalises over channels, casts back to the torso dtype.
"""
import json
import struct
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

LN_EPS = 1e-5  # nn.LayerNorm default
_DN = ('NDHWC', 'DHWIO', 'NDHWC')

# ---------------------------------------------------------------- weights


def read_safetensors_bytes(buf):
    """Minimal safetensors parser over an in-memory buffer -> {name: np.ndarray} (dtype preserved)."""
    dt = {'F16': np.float16, 'F32': np.float32, 'F64': np.float64}
    n = struct.unpack('<Q', bytes(buf[:8]))[0]
    hdr = json.loads(bytes(buf[8:8 + n]))
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        if v['dtype'] == 'BF16':
            import ml_dtypes
            d = ml_dtypes.bfloat16
        else:
            d = dt[v['dtype']]
        s, e = v['data_offsets']
        out[k] = np.frombuffer(buf[base + s:base + e], dtype=d).reshape(v['shape']).copy()
    return out


def read_safetensors(path):
    """Minimal safetensors reader -> {name: np.ndarray} (dtype preserved)."""
    with open(path, 'rb') as f:
        return read_safetensors_bytes(memoryview(f.read()))


def default_config():
    return dict(out_channels=1, latent_channels=8, num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32])


def block_plan(cfg):
    """Ordered list of ('res', idx) / ('up', idx) for the `blocks` ModuleList, same order as the reference __init__."""
    plan, i = [], 0
    chs = cfg['channels']
    for li, ch in enumerate(chs):
        for _ in range(cfg['num_res_blocks']):
            plan.append(('res', i)); i += 1
        if li < len(chs) - 1:
            plan.append(('up', i)); i += 1
    return plan


def _conv_w(w):
    # torch Conv3d weight [O, I, kd, kh, kw] -> lax DHWIO [kd, kh, kw, I, O]
    return np.ascontiguousarray(np.transpose(w, (2, 3, 4, 1, 0)))


def convert_params(sd, cfg=None, torso_dtype=jnp.float32):
    """torch state_dict (numpy) -> nested dict of jnp arrays.

    Conv weights of the torso are stored in `torso_dtype` (the reference stores them fp16 and runs fp16; for the fp32
    parity run they are upcast exactly like convert_to_fp32()). Norm params and the input/out layers stay float32.
    """
    cfg = cfg or default_config()
    f32 = jnp.float32

    def conv(prefix, dtype):
        return {'w': jnp.asarray(_conv_w(sd[prefix + '.weight'].astype(np.float32)), dtype=dtype),
                'b': jnp.asarray(sd[prefix + '.bias'].astype(np.float32), dtype=dtype)}

    def norm(prefix):
        return {'g': jnp.asarray(sd[prefix + '.weight'], f32), 'b': jnp.asarray(sd[prefix + '.bias'], f32)}

    def res(prefix):
        return {'norm1': norm(prefix + '.norm1'), 'conv1': conv(prefix + '.conv1', torso_dtype),
                'norm2': norm(prefix + '.norm2'), 'conv2': conv(prefix + '.conv2', torso_dtype)}

    p = {'input_layer': conv('input_layer', f32),
         'middle': [res(f'middle_block.{i}') for i in range(cfg['num_res_blocks_middle'])],
         'blocks': [],
         'out_norm': norm('out_layer.0'),
         'out_conv': conv('out_layer.2', f32)}
    for kind, i in block_plan(cfg):
        p['blocks'].append(res(f'blocks.{i}') if kind == 'res' else {'conv': conv(f'blocks.{i}.conv', torso_dtype)})
    return p


def load_params(path_prefix, torso_dtype=jnp.float32):
    cfg = json.load(open(path_prefix + '.json'))['args']
    sd = read_safetensors(path_prefix + '.safetensors')
    return convert_params(sd, cfg, torso_dtype), cfg


def check_keys(sd, cfg):
    """Every tensor of the checkpoint is consumed by convert_params and nothing is missing (strict load)."""
    want = {'input_layer.weight', 'input_layer.bias', 'out_layer.0.weight', 'out_layer.0.bias',
            'out_layer.2.weight', 'out_layer.2.bias'}
    rb = lambda p: {f'{p}.{a}.{b}' for a in ('norm1', 'conv1', 'norm2', 'conv2') for b in ('weight', 'bias')}  # noqa: E731
    for i in range(cfg['num_res_blocks_middle']):
        want |= rb(f'middle_block.{i}')
    for kind, i in block_plan(cfg):
        want |= rb(f'blocks.{i}') if kind == 'res' else {f'blocks.{i}.conv.weight', f'blocks.{i}.conv.bias'}
    return sorted(want - set(sd)), sorted(set(sd) - want)

# ---------------------------------------------------------------- layers


def conv3d(x, p, precision):
    """Same-padded 3D cross-correlation (torch Conv3d semantics), step 1, channels-last."""
    k = p['w'].shape[0]
    pad = (k - 1) // 2
    y = lax.conv_general_dilated(x, p['w'], window_strides=(1, 1, 1), padding=[(pad, pad)] * 3,
                                 dimension_numbers=_DN, precision=precision,
                                 preferred_element_type=jnp.float32 if x.dtype != jnp.float32 else None)
    return (y + p['b'].astype(y.dtype)).astype(x.dtype)


def channel_ln32(x, p):
    """ChannelLayerNorm32: LayerNorm over C in float32, cast back to the input dtype."""
    dt = x.dtype
    x = x.astype(jnp.float32)
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
    y = (x - mu) * lax.rsqrt(var + LN_EPS) * p['g'] + p['b']
    return y.astype(dt)


def silu(x):
    return x * jax.nn.sigmoid(x)


def res_block(x, p, precision):
    h = silu(channel_ln32(x, p['norm1']))
    h = conv3d(h, p['conv1'], precision)
    h = silu(channel_ln32(h, p['norm2']))
    h = conv3d(h, p['conv2'], precision)
    return h + x  # channels equal inside every decoder ResBlock -> identity skip


def pixel_shuffle_3d(x, r=2):
    """Channels-last version of trellis2.modules.spatial.pixel_shuffle_3d.

    Reference (NCHWD): channel c = c_*r^3 + i*r^2 + j*r + k  ->  output voxel (h*r+i, w*r+j, d*r+k), channel c_.
    """
    B, H, W, D, C = x.shape
    c_ = C // (r ** 3)
    x = x.reshape(B, H, W, D, c_, r, r, r)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # B, H, i, W, j, D, k, c_
    return x.reshape(B, H * r, W * r, D * r, c_)


def upsample_block(x, p, precision):
    return pixel_shuffle_3d(conv3d(x, p['conv'], precision), 2)

# ---------------------------------------------------------------- model


@partial(jax.jit, static_argnames=('torso_dtype', 'precision'))
def decode_logits_ndhwc(params, z, torso_dtype=jnp.float32, precision=lax.Precision.HIGHEST):
    """z [B,16,16,16,8] float32 (channels-last) -> logits [B,64,64,64,1] float32."""
    h = conv3d(z.astype(jnp.float32), params['input_layer'], precision)
    h = h.astype(torso_dtype)
    for p in params['middle']:
        h = res_block(h, p, precision)
    for p in params['blocks']:
        h = res_block(h, p, precision) if 'norm1' in p else upsample_block(h, p, precision)
    h = h.astype(jnp.float32)
    h = silu(channel_ln32(h, params['out_norm']))
    return conv3d(h, params['out_conv'], precision)


def decode_trace(params, z_ncdhw, torso_dtype=jnp.float32, precision=lax.Precision.HIGHEST):
    """Un-jitted debug path: returns [(name, NCDHW float32 np array)] after input_layer, every middle/blocks entry
    and the final logits, named like the torch submodules so hooks can be matched one to one."""
    out = []
    to_ref = lambda a: np.asarray(jnp.transpose(a.astype(jnp.float32), (0, 4, 1, 2, 3)))  # noqa: E731
    h = conv3d(jnp.transpose(jnp.asarray(z_ncdhw, jnp.float32), (0, 2, 3, 4, 1)), params['input_layer'], precision)
    out.append(('input_layer', to_ref(h)))
    h = h.astype(torso_dtype)
    for i, p in enumerate(params['middle']):
        h = res_block(h, p, precision)
        out.append((f'middle_block.{i}', to_ref(h)))
    for i, p in enumerate(params['blocks']):
        h = res_block(h, p, precision) if 'norm1' in p else upsample_block(h, p, precision)
        out.append((f'blocks.{i}', to_ref(h)))
    h = h.astype(jnp.float32)
    h = conv3d(silu(channel_ln32(h, params['out_norm'])), params['out_conv'], precision)
    out.append(('out_layer', to_ref(h)))
    return out


def decode_logits(params, z_ncdhw, torso_dtype=jnp.float32, precision=lax.Precision.HIGHEST):
    """Reference-layout wrapper: z [B,8,16,16,16] -> logits [B,1,64,64,64]."""
    z = jnp.transpose(jnp.asarray(z_ncdhw, jnp.float32), (0, 2, 3, 4, 1))
    y = decode_logits_ndhwc(params, z, torso_dtype=torso_dtype, precision=precision)
    return jnp.transpose(y, (0, 4, 1, 2, 3))

# ---------------------------------------------------------------- post-processing (sample_sparse_structure)


def occupancy(logits_ncdhw, resolution=32):
    """logits [B,1,R,R,R] -> bool occupancy [B,resolution^3 grid] exactly as the reference:
    decoded = logits > 0; if resolution != R: max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5 (= any over the block)."""
    occ = logits_ncdhw[:, 0] > 0
    B, R = occ.shape[0], occ.shape[1]
    if resolution != R:
        r = R // resolution
        occ = occ.reshape(B, resolution, r, resolution, r, resolution, r).any(axis=(2, 4, 6))
    return occ


@partial(jax.jit, static_argnames=('cap',))
def coords_padded(occ, cap):
    """Fixed-cap padded nonzero of occ [B,X,Y,Z] -> (coords [cap,4] int32 in argwhere order (b,x,y,z), n, overflow).

    Rows >= n are -1. Order equals torch.argwhere (row-major), which is what the reference returns. Implemented as a
    cumsum compaction (prefix sum -> scatter), which is static-shaped and maps to plain XLA ops on TPU.
    cap = B*X*Y*Z can never overflow (32768 per sample at the 32^3 grid); smaller caps report overflow=True.
    """
    B, X, Y, Z = occ.shape
    flat = occ.reshape(-1)
    n = jnp.sum(flat, dtype=jnp.int32)
    pos = jnp.cumsum(flat, dtype=jnp.int32) - 1
    idx = jnp.arange(flat.shape[0], dtype=jnp.int32)
    tgt = jnp.where(flat & (pos < cap), pos, cap)  # cap = out-of-range slot, dropped
    lin = jnp.full((cap,), -1, jnp.int32).at[tgt].set(idx, mode='drop')
    valid = lin >= 0
    b = lin // (X * Y * Z)
    x = (lin // (Y * Z)) % X
    y = (lin // Z) % Y
    z = lin % Z
    coords = jnp.where(valid[:, None], jnp.stack([b, x, y, z], axis=1), -1)
    return coords, n, n > cap


def decode_to_coords(params, z_ncdhw, resolution=32, cap=None, torso_dtype=jnp.float32,
                     precision=lax.Precision.HIGHEST):
    """Full stage 3: latent -> logits -> occupancy -> pooled occupancy -> padded coords."""
    logits = decode_logits(params, z_ncdhw, torso_dtype, precision)
    occ = occupancy(logits, resolution)
    cap = cap or int(np.prod(occ.shape))
    coords, n, overflow = coords_padded(occ, cap)
    return dict(logits=logits, occ=occ, coords=coords, n=n, overflow=overflow)
