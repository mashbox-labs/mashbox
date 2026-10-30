"""TRELLIS.2 sparse 3D conv stack in JAX (replaces FlexGEMM sparse_submanifold_conv3d + o-voxel hashmap).

Written from the documented semantics of the official op (TRELLIS.2 @75fbf01 and the FlexGEMM @6dd94a8 (LICENSE-FlexGEMM beside this module) explicit GEMM
definition it calls), not from any CUDA/Triton source:

  out[i] = bias + sum_{k} W[:, a, b, c, :] @ in[j(i, k)]     k = a*9 + b*3 + c, a,b,c in {0,1,2}
  j(i, k) = index of the voxel at coords[i] + (a-1, b-1, c-1) (x, y, z order; correlation, no flip), zero if absent.

Weight layout as stored in the checkpoints: (Co, 3, 3, 3, Ci), axis 1 <-> x (coords[:,1] in TRELLIS), axis 2 <-> y, axis 3 <-> z.

Data layout used here: one batch item at a time (the pipeline runs batch 1). coords int32 [N, 3] (x, y, z), feats [N, C],
valid bool [N] (padded rows are False so N can be a static bucket size). All index work is sort + searchsorted (no hashmap).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

# 27 offsets in the official kernel order: meshgrid(ij) over (x, y, z), x slowest.
OFFSETS = np.stack(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2), indexing='ij'), -1).reshape(27, 3).astype(np.int32)
SENTINEL = np.int32(np.iinfo(np.int32).max)
MAX_GRID = 1024  # (1024 + 2)^3 < 2^31, so keys stay int32 (TPU-friendly, no x64)


# ----------------------------------------------------------------------------------------------------------------------
# Neighbour map (replaces o-voxel / FlexGEMM hashmap build)
# ----------------------------------------------------------------------------------------------------------------------
def _keys(c: jnp.ndarray, grid: int) -> jnp.ndarray:
    """Linear key of (x, y, z) with a 1-voxel guard band so coords in [-1, grid] never alias."""
    g = grid + 2
    c = c + 1
    return (c[..., 0] * g + c[..., 1]) * g + c[..., 2]


def neighbor_map(coords: jnp.ndarray, valid: jnp.ndarray, grid: int, offsets: np.ndarray = OFFSETS) -> jnp.ndarray:
    """[N, 27] int32 indices into feats; N (one past the end) means 'no neighbour' (gathers the zero row).

    Sort the N keys once, then 27 searchsorted queries per voxel. Padded rows have key SENTINEL and never match a real
    query (real keys < (grid+2)^3 < SENTINEL); padded rows themselves get only misses."""
    assert grid <= MAX_GRID, grid
    n = coords.shape[0]
    keys = jnp.where(valid, _keys(coords, grid), SENTINEL)
    order = jnp.argsort(keys)
    sk = keys[order]
    q = _keys(coords[:, None, :] + jnp.asarray(offsets)[None], grid)          # [N, 27]
    pos = jnp.searchsorted(sk, q.reshape(-1), method='sort').reshape(q.shape)
    pos = jnp.clip(pos, 0, n - 1)
    hit = (sk[pos] == q) & valid[:, None]
    return jnp.where(hit, order[pos], n).astype(jnp.int32)


# ----------------------------------------------------------------------------------------------------------------------
# Submanifold conv: gather -> GEMM (MXU) -> sum over 27 offsets
# ----------------------------------------------------------------------------------------------------------------------
def conv_weight_to_kio(w: np.ndarray) -> np.ndarray:
    """(Co, 3, 3, 3, Ci) checkpoint layout -> (27, Ci, Co)."""
    co, ci = w.shape[0], w.shape[-1]
    return np.ascontiguousarray(np.asarray(w).reshape(co, 27, ci).transpose(1, 2, 0))


def submconv(feats: jnp.ndarray, nbr: jnp.ndarray, wk: jnp.ndarray, bias: Optional[jnp.ndarray], mode: str = 'accumulate',
             precision=jax.lax.Precision.HIGHEST) -> jnp.ndarray:
    """feats [N, Ci], nbr [N, 27], wk [27, Ci, Co] -> [N, Co].

    mode 'explicit': im2col [N, 27*Ci] then one GEMM (the FlexGEMM explicit_gemm math; memory N*27*Ci).
    mode 'accumulate': 27 gathers + 27 GEMMs accumulated in fp32 (memory N*Ci)."""
    n, ci = feats.shape
    fp = jnp.concatenate([feats, jnp.zeros((1, ci), feats.dtype)], 0)
    if mode == 'explicit':
        g = fp[nbr].reshape(n, 27 * ci)
        out = jnp.dot(g, wk.reshape(27 * ci, -1), precision=precision, preferred_element_type=jnp.float32)
    elif mode == 'accumulate':
        def body(k, acc):
            g = jnp.take(fp, nbr[:, k], axis=0)
            return acc + jnp.dot(g, wk[k], precision=precision, preferred_element_type=jnp.float32)
        out = jax.lax.fori_loop(0, 27, body, jnp.zeros((n, wk.shape[-1]), jnp.float32))
    else:
        raise ValueError(mode)
    if bias is not None:
        out = out + bias.astype(jnp.float32)
    return out.astype(feats.dtype)


# ----------------------------------------------------------------------------------------------------------------------
# Channel2Spatial (C2S) subdivision with a static output cap
# ----------------------------------------------------------------------------------------------------------------------
def c2s_index(coords: jnp.ndarray, valid: jnp.ndarray, sub_mask: jnp.ndarray, cap: int):
    """Official SparseChannel2Spatial(2) order: children are emitted parent-major, child index s ascending, where
    s = x_bit + 2*y_bit + 4*z_bit. That is exactly the row-major order of the flattened [N, 8] mask, so one nonzero().

    Returns (flat_idx [cap] = parent*8 + s, new_coords [cap, 3], new_valid [cap], count)."""
    flat = (sub_mask & valid[:, None]).reshape(-1)
    count = flat.sum()
    (f,) = jnp.nonzero(flat, size=cap, fill_value=0)
    new_valid = jnp.arange(cap) < count
    parent, s = f // 8, f % 8
    bits = jnp.stack([s & 1, (s >> 1) & 1, (s >> 2) & 1], -1).astype(jnp.int32)
    new_coords = coords[parent] * 2 + bits
    new_coords = jnp.where(new_valid[:, None], new_coords, 0)
    return f.astype(jnp.int32), new_coords.astype(jnp.int32), new_valid, count


def c2s_feats(feats: jnp.ndarray, flat_idx: jnp.ndarray) -> jnp.ndarray:
    """[N, 8*C] -> [cap, C]: channel block s of parent p becomes child (p, s)."""
    n, c8 = feats.shape
    return jnp.take(feats.reshape(n * 8, c8 // 8), flat_idx, axis=0)


def s2c(feats: jnp.ndarray, coords: np.ndarray):
    """SparseSpatial2Channel(2) (encoder side; host numpy index, kept for completeness). Returns (feats [M, 8C], coords [M, 3])."""
    coords = np.asarray(coords)
    parent = coords // 2
    s = (coords[:, 0] % 2) + 2 * (coords[:, 1] % 2) + 4 * (coords[:, 2] % 2)
    uniq, inv = np.unique(parent, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    out = jnp.zeros((uniq.shape[0] * 8, feats.shape[1]), feats.dtype).at[inv * 8 + s].set(feats)
    return out.reshape(uniq.shape[0], -1), uniq.astype(np.int32)


# ----------------------------------------------------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------------------------------------------------
def layer_norm(x, w=None, b=None, eps=1e-6):
    x32 = x.astype(jnp.float32)
    mu = x32.mean(-1, keepdims=True)
    var = ((x32 - mu) ** 2).mean(-1, keepdims=True)
    y = (x32 - mu) * jax.lax.rsqrt(var + eps)
    if w is not None:
        y = y * w.astype(jnp.float32) + b.astype(jnp.float32)
    return y.astype(x.dtype)


def linear(x, w, b, precision=jax.lax.Precision.HIGHEST):
    """torch nn.Linear: w [out, in]."""
    y = jnp.dot(x, w.T, precision=precision, preferred_element_type=jnp.float32)
    if b is not None:
        y = y + b.astype(jnp.float32)
    return y.astype(x.dtype)


def convnext_block(p: Dict, x, nbr, mode='accumulate'):
    """SparseConvNeXtBlock3d: conv -> LN(affine, 1e-6) -> Linear(C,4C) -> SiLU -> Linear(4C,C) -> + x."""
    h = submconv(x, nbr, p['conv.wk'], p['conv.bias'], mode)
    h = layer_norm(h, p['norm.weight'], p['norm.bias'], 1e-6)
    h = jax.nn.silu(linear(h, p['mlp.0.weight'], p['mlp.0.bias']))
    h = linear(h, p['mlp.2.weight'], p['mlp.2.bias'])
    return h + x


def c2s_block_pre(p: Dict, x, nbr, pred_subdiv: bool, mode='accumulate'):
    """First half of SparseResBlockC2S3d (before the data-dependent subdivision). Returns (h [N, 8*Co], subdiv logits or None)."""
    subdiv = linear(x, p['to_subdiv.weight'], p['to_subdiv.bias']) if pred_subdiv else None
    h = jax.nn.silu(layer_norm(x, p['norm1.weight'], p['norm1.bias'], 1e-6))
    h = submconv(h, nbr, p['conv1.wk'], p['conv1.bias'], mode)
    return h, subdiv


def c2s_block_post(p: Dict, h, x, flat_idx, nbr_new, out_channels: int, mode='accumulate'):
    """Second half of SparseResBlockC2S3d, at the new (2x) resolution."""
    hs = c2s_feats(h, flat_idx)                                   # [cap, Co]
    xs = c2s_feats(x, flat_idx)                                   # [cap, Ci/8]
    hs = jax.nn.silu(layer_norm(hs, None, None, 1e-6))
    hs = submconv(hs, nbr_new, p['conv2.wk'], p['conv2.bias'], mode)
    rep = out_channels // xs.shape[1]
    return hs + jnp.repeat(xs, rep, axis=1)                       # torch repeat_interleave(dim=1)


# jitted kernels (static shapes; one compile per bucket size)
_nbr_jit = jax.jit(neighbor_map, static_argnames=('grid',))
_convnext_jit = jax.jit(convnext_block, static_argnames=('mode',))
_c2s_pre_jit = jax.jit(c2s_block_pre, static_argnames=('pred_subdiv', 'mode'))
_c2s_index_jit = jax.jit(c2s_index, static_argnames=('cap',))
_c2s_post_jit = jax.jit(c2s_block_post, static_argnames=('out_channels', 'mode'))


def bucket(n: int, quantum: int = 256) -> int:
    """Static row bucket: next multiple of `quantum` at or above max(n, 1), rounded up to 1/8 powers of two above 8*quantum
    so the number of compiled shapes stays small."""
    n = max(int(n), 1)
    b = -(-n // quantum) * quantum
    if b > 8 * quantum:
        step = 2 ** (int(math.log2(b)) - 3)
        b = -(-b // step) * step
    return b


# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
def load_decoder_params(state: Dict[str, np.ndarray], dtype=jnp.float32) -> Dict[str, jnp.ndarray]:
    """Checkpoint dict (numpy) -> flat dict with conv weights in (27, Ci, Co) layout under '<prefix>.wk'."""
    out = {}
    for k, v in state.items():
        v = np.asarray(v, np.float32)
        if k.endswith('.weight') and v.ndim == 5:
            out[k[:-len('.weight')] + '.wk'] = jnp.asarray(conv_weight_to_kio(v), dtype)
        else:
            out[k] = jnp.asarray(v, dtype)
    return out


def block_params(params: Dict, prefix: str) -> Dict:
    pl = prefix + '.'
    return {k[len(pl):]: v for k, v in params.items() if k.startswith(pl)}


# ----------------------------------------------------------------------------------------------------------------------
# SparseUnetVaeDecoder (stage 5 upsample, stage 8 FlexiDualGrid shape decoder, stage 9 tex decoder), host-driven buckets
# ----------------------------------------------------------------------------------------------------------------------
class SparseUnetDecoderJax:
    def __init__(self, params: Dict, model_channels: List[int], num_blocks: List[int], pred_subdiv: bool, mode='accumulate',
                 quantum: int = 256):
        self.p, self.mc, self.nb, self.pred_subdiv, self.mode, self.q = params, model_channels, num_blocks, pred_subdiv, mode, quantum

    def run(self, feats: np.ndarray, coords: np.ndarray, grid: int, guide_subs: Optional[List[np.ndarray]] = None,
            upsample_times: Optional[int] = None, record: Optional[list] = None, teacher: Optional[dict] = None,
            force_subs: Optional[List[np.ndarray]] = None):
        """feats [N, latent], coords [N, 3] (x, y, z) at `grid`.

        Returns dict(h=[M, out] pre-head output or None, coords=[M, 3], subs=[(coords, logits)] per C2S, counts).
        guide_subs: list of bool masks [N_i, 8] (tex decoder, pred_subdiv=False).
        force_subs: optional list of bool masks [N_i, 8] (shape decoder, pred_subdiv=True): our logits are still
            computed and returned in subs, but the subdivision follows these masks (teacher forcing within stage 8,
            an opt-in check). Default None = unchanged behaviour.
        upsample_times: stop at that level and return coords (the official .upsample()).
        record: list to append (name, valid feats) per block output (for per-block parity).
        teacher: optional dict name -> (feats_in numpy) to teacher-force each block's input."""
        n = feats.shape[0]
        cap = bucket(n, self.q)
        valid = np.zeros(cap, bool); valid[:n] = True
        c = np.zeros((cap, 3), np.int32); c[:n] = coords
        x = np.zeros((cap, feats.shape[1]), np.float32); x[:n] = feats
        valid, c, h = jnp.asarray(valid), jnp.asarray(c), jnp.asarray(x)
        h = jax.jit(linear)(h, self.p['from_latent.weight'], self.p['from_latent.bias'])
        subs, counts = [], [n]
        nbr = _nbr_jit(c, valid, grid=grid)
        for i in range(len(self.nb)):
            if upsample_times is not None and i == upsample_times:
                return {'h': None, 'coords': np.asarray(c)[:counts[-1]], 'subs': subs, 'counts': counts, 'grid': grid}
            nblk = self.nb[i] + (1 if i < len(self.nb) - 1 else 0)
            for j in range(nblk):
                name = f'blocks.{i}.{j}'
                if teacher is not None and name in teacher:
                    t = np.zeros(h.shape, np.float32); t[:counts[-1]] = teacher[name]; h = jnp.asarray(t)
                bp = block_params(self.p, name)
                if j < self.nb[i]:
                    h = _convnext_jit(bp, h, nbr, mode=self.mode)
                else:
                    hh, logit = _c2s_pre_jit(bp, h, nbr, pred_subdiv=self.pred_subdiv, mode=self.mode)
                    m = counts[-1]
                    if self.pred_subdiv:
                        mask = logit > 0
                        subs.append((np.asarray(c)[:m], np.asarray(logit)[:m]))
                        if force_subs is not None:
                            g = np.zeros((h.shape[0], 8), bool); g[:m] = force_subs[len(subs) - 1]; mask = jnp.asarray(g)
                    else:
                        g = np.zeros((h.shape[0], 8), bool); g[:m] = guide_subs[i]; mask = jnp.asarray(g)
                    cnt = int(jnp.sum(mask & valid[:, None]))
                    ncap = bucket(cnt, self.q)
                    f, c, valid, _ = _c2s_index_jit(c, valid, mask, cap=ncap)
                    grid = grid * 2
                    nbr = _nbr_jit(c, valid, grid=grid)
                    h = _c2s_post_jit(bp, hh, h, f, nbr, out_channels=self.mc[i + 1], mode=self.mode)
                    counts.append(cnt)
                if record is not None:
                    record.append((name, np.asarray(h)[:counts[-1]], np.asarray(c)[:counts[-1]]))
        m = counts[-1]
        h = layer_norm(h, None, None, 1e-5)
        h = jax.jit(linear)(h, self.p['output_layer.weight'], self.p['output_layer.bias'])
        return {'h': np.asarray(h)[:m], 'coords': np.asarray(c)[:m], 'subs': subs, 'counts': counts, 'grid': grid}


def fdg_heads(h: np.ndarray, voxel_margin: float = 0.5):
    """FlexiDualGridVaeDecoder eval heads (inputs to o_voxel flexible_dual_grid_to_mesh)."""
    h = jnp.asarray(h)
    vertices = (1 + 2 * voxel_margin) * jax.nn.sigmoid(h[:, 0:3]) - voxel_margin
    intersected = h[:, 3:6] > 0
    quad_lerp = jax.nn.softplus(h[:, 6:7])
    return np.asarray(vertices), np.asarray(intersected), np.asarray(quad_lerp)


def cascade_quantize(hr_coords: np.ndarray, lr_resolution: int, hr_resolution: int) -> np.ndarray:
    """Pipeline cascade step after .upsample(slat, 4): ((c + 0.5) / lr * (hr // 16)).int(), then unique rows (torch unique
    dim=0 sorts lexicographically). hr_coords [M, 3] int (x, y, z). Host numpy, float32 like the reference."""
    q = ((hr_coords.astype(np.float32) + np.float32(0.5)) / np.float32(lr_resolution) * np.float32(hr_resolution // 16))
    q = q.astype(np.int32)  # trunc toward zero; values are positive
    return np.unique(q, axis=0)
