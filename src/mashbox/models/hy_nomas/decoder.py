"""Hy-Nomas pixel decoder in JAX: latents [1, 32, t, 22, 40] -> video frames [4t, 704, 1280, 3] uint8, one pass.

This is the one-step pixel decoder of Tencent's Hy-Nomas release (VAE/PixelUMM_FT): 36 transformer layers over
16 x 16 x 4 pixel patches, attention local to 3 x 3 neighbourhoods of 8 x 8 tiles, conditioned on the latents, starting
from noise. Weights are bf16 and one full copy sits on each decoder chip; calls on different chips run in parallel.
The noise is made on the chip from a fixed seed folded with the absolute latent index, so a given latent always gets
the same noise.

Windows are decoded in chunks of latent frames (10, 10, 10 by default). With overlap 1, each chunk also decodes the
latent before it; the player crossfades the 4 frames that overlap (see server.py), which removes the seam at chunk edges.
"""
import json
import math
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

BF = jnp.bfloat16
TILE = 8
HI = jax.lax.Precision.HIGHEST


@dataclass(frozen=True)
class Arch:
    hidden: int = 4096
    intermediate: int = 12288
    layers: int = 36
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    eps: float = 1e-6
    theta_t: float = 1_000_000.0
    theta_hw: float = 10_000.0

    @property
    def rope_sections(self):
        return (self.head_dim // 2, self.head_dim // 4, self.head_dim // 4)


ARCH = Arch()


# ------------------------------------------------------------------ weights
class Weights:
    """Reads the decoder tensors ('decoder.' prefix) of the safetensors file, one tensor at a time, as bf16."""

    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            if not 0 < n < 64 * 1024 * 1024:
                raise ValueError('invalid safetensors header length')
            self.header = json.loads(f.read(n))
        self.header.pop('__metadata__', None)
        self.start = 8 + n

    def names(self, prefix):
        start = 'decoder.' + prefix + '.'
        return start, sorted(k for k in self.header if k.startswith(start))

    def read(self, name):
        """One tensor as bf16 numpy (F32 and F16 are rounded to nearest even, as torch's .to(bfloat16))."""
        import ml_dtypes
        spec = self.header[name]
        s, e = spec['data_offsets']
        count = math.prod(spec['shape'])
        stored = {'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}[spec['dtype']]
        if e - s != count * np.dtype(stored).itemsize:
            raise ValueError('bad tensor extent: ' + name)
        with open(self.path, 'rb') as f:
            f.seek(self.start + s)
            a = np.fromfile(f, dtype=stored, count=count)
        if a.size != count:
            raise ValueError('truncated tensor: ' + name)
        a = a.view(ml_dtypes.bfloat16) if spec['dtype'] == 'BF16' else a.astype(ml_dtypes.bfloat16)
        return a.reshape(spec['shape'])

    def group(self, prefix):
        start, names = self.names(prefix)
        if not names:
            raise KeyError(prefix)
        return {n[len(start):]: self.read(n) for n in names}

    def load(self, arch=ARCH):
        """-> (top {embed, cond_proj, final_norm, head}, stacked layer weights [layers, ...]) as bf16 numpy."""
        top = {p: self.group(p) for p in ('embed', 'cond_proj', 'final_norm', 'head')}
        stacked = {}
        for i in range(arch.layers):
            for k, v in self.group('layers.%d' % i).items():
                if i == 0:
                    stacked[k] = np.empty((arch.layers,) + v.shape, v.dtype)
                stacked[k][i] = v
        return top, stacked


# ------------------------------------------------------------------ math
def patchify(video):
    b, _, t, h, w = video.shape
    return video.reshape(b, 3, t // 4, 4, h // 16, 16, w // 16, 16).transpose(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, -1, 3072)


def unpatchify(tokens, grid):
    b, n, dim = tokens.shape
    t, h, w = grid
    return tokens.reshape(b, t, h, w, 4, 16, 16, 3).transpose(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, 3, t * 4, h * 16, w * 16)


def linear(x, weight, bias=None):
    out = jnp.matmul(x, weight.T, precision=HI)
    return out if bias is None else out + bias


def rms_norm(x, weight, eps):
    xf = x.astype(jnp.float32)
    return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype) * weight


def mrope_tables(arch, grid):
    pos = jnp.meshgrid(*(jnp.arange(n) for n in grid), indexing='ij')
    cos, sin = [], []
    for p, dim, theta in zip(pos, arch.rope_sections, (arch.theta_t, arch.theta_hw, arch.theta_hw)):
        inv = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        ang = p.reshape(-1).astype(jnp.float32)[:, None] * inv[None]
        ang = jnp.concatenate((ang, ang), axis=-1)
        cos.append(jnp.cos(ang))
        sin.append(jnp.sin(ang))
    return jnp.concatenate(cos, axis=-1)[None, :, None, :], jnp.concatenate(sin, axis=-1)[None, :, None, :]


def condition_tokens(latents, weight, bias, arch):
    b, _, t, h, w = latents.shape
    x = linear(latents.transpose(0, 2, 3, 4, 1), weight, bias)
    return x.reshape(b, t, h, w, 2, 2, arch.hidden).transpose(0, 1, 2, 4, 3, 5, 6).reshape(b, -1, arch.hidden)


def local_layout(grid, tile):
    """Token order (tile-major) and, per tile, the keys of its 3 x 3 tile neighbourhood (edges use the nearest full one)."""
    t, h, w = grid
    nh, nw = (h + tile - 1) // tile, (w + tile - 1) // tile
    if nh <= 3 and nw <= 3:
        return None
    tt, hh, ww = np.meshgrid(np.arange(t), np.arange(h), np.arange(w), indexing='ij')
    th, tw = (hh // tile).ravel(), (ww // tile).ravel()
    order = (((th * nw + tw) * t + tt.ravel()) * tile + (hh % tile).ravel()) * tile + (ww % tile).ravel()
    perm = np.argsort(order)
    th, tw = th[perm], tw[perm]
    nb = []
    for row in range(nh):
        for col in range(nw):
            q = np.flatnonzero((th == row) & (tw == col))
            ch, cw = min(max(row, 1), max(nh - 2, 1)), min(max(col, 1), max(nw - 2, 1))
            k = np.flatnonzero((np.abs(th - ch) <= 1) & (np.abs(tw - cw) <= 1))
            nb.append((q, k))
    return perm, np.argsort(perm), nb


def layout(grid, arch=ARCH):
    n = grid[0] * grid[1] * grid[2]
    lay = local_layout(grid, TILE)
    if lay is None:
        perm = inv = np.arange(n)
        nb = [(np.arange(n), np.arange(n))]
    else:
        perm, inv, nb = lay
    qm, km, g = max(len(q) for q, _ in nb), max(len(k) for _, k in nb), len(nb)
    qi = np.full((g, qm), n, np.int32)
    ki = np.zeros((g, km), np.int32)
    kv = np.zeros((g, km), bool)
    for j, (q, k) in enumerate(nb):
        qi[j, :len(q)] = q
        ki[j, :len(k)] = k
        kv[j, :len(k)] = True
    cos, sin = mrope_tables(arch, grid)
    return dict(perm=np.asarray(perm, np.int32), inv=np.asarray(inv, np.int32), qi=qi, qg=np.minimum(qi, n - 1), ki=ki, kv=kv,
                cos=np.asarray(cos)[:, perm], sin=np.asarray(sin)[:, perm], n=n)


def make_decode(grid, ops, arch=ARCH):
    """-> jitted fwd(top, stacked, latents bf16 [1, 32, t, h, w], start int32, key, lay) = frames uint8 [4t, 32h, 32w, 3]."""
    t = grid[0]

    def layer(x, w, lay):
        h = rms_norm(x, w['input_norm.weight'], arch.eps)
        b, n, _ = h.shape
        v = linear(h, w['attn.v_proj.weight']).reshape(b, n, arch.kv_heads, arch.head_dim)
        q = jnp.matmul(h, w['attn.q_proj.weight'].T, precision=HI, preferred_element_type=jnp.float32)
        k = jnp.matmul(h, w['attn.k_proj.weight'].T, precision=HI, preferred_element_type=jnp.float32)
        c2, s2 = lay['cos'].reshape(n, arch.head_dim), lay['sin'].reshape(n, arch.head_dim)
        q = ops.dec_qk_norm_rope(q.reshape(n, -1), w['attn.q_norm_t.weight'], w['attn.q_norm_hw.weight'], c2, s2, H=arch.heads,
                                 which='q', frames=t).reshape(n, arch.heads, arch.head_dim)
        k = ops.dec_qk_norm_rope(k.reshape(n, -1), w['attn.k_norm_t.weight'], w['attn.k_norm_hw.weight'], c2, s2, H=arch.kv_heads,
                                 which='k', frames=t).reshape(n, arch.kv_heads, arch.head_dim)
        h = ops.dec_local_attention(q, k, v[0], lay, frames=t, heads=arch.heads, kv_heads=arch.kv_heads, head_dim=arch.head_dim)
        x = x + linear(h.astype(x.dtype), w['attn.o_proj.weight'])
        h = rms_norm(x, w['post_attn_norm.weight'], arch.eps)
        gate = linear(h, w['mlp.gate_proj.weight'])
        up = linear(h, w['mlp.up_proj.weight'])
        gf = gate.astype(jnp.float32)
        return x + linear((gf * jax.nn.sigmoid(gf)).astype(gate.dtype) * up, w['mlp.down_proj.weight'])

    def fwd(top, stacked, latents, start, key, lay):
        b, _, tt, h, w_ = latents.shape
        idx = start + jnp.arange(tt)
        st = jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(key, i), (1, 3, 4, h * 32, w_ * 32), jnp.float32))(idx)
        state = st.transpose(1, 2, 0, 3, 4, 5).reshape(1, 3, 4 * tt, h * 32, w_ * 32)
        dtype = top['embed']['weight'].dtype
        tokens = linear(patchify(state.astype(dtype)), top['embed']['weight'], top['embed']['bias'])
        tokens = tokens + condition_tokens(latents.astype(dtype), top['cond_proj']['weight'], top['cond_proj']['bias'], arch)
        tokens = tokens[:, lay['perm']]
        tokens, _ = jax.lax.scan(lambda x, w: (layer(x, w, lay), None), tokens, stacked)
        tokens = rms_norm(tokens, top['final_norm']['weight'], arch.eps)
        tokens = tokens[:, lay['inv']]
        pixels = linear(rms_norm(tokens, top['head']['norm.weight'], 1e-6), top['head']['linear.weight'], top['head']['linear.bias'])
        video = unpatchify(pixels, grid)
        vid = jnp.clip(video * jnp.asarray(0.5, video.dtype) + jnp.asarray(0.5, video.dtype), 0, 1)
        return jnp.clip(vid[0].transpose(1, 2, 3, 0) * jnp.asarray(255, video.dtype), 0, 255).astype(jnp.uint8)
    return jax.jit(fwd)


class Decoder:
    """One resident bf16 copy per device; call(ci, latents, start) decodes on device ci."""

    def __init__(self, weights, devices, ops, arch=ARCH, seed=42, log=print):
        """weights: (top, stacked) bf16 numpy from Weights.load (or the same structure built for tests)."""
        self.arch, self.ops, self.devices, self.log = arch, ops, list(devices), log
        t0 = time.time()
        top, stacked = weights
        self.W = [(jax.tree_util.tree_map(lambda x: jax.device_put(x, d), top), {k: jax.device_put(v, d) for k, v in stacked.items()})
                  for d in self.devices]
        jax.block_until_ready(self.W)
        self.load_s = round(time.time() - t0, 1)
        self.key = jax.random.PRNGKey(seed)
        self._fn, self._lay = {}, {}
        self.lock = threading.Lock()
        log(json.dumps(dict(event='decoder_loaded', chips=len(self.devices), load_s=self.load_s)))

    def program(self, ci, t, h, w):
        grid = (t, 2 * h, 2 * w)
        with self.lock:
            if grid not in self._lay:
                self._lay[grid] = layout(grid, self.arch)
            if (grid, ci) not in self._fn:
                d = self.devices[ci]
                lay = {k: jax.device_put(self._lay[grid][k], d) for k in ('perm', 'inv', 'qi', 'qg', 'ki', 'kv', 'cos', 'sin')}
                self._fn[(grid, ci)] = (make_decode(grid, self.ops, self.arch), lay)
            return self._fn[(grid, ci)]

    def call(self, ci, latents, start):
        """latents [1, 32, t, h, w] (any device or host), start = absolute latent index -> uint8 frames on device ci."""
        t, h, w = latents.shape[2:]
        fn, lay = self.program(ci, t, h, w)
        d = self.devices[ci]
        top, stacked = self.W[ci]
        lat = jax.device_put(latents, d).astype(BF)
        return fn(top, stacked, lat, jax.device_put(np.int32(start), d), jax.device_put(self.key, d), lay)
