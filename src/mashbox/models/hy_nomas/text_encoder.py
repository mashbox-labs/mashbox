"""Hy-Nomas text encoder in JAX: Qwen2.5-VL-7B-Instruct's language model, text path only, as Tencent's TextEncoder runs it.

  * tokens: Tencent's own TextEncoder.text2tokens (video JSON prompt template, crop from the user marker, 1280 tokens
    after the crop, right padding), see upstream.TencentText;
  * output: the hidden state after layer 26 of 28 (hidden_states[-3]), no final norm, rows [crop, crop + 1280),
    as float32 [1280, 3584] plus the attention mask;
  * arithmetic: bf16 like the bf16 torch model, every rounding point kept, matmuls in bf16 with f32 accumulation.

The 26 layers are split over the chips in groups (7, 7, 6, 6 on four chips, about 3 GB each) so the encoder fits beside
the decoder; the hidden state moves chip to chip between groups.
"""
import json
import struct
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

LAYERS_USED = 26
TEXT_LEN = 1280


class Arch:
    def __init__(self, hidden=3584, heads=28, kv_heads=4, ff=18944, head_dim=128, theta=1e6, eps=1e-6):
        self.hidden, self.heads, self.kv_heads, self.ff, self.head_dim, self.theta, self.eps = hidden, heads, kv_heads, ff, head_dim, theta, eps


QWEN = Arch()


def check_config(encoder_dir):
    c = json.loads((Path(encoder_dir) / 'config.json').read_text())
    t = c.get('text_config', c)
    g = lambda k: t.get(k, c.get(k))
    got = (g('hidden_size'), g('num_attention_heads'), g('num_key_value_heads'), g('intermediate_size'), g('num_hidden_layers'),
           float(g('rope_theta')), float(g('rms_norm_eps')))
    if got != (3584, 28, 4, 18944, 28, 1e6, 1e-6):
        raise RuntimeError('unexpected text encoder config %s' % (got,))


class Shards:
    """Safetensors reader over the sharded checkpoint (numpy memmap, bf16 kept as bf16, no torch)."""

    def __init__(self, encoder_dir):
        self.dir = Path(encoder_dir)
        self.index = json.loads((self.dir / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def get(self, key):
        import ml_dtypes
        fn = self.index[key]
        if fn not in self.headers:
            with (self.dir / fn).open('rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[fn] = (n + 8, json.loads(f.read(n)))
        start, hdr = self.headers[fn]
        info = hdr[key]
        if info['dtype'] != 'BF16':
            raise RuntimeError('%s: expected BF16, got %s' % (key, info['dtype']))
        return np.memmap(self.dir / fn, mode='r', dtype=np.uint16, offset=start + info['data_offsets'][0],
                         shape=tuple(info['shape'])).view(ml_dtypes.bfloat16)


def layer_getters(rd):
    L = lambda i, n: rd.get('model.layers.%d.%s' % (i, n))
    return {
        'ln1': lambda i: L(i, 'input_layernorm.weight'),
        'wqkv': lambda i: np.concatenate([L(i, 'self_attn.q_proj.weight'), L(i, 'self_attn.k_proj.weight'), L(i, 'self_attn.v_proj.weight')], 0),
        'bqkv': lambda i: np.concatenate([L(i, 'self_attn.q_proj.bias'), L(i, 'self_attn.k_proj.bias'), L(i, 'self_attn.v_proj.bias')], 0),
        'wo': lambda i: L(i, 'self_attn.o_proj.weight'),
        'ln2': lambda i: L(i, 'post_attention_layernorm.weight'),
        'wgu': lambda i: np.concatenate([L(i, 'mlp.gate_proj.weight'), L(i, 'mlp.up_proj.weight')], 0),
        'wd': lambda i: L(i, 'mlp.down_proj.weight'),
    }


# ------------------------------------------------------------------ math (bf16 rounding points of the torch model)
def _r(x, dt):
    return x.astype(dt)


def _rms(x, w, dt, eps):
    xf = x.astype(jnp.float32)
    n = _r(xf * lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + jnp.float32(eps)), dt)
    return _r(n.astype(jnp.float32) * w.astype(jnp.float32), dt)


def _lin(x, w, prec, b=None):
    y = lax.dot_general(x, w.astype(x.dtype), (((1,), (1,)), ((), ())), precision=prec, preferred_element_type=jnp.float32)
    if b is not None:
        y = y + b.astype(jnp.float32)
    return _r(y, x.dtype)


def rotary(length, dt, head_dim=128, theta=1e6):
    # the bf16 model also stores the rotary inverse frequencies in bf16
    inv = _r(jnp.float32(1) / jnp.power(jnp.float32(theta), jnp.arange(0, head_dim, 2, dtype=jnp.float32) / jnp.float32(head_dim)), dt).astype(jnp.float32)
    f = jnp.arange(length, dtype=jnp.float32)[:, None] * inv[None, :]
    ang = jnp.concatenate([f, f], axis=-1)
    return _r(jnp.cos(ang), dt)[:, None, :], _r(jnp.sin(ang), dt)[:, None, :]


def _rope(x, cos, sin):
    dt = x.dtype
    h = x.shape[-1] // 2
    rot = jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    a = _r(x.astype(jnp.float32) * cos.astype(jnp.float32), dt)
    b = _r(rot.astype(jnp.float32) * sin.astype(jnp.float32), dt)
    return _r(a.astype(jnp.float32) + b.astype(jnp.float32), dt)


def _attn(q, k, v, allowed, dt, prec, head_dim, tile=512):
    """Grouped-query causal attention, keys in tiles of 512 with a running max (the CPU torch kernel's order)."""
    logits = jnp.einsum('qgrd,kgd->grqk', q, k, precision=prec, preferred_element_type=jnp.float32) * jnp.float32(head_dim ** -0.5)
    logits = jnp.where(allowed[None, None], logits, -jnp.inf)
    Lk = logits.shape[-1]
    m = jnp.full(logits.shape[:-1] + (1,), -jnp.inf, jnp.float32)
    tot = jnp.zeros_like(m)
    out = jnp.zeros(logits.shape[:-1] + (head_dim,), jnp.float32)
    for s in range(0, Lk, tile):
        blk = logits[..., s:s + tile]
        cm = jnp.maximum(m, jnp.max(blk, axis=-1, keepdims=True))
        has = jnp.isfinite(cm)
        sm = jnp.where(has, cm, jnp.float32(0))
        e = jnp.exp(blk - sm)
        resc = jnp.where(has, jnp.exp(m - sm), jnp.float32(0))
        tot = jnp.sum(e, axis=-1, keepdims=True) + resc * tot
        pv = jnp.einsum('grqk,kgd->grqd', _r(e, dt), v[s:s + tile], precision=prec, preferred_element_type=jnp.float32)
        out = out * resc + pv
        m = cm
    return out * (jnp.float32(1) / jnp.where(tot == 0, jnp.float32(1), tot))


def layer(h, w, cos, sin, allowed, dt, prec, a=QWEN):
    L = h.shape[0]
    H, G, D = a.heads, a.kv_heads, a.head_dim
    x = _rms(h, w['ln1'], dt, a.eps)
    qkv = _lin(x, w['wqkv'], prec, w['bqkv'])
    q = qkv[:, :H * D].reshape(L, H, D)
    k = qkv[:, H * D:(H + G) * D].reshape(L, G, D)
    v = qkv[:, (H + G) * D:].reshape(L, G, D)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = _attn(q.reshape(L, G, H // G, D), k, v, allowed, dt, prec, D)
    o = _r(o, dt).transpose(2, 0, 1, 3).reshape(L, H * D)
    h = _r(h.astype(jnp.float32) + _lin(o, w['wo'], prec).astype(jnp.float32), dt)
    x = _rms(h, w['ln2'], dt, a.eps)
    gu = _lin(x, w['wgu'], prec)
    g = _r(jax.nn.silu(gu[:, :a.ff].astype(jnp.float32)), dt)
    p = _r(g.astype(jnp.float32) * gu[:, a.ff:].astype(jnp.float32), dt)
    return _r(h.astype(jnp.float32) + _lin(p, w['wd'], prec).astype(jnp.float32), dt)


def _part(W, h, mask, mode, a):
    dt = jnp.bfloat16 if mode == 'bf16' else jnp.float32
    prec = lax.Precision.DEFAULT if mode == 'bf16' else lax.Precision.HIGHEST
    L = h.shape[0]
    cos, sin = rotary(L, dt, a.head_dim, a.theta)
    pos = jnp.arange(L)
    allowed = (pos[:, None] >= pos[None, :]) & mask[None, :]
    h, _ = lax.scan(lambda h, w: (layer(h, w, cos, sin, allowed, dt, prec, a), None), h.astype(dt), W)
    return h


part = jax.jit(_part, static_argnames=('mode', 'a'))


def split(layers, n):
    """Layer groups over n chips, larger groups first (26 over 4 -> 7, 7, 6, 6)."""
    base, extra = divmod(layers, n)
    return tuple(base + (1 if i < extra else 0) for i in range(n))


class TextEncoder:
    """enc = TextEncoder(text, encoder_dir, devices); emb, mask, info, timing = enc.encode(prompt)

    ``text`` is an upstream.TencentText (tokenizer, template, crop, action offsets)."""

    def __init__(self, text, encoder_dir, devices, mode='bf16', layers=None, arch=QWEN, warmup=True, log=print):
        self.tt, self.mode, self.arch, self.log = text, mode, arch, log
        self.devices = list(devices)
        t0 = time.time()
        if layers is None:
            check_config(encoder_dir)
            rd = Shards(encoder_dir)
            get, self.emb = layer_getters(rd), rd.get('model.embed_tokens.weight')
            n_layers = LAYERS_USED
        else:                                   # tests: (getters, embedding, layer count)
            get, self.emb, n_layers = layers
        groups = split(n_layers, len(self.devices))
        self.ranges = [(sum(groups[:i]), sum(groups[:i + 1])) for i in range(len(groups))]
        self.W = [dict() for _ in self.devices]
        nb = 0
        for k, fn in get.items():
            first = fn(0)
            arr = np.empty((n_layers,) + first.shape, first.dtype)
            arr[0] = first
            for i in range(1, n_layers):
                arr[i] = fn(i)
            for g, (a, b) in enumerate(self.ranges):
                self.W[g][k] = jax.device_put(arr[a:b], self.devices[g])
            jax.block_until_ready([w[k] for w in self.W])
            nb += arr.nbytes
            del arr
        self.weight_gb = round(nb / 1e9, 2)
        self.load_s = round(time.time() - t0, 1)
        self.compile_s = None
        log(json.dumps(dict(event='text_encoder_loaded', gb=self.weight_gb, chips=len(self.devices), layers=self.ranges, load_s=self.load_s)))
        if warmup:
            t1 = time.time()
            self.encode('{"segments": [{"description_change": "warmup", "contained_actions": []}]}')
            self.compile_s = round(time.time() - t1, 1)

    def states(self, ids, am):
        h = jax.device_put(np.asarray(self.emb[ids]), self.devices[0])
        for g, dv in enumerate(self.devices):
            if g:
                h = jax.device_put(h, dv)
            h = part(self.W[g], h, jax.device_put(np.asarray(am), dv), mode=self.mode, a=self.arch)
        c = self.tt.crop
        return np.asarray(h)[c:c + TEXT_LEN].astype(np.float32)

    def encode(self, prompt):
        t0 = time.time()
        ids, am = self.tt.tokens(prompt)
        text = self.states(ids, am)
        mask = am[self.tt.crop:]
        info = self.tt.offsets(prompt, int(am.sum()))
        if text.shape != (TEXT_LEN, self.arch.hidden) or not np.isfinite(text).all():
            raise RuntimeError('text encoder output %s' % (text.shape,))
        if int(mask.sum()) != info['valid_post_crop']:
            raise RuntimeError('valid tokens %d vs %d' % (int(mask.sum()), info['valid_post_crop']))
        return text, mask, info, dict(s=round(time.time() - t0, 3))
