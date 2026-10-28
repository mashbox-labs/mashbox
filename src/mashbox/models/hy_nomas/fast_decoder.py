"""MASHBOX Fast Decoder for Hy-Nomas: our own small decoder, trained by us, for when
speed matters more than fine detail. latents [1, 32, t, 22, 40] -> frames [4t, 704, 1280, 3] uint8, same as decoder.py.

The weights (about 42 MB) come from our Hugging Face repo at the pinned revision (upstream.json, repo 'mashbox').
The decoder is causal: each latent gives 4 frames and sees only itself and the latents before it. Every output frame
depends on its own latent and the 6 latents before it, nothing older, so a part of a window decoded with those 6 latents
in front of it, from a zero state, gives the same frames as one stream from the first latent. That keeps the call shape
of the full decoder: the three parts of a window decode on the three decoder chips at once.
"""
import json
import struct
import threading
import time

import numpy as np

WARM = 6                                                    # latents of context in front of each part


def read_safetensors(path):
    """{name: float32 numpy} from a safetensors file holding F32 tensors only."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 0 < n < 16 * 1024 * 1024:
            raise ValueError('invalid safetensors header length')
        head = json.loads(f.read(n))
        head.pop('__metadata__', None)
        blob = f.read()
    out = {}
    for k, spec in head.items():
        if spec['dtype'] != 'F32':
            raise ValueError('expected F32 tensors: ' + k)
        s, e = spec['data_offsets']
        a = np.frombuffer(blob[s:e], np.float32)
        if a.size != int(np.prod(spec['shape'])):
            raise ValueError('bad tensor extent: ' + k)
        out[k] = a.reshape(spec['shape'])
    return out


def tree(flat):
    """Flat names (stem, temporal.0, stages.1.project, stages.1.res.0.expand, head) -> parameter tree."""
    n_st = 1 + max(int(k.split('.')[1]) for k in flat if k.startswith('stages.'))
    p = {'stem': flat['stem'], 'head': flat['head'], 'temporal': [flat['temporal.0'], flat['temporal.1']], 'stages': []}
    for i in range(n_st):
        st = {'res': []}
        if i:
            st['project'] = flat['stages.%d.project' % i]
        b = 0
        while 'stages.%d.res.%d.expand' % (i, b) in flat:
            st['res'].append({k: flat['stages.%d.res.%d.%s' % (i, b, k)] for k in ('expand', 'depthwise', 'contract')})
            b += 1
        p['stages'].append(st)
    return p


def _conv(x, w, groups=1):
    import jax
    return jax.lax.conv_general_dilated(x[None], w, (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                        feature_group_count=groups)[0]


def initial_state(p, h, w, dtype):
    import jax.numpy as jnp
    c = p['stem'].shape[-1]
    return (jnp.zeros((2, h, w, c), dtype), jnp.zeros((4, h, w, c), dtype))


def step(p, state, z):
    """One latent [h, w, 32] -> (state, 4 frames [4, 32h, 32w, 3] in [-1, 1])."""
    import jax
    import jax.numpy as jnp
    x = _conv(z, p['stem'])
    new = []
    for w, hist, dil in zip(p['temporal'], state, (1, 2)):
        seq = jnp.concatenate((hist, x[None]), axis=0)      # the block's inputs are kept, never its outputs
        y = sum(seq[-1 - i * dil] * w[2 - i] for i in range(3))
        new.append(seq[1:])
        x = x + .1 * jax.nn.silu(y)
    for i, st in enumerate(p['stages']):
        if i:
            x = _conv(x, st['project'])
            x = jnp.repeat(jnp.repeat(x, 2, axis=0), 2, axis=1)
        for b in st['res']:
            y = jax.nn.silu(_conv(x, b['expand']))
            y = jax.nn.silu(_conv(y, b['depthwise'], groups=y.shape[-1]))
            x = x + .1 * _conv(y, b['contract'])
    y = _conv(x, p['head']).reshape(x.shape[0], x.shape[1], 4, 3).transpose(2, 0, 1, 3)
    return tuple(new), jnp.tanh(y)


def make_decode(dtype):
    """-> fn(p, latents [1, 32, t, h, w], warm static) = uint8 frames [4 (t - warm), 32h, 32w, 3]."""
    import jax
    import jax.numpy as jnp

    def fn(p, lat, warm):
        z = jnp.transpose(lat[0], (1, 2, 3, 0)).astype(dtype)
        st = initial_state(p, z.shape[1], z.shape[2], dtype)
        _, fr = jax.lax.scan(lambda s, x: step(p, s, x), st, z)
        fr = fr[warm:].reshape(-1, fr.shape[2], fr.shape[3], 3)
        return jnp.clip(jnp.round((fr.astype(jnp.float32) + 1) * 127.5), 0, 255).astype(jnp.uint8)
    return jax.jit(fn, static_argnums=2)


class FastDecoder:
    """One copy per decoder chip; call(ci, latents, start, warm) decodes on device ci and drops the first warm latents."""
    warm = WARM
    name = 'fast'

    def __init__(self, path, devices, dtype='bfloat16', log=print):
        import jax
        import jax.numpy as jnp
        t0 = time.time()
        self.devices, self.log = list(devices), log
        self.dtype = getattr(jnp, dtype)
        p = tree(read_safetensors(path))
        self.P = [jax.device_put(jax.tree_util.tree_map(lambda x: np.asarray(x).astype(self.dtype), p), d) for d in self.devices]
        jax.block_until_ready(self.P)
        self.fn = make_decode(self.dtype)
        self.lock = threading.Lock()
        self.mb = round(sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(p)) * 4 / 1e6, 1)
        log(json.dumps(dict(event='fast_decoder_loaded', chips=len(self.devices), mb_f32=self.mb, load_s=round(time.time() - t0, 1))))

    def call(self, ci, latents, start, warm=0):
        import jax
        d = self.devices[ci]
        return self.fn(self.P[ci], jax.device_put(np.asarray(latents, np.float32), d), int(warm))
