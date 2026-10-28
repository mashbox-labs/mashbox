"""MASHBOX Fast Decoder: the parts the app decodes in parallel give the same frames as one stream from the first latent."""
import threading

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

from mashbox.models.hy_nomas import fast_decoder as FD  # noqa: E402


def tiny_params(seed=0, widths=(16, 12, 8), blocks=(1, 1, 1)):
    r = np.random.default_rng(seed)

    def w(*shape):
        return (r.standard_normal(shape) / np.sqrt(np.prod(shape[:-1]))).astype(np.float32)
    flat = {'stem': w(3, 3, 32, widths[0]), 'temporal.0': w(3, widths[0]), 'temporal.1': w(3, widths[0]), 'head': w(1, 1, widths[-1], 12)}
    for i, c in enumerate(widths):
        if i:
            flat['stages.%d.project' % i] = w(1, 1, widths[i - 1], c)
        for b in range(blocks[i]):
            flat['stages.%d.res.%d.expand' % (i, b)] = w(1, 1, c, 2 * c)
            flat['stages.%d.res.%d.depthwise' % (i, b)] = w(3, 3, 1, 2 * c)
            flat['stages.%d.res.%d.contract' % (i, b)] = w(1, 1, 2 * c, c)
    return flat


def test_tree_and_warm_context():
    p = jax.tree_util.tree_map(jnp.asarray, FD.tree(tiny_params()))
    fn = FD.make_decode(jnp.float32)
    lat = np.random.default_rng(1).standard_normal((1, 32, 14, 2, 3)).astype(np.float32)
    full = np.asarray(fn(p, lat, 0))
    assert full.shape == (56, 8, 12, 3) and full.dtype == np.uint8
    assert np.array_equal(np.asarray(fn(p, lat, FD.WARM)), full[4 * FD.WARM:])          # 6 latents of context = exact
    assert not np.array_equal(np.asarray(fn(p, lat[:, :, 1:], FD.WARM - 1)), full[4 * FD.WARM:])  # 5 are not enough


class _RT:
    steps, preset = 3, 'quality'

    def __init__(self, lats):
        self.lats = lats

    def cam_tokens(self, blk):
        return None

    def tables(self, w):
        return w

    def window(self, cam, tabs, it):
        return self.lats[tabs], dict(text=1, action_mask=False)


class _Fast(FD.FastDecoder):
    def __init__(self, flat, n):
        self.devices = [jax.devices('cpu')[0]] * n
        p = jax.tree_util.tree_map(jnp.asarray, FD.tree(flat))
        self.P = [p] * n
        self.fn = FD.make_decode(jnp.float32)
        self.lock = threading.Lock()


class _Model:
    def __init__(self, lats, flat):
        self.dit, self.decoder, self.text = _RT(lats), None, None
        self.split, self.overlap = (10, 10, 10), 1
        self.fast_decoder, self.decoder_kind = _Fast(flat, 3), 'fast'


def test_engine_parts_match_one_stream():
    from mashbox.models.hy_nomas.server import Engine
    flat = tiny_params(2)
    rng = np.random.default_rng(3)
    lats = [rng.standard_normal((32, 30, 2, 3)).astype(np.float32) for _ in range(3)]
    eng = Engine(_Model(lats, flat), log=lambda s: None)
    assert eng.decoders == ['full', 'fast'] and eng.decoder == 'fast'
    got = {}
    for w in range(3):
        eng.window(0, w, None, None, lambda sess, wi, j, fr, info: got.__setitem__((wi, j), fr))
        eng.barrier(0, w)
    p = jax.tree_util.tree_map(jnp.asarray, FD.tree(flat))
    stream = np.asarray(FD.make_decode(jnp.float32)(p, np.concatenate(lats, axis=1)[None], 0))
    for w in range(3):
        for j, (s, e) in enumerate(eng.split):
            fr = got[(w, j)]
            lead = 0 if (w == 0 and j == 0) else 4                 # the overlap latent's frames come first, as with the full decoder
            assert fr.shape[0] == 4 * (e - s) + lead
            g0 = 4 * (30 * w + s)
            assert np.array_equal(fr[lead:], stream[g0:g0 + 4 * (e - s)])
            if lead:
                assert np.array_equal(fr[:lead], stream[g0 - 4:g0])
