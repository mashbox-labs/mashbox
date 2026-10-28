"""Engine.barrier() returns only after every part of the window has been handed to the callback (the --out render reads
them right after it). Part 0's copy to the host is slow on purpose."""
import time

import numpy as np

from mashbox.models.hy_nomas.server import Engine, StubEngine


class _RT:
    steps, preset = 3, 'quality'

    def cam_tokens(self, blk):
        return None

    def tables(self, w):
        return w

    def window(self, cam, tabs, it):
        return np.zeros((2, 30, 1, 1), np.float32), dict(text=1, action_mask=False)


class _Frames:
    def __init__(self, ci, n):
        self.ci, self.n = ci, n

    def block_until_ready(self):
        return self

    def __array__(self, dtype=None, copy=None):
        if self.ci == 0:
            time.sleep(0.3)
        return np.full((self.n, 1, 1, 3), self.ci, np.uint8)


class _Decoder:
    def call(self, ci, lat, start):
        return _Frames(ci, 4 * lat.shape[2])


class _Model:
    def __init__(self):
        self.dit, self.decoder, self.text = _RT(), _Decoder(), None
        self.split, self.overlap = (10, 10, 10), 1


def test_engine_parts_handed_over_before_barrier_returns():
    eng = Engine(_Model(), log=lambda s: None)
    got = {}
    for w in range(3):
        eng.window(0, w, None, None, lambda sess, wi, j, fr, info: got.__setitem__((wi, j), fr))
        eng.barrier(0, w)
        assert sorted(k for k in got if k[0] == w) == [(w, 0), (w, 1), (w, 2)]


def test_stub_parts_handed_over_before_barrier_returns():
    eng = StubEngine(lambda *a: None, dit_s=0.0, dec_s=0.05, encode_s=0.0)
    got = {}
    for w in range(2):
        eng.window(0, w, None, dict(key='seg1', intra=False), lambda sess, wi, j, fr, info: got.__setitem__((wi, j), fr))
        eng.barrier(0, w)
        assert sorted(k for k in got if k[0] == w) == [(w, j) for j in range(eng.nch)]
