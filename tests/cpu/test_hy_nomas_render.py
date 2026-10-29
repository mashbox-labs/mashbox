"""`mashbox run hy-nomas --out`: every part of every window reaches the clip.

The decoder threads count a part as done before its frames are copied to the host and handed on, so the window barrier can
return while the last part's frames are still on their way (on a TPU v6e-4 this stopped the documented
--out command at window 2 with KeyError (1, 0)). Here part 0's host copy is slow on purpose; the clip must still get every
frame in order.
"""
import argparse
import time

import numpy as np
import pytest

from mashbox.models.hy_nomas import app, cameras, server
from mashbox.models.hy_nomas.server import Engine

FRAMES, OV = 30, 1


class _Slow:
    """A decoded part: ready at once, but the copy to the host takes a while for part 0."""

    def __init__(self, arr, delay):
        self.arr, self.delay = arr, delay

    def block_until_ready(self):
        return self

    def __array__(self, dtype=None, copy=None):
        time.sleep(self.delay)
        return self.arr


class _Dec:
    def call(self, ci, lat, start):
        n = lat.shape[2]
        fr = np.zeros((4 * n, 2, 2, 3), np.uint8)
        fr[..., 0] = ci                                    # which part
        fr[..., 1] = np.arange(4 * n)[:, None, None] % 251  # frame number inside the part
        fr[..., 2] = (start + OV) // FRAMES                 # which window (a part after the first starts one overlap latent early)
        return _Slow(fr, 0.3 if ci == 0 else 0.0)


class _RT:
    steps, preset = 3, 'quality'

    def cam_tokens(self, blk):
        return None

    def tables(self, w):
        return None

    def window(self, cam, tabs, it):
        return np.zeros((2, FRAMES, 1, 1), np.float32), dict(text=it, action_mask=None)

    def make_item(self, key, text, mask, info):
        return key

    def set_preset(self, p):
        pass

    def reset(self, seed):
        pass


class _Text:
    def encode(self, prompt):
        return np.zeros((4, 2), np.float32), np.ones(4, bool), {}, {}


class _Model:
    def __init__(self):
        self.dit, self.decoder, self.text = _RT(), _Dec(), _Text()
        self.split, self.overlap = (10, 10, 10), OV
        self.tencent_cam = None


class _Cams:
    def __init__(self, *a, **k):
        pass

    def next_window(self, keys):
        return keys


class _FF:
    """Stands in for ffmpeg: keeps the frames written to it."""
    last = None

    def __init__(self):
        self.chunks = []
        self.stdin = self
        _FF.last = self

    def write(self, b):
        self.chunks.append(b)

    def close(self):
        pass

    def wait(self):
        return 0


def test_out_clip_gets_every_part(monkeypatch, tmp_path):
    monkeypatch.setattr(cameras, 'Cameras', _Cams)
    real = app.subprocess.Popen
    monkeypatch.setattr(server, 'ffmpeg_exe', lambda: 'fake-ffmpeg')
    monkeypatch.setattr(app.subprocess, 'Popen', lambda cmd, *x, **k: _FF() if cmd[0] == 'fake-ffmpeg' else real(cmd, *x, **k))
    model = _Model()
    eng = Engine(model, log=lambda s: None)
    a = argparse.Namespace(world='eagle', windows=3, keys='w;w,left;w', preset='quality', seed=2701, out=str(tmp_path / 'out.mp4'))
    assert app.render(eng, model, a) == 0
    fr = np.frombuffer(b''.join(_FF.last.chunks), np.uint8).reshape(-1, 2, 2, 3)
    assert fr.shape[0] == 4 * FRAMES * a.windows                # every latent's 4 frames once: overlaps are blended, not repeated
    # the parts in stream order (window by window, parts 0, 1, 2), leaving out the 4 blended frames at each seam
    first, per = 4 * 10 - 4, 4 * 10
    blended = {first + per * m + i for m in range(3 * a.windows - 1) for i in range(4)}
    seq = [(int(f[0, 0, 2]), int(f[0, 0, 0])) for k, f in enumerate(fr) if k not in blended]
    runs = [x for k, x in enumerate(seq) if k == 0 or x != seq[k - 1]]
    assert runs == [(w, j) for w in range(a.windows) for j in range(3)]


def test_frog_world_is_gone(capsys):
    """The presets are eagle (default), fish, mars and moon. `--world frog` is refused like any unknown id: the command
    line stops before any model loads, and the app's world call raises KeyError, which the server answers with 404."""
    assert list(server.worlds()) == ['eagle', 'fish', 'mars', 'moon']
    assert app.parser('run').parse_args([]).world == 'eagle'
    with pytest.raises(SystemExit) as e:
        app.main('run', ['--out', 'out.mp4', '--world', 'frog'])
    assert e.value.code == 2 and "invalid choice: 'frog'" in capsys.readouterr().err
    live = server.Live(server.StubEngine(lambda *a: None), lambda *a: None, hw='STUB', stub=True)
    with pytest.raises(KeyError):
        live.api('/api/world', {'mode': 'preset', 'id': 'frog'})


def test_crossfade_does_not_copy_the_part_and_blends_frame_by_frame():
    """The live pacer crossfades each part between two frames 1/24 s apart. Blending 4 frames in f32 and copying the
    whole 40-frame 720p part took 76-87 ms on a busy CPU build box (load 57), so the first frame after each part's held
    tail went out late (8 of 365 frames in a stub run slower than real time, at g = 36, 76, 156, ...). The part is now
    read in place: frames past the overlap are the decoder's own frames, overlap frames are blended when read, pixel for
    pixel the same as before; np.asarray gives the whole part (the --out writer)."""
    rng = np.random.default_rng(5)
    frames = rng.integers(0, 256, (40, 8, 16, 3), dtype=np.uint8)
    hold = rng.integers(0, 256, (4, 8, 16, 3), dtype=np.uint8)
    wv = ((np.arange(4) + 1) / 5).astype(np.float32)[:, None, None, None]
    bl = np.clip(np.round(hold.astype(np.float32) * (1 - wv) + frames[:4].astype(np.float32) * wv), 0, 255).astype(np.uint8)
    want = np.concatenate([bl, frames[4:]])[:-4]
    out, tail = server.crossfade(frames, hold, 4)
    assert out.shape == want.shape and len(out) == 36
    assert all(np.array_equal(out[f], want[f]) for f in range(36))
    assert np.array_equal(np.asarray(out), want) and np.array_equal(np.ascontiguousarray(out), want)
    assert all(np.shares_memory(out[f], frames) for f in range(4, 36)) and np.shares_memory(tail, frames)
    first, tail0 = server.crossfade(frames, None, 4)               # first part: nothing to blend
    assert np.shares_memory(first[0], frames) and np.array_equal(np.asarray(first), frames[:-4])
    assert server.crossfade(frames, None, 0)[0] is frames
