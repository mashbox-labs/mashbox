"""Camera path from held keys, normalized for the model with Tencent's own prepare_camera_inputs.

Keys: w/s move along the camera's forward axis, a/d along its right axis (2.8 units per second, Tencent's move speed);
left/right turn about the camera's own up axis and up/down about its own right axis (25 degrees per second by default,
times Tencent's rotation gain 2.5 in the model input). Turning about the camera's own axes keeps a turn after a climb a
turn; the summed pitch the model sees is capped (60 degrees by default, never past 85) so a climb never goes past vertical.

Each window is 5 s: 117 video frames for the first window, 120 after. The normalization runs on at most ``cap`` windows of
path (60 = 5 minutes) so its cost stays flat in long sessions.
"""
import copy
import math
import os
import tempfile

import numpy as np

KEYS = ('w', 's', 'a', 'd', 'up', 'down', 'left', 'right')
FPS = 24.0
MOVE = 2.8 / FPS                      # Tencent's move speed per frame


def _ry(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def block_angles(blk):
    """(yaw, pitch) in degrees across one block of camera-to-world matrices, in the block's first camera frame."""
    r = blk[0, :3, :3].T @ blk[-1, :3, :3]
    return float(np.degrees(np.arctan2(r[0, 2], r[2, 2]))), float(np.degrees(np.arcsin(-np.clip(r[1, 2], -1, 1))))


class Cameras:
    """next_window(keys) -> float32 [n, 4, 4] normalized extrinsics of the next window (n = 117 first, then 120).

    ``tencent`` = upstream.camera_module(src); None gives the raw relative path (tests only)."""

    def __init__(self, tencent=None, rate=25.0, pitch_max=60.0, gain=2.5, cap=60, move=1.0, tmpdir=None):
        self.rate, self.gain, self.cap, self.move = float(rate), float(gain), int(cap), float(move)
        self.pmax = min(85.0, float(pitch_max)) / self.gain
        self.R, self.p, self.pitch = np.eye(3), np.zeros(3), 0.0
        self.c2w = [np.eye(4)]
        self.inputs = []
        self.w = 0
        self.tmp = tmpdir or tempfile.mkdtemp(prefix='hy-nomas-cam-')
        self.tencent = tencent
        self.intr = None
        if tencent is not None:
            to_extr, Config, self.prepare = tencent
            cfg = Config()
            if abs(cfg.move_speed - MOVE) > 1e-9:
                raise RuntimeError('unexpected camera move speed %s' % cfg.move_speed)
            _, intr = to_extr([{'keys': [], 'duration': 2}], cfg)
            self.intr = intr.numpy().astype(np.float32)[0]

    def clone(self):
        n = copy.copy(self)
        n.c2w, n.inputs = list(self.c2w), list(self.inputs)
        return n

    def set(self, rate=None, pitch_max=None, gain=None, move=None):
        if rate is not None:
            self.rate = float(rate)
        if gain is not None:
            self.gain = float(gain)
        if move is not None:
            self.move = float(move)
        if pitch_max is not None or gain is not None:
            pm = pitch_max if pitch_max is not None else self.pmax * (self.gain if gain is None else 2.5)
            self.pmax = min(85.0, float(pm)) / self.gain

    def _step(self, keys):
        k = set(keys)
        d = self.rate / FPS
        dy = d * (('right' in k) - ('left' in k))
        dp = d * (('up' in k) - ('down' in k))
        new = max(-self.pmax, min(self.pmax, self.pitch + dp))
        dp, self.pitch = new - self.pitch, new
        if dy:
            self.R = self.R @ _ry(dy)
        if dp:
            self.R = self.R @ _rx(dp)
        mv = np.zeros(3)
        fwd, right = self.R[:, 2], self.R[:, 0]
        mv += fwd * (('w' in k) - ('s' in k))
        mv += right * (('d' in k) - ('a' in k))
        self.p = self.p + mv * MOVE
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.R, self.p
        self.c2w.append(m)

    def next_window(self, keys):
        """keys: one key list held for the whole window, or one key list per video frame (n of them)."""
        n = 117 if self.w == 0 else 120
        if keys and isinstance(keys[0], (list, tuple, set, frozenset)):
            if len(keys) != n:
                raise ValueError('need %d per-frame key lists, got %d' % (n, len(keys)))
            pf = [sorted(set(k for k in ks if k in KEYS)) for ks in keys]
        else:
            pf = [sorted(set(k for k in keys if k in KEYS))] * n
        self.inputs.extend(pf)                 # frame i of the path reflects inputs 0..i-1
        while len(self.c2w) < len(self.inputs):
            self._step(self.inputs[len(self.c2w) - 1])
        self.w += 1
        nf = 120 * self.w - 3
        if self.tencent is None:
            blk = np.stack(self.c2w[nf - n:nf]).astype(np.float32)
            return np.linalg.inv(blk[0])[None] @ blk
        import torch
        w0 = max(0, self.w - self.cap)
        nk, f0 = self.w - w0, 120 * w0
        nfk = 120 * nk - 3
        path = np.stack(self.c2w[f0:nf]).astype(np.float32)
        pix = [120 * i for i in range(nk)] + [nfk]
        p = os.path.join(self.tmp, 'path.npz')
        np.savez(p, c2ws=path, intrinsics=np.repeat(self.intr[None], len(path), 0))
        cam = self.prepare(p, len(path), torch.device('cpu'), torch.float32, normalize_relative_to_first_frame=True,
                           normalize_framewise=True, normalize_displacement=True, normalize_displacement_per_video=True,
                           segment_boundaries_pixel=pix, translation_scale=self.move, rotation_scale=self.gain)
        ne = cam['norm_extrinsics'][0].numpy().astype(np.float32)
        if ne.shape != (len(path), 4, 4) or not np.isfinite(ne).all():
            raise RuntimeError('camera normalization gave %s' % (ne.shape,))
        return ne[len(path) - n:]
