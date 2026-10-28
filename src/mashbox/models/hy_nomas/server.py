"""Live Hy-Nomas server: play a world in the browser, steer it with the keyboard, add timed actions.

One process owns the chips. Each 5 s window is generated ahead of the play clock (``lead`` seconds before its first frame
plays), decoded in three parts in parallel on the decoder chips, crossfaded at the part seams, and sent at its 24 fps play
time as H.264 over a WebSocket (WebCodecs in the page). Keys held during the last 5 s drive the next window, frame by frame.

The server binds 127.0.0.1 only. Open it from your machine through an SSH tunnel (see docs/models/hy-nomas.md).
``StubEngine`` stands in for the chips (synthetic frames, fixed stage times) to test the page, controls and streaming on CPU.
"""
import base64
import hashlib
import json
import os
import queue
import re
import socket
import struct
import subprocess
import threading
import time
from importlib import resources

import numpy as np

from . import prompts as PR
from .cameras import Cameras, block_angles

HOST = '127.0.0.1'
LOOPBACK = {'127.0.0.1', 'localhost', '::1'}


def _hostname(v):
    """'127.0.0.1:8790', '[::1]:8790' or 'http://localhost:18873' -> the bare lower-case host name ('' when missing)."""
    v = (v or '').strip().lower()
    v = v.split('://', 1)[1] if '://' in v else v
    v = v.split('/', 1)[0]
    if v.startswith('['):
        return v[1:v.find(']')] if ']' in v else ''
    return v.rsplit(':', 1)[0] if v.count(':') == 1 else v


def request_allowed(headers, allow):
    """Pages on this machine only: the Host must be a loopback name (any port, so an SSH tunnel on another local port works) or
    one listed in MASHBOX_ALLOW_HOSTS, which refuses DNS-rebinding names; an Origin, when the browser sends one, must pass the
    same test, which refuses pages of other sites (their POSTs and their WebSockets)."""
    o = headers.get('Origin')
    return _hostname(headers.get('Host')) in allow and (o is None or _hostname(o) in allow)


def seed_of(v, default):
    """Seed from a request: a whole number from 0 to 2**32 - 1 (numpy's range), else ValueError (answered 400)."""
    v = default if v is None or v == '' else v                    # 0 is a seed, not "no seed"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise ValueError('seed must be a whole number from 0 to 4294967295')
    s = int(v)                                                   # a string that is not a whole number: ValueError
    if not 0 <= s < 2 ** 32:
        raise ValueError('seed must be a whole number from 0 to 4294967295')
    return s
FPS = 24.0
GUID = b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def page_bytes():
    return resources.files(__package__).joinpath('page.html').read_bytes()


def worlds():
    return {w['id']: w for w in json.loads(resources.files(__package__).joinpath('worlds.json').read_text())['worlds']}


# ------------------------------------------------------------------ engines
class Engine:
    """Main model + decoder + text encoder on the chips. ``model`` is app.Model (runtime, decoder, text encoder, settings)."""

    def __init__(self, model, log):
        self.m, self.log = model, log
        self.rt, self.dec, self.te = model.dit, model.decoder, model.text
        sp = model.split
        self.split = [(sum(sp[:i]), sum(sp[:i + 1])) for i in range(len(sp))]
        self.nch, self.ov = len(sp), model.overlap
        self.fast = getattr(model, 'fast_decoder', None)   # MASHBOX Fast Decoder (ours), None when not loaded
        self.decoders = ['full'] + (['fast'] if self.fast is not None else [])
        self.decoder = getattr(model, 'decoder_kind', 'full') if getattr(model, 'decoder_kind', 'full') in self.decoders else 'full'
        self.prevx = {}
        self.decq = [queue.Queue() for _ in range(self.nch)]
        self.done = threading.Condition()
        self.ndone = {}
        for ci in range(self.nch):
            threading.Thread(target=self._decw, args=(ci,), daemon=True).start()

    @property
    def steps(self):
        return self.rt.steps

    @property
    def preset(self):
        return self.rt.preset

    def encode(self, prompt):
        text, mask, info, timing = self.te.encode(prompt)
        return text, mask, info, timing

    def _decw(self, ci):
        while True:
            sess, w, j, dec, lat, start, warm, cb = self.decq[ci].get()
            t0 = time.time()
            fr = dec.call(ci, lat, start, warm) if dec is self.fast else dec.call(ci, lat, start)
            fr.block_until_ready()
            t1 = time.time()
            # hand the frames over before the part counts as done: barrier() returns on the count, and the --out render
            # reads every part right after it (the copy to the host can take longer than the other parts' decode)
            cb(sess, w, j, np.asarray(fr), dict(dec_s=round(t1 - t0, 4)))
            with self.done:
                self.ndone[(sess, w)] = self.ndone.get((sess, w), 0) + 1
                self.done.notify_all()

    def make_item(self, key, text, mask, info):
        return self.rt.make_item(key, text, mask, info)

    def hist_after(self, it):
        return self.rt.hist_after(it)

    def prepare(self, it, hist_len):
        t0 = time.time()
        self.rt.compile(it['bucket'], hist_len)
        return round(time.time() - t0, 2)

    def reset(self, seed):
        self.rt.reset(seed)
        self.prevx = {}

    def set_preset(self, preset):
        self.rt.set_preset(preset)

    def set_decoder(self, kind):
        if kind in self.decoders:
            self.decoder = kind

    def window(self, sess, w, blk, it, on_part):
        t0 = time.time()
        cam = self.rt.cam_tokens(blk)
        tabs = self.rt.tables(w)
        x, T = self.rt.window(cam, tabs, it)
        xd = np.asarray(x)
        frames = xd.shape[1]
        pvx = self.prevx.get(sess) if w > 0 else None
        dec = self.fast if self.decoder == 'fast' else self.dec
        if dec is self.fast:                               # context for the fast decoder: last window + this one
            off = pvx.shape[1] if pvx is not None else 0
            ctx = np.concatenate([pvx, xd], axis=1) if pvx is not None else xd
        for j, (s, e) in enumerate(self.split):
            if self.ov and (j > 0 or pvx is not None):
                pv = pvx[:, frames - self.ov:frames] if j == 0 else xd[:, s - self.ov:s]
                lat, start = np.concatenate([pv, xd[:, s:e]], axis=1)[None], w * frames + s - self.ov
            else:
                lat, start = xd[:, s:e][None], w * frames + s
            warm = 0
            if dec is self.fast:                           # same frames out; up to WARM latents in front, dropped after
                a = off + s - (lat.shape[2] - (e - s))
                b = max(0, a - dec.warm)
                warm, lat = a - b, ctx[:, b:off + e][None]
            self.decq[j].put((sess, w, j, dec, lat, start, warm, on_part))
        self.prevx = {sess: xd}
        return dict(dit_s=round(time.time() - t0, 4), text=T['text'], action_mask=T['action_mask'], decoder=self.decoder)

    def barrier(self, sess, w):
        with self.done:
            self.done.wait_for(lambda: self.ndone.get((sess, w), 0) >= self.nch)
            self.ndone.pop((sess, w), None)

    def memory(self):
        out = []
        for d in self.rt.devices:
            try:
                m = d.memory_stats() or {}
                out.append([round(m.get('bytes_in_use', 0) / 2**30, 2), round(m.get('peak_bytes_in_use', 0) / 2**30, 2)])
            except Exception:                                    # not every backend reports memory
                out.append(None)
        return out


class StubEngine:
    """CPU stand-in: no model. Sleeps fixed stage times and returns synthetic frames that show the window number and the
    camera turn of that window. Tests pacing, controls and streaming only; it says STUB on every frame."""

    def __init__(self, log, dit_s=3.1, dec_s=1.6, encode_s=0.2, split=(10, 10, 10), overlap=1):
        self.log, self.dit_s, self.dec_s, self.encode_s = log, dit_s, dec_s, encode_s
        self.nch, self.ov = len(split), overlap
        self.split = [(sum(split[:i]), sum(split[:i + 1])) for i in range(len(split))]
        self.steps, self.preset = 3, 'quality'
        self.decoders, self.decoder = ['full', 'fast'], 'full'
        self.decq = queue.Queue()
        self.done = threading.Condition()
        self.ndone = {}
        for _ in range(self.nch):
            threading.Thread(target=self._decw, daemon=True).start()

    def encode(self, prompt):
        time.sleep(self.encode_s)
        n = max(40, min(500, len(prompt) // 4))
        acts = [m.start() // 4 for m in re.finditer('<Action', prompt)]
        offs = (acts + [n]) if len(acts) >= 2 else None
        info = dict(valid_post_crop=n, offs=offs, win=None, action_start=acts[0] if acts else -1)
        mask = np.ones(n, bool)
        return np.zeros((n, 8), np.float32), mask, info, dict(s=self.encode_s)

    def _decw(self):
        while True:
            sess, w, j, n, yaw, pitch, cb = self.decq.get()
            t0 = time.time()
            time.sleep(self.dec_s)
            cb(sess, w, j, synthetic(w, j, n, yaw, pitch), dict(dec_s=round(time.time() - t0, 4)))
            with self.done:                                # counted after the hand-over, as in Engine
                self.ndone[(sess, w)] = self.ndone.get((sess, w), 0) + 1
                self.done.notify_all()

    def make_item(self, key, text, mask, info):
        offs = (info or {}).get('offs')
        return dict(key=key, n=int(np.asarray(mask).sum()), intra=bool(offs and len(offs) >= 2), head=None, bucket=256)

    def hist_after(self, it):
        return 18 * 880 + it['bucket']

    def prepare(self, it, hist_len):
        return 0.0

    def reset(self, seed):
        pass

    def set_preset(self, preset):
        self.preset, self.steps = preset, (3 if preset == 'quality' else 4)

    def set_decoder(self, kind):
        if kind in self.decoders:
            self.decoder = kind

    def window(self, sess, w, blk, it, on_part):
        t0 = time.time()
        time.sleep(self.dit_s)
        yaw, pitch = block_angles(blk) if blk is not None else (0.0, 0.0)
        for j, (s, e) in enumerate(self.split):
            n = e - s + (self.ov if (self.ov and (j > 0 or w > 0)) else 0)
            self.decq.put((sess, w, j, 4 * n, yaw, pitch, on_part))
        return dict(dit_s=round(time.time() - t0, 4), text=it['key'], action_mask=it['intra'])

    def barrier(self, sess, w):
        with self.done:
            self.done.wait_for(lambda: self.ndone.get((sess, w), 0) >= self.nch)
            self.ndone.pop((sess, w), None)

    def memory(self):
        return []


# 5 x 7 digits for the stub frames (no font library needed)
_DIGITS = ['111101101101111', '010110010010111', '111001111100111', '111001111001111', '101101111001001',
           '111100111001111', '111100111101111', '111001001001001', '111101111101111', '111101111001111']


def synthetic(w, j, n, yaw, pitch, h=704, wd=1280):
    """n stub frames: a gradient that slides with the window's camera turn, the window number in big digits, a STUB bar."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, wd, dtype=np.float32)[None, :]
    out = np.empty((n, h, wd, 3), np.uint8)
    for f in range(n):
        ph = (w * 120 + j * 40 + f) / 120.0
        shift = ph * (yaw / 90.0)
        r = 0.5 + 0.5 * np.sin(2 * np.pi * (x + shift) * 2)
        g = 0.5 + 0.5 * np.sin(2 * np.pi * (y + pitch / 180.0) * 2)
        img = np.stack([r * np.ones_like(y), g * np.ones_like(x), 0.35 * np.ones((h, wd), np.float32)], -1)
        out[f] = (img * 255).astype(np.uint8)
    for k, ch in enumerate(str(w + 1)):
        bits = _DIGITS[int(ch)]
        for i, b in enumerate(bits):
            if b == '1':
                r0, c0 = 40 + (i // 3) * 24, 40 + k * 90 + (i % 3) * 24
                out[:, r0:r0 + 22, c0:c0 + 22] = 255
    out[:, h - 40:h - 20, 40:400] = (255, 200, 0)
    return out


# ------------------------------------------------------------------ text cache
class Texts:
    """prompt -> (text, mask, offsets), cached by the sha256 of the exact prompt string (in memory)."""

    def __init__(self, engine, limit=256):
        self.eng, self.limit = engine, limit
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, prompt):
        sha = hashlib.sha256(prompt.encode()).hexdigest()
        with self.lock:
            hit = self.cache.get(sha)
        if hit is not None:
            return hit + (dict(sha=sha[:12], cached=True, s=0.0),)
        with self.lock:                       # one encode at a time (the encoder uses every chip)
            t0 = time.time()
            text, mask, info, _ = self.eng.encode(prompt)
            if len(self.cache) >= self.limit:
                self.cache.pop(next(iter(self.cache)))
            self.cache[sha] = (text, mask, info)
            return text, mask, info, dict(sha=sha[:12], cached=False, s=round(time.time() - t0, 3))


# ------------------------------------------------------------------ play clock and stream
class Clock:
    def __init__(self):
        self.g0 = self.t0 = None
        self.lock = threading.Lock()

    def anchor(self, g, t):
        with self.lock:
            self.g0, self.t0 = g, t

    def clear(self):
        with self.lock:
            self.g0 = self.t0 = None

    def play(self, g):
        with self.lock:
            return None if self.t0 is None else self.t0 + (g - self.g0) / FPS


def nals(buf):
    """Annex-B byte string -> NAL payloads (no start codes)."""
    out, n = [], len(buf)
    starts = [m.start() for m in re.finditer(b'\x00\x00\x01', buf)]
    for k, s in enumerate(starts):
        b = s + 3
        e = starts[k + 1] if k + 1 < len(starts) else n
        if k + 1 < len(starts) and buf[e - 1] == 0:
            e -= 1
        out.append(buf[b:e])
    return out


class Stream:
    """Frames at play time -> x264 (zero latency, no B-frames) -> access units (AVCC) -> every client."""

    def __init__(self, log, bitrate='6M', gop=48, size=(1280, 704)):
        self.log, self.size = log, size
        self.p = subprocess.Popen([ffmpeg_exe(), '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % size,
                                   '-r', '24', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
                                   '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-g', str(gop), '-keyint_min', str(gop), '-bf', '0',
                                   '-b:v', bitrate, '-maxrate', bitrate, '-bufsize', bitrate,
                                   '-x264-params', 'aud=1:repeat-headers=1:scenecut=0', '-f', 'h264', '-'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.meta = queue.Queue()
        self.clients = set()
        self.cl = threading.Lock()
        self.config = None
        self.sent = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def push(self, img, meta):
        self.meta.put(meta)
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())
        if any(c.mode == 'jpeg' for c in list(self.clients)):
            jpg = to_jpeg(img)
            if jpg is not None:
                hdr = struct.pack('<BBHIdd', 2, 1, meta['sess'] & 0xFFFF, meta['g'], meta['play'], time.time())
                self._send(hdr + jpg, True, 'jpeg')

    def _send(self, msg, key, mode):
        with self.cl:
            cs = [c for c in self.clients if c.mode == mode]
        for c in cs:
            c.offer(msg, key)

    def _reader(self):
        buf, au = b'', []
        while True:
            chunk = self.p.stdout.read(65536)
            if not chunk:
                self.log(json.dumps(dict(event='encoder_exit')))
                return
            buf += chunk
            last = buf.rfind(b'\x00\x00\x01')
            if last <= 0:
                continue
            head, buf = buf[:last], buf[last:]
            if head.endswith(b'\x00'):
                head, buf = head[:-1], b'\x00' + buf
            for nal in nals(head):
                if not nal:
                    continue
                if nal[0] & 0x1F == 9 and au:
                    self._emit(au)
                    au = []
                au.append(nal)

    def _emit(self, nl):
        meta = self.meta.get()
        sps = [n for n in nl if n[0] & 0x1F == 7]
        pps = [n for n in nl if n[0] & 0x1F == 8]
        key = any(n[0] & 0x1F == 5 for n in nl)
        if sps and pps:
            s, p = sps[0], pps[0]
            desc = bytes([1, s[1], s[2], s[3], 0xFF, 0xE1]) + len(s).to_bytes(2, 'big') + s + bytes([1]) + len(p).to_bytes(2, 'big') + p
            cfg = json.dumps(dict(type='config', codec='avc1.%02x%02x%02x' % (s[1], s[2], s[3]), desc=base64.b64encode(desc).decode(),
                                  w=self.size[0], h=self.size[1]))
            if cfg != self.config:
                self.config = cfg
                with self.cl:
                    cs = list(self.clients)
                for c in cs:
                    c.send_text(cfg)
        body = b''.join(len(n).to_bytes(4, 'big') + n for n in nl if n[0] & 0x1F in (1, 5))
        hdr = struct.pack('<BBHIdd', 1, int(key), meta['sess'] & 0xFFFF, meta['g'], meta['play'], time.time())
        self._send(hdr + body, key, 'h264')
        self.sent += 1


def to_jpeg(img):
    try:
        from PIL import Image
    except ImportError:
        return None
    import io
    b = io.BytesIO()
    Image.fromarray(img).resize((960, 528)).save(b, format='JPEG', quality=78)
    return b.getvalue()


class Client:
    def __init__(self, sock, mode):
        self.sock, self.mode = sock, mode
        self.q = queue.Queue(maxsize=96)
        self.need_key, self.alive, self.drops = True, True, 0

    @staticmethod
    def frame(op, data):
        n = len(data)
        h = bytes([0x80 | op]) + (bytes([n]) if n < 126 else bytes([126]) + n.to_bytes(2, 'big') if n < 65536 else bytes([127]) + n.to_bytes(8, 'big'))
        return h + data

    def offer(self, msg, key):
        if self.need_key and not key:
            return
        self.need_key = False
        try:
            self.q.put_nowait(self.frame(2, msg))
        except queue.Full:                    # slow viewer: drop to the next key frame
            self.need_key = True
            self.drops += 1
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

    def send_text(self, s):
        try:
            self.q.put_nowait(self.frame(1, s.encode()))
        except queue.Full:
            pass

    def writer(self):
        try:
            while self.alive:
                m = self.q.get()
                if m is None:
                    return
                self.sock.sendall(m)
        except OSError:
            self.alive = False


def read_ws(sock):
    def rd(n):
        b = b''
        while len(b) < n:
            c = sock.recv(n - len(b))
            if not c:
                raise ConnectionError
            b += c
        return b
    h = rd(2)
    op, n = h[0] & 0x0F, h[1] & 0x7F
    if n == 126:
        n = int.from_bytes(rd(2), 'big')
    elif n == 127:
        n = int.from_bytes(rd(8), 'big')
    if n > 65536:
        raise ConnectionError('frame too big')
    mask = rd(4) if h[1] & 0x80 else None
    d = rd(n)
    if mask:
        d = bytes(b ^ mask[i % 4] for i, b in enumerate(d))
    return op, d


# ------------------------------------------------------------------ the live session
class Live:
    def __init__(self, engine, log, hw='', seed=2701, preset='quality', lead=5.6, slow_extra=0.9, buffer=3.0, idle_min=15.0,
                 camera=None, tencent_cam=None, stub=False):
        self.eng, self.log, self.hw, self.seed, self.stub = engine, log, hw, seed, stub
        self.lead, self.slow_extra, self.buffer, self.idle_min = lead, slow_extra, buffer, idle_min
        self.tencent_cam = tencent_cam
        c = camera or {}
        self.cv = threading.Condition()
        self.sess, self.mode, self.world, self.w = 0, 'idle', None, 0
        self.cams, self.req = None, []
        self.settings = dict(rate=float(c.get('turn_deg_per_s', 25.0)), pitch_max=float(c.get('pitch_limit_deg', 60)), move=1.0,
                             gain=float(c.get('rotation_scale', 2.5)), preset=preset, steps=engine.steps,
                             decoder=getattr(engine, 'decoder', 'full'))
        self.actions, self.next_aid, self.win = [], 1, []
        self.late = self.frames = 0
        self.msg, self.loading, self.pending_world, self.load_est = '', None, None, None
        self.held, self.next_lock = set(), None
        self.kev = [(0.0, frozenset())]
        self.clients, self.t_last_client, self.idle_paused = 0, time.time(), False
        self.clock = Clock()
        self.parts, self.pl = {}, threading.Condition()
        self.texts = Texts(engine)
        self.lib = {}
        self.worlds = worlds()
        self.stream = None
        self.events = []

    def event(self, kind, **kw):
        kw.update(kind=kind, t=round(time.time(), 3))
        self.events = (self.events + [kw])[-500:]

    # ---------------------------------------------------------- frames
    def on_part(self, sess, w, j, frames, info):
        with self.pl:
            self.parts[(sess, w, j)] = (frames, info)
            self.pl.notify_all()

    def pacer(self):
        """Parts in stream order -> each frame to the encoder at its play time; crossfade 4 frames at each part seam."""
        cur, w, j, hold, gb = None, 0, 0, None, 0
        nch, ovn = self.eng.nch, 4 * self.eng.ov
        while True:
            with self.pl:
                while True:
                    s = self.sess
                    if s != cur:
                        for k in [k for k in self.parts if k[0] != s]:
                            self.parts.pop(k)
                        cur, w, j, hold, gb = s, 0, 0, None, 0
                        self.clock.clear()
                    if (cur, w, j) in self.parts:
                        frames, _ = self.parts.pop((cur, w, j))
                        break
                    self.pl.wait(timeout=0.5)
            frames, hold = crossfade(frames, hold, ovn)
            for f in range(frames.shape[0]):
                if self.sess != cur:
                    break
                g = gb + f
                if self.clock.play(g) is None:
                    self.clock.anchor(g, time.time() + self.buffer)
                play = self.clock.play(g)
                d = play - time.time()
                if d > 0:
                    time.sleep(d)
                now = time.time()
                late = now > play + 1 / FPS
                if now > play + 0.5:                 # a stall: re-anchor so the stream does not stay behind
                    self.clock.anchor(g, now)
                    self.event('reanchor', g=g, behind_s=round(now - play, 3))
                self.late += int(late)
                self.frames += 1
                self.stream.push(frames[f], dict(sess=cur, g=g, play=play))
            gb += frames.shape[0]
            j += 1
            if j == nch:
                j, w = 0, w + 1

    # ---------------------------------------------------------- keys
    def timeline(self, n, t_end, now=None):
        """Per-frame keys for a window of n frames = what was held over the n/24 s before t_end."""
        with self.cv:
            ev = list(self.kev)
        now = time.time() if now is None else now
        out = []
        for i in range(n):
            t = min(t_end - (n - i) / FPS, now)
            k = frozenset()
            for t0, ks in ev:
                if t0 > t:
                    break
                k = ks
            out.append(sorted(k))
        return out

    def keys_lock(self, n):
        now = time.time()
        pf = self.timeline(n, now, now)
        with self.cv:
            cut = now - 60.0
            keep = [e for e in self.kev if e[0] >= cut]
            before = [e for e in self.kev if e[0] < cut]
            self.kev = (before[-1:] + keep) if before else keep
        return pf

    def on_keys(self, held):
        held = set(k for k in held if k in ('w', 's', 'a', 'd', 'up', 'down', 'left', 'right'))
        with self.cv:
            if held != self.held:
                self.held = held
                self.kev.append((time.time(), frozenset(held)))

    # ---------------------------------------------------------- camera
    def new_cams(self):
        s = self.settings
        return Cameras(self.tencent_cam, rate=s['rate'], pitch_max=s['pitch_max'], gain=s['gain'], move=s['move'])

    def apply_cam(self, c):
        s = self.settings
        c.rate, c.gain, c.move = s['rate'], s['gain'], s['move']
        c.pmax = min(85.0, s['pitch_max']) / c.gain
        return c

    # ---------------------------------------------------------- the window loop
    def producer(self):
        ahead = None
        while True:
            with self.cv:
                while True:
                    if self.req:
                        r = self.req.pop(0)
                        break
                    if self.mode == 'playing' and not self.idle_paused:
                        r = None
                        break
                    self.cv.wait(timeout=0.5)
            if r is not None:
                if r['kind'] == 'world':
                    self.world = r['world']
                try:
                    self.eng.reset(r.get('seed', self.world['seed']))
                except Exception as e:  # one bad request never ends this thread
                    self.log(json.dumps(dict(event='reset_failed', err=repr(e)[:300])))
                    with self.cv:
                        self.mode, self.msg = 'paused', 'That start failed; pick a world or press Restart.'
                    continue
                with self.cv:
                    self.sess += 1
                    self.w, self.cams, self.mode, self.msg, ahead = 0, self.new_cams(), 'playing', '', None
                    for x in self.actions:
                        if x['status'] in ('ready', 'queued', 'encoding'):
                            x['status'] = 'dropped (new session)'
                self.clock.clear()
                self.event('session', sess=self.sess, world=self.world['id'])
                continue
            sess, w = self.sess, self.w
            if w > 0 and not self._wait_turn(sess, w):
                continue
            if self.settings['preset'] != self.eng.preset:
                self.eng.set_preset(self.settings['preset'])
                self.settings['steps'] = self.eng.steps
            if self.settings.get('decoder', 'full') != getattr(self.eng, 'decoder', 'full'):
                self.eng.set_decoder(self.settings['decoder'])
            nfr = 117 if w == 0 else 120
            keys = self.keys_lock(nfr)
            t_sub = time.time()
            it, act = self.world['items'][0 if w == 0 else 1], None
            if w > 0:
                with self.cv:
                    act = next((x for x in self.actions if x['status'] == 'ready' and x['sess'] == sess), None)
                if act is not None:
                    it = act['item']
                    act['status'], act['window'] = 'playing', w + 1
                    act['plays_at'] = self.clock.play(w * 120 + int(act['span'][0] * FPS))
            sig = (tuple(tuple(k) for k in keys), self.settings['rate'], self.settings['pitch_max'], self.settings['move'], self.settings['gain'])
            if ahead is not None and ahead['w'] == w and ahead['sig'] == sig:
                self.cams, blk = ahead['cams'], ahead['blk']
            else:
                blk = self.apply_cam(self.cams).next_window(keys)
            ahead = None
            try:
                T = self.eng.window(sess, w, blk, it, self.on_part)
            except Exception as e:
                self.log(json.dumps(dict(event='window_failed', w=w, err=repr(e)[:400])))
                with self.cv:
                    self.mode, self.msg = 'paused', 'Generation stopped on an error; press Restart.'
                continue
            if self.mode == 'playing':                    # next window's camera, made while the decoder runs
                c2 = self.apply_cam(self.cams.clone())
                pn = self.clock.play((w + 1) * 120)
                k2 = self.timeline(120, (pn - self.lead) if pn is not None else time.time() + 5.0, time.time())
                ahead = dict(w=w + 1, sig=(tuple(tuple(k) for k in k2),) + sig[1:], cams=c2, blk=c2.next_window(k2))
            self.eng.barrier(sess, w)
            t_done = time.time()
            rec = dict(sess=sess, w=w, t_sub=round(t_sub, 3), t_done=round(t_done, 3), compute=round((t_done - t_sub) / 5.0, 4),
                       steps=self.eng.steps, preset=self.eng.preset, action=act['id'] if act else None, **T)
            if w < 3 or w % 6 == 0:
                rec['memory_gib'] = self.eng.memory()
            self.event('window', **rec)
            with self.cv:
                self.win = (self.win + [rec])[-40:]
                self.w = w + 1
            if act is not None:
                act['status'] = 'done'

    def _wait_turn(self, sess, w):
        """Wait until 'lead' s before window w plays (earlier for the slower first windows and action windows)."""
        slow = w <= 2 or any(x['status'] == 'ready' and x['sess'] == sess for x in list(self.actions))
        while True:
            p = self.clock.play(w * 120)
            tgt = None if p is None else p - self.lead - (self.slow_extra if slow else 0.0)
            with self.cv:
                self.next_lock = tgt
                if self.req or self.sess != sess or self.mode != 'playing':
                    return False
            if tgt is None:
                if self.win and time.time() - self.win[-1]['t_done'] > 30:
                    return True
                time.sleep(0.02)
                continue
            if time.time() >= tgt:
                return True
            time.sleep(min(0.02, tgt - time.time()))

    # ---------------------------------------------------------- worlds and actions
    def prep_world(self, spec):
        t_start, wd = time.time(), None
        try:
            with self.cv:
                self.loading = dict(title=spec['title'], status='reading the prompt', t0=time.time())
            t1, m1, i1, e1 = self.texts.get(spec['seg1'])
            t2, m2, i2, e2 = self.texts.get(spec['seg2'])
            s1 = self.eng.make_item(spec['id'] + '_seg1', t1, m1, i1)
            s2 = self.eng.make_item(spec['id'] + '_seg2', t2, m2, i2)
            self.loading['status'] = 'preparing the model for this prompt'
            c = [self.eng.prepare(s1, 0), self.eng.prepare(s2, self.eng.hist_after(s1)), self.eng.prepare(s2, self.eng.hist_after(s2))]
            wd = dict(spec, items=[s1, s2], base=PR.base_action(spec['seg2']), name=PR.display_name(spec['seg1']))
            self.event('world_ready', id=spec['id'], encode=[e1, e2], compile_s=c)
            with self.cv:
                self.req.append(dict(kind='world', world=wd, seed=spec['seed']))
                self.loading = None
                self.load_est = round(time.time() - t_start, 1)
                self.cv.notify_all()
            threading.Thread(target=self.prewarm, args=(wd,), daemon=True).start()
        except Exception as e:
            self.log(json.dumps(dict(event='world_failed', err=repr(e)[:400])))
            with self.cv:
                self.loading, self.msg = None, 'World failed to load: ' + str(e)[:160]
        with self.cv:
            spec, self.pending_world = self.pending_world, None
            if spec is not None and wd is not None:
                self.loading = dict(title=spec['title'], status='starts after ' + wd['title'], t0=time.time())
        if spec is not None:
            # The page says a queued world "starts after this one": let the world just loaded make its first two windows
            # (at most 30 s) before the next one is prepared. Without this a fast second prompt lands in the queue right
            # behind the first and the producer takes both back to back, so the first world never shows.
            t_hold = time.time()
            while wd is not None and time.time() - t_hold < 30 and not (self.world is wd and self.w >= 2):
                time.sleep(0.1)
            threading.Thread(target=self.prep_world, args=(spec,), daemon=True).start()

    def prewarm(self, wd):
        """Encode the preset actions at default timing in the background, so those buttons play in the next window."""
        for _ in range(600):
            if self.world is wd:
                break
            time.sleep(0.1)
        for line in PR.PRESET_ACTIONS:
            if self.world is not wd:
                return
            try:
                seg2, _ = PR.action_seg2(wd['seg2'], line, 1.0, 1.0, True)
                h = hashlib.sha256(seg2.encode()).hexdigest()
                if h not in self.lib:
                    t, m, info, _ = self.texts.get(seg2)
                    it = self.eng.make_item('pre_' + h[:8], t, m, info)
                    self.eng.prepare(it, self.eng.hist_after(wd['items'][1]))
                    self.lib[h] = it
            except Exception as e:
                self.log(json.dumps(dict(event='prewarm_failed', line=line, err=repr(e)[:200])))

    def prep_action(self, x):
        try:
            wd = self.world
            h = hashlib.sha256(x['seg2'].encode()).hexdigest()
            if h in self.lib:
                x['item'] = self.lib[h]
            else:
                x['status'] = 'encoding'
                t, m, info, ti = self.texts.get(x['seg2'])
                x['item'] = self.eng.make_item('act%d' % x['id'], t, m, info)
                x['status'] = 'preparing'
                self.eng.prepare(x['item'], self.eng.hist_after(wd['items'][1]))
            x['amask'] = bool(x['item'].get('intra'))
            if not x['amask']:
                x['note'] = 'no timed split found by the tokenizer; the line applies to the whole window'
            if x['status'] != 'dropped (new session)':
                x['status'] = 'ready'
        except Exception as e:
            x['status'] = 'failed: ' + str(e)[:160]

    # ---------------------------------------------------------- HTTP API
    def stats(self):
        with self.cv:
            wins = [x for x in self.win if x['sess'] == self.sess]
            last = wins[-1] if wins else None
            acts = [dict(id=x['id'], text=x['text'], span=x['span'], status=x['status'], window=x.get('window'),
                         plays_at=x.get('plays_at'), amask=x.get('amask'), note=x.get('note')) for x in self.actions[-6:]]
            p = self.clock.play(self.w * 120) if self.w else None
            wd = self.world
            return dict(type='stats', now=time.time(), mode='paused' if self.idle_paused else self.mode, sess=self.sess, hw=self.hw,
                        world=dict(id=wd['id'], title=wd['title'], seed=wd['seed'], base=wd['base'], name=wd['name']) if wd else None,
                        window=(last['w'] + 1) if last else 0, compute=last['compute'] if last else None,
                        compute_avg=round(float(np.mean([x['compute'] for x in wins[1:]])), 3) if len(wins) > 1 else None,
                        steps=self.eng.steps, preset=self.eng.preset, keys_held=sorted(self.held),
                        decoder=getattr(self.eng, 'decoder', 'full'), decoders=list(getattr(self.eng, 'decoders', ['full'])),
                        next_lock_in=round(self.next_lock - time.time(), 2) if self.next_lock and self.mode == 'playing' else None,
                        next_window_plays_in=round(p - time.time(), 2) if p else None,
                        late=self.late, frames=self.frames, actions=acts, settings=dict(self.settings), msg=self.msg,
                        loading=dict(title=self.loading['title'], status=self.loading['status'], s=round(time.time() - self.loading['t0'], 1),
                                     est=self.load_est) if self.loading else None,
                        queued_world=self.pending_world['title'] if self.pending_world else None, stub=self.stub)

    def api(self, path, body):
        if path == '/api/world':
            mode = body.get('mode', 'preset')
            seed = seed_of(body.get('seed'), self.seed)
            if mode == 'preset':
                w = self.worlds[body['id']]
                spec = dict(id=w['id'], title=w['title'], seg1=w['seg1'], seg2=w['seg2'])
            elif mode == 'form':
                s1, s2 = PR.world_from_form(body)
                spec = dict(id='custom', title=PR.clean_text(body.get('name'), 40) or 'Custom world', seg1=s1, seg2=s2)
            else:
                s1, s2 = str(body.get('seg1', '')), str(body.get('seg2', ''))
                PR.check_json_world(s1, s2)
                spec = dict(id='json', title=PR.clean_text(PR.display_name(s1), 40) or 'JSON world', seg1=s1, seg2=s2)
            spec['seed'] = seed
            with self.cv:
                busy = self.loading is not None
                if busy:
                    self.pending_world = spec
            if not busy:
                threading.Thread(target=self.prep_world, args=(spec,), daemon=True).start()
            return dict(ok=True, loading=spec['title'], queued=busy, seg1=spec['seg1'], seg2=spec['seg2'])
        if path == '/api/action':
            if not self.world:
                raise ValueError('start a world first')
            subj = bool(body.get('subject', True))
            line = PR.action_text(self.world['seg1'], body.get('text', ''), subj)
            seg2, span = PR.action_seg2(self.world['seg2'], line, body.get('start', 1.0), body.get('dur', 1.0), subj)
            with self.cv:
                x = dict(id=self.next_aid, text=line, span=span, seg2=seg2, status='queued', sess=self.sess)
                self.next_aid += 1
                self.actions = (self.actions + [x])[-30:]
            threading.Thread(target=self.prep_action, args=(x,), daemon=True).start()
            return dict(ok=True, id=x['id'], line=line, span=span, seg2=seg2)
        if path == '/api/restart':
            if not self.world:
                raise ValueError('no world yet')
            with self.cv:
                seed = seed_of(body.get('seed'), self.world['seed'])
                self.world['seed'] = seed
                self.req.append(dict(kind='restart', seed=seed))
                self.cv.notify_all()
            return dict(ok=True, seed=seed)
        if path == '/api/pause':
            with self.cv:
                if self.world:
                    self.mode = 'paused' if body.get('on') else 'playing'
                    if self.mode == 'playing':
                        self.clock.clear()
                    self.cv.notify_all()
            return dict(ok=True, mode=self.mode)
        if path == '/api/settings':
            s = self.settings
            with self.cv:
                if 'rate' in body:
                    s['rate'] = float(min(90, max(5, float(body['rate']))))
                if 'pitch_max' in body:
                    s['pitch_max'] = float(min(85, max(5, float(body['pitch_max']))))
                if 'move' in body:
                    s['move'] = float(min(3, max(0.25, float(body['move']))))
                if 'gain' in body:
                    s['gain'] = float(min(4, max(1, float(body['gain']))))
                if body.get('preset') in ('quality', 'live'):
                    s['preset'] = body['preset']
                if body.get('decoder') in getattr(self.eng, 'decoders', ['full']):
                    s['decoder'] = body['decoder']
            return dict(ok=True, settings=dict(s))
        raise KeyError(path)

    def ticker(self):
        while True:
            time.sleep(0.5)
            s = json.dumps(self.stats())
            with self.stream.cl:
                cs = list(self.stream.clients)
            for c in cs:
                c.send_text(s)
            with self.cv:
                if self.clients == 0 and not self.idle_paused and self.mode == 'playing' and time.time() - self.t_last_client > self.idle_min * 60:
                    self.idle_paused = True
                    self.log(json.dumps(dict(event='idle_pause')))

    def serve(self, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        page = page_bytes()
        live = self
        wl = [dict(id=w['id'], title=w['title'], seg1=w['seg1'], seg2=w['seg2']) for w in self.worlds.values()]
        allow = LOOPBACK | {h.strip().lower() for h in os.environ.get('MASHBOX_ALLOW_HOSTS', '').split(',') if h.strip()}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *x):
                pass

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _refuse(self, code, err):
                self.close_connection = True
                self._json(code, dict(err=err))

            def do_GET(self):
                if not request_allowed(self.headers, allow):
                    return self._refuse(403, 'host or origin not allowed: open the page through localhost (see the docs)')
                path = self.path.split('?')[0]
                if path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif path == '/api/state':
                    self._json(200, dict(live.stats(), worlds=wl, styles=PR.STYLES, views=PR.VIEWS))
                elif path == '/ws':
                    self._ws()
                elif path == '/favicon.ico':
                    self.send_response(204)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                else:
                    self._json(404, dict(err='not found'))

            def do_POST(self):
                if not request_allowed(self.headers, allow):
                    return self._refuse(403, 'host or origin not allowed: open the page through localhost (see the docs)')
                if not (self.headers.get('Content-Type') or '').lower().startswith('application/json'):
                    return self._refuse(415, 'send JSON (Content-Type: application/json)')
                n = int(self.headers.get('Content-Length', 0) or 0)
                if n > 20000:
                    return self._json(413, dict(err='too big'))
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                    self._json(200, live.api(self.path.split('?')[0], body))
                except KeyError as e:
                    self._json(404, dict(err='unknown ' + str(e)[:80]))
                except Exception as e:
                    self._json(400, dict(err=str(e)[:200]))

            def _ws(self):
                k = self.headers.get('Sec-WebSocket-Key', '')
                if self.headers.get('Upgrade', '').lower() != 'websocket' or not k:
                    return self._json(400, dict(err='websocket only'))
                acc = base64.b64encode(hashlib.sha1(k.encode() + GUID).digest()).decode()
                self.send_response(101)
                self.send_header('Upgrade', 'websocket')
                self.send_header('Connection', 'Upgrade')
                self.send_header('Sec-WebSocket-Accept', acc)
                self.end_headers()
                self.wfile.flush()
                mode = 'jpeg' if 'mode=jpeg' in self.path else 'h264'
                c = Client(self.connection, mode)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=c.writer, daemon=True).start()
                if live.stream.config and mode == 'h264':
                    c.send_text(live.stream.config)
                with live.stream.cl:
                    live.stream.clients.add(c)
                with live.cv:
                    live.clients += 1
                    live.t_last_client = time.time()
                    if live.idle_paused:
                        live.idle_paused = False
                        live.clock.clear()
                        live.cv.notify_all()
                try:
                    while c.alive:
                        op, d = read_ws(self.connection)
                        if op == 8:
                            break
                        if op == 9:
                            c.q.put(c.frame(10, d))
                            continue
                        if op == 1:
                            j = json.loads(d.decode())
                            if j.get('t') == 'keys':
                                live.on_keys(j.get('held', []))
                except (ConnectionError, OSError, ValueError):
                    pass
                finally:
                    c.alive = False
                    c.q.put(None)
                    with live.stream.cl:
                        live.stream.clients.discard(c)
                    with live.cv:
                        live.clients -= 1
                        live.t_last_client = time.time()
                    self.close_connection = True

        ThreadingHTTPServer.daemon_threads = True
        srv = ThreadingHTTPServer((HOST, port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.log(json.dumps(dict(event='serving', url='http://%s:%d/' % (HOST, port))))
        return srv

    def start(self, port, autostart=None):
        self.stream = Stream(self.log)
        for f in (self.producer, self.pacer, self.ticker):
            threading.Thread(target=f, daemon=True).start()
        srv = self.serve(port)
        if autostart:
            self.api('/api/world', dict(mode='preset', id=autostart, seed=self.seed))
        return srv


def crossfade(frames, hold, ovn):
    """Blend the first ovn frames of a part into the held tail of the previous part; hold back this part's own tail."""
    if not ovn:
        return frames, None
    if hold is not None:
        wv = ((np.arange(ovn) + 1) / (ovn + 1)).astype(np.float32)[:, None, None, None]
        bl = np.clip(np.round(hold.astype(np.float32) * (1 - wv) + frames[:ovn].astype(np.float32) * wv), 0, 255).astype(np.uint8)
        frames = np.concatenate([bl, frames[ovn:]])
    return frames[:-ovn], frames[-ovn:]
