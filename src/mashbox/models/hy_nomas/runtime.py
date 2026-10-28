"""Window loop of the main model: noise schedule, history, text items, and one compiled program per shape.

A world is generated 5 s (30 latent frames) at a time. Each window runs ``steps`` denoising passes. Its keys and values
become the next window's history: 6 "sink" frames from the first window plus the window's last 12 frames and its text,
stored as int8. Two ways to make that history:
  quality: after the denoising steps, one extra pass on the clean result renoised to sigma 0.206897 (Tencent's
           history pass), 3 steps by default;
  live:    the history comes from the last denoising pass (no extra pass), 4 steps on a schedule whose last input
           sigma is 0.206897.
Both run 4 passes per window.

Text: every text is cut to its real tokens, padded with zero rows to a bucket (256 or 512 tokens) and masked by group
ids, so the compiled programs depend only on (text bucket, history length). Tencent's two text masks are applied the
same way: the per-action text mask inside a window (intra_segment_action_mask) and, in the history, the previous text
cut to its head (cross_segment_text_head_only).
"""
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from . import buckets as B
from . import dit as M

LIVE_SIGMAS = (1.0, 0.915129, 0.743017, 0.206897, 0.0)     # 4 steps; the last input sigma is the history sigma
HISTORY_SIGMA = 0.206897


def sigmas(steps, preset):
    if preset == 'live' and steps == 4:
        return np.array(LIVE_SIGMAS, np.float32)
    s = np.linspace(1, 0, steps + 1, dtype=np.float32)
    return (5.0 * s / (1 + 4.0 * s)).astype(np.float32)          # Euler shift 5


def text_info(n, info, frames):
    """Tencent's action offsets for one text of n real tokens -> dict(intra, frame_act, offs, head).

    intra: the text has 2+ timed actions and video sub-windows (per-action mask on); frame_act: action index per latent
    frame; head: tokens of the text kept visible in the next window's history (None = all)."""
    if info is None:
        return dict(intra=False, frame_act=None, offs=None, head=None)
    if info.get('valid_post_crop') not in (None, n):
        raise ValueError('text offsets are for %s tokens, the text has %d' % (info.get('valid_post_crop'), n))
    offs, win, a0 = info.get('offs'), info.get('win'), info.get('action_start', -1)
    if offs is not None and len(offs) >= 1:
        head = min(int(offs[0]), n)
    elif a0 is not None and a0 >= 0:
        head = min(int(a0), n)
    else:
        head = None
    intra = offs is not None and len(offs) >= 2 and win is not None and len(win) > 0
    fa = None
    if intra:
        if offs[-1] != n:
            raise ValueError('last action offset %d != text length %d' % (offs[-1], n))
        fa = np.full(frames, -9, np.int32)
        cur = 0
        for vs, ve, k in win:
            if vs != cur or ve <= vs:
                raise ValueError('action video windows %s do not tile the window' % (win,))
            fa[vs:ve] = k
            cur = ve
        if cur != frames or (fa < -1).any():
            raise ValueError('action video windows %s do not cover %d frames' % (win, frames))
        fa = fa.tolist()
    return dict(intra=intra, frame_act=fa, offs=[int(o) for o in offs] if intra else None, head=head)


class DiTRuntime:
    def __init__(self, devices, params, ops, arch=M.ARCH, steps=None, preset='quality', sink=6, hist=18,
                 buckets=B.BUCKETS, bq=1024, bkv=7168, log=print):
        """params: (top, blocks, acts) f32 numpy from dit.build_params (LoRA already merged)."""
        self.arch, self.ops, self.log, self.sink, self.hist, self.buckets, self.bq, self.bkv = arch, ops, log, sink, hist, tuple(buckets), bq, bkv
        self.devices = list(devices)
        self.mesh = Mesh(np.array(self.devices), ('tp',))
        self.rep = NamedSharding(self.mesh, P())
        t0 = time.time()
        top, blocks, acts = params
        self.bspec, self.aspec = M.param_specs(blocks, acts)
        put = lambda x, spec: jax.device_put(np.asarray(x).astype(jnp.bfloat16), NamedSharding(self.mesh, spec))
        self.blocks = {k: put(v, self.bspec[k]) for k, v in blocks.items()}
        self.acts = {k: put(v, self.aspec[k]) for k, v in acts.items()}
        self.top = {k: put(v, P(*([None] * np.ndim(v)))) for k, v in top.items()}
        jax.block_until_ready((self.top, self.blocks, self.acts))
        self.gib = sum(x.nbytes for d in (self.top, self.blocks, self.acts) for x in d.values()) / 2**30
        self.load_s = round(time.time() - t0, 1)
        self.hsp = {'q': NamedSharding(self.mesh, P(None, None, 'tp', None)), 's': NamedSharding(self.mesh, P(None, None, 'tp'))}
        self._fw, self._exe, self._ex = {}, {}, None
        self.compile_lock = __import__('threading').Lock()
        self.f_cam = jax.jit(lambda t_, e: M.cam_embed(t_, e, jnp.bfloat16))

        def concat_hist(sk, sv, rk, rv):
            cat = lambda a, b: {n: jnp.concatenate([a[n], b[n]], axis=1) for n in ('q', 's')} if a is not None else b
            return cat(sk, rk), cat(sv, rv)
        self.f_hist = jax.jit(concat_hist, out_shardings=(self.hsp, self.hsp))
        self.set_preset(preset, steps)
        log(json.dumps(dict(event='dit_loaded', chips=len(self.devices), gib=round(self.gib, 2), load_s=self.load_s,
                            preset=self.preset, steps=self.steps)))
        self.reset(0)

    # -------------------------------------------------------------- settings
    def set_preset(self, preset, steps=None):
        if preset not in ('quality', 'live'):
            raise ValueError('preset must be quality or live')
        self.preset = preset
        self.steps = int(steps or (3 if preset == 'quality' else 4))      # quality: 3 steps + history pass; live: 4 steps
        self.SIG = sigmas(self.steps, preset)

    # -------------------------------------------------------------- programs
    def forward(self, text_len, hist_len, kind):
        """Compiled forward for (text bucket, history rows, kind): kind 'plain' (velocity only), 'kv' (velocity + the
        recent-frames history), 'sink' (velocity + the sink frames, first window only)."""
        key = (text_len, hist_len, kind)
        if key not in self._fw:
            a = self.arch
            FR, TT = a.frame_tokens, a.frames
            if kind == 'plain':
                kw = dict(want_kv=False)
            elif kind == 'kv':
                kw = dict(want_kv=True, kv_from=(TT - (self.hist - self.sink)) * FR)
            else:
                kw = dict(want_kv=True, kv_from=0, kv_to=self.sink * FR)
            self._fw[key] = jax.jit(M.make_forward(self.mesh, self.bspec, self.aspec, self.ops, a, bq=self.bq, bkv=self.bkv,
                                                   with_hist=hist_len > 0, **kw))
        return self._fw[key]

    def call(self, text_len, hist_len, kind, *args):
        exe = self._exe.get((text_len, hist_len, kind))
        if exe is not None:
            try:
                return exe(*args)
            except (TypeError, ValueError) as e:          # argument layout differs from the compiled one: use the jit
                self.log(json.dumps(dict(event='compiled_call_fallback', err=repr(e)[:200])))
                self._exe.pop((text_len, hist_len, kind), None)
        return self.forward(text_len, hist_len, kind)(*args)

    def compile(self, text_len, hist_len, kinds=None):
        """Compile (or load from the compile cache) the programs for one (text bucket, history rows) shape."""
        kinds = kinds or (('plain', 'kv', 'sink') if hist_len == 0 else ('plain', 'kv'))
        with self.compile_lock:
            todo = [k for k in kinds if (text_len, hist_len, k) not in self._exe]
            if not todo:
                return
            args = self._abstract_args(text_len, hist_len)
            for kind in todo:
                self._exe[(text_len, hist_len, kind)] = self.forward(text_len, hist_len, kind).lower(*args).compile()

    def shapes(self):
        """Every (text bucket, history rows) pair a session can use."""
        FR = self.arch.frame_tokens
        out = [(b, 0) for b in self.buckets]
        out += [(b, self.hist * FR + p) for b in self.buckets for p in self.buckets]
        return out

    def _abstract_args(self, text_len, hist_len):
        a, SDS = self.arch, jax.ShapeDtypeStruct
        LQ, LK, nv = B.layout(a.tokens, text_len, self.hist * a.frame_tokens if hist_len else 0, hist_len - self.hist * a.frame_tokens if hist_len else 0,
                              block_q=self.bq, block_kv=self.bkv)
        hk = {'q': SDS((a.layers, hist_len, a.heads, a.head_dim), np.int8, sharding=self.hsp['q']),
              's': SDS((a.layers, hist_len, a.heads), np.float32, sharding=self.hsp['s'])} if hist_len else None
        if self._ex is None:                    # example camera tokens and tables: the exact layouts the session passes
            self._ex = (self.cam_tokens(np.tile(np.eye(4, dtype=np.float32), (4 * a.frames, 1, 1))), self.tables(1))
        return (self.top, self.blocks, self.acts, SDS((a.latent_ch, a.frames, a.height, a.width), np.float32, sharding=self.rep),
                jnp.float32(500.0), SDS((text_len, a.text_dim), np.float32, sharding=self.rep), SDS((text_len,), np.bool_, sharding=self.rep),
                self._ex[0], self._ex[1], hk, hk, SDS((LQ, 1), np.int32, sharding=self.rep), SDS((1, LK), np.int32, sharding=self.rep))

    def compile_all(self, log=None):
        """Compile (or load from the compile cache) every program a session can use. Returns seconds per shape."""
        out = {}
        for tl, hl in self.shapes():
            t0 = time.time()
            self.compile(tl, hl)
            out['%d/%d' % (tl, hl)] = round(time.time() - t0, 1)
            if log:
                log(json.dumps(dict(event='compiled', text=tl, hist=hl, s=out['%d/%d' % (tl, hl)])))
        return out

    # -------------------------------------------------------------- state
    def put(self, x):
        return jax.device_put(x, self.rep)

    def reset(self, seed):
        a = self.arch
        self.rng = np.random.RandomState(seed)
        self.w = 0
        self.sinkkv = None
        self.hk = self.hv = None                 # no history before the first window
        self.prev = None                         # (text bucket, real tokens, tokens kept visible) of the text in the history

    def make_item(self, key, text, mask, info):
        """Encoder output (text [N, text_dim] f32, mask [N] bool, Tencent offsets) -> item on the chips."""
        m = np.asarray(mask, bool)
        n = int(m.sum())
        if n == 0 or not m[:n].all() or m[n:].any():
            raise ValueError('valid text tokens must be a non-empty prefix')
        tb = B.bucket(n, self.buckets)
        t = B.pad_rows(np.asarray(text, np.float32)[:n], tb)
        tm = np.zeros(tb, bool)
        tm[:n] = True
        it = dict(key=key, n=n, bucket=tb, text=self.put(t), tmask=self.put(tm), **text_info(n, info, self.arch.frames))
        return it

    def tables(self, w):
        return {k: self.put(v) for k, v in M.seg_tables(self.arch, t0=w * self.arch.frames).items()}

    def cam_tokens(self, extr_block):
        """Normalized extrinsics of this window [117 or 120, 4, 4] -> camera tokens [frames, 64] on the chips."""
        return self.f_cam(self.top, self.put(np.asarray(extr_block, np.float32)))

    def groups(self, it):
        a = self.arch
        hist_img = self.hist * a.frame_tokens if self.prev else 0
        pb, n_prev, keep = self.prev if self.prev else (0, 0, 0)
        LQ, LK, nv = B.layout(a.tokens, it['bucket'], hist_img, pb, block_q=self.bq, block_kv=self.bkv)
        gq, gk = B.groups(a.tokens, a.frame_tokens, it['frame_act'] if it['intra'] else None, it['n'],
                          it['offs'] if it['intra'] else None, it['bucket'], LQ, LK, hist_img, n_prev, keep, pb)
        return self.put(gq), self.put(gk)

    # -------------------------------------------------------------- one window
    def window(self, cam, tabs, it, last=False):
        """Generate the next window with text item ``it``. Returns (x0 latents f32 on the chips [32, 30, 22, 40], timing)."""
        a = self.arch
        w = self.w
        T = {}
        n0 = self.rng.randn(a.latent_ch, a.frames, a.height, a.width).astype(np.float32)
        n1 = self.rng.randn(a.latent_ch, a.frames, a.height, a.width).astype(np.float32)
        tb, hl = it['bucket'], (int(self.hk['q'].shape[1]) if self.hk is not None else 0)
        gq, gk = self.groups(it)
        f_plain = lambda *a: self.call(tb, hl, 'plain', *a)
        f_kv = lambda *a: self.call(tb, hl, 'kv', *a)
        f_sink = lambda *a: self.call(tb, hl, 'sink', *a)
        t0 = time.time()
        x = self.put(n0)
        hk, hv = self.hk, self.hv
        fold = self.preset == 'live' and not last
        r = None
        S = self.SIG
        for i in range(self.steps):
            args = (self.top, self.blocks, self.acts, x, jnp.float32(S[i] * 1000), it['text'], it['tmask'], cam, tabs, hk, hv, gq, gk)
            if fold and i == self.steps - 1:
                if w == 0:
                    self.sinkkv = f_sink(*args)[1]
                v, r = f_kv(*args)
            else:
                v = f_plain(*args)[0]
            x = x.astype(jnp.float32) + v.astype(jnp.float32) * (S[i + 1] - S[i])
        x.block_until_ready()
        T['steps_s'] = round(time.time() - t0, 4)
        if not last:
            if not fold:
                cin = ((1 - HISTORY_SIGMA) * x + HISTORY_SIGMA * self.put(n1)).astype(jnp.float32)
                args = (self.top, self.blocks, self.acts, cin, jnp.float32(206.897), it['text'], it['tmask'], cam, tabs, hk, hv, gq, gk)
                if w == 0:
                    self.sinkkv = f_sink(*args)[1]
                r = f_kv(*args)[1]
            args = hk = hv = None
            self.hk = self.hv = None
            sk, sv = self.sinkkv
            self.hk, self.hv = self.f_hist(sk, sv, r[0], r[1])
            jax.block_until_ready((self.hk, self.hv))
            keep = it['head'] if (it['head'] is not None and it['head'] < it['n']) else it['n']
            self.prev = (it['bucket'], it['n'], keep)
        T['end_s'] = round(time.time() - t0, 4)
        T['text'] = it['key']
        T['action_mask'] = bool(it['intra'])
        self.w += 1
        return x, T

    def hist_after(self, it):
        return self.hist * self.arch.frame_tokens + it['bucket']
