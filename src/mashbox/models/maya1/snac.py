"""SNAC 24 kHz decoder (codes -> 24 kHz mono audio) in plain JAX, and maya1's 7-slot code unpack.

The decoder follows snac 1.2.1 (github.com/hubertsiuzdak/snac at 8f79a718; its MIT licence is LICENSE-SNAC next to this
file, unchanged): vq.py from_codes, layers.py Decoder, ResidualUnit, NoiseBlock, DecoderBlock and snake. The weights are
hubertsiuzdak/snac_24khz at d73ad176, with weight norm folded in, as the snac_24khz_decoder.npz arrays in
mashbox-labs/maya1-TPU. The unpack follows maya-research/maya1's example (7 tokens per frame: level1 = slot 0,
level2 = slots 1 and 4, level3 = slots 2, 3, 5 and 6).

Layout: [batch, channels, time] everywhere, as in torch.

Precision (`cd`):
  'f32'   float32 operands at HIGHEST precision, float32 everywhere (the default).
  'bf16'  conv operands in bfloat16 with float32 accumulation; snake, noise, residual adds and tanh in float32.

Lengths: every decode runs on a fixed window of Fw frames with a valid frame range [lo, hi). Each conv output is
multiplied by the validity mask at its own resolution, so positions outside [lo, hi) are exactly zero at the input of
every conv, as the zero padding SNAC sees at the true edges of a take. So a take decodes in a padded shape bucket (one
compile per bucket, not per length), and streamed windows of one fixed shape give the same audio as the whole-take
decode, as long as each emitted frame is at least the receptive field away from any window edge that is not a take edge
(context_frames()).

Noise: SNAC's NoiseBlocks add random noise. KeyNoise(seed) draws it per (block, frame) from the seed, so one seed gives
the same audio for a whole-take decode and a streamed decode of any chunking.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# SNAC 24 kHz constants (snac_24khz config.json) and maya1's code tokens
SR = 24000
CODE_START = 128257          # SOS
CODE_END = 128258            # EOS (stop token)
OFF = 128266                 # first SNAC code token
N_SLOTS = 7
CB_SIZE = 4096
LATENT = 768                 # encoder_dim 48 * 2**4
DEC_DIM = 1024
DEC_RATES = (8, 8, 4, 2)
VQ_STRIDES = (4, 2, 1)
DILATIONS = (1, 3, 9)
HOP = 512                    # prod(encoder_rates)
LAT_PER_FRAME = 4            # one maya1 frame = 1 level1 + 2 level2 + 4 level3 codes = 4 latent steps (level3 at 1 latent step per code)
FRAME = LAT_PER_FRAME * HOP  # 2048 samples = 85.33 ms
MAX_NEW_TOKENS = 2048
MAX_FRAMES = MAX_NEW_TOKENS // N_SLOTS   # 292
# time steps per frame at the output of each NoiseBlock (block k output resolution)
NOISE_PER_FRAME = tuple(LAT_PER_FRAME * int(np.prod(DEC_RATES[:k + 1])) for k in range(4))   # (32, 256, 1024, 2048)
# whole-take shape buckets (frames). One compile per (bucket, cd).
BUCKETS = (8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, MAX_FRAMES)


# ============================================================================ token unpack (host side, numpy)
def extract_snac_tokens(token_ids):
    """Generated ids -> SNAC code tokens: cut at the first 128258 and keep only ids in [128266, 128266 + 7*4096), as
    maya1's example does."""
    ids = [int(t) for t in token_ids]
    if CODE_END in ids:
        ids = ids[:ids.index(CODE_END)]
    return [t for t in ids if OFF <= t < OFF + N_SLOTS * CB_SIZE]


def frames_from_tokens(token_ids):
    """-> int32 [n_frames, 7] of codes (t - 128266) % 4096, whole frames only (trailing partial frame dropped,
    as maya1's example)."""
    c = extract_snac_tokens(token_ids)
    n = len(c) // N_SLOTS
    a = np.asarray(c[:n * N_SLOTS], np.int64).reshape(n, N_SLOTS)
    return ((a - OFF) % CB_SIZE).astype(np.int32)


def levels_from_frames(q):
    """[n, 7] slot codes -> (level1 [n], level2 [2n], level3 [4n]) as maya1's unpack:
    slot0 -> level1[i]; slot1, slot4 -> level2[2i], level2[2i+1]; slot2, slot3, slot5, slot6 -> level3[4i..4i+3]."""
    q = np.asarray(q, np.int32).reshape(-1, N_SLOTS)
    return q[:, 0].copy(), q[:, [1, 4]].reshape(-1), q[:, [2, 3, 5, 6]].reshape(-1)


def unpack(token_ids):
    """Generated ids -> (level1, level2, level3) int32 numpy arrays (maya1's 7-slot unpack)."""
    return levels_from_frames(frames_from_tokens(token_ids))


def slot_layout_report(token_ids):
    """Diagnostic only (decoding never depends on it): how many SNAC tokens sit outside the INFERRED per-slot
    sub-range 128266 + 4096*j .. +4095 for their slot j. maya1's example does not check this."""
    c = extract_snac_tokens(token_ids)
    n = len(c) // N_SLOTS
    a = np.asarray(c[:n * N_SLOTS], np.int64).reshape(n, N_SLOTS)
    slot_of_id = (a - OFF) // CB_SIZE
    off = slot_of_id != np.arange(N_SLOTS)[None, :]
    return {'frames': int(n), 'tokens': int(n * N_SLOTS), 'off_slot': int(off.sum()),
            'off_slot_per_slot': off.sum(0).astype(int).tolist(), 'dropped_partial_tail': int(len(c) - n * N_SLOTS)}


# ============================================================================ params
def load_params(npz_path_or_dict, cd='f32'):
    """The weight-norm-folded torch-layout arrays (snac_24khz_decoder.npz) -> JAX param tree. Conv weights cast to the variant's operand
    dtype; biases, snake alphas and codebooks stay f32."""
    sd = dict(np.load(npz_path_or_dict)) if isinstance(npz_path_or_dict, str) else dict(npz_path_or_dict)
    wdt = jnp.float32 if cd == 'f32' else jnp.bfloat16
    F = lambda k: jnp.asarray(np.asarray(sd[k], np.float32))
    W = lambda k: jnp.asarray(np.asarray(sd[k], np.float32), wdt)

    def wT(k):  # ConvTranspose1d weight (Cin, Cout, K) -> equivalent forward-conv weight (Cout, Cin, K), flipped
        w = np.asarray(sd[k], np.float32)
        return jnp.asarray(np.ascontiguousarray(np.flip(w, -1).transpose(1, 0, 2)), wdt)

    p = {'q': [], 'blocks': []}
    for i in range(len(VQ_STRIDES)):
        pre = f'quantizer.quantizers.{i}.'
        p['q'].append({'cb': F(pre + 'codebook.weight'), 'w': W(pre + 'out_proj.weight'), 'b': F(pre + 'out_proj.bias')})
    p['in_dw'] = {'w': W('decoder.model.0.weight'), 'b': F('decoder.model.0.bias')}
    p['in_pw'] = {'w': W('decoder.model.1.weight'), 'b': F('decoder.model.1.bias')}
    for k in range(len(DEC_RATES)):
        pre = f'decoder.model.{2 + k}.block.'
        blk = {'alpha': F(pre + '0.alpha'), 'up_w': wT(pre + '1.weight'), 'up_b': F(pre + '1.bias'),
               'noise_w': W(pre + '2.linear.weight'), 'res': []}
        for r in range(len(DILATIONS)):
            rp = f'{pre}{3 + r}.block.'
            blk['res'].append({'a0': F(rp + '0.alpha'), 'w1': W(rp + '1.weight'), 'b1': F(rp + '1.bias'),
                               'a1': F(rp + '2.alpha'), 'w2': W(rp + '3.weight'), 'b2': F(rp + '3.bias')})
        p['blocks'].append(blk)
    p['out_alpha'] = F('decoder.model.6.alpha')
    p['out'] = {'w': W('decoder.model.7.weight'), 'b': F('decoder.model.7.bias')}
    return p


# ============================================================================ ops
_DN = ('NCH', 'OIH', 'NCH')


def _conv(x, w, b, cd, *, pad=0, dil=1, groups=1, lhs_dil=1, padding=None):
    prec = lax.Precision.HIGHEST if cd == 'f32' else lax.Precision.DEFAULT
    y = lax.conv_general_dilated(x.astype(w.dtype), w, window_strides=(1,),
                                 padding=padding if padding is not None else [(pad, pad)],
                                 lhs_dilation=(lhs_dil,), rhs_dilation=(dil,), dimension_numbers=_DN,
                                 feature_group_count=groups, precision=prec, preferred_element_type=jnp.float32)
    return y if b is None else y + b[None, :, None]


def _snake(x, alpha):
    # layers.py:169-175: x + (alpha + 1e-9).reciprocal() * sin(alpha * x).pow(2)
    return x + jnp.reciprocal(alpha + 1e-9) * jnp.square(jnp.sin(alpha * x))


def _conv_transpose(x, w_eq, b, s, cd):
    # torch ConvTranspose1d(k=2s, step s, padding=ceil(s/2), output_padding=s%2) as a forward conv on the
    # s-dilated input with the flipped/transposed kernel: padding (K-1-p, K-1-p+output_padding). Output length = T*s.
    K = w_eq.shape[-1]
    p = math.ceil(s / 2)
    op = s % 2
    return _conv(x, w_eq, b, cd, lhs_dil=s, padding=[(K - 1 - p, K - 1 - p + op)])


def _mask(T, per_frame, lo, hi):
    t = jnp.arange(T, dtype=jnp.int32)
    return ((t >= lo * per_frame) & (t < hi * per_frame)).astype(jnp.float32)[None, None, :]


def _decode_core(p, l1, l2, l3, n0, n1, n2, n3, lo, hi, cd):
    """l1 [B, Fw], l2 [B, 2Fw], l3 [B, 4Fw] int32; nk [B, 1, Fw*NOISE_PER_FRAME[k]] f32; lo/hi int32 scalars
    (valid frame range inside the window). -> audio [B, Fw*2048] f32 (zero outside [lo, hi))."""
    Fw = l1.shape[1]
    T = Fw * LAT_PER_FRAME
    m = _mask(T, LAT_PER_FRAME, lo, hi)
    # vq.py:91-98 from_codes: per level codebook lookup -> out_proj (1x1, 8 -> 768) -> repeat_interleave(the level rate); sum
    z = 0.0
    for q, codes, s in zip(p['q'], (l1, l2, l3), VQ_STRIDES):
        e = jnp.take(q['cb'], codes, axis=0).transpose(0, 2, 1)          # [B, 8, T/s]
        zi = _conv(e, q['w'], q['b'], cd)                                 # [B, 768, T/s]
        z = z + jnp.repeat(zi, s, axis=-1)
    x = z * m
    # layers.py:47-50 depthwise input: k7 depthwise conv, then 1x1 768 -> 1024 (no attention: attn_window_size null)
    x = _conv(x, p['in_dw']['w'], p['in_dw']['b'], cd, pad=3, groups=LATENT) * m
    x = _conv(x, p['in_pw']['w'], p['in_pw']['b'], cd) * m
    per = LAT_PER_FRAME
    for k, (blk, s, nz) in enumerate(zip(p['blocks'], DEC_RATES, (n0, n1, n2, n3))):
        C = DEC_DIM // 2 ** (k + 1)
        per *= s
        x = _snake(x, blk['alpha'])
        x = _conv_transpose(x, blk['up_w'], blk['up_b'], s, cd)
        m = _mask(x.shape[-1], per, lo, hi)
        x = x * m
        # NoiseBlock (layers.py:118-129): x + noise * linear(x), linear = 1x1 no bias; noise [B, 1, T] shared over C
        x = x + nz.astype(jnp.float32) * _conv(x, blk['noise_w'], None, cd)
        for r, d in zip(blk['res'], DILATIONS):                           # ResidualUnit (layers.py:77-93), kernel 7
            y = _snake(x, r['a0'])
            y = _conv(y, r['w1'], r['b1'], cd, pad=3 * d, dil=d, groups=C) * m
            y = _snake(y, r['a1'])
            y = _conv(y, r['w2'], r['b2'], cd) * m
            x = x + y
    x = _snake(x, p['out_alpha'])
    x = _conv(x, p['out']['w'], p['out']['b'], cd, pad=3) * m
    return jnp.tanh(x)[:, 0, :]


_decode_jit = jax.jit(_decode_core, static_argnames=('cd',))


def decode_window(p, l1, l2, l3, noises, lo, hi, cd='f32'):
    """Jitted window decode. noises: 4 arrays [B, 1, Fw*NOISE_PER_FRAME[k]]."""
    return _decode_jit(p, jnp.asarray(l1, jnp.int32), jnp.asarray(l2, jnp.int32), jnp.asarray(l3, jnp.int32),
                       *[jnp.asarray(n, jnp.float32) for n in noises], jnp.int32(lo), jnp.int32(hi), cd=cd)


# ============================================================================ noise providers
class FixedNoise:
    """Position-indexed noise from fixed arrays (tests): block k sample t of the take uses arr[k][t]. Positions
    outside the arrays (left of the take start) are zero; they are masked anyway."""

    def __init__(self, arrays):
        self.a = [np.asarray(a, np.float32).reshape(-1) for a in arrays]

    def window(self, f_start, n_frames):
        out = []
        for a, pf in zip(self.a, NOISE_PER_FRAME):
            s, e = f_start * pf, (f_start + n_frames) * pf
            w = np.zeros(e - s, np.float32)
            cs, ce = max(s, 0), min(e, a.shape[0])
            if ce > cs:
                w[cs - s:ce - s] = a[cs:ce]
            out.append(w[None, None, :])
        return out


_NOISE_FN = {}      # (pf, n_frames) -> jitted noise program, shared by every seed and every KeyNoise


def _noise_fn(pf, n_frames):
    """One program per (block resolution, window length) for every seed: the block key is a traced argument, so a new
    seed compiles nothing. Per frame: fold_in(block key, frame), then normal."""
    if (pf, n_frames) not in _NOISE_FN:
        def f(kk, f0):
            idx = jnp.maximum(f0 + jnp.arange(n_frames, dtype=jnp.int32), 0).astype(jnp.uint32)
            v = jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(kk, i), (pf,), jnp.float32))(idx)
            return v.reshape(1, 1, n_frames * pf)
        _NOISE_FN[(pf, n_frames)] = jax.jit(f)
    return _NOISE_FN[(pf, n_frames)]


class KeyNoise:
    """Seeded noise: N(0,1) per (block, frame) from fold_in(fold_in(key, block), frame), so the same seed gives the
    same audio for whole-take and streamed decodes of any chunking, and the take length need not be known."""

    def __init__(self, seed):
        self.key = jax.random.PRNGKey(int(seed))
        self._kk = {}

    def _block_key(self, k):
        if k not in self._kk:
            self._kk[k] = jax.random.fold_in(self.key, k)
        return self._kk[k]

    def window(self, f_start, n_frames):
        return [_noise_fn(pf, n_frames)(self._block_key(k), jnp.int32(f_start)) for k, pf in enumerate(NOISE_PER_FRAME)]


# ============================================================================ receptive field
def context_frames():
    """Exact dependency reach of one output frame onto latent frames, computed on the actual layer list with
    concrete frame-aligned indices (shift-invariant at frame alignment). -> (left_frames, right_frames)."""
    a = 1000
    lo, hi = a * FRAME, (a + 1) * FRAME - 1
    lo, hi = lo - 3, hi + 3                                   # output conv k7 pad3
    for s in reversed(DEC_RATES):
        for d in reversed(DILATIONS):                         # residual units: k7 dil d pad 3d (+ 1x1)
            lo, hi = lo - 3 * d, hi + 3 * d
        K, p = 2 * s, math.ceil(s / 2)                        # conv transpose: o = i*s - p + j, j in [0, K)
        lo, hi = -((-(lo + p - K + 1)) // s), (hi + p) // s
    lo, hi = lo - 3, hi + 3                                   # input depthwise k7 pad3
    L = a * LAT_PER_FRAME - lo
    R = hi - ((a + 1) * LAT_PER_FRAME - 1)
    return -(-L // LAT_PER_FRAME), -(-R // LAT_PER_FRAME)


# ============================================================================ whole take
def bucket_for(n):
    for b in BUCKETS:
        if n <= b:
            return b
    raise ValueError(f'{n} frames > MAX_FRAMES {MAX_FRAMES}')


def _window_levels(q, ws, Fw):
    """Rows [ws, ws+Fw) of the [n, 7] frame table; frames outside [0, n) get code 0 (masked anyway)."""
    w = np.zeros((Fw, N_SLOTS), np.int32)
    n = q.shape[0]
    cs, ce = max(ws, 0), min(ws + Fw, n)
    if ce > cs:
        w[cs - ws:ce - ws] = q[cs:ce]
    l1, l2, l3 = levels_from_frames(w)
    return l1[None], l2[None], l3[None]


def decode_frames(p, q, noise, cd='f32', bucket=True):
    """Whole-take decode of a [n, 7] frame table -> np.float32 [n*2048]. bucket=False decodes at the exact length."""
    q = np.asarray(q, np.int32).reshape(-1, N_SLOTS)
    n = q.shape[0]
    if n == 0:
        return np.zeros(0, np.float32)
    Fw = bucket_for(n) if bucket else n
    l1, l2, l3 = _window_levels(q, 0, Fw)
    y = decode_window(p, l1, l2, l3, noise.window(0, Fw), 0, n, cd)
    return np.asarray(y, np.float32)[0, :n * FRAME]     # host slice: no device program per take length


def decode_tokens(p, token_ids, noise, cd='f32', bucket=True):
    return decode_frames(p, frames_from_tokens(token_ids), noise, cd, bucket)


# ============================================================================ streaming
class SnacStream:
    """Incremental decoder fed with generated token ids as they arrive. Emits frames [a, b) once frames up to
    b + CR are known (or at end of take), decoding one fixed window shape [a - CL, a - CL + Fw) with the validity
    mask covering only known in-take frames. Same length as the whole-take decode (no frame dropped, no crossfade); the
    samples match it to within 1e-6, not bit for bit."""

    def __init__(self, p, noise, cd='f32', chunk_frames=8, first_chunk_frames=None, ctx=None):
        self.p, self.noise, self.cd = p, noise, cd
        self.CL, self.CR = ctx if ctx is not None else context_frames()
        self.F = int(chunk_frames)
        self.F1 = int(first_chunk_frames or chunk_frames)
        assert 1 <= self.F1 <= self.F
        self.Fw = self.CL + self.F + self.CR
        self.codes = []          # flat SNAC code tokens seen so far (after filtering)
        self.ended = False
        self.done = 0            # frames emitted

    def _frames(self):
        n = len(self.codes) // N_SLOTS
        a = np.asarray(self.codes[:n * N_SLOTS], np.int64).reshape(n, N_SLOTS)
        return ((a - OFF) % CB_SIZE).astype(np.int32)

    def _emit(self, q, a, b, n_known):
        ws = a - self.CL
        l1, l2, l3 = _window_levels(q, ws, self.Fw)
        lo, hi = max(0, -ws), min(self.Fw, n_known - ws)
        y = decode_window(self.p, l1, l2, l3, self.noise.window(ws, self.Fw), lo, hi, self.cd)
        return np.asarray(y, np.float32)[0, self.CL * FRAME:(self.CL + b - a) * FRAME]   # host slice: no program per chunk length

    def push(self, token_ids):
        """Feed newly generated ids. Returns a list of audio chunks (np.float32) ready to send."""
        for t in token_ids:
            t = int(t)
            if self.ended:
                break
            if t == CODE_END:
                self.ended = True
            elif OFF <= t < OFF + N_SLOTS * CB_SIZE:
                self.codes.append(t)
        out = []
        q = self._frames()
        n = q.shape[0]
        while True:
            want = self.F1 if self.done == 0 else self.F
            if self.done + want + self.CR > n:
                break
            out.append(self._emit(q, self.done, self.done + want, n))
            self.done += want
        if self.ended:
            out += self.finish()
        return out

    def finish(self):
        """End of take (EOS or max tokens): flush the rest with the true right edge."""
        self.ended = True
        q = self._frames()
        n = q.shape[0]
        out = []
        while self.done < n:
            b = min(self.done + self.F, n)
            out.append(self._emit(q, self.done, b, n))
            self.done = b
        return out
