"""Hy-Nomas main model (the distilled DiT with the camera action module) in JAX, split over chips by attention heads.

One call runs one denoising pass over one 5 s window: 30 latent frames of 22 x 40 tokens, the window's text, its camera
tokens and the history (keys and values kept from earlier windows, stored as int8 with one scale per token and head).
It follows Tencent's ``forward_with_cache`` for one segment: image embed, timestep embed, text refiner (2 blocks), 54
double-stream blocks each followed by the camera action module, final layer.

Layout per chip (heads split 4 ways on a v6e-4): queries are [image | text | zero rows], keys and values are
[image | text | history | zero rows]. Text is padded to a fixed length (buckets.py) and the padded rows are zeroed, so
every text of the same bucket reuses the same compiled program.
"""
import json
import math
import struct
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map as _sm

    def smap(f, mesh, i, o):
        return _sm(f, mesh=mesh, in_specs=i, out_specs=o, check_vma=False)
except ImportError:                                    # older JAX
    from jax.experimental.shard_map import shard_map as _sm

    def smap(f, mesh, i, o):
        return _sm(f, mesh=mesh, in_specs=i, out_specs=o, check_rep=False)

from .ops import q_scale

F32 = jnp.float32


@dataclass(frozen=True)
class Arch:
    hidden: int = 2048
    heads: int = 16
    head_dim: int = 128
    ff: int = 8192
    layers: int = 54
    latent_ch: int = 32
    text_dim: int = 3584
    cam_heads: int = 16
    cam_head_dim: int = 64
    frames: int = 30                     # latent frames per window
    height: int = 22                     # latent tokens per frame, vertical
    width: int = 40
    rope_dims: tuple = (16, 56, 56)
    cam_rope_dims: tuple = (8, 28, 28)
    theta: float = 256.0

    @property
    def frame_tokens(self):
        return self.height * self.width

    @property
    def tokens(self):
        return self.frames * self.frame_tokens


ARCH = Arch()


# ------------------------------------------------------------------ safetensors (no torch)
def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return h, 8 + n


def st_read(path, want=None):
    """{name: f32 numpy array} for every tensor (or those where want(name) is true). BF16/F16 are widened to f32."""
    h, off = st_header(path)
    mm = np.memmap(path, dtype=np.uint8, mode='r', offset=off)
    out = {}
    for k, v in h.items():
        if want is not None and not want(k):
            continue
        s, e = v['data_offsets']
        b = mm[s:e]
        dt = v['dtype']
        if dt == 'F32':
            a = b.view(np.float32)
        elif dt == 'BF16':
            a = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == 'F16':
            a = b.view(np.float16).astype(np.float32)
        else:
            raise ValueError('%s: dtype %s' % (k, dt))
        out[k] = a.reshape(v['shape'])
    return out


# ------------------------------------------------------------------ parameters
BLK_LIN = ('mod', 'q', 'k', 'v', 'proj', 'fc1', 'fc2')
BLK_SRC = {'mod': 'mod.linear', 'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'proj': 'attn_proj', 'fc1': 'mlp.fc1', 'fc2': 'mlp.fc2'}
# tensor-parallel split of each linear's [in, out] weight: 'col' splits the outputs, 'row' splits the inputs (+ sum)
BLK_TP = {'mod': 'col', 'q': 'col', 'k': 'col', 'v': 'col', 'proj': 'row', 'fc1': 'col', 'fc2': 'row'}
ACT_TP = {'aq': 'col', 'ak': 'col', 'av': 'col', 'aproj': 'row'}


def build_params(sd, arch=ARCH):
    """Tencent's DiT state dict (f32 numpy, LoRA already merged) -> (top, blocks, acts) as f32 numpy; blocks and acts are
    stacked over layers with every linear stored [in, out]."""
    NL = arch.layers
    blocks = {}
    for s in ('img', 'txt'):
        for nm in BLK_LIN:
            key = lambda i, s=s, nm=nm: 'double_blocks.%d.%s_%s' % (i, s, BLK_SRC[nm])
            blocks['%s_%s_w' % (s, nm)] = np.stack([sd[key(i) + '.weight'].T for i in range(NL)])
            blocks['%s_%s_b' % (s, nm)] = np.stack([sd[key(i) + '.bias'] for i in range(NL)])
        blocks[s + '_qn'] = np.stack([sd['double_blocks.%d.%s_attn_q_norm.weight' % (i, s)] for i in range(NL)])
        blocks[s + '_kn'] = np.stack([sd['double_blocks.%d.%s_attn_k_norm.weight' % (i, s)] for i in range(NL)])
    ckh = arch.cam_heads * arch.cam_head_dim
    a = lambda i, n: sd['action_modules.%d.%s' % (i, n)]
    acts = {'aq_w': np.stack([a(i, 'img_attn_q.weight').T for i in range(NL)])}
    kv = [a(i, 'keyboard_attn_kv.weight').T for i in range(NL)]
    acts['ak_w'] = np.stack([x[:, :ckh] for x in kv])
    acts['av_w'] = np.stack([x[:, ckh:] for x in kv])
    acts['aproj_w'] = np.stack([a(i, 'proj_keyboard.weight').T for i in range(NL)])
    acts['aqn'] = np.stack([a(i, 'key_attn_q_norm.weight') for i in range(NL)])
    acts['akn'] = np.stack([a(i, 'key_attn_k_norm.weight') for i in range(NL)])
    for j in (0, 2):
        acts['kb%d_w' % j] = np.stack([a(i, 'keyboard_embed.%d.weight' % j).T for i in range(NL)])
        acts['kb%d_b' % j] = np.stack([a(i, 'keyboard_embed.%d.bias' % j) for i in range(NL)])
    t = {}
    L = lambda n: sd[n + '.weight'].T
    B = lambda n: sd[n + '.bias']
    t['img_in_w'] = sd['img_in.proj.weight'].reshape(arch.hidden, arch.latent_ch).T
    t['img_in_b'] = B('img_in.proj')
    for p, n in (('time', 'time_in.mlp'), ('rt', 'txt_in.t_embedder.mlp')):
        t[p + '0_w'], t[p + '0_b'], t[p + '2_w'], t[p + '2_b'] = L(n + '.0'), B(n + '.0'), L(n + '.2'), B(n + '.2')
    t['rin_w'], t['rin_b'] = L('txt_in.input_embedder'), B('txt_in.input_embedder')
    t['rc1_w'], t['rc1_b'] = L('txt_in.c_embedder.linear_1'), B('txt_in.c_embedder.linear_1')
    t['rc2_w'], t['rc2_b'] = L('txt_in.c_embedder.linear_2'), B('txt_in.c_embedder.linear_2')
    for j in range(2):
        n = 'txt_in.individual_token_refiner.blocks.%d' % j
        for nm, src in (('n1', 'norm1'), ('n2', 'norm2')):
            t['r%d_%s_w' % (j, nm)], t['r%d_%s_b' % (j, nm)] = sd[n + '.%s.weight' % src], sd[n + '.%s.bias' % src]
        for nm, src in (('qkv', 'self_attn_qkv'), ('proj', 'self_attn_proj'), ('fc1', 'mlp.fc1'), ('fc2', 'mlp.fc2'),
                        ('ada', 'adaLN_modulation.1')):
            t['r%d_%s_w' % (j, nm)], t['r%d_%s_b' % (j, nm)] = L(n + '.' + src), B(n + '.' + src)
    t['fin_ada_w'], t['fin_ada_b'] = L('final_layer.adaLN_modulation.1'), B('final_layer.adaLN_modulation.1')
    t['fin_w'], t['fin_b'] = L('final_layer.linear'), B('final_layer.linear')
    for nm, src in (('ext', 'extrinsics_embedding_camctrl'), ('c1', 'c2ws_hidden_states_layer1'), ('c2', 'c2ws_hidden_states_layer2')):
        t[nm + '_w'], t[nm + '_b'] = L(src), B(src)
    return t, blocks, acts


def param_specs(blocks, acts):
    """PartitionSpecs over mesh axis 'tp' for the stacked tensors (leading axis = layer)."""
    bs = {}
    for k in blocks:
        parts = k.split('_')
        kind = parts[-1]
        if kind in ('qn', 'kn'):
            bs[k] = P(None, None)
        elif kind == 'b':
            bs[k] = P(None, 'tp') if BLK_TP[parts[1]] == 'col' else P(None, None)
        else:
            bs[k] = P(None, None, 'tp') if BLK_TP[parts[1]] == 'col' else P(None, 'tp', None)
    as_ = {}
    for k, v in acts.items():
        nm = k.rsplit('_', 1)[0]
        if nm in ACT_TP:
            as_[k] = P(None, None, 'tp') if ACT_TP[nm] == 'col' else P(None, 'tp', None)
        else:
            as_[k] = P(*([None] * np.ndim(v)))
    return bs, as_


# ------------------------------------------------------------------ position tables (host, f32)
def rope_nd(dims, start, stop, theta):
    axes = [np.linspace(a, b, b - a + 1, dtype=np.float32)[: b - a] for a, b in zip(start, stop)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), 0)
    cs, ss = [], []
    for i, d in enumerate(dims):
        fr = (np.float32(1.0) / (np.float32(theta) ** (np.arange(0, d, 2, dtype=np.float32)[: d // 2] / np.float32(d)))).astype(np.float32)
        f = np.outer(grid[i].reshape(-1).astype(np.float32), fr).astype(np.float32)
        cs.append(np.repeat(np.cos(f), 2, axis=1))
        ss.append(np.repeat(np.sin(f), 2, axis=1))
    return np.concatenate(cs, 1), np.concatenate(ss, 1)


def seg_tables(arch, t0=0):
    """Rotary tables for one window's image tokens (time positions t0..t0+frames) and the camera module's tables."""
    tt, th, tw = arch.frames, arch.height, arch.width
    cos, sin = rope_nd(arch.rope_dims, (t0, 0, 0), (t0 + tt, th, tw), arch.theta)
    ccos, csin = rope_nd(arch.cam_rope_dims, (0, 0, 0), (7500, 1, 1), arch.theta)
    S = th * tw
    fid = np.minimum(np.arange(tt * S) // S, 7499)
    kid = np.minimum(np.arange(tt), 7499)
    return dict(cos=cos, sin=sin, cq_cos=ccos[fid], cq_sin=csin[fid], ck_cos=ccos[kid], ck_sin=csin[kid],
                fid=(np.arange(tt * S) // S).astype(np.int32))


# ------------------------------------------------------------------ math helpers (torch semantics)
def lin(x, w, b=None):
    y = jnp.dot(x, w.astype(x.dtype), preferred_element_type=F32)
    if b is not None:
        y = y + b.astype(F32)
    return y.astype(x.dtype)


def silu(x):
    return jax.nn.silu(x.astype(F32)).astype(x.dtype)


def ln(x, w=None, b=None, eps=1e-6):
    x32 = x.astype(F32)
    m = x32.mean(-1, keepdims=True)
    v = ((x32 - m) ** 2).mean(-1, keepdims=True)
    y = (x32 - m) * jax.lax.rsqrt(v + eps)
    if w is not None:
        y = y * w.astype(F32) + b.astype(F32)
    return y.astype(x.dtype)


def rms(x, w, eps=1e-6):
    x32 = x.astype(F32)
    return (x32 * jax.lax.rsqrt((x32 * x32).mean(-1, keepdims=True) + eps)).astype(x.dtype) * w.astype(x.dtype)


def rope(x, cos, sin):
    x32 = x.astype(F32)
    xr = x32.reshape(*x.shape[:-1], -1, 2)
    rot = jnp.stack([-xr[..., 1], xr[..., 0]], -1).reshape(x.shape)
    return (x32 * cos[:, None, :] + rot * sin[:, None, :]).astype(x.dtype)


def gelu_tanh(x):
    return jax.nn.gelu(x.astype(F32), approximate=True).astype(x.dtype)


def tstep(t, dim=256, max_period=10000):
    half = dim // 2
    fr = jnp.exp(F32(-math.log(max_period)) * jnp.arange(half, dtype=F32) / half)
    a = t.astype(F32)[..., None] * fr
    return jnp.concatenate([jnp.cos(a), jnp.sin(a)], -1)


def attn_ref(q, k, v, kvmask):
    """softmax(q k^T / sqrt(d)) v over valid keys in f32. q [Lq, h, d], k/v [Lk, h, d], kvmask [Lk] bool."""
    s = jnp.einsum('qhd,khd->hqk', q.astype(F32), k.astype(F32)) / math.sqrt(q.shape[-1])
    s = jnp.where(kvmask[None, None, :], s, -jnp.inf)
    p = jax.nn.softmax(s, -1)
    return jnp.einsum('hqk,khd->qhd', p, v.astype(F32)).astype(q.dtype)


def quant_kv(x):
    """[..., d] -> int8 + f32 scale per row (absmax / 127, round half away from zero)."""
    x32 = x.astype(F32)
    pk = jnp.abs(x32).max(-1)
    sc = jnp.where(pk > 0, pk / 127.0, 1.0)
    y = x32 / sc[..., None]
    r = jnp.where(y >= 0, jnp.floor(y + 0.5), jnp.ceil(y - 0.5))
    return {'q': jnp.clip(r, -127, 127).astype(jnp.int8), 's': sc}


def rup(x, m):
    return (x + m - 1) // m * m


# ------------------------------------------------------------------ the layer stack (runs per chip inside shard_map)
def core_local(blocks, acts, img, txt, tvalid, vec, cam, cos, sin, cq_cos, cq_sin, ck_cos, ck_sin, fid, hk, hv, gq, gk,
               *, ops, arch, want_kv, kv_from, kv_to, dt, bq, bkv):
    HD = arch.head_dim
    hl = blocks['img_q_w'].shape[-1] // HD                 # heads on this chip
    chl = acts['ak_w'].shape[-1] // arch.cam_head_dim
    Li, Lt, Tk = img.shape[0], txt.shape[0], cam.shape[0]
    Lh = hk['q'].shape[1] if hk is not None else 0         # no history (first window): None, never a zero-length array
    Lk = Li + Lt + Lh
    LQ, LK = rup(Li + Lt, bq), rup(Lk, bkv)
    HDl = hl * HD
    c_ = q_scale(HD)
    sv = silu(vec)
    tv = tvalid[:, None]
    dus = jax.lax.dynamic_update_slice

    def layer(carry, xs):
        img, txt = carry
        lp, ap, hkl, hvl = xs
        if Lh:
            hkl = (hkl['q'].astype(F32) * hkl['s'][..., None]).astype(dt).reshape(Lh, HDl)
            hvl = (hvl['q'].astype(F32) * hvl['s'][..., None]).astype(dt).reshape(Lh, HDl)

        def mod(s):
            m = lin(sv[None], lp[s + '_mod_w'], lp[s + '_mod_b'])[0]
            return jnp.split(jax.lax.all_gather(m, 'tp', tiled=True), 6)
        mi, mt = mod('img'), mod('txt')
        xm = ln(img) * (1 + mi[1]) + mi[0]
        vi = lin(xm, lp['img_v_w'], lp['img_v_b'])
        xt = ln(txt) * (1 + mt[1]) + mt[0]
        qt = rms(lin(xt, lp['txt_q_w'], lp['txt_q_b']).reshape(-1, hl, HD), lp['txt_qn'])
        kt = rms(lin(xt, lp['txt_k_w'], lp['txt_k_b']).reshape(-1, hl, HD), lp['txt_kn'])
        vt = lin(xt, lp['txt_v_w'], lp['txt_v_b'])
        # padded text rows are zeros in q, k and v (they are also masked out by the group ids)
        qt2 = jnp.where(tv, (qt.astype(F32) * c_).astype(dt).reshape(Lt, HDl), 0)
        kt2 = jnp.where(tv, kt.reshape(Lt, HDl), 0)
        vt2 = jnp.where(tv, vt, 0)
        Q2 = ops.dit_qk_norm_rope(lin(xm, lp['img_q_w'], lp['img_q_b']), lp['img_qn'], cos, sin, H=hl, scale=c_, out_rows=LQ)
        Q2 = dus(dus(Q2, qt2, (Li, 0)), jnp.zeros((LQ - Li - Lt, HDl), dt), (Li + Lt, 0))
        K2 = ops.dit_qk_norm_rope(lin(xm, lp['img_k_w'], lp['img_k_b']), lp['img_kn'], cos, sin, H=hl, out_rows=LK)
        K2 = dus(K2, kt2, (Li, 0))
        if Lh:
            K2 = dus(K2, hkl, (Li + Lt, 0))
        if LK > Lk:
            K2 = dus(K2, jnp.zeros((LK - Lk, HDl), dt), (Lk, 0))
        V2 = jnp.concatenate([vi, vt2] + ([hvl] if Lh else []) + ([jnp.zeros((LK - Lk, HDl), dt)] if LK > Lk else []))
        o = ops.dit_attention(Q2, K2, V2, gq, gk, H=hl, n_img=Li, nvalid=Lk, bq=bq, bkv=bkv)
        oi, ot = o[:Li], o[Li:Li + Lt]

        def rowlin(x, pn):
            return (jax.lax.psum(jnp.dot(x, lp[pn + '_w'].astype(dt), preferred_element_type=F32), 'tp') + lp[pn + '_b'].astype(F32)).astype(dt)

        def post(x, o, m, s):
            x = x + rowlin(o, s + '_proj') * m[2]
            h = gelu_tanh(lin(ln(x) * (1 + m[4]) + m[3], lp[s + '_fc1_w'], lp[s + '_fc1_b']))
            return x + rowlin(h, s + '_fc2') * m[5]
        img = post(img, oi, mi, 'img')
        txt = post(txt, ot, mt, 'txt')
        # camera action module, one camera token per latent frame: the image tokens of frame j attend only to token j,
        # so the attention output is that token's value (softmax over one key is exactly 1)
        kb = lin(silu(lin(cam, ap['kb0_w'], ap['kb0_b'])), ap['kb2_w'], ap['kb2_b'])
        cv = lin(kb, ap['av_w']).reshape(Tk, chl, arch.cam_head_dim)
        co = cv[fid]
        img = img + jax.lax.psum(jnp.dot(co.reshape(Li, chl * arch.cam_head_dim), ap['aproj_w'].astype(dt),
                                         preferred_element_type=F32), 'tp').astype(dt)
        kvo = None
        if want_kv:
            kc = K2[:Li + Lt].reshape(Li + Lt, hl, HD)
            vc = jnp.concatenate([vi, vt2]).reshape(Li + Lt, hl, HD)
            kvo = (quant_kv(kc[kv_from:kv_to]), quant_kv(vc[kv_from:kv_to]))
        return (img, txt), kvo

    (img, txt), kv = jax.lax.scan(layer, (img, txt), (blocks, acts, hk, hv))
    return img, kv


# ------------------------------------------------------------------ full forward for one window and one sigma
def refiner(t_, x, t, mask, dt, arch):
    temb = lin(silu(lin(tstep(t)[None].astype(dt), t_['rt0_w'], t_['rt0_b'])), t_['rt2_w'], t_['rt2_b'])[0]
    mf = mask.astype(dt)[:, None]
    ctx = ((x * mf).sum(0) / mf.sum(0))[None]
    c = temb + lin(silu(lin(ctx, t_['rc1_w'], t_['rc1_b'])), t_['rc2_w'], t_['rc2_b'])[0]
    h = lin(x, t_['rin_w'], t_['rin_b'])
    m = mask.at[0].set(True)
    S = x.shape[0]
    for j in range(2):
        g_msa, g_mlp = jnp.split(lin(silu(c)[None], t_['r%d_ada_w' % j], t_['r%d_ada_b' % j])[0], 2)
        qkv = lin(ln(h, t_['r%d_n1_w' % j], t_['r%d_n1_b' % j]), t_['r%d_qkv_w' % j], t_['r%d_qkv_b' % j]).reshape(S, 3, arch.heads, arch.head_dim)
        a = attn_ref(qkv[:, 0], qkv[:, 1], qkv[:, 2], m)
        a = jnp.where(m[:, None, None], a, 0).reshape(S, arch.hidden)
        h = h + lin(a, t_['r%d_proj_w' % j], t_['r%d_proj_b' % j]) * g_msa
        f = lin(silu(lin(ln(h, t_['r%d_n2_w' % j], t_['r%d_n2_b' % j]), t_['r%d_fc1_w' % j], t_['r%d_fc1_b' % j])), t_['r%d_fc2_w' % j], t_['r%d_fc2_b' % j])
        h = h + f * g_mlp
    return h


def cam_embed(t_, extr, dt):
    """Normalized camera extrinsics [frames, 4, 4] -> camera tokens [frames / 4, 64]."""
    e = extr.astype(dt)
    pad = (4 - e.shape[0] % 4) % 4
    if pad:
        e = jnp.concatenate([jnp.repeat(e[:1], pad, 0), e])
    e = e.reshape(e.shape[0] // 4, 64)
    e = lin(e, t_['ext_w'], t_['ext_b'])
    return e + lin(silu(lin(e, t_['c1_w'], t_['c1_b'])), t_['c2_w'], t_['c2_b'])


def make_forward(mesh, bspec, aspec, ops, arch=ARCH, *, want_kv=False, kv_from=0, kv_to=None, dt=jnp.bfloat16,
                 bq=1024, bkv=7168, with_hist=True):
    """-> fwd(top, blocks, acts, lat, t, text, tmask, cam, tabs, hk, hv, gq, gk) = (velocity [32, frames, h, w], kv).

    text [Lt, text_dim] f32 padded rows zero, tmask [Lt] bool, cam [frames, 64] camera tokens, hk/hv int8 history
    {'q': [layers, Lh, heads, 128], 's': [layers, Lh, heads]} or None for the first window (with_hist=False),
    gq [LQ, 1] / gk [1, LK] group ids (buckets.groups).
    kv (want_kv only): int8 keys and values of image rows kv_from..kv_to (and the text), the next window's history.
    """
    hsp = {'q': P(None, None, 'tp', None), 's': P(None, None, 'tp')}
    hin = hsp if with_hist else P()
    core = smap(lambda *a: core_local(*a, ops=ops, arch=arch, want_kv=want_kv, kv_from=kv_from, kv_to=kv_to, dt=dt, bq=bq, bkv=bkv),
                mesh, (bspec, aspec) + (P(),) * 12 + (hin, hin, P(), P()), (P(), (hsp, hsp) if want_kv else None))

    def fwd(top, blocks, acts, lat, t, text, tmask, cam, tabs, hk, hv, gq, gk):
        C, tt, th, tw = lat.shape
        img = lin(lat.reshape(C, -1).T.astype(dt), top['img_in_w'], top['img_in_b'])
        vec = lin(silu(lin(tstep(t)[None].astype(dt), top['time0_w'], top['time0_b'])), top['time2_w'], top['time2_b'])[0]
        txt = refiner(top, text.astype(dt), t, tmask, dt, arch)
        img, kv = core(blocks, acts, img, txt, tmask, vec, cam.astype(dt), tabs['cos'], tabs['sin'], tabs['cq_cos'], tabs['cq_sin'],
                       tabs['ck_cos'], tabs['ck_sin'], tabs['fid'], hk, hv, gq, gk)
        sh, sc = jnp.split(lin(silu(vec)[None], top['fin_ada_w'], top['fin_ada_b'])[0], 2)
        out = lin(ln(img) * (1 + sc) + sh, top['fin_w'], top['fin_b'])
        return out.reshape(tt, th, tw, arch.latent_ch).transpose(3, 0, 1, 2), kv
    return fwd
