"""Qwen3.8-27B (Hugging Face model_type qwen3_5_text) in plain JAX.

Layers come in groups of `full_attention_interval`: (FI - 1) gated delta rule linear-attention layers, then one gated
full-attention layer. All math follows the Hugging Face reference: zero-centred RMSNorm, partial rotary embedding, causal
depthwise conv + SiLU in front of the linear attention, gated RMSNorm on its output.

Weights: every big linear is either a bf16 array (out, in) or {"q": int8 (out, in), "s": f32 (out,)} with y = (x . q) * s.

State per slot (S slots + 1 scratch slot used by padding rows):
  full attention   K, V caches (NG, S + 1, CTX, KV, HD) bf16
  linear attention recurrent state (NG, FI - 1, S + 1, Hv, Dk, Dv) f32 and conv tail (NG, FI - 1, S + 1, K - 1, conv dim) bf16
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ...runtime import activation_edits as AE
from ...runtime import kvcache
from ...runtime.activation_edits import ActivationEdits  # noqa: F401  (the engine interface; re-exported for model users)

f32, bf16 = jnp.float32, jnp.bfloat16
SPEC_VERIFY = True       # forward(verify=True) with gated_delta_commit / conv_commit (runtime/spec_step.py)
SPEC_MTP_HEAD = "mashbox.models.qwen3_8_27b.spec"    # its MTP head (make_mtp, load_mtp) for BSHJ files
HI = lax.Precision.HIGHEST
CHUNK = 64
NEG = -1e30


class Config:
    """Shapes from the model's config.json (nested text_config or flat). Anything this port does not implement is refused."""

    def __init__(self, model_dir, layers=None):
        raw = json.load(open(os.path.join(model_dir, "config.json")))
        c = raw.get("text_config") or raw
        if c.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
            raise NotImplementedError(f"model_type {c.get('model_type')}")
        rp = c.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope {rp}")
        if c.get("hidden_act", "silu") != "silu" or not c.get("attn_output_gate", True):
            raise NotImplementedError("hidden_act / attn_output_gate")
        if c.get("attention_bias"):
            raise NotImplementedError("attention_bias")
        self.L, self.D, self.F = c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]
        self.H, self.KV, self.HD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.V = c["vocab_size"]
        self.theta = float(rp.get("rope_theta") or c.get("rope_theta") or 10000.0)
        self.prf = float(rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        self.RD = int(self.HD * self.prf)
        self.eps = float(c.get("rms_norm_eps", 1e-6))
        self.tie = bool(raw.get("tie_word_embeddings", c.get("tie_word_embeddings", False)))
        self.HK, self.HV = c["linear_num_key_heads"], c["linear_num_value_heads"]
        self.DK, self.DV, self.CK = c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
        self.CONV = 2 * self.HK * self.DK + self.HV * self.DV
        self.FI = int(c.get("full_attention_interval", 4))
        want = [("full_attention" if (i + 1) % self.FI == 0 else "linear_attention") for i in range(self.L)]
        if (c.get("layer_types") or want) != want or self.L % self.FI:
            raise NotImplementedError("layer_types is not (FI-1) x linear + 1 x full, repeated")
        if layers is not None:                      # tests only: the first `layers` layers (a multiple of FI)
            assert layers % self.FI == 0 and 0 < layers <= self.L
            self.L = layers
        self.NG = self.L // self.FI
        self.types = want[:self.L]


# ---------------------------------------------------------------------------------------------------------------- math
def linear(x, w, pj=None):
    """(.., in) x (out, in) with f32 accumulation, rounded to x.dtype. w = bf16 array or {"q": int8, "s": f32 (out,)}.
    pj (activation edits only): a function on the f32 output before the rounding; None = the plain ops."""
    if isinstance(w, dict):
        y = lax.dot_general(x, w["q"].astype(x.dtype), (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=f32)
        if pj is None:
            return (y * w["s"]).astype(x.dtype)
        return pj(y * w["s"]).astype(x.dtype)
    if pj is None:
        return lax.dot_general(x, w, (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=f32).astype(x.dtype)
    return pj(lax.dot_general(x, w, (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=f32)).astype(x.dtype)


def rms_norm(x, w, eps):
    """Zero-centred RMSNorm: normalise in f32, times (1 + w) in f32, cast back."""
    x32 = x.astype(f32)
    return (x32 * lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps) * (1.0 + w.astype(f32))).astype(x.dtype)


def rms_norm_gated(x, z, w, eps):
    """Gated RMSNorm: normalise in f32, cast back, times the weight, times silu(z) in f32, cast back."""
    x32 = x.astype(f32)
    h = w * (x32 * lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype)
    return (h.astype(f32) * jax.nn.silu(z.astype(f32))).astype(x.dtype)


def rope_partial(x, cos, sin, rd):
    xr, xp = x[..., :rd], x[..., rd:]
    h = rd // 2
    xr = xr * cos + jnp.concatenate([-xr[..., h:], xr[..., :h]], -1) * sin
    return jnp.concatenate([xr, xp], -1)


def l2norm(x):
    return x * lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)


def softplus(x):   # torch softplus(beta=1, threshold=20)
    return jnp.where(x > 20.0, x, jnp.log1p(jnp.exp(jnp.minimum(x, 20.0))))


EMBED_TILE = 32         # rows of an int8 (32, 128) TPU tile
EMBED_TILE_MAX_ROWS = 8  # lookups of at most this many rows read whole tiles (see int8_rows)


def int8_rows(q, toks):
    """q[toks] for an int8 (N, D) table, exactly. For a few rows (decode), each row is read from its whole (32, D) tile:
    q viewed as (N / 32, 32, D) (the same bytes in the TPU's int8 tiling), the tile of each token gathered, its row
    picked by a one-hot sum in int32 (exact). A plain gather of one row makes XLA relayout the whole table from
    T(32,128) to T(8,128) on every call (v6e-1 int8 served path, B=1: 2.12 ms/step; 26.45 -> 24.24 ms/step with this form,
    logits bit-equal). Larger lookups (prefill) keep the plain gather, which needs no relayout there and reads 32x fewer
    bytes."""
    N, D = q.shape
    if q.dtype != jnp.int8 or N % EMBED_TILE or toks.size > EMBED_TILE_MAX_ROWS:
        return q[toks]
    t = q.reshape(N // EMBED_TILE, EMBED_TILE, D)[toks // EMBED_TILE]
    m = lax.broadcasted_iota(jnp.int32, t.shape, t.ndim - 2) == (toks % EMBED_TILE)[..., None, None]
    return jnp.where(m, t, jnp.int8(0)).astype(jnp.int32).sum(-2).astype(jnp.int8)


def embed(E, toks, pj=None):
    """Embedding rows of toks. pj (activation edits only): a function on the rows in f32 before the rounding."""
    if isinstance(E, dict):
        x = int8_rows(E["q"], toks).astype(f32) * E["s"][toks][..., None]
        return (x if pj is None else pj(x)).astype(bf16)
    if pj is None:
        return E[toks]
    return pj(E[toks].astype(f32)).astype(bf16)


# Activation edits (mashbox.runtime.activation_edits, the engine interface): the sites this model runs.
ACTIVATION_EDIT_SITES = AE.SITES
# Drafter layer taps: forward(taps=) returns the residual stream after the named decoder layers, for a drafter that
# reads the target's layer outputs (DFlash 2). One chip only: the tensor-parallel forward has none.
# The DFlash 2 drafter published for this model (runtime/drafters/dflash.py; it reads layers 5, 19, 33, 47, 61):
# z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 (a mirror of incoai/Qwen3.8-27B-DFlash2 @
# 015e795645c74b1a0eeef3b570031fb62e769bc5, model.safetensors sha256
# 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c), loaded unchanged, under the licence its publisher
# states (README front matter: apache-2.0).
LAYER_TAPS = True


def gated_delta_chunk(q, k, v, g, beta, s0, CH=CHUNK):
    """Gated delta rule, chunked form (the Hugging Face torch reference algorithm), f32 at HIGHEST precision.
    q, k (B,T,H,Dk) l2-normalised, v (B,T,H,Dv), g / beta (B,T,H) (0 / 0 at padded positions), s0 (B,H,Dk,Dv)
    -> o (B,T,H,Dv), s1 (B,H,Dk,Dv)."""
    B, T, Hh, Dk = k.shape
    Dv = v.shape[-1]
    tr = lambda x: jnp.swapaxes(x, 1, 2)
    q, k, v, beta, g = tr(q), tr(k), tr(v), tr(beta), tr(g)
    q = q * (Dk ** -0.5)
    pad = (-T) % CH
    if pad:
        q, k, v = (jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (q, k, v))
        beta, g = (jnp.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (beta, g))
    N = (T + pad) // CH
    vb, kb = v * beta[..., None], k * beta[..., None]
    ch = lambda x: x.reshape(B, Hh, N, CH, x.shape[-1])
    q, k, kb, vb = ch(q), ch(k), ch(kb), ch(vb)
    cum = jnp.cumsum(g.reshape(B, Hh, N, CH), -1)
    up = jnp.triu(jnp.ones((CH, CH), bool), 1)
    pdec = jnp.exp(jnp.where(up, -jnp.inf, cum[..., :, None] - cum[..., None, :]))
    ein = lambda s, a, b: jnp.einsum(s, a, b, precision=HI, preferred_element_type=f32)
    ut = ein("bhncd,bhned->bhnce", kb, k) * pdec
    intra = ein("bhncd,bhned->bhnce", q, k) * pdec
    dkb = kb * jnp.exp(cum)[..., None]
    nv = lax.linalg.triangular_solve(ut, vb, left_side=True, lower=True, unit_diagonal=True)
    kcd = lax.linalg.triangular_solve(ut, dkb, left_side=True, lower=True, unit_diagonal=True)
    q2 = q * jnp.exp(cum)[..., None]
    k2 = k * jnp.exp(cum[..., -1:] - cum)[..., None]
    cd = jnp.exp(cum[..., -1])

    def step(s, xs):
        nv_i, kcd_i, q2_i, in_i, k2_i, cd_i = xs
        vn = nv_i - ein("bhcd,bhde->bhce", kcd_i, s)
        o = ein("bhcd,bhde->bhce", q2_i, s) + ein("bhce,bhed->bhcd", in_i, vn)
        s = s * cd_i[..., None, None] + ein("bhcd,bhce->bhde", k2_i, vn)
        return s, o
    mv = lambda x: jnp.moveaxis(x, 2, 0)
    s1, o = lax.scan(step, s0, (mv(nv), mv(kcd), mv(q2), mv(intra), mv(k2), mv(cd)))
    o = jnp.moveaxis(o, 0, 2).reshape(B, Hh, N * CH, Dv)[:, :, :T]
    return tr(o), s1


def gated_delta_state(k, v, g, beta, s):
    """The state update of one recurrent step (k (B,H,Dk), v (B,H,Dv), g / beta (B,H), s (B,H,Dk,Dv) f32). With g = 0 and
    beta = 0 it returns s bit for bit (s * 1, + k * 0): speculative decoding masks rejected steps that way."""
    s = s * jnp.exp(g)[..., None, None]
    kv = jnp.einsum("bhkv,bhk->bhv", s, k, precision=HI, preferred_element_type=f32)
    u = (v - kv) * beta[..., None]
    return s + k[..., :, None] * u[..., None, :]


def gated_delta_step(q, k, v, g, beta, s):
    """One recurrent step: q, k (B,H,Dk), v (B,H,Dv), g / beta (B,H), s (B,H,Dk,Dv) f32."""
    q = q / (q.shape[-1] ** 0.5)
    s = gated_delta_state(k, v, g, beta, s)
    o = jnp.einsum("bhkv,bhk->bhv", s, q, precision=HI, preferred_element_type=f32)
    return o, s


def gated_delta_steps(qr, kr, v, g, beta, s0):
    """Speculative verify: T tokens as T one-token steps (the decode math, so the outputs equal T decode steps), state not
    returned. qr, kr (B,T,Hk,Dk) before l2norm, v (B,T,Hv,Dv), g / beta (B,T,Hv) f32 -> o (B,T,Hv,Dv) f32, kn = l2norm(kr) f32."""
    rep = v.shape[2] // qr.shape[2]
    qn, kn = l2norm(qr.astype(f32)), l2norm(kr.astype(f32))
    q, k = jnp.repeat(qn, rep, axis=2), jnp.repeat(kn, rep, axis=2)
    mv = lambda x: jnp.moveaxis(x, 1, 0)

    def st(s, xs):
        o, s = gated_delta_step(*xs, s)
        return s, o
    _, o = lax.scan(st, s0, (mv(q), mv(k), mv(v.astype(f32)), mv(g), mv(beta)))
    return jnp.moveaxis(o, 0, 1), kn


def gated_delta_commit(R, rec, n, rep):
    """Speculative commit: replay the first n[b] accepted steps of every linear-attention layer from the state before the
    verify pass; steps past n[b] get g = beta = 0 and leave the state as it is. R (NG, FI - 1, S + 1, Hv, Dk, Dv) f32, rows
    0..B-1; rec["k"] (NG, FI - 1, B, T, Hk, Dk), rec["v"] (.., Hv, Dv), rec["g"] / rec["b"] (NG, FI - 1, B, T, Hv); n (B,) in
    1..T."""
    NG, FI1, B, T = rec["g"].shape[:4]
    m = (jnp.arange(T)[None] < n[:, None])[..., None]

    def layer(l, R):
        g_, i_ = l // FI1, l % FI1
        s = lax.dynamic_slice(R, (g_, i_, 0, 0, 0, 0), (1, 1, B) + R.shape[3:])[0, 0]
        k = jnp.repeat(rec["k"][g_, i_], rep, axis=2)
        v, g, b = rec["v"][g_, i_], jnp.where(m, rec["g"][g_, i_], 0.0), jnp.where(m, rec["b"][g_, i_], 0.0)
        for t in range(T):
            s = gated_delta_state(k[:, t], v[:, t], g[:, t], b[:, t], s)
        return lax.dynamic_update_slice(R, s[None, None], (g_, i_, 0, 0, 0, 0))
    return lax.fori_loop(0, NG * FI1, layer, R)


def conv_commit(C, x, n):
    """Conv tail after n[b] accepted tokens: the last K - 1 rows of [old tail, x[:, :n]]. C (NG, FI - 1, S + 1, K - 1, CONV)
    rows 0..B-1, x (NG, FI - 1, B, T, CONV) the verify pass's conv inputs."""
    B, K1 = x.shape[2], C.shape[3]
    xe = jnp.concatenate([C[:, :, :B], x.astype(C.dtype)], 3)
    idx = (n[:, None] + jnp.arange(K1)[None])[None, None, :, :, None]
    return C.at[:, :, :B].set(jnp.take_along_axis(xe, jnp.broadcast_to(idx, xe.shape[:3] + (K1, 1)), 3))


def gated_delta(qr, kr, v, g, beta, s0):
    """qr, kr (B,T,Hk,Dk) before l2norm, v (B,T,Hv,Dv), g / beta (B,T,Hv) f32, s0 (B,Hv,Dk,Dv) f32 -> o (B,T,Hv,Dv) f32, s1.
    Value head h reads key head h // (Hv / Hk)."""
    rep = v.shape[2] // qr.shape[2]
    q = jnp.repeat(l2norm(qr.astype(f32)), rep, axis=2)
    k = jnp.repeat(l2norm(kr.astype(f32)), rep, axis=2)
    v = v.astype(f32)
    if q.shape[1] == 1:
        o, s1 = gated_delta_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], s0)
        return o[:, None], s1
    return gated_delta_chunk(q, k, v, g, beta, s0)


# ---------------------------------------------------------------------------------------------------------------- state
def new_state(c, S, CTX, extra_groups=0):
    """Zeroed per-slot state for S slots (+1 scratch slot). extra_groups: K/V groups after the model's NG (speculative
    decoding keeps the MTP layer's K/V in group NG). K/V: runtime/kvcache.py's layout (NG, S+1, CTX, KV x HD)."""
    return (kvcache.zeros(c.NG + extra_groups, S + 1, CTX, c.KV, c.HD, bf16),
            (jnp.zeros((c.NG, c.FI - 1, S + 1, c.HV, c.DK, c.DV), f32),
             jnp.zeros((c.NG, c.FI - 1, S + 1, c.CK - 1, c.CONV), bf16)))


# ---------------------------------------------------------------------------------------------------------------- forward
def attend_one(Kc, Vc, g, q, k, v, mask, slots, Tc, scale, contig=False):
    """Full-attention core on one chip: read slot rows of group g's K/V cache (positions < Tc; contig: the rows are slots
    0..B-1, one slice), append this call's k, v, masked softmax. q (B,T,KV,G,HD), k / v (B,T,KV,HD), mask (B,T,Tc+T) ->
    o (B,T,KV,G,HD) bf16."""
    if Tc:
        kc, vc = kvcache.read_heads(Kc, Vc, g, slots, Tc, contig, k.shape[2], k.shape[3])
        kk, vv = jnp.concatenate([kc, k], 1), jnp.concatenate([vc, v], 1)
    else:
        kk, vv = k, v
    s = jnp.einsum("btkgd,bskd->bkgts", q, kk, preferred_element_type=f32) * scale
    p = jax.nn.softmax(jnp.where(mask[:, None, None], s, NEG), -1).astype(bf16)
    return jnp.einsum("bkgts,bskd->btkgd", p, vv, preferred_element_type=f32).astype(bf16)


def kv_write_one(Kc, Vc, slots, pos, ks, vs):
    """Write this call's K/V (NG,B,T,KV,HD) at (slot, position) in place; positions past the cache are dropped (an MTP
    group after the model's groups is written by its own layer). runtime/kvcache.write."""
    return kvcache.write(Kc, Vc, slots, pos, ks, vs)


def kv_write_group(Kc, Vc, g, slots, pos, k, v):
    """Write group g's K/V (B,T,KV,HD) at (slot, position) inside the layer loop, in place; positions past the cache are
    dropped."""
    return kvcache.write(Kc, Vc, slots, pos, k[None], v[None], g=g)


def make_forward(c, decode_kernels=None, attend=attend_one, kv_write=kv_write_one, kv_in_loop=False, kv_write_g=kv_write_group,
                 edits=None):
    """-> forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False) -> dict(ck, cv, nxt, lg).

    toks (B, T) int32 at positions p0 + 0..T-1 (lens = valid tokens per row, rows padded on the right), slots (B,) = state
    slot per row. Tc (static) = how many cached positions full attention reads. contig (static) = slots are arange(B).
    nxt = greedy next token per row, lg = its logits as f32 (all positions when all_logits).
    decode_kernels: optional {B: fn(q, k, v, g, beta, pool, slots, layer) -> (o, pool)} compiled linear-attention decode
    for a batch of B rows; used only for one-token steps with contig slots. Without it, decode runs on plain JAX.
    attend / kv_write: the K/V cache read + attention core and the cache write (defaults: one chip). The tensor-parallel
    model (shard.py) passes shard_map versions so each chip reads and writes only its own KV heads.
    kv_in_loop (tensor parallel): the K/V cache rides in the layer-loop carry and each group writes its K/V in place inside
    the loop (kv_write_g), the JetStream / MaxText decode layout. Off (one chip): the cache is read inside the loop and
    written once after it, as before. With the cache read-only inside the loop and updated after it, XLA on TPU copies the
    whole cache twice per decode program (v5e-4, ctx 32768 x 8 slots: 9.02 GiB temp per chip).
    edits: an ActivationEdits (mashbox.runtime.activation_edits) = forward takes ed= and returns out["edit_reads"]; None = no edit site at all, the
    program is the one built without the argument."""
    AE.check_make_forward_args(edits, kv_in_loop)
    L, D, H, KV, HD, V = c.L, c.D, c.H, c.KV, c.HD, c.V
    FI, NG, RD = c.FI, c.NG, c.RD
    HK, HV, DK, DV, CK = c.HK, c.HV, c.DK, c.DV, c.CK
    KD, VD = HK * DK, HV * DV
    G, scale, eps = H // KV, HD ** -0.5, c.eps
    inv = (1.0 / (np.float32(c.theta) ** (np.arange(0, RD, 2, dtype=np.int64).astype(np.float32) / np.float32(RD)))).astype(np.float32)

    def at(t, i):
        """position i inside a layer group: a list by index, a stacked array by its leading axis"""
        if isinstance(t, list):
            return t[i]
        if isinstance(t, dict) and set(t) != {"q", "s"}:
            return {k: at(v, i) for k, v in t.items()}
        return jax.tree_util.tree_map(lambda a: a[i], t)

    def forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, verify=False, hidden=False, keep=None,
                ed=None, taps=()):
        """taps (static): decoder layer ids whose outputs (the residual stream after that layer's MLP, before the final norm;
        Hugging Face hidden_states[l + 1]) come back concatenated in that order as out["taps"] (B, T, len(taps) * D), from the
        verify pass and from prefill (a block drafter's context, runtime/drafters/dflash.py); () = the program without them.
        keep (B,) bool or None: rows whose linear-attention state this call may change; the state rows of the others
        (padding rows of a decode bucket, rows paused mid-prefill) are written back as they were. The verify pass writes no
        linear state (speculative decoding commits the accepted part after it), so it takes no keep.
        ed: the activation-edit operand (make_forward(edits=...) only)."""
        AE.check_forward_args(edits, ed, verify)
        B, T = toks.shape
        taps = tuple(int(l) for l in taps)
        assert all(0 <= l < L for l in taps), taps
        assert not (taps and (kv_in_loop or edits is not None)), "taps: one chip, no activation edits"
        tsub = sorted({l % FI for l in taps})     # sub-layers of a group whose output is tapped
        assert not verify or (contig and not kv_in_loop), "verify runs on contig rows, one chip"
        assert not (verify and keep is not None), "the verify pass takes no keep mask"
        Kc, Vc = ck
        R0, C0 = cv
        kern = decode_kernels.get(B) if (decode_kernels and contig and T == 1 and not verify) else None
        assert not (kern is not None and kv_in_loop), "kv_in_loop is the plain JAX decode path"
        pos = p0[:, None] + jnp.arange(T, dtype=jnp.int32)[None]
        valid = jnp.arange(T)[None] < lens[:, None]
        fr = pos.astype(f32)[..., None] * jnp.asarray(inv)
        emb = jnp.concatenate([fr, fr], -1)
        cos, sin = jnp.cos(emb).astype(bf16)[:, :, None], jnp.sin(emb).astype(bf16)[:, :, None]
        m2 = (jnp.arange(T)[None, :, None] >= jnp.arange(T)[None, None, :]) & valid[:, None, :]
        if Tc:
            m1 = jnp.broadcast_to((jnp.arange(Tc)[None, :] < p0[:, None])[:, None, :], (B, T, Tc))
            mask = jnp.concatenate([m1, m2], -1)
        else:
            mask = m2
        fresh = p0 == 0
        if edits is None:
            x = embed(W["embed"], toks)
        else:
            ctx = dict(pos=pos, valid=valid, slots=slots, p0=p0, lens=lens)
            e_call, e_lay = AE.split_op(ed)
            x = embed(W["embed"], toks, edits.writer("embed", -1, e_call, None, ctx))

        def lin_attn(h, w, r0, c0, pool=None, layer=None, pj=None):
            qkv = linear(h, w["qkv"])
            z = linear(h, w["z"]).reshape(B, T, HV, DV)
            bb, aa = linear(h, w["b"]), linear(h, w["a"])
            xe = jnp.concatenate([c0, qkv], 1)
            cw = w["conv"].astype(f32)
            acc = xe[:, 0:T].astype(f32) * cw[:, 0]
            for t_ in range(1, CK):
                acc = acc + xe[:, t_:t_ + T].astype(f32) * cw[:, t_]
            co = jax.nn.silu(acc.astype(bf16).astype(f32)).astype(bf16)
            c1 = jnp.take_along_axis(xe, (lens[:, None] + jnp.arange(CK - 1)[None])[..., None], 1)
            qr, kr = co[..., :KD].reshape(B, T, HK, DK), co[..., KD:2 * KD].reshape(B, T, HK, DK)
            v = co[..., 2 * KD:].reshape(B, T, HV, DV)
            beta = jax.nn.sigmoid(bb.astype(f32)).astype(bf16).astype(f32)
            gg = -jnp.exp(w["A"].astype(f32)) * softplus(aa.astype(f32) + w["dt"])
            vm = valid[..., None]
            beta, gg = jnp.where(vm, beta, 0.0), jnp.where(vm, gg.astype(f32), 0.0)
            if pool is not None:
                o, r1 = kern(qr[:, 0], kr[:, 0], v[:, 0], gg[:, 0], beta[:, 0], pool, jnp.arange(B, dtype=jnp.int32), layer)
                o = o[:, None]
            elif verify:              # state not committed: keep what the commit replays (k, v, g, beta, conv input)
                o, kn = gated_delta_steps(qr, kr, v, gg, beta, r0)
                r1, c1 = dict(k=kn, v=v.astype(f32), g=gg, b=beta), qkv
            else:
                o, r1 = gated_delta(qr, kr, v, gg, beta, r0)
            o = rms_norm_gated(o.astype(bf16), z, w["nw"], eps).reshape(B, T, VD)
            return linear(o, w["out"], pj), r1, c1

        def full_attn(h, w, g, Kl=None, Vl=None, pj=None):
            qg = linear(h, w["wq"]).reshape(B, T, H, 2 * HD)
            q, gate = qg[..., :HD], qg[..., HD:].reshape(B, T, H * HD)
            k, v = linear(h, w["wk"]).reshape(B, T, KV, HD), linear(h, w["wv"]).reshape(B, T, KV, HD)
            q, k = rms_norm(q, w["qn"], eps), rms_norm(k, w["kn"], eps)
            q, k = rope_partial(q, cos, sin, RD), rope_partial(k, cos, sin, RD)
            o = attend(Kc if Kl is None else Kl, Vc if Vl is None else Vl, g, q.reshape(B, T, KV, G, HD), k, v, mask, slots, Tc,
                       scale, contig=contig).reshape(B, T, H * HD)
            o = o * jax.nn.sigmoid(gate.astype(f32)).astype(bf16)
            return linear(o, w["wo"], pj), k, v

        def mlp(x, wm, i):
            h = rms_norm(x, wm["ln2"][i], eps)
            a = jax.nn.silu(linear(h, at(wm["wg"], i)).astype(f32)).astype(bf16) * linear(h, at(wm["wu"], i))
            return x + linear(a, at(wm["wd"], i))

        def mlp_edited(x, wm, i, l, lay):
            """mlp with the act, writer("mlp") and resid sites -> (x, reads)."""
            h = rms_norm(x, wm["ln2"][i], eps)
            a = jax.nn.silu(linear(h, at(wm["wg"], i)).astype(f32)).astype(bf16) * linear(h, at(wm["wu"], i))
            a, ra = edits.act(l, e_call, lay, a, ctx)
            x = x + linear(a, at(wm["wd"], i), edits.writer("mlp", l, e_call, lay, ctx))
            x, rr = edits.resid(l, e_call, lay, x, ctx)
            return x, AE.merge_reads(ra, rr)

        def group(carry, xg):
            if kv_in_loop:
                x, R, C, Kl, Vl = carry
            else:
                x, R, C = carry
            g, w = xg["g"], xg["w"]
            rl, cl = [], []
            tp = {}                   # tapped sub-layer outputs of this group (taps only)
            erl = []                  # activation-edit reads of the group's layers (edits only)
            if contig:
                Cg = lax.dynamic_slice(C, (g, 0, 0, 0, 0), (1, FI - 1, B) + C.shape[3:])[0]
                if kern is not None:
                    Rg = [None] * (FI - 1)
                    Cg = lax.optimization_barrier(Cg)
                else:
                    Rg = lax.dynamic_slice(R, (g, 0, 0, 0, 0, 0), (1, FI - 1, B) + R.shape[3:])[0]
                    Rg, Cg = lax.optimization_barrier((Rg, Cg))
                    Rg0 = Rg
                    Rg = jnp.where(fresh[None, :, None, None, None], 0.0, Rg)
                Cg0 = Cg
                Cg = jnp.where(fresh[None, :, None, None], jnp.zeros((), C.dtype), Cg)
            else:
                gi = jnp.full((FI - 1, B), g, jnp.int32)
                ii = jnp.broadcast_to(jnp.arange(FI - 1, dtype=jnp.int32)[:, None], (FI - 1, B))
                si = jnp.broadcast_to(slots[None, :], (FI - 1, B))
                Rg, Cg = lax.optimization_barrier((R[gi, ii, si], C[gi, ii, si]))
                Rg0, Cg0 = Rg, Cg
                Rg = jnp.where(fresh[None, :, None, None, None], 0.0, Rg)
                Cg = jnp.where(fresh[None, :, None, None], jnp.zeros((), C.dtype), Cg)
            for i in range(FI):
                h = rms_norm(x, w["mlp"]["ln1"][i], eps)
                if edits is not None:
                    l_ = g * FI + i
                    lay = None if e_lay is None else jax.tree_util.tree_map(lambda a: a[i], xg["ed"])
                    pj = edits.writer("attn", l_, e_call, lay, ctx)
                else:
                    pj = None
                if i < FI - 1:
                    if kern is not None:
                        o, R, c1 = lin_attn(h, at(w["lin"], i), None, Cg[i], pool=R, layer=g * (FI - 1) + i, pj=pj)
                    else:
                        o, r1, c1 = lin_attn(h, at(w["lin"], i), Rg[i], Cg[i], pj=pj)
                        rl.append(r1 if verify else r1.astype(R0.dtype))
                    cl.append(c1 if verify else c1.astype(C0.dtype))
                elif kv_in_loop:
                    o, k, v = full_attn(h, w["full"], g, Kl, Vl)
                    Kl, Vl = kv_write_g(Kl, Vl, g, slots, pos, k, v)
                else:
                    o, k, v = full_attn(h, w["full"], g, pj=pj)
                if edits is None:
                    x = mlp(x + o, w["mlp"], i)
                else:
                    x, er = mlp_edited(x + o, w["mlp"], i, l_, lay)
                    erl.append(er)
                if i in tsub:
                    tp[f"tap{i}"] = x
            km = None if keep is None else keep[None, :, None, None]
            if kern is not None:
                Cn = jnp.stack(cl) if km is None else jnp.where(km, jnp.stack(cl), Cg0)
                C = lax.dynamic_update_slice(C, lax.optimization_barrier(Cn)[None], (g, 0, 0, 0, 0))
                if edits is not None:
                    return (x, R, C), dict(k=k, v=v, er={n: jnp.stack([e[n] for e in erl]) for n in erl[0]})
                return (x, R, C), dict(k=k, v=v, **tp)
            if verify:
                rec = jax.tree.map(lambda *a: jnp.stack(a), *rl)
                rec["x"] = jnp.stack(cl)
                return (x, R, C), dict(k=k, v=v, rec=rec, **tp)
            Rn, Cn = jnp.stack(rl), jnp.stack(cl)
            if km is not None:
                Rn, Cn = jnp.where(km[..., None], Rn, Rg0), jnp.where(km, Cn, Cg0)
            Rn, Cn = lax.optimization_barrier((Rn, Cn))
            if contig:
                R = lax.dynamic_update_slice(R, Rn[None], (g, 0, 0, 0, 0, 0))
                C = lax.dynamic_update_slice(C, Cn[None], (g, 0, 0, 0, 0))
            else:
                R, C = R.at[gi, ii, si].set(Rn), C.at[gi, ii, si].set(Cn)
            if kv_in_loop:
                return (x, R, C, Kl, Vl), None
            if edits is not None:
                return (x, R, C), dict(k=k, v=v, er={n: jnp.stack([e[n] for e in erl]) for n in erl[0]})
            return (x, R, C), dict(k=k, v=v, **tp)

        xs = dict(w=W["g"], g=jnp.arange(NG, dtype=jnp.int32))
        if edits is not None and e_lay is not None:      # (L, ..) -> (NG, FI, ..): each group scans its FI layers' rows
            xs["ed"] = jax.tree_util.tree_map(lambda a: a.reshape((NG, FI) + a.shape[1:]), e_lay)
        if kern is not None:
            (x, R1, C1), ys = lax.scan(group, (x, R0.reshape((NG * (FI - 1),) + R0.shape[2:]), C0), xs)
            R1 = R1.reshape(R0.shape)
        elif kv_in_loop:
            (x, R1, C1, Kc, Vc), _ = lax.scan(group, (x, R0, C0, Kc, Vc), xs)
        else:
            (x, R1, C1), ys = lax.scan(group, (x, R0, C0), xs)
        if not kv_in_loop:
            Kc, Vc = kv_write(Kc, Vc, slots, pos, ys["k"], ys["v"])
        out = dict(ck=(Kc, Vc), cv=(R1, C1))
        if taps:                  # (NG, B, T, D) per tapped sub-layer -> layer l = group l // FI, sub-layer l % FI
            out["taps"] = jnp.concatenate([ys[f"tap{l % FI}"][l // FI] for l in taps], -1)
        if edits is not None:     # (NG, FI, ..) -> (L, ..)
            out["edit_reads"] = {n: a.reshape((NG * FI,) + a.shape[2:]) for n, a in ys["er"].items()}
        if verify:                # bf16 logits at every position, as one-token decode computes them for its argmax
            xn = rms_norm(x, W["norm"], eps)
            lg = linear(xn, W["head"])
            out.update(rec=ys["rec"], hid=xn, tgt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg.astype(f32))
            return out
        if hidden:                # post-norm hidden at every position (the MTP head's input)
            out["hid"] = rms_norm(x, W["norm"], eps)
        if all_logits:
            out["lg"] = linear(rms_norm(x, W["norm"], eps), W["head"]).astype(f32)
            return out
        hl = rms_norm(x[jnp.arange(B), jnp.maximum(lens - 1, 0)], W["norm"], eps)
        lg = linear(hl, W["head"])               # bf16 logits; argmax takes the first index on a tie
        out.update(nxt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg.astype(f32))
        return out

    return forward
