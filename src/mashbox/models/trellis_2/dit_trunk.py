"""TRELLIS.2 flow DiT trunk in JAX (MASHBOX Labs TPU port).

One trunk serves all four flow stages:
  stage 2 SparseStructureFlowModel (dense, 16^3 = 4096 tokens, in/out 8)
  stage 4/stage 6 SLatFlowModel shape (sparse, variable N, in/out 32)
  stage 7 SLatFlowModel tex (sparse, in 64 = noise 32 + shape slat 32, out 32)
30 blocks, d1536, 12 heads (head_dim 128), MLP 8192, 3D RoPE, qk RMSNorm, share_mod adaLN, cross-attention
to the image condition. Math written to match the MIT TRELLIS.2 code at commit 75fbf018
(trellis2/models/sparse_structure_flow.py, structured_latent_flow.py, modules/transformer/modulated.py,
modules/sparse/transformer/modulated.py, modules/attention/*, modules/sparse/attention/*).

Sparse variable-length tokens are handled as a padded batch [B, Lpad, C] plus a valid mask [B, Lpad]. Every
op except self-attention is per token, so padding only has to be masked out of the attention keys. The dense
SS case is the same trunk with mask=None.

Precision policy mirrors the vendor code (model dtype `dtype`, bf16 on chip, f32 for the float32 checks against the vendor code):
  f32: t embedder, adaLN, input/out layers, every LayerNorm (LayerNorm32), the qk RMSNorm, RoPE, softmax.
  dtype: the block linears and the residual stream between ops.
Pure functional JAX, no framework. Licences: jax Apache-2.0, numpy BSD-3-Clause, ml_dtypes Apache-2.0.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

F32 = jnp.float32


# ----------------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiTConfig:
    in_channels: int
    out_channels: int
    model_channels: int = 1536
    cond_channels: int = 1024
    num_blocks: int = 30
    num_heads: int = 12
    mlp_ratio: float = 5.3334
    resolution: int = 16            # grid side (16 for SS, 32 for SLat 512, 64 for SLat 1024); dense uses it
    rope_freq: Tuple[float, float] = (1.0, 10000.0)
    qk_rms_norm: bool = True
    qk_rms_norm_cross: bool = True
    dtype: str = "bfloat16"         # model (block) dtype: "bfloat16" on chip, "float32" for the float32 checks
    dense: bool = False             # True = SparseStructureFlowModel input/output layout [B, C, R, R, R]
    # image_attn_mode "proj" (ProjectAttention), which TRELLIS.2 never uses: per-token proj cond width.
    # None = TRELLIS.2 "cross" mode, the default; nothing changes then (keys, block math, outputs).
    proj_in_channels: Optional[int] = None

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.model_channels * self.mlp_ratio)   # int(1536 * 5.3334) = 8192, same as torch

    @property
    def jdtype(self):
        return {"bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16, "float32": F32, "fp32": F32, "f32": F32,
                "float16": jnp.float16, "fp16": jnp.float16}[self.dtype]

    @property
    def precision(self):
        # On TPU an f32 dot at DEFAULT precision runs bf16 passes; the float32 checks need real f32.
        return jax.lax.Precision.HIGHEST if self.jdtype == F32 else jax.lax.Precision.DEFAULT

    @staticmethod
    def from_trellis_json(cfg: Mapping, dtype: Optional[str] = None) -> "DiTConfig":
        """Build from the official ckpt config json ({"name": ..., "args": {...}})."""
        a = cfg["args"]
        assert a.get("pe_mode", "ape") == "rope", "only pe_mode=rope is ported (all TRELLIS.2 flows use it)"
        assert a.get("share_mod", False), "only share_mod=True is ported (all TRELLIS.2 flows use it)"
        heads = a.get("num_heads") or a["model_channels"] // a.get("num_head_channels", 64)
        mode = a.get("image_attn_mode", "cross")
        assert mode in ("cross", "proj"), f"image_attn_mode {mode!r} is not ported (cross and proj are)"
        proj_in = None
        if mode == "proj":   # vendor: proj_in_channels if proj_in_channels is not None else ctx_channels
            proj_in = a["proj_in_channels"] if a.get("proj_in_channels") is not None else a["cond_channels"]
        return DiTConfig(
            in_channels=a["in_channels"], out_channels=a["out_channels"], model_channels=a["model_channels"],
            cond_channels=a["cond_channels"], num_blocks=a["num_blocks"], num_heads=heads,
            mlp_ratio=a.get("mlp_ratio", 4.0), resolution=a["resolution"],
            rope_freq=tuple(a.get("rope_freq", (1.0, 10000.0))), qk_rms_norm=a.get("qk_rms_norm", False),
            qk_rms_norm_cross=a.get("qk_rms_norm_cross", False), dtype=dtype or a.get("dtype", "float32"),
            dense=(cfg["name"] == "SparseStructureFlowModel"), proj_in_channels=proj_in)


# ----------------------------------------------------------------------------------------------------------------
# Weights: official state_dict (torch names) -> stacked JAX pytree
# ----------------------------------------------------------------------------------------------------------------
_BLOCK_KEYS = {
    # ours: (torch suffix, kind) ; kind "lin_w" = Linear weight [out,in] -> [in,out] in model dtype,
    # "lin_b" = Linear bias in model dtype, "f32" = parameter kept float32 (vendor convert_to only casts nn.Linear)
    "mod": ("modulation", "f32"),
    "qkv_w": ("self_attn.to_qkv.weight", "lin_w"), "qkv_b": ("self_attn.to_qkv.bias", "lin_b"),
    "sa_qg": ("self_attn.q_rms_norm.gamma", "f32"), "sa_kg": ("self_attn.k_rms_norm.gamma", "f32"),
    "sa_o_w": ("self_attn.to_out.weight", "lin_w"), "sa_o_b": ("self_attn.to_out.bias", "lin_b"),
    "n2_w": ("norm2.weight", "f32"), "n2_b": ("norm2.bias", "f32"),
    "q_w": ("cross_attn.to_q.weight", "lin_w"), "q_b": ("cross_attn.to_q.bias", "lin_b"),
    "kv_w": ("cross_attn.to_kv.weight", "lin_w"), "kv_b": ("cross_attn.to_kv.bias", "lin_b"),
    "ca_qg": ("cross_attn.q_rms_norm.gamma", "f32"), "ca_kg": ("cross_attn.k_rms_norm.gamma", "f32"),
    "ca_o_w": ("cross_attn.to_out.weight", "lin_w"), "ca_o_b": ("cross_attn.to_out.bias", "lin_b"),
    "fc1_w": ("mlp.mlp.0.weight", "lin_w"), "fc1_b": ("mlp.mlp.0.bias", "lin_b"),
    "fc2_w": ("mlp.mlp.2.weight", "lin_w"), "fc2_b": ("mlp.mlp.2.bias", "lin_b"),
}
_TOP_KEYS = {  # all float32 in the vendor model (outside the converted blocks)
    "t0_w": "t_embedder.mlp.0.weight", "t0_b": "t_embedder.mlp.0.bias",
    "t2_w": "t_embedder.mlp.2.weight", "t2_b": "t_embedder.mlp.2.bias",
    "ada_w": "adaLN_modulation.1.weight", "ada_b": "adaLN_modulation.1.bias",
    "in_w": "input_layer.weight", "in_b": "input_layer.bias",
    "out_w": "out_layer.weight", "out_b": "out_layer.bias",
}
_IGNORED = ("rope_phases",)  # dense model buffer (complex); recomputed here, never in the HF safetensors


def block_keys(cfg: DiTConfig) -> dict:
    """Per-block key map. TRELLIS.2 (proj_in_channels None) = _BLOCK_KEYS as is. The proj mode
    (ProjectAttention wraps the cross-attn): cross_attn.* -> cross_attn.cross_attn_block.*, plus the block's
    proj_linear (an nn.Linear inside blocks, so convert_to casts it like the others)."""
    if cfg.proj_in_channels is None:
        return _BLOCK_KEYS
    m = {o: (("cross_attn.cross_attn_block." + s[len("cross_attn."):]) if s.startswith("cross_attn.") else s, k)
         for o, (s, k) in _BLOCK_KEYS.items()}
    m["pj_w"] = ("cross_attn.proj_linear.weight", "lin_w")
    m["pj_b"] = ("cross_attn.proj_linear.bias", "lin_b")
    return m


def expected_state_dict_keys(cfg: DiTConfig) -> list:
    keys = list(_TOP_KEYS.values())
    for i in range(cfg.num_blocks):
        for ours, (suf, _) in block_keys(cfg).items():
            if not cfg.qk_rms_norm and ours in ("sa_qg", "sa_kg"):
                continue
            if not cfg.qk_rms_norm_cross and ours in ("ca_qg", "ca_kg"):
                continue
            keys.append(f"blocks.{i}.{suf}")
    return keys


def params_from_state_dict(sd: Mapping[str, np.ndarray], cfg: DiTConfig, strict: bool = True) -> dict:
    """sd: torch-named arrays (numpy, any float dtype incl. ml_dtypes.bfloat16). Returns the stacked pytree.
    Values are cast exactly as the vendor model holds them: block nn.Linear -> model dtype, the rest float32."""
    want = set(expected_state_dict_keys(cfg))
    have = {k for k in sd if not k.endswith(_IGNORED)}
    if strict and want != have:
        raise KeyError(f"state_dict mismatch: missing {sorted(want - have)[:8]} unexpected {sorted(have - want)[:8]}")
    dt = cfg.jdtype
    f32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    p = {}
    for ours, k in _TOP_KEYS.items():
        a = f32(sd[k])
        p[ours] = a.T if ours.endswith("_w") else a
    blocks = {}
    for ours, (suf, kind) in block_keys(cfg).items():
        if ours in ("sa_qg", "sa_kg") and not cfg.qk_rms_norm:
            continue
        if ours in ("ca_qg", "ca_kg") and not cfg.qk_rms_norm_cross:
            continue
        arrs = [np.asarray(sd[f"blocks.{i}.{suf}"], dtype=np.float32) for i in range(cfg.num_blocks)]
        a = jnp.asarray(np.stack(arrs, 0))
        if kind == "lin_w":
            a = jnp.swapaxes(a, 1, 2).astype(dt)
        elif kind == "lin_b":
            a = a.astype(dt)
        blocks[ours] = a
    p["blocks"] = blocks
    return p


def top_params_from_state_dict(sd: Mapping[str, np.ndarray]) -> dict:
    """Only the non-block (float32) parameters: t embedder, adaLN, input and output layers."""
    p = {}
    for ours, k in _TOP_KEYS.items():
        a = jnp.asarray(np.asarray(sd[k], dtype=np.float32))
        p[ours] = a.T if ours.endswith("_w") else a
    return p


def block_params_from_state_dict(sd: Mapping[str, np.ndarray], i: int, cfg: DiTConfig) -> dict:
    """Parameters of ONE block (no leading stack axis), cast like params_from_state_dict. Used to stream the
    30 blocks one at a time (weights fetched per block) for a block-by-block check against the original weights; on chip the stacked
    pytree from params_from_state_dict is used instead."""
    dt = cfg.jdtype
    bp = {}
    for ours, (suf, kind) in block_keys(cfg).items():
        if ours in ("sa_qg", "sa_kg") and not cfg.qk_rms_norm:
            continue
        if ours in ("ca_qg", "ca_kg") and not cfg.qk_rms_norm_cross:
            continue
        a = jnp.asarray(np.asarray(sd[f"blocks.{i}.{suf}"], dtype=np.float32))
        if kind == "lin_w":
            a = a.T.astype(dt)
        elif kind == "lin_b":
            a = a.astype(dt)
        bp[ours] = a
    return bp


def load_safetensors(path: str, keys: Optional[Sequence[str]] = None) -> dict:
    """Minimal safetensors reader (format: u64 header length, JSON header, raw little-endian data).
    BF16 is returned as ml_dtypes.bfloat16 (exact). Written from the published format description."""
    import ml_dtypes
    dmap = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16, "F64": np.float64,
            "I64": np.int64, "I32": np.int32}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__" or (keys is not None and k not in keys):
            continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(mm[s:e].tobytes(), dtype=dmap[v["dtype"]]).reshape(v["shape"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Small ops (each mirrors a vendor op; comments name it)
# ----------------------------------------------------------------------------------------------------------------
def _linear(x, w, b, out_dtype, precision):
    """nn.Linear: x @ W^T + b. f32 accumulation, result in out_dtype."""
    y = jnp.einsum("...i,io->...o", x.astype(w.dtype), w, precision=precision, preferred_element_type=F32)
    return (y + b.astype(F32)).astype(out_dtype)


def _layer_norm32(x, eps, w=None, b=None):
    """LayerNorm32: upcast to f32, LayerNorm (biased var), optional affine, cast back."""
    xd = x.dtype
    x = x.astype(F32)
    mu = jnp.mean(x, -1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), -1, keepdims=True)
    y = (x - mu) * jax.lax.rsqrt(var + eps)
    if w is not None:
        y = y * w + b
    return y.astype(xd)


def _rms_norm_heads(x, gamma):
    """MultiHeadRMSNorm: F.normalize(x.float(), dim=-1) * gamma * sqrt(D), cast back. x [..., H, D], gamma [H, D]."""
    xd = x.dtype
    xf = x.astype(F32)
    n = jnp.sqrt(jnp.sum(xf * xf, -1, keepdims=True))
    y = xf / jnp.maximum(n, 1e-12) * gamma * math.sqrt(x.shape[-1])
    return y.astype(xd)


def timestep_embedding(t, dim=256, max_period=10000.0):
    """TimestepEmbedder.timestep_embedding: [cos(t f), sin(t f)], f = exp(-ln(P) i / half)."""
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=F32) / half)
    args = t.astype(F32)[:, None] * freqs[None]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], -1)


def rope_tables(coords, head_dim, rope_freq=(1.0, 10000.0)):
    """RotaryPositionEmbedder / SparseRotaryPositionEmbedder phases for 3D integer coords [..., 3].
    freq_dim = head_dim // 2 // 3 per axis, layout [x f0..f(k-1), y ..., z ...], padded with phase 0 (identity)
    up to head_dim // 2. Returns (cos, sin) each [..., head_dim // 2] float32."""
    fd = head_dim // 2 // 3
    freqs = rope_freq[0] / (rope_freq[1] ** (jnp.arange(fd, dtype=F32) / fd))
    ph = coords.astype(F32)[..., :, None] * freqs        # [..., 3, fd]
    ph = ph.reshape(*coords.shape[:-1], 3 * fd)
    pad = head_dim // 2 - 3 * fd
    if pad:
        ph = jnp.concatenate([ph, jnp.zeros((*ph.shape[:-1], pad), F32)], -1)
    return jnp.cos(ph), jnp.sin(ph)


def apply_rope(x, cos, sin):
    """apply_rotary_embedding: view last dim as complex pairs (x[2i] + i x[2i+1]) times exp(i phase), in f32.
    x [B, L, H, D]; cos/sin [B, L, D/2] or [L, D/2]."""
    xd = x.dtype
    xf = x.astype(F32).reshape(*x.shape[:-1], -1, 2)
    a, b = xf[..., 0], xf[..., 1]
    c = cos[..., None, :]
    s = sin[..., None, :]
    if c.ndim == 3:  # [L, 1, D/2] -> broadcast over batch
        c, s = c[None], s[None]
    out = jnp.stack([a * c - b * s, a * s + b * c], -1).reshape(x.shape)
    return out.astype(xd)


def _gelu_tanh(x):
    xd = x.dtype
    return jax.nn.gelu(x.astype(F32), approximate=True).astype(xd)


# ----------------------------------------------------------------------------------------------------------------
# Attention. One signature for every backend:
#   attn(q, k, v, kv_mask) with q [B, Lq, H, D], k/v [B, Lk, H, D], kv_mask [B, Lk] bool or None -> [B, Lq, H, D]
# attention_xla below is the reference form; DiTRuntime.attn_self / attn_cross hold the functions the model calls.
# ----------------------------------------------------------------------------------------------------------------
_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)


def attention_xla(q, k, v, kv_mask=None, precision=None):
    """Exact softmax attention, scale 1/sqrt(D), logits + softmax in f32 (sdpa / flash semantics).
    kv_mask False entries are excluded (block-diagonal varlen semantics after packing one sequence per row)."""
    od = q.dtype
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision=precision, preferred_element_type=F32) * scale
    if kv_mask is not None:
        s = jnp.where(kv_mask[:, None, None, :], s, _MASK_VALUE)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum("bhqk,bkhd->bqhd", p, v.astype(F32), precision=precision, preferred_element_type=F32)
    return o.astype(od)


def _pad_axis1(a, n, value=0):
    pad = n - a.shape[1]
    if pad <= 0:
        return a
    widths = [(0, 0)] * a.ndim
    widths[1] = (0, pad)
    return jnp.pad(a, widths, constant_values=value)


def attention_chunked(q, k, v, kv_mask=None, precision=None, block_q: int = 1024, block_k: int = 1024):
    """Same math as attention_xla, computed flash-style in pure JAX: lax.scan over key blocks with an online
    softmax (running max m, running sum l, f32 accumulator), lax.map over query blocks. Peak extra memory is one
    [B, H, block_q, block_k] f32 tile instead of the [B, H, Lq, Lk] score matrix (at stage 6, 49152 tokens, the full
    matrix is 12 x 49152^2 x 4 B = 116 GB, so attention_xla cannot run stage 6 on a chip; this path can). It is plain
    JAX, the same code on CPU and on the chip.
    Pads Lq / Lk up to the block sizes internally (padded keys masked, padded query rows dropped)."""
    od = q.dtype
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    scale = 1.0 / math.sqrt(D)
    bq, bk = min(block_q, Lq), min(block_k, Lk)
    nq, nk = -(-Lq // bq), -(-Lk // bk)
    mask = jnp.ones((B, Lk), bool) if kv_mask is None else kv_mask
    qp = _pad_axis1(q, nq * bq)
    kp, vp = _pad_axis1(k, nk * bk), _pad_axis1(v, nk * bk)
    mp = _pad_axis1(mask, nk * bk, False)
    kb = jnp.moveaxis(kp.reshape(B, nk, bk, H, D), 1, 0)
    vb = jnp.moveaxis(vp.reshape(B, nk, bk, H, D), 1, 0)
    mb = jnp.moveaxis(mp.reshape(B, nk, bk), 1, 0)

    def one_q_block(qq):                                     # qq [B, bq, H, D]
        def step(carry, blk):
            m, l, acc = carry
            kk, vv, mk = blk
            s = jnp.einsum("bqhd,bkhd->bhqk", qq, kk, precision=precision, preferred_element_type=F32) * scale
            s = jnp.where(mk[:, None, None, :], s, _MASK_VALUE)
            m_new = jnp.maximum(m, jnp.max(s, -1))
            a = jnp.exp(m - m_new)
            p = jnp.exp(s - m_new[..., None])
            l = l * a + jnp.sum(p, -1)
            acc = acc * jnp.moveaxis(a, 1, 2)[..., None] + jnp.einsum(
                "bhqk,bkhd->bqhd", p, vv.astype(F32), precision=precision, preferred_element_type=F32)
            return (m_new, l, acc), None

        init = (jnp.full((B, H, bq), _MASK_VALUE, F32), jnp.zeros((B, H, bq), F32), jnp.zeros((B, bq, H, D), F32))
        (m, l, acc), _ = jax.lax.scan(step, init, (kb, vb, mb))
        return acc / jnp.moveaxis(l, 1, 2)[..., None]

    qb = jnp.moveaxis(qp.reshape(B, nq, bq, H, D), 1, 0)
    o = jax.lax.map(one_q_block, qb)                          # [nq, B, bq, H, D]
    o = jnp.moveaxis(o, 0, 1).reshape(B, nq * bq, H, D)[:, :Lq]
    return o.astype(od)


# ----------------------------------------------------------------------------------------------------------------
# Trunk
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiTRuntime:
    """Non-weight knobs. attn_self / attn_cross default to the exact XLA path."""
    attn_self: Callable = attention_xla
    attn_cross: Callable = attention_xla
    return_hidden: bool = False      # also return every block output (per-block parity), [num_blocks, B, L, C]


def modulation_vector(params, t, cfg: DiTConfig):
    """t [B] (already x1000 by the sampler) -> shared adaLN vector [B, 6C] in model dtype."""
    hp = cfg.precision if cfg.jdtype == F32 else jax.lax.Precision.HIGHEST  # this path is f32 in the vendor model
    e = timestep_embedding(t, 256)
    e = _linear(e, params["t0_w"], params["t0_b"], F32, hp)
    e = jax.nn.silu(e)
    e = _linear(e, params["t2_w"], params["t2_b"], F32, hp)
    e = _linear(jax.nn.silu(e), params["ada_w"], params["ada_b"], F32, hp)
    return e.astype(cfg.jdtype)


def _block(x, bp, mod, cond, cos, sin, kv_mask, cfg: DiTConfig, rt: DiTRuntime, proj=None):
    """ModulatedTransformerCrossBlock / ModulatedSparseTransformerCrossBlock._forward (share_mod=True).
    proj [B, L, proj_in] model dtype (optional): ProjectAttention, h = cross_attn_block(norm2(x), global)
    + proj_linear(proj); x = x + h. proj None = TRELLIS.2, the same code path as before this option existed."""
    dt, pr = cfg.jdtype, cfg.precision
    B, L, C = x.shape
    H, D = cfg.num_heads, cfg.head_dim
    m = (bp["mod"][None].astype(F32) + mod.astype(F32)).astype(dt)          # (modulation + mod).type(mod.dtype)
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = [a[:, None, :] for a in jnp.split(m, 6, axis=-1)]

    # self-attention
    h = _layer_norm32(x, 1e-6)
    h = h * (1 + sc_msa) + sh_msa
    qkv = _linear(h, bp["qkv_w"], bp["qkv_b"], dt, pr).reshape(B, L, 3, H, D)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    if cfg.qk_rms_norm:
        q = _rms_norm_heads(q, bp["sa_qg"])
        k = _rms_norm_heads(k, bp["sa_kg"])
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    h = rt.attn_self(q, k, v, kv_mask, precision=pr).reshape(B, L, C)
    h = _linear(h, bp["sa_o_w"], bp["sa_o_b"], dt, pr)
    x = x + h * g_msa

    # cross-attention to the condition (dense, unmasked; kv length 1029 or 4101)
    h = _layer_norm32(x, 1e-6, bp["n2_w"], bp["n2_b"])
    q = _linear(h, bp["q_w"], bp["q_b"], dt, pr).reshape(B, L, H, D)
    kv = _linear(cond, bp["kv_w"], bp["kv_b"], dt, pr).reshape(B, cond.shape[1], 2, H, D)
    k, v = kv[:, :, 0], kv[:, :, 1]
    if cfg.qk_rms_norm_cross:
        q = _rms_norm_heads(q, bp["ca_qg"])
        k = _rms_norm_heads(k, bp["ca_kg"])
    h = rt.attn_cross(q, k, v, None, precision=pr).reshape(B, L, C)
    if proj is None:
        x = x + _linear(h, bp["ca_o_w"], bp["ca_o_b"], dt, pr)
    else:   # vendor order: x + (proj_linear(proj) + global_out); fp add is commutative, so only the grouping counts
        x = x + (_linear(h, bp["ca_o_w"], bp["ca_o_b"], dt, pr) + _linear(proj, bp["pj_w"], bp["pj_b"], dt, pr))

    # MLP
    h = _layer_norm32(x, 1e-6)
    h = h * (1 + sc_mlp) + sh_mlp
    h = _gelu_tanh(_linear(h, bp["fc1_w"], bp["fc1_b"], dt, pr))
    h = _linear(h, bp["fc2_w"], bp["fc2_b"], dt, pr)
    return x + h * g_mlp


def _check_proj(proj, cfg: DiTConfig):
    """A proj-mode model run without proj (or the reverse) would silently drop or invent a term: refuse."""
    if (proj is None) != (cfg.proj_in_channels is None):
        raise ValueError(f"proj given={proj is not None} but cfg.proj_in_channels={cfg.proj_in_channels}")
    if proj is not None and proj.shape[-1] != cfg.proj_in_channels:
        raise ValueError(f"proj width {proj.shape[-1]} != cfg.proj_in_channels {cfg.proj_in_channels}")


def trunk(params, h, mod, cond, coords, kv_mask, cfg: DiTConfig, rt: DiTRuntime = DiTRuntime(), proj=None):
    """The shared 30-block stack. h [B, L, C] model dtype; mod [B, 6C]; cond [B, Lc, Ccond];
    coords [B, L, 3] or [L, 3] int; kv_mask [B, L] bool or None. lax.scan over stacked blocks (compiles once).
    proj [B, L, proj_in] (proj mode only, same token order as h), cast to model dtype like cond."""
    _check_proj(proj, cfg)
    dt = cfg.jdtype
    cos, sin = rope_tables(coords, cfg.head_dim, cfg.rope_freq)
    cond = cond.astype(dt)
    mod = mod.astype(dt)
    proj = None if proj is None else proj.astype(dt)

    def step(x, bp):
        y = _block(x, bp, mod, cond, cos, sin, kv_mask, cfg, rt, proj)
        return y, (y if rt.return_hidden else None)

    h, hs = jax.lax.scan(step, h.astype(dt), params["blocks"])
    return h, hs


def _head(params, h, out_dtype, cfg):
    hp = cfg.precision if cfg.jdtype == F32 else jax.lax.Precision.HIGHEST
    h = _layer_norm32(h.astype(out_dtype), 1e-5)                              # F.layer_norm default eps
    return _linear(h, params["out_w"], params["out_b"], out_dtype, hp)


def dense_grid_coords(res: int):
    g = np.stack(np.meshgrid(*[np.arange(res)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return jnp.asarray(g, jnp.int32)


def ss_embed(params, x, cfg: DiTConfig):
    """SparseStructureFlowModel.forward up to the blocks: x [B, Cin, R, R, R] -> h [B, R^3, C] in x.dtype
    (the trunk casts to model dtype, = manual_cast(h, self.dtype))."""
    B, Cin = x.shape[0], x.shape[1]
    hp = cfg.precision if cfg.jdtype == F32 else jax.lax.Precision.HIGHEST
    h = jnp.transpose(x.reshape(B, Cin, -1), (0, 2, 1))
    return _linear(h, params["in_w"], params["in_b"], x.dtype, hp)


def ss_head(params, h, x_dtype, R, cfg: DiTConfig):
    """manual_cast back, F.layer_norm, out_layer, back to [B, Cout, R, R, R]."""
    B = h.shape[0]
    h = _head(params, h, x_dtype, cfg)
    return jnp.transpose(h, (0, 2, 1)).reshape(B, -1, R, R, R)


def slat_embed(params, feats, cfg: DiTConfig, concat_feats=None):
    """SLatFlowModel.forward up to the blocks (sparse_cat on channels, input_layer). [B, Lpad, Cin] -> [B, Lpad, C]."""
    if concat_feats is not None:
        feats = jnp.concatenate([feats, concat_feats.astype(feats.dtype)], -1)
    hp = cfg.precision if cfg.jdtype == F32 else jax.lax.Precision.HIGHEST
    return _linear(feats, params["in_w"], params["in_b"], feats.dtype, hp)


def slat_head(params, h, mask, feats_dtype, cfg: DiTConfig):
    out = _head(params, h, feats_dtype, cfg)
    return jnp.where(mask[..., None], out, 0)


def block_forward(x, bp, mod, cond, coords, kv_mask, cfg: DiTConfig, rt: DiTRuntime = DiTRuntime(), proj=None):
    """ONE block with unstacked params bp (streaming / per-block tests). Same _block the scan runs."""
    _check_proj(proj, cfg)
    dt = cfg.jdtype
    cos, sin = rope_tables(coords, cfg.head_dim, cfg.rope_freq)
    proj = None if proj is None else proj.astype(dt)
    return _block(x.astype(dt), bp, mod.astype(dt), cond.astype(dt), cos, sin, kv_mask, cfg, rt, proj)


def ss_flow_forward(params, x, t, cond, cfg: DiTConfig, rt: DiTRuntime = DiTRuntime(), proj=None):
    """SparseStructureFlowModel.forward. x [B, Cin, R, R, R] f32, t [B] (x1000), cond [B, Lc, 1024].
    proj [B, R^3, proj_in] (proj mode only): token order = x.reshape(B, Cin, -1) order, i.e. the
    vendor's dense proj cond as passed to the blocks. Returns [B, Cout, R, R, R] (and hidden states if
    rt.return_hidden)."""
    R = x.shape[2]
    h = ss_embed(params, x, cfg)
    mod = modulation_vector(params, t, cfg)
    h, hs = trunk(params, h, mod, cond, dense_grid_coords(R), None, cfg, rt, proj)
    out = ss_head(params, h, x.dtype, R, cfg)
    return (out, hs) if rt.return_hidden else out


def slat_flow_forward(params, feats, coords, mask, t, cond, cfg: DiTConfig, concat_feats=None,
                      rt: DiTRuntime = DiTRuntime(), proj=None):
    """SLatFlowModel.forward on a packed batch. feats [B, Lpad, Cin] f32, coords [B, Lpad, 3] int (xyz, no
    batch column), mask [B, Lpad] bool (valid tokens), t [B] (x1000), cond [B, Lc, 1024].
    concat_feats [B, Lpad, Cc] = concat_cond feats (tex flow: the shape slat), concatenated on channels like
    sp.sparse_cat(dim=-1). proj [B, Lpad, proj_in] (proj mode only) packed like feats (pack_varlen).
    Returns [B, Lpad, Cout] (padding rows are zeroed; use unpack_varlen)."""
    h = slat_embed(params, feats, cfg, concat_feats)
    mod = modulation_vector(params, t, cfg)
    h, hs = trunk(params, h, mod, cond, coords, mask, cfg, rt, proj)
    out = slat_head(params, h, mask, feats.dtype, cfg)
    return (out, hs) if rt.return_hidden else out


# ----------------------------------------------------------------------------------------------------------------
# Varlen packing (host side): SparseTensor (coords [T, 4] = batch, x, y, z; feats [T, C]) <-> padded batch
# ----------------------------------------------------------------------------------------------------------------
def bucket_for(n: int, granule: int = 512, steps_per_octave: int = 8, cap: Optional[int] = None) -> int:
    """Padded length for n tokens: a multiple of `granule` and of 2^floor(log2 n) / steps_per_octave, so padding
    is at most ~1/steps_per_octave of n (12.5% at 8) and the number of compiled shapes stays ~8 per octave.
    The 1024_cascade pipeline caps hr tokens at 49152 = 96 x 512, which is itself a bucket."""
    n = max(int(n), 1)
    step = max(granule, (1 << (n.bit_length() - 1)) // steps_per_octave)
    step = ((step + granule - 1) // granule) * granule
    b = ((n + step - 1) // step) * step
    if cap is not None and n <= cap:
        b = min(b, ((cap + granule - 1) // granule) * granule)
    return b


def pack_varlen(coords_b: np.ndarray, feats: np.ndarray, batch_size: Optional[int] = None,
                lpad: Optional[int] = None, **bucket_kw):
    """coords_b [T, 4] int (batch index first, rows of a batch contiguous as in SparseTensor), feats [T, C].
    Returns (feats_p [B, Lpad, C], coords_p [B, Lpad, 3] int32, mask [B, Lpad] bool, seqlens list)."""
    coords_b = np.asarray(coords_b)
    feats = np.asarray(feats)
    B = int(batch_size if batch_size is not None else coords_b[:, 0].max() + 1)
    seqlens = [int((coords_b[:, 0] == i).sum()) for i in range(B)]
    L = lpad if lpad is not None else bucket_for(max(seqlens), **bucket_kw)
    assert max(seqlens) <= L
    fp = np.zeros((B, L, feats.shape[1]), feats.dtype)
    cp = np.zeros((B, L, 3), np.int32)
    mk = np.zeros((B, L), bool)
    for i in range(B):
        sel = coords_b[:, 0] == i
        n = seqlens[i]
        fp[i, :n] = feats[sel]
        cp[i, :n] = coords_b[sel, 1:]
        mk[i, :n] = True
    return fp, cp, mk, seqlens


def unpack_varlen(padded: np.ndarray, seqlens: Sequence[int]) -> np.ndarray:
    """[B, Lpad, C] -> [T, C] in SparseTensor batch-contiguous order."""
    padded = np.asarray(padded)
    return np.concatenate([padded[i, :n] for i, n in enumerate(seqlens)], 0)
