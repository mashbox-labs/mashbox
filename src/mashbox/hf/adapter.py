"""A Hugging Face model folder presented as the GGUF File the architecture modules read (mashbox/gguf/archs).

The HF path has no forward of its own: config.json becomes the GGUF metadata keys, the safetensors tensors get the GGUF
names, and the few weights llama.cpp's converter (convert_hf_to_gguf.py, MIT, github.com/ggml-org/llama.cpp) changes on
the way into a GGUF file are changed the same way here, in memory:

  llama (LlamaForCausalLM, MistralForCausalLM)  q/k rows permuted for rope on adjacent pairs (the converter's `permute`);
                                                 Llama 3 rope scaling becomes the `rope_freqs` factors
  qwen2 (Qwen2ForCausalLM), qwen3 (Qwen3ForCausalLM)  names only
  qwen35 (Qwen3_5ForConditionalGeneration / Qwen3_5ForCausalLM, text model only)  RMSNorm weights + 1 (not the
                                                 linear-attention output norm), ssm_a = -exp(A_log), conv1d squeezed,
                                                 linear-attention value heads from grouped to tiled order

Anything this map does not cover stops with one line naming the config field or tensor.
"""
import math

import numpy as np

from .source import HFError

TYPE = {"F32": 0, "F16": 1, "BF16": 30}
ITEM = {"F32": 4, "F16": 2, "BF16": 2}

ARCHITECTURES = {
    "LlamaForCausalLM": "llama", "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen2", "Qwen3ForCausalLM": "qwen3",
    "Qwen3_5ForConditionalGeneration": "qwen35", "Qwen3_5ForCausalLM": "qwen35",
}
TEXT_TYPES = {"qwen35": ("qwen3_5_text", "qwen3_5")}
MOE_FIELDS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size")
SKIP_PREFIXES = ("model.visual.", "visual.", "mtp.", "model.mtp.")


class Tensor:
    """the gguf.reader.Tensor interface (name, type, shape, nbytes, raw, rows()) with the bytes made on first use"""

    __slots__ = ("name", "type", "shape", "nbytes", "make")

    def __init__(self, name, t, shape, nbytes, make):
        self.name, self.type, self.shape, self.nbytes, self.make = name, int(t), tuple(shape), int(nbytes), make

    @property
    def raw(self):
        """the bytes, made on each use and not kept (the loader stacks and sends them to the device once)"""
        return self.make()

    @property
    def type_name(self):
        from ..ops import reference as R
        return R.TYPES[self.type][0]

    def rows(self):
        from ..ops import reference as R
        rb = R.row_bytes(self.type, self.shape[-1])
        return self.raw.reshape(tuple(self.shape[:-1]) + (rb,))

    def __repr__(self):
        return f"Tensor({self.name}, {self.type_name}, {self.shape})"


def text_config(cfg):
    """-> (architecture name, the text model's config) or a refusal"""
    archs = cfg.get("architectures") or []
    if not archs:
        raise HFError("mashbox: config.json has no `architectures` entry")
    a = archs[0]
    if a not in ARCHITECTURES:
        raise HFError(f'mashbox: config.json architectures "{a}" is not read by the Hugging Face path in this version '
                      f"(it reads {', '.join(sorted(ARCHITECTURES))})")
    arch = ARCHITECTURES[a]
    tc = cfg.get("text_config") or cfg if arch == "qwen35" else cfg
    if arch == "qwen35" and tc.get("model_type") not in TEXT_TYPES["qwen35"]:
        raise HFError(f'mashbox: config.json text_config.model_type "{tc.get("model_type")}" is not read by the Hugging Face '
                      f"path (models with routed experts do not run from Hugging Face safetensors in this version)")
    for k in MOE_FIELDS:
        if tc.get(k):
            raise HFError(f"mashbox: config.json {k} = {tc[k]}: models with routed experts do not run from Hugging Face "
                          f"safetensors in this version")
    return arch, tc


def _rope(tc):
    """-> (theta, scaling dict or None) from rope_parameters (transformers 5) or rope_theta + rope_scaling"""
    rp = tc.get("rope_parameters") or {}
    theta = rp.get("rope_theta", tc.get("rope_theta", 10000.0))
    sc = tc.get("rope_scaling") or None
    if rp.get("rope_type") not in (None, "default"):
        sc = rp
    return float(theta), sc


def _no(field, value, arch):
    return HFError(f"mashbox: config.json {field} = {value!r} is not supported for {arch} in this version")


def metadata(arch, tc, n_vocab):
    """config.json -> the GGUF keys `<arch>.*` the architecture module reads"""
    A = arch
    H = int(tc["num_attention_heads"])
    KV = int(tc.get("num_key_value_heads") or H)
    D = int(tc["hidden_size"])
    HD = int(tc.get("head_dim") or D // H)
    act = tc.get("hidden_act", tc.get("hidden_activation", "silu"))
    if act != "silu":
        raise _no("hidden_act", act, arch)
    if tc.get("quantization_config"):
        raise _no("quantization_config", tc["quantization_config"].get("quant_method", "set"), arch)
    if tc.get("mlp_bias"):
        raise _no("mlp_bias", True, arch)
    if tc.get("use_sliding_window"):
        raise _no("use_sliding_window", True, arch)
    lt = tc.get("layer_types") or []
    if arch != "qwen35" and any(t != "full_attention" for t in lt):
        raise _no("layer_types", sorted(set(lt)), arch)
    if arch == "llama" and tc.get("sliding_window"):
        raise _no("sliding_window", tc["sliding_window"], arch)
    theta, sc = _rope(tc)
    m = {"general.architecture": A, "general.name": tc.get("_name_or_path")}
    k = lambda s, v: m.__setitem__(f"{A}.{s}", v)
    k("block_count", int(tc["num_hidden_layers"]))
    k("embedding_length", D)
    k("feed_forward_length", int(tc["intermediate_size"]))
    k("attention.head_count", H)
    k("attention.head_count_kv", KV)
    k("attention.key_length", HD)
    k("attention.value_length", HD)
    k("attention.layer_norm_rms_epsilon", float(tc.get("rms_norm_eps", 1e-6)))
    k("rope.freq_base", theta)
    k("context_length", int(tc.get("max_position_embeddings") or 0))
    k("vocab_size", n_vocab)
    rt = (sc or {}).get("rope_type", (sc or {}).get("type"))
    freqs = None
    if arch == "qwen35":
        prf = float((tc.get("rope_parameters") or {}).get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
        k("rope.dimension_count", int(HD * prf))
        if rt not in (None, "default"):
            raise _no("rope type", rt, arch)
        k("ssm.group_count", int(tc["linear_num_key_heads"]))
        k("ssm.time_step_rank", int(tc["linear_num_value_heads"]))
        k("ssm.state_size", int(tc["linear_key_head_dim"]))
        k("ssm.inner_size", int(tc["linear_value_head_dim"]) * int(tc["linear_num_value_heads"]))
        k("ssm.conv_kernel", int(tc["linear_conv_kernel_dim"]))
        FI = int(tc.get("full_attention_interval") or 4)
        k("full_attention_interval", FI)
        if lt:
            k("attention.recurrent_layers", [t == "linear_attention" for t in lt])
        k("nextn_predict_layers", 0)              # the MTP layer is not read on this path
    else:
        if rt in (None, "default"):
            pass
        elif rt == "linear":
            k("rope.scaling.type", "linear")
            k("rope.scaling.factor", float(sc["factor"]))
        elif rt == "llama3" and arch == "llama":
            freqs = llama3_factors(theta, HD, sc)
        else:
            raise _no("rope type", rt, arch)
    return m, freqs


def llama3_factors(theta, dim, sc):
    """llama.cpp convert_hf_to_gguf LlamaModel.generate_extra_tensors: one factor per rotated pair (float32)"""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    factor = float(sc.get("factor", 8.0))
    lo, hi = float(sc.get("low_freq_factor", 1.0)), float(sc.get("high_freq_factor", 4.0))
    old = float(sc.get("original_max_position_embeddings", 8192))
    lo_w, hi_w = old / lo, old / hi
    out = []
    for f in freqs:
        w = 2 * math.pi / float(f)
        if w < hi_w:
            out.append(1.0)
        elif w > lo_w:
            out.append(factor)
        else:
            s = (old / w - lo) / (hi - lo)
            out.append(1.0 / ((1 - s) / factor + s))
    return np.asarray(out, np.float32)


# ------------------------------------------------------------------------------------------------------- tensors
DENSE = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
         "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
QWEN35 = dict(DENSE, **{"post_attention_layernorm": "post_attention_norm", "linear_attn.in_proj_qkv": "attn_qkv",
                        "linear_attn.in_proj_z": "attn_gate", "linear_attn.in_proj_b": "ssm_beta",
                        "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.conv1d": "ssm_conv1d",
                        "linear_attn.norm": "ssm_norm", "linear_attn.out_proj": "ssm_out"})
QWEN35_PARAMS = {"linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.A_log": "ssm_a"}
GLOBAL = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight"}


def _strip(name):
    for p in ("model.language_model.", "language_model.model.", "model."):
        if name.startswith(p):
            return name[len(p):], True
    return name, False


def gguf_name(arch, name):
    """HF tensor name -> GGUF name, or None for a tensor this path does not read (vision tower, MTP layer, rotary
    buffers); a refusal for anything else"""
    if name.startswith(SKIP_PREFIXES) or name.endswith("rotary_emb.inv_freq"):
        return None
    if name == "lm_head.weight":
        return "output.weight"
    s, had = _strip(name)
    if s in GLOBAL and had:
        return GLOBAL[s]
    parts = s.split(".")
    if had and len(parts) >= 4 and parts[0] == "layers" and parts[1].isdigit():
        role, tail = ".".join(parts[2:-1]), parts[-1]
        table = QWEN35 if arch == "qwen35" else DENSE
        if role in table and tail in ("weight", "bias"):
            return f"blk.{parts[1]}.{table[role]}.{tail}"
        if arch == "qwen35" and f"{role}.{tail}" in QWEN35_PARAMS:
            return f"blk.{parts[1]}.{QWEN35_PARAMS[role + '.' + tail]}"
    raise HFError(f"mashbox: tensor {name} has no place in the {arch} map of this version")


def _llama_permute(a, n_head):
    """llama.cpp convert_hf_to_gguf LlamaModel.permute on rows: (n_head, 2, rows/n_head/2, ...) -> swap the two middle axes"""
    return a.reshape((n_head, 2, a.shape[0] // n_head // 2) + a.shape[1:]).swapaxes(1, 2).reshape(a.shape)


def _bits(dt):
    return {2: np.uint16, 4: np.uint32}[ITEM[dt]]


def _f32(dt, raw, shape):
    """exact float32 values of a BF16 / F16 / F32 tensor"""
    if dt == "F32":
        return np.asarray(raw).view(np.float32).reshape(shape).copy()
    if dt == "F16":
        return np.asarray(raw).view(np.float16).reshape(shape).astype(np.float32)
    return (np.asarray(raw).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


class File:
    """The GGUF File interface over a Hugging Face model folder: .arch, .get/.keys (GGUF metadata keys), .tensors (GGUF
    names, the safetensors dtype as the ggml type), .census(), .unreadable(), .path, .close()."""

    def __init__(self, path, cfg, src):
        self.path, self.src, self.cfg = path, src, cfg
        self.arch, tc = text_config(cfg)
        self.text = tc
        names = {n: gguf_name(self.arch, n) for n in src.tensors}
        emb = [n for n, g in names.items() if g == "token_embd.weight"]
        if not emb:
            raise HFError("mashbox: the model has no embed_tokens.weight tensor")
        V = int(src.tensors[emb[0]][1][0])
        self.meta, freqs = metadata(self.arch, tc, V)
        self.H, self.KV = self.meta[f"{self.arch}.attention.head_count"], self.meta[f"{self.arch}.attention.head_count_kv"]
        tie = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))
        self.tensors = {}
        for n, g in names.items():
            if g is None or (g == "output.weight" and tie):
                continue
            dt, shape, raw = src.tensors[n]
            self._add(g, n, dt, shape, raw)
        if freqs is not None:
            self.tensors["rope_freqs.weight"] = Tensor("rope_freqs.weight", 0, freqs.shape, freqs.nbytes,
                                                       lambda v=freqs: v.view(np.uint8))
        self.size = sum(t.nbytes for t in self.tensors.values())

    # GGUF File interface
    def get(self, key, default=None):
        return self.meta.get(key, default)

    def keys(self):
        return list(self.meta)

    def census(self):
        from ..gguf.reader import File as G
        return G.census(self)

    def unreadable(self):
        return []

    def close(self):
        self.src.close()

    # ------------------------------------------------------------------------------------------------------- maps
    def _add(self, g, hf, dt, shape, raw):
        A, t = self.arch, TYPE[dt]
        nb = int(np.prod(shape, dtype=np.int64)) * ITEM[dt]
        same = lambda: np.asarray(raw)
        role = g.split(".")[2] if g.startswith("blk.") else g
        if A == "llama" and role in ("attn_q", "attn_k"):
            nh = self.H if role == "attn_q" else self.KV
            make = lambda: np.ascontiguousarray(_llama_permute(np.asarray(raw).view(_bits(dt)).reshape(shape), nh)).view(np.uint8).reshape(-1)
        elif A == "qwen35":
            make, t, shape, nb = self._qwen35(g, role, dt, shape, raw, t, nb)
        else:
            make = same
        self.tensors[g] = Tensor(g, t, shape, nb, make)

    def _qwen35(self, g, role, dt, shape, raw, t, nb):
        """the converter's Qwen3.5 changes; -> (make, type, shape, nbytes)"""
        A = self.arch
        HK, HV = self.meta[f"{A}.ssm.group_count"], self.meta[f"{A}.ssm.time_step_rank"]
        DK, DV = self.meta[f"{A}.ssm.state_size"], self.meta[f"{A}.ssm.inner_size"] // HV
        rep, KD = HV // HK, HK * DK
        # GGUF value head j = HF value head (j % HK) * rep + j // HK  (tiled: head j reads key head j % HK)
        perm = np.asarray([(j % HK) * rep + j // HK for j in range(HV)])
        bits = lambda: np.asarray(raw).view(_bits(dt)).reshape(shape)
        u8 = lambda a: np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        f32n = int(np.prod(shape, dtype=np.int64)) * 4
        if role in ("attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "output_norm.weight"):
            return (lambda: u8(_f32(dt, raw, shape) + np.float32(1.0))), 0, shape, f32n
        if role == "ssm_a":
            return (lambda: u8(-np.exp(_f32(dt, raw, shape))[perm])), 0, shape, f32n
        if role in ("ssm_dt", "ssm_beta", "ssm_alpha"):           # one row (or value) per value head
            return (lambda: u8(bits()[perm])), t, shape, nb
        if role == "attn_gate":                                      # (HV * DV, D) rows by value head
            return (lambda: u8(bits().reshape((HV, DV) + shape[1:])[perm].reshape(shape))), t, shape, nb
        if role == "ssm_out":                                        # (D, HV * DV) columns by value head
            return (lambda: u8(bits().reshape(shape[0], HV, DV)[:, perm].reshape(shape))), t, shape, nb
        if role in ("attn_qkv", "ssm_conv1d"):                       # q | k | v rows, v by value head
            sh = (shape[0], int(np.prod(shape[1:])))

            def make():
                a = bits().reshape(sh)
                v = a[2 * KD:].reshape((HV, DV, sh[1]))[perm].reshape(HV * DV, sh[1])
                return u8(np.concatenate([a[:2 * KD], v], 0))
            return make, t, sh, nb
        return (lambda: np.asarray(raw)), t, shape, nb
