"""Mixture of experts or not: read from the model's own metadata and tensor names, never from its name.

  detect_gguf(meta, tensor_names, arch) -> MoESpec | None     meta(key, default) reads the GGUF metadata
  detect_hf(config, tensor_names=None) -> MoESpec | None      config = the Hugging Face config.json as a dict

A GGUF file is MoE when `<arch>.expert_count` > 0 and its layers carry `ffn_*_exps` tensors; one without the other is
refused, naming what is missing. What the metadata does not say comes from llama.cpp's graph for that
`general.architecture` (ARCH_DEFAULTS, each with its file:line in llama.cpp a37cc46), because llama.cpp itself
hard-codes those values per architecture: a file runs here the way it runs there. A value we do not understand is
refused with a message that names the field; nothing is guessed.
"""
from dataclasses import dataclass

LLAMA_CPP = "llama.cpp a37cc46"
CLAMP = 6.103515625e-05                 # llama.cpp build_moe_ffn: renormalised weights divide by max(sum, this)

# llama.cpp src/llama-hparams.h:16-22 (llama_expert_gating_func_type)
GATING = {1: "softmax", 2: "sigmoid", 3: "softmax_weight"}
GATING_NAMES = {0: "none", 4: "sqrt_softplus"}      # known to llama.cpp, not run here: refused by name

# per architecture, what llama.cpp's graph passes to build_moe_ffn (src/models/<file>:<lines> at a37cc46). A value of
# META means llama.cpp reads it from the file's metadata there (hparams.*), so we do too.
META = "metadata"
ARCH_DEFAULTS = {
    "qwen35moe": dict(gating="softmax", norm_w=True, act="silu", src="src/models/qwen35moe.cpp:498-502"),
    "qwen3moe": dict(gating="softmax", norm_w=True, act="silu", src="src/models/qwen3moe.cpp:142-146"),
    "llama": dict(gating="softmax", norm_w=True, act="silu", src="src/models/llama.cpp:208-212"),
    "olmoe": dict(gating="softmax", norm_w=False, act="silu", src="src/models/olmoe.cpp:141-145"),
    "qwen2moe": dict(gating="softmax", norm_w=False, act="silu", src="src/models/qwen2moe.cpp:137-141"),
    "gpt-oss": dict(gating="softmax_weight", norm_w=False, act="swiglu_oai", src="src/models/openai-moe.cpp:137-142"),
    "deepseek2": dict(gating=META, norm_w=META, act="silu", src="src/models/deepseek2.cpp:358-362"),
}


class MoEError(ValueError):
    """a MoE layout this version cannot read; the message names the field or tensor"""


@dataclass(frozen=True)
class MoESpec:
    n_expert: int            # E
    n_used: int              # K
    expert_ff: int           # rows of one expert's gate/up
    gating: str              # "softmax" | "sigmoid" | "softmax_weight"
    norm_w: bool             # renormalise the K weights to sum 1 (clamped at CLAMP)
    w_scale: float           # multiply the K weights (1.0 = none)
    n_group: int             # 1 = no group routing
    n_group_used: int
    sel_bias: bool           # selection-only bias added to the scores before top-K (exp_probs_b)
    router_bias: bool        # router (gate_inp) bias
    expert_bias: bool        # per-expert gate/up/down biases
    act: str                 # "silu" | "gelu" | "swiglu_oai" (gated) | "relu2" (ungated: up / down only, NemotronH)
    fused_gate_up: bool      # one gate|up expert tensor
    n_shared: int            # shared experts (0 = none)
    shared_ff: int
    shared_gate: bool        # sigmoid gate on the shared expert
    moe_layers: tuple        # per layer: True = routed experts, False = a dense FFN
    source: tuple            # ((field, where it came from), ...) for `mashbox inspect`

    def describe(self):
        """one short line: what `mashbox inspect` prints"""
        sh = f", {self.n_shared} shared expert{'s' if self.n_shared > 1 else ''}" if self.n_shared else ""
        grp = f", groups {self.n_group_used} of {self.n_group}" if self.n_group > 1 else ""
        dense = sum(1 for m in self.moe_layers if not m)
        dl = f", {dense} dense layer{'s' if dense > 1 else ''}" if dense else ""
        return (f"mixture of experts: {self.n_used} of {self.n_expert} experts per token{sh}{grp}, {self.gating} router"
                f"{', weights renormalised' if self.norm_w else ''}{dl}")

    def as_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("source", "moe_layers")}
        d["moe_layers"] = sum(1 for m in self.moe_layers if m)
        d["dense_layers"] = sum(1 for m in self.moe_layers if not m)
        d["source"] = dict(self.source)
        return d


def _layers(names):
    out = {}
    for n in names:
        if n.startswith("blk."):
            p = n.split(".", 2)
            if len(p) == 3 and p[1].isdigit():
                out.setdefault(int(p[1]), set()).add(p[2])
    return out


def _deepseek2_absent_gating(k, names, shapes):
    """llama.cpp a37cc46 src/models/deepseek2.cpp:23-31, for a 'deepseek2' file with no (or a none) expert_gating_func:
    softmax (1), except when n_layer (block_count - nextn_predict_layers, src/llama-hparams.cpp:348-350) is 47 or 48 and the
    vocabulary has 154,880 tokens: sigmoid (2). The vocabulary is token_embd.weight's row count; a 47 or 48 layer file
    whose shapes are not given is refused, since the rule cannot be read without it."""
    nl = int(k("block_count", 0) or 0) - int(k("nextn_predict_layers", 0) or 0)
    if nl not in (47, 48):
        return 1
    if shapes is None or "token_embd.weight" not in names:
        raise MoEError(f"mashbox: deepseek2.expert_gating_func is absent in a {nl}-layer file and llama.cpp then picks "
                       f"the router from the vocabulary size, which is not known here (no token_embd.weight shape)")
    return 2 if int(shapes("token_embd.weight")[0]) == 154880 else 1


def detect_gguf(meta, tensor_names, arch, shapes=None, n_layers=None):
    """-> MoESpec, or None for a dense file. `meta(key, default)` reads metadata, `tensor_names` lists the tensors;
    `shapes` (optional) maps a name to its shape as the reader gives it (ggml dims reversed: experts first)."""
    k = lambda s, d=None: meta(f"{arch}.{s}", d)
    names = list(tensor_names)
    per = _layers(names)
    L = int(n_layers if n_layers is not None else (k("block_count", 0) or (max(per) + 1 if per else 0)))
    E = int(k("expert_count", 0) or 0)
    exps = sorted(i for i, ts in per.items() if any("_exps." in t for t in ts))
    if E <= 0 and not exps:
        return None
    if E <= 0:
        raise MoEError(f"mashbox: {arch} file with routed experts (blk.{exps[0]}.ffn_*_exps) but no {arch}.expert_count; "
                       f"the file is incomplete")
    if not exps:
        raise MoEError(f"mashbox: {arch}.expert_count is {E} but the file has no ffn_*_exps expert tensors")
    odd = sorted(t for t in names if ("_exps." in t or "_shexp." in t) and not t.endswith((".weight", ".bias")))
    if odd:
        raise MoEError(f"mashbox: {odd[0]} ({len(odd)} expert tensor{'s' if len(odd) > 1 else ''} other than .weight "
                       f"and .bias, e.g. llama.cpp's per-expert .scale): this version does not read them, so the file "
                       f"would run without them")
    src = [("n_expert", f"metadata {arch}.expert_count")]
    K = int(k("expert_used_count", 0) or 0)
    if K <= 0:
        raise MoEError(f"mashbox: {arch}.expert_used_count is missing or 0 for a file with {E} experts")
    if K > E:
        raise MoEError(f"mashbox: {arch}.expert_used_count = {K} is more than {arch}.expert_count = {E}")
    src.append(("n_used", f"metadata {arch}.expert_used_count"))
    d = ARCH_DEFAULTS.get(arch)
    if d is None:
        raise MoEError(f"mashbox: routed experts in a '{arch}' file: this version has no routing rule for that "
                       f"architecture (llama.cpp's graph for it is not ported)")

    gf = k("expert_gating_func")
    if d["gating"] == META:
        g = int(gf or 0)
        if g == 0 and arch == "deepseek2":
            g = _deepseek2_absent_gating(k, names, shapes)
            src.append(("gating", f"absent: {GATING[g]} ({LLAMA_CPP} src/models/deepseek2.cpp:23-31)"))
        if g not in GATING:
            raise MoEError(f"mashbox: {arch}.expert_gating_func = {g} ({GATING_NAMES.get(g, 'unknown')}) is not "
                           f"supported in this version")
        gating = GATING[g]
        if not any(f == "gating" for f, _ in src):
            src.append(("gating", f"metadata {arch}.expert_gating_func"))
    else:
        if gf is not None and int(gf) not in GATING and int(gf) != 0:
            raise MoEError(f"mashbox: {arch}.expert_gating_func = {int(gf)} is not a value llama.cpp knows")
        gating = d["gating"]
        src.append(("gating", f"arch default {LLAMA_CPP} {d['src']}"))
    if d["norm_w"] == META:
        norm_w = bool(k("expert_weights_norm", False))
        src.append(("norm_w", f"metadata {arch}.expert_weights_norm"))
    else:
        norm_w = d["norm_w"]
        src.append(("norm_w", f"arch default {LLAMA_CPP} {d['src']}"))
    ws = float(k("expert_weights_scale", 0.0) or 0.0)
    w_scale = 1.0 if ws == 0.0 else ws
    src.append(("w_scale", f"metadata {arch}.expert_weights_scale" if ws else "absent: 1.0"))
    ng, ngu = int(k("expert_group_count", 0) or 0), int(k("expert_group_used_count", 0) or 0)
    if ng > 1:
        if E % ng:
            raise MoEError(f"mashbox: {arch}.expert_group_count = {ng} does not divide expert_count = {E}")
        if not 0 < ngu <= ng:
            raise MoEError(f"mashbox: {arch}.expert_group_used_count = {ngu} is not in 1..{ng}")
        src.append(("n_group", f"metadata {arch}.expert_group_count"))
    else:
        ng = ngu = 1

    moe_layers = tuple(i in per and "ffn_gate_inp.weight" in per[i] for i in range(L))
    lead = int(k("leading_dense_block_count", 0) or 0)
    for i in range(L):
        ts = per.get(i, set())
        has_exp = any("_exps." in t for t in ts)
        if has_exp != moe_layers[i]:
            what = "ffn_*_exps without ffn_gate_inp" if has_exp else "ffn_gate_inp without ffn_*_exps"
            raise MoEError(f"mashbox: layer {i} has {what}; a routed layer needs both")
        if i < lead and moe_layers[i]:
            raise MoEError(f"mashbox: {arch}.leading_dense_block_count = {lead} but layer {i} has routed experts")
    first = min(i for i in range(L) if moe_layers[i])
    t = per[first]
    fused = "ffn_gate_up_exps.weight" in t
    if fused and ("ffn_gate_exps.weight" in t or "ffn_up_exps.weight" in t):
        raise MoEError(f"mashbox: layer {first} has both a fused ffn_gate_up_exps and separate gate/up expert tensors")
    if not fused and not {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} <= t:
        miss = sorted({"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} - t)
        raise MoEError(f"mashbox: layer {first} is missing {', '.join('blk.%d.%s' % (first, m) for m in miss)}")
    if fused and "ffn_down_exps.weight" not in t:
        raise MoEError(f"mashbox: layer {first} is missing blk.{first}.ffn_down_exps.weight")

    ff_meta = int(k("expert_feed_forward_length", 0) or 0)
    ff = ff_meta
    if shapes is not None:
        for i in (i for i in range(L) if moe_layers[i]):
            up = shapes(f"blk.{i}.ffn_{'gate_up' if fused else 'up'}_exps.weight")
            if int(up[0]) != E:
                raise MoEError(f"mashbox: blk.{i}.ffn_{'gate_up' if fused else 'up'}_exps.weight holds {int(up[0])} "
                               f"experts, not {arch}.expert_count = {E}")
            rows = int(up[-2]) // (2 if fused else 1)
            if ff and rows != ff:
                raise MoEError(f"mashbox: blk.{i} experts have {rows} rows but {arch}.expert_feed_forward_length = {ff}")
            ff = rows
    if not ff:
        raise MoEError(f"mashbox: {arch}.expert_feed_forward_length is missing and the expert shapes were not given")
    src.append(("expert_ff", f"metadata {arch}.expert_feed_forward_length" if ff_meta else "expert tensor shape"))

    shexp = "ffn_up_shexp.weight" in t
    n_shared = int(k("expert_shared_count", 0) or 0) or (1 if shexp else 0)
    if n_shared and not shexp:
        raise MoEError(f"mashbox: {arch}.expert_shared_count = {n_shared} but the file has no ffn_*_shexp tensors")
    shared_ff = 0
    if shexp:
        shared_ff = int(k("expert_shared_feed_forward_length", 0) or 0)
        if shapes is not None:
            rows = int(shapes(f"blk.{first}.ffn_up_shexp.weight")[-2])
            shared_ff = rows if not shared_ff else shared_ff
        src.append(("n_shared", "ffn_*_shexp tensors"))
    shared_gate = "ffn_gate_inp_shexp.weight" in t
    sel_bias = "exp_probs_b.bias" in t
    router_bias = "ffn_gate_inp.bias" in t
    expert_bias = any(x in t for x in ("ffn_gate_exps.bias", "ffn_up_exps.bias", "ffn_down_exps.bias",
                                       "ffn_gate_up_exps.bias"))
    src.append(("act", f"arch default {LLAMA_CPP} {d['src']}"))
    return MoESpec(n_expert=E, n_used=K, expert_ff=ff, gating=gating, norm_w=norm_w, w_scale=w_scale, n_group=ng,
                   n_group_used=ngu, sel_bias=sel_bias, router_bias=router_bias, expert_bias=expert_bias, act=d["act"],
                   fused_gate_up=fused, n_shared=n_shared, shared_ff=shared_ff, shared_gate=shared_gate,
                   moe_layers=moe_layers, source=tuple(src))


# Hugging Face config.json -> the same fields. model_type -> the llama.cpp architecture whose graph (and defaults) the
# converter maps it to (convert_hf_to_gguf.py at a37cc46); it is used only for defaults the config does not state.
HF_ARCH = {"qwen3_moe": "qwen3moe", "qwen3_5_moe": "qwen35moe", "qwen2_moe": "qwen2moe",
           "olmoe": "olmoe", "mixtral": "llama", "gpt_oss": "gpt-oss", "deepseek_v3": "deepseek2",
           "deepseek_v2": "deepseek2"}
HF_SCORING = {"softmax": "softmax", "sigmoid": "sigmoid"}


def detect_hf(config, tensor_names=None):
    """-> MoESpec, or None for a dense model. Reads the counts and routing options config.json states; the rest from the
    llama.cpp architecture the converter maps model_type to."""
    c = dict(config.get("text_config") or {}, **{k: v for k, v in config.items() if k != "text_config"})
    pick = lambda *ks: next(((k, c[k]) for k in ks if c.get(k) is not None), (None, None))
    ek, E = pick("num_experts", "num_local_experts", "n_routed_experts")
    E = int(E or 0)
    if E <= 0:
        return None
    mt = c.get("model_type")
    arch = HF_ARCH.get(mt)
    if arch is None:
        raise MoEError(f"mashbox: config.json model_type '{mt}' has {ek} = {E}, and this version has no routing rule "
                       f"for it")
    d = ARCH_DEFAULTS[arch]
    src = [("n_expert", f"config.json {ek}")]
    kk, K = pick("num_experts_per_tok", "num_experts_per_token", "moe_topk")
    K = int(K or 0)
    if not 0 < K <= E:
        raise MoEError(f"mashbox: config.json {kk or 'num_experts_per_tok'} = {K} is not in 1..{E}")
    src.append(("n_used", f"config.json {kk}"))
    fk, ff = pick("moe_intermediate_size", "intermediate_size")
    src.append(("expert_ff", f"config.json {fk}"))
    sc = c.get("scoring_func")
    if sc is not None:
        if sc not in HF_SCORING:
            raise MoEError(f"mashbox: config.json scoring_func '{sc}' is not supported in this version")
        gating = HF_SCORING[sc]
        src.append(("gating", "config.json scoring_func"))
    elif d["gating"] == META:
        gating = "softmax"
        src.append(("gating", "absent: softmax (llama.cpp's default for old files)"))
    else:
        gating = d["gating"]
        src.append(("gating", f"arch default {LLAMA_CPP} {d['src']}"))
    if c.get("norm_topk_prob") is not None or d["norm_w"] == META:     # the config's own value wins for HF weights
        norm_w = bool(c.get("norm_topk_prob", False))
        src.append(("norm_w", "config.json norm_topk_prob" if "norm_topk_prob" in c else "absent: false"))
    else:
        norm_w = d["norm_w"]
        src.append(("norm_w", f"arch default {LLAMA_CPP} {d['src']}"))
    ws = float(c.get("routed_scaling_factor") or 0.0)
    w_scale = 1.0 if ws == 0.0 else ws
    ng, ngu = int(c.get("n_group") or 1), int(c.get("topk_group") or 1)
    if ng > 1 and (E % ng or not 0 < ngu <= ng):
        raise MoEError(f"mashbox: config.json n_group = {ng} / topk_group = {ngu} do not fit {E} experts")
    L = int(c.get("num_hidden_layers") or 0)
    fkd = int(c.get("first_k_dense_replace") or 0)
    step = int(c.get("decoder_sparse_step") or 1)
    only = set(c.get("mlp_only_layers") or [])
    moe_layers = tuple(i >= fkd and i not in only and (i + 1) % step == 0 for i in range(L))
    sk, n_shared = pick("n_shared_experts", "num_shared_experts")
    sfk, shared_ff = pick("shared_expert_intermediate_size")
    n_shared = int(n_shared or (1 if shared_ff else 0))
    shared_ff = int(shared_ff or (n_shared * int(ff or 0) if n_shared else 0))
    names = set(tensor_names or ())
    has = lambda frag: any(frag in n for n in names)
    return MoESpec(n_expert=E, n_used=K, expert_ff=int(ff or 0), gating=gating, norm_w=norm_w, w_scale=w_scale,
                   n_group=ng if ng > 1 else 1, n_group_used=ngu if ng > 1 else 1,
                   sel_bias=has("e_score_correction_bias") or c.get("topk_method") == "noaux_tc",
                   router_bias=has("router.bias"), expert_bias=arch == "gpt-oss" or has("experts.gate_up_proj_bias"),
                   act=d["act"], fused_gate_up=has("experts.gate_up_proj"), n_shared=n_shared, shared_ff=shared_ff,
                   shared_gate=arch in ("qwen2moe", "qwen35moe") and n_shared > 0, moe_layers=moe_layers,
                   source=tuple(src))
