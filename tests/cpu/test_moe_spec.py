"""Mixture of experts or not, from metadata only (mashbox.moe.spec): GGUF fixtures, pinned Hugging Face config.json
files, name independence and the refusals. No chip, no network: the configs are committed with their Hub revision
and sha256 (tests/fixtures/moe/hf/PINS.json)."""
import ast
import hashlib
import json
import os
import re

import pytest

from mashbox.moe import spec as S
from mashbox.moe.spec import MoEError, detect_gguf, detect_hf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF = os.path.join(ROOT, "tests", "fixtures", "moe", "hf")


def meta(arch, **kv):
    d = {f"{arch}.{k}": v for k, v in kv.items()}
    return lambda key, default=None: d.get(key, default)


def names(L, *per_layer, shexp=False, extra=()):
    out = ["token_embd.weight", "output_norm.weight"]
    for i in range(L):
        b = f"blk.{i}."
        out += [b + "attn_norm.weight", b + "attn_q.weight"]
        out += [b + t for t in per_layer]
        if shexp:
            out += [b + t for t in ("ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                    "ffn_down_shexp.weight")]
        out += [b + t for t in extra]
    return out


ROUTED = ("ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
DENSE = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")


def shapes_for(E, FF, D=64):
    def sh(n):
        if "_exps." in n:
            return (E, D, FF) if "down" in n else (E, 2 * FF if "gate_up" in n else FF, D)
        if "_shexp." in n:
            return (D, 2 * FF) if "down" in n else (2 * FF, D)
        return (D,)
    return sh


def test_dense_file_is_dense():
    assert detect_gguf(meta("qwen3", block_count=2), names(2, *DENSE), "qwen3") is None


def test_qwen3moe_metadata_and_arch_defaults():
    sp = detect_gguf(meta("qwen3moe", block_count=3, expert_count=16, expert_used_count=4,
                          expert_feed_forward_length=32), names(3, *ROUTED), "qwen3moe", shapes=shapes_for(16, 32))
    assert (sp.n_expert, sp.n_used, sp.expert_ff, sp.gating, sp.norm_w, sp.w_scale) == (16, 4, 32, "softmax", True, 1.0)
    assert (sp.n_shared, sp.shared_gate, sp.sel_bias, sp.router_bias, sp.fused_gate_up) == (0, False, False, False, False)
    assert sp.moe_layers == (True, True, True)
    src = dict(sp.source)
    assert src["n_expert"] == "metadata qwen3moe.expert_count"
    assert src["norm_w"].startswith("arch default llama.cpp a37cc46 src/models/qwen3moe.cpp:")


def test_qwen2moe_shared_expert_and_no_renorm():
    sp = detect_gguf(meta("qwen2moe", block_count=2, expert_count=8, expert_used_count=2, expert_feed_forward_length=16,
                          expert_shared_feed_forward_length=32), names(2, *ROUTED, shexp=True), "qwen2moe",
                     shapes=shapes_for(8, 16))
    assert (sp.norm_w, sp.n_shared, sp.shared_ff, sp.shared_gate) == (False, 1, 32, True)


def test_mixtral_is_llama_with_experts():
    """llama.cpp writes Mixtral as architecture `llama` + llama.expert_count; the experts' size is the tensor's"""
    sp = detect_gguf(meta("llama", block_count=2, expert_count=8, expert_used_count=2, feed_forward_length=24),
                     names(2, *ROUTED), "llama", shapes=shapes_for(8, 24))
    assert (sp.n_expert, sp.n_used, sp.expert_ff, sp.norm_w) == (8, 2, 24, True)
    assert dict(sp.source)["expert_ff"] == "expert tensor shape"


def test_leading_dense_layers_from_the_tensors():
    nm = names(1, *DENSE) + [n.replace("blk.0.", "blk.1.") for n in names(1, *ROUTED) if n.startswith("blk.")]
    sp = detect_gguf(meta("deepseek2", block_count=2, expert_count=8, expert_used_count=2, expert_feed_forward_length=16,
                          leading_dense_block_count=1, expert_gating_func=2, expert_weights_norm=True,
                          expert_weights_scale=2.5, expert_group_count=4, expert_group_used_count=2),
                     nm, "deepseek2", shapes=shapes_for(8, 16))
    assert sp.moe_layers == (False, True)
    assert (sp.gating, sp.norm_w, sp.w_scale, sp.n_group, sp.n_group_used) == ("sigmoid", True, 2.5, 4, 2)
    assert dict(sp.source)["gating"] == "metadata deepseek2.expert_gating_func"


def test_selection_and_router_biases_and_fused_gate_up():
    t = ("ffn_gate_inp.weight", "ffn_gate_inp.bias", "exp_probs_b.bias", "ffn_gate_up_exps.weight",
         "ffn_down_exps.weight", "ffn_down_exps.bias")
    sp = detect_gguf(meta("gpt-oss", block_count=1, expert_count=4, expert_used_count=2, expert_feed_forward_length=8),
                     names(1, *t), "gpt-oss", shapes=shapes_for(4, 8))
    assert (sp.sel_bias, sp.router_bias, sp.expert_bias, sp.fused_gate_up, sp.gating, sp.act) == (
        True, True, True, True, "softmax_weight", "swiglu_oai")


@pytest.mark.parametrize("why,m,nm,arch,needle", [
    ("expert_count without _exps", dict(expert_count=8, expert_used_count=2), names(2, *DENSE), "qwen3moe",
     "qwen3moe.expert_count is 8 but the file has no ffn_*_exps"),
    ("_exps without expert_count", dict(), names(2, *ROUTED), "qwen3moe", "but no qwen3moe.expert_count"),
    ("K > E", dict(expert_count=4, expert_used_count=8), names(2, *ROUTED), "qwen3moe",
     "qwen3moe.expert_used_count = 8 is more than qwen3moe.expert_count = 4"),
    ("unknown gating", dict(expert_count=8, expert_used_count=2, expert_gating_func=4), names(2, *ROUTED), "deepseek2",
     "deepseek2.expert_gating_func = 4 (sqrt_softplus)"),
    ("gating not llama.cpp's", dict(expert_count=8, expert_used_count=2, expert_gating_func=9), names(2, *ROUTED),
     "qwen3moe", "qwen3moe.expert_gating_func = 9"),
    ("groups do not divide E", dict(expert_count=8, expert_used_count=2, expert_group_count=3,
                                    expert_group_used_count=1), names(2, *ROUTED), "qwen3moe",
     "qwen3moe.expert_group_count = 3 does not divide expert_count = 8"),
    ("fused and separate", dict(expert_count=8, expert_used_count=2), names(2, *ROUTED, extra=("ffn_gate_up_exps.weight",)),
     "qwen3moe", "both a fused ffn_gate_up_exps and separate"),
    ("router without experts", dict(expert_count=8, expert_used_count=2),
     names(1, *ROUTED) + ["blk.1.ffn_gate_inp.weight"], "qwen3moe", "layer 1 has ffn_gate_inp without ffn_*_exps"),
    ("an arch with no routing rule", dict(expert_count=8, expert_used_count=2), names(2, *ROUTED), "phimoe",
     "'phimoe' file: this version has no routing rule"),
])
def test_refusals_name_the_field(why, m, nm, arch, needle):
    with pytest.raises(MoEError) as e:
        detect_gguf(meta(arch, block_count=2, expert_feed_forward_length=16, **m), nm, arch, shapes=shapes_for(8, 16))
    assert needle in str(e.value), (why, str(e.value))


def test_expert_count_must_match_the_tensor():
    with pytest.raises(MoEError, match="holds 4 experts, not qwen3moe.expert_count = 8"):
        detect_gguf(meta("qwen3moe", block_count=1, expert_count=8, expert_used_count=2), names(1, *ROUTED), "qwen3moe",
                    shapes=shapes_for(4, 16))


def test_name_never_decides():
    """the same metadata under another model's name, file name and repo id gives the identical spec; a dense file
    whose name says MoE stays dense"""
    base = dict(block_count=2, expert_count=8, expert_used_count=2, expert_feed_forward_length=16)
    a = detect_gguf(meta("qwen3moe", **base), names(2, *ROUTED), "qwen3moe", shapes=shapes_for(8, 16))
    g = meta("qwen3moe", **base)
    renamed = lambda k, d=None: {"general.name": "Some-Other-Model-8x7B-MoE", "general.basename": "x"}.get(k, g(k, d))
    b = detect_gguf(renamed, names(2, *ROUTED), "qwen3moe", shapes=shapes_for(8, 16))
    assert a == b
    dm = lambda k, d=None: {"general.name": "Tiny-MoE-A3B-Experts"}.get(k, d)
    assert detect_gguf(dm, names(2, *DENSE), "qwen3") is None


def test_no_model_names_in_the_moe_code():
    """0 model-name literals (sizes like 30B / A3B, product names) in mashbox/moe/*.py; architecture keys are allowed"""
    bad = re.compile(r"\b\d+(\.\d+)?[BbMm]\b|A\d+B|Qwen|Mixtral|OLMoE|DeepSeek|Llama|Gemma|GPT|Phi-|Mistral", re.I)
    allowed = set(S.ARCH_DEFAULTS) | set(S.HF_ARCH) | {"llama.cpp", "Llama.cpp"}
    d = os.path.join(ROOT, "src", "mashbox", "moe")
    hits = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(d, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in allowed:
                    continue
                txt = re.sub(r"llama\.cpp|src/[\w/.-]+\.(cpp|h)\b|convert_hf_to_gguf\.py", "", node.value)
                for k in allowed:
                    txt = txt.replace(f"'{k}'", "").replace(f"{k}.", "")
                if bad.search(txt):
                    hits.append((fn, getattr(node, "lineno", 0), node.value[:60]))
    assert not hits, hits


def _hf(name):
    pins = {p["file"]: p for p in json.load(open(os.path.join(HF, "PINS.json")))["configs"]}
    fn = f"{name}.config.json"
    raw = open(os.path.join(HF, fn), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == pins[fn]["sha256"], f"{fn} changed since it was pinned"
    return json.loads(raw)


@pytest.mark.parametrize("name,want", [
    ("qwen3-30b-a3b", dict(n_expert=128, n_used=8, expert_ff=768, gating="softmax", norm_w=True, n_shared=0)),
    ("qwen3.6-35b-a3b", dict(n_expert=256, n_used=8, expert_ff=512, gating="softmax", norm_w=True, n_shared=1,
                             shared_ff=512, shared_gate=True)),
    ("olmoe-1b-7b-0125", dict(n_expert=64, n_used=8, expert_ff=1024, gating="softmax", norm_w=False, n_shared=0)),
    ("mixtral-8x7b", dict(n_expert=8, n_used=2, expert_ff=14336, gating="softmax", norm_w=True, n_shared=0)),
    ("gpt-oss-20b", dict(n_expert=32, n_used=4, expert_ff=2880, gating="softmax_weight", norm_w=False, act="swiglu_oai")),
    ("deepseek-v3", dict(n_expert=256, n_used=8, expert_ff=2048, gating="sigmoid", norm_w=True, w_scale=2.5, n_group=8,
                         n_group_used=4, n_shared=1, shared_ff=2048, sel_bias=True)),
    ("qwen1.5-moe-a2.7b", dict(n_expert=60, n_used=4, expert_ff=1408, norm_w=False, n_shared=1, shared_ff=5632,
                               shared_gate=True)),
])
def test_hf_configs(name, want):
    sp = detect_hf(_hf(name))
    got = {k: getattr(sp, k) for k in want}
    assert got == want
    if name == "deepseek-v3":
        assert sp.moe_layers[:3] == (False, False, False) and all(sp.moe_layers[3:])


def test_hf_dense_config_is_dense():
    assert detect_hf(dict(model_type="qwen3", num_hidden_layers=4, intermediate_size=64)) is None


def test_hf_unknown_moe_type_is_refused_by_name():
    with pytest.raises(MoEError, match="model_type 'brandnew_moe' has num_experts = 16"):
        detect_hf(dict(model_type="brandnew_moe", num_experts=16, num_experts_per_tok=2))


def test_describe_line():
    sp = detect_hf(_hf("qwen3.6-35b-a3b"))
    assert sp.describe() == ("mixture of experts: 8 of 256 experts per token, 1 shared expert, softmax router, weights "
                             "renormalised")


def _tl35():
    import json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "moe", "tensor-lists",
                     "Qwen3.6-35B-A3B-UD-Q4_K_M.tensors.json")
    with open(p) as fh:
        return json.load(fh)


def test_expert_tensors_other_than_weight_and_bias_are_refused_by_name():
    """a per-expert tensor llama.cpp would apply (blk.N.ffn_*_exps.scale) is refused, never ignored"""
    from mashbox.moe import MoEError, detect_gguf
    d = _tl35()
    names = [t["name"] for t in d["tensors"]]
    get = d["metadata"].get
    assert detect_gguf(get, names, "qwen35moe").n_expert == 256          # control: the real file's names detect
    with pytest.raises(MoEError, match=r"blk\.3\.ffn_down_exps\.scale"):
        detect_gguf(get, names + ["blk.3.ffn_down_exps.scale"], "qwen35moe")
    with pytest.raises(MoEError, match=r"blk\.0\.ffn_up_shexp\.scale"):
        detect_gguf(get, names + ["blk.0.ffn_up_shexp.scale"], "qwen35moe")


def test_hf_qwen3_next_is_not_mapped_to_another_graph():
    """llama.cpp converts Qwen3-Next to its own arch (qwen3next), which this version does not port, so
    detect_hf does not borrow qwen35moe's defaults for it"""
    from mashbox.moe.spec import HF_ARCH
    assert "qwen3_next" not in HF_ARCH


def test_deepseek2_absent_gating_reads_as_llama_cpp_reads_it():
    """llama.cpp a37cc46 src/models/deepseek2.cpp:23-31: an absent (or none) expert_gating_func is softmax, except a file
    of 47 or 48 layers (block_count - nextn_predict_layers) with a 154,880-token vocabulary, which gets sigmoid. GGUF
    detection reads it the same way (the vocabulary from token_embd.weight's shape), detect_hf gives softmax for a
    config with no scoring_func, and the 47/48-layer case with no shapes given is refused naming the field."""
    base = dict(expert_count=8, expert_used_count=2, expert_feed_forward_length=16, expert_weights_norm=True)
    plain = shapes_for(8, 16)

    def vocab(n_vocab):
        return lambda n: (n_vocab, 64) if n == "token_embd.weight" else plain(n)
    for gf in ({}, dict(expert_gating_func=0)):
        sp = detect_gguf(meta("deepseek2", block_count=2, **base, **gf), names(2, *ROUTED), "deepseek2", shapes=plain)
        assert sp.gating == "softmax" and "deepseek2.cpp:23-31" in dict(sp.source)["gating"], sp.source
    m49 = meta("deepseek2", block_count=49, nextn_predict_layers=1, **base)
    assert detect_gguf(m49, names(49, *ROUTED), "deepseek2", shapes=vocab(154880)).gating == "sigmoid"
    assert detect_gguf(m49, names(49, *ROUTED), "deepseek2", shapes=vocab(129280)).gating == "softmax"
    assert detect_gguf(meta("deepseek2", block_count=49, **base), names(49, *ROUTED), "deepseek2",
                       shapes=vocab(154880)).gating == "softmax"            # 49 layers, no nextn: not the rule
    with pytest.raises(MoEError, match="deepseek2.expert_gating_func"):
        detect_gguf(meta("deepseek2", block_count=48, **base), names(48, *ROUTED), "deepseek2")
    cfg = dict(_hf("deepseek-v3"))
    cfg.pop("scoring_func")
    assert detect_hf(cfg).gating == "softmax"                               # the two detectors agree
