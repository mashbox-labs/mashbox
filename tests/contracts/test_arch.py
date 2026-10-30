"""Architecture contracts `arch/<arch>`: one per key of mashbox.gguf.archs.ARCHS. A GGUF file of an architecture runs here
when its contract passes (with its op/ and tok/ contracts), as llama.cpp supports files by type, never by file.

Each contract is a set of tiny random-weight GGUF fixtures of that architecture (tools/gguf_fixtures/: seeded, small
shapes, the metadata keys llama.cpp reads), chosen so the features the architecture's graph reads differently are in a
file: RoPE kind, base and scaling, norm placement and eps, attention (GQA, sliding window, per-layer heads, hybrid linear
layers), biases, tied or own output head, soft-cap and scalars. The references are llama.cpp a37cc46's CPU backend on the
same files, made once (fixtures.py / more.py make: logits; greedy.py make: greedy ids) and committed with their sha256.
The tests need neither llama.cpp nor a chip; the engine runs its exact float32 path (use_kernels=False).

BARS: written 2026-10-30T06:10Z, before the first run of this file; never moved after a result. Items 1 and 2 are the
bars tests/cpu/test_gguf_dense.py and test_gguf_qwen35.py already lock; item 3 is the locked near-tie rule of
tests/cpu/test_catalog.py (NEAR_TIE_GAP).
  1. Float fixture: our logits for one prefill of the 16 prompt ids, and for a prefill of 8 then 8 one-token decode
     steps (K/V cache path): NMSE <= 1e-4 vs llama.cpp's logits.
  2. Block-type fixture: the same two runs; NMSE <= 1e-4 vs llama.cpp on the F32 copy holding the identical
     dequantized values (<name>.f32ref), the float bar. Mean KL(llama.cpp || ours) and top-1 vs llama.cpp on the block
     file are printed, never gating: llama.cpp's CPU block-type dot products first round the activations to 8 bits
     (Q8_K for the K-quants and IQ4_XS, Q8_0 for Q4_0/Q5_0/Q8_0/IQ4_NL/MXFP4, Q8_1 for Q4_1/Q5_1), so that comparison
     measures llama.cpp's rounding, not the architecture. The quant types themselves are gated by op/<TYPE> against an
     exact float64 reference. Corrected 2026-10-30, before its re-run. The record of the old bar 2 (KL <= 1e-3 and
     top-1 >= 15/16 vs the block file) stays: it FAILED on qwen35-kquant (KL 1.350e-3, top-1 16/16) and
     qwen35moe-pyquant (KL 4.923e-3, top-1 12/16). Measured the same day: our logits on those block files vs an
     independent float64 forward NMSE ~3e-12, top-1 16/16; llama.cpp's block path vs that float64 forward KL 1.352e-3
     and 4.912e-3; the float64 forward with llama.cpp's activation rounding reproduces llama.cpp's block logits to NMSE
     4.5e-6 and 5.1e-15.
  3. Greedy: 16 greedy ids after the 16 prompt ids (one prefill, then one-token decode steps on our K/V cache, each
     step's id = the lowest id of the largest logit, as llama.cpp's tool picks). PASS when all 16 ids equal llama.cpp's,
     or when the first different id is at a step where llama.cpp's top-two gap is under 0.5 logit (a near tie).
     And the logits of every step up to and including the first difference (both sides still have the same inputs
     there): NMSE <= 1e-4 vs llama.cpp's. A block-type fixture is compared with llama.cpp's greedy run on its F32 copy.
  4. Feature truth: every feature a case lists holds in the file it runs, so a coverage line is a fact of the file.
  Fingerprints (tests/contracts/conftest.py) record exact outputs only, never float logits (summation order differs
  between CPUs): the top-1 id per position and the greedy ids, each up to llama.cpp's first near tie (gap < 0.5, where
  another CPU may pick the other id), plus the positions left out.
  5. Coverage (not a contract): every ARCHS key has arch/<key>; every contract has a float case, an eps-sensitive case
     (rms eps EPS_STRESS: near the activations' mean square, so a norm that does not use the file's eps moves the logits;
     at a real eps of 1e-6..1e-5 random weights hide it) and committed references at the pinned llama.cpp commit.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

import gguf  # noqa: F401  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
import jax
import jax.numpy as jnp

from mashbox.gguf import archs
from mashbox.gguf.reader import File

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", "gguf_fixtures", file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


GREEDY = _load("gguf_fixtures_greedy_contracts", "greedy.py")
BASE, MORE = GREEDY.base, GREEDY.more

NMSE_MAX = 1e-4
NEAR_TIE_GAP = 0.5
EPS_STRESS = 0.25
CTX = 32                    # 16 prompt ids + 15 fed greedy ids fit; the decode steps attend Tc = CTX
SPLIT = 8


# ------------------------------------------------------------------------------------------------ features of a file
def key(f, k):
    return f.get(f"{f.arch}.{k}")


def has(*names):
    return lambda f: all(n in f.tensors for n in names)


def lacks(*names):
    return lambda f: not any(n in f.tensors for n in names)


def is_(k, v):
    return lambda f: key(f, k) == v


def set_(*ks):
    return lambda f: all(key(f, k) is not None for k in ks)


def gqa(f):
    h, kv = key(f, "attention.head_count"), key(f, "attention.head_count_kv")
    return bool(np.all(np.asarray(kv) < np.asarray(h)))


def head_size_not_d_over_h(f):
    hd = key(f, "attention.key_length")
    return hd is not None and int(hd) != int(key(f, "embedding_length")) // int(key(f, "attention.head_count"))


def no_scaling(f):
    return key(f, "rope.scaling.type") in (None, "none")


def blocks(f):
    return any(t.type_name not in ("F32", "F16", "BF16") for t in f.tensors.values())


def eps_is(v):
    return lambda f: abs(float(key(f, "attention.layer_norm_rms_epsilon")) - v) <= 1e-7 * v


TIED = ("tied head: no output.weight, the head is token_embd", lacks("output.weight"))
OWN = ("own output head (output.weight)", has("output.weight"))
EPS = (f"rms eps {EPS_STRESS} (eps-sensitive)", eps_is(EPS_STRESS))
GQA = ("grouped-query attention (head_count_kv < head_count)", gqa)
BLOCKS = ("block-type weights", blocks)
NOSCALE = ("no rope scaling", no_scaling)
YARN = ("YaRN rope scaling", lambda f: key(f, "rope.scaling.type") == "yarn" and key(f, "rope.scaling.factor") > 1)
LINEAR = ("linear rope scaling", lambda f: key(f, "rope.scaling.type") == "linear" and key(f, "rope.scaling.factor") > 1)
FREQS = ("rope_freqs.weight frequency factors", has("rope_freqs.weight"))
QKNORM = ("per-head q/k RMSNorm", has("blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"))
HDNE = ("head size != embedding_length / head_count", head_size_not_d_over_h)


def emb_unit(f):
    e = np.frombuffer(np.asarray(f.tensors["token_embd.weight"].rows()).tobytes(), np.float32)
    return 0.2 <= float(e.std()) * float(key(f, "embedding_scale") or 1.0) <= 1.0


EMB_UNIT = ("embedding x embedding_scale near unit scale, so attention and norms move the residual", emb_unit)


def routed(f):
    return has("blk.0.ffn_gate_exps.weight")(f) and int(key(f, "expert_used_count")) < int(key(f, "expert_count"))


ROUTED = ("routed experts (top-k of expert_count)", routed)
SHEXP = ("sigmoid-gated shared expert", has("blk.0.ffn_gate_inp_shexp.weight", "blk.0.ffn_up_shexp.weight"))

# arch -> [(fixture, "float" | "block", [(feature, predicate)])]. The rope kind (adjacent pairs vs halves), the norm
# formula and the graph are the architecture's own: every case of the contract runs them.
CONTRACTS = {
    "llama": [
        ("llama-f32", "float", [FREQS, ("q/k/v/output biases", has(*[f"blk.0.attn_{t}.bias" for t in "qkv"],
                                                                      "blk.0.attn_output.bias")), GQA, OWN, NOSCALE]),
        ("llama-yarn-f32", "float", [YARN, OWN]),
        ("llama-linear-f32", "float", [LINEAR, OWN]),
        ("llama-tied-f32", "float", [TIED, FREQS]),
        ("llama-eps-f32", "float", [EPS]),
        ("llama-pyquant", "block", [BLOCKS]),
        ("llama-moe-f32", "float", [ROUTED, ("experts under the llama name (feed_forward_length = expert size)",
                                             lacks("blk.0.ffn_gate.weight")), OWN]),
    ],
    "qwen2": [
        ("qwen2-f32", "float", [("q/k/v biases, no output bias", lambda f: has(*[f"blk.0.attn_{t}.bias" for t in "qkv"])(f)
                                 and lacks("blk.0.attn_output.bias")(f)), GQA, OWN, NOSCALE]),
        ("qwen2-tied-f32", "float", [TIED]),
        ("qwen2-eps-f32", "float", [EPS]),
    ],
    "qwen3": [
        ("qwen3-f32", "float", [QKNORM, HDNE, ("no biases", lacks("blk.0.attn_q.bias")), GQA, OWN, NOSCALE]),
        ("qwen3-yarn-f32", "float", [YARN, QKNORM]),
        ("qwen3-tied-f32", "float", [TIED, QKNORM]),
        ("qwen3-eps-f32", "float", [EPS, QKNORM]),
        ("qwen3-kquant", "block", [BLOCKS]),
    ],
    "qwen2moe": [
        ("qwen2moe-f32", "float", [ROUTED, SHEXP, ("q/k/v biases", has(*[f"blk.0.attn_{t}.bias" for t in "qkv"])),
                                   GQA, OWN]),
        ("qwen2moe-eps-f32", "float", [EPS, ROUTED]),
    ],
    "qwen3moe": [
        ("qwen3moe-f32", "float", [ROUTED, QKNORM, HDNE, ("no shared expert", lacks("blk.0.ffn_up_shexp.weight")),
                                   GQA, OWN]),
        ("qwen3moe-eps-f32", "float", [EPS, ROUTED, QKNORM]),
    ],
    "exaone": [
        ("exaone-f32", "float", [FREQS, GQA, OWN]),
        ("exaone-eps-f32", "float", [EPS]),
        ("exaone-pyquant", "block", [BLOCKS]),
    ],
    "granite": [
        ("granite-f32", "float", [("attention.scale, embedding_scale, residual_scale, logit_scale",
                                   set_("attention.scale", "embedding_scale", "residual_scale", "logit_scale")),
                                  TIED, GQA]),
        ("granite-eps-f32", "float", [EPS]),
        ("granite-attn-f32", "float", [EMB_UNIT, TIED]),
        ("granite-attn-eps-f32", "float", [EMB_UNIT, EPS]),
        ("granite-pyquant", "block", [BLOCKS]),
    ],
    "gemma3": [
        ("gemma3-f32", "float", [("sliding window 4 (the 16 prompt ids cross it)", is_("attention.sliding_window", 4)),
                                 ("linear rope scaling on the full layer", LINEAR[1]), QKNORM, HDNE,
                                 ("post-attention and post-FFN norms", has("blk.0.post_attention_norm.weight",
                                                                            "blk.0.post_ffw_norm.weight")),
                                 TIED, GQA]),
        ("gemma3-eps-f32", "float", [EPS]),
        ("gemma3-pyquant", "block", [BLOCKS]),
    ],
    "gemma4": [
        ("gemma4-f32", "float", [("sliding-window pattern and window 4", set_("attention.sliding_window_pattern",
                                                                              "attention.sliding_window")),
                                 ("per-layer K/V head counts", lambda f: not np.isscalar(key(f, "attention.head_count_kv"))),
                                 ("full layers with no attn_v (v = k projection)",
                                  lambda f: has("blk.0.attn_v.weight")(f) and lacks("blk.2.attn_v.weight")(f)),
                                 ("sliding-window rope base and head size", set_("rope.freq_base_swa",
                                                                                 "attention.key_length_swa")),
                                 FREQS, ("final logit soft-cap", set_("final_logit_softcapping")),
                                 ("layer_output_scale", has("blk.0.layer_output_scale.weight")), QKNORM, TIED]),
        ("gemma4-eps-f32", "float", [EPS]),
        ("gemma4-kquant", "block", [BLOCKS]),
    ],
    "qwen35": [
        ("qwen35-f32", "float", [("hybrid: linear-attention layers + a full-attention layer every 4th",
                                  has("blk.0.ssm_a", "blk.0.ssm_conv1d.weight", "blk.3.attn_q.weight")),
                                 ("multi-section rope", set_("rope.dimension_sections")),
                                 ("partial rope (dimension_count < head size)",
                                  lambda f: int(key(f, "rope.dimension_count")) < int(key(f, "attention.key_length"))),
                                 ("gated attention q (2 x heads x head size)", lambda f: f.tensors["blk.3.attn_q.weight"].shape[0]
                                  == 2 * int(key(f, "attention.head_count")) * int(key(f, "attention.key_length"))),
                                 ("q/k RMSNorm on full layers", has("blk.3.attn_q_norm.weight")), OWN]),
        ("qwen35-mtp", "float", [("an MTP (nextn) layer the main logits skip", has("blk.4.nextn.eh_proj.weight"))]),
        ("qwen35-eps-f32", "float", [EPS]),
        ("qwen35-pyquant", "block", [BLOCKS]),
        ("qwen35-kquant", "block", [BLOCKS]),
    ],
    "qwen35moe": [
        ("qwen35moe-f32", "float", [("routed experts (top-k of expert_count)", lambda f: has("blk.0.ffn_gate_exps.weight")(f)
                                     and int(key(f, "expert_used_count")) < int(key(f, "expert_count"))),
                                    ("sigmoid-gated shared expert", has("blk.0.ffn_gate_inp_shexp.weight",
                                                                        "blk.0.ffn_up_shexp.weight")),
                                    ("hybrid linear-attention layers", has("blk.0.ssm_a", "blk.3.attn_q.weight"))]),
        ("qwen35moe-eps-f32", "float", [EPS]),
        ("qwen35moe-pyquant", "block", [BLOCKS]),
    ],
}
CASES = [(f"arch/{a}", name) for a, cs in CONTRACTS.items() for name, _, _ in cs]
KIND = {name: kind for cs in CONTRACTS.values() for name, kind, _ in cs}
FEATURES = {name: feats for cs in CONTRACTS.values() for name, _, feats in cs}


# ------------------------------------------------------------------------------------------------------ the engine side
def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def kl_top1(ref, ours):
    ref, ours = ref.astype(np.float64), ours.astype(np.float64)    # float32 exp underflows to 0 (as test_gguf_more_archs.py)
    p = np.exp(ref - ref.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    kl = np.sum(p * (np.log(np.maximum(p, 1e-300)) - lq), -1)
    return float(kl.mean()), int(np.sum(ref.argmax(-1) == ours.argmax(-1)))


def owner(name):
    return GREEDY.owner(name)


def meta(name, kind=""):
    return json.load(open(os.path.join(FX, f"{name}{kind}.json")))


def ref(name, kind, rows):
    p = os.path.join(FX, f"{name}{kind}.logits.f32")
    return np.fromfile(p, np.float32).reshape(rows, -1), BASE.sha256(p)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """fixture name -> its .gguf path: rebuilt from its seed (checked against the committed sha256), or the committed file"""
    d, out = tmp_path_factory.mktemp("arch_contracts"), {}

    def get(name):
        if name not in out:
            mod, fx = owner(name)
            p = mod.build(name, str(d)) if fx["make"] == "gguf-py" else os.path.join(FX, f"{name}.gguf")
            assert BASE.sha256(p) == meta(name)["gguf_sha256"], f"{name}: fixture bytes changed; remake its references"
            out[name] = p
        return out[name]
    return get


_MODELS = {}


def model(path):
    """(module, config, weights, jitted forward) for a fixture file, float32 activations, the exact (no-kernel) path"""
    if path not in _MODELS:
        f = File(path)
        M = archs.module(f.arch)
        c = M.Config(f, act=jnp.float32)
        W, _ = M.load(f, c, use_kernels=False)
        _MODELS[path] = (M, c, W, jax.jit(M.make_forward(c), static_argnames=("Tc", "contig", "all_logits")))
    return _MODELS[path]


I32 = jnp.int32


def logits(path, toks, split=None):
    M, c, W, fwd = model(path)
    ck, cv = M.new_state(c, 1, CTX)
    z, T, t = jnp.zeros((1,), I32), len(toks), jnp.asarray([toks], I32)
    n = T if split is None else split
    r = fwd(W, ck, cv, t[:, :n], z, jnp.asarray([n], I32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(n, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], I32), jnp.asarray([1], I32), z, CTX, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0)


def greedy(path, toks, n_new):
    """-> (ids (n_new,) int32, logits (n_new, V) float32): one prefill, then one-token decode steps of our own argmax"""
    M, c, W, fwd = model(path)
    ck, cv = M.new_state(c, 1, CTX)
    z, T = jnp.zeros((1,), I32), len(toks)
    r = fwd(W, ck, cv, jnp.asarray([toks], I32), z, jnp.asarray([T], I32), z, 0, all_logits=True)
    rows, ck, cv = [np.asarray(r["lg"][0, T - 1])], r["ck"], r["cv"]
    ids = [int(np.argmax(rows[-1]))]
    for s in range(1, n_new):
        r = fwd(W, ck, cv, jnp.asarray([[ids[-1]]], I32), jnp.asarray([T + s - 1], I32), jnp.asarray([1], I32), z, CTX,
                all_logits=True)
        ck, cv = r["ck"], r["cv"]
        rows.append(np.asarray(r["lg"][0, 0]))
        ids.append(int(np.argmax(rows[-1])))
    return np.asarray(ids, np.int32), np.stack(rows)


# ----------------------------------------------------------------------------------------------------------- contracts
pytestmark = pytest.mark.contract
IDS = [f"{c}:{n}" for c, n in CASES]


@pytest.mark.parametrize("cid,name", CASES, ids=IDS)
def test_arch_case_features_are_in_the_file(cid, name, built, contract):
    c = contract(cid)
    f = File(built(name))
    assert f"arch/{f.arch}" == cid, (name, f.arch)
    missing = [what for what, holds in FEATURES[name] if not holds(f)]
    c.record("\n".join(what for what, _ in FEATURES[name]))
    assert not missing, f"{name} does not have: {missing}"


@pytest.mark.parametrize("split", [None, SPLIT], ids=["prefill16", "prefill8+decode8"])
@pytest.mark.parametrize("cid,name", [pytest.param(c, n, id=f"{c}:{n}") for c, n in CASES])
def test_arch_case_logits_match_llama_cpp(cid, name, split, built, contract):
    c = contract(cid)
    m = meta(name)
    assert m["llama_cpp_commit"] == BASE.LLAMA_CPP_COMMIT
    r, sha = ref(name, "", len(m["tokens"]))
    assert sha == m["logits_sha256"], f"{name}: reference logits bytes changed"
    ours = logits(built(name), m["tokens"], split)
    assert ours.shape == r.shape, (ours.shape, r.shape)
    srt = np.sort(r, -1)
    tie = np.nonzero(srt[:, -1] - srt[:, -2] < NEAR_TIE_GAP)[0].astype(np.int32)
    top1 = ours.argmax(-1).astype(np.int32)
    top1[tie] = -1
    c.record(top1)
    c.record(tie)
    if KIND[name] == "float":
        e = nmse(ours, r)
        print(f"{cid} {name} {split}: NMSE {e:.3e}")
        assert e <= NMSE_MAX, e
        return
    r32 = np.fromfile(os.path.join(FX, f"{name}.f32ref.logits.f32"), np.float32).reshape(r.shape)
    assert BASE.sha256(os.path.join(FX, f"{name}.f32ref.logits.f32")) == m["f32ref_logits_sha256"]
    kl, top1 = kl_top1(r, ours)      # vs llama.cpp's block path: printed, never gating (bar 2)
    e = nmse(ours, r32)
    print(f"{cid} {name} {split}: KL {kl:.3e} top1 {top1}/{len(r)} NMSE vs f32ref {e:.3e}")
    assert e <= NMSE_MAX, e


@pytest.mark.parametrize("cid,name", CASES, ids=IDS)
def test_arch_case_greedy_ids_match_llama_cpp(cid, name, built, contract):
    c = contract(cid)
    g = meta(name, ".greedy")
    m = meta(name)
    assert g["llama_cpp_commit"] == BASE.LLAMA_CPP_COMMIT and g["prompt"] == m["tokens"]
    # the file llama.cpp's greedy run used: the fixture itself, or (block types) the F32 copy its .f32ref logits came from
    assert g["gguf_sha256"] == (m["gguf_sha256"] if KIND[name] == "float" else m["f32ref_gguf_sha256"]), name
    r, sha = ref(name, ".greedy", g["n_new"])
    assert sha == g["logits_sha256"], f"{name}: reference greedy logits bytes changed"
    ref_ids = r.argmax(-1)
    srt = np.sort(r, -1)
    ref_gap = srt[:, -1] - srt[:, -2]
    assert ref_ids.tolist() == g["ids"], "reference ids are not the argmax of the reference logits"
    ids, lg = greedy(built(name), g["prompt"], g["n_new"])
    near = np.nonzero(ref_gap < NEAR_TIE_GAP)[0]
    stable = int(near[0]) if len(near) else g["n_new"]
    c.record(ids[:stable])
    c.record(np.asarray([stable], np.int32))
    diff = np.nonzero(ids != ref_ids)[0]
    first = int(diff[0]) if len(diff) else None
    upto = g["n_new"] if first is None else first + 1
    e = nmse(lg[:upto], r[:upto])
    print(f"{cid} {name}: greedy first difference {first} (llama.cpp gap there "
          f"{'-' if first is None else f'{ref_gap[first]:.3g}'}), NMSE over {upto} steps {e:.3e}; ours {ids.tolist()}")
    assert first is None or ref_gap[first] < NEAR_TIE_GAP, \
        f"{name}: greedy id {first} is {ids[first]}, llama.cpp {ref_ids[first]} (gap {ref_gap[first]:.3g} >= {NEAR_TIE_GAP})"
    assert e <= NMSE_MAX, e


def test_every_architecture_has_a_contract_with_its_cases_and_references():
    assert set(CONTRACTS) == set(archs.ARCHS), (sorted(set(archs.ARCHS) - set(CONTRACTS)),
                                                 sorted(set(CONTRACTS) - set(archs.ARCHS)))
    problems = []
    for a, cs in CONTRACTS.items():
        names = [n for n, _, _ in cs]
        if not any(k == "float" for _, k, _ in cs):
            problems.append(f"arch/{a}: no float case")
        if not any(EPS in feats for _, _, feats in cs):
            problems.append(f"arch/{a}: no eps-sensitive case")
        if not any(EPS not in feats and k == "float" for _, k, feats in cs):
            problems.append(f"arch/{a}: no float case at the architecture's own eps")
        for n in names:
            if owner(n)[1]["arch"] != a:
                problems.append(f"arch/{a}: {n} is a {owner(n)[1]['arch']} fixture")
            if (KIND[n] == "block") != GREEDY.blocks(owner(n)[1]):
                problems.append(f"arch/{a}: {n} kind {KIND[n]} disagrees with its block types")
            for suffix in (".json", ".logits.f32", ".greedy.json", ".greedy.logits.f32") + \
                    ((".f32ref.logits.f32",) if KIND[n] == "block" else ()):
                if not os.path.exists(os.path.join(FX, n + suffix)):
                    problems.append(f"arch/{a}: {n}{suffix} missing (tools/gguf_fixtures/*.py make, greedy.py make)")
    assert not problems, "\n".join(problems)


def test_every_fixture_is_in_a_contract():
    """a fixture with llama.cpp references and no contract would be a check no row stands on"""
    listed = {n for cs in CONTRACTS.values() for n, _, _ in cs}
    fixtures = set(BASE.FIXTURES) | set(MORE.FIXTURES)
    assert fixtures <= listed, sorted(fixtures - listed)
