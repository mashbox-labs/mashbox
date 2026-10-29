"""GGUF architectures llama, qwen2, qwen3 (gguf/archs/dense.py): our logits vs llama.cpp's CPU logits on tiny random
fixtures (no chip, no network, no secrets). tests/fixtures/gguf/<name>.json says how each was made (llama.cpp commit,
file sha256).

  llama-f32 / qwen2-f32 / qwen3-f32  float32 weights, 4 layers: the graph test (llama: adjacent-pair rope, rope_freqs
                   factors, q/k/v/output biases; qwen2: halves rope, q/k/v biases; qwen3: per-head q/k norms, head size
                   != embedding_length / head_count). Bound NMSE <= 1e-4, llama.cpp's cross-backend bar.
  llama-pyquant    Q8_0 / Q4_0 / Q5_0 / Q4_1 / Q5_1 blocks written by gguf-py (different types per layer: unrolled path).
  qwen3-kquant     Q4_K / Q5_K / Q6_K / IQ4_NL / IQ4_XS / Q8_0 blocks written by llama.cpp's llama-quantize.
  Block-type bars, the same as test_gguf_qwen35.py and fixed before the first run: mean KL(llama.cpp || ours) <= 1e-3 and
  the same top-1 on >= 15 of 16 vs llama.cpp's block path, and NMSE <= 1e-4 vs llama.cpp on an F32 copy holding the
  identical dequantized weights (<name>.f32ref).
Every fixture runs twice: one prefill of all 16 tokens, and a prefill of 8 then 8 one-token decode steps (K/V cache path).
"""
import json
import os

import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

from mashbox.gguf import open_model, GGUFError  # noqa: E402
from mashbox.gguf.reader import File  # noqa: E402
from mashbox.gguf.archs import dense as A  # noqa: E402

NMSE_MAX = 1e-4
KL_MAX, TOP1_MIN = 1e-3, 15


def fixture(name, tmp):
    meta = json.load(open(os.path.join(FX, f"{name}.json")))
    p = FXT.build(name, str(tmp)) if meta["made_by"] == "gguf-py" else os.path.join(FX, f"{name}.gguf")
    assert FXT.sha256(p) == meta["gguf_sha256"], f"{name}: fixture bytes changed; rerun tools/gguf_fixtures/fixtures.py make"
    ref = np.fromfile(os.path.join(FX, f"{name}.logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    return p, meta, ref


def run(path, toks, act=jnp.float32, split=None):
    f, M, c = open_model(path, act=act)
    assert M is A
    W, info = A.load(f, c, use_kernels=False)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = A.new_state(c, 1, 32)
    i32 = jnp.int32
    z = jnp.zeros((1,), i32)
    T = len(toks)
    t = jnp.asarray([toks], i32)
    if split is None:
        r = fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)
        return np.asarray(r["lg"][0]), c, info
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out = [np.asarray(r["lg"][0])]
    ck, cv = r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0), c, info


def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def kl_top1(ref, ours):
    p = np.exp(ref - ref.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    kl = np.sum(p * (np.log(np.maximum(p, 1e-300)) - lq), -1)
    return float(kl.mean()), int(np.sum(ref.argmax(-1) == ours.argmax(-1)))


@pytest.mark.parametrize("name", ["llama-f32", "qwen2-f32", "qwen3-f32"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_float_logits_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = fixture(name, tmp_path)
    ours, c, info = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"{name} {split}: NMSE {e:.3e} stacked {info['layers_stacked']}")
    assert info["layers_stacked"]                         # one lax.scan over the layers
    assert e <= NMSE_MAX, e


@pytest.mark.parametrize("mixed_scan", [True, False], ids=["mixed-scan", "unrolled"])
@pytest.mark.parametrize("name", ["llama-pyquant", "qwen3-kquant"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_type_logits_match_llama_cpp(name, split, mixed_scan, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "MIXED_SCAN", mixed_scan)
    p, meta, ref = fixture(name, tmp_path)
    ours, c, info = run(p, meta["tokens"], split=split)
    kl, top1 = kl_top1(ref, ours)
    print(f"{name} {split}: KL {kl:.3e} top1 {top1}/{len(ref)} NMSE {nmse(ours, ref):.3e} types {info['reference_types']}")
    assert kl <= KL_MAX and top1 >= TOP1_MIN, (kl, top1)


@pytest.mark.parametrize("mixed_scan", [True, False], ids=["mixed-scan", "unrolled"])
@pytest.mark.parametrize("name", ["llama-pyquant", "qwen3-kquant"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_types_match_llama_cpp_on_identical_weights(name, split, mixed_scan, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "MIXED_SCAN", mixed_scan)
    p, meta, _ = fixture(name, tmp_path)
    ref = np.fromfile(os.path.join(FX, f"{name}.f32ref.logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    ours, c, info = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"{name} {split} vs f32ref: NMSE {e:.3e} stacked {info['layers_stacked']}")
    assert e <= NMSE_MAX, e


def test_fixture_paths_cover_scan_and_unrolled(tmp_path, monkeypatch):
    """the block-type fixtures exercise every layer path: llama-pyquant cycles types over layers (one lax.scan with
    ops.Mixed roles; with MIXED_SCAN False an unrolled list), qwen3-kquant has one type per role (stacked)"""
    for name, mixed_scan, want, mixed in (("llama-pyquant", True, True, True), ("llama-pyquant", False, False, False),
                                          ("qwen3-kquant", True, True, False)):
        monkeypatch.setattr(A, "MIXED_SCAN", mixed_scan)
        p, meta, _ = fixture(name, tmp_path)
        f, M, c = open_model(p, act=jnp.float32)
        W, info = A.load(f, c, use_kernels=False)
        assert info["layers_stacked"] is want and isinstance(W["g"], list) is (not want), (name, info["layers_stacked"])
        assert bool(info["mixed_roles"]) is mixed, (name, mixed_scan, info["mixed_roles"])
    p, _, _ = fixture("qwen3-kquant", tmp_path)
    names = {File(p).tensors[f"blk.0.{r}.weight"].type_name for r in ("ffn_down", "ffn_up", "ffn_gate", "attn_v",
                                                                      "attn_output", "attn_q")}
    assert {"Q6_K", "Q5_K", "Q4_K", "IQ4_NL", "IQ4_XS", "Q8_0"} <= names, names


def _header_only(path, arch, extra=(), tensors=("token_embd.weight",)):
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    w.add_embedding_length(8)
    w.add_feed_forward_length(16)
    w.add_head_count(2)
    w.add_layer_norm_rms_eps(1e-6)
    for k, v in extra:
        w.add_string(f"{arch}.{k}", v)
    for n in tensors:
        w.add_tensor(n, np.zeros((4, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.mark.parametrize("arch,extra,tensors,words", [
    ("qwen2", [("rope.scaling.type", "longrope")], ("token_embd.weight",), "rope scaling 'longrope'"),
    ("llama", [], ("token_embd.weight", "blk.0.ffn_gate_exps.weight"), "routed experts"),  # gitleaks:allow
    ("llama", [], ("token_embd.weight", "blk.0.attn_qkv.weight"), "a fused qkv tensor"),  # gitleaks:allow
    ("qwen3", [], ("token_embd.weight",), "no q/k norms")])
def test_refusals_are_one_line(arch, extra, tensors, words, tmp_path):
    p = _header_only(tmp_path / "m.gguf", arch, extra, tensors)
    with pytest.raises(GGUFError) as e:
        open_model(p)
    msg = str(e.value)
    assert words in msg and f"mashbox: {arch} file with" in msg and "\n" not in msg, msg


def test_config_reads_each_arch(tmp_path):
    for name, neox, qk, fr in (("llama-f32", False, False, True), ("qwen2-f32", True, False, False),
                               ("qwen3-f32", True, True, False)):
        f, M, c = open_model(FXT.build(name, str(tmp_path)), act=jnp.float32)
        assert (c.neox, c.qknorm, c.freqs, c.FI, c.NG, c.nextn) == (neox, qk, fr, 1, c.L, 0), name
    f, M, c = open_model(FXT.build("qwen3-f32", str(tmp_path)), act=jnp.float32)
    assert c.HD == 64 and c.D // c.H == 32                 # key_length read, not embedding_length / head_count


def test_tokenizer_bos_follows_llama_cpp(tmp_path):
    """llama files (add_bos_token) start every chat prompt with exactly one BOS, as llama.cpp; qwen files get none"""
    from mashbox.gguf import tokenizer as GT
    from mashbox.runtime.chat import Chat
    toks, types, merges = FXT.vocab()
    bos = len(toks) - 3
    for name, want in (("llama-f32", True), ("qwen2-f32", False), ("qwen3-f32", False)):
        p = FXT.build(name, str(tmp_path))
        ch = Chat(GT.model_dir(File(p), p, root=str(tmp_path / ("tok-" + name))))
        ids = ch.chat_ids([dict(role="user", content="in the hat")])
        assert (ids[0] == bos) is want and ids.count(bos) == (1 if want else 0), (name, ids[:4])
        assert ch.decode(ch.encode("the in a hat\n")) == "the in a hat\n"


def test_run_prompt_end_to_end_on_cpu(tmp_path, capsys, monkeypatch):
    from mashbox import cli
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = FXT.build("llama-f32", str(tmp_path))
    rc = cli.main(["run", p, "--prompt", "the hat", "--max-tokens", "6", "--ctx", "64", "--slots", "2", "--steps", "2",
                   "--cache-dir", str(tmp_path / "xla")])
    cap = capsys.readouterr()
    assert rc in (None, 0), cap.err
    events = [json.loads(l) for l in cap.err.splitlines() if l.startswith("{")]
    kinds = [e["event"] for e in events]
    assert "gguf_plan" in kinds and "loaded" in kinds and "done" in kinds, cap.err
    assert next(e for e in events if e["event"] == "gguf_plan")["arch"] == "llama"
    assert next(e for e in events if e["event"] == "done")["new_tokens"] >= 1
