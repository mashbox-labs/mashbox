"""The Hugging Face safetensors path (mashbox/hf): our logits vs transformers' float32 logits on tiny random models, and
the pieces around it (no chip, no network, no secrets). tests/fixtures/hf/<name>/meta.json says how each was made.

  llama / mistral / qwen2 / qwen3 / qwen35   float32 weights; bound NMSE <= 1e-4 (llama.cpp's cross-backend bar), one
                   prefill of 16 tokens and a prefill of 8 then 8 one-token decode steps
  Q8_0 on load     the blocks equal gguf-py's quantize of the same values, bit for bit; logits NMSE <= 1e-4 vs
                   transformers on the same dequantized values (logits.q8_0.f32)
  .bshj            a lossless .bshj copy gives the same tensor bytes as the safetensors
  bf16 matmul      the native BF16 matmul runs (no fallback line) and equals the dequantized path; forced to fail,
                   the output is the same; the load log counts the handles each path holds
  old path         no type but BF16 gets the native handle, and a GGUF file without BF16 weights loads the same handles
  Qwen3.5 layout   Qwen3_5ForConditionalGeneration (text_config, model.language_model.*, vision and MTP tensors) reads
                   as the same File and logits as the text-only fixture
  form pick        bf16 when it fits, Q8_0 when only that fits, else the refusal naming the chips that fit
  chat template    taken from tokenizer_config.json when there is no chat_template.jinja: renders as transformers does
  refusals         every config field or tensor the map does not cover stops in one line; a gated, missing or
                   unmapped Hub repo stops in one line before any weights download (Hub calls monkeypatched)
  run entry        parsed `run` args -> hf.resolve -> hf.load -> Engine.generate on CPU
"""
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

import gguf
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "hf")
_spec = importlib.util.spec_from_file_location("hf_fixtures", os.path.join(ROOT, "tools", "hf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

from mashbox import hf, ops  # noqa: E402
from mashbox.hf import adapter, quant, source  # noqa: E402
from mashbox.ops import dense16  # noqa: E402

NMSE_MAX = 1e-4
NAMES = ["llama", "mistral", "qwen2", "qwen3", "qwen35"]
i32 = jnp.int32


def fixture(name):
    d = os.path.join(FX, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    for f, h in meta["sha256"].items():
        assert FXT.sha256(os.path.join(d, f)) == h, f"{name}/{f} changed; rerun tools/hf_fixtures/fixtures.py make"
    ref = np.fromfile(os.path.join(d, "logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    return d, meta, ref


def run(d, toks, split=None, use_kernels=False, form=None):
    f, A, c = hf.open_model(d, act=jnp.float32, form=form)
    W, info = A.load(f, c, use_kernels=use_kernels)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = A.new_state(c, 1, 32)
    z = jnp.zeros((1,), i32)
    t = jnp.asarray([toks], i32)
    T = len(toks)
    if split is None:
        return np.asarray(fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)["lg"][0]), f, info
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0), f, info


def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def kl_top1(ref, ours):
    p = np.exp(ref - ref.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True)
    lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    return float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - lq), -1).mean()), int(np.sum(ref.argmax(-1) == ours.argmax(-1)))


def as_bf16(d, out):
    """a copy of fixture folder d with its float32 weights rounded to bfloat16 (round to nearest even)"""
    import ml_dtypes
    shutil.copytree(d, out)
    src = source.open_source(d)
    ts = {}
    for n, (dt, shape, raw) in src.tensors.items():
        b = np.asarray(raw).view(np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        ts[n] = ("BF16", shape, b.reshape(-1).view(np.uint8))
    src.close()
    FXT.write_safetensors(os.path.join(out, "model.safetensors"), ts)
    return out


# ------------------------------------------------------------------------------------------------------- logits
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_float_logits_match_transformers(name, split):
    d, meta, ref = fixture(name)
    ours, f, info = run(d, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"{name} {split}: NMSE {e:.3e}")
    assert f.arch == {"mistral": "llama"}.get(name, name)
    assert e <= NMSE_MAX, e


def test_llama_permute_and_rope_factors_are_the_converters():
    d, _, _ = fixture("llama")
    f, _, c = hf.open_model(d)
    assert c.freqs and not c.neox and not c.tie
    src = source.open_source(d)
    dt, shape, raw = src.tensors["model.layers.0.self_attn.q_proj.weight"]
    w = np.asarray(raw).view(np.float32).reshape(shape)
    H = 4
    want = w.reshape(H, 2, shape[0] // H // 2, shape[1]).swapaxes(1, 2).reshape(shape)
    got = np.asarray(f.tensors["blk.0.attn_q.weight"].raw).view(np.float32).reshape(shape)
    assert np.array_equal(got, want)
    fr = np.asarray(f.tensors["rope_freqs.weight"].raw).view(np.float32)
    assert fr.shape == (8,) and fr[0] == 1.0 and fr[-1] == 8.0


# ------------------------------------------------------------------------------------------------------- Q8_0
def test_q8_0_on_load_is_gguf_py_quantize_bit_for_bit(tmp_path):
    d, meta, ref = fixture("qwen3")
    b = as_bf16(d, str(tmp_path / "bf16"))
    f, A, c = hf.open_model(b, form="q8_0")
    g, _, _ = hf.open_model(b)
    names = quant.q8_0_names(g, A, c)
    assert names and all(f.tensors[n].type == quant.Q8_0 for n in names)
    assert f.tensors["blk.0.attn_norm.weight"].type == 30                  # norms stay as stored
    for n in names:
        t = g.tensors[n]
        want = gguf.quants.quantize(adapter._f32("BF16", t.raw, t.shape), gguf.GGMLQuantizationType.Q8_0)
        assert np.array_equal(np.asarray(f.tensors[n].raw).reshape(want.shape), want), n


@pytest.mark.parametrize("name", ["qwen3", "qwen35"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_q8_0_on_load_logits_match_transformers_on_the_same_values(name, split, tmp_path):
    d, meta, ref = fixture(name)
    qref = np.fromfile(os.path.join(d, "logits.q8_0.f32"), np.float32).reshape(ref.shape)
    ours, f, _ = run(as_bf16(d, str(tmp_path / "bf16")), meta["tokens"], split=split, form="q8_0")
    e = nmse(ours, qref)
    kl, top1 = kl_top1(ref, ours)
    print(f"{name} Q8_0 on load: NMSE {e:.3e} vs the same values; vs float32: KL {kl:.2e}, top-1 {top1}/16")
    assert f.form == "q8_0" and e <= NMSE_MAX, e


# ------------------------------------------------------------------------------------------------------- .bshj
@pytest.mark.parametrize("name", ["llama", "qwen35"])
def test_lossless_bshj_gives_the_same_tensors(name, tmp_path):
    d, meta, ref = fixture(name)
    out = tmp_path / "bj"
    out.mkdir()
    shutil.copy(os.path.join(d, "config.json"), out)
    src = source.open_source(d)
    FXT.write_bshj(str(out / "model.bshj"), dict(src.tensors))
    a, _, _ = hf.open_model(d)
    b, _, _ = hf.open_model(str(out))
    assert b.src.kind == "bshj" and sorted(a.tensors) == sorted(b.tensors)
    for n in a.tensors:
        assert a.tensors[n].type == b.tensors[n].type and a.tensors[n].shape == b.tensors[n].shape
        assert np.asarray(a.tensors[n].raw).tobytes() == np.asarray(b.tensors[n].raw).tobytes(), n
    ours, _, _ = run(str(out), meta["tokens"])
    assert nmse(ours, ref) <= NMSE_MAX


def test_device_form_bshj_is_refused(tmp_path):
    d, _, _ = fixture("qwen3")
    out = tmp_path / "bj"
    out.mkdir()
    shutil.copy(os.path.join(d, "config.json"), out)
    FXT.write_bshj(str(out / "model.bshj"), {"model.embed_tokens.weight": ("F32", (2, 2), np.zeros(16, np.uint8))})
    p = out / "model.bshj"
    raw = bytearray(p.read_bytes())
    raw[6:8] = (2).to_bytes(2, "little")                                     # container minor 2 = a device form
    p.write_bytes(bytes(raw))
    with pytest.raises(hf.HFError, match="newer than this reader"):
        hf.open_model(str(out))


# ------------------------------------------------------------------------------------------------------- bf16 matmul
@pytest.mark.parametrize("name", ["qwen3", "qwen35"])
def test_native_bf16_matmul_equals_the_dequantized_path(name, tmp_path, monkeypatch, capfd):
    d, meta, _ = fixture(name)
    b = as_bf16(d, str(tmp_path / "bf16"))
    exact, _, i0 = run(b, meta["tokens"], use_kernels=False)
    monkeypatch.delenv("MASHBOX_DENSE16", raising=False)
    calls, real = [], dense16.native
    monkeypatch.setattr(dense16, "native", lambda x, w: calls.append(w.shape) or real(x, w))
    dense16._STATE["warned"].clear()
    capfd.readouterr()
    fast, _, i1 = run(b, meta["tokens"], use_kernels=True)
    assert calls, "the native bf16 matmul never ran"
    assert "dequantized path" not in capfd.readouterr().err
    assert nmse(fast, exact) <= 1e-6


@pytest.mark.standards
def test_native_bf16_matmul_falls_back_when_forced_to_fail(tmp_path, monkeypatch, capfd):
    d, meta, _ = fixture("qwen3")
    b = as_bf16(d, str(tmp_path / "bf16"))
    exact, _, _ = run(b, meta["tokens"], use_kernels=False)

    def boom(x, w):
        raise RuntimeError("forced")
    monkeypatch.setattr(dense16, "native", boom)
    dense16._STATE["warned"].clear()
    fell, _, _ = run(b, meta["tokens"], use_kernels=True)
    assert np.array_equal(fell, exact) or nmse(fell, exact) <= 1e-9
    assert "dequantized path" in capfd.readouterr().err
    monkeypatch.setenv("MASHBOX_DENSE16", "off")
    raw = np.zeros((4, 64), np.uint8)
    assert isinstance(ops.prepare(raw, 30, (4, 32)), ops.Raw)


@pytest.mark.standards
def test_native_bf16_prepare_falls_back_when_forced_to_fail(monkeypatch, capfd):
    def boom(raw, shape):
        raise RuntimeError("forced")
    monkeypatch.delenv("MASHBOX_DENSE16", raising=False)
    monkeypatch.setattr(dense16, "_host_values", boom)
    dense16._STATE["warned"].clear()
    assert isinstance(ops.prepare(np.zeros((4, 64), np.uint8), 30, (4, 32)), ops.Raw)
    assert "BF16 weights run on the dequantized path" in capfd.readouterr().err


def test_bf16_handles_through_ops():
    import ml_dtypes
    w = (np.random.default_rng(0).standard_normal((6, 32)).astype(np.float32)).astype(ml_dtypes.bfloat16)
    raw = w.view(np.uint8).reshape(6, 64)
    h = ops.prepare(raw, 30, (6, 32))
    assert isinstance(h, dense16.Dense16)
    assert np.array_equal(np.asarray(ops.dequant(h)), w.astype(np.float32))
    assert np.array_equal(np.asarray(ops.take_rows(h, jnp.asarray([4, 1]))), w.astype(np.float32)[[4, 1]])
    assert isinstance(ops.prepare(raw, 30, (6, 32), use_kernel=False), ops.Raw)


# ------------------------------------------------------------------------------------------------------- form pick
@pytest.mark.standards
def test_form_pick_bf16_then_q8_0_then_refusal(monkeypatch):
    d, _, _ = fixture("qwen3")
    f, A, c = hf.open_model(d)
    from mashbox import gguf as G
    need = sum(G.need_bytes(f, c, 32, 1))
    q = hf.q8_0_need(f, A, c, 32, 1)
    assert q < need and f.form == "bf16" and f.tensors["blk.0.attn_q.weight"].type == 0
    assert hf.pick_form(f, A, c, 32, 1, lim=need * 2)[0] == "bf16"
    assert hf.pick_form(f, A, c, 32, 1, lim=(need + q) / 2 / G.HEADROOM)[0] == "q8_0"
    with pytest.raises(hf.HFError, match="It fits: TPU v5e"):
        hf.pick_form(f, A, c, 32, 1, lim=q / 2)
    assert hf.pick_form(f, A, c, 32, 1, lim=0)[0] == "bf16"           # no chip limit: CPU
    monkeypatch.setenv("MASHBOX_HF_LOAD", "q8_0")
    assert hf.pick_form(f, A, c, 32, 1, lim=need * 2) == ("q8_0", "MASHBOX_HF_LOAD")
    monkeypatch.setenv("MASHBOX_HF_LOAD", "int4")
    with pytest.raises(hf.HFError):
        hf.pick_form(f, A, c, 32, 1, lim=need * 2)


def test_default_slots_step_down_before_a_refusal():
    """--slots left at 8: when no form fits, 4, 2 and 1 slots are tried before refusing, as a GGUF file's fit_slots does
    (one v6e chip refused Qwen3.5-27B and Qwen3.6-27B at zero flags; Q8_0 made on load fits there with 2 slots). An
    explicit --slots is kept. The refusal names the usable share of the chip, not only its size."""
    d, _, _ = fixture("qwen3")
    f, A, c = hf.open_model(d)
    from mashbox import gguf as G
    q = {s: hf.q8_0_need(f, A, c, 2048, s) for s in (1, 2, 4, 8)}
    bf2 = sum(G.need_bytes(f, c, 2048, 2))
    lim = q[2] / G.HEADROOM * (1 + 1e-9)               # Q8_0 fits with 2 slots, not with 4; bf16 does not fit with 2
    assert q[4] > G.HEADROOM * lim and bf2 > G.HEADROOM * lim and q[2] <= G.HEADROOM * lim
    form, slots, why = hf.fit_form(f, A, c, 2048, 8, lim=lim)
    assert (form, slots) == ("q8_0", 2), why
    with pytest.raises(hf.HFError, match=r"with 8 slots of 2048 tokens, and [0-9.]+ GB of this chip's [0-9.]+ GB is usable"):
        hf.fit_form(f, A, c, 2048, 8, lim=lim, step_down=False)
    with pytest.raises(hf.HFError, match="with 1 slots of 2048 tokens"):
        hf.fit_form(f, A, c, 2048, 8, lim=q[1] / G.HEADROOM / 2)


# ------------------------------------------------------------------------------------------------------- chat
def _tiny_tokenizer(path, specials):
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[UNK]": 0, **{s: i + 1 for i, s in enumerate(specials)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.add_special_tokens(specials)
    t.save(path)


def test_chat_template_from_tokenizer_config_renders_as_transformers(tmp_path):
    from mashbox.runtime.chat import Chat
    src = os.path.join(FX, "chat-a2")
    meta = json.load(open(os.path.join(src, "meta.json")))
    for f, h in meta["sha256"].items():
        assert FXT.sha256(os.path.join(src, f)) == h
    d = tmp_path / "m"
    d.mkdir()
    shutil.copy(os.path.join(src, "tokenizer_config.json"), d)
    _tiny_tokenizer(str(d / "tokenizer.json"), ["<|im_start|>", "<|im_end|>"])
    (d / "config.json").write_text(json.dumps(dict(eos_token_id=[1, 2])))
    md = hf.model_dir(str(d), root=str(tmp_path / "cache"))
    ch = Chat(md)
    R = json.load(open(os.path.join(src, "renders.json")))
    for conv, want in zip(R["conversations"], R["renders"]):
        assert ch.render(conv) == want
    assert ch.stop_ids == (1, 2)
    assert hf.model_dir(str(d), root=str(tmp_path / "cache")) == md                 # written once, keyed by content


def test_folder_chat_template_jinja_wins(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    _tiny_tokenizer(str(d / "tokenizer.json"), [])
    (d / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    (d / "tokenizer_config.json").write_text(json.dumps(dict(chat_template="other")))
    (d / "generation_config.json").write_text(json.dumps(dict(eos_token_id=7)))
    fs = hf.chat_files(str(d))
    assert fs["chat_template.jinja"] == "{{ messages[0]['content'] }}" and json.loads(fs["generation_config.json"])["eos_token_id"] == 7


@pytest.mark.parametrize("cfg,eos", [(dict(eos_token_id=1), "<|im_end|>"),
                                     (dict(text_config=dict(eos_token_id=1)), dict(content="<|im_end|>", special=True))])
def test_stop_ids_without_generation_config_add_the_turn_end(tmp_path, cfg, eos):
    """a repo with no generation_config.json (Qwen/Qwen3.5-9B): config.json names <|endoftext|>, tokenizer_config.json's
    eos_token <|im_end|> ends every assistant turn; both stop a reply"""
    from mashbox.runtime.chat import Chat
    d = tmp_path / "m"
    d.mkdir()
    _tiny_tokenizer(str(d / "tokenizer.json"), ["<|endoftext|>", "<|im_end|>"])
    (d / "tokenizer_config.json").write_text(json.dumps(dict(chat_template="{{ messages[0]['content'] }}", eos_token=eos)))
    (d / "config.json").write_text(json.dumps(cfg))
    assert sorted(Chat(hf.model_dir(str(d), root=str(tmp_path / "cache"))).stop_ids) == [1, 2]


# ------------------------------------------------------------------------------------------------------- refusals
def _cfg(name):
    return json.load(open(os.path.join(FX, name, "config.json")))


REFUSE = [
    ("architectures", lambda c: c.update(architectures=["GPTNeoXForCausalLM"]), "architectures"),
    ("moe", lambda c: c.update(num_experts=8), "num_experts"),
    ("act", lambda c: c.update(hidden_act="gelu"), "hidden_act"),
    ("quant", lambda c: c.update(quantization_config=dict(quant_method="awq")), "quantization_config"),
    ("mlp_bias", lambda c: c.update(mlp_bias=True), "mlp_bias"),
    ("sliding", lambda c: c.update(use_sliding_window=True), "use_sliding_window"),
    ("yarn", lambda c: c.update(rope_parameters=dict(rope_type="yarn", rope_theta=1e6, factor=4.0)), "rope type"),
    ("layer_types", lambda c: c.update(layer_types=["sliding_attention"] * 3), "layer_types"),
]


@pytest.mark.parametrize("case", REFUSE, ids=[r[0] for r in REFUSE])
def test_unmapped_config_refuses_in_one_line(case, tmp_path):
    _, change, field = case
    d = tmp_path / "m"
    shutil.copytree(os.path.join(FX, "qwen3"), d)
    c = _cfg("qwen3")
    change(c)
    (d / "config.json").write_text(json.dumps(c))
    with pytest.raises(hf.HFError) as e:
        hf.open_model(str(d))
    assert field in str(e.value) and "\n" not in str(e.value)
    out = hf.inspect(str(d))
    assert out["runs_here"] is False and field in out["why"]


def test_unmapped_tensor_and_dtype_refuse(tmp_path):
    d = tmp_path / "m"
    shutil.copytree(os.path.join(FX, "qwen3"), d)
    FXT.write_safetensors(str(d / "extra.safetensors"), {"model.layers.0.self_attn.rel_bias.weight": np.zeros(4, np.float32)})
    with pytest.raises(hf.HFError, match="rel_bias.weight has no place in the qwen3 map"):
        hf.open_model(str(d))
    (d / "extra.safetensors").unlink()
    FXT.write_safetensors(str(d / "extra.safetensors"), {"x": ("I8", (4,), np.zeros(4, np.uint8))})
    with pytest.raises(hf.HFError, match="stored as I8"):
        hf.open_model(str(d))


def test_vision_and_mtp_tensors_are_skipped():
    assert adapter.gguf_name("qwen35", "model.visual.blocks.0.attn.qkv.weight") is None
    assert adapter.gguf_name("qwen35", "mtp.layers.0.mlp.up_proj.weight") is None
    assert adapter.gguf_name("qwen35", "model.language_model.layers.3.linear_attn.A_log") == "blk.3.ssm_a"
    assert adapter.gguf_name("qwen35", "model.language_model.embed_tokens.weight") == "token_embd.weight"


def test_conditional_generation_layout_runs_the_text_tower(tmp_path):
    """the published Qwen3.5 repos' layout: Qwen3_5ForConditionalGeneration with text_config (model_type qwen3_5_text),
    tie_word_embeddings at the top, model.language_model.* names, vision and MTP tensors beside the text tower. It reads
    as the same File (keys, tensors, bytes) and gives the same logits as the text-only fixture"""
    d, meta, ref = fixture("qwen35")
    cg = tmp_path / "cg"
    shutil.copytree(d, cg)
    c = _cfg("qwen35")
    text = {k: v for k, v in c.items() if k not in ("architectures", "tie_word_embeddings", "transformers_version")}
    text["model_type"] = "qwen3_5_text"
    (cg / "config.json").write_text(json.dumps(dict(
        architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5", text_config=text,
        tie_word_embeddings=c.get("tie_word_embeddings", False), vision_config=dict(depth=1, hidden_size=8))))
    src = source.open_source(d)
    lm = lambda n: n if n.startswith(("model.language_model.", "lm_head.")) else "model.language_model." + n[len("model."):]
    ts = {lm(n): (dt, shape, np.asarray(raw).reshape(-1).view(np.uint8)) for n, (dt, shape, raw) in src.tensors.items()}
    assert all(n.startswith(("model.language_model.", "lm_head.")) for n in ts)       # the published repos' text names
    ts["model.visual.blocks.0.mlp.linear_fc1.weight"] = np.ones((8, 4), np.float32)
    ts["mtp.fc.weight"] = np.ones((4, 8), np.float32)
    FXT.write_safetensors(str(cg / "model.safetensors"), ts)        # the views read src's memory map: close it after
    src.close()
    f1, _, _ = hf.open_model(d, act=jnp.float32)
    f2, _, _ = hf.open_model(str(cg), act=jnp.float32)
    assert f2.arch == f1.arch == "qwen35" and f2.meta == f1.meta and sorted(f2.tensors) == sorted(f1.tensors)
    for n, t in f1.tensors.items():
        u = f2.tensors[n]
        assert (u.type, tuple(u.shape)) == (t.type, tuple(t.shape)), n
        assert np.asarray(u.rows()).tobytes() == np.asarray(t.rows()).tobytes(), n
    out, _, _ = run(str(cg), meta["tokens"])
    base, _, _ = run(d, meta["tokens"])
    assert np.array_equal(out, base) and nmse(out, ref) <= NMSE_MAX


def test_routed_experts_refusal_names_no_other_path(tmp_path):
    """a model with routed experts stops in one line saying it does not run from Hugging Face safetensors (there is no
    Hugging Face MoE path to send it to)"""
    for name, change in (("qwen3", lambda c: c.update(num_experts=8)),
                         ("qwen35", lambda c: c.update(model_type="qwen3_5_moe_text"))):
        d = tmp_path / name
        shutil.copytree(os.path.join(FX, name), d)
        c = _cfg(name)
        change(c)
        (d / "config.json").write_text(json.dumps(c))
        with pytest.raises(hf.HFError) as e:
            hf.open_model(str(d))
        msg = str(e.value)
        assert "do not run from Hugging Face safetensors" in msg and "MoE path" not in msg and "\n" not in msg, msg


def test_files_without_bf16_weights_take_the_old_path(monkeypatch):
    """ops.prepare routes only ggml type 30 (BF16) to the native op: no other type gets a Dense16 handle, and a GGUF file
    with no BF16 tensor (the committed qwen35 K-quant fixture) loads the same handles and bytes with the native op on or
    off, so every later ops call on it takes the lines it took before the change"""
    from mashbox import gguf as G
    monkeypatch.delenv("MASHBOX_DENSE16", raising=False)
    for t, (tname, bs, tsz) in ops.reference.TYPES.items():
        if t not in dense16.TYPES:
            cols = bs * 2 if bs > 1 else 8
            h = ops.prepare(np.zeros((2, cols // bs * tsz), np.uint8), t, (2, cols), use_kernel=True)
            assert not isinstance(h, dense16.Dense16), tname
    p = os.path.join(ROOT, "tests", "fixtures", "gguf", "qwen35-kquant.gguf")
    is_h = lambda x: isinstance(x, (ops.Raw, ops.Packed, dense16.Dense16))
    got = {}
    for mode in ("on", "off"):
        monkeypatch.setenv("MASHBOX_DENSE16", mode)
        f, A, c = G.open_model(p, act=jnp.float32)
        assert all(t.type not in dense16.TYPES for t in f.tensors.values())
        W, _ = A.load(f, c, use_kernels=True)
        got[mode] = [(type(x).__name__, getattr(x, "t", None), getattr(x, "shape", None),
                      None if isinstance(x, ops.Packed) else np.asarray(x.raw if isinstance(x, ops.Raw) else x).tobytes())
                     for x in jax.tree_util.tree_leaves(W, is_leaf=is_h)]
    assert got["on"] == got["off"] and all(k != "Dense16" for k, *_ in got["on"])


# ------------------------------------------------------------------------------------------------------- claims
def test_claims_and_inspect(tmp_path):
    d, _, _ = fixture("qwen35")
    assert hf.claims(d)
    assert not hf.claims(str(tmp_path))
    assert hf.claims("Qwen/Qwen3-4B", files=["config.json", "model-00001-of-00002.safetensors"])
    assert not hf.claims("unsloth/Qwen3-4B-GGUF", files=["config.json", "a.safetensors", "Qwen3-4B-Q4_K_M.gguf"])
    assert not hf.claims("unsloth/Qwen3-4B-GGUF:Q4_K_M", files=["config.json", "a.safetensors"])
    assert not hf.claims("qwen3-8-27b") and not hf.claims("./x.gguf")
    out = hf.inspect(d)
    assert out["runs_here"] and out["arch"] == "qwen35" and out["layers"] == 4
    assert set(out["chips_ctx2048_slots8"]) == {n for n, _ in __import__("mashbox.gguf", fromlist=["CHIPS"]).CHIPS}
    r = hf.resolve(d)
    assert r["path"] == os.path.abspath(d) and r["files"] == ["model.safetensors"]


# ------------------------------------------------------------------------------------------------------- hub
def _hub_err(name):
    from huggingface_hub import errors as E
    cls = getattr(E, name)
    e = cls.__new__(cls)                    # the HTTP error classes want a live response; only the class matters here
    Exception.__init__(e, name)
    return e


def _fake_hub(monkeypatch, cfg_dir, info_err=None, file_err=None):
    import huggingface_hub as H
    from types import SimpleNamespace as NS
    calls = []

    def model_info(self, ref, **kw):
        calls.append(("info", ref))
        if info_err:
            raise _hub_err(info_err)
        return NS(sha="0" * 40, siblings=[NS(rfilename=p, size=1000) for p in ("config.json", "model.safetensors")])

    def hf_hub_download(ref, filename, revision=None, **kw):
        calls.append(("file", filename))
        if file_err:
            raise _hub_err(file_err)
        return os.path.join(cfg_dir, filename)

    def snapshot_download(ref, **kw):
        calls.append(("snapshot", ref))
        return cfg_dir

    def get_safetensors_metadata(self, ref, **kw):
        calls.append(("meta", ref))
        raise AssertionError("not reached in these tests")
    monkeypatch.setattr(H.HfApi, "model_info", model_info)
    monkeypatch.setattr(H.HfApi, "get_safetensors_metadata", get_safetensors_metadata)
    monkeypatch.setattr(H, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(H, "snapshot_download", snapshot_download)
    return calls


HUB_ERRORS = [
    ("info", "GatedRepoError", "is gated: accept its terms"),
    ("info", "RepositoryNotFoundError", "no such model on Hugging Face"),
    ("info", "RevisionNotFoundError", "no such model on Hugging Face"),
    ("file", "GatedRepoError", "is gated: accept its terms"),
    ("file", "RepositoryNotFoundError", "no such model on Hugging Face"),
    ("file", "EntryNotFoundError", "has no config.json with safetensors weights"),
]


@pytest.mark.parametrize("case", HUB_ERRORS, ids=[f"{w}-{e}" for w, e, _ in HUB_ERRORS])
def test_hub_errors_refuse_in_one_line(case, monkeypatch, tmp_path):
    where, err, msg = case
    calls = _fake_hub(monkeypatch, str(tmp_path), info_err=err if where == "info" else None,
                      file_err=err if where == "file" else None)
    out = hf.inspect("some-org/some-model")
    assert out["runs_here"] is False and msg in out["why"] and "\n" not in out["why"]
    for download in (False, True):
        with pytest.raises(hf.HFError, match=msg):
            hf.resolve("some-org/some-model", log=lambda *_: None, download=download)
    assert ("snapshot", "some-org/some-model") not in calls


def test_resolve_refuses_an_unmapped_architecture_before_the_weights(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(architectures=["Phi3ForCausalLM"])))
    calls = _fake_hub(monkeypatch, str(tmp_path))
    for download in (False, True):
        with pytest.raises(hf.HFError, match='architectures "Phi3ForCausalLM" is not read'):
            hf.resolve("microsoft/phi-4", log=lambda *_: None, download=download)
    assert ("file", "config.json") in calls and not any(k == "snapshot" for k, _ in calls)
    d, _, _ = fixture("qwen3")
    calls = _fake_hub(monkeypatch, d)
    r = hf.resolve("Qwen/Qwen3-4B", log=lambda *_: None)
    assert r["path"] == d and r["revision"] == "0" * 40 and r["files"] == ["model.safetensors"]
    assert [k for k, _ in calls] == ["info", "file", "snapshot"]


# ------------------------------------------------------------------------------------------------------- run entry
def _hf_load(monkeypatch, tmp_path, name, tag):
    """hf.load of a bf16 copy of fixture `name` through the parsed `run` args, as `mashbox run <folder>` does"""
    from mashbox import cli
    d, meta, _ = fixture(name)
    m = tmp_path / tag
    as_bf16(d, str(m))
    _tiny_tokenizer(str(m / "tokenizer.json"), ["<|im_start|>", "<|im_end|>"])
    (m / "chat_template.jinja").write_text("{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}")
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "setup_cache", lambda d=None: str(tmp_path / "xla"))
    a = cli.build_parser().parse_args(["run", str(m)])
    a.hf = hf.resolve(str(m))
    logs = []
    info = hf.load(a, log=logs.append)[3]
    return str(m), info, {json.loads(x)["event"]: json.loads(x) for x in logs if x.startswith("{")}


@pytest.mark.parametrize("name", ["qwen3", "qwen35"])
def test_load_log_counts_the_handles_each_path_holds(name, monkeypatch, tmp_path):
    """the weights and loaded events count the handles load made for the weights a contraction multiplies (Dense16 =
    native bf16, Raw = the exact path), never the file's types, and never the embedding table: its rows are gathered, not
    multiplied, as on the GGUF path. With the native op every multiplied weight is native bf16 and none is on the exact
    path; with it off, exactly the native bytes move to the exact path"""
    monkeypatch.delenv("MASHBOX_DENSE16", raising=False)
    m, on, ev_on = _hf_load(monkeypatch, tmp_path, name, "on")
    f, _, _ = hf.open_model(m)
    assert f.tensors["token_embd.weight"].nbytes > 0
    assert on["native_bf16_bytes"] > 0 and on["reference_bytes"] == 0 and on["reference_types"] == {}
    monkeypatch.setenv("MASHBOX_DENSE16", "off")
    _, off, ev_off = _hf_load(monkeypatch, tmp_path, name, "off")
    assert off["native_bf16_bytes"] == 0 and off["reference_bytes"] == on["native_bf16_bytes"]
    assert off["reference_types"] == {"BF16": on["native_bf16_bytes"]}
    for info, ev in ((on, ev_on), (off, ev_off)):
        assert all(ev["weights"][k] == info[k] for k in ("native_bf16_bytes", "reference_bytes", "reference_types"))
        assert ev["loaded"]["reference_gb"] == round(info["reference_bytes"] / 1e9, 2) >= 0


def test_run_args_resolve_load_and_generate_on_cpu(monkeypatch, tmp_path):
    from mashbox import cli
    d, meta, _ = fixture("qwen3")
    m = tmp_path / "qwen3"
    shutil.copytree(d, m)
    _tiny_tokenizer(str(m / "tokenizer.json"), ["<|im_start|>", "<|im_end|>"])
    (m / "chat_template.jinja").write_text("{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}")
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "setup_cache", lambda d=None: str(tmp_path / "xla"))     # no global JAX cache dir in tests
    a = cli.build_parser().parse_args(["run", str(m)])
    assert (a.ctx, a.slots, a.no_kernels) == (2048, 8, False)
    a.hf = hf.resolve(str(m))
    logs = []
    model, eng, chat, info, _ = hf.load(a, log=logs.append)
    ev = {json.loads(x)["event"]: json.loads(x) for x in logs if x.startswith("{")}
    assert ev["hf_plan"]["form"] == "bf16" and ev["hf_plan"]["arch"] == "qwen3" and ev["loaded"]["form"] == "bf16"
    assert info["form"] == "bf16" and model["id"] == str(m)
    ids = chat.chat_ids([dict(role="user", content="hello")])
    assert ids and all(0 <= i < meta["n_vocab"] for i in ids)
    r = eng.generate([ids], max_new=4, stop_ids=())
    assert len(r["ids"][0]) == 4 and all(0 <= int(i) < meta["n_vocab"] for i in r["ids"][0])


def test_hf_run_path_refuses_an_unlisted_moe_arch(monkeypatch, tmp_path):
    import types

    from mashbox.gguf import archs

    monkeypatch.setattr(hf, "open_model", lambda path, act=None, form=None: (types.SimpleNamespace(arch="qwen35moe", tensors={}), None, None))
    monkeypatch.setattr(hf, "fit_form", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a held MoE model (no passing row of its own) was loaded")))
    a = types.SimpleNamespace(hf=dict(path=str(tmp_path)), tensor_parallel_size=0, no_kernels=True, ctx=2048, slots=8)
    with pytest.raises(hf.HFError) as e:
        hf.load(a, log=lambda *x: None)
    assert str(e.value) == archs.unsupported("qwen35moe", str(tmp_path))


# ------------------------------------------------------------------------------------------------ mashbox run / inspect
def _model_folder(monkeypatch, tmp_path):
    """a bf16 copy of the qwen3 fixture with a tokenizer and chat template: what `mashbox run <folder>` is given"""
    from mashbox import cli
    d, meta, _ = fixture("qwen3")
    m = tmp_path / "Qwen3-tiny"
    as_bf16(d, str(m))
    _tiny_tokenizer(str(m / "tokenizer.json"), ["<|im_start|>", "<|im_end|>"])
    (m / "chat_template.jinja").write_text("{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}")
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "setup_cache", lambda d=None: str(tmp_path / "xla"))
    return str(m)


class _Hub:
    """huggingface_hub.HfApi with a fixed listing; counts the calls; no network"""
    calls = []

    def __init__(self, files):
        self.files = files

    def __call__(self, *a, **k):
        return self

    def model_info(self, repo, **k):
        import types
        _Hub.calls.append(("model_info", repo))
        if self.files is None:
            raise AssertionError(f"the Hub was asked about {repo}")
        return types.SimpleNamespace(sha="1" * 40, siblings=[types.SimpleNamespace(rfilename=p, size=n) for p, n in self.files])

    def list_repo_tree(self, repo, revision=None, recursive=False):
        from huggingface_hub.hf_api import RepoFile
        _Hub.calls.append(("list_repo_tree", repo))
        return [RepoFile(path=p, size=n, oid="0" * 40) for p, n in self.files]


def _hub(monkeypatch, files):
    import huggingface_hub
    _Hub.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _Hub(files))
    return _Hub.calls


def test_mashbox_run_folder_prompt_answers_through_the_hf_path(monkeypatch, tmp_path, capsys):
    """`mashbox run <folder> --prompt` loads the folder's safetensors with hf.load and answers on CPU, zero flags"""
    from mashbox import cli
    m = _model_folder(monkeypatch, tmp_path)
    calls = _hub(monkeypatch, None)                         # a local folder never asks the Hub
    loads = []
    real = hf.load
    monkeypatch.setattr(hf, "load", lambda a, log=None: loads.append(a.hf["path"]) or real(a, log))
    rc = cli.main(["run", m, "--prompt", "hello", "--max-tokens", "4"])
    err = capsys.readouterr().err
    ev = {json.loads(x)["event"]: json.loads(x) for x in err.splitlines() if x.startswith("{")}
    assert rc in (None, 0) and loads == [m] and calls == []
    assert ev["hf_plan"]["form"] == "bf16" and ev["hf_plan"]["arch"] == "qwen3"
    assert ev["done"]["new_tokens"] == 4


def test_mashbox_run_folder_serves_in_this_process(monkeypatch, tmp_path):
    """without --prompt a Hugging Face model is served in this process (as a GGUF file), never through the worker front"""
    from mashbox import cli
    from mashbox.runtime import front, server
    m = _model_folder(monkeypatch, tmp_path)
    monkeypatch.setattr(front, "serve_cli", lambda *a, **k: (_ for _ in ()).throw(AssertionError("worker front used")))
    served = []
    monkeypatch.setattr(server, "serve", lambda srv, host, port, log, ui=None, keys=None: served.append((srv, host, port)))
    monkeypatch.setattr(cli, "api_keys", lambda a: [])
    rc = cli.main(["run", m, "--no-warm"])
    assert rc in (None, 0) and len(served) == 1
    srv, host, port = served[0]
    assert srv.ready and host == "127.0.0.1"


def test_a_hub_repo_with_safetensors_and_no_gguf_runs_through_hf(monkeypatch, tmp_path):
    """owner/name whose listing has config.json + safetensors and no .gguf goes to hf.resolve (the GGUF listing is reused)"""
    from mashbox import cli
    calls = _hub(monkeypatch, [("config.json", 700), ("model.safetensors", 1 << 20), ("tokenizer.json", 9)])
    seen = []

    def stop(ref, log=print, download=True):
        seen.append(ref)
        raise hf.HFError("mashbox: stopped by the test")
    monkeypatch.setattr(hf, "resolve", stop)
    assert cli.main(["run", "someone/Tiny-Model", "--prompt", "hi"]) == 2
    assert seen == ["someone/Tiny-Model"] and calls.count(("model_info", "someone/Tiny-Model")) == 1


def test_a_gguf_repo_is_listed_once_and_never_goes_to_hf(monkeypatch):
    from mashbox import cli
    calls = _hub(monkeypatch, [("config.json", 700), ("model.safetensors", 1 << 20), ("model-Q8_0.gguf", 1 << 20)])
    monkeypatch.setattr(hf, "resolve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hf path used")))
    assert cli.main(["inspect", "someone/Tiny-Model-GGUF", "--no-download"]) == 0
    assert calls == [("model_info", "someone/Tiny-Model-GGUF"), ("list_repo_tree", "someone/Tiny-Model-GGUF")]


def test_inspect_a_hub_safetensors_repo_prints_hf_inspect(monkeypatch, capsys):
    from mashbox import cli
    _hub(monkeypatch, [("config.json", 700), ("model.safetensors", 1 << 20)])
    monkeypatch.setattr(hf, "inspect", lambda ref, log=print: dict(model=ref, runs_here=True))
    assert cli.main(["inspect", "someone/Tiny-Model", "--no-download"]) == 0
    assert json.loads(capsys.readouterr().out) == dict(model="someone/Tiny-Model", runs_here=True)


def test_a_models_json_id_never_asks_the_hub(monkeypatch):
    """guard: catalog ids keep their own path and make no Hub call to decide it"""
    from mashbox import cli
    calls = _hub(monkeypatch, None)
    monkeypatch.setattr(cli, "cmd_run", lambda a, argv=None: 0)
    assert cli.main(["run", "qwen3-8-27b", "--prompt", "hi"]) == 0 and calls == []


@pytest.mark.parametrize("args, says", [
    (["--speculation", "ngram"], "--speculation"),
    (["--weights", "x.bshj"], "--weights"),
    (["-tp", "2"], "one chip"),
])
def test_hf_run_refusals_are_one_line_exit_2(args, says, monkeypatch, tmp_path, capsys):
    from mashbox import cli
    m = _model_folder(monkeypatch, tmp_path)
    monkeypatch.setattr(hf, "open_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("weights opened")))
    assert cli.main(["run", m, "--prompt", "hi"] + args) == 2
    err = [x for x in capsys.readouterr().err.splitlines() if x.strip() and not x.startswith("{")]
    assert len(err) == 1 and err[0].startswith("mashbox: ") and says in err[0], err


def test_a_folder_that_is_not_a_model_refuses_in_one_line(monkeypatch, tmp_path, capsys):
    from mashbox import cli
    d = tmp_path / "notes"
    d.mkdir()
    (d / "readme.txt").write_text("x")
    _hub(monkeypatch, None)
    assert cli.main(["run", str(d), "--prompt", "hi"]) == 2
    err = [x for x in capsys.readouterr().err.splitlines() if x.strip()]
    assert len(err) == 1 and "config.json" in err[0] and "safetensors" in err[0], err
