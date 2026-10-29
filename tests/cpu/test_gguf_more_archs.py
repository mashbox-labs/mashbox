"""GGUF architectures gemma3 (gguf/archs/gemma3.py on the gemma4 graph), exaone and granite (gguf/archs/dense.py), and
SentencePiece vocabularies (gguf/spm.py): our logits and token ids vs llama.cpp's on tiny random fixtures (no chip, no
network, no secrets). tests/fixtures/gguf/<name>.json says how each was made (llama.cpp commit, file sha256); the
files are rebuilt from their seeds by tools/gguf_fixtures/more.py.

  gemma3-f32      6 layers: 5 sliding-window layers (window 4) + 1 full layer with linear rope scaling 8, the default
                  sliding-window rope base, q scaled by 1/sqrt(key_length) with key_length != embedding_length /
                  head_count, no v norm, tied head, a SentencePiece vocabulary.
  exaone-f32      llama tensors with the rotation on halves (NEOX) and rope frequency factors.
  granite-f32     llama tensors with attention.scale, embedding_scale, residual_scale, logit_scale, tied head.
  *-pyquant       Q8_0 / Q4_0 / Q5_0 / Q4_1 / Q5_1 blocks written by gguf-py.
Bars (fixed before the first run, as test_gguf_dense.py): NMSE <= 1e-4 on float weights and vs llama.cpp on an F32 copy
holding the identical dequantized weights; mean KL <= 1e-3 and the same top-1 on >= 15 of 16 vs llama.cpp's block path.
Token ids: identical to llama-tokenize on every text of more.TEXTS, with and without a space prefix.
exaone and granite are not in ARCHS while their files' pre-tokenizers ("exaone", "refact") are not read: their graphs
are tested here through dense.py directly, and `inspect` refuses their files.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

import gguf  # noqa: F401  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")
_spec = importlib.util.spec_from_file_location("gguf_fixtures_more", os.path.join(ROOT, "tools", "gguf_fixtures", "more.py"))
MORE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MORE)

from mashbox.gguf import open_model, GGUFError  # noqa: E402
from mashbox.gguf.reader import File  # noqa: E402

NMSE_MAX = 1e-4
KL_MAX, TOP1_MIN = 1e-3, 15
MODULE = {"gemma3": "mashbox.gguf.archs.gemma3", "exaone": "mashbox.gguf.archs.dense",
          "granite": "mashbox.gguf.archs.dense"}


def fixture(name, tmp, ref="logits"):
    meta = json.load(open(os.path.join(FX, f"{name}.json")))
    p = MORE.build(name, str(tmp))
    assert MORE.sha256(p) == meta["gguf_sha256"], f"{name}: fixture bytes changed; rerun tools/gguf_fixtures/more.py make"
    lg = np.fromfile(os.path.join(FX, f"{name}.{ref}.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    return p, meta, lg


def open_graph(path):
    """open_model, or for an architecture with a graph but no ARCHS line yet, its module's Config directly"""
    from mashbox.gguf import archs
    f = File(path)
    if f.arch in archs.ARCHS:
        return open_model(path, act=jnp.float32)
    M = importlib.import_module(MODULE[f.arch])
    return f, M, M.Config(f, act=jnp.float32)


def run(path, toks, split=None):
    f, M, c = open_graph(path)
    assert M.__name__ == MODULE[f.arch], (f.arch, M.__name__)
    W, info = M.load(f, c, use_kernels=False)
    fwd = jax.jit(M.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = M.new_state(c, 1, 32)
    i32 = jnp.int32
    z = jnp.zeros((1,), i32)
    T = len(toks)
    t = jnp.asarray([toks], i32)
    if split is None:
        return np.asarray(fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)["lg"][0]), c
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out = [np.asarray(r["lg"][0])]
    ck, cv = r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0), c


def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def kl_top1(ref, ours):
    ref, ours = ref.astype(np.float64), ours.astype(np.float64)    # float32 exp underflows to 0 on these logits
    p = np.exp(ref - ref.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    kl = np.sum(p * (np.log(np.maximum(p, 1e-300)) - lq), -1)
    return float(kl.mean()), int(np.sum(ref.argmax(-1) == ours.argmax(-1)))


@pytest.mark.parametrize("name", ["gemma3-f32", "exaone-f32", "granite-f32"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_float_logits_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = fixture(name, tmp_path)
    ours, c = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"{name} {split}: NMSE {e:.3e}")
    assert e <= NMSE_MAX, e


@pytest.mark.parametrize("name", ["gemma3-pyquant", "exaone-pyquant", "granite-pyquant"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_types_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = fixture(name, tmp_path)
    ours, c = run(p, meta["tokens"], split=split)
    kl, top1 = kl_top1(ref, ours)
    _, _, ref32 = fixture(name, tmp_path, ref="f32ref.logits")
    e = nmse(ours, ref32)
    print(f"{name} {split}: KL {kl:.3e} top1 {top1}/{len(ref)} NMSE vs f32ref {e:.3e}")
    assert kl <= KL_MAX and top1 >= TOP1_MIN, (kl, top1)
    assert e <= NMSE_MAX, e


def test_config_reads_each_arch(tmp_path):
    f, M, c = open_model(MORE.build("gemma3-f32", str(tmp_path)), act=jnp.float32)
    assert c.swa == [True] * 5 + [False] and c.window == 4 and c.theta == [1e4] * 5 + [1e6], (c.swa, c.theta)
    assert c.fscale == [1.0] * 5 + [1.0 / 8.0] and not c.vnorm and not c.freqs and c.softcap == 0.0
    assert abs(c.qscale - 32 ** -0.5) < 1e-7 and c.D // c.H == 64 and c.hd == [32] * 6
    f, M, c = open_graph(MORE.build("exaone-f32", str(tmp_path)))
    assert (c.neox, c.freqs, c.tie, c.emb_scale, c.res_scale, c.logit_scale) == (True, True, False, 1.0, 1.0, 1.0)
    f, M, c = open_graph(MORE.build("granite-f32", str(tmp_path)))
    assert (c.neox, c.tie, c.scale, c.emb_scale, c.logit_scale) == (False, True, 0.0625, 12.0, 8.0)
    assert abs(c.res_scale - 0.22) < 1e-7


def test_gemma4_config_keeps_its_graph(tmp_path):
    """the three gemma3 switches default to gemma4's own graph (v norm, no q scale, no rope scaling)"""
    _fx = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
    base = importlib.util.module_from_spec(_fx)
    _fx.loader.exec_module(base)
    f, M, c = open_model(base.build("gemma4-f32", str(tmp_path)), act=jnp.float32)
    assert M.__name__ == "mashbox.gguf.archs.gemma4"
    assert (c.vnorm, c.qscale, c.fscale) == (True, 1.0, [1.0] * c.L)


def _gemma3_header(path, extra=(), tensors=("token_embd.weight",)):
    w = gguf.GGUFWriter(str(path), "gemma3")
    w.add_block_count(1)
    w.add_embedding_length(8)
    w.add_feed_forward_length(16)
    w.add_head_count(2)
    w.add_layer_norm_rms_eps(1e-6)
    for k, v in extra:
        w.add_string(f"gemma3.{k}", v)
    for n in tensors:
        w.add_tensor(n, np.zeros((4, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.mark.parametrize("extra,tensors,words", [
    ([("rope.scaling.type", "yarn")], ("token_embd.weight",), "rope scaling 'yarn'"),
    ([], ("token_embd.weight", "blk.0.attn_qkv.weight"), "a fused qkv tensor"),  # gitleaks:allow
    ([], ("token_embd.weight", "dense_2.weight"), "dense output projections")])
def test_gemma3_refusals_are_one_line(extra, tensors, words, tmp_path):
    with pytest.raises(GGUFError) as e:
        open_model(_gemma3_header(tmp_path / "m.gguf", extra, tensors))
    msg = str(e.value)
    assert words in msg and "mashbox: gemma3 file with" in msg and "\n" not in msg, msg


def test_spm_tokenizer_matches_llama_tokenize(tmp_path):
    """SentencePiece ids (no BOS, special tokens parsed) == llama.cpp's llama-tokenize on the fixture's vocabulary"""
    from tokenizers import Tokenizer
    from mashbox.gguf import tokenizer as GT
    want = json.load(open(os.path.join(FX, "gemma3-spm.tokens.json"), encoding="utf-8"))
    p = MORE.build("gemma3-f32", str(tmp_path))
    assert MORE.sha256(p) == want["gguf_sha256"]
    tok = Tokenizer.from_str(GT.files_dict(File(p), p)["tokenizer.json"])
    bad = [(r["text"], tok.encode(r["text"], add_special_tokens=False).ids, r["ids"]) for r in want["rows"]
           if tok.encode(r["text"], add_special_tokens=False).ids != r["ids"]]
    assert len(want["rows"]) == len(MORE.TEXTS) and not bad, bad


@pytest.mark.parametrize("variant", ["true", "absent"])
def test_spm_space_prefix_matches_llama_tokenize(variant, tmp_path):
    """the space prefix of Llama 2 and Mistral 7B style vocabularies: tokenizer.ggml.add_space_prefix true, and the key
    left out (llama.cpp reads that as true): ids == llama-tokenize on every text"""
    from tokenizers import Tokenizer
    from mashbox.gguf import tokenizer as GT
    want = json.load(open(os.path.join(FX, "spm-prefix.tokens.json"), encoding="utf-8"))["variants"][variant]
    p = MORE.write_gemma3(str(tmp_path / f"prefix-{variant}.gguf"), seed=MORE.FIXTURES["gemma3-f32"]["seed"], layers=1,
                          prefix=MORE.PREFIX[variant])
    assert MORE.sha256(p) == want["gguf_sha256"]
    tok = Tokenizer.from_str(GT.files_dict(File(p), p)["tokenizer.json"])
    bad = [(r["text"], tok.encode(r["text"], add_special_tokens=False).ids, r["ids"]) for r in want["rows"]
           if tok.encode(r["text"], add_special_tokens=False).ids != r["ids"]]
    assert len(want["rows"]) == len(MORE.TEXTS) and not bad, bad


@pytest.mark.parametrize("arch,pre", [("granite", "refact"), ("exaone", "exaone")])
def test_inspect_says_runs_only_when_the_tokenizer_is_read(arch, pre, tmp_path):
    """the real granite and EXAONE files carry pre-tokenizers "refact" and "exaone": inspect may say runs_here only when
    this version reads the file's tokenizer too, since run stops at it"""
    from mashbox import gguf as G
    from mashbox.gguf import tokenizer as GT
    fx = MORE.FIXTURES[f"{arch}-f32"]
    p = MORE.write_dense2(str(tmp_path / f"{arch}-{pre}.gguf"), arch, seed=fx["seed"], layers=fx["layers"], pre=pre)
    try:
        GT.files_dict(File(p), p)
        read = True
    except GGUFError:
        read = False
    r = G.inspect(p)
    assert r["runs_here"] == read, r.get("why") or r.get("plan")


def test_spm_chat_has_one_bos_and_round_trips(tmp_path):
    from mashbox.gguf import tokenizer as GT
    from mashbox.runtime.chat import Chat
    p = MORE.build("gemma3-f32", str(tmp_path))
    ch = Chat(GT.model_dir(File(p), p, root=str(tmp_path / "tok")))
    ids = ch.chat_ids([dict(role="user", content="the hat")])
    toks, _, _ = MORE.spm_vocab()
    assert ids[0] == 2 and ids.count(2) == 1 and ids[1] == toks.index("<start_of_turn>"), ids[:4]
    for t in ("the hat sat on the mat.\n", "über 東京 🙂", "  two  spaces"):
        assert ch.decode(ch.encode(t)) == t


def test_spm_merges_follow_scores():
    toks = ["a", "b", "ab", "ba", "aba", "c"]
    scores = [0.0, 0.0, -1.0, -3.0, -2.0, 0.0]
    from mashbox.gguf.spm import merges_from_scores
    m = merges_from_scores(toks, scores)          # by score, then each piece's splits by the halves' ids
    assert m == [["a", "b"], ["a", "ba"], ["ab", "a"], ["b", "a"]], m


def test_run_gemma3_prompt_end_to_end_on_cpu(tmp_path, capsys, monkeypatch):
    from mashbox import cli
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = MORE.build("gemma3-f32", str(tmp_path))
    rc = cli.main(["run", p, "--prompt", "the hat", "--max-tokens", "6", "--ctx", "64", "--slots", "2", "--steps", "2",
                   "--cache-dir", str(tmp_path / "xla")])
    cap = capsys.readouterr()
    assert rc in (None, 0), cap.err
    events = [json.loads(l) for l in cap.err.splitlines() if l.startswith("{")]
    assert next(e for e in events if e["event"] == "gguf_plan")["arch"] == "gemma3"
    assert next(e for e in events if e["event"] == "done")["new_tokens"] >= 1
