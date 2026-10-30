"""GGUF architecture gemma4 (gguf/archs/gemma4.py): our logits vs llama.cpp's CPU logits on tiny random fixtures (no chip,
no network, no secrets). tests/fixtures/gguf/<name>.json says how each was made (llama.cpp commit, file sha256).

  gemma4-f32     float32 weights, 6 layers (sliding, sliding, full, sliding, sliding, full): the graph test. Sliding
                 layers have 2 K/V heads of 32 and a window of 4 (the 16 test tokens cross it); full layers 1 K/V head of
                 64, no attn_v (v is the k projection) and rope_freqs factors; q/k norms, post norms, layer_output_scale,
                 tied head, soft-capped logits. Bound NMSE <= 1e-4, llama.cpp's cross-backend bar.
  gemma4-kquant  the same graph in Q4_K / Q5_K / Q6_K / Q8_0 blocks written by llama.cpp's llama-quantize. Bars as
                 test_gguf_dense.py, fixed before the first run: mean KL(llama.cpp || ours) <= 1e-3 and the same top-1 on
                 >= 15 of 16 vs llama.cpp's block path, and NMSE <= 1e-4 vs llama.cpp on an F32 copy holding the identical
                 dequantized weights (<name>.f32ref).
Every fixture runs twice: one prefill of all 16 tokens, and a prefill of 8 then 8 one-token decode steps (K/V cache path,
sliding window across the cache).
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
from mashbox.gguf.archs import gemma4 as A  # noqa: E402

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


@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_float_logits_match_llama_cpp(split, tmp_path):
    p, meta, ref = fixture("gemma4-f32", tmp_path)
    ours, c, info = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"gemma4-f32 {split}: NMSE {e:.3e}")
    assert e <= NMSE_MAX, e


@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_type_logits_match_llama_cpp(split, tmp_path):
    p, meta, ref = fixture("gemma4-kquant", tmp_path)
    ours, c, info = run(p, meta["tokens"], split=split)
    kl, top1 = kl_top1(ref, ours)
    print(f"gemma4-kquant {split}: KL {kl:.3e} top1 {top1}/{len(ref)} NMSE {nmse(ours, ref):.3e} types {info['reference_types']}")
    assert kl <= KL_MAX and top1 >= TOP1_MIN, (kl, top1)


@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_types_match_llama_cpp_on_identical_weights(split, tmp_path):
    p, meta, _ = fixture("gemma4-kquant", tmp_path)
    ref = np.fromfile(os.path.join(FX, "gemma4-kquant.f32ref.logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    ours, c, info = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"gemma4-kquant {split} vs f32ref: NMSE {e:.3e}")
    assert e <= NMSE_MAX, e


def test_window_is_crossed_and_layers_differ(tmp_path):
    """the fixture exercises what makes gemma4 different: 16 tokens > the window, two head shapes, a layer without attn_v"""
    f, M, c = open_model(FXT.build("gemma4-f32", str(tmp_path)), act=jnp.float32)
    assert c.window == 4 and c.swa == [True, True, False, True, True, False] and c.softcap == 30.0
    assert (c.kv[0], c.hd[0], c.kv[2], c.hd[2]) == (2, 32, 1, 64) and (c.KV, c.HD) == (1, 64)
    assert "blk.2.attn_v.weight" not in f.tensors and "blk.0.attn_v.weight" in f.tensors
    p, _, _ = fixture("gemma4-kquant", tmp_path)
    names = {t.type_name for n, t in File(p).tensors.items() if n.startswith("blk.")}
    assert {"Q4_K", "Q8_0"} <= names, names


def _header_only(path, extra, ff=16):
    w = gguf.GGUFWriter(str(path), "gemma4")
    w.add_block_count(1)
    w.add_embedding_length(8)
    w.add_feed_forward_length(ff)
    w.add_head_count(2)
    w.add_key_length(4)
    w.add_value_length(4)
    w.add_layer_norm_rms_eps(1e-6)
    names = ["token_embd.weight"]
    for k, v in extra:
        if k.startswith("tensor:"):
            names.append(k[7:])
        else:
            w.add_uint32(f"gemma4.{k}", v)
    for n in names:
        w.add_tensor(n, np.zeros((4, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.mark.parametrize("extra,words", [
    ([("embedding_length_per_layer_input", 256)], "per-layer token embeddings"),
    ([("attention.shared_kv_layers", 2)], "shared K/V layers"),
    ([("tensor:blk.0.ffn_gate_inp.weight", 0)], "routed experts")])
def test_refusals_are_one_line(extra, words, tmp_path):
    p = _header_only(tmp_path / "m.gguf", extra)
    with pytest.raises(GGUFError) as e:
        open_model(p)
    msg = str(e.value)
    assert words in msg and "mashbox: gemma4 file with" in msg and "\n" not in msg, msg


@pytest.mark.parametrize("extra,words", [
    ([("embedding_length_per_layer_input", 256)], "per-layer token embeddings"),
    ([], "per-layer feed-forward lengths")])
def test_a_per_layer_ffn_list_is_refused_in_one_line(extra, words, tmp_path):
    """feed_forward_length stored as one value per layer (an array, as in unsloth's gemma-4-E2B mobile files) gets the
    one-line refusal of any other unsupported gemma4 file, never a traceback; with per-layer token embeddings too (the
    E2B file's case) the message names those first, as it does for an E4B file"""
    p = _header_only(tmp_path / "m.gguf", extra, ff=[16])
    with pytest.raises(GGUFError) as e:
        open_model(p)
    msg = str(e.value)
    assert words in msg and "mashbox: gemma4 file with" in msg and "\n" not in msg, msg


def test_run_prompt_end_to_end_on_cpu(tmp_path, capsys, monkeypatch):
    from mashbox import cli
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = FXT.build("gemma4-f32", str(tmp_path))
    rc = cli.main(["run", p, "--prompt", "the hat", "--max-tokens", "6", "--ctx", "64", "--slots", "2", "--steps", "2",
                   "--cache-dir", str(tmp_path / "xla")])
    cap = capsys.readouterr()
    assert rc in (None, 0), cap.err
    events = [json.loads(l) for l in cap.err.splitlines() if l.startswith("{")]
    kinds = [e["event"] for e in events]
    assert "gguf_plan" in kinds and "loaded" in kinds and "done" in kinds, cap.err
    assert next(e for e in events if e["event"] == "gguf_plan")["arch"] == "gemma4"
    assert next(e for e in events if e["event"] == "done")["new_tokens"] >= 1


# ------------------------------------------------------------------------------- sliding layers: the K/V ring
# The sliding-window layers keep a ring of min(window, ctx) positions (runtime/kvcache.py ring_write / ring_pos), held by
# the engine as snapshotted slot state. The checks compare logits, not only the argmax (an argmax-only check let 4 of 5
# planted ring bugs pass on this random fixture). Against a
# no-cache forward of the same ids (one call over the whole sequence, nothing read from a cache),
#   * the engine's teacher-forced chunked logits (Engine.logits: 16-token chunks through the cache, a padded last chunk)
#     differ by at most 1e-3 at every position;
#   * every token the engine generates (batched rows of different lengths, chunked prefill, decode steps across the
#     window, a prefix-cache resume, a host-cache restore) has top-4 logits (the decode program's own, MASHBOX_RECORD_TOPK)
#     within 1e-3 of the no-cache logits of those ids at that position, and is their argmax unless the no-cache top-two
#     gap is under 1e-3.
# The window is crossed in every run. Window 4 is the file's (every prefill chunk is longer than the ring); 24 makes a
# 16-token chunk wrap around the ring's end.
TOL = 1e-3


@pytest.mark.parametrize("window", [4, 24])
def test_ring_engine_runs_equal_a_no_cache_forward(window, tmp_path, monkeypatch):
    from mashbox.runtime.engine import Engine
    monkeypatch.setenv("MASHBOX_RECORD_TOPK", "4")
    f, M, c = open_model(FXT.build("gemma4-f32", str(tmp_path)), act=jnp.float32)
    c.window = window
    W, _ = A.load(f, c, use_kernels=False)
    CTX, i32 = 64, jnp.int32
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck0, cv0 = A.new_state(c, 1, CTX)
    assert cv0[0].shape[2] == min(window, CTX) and ck0[0].shape[0] == c.swa.count(False)

    def nocache(seq):
        t = np.zeros((1, CTX), np.int32)
        t[0, :len(seq)] = seq
        return np.asarray(fwd(W, ck0, cv0, jnp.asarray(t), jnp.zeros((1,), i32), jnp.asarray([len(seq)], i32),
                              jnp.zeros((1,), i32), 0, all_logits=True)["lg"][0])[:len(seq)]

    def check(prompt, ids, topk):
        """-> (largest top-4 logit difference, generated ids that are not the no-cache argmax outside a near tie)"""
        seq = prompt + ids
        lg = nocache(seq)
        worst, bad = 0.0, 0
        for k, (tid, tv) in enumerate(topk):
            row = lg[len(prompt) + k - 1]
            worst = max(worst, float(np.abs(row[np.asarray(tid)] - np.asarray(tv)).max()))
            top = np.sort(row)[-2:]
            bad += int(int(row.argmax()) != ids[k] and top[1] - top[0] >= TOL)
        return worst, bad

    rng = np.random.default_rng(7)
    V = c.V
    ps = [rng.integers(1, V, n).tolist() for n in (5, 23, 40)]
    eng = Engine(c, W, A, CTX=CTX, S=3, K=3, chunk=16, prefix_cache=True, cache_ram=1 << 26, ckpt_step=8, trace=True)
    seq = ps[2] + rng.integers(1, V, 20).tolist()
    assert np.abs(eng.logits(seq) - nocache(seq)).max() <= TOL
    r = eng.generate(ps, max_new=20)
    got = [check(p, g, k) for p, g, k in zip(ps, r["ids"], r["topk"])]
    assert all(w <= TOL and b == 0 for w, b in got), got
    assert all(len(p) + len(g) > window for p, g in zip(ps, r["ids"]))
    # a second turn of the 23-token conversation: resumed at its prompt end (the ring put back from its snapshot)
    t2 = ps[1] + r["ids"][1] + rng.integers(1, V, 6).tolist()
    r2 = eng.generate([t2], max_new=12)
    w, b = check(t2, r2["ids"][0], r2["topk"][0])
    assert r2["cached"] == [len(ps[1])] and w <= TOL and b == 0, (r2["cached"], w, b)
    # three other conversations take every slot (t2's slot goes to host memory), then t2's next turn comes back restored
    eng.generate([rng.integers(1, V, 9).tolist() for _ in range(3)], max_new=4)
    t3 = t2 + r2["ids"][0][:5]
    r3 = eng.generate([t3], max_new=8)
    w, b = check(t3, r3["ids"][0], r3["topk"][0])
    assert r3["restored"] == [len(t2)] and w <= TOL and b == 0, (r3["restored"], w, b)
