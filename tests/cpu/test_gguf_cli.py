"""GGUF from the command line, on CPU with no network: llama.cpp's file pick, the refusals (one line, exit 2, never a
stack trace), the tokenizer built from the file, and `mashbox run <file.gguf> --prompt` end to end on the tiny fixture."""
import argparse
import importlib.util
import json
import os

import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
pytest.importorskip("jax")

from mashbox import cli  # noqa: E402
from mashbox.gguf import pick as GR, tokenizer as GT, File  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

# unsloth/Qwen3.8-27B-GGUF @ 4ca7207, in the Hub's order (listed 2026-10-28)
UNSLOTH = ["BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
           "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q4_1.gguf", "Qwen3.8-27B-Q8_0.gguf",
           "Qwen3.8-27B-UD-IQ1_M.gguf", "Qwen3.8-27B-UD-IQ1_S.gguf", "Qwen3.8-27B-UD-IQ2_S.gguf",
           "Qwen3.8-27B-UD-IQ2_XXS.gguf", "Qwen3.8-27B-UD-IQ3_S.gguf", "Qwen3.8-27B-UD-IQ3_XXS.gguf",
           "Qwen3.8-27B-UD-IQ4_XS.gguf", "Qwen3.8-27B-UD-Q2_K_XL.gguf", "Qwen3.8-27B-UD-Q3_K_XL.gguf",
           "Qwen3.8-27B-UD-Q4_K_M.gguf", "Qwen3.8-27B-UD-Q4_K_S.gguf", "Qwen3.8-27B-UD-Q4_K_XL.gguf",
           "Qwen3.8-27B-UD-Q5_K_M.gguf", "Qwen3.8-27B-UD-Q8_K_XL.gguf", "README.md", "config.json",
           "imatrix_unsloth.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf"]


@pytest.mark.parametrize("tag,want", [
    (None, "Qwen3.8-27B-UD-Q4_K_M.gguf"),           # llama.cpp's default: first Q4_K_M ...
    ("Q8_0", "Qwen3.8-27B-Q8_0.gguf"),
    ("q4_0", "Qwen3.8-27B-Q4_0.gguf"),              # any case; the MTP sidecar is never the model
    ("BF16", "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"),
    ("UD-Q4_K_XL", "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
    ("F16", "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"),   # as llama.cpp: an unanchored search, "F16-" is in "BF16-"
    ("Q9_9", None),
])
def test_pick_follows_llama_cpp(tag, want):
    assert GR.pick(UNSLOTH, tag) == want


def test_pick_fallbacks_and_shards():
    no_default = ["a/model-IQ2_M.gguf", "imatrix.gguf", "mmproj-x.gguf", "b/model-00002-of-00002.gguf",
                  "b/model-00001-of-00002.gguf"]
    assert GR.pick(no_default) == "a/model-IQ2_M.gguf"              # no Q4_K_M / Q8_0: the first model file
    assert GR.pick(["m-Q8_0-00002-of-00002.gguf", "m-Q8_0-00001-of-00002.gguf"]) == "m-Q8_0-00001-of-00002.gguf"
    assert GR.shards("m-Q8_0-00001-of-00003.gguf") == [f"m-Q8_0-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    assert GR.split_repo_tag("hf.co/unsloth/Qwen3.8-27B-GGUF:Q8_0") == ("unsloth/Qwen3.8-27B-GGUF", "Q8_0")
    assert GR.is_gguf_request("unsloth/Qwen3.8-27B-GGUF") and not GR.is_gguf_request("qwen3-8-27b")


def _tiny(path, arch, tensors=True):
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    w.add_embedding_length(8)
    if tensors:
        w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.mark.parametrize("arch,words", [("t5", "does not run on TPU in mashbox"), ("qwen9next", "is newer than the GGUF reader")])
def test_unsupported_arch_is_one_line_with_the_llama_cpp_command(arch, words, tmp_path, capsys):
    p = _tiny(tmp_path / "m.gguf", arch)
    assert cli.main(["run", p]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and words in err[0] and f"llama-server -m {p}" in err[0], err
    assert "Traceback" not in "\n".join(err)


def _draft(path, d2t=True):
    """a draft head for speculative decoding laid out as HauhauCS's ...-FastMTP-32K.gguf (header read 2026-10-30): qwen35,
    block_count 3 with 1 MTP block, its embedding_length (the real file has 5120), tensors only in that block, a reduced
    output vocabulary and, with d2t, the int32 draft-to-target vocabulary map"""
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_block_count(3)
    w.add_embedding_length(8)
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_tensor("token_embd.weight", np.zeros((16, 8), np.float32))
    w.add_tensor("output.weight", np.zeros((4, 8), np.float32))
    w.add_tensor("blk.2.attn_norm.weight", np.zeros(8, np.float32))
    w.add_tensor("blk.2.nextn.eh_proj.weight", np.zeros((8, 16), np.float32))
    if d2t:
        w.add_tensor("d2t", np.arange(4, dtype=np.int32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.mark.parametrize("d2t", [True, False])
def test_a_draft_head_is_one_line_that_says_so(d2t, tmp_path, capsys):
    """A draft file (MTP block only, d2t map or not) is not a whole model: one line naming it, exit 2, and no claim that
    llama.cpp runs it (llama.cpp a37cc46 refuses HauhauCS's FastMTP-32K file too: 'output.weight' has wrong shape)"""
    p = _draft(tmp_path / "m-FastMTP-32K.gguf", d2t)
    assert cli.main(["run", p]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "draft for speculative decoding, not a whole model" in err[0], err
    assert "block 2 of 3" in err[0] and ("4-token draft vocabulary (d2t)" in err[0]) == d2t, err
    assert "llama-server" not in err[0] and "Traceback" not in "\n".join(err), err


@pytest.mark.parametrize("flags", [["--mtp", "2"], ["--mtp-file", "draft.gguf"]])
def test_speculation_flags_on_a_gguf_file_are_one_line(flags, tmp_path, capsys):
    p = FXT.build("qwen35-f32", str(tmp_path))
    assert cli.main(["run", p, *flags]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "--mtp and --mtp-file are for models.json models" in err[0], err


def test_missing_file_and_inspect(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path / "nope.gguf")]) == 2
    assert "no such file" in capsys.readouterr().err
    p = FXT.build("qwen35-f32", str(tmp_path))
    assert cli.main(["inspect", p]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["arch"] == "qwen35" and out["runs_here"] and out["plan"]["layers"] == 8 and out["tokenizer"]["pre"] == "qwen35"


def test_tokenizer_from_the_file(tmp_path):
    p = FXT.build("qwen35-f32", str(tmp_path))
    from mashbox.runtime.chat import Chat
    c = Chat(GT.model_dir(File(p), p, root=str(tmp_path / "tok")))
    toks, types, merges = FXT.vocab()
    assert c.stop_ids == (len(toks) - 1, len(toks) - 3)                        # eos <|im_end|>, then <|endoftext|>
    ids = c.chat_ids([dict(role="user", content="in the hat")])
    assert ids[0] == toks.index("<|im_start|>") and toks.index("<|im_end|>") in ids
    assert toks.index("in") in ids and toks.index("\u0120the") in ids                  # merges applied, then " the"
    assert c.decode(c.encode("the in a hat\n")) == "the in a hat\n"


def test_qwen35_tokenizer_keeps_decomposed_text_as_llama_cpp_does(tmp_path):
    """No NFC for pre-tokenizer qwen35: llama.cpp a37cc46 and the model's own tokenizer (llama.cpp's
    models/ggml-vocab-qwen35.gguf.out, tests 46-48) keep "e" + U+0301 apart from the composed U+00E9 (llama-tokenize on that
    vocabulary: [68, 53839] vs [963]). With NFC, 3 of those 50 id lists differed."""
    p = FXT.build("qwen35-f32", str(tmp_path))
    from mashbox.runtime.chat import Chat
    c = Chat(GT.model_dir(File(p), p, root=str(tmp_path / "tok")))
    assert json.loads(GT.files_dict(File(p), p)["tokenizer.json"])["normalizer"] is None
    assert c.encode("e\u0301") != c.encode("\u00e9")
    assert c.decode(c.encode("re\u0301sume\u0301 a\u0300")) == "re\u0301sume\u0301 a\u0300"


def test_run_prompt_end_to_end_on_cpu(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = FXT.build("qwen35-f32", str(tmp_path))
    rc = cli.main(["run", p, "--prompt", "the hat", "--max-tokens", "6", "--ctx", "64", "--slots", "2", "--steps", "2",
                   "--cache-dir", str(tmp_path / "xla")])
    cap = capsys.readouterr()
    assert rc in (None, 0), cap.err
    events = [json.loads(l) for l in cap.err.splitlines() if l.startswith("{")]
    kinds = [e["event"] for e in events]
    assert "gguf_plan" in kinds and "loaded" in kinds and "done" in kinds, cap.err
    assert next(e for e in events if e["event"] == "done")["new_tokens"] >= 1


def _needs(p, ctx=64):
    from mashbox import gguf as G
    f, _, c = G.open_model(p)
    return G, f, c, {s: max(G.slot_need(f, c, ctx, s)[0], G.HEADROOM * G.slot_need(f, c, ctx, s)[1]) for s in (8, 4, 2, 1)}


def test_fit_slots_keeps_the_most_slots_whose_estimate_is_inside_the_headroom(tmp_path, monkeypatch):
    G, f, c, need = _needs(FXT.build("qwen35-f32", str(tmp_path)))
    assert need[8] > need[4] > need[2] > need[1], need
    h = G.HEADROOM
    assert G.fit_slots(f, c, 64, 8, lim=need[8] / h) == 8                          # 8 fit: unchanged
    assert G.fit_slots(f, c, 64, 8, lim=(need[4] + need[8]) / 2 / h) == 4
    assert G.fit_slots(f, c, 64, 8, lim=(need[2] + need[4]) / 2 / h) == 2
    assert G.fit_slots(f, c, 64, 8, lim=need[1] / h) == 1
    assert G.fit_slots(f, c, 64, 8, lim=max(G.slot_need(f, c, 64, 1))) == 1          # only inside the limit: the best chance
    assert G.fit_slots(f, c, 64, 8, lim=need[1] - 1) == 8                          # nothing fits: check_fits refuses
    monkeypatch.setattr(G, "_limit", lambda: None)
    assert G.fit_slots(f, c, 64, 8) == 8                                           # limit unknown (CPU): unchanged


def test_run_with_the_default_slots_lowers_them_to_fit_the_chip(tmp_path, capsys, monkeypatch):
    """zero flags never load a file whose estimate at 8 slots is over the headroom (Qwen3.8-27B Q8_0 on one v6e chip
    warned at 32.8 of 33.5 GB, then ran out of memory at load); an explicit --slots is kept"""
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = FXT.build("qwen35-f32", str(tmp_path))
    G, f, c, need = _needs(p)
    monkeypatch.setattr(G, "_limit", lambda: (need[2] + need[4]) / 2 / G.HEADROOM)
    for flags, want in (([], 2), (["--slots", "1"], None)):
        rc = cli.main(["run", p, "--prompt", "the hat", "--max-tokens", "4", "--ctx", "64", "--steps", "2",
                       "--cache-dir", str(tmp_path / "xla"), *flags])
        cap = capsys.readouterr()
        assert rc in (None, 0), cap.err
        events = [json.loads(l) for l in cap.err.splitlines() if l.startswith("{")]
        got = [e for e in events if e["event"] == "gguf_slots"]
        if want is None:
            assert not got, got
        else:
            assert len(got) == 1 and got[0]["slots"] == want and got[0]["asked"] == 8, got
            assert "--slots" in cap.err and "fit" in cap.err
        assert next(e for e in events if e["event"] == "done")["new_tokens"] >= 1


def test_run_and_download_help_names_kernels_only_for_a_catalog_model():
    """This version installs no kernel package for GGUF block types (models.json engine_kernels is empty), so a GGUF file
    runs on the exact reference path: the run/download help shared by both must not promise a GGUF file a kernel bundle
    or a speed change. Every option help that names kernels or a speed says it is about a catalog model."""
    sub = next(x for x in cli.build_parser()._actions if isinstance(x, argparse._SubParsersAction))
    for name in ("run", "download"):
        named = 0
        for act in sub.choices[name]._actions:
            h = " ".join((act.help or "").split())
            if "kernel" in h or "slower" in h or "faster" in h:
                named += 1
                assert "catalog model" in h and "bundle" not in h, (name, act.option_strings, h)
        assert named, name
