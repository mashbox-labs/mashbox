"""GGUF files that are not llama.cpp model files stop with one clear line, never a crash or a wrong pointer.

GGUF is also the weights format of other tools: stable-diffusion.cpp video files and ComfyUI-GGUF files hold weights
under the original model's tensor names and carry none of the keys llama.cpp needs to load a model. llama.cpp a37cc46
refuses them ("unknown model architecture: ''" with no general.architecture, "key not found in model" with no
<arch>.block_count or <arch>.embedding_length), so `mashbox run` must not say llama.cpp runs them, and must not stop
with a TypeError while reading hyperparameters that are not there. The three files below have the key layout of real
files on the Hub: no metadata at all (unsloth's MiniMax-H3 video file), general.architecture plus comfy.gguf.* keys
(a ComfyUI-GGUF Qwen3-VL text encoder), general.architecture only (a ComfyUI-GGUF Qwen3.5 text encoder).
"""
import json

import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips

from mashbox import cli  # noqa: E402
from mashbox import gguf as G  # noqa: E402


def _write(path, arch, keys, tensors):
    w = gguf.GGUFWriter(str(path), arch or "llama")
    if arch is None:
        w.kv_data[0].pop("general.architecture")                  # a file with no metadata at all
    for k, v in keys.items():
        if isinstance(v, list):
            w.add_array(k, v)
        else:
            w.add_uint32(k, v)
    for name, shape in tensors:
        w.add_tensor(name, np.zeros(shape, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


HF_NAMES = [("model.embed_tokens.weight", (16, 8)), ("model.layers.0.mlp.down_proj.weight", (8, 8))]
FILES = {
    "no-metadata": (None, {}, HF_NAMES, ["general.architecture"]),
    "comfy-qwen3vl": ("qwen3vl", {"comfy.gguf.orig_shape.visual.blocks.0.attn.qkv.weight": [24, 8]}, HF_NAMES,
                      ["qwen3vl.block_count", "qwen3vl.embedding_length"]),
    "arch-only-qwen35": ("qwen35", {}, [("token_embd.weight", (16, 8)), ("blk.0.attn_norm.weight", (8,))],
                         ["qwen35.block_count", "qwen35.embedding_length"]),
}


@pytest.mark.parametrize("kind", sorted(FILES))
def test_not_a_llama_cpp_model_file_is_one_clear_line(kind, tmp_path, capsys):
    arch, keys, tensors, missing = FILES[kind]
    p = _write(tmp_path / f"{kind}.gguf", arch, keys, tensors)
    with pytest.raises(G.GGUFError) as e:
        G.open_model(p)
    line = str(e.value)
    assert line.startswith(f"mashbox: {kind}.gguf is not a llama.cpp model file"), line
    assert all(k in line for k in missing), (missing, line)
    assert "llama.cpp does not run it either" in line, line
    for wrong in ("llama.cpp runs it", "It runs in llama.cpp", "llama-server", "does not run on TPU"):
        assert wrong not in line, (wrong, line)
    assert cli.main(["run", p]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [line], err
    assert cli.main(["inspect", p]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runs_here"] is False and out["why"] == line, out


def test_a_model_file_of_an_architecture_we_do_not_run_keeps_the_llama_cpp_line(tmp_path):
    """the check names only files llama.cpp cannot load: a llama.cpp model file of another architecture still gets the
    unsupported-architecture line with the llama.cpp command"""
    p = _write(tmp_path / "t5.gguf", "t5", {"t5.block_count": 1, "t5.embedding_length": 8}, [("token_embd.weight", (16, 8))])
    with pytest.raises(G.GGUFError) as e:
        G.open_model(p)
    assert "does not run on TPU" in str(e.value) and f"llama-server -m {p}" in str(e.value), str(e.value)


def _scan():
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location("gguf_scan", os.path.join(root, "tools", "gguf_scan.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("kind", sorted(FILES))
def test_the_hub_scan_probe_gives_the_same_line_and_files_it_as_not_a_model(kind, tmp_path):
    """tools/gguf_scan.py probes a header on a sparse copy and lists what blocks each file: its probe is the engine's
    own answer (open_model's line), and a file that is not a model is filed as not-a-model, not as a missing arch"""
    S = _scan()
    arch, keys, tensors, _ = FILES[kind]
    p = _write(tmp_path / f"{kind}.gguf", arch, keys, tensors)
    with pytest.raises(G.GGUFError) as e:
        G.open_model(p)
    off = int(gguf.GGUFReader(p, "r").data_offset)
    with open(p, "rb") as fh:
        hdr = fh.read(off)
    pr = S.probe(hdr, len(open(p, "rb").read()), p)
    assert pr == dict(supported=False, stage="load", why=str(e.value)), pr
    assert S.blocked_by(S.parse(hdr)[0], pr) == "not-a-model"


def _without_tokenizer(src, dst, model=None):
    """a copy of the GGUF file src with every tokenizer.* key left out (as maya1's GGUF: 15 keys, none of them
    tokenizer.*), or with tokenizer.ggml.model set to `model` and no vocabulary"""
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, r.fields["general.architecture"].contents())
    for name, fd in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture" or name.startswith("tokenizer."):
            continue
        if fd.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(name, fd.contents(), fd.types[0], sub_type=fd.types[-1])
        else:
            w.add_key_value(name, fd.contents(), fd.types[0])
    if model is not None:
        w.add_tokenizer_model(model)
    for t in r.tensors:
        w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return dst


@pytest.mark.parametrize("model", [None, "none", "no_vocab"], ids=["no tokenizer keys", "model none", "model no_vocab"])
def test_a_model_file_with_no_tokenizer_is_one_true_line(model, tmp_path, capsys, monkeypatch):
    """A model file with no tokenizer (maya1's GGUF) stopped `run` with 'the tokenizer in this file (model "None", ...) is
    not read by this version. llama.cpp runs it: llama-server ...', and inspect said runs_here. llama.cpp a37cc46 does not
    load such a file either: tokenizer.ggml.model is a key it requires (src/llama-vocab.cpp:2201; "error loading model
    vocabulary: key not found in model: tokenizer.ggml.model"); "none" and "no_vocab" mean a file with no vocabulary,
    which takes no text. One line says the file has no tokenizer and mashbox does not run it, never that llama.cpp runs
    it; `run` stops before the weights load and inspect gives the same line."""
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(root, "tools", "gguf_fixtures", "fixtures.py"))
    FXT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FXT)
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = _without_tokenizer(FXT.build("llama-f32", str(tmp_path)), str(tmp_path / "maya-like.gguf"), model)
    G.open_model(p)                                   # a whole llama model file in every other way
    assert cli.main(["run", p, "--prompt", "Hi", "--ctx", "64", "--slots", "2", "--cache-dir", str(tmp_path / "xla")]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1, err                         # nothing before it: no plan, no kernels, no weights
    line = err[0]
    head = ('mashbox: maya-like.gguf has no tokenizer (no tokenizer.ggml.model key)' if model is None else
            f'mashbox: maya-like.gguf has no tokenizer (tokenizer.ggml.model is "{model}")')
    assert line.startswith(head + ", so mashbox cannot turn text into its tokens and does not run it."), line
    assert ("llama.cpp does not load it either" in line) == (model is None), line
    for wrong in ("llama.cpp runs it", "It runs in llama.cpp", "llama-server", "is not read by this version"):
        assert wrong not in line, (wrong, line)
    assert cli.main(["inspect", p]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runs_here"] is False and out["why"] == line, out
