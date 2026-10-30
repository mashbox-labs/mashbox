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
