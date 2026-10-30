"""tools/gguf_scan.py, tools/gguf_conform.py and tools/gguf_rows.py on the committed fixtures (no network, no llama.cpp
build, no chip): the header-only parse equals gguf-py's, the probe is the engine's own answer, the CPU bars are the
locked ones, and a row is only generated from a sealed record."""
import importlib.util
import json
import os
import shutil

import pytest

import gguf
pytest.importorskip("jax")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "gguf", "qwen3-kquant.gguf")


def tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def header(path):
    off = int(gguf.GGUFReader(path, "r").data_offset)
    with open(path, "rb") as fh:
        return fh.read(off)


def test_header_parse_equals_gguf_py():
    S = tool("gguf_scan")
    hdr = header(FIXTURE)
    meta, tensors, start = S.parse(hdr)
    r = gguf.GGUFReader(FIXTURE, "r")
    assert start == int(r.data_offset)
    assert meta["general.architecture"] == r.fields["general.architecture"].contents()
    assert [(t[0], t[1], t[2]) for t in tensors] == [
        (t.name, int(t.tensor_type), tuple(reversed([int(x) for x in t.shape]))) for t in r.tensors]
    with pytest.raises(S.More):
        S.parse(hdr[:len(hdr) // 2])


def test_probe_is_the_engines_answer(tmp_path):
    S = tool("gguf_scan")
    from mashbox.gguf import archs as GA
    hdr = header(FIXTURE)
    ok = S.probe(hdr, os.path.getsize(FIXTURE), FIXTURE)
    assert ok["supported"] is True and ok["stage"] == "ok"
    w = gguf.GGUFWriter(str(tmp_path / "x.gguf"), arch="nosucharch")
    w.add_block_count(1)
    import numpy as np
    w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    p = str(tmp_path / "x.gguf")
    no = S.probe(header(p), os.path.getsize(p), p)
    assert no == dict(supported=False, stage="load", why=GA.unsupported("nosucharch", p))


def test_bars_are_the_locked_ones():
    C = tool("gguf_conform")
    dense = importlib.util.spec_from_file_location("tgd", os.path.join(ROOT, "tests", "cpu", "test_gguf_dense.py"))
    src = open(dense.origin).read()
    assert "NMSE_MAX = 1e-4" in src and "KL_MAX, TOP1_MIN = 1e-3, 15" in src
    assert C.BARS["identical_weights_nmse_max"] == 1e-4
    assert C.BARS["top1_min_share"] == 15 / 16
    assert C.BARS["near_tie_gap"] == 0.5 and C.BARS["other_divergences_max"] == 0
    assert C.BARS["locked"].startswith("2026-10-29T22:04Z")
    assert "not gated" in C.BARS["block_path_kl"]


def test_rows_only_from_a_sealed_record(tmp_path):
    R = tool("gguf_rows")
    C = tool("gguf_conform")
    d = tmp_path / "rec" / "someone__X.gguf"
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps({"verdict": "cpu-checked"}))
    (d / "header.bin").write_bytes(header(FIXTURE))
    C.seal(str(d))
    assert R.sums_ok(str(d))
    (d / "record.json").write_text(json.dumps({"verdict": "cpu-checked", "edited": True}))
    assert not R.sums_ok(str(d))
    shutil.copy(FIXTURE, d / "extra.gguf")
    C.seal(str(d))
    assert R.sums_ok(str(d))
    (d / "unlisted.txt").write_text("x")
    assert not R.sums_ok(str(d))
