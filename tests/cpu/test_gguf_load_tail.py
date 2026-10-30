"""A GGUF file loads through the same deployment tail as a models.json model (PLAN B-a): deployment id and component set
(/health, the startup log), the prefix cache namespace, and MASHBOX_TRACE. The file's sha256 stands for its weights,
tokenizer and chat template, so a one-byte change to the file is another deployment. CPU, tiny fixture, no network."""
import hashlib
import importlib.util
import json
import os
import shutil

import pytest

pytest.importorskip("jax")

from mashbox import cli  # noqa: E402
from mashbox.gguf import pick as GR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)


def _load(path, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    a = cli.build_parser().parse_args(["run", path, "--prompt", "hi", "--ctx", "64", "--slots", "2", "--steps", "2",
                                       "--cache-dir", str(tmp_path / "xla")])
    a.gguf = GR.resolve(path, log=cli.log)
    m, eng, chat, info, _ = cli.load_gguf(a)
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    return eng, events


def test_gguf_load_sets_the_deployment_namespace_and_trace(tmp_path, monkeypatch, capsys):
    p = FXT.build("qwen35-f32", str(tmp_path))
    monkeypatch.setenv("MASHBOX_TRACE", "1")
    eng, ev = _load(p, tmp_path, monkeypatch, capsys)
    assert eng.deployment_id and eng.component_set_sha, "no deployment id or component set for a GGUF file"
    assert eng.trace is True
    dep = [e for e in ev if e["event"] == "deployment"]
    assert dep == [dict(event="deployment", deployment_id=eng.deployment_id, component_set_sha=eng.component_set_sha)]
    ident = next(e for e in ev if e["event"] == "gguf_identity")
    assert ident["sha256"] == hashlib.sha256(open(p, "rb").read()).hexdigest() and ident["how"] == ["read"]
    assert getattr(eng, "cache_ns", None) or getattr(eng, "namespace", None), "no prefix cache namespace"


def test_one_byte_in_the_file_is_another_deployment(tmp_path, monkeypatch, capsys):
    (tmp_path / "a").mkdir()
    p = FXT.build("qwen35-f32", str(tmp_path / "a"))
    (tmp_path / "b").mkdir()
    q = str(tmp_path / "b" / os.path.basename(p))   # the same name, so only the file's bytes differ (a local file's id is its name)
    shutil.copyfile(p, q)
    with open(q, "r+b") as f:                       # the last byte is tensor data (f32 weights): still a valid file
        f.seek(-1, os.SEEK_END)
        b = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([b[0] ^ 1]))
    e1, _ = _load(p, tmp_path, monkeypatch, capsys)
    monkeypatch.delenv("MASHBOX_TRACE", raising=False)
    e2, _ = _load(q, tmp_path, monkeypatch, capsys)
    assert e1.deployment_id and e2.deployment_id and e1.deployment_id != e2.deployment_id
    ns = lambda e: getattr(e, "cache_ns", None) or getattr(e, "namespace", None)  # noqa: E731
    assert ns(e1) != ns(e2)


def test_hub_cache_blob_name_is_its_sha256(tmp_path):
    """a file in the Hugging Face cache is a link to blobs/<sha256 of its bytes>: that name is used, not a re-read"""
    data = b"GGUF tiny"
    sha = hashlib.sha256(data).hexdigest()
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / sha).write_bytes(data)
    (tmp_path / "snap").mkdir()
    os.symlink(tmp_path / "blobs" / sha, tmp_path / "snap" / "m.gguf")
    assert cli.file_identity(str(tmp_path / "snap" / "m.gguf")) == (sha, "hub_blob_name")
    (tmp_path / "plain.gguf").write_bytes(data)
    assert cli.file_identity(str(tmp_path / "plain.gguf")) == (sha, "read")
