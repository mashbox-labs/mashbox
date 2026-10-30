"""The shipped XLA compile cache is an optimisation, so a bad one must never stop the server: hub.compile_cache unpacks a
matching tar into the compile-cache folder, and any error on that path (a tar whose sha256 does not match the pinned
models.json row, a failed download) is one log line and compiling at first use. Before this guard, verify's SystemExit on
a sha256 mismatch went past `except Exception` and ended the run. Tiny tars on CPU; no download, no JAX."""
import hashlib
import io
import json
import sys
import tarfile
import types

import pytest

import mashbox
from mashbox import hub

ROW = dict(name="xla-cache/v6e-jax0.0.0-libtpu0.0.0.tar", kind="xla-cache", chips=["v6e"], accelerator="v6e-1", jax="0.0.0",
           jaxlib="0.0.0", libtpu="0.0.0")
V6E1 = dict(ROW, tp=1)
V6E4 = dict(ROW, name="xla-cache/v6e4-tp4-jax0.0.0-libtpu0.0.0.tar", accelerator="v6e-4", tp=4)


def make_tar(path, entries):
    with tarfile.open(path, "w") as t:
        for name, data in entries.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model(sha):
    return {"id": "m", "hf_repo": "org/m", "hf_revision": "0" * 40, "files": [dict(ROW, sha256=sha)]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "home"))
    fake = types.ModuleType("mashbox.kernels")          # this chip and versions match ROW; no JAX needed
    fake.chip_family = lambda: "v6e"
    fake.accelerator = lambda: "v6e-1"
    fake.versions = lambda: {"jax": "0.0.0", "jaxlib": "0.0.0", "libtpu": "0.0.0"}
    monkeypatch.setitem(sys.modules, "mashbox.kernels", fake)
    monkeypatch.setattr(mashbox, "kernels", fake, raising=False)
    tar = tmp_path / "cache.tar"
    good = make_tar(tar, {"prog-a": b"a" * 100, "prog-b": b"b" * 50})
    monkeypatch.setattr(hub, "fetch", lambda repo, filename, revision, local_dir=None: str(tar))
    cache_dir = tmp_path / "xla"
    cache_dir.mkdir()
    lines = []
    return types.SimpleNamespace(tar=tar, good=good, cache_dir=cache_dir, lines=lines, log=lines.append)


def model_rows(*rows):
    return {"id": "m", "hf_repo": "org/m", "hf_revision": "0" * 40, "files": list(rows)}


def on_host(monkeypatch, acc):
    monkeypatch.setattr(sys.modules["mashbox.kernels"], "accelerator", lambda: acc)


def events(lines):
    return [json.loads(x) for x in lines]


def test_matching_tar_is_unpacked(env):
    assert hub.compile_cache(model(env.good), str(env.cache_dir), env.log) == str(env.tar)
    assert sorted(p.name for p in env.cache_dir.iterdir()) == ["prog-a", "prog-b"]
    ev = [e for e in events(env.lines) if e["event"] == "compile_cache"]
    assert len(ev) == 1 and ev[0]["used"] is True and ev[0]["new_entries"] == 2


def test_bad_hash_tar_is_one_line_and_compile_at_first_use(env):
    assert hub.compile_cache(model("f" * 64), str(env.cache_dir), env.log) is None
    assert list(env.cache_dir.iterdir()) == []
    ev = events(env.lines)
    assert len(ev) == 1 and ev[0]["event"] == "compile_cache" and ev[0]["used"] is False, ev
    assert "does not match the pinned" in ev[0]["reason"], ev


def test_download_error_is_one_line(env, monkeypatch):
    def fail(*a, **k):
        raise OSError("HTTP 404")
    monkeypatch.setattr(hub, "fetch", fail)
    assert hub.compile_cache(model(env.good), str(env.cache_dir), env.log) is None
    ev = events(env.lines)
    assert len(ev) == 1 and ev[0]["used"] is False and "HTTP 404" in ev[0]["reason"], ev


def test_weights_sha256_mismatch_still_stops(tmp_path, monkeypatch):
    # only the cache path forgives a bad file: weights that fail their sha256 still stop the run
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "home"))
    f = tmp_path / "w.bshj"
    f.write_bytes(b"x")
    with pytest.raises(SystemExit, match="does not match the pinned"):
        hub.verify(str(f), "0" * 64)


# The compile cache holds programs for one deployment shape. Keyed by chip family alone, a v6e-4 tensor-parallel start
# unpacked the one-chip v6e cache and still compiled every program (its programs are other programs). The row names its
# accelerator (chip family + chips: v6e-1, v6e-4) and the tensor-parallel size it was built for (1 when absent).
def test_a_v6e4_host_never_unpacks_a_v6e1_cache(env, monkeypatch):
    on_host(monkeypatch, "v6e-4")
    assert hub.compile_cache(model_rows(dict(V6E1, sha256=env.good)), str(env.cache_dir), env.log) is None
    assert list(env.cache_dir.iterdir()) == []
    ev = events(env.lines)
    assert len(ev) == 1 and ev[0]["event"] == "compile_cache" and ev[0]["used"] is False and "v6e-4" in ev[0]["reason"], ev


@pytest.mark.parametrize("row_tp,tp", [(1, 4), (4, 1), (4, 2)])
def test_a_cache_built_at_another_tensor_parallel_size_is_not_unpacked(env, monkeypatch, row_tp, tp):
    on_host(monkeypatch, "v6e-4")
    m = model_rows(dict(V6E4, tp=row_tp, sha256=env.good))
    assert hub.compile_cache(m, str(env.cache_dir), env.log, tp=tp) is None
    assert list(env.cache_dir.iterdir()) == []


def test_the_row_of_this_deployment_is_the_one_unpacked(env, monkeypatch, tmp_path):
    tar4 = tmp_path / "cache4.tar"
    good4 = make_tar(tar4, {"prog-4": b"4" * 10})
    paths = {V6E1["name"]: str(env.tar), V6E4["name"]: str(tar4)}
    monkeypatch.setattr(hub, "fetch", lambda repo, filename, revision, local_dir=None: paths[filename])
    m = model_rows(dict(V6E1, sha256=env.good), dict(V6E4, sha256=good4))     # the one-chip row comes first
    on_host(monkeypatch, "v6e-4")
    assert hub.compile_cache(m, str(env.cache_dir), env.log, tp=4) == str(tar4)
    assert sorted(p.name for p in env.cache_dir.iterdir()) == ["prog-4"]
    one = tmp_path / "xla1"
    one.mkdir()
    on_host(monkeypatch, "v6e-1")
    assert hub.compile_cache(m, str(one), env.log, tp=1) == str(env.tar)
    assert sorted(p.name for p in one.iterdir()) == ["prog-a", "prog-b"]


def test_a_row_that_names_no_accelerator_is_never_unpacked(env):
    family_only = {k: v for k, v in ROW.items() if k != "accelerator"}
    assert hub.compile_cache(model_rows(dict(family_only, sha256=env.good)), str(env.cache_dir), env.log) is None
    assert list(env.cache_dir.iterdir()) == []


def test_every_shipped_cache_row_names_its_deployment():
    from pathlib import Path
    import re
    data = json.loads((Path(hub.__file__).parent / "models.json").read_text())
    entries = data.get("models", data) if isinstance(data, dict) else data
    entries = entries.values() if isinstance(entries, dict) else entries
    rows = {e.get("id"): f for e in entries if isinstance(e, dict) for f in e.get("files", []) if f.get("kind") == "xla-cache"}
    assert rows, "no compile cache row in models.json"
    for mid, f in rows.items():
        assert isinstance(f.get("accelerator"), str) and re.fullmatch(r"[a-z0-9]+-[0-9]+", f["accelerator"]), (mid, f["name"])
        assert isinstance(f.get("tp", 1), int) and f.get("tp", 1) >= 1, (mid, f["name"])
    assert (rows["qwen3-8-27b"]["accelerator"], rows["qwen3-8-27b"].get("tp", 1)) == ("v6e-1", 1)   # built on one v6e chip
    assert rows["hy-nomas"]["accelerator"] == "v6e-4"                                               # built on a v6e-4
