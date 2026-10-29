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

ROW = dict(name="xla-cache/v6e-jax0.0.0-libtpu0.0.0.tar", kind="xla-cache", chips=["v6e"], jax="0.0.0", jaxlib="0.0.0",
           libtpu="0.0.0")


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
