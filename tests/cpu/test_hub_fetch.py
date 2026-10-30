"""A download that fails is one line. hub.fetch was a bare hf_hub_download, so an offline run, a DNS or connection
failure, or a repo the Hub refuses (401, 403, 404) ended `mashbox run` and `mashbox download` with a huggingface_hub
traceback (LocalEntryNotFoundError, cause OfflineModeIsEnabled) for the Qwen3.8-27B upstream files, weights, compile cache
and kernel bundle, and never said whether running again was safe. Every hub.fetch now raises hub.DownloadError, whose
text is that one line, and `mashbox` prints it and exits 1; paths that go on without a file (the compile cache, a drafter
that is not gated) still do."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from mashbox import hub

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_an_offline_download_is_one_line(tmp_path):
    """`mashbox download qwen3-8-27b` with no network (HF_HUB_OFFLINE=1, empty caches): one line naming the file, its
    repo and revision, why, and that running the same command again is safe; exit 1, no traceback."""
    m = hub.model_entry("qwen3-8-27b")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf"),
           "MASHBOX_CACHE": str(tmp_path / "cache"), "JAX_PLATFORMS": "cpu"}
    p = subprocess.run([sys.executable, "-m", "mashbox", "download", "qwen3-8-27b"], capture_output=True, text=True,
                       timeout=300, cwd=tmp_path, env=env)
    err = p.stderr.strip().splitlines()
    assert p.returncode == 1 and len(err) == 1 and p.stdout == "", p.stderr[-2000:]
    up = m["upstream"]
    assert err[0].startswith(f"mashbox: could not download config.json from https://huggingface.co/{up['hf']} "
                             f"(revision {up['revision'][:8]}): LocalEntryNotFoundError: "), err
    assert err[0].endswith("Check the network and run the same command again (files already downloaded are kept)."), err


def refusals():
    """(the error hf_hub_download raises, what the line must say why) for a refusal and for each network failure the
    installed huggingface_hub can raise"""
    errors = pytest.importorskip("huggingface_hub.errors")
    out = [(OSError(28, "No space left on device"), "OSError: [Errno 28] No space left on device")]
    try:
        import requests                                    # huggingface_hub 0.x: its HTTP errors are requests' (OSErrors)
        r = requests.Response()
        r.status_code = 401
        out += [(errors.RepositoryNotFoundError("401 Client Error. Repository Not Found", response=r), "Hugging Face answered HTTP 401"),
                (requests.exceptions.ConnectionError("Max retries exceeded with url"), "ConnectionError: Max retries exceeded")]
    except ImportError:
        pass
    try:
        import httpx2                                      # huggingface_hub 1.x and later: httpx2, whose errors are not OSErrors
        out += [(httpx2.ConnectError("[Errno -3] Temporary failure in name resolution"), "ConnectError: [Errno -3]"),
                (httpx2.ReadTimeout("The read operation timed out"), "ReadTimeout: The read operation timed out")]
    except ImportError:
        pass
    return out


def test_a_refused_or_broken_download_is_one_line(monkeypatch):
    """Each failure hf_hub_download raises (a refusal with its HTTP code, a connection or read failure, a full disk) is a
    DownloadError with one line; a programming error is not caught."""
    import huggingface_hub
    rev = "0123456789abcdef0123456789abcdef01234567"
    for err, why in refusals():
        def fail(*a, err=err, **k):
            raise err
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
        with pytest.raises(hub.DownloadError) as e:
            hub.fetch("org/Repo", "w.bshj", rev)
        msg = str(e.value)
        assert msg.startswith(f"mashbox: could not download w.bshj from https://huggingface.co/org/Repo (revision 01234567): {why}"), msg
        assert "\n" not in msg and "Traceback" not in msg and "run the same command again" in msg, msg

    def long(*a, **k):                                     # huggingface_hub's offline text is two sentences over 200 characters
        raise OSError("We cannot find the requested files in the local cache. " + "Please check your connection " * 9)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", long)
    with pytest.raises(hub.DownloadError) as e:
        hub.fetch("org/Repo", "w.bshj", rev)
    assert ": OSError: We cannot find the requested files in the local cache. Check the network and run" in str(e.value), str(e.value)

    def bug(*a, **k):
        raise TypeError("hf_hub_download() got an unexpected keyword argument")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", bug)
    with pytest.raises(TypeError):
        hub.fetch("org/Repo", "w.bshj", rev)


@pytest.mark.parametrize("model", ["qwen3-8-27b", "hy-nomas"])
def test_a_kernel_bundle_download_error_names_the_way_out(monkeypatch, model):
    """The kernel bundle's line also says the run can go on without it: --no-kernels (plain JAX, slower)."""
    import huggingface_hub
    from mashbox import kernels as K
    m = hub.model_entry(model)
    row = m["kernels"][0]
    monkeypatch.setattr(K, "chip_family", lambda: row["chips"][0])
    monkeypatch.setattr(K, "versions", lambda: dict(jax=row["jax"], jaxlib=None, libtpu=None))

    def offline(*a, **k):
        raise OSError("outgoing traffic has been disabled")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(hub.DownloadError) as e:
        hub.kernel_bundle(m, print)
    msg = str(e.value)
    assert f"could not download {row['file']} from https://huggingface.co/{m['hf_repo']} " in msg, msg
    assert msg.endswith(", or start with --no-kernels to run without it, in plain JAX (slower)."), msg


def test_paths_that_go_on_without_a_file_still_do(tmp_path, monkeypatch):
    """The compile cache is an optimisation: a DownloadError there is one log line and compiling at first use."""
    import huggingface_hub
    lines = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    from mashbox import kernels as K
    row = dict(name="xla-cache/x.tar", kind="xla-cache", accelerator="v6e-1", jax="0.0.0", jaxlib="0.0.0", libtpu="0.0.0",
               sha256="0" * 64)
    monkeypatch.setattr(K, "accelerator", lambda: "v6e-1")
    monkeypatch.setattr(K, "versions", lambda: dict(jax="0.0.0", jaxlib="0.0.0", libtpu="0.0.0"))
    m = dict(id="t", hf_repo="org/Repo", hf_revision="0" * 40, files=[row])
    assert hub.compile_cache(m, str(tmp_path), lines.append) is None
    ev = [json.loads(x) for x in lines]
    assert len(ev) == 1 and ev[0]["used"] is False and ev[0]["reason"].startswith("DownloadError: mashbox: could not download"), ev
