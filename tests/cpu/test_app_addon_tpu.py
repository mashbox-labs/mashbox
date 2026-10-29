"""TPU rows of app add-ons (ui/app/addons.py tpu_rows / tpu_payload) through `mashbox tpu` (model, remote_setup) and the
app (catalog, Start, server rows). A made-up add-on; no cloud, no network."""
import importlib
import io
import os
import re
import tarfile
import types

import pytest

from mashbox import tpu
from mashbox.ui import app as App
from mashbox.ui.app import addons

ROW = {"id": "demo-big-model", "name": "Demo", "cloud": {"accelerator": "v6e-1", "runtime": "v2-alpha-tpuv6e",
       "disk_gb": 10, "port": 8792, "health": "/health", "zones": ["us-east5-b"]}}


def fake(tmp_path, rows=(ROW,), data=b"x" * 1000):
    a = types.SimpleNamespace(ID="demo", LABEL="Demo", TPU=list(rows))
    f = tmp_path / "data.bin"
    f.write_bytes(data)

    def tpu_payload(row_id, tmp):
        out = os.path.join(tmp, "code.tar")
        with tarfile.open(out, "w") as t:
            ti = tarfile.TarInfo("start.py"); b = b"print(1)\n"; ti.size = len(b); t.addfile(ti, io.BytesIO(b))
        return {"code": out, "data": [str(f)], "args": ["--x"]}
    a.tpu_payload = tpu_payload
    return a


@pytest.fixture
def demo(tmp_path, monkeypatch):
    a = fake(tmp_path)
    monkeypatch.setattr(addons, "_loaded", [a])
    monkeypatch.setattr(addons, "_problems", [])
    monkeypatch.setenv("MASHBOX_APP_ADDONS", str(tmp_path / "x.whl"))
    return a


def test_rows_checked(tmp_path, monkeypatch):
    bad = [dict(ROW, id="big-model"), dict(ROW, run="mashbox run x"), dict(ROW, cloud=dict(ROW["cloud"], port=80)),
           dict(ROW, cloud=dict(ROW["cloud"], zones=["nowhere"])), dict(ROW, cloud=dict(ROW["cloud"], agents={})),
           dict(ROW, cloud=dict(ROW["cloud"], health="health")), ROW]
    monkeypatch.setattr(addons, "_loaded", [fake(tmp_path, rows=bad)])
    got = addons.tpu_rows()
    assert [r["id"] for r in got] == ["demo-big-model"]
    assert got[0]["kind"] == "addon" and got[0]["module"] == "demo"


def test_cli_finds_row_only_in_app_runtime(demo, monkeypatch):
    m = tpu.model("demo-big-model")
    assert m["addon"] == "demo" and m["cloud"]["port"] == 8792
    monkeypatch.delenv("MASHBOX_APP_ADDONS")
    with pytest.raises(tpu.Stop):
        tpu.model("demo-big-model")


def test_catalog_row_never_marked_addon(demo, monkeypatch):
    reg = tpu.registry()
    reg["models"][0]["addon"] = "demo"
    monkeypatch.setattr(tpu, "registry", lambda: reg)
    assert "addon" not in tpu.model(reg["models"][0]["id"])


def test_problem_addon_stops_cli(demo, monkeypatch):
    monkeypatch.setattr(addons, "_problems", ["x.whl: broken"])
    with pytest.raises(tpu.Stop, match="did not load"):
        tpu.model("demo-big-model")


def run_setup(monkeypatch, m, have=""):
    calls = []

    def gc(args, project=None, timeout=None):
        calls.append(args)
        cmd = args[args.index("--command") + 1] if "--command" in args else ""
        out = have if "MBX_HAVE" in cmd and "MBX_SERVER" not in cmd else "MBX_SERVER=started\nMBX_DISK free_gb=90 have_gb=90\n"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(tpu, "gc", gc)
    monkeypatch.setattr(tpu, "still_ours", lambda *a, **k: None)
    monkeypatch.delenv("MASHBOX_WHEEL", raising=False)
    res = tpu.remote_setup(m, "p", "us-east5-b", "c1")
    return calls, res


def test_remote_setup_sends_payload_and_launches_it(demo, monkeypatch):
    m = tpu.model("demo-big-model")
    calls, res = run_setup(monkeypatch, m)
    scp = next(c for c in calls if c[3] == "scp")
    names = [os.path.basename(x) for x in scp[4:-4]]
    assert "mashbox-addon-code.tar" in names and any(re.fullmatch(r"mashbox-data-[0-9a-f]{64}", n) for n in names)
    script = calls[-1][calls[-1].index("--command") + 1]
    assert "LAUNCH" not in script and "start.py" in script and "mashbox download" not in script
    assert re.search(r"MBX_ADDON_SHA=code", script) and re.search(r'mv -T -f ~/mashbox-data-[0-9a-f]{64} "\$f"', script)
    assert "--x" in script and res["server"] == "started"


def test_data_already_on_chip_is_not_sent(demo, monkeypatch, tmp_path):
    h = tpu.sha256_file(str(tmp_path / "data.bin"))
    calls, _ = run_setup(monkeypatch, tpu.model("demo-big-model"), have=f"MBX_HAVE={h}\n")
    scp = next(c for c in calls if c[3] == "scp")
    assert not any("mashbox-data-" in x for x in scp)
    script = calls[-1][calls[-1].index("--command") + 1]
    assert f'[ "$(sha256sum "$f" | cut -c1-64)" = {h} ]' in script        # checked again at the start, by its bytes


def test_bad_payload_stops(demo, monkeypatch, tmp_path):
    demo.tpu_payload = lambda row_id, tmp: {"code": str(tmp_path / "data.bin"), "data": []}
    with pytest.raises(tpu.Stop, match="no usable files"):
        run_setup(monkeypatch, tpu.model("demo-big-model"))
    link = tmp_path / "link.bin"
    link.symlink_to(tmp_path / "data.bin")

    def p(row_id, tmp):
        out = os.path.join(tmp, "c.tar"); tarfile.open(out, "w").close()
        return {"code": out, "data": [str(link)]}
    demo.tpu_payload = p
    with pytest.raises(tpu.Stop, match="not a link"):
        run_setup(monkeypatch, tpu.model("demo-big-model"))


def test_catalog_row_launch_unchanged(demo, monkeypatch):
    m = tpu.model("qwen3-8-27b")
    calls, _ = run_setup(monkeypatch, m)
    script = calls[-1][calls[-1].index("--command") + 1]
    assert "mashbox-addon-code.tar" not in script and "start.py" not in script


def test_app_lists_starts_and_files_addon_row(demo, monkeypatch, tmp_path):
    monkeypatch.setattr(App, "registry", lambda refresh=False: (tpu.registry(), {"from": "package"}))
    assert App.tpu_row("demo-big-model")["module"] == "demo"
    assert App.health_path("demo-big-model") == "/health"
    st = {"servers": [], "active": None}
    monkeypatch.setattr(App, "state", lambda: st)
    monkeypatch.setattr(App, "_save", lambda *a: None)
    App.add_tpu_server("demo-big-model", "c1", "http://127.0.0.1:8792")
    assert st["servers"][0]["module"] == "demo" and st["active"] is None
    App.add_tpu_server("qwen3-8-27b", "c2", "http://127.0.0.1:8000")
    assert st["servers"][1]["module"] == "chat" and st["active"] == "tpu-c2"


def test_payload_mismatch_on_chip_stops(demo, monkeypatch):
    def gc(args, project=None, timeout=None):
        bad = "--command" in args and "MBX_ADDON_SHA" in args[args.index("--command") + 1]
        import types as T
        return T.SimpleNamespace(returncode=5 if bad else 0, stdout="MBX_ADDON_SHA=code\n" if bad else "", stderr="")
    monkeypatch.setattr(tpu, "gc", gc)
    monkeypatch.setattr(tpu, "still_ours", lambda *a, **k: None)
    with pytest.raises(tpu.Stop, match="not the ones sent from here"):
        tpu.remote_setup(tpu.model("demo-big-model"), "p", "us-east5-b", "c1")


def test_catalog_lists_addon_rows(demo, monkeypatch):
    monkeypatch.setattr(App, "registry", lambda refresh=False: (tpu.registry(), {"from": "package"}))
    monkeypatch.setattr(App, "active_server", lambda *a: None)
    rows = {r["id"]: r for r in App.catalog()["models"]}
    assert rows["demo-big-model"]["module"] == "demo" and rows["demo-big-model"]["tpu"] is True


def test_probe_reports_only_hashed_files(demo, monkeypatch, tmp_path):
    """S25: the chip's answer names a file as held only after hashing it; a bare name (the old `ls` answer) or a hash
    that was not asked for never skips a send."""
    h = tpu.sha256_file(str(tmp_path / "data.bin"))
    for have in (h + "\n", "MBX_HAVE=" + "0" * 64 + "\n"):
        calls, _ = run_setup(monkeypatch, tpu.model("demo-big-model"), have=have)
        scp = next(c for c in calls if c[3] == "scp")
        assert any(f"mashbox-data-{h}" in x for x in scp)
    probe = next(c[c.index("--command") + 1] for c in calls if "--command" in c and "MBX_HAVE" in c[c.index("--command") + 1])
    assert "sha256sum" in probe and "! -L" in probe


def _sh(script, home):
    import subprocess
    return subprocess.run(["bash", "-c", script], env={"HOME": str(home), "PATH": os.environ["PATH"]},
                          capture_output=True, text=True)


@pytest.mark.parametrize("bad", ["corrupt", "link", "folder"])
def test_chip_scripts_replace_a_bad_cached_entry(demo, monkeypatch, tmp_path, bad):
    """S25 on a real shell: a corrupt file, a link (even to the right bytes) or a folder under the expected name is never
    reported as held; the sent file is checked and put in place in one step (a folder of that name set aside)."""
    import hashlib
    body = b"x" * 1000
    h = hashlib.sha256(body).hexdigest()
    home = tmp_path / "home"
    (home / ".mashbox" / "data").mkdir(parents=True)
    entry = home / ".mashbox" / "data" / h
    if bad == "corrupt":
        entry.write_bytes(b"y" * 1000)
    elif bad == "link":
        (tmp_path / "right").write_bytes(body)
        entry.symlink_to(tmp_path / "right")
    else:
        entry.mkdir()
    calls, _ = run_setup(monkeypatch, tpu.model("demo-big-model"), have="")
    probe = next(c[c.index("--command") + 1] for c in calls if "--command" in c and "MBX_HAVE" in c[c.index("--command") + 1])
    r = _sh(probe, home)
    assert r.returncode == 0 and "MBX_HAVE" not in r.stdout
    script = calls[-1][calls[-1].index("--command") + 1]
    lines = [ln for ln in script.splitlines() if "mashbox-data-" in ln or ln.startswith("if [ -d") or ln.startswith("mkdir -p ~/.mashbox/data")]
    (home / f"mashbox-data-{h}").write_bytes(body)
    r = _sh("set -e\n" + "\n".join(lines), home)
    assert r.returncode == 0, r.stderr
    assert entry.is_file() and not entry.is_symlink() and entry.read_bytes() == body
    assert (bad == "folder") == any(p.name.startswith(h + ".set-aside-") for p in entry.parent.iterdir())
    if bad == "link":
        assert (tmp_path / "right").read_bytes() == body                  # the link itself was replaced, not its target
    r = _sh(probe, home)
    assert f"MBX_HAVE={h}" in r.stdout


def test_held_entry_is_hashed_again_at_start(demo, monkeypatch, tmp_path):
    """S25: between the probe and the start the entry changed: the start's check refuses it (exit 5)."""
    import hashlib
    body = b"x" * 1000
    h = hashlib.sha256(body).hexdigest()
    calls, _ = run_setup(monkeypatch, tpu.model("demo-big-model"), have=f"MBX_HAVE={h}\n")
    script = calls[-1][calls[-1].index("--command") + 1]
    line = next(ln for ln in script.splitlines() if ln.startswith(f"f=~/.mashbox/data/{h}"))
    home = tmp_path / "home"
    (home / ".mashbox" / "data").mkdir(parents=True)
    (home / ".mashbox" / "data" / h).write_bytes(b"y" * 1000)
    r = _sh(line, home)
    assert r.returncode == 5 and "MBX_ADDON_SHA=data" in r.stdout
    (home / ".mashbox" / "data" / h).write_bytes(body)
    assert _sh(line, home).returncode == 0


def test_restart_identity_covers_data_and_arguments(demo, monkeypatch, tmp_path):
    """S26: the same code with other data or other arguments is another payload: the server starts again."""
    def pkg_line(m):
        calls, _ = run_setup(monkeypatch, m)
        script = calls[-1][calls[-1].index("--command") + 1]
        return next(ln for ln in script.splitlines() if ln.startswith("pkg="))
    m = tpu.model("demo-big-model")
    a = pkg_line(m)
    assert pkg_line(m) == a                                             # same payload, same identity
    (tmp_path / "data.bin").write_bytes(b"z" * 1000)
    b = pkg_line(m)
    orig = demo.tpu_payload

    def other_args(row_id, tmp):
        p = orig(row_id, tmp)
        return dict(p, args=["--y"])
    demo.tpu_payload = other_args
    c = pkg_line(m)
    assert len({a, b, c}) == 3
