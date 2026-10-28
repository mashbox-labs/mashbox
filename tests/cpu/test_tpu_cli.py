"""`mashbox tpu` against a fake gcloud (tests/cpu/fake_gcloud.py), and install.sh checks. No cloud, no network."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import resources

import re

import pytest

from mashbox import tpu

HERE = os.path.dirname(os.path.abspath(__file__))
Z = ["us-east5-b", "europe-west4-a", "us-east5-a", "asia-northeast1-b", "us-south1-a", "us-east1-d", "us-central1-b"]


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """Returns setup(**scenario) -> state reader. Fake gcloud on MASHBOX_GCLOUD, state file in tmp_path."""
    st = tmp_path / "gcloud.json"
    g = tmp_path / "gcloud"
    g.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE}/fake_gcloud.py" "$@"\n')
    g.chmod(0o755)
    monkeypatch.setenv("MASHBOX_GCLOUD", str(g))
    monkeypatch.setenv("FAKE_GCLOUD_STATE", str(st))
    monkeypatch.delenv("MASHBOX_WHEEL", raising=False)
    monkeypatch.delenv("MASHBOX_HF_TOKEN_FILE", raising=False)
    monkeypatch.setattr(tpu, "STATE", str(tmp_path / "tpu.json"))
    monkeypatch.setattr(tpu, "POLL", 0.01)
    held = []

    def hold(tun, *a):
        held.append(a)
        tun.terminate()
        tun.wait()
        return 0
    monkeypatch.setattr(tpu, "hold", hold)

    def setup(**kw):
        s = dict(account="user@example.com", project="p1", api=True, plain={}, grant={})
        s.update(kw)
        st.write_text(json.dumps(s))
        return lambda: json.loads(st.read_text())
    setup.held = held
    return setup


def creates(S, verb="tpu-vm"):
    return [c[c.index("--zone") + 1] for c in S["calls"] if c[2:4] == [verb, "create"]]


def up(*extra):
    return tpu.main(["up", "qwen3-8-27b", "--yes", *extra])


def test_zone_exhaustion_moves_to_next_zone(cloud, capsys):
    read = cloud(plain={Z[0]: "capacity", Z[1]: "capacity", Z[2]: "ok"})
    assert up() == 0
    S = read()
    assert creates(S) == Z[:3]
    assert f"{Z[2]}/mashbox-qwen3-8-27b" in S["nodes"]
    out = capsys.readouterr().out
    assert out.count(": no free chip (") == 2 and "$2.70 per hour" in out and "ESTIMATE" in out
    assert re.search(r"Local URL +http://127\.0\.0\.1:80\d\d/v1 ", out) and "mashbox tpu down mashbox-qwen3-8-27b" in out


def test_quota_error_prints_fix_and_continues(cloud, capsys):
    read = cloud(plain={Z[0]: "no-access", Z[1]: "quota", Z[2]: "ok"})
    assert up() == 0
    assert creates(read()) == Z[:3]
    out = capsys.readouterr().out
    assert "iam-admin/quotas?project=p1" in out and "no quota" in out


def test_created_then_gone_counts_as_no_capacity(cloud, capsys):
    read = cloud(plain={Z[0]: "vanish", Z[1]: "ok"})
    assert up() == 0
    assert f"{Z[1]}/mashbox-qwen3-8-27b" in read()["nodes"]


def test_queued_fallback_keeps_first_grant_deletes_others(cloud, capsys):
    read = cloud(plain={}, grant={Z[1]: 2})                      # every plain create: no capacity
    assert up() == 0
    S = read()
    assert creates(S) == Z[:3] and creates(S, "queued-resources") == Z[:3]
    assert list(S["qrs"]) == [f"{Z[1]}/mashbox-qwen3-8-27b-q"]    # the other two deleted
    n = S["nodes"][f"{Z[1]}/mashbox-qwen3-8-27b"]
    assert n["metadata"]["mashbox-qr"] == "mashbox-qwen3-8-27b-q" and n["metadata"]["mashbox-max-seconds"] == "28800"
    q = [c for c in S["calls"] if c[2:4] == ["queued-resources", "create"]][0]
    assert "--spot" not in q and q[q.index("--valid-until-duration") + 1] == "60m"
    assert "granted after" in capsys.readouterr().out


def test_queue_skips_zones_without_quota(cloud):
    read = cloud(plain={Z[0]: "quota", Z[1]: "no-access"}, grant={Z[2]: 1})
    assert up() == 0
    assert creates(read(), "queued-resources") == [Z[2], Z[3], Z[4]]


def test_queue_timeout_deletes_every_request(cloud, capsys):
    read = cloud(plain={})
    assert up("--queue-minutes", "0.002") == 2
    S = read()
    assert S["qrs"] == {} and S["nodes"] == {}
    assert "no chip after" in capsys.readouterr().err


def test_ctrl_c_while_queued_deletes_every_request(cloud, monkeypatch, capsys):
    read = cloud(plain={})
    n = {"i": 0}

    def sleep(s):
        n["i"] += 1
        if n["i"] > 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(tpu.time, "sleep", sleep)
    assert up() == 130
    assert read()["qrs"] == {}
    assert "Deleting the queued requests" in capsys.readouterr().out


def test_up_again_reuses_chip_and_server(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    assert up() == 0
    S = read()
    assert creates(S) == [Z[0]]                                    # one create only
    assert S["starts"] == 2 and "already exists" in capsys.readouterr().out
    install = [c for c in S["calls"] if c[2:4] == ["tpu-vm", "ssh"] and "--command" in c and "mashbox-install.sh" in c[-3]]
    assert len(install) == 2                                        # install.sh re-run (idempotent), server not restarted


def test_down_with_nothing_up(cloud, capsys):
    cloud()
    assert tpu.main(["down"]) == 0
    assert "Nothing to take down" in capsys.readouterr().out
    assert tpu.main(["status"]) == 0
    assert "No mashbox chips" in capsys.readouterr().out


def test_status_then_down(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    capsys.readouterr()
    assert tpu.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "mashbox-qwen3-8-27b" in out and "$2.70" in out and "UTC" in out
    assert tpu.main(["down"]) == 0
    assert read()["nodes"] == {} and "gone" in capsys.readouterr().out
    assert json.load(open(tpu.STATE)) == {}


def test_chip_gone_elsewhere_is_forgotten(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    S = read()
    S["nodes"] = {}                                                  # deleted by its max lifetime, not by us
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    assert tpu.main(["status"]) == 0 and "No mashbox chips" in capsys.readouterr().out
    assert json.load(open(tpu.STATE)) == {}


def test_down_queued_chip_deletes_request_and_node(cloud):
    read = cloud(plain={}, grant={Z[0]: 1})
    assert up() == 0
    assert tpu.main(["down", "mashbox-qwen3-8-27b"]) == 0
    S = read()
    assert S["nodes"] == {} and S["qrs"] == {}


def test_preflight_messages(cloud, capsys):
    cloud(account="")
    assert up() == 2 and "gcloud auth login" in capsys.readouterr().err
    cloud(project="")
    assert up() == 2 and "gcloud config set project" in capsys.readouterr().err
    cloud(api=False)
    assert up() == 2 and "gcloud services enable tpu.googleapis.com --project p1" in capsys.readouterr().err


def test_no_gcloud(monkeypatch, capsys):
    monkeypatch.setenv("MASHBOX_GCLOUD", "")
    monkeypatch.setenv("PATH", "/nonexistent")
    assert tpu.main(["status"]) == 2 and "cloud.google.com/sdk" in capsys.readouterr().err


def test_self_delete_not_allowed_is_said(cloud, capsys):
    cloud(plain={Z[0]: "ok"}, probe="403")
    assert up() == 0
    out = capsys.readouterr().out
    assert "NOT ACTIVE" in out and "add-iam-policy-binding" in out and "roles/tpu.admin" in out


def test_max_hours_zero_and_spot(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up("--max-hours", "0", "--spot") == 0
    n = read()["nodes"][f"{Z[0]}/mashbox-qwen3-8-27b"]
    assert n["metadata"]["mashbox-max-seconds"] == "0" and n["schedulingConfig"]["spot"]
    assert "off (--max-hours 0)" in capsys.readouterr().out


def test_needs_yes_without_terminal(cloud, monkeypatch, capsys):
    read = cloud(plain={Z[0]: "ok"})
    monkeypatch.setattr(sys, "stdin", open(os.devnull))
    assert tpu.main(["up", "qwen3-8-27b"]) == 2
    assert "--yes" in capsys.readouterr().err and creates(read()) == []


def test_hy_nomas_row_and_staging_wheel(cloud, tmp_path, monkeypatch, capsys):
    w = tmp_path / "mashbox-0.1.0-py3-none-any.whl"
    w.write_bytes(b"x")
    monkeypatch.setenv("MASHBOX_WHEEL", str(w))
    read = cloud(plain={"asia-northeast1-b": "ok"})
    assert tpu.main(["up", "hy-nomas", "--yes"]) == 0
    S = read()
    c = [c for c in S["calls"] if c[2:4] == ["tpu-vm", "create"]][0]
    assert c[c.index("--accelerator-type") + 1] == "v6e-4" and creates(S) == ["asia-northeast1-b"]
    assert S["copied"] == ["mashbox-install.sh", w.name]
    cmd = [c for c in S["calls"] if "--command" in c and "mashbox-install.sh" in c[c.index("--command") + 1]][0]
    script = cmd[cmd.index("--command") + 1]
    assert "MASHBOX_EXTRAS=tpu,hy-nomas" in script and f"MASHBOX_WHEEL=$HOME/{w.name}" in script and "-lt 92" in script
    assert re.search(r"Open +http://127\.0\.0\.1:8[78]\d\d/ in your browser", capsys.readouterr().out)


def test_agents_settings_and_run_args_reach_the_chip(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up("--agents", "--", "--max-tokens", "64") == 0
    cmd = [c for c in read()["calls"] if "--command" in c and "mashbox-install.sh" in c[c.index("--command") + 1]][0]
    script = cmd[cmd.index("--command") + 1]
    assert "run qwen3-8-27b --ctx 32768 --slots 1 --max-tokens 64" in script and "kill -INT" in script
    assert "if [ 0 = 0 ]" in script                                 # no --restart: a running server is kept
    out = capsys.readouterr().out
    assert "Coding agents base URL" in out and "agents-on-tpu" in out and "--agents --restart" not in out


def test_restart_flag(cloud):
    read = cloud(plain={Z[0]: "ok"})
    assert up("--restart") == 0
    script = [c for c in read()["calls"] if "--command" in c and "mashbox-install.sh" in c[c.index("--command") + 1]][0]
    assert "if [ 1 = 0 ]" in script[script.index("--command") + 1]


# ------------------------------------------------------------------------------------------------------------ install.sh
def install_sh():
    return str(resources.files("mashbox.tpu").joinpath("install.sh"))


def test_install_sh_sha256_file_matches():
    d = resources.files("mashbox.tpu")
    want = d.joinpath("install.sh.sha256").read_text().split()
    assert want[1] == "install.sh"
    assert hashlib.sha256(d.joinpath("install.sh").read_bytes()).hexdigest() == want[0]


def test_install_sh_lints():
    subprocess.run(["sh", "-n", install_sh()], check=True)
    sc = shutil.which("shellcheck")
    if not sc:
        pytest.skip("shellcheck not installed")
    for f in ("install.sh", "life.sh"):
        subprocess.run([sc, "-s", "sh", str(resources.files("mashbox.tpu").joinpath(f))], check=True)


@pytest.mark.parametrize("uname_s,uname_m,uid,msg", [
    ("MINGW64_NT-10.0", "x86_64", "1000", "supports Linux and macOS"),
    ("Linux", "ppc64le", "1000", "supports x86_64 and arm64"),
    ("Linux", "x86_64", "0", "not as root"),
])
def test_install_sh_refuses(tmp_path, uname_s, uname_m, uid, msg):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "uname").write_text(f'#!/bin/sh\n[ "$1" = -m ] && echo {uname_m} || echo {uname_s}\n')
    (b / "id").write_text(f"#!/bin/sh\necho {uid}\n")
    for f in b.iterdir():
        f.chmod(0o755)
    r = subprocess.run(["sh", install_sh()], capture_output=True, text=True,
                       env=dict(PATH=f"{b}:/usr/bin:/bin", HOME=str(tmp_path)))
    assert r.returncode == 1 and msg in r.stderr and "uv" not in r.stdout
