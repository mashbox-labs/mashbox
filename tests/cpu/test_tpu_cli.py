"""`mashbox tpu` against a fake gcloud (tests/cpu/fake_gcloud.py), and install.sh checks. No cloud, no network."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources

import re

import pytest

from mashbox import tpu

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_HOLD = tpu.hold                                                 # the fixture replaces tpu.hold
Z = ["us-east5-b", "europe-west4-a", "us-east5-a", "asia-northeast1-b", "us-south1-a", "us-east1-d", "us-central1-b"]


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """Returns setup(**scenario) -> state reader. Fake gcloud on MASHBOX_GCLOUD, state file in tmp_path.
    Local ports: `tpu up` and `tpu tunnel` take free_port(the model's cloud port), and a tpu.Port stays bound until its
    process exits (by design: Port.release). In a test they get a port the test holds instead (bound with SO_REUSEADDR,
    never listening, as api_stub_engine.held_port), and every tpu.Port the test made is closed when it ends, so tests
    never use 8000 to 8019 (the model's range; several test copies on one host, or a server on 8000, ran it out:
    'ports 8000 to 8019 on 127.0.0.1 are all in use here') and no port outlives its test."""
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
        if hasattr(a[4], "close"):                    # the held local port (tpu.Port)
            a[4].close()
        return 0
    monkeypatch.setattr(tpu, "hold", hold)

    def setup(**kw):
        s = dict(account="user@example.com", project="p1", api=True, plain={}, grant={})
        s.update(kw)
        st.write_text(json.dumps(s))
        return lambda: json.loads(st.read_text())
    setup.held = held
    made, kept = [], []
    setup.ports = []                                  # each port handed to free_port in place of a model's port
    setup.asked = []                                  # the model's cloud port each of those took the place of
    real_port, real_free = tpu.Port, tpu.free_port
    setup.model_port = {m["id"]: m["cloud"]["port"] for m in tpu.registry()["models"] if "port" in (m.get("cloud") or {})}
    model_ports = set(setup.model_port.values())

    class Port(real_port):
        def __init__(self, port):
            super().__init__(port)
            made.append(self)

    def free_port(p):
        if p in model_ports:
            setup.asked.append(p)
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            kept.append(s)
            p = s.getsockname()[1]
            setup.ports.append(p)
        return real_free(p)
    monkeypatch.setattr(tpu, "Port", Port)
    monkeypatch.setattr(tpu, "free_port", free_port)
    yield setup
    for x in made:
        x.close()
    for s in kept:
        s.close()


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
    assert out.count(": no free chip (") == 2 and "$2.70 per hour" in out and "estimate at Google's on-demand list price" in out
    assert cloud.asked == [cloud.model_port["qwen3-8-27b"]]                   # up asked for the port of its own model
    assert re.search(rf"Local URL +http://127\.0\.0\.1:{cloud.ports[-1]}/v1 ", out) and "mashbox tpu down mashbox-qwen3-8-27b" in out


def test_internal_error_shows_googles_message_and_continues(cloud, capsys):
    """Seen on a real create (europe-west4-a, after 3 min): code 13 JSON error; the user saw only "}"."""
    read = cloud(plain={Z[0]: "internal", Z[1]: "ok"})
    assert up() == 0 and creates(read()) == Z[:2]
    out = capsys.readouterr().out
    assert "Google said: an internal error has occurred" in out and "\n    }\n" not in out


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


def test_create_accepted_then_sign_in_expired_names_the_chip(cloud, capsys):
    """Seen on a real create: the sign-in expired during a 9 min create; `up` stopped and Google kept creating."""
    read = cloud(plain={Z[0]: "accepted-auth", Z[1]: "ok"})
    assert up() == 2
    S = read()
    assert creates(S) == Z[:1] and list(S["nodes"]) == [f"{Z[0]}/mashbox-qwen3-8-27b"]   # no second chip
    looks = [c for c in S["calls"] if c[2:4] == ["tpu-vm", "describe"] and c[4] == "mashbox-qwen3-8-27b"]
    assert [d[d.index("--zone") + 1] for d in looks] == [Z[0]]                            # looked up by its exact name
    assert not [c for c in S["calls"] if c[2:4] == ["tpu-vm", "delete"]]                  # never deleted unchecked
    err = capsys.readouterr().err
    assert "gcloud auth login, then delete it: mashbox tpu down mashbox-qwen3-8-27b" in err
    assert f"gcloud compute tpus tpu-vm delete mashbox-qwen3-8-27b --zone {Z[0]} --project p1 --quiet" in err


def test_create_accepted_then_error_is_deleted_before_the_next_zone(cloud, capsys):
    read = cloud(plain={Z[0]: "accepted-internal", Z[1]: "ok"})
    assert up() == 0
    S = read()
    assert list(S["nodes"]) == [f"{Z[1]}/mashbox-qwen3-8-27b"]
    order = [(c[3], c[c.index("--zone") + 1]) for c in S["calls"] if c[2] == "tpu-vm" and c[3] in ("create", "delete")]
    assert order[:3] == [("create", Z[0]), ("delete", Z[0]), ("create", Z[1])]
    assert f"{Z[0]}: mashbox-qwen3-8-27b deleted" in capsys.readouterr().out


def test_queued_request_after_an_error_counts_as_made(cloud, capsys):
    read = cloud(plain={}, qr_create={Z[0]: "accepted-internal"}, grant={Z[1]: 2})
    assert up() == 0
    assert list(read()["qrs"]) == [f"{Z[1]}/mashbox-qwen3-8-27b-q"]        # the one made with an error is deleted
    assert "it may have been made" in capsys.readouterr().out


def test_queued_request_after_an_error_is_watched_and_can_win(cloud, capsys):
    read = cloud(plain={}, qr_create={Z[0]: "accepted-internal"}, grant={Z[0]: 1})
    assert up() == 0
    S = read()
    assert list(S["qrs"]) == [f"{Z[0]}/mashbox-qwen3-8-27b-q"] and list(S["nodes"]) == [f"{Z[0]}/mashbox-qwen3-8-27b"]


def test_existing_queued_request_of_that_name_is_left_alone(cloud):
    read = cloud(plain={}, grant={Z[1]: 1})
    S = read()
    S["qrs"] = {f"{Z[0]}/mashbox-qwen3-8-27b-q": {"name": f"projects/p/locations/{Z[0]}/queuedResources/mashbox-qwen3-8-27b-q",
                "state": {"state": "WAITING_FOR_RESOURCES"}, "describes": 0, "args": [],
                "tpu": {"nodeSpec": [{"nodeId": "someone-else", "node": {}}]}}}           # no mashbox label
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    assert up() == 0
    assert f"{Z[0]}/mashbox-qwen3-8-27b-q" in read()["qrs"]


def test_queued_request_then_sign_in_expired_prints_the_delete(cloud, capsys):
    read = cloud(plain={}, qr_create={Z[0]: "accepted-auth"})
    assert up() == 2
    assert list(read()["qrs"]) == [f"{Z[0]}/mashbox-qwen3-8-27b-q"]        # Google holds it; the user is told how
    out, err = capsys.readouterr()
    assert (f"gcloud compute tpus queued-resources delete mashbox-qwen3-8-27b-q --zone {Z[0]} --project p1 --force --quiet"
            in out) and "gcloud auth login" in err


def test_existing_node_of_that_name_is_never_deleted(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    S = read()
    S["nodes"] = {f"{Z[0]}/mine": {"name": f"projects/p/locations/{Z[0]}/nodes/mine", "state": "STOPPED"}}   # no mashbox label
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    tpu.main(["up", "qwen3-8-27b", "--yes", "--name", "mine", "--queue-minutes", "0"])
    S = read()
    assert f"{Z[0]}/mine" in S["nodes"] and not [c for c in S["calls"] if c[2:4] == ["tpu-vm", "delete"]]


def test_existing_ready_node_of_that_name_is_never_used(cloud, monkeypatch, capsys):
    read = cloud(plain={Z[0]: "ok"})
    S = read()
    S["nodes"] = {f"{Z[0]}/mine": {"name": f"projects/p/locations/{Z[0]}/nodes/mine", "state": "READY"}}     # no mashbox label
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    touched = []
    monkeypatch.setattr(tpu, "wait_ready_node", lambda *a, **k: touched.append(("wait", a)))
    monkeypatch.setattr(tpu, "remote_setup", lambda *a, **k: touched.append(("setup", a)))
    assert tpu.main(["up", "qwen3-8-27b", "--yes", "--name", "mine", "--queue-minutes", "0"]) == 2
    S = read()
    assert not touched and f"{Z[0]}/mine" in S["nodes"] and creates(S) == []       # found by name first: no create sent
    assert not [c for c in S["calls"] if c[2:4] == ["tpu-vm", "delete"]] and "mine" not in tpu.load_state()
    assert "already exists in " + Z[0] in capsys.readouterr().err


def test_one_failed_describe_is_not_gone(cloud, monkeypatch):
    read = cloud(plain={Z[0]: "ok"})
    real, n = tpu.gj, {"i": 0}

    def gj(args, project=None):
        if args[3] == "describe" and n["i"] < 2:
            n["i"] += 1
            return None
        return real(args, project)
    monkeypatch.setattr(tpu, "gj", gj)
    assert up("--queue-minutes", "0") == 0 and creates(read()) == Z[:1] and f"{Z[0]}/mashbox-qwen3-8-27b" in read()["nodes"]


def test_stop_after_the_chip_exists_names_it(cloud, monkeypatch, capsys):
    cloud(plain={Z[0]: "ok"})

    def setup(*a, **k):
        raise tpu.Stop("install failed")
    monkeypatch.setattr(tpu, "remote_setup", setup)
    assert up() == 2
    err = capsys.readouterr().err
    assert "install failed" in err and "still running and billing. Take it down: mashbox tpu down mashbox-qwen3-8-27b" in err


def test_down_goes_on_when_the_check_list_fails(cloud, monkeypatch, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    real, n = tpu.scan, {"i": 0}

    def scan(project, zones, unchanged=True, names=()):
        n["i"] += 1
        if n["i"] == 2:                                              # the check list after the delete
            raise tpu.Stop("could not list chips in " + zones[0])
        return real(project, zones, unchanged, names)
    monkeypatch.setattr(tpu, "scan", scan)
    capsys.readouterr()
    assert tpu.main(["down", "mashbox-qwen3-8-27b"]) == 1
    out = capsys.readouterr().out
    assert read()["nodes"] == {} and "delete sent, but the check failed" in out and "Nothing was changed" not in out


def test_down_with_zone_looks_only_there(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    S = read()
    S["list_error"] = {Z[3]: "other"}                                # a zone whose list always fails
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    assert tpu.main(["down", "mashbox-qwen3-8-27b"]) == 2 and "add --zone ZONE" in capsys.readouterr().err
    assert tpu.main(["down", "mashbox-qwen3-8-27b", "--zone", Z[0]]) == 0 and read()["nodes"] == {}


def test_googles_message_with_quotes_is_read_whole():
    e = 'ERROR: (gcloud) {"code": 3, "message": "Invalid value for field \\"node.runtime_version\\": v9"}'
    assert tpu.gerr(e) == 'Invalid value for field "node.runtime_version": v9'


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


@pytest.mark.parametrize("err,says", [("auth", "gcloud auth login"), ("other", "could not list chips in us-east5-a")])
def test_failed_list_never_reads_as_nothing_up(cloud, capsys, err, says):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    S = read()
    S["list_error"] = {Z[2]: err}                                    # one zone's list fails; the chip is in Z[0]
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    capsys.readouterr()
    for cmd in (["status"], ["down"]):
        assert tpu.main(cmd) == 2
        o = capsys.readouterr()
        assert says in o.err and "Nothing was changed" in o.err and "No mashbox chips" not in o.out + o.err
    assert read()["nodes"] and "mashbox-qwen3-8-27b" in json.load(open(tpu.STATE))   # not deleted, not forgotten


def test_held_tunnel_stops_when_the_chip_is_gone(cloud, capsys, monkeypatch):
    """Seen in real use: `down` in a second terminal left the first one reopening a tunnel to a deleted chip
    while it still said the chip bills. hold() must notice the chip is gone and stop."""
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    S = read()
    S["nodes"] = {}                                                  # taken down from another terminal
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    dead = subprocess.Popen(["true"])
    dead.wait()
    m = tpu.model("qwen3-8-27b")
    assert REAL_HOLD(dead, "p1", Z[0], "mashbox-qwen3-8-27b", m, 8000, None) == 0
    assert "is gone" in capsys.readouterr().out
    assert "mashbox-qwen3-8-27b" not in json.load(open(tpu.STATE))


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
    assert "MASHBOX_EXTRAS=tpu,hy-nomas" in script and f'MASHBOX_WHEEL="$HOME"/{w.name}' in script and "-lt 92" in script
    assert cloud.asked == [cloud.model_port["hy-nomas"]]                      # up asked for the port of its own model
    assert re.search(rf"Open +http://127\.0\.0\.1:{cloud.ports[-1]}/ in your browser", capsys.readouterr().out)


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


# ---------------------------------------------------------------------------------------------------------- security
def test_f19_another_local_user_cannot_answer_for_the_chip(cloud, monkeypatch):
    """The local port was checked free, released, and bound by ssh a moment later; another user of this machine who
    bound it first got "Server ready", then every prompt and the agent's tool calls. Now this process binds the port and
    keeps it, and the forward is a Unix socket in a private directory: the race has nothing to win."""
    cloud(plain={Z[0]: "ok"})
    real, hits, evil = tpu.open_tunnel, [], []

    class Evil(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"evil")

        def log_message(self, *a):
            pass

    def racing(*a):                                     # runs right before ssh binds its forward
        port = a[4] if isinstance(a[4], int) else a[4].port
        try:
            srv = HTTPServer(("127.0.0.1", port), Evil)
        except OSError:
            return real(*a)                             # the port is held: nobody else can bind it
        evil.append(srv)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return real(*a)
    monkeypatch.setattr(tpu, "open_tunnel", racing)
    try:
        assert up() == 0
    finally:
        for srv in evil:
            srv.shutdown()
            srv.server_close()
    assert evil == [] and hits == []                    # "Server ready" came from the chip, through our own forward


def test_f19_forward_failure_stops_loudly(cloud, monkeypatch, capsys):
    """A forward SSH cannot set up (ExitOnForwardFailure) ends `up` with SSH's message; it is never retried in silence
    while something else may answer on the port."""
    cloud(plain={Z[0]: "ok"}, tunnel_fail=True)
    monkeypatch.setattr(tpu, "READY_MIN", 0.05)
    assert up() == 2
    err = capsys.readouterr().err
    assert "could not open its forward" in err and "forwarding failed" in err.lower()


def a_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_held_port_is_free_again_right_after_its_session():
    """The port a session held and closed is taken again at once (the same URL as before), although the connections it
    relayed sit in TIME_WAIT on that port for a minute; a port another socket listens on is still refused."""
    hold = socket.socket()          # p stays bound here (SO_REUSEADDR, never listening) for the whole test: no other
    hold.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)            # process can take it, and Port still binds it
    hold.bind(("127.0.0.1", 0))
    p = hold.getsockname()[1]
    lp = tpu.free_port(p)
    assert lp.port == p
    c = socket.create_connection(("127.0.0.1", p), timeout=5)
    assert c.recv(1) == b""                             # no forward yet: our relay closes first, so TIME_WAIT is ours
    c.close()
    lp.close()
    again = tpu.free_port(p)
    try:
        assert again.port == p
        other = socket.socket()
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            other.bind(("127.0.0.1", p))
        other.close()
    finally:
        again.close()
        hold.close()


def test_port_in_use_leaves_no_private_folder(tmp_path, monkeypatch):
    """A port that cannot be bound makes no private folder: none is left behind for each busy port tried."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        with pytest.raises(OSError):
            tpu.Port(busy.getsockname()[1])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("argv", [["down", "Mashbox_X"], ["tunnel", "a" * 64], ["ssh", "x;id"],
                                  ["up", "qwen3-8-27b", "--yes", "--name", "x y"],
                                  ["status", "--zone", "us-east5-b --verbosity=debug"],
                                  ["status", "--project=--impersonate-service-account=e@x.iam.gserviceaccount.com"]])
def test_f10_names_zones_projects_are_checked_before_gcloud(cloud, argv, capsys):
    """On the mashbox tpu side, a chip name, zone or project that is not one can never reach gcloud's argv, where it
    could be read as an option or another chip."""
    read = cloud(plain={Z[0]: "ok"})
    with pytest.raises(SystemExit) as e:
        tpu.main(argv)
    assert e.value.code == 2 and "not a valid" in capsys.readouterr().err
    assert read().get("calls", []) == []


def test_chip_runs_the_wheel_it_was_given_and_a_server_from_another_mashbox_restarts(cloud, tmp_path, monkeypatch, capsys):
    """The Mac app passes its own wheel (MASHBOX_WHEEL): the chip installs that wheel, never PyPI's release of the same
    version number, and a server that was started from another mashbox (PyPI 0.1.0 had no request guard) is started
    again instead of being reused under the package the install just replaced."""
    w = tmp_path / "mashbox-0.1.0+gabc1234-py3-none-any.whl"
    w.write_bytes(b"wheel bytes")
    monkeypatch.setenv("MASHBOX_WHEEL", str(w))
    read = cloud(plain={Z[0]: "ok"}, server="running", server_pkg="mashbox==0.1.0")
    assert up("--agents") == 0
    S = read()
    assert S["copied"] == ["mashbox-install.sh", w.name]
    script = [c for c in S["calls"] if "--command" in c and "mashbox-install.sh" in c[c.index("--command") + 1]][0]
    script = script[script.index("--command") + 1]
    sha = hashlib.sha256(b"wheel bytes").hexdigest()
    assert f'MASHBOX_WHEEL="$HOME"/{w.name}' in script and "MASHBOX_VERSION" not in script
    assert f"pkg='wheel sha256={sha}'" in script and 'echo "$pkg" > ~/.mashbox/serve.pkg' in script
    out = capsys.readouterr().out
    assert "came from another mashbox (mashbox==0.1.0)" in out and "server started" in out
    assert S["server_pkg"] == f"wheel sha256={sha}"
    assert up("--agents") == 0                                      # same wheel again: the server is kept
    assert "server running" in capsys.readouterr().out and read()["starts"] == 2


def test_without_a_wheel_the_chip_pins_this_version(cloud):
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    script = [c for c in read()["calls"] if "--command" in c and "mashbox-install.sh" in c[c.index("--command") + 1]][0]
    script = script[script.index("--command") + 1]
    v = __import__("mashbox").__version__
    assert f"MASHBOX_VERSION={v}" in script and f"pkg=mashbox=={v}" in script and "MASHBOX_WHEEL" not in script


def test_a_second_start_with_other_settings_starts_the_server_again(cloud, capsys):
    """A second Start on the same chip with other settings (the app's Coding agents box) starts the server again with
    them; the same settings keep it running."""
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    capsys.readouterr()
    assert up("--agents") == 0
    out = capsys.readouterr().out
    assert "had other settings (run qwen3-8-27b)" in out and "server started" in out
    assert up("--agents") == 0
    assert "server running" in capsys.readouterr().out
    S = read()
    assert S["starts"] == 3 and len(creates(S)) == 1 and S["server_args"] == "run qwen3-8-27b --ctx 32768 --slots 1"


def test_chip_script_keeps_a_server_only_for_the_same_model_and_settings(monkeypatch, tmp_path):
    """The script `tpu up` runs on the chip, run by a real shell: a server keeps running only when it runs the same
    mashbox, model and settings; otherwise it gets Ctrl-C (SIGINT) and starts again with the new ones."""
    import shutil
    import subprocess
    import threading
    import types
    calls = []

    def gc(args, project=None, timeout=None):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="MBX_SERVER=started\n", stderr="")
    monkeypatch.setattr(tpu, "gc", gc)
    monkeypatch.setattr(tpu, "still_ours", lambda *a, **k: None)
    monkeypatch.delenv("MASHBOX_WHEEL", raising=False)
    monkeypatch.delenv("MASHBOX_HF_TOKEN_FILE", raising=False)

    def script(mid, extra=()):
        tpu.remote_setup(tpu.model(mid), "p", Z[0], "c1", list(extra))
        return calls[-1][calls[-1].index("--command") + 1]
    home, stub = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    stub.mkdir()
    (home / "mashbox-install.sh").write_text("exit 0\n")
    sleep = shutil.which("sleep")
    for name, body in {"python3": 'echo started >> "$HOME/launched"\n',                # the server launch, recorded
                       "df": "echo 'Filesystem 1024-blocks Used Available Capacity Mounted'\n"
                             "echo '/dev/x 1 1 999999999999 1% /'\n",
                       "sleep": f"exec {sleep} 0.05\n"}.items():
        (stub / name).write_text("#!/bin/sh\n" + body)
        (stub / name).chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{stub}:{os.environ['PATH']}"}

    def run(text):
        r = subprocess.run(["bash", "-c", text], env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    def launched(n):                                # the launch runs in the background (&): wait for its record
        import time
        for _ in range(200):
            f = home / "launched"
            if f.exists() and f.read_text().count("started") >= n:
                break
            time.sleep(0.05)
        assert (home / "launched").read_text().count("started") == n

    def serving():                                  # a server process in its own group, as LAUNCH starts one
        p = subprocess.Popen([sleep, "600"], start_new_session=True)
        threading.Thread(target=p.wait, daemon=True).start()
        (home / ".mashbox" / "serve.pid").write_text(f"{p.pid}\n")
        return p
    plain, agents = script("qwen3-8-27b"), script("qwen3-8-27b", ["--ctx", "32768", "--slots", "1"])
    assert "MBX_SERVER=started" in run(plain)
    assert (home / ".mashbox" / "serve.args").read_text() == "run qwen3-8-27b\n"
    launched(1)
    p = serving()
    out = run(plain)                                 # same mashbox, model and settings: kept
    assert "MBX_SERVER=running" in out and "MBX_ARGS=run qwen3-8-27b" in out and p.poll() is None
    launched(1)
    out = run(agents)                                # the Coding agents box: other settings
    assert "MBX_CHANGED=run qwen3-8-27b" in out and "MBX_SERVER=started" in out
    p.wait(timeout=10)
    assert (home / ".mashbox" / "serve.args").read_text() == "run qwen3-8-27b --ctx 32768 --slots 1\n"
    p = serving()
    out = run(script("hy-nomas"))                    # another model on the same chip name
    assert "MBX_CHANGED=run qwen3-8-27b --ctx 32768 --slots 1" in out and "MBX_SERVER=started" in out
    p.wait(timeout=10)
    launched(3)

def test_up_chip_choices_come_from_passing_chip_rows():
    """`tpu up --chip` offers exactly the models.json chips rows that pass (one source with the docs chip tables)."""
    q = tpu.model("qwen3-8-27b")
    names = [c for c, _ in tpu.offered(q)]
    assert names[0] == "v6e-1" and set(names) == {"v6e-1", "v5e-4", "v6e-4"}
    assert tpu.pick_chip(q, None) is q
    v5e = tpu.pick_chip(q, "v5e-4")["cloud"]
    assert (v5e["accelerator"], v5e["runtime"]) == ("v5litepod-4", "v2-alpha-tpuv5-lite") and v5e["zones"]
    assert tpu.price(v5e["accelerator"]) == pytest.approx(4.80)
    assert q["cloud"]["accelerator"] == "v6e-1"          # the registry entry itself is unchanged
    for chip in ("v5p-8", "v5e-1"):                       # not passing / not a row: refused with the list
        with pytest.raises(tpu.Stop, match="runs on: v6e-1"):
            tpu.pick_chip(q, chip)
    # a chip nobody ran reads "Not tested yet", a measured chip prints its measured line, a passing chip type names its
    # host (test_up_chip_never_says_not_on_for_a_chip_the_model_ran_on); "Not on" stays for a chip with no row
    with pytest.raises(tpu.Stop) as e:
        tpu.pick_chip(q, "v5p-8")
    assert "Not tested yet" in str(e.value) and "Not on" not in str(e.value), e.value
    with pytest.raises(tpu.Stop, match="Not on v5e-1"):
        tpu.pick_chip(q, "v5e-1")
    hy = tpu.model("hy-nomas")
    for chip, row in ((c.get("accelerator") or k, c) for k, c in hy["chips"].items() if c.get("status") == "measured"):
        with pytest.raises(tpu.Stop) as e:
            tpu.pick_chip(hy, chip)
        assert row["run_note"] in str(e.value) and "Not on" not in str(e.value), e.value
    assert [c for c, _ in tpu.offered(tpu.model("hy-nomas"))] == ["v6e-4"]


def test_up_chip_never_says_not_on_for_a_chip_the_model_ran_on():
    """`tpu up --chip` with any chips row's key or host (v6e, v5e, v6e-4, v6e-1, v5e-4, every model with TPU settings)
    never answers "Not on" when that row has a run (passing or measured): a passing row named by its chip type names the
    host to ask for ("On v6e: use --chip v6e-1"), a measured one prints its measured line. "Not on" is left for a chip
    with no row or a failing one."""
    seen = 0
    for m in (tpu.model(r["id"]) for r in tpu.registry()["models"] if "cloud" in r):
        for key, row in m["chips"].items():
            if not isinstance(row, dict) or row.get("status") not in ("passing", "measured"):
                continue
            for chip in dict.fromkeys((key, row.get("accelerator") or key)):
                seen += 1
                try:
                    out = tpu.pick_chip(m, chip)
                except tpu.Stop as e:
                    said = str(e)
                    assert "Not on" not in said, (m["id"], chip, said)
                    if row["status"] == "passing":
                        assert f"use --chip {row['accelerator']}" in said, (m["id"], chip, said)
                    continue
                assert row["status"] == "passing" and out["cloud"]["accelerator"] == row["gcloud_type"], (m["id"], chip)
    assert seen >= 6


# ------------------------------------------------------------------------------------- run on chips that are too small
# `mashbox run` on a chip split that cannot hold the model stops with one line before any weights are read (CHIPS-VERIFY
# case C: `-tp 1` on a v5e-4 loaded ~14.7 GB onto chip 0, then a raw RESOURCE_EXHAUSTED traceback). Chip memory as
# JAX reports it (bytes_limit, MEASURED: v5e 15.75 GiB, v6e 31.24 GiB); the config has the real Qwen3.8-27B shapes.
QWEN_CFG = dict(model_type="qwen3_5_text", num_hidden_layers=64, hidden_size=5120, intermediate_size=17408,
                num_attention_heads=24, num_key_value_heads=4, head_dim=256, vocab_size=248320, linear_num_key_heads=16,
                linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                full_attention_interval=4, tie_word_embeddings=False,
                rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25))
GIB = 2 ** 30
CHIP = {"v5e": ("TPU v5 lite", int(15.75 * GIB)), "v6e": ("TPU v6 lite", int(31.24 * GIB))}


class WeightsReached(Exception):
    pass


@pytest.fixture
def host(tmp_path, monkeypatch):
    """host(family, chips) -> run(*args): cli.load on a fake TPU host; WeightsReached when it gets to the weights."""
    jax = pytest.importorskip("jax")
    from mashbox import cli, hub
    up = tmp_path / "upstream"
    up.mkdir()
    (up / "config.json").write_text(json.dumps(QWEN_CFG))
    monkeypatch.setattr(hub, "upstream_dir", lambda m, log=print: str(up))
    monkeypatch.setattr(cli, "setup_cache", lambda d=None: str(tmp_path / "xla"))     # no global JAX cache dir in tests

    def weights(*a, **k):
        raise WeightsReached()
    monkeypatch.setattr(hub, "weights_file", weights)

    def setup(fam, n):
        kind, limit = CHIP[fam]

        class Chip:
            platform, device_kind = "tpu", kind

            def memory_stats(self):
                return {"bytes_limit": limit}
        monkeypatch.setattr(jax, "devices", lambda *a: [Chip() for _ in range(n)])
        return lambda *args: cli.load(cli.build_parser().parse_args(["run", "qwen3-8-27b", "--prompt", "Hi", *args]))
    return setup


def stop_line(run, *args):
    with pytest.raises(SystemExit) as e:
        run(*args)
    msg = str(e.value.code)
    assert "\n" not in msg and "Traceback" not in msg and "RESOURCE_EXHAUSTED" not in msg
    return msg


def test_one_v5e_chip_of_four_stops_at_once_with_the_split_and_chip_that_work(host):
    msg = stop_line(host("v5e", 4), "--tensor-parallel-size", "1")
    assert msg.startswith("qwen3-8-27b does not fit on 1 TPU v5e chip: it needs at least 29.53 GB per chip")
    assert "weights 26.94 GB + context 2.59 GB for --ctx 2048 x --slots 8" in msg and "each chip here has 16.91 GB" in msg
    assert "leave out --tensor-parallel-size (it then uses 4)" in msg
    assert msg.endswith("mashbox tpu up qwen3-8-27b --chip v5e-4 (from your own machine).")
    from mashbox import hub
    m = hub.model_entry("qwen3-8-27b")
    assert tpu.pick_chip(m, "v5e-4")["cloud"]["accelerator"] == "v5litepod-4"      # the line it gives is one tpu up takes


def test_a_one_chip_v5e_host_points_to_the_v5e_4(host):
    msg = stop_line(host("v5e", 1))
    assert msg.startswith("qwen3-8-27b does not fit on 1 TPU v5e chip: it needs at least 29.53 GB per chip")
    assert msg.endswith("It runs on a v5e-4: mashbox tpu up qwen3-8-27b --chip v5e-4 (from your own machine).")


def test_too_long_a_context_names_ctx_and_slots(host):
    msg = stop_line(host("v5e", 4), "--ctx", "262144")
    assert "does not fit on 4 TPU v5e chips" in msg and "Lower --ctx or --slots" in msg


@pytest.mark.parametrize("fam,n,args", [
    ("v5e", 4, ()),                                       # auto tp 4: the P7 / CHIPS-VERIFY B run
    ("v5e", 4, ("--ctx", "32768")),                       # 32768 x 8 slots: peak 12.06 GB per chip measured
    ("v5e", 4, ("--ctx", "131072", "--slots", "1")),      # 131072 x 1: needle recalled
    ("v6e", 1, ()),                                       # v6e-1, the public default
    ("v6e", 1, ("--ctx", "32768", "--slots", "1")),       # agents setting on one v6e chip
    ("v6e", 4, ("--ctx", "131072")),                      # v6e-4 tp 4 at 131072 x 8: peak 26.56 GB per chip
])
def test_settings_that_ran_on_real_chips_are_never_stopped(host, fam, n, args):
    with pytest.raises(WeightsReached):
        host(fam, n)(*args)


def test_top_help_lists_tpu_and_every_tpu_command_has_help(capsys):
    """`mashbox --help` names the tpu command family, and `mashbox tpu --help` describes each of its commands."""
    from mashbox import cli
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    flat = " ".join(out.split())        # argparse wraps help lines at the terminal width
    assert "{env,tpu,run,download,models,inspect}" in flat and "mashbox tpu --help" in flat, out
    with pytest.raises(SystemExit):
        cli.main(["tpu", "--help"])
    out = capsys.readouterr().out
    for n in ("up", "down", "status", "ssh", "tunnel"):
        line = next((l for l in out.splitlines() if l.strip().startswith(n + " ") or l.strip() == n), None)
        assert line is not None and len(line.split(None, 1)) == 2, (n, out)
