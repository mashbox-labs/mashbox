"""`mashbox tpu` acts only on chips mashbox made, in the zone it was told, opens the local port only after the chip's
server refused a request from another web site, and installs on the chip exactly the wheel bytes it hashed.
Fake gcloud (tests/cpu/fake_gcloud.py); no cloud, no network."""
import hashlib
import json
import os
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mashbox import tpu
from test_tpu_cli import REAL_HOLD, Z, cloud, creates, up  # noqa: F401  (cloud is the fixture)


def put(read, **kw):
    S = read()
    S.update(kw)
    with open(os.environ["FAKE_GCLOUD_STATE"], "w") as f:
        json.dump(S, f)


def plain_node(zone, name, labels=None, **kw):
    return {"name": f"projects/p1/locations/{zone}/nodes/{name}", "state": "READY", "acceleratorType": "v6e-1",
            "createTime": "2026-01-01T00:00:00.000000Z", "labels": labels or {}, **kw}


def ours(zone, name):
    return plain_node(zone, name, {"mashbox": "tpu-up", "mashbox-model": "qwen3-8-27b"})


def deletes(S):
    return [c for c in S["calls"] if c[2:4] in (["tpu-vm", "delete"], ["queued-resources", "delete"])]


def installs(S):
    return [c for c in S["calls"] if "--command" in c and "mashbox-install.sh" in c[c.index("--command") + 1]]


def script_of(S):
    c = installs(S)[0]
    return c[c.index("--command") + 1]


# ------------------------------------------------------------------ chips mashbox did not make (F7)
def test_up_name_never_uses_a_chip_made_outside_mashbox_in_any_zone(cloud, capsys):
    """A chip with that name made outside mashbox (plain gcloud, the console, someone else) is found in every zone, and
    `up` neither uses it (no install, no server restart) nor creates a second chip with that name in an earlier zone."""
    read = cloud(plain={Z[0]: "ok", Z[3]: "ok"})
    put(read, nodes={f"{Z[3]}/handed-1": plain_node(Z[3], "handed-1")})
    assert up("--name", "handed-1") == 2
    S = read()
    assert creates(S) == [] and not installs(S) and not deletes(S) and list(S["nodes"]) == [f"{Z[3]}/handed-1"]
    assert "handed-1 already exists in asia-northeast1-b and was not made by mashbox" in capsys.readouterr().err
    assert "handed-1" not in tpu.load_state()


def test_down_name_finds_a_chip_made_outside_mashbox_and_leaves_it(cloud, capsys):
    """`down <name>` never says "nothing to take down" while a chip of that name bills (the app reads exit 0 as gone
    and forgets it); it names the chip and its zone, deletes nothing, and exits non-zero."""
    read = cloud(plain={})
    put(read, nodes={f"{Z[3]}/handed-1": plain_node(Z[3], "handed-1")})
    assert tpu.main(["down", "handed-1"]) == 2
    o = capsys.readouterr()
    assert "Nothing to take down" not in o.out and "handed-1 already exists in asia-northeast1-b" in o.err
    assert "gcloud compute tpus tpu-vm delete handed-1" in o.err
    S = read()
    assert not deletes(S) and list(S["nodes"]) == [f"{Z[3]}/handed-1"]


# ------------------------------------------------------------------ the local list is not proof of ownership (F14)
@pytest.mark.parametrize("entry", [dict(project="other-project", zone=Z[5]),       # stale: another project and zone
                                   dict(project="p1", zone=Z[3])])                # same place: the name was reused since
def test_a_name_in_the_local_list_never_makes_a_chip_ours(cloud, capsys, entry):
    """tpu.json can hold a name from another project or zone, or from a mashbox chip deleted since whose name an
    unlabelled chip now has. Only our label makes a chip mashbox's: up, down (named, with --zone, --all), ssh (named or
    not), tunnel and status all leave that chip alone and never reach it."""
    read = cloud(plain={Z[0]: "ok"})
    put(read, nodes={f"{Z[3]}/victim": plain_node(Z[3], "victim")})
    runs = [(["up", "qwen3-8-27b", "--yes", "--name", "victim"], 2), (["down", "victim"], 2),
            (["down", "victim", "--zone", Z[3]], 2), (["down", "--all"], 0), (["ssh", "victim"], 2), (["ssh"], 2),
            (["tunnel", "victim"], 2), (["tunnel"], 2), (["status"], 0)]
    for argv, rc in runs:
        tpu.save_state("victim", model="qwen3-8-27b", qr=None, max_hours=2, created=1.0, **entry)   # before every run
        capsys.readouterr()
        assert tpu.main(argv) == rc, argv
        o = capsys.readouterr()
        if argv[0] == "status":
            assert "victim" not in o.out and "No mashbox chips" in o.out
        if argv[:2] == ["down", "--all"]:
            assert "Nothing to take down" in o.out
    S = read()
    assert creates(S) == [] and not installs(S) and not deletes(S) and list(S["nodes"]) == [f"{Z[3]}/victim"]
    assert not [c for c in S["calls"] if c[2:4] == ["tpu-vm", "ssh"]]           # never reached: no ssh, no tunnel


def test_reused_chip_says_it_keeps_the_lifetime_it_was_made_with(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    assert up("--max-hours", "3") == 0
    capsys.readouterr()
    assert up("--max-hours", "1") == 0
    assert "mashbox-qwen3-8-27b keeps the max lifetime it was made with (3 h)" in capsys.readouterr().out
    assert creates(read()) == [Z[0]]


def test_a_zone_whose_list_raises_is_never_read_as_empty(cloud, monkeypatch, capsys):
    """An error inside one zone's list (gcloud gone mid-run, a failed exec) stops the command; it is never a zone with
    no chips, which would let up create a second chip of that name or down say there is nothing."""
    read = cloud(plain={Z[0]: "ok"})
    put(read, nodes={f"{Z[3]}/mashbox-qwen3-8-27b": ours(Z[3], "mashbox-qwen3-8-27b")})
    real = tpu.gc

    def gc(args, project=None, **kw):
        if args[3:4] == ["list"] and Z[3] in args:
            raise OSError("exec failed")
        return real(args, project, **kw)
    monkeypatch.setattr(tpu, "gc", gc)
    assert up() == 2 and "could not list chips in asia-northeast1-b" in capsys.readouterr().err
    assert tpu.main(["down", "mashbox-qwen3-8-27b"]) == 2
    S = read()
    assert creates(S) == [] and not deletes(S)


# ------------------------------------------------------------------ --zone, and one name in two zones (F9)
def test_zone_flag_searches_only_that_zone(cloud, capsys):
    """--zone: that zone only. A chip of the same name in another zone is not found, used or changed."""
    read = cloud(plain={Z[0]: "ok"})
    put(read, nodes={f"{Z[3]}/mashbox-qwen3-8-27b": ours(Z[3], "mashbox-qwen3-8-27b")})
    assert up("--zone", Z[0]) == 0
    S = read()
    listed = {c[c.index("--zone") + 1] for c in S["calls"] if c[3:4] == ["list"]}
    assert listed == {Z[0]} and creates(S) == [Z[0]]
    assert "already exists in asia-northeast1-b" not in capsys.readouterr().out
    assert all(Z[3] not in " ".join(c) for c in installs(S))


def test_a_name_in_two_zones_is_never_guessed(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"})
    two = {f"{z}/twin": ours(z, "twin") for z in (Z[0], Z[3])}
    put(read, nodes=two)
    assert up("--name", "twin") == 2
    assert "chips named twin are in asia-northeast1-b, us-east5-b. Name the zone" in capsys.readouterr().err
    assert tpu.main(["down", "twin"]) == 2 and "Name the zone" in capsys.readouterr().err
    S = read()
    assert not installs(S) and not deletes(S) and sorted(S["nodes"]) == sorted(two)
    assert tpu.main(["down", "twin", "--zone", Z[3]]) == 0                      # the zone named: only that one goes
    assert list(read()["nodes"]) == [f"{Z[0]}/twin"]


# ------------------------------------------------------------------ the local port opens after the guard check (F11)
def unix_server(path, body=b"secret"):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def address_string(self):
            return "u"

        def log_message(self, *a):
            pass
    import socketserver

    class U(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True
    s = U(path, H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def tcp_get(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as c:
            c.sendall(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            data = b""
            while True:
                d = c.recv(65536)
                if not d:
                    return data
                data += d
    except OSError:
        return b""


def test_port_relays_nothing_before_it_is_opened():
    lp = tpu.free_port(18600)
    srv = unix_server(lp.sock)
    try:
        assert tcp_get(lp.port) == b""                         # the forward answers, but the port was not opened yet
        lp.open()
        assert tcp_get(lp.port).endswith(b"secret")
        lp.shut()
        assert tcp_get(lp.port) == b""
    finally:
        srv.shutdown()
        lp.close()


def raw_unix_server(path, reply):
    """A forward's socket that records every byte it is sent and answers each connection with `reply`."""
    got, ls = [], socket.socket(socket.AF_UNIX)
    ls.bind(path)
    ls.listen(8)

    def serve():
        while True:
            try:
                c, _ = ls.accept()
            except OSError:
                return
            c.settimeout(2)
            try:
                d = c.recv(65536)
                got.append(d)
                if d:
                    c.sendall(reply)
            except OSError:
                got.append(b"")
            c.close()
    threading.Thread(target=serve, daemon=True).start()
    return ls, got


@pytest.mark.parametrize("reopened_and_opened", [False, True])
def test_a_relay_paused_across_a_reopen_never_reaches_the_new_forward(monkeypatch, reopened_and_opened):
    """F15: a relay passes the open check, then pauses inside connect(). Meanwhile the forward drops, shut() runs and a
    new forward listens on the socket (before its guard check, or checked and opened again after this relay's own
    check). The paused relay must not pass a byte to it: closed, nothing sent, nothing answered."""
    lp = tpu.free_port(18640)
    reached, go = threading.Event(), threading.Event()

    class Slow(socket.socket):
        def connect(self, addr):
            if self.family == socket.AF_UNIX and not go.is_set():
                reached.set()
                go.wait(5)                                       # the relay sits between its check and its connect
            return super().connect(addr)
    import types
    shim = types.SimpleNamespace(**{k: getattr(socket, k) for k in dir(socket) if not k.startswith("__")})
    shim.socket = Slow
    monkeypatch.setattr(tpu, "socket", shim)                     # only tpu's own sockets (the relay's) are slowed
    old, old_got = raw_unix_server(lp.sock, b"OLD")
    out = {}
    try:
        lp.open()
        t = threading.Thread(target=lambda: out.update(r=tcp_get(lp.port)))
        t.start()
        assert reached.wait(5)                                   # the relay passed the open check
        lp.shut()                                                # the forward dropped
        old.close()
        os.unlink(lp.sock)
        new, new_got = raw_unix_server(lp.sock, b"NEW")          # the reopened forward
        if reopened_and_opened:
            lp.open()                                            # checked and opened again, after the relay's check
        go.set()
        t.join(10)
        assert out["r"] == b"" and new_got in ([], [b""]) and old_got == []
        new.close()
    finally:
        go.set()
        lp.close()


def test_shut_cuts_a_relay_in_flight():
    """The forward behind a relay in flight is gone or not checked: shut() ends that relay at once."""
    lp = tpu.free_port(18660)
    ls = socket.socket(socket.AF_UNIX)
    ls.bind(lp.sock)
    ls.listen(4)
    try:
        lp.open()
        c = socket.create_connection(("127.0.0.1", lp.port), timeout=5)
        c.sendall(b"ping")
        s, _ = ls.accept()                                       # the relay reached the forward
        s.settimeout(5)
        assert s.recv(16) == b"ping"
        t0 = time.time()
        lp.shut()
        assert c.recv(16) == b"" and time.time() - t0 < 3        # the local side is closed by shut(), not by a timeout
        s.close()
        c.close()
    finally:
        ls.close()
        lp.close()


@pytest.mark.parametrize("guard", ["200", "404", "drop"])
def test_server_without_the_guard_never_gets_the_local_port(cloud, monkeypatch, capsys, guard):
    """The chip's server is checked through the private socket with another web site's Origin. Unless it refuses
    (403/401) the tunnel closes and the local port never relays, also while `up` waits for the server."""
    read = cloud(plain={Z[0]: "ok"}, guard=guard)
    ports, got, done = [], [], threading.Event()
    real = tpu.free_port

    def free_port(p):
        lp = real(p)
        ports.append(lp.port)
        return lp
    monkeypatch.setattr(tpu, "free_port", free_port)

    def page():                                            # a page in a browser calling the port the whole time
        while not done.is_set():
            if ports:
                r = tcp_get(ports[0])
                if r:
                    got.append(r)
            time.sleep(0.01)
    t = threading.Thread(target=page, daemon=True)
    t.start()
    try:
        rc = up()
    finally:
        done.set()
        t.join(5)
    assert rc == 2 and got == [] and not cloud.held
    assert "did not refuse a request from another web site" in capsys.readouterr().err
    assert read()["nodes"]                                  # the chip is not touched (still up, said so)


def test_guarded_server_gets_the_port(cloud, capsys):
    cloud(plain={Z[0]: "ok"})                                # guard 403 (the default): mashbox's request guard
    assert up() == 0 and cloud.held
    assert "Server ready" in capsys.readouterr().out


# ------------------------------------------------------------------ the wheel the chip installs (F12, F13; F8 in the CLI)
def pkg_sha(script):
    return re.search(r"^pkg='wheel sha256=([0-9a-f]{64})'$", script, re.M).group(1)


def test_chip_gets_the_bytes_that_were_hashed(cloud, tmp_path, monkeypatch):
    """The wheel is copied once into a private folder and hashed there; that copy is what scp sends, so a file that
    changes after the hash (a rebuild, an update) cannot put other bytes on the chip under the recorded hash."""
    w = tmp_path / "mashbox-0.1.0+gabc1234-py3-none-any.whl"
    w.write_bytes(b"wheel one")
    monkeypatch.setenv("MASHBOX_WHEEL", str(w))
    read = cloud(plain={Z[0]: "ok"})
    real = tpu.gc

    def gc(args, project=None, **kw):
        if args[3:4] == ["scp"]:
            w.write_bytes(b"wheel two")                     # the file changes between the hash and the copy
        return real(args, project, **kw)
    monkeypatch.setattr(tpu, "gc", gc)
    assert up() == 0
    S = read()
    sha = pkg_sha(script_of(S))
    assert S["copied_sha"][w.name] == sha == hashlib.sha256(b"wheel one").hexdigest()
    assert f'[ "$got" = {sha} ]' in script_of(S)          # the chip checks it before installing


def test_chip_refuses_a_wheel_whose_bytes_differ(cloud, tmp_path, monkeypatch, capsys):
    w = tmp_path / "mashbox-0.1.0+gabc1234-py3-none-any.whl"
    w.write_bytes(b"wheel one")
    monkeypatch.setenv("MASHBOX_WHEEL", str(w))
    read = cloud(plain={Z[0]: "ok"})
    real = tpu.gc

    def gc(args, project=None, **kw):
        r = real(args, project, **kw)
        if args[3:4] == ["scp"]:                            # other bytes arrive on the chip
            put(read, copied_sha={w.name: "0" * 64})
        return r
    monkeypatch.setattr(tpu, "gc", gc)
    assert up() == 2
    assert "the wheel on the chip is not the one sent from here" in capsys.readouterr().err
    assert "starts" not in read()                           # nothing installed, no server started


def test_url_wheel_is_downloaded_hashed_and_a_changed_one_restarts_the_server(cloud, tmp_path, monkeypatch, capsys):
    body = {"b": b"url wheel one"}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body["b"])))
            self.end_headers()
            self.wfile.write(body["b"])

        def log_message(self, *a):
            pass
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    name = "mashbox-0.1.0+gabc1234-py3-none-any.whl"
    monkeypatch.setenv("MASHBOX_WHEEL", f"http://127.0.0.1:{s.server_address[1]}/w/{name}")
    try:
        read = cloud(plain={Z[0]: "ok"})
        assert up() == 0
        S = read()
        assert S["copied"][-1] == name and pkg_sha(script_of(S)) == hashlib.sha256(b"url wheel one").hexdigest()
        body["b"] = b"url wheel two"                       # same URL, other bytes
        capsys.readouterr()
        assert up() == 0
        assert "came from another mashbox" in capsys.readouterr().out and read()["starts"] == 2
    finally:
        s.shutdown()


def test_a_wheel_that_is_not_there_stops_before_the_chip(cloud, tmp_path, monkeypatch, capsys):
    """Fail closed: a named wheel that is missing (the Mac app's copy failed) never becomes an install of something
    else on the chip."""
    monkeypatch.setenv("MASHBOX_WHEEL", str(tmp_path / "gone" / "mashbox-0.1.0+gabc1234-py3-none-any.whl"))
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 2
    assert "MASHBOX_WHEEL names no file on this machine" in capsys.readouterr().err
    S = read()
    assert not installs(S) and "copied" not in S


@pytest.mark.parametrize("bad", ["mashbox-0.1.0-py3-none-any$(touch pwned).whl", "mashbox 0.1.0-py3-none-any.whl",
                                 "mashbox-0.1.0-py3-none-any.whl;id", "notawheel.zip"])
def test_wheel_name_never_reaches_a_shell(cloud, tmp_path, monkeypatch, capsys, bad):
    w = tmp_path / bad
    w.write_bytes(b"x")
    monkeypatch.setenv("MASHBOX_WHEEL", str(w))
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 2 and "MASHBOX_WHEEL must be a wheel file" in capsys.readouterr().err
    S = read()
    assert not installs(S) and "copied" not in S


def test_wheel_path_is_quoted_for_the_chips_shell(cloud, tmp_path, monkeypatch):
    w = tmp_path / "mashbox-0.1.0+gabc1234-py3-none-any.whl"
    w.write_bytes(b"x")
    monkeypatch.setenv("MASHBOX_WHEEL", str(w))
    read = cloud(plain={Z[0]: "ok"})
    assert up() == 0
    assign = re.search(r"(?<=\s)MASHBOX_WHEEL=\S+", script_of(read())).group(0)
    out = subprocess.run(["sh", "-c", f'{assign}; printf %s "$MASHBOX_WHEEL"'], env={"HOME": "/home/u w"},
                         capture_output=True, text=True).stdout
    assert out == "/home/u w/" + w.name


# ------------------------------------------------------------------ an unclear create answer and a name taken meanwhile (F17)
def node_calls(S, verb, zone=None):
    return [c for c in S["calls"] if c[2:4] == ["tpu-vm", verb] and (zone is None or c[c.index("--zone") + 1] == zone)]


def test_unclear_create_never_deletes_a_chip_someone_else_made_meanwhile(cloud, capsys):
    """After the scan, someone else makes a chip of that name (no mashbox label) and this command's create ends in an
    error. The chip of that name is described, not deleted, not used, and up stops naming it and its zone (it does not
    go on to make a chip in the next zone either)."""
    read = cloud(plain={Z[0]: "raced-internal", Z[1]: "ok"})
    assert up() == 2
    S = read()
    assert not deletes(S) and not installs(S) and creates(S) == [Z[0]]
    assert list(S["nodes"]) == [f"{Z[0]}/mashbox-qwen3-8-27b"] and not S["nodes"][f"{Z[0]}/mashbox-qwen3-8-27b"]["labels"]
    assert f"a TPU named mashbox-qwen3-8-27b appeared in {Z[0]}" in capsys.readouterr().err


def test_unclear_create_with_the_sign_in_gone_deletes_nothing_it_cannot_check(cloud, capsys):
    """The create errors and the sign-in has expired, so the chip of that name cannot be read: nothing is deleted (it
    may be someone else's); the user is told to check our label before a raw delete."""
    read = cloud(plain={Z[0]: "raced-auth"})
    assert up() == 2
    S = read()
    assert not deletes(S) and list(S["nodes"]) == [f"{Z[0]}/mashbox-qwen3-8-27b"]
    assert node_calls(S, "describe", Z[0])
    err = capsys.readouterr().err
    assert "shows the label mashbox=tpu-up" in err and "it deletes the chip only if mashbox made it" in err


def test_unclear_queued_create_never_uses_a_request_someone_else_made(cloud, capsys):
    """A queued request of that name made by someone else (no label) in the first zone is granted first: it is never
    used (no install on its chip) and never deleted; this command's own request in the next zone is used."""
    read = cloud(plain={}, qr_create={Z[0]: "raced-internal"}, grant={Z[0]: 1, Z[1]: 3})
    assert up() == 0
    S = read()
    assert f"{Z[0]}/mashbox-qwen3-8-27b-q" in S["qrs"] and f"{Z[0]}/mashbox-qwen3-8-27b" in S["nodes"]
    assert not [d for d in deletes(S) if d[d.index("--zone") + 1] == Z[0]]
    assert installs(S) and all(c[c.index("--zone") + 1] == Z[1] for c in installs(S))
    assert f"{Z[0]}: the queued request mashbox-qwen3-8-27b-q there was not made by mashbox" in capsys.readouterr().out


def test_unclear_queued_create_never_force_deletes_a_request_someone_else_made(cloud, capsys):
    """When this command's request in another zone is granted, the others it may have made are deleted with --force
    (which deletes their chips too): a request of that name without our label is never one of them."""
    read = cloud(plain={}, qr_create={Z[0]: "raced-internal"}, grant={Z[1]: 1})
    assert up() == 0
    S = read()
    assert f"{Z[0]}/mashbox-qwen3-8-27b-q" in S["qrs"]
    assert not [d for d in deletes(S) if d[d.index("--zone") + 1] == Z[0]]


def test_a_queued_request_that_cannot_be_checked_is_named_not_deleted(cloud, capsys):
    read = cloud(plain={}, qr_create={Z[0]: "raced-auth"})
    assert up() == 2
    S = read()
    assert not deletes(S) and list(S["qrs"]) == [f"{Z[0]}/mashbox-qwen3-8-27b-q"]
    out = capsys.readouterr().out
    assert "could not be checked" in out and "If it carries the label mashbox=tpu-up, delete it: gcloud compute tpus " \
        f"queued-resources delete mashbox-qwen3-8-27b-q --zone {Z[0]} --project p1 --force --quiet" in out


# ------------------------------------------------------------------ a long TMPDIR (F20)
def a_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def listener_could_bind(port):
    """True when a new listener could bind the port (SO_REUSEADDR, as Port: old TIME_WAIT pairs do not count, a socket
    listening on it does)."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()


def test_a_long_tmpdir_still_gives_a_socket_path_ssh_can_bind(tmp_path, monkeypatch):
    long = tmp_path / ("d" * 60) / ("e" * 60)
    long.mkdir(parents=True)
    monkeypatch.setattr(tpu.tempfile, "tempdir", str(long))
    lp = tpu.free_port(a_free_port())
    try:
        assert len(os.fsencode(lp.sock)) <= tpu.SUN_MAX and not lp.dir.startswith(str(long))
        assert os.stat(lp.dir).st_mode & 0o777 == 0o700
        u = socket.socket(socket.AF_UNIX)
        u.bind(lp.sock)                                                   # what ssh does with -L SOCKET:...
        u.close()
    finally:
        lp.close()


def test_no_short_folder_stops_before_ssh_and_frees_the_port(tmp_path, monkeypatch):
    long = tmp_path / ("d" * 60) / ("e" * 60)
    long.mkdir(parents=True)
    monkeypatch.setattr(tpu.tempfile, "tempdir", str(long))
    monkeypatch.setattr(tpu, "SHORT_BASES", (str(long),))
    with socket.socket() as hold:   # p stays bound here (SO_REUSEADDR, never listening): no other process can take
        hold.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        # it before Port binds it, and Port still can
        hold.bind(("127.0.0.1", 0))
        p = hold.getsockname()[1]
        with pytest.raises(tpu.Stop, match="Set TMPDIR to a short folder"):
            tpu.Port(p)
        assert listener_could_bind(p)                                     # the port was given back


def test_ssh_saying_the_socket_path_is_too_long_stops_at_once(cloud, monkeypatch, capsys):
    """OpenSSH's own lines for a forward it cannot bind ("unix_listener: path ... too long", "Could not request local
    forwarding.") stop up with that message: one ssh -L, never a retry loop while the chip bills."""
    read = cloud(plain={Z[0]: "ok"}, tunnel_fail="long")
    monkeypatch.setattr(tpu, "READY_MIN", 0.1)                            # the old loop gives up after 6 s here
    assert up() == 2
    err = capsys.readouterr().err
    assert "could not open its forward" in err and "too long for Unix domain socket" in err
    assert len([c for c in read()["calls"] if "-L" in c]) == 1


def test_the_port_stays_bound_after_the_session_until_the_process_exits():
    """release() (the end of a session) cuts relays and removes the private socket, but nothing else can take the port
    while this process runs: the Mac app relies on it (a job still running after a connect held the port during it)."""
    lp = tpu.free_port(a_free_port())
    d = lp.dir
    lp.release()
    try:
        assert not os.path.exists(d) and not listener_could_bind(lp.port)
    finally:
        lp.close()
    assert listener_could_bind(lp.port)


# ------------------------------------------------------------------ a chip replaced after the list (F23)
SWAPPED = f"{Z[3]}/handed-1"


def listed_then_replaced(cloud, qr=False):
    read = cloud(plain={}, swap_after_list=[SWAPPED + ("-q" if qr else "")])
    S = read()
    S["nodes"] = {SWAPPED: ours(Z[3], "handed-1")}
    if qr:
        S["nodes"][SWAPPED]["queuedResource"] = f"projects/p1/locations/{Z[3]}/queuedResources/handed-1-q"
        S["qrs"] = {SWAPPED + "-q": {"name": f"projects/p1/locations/{Z[3]}/queuedResources/handed-1-q",
                                     "state": {"state": "ACTIVE"}, "describes": 0, "args": [],
                                     "tpu": {"nodeSpec": [{"nodeId": "handed-1", "node": {"labels": {"mashbox": "tpu-up"}}}]}}}
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    return read


def test_up_never_installs_on_a_chip_replaced_after_the_list(cloud, capsys):
    read = listed_then_replaced(cloud)
    assert up("--name", "handed-1") == 2
    S = read()
    assert not installs(S) and not deletes(S) and not [c for c in S["calls"] if "-L" in c]
    assert "not the one mashbox listed" in capsys.readouterr().err


@pytest.mark.parametrize("qr", [False, True])
def test_down_never_deletes_a_chip_replaced_after_the_list(cloud, capsys, qr):
    read = listed_then_replaced(cloud, qr)
    assert tpu.main(["down", "handed-1"]) != 0
    S = read()
    assert not deletes(S) and SWAPPED in S["nodes"]
    assert "NOT deleted" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["ssh", "tunnel"])
def test_ssh_and_tunnel_never_reach_a_chip_replaced_after_the_list(cloud, capsys, cmd):
    read = listed_then_replaced(cloud)
    assert tpu.main([cmd, "handed-1"]) == 2
    assert not [c for c in read()["calls"] if c[2:4] == ["tpu-vm", "ssh"]]
    assert "not the one mashbox listed" in capsys.readouterr().err


def test_down_of_a_chip_gone_since_the_list_says_gone(cloud, capsys):
    """Deleted by its max lifetime between the list and the delete: down says it is gone (exit 0), deletes nothing."""
    read = cloud(plain={})
    S = read()
    S["nodes"] = {SWAPPED: ours(Z[3], "handed-1")}
    S["vanish_after_list"] = [SWAPPED]
    open(os.environ["FAKE_GCLOUD_STATE"], "w").write(json.dumps(S))
    assert tpu.main(["down", "handed-1"]) == 0
    assert not deletes(read()) and "handed-1: gone" in capsys.readouterr().out


# ------------------------------------------------------------------ replaced later: every call by name describes first (F24, F25)
NAME = "mashbox-qwen3-8-27b"
KEY = f"{Z[0]}/{NAME}"


def kinds(S):
    """What each call did to a node by name (scp, install, probe, tail, tunnel, ssh), in order."""
    import fake_gcloud
    return [k for k in (fake_gcloud.call_kind(c)[1] for c in S["calls"]) if k]


def test_up_never_installs_on_a_chip_replaced_during_the_copy(cloud, capsys):
    """The copy takes a while: a chip of that name deleted and made again without our label meanwhile never gets the
    install (nor a probe, a tunnel or a delete)."""
    read = cloud(plain={Z[0]: "ok"}, swap_on={KEY: "scp"})
    assert up() == 2
    S = read()
    assert S["swapped"] == ["scp"] and kinds(S) == ["scp"] and not deletes(S)
    assert "not the one mashbox listed" in capsys.readouterr().err


def test_lifetime_probe_never_reaches_a_chip_replaced_during_the_install(cloud, capsys):
    read = cloud(plain={Z[0]: "ok"}, swap_on={KEY: "install"})
    assert up() == 2
    S = read()
    assert S["swapped"] == ["install"] and kinds(S) == ["scp", "install"] and not deletes(S)
    assert "not the one mashbox listed" in capsys.readouterr().err


def test_a_dropped_forward_is_never_reopened_to_a_chip_replaced_meanwhile(cloud, capsys):
    """wait_server's retry after a dropped forward describes the chip first: no second forward to the replacement."""
    read = cloud(plain={Z[0]: "ok"}, swap_on={KEY: "tunnel"}, tunnel_drop=1)
    assert up() == 2
    S = read()
    assert S["swapped"] == ["tunnel"] and kinds(S).count("tunnel") == 1
    assert "not the one mashbox listed" in capsys.readouterr().err


def test_the_log_look_never_reaches_a_chip_replaced_while_the_server_loads(cloud, capsys, monkeypatch):
    """A server still loading: once a minute up reads the chip's log by name. A replacement made meanwhile is never
    reached, and the forward to it is closed."""
    read = cloud(plain={Z[0]: "ok"}, swap_on={KEY: "tunnel"}, ready=False)
    clock, t0 = [1e9], time.time()

    def swapped():
        try:
            return bool(read().get("swapped"))
        except ValueError:                              # the fake is writing the file right now
            return False

    def later():                                        # 7 s per look at the clock, once the forward's ssh has run
        if swapped() or time.time() - t0 > 60:          # (it starts in its own process; under load that takes a while)
            clock[0] += 7
        return clock[0]
    monkeypatch.setattr(tpu, "now", later)
    procs, real = [], getattr(tpu, "forward", None)

    def rec(*a):
        p = real(*a)
        procs.append(p)
        return p
    monkeypatch.setattr(tpu, "forward", rec, raising=False)
    assert up() == 2
    S = read()
    assert S["swapped"] == ["tunnel"] and "tail" not in kinds(S)
    assert "not the one mashbox listed" in capsys.readouterr().err
    assert len(procs) == 1 and procs[0].wait(timeout=10) is not None      # the forward was closed


def held_tunnel(read, monkeypatch, until):
    """Run the real hold() on a dropped forward until `until(naps, lp)` says stop (KeyboardInterrupt, as Ctrl-C)."""
    dead = subprocess.Popen(["true"])
    dead.wait()
    lp = tpu.free_port(a_free_port())
    naps, procs, real_sleep = [], [], time.sleep
    for f in ("forward", "open_tunnel"):                # every forward started here is closed at the end
        real = getattr(tpu, f, None)
        if real:
            def rec(*a, real=real):
                p = real(*a)
                procs.append(p)
                return p
            monkeypatch.setattr(tpu, f, rec)

    class Clock:                                        # tpu's own time module only: threads other tests left
        def __getattr__(self, k):                       # running keep the real time.sleep
            return getattr(time, k)

        def sleep(self, s):
            naps.append(lp.ok.is_set())
            real_sleep(0.02)
            if until(naps, lp):
                raise KeyboardInterrupt
    monkeypatch.setattr(tpu, "time", Clock())
    try:
        return REAL_HOLD(dead, "p1", Z[0], NAME, tpu.model("qwen3-8-27b"), lp, None), naps
    finally:
        monkeypatch.setattr(tpu, "time", time)
        for p in procs:
            p.terminate()
            p.wait(timeout=10)
        lp.close()


def test_held_tunnel_is_never_reopened_to_a_chip_replaced_meanwhile(cloud, capsys, monkeypatch):
    """Hours later the forward drops and the name belongs to a READY chip without our label: hold() stops with the
    reason (non-zero for the CLI) and never opens a forward to it."""
    read = cloud(plain={})
    put(read, nodes={KEY: plain_node(Z[0], NAME)})
    with pytest.raises(tpu.Stop, match="not the one this tunnel was opened to"):
        held_tunnel(read, monkeypatch, lambda naps, lp: len(naps) > 50)
    assert "tunnel" not in kinds(read())


def test_held_tunnel_opens_no_forward_while_the_chip_cannot_be_read(cloud, capsys, monkeypatch):
    """The forward drops and the chip cannot be described (sign-in expired, network): no forward to a chip that was
    not checked; hold() describes it again every 10 s and reopens once it reads as ours."""
    read = cloud(plain={})
    put(read, nodes={KEY: ours(Z[0], NAME)}, expired=True)

    def until(naps, lp):
        if len(naps) == 3:
            put(read, expired=False)                    # readable again
        return lp.ok.is_set() or len(naps) > 300
    rc, naps = held_tunnel(read, monkeypatch, until)
    S = read()
    assert rc == 0 and naps[-1] is True                 # reopened and checked once readable
    assert kinds(S).count("tunnel") == 1 and S["calls"].index(next(c for c in S["calls"] if "-L" in c)) > 3
    assert capsys.readouterr().out.count("could not be checked") == 1
