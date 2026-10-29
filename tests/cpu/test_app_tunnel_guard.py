"""The servers the Mac app tunnels to: the chip runs the wheel this app was installed from, and a tunnel to a server
without the request guard (PyPI mashbox 0.1.0 had none) is closed before the app or a coding agent uses it."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mashbox.runtime.guard import Guard
from mashbox.ui import app


def serve(guarded, routes=None):
    """A test server: with the request guard or without it; `routes` = the paths it serves (others 404)."""
    g = Guard(hosts=["127.0.0.1"])

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            err = g.check(self.command, self.path, self.headers, auth=False) if guarded else None
            if err is None and routes is not None and self.path not in routes:
                err = (404, "not found")
            code, body = err if err else (200, "ok")
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


def test_guarded_tells_the_two_apart():
    (open_s, open_url), (g_s, g_url) = serve(False), serve(True)
    try:
        assert app.guarded(open_url) is False and app.guarded(g_url) is True
        open_s.shutdown()
        open_s.server_close()
        assert app.guarded(open_url, timeout=2) is False          # no answer: not used (fail closed)
    finally:
        g_s.shutdown()


def job_printing(url, chip="c1"):
    code = (f"import time; print('Qwen is running on {chip} (TPU v6e-1, z).'); "
            f"print('  Local URL     {url}/   (SSH tunnel)'); time.sleep(30)")
    return app.Job("up", "m1", [sys.executable, "-u", "-c", code, "tpu", "up"])   # view() names the command from "tpu"


def wait(cond, t=15):
    t0 = time.time()
    while not cond() and time.time() - t0 < t:
        time.sleep(0.05)
    return cond()


def test_tunnel_to_an_unguarded_server_is_closed_and_never_used(monkeypatch):
    added = []
    monkeypatch.setattr(app, "add_tpu_server", lambda *a: added.append(a))
    s, url = serve(False)
    try:
        j = job_printing(url)
        assert wait(lambda: not j.running())
        assert j.refused and not added and j.view()["url"] is None
        assert any("no request guard" in x for x in j.lines)
    finally:
        s.shutdown()


def test_tunnel_to_a_guarded_server_is_used(monkeypatch):
    added = []
    monkeypatch.setattr(app, "add_tpu_server", lambda *a: added.append(a))
    s, url = serve(True)
    try:
        j = job_printing(url)
        assert wait(lambda: added)
        assert added == [("m1", "c1", url)] and not j.refused and j.running()
        j.stop()
    finally:
        s.shutdown()


def test_tpu_up_gets_the_apps_own_wheel(tmp_path, monkeypatch):
    w = tmp_path / "mashbox-0.1.0+gabc1234-py3-none-any.whl"
    monkeypatch.delenv("MASHBOX_WHEEL", raising=False)
    monkeypatch.setenv("MASHBOX_APP_WHEEL", str(w))
    assert app.child_env()["MASHBOX_WHEEL"] == str(w)               # named but missing: passed on, so tpu up stops
    w.write_bytes(b"x")
    assert app.child_env()["MASHBOX_WHEEL"] == str(w)


def test_v1_answers_made_here_are_data(tmp_path, monkeypatch):
    """No server in use (503) and an unreachable server (502): the same nosniff + sandbox as proxied answers."""
    import re
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()

        def get():
            try:
                urllib.request.urlopen(urllib.request.Request(url + "/v1/models", headers={"X-Mashbox-Token": tok}), timeout=10)
            except urllib.error.HTTPError as e:
                return e.code, e.headers
        code, h = get()
        assert code == 503 and h["Content-Security-Policy"] == "sandbox" and h["X-Content-Type-Options"] == "nosniff"
        dead = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = dead.server_address[1]
        dead.server_close()                                          # a port nothing answers on
        (tmp_path / "state.json").write_text(json.dumps({"servers": [{"id": "s1", "name": "dead", "url": f"http://127.0.0.1:{port}"}], "active": "s1"}))
        code, h = get()
        assert code == 502 and h["Content-Security-Policy"] == "sandbox" and h["X-Content-Type-Options"] == "nosniff"
    finally:
        for s in (srv, wsrv):
            s.shutdown()


def test_a_tunnel_without_a_refusal_is_closed(monkeypatch):
    """No verdict is no approval: a 404 (a server without the route or the guard) and no answer close the tunnel."""
    added = []
    monkeypatch.setattr(app, "add_tpu_server", lambda *a: added.append(a))
    s, url = serve(False, routes=["/"])                              # no /health: 404 to the check
    dead = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    gone = f"http://127.0.0.1:{dead.server_address[1]}"
    dead.server_close()                                              # nothing answers there
    try:
        jobs = [job_printing(url, "c404"), job_printing(gone, "cdead")]
        assert wait(lambda: all(not j.running() for j in jobs), 30)
        assert all(j.refused and j.view()["url"] is None for j in jobs) and not added
    finally:
        s.shutdown()


def test_world_tunnel_is_checked_on_the_world_servers_route(monkeypatch):
    """The world model server has no /health; it is checked on its own route (/, its health route in the catalog),
    which a guarded world server refuses to another web site and an unguarded one serves."""
    monkeypatch.setattr(app, "registry", lambda *a, **k: ({"models": [{"id": "hy-nomas", "cloud": {"health": "/"}}]}, "test"))
    added = []
    monkeypatch.setattr(app, "add_tpu_server", lambda *a: added.append(a))
    (open_s, open_url), (g_s, g_url) = serve(False, routes=["/"]), serve(True, routes=["/"])
    try:
        bad = app.Job("up", "hy-nomas", [sys.executable, "-u", "-c",
                      f"import time; print('W is running on w1 (TPU v6e-4, z).'); print('  Local URL     {open_url}/'); time.sleep(30)", "tpu", "up"])
        good = app.Job("up", "hy-nomas", [sys.executable, "-u", "-c",
                       f"import time; print('W is running on w2 (TPU v6e-4, z).'); print('  Local URL     {g_url}/'); time.sleep(30)", "tpu", "up"])
        assert wait(lambda: not bad.running() and added)
        assert bad.refused and added == [("hy-nomas", "w2", g_url)] and not good.refused
        good.stop()
    finally:
        open_s.shutdown()
        g_s.shutdown()


@pytest.mark.parametrize("latest,version,newer", [
    ("0.1.1", "0.1.0", True), ("0.1.10", "0.1.9", True), ("0.1.9", "0.1.10", False), ("0.1.1", "0.1.1", False),
    ("0.1.1", "0.2.0", False), ("0.2.0", "0.2.0rc1", True), ("0.2.0rc1", "0.1.9", True), ("0.2.0.dev1", "0.2.0a1", False),
    ("0.1.1.post1", "0.1.1", True), ("0.1.1", "0.1.0+g3006b09", None), ("0.1.1", "0.2.0+g3006b09", None),
    (None, "0.1.0", None), ("garbage", "0.1.0", None)])
def test_newer_release_compares_versions(latest, version, newer):
    assert app.newer_release(latest, version) is newer


def test_the_mac_app_without_its_wheel_starts_no_tpu_up(tmp_path, monkeypatch):
    """Fail closed: run by the Mac app (MASHBOX_APP_VERSION) without the wheel it installed from, Start refuses with a
    clear error instead of letting `tpu up` put another mashbox on the chip."""
    import re
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    monkeypatch.setenv("MASHBOX_APP_VERSION", "0.2.0")
    monkeypatch.setenv("MASHBOX_APP_WHEEL", str(tmp_path / "runtime" / "mashbox-0.1.0+gabc1234-py3-none-any.whl"))
    started = []
    monkeypatch.setattr(app, "Job", lambda *a: started.append(a))
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()
        req = urllib.request.Request(url + "/api/tpu/up", data=json.dumps({"model": "qwen3-8-27b"}).encode(),
                                     headers={"X-Mashbox-Token": tok, "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 409 and "wheel is missing" in json.load(e.value)["error"] and not started
    finally:
        for s in (srv, wsrv):
            s.shutdown()


# ------------------------------------------------------------------ a saved TPU row is no tunnel (F16)
def counting_server():
    """Any server on a port a TPU row names (another tunnel, another program): it answers everything and counts."""
    hits = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b'{"data": [{"id": "other"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}", hits


def running_app(tmp_path, monkeypatch):
    import re
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    with urllib.request.urlopen(url + "/", timeout=10) as r:
        tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()

    def get(path):
        try:
            with urllib.request.urlopen(urllib.request.Request(url + path, headers={"X-Mashbox-Token": tok}), timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, None
    return (srv, wsrv), get


def tpu_rows(other_url):
    return [{"id": "tpu-c1", "from": "tpu", "name": "TPU c1", "url": other_url, "model": "qwen3-8-27b", "chip": "c1",
             "module": "chat"},
            {"id": "tpu-w1", "from": "tpu", "name": "TPU w1", "url": other_url, "model": "hy-nomas", "chip": "w1",
             "module": "world"}]


@pytest.mark.parametrize("when", ["saved by an earlier launch", "left after its job ended"])
def test_a_saved_tpu_row_without_its_job_is_never_used(tmp_path, monkeypatch, when):
    """A TPU row names a local port, which is a server of ours only while the job holding the tunnel runs and its guard
    check passed. Saved by an earlier launch, or left in state.json with no job, the row is not used: /v1 answers 503,
    nothing reaches the server now on that port, the Connect agent page gets no server, and the world page none."""
    other, other_url, hits = counting_server()
    keep = {"id": "s9", "name": "mine", "url": "http://127.0.0.1:9", "from": "url"}       # added by address: kept
    saved = {"servers": tpu_rows(other_url) + [keep], "active": "tpu-c1"}
    if when == "saved by an earlier launch":
        (tmp_path / "state.json").write_text(json.dumps(saved))
    servers, get = running_app(tmp_path, monkeypatch)
    try:
        if when == "saved by an earlier launch":
            st = json.loads((tmp_path / "state.json").read_text())
            assert st["servers"] == [keep] and st["active"] == "s9"            # dropped at launch, the rest kept
            (tmp_path / "state.json").write_text(json.dumps(saved))            # and even if written back
        else:
            (tmp_path / "state.json").write_text(json.dumps(saved))
        code, _ = get("/v1/models")
        assert code == 503
        code, st = get("/api/state")
        assert code == 200 and st["active"] is None and [x["id"] for x in st["servers"]] == ["s9"]
        code, chk = get("/api/check")
        assert chk == {"ok": False, "error": "no server"}
        code, chk = get("/api/check?id=tpu-c1")
        assert chk == {"ok": False, "error": "no server"}
        assert app.active_server() is None and app.world_server() is None and app.world_server(check=True) is None
        assert hits == []                                                    # nothing reached that port
    finally:
        other.shutdown()
        for s in servers:
            s.shutdown()


def test_a_tpu_row_is_used_while_its_job_runs_and_dropped_when_it_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    monkeypatch.setattr(app, "registry", lambda *a, **k: ({"models": [{"id": "m1", "kind": "llm"}]}, "test"))
    monkeypatch.setattr(app, "JOBS", {})
    s, url = serve(True)
    try:
        j = app.JOBS["m1"] = job_printing(url)
        assert wait(lambda: app.active_server() is not None)
        assert app.active_server()["url"] == url and j.running()
        j.stop()
        assert wait(lambda: not app.state()["servers"])                      # the row goes with its job
        assert app.active_server() is None and app.state()["active"] is None
    finally:
        s.shutdown()


# ------------------------------------------------------------------ a job that ends while the app connects (F18)
class EndingJob:
    """A checked tunnel job that exits while the app is connecting to its port (another program then has the port)."""
    def __init__(self, chip, url):
        self.ok, self.chip, self.url, self.alive, self.started = True, chip, url, True, time.time()

    def running(self):
        return self.alive


def ends_on_connect(monkeypatch, port, jobs):
    """Every connection the app makes to `port` ends the jobs first: the race of F18, made certain."""
    import socket
    real = socket.create_connection

    def create_connection(addr, *a, **k):
        if addr[1] == port:
            for j in jobs:
                j.alive = False
        return real(addr, *a, **k)
    monkeypatch.setattr(socket, "create_connection", create_connection)


def test_a_job_that_ends_while_the_app_connects_gets_nothing_sent(tmp_path, monkeypatch):
    """The row was live when it was chosen; the job ends and another program takes its port during the connect. Nothing
    is sent there: /v1 answers 503, /api/check says not running, a world file and the world stream get 503."""
    import socket
    other, other_url, hits = counting_server()
    jc, jw = EndingJob("c1", other_url), EndingJob("w1", other_url)
    monkeypatch.setattr(app, "JOBS", {"qwen3-8-27b": jc, "hy-nomas": jw})
    servers, get = running_app(tmp_path, monkeypatch)
    (tmp_path / "state.json").write_text(json.dumps({"servers": tpu_rows(other_url), "active": "tpu-c1"}))
    wport = servers[1].server_address[1]

    def world_file():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{wport}/x.js", timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def world_stream():
        s = socket.socket()
        s.connect(("127.0.0.1", wport))
        s.sendall(f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{wport}\r\nOrigin: http://127.0.0.1:{wport}\r\n"
                  "Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"  # gitleaks:allow
                  "Sec-WebSocket-Version: 13\r\n\r\n".encode())
        s.settimeout(10)
        head = s.recv(4096)
        s.close()
        return head.split(b" ", 2)[1]
    try:
        assert app.active_server()["id"] == "tpu-c1" and app.world_server()["id"] == "tpu-w1"   # live when chosen
        ends_on_connect(monkeypatch, other.server_address[1], [jc, jw])
        assert get("/v1/models")[0] == 503
        jc.alive = jw.alive = True
        code, chk = get("/api/check?id=tpu-c1")
        assert code == 200 and chk["ok"] is False and chk["error"].startswith("not running")
        jc.alive = jw.alive = True
        assert world_file() == 503
        jc.alive = jw.alive = True
        assert world_stream() == b"503"
        assert hits == []                                                  # not one request reached that port
    finally:
        other.shutdown()
        for x in servers:
            x.shutdown()


def test_a_job_that_ends_during_its_guard_check_is_never_used(monkeypatch, tmp_path):
    """The guard check reached the job's port only if the job still runs after it: one that ended during the check
    (its port may be another program's by then) is refused, and no row is added."""
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    monkeypatch.setattr(app, "JOBS", {})
    added = []
    monkeypatch.setattr(app, "add_tpu_server", lambda *a: added.append(a))
    holder = {}

    def guarded(url, path="/health", timeout=10):
        holder["job"].proc.wait(10)                                       # the job exits during the check
        return True
    monkeypatch.setattr(app, "guarded", guarded)
    code = ("import sys; print('Qwen is running on c1 (TPU v6e-1, z).'); "
            "print('  Local URL     http://127.0.0.1:9/   (SSH tunnel)'); sys.stdout.flush()")
    j = holder["job"] = app.Job("up", "m1", [sys.executable, "-u", "-c", code, "tpu", "up"])
    assert wait(lambda: j.checked and not j.running())
    time.sleep(0.2)
    assert j.refused and not j.ok and added == []


# ------------------------------------------------------------------ two Start clicks at once (F19)
def test_two_starts_at_once_make_one_job(tmp_path, monkeypatch):
    import re
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    monkeypatch.setattr(app, "JOBS", {})
    monkeypatch.setattr(app, "wheel_missing", lambda: None)
    made = []

    class SlowJob:
        def __init__(self, kind, model, cmd):
            made.append(model)
            time.sleep(0.5)                                               # starting a subprocess takes a moment
            self.kind, self.model, self.chip, self.url, self.ok = kind, model, None, None, False

        def running(self):
            return True

        def view(self):
            return {"kind": self.kind, "model": self.model, "running": True}

        def stop(self):
            pass
    monkeypatch.setattr(app, "Job", SlowJob)
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    codes = []
    try:
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()

        def start():
            req = urllib.request.Request(url + "/api/tpu/up", data=json.dumps({"model": "qwen3-8-27b"}).encode(),
                                         headers={"X-Mashbox-Token": tok, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    codes.append(r.status)
            except urllib.error.HTTPError as e:
                codes.append(e.code)
        ts = [threading.Thread(target=start) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert sorted(codes) == [200, 409] and made == ["qwen3-8-27b"] and len(app.JOBS) == 1
    finally:
        app.CLOSING.clear()
        for s in (srv, wsrv):
            s.shutdown()


# ------------------------------------------------------------------ Stop or Delete while a Start is being made (F22)
@pytest.mark.parametrize("call,body", [("/api/tpu/stop", {"model": "qwen3-8-27b"}), ("/api/tpu/down", {"name": "c-new-1"})])
def test_stop_or_delete_during_a_start_stops_that_start(tmp_path, monkeypatch, call, body):
    import re
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    monkeypatch.setattr(app, "JOBS", {})
    monkeypatch.setattr(app, "wheel_missing", lambda: None)
    monkeypatch.setattr(app, "run_quick", lambda cmd, timeout=120: {"code": 0, "out": "gone"})
    stopped = []

    class SlowJob:
        def __init__(self, kind, model, cmd):
            time.sleep(0.6)                                               # starting a subprocess takes a moment
            self.kind, self.model, self.cmd, self.chip, self.url, self.ok = kind, model, cmd, None, None, False

        def running(self):
            return not stopped

        def view(self):
            return {"kind": self.kind, "model": self.model, "running": self.running()}

        def stop(self):
            stopped.append(self.model)
    monkeypatch.setattr(app, "Job", SlowJob)
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()

        def post(path, b):
            req = urllib.request.Request(url + path, data=json.dumps(b).encode(),
                                         headers={"X-Mashbox-Token": tok, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status
        t = threading.Thread(target=post, args=("/api/tpu/up", {"model": "qwen3-8-27b", "name": "c-new-1"}))
        t.start()
        time.sleep(0.2)                                                   # the start is inside JOBS_LOCK now
        assert post(call, body) == 200
        t.join()
        assert stopped == ["qwen3-8-27b"]                                 # the start that was being made is stopped
    finally:
        for s in (srv, wsrv):
            s.shutdown()
