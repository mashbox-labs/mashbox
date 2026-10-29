"""The MASHBOX app server (src/mashbox/ui/app): its pages and routes, the chat page for the server in use, and the
model catalog read at run time (a new model of a known kind shows up with no app update; a new kind does not)."""
import base64
import json
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mashbox.ui import app, shell

WS_KEY = base64.b64encode(b"the sample nonce").decode()     # RFC 6455's sample handshake key, built at run time


def serve(handler):
    s = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


class Model(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "qwen3-8-27b", "max_model_len": 4096}]} if self.path == "/v1/models" else {"ok": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    reg = shell.registry()
    reg["models"].append(dict(reg["models"][-1], id="new-chat-model", name="New Chat Model", kind="llm"))
    reg["models"].append(dict(reg["models"][-1], id="new-kind-model", name="New Kind", kind="voice"))
    (tmp_path / "catalog.json").write_text(json.dumps(reg))
    app._cat.clear()
    app._world.clear()
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    with urllib.request.urlopen(url + "/", timeout=10) as r:     # the window page carries this launch's token
        HDR["X-Mashbox-Token"] = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()
    HDR["world"] = f"http://127.0.0.1:{wsrv.server_address[1]}"
    yield url, tmp_path
    srv.shutdown()
    wsrv.shutdown()


def test_registry_opens_the_public_catalog(tmp_path, monkeypatch):
    """With no catalog set in Settings, registry() opens models.json on the main branch of the GitHub repo in models.json:
    the one catalog (the sync check reads the CATALOG_URL line; this checks the URL the app really opens)."""
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    gh, seen = shell.registry()["github"], []

    def offline(url, *a, **k):
        seen.append(getattr(url, "full_url", url))
        raise OSError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    app._cat.clear()
    app.registry()               # as the app calls it at start and on every page
    app._cat.clear()
    app.registry(refresh=True)   # as Settings' refresh calls it
    app._cat.clear()
    assert seen == [f"https://raw.githubusercontent.com/{gh['org']}/{gh['repo']}/main/models.json"] * 2


HDR: dict = {}


def get(url, data=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json", "X-Mashbox-Token": HDR["X-Mashbox-Token"],
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.geturl(), r.read()


def test_pages_and_routes(base):
    url, _ = base
    for p in ("/", "/window/frame.js", "/ui/app/app.css", "/ui/shell/shell.css", "/p/models", "/p/servers",
              "/p/agents", "/p/settings", "/p/world"):
        assert get(url + p)[0] == 200, p
    assert get(url + "/m/chat/")[1].endswith("/p/servers?need=chat")       # no server yet
    assert shell.check_page(app.own_file("index.html").decode()) == []
    assert get(url + "/ui/app/inapp.css")[0] == 200
    assert app.INAPP in get(url + "/p/models")[2]           # module pages shown in the window get inapp.css


def test_chat_through_the_server_in_use(base):
    url, _ = base
    s, model = serve(Model)
    try:
        r = json.loads(get(url + "/api/servers/add", {"name": "Test", "url": model})[2])
        assert r["probe"]["models"][0]["id"] == "qwen3-8-27b"
        code, _, page = get(url + "/m/chat/")
        assert code == 200 and b"Qwen3.8-27B" in page
        assert json.loads(get(url + "/v1/models")[2])["data"][0]["id"] == "qwen3-8-27b"
        assert get(url + "/m/chat/ui/chat/chat.js")[0] == 200
    finally:
        s.shutdown()


def test_catalog_from_url_new_model_no_update(base):
    url, tmp = base
    get(url + "/api/settings", {"catalog_url": (tmp / "catalog.json").as_uri()})
    c = json.loads(get(url + "/api/catalog?refresh=1")[2])
    rows = {m["id"]: m for m in c["models"]}
    assert c["source"]["from"] == "url"
    assert rows["new-chat-model"]["module"] == "chat"         # known kind: shows up, no app update
    assert rows["new-kind-model"]["module"] is None           # new kind: needs a newer app
    get(url + "/api/settings", {"catalog_url": (tmp / "missing.json").as_uri()})
    c = json.loads(get(url + "/api/catalog?refresh=1")[2])
    assert c["source"]["from"] == "package" and c["source"]["error"]


class World(BaseHTTPRequestHandler):
    """A world model server as the app sees it: a page at /, its own API, a WebSocket stream, no /v1 and no /health."""
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.end_headers()
            self.wfile.flush()
            while True:
                d = self.connection.recv(1024)
                if not d:
                    break
                self.connection.sendall(d)
            return
        if self.path in ("/", "/?mode=jpeg"):
            return self.reply(200, b"<html><head><title>w</title></head><body>world page</body></html>", "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self.reply(200, b'{"mode": "playing"}')
        self.reply(404, b'{"err": "not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.reply(200, self.rfile.read(n))


def test_world_module_through_the_app(base):
    import socket
    url, _ = base
    s, world = serve(World)
    try:
        add = json.loads(get(url + "/api/servers/add", {"name": "World", "url": world})[2])
        assert add["probe"]["ok"] and add["probe"]["models"] == []
        st = json.loads(get(url + "/api/state")[2])
        assert [x["module"] for x in st["servers"]] == ["world"] and st["active"] is None   # chat keeps its own server
        wurl = HDR["world"]                     # the world page's own origin: no /api, no token, no keys
        code, where, body = get(url + "/m/world")
        assert code == 200 and where == wurl + "/" and b"world page" in body and app.INAPP in body
        assert json.loads(get(url + "/m/world/api/state")[2]) == {"mode": "playing"}
        assert json.loads(get(wurl + "/api/action", {"keys": ["w"]}, {"Origin": wurl})[2]) == {"keys": ["w"]}
        port = int(wurl.rsplit(":", 1)[1])
        with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
            c.sendall(f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: {wurl}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {WS_KEY}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
            head = b""
            while b"\r\n\r\n" not in head:
                head += c.recv(1024)
            assert head.split(b"\r\n")[0].split()[1] == b"101"
            c.sendall(b"frame bytes")
            got = b""
            while len(got) < len(b"frame bytes"):
                got += c.recv(1024)
            assert got == b"frame bytes"
        assert get(url + "/m/chat/")[1].endswith("/p/servers?need=chat")
    finally:
        s.shutdown()


def test_delete_chip_drops_its_row_once_google_has_deleted_it(base, monkeypatch):
    """Deleting a queued request returns before Google deletes its chip, so `tpu down` can say STILL LISTED and exit 1
    (seen on a real v6e-1 delete from the app). The row goes once `tpu status` no longer lists the chip; a failed
    delete keeps it."""
    url, _ = base
    with app.LOCK:                                          # a world server first in the list never becomes the chat's
        st = app.state()
        st["servers"].append({"id": "w1", "name": "World", "url": "http://127.0.0.1:11", "module": "world"})
        app._save("state.json", st)
    app.add_tpu_server("qwen3-8-27b", "other-chip", "http://127.0.0.1:10")
    app.add_tpu_server("qwen3-8-27b", "desk-chip", "http://127.0.0.1:9")    # the chat's server in use
    status = iter(["desk-chip  us-east5-b  v6e-1  DELETING  60 min", "No mashbox chips or queued requests."])
    calls = []

    def fake(cmd, timeout=120):
        calls.append(cmd[cmd.index("tpu") + 1])
        if "down" in cmd:
            return {"code": 1, "out": "  us-east5-b: queued request desk-chip deleted\ndesk-chip: STILL LISTED after delete"}
        return {"code": 0, "out": next(status)}

    monkeypatch.setattr(app, "run_quick", fake)
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    r = json.loads(get(url + "/api/tpu/down", {"name": "desk-chip"})[2])
    assert r["code"] == 0 and "desk-chip: gone" in r["out"] and calls == ["down", "status", "status"]
    assert [x.get("chip") for x in app.state()["servers"]] == [None, "other-chip"]
    assert app.state()["active"] == next(x["id"] for x in app.state()["servers"] if x.get("chip") == "other-chip")

    monkeypatch.setattr(app, "run_quick", lambda cmd, timeout=120: {"code": 1, "out": "other-chip: NOT deleted: denied"})
    r = json.loads(get(url + "/api/tpu/down", {"name": "other-chip"})[2])
    assert r["code"] == 1 and [x.get("chip") for x in app.state()["servers"]] == [None, "other-chip"]
