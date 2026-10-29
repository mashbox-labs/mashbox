"""MASHBOX app: the local server behind the MASHBOX desktop window (and any browser).

    python -m mashbox.ui.app                      # then open http://127.0.0.1:18930/

The window (window/index.html) holds the module sidebar and shows one page at a time. Every page it shows is a module
shell page from this package: the chat module (mashbox.ui.chat) for chat models, the world model's own page from its
server, and this module's screens (index.html with view.js): Models, Servers, Connect your agent, Settings.

Where models run: the user's own Cloud TPU through `mashbox tpu up | status | tunnel | down` (this package's CLI, run
as a child process), or any MASHBOX or OpenAI-compatible server URL. /v1/... and /health go to the server in use.

Models load at run time: the catalog is models.json from CATALOG_URL (cached for offline, else the copy in this
package), merged with what the server in use lists at /v1/models. A row's `kind` picks its module, so a new model of a
known kind shows up with no app update; a new kind needs a newer app.

State lives in the app's data folder (state.json; server keys in keys.json, mode 600).

Who may call it (mashbox.runtime.guard): every request must name this machine as Host and, from a browser, come from
this origin; a POST must be application/json. /api, /v1 and /health also need this launch's token in the X-Mashbox-Token
header; the app's own pages carry it (a page from another site can neither read it nor send that header without a CORS
preflight, which this server never approves). The world model server's page, files and stream are served from a second
origin, the next port (`--world-port`), which has no /api or /v1, so that page never runs where the token and the
stored keys are.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from ... import __version__
from ...runtime.guard import HEADER, Guard, new_token
from .. import chat, shell

CATALOG_URL = "https://raw.githubusercontent.com/mashbox-labs/mashbox/main/models.json"
MODULES = {"llm": "chat", "llm-tpu": "chat", "world": "world"}      # model kind -> module page
VIEWS = {"models": "Models", "servers": "Servers", "agents": "Connect your agent", "settings": "Settings",
         "world": "World model"}
TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml"}
BLOCK = shell.BLOCK
TPU_NAME = re.compile(r"[a-z]([-a-z0-9]{0,61}[a-z0-9])?")     # a Cloud TPU node name, which never starts with "-"
BAD_NAME = "A chip name is lowercase letters, digits and dashes, starts with a letter and has at most 63 characters."
PROXY_TYPES = ("application/json", "text/event-stream", "text/plain")   # what /v1 and /health may answer as
EXTRA_PATH = ["/opt/homebrew/bin", "/usr/local/bin", os.path.expanduser("~/google-cloud-sdk/bin"),
              os.path.expanduser("~/.local/bin")]


def data_dir() -> str:
    if os.environ.get("MASHBOX_APP_DATA"):
        d = os.environ["MASHBOX_APP_DATA"]
    elif sys.platform == "darwin":
        d = os.path.expanduser("~/Library/Application Support/MASHBOX")
    else:
        d = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "mashbox-app")
    os.makedirs(d, exist_ok=True)
    return d


def _path(name):
    return os.path.join(data_dir(), name)


def _load(name, default):
    try:
        with open(_path(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(name, obj, private=False):
    tmp = _path(name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, _path(name))


LOCK = threading.RLock()


def state() -> dict:
    s = _load("state.json", {})
    s.setdefault("servers", [])
    s.setdefault("active", None)
    s.setdefault("catalog_url", None)
    s.setdefault("project", None)
    return s


def keys() -> dict:
    return _load("keys.json", {})


def active_server(s=None):
    s = s or state()
    return next((x for x in s["servers"] if x["id"] == s["active"]), None)


_world: dict = {}


def world_server(check=False):
    """The world model server the World model module shows: the first world server in the list that answers (checked
    when the page opens; its page, files and stream then come through /m/world/ from this one)."""
    rows = [x for x in state()["servers"] if x.get("module") == "world"]
    if check:
        _world["srv"] = next((x for x in rows if probe(x["url"], keys().get(x["id"]), timeout=3)["ok"]), None)
    elif _world.get("srv") not in rows:
        _world["srv"] = rows[0] if rows else None
    return _world.get("srv")


_rel: dict = {}


def latest_release(timeout=4):
    """The newest mashbox on PyPI, checked at most once an hour (best effort; None when offline)."""
    if time.time() - _rel.get("t", 0) < 3600:
        return _rel.get("v")
    _rel["t"] = time.time()
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/mashbox/json", timeout=timeout) as r:
            _rel["v"] = json.load(r)["info"]["version"]
    except (OSError, ValueError, KeyError):
        _rel["v"] = None
    return _rel["v"]


def runtime_source():
    """What the app installed as its runtime: the stamp the host writes next to the wheel (release file or source build)."""
    try:
        with open(os.path.join(data_dir(), "runtime-wheel.txt")) as f:
            return f.read().strip() or None
    except OSError:
        return None


# ------------------------------------------------------------------------------------------------ talking to a server
def _conn(url, timeout):
    u = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


def probe(url, key=None, timeout=6) -> dict:
    """What a server answers: its models (/v1/models) and /health, with the time it took."""
    out = {"url": url, "ok": False, "models": [], "health": None, "error": None}
    t0 = time.time()
    hdr = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        c, base = _conn(url, timeout)
        c.request("GET", base + "/v1/models", headers=hdr)
        r = c.getresponse()
        body = r.read()
        if r.status == 200:
            data = json.loads(body).get("data") or []
            out["models"] = [{"id": m.get("id"), "context": m.get("max_model_len") or m.get("context_length")}
                             for m in data if m.get("id")]
            out["ok"] = True
        else:
            out["error"] = f"/v1/models answered HTTP {r.status}"
        c.close()
        c, base = _conn(url, timeout)
        c.request("GET", base + "/health", headers=hdr)
        r = c.getresponse()
        r.read()
        out["health"] = r.status
        if not out["ok"] and r.status in (200, 503):
            out["ok"], out["error"] = True, None          # a server with a page and no model list (a world model)
        c.close()
        if not out["ok"]:                                   # no /health either: a page at its address (the world model)
            c, base = _conn(url, timeout)
            c.request("GET", base + "/", headers=hdr)
            r = c.getresponse()
            r.read()
            if r.status == 200 and "html" in (r.getheader("Content-Type") or ""):
                out["ok"], out["error"], out["page"] = True, None, True
            c.close()
    except (OSError, ValueError) as e:
        out["error"] = out["error"] or f"not reachable: {e}"
    out["ms"] = round((time.time() - t0) * 1000)
    return out


# ---------------------------------------------------------------------------------------------------------- catalog
_cat: dict = {}


def _valid(reg):
    return isinstance(reg, dict) and isinstance(reg.get("models"), list) and all(
        isinstance(m, dict) and m.get("id") and m.get("kind") for m in reg["models"])


def registry(refresh=False) -> tuple[dict, dict]:
    """(catalog, where it came from). The URL first (cached 10 minutes), else the last good copy, else this package's."""
    s = state()
    url = s.get("catalog_url") or CATALOG_URL
    with LOCK:
        if not refresh and _cat.get("url") == url and time.time() - _cat.get("at", 0) < 600:
            return _cat["reg"], _cat["src"]
        pkg = shell.registry()
        reg, src = None, None
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                got = json.loads(r.read())
            if not _valid(got):
                raise ValueError("not a MASHBOX models.json (needs a models list with id and kind)")
            reg = got
            src = {"from": "url", "url": url, "at": time.time()}
            _save("catalog-cache.json", {"url": url, "at": src["at"], "reg": got})
        except (OSError, ValueError) as e:
            err = str(e)
            cache = _load("catalog-cache.json", None)
            if cache and cache.get("url") == url and _valid(cache.get("reg")):
                reg, src = cache["reg"], {"from": "cache", "url": url, "at": cache.get("at"), "error": err}
            else:
                reg, src = pkg, {"from": "package", "url": url, "at": None, "error": err}
        for k in ("github", "hf", "version", "license", "schema"):    # what the shell needs for its links
            reg.setdefault(k, pkg.get(k))
        _cat.update(url=url, at=time.time(), reg=reg, src=src)
        return reg, src


def catalog(refresh=False) -> dict:
    reg, src = registry(refresh)
    srv = active_server()
    served = []
    if srv:
        p = probe(srv["url"], keys().get(srv["id"]), timeout=4)
        served = [m["id"] for m in p["models"]]
    rows = []
    for m in reg["models"]:
        rows.append({"id": m["id"], "name": m.get("name", m["id"]), "kind": m["kind"], "module": MODULES.get(m["kind"]),
                     "summary": m.get("summary", ""), "launch": m.get("launch"),
                     "maker": (m.get("upstream") or {}).get("maker"),
                     "chips": [f"TPU {c}" for c, v in (m.get("chips") or {}).items()
                               if isinstance(v, dict) and v.get("status") == "passing"],
                     "accelerator": (m.get("cloud") or {}).get("accelerator"),
                     "tpu": bool(m.get("cloud")), "agents": bool((m.get("cloud") or {}).get("agents")),
                     "served": m["id"] in served, "has_chat_settings": bool(m.get("chat"))})
    known = {r["id"] for r in rows}
    extra = [{"id": i, "name": i, "kind": "llm", "module": "chat", "summary": "Listed by the server in use.",
              "served": True, "from_server": True, "chips": [], "tpu": False} for i in served if i not in known]
    return {"source": src, "models": rows + extra, "modules": sorted(set(MODULES.values()))}


# ------------------------------------------------------------------------------------------------------ the TPU CLI
def child_env():
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    parts = env.get("PATH", "").split(os.pathsep)
    env["PATH"] = os.pathsep.join(parts + [p for p in EXTRA_PATH if p not in parts])
    return env


def tpu_cmd(*args, project=None):
    cmd = [sys.executable, "-u", "-m", "mashbox", "tpu", *args]
    if project:
        cmd += ["--project", project]
    return cmd


class Job:
    """One long `mashbox tpu up` or `tunnel` run: its output, the local URL it opens, the chip it names."""

    def __init__(self, kind, model, cmd):
        self.kind, self.model, self.cmd = kind, model, cmd
        self.lines: list[str] = []
        self.url = self.chip = None
        self.started = time.time()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     text=True, bufsize=1, env=child_env(), start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.lines = (self.lines + [line])[-400:]
            m = re.search(r"Local URL\s+(http://127\.0\.0\.1:\d+)", line)
            if m:
                self.url = m.group(1)
            m = re.search(r" is running on (\S+) \(TPU", line)
            if m:
                self.chip = m.group(1)
            if self.url and self.chip and self.model:
                add_tpu_server(self.model, self.chip, self.url)
            if self.chip and line.startswith(f"{self.chip} is gone"):    # deleted elsewhere or by its max lifetime
                forget_chip(self.chip)
        self.proc.wait()

    def running(self):
        return self.proc.poll() is None

    def stop(self):
        if self.running():
            try:
                os.killpg(self.proc.pid, signal.SIGINT)        # what Ctrl-C does: the CLI closes the tunnel cleanly
                self.proc.wait(10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()

    def view(self):
        return {"kind": self.kind, "model": self.model, "running": self.running(), "code": self.proc.poll(),
                "url": self.url, "chip": self.chip, "lines": self.lines[-120:], "started": self.started,
                "cmd": " ".join(["mashbox"] + self.cmd[self.cmd.index("tpu"):])}


JOBS: dict = {}          # one `mashbox tpu` run per model: a chat model and a world model can each hold a tunnel


def job_for(model=None, chip=None):
    js = list(JOBS.values())
    if chip:
        return next((j for j in js if j.chip == chip), None)
    if model:
        return JOBS.get(model)
    return max(js, key=lambda j: j.started) if js else None


def add_tpu_server(model, chip, url):
    with LOCK:
        s = state()
        sid = "tpu-" + chip
        row = next((x for x in s["servers"] if x["id"] == sid), None)
        if row is None:
            row = {"id": sid, "from": "tpu"}
            s["servers"].append(row)
        reg, _ = registry()
        kind = next((m["kind"] for m in reg["models"] if m["id"] == model), "llm")
        module = MODULES.get(kind, "chat")
        row.update(name=f"TPU {chip}", url=url, model=model, chip=chip, module=module)
        if module == "chat":                    # the server in use is the chat's; a world server is the World model's
            s["active"] = sid
        _save("state.json", s)


def forget_chip(name):
    """Drop the server rows of a chip that is gone; the server in use moves to the first row left."""
    with LOCK:
        s = state()
        s["servers"] = [x for x in s["servers"] if x.get("chip") != name]
        if not any(x["id"] == s["active"] for x in s["servers"]):
            s["active"] = next_active(s)
        _save("state.json", s)


def next_active(s):
    """The chat's next server in use after one goes: the first row that is not a world server, else none."""
    return next((x["id"] for x in s["servers"] if x.get("module") != "world"), None)


def chip_listed(name, project):
    """How `mashbox tpu status` lists the chip now: "" when it is not listed, "DELETING" while Google deletes it,
    "LISTED" for any other state, None when the check itself failed."""
    r = run_quick(tpu_cmd("status", project=project), 120)
    if r["code"] != 0:
        return None
    for line in r["out"].splitlines():
        words = line.split()
        if words[:1] == [name]:
            return "DELETING" if "DELETING" in words else "LISTED"
    return ""


def run_quick(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=child_env(), stdin=subprocess.DEVNULL)
        return {"code": r.returncode, "out": (r.stdout + r.stderr).strip()}
    except subprocess.TimeoutExpired:
        return {"code": -1, "out": f"no answer within {timeout} s"}
    except OSError as e:
        return {"code": -1, "out": str(e)}


def gcloud_info():
    g = shutil.which("gcloud", path=child_env()["PATH"])
    if not g:
        return {"gcloud": None}
    acc = run_quick([g, "config", "get-value", "account"], 30)["out"].splitlines()
    prj = run_quick([g, "config", "get-value", "project"], 30)["out"].splitlines()
    pick = lambda xs: next((x.strip() for x in xs if x.strip() and not x.startswith(("(", "WARNING"))), None)
    return {"gcloud": g, "account": pick(acc), "project": pick(prj)}


# ------------------------------------------------------------------------------------------------------------ pages
def own_file(rel):
    f = resources.files(__package__)
    for p in rel.split("/"):
        if p in ("", ".", "..") or p.startswith("."):
            return None
        f = f.joinpath(p)
    return f.read_bytes() if f.is_file() else None


def asset(path):
    """Any .../ui/shell/..., .../ui/chat/... or .../ui/app/... file, wherever the page that asks for it sits."""
    i = path.find("ui/")
    if i < 0:
        return None
    rel = path[i:]
    if rel.startswith("ui/app/"):
        sub = rel[len("ui/app/"):]
        ext = "." + sub.rsplit(".", 1)[-1] if "." in sub else ""
        body = own_file(sub) if ext in TYPES else None
        return (body, TYPES[ext]) if body is not None else None
    return shell.asset(rel) or chat.asset(rel)


def view_page(view, reg, token=None):
    raw = own_file("index.html")
    own = shell.page_config(raw) or {}
    own["module"] = VIEWS[view]
    raw = BLOCK.sub(lambda m: m.group(1) + json.dumps(own).encode() + m.group(3), raw, count=1)
    return in_app(shell.render(raw, reg), token)


INAPP = b'<link rel="stylesheet" href="/ui/app/inapp.css">'


def token_meta(token):
    """The launch token as the app's own pages carry it (read by frame.js, view.js and chat.js for their requests)."""
    return f'<meta name="mbx-token" content="{token}">'.encode() if token else b""


def in_app(html, token=None):
    """The page as the window shows it: the module page itself plus inapp.css (no web top bar or site links,
    the sidebar has them) and, on the app's own pages only, the launch token. The page files are never changed."""
    if isinstance(html, str):
        html = html.encode()
    i = html.lower().rfind(b"</head>")
    return html[:i] + INAPP + token_meta(token) + html[i:] if i >= 0 else html


def chat_page(reg, token=None):
    srv = active_server()
    model = srv.get("model") if srv else None
    if srv:
        p = probe(srv["url"], keys().get(srv["id"]), timeout=4)
        if p["models"]:
            model = p["models"][0]["id"]
    try:
        return in_app(chat.page(model, reg), token)
    except (ValueError, KeyError):          # a catalog row this package cannot read: the page still works, plainer
        return in_app(chat.page(model, dict(reg, models=[m for m in reg["models"] if m["id"] != model])), token)


# ----------------------------------------------------------------------------------------------------------- server
PAGE_HDR = {"Content-Security-Policy": "frame-ancestors 'self'", "X-Frame-Options": "SAMEORIGIN"}   # no framing by other sites
WS_HEADERS = ("upgrade", "connection", "sec-websocket-key", "sec-websocket-version", "sec-websocket-protocol",
              "sec-websocket-extensions", "user-agent", "origin", HEADER.lower())      # what a world stream handshake forwards
DROP = ("connection", "transfer-encoding", "keep-alive", "content-length", "set-cookie")  # never relayed from a server


def _head(html, add):
    i = html.lower().rfind(b"</head>")
    return html[:i] + add + html[i:] if i >= 0 else html


class _Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "MASHBOX"
    guard: Guard
    ports: dict

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send(self, code, body, ctype="application/json", extra=None):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def go(self, where):
        self.send(302, b"", "text/plain", {"Location": where})

    def refused(self, auth):
        """Answers and returns True when the guard turns this request away (wrong Host or Origin, a POST that is not
        JSON, or, with auth, no launch token)."""
        err = self.guard.check(self.command, self.path, self.headers, auth=auth)
        if err:
            self.close_connection = True
            self.send(err[0], {"error": {"message": err[1]}})
        return bool(err)

    def origin_of(self, which):
        """This machine's address as the browser named it in Host, on the app's or the world page's port."""
        h = self.headers.get("Host") or "127.0.0.1"
        name = h[: h.index("]") + 1] if h.startswith("[") and "]" in h else h.split(":")[0]
        return f"http://{name}:{self.ports[which]}"

    # ---- the world model server's own page, files and stream, shown from the world origin (never the app's)
    def world(self, srv, method, rest, page=False):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else None
        c, base = _conn(srv["url"], 60)
        try:
            hdr = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept", "range", "origin", HEADER.lower())}
            key = keys().get(srv["id"])
            if key:
                hdr["Authorization"] = f"Bearer {key}"
            c.request(method, base + rest, body=data, headers=hdr)
            r = c.getresponse()
            ctype = r.getheader("Content-Type") or "application/octet-stream"
            if page and r.status == 200 and "html" in ctype:
                return self.send(200, in_app(r.read()), ctype)
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in DROP:
                    self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as e:
            try:
                self.send(502, {"error": f"{srv['name']} is not reachable: {e}"})
            except OSError:
                pass
        finally:
            c.close()

    def world_ws(self, srv, rest):
        """The page's video and key stream: the handshake goes to the world server with the browser's own Origin and
        only the WebSocket headers (never its cookies), then bytes both ways."""
        u = urllib.parse.urlsplit(srv["url"])
        host = u.hostname + (f":{u.port}" if u.port else "")
        try:
            up = socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout=10)
            if u.scheme == "https":
                up = ssl.create_default_context().wrap_socket(up, server_hostname=u.hostname)
        except OSError as e:
            return self.send(502, {"error": f"{srv['name']} is not reachable: {e}"})
        lines = [f"GET {u.path.rstrip('/')}{rest} HTTP/1.1", f"Host: {host}"]
        lines += [f"{k}: {v}" for k, v in self.headers.items() if k.lower() in WS_HEADERS]
        key = keys().get(srv["id"])
        if key:
            lines.append(f"Authorization: Bearer {key}")
        up.settimeout(None)
        me = self.connection
        for x in (up, me):
            try:
                x.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        def pump(a, b):
            try:
                while True:
                    d = a.recv(65536)
                    if not d:
                        break
                    b.sendall(d)
            except OSError:
                pass
            for x in (a, b):
                try:
                    x.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        try:
            up.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            t = threading.Thread(target=pump, args=(up, me), daemon=True)
            t.start()
            pump(me, up)
            t.join(5)
        finally:
            up.close()
            self.close_connection = True


def make_handler(token=None, ports=None, host="127.0.0.1"):
    """The app's own origin: the window, the module pages, /api, and /v1 + /health for the server in use."""
    token = token or new_token()

    class H(_Base):
        guard = Guard(hosts=[host], token=token)

        def page(self, html):
            self.send(200, html, TYPES[".html"], PAGE_HDR)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                return {}

        # ---- the server in use, as it is (its answers as data only: never a page that runs here)
        def proxy(self, method):
            srv = active_server()
            if not srv:
                return self.send(503, {"error": {"message": "No server in use. Add one or start a TPU in Servers."}})
            n = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(n) if n else None
            c, base = _conn(srv["url"], 600)
            try:
                hdr = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept")}
                key = keys().get(srv["id"])
                if key:
                    hdr["Authorization"] = f"Bearer {key}"
                c.request(method, base + self.path, body=data, headers=hdr)
                r = c.getresponse()
                ctype = r.getheader("Content-Type") or ""
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in DROP + ("content-type",):
                        self.send_header(k, v)
                self.send_header("Content-Type", ctype if ctype.split(";")[0].strip().lower() in PROXY_TYPES
                                 else "application/octet-stream")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "sandbox")
                self.end_headers()
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError as e:
                try:
                    self.send(502, {"error": {"message": f"{srv['name']} is not reachable: {e}"}})
                except OSError:
                    pass
            finally:
                c.close()

        def do_GET(self):
            p = urllib.parse.urlsplit(self.path)
            path, q = p.path, urllib.parse.parse_qs(p.query)
            if self.refused(auth=path.startswith(("/v1/", "/api/")) or path == "/health"):
                return
            if (self.headers.get("Upgrade") or "").lower() == "websocket":
                return self.send(404, {"error": {"message": "no stream here"}})
            if path.startswith("/v1/") or path == "/health":
                return self.proxy("GET")
            if path in ("/", "/index.html"):
                return self.page(_head(own_file("window/index.html"), token_meta(token)))
            if path.startswith("/window/"):
                body = own_file(path[1:])
                ext = "." + path.rsplit(".", 1)[-1]
                return self.send(200, body, TYPES[ext]) if body and ext in TYPES else self.send(404, b"", "text/plain")
            if path.startswith("/api/"):
                return self.api_get(path, q)
            if path == "/m/world" or path.startswith("/m/world/"):      # the world page lives on its own origin
                if not world_server():
                    return self.go("/p/world")
                return self.go(self.origin_of("world") + (path[len("/m/world"):] or "/") + ("?" + p.query if p.query else ""))
            reg, _ = registry()
            if path in ("/m/chat", "/m/chat/"):
                if not active_server():
                    return self.go("/p/servers?need=chat")
                return self.page(chat_page(reg, token))
            m = re.fullmatch(r"/p/(\w+)/?", path)
            if m and m.group(1) in VIEWS:
                return self.page(view_page(m.group(1), reg, token))
            hit = asset(path)
            if hit:
                return self.send(200, hit[0], hit[1], {"Cache-Control": "no-cache"})
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if self.refused(auth=True):
                return
            if path.startswith("/v1/"):
                return self.proxy("POST")
            if path.startswith("/api/"):
                return self.api_post(path, self.body())
            self.send(404, b"", "text/plain")

        # ---- app API
        def api_get(self, path, q):
            if path == "/api/state":
                s = state()
                k = keys()
                job = job_for()
                return self.send(200, {
                    "version": __version__, "app_version": os.environ.get("MASHBOX_APP_VERSION"),
                    "data_dir": data_dir(), "catalog_url": s["catalog_url"], "catalog_default": CATALOG_URL,
                    "project": s["project"], "active": s["active"],
                    "servers": [dict(x, has_key=x["id"] in k) for x in s["servers"]],
                    "job": job.view() if job else None})
            if path == "/api/catalog":
                return self.send(200, catalog(refresh="refresh" in q))
            if path == "/api/check":
                sid = (q.get("id") or [None])[0]
                srv = next((x for x in state()["servers"] if x["id"] == sid), None) or active_server()
                if not srv:
                    return self.send(200, {"ok": False, "error": "no server"})
                return self.send(200, probe(srv["url"], keys().get(srv["id"])))
            if path == "/api/gcloud":
                return self.send(200, gcloud_info())
            if path == "/api/tpu/status":
                return self.send(200, run_quick(tpu_cmd("status", project=state()["project"]), 120))
            if path == "/api/release":
                src = runtime_source()
                return self.send(200, {"version": __version__, "source": src, "latest": latest_release(),
                                       "release_file": bool(src and " release " in f" {src} ")})
            if path == "/api/tpu/job":
                job = job_for((q.get("model") or [None])[0])
                return self.send(200, job.view() if job else None)
            self.send(404, {"error": "no such call"})

        def api_post(self, path, b):
            if path == "/api/servers/check":
                return self.send(200, probe(str(b.get("url", "")).rstrip("/"), b.get("key") or None))
            if path == "/api/servers/add":
                url = str(b.get("url", "")).strip().rstrip("/")
                u = urllib.parse.urlsplit(url)
                if u.scheme not in ("http", "https") or not u.hostname:
                    return self.send(400, {"error": "The address needs http:// or https:// and a host."})
                for suffix in ("/v1",):
                    url = url[: -len(suffix)] if url.endswith(suffix) else url
                p = probe(url, b.get("key") or None)
                with LOCK:
                    s = state()
                    sid = "srv-" + uuid.uuid4().hex[:8]
                    module = b.get("module") if b.get("module") in ("chat", "world") else (
                        "chat" if p["models"] else "world" if p["ok"] else "chat")
                    s["servers"].append({"id": sid, "name": b.get("name") or u.hostname, "url": url, "from": "url",
                                         "module": module, "model": p["models"][0]["id"] if p["models"] else None})
                    s["active"] = sid if module == "chat" else s["active"]
                    _save("state.json", s)
                    if b.get("key"):
                        k = keys()
                        k[sid] = b["key"]
                        _save("keys.json", k, private=True)
                return self.send(200, {"id": sid, "probe": p})
            if path == "/api/servers/use":
                with LOCK:
                    s = state()
                    if any(x["id"] == b.get("id") for x in s["servers"]):
                        s["active"] = b["id"]
                        _save("state.json", s)
                return self.send(200, {"active": state()["active"]})
            if path == "/api/servers/remove":
                with LOCK:
                    s = state()
                    s["servers"] = [x for x in s["servers"] if x["id"] != b.get("id")]
                    if s["active"] == b.get("id"):
                        s["active"] = next_active(s)
                    _save("state.json", s)
                    k = keys()
                    if k.pop(b.get("id"), None) is not None:
                        _save("keys.json", k, private=True)
                return self.send(200, {"ok": True})
            if path == "/api/settings":
                with LOCK:
                    s = state()
                    for key in ("catalog_url", "project"):
                        if key in b:
                            s[key] = (str(b[key]).strip() or None) if b[key] is not None else None
                    _save("state.json", s)
                _cat.clear()
                return self.send(200, {"ok": True})
            if path == "/api/open":
                url = str(b.get("url", ""))
                if re.match(r"https?://", url) and sys.platform == "darwin":
                    subprocess.Popen(["/usr/bin/open", url])
                return self.send(200, {"ok": bool(re.match(r"https?://", url))})
            if path == "/api/gcloud/login":
                g = gcloud_info().get("gcloud")
                if not g:
                    return self.send(200, {"ok": False, "error": "gcloud is not installed"})
                subprocess.Popen([g, "auth", "login", "--brief"], env=child_env(), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return self.send(200, {"ok": True})
            if path in ("/api/tpu/up", "/api/tpu/tunnel"):
                prj = state()["project"]
                name = str(b.get("name") or "")
                model = str(b.get("model") or "")
                if name and not TPU_NAME.fullmatch(name):
                    return self.send(400, {"error": BAD_NAME})
                if path == "/api/tpu/tunnel":
                    model = next((x.get("model") for x in state()["servers"] if x.get("chip") == name), None) or model
                job = JOBS.get(model)
                if job and job.running():
                    return self.send(409, {"error": "A TPU run for this model is already going. Stop it first."})
                if path == "/api/tpu/up":
                    row = next((m for m in registry()[0]["models"] if m["id"] == model and m.get("cloud")), None)
                    if row is None or not re.fullmatch(r"[a-z0-9][-a-z0-9._]*", model):
                        return self.send(400, {"error": "Pick a model from the list."})
                    zone = str(b.get("zone") or "")
                    if zone and zone not in (row["cloud"].get("zones") or []):
                        return self.send(400, {"error": "Pick one of the model's zones: " + ", ".join(row["cloud"].get("zones") or [])})
                    try:                                # empty means the default 2 h; 0 (no time limit) is the CLI's call only
                        hours = 2.0 if b.get("max_hours") in (None, "") else float(b["max_hours"])
                    except (TypeError, ValueError):
                        hours = float("nan")
                    if not (0 < hours < float("inf")):
                        return self.send(400, {"error": "Deletes itself after: a number of hours above 0."})
                    args = ["up", model, "--yes", f"--max-hours={hours}"]
                    if zone:
                        args.append(f"--zone={zone}")
                    if name:
                        args.append(f"--name={name}")
                    if b.get("spot"):
                        args.append("--spot")
                    if b.get("agents"):
                        args.append("--agents")
                    job = Job("up", model, tpu_cmd(*args, project=prj))
                else:
                    job = Job("tunnel", model, tpu_cmd("tunnel", *([name] if name else []), project=prj))
                JOBS[model] = job
                return self.send(200, job.view())
            if path == "/api/tpu/stop":
                job = job_for(b.get("model"))
                if job:
                    job.stop()
                return self.send(200, job.view() if job else None)
            if path == "/api/tpu/down":
                name = str(b.get("name") or "")
                if name and not TPU_NAME.fullmatch(name):
                    return self.send(400, {"error": BAD_NAME})
                job = job_for(chip=name) if name else job_for(b.get("model"))
                name = name or (job.chip if job else "")
                if job and job.chip == name:
                    job.stop()
                prj = state()["project"]
                r = run_quick(tpu_cmd("down", *([name] if name else []), project=prj), 300)
                gone = r["code"] == 0
                if name and not gone and "STILL LISTED" in r["out"]:
                    # Deleting a queued request returns before Google has deleted its chip (about a minute later), so
                    # the CLI's check right after can still list it. Ask again until it is gone, at most 3 minutes.
                    t0, left = time.time(), "LISTED"
                    while time.time() - t0 < 180:
                        time.sleep(10)
                        left = chip_listed(name, prj)
                        if left == "":
                            break
                    if left == "":
                        r = {"code": 0, "out": r["out"] + f"\n{name}: gone ({time.time() - t0:.0f} s later). Nothing of it bills now."}
                    elif left == "DELETING":
                        r = {"code": 0, "out": r["out"] + f"\n{name}: Google is still deleting it. Press Status to see it leave the list."}
                    gone = left in ("", "DELETING")
                if gone and name:
                    forget_chip(name)
                return self.send(200, r)
            self.send(404, {"error": "no such call"})

    H.ports = ports if ports is not None else {}
    return H


def make_world_handler(ports, host="127.0.0.1"):
    """The world origin: the world model server's page, files, API and stream, and nothing of the app's. Its page is
    another party's code, so it runs here, where the app's token, keys and /api are out of its reach."""

    class WH(_Base):
        guard = Guard(hosts=[host])

        def do_GET(self):
            if self.refused(auth=False):
                return
            p = urllib.parse.urlsplit(self.path)
            if p.path == "/ui/app/inapp.css":                       # the one file the app adds to the world page
                return self.send(200, own_file("inapp.css"), TYPES[".css"], {"Cache-Control": "no-cache"})
            ws = (self.headers.get("Upgrade") or "").lower() == "websocket"
            srv = world_server(check=p.path == "/" and not ws)
            if not srv:
                if p.path == "/" and not ws:
                    return self.go(self.origin_of("app") + "/p/world")
                return self.send(503, {"error": "No world model server. Add one or start one in Servers."})
            if ws:
                return self.world_ws(srv, self.path)
            return self.world(srv, "GET", self.path, page=p.path == "/")

        def do_POST(self):
            if self.refused(auth=False):
                return
            srv = world_server()
            if not srv:
                return self.send(503, {"error": "No world model server. Add one or start one in Servers."})
            return self.world(srv, "POST", self.path)

    WH.ports = ports
    return WH


def start(host="127.0.0.1", port=18930, world_port=None):
    """Binds the app (host:port) and the world page's own origin (world_port; default port + 1, any free port when that
    one is taken). Returns (app server, world server), not yet serving. The launch token never leaves this process
    except inside the app's own pages."""
    ports: dict = {}
    srv = ThreadingHTTPServer((host, port), make_handler(new_token(), ports, host))
    ports["app"] = srv.server_address[1]
    want = world_port if world_port is not None else (ports["app"] + 1 if port else 0)
    try:
        wsrv = ThreadingHTTPServer((host, want), make_world_handler(ports, host))
    except OSError:
        wsrv = ThreadingHTTPServer((host, 0), make_world_handler(ports, host))
    ports["world"] = wsrv.server_address[1]
    for s in (srv, wsrv):
        s.daemon_threads = True
    return srv, wsrv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mashbox.ui.app", description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18930)
    ap.add_argument("--world-port", type=int, default=None, help="the world model page's own origin (default: --port + 1)")
    a = ap.parse_args(argv)
    srv, wsrv = start(a.host, a.port, a.world_port)
    threading.Thread(target=wsrv.serve_forever, daemon=True).start()

    def bye(*_):
        for job in list(JOBS.values()):
            job.stop()                          # closes the tunnel; a chip keeps its own max lifetime
        os._exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    print(json.dumps({"event": "ready", "url": f"http://{a.host}:{a.port}/", "world_port": wsrv.server_address[1],
                      "version": __version__}), flush=True)
    srv.serve_forever()
