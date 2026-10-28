"""Request checks for the package's local HTTP servers (the inference server, runtime/server.py, uses them). Stdlib
only.

A browser lets any web page send requests to 127.0.0.1, and a page whose own domain is re-pointed at 127.0.0.1 (DNS
rebinding) can also read the answers. These checks stop both, while clients that are not browsers (curl, coding
agents, SDKs, scripts) keep working unchanged:

- Host: the name the client used must be localhost (or a name under .localhost), an IP address, or a name the server
  was told to answer to (`hosts`: the server's own --host, plus MASHBOX_ALLOW_HOSTS). A rebinding page always sends
  its own domain, so it fails here. The port is not checked: an SSH tunnel may forward another local port (`mashbox
  tpu up` takes 8001 when 8000 is in use). No Host header means no browser, and passes.
- Origin: absent passes (only browsers send it). Present, it must be the server's own origin (the Host header's name
  and port; localhost, 127.0.0.1 and [::1] count as one name) or one listed in `origins` / MASHBOX_ALLOW_ORIGINS,
  written as the browser sends it ("http://127.0.0.1:18931"). "null" (sandboxed frames, file: pages) passes only if listed.
- Content-Type: a POST must be application/json. A web page can send that to another origin only after a CORS
  preflight (OPTIONS), which these servers never approve, so a "simple" text/plain or form POST never reaches a handler.
- WebSocket: browsers send Origin on every handshake, so an upgrade must carry an allowed one.
- token (servers that have one, made by new_token() at launch): a request checked with auth=True carries it in the
  X-Mashbox-Token header, in the cookie named `cookie`, or, on a WebSocket handshake only (a browser cannot set its
  headers), in a `token` query parameter. A cookie is sent by pages on the same host at any port, so a GET must never
  change anything.
- key (servers that have an API key): a request checked with auth=True carries it as `Authorization: Bearer <key>`
  or `x-api-key: <key>`, the two ways OpenAI and Anthropic clients send one.

The Host rule is the default allowedHosts of Vite and webpack-dev-server (localhost, *.localhost, IP addresses); the
Origin rule is Jupyter's Origin-matches-Host check. Use it first thing in every handler method:

    G = Guard(hosts=[args.host], token=new_token())
    err = G.check(self.command, self.path, self.headers, auth=...)   # auth=False for pages and static files
    if err:
        return <answer err[0] with the message err[1]>                # never holds a secret

Never answer OPTIONS with Access-Control-Allow-* headers: the Content-Type rule relies on preflights failing.
"""
import hmac
import ipaddress
import os
import re
import secrets
import urllib.parse

HEADER = "X-Mashbox-Token"
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_NETLOC = re.compile(r"(\[[0-9a-f:.]+\]|[a-z0-9_.-]+)(?::([0-9]{1,5}))?")
_PORT = {"http": 80, "https": 443}


def new_token():
    """A random per-launch secret: 32 bytes, URL-safe text."""
    return secrets.token_urlsafe(32)


def set_cookie(name, token):
    """The Set-Cookie value that gives a page its token: page scripts cannot read it, other sites' requests never carry it."""
    return f"{name}={token}; Path=/; HttpOnly; SameSite=Strict"


def loopback(host):
    """True for an address or name only this machine can reach (127.0.0.0/8, ::1, localhost); 0.0.0.0 is not."""
    h = (host or "").strip().strip("[]").lower().rstrip(".")
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _netloc(s):
    """'LocalHost.:8000' -> ('localhost', 8000), '[::1]' -> ('::1', None); anything malformed -> None."""
    m = _NETLOC.fullmatch((s or "").strip().lower())
    if not m or (m.group(2) and int(m.group(2)) > 65535):
        return None
    return m.group(1).strip("[]").rstrip("."), int(m.group(2)) if m.group(2) else None


def _is_ip(h):
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _env(name):
    return [x for x in os.environ.get(name, "").split(",") if x.strip()]


def _same(given, secret):
    return given is not None and hmac.compare_digest(given.encode(), secret.encode())


class Guard:
    """One server's checks. hosts: extra names it answers to (pass its --host); origins: other origins allowed to call
    it; token: its per-launch secret or None; key: its API key or None; cookie: the cookie name that may carry the token."""

    def __init__(self, hosts=(), origins=(), token=None, key=None, cookie=None):
        self.hosts = {h.strip().strip("[]").lower().rstrip(".") for h in [*hosts, *_env("MASHBOX_ALLOW_HOSTS")] if h}
        self.origins = {o.strip().lower().rstrip("/") for o in [*origins, *_env("MASHBOX_ALLOW_ORIGINS")]}
        self.token, self.key, self.cookie = token, key, cookie

    def host_ok(self, host):
        """The Host header: localhost or *.localhost, an IP address, or a configured name; any port."""
        if host is None:
            return True
        hp = _netloc(host)
        return hp is not None and (hp[0] == "localhost" or hp[0].endswith(".localhost") or _is_ip(hp[0]) or hp[0] in self.hosts)

    def origin_ok(self, origin, host):
        """The Origin header: absent, the server's own origin (as named by `host`), or a listed one."""
        if origin is None:
            return True
        o = origin.strip().lower().rstrip("/")
        if o in self.origins:
            return True
        m = re.fullmatch(r"(https?)://(.+)", o)
        theirs, mine = m and _netloc(m.group(2)), _netloc(host)
        if not (theirs and mine):
            return False
        same = theirs[0] == mine[0] or {theirs[0], mine[0]} <= LOOPBACK
        return same and (theirs[1] or _PORT[m.group(1)]) == (mine[1] or _PORT[m.group(1)])

    def _token(self, headers, path, ws):
        t = headers.get(HEADER)
        if t is None and self.cookie:
            for part in (headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == self.cookie:
                    t = v
                    break
        if t is None and ws:
            t = (urllib.parse.parse_qs(urllib.parse.urlsplit(path).query).get("token") or [None])[0]
        return t

    def check(self, method, path, headers, auth=True):
        """None when the request may go on, else (HTTP status, message). `headers` is the handler's self.headers.
        auth=False skips the token and key (pages and static files); Host, Origin, Content-Type are always checked."""
        host, origin = headers.get("Host"), headers.get("Origin")
        ws = (headers.get("Upgrade") or "").lower() == "websocket"
        if not self.host_ok(host):
            return 403, "Host not allowed: open this server as 127.0.0.1 or localhost (or allow the name)"
        if (ws and origin is None) or not self.origin_ok(origin, host):
            return 403, "Origin not allowed: requests from other web pages are refused"
        if method == "POST" and (headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return 415, "Content-Type must be application/json"
        if auth and self.token and not _same(self._token(headers, path, ws), self.token):
            return 401, "missing or wrong session token"
        if auth and self.key:
            a = headers.get("Authorization") or ""
            given = [a[7:].strip() if a[:7].lower() == "bearer " else None, headers.get("x-api-key")]
            if not any(_same(g, self.key) for g in given):
                return 401, "missing or wrong API key"
        return None
