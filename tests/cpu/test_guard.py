"""runtime/guard.py, the request checks for the local servers: guards against a web page driving them.

Each refused case is a request shape a web page can send: a cross-site text/plain POST, a DNS-rebound Host, a
WebSocket from another origin, a missing token or key.
Each passing case is a client that must keep working: curl as in docs/server-api.md, coding agents (no Origin), a
tunnel on another local port, the server's own page, IPv6, a configured --host. Headers are parsed by
http.client.parse_headers, the parser BaseHTTPRequestHandler uses for self.headers, so case and lookup match a handler.
"""
import base64
import http.client
import io

import pytest

from mashbox.runtime import guard as G

TOKEN = G.new_token()
KEY = G.new_token()                     # a fresh key each run, as a server makes one


def H(**h):
    """Header lines as a server handler sees them; '_' in a name becomes '-'."""
    raw = "".join(f"{k.replace('_', '-')}: {v}\r\n" for k, v in h.items()) + "\r\n"
    return http.client.parse_headers(io.BytesIO(raw.encode("latin-1")))


JSON = "application/json"
WS = dict(Upgrade="websocket", Connection="Upgrade", Sec_WebSocket_Key=base64.b64encode(b"the sample nonce").decode(), Sec_WebSocket_Version="13")

REFUSED = [   # (why, method, path, headers, expected status)
    ("no-cors text/plain POST from a web page", "POST", "/v1/chat/completions",
     H(Host="127.0.0.1:18930", Origin="https://attacker.example", Content_Type="text/plain;charset=UTF-8"), 403),
    ("same, page with Referrer-Policy no-referrer (Origin: null)", "POST", "/v1/messages",
     H(Host="127.0.0.1:8000", Origin="null", Content_Type="text/plain"), 403),
    ("same, a browser that sends no Origin: the body is not JSON", "POST", "/v1/messages",
     H(Host="127.0.0.1:8000", Content_Type="text/plain"), 415),
    ("a form POST", "POST", "/api/world", H(Host="127.0.0.1:8790", Content_Type="application/x-www-form-urlencoded"), 415),
    ("a POST with no Content-Type", "POST", "/api/restart", H(Host="127.0.0.1:8790"), 415),
    ("DNS-rebound page reading the API (same-origin to the browser)", "GET", "/v1/models",
     H(Host="rebind.attacker.tld:18930"), 403),
    ("rebound page POSTing JSON to /v1", "POST", "/v1/chat/completions",
     H(Host="rebind.attacker.tld:18930", Origin="http://rebind.attacker.tld:18930", Content_Type=JSON), 403),
    ("WebSocket from another site", "GET", "/ws?mode=jpeg",
     H(Host="127.0.0.1:8790", Origin="https://attacker.example", **WS), 403),
    ("WebSocket with no Origin", "GET", "/ws", H(Host="127.0.0.1:8790", **WS), 403),
    ("another local web app on another port", "POST", "/api/settings",
     H(Host="127.0.0.1:18930", Origin="http://127.0.0.1:9999", Content_Type=JSON), 403),
    ("malformed Origin", "POST", "/v1/messages", H(Host="127.0.0.1:8000", Origin="http://127.0.0.1:8000/x", Content_Type=JSON), 403),
    ("malformed Host: port out of range", "GET", "/health", H(Host="127.0.0.1:99999"), 403),
    ("malformed Host: userinfo", "GET", "/health", H(Host="evil@127.0.0.1"), 403),
    ("empty Host", "GET", "/health", H(Host=""), 403),
]

PASSING = [   # (who, method, path, headers)
    ("curl as in docs/server-api.md", "POST", "/v1/chat/completions", H(Host="127.0.0.1:8000", Content_Type=JSON)),
    ("Claude Code / Anthropic SDK", "POST", "/v1/messages",
     H(Host="localhost:8000", Content_Type="application/json; charset=utf-8", x_api_key="anything", Sec_Fetch_Mode="cors")),
    ("Codex through a tunnel that took local port 8001", "POST", "/v1/responses", H(Host="127.0.0.1:8001", Content_Type=JSON)),
    ("a GET with no Host (HTTP/1.0 client)", "GET", "/health", H()),
    ("the server's own chat page", "POST", "/v1/chat/completions",
     H(Host="127.0.0.1:8000", Origin="http://127.0.0.1:8000", Content_Type=JSON)),
    ("own page opened as localhost, calling 127.0.0.1 on the same port", "POST", "/v1/chat/completions",
     H(Host="127.0.0.1:8000", Origin="http://localhost:8000", Content_Type=JSON)),
    ("IPv6 loopback, same origin", "POST", "/api/world", H(Host="[::1]:8790", Origin="http://[::1]:8790", Content_Type=JSON)),
    ("own page's WebSocket", "GET", "/ws?mode=jpeg", H(Host="127.0.0.1:8790", Origin="http://127.0.0.1:8790", **WS)),
    ("a LAN client of a server on --host 0.0.0.0, by IP", "POST", "/v1/messages",
     H(Host="192.0.2.20:8000", Origin="http://192.0.2.20:8000", Content_Type=JSON)),
    ("a name under .localhost", "GET", "/health", H(Host="mashbox.localhost:8000")),
]


@pytest.mark.parametrize("why,method,path,headers,code", REFUSED, ids=[r[0] for r in REFUSED])
def test_refused(why, method, path, headers, code):
    err = G.Guard(hosts=["127.0.0.1"]).check(method, path, headers, auth=False)
    assert err is not None and err[0] == code, why


@pytest.mark.parametrize("who,method,path,headers", PASSING, ids=[p[0] for p in PASSING])
def test_plain_clients_pass(who, method, path, headers):
    assert G.Guard(hosts=["0.0.0.0"]).check(method, path, headers) is None, who


def test_configured_names_and_origins(monkeypatch):
    """A name the server was told about (its --host, MASHBOX_ALLOW_HOSTS) and a listed origin pass; nothing else does."""
    box = H(Host="box.lan:8000", Content_Type=JSON)
    assert G.Guard().check("POST", "/v1/messages", box)[0] == 403
    assert G.Guard(hosts=["box.lan"]).check("POST", "/v1/messages", box) is None
    monkeypatch.setenv("MASHBOX_ALLOW_HOSTS", "world.example, box.lan")
    monkeypatch.setenv("MASHBOX_ALLOW_ORIGINS", "http://127.0.0.1:18931")
    g = G.Guard()
    assert g.check("POST", "/v1/messages", box) is None
    behind_tls = H(Host="world.example", Origin="https://world.example", Content_Type=JSON)   # a TLS proxy in front
    assert g.check("POST", "/api/world", behind_tls) is None
    assert g.check("POST", "/api/world", H(Host="world.example", Origin="http://world.example:8080", Content_Type=JSON))[0] == 403
    app_world = H(Host="127.0.0.1:8790", Origin="http://127.0.0.1:18931", **WS)              # a listed origin's WebSocket
    assert g.check("GET", "/ws", app_world) is None
    assert G.Guard(hosts=["127.0.0.1"]).check("POST", "/api/x", H(Host="127.0.0.1:8790", Origin="null", Content_Type=JSON))[0] == 403
    assert G.Guard(origins=["null"]).check("POST", "/api/x", H(Host="127.0.0.1:8790", Origin="null", Content_Type=JSON)) is None


def test_token():
    """With a token, auth=True needs it: header always, the named cookie, the query only on a WebSocket handshake."""
    g = G.Guard(token=TOKEN, cookie="mashbox_app")
    own = dict(Host="127.0.0.1:18930", Origin="http://127.0.0.1:18930")
    assert g.check("POST", "/api/tpu/up", H(**own, Content_Type=JSON)) == (401, "missing or wrong session token")
    assert g.check("POST", "/api/tpu/up", H(**own, Content_Type=JSON, X_Mashbox_Token=TOKEN[:-1] + "x"))[0] == 401
    assert g.check("POST", "/api/tpu/up", H(**own, Content_Type=JSON, X_Mashbox_Token=TOKEN)) is None
    assert g.check("GET", "/m/chat", H(Host="127.0.0.1:18930", Cookie=f"_xsrf=1; mashbox_app={TOKEN}; other=2")) is None
    assert g.check("GET", "/m/chat", H(Host="127.0.0.1:18930", Cookie=f"mashbox_other={TOKEN}"))[0] == 401
    assert g.check("GET", f"/api/state?token={TOKEN}", H(Host="127.0.0.1:18930"))[0] == 401     # query: WebSocket only
    assert g.check("GET", f"/m/world/ws?token={TOKEN}", H(**own, **WS)) is None
    assert g.check("GET", "/m/world/ws?token=nope", H(**own, **WS))[0] == 401
    assert g.check("GET", "/", H(Host="127.0.0.1:18930"), auth=False) is None                  # the page that hands it out


def test_key():
    """With an API key (on by default off loopback), auth=True needs it as a Bearer or x-api-key, the OpenAI and
    Anthropic ways; health stays open with auth=False."""
    g = G.Guard(hosts=["0.0.0.0"], key=KEY)
    req = dict(Host="198.51.100.5:8000", Content_Type=JSON)
    assert g.check("POST", "/v1/chat/completions", H(**req)) == (401, "missing or wrong API key")
    assert g.check("POST", "/v1/chat/completions", H(**req, Authorization="Bearer wrong"))[0] == 401
    assert g.check("POST", "/v1/chat/completions", H(**req, Authorization=f"Bearer {KEY}")) is None
    assert g.check("POST", "/v1/messages", H(**req, x_api_key=KEY)) is None
    assert g.check("POST", "/v1/messages", H(**req, Authorization="Bearer stale", x_api_key=KEY)) is None
    assert g.check("POST", "/v1/messages", H(**req, Authorization=f"Basic {KEY}"))[0] == 401
    assert g.check("GET", "/health", H(Host="198.51.100.5:8000"), auth=False) is None


def test_messages_hold_no_secret_and_helpers():
    """Refusals never echo a secret; loopback() picks when a key is on by default; tokens are fresh and cookie-safe."""
    g = G.Guard(token=TOKEN, key=KEY)
    for h in (H(Host="evil.example"), H(Host="127.0.0.1", X_Mashbox_Token=TOKEN[:5]), H(Host="127.0.0.1", X_Mashbox_Token=TOKEN)):
        err = g.check("GET", "/x", h)
        assert err is not None and TOKEN not in err[1] and KEY not in err[1]
    assert all(G.loopback(h) for h in ("127.0.0.1", "127.0.0.2", "localhost", "LocalHost.", "::1", "[::1]", "a.localhost"))
    assert not any(G.loopback(h) for h in ("0.0.0.0", "::", "", "198.51.100.5", "example.com", "localhost.example.com"))
    toks = {G.new_token() for _ in range(50)}
    assert len(toks) == 50 and all(len(t) >= 43 and t.replace("-", "").replace("_", "").isalnum() for t in toks)
    c = G.set_cookie("mashbox_app", TOKEN)
    assert c.startswith(f"mashbox_app={TOKEN};") and "HttpOnly" in c and "SameSite=Strict" in c and "Path=/" in c
