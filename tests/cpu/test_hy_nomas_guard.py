"""World model server (models/hy_nomas/server.py): other web
pages must not drive or watch the session (no-cors POSTs, WebSockets, DNS rebinding), and no request may stop generation.
Real handler on a real socket with the stand-in engine. No ffmpeg: the stream object is only a holder for the client set."""
import base64
import http.client
import json
import re
import socket
import threading
import time
import types

import numpy as np
import pytest

from mashbox.models.hy_nomas import server as S
from mashbox.runtime import guard as G

EVIL = "https://attacker.example"
NO_CORS = {"Origin": EVIL, "Content-Type": "text/plain;charset=UTF-8"}   # what fetch(..., {mode: 'no-cors'}) sends


class SeededStub(S.StubEngine):
    """The stand-in engine, but reset() seeds numpy's RandomState as the real runtime's reset does
    (runtime.DiTRuntime.reset), so a seed outside 0..2**32-1 raises here as it does on the chips."""

    def reset(self, seed):
        self.rng = np.random.RandomState(seed)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def world():
    eng = SeededStub(lambda *a: None, dit_s=0.05, dec_s=0.01, encode_s=0.0)
    live = S.Live(eng, lambda *a: None, hw="STUB", stub=True, buffer=0.2)
    live.stream = types.SimpleNamespace(config=None, cl=threading.Lock(), clients=set())
    threading.Thread(target=live.producer, daemon=True).start()
    port = free_port()
    srv = live.serve(port)
    yield live, port
    srv.shutdown()
    srv.server_close()


def call(port, method, path, headers=None, body=None):
    """One request with exactly these headers (Host defaults to 127.0.0.1:port) -> (status, body bytes)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    hs = {"Host": "127.0.0.1:%d" % port, **(headers or {})}
    data = None if body is None else json.dumps(body).encode()
    for k, v in hs.items():
        c.putheader(k, v)
    if data is not None:
        c.putheader("Content-Length", str(len(data)))
    c.endheaders(data)
    r = c.getresponse()
    out = r.status, r.read()
    c.close()
    return out


def token(port):
    """The token as the page gets it: from the page the server serves ('' when the page carries none)."""
    m = re.search(rb'<meta name="mbx-token" content="([^"]+)">', call(port, "GET", "/")[1])
    return m.group(1).decode() if m else ""


def page_headers(port, tok, origin=None):
    """What the page's own fetch sends: its origin, JSON, the token it was served."""
    return {"Origin": origin or "http://127.0.0.1:%d" % port, "Content-Type": "application/json", "X-Mashbox-Token": tok}


def ws(port, path, headers):
    """A WebSocket handshake -> the HTTP status of the answer (101 = upgraded). The socket is closed after."""
    hs = {"Host": "127.0.0.1:%d" % port, "Upgrade": "websocket", "Connection": "Upgrade",
          "Sec-WebSocket-Key": base64.b64encode(b"the sample nonce").decode(), "Sec-WebSocket-Version": "13", **headers}
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(("GET %s HTTP/1.1\r\n" % path + "".join("%s: %s\r\n" % kv for kv in hs.items()) + "\r\n").encode())
        return int(s.recv(4096).split(b" ", 2)[1])


def until(ok, s=10.0):
    t0 = time.time()
    while time.time() - t0 < s and not ok():
        time.sleep(0.02)
    return ok()


def test_other_sites_cannot_drive_the_session(world):
    """a no-cors text/plain POST from any site changed the world, actions, seed, pause and settings."""
    live, port = world
    before = dict(live.settings)
    wid = next(iter(live.worlds))
    for path, body in [("/api/settings", {"rate": 90, "preset": "live", "decoder": "fast", "sync": False}),
                       ("/api/world", {"mode": "preset", "id": wid}), ("/api/pause", {"on": True}),
                       ("/api/restart", {"seed": 7}), ("/api/action", {"text": "leaps"})]:
        assert call(port, "POST", path, NO_CORS, body)[0] == 403, path
    assert call(port, "POST", "/api/settings", {"Content-Type": "text/plain"}, {"rate": 90})[0] == 415   # no Origin sent
    assert call(port, "POST", "/api/settings", {"Content-Type": "application/json"}, {"rate": 90})[0] == 401  # no token
    bad = page_headers(port, "not-the-token")
    assert call(port, "POST", "/api/settings", bad, {"rate": 90})[0] == 401
    time.sleep(0.2)
    assert live.settings == before and live.loading is None and live.pending_world is None and live.sess == 0
    tok = token(port)                                        # the page itself still works
    st, b = call(port, "POST", "/api/settings", page_headers(port, tok), {"rate": 30})
    assert st == 200 and json.loads(b)["settings"]["rate"] == 30.0 and live.settings["rate"] == 30.0
    st, _ = call(port, "POST", "/api/world", page_headers(port, tok), {"mode": "preset", "id": wid})
    assert st == 200 and until(lambda: live.sess == 1)


def test_rebinding_page_gets_nothing(world):
    """a page whose own domain now points at 127.0.0.1 is same-origin to the browser; its Host gives it away."""
    live, port = world
    rb = "rebind.attacker.example:%d" % port
    st, b = call(port, "GET", "/", {"Host": rb})
    assert st == 403 and b"mbx-token" not in b              # it cannot read the page, so it never has the token
    assert call(port, "GET", "/api/state", {"Host": rb})[0] == 403
    h = {"Host": rb, "Origin": "http://" + rb, "Content-Type": "application/json"}
    assert call(port, "POST", "/api/settings", h, {"rate": 90})[0] == 403
    assert ws(port, "/ws?mode=jpeg", {"Host": rb, "Origin": "http://" + rb}) == 403
    assert live.settings["rate"] != 90 and live.clients == 0
    assert call(port, "GET", "/")[0] == 200 and call(port, "GET", "/api/state")[0] == 200   # the plain route is unchanged


def test_other_sites_cannot_watch_or_drive(world):
    """any page could open /ws, read the video and stats, send keys and wake an idle-paused session."""
    live, port = world
    live.idle_paused = True
    own = "http://127.0.0.1:%d" % port
    tok = token(port)
    assert ws(port, "/ws?mode=jpeg", {"Origin": EVIL}) == 403
    assert ws(port, "/ws?mode=jpeg&token=" + tok, {"Origin": EVIL}) == 403                # even holding the token
    assert ws(port, "/ws?mode=jpeg", {"Origin": own}) == 401                              # own origin, no token
    assert ws(port, "/ws?mode=jpeg&token=nope", {"Origin": own}) == 401
    assert ws(port, "/ws?mode=jpeg&token=" + tok, {}) == 403                              # browsers always send Origin
    assert live.idle_paused and live.clients == 0
    assert ws(port, "/ws?mode=jpeg&token=" + tok, {"Origin": own}) == 101                 # the page itself
    assert until(lambda: not live.idle_paused)
    # the MASHBOX app shows the page from its own 127.0.0.1 port and passes the browser's Origin and the token on
    assert ws(port, "/ws?token=" + tok, {"Origin": "http://127.0.0.1:18931"}) == 101
    assert ws(port, "/ws", {"Origin": "http://localhost:18931"}) == 401
    assert G.Guard().origin_ok("http://127.0.0.1:18931", "127.0.0.1:%d" % port) is False   # why WorldGuard adds that rule
    assert until(lambda: live.clients == 0)


def test_a_bad_seed_cannot_stop_generation(world):
    """seed -1 (or >= 2**32) passed int() and killed the producer thread in RandomState; the stream froze for good."""
    live, port = world
    h = page_headers(port, token(port))
    wid = next(iter(live.worlds))
    assert call(port, "POST", "/api/world", h, {"mode": "preset", "id": wid, "seed": 5})[0] == 200
    assert until(lambda: live.sess == 1)
    for path, body in [("/api/restart", {"seed": -1}), ("/api/restart", {"seed": 2**32}),
                       ("/api/world", {"mode": "preset", "id": wid, "seed": -1})]:
        st, b = call(port, "POST", path, h, body)
        assert st == 400 and b"seed" in b, (path, body, st)
    with live.cv:                                            # a bad request that reaches the producer some other way
        live.req.append(dict(kind="restart", seed=-1))
        live.cv.notify_all()
    assert until(lambda: "Could not start" in live.msg)
    st, b = call(port, "POST", "/api/restart", h, {"seed": 2**32 - 1})
    assert st == 200 and until(lambda: live.sess == 2)     # the producer is still alive and takes the next request
    assert live.world["seed"] == 2**32 - 1 and live.eng.rng.randint(1 << 30) >= 0


def test_a_failed_start_pauses_with_the_error(world):
    """A start the engine refuses pauses the session and shows why (as 0.1.1 did): the page gets mode 'paused' and the
    error, generation stops, and the next good start plays again."""
    live, port = world
    h = page_headers(port, token(port))
    wid = next(iter(live.worlds))
    assert call(port, "POST", "/api/world", h, {"mode": "preset", "id": wid, "seed": 5})[0] == 200
    assert until(lambda: live.sess == 1 and live.mode == "playing")
    with live.cv:                                            # reset() raises in RandomState, as on the chips
        live.req.append(dict(kind="restart", seed=-1))
        live.cv.notify_all()
    assert until(lambda: "Could not start" in live.msg)
    assert live.mode == "paused" and live.stats()["mode"] == "paused"
    assert live.msg.startswith("Could not start that: ") and len(live.msg) > len("Could not start that: ")   # the error
    w_paused = live.w
    time.sleep(0.5)                                          # paused: the producer makes no more windows
    assert live.w == w_paused and live.mode == "paused" and "Could not start" in live.msg
    st, b = call(port, "POST", "/api/restart", h, {"seed": 7})
    assert st == 200 and until(lambda: live.sess == 2 and live.mode == "playing")


def test_allowed_host_names_also_pass_as_origin(monkeypatch):
    """MASHBOX_ALLOW_HOSTS alone lets your own tunnel or proxy
    name open the page (as Host and as the browser's Origin); other names stay refused."""
    monkeypatch.setenv("MASHBOX_ALLOW_HOSTS", "wm.example.org")
    g = S.WorldGuard(token="t")
    assert g.host_ok("wm.example.org") and not g.host_ok("evil.example")
    assert g.origin_ok("https://wm.example.org", "127.0.0.1:8790")
    assert g.origin_ok("https://localhost:18898", "127.0.0.1:8790")
    assert not g.origin_ok("https://evil.example", "127.0.0.1:8790")
    assert not g.origin_ok("https://wm.example.org.evil.example", "127.0.0.1:8790")
    assert not g.origin_ok("null", "127.0.0.1:8790")
