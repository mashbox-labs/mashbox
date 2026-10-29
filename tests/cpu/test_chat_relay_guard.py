"""`python -m mashbox.ui.chat --api ...` (port 8801 by default) relays /v1 to a model server and sets the upstream Host
itself, so the model server's own guard never sees the browser's Host or Origin. Found apart from the other
security checks: a page whose domain is re-pointed at 127.0.0.1 could read the model's replies through the relay, and
any site could push "simple" text/plain POSTs through it. Real relay (its main()) on a real socket, a stand-in upstream."""
import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mashbox.ui.chat import __main__ as relay


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def chat():
    hits = []

    class Up(BaseHTTPRequestHandler):
        def _reply(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            hits.append(("GET", self.path))
            self._reply({"data": [{"id": "stub-model"}]})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            hits.append(("POST", self.path, self.rfile.read(n)))
            self._reply({"choices": [{"message": {"role": "assistant", "content": "secret reply"}}]})

        def log_message(self, *a):
            pass
    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    port = free_port()
    threading.Thread(target=relay.main, args=(["--api", "http://127.0.0.1:%d" % up.server_address[1], "--port", str(port)],),
                     daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    hits.clear()
    yield port, hits
    up.shutdown()


def call(port, method, path, headers, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for k, v in headers.items():
        c.putheader(k, v)
    b = json.dumps(body).encode() if body is not None else b""
    if method == "POST":
        c.putheader("Content-Length", str(len(b)))
    c.endheaders(b if method == "POST" else None)
    r = c.getresponse()
    out = r.status, r.read()
    c.close()
    return out


CHAT = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}]}


def test_rebinding_page_cannot_read_replies_through_the_relay(chat):
    port, hits = chat
    rebound = {"Host": "rebind.attacker.example:%d" % port, "Origin": "http://rebind.attacker.example:%d" % port}
    assert call(port, "GET", "/v1/models", {"Host": rebound["Host"]})[0] == 403
    st, b = call(port, "POST", "/v1/chat/completions", dict(rebound, **{"Content-Type": "application/json"}), CHAT)
    assert st == 403 and b"secret reply" not in b
    assert hits == []                                                       # nothing reached the model server


def test_other_sites_cannot_post_through_the_relay(chat):
    port, hits = chat
    own = {"Host": "127.0.0.1:%d" % port}
    st, _ = call(port, "POST", "/v1/chat/completions",
                 dict(own, Origin="https://attacker.example", **{"Content-Type": "text/plain;charset=UTF-8"}), CHAT)
    assert st in (403, 415)
    st, _ = call(port, "POST", "/v1/chat/completions", dict(own, **{"Content-Type": "text/plain;charset=UTF-8"}), CHAT)
    assert st == 415
    assert hits == []


def test_the_page_and_plain_clients_still_work(chat):
    port, hits = chat
    own = {"Host": "127.0.0.1:%d" % port}
    assert call(port, "GET", "/", own)[0] == 200
    J = dict(own, **{"Content-Type": "application/json"})
    assert call(port, "POST", "/v1/chat/completions", dict(J, Origin="http://127.0.0.1:%d" % port), CHAT)[0] == 200
    assert call(port, "POST", "/v1/chat/completions", dict(J, Host="localhost:%d" % port), CHAT)[0] == 200   # curl, scripts
    assert call(port, "GET", "/v1/models", own)[0] == 200
    assert [h[0] for h in hits] == ["POST", "POST", "GET"]
