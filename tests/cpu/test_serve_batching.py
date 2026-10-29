"""Server waiting limits and client hang-ups on the continuous-batching loop (skill mashbox-engine-batching E7, E8).

Guards these failures, each seen on this server:
- a client that hangs up while its request waits for a slot keeps its owner's waiting count (the legacy streaming route
  /v1/completions held it until the request was admitted and its first write failed, so 32 dead clients locked an owner
  out with 429s);
- the per-owner (429) and overall (429, Anthropic 529; never 503, which means only "not ready") waiting limits, the longest-queue drop, Retry-After on every
  refusal, and the legacy streaming route answering 429 itself instead of after a 200;
- a raw JSON Infinity, -Infinity, NaN or 1e999 in a numeric field answered 500 (OverflowError in int()), not 400;
- the standard library's accept queue of 5 resetting connections when many clients connect at once.
The engine here is a scripted stand-in with the real engine's admit / step / cancel / check interface, so the server's
loop_live path runs; the tiny-model checks (engine_batch_checks.py) cover the same limits on the real engine.
"""
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from mashbox.runtime import server as SV
import api_stub_engine as STUB


class IdChat:
    """Token ids as text ('12 7 301'): the server path without a tokenizer."""
    stop_ids = ()

    def encode(self, text):
        return [int(x) for x in str(text).split() if x.isdigit()]

    def decode(self, ids):
        return "".join(f"{int(t)} " for t in ids)

    def chat_ids(self, msgs, *a, **k):
        c = msgs[-1].get("content")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        return self.encode(c)


class LiveEngine:
    """Scripted engine with the continuous-batching interface (Engine.admit / step / cancel / check / free_slots / busy):
    each live row gets token 7 per step and ends at max_new. `gate` holds every step until set."""
    CTX, K, prefix_cache, V = 4096, 4, False, 1000

    def __init__(self, S=1):
        self.S, self.rows = S, [None] * S
        self.gate = threading.Event()
        self.gate.set()

    def check(self, q):
        if max(q["ids"] or [0]) >= self.V:
            return f"token ids must be 0..{self.V - 1}"
        return None

    def free_slots(self):
        return sum(r is None for r in self.rows)

    def busy(self):
        return any(r is not None for r in self.rows)

    def admit(self, reqs, stop_ids=()):
        out = []
        for q in reqs:
            now = time.perf_counter()
            why = self.check(q)
            if why is not None:
                out.append(dict(slot=None, ids=[], fin="error", error=why, cached=0, restored=0, sampled=False,
                                t_admit=now, t_first=None, t_end=now, calls=0))
                continue
            s = self.rows.index(None)
            r = dict(slot=s, ids=[], fin=None, cached=0, restored=0, sampled=False, t_admit=now, t_first=None,
                     t_end=None, calls=0, max_new=q["max_new"])
            self.rows[s] = r
            out.append(r)
        return out

    def _end(self, r, fin):
        r["fin"], r["t_end"] = fin, time.perf_counter()
        self.rows[r["slot"]] = None

    def cancel(self, rows):
        out = []
        for r in rows:
            if r is not None and r["fin"] is None and self.rows[r["slot"]] is r:
                self._end(r, "cancel")
                out.append(r)
        return out

    def step(self, waiting=False):
        self.gate.wait()
        ended = []
        for r in [r for r in self.rows if r is not None]:
            r["calls"] += 1
            r["ids"].append(7)
            if r["t_first"] is None:
                r["t_first"] = time.perf_counter()
            if len(r["ids"]) >= r["max_new"]:
                self._end(r, "length")
                ended.append(r)
        return ended


def start(S=1, keys=None, log=None):
    eng = LiveEngine(S)
    srv = SV.Server(eng, IdChat(), "m", {}, log=log or (lambda m: None))
    srv.ready = True
    port = STUB.serve_on_free_port(srv, keys=keys)
    return eng, srv, port


def hold(eng, srv):
    """Every slot busy for the whole test: the engine's next step waits for the gate. -> the holding job."""
    eng.gate.clear()
    j = srv.submit([1, 2, 3], SV.knobs_of(dict(max_tokens=3000)), True, False, scope="hold")
    t = time.time()
    while eng.free_slots() and time.time() - t < 5:
        time.sleep(0.005)
    assert not eng.free_slots()
    return j


def release(eng, srv, jobs):
    for j in jobs:
        srv.hangup(j)
    eng.gate.set()


def call(port, path, body, headers=None):
    """POST with the body sent as written (a str keeps raw JSON literals such as Infinity) -> (status, body, headers)."""
    data = (body if isinstance(body, str) else json.dumps(body)).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data, h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


ROUTES = {
    "completions": ("/v1/completions", dict(prompt="5 6 7 8", max_tokens=4)),
    "chat": ("/v1/chat/completions", dict(model="m", max_tokens=4, messages=[dict(role="user", content="5 6 7 8")])),
    "anthropic": ("/v1/messages", dict(model="m", max_tokens=4, messages=[dict(role="user", content="5 6 7 8")])),
    "responses": ("/v1/responses", dict(model="m", max_output_tokens=4, input="5 6 7 8")),
}


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("route", list(ROUTES))
def test_waiting_client_that_hangs_up_frees_its_count(route, stream):
    """E7: 8 clients send a request while the only slot is busy, then close their connection; every waiting count is
    freed within 0.5 s (spec E7 bound), on every route, streamed or not, with the slot still busy."""
    eng, srv, port = start()
    held = hold(eng, srv)
    path, body = ROUTES[route]
    data = json.dumps(dict(body, stream=stream)).encode()
    socks = []
    for _ in range(8):
        c = socket.create_connection(("127.0.0.1", port), timeout=10)
        c.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
        socks.append(c)
    t = time.time()
    while srv.waiting < 8 and time.time() - t < 5:
        time.sleep(0.005)
    assert srv.waiting == 8
    for c in socks:
        c.close()
    t = time.time()
    while srv.waiting and time.time() - t < 3:
        time.sleep(0.005)
    freed_s = time.time() - t
    still_busy = eng.busy() and not held["ev"].is_set()
    release(eng, srv, [held])
    assert srv.waiting == 0 and freed_s <= 0.5, (srv.waiting, round(freed_s, 3))
    assert still_busy


def test_waiting_limits_per_owner_and_overall(monkeypatch):
    """E7: with the only slot busy, owner A (key-a's tenant) gets 32 waiting and then 429 with Retry-After; owner B (key-b's) is not
    refused; the legacy streaming route answers 429 itself (no 200 first). With the overall limit full, owner C (under
    its share) gets in by dropping A's newest waiting request (that request ends with Overloaded), and A then gets 429 on
    the OpenAI routes and 529 overloaded_error on the Anthropic route, both with Retry-After (A1: 503 only while not
    ready)."""
    eng, srv, port = start(keys=["key-a", "key-b", "key-c"])
    from mashbox.runtime import tenancy as T
    ta, tb, tc = (T.current().tenant_id(k) for k in ("key-a", "key-b", "key-c"))   # each key's owner (its tenant id)
    held = hold(eng, srv)
    jobs, refused = {ta: [], tb: []}, {ta: [], tb: []}
    for owner, n in ((ta, 40), (tb, 5)):
        for _ in range(n):
            try:
                jobs[owner].append(srv.submit([5, 6], SV.knobs_of(dict(max_tokens=2)), True, False, scope=owner))
            except SV.Busy as e:
                refused[owner].append((type(e).__name__, e.retry_after))
    assert len(jobs[ta]) == 32 and len(refused[ta]) == 8 and {k for k, _ in refused[ta]} == {"Busy"}
    assert all(1 <= ra <= 60 for _, ra in refused[ta])
    assert jobs[tb] and not refused[tb]
    a = {"Authorization": "Bearer key-a"}
    code, body, h = call(port, "/v1/completions", dict(prompt="5 6", max_tokens=2, stream=True), a)
    assert code == 429 and h.get("Retry-After") and not body.startswith(b"data:")
    code, _, h = call(port, "/v1/chat/completions", dict(ROUTES["chat"][1]), a)
    assert code == 429 and h.get("Retry-After")
    monkeypatch.setattr(SV, "MAX_WAITING", srv.waiting)
    monkeypatch.setattr(SV, "MAX_WAITING_OWNER", 100)
    newest_a = jobs[ta][-1]
    c_job = srv.submit([5, 6], SV.knobs_of(dict(max_tokens=2)), True, False, scope=tc)
    assert isinstance(newest_a.get("error"), SV.Overloaded) and newest_a["ev"].is_set()
    code, body, h = call(port, "/v1/chat/completions", dict(ROUTES["chat"][1]), a)
    assert code == 429 and h.get("Retry-After") and json.loads(body)["error"]["type"] == "server_overloaded"
    code, body, h = call(port, "/v1/messages", dict(ROUTES["anthropic"][1]), {"x-api-key": "key-a"})
    assert code == 529 and h.get("Retry-After") and json.loads(body)["error"]["type"] == "overloaded_error"
    release(eng, srv, jobs[ta] + jobs[tb] + [c_job, held])
    t = time.time()
    while srv.waiting and time.time() - t < 3:
        time.sleep(0.01)
    assert srv.waiting == 0


NONFINITE = ("Infinity", "-Infinity", "NaN", "1e999", "-1e999")
EDGES = NONFINITE + ("2147483648", "9223372036854775808", "18446744073709551616", "-1", "0", "1.5", "true", '"x"',
                     "null", "[]")
FIELDS = {"completions": ("max_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                          "frequency_penalty", "presence_penalty"),
          "chat": ("max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "seed", "presence_penalty"),
          "anthropic": ("max_tokens", "temperature", "top_p", "top_k"),
          "responses": ("max_output_tokens", "temperature", "top_p")}


def test_numeric_field_edges_get_400_never_500():
    """E8: every numeric field of every generating route, sent as a raw JSON literal at float and integer edges
    (Infinity, NaN, 1e999, 2^31, 2^63, 2^64, ...), answers 400 or 200, never 500; non-finite numbers always 400."""
    eng, srv, port = start(S=2)
    bad = {}
    for route, fields in FIELDS.items():
        path, body = ROUTES[route]
        for f in fields:
            for lit in EDGES:
                b = dict(body, temperature=0.5)
                b.pop(f, None)
                raw = json.dumps(b)[:-1] + f', "{f}": {lit}}}'
                code, out, _ = call(port, path, raw)
                if code not in (200, 400) or (lit in NONFINITE and code != 400):
                    bad[f"{route}.{f}={lit}"] = (code, out[:120])
    assert not bad, bad
    assert srv.waiting == 0


def test_accept_queue_takes_256_clients_connecting_at_once():
    """256 clients connect at the same moment: every one gets its answer. The standard library's accept queue of 5
    reset 15 of 128 requests at 64 users on a v6e-1; the server's queue is 1024 (the kernel caps it at somaxconn)."""
    assert SV.Listener.request_queue_size >= 1024
    _, _, port = start()
    go, res = threading.Event(), []

    def hit():
        go.wait()
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=10)
            c.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            b = b""
            while True:
                x = c.recv(65536)
                if not x:
                    break
                b += x
            c.close()
            res.append(b.split(b"\r\n", 1)[0])
        except OSError as e:
            res.append(type(e).__name__.encode())
    th = [threading.Thread(target=hit, daemon=True) for _ in range(256)]
    for t in th:
        t.start()
    go.set()
    for t in th:
        t.join(20)
    ok = sum(1 for r in res if b" 200 " in r)
    assert ok == 256, sorted(set(res))[:5]


# ---------------------------------------------------------------------------- admit trace line (CERT B1 order arm)
def body_for(route, text, n, stream):
    """The ROUTES request with its own prompt text and token limit."""
    path, body = ROUTES[route]
    b = dict(body, stream=stream)
    if route == "completions":
        b["prompt"] = text
    elif route == "responses":
        b["input"] = text
    else:
        b["messages"] = [dict(role="user", content=text)]
    b["max_output_tokens" if route == "responses" else "max_tokens"] = n
    return path, b


def response_id(raw, stream):
    """The id the client got: the body's id, or the first stream event that names one (chunk id, message_start's
    message.id, response.created's response.id)."""
    if not stream:
        return json.loads(raw)["id"]
    for line in raw.decode().splitlines():
        if line.startswith("data: {"):
            d = json.loads(line[6:])
            rid = d.get("id") or (d.get("message") or {}).get("id") or (d.get("response") or {}).get("id")
            if rid:
                return rid
    return None


def admits(lines):
    return [e for e in (json.loads(m) for m in lines if m.startswith("{")) if e.get("event") == "admit"]


def wait_for(cond, s=5.0):
    t = time.time()
    while not cond() and time.time() - t < s:
        time.sleep(0.005)
    return cond()


PREFIX = dict(completions="cmpl-", chat="chatcmpl-", anthropic="msg_", responses="resp_")


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("route", list(ROUTES))
def test_admit_trace_line_names_the_clients_response_id_and_slot_in_admission_order(route, stream, monkeypatch):
    """CERT B1 order arm: under MASHBOX_TRACE=1 the server logs one line per admitted request when the
    loop admits it, {"event": "admit", "rid": <the id this request's client receives>, "slot": <its slot>, "seq": n,
    "tenant": <short tenant id or null>}. Checked against the engine's own admit calls: 5 requests arrive one by one
    at a 2-slot engine whose steps are held, so 4 wait; one more client hangs up while waiting and is never admitted;
    token limits differ, so requests finish in another order than they were admitted."""
    monkeypatch.setenv("MASHBOX_TRACE", "1")
    lines = []
    eng, srv, port = start(S=2, log=lines.append)
    real, got = eng.admit, []

    def admit(reqs, stop_ids=()):       # the engine's own record: (prompt, slot) of each admitted row, in order
        out = real(reqs, stop_ids)
        got.extend((tuple(q["ids"]), r["slot"]) for q, r in zip(reqs, out) if r["fin"] != "error")
        return out
    eng.admit = admit
    eng.gate.clear()
    limits = [6, 1, 4, 2, 3]
    texts = [f"5 6 7 {20 + i}" for i in range(len(limits))]
    res = [None] * len(limits)

    def one(i):
        res[i] = call(port, *body_for(route, texts[i], limits[i], stream))
    ths = []
    for i in range(len(limits)):
        ths.append(threading.Thread(target=one, args=(i,), daemon=True))
        ths[-1].start()
        assert wait_for(lambda: len(got) == 1 if i == 0 else srv.waiting == i), (i, len(got), srv.waiting)
    early = admits(lines)               # request 0 is admitted and running; nothing has finished
    path, b = body_for(route, "5 6 7 99", 3, stream)
    data = json.dumps(b).encode()
    c = socket.create_connection(("127.0.0.1", port), timeout=10)
    c.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
    assert wait_for(lambda: srv.waiting == len(limits))
    c.close()                           # the 6th client leaves while it waits: never admitted
    assert wait_for(lambda: srv.waiting == len(limits) - 1)
    eng.gate.set()
    for t in ths:
        t.join(20)
    assert [r[0] for r in res] == [200] * len(limits), [(r[0], r[1][:200]) for r in res]
    rids = [response_id(r[1], stream) for r in res]
    assert all(isinstance(x, str) and x.startswith(PREFIX[route]) for x in rids), rids
    by_text = {tuple(int(x) for x in t.split()): rids[i] for i, t in enumerate(texts)}
    want = [(by_text[p], s) for p, s in got]
    ev = admits(lines)
    assert [(e["rid"], e["slot"]) for e in ev] == want, (ev, want)
    assert [e["rid"] for e in ev] == rids                   # first come, first served; not the finish order
    assert [e["seq"] for e in ev] == list(range(ev[0]["seq"], ev[0]["seq"] + len(ev)))
    assert all("tenant" in e and e["tenant"] is None for e in ev)       # no API keys: one local owner
    assert [(e["rid"], e["slot"]) for e in early] == [(rids[0], got[0][1])]


def test_admit_trace_line_is_off_without_mashbox_trace(monkeypatch):
    """The admit line is a check hook: without MASHBOX_TRACE=1 the server writes none."""
    monkeypatch.delenv("MASHBOX_TRACE", raising=False)
    lines = []
    eng, srv, port = start(S=2, log=lines.append)
    st, raw, _ = call(port, *body_for("chat", "5 6 7 8", 3, False))
    assert st == 200 and admits(lines) == []


class WholeBatchEngine:
    """generate() only (no admit / step): the server runs whole batches (Server.loop_batches -> run)."""
    CTX, K, prefix_cache, V, S = 4096, 4, False, 1000, 2

    def __init__(self):
        self.batches = []

    def generate(self, prompts, max_new, stop_ids=(), samp=None, on_step=None, **kw):
        self.batches.append([tuple(p) for p in prompts])
        return dict(ids=[[7] * max_new for _ in prompts], finish=[None] * len(prompts), timing={},
                    cached=[0] * len(prompts))


def test_admit_trace_line_on_the_whole_batch_path(monkeypatch):
    """The whole-batch loop admits too: one admit line per request it runs, slot = the request's row in its batch,
    tenant = the first 11 characters of the tenant id ("v1." + 8 hex)."""
    monkeypatch.setenv("MASHBOX_TRACE", "1")
    lines, eng = [], WholeBatchEngine()
    srv = SV.Server(eng, IdChat(), "m", {}, log=lines.append)
    srv.ready = True
    scope = "v1." + "0123456789abcdef" * 2
    jobs = [srv.submit([5, 6, 30 + i], SV.knobs_of(dict(max_tokens=2)), True, False, scope=scope, rid=f"chatcmpl-r{i}")
            for i in range(3)]
    for j in jobs:
        assert j["ev"].wait(10) and "error" not in j
    want = [(f"chatcmpl-r{p[2] - 30}", row) for batch in eng.batches for row, p in enumerate(batch)]
    ev = admits(lines)
    assert [(e["rid"], e["slot"]) for e in ev] == want
    assert all(e["tenant"] == "v1.01234567" for e in ev)
