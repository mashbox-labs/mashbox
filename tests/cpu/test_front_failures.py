"""Failure rules for the HTTP front and its worker process (runtime/front.py, runtime/worker.py, runtime/faults.py).

Each test starts a real worker process (the scripted engine in api_stub_engine.py) behind a real HTTP front, injects one
fault, and checks what clients see and that the server comes back by itself:
- a device fault, out of memory, a hung call (watchdog) or kill -9 restarts the worker;
- only requests whose client has received nothing are sent again (from the start), and only once: a non-streamed request
  until its response is written, even when the worker had made some of its tokens; a stream until a token was sent;
- a stream that already has output ends with the route's error event carrying the response id;
- while no worker is ready, requests get 503 (Anthropic route 529) with Retry-After and x-should-retry: true;
- after the fault the server is ready again with no step by hand, and answers new requests in full.
Crashes covered: mid-stream (raise, oom, kill -9 from inside and from outside, hang) and during the compile warm-up.
A client that stops reading a stream is ended after MASHBOX_STALL_S (INTERFACES §3.3 P6, consumer_stalled) and its row
is freed, while other clients keep being answered.
"""
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from mashbox.runtime import front as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402

OUT = list(range(11, 51))                      # 40 scripted tokens: text "11 12 13 ... 50 "
TEXT = "".join(f"{t} " for t in OUT)
WHOLE = TEXT.rstrip()                          # a chat answer drops the trailing space


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


FRONTS = []


@pytest.fixture(autouse=True)
def _close_fronts():
    """Each test's fronts end with it: Front.close() ends their worker processes (none outlives its test)."""
    yield
    while FRONTS:
        FRONTS.pop().close()


def start(tmp_path, fault="", S=1, step_s=0.03, warm=False, outs=None, lives=None, extra_env=None, chat=None, kw=None):
    log = str(tmp_path / "engine.jsonl")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]
                                                      + [os.environ.get("PYTHONPATH", "")]),
               MASHBOX_FAULT=fault, MASHBOX_FAULT_STATE=str(tmp_path / "faults"), JAX_PLATFORMS="cpu",
               MASHBOX_WATCHDOG_MIN_S="1")
    env.update(extra_env or {})
    logs = []
    fr = F.Front(chat or IdChat(), "stub-model", {}, "api_stub_engine:make",
                 dict(S=S, step_s=step_s, outs=outs or [OUT], log=log, **(kw or {})), warm=warm, env=env, log=logs.append)
    FRONTS.append(fr)
    port = STUB.serve_on_free_port(fr)
    return fr, port, log, logs


def wait_ready(fr, s=30):
    t = time.time()
    while not fr.ready and time.time() - t < s:
        time.sleep(0.02)
    assert fr.ready, "the server did not come back ready by itself"


def post(port, path, body, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e.headers


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def events(raw):
    out = []
    for blk in raw.split("\n\n"):
        name = next((ln[7:] for ln in blk.split("\n") if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in blk.split("\n") if ln.startswith("data: ")), None)
        if data is not None:
            out.append((name, data if data == "[DONE]" else json.loads(data)))
    return out


CHAT = ("/v1/chat/completions", dict(model="stub-model", max_tokens=40, stream=True, messages=[dict(role="user", content="1 2 3")]))
ANTH = ("/v1/messages", dict(model="stub-model", max_tokens=40, stream=True, messages=[dict(role="user", content="1 2 3")]))
RESP = ("/v1/responses", dict(model="stub-model", max_output_tokens=40, stream=True, input="1 2 3"))


def chat_text(ev):
    return "".join((e["choices"][0]["delta"].get("content") or "") for _, e in ev if isinstance(e, dict) and e.get("choices"))


def full_answer_ok(port):
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False))
    assert code == 200, raw
    r = json.loads(raw)
    assert r["choices"][0]["message"]["content"] == WHOLE and r["usage"]["completion_tokens"] == len(OUT)


@pytest.mark.parametrize("kind", ["raise", "oom", "kill", "hang"])
def test_crash_mid_stream_ends_stream_with_error_event_and_recovers(tmp_path, kind):
    """A stream with output gets text, then one error event with its response id; the worker restarts by itself."""
    fr, port, log, logs = start(tmp_path, fault=f"step:6:{kind}")
    wait_ready(fr)
    pid0 = fr.link.proc.pid
    code, raw, h = post(port, *CHAT)
    assert code == 200
    ev = events(raw)
    errs = [e for _, e in ev if isinstance(e, dict) and "error" in e]
    got = chat_text(ev)
    assert got and TEXT.startswith(got) and got != TEXT, got            # output, cut off, nothing invented
    assert len(errs) == 1 and errs[0]["error"]["type"] == "server_error" and errs[0]["request_id"].startswith("chatcmpl-")
    assert ev[-1][1] == "[DONE]" and not [e for _, e in ev if isinstance(e, dict) and e.get("choices")
                                          and e["choices"][0].get("finish_reason")]   # no fake finish_reason
    wait_ready(fr)
    assert fr.link.proc.pid != pid0
    full_answer_ok(port)
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert down and down[0]["fault"] == {"raise": "device_fault", "oom": "oom", "kill": None, "hang": "watchdog"}[kind]


def test_kill_9_from_outside_mid_stream(tmp_path):
    """kill -9 of the worker process (the kernel's out-of-memory killer, an operator) mid-answer, on all three APIs."""
    fr, port, log, logs = start(tmp_path, step_s=0.05)
    for path, body in (CHAT, ANTH, RESP):
        wait_ready(fr)
        out = {}
        t = threading.Thread(target=lambda: out.update(r=post(port, path, body)))
        t.start()
        t0 = time.time()
        while not [e for e in STUB.read_log(log) if e["ev"] == "admit" and e["pid"] == fr.link.proc.pid] and time.time() - t0 < 10:
            time.sleep(0.01)
        time.sleep(0.4)                                      # several tokens out
        os.kill(fr.link.proc.pid, signal.SIGKILL)
        t.join(30)
        code, raw, _ = out["r"]
        ev = events(raw)
        assert code == 200
        if path == ANTH[0]:
            text = "".join(e["delta"].get("text", "") for n, e in ev if n == "content_block_delta")
            err = [e for n, e in ev if n == "error"]
            assert len(err) == 1 and err[0]["error"]["type"] == "api_error" and err[0]["request_id"].startswith("msg_")
            assert not [e for n, e in ev if n == "message_stop"]
        elif path == RESP[0]:
            text = "".join(e["delta"] for n, e in ev if n == "response.output_text.delta")
            failed = [e for n, e in ev if n == "response.failed"]
            assert [n for n, _ in ev][-2:] == ["error", "response.failed"]
            assert failed[0]["response"]["status"] == "failed" and failed[0]["response"]["id"].startswith("resp_")
            assert not [e for n, e in ev if n == "response.completed"]
        else:
            text = chat_text(ev)
            assert [e for _, e in ev if isinstance(e, dict) and "error" in e]
        assert text and TEXT.startswith(text) and text != TEXT, (path, text)
    wait_ready(fr)
    full_answer_ok(port)


def test_request_without_output_is_retried_once_and_completes(tmp_path):
    """Two live rows: A has output, B was just admitted (no token yet) when the device call fails. A ends with an error
    event; B runs again from the start on the restarted worker and gets the whole answer, as if nothing happened."""
    fr, port, log, logs = start(tmp_path, fault="admit:2:raise", S=2, step_s=0.05)
    wait_ready(fr)
    a = {}
    ta = threading.Thread(target=lambda: a.update(r=post(port, *CHAT)))
    ta.start()
    time.sleep(0.5)                                          # A streams
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False))     # B: admit #2 fails
    ta.join(30)
    assert code == 200, raw
    assert json.loads(raw)["choices"][0]["message"]["content"] == WHOLE
    ev = events(a["r"][1])
    assert chat_text(ev) and [e for _, e in ev if isinstance(e, dict) and "error" in e]
    lost = [json.loads(m) for m in logs if '"worker_lost_requests"' in m]
    assert lost == [dict(event="worker_lost_requests", code="device_fault", retried=1, ended=1)]


def test_retry_that_fails_again_gets_500(tmp_path):
    """The one retry fails on a device fault too: 500 (OpenAI server_error, Anthropic api_error), never a third try."""
    fr, port, log, logs = start(tmp_path, fault="admit:1:raise:2")
    wait_ready(fr)
    code, raw, h = post(port, CHAT[0], dict(CHAT[1], stream=False))
    assert code == 500 and json.loads(raw)["error"]["type"] == "server_error"
    admits = [e for e in STUB.read_log(log) if e["ev"] == "admit"]
    assert len({e["pid"] for e in admits}) == 0           # both admits failed before the engine took the row
    wait_ready(fr)
    starts = [m for m in logs if '"worker_ready"' in m]
    assert len(starts) == 3                                  # the first worker, then two restarts
    (tmp_path / "b").mkdir()
    fr2, port2, log2, logs2 = start(tmp_path / "b", fault="admit:1:raise:2")
    wait_ready(fr2)
    code, raw, h = post(port2, ANTH[0], dict(ANTH[1], stream=False))
    assert code == 500 and json.loads(raw)["error"]["type"] == "api_error"
    wait_ready(fr)
    full_answer_ok(port)


def tokens_seen(fr):
    """The TOKENS frames the front gets from its workers from now on: [(pid, number of ids)]."""
    seen, real = [], fr.on_frame

    def on_frame(link, m):
        if m.get("t") == "TOKENS":
            seen.append((link.proc.pid, len(m.get("ids") or [])))
        return real(link, m)
    fr.on_frame = on_frame                        # read_loop looks the method up for each frame
    return seen


def test_non_streamed_request_whose_worker_made_tokens_is_sent_again_and_completes(tmp_path):
    """A13: a request whose client has received no bytes is sent once more. Not streamed, the client receives nothing
    until the answer ends, so the worker's first tokens (here 5, then kill -9 in step 6) leave the request eligible: it
    runs again from the start on the restarted worker and its client gets the whole answer, once. Old code: 500 (the
    front counted the worker's tokens as output the client had)."""
    fr, port, log, logs = start(tmp_path, fault="step:6:kill")
    wait_ready(fr)
    seen = tokens_seen(fr)
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False))
    assert code == 200, raw
    assert json.loads(raw)["choices"][0]["message"]["content"] == WHOLE
    pids = list(dict.fromkeys(p for p, _ in seen))
    assert len(pids) == 2 and sum(n for p, n in seen if p == pids[0]) >= 5, seen     # the first life made tokens
    assert sum(n for p, n in seen if p == pids[1]) == len(OUT), seen                # the second made the answer
    lost = [json.loads(m) for m in logs if '"worker_lost_requests"' in m]
    assert lost == [dict(event="worker_lost_requests", code="worker_exit", retried=1, ended=0)], lost
    full_answer_ok(port)


def held_at_publication(fr):
    """Holds each request's own thread right after Server.submit puts its job on the front's queue (the one call that
    publishes it; the front's loop takes the job from there and runs it) until release is set. -> (state, release):
    state["jobs"] = the jobs published, state["held"] = True while a request thread is held there."""
    real, state, release = fr.q.put, dict(jobs=[], held=False), threading.Event()

    def put(job, *a, **kw):
        real(job, *a, **kw)
        state["jobs"].append(job)
        state["held"] = True
        release.wait(60)
        state["held"] = False
    fr.q.put = put                                # an instance attribute: Server.submit's self.q.put finds it first
    return state, release


def answer_text(route, raw, stream):
    if route == "chat":
        return chat_text(events(raw)) if stream else json.loads(raw)["choices"][0]["message"]["content"]
    if stream:
        return "".join(e["delta"].get("text") or "" for n, e in events(raw) if n == "content_block_delta")
    return "".join(b.get("text") or "" for b in json.loads(raw)["content"] if b.get("type") == "text")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["chat", "anthropic"])
def test_retry_rule_reads_the_clients_stream_mode_from_the_job_as_it_is_published(tmp_path, route, stream):
    """The front may run a job, get tokens and lose its worker before Server.submit returns to its route, so the
    client's stream mode must be in the job when it is queued. The request's thread is held right after the queue put,
    before submit returns; meanwhile the worker makes 5 tokens, is killed in step 6, and the front decides. Not
    streamed: the client has received nothing, so the request runs once more from the start and its client gets the
    whole answer (200). Streamed: its tokens are queued for its handler, so it is not sent again; once released, the
    client gets that output and then the route's error event. Old code (each route marked the stream mode after submit
    returned, and an unmarked job counts as a stream): the request that is not streamed got 500."""
    fr, port, log, logs = start(tmp_path, fault="step:6:kill")
    wait_ready(fr)
    path, body = {"chat": CHAT, "anthropic": ANTH}[route]
    seen = tokens_seen(fr)
    state, release = held_at_publication(fr)
    res = {}
    th = threading.Thread(target=lambda: res.update(r=post(port, path, dict(body, stream=stream))))
    th.start()
    t = time.time()
    while not [m for m in logs if '"worker_lost_requests"' in m] and time.time() - t < 30:
        time.sleep(0.02)
    lost = [json.loads(m) for m in logs if '"worker_lost_requests"' in m]
    held = state["held"]
    release.set()
    th.join(60)
    assert held and len(state["jobs"]) == 1, (held, state["jobs"])   # the front decided while the thread was held
    assert state["jobs"][0].get("streamed") is stream
    seen = list(seen)                             # the token frames until this request ended (before the next one)
    pids = list(dict.fromkeys(p for p, _ in seen))
    assert len(pids) >= 1 and sum(n for p, n in seen if p == pids[0]) >= 5, seen      # the lost worker made tokens
    code, raw, _ = res["r"]
    assert code == 200, raw
    wait_ready(fr)
    code0, ref, _ = post(port, path, dict(body, stream=False))       # the same request on the healthy worker
    assert code0 == 200, ref
    whole = answer_text(route, ref, False)
    assert whole and TEXT.startswith(whole), whole
    if not stream:
        assert lost == [dict(event="worker_lost_requests", code="worker_exit", retried=1, ended=0)], lost
        assert answer_text(route, raw, False) == whole, raw
        assert len(pids) == 2 and sum(n for p, n in seen if p == pids[1]) == len(OUT), seen     # run once more, once
    else:
        assert lost == [dict(event="worker_lost_requests", code="worker_exit", retried=0, ended=1)], lost
        ev = events(raw)
        got = answer_text(route, raw, True)
        assert got and whole.startswith(got.rstrip()) and got.rstrip() != whole, (got, raw[-600:])   # cut off
        assert [e for n, e in ev if n == "error" or (isinstance(e, dict) and "error" in e)], raw[-600:]
        assert len(pids) == 1, seen                                                       # never sent again
    full_answer_ok(port)


def test_crash_during_compile_warm_up_comes_back_ready(tmp_path):
    """The worker dies while it compiles (kill -9 inside warm()). Until a worker is ready, requests get 503 / 529 with
    Retry-After and x-should-retry: true; the next worker comes up by itself and answers."""
    fr, port, log, logs = start(tmp_path, fault="warm:1:kill", warm=True)
    saw = {"503": None, "529": None}
    t = time.time()
    while not fr.ready and time.time() - t < 30:
        for code_want, (path, body) in (("503", CHAT), ("529", ANTH)):
            code, raw, h = post(port, path, dict(body, stream=False))
            if not fr.ready and str(code) == code_want:
                saw[code_want] = (raw, h.get("Retry-After"), h.get("x-should-retry"))
        time.sleep(0.01)
    wait_ready(fr)
    assert saw["503"] and saw["503"][1] and saw["503"][2] == "true", saw
    assert saw["529"] and json.loads(saw["529"][0])["error"]["type"] == "overloaded_error" and saw["529"][2] == "true"
    warms = [e for e in STUB.read_log(log) if e["ev"] == "warm"]
    assert len(warms) == 1                                   # only the second worker finished its warm-up
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert down and down[0]["was_ready"] is False and down[0]["status"] == -signal.SIGKILL
    full_answer_ok(port)


@pytest.mark.parametrize("kind", ["raise", "kill"])
def test_requests_waiting_in_the_front_survive_a_crash(tmp_path, kind):
    """With one row, B waits in the front's queue while A streams; A's worker dies (a device fault inside it, or kill -9
    from outside); B is not touched by it and gets its whole answer from the next worker. The crash comes only once A
    has output and B is in the front's queue (the engine holds A's row after 4 tokens until the test says go), so the
    test does not depend on how fast this machine is."""
    go = str(tmp_path / "go")
    fr, port, log, logs = start(tmp_path, step_s=0.03,
                                kw=dict(hold=dict(after=4, path=go, then="raise" if kind == "raise" else "go")))
    wait_ready(fr)
    pid0 = fr.link.proc.pid
    a, b = {}, {}
    ta = threading.Thread(target=lambda: a.update(r=post(port, *CHAT)))
    ta.start()
    t0 = time.time()
    while not any(j.get("got") for j in list(fr.link.inflight.values())) and time.time() - t0 < 30:
        time.sleep(0.01)                                     # A streams: it has output
    tb = threading.Thread(target=lambda: b.update(r=post(port, CHAT[0], dict(CHAT[1], stream=False))))
    tb.start()
    while len(fr.pending) < 1 and time.time() - t0 < 30:
        time.sleep(0.01)                                     # B waits in the front's queue for the one row
    with fr.lock:
        held = [j for j in fr.link.inflight.values()]
        assert fr.link.proc.pid == pid0 and len(held) == 1 and held[0].get("got") and len(fr.pending) == 1 \
            and fr.q.qsize() == 0, (f"precondition: A in the worker with output and B waiting in the front before the "
                                    f"crash; in worker {len(held)}, got {[j.get('got') for j in held]}, waiting "
                                    f"{len(fr.pending)} + {fr.q.qsize()}")
    if kind == "raise":
        open(go, "w").close()                                # the worker's next step raises a device fault
    else:
        os.kill(pid0, signal.SIGKILL)
        open(go, "w").close()                                # the next worker does not hold B
    ta.join(30)
    tb.join(30)
    assert [e for _, e in events(a["r"][1]) if isinstance(e, dict) and "error" in e]
    assert b["r"][0] == 200 and json.loads(b["r"][1])["choices"][0]["message"]["content"] == WHOLE
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert down and down[0]["pid"] == pid0 and down[0]["fault"] == {"raise": "device_fault", "kill": None}[kind], down
    admits = [e["pid"] for e in STUB.read_log(log) if e["ev"] == "admit"]
    assert admits.count(pid0) == 1 and len(admits) == 2 and admits[1] != pid0, admits   # B never reached the dead worker


def test_close_ends_the_worker_and_its_requests(tmp_path):
    """Front.close(): a live stream ends with its error event, a waiting request gets 503, the worker process exits and
    no new one starts; a request after close gets 503 at once."""
    fr, port, log, logs = start(tmp_path, S=1, step_s=0.05)
    wait_ready(fr)
    proc = fr.link.proc
    a, b = {}, {}
    ta = threading.Thread(target=lambda: a.update(r=post(port, *CHAT)))
    ta.start()
    t0 = time.time()
    while not any(j.get("got") for j in list(fr.link.inflight.values())) and time.time() - t0 < 20:
        time.sleep(0.01)                                     # A streams: it has output
    tb = threading.Thread(target=lambda: b.update(r=post(port, CHAT[0], dict(CHAT[1], stream=False))))
    tb.start()                                               # B waits for the one row
    while len(fr.pending) + fr.q.qsize() < 1 and time.time() - t0 < 20:
        time.sleep(0.01)
    n0 = fr.workers_started
    fr.close()
    ta.join(30)
    tb.join(30)
    assert proc.poll() is not None, "the worker outlived close()"
    ev = events(a["r"][1])
    assert chat_text(ev) and [e for _, e in ev if isinstance(e, dict) and "error" in e]
    assert b["r"][0] == 503 and b["r"][2].get("x-should-retry") == "true", b["r"][:2]
    time.sleep(1.0)
    assert fr.workers_started == n0 and fr.link is None and not fr.ready
    code, raw, h = post(port, CHAT[0], dict(CHAT[1], stream=False), timeout=10)
    assert code == 503, raw


# ---- P6 consumer_stalled (INTERFACES §3.3): a client that stops reading a stream ---------------------------------------
W_BIG, N_BIG = 8192, 2000
BIG = ("y" * (W_BIG - 1) + " ") * N_BIG


class BigChat(IdChat):
    """Each output token is 8 KiB of text, so a few hundred tokens fill a connection's buffers (one slice per decode:
    Text re-decodes the whole answer at every event)."""

    def decode(self, ids):
        return BIG[:len(ids) * W_BIG]


STALL_ROUTES = {"chat": (CHAT[0], dict(CHAT[1], max_tokens=N_BIG)),
                "completions": ("/v1/completions", dict(model="stub-model", max_tokens=N_BIG, stream=True, prompt="1 2 3")),
                "anthropic": (ANTH[0], dict(ANTH[1], max_tokens=N_BIG))}


@pytest.mark.parametrize("route", sorted(STALL_ROUTES))
def test_client_that_stops_reading_is_ended_and_its_row_freed(tmp_path, monkeypatch, route):
    """P6: a client asks for a long stream with a 4 KiB receive window and never reads. Once a write cannot finish for
    MASHBOX_STALL_S seconds, the request ends as a hang-up (log event consumer_stalled), the worker cancels its row at
    the next step (long before max_tokens), the server closes that connection, and another client is answered in full
    the whole time."""
    stall = 2.0
    monkeypatch.setenv("MASHBOX_STALL_S", str(stall))
    fr, port, log, logs = start(tmp_path, S=2, step_s=0.005, outs=[[5] * N_BIG], chat=BigChat())
    wait_ready(fr)
    path, body = STALL_ROUTES[route]
    raw = json.dumps(body).encode()
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.connect(("127.0.0.1", port))
    try:
        s.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
                  f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)                  # ... and never reads
        t0, big, served = time.time(), [], 0
        while not big and time.time() - t0 < stall + 15:
            code, out, _ = post(port, CHAT[0], dict(CHAT[1], stream=False, max_tokens=3), timeout=30)
            assert code == 200 and json.loads(out)["usage"]["completion_tokens"] == 3, (code, out[:200])
            served += 1
            big = [e for e in STUB.read_log(log) if e["ev"] == "end" and e["n"] > 3]      # the long row's end
        assert big, f"the stalled client's row still runs {time.time() - t0:.1f} s after its request"
        assert big[0]["fin"] == "cancel" and big[0]["n"] < N_BIG, big[0]
        assert served >= 1
        stalled = [json.loads(m) for m in logs if '"consumer_stalled"' in m]
        assert stalled and stalled[0]["stall_s"] == stall, logs[-5:]
        s.settimeout(15)                        # the server has closed this connection: reading drains it to its end
        while s.recv(1 << 20):
            pass
    finally:
        s.close()
    code, out, _ = post(port, CHAT[0], dict(CHAT[1], stream=False, max_tokens=3))
    assert code == 200


def test_keep_alive_connection_outlives_the_write_timeout(tmp_path, monkeypatch):
    """The write timeout covers an answer only: a keep-alive connection that sits idle longer than MASHBOX_STALL_S after
    an answer still takes the next request (the connection's own timeout is restored after each answer)."""
    import http.client
    monkeypatch.setenv("MASHBOX_STALL_S", "1")
    fr, port, log, logs = start(tmp_path, S=1, step_s=0.0)
    wait_ready(fr)
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        for i in range(2):
            c.request("POST", CHAT[0], json.dumps(dict(CHAT[1], stream=False)), {"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
            assert r.status == 200 and json.loads(body)["choices"][0]["message"]["content"] == WHOLE, (i, body[:200])
            if i == 0:
                time.sleep(2.5)                   # idle on the open connection, past the 1 s write timeout
    finally:
        c.close()
    assert not [m for m in logs if '"consumer_stalled"' in m]


def test_health_reports_the_workers_engine_and_compiles_over_all_lives(tmp_path):
    """/health's load fields come from the worker's engine, not the front's stand-in (deployment id, component set,
    cache namespace, live rows), and compiles_after_ready is the worker's own count summed over every worker life: a
    worker killed with -9 never takes its compiles with it. The front itself never compiles. The front prints the
    "ready" line once, when it first takes requests (a restarted worker logs worker_ready only)."""
    fr, port, log, logs = start(tmp_path, kw=dict(compile_each_admit=True, deployment_id="dep-test",
                                                  component_set_sha="cs-test", cache_ns="ns-test"))
    wait_ready(fr)
    h = get(port, "/health")
    assert (h["deployment_id"], h["component_set_sha"], h["cache_namespace"]) == ("dep-test", "cs-test", "ns-test"), h
    assert h["live_rows"] == 0 and h["compiles_after_ready"] == 0, h
    lives = []
    for life in range(2):
        pid = fr.link.proc.pid
        full_answer_ok(port)
        got = [e["after"] for e in STUB.read_log(log) if e["ev"] == "compiled" and e["pid"] == pid]
        assert got and got[-1] >= 1, got                     # this worker compiled after it said ready
        lives.append(got[-1])
        t = time.time()
        while get(port, "/health")["compiles_after_ready"] != sum(lives) and time.time() - t < 10:
            time.sleep(0.05)
        h = get(port, "/health")
        assert h["compiles_after_ready"] == sum(lives), (h, lives)
        if life == 0:
            os.kill(pid, signal.SIGKILL)
            t = time.time()
            while not (fr.link is not None and fr.link.proc.pid != pid) and time.time() - t < 30:
                time.sleep(0.02)
            wait_ready(fr)
            h = get(port, "/health")
            assert h["worker_restarts"] == 1 and h["compiles_after_ready"] == lives[0], (h, lives)
            assert h["deployment_id"] == "dep-test", h
    ready = [json.loads(m) for m in logs if '"event": "ready"' in m]     # the line the docs say to wait for: once
    assert len(ready) == 1 and ready[0]["seconds_since_start"] >= 0, logs


def test_models_list_shows_the_workers_weights_info(tmp_path):
    """/v1/models carries the served weights' fields (file, size, bits per weight, slots), which only the worker knows."""
    info = dict(weights="stub-q4.bshj", gb=1.25, bpw=4.5, prefix_cache=False, slots=1, cache_ram_mib=0)
    fr, port, log, logs = start(tmp_path, kw=dict(serve_info=info))
    wait_ready(fr)
    m = get(port, "/v1/models")["data"][0]
    assert m["id"] == "stub-model" and {k: m.get(k) for k in info} == info, m


def test_owner_and_edit_variant_reach_the_engine_through_the_worker(tmp_path):
    """A request's owner (its tenant scope) and its edit variant (the hash of its activation-edit table) are in its
    prefix-cache key, so both must reach engine.admit inside the worker exactly as the one-process server passes them."""
    from mashbox.runtime import server as SV
    fr, port, log, logs = start(tmp_path)
    wait_ready(fr)
    j = fr.submit([1, 2, 3], SV.knobs_of(dict(max_tokens=3)), False, False, scope="tenant-a", variant="edit-v1")
    assert j["ev"].wait(30)
    adm = [e for e in STUB.read_log(log) if e["ev"] == "admit"]
    assert adm and (adm[-1]["scope"], adm[-1]["variant"]) == ("tenant-a", "edit-v1"), adm


ADMIT_ROUTES = [CHAT, ("/v1/completions", dict(model="stub-model", max_tokens=40, stream=True, prompt="1 2 3")), ANTH, RESP]


def response_id(raw, stream):
    """The id the client got: the body's id, or the first stream event that names one (chunk id, message_start's
    message.id, response.created's response.id)."""
    if not stream:
        return json.loads(raw)["id"]
    for _, d in events(raw):
        if isinstance(d, dict):
            rid = d.get("id") or (d.get("message") or {}).get("id") or (d.get("response") or {}).get("id")
            if rid:
                return rid
    return None


def admit_lines(err):
    """The MASHBOX_TRACE admit lines on stderr: the worker writes them (its stderr is the server's)."""
    out = []
    for ln in err.splitlines():
        if ln.startswith("{") and '"admit"' in ln:
            e = json.loads(ln)
            if e.get("event") == "admit":
                out.append(e)
    return out


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_admit_trace_line_names_the_clients_response_id_behind_the_worker(tmp_path, capfd, stream):
    """MASHBOX_TRACE=1 behind the front: the worker's loop admits the rows, so its admit line must name the id each
    client receives, as the one-process server's line does (CERT B1 order arm). The id crosses the SUBMIT frame. One
    request per generating route (chat, text completions, Anthropic messages, Responses), one after another: one line
    each, in that order, seq 1..4, tenant null for the one local owner."""
    fr, port, log, logs = start(tmp_path, S=2, extra_env=dict(MASHBOX_TRACE="1"))
    wait_ready(fr)
    want = []
    for path, body in ADMIT_ROUTES:
        code, raw, _ = post(port, path, dict(body, stream=stream))
        assert code == 200, raw
        want.append(response_id(raw, stream))
    assert all(want) and len(set(want)) == 4, want
    got = admit_lines(capfd.readouterr().err)
    assert [e["rid"] for e in got] == want, got
    assert [e["seq"] for e in got] == [1, 2, 3, 4] and all(e["tenant"] is None for e in got), got
    assert not [m for m in logs if '"worker_down"' in m], logs


def test_admit_trace_line_of_a_retried_request_names_the_same_response_id(tmp_path, capfd):
    """A request with no output is sent again after the worker restarts (A13); its admit line in the new worker's life
    names the same id its client receives, so a trace reader can follow it across the restart."""
    fr, port, log, logs = start(tmp_path, fault="admit:2:raise", S=2, step_s=0.05, extra_env=dict(MASHBOX_TRACE="1"))
    wait_ready(fr)
    a = {}
    ta = threading.Thread(target=lambda: a.update(r=post(port, *CHAT)))
    ta.start()
    time.sleep(0.5)                                          # A streams
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False))     # B: admit #2 fails, then B runs again
    ta.join(30)
    assert code == 200, raw
    lost = [json.loads(m) for m in logs if '"worker_lost_requests"' in m]
    assert lost == [dict(event="worker_lost_requests", code="device_fault", retried=1, ended=1)]
    got = admit_lines(capfd.readouterr().err)
    assert [e["rid"] for e in got] == [response_id(a["r"][1], True), json.loads(raw)["id"]], got
    assert [e["seq"] for e in got] == [1, 1], got                     # one line in each worker life


def test_serve_cli_keeps_the_front_off_the_chip_and_gives_the_worker_the_command(monkeypatch):
    """`mashbox run` serving (cli.cmd_run -> front.serve_cli): the worker gets the environment the user started with
    (its chip platform) and the same command line, so it loads and warms the engine as the one-process server did; this
    process, the front, is kept on the CPU, so it can never hold the chip the worker needs."""
    import argparse
    import jax
    from mashbox import hub
    from mashbox.runtime import chat as CH
    seen = {}
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setattr(hub, "model_entry", lambda mid: dict(id="qwen3-8-27b"))
    monkeypatch.setattr(hub, "upstream_dir", lambda m, log=print: "/no/such/dir")
    monkeypatch.setattr(CH, "Chat", lambda d: ("chat", d))
    monkeypatch.setattr(F, "serve_split", lambda *a, **k: seen.update(a=a, k=k))
    before = jax.config.jax_platforms
    try:
        a = argparse.Namespace(model="qwen3-8-27b", host="127.0.0.1", port=8123, ui=False)
        argv = ["run", "qwen3-8-27b", "--port", "8123"]
        F.serve_cli(a, argv, ["key-1"], print, t0=1.0)
        assert os.environ["JAX_PLATFORMS"] == "cpu" and jax.config.jax_platforms == "cpu"
    finally:
        jax.config.update("jax_platforms", before)
    chat, mid, info, factory, kw = seen["a"]
    assert (chat, mid, factory, kw) == (("chat", "/no/such/dir"), "qwen3-8-27b", "mashbox.runtime.worker:from_cli",
                                        dict(argv=argv))
    k = seen["k"]
    assert k["env"]["JAX_PLATFORMS"] == "tpu" and (k["host"], k["port"], k["keys"], k["t0"]) == ("127.0.0.1", 8123, ["key-1"], 1.0)


# ------------------------------------------------------------------------------- many connections, half-sent requests
def _fill_fds(n=1100):
    """Hold n descriptors in this process (the server runs here), so the next socket the server accepts is numbered
    1024 or more (the kernel gives the lowest free number)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    need = n + 400
    if soft != resource.RLIM_INFINITY and soft < need:
        if hard != resource.RLIM_INFINITY and hard < need:
            pytest.skip(f"this process may open only {hard} files; the check needs {need}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (need, hard))
    fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(n)]
    probe = os.open(os.devnull, os.O_RDONLY)
    os.close(probe)
    assert probe >= 1024, probe
    return fds


@pytest.mark.parametrize("mode", ["one_process", "worker"])
def test_answers_stay_whole_when_the_server_holds_over_1024_open_files(tmp_path, mode):
    """A server holding more than 1,024 open files (connections, logs) answers in full: a stream gets every token, its
    finish reason and [DONE], a non-streamed answer gets 200. The client-gone check must not read a socket numbered
    1024 or more as a client that hung up (select() cannot watch it; poll() can)."""
    if mode == "worker":
        fr, port, log, logs = start(tmp_path, S=1, step_s=0.0)
        wait_ready(fr)
    else:
        from mashbox.runtime import server as SV
        logs = []
        fr = SV.Server(STUB.make(S=1, outs=[OUT]), IdChat(), "stub-model", {}, log=logs.append)
        fr.ready = True
        port = STUB.serve_on_free_port(fr)
    fds = _fill_fds()
    try:
        code, raw, _ = post(port, *CHAT)
        ev = events(raw)
        assert code == 200 and chat_text(ev).rstrip() == WHOLE, (code, raw[:300])
        fins = [e["choices"][0].get("finish_reason") for _, e in ev if isinstance(e, dict) and e.get("choices")]
        assert fins[-1] == "length" and ev[-1] == (None, "[DONE]"), raw[-300:]
        full_answer_ok(port)
    finally:
        for fd in fds:
            os.close(fd)
    assert not [m for m in logs if '"client_closed"' in m], logs


def _held_open_s(port, first, then=None, every=0.25, limit=10.0):
    """Send `first`, then (if given) one byte of `then` every `every` seconds, until the server closes the connection;
    seconds until it closed, or None if it was still open after `limit`."""
    s = socket.create_connection(("127.0.0.1", port), timeout=limit)
    t0, i = time.time(), 0
    try:
        s.sendall(first)
        s.settimeout(every if then else limit)
        while time.time() - t0 < limit:
            try:
                if s.recv(4096) == b"":
                    return time.time() - t0
            except socket.timeout:
                if then:
                    try:
                        s.sendall(then[i % len(then):i % len(then) + 1])
                    except OSError:
                        return time.time() - t0
                    i += 1
            except OSError:
                return time.time() - t0
        return None
    finally:
        s.close()


HALF = {
    "idle": (b"", None),
    "half_headers": (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n", None),
    "half_body": (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: 200\r\n\r\n{\"model\":", None),
    "byte_drip": (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Pad: ", b"a"),
    "line_drip": (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n", b"X-Pad: a\r\n"),
}


@pytest.mark.parametrize("kind", list(HALF))
def test_idle_and_half_sent_requests_are_closed_after_the_read_timeout(tmp_path, monkeypatch, kind):
    """A connection that sends nothing, half a request (headers or body), or a request a byte or a line at a time is
    closed once MASHBOX_READ_S has passed (first byte, then the whole request), so such connections cannot pile up and
    starve the server. A half-sent request is logged as request_read_timeout; whole requests are answered as before."""
    monkeypatch.setenv("MASHBOX_READ_S", "1")
    fr, port, log, logs = start(tmp_path, S=1, step_s=0.0)
    wait_ready(fr)
    took = _held_open_s(port, *HALF[kind])
    assert took is not None and took < 6.0, f"{kind}: still open after 10 s"
    full_answer_ok(port)
    timeouts = [json.loads(m) for m in logs if '"request_read_timeout"' in m]
    assert (len(timeouts) == 1 and timeouts[0]["bytes"] > 0) if kind != "idle" else not timeouts, timeouts


# ------------------------------------------------------------------------------- a worker that never gets ready
def test_a_load_that_fails_every_time_ends_the_server(tmp_path):
    """INTERFACES §3.5: a load that fails in every worker life (a wrong path, too small a chip) ends the server after
    START_TRIES lives instead of answering 503 forever: the front gives up with the worker's own error (code
    load_failed), ends what waits, and stops serving HTTP."""
    from mashbox.runtime import protocol as P
    fr, port, log, logs = start(tmp_path, fault="load:1:raise:99")
    t = time.time()
    while getattr(fr, "gave_up", None) is None and time.time() - t < 30:
        time.sleep(0.05)
    starts = fr.workers_started
    assert getattr(fr, "gave_up", None), f"still restarting after 30 s: {starts} worker starts, no give-up"
    assert "InjectedFault" in fr.gave_up and "load_failed" in fr.gave_up, fr.gave_up
    assert starts == P.START_TRIES and fr.closed.is_set(), (starts, fr.gave_up)
    t = time.time()
    while time.time() - t < 10:                     # the HTTP server stops: nothing listens on the port any more
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            time.sleep(0.1)
        except OSError:
            break
    else:
        raise AssertionError("the HTTP server still accepts connections after the front gave up")
    time.sleep(1.0)
    assert fr.workers_started == starts             # no worker started after the give-up


def test_serve_split_exits_non_zero_with_the_workers_error(tmp_path):
    """What `mashbox run` does through serve_split when the model never loads: the process exits non-zero with the
    worker's own error on stderr, within a few seconds (not an endless 503)."""
    import subprocess
    code = ("import sys, api_stub_engine as S; from mashbox.runtime import front as F; "
            "F.serve_split(S.IdChat(), 'stub-model', {}, 'api_stub_engine:make', dict(S=1), port=int(sys.argv[1]))")
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]
                                                      + [os.environ.get("PYTHONPATH", "")]),
               MASHBOX_FAULT="load:1:raise:99", MASHBOX_FAULT_STATE=str(tmp_path / "faults"), JAX_PLATFORMS="cpu")
    try:
        p = subprocess.run([sys.executable, "-c", code, str(port)], env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError("serve_split still serving 60 s after every worker load failed")
    assert p.returncode != 0 and "mashbox: " in p.stderr and "InjectedFault" in p.stderr, (p.returncode, p.stderr[-800:])


def test_a_restart_slower_than_twice_the_first_start_ends_the_server(tmp_path):
    """INTERFACES §3.5: after a fault, a worker that takes longer than START_SLOW_X x the first start to get ready
    (MASHBOX_START_FLOOR_S lowered to 1 s here) ends the server with that reason, instead of hanging in 503."""
    fr, port, log, logs = start(tmp_path, fault="step:1:kill", extra_env=dict(MASHBOX_START_FLOOR_S="1"),
                                kw=dict(life_dir=str(tmp_path / "lives"), slow_from_life=2, slow_s=60))
    wait_ready(fr)
    threading.Thread(target=post, args=(port, CHAT[0], dict(CHAT[1], stream=False), 30), daemon=True).start()
    t = time.time()
    while getattr(fr, "gave_up", None) is None and time.time() - t < 30:
        time.sleep(0.05)
    assert getattr(fr, "gave_up", None), "a restart 60 s slow did not end the server within 30 s"
    assert "not ready after" in fr.gave_up and fr.workers_started == 2, (fr.gave_up, fr.workers_started)
    assert [json.loads(m) for m in logs if '"worker_slow_start"' in m]


def test_server_start_raises_the_open_files_soft_limit_to_its_hard_limit():
    """A server started under the usual low soft limit on open files (Ubuntu's 1,024; here 256 under a hard limit of
    4,096) raises it to the hard limit and logs it, so connections past the soft limit are not refused (EMFILE)."""
    import resource
    import subprocess
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    want = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
    assert want > 256, f"this machine's hard limit on open files is {hard}"
    code = (
        "import json, resource, sys, threading\n"
        f"resource.setrlimit(resource.RLIMIT_NOFILE, (256, {want}))\n"
        "import api_stub_engine as STUB\n"
        "from mashbox.runtime import server as SV\n"
        "logs, up = [], threading.Event()\n"
        "def log(m):\n"
        "    logs.append(m)\n"
        "    if '\"listening\"' in str(m):\n"
        "        up.set()\n"
        "srv = SV.Server(STUB.make(S=1), STUB.IdChat(), 'stub-model', {}, log=log)\n"
        "srv.ready = True\n"
        "threading.Thread(target=SV.serve, args=(srv, '127.0.0.1', 0, log), daemon=True).start()\n"
        "assert up.wait(30), logs\n"
        "print(json.dumps(dict(limit=list(resource.getrlimit(resource.RLIMIT_NOFILE)), logs=logs)))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]
                                                      + [os.environ.get("PYTHONPATH", "")]), JAX_PLATFORMS="cpu")
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["limit"] == [want, want], f"soft limit after the server started: {out['limit'][0]} (hard {want})"
    lim = [json.loads(m) for m in out["logs"] if '"open_files_limit"' in m]
    assert lim == [dict(event="open_files_limit", before=256, soft=want, hard=want)], out["logs"]


def test_mashbox_run_serves_through_the_front_with_its_command_line(monkeypatch):
    """`mashbox run MODEL` (no --prompt) serves through front.serve_cli with the command line it was given, so the worker
    loads and warms the engine; this process (the front) never loads it. A worker that never gets ready makes serve_cli
    exit non-zero (test_serve_split_exits_non_zero_with_the_workers_error)."""
    from mashbox import cli
    from mashbox.runtime import front as Fr
    seen = {}

    def no_load(a):
        raise AssertionError("the front loaded the engine")
    monkeypatch.setattr(Fr, "serve_cli", lambda a, argv, keys, log, t0=None: seen.update(argv=list(argv), model=a.model,
                                                                                        keys=keys))
    monkeypatch.setattr(cli, "load", no_load)
    monkeypatch.setattr(cli, "api_keys", lambda a: ["k-test"])
    argv = ["run", "qwen3-8-27b", "--port", "8123"]
    cli.cmd_run(cli.build_parser().parse_args(argv), argv)
    assert seen == dict(argv=argv, model="qwen3-8-27b", keys=["k-test"]), seen


def test_the_front_takes_ready_only_with_the_workers_limits(tmp_path):
    """A heartbeat that says ready must not make the front ready before the worker's ready frame, which carries its rows,
    context and vocabulary: here the engine's first read takes 1.5 s, so a heartbeat (every second) comes in between.
    The front must still take 4 rows, the vocabulary and the context from the worker, and run 4 requests at once."""
    fr, port, log, logs = start(tmp_path, S=4, step_s=0.05, kw=dict(first_free_slots_s=1.5))
    wait_ready(fr)
    E = fr.engine
    assert (E.S, E.V, E.CTX) == (4, 1000, 4096), (E.S, E.V, E.CTX)
    out = []
    ths = [threading.Thread(target=lambda: out.append(post(port, *CHAT))) for _ in range(4)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)
    assert [c for c, _, _ in out] == [200] * 4
    assert time.time() - t0 < 3.5, "4 requests of 40 tokens at 0.05 s a step ran one after another"
