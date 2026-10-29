"""The request queue under load and the server's stop.

Each test drives the real server code (runtime/server.py, and runtime/front.py with a real worker process where named)
with the scripted engine in api_stub_engine.py:
- a streamed request dropped from a full queue (longest-queue drop) gets its route's status code with Retry-After and
  never a 200 followed by an error event (seen under load: 126 streams got 200, then "dropped from the queue", 0 tokens);
- waiting requests are admitted round robin across owners, first in first out within one, so an owner
  with nothing else waiting joins at the next free row (seen under load: 29 of 70 probes late);
- SIGINT while a row is mid-step: the server's process group is gone 1 s after it exits, rc 130 (seen before:
  rc -2, orphans left).
- /health counts each host array of the prompt cache once (a point shared by a slot and its saved entries was counted
  two or three times), so a soak can subtract every host-RAM cache from RSS;
- a request that ends with no engine event keeps its answer and finish on every route (a stop was reported as
  'length', Responses incomplete), and its tool calls, stop string and usage are those of the same answer given with an
  event;
- a request a longest-queue drop refuses in the moment the loop takes it from the queue is never sent to the engine,
  in this process and behind a worker;
- /health counts one moment of the prompt cache while the engine evicts and replaces entries;
- the stop: a SIGTERM right after its handler is installed still closes the server, and the stop waits for a request
  still being written, so its client gets the whole shutdown error event.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from collections import deque

import pytest

from mashbox.runtime import server as SV

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402

KA, KB = "key-tenant-a-0001", "key-tenant-b-0002"
ROUTES = {
    "chat": ("/v1/chat/completions", lambda n: dict(model="m", max_tokens=n, stream=True,
                                                     messages=[dict(role="user", content="1 2 3")]), 429),
    "anthropic": ("/v1/messages", lambda n: dict(model="m", max_tokens=n, stream=True,
                                                  messages=[dict(role="user", content="1 2 3")]), 529),
    "responses": ("/v1/responses", lambda n: dict(model="m", max_output_tokens=n, stream=True, input="1 2 3"), 429),
}


def start(S=1, keys=(KA, KB), outs=None, log=None):
    eng = STUB.StubEngine(S=S, outs=outs or [list(range(11, 21))])
    srv = SV.Server(eng, STUB.IdChat(), "m", {}, log=log or (lambda m: None))
    srv.ready = True
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    threading.Thread(target=SV.serve, args=(srv, "127.0.0.1", port, lambda m: None), kwargs=dict(keys=list(keys)),
                     daemon=True).start()
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port)).close()
            break
        except OSError:
            time.sleep(0.02)
    return eng, srv, port


def post(port, path, body, key, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e.headers


def until(cond, s=10):
    t = time.time()
    while not cond() and time.time() - t < s:
        time.sleep(0.01)
    return cond()


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_b5_a_stream_dropped_while_waiting_gets_its_status_code_never_200_then_error(route, monkeypatch):
    path, body, code = ROUTES[route]
    monkeypatch.setattr(SV, "MAX_WAITING", 2)
    eng, srv, port = start()
    eng.gate.clear()                                     # the engine holds: A's first request keeps the one row
    res = {}

    def go(name, key, n):
        res[name] = post(port, path, body(n), key)
    ths = [threading.Thread(target=go, args=("a0", KA, 4), daemon=True)]
    ths[0].start()
    assert until(lambda: eng.n_admitted == 1)
    for name in ("a1", "a2"):                            # two more of A's wait: the queue is full (MAX_WAITING 2)
        ths.append(threading.Thread(target=go, args=(name, KA, 4), daemon=True))
        ths[-1].start()
        assert until(lambda: srv.waiting == int(name[1]))
    ths.append(threading.Thread(target=go, args=("b", KB, 4), daemon=True))
    ths[-1].start()                                      # B is under its share: A's newest waiting (a2) is dropped
    assert until(lambda: "a2" in res), "the dropped request got no answer"
    st, raw, hd = res["a2"]
    assert st == code and hd.get("Retry-After"), (st, raw[:300])
    assert "dropped from the queue" in raw and "event:" not in raw and "data:" not in raw
    eng.gate.set()
    for t in ths:
        t.join(30)
    for name in ("a0", "a1", "b"):                       # the others are answered whole
        assert res[name][0] == 200 and "error" not in res[name][1], (name, res[name][1][:300])


def test_b5_an_owner_with_nothing_waiting_is_admitted_before_a_long_queue_of_another():
    lines = []
    old = os.environ.get("MASHBOX_TRACE")
    os.environ["MASHBOX_TRACE"] = "1"
    try:
        eng, srv, port = start(log=lines.append)
        eng.gate.clear()
        done = []

        def go(tag, key):
            st, raw, _ = post(port, "/v1/chat/completions", dict(model="m", max_tokens=2, messages=[
                dict(role="user", content="1 2 3")]), key)
            done.append((tag, st))
        ths = [threading.Thread(target=go, args=("a0", KA), daemon=True)]
        ths[0].start()
        assert until(lambda: eng.n_admitted == 1)
        for i in range(1, 6):                            # five more of A's wait, then one of B's arrives last
            ths.append(threading.Thread(target=go, args=(f"a{i}", KA), daemon=True))
            ths[-1].start()
            assert until(lambda: srv.waiting == i)
        ths.append(threading.Thread(target=go, args=("b", KB), daemon=True))
        ths[-1].start()
        assert until(lambda: srv.waiting == 6)
        eng.gate.set()
        for t in ths:
            t.join(30)
    finally:
        if old is None:
            os.environ.pop("MASHBOX_TRACE", None)
        else:
            os.environ["MASHBOX_TRACE"] = old
    adm = [json.loads(x) for x in lines if '"event": "admit"' in x]
    ten = [a["tenant"] for a in adm]
    assert len(ten) == 7 and all(st == 200 for _, st in done)
    a = ten[0]
    b = next(t for t in ten if t != a)
    assert ten == [a, b, a, a, a, a, a], ten             # first come first served would put B last


def test_next_waiting_rules():
    srv = SV.Server.__new__(SV.Server)
    srv.last_admit, srv.n_taken = {}, 0
    q = deque(dict(scope="ABC"[i % 3], n=i) for i in range(9))     # A, B, C in turn: the same as arrival order
    assert [srv.next_waiting(q)["n"] for _ in range(9)] == list(range(9))
    srv.last_admit, srv.n_taken = {}, 0
    q = deque([dict(scope="A", n=0), dict(scope="A", n=1), dict(scope="A", n=2), dict(scope="B", n=3)])
    srv.last_admit["A"], srv.n_taken = 1, 1
    assert [srv.next_waiting(q)["n"] for _ in range(4)] == [3, 0, 1, 2]           # B was never taken: it goes first
    srv.last_admit, srv.n_taken = {"A": 1, "B": 9}, 9                            # B was taken last: A would go first
    q = deque([dict(scope="A", n=0), dict(scope="B", n=1, retried=True)])
    assert srv.next_waiting(q)["n"] == 1                                          # a retried request goes first
    q = deque(dict(scope=None, n=i) for i in range(5))                            # one owner: first come, first served
    assert [srv.next_waiting(q)["n"] for _ in range(5)] == list(range(5))
    with pytest.raises(IndexError):
        srv.next_waiting(deque())


def _front_script(tmp_path, step_s):
    return f"""
import os, sys
sys.path[:0] = {[HERE] + [p for p in sys.path if p.endswith('src')]!r}
from mashbox.runtime import front as F, server as SV
import api_stub_engine as STUB
env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path[:3]), JAX_PLATFORMS="cpu")
fr = F.Front(STUB.IdChat(), "stub-model", {{}}, "api_stub_engine:make",
             dict(S=1, step_s={step_s}, outs=[list(range(11, 51))], log={str(tmp_path / 'engine.jsonl')!r}), env=env,
             log=lambda m: print(m, flush=True))
SV.serve(fr, "127.0.0.1", int(sys.argv[1]), lambda m: print(m, flush=True))
"""


def test_b3_sigint_mid_step_leaves_no_process_of_the_server(tmp_path):
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    script = tmp_path / "front.py"
    script.write_text(_front_script(tmp_path, 3.0))      # each engine step takes 3 s: SIGINT lands mid-step
    out = open(tmp_path / "front.log", "w")
    p = subprocess.Popen([sys.executable, str(script), str(port)], stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
    try:
        assert until(lambda: '"ready"' in (tmp_path / "front.log").read_text(), 60), (tmp_path / "front.log").read_text()
        got = dict(lines=[])
        th = threading.Thread(target=read_stream, args=(port, "/v1/chat/completions", dict(model="stub-model",
                              max_tokens=40, stream=True, messages=[dict(role="user", content="1 2 3")]), got), daemon=True)
        th.start()
        assert until(lambda: any(ln.startswith("data: {") for ln in got["lines"]), 20), got   # its first token is out
        time.sleep(0.5)                                  # the worker is inside its next 3 s step
        pg = os.getpgid(p.pid)
        p.send_signal(signal.SIGINT)
        rc = p.wait(timeout=30)

        def alive():
            try:
                os.killpg(pg, 0)
                return True
            except ProcessLookupError:
                return False
        at_exit = alive()                                # the moment the front is gone: its worker must be gone too
        time.sleep(1.0)
        later = alive()                                  # the certification check (1 s); on a chip the worker's teardown takes longer
        log = (tmp_path / "front.log").read_text()
        # the front ended its worker itself before exiting (Front.close: the worker exits, killed after the timeout)
        assert '"front_closed"' in log and '"stopping"' in log, log[-800:]
        assert rc in (0, 130) and not at_exit and not later, (rc, at_exit, later, log[-800:])
        th.join(10)                                      # the client's stream is whole: tokens, the error event, [DONE]
        data = [ln for ln in got["lines"] if ln.startswith("data: ")]
        assert not th.is_alive() and got.get("status") == 200 and got.get("eof"), got
        assert data[-1] == "data: [DONE]" and "the server stopped mid-answer" in data[-2], data[-3:]
        assert '"stopped"' in log and '"answering": 0' in log, log[-800:]
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        reap_own(tmp_path, [p.pid] + worker_pids(tmp_path / "front.log"))   # a failed run leaves nothing behind
        out.close()


def read_stream(port, path, body, got, key="x"):
    """POST a streamed request; got["status"], then each line as it comes (got["lines"]), got["eof"] once the server
    has closed the stream."""
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    r = c.getresponse()
    got["status"] = r.status
    while True:
        ln = r.readline()
        if not ln:
            break
        got["lines"].append(ln.decode().rstrip("\n"))
    got["eof"] = True


def worker_pids(log):
    try:
        return [json.loads(ln)["pid"] for ln in open(log) if ln.startswith('{"event": "worker_ready"')]
    except OSError:
        return []


def reap_own(tmp_path, pids):
    """SIGTERM each pid that is still this test's own process (its command line names the test's tmp_path), so a pid
    the system has reused for another process is never touched."""
    for pid in pids:
        try:
            if str(tmp_path) in open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace"):
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def test_health_counts_each_cached_array_once():
    """/health's cache bytes are what a soak subtracts from RSS (net of every host-RAM
    cache), so each host array counts once. A slot saved to the host cache keeps the points the next
    request on that slot shares (engine.py Engine._save builds the entry from the slot's own point dicts; _admit keeps
    those at or below the new request's start), and a second save of that slot holds the shared point again in a second
    entry (HostCache.add keeps the first: one of its points is not a prefix of the new one)."""
    import types

    import numpy as np
    from mashbox.runtime import front as F
    from mashbox.runtime.engine import HostCache

    sysp = tuple(range(100))

    def point(tail):
        return dict(ids=sysp + tail, r=np.ones(1000, np.float32), c=np.ones(100, np.float32), scope="own")

    def save(slot):                                       # the entry Engine._save makes of a slot
        kv = [(0, (np.zeros(3000, np.uint8), np.zeros(3000, np.uint8)))]
        return dict(kv=kv, snaps=list(slot), scope="own",
                    bytes=6000 + sum(x["r"].nbytes + x["c"].nbytes for x in slot))
    host = HostCache(10 ** 9)
    sp, a, b, c = point(()), point((1001,) * 20), point((2001,) * 30), point((3001,) * 40)
    host.add(save([sp, a]))                               # conversation A on the slot, displaced by B (same system prompt)
    host.add(save([sp, b]))                               # B displaced by C: the entry of A stays (A is not a prefix of B)
    assert len(host.items) == 2
    eng = types.SimpleNamespace(host=host, snaps=[[sp, c]], S=1, free_slots=lambda: 0)
    h = SV.Server.health(types.SimpleNamespace(engine=eng, waiting=0))
    pt = 4400                                             # one point: r 4,000 + c 400 bytes
    assert h["cache_bytes"] == 2 * 6000 + 4 * pt, h       # two K/V blocks + the points sp, a, b, c, each once
    assert h["host_cache_bytes"] == 2 * 6000 + 3 * pt and h["snapshot_bytes"] == pt, h
    assert h["cache_bytes"] == h["host_cache_bytes"] + h["snapshot_bytes"]
    assert {"cache_bytes", "host_cache_bytes", "snapshot_bytes"} <= set(F.Front.HEALTH_FIELDS)   # the split server too


class NoEventEngine:
    """generate()-only engine (the whole-batch loop) that ends every request with no step event and no token."""
    CTX, S, K, prefix_cache = 4096, 1, 4, False

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, **kw):
        return dict(ids=[[] for _ in prompts], finish=["stop"] * len(prompts), cached=[0] * len(prompts), timing={})


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_stream_that_ends_with_no_event_still_gets_its_200_and_its_end(route):
    path, body, _ = ROUTES[route]
    srv = SV.Server(NoEventEngine(), STUB.IdChat(), "m", {}, log=lambda m: None)
    srv.ready = True
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    threading.Thread(target=SV.serve, args=(srv, "127.0.0.1", port, lambda m: None), daemon=True).start()
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port)).close()
            break
        except OSError:
            time.sleep(0.02)
    st, raw, hd = post(port, path, body(4), "x")
    assert st == 200 and hd.get("Content-Type") == "text/event-stream", (st, raw[:300])
    end = {"chat": "[DONE]", "anthropic": "message_stop", "responses": "response.completed"}[route]
    assert end in raw and "error" not in raw, raw[-300:]


class NoEventTextEngine(NoEventEngine):
    """As NoEventEngine, with an answer: generate() returns `ids` and `fin` and never calls on_step."""

    def __init__(self, ids, fin):
        self.ids, self.fin = list(ids), fin

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, **kw):
        return dict(ids=[list(self.ids) for _ in prompts], finish=[self.fin] * len(prompts), cached=[0] * len(prompts),
                    timing={})


class OneEventTextEngine(NoEventTextEngine):
    """The same answer, given to the server the usual way: one on_step with every token and the finish, then the result."""

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, **kw):
        if on_step is not None:
            on_step([list(self.ids) for _ in prompts], [self.fin] * len(prompts))
        return super().generate(prompts, max_new, stop_ids, samp, None, **kw)


def answer_of(engine, route, stream):
    path, body, _ = ROUTES[route]
    srv = SV.Server(engine, STUB.IdChat(), "m", {}, log=lambda m: None)
    srv.ready = True
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    threading.Thread(target=SV.serve, args=(srv, "127.0.0.1", port, lambda m: None), daemon=True).start()
    assert until(lambda: _open(port), 4)
    st, raw, _ = post(port, path, dict(body(4), stream=stream), "x")
    assert st == 200, (st, raw[:300])
    return (sse_answer(route, raw) if stream else json_answer(route, json.loads(raw))), raw


def sse_answer(route, raw):
    """(text, finish) of a streamed answer: chat finish_reason, Anthropic stop_reason, Responses status."""
    evs = [json.loads(ln[6:]) for ln in raw.splitlines() if ln.startswith("data: {")]
    if route == "chat":
        text = "".join((e["choices"][0].get("delta") or {}).get("content") or "" for e in evs if e.get("choices"))
        fin = [e["choices"][0]["finish_reason"] for e in evs if e.get("choices") and e["choices"][0].get("finish_reason")]
        return text, fin[-1] if fin else None
    if route == "anthropic":
        text = "".join(e["delta"].get("text", "") for e in evs if e.get("type") == "content_block_delta")
        fin = [e["delta"].get("stop_reason") for e in evs if e.get("type") == "message_delta"]
        return text, fin[-1] if fin else None
    text = "".join(e["delta"] for e in evs if e.get("type") == "response.output_text.delta")
    end = [e["response"]["status"] for e in evs if e.get("type") in ("response.completed", "response.incomplete")]
    return text, end[-1] if end else None


def json_answer(route, j):
    if route == "chat":
        return j["choices"][0]["message"].get("content") or "", j["choices"][0]["finish_reason"]
    if route == "anthropic":
        return "".join(c.get("text", "") for c in j["content"]), j["stop_reason"]
    return "".join(c.get("text", "") for it in j["output"] if it["type"] == "message" for c in it["content"]), j["status"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ids,fin", [((), "stop"), ((11, 12), "stop"), ((11, 12, 13, 14), "length")])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_an_answer_that_ends_with_no_event_keeps_its_text_and_finish(route, ids, fin, stream):
    """An engine that ends a request with no step event (the whole-batch loop, generate() without on_step): the answer
    and its finish are the request's result, on every route, streamed or not, as the legacy route takes them: the client
    gets what the same answer given with an event gives. Seen before: the finish stayed 'length' (Responses:
    incomplete, max_output_tokens, for a stop) and no text came out."""
    (text, got), raw = answer_of(NoEventTextEngine(ids, fin), route, stream)
    (ref_text, ref_got), _ = answer_of(OneEventTextEngine(ids, fin), route, stream)
    want = {"chat": fin, "anthropic": "end_turn" if fin == "stop" else "max_tokens",
            "responses": "completed" if fin == "stop" else "incomplete"}[route]
    assert ref_got == want and (ref_text != "") == bool(ids), (ref_text, ref_got)    # the reference is a real answer
    assert (text, got) == (ref_text, ref_got), raw[-400:]

def _open(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
        return True
    except OSError:
        return False


# ---- a request dropped in the moment the loop takes it ----------------------------------------------------------
def _env():
    return dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.abspath("src"), os.environ.get("PYTHONPATH", "")]),
                MASHBOX_FAULT="", JAX_PLATFORMS="cpu")


@pytest.mark.parametrize("mode", ["in_process", "worker"])
def test_a_request_dropped_while_it_is_being_taken_is_never_sent_to_the_engine(mode, monkeypatch, tmp_path):
    """The loop takes a waiting request (next_waiting removes it from its queue) and only then takes it off its owner's
    waiting list (_unwait). A longest-queue drop in that moment can pick it: its client gets 429 + Retry-After at once.
    The loop must not then send it to the engine (a row run for nobody). The test stops the loop right after next_waiting
    returns, lets B's request drop A's, then lets the loop go on: the engine admits A's first request and B's only
    (before, it admitted A's dropped request too)."""
    monkeypatch.setattr(SV, "MAX_WAITING", 1)
    taken, go, armed = threading.Event(), threading.Event(), []
    real = SV.Server.next_waiting

    def paused(self, wait):
        j = real(self, wait)
        if armed:
            armed.clear()
            taken.set()
            go.wait(30)
        return j
    monkeypatch.setattr(SV.Server, "next_waiting", paused)
    elog = tmp_path / "engine.jsonl"
    front = None
    if mode == "in_process":
        eng = STUB.StubEngine(S=1, outs=[list(range(11, 21))], log=str(elog))
        srv = SV.Server(eng, STUB.IdChat(), "m", {}, log=lambda m: None)
        srv.ready = True
        eng.gate.clear()                                 # A's first request keeps the one row
        release = eng.gate.set
    else:
        from mashbox.runtime import front as F
        hold = tmp_path / "go"
        front = srv = F.Front(STUB.IdChat(), "m", {}, "api_stub_engine:make",
                              dict(S=1, step_s=0.02, outs=[list(range(11, 21))], log=str(elog),
                                   hold=dict(after=1, path=str(hold))), env=_env(), log=lambda m: None)
        release = hold.touch                             # A's first request keeps the one row until this file exists
    try:
        with socket.socket() as s_:
            s_.bind(("127.0.0.1", 0))
            port = s_.getsockname()[1]
        threading.Thread(target=SV.serve, args=(srv, "127.0.0.1", port, lambda m: None), kwargs=dict(keys=[KA, KB]),
                         daemon=True).start()
        assert until(lambda: srv.ready and _open(port), 60)
        path, body, code = ROUTES["chat"]
        res = {}

        def go_post(name, key):
            res[name] = post(port, path, body(4), key)
        admits = lambda: sum(e.get("ev") == "admit" for e in STUB.read_log(str(elog)))   # noqa: E731
        ths = [threading.Thread(target=go_post, args=("a0", KA), daemon=True)]
        ths[0].start()
        assert until(lambda: admits() == 1, 20)
        ths.append(threading.Thread(target=go_post, args=("a1", KA), daemon=True))
        ths[-1].start()
        assert until(lambda: srv.waiting == 1)           # the queue is full (MAX_WAITING 1)
        armed.append(True)
        release()                                        # a0 ends; the loop takes a1 and stops there
        assert taken.wait(30), "the loop did not take a1"
        ths.append(threading.Thread(target=go_post, args=("b", KB), daemon=True))
        ths[-1].start()                                  # B is under its share: A's newest waiting (a1) is dropped
        assert until(lambda: "a1" in res, 10), "the dropped request got no answer"
        go.set()
        for th in ths:
            th.join(30)
        st, raw, hd = res["a1"]
        assert st == code and hd.get("Retry-After") and "dropped from the queue" in raw, (st, raw[:300])
        assert res["a0"][0] == 200 and res["b"][0] == 200, (res["a0"][:2], res["b"][:2])
        time.sleep(0.3)                                  # an admission of a1 after b would show here
        assert admits() == 2, STUB.read_log(str(elog))   # a0 and b; never the refused a1
    finally:
        go.set()
        if front is not None:
            front.close()


def test_the_whole_batch_loop_keeps_a_dropped_requests_refusal():
    """The whole-batch loop (engines with generate() only) gathers requests before run() takes them off the waiting
    lists, so a drop can pick one in between: it has its 429 / 529 already and also carries "cancel". run() must leave
    its refusal as it is; before, it replaced it with a client hang-up (ConnectionResetError), so a handler that read
    it then wrote no answer at all (the whole-batch side)."""
    srv = SV.Server(NoEventEngine(), STUB.IdChat(), "m", {}, log=lambda m: None)
    over = SV.Overloaded("server overloaded: this request was dropped from the queue; try again", 3)
    j = dict(ids=[1], max_new=4, samp=None, ev=threading.Event(), stream=None, scope="a", cancel=True, unwaited=True,
             error=over)
    j["ev"].set()                                        # as submit()'s longest-queue drop leaves it
    srv.run([j])
    assert j["error"] is over and "result" not in j, j


# ---- /health counts one moment of the cache ---------------------------------------------------------------------
def test_health_counts_one_moment_of_the_cache_while_the_engine_changes_it():
    """/health runs on another thread (the HTTP handler; in a worker the heartbeat) while the engine evicts host entries
    and replaces a slot's points. It must report the cache of one moment: the loop's last call boundary. Here /health is
    stopped in the middle of its count (on the first array's byte count) while a request runs a step that evicts entry
    e2 and gives the slot a new point; then it goes on. It is stopped the same way if it iterates the engine's live slot
    list itself (a copy taken on its own thread can straddle the step too). Before, it summed the old entry list (e2
    included) with the slot's new point: 20,800 bytes, a cache that never existed."""
    from mashbox.runtime.engine import HostCache
    inside, resume, who = threading.Event(), threading.Event(), {}

    def stop_here():                         # the first time /health reads the cache: wait while the step runs
        if not inside.is_set():
            inside.set()
            resume.wait(10)

    class Live(list):
        """The engine's live slot point lists: /health's thread waits when it iterates them."""
        def __iter__(self):
            if threading.current_thread() is who.get("health"):
                stop_here()
            return super().__iter__()

    class Arr:
        """An array's byte count; the first read of the marked one waits (the /health thread is then mid-count)."""
        def __init__(self, n, stop=False):
            self.n, self.stop = n, stop

        @property
        def nbytes(self):
            if self.stop:
                self.stop = False
                stop_here()
            return self.n

    def entry(scope, first):
        return dict(kv=[(0, (first, Arr(3000)))], snaps=[dict(ids=(len(scope),), r=Arr(0), c=Arr(0), scope=scope)],
                    scope=scope, bytes=6000)
    e1, e2 = entry("a", Arr(3000, stop=True)), entry("bb", Arr(3000))
    p1, p2 = dict(ids=(1, 2), r=Arr(4000), c=Arr(400)), dict(ids=(1, 2, 3), r=Arr(8000), c=Arr(800))

    class CacheEngine(STUB.StubEngine):
        def __init__(self):
            super().__init__(S=1, outs=[[11, 12]])
            self.host, self.snaps, self.changed = HostCache(10 ** 9), Live([[p1]]), False
            self.host.add(e1)
            self.host.add(e2)

        def step(self, waiting=False):
            if not self.changed:             # this call evicts e2 and gives the slot its new point, as a retire does
                self.changed = True
                self.host.take(e2)
                self.snaps[0] = [p2]
            return super().step(waiting)
    eng = CacheEngine()
    srv = SV.Server(eng, STUB.IdChat(), "m", {}, log=lambda m: None)
    srv.ready = True
    port = STUB.serve_on_free_port(srv)
    until(lambda: getattr(srv, "cache_view", None) is not None, 2)     # the loop is at its first call boundary
    got = {}
    th = who["health"] = threading.Thread(target=lambda: got.update(h=srv.health()), daemon=True)
    th.start()
    assert inside.wait(5)                                # /health is in the middle of its count
    st, raw, _ = post(port, "/v1/chat/completions", dict(model="m", max_tokens=2,
                                                          messages=[dict(role="user", content="1 2 3")]), "x")
    assert st == 200 and eng.changed, (st, raw[:300])    # the engine evicted e2 and replaced the slot's points
    resume.set()
    th.join(5)
    keys = ("cache_bytes", "host_cache_bytes", "snapshot_bytes")
    g1 = dict(cache_bytes=12000 + 4400, host_cache_bytes=12000, snapshot_bytes=4400)   # before the step
    g2 = dict(cache_bytes=6000 + 8800, host_cache_bytes=6000, snapshot_bytes=8800)     # after it
    assert {k: got["h"][k] for k in keys} == g1, got["h"]
    assert {k: srv.health()[k] for k in keys} == g2      # the next boundary's moment


# ---- the stop -----------------------------------------------------------------------------------------------------
def test_b3_a_sigterm_right_after_its_handler_is_installed_still_stops_the_server(monkeypatch):
    """A SIGTERM that lands the moment serve() has installed its handler, before it serves anything, still runs the
    stop: the server (a front: its requests and its worker) is closed and the process exits 143. Before, the handler
    was installed before the protected block, so this signal's KeyboardInterrupt left serve() with nothing closed."""
    assert threading.current_thread() is threading.main_thread()
    real = signal.signal
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    sent, closed, lines = [], [], []

    def install(sig, handler):
        old = real(sig, handler)
        if sig == signal.SIGTERM and callable(handler) and not sent:
            sent.append(1)
            os.kill(os.getpid(), signal.SIGTERM)          # lands right after its handler is installed
        return old
    fake = types.SimpleNamespace(close=lambda: closed.append(1), ready=True, model_id="m")
    monkeypatch.setattr(SV.signal, "signal", install)
    out = None
    try:
        SV.serve(fake, "127.0.0.1", 0, lines.append)
    except BaseException as e:                           # noqa: BLE001 - a KeyboardInterrupt here is the failure
        out = e
    finally:
        for s, h in before.items():
            real(s, h)
    assert sent and isinstance(out, SystemExit) and out.code == 128 + signal.SIGTERM, (repr(out), lines)
    assert closed == [1] and any('"stopping"' in m for m in lines), (closed, lines)


def test_b3_the_stop_waits_for_an_answer_still_being_written(monkeypatch):
    """stop_serving closes the server (the front ends each request: a stream with output gets its error event), and the
    process exits right after. A handler thread can still be writing that error event when close() returns; the stop
    must wait for it, so the client gets the whole event and the end of its stream, not a cut. Here the request's own
    thread writes its error event 0.5 s after close() returns. Before, the stop exited before it."""
    wrote = []
    real_error = SV.SSE.error

    def error(self, *a, **k):
        r = real_error(self, *a, **k)
        wrote.append(time.monotonic())
        return r
    monkeypatch.setattr(SV.SSE, "error", error)

    class Closing:
        """A server whose one streamed answer ends with the shutdown error once close() has run."""
        model_id, ready = "m", True

        def __init__(self):
            self.closed, self.lines = threading.Event(), []

        def log(self, m):
            self.lines.append(m)

        def openai_chat(self, b, sse, meta):
            sse.start()
            sse.data(dict(id="c1", object="chat.completion.chunk", model="m",
                          choices=[dict(index=0, delta=dict(content="11 "), finish_reason=None)]))
            self.closed.wait(10)
            time.sleep(0.5)                              # this handler's thread goes on after close() has returned
            raise SV.WorkerFault("the server stopped mid-answer; this answer is cut off here", code="shutdown")
        anthropic = responses = openai_chat

        def close(self):
            self.closed.set()
    srv = Closing()
    port = STUB.serve_on_free_port(srv)
    got = dict(lines=[])
    th = threading.Thread(target=read_stream, args=(port, "/v1/chat/completions", dict(
        model="m", stream=True, messages=[dict(role="user", content="1 2 3")]), got), daemon=True)
    th.start()
    assert until(lambda: any(ln.startswith("data: {") for ln in got["lines"]), 10), got
    srv.httpd.shutdown()                                 # serve_forever returns, as it does on the signal
    out = None
    try:
        SV.stop_serving(srv, srv.httpd, srv.log, signal.SIGTERM, main=False)
    except BaseException as e:                           # noqa: BLE001
        out = e
    t_exit = time.monotonic()                            # the process would exit here
    th.join(10)
    assert isinstance(out, SystemExit) and out.code == 128 + signal.SIGTERM, repr(out)
    assert wrote and wrote[0] <= t_exit, (wrote, t_exit)  # its error event was written before the stop ended
    data = [ln for ln in got["lines"] if ln.startswith("data: ")]
    assert data[-1] == "data: [DONE]" and "the server stopped mid-answer" in data[-2] and got.get("eof"), data[-3:]


# ---- no-event answers with tool calls, stop strings and usage ------------------------------------------------------
class CharChat:
    """Characters as token ids: the server path with any text an answer needs (a tool call, a stop string)."""
    stop_ids = ()

    def encode(self, text):
        return [ord(c) for c in str(text)]

    def decode(self, ids):
        return "".join(chr(int(t)) for t in ids)

    def chat_ids(self, msgs, *a, **k):
        c = msgs[-1].get("content")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        return self.encode(c or "x")


CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
SCHEMA = dict(type="object", properties=dict(command=dict(type="string")), required=["command"])
CASES = {   # route -> case -> (path, body)
    "chat": ("/v1/chat/completions", dict(
        tool=dict(tools=[dict(type="function", function=dict(name="Bash", description="run", parameters=SCHEMA))]),
        stop=dict(stop=["XY"]))),
    "anthropic": ("/v1/messages", dict(tool=dict(tools=[dict(name="Bash", description="run", input_schema=SCHEMA)]),
                                       stop=dict(stop_sequences=["XY"]))),
    "responses": ("/v1/responses", dict(tool=dict(tools=[dict(type="function", name="Bash", description="run",
                                                              parameters=SCHEMA)]), stop=dict(stop=["XY"]))),
}
VARY = {"id", "created", "created_at", "completed_at", "call_id", "item_id", "request_id", "timings",
        "system_fingerprint"}


def _norm(o):
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items() if k not in VARY}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    return o


def _shape(raw, stream):
    """The whole answer without its ids and times: the JSON body, or every event name and data object in order."""
    if not stream:
        return _norm(json.loads(raw))
    return [_norm(json.loads(ln[6:])) if ln.startswith("data: {") else ln for ln in raw.splitlines()
            if ln.startswith("event: ") or ln.startswith("data: ")]


def _out_tokens(raw, stream):
    objs = [json.loads(raw)] if not stream else [json.loads(ln[6:]) for ln in raw.splitlines() if ln.startswith("data: {")]
    for o in reversed(objs):
        u = o.get("usage") or (o.get("response") or {}).get("usage")
        if u:
            return u.get("completion_tokens", u.get("output_tokens"))
    return None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["tool", "stop"])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_no_event_answer_keeps_its_tool_call_stop_and_usage(route, case, stream):
    """The no-event answer (generate() without on_step) against the same answer given with one event, on every route,
    streamed and not: one tool call, and a stop string the engine ran past. The whole answer must be the same (every
    field and event but ids and times: tool call structure, finish, stop_sequence, usage), with the call parsed (never
    text) and the stop answer cut at the stop, with usage at the stop point as stop_point counts it: through the stop
    string ("ab XY" = 5 tokens), never the 8 the engine made.
    The earlier no-event test compared plain text and finish only."""
    path, extra = CASES[route]
    ids = CharChat().encode(CALL if case == "tool" else "ab XY ~~")
    fin = "stop" if case == "tool" else "length"
    n = "max_output_tokens" if route == "responses" else "max_tokens"
    body = dict(model="m", stream=stream, **{n: 400}, **extra[case])
    body.update(input="hi") if route == "responses" else body.update(messages=[dict(role="user", content="hi")])
    if stream and route == "chat":
        body["stream_options"] = dict(include_usage=True)
    raws = []
    for engine in (NoEventTextEngine(ids, fin), OneEventTextEngine(ids, fin)):
        srv = SV.Server(engine, CharChat(), "m", {}, log=lambda m: None)
        srv.ready = True
        port = STUB.serve_on_free_port(srv)
        st, raw, _ = post(port, path, body, "x")
        assert st == 200, (st, raw[:400])
        raws.append(raw)
    got, ref = raws
    assert _shape(got, stream) == _shape(ref, stream), (got[-600:], ref[-600:])
    if case == "tool":
        assert '"Bash"' in ref and "pwd" in ref and "<tool_call>" not in ref and "<function" not in ref, ref[-600:]
    else:
        assert "~~" not in ref and _out_tokens(ref, stream) == 5, ref[-600:]
        if route == "anthropic":
            assert '"stop_sequence": "XY"' in ref, ref[-600:]
