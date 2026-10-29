"""Admission order of the HTTP front (runtime/front.py Front.loop) with a real worker process (the scripted engine in
api_stub_engine.py): waiting requests go to the worker in Server.next_waiting's order, round robin across owners (the
owner taken least recently first), first in first out within one, so a request of an owner with nothing else waiting
joins at the next free row however long another owner's queue is. Old code: first come, first served over all owners.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

from mashbox.runtime import front as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402

KA, KB = "key-tenant-a-0001", "key-tenant-b-0002"


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


def admit_lines(err):
    """The MASHBOX_TRACE=1 admit lines on stderr, in the order the worker admitted (its stderr is the server's)."""
    out = []
    for ln in err.splitlines():
        if ln.startswith("{") and '"admit"' in ln:
            e = json.loads(ln)
            if e.get("event") == "admit":
                out.append(e)
    return out


def test_front_admits_round_robin_across_owners(tmp_path, capfd):
    """One worker row (S=1). Owner A's first request runs; three more of A's wait, then one of B's arrives last. B's
    request is admitted next, before A's waiting ones, and every request gets 200. The order is read where it is made:
    the worker's MASHBOX_TRACE=1 admit lines, one per request, named by the response id each client receives, with the
    admission counter 1..5 (the order responses come back is checked too, but a client thread can be late to record)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]),
               JAX_PLATFORMS="cpu", MASHBOX_TRACE="1")
    lines = []
    fr = F.Front(STUB.IdChat(), "m", {}, "api_stub_engine:make",
                 dict(S=1, step_s=0.3, outs=[list(range(11, 13))], log=str(tmp_path / "engine.jsonl")), env=env,
                 log=lines.append)
    try:
        port = STUB.serve_on_free_port(fr, keys=[KA, KB])
        assert until(lambda: fr.ready, 60)
        order, codes, rid_of = [], [], {}

        def go(tag, key):
            st, raw, _ = post(port, "/v1/chat/completions", dict(model="m", max_tokens=2,
                                                                messages=[dict(role="user", content="1")]), key)
            codes.append(st)
            rid_of[tag] = json.loads(raw).get("id") if st == 200 else None
            order.append(tag)
        ths = []
        for i, (tag, key) in enumerate([("a0", KA), ("a1", KA), ("a2", KA), ("a3", KA), ("b", KB)]):
            ths.append(threading.Thread(target=go, args=(tag, key), daemon=True))
            ths[-1].start()
            assert until(lambda: fr.waiting + len(fr.link.inflight if fr.link else ()) == i + 1, 10)
        for t in ths:
            t.join(60)
        assert codes == [200] * 5, codes
        assert order.index("b") <= 1, order             # B waited behind nobody of its own (FIFO: last)
        assert [t for t in order if t != "b"] == ["a0", "a1", "a2", "a3"], order     # first in, first out within A
        assert all(rid_of.values()) and len(set(rid_of.values())) == 5, rid_of
        tag_of = {r: t for t, r in rid_of.items()}
        got = admit_lines(capfd.readouterr().err)
        assert [tag_of.get(e["rid"]) for e in got] == ["a0", "b", "a1", "a2", "a3"], got   # admitted: B second, A in order
        assert [e["seq"] for e in got] == [1, 2, 3, 4, 5], got
        ta = {e["tenant"] for e in got if tag_of.get(e["rid"]) != "b"}
        tb = {e["tenant"] for e in got if tag_of.get(e["rid"]) == "b"}
        assert len(ta) == len(tb) == 1 and None not in ta | tb and ta != tb, got   # one tenant per key, two owners
    finally:
        fr.close()
