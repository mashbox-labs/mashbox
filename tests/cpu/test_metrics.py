"""GET /metrics (runtime/metrics.py): the Prometheus text format, counts that equal what the clients got, live gauges,
worker restarts, and no label that names a tenant, a key or a session. In one process and through the worker."""
import json
import math
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402
import test_front_failures as TF  # noqa: E402  (the stub server helpers: start, wait_ready, post, IdChat)

from mashbox.runtime import server as SV  # noqa: E402

_close_fronts = TF._close_fronts            # each test's fronts (and their worker processes) end with it

NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
SAMPLE = re.compile(rf"^({NAME})(\{{(.*)\}})? (\S+)$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(,|$)')
FAMILIES = {   # name -> type: the families every server exposes (running / kv use need a slot count, always there here)
    "mashbox:num_requests_running": "gauge", "mashbox:num_requests_waiting": "gauge",
    "mashbox:kv_cache_usage_perc": "gauge", "mashbox:cache_config_info": "gauge",
    "mashbox:request_success_total": "counter", "mashbox:prompt_tokens_total": "counter",
    "mashbox:generation_tokens_total": "counter", "mashbox:prefix_cache_queries_total": "counter",
    "mashbox:prefix_cache_hits_total": "counter", "mashbox:num_preemptions_total": "counter",
    "mashbox:worker_restarts_total": "counter", "mashbox:time_to_first_token_seconds": "histogram",
    "mashbox:request_queue_time_seconds": "histogram", "mashbox:e2e_request_latency_seconds": "histogram",
    "mashbox:request_time_per_output_token_seconds": "histogram"}


def parse(text):
    """Strict reader of the Prometheus text format 0.0.4 -> {family: dict(type, help, samples=[(name, labels, value)])}.
    Fails on any line that is not HELP, TYPE or a sample, on a sample before its family's TYPE, on a name that does not
    belong to its family, and on a histogram whose buckets are not cumulative or whose +Inf bucket is not its count."""
    assert text.endswith("\n"), "the exposition ends with a line feed"
    fams, cur = {}, None
    for ln in text.split("\n")[:-1]:
        if ln.startswith("# HELP "):
            name, _, h = ln[7:].partition(" ")
            assert re.fullmatch(NAME, name) and h, ln
            fams.setdefault(name, dict(type=None, help=h, samples=[]))
            cur = name
            continue
        if ln.startswith("# TYPE "):
            name, typ = ln[7:].split(" ")
            assert typ in ("counter", "gauge", "histogram", "summary", "untyped"), ln
            assert name == cur and fams[name]["type"] is None, f"TYPE after its HELP, once: {ln}"
            fams[name]["type"] = typ
            continue
        assert not ln.startswith("#") and ln.strip(), f"no other comments or blank lines: {ln!r}"
        m = SAMPLE.match(ln)
        assert m, f"not a sample line: {ln!r}"
        name, labels, value = m.group(1), {}, float(m.group(4))
        if m.group(3):
            pos = 0
            for lm in LABEL.finditer(m.group(3)):
                assert lm.start() == pos, ln
                labels[lm.group(1)] = lm.group(2)
                pos = lm.end()
            assert pos == len(m.group(3)), f"bad labels: {ln!r}"
        assert cur is not None and fams[cur]["type"], f"sample before its TYPE: {ln!r}"
        ok = {cur} if fams[cur]["type"] != "histogram" else {cur + s for s in ("_bucket", "_sum", "_count")}
        assert name in ok, f"{name} is not in family {cur}"
        fams[cur]["samples"].append((name, labels, value))
    for name, f in fams.items():
        assert f["type"] and f["samples"], f"family {name} has a TYPE and samples"
        if f["type"] == "histogram":
            b = [(lab["le"], v) for n, lab, v in f["samples"] if n.endswith("_bucket")]
            les = [math.inf if le == "+Inf" else float(le) for le, _ in b]
            assert les == sorted(les) and les[-1] == math.inf, name
            assert [v for _, v in b] == sorted(v for _, v in b), f"{name}: buckets cumulative"
            count = [v for n, _, v in f["samples"] if n.endswith("_count")]
            assert count == [b[-1][1]], f"{name}: +Inf bucket equals _count"
    return fams


def one(fams, name, **labels):
    fam = name if name in fams else next(k for k in fams if name in (k + "_sum", k + "_count", k + "_bucket"))
    got = [v for n, lab, v in fams[fam]["samples"] if n == name and all(lab.get(k) == str(x) for k, x in labels.items())]
    assert len(got) == 1, (name, labels, fams[fam]["samples"])
    return got[0]


def count(fams, name):
    return one(fams, name + "_count")


def scrape(port, key=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


def server(tmp_path, mode, S=2, step_s=0.005):
    if mode == "worker":
        fr, port, _, logs = TF.start(tmp_path, S=S, step_s=step_s)
        TF.wait_ready(fr)
        return fr, port
    fr = SV.Server(STUB.make(S=S, step_s=step_s, outs=[TF.OUT]), TF.IdChat(), "stub-model", {}, log=lambda m: None)
    fr.ready = True
    return fr, STUB.serve_on_free_port(fr)


@pytest.mark.parametrize("mode", ["one_process", "worker"])
def test_metrics_count_what_the_clients_got(tmp_path, mode):
    """Six requests on four routes, streamed and not, one cut by a stop string: the counters equal the usage the clients
    got, every request lands in each latency histogram once, the gauges read idle afterwards, and the text parses."""
    fr, port = server(tmp_path, mode)
    st, ctype, txt = scrape(port)
    assert st == 200 and ctype == "text/plain; version=0.0.4; charset=utf-8", (st, ctype)
    f0 = parse(txt)
    assert {k: f0[k]["type"] for k in FAMILIES} == FAMILIES
    assert one(f0, "mashbox:request_success_total", finished_reason="stop") == 0
    usage = []                    # (finish, prompt tokens, completion tokens) as each client got them
    code, raw, _ = TF.post(port, "/v1/chat/completions", dict(model="stub-model", max_tokens=40,
                                                             messages=[dict(role="user", content="1 2 3")]))
    r = json.loads(raw)
    usage.append((r["choices"][0]["finish_reason"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]))
    code, raw, _ = TF.post(port, "/v1/chat/completions", dict(model="stub-model", max_tokens=40, stop=["15"],
                                                             messages=[dict(role="user", content="1 2 3")]))
    r = json.loads(raw)
    assert r["choices"][0]["finish_reason"] == "stop" and "15" not in r["choices"][0]["message"]["content"], r
    usage.append(("stop", r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]))
    code, raw, _ = TF.post(port, "/v1/completions", dict(model="stub-model", max_tokens=5, prompt="4 5"))
    r = json.loads(raw)
    usage.append((r["choices"][0]["finish_reason"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]))
    code, raw, _ = TF.post(port, "/v1/completions", dict(model="stub-model", max_tokens=7, prompt="4 5 6", stream=True,
                                                        stream_options=dict(include_usage=True)))
    u = [e for _, e in TF.events(raw) if isinstance(e, dict) and e.get("usage")][-1]["usage"]
    usage.append(("length", u["prompt_tokens"], u["completion_tokens"]))
    code, raw, _ = TF.post(port, "/v1/messages", dict(model="stub-model", max_tokens=9,
                                                     messages=[dict(role="user", content="1 2 3 4")]))
    r = json.loads(raw)
    assert r["stop_reason"] == "max_tokens"
    usage.append(("length", r["usage"]["input_tokens"], r["usage"]["output_tokens"]))
    code, raw, _ = TF.post(port, *TF.RESP)
    done = [e for n, e in TF.events(raw) if n in ("response.completed", "response.incomplete")][-1]["response"]
    usage.append(("length", done["usage"]["input_tokens"], done["usage"]["output_tokens"]))
    assert [u[0] for u in usage] == ["length", "stop", "length", "length", "length", "length"], usage
    assert [u[2] for u in usage[:1] + usage[2:]] == [40, 5, 7, 9, 40] and 0 < usage[1][2] < 40, usage
    f = parse(scrape(port)[2])
    assert one(f, "mashbox:request_success_total", finished_reason="stop") == 1
    assert one(f, "mashbox:request_success_total", finished_reason="length") == 5
    assert sum(v for n, _, v in f["mashbox:request_success_total"]["samples"]) == 6
    assert one(f, "mashbox:prompt_tokens_total") == sum(u[1] for u in usage)
    assert one(f, "mashbox:generation_tokens_total") == sum(u[2] for u in usage)
    assert one(f, "mashbox:prefix_cache_queries_total") == sum(u[1] for u in usage)
    assert one(f, "mashbox:prefix_cache_hits_total") == 0          # the stub engine caches nothing
    for h in ("mashbox:time_to_first_token_seconds", "mashbox:request_queue_time_seconds",
              "mashbox:e2e_request_latency_seconds", "mashbox:request_time_per_output_token_seconds"):
        assert count(f, h) == 6, h
    assert one(f, "mashbox:e2e_request_latency_seconds_sum") >= one(f, "mashbox:time_to_first_token_seconds_sum") > 0
    assert one(f, "mashbox:num_requests_running") == 0 and one(f, "mashbox:num_requests_waiting") == 0
    assert one(f, "mashbox:kv_cache_usage_perc") == 0
    assert one(f, "mashbox:cache_config_info", block_size=4096, num_blocks=2) == 1
    assert one(f, "mashbox:num_preemptions_total") == 0 and one(f, "mashbox:worker_restarts_total") == 0
    if mode == "worker":
        assert one(f, "mashbox:compiles_after_ready") == 0


@pytest.mark.parametrize("mode", ["one_process", "worker"])
def test_metrics_running_and_waiting_while_busy(tmp_path, mode):
    """One slot, three streams at once: a scrape mid-answer reads 1 running, 2 waiting and the cache full; afterwards
    0 / 0 / 0 and three answered requests."""
    fr, port = server(tmp_path, mode, S=1, step_s=0.05)
    out = []
    ths = [threading.Thread(target=lambda: out.append(TF.post(port, *TF.CHAT))) for _ in range(3)]
    for t in ths:
        t.start()
    seen, t0 = None, time.time()
    while time.time() - t0 < 10:
        f = parse(scrape(port)[2])
        seen = (one(f, "mashbox:num_requests_running"), one(f, "mashbox:num_requests_waiting"),
                one(f, "mashbox:kv_cache_usage_perc"))
        if seen == (1, 2, 1.0):
            break
        time.sleep(0.02)
    assert seen == (1, 2, 1.0), seen
    for t in ths:
        t.join(60)
    assert [c for c, _, _ in out] == [200, 200, 200]
    f = parse(scrape(port)[2])
    assert (one(f, "mashbox:num_requests_running"), one(f, "mashbox:num_requests_waiting"),
            one(f, "mashbox:kv_cache_usage_perc")) == (0, 0, 0)
    assert one(f, "mashbox:request_success_total", finished_reason="length") == 3
    assert count(f, "mashbox:request_queue_time_seconds") == 3 and one(f, "mashbox:request_queue_time_seconds_sum") > 1.0


def test_metrics_count_a_worker_restart(tmp_path):
    """kill -9 of the worker: once the front is ready again, worker_restarts_total reads 1 and an answer counts."""
    fr, port = server(tmp_path, "worker")
    pid = fr.link.proc.pid
    os.kill(pid, signal.SIGKILL)
    t0 = time.time()
    while (fr.link is None or fr.link.proc.pid == pid or not fr.ready) and time.time() - t0 < 30:
        time.sleep(0.02)
    TF.wait_ready(fr)
    TF.full_answer_ok(port)
    f = parse(scrape(port)[2])
    assert one(f, "mashbox:worker_restarts_total") == 1
    assert one(f, "mashbox:request_success_total", finished_reason="length") == 1


def _serve_with_keys(srv, keys):
    return STUB.serve_on_free_port(srv, keys=keys)


@pytest.mark.parametrize("mode", ["one_process", "worker"])
def test_metrics_name_no_tenant_key_or_session(tmp_path, mode):
    """With two API keys, sessions and subagents sending, /metrics needs a key, and no label or value in it names
    a key, a tenant, a session or an agent. Its labels are the server's own (finish reason, bucket bound, cache layout)."""
    keys = ["mbx-test-key-alpha-7f3a9c", "mbx-test-key-bravo-21d8e4"]  # gitleaks:allow
    if mode == "worker":
        fr, _, _, _ = TF.start(tmp_path, S=2)
        TF.wait_ready(fr)
    else:
        fr = SV.Server(STUB.make(S=2, outs=[TF.OUT]), TF.IdChat(), "stub-model", {}, log=lambda m: None)
        fr.ready = True
    port = _serve_with_keys(fr, keys)
    sessions = ["sess-4b1d0e-alpha", "sess-9c2f7a-bravo"]
    for key, sess in zip(keys, sessions):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json.dumps(dict(model="stub-model", max_tokens=4, messages=[dict(role="user", content="1 2")])).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "X-Session-Id": sess,
             "x-claude-code-agent-id": "agent-" + sess})
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.status == 200
    st, _, body = scrape(port)
    assert st == 401, (st, body)
    st, _, txt = scrape(port, keys[1])
    assert st == 200
    f = parse(txt)
    names = {k for fam in f.values() for _, lab, _ in fam["samples"] for k in lab}
    assert names <= {"finished_reason", "le", "block_size", "num_blocks"}, names
    for secret in keys + sessions:
        assert secret not in txt and secret[-6:] not in txt
    assert one(f, "mashbox:request_success_total", finished_reason="length") == 2
