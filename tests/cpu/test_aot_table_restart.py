"""A table failure after ready never compiles in a ready process.

A program loaded from the saved executable table that fails in a request (programs.TableFailed) ends the process that
holds the engine; it never warms again in that process, where every compile would be a compile after ready, and the failure rule
restarts the engine's process after a failed device call:
- behind the HTTP front (`mashbox run`): the worker sends one fatal ERROR frame that names the table failure and exits;
  the front answers 503 while it starts a new worker without the table (MASHBOX_AOT=0 for every later worker, so a
  folder that could not be closed is never loaded again), times that start as a first start (it compiles every program,
  so the restart limit of runtime/front.py does not apply to it), and sends a request whose client has received nothing
  once more (a request that is not streamed even when the worker had made some of its tokens).
  The new worker compiles before it says ready: /health's compiles_after_ready, summed over every worker life, stays 0.
- in one process (server.Server): every job of the call gets the error and the process exits (code 70) with no warm-up.
Real worker processes (aot_table_stub.py over api_stub_engine.py) and the real programs counter."""
import json
import os
import subprocess
import sys
import threading
import time

from mashbox.runtime import front as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402
from test_front_failures import CHAT, OUT, WHOLE, FRONTS, _close_fronts, IdChat, chat_text, events, get, post  # noqa: E402,F401


def start(tmp_path, fail="first", closed=True, slow_s=0.0, S=1, step_s=0.03):
    table = tmp_path / "table"
    table.mkdir()
    (table / "manifest.json").write_text("{}")
    log = str(tmp_path / "engine.jsonl")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]
                                                      + [os.environ.get("PYTHONPATH", "")]),
               JAX_PLATFORMS="cpu", MASHBOX_WATCHDOG_MIN_S="5", MASHBOX_START_FLOOR_S="1")
    env.pop("MASHBOX_AOT", None)
    logs = []
    fr = F.Front(IdChat(), "stub-model", {}, "aot_table_stub:make",
                 dict(table=str(table), fail=fail, closed=closed, slow_s=slow_s, S=S, step_s=step_s, outs=[OUT], log=log),
                 warm=True, env=env, log=logs.append)
    FRONTS.append(fr)
    port = STUB.serve_on_free_port(fr)
    return fr, port, log, logs, table


def ready(fr, s=60):
    t = time.time()
    while not fr.ready and time.time() - t < s and not fr.gave_up:
        time.sleep(0.02)
    return fr.ready


def lives(log):
    return [e for e in STUB.read_log(log) if e["ev"] == "warm"]


def test_a_table_failure_restarts_the_worker_cold_and_runs_the_request_again(tmp_path):
    """The loaded program fails before the request's first token. The worker exits; the front answers 503 while the
    next worker compiles (slow_s: far past 2 x the first start, which loaded the table), never gives up on that start,
    and answers the request in full on it. Old code: the front killed the restart as too slow and stopped serving."""
    fr, port, log, logs, table = start(tmp_path, slow_s=12.0)
    assert ready(fr), fr.gave_up
    first = [json.loads(m) for m in logs if '"worker_ready"' in m][0]["ready_s"]
    assert first < 6.0, f"the first start took {first} s: the restart would not be slow enough to test the limit"
    out = {}
    ta = threading.Thread(target=lambda: out.update(r=post(port, CHAT[0], dict(CHAT[1], stream=False), timeout=120)))
    ta.start()
    saw = None
    t = time.time()
    while ta.is_alive() and time.time() - t < 90:
        if not fr.ready and fr.restarts:
            code, raw, h = post(port, CHAT[0], dict(CHAT[1], stream=False))
            if code == 503:
                saw = (code, h.get("Retry-After"), h.get("x-should-retry"))
                break
        time.sleep(0.05)
    ta.join(120)
    assert fr.gave_up is None, fr.gave_up
    assert out["r"][0] == 200, out["r"]
    assert json.loads(out["r"][1])["choices"][0]["message"]["content"] == WHOLE
    assert saw and saw[1] and saw[2] == "true", saw                   # 503 with Retry-After while the worker is down
    w = lives(log)
    assert [x["loaded"] for x in w] == [True, False], w              # the second life never loaded the table
    assert w[1]["aot"] == "0" and w[1]["compiles"] >= 1, w            # it compiled, before it said ready
    assert not table.exists() and [p for p in os.listdir(tmp_path) if p.startswith(".refused-table-")]
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert len(down) == 1 and down[0]["was_ready"] is True and down[0]["table"] == dict(closed=True), down
    lost = [json.loads(m) for m in logs if '"worker_lost_requests"' in m]
    assert lost == [dict(event="worker_lost_requests", code="device_fault", retried=1, ended=0)], lost
    h = get(port, "/health")
    assert h["compiles_after_ready"] == 0 and h["worker_restarts"] == 1, h
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False))
    assert code == 200 and json.loads(raw)["choices"][0]["message"]["content"] == WHOLE
    assert get(port, "/health")["compiles_after_ready"] == 0


def test_a_table_that_could_not_be_closed_is_not_loaded_by_the_restarted_worker(tmp_path):
    """The folder could be neither renamed nor refused (closed False): the next worker of this front must not load it.
    Old code: the restarted worker loaded it again, the retried request failed on it again and got 500."""
    fr, port, log, logs, table = start(tmp_path, closed=False)
    assert ready(fr), fr.gave_up
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False), timeout=120)
    assert code == 200, raw
    assert json.loads(raw)["choices"][0]["message"]["content"] == WHOLE
    w = lives(log)
    assert [x["loaded"] for x in w] == [True, False] and w[1]["aot"] == "0", w
    assert table.is_dir() and not (table / "REFUSED").exists()       # the folder stayed loadable on disk
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert down and down[0]["table"] == dict(closed=False), down
    assert get(port, "/health")["compiles_after_ready"] == 0


def test_a_streamed_request_with_tokens_gets_the_error_and_the_next_worker_answers(tmp_path):
    fr, port, log, logs, table = start(tmp_path, fail="token")
    assert ready(fr), fr.gave_up
    code, raw, _ = post(port, *CHAT, timeout=120)
    ev = events(raw)
    assert chat_text(ev), raw                                         # tokens went out before the failure
    assert [e for _, e in ev if isinstance(e, dict) and "error" in e], raw
    assert ready(fr), fr.gave_up
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False), timeout=120)
    assert code == 200 and json.loads(raw)["choices"][0]["message"]["content"] == WHOLE, raw
    w = lives(log)
    assert [x["loaded"] for x in w] == [True, False], w
    assert get(port, "/health")["compiles_after_ready"] == 0


def test_a_non_streamed_request_with_tokens_is_sent_again_after_a_table_failure(tmp_path):
    """The no-bytes rule on the table's fatal frame: the loaded program fails once the row has tokens; the client of a
    request that is not streamed has received nothing, so the request runs again on the worker that starts without the
    table and gets the whole answer. Old code: 500 (the front counted the worker's tokens as output the client had)."""
    from test_front_failures import tokens_seen
    fr, port, log, logs, table = start(tmp_path, fail="token")
    assert ready(fr), fr.gave_up
    seen = tokens_seen(fr)
    code, raw, _ = post(port, CHAT[0], dict(CHAT[1], stream=False), timeout=120)
    assert code == 200, raw
    assert json.loads(raw)["choices"][0]["message"]["content"] == WHOLE
    pids = list(dict.fromkeys(p for p, _ in seen))
    assert len(pids) == 2 and sum(n for p, n in seen if p == pids[0]) >= 1, seen     # the first life sent tokens
    w = lives(log)
    assert [x["loaded"] for x in w] == [True, False] and w[1]["aot"] == "0", w
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert len(down) == 1 and down[0]["table"] == dict(closed=True) and down[0]["was_ready"] is True, down
    lost = [json.loads(m) for m in logs if '"worker_lost_requests"' in m]
    assert lost == [dict(event="worker_lost_requests", code="device_fault", retried=1, ended=0)], lost
    assert get(port, "/health")["compiles_after_ready"] == 0


ONE_PROCESS = r"""
import json, os, pathlib, sys, threading
sys.path.insert(0, sys.argv[1])
from mashbox.runtime import programs, server as SV
import test_aot_table_first_use as T
tmp = pathlib.Path(sys.argv[2])
e, skipped, d = T.skipped_after_warm(tmp)
programs.install()
programs.mark_ready()                      # what cli.watch_compiles does where the engine's process says ready
logs, hits, codes, warms, ended = [], [], [], [], threading.Event()
srv = SV.Server(e, T.Chat(), "m", {}, log=logs.append)
srv.ready = True
srv.exit = lambda c: (codes.append(c), ended.set())
real = e.warm
e.warm = lambda *a, **k: (warms.append(1), real(*a, **k))[1]
for k in skipped:
    e.aot.ex[k] = T.Broken(RuntimeError("device fault at launch"), hits)
j, L = T.serve_until_failure(srv)
ended.wait(60)
print(json.dumps(dict(codes=codes, warms=len(warms), hits=len(hits), ready=srv.ready,
                      after=programs.counts()["compiles_after_ready"], error=repr(j and j.get("error")),
                      failed_lines=sum('"aot_table"' in x and '"failed"' in x for x in logs))))
"""


def test_one_process_server_ends_on_a_table_failure_with_no_compile_after_ready(tmp_path):
    """The counter is the real programs.counts()["compiles_after_ready"] after mark_ready. Old code warmed every
    program again in the ready process (each compile counted after ready) and served on."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]
                                                      + [os.environ.get("PYTHONPATH", "")]), JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-c", ONE_PROCESS, HERE, str(tmp_path)], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["hits"] == 1 and "TableFailed" in got["error"], got
    assert got["after"] == 0 and got["warms"] == 0, got
    assert got["codes"] == [70] and got["ready"] is False and got["failed_lines"] == 1, got
