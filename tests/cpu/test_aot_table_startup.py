"""A table failure in the warm-up, before ready, follows A13's one recovery rule: the worker ends.

A program loaded from the saved executable table that fails in cli.warm's two short requests or in Engine.warm (here at
its final barrier) raises programs.TableFailed out of the warm-up; the worker sends the table's fatal frame with
closed (whether its folder was renamed or refused in place) and exits, never warming a second time in that process. The
front starts the next worker at once without the table (MASHBOX_AOT=0 for every later worker, so a folder that could not
be closed is never loaded again); that failure is not counted as a failed start. The next worker compiles every program
before it says ready and answers. Real worker processes (aot_startup_stub.py: the stand-in engine of test_aot_table.py
through the real cli.warm and Engine.warm), the real quarantine (closed=False forces its rename and REFUSED write to
fail) and the real programs counter. Old code: the worker caught the failure and warmed again in the same process, and
with closed=False the front never heard of the table, so later workers loaded it again."""
import json
import os
import sys
import time

import pytest

pytest.importorskip("jax")

from mashbox.runtime import front as F  # noqa: E402
from mashbox.runtime import programs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402
from test_aot_table import cli_start, engine  # noqa: E402
from test_front_failures import FRONTS, _close_fronts, IdChat, get, post  # noqa: E402,F401

BODY = dict(messages=[dict(role="user", content="5 6 7")], max_tokens=6, temperature=0, stream=False)


def start(tmp_path, capsys, where, closed):
    cache = tmp_path / "cache"
    (cache / "aot").mkdir(parents=True, mode=0o700)
    cli_start(cache / "aot" / "dep-a", capsys)            # an earlier start saved the table
    assert (cache / "aot" / "dep-a" / "manifest.json").is_file()
    log = str(tmp_path / "lives.jsonl")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in sys.path if p.endswith("src")]
                                                      + [os.environ.get("PYTHONPATH", "")]),
               JAX_PLATFORMS="cpu", MASHBOX_WATCHDOG_MIN_S="30")
    env.pop("MASHBOX_AOT", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    logs = []
    fr = F.Front(IdChat(), "stub-model", {}, "aot_startup_stub:make",
                 dict(cache=str(cache), where=where, closed=closed, log=log), warm=False, env=env, log=logs.append)
    FRONTS.append(fr)
    fr.counted, real = [], fr._start_failed           # set long before the first worker life ends (it imports jax)
    fr._start_failed = lambda *a: (fr.counted.append(a), real(*a))[1]
    port = STUB.serve_on_free_port(fr)
    return fr, port, logs, log, cache / "aot"


def ready(fr, s=600):
    t = time.time()
    while not fr.ready and time.time() - t < s and not fr.gave_up:
        time.sleep(0.05)
    return fr.ready


@pytest.mark.parametrize("closed", [True, False], ids=["closed", "unclosable"])
@pytest.mark.parametrize("where", ["short", "barrier"])
def test_a_table_failure_in_the_warm_up_ends_the_worker_and_the_next_starts_without_the_table(tmp_path, capsys, where,
                                                                                              closed):
    fr, port, logs, log, aot = start(tmp_path, capsys, where, closed)
    assert ready(fr), (fr.gave_up, logs[-20:])
    lives = STUB.read_log(log)
    assert [x["ev"] for x in lives] == ["table_failed", "warm"], lives
    bad, good = lives
    assert bad["aot"] is None and bad["closed"] is closed and (bad["program"] == "warm") is (where == "barrier"), bad
    assert ("final barrier" if where == "barrier" else "at launch") in bad["error"], bad
    assert good["aot"] == "0" and good["used"] is False and good["compiles"] >= 1 and good["pid"] != bad["pid"], good
    down = [json.loads(m) for m in logs if '"worker_down"' in m]
    assert len(down) == 1 and down[0]["was_ready"] is False and down[0]["table"] == dict(closed=closed), down
    assert down[0]["fault"] == "device_fault", down
    off = [json.loads(m) for m in logs if '"aot_table_off"' in m]
    assert off == [dict(event="aot_table_off", closed=closed)], off
    assert fr.env["MASHBOX_AOT"] == "0" and fr.counted == [] and fr.gave_up is None      # not a failed start
    assert not [m for m in logs if '"worker_start_failed"' in m or '"worker_gave_up"' in m], logs
    refused = [n for n in os.listdir(aot) if n.startswith(".refused-dep-a-")]
    if closed:
        assert len(refused) == 1 and not (aot / "dep-a").exists(), os.listdir(aot)
    else:       # the folder stayed loadable on disk; only the front's MASHBOX_AOT=0 keeps later workers off it
        assert refused == [] and (aot / "dep-a" / "manifest.json").is_file() and not (aot / "dep-a" / "REFUSED").exists()
        assert engine().aot.load(aot / "dep-a", programs.table_meta("dep-a"))["loaded"] > 0
    code, raw, _ = post(port, "/v1/chat/completions", BODY, timeout=120)
    assert code == 200, raw
    want = engine()
    want.warm(workers=2)
    ids = want.generate([[5, 6, 7]], max_new=6)["ids"][0]
    assert json.loads(raw)["choices"][0]["message"]["content"] == "".join(f"{t} " for t in ids).rstrip(), raw
    h = get(port, "/health")
    assert h["compiles_after_ready"] == 0 and h["worker_restarts"] == 1, h
