"""Qwen3.8-27B port: the CPU checks in qwen3_8_27b_checks.py (tiny model of the same architecture vs transformers), and on
the same tiny model the engine checks that guard batching: prefix_cache_checks.py, continuous_batching_checks.py
(each check its own label), engine_batch_checks.py (the edge cases) and determinism_checks.py (the CPU
form of single-user determinism: one user alone, greedy, same tokens run to run, from its own cache and after a
restart), each gate as its own test.

All of these are slow (tiny-model builds and many compiles on CPU, minutes in all): the scripts run once per module
(shared fixture) and the per-gate tests only read their JSON. Gates that fail today for a known open engine item are
xfail(strict=True) with the item named, so they show as expected failures and turn red the day they start passing
without the marker being removed.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CPU = ROOT / "tests" / "cpu"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tiny_wd(tmp_path_factory):
    pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5TextConfig"):
        pytest.skip("transformers without qwen3_5")
    wd = tmp_path_factory.mktemp("qwen3_8_27b")
    tok = os.environ.get("MASHBOX_QWEN_TOKENIZER_DIR")      # a Qwen tokenizer dir turns on the template and server checks
    r = subprocess.run([sys.executable, str(CPU / "qwen3_8_27b_checks.py"), str(wd)] + ([tok] if tok else []),
                       capture_output=True, text=True, timeout=3600)
    return wd, r


def test_cpu_checks(tiny_wd):
    _, r = tiny_wd
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


SCRIPTS = ("prefix_cache_checks.py", "continuous_batching_checks.py", "engine_batch_checks.py", "determinism_checks.py")
GATES = {"prefix_cache_checks.py": ["P1", "P2", "P3", "P4", "P5", "P6"],
         "continuous_batching_checks.py": ["C1", "C2", "C3", "C4", "C5", "C6"],
         "engine_batch_checks.py": ["E2", "E4", "E18_engine", "E18_server", "E6", "E8", "E7", "E19", "E26", "E33",
                                    "E7_lqd", "E7_owner", "E7_hangup", "E_connect", "E14", "E14_cache"],
         "determinism_checks.py": ["D1", "D2", "D3"]}
OPEN = {("continuous_batching_checks.py", "C6"): "batch-invariant decode not built: greedy rows decode on row buckets "
                                                  "1/4/S and the bucket changes the rounding (one row of 6 flips at a "
                                                  "near tie)"}


# Gates that pass or fail on wall-clock time, so on box speed: the sweep runs them, PR CI deselects them and skips their
# runs (MASHBOX_SKIP_TIMING, below; CI.md).
# The "merge": false latency gate: B's first token comes before A ends and B's p95 first-token time stays under double.
# Its other condition (no compile in the measured runs) is the serve-time compile gates' class; those stay in PR CI.
TIMING = {("engine_batch_checks.py", "E33")}
# PR CI sets MASHBOX_SKIP_TIMING=1: the scripts then skip the runs of their TIMING gates too (deselecting the test alone
# left the wall-clock gate's warm pairs and 8 runs inside the script's time); the sweep runs without it and checks every gate.
SKIP_TIMING = os.environ.get("MASHBOX_SKIP_TIMING") == "1"


def expected_gates(script):
    """the gates a script reports in this run: GATES[script], less its TIMING gates under MASHBOX_SKIP_TIMING=1"""
    return [g for g in GATES[script] if not (SKIP_TIMING and (script, g) in TIMING)]


class _Gates(dict):
    """script -> its gates, the script run once on first use (the scripts read only the tiny model, not each other's
    output), so a CI part that holds one script's tests runs that script alone (ci_part below)"""
    def __init__(self, wd):
        super().__init__()
        self.wd = wd

    def __missing__(self, s):
        p = subprocess.run([sys.executable, str(CPU / s), str(self.wd)], capture_output=True, text=True, timeout=3600)
        lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
        R = json.loads(lines[-1]) if lines else {}
        g = R.get("gates") or {k[:-5]: v for k, v in R.items() if k.endswith("_pass")}
        self[s] = dict(gates=g, result=R, tail=p.stdout[-2000:] + p.stderr[-2000:])
        return self[s]


@pytest.fixture(scope="module")
def gates(tiny_wd):
    wd, r = tiny_wd
    if not (wd / "tiny-bf16.bshj").exists():
        pytest.fail("qwen3_8_27b_checks.py did not write the tiny model: " + r.stdout[-1000:] + r.stderr[-1000:])
    return _Gates(wd)


def ci_part(item):
    """the CI part group of a test (tests/conftest.py --ci-part): its script, so one part runs each script once"""
    cs = getattr(item, "callspec", None)
    return cs.params.get("script", "") if cs else ""


@pytest.mark.parametrize("script", SCRIPTS)
def test_gate_list_is_complete(gates, script):
    """Every gate a script reports has a test below, and every gate it should report in this run is reported: all of
    GATES, or all but the TIMING gates under MASHBOX_SKIP_TIMING=1 (PR CI), so none is silently dropped."""
    assert sorted(gates[script]["gates"]) == sorted(expected_gates(script)), gates[script]["tail"]


@pytest.mark.parametrize("script,gate", [
    pytest.param(s, g, marks=([pytest.mark.xfail(strict=True, reason=OPEN[(s, g)])] if (s, g) in OPEN else [])
                 + ([pytest.mark.timing] if (s, g) in TIMING else []))
    for s in SCRIPTS for g in GATES[s]])
def test_gate(gates, script, gate):
    r = gates[script]
    detail = {k: v for k, v in r["result"].items() if k == gate or k.startswith(gate + "_")}
    assert r["gates"].get(gate) is True, json.dumps(detail, default=str)[:3000] or r["tail"]
