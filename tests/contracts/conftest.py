"""Contract results: what a GGUF row's record names instead of the bytes of every engine file it imported.

A contract is one id for one supported kind: `op/<ggml type>`, `tok/<tokenizer>`, `arch/<architecture>`, `path/run`.
Every test case of a contract records its outputs with `contract(id).record(x)` and asserts its bar (written in the test
before its first run). At the end of the session the results are written to MASHBOX_CONTRACTS_OUT (default
build/contracts.json):

  {"schema": 1, "code_commit": <git HEAD or null>, "dirty": <bool or null>,
   "contracts": {id: {"status": "PASS" | "FAIL" | "SKIP", "fingerprint": sha256 hex, "cases": n}}}

status: FAIL when any case of the id failed (setup, call or teardown), else SKIP when any case skipped, else PASS.
A known failure (pytest.mark.xfail(strict=True) naming the gap) is FAIL here too: pytest stays green for unrelated
changes, and every row that names the id is BROKEN until the gap is fixed.
fingerprint: sha256 over the cases sorted by pytest node id, each case = its node id and the sha256 of every value it
recorded, in call order. It must be the same on every machine, so only exact outputs are recorded: integers (token
ids, greedy ids), bytes, str, bool, and float arrays only with exact=True where the bar is bit for bit (dequantized
values). A float output under a tolerance is never recorded: its bar decides PASS or FAIL, and float summation order
differs between CPU types. The same code gives the same fingerprint; an exact output that moves changes it.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
_CASES = {}        # id -> {nodeid: sha256 object}
_OUTCOME = {}      # id -> {nodeid: "passed" | "failed" | "skipped"}
_USED = {}         # nodeid -> [ids]


def _value_bytes(x, exact=False):
    if isinstance(x, (bytes, bytearray, memoryview)):
        return b"b" + bytes(x)
    if isinstance(x, str):
        return b"s" + x.encode("utf-8")
    a = np.ascontiguousarray(np.asarray(x))
    if a.dtype == object:
        raise TypeError(f"contract.record: an object array cannot be fingerprinted ({a!r:.80})")
    if a.dtype.kind in "fc" and not exact:
        raise TypeError("contract.record: a float array is fingerprinted only with exact=True (a bit-for-bit bar); "
                        "a float output under a tolerance is checked by its bar, not recorded")
    if a.dtype.kind not in "biufc":
        raise TypeError(f"contract.record: dtype {a.dtype} cannot be fingerprinted")
    return b"a" + f"{a.dtype.str}{a.shape}".encode() + a.tobytes()


class Contract:
    def __init__(self, cid, nodeid):
        self.id, self.nodeid = cid, nodeid
        self._h = _CASES.setdefault(cid, {}).setdefault(nodeid, hashlib.sha256())

    def record(self, x, exact=False):
        self._h.update(hashlib.sha256(_value_bytes(x, exact)).digest())
        return x


@pytest.fixture
def contract(request):
    """contract(id) -> Contract for this test case; one case may serve several ids"""
    def get(cid):
        if not isinstance(cid, str) or cid.count("/") != 1 or not all(cid.split("/")):
            raise ValueError(f"contract id {cid!r}: want <kind>/<name>")
        _USED.setdefault(request.node.nodeid, [])
        if cid not in _USED[request.node.nodeid]:
            _USED[request.node.nodeid].append(cid)
        return Contract(cid, request.node.nodeid)
    return get


def pytest_configure(config):
    config.addinivalue_line("markers", "contract: a contract test (tests/contracts/conftest.py)")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep = (yield).get_result()
    for cid in _USED.get(item.nodeid, []):
        seen = _OUTCOME.setdefault(cid, {})
        prev = seen.get(item.nodeid)
        if rep.failed or prev == "failed" or hasattr(rep, "wasxfail"):     # a known failure is a failure
            seen[item.nodeid] = "failed"
        elif rep.skipped:
            seen[item.nodeid] = "skipped"
        elif rep.when == "call":
            seen[item.nodeid] = "passed"


def results():
    """-> {id: {status, fingerprint, cases}} for every id a test asked for"""
    out = {}
    for cid in sorted(_CASES):
        cases, seen = _CASES[cid], _OUTCOME.get(cid, {})
        st = [seen.get(n, "failed") for n in cases]          # a case with no report did not pass
        status = "FAIL" if "failed" in st else "SKIP" if "skipped" in st else "PASS"
        h = hashlib.sha256()
        for n in sorted(cases):
            h.update(n.encode() + b"\0" + cases[n].digest())
        out[cid] = {"status": status, "fingerprint": h.hexdigest(), "cases": len(cases)}
    return out


def _git(*args):
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def pytest_sessionfinish(session, exitstatus):
    if not _CASES:
        return
    path = Path(os.environ.get("MASHBOX_CONTRACTS_OUT") or ROOT / "build" / "contracts.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD")
    st = _git("status", "--porcelain", "--untracked-files=no")
    doc = {"schema": SCHEMA, "code_commit": head, "dirty": None if st is None else bool(st), "contracts": results()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
