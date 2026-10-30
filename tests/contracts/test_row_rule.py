"""The row-validity rule (tools/contracts.py), written before the code: a GGUF row stands on the contracts of its kinds.

  contract set   arch/<arch>, op/<type> for every tensor type (MoE rows also op/<type>:experts), the tokenizer's
                 tok/ id, and path/run; nothing else
  VALID          every id of the set is named by the record and PASS now with the fingerprint the record names
  FLAGGED        every id PASS, and at least one fingerprint moved: one QUICK re-run on the real file
  BROKEN         an id of the set FAILs, SKIPs, is missing from the results, or is not named by the record
A change that moves no contract result (a catalog view, docs, a wording) leaves every row VALID.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("mashbox_contracts", os.path.join(ROOT, "tools", "contracts.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

SET = ["arch/qwen2", "op/F32", "op/Q4_K", "op/Q6_K", "path/run", "tok/qwen2"]


def _res(fp=None, status=None):
    fp, status = fp or {}, status or {}
    return {i: {"status": status.get(i, "PASS"), "fingerprint": fp.get(i, "f-" + i), "cases": 1} for i in SET}


def test_contract_set_of_a_row():
    assert C.row_contracts("qwen2", ["Q6_K", "F32", "Q4_K", "Q4_K"], "gpt2", "qwen2") == SET
    assert C.row_contracts("gemma3", ["BF16"], "llama", None) == ["arch/gemma3", "op/BF16", "path/run", "tok/spm"]
    assert C.row_contracts("qwen35moe", ["Q8_0"], "gpt2", "qwen35", moe=True) == [
        "arch/qwen35moe", "op/Q8_0", "op/Q8_0:experts", "path/run", "tok/qwen35"]


def test_valid_flagged_broken():
    named = {i: "f-" + i for i in SET}
    assert C.row_state(SET, named, _res())[0] == "VALID"
    st, why = C.row_state(SET, named, _res(fp={"tok/qwen2": "moved"}))
    assert st == "FLAGGED" and any("tok/qwen2" in w for w in why), why
    for bad in ("FAIL", "SKIP"):
        st, why = C.row_state(SET, named, _res(status={"op/Q4_K": bad}))
        assert st == "BROKEN" and any("op/Q4_K" in w for w in why), why
    r = _res()
    del r["path/run"]
    assert C.row_state(SET, named, r)[0] == "BROKEN"                     # not run = not proven
    short = dict(named)
    del short["op/Q6_K"]
    st, why = C.row_state(SET, short, _res())
    assert st == "BROKEN" and any("op/Q6_K" in w for w in why), why      # a record that skips a contract never stands
    st, _ = C.row_state(SET, named, _res(fp={"tok/qwen2": "moved"}, status={"arch/qwen2": "FAIL"}))
    assert st == "BROKEN"                                                # BROKEN wins over FLAGGED


def test_ids_outside_the_set_change_nothing():
    named = {i: "f-" + i for i in SET}
    r = _res()
    r["arch/gemma4"] = {"status": "FAIL", "fingerprint": "x", "cases": 1}
    r["op/IQ1_S"] = {"status": "FAIL", "fingerprint": "y", "cases": 1}
    assert C.row_state(SET, named, r)[0] == "VALID"


def test_only_a_named_case_that_moved_flags_a_row():
    """per case: the record's results file holds each named id's case digests (case_sha256); a moved id fingerprint
    flags the row only when a case the record named changed its exact outputs or is gone. A case added since the record
    moves no output the record named (it must still PASS its bar). No case digests to compare: the id moved."""
    named = {i: "f-" + i for i in SET}
    cases = {i: {"t.py::a": "h1"} for i in SET}
    r = _res(fp={"tok/qwen2": "new"})
    r["tok/qwen2"]["case_sha256"] = {"t.py::a": "h1", "t.py::b": "h2"}         # a case added, the named one equal
    assert C.row_state(SET, named, r, cases) == ("VALID", [])
    r["tok/qwen2"]["case_sha256"] = {"t.py::a": "hX", "t.py::b": "h2"}         # the named case's output moved
    st, why = C.row_state(SET, named, r, cases)
    assert st == "FLAGGED" and any("t.py::a" in w for w in why), why
    r["tok/qwen2"]["case_sha256"] = {"t.py::b": "h2"}                          # the named case is gone
    assert C.row_state(SET, named, r, cases)[0] == "FLAGGED"
    del r["tok/qwen2"]["case_sha256"]
    assert C.row_state(SET, named, r, cases)[0] == "FLAGGED"                   # nothing to compare per case
    assert C.row_state(SET, named, _res(fp={"tok/qwen2": "new"}))[0] == "FLAGGED"   # no named cases: id decides
    r = _res(fp={"tok/qwen2": "new"}, status={"tok/qwen2": "FAIL"})
    r["tok/qwen2"]["case_sha256"] = {"t.py::a": "h1"}
    assert C.row_state(SET, named, r, cases)[0] == "BROKEN"                    # a FAIL is never saved by a case


def test_a_new_architecture_only_adds_path_run_cases():
    """path/run's catalog refusal records one case per ARCHS architecture, so adding an architecture adds a case and
    moves none a record named (GGUF-ENGINE 2, 10-30: one record over every architecture moved a named case, so adding
    qwen3vl would have flagged every row)"""
    pytest.importorskip("jax")
    from mashbox.gguf import archs
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider",
                        os.path.join(ROOT, "tests", "contracts", "test_path_run.py")],
                       capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    got = {ln.split("::", 1)[1] for ln in r.stdout.splitlines() if "::test_catalog_refusal_on_the_run_path" in ln}
    assert got == {f"test_catalog_refusal_on_the_run_path[{a}]" for a in archs.ARCHS}, r.stdout[-3000:] + r.stderr[-2000:]
