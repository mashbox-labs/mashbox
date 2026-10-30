"""Tensor parallel for the batched engine: the gates of tp_batched_checks.py (4 forced CPU devices, the v6e-4 / v5e-4
layout, against one device on a tiny Qwen3.8 of the same architecture), each as its own test.

Slow (a tiny-model build, then the engine's warm-up and requests on 1 and 4 devices, several minutes on CPU): the script
runs once per module and the per-gate tests read its JSON lines. In PR CI the file runs as parts (tools/ci_select.py
SPLIT); a part runs the script with TP_ONLY = the gates it holds (ci_part keeps TOGETHER gates in one part). Needs torch and transformers with qwen3_5 to build the
tiny model, as test_qwen3_8_27b.py.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow

GATES = ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "TP7", "TP8", "TP9"]
SCRIPT = ROOT / "tests" / "cpu" / "tp_batched_checks.py"
TOGETHER = [("TP6", "TP7")]     # TP7 reads the prefix-cache engine TP6's block builds: one CI part, one script run


def ci_part(item):
    """the CI part group of a test (tests/conftest.py --ci-part): its gate, or its TOGETHER group; a part's script run
    computes only the gates its tests read (TP_ONLY)"""
    cs = getattr(item, "callspec", None)
    g = cs.params.get("gate", "") if cs else ""
    return next(("+".join(t) for t in TOGETHER if g in t), g)


def _asked(request):
    """the gates of this file's test_gate items in this pytest run, in GATES order (a CI part holds some of them)"""
    here = request.node.path
    got = {it.callspec.params["gate"] for it in request.session.items
           if it.path == here and "gate" in getattr(getattr(it, "callspec", None), "params", {})}
    return [g for g in GATES if g in got]


@pytest.fixture(scope="module")
def result(request, tmp_path_factory):
    pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5TextConfig"):
        pytest.skip("transformers without qwen3_5")
    wd = tmp_path_factory.mktemp("tp_batched")
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    env.pop("TP_ONLY", None)
    asked = _asked(request)
    if asked and asked != GATES:                    # a CI part: the script computes only its gates
        env["TP_ONLY"] = " ".join(asked)
    p = subprocess.run([sys.executable, str(ROOT / "tests" / "cpu" / "tp_batched_checks.py"), str(wd)],
                       capture_output=True, text=True, timeout=3600, env=env)
    gates = {}
    for line in p.stdout.splitlines():
        if line.startswith("{") and '"gate"' in line:
            d = json.loads(line)
            gates[d["gate"]] = d
    return gates, p.stdout[-3000:] + p.stderr[-3000:]


def _block_names(node):
    """(names a top-level statement stores, names it loads): Name targets, function and lambda arguments, imports"""
    import ast
    st, ld = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            (ld if isinstance(n.ctx, ast.Load) else st).add(n.id)
        elif isinstance(n, ast.arg):
            st.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            st.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            st |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return st, ld


def test_gate_list_is_complete():
    """Every gate tp_batched_checks.py emits has a test here (GATES) and every listed gate is emitted, each in its own
    top-level `if want("TPn"):` block and nowhere else, so TP_ONLY runs exactly the gates a CI part holds and none is
    silently dropped. A block that reads a name only another gate's block sets (TP7 reads TP6's prefix-cache engine)
    runs in that gate's CI part (ci_part), so a part never runs it without the block it needs. A static read of the
    script: a CI part runs some gates, so no one run reports them all; the sweep runs the whole file."""
    import ast
    tree = ast.parse(SCRIPT.read_text())

    def want_gate(node):
        t = node.test if isinstance(node, ast.If) else None
        if isinstance(t, ast.Call) and getattr(t.func, "id", "") == "want" and len(t.args) == 1:
            return t.args[0].value
        return None

    def emits(node):
        return [n.args[0].value for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "emit"
                and n.args and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith("TP")]
    blocks, module = {}, set()
    for node in tree.body:
        g = want_gate(node)
        if g is None:
            assert not emits(node) or isinstance(node, ast.FunctionDef), ast.unparse(node)[:200]   # emit only in its block
            module |= _block_names(node)[0] if not isinstance(node, ast.FunctionDef) else {node.name}
        else:
            assert g not in blocks, f"two want({g!r}) blocks"
            assert emits(node) == [g], (g, emits(node))
            blocks[g] = _block_names(node)
    assert sorted(blocks, key=GATES.index) == GATES and sorted(blocks) == sorted(set(GATES))
    group = {g: next(("+".join(t) for t in TOGETHER if g in t), g) for g in GATES}
    for g, (st, ld) in blocks.items():
        for other, (st2, _) in blocks.items():
            needs = (ld - st - module) & st2
            if other != g and needs:
                assert group[g] == group[other], f"{g} reads {sorted(needs)} from {other}'s block: put them in TOGETHER"


@pytest.mark.parametrize("gate", GATES)
def test_gate(result, gate):
    gates, tail = result
    assert gate in gates, tail
    assert gates[gate]["ok"] is True, json.dumps(gates[gate], default=str)[:3000]
