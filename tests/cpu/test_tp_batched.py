"""Tensor parallel for the batched engine: the gates of tp_batched_checks.py (4 forced CPU devices, the v6e-4 / v5e-4
layout, against one device on a tiny Qwen3.8 of the same architecture), each as its own test.

Slow (a tiny-model build, then the engine's warm-up and requests on 1 and 4 devices, several minutes on CPU): the script
runs once per module and the per-gate tests read its JSON lines. Needs torch and transformers with qwen3_5 to build the
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


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5TextConfig"):
        pytest.skip("transformers without qwen3_5")
    wd = tmp_path_factory.mktemp("tp_batched")
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    env.pop("TP_ONLY", None)
    p = subprocess.run([sys.executable, str(ROOT / "tests" / "cpu" / "tp_batched_checks.py"), str(wd)],
                       capture_output=True, text=True, timeout=3600, env=env)
    gates = {}
    for line in p.stdout.splitlines():
        if line.startswith("{") and '"gate"' in line:
            d = json.loads(line)
            gates[d["gate"]] = d
    return gates, p.stdout[-3000:] + p.stderr[-3000:]


def test_gate_list_is_complete(result):
    gates, tail = result
    assert sorted(g for g in gates if g.startswith("TP")) == GATES, tail


@pytest.mark.parametrize("gate", GATES)
def test_gate(result, gate):
    gates, tail = result
    assert gate in gates, tail
    assert gates[gate]["ok"] is True, json.dumps(gates[gate], default=str)[:3000]
