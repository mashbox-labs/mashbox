"""Qwen3.8-27B port: the CPU checks in qwen3_8_27b_checks.py (tiny model of the same architecture vs transformers)."""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_cpu_checks(tmp_path):
    pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5TextConfig"):
        pytest.skip("transformers without qwen3_5")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "cpu" / "qwen3_8_27b_checks.py"), str(tmp_path)],
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
