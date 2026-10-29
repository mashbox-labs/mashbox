"""The CPU standards tier's own folder (tools/standards_cpu.sh). The marker, the family parametrization, the per-family
process split and the skip = FAIL rule live in tests/conftest.py, so they hold for a standards test in any file. Here: a
family process prints its fixture's tokenizer hash (tiny.tokenizer_hash) at the end of its run, for the runner's table.
"""
import os

import tiny


def pytest_terminal_summary(terminalreporter):
    fam = os.environ.get("STD_FAMILY") or None
    if fam and fam != "-":
        terminalreporter.write_line(f"tokenizer_sha256 {fam} {tiny.tokenizer_hash(fam)}")
