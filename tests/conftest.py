"""Test markers for CI (CI.md), added from file names so no test file has to move.

- `<area>`: the model id with underscores when the file name carries it (test_hy_nomas_port.py -> hy_nomas), else `core`.
- `t0`: tests/cpu/test_core.py, the static and registry checks that need nothing installed but pytest.
Markers a test can set itself:
- `slow`: over 10 s on the CI runner; the docstring says why.
- `cpu_only`: needs CPU-only stand-ins; skipped on the chip tier (JAX_PLATFORMS=tpu,cpu).
- `quarantine(reason=..., until="YYYY-MM-DD")`: flaky, so CI deselects it; after `until` collection fails until the
  owner fixes or deletes it.
"""
import datetime
import importlib.util
import os
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location("ci_select", pathlib.Path(__file__).resolve().parents[1] / "tools" / "ci_select.py")
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)
AREAS = CS.model_areas()


def pytest_configure(config):
    for m in ["t0: static and registry checks (tier T0)", "core: tests of the shared core",
              "slow: over 10 s on the CI runner, reason in the docstring", "cpu_only: skipped on the chip tier",
              "quarantine(reason, until): flaky; deselected in CI until fixed or deleted"] + [f"{a}: tests of model {a}" for a in AREAS]:
        config.addinivalue_line("markers", m)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    today = datetime.date.today().isoformat()
    on_chip = os.environ.get("JAX_PLATFORMS", "").split(",")[0] == "tpu"
    for it in items:
        it.add_marker(CS.test_area(it.path.name, AREAS))
        if it.path.name in CS.T0_TESTS:
            it.add_marker("t0")
        q = it.get_closest_marker("quarantine")
        if q is not None and not str(q.kwargs.get("until", "")) >= today:
            raise pytest.UsageError(f"{it.nodeid}: quarantine ended ({q.kwargs.get('until')!r}); fix or delete the test (CI.md)")
        if on_chip and it.get_closest_marker("cpu_only") is not None:
            it.add_marker(pytest.mark.skip(reason="cpu_only"))
