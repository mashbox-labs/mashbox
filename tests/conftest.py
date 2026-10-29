"""Test markers for CI (CI.md), added from file names so no test file has to move.

- `<area>`: the model id with underscores when the file name carries it (test_hy_nomas_port.py -> hy_nomas), else `core`.
- `t0`: tests/cpu/test_core.py, the static and registry checks that need nothing installed but pytest.
Markers a test can set itself:
- `slow`: over 10 s on the CI runner; the docstring says why.
- `cpu_only`: needs CPU-only stand-ins; skipped on the chip tier (JAX_PLATFORMS=tpu,cpu).
- `quarantine(reason=..., until="YYYY-MM-DD")`: flaky, so CI deselects it; after `until` collection fails until the
  owner fixes or deletes it.
- `standards` / `standards_slow`: the CPU standards tier (tools/standards_cpu.sh), in any test file. A standards test
  with a `family` argument runs once per tiny family (tests/standards/tiny.py); STD_FAMILY=<name> keeps that family's
  tests, STD_FAMILY=- the standards tests that take no family. In a run that selects the marker (-m standards ...), a
  skip inside it is a FAIL, at collection too: a missing dependency or an unbuilt feature never reads green (an unbuilt
  feature is a strict xfail naming the code that must change).
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
STANDARDS = pathlib.Path(__file__).resolve().parent / "standards"


def pytest_configure(config):
    for m in ["t0: static and registry checks (tier T0)", "core: tests of the shared core",
              "slow: over 10 s on the CI runner, reason in the docstring", "cpu_only: skipped on the chip tier",
              "quarantine(reason, until): flaky; deselected in CI until fixed or deleted",
              "standards: the CPU standards tier (tools/standards_cpu.sh); a skip inside it is a FAIL",
              "standards_slow: standards rows run only when speculation code changes"] + [f"{a}: tests of model {a}" for a in AREAS]:
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
    only = os.environ.get("STD_FAMILY") or None
    if only is not None:            # one standards process per family, and one (-) for the tests that take no family
        drop = [it for it in items if it.get_closest_marker("standards") is not None
                and ("family" in getattr(it, "fixturenames", ())) == (only == "-")]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = [it for it in items if it not in drop]


def _tiny():
    import sys
    if str(STANDARDS) not in sys.path:
        sys.path.insert(0, str(STANDARDS))
    import tiny
    return tiny


def pytest_generate_tests(metafunc):
    if "family" in metafunc.fixturenames and metafunc.definition.get_closest_marker("standards") is not None:
        only = os.environ.get("STD_FAMILY") or None
        T = _tiny()
        metafunc.parametrize("family", [T.param(f) for f in T.families(None if only == "-" else only)], scope="module")


def _standards_run(config):
    """True when the run's -m expression selects a standards test (one with the marker, or with standards_slow too);
    `-m "not standards"` is not a standards run"""
    mx = (config.getoption("markexpr", "") or "").strip()
    if not mx:
        return False
    from _pytest.mark.expression import Expression
    e = Expression.compile(mx)
    return any(e.evaluate(lambda n, *_a, _m=m, **_k: n in _m) for m in ({"standards"}, {"standards", "standards_slow"}))


def _standards_file(path):
    p = pathlib.Path(str(path)).resolve()
    if p == STANDARDS or STANDARDS in p.parents:
        return True
    try:
        return p.is_file() and "mark.standards" in p.read_text(errors="ignore")
    except OSError:
        return False


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    out = yield
    rep = out.get_result()
    if rep.skipped and _standards_run(collector.config) and _standards_file(collector.path):
        rep.outcome = "failed"
        rep.longrepr = f"a skip inside the standards marker is a FAIL (at collection): {rep.longrepr}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    rep = out.get_result()
    if (rep.skipped and not hasattr(rep, "wasxfail") and item.get_closest_marker("standards") is not None
            and _standards_run(item.config)):
        rep.outcome = "failed"
        rep.longrepr = f"a skip inside the standards marker is a FAIL: {rep.longrepr}"
