"""Test markers for CI (CI.md), added from file names so no test file has to move.

- `<area>`: the model id with underscores when the file name carries it (test_hy_nomas_port.py -> hy_nomas), else `core`.
- `gguf_rows`: also on the tests of the rows leg (tools/ci_select.py rows_files), which a gguf-rows change runs.
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
Every test leaves each mashbox module that was loaded when it started as the same object (_mashbox_modules_kept): a test
that drops or re-imports one fails at its teardown, by name.
"""
import datetime
import importlib.util
import os
import pathlib
import sys
import types

import pytest

_spec = importlib.util.spec_from_file_location("ci_select", pathlib.Path(__file__).resolve().parents[1] / "tools" / "ci_select.py")
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)
AREAS = CS.model_areas()
ROWS_FILES = {str(pathlib.Path(__file__).resolve().parents[1] / f) for f in CS.area_files(CS.ROWS, AREAS)}
STANDARDS = pathlib.Path(__file__).resolve().parent / "standards"


def pytest_configure(config):
    for m in ["t0: static and registry checks (tier T0)", "core: tests of the shared core",
              "slow: over 10 s on the CI runner, reason in the docstring", "cpu_only: skipped on the chip tier",
              "quarantine(reason, until): flaky; deselected in CI until fixed or deleted",
              "standards: the CPU standards tier (tools/standards_cpu.sh); a skip inside it is a FAIL",
              "standards_slow: standards rows run only when speculation code changes"] + [f"{a}: tests of model {a}" for a in AREAS] \
            + [f"{CS.ROWS}: tests outside gguf that read the gguf rows of models.json (tools/ci_select.py rows_files)"]:
        config.addinivalue_line("markers", m)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    today = datetime.date.today().isoformat()
    on_chip = os.environ.get("JAX_PLATFORMS", "").split(",")[0] == "tpu"
    for it in items:
        it.add_marker(CS.test_area(it.path.name, AREAS))
        if str(it.path) in ROWS_FILES:
            it.add_marker(CS.ROWS)
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


def mashbox_modules():
    """{name: module} of every loaded mashbox module (mashbox and mashbox.*)."""
    return {n: m for n, m in list(sys.modules.items()) if n == "mashbox" or n.startswith("mashbox.")}


def put_back_mashbox_modules(before):
    """Make the mashbox modules exactly `before` again (a mashbox_modules() taken earlier); returns the names that were
    dropped or replaced since. Only when one was: each package put back holds its own submodules again (importing the
    other copy of a submodule sets the kept parent's attribute to it), and the mashbox modules loaded since then are
    dropped from sys.modules and from the packages put back (each may be of the other copy or hold names taken from it),
    so the next import makes them again under the modules put back."""
    changed = sorted(n for n, m in before.items() if sys.modules.get(n) is not m)
    if changed:
        for n in mashbox_modules():
            if n not in before:
                del sys.modules[n]
        sys.modules.update(before)
        for n, m in before.items():
            if not isinstance(m, types.ModuleType):
                continue
            parent, _, child = n.rpartition(".")
            held = getattr(before.get(parent), child, None)
            if isinstance(held, types.ModuleType) and held.__name__ == n:
                setattr(before[parent], child, m)
            for k, v in list(vars(m).items()):
                if isinstance(v, types.ModuleType) and v.__name__ == f"{n}.{k}" and v.__name__ not in before:
                    delattr(m, k)
    return changed


@pytest.fixture(autouse=True)
def _mashbox_modules_kept():
    """A test that drops or re-imports a mashbox module that was loaded when it started leaves the tests after it two
    copies of that module: what was made from the old copy (a Front raising the old agents.NotReady) no longer matches
    the new copy (the server's `except NotReady` misses it and answers 500, not 503). Such a test fails here at its
    teardown, naming the modules; the old modules are put back for the tests after it. Not caught here: importlib.reload
    (the module object stays, its classes are new), a drop at collection or in a module- or session-scoped fixture
    (before the test's own snapshot), and a test that puts sys.modules back itself but leaves a package's attribute on
    the other copy."""
    before = mashbox_modules()
    yield
    changed = put_back_mashbox_modules(before)
    if changed:
        pytest.fail(f"this test dropped or re-imported {len(changed)} mashbox module(s) loaded before it: "
                    f"{', '.join(changed[:8])}{' ...' if len(changed) > 8 else ''} (the tests after it would get a "
                    "second copy of each)",
                    pytrace=False)
