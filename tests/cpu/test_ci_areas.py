"""tools/ci_select.py: the GGUF reader, its ops, fixtures and test_gguf_*.py are their own CI area (`gguf`), so a GGUF
change runs the GGUF tests and a core change runs them too; nothing GGUF is left to an area that never runs it."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ci_select", ROOT / "tools" / "ci_select.py")
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)


def test_gguf_is_an_area_with_its_files_and_tests():
    areas = CS.model_areas()
    assert "gguf" in areas
    tests = sorted(p.name for p in (ROOT / "tests" / "cpu").glob("test_gguf_*.py"))
    assert tests and all(CS.test_area(t, areas) == "gguf" for t in tests)
    for p in ("src/mashbox/gguf/reader.py", "src/mashbox/gguf/archs/dense.py", "src/mashbox/ops/__init__.py",
              "tools/gguf_check.py", "tools/gguf_fixtures/fixtures.py", "tests/fixtures/gguf/qwen3-f32.json"):
        assert CS.select([p], areas) == ["gguf"], p
    assert "gguf" in CS.select(["src/mashbox/cli.py"], areas)            # core changes run the GGUF tests too
    # cli.main imports gguf/__init__.py and gguf/resolve.py on every run, for every model: a change to them runs every area
    for p in ("src/mashbox/gguf/__init__.py", "src/mashbox/gguf/resolve.py"):
        assert set(CS.select([p], areas)) >= {"gguf", "hy_nomas", "qwen3_8_27b"}, p
    assert "gguf" in CS.select(["src/mashbox/runtime/engine.py"], areas)


def test_t1_shards_run_every_file_of_an_area_exactly_once():
    """A sharded area (SHARDS) runs as several T1 jobs; a file dealt to no job is never tested and a file dealt twice
    wastes runner time, so the shards of every area partition its test files, and the matrix has one row per shard."""
    areas = CS.model_areas()
    for area in list(areas) + ["core"]:
        files = CS.area_files(area, areas)
        n = CS.SHARDS.get(area, 1)
        dealt = [f for i in range(1, n + 1) for f in CS.shard_files(files, f"{i}/{n}")]
        assert sorted(dealt) == sorted(files) and len(dealt) == len(set(dealt)), area
        assert all(CS.shard_files(files, f"{i}/{n}") for i in range(1, n + 1)), f"{area}: an empty shard"
        rows = [r for r in CS.jobs([area]) if r["area"] == area]
        assert [r["shard"] for r in rows] == [f"{i}/{n}" for i in range(1, n + 1)], area
    every = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests" / "cpu").rglob("test_*.py") if p.name not in CS.T0_TESTS}
    assert every == {f for a in list(areas) + ["core"] for f in CS.area_files(a, areas)}   # every T1 file has an area


def test_t1_shards_split_an_area_by_time():
    """The shards of an area share its time (SECONDS, per test file): the slowest shard is at most its longest file or
    a quarter over an even split, whichever is more, and stays under 60% of the T1 timeout, so no job runs close to it."""
    import re
    t1 = (ROOT / ".github" / "workflows" / "ci.yml").read_text().split("\n  t1:\n", 1)[1]
    limit = 60 * int(re.search(r"^    timeout-minutes: (\d+)$", t1, re.M).group(1))
    areas = CS.model_areas()
    for area, n in CS.SHARDS.items():
        files = CS.area_files(area, areas)
        sec = {f: CS.SECONDS.get(f, CS.SECONDS_DEFAULT) for f in files}
        loads = [sum(sec[f] for f in CS.shard_files(files, f"{i}/{n}")) for i in range(1, n + 1)]
        bound = max(max(sec.values()), 1.25 * sum(loads) / n)
        assert max(loads) <= bound, f"{area}: shard seconds {loads}, bound {bound:.0f}"
        assert max(loads) <= 0.6 * limit, f"{area}: slowest shard {max(loads)} s of a {limit} s T1 timeout"
