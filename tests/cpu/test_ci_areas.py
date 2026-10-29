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
