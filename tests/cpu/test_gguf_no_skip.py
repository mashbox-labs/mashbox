"""gguf (gguf-py) is a hard dependency of the package (pyproject gguf==0.19.0), so no GGUF test may skip when it is missing:
a broken install has to fail the suite, not hide the GGUF tests (catalog PLAN B-c)."""
import pathlib
import re

TESTS = pathlib.Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"importorskip\(\s*['\"]" + "gguf" + r"['\"]")


def test_no_gguf_skip():
    hits = [f"{p.relative_to(TESTS)}:{n}" for p in sorted(TESTS.rglob("*.py"))
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if PATTERN.search(line)]
    assert hits == [], f"GGUF tests that skip without gguf: {hits}"


def test_gguf_is_a_declared_dependency():
    pyproject = (TESTS.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'"gguf==[0-9.]+"', pyproject), "gguf is not pinned in pyproject dependencies"
