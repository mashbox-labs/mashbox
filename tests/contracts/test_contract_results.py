"""The contract results writer (tests/contracts/conftest.py) on its own: status, fingerprint and record rules."""
import json
import subprocess
import sys
import textwrap


def _run(tmp_path, body):
    (tmp_path / "conftest.py").write_text((__import__("pathlib").Path(__file__).with_name("conftest.py")).read_text())
    (tmp_path / "test_x.py").write_text(textwrap.dedent(body))
    __import__("os").utime(tmp_path / "test_x.py", (1_000_000_000, 1_000_000_000))   # same mtime every run: a fast
    # machine rewrites the file within one second, so a bytecode cache keyed on (mtime, size) must never be read
    out = tmp_path / "c.json"
    out.unlink(missing_ok=True)            # a run that writes no results must never read the previous run's file
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                       capture_output=True, text=True, cwd=tmp_path,
                       env={"MASHBOX_CONTRACTS_OUT": str(out), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.exists(), (r.stdout + r.stderr)[-2000:]
    return json.loads(out.read_text())["contracts"]


def test_status_and_fingerprint_rules(tmp_path):
    r = _run(tmp_path, """
        import numpy as np, pytest
        def test_a(contract): contract("k/pass").record([1, 2, 3])
        def test_b(contract): contract("k/fail").record(b"x"); assert False
        @pytest.mark.xfail(strict=True, reason="known gap")
        def test_c(contract): contract("k/known").record("y"); assert False
        def test_d(contract): contract("k/float").record(np.ones(2, np.float32))
        def test_e(contract): contract("k/exact").record(np.ones(2, np.float32), exact=True)
        def test_f(contract): contract("k/skip"); pytest.skip("no")
    """)
    assert {k: v["status"] for k, v in r.items()} == {"k/pass": "PASS", "k/fail": "FAIL", "k/known": "FAIL",
                                                     "k/float": "FAIL", "k/exact": "PASS", "k/skip": "SKIP"}
    r2 = _run(tmp_path, """
        def test_a(contract): contract("k/pass").record([1, 2, 3])
    """)
    assert r2["k/pass"]["fingerprint"] == r["k/pass"]["fingerprint"]
    r3 = _run(tmp_path, """
        def test_a(contract): contract("k/pass").record([1, 2, 4])
    """)
    assert r3["k/pass"]["fingerprint"] != r["k/pass"]["fingerprint"]


def test_each_case_has_its_own_digest(tmp_path):
    """case_sha256 = {node id: sha256 of the values that case recorded}; a case added to an id leaves the digests of
    the cases already there unchanged and moves the id fingerprint"""
    one = _run(tmp_path, """
        def test_a(contract): contract("k/x").record([1, 2])
    """)["k/x"]
    two = _run(tmp_path, """
        def test_a(contract): contract("k/x").record([1, 2])
        def test_b(contract): contract("k/x").record([3])
    """)["k/x"]
    assert set(one["case_sha256"]) == {"test_x.py::test_a"} and two["cases"] == 2
    assert two["case_sha256"]["test_x.py::test_a"] == one["case_sha256"]["test_x.py::test_a"]
    assert two["fingerprint"] != one["fingerprint"]
