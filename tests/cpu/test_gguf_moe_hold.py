"""A mixture-of-experts architecture runs only when the catalog lists it (catalog.arch_rows: a passing chip row of a MoE
file of it). Until then `mashbox run` and `mashbox serve` refuse it with the one unsupported-architecture line, exit 2,
before anything is loaded, so a MoE file is never served on the exact reference path in place of its routed-expert
kernel; `mashbox inspect` says it does not run here, and the Hugging Face run path refuses the same way. A dense
architecture is never held. On CPU with no network, on the tiny qwen35moe fixture."""
import importlib.util
import json
import os

import pytest

import gguf  # noqa: F401,E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
pytest.importorskip("jax")

from mashbox import catalog, cli  # noqa: E402
from mashbox.gguf import archs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)


def _listed(reg):
    return {r["arch"] for r in catalog.arch_rows(reg)}


def _moe(a):
    return catalog.moe_arch(a, catalog.arch_models().get(a))


def _reg_with(base, rows):
    reg = json.loads(json.dumps(base))
    reg["gguf"] = list(reg.get("gguf") or []) + rows
    return reg


def _no_serving(monkeypatch):
    """the load must never be reached: a held file that got past the refusal fails here instead of serving forever"""
    def reached(*a, **k):
        raise AssertionError("an unlisted MoE file was loaded")
    monkeypatch.setattr(cli, "setup_cache", reached)


def test_this_tree_holds_every_unlisted_moe_arch():
    reg = catalog.registry()
    moe = [a for a in catalog.archs() if _moe(a)]
    assert "qwen35moe" in moe, moe
    for a in catalog.archs():
        line = archs.held(a, "m.gguf")
        if _moe(a) and a not in _listed(reg):
            assert line == archs.unsupported(a, "m.gguf"), (a, line)
        else:
            assert line is None, (a, line)                              # dense, or a MoE arch the catalog lists


@pytest.mark.parametrize("cmd", [["--prompt", "hi"], []], ids=["run", "serve"])
def test_run_and_serve_refuse_an_unlisted_moe_file_in_one_line(cmd, tmp_path, capsys, monkeypatch):
    p = FXT.build("qwen35moe-f32", str(tmp_path))
    _no_serving(monkeypatch)
    assert cli.main(["run", p, *cmd]) == 2
    cap = capsys.readouterr()
    err = cap.err.strip().splitlines()
    assert err == [archs.unsupported("qwen35moe", p)], err                # one line, nothing logged before it
    assert "does not run on TPU in mashbox" in err[0] and f"llama-server -m {p}" in err[0]
    assert cap.out == "" and "Traceback" not in cap.err


def test_inspect_agrees_with_run(tmp_path, capsys):
    p = FXT.build("qwen35moe-f32", str(tmp_path))
    assert cli.main(["inspect", p]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["arch"] == "qwen35moe" and out["runs_here"] is False and out["why"] == archs.unsupported("qwen35moe", p)
    d = FXT.build("qwen35-f32", str(tmp_path))
    assert cli.main(["inspect", d]) == 0
    assert json.loads(capsys.readouterr().out)["runs_here"] is True    # a dense file is not held


def test_a_passing_chip_row_of_a_moe_file_lists_and_runs_the_arch(monkeypatch):
    passing = {"v6e": {"status": "passing", "accelerator": "v6e-1"}}
    cases = [
        ([dict(id="t-moe", arch="qwen35moe", moe=True, chips=passing)], True),
        ([dict(id="t-moe", arch="qwen35moe", moe=True, chips={"v6e": {"status": "failing"}})], False),
        ([dict(id="t-moe", arch="qwen35moe", moe=True, chips={})], False),
        ([dict(id="t-dense", arch="qwen35moe", moe=False, chips=passing)], False),   # not a MoE file of it
    ]
    base = catalog.registry()                                           # read once: each case adds its rows to this tree's
    for rows, runs in cases:
        reg = _reg_with(base, rows)
        monkeypatch.setattr(catalog, "registry", lambda path=None, reg=reg: reg)
        assert ("qwen35moe" in _listed(reg)) is runs, rows
        assert (archs.held("qwen35moe") is None) is runs, rows             # listed <=> runs: one rule


def test_the_hold_is_for_moe_archs_only(monkeypatch):
    rows = catalog.arch_rows
    monkeypatch.setattr(catalog, "arch_rows", lambda reg, src=None: [r for r in rows(reg, src) if r["arch"] != "qwen3"])
    assert "qwen3" not in _listed(catalog.registry())
    assert archs.held("qwen3") is None                                  # a dense arch is never held, listed or not
