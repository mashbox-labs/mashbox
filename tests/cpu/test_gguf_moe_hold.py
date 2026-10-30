"""A mixture-of-experts file runs only when the catalog has a passing chip row of that file: its sha256 (the bytes the
chip record ran, which also fix its repo, revision and name). A passing row lists the architecture (catalog.arch_rows),
and releases no other file of it: the routed-expert kernel's payloads are keyed to the proven file's shapes, so another
file of the architecture could fall to the exact routed path. Until its own row passes, `mashbox run`
and `mashbox serve` refuse a MoE file in one line, exit 2, before anything is loaded; `mashbox inspect` says it does not
run here, and the Hugging Face run path refuses the same way. A dense file is never held. On CPU with no network, on the
tiny qwen35moe and llama + experts fixtures."""
import hashlib
import importlib.util
import json
import os
import shutil

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


PASSING = {"v6e": {"status": "passing", "accelerator": "v6e-1"}}


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _row(p, arch, moe=True, chips=PASSING, sha256=None):
    """a catalog gguf[] row of the file at p, as the record guard holds it: the file's name and its sha256"""
    return dict(id=f"t-{os.path.basename(p)}", arch=arch, moe=moe, chips=chips, file=os.path.basename(p),
                sha256=sha256 or _sha(p))


def _build(name, d):
    d.mkdir(parents=True, exist_ok=True)
    return FXT.build(name, str(d))


def _use(monkeypatch, reg):
    monkeypatch.setattr(catalog, "registry", lambda path=None, reg=reg: reg)


def _no_serving(monkeypatch):
    """the load must never be reached: a held file that got past the refusal fails here instead of serving forever"""
    def reached(*a, **k):
        raise AssertionError("a held MoE file (no passing row of its own) was loaded")
    monkeypatch.setattr(cli, "setup_cache", reached)


def test_this_tree_holds_every_moe_file_without_its_own_row():
    reg = catalog.registry()
    moe = [a for a in catalog.archs() if _moe(a)]
    assert "qwen35moe" in moe, moe
    for a in catalog.archs():
        line = archs.held(a, "m.gguf")                                  # m.gguf: no row of this tree is of it
        if not _moe(a):
            assert line is None, (a, line)                              # a dense file is never held
        elif a not in _listed(reg):
            assert line == archs.unsupported(a, "m.gguf"), (a, line)
        else:
            assert line is not None, (a, line)                          # a listed MoE arch runs only its proven files


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


def test_a_passing_chip_row_of_a_moe_file_lists_the_arch_and_runs_that_file(tmp_path, monkeypatch):
    p = _build("qwen35moe-f32", tmp_path)
    cases = [
        ([_row(p, "qwen35moe")], True),
        ([_row(p, "qwen35moe", chips={"v6e": {"status": "failing"}})], False),
        ([_row(p, "qwen35moe", chips={})], False),
        ([_row(p, "qwen35moe", moe=False)], False),                    # not a MoE file of it
    ]
    base = catalog.registry()                                           # read once: each case adds its rows to this tree's
    for rows, runs in cases:
        reg = _reg_with(base, rows)
        _use(monkeypatch, reg)
        assert ("qwen35moe" in _listed(reg)) is runs, rows
        assert (archs.held("qwen35moe", p, moe=True) is None) is runs, rows   # the row's own file: listed <=> it runs


def test_the_hold_is_for_moe_archs_only(monkeypatch):
    rows = catalog.arch_rows
    monkeypatch.setattr(catalog, "arch_rows", lambda reg, src=None: [r for r in rows(reg, src) if r["arch"] != "qwen3"])
    assert "qwen3" not in _listed(catalog.registry())
    assert archs.held("qwen3") is None                                  # a dense arch is never held, listed or not


# A file with routed experts under a dense architecture's name (llama.cpp writes Mixtral as `llama` + llama.expert_count,
# and mashbox.moe reads it): the hold is the file's, not the name's. It is held until the catalog has a passing chip row
# of a MoE file of that architecture; a dense row of it lists the architecture for dense files only. On the hold by name
# alone, the tiny llama + experts fixture ran on the exact path with no chip record (it was never held).


def test_a_moe_file_of_a_dense_arch_is_held_until_a_moe_row_of_it_passes(tmp_path, capsys, monkeypatch):
    p = FXT.build("llama-moe-f32", str(tmp_path))
    d = FXT.build("llama-f32", str(tmp_path))
    assert not _moe("llama")                                            # the name says dense
    _no_serving(monkeypatch)
    assert cli.main(["run", p, "--prompt", "hi"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [archs.held("llama", p, moe=True)], err
    assert '"llama" does not run on TPU' not in err[0], err             # llama files without routed experts DO run
    assert "with routed experts" in err[0] and '"llama" files without routed experts do' in err[0], err
    assert f"llama-server -m {p}" in err[0], err
    assert cli.main(["inspect", p]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runs_here"] is False and out["why"] == err[0], out
    assert archs.held("llama", d) is None and archs.held("llama", d, moe=False) is None    # a dense llama file: never
    base = catalog.registry()
    for rows, runs in (([_row(p, "llama", moe=False)], False),
                       ([_row(p, "llama", chips={"v6e": {"status": "failing"}})], False),
                       ([_row(p, "llama")], True)):
        reg = _reg_with(base, rows)
        monkeypatch.setattr(catalog, "registry", lambda path=None, reg=reg: reg)
        assert (archs.held("llama", p, moe=True) is None) is runs, rows
        assert archs.held("llama", d, moe=False) is None, rows


# One passing row proves one file. Its sha256 is the identity the record guard binds (with its repo,
# revision and name), and the kernel's expert payloads are keyed to that file's shapes. Another file of the same
# architecture, named MoE or dense, stays held until its own row passes; the same bytes under another name or folder run.


def test_a_passing_row_releases_its_own_file_and_no_other_file_of_the_arch(tmp_path, capsys, monkeypatch):
    a = _build("qwen35moe-f32", tmp_path / "a")
    b = _build("qwen35moe-pyquant", tmp_path / "b")                    # another qwen35moe file: other bytes and shapes
    reg = _reg_with(catalog.registry(), [_row(a, "qwen35moe")])
    _use(monkeypatch, reg)
    assert "qwen35moe" in _listed(reg)                                  # the listing names the architecture
    assert archs.held("qwen35moe", a, moe=True) is None                 # the proven file runs
    line = archs.held("qwen35moe", b, moe=True)
    assert line is not None, "a passing row of file A released file B of the same MoE architecture"
    assert archs.held("qwen35moe", moe=True) is not None                # no file named: nothing is proven
    c = tmp_path / "c" / "renamed.gguf"
    c.parent.mkdir()
    shutil.copyfile(a, c)
    assert archs.held("qwen35moe", str(c), moe=True) is None            # the same bytes: the same proven file
    _use(monkeypatch, _reg_with(reg, [_row(b, "qwen35moe", sha256="0" * 64)]))
    assert archs.held("qwen35moe", b, moe=True) is not None             # a row naming other bytes proves nothing here
    _use(monkeypatch, reg)
    assert "with routed experts" in line and f"llama-server -m {b}" in line, line
    assert '"qwen35moe" does not run on TPU' not in line, line          # the architecture runs: its proven file does
    _no_serving(monkeypatch)
    assert cli.main(["run", b, "--prompt", "hi"]) == 2
    cap = capsys.readouterr()
    assert cap.err.strip().splitlines() == [line], cap.err
    assert cap.out == "" and "Traceback" not in cap.err
    assert cli.main(["inspect", b]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runs_here"] is False and out["why"] == line, out
    assert cli.main(["inspect", a]) == 0
    assert json.loads(capsys.readouterr().out)["runs_here"] is True


def test_a_passing_moe_row_of_a_dense_arch_releases_only_its_own_file(tmp_path, monkeypatch):
    a = _build("llama-moe-f32", tmp_path / "a")
    b = tmp_path / "b" / "llama-moe-other.gguf"
    b.parent.mkdir()
    FXT.write_dense(str(b), "llama", seed=23, layers=4, moe=dict(E=8, K=2, FE=32))   # another llama + experts file
    d = _build("llama-f32", tmp_path / "d")
    _use(monkeypatch, _reg_with(catalog.registry(), [_row(a, "llama")]))
    assert archs.held("llama", a, moe=True) is None                     # the proven file runs
    line = archs.held("llama", str(b), moe=True)
    assert line is not None, "a passing row of llama MoE file A released llama MoE file B"
    assert "with routed experts" in line and '"llama" does not run on TPU' not in line, line
    assert archs.held("llama", d, moe=False) is None                    # a dense llama file: never held
