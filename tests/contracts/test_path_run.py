"""path/run: the real `mashbox run` path on a tiny fixture, and the catalog's refusal that sits on that path.

Bars, written before the first run:
  1. `mashbox run <llama-f32 fixture> --prompt "the hat" --max-tokens 6 --temperature 0` exits 0, its plan names arch
     llama, it loads, and it prints the same answer twice (greedy decoding is deterministic on one machine).
  2. gguf.archs.held (run's refusal, which reads catalog.py), for every ARCHS architecture: a file with no name (path
     None) of a dense-named architecture is never held; of a mixture-of-experts one (catalog.moe_arch) it is always held,
     with one line naming llama.cpp (a file with no name proves nothing).
  3. The MoE hold is per file (MOE 33, DESIGN.md item 14): for every ARCHS architecture, two files F1, F2 with routed
     experts (moe=True) and different bytes, on the shipped registry minus the architecture's rows plus, per sub-case,
     (a) a passing MoE row naming F1's sha256: F1 runs and F2 is held; (b) the same row with no sha256: both held (a row
     with no sha256 lifts nothing); (c) no MoE row: both held. A held file gets one line naming llama.cpp. A file of a
     dense-named architecture without routed experts (moe=False) is never held, in every sub-case.
Recorded (exact): the plan's arch; per ARCHS architecture, one case (so an architecture added to ARCHS adds a case and
moves none a record named): whether it is MoE-named, the no-name decision, the 6 F1/F2 decisions of bar 3 and the dense-file decision on the fixed
registry. The shipped registry's MoE decisions follow models.json's rows and are never recorded: a passing MoE row added
to the catalog must not move the path/run fingerprint of every row. The answer text is a float argmax and is checked by
bar 1, never recorded. Not asserted, by design since the per-file hold: "listed == runnable" (catalog.arch_rows lists
per architecture, held releases per file; the catalog view's tests cover the listing).
"""
import importlib.util
import json
import os

import pytest

jax = pytest.importorskip("jax")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)


def _run(p, tmp_path, capsys):
    from mashbox import cli
    rc = cli.main(["run", p, "--prompt", "the hat", "--max-tokens", "6", "--temperature", "0", "--ctx", "64",
                   "--slots", "2", "--steps", "2", "--cache-dir", str(tmp_path / "xla")])
    cap = capsys.readouterr()
    assert rc in (None, 0), cap.err[-2000:]
    events = [json.loads(line) for line in cap.err.splitlines() if line.startswith("{")]
    return cap.out, events


@pytest.mark.contract
def test_run_path_on_a_tiny_file(contract, tmp_path, capsys, monkeypatch):
    c = contract("path/run")
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p = FXT.build("llama-f32", str(tmp_path))
    out1, ev = _run(p, tmp_path, capsys)
    out2, _ = _run(p, tmp_path, capsys)
    kinds = [e["event"] for e in ev]
    assert "gguf_plan" in kinds and "loaded" in kinds and "done" in kinds, kinds
    arch = next(e for e in ev if e["event"] == "gguf_plan")["arch"]
    assert arch == "llama" and next(e for e in ev if e["event"] == "done")["new_tokens"] >= 1
    assert out1 == out2 and out1.strip(), (out1, out2)
    c.record(arch)


from mashbox.gguf import archs as _GA    # noqa: E402 - after the jax skip; the case list is ARCHS at collection

ARCHS = sorted(_GA.ARCHS)


@pytest.mark.contract
def test_there_is_a_moe_architecture_for_bar_3(contract):
    contract("path/run")
    from mashbox import catalog
    assert any(catalog.moe_arch(a, catalog.arch_models().get(a)) for a in ARCHS), \
        "no MoE architecture in ARCHS: bar 3 has nothing to check"


@pytest.mark.contract
@pytest.mark.parametrize("arch", ARCHS)
def test_catalog_refusal_on_the_run_path(contract, monkeypatch, tmp_path, arch):
    c = contract("path/run")
    from mashbox import catalog
    from mashbox.gguf import archs as GA
    from mashbox.gguf.pick import identity
    reg = catalog.registry()
    moe = bool(catalog.moe_arch(arch, catalog.arch_models().get(arch)))
    c.record(json.dumps([arch, moe]))

    def runs(path, is_moe):
        line = GA.held(arch, path=path, moe=is_moe)
        assert line is None or ("llama.cpp" in line and "\n" not in line), (arch, line)
        return line is None
    noname = runs(None, False)
    assert noname is (not moe), (arch, noname)                                        # bar 2
    f1, f2 = tmp_path / "f1.gguf", tmp_path / "f2.gguf"
    f1.write_bytes(b"GGUF path/run contract file one")
    f2.write_bytes(b"GGUF path/run contract file two")
    sha1 = identity(str(f1))[0]
    assert sha1 and sha1 != identity(str(f2))[0]
    base = [g for g in reg.get("gguf") or [] if g["arch"] != arch]
    row = {"id": "contract-moe-row", "arch": arch, "moe": True, "chips": {"v6e": {"status": "passing"}}}
    got = []
    for sub, gg in (("a", base + [dict(row, sha256=sha1)]), ("b", base + [row]), ("c", base)):
        monkeypatch.setattr(catalog, "registry", lambda gg=gg: dict(reg, gguf=gg))
        d = [runs(str(f1), True), runs(str(f2), True)]
        assert d == ([True, False] if sub == "a" else [False, False]), (arch, sub, d)     # bar 3
        if not moe:
            assert runs(str(f1), False) and runs(str(f2), False), (arch, sub)          # a dense file is never held
        got.append(d)
    c.record(json.dumps([arch, noname, got, True if not moe else None]))
