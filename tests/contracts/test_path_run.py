"""path/run: the real `mashbox run` path on a tiny fixture, and the catalog's refusal that sits on that path.

Bars, written before the first run:
  1. `mashbox run <llama-f32 fixture> --prompt "the hat" --max-tokens 6 --temperature 0` exits 0, its plan names arch
     llama, it loads, and it prints the same answer twice (greedy decoding is deterministic on one machine).
  2. gguf.archs.held (run's refusal, which reads catalog.py): None for every dense architecture in ARCHS; for a
     mixture-of-experts one, None exactly when catalog.arch_rows lists it, else one line naming llama.cpp.
  3. The MoE decision follows the registry: adding a passing MoE chip row lifts the refusal, removing it restores it.
Recorded (exact): the plan's arch, the held decision of every dense ARCHS architecture, and of one MoE architecture
with and without a passing MoE row. The shipped registry's MoE decisions follow models.json's rows (bar 2 checks them)
and are never recorded: a passing MoE row added to the catalog must not move the path/run fingerprint of every row. The answer text is a float argmax and is checked by bar 1, never recorded.
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


def _decisions(archs):
    from mashbox.gguf import archs as GA
    return {a: GA.held(a) for a in sorted(archs.ARCHS)}


@pytest.mark.contract
def test_catalog_refusal_on_the_run_path(contract, monkeypatch):
    c = contract("path/run")
    from mashbox import catalog
    from mashbox.gguf import archs as GA
    reg = catalog.registry()
    now = _decisions(GA)
    for a, line in now.items():
        moe = catalog.moe_arch(a, catalog.arch_models().get(a))
        listed = any(r["arch"] == a for r in catalog.arch_rows(reg))
        if not moe:
            assert line is None, (a, line)
        else:
            assert (line is None) == listed, (a, line, listed)
            assert line is None or ("llama.cpp" in line and "\n" not in line), line
    dense = {a: v is None for a, v in now.items() if not catalog.moe_arch(a, catalog.arch_models().get(a))}
    c.record(json.dumps(dense, sort_keys=True))
    moe_archs = [a for a in GA.ARCHS if catalog.moe_arch(a, catalog.arch_models().get(a))]
    assert moe_archs, "no MoE architecture in ARCHS: bar 3 has nothing to check"
    a = moe_archs[0]
    base = [g for g in reg.get("gguf") or [] if g["arch"] != a]
    row = {"id": "contract-moe-row", "arch": a, "moe": True, "chips": {"v6e": {"status": "passing"}}}
    for gg, want in ((base + [row], True), (base, False)):
        monkeypatch.setattr(catalog, "registry", lambda gg=gg: dict(reg, gguf=gg))
        got = GA.held(a) is None
        assert got == want and got == any(r["arch"] == a for r in catalog.arch_rows(catalog.registry())), (a, gg[-1:])
        c.record(json.dumps([a, got]))
