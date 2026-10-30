"""Routed-expert GGUF files of the dense-attention architectures vs llama.cpp a37cc46's CPU logits (tiny random
fixtures, tools/gguf_fixtures; no chip, no network):

  qwen3moe-f32   qwen3 attention, 8 routed experts (2 used), softmax router, weights renormalised (llama.cpp's graph)
  qwen2moe-f32   qwen2 attention, 8 experts (3 used), no renormalisation, a sigmoid-gated shared expert
  llama-moe-f32  architecture `llama` + llama.expert_count (how llama.cpp writes Mixtral), 8 experts (2 used)

Bound: NMSE <= 1e-9, fixed before the first run (the qwen35moe float graph measures 2.2e-11). Each fixture runs as one
prefill of 16 tokens and as a prefill of 8 then 8 one-token decode steps (the K/V cache path)."""
import json
import os

import numpy as np
import pytest

import gguf  # noqa: E402  (a hard dependency: never skipped)
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox.gguf import open_model  # noqa: E402
from mashbox.gguf.reader import GGUFError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

NMSE_MAX = 1e-9
MOE = ["qwen3moe-f32", "qwen2moe-f32", "llama-moe-f32"]


def fixture(name, tmp):
    meta = json.load(open(os.path.join(FX, f"{name}.json")))
    p = FXT.build(name, str(tmp))
    assert FXT.sha256(p) == meta["gguf_sha256"], f"{name}: fixture bytes changed; rerun tools/gguf_fixtures/fixtures.py make"
    ref = np.fromfile(os.path.join(FX, f"{name}.logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    return p, meta, ref


def listed(monkeypatch, arch, path):
    """The catalog releases the MoE file at `path` the one way it does (gguf.archs.held): this tree's registry plus a
    passing chip row of `arch` whose sha256 is that file's identity (gguf.resolve.identity), so the row proves this file
    and no other file of the architecture. A test that inspects a tiny MoE fixture as runnable calls this first; without
    it inspect, run and serve refuse the file in the unsupported-architecture line (tests/cpu/test_gguf_moe_hold.py)."""
    from mashbox import catalog
    from mashbox.gguf.resolve import identity
    reg = json.loads(json.dumps(catalog.registry()))
    reg["gguf"] = list(reg.get("gguf") or []) + [dict(id=f"test-{arch}-moe", arch=arch, moe=True, sha256=identity(path)[0],
                                                      chips={"v6e": {"status": "passing", "accelerator": "v6e-1"}})]
    monkeypatch.setattr(catalog, "registry", lambda path=None, reg=reg: reg)


def run(path, toks, split=None):
    f, A, c = open_model(path, act=jnp.float32)
    W, _ = A.load(f, c, use_kernels=False)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = A.new_state(c, 1, 32)
    i32 = jnp.int32
    z = jnp.zeros((1,), i32)
    t = jnp.asarray([toks], i32)
    T = len(toks)
    if split is None:
        return np.asarray(fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)["lg"][0]), c
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0), c


def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


@pytest.mark.parametrize("name,want", [
    ("qwen3moe-f32", dict(n_expert=8, n_used=2, expert_ff=32, norm_w=True, n_shared=0)),
    ("qwen2moe-f32", dict(n_expert=8, n_used=3, expert_ff=32, norm_w=False, n_shared=1, shared_ff=48, shared_gate=True)),
    ("llama-moe-f32", dict(n_expert=8, n_used=2, expert_ff=32, norm_w=True, n_shared=0)),
])
def test_moe_fixture_is_detected_from_metadata(name, want, tmp_path):
    p, _, _ = fixture(name, tmp_path)
    _, _, c = open_model(p, act=jnp.float32)
    assert c.moe and {k: getattr(c.spec, k) for k in want} == want
    assert all(c.spec.moe_layers)


@pytest.mark.parametrize("name", MOE)
@pytest.mark.parametrize("split", [None, 8])
def test_moe_logits_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = fixture(name, tmp_path)
    ours, _ = run(p, meta["tokens"], split)
    e = nmse(ours, ref)
    assert e <= NMSE_MAX, f"{name} split={split}: NMSE {e:.3e} > {NMSE_MAX}"
    assert np.array_equal(ours.argmax(-1), ref.argmax(-1))


def test_moe_name_does_not_matter(tmp_path, monkeypatch):
    """the qwen3moe fixture under a dense model's name and file name: identical spec and identical logits"""
    p, meta, ref = fixture("qwen3moe-f32", tmp_path)
    a, ca = run(p, meta["tokens"])
    q = tmp_path / "Dense-Chat-7B-Instruct.gguf"
    os.link(p, q)
    from mashbox.gguf.reader import File
    get = File.get
    monkeypatch.setattr(File, "get", lambda self, k, d=None: "Dense Chat 7B Instruct" if k in (
        "general.name", "general.basename") else get(self, k, d))
    b, cb = run(str(q), meta["tokens"])
    assert ca.spec == cb.spec and np.array_equal(a, b)


def test_dense_file_with_moe_in_its_name_stays_dense(tmp_path):
    p = FXT.build("qwen3-f32", str(tmp_path))
    q = tmp_path / "Tiny-MoE-A3B-Experts.gguf"
    os.link(p, q)
    _, _, c = open_model(str(q), act=jnp.float32)
    assert not getattr(c, "moe", False) and getattr(c, "spec", None) is None       # a guard: main's dense Config passes too


@pytest.mark.parametrize("arch,piece", [("olmoe", "q/k RMSNorm over all heads"),
                                        ("deepseek2", "multi-head latent attention"),
                                        ("gpt-oss", "attention sinks")])
def test_moe_archs_not_ported_name_the_missing_piece(arch, piece):
    from mashbox.gguf import archs
    with pytest.raises(GGUFError, match=piece):
        archs.module(arch)
