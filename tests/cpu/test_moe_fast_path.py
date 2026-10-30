"""The MoE fast path is required: when the kernel package's expert_ffn covers a file's expert weights, every routed
call site takes the kernel; the exact routed path is the tested failure fallback only (F-moe-routed). Each tiny MoE
GGUF fixture loads with kernels on through a stub package whose EXPERT_TYPES cover the fixtures' F32 experts and whose
expert_ffn computes the experts with the exact math (so the logits still match llama.cpp's); the engine's exact
routed path is replaced by one that fails. Prefill and decode must reach the kernel, ops.expert_report() must say path
"kernel" with the package's EXPERT_KERNEL and 0 exact sites, and inspect must name the same path. Controls: kernels
off reports "exact" with its reason, and a package without the experts' type reports "exact" naming the type."""
import sys
import types

import numpy as np
import pytest
import gguf  # noqa: F401  (a hard dependency: never skipped)
import jax
import jax.numpy as jnp

from mashbox import ops
from mashbox.moe import record as RX   # the routed-expert report's names and shape check
from mashbox.gguf import inspect, open_model
from mashbox.runtime import front as FR
from mashbox.runtime import openapi as O
from mashbox.runtime import server as SV

pytestmark = pytest.mark.standards          # the standards CPU tier runs this file in its common process

sys.path.insert(0, __import__("os").path.dirname(__file__))
import test_gguf_moe as M  # noqa: E402  (its fixtures, reference logits and bound)

NAME = "_stub_moe_fastpath_kernels"
KERNEL = {"id": "stub.expert-ffn", "version": "0.0.0"}


def _stub(monkeypatch, expert_types=(0,)):
    stub = types.ModuleType(NAME)
    stub.TYPES, stub.EXPERT_TYPES, stub.EXPERT_KERNEL, stub.calls = (0,), tuple(expert_types), KERNEL, 0
    ref = ops._expert_ref

    def repack(t, raw, shape):
        return {"w": raw, "meta": jnp.zeros((t, shape[-2], shape[-1], 0), jnp.int8)}
    stub.repack = repack
    stub.dequant = lambda p: ops.reference.dequant(p["w"], p["meta"].shape[-4], p["meta"].shape[-2])
    stub.matmul = lambda x, p, layer=None: ops.qmatmul(x, ops.Raw(p["w"] if layer is None else p["w"][layer],
                                                                  p["meta"].shape[-4],
                                                                  (p["meta"].shape[-3], p["meta"].shape[-2])))

    def expert_ffn(xs, es, ws, W, act, E, layer=None):
        stub.calls += 1
        hd = {k: (ops.Packed(v, v["meta"].shape[-4], (E,) + tuple(v["meta"].shape[-3:-1])) if isinstance(v, dict) else v)
              for k, v in W.items()}
        return ref(xs, es, ws, hd, act, E, **({} if layer is None else {"layer": layer}))
    stub.expert_ffn = expert_ffn

    def exact_is_forbidden(*a, **k):
        raise AssertionError("the exact routed path ran although the kernel package covers these experts")
    monkeypatch.setitem(sys.modules, NAME, stub)
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", NAME)            # inspect() loads the package by itself
    monkeypatch.setattr(ops, "_expert_ref", exact_is_forbidden)
    return stub


def _run_kernels(path, toks, split):
    f, A, c = open_model(path, act=jnp.float32)
    W, _ = A.load(f, c, use_kernels=True)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = A.new_state(c, 1, 32)
    i32, z, t = jnp.int32, jnp.zeros((1,), jnp.int32), jnp.asarray([toks], jnp.int32)
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(split, len(toks)):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0)


def _health():
    """Server.health on this process, called as the worker calls it (runtime/worker.py: Server.health(self)): the load
    fields /health sends an API caller, read where the engine runs"""
    return SV.Server.health(types.SimpleNamespace(engine=types.SimpleNamespace(), waiting=0))


def _fits_health_schema(h):
    s = O.SCHEMAS["Health"]["properties"]["routed_experts"]
    assert set(h) == set(s["properties"]), h
    assert h["path"] in s["properties"]["path"]["enum"], h
    assert set(h["sites"]) == set(s["properties"]["sites"]["properties"]), h
    assert RX.problems(h) == [], h


@pytest.mark.parametrize("name", M.MOE)
def test_moe_with_the_kernel_never_runs_the_exact_path(name, tmp_path, monkeypatch):
    p, meta, ref = M.fixture(name, tmp_path)
    stub = _stub(monkeypatch)
    try:
        assert ops.load_kernels(module=NAME) == frozenset({0})
        assert ops.expert_report()["path"] is None
        got = _run_kernels(p, meta["tokens"], 8)               # one prefill of 8, then 8 one-token decode steps
        rep = ops.expert_report()
        assert rep["path"] == "kernel" and rep["sites"]["exact"] == 0, rep
        assert rep["sites"]["kernel"] >= 2, rep                  # prefill and decode programs both took the kernel
        assert stub.calls >= rep["sites"]["kernel"], stub.calls  # (+ ops' one compile check per call shape)
        assert rep["kernel"] == dict(KERNEL, package=NAME) and rep[RX.WHY] == [], rep
        assert RX.problems(rep) == [], rep
        assert M.nmse(got, ref) <= M.NMSE_MAX
        h = _health()["routed_experts"]
        assert h == rep, h                                       # /health names the same path, kernel and sites
        _fits_health_schema(h)
        M.listed(monkeypatch, meta["arch"], p)                  # inspect runs a MoE file only once a passing row names it
        ins = inspect(p)
        assert ins["routed_experts"] == dict(path="kernel", why=None, kernel=dict(KERNEL, package=NAME)), ins
    finally:
        ops.load_kernels(module="off")


@pytest.mark.parametrize("name", M.MOE)
def test_the_kernel_is_traced_once_per_shape_not_once_per_moe_layer(name, tmp_path, monkeypatch):
    """warm-up: one forward traces the package's expert_ffn once per argument shapes (plus at most
    the compile check, which shares that trace), however many MoE layers call it. Before, every MoE layer traced and
    lowered the kernel's Pallas body again: on the chip the 35B target's warm-up spent 77% of its compile workers'
    samples there (40 layers x every warm program). That file loads with its layers unrolled (its layers' block types
    differ), so this fixture's copy differs in one dense weight of layer 0 (attn_q F16, every expert F32) and must load
    unrolled too (MIXED_SCAN off: with it, a file whose only differing role is a dense one runs in one lax.scan); every
    MoE layer still counts as a kernel site."""
    fx = M.FXT.FIXTURES[name]
    per = 7 + (3 if fx["moe"].get("FS") else 0)         # big tensors per layer in write order: q k v o, 3 experts, shexp
    quant = ["F32"] + ["F16"] + ["F32"] * (per - 1) + ["F32"] * per * (fx["layers"] - 1)
    p = M.FXT.write_dense(str(tmp_path / f"{name}-unrolled.gguf"), fx["arch"], seed=fx["seed"], layers=fx["layers"],
                          quant=quant, moe=fx["moe"])
    stub = _stub(monkeypatch)
    try:
        ops.load_kernels(module=NAME)
        f, A, c = open_model(p, act=jnp.float32)
        monkeypatch.setattr(A, "MIXED_SCAN", False)
        W, info = A.load(f, c, use_kernels=True)
        assert info["layers_stacked"] is False and isinstance(W["g"], list), info   # unrolled, as the 35B target
        ck, cv = A.new_state(c, 1, 32)
        z = jnp.zeros((1,), jnp.int32)
        fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
        fwd.lower(W, ck, cv, jnp.zeros((1, 8), jnp.int32), z, jnp.asarray([8], jnp.int32), z, 0, all_logits=True)
        sites, traces = ops.expert_report()["sites"], stub.calls
        assert sites == {"kernel": c.L, "exact": 0}, sites               # every MoE layer reached the kernel
        assert traces <= 2, (traces, sites)                             # not one trace per MoE layer
    finally:
        ops.load_kernels(module="off")


def test_controls_name_the_exact_path_and_why(tmp_path, monkeypatch):
    p, meta, _ = M.fixture("qwen3moe-f32", tmp_path)
    ops.load_kernels(module="off")
    assert ops.expert_plan([0]) == ("exact", "no kernel package")
    _stub(monkeypatch, expert_types=(8,))                        # the package's experts take Q8_0 only, not F32
    try:
        ops.load_kernels(module=NAME)
        path, why = ops.expert_plan([0])
        assert path == "exact" and "F32" in why and "EXPERT_TYPES" in why, why
        M.listed(monkeypatch, meta["arch"], p)
        ins = inspect(p)["routed_experts"]
        assert ins["path"] == "exact" and "F32" in ins["why"] and "EXPERT_TYPES" in ins["why"], ins
    finally:
        ops.load_kernels(module="off")



def test_health_routed_experts_is_null_until_a_routed_op_runs_and_the_front_passes_it_on(tmp_path):
    """/health's routed_experts is null for a dense model and before any routed op was traced (load_kernels resets the
    count); on an MoE file it is the path the experts took. The front keeps it among the worker's load fields, so it is
    null (never missing) while no worker is ready."""
    ops.load_kernels(module="off")
    assert _health()["routed_experts"] is None
    p, meta, ref = M.fixture("qwen3moe-f32", tmp_path)
    got = _run_kernels(p, meta["tokens"], 8)                     # kernels off: the exact path, said so with its reason
    h = _health()["routed_experts"]
    assert h["path"] == "exact" and h["sites"]["kernel"] == 0 and h["sites"]["exact"] >= 2, h
    assert h[RX.WHY] == ["no kernel package"] and h["kernel"] is None, h
    _fits_health_schema(h)
    assert M.nmse(got, ref) <= M.NMSE_MAX
    assert "routed_experts" in FR.Front.HEALTH_FIELDS
    ops.load_kernels(module="off")
