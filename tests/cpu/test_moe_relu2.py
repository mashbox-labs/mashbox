"""Ungated relu2 routed experts (NemotronH): ops.expert_act "relu2" (relu(u)^2, no gate), the exact
routed path with exactly up / down, a kernel package that does not list relu2 in EXPERT_ACTS (the routed-expert kernel before
0.2.0) sends the layer to the exact path with the reason (never a crash, never a dropped term), one that lists it takes
the kernel, and moe.routed_fn / prepare_ungated give a model that routes by itself (NemotronH's moe.ROUTED contract:
x (N, D), ids (N, K), wts (N, K), up, down -> (N, D) float32) the engine's routed op.

Bound (fixed before the first run: the float64 reference derivation with the exact path's roundings): eps = the activation dtype's unit roundoff (2^-9 bfloat16, 2^-24 float32); u is rounded to the activation
dtype after a float32 accumulation over D, du = eps |u| + 1.01 D 2^-24 (|Wu|.|x|); the square turns du into at most
2 relu(u) du + du^2; h is rounded to the activation dtype and accumulated in float32 over F, and the exact path rounds
each pick's y to the activation dtype: B = (2 eps + 1.01 F 2^-24)(|Wd|.h64) + |Wd|.(2 relu(u) du + du^2) + 2 eps |y64|;
times the routing weight |w| B + 2^-24 |w| (|y64| + B), and the K picks added in float32: + K 2^-24 sum_k |w_k y64_k|.
The expert weights are drawn bfloat16-representable for both activation dtypes, since prepare_ungated stores BF16 blocks
(a float32 reference drawn from float32 weights is not the weights the engine holds; the bound is unchanged)."""
import sys
import types

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from mashbox import moe, ops
from mashbox.moe import record as RX   # the routed-expert report's names and shape check

NAME = "_stub_moe_relu2_kernels"


def _bf(a):
    return np.asarray(jnp.asarray(a, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32), np.float64)


def _case(E=8, D=64, F=48, N=12, K=3, dtype=jnp.bfloat16, seed=0):
    r = np.random.default_rng(seed)
    q = _bf if dtype == jnp.bfloat16 else (lambda a: np.asarray(a, np.float32).astype(np.float64))
    x = q(r.standard_normal((N, D)))
    ids = np.stack([r.choice(E, K, replace=False) for _ in range(N)]).astype(np.int32)
    wts = r.uniform(0.05, 1.0, (N, K)).astype(np.float32)
    up = _bf(r.standard_normal((E, D, F)) * D ** -0.5)            # (E, in, out): NemotronH's checkpoint layout
    down = _bf(r.standard_normal((E, F, D)) * F ** -0.5)          # bf16-representable: prepare_ungated stores BF16
    return x, ids, wts, up, down


def ref64(x, ids, wts, up, down, eps):
    N, K = ids.shape
    D, F = up.shape[1], up.shape[2]
    y, bnd, s = np.zeros((N, D)), np.zeros((N, D)), np.zeros((N, D))
    for n in range(N):
        for k in range(K):
            e, w = ids[n, k], float(wts[n, k])
            u = x[n] @ up[e]
            r = np.maximum(u, 0.0)
            h = r * r
            yk = h @ down[e]
            du = eps * np.abs(u) + 1.01 * D * 2 ** -24 * (np.abs(x[n]) @ np.abs(up[e]))
            B = (2 * eps + 1.01 * F * 2 ** -24) * (h @ np.abs(down[e])) + (2 * r * du + du * du) @ np.abs(down[e]) \
                + 2 * eps * np.abs(yk)
            y[n] += w * yk
            bnd[n] += abs(w) * B + 2 ** -24 * abs(w) * (np.abs(yk) + B)
            s[n] += abs(w) * np.abs(yk)
    return y, bnd + K * 2 ** -24 * s


def _handles(up, down):
    return moe.prepare_ungated(jnp.asarray(up, jnp.float32), jnp.asarray(down, jnp.float32))


def _f32_handles(up, down):
    """(E, in, out) arrays -> F32 expert stacks (E, out, in) through ops.prepare, kernel path on (the stub's kq)"""
    out = []
    for a in (up, down):
        t = np.ascontiguousarray(np.swapaxes(np.asarray(a, np.float32), 1, 2))
        E, rows, cols = t.shape
        out.append(ops.prepare(t.view(np.uint8).reshape(E, rows, cols * 4), 0, (E, rows, cols), experts=True))
    return tuple(out)


def _stub(monkeypatch, acts):
    """a kernel package (tests/cpu/test_moe_fast_path.py's stub form) whose expert_ffn covers F32 experts with the exact
    math; acts None = no EXPERT_ACTS (the surface of the routed-expert kernel's 0.1.4 binding)"""
    stub = types.ModuleType(NAME)
    stub.TYPES, stub.EXPERT_TYPES, stub.calls = (0,), (0,), 0
    stub.EXPERT_KERNEL = {"id": "stub.expert-ffn", "version": "0.0.0"}
    if acts is not None:
        stub.EXPERT_ACTS = tuple(acts)
    ref = ops._expert_ref
    stub.repack = lambda t, raw, shape: {"w": raw, "meta": jnp.zeros((t, shape[-2], shape[-1], 0), jnp.int8)}
    stub.dequant = lambda p: ops.reference.dequant(p["w"], p["meta"].shape[-4], p["meta"].shape[-2])
    stub.matmul = lambda x, p, layer=None: ops.qmatmul(x, ops.Raw(p["w"], p["meta"].shape[-4],
                                                                  (p["meta"].shape[-3], p["meta"].shape[-2])))

    def expert_ffn(xs, es, ws, W, act, E, layer=None):
        stub.calls += 1
        hd = {k: (ops.Packed(v, v["meta"].shape[-4], (E,) + tuple(v["meta"].shape[-3:-1])) if isinstance(v, dict) else v)
              for k, v in W.items()}
        return ref(xs, es, ws, hd, act, E)
    stub.expert_ffn = expert_ffn
    monkeypatch.setitem(sys.modules, NAME, stub)
    return stub


def test_relu2_act_is_relu_squared_with_no_gate():
    u = jnp.asarray([-2.0, -0.0, 0.0, 0.5, 3.0], jnp.float32)
    assert np.array_equal(np.asarray(ops.expert_act(None, u, "relu2")), np.asarray([0.0, 0.0, 0.0, 0.25, 9.0], np.float32))
    assert ops.UNGATED_ACTS == ("relu2",)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "f32"])
def test_routed_fn_relu2_exact_path_matches_float64_within_bound(dtype):
    ops.load_kernels(module="off")
    x, ids, wts, up, down = _case(dtype=dtype)
    fn = moe.routed_fn("relu2")
    assert fn.keys == ("up", "down")
    y = np.asarray(fn(jnp.asarray(x, dtype), jnp.asarray(ids), jnp.asarray(wts), *_handles(up, down)), np.float64)
    y64, bnd = ref64(x, ids, wts, up, down, 2 ** -9 if dtype == jnp.bfloat16 else 2 ** -24)
    bad = np.abs(y - y64) > bnd
    print(f"relu2 exact {np.dtype(dtype).name}: max err/bound {np.max(np.abs(y - y64) / (bnd + 1e-300)):.3f}")
    assert not bad.any(), int(bad.sum())
    rep = ops.expert_report()
    assert rep["sites"] == {"kernel": 0, "exact": 1} and rep[RX.WHY] == ["no kernel package"], rep
    assert RX.problems(rep) == [], rep


def test_relu2_with_a_gate_is_refused_on_the_exact_path():
    ops.load_kernels(module="off")
    x, ids, wts, up, down = _case(N=2)
    u, d = _handles(up, down)
    with pytest.raises(ValueError, match="ungated"):
        ops.expert_ffn(jnp.zeros((6, 64), jnp.bfloat16), jnp.zeros((6,), jnp.int32), jnp.ones((6,), jnp.float32),
                       dict(gate=u, up=u, down=d), act="relu2", n_expert=8)
    with pytest.raises(TypeError, match="up, down"):
        moe.routed_fn("relu2")(jnp.asarray(x, jnp.bfloat16), jnp.asarray(ids), jnp.asarray(wts), u, u, d)


def test_a_package_without_relu2_sends_it_to_the_exact_path_with_the_reason(monkeypatch):
    stub = _stub(monkeypatch, acts=None)
    try:
        ops.load_kernels(module=NAME)
        assert ops.expert_kernel_acts() == ("silu", "gelu", "swiglu_oai")
        assert ops.expert_plan([0], act="relu2") == ("exact", "act 'relu2' not in the kernel package's EXPERT_ACTS")
        assert ops.expert_plan([0], act="silu") == ("kernel", None)
        x, ids, wts, up, down = _case(seed=1)
        y = np.asarray(moe.routed_fn("relu2")(jnp.asarray(x, jnp.bfloat16), jnp.asarray(ids), jnp.asarray(wts),
                                              *_f32_handles(up, down)))
        rep = ops.expert_report()
        assert stub.calls == 0 and rep["path"] == "exact", (stub.calls, rep)
        assert rep[RX.WHY] == ["act 'relu2' not in the kernel package's EXPERT_ACTS"], rep
        assert RX.problems(rep) == [], rep
        ops.load_kernels(module="off")
        y0 = np.asarray(moe.routed_fn("relu2")(jnp.asarray(x, jnp.bfloat16), jnp.asarray(ids), jnp.asarray(wts),
                                               *_handles(up, down)))
        assert np.array_equal(y, y0)
    finally:
        ops.load_kernels(module="off")


def test_a_package_with_relu2_takes_the_kernel(monkeypatch):
    stub = _stub(monkeypatch, acts=("silu", "gelu", "swiglu_oai", "relu2"))
    try:
        ops.load_kernels(module=NAME)
        assert ops.expert_plan([0], act="relu2") == ("kernel", None)
        x, ids, wts, up, down = _case(seed=2)
        y = np.asarray(moe.routed_fn("relu2")(jnp.asarray(x, jnp.bfloat16), jnp.asarray(ids), jnp.asarray(wts),
                                              *_f32_handles(up, down)), np.float64)
        rep = ops.expert_report()
        assert stub.calls >= 1 and rep["path"] == "kernel" and rep["sites"]["exact"] == 0, (stub.calls, rep)
        y64, bnd = ref64(x, ids, wts, up, down, 2 ** -9)
        assert not (np.abs(y - y64) > bnd).any()
    finally:
        ops.load_kernels(module="off")
