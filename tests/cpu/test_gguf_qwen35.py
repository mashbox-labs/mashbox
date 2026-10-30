"""GGUF architecture qwen35: our logits vs llama.cpp's CPU logits on tiny random fixtures (no chip, no network, no secrets).

Fixtures (tests/fixtures/gguf/<name>.json says how each was made, with llama.cpp's commit and the file sha256):
  qwen35-f32       float32 weights, 8 layers (2 layer groups): the graph test. Bound: NMSE <= 1e-4, llama.cpp's own
                   cross-backend bar (tests/test-llama-archs.cpp). Both sides compute in float32 here.
  qwen35-mtp       float32, 4 layers + an MTP (nextn) layer that the main pass must skip. Same bound.
  qwen35moe-f32    architecture qwen35moe: float32, 8 layers, 8 routed experts (2 used) + a gated shared expert. Same bound.
  qwen35moe-pyquant qwen35moe with Q8_0 / Q4_0 / Q5_0 blocks (experts included); bars as for the block-type fixtures.
  qwen35-pyquant   Q8_0 / Q4_0 / Q5_1 / Q4_1 / Q5_0 / BF16 / MXFP4 blocks written by gguf-py.
  qwen35-kquant    Q4_K / Q5_K / Q6_K / Q3_K / Q2_K / IQ4_XS / IQ4_NL / Q8_0 blocks written by llama.cpp's llama-quantize.
                   For block types our weights are exact (test_gguf_dequant.py), but llama.cpp's CPU dot products
                   round the activations to 8-bit blocks, so the bound is a distribution bound, fixed before the
                   first run: mean KL(llama.cpp || ours) <= 1e-3 per position and the same top-1 token on >= 15 of 16.
                   MEASURED 2026-10-28 (CPU): pyquant KL 8.86e-4, 16/16 (pass); kquant KL 1.350e-3, 16/16, over
                   the bound by 35%, kept as a strict xfail. Cause, measured the same day: llama.cpp's own K-quant path
                   vs llama.cpp on an F32 copy holding the identical dequantized weights gives KL 1.350e-3 too, so the
                   whole gap is llama.cpp's activation rounding. The strict test is therefore the one below:
  <name>.f32ref    llama.cpp on that F32 copy (identical weight values on both sides): NMSE <= 1e-4 for the block-type
                   fixtures too, the same bar as the float graph.
Every fixture runs twice: one prefill of all 16 tokens, and a prefill of 8 then 8 one-token decode steps (the
recurrent state, conv tail and K/V cache paths).
"""
import json
import os

import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

from mashbox.gguf.reader import File  # noqa: E402
from mashbox.gguf.archs import qwen35 as A  # noqa: E402

NMSE_MAX = 1e-4
KL_MAX, TOP1_MIN = 1e-3, 15


def fixture(name, tmp):
    meta = json.load(open(os.path.join(FX, f"{name}.json")))
    if meta["made_by"] == "gguf-py":
        p = FXT.build(name, str(tmp))
    else:
        p = os.path.join(FX, f"{name}.gguf")
    assert FXT.sha256(p) == meta["gguf_sha256"], f"{name}: fixture bytes changed; rerun tools/gguf_fixtures/fixtures.py make"
    ref = np.fromfile(os.path.join(FX, f"{name}.logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    return p, meta, ref


def run(path, toks, act=jnp.float32, split=None, use_kernels=False):
    """our logits for every position: one prefill, or a prefill of `split` tokens then one-token decode steps"""
    f = File(path)
    c = A.Config(f, act=act)
    W, info = A.load(f, c, use_kernels=use_kernels)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = A.new_state(c, 1, 32)
    i32 = jnp.int32
    z = jnp.zeros((1,), i32)
    T = len(toks)
    t = jnp.asarray([toks], i32)
    if split is None:
        r = fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)
        return np.asarray(r["lg"][0]), c, info
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out = [np.asarray(r["lg"][0])]
    ck, cv = r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0), c, info


def nmse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def kl_top1(ref, ours):
    p = np.exp(ref - ref.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    lp = np.log(np.maximum(p, 1e-300))
    kl = np.sum(p * (lp - lq), -1)
    return float(kl.mean()), int(np.sum(ref.argmax(-1) == ours.argmax(-1)))


@pytest.mark.parametrize("name", ["qwen35-f32", "qwen35-mtp", "qwen35moe-f32"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_float_logits_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = fixture(name, tmp_path)
    ours, c, info = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"{name} {split}: NMSE {e:.3e}")
    assert e <= NMSE_MAX, e


KQ_XFAIL = pytest.mark.xfail(strict=True, reason="MEASURED KL 1.350e-3 > 1e-3 bound set before the run (top-1 16/16); "
                              "llama.cpp vs itself on the same weights in F32 gives the same 1.350e-3 (activation rounding)")


MOEQ_XFAIL = pytest.mark.xfail(strict=True, reason="MEASURED KL 4.923e-3 > 1e-3, top-1 12/16 < 15 (bound set before the "
                                "run); llama.cpp vs itself on the same weights in F32 gives the same KL 4.923e-3 and 12/16 "
                                "(its 8-bit activation rounding, which also moves expert picks); ours vs that F32 run: "
                                "NMSE 4.6e-12")


@pytest.mark.parametrize("name", ["qwen35-pyquant", pytest.param("qwen35-kquant", marks=KQ_XFAIL),
                                  pytest.param("qwen35moe-pyquant", marks=MOEQ_XFAIL)])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_type_logits_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = fixture(name, tmp_path)
    ours, c, info = run(p, meta["tokens"], split=split)
    kl, top1 = kl_top1(ref, ours)
    print(f"{name} {split}: KL {kl:.3e} top1 {top1}/{len(ref)} NMSE {nmse(ours, ref):.3e} types {info['reference_types']}")
    assert kl <= KL_MAX and top1 >= TOP1_MIN, (kl, top1)


@pytest.mark.parametrize("name", ["qwen35-pyquant", "qwen35-kquant", "qwen35moe-pyquant"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_block_types_match_llama_cpp_on_identical_weights(name, split, tmp_path):
    """ours on the block-type file vs llama.cpp on an F32 file with exactly the dequantized weights: NMSE <= 1e-4"""
    p, meta, _ = fixture(name, tmp_path)
    ref = np.fromfile(os.path.join(FX, f"{name}.f32ref.logits.f32"), np.float32).reshape(len(meta["tokens"]), meta["n_vocab"])
    ours, c, info = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"{name} {split} vs f32ref: NMSE {e:.3e}")
    assert e <= NMSE_MAX, e


def test_fixture_types_cover_the_unsloth_mixes(tmp_path):
    """the K-quant fixture holds the block types of Unsloth's UD-Q4_K_XL / UD-Q4_K_M files except IQ3_S"""
    p, meta, ref = fixture("qwen35-kquant", tmp_path)
    have = set(File(p).census()[0])
    assert {"Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "Q8_0"} <= have, have


def test_uniform_layers_scan(tmp_path):
    p, meta, _ = fixture("qwen35-f32", tmp_path)
    f = File(p)
    W, info = A.load(f, A.Config(f, act=jnp.float32), use_kernels=False)
    assert info["layers_stacked"] is True and not isinstance(W["g"], list)


def test_stacked_load_never_holds_the_weights_twice(tmp_path, monkeypatch):
    """Stacking the layer groups for the scan never holds the weights twice: device bytes alive after every step of the
    load (each handle made, each stack) stay under 1.25x the loaded weights (keeping every group until the end and then
    stacking peaks at 2x). Bar fixed before the first run."""
    from mashbox import ops
    p, meta, _ = fixture("qwen35-f32", tmp_path)
    f = File(p)
    live = lambda: sum(int(x.nbytes) for x in jax.live_arrays() if not x.is_deleted())
    base, peak = live(), [0]

    def watch(fn):
        def w(*a, **k):
            y = fn(*a, **k)
            peak[0] = max(peak[0], live() - base)
            return y
        return w
    for mod, name in ((jnp, "stack"), (ops, "prepare"), (ops.reference, "dequant")):
        monkeypatch.setattr(mod, name, watch(getattr(mod, name)))
    W, info = A.load(f, A.Config(f, act=jnp.float32), use_kernels=False)
    monkeypatch.undo()
    held = sum(int(x.nbytes) for x in jax.tree_util.tree_leaves(W))
    print(f"peak during load {peak[0]:,} B, loaded {held:,} B")
    assert info["layers_stacked"] is True and NG_OK(W) and 0 < peak[0] <= 1.25 * held, (peak[0], held)


def NG_OK(W):
    return all(x.shape[0] == 2 for x in jax.tree_util.tree_leaves(W["g"]))


@pytest.mark.parametrize("host", [False, True], ids=["device", "host"])
def test_stacked_kernel_handles_repack_in_one_program(host, tmp_path, monkeypatch):
    """A kernel package's repack takes one tensor (rows, row_bytes), as the private binding does. Stacked layer groups are
    repacked through it mapped over the group axis: one Packed handle per role with the group axis in front, and logits
    equal to the exact path (stub kernel = the reference math; NMSE <= 1e-10, fixed before the first run). host: the
    repack runs on the host CPU device from the loader's host arrays (what a TPU run does), then the planes are sent."""
    import sys
    import types
    from mashbox import ops
    monkeypatch.setattr(ops, "HOST_REPACK", True if host else None)
    stub = types.ModuleType("_stub_gguf_kernels")
    stub.TYPES = (8,)

    def repack(t, raw, shape):
        assert raw.ndim == 2 and raw.shape[0] == shape[0], raw.shape
        return {"w": raw[::-1], "meta": jnp.zeros((t, shape[0], shape[1], 0), jnp.int8)}
    stub.repack = repack
    stub.dequant = lambda p: ops.reference.dequant(p["w"][..., ::-1, :], p["meta"].shape[-4], p["meta"].shape[-2])
    stub.matmul = lambda x, p: ops.qmatmul(x, ops.Raw(p["w"][::-1], p["meta"].shape[-4],
                                                       (p["meta"].shape[-3], p["meta"].shape[-2])))
    monkeypatch.setitem(sys.modules, "_stub_gguf_kernels", stub)
    q = FXT.write_qwen35(str(tmp_path / "q8.gguf"), seed=5, layers=8, quant=["Q8_0"])
    toks = list(range(3, 19))
    a, _, info0 = run(q, toks, split=8)
    try:
        assert ops.load_kernels(module="_stub_gguf_kernels") == frozenset({8})
        f = File(q)
        W, info = A.load(f, A.Config(f, act=jnp.float32), use_kernels=True)
        b, _, _ = run(q, toks, split=8, use_kernels=True)
    finally:
        ops.load_kernels(module="off")
    h = W["g"]["mlp"][0]["wg"]
    assert info["layers_stacked"] and isinstance(h, ops.Packed) and h.packed["w"].shape[0] == 2, h
    assert info["kernel_bytes"] > 0 and "Q8_0" not in info["reference_types"], info     # every Q8_0 tensor: the kernel
    assert isinstance(h.packed["w"], jax.Array) and h.packed["w"].devices() == {jax.devices()[0]}
    e = nmse(b, a)
    print(f"stacked stub-kernel path vs exact path: NMSE {e:.3e}")
    assert e <= 1e-10, e


def test_kernel_repack_host_and_plane_bytes(tmp_path, monkeypatch):
    """A kernel package with repack_host gets every host array there (never its jitted repack), stacked layer groups
    included (leading group axis); logits equal the exact path (stub kernel = the reference math; NMSE <= 1e-10, fixed
    before the first run). With plane_bytes, the memory estimate counts the planes of every multiplied weight of a kernel
    type and the file's blocks for everything else (the embedding table is gathered, never multiplied)."""
    import sys
    import types
    from mashbox import gguf as G
    from mashbox import ops
    stub = types.ModuleType("_stub_gguf_kernels_host")
    stub.TYPES = (8,)
    calls = {"host": 0, "jit": 0}

    def repack(t, raw, shape):
        calls["jit"] += 1
        return {"w": raw[::-1], "meta": jnp.zeros((t, shape[0], shape[1], 0), jnp.int8)}

    def repack_host(t, raw, shape):
        assert isinstance(raw, np.ndarray) and raw.shape[-2] == shape[0], (type(raw), raw.shape)
        calls["host"] += 1
        return {"w": np.ascontiguousarray(raw[..., ::-1, :]), "meta": np.zeros(raw.shape[:-2] + (t, shape[0], shape[1], 0), np.int8)}
    stub.repack, stub.repack_host = repack, repack_host
    stub.plane_bytes = lambda t, shape: 3 * shape[0] * shape[1]
    stub.dequant = lambda p: ops.reference.dequant(p["w"][..., ::-1, :], p["meta"].shape[-4], p["meta"].shape[-2])
    stub.matmul = lambda x, p: ops.qmatmul(x, ops.Raw(p["w"][::-1], p["meta"].shape[-4],
                                                       (p["meta"].shape[-3], p["meta"].shape[-2])))
    monkeypatch.setitem(sys.modules, "_stub_gguf_kernels_host", stub)
    q = FXT.write_qwen35(str(tmp_path / "q8.gguf"), seed=5, layers=8, quant=["Q8_0"])
    toks = list(range(3, 19))
    a, _, _ = run(q, toks, split=8)
    try:
        assert ops.load_kernels(module="_stub_gguf_kernels_host") == frozenset({8})
        f = File(q)
        c = A.Config(f, act=jnp.float32)
        W, info = A.load(f, c, use_kernels=True)
        b, _, _ = run(q, toks, split=8, use_kernels=True)
        loaded, mult = A.tensor_roles(f, c)
        w_k, st_k, _ = G.need_bytes(f, c, 64, 2, frozenset({8}))
        w_r, st_r, _ = G.need_bytes(f, c, 64, 2, frozenset())
    finally:
        ops.load_kernels(module="off")
    h = W["g"]["mlp"][0]["wg"]
    assert calls["host"] > 0 and calls["jit"] == 0, calls
    assert info["layers_stacked"] and isinstance(h, ops.Packed) and h.packed["w"].shape[0] == 2, h
    assert isinstance(h.packed["w"], jax.Array) and h.packed["w"].devices() == {jax.devices()[0]}
    want = sum(3 * f.tensors[n].shape[-2] * f.tensors[n].shape[-1] if (n in mult and f.tensors[n].type == 8)
               else f.tensors[n].nbytes for n in loaded)
    assert w_r == sum(f.tensors[n].nbytes for n in loaded) and w_k == want and w_k != w_r and st_k == st_r, (w_k, want, w_r)
    e = nmse(b, a)
    print(f"repack_host stub-kernel path vs exact path: NMSE {e:.3e}; estimate weights {w_k:,} (planes) vs {w_r:,} (file)")
    assert e <= 1e-10, e


def _kq_stub(name, fail, calls):
    """A kq stub package for Q8_0 (the reference math, as the stubs above) whose function `fail` raises: "repack",
    "repack_host" (the package has repack_host), "matmul" (the kernel raises when it is traced) or "matmul_run" (the
    kernel traces and compiles, then fails while it runs)."""
    import types
    from mashbox import ops
    stub = types.ModuleType(name)
    stub.TYPES = (8,)

    def boom(what):
        raise RuntimeError(f"forced failure of the kernel's {what}")

    def repack(t, raw, shape):
        calls["repack"] += 1
        if fail == "repack":
            boom("repack")
        return {"w": raw[::-1], "meta": jnp.zeros((t, shape[0], shape[1], 0), jnp.int8)}

    def repack_host(t, raw, shape):
        calls["repack"] += 1
        if fail == "repack_host":
            boom("repack_host")
        return {"w": np.ascontiguousarray(raw[..., ::-1, :]), "meta": np.zeros(raw.shape[:-2] + (t, shape[0], shape[1], 0), np.int8)}

    def matmul(x, p):
        calls["matmul"] += 1
        if fail == "matmul":
            boom("matmul")
        y = ops.qmatmul(x, ops.Raw(p["w"][::-1], p["meta"].shape[-4], (p["meta"].shape[-3], p["meta"].shape[-2])))
        if fail == "matmul_run":
            def run_fails(v):
                raise RuntimeError("forced failure of the kernel while it runs")
            y = jax.pure_callback(run_fails, jax.ShapeDtypeStruct(y.shape, y.dtype), y)
        return y
    stub.repack, stub.matmul = repack, matmul
    if fail == "repack_host":
        stub.repack_host = repack_host
    stub.dequant = lambda p: ops.reference.dequant(p["w"][..., ::-1, :], p["meta"].shape[-4], p["meta"].shape[-2])
    return stub


@pytest.mark.parametrize("fail", ["repack", "repack_host", "matmul", "matmul_run"])
def test_kq_failure_falls_back_to_the_exact_path(fail, tmp_path, monkeypatch, capsys):
    """The fallback rule for kq (every fast path needs a forced-failure test that returns the working path's output):
    force the kernel package's repack, or its matmul at trace or at run, to fail. The file still loads, every Q8_0 weight
    runs on the exact path, one warning per (type, shape) names the type and the error, and the logits equal the exact
    path's bit for bit, for a prefill and for one-token decode steps (stacked layer groups and the unstacked tensors)."""
    import sys
    from mashbox import ops
    calls = {"repack": 0, "matmul": 0}
    name = f"_stub_gguf_kernels_fail_{fail}"
    monkeypatch.setitem(sys.modules, name, _kq_stub(name, fail, calls))
    q = FXT.write_qwen35(str(tmp_path / "q8.gguf"), seed=5, layers=8, quant=["Q8_0"])
    toks = list(range(3, 19))
    a, _, info0 = run(q, toks, split=8)
    try:
        assert ops.load_kernels(module=name) == frozenset({8})
        capsys.readouterr()
        f = File(q)
        W, info = A.load(f, A.Config(f, act=jnp.float32), use_kernels=True)
        err = capsys.readouterr().err
        ran = dict(ops._KQ_RAN)
        b, _, info1 = run(q, toks, split=8, use_kernels=True)
    finally:
        ops.load_kernels(module="off")
    h = W["g"]["mlp"][0]["wg"]
    assert isinstance(h, ops.Raw), h
    assert ran and all(v is not True and "forced failure" in v for v in ran.values()), ran
    assert err.count("kq planes for") == len(ran), err                     # one warning per (type, shape)
    assert info["kernel_bytes"] == 0 and info1["kernel_bytes"] == 0, (info, info1)
    assert info["reference_types"] == info0["reference_types"] and "Q8_0" in info["reference_types"], info
    assert "kq planes for Q8_0" in err and "forced failure" in err and "exact path" in err, err
    assert calls["repack"] > 0 and (calls["matmul"] > 0) == fail.startswith("matmul"), calls
    assert b.shape == a.shape and np.array_equal(b.view(np.uint32), a.view(np.uint32)), float(np.abs(b - a).max())


def test_kq_preflight_runs_once_per_type_and_shape(tmp_path, monkeypatch):
    """A working kq package: the preflight runs the kernel once per (type, shape), never per weight, waits for each
    result (on TPU a kernel fails while it runs only when its result is waited for; the CPU here runs it at the call),
    and a kernel that passes keeps its planes (the stub's matmul is counted while the load runs, before any forward)."""
    import sys
    from mashbox import ops
    calls = {"repack": 0, "matmul": 0}
    waited, wait = [], jax.block_until_ready
    monkeypatch.setattr(jax, "block_until_ready", lambda v: (waited.append(tuple(getattr(v, "shape", ()))), wait(v))[1])
    monkeypatch.setitem(sys.modules, "_stub_gguf_kernels_ok", _kq_stub("_stub_gguf_kernels_ok", None, calls))
    q = FXT.write_qwen35(str(tmp_path / "q8.gguf"), seed=5, layers=8, quant=["Q8_0"])
    try:
        assert ops.load_kernels(module="_stub_gguf_kernels_ok") == frozenset({8})
        f = File(q)
        W, info = A.load(f, A.Config(f, act=jnp.float32), use_kernels=True)
        ran = dict(ops._KQ_RAN)
    finally:
        ops.load_kernels(module="off")
    assert isinstance(W["g"]["mlp"][0]["wg"], ops.Packed) and info["kernel_bytes"] > 0, info
    assert ran and all(v is True for v in ran.values()), ran
    assert calls["matmul"] == len(ran) and calls["repack"] >= len(ran), (calls, sorted(ran))
    assert all((1, k[1][-2]) in waited for k in ran), (waited, sorted(ran))      # each preflight result waited for
    assert ops._KQ_RAN == {}, "load_kernels starts the preflight over"


@pytest.mark.parametrize("mixed_scan", [False, True], ids=["unrolled", "mixed-scan"])
def test_mixed_layer_groups_unroll_and_match_the_scan(mixed_scan, tmp_path, monkeypatch):
    """8 layers (2 groups) with a Q8_0 / Q4_0 cycle that gives the two groups different block types -> the unrolled path
    (MIXED_SCAN False), or one lax.scan with ops.Mixed roles (MIXED_SCAN True). Its logits vs the scan path on an all-F32
    copy holding exactly gguf-py's dequantized weights: NMSE <= 1e-4 (the graph bar, fixed before the first run),
    prefill and decode."""
    monkeypatch.setattr(A, "MIXED_SCAN", mixed_scan)
    q = FXT.write_qwen35(str(tmp_path / "mixed.gguf"), seed=4, layers=8, quant=["Q8_0", "Q4_0"])
    FXT.FIXTURES["_mixed8"] = dict(arch="qwen35", seed=4, layers=8, make="gguf-py", quant=["Q8_0", "Q4_0"])
    try:
        fq = FXT.dequant_copy("_mixed8", q, str(tmp_path / "mixed-f32.gguf"))
    finally:
        del FXT.FIXTURES["_mixed8"]
    f = File(q)
    W, info = A.load(f, A.Config(f, act=jnp.float32), use_kernels=False)
    if mixed_scan:
        assert info["layers_stacked"] is True and not isinstance(W["g"], list) and info["mixed_roles"], info
    else:
        assert info["layers_stacked"] is False and isinstance(W["g"], list) and len(W["g"]) == 2
    f2 = File(fq)
    W2, info2 = A.load(f2, A.Config(f2, act=jnp.float32), use_kernels=False)
    assert info2["layers_stacked"] is True
    toks = list(range(3, 19))
    for split in (None, 8):
        a, _, _ = run(q, toks, split=split)
        b, _, _ = run(fq, toks, split=split)
        e = nmse(a, b)
        print(f"unrolled (block types) vs scan (F32 copy) split={split}: NMSE {e:.3e}")
        assert e <= NMSE_MAX, e


def test_unrolled_forward_memory_does_not_grow_with_layers(tmp_path, monkeypatch):
    """Files that mix block types per layer run the unrolled path when MIXED_SCAN is False. Its compiled scratch must hold
    the float32 weights of about one matmul at a time, not of every layer: going from 8 to 16 layers may add at most 10%
    of the extra layers' float32 weight bytes (all of them live at once would add 100%). Decode (T=1) and a 16-token
    prefill."""
    monkeypatch.setattr(A, "MIXED_SCAN", False)
    temp, f32b = {}, {}
    for L in (8, 16):
        p = FXT.write_qwen35(str(tmp_path / f"m{L}.gguf"), seed=4, layers=L, quant=["Q8_0", "Q4_0"])
        f = File(p)
        c = A.Config(f, act=jnp.bfloat16)
        W, info = A.load(f, c, use_kernels=False)
        assert info["layers_stacked"] is False
        f32b[L] = sum(4 * int(np.prod(h.shape)) for g in W["g"] for h in jax.tree_util.tree_leaves(
            g, is_leaf=lambda x: hasattr(x, "raw")) if hasattr(h, "raw"))
        fwd = jax.jit(A.make_forward(c), static_argnums=(7,))
        for T in (1, 16):
            ck, cv = A.new_state(c, 2, 64)
            args = (W, ck, cv, jnp.zeros((2, T), jnp.int32), jnp.zeros((2,), jnp.int32), jnp.full((2,), T, jnp.int32),
                    jnp.arange(2, dtype=jnp.int32), 0)
            temp[L, T] = int(fwd.lower(*args).compile().memory_analysis().temp_size_in_bytes)
    assert f32b[16] > f32b[8] > 0
    for T in (1, 16):
        grow = temp[16, T] - temp[8, T]
        print(f"T={T}: temp 8 layers {temp[8, T]:,} B, 16 layers {temp[16, T]:,} B; extra float32 weights "
              f"{f32b[16] - f32b[8]:,} B")
        assert grow <= 0.10 * (f32b[16] - f32b[8]), (T, temp)


def test_exact_path_rebuilds_big_weights_in_row_chunks(monkeypatch):
    """A weight over ops.EXACT_CHUNK float32 values runs the exact path a chunk of rows at a time (a lax.map), so its
    compiled scratch is about one chunk, not the whole float32 weight. Q6_K (the output head type of the 27B UD file),
    Q4_K and IQ4_XS weights, decode (2 rows) and prefill (16 rows). Values: chunked and whole-weight results both stay
    inside the textbook float32 dot-product error bound (gamma_n * sum |x w|, n = cols) of a float64 product of the same
    exact weights; they are not bit-equal (the CPU dot's blocking depends on the row count). Scratch: under a quarter of
    the whole-weight path's. (On a TPU the whole-weight path of the 27B head needed 57.36 GB of scratch.)"""
    from mashbox import ops
    from mashbox.ops import reference as R
    rng = np.random.default_rng(7)
    rows, cols = 1024, 512
    eps = np.finfo(np.float32).eps
    gamma = cols * eps / (1 - cols * eps)
    for t in (14, 12, 23):                                          # Q6_K, Q4_K, IQ4_XS
        raw = jnp.asarray(_random_blocks(rng, t, rows, cols))
        h = ops.Raw(raw, t, (rows, cols))
        w64 = np.asarray(R.dequant(raw, t, cols)).astype(np.float64)
        for n in (2, 16):
            x = jnp.asarray(rng.standard_normal((n, cols)).astype(np.float32))
            x64 = np.asarray(x).astype(np.float64)
            y64, bound = x64 @ w64.T, gamma * (np.abs(x64) @ np.abs(w64).T)
            monkeypatch.setattr(ops, "EXACT_CHUNK", 1 << 30)
            f_all = jax.jit(lambda x, h: ops.qmatmul(x, h))         # a new function: jit's trace cache does not
                                                                    # key on EXACT_CHUNK
            y_all = np.asarray(f_all(x, h)).astype(np.float64)
            m_all = f_all.lower(x, h).compile().memory_analysis().temp_size_in_bytes
            monkeypatch.setattr(ops, "EXACT_CHUNK", 1 << 16)
            assert ops.exact_chunk_rows(rows, cols) == 128
            prims = {e.primitive.name for e in jax.make_jaxpr(lambda x, h: ops.qmatmul(x, h))(x, h).jaxpr.eqns}
            assert prims & {"scan", "while"}, ("the chunked path runs no loop", prims)
            f_ch = jax.jit(lambda x, h: ops.qmatmul(x, h))
            y_ch = np.asarray(f_ch(x, h)).astype(np.float64)
            m_ch = f_ch.lower(x, h).compile().memory_analysis().temp_size_in_bytes
            print(f"{R.TYPES[t][0]} rows {n}: scratch whole {m_all:,} B, chunked {m_ch:,} B; max abs vs float64: whole "
                  f"{float(np.max(np.abs(y_all - y64))):.3e}, chunked {float(np.max(np.abs(y_ch - y64))):.3e}, "
                  f"bound {float(np.min(bound)):.3e}..{float(np.max(bound)):.3e}")
            assert np.all(np.abs(y_all - y64) <= bound), R.TYPES[t][0]
            assert np.all(np.abs(y_ch - y64) <= bound), R.TYPES[t][0]
            assert m_ch <= 0.25 * m_all, (R.TYPES[t][0], n, m_all, m_ch)


def _random_blocks(rng, t, rows, cols):
    """random but valid blocks of type t: random bytes with every float16 scale field set to a small finite value"""
    from mashbox.ops import reference as R
    _, bs, ts = R.TYPES[t]
    b = rng.integers(0, 256, size=(rows, cols // bs, ts), dtype=np.uint8)
    h = np.array([0.01], np.float16).view(np.uint8)
    pos = {14: [208], 12: [0, 2], 23: [0], 8: [0]}[t]              # byte offsets of the float16 fields
    for p in pos:
        b[..., p], b[..., p + 1] = h[0], h[1]
    return b.reshape(rows, -1)


@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_exact_path_chunks_match_llama_cpp(split, tmp_path, monkeypatch):
    """The K-quant fixture with a chunk small enough that every multiplied weight is split: the same bar as the
    whole-weight path, NMSE <= 1e-4 vs llama.cpp on the F32 copy with identical weights (prefill and decode)."""
    from mashbox import ops
    p, meta, _ = fixture("qwen35-kquant", tmp_path)
    ref = np.fromfile(os.path.join(FX, "qwen35-kquant.f32ref.logits.f32"), np.float32).reshape(len(meta["tokens"]),
                                                                                                meta["n_vocab"])
    monkeypatch.setattr(ops, "EXACT_CHUNK", 1 << 12)
    ours, c, info = run(p, meta["tokens"], split=split)
    monkeypatch.setattr(ops, "EXACT_CHUNK", 1 << 30)
    whole, _, _ = run(p, meta["tokens"], split=split)
    e = nmse(ours, ref)
    print(f"chunked {split} vs f32ref: NMSE {e:.3e} (whole-weight path {nmse(whole, ref):.3e}); chunked vs whole max abs "
          f"{float(np.max(np.abs(ours - whole))):.3e}")
    assert e <= NMSE_MAX, e


def test_mtp_layer_is_found(tmp_path):
    p, meta, _ = fixture("qwen35-mtp", tmp_path)
    f = File(p)
    c = A.Config(f)
    assert (c.L_all, c.nextn, c.L) == (5, 1, 4)
    roles = A.mtp_tensors(f, c)
    assert {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
            "attn_q.weight", "ffn_down.weight"} <= set(roles), sorted(roles)


def test_bf16_serving_path_close_to_float(tmp_path):
    """the bfloat16 activations the server uses vs float32 on the same file (bound fixed before the run)"""
    p, meta, ref = fixture("qwen35-f32", tmp_path)
    ours16, _, _ = run(p, meta["tokens"], act=jnp.bfloat16)
    kl, top1 = kl_top1(ref, ours16)
    print(f"bf16 vs llama.cpp f32: KL {kl:.3e} top1 {top1}/16")
    assert kl <= 1e-2 and top1 >= 14, (kl, top1)


def test_exact_handles_keep_each_tensor_as_one_byte_run_with_the_same_bits():
    """prepare keeps a tensor's blocks as one 1-D byte run (one device layout); dequant, take_rows and qmatmul (whole and
    chunked) equal the (rows, row_bytes) handle bit for bit"""
    import jax
    from mashbox import ops
    rng = np.random.default_rng(11)
    rows, cols = 64, 512
    for t in (14, 12, 23):                                          # Q6_K, Q4_K, IQ4_XS
        raw = _random_blocks(rng, t, rows, cols)
        h = ops.prepare(raw, t, (rows, cols), use_kernel=False)
        h2 = ops.Raw(jnp.asarray(raw), t, (rows, cols))
        assert h.raw.shape == (raw.size,) and (np.asarray(h.raw) == raw.reshape(-1)).all()
        ids = jnp.asarray([3, 0, 63, 3], jnp.int32)
        x = jnp.asarray(rng.standard_normal((2, cols)).astype(np.float32))
        pairs = [(ops.dequant(h), ops.dequant(h2)), (ops.take_rows(h, ids), ops.take_rows(h2, ids)),
                 (jax.jit(ops.qmatmul)(x, h), jax.jit(ops.qmatmul)(x, h2))]
        old = ops.EXACT_CHUNK
        try:
            ops.EXACT_CHUNK = 16 * cols                             # 16-row chunks: the lax.map path
            pairs.append((jax.jit(lambda x, h: ops.qmatmul(x, h))(x, h), jax.jit(lambda x, h: ops.qmatmul(x, h))(x, h2)))
        finally:
            ops.EXACT_CHUNK = old
        for a, b in pairs:
            a, b = np.asarray(a), np.asarray(b)
            assert a.shape == b.shape and (a.view(np.uint32) == b.view(np.uint32)).all()


def test_take_rows_of_a_byte_run_gathers_whole_rows():
    """the embedding lookup of a byte run is one gather whose slice is one whole row of blocks, not one index per byte
    (Qwen3.6-35B-A3B's Q8_0 table, 2176 bytes a row, on one v6e: 8 ids took 983 us one index per byte, 123 us as rows)"""
    import re
    import jax
    from mashbox import ops
    from mashbox.ops import reference as R
    rng = np.random.default_rng(12)
    rows, cols = 64, 512
    for t in (8, 14, 12, 23):                                       # Q8_0, Q6_K, Q4_K, IQ4_XS
        h = ops.prepare(_random_blocks(rng, t, rows, cols), t, (rows, cols), use_kernel=False)
        rb = cols // R.TYPES[t][1] * R.TYPES[t][2]
        def sizes(txt):
            return sorted(re.findall(r"stablehlo\.gather.*?slice_sizes = array<i64: ([0-9, ]+)>", txt))
        for shape in ((1,), (8,), (3, 5)):
            got = sizes(jax.jit(ops.take_rows).lower(h, jnp.zeros(shape, jnp.int32)).as_text())
            own = sizes(jax.jit(lambda b: R.dequant(b, t, cols)).lower(jnp.zeros(shape + (rb,), jnp.uint8)).as_text())
            assert got == sorted(own + [str(rb)]), (t, shape, got, own)       # the dequant's own lookups, plus one row gather


def test_take_rows_of_a_byte_run_equals_the_row_handle_for_every_id_form():
    """the byte run's lookup equals the (rows, row_bytes) handle's bit for bit for 1-D and 2-D ids, the first and last
    rows, repeated ids and negative ids (counted from the end, as numpy indexing)"""
    import jax
    from mashbox import ops
    rng = np.random.default_rng(13)
    rows, cols = 64, 512
    for t in (8, 14, 12, 23):
        raw = _random_blocks(rng, t, rows, cols)
        h = ops.prepare(raw, t, (rows, cols), use_kernel=False)
        h2 = ops.Raw(jnp.asarray(raw), t, (rows, cols))
        for ids in ([0], [rows - 1, 0, 5, 5], [-1, -rows, 7], [[3, -2, 0], [63, 1, 1]]):
            ids = jnp.asarray(ids, jnp.int32)
            a, b = np.asarray(jax.jit(ops.take_rows)(h, ids)), np.asarray(ops.take_rows(h2, ids))
            assert a.shape == b.shape == ids.shape + (cols,) and (a.view(np.uint32) == b.view(np.uint32)).all(), (t, ids)


def test_qwen35moe_file_without_feed_forward_length(tmp_path):
    """llama.cpp's converter writes no qwen35moe.feed_forward_length (unsloth/Qwen3.6-35B-A3B-GGUF has only the expert
    and shared-expert lengths). The MoE fixture with that key hidden opens and still matches llama.cpp's float logits."""
    p, meta, ref = fixture("qwen35moe-f32", tmp_path)
    f = File(p)
    get = f.get
    f.get = lambda key, default=None: default if key == "qwen35moe.feed_forward_length" else get(key, default)
    assert f.get("qwen35moe.feed_forward_length") is None and f.get("qwen35moe.expert_feed_forward_length")
    c = A.Config(f, act=jnp.float32)
    assert c.moe and c.F == 0
    W, _ = A.load(f, c, use_kernels=False)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    ck, cv = A.new_state(c, 1, 32)
    z = jnp.zeros((1,), jnp.int32)
    toks = meta["tokens"]
    r = fwd(W, ck, cv, jnp.asarray([toks], jnp.int32), z, jnp.asarray([len(toks)], jnp.int32), z, 0, all_logits=True)
    assert nmse(np.asarray(r["lg"][0]), ref) < NMSE_MAX


def test_kernel_planes_count_every_expert_of_a_stacked_weight(tmp_path, monkeypatch):
    """a kernel type's planes for an (experts, rows, cols) weight are experts x one matrix's planes; the memory estimate
    counted one expert per stacked weight"""
    import sys
    import types
    from mashbox import gguf as G, ops
    stub = types.ModuleType("_stub_gguf_kernels_planes")
    stub.TYPES = (8,)
    stub.plane_bytes = lambda t, shape: 3 * shape[0] * shape[1]
    monkeypatch.setitem(sys.modules, "_stub_gguf_kernels_planes", stub)
    p = FXT.build("qwen35moe-pyquant", str(tmp_path))
    f, M, c = G.open_model(p)
    loaded, mult = M.tensor_roles(f, c)
    try:
        assert ops.load_kernels(module="_stub_gguf_kernels_planes") == frozenset({8})
        one = ops.device_nbytes(8, (4, 6, 64), 4 * 6 * 2 * 34)
        w, _, _ = G.need_bytes(f, c, 64, 2, frozenset({8}))
    finally:
        ops.load_kernels(module="off")
    assert one == 4 * 3 * 6 * 64, one
    stacked = [n for n in mult if len(f.tensors[n].shape) == 3 and f.tensors[n].type == 8]
    assert stacked, "the fixture has Q8_0 expert weights"
    want = sum(3 * int(np.prod(f.tensors[n].shape)) if (n in mult and f.tensors[n].type == 8) else f.tensors[n].nbytes
               for n in loaded)
    assert w == want, (w, want)


_CALLS = ("jit", "pjit", "closed_call", "core_call", "remat", "checkpoint", "custom_jvp_call", "custom_vjp_call")


def _plane_readers(jaxpr, pv, kernel):
    """-> names of the ops that read any of the plane vars pv anywhere but in the kernel (a jit named `kernel`), looking
    inside nested calls and lax.scan bodies: a plane var among a scan's consts maps to the body's const; among its carry
    or xs, the scan itself reads (slices) it"""
    readers = set()
    for e in jaxpr.eqns:
        hit = [i for i, v in enumerate(e.invars) if type(v).__name__ != "Literal" and v in pv]
        if not hit:
            continue
        name = e.primitive.name
        if name in ("jit", "pjit") and e.params.get("name") == kernel:
            continue
        if name == "scan":
            body = e.params["jaxpr"].jaxpr
            if "num_consts" in e.params:
                nc = e.params["num_consts"]
            else:                                   # JAX 0.11: ft_in = (consts, carry, xs) as flat trees
                nc = len(list(e.params["ft_in"].elts[0].vals))
            if any(i >= nc for i in hit):
                readers.add("scan (slices the stack per step)")
            readers |= _plane_readers(body, {body.invars[i] for i in hit if i < nc}, kernel)
            continue
        inner = e.params.get("jaxpr") or e.params.get("call_jaxpr")
        if name in _CALLS and inner is not None:
            ij = getattr(inner, "jaxpr", inner)
            readers |= _plane_readers(ij, {ij.invars[i] for i in hit}, kernel)
            continue
        readers.add(name)
    return readers


def _layer_stub(name, picks_layer, calls):
    """kernel package stub on Q8_0 (the reference math on the file's blocks; rows reversed as its 'layout'). Its kernel is
    a jit named _stub_kq_kernel. picks_layer: matmul takes layer= (planes stacked on axis 0, the layer read inside the
    kernel), as the TPU kernel package does; else planes must arrive as one tensor. calls: (planes ndim, layer given)."""
    import types
    from jax import lax
    from mashbox import ops
    stub = types.ModuleType(name)
    stub.TYPES = (8,)
    stub.repack = lambda t, raw, shape: {"w": raw[::-1], "meta": jnp.zeros((t, shape[-2], shape[-1], 0), jnp.int8)}
    stub.dequant = lambda p: ops.reference.dequant(p["w"][..., ::-1, :], p["meta"].shape[-4], p["meta"].shape[-2])

    def _stub_kq_kernel(x, p, layer):
        w = p["w"] if layer is None else lax.dynamic_index_in_dim(p["w"], layer, 0, keepdims=False)
        m = p["meta"]
        return ops.qmatmul(x, ops.Raw(w[::-1], m.shape[-4], (m.shape[-3], m.shape[-2])))
    kern = jax.jit(_stub_kq_kernel)
    if picks_layer == "raises":                  # takes layer= but cannot run it (e.g. a payload for one tensor)
        def matmul(x, p, layer=None):
            if layer is not None:
                raise ValueError("this kernel build has no stacked-planes program")
            calls.append((p["w"].ndim, False))
            return kern(x, p, None)
    elif picks_layer:
        def matmul(x, p, layer=None):
            calls.append((p["w"].ndim, layer is not None))
            return kern(x, p, layer)
    else:
        def matmul(x, p):
            calls.append((p["w"].ndim, False))
            return kern(x, p, None)
    stub.matmul = matmul
    return stub


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe", "llama"])
def test_kq_layer_is_picked_in_the_kernel(arch, tmp_path, monkeypatch):
    """Weights stacked over layer groups run in one lax.scan. With a kernel package whose matmul takes layer=, every
    stacked kernel weight reaches the kernel whole with the layer index, and no op of the engine reads a plane stack
    (the scan does not slice it: on TPU a slice is a copy of the layer's planes at every call). (1) jaxpr walk of one
    decode step: no reader of a stacked plane array outside the kernel. (2) calls: stacked planes + layer, and the
    unstacked output weight without. (3) logits, prefill + decode, bit for bit those of a kernel without layer= (whose
    planes the scan slices), and NMSE <= 1e-10 vs the exact path (the stub is the reference math). qwen35 and the dense
    archs (llama here) share the scan helpers. qwen35moe: its expert stacks (a stack per group of a stack of experts) are
    on kq too and stay scanned (layer= takes one stacked axis), while its other weights are picked in place."""
    import sys
    from mashbox import ops
    from mashbox.gguf import open_model
    if arch.startswith("qwen35"):
        moe = dict(E=8, K=2, FE=32, FS=32) if arch == "qwen35moe" else None
        q = FXT.write_qwen35(str(tmp_path / "q8.gguf"), seed=5, layers=8, quant=["Q8_0"], moe=moe)
    else:
        q = FXT.write_dense(str(tmp_path / "q8.gguf"), arch=arch, seed=5, layers=4, quant=["Q8_0"])
    toks = list(range(3, 19))

    def logits(use_kernels):
        f, M, c = open_model(q, act=jnp.float32)
        W, _ = M.load(f, c, use_kernels=use_kernels)
        fwd0 = M.make_forward(c)
        fwd = jax.jit(fwd0, static_argnames=("Tc", "contig", "all_logits"))
        ck, cv = M.new_state(c, 1, 32)
        i32, z, t = jnp.int32, jnp.zeros((1,), jnp.int32), jnp.asarray([toks], jnp.int32)
        r = fwd(W, ck, cv, t[:, :8], z, jnp.asarray([8], i32), z, 0, all_logits=True)
        out = [np.asarray(r["lg"][0])]
        ck, cv = r["ck"], r["cv"]
        for i in range(8, len(toks)):
            r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
            ck, cv = r["ck"], r["cv"]
            out.append(np.asarray(r["lg"][0]))
        step = lambda W_: fwd0(W_, ck, cv, t[:, 8:9], jnp.asarray([8], i32), jnp.asarray([1], i32), z, 32)
        return np.concatenate(out, 0), W, step

    exact, _, _ = logits(False)
    got = {}
    for picks in (False, True):
        calls = []
        name = f"_stub_gguf_kernels_layer_{int(picks)}"
        monkeypatch.setitem(sys.modules, name, _layer_stub(name, picks, calls))
        try:
            assert ops.load_kernels(module=name) == frozenset({8})
            lg, W, step = logits(True)
            if picks:
                assert isinstance(W["g"], dict), "the fixture's layers are stacked for one lax.scan"
                ps = [h for h in jax.tree_util.tree_leaves(W["g"], is_leaf=lambda v: isinstance(v, ops.Packed))
                      if isinstance(h, ops.Packed)]
                hs = [h for h in ps if len(h.shape) == 2]
                assert hs and all(h.packed["w"].ndim == 3 for h in hs), hs
                ex = [h for h in ps if len(h.shape) == 3]
                assert bool(ex) == (arch == "qwen35moe") and all(h.packed["w"].ndim == 4 for h in ex), ex
                leaves, tree = jax.tree_util.tree_flatten(W)
                stack_ids = {id(a) for h in hs for a in jax.tree_util.tree_leaves(h.packed)}
                cj = jax.make_jaxpr(lambda *lv: step(jax.tree_util.tree_unflatten(tree, lv)))(*leaves)
                pv = {v for v, a in zip(cj.jaxpr.invars, leaves) if id(a) in stack_ids}
                assert len(pv) == 2 * len(hs), (len(pv), len(hs))
                readers = _plane_readers(cj.jaxpr, pv, "_stub_kq_kernel")
                assert not readers, f"ops that read a stacked plane array outside the kernel: {sorted(readers)}"
        finally:
            ops.load_kernels(module="off")
        got[picks] = (lg, set(calls))
    assert got[False][1] == {(2, False)}, got[False][1]
    assert got[True][1] == {(3, True), (2, False)}, f"kernel calls (planes ndim, layer given): {got[True][1]}"
    assert np.array_equal(got[True][0], got[False][0]), "logits differ from the sliced-planes path"
    e = nmse(got[True][0], exact)
    print(f"{arch}: layer picked in the kernel vs sliced planes: bit-equal; vs exact path NMSE {e:.3e}")
    assert e <= 1e-10, e


def test_kq_layer_failure_keeps_the_sliced_path(tmp_path, monkeypatch, capsys):
    """A kernel package whose matmul names layer= but raises on a stack: the first stack of each (type, shape) runs the
    layer= form once at load (the layer preflight); it fails, so those stacks are sliced per layer as before, with one
    warning per (type, shape), and logits (prefill + decode) are bit for bit those of a kernel without layer=."""
    import sys
    from mashbox import ops
    q = FXT.write_qwen35(str(tmp_path / "q8.gguf"), seed=5, layers=8, quant=["Q8_0"])
    toks = list(range(3, 19))
    got = {}
    for mode in (False, "raises"):
        calls = []
        name = f"_stub_gguf_kernels_layerfail_{mode}"
        monkeypatch.setitem(sys.modules, name, _layer_stub(name, mode, calls))
        try:
            assert ops.load_kernels(module=name) == frozenset({8})
            capsys.readouterr()
            lg, _, _ = run(q, toks, split=8, use_kernels=True)
            err = capsys.readouterr().err
            failed = {k for k, v in getattr(ops, "_KQ_LAYER", {}).items() if v is not True}
        finally:
            ops.load_kernels(module="off")
        got[mode] = (lg, set(calls), err, failed)
    lg, calls, err, failed = got["raises"]
    assert failed and calls == {(2, False)}, (failed, calls)
    warns = [ln for ln in err.splitlines() if "kq layer= for" in ln]
    assert len(warns) == len(failed) and all("sliced per layer" in ln for ln in warns), err
    assert np.array_equal(lg, got[False][0]), "logits differ from the sliced-planes path"


# ------------------------------------------------------------------------------------------- K/V cache read and write
# The K/V cache's layout, read and write live in runtime/kvcache.py (every model the engine runs) and are checked in
# tests/cpu/test_kvcache.py; this file keeps the qwen35 entry points on it.
def test_qwen35_cache_is_the_shared_layout():
    from mashbox.runtime import kvcache
    assert A.kv_write.__module__ == A.__name__ and "kvcache.write" in __import__("inspect").getsource(A.kv_write)
    assert "kvcache.read_heads" in __import__("inspect").getsource(A.kv_read)
    Kc, Vc = kvcache.zeros(3, 5, 16, 2, 4, jnp.float32)
    ks = jnp.ones((3, 1, 1, 2, 4))
    K2, _ = A.kv_write(Kc, Vc, jnp.array([2]), jnp.array([[7]]), ks, ks)
    assert K2.shape == (3, 5, 16, 8) and float(K2[:, 2, 7].min()) == 1.0 and float(jnp.abs(K2).sum()) == 3 * 8
    k, v = A.kv_read(K2, K2, 1, jnp.array([2, 0]), 8, False, 2, 4)
    assert k.shape == (2, 8, 2, 4) and float(k[0, 7].min()) == 1.0
