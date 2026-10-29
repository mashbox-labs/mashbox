"""GGUF dense-attention files whose block types differ by layer (Q4_K_M keeps some layers' attn_v / ffn_down in Q6_K;
Unsloth's dynamic quants mix more): one lax.scan runs the layers, with one weight rebuild per type of each role
(ops.Mixed, picked per layer through lax.switch), not one unrolled copy of every layer per program. Unrolled, each
program grew with the layer count, and the serve warm-up's ~140 programs filled host RAM and the compile cache.

Archs: dense (llama, qwen2, qwen3) and qwen35 (layer groups of 4). Bars, fixed before the first run:
  - the program does not grow with the layer count: HLO ops of the decode program at 12 layers <= 1.10 x at 4 layers
    (qwen35: 24 layers vs 8; unrolled it is ~3x);
  - logits equal the unrolled path's (MIXED_SCAN = False) on the same file: the same argmax at every position and
    max |diff| <= 1e-5 x max |logit| (both float32 at HIGHEST), prefill and K/V-cache decode;
  - with a stub kernel package for one of the two types (a role then holds a kq-plane stack and a block stack), logits
    equal the exact path's (NMSE <= 1e-10, the bar of test_gguf_qwen35.py's stub-kernel tests).
"""
import os
import sys
import types

import numpy as np
import pytest

import gguf  # noqa: F401  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

from mashbox import ops  # noqa: E402
from mashbox.gguf import open_model  # noqa: E402
from mashbox.gguf.archs import dense as A  # noqa: E402

i32 = jnp.int32
STATIC = ("Tc", "contig", "all_logits", "verify", "hidden")
QUANT = ["Q8_0", "Q4_0"]         # cycled over the big linears: 7 per layer, so every role alternates by layer


def mixed_file(tmp, layers, arch="llama", quant=QUANT):
    return FXT.write_dense(str(tmp / f"{arch}-{layers}.gguf"), arch=arch, seed=3, layers=layers, quant=quant)


def load(path, use_kernels=False):
    f, M, c = open_model(path, act=jnp.float32)
    assert M is A
    W, info = A.load(f, c, use_kernels=use_kernels)
    return c, W, info


def logits(c, W, toks, split=None):
    fwd = jax.jit(A.make_forward(c), static_argnames=STATIC)
    ck, cv = A.new_state(c, 1, 32)
    z = jnp.zeros((1,), i32)
    t = jnp.asarray([toks], i32)
    T = len(toks)
    if split is None:
        return np.asarray(fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)["lg"][0])
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0)


def decode_ops(c, W):
    """HLO ops of one decode program (8 rows, one token, attention window 32), as the engine's dec lowers it"""
    fwd = jax.jit(A.make_forward(c), static_argnames=STATIC)
    ck, cv = A.new_state(c, 8, 64)
    z = jnp.zeros((8,), i32)
    txt = fwd.lower(W, ck, cv, jnp.zeros((8, 1), i32), z, jnp.ones((8,), i32), jnp.arange(8, dtype=i32), 32,
                    contig=True).as_text()
    return txt.count(" = ")


def test_program_does_not_grow_with_layers(tmp_path):
    n = {}
    for L in (4, 12):
        c, W, info = load(mixed_file(tmp_path, L))
        assert info["layers_stacked"] and not isinstance(W["g"], list), info
        assert set(info["mixed_roles"]) == {r + ".weight" for r in ("attn_q", "attn_k", "attn_v", "attn_output",
                                                                     "ffn_gate", "ffn_up", "ffn_down")}, info
        assert isinstance(W["g"]["wd"], ops.Mixed) and len(W["g"]["wd"].stacks) == 2, W["g"]["wd"]
        n[L] = decode_ops(c, W)
    print(f"decode program HLO ops: 4 layers {n[4]}, 12 layers {n[12]}")
    assert n[12] <= 1.10 * n[4], n


@pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_mixed_scan_equals_unrolled(arch, split, tmp_path, monkeypatch):
    p = mixed_file(tmp_path, 6, arch=arch, quant=["Q8_0", "Q4_0", "Q5_1"])
    toks = list(range(3, 19))
    c, W, info = load(p)
    assert info["layers_stacked"] and info["mixed_roles"], info
    a = logits(c, W, toks, split)
    monkeypatch.setattr(A, "MIXED_SCAN", False)
    c2, W2, info2 = load(p)
    assert info2["layers_stacked"] is False and isinstance(W2["g"], list) and info2["mixed_roles"] == [], info2
    b = logits(c2, W2, toks, split)
    d = float(np.max(np.abs(a - b)))
    print(f"{arch} {split}: max |scan - unrolled| {d:.3e} of max |logit| {float(np.max(np.abs(b))):.3e}")
    assert np.array_equal(a.argmax(-1), b.argmax(-1))
    assert d <= 1e-5 * float(np.max(np.abs(b))), d


def test_mixed_scan_with_a_kernel_type(tmp_path, monkeypatch):
    """a stub kernel package for Q8_0 (the reference math on reversed rows, as test_gguf_qwen35.py's stub): every role
    holds a Q8_0 kq-plane stack and a Q4_0 block stack; logits equal the exact path's"""
    stub = types.ModuleType("_stub_gguf_kernels_mixed")
    stub.TYPES = (8,)

    def repack(t, raw, shape):
        return {"w": raw[..., ::-1, :], "meta": jnp.zeros(raw.shape[:-2] + (t, shape[0], shape[1], 0), jnp.int8)}
    stub.repack = repack
    stub.dequant = lambda p: ops.reference.dequant(p["w"][..., ::-1, :], p["meta"].shape[-4], p["meta"].shape[-2])
    stub.matmul = lambda x, p: ops.qmatmul(x, ops.Raw(p["w"][::-1], p["meta"].shape[-4],
                                                       (p["meta"].shape[-3], p["meta"].shape[-2])))
    monkeypatch.setitem(sys.modules, "_stub_gguf_kernels_mixed", stub)
    p = mixed_file(tmp_path, 6)
    toks = list(range(3, 19))
    c, W, _ = load(p)
    a = logits(c, W, toks, 8)
    try:
        assert ops.load_kernels(module="_stub_gguf_kernels_mixed") == frozenset({8})
        c, W, info = load(p, use_kernels=True)
        h = W["g"]["wg"]
        assert isinstance(h, ops.Mixed) and {type(s) for s in h.stacks} == {ops.Packed, ops.Raw}, h
        assert info["kernel_bytes"] > 0 and "Q8_0" not in info["reference_types"], info
        b = logits(c, W, toks, 8)
    finally:
        ops.load_kernels(module="off")
    e = float(np.sum((b.astype(np.float64) - a) ** 2) / np.sum(a.astype(np.float64) ** 2))
    print(f"mixed scan, stub kernel for Q8_0 vs exact path: NMSE {e:.3e}")
    assert e <= 1e-10, e


# ------------------------------------------------------------------------------------------------ qwen35 (layer groups)
from mashbox.gguf.archs import qwen35 as Q  # noqa: E402
from mashbox.gguf.reader import File  # noqa: E402


def q35_file(tmp, layers, quant=QUANT):
    return FXT.write_qwen35(str(tmp / f"q35-{layers}.gguf"), seed=4, layers=layers, quant=quant)


def q35_load(path):
    f = File(path)
    c = Q.Config(f, act=jnp.float32)
    W, info = Q.load(f, c, use_kernels=False)
    return c, W, info


def q35_logits(c, W, toks, split=None):
    fwd = jax.jit(Q.make_forward(c), static_argnames=STATIC)
    ck, cv = Q.new_state(c, 1, 32)
    z = jnp.zeros((1,), i32)
    t = jnp.asarray([toks], i32)
    T = len(toks)
    if split is None:
        return np.asarray(fwd(W, ck, cv, t, z, jnp.asarray([T], i32), z, 0, all_logits=True)["lg"][0])
    r = fwd(W, ck, cv, t[:, :split], z, jnp.asarray([split], i32), z, 0, all_logits=True)
    out, ck, cv = [np.asarray(r["lg"][0])], r["ck"], r["cv"]
    for i in range(split, T):
        r = fwd(W, ck, cv, t[:, i:i + 1], jnp.asarray([i], i32), jnp.asarray([1], i32), z, 32, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        out.append(np.asarray(r["lg"][0]))
    return np.concatenate(out, 0)


def q35_decode_ops(c, W):
    fwd = jax.jit(Q.make_forward(c), static_argnames=STATIC)
    ck, cv = Q.new_state(c, 8, 64)
    z = jnp.zeros((8,), i32)
    txt = fwd.lower(W, ck, cv, jnp.zeros((8, 1), i32), z, jnp.ones((8,), i32), jnp.arange(8, dtype=i32), 32,
                    contig=True).as_text()
    return txt.count(" = ")


def test_qwen35_program_does_not_grow_with_groups(tmp_path):
    n = {}
    for L in (8, 24):                   # 2 and 6 layer groups of 4
        c, W, info = q35_load(q35_file(tmp_path, L))
        assert info["layers_stacked"] and not isinstance(W["g"], list) and info["mixed_roles"], info
        n[L] = q35_decode_ops(c, W)
    print(f"qwen35 decode program HLO ops: 8 layers {n[8]}, 24 layers {n[24]}")
    assert n[24] <= 1.10 * n[8], n


@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_qwen35_mixed_scan_equals_unrolled(split, tmp_path, monkeypatch):
    p = q35_file(tmp_path, 12, quant=["Q8_0", "Q4_0", "Q5_1"])
    toks = list(range(3, 19))
    c, W, info = q35_load(p)
    assert info["layers_stacked"] and info["mixed_roles"], info
    a = q35_logits(c, W, toks, split)
    monkeypatch.setattr(Q, "MIXED_SCAN", False)
    c2, W2, info2 = q35_load(p)
    assert info2["layers_stacked"] is False and isinstance(W2["g"], list) and info2["mixed_roles"] == [], info2
    b = q35_logits(c2, W2, toks, split)
    d = float(np.max(np.abs(a - b)))
    print(f"qwen35 {split}: max |scan - unrolled| {d:.3e} of max |logit| {float(np.max(np.abs(b))):.3e}")
    assert np.array_equal(a.argmax(-1), b.argmax(-1))
    assert d <= 1e-5 * float(np.max(np.abs(b))), d
