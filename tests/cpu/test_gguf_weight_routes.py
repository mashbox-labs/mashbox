"""Every weight a contraction multiplies is counted once in the load's weights event, in the bucket of the path it
runs on.

kernel: kq or native planes. reference: the exact path, which rebuilds the values from the file's blocks at every call.
resident_f32: float32 values held on the device and multiplied by the exact path's own float32 dot with no rebuild (an
F32 weight held as values, the routed-expert router, the shared expert's gate): exact, and never counted as a kernel.
native_bf16: BF16 weights on the native bfloat16 matmul. A tied output head is the token embedding multiplied, so it is
counted too. Elementwise tensors (norms, conv taps, ssm_a, ssm_dt, biases) are not multiplied and count nowhere.

The tensors each graph multiplies are written out here from llama.cpp's graphs, not read from the loaders, so a loader
cannot agree with itself."""
import importlib.util
import os
import re

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures",
                                               os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

from mashbox import ops
from mashbox.gguf.archs import dense, gemma4, qwen35
from mashbox.gguf.reader import File

CONTRACTED = {"attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight",
              "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight",
              "ffn_up.weight", "ffn_down.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp.weight",
              "ffn_gate_inp_shexp.weight", "output.weight"}
MOE = {"E": 8, "K": 2, "FE": 32, "FS": 32}
F32 = 0


@pytest.fixture(autouse=True)
def _no_kernels():
    ops.load_kernels(module="off")
    yield
    ops.load_kernels(module="off")


def role(name):
    return re.sub(r"^blk\.\d+\.", "", name)


def multiplied(f):
    """{role: file bytes} of every tensor the graph multiplies; a file without output.weight multiplies its embedding"""
    out = {}
    for n, t in f.tensors.items():
        if role(n) in CONTRACTED:
            out[role(n)] = out.get(role(n), 0) + int(t.nbytes)
    if "output.weight" not in f.tensors:
        out["token_embd.weight"] = int(f.tensors["token_embd.weight"].nbytes)
    return out


def f32_roles(f, roles):
    return {r for r in roles if all(t.type == F32 for n, t in f.tensors.items() if role(n) == r)
            and any(role(n) == r for n in f.tensors)}


def check_partition(info, want):
    """nothing multiplied is missing, nothing is counted twice, nothing elementwise is counted (kernels off)"""
    assert info["kernel_bytes"] == 0, info
    got = info["reference_bytes"] + info["resident_f32_bytes"] + info["native_bf16_bytes"]
    assert got == sum(want.values()), (got, sum(want.values()), want, info)
    assert info["resident_f32_bytes"] == sum(info["resident_f32_roles"].values()), info
    assert info["resident_f32_bytes"] == sum(info["resident_f32_types"].values()), info


def test_qwen35moe_unrolled_f32_gates_and_router_are_resident_f32_not_reference(tmp_path):
    p = FXT.write_qwen35(str(tmp_path / "m.gguf"), seed=4, layers=8, quant=["Q8_0", "Q4_0"], moe=MOE)
    f = File(p)
    _, info = qwen35.load(f, qwen35.Config(f, act=jnp.float32), use_kernels=True)
    assert not info["layers_stacked"], info             # the mixed file loads unrolled: F32 2-D weights are Dense32
    want = multiplied(f)
    res = f32_roles(f, want)
    assert res == {"ssm_alpha.weight", "ssm_beta.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"}, res
    assert info["resident_f32_roles"] == {r: want[r] for r in res}, info
    assert info["resident_f32_types"] == {"F32": sum(want[r] for r in res)}, info
    assert "F32" not in info["reference_types"], info
    assert info["reference_bytes"] == sum(n for r, n in want.items() if r not in res), info
    check_partition(info, want)


def test_qwen35moe_stacked_router_and_shared_gate_are_counted(tmp_path):
    p = FXT.write_qwen35(str(tmp_path / "s.gguf"), seed=4, layers=8, quant=None, moe=MOE)
    f = File(p)
    _, info = qwen35.load(f, qwen35.Config(f, act=jnp.float32), use_kernels=True)
    assert info["layers_stacked"], info                 # a stack of F32 linears stays Raw (the exact path chunks it)
    want = multiplied(f)
    small = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"}
    assert {r: info["resident_f32_roles"].get(r) for r in small} == {r: want[r] for r in small}, info
    check_partition(info, want)


def test_qwen2moe_router_and_shared_gate_are_counted(tmp_path):
    p = FXT.write_dense(str(tmp_path / "d.gguf"), arch="qwen2moe", seed=4, layers=4, quant=["Q8_0", "Q4_0"], moe=MOE)
    f = File(p)
    _, info = dense.load(f, dense.Config(f, act=jnp.float32), use_kernels=True)
    want = multiplied(f)
    small = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"}
    assert {r: info["resident_f32_roles"].get(r) for r in small} == {r: want[r] for r in small}, info
    check_partition(info, want)


def test_a_tied_head_is_counted_as_the_embedding_it_multiplies(tmp_path):
    p = FXT.write_gemma4(str(tmp_path / "g.gguf"), seed=4, layers=6, quant=["Q8_0", "Q4_0"])
    f = File(p)
    assert "output.weight" not in f.tensors             # the fixture's head is tied
    _, info = gemma4.load(f, gemma4.Config(f, act=jnp.float32), use_kernels=True)
    want = multiplied(f)
    check_partition(info, want)
    t = f.tensors["token_embd.weight"]
    where = info["resident_f32_roles"] if t.type == F32 else info["reference_types"]
    assert where.get("token_embd.weight" if t.type == F32 else t.type_name, 0) >= int(t.nbytes), info


def test_bf16_weights_on_the_native_matmul_are_not_reference(tmp_path):
    p = FXT.write_qwen35(str(tmp_path / "b.gguf"), seed=4, layers=8, quant=["BF16", "Q8_0"])
    f = File(p)
    W, info = qwen35.load(f, qwen35.Config(f, act=jnp.float32), use_kernels=True)
    hs = [x for x in jax.tree_util.tree_leaves(W, is_leaf=lambda x: isinstance(x, ops.dense16.Dense16))
          if isinstance(x, ops.dense16.Dense16)]
    assert hs, "the fixture has BF16 weights on the native matmul"
    assert info["native_bf16_bytes"] == sum(int(x.w.nbytes) for x in hs), info
    assert "BF16" not in info["reference_types"], info
    check_partition(info, multiplied(f))
