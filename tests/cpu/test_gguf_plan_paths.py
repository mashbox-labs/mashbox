"""GGUF plan and memory estimate follow the path each tensor takes (ops.weight_path), not its block type alone. With
native planes for a type that kq lacks, stacked experts of that type run native while dense weights of the same type run
on the exact float32 reference path: the plan names those in its warning and the estimate counts their float32 scratch."""
import importlib.util
import os
import sys
import types

import numpy as np
import jax.numpy as jnp

from mashbox import ops
from mashbox import gguf as G
from mashbox.gguf.reader import File
from mashbox.gguf.archs import qwen35 as A
from mashbox.ops import reference as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q4_0, Q8_0 = 2, 8


def _fixture(tmp):
    spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    return fx.build("qwen35moe-pyquant", str(tmp))


def test_plan_and_scratch_follow_the_per_tensor_path(tmp_path, monkeypatch):
    f = File(_fixture(tmp_path))
    c = A.Config(f, act=jnp.float32)
    _, mult = A.tensor_roles(f, c)
    m = types.ModuleType("stub_nb_only")
    m.TYPES = frozenset()                                               # no kq planes
    m.nb_types = lambda: frozenset({Q4_0, Q8_0})                        # native planes for Q4_0 and Q8_0
    m.nb_bytes = lambda t, shape: int(np.prod(shape[:-1])) * (shape[-1] // R.TYPES[t][1]) * R.TYPES[t][2]
    monkeypatch.setitem(sys.modules, "stub_nb_only", m)
    ops.load_kernels(module="stub_nb_only")
    try:
        kt = ops.kernel_types()
        _, total = f.census()
        gathered = {"token_embd.weight"} if "output.weight" in f.tensors else set()     # read by row, not multiplied
        exact = {n for n, t in f.tensors.items()
                 if t.type_name != "F32" and n not in gathered and ops.weight_path(t.type, t.shape, kernel_types=kt) == "exact"}
        assert {n for n in exact & mult if f.tensors[n].type in (Q4_0, Q8_0)}   # dense Q4_0 / Q8_0 on the exact path
        line, warn = G.plan(f, c)
        assert line["reference_share"] == round(100.0 * sum(f.tensors[n].nbytes for n in exact) / max(1, total), 1)
        for name in {f.tensors[n].type_name for n in exact}:
            assert warn and name in warn, name
        sh = [tuple(int(v) for v in f.tensors[n].shape) for n in mult & exact]
        big = max([ops.exact_chunk_rows(s[-2], s[-1]) * s[-1] * 4 for s in sh if len(s) >= 2], default=0)
        from mashbox import moe                     # + the routed op's own scratch at tokens_per_call 2048 (fit item a)
        ex = [(f.tensors[n].type, f.tensors[n].shape, f.tensors[n].nbytes) for n in mult if len(f.tensors[n].shape) > 2]
        routed = moe.need_bytes(c.spec, c.D, 2048, ex, np.dtype(c.act).itemsize, kt)
        assert big > 0 and routed > 0 and G.need_bytes(f, c, 64, 2, kt)[2] == big + routed
    finally:
        ops.load_kernels(module="off")



def test_plan_leaves_out_the_gathered_embedding_table():
    """On the header of Qwen3.6-35B-A3B UD-Q4_K_M (its tensor list): token_embd.weight (Q8_0, 2.4% of the bytes) is only
    gathered, one row per token, because the file has its own output.weight, so the plan's reference share and warning
    leave it out, as the load's weights event does. Without output.weight it is the head and counts. Before, that file
    on v6e with every multiplied weight on a kernel printed 'Q8_0 (2.4% of the weights) ... slow'."""
    import json
    d = json.load(open(os.path.join(ROOT, "tests", "fixtures", "moe", "tensor-lists",
                                    "Qwen3.6-35B-A3B-UD-Q4_K_M.tensors.json")))

    def header(drop=()):
        h = types.SimpleNamespace(arch=d["metadata"]["general.architecture"],
                                  tensors={t["name"]: types.SimpleNamespace(**t) for t in d["tensors"] if t["name"] not in drop})
        h.census = lambda: File.census(h)
        return h

    def share(h, skip):                     # no kernels: every non-F32 tensor that is multiplied is on the exact path
        _, total = h.census()
        return round(100.0 * sum(t.nbytes for n, t in h.tensors.items() if t.type_name != "F32" and n not in skip) / total, 1)

    c = types.SimpleNamespace(L=40, nextn=0)
    h = header()
    emb = h.tensors["token_embd.weight"]
    assert "output.weight" in h.tensors and emb.type_name == "Q8_0"
    line, _ = G.plan(h, c, kernel_types=frozenset())
    assert line["reference_share"] == share(h, {"token_embd.weight"}), line["reference_share"]
    tied = header(drop=("output.weight",))
    line_t, _ = G.plan(tied, c, kernel_types=frozenset())
    assert line_t["reference_share"] == share(tied, set()), line_t["reference_share"]
