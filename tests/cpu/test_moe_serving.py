"""The tiny MoE GGUF fixtures (tests/cpu/test_gguf_moe.py) through inspect and the serving engine,
kernels off, so the routed op runs its exact path (tests/cpu/test_moe_fast_path.py covers the kernel path).

  inspect: its output per fixture equals the expected text: architecture, runs here, the plan line (4 layers, no MTP,
      F32 100.0%, no reference share, no kernel types), no warning, the tensor count, and the routed experts' path on
      this install ("exact", why "no kernel package", no kernel).
  no compile after ready, on MoE: after cli.warm (what `mashbox run` does before it reports ready), random traffic of
      rows 1..S x prompt lengths 1, 8, 64, 200 (CTX 256) x greedy / sampled compiles nothing: the delta of
      programs.counts()["compiles"] is 0. The delta, not compiles_after_ready: another module may have marked this
      process ready. The routed op must have been traced (expert_report path "exact", at least one site).
  one user alone, on MoE: greedy, gives identical ids run to run. Alone vs batched is recorded (one JSON line on
      stdout), never asserted equal (batched output may differ from alone)."""
import json
import os
import sys

import numpy as np
import pytest
import gguf  # noqa: F401  (a hard dependency: never skipped)

from mashbox import cli as CLI
from mashbox import ops
from mashbox.moe import record as RX   # the routed-expert report's names and shape check
from mashbox.gguf import File, inspect, open_model
from mashbox.runtime import programs
from mashbox.runtime.engine import Engine

sys.path.insert(0, os.path.dirname(__file__))
import test_gguf_moe as M  # noqa: E402  (its fixtures)

ARCH = {"qwen3moe-f32": "qwen3moe", "qwen2moe-f32": "qwen2moe", "llama-moe-f32": "llama"}
TENSORS = {"qwen3moe-f32": 51, "qwen2moe-f32": 71, "llama-moe-f32": 60}
SERVE = "qwen2moe-f32"          # routed experts + a gated shared expert + unnormalised weights: the most parts of the three
CTX, S = 256, 4


@pytest.mark.parametrize("name", M.MOE)
def test_i5_inspect_names_the_moe_file_and_its_routed_path(name, tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    p, _, _ = M.fixture(name, tmp_path)
    _, total = File(p).census()
    assert inspect(p)["runs_here"] is False                     # no passing row of its own: held (gguf.archs.held)
    M.listed(monkeypatch, ARCH[name], p)
    ins = inspect(p)
    assert ins["runs_here"] is True, ins.get("why")
    assert ins["arch"] == ARCH[name]
    assert ins["tensors"] == TENSORS[name]
    assert list(ins["types"]) == ["F32"]
    assert ins["types"]["F32"]["tensors"] == TENSORS[name] and ins["types"]["F32"]["share"] == 100.0
    assert ins["plan"] == dict(arch=ARCH[name], layers=4, mtp_layers=0, weights_gb=round(total / 1e9, 2),
                               types="F32 100.0%", kernel_types=[], reference_share=0.0, chip=None)
    assert ins["warning"] is None and ins["mtp_layers"] == 0
    assert ins["routed_experts"] == dict(path="exact", why="no kernel package", kernel=None)


class _Chat:
    def chat_ids(self, msgs, *a, **k):
        return [1, 2, 3, 4, 5]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    ops.load_kernels(module="off")
    p, meta, _ = M.fixture(SERVE, tmp_path_factory.mktemp("moe-serve"))
    f, A, c = open_model(p)
    W, _ = A.load(f, c, use_kernels=False)
    e = Engine(c, W, A, CTX=CTX, S=S, K=8, decode_kernels=None)
    CLI.warm(e, _Chat())
    return e, int(meta["n_vocab"])


@pytest.mark.slow
def test_i7_no_compile_after_warm_under_random_moe_traffic(served):
    """slow: cli.warm compiles the serving programs on CPU (about 25 s), then 32 requests"""
    e, V = served
    rng = np.random.default_rng(7)
    c0, hits = programs.counts()["compiles"], []
    assert c0 is not None, "cli.warm did not install the compile counter"
    for L in (1, 8, 64, 200):
        for n in range(1, S + 1):
            for samp in (None, dict(temperature=0.7, top_p=0.9, seed=3)):
                k0 = programs.counts()["compiles"]
                e.generate([rng.integers(1, V, L).tolist() for _ in range(n)], max_new=min(8, CTX - L), samp=[samp] * n)
                if programs.counts()["compiles"] > k0:
                    hits.append(dict(prompt=L, rows=n, sampled=samp is not None, compiles=programs.counts()["compiles"] - k0))
    assert programs.counts()["compiles"] - c0 == 0, hits[:3]
    rep = ops.expert_report()
    assert rep["path"] == "exact" and rep["sites"]["exact"] >= 1 and rep[RX.WHY] == ["no kernel package"], rep
    assert RX.problems(rep) == [], rep


@pytest.mark.slow
def test_i8_one_user_alone_greedy_repeats_run_to_run(served):
    """slow: shares the no-compile test's warmed engine (cli.warm on CPU, about 25 s, when run alone)"""
    e, V = served
    rng = np.random.default_rng(8)
    prompt = rng.integers(1, V, 24).tolist()
    a = e.generate([prompt], max_new=16, samp=[None])["ids"][0]
    b = e.generate([prompt], max_new=16, samp=[None])["ids"][0]
    assert len(a) == 16 and len(set(a)) >= 2, a       # a comparison of a constant sequence would prove nothing
    assert a == b
    others = [rng.integers(1, V, L).tolist() for L in (5, 40, 90)]
    batched = e.generate([prompt] + others, max_new=16, samp=[None] * 4)["ids"][0]
    print(json.dumps(dict(I8="alone_vs_batched", fixture=SERVE, alone=a, batched=batched, equal=a == batched)))
