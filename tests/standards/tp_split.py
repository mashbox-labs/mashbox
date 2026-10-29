"""ST-B6's own process: tensor parallel 4 against 1 on 4 forced CPU devices, through the engine cli.load builds for a
models.json model with --tensor-parallel-size N (shard.mesh_for, shard.load, shard.TPModel, then Engine).

  XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 tests/standards/tp_split.py <workdir>   -> one JSON line

The tier's BSHJ fixture has 2 KV heads, so a 4-way split cannot give each chip one; this process builds the same tiny
Qwen3.8 architecture with the 4-chip layout (8 query / 4 KV / 4 linear key / 8 linear value heads, as
tests/cpu/tp_batched_checks.py) from the spec tests' builder, near-bigram weights. For tp 1 and tp 4: cli.warm (the
request path, then Engine.warm's program list), then one arrival schedule (rows join mid-batch, a stop token, a
cancel) through tiny.drive (the stop token: the first fresh token, position 2 or later, of tp 1's alone run of row 0), counting XLA compiles after the warm-up with the tier's own listener. Prints the rows, the
compiles, how many devices the weights span and how many K/V heads each device holds.

The schedule holds a penalised greedy row (temperature 0 with repetition / frequency / presence penalties) and a seeded
sampled row (temperature + top_k 5, within the 8 candidates, so the merge's cut is exact) next to plain greedy rows, as
tp_batched_checks.py TP5. Without them every token comes from the greedy decode program (the forward's own argmax,
runtime/engine.py) and the sharded sampler (runtime/sampler_tp.py) never decides a token: while one of them is in the
batch, every row's token (plain greedy rows too) comes from the sampler's merge (greedy rows: the first chip with the
largest value; the sampled row: the merged draw), so a wrong global id there changes tp 4's rows.
"""
import importlib.util
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import jax                                                          # noqa: E402

import tiny                                                         # noqa: E402
from mashbox import cli                                             # noqa: E402
from mashbox.models import qwen3_8_27b as Q                         # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW, shard as QS   # noqa: E402
from mashbox.runtime.engine import Engine                           # noqa: E402

TP = 4
LAYOUT = dict(H=8, KV=4, HK=4, HV=8)        # one KV head, one linear key head and two linear value heads per chip
# row 1: penalised greedy (deterministic, the sampler's greedy merge on penalised logits); row 4: seeded sampled draw
# (temperature + top_k 5 <= the 8 candidates: exact, the same tokens as sampler.py for the same logits); the rest greedy
SAMP = [None, dict(temperature=0.0, repetition_penalty=1.3, frequency_penalty=0.5, presence_penalty=0.2), None, None,
        dict(temperature=0.9, top_k=5, seed=3), None]


def spec_copy():
    """a private copy of the spec tests' builder with the 4-chip head layout (its tensors() reads these at call time)"""
    sp = importlib.util.spec_from_file_location("std_spec_tp", os.path.join(ROOT, "tests", "cpu", "test_spec_in_step.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    for k, v in LAYOUT.items():
        setattr(m, k, v)
    return m


def spans(tree):
    return max(len(x.sharding.device_set) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "sharding"))


def kv_share(eng):
    """K/V heads each device holds in the slot state (TP7's read): cfg.KV // tp when the cache is split"""
    (Kc, _Vc), _rest = eng.state if eng.state is not None else eng.new_state()
    return sorted({sh.data.shape[3] for sh in Kc.addressable_shards})


def main(wd):
    assert len(jax.devices()) >= TP, f"needs {TP} host devices, has {len(jax.devices())}"
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_compilation_cache", False)
    os.makedirs(wd, exist_ok=True)
    S = spec_copy()
    tiny._bshj_config(wd, S)
    path = os.path.join(wd, "bigram.bshj")
    S.write_bshj(path, S.tensors(11, tiny.LO))
    cfg = Q.Config(wd)
    tiny.compile_count()
    out, stop = dict(layout=LAYOUT, tp=TP, samp=SAMP), None
    for tp in (1, TP):
        if tp == 1:
            W, _ = QW.load(path, cfg)
            model = Q
        else:
            mesh = QS.mesh_for(tp)
            W, _ = QS.load(path, cfg, mesh)
            # 8 candidates of 128 per chip (as tp_batched_checks.py): the sharded sampler's merge runs, the path every row
            # takes on chip, where a chip's vocabulary share is far above the candidate count. With the default, every
            # tiny-vocabulary token is a candidate and greedy rows skip the merge.
            model = QS.TPModel(Q, mesh, candidates=8)
        eng = Engine(cfg, W, model, CTX=256, S=4, K=4, prefix_cache=False, cache_ram=64 << 20)
        ps = tiny.prompts(eng, lengths=(5, 17, 9, 30, 3, 12), seed=21)
        a0 = tiny.alone(eng, ps[0], 10)               # before the warm-up, so its programs are not counted after it
        if tp == 1:                                   # the stop: tp 1's first fresh token of row 0's alone run (B3's rule)
            stop = next((t for k, t in enumerate(a0) if k >= 2 and t not in a0[:k]), None)
        c0 = tiny.compile_count()
        log, cli.log = cli.log, (lambda s: None)
        try:
            cli.warm(eng, tiny.IdChat())
        finally:
            cli.log = log
        c1 = tiny.compile_count()
        plan = [(at, dict(ids=p, max_new=m, samp=sp)) for at, p, m, sp in zip([0, 0, 1, 2, 4, 6], ps, [10, 8, 12, 6, 10, 9], SAMP)]
        rows = tiny.drive(eng, plan, stop_ids=() if stop is None else (stop,), cancel_at={2: 4})
        out[f"tp{tp}"] = dict(alone0=a0, stop=stop, rows=[dict(ids=r["ids"], fin=r["fin"]) for r in rows], warm_compiles=c1 - c0,
                              compiles_after_warm=tiny.compile_count() - c1, weight_devices=spans(W),
                              kv_heads_per_chip=kv_share(eng))
        del eng, W
    print("ST_B6 " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
