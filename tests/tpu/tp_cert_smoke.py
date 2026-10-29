"""Chip smoke for the batched engine under tensor parallel (v6e-4, v5e-4): one process per arm, then a compare step.

  python3 tests/tpu/tp_cert_smoke.py run --tp 4 --weights <.bshj> --model-dir <dir with config.json> --out <arm.json>
  python3 tests/tpu/tp_cert_smoke.py run --tp 1 ...            (one chip of the same host, same requests)
  python3 tests/tpu/tp_cert_smoke.py compare <tp1.json> <tp4.json>

Each arm, plain JAX decode (no exported kernels, so both arms run the same math): Engine.warm, then (a) 8 prompts greedy
alone and together, (b) 8 sampled rows twice (a seeded rerun must repeat), (c) logits at 16 positions of 2 prompts, (d)
engine tok/s at 1 / 8 / 64 users (64 queue for 8 slots; prompt 128 tokens, 128 new, greedy; total and per user, time
to first token after admission p50 / p95), counting XLA compiles after the warm-up. The tp 4 arm also checks the vocabulary-sharded
sampler against the one-chip sampler on the model's own logits. The compare step prints KL of tp 4 vs tp 1 and token
equality. Every number lands in the JSON with the chip, jax version and the code sha from $CODE_SHA.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))


def run(a):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from mashbox.models import qwen3_8_27b as Q
    from mashbox.models.qwen3_8_27b import weights as QW, shard as QS
    from mashbox.runtime.engine import Engine
    from mashbox.runtime import sampler as SP, sampler_tp as SPT
    try:                # the engine's compile watch (names every compile after ready), when this tree has it
        from mashbox.runtime import programs as PG
    except ImportError:
        PG = None

    comp = [0]
    jax.monitoring.register_event_duration_secs_listener(
        lambda ev, d, **k: comp.__setitem__(0, comp[0] + (ev == "/jax/core/compile/backend_compile_duration")))
    out = dict(tp=a.tp, ctx=a.ctx, slots=a.slots, jax=jax.__version__, chip=jax.devices()[0].device_kind,
               devices=len(jax.devices()), code_sha=os.environ.get("CODE_SHA"), started=time.strftime("%FT%TZ", time.gmtime()))
    cfg = Q.Config(a.model_dir)
    t0 = time.time()
    if a.tp == 1:
        W, info = QW.load(a.weights, cfg)
        model = Q
    else:
        mesh = QS.mesh_for(a.tp)
        W, info = QS.load(a.weights, cfg, mesh)
        model = QS.TPModel(Q, mesh)
    out["load"] = dict(info, wall_s=round(time.time() - t0, 1))
    e = Engine(cfg, W, model, CTX=a.ctx, S=a.slots, K=8, prefix_cache=True)
    t0 = time.time()
    out["warm"] = dict(e.warm(workers=8), wall_s=round(time.time() - t0, 1))
    c_ready = comp[0]
    if PG is not None:
        PG.mark_ready()
    names = [{}]

    def new_names():  # compile names after ready since the last call (from the engine's compile watch)
        if PG is None:
            return None
        now = PG.counts()["after_ready_by_name"]
        d = {k: v - names[0].get(k, 0) for k, v in now.items() if v != names[0].get(k, 0)}
        names[0] = now
        return d
    cnames = {}
    rng = np.random.default_rng(20261028)
    prompts = [rng.integers(1000, 150000, n).tolist() for n in (37, 96, 150, 211, 64, 300, 128, 77)]

    def drive(reqs, max_steps=100000):
        rows, pend, k, t_first, t_start = [], list(reqs), 0, {}, time.perf_counter()
        while pend or e.busy():
            n = min(e.free_slots(), len(pend))
            if n:
                rows += e.admit(pend[:n])
                pend = pend[n:]
            e.step()
            k += 1
            assert k < max_steps
        return rows, time.perf_counter() - t_start

    q = lambda p, n, **kw: dict(ids=p, max_new=n, **kw)
    cph = {}            # compiles per phase after the warm-up: serving phases (requests) and test-only phases apart

    def mark(name, c0):
        cph[name] = comp[0] - c0
        cnames[name] = new_names()
        return comp[0]
    c0 = comp[0]
    alone = [drive([q(p, 48)])[0][0]["ids"] for p in prompts]
    together = [r["ids"] for r in drive([q(p, 48) for p in prompts])[0]]
    c0 = mark("serve_greedy", c0)
    out["greedy"] = dict(alone=alone, together=together, alone_eq_together=[x == y for x, y in zip(alone, together)])
    samp = [dict(temperature=0.8, top_p=0.9, seed=1), dict(temperature=1.0, top_k=20, seed=2), dict(temperature=0.7, seed=3),
            dict(temperature=1.0, min_p=0.05, seed=4), dict(temperature=0.6, top_k=40, top_p=0.95, seed=5),
            dict(temperature=1.0, top_p=0.5, repetition_penalty=1.1, seed=6), None, dict(temperature=1.2, top_p=0.99, seed=8)]
    s1 = [r["ids"] for r in drive([q(p, 48, samp=s) for p, s in zip(prompts, samp)])[0]]
    s2 = [r["ids"] for r in drive([q(p, 48, samp=s) for p, s in zip(prompts, samp)])[0]]
    c0 = mark("serve_sampled", c0)
    out["sampled"] = dict(run1=s1, run2=s2, repeats=s1 == s2)
    pos = {}
    for i in (() if a.skip_test_phases else (1, 5)):
        lg = np.asarray(e.logits(prompts[i]), np.float32)
        idx = np.linspace(0, len(prompts[i]) - 1, 16).astype(int)
        np.save(a.out + f".lg{i}.npy", lg[idx])
        pos[i] = idx.tolist()
    out["logits_positions"] = pos
    c0 = mark("test_logits", c0)
    if a.tp > 1 and not a.skip_test_phases:        # the sharded sampler vs the one-chip sampler on the same (real vocabulary) logits
        lg = np.asarray(e.logits(prompts[2]), np.float32)[-8:]
        x1 = jax.device_put(jnp.asarray(lg), jax.devices()[0])
        xs = jax.device_put(jnp.asarray(lg), NamedSharding(model.mesh, P(None, "tp")))
        eq, n, fb = 0, 0, 0
        for st in range(16):
            for sett in ([None] * 8, samp, [dict(temperature=1.0, top_k=k, seed=st) for k in (1, 5, 50, 255, 256, 257, 1000, 4096)],
                         [dict(temperature=1.0, top_p=p, seed=st) for p in (0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999)]):
                sp = SP.to_device(SP.pack(sett, 8))
                stp = jnp.full((8,), st, jnp.int32)
                ref = np.asarray(jax.jit(SP.sample)(x1, jax.device_put(sp, jax.devices()[0]), jax.device_put(stp, jax.devices()[0])))
                got, need = jax.jit(model.sampler.sample_rows)(xs, sp, stp)
                eq += int((ref == np.asarray(got)).sum())
                n += 8
                fb += int(np.asarray(need).sum())
        out["sampler_vs_one_chip"] = dict(equal=eq, draws=n, fallback_rows=fb, candidates=model.sampler.C)
    c0 = mark("test_sampler_check", c0)
    speed = {}
    for users in (1, 8, 64):
        reqs = [q(rng.integers(1000, 150000, 128).tolist(), 128) for _ in range(users)]
        rows, wall = drive(reqs)
        toks = sum(len(r["ids"]) for r in rows)
        ttft = sorted(r["t_first"] - r["t_admit"] for r in rows)
        dec = [(len(r["ids"]) - 1) / (r["t_end"] - r["t_first"]) for r in rows if r["t_end"] and r["t_end"] > r["t_first"]]
        speed[users] = dict(tokens=toks, wall_s=round(wall, 2), total_tok_s=round(toks / wall, 1),
                            per_user_decode_tok_s_median=round(float(np.median(dec)), 2),
                            ttft_after_admit_p50_s=round(ttft[len(ttft) // 2], 3), ttft_after_admit_p95_s=round(ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))], 3))
    out["speed"] = speed
    c0 = mark("serve_speed", c0)
    if a.long:          # a.long prompts near the context length, all live at once (every slot holds a long cache)
        L = [max(256, a.ctx - 80 - 997 * i) for i in range(a.long)]
        reqs = [q(rng.integers(1000, 150000, n).tolist(), 48) for n in L]
        rows, wall = drive(reqs)
        ttft = sorted(r["t_first"] - r["t_admit"] for r in rows)
        dec = [(len(r["ids"]) - 1) / (r["t_end"] - r["t_first"]) for r in rows if r["t_end"] and r["t_end"] > r["t_first"]]
        out["long"] = dict(prompt_tokens=L, new_tokens=[len(r["ids"]) for r in rows], all_48=all(len(r["ids"]) == 48 for r in rows),
                           prompt_tokens_total=sum(L), wall_s=round(wall, 2), prefill_plus_decode_tok_s=round((sum(L) + sum(len(r["ids"]) for r in rows)) / wall, 1),
                           per_user_decode_tok_s_median=round(float(np.median(dec)), 2) if dec else None,
                           ttft_after_admit_p50_s=round(ttft[len(ttft) // 2], 3), ttft_after_admit_p95_s=round(ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))], 3),
                           first16=[r["ids"][:16] for r in rows])
        mark("serve_long", c0)
    out["compiles_by_phase"] = cph
    out["compiles_after_warm"] = comp[0] - c_ready
    out["compile_names_by_phase"] = cnames
    if PG is not None:
        out["compile_watch"] = PG.counts()
    try:
        out["hbm"] = [{k: d.memory_stats().get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")} for d in jax.devices()]
    except Exception as x:
        out["hbm"] = repr(x)
    out["stats"] = e.stats
    out["finished"] = time.strftime("%FT%TZ", time.gmtime())
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("greedy", "sampled")}, default=str))


def compare(p1, p4):
    A, B = json.load(open(p1)), json.load(open(p4))
    kls = []
    for i in A["logits_positions"]:
        a, b = np.load(p1 + f".lg{i}.npy").astype(np.float64), np.load(p4 + f".lg{i}.npy").astype(np.float64)
        la = a - a.max(-1, keepdims=True); la -= np.log(np.exp(la).sum(-1, keepdims=True))
        lb = b - b.max(-1, keepdims=True); lb -= np.log(np.exp(lb).sum(-1, keepdims=True))
        kls += list((np.exp(la) * (la - lb)).sum(-1))
        t1 = a.argmax(-1) == b.argmax(-1)
    g = lambda X, k: X["greedy"][k]
    first_diff = lambda x, y: next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), None)
    r = dict(kl_tp4_vs_tp1_mean=float(np.mean(kls)), kl_max=float(np.max(kls)), top1_last=bool(t1.all()),
             greedy_alone_equal=[x == y for x, y in zip(g(A, "alone"), g(B, "alone"))],
             greedy_alone_first_diff=[first_diff(x, y) for x, y in zip(g(A, "alone"), g(B, "alone"))],
             greedy_together_equal=[x == y for x, y in zip(g(A, "together"), g(B, "together"))],
             sampled_equal=[x == y for x, y in zip(A["sampled"]["run1"], B["sampled"]["run1"])],
             tp4_sampled_repeats=B["sampled"]["repeats"], tp1_sampled_repeats=A["sampled"]["repeats"])
    print(json.dumps(r))


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)     # stoppable by SIGINT even when started from a shell that ignores it
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r_ = sub.add_parser("run")
    r_.add_argument("--tp", type=int, default=4)
    r_.add_argument("--weights", required=True)
    r_.add_argument("--model-dir", required=True)
    r_.add_argument("--out", required=True)
    r_.add_argument("--ctx", type=int, default=4096)
    r_.add_argument("--slots", type=int, default=8)
    r_.add_argument("--skip-test-phases", action="store_true", help="skip the test-only logits and sampler phases (they hold a second slot state)")
    r_.add_argument("--long", type=int, default=0, help="this many prompts near the context length, served together")
    c_ = sub.add_parser("compare")
    c_.add_argument("a")
    c_.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a.a, a.b)
