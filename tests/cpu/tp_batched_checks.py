"""CPU checks of the batched engine under tensor parallel: 4 forced host devices (the v6e-4 / v5e-4 layout, one KV head per
chip) against one device, on a tiny random Qwen3.8 of the same architecture (as the 4-device check in qwen3_8_27b_checks.py).

  python3 tests/cpu/tp_batched_checks.py <workdir>          -> prints one JSON line per gate, then ALL_PASS or FAILED

  TP1 noise        the sharded sampler's noise, made from the global token id, is bit-equal to sampler.py's draw noise
  TP2 sampler      sharded sampler == sampler.py on the same logits: greedy, temperature, top_k, top_p, min_p, mixes,
                   ties across a shard boundary; small candidate lists so the fallback runs too
  TP3 collectives  the sampler's compiled program: 2 collectives on its main path, 1 more inside the fallback;
                   the decode programs' collectives per pass are recorded (not a gate)
  TP4 greedy       continuous batching (rows join mid-answer, chunked prefill) under tp 4 == tp 1, token for token
  TP5 sampled      sampled and penalised rows next to greedy rows under tp 4 == tp 1, and a tp 4 rerun repeats
  TP6 cache        prefix cache + host cache (more conversations than slots: saves, restores, slot swaps) under tp 4 ==
                   tp 1, same cached and restored counts, and a continued turn == the same prompt cold
  TP7 shares       after all of it each chip holds its own share of the slot state (K/V heads / 4, value heads / 4), and
                   the decode program's temp memory per chip is below the whole K/V (no chip gathers the cache)
  TP8 compiles     after Engine.warm on the mesh, requests compile NOTHING under tp 4 (no compile at serve time, on the mesh), and no more than
                   under tp 1; the requests include sampled first tokens at every prefill row bucket (alone, 2 and 4 rows at
                   once), whose logits-by-slot buffer must keep one placement (on chip it compiled once more under tp 4)
  TP9 ctx reads    the decode (1 and 8 rows) and prefill programs under tp 4 read only the cache positions they attend: no
                   gather takes, and no slice makes, a whole context axis (on a v6e-4 the 8-row step copied each full-attention
                   layer's whole cache group: 14.33 ms at ctx 4096, 42.78 ms at 32768); the one-device count is recorded
"""
import json
import os
import struct
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                     # noqa: E402
import torch                           # noqa: E402
import jax                             # noqa: E402
import jax.numpy as jnp                # noqa: E402
import ml_dtypes                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P   # noqa: E402

from mashbox.models import qwen3_8_27b as Q                  # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW, shard as QS   # noqa: E402
from mashbox.runtime.engine import Engine                    # noqa: E402
from mashbox.runtime import sampler as SP, sampler_tp as SPT  # noqa: E402

WD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mashbox-tp-test"
os.makedirs(WD, exist_ok=True)
R, G = {}, {}
ONLY = set(os.environ.get("TP_ONLY", "").split())     # e.g. TP_ONLY="TP8": run only those gates


def want(g):
    return not ONLY or g in ONLY


_COMPILES = [0]
jax.monitoring.register_event_duration_secs_listener(
    lambda ev, d, **k: _COMPILES.__setitem__(0, _COMPILES[0] + (ev == "/jax/core/compile/backend_compile_duration")))
assert len(jax.devices()) >= 4, "needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)"


def emit(name, ok, **rec):
    R[name], G[name] = rec, bool(ok)
    print(json.dumps(dict(gate=name, ok=bool(ok), **rec), default=str), flush=True)


# ---- tiny model (the 4-device layout of qwen3_8_27b_checks.py: 8 q / 4 KV / 4 linear key / 8 linear value heads, so each of 4 chips holds one KV head)
def build(md):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=8,
                            num_key_value_heads=4, head_dim=32, linear_num_key_heads=4, linear_num_value_heads=8,
                            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
                            rms_norm_eps=1e-6, rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                                                    mrope_section=[2, 1, 1], mrope_interleaved=True),
                            tie_word_embeddings=False, attn_output_gate=True, pad_token_id=0)
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif "dt_bias" in n:
                p.normal_(0, 1)
            elif p.ndim == 1:
                p.normal_(0, 0.3)
            elif "conv1d" in n:
                p.normal_(0, 0.4)
            else:
                p.normal_(0, 0.08)
    m.to(torch.bfloat16).eval().save_pretrained(md)


def write_bshj(path, tensors, meta):       # container layout as documented in mashbox/bshj.py (same as qwen3_8_27b_checks)
    align, pos, ents, bufs = 4096, 0, {}, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {"bfloat16": "bf16", "float32": "f32", "int8": "i8"}[a.dtype.name]
        pos = -(-pos // align) * align
        ents[name] = dict(dtype=dt, shape=list(a.shape), offset=pos, nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    raw = json.dumps(dict(meta, format="bshj", version="1.0", data_len=pos, tensors=ents), separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p_, b in bufs:
            f.seek(off + p_)
            f.write(b)
        f.truncate(off + pos)


def int8_bshj(md, path):
    from safetensors import safe_open
    import glob
    big = ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight",
           "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight")
    out = {}
    for f in sorted(glob.glob(os.path.join(md, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                t = s.get_tensor(k)
                v = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.float().numpy()
                if k.endswith(big) or k.endswith("embed_tokens.weight") or k == "lm_head.weight":
                    w32 = v.astype(np.float32)
                    am = np.abs(w32).max(-1)
                    sc = np.where(am > 0, am / np.float32(127.0), np.float32(1.0)).astype(np.float32)
                    out[k], out[k + "_scale"] = np.clip(np.round(w32 / sc[:, None]), -127, 127).astype(np.int8), sc
                else:
                    out[k] = v
    write_bshj(path, out, {"bshj": {"quant": {"type": "TEST"}}})


def kl(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    la = a - a.max(-1, keepdims=True); la -= np.log(np.exp(la).sum(-1, keepdims=True))
    lb = b - b.max(-1, keepdims=True); lb -= np.log(np.exp(lb).sum(-1, keepdims=True))
    return float((np.exp(la) * (la - lb)).sum(-1).mean())


COLL = ("all-gather", "all-reduce", "reduce-scatter", "collective-permute", "all-to-all", "collective-broadcast")


def collectives(compiled):
    """-> (main, in_branches): collective instructions in the optimised HLO, outside / inside conditional branches."""
    txt = compiled.as_text()
    comps, cur, branches = {}, None, set()
    for line in txt.splitlines():
        s = line.strip()
        if (s.startswith("%") or s.startswith("ENTRY")) and s.endswith("{") and "=" not in s.split("(")[0]:
            cur = s.split()[1 if s.startswith("ENTRY") else 0].lstrip("%")
            comps[cur] = []
        elif cur is not None and " = " in s:
            comps[cur].append(s)
            if " conditional(" in s:
                for part in s.split("branch_computations={")[1:] + s.split("true_computation=")[1:] + s.split("false_computation=")[1:]:
                    for nm in part.split("}")[0].split(","):
                        branches.add(nm.strip().lstrip("%").split(" ")[0].rstrip(","))

    def calls(c, seen):
        out = set()
        for s in comps.get(c, []):
            for key in ("calls=", "to_apply=", "body=", "condition="):
                if key in s:
                    out.add(s.split(key)[1].split(",")[0].split(" ")[0].lstrip("%").rstrip("}"))
        return out

    inside = set()
    todo = list(branches)
    while todo:
        c = todo.pop()
        if c in inside:
            continue
        inside.add(c)
        todo += list(calls(c, inside))
    count = lambda cs: sum(1 for c in cs for s in comps.get(c, []) for op in COLL
                           if f" {op}(" in s or f" {op}-start(" in s)
    return count(set(comps) - inside), count(inside)


t_all = time.time()
MD = os.path.join(WD, "tiny4")
BP = os.path.join(WD, "tiny4-int8.bshj")
if not os.path.exists(BP):
    build(MD)
    int8_bshj(MD, BP)
cfg = Q.Config(MD)
V = cfg.V
mesh = QS.mesh_for(4)

# TP1 ---------------------------------------------------------------------------------------------------------------------
if want("TP1"):
    ok, worst = True, []
    for s_, st in ((0, 0), (7, 3), (2 ** 63 + 5, 1000), (123456789, 2 ** 20)):
        seed = jnp.asarray(SP.pack([dict(seed=s_)], 1)["seed"][0])
        ref = jax.random.gumbel(jax.random.fold_in(jax.random.wrap_key_data(seed, impl="threefry2x32"), st), (V,), jnp.float32)
        got = SPT.noise(seed, jnp.int32(st), jnp.arange(V, dtype=jnp.int32))
        same = bool(np.array_equal(np.asarray(ref).view(np.uint32), np.asarray(got).view(np.uint32)))
        ok &= same
        worst.append(dict(seed=s_, step=st, bit_equal=same))
    emit("TP1", ok, cases=worst, threefry_partitionable=bool(jax.config.jax_threefry_partitionable))

# TP2 ---------------------------------------------------------------------------------------------------------------------
if want("TP2"):
    rng = np.random.default_rng(3)
    B = 16
    rows = []
    for kind in ("greedy", "temp", "k3", "k8", "k20", "p05", "p09", "p099", "min01", "min03", "k8p09", "p09min02", "k20p05min01",
                 "hot", "flat", "tie"):
        rows.append(kind)
    settings = dict(greedy=None, temp=dict(temperature=0.9), k3=dict(temperature=1.0, top_k=3), k8=dict(temperature=1.0, top_k=8),
                    k20=dict(temperature=0.7, top_k=20), p05=dict(temperature=1.0, top_p=0.5), p09=dict(temperature=1.0, top_p=0.9),
                    p099=dict(temperature=1.2, top_p=0.99), min01=dict(temperature=1.0, min_p=0.1), min03=dict(temperature=1.0, min_p=0.3),
                    k8p09=dict(temperature=0.8, top_k=8, top_p=0.9), p09min02=dict(temperature=1.0, top_p=0.9, min_p=0.2),
                    k20p05min01=dict(temperature=1.0, top_k=20, top_p=0.5, min_p=0.1), hot=dict(temperature=2.0, top_p=0.95),
                    flat=dict(temperature=1.0, top_k=5), tie=dict(temperature=1.0, top_k=2))
    res = {}
    for C in (8, 64, SPT.CANDIDATES):
        smp = SPT.Sampler(mesh, "tp", C)
        eq, n, fb, miss = 0, 0, 0, []
        for st in range(12):
            x = rng.normal(size=(B, V)).astype(np.float32) * 3
            x[rows.index("flat")] = rng.normal(size=V).astype(np.float32) * 0.01
            x[rows.index("tie"), :] = -5.0
            x[rows.index("tie"), [127, 128, 300]] = 4.0              # equal maxima on both sides of the shard boundary 127 | 128
            x[rows.index("greedy"), [255, 256]] = 9.0                # greedy tie across a boundary: the first index wins
            samp = [dict(settings[k] or {}, seed=1000 * i + 17) if settings[k] else None for i, k in enumerate(rows)]
            sp = SP.to_device(SP.pack(samp, B))
            stp = jnp.full((B,), st, jnp.int32)
            xs = jax.device_put(jnp.asarray(x), NamedSharding(mesh, P(None, "tp")))
            a = np.asarray(SP.sample(jnp.asarray(x), sp, stp))
            b = np.asarray(jax.jit(smp.sample)(xs, sp, stp))
            fb += int(np.asarray(jax.jit(smp.sample_rows)(xs, sp, stp)[1]).sum())
            eq += int((a == b).sum())
            n += B
            miss += [dict(step=st, row=rows[i], ref=int(a[i]), got=int(b[i])) for i in np.nonzero(a != b)[0]]
        res[C] = dict(equal=eq, draws=n, fallback_rows=fb, misses=miss[:8])
    emit("TP2", all(r["equal"] == r["draws"] for r in res.values()), by_candidates=res)

# TP3 ---------------------------------------------------------------------------------------------------------------------
if want("TP3"):
    smp8 = SPT.Sampler(mesh, "tp", 8)
    xs = jax.device_put(jnp.zeros((8, V), jnp.float32), NamedSharding(mesh, P(None, "tp")))
    sp8 = SP.to_device(SP.pack([dict(temperature=1.0, top_p=0.9, seed=1)] * 8, 8))
    main, br = collectives(jax.jit(smp8.sample).lower(xs, sp8, jnp.zeros(8, jnp.int32)).compile())
    rec3 = dict(sampler_main=main, sampler_fallback_branch=br)
    emit("TP3", main <= 2 and main + br <= 3, **rec3)

# engines -----------------------------------------------------------------------------------------------------------------
W1, i1 = QW.load(BP, cfg)
W4, i4 = QS.load(BP, cfg, mesh)
tpm = QS.TPModel(Q, mesh, candidates=8)          # 8 candidates of 128 per chip: the merge and its fallback both run


def engines(**kw):
    return Engine(cfg, W1, Q, **kw), Engine(cfg, W4, tpm, **kw)


prng = np.random.default_rng(1)
P_ = [prng.integers(1, V, n).tolist() for n in (12, 70, 33, 140, 5, 200)]


def run(e, reqs, joins=(), steps_max=400):
    """admit reqs[0..] now, the requests in joins (step index -> list) when that step comes; -> rows by request order."""
    out, pend, k = [], sorted(dict(joins).items()), 0
    out += e.admit(reqs)
    while e.busy() or pend:
        if pend and pend[0][0] <= k and e.free_slots() >= len(pend[0][1]):
            out += e.admit(pend.pop(0)[1])
        e.step()
        k += 1
        assert k < steps_max
    return [(r["ids"], r["fin"], r.get("cached", 0), r.get("restored", 0)) for r in out]


q = lambda p, n=16, **k: dict(ids=p, max_new=n, **k)
samp = [dict(temperature=0.8, top_p=0.9, seed=7), dict(temperature=1.0, top_k=5, seed=8), None,
        dict(temperature=1.0, min_p=0.1, repetition_penalty=1.3, presence_penalty=0.5, seed=9),
        dict(temperature=0.7, top_k=40, top_p=0.95, frequency_penalty=0.4, seed=10), dict(temperature=1.2, seed=11)]


# TP4 ---------------------------------------------------------------------------------------------------------------------
if want("TP4"):
    e1, e4 = engines(CTX=512, S=4, K=4, budget_busy=64, budget_idle=128)
    plan = ([q(P_[0]), q(P_[1])], {3: [q(P_[2])], 6: [q(P_[3], 12)]})
    g1, g4 = run(e1, *plan), run(e4, *plan)
    l1, l4 = e1.logits(P_[3]), e4.logits(P_[3])
    emit("TP4", g1 == g4 and kl(l1, l4) < 1e-4, equal_rows=[a == b for a, b in zip(g1, g4)], kl_vs_tp1=kl(l1, l4),
         tokens=sum(len(r[0]) for r in g1))

# TP5 ---------------------------------------------------------------------------------------------------------------------
if want("TP5"):
    e1, e4 = engines(CTX=512, S=4, K=4, budget_busy=64, budget_idle=128)
    plan5 = ([q(P_[i], 14, samp=samp[i]) for i in range(3)], {2: [q(P_[3], 14, samp=samp[3])], 5: [q(P_[4], 14, samp=samp[4])]})
    s1, s4, s4b = run(e1, *plan5), run(e4, *plan5), run(e4, *plan5)
    emit("TP5", s1 == s4 and s4 == s4b, equal_rows=[a == b for a, b in zip(s1, s4)], tp4_repeats=s4 == s4b,
         tokens=sum(len(r[0]) for r in s1))

# TP6 ---------------------------------------------------------------------------------------------------------------------
if want("TP6"):
    c1, c4 = engines(CTX=512, S=2, K=4, prefix_cache=True, cache_ram=64 << 20, budget_busy=64, budget_idle=128)
    convs = [P_[1], P_[3], P_[5]]                                   # 3 conversations, 2 slots: saves to and restores from host
    seq6 = [q(c_ + [7, 9 + ci], 8, scope="test-key") for ci, c_ in enumerate(convs)]
    seq6 += [q(r["ids"] + [11, 12, 13 + ci], 8, scope="test-key") for ci, r in enumerate(list(seq6))]     # each continues its turn
    o1 = [run(c1, [r])[0] for r in seq6]
    o4 = [run(c4, [r])[0] for r in seq6]
    f4 = Engine(cfg, W4, tpm, CTX=512, S=2, K=4)                     # no prefix cache: the last turn cold
    cold = run(f4, [dict(seq6[-1])])[0]
    emit("TP6", o1 == o4 and o4[-1][0] == cold[0] and c4.stats["restored_tokens"] > 0 and c4.stats == c1.stats,
         equal_turns=[a == b for a, b in zip(o1, o4)], stats_tp1=c1.stats, stats_tp4=c4.stats, cached_eq_cold=o4[-1][0] == cold[0])

# TP7 ---------------------------------------------------------------------------------------------------------------------
if want("TP7"):
    (K_, V_), (R_, C_) = c4.state
    kvd = sorted({s.data.shape[3] // cfg.HD for s in K_.addressable_shards})     # K/V heads flat in the cache row
    rvd = sorted({s.data.shape[3] for s in R_.addressable_shards})
    big1, big4 = engines(CTX=4096, S=8, K=1)
    st1, st4 = big1.new_state(), big4.new_state()
    args = lambda e, st: (e.W, st[0], st[1], jnp.zeros(8, jnp.int32), jnp.full(8, 100, jnp.int32))
    m1 = big1.dec.lower(*args(big1, st1), Tc=4096, K=1).compile().memory_analysis()
    m4 = big4.dec.lower(*args(big4, st4), Tc=4096, K=1).compile().memory_analysis()
    kv_all = sum(int(a.nbytes) for a in jax.tree.leaves(st4[0]))
    t1_, t4_ = int(m1.temp_size_in_bytes), int(m4.temp_size_in_bytes)
    emit("TP7", kvd == [cfg.KV // 4] and rvd == [cfg.HV // 4] and t4_ < kv_all // 2,
         kv_heads_per_chip=kvd, value_heads_per_chip=rvd, decode_temp_per_chip_tp4=t4_, decode_temp_tp1=t1_, kv_cache_all_chips=kv_all)
    main_d, _ = collectives(big4.dec.lower(*args(big4, st4), Tc=4096, K=1).compile())
    R["E29_info"] = dict(decode_greedy_collectives_per_pass=main_d, layers=cfg.L)
    print(json.dumps(dict(gate="E29_info", **R["E29_info"])), flush=True)
    del st1, st4, big1, big4

# TP8 ---------------------------------------------------------------------------------------------------------------------
if want("TP8"):
    w1, w4 = engines(CTX=256, S=4, K=4, prefix_cache=True, cache_ram=16 << 20)
    wr1, wr4 = w1.warm(workers=1), w4.warm(workers=1)
    seq8 = [[q(P_[0], 6)], [q(P_[1], 6, samp=samp[0]), q(P_[2], 6)], [q(P_[1] + [5, 6], 6, samp=samp[1])],
            [q(P_[4], 6), q(P_[5][:150], 6, samp=samp[3]), q(P_[0] + [3], 6)],
            [q(P_[2][:20], 6, samp=samp[4]), q(P_[3][:20], 6, samp=samp[5]), q(P_[5][:20], 6, samp=samp[0]), q(P_[1][:20], 6, samp=samp[1])],
            [q(P_[4] + [8], 6, samp=samp[5])]]
    cnt, drift = {}, []
    canon = [tpm.kv_sh, tpm.kv_sh, tpm.r_sh, tpm.c_sh]
    for name, e in (("tp1", w1), ("tp4", w4)):
        k0 = _COMPILES[0]
        for j, reqs in enumerate(seq8):
            run(e, reqs)
            if name == "tp4":       # the slot state is in the model's own placement after every request
                drift += [dict(request=j, leaf=i, got=str(a.sharding.spec)) for i, (a, sh) in enumerate(zip(jax.tree.leaves(e.state), canon))
                          if not a.sharding.is_equivalent_to(sh, a.ndim)]
        cnt[name] = _COMPILES[0] - k0
    emit("TP8", cnt["tp4"] == 0 and cnt["tp4"] <= cnt["tp1"] and wr4["compile_failed"] == 0 and not drift, compiles_after_warm=cnt,
         state_placement_drift=drift[:6], warm_tp1=wr1, warm_tp4=wr4)

# TP9 ---------------------------------------------------------------------------------------------------------------------
def ctx_reads(closed, ctx):
    """equations of a traced program (into scan / while / shard_map / jit bodies) that read a whole context axis: a gather
    whose operand carries it, or a slice / dynamic_slice / squeeze whose output carries it (a whole cache group copied
    before the read). Scatter and dynamic_update_slice into the cache are writes, not counted."""
    out = []

    def walk(jx):
        for eq in jx.eqns:
            p = eq.primitive.name
            ins = [tuple(getattr(v.aval, "shape", ())) for v in eq.invars]
            outs = [tuple(getattr(v.aval, "shape", ())) for v in eq.outvars]
            if (p == "gather" and ctx in ins[0]) or (p in ("slice", "dynamic_slice", "squeeze") and any(ctx in s for s in outs)):
                out.append(dict(op=p, operand=ins[0], out=outs[0]))
            for v in eq.params.values():
                for sub in (v if isinstance(v, (list, tuple)) else [v]):
                    if hasattr(sub, "eqns"):
                        walk(sub)
                    elif hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                        walk(sub.jaxpr)
    walk(closed.jaxpr)
    return out


if want("TP9"):
    # On a v6e-4 the 8-row decode step slowed 3.9x from ctx 4096 to 32768 (14.33 -> 42.78 ms; one row: 9.14 both): every
    # full-attention layer copied its whole cache group (all slots x CTX positions) before reading Tc of them. The decode and
    # prefill programs under tp 4 must read only the positions they attend.
    CT = 2304                       # a length no other axis of the tiny model has
    r1, r4 = engines(CTX=CT, S=8, K=1)
    st1, st4 = r1.new_state(), r4.new_state()
    found, info = {}, {}
    for B in (1, 8):
        a4 = (r4.W, st4[0], st4[1], jnp.zeros(B, jnp.int32), jnp.full(B, 100, jnp.int32))
        found[f"decode_B{B}"] = ctx_reads(r4.dec.trace(*a4, Tc=256, K=1).jaxpr, CT)
        a1 = (r1.W, st1[0], st1[1], jnp.zeros(B, jnp.int32), jnp.full(B, 100, jnp.int32))
        info[f"tp1_decode_B{B}"] = len(ctx_reads(r1.dec.trace(*a1, Tc=256, K=1).jaxpr, CT))
    pf = (r4.W, st4[0], st4[1], jnp.zeros((2, 32), jnp.int32), jnp.full(2, 256, jnp.int32), jnp.full(2, 32, jnp.int32),
          jnp.asarray([5, 2], jnp.int32))
    found["prefill_B2_T32"] = ctx_reads(r4.fwd.trace(*pf, Tc=256).jaxpr, CT)
    emit("TP9", not any(found.values()), whole_ctx_reads_tp4={k: len(v) for k, v in found.items()},
         examples={k: v[:3] for k, v in found.items() if v}, tp1_info_not_a_gate=info, ctx=CT)
    del st1, st4, r1, r4

print(json.dumps(dict(seconds=round(time.time() - t_all, 1), jax=jax.__version__, devices=len(jax.devices()))))
print("ALL_PASS" if all(G.values()) else "FAILED " + " ".join(k for k, v in G.items() if not v))
