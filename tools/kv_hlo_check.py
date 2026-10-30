"""Offline check of how the engine's programs hold the K/V cache on a TPU, with no chip: each program is compiled for a TPU
v6e by XLA's TPU compiler from libtpu on the host (a deviceless AOT compile, jax.experimental.topologies), and the
optimized HLO and the compile's own memory analysis are read.

Per program it reports:
  temp_gib        the HLO temporaries XLA asks for (memory_analysis().temp_size_in_bytes), the number an OOM names;
  cache_copies    copy instructions whose result is a whole slot-state array of 1 MiB or more (the K/V cache, a
                  sliding-window ring, the linear-attention state; any layout), counted in the optimized HLO, inside and
                  outside loops;
  cache_layouts   every layout the whole cache takes in the program;
  aliased         the cache arguments are aliased to the cache outputs (input_output_alias), so the donated buffer is
                  updated in place;
  pool_scatters   scatter / gather instructions (fused computations included) whose result or operand has the shape of
  pool_gathers    a whole slot-state array (the linear-attention pools R and C, a cache): a gather / scatter on the pool
                  instead of a slice and an in-place dynamic_update_slice of the call's rows;
  bytes_accessed  XLA's cost analysis of the program (cost_analysis()['bytes accessed']), GiB.

A CPU compile does not show any of this (TPU tiles the two minor dims), so
this needs libtpu installed (`pip install "jax[tpu]==0.11.2"`); it never needs a TPU.

The model files are tiny random GGUF files written by tools/gguf_fixtures with a real model's attention shapes (layer
count, head counts, head sizes) and a small embedding / FFN / vocabulary: the cache is the real model's cache at the
stated slots x context; the big linears are BF16 (held as bfloat16 arrays), so the temporaries measured are the
cache's.

    python tools/kv_hlo_check.py --model g14 --slots 8 --ctx 2048 [--programs dec1,decS,decs,pre1,preS] [--json out.json]
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# attention shapes of real catalog models (from each file's GGUF metadata); D, F and the vocabulary are small on purpose
MODELS = {
    # unsloth/gemma-4-12b-it-GGUF: 48 layers, sliding layers 16 q heads / 8 K/V heads of 256, full layers 4 K/V heads of 512
    # (every sixth), so the flat K/V row is 2,048 wide
    "g14": ("gemma4", dict(layers=48, hp=dict(D=256, F=256, H=16, swa=[True] * 5 + [False], kv=[8] * 5 + [4], hd_swa=256,
                                               hd=512, window=1024))),
    # Qwen3-8B (unsloth/Qwen3-8B-GGUF): 36 layers, 32 q heads, 8 K/V heads of 128
    "qwen3-8b": ("qwen3", dict(layers=36, hp=dict(D=256, F=256, H=32, KV=8, HD=128))),
    # Llama-3.2-3B: 28 layers, 24 q heads, 8 K/V heads of 128
    "llama-3b": ("llama", dict(layers=28, hp=dict(D=256, F=256, H=24, KV=8, HD=128))),
    # Qwen2.5-0.5B: 24 layers, 14 q heads, 2 K/V heads of 64
    "qwen25-05b": ("qwen2", dict(layers=24, hp=dict(D=256, F=256, H=14, KV=2, HD=64))),
    # Qwen3.5-9B (qwen35 hybrid): 32 layers (every 4th full attention), 16 q heads, 4 K/V heads of 256
    "qwen35-9b": ("qwen35", dict(layers=32, hp=dict(D=256, F=256, H=16, KV=4, HD=256, RD=64, HK=2, HV=4, DK=64))),
    # unsloth/Qwen3.5-4B-GGUF: 32 layers (every 4th full attention: 16 q heads, 4 K/V heads of 256); linear attention
    # 16 key heads, 32 value heads of 128, so the recurrent pool is (8, 3, slots + 1, 32, 128, 128) float32
    "qwen35-4b": ("qwen35", dict(layers=32, hp=dict(D=256, F=256, H=16, KV=4, HD=256, RD=64, HK=16, HV=32, DK=128))),
}

# the MASHBOX Qwen3.8-27B build (models/qwen3_8_27b, BQ8_R int8 weights) at its real size: the weight tree weights.load
# makes, as abstract arrays (nothing is allocated), from the model's own config.json text fields
Q38 = dict(model_type="qwen3_5_text", num_hidden_layers=64, hidden_size=5120, intermediate_size=17408,
           num_attention_heads=24, num_key_value_heads=4, head_dim=256, vocab_size=248320, linear_num_key_heads=16,
           linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
           full_attention_interval=4, tie_word_embeddings=False, rms_norm_eps=1e-6, rope_parameters=dict(rope_theta=1e7,
           partial_rotary_factor=0.25, rope_type="default"))
MODELS["q38-27b"] = ("bshj", {})

PROGRAMS = ("dec1", "decS", "decs", "pre1", "preS")
SPEC = "specS"      # speculation verify (n-gram drafter, 3 drafts, S rows, greedy): the engine's spec step program


def q38_weights(c, sh):
    """weights.load's device tree for config c, as ShapeDtypeStructs on sharding sh (int8 + per-row f32 scale)"""
    import jax
    import jax.numpy as jnp
    S = lambda shape, dt: jax.ShapeDtypeStruct(tuple(shape), dt, sharding=sh)
    big = lambda lead, rows, cols: dict(q=S(lead + (rows, cols), jnp.int8), s=S(lead + (rows,), jnp.float32))
    NG, FI, D, F = c.NG, c.FI, c.D, c.F
    Gl, Gf, Gm = (NG, FI - 1), (NG,), (NG, FI)
    VD, CONV = c.HV * c.DV, c.CONV
    per = lambda n, rows, cols: [big((NG,), rows, cols) for _ in range(n)]
    W = dict(g=dict(
        lin=dict(qkv=per(FI - 1, CONV, D), z=per(FI - 1, VD, D), a=big(Gl, c.HV, D), b=big(Gl, c.HV, D),
                 out=per(FI - 1, D, VD), conv=S(Gl + (CONV, c.CK), jnp.bfloat16), A=S(Gl + (c.HV,), jnp.float32),
                 dt=S(Gl + (c.HV,), jnp.float32), nw=S(Gl + (c.DV,), jnp.float32)),
        full=dict(wq=big(Gf, 2 * c.H * c.HD, D), wk=big(Gf, c.KV * c.HD, D), wv=big(Gf, c.KV * c.HD, D),
                  wo=big(Gf, D, c.H * c.HD), qn=S(Gf + (c.HD,), jnp.float32), kn=S(Gf + (c.HD,), jnp.float32)),
        mlp=dict(ln1=S(Gm + (D,), jnp.float32), ln2=S(Gm + (D,), jnp.float32), wg=per(FI, F, D), wu=per(FI, F, D),
                 wd=per(FI, D, F))))
    W["embed"] = big((), c.V, D)
    W["norm"] = S((D,), jnp.float32)
    W["head"] = big((), c.V, D)
    return W


def _fixtures():
    spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_model(name, d):
    arch, kw = MODELS[name]
    fx = _fixtures()
    p = os.path.join(d, f"{name}.gguf")
    q = ["BF16"]            # big linears as BF16: ops.dense16 multiplies them as stored, nothing is rebuilt in the program
    if arch == "gemma4":
        fx.write_gemma4(p, layers=kw["layers"], hp=kw["hp"], quant=q)
    elif arch == "qwen35":
        fx.write_qwen35(p, layers=kw["layers"], hp=kw["hp"], quant=q)
    else:
        fx.write_dense(p, arch=arch, layers=kw["layers"], hp=kw["hp"], quant=q)
    return p


def topology_device():
    from jax.experimental import topologies
    topo = topologies.get_topology_desc(platform="tpu", topology_name="v6e:1x1", chip_config_name="default",
                                        chips_per_host_bounds=(1, 1, 1), num_slices=1)
    return topo.devices[0]


def cache_shapes(state):
    """the shapes of the slot state's arrays of 1 MiB or more: the K/V cache, a sliding-window ring (gemma4) and the
    linear-attention state (qwen35, Qwen3.8-27B)"""
    import jax
    return {tuple(a.shape) for a in jax.tree.leaves(state) if int(a.size) * a.dtype.itemsize >= 1 << 20}


def analyse(compiled, shapes):
    """-> dict of the numbers above from one compiled program; shapes = the whole cache arrays' shapes"""
    m = compiled.memory_analysis()
    txt = compiled.as_text()
    dims = {",".join(map(str, s)) for s in shapes}
    copies, layouts = 0, set()
    pat = re.compile(r"=\s*\w+\[([0-9,]+)\]\{([^}]*)\}\s+copy\(")
    for line in txt.splitlines():
        mm = pat.search(line)
        if mm and mm.group(1) in dims:
            copies += 1
        for d_ in dims:
            for lay in re.findall(r"\w+\[" + re.escape(d_) + r"\]\{([^}]*)\}", line):
                layouts.add(lay)
    head = txt.split("\n", 1)[0]
    alias = re.search(r"input_output_alias=\{(.*?)\}\s*,\s*entry_computation_layout", head)
    shp = re.compile(r"\w+\[([0-9,]+)\]")
    pool = {"scatter": 0, "gather": 0}
    for line in txt.splitlines():
        for op in pool:
            if re.search(r"\s" + op + r"\(", line) and dims & set(shp.findall(line)):
                pool[op] += 1
    try:
        ca = compiled.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        ba = round(float(ca.get("bytes accessed", 0.0)) / 2 ** 30, 3)
    except Exception:                   # a backend without a cost model
        ba = None
    return dict(temp_gib=round(m.temp_size_in_bytes / 2 ** 30, 3), arg_gib=round(m.argument_size_in_bytes / 2 ** 30, 3),
                out_gib=round(m.output_size_in_bytes / 2 ** 30, 3), alias_gib=round(m.alias_size_in_bytes / 2 ** 30, 3),
                cache_copies=copies, cache_layouts=sorted(layouts), aliased=bool(alias and alias.group(1).strip()),
                pool_scatters=pool["scatter"], pool_gathers=pool["gather"], bytes_accessed=ba)


def build(name, S, CTX, K=8, d=None, spec=False):
    """-> (engine, abstract weights, abstract cache) for model `name` at S slots x CTX positions (spec: with the n-gram
    drafter, 3 drafts)"""
    import jax
    import jax.numpy as jnp
    from mashbox.gguf import open_model
    from mashbox.runtime.engine import Engine
    dev = topology_device()
    sh = jax.sharding.SingleDeviceSharding(dev)
    ab = lambda t: jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sh), t)
    if MODELS[name][0] == "bshj":
        from mashbox.models import qwen3_8_27b as A
        json.dump(dict(text_config=Q38), open(os.path.join(d, "config.json"), "w"))
        c = A.Config(d)
        W = q38_weights(c, sh)
        eng = Engine(c, W, A, CTX=CTX, S=S, K=K)
    else:
        path = write_model(name, d)
        f, A, c = open_model(path, act=jnp.bfloat16)
        # BF16 linears as bfloat16 device arrays (ops.dense16), small F32 tensors as float32: no weight is rebuilt from
        # blocks inside the program, so the temporaries measured are the cache's (no kernel package is loaded here)
        W, _ = A.load(f, c, use_kernels=True)
        kw = {}
        if spec:
            from mashbox.runtime.drafters.ngram import NGram
            kw = dict(draft_k=3, drafter=NGram(3))
        eng = Engine(c, W, A, CTX=CTX, S=S, K=K, **kw)
        W = ab(W)
    st = jax.eval_shape(lambda: A.new_state(c, S - 1 if S <= 2 else S, CTX))
    return eng, W, ab(st), sh


def run(name, S=8, CTX=2048, K=8, programs=PROGRAMS, log=print):
    import jax
    import jax.numpy as jnp
    with tempfile.TemporaryDirectory() as d:
        eng, W, (ck, cv), sh = build(name, S, CTX, K, d, spec=SPEC in programs)
        shapes = cache_shapes((ck, cv))
        i32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.int32, sharding=sh)
        gib = lambda t: round(sum(int(a.size) * a.dtype.itemsize for a in jax.tree.leaves(t)) / 2 ** 30, 3)
        out = dict(model=name, slots=S, ctx=CTX, K=K, cache=[list(s) for s in sorted(shapes)], cache_gib=gib(ck),
                   state2_gib=gib(cv), programs={})
        for p in programs:
            t0 = time.time()
            if p in ("dec1", "decS"):
                B = 1 if p == "dec1" else S
                low = eng.dec.jitted.lower(W, ck, cv, i32(B), i32(B), Tc=CTX, K=K)
            elif p == "decs":
                from mashbox.runtime import sampler as SP
                sp = jax.eval_shape(lambda: SP.to_device(SP.pack([dict(temperature=0.7, top_p=0.9, top_k=20, seed=1)] * S, S)))
                sp = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sh), sp)
                V = eng.c.V
                low = eng.decs.jitted.lower(W, ck, cv, i32(S), i32(S), sp, i32(S),
                                            jax.ShapeDtypeStruct((S, V), jnp.float32, sharding=sh),
                                            jax.ShapeDtypeStruct((S, V), jnp.bool_, sharding=sh), Tc=CTX, K=K)
            elif p == SPEC:
                ab = lambda t: jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sh), t)
                z = i32(S)
                low = eng.sstep.lower(W, ck, cv, None, ab(eng.dstate), (), z, z, z, z, None, None, None, None,
                                      Tc=CTX, K=3, N=eng.NR, trace=False)
            elif p in ("pre1", "preS"):
                B, T = (1, 512) if p == "pre1" else (S, 128)
                low = eng.fwd.jitted.lower(W, ck, cv, i32(B, T), i32(B), i32(B), i32(B), Tc=CTX)
            else:
                raise ValueError(p)
            try:
                r = analyse(low.compile(), shapes)
            except Exception as e:          # an OOM at compile is a result: the message names the temporaries
                msg = str(e).split("\n", 1)[0]
                g = re.search(r"HLO temporaries \(([0-9.]+)G\)", msg)
                r = dict(error=msg[:300], temp_gib=float(g.group(1)) if g else None)
            r["compile_s"] = round(time.time() - t0, 1)
            out["programs"][p] = r
            log(json.dumps({p: r}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--programs", default=",".join(PROGRAMS), help=f"also {SPEC} (not for gemma4: no verify)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")    # a deviceless compile holds no chip: several may run at once
    out = run(a.model, a.slots, a.ctx, a.steps, tuple(x for x in a.programs.split(",") if x))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
