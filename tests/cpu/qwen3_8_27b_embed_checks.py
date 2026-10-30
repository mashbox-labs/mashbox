"""CPU checks of the Qwen3.8-27B int8 embedding lookup (model.int8_rows): the tile-aligned form equals the plain gather
bit for bit, and a decode-sized lookup never gathers single rows of the full table (the gather that makes XLA relayout
the whole table on TPU every decode step).

  python3 tests/cpu/qwen3_8_27b_embed_checks.py <workdir>

Needs: jax (CPU), ml_dtypes; E3 also torch + transformers with qwen3_5 support (the tiny model of qwen3_8_27b_checks.py).
  E1 embed() == q[toks] * s[toks] (the plain gather) bit for bit: every row residue mod 32, the last row, shapes from
     one row to prefill size, with and without an activation-edit function
  E2 a lookup of at most EMBED_TILE_MAX_ROWS rows gathers whole 32-row tiles, never (1, D) rows of the (N, D) table
  E3 tiny int8 model: logits (prefill of 1..8 tokens and longer) and greedy decode tokens + per-token logit hashes with
     the tile form == with the plain gather (EMBED_TILE_MAX_ROWS = 0), bit for bit
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                     # noqa: E402
import jax                             # noqa: E402
import jax.numpy as jnp                # noqa: E402

from mashbox.models.qwen3_8_27b import model as QM   # noqa: E402

WD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mashbox-cpu-embed-test"
os.makedirs(WD, exist_ok=True)
R = {}
f32, bf16 = jnp.float32, jnp.bfloat16
bits = lambda x: np.asarray(x).view(np.uint16)


def plain(E, toks, pj=None):            # the lookup as written before int8_rows (reference)
    x = E["q"][toks].astype(f32) * E["s"][toks][..., None]
    return (x if pj is None else pj(x)).astype(bf16)


# E1
rng = np.random.default_rng(0)
N, D = 4096, 192
E = dict(q=jnp.asarray(rng.integers(-128, 128, (N, D)).astype(np.int8)),
         s=jnp.asarray(rng.uniform(1e-4, 3e-2, N).astype(np.float32)))
pj = lambda x: x * jnp.float32(1.5) + jnp.float32(0.25)
cases = [np.array([[r]]) for r in range(32)] + [np.array([[N - 1]]), np.array([[N - 32]]), np.array([[31 + 32 * 7]])]
cases += [rng.integers(0, N, s) for s in ((2, 1), (4, 1), (8, 1), (1, 4), (1, 8), (2, 4), (3, 5), (1, 40), (8, 64))]
cases += [np.arange(N).reshape(1, N)]
e1 = []
for t in cases:
    t = jnp.asarray(t, jnp.int32)
    for f in (None, pj):
        a, b = bits(jax.jit(lambda E, t: QM.embed(E, t, f))(E, t)), bits(jax.jit(lambda E, t: plain(E, t, f))(E, t))
        e1.append(dict(shape=list(t.shape), pj=f is not None, n_diff=int((a != b).sum())))
R["E1"] = dict(cases=len(e1), fails=[c for c in e1 if c["n_diff"]])
R["E1_pass"] = not R["E1"]["fails"]


# E2
def table_gathers(fn, toks):
    """(operand shape, slice sizes) of every gather / dynamic_slice in the jaxpr of fn(E, toks) that reads the table"""
    out = []

    def walk(jx):
        for e in jx.eqns:
            if e.primitive.name in ("gather", "dynamic_slice"):
                op = e.invars[0].aval
                if op.dtype == jnp.int8:
                    ss = e.params.get("slice_sizes") or e.params.get("dimension_numbers")
                    out.append((tuple(op.shape), tuple(e.params["slice_sizes"]) if "slice_sizes" in e.params else str(ss)))
            for p in e.params.values():
                if hasattr(p, "jaxpr"):
                    walk(p.jaxpr)
    walk(jax.make_jaxpr(fn)(E, toks).jaxpr)
    return out


e2 = []
for shp in ((1, 1), (2, 1), (8, 1), (1, 8)):
    g = table_gathers(lambda E, t: QM.embed(E, t), jnp.zeros(shp, jnp.int32))
    rows = [x for x in g if x[0] == (N, D)]                         # single rows of the full (N, D) table
    tiles = [x for x in g if len(x[0]) == 3 and x[1][-2:] == (getattr(QM, "EMBED_TILE", 32), D)]  # whole tiles
    e2.append(dict(shape=list(shp), gathers=[[list(a), list(b) if isinstance(b, tuple) else b] for a, b in g],
                   ok=bool(tiles) and not rows))
R["E2"] = e2
R["E2_pass"] = all(x["ok"] for x in e2)
print(json.dumps(dict(E1_pass=R["E1_pass"], E1_cases=R["E1"]["cases"], E2_pass=R["E2_pass"])), flush=True)


# E3: the tiny int8 model of qwen3_8_27b_checks.py (its fixture code, run up to the model build)
def e3():
    import importlib.util
    src = open(os.path.join(HERE, "qwen3_8_27b_checks.py")).read()
    pre = src[:src.index("\nm = build()")]                           # imports + fixtures only, no checks
    ns = dict(__name__="qwen3_8_27b_fixture", __file__=os.path.join(HERE, "qwen3_8_27b_checks.py"))
    sys.argv[1:2] = [WD]
    exec(compile(pre, "qwen3_8_27b_checks.py (fixture prefix)", "exec"), ns)
    md = os.path.join(WD, "tiny")
    ns["build"](md)
    T = ns["hf_tensors"](md)
    path = os.path.join(WD, "tiny-int8.bshj")
    BIG, rows8 = ns["BIG"], ns["int8_rows"]
    T8 = {}
    for k, v in T.items():               # as int8_tree in qwen3_8_27b_checks.py: big linears, embedding, lm_head per-row int8
        if k.endswith(BIG) or k.endswith("embed_tokens.weight") or k == "lm_head.weight":
            T8[k], T8[k + "_scale"] = rows8(v)
        else:
            T8[k] = v
    ns["write_bshj"](path, T8, {"bshj": {"quant": {"type": "TEST"}}})
    from mashbox.models import qwen3_8_27b as Q
    from mashbox.models.qwen3_8_27b import weights as QW
    from mashbox.runtime.engine import Engine
    cfg = Q.Config(md)
    W, _ = QW.load(path, cfg)
    assert isinstance(W["embed"], dict) and W["embed"]["q"].dtype == jnp.int8 and W["embed"]["q"].shape[0] % 32 == 0
    r = np.random.default_rng(1)
    seqs = [r.integers(1, cfg.V if hasattr(cfg, "V") else 512, n).tolist() for n in (1, 3, 8, 40)]
    seqs[0] = [511]
    prompts = [seqs[3][:12], seqs[3][:20], [5, 7, 480, 31, 32]]

    def run(max_rows):
        QM.EMBED_TILE_MAX_ROWS = max_rows
        eng = Engine(cfg, W, Q, CTX=256, S=4, K=4, trace=True)
        lg = [np.asarray(eng.logits(s), np.float32) for s in seqs]
        g1 = eng.generate([prompts[0]], max_new=24, stop_ids=())
        g3 = eng.generate(prompts, max_new=24, stop_ids=())
        return lg, [g1, g3]

    keep = getattr(QM, "EMBED_TILE_MAX_ROWS", None)
    try:
        lt, gt = run(keep)
        lp, gp = run(0)
    finally:
        QM.EMBED_TILE_MAX_ROWS = keep
    lg = [dict(n=len(s), max_abs=float(np.abs(a - b).max()), bit_equal=bool((a.view(np.uint32) == b.view(np.uint32)).all()))
          for s, a, b in zip(seqs, lt, lp)]
    gen = []
    for a, b in zip(gt, gp):
        ia, ib = [list(map(int, x)) for x in a["ids"]], [list(map(int, x)) for x in b["ids"]]
        ha, hb = a.get("lh"), b.get("lh")
        gen.append(dict(tokens_equal=ia == ib, hashes_present=ha is not None, hashes_equal=str(ha) == str(hb),
                        n_tokens=sum(len(x) for x in ia)))
    return dict(logits=lg, generate=gen)


try:
    R["E3"] = e3()
    R["E3_pass"] = (all(x["bit_equal"] and x["max_abs"] == 0.0 for x in R["E3"]["logits"])
                    and all(x["tokens_equal"] and x["hashes_present"] and x["hashes_equal"] for x in R["E3"]["generate"]))
except ImportError as x:
    R["E3"] = dict(skipped=repr(x))
    R["E3_pass"] = None
R["pass"] = bool(R["E1_pass"] and R["E2_pass"] and R["E3_pass"] is not False)
json.dump(R, open(os.path.join(WD, "embed_checks.json"), "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ("E1_pass", "E2_pass", "E3_pass", "pass")}), flush=True)
print(json.dumps(R.get("E3"), default=str)[:1500], flush=True)
sys.exit(0 if R["pass"] else 1)
