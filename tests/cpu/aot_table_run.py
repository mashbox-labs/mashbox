"""One start of the stand-in engine as `mashbox run` starts (test_aot_table.py runs it twice, each in a fresh process):
load the executable table from TABLE when the engine has one, two short requests, warm(), save the table, then the same
fixed requests, greedy and sampled. Usage: aot_table_run.py TABLE OUT.json. Writes the compiles and the traces of the
engine's programs (by name) during the start and after it, and every answer's tokens."""
import json
import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import jax                                  # noqa: E402
import jax.numpy as jnp                     # noqa: E402

import engine_standin as M                  # noqa: E402
from mashbox.runtime import programs        # noqa: E402
from mashbox.runtime.engine import Engine   # noqa: E402

TABLE, OUT = sys.argv[1:3]
NAMES = {"fwd", "dec", "decs", "first_rows", "first", "gather_last", "put", "take", "kv_take", "kv_put", "swap"}
seen = {"start": {}, "after": {}}
traced = {"start": {}, "after": {}}
phase = ["start"]


def listen(ev, secs, **k):
    n = str(k.get("fun_name"))
    n = n[4:-1] if n.startswith("jit(") and n.endswith(")") else n     # compiles name "jit(fwd)", traces "fwd"
    if n in NAMES and ev in (programs.EVENT, "/jax/core/compile/jaxpr_trace_duration"):
        d = (seen if ev == programs.EVENT else traced)[phase[0]]
        d[n] = d.get(n, 0) + 1


jax.monitoring.register_event_duration_secs_listener(listen)
e = Engine(types.SimpleNamespace(V=M.V, H=2), {"w": jnp.ones((), jnp.float32)}, M.MODEL, CTX=256, S=2, K=4,
           prefix_cache=True, cache_ram=1 << 30)
has = hasattr(e, "aot")
meta = programs.table_meta("test-deployment") if has else None
load = e.aot.load(TABLE, meta, workers=4) if has else None


class Counted:
    """An executable of the table that records each run that returns (the program really executed), and is otherwise
    the executable itself (programs.proven reads its fields)."""
    def __init__(self, exe, k):
        self.exe, self.k = exe, k

    def __call__(self, *a):
        out = self.exe(*a)
        runs.append(self.k)
        return out

    def __getattr__(self, n):
        return getattr(self.exe, n)


SAMP = dict(temperature=0.7, top_p=0.9, top_k=20, presence_penalty=1.0, seed=3)
e.generate([[5, 6, 7]], max_new=5)
e.generate([[5, 6, 7]], max_new=5, samp=[SAMP])
runs = []
if has:     # during warm() every executable of the table, loaded or compiled there, counts its runs
    e.aot.ex.update({k: Counted(x, k) for k, x in e.aot.ex.items()})
    put = e.aot.put
    e.aot.put = lambda k, exe, name: put(k, Counted(exe, k), name)
w = e.warm(workers=4)
if has:
    e.aot.put = put
    e.aot.ex.update({k: x.exe for k, x in e.aot.ex.items() if isinstance(x, Counted)})
warm_runs = [k for k in runs if k[0] in ("fwd", "dec", "decs")]      # prefill / decode programs the warm-up executed
ran_in_warm = len(warm_runs)
ran_recorded = all(k in e.aot.ran for k in warm_runs) if has else None      # and the table's record (Table.ran) has each
save = e.aot.save(TABLE, meta) if has else None
phase[0] = "after"
ans = []
for i, L in enumerate((3, 17, 64, 100, 129, 200)):
    p = [(7 * j + L) % (M.V - 1) + 1 for j in range(L)]
    ans.append(e.generate([p], max_new=12)["ids"][0])
    ans.append(e.generate([p], max_new=12, samp=[dict(SAMP, seed=i)])["ids"][0])
ans.append(e.generate([[3] * 40, [9] * 70], max_new=10, samp=[None, SAMP])["ids"])
with open(OUT, "w") as f:
    json.dump(dict(has_table=has, load=load, save=save, warm=dict(w), compiles=seen, traces=traced, answers=ans,
                   prefill_shapes=e.prefill_shapes(), ran_in_warm=ran_in_warm,
                   ran_recorded=ran_recorded), f, default=str)
