"""Single-user greedy determinism (certified-base gate B6, skill mashbox-engine-batching section 13.13), the CPU form on the
tiny Qwen3.8 model. Usage: determinism_checks.py WD   (WD = the work folder of a qwen3_8_27b_checks.py run: tiny/ and
tiny-bf16.bshj). Prints one JSON line; exit 0 when every gate passes. Keys ending in _info are reported, never gates.

The engine runs at the defaults of `mashbox run` (context, slots, steps per call, prefix cache on, host cache size,
context checkpoints: read from cli.build_parser, not copied) with trace=True (MASHBOX_TRACE=1). One user, alone, greedy:
8 chat-shaped prompts of 24 to 1500 tokens (cache points at each message start, as the server's marks_of gives), each run
on its own, max_new tokens each.

D1  three runs of the 8 prompts on one engine give identical tokens; runs 2 and 3 are served from the user's own prefix
    cache (cached > 0 for every prompt), so a repeat that hits its own cache gives the cold run's tokens.
D2  a run on a new engine from the same weights (a restart) gives run 1's tokens.
D3  the host-cache path: between the cold run and the repeat of each prompt, 8 other prompts of the same user take every
    slot, so the repeat is restored from host memory (restored > 0) and gives the cold run's tokens.
D_hash_info per-token logit hashes of every repeat against run 1: a hash that differs while the tokens are equal is a
    margin (the step and its top-1 / top-2 gap are listed), reported and never hidden.
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                                          # noqa: E402

from mashbox import cli                                     # noqa: E402
from mashbox.models import qwen3_8_27b as Q                 # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW        # noqa: E402
from mashbox.runtime.engine import Engine                   # noqa: E402

WD = sys.argv[1]
MAX_NEW = int(os.environ.get("D_MAX_NEW", "256"))
cfg = Q.Config(os.path.join(WD, "tiny"))
W, _ = QW.load(os.path.join(WD, "tiny-bf16.bshj"), cfg)
a = cli.build_parser().parse_args(["run", "qwen3-8-27b"])
DEFAULTS = dict(CTX=a.ctx, S=a.slots, K=a.steps, prefix_cache=True, cache_ram=max(0, a.cache_ram) << 20,
                ckpts=a.ctx_checkpoints, ckpt_step=a.checkpoint_min_step, trace=True)
rng = np.random.default_rng(61)


def chat_prompt(n):
    """n token ids in 2-4 'messages'; -> (ids, marks): the cache points at each message start after the first."""
    cuts = sorted(set(int(x) for x in rng.integers(1, n, int(rng.integers(1, 4))))) if n > 4 else []
    return rng.integers(1, cfg.V, n).tolist(), cuts


P = [chat_prompt(n) for n in (24, 90, 200, 450, 640, 900, 1200, 1500)]
assert all(len(p) + MAX_NEW <= DEFAULTS["CTX"] for p, _ in P)
FILLER = [chat_prompt(n) for n in (60, 70, 80, 90, 100, 110, 120, 130)]


def one(e, pm):
    p, m = pm
    r = e.generate([p], max_new=MAX_NEW, marks=[m])
    return dict(ids=r["ids"][0], cached=r["cached"][0], restored=r["restored"][0], lh=r["lh"][0])


def margins(ref, x):
    """steps where the tokens agree but the logit hash differs: (step, top-1/top-2 gap of the reference)"""
    return [(k, round(g0, 4)) for k, ((h0, g0), (h1, _)) in enumerate(zip(ref["lh"], x["lh"]))
            if h0 != h1 and ref["ids"][k] == x["ids"][k]]


R, G = dict(defaults={k: v for k, v in DEFAULTS.items()}, max_new=MAX_NEW, lens=[len(p) for p, _ in P]), {}

# D1: three runs on one engine --------------------------------------------------------------------------------------
e = Engine(cfg, W, Q, **DEFAULTS)
runs = [[one(e, pm) for pm in P] for _ in range(3)]
same = [[runs[k][i]["ids"] == runs[0][i]["ids"] for i in range(len(P))] for k in (1, 2)]
R["D1"] = dict(equal=same, cached=[[x["cached"] for x in run] for run in runs],
               lens_out=[len(x["ids"]) for x in runs[0]])
G["D1"] = (all(all(s) for s in same) and all(x["cached"] == 0 for x in runs[0])
           and all(x["cached"] > 0 for run in runs[1:] for x in run)
           and all(len(x["ids"]) == MAX_NEW for x in runs[0]))

# D2: a restart -----------------------------------------------------------------------------------------------------
e2 = Engine(cfg, W, Q, **DEFAULTS)
restart = [one(e2, pm) for pm in P]
R["D2"] = dict(equal=[restart[i]["ids"] == runs[0][i]["ids"] for i in range(len(P))],
               cached=[x["cached"] for x in restart])
G["D2"] = all(R["D2"]["equal"]) and all(x["cached"] == 0 for x in restart)

# D3: the repeat restored from host memory ------------------------------------------------------------------------
e3 = Engine(cfg, W, Q, **DEFAULTS)
cold, warm_ = [], []
for pm in P:
    cold.append(one(e3, pm))
    for f in FILLER:
        one(e3, (f[0], f[1]))
    warm_.append(one(e3, pm))
R["D3"] = dict(equal=[warm_[i]["ids"] == cold[i]["ids"] for i in range(len(P))],
               restored=[x["restored"] for x in warm_], cached=[x["cached"] for x in warm_],
               equal_run1=[cold[i]["ids"] == runs[0][i]["ids"] for i in range(len(P))])
G["D3"] = (all(R["D3"]["equal"]) and all(R["D3"]["equal_run1"]) and all(x["restored"] > 0 for x in warm_))

# logit hashes ------------------------------------------------------------------------------------------------------
mg = {"run2": [margins(runs[0][i], runs[1][i]) for i in range(len(P))],
      "run3": [margins(runs[0][i], runs[2][i]) for i in range(len(P))],
      "restart": [margins(runs[0][i], restart[i]) for i in range(len(P))],
      "host_restore": [margins(cold[i], warm_[i]) for i in range(len(P))]}
R["D_hash_info"] = dict(margin_steps={k: sum(len(x) for x in v) for k, v in mg.items()},
                        first={k: next(([i] + x[0:1] for i, x in enumerate(v) if x), None) for k, v in mg.items()},
                        tokens_checked=sum(len(x["ids"]) for x in runs[0]) * 4)
R["gates"] = G
print(json.dumps(R))
sys.exit(0 if all(G.values()) else 1)
