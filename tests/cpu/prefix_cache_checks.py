"""Prefix cache checks on the tiny Qwen3.8 model (CPU). Usage: prefix_cache_checks.py WD
WD = the work folder of a qwen3_8_27b_checks.py run (it holds tiny/ and tiny-bf16.bshj). Prints one JSON line, exit 0 on pass.

P1 snapshot and restore: prompt B prefilled after restoring the cached state of its first part A (with the slot state
   overwritten in between, as decoding does) gives the same last logits as B prefilled from scratch.
P2 generate: a second turn that resends turn 1 + the answer + new tokens is served from the cache (cached = turn 1
   prompt length); every token it decodes is the reference argmax within 0.05 logits (the T1 tolerance), with the
   reference = teacher-forced logits of an engine without the cache. A third turn that changes the last message is
   served from the mark (start of the last message).
P3 two conversations on two slots keep their own cache; a third, new prompt takes the least recently used slot.
"""
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

from mashbox.models import qwen3_8_27b as Q
from mashbox.models.qwen3_8_27b import weights as QW
from mashbox.runtime.engine import Engine

WD = sys.argv[1]
cfg = Q.Config(os.path.join(WD, "tiny"))
W, _ = QW.load(os.path.join(WD, "tiny-bf16.bshj"), cfg)
rng = np.random.default_rng(7)
R = {}


def gaps(e, prompt, ids):
    """reference logit gap of each decoded token: max logit - logit of the token (0 = the reference argmax)"""
    lg = e.logits(list(prompt) + list(ids))
    return [float(lg[len(prompt) - 1 + j].max() - lg[len(prompt) - 1 + j][t]) for j, t in enumerate(ids)]

# P1 ------------------------------------------------------------------------------------------------------------------
eng = Engine(cfg, W, Q, CTX=512, S=2, K=4)
a = rng.integers(1, 512, 150).tolist()
b = a + rng.integers(1, 512, 70).tolist()
_, _, la = eng.prefill(eng.new_state(), [b])
st, _, _ = eng.prefill(eng.new_state(), [a])
snap = eng._snap(st, 0, a)
st, _, _ = eng.prefill(st, [a + rng.integers(1, 512, 40).tolist()], [0], [len(a)])     # overwrite the slot state past A
st = (st[0], eng.put(st[1], 0, jnp.asarray(snap["r"]), jnp.asarray(snap["c"])))
_, _, lb = eng.prefill(st, [b], [0], [len(a)])
x, y = np.asarray(la[0]), np.asarray(lb[0])
R["P1_max_abs_diff"] = float(np.abs(x - y).max())
R["P1_logit_scale"] = float(np.abs(x).max())
R["P1_argmax_same"] = bool(x.argmax() == y.argmax())
R["P1_pass"] = R["P1_argmax_same"] and R["P1_max_abs_diff"] <= 0.02 * R["P1_logit_scale"]

# P2 ------------------------------------------------------------------------------------------------------------------
ref = Engine(cfg, W, Q, CTX=512, S=2, K=4)
pc = Engine(cfg, W, Q, CTX=512, S=2, K=4, prefix_cache=True)
t1 = rng.integers(1, 512, 120).tolist()
r1 = pc.generate([t1], max_new=16, marks=[100])
t2 = t1 + r1["ids"][0] + rng.integers(1, 512, 30).tolist()
r2 = pc.generate([t2], max_new=16, marks=[len(t1) + len(r1["ids"][0])])
g2 = ref.generate([t2], max_new=16)
R["P2_cached"] = r2["cached"][0]
R["P2_expect_cached"] = len(t1)
R["P2_same_tokens"] = int(sum(p == q for p, q in zip(r2["ids"][0], g2["ids"][0])))
R["P2_max_gap"] = max(gaps(ref, t2, r2["ids"][0]))
R["P2_pass"] = r2["cached"][0] == len(t1) and R["P2_max_gap"] < 0.05 and r1["cached"][0] == 0
# a third turn that changes the last message: served from the mark (start of the last message)
t3 = t2[:len(t1) + len(r1["ids"][0])] + rng.integers(1, 512, 25).tolist()
r3 = pc.generate([t3], max_new=8)
R["P2_mark_cached"] = r3["cached"][0]
R["P2_mark_max_gap"] = max(gaps(ref, t3, r3["ids"][0]))
R["P2_pass"] = R["P2_pass"] and r3["cached"][0] == len(t1) + len(r1["ids"][0]) and R["P2_mark_max_gap"] < 0.05

# P3 ------------------------------------------------------------------------------------------------------------------
u = [rng.integers(1, 512, 90).tolist(), rng.integers(1, 512, 60).tolist()]
ra = pc.generate(u, max_new=6)
u2 = [u[i] + ra["ids"][i] + rng.integers(1, 512, 10).tolist() for i in range(2)]
rb = pc.generate(u2[::-1], max_new=6)            # order swapped: each must still find its own slot
R["P3_cached"] = rb["cached"]
R["P3_slots"] = [ra["slots"], rb["slots"]]
R["P3_max_gap"] = max(max(gaps(ref, p, i)) for p, i in zip(u2[::-1], rb["ids"]))
R["P3_pass"] = rb["cached"] == [len(u[1]), len(u[0])] and R["P3_max_gap"] < 0.05 and rb["slots"] == ra["slots"][::-1]
rc = pc.generate([rng.integers(1, 512, 40).tolist()], max_new=4)
R["P3_new_prompt"] = dict(cached=rc["cached"][0], slot=rc["slots"][0])
R["P3_pass"] = R["P3_pass"] and rc["cached"][0] == 0
R["stats"] = pc.stats
R["ALL_PASS"] = all(v for k, v in R.items() if k.endswith("_pass"))
print(json.dumps(R))
sys.exit(0 if R["ALL_PASS"] else 1)
