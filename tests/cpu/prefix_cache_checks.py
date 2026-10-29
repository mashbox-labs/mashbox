"""Prefix cache checks on the tiny Qwen3.8 model (CPU). Usage: prefix_cache_checks.py WD
WD = the work folder of a qwen3_8_27b_checks.py run (it holds tiny/ and tiny-bf16.bshj). Prints one JSON line, exit 0 on pass.

P1 snapshot and restore: prompt B prefilled after restoring the cached state of its first part A (with the slot state
   overwritten in between, as decoding does) gives the same last logits as B prefilled from scratch.
P2 generate: a second turn that resends turn 1 + the answer + new tokens is served from the cache (cached = turn 1
   prompt length); every token it decodes is the reference argmax within 0.05 logits (the T1 tolerance), with the
   reference = teacher-forced logits of an engine without the cache. A third turn that changes the last message is
   served from the mark (start of the last message).
P3 two conversations on two slots keep their own cache; a third, new prompt takes the least recently used slot.
P4 host-memory prompt cache (llama-server --cache-ram): on one slot, conversation A, then an unrelated prompt B takes the
   slot, then A's next turn is restored from host memory (cached = restored = A's turn-1 length) and decodes exactly
   the tokens of an engine that served A's two turns back to back; B comes back restored too. With cache_ram=0 the
   same sequence re-prefills A (cached 0).
P5 user-message checkpoints (llama-server PR 24176): over many turns the point at the first user message (the end of the
   tools and system prompt) is kept within the per-slot limit, and a compacted conversation (same start, new history)
   resumes there; its tokens are the reference argmax within 0.05 logits.
P6 a row ended by on_step (client gone, stop string) stops at that step; the other row decodes the reference tokens and the
   ended conversation's next turn is served from the cache.
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

# P4 ------------------------------------------------------------------------------------------------------------------
h = Engine(cfg, W, Q, CTX=512, S=1, K=4, prefix_cache=True)
u1 = Engine(cfg, W, Q, CTX=512, S=1, K=4, prefix_cache=True)
off = Engine(cfg, W, Q, CTX=512, S=1, K=4, prefix_cache=True, cache_ram=0)
a1 = rng.integers(1, 512, 140).tolist()
b1 = rng.integers(1, 512, 90).tolist()
ra1 = h.generate([a1], max_new=12, marks=[[60, 120]])
rb1 = h.generate([b1], max_new=6, marks=[[50, 80]])
a2 = a1 + ra1["ids"][0] + rng.integers(1, 512, 40).tolist()
ra2 = h.generate([a2], max_new=16, marks=[[60, 120, 152]])
q1 = u1.generate([a1], max_new=12, marks=[[60, 120]])
q2 = u1.generate([a2], max_new=16, marks=[[60, 120, 152]])
o = [off.generate([x], max_new=6) for x in (a1, b1, a2)]
b2 = b1 + rb1["ids"][0] + rng.integers(1, 512, 9).tolist()
rb2 = h.generate([b2], max_new=4)
R["P4"] = dict(cached=ra2["cached"][0], restored=ra2["restored"][0], expect=len(a1), same_tokens_as_uninterrupted=ra2["ids"][0] == q2["ids"][0],
               uninterrupted_cached=q2["cached"][0], off_cached=o[2]["cached"][0], b_restored=rb2["restored"][0], saves=h.stats["saves"],
               host_entries=len(h.host.items), host_mb=round(h.host.nbytes() / 2 ** 20, 2))
# three conversations taking turns on one slot (several saved entries at once), each comes back restored
cs = [rng.integers(1, 512, 60 + 10 * k).tolist() for k in range(3)]
for c_ in cs:
    h.generate([c_], max_new=4)
back = [h.generate([c_ + [7, 8, 9]], max_new=4)["restored"][0] for c_ in cs]
R["P4"]["three_rotating_restored"] = back
R["P4_pass"] = back == [len(c_) for c_ in cs] and (ra2["cached"][0] == ra2["restored"][0] == len(a1) == q2["cached"][0] and ra1["ids"][0] == q1["ids"][0]
                and ra2["ids"][0] == q2["ids"][0] and o[2]["cached"][0] == 0 and rb2["restored"][0] == len(b1) and h.stats["saves"] >= 2)

# P5 ------------------------------------------------------------------------------------------------------------------
cp = Engine(cfg, W, Q, CTX=512, S=1, K=4, prefix_cache=True, ckpt_step=64)
sysp = rng.integers(1, 512, 100).tolist()
conv, msgs = sysp + rng.integers(1, 512, 20).tolist(), [100]
kept = []
for turn in range(9):
    r_ = cp.generate([conv], max_new=4, marks=[msgs + [len(conv) - 2]])     # user starts, then a last-message point
    kept.append(sorted(len(x["ids"]) for x in cp.snaps[0]))
    msgs = msgs + [len(conv) + len(r_["ids"][0])]
    conv = conv + r_["ids"][0] + rng.integers(1, 512, 25).tolist()
comp = sysp + rng.integers(1, 512, 30).tolist()
rc_ = cp.generate([comp], max_new=8, marks=[[100, len(comp) - 2]])
R["P5"] = dict(points_per_turn=kept, last_prompt=len(conv), compacted_cached=rc_["cached"][0], expect=100,
               max_gap=max(gaps(ref, comp, rc_["ids"][0])))
R["P5_pass"] = (rc_["cached"][0] == 100 and R["P5"]["max_gap"] < 0.05 and all(100 in k and len(k) <= cp.ckpts for k in kept))
# P6 ------------------------------------------------------------------------------------------------------------------
# a row that on_step ends (client gone, stop string found) stops at that step; the other row of the batch decodes the
# reference tokens, and the ended conversation's next turn is still served from the cache
cx = Engine(cfg, W, Q, CTX=512, S=2, K=4, prefix_cache=True)
v = [rng.integers(1, 512, 80).tolist(), rng.integers(1, 512, 70).tolist()]


def end_row0(ids, fin):
    return [0] if len(ids[0]) >= 5 else []


rx = cx.generate(v, max_new=40, on_step=end_row0)
v2 = v[0] + rx["ids"][0] + rng.integers(1, 512, 12).tolist()
rx2 = cx.generate([v2], max_new=8)
R["P6"] = dict(finish=rx["finish"], row0_tokens=len(rx["ids"][0]), row1_tokens=len(rx["ids"][1]),
               row1_max_gap=max(gaps(ref, v[1], rx["ids"][1])), next_cached=rx2["cached"][0], expect=len(v[0]),
               next_max_gap=max(gaps(ref, v2, rx2["ids"][0])))
R["P6_pass"] = (rx["finish"][0] == "cancel" and len(rx["ids"][0]) == 5 and rx["finish"][1] in ("stop", "length")
                and R["P6"]["row1_max_gap"] < 0.05 and rx2["cached"][0] == len(v[0]) and R["P6"]["next_max_gap"] < 0.05)
R["ALL_PASS"] = all(v for k, v in R.items() if k.endswith("_pass"))
print(json.dumps(R))
sys.exit(0 if R["ALL_PASS"] else 1)
