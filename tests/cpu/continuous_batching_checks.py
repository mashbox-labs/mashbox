"""Continuous batching checks on the tiny Qwen3.8 model (CPU). Usage: continuous_batching_checks.py WD
WD = the work folder of a qwen3_8_27b_checks.py run (it holds tiny/ and tiny-bf16.bshj). Prints one JSON line, exit 0 on pass.

Batched output may differ from the same request run alone at a near tie (vLLM precedent),
so C1-C5 never gate a batched row against an alone run: that comparison
is recorded as *_alone_equal (with the alone top-1/top-2 logit gap where the tokens first differ), information only, and
the identity itself is C6's, the strict expected failure of the opt-in batch-invariant mode. What C1-C5 gate, per schedule:
  replay     the same arrival schedule in a fresh engine gives every row the same tokens, the same slot and the same finish;
  rows       every row keeps its own rules: it ends by itself (fin "length" exactly at its max_new, or "stop" before it and
             only with a stop token set), no stop token in its answer, every token in the vocabulary, a slot, an end time;
  state      after the schedule no row is live, every slot is free, nothing waits in the admission order, and the engine's
             usage counters count each request and its prompt tokens once;
  isolation  (C1, C2, C4, C5) for each row, the same schedule with every OTHER row's prompt replaced by different tokens of the
             same length (same max_new and settings, so the same batch shapes at every step) gives that row the same tokens
             and slot. The model is dense, so a row's numbers depend only on its own inputs and the batch shapes: a row that
             reads another row's tokens, cache or state reads false here. This is not alone vs batched (the batch
             composition is the same in both runs).

C1 greedy: a request that joins while another row is mid-answer (admit between two step() calls), and the row it joined.
C2 sampled with a seed (temperature, top_k, top_p) joining mid-answer next to a greedy row and a penalised row
   (repetition/frequency/presence); also gated: the seeded sampled answer differs from the greedy one (both alone).
C3 cancel: a row ended mid-answer ends at once (fin "cancel") and frees its slot; a new request takes the slot; the other live
   row AND the new row decode the same tokens as a replay of the same schedule in a fresh engine and as the schedule where
   the cancelled row ends by itself at that step (the cancel changes nothing for them). Their comparisons with the same
   requests run alone are recorded (alone_equal, new_alone_equal), information only.
C4 prefix cache: the second turn of a conversation admitted while another row is live is served from the cache (cached =
   turn 1 prompt length, gated, also in the replay); its tokens vs an engine that served the two turns back to back are
   recorded (serial_equal), information only.
C5 per-row max_new and stop: rows of one admit end at their own limits; a stop token ends a row without the token (alone).
C6 server: 6 requests on 2 slots, sent 0.05 s apart, each answers exactly the tokens it decodes ALONE (spec section 1: same
   answer batched or alone); a late request gets its first token before the first request ends (it did not wait for the
   batch in flight), and the slots never hold more than 2 rows. Also reported (information, not the gate): equality with the
   same row decoded in decode bucket 2, and the solo top-1/top-2 logit gap where a row first differs. KNOWN FAIL: one row
   of 6 flips at a near tie (gap 0.0078) because greedy rows decode on buckets 1/4/S and the bucket changes the rounding;
   the fix is batch-invariant decode (open item), not a weaker test.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                     # noqa: E402

from mashbox.models import qwen3_8_27b as Q                  # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW        # noqa: E402
from mashbox.runtime.engine import Engine                   # noqa: E402

WD = sys.argv[1]
cfg = Q.Config(os.path.join(WD, "tiny"))
W, _ = QW.load(os.path.join(WD, "tiny-bf16.bshj"), cfg)
rng = np.random.default_rng(11)
R = {}
P = [rng.integers(1, 512, n).tolist() for n in (30, 57, 41, 90)]
STEP_CAP = 4000          # a schedule that is still busy after this many steps (a row that never ends, a slot never freed) stops


def run_live(eng, plan, stop_ids=()):
    """plan: list of (step index at which to admit, request dict); requests of the same step go in one admit call.
    -> list of rows (one per request, plan order)."""
    rows, k = [None] * len(plan), 0
    while k <= STEP_CAP:
        now = [i for i, (at, q) in enumerate(plan) if at == k and rows[i] is None]
        if now:
            for i, r in zip(now, eng.admit([plan[i][1] for i in now], stop_ids)):
                rows[i] = r
        if not eng.busy() and all(r is not None for r in rows):
            break
        eng.step()
        k += 1
    return [r if r is not None else dict(slot=None, ids=[], fin=None) for r in rows]


def drain(eng):
    n = 0
    while eng.busy() and n < STEP_CAP:
        eng.step()
        n += 1


def row_ok(r, q, stop_ids=()):
    """The row's own rules: it ended by itself ("length" exactly at max_new, or "stop" before it with a stop token set), no
    stop token in its answer, every token in the vocabulary, a slot, an end time."""
    ids, mx, fin = r["ids"], q["max_new"], r.get("fin")
    return bool(r.get("slot") is not None and fin in ("length", "stop") and len(ids) <= mx
                and (fin != "length" or len(ids) == mx) and (fin != "stop" or (len(stop_ids) > 0 and len(ids) < mx))
                and not set(ids) & set(stop_ids) and all(0 <= t < cfg.V for t in ids) and r.get("t_end") is not None)


def idle(eng, s0, reqs):
    """Usage and state after a schedule: no live row, every slot free, nothing in the admission order, no slot marked
    decoding, and the usage counters count each request and its prompt tokens once."""
    return bool(not eng.busy() and eng.free_slots() == eng.S and not eng.order and not np.any(eng.act)
                and eng.stats["requests"] - s0["requests"] == len(reqs)
                and eng.stats["prompt_tokens"] - s0["prompt_tokens"] == sum(len(q["ids"]) for q in reqs))


def schedule(make, plan, stop_ids=()):
    """make() -> an engine (fresh, with any history the check needs); run plan on it. -> dict(rows, state_ok)."""
    eng = make()
    s0 = dict(eng.stats)
    rows = run_live(eng, plan, stop_ids)
    return dict(rows=rows, state_ok=idle(eng, s0, [q for _, q in plan]))


def other_tokens(ids):
    """Different tokens of the same length (the isolation control's stand-in for another row's prompt)."""
    return [(t * 7 + 13) % 511 + 1 for t in ids]


def gated(make, plan, stop_ids=(), vary=None):
    """Run plan twice in fresh engines (the run and its replay), and with isolation, once per row with every other row's
    request replaced by vary(i, q). -> (the run's rows, the replay's rows, the gate record). Gated: replay_equal, slots_equal, fins_equal,
    rows_ok, state_ok, isolation_equal."""
    a, b = schedule(make, plan, stop_ids), schedule(make, plan, stop_ids)
    ra, rb = a["rows"], b["rows"]
    g = dict(replay_equal=[r["ids"] for r in ra] == [r["ids"] for r in rb],
             slots_equal=[r["slot"] for r in ra] == [r["slot"] for r in rb],
             fins_equal=[r["fin"] for r in ra] == [r["fin"] for r in rb],
             rows_ok=[row_ok(r, q, stop_ids) for r, (_, q) in zip(ra, plan)], state_ok=a["state_ok"] and b["state_ok"],
             slots=[r["slot"] for r in ra], fins=[r["fin"] for r in ra], lens=[len(r["ids"]) for r in ra])
    if vary is not None:
        iso = []
        for k in range(len(plan)):
            alt = [(at, q if i == k else vary(i, q)) for i, (at, q) in enumerate(plan)]
            c = schedule(make, alt, stop_ids)["rows"][k]
            iso.append(c["ids"] == ra[k]["ids"] and c["slot"] == ra[k]["slot"] and c["fin"] == ra[k]["fin"])
        g["isolation_equal"] = iso
    return ra, rb, g


def gate_pass(g):
    return bool(g["replay_equal"] and g["slots_equal"] and g["fins_equal"] and all(g["rows_ok"]) and g["state_ok"]
                and all(g.get("isolation_equal", [True])))


def alone_info(p, a, b, samp=None):
    """Information only: does batched answer a equal alone answer b, and if not, the alone top-1/top-2 gap where they first
    differ (greedy rows; a sampled row's gap is not a tie measure)."""
    if a == b:
        return dict(equal=True)
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    gap = None
    if samp is None and k < min(len(a), len(b)):
        lg = np.sort(Engine(cfg, W, Q, CTX=512, S=2, K=4).logits(list(p) + list(b[:k + 1]))[len(p) + k - 1])
        gap = float(lg[-1] - lg[-2])
    return dict(equal=False, first_diff=k, alone_gap=gap)


def vary_ids(i, q):
    return dict(q, ids=other_tokens(q["ids"]))


# C1 ------------------------------------------------------------------------------------------------------------------
eng = Engine(cfg, W, Q, CTX=512, S=4, K=4)
solo = [eng.generate([p], max_new=24)["ids"][0] for p in P[:2]]
plan1 = [(0, dict(ids=P[0], max_new=24)), (2, dict(ids=P[1], max_new=24))]
rows, _, g = gated(lambda: Engine(cfg, W, Q, CTX=512, S=4, K=4), plan1, vary=vary_ids)
R["C1"] = dict(g, joined_alone_equal=alone_info(P[1], rows[1]["ids"], solo[1]),
               first_alone_equal=alone_info(P[0], rows[0]["ids"], solo[0]))
R["C1_pass"] = gate_pass(g) and len(set(g["slots"])) == 2

# C2 ------------------------------------------------------------------------------------------------------------------
sp = dict(temperature=0.9, top_k=40, top_p=0.9, seed=1234)
pen = dict(temperature=0.0, repetition_penalty=1.3, frequency_penalty=0.5, presence_penalty=0.4, seed=5)
a_s = eng.generate([P[2]], max_new=20, samp=[sp])["ids"][0]
a_p = eng.generate([P[3]], max_new=20, samp=[pen])["ids"][0]
a_g = eng.generate([P[0]], max_new=28)["ids"][0]
plan2 = [(0, dict(ids=P[0], max_new=28)), (1, dict(ids=P[2], max_new=20, samp=sp)), (3, dict(ids=P[3], max_new=20, samp=pen))]
rows, _, g = gated(lambda: Engine(cfg, W, Q, CTX=512, S=4, K=4), plan2, vary=vary_ids)
R["C2"] = dict(g, sampled_differs_from_greedy=a_s != eng.generate([P[2]], max_new=20)["ids"][0],
               sampled_alone_equal=alone_info(P[2], rows[1]["ids"], a_s, samp=sp),
               penalised_alone_equal=alone_info(P[3], rows[2]["ids"], a_p, samp=pen),
               greedy_alone_equal=alone_info(P[0], rows[0]["ids"], a_g))
R["C2_pass"] = gate_pass(g) and R["C2"]["sampled_differs_from_greedy"]


# C3 ------------------------------------------------------------------------------------------------------------------
# A cancel ends its row at once and frees the slot, the next request takes that slot, and the other live row and the new row
# go on as if nothing had happened: each equals the same schedule replayed in a fresh engine, and the schedule where r0 ends
# by itself at that step instead of by cancel. Neither is compared with the same request run alone in the gate (batched
# output may differ from running alone at a near tie); those comparisons stay in the record as
# alone_equal / new_alone_equal, and the identity is the opt-in batch-invariant mode's (C6).
def c3_schedule(r0_max_new=40, cancel=True):
    e = Engine(cfg, W, Q, CTX=512, S=2, K=4)
    alone1 = e.generate([P[1]], max_new=30)["ids"][0]
    alone2 = e.generate([P[2]], max_new=12)["ids"][0]
    s0 = dict(e.stats)
    r0 = e.admit([dict(ids=P[0], max_new=r0_max_new)])[0]
    r1 = e.admit([dict(ids=P[1], max_new=30)])[0]
    e.step()
    gone = e.cancel([r0]) if cancel else []
    free_after = e.free_slots()
    # a cancel that frees no slot leaves no room for the new request: it is not admitted (took_slot and the new row's
    # gates read false) and C4-C6 still report, instead of the schedule stopping on admit's error
    r2 = e.admit([dict(ids=P[2], max_new=12)])[0] if free_after else dict(slot=None, ids=[], fin=None)
    drain(e)
    reqs = [dict(ids=P[0]), dict(ids=P[1])] + ([dict(ids=P[2])] if free_after else [])
    return dict(alone1=alone1, alone2=alone2, r0=r0, r1=r1, r2=r2, gone=gone, free_after=free_after,
                state_ok=idle(e, s0, reqs))


c3 = c3_schedule()
c3_replay = c3_schedule()
c3_natural = c3_schedule(r0_max_new=max(1, len(c3["r0"]["ids"])), cancel=False)
R["C3"] = dict(cancelled=[g["fin"] for g in c3["gone"]], free_after=c3["free_after"],
               took_slot=c3["r2"]["slot"] == c3["r0"]["slot"],
               cancel_len=len(c3["r0"]["ids"]), replay_equal=c3_replay["r1"]["ids"] == c3["r1"]["ids"],
               natural_end_equal=c3_natural["r1"]["ids"] == c3["r1"]["ids"],
               natural_end_fin=c3_natural["r0"].get("fin"), natural_end_len=len(c3_natural["r0"]["ids"]),
               new_replay_equal=c3_replay["r2"]["ids"] == c3["r2"]["ids"],
               new_natural_equal=c3_natural["r2"]["ids"] == c3["r2"]["ids"],
               slots_equal=[c3[k]["slot"] for k in ("r0", "r1", "r2")] == [c3_replay[k]["slot"] for k in ("r0", "r1", "r2")],
               rows_ok=[row_ok(c3["r1"], dict(max_new=30)), row_ok(c3["r2"], dict(max_new=12))],
               state_ok=c3["state_ok"] and c3_replay["state_ok"] and c3_natural["state_ok"],
               alone_equal=alone_info(P[1], c3["r1"]["ids"], c3["alone1"]),
               new_alone_equal=alone_info(P[2], c3["r2"]["ids"], c3["alone2"]))
R["C3_pass"] = (R["C3"]["cancelled"] == ["cancel"] and R["C3"]["free_after"] == 1 and R["C3"]["took_slot"]
                and R["C3"]["cancel_len"] < 40 and R["C3"]["replay_equal"]
                and R["C3"]["natural_end_equal"] and R["C3"]["natural_end_len"] == R["C3"]["cancel_len"]
                and R["C3"]["new_replay_equal"] and R["C3"]["new_natural_equal"] and R["C3"]["slots_equal"]
                and all(R["C3"]["rows_ok"]) and R["C3"]["state_ok"])

# C4 ------------------------------------------------------------------------------------------------------------------
t1 = P[1]
ref = Engine(cfg, W, Q, CTX=512, S=2, K=4, prefix_cache=True)
a1 = ref.generate([t1], max_new=10)["ids"][0]
t2 = t1 + a1 + rng.integers(1, 512, 25).tolist()
b_ref = ref.generate([t2], max_new=10)


def pc_engine():
    """an engine that served turn 1 of the conversation (its prefix cache holds it)"""
    e = Engine(cfg, W, Q, CTX=512, S=2, K=4, prefix_cache=True)
    e.generate([t1], max_new=10)
    return e


def vary4(i, q):
    """isolation stand-ins: the live other row gets other tokens; turn 2 keeps its cached conversation (t1 + answer) and
    gets other tokens only in its new user part, so its cache hit stays the same"""
    return dict(q, ids=other_tokens(q["ids"])) if i == 0 else dict(q, ids=t2[:len(t1) + len(a1)] + other_tokens(t2[len(t1) + len(a1):]))


plan4 = [(0, dict(ids=P[3], max_new=40)), (1, dict(ids=t2, max_new=10))]
rows, again, g = gated(pc_engine, plan4, vary=vary4)
R["C4"] = dict(g, ref_cached=b_ref["cached"][0], live_cached=rows[1]["cached"], replay_cached=again[1]["cached"],
               serial_equal=alone_info(t2, rows[1]["ids"], b_ref["ids"][0]), other_slot=rows[0]["slot"],
               turn2_slot=rows[1]["slot"])
R["C4_pass"] = (R["C4"]["live_cached"] == len(t1) and R["C4"]["ref_cached"] == len(t1)
                and R["C4"]["replay_cached"] == len(t1) and gate_pass(g))

# C5 ------------------------------------------------------------------------------------------------------------------
e5 = Engine(cfg, W, Q, CTX=512, S=4, K=4)
full = e5.generate([P[0]], max_new=20)["ids"][0]
stop_tok = full[9]
plan5 = [(0, dict(ids=P[0], max_new=5)), (0, dict(ids=P[0], max_new=20)), (0, dict(ids=P[1], max_new=11))]
rows, _, g = gated(lambda: Engine(cfg, W, Q, CTX=512, S=4, K=4), plan5, vary=vary_ids)
rs = e5.admit([dict(ids=P[0], max_new=20)], stop_ids=(stop_tok,))[0]
drain(e5)
k_stop = full.index(stop_tok)
R["C5"] = dict(g, short_alone_equal=alone_info(P[0], rows[0]["ids"], full[:5]),
               stop_ids=rs["ids"] == full[:k_stop], stop_fin=rs["fin"],
               stop_row_ok=row_ok(rs, dict(max_new=20), (stop_tok,)))
R["C5_pass"] = (g["lens"] == [5, 20, 11] and g["fins"] == ["length"] * 3 and gate_pass(g)
                and R["C5"]["stop_ids"] and rs["fin"] == "stop" and R["C5"]["stop_row_ok"])

print(json.dumps(R, default=str), file=sys.stderr, flush=True)
# C6 ------------------------------------------------------------------------------------------------------------------
from mashbox.runtime.server import Server, serve   # noqa: E402


class IdChat:
    """Token ids as text ('12 7 301'): the server path without a tokenizer."""
    stop_ids = ()

    def encode(self, text):
        return [int(x) for x in text.split() if x.isdigit()]

    def decode(self, ids):
        return "".join(f"{int(t)} " for t in ids)

    def chat_ids(self, msgs, *a, **k):
        return self.encode(msgs[-1]["content"])


e6 = Engine(cfg, W, Q, CTX=512, S=2, K=4)
srv = Server(e6, IdChat(), "tiny", {}, log=lambda m: None)
srv.ready = True
port = 18000 + os.getpid() % 1000
threading.Thread(target=serve, args=(srv, "127.0.0.1", port, lambda m: None), daemon=True).start()
time.sleep(0.5)
prompts6 = [P[i % 4][:20 + 3 * i] for i in range(6)]
solo6 = [Engine(cfg, W, Q, CTX=512, S=2, K=4).generate([p], max_new=120)["ids"][0] for p in prompts6]


def bucket2(p):
    """p decoded in slot 1 next to a live row in slot 0 (decode bucket 2)."""
    e = Engine(cfg, W, Q, CTX=512, S=2, K=4)
    e.admit([dict(ids=P[3], max_new=400)])
    r = e.admit([dict(ids=p, max_new=120)])[0]
    while r["fin"] is None:
        e.step()
    return r["ids"]
res, times = [None] * 6, [None] * 6
peak = [0]


def watch():
    while not all(r is not None for r in res):
        peak[0] = max(peak[0], sum(r is not None for r in e6.rows))
        time.sleep(0.002)


def call(i):
    body = dict(messages=[dict(role="user", content=" ".join(map(str, prompts6[i])))], max_tokens=120, stream=True)
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    t0, first, text = time.monotonic(), None, ""
    try:
        f = urllib.request.urlopen(rq, timeout=600)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:400], file=sys.stderr, flush=True)
        raise
    with f:
        for line in f:
            line = line.decode()
            if line.startswith("data: {"):
                d = (json.loads(line[6:])["choices"][0].get("delta") or {}).get("content") or ""
                if d and first is None:
                    first = time.monotonic()
                text += d
    times[i] = (t0, first, time.monotonic())
    res[i] = [int(x) for x in text.split()]


wt = threading.Thread(target=watch, daemon=True)
wt.start()
th = []
for i in range(6):
    th.append(threading.Thread(target=call, args=(i,)))
    th[-1].start()
    time.sleep(0.05)
[t.join() for t in th]
same = [a == b or a == bucket2(p) for a, b, p in zip(res, solo6, prompts6)]
alone = [a == b for a, b in zip(res, solo6)]


def first_gap(p, a, b):
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if k is None:
        return None
    lg = np.sort(Engine(cfg, W, Q, CTX=512, S=2, K=4).logits(list(p) + list(b[:k + 1]))[len(p) + k - 1])
    return dict(token=k, gap=float(lg[-1] - lg[-2]))


R["C6"] = dict(equal_alone=alone, equal_same_bucket_info=same, first_diff_info=[first_gap(p, a, b) for p, a, b in zip(prompts6, res, solo6)],
               peak_rows=peak[0],
               late_first_before_first_end=times[1][1] < times[0][2], first_end=round(times[0][2] - times[0][0], 3),
               late_ttft=round(times[1][1] - times[1][0], 3))
R["C6_pass"] = all(alone) and peak[0] <= 2 and R["C6"]["late_first_before_first_end"]

R["ALL_PASS"] = all(v for k, v in R.items() if k.endswith("_pass"))
print(json.dumps(R, default=str))
sys.exit(0 if R["ALL_PASS"] else 1)
