"""Engine batching checks (skill mashbox-engine-batching, edge cases E2-E33) on the tiny Qwen3.8 model (CPU).
Usage: engine_batch_checks.py WD   (WD = the work folder of a qwen3_8_27b_checks.py run: tiny/ and tiny-bf16.bshj)
Prints one JSON line; exit 0 when every gate passes. Lines marked _info are reported, never gates.

E2  a long prompt joins while a row decodes: the decoding row gets a token at every step of the join, the prompt is read
    one chunk per step inside the busy token budget, and chunks cut by the budget end on the 64-token grid.
E4  per-row finish: stop token, max_new and cancel end only their own row at the step the engine learns of them.
E18 a row finishing inside a K-step call: same tokens at K=1 and K=8 (engine); same text and completion_tokens for a stop
    string through the server at K=1 and K=8 (usage at the finish point).
E6  a request longer than the context gets 400 at once and nothing else is affected.
E7  the waiting count returns to 0 after requests ran through the continuous-batching loop.
E7_owner / E7_lqd / E7_hangup (amended spec): with every slot busy, owner A submits 300 and B 5: A is refused after 32
    (429), B never; the legacy streaming route answers 429 itself; with the overall limit full an owner under its share
    gets in by dropping the newest waiting request of the longest queue, an owner over its share gets 429 (Anthropic
    529; A1); 256 waiting
    clients that hang up free their count within 0.5 s.
E_connect 256 clients connecting at the same moment all get their answer (accept queue).
E8  three bad rows (top_k = 2^31, a NaN temperature sent as the raw JSON literal, a token id past the vocabulary) sent
    together with three good rows are each refused alone with 400; the good rows answer as they answer alone; two valid
    requests with different prompt lengths and max_tokens run together; numeric fields fuzzed at int32 / int64 / float
    edges, as raw JSON literals (Infinity, -Infinity, NaN, 1e999) and as quoted strings, get 400 or 200, never a 500,
    and a non-finite literal always gets 400.
E14 no compile at serve time: after the start-up warm-up of `mashbox run` (cli.warm), requests at log-spaced prompt
    lengths 1..ctx with 1..S rows, greedy and sampled, compile nothing (JAX backend compile counter: every jit program
    and eager op). E14_cache: the same with the prefix cache and host cache on and trace=True (MASHBOX_TRACE=1), over
    conversation turns of 3 tenants that resume from their own points, are displaced and restored from host memory, and
    run 1-3 rows at once, greedy and sampled mixed: nothing compiles either, and the turns did hit, save and restore.
E19 cancel during prefill frees the row before its next chunk; the slot is taken by the next request at once.
E26 padding rows (free slots inside a decode bucket) and a row paused mid-prefill: their K/V and linear state are unchanged
    by a step they are not part of (checksums of every slot the live rows do not write).
E33 "merge": false with max_tokens = room changes nothing for others: B's first token arrives before A ends, and B's p95
    first-token time with the field is within 2x of its p95 without it (4 runs per arm in ABBA order). The measured runs
    start only after a with/without pair compiled nothing (compile counter), and the gate also requires 0 compiles in
    the measured runs, so a serve-time compile (E14) can neither fail nor pass this check. The server's engine is warmed
    as `mashbox run` warms it (cli.warm) before any run: where B joins depends on how far A got in 0.3 s (box speed),
    and without the warm-up a join at a window no earlier pair visited compiled inside the measured runs. The single
    pair measured right after one warm-up run (the earlier form of this check) is reported as E33_first_pair_info with
    its compiles.
ID_info identity (spec section 1, gate E25 is on chip): per-step logit hashes of rows batched vs alone, and cached-prefill
    budgets 64 vs 1024; the first differing step is reported with its top-1/top-2 gap (known FAIL class: bucket rounding).
"""
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                     # noqa: E402
import jax                             # noqa: E402

from mashbox.models import qwen3_8_27b as Q                  # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW        # noqa: E402
from mashbox.runtime.engine import Engine                   # noqa: E402
from mashbox.runtime.server import Server                   # noqa: E402
from jax import monitoring                                  # noqa: E402
import api_stub_engine as STUB                              # noqa: E402

_COMPILES = [0]
monitoring.register_event_duration_secs_listener(
    lambda ev, d, **k: _COMPILES.__setitem__(0, _COMPILES[0] + (ev == "/jax/core/compile/backend_compile_duration")))


def compiles():
    """XLA backend compiles so far in this process (every jit program and eager op compiled)."""
    return _COMPILES[0]

WD = sys.argv[1]
cfg = Q.Config(os.path.join(WD, "tiny"))
W, _ = QW.load(os.path.join(WD, "tiny-bf16.bshj"), cfg)
rng = np.random.default_rng(23)
R, G = {}, {}
P = [rng.integers(1, 512, n).tolist() for n in (30, 57, 41, 90, 300, 260)]


def E(**k):
    a = dict(CTX=1024, S=4, K=4, chunk=512, budget_busy=64, budget_idle=1024)
    a.update(k)
    return Engine(cfg, W, Q, **a)


def solo(p, n=24, **k):
    return E(**k).generate([p], max_new=n)["ids"][0]


def digest(state, s_):
    (Kc, Vc), (Rr, Cc) = state
    h = hashlib.sha256()
    for a, ax in ((Kc, 1), (Vc, 1), (Rr, 2), (Cc, 2)):
        h.update(np.asarray(jax.numpy.take(a, s_, axis=ax)).tobytes())
    return h.hexdigest()[:16]


# E2 ------------------------------------------------------------------------------------------------------------------
e = E()
e.shapes = []
a = e.admit([dict(ids=P[0], max_new=200)])[0]
while not a["ids"]:
    e.step()
b = e.admit([dict(ids=P[4], max_new=8)])[0]
gaps, chunks = [], []
while b["t_first"] is None:
    n0, c0 = len(a["ids"]), b["cur"]
    e.step()
    gaps.append(len(a["ids"]) - n0)
    chunks.append((c0, b["cur"]))
sizes = [y - x for x, y in chunks]
R["E2"] = dict(steps=len(chunks), tokens_per_step=sorted(set(gaps)), chunk_sizes=sorted(set(sizes)),
               ends=[y for _, y in chunks])
G["E2"] = (all(g_ == 1 for g_ in gaps) and max(sizes) <= 64 and len(chunks) == -(-len(P[4]) // 64)
           and all(y % 64 == 0 or y == len(P[4]) for _, y in chunks))

# E4 ------------------------------------------------------------------------------------------------------------------
ref = solo(P[1], 30)
stop_tok = ref[5]
e = E()
rows = e.admit([dict(ids=P[1], max_new=30)], stop_ids=[stop_tok]) + e.admit([dict(ids=P[2], max_new=7),
                                                                              dict(ids=P[3], max_new=30)])
k = 0
while e.busy():
    e.step()
    k += 1
    if k == 3 and rows[2]["fin"] is None:
        e.cancel([rows[2]])
        free_after_cancel = e.free_slots()
R["E4"] = dict(fin=[r["fin"] for r in rows], lens=[len(r["ids"]) for r in rows], free_after_cancel=free_after_cancel)
G["E4"] = (rows[0]["fin"] == "stop" and rows[0]["ids"] == ref[:ref.index(stop_tok)] and rows[1]["fin"] == "length"
           and len(rows[1]["ids"]) == 7 and rows[2]["fin"] == "cancel" and free_after_cancel >= 1)

# E18 (engine) --------------------------------------------------------------------------------------------------------
k1 = E(K=1).generate([P[1], P[2]], max_new=19)["ids"]
k8 = E(K=8).generate([P[1], P[2]], max_new=19)["ids"]
R["E18_engine"] = dict(equal=k1 == k8, lens=[len(x) for x in k8])
G["E18_engine"] = k1 == k8 and all(len(x) == 19 for x in k8)


# server helpers ------------------------------------------------------------------------------------------------------
class IdChat:
    """Token ids as text ('12 7 301'): the server path without a tokenizer."""
    stop_ids = ()

    def encode(self, text):
        return [int(x) for x in text.split() if x.lstrip("-").isdigit()]

    def decode(self, ids):
        return "".join(f"{int(t)} " for t in ids)

    def chat_ids(self, msgs, *a, **k):
        return self.encode(msgs[-1]["content"])


def server(eng):
    srv = Server(eng, IdChat(), "tiny", {}, log=lambda m: None)
    srv.ready = True
    port = STUB.serve_on_free_port(srv)  # port 0: the system's free port (a fixed port can hold another server: 404s)
    return srv, port


def hold_engine(srv):
    """Keep every slot busy for the whole check (the check's precondition): the server loop's next engine step waits for
    the returned event. Without it a fast host finishes the holding request early and the queue drains (the legacy
    request then got 200)."""
    gate, step = threading.Event(), srv.engine.step

    def held(*a, **k):
        gate.wait()
        return step(*a, **k)
    srv.engine.step = held
    return gate


def post(port, body, path="/v1/completions", stream=False):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(dict(body, stream=stream)).encode(),
                                {"Content-Type": "application/json"})
    t0, first = time.monotonic(), None
    try:
        with urllib.request.urlopen(rq, timeout=600) as f:
            if not stream:
                return 200, json.loads(f.read()), (t0, None, time.monotonic())
            text = ""
            for line in f:
                line = line.decode()
                if line.startswith("data: {"):
                    d = json.loads(line[6:])["choices"][0].get("text") or ""
                    if d and first is None:
                        first = time.monotonic()
                    text += d
            return 200, text, (t0, first, time.monotonic())
    except urllib.error.HTTPError as x:
        return x.code, x.read().decode()[:300], (t0, None, time.monotonic())


def prompt(p):
    return " ".join(map(str, p))


# E18 (server, stop string) -------------------------------------------------------------------------------------------
ref = solo(P[3], 40)
stop_s = f"{ref[6]} {ref[7]} "
out = {}
for K in (1, 8):
    srv, port = server(E(K=K))
    out[K] = post(port, dict(prompt=prompt(P[3]), max_tokens=40, stop=[stop_s]))[1]
R["E18_server"] = {K: dict(text=o["choices"][0]["text"][:40], usage=o["usage"]["completion_tokens"]) for K, o in out.items()}
want = len(IdChat().decode(ref).split(stop_s)[0].split()) + 2      # tokens up to and including the stop string's last token
G["E18_server"] = (out[1]["choices"][0]["text"] == out[8]["choices"][0]["text"]
                   and out[1]["usage"]["completion_tokens"] == out[8]["usage"]["completion_tokens"] == want)

# E6 / E7 / E8 --------------------------------------------------------------------------------------------------------
eng8 = E(CTX=256)
srv, port = server(eng8)
c6, _, _ = post(port, dict(prompt=prompt(rng.integers(1, 512, 300).tolist()), max_tokens=4))
R["E6"] = c6
G["E6"] = c6 == 400
alone = [solo(P[i], 12, CTX=256) for i in (0, 1, 2)]
res = [None] * 6


def go(i, body):
    res[i] = post(port, body)


# 3 good rows sent together with 3 bad ones: top_k = 2^31, temperature NaN (json.dumps writes the raw literal NaN, which
# the server's JSON reader accepts), a token id past the vocabulary (refused by the engine's own admission check)
bodies = [dict(prompt=prompt(P[i]), max_tokens=12) for i in (0, 1, 2)] + [
    dict(prompt=prompt(P[2]), max_tokens=12, temperature=0.7, top_k=1 << 31),
    dict(prompt=prompt(P[1]), max_tokens=12, temperature=float("nan")),
    dict(prompt=prompt(P[0][:5] + [cfg.V + 5]), max_tokens=12)]
th = [threading.Thread(target=go, args=(i, b_)) for i, b_ in enumerate(bodies)]
[t.start() for t in th]
[t.join() for t in th]
good = [res[i][0] == 200 and IdChat().encode(res[i][1]["choices"][0]["text"]) == alone[i] for i in range(3)]
mixed = [None, None]
th = [threading.Thread(target=lambda i=i: mixed.__setitem__(i, post(port, dict(prompt=prompt(P[i][:20 + 30 * i]),
                                                                                 max_tokens=[200, 5][i])))) for i in (0, 1)]
[t.start() for t in th]
[t.join() for t in th]
fuzz, NONFIN = {}, ("Infinity", "-Infinity", "NaN", "1e999", "-1e999")
for field, vals in (("top_k", [-1, 0, 1, 512, 513, 2**31 - 1, 2**31, 2**63]), ("max_tokens", [0, 1, 2**31 - 1, 2**31, -5]),
                    ("temperature", [0, 1e-30, 4, 4.0001]), ("top_p", [0, 1e-30, 1, 1.5]), ("min_p", [0, 0.5, 1]),
                    ("seed", [-(2**63), 2**64 - 1, 2**64, 2**70]), ("presence_penalty", [-2, 2, 2.5]),
                    ("frequency_penalty", [-2.5, 2]), ("repetition_penalty", [0, 1e-30, 1e30])):
    # each value as a JSON number, each non-finite one as the raw JSON literal and as a quoted string
    lits = [json.dumps(v) for v in vals] + list(NONFIN) + [json.dumps(x) for x in NONFIN]
    for lit in lits:
        body = dict(prompt=prompt(P[0][:10]), max_tokens=3, temperature=0.5, stream=False)
        body.pop(field, None)
        rq = json.dumps(body)[:-1] + f', "{field}": {lit}}}'
        try:
            with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", rq.encode(),
                                                               {"Content-Type": "application/json"}), timeout=120) as f:
                code = f.status
        except urllib.error.HTTPError as x:
            code = x.code
        fuzz[f"{field}={lit}"] = code
time.sleep(0.3)
R["E8"] = dict(bad_rows=[res[i][0] for i in (3, 4, 5)], good_equal_alone=good, mixed_pair=[m[0] for m in mixed],
               fuzz_sent=len(fuzz), fuzz_codes=sorted(set(fuzz.values())),
               fuzz_500=[k_ for k_, c in fuzz.items() if c >= 500],
               nonfinite_not_400=[k_ for k_, c in fuzz.items() if k_.split("=", 1)[1] in NONFIN and c != 400])
G["E8"] = ([res[i][0] for i in (3, 4, 5)] == [400, 400, 400] and all(good) and [m[0] for m in mixed] == [200, 200]
           and not R["E8"]["fuzz_500"] and not R["E8"]["nonfinite_not_400"])
R["E7"] = dict(waiting_after=srv.waiting)
G["E7"] = srv.waiting == 0

# E19 -----------------------------------------------------------------------------------------------------------------
e = E(S=2)
a = e.admit([dict(ids=P[0], max_new=100)])[0]
e.step()
b = e.admit([dict(ids=P[4], max_new=10)])[0]
e.step()
cur_at_cancel = b["cur"]
e.cancel([b])
freed = e.free_slots()
c = e.admit([dict(ids=P[2], max_new=10)])[0]
e.step()
R["E19"] = dict(cur_at_cancel=cur_at_cancel, prompt=len(P[4]), free_after=freed, next_slot=c["slot"], next_cur=c["cur"])
G["E19"] = 0 < cur_at_cancel < len(P[4]) and freed == 1 and c["slot"] == b["slot"] and c["cur"] > 0

# E26 -----------------------------------------------------------------------------------------------------------------
e = E(S=4, budget_busy=64, chunk=64)
x, a = e.admit([dict(ids=P[0], max_new=300), dict(ids=P[1], max_new=300)])
while not a["ids"]:
    e.step()
e.cancel([x])                                  # slot 0 free: a decodes in slot 1, so the decode bucket (4) pads slots 0, 2, 3
b, c = e.admit([dict(ids=P[4], max_new=4), dict(ids=P[5], max_new=4)])
ok, seen_steps = [], 0
while c["cur"] < c["prompt_len"] and seen_steps < 20:
    paused = [r for r in (b, c) if r["fin"] is None and r["cur"] < r["prompt_len"]]
    plan = {w[0]["slot"] for w in e._plan(e.prefilling(), True)}
    still = [r["slot"] for r in paused if r["slot"] not in plan]
    others = [s_ for s_ in range(4) if e.rows[s_] is None] + still
    before = {s_: digest(e.state, s_) for s_ in others}
    e.step()
    ok.append(all(digest(e.state, s_) == before[s_] for s_ in others))
    seen_steps += 1
R["E26"] = dict(steps=seen_steps, unchanged=ok, slots=[a["slot"], b["slot"], c["slot"]])
G["E26"] = len(ok) > 2 and all(ok)

# E33 -----------------------------------------------------------------------------------------------------------------
from mashbox import cli as CLI   # noqa: E402


class WarmChat(IdChat):
    def chat_ids(self, msgs, *a, **k):      # cli.warm sends "Hello": any short prompt
        return P[1][:5]


e33 = E(S=2, CTX=1024)
CLI.warm(e33, WarmChat())       # what `mashbox run` does before it reports ready: B joins A at whatever window A has
srv, port = server(e33)         # reached (box speed decides it), and every such window was warmed, as in a real start


def run33(flag):
    """A (max_tokens = room, merge false when flag) then B 0.3 s later. -> (A, B) post results, compiles during the run."""
    out, c0 = [None, None], compiles()
    body_a = dict(prompt=prompt(P[0]), max_tokens=1024 - len(P[0]))
    if flag:
        body_a["merge"] = False
    ta = threading.Thread(target=lambda: out.__setitem__(0, post(port, body_a, stream=True)))
    ta.start()
    time.sleep(0.3)
    out[1] = post(port, dict(prompt=prompt(P[1]), max_tokens=8), stream=True)
    ta.join()
    return out, compiles() - c0


ttft = lambda o: o[1][2][1] - o[1][2][0]
first_before_end = lambda o: o[1][2][1] < o[0][2][2]
# the earlier form of the check, kept as information: one warm-up run, then one run with the field and one without
w0, c_w0 = run33(False)
(f1, c_f1), (n1, c_n1) = run33(True), run33(False)
R["E33_first_pair_info"] = dict(b_ttft_with=round(ttft(f1), 3), b_ttft_without=round(ttft(n1), 3),
                                compiles_warmup=c_w0, compiles_with=c_f1, compiles_without=c_n1)
# warm until a whole with/without pair compiles nothing (at most 4 more pairs), then 4 runs per arm in ABBA order
warm_pairs, quiet = 0, c_f1 + c_n1 == 0
while not quiet and warm_pairs < 4:
    (_, c_a), (_, c_b) = run33(True), run33(False)
    warm_pairs, quiet = warm_pairs + 1, c_a + c_b == 0
runs = {True: [], False: []}
for flag in (True, False, False, True, True, False, False, True):
    o, c_ = run33(flag)
    runs[flag].append((ttft(o), first_before_end(o), c_))
p95 = lambda xs: float(np.percentile([x[0] for x in xs], 95))
R["E33"] = dict(b_ttft_with=[round(x[0], 3) for x in runs[True]], b_ttft_without=[round(x[0], 3) for x in runs[False]],
                p95_with=round(p95(runs[True]), 3), p95_without=round(p95(runs[False]), 3),
                b_first_before_a_end=all(x[1] for x in runs[True]), warm_pairs=warm_pairs, quiet_before_measure=quiet,
                compiles_measured=sum(x[2] for x in runs[True] + runs[False]))
G["E33"] = (R["E33"]["b_first_before_a_end"] and R["E33"]["compiles_measured"] == 0
            and p95(runs[True]) <= 2 * max(p95(runs[False]), 0.05))

# E7 waiting limits (amended spec): A submits 300, B 5 while every slot is busy: B gets 0 rejections, A is refused after
# 32 with 429; the legacy streaming route answers 429 itself (before any 200); hung-up waiters free their count within 0.5 s
from mashbox.runtime import server as SV   # noqa: E402
from mashbox.runtime.server import knobs_of, Busy, Overloaded   # noqa: E402
srv, port = server(E(S=1, CTX=1024))
gate7 = hold_engine(srv)
hold = srv.submit(P[0], knobs_of(dict(max_tokens=1024 - len(P[0]))), True, False)
t_ = time.time()
while not srv.engine.busy() and time.time() - t_ < 30:
    time.sleep(0.01)
acc, rej = {"A": [], "B": []}, {"A": [], "B": []}
for o, n in (("A", 300), ("B", 5)):
    for i in range(n):
        try:
            acc[o].append(srv.submit(P[1][:8], knobs_of(dict(max_tokens=2)), True, False, scope=o))
        except Busy as e:
            rej[o].append((type(e).__name__, e.retry_after))
code_legacy = None
for i in range(32):
    acc.setdefault(None, []).append(srv.submit(P[1][:8], knobs_of(dict(max_tokens=2)), True, False))
rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(dict(prompt="1 2 3", max_tokens=2,
                            stream=True)).encode(), {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(rq, timeout=30) as f:
        code_legacy = f.status
except urllib.error.HTTPError as x:
    code_legacy, ra = x.code, x.headers.get("Retry-After")
# overall limit full: an owner under its share gets in by dropping the newest waiting request of the longest queue;
# an owner over its share (under its own limit) gets Overloaded (429, Anthropic route 529; A1)
SV.MAX_WAITING, SV.MAX_WAITING_OWNER = srv.waiting, 100
before = {o: list(js) for o, js in srv.wait_by.items()}
try:
    acc.setdefault("C", []).append(srv.submit(P[1][:8], knobs_of(dict(max_tokens=2)), True, False, scope="C"))
    c_in = True
except Busy:
    c_in = False
dropped = [j for js in before.values() for j in js if isinstance(j.get("error"), Overloaded)]
try:
    srv.submit(P[1][:8], knobs_of(dict(max_tokens=2)), True, False, scope="A")
    a_over = None
except Busy as e:
    a_over = type(e).__name__
SV.MAX_WAITING, SV.MAX_WAITING_OWNER = 256, 32
R["E7_lqd"] = dict(c_admitted=c_in, dropped=len(dropped), dropped_owner_len=max(len(js) for js in before.values()),
                   a_over_share=a_over)
G["E7_lqd"] = c_in and len(dropped) == 1 and a_over == "Overloaded"
t_ = time.time()
for js in acc.values():
    for j in js:
        srv.hangup(j)                      # the server's hang-up path (wait() / stream writers call it)
srv.hangup(hold)
while srv.waiting and time.time() - t_ < 2:
    time.sleep(0.01)
freed_s = time.time() - t_
gate7.set()
R["E7_owner"] = dict(A_accepted=len(acc["A"]), A_rejected=len(rej["A"]), A_kinds=sorted(set(k_ for k_, _ in rej["A"])),
                     B_rejected=len(rej["B"]), legacy_stream_status=code_legacy, retry_after=ra if code_legacy != 200 else None,
                     waiting_after_hangup=srv.waiting, freed_s=round(freed_s, 3))
G["E7_owner"] = (len(acc["A"]) == 32 and len(rej["A"]) == 268 and R["E7_owner"]["A_kinds"] == ["Busy"] and not rej["B"]
                 and code_legacy == 429 and srv.waiting == 0 and freed_s <= 0.5)

# E7 hang-up: 256 requests (8 owners x 32) wait while the only slot is busy; their clients hang up (the real wait()
# path, polling the client every 0.25 s); every waiting count is freed within 0.5 s, with the slot still busy
srv2, _ = server(E(S=1, CTX=1024))
gate7b = hold_engine(srv2)
hold2 = srv2.submit(P[0], knobs_of(dict(max_tokens=1024 - len(P[0]))), True, False)
t_ = time.time()
while not srv2.engine.busy() and time.time() - t_ < 30:
    time.sleep(0.01)
jobs2 = [srv2.submit(P[1][:8], knobs_of(dict(max_tokens=2)), True, False, scope=f"owner{i % 8}") for i in range(256)]
left = threading.Event()


def _waiter(j):
    try:
        srv2.wait(j, left.is_set)
    except ConnectionResetError:
        pass


ths = [threading.Thread(target=_waiter, args=(j,), daemon=True) for j in jobs2]
for th in ths:
    th.start()
w0 = srv2.waiting
t_ = time.time()
left.set()
while srv2.waiting and time.time() - t_ < 5:
    time.sleep(0.002)
hang_s = time.time() - t_
slot_busy = bool(srv2.engine.busy()) and "t_run" in hold2 and not hold2["ev"].is_set()
srv2.hangup(hold2)
gate7b.set()
R["E7_hangup"] = dict(waiting_before=w0, waiting_after=srv2.waiting, freed_s=round(hang_s, 3), slot_still_busy=slot_busy)
G["E7_hangup"] = w0 == 256 and srv2.waiting == 0 and hang_s <= 0.5 and slot_busy

# E_connect: 256 clients connect at the same moment; every one gets its answer (no connection reset from a full
# accept queue; the standard library default backlog of 5 reset 15 of 128 requests at 64 users on the chip)
srv3, port3 = server(E(S=1, CTX=1024))
go, res3 = threading.Event(), []


def _hit():
    go.wait()
    try:
        c = socket.create_connection(("127.0.0.1", port3), timeout=30)
        c.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        b = b""
        while True:
            x = c.recv(65536)
            if not x:
                break
            b += x
        c.close()
        res3.append(b.split(b"\r\n", 1)[0])
    except OSError as e:
        res3.append(type(e).__name__.encode())


th3 = [threading.Thread(target=_hit, daemon=True) for _ in range(256)]
for th in th3:
    th.start()
go.set()
for th in th3:
    th.join(60)
ok3 = sum(1 for r in res3 if r.startswith(b"HTTP/1.") and b" 200 " in r)
R["E_connect"] = dict(clients=256, ok=ok3, other=sorted(set(r.decode(errors="replace") for r in res3 if b" 200 " not in r))[:5])
G["E_connect"] = ok3 == 256

# E14 -----------------------------------------------------------------------------------------------------------------
e14, rng14 = E(S=4, CTX=1024), np.random.default_rng(14)
CLI.warm(e14, WarmChat())                   # what `mashbox run` does before it reports ready
e14.shapes, c14, hits = [], compiles(), []
for L in (1, 8, 64, 300, 1000):             # log-spaced prompt lengths up to the context
    for n in (1, 2, 3, 4):                  # rows 1..S
        for samp in (None, dict(temperature=0.7, top_p=0.9, seed=3)):
            k0, s0 = compiles(), len(e14.shapes)
            e14.generate([rng14.integers(1, 512, L).tolist() for _ in range(n)], max_new=min(8, 1024 - L), samp=[samp] * n)
            if compiles() > k0:
                hits.append(dict(prompt=L, rows=n, sampled=samp is not None, compiles=compiles() - k0,
                                 shapes=[list(x) for x in sorted(set(map(tuple, e14.shapes[s0:])))][:6]))
R["E14"] = dict(compiles_after_ready=compiles() - c14, requests=40, requests_that_compiled=len(hits), first=hits[:3])
G["E14"] = R["E14"]["compiles_after_ready"] == 0

e14c, rng14 = E(S=4, CTX=1024, prefix_cache=True, cache_ram=64 << 20, trace=True), np.random.default_rng(141)
CLI.warm(e14c, WarmChat())
c14, hits, nreq = compiles(), [], 0
conv = [rng14.integers(1, 512, 150 + 7 * i).tolist() for i in range(6)]
for turn in range(4):
    for i in range(6):                      # 6 conversations on 4 slots: each turn displaces and restores some
        k0 = compiles()
        r = e14c.generate([conv[i]], max_new=4, samp=[None if i % 2 else dict(temperature=0.7, top_p=0.9, seed=i)],
                          marks=[[len(conv[i]) - 1]], scopes=[f"t{i % 3}"])
        nreq += 1
        conv[i] = conv[i] + r["ids"][0] + rng14.integers(1, 512, 30).tolist()
        if compiles() > k0:
            hits.append(dict(turn=turn, conv=i, compiles=compiles() - k0))
    for n in (2, 3):
        k0 = compiles()
        e14c.generate([conv[j] for j in range(n)], max_new=3, samp=[None, dict(temperature=0.5, seed=1), None][:n],
                      scopes=[f"t{j % 3}" for j in range(n)])
        nreq += 1
        if compiles() > k0:
            hits.append(dict(turn=turn, rows=n, compiles=compiles() - k0))
st14 = e14c.stats
R["E14_cache"] = dict(compiles_after_ready=compiles() - c14, requests=nreq, requests_that_compiled=len(hits), first=hits[:3],
                      cached_tokens=st14["cached_tokens"], restored_tokens=st14["restored_tokens"], saves=st14["saves"])
G["E14_cache"] = (R["E14_cache"]["compiles_after_ready"] == 0 and st14["cached_tokens"] > 0 and st14["restored_tokens"] > 0
                  and st14["saves"] > 0)

# ID_info -------------------------------------------------------------------------------------------------------------
def traced(plan, **k):
    eng = E(trace=True, **k)
    rows, st = [None] * len(plan), 0
    while True:
        for i, (at, q) in enumerate(plan):
            if at == st and rows[i] is None:
                rows[i] = eng.admit([q])[0]
        if not eng.busy() and all(r is not None for r in rows):
            return rows
        eng.step()
        st += 1


lone = [traced([(0, dict(ids=P[i], max_new=40))])[0]["lh"] for i in range(4)]
bat = traced([(0, dict(ids=P[0], max_new=40)), (1, dict(ids=P[1], max_new=40)), (3, dict(ids=P[2], max_new=40)),
              (5, dict(ids=P[3], max_new=40))])


def first_diff(a_, b_):
    for j, (u, v) in enumerate(zip(a_, b_)):
        if u[0] != v[0]:
            return dict(step=j, gap_alone=round(u[1], 5))
    return None


cold = traced([(0, dict(ids=P[4], max_new=8))], budget_idle=1024)[0]["lh"]
small = traced([(0, dict(ids=P[4], max_new=8))], budget_idle=64)[0]["lh"]
R["ID_info"] = dict(batched_vs_alone=[first_diff(x, r["lh"]) for x, r in zip(lone, bat)],
                    budget64_vs_1024=first_diff(cold, small))

R["gates"] = G
R["pass"] = all(G.values())
print(json.dumps(R, default=str))
sys.exit(0 if R["pass"] else 1)
