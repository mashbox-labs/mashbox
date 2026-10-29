"""S-BATCH on every tiny family (tiny.families()), through the real Engine admit / step / cancel. ST-B1 and ST-B2 call
Engine.generate; ST-B3 and ST-B5 run arrival schedules with tiny.drive, a fixed-clock copy of the server's loop, and ST-B3
replays its schedule through the real loop (Server.submit and Server.loop_live, tiny.serve); ST-B7 and ST-B8 go through
the real server.

ST-B1  one user alone, greedy: the same tokens run to run, and in a fresh engine (the CPU form of B6); a fixture whose
       runs each repeat one token is degenerate and fails
ST-B2  batched rows equal the same requests alone, on the near-bigram weights (no near ties: exact; a degenerate fixture
       fails). On the random weights the comparison is printed (a JSON line in the log: per row, the first position that
       differs from alone and alone's top-2 gap there against one bf16 step of its top logit), not gated
ST-B3  rows that join and leave mid-batch, each with its own max_new, a stop token and a cancel, on both weight sets:
       every row keeps its own rules, a freed slot is reused and never holds two live rows (rows that share a slot have
       disjoint admit..end intervals), usage counts each request and its prompt tokens once, the engine ends idle.
       Near-bigram: each row equals its alone run exactly (the cancelled row: its tokens are alone's first ones), a fresh
       engine replays the schedule exactly (its own row, test_b3_fresh_engine_replays), and the schedule run through the
       real server loop (Server.submit + loop_live, the engine's step as the clock) gives tiny.drive's tokens, slot,
       finish and usage. Random, the chip's tie-aware rule: every committed token of every row, teacher-forced
       through the plain forward from a fresh state (Engine.logits), is its argmax or within one bf16 step of its top
       logit, and a row differs from its alone run only from a position where alone's top-2 gap is within one bf16 step.
       The stop token is the first fresh token (position 2 or later) of the first row whose alone run has one; no such
       row = a degenerate fixture, a FAIL
ST-B4  prefix-cache owner isolation, on the near-bigram engine with the prefix cache on (Engine.admit with each
       request's owner, `scope`, as the server passes it): two conversations of one owner that share a prompt prefix
       (subagents; each request marks the start of its own message, as the server's marks_of does) share it (the second
       reads cached tokens), another owner sending the very same prompts reads 0
       cached tokens until its own conversation made them, every answer equals the same request alone on the
       cache-off engine, and every resume passed the engine's owner check
ST-B5  0 compiles after the warm-up (cli.warm: the request path, then Engine.warm's program list) over a mixed schedule:
       prompts of 1, 3, 9, 27, 81 and 200 tokens, joins mid-answer, greedy + sampled + penalised rows, a stop, a cancel,
       a second turn served from the prefix cache, then a short row decoding alone (the smallest decode window) (the E14
       counter: jax's backend compile events, counted by the tier's own listener, tiny.compile_count)
ST-B7  not ready: 503 (OpenAI route) / 529 overloaded_error (Anthropic route), with Retry-After. Overload, with two
       owners (two API keys) and the waiting caps scaled to the tiny model (overall 4, per owner 3): an owner over its
       own cap gets 429 (Anthropic: rate_limit_error); the overall cap and the longest-queue drop answer 429 (OpenAI) /
       529 overloaded_error (Anthropic), with Retry-After, never 503 while ready; a queued stream dropped before its
       first byte gets that refusal as its status, never a 200 then an error event. The engine's device call is held
       while the queue fills, so every waiting count is exact (no race)
ST-B8  one load tail: a GGUF file served by `mashbox run`, and the BSHJ fixture loaded through cli.load from a models.json
       row, each report a non-null deployment_id and component_set_sha and the /health keys of runtime/front.py
       HEALTH_FIELDS
"""
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

import tiny

pytestmark = pytest.mark.standards
MAX_NEW = 10


@pytest.fixture(scope="module")
def engines(family):
    """the family's near-bigram and random engines (4 slots, 4 decode steps per call, no prefix cache)"""
    return {k: tiny.engine(family, k) for k in ("bigram", "random")}


def _open_fixture(family, request):
    if family in tiny.DEGENERATE:
        request.applymarker(pytest.mark.xfail(strict=True, reason=tiny.DEGENERATE[family]))


@pytest.mark.parametrize("kind", ["bigram", "random"])
@pytest.mark.slow
def test_b1_lone_greedy_same_run_to_run(family, engines, kind, request):
    """Slow (CI.md: over 10 s, marked `slow`; the tier still runs it): its first calls compile the module engine's
    generate programs and a fresh engine's. Sealed calls over 10 s on the gate VM (n2-highcpu-32, one process per
    family on 2-3 cores): qwen38-bshj-bigram 10.28 s (r4 pre-check). The cost is XLA compiles on CPU: every Engine
    compiles its own programs."""
    _open_fixture(family, request)
    eng = engines[kind][0]
    runs = []
    for p in tiny.prompts(eng, 3, seed=1):
        a, b = tiny.alone(eng, p, MAX_NEW), tiny.alone(eng, p, MAX_NEW)
        assert a == b, (family, kind, p)
        assert len(a) == MAX_NEW
        runs.append(a)
    fresh = tiny.engine(family, kind)[0]
    p = tiny.prompts(eng, 1, seed=1)[0]
    assert tiny.alone(fresh, p, MAX_NEW) == tiny.alone(eng, p, MAX_NEW), (family, kind, "fresh engine")
    assert any(len(set(r)) > 1 for r in runs), ("degenerate fixture: every run repeats one token", family, kind, runs)


@pytest.mark.parametrize("kind", ["bigram", "random"])
@pytest.mark.slow
def test_b2_batched_equals_alone(family, engines, kind, request):
    """Slow (CI.md: over 10 s, marked `slow`; the tier still runs it): its first calls compile the batched programs; on
    the random weights each tie record teacher-forces a new length through Engine.logits. Sealed calls over 10 s on
    the gate VM (n2-highcpu-32, one process per family on 2-3 cores): qwen35moe-f32-bigram 12.24 s and qwen38-bshj-
    random 11.49 s (gate g-f8a6847); qwen35-f32-bigram 13.86 s, qwen38-bshj-random 12.70 s, gemma4-f32-bigram 11.98
    s, qwen35moe-f32-bigram 11.75 s, qwen38-bshj-bigram 10.19 s (r4 pre-check). The cost is XLA compiles on CPU:
    every Engine compiles its own programs."""
    if kind == "bigram":
        _open_fixture(family, request)
    eng = engines[kind][0]
    ps = tiny.prompts(eng, 3, seed=2)
    batched = eng.generate(ps, max_new=MAX_NEW)["ids"]
    solo = [tiny.alone(eng, p, MAX_NEW) for p in ps]
    eq = [a == b for a, b in zip(batched, solo)]
    rec = dict(test="ST-B2", family=family, kind=kind, batched_equals_alone=eq)
    if kind == "random":            # recorded, not gated: where each row first differs, and whether alone has a tie there
        rec["rows"] = [tiny.tie_record(eng, p, dict(ids=b, fin="length"), a) for p, b, a in zip(ps, batched, solo)]
    print(json.dumps(rec))
    if kind == "bigram":
        assert any(len(set(r)) > 1 for r in solo), ("degenerate fixture: every alone row repeats one token", family, solo)
        assert all(eq), (family, batched, solo)


def _schedule(eng):
    ps = tiny.prompts(eng, 6, lo=3, hi=30, seed=3)
    mx = [MAX_NEW, 4, 12, 7, 3, 9]
    at = [0, 0, 2, 3, 5, 6]
    return [(at[i], dict(ids=ps[i], max_new=mx[i])) for i in range(6)]


def _stop_row(eng, plan, skip):
    """-> (row k, position j, token): the first row (not `skip`) whose alone run has a token at position j >= 2 that it
    did not produce before; None when no row has one"""
    for k, (_at, q) in enumerate(plan):
        if k == skip:
            continue
        run = tiny.alone(eng, q["ids"], q["max_new"])
        j = next((n for n in range(2, len(run)) if run[n] not in run[:n]), None)
        if j is not None:
            return k, j, run[j]
    return None


@pytest.mark.parametrize("kind", ["bigram", "random"])
@pytest.mark.slow
def test_b3_join_leave_stop_cancel(family, engines, kind, request):
    """Slow (CI.md: over 10 s, marked `slow`; the tier still runs it): an arrival schedule compiles the prefill chunk
    and decode window programs it reaches first, then 6 alone runs. Sealed calls over 10 s on the gate VM
    (n2-highcpu-32, one process per family on 2-3 cores): qwen38-bshj-bigram 21.52 s, qwen35moe-f32-bigram 17.10 s,
    qwen35-f32-bigram 15.84 s, qwen35-mtp-bigram 10.91 s (gate g-f8a6847, when this row also held the fresh-engine
    replay); qwen35-f32-bigram 13.86 s, qwen38-bshj-bigram 12.24 s, qwen35moe-f32-bigram 10.72 s (r4 pre-check,
    without it). The cost is XLA compiles on CPU: every Engine compiles its own programs."""
    _open_fixture(family, request)
    eng, chat, _info = engines[kind]
    plan = _schedule(eng)
    cancel_at = {2: 4}
    picked = _stop_row(eng, plan, skip=2)
    assert picked is not None, ("degenerate fixture: no row's alone run has a fresh token to stop on", family, kind,
                                [tiny.alone(eng, q["ids"], q["max_new"]) for _a, q in plan])
    k, j, tok = picked
    stop = (tok,)
    s0 = dict(eng.stats)
    rows = tiny.drive(eng, plan, stop_ids=stop, cancel_at=cancel_at)
    _slots_reused_never_shared(rows, eng)
    assert eng.stats["requests"] - s0["requests"] == len(plan)                  # usage: each request once
    assert eng.stats["prompt_tokens"] - s0["prompt_tokens"] == sum(len(q["ids"]) for _a, q in plan)
    assert not eng.busy() and eng.free_slots() == eng.S and eng.order == []
    solo = [tiny.alone(eng, q["ids"], q["max_new"], stop_ids=stop) for _a, q in plan]
    ties = []
    for i, ((_at, q), r, a) in enumerate(zip(plan, rows, solo)):
        assert r["fin"] in ("stop", "length", "cancel"), (i, r["fin"])
        assert all(0 <= t < tiny.vocab_size(eng) for t in r["ids"])
        assert stop[0] not in r["ids"], (i, "stop token in the answer")
        if r["fin"] == "length":
            assert len(r["ids"]) == q["max_new"], (i, len(r["ids"]))
        elif r["fin"] == "stop":
            assert len(r["ids"]) < q["max_new"], (i, "stop at the limit")
        else:
            assert i == 2 and len(r["ids"]) < q["max_new"], (i, "cancel")
        assert r["t_end"] is not None and r["slot"] is not None
        if kind == "bigram":
            if i != 2:
                assert r["ids"] == a, (family, i, "near-bigram alone", r["ids"], a)
            else:
                assert r["ids"] == a[:len(r["ids"])], (family, i, "near-bigram alone (cancelled row)", r["ids"], a)
        else:
            t = tiny.tie_record(eng, q["ids"], r, a, stop)
            ties.append(t)
            assert t["beyond"] == 0, (family, i, "a committed token is more than one bf16 step under the plain "
                                                 "forward's top logit", t, r["ids"])
            assert t["first"] is None or t["tie"], (family, i, "differs from its alone run where alone has no tie", t,
                                                    r["ids"], a)
    if ties:
        print(json.dumps(dict(test="ST-B3", family=family, kind=kind, rows=ties)))
    if kind == "bigram" or rows[k]["ids"][:j] == solo[k][:j]:       # random: unless the row left alone's path at a tie
        assert rows[k]["fin"] == "stop" and len(rows[k]["ids"]) == j, ("the stop row", k, rows[k]["fin"], len(rows[k]["ids"]))
    assert rows[2]["fin"] == "cancel"


def _slots_reused_never_shared(rows, eng):
    """a freed slot was reused (more rows than slots, every row had one), and rows that share a slot never overlap: each
    one was admitted after the one before it in that slot ended"""
    assert len(rows) > eng.S and all(r["slot"] is not None for r in rows), [r["slot"] for r in rows]
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["slot"], []).append((r["t_admit"], r["t_end"], i))
    assert any(len(v) > 1 for v in by.values()), ("no slot hosted two rows", by)
    for s_, v in by.items():
        v.sort(key=lambda x: x[0])
        for (a0, e0, i0), (a1, _e1, i1) in zip(v, v[1:]):
            assert e0 is not None and e0 <= a1, ("two live rows in one slot", s_, i0, i1, v)


@pytest.mark.slow
def test_b3_fresh_engine_replays(family, engines, request):
    """ST-B3's near-bigram schedule on a fresh engine (a new tiny.engine, no history) gives the module engine's tokens,
    slot and finish row for row (the CPU form of B6 for a schedule).

    Slow (CI.md: over 10 s, marked `slow`; the tier still runs it): the fresh engine compiles every program the
    schedule reaches. Sealed: qwen38-bshj 19.89 s on the gate VM (r4 pre-check, 2-3 cores per family process)."""
    _open_fixture(family, request)
    eng = engines["bigram"][0]
    plan = _schedule(eng)
    cancel_at = {2: 4}
    picked = _stop_row(eng, plan, skip=2)
    assert picked is not None, ("degenerate fixture: no row's alone run has a fresh token to stop on", family)
    stop = (picked[2],)
    rows = tiny.drive(eng, plan, stop_ids=stop, cancel_at=cancel_at)
    replay = tiny.drive(tiny.engine(family, "bigram")[0], plan, stop_ids=stop, cancel_at=cancel_at)
    for i, (r, r2) in enumerate(zip(rows, replay)):
        assert (r["ids"], r["slot"], r["fin"]) == (r2["ids"], r2["slot"], r2["fin"]), (family, i, "replay", r["ids"], r2["ids"])


def test_b3_server_loop_equals_drive(family, engines, request):
    """ST-B3's near-bigram schedule through the real server loop: submitted with Server.submit and run by
    Server.loop_live (tiny.serve, the engine's step as the clock), every row has the tokens, slot and finish tiny.drive
    gives, usage counts each request and its prompt tokens once, and the engine ends idle"""
    if family == tiny.BSHJ:
        request.applymarker(pytest.mark.xfail(strict=True, reason=NO_TOKENIZER))
    _open_fixture(family, request)
    eng, chat, _info = engines["bigram"]
    plan = _schedule(eng)
    cancel_at = {2: 4}
    picked = _stop_row(eng, plan, skip=2)
    assert picked is not None, ("degenerate fixture: no row's alone run has a fresh token to stop on", family)
    stop = (picked[2],)
    rows = tiny.drive(eng, plan, stop_ids=stop, cancel_at=cancel_at)
    s0 = dict(eng.stats)
    served = tiny.serve(eng, chat, plan, stop_ids=stop, cancel_at=cancel_at)
    for i, (r, v) in enumerate(zip(rows, served)):
        assert (r["ids"], r["slot"], r["fin"]) == (v["ids"], v["slot"], v["fin"]), (family, i, "server loop", r, v)
    assert eng.stats["requests"] - s0["requests"] == len(plan)
    assert eng.stats["prompt_tokens"] - s0["prompt_tokens"] == sum(len(q["ids"]) for _a, q in plan)
    assert not eng.busy() and eng.free_slots() == eng.S and eng.order == []


@pytest.fixture(scope="module")
def warmed(family):
    """the family's near-bigram engine with the prefix cache on, after cli.warm (the request path, then Engine.warm's
    program list); built once per module, so ST-B5's own time is its schedule"""
    from mashbox import cli
    tiny.compile_count()
    eng, chat, _info = tiny.engine(family, "bigram", prefix_cache=True)
    log, cli.log = cli.log, (lambda s: None)
    try:
        cli.warm(eng, chat)
    finally:
        cli.log = log
    return eng


LENGTHS = (1, 3, 9, 27, 81, 200)         # log-spaced: a one-token prompt, then every prefill chunk bucket up to 200


def test_b5_no_compile_after_warm(family, warmed):
    eng = warmed
    c0 = tiny.compile_count()
    ps = tiny.prompts(eng, lengths=LENGTHS, seed=5)
    samp = [None, dict(temperature=0.8, top_p=0.9, top_k=20, seed=3), None,
            dict(temperature=0.0, repetition_penalty=1.3, frequency_penalty=0.5, presence_penalty=0.2), None, None]
    plan = [(at, dict(ids=p, max_new=m, samp=s)) for at, p, m, s in zip([0, 1, 1, 3, 4, 8], ps, [9, 6, 12, 5, 8, 4], samp)]
    rows = tiny.drive(eng, plan, stop_ids=(ps[0][0],), cancel_at={1: 3})
    turn2 = ps[2] + rows[2]["ids"] + tiny.prompts(eng, 1, lo=3, hi=9, seed=6)[0]
    r2 = tiny.drive(eng, [(0, dict(ids=turn2, max_new=5)), (1, dict(ids=ps[4], max_new=3))])
    assert r2[0]["cached"] > 0, ("the second turn was not served from the prefix cache", r2[0]["cached"])
    short = tiny.drive(eng, [(0, dict(ids=tiny.prompts(eng, lengths=(3,), seed=7)[0], max_new=8))])  # decodes alone
    assert short[0]["fin"] in ("stop", "length"), short[0]["fin"]
    after = tiny.compile_count() - c0
    assert after == 0, (family, f"{after} compiles after warm-up")


def test_b4_prefix_cache_owner_isolation(family, warmed, engines):
    """gated on every family, degenerate fixtures included: its checks are cached-token counts and equality with the
    cache-off engine, which need no varied tokens"""
    eng, ref = warmed, engines["bigram"][0]
    base = tiny.prompts(eng, lengths=(48,), seed=8)[0]
    tails = tiny.prompts(eng, 3, lo=3, hi=6, seed=9)
    turn = lambda k: base + tails[k]                    # noqa: E731 (three conversations sharing a 48-token prefix)
    checks0 = eng.stats["owner_checks"]

    def run(owner, k):
        r = tiny.drive(eng, [(0, dict(ids=turn(k), max_new=6, scope=owner, mark=len(base)))])[0]   # mark: the server's
        # point at a message start (Server.marks_of), here the start of each conversation's own message
        assert r["ids"] == tiny.alone(ref, turn(k), 6), (family, owner, k, "differs from alone on the cache-off engine")
        return r["cached"]
    first = run("std-owner-a", 0)
    shared = run("std-owner-a", 1)                      # a subagent of the same owner: the prefix is shared
    other = [run("std-owner-b", 0), run("std-owner-c", 1)]   # a's very prompts, from owners with nothing cached yet
    own_b = run("std-owner-b", 2)                       # owner b's next conversation: its own prefix
    again_a = run("std-owner-a", 2)
    assert first == 0, (family, "owner a's first prompt read cached tokens it never made", first)
    assert shared >= len(base) // 2, (family, "one owner's conversations do not reuse the prefix", first, shared)
    assert other == [0, 0], (family, "another owner's prompts were served from this owner's cache", other)
    assert own_b >= len(base) // 2 and again_a >= len(base) // 2, (family, "an owner's own cache", own_b, again_a)
    assert eng.stats["owner_checks"] > checks0, ("no resume passed the owner check", eng.stats["owner_checks"])


def _serve(eng, chat, monkeypatch=None, keys=None, **caps):
    from mashbox.runtime import server as SV
    for k, v in caps.items():
        monkeypatch.setattr(SV, k, v)
    srv = SV.Server(eng, chat, "tiny", dict(weights="tiny"), log=lambda m: None)
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    threading.Thread(target=SV.serve, args=(srv, "127.0.0.1", port, lambda m: None), kwargs=dict(keys=keys),
                     daemon=True).start()
    for _ in range(500):
        try:
            socket.create_connection(("127.0.0.1", port)).close()
            break
        except OSError:
            time.sleep(0.02)
    return srv, port


def _post(port, path, body, timeout=120, key=None):
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


ROUTES = [("/v1/chat/completions", lambda n, s: dict(model="tiny", messages=[dict(role="user", content=f"hi {n}")],
                                                       max_tokens=48, stream=s)),
          ("/v1/messages", lambda n, s: dict(model="tiny", messages=[dict(role="user", content=f"hi {n}")],
                                              max_tokens=48, stream=s))]
KEYS = ("std-owner-a-key", "std-owner-b-key")       # two API keys: two owners, each with its own waiting queue


def _until(cond, what, secs=60):
    t = time.monotonic() + secs
    while not cond():
        assert time.monotonic() < t, ("timed out waiting for", what)
        time.sleep(0.005)


def _err_type(body):
    try:
        return json.loads(body)["error"]["type"]
    except (ValueError, KeyError, TypeError):
        return None


NO_TOKENIZER = ("open: the BSHJ fixture has no tokenizer for the chat routes (the tiny tokenizer and template come "
                "with ST-A1)")


def test_b7_not_ready_answers_503_529(family, monkeypatch, request):
    if family == tiny.BSHJ:
        request.applymarker(pytest.mark.xfail(strict=True, reason=NO_TOKENIZER))
    eng, chat, _ = tiny.engine(family, "random", S=1)
    srv, port = _serve(eng, chat, monkeypatch)
    srv.ready = False
    for path, body in ROUTES:
        code, _b, h = _post(port, path, body(0, False))
        assert code == (529 if path == "/v1/messages" else 503) and "Retry-After" in h, (path, code)


@pytest.mark.slow
@pytest.mark.parametrize("row", ["caps", "drop"])
def test_b7_overload_answers_429_529_never_503(family, monkeypatch, request, row):
    """row "caps": the per-owner cap (a5) and the overall cap (b3) refuse with their codes, every other request is
    served; row "drop": the queued stream a4, dropped for b2 (longest-queue drop), gets its refusal as its status.

    Slow (CI.md: over 10 s, marked `slow`; the tier still runs it): a new 1-slot engine compiles its programs, then 8
    requests on each of 2 routes. Sealed: gemma4-f32-caps 12.73 s, qwen35moe-f32-caps 10.02 s on the gate VM (r4
    pre-check, 2-3 cores per family process)."""
    eng, chat, _ = tiny.engine(family, "random", S=1)
    if family == tiny.BSHJ:
        request.applymarker(pytest.mark.xfail(strict=True, reason=NO_TOKENIZER))
    chat.decode(chat.chat_ids([dict(role="user", content="hi")]))     # the chat routes need a tokenizer that decodes
    held, entered, real_step = threading.Event(), threading.Event(), eng.step

    def held_step(*a, **k):             # the loop's device call is held while the queue fills: exact waiting counts
        entered.set()
        assert held.wait(120), "the held device call was never released"
        return real_step(*a, **k)
    monkeypatch.setattr(eng, "step", held_step)
    srv, port = _serve(eng, chat, monkeypatch, keys=KEYS, MAX_WAITING=4, MAX_WAITING_OWNER=3)
    srv.ready = True
    A, B = KEYS
    for path, body in ROUTES:
        anth = path == "/v1/messages"
        over = 529 if anth else 429
        held.clear()
        entered.clear()
        res, ts = {}, []

        def send(name, key, stream=False):
            def go():
                res[name] = _post(port, path, body(name, stream), key=key)
            ts.append(threading.Thread(target=go))
            ts[-1].start()
        send("a1", A)                                                   # takes the one slot; its device call is held
        _until(entered.is_set, ("the first request in the engine", res))
        for n, (name, key, stream) in enumerate([("a2", A, False), ("a3", A, False), ("a4", A, True), ("b1", B, False)], 1):
            send(name, key, stream)
            _until(lambda: srv.waiting == n, (f"{name} waiting", srv.waiting, res))
        send("a5", A)                   # owner a has 3 waiting = its own cap: refused alone
        _until(lambda: "a5" in res, ("a5's answer", res))
        send("b2", B)                   # queue full (4), owner b under its share (1 < 4 / 2): a's newest (stream a4) dropped
        _until(lambda: "a4" in res, ("a4's answer (dropped from the queue)", res))
        send("b3", B)                   # queue full, owner b at its share (2 of 4 / 2): the overall cap refuses it
        _until(lambda: "b3" in res, ("b3's answer", res))
        held.set()
        for t in ts:
            t.join(180)
        codes = {n: res.get(n, (None,))[0] for n in ("a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3")}
        refused = ("a5", "b3") if row == "caps" else ("a4",)
        if row == "caps":
            assert {n: codes[n] for n in ("a1", "a2", "a3", "a5", "b1", "b2", "b3")} == dict(
                a1=200, a2=200, a3=200, a5=429, b1=200, b2=200, b3=over), (path, codes)
            for n in ("a1", "a2", "a3", "b1", "b2"):
                assert b"error" not in res[n][1].split(b"\n", 1)[0], (path, n, res[n][1][:200])
        else:
            assert codes["a4"] == over, (path, codes, "a4 (the dropped stream):", res.get("a4", (None, b""))[1][:300])
        for n in refused:
            code, b, h = res[n]
            assert "Retry-After" in h, (path, n, h)
            if anth:
                assert _err_type(b) == ("rate_limit_error" if n == "a5" else "overloaded_error"), (path, n, b[:200])
        _until(lambda: srv.waiting == 0 and not eng.busy(), "the server idle")


NO_TINY_ROW = (f"open: no models.json row {tiny.TINY_ROW!r} loads the BSHJ fixture through cli.load (the models.json path "
               "of the load tail)")


def test_b8_one_load_tail(family, request):
    from mashbox.runtime import server as SV
    from mashbox.runtime.front import Front
    if family == tiny.BSHJ:
        request.applymarker(pytest.mark.xfail(strict=True, reason=NO_TINY_ROW))
        eng, chat, _ = tiny.catalog_engine(S=1)
    else:
        eng, chat, _ = tiny.engine(family, "bigram", S=1)
    h = SV.Server(eng, chat, "tiny", dict(weights="tiny"), log=lambda m: None).health()
    assert h["deployment_id"] and h["component_set_sha"], h
    assert set(Front.HEALTH_FIELDS) <= set(h), sorted(set(Front.HEALTH_FIELDS) - set(h))
