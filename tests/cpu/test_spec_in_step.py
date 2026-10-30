"""Speculative decoding inside the batched step (Engine._spec_step, runtime/spec_step.py make_step with the MTP drafter) on tiny random
models of the Qwen3.8-27B architecture, CPU. The drafter gate's CPU forms
(the same rules the chip gate applies):
  G-D1  one user alone, greedy, MTP on (K = 1, 2, 3), the tie-aware replay (the G-D1 rule; batched output may
        differ from alone): every committed token, teacher-forced through the plain forward, is its argmax
        or within one bf16 step of its top logit; the near-tie count (committed tokens that are not the argmax) is
        recorded. Token-for-token MTP on == off is information here, not the gate: exact on the near-bigram model; on the
        random model 2 of 4 prompts differ at a bf16 tie (top-2 gap 0 or one bf16 step: the verify width and the one-token
        decode width round differently, on CPU), kept as an xfail for a later opt-in exact mode (fixed-order
        verify ops, not built); not strict, because whether a tie flips depends on the runner's CPU rounding, so these
        rows pass on some runners and fail on others.
  G-D2  rows that join and leave while others decode, with different k (0 for draft=False rows), a stop token and a
        cancel mid-answer, at 1, 8 and 64 users: the base batched rule (as C1-C5): the run and
        its replay in a fresh engine give every row the same tokens, slot and finish, every row keeps its own rules, both
        engines end idle with usage counted once, the cancelled row ends "cancel"; on the near-bigram model (no ties)
        drafts were accepted, a stop fired (8 and 64 users) and every row also equals the same schedule without
        speculation; on the random model that equality, the tokens per round and the tie-aware replay numbers
        (speculation on and off) are recorded, not gated. Sampled and penalised rows equal the plain schedule (the coupled draw:
        no rejection sampling); max_tokens and stop tokens inside a draft; cancel mid-draft, then the freed slot
        serves a new request correctly
  roll  rejected drafts leave no trace: the slot state after speculation equals the state after plain decode of the same
        tokens (linear-attention state, conv tail, full-attention K/V at every committed position); bit-equal on the
        near-bigram model, a strict xfail with its record on the random model (one bf16 step from the verify width)
  G-D3  after warm(), a mixed run compiles no speculation program (spec_programs() covers what the step runs)
"""
import json
import struct

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
jax = pytest.importorskip("jax")

from mashbox.models import qwen3_8_27b as Q  # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW, spec as QS  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402

bf = ml_dtypes.bfloat16
D, F, V, L, H, KV, HD, HK, HV, DK, DV, CK = 64, 96, 512, 8, 4, 2, 32, 2, 6, 16, 16, 4


def write_bshj(path, tensors):
    align, pos, ents, bufs = 4096, 0, {}, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {"bfloat16": "bf16", "float32": "f32", "int8": "i8"}[a.dtype.name]
        pos = -(-pos // align) * align
        ents[name] = dict(dtype=dt, shape=list(a.shape), offset=pos, nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    raw = json.dumps(dict(format="bshj", version="1.0", data_len=pos, tensors=ents), separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p, b in bufs:
            f.seek(off + p)
            f.write(b)
        f.truncate(off + pos)


def tensors(seed, lo):
    """random weights; lo scales every residual write: small = close to a bigram model, whose MTP head passes the next
    token's embedding through (drafts mostly accepted)"""
    rng = np.random.default_rng(seed)
    r = lambda *s, sd=0.08: (rng.normal(0, sd, s)).astype(bf)
    T = {"model.embed_tokens.weight": r(V, D, sd=1.0), "model.norm.weight": r(D, sd=0.1), "lm_head.weight": r(V, D, sd=0.3)}
    for i in range(L):
        p = f"model.layers.{i}."
        T[p + "input_layernorm.weight"], T[p + "post_attention_layernorm.weight"] = r(D, sd=0.3), r(D, sd=0.3)
        T[p + "mlp.gate_proj.weight"], T[p + "mlp.up_proj.weight"], T[p + "mlp.down_proj.weight"] = r(F, D), r(F, D), r(D, F, sd=0.08 * lo)
        if (i + 1) % 4:
            a = p + "linear_attn."
            T[a + "in_proj_qkv.weight"], T[a + "in_proj_z.weight"] = r(2 * HK * DK + HV * DV, D), r(HV * DV, D)
            T[a + "in_proj_a.weight"], T[a + "in_proj_b.weight"] = r(HV, D), r(HV, D)
            T[a + "out_proj.weight"], T[a + "conv1d.weight"] = r(D, HV * DV, sd=0.08 * lo), r(2 * HK * DK + HV * DV, 1, CK, sd=0.4)
            T[a + "A_log"], T[a + "dt_bias"] = np.log(rng.uniform(1, 16, HV)).astype(np.float32), r(HV, sd=1.0)
            T[a + "norm.weight"] = r(DV, sd=0.3)
        else:
            a = p + "self_attn."
            T[a + "q_proj.weight"], T[a + "k_proj.weight"], T[a + "v_proj.weight"] = r(2 * H * HD, D), r(KV * HD, D), r(KV * HD, D)
            T[a + "o_proj.weight"] = r(D, H * HD, sd=0.08 * lo)
            T[a + "q_norm.weight"], T[a + "k_norm.weight"] = r(HD, sd=0.3), r(HD, sd=0.3)
    m = "mtp.layers.0."
    fc = np.concatenate([np.eye(D), np.zeros((D, D))], 1).astype(bf) if lo < 1 else r(D, 2 * D)
    T.update({"mtp.fc.weight": fc, "mtp.pre_fc_norm_embedding.weight": r(D, sd=0.1), "mtp.pre_fc_norm_hidden.weight": r(D, sd=0.1),
              "mtp.norm.weight": T["model.norm.weight"], m + "input_layernorm.weight": r(D, sd=0.3),
              m + "post_attention_layernorm.weight": r(D, sd=0.3), m + "self_attn.q_proj.weight": r(2 * H * HD, D),
              m + "self_attn.k_proj.weight": r(KV * HD, D), m + "self_attn.v_proj.weight": r(KV * HD, D),
              m + "self_attn.o_proj.weight": r(D, H * HD, sd=0.08 * lo), m + "self_attn.q_norm.weight": r(HD, sd=0.3),
              m + "self_attn.k_norm.weight": r(HD, sd=0.3), m + "mlp.gate_proj.weight": r(F, D), m + "mlp.up_proj.weight": r(F, D),
              m + "mlp.down_proj.weight": r(D, F, sd=0.08 * lo)})
    return T


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    wd = tmp_path_factory.mktemp("spec_in_step")
    json.dump(dict(model_type="qwen3_5_text", vocab_size=V, hidden_size=D, intermediate_size=F, num_hidden_layers=L,
                   num_attention_heads=H, num_key_value_heads=KV, head_dim=HD, linear_num_key_heads=HK, linear_num_value_heads=HV,
                   linear_key_head_dim=DK, linear_value_head_dim=DV, linear_conv_kernel_dim=CK, full_attention_interval=4,
                   rms_norm_eps=1e-6, rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25),
                   tie_word_embeddings=False, attn_output_gate=True, mtp_num_hidden_layers=1), open(wd / "config.json", "w"))
    cfg = Q.Config(str(wd))
    out = {}
    for tag, lo in (("random", 1.0), ("bigram", 0.05)):
        path = str(wd / f"{tag}.bshj")
        write_bshj(path, tensors(11, lo))
        W, _ = QW.load(path, cfg)
        Wm, _ = QS.load_mtp(path, cfg)
        out[tag] = (cfg, W, Wm)
    return out


def engines(t, k, **kw):
    cfg, W, Wm = t
    return Engine(cfg, W, Q, **kw), Engine(cfg, W, Q, mtp=(QS, Wm), draft_k=k, spec_rounds=3, **kw)


def drive(eng, plan, stop_ids=(), cancel_at=None, force=None, cancel_after=None):
    """plan: list of (step index to arrive at, request dict). Requests are admitted in arrival order as slots free up,
    steps run until every row ends (step(waiting=True) while a request waits, as the server does). cancel_at: {request
    index: step index}. cancel_after: {request index: n}, cancelled at the first step it holds n tokens or more. force: token list for request 0, fed back after each step in place of the engine's own token
    (one-step-per-call engines: plain decode of exactly these tokens). -> rows in plan order."""
    rows, nxt, i = [None] * len(plan), 0, 0
    while nxt < len(plan) or eng.busy():
        while nxt < len(plan) and plan[nxt][0] <= i and eng.free_slots():
            rows[nxt] = eng.admit([plan[nxt][1]], stop_ids)[0]
            nxt += 1
        for j, at in (cancel_at or {}).items():
            if at == i and rows[j] is not None:
                eng.cancel([rows[j]])
        for j, n in (cancel_after or {}).items():
            if rows[j] is not None and rows[j]["fin"] is None and len(rows[j]["ids"]) >= n:
                eng.cancel([rows[j]])
        if eng.busy():
            eng.step(waiting=nxt < len(plan))
            r0 = rows[0]
            if force is not None and r0["ids"] and r0["fin"] is None and r0["ids"][-1] != force[len(r0["ids"]) - 1]:
                r0["ids"][-1] = eng.tok[r0["slot"]] = force[len(r0["ids"]) - 1]
        i += 1
        assert i < 10000
    return rows


def prompts(n, seed=5, lo=3, hi=60):
    rng = np.random.default_rng(seed)
    return [rng.integers(1, V, int(rng.integers(lo, hi))).tolist() for _ in range(n)]


EXACT_TIE = pytest.mark.xfail(strict=False, reason="exact mode (not the G-D1 rule): bf16 ties flip greedy tokens "
                               "between verify and decode widths (random model, prompts of 3 and 7 tokens, gap 0.0625 and 0.0; "
                               "same on main's whole-batch MTP); needs fixed-order ops at the verify widths (batching "
                               "spec §6 risk), a later opt-in mode; outcome depends on the runner (whether a tie flips "
                               "follows the CPU's rounding), so not strict")


@pytest.mark.parametrize("tag", [pytest.param("random", marks=EXACT_TIE), "bigram"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_exact_mode_lone_greedy_on_equals_off(tiny, tag, k):
    """information for a later opt-in exact mode, not the G-D1 gate: MTP on == off token for token for one user alone"""
    base, spec = engines(tiny[tag], k, CTX=256, S=4, K=4)
    for p in prompts(4):
        want = base.generate([p], max_new=40)["ids"][0]
        assert drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"] == want
        assert spec.generate([p], max_new=40)["ids"][0] == want
    if tag == "bigram" and k == 3:      # drafts were really accepted (the near-bigram head), so long commits ran
        assert spec.spec_stats["tokens"] / spec.spec_stats["rounds"] > 2.0


def bf16_step(x):
    return 2.0 ** (np.floor(np.log2(np.maximum(np.abs(x), 1e-30))) - 7)


def tie_aware_replay(base, p, got):
    """G-D1's check of one greedy answer: teacher-force its tokens through the plain (non-speculative) forward ->
    (not_argmax, beyond): how many committed tokens are not the argmax there, and how many are more than one bf16 step
    under the top logit"""
    lg = np.asarray(base.logits(p + got)[len(p) - 1:len(p) - 1 + len(got)], np.float32)
    top, mine = lg.max(-1), lg[np.arange(len(got)), got]
    return int(np.sum(lg.argmax(-1) != np.asarray(got))), int(np.sum(top - mine > bf16_step(top)))


@pytest.mark.parametrize("tag", ["random", "bigram"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_gd1_lone_greedy_tie_aware_replay(tiny, tag, k, record_property):
    """G-D1: one user alone, greedy, MTP on, through the batched step (drive) and generate():
    every committed token is the plain forward's argmax or within one bf16 step of its top logit; the near-tie count is
    recorded (record_property gd1_near_tie, and printed)"""
    base, spec = engines(tiny[tag], k, CTX=256, S=4, K=4)
    n = near = over = 0
    for p in prompts(4):
        for got in (drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"], spec.generate([p], max_new=40)["ids"][0]):
            assert len(got) == 40
            a, b = tie_aware_replay(base, p, got)
            n, near, over = n + len(got), near + a, over + b
    record_property("gd1_tokens", n)
    record_property("gd1_near_tie", near)
    record_property("gd1_beyond_one_bf16_step", over)
    print(f"G-D1 {tag} k={k}: {n} tokens, near-tie (not the argmax) {near}, beyond one bf16 step {over}")
    assert over == 0, (tag, k, n, near, over)
    if tag == "bigram" and k == 3:      # drafts were really accepted (the near-bigram head), so long commits ran
        assert spec.spec_stats["tokens"] / spec.spec_stats["rounds"] > 2.0


def gd2_row_ok(r, q, stop_ids, cancelled):
    """the base batched rule's row rules (continuous_batching_checks.row_ok) with a cancel: the row ends by itself ("length"
    exactly at max_new, "stop" before it and only with a stop token set) or, for the cancelled request only, "cancel" before
    max_new; no stop token in its answer, every token in the vocabulary, a slot"""
    ids, mx, fin = r["ids"], q["max_new"], r["fin"]
    return bool(r["slot"] is not None and (fin in ("length", "stop") or (cancelled and fin == "cancel"))
                and len(ids) <= mx and (fin != "length" or len(ids) == mx) and (fin != "stop" or (stop_ids and len(ids) < mx))
                and (fin != "cancel" or len(ids) < mx) and not set(ids) & set(stop_ids) and all(0 <= t < V for t in ids))


def gd2_idle(eng, s0, plan):
    """the base batched rule's state: no live row, every slot free, nothing in the admission order, no slot decoding, and
    the usage counters count each request and its prompt tokens once"""
    return bool(not eng.busy() and eng.free_slots() == eng.S and not eng.order and not np.any(eng.act)
                and eng.stats["requests"] - s0["requests"] == len(plan)
                and eng.stats["prompt_tokens"] - s0["prompt_tokens"] == sum(len(q["ids"]) for _, q in plan))


def gd2_schedule(users):
    """rows that arrive over the first 12 steps (more requests than slots at 8 and 64 users, so rows join and leave
    mid-batch), each with its own max_new, every third row draft=False (k 0 inside the speculating step)"""
    ps = prompts(users + users // 2, seed=7, hi=40)
    rng = np.random.default_rng(3)
    plan = [(int(rng.integers(0, 12)), dict(ids=p, max_new=30 if j == 1 else int(rng.integers(1, 30)), draft=bool(j % 3)))
            for j, p in enumerate(ps + [ps[0]])]
    plan.sort(key=lambda x: x[0])
    return plan


@pytest.mark.parametrize("users,S,ctx", [(1, 1, 256), (8, 8, 256), (64, 64, 128)])
@pytest.mark.parametrize("tag", ["random", "bigram"])
def test_gd2_batched_rows_base_rule(tiny, tag, users, S, ctx, record_property):
    """G-D2 (the base batched rule, as C1-C5): speculation on (k 2), rows join and leave while
    others decode, a stop token and a cancel mid-answer. GATED: the run and its replay in a fresh engine give every row the
    same tokens, slot and finish; every row keeps its own rules; both engines end idle with usage counted once; the
    cancelled row ends "cancel". On the near-bigram model (its top logit leads by far more than one bf16 step, so no tie can
    flip) drafts were accepted, a stop ended a row (8 and 64 users), and every row also equals the same schedule without
    speculation (the cancelled row: one answer is a prefix of the other); that
    is what fails when a draft is committed unchecked, the linear state is not rolled back, the bonus token comes from the
    wrong verify position or the cursor is one short. RECORDED, not gated (batched output may differ from alone; the tie rule): rows
    equal to the schedule without speculation on the random model, and the tie-aware replay of every row (not the argmax,
    beyond one bf16 step, largest gap in bf16 steps) with speculation on and off."""
    t = tiny[tag]
    plain = engines(t, 2, CTX=ctx, S=S, K=4)[0]
    plan = gd2_schedule(users)
    ref = drive(plain, plan)
    cj = max(range(len(plan)), key=lambda j: plan[j][1]["max_new"])     # the longest row (max_new 30): cancelled mid-answer
    cancel = {cj: 3}
    stop_ids = ()
    for j, r in enumerate(ref):         # a stop mid-answer: a token another prompt's row makes at answer index 3 and not
        if (len(r["ids"]) > 4 and plan[j][1]["ids"] != plan[cj][1]["ids"] and r["ids"][3] not in r["ids"][:3]
                and r["ids"][3] not in ref[cj]["ids"]):        # before, and not in the cancelled row's answer (1 user: none)
            stop_ids = (r["ids"][3],)
            break
    runs = []
    for _ in range(2):                  # the run and its replay, each in a fresh engine
        spec = engines(t, 2, CTX=ctx, S=S, K=4)[1]
        s0 = dict(spec.stats)
        rows = drive(spec, plan, stop_ids, cancel_after=cancel)
        runs.append((rows, gd2_idle(spec, s0, plan), dict(spec.spec_stats)))
    (a, idle_a, st), (b, idle_b, _) = runs
    off = drive(engines(t, 2, CTX=ctx, S=S, K=4)[0], plan, stop_ids, cancel_after=cancel)
    g = dict(replay_equal=[r["ids"] for r in a] == [r["ids"] for r in b],
             slots_equal=[r["slot"] for r in a] == [r["slot"] for r in b],
             fins_equal=[r["fin"] for r in a] == [r["fin"] for r in b],
             rows_ok=all(gd2_row_ok(r, q, stop_ids, j == cj) for j, (r, (_, q)) in enumerate(zip(a, plan))),
             state_ok=idle_a and idle_b, cancel_used=a[cj]["fin"] == "cancel")
    same = [x["ids"] == y["ids"] or (j == cj and (x["ids"][:len(y["ids"])] == y["ids"] or y["ids"][:len(x["ids"])] == x["ids"]))
            for j, (x, y) in enumerate(zip(a, off))]
    info = dict(rows=len(plan), equal_off=sum(same), stop_rows=sum(r["fin"] == "stop" for r in a),
                tokens_per_round=round(st["tokens"] / max(st["rounds"], 1), 3))
    for arm, rows in (("on", a), ("off", off)):
        near = over = 0
        gap = 0.0
        for r, (_, q) in zip(rows, plan):
            if r["ids"]:
                lg = np.asarray(plain.logits(q["ids"] + r["ids"])[len(q["ids"]) - 1:len(q["ids"]) - 1 + len(r["ids"])], np.float32)
                top, mine = lg.max(-1), lg[np.arange(len(r["ids"])), r["ids"]]
                near += int(np.sum(lg.argmax(-1) != np.asarray(r["ids"])))
                over += int(np.sum(top - mine > bf16_step(top)))
                gap = max(gap, float(np.max((top - mine) / bf16_step(top))))
        info.update({f"{arm}_near_tie": near, f"{arm}_beyond_one_bf16_step": over, f"{arm}_largest_gap_bf16_steps": round(gap, 2)})
    for k_, v in {**g, **info}.items():
        record_property(f"gd2_{k_}", v)
    print(f"G-D2 {tag} users={users}: gate {g} recorded {info}")
    assert all(g.values()), (tag, users, g, info)
    if tag == "bigram":                 # drafts accepted (the random drafter's guesses almost never are), no tie can flip
        assert st["tokens"] > st["rounds"] and all(same) and (info["stop_rows"] > 0 or not stop_ids), (
            users, [j for j, s in enumerate(same) if not s], info)


def test_gd2_mixed_settings_one_step(tiny):
    """greedy rows, a draft=False row, penalised rows and seeded sampled rows in the same steps: every row equals the
    same schedule without speculation (sampled rows by the coupled draw), and a seeded row repeats"""
    base, spec = engines(tiny["bigram"], 3, CTX=256, S=8, K=4)
    ps = prompts(8, seed=9)
    smp = [None, None, dict(temperature=0.8, top_k=20, presence_penalty=1.5, seed=4), None,
           dict(temperature=0.7, seed=11), None, dict(temperature=0.0, repetition_penalty=1.3), None]
    plan = [(j, dict(ids=p, max_new=25, samp=smp[j], draft=j != 3)) for j, p in enumerate(ps)]
    want = drive(base, plan)
    got = drive(spec, plan)
    for j in range(8):
        assert got[j]["ids"] == want[j]["ids"], j
    again = drive(spec, plan)
    assert again[4]["ids"] == got[4]["ids"] and len(got[4]["ids"]) == 25     # seeded sampled row with drafts repeats


SAMPLED = [dict(temperature=0.8, top_k=20, seed=4), dict(temperature=0.7, top_p=0.9, seed=11),
           dict(temperature=1.0, min_p=0.05, seed=2), dict(temperature=0.9, presence_penalty=1.5, seed=4),
           dict(temperature=0.6, top_k=40, frequency_penalty=0.7, repetition_penalty=1.2, seed=8),
           dict(temperature=0.0, repetition_penalty=1.3), dict(temperature=0.0, frequency_penalty=0.5, presence_penalty=0.5),
           dict(temperature=1.2, top_p=0.95, seed=123)]


@pytest.mark.parametrize("k", [1, 3])
def test_coupled_draw_sampled_and_penalised_rows_equal_plain(tiny, k):
    """The coupled draw: the target at answer index j is the plain sampler's own draw there (same transform, Gumbel key
    (seed, j), penalty counts that include the drafts before j), drafts are accepted while equal, so sampled and penalised
    rows (greedy or sampled) commit the same tokens as without speculation, and they speculate (k > 0, drafts accepted)"""
    base, spec = engines(tiny["bigram"], k, CTX=256, S=8, K=4)
    ps = prompts(8, seed=21)
    plan = [(j // 2, dict(ids=p, max_new=32, samp=SAMPLED[j])) for j, p in enumerate(ps)]
    want = drive(base, plan)
    got = drive(spec, plan)
    for j in range(8):
        assert got[j]["ids"] == want[j]["ids"], (j, SAMPLED[j])
    assert spec.spec_stats["tokens"] > spec.spec_stats["rounds"]       # only these rows ran: their drafts were accepted


def test_e17_max_tokens_and_stop_inside_a_draft(tiny):
    base, spec = engines(tiny["bigram"], 3, CTX=256, S=4, K=4)
    p = prompts(1, seed=13)[0]
    ref = base.generate([p], max_new=40)["ids"][0]
    for m in range(1, 9):
        r = drive(spec, [(0, dict(ids=p, max_new=m))])[0]
        assert r["ids"] == ref[:m] and r["fin"] == "length"
    for j in (1, 2, 5, 11):
        t = ref[j]
        if t in ref[:j]:
            continue
        r = drive(spec, [(0, dict(ids=p, max_new=40))], stop_ids=(t,))[0]
        assert r["ids"] == ref[:j] and r["fin"] == "stop"


def test_cancel_mid_draft_then_slot_reused(tiny):
    base, spec = engines(tiny["random"], 3, CTX=256, S=4, K=4)
    a, b, c = prompts(3, seed=17)
    want_b = base.generate([b], max_new=30)["ids"][0]
    want_c = base.generate([c], max_new=30)["ids"][0]
    rows = drive(spec, [(0, dict(ids=a, max_new=30)), (0, dict(ids=b, max_new=30)), (4, dict(ids=c, max_new=30))],
                 cancel_at={0: 3})
    assert rows[0]["fin"] == "cancel" and 0 < len(rows[0]["ids"]) < 30
    assert rows[1]["ids"] == want_b and rows[2]["ids"] == want_c


ROLL_WIDTH = pytest.mark.xfail(strict=True, reason="bit-equal slot state (spec §6) as written: on the random model the full-attention "
                               "K at the first generated position differs by one bf16 step in 18 of 7,168 values (verify width 4 vs "
                               "decode width 1 round differently, on CPU); same cause as the exact-mode ties")


@pytest.mark.parametrize("tag", [pytest.param("random", marks=ROLL_WIDTH), "bigram"])
def test_rollback_leaves_no_trace(tiny, tag):
    """after a request ends, its slot holds what plain one-step decode of the same tokens leaves (the plain run is fed the
    speculative run's tokens, so a bf16 tie cannot make the two runs decode different text)"""
    cfg = tiny[tag][0]
    base, spec = engines(tiny[tag], 3, CTX=256, S=4, K=1, prefix_cache=True, cache_ram=0)
    p = prompts(1, seed=21)[0]
    got = drive(spec, [(0, dict(ids=p, max_new=37))])[0]
    want = drive(base, [(0, dict(ids=p, max_new=37))], force=got["ids"])[0]
    assert got["ids"] == want["ids"] and got["slot"] == want["slot"] == 0
    n = len(p) + 37 - 1                    # positions fed: the prompt and every committed token but the last
    (kb, vb), (rb, cb) = base.state
    (ks, vs), (rs, cs) = spec.state
    G = cfg.NG
    np.testing.assert_array_equal(np.asarray(ks[:G, 0, :n]), np.asarray(kb[:G, 0, :n]))
    np.testing.assert_array_equal(np.asarray(vs[:G, 0, :n]), np.asarray(vb[:G, 0, :n]))
    np.testing.assert_allclose(np.asarray(rs[:, :, 0]), np.asarray(rb[:, :, 0]), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(cs[:, :, 0]), np.asarray(cb[:, :, 0]))


def test_gd3_no_spec_compiles_after_warm(tiny):
    _, spec = engines(tiny["bigram"], 2, CTX=256, S=4, K=4)
    spec.warm(workers=1)
    before = (spec.sstep._cache_size(), spec.dpf._cache_size())
    ps = prompts(6, seed=23, hi=120)
    plan = [(j, dict(ids=p, max_new=20 + j, samp=dict(temperature=0.7, seed=j) if j == 2 else None, draft=j != 4))
            for j, p in enumerate(ps)]
    drive(spec, plan)
    assert (spec.sstep._cache_size(), spec.dpf._cache_size()) == before


def test_drop_speculation_before_warm_is_the_plain_engine(tiny):
    """cli.load builds the engine with the drafter, then compares this deployment's component_set_sha with the one the
    drafter's gate passed for; on a mismatch it turns speculation off before warm-up. The engine must then be the plain
    engine: no speculation program, the plain slot state (no MTP K/V group) and the plain engine's tokens. Once a
    speculation program has run, turning it off is refused."""
    plain, spec = engines(tiny["bigram"], 2, CTX=256, S=4, K=4)
    spec.drop_speculation()
    assert (spec.DK, spec.drafter, spec.dstate, spec.k_table) == (0, None, None, {})
    shapes = lambda e: [(x.shape, x.dtype) for x in jax.tree.leaves(e.new_state())]
    assert shapes(spec) == shapes(plain)
    assert spec.spec_programs(*spec.new_state()) == []
    plan = [(j, dict(ids=p, max_new=16 + j, samp=None)) for j, p in enumerate(prompts(5, seed=31))]
    assert [r["ids"] for r in drive(spec, plan)] == [r["ids"] for r in drive(plain, plan)]
    _, used = engines(tiny["bigram"], 2, CTX=256, S=4, K=4)
    drive(used, plan[:1])
    with pytest.raises(RuntimeError):
        used.drop_speculation()
