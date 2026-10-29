"""Speculation for any model (runtime/spec_step.py, runtime/drafters/): the n-gram drafter on every GGUF dense family
(llama, qwen2, qwen3: tiny f32 fixtures from tools/gguf_fixtures) and on the tiny hybrid Qwen3.8 BSHJ model, the dense
verify pass, the drafter choice from metadata, the forced fallback, the per-row backoff and the call-level plain choice.
CPU, no network.

  ST-S1  greedy, one user: every committed token is the plain forward's argmax or within one bf16 step of its top logit
         (the tie rule); on f32 fixtures on == off token for token is also checked
  ST-S2  rows join and leave, stop, cancel and draft=False rows: every row equals the same schedule without speculation,
         at 1, 8 and 64 users
  ST-S3  the 4 step mutants (accept unchecked, no rollback, bonus from the wrong position, cursor one short) are killed
         for the n-gram drafter on every family and the MTP drafter on the hybrid BSHJ model
  ST-S4  rollback leaves no trace: the n-gram state holds exactly the row's prompt and committed tokens
  ST-S5  0 speculation compiles after warm(), and no more compiles of base programs than the plain engine's
  ST-S6  the device matcher equals the host reference on random and repetitive histories; rows never read each other
  ST-S8  a call where no row drafts runs the plain program (counted), with the same output
  ST-S9  the pick reads metadata only (no name), in the order mtp, draft_model, ngram; MoE and no-verify refused
  ST-S10 the drafter key carries the speculation step version; a gate record turns speculation on only for this
         component_set_sha, this step version and k > 0 at a row bucket, and the GGUF engine checks it before the warm-up
  ST-S11 MASHBOX_FORCE_FAIL=speculation, a drafter that cannot be built and a speculation program that fails before
         ready give plain decoding (output equal to the plain engine's) with a why line
  ST-S12 per-row backoff: a row whose drafts keep failing drops to k = 0 (4 misses, 16 rounds, doubling to 256, reset
         by an accepted draft); another row's rounds are its own; output unchanged
  MoE    a MoE file is refused (cfg.E or cfg.moe), forced n-gram included
  server one GGUF request with --speculation ngram through the real server equals the plain server's
Slow rows (the step mutants, 64 users) run with MASHBOX_SPEC_SLOW=1, where speculation code changes (SPEC_SLOW below).
"""
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import gguf  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mashbox.runtime.engine import Engine  # noqa: E402
from mashbox.runtime import drafter as DR, drafters as DRS, spec_step as SS  # noqa: E402
from mashbox.runtime.drafters import ngram as NG, select as SEL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_fx = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_fx)
_fx.loader.exec_module(FXT)
FAMILIES = ("llama-f32", "qwen2-f32", "qwen3-f32", "qwen35-f32")     # full attention, and hybrid (gated delta + conv)
# near-bigram weights for the hybrid fixture (every residual write x 0.05, as tests/standards' "bigram" weights): with
# its random weights the greedy text never repeats a 3-gram of these prompts, so prompt lookup would never draft there
BIGRAM, LO = ("qwen35-f32",), 0.05
RESIDUAL = (".attn_output.weight", ".ssm_out.weight", ".ffn_down.weight")
PATHS = {}
_COMPILES = [0]
# The slow speculation rows (ST-S3's step mutants, ST-S2 at 64 users) run where speculation code changes:
# runtime/spec_step.py, runtime/drafters/, runtime/drafter.py or the speculation parts of runtime/engine.py (plan 4.3:
# mutants only when speculation paths change). MASHBOX_SPEC_SLOW=1 runs them; every other run skips them with this reason.
SPEC_SLOW = [pytest.mark.slow, pytest.mark.skipif(os.environ.get("MASHBOX_SPEC_SLOW") != "1", reason=(
    "speculation slow row: runs with MASHBOX_SPEC_SLOW=1 where runtime/spec_step.py, runtime/drafters/, "
    "runtime/drafter.py or engine.py speculation code changes"))]
jax.monitoring.register_event_duration_secs_listener(
    lambda ev, d, **k: _COMPILES.__setitem__(0, _COMPILES[0] + (ev == "/jax/core/compile/backend_compile_duration")))


def scaled(path):
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        a = gguf.quants.dequantize(t.data, t.tensor_type).reshape(tuple(reversed([int(x) for x in t.shape])))
        out[t.name] = (a * LO if t.name.endswith(RESIDUAL) else a).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def fams(tmp_path_factory):
    from mashbox.gguf import open_model
    out = {}
    wd = tmp_path_factory.mktemp("drafters")
    for name in FAMILIES:
        path = FXT.build(name, str(wd))
        if name in BIGRAM:
            path = FXT.write(name, str(wd / f"{name}.bigram.gguf"), override=scaled(path), quant=False)
        PATHS[name] = path
        f, A, c = open_model(path, act=jnp.float32)
        W, _ = A.load(f, c, use_kernels=False)
        out[name] = (c, W, A)
    return out


def engines(t, k=3, **kw):
    c, W, A = t
    kw = dict(dict(CTX=128, S=4, K=4), **kw)
    return Engine(c, W, A, **kw), Engine(c, W, A, drafter=NG.NGram(k), draft_k=k, spec_rounds=3, **kw)


def no_fault(eng, faults_ok=False):
    """A speculation call that raises before it took the slot state is replayed plain and turns speculation off
    (Engine._spec_fault), so the rows still equal the plain engine's: an output check alone cannot see it. Every row
    that does not force a failure checks that none happened."""
    st = getattr(eng, "spec_stats", None) or {}
    assert faults_ok or not st.get("faults"), f"a speculation call failed and was replayed plain: {eng.spec_why}"


def drive(eng, plan, stop_ids=(), cancel_at=None, faults_ok=False):
    rows, nxt, i = [None] * len(plan), 0, 0
    while nxt < len(plan) or eng.busy():
        while nxt < len(plan) and plan[nxt][0] <= i and eng.free_slots():
            rows[nxt] = eng.admit([plan[nxt][1]], stop_ids)[0]
            nxt += 1
        for j, at in (cancel_at or {}).items():
            if at == i and rows[j] is not None:
                eng.cancel([rows[j]])
        if eng.busy():
            eng.step(waiting=nxt < len(plan))
        i += 1
        assert i < 5000
    no_fault(eng, faults_ok)
    return rows


def loop_prompts(V, n, seed):
    """prompts that repeat a short phrase (as code and agent turns repeat text), so prompt lookup finds matches"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ph = rng.integers(1, V, int(rng.integers(3, 7))).tolist()
        out.append((rng.integers(1, V, int(rng.integers(2, 6))).tolist() + ph * int(rng.integers(2, 4)))[:40])
    return out


def bf16_step(x):
    return 2.0 ** (np.floor(np.log2(np.maximum(np.abs(x), 1e-30))) - 7)


def tie_aware(base, p, got):
    lg = np.asarray(base.logits(p + got)[len(p) - 1:len(p) - 1 + len(got)], np.float32)
    top, mine = lg.max(-1), lg[np.arange(len(got)), got]
    return int(np.sum(lg.argmax(-1) != np.asarray(got))), int(np.sum(top - mine > bf16_step(top)))


# ---------------------------------------------------------------------------------------------------------------- matcher
def test_ngram_matcher_rules():
    """longest n first, most recent match, continuation at or before the cursor, no match gives nothing"""
    m = NG.match
    assert m([1, 2, 3, 4, 9, 1, 2, 3, 4], 3) == [9, 1, 2]                   # the 4-gram 1 2 3 4 recurs: its continuation
    assert m([5, 1, 2, 3, 7, 1, 2, 3, 8, 6, 1, 2, 3], 2) == [8, 6]          # most recent of two 3-gram matches
    assert m([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], 3) == [1, 2, 3]                # 5-gram before 4-gram
    assert m([1, 2, 3, 1, 2, 3], 4) == [1, 2, 3]                            # continuation stops at the cursor
    assert m([1, 2, 3, 4, 5, 6], 3) == [] and m([1, 2], 3) == []           # no match, too short
    assert m([7, 7, 7, 7], 2) == [7]                                         # overlapping window: only up to the cursor


def device_drafts(hist_rows, toks, poss, K):
    dr = NG.NGram(K)
    d, kd, _ = jax.jit(lambda h, t, p: dr.draft(None, None, None, h, t, p, 0, K, None, None))(
        jnp.asarray(hist_rows), jnp.asarray(toks), jnp.asarray(poss))
    return np.asarray(d), np.asarray(kd)


def test_ngram_device_equals_host_reference():
    """10,000 histories (random over small vocabularies, and repeated phrases): the device drafts equal match() token for
    token; changing another row never changes a row's drafts"""
    rng = np.random.default_rng(0)
    C, B, K = 64, 100, 4
    n = 0
    for it in range(100):
        rows, toks, poss, want = np.full((B, C), -1, np.int32), np.zeros(B, np.int32), np.zeros(B, np.int32), []
        for b in range(B):
            L = int(rng.integers(1, C))
            if it % 2:
                ph = rng.integers(0, 50, int(rng.integers(2, 8)))
                seq = np.concatenate([rng.integers(0, 50, int(rng.integers(0, 6))), np.tile(ph, 12)])[:L]
            else:
                seq = rng.integers(0, int(rng.integers(2, 6)), L)
            seq = np.asarray(seq, np.int32)
            rows[b, :len(seq) - 1], toks[b], poss[b] = seq[:-1], seq[-1], len(seq) - 1
            want.append(NG.match(seq.tolist(), K))
        d, kd = device_drafts(rows, toks, poss, K)
        for b in range(B):
            assert d[b, :kd[b]].tolist() == want[b], (it, b)
            n += 1
        if it == 0:         # rows are independent: scramble row 1 and row 0's drafts stay
            r2 = rows.copy()
            r2[1] = rng.integers(0, 3, C)
            d2, kd2 = device_drafts(r2, toks, poss, K)
            assert d2[0].tolist() == d[0].tolist() and kd2[0] == kd[0]
    assert n == 10000


# ---------------------------------------------------------------------------------------------------------------- engine
# k = 3 on every family, and k = 1 .. 3 on the hybrid family (near-bigram weights, so drafts are accepted at every k)
S1_ROWS = [(f, 3) for f in FAMILIES] + [("qwen35-f32", 1), ("qwen35-f32", 2)]


@pytest.mark.parametrize("name,k", S1_ROWS)
def test_st_s1_lone_greedy_tie_rule_and_on_equals_off(fams, name, k, record_property):
    base, spec = engines(fams[name], k=k)
    c = fams[name][0]
    n = near = over = acc = 0
    for p in loop_prompts(c.V, 3, 11):
        want = drive(base, [(0, dict(ids=p, max_new=48))])[0]["ids"]
        got = drive(spec, [(0, dict(ids=p, max_new=48))])[0]["ids"]
        assert len(got) == 48
        a, b = tie_aware(base, p, got)
        n, near, over = n + len(got), near + a, over + b
        assert got == want          # f32 fixture: on == off token for token as well
    acc = sum(spec.spec_stats["hist"][2:])
    record_property("ngram_tokens", n)
    record_property("ngram_rounds_with_accepts", acc)
    print(f"ST-S1 {name} k={k}: {n} tokens, near-tie {near}, beyond one bf16 step {over}, stats {spec.spec_stats}")
    assert over == 0 and near == 0
    assert acc > 0, "no draft was ever accepted: the drafter ran without effect"


@pytest.mark.parametrize("name", FAMILIES)
def test_st_s2_batched_schedule_equals_plain(fams, name):
    """rows join and leave mid-batch, one stops on a token, one is cancelled, one has draft=False, one is sampled with
    a presence penalty (the coupled draw): every row equals the same schedule without speculation (f32), both engines
    end idle"""
    base, spec = engines(fams[name])
    c = fams[name][0]
    ps = loop_prompts(c.V, 7, 21)
    plan = [(j, dict(ids=p, max_new=20 + 3 * j, draft=j != 3)) for j, p in enumerate(ps)]
    plan[5][1]["samp"] = dict(temperature=0.8, seed=5, presence_penalty=0.6)
    stop = (int(ps[0][-1]),)
    a = drive(base, plan, stop, cancel_at={1: 3})
    b = drive(spec, plan, stop, cancel_at={1: 3})
    assert [(r["ids"], r["fin"]) for r in a] == [(r["ids"], r["fin"]) for r in b]
    assert not spec.busy() and not base.busy() and b[1]["fin"] == "cancel"


@pytest.mark.parametrize("name", FAMILIES)
def test_st_s4_rollback_leaves_no_trace(fams, name):
    """after speculation the n-gram state of each slot holds the row's prompt and committed tokens before the one being
    fed, and -1 after: rejected drafts were never written"""
    _, spec = engines(fams[name])
    c = fams["qwen3-f32"][0]
    rows = [spec.admit([dict(ids=p, max_new=90)])[0] for p in loop_prompts(c.V, 3, 5)]
    checked = 0
    for _ in range(40):
        spec.step()
        hist = np.asarray(spec.dstate)
        for r in rows:
            if r["fin"] is not None or not r.get("hist_ok"):     # a plain call since: rebuilt before the next spec call
                continue
            seq = list(r["prompt"]) + list(r["ids"][:-1])
            assert hist[r["slot"], :len(seq)].tolist() == seq
            assert (hist[r["slot"], len(seq):] == -1).all()
            checked += 1
    assert checked > 0 and spec.spec_stats["rounds"] > 0 and sum(spec.spec_stats["hist"][2:]) > 0


def test_st_s8_plain_call_when_no_row_drafts(fams):
    """a prompt with no repeated 3-gram and a short answer: no row finds a draft, so calls run the plain program
    (counted), and the output equals the plain engine's"""
    base, spec = engines(fams["qwen2-f32"])
    p = list(range(1, 30))
    want = drive(base, [(0, dict(ids=p, max_new=3))])[0]["ids"]
    got = drive(spec, [(0, dict(ids=p, max_new=3))])[0]["ids"]
    assert got == want and spec.spec_stats["plain_calls"] > 0


def test_st_s12_backoff_rows_own_history(fams):
    """a row whose drafts keep failing drops to k = 0 by the backoff rule (counted); its output is unchanged"""
    base, spec = engines(fams["llama-f32"])
    spec._k_row_calls = []
    c = fams["llama-f32"][0]
    p = loop_prompts(c.V, 1, 2)[0]
    real = spec.drafter.found
    spec.drafter.found = lambda eng, r, K: 1         # the call-level plain choice never skips: rounds always verify
    spec.drafter.draft = (lambda f: lambda *a: (lambda d, kd, ck: (jnp.full_like(d, c.V - 1), jnp.full_like(kd, 3), ck))(
        *f(*a)))(spec.drafter.draft)                  # drafts that are never the target's token
    spec.sstep = jax.jit(SS.make_step(spec.model.make_forward(c), SS.arch_commit(spec.model, c), spec.drafter),
                         static_argnames=("Tc", "K", "N", "trace"), donate_argnums=(1, 2))
    got = drive(spec, [(0, dict(ids=p, max_new=60))])[0]["ids"]
    spec.drafter.found = real
    assert got == drive(base, [(0, dict(ids=p, max_new=60))])[0]["ids"]
    assert spec.spec_stats["backoffs"] >= 1


def test_ngram_on_hybrid_bshj_model(tmp_path):
    """the same drafter on the hybrid (linear-attention) Qwen3.8 tiny model: its commit replays only the accepted
    steps, so greedy output equals the plain engine's on the near-bigram weights"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_spec_in_step as T
    cfg_dir = tmp_path
    import json
    json.dump(dict(model_type="qwen3_5_text", vocab_size=T.V, hidden_size=T.D, intermediate_size=T.F,
                   num_hidden_layers=T.L, num_attention_heads=T.H, num_key_value_heads=T.KV, head_dim=T.HD,
                   linear_num_key_heads=T.HK, linear_num_value_heads=T.HV, linear_key_head_dim=T.DK,
                   linear_value_head_dim=T.DV, linear_conv_kernel_dim=T.CK, full_attention_interval=4, rms_norm_eps=1e-6,
                   rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25),
                   tie_word_embeddings=False, attn_output_gate=True, mtp_num_hidden_layers=1),
              open(cfg_dir / "config.json", "w"))
    cfg = T.Q.Config(str(cfg_dir))
    path = str(tmp_path / "bigram.bshj")
    T.write_bshj(path, T.tensors(11, 0.05))
    W, _ = T.QW.load(path, cfg)
    base = Engine(cfg, W, T.Q, CTX=128, S=2, K=4)
    spec = Engine(cfg, W, T.Q, CTX=128, S=2, K=4, drafter=NG.NGram(3), draft_k=3, spec_rounds=3)
    for p in loop_prompts(T.V, 2, 7):
        assert drive(spec, [(0, dict(ids=p, max_new=40))])[0]["ids"] == drive(base, [(0, dict(ids=p, max_new=40))])[0]["ids"]
    assert spec.spec_stats["rounds"] > 0


# ---------------------------------------------------------------------------------------------------------------- choice
def test_st_s9_pick_from_metadata_only(fams):
    c, W, A = fams["qwen3-f32"]
    m = SEL.meta_of(A, c, "gguf", 0)
    assert set(m) == {"format", "verify", "mtp_head", "experts", "mtp_layers", "draft_given", "trained", "trained_why",
                      "draft_models"} \
        and m["verify"] and not m["mtp_head"]
    assert m["trained"] == m["trained_why"] == m["draft_models"] == []     # detection: nothing declared for this file
    from mashbox.models import qwen3_8_27b as Q
    assert SEL.meta_of(Q, SimpleNamespace(), "bshj", 1)["mtp_head"]
    css = "c" * 64
    ok = dict(status="passed", record="r", record_sha256="e" * 64, component_set_sha=css, spec_step_version=SS.SPEC_STEP_VERSION, k_table={"8": 3})
    rec = {"ngram": {"v6e-1": ok}}
    assert SEL.pick(m, "v6e-1", rec, css=css)[:2] == ("ngram", True)
    kind, g, why, _ = SEL.pick(m, "v6e-1", None)
    assert kind is None and any("not gated on this chip (v6e-1)" in w for w in why)
    assert SEL.pick(m, None, None, forced="ngram")[:2] == ("ngram", False)
    assert SEL.pick(m, "v6e-1", None, forced="off")[0] is None
    # an MTP head on an architecture with one registered comes first; on GGUF it is not built, so n-gram follows
    q = dict(m, mtp_head=True, mtp_layers=1)
    rec2 = {"mtp": {"v6e-1": ok}, "ngram": {"v6e-1": ok}}
    assert SEL.pick(q, "v6e-1", rec2, css=css)[0] == "mtp"
    g2 = dict(m, mtp_layers=1)
    k2, _, why2, _ = SEL.pick(g2, "v6e-1", rec2, css=css)
    assert k2 == "ngram" and why2[0].startswith("trained: no trained drafter declares this target")   # trained is ranked first
    assert why2[1].startswith("mtp: the file has an MTP head, but no MTP implementation")
    assert SEL.pick(dict(m, experts=8), "v6e-1", rec)[0] is None
    assert SEL.pick(dict(m, verify=False), "v6e-1", rec)[0] is None


def test_no_model_name_in_speculation_layer():
    """drafters are chosen from metadata and records: no model family or file name in the speculation layer"""
    import re
    here = os.path.join(ROOT, "src", "mashbox", "runtime")
    files = [os.path.join(here, "spec_step.py")] + [os.path.join(here, "drafters", f)
                                                     for f in sorted(os.listdir(os.path.join(here, "drafters"))) if f.endswith(".py")]
    bad = re.compile(r"qwen|llama|gemma|mistral|deepseek|phi-?\d|hy-?nomas|mimo|glm|kimi", re.I)
    for f in files:
        for i, line in enumerate(open(f), 1):
            assert not bad.search(line), f"{f}:{i}: {line.strip()}"


def test_st_s10_key_carries_step_version():
    dr = dict(kind="mtp", id="m", version=1, sha256="ab", k_default=2, k_max=3)
    assert DR.key(dr)["spec_step_version"] == SS.SPEC_STEP_VERSION
    assert NG.NGram(3).key()["spec_step_version"] == SS.SPEC_STEP_VERSION and NG.NGram(3).key()["weights_sha256"] is None


def test_st_s11_forced_failure_and_build_failure_decode_plain(fams, monkeypatch):
    from mashbox import cli
    c, W, A = fams["llama-f32"]
    lines = []
    monkeypatch.setattr(cli, "log", lines.append)
    a = SimpleNamespace(speculation="ngram", draft_k=2)
    monkeypatch.setenv("MASHBOX_FORCE_FAIL", "speculation")
    assert cli.gguf_speculation(a, c, A) == {} and "forced failure" in lines[-1]
    monkeypatch.delenv("MASHBOX_FORCE_FAIL")
    got = cli.gguf_speculation(a, c, A)
    assert got["draft_k"] == 2 and got["drafter"].kind == "ngram" and "not gated on this chip" in lines[-2]
    monkeypatch.setattr(NG, "NGram", lambda k: (_ for _ in ()).throw(RuntimeError("no")))
    assert cli.gguf_speculation(a, c, A) == {} and "could not be built" in lines[-1]
    assert cli.gguf_speculation(SimpleNamespace(speculation=None, draft_k=None), c, A) == {}     # auto, no record: off
    assert "not gated" in lines[-1] or "not on a TPU" in lines[-1]


# ---------------------------------------------------------------------------------------------------------- added in r2
def test_ngram_find_equals_match():
    """find() (NumPy windows, run on the host for each live row before a call) equals match() on 3,000 histories"""
    rng = np.random.default_rng(3)
    for it in range(3000):
        L = int(rng.integers(1, 300))
        if it % 2:
            seq = np.concatenate([rng.integers(0, 40, int(rng.integers(0, 9))), np.tile(rng.integers(0, 40, int(rng.integers(2, 9))), 60)])[:L]
        else:
            seq = rng.integers(0, int(rng.integers(2, 6)), L)
        K = int(rng.integers(1, 6))
        assert NG.find(seq, K) == NG.match(seq.tolist(), K), it


@pytest.mark.parametrize("users", (1, 8, pytest.param(64, marks=SPEC_SLOW)))
@pytest.mark.parametrize("name", ("llama-f32", "qwen35-f32"))
def test_st_s2_users(fams, name, users):
    """the batched schedule at 1, 8 and 64 users (rows join over 5 steps, one in 9 with draft=False): every row equals
    the plain engine's. 64 users is a slow row (SPEC_SLOW): both 64-slot engines compile every row bucket up to 64. The
    row lengths stay long enough for prompt lookup to draft (rows of 6 to 10 tokens ran 0 speculation rounds at 64
    users on the hybrid family, so the check would hold without speculation running)"""
    c = fams[name][0]
    base, spec = engines(fams[name], S=users)
    plan = [(j % 5, dict(ids=p, max_new=40 if users == 1 else 12 + j % 7, draft=j % 9 != 4))
            for j, p in enumerate(loop_prompts(c.V, users, 100 + users))]
    assert [r["ids"] for r in drive(spec, plan)] == [r["ids"] for r in drive(base, plan)]
    assert spec.spec_stats["rounds"] > 0 and not spec.busy()


@pytest.mark.parametrize("name", [FAMILIES[0]] + [pytest.param(f, marks=SPEC_SLOW) for f in FAMILIES[1:]])
def test_st_s5_no_spec_compiles_after_warm(fams, name):
    """0 speculation programs compiled after warm(), and no more compiles in all (base helpers included) than the plain
    engine has on the same schedule: switching between the plain and the speculation program compiles nothing new"""
    base, spec = engines(fams[name])
    base.warm(workers=1)
    spec.warm(workers=1)
    c = fams["llama-f32"][0]
    plan = [(j, dict(ids=p, max_new=12 + 5 * j, samp=dict(temperature=0.7, seed=j) if j == 2 else None, draft=j != 4))
            for j, p in enumerate(loop_prompts(c.V, 6, 9))]
    c0 = _COMPILES[0]
    drive(base, plan)
    plain = _COMPILES[0] - c0
    before, c1 = (spec.sstep._cache_size(), spec.drafter.put._cache_size()), _COMPILES[0]
    drive(spec, plan)
    total = _COMPILES[0] - c1
    print(f"ST-S5: compiles after warm: speculation engine {total} (speculation programs "
          f"{spec.sstep._cache_size() - before[0] + spec.drafter.put._cache_size() - before[1]}), plain engine {plain}; "
          f"plain calls {spec.spec_stats['plain_calls']}, rounds {spec.spec_stats['rounds']}")
    assert (spec.sstep._cache_size(), spec.drafter.put._cache_size()) == before
    assert spec.spec_stats["plain_calls"] > 0 and spec.spec_stats["rounds"] > 0
    assert total <= plain


def test_st_s10_gate_needs_component_set_step_version_and_k(fams, monkeypatch):
    from mashbox import cli
    from mashbox.gguf import open_model
    css = "c" * 64
    ok = dict(status="passed", record="r", record_sha256="e" * 64, component_set_sha=css, spec_step_version=SS.SPEC_STEP_VERSION, k_table={"1": 3, "8": 0})
    assert SEL.gated({"v6e-1": ok}, "v6e-1", css)[0]
    for bad, word in ((dict(ok, component_set_sha="d" * 64), "another component set"),
                      (dict(ok, spec_step_version=SS.SPEC_STEP_VERSION + 1), "step version"),
                      (dict(ok, k_table={"1": 0, "8": 0}), "k = 0"), (dict(ok, k_table=None), "k = 0"),
                      (dict(ok, status="failed"), "not gated")):
        g, why = SEL.gated({"v6e-1": bad}, "v6e-1", css)
        assert not g and word in why, why
    assert not SEL.gated({"v6e-1": ok}, "v6e-1", None)[0] and not SEL.gated({"v6e-1": ok}, None, css)[0]
    # GGUF: the record's component set stands in for the pick; the built engine's own one decides before the warm-up
    c, W, A = fams["llama-f32"]
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    lines = []
    monkeypatch.setattr(cli, "log", lines.append)
    a = SimpleNamespace(speculation=None, draft_k=None)
    got = cli.gguf_speculation(a, c, A, records={"ngram": {"v6e-1": ok}})
    assert got.pop("gate_css") == css and got["drafter"].kind == "ngram"
    assert got["k_table"] == {1: 3, 8: 0} and got["draft_k"] == 3      # the record's k per row bucket reaches the engine
    redo = []
    eng = Engine(c, W, A, CTX=128, S=2, K=4, **got)
    assert eng.k_table == {1: 3, 8: 0} and eng.DK == 3
    eng.component_set_sha = "d" * 64
    cli.gguf_spec_gate(eng, css, lambda: redo.append(1))
    assert eng.DK == 0 and eng.drafter is None and redo == [1] and "another component set" in lines[-1]
    eng2 = Engine(c, W, A, CTX=128, S=2, K=4, drafter=NG.NGram(3), draft_k=3)
    eng2.component_set_sha = css
    cli.gguf_spec_gate(eng2, css, lambda: redo.append(2))
    assert eng2.DK == 3 and redo == [1]
    assert cli.gguf_speculation(a, c, A, records={"ngram": {"v6e-1": dict(ok, spec_step_version=0)}}) == {}
    # the GGUF deployment's component set holds the drafter's key
    f, _, _ = open_model(PATHS["llama-f32"], act=jnp.float32)
    r = dict(path=PATHS["llama-f32"], repo=None, file=os.path.basename(PATHS["llama-f32"]))
    plain, on = Engine(c, W, A, CTX=128, S=2, K=4), Engine(c, W, A, CTX=128, S=2, K=4, drafter=NG.NGram(3), draft_k=3)
    cli.gguf_deployment(plain, f, r, W)
    cli.gguf_deployment(on, f, r, W)
    assert on.component_set_sha != plain.component_set_sha


def test_forced_ngram_on_a_catalog_model_checks_applies(monkeypatch):
    """--speculation ngram on a models.json model runs only where select.applies says so (a verify pass, not MoE)"""
    from mashbox import cli
    from mashbox.models import qwen3_8_27b as Q
    monkeypatch.setattr(cli, "log", lambda *a: None)
    assert cli.forced_drafter("ngram", 3, 1, Q, SimpleNamespace())["drafter"].kind == "ngram"
    with pytest.raises(SystemExit, match="MoE"):
        cli.forced_drafter("ngram", 3, 1, Q, SimpleNamespace(E=8))
    with pytest.raises(SystemExit, match="no verify pass"):
        cli.forced_drafter("ngram", 3, 1, SimpleNamespace(), SimpleNamespace())


def broken_step(eng, c):
    def draft(*a, **k):
        raise RuntimeError("speculation program failed to compile (forced)")
    eng.drafter.draft = draft
    eng.sstep = jax.jit(SS.make_step(eng.model.make_forward(c, None), SS.arch_commit(eng.model, c), eng.drafter),
                        static_argnames=("Tc", "K", "N", "trace"), donate_argnums=(1, 2))


def test_st_s11_compile_error_before_ready_decodes_plain(fams, monkeypatch):
    """a speculation program that fails before ready (cli.warm): drop_speculation, the plain warm-up, plain output"""
    from mashbox import cli
    c = fams["llama-f32"][0]
    base, spec = engines(fams["llama-f32"])
    lines = []
    monkeypatch.setattr(cli, "log", lines.append)
    broken_step(spec, c)
    p0 = loop_prompts(c.V, 1, 4)[0]
    cli.warm(spec, SimpleNamespace(chat_ids=lambda msgs: p0))
    assert spec.DK == 0 and spec.drafter is None and any("failed before ready" in x for x in lines)
    for p in loop_prompts(c.V, 3, 12):
        assert drive(spec, [(0, dict(ids=p, max_new=24))])[0]["ids"] == drive(base, [(0, dict(ids=p, max_new=24))])[0]["ids"]



def test_spec_health_reason_after_a_program_failed_before_ready(fams, monkeypatch):
    """The reason (spec_why) after a speculation program failed before ready: cli.warm's why line (the error named), and
    the engine's own reason for drop_speculation(failed=True) when no cli sets one"""
    from mashbox import cli
    c = fams["llama-f32"][0]
    _, spec = engines(fams["llama-f32"])
    lines = []
    monkeypatch.setattr(cli, "log", lines.append)
    broken_step(spec, c)
    p0 = loop_prompts(c.V, 1, 4)[0]
    cli.warm(spec, SimpleNamespace(chat_ids=lambda msgs: p0))
    why = [json.loads(x)["reason"] for x in lines if x.startswith('{"event": "speculation"')]
    h = spec.spec_health()
    assert h["on"] is False and why and h["reason"] == why[-1], (h, why)
    assert h["reason"].startswith("a speculation program failed before ready: ")
    _, eng = engines(fams["llama-f32"])
    eng.drop_speculation(failed=True)
    assert eng.spec_health()["reason"] == "speculation off: a speculation program failed before ready"

# ST-S3: each mutant changes one line of runtime/spec_step.py; a mutant is killed when a row's output differs from the
# plain engine's on a batched schedule or the n-gram state holds a token the row did not commit
MUTANTS = {
    "accept unchecked": [("a = jnp.minimum(jnp.sum(jnp.cumprod((d == tgt[:, :K]).astype(i32), 1), 1), kc)", "a = kc")],
    "no rollback": [('cv = commit(r["cv"], r["rec"], n)', 'cv = commit(r["cv"], r["rec"], jnp.where(rem > 0, K1, 0))'),
                    ("Dc = drafter.after_commit(Dc, r, emit, n, go, tok, pos, K1)",
                     "Dc = drafter.after_commit(Dc, r, emit, jnp.where(rem > 0, K1, 0), go, tok, pos, K1)")],
    "bonus from the wrong position": [("bonus = jnp.take_along_axis(tgt, a[:, None], 1)[:, 0]",
                                       "bonus = jnp.take_along_axis(tgt, jnp.minimum(a + 1, K)[:, None], 1)[:, 0]")],
    "cursor one short": [("pos + n, rem - n,", "pos + jnp.maximum(n - 1, 0), rem - n,")],
}


def mutant_module(name):
    import types
    src = open(SS.__file__).read()
    for old, new in MUTANTS[name]:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    m = types.ModuleType("mashbox.runtime.spec_step_mutant")
    m.__package__ = "mashbox.runtime"
    exec(compile(src, f"{SS.__file__} [{name}]", "exec"), m.__dict__)
    return m


@pytest.fixture(scope="module")
def bshj(tmp_path_factory):
    import json
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_spec_in_step as T
    wd = tmp_path_factory.mktemp("drafters_bshj")
    json.dump(dict(model_type="qwen3_5_text", vocab_size=T.V, hidden_size=T.D, intermediate_size=T.F,
                   num_hidden_layers=T.L, num_attention_heads=T.H, num_key_value_heads=T.KV, head_dim=T.HD,
                   linear_num_key_heads=T.HK, linear_num_value_heads=T.HV, linear_key_head_dim=T.DK,
                   linear_value_head_dim=T.DV, linear_conv_kernel_dim=T.CK, full_attention_interval=4, rms_norm_eps=1e-6,
                   rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25),
                   tie_word_embeddings=False, attn_output_gate=True, mtp_num_hidden_layers=1), open(wd / "config.json", "w"))
    cfg = T.Q.Config(str(wd))
    out = {}
    for tag, lo in (("random", 1.0), ("bigram", 0.05)):
        path = str(wd / f"{tag}.bshj")
        T.write_bshj(path, T.tensors(11, lo))
        out[tag] = (T.QW.load(path, cfg)[0], T.QS.load_mtp(path, cfg)[0])
    return T, cfg, out


# one combination per drafter and architecture module (gguf/archs/dense.py runs llama, qwen2 and qwen3, whose families
# ST-S1 and ST-S2 cover). The BSHJ model runs its random weights: on its near-bigram ones every n-gram draft is right,
# so a step that accepts drafts unchecked would give the same text there
# slow (SPEC_SLOW): 4 combinations x 6 engines (plain, clean, 4 mutants), each compiling its programs
COMBOS = [pytest.param(*x, marks=SPEC_SLOW) for x in
          (("ngram", "llama-f32"), ("ngram", "qwen35-f32"), ("ngram", "qwen38-bshj"), ("mtp", "qwen38-bshj"))]


@pytest.mark.parametrize("drafter,family", COMBOS)
def test_st_s3_step_mutants_are_killed(fams, bshj, drafter, family):
    kw = dict(CTX=128, S=4, K=4)
    if family == "qwen38-bshj":
        T, c, ws = bshj
        W, Wm = ws["random"]
        model, V = T.Q, T.V
        dk = dict(mtp=(T.QS, Wm)) if drafter == "mtp" else dict(drafter=NG.NGram(3))
        base, make = Engine(c, W, model, **kw), lambda: Engine(c, W, model, draft_k=3, spec_rounds=3, **dict(kw, **dk))
    else:
        c, W, model = fams[family]
        V = c.V
        base, make = Engine(c, W, model, **kw), lambda: Engine(c, W, model, drafter=NG.NGram(3), draft_k=3, spec_rounds=3, **kw)
    plan = [(j, dict(ids=p, max_new=48)) for j, p in enumerate(loop_prompts(V, 3, 11))]
    want = [r["ids"] for r in drive(base, plan)]
    clean = make()
    assert [r["ids"] for r in drive(clean, plan)] == want
    assert clean.spec_stats["rounds"] > 0
    survived = []
    for name in MUTANTS:
        M = mutant_module(name)
        eng = make()
        eng.sstep = jax.jit(M.make_step(eng.model.make_forward(c, None), M.arch_commit(eng.model, c), eng.drafter),
                            static_argnames=("Tc", "K", "N", "trace"), donate_argnums=(1, 2))
        try:
            got = [r["ids"] for r in drive_hist(eng, plan)]
            why = "output differs" if got != want else None
        except AssertionError as e:
            why = f"state check: {str(e)[:80]}"
        print(f"ST-S3 {drafter} x {family}: {name}: {'killed (' + why + ')' if why else 'SURVIVED'}")
        if why is None:
            survived.append(name)
    assert not survived, f"mutants not killed: {survived}"


def drive_hist(eng, plan):
    """drive(), checking after each step that the n-gram state of every row holds exactly its prompt and committed
    tokens before the one being fed (ST-S4's rule)"""
    rows, nxt, i = [None] * len(plan), 0, 0
    while nxt < len(plan) or eng.busy():
        while nxt < len(plan) and plan[nxt][0] <= i and eng.free_slots():
            rows[nxt] = eng.admit([plan[nxt][1]], ())[0]
            nxt += 1
        if eng.busy():
            eng.step(waiting=nxt < len(plan))
        if eng.drafter is not None and eng.drafter.kind == "ngram":
            hist = np.asarray(eng.dstate)
            for r in rows:
                if r is not None and r["fin"] is None and r.get("hist_ok"):
                    seq = list(r["prompt"]) + list(r["ids"][:-1])
                    assert hist[r["slot"], :len(seq)].tolist() == seq and (hist[r["slot"], len(seq):] == -1).all(), "hist"
        i += 1
        assert i < 5000
    no_fault(eng)
    return rows


# ---------------------------------------------------------------------------------------------------------- added in r3
def test_st_s12_backoff_wait_sequence():
    """the POLICY numbers: 4 rounds in a row with drafts and 0 accepted -> 16 rounds at k = 0; each further failure
    doubles the wait up to 256; an accepted draft resets it (Engine._backoff, the row's own record only)"""
    assert (Engine.BO_MISS, Engine.BO_WAIT, Engine.BO_MAX) == (4, 16, 256)
    eng = SimpleNamespace(drafter=SimpleNamespace(backoff=True), BO_MISS=Engine.BO_MISS, BO_WAIT=Engine.BO_WAIT,
                          BO_MAX=Engine.BO_MAX, spec_stats=dict(backoffs=0))
    r, waits = {}, []
    for _ in range(7):
        for _ in range(4):
            Engine._backoff(eng, r, 1, 3)          # a round with drafts, none accepted: 1 token committed
        waits.append(r["bo"]["wait"])
        while r["bo"]["wait"]:
            Engine._backoff(eng, r, 1, 0)          # a waiting round (k = 0)
    assert waits == [16, 32, 64, 128, 256, 256, 256] and eng.spec_stats["backoffs"] == 7
    Engine._backoff(eng, r, 1, 3)
    Engine._backoff(eng, r, 3, 3)                  # 2 drafts accepted: the wait goes back to 16
    for _ in range(4):
        Engine._backoff(eng, r, 1, 3)
    assert r["bo"]["wait"] == 16
    other = {}
    Engine._backoff(eng, other, 1, 3)
    assert other["bo"] == dict(miss=1, wait=0, span=16)     # another row starts from its own state


def test_st_s12_backoff_is_per_row(fams):
    """two rows in one batch: row 0's drafts never match, so it backs off (rounds at k = 0); row 1 drafts by prompt
    lookup as it does when row 0's drafts are its own. Row 1's (committed, cap) per round is the same in both runs,
    and both rows equal the plain engine's"""
    c = fams["llama-f32"][0]
    ps = loop_prompts(c.V, 2, 2)
    plan = [(0, dict(ids=ps[0], max_new=60)), (0, dict(ids=ps[1], max_new=60))]
    want = [r["ids"] for r in drive(engines(fams["llama-f32"])[0], plan)]
    rounds = {}
    for bad in (False, True):
        _, spec = engines(fams["llama-f32"])
        spec.drafter.found = lambda eng, r, K: 1          # the call-level plain choice never skips: rounds always verify
        spec._spec_call = lambda live, K: True            # also while both rows wait: row 1's rounds keep their order
        if bad:     # slot 0 drafts 3 tokens that are never the target's; slot 1 keeps its own drafts
            spec.drafter.draft = (lambda f: lambda *a: (lambda d, kd, ck: (d.at[0].set(c.V - 1), kd.at[0].set(3), ck))(
                *f(*a)))(spec.drafter.draft)
            spec.sstep = jax.jit(SS.make_step(spec.model.make_forward(c), SS.arch_commit(spec.model, c), spec.drafter),
                                 static_argnames=("Tc", "K", "N", "trace"), donate_argnums=(1, 2))
        seen, real = [], spec._backoff
        spec._backoff = lambda r, n_, kr: (seen.append((r["slot"], n_, kr)), real(r, n_, kr))[1]
        assert [r["ids"] for r in drive(spec, plan)] == want
        rounds[bad] = seen
    mine = lambda log, s: [(n_, kr) for s_, n_, kr in log if s_ == s]       # noqa: E731
    assert mine(rounds[True], 1) == mine(rounds[False], 1) and len(mine(rounds[True], 1)) > 0
    assert any(kr == 0 for _, kr in mine(rounds[True], 0)) and spec.spec_stats["backoffs"] >= 1


def _gguf_name(path):
    f = gguf.GGUFReader(path).fields["general.name"]
    return bytes(f.parts[f.data[0]]).decode()


def test_st_s9_renamed_file_gives_the_same_pick(fams, tmp_path, monkeypatch):
    """the same weights under another general.name and another file name: the same metadata, pick, draft length,
    k_table and why lines (auto with a passed record, auto without one, forced n-gram)"""
    from mashbox import cli
    from mashbox.gguf import open_model
    src = PATHS["llama-f32"]
    data = bytearray(open(src, "rb").read())
    old = _gguf_name(src).encode()
    i = data.index(old, data.index(b"general.name"))
    data[i:i + len(old)] = b"z" * len(old)         # same length: every offset in the file stays
    dst = str(tmp_path / "another-file-name.gguf")
    open(dst, "wb").write(bytes(data))
    assert _gguf_name(dst) == "z" * len(old) != old.decode()
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    ok = dict(status="passed", record="r", record_sha256="e" * 64, component_set_sha="c" * 64, spec_step_version=SS.SPEC_STEP_VERSION,
              k_table={"1": 3, "8": 2})
    seen = []
    for path in (src, dst):
        _, A, c = open_model(path, act=jnp.float32)
        lines, out = [], []
        monkeypatch.setattr(cli, "log", lines.append)
        for a, rec in ((SimpleNamespace(speculation=None, draft_k=None), {"ngram": {"v6e-1": ok}}),
                       (SimpleNamespace(speculation=None, draft_k=None), None),
                       (SimpleNamespace(speculation="ngram", draft_k=2), None)):
            got = cli.gguf_speculation(a, c, A, records=rec)
            out.append((sorted(got), getattr(got.get("drafter"), "kind", None), got.get("draft_k"), got.get("k_table")))
        seen.append((SEL.meta_of(A, c, "gguf", getattr(c, "nextn", 0)), out, lines))
    assert seen[0] == seen[1] and seen[0][1][0][1] == "ngram" and seen[0][1][1][1] is None


def test_st_s9_order_on_files_that_have_each(tmp_path, bshj):
    """the order on real files: a GGUF file with an MTP layer (nextn) reads trained (none declared for it), then mtp (no
    GGUF MTP implementation in this version), then draft_model, then n-gram, which it picks; the BSHJ model with its MTP
    head reads trained, then picks mtp"""
    from mashbox.gguf import open_model
    css = "c" * 64
    ok = dict(status="passed", record="r", record_sha256="e" * 64, component_set_sha=css, spec_step_version=SS.SPEC_STEP_VERSION, k_table={"1": 2})
    rec = {k: {"v6e-1": ok} for k in DRS.ORDER}
    _, A, c = open_model(FXT.build("qwen35-mtp", str(tmp_path)), act=jnp.float32)
    m = SEL.meta_of(A, c, "gguf", getattr(c, "nextn", 0))
    assert m["mtp_layers"] > 0 and not m["mtp_head"] and m["verify"]
    kind, g, why, _ = SEL.pick(m, "v6e-1", rec, css=css)
    assert (kind, g) == ("ngram", True) and [w.split(":")[0] for w in why] == ["trained", "mtp", "draft_model", "ngram"]
    T, cfg, _ = bshj
    kind, g, why, _ = SEL.pick(SEL.meta_of(T.Q, cfg, "bshj", 1), "v6e-1", rec, css=css)    # config: 1 MTP layer
    assert (kind, g) == ("mtp", True) and [w.split(":")[0] for w in why] == ["trained", "mtp"]


def test_moe_refused_from_both_config_forms(tmp_path, monkeypatch):
    """MoE verify at T = K+1 is not built: a MoE file is refused whether its config gives an expert count (cfg.E) or
    the MoE flag (cfg.moe), forced n-gram included (through cli.gguf_speculation on the real qwen35moe fixture)"""
    from mashbox import cli
    from mashbox.gguf import GGUFError, open_model
    _, A, c = open_model(FXT.build("qwen35moe-f32", str(tmp_path)), act=jnp.float32)
    lines = []
    monkeypatch.setattr(cli, "log", lines.append)
    with pytest.raises(GGUFError, match="MoE"):
        cli.gguf_speculation(SimpleNamespace(speculation="ngram", draft_k=3), c, A)
    assert cli.gguf_speculation(SimpleNamespace(speculation=None, draft_k=None), c, A) == {} and "MoE" in lines[-1]
    dense = SimpleNamespace(make_forward=None, SPEC_VERIFY=True)
    for cfg in (SimpleNamespace(E=8), SimpleNamespace(moe=True), SimpleNamespace(E=0, moe=True)):
        m = SEL.meta_of(dense, cfg, "gguf")
        assert m["experts"] > 0 and not SEL.applies("ngram", m)[0], cfg
    assert SEL.applies("ngram", SEL.meta_of(dense, SimpleNamespace(moe=False), "gguf"))[0]


def test_server_loop_ngram_equals_plain(fams, tmp_path, monkeypatch):
    """one GGUF request through the real server (cli.load_gguf, Server.complete and its scheduler thread) with
    --speculation ngram: the same text and token count as the plain server, and speculation rounds ran"""
    from mashbox import cli
    from mashbox.gguf import resolve as GR
    from mashbox.runtime.server import Server
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    path, out = PATHS["llama-f32"], {}
    for extra in ([], ["--speculation", "ngram", "--draft-k", "3"]):
        a = cli.build_parser().parse_args(["run", path, "--prompt", "hi", "--ctx", "128", "--slots", "2", "--steps", "4",
                                           "--cache-dir", str(tmp_path / "xla")] + extra)
        a.gguf = GR.resolve(path, log=cli.log)
        m, eng, chat, info, _ = cli.load_gguf(a)
        srv = Server(eng, chat, m["id"], {}, log=lambda s: None)
        srv.ready = True
        r = srv.complete(dict(prompt="one two three four five " * 3, max_tokens=24, temperature=0), False)
        out[bool(extra)] = (r["choices"][0]["text"], r["usage"]["completion_tokens"], eng.DK, eng.spec_stats["rounds"])
    assert out[True][:2] == out[False][:2] and out[True][1] > 0
    assert (out[False][2], out[True][2]) == (0, 3) and out[True][3] > 0


# ---------------------------------------------------------------------------------------------------------- added in r4
def bucket_plan(V, seed=21):
    """one row alone first (row bucket 1), then 4 rows at once (row bucket 4)"""
    ps = loop_prompts(V, 5, seed)
    return [(0, dict(ids=ps[0], max_new=40))] + [(60, dict(ids=p, max_new=40)) for p in ps[1:]]


def test_k0_row_bucket_runs_plain(fams):
    """a k_table with k = 0 at a row bucket (gate_apply writes it where speculation measured slower than plain): the
    warm-up lists no speculation program there and runs, calls at that bucket run the plain decode program, the other
    bucket speculates, and every row equals the plain engine's"""
    c = fams["llama-f32"][0]
    plan = bucket_plan(c.V)
    base, _ = engines(fams["llama-f32"])
    want = [r["ids"] for r in drive(base, plan)]
    _, spec = engines(fams["llama-f32"], k_table={4: 0})
    progs = spec.spec_programs(*spec.new_state())
    assert progs and all(kw["K"] > 0 for _, _, _, kw in progs), [kw["K"] for _, _, _, kw in progs]
    spec.warm(workers=1)
    spec.shapes = []
    assert [r["ids"] for r in drive(spec, plan)] == want
    calls = {x[:2] for x in spec.shapes}
    assert ("spec", 1) in calls and ("spec", 4) not in calls and ("decode", 4) in calls, sorted(calls)
    assert spec.spec_stats["rounds"] > 0 and spec.spec_stats["plain_calls"] > 0
    assert not spec.spec_off and not spec.spec_stats.get("faults"), spec.spec_why


def test_mtp_k0_row_bucket_decodes_plain(bshj):
    """the MTP drafter needs every committed token, so no plain call may run between its calls: an engine given a
    k_table with k = 0 at a row bucket warms, decodes plain with the plain engine's output, and says why (spec_why)"""
    T, c, ws = bshj
    W, Wm = ws["bigram"]
    kw = dict(CTX=128, S=4, K=4)
    plan = bucket_plan(T.V)
    want = [r["ids"] for r in drive(Engine(c, W, T.Q, **kw), plan)]
    eng = Engine(c, W, T.Q, mtp=(T.QS, Wm), draft_k=2, spec_rounds=2, k_table={4: 0}, **kw)
    why = getattr(eng, "spec_why", None) or ""
    assert eng.DK == 0 and eng.drafter is None and "k = 0" in why, (eng.DK, why)
    eng.warm(workers=1)
    assert [r["ids"] for r in drive(eng, plan)] == want
    assert eng.DK == 0 and eng.drafter is None and "k = 0" in eng.spec_why


def raise_at(eng, at):
    """eng's speculation program raises at its at-th call before it runs (as a compile error for a new shape would)"""
    real, n = eng.sstep, [0]

    def sstep(*a, **k):
        n[0] += 1
        if n[0] == at:
            raise RuntimeError("speculation program failed (forced)")
        return real(*a, **k)
    eng.sstep = sstep
    return n


@pytest.mark.parametrize("drafter", ("ngram", "mtp"))
def test_spec_call_error_replays_plain(fams, bshj, drafter):
    """a speculation call that raises before it took the slot state replays that call with plain decoding from the
    unchanged state; speculation stays off after it (spec_off, spec_why, faults 1) and every row equals the plain
    engine's (n-gram on a GGUF family; MTP on the hybrid BSHJ model, whose rows may not take a plain call otherwise)"""
    kw = dict(CTX=128, S=4, K=4)
    if drafter == "mtp":
        T, c, ws = bshj
        W, Wm = ws["bigram"]
        V, base, eng = T.V, Engine(c, W, T.Q, **kw), Engine(c, W, T.Q, mtp=(T.QS, Wm), draft_k=2, spec_rounds=2, **kw)
    else:
        V = fams["llama-f32"][0].V
        base, eng = engines(fams["llama-f32"])
    plan = [(j, dict(ids=p, max_new=40)) for j, p in enumerate(loop_prompts(V, 3, 31))]
    want = [r["ids"] for r in drive(base, plan)]
    n = raise_at(eng, 3)
    try:
        got = [r["ids"] for r in drive(eng, plan, faults_ok=True)]
    except RuntimeError as e:
        pytest.fail(f"the speculation call that failed before it took the slot state was not replayed plain: {e}")
    assert got == want
    assert n[0] == 3 and eng.spec_off and eng.spec_stats["faults"] == 1 and "failed speculation call" in eng.spec_why
    assert eng.spec_stats["plain_calls"] > 0


def test_spec_call_error_after_it_took_the_state_is_not_replayed(fams):
    """a speculation call that fails after the device took its donated slot state (deleted buffers), or after it was
    dispatched, is never re-run in silence: the step raises (the A13 failure rules end the streams)"""
    c = fams["llama-f32"][0]
    plan = [(0, dict(ids=loop_prompts(c.V, 1, 33)[0], max_new=40))]
    _, eng = engines(fams["llama-f32"])
    real = eng.sstep

    def consumed(W, ck, cv, *a, **k):
        for x in jax.tree.leaves((ck, cv)):
            x.delete()
        raise RuntimeError("device error after the state was taken (forced)")
    eng.sstep = consumed
    with pytest.raises(RuntimeError, match="after the state was taken"):
        drive(eng, plan)
    assert not getattr(eng, "spec_off", False)
    _, eng = engines(fams["llama-f32"])
    note = eng._note
    eng._note = lambda *x: (_ for _ in ()).throw(RuntimeError("error after dispatch (forced)")) if x[0] == "spec" else note(*x)
    with pytest.raises(RuntimeError, match="after dispatch"):
        drive(eng, plan)
    assert not getattr(eng, "spec_off", False) and real is not None


def test_low_acceptance_switches_rows_to_plain(fams):
    """the low-acceptance switch-off is per row (each row's own record; never across rows): when every row's drafts are
    wrong, each row backs off and calls where no row drafts run the plain program, so far fewer verify rounds run than
    tokens are committed, no draft is accepted, and every row equals the plain engine's"""
    c = fams["llama-f32"][0]
    plan = [(0, dict(ids=p, max_new=80)) for p in loop_prompts(c.V, 4, 41)]
    base, spec = engines(fams["llama-f32"])
    want = [r["ids"] for r in drive(base, plan)]
    spec.drafter.found = lambda eng, r, K: K           # the host sees drafts for every row
    spec.drafter.draft = (lambda f: lambda *a: (lambda d, kd, ck: (jnp.full_like(d, c.V - 1), jnp.full_like(kd, 3), ck))(
        *f(*a)))(spec.drafter.draft)                   # drafts that are never the target's token, for every row
    spec.sstep = jax.jit(SS.make_step(spec.model.make_forward(c), SS.arch_commit(spec.model, c), spec.drafter),
                         static_argnames=("Tc", "K", "N", "trace"), donate_argnums=(1, 2))
    got = [r["ids"] for r in drive(spec, plan)]
    assert got == want
    st, tokens = spec.spec_stats, sum(len(x) for x in got)
    print(f"low acceptance: {tokens} tokens, {st}")
    assert sum(st["hist"][2:]) == 0 and st["backoffs"] >= len(plan) and st["plain_calls"] > 0
    assert st["rounds"] <= tokens // 2, (st["rounds"], tokens)


# -------------------------------------------------------------------------------------------------
def sealed_entry(css, k_table, chip="v6e-1"):
    return {chip: dict(status="passed", record="release/engine/cert/r", record_sha256="e" * 64, component_set_sha=css,
                       spec_step_version=SS.SPEC_STEP_VERSION, k_table=k_table)}


def test_unsealed_gguf_gate_entry_is_not_gated():
    """select.gated counts a passed entry only with its sealed record (record + record_sha256)"""
    ent = sealed_entry("c" * 64, {"1": 3})["v6e-1"]
    assert SEL.gated({"v6e-1": ent}, "v6e-1", "c" * 64)[0] is True
    for k in ("record_sha256", "record"):
        bad = {x: v for x, v in ent.items() if x != k}
        ok, why = SEL.gated({"v6e-1": bad}, "v6e-1", "c" * 64)
        assert ok is False and "not sealed" in why, (k, why)


def test_catalog_records_match_the_file_identity_never_a_name():
    """the gguf[] row is matched on repo + revision + file (a Hub file) or sha256, never on a name: a renamed file of
    the same repo, another revision or a row of another repo gives no records"""
    rec = dict(ngram=sealed_entry("c" * 64, {"1": 3}))
    reg = dict(gguf=[dict(repo="org/m-GGUF", revision="r1", file="m-Q4_K_M.gguf", sha256="a" * 64, speculation=rec)])
    hub_file = dict(repo="org/m-GGUF", revision="r1", file="m-Q4_K_M.gguf", path="/x/m-Q4_K_M.gguf")
    assert SEL.catalog_records(reg, hub_file) == rec
    assert SEL.catalog_records(reg, dict(repo=None, file="m-Q4_K_M.gguf", path="/y/m-Q4_K_M.gguf")) == {}
    assert SEL.catalog_records(reg, dict(hub_file, revision="r2")) == {}
    assert SEL.catalog_records(reg, dict(hub_file, repo="other/m-GGUF")) == {}
    assert SEL.catalog_records(reg, dict(repo=None, file="renamed.gguf", sha256="a" * 64)) == rec
    assert SEL.catalog_records({}, hub_file) == {} and SEL.catalog_records(dict(gguf=[]), hub_file) == {}


def test_gguf_load_turns_ngram_on_with_no_flag_only_from_a_sealed_catalog_record(fams, tmp_path, monkeypatch):
    """The sealed-record path through cli.load_gguf: no flag, a TPU chip key, and the catalog's gguf[] row for this file with a passed,
    sealed n-gram record for this deployment's component set -> n-gram on with the record's k_table; the same row with
    the record unsealed, or no row -> plain decoding with a why line. (The chip key is set; the engine runs on CPU.)"""
    from mashbox import cli, hub
    from mashbox.gguf import resolve as GR
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    path = PATHS["llama-f32"]
    lines = []
    monkeypatch.setattr(cli, "log", lambda s: lines.append(s))

    def load(reg, extra=()):
        monkeypatch.setattr(hub, "registry", lambda: reg)
        a = cli.build_parser().parse_args(["run", path, "--prompt", "hi", "--ctx", "128", "--slots", "2", "--steps", "4",
                                           "--cache-dir", str(tmp_path / "xla")] + list(extra))
        a.gguf = dict(GR.resolve(path, log=cli.log), repo="org/llama-GGUF", revision="r1")
        return cli.load_gguf(a)[1]
    css = load({}, ["--speculation", "ngram", "--draft-k", "3"]).component_set_sha    # this deployment's set, n-gram on
    row, kt = dict(repo="org/llama-GGUF", revision="r1", file=os.path.basename(path), sha256="0" * 64), {"1": 3, "2": 2}
    eng = load(dict(gguf=[dict(row, speculation=dict(ngram=sealed_entry(css, kt)))]))
    assert eng.DK == 3 and eng.drafter is not None and eng.drafter.kind == "ngram" and eng.k_table == {1: 3, 2: 2}
    assert eng.component_set_sha == css
    eng = load(dict(gguf=[dict(row, speculation=dict(ngram=sealed_entry("d" * 64, kt)))]))
    assert eng.DK == 0 and eng.drafter is None          # a sealed record for another component set: plain decoding
    unsealed = sealed_entry(css, {"1": 3})
    del unsealed["v6e-1"]["record_sha256"]
    for reg in (dict(gguf=[dict(row, speculation=dict(ngram=unsealed))]), dict(gguf=[]), {}):
        lines.clear()
        eng = load(reg)
        assert eng.DK == 0 and eng.drafter is None, reg
        why = [json.loads(x) for x in lines if x.startswith("{") and '"speculation"' in x]
        assert why and why[-1]["used"] is False, lines


def test_spec_health_names_the_speculation_state(fams, capsys):
    """The speculation block (Engine.spec_health): while speculation runs, on with its method, draft
    length, rounds and faults 0 (a copy: reading it changes no counter); after a call that failed before it took the
    slot state, off with the fault's why line, faults 1 and one log line; a plain engine names the load's reason, and
    drop_speculation's why reaches it"""
    c, W, A = fams["llama-f32"]
    plan = [(j, dict(ids=p, max_new=40)) for j, p in enumerate(loop_prompts(c.V, 3, 31))]
    _, eng = engines(fams["llama-f32"])
    h = eng.spec_health()
    assert (h["on"], h["method"], h["draft_k"], h["reason"]) == (True, "ngram", 3, None)
    assert h["stats"]["faults"] == 0 and h["stats"]["rounds"] == 0 and h["spec_step_version"] == SS.SPEC_STEP_VERSION
    drive(eng, plan)
    h = eng.spec_health()
    assert h["on"] is True and h["stats"]["rounds"] > 0 and h["stats"]["faults"] == 0, h
    h["stats"]["hist"].append(99)
    assert eng.spec_health()["stats"]["hist"] == eng.spec_stats["hist"] and 99 not in eng.spec_stats["hist"]
    _, bad = engines(fams["llama-f32"])
    raise_at(bad, 2)
    capsys.readouterr()
    drive(bad, plan, faults_ok=True)
    h = bad.spec_health()
    assert h["on"] is False and h["stats"]["faults"] == 1 and "failed speculation call" in h["reason"], h
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith('{"event": "speculation"')]
    assert [(x["used"], x["faults"]) for x in lines] == [(False, 1)] and lines[0]["reason"] == h["reason"], lines
    plain = Engine(c, W, A, CTX=128, S=4, K=4, spec_why="no passed gate record on this chip")
    h = plain.spec_health()
    assert (h["on"], h["method"], h["draft_k"], h["reason"]) == (False, None, 0, "no passed gate record on this chip")
    assert Engine(c, W, A, CTX=128, S=4, K=4).spec_health()["reason"] == "no drafter: plain decoding"
    _, drop = engines(fams["llama-f32"])
    drop.drop_speculation()
    assert (drop.spec_health()["on"], drop.spec_health()["reason"]) == (False, "speculation off before the warm-up")


def test_spec_health_reason_through_load_gguf(fams, tmp_path, monkeypatch):
    """through cli.load_gguf with no flag and no catalog row: plain decoding, and Engine.spec_health's reason is the load's
    speculation why line"""
    from mashbox import cli, hub
    from mashbox.gguf import resolve as GR
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    monkeypatch.setattr(hub, "registry", lambda: {})
    lines = []
    monkeypatch.setattr(cli, "log", lambda s: lines.append(s))
    path = PATHS["llama-f32"]
    a = cli.build_parser().parse_args(["run", path, "--prompt", "hi", "--ctx", "128", "--slots", "2", "--steps", "4",
                                       "--cache-dir", str(tmp_path / "xla")])
    a.gguf = dict(GR.resolve(path, log=cli.log), repo="org/llama-GGUF", revision="r1")
    eng = cli.load_gguf(a)[1]
    why = [json.loads(x) for x in lines if x.startswith("{") and '"speculation"' in x]
    h = eng.spec_health()
    assert eng.DK == 0 and h["on"] is False and why and why[-1]["used"] is False and h["reason"] == why[-1]["reason"], (h, why)


def test_speculation_draft_is_refused_on_both_load_paths(tmp_path, monkeypatch):
    """--speculation draft stops with an error on a models.json model (cli.forced_drafter, before any file is read) and on
    a GGUF file without a draft model (cli.gguf_speculation), as docs/speculation.md says: a draft model runs on a GGUF
    file, given with --draft-model"""
    from mashbox import cli
    from mashbox.gguf import GGUFError, open_model
    from mashbox.models import qwen3_8_27b as Q
    monkeypatch.setattr(cli, "log", lambda *a: None)
    with pytest.raises(SystemExit, match="--speculation draft runs on a GGUF file in this version"):
        cli.forced_drafter("draft", None, 1, Q, SimpleNamespace())
    _, A, c = open_model(FXT.build("llama-f32", str(tmp_path)), act=jnp.float32)
    with pytest.raises(GGUFError, match=r"--speculation draft: draft_model: no draft model given \(--draft-model\)"):
        cli.gguf_speculation(SimpleNamespace(speculation="draft", draft_k=None), c, A)
