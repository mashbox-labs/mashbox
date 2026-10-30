"""The standards tier's tiny models: one builder, seeded, f32, built in seconds, no network, no chip.

Families (families()):
  every GGUF fixture tools/gguf_fixtures makes from a seed with float weights (FIXTURES entries with make "gguf-py" and
  no block types, in fixtures.py and in every other table there that keeps fixtures.py's contract (more.py): llama,
  qwen2, qwen3, qwen35, qwen35 with an MTP layer, qwen35moe, gemma4, gemma3, exaone, granite, and every family added
  there later), picked from those tables, never from a list kept here; plus "qwen38-bshj", the tiny Qwen3.8
  architecture model in the BSHJ format as tests/cpu/test_spec_in_step.py builds it (its builder is loaded from that
  file, one source). A family whose architecture `mashbox run` refuses (no line in src/mashbox/gguf/archs ARCHS) stays
  in the tier: its rows are strict xfails on that refusal while the line is missing (refused()), and run as gated rows
  the day it comes back.

Two weight sets per family (both f32 on the host, jax_default_matmul_precision=highest):
  "bigram"  every residual write (attention output, linear-attention output, FFN and expert down projections) is scaled by
            LO = 0.05, so the model is close to a bigram model: top-1 logit gaps are wide, no near ties, checks use
            exact equality. Where a norm sits between a write and its residual add (sandwich norms: post_ffw_norm, and
            post_attention_norm in a layer that also has post_ffw_norm), that norm would cancel the scale, so its weight
            is scaled too (an RMSNorm's output is linear in its weight). Without post_ffw_norm, post_attention_norm is the
            FFN's input norm (as in qwen35) and stays as written;
  "random"  the fixture as written: it has near ties, so a check that compares two different batch shapes is recorded,
            not gated (batched output may differ from alone at a near tie, as in vLLM).

GGUF files load through the CLI's own GGUF path (cli.load_gguf: resolver, plan, fit check, loader, tokenizer, Engine),
so a test runs exactly what `mashbox run <file>.gguf` runs, with --no-kernels on CPU. Nothing here is a trained model.
"""
import hashlib
import importlib.util
import json
import os
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LO = 0.05
BSHJ = "qwen38-bshj"
TINY_ROW = "std-tiny-qwen38"     # the row id the load-tail test asks the catalog for (the BSHJ fixture through cli.load)
RESIDUAL = (".attn_output.weight", ".ssm_out.weight", ".ffn_down.weight", ".ffn_down_exps.weight", ".ffn_down_shexp.weight")
POST_FFN, POST_ATTN = ".post_ffw_norm.weight", ".post_attention_norm.weight"
# Fixtures whose generations say nothing yet: every row that needs varied tokens (the near-bigram batched-equals-alone
# rows, the join-and-leave schedule) is a strict xfail on them, so the run fails the day the fixture changes and the
# entry is removed. Never a skip.
DEGENERATE = {
    "gemma4-f32": "open: this fixture copies the last token. tools/gguf_fixtures write_gemma4 draws token_embd as "
                  "0.5 x randn, the graph multiplies it by sqrt(256) and the output head is tied, so the embedding "
                  "dominates the residual stream and both weight sets continue every prompt with its last token "
                  "(measured on this fixture). Measured on all three tied-head fixtures (gemma4-f32, gemma3-f32, "
                  "granite-f32) and on none of the fixtures with their own output head (exaone-f32 included); INFERRED "
                  "for other tied-head models: near-bigram weights (residual writes x 0.05) keep that copy map, so this "
                  "tier needs a tied-head near-bigram construction",
    "gemma3-f32": "open: this fixture copies the last token, as gemma4-f32 does. tools/gguf_fixtures/more.py "
                  "write_gemma3 has a tied output head and the graph scales the embedding by sqrt(hidden size); both "
                  "weight sets continue every prompt with its last token (measured: three prompts ending in 209, 251 "
                  "and 16 give 209 x 10, 251 x 10 and 16 x 10). The same open item as gemma4-f32",
    "granite-f32": "open: this fixture copies the last token, as gemma4-f32 does. tools/gguf_fixtures/more.py "
                   "write_dense2 gives granite a tied output head (token_embd 0.5 x randn); measured with granite's "
                   "ARCHS line added on a copy of the tree: both weight sets continue every prompt with its last token "
                   "(209 x 10, 251 x 10, 16 x 10). While the line is missing its rows are refused() xfails first. The "
                   "same open item as gemma4-f32",
}
_mods, _built = {}, {}


def _load(name, rel):
    if name not in _mods:
        sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        _mods[name] = m
    return _mods[name]


TABLES = ("fixtures.py", "more.py")      # tools/gguf_fixtures files with a FIXTURES table and fixtures.py's build / write


def fixtures():
    """tools/gguf_fixtures/fixtures.py (the GGUF fixture builder; this tier calls it and never edits it)"""
    return _load("std_gguf_fixtures", os.path.join("tools", "gguf_fixtures", "fixtures.py"))


def _tables():
    """-> {fixture name: (its builder module, its FIXTURES entry)} over every table file present (an old tree has fewer)"""
    out = {}
    for f in TABLES:
        if os.path.exists(os.path.join(ROOT, "tools", "gguf_fixtures", f)):
            m = fixtures() if f == "fixtures.py" else _load("std_gguf_" + f[:-3], os.path.join("tools", "gguf_fixtures", f))
            for n, e in m.FIXTURES.items():
                assert n not in out, f"fixture {n!r} is in two tables"
                out[n] = (m, e)
    return out


def families(only=None):
    """-> family names, in a fixed order. only: one name (the runner's STD_FAMILY) or None for all."""
    names = [n for n, (_m, f) in _tables().items() if f["make"] == "gguf-py" and not f.get("quant")] + [BSHJ]
    if only is not None:
        if only not in names:
            raise ValueError(f"unknown family {only!r}; families: {', '.join(names)}")
        return [only]
    return names


def is_moe(family):
    return family != BSHJ and bool(_tables()[family][1].get("moe"))


def refused(family):
    """-> None, or the strict-xfail reason for a family whose architecture has no line in mashbox.gguf.archs ARCHS, or
    that the hold refuses (archs.held with the family's moe: a mixture-of-experts file with no passing chip row of its
    own, under a MoE architecture or a dense one's name): `mashbox run` refuses its file (cli.load_gguf: GGUFError,
    archs.module / archs.held). Read from the code each run, so the rows run gated the day the architecture's line comes
    back (no xfail to remove, nothing to flip). A tiny fixture is never a proven file (held reads no path here, and a
    passing row releases only the file whose sha256 it names), so a MoE family stays refused. Never a bypass of the
    refusal."""
    if family == BSHJ:
        return None
    from mashbox.gguf import archs
    arch = _tables()[family][1]["arch"]
    if arch not in archs.ARCHS:
        return (f"open: src/mashbox/gguf/archs/__init__.py ARCHS has no {arch!r} line, so `mashbox run` refuses this "
                "file (the architecture's graph exists; its line comes back when the file's pre-tokenizer is read)")
    if archs.held(arch, moe=is_moe(family)):
        return (f"held: a {arch!r} file with routed experts runs only when a passing chip row names that file (its "
                "sha256), and this tiny fixture is no proven file, so `mashbox run` refuses it with the held line "
                "(gguf.archs.held, cli.load_gguf)")
    return None


def param(family):
    """the family as a pytest parameter: a refused family's rows are strict xfails raising the refusal (refused())"""
    import pytest
    from mashbox.gguf.reader import GGUFError
    why = refused(family)
    marks = [pytest.mark.xfail(strict=True, raises=GGUFError, reason=why)] if why else []
    return pytest.param(family, marks=marks, id=family)


def _outdir():
    d = os.environ.get("STD_TINY_DIR")
    if not d:
        d = tempfile.mkdtemp(prefix="std-tiny-")
        os.environ["STD_TINY_DIR"] = d
    os.makedirs(d, exist_ok=True)
    return d


def _scaled_override(path):
    """{tensor name: f32 array} of a float GGUF file with every residual write scaled by LO (all other tensors as read)"""
    import gguf
    r = gguf.GGUFReader(path)
    names = {t.name for t in r.tensors}
    out = {}
    for t in r.tensors:
        a = gguf.quants.dequantize(t.data, t.tensor_type).reshape(tuple(reversed([int(x) for x in t.shape])))
        out[t.name] = (a * LO if _residual(t.name, names) else a).astype(np.float32)
    return out


def _residual(name, names):
    """True for a tensor whose scale is a residual write's scale: an output projection, or a norm on a sub-layer's
    output before its residual add (post_ffw_norm; post_attention_norm only when its layer also has post_ffw_norm)"""
    if name.endswith(RESIDUAL) or name.endswith(POST_FFN):
        return True
    return name.endswith(POST_ATTN) and name[:-len(POST_ATTN)] + POST_FFN in names


def weights(family, kind):
    """-> the path of family's weights of `kind` ("bigram" or "random"); built once per process"""
    assert kind in ("bigram", "random"), kind
    key = (family, kind)
    if key in _built:
        return _built[key]
    d = _outdir()
    if family == BSHJ:
        S = _spec_builder()
        wd = os.path.join(d, BSHJ)
        os.makedirs(wd, exist_ok=True)
        _bshj_config(wd)
        p = os.path.join(wd, f"{kind}.bshj")
        S.write_bshj(p, S.tensors(11, LO if kind == "bigram" else 1.0))
    else:
        F = _tables()[family][0]
        p = F.build(family, d) if kind == "random" else None
        if kind == "bigram":
            src = _built.get((family, "random")) or weights(family, "random")
            p = F.write(family, os.path.join(d, f"{family}.bigram.gguf"), override=_scaled_override(src), quant=False)
    _built[key] = p
    return p


def _spec_builder():
    return _load("std_spec_in_step", os.path.join("tests", "cpu", "test_spec_in_step.py"))


def _bshj_config(wd, S=None):
    """config.json of the BSHJ fixture whose dimensions are module S's (the spec tests' builder by default)"""
    S = S or _spec_builder()
    json.dump(dict(model_type="qwen3_5_text", vocab_size=S.V, hidden_size=S.D, intermediate_size=S.F, num_hidden_layers=S.L,
                   num_attention_heads=S.H, num_key_value_heads=S.KV, head_dim=S.HD, linear_num_key_heads=S.HK,
                   linear_num_value_heads=S.HV, linear_key_head_dim=S.DK, linear_value_head_dim=S.DV,
                   linear_conv_kernel_dim=S.CK, full_attention_interval=4, rms_norm_eps=1e-6,
                   rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25),
                   tie_word_embeddings=False, attn_output_gate=True, mtp_num_hidden_layers=1),
              open(os.path.join(wd, "config.json"), "w"))


class IdChat:
    """The BSHJ fixture has no tokenizer: prompts are token ids. chat_ids gives a fixed short prompt (cli.warm's two
    request-path calls)."""
    stop_ids = ()

    def chat_ids(self, messages, enable_thinking=False, tools=None):
        return [7, 11, 13, 17, 19, 23]


def _quiet(*a, **k):
    pass


def engine(family, kind="bigram", S=4, K=4, CTX=256, prefix_cache=False, cache_ram_mib=64, extra_argv=()):
    """-> (Engine, chat, info) for family's `kind` weights. GGUF: cli.load_gguf with the arguments `mashbox run <file>
    --no-kernels --ctx CTX --slots S --steps K` (+ extra_argv) parses to; BSHJ: the Qwen3.8 port's Engine as the spec
    tests build it.
    The persistent compile cache is off in this process, so a run builds its own programs and never reads programs an
    earlier run left on disk. (The compile counter would see a cache hit too: jax's backend-compile event wraps
    compile_or_get_cached.)"""
    import jax
    from mashbox import cli
    jax.config.update("jax_default_matmul_precision", "highest")
    if family == BSHJ:
        from mashbox.models import qwen3_8_27b as Q
        from mashbox.models.qwen3_8_27b import weights as QW
        from mashbox.runtime.engine import Engine
        wd = os.path.join(_outdir(), BSHJ)
        p = weights(family, kind)
        cfg = Q.Config(wd)
        W, _ = QW.load(p, cfg)
        eng = Engine(cfg, W, Q, CTX=CTX, S=S, K=K, prefix_cache=prefix_cache, cache_ram=cache_ram_mib << 20)
        chat = IdChat()
        info = dict(file=os.path.basename(p))
    else:
        from mashbox.gguf import pick as GR
        p = weights(family, kind)
        argv = ["run", p, "--no-kernels", "--ctx", str(CTX), "--slots", str(S), "--steps", str(K),
                "--cache-dir", os.path.join(_outdir(), "xla"), "--cache-ram", str(cache_ram_mib)]
        if not prefix_cache:
            argv.append("--no-prefix-cache")
        argv += list(extra_argv)
        a = cli.build_parser().parse_args(argv)
        a.gguf = GR.resolve(a.model, log=_quiet, download=False)
        log, cli.log = cli.log, _quiet
        try:
            _m, eng, chat, info, _t0 = cli.load_gguf(a)
        finally:
            cli.log = log
    jax.config.update("jax_enable_compilation_cache", False)
    return eng, chat, info


TOKENIZER_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
                  "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                  "tokenizer.ggml.padding_token_id")


def tokenizer_hash(family):
    """GGUF: sha256 of the tokenizer read back from the family's written file (its GGUF KV: tokens, token_type, merges
    and the bos / eos / eot / pad ids, a missing key as null), the speculation plan's tokenizer hash. The BSHJ fixture
    has no tokenizer: the label is the sha256 of its vocabulary size (ids only)."""
    if family == BSHJ:
        return hashlib.sha256(json.dumps(dict(ids_only=_spec_builder().V)).encode()).hexdigest()
    import gguf
    fields = gguf.GGUFReader(weights(family, "random")).fields
    kv = {k: (_plain(fields[k].contents()) if k in fields else None) for k in TOKENIZER_KEYS}
    return hashlib.sha256(json.dumps(kv, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _plain(v):
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", "surrogateescape")
    return v.item() if hasattr(v, "item") else v


def vocab_size(eng):
    return int(eng.c.V)


def prompts(eng, n=None, lo=5, hi=40, seed=0, lengths=None):
    """n random prompts of lengths lo..hi (or one prompt per entry of `lengths`) over the model's vocabulary, below the
    special tokens"""
    rng = np.random.default_rng(seed)
    top = min(vocab_size(eng), 256)
    if lengths is None:
        lengths = [int(rng.integers(lo, hi + 1)) for _ in range(n)]
    return [rng.integers(1, top, int(m)).tolist() for m in lengths]


def drive(eng, plan, stop_ids=(), cancel_at=None):
    """Run an arrival schedule through Engine.admit / step / cancel. This mirrors Server.loop_live (runtime/server.py) on
    a fixed clock, so a schedule replays exactly: admit what has arrived into free slots (plan order), then cancels,
    then one step, with waiting=True only while a request that has arrived is not admitted (loop_live: a job in its
    wait queue or in the server queue). plan: [(arrival step, request dict(ids, max_new, samp=None))].
    cancel_at: {plan index: step index}. -> rows in plan order."""
    rows, nxt, i = [None] * len(plan), 0, 0
    cancel_at = dict(cancel_at or {})
    while True:
        batch = []
        while nxt < len(plan) and plan[nxt][0] <= i and len(batch) < eng.free_slots():
            batch.append(nxt)
            nxt += 1
        if batch:
            got = eng.admit([plan[j][1] for j in batch], stop_ids=stop_ids)
            for j, r in zip(batch, got):
                rows[j] = r
        for j, at in list(cancel_at.items()):
            if at <= i and rows[j] is not None and rows[j]["fin"] is None:
                eng.cancel([rows[j]])
                del cancel_at[j]
        if not eng.busy() and nxt >= len(plan):
            return rows
        if eng.busy():
            eng.step(waiting=nxt < len(plan) and plan[nxt][0] <= i)
        i += 1
        if i > 10000:
            raise RuntimeError("schedule did not end in 10000 steps")


_compiles = dict(n=0, on=False)


def compile_count():
    """XLA backend compiles in this process so far, counted by this tier's own jax.monitoring listener (the event the
    engine's runtime/programs.py counts, but not read through it, so a broken engine counter cannot hide a compile).
    The first call starts the count."""
    if not _compiles["on"]:
        from jax import monitoring

        def listen(ev, secs, **k):
            if ev == "/jax/core/compile/backend_compile_duration":
                _compiles["n"] += 1
        monitoring.register_event_duration_secs_listener(listen)
        _compiles["on"] = True
    return _compiles["n"]


def alone(eng, ids, max_new, samp=None, stop_ids=()):
    return eng.generate([ids], max_new=max_new, stop_ids=stop_ids, samp=None if samp is None else [samp])["ids"][0]


def bf16_step(x):
    """one bf16 step (8 bits of mantissa) at the magnitude of x, as the chip's tie-aware rule counts it"""
    return 2.0 ** (np.floor(np.log2(np.maximum(np.abs(x), 1e-30))) - 7)


def tie_check(eng, prompt, ids, stop_tok=None):
    """The chip's tie-aware rule for one answer: teacher-force prompt + ids (+ the stop token that ended it)
    through Engine.logits, the plain forward from a fresh zero state (no slot, no cache) -> (beyond, near): how many
    committed tokens are more than one bf16 step under the top logit there, and how many are not the argmax"""
    got = list(ids) + ([stop_tok] if stop_tok is not None else [])
    if not got:
        return 0, 0
    lg = np.asarray(eng.logits(list(prompt) + got)[len(prompt) - 1:len(prompt) - 1 + len(got)], np.float32)
    top, mine = lg.max(-1), lg[np.arange(len(got)), got]
    return int(np.sum(top - mine > bf16_step(top))), int(np.sum(lg.argmax(-1) != np.asarray(got)))


def first_diff(row, alone_ids, cut=False):
    """the first position where a row differs from its alone run (a finish at a different length counts), None when
    equal; cut=True (a cancelled row): only the row's own length is compared"""
    n = min(len(row), len(alone_ids))
    d = next((i for i in range(n) if row[i] != alone_ids[i]), None)
    if d is None and not cut and len(row) != len(alone_ids):
        d = n
    return d


def alone_gap(eng, prompt, alone_ids, d):
    """-> (gap, step): the alone run's top-2 logit gap at position d (teacher-forced, fresh state) and one bf16 step of
    its top logit there"""
    lg = np.sort(np.asarray(eng.logits(list(prompt) + list(alone_ids[:d]))[-1], np.float32))
    return float(lg[-1] - lg[-2]), float(bf16_step(lg[-1]))


def tie_record(eng, prompt, row, alone_ids, stop_ids=()):
    """one row of a batched run against the tie-aware rule: beyond / near (tie_check over its committed tokens and its
    stop token), first (first_diff with its alone run), gap / step / tie (alone's top-2 gap there, one bf16 step,
    gap <= step)"""
    stop_tok = next(iter(stop_ids)) if row["fin"] == "stop" and len(stop_ids) == 1 else None
    beyond, near = tie_check(eng, prompt, row["ids"], stop_tok)
    d = first_diff(row["ids"], alone_ids, cut=row["fin"] == "cancel")
    rec = dict(beyond=beyond, near=near, first=d, gap=None, step=None, tie=None)
    if d is not None:
        rec["gap"], rec["step"] = alone_gap(eng, prompt, alone_ids, d)
        rec["tie"] = rec["gap"] <= rec["step"]
    return rec


class _Stops:
    """the family's chat object with this schedule's stop ids (the server loop admits with chat.stop_ids)"""

    def __init__(self, chat, stop_ids):
        self._chat, self.stop_ids = chat, tuple(stop_ids)

    def __getattr__(self, name):
        return getattr(self._chat, name)


def serve(eng, chat, plan, stop_ids=(), cancel_at=None, secs=300):
    """drive()'s schedule through the real server: Server.submit (caps, waiting counts) and the Server.loop_live thread
    a Server starts (admit, cancel, step, finish). The engine's step is the clock: the loop's device call i is held
    while the requests that arrive at i + 1 are submitted and the hang-ups due at i + 1 are made (Server.hangup), so the
    loop takes them before its call i + 1, as drive() does. The requests that arrive at 0 are put on the server's queue
    together (under the queue's own lock), so the loop's first admit takes them all. Greedy requests only.
    -> rows in plan order: dict(ids, fin, slot, cached) from each job's result."""
    import queue
    import threading
    import time
    from mashbox.runtime import server as SV
    assert all(q.get("samp") is None for _at, q in plan), "serve() submits greedy requests only"
    entered, go, real = threading.Semaphore(0), threading.Semaphore(0), eng.step

    def held(*a, **k):
        entered.release()
        if not go.acquire(timeout=secs):
            raise RuntimeError("the held device call was never released")
        return real(*a, **k)
    eng.step = held
    try:
        srv = SV.Server(eng, _Stops(chat, stop_ids), "tiny", dict(weights="tiny"), log=_quiet)
        jobs = [None] * len(plan)

        def submit(j):
            k = dict(max_tokens=plan[j][1]["max_new"], temperature=0, repetition_penalty=1.0, frequency_penalty=0,
                     presence_penalty=0)
            jobs[j] = srv.submit(list(plan[j][1]["ids"]), k, False, False)
        live, srv.q = srv.q, queue.Queue()
        for j in range(len(plan)):
            if plan[j][0] == 0:
                submit(j)
        first, srv.q = srv.q, live
        with live.mutex:
            live.queue.extend(first.queue)
            live.unfinished_tasks += len(first.queue)
            live.not_empty.notify()
        cancel_at, i, end = dict(cancel_at or {}), 0, time.monotonic() + secs
        while True:
            pending = [j for j in range(len(plan)) if jobs[j] is None]
            done = all(jb is not None and jb["ev"].is_set() for jb in jobs)
            if done and not pending:
                break
            if entered.acquire(timeout=0.01):
                for j in pending:
                    if plan[j][0] == i + 1:
                        submit(j)
                for j, at in list(cancel_at.items()):
                    if at == i + 1 and jobs[j] is not None:
                        srv.hangup(jobs[j])
                        del cancel_at[j]
                i += 1
                go.release()
            elif done:
                raise AssertionError(f"the engine went idle after call {i} with arrivals {pending} still to come: the "
                                     "schedule's clock and the server's cannot align")
            if time.monotonic() > end:
                raise RuntimeError(f"the served schedule did not end in {secs} s (call {i})")
    finally:
        eng.step = real
    for jb in jobs:
        if jb.get("error") is not None:
            raise jb["error"]
    return [dict(ids=jb["result"]["ids"], fin=jb["result"]["finish"], slot=jb["row"]["slot"],
                 cached=jb["result"]["cached"]) for jb in jobs]


def serve_http(eng, chat, monkeypatch=None, keys=None, **caps):
    """the real server (runtime/server.py Server + serve) on a free local port, in a daemon thread -> (Server, port);
    caps: server module constants set through monkeypatch for this test only"""
    import threading
    from mashbox.runtime import server as SV
    for k, v in caps.items():
        monkeypatch.setattr(SV, k, v)
    srv = SV.Server(eng, chat, "tiny", dict(weights="tiny"), log=lambda m: None)
    up = threading.Event()                 # port 0: serve() binds the system's free port (never a number picked earlier)
    threading.Thread(target=SV.serve, args=(srv, "127.0.0.1", 0, lambda m: up.set() if '"listening"' in str(m) else None),
                     kwargs=dict(keys=keys), daemon=True).start()
    assert up.wait(60), "serve() is not listening"
    return srv, srv.httpd.server_address[1]


def post(port, path, body, timeout=120, key=None):
    """one HTTP POST of a JSON body -> (status, body bytes, headers); an HTTP error status is returned, not raised"""
    import urllib.error
    import urllib.request
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def catalog_engine(S=1, CTX=256):
    """-> (Engine, chat, info): the BSHJ fixture loaded through cli.load, the models.json path (`mashbox run TINY_ROW
    --weights <fixture> --no-kernels`); it raises until the catalog has that row"""
    from mashbox import cli
    p = weights(BSHJ, "bigram")
    a = cli.build_parser().parse_args(["run", TINY_ROW, "--weights", p, "--no-kernels", "--ctx", str(CTX), "--slots", str(S),
                                       "--cache-dir", os.path.join(_outdir(), "xla")])
    log, cli.log = cli.log, _quiet
    try:
        _m, eng, chat, info, _t0 = cli.load(a)
    finally:
        cli.log = log
    return eng, chat, info


# ---- ST-A fixtures: scripted logits ----------------------------------------------------------------------------------
# The coding-agent rows test the plumbing (the API routes, tool-call parsing, streaming, the pinned clients' contract),
# never model quality. Their engine is the BSHJ fixture's real Engine (admit / step / cancel, slots, K/V cache, the
# sampler, the real server loop) whose forward keeps running but whose next-token logits are replaced by a scripted
# logits table (ScriptedLogits: this counts as the real tiny engine for these rows). The table is keyed on the row's
# last input token, and the scripted answer is cut into
# pieces that are distinct tokens of a local tokenizer, so the chain spells an exact tool call. Nothing in these rows
# is model behaviour; every row and docstring that uses them says "scripted logits".
SCRIPT_PIECES = ("Checking", " the folder.", "\n\n<to", "ol_call>\n<fun", "ction=Ba", "sh>\n<param", "eter=comm", "and>\npw",
                 "d\n</para", "meter>\n</func", "tion>\n</tool_", "call>")     # cut inside every tag: streamed chunks split them
SCRIPT_TEXT = "".join(SCRIPT_PIECES)
SCRIPT_CALL = ("Bash", {"command": "pwd"})
SCRIPT_SAYS = "Checking the folder."
AGENT_SPECIALS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>",
                  "</tool_response>", "<think>", "</think>")
AGENT_TEMPLATE = (
    '{%- set NL = "\\n" -%}'
    "{%- if messages and messages[0].role == 'system' -%}{%- set sys = messages[0].content -%}{%- set rest = messages[1:] -%}"
    "{%- else -%}{%- set sys = '' -%}{%- set rest = messages -%}{%- endif -%}"
    "{%- if tools or sys -%}<|im_start|>system{{ NL }}{{ sys }}{%- if tools -%}{{ NL }}# Tools{{ NL }}"
    "{%- for t in tools -%}{{ t | tojson }}{{ NL }}{%- endfor -%}{%- endif -%}<|im_end|>{{ NL }}{%- endif -%}"
    "{%- for m in rest -%}"
    "{%- if m.role == 'user' -%}<|im_start|>user{{ NL }}{{ m.content }}<|im_end|>{{ NL }}"
    "{%- elif m.role == 'assistant' -%}<|im_start|>assistant{{ NL }}{{ m.content }}"
    "{%- for c in m.tool_calls or [] -%}<tool_call>{{ NL }}<function={{ c.function.name }}>{{ NL }}"
    "{%- for k, v in c.function.arguments.items() -%}<parameter={{ k }}>{{ NL }}{{ v if v is string else v | tojson }}"
    "{{ NL }}</parameter>{{ NL }}{%- endfor -%}</function>{{ NL }}</tool_call>{%- endfor -%}<|im_end|>{{ NL }}"
    "{%- elif m.role == 'tool' -%}<|im_start|>user{{ NL }}<tool_response>{{ NL }}{{ m.content }}{{ NL }}</tool_response>"
    "<|im_end|>{{ NL }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant{{ NL }}{%- endif -%}")


def _byte_chars():
    """GPT-2's byte-level alphabet (byte value -> printable character), as tokenizers' ByteLevel uses it"""
    return fixtures().byte_chars()


def agent_tokenizer(folder):
    """A local byte-level tokenizer folder for Chat (runtime/chat.py): tokenizer.json (ids 0-255 the bytes, then the
    script pieces as tokens of their own, then the special tokens), chat_template.jinja (AGENT_TEMPLATE: system,
    tools as JSON lines, tool calls in the <tool_call><function=..><parameter=..> format, tool results as
    <tool_response>) and generation_config.json (stop on <|im_end|> and <|endoftext|>). -> {piece or special: id}"""
    os.makedirs(folder, exist_ok=True)
    bc = _byte_chars()
    vocab = {bc[b]: b for b in range(256)}
    ids = {}
    for p in SCRIPT_PIECES:
        s = "".join(bc[b] for b in p.encode())
        assert s not in vocab, ("script pieces must be distinct tokens", p)
        ids[p] = vocab[s] = len(vocab)
    added = []
    for s in AGENT_SPECIALS:
        ids[s] = len(vocab) + len(added)
        added.append(dict(id=ids[s], content=s, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True))
    bl = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=True, use_regex=True)
    spec = dict(version="1.0", truncation=None, padding=None, added_tokens=added, normalizer=None, pre_tokenizer=bl,
                post_processor=None, decoder=bl,
                model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None,
                           fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab=vocab, merges=[]))
    json.dump(spec, open(os.path.join(folder, "tokenizer.json"), "w"))
    open(os.path.join(folder, "chat_template.jinja"), "w").write(AGENT_TEMPLATE)
    json.dump(dict(eos_token_id=[ids["<|im_end|>"], ids["<|endoftext|>"]]), open(os.path.join(folder, "generation_config.json"), "w"))
    return ids


class ScriptedLogits:
    """Scripted logits: the model module `model` (its forward, state, everything) with the forward's next-token logits
    replaced by table[the row's last input token] (and nxt = their argmax). The forward itself still runs, so the
    engine's K/V cache, slot state and programs are the real ones; only the logits are scripted. Speculation's verify
    and all-position logits are not scripted (the agent rows run without speculation)."""

    def __init__(self, model, table):
        self._m, self.table = model, np.asarray(table, np.float32)

    def __getattr__(self, k):
        return getattr(self._m, k)

    def make_forward(self, c, decode_kernels=None, **kw):
        import jax.numpy as jnp
        f, T = self._m.make_forward(c, decode_kernels, **kw), self.table

        def forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, verify=False, hidden=False,
                    keep=None, ed=None):
            if all_logits or verify:
                raise NotImplementedError("scripted logits: next-token logits only")
            k = dict(contig=contig, hidden=hidden)
            if keep is not None:
                k["keep"] = keep
            if ed is not None:
                k["ed"] = ed
            r = f(W, ck, cv, toks, p0, lens, slots, Tc, **k)
            last = toks[jnp.arange(toks.shape[0]), jnp.maximum(lens - 1, 0)]
            lg = jnp.asarray(T)[last]
            return dict(r, lg=lg, nxt=jnp.argmax(lg, -1).astype(jnp.int32))
        return forward


def script_table(V, ids, after="\n"):
    """(V, V) scripted logits: the token `after` (the template's last generation-prompt token, the newline after
    "assistant") -> the first piece, each piece -> the next, the last piece -> <|im_end|>; every other token ->
    <|im_end|>. The chosen token's logit is 0 and every other is -1e4, so greedy and sampled rows take it alike."""
    nxt = np.full(V, ids["<|im_end|>"], np.int64)
    nxt[ord(after)] = ids[SCRIPT_PIECES[0]]                  # ids 0-255 are the bytes (agent_tokenizer)
    for a, b in zip(SCRIPT_PIECES, SCRIPT_PIECES[1:]):
        nxt[ids[a]] = ids[b]
    t = np.full((V, V), -1e4, np.float32)
    t[np.arange(V), nxt] = 0.0
    return t


def agent_engine(S=4, K=4, CTX=1024, prefix_cache=False):
    """-> (Engine, Chat, script ids): the BSHJ fixture's real Engine with scripted logits (ScriptedLogits over the
    near-bigram weights, script_table) and the local tokenizer folder (agent_tokenizer). Plumbing only."""
    import jax
    from mashbox.models import qwen3_8_27b as Q
    from mashbox.models.qwen3_8_27b import weights as QW
    from mashbox.runtime.chat import Chat
    from mashbox.runtime.engine import Engine
    jax.config.update("jax_default_matmul_precision", "highest")
    wd = os.path.join(_outdir(), BSHJ)
    p = weights(BSHJ, "bigram")
    cfg = Q.Config(wd)
    folder = os.path.join(_outdir(), "agent-tokenizer")
    ids = agent_tokenizer(folder)
    W, _ = QW.load(p, cfg)
    eng = Engine(cfg, W, ScriptedLogits(Q, script_table(cfg.V, ids)), CTX=CTX, S=S, K=K, prefix_cache=prefix_cache,
                 cache_ram=64 << 20)
    jax.config.update("jax_enable_compilation_cache", False)
    return eng, Chat(folder), ids
