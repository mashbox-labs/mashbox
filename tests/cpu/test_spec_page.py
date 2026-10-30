"""docs/speculation.md states only what the code does and what was measured:

- every speed claim on the page quotes a measured speculative-decoding result of models.json (its number, the number
  without speculation, its chip and its batch), so no claim of speed stands without a record;
- the back-off sentence names exactly the drafters that back off (Drafter.backoff) with the engine's own numbers, and
  every drafter that does not back off is said to guess on every pass;
- the "Where it runs" table names the architectures the tests run each drafter on as tested, and every other
  architecture whose module runs it as not tested, never as not supported;
- the page says auto turns nothing on exactly when no catalog gate has passed on any chip;
- auto's list on the page follows drafters.ORDER, and says which kinds this release does not run.
CPU, no network."""
import ast
import importlib
import json
import os
import re

import pytest

pytest.importorskip("jax")
from mashbox.gguf import archs as ARCHS  # noqa: E402
from mashbox.runtime import drafters as D  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGE = open(os.path.join(ROOT, "docs", "speculation.md")).read()
REG = json.load(open(os.path.join(ROOT, "models.json")))
# what the page calls each drafter kind in running text
NAME = {"mtp": "MTP head", "ngram": "copy from the conversation", "draft_model": "a draft model"}
SPEED = re.compile(r"(?i)\bfaster\b|\bquicker\b|\bmore tokens\b|tokens per second|tokens/s|tok/s|\bthroughput\b|\btwice\b"
                   r"|\btriples?\b|\b(?:doubles|halves)\s+(?:the\s+)?(?:speed|throughput|tokens|latency)\b|\bspeeds? up\b"
                   r"|\b\d+(?:\.\d+)?x\b|\d+(?:\.\d+)?\s?%")
NOTHING_ON = "No drafter has passed its checks on a chip"


def _num(s):
    return re.search(r"\d[\d,]*(?:\.\d+)?", s).group(0)


def _spec_results():
    return [(m, r) for m in REG["models"] for r in m.get("results") or []
            if "speculative decoding" in r.get("metric", "").lower() and r.get("short")]


def test_every_speed_claim_quotes_a_measured_result():
    """a line of the page, or of the docs index that links it, that claims speed carries the numbers with and without speculation of one measured result, its chip,
    its batch ("1 request" / "N requests"), the model's name and its drafter (MTP); that result has a date in models.json"""
    rows = _spec_results()
    assert rows, "models.json has no measured speculative-decoding result"
    index = [ln for p in ("README.md", "SUMMARY.md") for ln in open(os.path.join(ROOT, "docs", p)).read().splitlines()
             if "(speculation.md)" in ln]
    assert index, "docs/README.md and docs/SUMMARY.md do not link the page"
    claims = [ln for ln in PAGE.splitlines() + index if SPEED.search(ln)]
    assert claims, "the page states no measured speed"
    for ln in claims:
        hit = [(m, r) for m, r in rows
               if _num(r["short"]["ours"]) in ln and _num(r["short"]["vs"]) in ln and r["chip"] in ln
               and re.search(r"\b(\d+) requests?\b", r["metric"]).group(0) in ln and m["name"] in ln
               and ("MTP" in ln) == ("MTP" in r["metric"])]
        assert hit, f"a speed claim without its measured result (numbers, chip, batch): {ln!r}"
        assert all(r.get("date") for _, r in hit), hit


def _backoff():
    out = {}
    for kind, mod in D.MODULES.items():
        m = importlib.import_module(mod)
        cls = [v for v in vars(m).values() if isinstance(v, type) and issubclass(v, D.Drafter) and v.kind == kind]
        assert len(cls) == 1, (kind, cls)
        out[kind] = bool(cls[0].backoff)
    return out


def test_backoff_sentence_names_exactly_the_drafters_that_back_off():
    sentences = [s for s in re.split(r"(?<=\.)\s+", PAGE.replace("\n", " "))]
    miss = [s for s in sentences if "keep missing" in s]
    assert len(miss) == 1, miss
    s, bo = miss[0], _backoff()
    assert any(bo.values()), bo
    for kind, backs_off in bo.items():
        assert (NAME[kind] in s.lower() or NAME[kind] in s) == backs_off, (kind, backs_off, s)
        if not backs_off:
            assert any(NAME[kind] in t and "every pass" in t for t in sentences), (kind, "never said to guess every pass")
    assert re.search(rf"after {Engine.BO_MISS} passes .* for {Engine.BO_WAIT} passes,.* each further {Engine.BO_MISS} .* "
                     rf"up to {Engine.BO_MAX} passes, and a pass that keeps a guess sets it back to {Engine.BO_WAIT}\b", s), s


def _ngram_engine(tmp_path):
    """a tiny llama GGUF on CPU with copy from the conversation (k = 3) and 3 passes per speculation call"""
    import importlib.util
    import jax.numpy as jnp
    from mashbox.gguf import open_model
    from mashbox.runtime.drafters import ngram as NG
    spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
    fxt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fxt)
    f, A, c = open_model(fxt.build("llama-f32", str(tmp_path)), act=jnp.float32)
    W, _ = A.load(f, c, use_kernels=False)
    return Engine(c, W, A, CTX=128, S=4, K=4, drafter=NG.NGram(3), draft_k=3, spec_rounds=3), c


def test_the_back_off_is_counted_per_device_call(tmp_path):
    """The engine runs several passes in one device call and picks plain or speculation once per call. Here (no row
    bucket at k = 0) a call runs the plain program only when no request has a guess; it changes no request's miss count,
    and a request that is already waiting counts the call's K passes toward its wait. In a call that speculates, every pass counts for each request
    that may guess, whether or not it has a guess in that pass: a pass in which the model keeps none of its guesses
    raises the request's miss count (or starts its wait), and a pass that keeps a guess sets the miss count to 0 and the
    next wait back to BO_WAIT. A pass of a request that is waiting (from the call's start, or for the rest of the call in
    which its wait began) only counts down its wait, whether or not the model keeps a guess. Measured on a tiny model:
    one request that repeats a phrase and one that never does, in one batch; the page's back-off sentence says the
    same."""
    import inspect
    import textwrap
    import numpy as np
    from mashbox.runtime.drafters import ngram as NG
    tree = ast.parse(textwrap.dedent(inspect.getsource(Engine)))
    callers = {f.name for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
               and any(isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_backoff" for n in ast.walk(f))}
    assert callers == {"_spec_step"}, callers       # only a speculation pass changes the miss count
    writes = {}       # back-off field -> the methods that write it: a plain call (_spec_call) only counts down a wait
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(f):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AugAssign) else []):
                    for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Constant) \
                                and e.slice.value in ("miss", "wait", "span"):
                            writes.setdefault(e.slice.value, set()).add(f.name)
    assert writes == {"miss": {"_backoff"}, "span": {"_backoff"}, "wait": {"_backoff", "_spec_call"}}, writes
    eng, c = _ngram_engine(tmp_path)
    k, N = eng.DK, eng.NR
    assert N > 1, N
    rng = np.random.default_rng(2)
    phrase = rng.integers(1, c.V, 5).tolist()
    plan = [dict(ids=(rng.integers(1, c.V, 3).tolist() + phrase * 3)[:40], max_new=24), dict(ids=list(range(1, 30)), max_new=40)]
    bo = lambda r: dict(r.get("bo") or dict(miss=0, wait=0, span=Engine.BO_WAIT))         # noqa: E731
    calls, cur = [], {}
    real_call, real_step, real_bo = eng._spec_call, eng._spec_step, eng._backoff

    def spec_call(live, K):
        cur.update(K=K, live=list(live), row={s: eng.rows[s] for s in live}, before={s: bo(eng.rows[s]) for s in live},
                   cap={s: eng._k_row(eng.rows[s]) for s in live}, guess={s: eng.drafter.found(eng, eng.rows[s], k) for s in live})
        cur["spec"] = real_call(live, K)
        return cur["spec"]

    def spec_step(ck, cv, live, K):
        cur["start"] = {s: dict(seq=list(eng.rows[s]["prompt"]) + list(eng.rows[s]["ids"]), kr=min(k, eng._k_row(eng.rows[s])),
                                rem=eng.rows[s]["max_new"] - len(eng.rows[s]["ids"])) for s in live}
        cur["rounds"] = []
        out = real_step(ck, cv, live, K)
        cur["toks"] = {s: [int(x) for x in out[0][0][:, s] if x >= 0] for s in live}
        return out

    def backoff(r, n_, kr):
        b = bo(r)
        real_bo(r, n_, kr)
        cur["rounds"].append((r["slot"], n_, kr, b, bo(r)))

    eng._spec_call, eng._spec_step, eng._backoff = spec_call, spec_step, backoff
    nxt, plain_wait_counted = 0, 0
    while nxt < len(plan) or eng.busy():
        while nxt < len(plan) and eng.free_slots():
            eng.admit([plan[nxt]], ())
            nxt += 1
        cur.clear()
        eng.step(waiting=False)
        if "spec" in cur:
            calls.append({x: cur[x] for x in cur})
            if not cur["spec"]:
                assert not cur.get("rounds"), "a plain call counted a pass"
                for s in cur["live"]:
                    b, a = cur["before"][s], bo(cur["row"][s])
                    assert a["miss"] == b["miss"], ("a plain call changed a miss count", s)
                    # the plain program runs K passes: a request already waiting counts them toward its wait
                    assert a == dict(b, wait=max(0, b["wait"] - cur["K"])), ("a plain call's passes and the wait", s, b, a)
                    plain_wait_counted += b["wait"] > 0

    plain = [x for x in calls if not x["spec"]]
    spec = [x for x in calls if x["spec"]]
    assert plain and spec, (len(plain), len(spec))
    for x in plain:     # a call runs plain only when no request has a guess
        assert all(x["guess"][s] == 0 or x["cap"][s] == 0 for s in x["live"]), (x["guess"], x["cap"])
    no_guess_counted = kept_in_wait = 0
    for x in spec:
        used = {s: 0 for s in x["live"]}
        for s, n_, kr, b, a in x["rounds"]:
            st = x["start"][s]
            seq = st["seq"] + x["toks"][s][:used[s]]
            had = min(len(NG.find(np.asarray(seq, np.int64), k)), kr, max(0, st["rem"] - used[s] - 1))
            used[s] += n_
            if b["wait"] > 0:   # waiting: the pass only counts down the wait, and a kept guess (rest of the call) resets nothing
                assert a == dict(b, wait=b["wait"] - 1), (s, n_, kr, b, a)
                kept_in_wait += kr > 0 and n_ > 1
                continue
            if kr == 0:
                assert a == b, (s, n_, kr, b, a)
                continue
            if n_ == 1:     # the model kept none of the request's guesses in this pass (or it had none): it counts
                assert a["miss"] == b["miss"] + 1 or (a["miss"] == 0 and a["wait"] > 0), (s, n_, b, a)
                no_guess_counted += had == 0
            else:           # a kept guess outside a wait: the miss count to 0 and the next wait back to BO_WAIT
                assert a == dict(b, miss=0, span=Engine.BO_WAIT), (s, n_, b, a)
    assert no_guess_counted > 0, "no pass of a speculating call without a guess for a request that may guess"
    assert plain_wait_counted > 0, "no plain call ran while a request was waiting"
    assert kept_in_wait > 0, "no pass kept a guess for the rest of the call in which a wait began"
    s = [t for t in re.split(r"(?<=\.)\s+", PAGE.replace("\n", " ")) if "keep missing" in t]
    assert len(s) == 1, s
    for said in ("decides once per device call whether it speculates", "in which the model keeps none of its guesses",
                 "a call in which no request has a guess runs without speculation, and its passes do not count toward "
                 "the misses, only toward the wait of a request that is already waiting",
                 "in a call that speculates, every pass counts for each request that may guess, whether or not it has a "
                 "guess in that pass", "a device call that starts during the wait makes no guesses for it",
                 "the rest of the call in which the wait began still guesses, and those passes count toward the "
                 f"{Engine.BO_WAIT} until the wait ends, whether or not the model keeps a guess",
                 f"a pass that keeps a guess sets it back to {Engine.BO_WAIT} unless the request is waiting"):
        assert said in s[0], (said, s[0])


def _section(title):
    return PAGE.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def _row(table, first):
    rows = [ln for ln in table.splitlines() if ln.startswith(f"| {first} |")]
    assert len(rows) == 1, (first, rows)
    return [c.strip() for c in rows[0].strip("|").split("|")]


def _ticked(cell):
    return set(re.findall(r"`([a-z0-9]+)`", cell))


def _test_archs(path, pattern):
    return set(re.findall(pattern, open(os.path.join(ROOT, path)).read()))


def test_where_it_runs_names_tested_and_untested_architectures():
    """copy from the conversation: tested = the families tests/cpu/test_drafters.py runs it on; not tested = every
    other architecture whose module has a verify pass (SPEC_VERIFY), less the mixture-of-experts names the page's
    limits list. A draft model (when built), for the model it drafts for and for the draft file apart: tested = the
    target architectures its tests run and the draft architectures they write; not tested = the rest of those verify
    architectures and the rest of archs.drafting(). Nothing on the page calls any of them not supported."""
    table = _section("Where it runs")
    tree = ast.parse(open(os.path.join(ROOT, "tests", "cpu", "test_drafters.py")).read())
    fam = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FAMILIES")
    tested = {ast.literal_eval(e).split("-")[0] for e in fam.elts}
    moe = _ticked(next(ln for ln in _section("Limits").splitlines() if ln.startswith("- Mixture-of-experts files")))
    verify = {a for a, m in ARCHS.ARCHS.items() if getattr(importlib.import_module(m), "SPEC_VERIFY", False)} - moe
    _, on, off = _row(table, "Copy from the conversation")
    assert _ticked(on) == tested and _ticked(off) == verify - tested, (on, off, tested, verify)
    if "draft_model" in D.MODULES:
        # test_draft_model.py writes its target and its draft with write_dense; test_draft_model_cli.py builds its
        # target (target = FXT.build) and writes its draft with write_dense
        dtested = (_test_archs("tests/cpu/test_draft_model.py", r'write_dense\([^)]*arch="([a-z0-9]+)"')
                   | _test_archs("tests/cpu/test_draft_model_cli.py", r'write_dense\([^)]*arch="([a-z0-9]+)"'))
        ttested = (_test_archs("tests/cpu/test_draft_model.py", r'write_dense\([^)]*arch="([a-z0-9]+)"')
                   | _test_archs("tests/cpu/test_draft_model_cli.py", r'target = FXT\.build\("([a-z0-9]+)-'))
        _, on, off = _row(table, "A smaller model with the same tokenizer")
        on_t, _, on_d = on.partition(" GGUF model with ")
        off_t, _, off_d = off.rpartition("; ")
        assert dtested and _ticked(on_d) == dtested and _ticked(off_d) == set(ARCHS.drafting()) - dtested, (on, off)
        assert ttested and _ticked(on_t) == ttested and _ticked(off_t) == verify - ttested, (on, off, ttested, verify)
    assert "not supported" not in PAGE.lower()


def test_page_says_auto_turns_nothing_on_exactly_when_no_gate_passed():
    passed = [m["id"] for m in REG["models"] for e in ((m.get("drafter") or {}).get("gate") or {}).values()
              if e.get("status") == "passed"]
    passed += [g["id"] for g in REG.get("gguf") or [] for chips in (g.get("speculation") or {}).values()
               for e in chips.values() if e.get("status") == "passed"]
    assert (NOTHING_ON in PAGE) == (not passed), passed


@pytest.mark.parametrize("line", ["Speculation decodes 40% quicker.", "Speculation decodes 40 % quicker.",
                                  "It raises throughput on one chip.", "It doubles throughput.", "It triples the speed.",
                                  "It runs 1.98x faster.", "About 2x the tokens."])
def test_the_speed_check_reads_every_form_of_speed_claim(line):
    """a claim of speed in any of these words is held to a measured result by the test above"""
    assert SPEED.search(line), line


def test_the_speed_check_leaves_the_back_off_wait_alone():
    """the back-off sentence's doubling wait is not a speed claim"""
    assert not SPEED.search("each further 4 such passes in a row double the wait, up to 256 passes")


# what the page calls each drafter kind in auto's list
AUTO_NAME = {"trained": "A trained drafter made for this model", "mtp": "The model's own MTP head",
             "draft_model": "A smaller model with the same tokenizer", "ngram": "Copy from the conversation"}
NOT_RUN = "This release does not run"
PICK_ONLY_TEXT = "`auto` does not pick one in this release"


def _auto_list_problems(page):
    """auto's list (How auto picks a drafter) against drafters.ORDER: one numbered item per kind, in ORDER's order,
    named as the page names it; an item says this release does not run its kind exactly when the kind has no module
    in this release (in neither drafters.MODULES nor drafters.PICK_ONLY), and says auto does not pick one exactly when
    the kind runs only when picked (drafters.PICK_ONLY)"""
    sec = page.split("\n## How auto picks a drafter\n", 1)[1].split("\n## ", 1)[0]
    items = re.findall(r"^(\d+)\. (.+)$", sec, re.M)
    out = []
    if [int(n) for n, _ in items] != list(range(1, len(D.ORDER) + 1)):
        out.append(f"items {[n for n, _ in items]}, want 1 to {len(D.ORDER)}")
    for (n, text), kind in zip(items, D.ORDER):
        if not text.startswith(f"**{AUTO_NAME[kind]}"):
            out.append(f"item {n} should be {AUTO_NAME[kind]!r}: {text[:60]!r}")
        absent = kind not in D.MODULES and kind not in D.PICK_ONLY
        if (NOT_RUN in text) != absent:
            out.append(f"item {n} ({kind}): says {NOT_RUN!r} = {NOT_RUN in text}, module in this release = {not absent}")
        if (PICK_ONLY_TEXT in text) != (kind in D.PICK_ONLY):
            out.append(f"item {n} ({kind}): says {PICK_ONLY_TEXT!r} = {PICK_ONLY_TEXT in text}, runs only when picked = {kind in D.PICK_ONLY}")
    return out


def test_auto_list_follows_the_order_and_says_what_this_release_does_not_run():
    assert not _auto_list_problems(PAGE), _auto_list_problems(PAGE)


def test_auto_list_check_catches_edits():
    """an edit to auto's list is caught: a kind this release does not run said to run, a built kind said not to run,
    a kind that runs only when picked said to be picked by auto or said not to run, two items swapped, an item added"""
    sec = PAGE.split("\n## How auto picks a drafter\n", 1)[1].split("\n## ", 1)[0]
    items = re.findall(r"^\d+\. .+$", sec, re.M)
    unbuilt = [i for i, k in enumerate(D.ORDER) if k not in D.MODULES and k not in D.PICK_ONLY]
    picked = [i for i, k in enumerate(D.ORDER) if k in D.PICK_ONLY]
    built = [i for i, k in enumerate(D.ORDER) if k in D.MODULES]
    edits = [PAGE.replace(items[built[0]], items[built[0]] + f" {NOT_RUN} it.")]
    edits += [PAGE.replace(items[i], items[i].split(NOT_RUN)[0].rstrip()) for i in unbuilt]
    edits += [PAGE.replace(items[i], items[i].replace(PICK_ONLY_TEXT, "`auto` picks it")) for i in picked]
    edits += [PAGE.replace(items[i], items[i] + f" {NOT_RUN} it.") for i in picked]
    a, b = items[built[0]], items[built[1]]
    edits.append(PAGE.replace(a, "\0").replace(b, a.replace(f"{built[0] + 1}. ", f"{built[1] + 1}. ", 1))
                 .replace("\0", b.replace(f"{built[1] + 1}. ", f"{built[0] + 1}. ", 1)))
    edits.append(PAGE.replace(items[-1], items[-1] + f"\n{len(items) + 1}. **Another drafter**."))
    for e in edits:
        assert e != PAGE
        assert _auto_list_problems(e), e.split("## How auto picks a drafter", 1)[1][:600]
