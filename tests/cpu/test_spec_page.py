"""docs/speculation.md states only what the code does and what was measured:

- every speed claim on the page quotes a measured speculative-decoding result of models.json (its number, the number
  without speculation, its chip and its batch), so no claim of speed stands without a record;
- the back-off sentence names exactly the drafters that back off (Drafter.backoff) with the engine's own numbers, and
  every drafter that does not back off is said to guess on every pass;
- the "Where it runs" table names the architectures the tests run each drafter on as tested, and every other
  architecture whose module runs it as not tested, never as not supported;
- the page says auto turns nothing on exactly when no catalog gate has passed on any chip.
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
SPEED = re.compile(r"(?i)\bfaster\b|\bmore tokens\b|tokens per second|tokens/s|tok/s|\btwice\b|\bspeeds? up\b|\b\d+(?:\.\d+)?x\b")
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
    assert re.search(rf"after {Engine.BO_MISS} passes .* for {Engine.BO_WAIT} passes,.* up to {Engine.BO_MAX} passes", s), s


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
