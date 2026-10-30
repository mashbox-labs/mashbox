#!/usr/bin/env python3
"""Which contracts a GGUF row stands on, and whether it still stands (tests/contracts/conftest.py writes the results).

A GGUF file runs here when its architecture, its tensor types and its tokenizer do, as llama.cpp supports files: by
type, never by file. So a row's record names the contracts of its kinds, not the bytes of the engine files it ran:

  arch/<general.architecture>     a tiny random-weight file of that architecture vs llama.cpp's logits
  op/<ggml type>                  every tensor type in the file: dequantization and matmul vs a NumPy reference
  tok/<tokenizer>                 the file's tokenizer: tok/<tokenizer.ggml.pre> for "gpt2" vocabularies, tok/spm for
                                  SentencePiece ("llama"), tok/gemma4 for gemma4 vocabularies; ids vs llama.cpp's
  path/run                        `mashbox run` through the CLI, resolver, loader, engine, sampler, chat and server
"""
import argparse
import hashlib
import json
import os
import sys

SPM, GEMMA4 = "tok/spm", "tok/gemma4"


def tok_id(model, pre):
    """the tokenizer contract of a file's tokenizer.ggml.model and tokenizer.ggml.pre"""
    if model == "llama":
        return SPM
    if model == "gemma4":
        return GEMMA4
    return f"tok/{pre or 'default'}"


def tokenizer_ids():
    """every tokenizer contract this version reads a file with (gguf/tokenizer.py PRE, SentencePiece, gemma4)"""
    from mashbox.gguf import tokenizer
    return sorted({tok_id("gpt2", p) for p in tokenizer.PRE} | {SPM, GEMMA4})


PATH_RUN = "path/run"
VALID, FLAGGED, BROKEN = "VALID", "FLAGGED", "BROKEN"


def row_contracts(arch, types, tok_model, pre, moe=False):
    """the contract set of a GGUF row: arch/<arch>, op/<type> for every tensor type (a MoE row also op/<type>:experts,
    the stacked expert path), the tokenizer's tok/ id, and path/run; sorted, nothing else"""
    ops = sorted(set(types))
    ids = {f"arch/{arch}", tok_id(tok_model, pre), PATH_RUN} | {f"op/{t}" for t in ops}
    if moe:
        ids |= {f"op/{t}:experts" for t in ops}
    return sorted(ids)


def row_state(contract_set, named, results, named_cases=None):
    """-> (VALID | FLAGGED | BROKEN, reasons) for a row whose record named `named` ({id: fingerprint}) against the
    current results ({id: {status, fingerprint, case_sha256}}). BROKEN: an id of the set is not named by the record, is
    missing from the results, or is not PASS. FLAGGED: all PASS and an exact output the record named moved (one QUICK
    re-run). Else VALID. Ids outside the set are never read.
    named_cases ({id: {node id: sha256}}, from the record's own results file): an id whose fingerprint moved has moved
    only when a case the record named changed or is gone; a case added since moves nothing the record named (it still
    has to PASS). An id with no named cases, or no case digests now, moved."""
    broken, moved = [], []
    for i in contract_set:
        r = results.get(i)
        if i not in named:
            broken.append(f"{i}: the record names no fingerprint for it")
        elif r is None:
            broken.append(f"{i}: not in the contract results (not run)")
        elif r.get("status") != "PASS":
            broken.append(f"{i}: {r.get('status')}")
        elif r.get("fingerprint") != named[i]:
            was, now = (named_cases or {}).get(i), r.get("case_sha256")
            if not was or not isinstance(now, dict):
                moved.append(f"{i}: fingerprint {str(named[i])[:12]} -> {str(r.get('fingerprint'))[:12]}")
                continue
            gone = sorted(n for n in was if n not in now)
            diff = sorted(n for n in was if n in now and now[n] != was[n])
            if gone or diff:
                moved.append(f"{i}: {len(diff)} named case(s) moved {diff[:3]}, {len(gone)} gone {gone[:3]}")
    if broken:
        return BROKEN, broken + moved
    return (FLAGGED, moved) if moved else (VALID, [])


# ---- the guard reader: `python3 tools/contracts.py check` (tests/contracts/test_guard_reader.py holds its rules) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NO_FIELD = "the record names no contracts (it ran before the contract guard): one QUICK, or the switch-over seed"


def _scan():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gguf_scan", os.path.join(ROOT, "tools", "gguf_scan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pinned(pin, what):
    """(bytes, None) when pin {path, sha256} reads with that sha256, else (None, why)"""
    if not isinstance(pin, dict) or not pin.get("path") or not pin.get("sha256"):
        return None, f"{what}: the record pins no {{path, sha256}}"
    try:
        data = open(pin["path"], "rb").read()
    except OSError as e:
        return None, f"{what}: {pin['path']} cannot be read ({e.__class__.__name__})"
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        return None, f"{what}: {pin['path']} is not the file the record pins (sha256)"
    return data, None


def vouches(doc):
    """None when a contract results document can vouch for code (schema 1, a commit, a clean tree), else why not"""
    if not isinstance(doc, dict) or doc.get("schema") != 1 or not isinstance(doc.get("contracts"), dict):
        return "not a schema 1 contract results file"
    if not doc.get("code_commit"):
        return "it names no code commit"
    if doc.get("dirty") is not False:
        return f"it ran on a tree that was not clean (dirty={doc.get('dirty')})"
    return None


def entry_state(row, ent, current, scan):
    """-> (state, reasons, contract set) of one chip record `ent` of catalog row `row` against `current` (results)"""
    data, why = _pinned(ent.get("header"), "header")
    if why:
        return BROKEN, [why], []
    try:
        meta = scan.parse(data)[0]
    except Exception as e:      # noqa: BLE001 - any parse failure of a pinned header is a broken record
        return BROKEN, [f"header: not a readable GGUF header ({e.__class__.__name__})"], []
    if meta.get("general.architecture") != row.get("arch"):
        return BROKEN, [f"header: general.architecture {meta.get('general.architecture')!r} is not the row's "
                        f"{row.get('arch')!r}"], []
    cset = row_contracts(row["arch"], row.get("types") or [], meta.get("tokenizer.ggml.model"),
                         meta.get("tokenizer.ggml.pre"), moe=row.get("moe") is True)
    c = ent.get("contracts")
    if c is None:
        bad = [f"{i}: {(current.get(i) or {}).get('status', 'not in the contract results (not run)')}"
               for i in cset if (current.get(i) or {}).get("status") != "PASS"]
        return (BROKEN, bad + [NO_FIELD], cset) if bad else (FLAGGED, [NO_FIELD], cset)
    named = c.get("fingerprints")
    if not isinstance(named, dict):
        return BROKEN, ["contracts: no fingerprints"], cset
    data, why = _pinned(c.get("results"), "contracts results")
    if why:
        return BROKEN, [why], cset
    try:
        doc = json.loads(data)
    except ValueError:
        return BROKEN, ["contracts results: not JSON"], cset
    why = vouches(doc)
    if why:
        return BROKEN, [f"contracts results: {why}"], cset
    held, bad = doc["contracts"], []
    for i, fp in sorted(named.items()):
        h = held.get(i) or {}
        if h.get("status") != "PASS" or h.get("fingerprint") != fp:
            bad.append(f"{i}: the record names {str(fp)[:12]}, its results file holds {h.get('status')} "
                       f"{str(h.get('fingerprint'))[:12]}")
    if bad:
        return BROKEN, bad, cset
    state, reasons = row_state(cset, named, current, {i: (held.get(i) or {}).get("case_sha256") for i in named})
    if c.get("seed"):
        reasons = reasons + [f"seeded at {str(c['seed'].get('commit'))[:12]}"]
    return state, reasons, cset


def check(results, evidence, models):
    """-> (exit code, report lines, rows): every chip record of every GGUF row of `models` judged on `results`"""
    doc = json.load(open(results))
    why = vouches(doc)
    head = (f"contracts: {results} sha256 {hashlib.sha256(open(results, 'rb').read()).hexdigest()[:12]} "
            f"code {str(doc.get('code_commit'))[:12]} dirty {doc.get('dirty')}")
    if why:
        return 2, [head, f"these results vouch for no code: {why}; nothing judged"], []
    cur = doc["contracts"]
    st = {s: sum(1 for v in cur.values() if v.get("status") == s) for s in ("PASS", "FAIL", "SKIP")}
    lines = [head + f", {len(cur)} ids ({st['PASS']} PASS, {st['FAIL']} FAIL, {st['SKIP']} SKIP)"]
    ev = json.load(open(evidence)).get("gguf") or {}
    scan, rows = _scan(), []
    for row in json.load(open(models)).get("gguf") or []:
        ents = ev.get(row["id"])
        if not ents:
            rows.append({"row": row["id"], "chip": "", "state": BROKEN, "set": [],
                         "reasons": ["the catalog lists the row and the evidence has no record of it"]})
            continue
        for chip in sorted(ents):
            s, why, cset = entry_state(row, ents[chip], cur, scan)
            rows.append({"row": row["id"], "chip": chip, "state": s, "set": cset, "reasons": why})
    for r in rows:
        lines.append(f"{r['state']:<8} {r['row']} {r['chip'] or '-'}" + (f"  {'; '.join(r['reasons'])}" if r["reasons"] else ""))
    n = {s: sum(1 for r in rows if r["state"] == s) for s in (VALID, FLAGGED, BROKEN)}
    lines.append(f"records: {n[VALID]} VALID, {n[FLAGGED]} FLAGGED (QUICK queue), {n[BROKEN]} BROKEN, of {len(rows)} "
                 f"chip records of {len({r['row'] for r in rows})} GGUF rows")
    return (1 if n[BROKEN] else 0), lines, rows


def _row(models, rid):
    for g in json.load(open(models)).get("gguf") or []:
        if g["id"] == rid:
            return g
    raise SystemExit(f"{models}: no GGUF row {rid}")


def field(results, row, header_path, scan=None):
    """-> (field, None) or (None, why): the contracts field of a chip record of `row` whose file's header is at
    header_path, from the clean results file `results` (every id of the set PASS there)"""
    data = open(results, "rb").read()
    doc = json.loads(data)
    why = vouches(doc)
    if why:
        return None, f"{results}: {why}"
    meta = (scan or _scan()).parse(open(header_path, "rb").read())[0]
    if meta.get("general.architecture") != row.get("arch"):
        return None, f"{header_path}: general.architecture {meta.get('general.architecture')!r} is not {row.get('arch')!r}"
    cset = row_contracts(row["arch"], row.get("types") or [], meta.get("tokenizer.ggml.model"),
                         meta.get("tokenizer.ggml.pre"), moe=row.get("moe") is True)
    bad = [f"{i}: {(doc['contracts'].get(i) or {}).get('status', 'not run')}" for i in cset
           if (doc["contracts"].get(i) or {}).get("status") != "PASS"]
    if bad:
        return None, f"{row['id']}: not every contract of its set PASSes in {results}: " + "; ".join(bad)
    return {"set": cset, "fingerprints": {i: doc["contracts"][i]["fingerprint"] for i in cset},
            "results": {"path": os.path.abspath(results), "sha256": hashlib.sha256(data).hexdigest()}}, None


def seed(results, evidence, models, entries, commit, old_guard, out):
    """the switch-over: a copy of `evidence` at `out` where each listed record ("<row> <chip>" lines, the records the old
    guard passed at `commit`) names its contracts from `results`, which must be that commit's clean results. -> (0, lines)
    or (1, why); nothing is written on any refusal."""
    doc = json.load(open(results))
    why = vouches(doc)
    if why or doc["code_commit"] != commit:
        return 1, [f"{results}: {why or 'its code commit ' + str(doc['code_commit'])[:12] + ' is not ' + commit[:12]}"]
    gdata = open(old_guard, "rb").read()
    pin = {"path": os.path.abspath(old_guard), "sha256": hashlib.sha256(gdata).hexdigest()}
    ev, scan, lines, bad = json.load(open(evidence)), _scan(), [], []
    for ln in open(entries).read().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        rid, chip = ln.split()[:2]
        ent = (ev.get("gguf") or {}).get(rid, {}).get(chip)
        if ent is None:
            bad.append(f"{rid} {chip}: no such record")
            continue
        data, why = _pinned(ent.get("header"), "header")
        if why:
            bad.append(f"{rid} {chip}: {why}")
            continue
        tmp = os.path.join(os.path.dirname(os.path.abspath(out)), f".hdr-{os.getpid()}")
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            fld, why = field(results, _row(models, rid), tmp, scan)
        finally:
            os.unlink(tmp)
        if why:
            bad.append(why)
            continue
        ent["contracts"] = dict(fld, seed={"commit": commit, "old_guard": pin})
        lines.append(f"SEEDED   {rid} {chip}  {len(fld['set'])} ids")
    if bad:
        return 1, bad
    with open(out + ".tmp", "w") as f:
        json.dump(ev, f, indent=1)
        f.write("\n")
    os.replace(out + ".tmp", out)
    return 0, lines + [f"seeded {len(lines)} records from {results} (code {commit[:12]}) into {out}"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="contracts.py", description="contract row rule: which GGUF records still stand")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="VALID / FLAGGED / BROKEN for every chip record of every GGUF row")
    c.add_argument("--results", required=True, help="this code's contract results (MASHBOX_CONTRACTS_OUT)")
    c.add_argument("--evidence", default=os.path.join(ROOT, "internal", "catalog-evidence.json"))
    c.add_argument("--models", default=os.path.join(ROOT, "models.json"))
    c.add_argument("--json", help="also write {results, rows} here")
    f = sub.add_parser("field", help="print a chip record's contracts field from a clean results file")
    f.add_argument("--results", required=True)
    f.add_argument("--row", required=True, help="the GGUF row id in models.json")
    f.add_argument("--header", required=True, help="the record's pinned header of the file")
    f.add_argument("--models", default=os.path.join(ROOT, "models.json"))
    s = sub.add_parser("seed", help="switch-over: name contracts in the records the old guard passed (a new file)")
    s.add_argument("--results", required=True, help="the clean contract results of --commit")
    s.add_argument("--evidence", default=os.path.join(ROOT, "internal", "catalog-evidence.json"))
    s.add_argument("--models", default=os.path.join(ROOT, "models.json"))
    s.add_argument("--entries", required=True, help='"<row> <chip>" per line: the records the old guard passed')
    s.add_argument("--commit", required=True, help="the commit the old guard and the results ran on")
    s.add_argument("--old-guard", required=True, help="the old guard's output at --commit (pinned by sha256)")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "field":
        fld, why = field(a.results, _row(a.models, a.row), a.header)
        if why:
            print(why, file=sys.stderr)
            return 1
        print(json.dumps(fld, indent=1, sort_keys=True))
        return 0
    if a.cmd == "seed":
        code, lines = seed(a.results, a.evidence, a.models, a.entries, a.commit, a.old_guard, a.out)
        print("\n".join(lines), file=sys.stderr if code else sys.stdout)
        return code
    code, lines, rows = check(a.results, a.evidence, a.models)
    print("\n".join(lines))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"results": a.results, "exit": code, "rows": rows}, f, indent=1, sort_keys=True)
            f.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
