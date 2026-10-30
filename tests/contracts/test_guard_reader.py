"""The guard reader, written before the code: `tools/contracts.py check` reads the contract results of this code and each
GGUF row's record, and prints every chip record of every row as VALID, FLAGGED or BROKEN.

A chip record of a row (internal/catalog-evidence.json gguf.<row>.<chip>) names its contracts in a `contracts` field:
  set           tools/contracts.py row_contracts of the row at record time
  fingerprints  {id: fingerprint}, each PASS with that fingerprint in the results file below
  results       {path, sha256}: the contract results (MASHBOX_CONTRACTS_OUT) of a clean tree that vouch for the record
  seed          optional {commit, old_guard {path, sha256}}: seeded at the switch-over from a tree the old guard passed
The tokenizer of a row is read from the record's pinned header ({path, sha256}), never from the row.

Rules (bars, written before the first run):
  1. the set is row_contracts(row arch, row types, header tokenizer.ggml.model, header tokenizer.ggml.pre, row moe);
     state = row_state(set, fingerprints, current results, case digests of the record's own results file)
  2. BROKEN, whatever else: the header or the record's results file is missing or not its sha256; the header names
     another architecture; the record's results file is from a dirty tree or does not hold a named fingerprint as PASS
  3. a record with no `contracts` field (it ran before the contract guard): BROKEN when a contract of its set is not
     PASS now, else FLAGGED (one QUICK, or the switch-over seed); never VALID
  4. a catalog row with no record is BROKEN
  5. exit 1 when any record is BROKEN, else 0; exit 2 (and judge nothing) when the current results are from a dirty
     tree or name no commit: they vouch for no code
"""
import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "tools", "contracts.py")


def _gguf_header(arch, model, pre):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", arch), ("tokenizer.ggml.model", model), ("tokenizer.ggml.pre", pre)]
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(s(k) + struct.pack("<I", 8) + s(v) for k, v in kv)


def _pin(path, data):
    data = data if isinstance(data, bytes) else (json.dumps(data, sort_keys=True) + "\n").encode()
    with open(path, "wb") as f:
        f.write(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def _res(ids, status=None, cases=None, dirty=False, commit="c0ffee"):
    out = {}
    for i in ids:
        cs = (cases or {}).get(i) or {f"t.py::{i}": "h-" + i}
        h = hashlib.sha256(json.dumps(cs, sort_keys=True).encode()).hexdigest()
        out[i] = {"status": (status or {}).get(i, "PASS"), "fingerprint": h, "cases": len(cs), "case_sha256": cs}
    return {"schema": 1, "code_commit": commit, "dirty": dirty, "contracts": out}


QW = ["arch/qwen2", "op/F32", "op/Q4_K", "path/run", "tok/qwen2"]
LL = ["arch/llama", "op/F32", "op/Q8_0", "path/run", "tok/spm"]


def _tree(tmp_path, cur, records):
    """records: {row id: (arch, types, (model, pre), contracts field builder or None)}; returns argv for check"""
    rr = tmp_path / "records"
    rr.mkdir(exist_ok=True)
    models, ev = {"gguf": []}, {"records_root": str(rr), "gguf": {}}
    for rid, (arch, types, tok, build) in records.items():
        models["gguf"].append({"id": rid, "arch": arch, "types": types, "moe": False})
        if build is False:
            continue
        ent = {"header": _pin(rr / f"{rid}.gguf", _gguf_header(*(tok[0] if isinstance(tok[0], tuple) else (arch,)),
                                                                 tok[-2], tok[-1]))}
        if build:
            ent["contracts"] = build(rr, rid)
        ev["gguf"][rid] = {"v6e": ent}
    paths = {"models": _pin(tmp_path / "models.json", models)["path"], "evidence": _pin(tmp_path / "ev.json", ev)["path"],
             "results": _pin(tmp_path / "now.json", cur)["path"]}
    return [sys.executable, TOOL, "check", "--results", paths["results"], "--evidence", paths["evidence"],
            "--models", paths["models"], "--json", str(tmp_path / "out.json")]


def _named(ids, doc_edit=None, named_edit=None):
    def build(rr, rid):
        doc = _res(ids)
        if doc_edit:
            doc_edit(doc)
        pin = _pin(rr / f"contracts-{rid}.json", doc)
        named = {i: _res(ids)["contracts"][i]["fingerprint"] for i in ids}
        if named_edit:
            named_edit(named, pin)
        return {"set": ids, "fingerprints": named, "results": pin}
    return build


def _check(tmp_path, argv):
    r = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path)
    out = json.loads((tmp_path / "out.json").read_text()) if (tmp_path / "out.json").exists() else None
    return r, {(x["row"], x["chip"]): x for x in (out or {}).get("rows", [])}


def test_each_record_gets_its_state(tmp_path):
    cur = _res(QW + LL)
    cur["contracts"]["tok/qwen2"]["case_sha256"] = {"t.py::tok/qwen2": "h-tok/qwen2", "t.py::new": "h"}   # case added
    cur["contracts"]["tok/qwen2"]["fingerprint"] = "moved-by-a-new-case"
    recs = {
        "valid": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), _named(QW)),
        "no-field": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), None),
        "spm-valid": ("llama", ["Q8_0", "F32"], ("llama", "default"), _named(LL)),
        "moved": ("llama", ["Q8_0", "F32"], ("llama", "default"),
                  _named(LL, doc_edit=lambda d: d["contracts"]["op/Q8_0"].update(
                      fingerprint="fp-old", case_sha256={"t.py::op/Q8_0": "old"}),
                         named_edit=lambda n, p: n.update({"op/Q8_0": "fp-old"}))),
        "no-record": ("qwen2", ["F32"], ("gpt2", "qwen2"), False),
    }
    r, rows = _check(tmp_path, _tree(tmp_path, cur, recs))
    assert rows[("valid", "v6e")]["state"] == "VALID", rows
    assert rows[("spm-valid", "v6e")]["state"] == "VALID" and rows[("spm-valid", "v6e")]["set"] == LL
    assert rows[("no-field", "v6e")]["state"] == "FLAGGED"
    assert any("names no contracts" in w for w in rows[("no-field", "v6e")]["reasons"])
    assert rows[("moved", "v6e")]["state"] == "FLAGGED"                       # a named case's output moved
    assert any("op/Q8_0" in w for w in rows[("moved", "v6e")]["reasons"])
    assert rows[("no-record", "")]["state"] == "BROKEN"
    assert r.returncode == 1, r.stdout + r.stderr
    for state in ("VALID", "FLAGGED", "BROKEN"):
        assert state in r.stdout


def test_a_named_fingerprint_must_be_a_pass_in_the_records_own_results(tmp_path):
    cur = _res(QW)
    bad = {
        "fp-not-in-file": _named(QW, named_edit=lambda n, p: n.update({"op/Q4_K": "typed-by-hand"})),
        "fail-in-file": _named(QW, doc_edit=lambda d: d["contracts"]["op/F32"].update(status="FAIL")),
        "dirty-file": _named(QW, doc_edit=lambda d: d.update(dirty=True)),
        "file-sha": _named(QW, named_edit=lambda n, p: p.update(sha256="0" * 64)),
    }
    recs = {k: ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), b) for k, b in bad.items()}
    recs["other-arch"] = ("qwen2", ["Q4_K", "F32"], (("llama",), "gpt2", "qwen2"), _named(QW))
    r, rows = _check(tmp_path, _tree(tmp_path, cur, recs))
    for k in recs:
        assert rows[(k, "v6e")]["state"] == "BROKEN", (k, rows.get((k, "v6e")))
    assert r.returncode == 1


def test_a_failing_contract_breaks_every_row_that_names_it_and_only_those(tmp_path):
    cur = _res(QW + LL, status={"op/Q8_0": "FAIL"})
    recs = {"q": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), _named(QW)),
            "l": ("llama", ["Q8_0", "F32"], ("llama", "default"), _named(LL)),
            "l-old": ("llama", ["Q8_0", "F32"], ("llama", "default"), None)}
    r, rows = _check(tmp_path, _tree(tmp_path, cur, recs))
    assert rows[("q", "v6e")]["state"] == "VALID"
    assert rows[("l", "v6e")]["state"] == "BROKEN" and rows[("l-old", "v6e")]["state"] == "BROKEN"
    assert r.returncode == 1


def test_results_from_a_dirty_tree_judge_nothing(tmp_path):
    recs = {"q": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), _named(QW))}
    for cur in (_res(QW, dirty=True), _res(QW, commit=None)):
        r, _ = _check(tmp_path, _tree(tmp_path, cur, recs))
        assert r.returncode == 2 and "vouch" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_all_valid_exits_0(tmp_path):
    recs = {"q": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), _named(QW))}
    r, rows = _check(tmp_path, _tree(tmp_path, _res(QW), recs))
    assert r.returncode == 0 and rows[("q", "v6e")]["state"] == "VALID", r.stdout + r.stderr


def _field_argv(tmp_path, results, rid, header, *extra):
    return [sys.executable, TOOL, "field", "--results", str(results), "--row", rid, "--header", str(header),
            "--models", str(tmp_path / "models.json"), *extra]


def test_the_field_writer_names_what_check_reads(tmp_path):
    """`field` prints a record's contracts field from a clean results file: the row's set, each id's fingerprint (all
    PASS), the results file pinned by sha256; check then reads that record as VALID. A set with an id not PASS in the
    results, or results from a dirty tree, write no field (exit 1)."""
    recs = {"q": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), None)}
    argv = _tree(tmp_path, _res(QW), recs)
    rr = tmp_path / "records"
    good = _pin(rr / "contracts-q.json", _res(QW))
    r = subprocess.run(_field_argv(tmp_path, good["path"], "q", rr / "q.gguf"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    field = json.loads(r.stdout)
    assert field["set"] == QW and field["results"] == good and sorted(field["fingerprints"]) == QW
    ev = json.loads((tmp_path / "ev.json").read_text())
    ev["gguf"]["q"]["v6e"]["contracts"] = field
    _pin(tmp_path / "ev.json", ev)
    out, rows = _check(tmp_path, argv)
    assert rows[("q", "v6e")]["state"] == "VALID", rows
    for bad in (_res(QW, status={"op/Q4_K": "FAIL"}), _res(QW, dirty=True), _res(QW[:-1])):
        p = _pin(rr / "bad.json", bad)
        r = subprocess.run(_field_argv(tmp_path, p["path"], "q", rr / "q.gguf"), capture_output=True, text=True)
        assert r.returncode == 1 and not r.stdout.strip(), (r.stdout, r.stderr)


def test_the_seed_names_only_the_listed_records_and_only_from_the_old_guards_commit(tmp_path):
    """`seed` writes the contracts field (with seed {commit, old_guard}) into a copy of the evidence for the listed
    records only (the ones the old guard passed at --commit), from results of that same commit; it refuses results of
    another commit or a dirty tree and writes nothing then."""
    recs = {"q": ("qwen2", ["Q4_K", "F32"], ("gpt2", "qwen2"), None),
            "l": ("llama", ["Q8_0", "F32"], ("llama", "default"), None)}
    argv = _tree(tmp_path, _res(QW + LL), recs)
    rr = tmp_path / "records"
    res = _pin(rr / "contracts-seed.json", _res(QW + LL, commit="c0ffee"))
    guard = _pin(rr / "old-guard.txt", b"q v6e: 0 problems\n")
    (tmp_path / "pass.txt").write_text("q v6e\n")
    seed = [sys.executable, TOOL, "seed", "--results", res["path"], "--evidence", str(tmp_path / "ev.json"),
            "--models", str(tmp_path / "models.json"), "--entries", str(tmp_path / "pass.txt"),
            "--old-guard", guard["path"], "--out", str(tmp_path / "ev2.json")]
    r = subprocess.run(seed + ["--commit", "c0ffee"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    ev2 = json.loads((tmp_path / "ev2.json").read_text())
    c = ev2["gguf"]["q"]["v6e"]["contracts"]
    assert c["seed"] == {"commit": "c0ffee", "old_guard": guard} and c["set"] == QW
    assert "contracts" not in ev2["gguf"]["l"]["v6e"]
    argv[argv.index("--evidence") + 1] = str(tmp_path / "ev2.json")
    _, rows = _check(tmp_path, argv)
    assert rows[("q", "v6e")]["state"] == "VALID" and rows[("l", "v6e")]["state"] == "FLAGGED", rows
    (tmp_path / "ev2.json").unlink()
    for bad_commit in ("deadbeef",):
        r = subprocess.run(seed + ["--commit", bad_commit], capture_output=True, text=True)
        assert r.returncode == 1 and not (tmp_path / "ev2.json").exists(), r.stdout + r.stderr
