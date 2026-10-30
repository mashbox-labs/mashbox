#!/usr/bin/env python3
"""Seal one quick-check chip run of a GGUF file (record_kind "quick-check") into a new folder under the records root, and
print its evidence entry and the row's chip block. Every value is copied from the run's own files, never typed:

  four runs of `mashbox run REPO:TAG --prompt P --max-tokens 64 --temperature 0` (runN.json/.rc/.stdout/.stderr, with
  the SHA256SUMS written next to them on the chip), the compare with llama.cpp's greedy ids (cmp.json, refN.ids), the
  chip's setup log, pip freeze and accelerator type, the prompts and the code commit the wheel was built from, the
  compare tool, and the file's CPU check record (its header, pinned card and pin: file, sha256, bytes, revision).

  python3 tools/gguf_quick_seal.py --runs DIR --compare DIR --chip DIR --kit DIR --tool cmp_p1.py --cpu-record DIR \
      --tag NAME --out NEW_DIR

A QUICK run (--kind quick, record_kind "quick") is the same four runs with --max-tokens 32, its compare tool sealed as
cmp_q.py, and no CPU record: --pin DIR holds the pick's pin.json (repo, revision, file, tag, lfs_sha256, bytes), the
file's header.bin and the card.json read at that revision, under their own SHA256SUMS. It also prints the row, built
from the header and the card the way tools/gguf_rows.py builds one (no cpu_check):

  python3 tools/gguf_quick_seal.py --kind quick --runs DIR --compare DIR --chip DIR --kit DIR --tool cmp_q.py \
      --pin DIR --tag NAME --out NEW_DIR

kernel_path comes from the runs' own events, never typed: "bundle" when all four ran the engine_kernels row's package
(--models: the code commit's models.json, --kernel-wheel: the sha256sum line of the package file the engine fetched)
with 0% of the weights and 0 bytes on the exact path, more than 0 bytes on a kernel and no fallback line (BF16 on the
native bf16 path counts as served: bf16_served); "exact" when
all four ran no kernel type and no kernel byte; anything else is mixed and gets no row (the tool stops). chip_count is
the runs' recorded device count (their observer's len(jax.devices())) when they recorded one.

--ngram-runs DIR --ngram-compare DIR (kind quick): the file's four `--speculation ngram` runs and their compare, sealed
in the same folder (ngram-run*, ngram-cmp.json), and the row's gate entry speculation.ngram.<chip key> printed: chip
key, component set, speculation step version and draft k from the n-gram runs' own events, k_table {"1": k} and 0 at
the engine's other row buckets, status passed only with the QUICK bars on the n-gram runs, a passing plain kernel-path
row, and n-gram decode tok/s (runs 1-3) at least the plain one; record gguf/<row id>/ngram/<chip key> (--row-id, else
the printed row's id) with the seal's SHA256SUMS sha256.

Every input is checked against its own SHA256SUMS first; the output folder must not exist (a seal is never edited).
The row's number is decode tok/s at 1 user = sum(new tokens) / sum(decode seconds) over runs 1-3 (run 0 holds the
first compile), 2 decimals: the record guard recomputes it.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

RUNS, TOKENS = 4, 64
KINDS = {"quick-check": (64, "cmp_p1.py"), "quick": (32, "cmp_q.py")}       # record kind -> (tokens, sealed tool name)
FALLBACK = ("did not load", "have no TPU kernel", "kq planes for")      # a type or the package on the exact path
BUCKETS = ("1", "4", "16", "64", "256")      # the engine's row buckets (runtime/engine.py _rows)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def checked(d, sums="SHA256SUMS", need=()):
    """{name: path} of the files a folder's seal lists, each with its sha256; every name in need must be there"""
    out = {}
    for line in open(os.path.join(d, sums)).read().splitlines():
        if not line.strip():
            continue
        h, n = line.split(None, 1)
        n = n.lstrip("*")
        p = os.path.join(d, n)
        if not os.path.isfile(p):
            continue                # a file kept on another host (llama.cpp's logits): listed, not copied
        if sha(p) != h:
            sys.exit(f"{p}: sha256 differs from {sums}")
        out[n] = p
    lost = [n for n in need if n not in out]
    if lost:
        sys.exit(f"{d}: {sums} does not hold {lost}")
    return out


def decode(runs):
    t = [r["generate_calls"][0]["timing"] for r in runs[1:]]
    return round(sum(x["new_tokens"] for x in t) / sum(x["decode_s"] for x in t), 2)


def events(path):
    """event name -> the first JSON event line of that name in a run's stderr"""
    out = {}
    for line in open(path, errors="replace"):
        if line.startswith('{"event"'):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            out.setdefault(x.get("event"), x)
    return out


BF16_RE = re.compile(r"^mashbox: BF16 \(([0-9.]+)% of the weights\) have no TPU kernel")


def bf16_served(gp, w, text):
    """BF16 on the native bf16 path (Dense16) counts as served (Dense16 reads a BF16 weight as fast as XLA's own plain
    read of the same bytes, measured on a v6e chip): True only when the
    run's one exact-path line names BF16 alone, no other fallback line, its load put 0 bytes on the exact path and more
    than 0 on Dense16, and the plan's share is Dense16's bytes over the plan's weights (to its rounding: share 0.1
    points, weights_gb 0.01 GB)"""
    lines = [x for x in text.splitlines() if "have no TPU kernel" in x]
    m = BF16_RE.match(lines[0]) if len(lines) == 1 else None
    nb, gb, share = w.get("native_bf16_bytes"), gp.get("weights_gb"), gp.get("reference_share")
    if not m or any(x in text for x in FALLBACK if x != "have no TPU kernel") or w.get("reference_bytes") != 0:
        return False
    if (not isinstance(nb, int) or isinstance(nb, bool) or nb <= 0 or isinstance(gb, bool) or not isinstance(gb, (int, float))
            or gb <= 0 or isinstance(share, bool) or not isinstance(share, (int, float)) or float(m.group(1)) != share):
        return False
    est = 100.0 * nb / (gb * 1e9)
    return abs(share - est) <= 0.05 + est * 0.005 / gb + 1e-9


NEAR_TIE_GAP = 0.5      # cmp_q.py: a first difference is a near tie when llama.cpp's own top-two gap is under 0.5


def fail_reason(summary, rows, forced=None):
    """why a QUICK did not pass (catalog.QUICK_FAIL_REASONS), first match: "run" (fewer than 4 runs, or a forced run,
    load and print), "first token" (a difference at step 0 that is no near tie), "divergence" (such a difference later,
    or a forced continuation with one), "near tie" (a token-0 near tie with no forced run)"""
    forced = forced or {}
    if summary.get("runs_ok") != RUNS or any(not v[0] for v in forced.values()):
        return "run"
    other = [x for x in rows if x.get("first_diff") is not None and not (
        isinstance(x.get("ref_gap"), (int, float)) and x["ref_gap"] < NEAR_TIE_GAP)]
    if any(x["first_diff"] == 0 for x in other):
        return "first token"
    return "divergence" if other or any(not v[1] for v in forced.values()) else "near tie"


def forced_runs(runs, cmp, truns, tcmp, tokens):
    """{prompt: (loads and prints, 0 other divergences)} of the forced runs (the token-0 near-tie rule):
    one per prompt whose plain first difference is a token-0 near tie; each ran the plain run's
    argv with llama.cpp's first token appended to the plain run's prompt ids, and its compare (tf-cmp.json) is [that
    token] + its ids against llama.cpp's 32. Stops on a forced run of another prompt or one that ran anything else."""
    rows = {x.get("i"): x for x in json.load(open(cmp["cmp.json"]))["rows"]}
    trows = {x.get("i"): x for x in json.load(open(tcmp["tf-cmp.json"]))["rows"]}
    out = {}
    for i in sorted({int(n[6:].split(".")[0]) for n in truns if n.startswith("tf-run")}):
        x = rows.get(i) or {}
        if x.get("first_diff") != 0 or not (isinstance(x.get("ref_gap"), (int, float)) and x["ref_gap"] < NEAR_TIE_GAP):
            sys.exit(f"forced run {i}: prompt {i} has no token-0 near tie (first difference {x.get('first_diff')}, gap "
                     f"{x.get('ref_gap')})")
        for ext in (".json", ".rc", ".stdout", ".stderr"):
            if f"tf-run{i}{ext}" not in truns:
                sys.exit(f"forced run {i}: tf-run{i}{ext} is not sealed")
        run, plain = json.load(open(truns[f"tf-run{i}.json"])), json.load(open(runs[f"run{i}.json"]))
        a_, _, b_ = open(cmp[f"ref{i}.ids"]).read().partition("\n\n")
        rg = [int(t) for t in b_.split()]
        if run.get("argv") != plain.get("argv") or run.get("env"):
            sys.exit(f"forced run {i} ran `mashbox {' '.join(map(str, run.get('argv') or []))}` with {run.get('env')}, the "
                     f"plain run{i} `mashbox {' '.join(map(str, plain.get('argv') or []))}`")
        if run.get("forced") != rg[0]:
            sys.exit(f"forced run {i} appended {run.get('forced')}, llama.cpp's first token is {rg[0]}")
        calls, pc = run.get("generate_calls") or [], (plain.get("generate_calls") or [{}])[0]
        c = calls[0] if len(calls) == 1 else {}
        p0 = (pc.get("prompt_ids") or [[]])[0]
        if (c.get("cli_prompt_ids") or [[]])[0] != p0 or (c.get("prompt_ids") or [[]])[0] != p0 + [rg[0]]:
            sys.exit(f"forced run {i} ran prompt ids {(c.get('prompt_ids') or [[]])[0]}, the plain run{i}'s + [{rg[0]}] "
                     f"are {p0 + [rg[0]]}")
        t = trows.get(i) or {}
        ours = ([rg[0]] + (c.get("ids") or [[]])[0])[:tokens]
        if t.get("ids_ours") != ours:
            sys.exit(f"forced run {i}: tf-cmp.json's ids are not [{rg[0]}] + the forced run's ids")
        rc = open(truns[f"tf-run{i}.rc"]).read().split()[:1]
        loads = rc == ["0"] and run.get("rc") in (0, None) and run.get("error") is None and len(calls) == 1 \
            and bool(open(truns[f"tf-run{i}.stdout"]).read().strip())
        out[i] = (loads, t.get("kind") in ("full", "near"))
    return out


def path_of(runs, n, krow, wheel, file):
    """-> "bundle", "exact" or "mixed: <why>" for the four runs (run{i}.stderr in runs, n the name prefix)"""
    why, exact = [], []
    for i in range(RUNS):
        f = runs[f"{n}run{i}.stderr"] if f"{n}run{i}.stderr" in runs else runs[f"run{i}.stderr"]
        ev, text = events(f), open(f, errors="replace").read()
        gp, w = ev.get("gguf_plan") or {}, ev.get("weights") or {}
        bf16 = bf16_served(gp, w, text)
        exact.append(gp.get("kernel_types") == [] and w.get("kernel_bytes") == 0)
        kt = sorted(map(str, (krow or {}).get("types") or [])) or None
        if gp.get("kernel_types") != kt:
            why.append(f"run{i} kernel types {gp.get('kernel_types')} != the row's {kt}")
        if (gp.get("reference_share") != 0.0 and not bf16) or w.get("reference_bytes") != 0 or w.get("file") != file:
            why.append(f"run{i} {gp.get('reference_share')}% / {w.get('reference_bytes')} bytes on the exact path")
        if not isinstance(w.get("kernel_bytes"), int) or w["kernel_bytes"] <= 0:
            why.append(f"run{i} {w.get('kernel_bytes')} bytes on a kernel")
        said = [x for x in FALLBACK if x in text]
        if said and not bf16:
            why.append(f"run{i} printed {said[0]!r}")
    if all(exact):
        return "exact"
    if krow is None or wheel[:1] != [krow.get("sha256")] or os.path.basename(wheel[1] if len(wheel) > 1 else "") != krow.get("file"):
        why.append(f"the package loaded ({wheel}) is not the engine_kernels row {krow and krow.get('file')}")
    return "bundle" if not why else "mixed: " + "; ".join(why)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--runs", "--compare", "--chip", "--kit", "--tool", "--tag", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--kind", default="quick-check", choices=sorted(KINDS))
    ap.add_argument("--cpu-record", help="quick-check: the file's CPU check record")
    ap.add_argument("--pin", help="quick: the pick's pin.json + header.bin + card.json with their SHA256SUMS")
    ap.add_argument("--models", help="the code commit's models.json (its engine_kernels rows)")
    ap.add_argument("--kernel-wheel", help="sha256sum line of the kernel package file the engine fetched")
    ap.add_argument("--ngram-runs", help="quick: the four --speculation ngram runs with their SHA256SUMS")
    ap.add_argument("--ngram-compare", help="quick: their compare (cmp.json, ref*.ids) with its SHA256SUMS")
    ap.add_argument("--tf-runs", help="quick: the forced runs tf-run<i>.* (the token-0 near-tie rule) of the prompts whose first "
                                      "difference is a token-0 near tie, with their SHA256SUMS")
    ap.add_argument("--tf-compare", help="quick: their compare (tf-cmp.json) with its SHA256SUMS")
    ap.add_argument("--row-id", help="the gguf[] row's id (an existing row), for the gate entry's record name")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: a seal is never edited")
    if (a.kind == "quick") != bool(a.pin) or (a.kind == "quick-check") != bool(a.cpu_record):
        sys.exit("--kind quick takes --pin; --kind quick-check takes --cpu-record")
    tokens, tool = KINDS[a.kind]
    runs = checked(a.runs, need=[f"run{i}{x}" for i in range(RUNS) for x in (".json", ".rc", ".stdout", ".stderr")])
    cmp = checked(a.compare, need=["cmp.json", *(f"ref{i}.ids" for i in range(RUNS))])
    chip = checked(a.chip, "SHA256SUMS-chipkit", need=["setup.log", "freeze.txt", "accelerator-type.txt"])
    kit = checked(a.kit, "SHA256SUMS-kit", need=["CODE_SHA", *(f"in/prompt{i}.txt" for i in range(RUNS))])
    if a.kind == "quick":
        cpu = checked(a.pin, need=["header.bin", "card.json", "pin.json"])
        pin = json.load(open(cpu["pin.json"]))
    else:
        cpu = checked(a.cpu_record, need=["header.bin", "card.json", "record.json"])
        pin = json.load(open(cpu["record.json"]))["pin"]
    for i in range(RUNS):
        argv_i = [str(x) for x in json.load(open(runs[f"run{i}.json"])).get("argv") or []]
        if argv_i[-4:] != ["--max-tokens", str(tokens), "--temperature", "0"] or "--speculation" in argv_i:
            sys.exit(f"run{i} ran {argv_i}: a {a.kind} run is --max-tokens {tokens} --temperature 0, no other flag")
    acc0 = open(chip["accelerator-type.txt"]).read().strip()
    krow = None
    if a.models:
        krow = next((k for k in json.load(open(a.models)).get("engine_kernels") or []
                     if acc0.split("-")[0] in (k.get("chips") or [])), None)
    wheel = open(a.kernel_wheel).read().split() if a.kernel_wheel else []
    kpath = path_of(runs, "", krow, wheel, pin["file"])
    if kpath.startswith("mixed"):
        sys.exit(f"{pin['repo']}:{pin['tag']}: kernel path {kpath}: no row (a mixed kernel path gets none)")
    nruns = ncmp = None
    if a.ngram_runs or a.ngram_compare:
        if a.kind != "quick" or not (a.ngram_runs and a.ngram_compare):
            sys.exit("--ngram-runs and --ngram-compare go together, with --kind quick")
        nruns = checked(a.ngram_runs, need=[f"run{i}{x}" for i in range(RUNS) for x in (".json", ".rc", ".stdout", ".stderr")])
        ncmp = checked(a.ngram_compare, need=["cmp.json", *(f"ref{i}.ids" for i in range(RUNS))])
        for i in range(RUNS):
            if sha(ncmp[f"ref{i}.ids"]) != sha(cmp[f"ref{i}.ids"]):
                sys.exit(f"the n-gram compare's ref{i}.ids is not the plain compare's")
            argv_i = [str(x) for x in json.load(open(nruns[f"run{i}.json"])).get("argv") or []]
            if argv_i[-6:] != ["--speculation", "ngram", "--max-tokens", str(tokens), "--temperature", "0"]:
                sys.exit(f"n-gram run{i} ran {argv_i}")
    truns = tcmp = None
    if a.tf_runs or a.tf_compare:
        if a.kind != "quick" or not (a.tf_runs and a.tf_compare):
            sys.exit("--tf-runs and --tf-compare go together, with --kind quick")
        truns = checked(a.tf_runs)
        tcmp = checked(a.tf_compare, need=["tf-cmp.json"])
    forced = forced_runs(runs, cmp, truns, tcmp, tokens) if truns else {}
    ids = set()
    for i in range(RUNS):
        for line in open(runs[f"run{i}.stderr"], errors="replace"):
            if line.startswith('{"event": "gguf_identity"'):
                ids.add(json.loads(line)["sha256"])
    for i in range(RUNS):
        for line in open(nruns[f"run{i}.stderr"], errors="replace") if nruns else ():
            if line.startswith('{"event": "gguf_identity"'):
                ids.add(json.loads(line)["sha256"])
    if ids != {pin["lfs_sha256"]}:
        sys.exit(f"the runs read {sorted(ids)}, the CPU record pins {pin['lfs_sha256']}")
    copies = {**{n: runs[n] for n in runs if n != "SHA256SUMS"}, "cmp.json": cmp["cmp.json"],
              **{f"ref{i}.ids": cmp[f"ref{i}.ids"] for i in range(RUNS)},
              **{n: chip[n] for n in ("setup.log", "freeze.txt", "accelerator-type.txt")},
              "CODE_SHA": kit["CODE_SHA"], **{f"prompt{i}.txt": kit[f"in/prompt{i}.txt"] for i in range(RUNS)},
              tool: a.tool, f"header-{a.tag}.gguf": cpu["header.bin"], "card-api.json": cpu["card.json"],
              **({"pin.json": cpu["pin.json"]} if a.kind == "quick" else {"cpu-record.json": cpu["record.json"]}),
              **({"kernel-wheel.sha256": a.kernel_wheel} if kpath == "bundle" else {}),
              **({f"ngram-{n}": p for n, p in (nruns or {}).items() if n != "SHA256SUMS"}),
              **({"ngram-cmp.json": ncmp["cmp.json"]} if ncmp else {}),
              **({n: p for n, p in (truns or {}).items() if n != "SHA256SUMS"}),
              **({"tf-cmp.json": tcmp["tf-cmp.json"]} if tcmp else {})}
    os.makedirs(a.out)
    for n, p in copies.items():
        shutil.copyfile(p, os.path.join(a.out, n))
    if "FILE" in cmp:               # the compare host's note of the file it read (same sha256 as the chip's)
        shutil.copyfile(cmp["FILE"], os.path.join(a.out, "compare-FILE.txt"))
    shutil.copyfile(os.path.join(a.compare, "SHA256SUMS"), os.path.join(a.out, "compare-SHA256SUMS.txt"))
    with open(os.path.join(a.out, f"file-{a.tag}.sha256"), "w") as f:
        f.write(f"{pin['lfs_sha256']}  {pin['file']}\n{pin['bytes']}\n")
    with open(os.path.join(a.out, f"plan-{a.tag}.txt"), "w") as f:
        f.write(f"{pin['repo']} {pin['revision']} {pin['file']} {pin['lfs_sha256']}\n")
    names = sorted(os.listdir(a.out))
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        f.write("".join(f"{sha(os.path.join(a.out, n))}  {n}\n" for n in names))

    def src(n):
        p = os.path.abspath(os.path.join(a.out, n))
        return {"path": p, "sha256": sha(p)}
    commit = open(kit["CODE_SHA"]).read().split()[0]
    acc = open(chip["accelerator-type.txt"]).read().strip()
    chip_key = acc.split("-")[0]
    evidence = {chip_key: {"kind": a.kind, "sha256sums": src("SHA256SUMS"), "compare": src("cmp.json"),
                           "file": src(f"file-{a.tag}.sha256"), "plan": src(f"plan-{a.tag}.txt"),
                           "card": src("card-api.json"), "header": src(f"header-{a.tag}.gguf"),
                           "code": {"commit": commit}}}
    s = json.load(open(cmp["cmp.json"]))["summary"]
    status = "passing" if s.get("runs_ok") == RUNS and s.get("other") == 0 else "failing"
    if a.kind == "quick" and any(x.get("first_diff") == 0 and forced.get(x.get("i")) != (True, True)
                                 for x in json.load(open(cmp["cmp.json"]))["rows"]):
        status = "failing"          # QUICK bar (3) with the token-0 near-tie rule: a near tie passes only with its forced run
    run_json = [json.load(open(runs[f"run{i}.json"])) for i in range(RUNS)]
    devs = {r.get("devices") for r in run_json}
    count = {}
    if devs != {None}:
        if len(devs) != 1 or None in devs:
            sys.exit(f"the runs recorded device counts {sorted(devs, key=str)}: one chip count per row")
        count = {"chip_count": devs.pop()}
    # pass/fail only: the runs' decode timing is each fresh process's first request (first-call cost in it), never a
    # row's number; a row's speed comes from its own steady-decode record
    row_chip = {chip_key: {"status": status, "accelerator": acc, **count, "kernel_path": kpath, "record_kind": a.kind}}
    if status == "failing" and a.kind == "quick":
        row_chip[chip_key]["reason"] = fail_reason(s, json.load(open(cmp["cmp.json"]))["rows"], forced)
    out = {"evidence": evidence, "chips": row_chip, "pin": {k: pin[k] for k in (
        "repo", "revision", "file", "tag", "lfs_sha256", "bytes")}}
    if nruns:
        out["speculation"], why = ngram_entry(nruns, json.load(open(ncmp["cmp.json"])), row_chip[chip_key], run_json,
                                              src("SHA256SUMS")["sha256"])
        out["ngram_why"] = why
        evidence["speculation"] = {"ngram": {ck: {"sha256sums": src("SHA256SUMS"), "compare": src("ngram-cmp.json")}
                                             for ck in out["speculation"]["ngram"]}}
    if a.kind == "quick":
        try:            # the seal stands either way; a row needs its maker and base from the card (gguf_rows rules)
            out["row"] = quick_row(pin, os.path.join(a.out, f"header-{a.tag}.gguf"),
                                   os.path.join(a.out, "card-api.json"), row_chip)
        except (ValueError, SystemExit) as e:
            out["row_error"] = str(e)
    if nruns:
        rid = a.row_id or (out.get("row") or {}).get("id")
        if not rid:
            sys.exit(f"no row id for the gate entry's record: --row-id ({out.get('row_error')})")
        for ck, e in out["speculation"]["ngram"].items():
            e["record"] = f"gguf/{rid}/ngram/{ck}"
    print(json.dumps(out, indent=1))


def ngram_entry(nruns, ncmp, plain_row, plain_runs, sums_sha):
    """({"ngram": {chip key: gate entry}}, why lines) from the n-gram runs' own events and compare (see the module
    doc); the record name is filled in by main"""
    ev = [events(nruns[f"run{i}.stderr"]) for i in range(RUNS)]
    rj = [json.load(open(nruns[f"run{i}.json"])) for i in range(RUNS)]
    keys = {r.get("chip_key") for r in rj} | {r.get("chip_key") for r in plain_runs}
    css = {(e.get("deployment") or {}).get("component_set_sha") for e in ev}
    steps = {(e.get("speculation") or {}).get("spec_step_version") for e in ev}
    ks = {(e.get("speculation") or {}).get("draft_k") for e in ev}
    meth = {((e.get("speculation") or {}).get("used"), (e.get("speculation") or {}).get("method")) for e in ev}
    if len(keys) != 1 or None in keys or len(css) != 1 or None in css or len(steps) != 1 or len(ks) != 1 \
            or meth != {(True, "ngram")}:
        sys.exit(f"the n-gram runs disagree or are not n-gram: chip keys {keys}, component sets {css}, step versions "
                 f"{steps}, draft_k {ks}, speculation {meth}")
    k = ks.pop()
    s = ncmp.get("summary") or {}
    why = []
    if s.get("runs_ok") != RUNS:
        why.append(f"{s.get('runs_ok')}/{RUNS} runs load and print")
    if s.get("other"):
        why.append(f"{s.get('other')} other divergence{'s' * (s.get('other') != 1)}")
    first = [x.get("i") for x in ncmp.get("rows") or [] if x.get("first_diff") == 0]
    if first:
        why.append(f"prompt {', '.join(map(str, first))} differ{'s' * (len(first) == 1)} at the first token")
    plain = decode(plain_runs)       # the plain runs sealed in the same folder, as the n-gram runs are timed
    if plain_row["status"] != "passing" or plain_row["kernel_path"] != "bundle":
        why.append(f"the row's plain kernel-path QUICK does not pass ({plain_row['status']}, {plain_row['kernel_path']})")
    fast = decode(rj)
    if fast < plain:
        why.append(f"n-gram {fast} tok/s below plain {plain}")
    ent = {"status": "not_passed" if why else "passed", "record": None, "record_sha256": sums_sha,
           "component_set_sha": css.pop(), "spec_step_version": steps.pop(),
           "k_table": {b: (k if b == "1" else 0) for b in BUCKETS}}
    return {"ngram": {keys.pop(): ent}}, why + [f"n-gram {fast} tok/s, plain {plain}"]


def quick_row(pin, header, card, chips):
    """the gguf[] row of a QUICK seal, from its own header and card only (tools/gguf_rows.py's fields and rules, no
    cpu_check): arch, block types and expert count read by the engine's reader on the header made sparse to the file's
    size; maker and base from the card (gguf_rows.upstream); a mixture of experts gets no row"""
    import re
    import tempfile
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             "..", "src")]
    import gguf_rows as R
    import gguf_scan as S
    from mashbox import gguf as G
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, os.path.basename(pin["file"]))
        with open(p, "wb") as f:
            f.write(open(header, "rb").read())
            f.truncate(pin["bytes"])
        g = G.File(p)
        types, experts = sorted(g.census()[0]), int(g.get(f"{g.arch}.expert_count") or 0)
        arch = g.arch
    if experts:
        sys.exit(f"{pin['repo']}: {experts} experts: no MoE row from a QUICK run (routed-expert kernel record first)")
    meta = S.parse(open(header, "rb").read())[0]
    cd = json.load(open(card)).get("cardData") or {}
    base = cd.get("base_model")
    base = base[0] if isinstance(base, list) else base
    name = (base or meta.get("general.basename") or pin["repo"].split("/")[1]).split("/")[-1]
    size = meta.get("general.size_label")
    tot = float(size[:-1]) if isinstance(size, str) and re.fullmatch(r"\d+(\.\d+)?B", size) else None
    up, maker = R.upstream(pin["repo"], base)
    return dict(id=R.slug(f"{pin['repo'].split('/')[0]}-{name}-{pin['tag']}"), model=R.slug(name),
                publisher=pin["repo"].split("/")[0], family=meta.get("general.basename") or name, maker=maker,
                repo=pin["repo"], revision=pin["revision"], file=pin["file"], tag=pin["tag"], sha256=pin["lfs_sha256"],
                bytes=pin["bytes"], arch=arch, moe=False, params_b=dict(total=tot, active=None), types=types,
                license=cd.get("license"),
                license_link=cd.get("license_link") or f"https://huggingface.co/{pin['repo']}/blob/{pin['revision']}/README.md",
                base=up, **{"in": "text", "out": "text"}, run=f"mashbox run {pin['repo']}:{pin['tag']}",
                experimental=False, chips=chips)


if __name__ == "__main__":
    main()
