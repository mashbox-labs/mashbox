#!/usr/bin/env python3
"""Seal one quick-check chip run of a GGUF file (record_kind "quick-check") into a new folder under the records root, and
print its evidence entry and the row's chip block. Every value is copied from the run's own files, never typed:

  four runs of `mashbox run REPO:TAG --prompt P --max-tokens 64 --temperature 0` (runN.json/.rc/.stdout/.stderr, with
  the SHA256SUMS written next to them on the chip), the compare with llama.cpp's greedy ids (cmp.json, refN.ids), the
  chip's setup log, pip freeze and accelerator type, the prompts and the code commit the wheel was built from, the
  compare tool, and the file's CPU check record (its header, pinned card and pin: file, sha256, bytes, revision).

  python3 tools/gguf_quick_seal.py --runs DIR --compare DIR --chip DIR --kit DIR --tool cmp_p1.py --cpu-record DIR \
      --tag NAME --out NEW_DIR

Every input is checked against its own SHA256SUMS first; the output folder must not exist (a seal is never edited).
The row's number is decode tok/s at 1 user = sum(new tokens) / sum(decode seconds) over runs 1-3 (run 0 holds the
first compile), 2 decimals: the record guard recomputes it.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

RUNS, TOKENS = 4, 64


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--runs", "--compare", "--chip", "--kit", "--tool", "--cpu-record", "--tag", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: a seal is never edited")
    runs = checked(a.runs, need=[f"run{i}{x}" for i in range(RUNS) for x in (".json", ".rc", ".stdout", ".stderr")])
    cmp = checked(a.compare, need=["cmp.json", *(f"ref{i}.ids" for i in range(RUNS))])
    chip = checked(a.chip, "SHA256SUMS-chipkit", need=["setup.log", "freeze.txt", "accelerator-type.txt"])
    kit = checked(a.kit, "SHA256SUMS-kit", need=["CODE_SHA", *(f"in/prompt{i}.txt" for i in range(RUNS))])
    cpu = checked(a.cpu_record, need=["header.bin", "card.json", "record.json"])
    pin = json.load(open(cpu["record.json"]))["pin"]
    ids = set()
    for i in range(RUNS):
        for line in open(runs[f"run{i}.stderr"], errors="replace"):
            if line.startswith('{"event": "gguf_identity"'):
                ids.add(json.loads(line)["sha256"])
    if ids != {pin["lfs_sha256"]}:
        sys.exit(f"the runs read {sorted(ids)}, the CPU record pins {pin['lfs_sha256']}")
    copies = {**{n: runs[n] for n in runs if n != "SHA256SUMS"}, "cmp.json": cmp["cmp.json"],
              **{f"ref{i}.ids": cmp[f"ref{i}.ids"] for i in range(RUNS)},
              **{n: chip[n] for n in ("setup.log", "freeze.txt", "accelerator-type.txt")},
              "CODE_SHA": kit["CODE_SHA"], **{f"prompt{i}.txt": kit[f"in/prompt{i}.txt"] for i in range(RUNS)},
              "cmp_p1.py": a.tool, f"header-{a.tag}.gguf": cpu["header.bin"], "card-api.json": cpu["card.json"],
              "cpu-record.json": cpu["record.json"]}
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
    evidence = {chip_key: {"kind": "quick-check", "sha256sums": src("SHA256SUMS"), "compare": src("cmp.json"),
                           "file": src(f"file-{a.tag}.sha256"), "plan": src(f"plan-{a.tag}.txt"),
                           "card": src("card-api.json"), "header": src(f"header-{a.tag}.gguf"),
                           "code": {"commit": commit}}}
    s = json.load(open(cmp["cmp.json"]))["summary"]
    status = "passing" if s.get("runs_ok") == RUNS and s.get("other") == 0 else "failing"
    run_json = [json.load(open(runs[f"run{i}.json"])) for i in range(RUNS)]
    row_chip = {chip_key: {"status": status, "accelerator": acc, "kernel_path": "exact", "record_kind": "quick-check",
                           "measured": {"decode_tok_s_1_user": decode(run_json)}}}
    print(json.dumps({"evidence": evidence, "chips": row_chip, "pin": {k: pin[k] for k in (
        "repo", "revision", "file", "tag", "lfs_sha256", "bytes")}}, indent=1))


if __name__ == "__main__":
    main()
