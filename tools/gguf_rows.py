#!/usr/bin/env python3
"""GGUF catalog rows generated from sealed gguf_conform.py records, never typed. Status is what was tested:

  cpu_check   "passing" only for a record with verdict cpu-checked (its numbers copied from the record);
  TPU         a row's v6e status is "passing" only from its own chip record (GGUF-REC); a file with no chip run reads
              "Not tested yet" (status "pending" + not_run_reason), and names the arch x block-type smoke that covers it
              (same architecture, every quantized type of the file inside the smoke's) with that smoke's own number.
  MoE         never listed until its routed-expert kernel record exists: counted, not written.

  python tools/gguf_rows.py RECORDS_DIR [RECORDS_DIR...] --models src/mashbox/models.json --out rows.json
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
UNQ = ("F32", "F16", "BF16")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def sums_ok(d):
    """every line of the folder's SHA256SUMS holds, and every file of the folder is listed"""
    lines = dict(reversed(x.split("  ", 1)) for x in open(os.path.join(d, "SHA256SUMS")).read().split("\n") if x)
    have = {n for n in os.listdir(d) if n != "SHA256SUMS"}
    return set(lines) == have and all(hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() == h
                                      for n, h in lines.items())


def smokes(models):
    """the chip runs the catalog already holds: (arch, quantized types, row id, accelerator, tok/s 1 user)"""
    out = []
    for g in json.load(open(models)).get("gguf") or []:
        r = (g.get("chips") or {}).get("v6e") or {}
        if r.get("status") == "passing":
            out.append((g["arch"], frozenset(t for t in g["types"] if t not in UNQ), g["id"], r.get("accelerator"),
                        (r.get("measured") or {}).get("decode_tok_s_1_user"), g["sha256"]))
    return out


# A GGUF row's maker: the organisation its base model's card names, keyed by the Hugging Face org of that base (base is
# org/name). The one table: rows get gguf[].maker from it and the release check reads it here.
MAKER_NAMES = {"google": "Google DeepMind", "ibm-granite": "IBM", "LGAI-EXAONE": "LG AI Research", "meta-llama": "Meta",
               "mistralai": "Mistral AI", "Qwen": "Qwen", "tiiuae": "Technology Innovation Institute"}
# A GGUF repo whose card metadata names no base model: the original model its own card text links (at the pinned revision).
ORIGINAL = {"bartowski/Mistral-7B-Instruct-v0.3-GGUF": "mistralai/Mistral-7B-Instruct-v0.3"}


def upstream(repo, card_base):
    """(base, maker): base is the base model's Hugging Face repo as org/name (a bare base_model is a repo of the card's
    own org); maker is MAKER_NAMES of that org. An unknown base or org raises: a row never carries a guessed maker."""
    base = card_base or ORIGINAL.get(repo)
    if not base:
        raise ValueError(f"{repo}: the card names no base model")
    if "/" not in base:
        base = f"{repo.split('/')[0]}/{base}"
    org = base.split("/")[0]
    if org not in MAKER_NAMES:
        raise ValueError(f"{repo}: base {base}: org {org} has no maker name in MAKER_NAMES")
    return base, MAKER_NAMES[org]


def row(d, rec, smk):
    import gguf_scan as S
    meta = S.parse(open(os.path.join(d, "header.bin"), "rb").read())[0]
    card = json.load(open(os.path.join(d, "card.json")))
    cd = card.get("cardData") or {}
    pin, c, tk = rec["pin"], rec["c_cpu"], rec["b_tokenizer"]
    base = cd.get("base_model")
    base = base[0] if isinstance(base, list) else base
    name = (base or meta.get("general.basename") or pin["repo"].split("/")[1]).split("/")[-1]
    size = meta.get("general.size_label")
    tot = float(size[:-1]) if isinstance(size, str) and re.fullmatch(r"\d+(\.\d+)?B", size) else None
    up, maker = upstream(pin["repo"], base)
    quants = frozenset(t for t in rec["types"] if t not in UNQ)
    same = c["identical_weights"]
    g = dict(id=slug(f"{pin['repo'].split('/')[0]}-{name}-{pin['tag'] or rec['tpu_key'].split(' x ')[1]}"),
             model=slug(name), publisher=pin["repo"].split("/")[0], family=meta.get("general.basename") or name,
             maker=maker, repo=pin["repo"], revision=pin["revision"], file=pin["file"],
             tag=pin["tag"], sha256=rec["sha256"], bytes=rec["bytes"], arch=rec["arch"], moe=rec["moe"],
             params_b=dict(total=tot, active=None), types=rec["types"], license=cd.get("license"),
             license_link=cd.get("license_link") or f"https://huggingface.co/{pin['repo']}/blob/{pin['revision']}/README.md",
             base=up, run=f"mashbox run {pin['repo']}:{pin['tag']}", experimental=False)
    g["cpu_check"] = dict(status="passing" if rec["verdict"] == "cpu-checked" else "failing",
                          reference=f"llama.cpp {rec['versions']['llama_cpp'][:7]} CPU",
                          identical_weights=dict(nmse=float(f"{same['nmse']:.2g}"), top1=f"{same['top1_agree']}/{same['positions']}"),
                          block_path=dict(top1=f"{c['top1_agree']}/{c['positions']}", kl=float(f"{c['kl_mean']:.2g}")),
                          greedy=dict(prompts=c["prompts"], tokens=c["n_gen"], full=c["full_match"],
                                      near_tie=c["near_tie_divergences"], other=c["other_divergences"]),
                          tokenizer=dict(texts=tk["texts"], equal=tk["nfc_equal"], explained=tk["nfc_explained"]))
    cover = [s for s in smk if s[0] == rec["arch"] and quants <= s[1]]
    own = [s for s in smk if s[5] == rec["sha256"]]
    if own:
        g["chips"] = {"v6e": dict(status="passing", accelerator=own[0][3], from_row=own[0][2])}
    else:
        g["chips"] = {"v6e": dict(status="pending", not_run_reason="no chip run of this file",
                                  **({"smoke": dict(row=cover[0][2], accelerator=cover[0][3], decode_tok_s_1_user=cover[0][4])}
                                     if cover else {}))}
    ok = g["cpu_check"]
    line = (f"Runs, checked against llama.cpp on CPU: same weights NMSE {ok['identical_weights']['nmse']}, top-1 "
            f"{ok['identical_weights']['top1']}; llama.cpp's own path top-1 {ok['block_path']['top1']}; greedy "
            f"{ok['greedy']['prompts']} x {ok['greedy']['tokens']} tokens: {ok['greedy']['full']} identical, "
            f"{ok['greedy']['near_tie']} near tie")
    if own:
        tpu = f"TPU {own[0][3]}: passing (its own chip record)"
    elif cover:
        tpu = f"TPU {cover[0][3]}: {cover[0][4]} tok/s, 1 user, measured on {cover[0][2]} (same architecture and block types)"
    else:
        tpu = "TPU: Not tested yet"
    return g, line, tpu


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    smk = smokes(a.models)
    rows, held, failed, bad = [], [], [], []
    for root in a.records:
        for d in sorted(glob.glob(os.path.join(root, "*__*"))):
            if not os.path.exists(os.path.join(d, "record.json")):
                continue
            if not sums_ok(d):
                bad.append(d)
                continue
            rec = json.load(open(os.path.join(d, "record.json")))
            if rec.get("moe"):
                held.append(d)
            elif rec["verdict"] != "cpu-checked":
                failed.append(dict(record=os.path.basename(d), verdict=rec["verdict"],
                                   why=rec.get("blocked_by") or (rec.get("c_cpu") or {}).get("error")))
            else:
                try:
                    g, line, tpu = row(d, rec, smk)
                except ValueError as e:
                    failed.append(dict(record=os.path.basename(d), verdict=rec["verdict"], why=str(e)))
                    continue
                rows.append(dict(row=g, text=[line, tpu], record=os.path.basename(d), code=rec["code"]["sha256"],
                                 seconds_total=rec["seconds_total"]))
    json.dump(dict(rows=rows, moe_held=held, not_listed=failed, seal_problems=bad), open(a.out, "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(dict(rows=len(rows), moe_held=len(held), not_listed=len(failed), seal_problems=len(bad))))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
