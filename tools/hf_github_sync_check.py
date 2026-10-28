#!/usr/bin/env python3
"""Keep GitHub and Hugging Face in sync from one file: models.json.

models.json (in the main repo) is the only source. This script renders every view
from it and checks that the views on disk, on Hugging Face and on GitHub match.

Views:
  README.md            the block between <!-- models:start --> and <!-- models:end -->
  profile README       same markers (optional, --profile PATH)
  llms.txt             the block between <!-- models:start --> and <!-- models:end -->
  hf/<id>/README.md    the Hugging Face card of every model that has an HF repo
                       (rendered from templates/hf/card-<kind>.md in the repo)

Usage:
  hf_github_sync_check.py render --repo ROOT [--profile PATH] [--templates DIR] [--write] [--out DIR]
  hf_github_sync_check.py check  --repo ROOT [--profile PATH] [--templates DIR] [--offline|--online]
                                  [--release] [--fixtures DIR]

  --offline  schema + rendered views equal the files on disk (no network). Default.
  --online   also asks the Hugging Face and GitHub APIs: repo public, pinned revision exists,
             card at that revision equals the rendered card, no repo in the HF org missing
             from models.json, the version tag exists on GitHub.
  --release  stricter: every model with an HF repo has a measured result and a passing chip.
  --fixtures replay API answers from DIR (tests). Tokens, if needed, come from the
             environment (HF_TOKEN, GH_TOKEN), never from arguments.

Exit status: 0 in sync, 1 out of sync or invalid, 2 usage error.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from string import Template

KINDS = {
    "llm-gguf": {"suffix": "-GGUF", "label": "LLM, GGUF"},
    "llm-tpu": {"suffix": "-TPU", "label": "LLM"},
    "world": {"suffix": "-TPU", "label": "World model"},
    "video": {"suffix": "-TPU", "label": "Video"},
    "image": {"suffix": "-TPU", "label": "Image"},
    "voice": {"suffix": "-TPU", "label": "Voice"},
}
CHIPS = ("v6e", "v5e")
CHIP_STATUS = ("pending", "passing", "failing", "not supported")
UNCONFIRMED = {"", "unconfirmed", "todo", "tbd", "unknown"}
MARK_START, MARK_END = "<!-- models:start -->", "<!-- models:end -->"
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BIT_LABEL_RE = re.compile(r"\d+[-_ ]?bit\b", re.I)  # our files are named by bpw, never "N-bit"
DEFAULT_TEMPLATES = Path(__file__).resolve().parent.parent / "org" / "templates" / "hf"


class Problems(list):
    def add(self, where, msg):
        self.append(f"{where}: {msg}")


# ---------------------------------------------------------------- validation

def validate(reg, repo_root, problems, release=False):
    for key in ("schema", "github", "hf", "version", "license", "models"):
        if key not in reg:
            problems.add("models.json", f"missing key {key!r}")
    if problems:
        return
    if reg["schema"] != 1:
        problems.add("models.json", f"schema {reg['schema']!r} is not 1")
    if not SEMVER_RE.match(str(reg["version"])):
        problems.add("models.json", f"version {reg['version']!r} is not X.Y.Z")
    hf_org = reg["hf"].get("org", "")
    seen = set()
    for i, m in enumerate(reg["models"]):
        where = f"models[{i}] {m.get('id', '?')}"
        for key in ("id", "name", "kind", "version", "summary", "in", "out", "path", "run",
                    "upstream", "hf_repo", "hf_revision", "pipeline_tag", "chips", "results", "files"):
            if key not in m:
                problems.add(where, f"missing key {key!r}")
        if any(p.startswith(where) and "missing key" in p for p in problems):
            continue
        if m.get("launch") not in (None, "coming soon"):
            problems.add(where, "launch must be \"coming soon\" or absent")
        if not ID_RE.match(m["id"]):
            problems.add(where, "id must be lower case words joined by hyphens")
        if m["id"] in seen:
            problems.add(where, "duplicate id")
        seen.add(m["id"])
        if m["kind"] not in KINDS:
            problems.add(where, f"kind {m['kind']!r} not in {sorted(KINDS)}")
            continue
        if not SEMVER_RE.match(str(m["version"])):
            problems.add(where, "version is not X.Y.Z")
        if repo_root is not None and not (repo_root / m["path"]).exists():
            problems.add(where, f"path {m['path']} does not exist in the repo")
        up = m["upstream"]
        for key in ("name", "maker", "hf", "license"):
            if not str(up.get(key, "")).strip():
                problems.add(where, f"upstream.{key} is empty")
        if str(up.get("license", "")).strip().lower() in UNCONFIRMED:
            problems.add(where, "upstream licence not confirmed (copy it as stated upstream)")
        if up.get("license") == "other" and not (up.get("license_name") and up.get("license_link")):
            problems.add(where, "licence 'other' needs license_name and license_link")
        if m["hf_repo"] is not None:
            want = f"{hf_org}/"
            if not m["hf_repo"].startswith(want):
                problems.add(where, f"hf_repo must be in the {hf_org} org")
            if not m["hf_repo"].endswith(KINDS[m["kind"]]["suffix"]):
                problems.add(where, f"hf_repo for kind {m['kind']} must end with {KINDS[m['kind']]['suffix']}")
            if m["hf_revision"] is not None and not SHA_RE.match(m["hf_revision"]):
                problems.add(where, "hf_revision must be a 40-character commit sha or null")
            if not m["files"] and not m.get("kernels"):
                problems.add(where, "an HF repo holds at least the full-precision weights or the kernel bundle")
        if m["kind"] == "llm-gguf" and m["hf_repo"] is None:
            problems.add(where, "a GGUF entry needs its HF repo")
        for chip in CHIPS:
            c = m["chips"].get(chip)
            if not c or c.get("status") not in CHIP_STATUS:
                if m["kind"] != "llm-gguf":
                    problems.add(where, f"chips.{chip}.status must be one of {CHIP_STATUS}")
                continue
            if c["status"] in ("passing", "failing") and not DATE_RE.match(str(c.get("date", ""))):
                problems.add(where, f"chips.{chip} {c['status']} needs a date YYYY-MM-DD")
            if c["status"] == "not supported" and not c.get("reason"):
                problems.add(where, f"chips.{chip} 'not supported' needs a reason")
        for j, r in enumerate(m["results"]):
            for key in ("chip", "metric", "value", "date", "baseline"):
                if not str(r.get(key, "")).strip():
                    problems.add(where, f"results[{j}].{key} is empty")
            if r.get("date") and not DATE_RE.match(str(r["date"])):
                problems.add(where, f"results[{j}].date is not YYYY-MM-DD")
        for j, r in enumerate(m["results"]):
            sh = r.get("short")
            if sh is None:
                continue
            for key in ("measure", "ours", "vs"):
                if not str(sh.get(key, "")).strip():
                    problems.add(where, f"results[{j}].short.{key} is empty")
            text = " ".join([sh.get("measure", ""), sh.get("ours", ""), sh.get("vs", "")] + sh.get("notes", []) + sh.get("vs_notes", []))
            bad = unbacked(text, result_sources(r))
            if bad:
                problems.add(where, f"results[{j}].short has numbers not in that result: {bad}")
        hl = m.get("headline") or []
        if len(hl) > 4:
            problems.add(where, "headline: at most 4 numbers")
        for j, h in enumerate(hl):
            idx = h.get("results") or []
            if not idx or any(not isinstance(i, int) or not 0 <= i < len(m["results"]) for i in idx):
                problems.add(where, f"headline[{j}].results must name results by index")
                continue
            text = f"{h.get('big', '')} {h.get('label', '')}"
            if not h.get("big") or not h.get("label"):
                problems.add(where, f"headline[{j}] needs big and label")
            if RATIO_RE.search(text) or BIT_LABEL_RE.search(text):
                problems.add(where, f"headline[{j}]: show both numbers, no ratios or N-bit labels")
            bad = unbacked(text, [x for i in idx for x in result_sources(m["results"][i])])
            if bad:
                problems.add(where, f"headline[{j}] has numbers not in its results: {bad}")
        if m.get("headline_note"):
            srcs = [x for r in m["results"] for x in result_sources(r)]
            srcs += [f"{f.get('name', '')} {f.get('gb', '')} {f.get('bpw', '')}" for f in m["files"]] + [m["upstream"]["hf"]]
            bad = unbacked(m["headline_note"], srcs)
            if bad:
                problems.add(where, f"headline_note has numbers not in results or files: {bad}")
        for j, f in enumerate(m["files"]):
            if not f.get("name") or not f.get("format") or not isinstance(f.get("gb"), (int, float)):
                problems.add(where, f"files[{j}] needs name, format and gb (number)")
            if BIT_LABEL_RE.search(str(f.get("name", ""))):
                problems.add(where, f"files[{j}] name says N-bit; name our files by bpw")
        kern = m.get("kernels", [])
        if kern and m["hf_repo"] is None:
            problems.add(where, "kernel bundles live in the model's HF repo; hf_repo is null")
        for j, k in enumerate(kern):
            for key in ("file", "sha256", "jax", "jaxlib", "libtpu", "chips", "form"):
                if not k.get(key):
                    problems.add(where, f"kernels[{j}].{key} is empty")
            if k.get("sha256") and not re.match(r"^[0-9a-f]{64}$", k["sha256"]):
                problems.add(where, f"kernels[{j}].sha256 is not 64 hex characters")
            if k.get("form") and k["form"] not in ("exported", "compiled"):
                problems.add(where, f"kernels[{j}].form must be exported or compiled")
            if any(c not in CHIPS for c in k.get("chips", [])):
                problems.add(where, f"kernels[{j}].chips must be from {CHIPS}")
            # one-off rule: a bundle belongs to exactly this model
            if k.get("file") and not k["file"].startswith(f"kernels/mashbox_{m['id'].replace('-', '_')}_kernels-"):
                problems.add(where, f"kernels[{j}].file must be kernels/mashbox_{m['id'].replace('-', '_')}_kernels-<version>-...whl")
        pj = m.get("plain_jax", {"status": "tested"})
        if pj.get("status") not in ("tested", "not supported") or (pj.get("status") == "not supported" and not pj.get("reason")):
            problems.add(where, "plain_jax.status must be tested, or not supported with a reason")
        if release:  # applies to EVERY model in a release (Codex F08)
            lic = str(reg["license"]).lower()
            if "proposed" in lic or "not chosen" in lic or not lic.strip():
                problems.add(where, "release: our licence is not chosen")
            if not m["results"]:
                problems.add(where, "release: no measured result")
            if m["hf_repo"] is None:
                problems.add(where, "release: every release has an HF repo (HF presence rule)")
            elif m["hf_revision"] is None:
                problems.add(where, "release: hf_revision not pinned")
            if m["kind"] != "llm-gguf":
                sts = [(m["chips"].get(c) or {}).get("status") for c in CHIPS]
                if "passing" not in sts:
                    problems.add(where, "release: no chip passing")
                if any(x in ("pending", "failing", None) for x in sts):
                    problems.add(where, "release: every chip must be passing or not supported (with reason)")
            for j, f in enumerate(m["files"]):
                if f.get("variant") and not (f.get("quality") and f.get("speed") and f.get("date")):
                    problems.add(where, f"release: variant files[{j}] needs measured quality, speed and date")


# ---------------------------------------------------------------- number guard (headline and short cells)

NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)")
RATIO_RE = re.compile(r"\d(?:\.\d+)?\s*x\b")


def _nums(s):
    return [x.rstrip(",").replace(",", "") for x in NUM_RE.findall(str(s))]


def result_sources(r):
    return [str(r.get(k, "")) for k in ("chip", "metric", "value", "baseline", "date")]


def unbacked(text, sources):
    """Numbers in text that are not in sources (a source number rounded to the decimals text shows counts)."""
    src = [Decimal(x) for s in sources for x in _nums(s)]
    bad = []
    for n in _nums(text):
        d = Decimal(n)
        q = Decimal(1).scaleb(d.as_tuple().exponent)
        if not any(x.quantize(q, rounding=ROUND_HALF_UP) == d for x in src):
            bad.append(n)
    return bad


# ---------------------------------------------------------------- rendering

def gh_base(reg):
    return f"https://github.com/{reg['github']['org']}/{reg['github']['repo']}"


def code_url(reg, m):
    return f"{gh_base(reg)}/tree/v{reg['version']}/{m['path']}"


def docs_url(m):
    return f"https://mashbox.gg/docs/models/{m['id']}"


def chip_cell(m, chip):
    c = m["chips"].get(chip) or {}
    st = c.get("status", "")
    if st in ("passing", "failing"):
        return f"{st}, {c['date']}"
    return st or "n/a"


def weights_cell(m):
    if m["hf_repo"]:
        return f"[{m['hf_repo'].split('/', 1)[1]}](https://huggingface.co/{m['hf_repo']})"
    return f"[original](https://huggingface.co/{m['upstream']['hf']})"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def live(reg):
    """Models shown with links in the views; a model with "launch": "coming soon" is named once, without links."""
    return [m for m in reg["models"] if not m.get("launch")]


def soon_line(reg):
    soon = [m for m in reg["models"] if m.get("launch") == "coming soon"]
    if not soon:
        return ""
    return "\n\nComing soon: " + "; ".join(
        f"**{m['name']}** by {m['upstream']['maker']} ({KINDS[m['kind']]['label'].lower()}: {m['in']} in, {m['out']} out)"
        for m in soon) + "."


def models_table(reg, absolute):
    rows = []
    for m in live(reg):
        link = code_url(reg, m) if absolute else m["path"]
        rows.append([f"**[{m['name']}]({link})**<br>by {m['upstream']['maker']}",
                     f"{KINDS[m['kind']]['label']}: {m['in']} in, {m['out']} out",
                     f"TPU v6e: {chip_cell(m, 'v6e')}<br>TPU v5e: {chip_cell(m, 'v5e')}", readme_measured(m), weights_cell(m)])
    return table(["Model", "What it does", "Chips", "Measured", "Weights"], rows) + soon_line(reg)


def short_metric(r):
    return r["metric"].split(" (", 1)[0]


def measured_cell(m, n=2):
    """Headline results for the README table: the first n results of the model, chip named."""
    if not m["results"]:
        return "chip tests running"
    return "<br>".join(f"{short_metric(r)}: **{r['value'].split(' (', 1)[0]}** ({r['chip']})" for r in m["results"][:n])


def readme_measured(m):
    """README cell: the headline numbers marked readme, on one line, plus a link to the card."""
    hl = [h for h in m.get("headline") or [] if h.get("readme")]
    if not hl or not m["hf_repo"]:
        return measured_cell(m)
    r = m["results"][hl[0]["results"][0]]
    return ("; ".join(f"**{h['big']}** {h['label']}" for h in hl) + f" ({r['chip']}, {r['date']}). "
            f"[All results](https://huggingface.co/{m['hf_repo']})")


def profile_table(reg):
    """Compact view for the org profile (reads on a phone): model and maker, in and out, chips passed."""
    rows = []
    for m in live(reg):
        passed = [f"TPU {c} ({m['chips'][c]['date']})" for c in CHIPS
                  if (m["chips"].get(c) or {}).get("status") == "passing"]
        rows.append([f"**[{m['name']}]({code_url(reg, m)})** by {m['upstream']['maker']}",
                     f"{m['in']} in, {m['out']} out", measured_cell(m, n=1), ", ".join(passed) or "pending"])
    return table(["Model", "What it does", "Measured", "Chip tests"], rows) + soon_line(reg)


def llms_block(reg):
    return "\n".join([f"- [{m['name']}]({docs_url(m)}): {m['in']} in, {m['out']} out. {KINDS[m['kind']]['label']}."
                      for m in live(reg)] + [f"- {m['name']}: coming soon. {KINDS[m['kind']]['label']}."
                                             for m in reg["models"] if m.get("launch")])


def front_matter(reg, m):
    up = m["upstream"]
    lines = [f"base_model: {up['hf']}"]
    if m.get("base_model_relation"):
        lines.append(f"base_model_relation: {m['base_model_relation']}")
    lines.append(f"license: {up['license']}")
    if up["license"] == "other":
        lines += [f"license_name: {up['license_name']}", f"license_link: {up['license_link']}"]
    if m.get("pipeline_tag"):
        lines.append(f"pipeline_tag: {m['pipeline_tag']}")
    tags = ["tpu", "jax", "mashbox"] if m["kind"] != "llm-gguf" else ["gguf", "llama.cpp", "mashbox"]
    lines.append("tags:")
    lines += [f"- {t}" for t in tags]
    return "\n".join(lines)


def results_table(m):
    if not m["results"]:
        return "No measured results yet."
    if all(r.get("short") for r in m["results"]):
        notes, rows = [], []

        def mark(ns):
            for n in ns:
                if n not in notes:
                    notes.append(n)
            return f"<sup>{','.join(str(notes.index(n) + 1) for n in ns)}</sup>" if ns else ""
        same = len({(r["chip"], r["date"]) for r in m["results"]}) == 1  # one chip and date: say it once (phone width)
        for r in m["results"]:
            sh = r["short"]
            rows.append([sh["measure"] + mark(sh.get("notes", [])), f"**{sh['ours']}**",
                         sh["vs"] + mark(sh.get("vs_notes", []))] + ([] if same else [f"{r['chip']}, {r['date']}"]))
        foot = "\n".join(f"<sub><sup>{i}</sup> {n}</sub><br>" for i, n in enumerate(notes, 1))
        r0 = m["results"][0]
        head = f"All {len(rows)} measurements" + (f", all on {r0['chip']}, {r0['date']}" if same else "")
        return (f"<details>\n<summary>{head}</summary>\n\n"
                + table(["Measure", "Ours", "Compared with"] + ([] if same else ["Chip, date"]), rows)
                + f"\n\n{foot}\n\n</details>")
    return table(["Chip", "Measure", "Result", "Baseline", "Date"],
                 [[r["chip"], r["metric"], r["value"], r["baseline"], r["date"]] for r in m["results"]])


def files_table(m):
    if not m["files"]:
        return "This port loads the original weights unchanged from " \
               f"[{m['upstream']['hf']}](https://huggingface.co/{m['upstream']['hf']})."
    rows = []
    for f in m["files"]:
        rows.append([f"`{f['name']}`", f["format"], f"{f['gb']:.2f}",
                     f"{f['bpw']:.2f}" if isinstance(f.get("bpw"), (int, float)) else "n/a",
                     f.get("quality") or ("reference" if not f.get("variant") else "not measured"),
                     f.get("speed") or "not measured", f.get("date") or ""])
    return table(["File", "Format", "Size (GB)", "Bits per weight", "Quality vs full precision", "Speed", "Measured"], rows)


def chips_table(m):
    rows = []
    for chip in CHIPS:
        c = m["chips"].get(chip) or {}
        note = c.get("reason") or c.get("note") or ""
        rows.append([f"TPU {chip}", chip_cell(m, chip), note])
    return table(["Chip", "Status", "Note"], rows)


def kernels_section(reg, m):
    kern = m.get("kernels", [])
    pj = m.get("plain_jax", {"status": "tested"})
    if not kern:
        return "This model runs on plain JAX. It ships no compiled kernels."
    tail = (" Without a match, the model runs on plain JAX, slower." if pj.get("status") == "tested"
            else f" This model needs its bundle: {pj.get('reason')}.")
    rows = [[f"`{k['file']}`", ", ".join(f"TPU {c}" for c in k["chips"]), k["jax"], k["jaxlib"], k["libtpu"]]
            for k in kern]
    return ("Compiled TPU kernels for this model only. `mashbox run` picks the file that matches your chip "
            "and JAX version and checks its sha256 against `models.json`." + tail + "\n\n" + table(["File", "Chips", "jax", "jaxlib", "libtpu"], rows))


def license_text(reg, m):
    up = m["upstream"]
    if up["license"] == "other":
        return f"[{up['license_name']}]({up['license_link']})"
    return up["license"]


def highlights(m):
    if not m["results"]:
        return "- Measured results are added here when the chip tests finish."
    out = [f"- **{r['value']}**: {r['metric'][0].lower() + r['metric'][1:]}, on {r['chip']} (measured {r['date']})"
           for r in m["results"]]
    for f in m["files"]:
        if f.get("quality") and f.get("default"):
            out.append(f"- **{f['quant_type'] if f.get('quant_type') else f['name']}** ({f['gb']:.2f} GB, "
                       f"{f['bpw']:.2f} bits per weight): {f['quality']}")
    return "\n".join(out)


def headline(m):
    """At most 4 headline numbers as a 2 x 2 strip (reads on a phone), chip and date under each; else the bullets."""
    hl = m.get("headline")
    if not hl:
        return highlights(m)
    cells = []
    for h in hl:
        r = m["results"][h["results"][0]]
        cells.append(f'<td width="50%"><b>{h["big"]}</b><br>{h["label"]}<br><sub>{r["chip"]}, {r["date"]}</sub></td>')
    rows = ["<tr>\n" + "\n".join(cells[i:i + 2]) + "\n</tr>" for i in range(0, len(cells), 2)]
    out = "<table>\n" + "\n".join(rows) + "\n</table>"
    return out + (f"\n\n<sub>{m['headline_note']}</sub>" if m.get("headline_note") else "")


def top_chart(repo, m):
    """Optional single chart under the headline: hf/<id>/assets/top/NN-<slug>-{light,dark}.png, drawn from the
    measurement ledger; never hand-made."""
    if repo is None or not m["hf_repo"]:
        return ""
    d = Path(repo) / "hf" / m["id"] / "assets" / "top"
    lights = sorted(d.glob("*-light.png")) if d.is_dir() else []
    if not lights:
        return ""
    light = lights[0]
    stem = light.name[:-len("-light.png")]
    alt = (d / f"{stem}.txt").read_text().strip() if (d / f"{stem}.txt").is_file() else stem
    base = f"https://huggingface.co/{m['hf_repo']}/resolve/main/assets/top/"
    dark = d / f"{stem}-dark.png"
    img = f'<img alt="{alt}" src="{base}{light.name}" width="100%">'
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base}{dark.name}">{img}</picture>'
            if dark.is_file() else img)


def hero(repo, m):
    """Optional looping preview: hf/<id>/assets/hero.gif plus a one-line caption in hero.txt."""
    if repo is None or not m["hf_repo"]:
        return ""
    gif = Path(repo) / "hf" / m["id"] / "assets" / "hero.gif"
    if not gif.is_file():
        return ""
    cap = (gif.parent / "hero.txt").read_text().strip() if (gif.parent / "hero.txt").is_file() else ""
    return (f'<img alt="{m["name"]} running on Google Cloud TPU" width="720" '
            f'src="https://huggingface.co/{m["hf_repo"]}/resolve/main/assets/hero.gif">\n\n' + (f"<sub>{cap}</sub>" if cap else ""))


def results_charts(repo, m):
    """Optional result charts: hf/<id>/assets/charts/NN-<slug>-light.png (+ -dark.png) with alt text in NN-<slug>.txt.
    Drawn from the measurement ledger; never hand-made."""
    if repo is None or not m["hf_repo"]:
        return ""
    d = Path(repo) / "hf" / m["id"] / "assets" / "charts"
    if not d.is_dir():
        return ""
    base = f"https://huggingface.co/{m['hf_repo']}/resolve/main/assets/charts/"
    out = []
    for light in sorted(d.glob("*-light.png")):
        stem = light.name[:-len("-light.png")]
        alt = (d / f"{stem}.txt").read_text().strip() if (d / f"{stem}.txt").is_file() else stem
        dark = d / f"{stem}-dark.png"
        src = f'<img alt="{alt}" src="{base}{light.name}" width="100%">'
        out.append(f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base}{dark.name}">{src}</picture>'
                   if dark.is_file() else src)
    if not out:
        return ""
    return "Each chart names its measurement date (UTC) and the hardware on both sides.\n\n" + "\n\n".join(out)


def badges(reg, m):
    gh = gh_base(reg)
    b = [(gh, f"https://img.shields.io/badge/GitHub-{reg['github']['org'].replace('-', '--')}%2F{reg['github']['repo'].replace('-', '--')}-181717?logo=github", "GitHub"),
         (docs_url(m), "https://img.shields.io/badge/Docs-mashbox.gg-ff5a1f", "Docs"),
         (f"{gh}/blob/v{reg['version']}/AGENTS.md", "https://img.shields.io/badge/For%20coding%20agents-AGENTS.md-555555", "AGENTS.md"),
         ("https://x.com/mashboxgg", "https://img.shields.io/badge/X-%40mashboxgg-000000?logo=x", "X")]
    return "<p>\n" + "\n".join(f'  <a href="{u}"><img alt="{a}" src="{i}"></a>' for u, i, a in b) + "\n</p>"


def links_table(reg, m):
    up = m["upstream"]
    rows = [["Code", f"[{reg['github']['org']}/{reg['github']['repo']}]({code_url(reg, m)})"],
            ["Guide", f"[{docs_url(m).replace('https://', '')}]({docs_url(m)})"],
            ["Original model", f"[{up['hf']}](https://huggingface.co/{up['hf']}) by {up['maker']}"],
            ["Setup from a new Google account", "[mashbox.gg/docs/google-cloud](https://mashbox.gg/docs/google-cloud)"],
            ["Companies", "hello@mashbox.gg"]]
    return table(["", ""], rows)


def agents_section(m):
    """LLM cards: one short section saying coding agents can use the model, from the model's cloud.agents row."""
    a = (m.get("cloud") or {}).get("agents")
    if not a:
        return ""
    run = " ".join([m["run"]] + [str(x) for x in a.get("run_args", [])])
    return ("### Works with Claude Code, Codex and Pi\n\n"
            "The `mashbox` server speaks the Anthropic Messages, OpenAI Responses and OpenAI chat completions APIs, "
            f"so coding agents can use {m['name']} as their LLM. Start it with `{run}` and point the agent at it: "
            f"[Use with coding agents]({a['guide']}).")


def card_vars(reg, m, repo=None):
    up = m["upstream"]
    return {
        "banner": (f"https://huggingface.co/{m['hf_repo']}/resolve/main/assets/banner-dark.png" if m["hf_repo"] else ""),
        "badges": badges(reg, m), "highlights": highlights(m), "headline": headline(m), "top_chart": top_chart(repo, m),
        "hero": hero(repo, m), "links_table": links_table(reg, m),
        "front_matter": front_matter(reg, m),
        "name": m["name"], "id": m["id"], "summary": m["summary"], "in": m["in"], "out": m["out"],
        "run": m["run"], "maker": up["maker"], "upstream_hf": up["hf"],
        "upstream_url": f"https://huggingface.co/{up['hf']}",
        "upstream_github": f"https://github.com/{up['github']}" if up.get("github") else f"https://huggingface.co/{up['hf']}",
        "hf_repo": m["hf_repo"] or "", "version": reg["version"], "code_url": code_url(reg, m),
        "docs_url": docs_url(m), "agents_url": f"{gh_base(reg)}/blob/v{reg['version']}/AGENTS.md",
        "results_table": results_table(m), "results_charts": results_charts(repo, m), "files_table": files_table(m), "chips_table": chips_table(m),
        "license_text": license_text(reg, m), "our_license": reg["license"],
        "notes": m.get("notes", ""), "agents_section": agents_section(m),
        "kernels_section": kernels_section(reg, m),
        "kernels_license": (f"The compiled kernels in `kernels/` are under their own licence: "
                            f"[LICENSE-KERNELS]({gh_base(reg)}/blob/v{reg['version']}/LICENSE-KERNELS).")
                           if m.get("kernels") else "",
        "default_tag": (m.get("default_tag") or ""),
    }


def render_card(reg, m, templates, repo=None):
    tpl = (templates / f"card-{m['kind']}.md").read_text()
    text = Template(tpl).substitute(card_vars(reg, m, repo))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip() + "\n"


def splice(text, block, where, problems):
    if MARK_START not in text or MARK_END not in text:
        problems.add(where, f"markers {MARK_START} / {MARK_END} not found")
        return None
    a = text.index(MARK_START) + len(MARK_START)
    b = text.index(MARK_END)
    return text[:a] + "\n" + block + "\n" + text[b:]


def render_all(reg, repo, profile, templates, problems):
    """Return {relative path or 'profile': rendered text} for every view."""
    views = {}
    readme = repo / "README.md"
    if readme.exists():
        out = splice(readme.read_text(), models_table(reg, absolute=False), "README.md", problems)
        if out is not None:
            views["README.md"] = out
    else:
        problems.add("README.md", "missing")
    llms = repo / "llms.txt"
    if llms.exists():
        out = splice(llms.read_text(), llms_block(reg), "llms.txt", problems)
        if out is not None:
            views["llms.txt"] = out
    else:
        problems.add("llms.txt", "missing")
    if profile is not None:
        out = splice(profile.read_text(), profile_table(reg), str(profile), problems)
        if out is not None:
            views["profile"] = out
    for m in live(reg):  # a "coming soon" model's card is not published yet; it is checked again at its launch
        if m["hf_repo"]:
            views[f"hf/{m['id']}/README.md"] = render_card(reg, m, templates, repo)
    return views


# ---------------------------------------------------------------- online

class Fetcher:
    def __init__(self, fixtures=None):
        self.fixtures = Path(fixtures) if fixtures else None

    def get(self, url):
        """Return (status, body_text)."""
        if self.fixtures is not None:
            f = self.fixtures / (hashlib.sha1(url.encode()).hexdigest() + ".json")
            if not f.exists():
                return 404, ""
            rec = json.loads(f.read_text())
            return rec["status"], rec["body"]
        headers = {"User-Agent": "mashbox-sync-check"}
        tok = os.environ.get("HF_TOKEN") if "huggingface.co" in url else os.environ.get("GH_TOKEN")
        if tok:
            headers["Authorization"] = "Bearer " + tok
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, ""
        except Exception as e:  # network down counts as a failure, never as a pass
            return 0, str(e)


def check_online(reg, views, fetch, problems):
    hf_org = reg["hf"]["org"]
    st, body = fetch.get(f"https://huggingface.co/api/models?author={hf_org}&limit=1000")
    listed = set()
    if st != 200:
        problems.add("hf", f"cannot list models of {hf_org} (HTTP {st})")
    else:
        listed = {x.get("id") or x.get("modelId") for x in json.loads(body)}
    wanted = {m["hf_repo"] for m in reg["models"] if m["hf_repo"]}
    for extra in sorted(listed - wanted):
        problems.add("hf", f"{extra} is on Hugging Face but not in models.json")
    for m in reg["models"]:
        if not m["hf_repo"]:
            continue
        where = f"hf {m['hf_repo']}"
        st, body = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}")
        if st != 200:
            problems.add(where, f"repo not found (HTTP {st})")
            continue
        info = json.loads(body)
        if info.get("private") or info.get("gated"):
            problems.add(where, "repo is private or gated")
        rev = m["hf_revision"]
        if not rev:
            problems.add(where, "hf_revision not pinned in models.json")
            continue
        if info.get("sha") != rev:  # the visible head must be the pinned, rendered revision (Codex F09)
            problems.add(where, f"HF head {str(info.get('sha'))[:12]} is not the pinned revision {rev[:12]}")
        st, _ = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}/revision/{rev}")
        if st != 200:
            problems.add(where, f"pinned revision {rev[:12]} not found (HTTP {st})")
            continue
        st, card = fetch.get(f"https://huggingface.co/{m['hf_repo']}/raw/{rev}/README.md")
        want = views.get(f"hf/{m['id']}/README.md")
        if st != 200:
            problems.add(where, f"card not readable at {rev[:12]} (HTTP {st})")
        elif want is not None and card.strip() != want.strip():
            problems.add(where, "card on Hugging Face differs from the card rendered from models.json")
    gh = reg["github"]
    st, _ = fetch.get(f"https://api.github.com/repos/{gh['org']}/{gh['repo']}/git/ref/tags/v{reg['version']}")
    if st != 200:
        problems.add("github", f"tag v{reg['version']} not found in {gh['org']}/{gh['repo']} (HTTP {st})")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["render", "check"])
    ap.add_argument("--repo", required=True, help="root of the main repo (holds models.json)")
    ap.add_argument("--profile", help="org profile README to keep in sync (optional)")
    ap.add_argument("--org-defaults", help="the .github repo: CONTRIBUTING.md and SECURITY.md must equal the main repo's")
    ap.add_argument("--templates", help="card templates; default: <repo>/templates/hf when it exists, else the org templates")
    ap.add_argument("--write", action="store_true", help="render: write the views in place")
    ap.add_argument("--out", help="render: write the views under this directory instead")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--online", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--fixtures")
    a = ap.parse_args(argv)

    repo = Path(a.repo)
    templates = Path(a.templates) if a.templates else \
        (repo / "templates" / "hf" if (repo / "templates" / "hf").is_dir() else DEFAULT_TEMPLATES)
    profile = Path(a.profile) if a.profile else None
    problems = Problems()
    try:
        reg = json.loads((repo / "models.json").read_text())
    except (OSError, ValueError) as e:
        print(f"FAIL models.json: {e}")
        return 1
    validate(reg, repo, problems, release=a.release)
    if problems:
        for p in problems:
            print("FAIL", p)
        return 1
    views = render_all(reg, repo, profile, templates, problems)

    if a.cmd == "render":
        for key, text in views.items():
            if a.out:
                dest = Path(a.out) / ("profile/README.md" if key == "profile" else key)
            elif a.write:
                dest = profile if key == "profile" else repo / key
            else:
                print(f"----- {key}\n{text}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            print("wrote", dest)
        for p in problems:
            print("FAIL", p)
        return 1 if problems else 0

    for key, text in views.items():
        path = profile if key == "profile" else repo / key
        if not path.exists():
            problems.add(key, "rendered view missing on disk (run render --write)")
        elif path.read_text() != text:
            problems.add(key, "differs from the view rendered from models.json (run render --write)")
    if a.org_defaults:
        for name in ("CONTRIBUTING.md", "SECURITY.md"):
            mine, theirs = repo / name, Path(a.org_defaults) / name
            if not theirs.exists() or not mine.exists() or mine.read_bytes() != theirs.read_bytes():
                problems.add(name, "org default copy differs from the main repo's (copy it at release)")
    if a.online:
        check_online(reg, views, Fetcher(a.fixtures), problems)
    for p in problems:
        print("FAIL", p)
    n = sum(1 for m in reg["models"])
    print(f"{'OK' if not problems else 'OUT OF SYNC'}: {n} models, {len(views)} views, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
