#!/usr/bin/env python3
"""Keep GitHub and Hugging Face in sync from one file: models.json.

models.json (in the main repo) is the only source. This script renders every view
from it and checks that the views on disk, on Hugging Face and on GitHub match.

Views:
  README.md            the block between <!-- models:start --> and <!-- models:end -->
  profile README       checked, never rendered (optional, --profile PATH; --org-card PATH for the Hugging Face org card):
                       an overview with no models block, no model named and no count; a family line names exactly
                       the GGUF families models.json has
  llms.txt             the block between <!-- models:start --> and <!-- models:end -->
  hf/<id>/README.md    the Hugging Face card of every model that has an HF repo
                       (rendered from templates/hf/card-<kind>.md in the repo)
  docs/openapi.json    the server's OpenAPI 3.1 description, from src/mashbox/runtime/openapi.py
  docs/server-api.md   the Server API page, from the same module (one section per route)
  docs/cli.md          every command and option, from the package's argparse parsers
  docs pages           a block between <!-- catalog:start --> and <!-- catalog:end -->: the catalog (mashbox/catalog.py,
                       the same render as `mashbox models`): by publisher, then model, then file
  docs/gguf.md         the block between <!-- archs:start --> and <!-- archs:end -->: catalog.arch_rows (ARCHS with a passing chip row)
  docs pages           blocks between <!-- cli:start CMD --> / <!-- cli:end --> (option table of `mashbox CMD`)
                       and <!-- api:start --> / <!-- api:end --> (routes and sampling fields) in docs/*.md, docs/models/*.md
  versions             models.json is the one source: src/mashbox/__init__.py reads it, pyproject.toml takes it
                       from mashbox.__version__ (dynamic), and no other file writes a version
  licences             LICENSE, pyproject [project].license, licenses/<id>/LICENSE sha256 and every page's licence
                       wording agree with models.json (report only: fails on a mismatch, never edits)
  cloud prices         blocks between <!-- cloud:start NAME --> / <!-- cloud:end --> in docs pages, from
                       docs/data/cloud-prices.json (dated Google Cloud list prices, memory bandwidth) and models.json;
                       fails when a price was read more than max_age_days (45) ago or models.json prices differ
  numbers in prose     <!-- r:MODEL/N -->TEXT<!-- /r --> in README.md, llms.txt and docs pages: TEXT is result N of
                       MODEL in models.json (short.ours, or its part before " ("); r:MODEL/NAME a figure, r:MODEL/chip-v5e
                       a chips row, r:MODEL/arg-ctx a run_args value; render fills it (pages and the app's HTML). Every
                       other measured number on every public page (git ls-files *.md, tests/ and hf/ too, but the
                       rendered cards), in the UI text of the package (src/mashbox and apps/ .html/.js) and in the
                       pyproject description fails T0 unless tools/numbers_allowed.txt names it.
                       The pages carry no markers: render --write moves them, and the named figures, into the number
                       map internal/numbers.json (maintainers only), which records where each number sits; the checks
                       put the markers back from it before they read a page. Without the map the number guard is
                       skipped.

Usage:
  hf_github_sync_check.py render --repo ROOT [--profile PATH] [--templates DIR] [--write] [--out DIR]
  hf_github_sync_check.py markup --repo ROOT PAGE...   (number markers back in these pages, to edit the words around them)
  hf_github_sync_check.py licence --repo ROOT   (the licence lines for a person, then the licence check; exit 1 on a mismatch)
  hf_github_sync_check.py check  --repo ROOT [--profile PATH] [--templates DIR] [--offline|--online]
                                  [--release] [--fixtures DIR]

  --offline  schema + rendered views equal the files on disk (no network). Default.
  --online   also asks the Hugging Face and GitHub APIs: repo public, pinned revision exists,
             card at that revision equals the rendered card, no repo in the HF org missing
             from models.json, the version tag exists on GitHub. A model with "launch" set
             (coming soon) only has to be private or absent: it must not be public yet.
  --release  stricter: every model with an HF repo has a measured result and a passing chip.
  --fixtures replay API answers from DIR (tests). Tokens, if needed, come from the
             environment (HF_TOKEN, GH_TOKEN), never from arguments.

Exit status: 0 in sync, 1 out of sync or invalid, 2 usage error.
"""
import argparse
import datetime
from decimal import Decimal, ROUND_HALF_UP
import functools
import hashlib
import html
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
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
    "rig": {"suffix": "", "label": "Rigging"},   # our own CPU models (Track2Rig): HF repo = the model name
}
CHIPS = ("v5e", "v6e", "v5p-8", "v6e-4")      # the chip types people rent: v5e, v6e, a v5p slice and a v6e slice
CHIP_LABEL = {"v5e": "TPU v5e", "v6e": "TPU v6e", "v5p-8": "TPU v5p slice (v5p-8)", "v6e-4": "TPU v6e slice (v6e-4)"}
CHIP_STATUS = ("pending", "passing", "measured", "failing", "not supported")
# measured: a dated measured run of the model on this chip (a dated results row on it) while this version's `mashbox run`
# does not start there; the pages print its note, llms.txt names the chip with its run_note, and the package's chip
# lists (catalog, app, tpu up), which read "passing" only, never offer it
UNCONFIRMED = {"", "unconfirmed", "todo", "tbd", "unknown"}
MARK_START, MARK_END = "<!-- models:start -->", "<!-- models:end -->"
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DATE_TAG_RE = re.compile(r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]$")   # a dated weights version (own models)
BIT_LABEL_RE = re.compile(r"\d+[-_ ]?bit\b", re.I)  # our files are named by bpw, never "N-bit"
PRIVATE_RE = re.compile(r"private", re.I)   # models.json is public: no private repository's name in it, at any depth
DEFAULT_TEMPLATES = Path(__file__).resolve().parent.parent / "org" / "templates" / "hf"


class Problems(list):
    def add(self, where, msg):
        self.append(f"{where}: {msg}")


# ---------------------------------------------------------------- validation
def private_labels(obj, path="models.json"):
    """Every (where, text) in models.json, keys and values at any depth, that says "private". models.json ships in the
    wheel and the export and every view renders from it, so a private repository's name must never be in it."""
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in private_labels(k, path) + private_labels(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [x for j, v in enumerate(obj) for x in private_labels(v, f"{path}[{j}]")]
    return [(path, obj)] if isinstance(obj, str) and PRIVATE_RE.search(obj) else []



def validate(reg, repo_root, problems, release=False):
    for key in ("schema", "github", "hf", "version", "license", "company_line", "install", "prices", "models"):
        if key not in reg:
            problems.add("models.json", f"missing key {key!r}")
    if problems:
        return
    for where, text in private_labels(reg):
        problems.add(where, f"{text[:80]!r} names a private repository: models.json is public (the wheel, the export and "
                            "every view carry it)")
    if reg["schema"] not in SCHEMAS:
        problems.add("models.json", f"schema {reg['schema']!r} is not one of {SCHEMAS}")
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
        if "stage" in m:
            problems.add(where, "stage is not a field: the Experimental label comes from catalog.experimental")
        if m["kind"] == "rig" and m["hf_revision"] is not None and not DATE_TAG_RE.match(str(m.get("hf_tag"))):
            problems.add(where, "hf_tag must name the dated weights version (YYYY-MM-DD): the Hugging Face tag of a "
                                "revision whose weights file is the pinned one")
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
            if c["status"] in ("passing", "failing", "measured") and not real_date(c.get("date", "")):
                problems.add(where, f"chips.{chip} {c['status']} needs a date YYYY-MM-DD")
            if c["status"] == "failing" and not c.get("reason"):
                problems.add(where, f"chips.{chip} 'failing' needs the measured reason")
            if "not_run_reason" in c and (c["status"] != "pending" or not str(c["not_run_reason"]).strip()):
                problems.add(where, f"chips.{chip}.not_run_reason says why a pending chip has not been run: only on a "
                                    "pending chip, never empty")
            if "run_note" in c and (c["status"] not in ("passing", "measured") or not str(c["run_note"]).strip()):
                problems.add(where, f"chips.{chip}.run_note says how the model runs on the chip (its measured speed and "
                                    "any condition): only on a passing or measured chip, never empty")
            if c["status"] == "not supported" and not (real_date(c.get("date", "")) and str(c.get("reason") or "").strip()):
                problems.add(where, f"chips.{chip}: a 'not supported' row states what a run found there, so it carries "
                                    "the day of the run and its reason; a chip nobody ran is pending with its "
                                    "not_run_reason")
            if c["status"] == "measured":
                if not RUN_NOTE_NUMBER_RE.search(row_words(c.get("run_note"))):
                    problems.add(where, f"chips.{chip} 'measured' run_note states the measured number with its unit")
                if chip not in chip_rows_measured(m):
                    problems.add(where, f"chips.{chip} 'measured' needs a dated results row on its chip (the run it "
                                        "states)")
                if not str(c.get("note") or "").strip() or not str(c.get("run_note") or "").strip():
                    problems.add(where, f"chips.{chip} 'measured' needs its note (the pages print it) and its run_note "
                                        "(llms.txt prints it): what was measured and that this version's `mashbox run` "
                                        "starts elsewhere")
            for bad in chip_row_problems(m, chip):
                problems.add(where, bad)
        for j, r in enumerate(m["results"]):
            for key in ("chip", "metric", "value", "date", "baseline"):
                if not str(r.get(key, "")).strip():
                    problems.add(where, f"results[{j}].{key} is empty")
            if not real_date(r.get("date", "")):
                problems.add(where, f"results[{j}].date is not a real YYYY-MM-DD date (not after tomorrow, UTC)")
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
            if "repo" in f:     # a row may pin its own revision of hf_repo, never another repo (a private one, say)
                problems.add(where, f"files[{j}] {f.get('name')} names its own repo {f['repo']!r}: every file of a model "
                                    "is in its one public repo (hf_repo)")
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
        if release:  # applies to EVERY model in a release
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
                if "passing" not in sts and (m["chips"].get("cpu") or {}).get("status") != "passing":
                    problems.add(where, "release: no chip passing (TPU, or the CPU for a CPU-only model)")
                # a released chip row is passing, measured, not supported, or pending with its not_run_reason; a chip
                # nobody ran is pending with its not_run_reason and reads "Not tested yet" on every page, and a chip
                # with a measured run on it is never "not supported" or pending: chip_row_problems, every mode
                if any(x in ("failing", None) or (x == "pending" and not not_run_reason(m, c)) for c, x in zip(CHIPS, sts)):
                    problems.add(where, "release: every chip must be passing, measured, not supported, or pending with "
                                        "its not_run_reason (a chip nobody ran reads 'Not tested yet')")
            for j, f in enumerate(m["files"]):
                if f.get("variant") and not (f.get("quality") and f.get("speed") and f.get("date")):
                    problems.add(where, f"release: variant files[{j}] needs measured quality, speed and date")
    validate_catalog(reg, problems)


# ---------------------------------------------------------------- catalog (schema 2: catalog blocks, gguf[], engine_kernels[])

SCHEMAS = (1, 2)
CATALOG_SOURCES = ("ours",)
CATALOG_FORMATS = ("bshj-build", "port", "own-model")
KERNELS_FILE_RE = re.compile(r"^mashbox_engine_kernels-[0-9A-Za-z_.+-]+\.whl$")
GGUF_KEYS = ("id", "model", "publisher", "family", "maker", "repo", "revision", "file", "tag", "sha256", "bytes", "arch", "moe", "params_b",
             "types", "license", "license_link", "base", "in", "out", "run", "experimental", "chips")
GGUF_CHIP_KEYS = ("status", "accelerator", "ctx", "slots", "kernel_path", "measured")
# a quick-check chip run (record_kind "quick-check": four zero-flag `mashbox run REPO:TAG --prompt P --max-tokens 64
# --temperature 0` runs compared with llama.cpp) measures decode tok/s at 1 user on the exact path, nothing else
GGUF_QUICK_KEYS = ("status", "accelerator", "kernel_path", "measured", "record_kind")
RECORD_KINDS = ("quick-check",)
QUICK_MEASURED = ("decode_tok_s_1_user",)
UNQUANTIZED = ("F32", "F16", "BF16")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


DATED_RE = re.compile(r"\d{4}-\d{2}-\d{2}")    # a date anywhere in a string (DATE_RE is the whole-field form)


def _dated(obj):
    """True when obj holds a date anywhere: a key that names one, or a string value with YYYY-MM-DD in it"""
    if isinstance(obj, dict):
        return any("date" in str(k).lower() or _dated(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(_dated(v) for v in obj)
    return isinstance(obj, str) and bool(DATED_RE.search(obj))


def _params_ok(p):
    return isinstance(p, dict) and set(p) == {"total", "active"} and all(
        v is None or (isinstance(v, (int, float)) and v > 0) for v in p.values())


def validate_catalog(reg, problems):
    """Schema 2. models[].catalog: source, format, family, params_b {total, active} (billions or null), experimental, number
    (null, {"headline": i} or {"result": i}). gguf[]: one GGUF file confirmed on a chip with this engine, pinned by repo,
    40-hex revision, file and sha256, its licence copied as stated on the repo card, the exact command, and per chip
    the run's outcome (passing or failing), accelerator, ctx, slots, kernel path and measured speed; no row stores a
    support level (mashbox/catalog.py computes it). engine_kernels[]: the engine's one kernels package
    (mashbox_engine_kernels), one row per (chip, JAX). No dates in any of them, as a key or in a value (the records
    behind a row are kept outside this repository)."""
    if reg["schema"] == 1:
        for key in ("gguf", "engine_kernels"):
            if key in reg:
                problems.add("models.json", f"{key} needs schema 2")
        return
    for rid, msg in _lists().maker_problems(reg):
        problems.add(f"models.json gguf {rid}", msg)
    for key in ("gguf", "engine_kernels"):
        if not isinstance(reg.get(key), list):
            problems.add("models.json", f"schema 2 needs {key} (a list, empty when there is none)")
            return
    ids = {m.get("id") for m in reg["models"]}
    for i, m in enumerate(reg["models"]):
        where = f"models[{i}] {m.get('id', '?')}"
        c = m.get("catalog")
        if not isinstance(c, dict):
            problems.add(where, "schema 2: missing the catalog block")
            continue
        if c.get("source") not in CATALOG_SOURCES:
            problems.add(where, f"catalog.source must be one of {CATALOG_SOURCES}")
        if c.get("format") not in CATALOG_FORMATS:
            problems.add(where, f"catalog.format must be one of {CATALOG_FORMATS}")
        if not str(c.get("family", "")).strip():
            problems.add(where, "catalog.family is empty")
        if not _params_ok(c.get("params_b")):
            problems.add(where, "catalog.params_b must be {total, active}, each a number of billions or null")
        if not isinstance(c.get("experimental"), bool):
            problems.add(where, "catalog.experimental must be true or false")
        if "status" in c or "level" in c:
            problems.add(where, "catalog: the support level is computed by mashbox/catalog.py, never stored")
        n = c.get("number")
        if n is not None:
            k = next(iter(n), None) if isinstance(n, dict) and len(n) == 1 else None
            arr = {"headline": m.get("headline") or [], "result": m.get("results") or []}.get(k)
            if arr is None or not isinstance(n[k], int) or not 0 <= n[k] < len(arr):
                problems.add(where, "catalog.number must be null, {\"headline\": i} or {\"result\": i} naming one that exists")
        if _dated(c):
            problems.add(where, "catalog: no dates (the records behind a row are kept outside this repository)")
    kern_types, builds = {}, set()
    for j, k in enumerate(reg["engine_kernels"]):
        where = f"engine_kernels[{j}]"
        for key in ("for", "file", "url", "sha256", "jax", "libtpu", "chips", "types"):
            if not k.get(key):
                problems.add(where, f"{key} is empty")
        if k.get("for") != "gguf":
            problems.add(where, "for must be \"gguf\"")
        if k.get("file") and not KERNELS_FILE_RE.match(k["file"].rsplit("/", 1)[-1]):
            problems.add(where, "file must be a mashbox_engine_kernels-*.whl (the engine's one kernels package)")
        build = (tuple(k.get("chips") or []), k.get("jax"))
        if len(build[0]) != 1 or build in builds:
            problems.add(where, "one row per (chip, JAX): chips names one chip, and no other row has that chip and jax")
        builds.add(build)
        if k.get("url") and not str(k["url"]).startswith("https://"):
            problems.add(where, "url must be https")
        if k.get("sha256") and not HEX64_RE.match(k["sha256"]):
            problems.add(where, "sha256 is not 64 hex characters")
        if any(ch not in CHIPS for ch in k.get("chips") or []):
            problems.add(where, f"chips must be from {CHIPS}")
        if _dated(k):
            problems.add(where, "no dates (the records behind a row are kept outside this repository)")
        for ch in k.get("chips") or []:
            kern_types.setdefault(ch, set()).update(k.get("types") or [])
    names = {}
    for j, g in enumerate(reg["gguf"]):
        where = f"gguf[{j}] {g.get('id', '?')}"
        miss = [key for key in GGUF_KEYS if key not in g]
        if miss:
            problems.add(where, f"missing keys {miss}")
            continue
        if not ID_RE.match(str(g["id"])):
            problems.add(where, "id must be lower case words joined by hyphens")
        if g["id"] in ids:
            problems.add(where, "duplicate id (ids are unique across models[] and gguf[])")
        ids.add(g["id"])
        if not SHA_RE.match(str(g["revision"])):
            problems.add(where, "revision must be the 40-character commit the run used")
        if not HEX64_RE.match(str(g["sha256"])):
            problems.add(where, "sha256 must be the 64-hex sha256 of the file that ran")
        if not isinstance(g["bytes"], int) or g["bytes"] <= 0:
            problems.add(where, "bytes must be the file's size")
        if not str(g["file"]).endswith(".gguf"):
            problems.add(where, "file must be a .gguf path in the repo")
        if "/" not in str(g["repo"]):
            problems.add(where, "repo must be owner/name on Hugging Face")
        if not ID_RE.match(str(g["model"])):
            problems.add(where, "model must be the model's id: lower case words joined by hyphens")
        if g["publisher"] != str(g["repo"]).split("/")[0]:
            problems.add(where, "publisher must be the Hugging Face owner of repo (the views group rows by it)")
        name = str(g["base"] or g["family"]).split("/")[-1]
        if names.setdefault((g["publisher"], g["model"]), name) != name:
            problems.add(where, f"one name per model: {g['publisher']}/{g['model']} is already "
                                f"{names[(g['publisher'], g['model'])]!r}, this row's base or family says {name!r}")
        if not isinstance(g["moe"], bool):
            problems.add(where, "moe must be true or false (from the file's expert count)")
        if not _params_ok(g["params_b"]):
            problems.add(where, "params_b must be {total, active}, each a number of billions or null")
        act, tot = (g["params_b"] or {}).get("active"), (g["params_b"] or {}).get("total")
        moe = g["moe"] is True or (isinstance(act, (int, float)) and act != tot)    # the file's metadata, never a name
        if moe and g["moe"] is not True:
            problems.add(where, f"params_b active {act} is not total {tot}: the file is a mixture of experts, so moe is true")
        if not isinstance(g["types"], list) or not g["types"]:
            problems.add(where, "types must list the block types in the file's header")
        if str(g["license"]).strip().lower() in UNCONFIRMED or not str(g["license_link"]).startswith("https://"):
            problems.add(where, "license and license_link: copy them as stated on the repo card at the revision")
        if not isinstance(g["experimental"], bool):
            problems.add(where, "experimental must be true or false")
        if "status" in g or "level" in g:
            problems.add(where, "the support level is computed by mashbox/catalog.py, never stored")
        want = f"mashbox run {g['repo']}:{g['tag']}"
        if not (g["run"] == want or str(g["run"]).startswith(want + " ")):
            problems.add(where, f"run must start with {want!r} (the command the record ran)")
        if _dated(g):
            problems.add(where, "no dates (the records behind a row are kept outside this repository)")
        chips = g["chips"] if isinstance(g["chips"], dict) else {}
        cpu_ok = "cpu_check" in g and not cpu_check_problems(g["cpu_check"], problems, where)
        pending = [ch for ch, r in chips.items() if isinstance(r, dict) and r.get("status") == "pending"]
        if not chips or any(ch not in CHIPS for ch in chips):
            problems.add(where, f"chips must name at least one chip from {CHIPS}")
        elif not any(isinstance(r, dict) and r.get("status") == "passing" for r in chips.values()):
            # Checked on CPU: no chip run, its CPU check against llama.cpp passed (sealed record), every chip pending
            # with its not_run_reason; never a mixture of experts (no MoE without its routed experts on the kernel)
            if moe:
                problems.add(where, "a mixture of experts is only listed with a passing chip record of its routed experts "
                                    "on the kernel, never from a CPU check alone")
            elif not cpu_ok or (g["cpu_check"] or {}).get("status") != "passing" or len(pending) != len(chips):
                problems.add(where, "no passing chip: a row is only added from a passing run (a failing outcome sits only "
                                    "beside a passing one), or from a passing CPU check (cpu_check status passing) with "
                                    "every chip pending and its not_run_reason")
        for ch in pending:
            # a not-run chip holds status and not_run_reason only: a row shows a speed only when measured on that exact
            # file, so no smoke from another row (same architecture or not) and nothing measured
            r = chips[ch]
            if not str(r.get("not_run_reason") or "").strip() or set(r) - {"status", "not_run_reason"}:
                problems.add(where, f"chips.{ch}: a pending chip holds its status and not_run_reason (never empty) only: "
                                    f"no smoke, nothing measured ({sorted(set(r) - {'status', 'not_run_reason'})})")
        for ch, r in chips.items():
            if ch in pending:
                continue
            quick = isinstance(r, dict) and "record_kind" in r
            keys = GGUF_QUICK_KEYS if quick else GGUF_CHIP_KEYS
            if not isinstance(r, dict) or any(key not in r for key in keys):
                problems.add(where, f"chips.{ch} needs {keys}")
                continue
            if quick:
                if r["record_kind"] not in RECORD_KINDS:
                    problems.add(where, f"chips.{ch}.record_kind must be one of {RECORD_KINDS}")
                if "ctx" in r or "slots" in r:
                    problems.add(where, f"chips.{ch}: a quick-check run records no ctx or slots, so its row has none")
                if r["kernel_path"] != "exact":
                    problems.add(where, f"chips.{ch}: a quick-check run is the zero-flag run: kernel_path exact")
                if sorted(r["measured"] or {}) != list(QUICK_MEASURED):
                    problems.add(where, f"chips.{ch}: a quick-check run measures decode_tok_s_1_user only (1 user, no "
                                        "8-user number)")
            if r["status"] not in ("passing", "failing"):
                problems.add(where, f"chips.{ch}.status must be passing or failing (the outcome of the run's record)")
            if r["kernel_path"] not in ("bundle", "exact"):
                problems.add(where, f"chips.{ch}.kernel_path must be bundle or exact")
            if moe and r["status"] == "passing":      # no mixture of experts without its experts on the kernel
                if r.get("routed_experts") != "kernel":
                    problems.add(where, f"chips.{ch}: a mixture of experts passes only with a sealed chip record of its "
                                        "routed experts on the kernel (routed_experts \"kernel\")")
                if not all(isinstance((r["measured"] or {}).get(f"goodput_roofline_pct_{n}"), (int, float))
                           for n in ("1_user", "8_users")):
                    problems.add(where, f"chips.{ch}: a mixture of experts passes only with its goodput as % of the "
                                        "roofline for 1 and 8 users (measured.goodput_roofline_pct_1_user and "
                                        "goodput_roofline_pct_8_users, from its record)")
            if not moe and "routed_experts" in r:
                problems.add(where, f"chips.{ch}.routed_experts on a row whose file has no experts")
            tok = (r["measured"] or {}).get("decode_tok_s_1_user")
            if not isinstance(tok, (int, float)) or tok <= 0:
                problems.add(where, f"chips.{ch}.measured.decode_tok_s_1_user must be the record's number")
            if r["kernel_path"] == "bundle":
                need = {t for t in g["types"] if t not in UNQUANTIZED}
                lack = sorted(need - kern_types.get(ch, set()))
                if lack:
                    problems.add(where, f"chips.{ch}: kernel_path bundle, but no engine_kernels row for {ch} has {lack}")


CPU_CHECK_KEYS = {"status", "reference", "identical_weights", "block_path", "greedy", "tokenizer"}
TOP1_RE = re.compile(r"^(\d+)/(\d+)$")


def cpu_check_problems(c, problems, where):
    """gguf[].cpu_check: the CPU check of the file against llama.cpp, copied from its sealed gguf_conform record (the
    record guard holds every value to it). Shape only here: status passing or failing, the reference named, same-weights
    NMSE <= 1e-4 with its top-1, llama.cpp's own path top-1 and KL, greedy prompts x tokens with full/near tie/other,
    tokenizer texts/equal/explained; a passing one has 0 other greedy divergences. Returns the number of problems."""
    bad = []
    if not isinstance(c, dict) or set(c) != CPU_CHECK_KEYS:
        bad.append(f"cpu_check must have exactly {sorted(CPU_CHECK_KEYS)}")
    else:
        same, block, g, tk = c["identical_weights"], c["block_path"], c["greedy"], c["tokenizer"]
        if c["status"] not in ("passing", "failing"):
            bad.append("cpu_check.status must be passing or failing (the record's verdict)")
        if not str(c["reference"]).startswith("llama.cpp "):
            bad.append("cpu_check.reference must name the llama.cpp build it was checked against")
        tops = [TOP1_RE.match(str((x or {}).get("top1"))) for x in (same, block)]
        if not all(tops) or any(int(m.group(1)) > int(m.group(2)) for m in tops):
            bad.append("cpu_check top1 must be 'agree/positions'")
        nm = (same or {}).get("nmse")
        if not isinstance(nm, (int, float)) or nm < 0 or not isinstance((block or {}).get("kl"), (int, float)):
            bad.append("cpu_check: identical_weights.nmse and block_path.kl must be numbers")
        if not all(isinstance((g or {}).get(k), int) for k in ("prompts", "tokens", "full", "near_tie", "other")) or \
                not all(isinstance((tk or {}).get(k), int) for k in ("texts", "equal", "explained")):
            bad.append("cpu_check: greedy {prompts, tokens, full, near_tie, other} and tokenizer {texts, equal, "
                       "explained} must be counts")
        elif c["status"] == "passing" and (g["other"] != 0 or g["full"] + g["near_tie"] != g["prompts"]
                                           or tk["equal"] + tk["explained"] != tk["texts"]
                                           or not isinstance(nm, (int, float)) or nm > 1e-4):
            bad.append("cpu_check passing needs 0 other greedy divergences, every tokenizer text equal or explained, "
                       "and same-weights NMSE <= 1e-4 (the locked bar)")
    for b in bad:
        problems.add(where, b)
    return len(bad)


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
    """The model's code at the release that ships it: the registry release, or the model's card_release when that is
    newer (a model added after the last tag lives in the next one; v0.1.3 has no track2rig folder)."""
    v, cr = reg["version"], m.get("card_release")
    return f"{gh_base(reg)}/tree/v{cr if ver_newer(cr, v) else v}/{m['path']}"


def ver_newer(a, b):
    """True when version a ("0.1.4") is set and newer than version b."""
    return bool(a) and tuple(map(int, a.split("."))) > tuple(map(int, b.split(".")))


def docs_url(m):
    return f"https://mashbox.gg/docs/models/{m['id']}"


def chip_label(m, chip):
    """"TPU v5e (v5e-4)": the chip column's name plus the host the model needs when it differs (one source: the row's accelerator)."""
    acc = (m["chips"].get(chip) or {}).get("accelerator")
    return CHIP_LABEL[chip] + (f" ({acc})" if acc and acc not in (chip, CHIP_LABEL[chip].split("(")[-1].rstrip(")")) else "")


def not_run_reason(m, chip):
    """Why a pending chip has not been run (models.json chips.<chip>.not_run_reason), "" when it states none. A released
    row may say "pending" only with one; the pages print "Not tested yet", never "Not supported"."""
    return str((m["chips"].get(chip) or {}).get("not_run_reason") or "").strip()


def real_date(s):
    """True for a whole-field YYYY-MM-DD that is a calendar date no later than tomorrow (UTC): a run's date, never
    9999-99-99, 2026-02-30, a date nobody has reached, or a date with words after it"""
    s = str(s)
    if not DATE_RE.match(s) or s != s.strip():
        return False
    try:
        d = datetime.date.fromisoformat(s)
    except ValueError:
        return False
    return d <= datetime.datetime.now(datetime.timezone.utc).date() + datetime.timedelta(days=1)


def chip_rows_measured(m):
    """The chips rows a results row of m was measured on: a result's chip names its row exactly (v6e-4, v5p-8) or by
    its type (v6e-1 -> v6e, v5e-4 -> v5e)."""
    out = set()
    for r in m.get("results") or []:
        if not real_date(r.get("date", "")):   # a result without a real date proves no run
            continue
        out |= chips_named(r.get("chip"))
    return out


def chips_named(text):
    """The chips rows a results row's chip names: exactly (v6e-4, v5p-8) or by type (v6e-1 -> v6e, v5e-4 -> v5e)."""
    return {t if t in CHIPS else t.split("-")[0] for t in CHIP_TOKEN_RE.findall(str(text or "").lower())}


# a measured number with its unit in a run_note ("2.46 s", "38.6 tok/s"); a chip name's digits (v6e-4) are not one
RUN_NOTE_NUMBER_RE = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)\s*(?:ms|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?|"
                                r"fps|tok/s|tokens/s|tokens per second|tokens|gib|gb|mb|%|x|\u00d7)(?![\w])")


def measured_run(m, chip):
    """Why a measured run of m exists on chips.<chip>, "" when none: a dated results row on that chip, or the row's
    measured block. Such a row is never "not supported": a chip a model ran on is never dropped."""
    if chip in chip_rows_measured(m):
        return "a dated results row on it"
    if ((m.get("chips") or {}).get(chip) or {}).get("measured"):
        return "its measured block"
    return ""


def never_run(m, chip):
    """True when nobody has run m on chips.<chip>: not passing or failing, no real date (a date of 'never' or
    '9999-99-99' is not a run), and no results row on that chip"""
    c = m["chips"].get(chip) or {}
    return (c.get("status") not in ("passing", "failing", "measured") and not real_date(c.get("date", ""))
            and chip not in chip_rows_measured(m))


# a sentence saying nobody ran the chip ("not tested", "hasn't been tested", "yet to be tested", "not yet run", "has not
# been run", "never run on this chip", "no run yet", "TBD"), never a run's finding that the model does not run there
# ("does not run on a 16 GB chip", "XLA would not run the decoder", "will never run in 16 GB"); read after row_words()
NOT_TESTED_RE = re.compile(
    r"\b(?:untested|untried|tbd)\b"
    r"|(?:\bnot|n't|\bnever)\s+(?:yet\s+)?(?:been\s+)?(?:yet\s+)?(?:tested|tried)\b"
    r"|\b(?:yet\s+)?to\s+be\s+(?:tested|tried)\b|\byet\s+to\s+be\s+run\b"
    r"|(?:\bnot|n't|\bnever)\s+(?:yet\s+)?been\s+run\b|\bnot\s+yet\s+run\b|(?<!does\s)(?<!did\s)(?<!do\s)\bnot\s+run\s+yet\b"
    r"|\b(?:was|were|is|are|has|have|had)(?:\s+not|n't)\s+(?:yet\s+)?run\b"
    r"|(?<!will\s)(?<!would\s)(?<!could\s)(?<!can\s)(?<!may\s)(?<!might\s)(?<!should\s)\bnever\s+run\b"
    r"|\b(?:nobody|no\s+one)\s+(?:has\s+|have\s+)?(?:ran|run|tested|tried)\b|\bno\s+runs?\s+(?:yet|so\s+far)\b")
NOT_SUPPORTED_RE = re.compile(r"\b(?:not\s+supported|unsupported)\b")
ZERO_WIDTH_RE = re.compile("[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")


def row_words(text):
    """A chips row's text as the row rules read it, with the chip spellings our own docs and records use: NFKC (full-width letters, no-break spaces), zero-width characters dropped, a curly apostrophe made
    straight, lower case, and the chip names our docs and records use made the catalog's own: v5litepod-4 -> v5e-4
    (gcloud), TPU v6 lite -> v6e (the device kind), Trillium -> v6e, TPUv6e / TPU-v6e -> tpu v6e."""
    s = ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFKC", str(text or ""))).replace("\u2019", "'").lower()
    s = re.sub(r"\bv5\s*lite\s*-?\s*pod", "v5e", s)
    s = re.sub(r"\bv([56])\s*lite\b", r"v\1e", s)
    s = re.sub(r"\btrillium\b", "v6e", s)
    return re.sub(r"\btpu\s*-?\s*(?=v[56])", "tpu ", s)


def chip_row_problems(m, chip):
    """What a chips row may say, read through row_words(): its reason, note and accelerator speak for its own
    chip type only (a v5e row never names v6e, v6e-1, v6e-4, Trillium or TPUv6e: 'runs on TPU v6e only' and every
    paraphrase of it); its date, when it has one, is a real date; a reason states a finding, never that the chip was
    not tested; a row nobody ran reads 'Not tested yet' on every page, carries no reason and prints 'Not supported'
    nowhere (status, note, reason, not_run_reason)."""
    c = m["chips"].get(chip) or {}
    out, kind = [], chip.split("-")[0]
    if "date" in c and c.get("status") not in ("passing", "failing", "measured") and not real_date(c["date"]):
        out.append(f"chips.{chip}.date {c['date']!r} is not a real YYYY-MM-DD date (not after tomorrow, UTC): a row "
                   "nobody ran has no date")
    for key in ("reason", "note", "accelerator", "run_note"):
        other = sorted({t for t in CHIP_TOKEN_RE.findall(row_words(c.get(key))) if t.split("-")[0] != kind})
        if other:
            out.append(f"chips.{chip}.{key} names another chip ({', '.join(other)}): a row speaks for its own chip only")
    if NOT_TESTED_RE.search(row_words(c.get("reason"))):
        out.append(f"chips.{chip}.reason says the chip was not tested: a reason states what a run found; a chip "
                   "nobody ran says 'Not tested yet' (note, or pending with its not_run_reason)")
    if c.get("status") == "not supported" and never_run(m, chip):
        out.append(f"chips.{chip} was never run, so it is \"pending\" with its not_run_reason, never status \"not "
                   "supported\": a chip nobody ran reads 'Not tested yet'")
    if c.get("status") == "pending" and measured_run(m, chip):
        out.append(f"chips.{chip} has a measured run on its chip ({measured_run(m, chip)}), so it never reads \"Not "
                   "tested yet\" and is never pending: passing, or measured when this version's `mashbox run` starts "
                   "elsewhere")
    if c.get("status") == "not supported" and measured_run(m, chip):
        out.append(f"chips.{chip} has a measured run on its chip ({measured_run(m, chip)}), so it never has status "
                   "\"not supported\": a chip the model ran on is passing, or measured when this version's `mashbox "
                   "run` starts elsewhere, with its measured number")
    if str(c.get("run_note") or "").strip():
        said = set(RUN_NOTE_NUMBER_RE.findall(row_words(c["run_note"])))
        have = {n for r in m.get("results") or [] if real_date(r.get("date", "")) and chip in chips_named(r.get("chip"))
                for n in re.findall(r"\d+(?:\.\d+)?", str(r.get("value") or ""))}
        if said - have:
            out.append(f"chips.{chip}.run_note states {', '.join(sorted(said - have))}, not a value of a dated results "
                       "row on this chip: a run_note copies its numbers from the measured result")
    if never_run(m, chip):
        if chip_text(m, chip) != "Not tested yet":
            out.append(f"chips.{chip} was never run (no date, no result on it) but reads {chip_text(m, chip)!r}: "
                       "a chip nobody ran reads 'Not tested yet', never 'Not supported'")
        if str(c.get("reason") or "").strip():
            out.append(f"chips.{chip} was never run, so it carries no reason: there is no finding to state")
    prints_not_tested = row_words(chip_text(m, chip)).lstrip().startswith("not tested yet")   # a note the pages print
    if never_run(m, chip) or chip_cell(m, chip) == "Not tested yet" or prints_not_tested:
        said = [k for k in ("note", "reason", "not_run_reason", "accelerator")
                if NOT_SUPPORTED_RE.search(row_words(c.get(k)))]
        if said:
            out.append(f"chips.{chip} reads 'Not tested yet' but its {' and '.join(said)} says Not supported: the chips "
                       "table prints the note and the chip label the accelerator; a chip nobody ran prints 'Not "
                       "supported' nowhere")
    return out


def chip_cell(m, chip):
    """A chip's status for the released version: Supported or Not supported. The test date stays in models.json
    (validate() needs it); public views never print it. A pending row reads "Not tested yet": it never claims a result,
    and the release gate (validate --release) stops a release that has one without its not_run_reason. A measured row
    reads "Measured" and the views print its note (chip_note) in its place, never "Not supported"."""
    st = (m["chips"].get(chip) or {}).get("status", "")
    if st == "passing":
        return "Supported"
    if st == "measured":   # the pages print its note (chip_note): what was measured and where `mashbox run` starts
        return "Measured"
    if st == "pending":
        return "Not tested yet"
    return "Not supported" if st in ("failing", "not supported") else "n/a"


def chip_note(m, chip):
    """The public note of a chips row that is not supported or measured (e.g. "One chip: the full pipeline fits ..."):
    views print it in place of "Not supported: <reason>" or "Measured". The status stays as it is, so mashbox run
    behaves the same. "" when none."""
    c = m["chips"].get(chip) or {}
    return c.get("note", "") if c.get("note") and chip_cell(m, chip) in ("Not supported", "Measured") else ""


def weights_cell(m):
    if m["hf_repo"]:
        return f"[{m['hf_repo'].split('/', 1)[1]}](https://huggingface.co/{m['hf_repo']})"
    return f"[original](https://huggingface.co/{m['upstream']['hf']})"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def live(reg):
    """Models shown in the views. A model with "launch" set is not released, so no public view names it."""
    return [m for m in reg["models"] if not m.get("launch")]


def models_table(reg, absolute):
    rows = []
    for m in live(reg):
        # A model newer than the registry release is not in that tag: its README link pins to the release that ships it.
        link = code_url(reg, m) if absolute or ver_newer(m.get("card_release"), reg["version"]) else m["path"]
        rows.append([f"**[{m['name']}]({link})**<br>by {m['upstream']['maker']}" + experimental_note(m, "<br>"),
                     f"{KINDS[m['kind']]['label']}: {m['in']} in, {m['out']} out",
                     ("CPU: Supported<br>" if runs_cpu(m) else "")
                     + "<br>".join(f"{chip_label(m, c)}: {chip_note(m, c) or chip_cell(m, c)}"
                                 + (f", {run_note(m, c)}" if chip_cell(m, c) == "Supported" and run_note(m, c) else "")
                                 for c in CHIPS), readme_measured(m),
                     weights_cell(m)])
    return table(["Model", "What it does", "Chips", "Measured", "Weights"], rows)


def short_metric(r):
    return r["metric"].split(" (", 1)[0]


def measured_cell(m, n=2):
    """Headline results for the README table: the first n results of the model, chip named."""
    if not m["results"]:
        return "n/a"
    return "<br>".join(f"{short_metric(r)}: **{r['value'].split(' (', 1)[0]}** {_lists().chip_note(r['chip'])}" for r in m["results"][:n])


def readme_measured(m):
    """README cell: the headline numbers marked readme, on one line, plus a link to the card."""
    hl = [h for h in m.get("headline") or [] if h.get("readme")]
    if not hl or not m["hf_repo"]:
        return measured_cell(m)
    r = m["results"][hl[0]["results"][0]]
    return ("; ".join(f"**{h['big']}** {h['label']}" for h in hl) + f" {_lists().chip_note(r['chip'])}. "
            f"[All results](https://huggingface.co/{m['hf_repo']})")


CATALOG_URL = "https://mashbox.gg/docs/supported-models"


@functools.lru_cache(maxsize=None)
def _lists():
    """tools/model_lists.py of this tool's own tree (standard library only), loaded by path: the one render of the
    model lists (counts, families, the tested list, makers) behind every model list on every page. It reads
    mashbox/catalog.py and never changes it (catalog.py is in the code set of every GGUF chip record)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mashbox_model_lists", Path(__file__).resolve().parent / "model_lists.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gguf_block(reg):
    """The overviews' GGUF tier (model_lists.gguf_block): every gguf[] row, one collapsed section per publisher, each
    file linking its exact file on Hugging Face, then the catalog link. Rendered, never typed: a new gguf[] row reaches
    the README with render --write."""
    return _lists().gguf_block(reg, CATALOG_URL)


def builds_head(reg):
    """The top of the README's models block: the lead line (the catalog's size from models.json), then the MASHBOX
    builds tier."""
    ml = _lists()
    return f"**{ml.lead_line(reg)}.**\n\n**{ml.BUILDS}**\n\n{ml.BUILDS_LINE}\n\n"


def readme_block(reg):
    """The README models block: the lead line, then two labelled tiers: MASHBOX builds (our ports table), then every
    GGUF file by publisher."""
    return builds_head(reg) + models_table(reg, absolute=False) + "\n\n" + gguf_block(reg)


# The Hugging Face org card is the org's static README Space. Like the GitHub org profile it is an overview with no
# model list (overview_problems), and the "TPU models" collection holds every live TPU port.
ORG_CARD_URL = "https://huggingface.co/spaces/{org}/README/raw/main/README.md"
TPU_COLLECTION = "TPU models"


def tpu_ports(reg):
    """The live models whose Hugging Face repo is a TPU port (named <Upstream>-TPU): the "TPU models" collection."""
    return {m["hf_repo"] for m in live(reg) if m["hf_repo"] and m["hf_repo"].endswith("-TPU")}


FAMILIES_RE = re.compile(r"\bFamilies covered(?: so far)?: ([^.\n]+)\.")


def overview_problems(reg, text, where):
    """[(where, problem)] for an org overview (the GitHub org profile, the Hugging Face org card: a
    general overview, the families we cover and links): no models block, no model named, no lead line, count line or
    typed count; a "Families covered: ..." line names exactly models.json's GGUF families (any order), so a new family
    shows as drift; the company line is models.json's. The model lists live in README, llms.txt and the docs catalog."""
    out = []
    if MARK_START in text or MARK_END in text:
        out.append((where, f"carries a {MARK_START} block: the overviews list no models (README, llms.txt and the docs "
                           "catalog do)"))
    for rx, what in ((LEAD_RE, "lead line"), (ENTRIES_RE, "count line")):
        for m in rx.finditer(text):
            out.append((where, f"prints the {what} {m.group(0)!r}: the overviews print no count"))
    out += [(w, msg) for w, msg in list_problems(reg, GEN_BLOCK_RE.sub(" ", text), where, names=True)]
    want = _lists().gguf_families(reg)
    for m in FAMILIES_RE.finditer(text):
        got = [f.strip() for f in re.split(r",\s*|\s+and\s+", m.group(1)) if f.strip()]
        lost = [f for f in want if f not in got]
        extra = [f for f in got if f not in want]
        if lost or extra:
            out.append((where, f"family line {m.group(0)!r} differs from models.json's GGUF families {want}"
                               + (f" (missing {lost})" if lost else "") + (f" (not in models.json: {extra})" if extra else "")))
    if reg.get("company_line") and reg["company_line"] not in text:
        out.append((where, "company line differs from models.json company_line"))
    return out


def collection_problems(reg, body, where):
    """[(where, problem)]: the org's "TPU models" collection holds exactly the live TPU ports of our org (other orgs'
    repos may sit in it; family collections are not checked here)."""
    cols = [c for c in json.loads(body) if c.get("title") == TPU_COLLECTION]
    if len(cols) != 1:
        return [(where, f"{len(cols)} collections titled {TPU_COLLECTION!r} (want 1)")]
    org = reg["hf"]["org"]
    ours = {i["id"] for i in cols[0].get("items") or [] if i.get("type") == "model" and i["id"].startswith(org + "/")}
    want = tpu_ports(reg)
    return ([(where, f"{TPU_COLLECTION!r} lacks {r}") for r in sorted(want - ours)]
            + [(where, f"{TPU_COLLECTION!r} holds {r}, which is not a live TPU port in models.json")
               for r in sorted(ours - want)])


# Lists and counts live only in rendered blocks. Outside them a page names no count of models (COUNT_RE), and the org
# profile's prose names no model (its one list is the block): a hand-typed list or count drifts the day a row lands.
GEN_BLOCK_RE = re.compile(r"<!-- ([\w-]+):start\b[^>]*-->.*?<!-- \1:end\b[^>]*-->", re.S)
COUNT_RE = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozens of|\d+)\s+(?:[\w.-]+\s+){0,2}?"
                      r"(?:models|model ports|ports|GGUF files|LLMs)\b", re.I)
ENTRIES_RE = re.compile(r"\b(\d+) entr(?:y|ies): ([^.\]\n]*)")
PART_RE = re.compile(r"\b(\d+) (?:MASHBOX ports?|MASHBOX models?|GGUF files?)\b")
LEAD_RE = re.compile(r"\b(\d+) models? and files? run in mashbox: (\d+) MASHBOX builds? and (\d+) GGUF files? of "
                     r"(\d+) models?\b")


def list_problems(reg, text, where, names=False):
    """[(where, problem)]: a count of models typed outside a rendered block; with names (the org profile) a model named
    outside it (name, id or its size such as 27B); and any printed count line whose parts do not add up to its total or
    whose total is not models.json's."""
    out, prose = [], GEN_BLOCK_RE.sub(" ", text)
    for m in COUNT_RE.finditer(prose):
        out.append((where, f"typed count {m.group(0)!r} outside a rendered block (counts come from models.json)"))
    if names:   # a command in a code block (the install lines) is not a list
        prose, ml = re.sub(r"```.*?```", " ", prose, flags=re.S), _lists()
        words = {m["name"] for m in live(reg)} | {m["id"] for m in live(reg)} | {ml.gguf_name(r) for r in
                                                                                 ml.catalog().gguf_rows(reg)}
        sizes = [((m.get("catalog") or {}).get("params_b") or {}).get("total") for m in live(reg)]
        words |= {f"{t:g}B" for t in sizes if t} | {f"{round(t)}B" for t in sizes if t}   # 26.9B and 27B
        words |= {w for m in live(reg) for w in re.findall(r"(?<![\d.])\d+(?:\.\d+)?B\b", m["name"])}   # 27B in Qwen3.8-27B
        for w in sorted(words):
            if re.search(rf"(?<![\w.-]){re.escape(w)}(?![\w-])", prose, re.I):
                out.append((where, f"names {w!r} outside the models block (the block is the one list)"))
    total = _lists().counts(reg)["entries"]
    for m in ENTRIES_RE.finditer(text):
        parts = [int(x) for x in PART_RE.findall(m.group(2))]
        if sum(parts) != int(m.group(1)) or int(m.group(1)) != total:
            out.append((where, f"count line {m.group(0)!r} does not add up to models.json's {total} entries"))
    c = _lists().counts(reg)
    want = (c["entries"], c["ports"] + c["own"], c["files"], c["gguf_models"])
    for m in LEAD_RE.finditer(text):
        if tuple(int(x) for x in m.groups()) != want or want[0] != want[1] + want[2]:
            out.append((where, (f"lead line {m.group(0)!r} is not models.json's catalog ({want[0]} models and files: "
                                f"{want[1]} MASHBOX builds and {want[2]} GGUF files of {want[3]} models)")))
    return out


def run_note(m, chip):
    """A passing or measured chips row's run_note (its measured speed and any condition, in a few words), "" otherwise."""
    c = m["chips"].get(chip) or {}
    return str(c.get("run_note") or "").strip() if c.get("status") in ("passing", "measured") else ""


def runs_cpu(m):
    """True when m's chips.cpu row passes (a model that runs on a CPU): every chip list names CPU first."""
    return (m["chips"].get("cpu") or {}).get("status") == "passing"


def runs_on(m):
    """" Runs on: TPU v5e (v5e-4), TPU v6e (v6e-1)." from the passing and measured chips rows, CPU first when its cpu
    row passes (agents read llms.txt to pick a chip); a row's run_note follows its chip ("TPU v6e (v6e-1): 2.46 s of
    compute per second of video (...)"), and the chips are then separated by semicolons."""
    keys = (["cpu"] if runs_cpu(m) else []) + [c for c in CHIPS
                                               if (m["chips"].get(c) or {}).get("status") in ("passing", "measured")]
    ok = [("CPU" if c == "cpu" else chip_label(m, c)) + (f": {run_note(m, c)}" if run_note(m, c) else "") for c in keys]
    sep = "; " if any(run_note(m, c) for c in keys) else ", "
    return f" Runs on: {sep.join(ok)}." if ok else ""


def experimental(m):
    """True when the row sets catalog.experimental (the one flag behind the "Experimental" label)."""
    return bool((m.get("catalog") or {}).get("experimental"))


def experimental_note(m, sep):
    """The plain label "Experimental", printed with the model's name when catalog.experimental is true; "" otherwise."""
    return f"{sep}Experimental" if experimental(m) else ""


def llms_block(reg):
    """llms.txt models block, the same two tiers: each MASHBOX build, then the GGUF files with a passing chip test and
    the catalog link with the count line (model_lists.llms_lines)."""
    ml = _lists()
    return "\n".join([f"{ml.BUILDS}:"] + [
        f"- [{m['name']}]({docs_url(m)}): {m['in']} in, {m['out']} out. {KINDS[m['kind']]['label']}."
        + experimental_note(m, " ") + ("." if experimental(m) else "") + runs_on(m) for m in live(reg)]
        + ["", f"{ml.GGUF}:"] + ml.llms_lines(reg, CATALOG_URL))


def front_matter(reg, m):
    up = m["upstream"]
    lines = [] if up.get("own") else [f"base_model: {up['hf']}"]   # our own model has no base model
    if m.get("base_model_relation"):
        lines.append(f"base_model_relation: {m['base_model_relation']}")
    lines.append(f"license: {up['license']}")
    if up["license"] == "other":
        lines += [f"license_name: {up['license_name']}", f"license_link: {up['license_link']}"]
    if m.get("pipeline_tag"):
        lines.append(f"pipeline_tag: {m['pipeline_tag']}")
    tags = ["tpu", "jax", "mashbox"] if m["kind"] != "llm-gguf" else ["gguf", "llama.cpp", "mashbox"]
    if m["kind"] == "rig":
        tags = ["jax", "mashbox", "rigging", "animation", "gltf"]   # CPU model: no tpu tag
    lines.append("tags:")
    lines += [f"- {t}" for t in tags]
    return "\n".join(lines)


def results_table(m):
    if not m["results"]:
        return "n/a"
    if all(r.get("short") for r in m["results"]):
        notes, rows = [], []

        def mark(ns):
            for n in ns:
                if n not in notes:
                    notes.append(n)
            return f"<sup>{','.join(str(notes.index(n) + 1) for n in ns)}</sup>" if ns else ""
        same = len({r["chip"] for r in m["results"]}) == 1  # one chip: say it once (phone width)
        for r in m["results"]:
            sh = r["short"]
            rows.append([sh["measure"] + mark(sh.get("notes", [])), f"**{sh['ours']}**",
                         sh["vs"] + mark(sh.get("vs_notes", []))] + ([] if same else [r["chip"]]))
        foot = "\n".join(f"<sub><sup>{i}</sup> {n}</sub><br>" for i, n in enumerate(notes, 1))
        r0 = m["results"][0]
        head = f"All {len(rows)} measurements" + (f", all on {r0['chip']}" if same else "")
        return (f"<details>\n<summary>{head}</summary>\n\n"
                + table(["Measure", "Ours", "Compared with"] + ([] if same else ["Chip"]), rows)
                + f"\n\n{foot}\n\n</details>")
    return table(["Chip", "Measure", "Result", "Baseline"],
                 [[r["chip"], r["metric"], r["value"], r["baseline"]] for r in m["results"]])


def files_table(m):
    lead = "This port loads the original weights unchanged from " \
           f"[{m['upstream']['hf']}](https://huggingface.co/{m['upstream']['hf']})."
    if not m["files"]:
        return lead
    if m["kind"] == "rig":   # weights and example clips, no reduced-size variants: no quality or speed columns
        size = lambda n: f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} KB"
        return table(["File", "Format", "Size", "Note"],
                     [[f"`{f['name']}`", f["format"], size(f["bytes"]), f.get("note") or f["kind"]] for f in m["files"]])
    rows = []
    for f in m["files"]:
        rows.append([f"`{f['name']}`", f["format"], f"{f['gb']:.2f}",
                     f"{f['bpw']:.2f}" if isinstance(f.get("bpw"), (int, float)) else "n/a",
                     f.get("quality") or ("reference" if not f.get("variant") else "n/a"), f.get("speed") or "n/a"])
    out = table(["File", "Format", "Size (GB)", "Bits per weight", "Quality vs full precision", "Speed"], rows)
    if not any(f.get("kind", "weights") == "weights" for f in m["files"]):  # e.g. only a compile cache of ours
        return lead + "\n\n" + out
    return out


def chips_table(m):
    rows = []
    for chip in CHIPS:
        c = m["chips"].get(chip) or {}
        own = chip_note(m, chip)
        if own:  # "One chip: the full pipeline fits ..." -> status "One chip", note the rest
            head, _, rest = own.partition(": ")
            rows.append([CHIP_LABEL[chip], head, rest] if rest else [CHIP_LABEL[chip], own, ""])
            continue
        note = c.get("reason") or c.get("note") or ""
        rows.append([CHIP_LABEL[chip], chip_cell(m, chip), note])
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


SPDX = {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-3-clause": "BSD-3-Clause", "cc-by-4.0": "CC-BY-4.0"}


def weights_licence(m):
    """The card's weights line: the upstream licence recorded at our pin, never our own (someone else's weights, and
    our conversions of them, follow their licence)."""
    up = m["upstream"]
    name = SPDX.get(str(up["license"]).lower(), up.get("license_name") or up["license"])
    rev = up.get("revision")
    pin = f"{up['hf']}@{rev[:8]}" if rev else up["hf"]
    url = f"https://huggingface.co/{up['hf']}/tree/{rev}" if rev else f"https://huggingface.co/{up['hf']}"
    return f"Weights: {name} licence, from [{pin}]({url})"


def upstream_licence_file(m, source=False):
    """The line that says the upstream LICENSE file travels with our repo, unchanged. source=True also names where it
    comes from (the world card); the LLM cards keep their words until their next card upload."""
    if not (m["upstream"].get("license_sha256") and m["hf_repo"]):
        return ""
    if source:
        up = m["upstream"]
        return (f"[LICENSE](https://huggingface.co/{m['hf_repo']}/blob/main/LICENSE) in this repo is {up['maker']}'s "
                f"licence file from [{up['hf']}](https://huggingface.co/{up['hf']}) on Hugging Face, unchanged. ")
    return (f"{m['upstream']['maker']}'s LICENSE file is in this repo, unchanged: "
            f"[LICENSE](https://huggingface.co/{m['hf_repo']}/blob/main/LICENSE). ")


def own_files(repo, m):
    """Our own files in the model's HF repo, as its upstream.json pins them: (name, title, licence). The licence of our
    own weights is set by a maintainer and recorded on the file's row; it is never derived."""
    f = Path(repo) / m["path"] / "upstream.json" if repo is not None else None
    if f is None or not f.is_file() or not m["hf_repo"]:
        return []
    return [(x["name"], x.get("title"), x.get("license")) for r in json.loads(f.read_text()).get("repos", {}).values()
            if r.get("hf") == m["hf_repo"] for x in r.get("files", [])]


def own_files_licence(repo, m):
    rows = [f"`{n}`" + (f" ({t})" if t else "") + f": {SPDX.get(str(lic).lower(), lic)}"
            for n, t, lic in own_files(repo, m) if lic]
    return ("Our own weights in this repo: " + "; ".join(rows) + ". ") if rows else ""


def highlights(m):
    out = [f"- **{r['value']}**: " + (r["highlight"] if r.get("highlight") else f"{r['metric'][0].lower() + r['metric'][1:]}, on {r['chip']}")
           for r in m["results"]]
    for f in m["files"]:
        if f.get("quality") and f.get("default"):
            out.append(f"- **{f['quant_type'] if f.get('quant_type') else f['name']}** ({f['gb']:.2f} GB, "
                       f"{f['bpw']:.2f} bits per weight): {f['quality']}")
    return "\n".join(out)


def headline(m):
    """At most 4 headline numbers as a 2 x 2 strip (reads on a phone), the chip under each; else the bullets."""
    hl = m.get("headline")
    if not hl:
        return highlights(m)
    cells = []
    for h in hl:
        r = m["results"][h["results"][0]]
        cells.append(f'<td width="50%"><b>{h["big"]}</b><br>{h["label"]}<br><sub>{r["chip"]}</sub></td>')
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
    alt = f"{m['name']} preview" if m["kind"] == "rig" else f"{m['name']} running on Google Cloud TPU"
    return (f'<img alt="{alt}" width="720" '
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
    return "Each chart names the hardware on both sides.\n\n" + "\n\n".join(out)


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
    # a card keeps the GitHub links of the release its pinned Hugging Face revision was published with (card_release)
    reg = dict(reg, version=m.get("card_release") or reg["version"])
    up = m["upstream"]
    return {
        "banner": (f"https://huggingface.co/{m['hf_repo']}/resolve/main/assets/banner-dark.png" if m["hf_repo"] else ""),
        "badges": badges(reg, m), "highlights": highlights(m), "headline": headline(m), "top_chart": top_chart(repo, m),
        "hero": hero(repo, m), "links_table": links_table(reg, m),
        "front_matter": front_matter(reg, m),
        "name": m["name"], "id": m["id"], "summary": m["summary"], "in": m["in"], "out": m["out"],
        "run": m["run"], "install": m.get("install") or reg["install"], "maker": up["maker"], "upstream_hf": up["hf"],
        "upstream_url": f"https://huggingface.co/{up['hf']}",
        "upstream_github": f"https://github.com/{up['github']}" if up.get("github") else f"https://huggingface.co/{up['hf']}",
        "hf_repo": m["hf_repo"] or "", "version": reg["version"], "code_url": code_url(reg, m),
        "docs_url": docs_url(m), "agents_url": f"{gh_base(reg)}/blob/v{reg['version']}/AGENTS.md",
        "results_table": results_table(m), "results_charts": results_charts(repo, m), "files_table": files_table(m), "chips_table": chips_table(m),
        "license_text": license_text(reg, m), "our_license": reg["license"],
        "weights_licence": weights_licence(m), "upstream_licence_file": upstream_licence_file(m),
        "upstream_licence_source": upstream_licence_file(m, source=True), "own_files_licence": own_files_licence(repo, m),
        "notes": m.get("notes", ""), "agents_section": agents_section(m),
        "kernels_section": kernels_section(reg, m),
        "kernels_license": ((f"The compiled kernels in `kernels/` are under their own licence: "
                             f"[LICENSE-KERNELS]({gh_base(reg)}/blob/v{reg['version']}/LICENSE-KERNELS).")
                            + (" The compile cache in `xla-cache/` is under the same licence."
                               if any(f.get("kind") == "xla-cache" for f in m.get("files", [])) else ""))
                           if m.get("kernels") else "",
        "default_tag": (m.get("default_tag") or ""),
        "weights_version": weights_version(m),
        "maturity_label": experimental_note(m, ""),
    }


def default_weights(m):
    """The default weights file row of files[] (kind weights, default true), or None."""
    return next((f for f in m.get("files") or [] if f.get("kind") == "weights" and f.get("default")), None)


def weights_version(m):
    """One line naming the weights of a model with a dated weights version (hf_tag) by file and sha256; "" without one.
    It never prints a Hugging Face tag or the revision sha: uploading the card makes a new revision that no tag points
    at (a tag is never moved), so a tag named in the card would name another revision (check --online refuses that)."""
    w = default_weights(m)
    if not m.get("hf_tag") or not w or not w.get("sha256"):
        return ""
    return f"The weights file `{w['name']}` has sha256 `{w['sha256']}`."


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


LINK_RE = re.compile(r'(!?\[[^\]]*\]\()([^)\s]+)((?:[ \t]+"[^"\n]*")?\))|((?:src|srcset|href)=["\'])([^"\']+)(["\'])')
REF_RE = re.compile(r"^( {0,3}\[[^\]]+\]:[ \t]*<?)([^\s>]+)", re.M)   # reference-style: [id]: target


def pin_links(reg, repo, text, where, problems):
    """README is also the PyPI page, where repo-relative paths do not resolve: every relative link and image becomes an
    absolute URL pinned to the release tag v<version> (images from raw.githubusercontent.com, files from blob/, folders
    from tree/), as the HF cards already pin code_url. Pinned URLs of an older version move to the current one."""
    org, name, ver = reg["github"]["org"], reg["github"]["repo"], reg["version"]
    old = re.compile(rf"(https://github\.com/{org}/{name}/(?:blob|tree)/|https://raw\.githubusercontent\.com/{org}/{name}/)"
                     rf"v(\d+\.\d+\.\d+)(?=/)")
    text = old.sub(lambda mt: mt.group(0) if ver_newer(mt.group(2), ver) else mt.group(1) + "v" + ver, text)

    def sub(mt):
        if mt.group(2) is not None:
            pre, target, post, image = mt.group(1), mt.group(2), mt.group(3), mt.group(1).startswith("!")
        else:
            pre, target, post = mt.group(4), mt.group(5), mt.group(6)
            image = not pre.startswith("href")
        if re.match(r"[a-z][a-z0-9+.-]*:|#|/", target):
            return mt.group(0)
        path, _, frag = target.partition("#")
        if not (repo / path).exists():
            problems.add(where, f"relative link to a missing path: {target}")
            return mt.group(0)
        if image:
            url = f"https://raw.githubusercontent.com/{org}/{name}/v{ver}/{path}"
        else:
            url = f"{gh_base(reg)}/{'tree' if (repo / path).is_dir() else 'blob'}/v{ver}/{path.rstrip('/')}"
        return pre + url + ("#" + frag if frag else "") + post

    def ref(mt):
        return sub(re.match(r"(\[\]\()(.*)(\))", f"[]({mt.group(2)})")).replace("[](", mt.group(1), 1)[:-1]
    return REF_RE.sub(ref, LINK_RE.sub(sub, text))


def render_all(reg, repo, profile, templates, problems, places=None, org_card=None):
    """Return {relative path: rendered text} for every view. places: the pages section of the private
    number map (None without one): pages get their markers back from it, are filled, and come out without markers;
    the new places are left in render_all.places."""
    views = {}
    readme = repo / "README.md"
    if readme.exists():
        out = splice(readme.read_text(), readme_block(reg), "README.md", problems)
        if out is not None:
            views["README.md"] = pin_links(reg, repo, out, "README.md", problems)
    else:
        problems.add("README.md", "missing")
    llms = repo / "llms.txt"
    if llms.exists():
        out = splice(llms.read_text(), llms_block(reg), "llms.txt", problems)
        if out is not None:
            views["llms.txt"] = out
    else:
        problems.add("llms.txt", "missing")
    for page in (profile, org_card):   # overviews: checked, never rendered (they carry no list to render)
        if page is not None:
            for w, msg in overview_problems(reg, page.read_text(), str(page)):
                problems.add(w, msg)
    for m in live(reg):  # a "coming soon" model's card is not published yet; it is checked again at its launch
        if m["hf_repo"]:
            views[f"hf/{m['id']}/README.md"] = render_card(reg, m, templates, repo)
    if (repo / "docs").is_dir() and (repo / "src" / "mashbox").is_dir():
        views.update(docs_views(reg, repo, problems))
    prose = set()
    for page in number_pages(reg, repo) + ui_pages(repo):   # every public page (not the cards) and the app's UI text
        rel = page.relative_to(repo).as_posix()
        if rel.startswith(("hf/", "docs/_templates/")):
            continue
        prose.add(rel)
        if rel not in views and (rel in (places or {}) or MARK_RE.search(page.read_text())):
            views[rel] = page.read_text()
    render_all.places = {}
    for key in [k for k in views if k in prose]:
        text = fill_numbers(reg, mark_up(views[key], (places or {}).get(key, []), key, problems), key, problems)
        if places is None:
            views[key] = text            # no private map (public export, a test tree): a page keeps its own markers
            continue
        views[key], spots = unmark(text)
        if mark_up(views[key], spots) != text:
            problems.add(key, "number markers cannot be placed back from the page text alone (two equal numbers with "
                              "equal words before them): reword one of them")
        if spots:
            render_all.places[key] = spots
    return views


# ---------------------------------------------------------------- generated references (docs)
# As llama.cpp's gen-docs and update-ops-docs: the docs are rendered from the code and CI fails when they differ.

CLI_RE = re.compile(r"(<!-- cli:start ([a-z0-9 -]+) -->)(.*?)(<!-- cli:end -->)", re.S)
API_RE = re.compile(r"(<!-- api:start -->)(.*?)(<!-- api:end -->)", re.S)
CHIPS_RE = re.compile(r"(<!-- chips:start -->)(.*?)(<!-- chips:end -->)", re.S)
CATALOG_RE = re.compile(r"(<!-- catalog:start -->)(.*?)(<!-- catalog:end -->)", re.S)   # a docs page (mashbox.catalog)
ARCHS_RE = re.compile(r"(<!-- archs:start -->)(.*?)(<!-- archs:end -->)", re.S)         # docs/gguf.md (gguf.archs.ARCHS)
CLI_NOTE = ("Generated from the argparse parsers in src/mashbox/cli.py and src/mashbox/tpu/__init__.py by "
            "tools/hf_github_sync_check.py render --write. Do not edit.")


class _Parser(Exception):
    pass


def _parser_of(main, argv):
    """The parser a main() builds, caught at its parse_args call (for a CLI without a build_parser())."""
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Parser(self)
    argparse.ArgumentParser.parse_args = grab
    try:
        main(argv)
    except _Parser as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError("main() returned without parsing its arguments")


def _package(repo, problems):
    """Import mashbox from <repo>/src (stdlib-only modules), or None with a problem. sys.path is left as it was found:
    the package's later imports go through mashbox.__path__, and a caller that hands sys.path to a child process (a test
    run building a worker's PYTHONPATH) must not get this tree's src in front of its own."""
    import importlib
    sys.dont_write_bytecode = True      # never write __pycache__ into the tree being checked (a release export stays == its manifest)
    src = str((repo / "src").resolve())
    added = src not in sys.path
    if added:
        sys.path.insert(0, src)
    try:
        cached = sys.modules.get("mashbox")
        where = getattr(cached, "__file__", None)
        if cached is not None and not (where and str(Path(where).resolve()).startswith(src)):
            # an earlier call (another tree, or an installed copy) left a different mashbox in the module cache: drop
            # it, else every later call refuses this tree's own package and renders no docs views
            for name in [n for n in sys.modules if n == "mashbox" or n.startswith("mashbox.")]:
                del sys.modules[name]
            importlib.invalidate_caches()
        try:
            mods = {n: importlib.import_module(f"mashbox{n}") for n in ("", ".cli", ".tpu", ".runtime.openapi", ".catalog")}
        except Exception as e:
            problems.add("src/mashbox", f"cannot import the package to render the docs: {type(e).__name__}: {e}")
            return None
    finally:
        if added and src in sys.path:
            sys.path.remove(src)
    if not str(Path(mods[""].__file__).resolve()).startswith(src):
        problems.add("src/mashbox", f"mashbox imported from {mods[''].__file__}, not {src}")
        return None
    return mods


def _mashbox_as_found(check):
    """A caller of _package() puts back the mashbox it found loaded, on every way out. _package() drops a different
    loaded mashbox from sys.modules to import the tree checked; a process that runs the check in place (a test run)
    keeps the module objects it imported before, and a later test that patches one of them must patch the one the code
    runs. The checked tree's modules stay loaded until the caller returns: its later imports (a model's own parser)
    come from that tree. When the loaded package is still the one found (the call checked that same tree), its newly
    imported submodules stay, as after any import."""
    @functools.wraps(check)
    def run(*args, **kwargs):
        found = {n: m for n, m in sys.modules.items() if n == "mashbox" or n.startswith("mashbox.")}
        try:
            return check(*args, **kwargs)
        finally:
            if sys.modules.get("mashbox") is not found.get("mashbox"):
                for name in [n for n in sys.modules if n == "mashbox" or n.startswith("mashbox.")]:
                    del sys.modules[name]
                sys.modules.update(found)
    return run


def _subparsers(p):
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction):
            helps = {c.dest: c.help for c in a._choices_actions}
            return [(name, sp, helps.get(name) or "") for name, sp in a.choices.items()]
    return []


def _usage(name, p):
    if p.usage:
        return p.usage
    pos = []
    for a in p._actions:
        if not a.option_strings and not isinstance(a, argparse._SubParsersAction):
            m = (a.metavar or a.dest).upper()
            pos.append(f"[{m}]" if a.nargs == "?" else m)
    opts = any(a.option_strings and not isinstance(a, argparse._HelpAction) for a in p._actions)
    return " ".join([name] + pos + (["[options]"] if opts else []))


def options_table(p):
    rows = []
    for a in p._actions:
        if isinstance(a, (argparse._HelpAction, argparse._VersionAction, argparse._SubParsersAction)):
            continue
        if a.help == argparse.SUPPRESS:     # argparse's own --help leaves these out, so the docs do too
            continue
        if a.option_strings:
            opt = max(a.option_strings, key=len)
            cell = opt if a.nargs == 0 else f"{opt} {a.metavar or a.dest.upper()}"
        else:
            cell = (a.metavar or a.dest).upper()
        flag = a.nargs == 0
        default = "off" if flag else "" if a.default in (None, argparse.SUPPRESS) else f"`{a.default}`"
        text = (a.help or "").replace("%(default)s", str(a.default))
        if a.choices and not flag:
            text += (" " if text else "") + "(" + ", ".join(f"`{c}`" for c in a.choices) + ")"
        rows.append([f"`{cell}`", default, text.replace("|", "\\|")])
    return table(["Option", "Default", "What it does"], rows) if rows else "No options."


def cli_parsers(mods):
    """[(command line, parser, one-line help)] for every documented command."""
    top = mods[".cli"].build_parser()
    # the top parser lists `tpu` for --help only; its commands are rendered from mashbox.tpu's own parser below
    out = [(f"mashbox {n}", sp, h) for n, sp, h in _subparsers(top) if n != "tpu"]
    tpu = _parser_of(mods[".tpu"].main, ["status"])
    out += [(f"mashbox tpu {n}", sp, h) for n, sp, h in _subparsers(tpu)]
    return out


def model_parser(mods, cmd):
    """`run MODEL [FORM]` for a model with its own command line (cli.MODEL_CLI): the parser its app.parser(FORM) builds
    (FORM defaults to run, as in `mashbox run track2rig --help`; `run track2rig eval` is its eval form). None for any
    other command."""
    import importlib
    parts = cmd.split()
    mod = getattr(mods[".cli"], "MODEL_CLI", {}).get(parts[1]) if len(parts) in (2, 3) and parts[0] == "run" else None
    if mod is None:
        return None
    app = importlib.import_module(f"{mod}.app")
    return app.parser(parts[2] if len(parts) == 3 else "run")


def cli_page(mods):
    out = [f"<!-- {CLI_NOTE} -->", "# Command line", "",
           "Every `mashbox` command and option, rendered from the package's own parsers (`mashbox <command> --help` "
           "prints the same). On a TPU VM the command is `~/.local/bin/mashbox` until `~/.local/bin` is on your PATH."]
    for cmd, p, h in cli_parsers(mods):
        out += ["", f"## {cmd}", ""] + ([h[0].upper() + h[1:] + ".", ""] if h else []) + \
               ["```", _usage(cmd, p), "```", "", options_table(p)]
    return "\n".join(out) + "\n"


def api_block(mods, page):
    O = mods[".runtime.openapi"]
    link = os.path.relpath("docs/server-api.md", os.path.dirname(page))
    routes = ", ".join(f"`{r['method'].upper()} {r['path']}`" for r in O.ROUTES)
    samp = ", ".join(f"`{k}`" for k in O.SAMPLING)
    return (f"Routes: {routes}. Each one, with every request field and response shape: [Server API]({link}); the same as "
            f"an OpenAPI file at `GET /openapi.json`. Send `\"stream\": true` for server-sent events. Sampling fields on "
            f"every generating route: {samp}; temperature 0 (the default) is greedy.")


def chip_text(m, chip):
    """A models.json chip row as one line for the released version: "Supported: a v5e-4 (4 chips) ...", a measured
    row's note (its measured result), "Not tested yet", or "Not supported: <measured failure>". The condition follows a colon, never in brackets, so a page that
    shows the line up to its first " (" still carries it."""
    c = m["chips"].get(chip) or {}
    cell = chip_cell(m, chip)
    if chip_note(m, chip):
        return chip_note(m, chip)
    why = c.get("note") if cell == "Supported" else (c.get("reason") or c.get("note"))
    speed = measured_speed(m, chip) if cell == "Supported" else ""
    return cell + (f": {why}" if why and cell != "Not tested yet" else "") + (f". {speed}" if speed else "")


def measured_speed(m, chip):
    """A passing chips row's measured decode speed, "76.73 tokens/s for 1 request, 470.4 tokens/s for 8" (its measured
    block); "" when it has none. A chip we ran on is supported with its measured speed."""
    got = (m["chips"].get(chip) or {}).get("measured") or {}
    one, eight = got.get("decode_tok_s_1_user"), got.get("decode_tok_s_8_users")
    if not isinstance(one, (int, float)):
        return ""
    return f"{one:g} tokens/s for 1 request" + (f", {eight:g} tokens/s for 8" if isinstance(eight, (int, float)) else "")


def starts_on(m):
    """["CPU", "TPU v5e-4", "TPU v6e-1", ...]: every chip the model starts on, the package catalog's list (catalog.chips_run,
    behind docs/supported-models.md and `mashbox models`): CPU when its cpu row passes, then every passing chips row's
    host in CHIPS order, each once. A chip it was only measured on keeps its note in its own column."""
    out = ["CPU"] if runs_cpu(m) else []
    for c in CHIPS:
        r = m["chips"].get(c) or {}
        if r.get("status") == "passing" and f"TPU {r.get('accelerator') or c}" not in out:
            out.append(f"TPU {r.get('accelerator') or c}")
    return out


def chips_block(reg):
    """Which chip each model runs on (the FAQ's and the chips page's table), from the models.json chips rows. Released
    models only, as every other view."""
    rows = [[m["name"], ", ".join(starts_on(m))] + [chip_text(m, c) for c in CHIPS] for m in live(reg)]
    return table(["Model", "Runs on"] + [CHIP_LABEL[c] for c in CHIPS], rows)


@_mashbox_as_found
def docs_views(reg, repo, problems):
    """Generated docs files and blocks: {relative path: text}."""
    views = {}
    mods = _package(repo, problems)
    if mods is None:
        return views
    pp = (repo / "pyproject.toml").read_text()
    if re.search(r'^version\s*=\s*"', pp, re.M):
        problems.add("pyproject.toml", "the version is written here; it has one source, models.json "
                     '(dynamic = ["version"] and version = {attr = "mashbox.__version__"})')
    elif not (re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', pp, re.M)
              and re.search(r'^version\s*=\s*\{\s*attr\s*=\s*"mashbox\.__version__"\s*\}', pp, re.M)):
        problems.add("pyproject.toml", 'the version must be dynamic = ["version"] with version = {attr = "mashbox.__version__"}')
    versions = {"models.json": reg["version"], "src/mashbox/__init__.py": mods[""].__version__}
    if len(set(versions.values())) != 1:
        problems.add("version", "versions differ: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    O = mods[".runtime.openapi"]
    views["docs/openapi.json"] = O.spec_text(version=reg["version"])
    ex = repo / "docs" / "api-examples.json"  # real answers per route (tools/record_api_examples.py on a chip)
    views["docs/server-api.md"] = O.markdown(version=reg["version"], recorded=json.loads(ex.read_text()) if ex.exists() else None)
    views["docs/cli.md"] = cli_page(mods)
    by_cmd = {cmd[len("mashbox "):]: p for cmd, p, _ in cli_parsers(mods)}
    cloud = cloud_data(repo, problems)
    for page in doc_pages(repo):
        rel = page.relative_to(repo).as_posix()
        if rel in views:
            continue
        text = page.read_text()

        def cli_sub(mt):
            p = by_cmd.get(mt.group(2).strip())
            if p is None:
                try:
                    p = model_parser(mods, mt.group(2).strip())
                except Exception as e:  # a model app that will not import stdlib-only, or a form its parser lacks
                    problems.add(rel, f"cli block {mt.group(2)!r}: {type(e).__name__}: {e}")
                    return mt.group(0)
            if p is None:
                problems.add(rel, f"cli block names an unknown command: {mt.group(2)!r}")
                return mt.group(0)
            return f"{mt.group(1)}\n{options_table(p)}\n{mt.group(4)}"
        new = CLI_RE.sub(cli_sub, text)
        new = API_RE.sub(lambda mt: f"{mt.group(1)}\n{api_block(mods, rel)}\n{mt.group(3)}", new)
        new = CHIPS_RE.sub(lambda mt: f"{mt.group(1)}\n{chips_block(reg)}\n{mt.group(3)}", new)
        C = mods[".catalog"]
        new = CATALOG_RE.sub(lambda mt: f"{mt.group(1)}\n{_lists().render_markdown(reg)}\n{mt.group(3)}", new)
        new = ARCHS_RE.sub(lambda mt: f"{mt.group(1)}\n{C.archs_block(reg)}\n{mt.group(3)}", new)
        if CLOUD_RE.search(text):
            if cloud is None:
                problems.add(rel, f"cloud blocks need {CLOUD.as_posix()}")
            else:
                new = CLOUD_RE.sub(lambda mt: f"{mt.group(1)}\n{cloud_block(mt.group(2), cloud, reg)}\n{mt.group(4)}", new)
        if new != text or any(rx.search(text) for rx in (CLI_RE, API_RE, CHIPS_RE, CLOUD_RE, CATALOG_RE, ARCHS_RE)):
            views[rel] = new
    return views


# ---------------------------------------------------------------- cloud prices and memory bandwidth (docs/tpu-vs-gpu.md)
# One dated source, docs/data/cloud-prices.json: Google Cloud list prices (column, region, source page, read UTC) and the
# vendors' memory bandwidth. Pages carry <!-- cloud:start NAME --> / <!-- cloud:end --> blocks rendered from it and from
# models.json (measured speed, weights size). T0 fails when a block differs from its render, a price was read more than
# max_age_days ago, or models.json prices.per_chip_hour differs from the on-demand price in the file.

CLOUD = Path("docs") / "data" / "cloud-prices.json"
CLOUD_RE = re.compile(r"(<!-- cloud:start ([a-z-]+) -->)(.*?)(<!-- cloud:end -->)", re.S)
CLOUD_BLOCK_RE = re.compile(r"<!-- cloud:start.*?<!-- cloud:end -->", re.S)
UTC_MIN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}Z$")
CENT = Decimal("0.01")


def _usd(d):
    return f"${d.quantize(CENT, ROUND_HALF_UP):,}"


def _dec(d, places=2):  # 1.60 -> "1.6", 1.956 -> "1.96"
    return f"{d.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP):f}".rstrip("0").rstrip(".")


def cloud_data(repo, problems):
    f = repo / CLOUD
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except ValueError as e:
        problems.add(CLOUD.as_posix(), f"not JSON: {e}")
        return None


def _unit_price(d, hw, column):
    """USD per hour for one chip or GPU in a price column (a VM price divided by its units), rounded to the cent."""
    p = next((p for p in d["prices"] if p["hardware"] == hw and p["column"] == column), None)
    if p is None:
        raise KeyError(f"no {hw} price in column {column!r}")
    return (Decimal(p["usd_per_hour"]) / p.get("units_per_vm", 1)).quantize(CENT, ROUND_HALF_UP), p


def _first_num(s):
    m = NUM_RE.search(s)
    return Decimal(m.group(0).replace(",", ""))


def cloud_rows(d, reg):
    """The page rows with their numbers: price per hour, memory, bandwidth, the one-user ceiling for the model."""
    m = next(x for x in reg["models"] if x["id"] == d["ceiling"]["model"])
    wf = next(f for f in m["files"] if f.get("kind") == "weights" and f.get("default"))
    gb = Decimal(str(wf["gb"]))
    rows = []
    for r in d["rows"]:
        hw = d["hardware"][r["hardware"]]
        unit, src = _unit_price(d, r["hardware"], d["options"]["on_demand"][r["hardware"]])
        bw = hw["bandwidth_gb_s"] * r["count"]
        exact = Decimal(bw) / gb  # cost per 1M uses this, so one chip and four chips (same GB/s per $) cost the same
        rows.append(dict(r, hw=hw, src=src, unit=unit, price=unit * r["count"], memory=hw["memory_gb"] * r["count"],
                         bw=bw, exact=exact, ceiling=exact.quantize(Decimal(1), ROUND_HALF_UP)))
    return m, wf, rows


def cloud_block(name, d, reg):
    m, wf, rows = cloud_rows(d, reg)
    src = {p["hardware"]: p["source"] for p in d["prices"]}
    if name == "prices":
        def listed(r):
            s = r["src"]
            if s.get("vm"):
                return f"{_usd(Decimal(s['usd_per_hour']))} per hour for {s['vm']} ({s['units_per_vm']} GPUs), divided by {s['units_per_vm']}"
            return (f"{r['count']} x " if r["count"] > 1 else "") + f"{_usd(Decimal(s['usd_per_hour']))} per {s['per']}"
        return (f"Google Cloud on-demand list prices in USD, from Google's [Cloud TPU pricing]({src['v6e']}) "
                f"and [GPU pricing]({src['h100']}) pages. Each price includes the host machine.\n\n"
                + table(["Hardware", "Memory", "Per hour", "Region", "As Google lists it"],
                        [[r["label"], f"{r['memory']} GB", _usd(r["price"]), r["src"]["region"], listed(r)] for r in rows]))
    if name == "options":
        one = [r for r in rows if r["count"] == 1]
        cols = list(d["options"].values())
        return (f"The same Google Cloud pages, per hour for one chip or one GPU:\n\n"
                + table(["Per hour"] + [c["label"] for c in cols],
                        [[r["label"]] + [_usd(_unit_price(d, r["hardware"], c[r["hardware"]])[0]) for c in cols] for r in one]))
    if name == "bandwidth":
        def spec(r):
            hw = r["hw"]
            maker = "Google" if hw["spec_source"].startswith("https://cloud.google.com/") else "NVIDIA"
            return f"[{maker}]({hw['spec_source']})" if r["count"] == 1 else f"{r['count']} x the chip"
        return table(["Hardware", "Memory", "Memory bandwidth", "Bandwidth per $1 an hour, on demand", "Spec"],
                     [[r["label"], f"{r['memory']} GB", f"{r['bw']:,} GB/s",
                       f"{(Decimal(r['bw']) / r['price']).quantize(Decimal(1), ROUND_HALF_UP):,} GB/s", spec(r)] for r in rows])
    if name == "pair":
        a, b = rows[1], rows[2]
        return (f"For about the same price per hour ({_usd(a['price'])} and {_usd(b['price'])}), a {a['label'].split(' (')[0]} "
                f"has {_dec(Decimal(a['memory']) / b['memory'])}x the memory ({a['memory']} GB and {b['memory']} GB) and "
                f"{_dec(Decimal(a['bw']) / b['bw'])}x the memory bandwidth ({a['bw']:,} GB/s and {b['bw']:,} GB/s) of one H100.")
    meas = []
    for x in d["measured"]:
        mm = next(y for y in reg["models"] if y["id"] == x["model"])
        res = mm["results"][x["result"]]
        r = next(r for r in rows if r["hardware"] == x["hardware"] and r["count"] == x["count"])
        meas.append((r, x["users"], _first_num((res.get("short") or {}).get("ours") or res["value"])))
    total = lambda users, n: f"{n:,}" + (" total" if users != "1" else "")
    if name == "speed":
        head = (f"{m['name']} ({d['ceiling']['params']}), our default weights file ({wf['gb']} GB, "
                f"{_dec(Decimal(str(wf['bpw'])))} bits per weight), output tokens per second:\n\n")
        body = [[r["label"], _usd(r["price"]), users, total(users, n), "measured"] for r, users, n in meas]
        body += [[r["label"], _usd(r["price"]), "1", f"~{r['ceiling']}", f"ceiling: {r['bw']:,} GB/s / {wf['gb']} GB"] for r in rows]
        return head + table(["Hardware", "Per hour", "Users", "Tokens per second", "Kind"], body)
    if name == "cost":
        per_m = lambda price, n: _usd(price / (n * 3600) * 1000000)
        body = sorted([[r["label"], users, f"{total(users, n)}, measured", per_m(r["price"], n)] for r, users, n in meas],
                      key=lambda row: Decimal(row[3][1:].replace(",", "")))
        body += [[r["label"], "1", f"~{r['ceiling']}, ceiling", per_m(r["price"], r["exact"])] for r in rows]
        return (f"{m['name']}, estimated as price per hour / output tokens per hour, at the Google Cloud on-demand list "
                f"prices above:\n\n" + table(["Hardware", "Users", "Tokens per second", "USD per 1M output tokens"], body))
    raise KeyError(f"unknown cloud block {name!r}")


def check_cloud(reg, repo, problems, today=None):
    """Prices are at most max_age_days old (models.json prices.read too) and models.json prices.per_chip_hour equals the
    on-demand price in docs/data/cloud-prices.json for each chip."""
    import datetime
    d = cloud_data(repo, problems)
    if d is None:
        return
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    age = lambda iso: (today - datetime.date.fromisoformat(iso)).days
    where, limit = CLOUD.as_posix(), d.get("max_age_days", 45)
    for p in d.get("prices", []):
        m = UTC_MIN_RE.match(p.get("read_utc", ""))
        if not m or not p.get("source", "").startswith("https://") or p.get("hardware") not in d.get("hardware", {}):
            problems.add(where, f"price {p.get('hardware')}/{p.get('column')}: needs hardware, read_utc YYYY-MM-DDTHH:MMZ and an https source")
        elif age(m.group(1)) > limit:
            problems.add(where, f"price {p['hardware']}/{p['column']} was read {age(m.group(1))} days ago (more than "
                                f"{limit}): read {p['source']} again and update the price and read_utc")
    if age(reg["prices"]["read"]) > limit:
        problems.add("models.json", f"prices.read is {age(reg['prices']['read'])} days old (more than {limit}): read the "
                                    f"price again (and {where})")
    for chip, v in reg["prices"]["per_chip_hour"].items():
        try:
            ours = _unit_price(d, chip, d["options"]["on_demand"].get(chip, "On Demand"))[0]
        except KeyError:
            problems.add(where, f"no on-demand price for {chip} (models.json prices.per_chip_hour has one)")
            continue
        if Decimal(v) != ours:
            problems.add("models.json", f"prices.per_chip_hour {chip} {v} differs from {where} ({ours})")


MARK_RE = re.compile(r"(<!-- r:([a-z0-9-]+)/([a-z0-9][a-z0-9_.-]*) -->)(.*?)(<!-- /r -->)", re.S)  # its own name: NUM_RE is the number guard


def figure_text(reg, owner, key):
    """The text a number marker shows: r:MODEL/N is results[N] of that model (its short "ours" cell), r:MODEL/NAME is
    its figures[NAME], r:MODEL/chip-v5e its chips row (status, why), r:MODEL/arg-ctx the value after --ctx in
    its cloud.agents.run_args, r:cloud/NAME is a top-level figure (chip waits, disk sizes: not one model's). None if
    absent."""
    if owner == "cloud":
        f = (reg.get("figures") or {}).get(key)
        return f and f["text"]
    m = next((x for x in reg["models"] if x["id"] == owner), None)
    if m is None:
        return None
    if key.startswith("chip-") and (key[5:] in CHIPS or key[5:] == "cpu"):   # chip-cpu: a CPU model's chips.cpu row
        return chip_text(m, key[5:])
    if key.startswith("arg-"):   # r:MODEL/arg-ctx: the value after --ctx in cloud.agents.run_args ("32,768")
        args = ((m.get("cloud") or {}).get("agents") or {}).get("run_args") or []
        flag = "--" + key[4:]
        v = args[args.index(flag) + 1] if flag in args[:-1] else None
        return v and (f"{int(v):,}" if v.isdigit() else v)
    if key.isdigit():
        res = m.get("results") or []
        return ((res[int(key)].get("short") or {}).get("ours") or res[int(key)]["value"]) if int(key) < len(res) else None
    f = (m.get("figures") or {}).get(key)
    return f and f["text"]


NUMBERS = Path("internal") / "numbers.json"   # the maintainers' number map: not in the public tree
PRIVATE_DIRS = ("addons/",)   # private app add-ons: held from the public tree, never a public page
_BEFORE = 40                                  # characters of the page kept before each number to find it again


def load_numbers(repo):
    """The private number map ({"figures": {owner: {key: figure}}, "pages": {page: [place]}}), or None without one."""
    f = repo / NUMBERS
    return json.loads(f.read_text()) if f.is_file() else None


def merge_figures(reg, nmap, problems=None):
    """The named figures live in the private map; the checks read them as models.json's. A figure still in models.json
    (a branch from before the map) is read too, and fails the check until render --write moves it."""
    figs = nmap.get("figures") or {}
    for owner, row in [("cloud", reg)] + [(m["id"], m) for m in reg["models"]]:
        inline = row.get("figures") or {}
        if inline and problems is not None:
            problems.add("models.json", f"{owner}: {len(inline)} figures belong in {NUMBERS.as_posix()} (run render --write)")
        row["figures"] = {**(figs.get(owner) or {}), **inline}


def unmark(text):
    """(text without number markers, [place]); a place is {"r": "MODEL/KEY", "text": shown text, "before": the text just
    before it on its line, up to _BEFORE characters, with the line start as a newline}."""
    out, spots, pos, plain = [], [], 0, ""
    for mt in MARK_RE.finditer(text):
        out.append(text[pos:mt.start()])
        plain = "".join(out)
        tail = plain[-_BEFORE:]
        k = tail.rfind("\n")
        spots.append({"r": f"{mt.group(2)}/{mt.group(3)}", "text": mt.group(4), "before": tail[k:] if k >= 0 else tail})
        out.append(mt.group(4))
        pos = mt.end()
    out.append(text[pos:])
    return "".join(out), spots


def mark_up(text, spots, where="", problems=None):
    """The page with its number markers back, from its places in the private map (in page order). A marker already in
    the page (written by hand, or put back to edit the words around it) wins over the map's place for that number."""
    plain, inline, pos = unmark(text)[0], [], 0
    for mt in MARK_RE.finditer(text):           # inline markers, at their offsets in the plain text
        pos += mt.start() - (inline[-1][4] if inline else 0)
        inline.append((pos, pos + len(mt.group(4)), f"{mt.group(2)}/{mt.group(3)}", mt.group(4), mt.end()))
        pos += len(mt.group(4))
    found, pos = [], 0
    for s in spots:
        b = len(s["before"])   # the words before a number can hold the number before it ("4.0 to 4.7 s")
        i = plain.find(s["before"] + s["text"], max(0, pos - b))
        while i >= 0 and i + b < pos:
            i = plain.find(s["before"] + s["text"], i + 1)
        if i < 0:
            if not inline and problems is not None:
                problems.add(where, f"number r:{s['r']} {s['text']!r} after {s['before'][-30:]!r} is not on the page any "
                                    f"more: put its marker back in the page (<!-- r:{s['r']} -->...<!-- /r -->) and run "
                                    f"render --write, or remove the number")
            continue
        found.append((i + b, i + b + len(s["text"]), s["r"], s["text"]))
        pos = i + b + len(s["text"])
    keep = [f for f in found if not any(x < f[1] and f[0] < y for x, y, *_ in inline)]
    out, pos = [], 0
    for x, y, r, shown, *_ in sorted([i[:4] for i in inline] + keep):
        out += [plain[pos:x], f"<!-- r:{r} -->{shown}<!-- /r -->"]
        pos = y
    out.append(plain[pos:])
    return "".join(out)


def _page(repo, f, places):
    """A page as the number checks read it: its markers back in place."""
    rel = f.relative_to(repo).as_posix()
    return mark_up(f.read_text(), (places or {}).get(rel, []))


def fill_numbers(reg, text, where, problems):
    """Every <!-- r:MODEL/KEY -->TEXT<!-- /r --> gets its figure from models.json (results by number, figures by name)."""
    def sub(mt):
        ours = figure_text(reg, mt.group(2), mt.group(3))
        if ours is None:
            problems.add(where, f"number marker r:{mt.group(2)}/{mt.group(3)}: no such result or figure in models.json")
            return mt.group(0)
        short = ours if mt.group(3).startswith("chip-") else ours.split(" (")[0]   # a chip status keeps its condition
        keep = mt.group(4) if mt.group(4) in (ours, short) else short
        return mt.group(1) + keep + mt.group(5)
    return MARK_RE.sub(sub, text)


# ---------------------------------------------------------------- measured numbers typed in prose (M4c)
# Every measured number in a hand-written page sits in a number marker, so it comes from models.json; numbers that are
# not measurements (Google's chip specs, settings, sizes in an example) are listed by page and text in
# tools/numbers_allowed.txt with the reason. A new or changed number anywhere else fails T0.

_NUM = r"\d[\d,]*(?:\.\d+)?"
_UNIT = (r"(?:[\s-]?(?:tokens/sec|tok/sec|tokens/s|tok/s|tokens a second|tokens per second|tokens?|frames? (?:a|per) second|"
         r"fps|requests? (?:a|per) second|times|ms|milliseconds?|secs?|s|seconds?|minutes?|mins?|hours?|hrs?|days?|"
         r"[KMGT]i?B/s|[KMGT]i?Bps|[KMG]bit/s|[KMG]bps|TB|TiB|GB|GiB|MB|MiB|KB|TFLOPS|TFLOPs|TOPs|%|percent|bpw|bits per weight|"
         r"fold)"
         r"|x|\s?\u00d7)")
# a number with its unit (also "5.2-second", "767-token"), a range ("6-7 s", "6 to 7 s") or a range that ends in a marker
# ("4.0 to <!-- r:... -->"); a bare number alone in a table cell (its unit is in the header); a dollar amount that is
# not a price an hour or a day (check_prose checks those against models.json prices); a number in words before a unit
_CELL = rf"\$?{_NUM}(?: ?(?:-|\u2013|to) ?{_NUM})?(?=\s?[|(,;*/+])"
MEASURE_RE = re.compile(rf"(?<![\w.$/:-]){_NUM}(?:\s?(?:-|\u2013|to|and)\s?(?:{_NUM}{_UNIT}|\x00)|{_UNIT})(?![\w/])"
                        rf"|(?:(?<=\|)|(?<=\| )|(?<=\| ~)|(?<=\| about ))(?<![\w.]){_CELL}"   # a table cell that starts with a number
                        rf"|\b(?:seconds|secs|minutes|mins|hours|ms|milliseconds|tokens|tok/s|fps|GB|GiB|MB|MiB|percent)\)?"
                        rf"\s?[:=]\s?{_NUM}"                                    # the unit first: "in seconds: 5.2"
                        r"|\d(?=\x00)|(?<=\x00)\d"                              # a digit typed against a marker
                        r"|(?<![\w.])\$\d[\d,]*(?:\.\d+)?[MKk]?(?![\w.]*\d)(?!\s(?:an|a|per)\s(?:chip-)?hour|/hour|\sa day)")
_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
          "eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million")
WORD_NUM_RE = re.compile(rf"\b(?:{_WORDS})(?:[\s-](?:and a half|{_WORDS}))*[\s-](?:seconds?|minutes?|hours?|milliseconds?|"
                         rf"times|tokens|frames|percent|gigabytes|megabytes)\b", re.I)
CODE_LANGS = r"(?:bash|sh|shell|zsh|json|jsonc|toml|python|py|yaml|yml|js|javascript|ts|html|css|ini|diff|powershell|ps1)"
STRIP_RES = (re.compile(rf"^```{CODE_LANGS}\b[^\n]*\n.*?^```", re.S | re.M),   # code; output blocks (text, console, plain) are read
             re.compile(r"<!-- (models|cli|api|chips|catalog|archs):start.*?<!-- (models|cli|api|chips|catalog|archs):end -->", re.S),
             CLOUD_BLOCK_RE,  # rendered from docs/data/cloud-prices.json and checked by render-and-diff (check_cloud)
             re.compile(r"\]\([^)\s]*\)"), re.compile(r"^```.*$", re.M))  # inline code is read: a number there is public too
TAG_RE = re.compile(r"<[^>]*>")   # a tag is dropped, the words it shows or exposes read (tag_words)
# alt and title text, ARIA text, a form value or placeholder, a meta content, data-* (CSS attr() puts it on screen)
TAG_WORDS_RE = re.compile(r"(?<![\w-])(alt|title|aria-[\w-]+|placeholder|value|content|label|"
                          r"data-[\w-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+))", re.I)


SHOWN_VALUE_RE = re.compile(r"(?is)<input\b(?![^>]*\btype\s*=\s*[\"']?(?:range|hidden|checkbox|radio|color|file|image|date|"
                            r"time|datetime-local|month|week)\b)")


COMPONENT_RE = re.compile(r"<[A-Z][a-z]\w*\b")   # an MDX (JSX) component: <Stat value="..." />
PROP_RE = re.compile(r"""(?<![\w-])[\w:.-]+\s*=\s*(?:"([^"]*)"|'([^']*)'|\{\s*["'`]([^"'`]*)["'`]\s*\})""")


def tag_words(tag):
    """The words a tag puts in front of a reader, quoted or not: what the number, chip and company checks read. A value
    is read where the browser prints it: an <input> box or button (a slider's or checkbox's value is not text). An MDX
    component shows its props as it likes: every string prop is read."""
    if COMPONENT_RE.match(tag):
        return " " + " ".join(a or b or c for a, b, c in PROP_RE.findall(tag)) + " "
    shown = SHOWN_VALUE_RE.match(tag)
    return " " + " ".join(a or b or c for k, a, b, c in TAG_WORDS_RE.findall(tag) if k.lower() != "value" or shown) + " "


CHIP_WORD_RE = re.compile(r"\b(?:TPU )?v[56][ep]\b")
STATUS_WORD_RE = re.compile(r"(?i)\b(?:not supported|not yet|supported|passing|failing|too little memory|do(?:es)? not fit|"
                            r"needs? more than)\b")
RUNS_ON_RE = re.compile(r"(?i)(?<!not )(?<!n't )\b(?:runs?|works?|fits?)(?: fine| well)? on (?:an? |one |the )?(?:single )?(?:TPU )?(v[56][ep](?:-\d+)?)\b")
ONLY_ON_RE = re.compile(r"(?i)(?<!not )\bonly (?:(?:runs?|works?|fits?) )?on (?:an? |one |the )?(?:single )?(?:TPU )?"
                        r"(v[56][ep](?:-\d+)?)\b|\b(?:runs?|works?|fits?) on (?:an? |one |the )?(?:single )?(?:TPU )?"
                        r"(v[56][ep](?:-\d+)?) only\b")   # read on whitespace-joined text
CHIP_TOKEN_RE = re.compile(r"\bv[56][ep](?:-\d+)?\b")
ARG_RE = re.compile(r"(--(?:ctx|slots))[ =](\d+)|(model_context_window\s*=\s*|\"contextWindow\":\s*)(\d+)")
ALLOWED = Path("tools") / "numbers_allowed.txt"


CARD_RE = re.compile(r"^hf/[^/]+/README\.md$")   # rendered from models.json and templates/hf (render-and-diff)


def number_pages(reg, repo):
    """Every public page but the HF cards (rendered from models.json) and the page templates. docs/cli.md and
    docs/server-api.md are read too: a number in a --help text or a route description lands there. A coming-soon
    model's page is read like the others: it is public from its first push. Any other file under hf/ is read too."""
    return [f for f in _prose(repo) if not CARD_RE.match(r := f.relative_to(repo).as_posix()) and not r.startswith("docs/_templates/")]


def ui_pages(repo):
    """The text people read in the package, the Mac app and the docs' own pictures and pages: every .html, .svg and .js
    file under src/mashbox, apps/ and docs/."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "--",
                              "src/mashbox", "apps", "docs"], capture_output=True, text=True, check=True).stdout.splitlines()
        files = [repo / x for x in out]
    except (OSError, subprocess.CalledProcessError):
        files = [f for d in ("src/mashbox", "apps", "docs") for f in (repo / d).rglob("*")]
    return sorted({f for f in files if f.suffix in (".html", ".svg", ".js") and f.is_file()}, key=lambda f: f.relative_to(repo).as_posix())


JS_STR_RE = re.compile(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"|`((?:[^`\\]|\\.)*)`")


def ui_text(f, text=None):
    """What a person can read in a UI file: an HTML or SVG page's text and the words its tags show (tag_words), and the
    string literals of its scripts (and of a .js file). Styles and code are left out; "$1" in a regex replacement too.
    text: the file with its number markers put back (the map's places), else the file as it is."""
    t, js = (f.read_text() if text is None else text), ""
    if f.suffix in (".html", ".svg"):   # an SVG's text, title and desc are its words
        t = re.sub(r"(?is)<style\b.*?</style>", " ", t)
        js = "\n".join(re.findall(r"(?is)<script\b[^>]*>(.*?)</script>", t))
        t = re.sub(r"(?is)<script\b.*?</script>", " ", t)
        t = re.sub(r"<(?!!-- (?:r:|/r ))[^>]*>",   # number markers stay (render fills them), other tags give their words
                   lambda mt: tag_words(mt.group(0)), t)
    else:
        t, js = "", t
    js = re.sub(r"(?m)^\s*//.*$", " ", js)
    lits = [a or b or c for a, b, c in JS_STR_RE.findall(js)]
    return re.sub(r"\$\d+\b(?![.,]\d)", " ", t + "\n" + "\n".join(lits))


def guard_pages(reg, repo, places=None):
    """(path, text) of everything the number guard reads: the public pages, the UI text and the PyPI summary line,
    with their number markers back from the map (places)."""
    out = [(f.relative_to(repo).as_posix(), _page(repo, f, places)) for f in number_pages(reg, repo)]
    seen = set()
    for f in ui_pages(repo):   # a byte-identical copy (the app's page in the package and in apps/) is read once
        raw = _page(repo, f, places)
        if raw not in seen:
            seen.add(raw)
            out.append((f.relative_to(repo).as_posix(), ui_text(f, raw)))
    pp = repo / "pyproject.toml"    # its description is the PyPI page's first line
    if pp.exists():
        out.append(("pyproject.toml", "\n".join(re.findall(r'^description\s*=\s*"(.*)"', pp.read_text(), re.M))))
    return out


def _guard_text(text):
    """A page as the number guard reads it: code blocks out (output blocks stay), markers as \x00, generated blocks
    and link targets out, tags as their alt/title text, bold marks out, whitespace joined."""
    keep = lambda mt: " ".join(re.findall(r"(?m)^[ \t]*(?:#|//)[ \t]+(.*)$", mt.group(0))) + " "   # comments are prose
    text = MARK_RE.sub(lambda mt: "\x00\x01" if mt.group(3).startswith("chip-") else "\x00",   # a marker stays a place
                       STRIP_RES[0].sub(keep, text))                                      # holder: "4.0 to <marker>"
    for rx in STRIP_RES[1:]:
        text = rx.sub(" ", text)
    text = TAG_RE.sub(lambda mt: tag_words(mt.group(0)), text)
    text = re.sub(r"\u00a0", " ", html.unescape(text)).replace("**", "").replace("__", "")
    return text


@_mashbox_as_found
def _run_arg_values(reg, repo):
    """{flag or config key: values a page may show}: models.json run_args, and the `mashbox run` defaults."""
    out = {"--ctx": set(), "--slots": set()}
    for m in reg["models"]:
        args = ((m.get("cloud") or {}).get("agents") or {}).get("run_args") or []
        for i, a in enumerate(args[:-1]):
            if a in out:
                out[a].add(args[i + 1])
    ctx = set(out["--ctx"])     # an agent's context setting (Codex, Pi) is the server's --ctx of run_args
    mods = _package(repo, Problems())
    if mods is not None:
        p = next((sp for n, sp, _ in _subparsers(mods[".cli"].build_parser()) if n == "run"), None)
        for a in (p._actions if p else []):
            for f in out:
                if f in a.option_strings and a.default not in (None, argparse.SUPPRESS):
                    out[f].add(str(a.default))
    out["ctx"] = ctx
    return out


def bad_chip_claims(m, piece):
    """The chips a sentence says model m runs on ("runs on a TPU v5e-8") while its models.json chips row says not
    supported, failing, or pending (nobody ran it there, so no run backs the sentence), or measured: a measured run
    exists there but this version's `mashbox run` does not start on it, so "fits on" holds and "runs on" does not."""
    out = []
    for hit in RUNS_ON_RE.finditer(piece):
        chip = hit.group(1)
        chip = chip if chip in CHIPS else chip.split("-")[0]   # v6e-4 is its own row; v6e-1, v5e-8: the chip's
        st = (m["chips"].get(chip) or {}).get("status")
        if st == "measured" and not hit.group(0).lower().startswith("fit"):
            out.append(chip)
        elif st in ("not supported", "failing", "pending"):
            out.append(chip)
    return out


def only_chip_claims(m, piece):
    """[(chip, why)] for each chip a sentence says model m runs ONLY on ("runs only on a TPU v6e-4", "only runs on",
    "works on a v6e-4 only") while m was measured on another chip (a results row, or a passing chips row) or has a chips
    row nobody has run it on yet: "only" then claims what no run showed. A chip covers its slices: "only on a TPU v6e"
    holds with results on v6e-1 and v6e-4; "only on a v6e-4" does not hold with a result on v6e-1."""
    out = []
    for said in (a or b for a, b in ONLY_ON_RE.findall(piece)):
        inside = lambda c: c == said or c.startswith(said + "-")
        row = said if said in CHIPS else said.split("-")[0]   # the chips row the chip belongs to, as bad_chip_claims
        why = [f"it was measured on {r['chip']}" for r in m.get("results") or []
               if any(not inside(c) for c in CHIP_TOKEN_RE.findall(r.get("chip") or ""))
               or (r.get("chip") or "").lower().startswith("cpu")]
        for key, c in (m.get("chips") or {}).items():
            acc = c.get("accelerator") or key   # a host of the row's own chip type (chip_row_problems)
            if key == row or (inside(acc) and acc.split("-")[0] == key.split("-")[0]):
                continue
            if c.get("status") in ("passing", "measured"):
                why.append(f"its {key} chips row is {c['status']}")
            elif c.get("status") == "pending":
                why.append(f"its {key} chips row is Not tested yet")
            elif key in CHIPS and never_run(m, key):
                why.append(f"its {key} chips row was never run")
        if why:
            out.append((said, "; ".join(dict.fromkeys(why))))
    return out


# a sentence that makes fitting one chip the condition to run on a chip type ("a model runs on v5e when its weights fit
# in 16 GB", "a model runs there when it fits"); read after row_words()
FIT_CONDITION_RE = re.compile(r"\b(?:when|if|once|as long as|unless)\b[^.;|]*\bfits?\b")
RUN_WORD_RE = re.compile(r"\b(?:runs?|running)\b")


def multi_chip_hosts(reg):
    """{chip type: the hosts of several chips a model passes on} from the released models' passing chips rows with a
    chip_count above 1: {"v5e": ["v5e-4"], "v6e": ["v6e-4"]}."""
    out = {}
    for m in live(reg):
        for key, c in (m.get("chips") or {}).items():
            if isinstance(c, dict) and c.get("status") == "passing" and int(c.get("chip_count") or 1) > 1:
                host = row_words(c.get("accelerator") or key)
                out.setdefault(host.split("-")[0], set()).add(host)
    return {t: sorted(h) for t, h in out.items()}


def chip_fit_problems(reg, line):
    """A page line that makes fitting one chip the condition to run on a chip type, while a model passes on a host of
    several chips of that type (Qwen3.8-27B on a v5e-4, split across 4 chips): the sentence names that host or the
    split, else it tells a reader the larger model does not run there. The chip type comes from the sentence, else from
    its line (a table row: "TPU v5e | ... A model runs there when it fits.")."""
    hosts, out = multi_chip_hosts(reg), []
    for piece in re.split(r"(?<=[.!?])\s+", line):
        words = row_words(piece)
        if not (FIT_CONDITION_RE.search(words) and RUN_WORD_RE.search(words)):
            continue
        kinds = {t.split("-")[0] for t in CHIP_TOKEN_RE.findall(words)} or \
                {t.split("-")[0] for t in CHIP_TOKEN_RE.findall(row_words(line))}
        for kind in sorted(kinds & set(hosts)):
            if not (any(re.search(rf"\b{re.escape(h)}\b", words) for h in hosts[kind]) or re.search(r"\bsplit|\bacross\b", words)):
                out.append(f"makes fitting one chip the condition to run on {kind} ({' '.join(piece.split())[:100]!r}), but "
                           f"a model passes on {', '.join(hosts[kind])} split across its chips: name the host or the split")
    return out


def sentences(text):
    """The sentences of a page with each paragraph, list item, table row and heading joined across its line breaks (a
    claim wrapped over two lines is read whole); a piece holding a chip marker (\x01) is left out."""
    out = []
    for block in re.split(r"\n\s*\n|\n(?=[ \t]*(?:[-*+|#>]|\d+[.)])\s)", text):
        out += [p for p in re.split(r"(?<=[.!?])\s+", " ".join(block.split())) if p and "\x01" not in p]
    return out


def docstrings(repo):
    """(path, text) of every docstring in the package (module, class, function): help() and API references show them."""
    import ast
    out = []
    for py in sorted((repo / "src" / "mashbox").rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except (SyntaxError, ValueError):
            continue   # a file that does not parse fails ruff and the tests
        for n in [tree] + [x for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
            d = ast.get_docstring(n, clean=True)
            if d:
                out.append((py.relative_to(repo).as_posix(), d))
    return out


def check_numbers(reg, repo, problems, places=None):
    """M4c: every measured number in hand-written prose is in a number marker or allowed by name; figures are
    checked against their result when they repeat one (figures[...]["from"]). places: the private map's pages."""
    for owner, figs in [("cloud", reg.get("figures") or {})] + [(m["id"], m.get("figures") or {}) for m in reg["models"]]:
        for key, f in figs.items():
            if not re.fullmatch(r"[a-z][a-z0-9_.-]*", key) or not f.get("text") or not f.get("what"):
                problems.add("models.json", f"figure {owner}/{key}: needs a lower-case name, text and what")
            elif "from" in f:
                m = next(x for x in reg["models"] if x["id"] == owner)
                bad = unbacked(f["text"], result_sources(m["results"][f["from"]]))
                if bad:
                    problems.add("models.json", f"figure {owner}/{key} {f['text']!r}: {bad} not in result {f['from']}")
    allowed, used = {}, set()
    f = repo / ALLOWED
    for i, line in enumerate(f.read_text().splitlines() if f.exists() else [], 1):
        if line.strip() and not line.startswith("#"):
            parts = line.split("|")
            page, phrase, why = parts[0].strip(), " ".join("|".join(parts[1:-1]).split()), parts[-1].strip()
            if len(parts) < 3 or not why or page == "*" or len(phrase.split()) < 3:
                problems.add(ALLOWED.as_posix(), f"line {i}: want 'page | the number with the words around it, as "
                                                 f"written (3 words or more) | why it is not a measurement' (no * lines)")
                continue
            allowed[(page, phrase)] = i
    args = _run_arg_values(reg, repo)
    names = {w: m for m in reg["models"] for w in (m["name"], m["id"], m["name"].split()[0])}
    name_re = re.compile("|".join(re.escape(w) for w in sorted(names, key=len, reverse=True)))
    for rel, raw in guard_pages(reg, repo, places):
        text = _guard_text(raw)
        own = next((m for m in reg["models"] if rel == f"docs/models/{m['id']}.md"), None)
        status = []   # a sentence (or table row) that gives a model's chip status: from the chips row (marker) only
        for line in text.splitlines():
            for why in chip_fit_problems(reg, line if "\x01" not in line else ""):   # \x01: a chips-row marker
                problems.add(rel, why)
            for piece in re.split(r"(?<=[.!?])\s+", line):
                said = {names[w]["id"]: names[w] for w in name_re.findall(piece)} or ({own["id"]: own} if own else {})
                if CHIP_WORD_RE.search(piece) and STATUS_WORD_RE.search(piece) and "\x01" not in piece and said:
                    status.append(" ".join(piece.split()))   # \x01: a chip marker
                for m in said.values():   # "Hy-Nomas runs on a TPU v5e-8" while its v5e row says not supported
                    for chip in bad_chip_claims(m, piece) if "\x01" not in line else []:
                        problems.add(rel, f"says {m['name']} runs on {chip} ({' '.join(piece.split())[:100]!r}), "
                                          f"but its models.json chips row is {chip_text(m, chip)!r}")
        for piece in sentences(text):   # "Hy-Nomas runs only on a TPU v6e-4" while it was measured on one v6e chip
            for m in ({names[w]["id"]: names[w] for w in name_re.findall(piece)} or ({own["id"]: own} if own else {})).values():
                for chip, why in only_chip_claims(m, piece):
                    problems.add(rel, f"says {m['name']} runs only on {chip} ({piece[:100]!r}), but {why}: say where it "
                                      f"runs (the chips rows), not that it runs nowhere else")
        text = " ".join(text.replace("\x01", "").split())
        spans = [(mt.start(), mt.end(), k) for k in allowed if k[0] == rel
                 for mt in re.finditer(re.escape(k[1]).replace("<marker>", "\x00"), text)]

        def allow(a, b):
            hit = next((k for x, y, k in spans if x <= a and b <= y), None)
            if hit:
                used.add(hit)
            return hit
        hits = [(mt.start(), mt.end(), mt.group(0)) for rx in (MEASURE_RE, WORD_NUM_RE) for mt in rx.finditer(text)]
        for a, b, num in sorted(hits):
            if num.startswith("$") and text[max(0, a - 13):a] == "for a day is ":
                continue   # "a day is $64.80": check_prose
            if num.isdigit() and text[max(0, a - 4):a] == "| | " and (f"| | {int(num) + 1} |" in text[b:b + 4000]
                                                                    or f"| | {int(num) - 1} |" in text[max(0, a - 4000):a]):
                continue   # a table's first column counting rows (turn 1, turn 2): the row's number, not a measurement
            if rel in ("docs/cli.md", "docs/server-api.md") and text[a - 2:a] == "| " and text[b:b + 2] == " |":
                continue   # a default in a table rendered from the parsers or the spec (their prose is read)
            if num.startswith("$") and re.search(r"\d+ minutes is about $", text[max(0, a - 25):a]):
                continue   # "30 minutes is about $1.35": check_prose
            if not allow(a, b):
                around = text[max(0, a - 30):b + 20].replace("\x00", "<marker>")
                problems.add(rel, f"measured number {num!r} typed by hand (...{around}...): put it in a number "
                                  f"marker (<!-- r:MODEL/KEY -->...<!-- /r -->, value in models.json) or, if it is not a "
                                  f"measurement, on its page in {ALLOWED.as_posix()} with the words around it")
        for s in status:
            a = text.find(s)
            if not allow(a, a + len(s)):
                problems.add(rel, f"chip status typed by hand: {s[:120]!r}: use the chips row "
                                  f"(<!-- r:MODEL/chip-v5e -->...<!-- /r -->), so it follows models.json")
        # --ctx/--slots and agent context settings: models.json run_args or the `mashbox run` default (n11b)
        for where, body, spanned in [("text", text, True)] + [("code", b, False) for b in FENCE_RE.findall(raw)]:
            for mt in ARG_RE.finditer(body):
                key, val = (mt.group(1), mt.group(2)) if mt.group(1) else ("ctx", mt.group(4))
                if val in args[key] or (spanned and allow(mt.start(), mt.end())):
                    continue
                problems.add(rel, f"{mt.group(0)!r} in {where}: not models.json cloud.agents.run_args "
                                  f"({sorted(args[key])}) or the mashbox run default; change it with run_args, or list the "
                                  f"example in {ALLOWED.as_posix()}")
    for rel, doc in docstrings(repo):   # a docstring (help(), API references) makes no chip claim against the chips rows
        own = next((m for m in reg["models"] if rel.startswith(f"src/mashbox/models/{m['id'].replace('-', '_')}/")), None)
        for piece in re.split(r"(?<=[.!?])\s+", " ".join(doc.split())):
            for m in ({names[w]["id"]: names[w] for w in name_re.findall(piece)} or ({own["id"]: own} if own else {})).values():
                for chip in bad_chip_claims(m, piece):
                    problems.add(rel, f"docstring says {m['name']} runs on {chip} ({piece[:100]!r}), "
                                      f"but its models.json chips row is {chip_text(m, chip)!r}")
                for chip, why in only_chip_claims(m, piece):
                    problems.add(rel, f"docstring says {m['name']} runs only on {chip} ({piece[:100]!r}), but {why}")
    which = re.compile(r"\b(first|later)\b", re.I)   # a result marker after "first start" must be a first-start result
    for f in number_pages(reg, repo):
        raw = _page(repo, f, places)
        for mt in MARK_RE.finditer(raw):
            m = next((x for x in reg["models"] if x["id"] == mt.group(2)), None)
            if m is None or not mt.group(3).isdigit() or int(mt.group(3)) >= len(m["results"]):
                continue
            r = m["results"][int(mt.group(3))]
            own = {w.lower() for w in which.findall((r.get("short") or {}).get("measure") or r["metric"])}
            said = {w.lower() for w in which.findall(re.split(r"[.;:(|]|<!--|-->", raw[max(0, mt.start() - 60):mt.start()])[-1])}
            words = set(re.findall(r"[a-z]{4,}", r["metric"].lower())) - {"with", "from", "that", "this", "time", "same"}
            line = raw[raw.rfind("\n", 0, mt.start()) + 1:mt.start()]   # prose (a table's column header names its measure)
            near = re.sub(r"<!--.*?-->", " ", raw[max(0, mt.start() - 160):mt.start()]).lower()
            if words and not line.lstrip().startswith("|") and not words & set(re.findall(r"[a-z]{4,}", near)):
                problems.add(f.relative_to(repo).as_posix(), f"marker r:{mt.group(2)}/{mt.group(3)} ({r['metric'][:60]}) "
                                                             f"has none of its words before it: is it the right result?")
            if own and said and not own & said:
                problems.add(f.relative_to(repo).as_posix(), f"marker r:{mt.group(2)}/{mt.group(3)} ({r['metric']}) follows "
                                                             f"words for the {'/'.join(sorted(said))} start: wrong result number?")
    refs = {(mt.group(2), mt.group(3)) for f in _prose(repo) + ui_pages(repo) for mt in MARK_RE.finditer(_page(repo, f, places))}
    for owner, figs in [("cloud", reg.get("figures") or {})] + [(m["id"], m.get("figures") or {}) for m in reg["models"]]:
        for key in figs:
            if (owner, key) not in refs:
                problems.add("models.json", f"figure {owner}/{key} is on no page (remove it)")
    for k, i in allowed.items():
        if k not in used:
            problems.add(ALLOWED.as_posix(), f"line {i}: {k[1]!r} holds no number on {k[0]} any more (remove the line)")


# ---------------------------------------------------------------- facts typed in prose (install line, prices, company line, release status)
# One value in models.json, every copy in the pages checked against it (as llama.cpp checks its generated docs).

FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)
UV_RE = re.compile(r"(?:python3? -m )?(?:uv tool install|uv pip install|uvx|pipx install|pip3? install)\b.*?(?=\s*(?:&&|'|`|$))", re.M)
PRICE_RE = re.compile(r"\$(\d+\.\d{2})(?: (?:an|a|per) (?:chip-)?hour\b|/hour\b)|\$(\d+\.\d{2}) a day\b|\bfor a day is \$(\d+\.\d{2})")
MINUTES_PRICE_RE = re.compile(r"\b(\d+) minutes is about \$(\d+\.\d{2})")
READ_RE = re.compile(r"\bread (\d{1,2} [A-Z][a-z]+ \d{4})")
COMPANY_RE = re.compile(r"\bMASHBOX\s+Labs(?:\s*[,(:]| (?:is|was|has|will|can|[a-z]+s)\b)|\b(?:we|us|team) at MASHBOX\b"
                        r"|(?-i:\bMASHBOX) (?:is|was|has been) (?:a|an|the|our)\b[^.]{0,40}?\b(?:lab|company|team|startup|studio)\b", re.I)
STAGING_RE = re.compile(r"(?i)\bstatus:\s*staging|pre-release package|\bnot released\b|\bbefore release\b")
VERSION_KEY_RE = re.compile(r"(?im)^[ \t]*[\w-]*(?:version|release)[\w-]*[ \t]*:[ \t]*[\"']?v?(\d+\.\d+\.\d+)\b")
VERSION_META_RE = re.compile(r"(?i)\b(?:name|property|itemprop)\s*=\s*[\"']?[\w:.-]*(?:version|release)")
HISTORY = {"CORE-CHANGES.md"}   # dated history: its entries keep the words and numbers of their date
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "build", "dist", ".pytest_cache", ".ruff_cache", "__pycache__"}


def _prose(repo):
    """The one page list: render, numbers, links, commands and prose checks all read it. Every Markdown (and MDX) page of the
    repo (git ls-files, committed and new; every *.md outside build and tool folders when the tree is no checkout) and
    llms.txt, without dated history, so a new page (a root BENCHMARKS.md, a package README) is checked from its first
    commit."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "--",
                              "*.md", "*.mdx", "llms.txt"], capture_output=True, text=True, check=True).stdout.splitlines()
        files = [repo / x for x in out if x]
    except (OSError, subprocess.CalledProcessError):
        files = [f for g in ("*.md", "*.mdx") for f in repo.rglob(g) if not _SKIP_DIRS & set(f.relative_to(repo).parts[:-1])] \
            + [repo / "llms.txt"]
    return sorted({f for f in files if f.is_file() and f.relative_to(repo).as_posix() not in HISTORY
                   and not f.relative_to(repo).as_posix().startswith(PRIVATE_DIRS)},
                  key=lambda f: f.relative_to(repo).as_posix())


def doc_pages(repo):
    """The docs pages of the one list, without the page templates in docs/_templates."""
    return [f for f in _prose(repo) if (r := f.relative_to(repo).as_posix()).startswith("docs/")
            and not r.startswith("docs/_templates/")]


def _long_date(iso):
    y, mo, d = (int(x) for x in iso.split("-"))
    months = ("January February March April May June July August September October November December").split()
    return f"{d} {months[mo - 1]} {y}"


def check_prose(reg, repo, problems):
    """install: every uv tool install line in a code block is models.json's install (or a model's own install).
    prices: every "$N.NN an hour" is a listed chip price times 1, 2, 4 or 8 chips, with no "read <date>" stamp (the
    date stays in models.json prices.read; tools/docs_lint.py fails every other date on a page). company_line: README, llms.txt (and the org profile) carry it word for word. A released model's page
    has no staging or before-release text."""
    allowed = {m.group(0) for src in [reg["install"]] + [m.get("install", "") for m in reg["models"]]
               for m in UV_RE.finditer(src) if "mashbox" in m.group(0)}
    per_chip = {k: Decimal(v) for k, v in reg["prices"]["per_chip_hour"].items()}
    ok_amounts = {(v * n).quantize(Decimal("0.01")) for v in per_chip.values() for n in (1, 2, 4, 8)}
    for f in _prose(repo):
        rel, text = f.relative_to(repo).as_posix(), CLOUD_BLOCK_RE.sub(" ", f.read_text())  # cloud blocks: check_cloud
        spans = [s for s in INLINE_RE.findall(FENCE_RE.sub("", text)) if not re.match(r"(?:python3? -m )?pip3? install(?!.*==)", s)]
        for block in FENCE_RE.findall(text) + spans:   # inline: plain `pip install "mashbox[tpu]"` is the documented failure
            for uv in UV_RE.finditer(block.replace("\\\n", " ")):
                if re.search(r"\bmashbox\b(?!-labs)", uv.group(0)) and uv.group(0) not in allowed:
                    problems.add(rel, f"install line differs from models.json install: {uv.group(0)!r}")
        for line in re.sub(r"&nbsp;|\u00a0", " ", text).splitlines():
            amounts = PRICE_RE.findall(line)
            for hour, d1, d2 in amounts:
                day = d1 or d2
                if hour and Decimal(hour) not in ok_amounts:
                    problems.add(rel, f"${hour} an hour is not a models.json prices.per_chip_hour price times 1, 2, 4 or 8 chips")
                if day and Decimal(day) not in {(v * 24).quantize(Decimal("0.01")) for v in ok_amounts}:
                    problems.add(rel, f"${day} a day is not 24 hours of a models.json prices.per_chip_hour price times 1, 2, 4 or 8")
            for mins, amt in MINUTES_PRICE_RE.findall(line):
                if Decimal(amt) not in {(v * int(mins) / 60).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) for v in ok_amounts}:
                    problems.add(rel, f"{mins} minutes is about ${amt}: not {mins} minutes of a models.json prices.per_chip_hour price")
            for d in READ_RE.findall(line):
                problems.add(rel, f"price read date {d!r} on a public page: a page states the price, the date it was "
                                  f"read stays in models.json prices.read")
    pins = set()
    def walk(x):
        if isinstance(x, dict):
            pins.update(str(v) for k, v in x.items() if k in ("jax", "jaxlib") and isinstance(v, str))
            [walk(v) for v in x.values()]
        elif isinstance(x, list):
            [walk(v) for v in x]
    walk(reg["models"])
    pp = repo / "pyproject.toml"
    jax_re = re.compile(r"\b(?:JAX|jax(?:lib)?(?:\[tpu\])?==|jax(?:lib)? )v?(\d+\.\d+\.\d+)")
    if len(pins) > 1:
        problems.add("models.json", f"kernel bundles pin different JAX versions: {sorted(pins)}")
    for rel, text in [(f.relative_to(repo).as_posix(), f.read_text()) for f in _prose(repo)] + \
            ([("pyproject.toml", pp.read_text())] if pp.exists() else []):
        for line in text.splitlines():
            for v in jax_re.findall(line) if pins and "Measured on" not in line else []:
                if v not in pins:
                    problems.add(rel, f"JAX {v} is not the models.json kernel bundle pin ({', '.join(sorted(pins))}): "
                                      f"{line.strip()[:100]!r}")
    for rel in ("README.md", "llms.txt"):
        f = repo / rel
        if f.exists() and reg["company_line"] not in f.read_text():
            problems.add(rel, "company line differs from models.json company_line")
    pages = [(f.relative_to(repo).as_posix(), f.read_text()) for f in _prose(repo)]
    pp = repo / "pyproject.toml"    # its description is the PyPI page's first line
    pages += [("pyproject.toml", "\n".join(re.findall(r'^description\s*=\s*"(.*)"', pp.read_text(), re.M)))] if pp.exists() else []
    pages += docstrings(repo)   # help() shows them
    for rel, text in pages:   # anywhere the company is described ("MASHBOX Labs makes/builds/is ...", "MASHBOX Labs, a ..."), it is the line itself
        text = CATALOG_RE.sub("", text)   # the generated catalog block ("## MASHBOX Labs (Hugging Face)") is checked against its render
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", TAG_RE.sub(lambda mt: tag_words(mt.group(0)), text)).replace("**", "").replace("__", "")
        text = " ".join(html.unescape(text).replace("\u00a0", " ").split())
        for mt in COMPANY_RE.finditer(text):
            if not text.startswith(reg["company_line"], mt.start()):
                problems.add(rel, f"describes the company in other words: "
                             f"{text[mt.start():mt.start() + 70]!r}... (use models.json company_line word for word)")
    for f in _prose(repo) + ui_pages(repo):   # a version a page declares as metadata (front matter, <meta>) is the release
        rel, text = f.relative_to(repo).as_posix(), f.read_text()
        fm = re.match(r"---\n(.*?)\n---", text, re.S)
        said = VERSION_KEY_RE.findall(fm.group(1)) if fm else []
        said += [v for tag in re.findall(r"(?i)<meta\b[^>]*>", text) if VERSION_META_RE.search(tag)
                 for v in re.findall(r"(?i)\bcontent\s*=\s*[\"']?v?(\d+\.\d+\.\d+)", tag)]
        for v in said:
            if v != reg["version"]:
                problems.add(rel, f"declares version {v} in its metadata; the release is models.json version {reg['version']}")
    for m in reg["models"]:
        page = repo / "docs" / "models" / f"{m['id']}.md"
        if not m.get("launch") and page.exists():
            hit = STAGING_RE.search(page.read_text())
            if hit:
                problems.add(page.relative_to(repo).as_posix(), f"released model page still says {hit.group(0)!r}")


HEAD_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)


def slug(text):
    """A heading's anchor as GitHub and the docs site make it (github-slugger, as gen_faq.slug)."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\- ]", "", text.strip().lower().replace("`", "")).replace(" ", "-")


def anchors(page):
    out, seen = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', page.read_text())), {}
    for h in HEAD_RE.findall(FENCE_RE.sub("", page.read_text())):
        s = slug(h)
        out.add(s if s not in seen else f"{s}-{seen[s]}")
        seen[s] = seen.get(s, 0) + 1
    return out


CSS_URL_RE = re.compile(r"url\(\s*([\"']?)([^\"')\s]+)\1\s*\)")
CSS_IMPORT_RE = re.compile(r"@import\s+([\"'])([^\"'\s]+)\1")   # @import "x.css" (the url() form is CSS_URL_RE)


def html_targets(body):
    """Local targets a browser follows in a page's raw HTML and CSS, quoted or not (html.parser reads attributes as a
    browser does): href, src, srcset, action, formaction, poster, object data, a meta refresh URL, url() in style
    blocks and style attributes, and @import in style blocks."""
    from html.parser import HTMLParser
    out = []

    class Tags(HTMLParser):
        def handle_starttag(self, tag, attrs):
            a = {k: v or "" for k, v in attrs}
            out.extend(a[k] for k in ("href", "src", "action", "formaction", "poster") if a.get(k))
            out.extend([a["data"]] if tag == "object" and a.get("data") else [])
            out.extend(s.split()[0] for s in a.get("srcset", "").split(",") if s.split())
            if tag == "meta" and a.get("http-equiv", "").lower() == "refresh":
                out.extend(re.findall(r"(?i)\burl\s*=\s*[\"']?([^\"';\s]+)", a.get("content", "")))
            out.extend(u for _, u in CSS_URL_RE.findall(a.get("style", "")))
        handle_startendtag = handle_starttag
    p = Tags(convert_charrefs=True)
    p.feed(body)
    p.close()
    return out + [u for css in re.findall(r"(?is)<style\b[^>]*>(.*?)</style>", body)
                  for _, u in CSS_URL_RE.findall(css) + CSS_IMPORT_RE.findall(css)]


def check_conditions(reg, repo, problems):
    """A results row with a "condition" (words that change what its number means, e.g. "main model quantized") never
    loses them: every sentence or table row that prints the row's number, on a public page, a model card, llms.txt or
    a models.json text field of that model, carries the condition words (a short words-only
    condition, never a bit count). The results row itself is the source (its metric names the condition)."""
    conds = []
    for m in reg["models"]:
        for i, r in enumerate(m.get("results") or []):
            if not r.get("condition"):
                continue
            num = re.match(r"\s*(\d[\d,]*(?:\.\d+)?(?:\s*(?:ms|s|min|tok/s|tokens/s|GB|GiB|%))?)", str(r.get("value", "")))
            if num:
                pat = re.compile(r"(?<![\d.])" + re.escape(num.group(1)) + r"(?![\d])")
                conds.append((m, i, pat, num.group(1), r["condition"]))
    def units(text):   # a table row is one unit, prose splits at sentence ends and <br>
        for n, line in enumerate(text.splitlines(), 1):
            parts = [line] if line.lstrip().startswith("|") else re.split(r"(?<=[.!?])\s+|<br\s*/?>", line)
            for part in parts:
                yield n, part
    for m, i, pat, num, cond in conds:
        if cond not in str(m["results"][i].get("metric", "")):
            problems.add("models.json", f"{m['id']} results[{i}] condition {cond!r} is not in its metric")
        def walk(x, where):
            if isinstance(x, dict):
                for k, v in x.items():
                    if not (where == m["id"] and k == "results"):
                        walk(v, f"{where}.{k}")
            elif isinstance(x, list):
                for j, v in enumerate(x):
                    walk(v, f"{where}[{j}]")
            elif isinstance(x, str):
                for _, part in units(x):
                    if pat.search(part) and cond not in part:
                        problems.add("models.json", f"{where} prints {num} ({m['id']} results[{i}]) without {cond!r}")
        walk(m, m["id"])
        for f in _prose(repo):
            for n, part in units(f.read_text()):
                if pat.search(part) and cond not in part and m["name"] in f.read_text():
                    problems.add(f.relative_to(repo).as_posix(), f"line {n}: {num} ({m['id']} results[{i}]) without {cond!r}: "
                                                                 f"{part.strip()[:120]}")


LEVELS_PAGE = "docs/supported-models.md"   # its "## Levels" table defines every level a catalog row prints


def check_levels(repo, problems):
    """Every level a catalog block prints (the Level column of each table between <!-- catalog:start --> and
    <!-- catalog:end -->, on any page) is a row of the "## Levels" table on docs/supported-models.md, so a reader can
    look up what each row claims. The level itself is computed by mashbox.catalog.level()."""
    page = repo / LEVELS_PAGE
    if not page.exists():
        return
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]
    sec = re.search(r"^## Levels[ \t]*\n(.*?)(?=^## |\Z)", page.read_text(), re.S | re.M)
    defined = {cells(x)[0] for x in (sec.group(1).splitlines() if sec else []) if x.lstrip().startswith("|")} \
        - {"Level", ""}
    defined = {d for d in defined if not set(d) <= set("-: ")}
    if not defined:
        problems.add(LEVELS_PAGE, "no \"## Levels\" table: nothing defines the levels the catalog rows print")
    for f in _prose(repo):
        if not f.exists():
            continue
        for block in CATALOG_RE.finditer(f.read_text()):
            col, printed = None, {}
            for line in block.group(2).splitlines():
                if not line.lstrip().startswith("|"):
                    col = None
                    continue
                row = cells(line)
                if col is None:
                    col = row.index("Level") if "Level" in row else -1
                elif 0 <= col < len(row) and not set(row[col]) <= set("-: "):
                    lvl = re.split(r"<br\s*/?>", row[col])[0].strip()
                    printed[lvl] = printed.get(lvl, 0) + 1
            for lvl, n in sorted(printed.items()):
                if lvl not in defined:
                    problems.add(f.relative_to(repo).as_posix(), f"the catalog prints level {lvl!r} on {n} row(s) but "
                                                                 f"the Levels table on {LEVELS_PAGE} does not define it")


def check_relative_links(repo, problems, reg=None):
    """Every relative link and image in the hand-written pages (inline or [id]: style) points at a file or folder in
    the repo, and a #fragment into a page at one of its headings (T0, offline). README's links are pinned to
    github.com/ORG/REPO/blob/vX.Y.Z/..., read as the same repo path. sync_docs turns a missing target into a GitHub link
    that answers 404 and GitHub answers 200 for any fragment, so a typo would otherwise ship."""
    own = re.compile(rf"https://github\.com/{reg['github']['org']}/{reg['github']['repo']}/(?:blob|tree)/[^/#?\s]+/"
                     if reg else r"(?!)")
    site = re.compile(r"https://mashbox\.gg/docs(?=[/#]|$)/?")

    def site_path(rest):   # mashbox.gg/docs/PATH is docs/PATH.md (or PATH/README.md) as sync_docs publishes it
        path, _, frag = rest.partition("#")
        path = path.split("?", 1)[0].strip("/")
        cands = [f"docs/{path}.md", f"docs/{path}/README.md", f"docs/{path}"] if path else ["docs/README.md"]
        return next((c for c in cands if (repo / c).is_file()), cands[0]) + ("#" + frag if frag else "")
    refs = ({"main", f"v{reg['version']}"} | {f"v{m['card_release']}" for m in reg["models"] if m.get("card_release")}) if reg else set()
    for f in _prose(repo):
        rel, body = f.relative_to(repo).as_posix(), FENCE_RE.sub("", f.read_text())
        targets = [mt.group(2) or mt.group(5) for mt in LINK_RE.finditer(body)] + [mt.group(2) for mt in REF_RE.finditer(body)]
        targets += html_targets(body)
        targets += [u for u in URL_RE.findall(body) if own.match(u) or site.match(u)]   # bare URLs too
        for u in URL_RE.findall(body):
            ref = own.match(u) and u.split("/")[6]
            if ref and ref not in refs and not SHA_RE.match(ref):
                problems.add(rel, f"GitHub link at {ref!r}: our links name main or this release's tag ({sorted(refs)}): {u}")
            sp = re.match(r"https?://(?:www\.)?mashbox\.gg(/[^#?]*)?", u)
            if sp and not site.match(u) and (sp.group(1) or "/").strip("/") not in SITE_PATHS:
                problems.add(rel, f"mashbox.gg link outside /docs that the site does not serve: {u} "
                                  f"(the site's own paths: {sorted(SITE_PATHS)})")
        for target in dict.fromkeys(t.rstrip(".,;:*") for t in targets):
            base = f.parent
            if own.match(target):
                base, target = repo, own.sub("", target)
            elif site.match(target):
                base, target = repo, site_path(site.sub("", target))
            elif re.match(r"/docs(?=[/#?]|$)", target):   # site root: /docs/PATH is mashbox.gg/docs/PATH
                base, target = repo, site_path(target[len("/docs"):])
            elif re.match(r"/(?!/)", target):
                if target.split("#")[0].split("?")[0].strip("/") not in SITE_PATHS:
                    problems.add(rel, f"root-relative link the site does not serve: {target} (its paths: /docs/..., "
                                      f"{sorted('/' + s for s in SITE_PATHS)})")
                continue
            elif re.match(r"([a-z][a-z0-9+.-]*:|/)", target, re.I) or re.search(r"[<{$]", target):
                continue
            path, _, frag = target.partition("#")
            dest = base / path.split("?", 1)[0] if path else f
            if not dest.exists():
                problems.add(rel, f"relative link to a missing path: {target}")
            elif frag and dest.suffix == ".md" and frag not in anchors(dest):
                problems.add(rel, f"link to a heading that is not there: {target}")


SITE_PATHS = {"", "install.sh"}   # mashbox.gg paths outside /docs: the home page and the install script
APP_CATALOG = Path("src/mashbox/ui/app/__init__.py")


def catalog_url(reg):
    gh = reg["github"]
    return f"https://raw.githubusercontent.com/{gh['org']}/{gh['repo']}/main/models.json"


def check_app(reg, repo, app, fetch, problems):
    """The MASHBOX app (Mac): its catalog is this models.json, and the runtime it bundles is the release.
    Offline (T0): the app's CATALOG_URL is the public models.json. With --app <desktop checkout>: the one bundled wheel
    has models.json's version and the same models.json inside; with --online also the PyPI wheel's sha256 for that
    version (the app ships the released wheel, never a build of a later commit)."""
    f = repo / APP_CATALOG
    if f.exists():   # one module-level CATALOG_URL, a plain string (registry() reading it is tests/cpu/test_app_module.py)
        import ast
        sets = [n for n in ast.walk(ast.parse(f.read_text())) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                and any(isinstance(t, ast.Name) and t.id == "CATALOG_URL" for t in getattr(n, "targets", [getattr(n, "target", None)]))]
        vals = [n.value.value for n in sets if isinstance(n.value, ast.Constant)]
        if len(sets) != 1 or vals != [catalog_url(reg)]:
            problems.add(APP_CATALOG.as_posix(), f"want one CATALOG_URL = {catalog_url(reg)!r}, found {len(sets)} assignment(s): {vals}")
    if app is None:
        return
    import zipfile
    wheels = sorted((Path(app) / "src-tauri" / "resources" / "wheels").glob("mashbox-*.whl"))
    if len(wheels) != 1:
        problems.add("app", f"want one bundled mashbox wheel, found {len(wheels)}")
        return
    w = wheels[0]
    ver = w.name.split("-")[1]
    if ver != reg["version"]:
        problems.add("app", f"bundled wheel {w.name} is not the release {reg['version']}")
    with zipfile.ZipFile(w) as z:
        inside = z.read("mashbox/models.json") if "mashbox/models.json" in z.namelist() else b""
    if json.loads(inside or b"{}") != reg:
        problems.add("app", f"bundled wheel {w.name}: its models.json differs from this models.json")
    if fetch is not None:
        st, body = fetch.get(f"https://pypi.org/pypi/mashbox/{ver}/json")
        sums = {u["digests"]["sha256"] for u in json.loads(body).get("urls", []) if u["filename"].endswith(".whl")} if st == 200 else set()
        if hashlib.sha256(w.read_bytes()).hexdigest() not in sums:
            problems.add("app", f"bundled wheel {w.name} is not PyPI's mashbox {ver} wheel (sha256); bundle the released wheel")


CMD_RE = re.compile(r"(?:^[ \t]*(?:\$ )?|(?<=&& )|(?<=; )|(?<=\| )|(?<=\()|(?<=='')|(?<==)(?=\S*/bin/mashbox\b)|(?<=-m ))"
                    r"(?:(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"\n]*\"|'[^'\n]*'|\S*)|nohup|sudo(?: -\S+)*|env|time|exec)[ \t]+)*"
                    r"(?:[~$%\w./{}-]*/bin/|/\S*/)?mashbox((?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s'\"&|;)`]+))*)", re.M)
INLINE_RE = re.compile(r"`([^`\n]+)`")
# a command given as a key's value in YAML, TOML or JSON (command: ..., "run": "...", exec = "..."): read as a command line
KEYED_CMD_RE = re.compile(r"""(?m)^[ \t]*(?:-[ \t]+)?["']?(?:command|cmd|run|exec|script|entrypoint)["']?[ \t]*[:=][ \t]*(["']?)(.*?)\1[ \t]*,?[ \t]*$""")
# a command in exec form, a list of strings (Dockerfile, Compose, Kubernetes, JSON): ["mashbox", "run", "x"]
EXEC_FORM_RE = re.compile(r"""\[[ \t]*(?:"[^"\n]*"|'[^'\n]*')(?:[ \t]*,[ \t]*(?:"[^"\n]*"|'[^'\n]*'))*[ \t]*,?[ \t]*\]""")
DOCKER_RE = re.compile(r"(?m)^[ \t]*(?:RUN|CMD|ENTRYPOINT)[ \t]+")   # a Dockerfile line runs the command after it


def _exec_line(mt):
    import ast
    import shlex
    try:
        return shlex.join(ast.literal_eval(mt.group(0)))
    except (ValueError, SyntaxError, TypeError):
        return mt.group(0)


def code_block(b):
    """A code block as the command lines it runs: a key's value (command: ...), a list in exec form read as its command
    line, and a Dockerfile RUN, CMD or ENTRYPOINT line read as the command it gives."""
    return DOCKER_RE.sub("", EXEC_FORM_RE.sub(_exec_line, KEYED_CMD_RE.sub(r"\2", b.replace("\\\n", " "))))


def example_files(repo):
    """Files people run as they are: everything under examples/ and every shell script outside tests/."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "--",
                              "examples", "*.sh", "*.bash"], capture_output=True, text=True, check=True).stdout.splitlines()
        files = [repo / x for x in out if x]
    except (OSError, subprocess.CalledProcessError):
        files = list((repo / "examples").rglob("*")) + [f for g in ("*.sh", "*.bash") for f in repo.rglob(g)]
    return sorted({f for f in files if f.is_file() and not _SKIP_DIRS & set(f.relative_to(repo).parts[:-1])
                   and not f.relative_to(repo).as_posix().startswith("tests/")}, key=lambda f: f.relative_to(repo).as_posix())


OTHER_FLAGS = {  # flags of other tools that the pages name in inline code: gcloud, Claude Code, Codex, vLLM
    "--accelerator-type", "--version", "--profile", "--settings", "--strict-mcp-config", "--permission-mode",
    "--allowedTools", "--dangerously-bypass-approvals-and-sandbox", "--max-num-seqs", "--gpu-memory-utilization",
    "--no-enable-prefix-caching", "--durations"}   # and pytest (CI.md); --mbx-* are the shell's CSS variables


def _package_flags(repo, rel):
    """A page under packages/<name>/ documents that package's own command line: the flags its source adds."""
    import ast
    parts = Path(rel).parts
    if len(parts) < 3 or parts[0] != "packages":
        return set()
    return {a.value for py in (Path(repo) / "packages" / parts[1]).rglob("*.py") for c in ast.walk(ast.parse(py.read_text()))
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument"
            for a in c.args if isinstance(a, ast.Constant) and str(a.value).startswith("-")}


@_mashbox_as_found
def check_commands(reg, repo, problems):
    """Every `mashbox ...` command in a page's code (blocks and inline) and every run_args list in models.json parses
    with the package's own parsers, and inline code made of flags names real flags (as vLLM checks its docs' commands).
    Renaming a flag or a typo in an example fails here, not on a user's chip."""
    mods = _package(repo, problems)
    if mods is None:
        return
    import contextlib
    import io
    import shlex
    by_cmd = {cmd[len("mashbox "):]: p for cmd, p, _ in cli_parsers(mods)}
    own = {}   # a model with its own command line (cli.MODEL_CLI): the flags its parser() adds, read from the source
    import ast
    for mid, mod in getattr(mods[".cli"], "MODEL_CLI", {}).items():
        own[mid] = {a.value for py in (repo / "src" / Path(*mod.split("."))).glob("*.py") for fn in ast.walk(ast.parse(py.read_text()))
                    if isinstance(fn, ast.FunctionDef) and fn.name == "parser" for c in ast.walk(fn)
                    if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument"
                    for a in c.args if isinstance(a, ast.Constant) and str(a.value).startswith("-")}
    flags = {o for p in by_cmd.values() for o in p._option_string_actions} | {f for v in own.values() for f in v} | OTHER_FLAGS

    def parse(argv, where, partial=False):
        if argv[:1] in (["run"], ["download"]) and len(argv) > 1 and argv[1] in own:
            for fl in argv[2:]:
                if fl.startswith("-") and fl.split("=")[0] not in own[argv[1]] | {"-h", "--help"}:
                    problems.add(where, f"mashbox {' '.join(argv)}: {fl} is not an option of mashbox {argv[0]} {argv[1]}")
            return
        key = " ".join(argv[:2]) if argv[:1] == ["tpu"] else " ".join(argv[:1])
        p = by_cmd.get(key)
        if p is None and partial and any(k.startswith(key + " ") for k in by_cmd):
            return   # `mashbox tpu` in prose: the command group
        if p is None:
            return problems.add(where, f"unknown command: mashbox {' '.join(argv)}")
        rest = argv[len(key.split()):]
        if key == "tpu up" and "--" in rest:    # after --: options for mashbox run on the chip
            parse(["run", "MODEL"] + rest[rest.index("--") + 1:], where)
            rest = rest[:rest.index("--")]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                p.parse_args(rest)
        except SystemExit as e:
            msg = (err.getvalue().strip().splitlines() or [""])[-1]
            if e.code and not (partial and "arguments are required" in msg):
                problems.add(where, f"mashbox {' '.join(argv)}: {msg}")

    pages = [(f.relative_to(repo).as_posix(), re.sub(r"<!--.*?-->", " ", f.read_text(), flags=re.S)) for f in number_pages(reg, repo)]
    pages += [(f.relative_to(repo).as_posix(), "```\n" + f.read_text(errors="replace") + "\n```\n") for f in example_files(repo)]
    for rel, text in pages:   # an example file is one code block
        if rel in ("docs/cli.md", "docs/server-api.md"):
            continue   # rendered from the parsers and the spec themselves
        blocks = [code_block(b) for b in FENCE_RE.findall(text)]
        spans = INLINE_RE.findall(FENCE_RE.sub("", text))
        for code, partial in [(b, False) for b in blocks] + [(s, True) for s in spans]:
            code = re.sub(r"\$\{[^}]*\}|<[^>\n]*>", "X", re.sub(r" \[[a-z .]+\]", "", code))  # ${id}, <model id>, [options]
            for mt in CMD_RE.finditer(code):
                try:
                    argv = shlex.split(mt.group(1))
                except ValueError as e:
                    problems.add(rel, f"cannot read the command mashbox{mt.group(1)}: {e}")
                    continue
                if argv:
                    parse(argv, rel, partial)
        prose = INLINE_RE.sub(" ", FENCE_RE.sub(" ", text))
        for s in [x for x in spans if x.startswith("--")] + [mt.group(0) for mt in re.finditer(r"(?<![\w`-])--[A-Za-z][\w-]*", prose)]:
            for fl in re.findall(r"(?<![\w-])--[A-Za-z][\w-]*", s):
                if fl not in flags and fl not in _package_flags(repo, rel) and not fl.startswith("--mbx-"):
                    problems.add(rel, f"`{s}`: {fl} is not an option of any mashbox command (docs/cli.md)")
    for m in reg["models"]:
        args = ((m.get("cloud") or {}).get("agents") or {}).get("run_args")
        if args is not None:
            parse(["run", m["id"]] + list(args), f"models.json {m['id']} cloud.agents.run_args")


def licence_report(reg, repo):
    """Lines that state a licence, beside LICENSE and the package metadata. A report for a person: it never edits and
    never fails (licence wording is decided by people, not by this tool)."""
    lic = (repo / "LICENSE").read_text().split() if (repo / "LICENSE").exists() else ["missing"]
    out = [f"LICENSE starts: {' '.join(lic[:4])}",
           f"models.json license: {reg.get('license')!r}"]
    py = repo / "pyproject.toml"
    if py.exists():
        lic = [x.strip() for x in py.read_text().splitlines() if re.match(r"\s*licen[cs]e(-files)?\s*=", x)]
        out.append(f"pyproject.toml: {lic or 'no license field'}")
    pat = re.compile(r"(?i)licen[cs]e|apache|\bMIT\b|not chosen")
    for f in _prose(repo):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line) and ("not chosen" in line.lower() or "licen" in line.lower()):
                out.append(f"{f.relative_to(repo).as_posix()}:{i}: {line.strip()[:200]}")
    return out


LICENCE_UNSTATED_RE = re.compile(r"(?i)not chosen|not decided|pending|all rights reserved|LICENSE-PENDING")
APACHE_HEAD = ("Apache License", "Version 2.0, January 2004")
LICENCE_HISTORY = {"CORE-CHANGES.md"}  # dated history: older entries keep the words they had on their date


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream_pins(repo, m):
    """{repo: revision} pinned by the model's upstream.json (its code commit and every repo it downloads), or None."""
    f = repo / m["path"] / "upstream.json"
    if not f.is_file():
        return None
    u = json.loads(f.read_text())
    pins = {u["code"]["repo"]: u["code"]["commit"]} if u.get("code") else {}
    pins.update({r["hf"]: r["revision"] for r in (u.get("repos") or {}).values() if r.get("hf")})
    return pins


def check_licences(reg, repo, problems):
    """Licence drift, report only: fails when the licence files, package metadata, upstream pins or the licence wording
    of a page or card source disagree with models.json. Never edits (a licence is changed by people, by hand).
    The rule: our stack is Apache-2.0; someone else's weights, and our conversions of them, keep the
    upstream licence recorded at our pin, with the upstream LICENSE file unchanged."""
    ours = str(reg["license"])
    lic = repo / "LICENSE"
    if not lic.is_file():
        problems.add("LICENSE", "missing")
    elif ours == "Apache-2.0" and not all(s in lic.read_text()[:400] for s in APACHE_HEAD):
        problems.add("LICENSE", "is not the Apache License 2.0 text, while models.json license is Apache-2.0")
    py = repo / "pyproject.toml"
    if py.is_file():
        import tomllib
        proj = tomllib.loads(py.read_text()).get("project", {})
        if proj.get("license") != ours:
            problems.add("pyproject.toml", f"[project].license is {proj.get('license')!r}, models.json license is {ours!r}")
        if "LICENSE" not in proj.get("license-files", []):
            problems.add("pyproject.toml", "[project].license-files does not list LICENSE")
    for m in reg["models"]:
        up, where = m["upstream"], f"models.json {m['id']}"
        f, want = repo / "licenses" / m["id"] / "LICENSE", up.get("license_sha256")
        if f.is_file() and not want:
            problems.add(where, f"upstream.license_sha256 missing (the sha256 of licenses/{m['id']}/LICENSE as pinned)")
        elif want and not f.is_file():
            problems.add(where, f"licenses/{m['id']}/LICENSE missing (copy the upstream file unchanged)")
        elif want and _sha256(f) != want:
            problems.add(f"licenses/{m['id']}/LICENSE", "differs from the pinned upstream file (upstream.license_sha256); never edit it")
        for name, _, lic in own_files(repo, m):
            if not lic:
                problems.add(where, f"{name}: our own file in {m['hf_repo']} has no licence in upstream.json (a maintainer decides it)")
            elif str(lic).lower() not in SPDX:
                problems.add(where, f"{name}: licence {lic!r} is not one of {sorted(SPDX)}")
        pins = _upstream_pins(repo, m)
        if pins is not None and up["hf"] in pins and pins[up["hf"]] != up.get("revision"):
            problems.add(where, f"upstream.revision differs from upstream.json ({str(pins[up['hf']])[:12]})")
        for c in up.get("components", []):
            src = c.get("hf") or c.get("github") or "?"
            for key in ("license", "revision"):
                if str(c.get(key) or "").strip().lower() in UNCONFIRMED:
                    problems.add(where, f"upstream.components {src}: {key} is empty")
            if pins is not None and pins.get(src) != c.get("revision"):
                problems.add(where, f"upstream.components {src}: revision differs from upstream.json ({str(pins.get(src))[:12]})")
    for f in _prose(repo) + sorted((repo / "templates" / "hf").glob("*.md")):
        rel = f.relative_to(repo).as_posix()
        if rel in LICENCE_HISTORY:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"(?i)licen[cs]e", line) and LICENCE_UNSTATED_RE.search(line):
                problems.add(f"{rel}:{i}", "licence wording says not chosen, pending or reserved; models.json states it")


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

    def status(self, url):
        """HTTP status of a link, logged out, without reading the body (a link may be a 27 GB weights file)."""
        if self.fixtures is not None:
            return self.get(url)[0]
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mashbox-sync-check link check)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 0


def public_file_problems(m, rev, listing, fetch):
    """Every files[] row of a released model is in its one public repo (hf_repo), at the pinned revision or at the row's
    own revision of that repo, read from the Hub's file list (siblings). The release export is checked logged out, so a
    file that only a private repo holds never passes."""
    out, lists = [], {rev: listing}
    for j, f in enumerate(m.get("files") or []):
        if f.get("repo", m["hf_repo"]) != m["hf_repo"]:
            out.append(f"files[{j}] {f['name']} is in {f['repo']}: every file of a released model is in its one public "
                       f"repo {m['hf_repo']}")
            continue
        r = f.get("revision") or rev
        if r not in lists:
            st, body = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}/revision/{r}")
            lists[r] = body if st == 200 else ""
        try:
            names = {s.get("rfilename") for s in json.loads(lists[r] or "{}").get("siblings") or []}
        except ValueError:
            names = set()
        if f["name"] not in names:
            out.append(f"files[{j}] {f['name']} is not in {m['hf_repo']} at {r[:12]}")
    return out


def card_tag_problems(m, rev, cards, fetch):
    """A card line that names a Hugging Face tag of its repo is true only when that tag points at the pinned revision
    (the one that carries the card). Uploading a card makes a new revision, and a tag is never moved (a move deletes it
    first), so after a card-only upload the card names no tag. cards: {label: card text}. Tags read logged out."""
    st, body = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}/refs")
    if st != 200:
        return [f"cannot read the Hugging Face tags (HTTP {st})"]
    out = []
    for name in sorted({t.get("name") for t in json.loads(body).get("tags") or [] if t.get("name")}):
        pat = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")
        hits = [(label, i + 1) for label, text in cards.items() for i, x in enumerate(text.splitlines()) if pat.search(x)]
        if not hits:
            continue
        st, rb = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}/revision/{urllib.parse.quote(name, safe='')}")
        sha = json.loads(rb).get("sha") if st == 200 else None
        if sha == rev:
            continue
        for label, n in hits:
            out.append(f"{label} line {n} names the Hugging Face tag {name}, which points at "
                       f"{str(sha)[:12] if sha else f'nothing readable (HTTP {st})'}, not at the pinned revision {rev[:12]}: "
                       f"a card names no tag, or only a tag of the revision it is on")
    return out


def weights_tag_problems(m, rev, fetch):
    """hf_tag (the dated weights version) points at a revision whose default weights file has the sha256 models.json
    pins, and so does the pinned revision: a card-only revision after the tag keeps the weights, so the tag may point at
    an older revision, never at other weights."""
    tag, w = m.get("hf_tag"), default_weights(m)
    if not tag:
        return []
    if not w or not w.get("sha256"):
        return [f"hf_tag {tag}: files[] has no default weights file with a sha256"]
    out = []
    for label, r in ((f"tag {tag}", urllib.parse.quote(tag, safe="")), (f"the pinned revision {rev[:12]}", rev)):
        st, body = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}/revision/{r}?blobs=true")
        if st != 200:
            out.append(f"hf_tag {tag}: cannot read {label} (HTTP {st})")
            continue
        sha = next(((s.get("lfs") or {}).get("sha256") for s in json.loads(body).get("siblings") or []
                    if s.get("rfilename") == w["name"]), None)
        if sha != w["sha256"]:
            out.append(f"hf_tag {tag}: {w['name']} at {label} has sha256 {str(sha)[:12]}, models.json pins {w['sha256'][:12]}")
    return out


def check_online(reg, views, fetch, problems, repo=None):
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
        if m.get("launch"):  # coming soon: private (401/403 without a token) or absent, never public before its launch
            if st == 200 and not (json.loads(body).get("private") or json.loads(body).get("gated")):
                problems.add(where, f"public before its launch ({m['launch']})")
            elif st not in (200, 401, 403, 404):
                problems.add(where, f"cannot read the repo state (HTTP {st})")
            continue
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
        if info.get("sha") != rev:  # the visible head must be the pinned, rendered revision
            problems.add(where, f"HF head {str(info.get('sha'))[:12]} is not the pinned revision {rev[:12]}")
        st, listing = fetch.get(f"https://huggingface.co/api/models/{m['hf_repo']}/revision/{rev}")
        if st != 200:
            problems.add(where, f"pinned revision {rev[:12]} not found (HTTP {st})")
            continue
        for x in public_file_problems(m, rev, listing, fetch):
            problems.add(where, x)
        st, card = fetch.get(f"https://huggingface.co/{m['hf_repo']}/raw/{rev}/README.md")
        want = views.get(f"hf/{m['id']}/README.md")
        if st != 200:
            problems.add(where, f"card not readable at {rev[:12]} (HTTP {st})")
        elif want is not None and card.strip() != want.strip():
            problems.add(where, "card on Hugging Face differs from the card rendered from models.json")
        cards = {f"the card at {rev[:12]}": card} if st == 200 else {}
        if want is not None and (st != 200 or card.strip() != want.strip()):
            cards["the card rendered from models.json"] = want
        for x in card_tag_problems(m, rev, cards, fetch) + weights_tag_problems(m, rev, fetch):
            problems.add(where, x)
        lic_sha = m["upstream"].get("license_sha256")
        if lic_sha:  # the upstream LICENSE travels with our conversion, unchanged (Apache-2.0 section 4)
            st, text = fetch.get(f"https://huggingface.co/{m['hf_repo']}/raw/{rev}/LICENSE")
            if st != 200:
                problems.add(where, f"no LICENSE at {rev[:12]} (HTTP {st}): upload licenses/{m['id']}/LICENSE unchanged")
            elif hashlib.sha256(text.encode()).hexdigest() != lic_sha:
                problems.add(where, f"LICENSE at {rev[:12]} differs from the pinned upstream file")
    gh = reg["github"]
    st, _ = fetch.get(f"https://api.github.com/repos/{gh['org']}/{gh['repo']}/git/ref/tags/v{reg['version']}")
    if st != 200:
        problems.add("github", f"tag v{reg['version']} not found in {gh['org']}/{gh['repo']} (HTTP {st})")
    # PyPI: the newest release is this version (pyproject, __version__ and models.json are held equal offline)
    st, body = fetch.get("https://pypi.org/pypi/mashbox/json")
    if st != 200:
        problems.add("pypi", f"cannot read the mashbox project (HTTP {st})")
    elif json.loads(body)["info"]["version"] != reg["version"]:
        problems.add("pypi", f"newest release is {json.loads(body)['info']['version']}, models.json says {reg['version']}")
    # The org profile is an overview: no model list, no count, the family line as models.json has it
    st, prof = fetch.get(f"https://raw.githubusercontent.com/{gh['org']}/.github/main/profile/README.md")
    if st != 200:
        problems.add("profile", f"org profile not readable (HTTP {st})")
    else:
        for w, msg in overview_problems(reg, prof, "profile (live)"):
            problems.add(w, msg)
    # The Hugging Face org card and the "TPU models" collection, as a logged-out visitor sees them
    st, card = fetch.get(ORG_CARD_URL.format(org=hf_org))
    if st != 200:
        problems.add("hf org card", f"not readable (HTTP {st})")
    else:
        for w, msg in overview_problems(reg, card, "hf org card (live)"):
            problems.add(w, msg)
    st, body = fetch.get(f"https://huggingface.co/api/collections?owner={hf_org}&limit=100")
    if st != 200:
        problems.add("hf collections", f"cannot list the collections of {hf_org} (HTTP {st})")
    else:
        for w, msg in collection_problems(reg, body, "hf collections"):
            problems.add(w, msg)
    pages = {f.relative_to(repo).as_posix(): f.read_text() for f in _prose(repo)} if repo is not None else {}
    check_links(reg, {**pages, **views}, fetch, problems)  # every page, hand-written or generated (views win: rendered text)


URL_RE = re.compile(r"https?://[^\s)\"'<>`\]|{}]+")


def check_links(reg, views, fetch, problems):
    """Every absolute link in the pages (README, llms.txt, every docs page, HF cards) answers 200 logged out.
    Skipped: local addresses, placeholders, and a coming-soon model's page, card and private repo (its own check is above)."""
    later = [m for m in reg["models"] if m.get("launch")]  # coming soon: its page, card and links are checked at its launch
    soon = tuple(f"https://huggingface.co/{m['hf_repo']}" for m in later if m["hf_repo"]) + \
        tuple(f"https://mashbox.gg/docs/models/{m['id']}" for m in later)
    skip = {f"hf/{m['id']}/README.md" for m in later} | {f"docs/models/{m['id']}.md" for m in later}
    urls = {}
    for key, text in views.items():
        if key in skip:
            continue
        for u in URL_RE.findall(text):
            u = u.rstrip(".,;:*")
            if re.match(r"https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)([:/]|$)", u) or u.startswith(soon) \
                    or re.search(r"YOUR_|<|\{|\$|^https://link$|/v1/?$", u) or "huggingface.co/api/" in u:
                continue
            urls.setdefault(u.split("#", 1)[0], key)
    for u, where in sorted(urls.items()):
        st = fetch.status(u)
        if st != 200:
            problems.add(f"link in {where}", f"{u} answers HTTP {st}")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["render", "check", "licence", "markup"],
                    help="licence: print every line that states a licence, then the licence check (never edits; exit 1 on a "
                         "mismatch). markup PAGE...: write the number markers back into these pages from the number map, to "
                         "edit the words around a number; render --write takes them out again")
    ap.add_argument("pages", nargs="*", help="markup: the pages (repo-relative)")
    ap.add_argument("--repo", required=True, help="root of the main repo (holds models.json)")
    ap.add_argument("--profile", help="org profile README to check: an overview with no model list (optional)")
    ap.add_argument("--org-card", help="Hugging Face org card (the org's README Space README.md) to check the same way (optional)")
    ap.add_argument("--org-defaults", help="the .github repo: CONTRIBUTING.md and SECURITY.md must equal the main repo's")
    ap.add_argument("--templates", help="card templates; default: <repo>/templates/hf when it exists, else the org templates")
    ap.add_argument("--write", action="store_true", help="render: write the views in place")
    ap.add_argument("--out", help="render: write the views under this directory instead")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--online", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--fixtures")
    ap.add_argument("--app", help="MASHBOX app (desktop) checkout: its bundled wheel must be the release (private nightly)")
    a = ap.parse_intermixed_args(argv)

    repo = Path(a.repo)
    if a.cmd == "markup":
        nmap = load_numbers(repo) or {"pages": {}}
        for rel in a.pages:
            f = repo / rel
            f.write_text(mark_up(f.read_text(), nmap["pages"].get(rel, [])))
            print("marked", rel, len(nmap["pages"].get(rel, [])), "numbers")
        return 0
    templates = Path(a.templates) if a.templates else \
        (repo / "templates" / "hf" if (repo / "templates" / "hf").is_dir() else DEFAULT_TEMPLATES)
    profile = Path(a.profile) if a.profile else None
    org_card = Path(a.org_card) if a.org_card else None
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
    if a.cmd == "licence":
        print("\n".join(licence_report(reg, repo)))
        check_licences(reg, repo, problems)
        for p in problems:
            print("FAIL", p)
        print(f"{'OK' if not problems else 'LICENCE MISMATCH'}: {len(problems)} problems")
        return 1 if problems else 0
    nmap = load_numbers(repo)
    if nmap is not None:
        moved = {o: dict(r.get("figures") or {}) for o, r in [("cloud", reg)] + [(m["id"], m) for m in reg["models"]]}
        merge_figures(reg, nmap, problems if a.cmd == "check" else None)
    views = render_all(reg, repo, profile, templates, problems, nmap["pages"] if nmap is not None else None,
                       org_card=org_card)

    if a.cmd == "render" and nmap is not None and (a.write or a.out):
        # the private map: figures move out of models.json, places follow the pages as rendered
        new = {"_about": nmap.get("_about", ""), "figures": {}, "pages": render_all.places}
        for o, r in [("cloud", reg)] + [(m["id"], m) for m in reg["models"]]:
            if r.get("figures"):
                new["figures"][o] = dict(sorted(r["figures"].items()))
        base = Path(a.out) if a.out else repo
        (base / NUMBERS).parent.mkdir(parents=True, exist_ok=True)
        (base / NUMBERS).write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        print("wrote", base / NUMBERS)
        if any(moved.values()):
            raw = json.loads((repo / "models.json").read_text())
            for row in [raw] + raw["models"]:
                row.pop("figures", None)
            text = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
            for dest in [base / "models.json"] + ([base / "src" / "mashbox" / "models.json"]
                                                 if (repo / "src" / "mashbox" / "models.json").exists() else []):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text)
                print("wrote", dest, "(figures moved to the number map)")

    if a.cmd == "render":
        for key, text in views.items():
            if a.out:
                dest = Path(a.out) / key
            elif a.write:
                dest = repo / key
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
        path = repo / key
        if not path.exists():
            problems.add(key, "rendered view missing on disk (run render --write)")
        elif path.read_text() != text:
            problems.add(key, "differs from its generated view: models.json, runtime/openapi.py or the CLI parsers (run render --write)")
    check_prose(reg, repo, problems)
    for f in _prose(repo):
        rel = f.relative_to(repo).as_posix()
        for w, msg in list_problems(reg, views.get(rel, f.read_text()), rel):
            problems.add(w, msg)
    check_conditions(reg, repo, problems)
    check_levels(repo, problems)
    check_relative_links(repo, problems, reg)
    if nmap is not None:
        check_numbers(reg, repo, problems, nmap["pages"])
        if nmap["pages"] != render_all.places:
            problems.add(NUMBERS.as_posix(), "differs from the pages (run render --write)")
    else:
        print(f"note: no {NUMBERS.as_posix()} in this tree: measured numbers in prose are checked by the maintainers")
    check_commands(reg, repo, problems)
    check_cloud(reg, repo, problems)
    check_licences(reg, repo, problems)
    if a.org_defaults:
        for name in ("CONTRIBUTING.md", "SECURITY.md"):
            mine, theirs = repo / name, Path(a.org_defaults) / name
            if not theirs.exists() or not mine.exists() or mine.read_bytes() != theirs.read_bytes():
                problems.add(name, "org default copy differs from the main repo's (copy it at release)")
    check_app(reg, repo, a.app, Fetcher(a.fixtures) if a.online else None, problems)
    if a.online:
        check_online(reg, views, Fetcher(a.fixtures), problems, repo)
        st, body = Fetcher(a.fixtures).get(catalog_url(reg))  # the catalog the MASHBOX app reads at run time
        if st != 200 or json.loads(body or "{}") != reg:
            problems.add("app", f"the app's catalog {catalog_url(reg)} differs from this models.json (HTTP {st})")
    for p in problems:
        print("FAIL", p)
    n = sum(1 for m in reg["models"])
    print(f"{'OK' if not problems else 'OUT OF SYNC'}: {n} models, {len(views)} views, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
