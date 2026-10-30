#!/usr/bin/env python3
"""Pick the CPU test areas (tier T1) a change needs. Standard library only; CI.md explains the tiers.

    python3 tools/ci_select.py --base <sha> [--head <sha>]    # the files changed between two commits
    python3 tools/ci_select.py --files src/mashbox/cli.py      # an explicit list
    python3 tools/ci_select.py --all
    python3 tools/ci_select.py --files-of core --shard 2/4      # the test files of one T1 job (an area's shard)

Areas are one per models.json row (the id with underscores, e.g. hy_nomas), `gguf` (the GGUF reader, its ops and
kernels, fixtures and test_gguf_*.py) plus `core`:
- a model area is its port folder, apps/<id>/, its own models.json row and the tests whose file names carry the id
- core (the shared package, pyproject.toml, test config, workflows, tools) runs every area, because it touches every model
- docs, licences, cards and templates run T0 only; a tests/tpu file runs T0 plus the area of any CPU test that reads it
- a models.json row runs its model and the core tests (the CLI and `mashbox tpu` tests read the rows)
- the gguf[] rows of models.json run the gguf area and the gguf_rows leg: the T1 tests outside gguf that read the rows
  (rows_files), unless the rest of the change already runs them; a rows-only change does not wait on every model
- a path nothing above claims runs every area, so a new kind of file is never silently untested
tests/conftest.py uses test_area() to mark each test with its area, so the same rule picks the tests.
An area in SHARDS runs as that many T1 jobs: its test files are split by their time on a CI runner (SECONDS), longest
first into the shard with the least time so far (--files-of), so every file runs in exactly one job and the jobs take
about the same time, within the T1 timeout.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
T0_TESTS = {"test_core.py"}                     # the T0 suite: static and registry checks, pytest only
# The T1 leg of the tests that drive the server through the official openai and anthropic clients: they skip without
# the clients at the versions they pin, so they get their own leg that installs exactly those (EXTRA) and the pinned
# Qwen3.8-27B tokenizer files (--upstream-files, MASHBOX_QWEN_UPSTREAM). A change that runs core runs it too.
CLIENTS = "api_clients"
CLIENTS_FILES = {"test_api_conformance.py", "test_api_loaded_clients.py"}
EXTRA = {"hy_nomas": "imageio-ffmpeg", "track2rig": "scipy",          # pip packages an area's tests need beyond .[dev] (the app server test)
         CLIENTS: "openai==3.40.0 anthropic==2.3.0"}
# The upstream files of UPSTREAM_MODEL that runtime/chat.py reads (tokenizer, chat template, stop ids), fetched at its
# models.json upstream revision for the CLIENTS leg and checked against these sha256 before any test reads them.
UPSTREAM_MODEL = "qwen3-8-27b"
UPSTREAM_FILES = {"tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
                  "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
                  "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"}
T0_ONLY_DIRS = ("docs/", "licenses/", "hf/", "templates/", ".github/ISSUE_TEMPLATE/",
                "tests/contracts/")          # the contracts job runs tests/contracts on every pull request
CORE_TESTS = "+core"                            # run the core T1 job without expanding to every model
T0_ONLY_FILES = {"llms.txt", "LICENSE", "LICENSE-KERNELS", ".github/pull_request_template.md", "tests/cpu/test_core.py",
                 "tools/docs_lint_allow.txt"}  # the docs-lint allow counts: only tools/docs_lint.py reads them, in T0
CORE_DIRS = ("src/mashbox/runtime/", ".github/workflows/", "tools/")
CORE_FILES = {"pyproject.toml", "tests/conftest.py"}
SHARDS = {"core": 12, "qwen3_8_27b": 4, CLIENTS: 2, "gguf_rows": 5}   # T1 jobs per area (default 1); core is ~70 test files, about 1,400 tests
# A file in SPLIT runs as that many parts, each its own shard unit (FILE@K/M; --args-of passes --ci-part to pytest, and
# tests/conftest.py keeps part K's tests: the file's test groups dealt round robin, a group = what the module's
# ci_part(item) returns, else each test alone). For a file whose one module fixture outlasts a shard, or whose many
# tests together do (test_aot_table, test_catalog, test_drafters: each test is short, so its parts split its time).
SPLIT = {"tests/cpu/test_spec_in_step.py": 3, "tests/cpu/test_qwen3_8_27b.py": 4, "tests/cpu/test_tp_batched.py": 3,
         "tests/cpu/test_aot_table.py": 3, "tests/cpu/test_catalog.py": 2, "tests/cpu/test_drafters.py": 2}
# Approximate seconds per T1 test file on a CI runner, used only to split an area's files into shards of even time.
# A file missing here counts as SECONDS_DEFAULT; a name here with no file is ignored. Refresh it from a CI run's
# per-file times when a shard runs long. Only files this repository ships are listed (a name here ships with it).
SECONDS_DEFAULT = 5
SECONDS = {
    "tests/cpu/test_activation_edits.py": 43,
    "tests/cpu/test_activation_edits_contract.py": 0,
    "tests/cpu/test_admission_fairness.py": 43,
    "tests/cpu/test_agent_api.py": 0,
    "tests/cpu/test_aot_table.py": 595,
    "tests/cpu/test_aot_table_first_use.py": 159,
    "tests/cpu/test_aot_table_prune.py": 130,
    "tests/cpu/test_aot_table_restart.py": 43,
    "tests/cpu/test_aot_table_startup.py": 228,
    "tests/cpu/test_api_conformance.py": 0,
    "tests/cpu/test_api_loaded_clients.py": 295,  # two 144 s queue tests (its own leg)
    "tests/cpu/test_app_addon_tpu.py": 0,
    "tests/cpu/test_app_addons.py": 5,
    "tests/cpu/test_app_catalog_sig.py": 1,
    "tests/cpu/test_app_module.py": 5,
    "tests/cpu/test_app_quit.py": 2,
    "tests/cpu/test_app_security.py": 10,
    "tests/cpu/test_app_tunnel_guard.py": 13,
    "tests/cpu/test_catalog.py": 488,
    "tests/cpu/test_chat_module.py": 2,
    "tests/cpu/test_chat_relay_guard.py": 2,
    "tests/cpu/test_ci_areas.py": 7,
    "tests/cpu/test_compile_key.py": 1,
    "tests/cpu/test_docs_supported_models.py": 3,
    "tests/cpu/test_draft_model.py": 27,
    "tests/cpu/test_draft_model_cli.py": 148,
    "tests/cpu/test_drafter_detect.py": 0,
    "tests/cpu/test_drafter_gate.py": 1,
    "tests/cpu/test_drafters.py": 437,
    "tests/cpu/test_engine_model_hooks.py": 6,
    "tests/cpu/test_engine_warm.py": 221,
    "tests/cpu/test_front_admission.py": 4,
    "tests/cpu/test_front_failures.py": 96,
    "tests/cpu/test_guard.py": 0,
    "tests/cpu/test_hub_compile_cache.py": 0,
    "tests/cpu/test_metrics.py": 17,
    "tests/cpu/test_model_lists.py": 0,
    "tests/cpu/test_module_shell.py": 0,
    "tests/cpu/test_moe_routed.py": 100,
    "tests/cpu/test_openapi.py": 2,
    "tests/cpu/test_org_card.py": 0,
    "tests/cpu/test_prefix_cache_owner.py": 0,
    "tests/cpu/test_qwen3_8_27b.py": 2148,  # 4 parts x its slowest part without the wall-clock gate's runs
    "tests/cpu/test_qwen3_8_27b_mtp_default.py": 0,
    "tests/cpu/test_sampler.py": 1,
    "tests/cpu/test_serve_batching.py": 10,
    "tests/cpu/test_serve_queue_api.py": 8,
    "tests/cpu/test_serve_security.py": 3,
    "tests/cpu/test_spec_in_step.py": 1344,  # 3 parts x its slowest part (each part repeats the module fixture)
    "tests/cpu/test_spec_options.py": 0,
    "tests/cpu/test_spec_page.py": 5,
    "tests/cpu/test_stop_strings.py": 18,
    "tests/cpu/test_subagents.py": 0,
    "tests/cpu/test_tenancy.py": 6,
    "tests/cpu/test_tp_batched.py": 861,
    "tests/cpu/test_tpu_chip_safety.py": 73,
    "tests/cpu/test_tpu_cli.py": 107,
    "tests/cpu/test_worker_protocol.py": 1,
}


AREAS_EXTRA = ("gguf", CLIENTS)                 # areas that are not a models.json row: GGUF files (reader, ops, kernels), the clients leg
GGUF_DIRS = ("src/mashbox/gguf/", "src/mashbox/ops/", "tools/gguf_fixtures/", "tests/fixtures/gguf/")
GGUF_FILES = {"tools/gguf_check.py"}
# cli.main imports these two on every run (is_gguf_request, the GGUF dispatch), so a change to them runs every area
GGUF_CORE_FILES = {"src/mashbox/gguf/__init__.py", "src/mashbox/gguf/pick.py"}
ROWS = "gguf_rows"                              # the T1 leg of the tests outside gguf that read models.json gguf[] rows
# How a test reaches the gguf[] rows: the rows themselves, the catalog module and its readers, the `models` command, the
# model lists, card and docs sync and GGUF row tools, the speculation records of a GGUF file, the MoE hold of an
# architecture. A test file that names one of these, or imports a tests/cpu helper that does, is in the rows leg.
READS_ROWS = re.compile(r"""\[["']gguf["']\]|get\(["']gguf["']|mashbox\.catalog\b|from mashbox import [^\n]*\bcatalog\b"""
                        r"""|\bcatalog\.(?:registry|entries|render_text|gguf_rows|arch_rows|arch_models|moe_arch)\b"""
                        r"""|(?<![\w)\]])\[["']models["'][,\]]|\bmodel_lists\b|\bhf_github_sync_check\b|\bgguf_rows\b"""
                        r"""|\bcatalog_records\b|drafters\.select\b|drafters import [^\n]*\bselect\b"""
                        r"""|archs\.held\b|archs import [^\n]*\bheld\b""")


def model_areas(text=None):
    reg = json.loads(text if text is not None else (ROOT / "models.json").read_text())
    return [m["id"].replace("-", "_") for m in reg["models"]] + list(AREAS_EXTRA)


def test_area(file_name, areas):
    """tests/cpu/test_hy_nomas_port.py -> hy_nomas; a CLIENTS_FILES name -> CLIENTS; a file name that carries no model
    id -> core."""
    if file_name in CLIENTS_FILES:
        return CLIENTS
    stem = file_name[:-3] if file_name.endswith(".py") else file_name
    stem = stem[5:] if stem.startswith("test_") else stem
    for a in sorted(areas, key=len, reverse=True):
        if stem == a or stem.startswith(a + "_"):
            return a
    return "core"


def registry_areas(base, head):
    """Areas whose models.json rows differ between two commits: a models[] row runs its model and the core tests, the
    gguf[] rows run the gguf area and the rows leg; the other top-level fields only feed the views (T0)."""
    def load(rev):
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:models.json"], capture_output=True, text=True)
        return json.loads(r.stdout) if r.returncode == 0 else None
    old, new = load(base), load(head) if head else json.loads((ROOT / "models.json").read_text())
    if old is None or new is None:
        return {"core"}
    om, nm = ({m["id"]: m for m in reg["models"]} for reg in (old, new))
    changed = {i for i in om.keys() | nm.keys() if om.get(i) != nm.get(i)}
    if changed - nm.keys():
        return {"core"}
    picked = {i.replace("-", "_") for i in changed} | ({CORE_TESTS} if changed else set())
    if old.get("gguf") != new.get("gguf"):
        picked |= {"gguf", ROWS}
    return picked


def tpu_file_users(path, areas):
    """tests/tpu/<file> -> the areas of the CPU tests that name it (test_hy_nomas_port.py runs the kernel check's
    stand-in bundle path on CPU), so an edit there runs those tests; no reader -> T0 only."""
    name = path.rsplit("/", 1)[-1]
    return {test_area(p.name, areas) for p in (ROOT / "tests" / "cpu").glob("*.py") if name in p.read_text()}


def classify(path, areas, base=None, head=None):
    """The areas one changed path selects: a set of area names, {'core'}, or an empty set (T0 only)."""
    if path.startswith("tests/tpu/"):
        return tpu_file_users(path, areas)
    if path in T0_ONLY_FILES or path.startswith(T0_ONLY_DIRS) or path.endswith(".md"):
        return set()
    if path in GGUF_CORE_FILES:
        return {"core"}
    if path in GGUF_FILES or path.startswith(GGUF_DIRS):
        return {"gguf"}
    if path in ("models.json", "src/mashbox/models.json"):
        return registry_areas(base, head) if base else {"core"}
    if path in CORE_FILES or path.startswith(CORE_DIRS):
        return {"core"}
    parts = path.split("/")
    if parts[:3] == ["src", "mashbox", "models"] and len(parts) > 4:
        return {parts[3]} if parts[3] in areas else {"core"}
    if parts[0] == "apps" and len(parts) > 2:
        a = parts[1].replace("-", "_")
        return {a} if a in areas else {"core"}
    if parts[:2] == ["tests", "cpu"] and len(parts) == 3:
        return {test_area(parts[2], areas)}
    return {"core"}                             # src/mashbox/*.py, models/__init__.py and anything unclaimed


def core_has_t1_tests(areas):
    return any(test_area(p.name, areas) == "core" and p.name not in T0_TESTS for p in (ROOT / "tests" / "cpu").glob("test_*.py"))


def select(files, areas, base=None, head=None):
    picked = set()
    for f in files:
        picked |= classify(f, areas, base, head)
    if "core" in picked:
        picked = set(areas) | {"core"}
    if CORE_TESTS in picked:
        picked = (picked - {CORE_TESTS}) | {"core"}
    if "core" in picked and not core_has_t1_tests(areas):
        picked.discard("core")
    if ROWS in picked and {test_area(f.rsplit("/", 1)[-1], areas) for f in area_files(ROWS, areas)} <= picked:
        picked.discard(ROWS)                    # the picked areas already run every test of the rows leg
    return sorted(picked)


def rows_files(areas, root=ROOT):
    """The rows leg: T1 test files outside the gguf area (its own leg runs those) that read the gguf[] rows (READS_ROWS),
    themselves or through a tests/cpu helper module they import; name order, like area_files"""
    cpu = root / "tests" / "cpu"
    helpers = [p.stem for p in cpu.rglob("*.py") if not p.name.startswith("test_") and p.name not in ("__init__.py", "conftest.py")
               and READS_ROWS.search(p.read_text())]
    def reads(text):
        return READS_ROWS.search(text) or any(re.search(rf"\b{h}\b", text) for h in helpers)
    return sorted((p.relative_to(root).as_posix() for p in cpu.rglob("test_*.py")
                   if p.name not in T0_TESTS and test_area(p.name, areas) != "gguf" and reads(p.read_text())),
                  key=lambda f: (f.rsplit("/", 1)[-1], f))


def area_files(area, areas, root=ROOT):
    """The T1 test files of one area: tests/cpu/**/test_*.py whose name gives that area (T0 files excluded), name order;
    for the rows leg, rows_files"""
    if area == ROWS:
        return rows_files(areas, root)
    cpu = root / "tests" / "cpu"
    return sorted((p.relative_to(root).as_posix() for p in cpu.rglob("test_*.py")
                   if p.name not in T0_TESTS and test_area(p.name, areas) == area), key=lambda f: (f.rsplit("/", 1)[-1], f))


def units(files):
    """files with each SPLIT file replaced by its parts FILE@K/M (K = 1..M), in place"""
    return [u for f in files for u in ([f"{f}@{k}/{SPLIT[f]}" for k in range(1, SPLIT[f] + 1)] if f in SPLIT else [f])]


def unit_seconds(u, sec):
    f, _, km = u.partition("@")
    return sec.get(f, SECONDS_DEFAULT) / (int(km.split("/")[1]) if km else 1)


def shard_args(files, shard, seconds=None, whole=False):
    """the pytest arguments of shard 'i/n' when SPLIT files run as parts: shard_files over units(files), then each file
    once, plus --ci-part=FILE@K/M for every part this shard runs (none when it runs every part of that file); whole:
    every file whole, no parts (the sweep)"""
    sec = SECONDS if seconds is None else seconds
    us = list(files) if whole else units(files)
    mine = shard_files(us, shard, {u: unit_seconds(u, sec) for u in us})
    out, parts = [], {}
    for u in mine:
        f, _, km = u.partition("@")
        if f not in out:
            out.append(f)
        if km:
            parts.setdefault(f, []).append(u)
    for f, us in parts.items():
        if len(us) < SPLIT[f]:
            out += [f"--ci-part={u}" for u in us]
    return out


def shard_files(files, shard, seconds=None):
    """files split into shard 'i/n' (1-based) by time: longest first (SECONDS), each to the shard with the least time so
    far, the lower shard on a tie; the n shards partition the list, and each keeps the list's order"""
    i, n = (int(x) for x in shard.split("/"))
    if not 1 <= i <= n:
        raise ValueError(f"shard {shard}: want i/n with 1 <= i <= n")
    sec = SECONDS if seconds is None else seconds
    load, mine = [0] * n, set()
    for k, f in sorted(enumerate(files), key=lambda kf: (-sec.get(kf[1], SECONDS_DEFAULT), kf[0])):
        j = min(range(n), key=lambda s: (load[s], s))
        load[j] += sec.get(f, SECONDS_DEFAULT)
        if j == i - 1:
            mine.add(k)
    return [f for k, f in enumerate(files) if k in mine]


def jobs(picked, whole=False):
    """the T1 matrix rows: one per area, or SHARDS[area] rows for a sharded area. whole (the sweep: every file whole, no
    SPLIT parts): at most one row per test file of the area, so no row is left with no file to run"""
    areas = model_areas() if whole else None
    count = {x: max(1, min(SHARDS.get(x, 1), len(area_files(x, areas)))) if whole else SHARDS.get(x, 1) for x in picked}
    return [{"area": x, "extra": extra(x), "shard": f"{i}/{count[x]}"} for x in picked for i in range(1, count[x] + 1)]


def extra(area):
    """the pip packages an area's T1 job installs beyond .[dev]; the rows leg takes those of the areas its files belong to"""
    if area != ROWS:
        return EXTRA.get(area, "")
    areas = model_areas()
    return " ".join(sorted({EXTRA[a] for a in {test_area(f.rsplit("/", 1)[-1], areas) for f in rows_files(areas)} if a in EXTRA}))


def contracts_summary(path):
    """contracts.json (tests/contracts/conftest.py) as a markdown table: counts by status, then one row per id, FAIL
    first, then SKIP, then PASS; the pytest step carries the pass or fail, this only shows it"""
    if not os.path.exists(path):
        return f"no contracts.json at {path} (the contracts did not write results)"
    doc = json.load(open(path))
    rows = doc.get("contracts", {})
    order = {"FAIL": 0, "SKIP": 1, "PASS": 2}
    counts = {}
    for r in rows.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = [f"### Contracts: {len(rows)} ids, " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: order.get(kv[0], 3)))
           + f" (commit {str(doc.get('code_commit'))[:12]})", "", "| id | status | fingerprint | cases |", "|---|---|---|---|"]
    for cid, r in sorted(rows.items(), key=lambda kv: (order.get(kv[1]["status"], 3), kv[0])):
        out.append(f"| {cid} | {r['status']} | {r['fingerprint'][:16]} | {r.get('cases', '')} |")
    return "\n".join(out)


def file_seconds(junit_xml):
    """seconds per test file from a pytest --junitxml file, longest first, as SECONDS lines (a missing file: a note)"""
    import xml.etree.ElementTree as ET
    if not os.path.exists(junit_xml):
        return f"no {junit_xml} (pytest did not run)"
    tot = {}
    for tc in ET.parse(junit_xml).iter("testcase"):
        f = tc.get("file") or "?"              # xunit1 (-o junit_family=xunit1) records each test's file
        tot[f] = tot.get(f, 0.0) + float(tc.get("time") or 0)
    return "\n".join(f'    "{f}": {round(s)},' for f, s in sorted(tot.items(), key=lambda kv: -kv[1]))


def _get(url):
    import urllib.request
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if attempt == 2:
                raise


def upstream_files(folder, fetch=_get, sums=None):
    """write UPSTREAM_FILES of UPSTREAM_MODEL, at its models.json upstream revision, into folder; stop (SystemExit,
    nothing written for that file) when a file's sha256 is not its pin"""
    import hashlib
    sums = UPSTREAM_FILES if sums is None else sums
    up = next(m["upstream"] for m in json.loads((ROOT / "models.json").read_text())["models"] if m["id"] == UPSTREAM_MODEL)
    os.makedirs(folder, exist_ok=True)
    for name, want in sorted(sums.items()):
        data = fetch(f"https://huggingface.co/{up['hf']}/resolve/{up['revision']}/{name}")
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit(f"{up['hf']}@{up['revision'][:8]} {name}: sha256 {got}, pinned {want}")
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(data)
    return folder


def changed_files(base, head):
    if not base or set(base) == {"0"}:
        return None                             # first push of a branch: no base to compare
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", f"{base}...{head or 'HEAD'}"], capture_output=True, text=True)
    return r.stdout.split() if r.returncode == 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--github-output", action="store_true", help="write matrix and any to $GITHUB_OUTPUT")
    ap.add_argument("--files-of", metavar="AREA", help="print the test files of AREA's T1 job --shard (default 1/1)")
    ap.add_argument("--args-of", metavar="AREA", help="print the pytest arguments of AREA's T1 job --shard, SPLIT files as parts")
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--whole-files", action="store_true", help="every file whole, no SPLIT parts (the sweep): with --args-of, "
                    "the arguments of one job; with --github-output, an area runs as at most one job per test file")
    ap.add_argument("--contracts-summary", metavar="CONTRACTS_JSON", help="print contracts.json as a table, failures first")
    ap.add_argument("--file-seconds", metavar="JUNIT_XML", help="print the seconds of each test file in a pytest junit file")
    ap.add_argument("--upstream-files", metavar="DIR", help=f"fetch the pinned {UPSTREAM_MODEL} tokenizer files into DIR")
    a = ap.parse_args(argv)
    if a.contracts_summary:
        print(contracts_summary(a.contracts_summary))
        return 0
    if a.file_seconds:
        print(file_seconds(a.file_seconds))
        return 0
    if a.upstream_files:
        print("upstream files in", upstream_files(a.upstream_files))
        return 0
    areas = model_areas()
    if a.args_of:
        args = shard_args(area_files(a.args_of, areas), a.shard, whole=a.whole_files)
        if not args:
            print(f"no test files for area {a.args_of} shard {a.shard}", file=sys.stderr)
            return 1
        print(" ".join(args))
        return 0
    if a.files_of:
        files = shard_files(area_files(a.files_of, areas), a.shard)
        if not files:
            print(f"no test files for area {a.files_of} shard {a.shard}", file=sys.stderr)
            return 1
        print(" ".join(files))
        return 0
    files = a.files if a.files is not None else (None if a.all else changed_files(a.base, a.head))
    if files is None:
        picked = select(["pyproject.toml"], areas)
        print("all areas (no base to compare, or --all)")
    else:
        picked = select(files, areas, a.base, a.head)
        for f in files:
            print(f"  {f}: {' '.join(sorted(classify(f, areas, a.base, a.head))) or 'T0 only'}")
    print("T1 areas:", " ".join(picked) or "none (T0 only)")
    if a.github_output:
        include = jobs(picked, whole=a.whole_files)
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"matrix={json.dumps({'include': include})}\nany={'true' if picked else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
