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
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
T0_TESTS = {"test_core.py"}                     # the T0 suite: static and registry checks, pytest only
EXTRA = {"hy_nomas": "imageio-ffmpeg"}          # pip packages an area's tests need beyond .[dev] (the app server test)
T0_ONLY_DIRS = ("docs/", "licenses/", "hf/", "templates/", ".github/ISSUE_TEMPLATE/")
CORE_TESTS = "+core"                            # run the core T1 job without expanding to every model
T0_ONLY_FILES = {"llms.txt", "LICENSE", "LICENSE-KERNELS", ".github/pull_request_template.md", "tests/cpu/test_core.py"}
CORE_DIRS = ("src/mashbox/runtime/", ".github/workflows/", "tools/")
CORE_FILES = {"pyproject.toml", "tests/conftest.py"}
SHARDS = {"core": 4}                            # T1 jobs per area (default 1); core is 55 test files, about 870 tests
# Approximate seconds per T1 test file on a CI runner, used only to split an area's files into shards of even time.
# A file missing here counts as SECONDS_DEFAULT; a name here with no file is ignored. Refresh it from a CI run's
# per-file times when a shard runs long. Only files this repository ships are listed (a name here ships with it).
SECONDS_DEFAULT = 5
SECONDS = {
    "tests/cpu/test_activation_edits.py": 47,
    "tests/cpu/test_activation_edits_contract.py": 0,
    "tests/cpu/test_admission_fairness.py": 44,
    "tests/cpu/test_agent_api.py": 0,
    "tests/cpu/test_aot_table.py": 350,
    "tests/cpu/test_aot_table_first_use.py": 98,
    "tests/cpu/test_aot_table_prune.py": 78,
    "tests/cpu/test_aot_table_restart.py": 36,
    "tests/cpu/test_aot_table_startup.py": 140,
    "tests/cpu/test_api_conformance.py": 4,
    "tests/cpu/test_api_loaded_clients.py": 295,
    "tests/cpu/test_app_addon_tpu.py": 0,
    "tests/cpu/test_app_addons.py": 3,
    "tests/cpu/test_app_catalog_sig.py": 1,
    "tests/cpu/test_app_module.py": 5,
    "tests/cpu/test_app_quit.py": 2,
    "tests/cpu/test_app_security.py": 11,
    "tests/cpu/test_app_tunnel_guard.py": 13,
    "tests/cpu/test_catalog.py": 88,
    "tests/cpu/test_chat_module.py": 1,
    "tests/cpu/test_chat_relay_guard.py": 2,
    "tests/cpu/test_ci_areas.py": 0,
    "tests/cpu/test_compile_key.py": 0,
    "tests/cpu/test_docs_supported_models.py": 1,
    "tests/cpu/test_drafter_gate.py": 0,
    "tests/cpu/test_engine_model_hooks.py": 4,
    "tests/cpu/test_engine_warm.py": 53,
    "tests/cpu/test_front_admission.py": 4,
    "tests/cpu/test_front_failures.py": 105,
    "tests/cpu/test_guard.py": 0,
    "tests/cpu/test_hub_compile_cache.py": 0,
    "tests/cpu/test_metrics.py": 16,
    "tests/cpu/test_module_shell.py": 0,
    "tests/cpu/test_moe_routed.py": 69,
    "tests/cpu/test_openapi.py": 1,
    "tests/cpu/test_prefix_cache_owner.py": 0,
    "tests/cpu/test_sampler.py": 1,
    "tests/cpu/test_serve_batching.py": 11,
    "tests/cpu/test_serve_queue_api.py": 12,
    "tests/cpu/test_serve_security.py": 2,
    "tests/cpu/test_spec_in_step.py": 926,
    "tests/cpu/test_stop_strings.py": 18,
    "tests/cpu/test_subagents.py": 0,
    "tests/cpu/test_tenancy.py": 4,
    "tests/cpu/test_tp_batched.py": 521,
    "tests/cpu/test_tpu_chip_safety.py": 53,
    "tests/cpu/test_tpu_cli.py": 71,
    "tests/cpu/test_worker_protocol.py": 1,
}


AREAS_EXTRA = ("gguf",)                         # areas that are not a models.json row: GGUF files (reader, ops, kernels)
GGUF_DIRS = ("src/mashbox/gguf/", "src/mashbox/ops/", "tools/gguf_fixtures/", "tests/fixtures/gguf/")
GGUF_FILES = {"tools/gguf_check.py"}
# cli.main imports these two on every run (is_gguf_request, the GGUF dispatch), so a change to them runs every area
GGUF_CORE_FILES = {"src/mashbox/gguf/__init__.py", "src/mashbox/gguf/resolve.py"}


def model_areas(text=None):
    reg = json.loads(text if text is not None else (ROOT / "models.json").read_text())
    return [m["id"].replace("-", "_") for m in reg["models"]] + list(AREAS_EXTRA)


def test_area(file_name, areas):
    """tests/cpu/test_hy_nomas_port.py -> hy_nomas; a file name that carries no model id -> core."""
    stem = file_name[:-3] if file_name.endswith(".py") else file_name
    stem = stem[5:] if stem.startswith("test_") else stem
    for a in sorted(areas, key=len, reverse=True):
        if stem == a or stem.startswith(a + "_"):
            return a
    return "core"


def registry_areas(base, head):
    """Areas whose models.json rows differ between two commits; top-level fields only feed the views (T0)."""
    def rows(rev):
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:models.json"], capture_output=True, text=True)
        return {m["id"]: m for m in json.loads(r.stdout)["models"]} if r.returncode == 0 else None
    old, new = rows(base), rows(head) if head else {m["id"]: m for m in json.loads((ROOT / "models.json").read_text())["models"]}
    if old is None or new is None:
        return {"core"}
    changed = {i for i in old.keys() | new.keys() if old.get(i) != new.get(i)}
    if changed - new.keys():
        return {"core"}
    return {i.replace("-", "_") for i in changed} | ({CORE_TESTS} if changed else set())


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
    return sorted(picked)


def area_files(area, areas, root=ROOT):
    """The T1 test files of one area: tests/cpu/**/test_*.py whose name gives that area (T0 files excluded), name order"""
    cpu = root / "tests" / "cpu"
    return sorted((p.relative_to(root).as_posix() for p in cpu.rglob("test_*.py")
                   if p.name not in T0_TESTS and test_area(p.name, areas) == area), key=lambda f: (f.rsplit("/", 1)[-1], f))


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


def jobs(picked):
    """the T1 matrix rows: one per area, or SHARDS[area] rows for a sharded area"""
    return [{"area": x, "extra": EXTRA.get(x, ""), "shard": f"{i}/{SHARDS.get(x, 1)}"}
            for x in picked for i in range(1, SHARDS.get(x, 1) + 1)]


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
    ap.add_argument("--shard", default="1/1")
    a = ap.parse_args(argv)
    areas = model_areas()
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
        include = jobs(picked)
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"matrix={json.dumps({'include': include})}\nany={'true' if picked else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
