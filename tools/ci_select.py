#!/usr/bin/env python3
"""Pick the CPU test areas (tier T1) a change needs. Standard library only; CI.md explains the tiers.

    python3 tools/ci_select.py --base <sha> [--head <sha>]    # the files changed between two commits
    python3 tools/ci_select.py --files src/mashbox/cli.py      # an explicit list
    python3 tools/ci_select.py --all

Areas are one per models.json row (the id with underscores, e.g. hy_nomas) plus `core`:
- a model area is its port folder, apps/<id>/, its own models.json row and the tests whose file names carry the id
- core (the shared package, pyproject.toml, test config, workflows, tools) runs every area, because it touches every model
- docs, licences, cards, templates and tests/tpu run T0 only
- a path nothing above claims runs every area, so a new kind of file is never silently untested
tests/conftest.py uses test_area() to mark each test with its area, so the same rule picks the tests.
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
T0_ONLY_DIRS = ("docs/", "licenses/", "hf/", "templates/", "tests/tpu/", ".github/ISSUE_TEMPLATE/")
T0_ONLY_FILES = {"llms.txt", "LICENSE", "LICENSE-KERNELS", ".github/pull_request_template.md", "tests/cpu/test_core.py"}
CORE_DIRS = ("src/mashbox/runtime/", ".github/workflows/", "tools/")
CORE_FILES = {"pyproject.toml", "tests/conftest.py"}


def model_areas(text=None):
    reg = json.loads(text if text is not None else (ROOT / "models.json").read_text())
    return [m["id"].replace("-", "_") for m in reg["models"]]


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
    return {"core"} if changed - new.keys() else {i.replace("-", "_") for i in changed}


def classify(path, areas, base=None, head=None):
    """The areas one changed path selects: a set of area names, {'core'}, or an empty set (T0 only)."""
    if path in T0_ONLY_FILES or path.startswith(T0_ONLY_DIRS) or path.endswith(".md"):
        return set()
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
        picked = set(areas) | ({"core"} if core_has_t1_tests(areas) else set())
    return sorted(picked)


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
    a = ap.parse_args(argv)
    areas = model_areas()
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
        include = [{"area": x, "extra": EXTRA.get(x, "")} for x in picked]
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"matrix={json.dumps({'include': include})}\nany={'true' if picked else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
