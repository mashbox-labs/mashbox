"""tools/ci_select.py: the GGUF reader, its ops, fixtures and test_gguf_*.py are their own CI area (`gguf`), so a GGUF
change runs the GGUF tests and a core change runs them too; nothing GGUF is left to an area that never runs it."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ci_select", ROOT / "tools" / "ci_select.py")
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)


def test_gguf_is_an_area_with_its_files_and_tests():
    areas = CS.model_areas()
    assert "gguf" in areas
    tests = sorted(p.name for p in (ROOT / "tests" / "cpu").glob("test_gguf_*.py"))
    assert tests and all(CS.test_area(t, areas) == "gguf" for t in tests)
    for p in ("src/mashbox/gguf/reader.py", "src/mashbox/gguf/archs/dense.py", "src/mashbox/ops/__init__.py",
              "tools/gguf_check.py", "tools/gguf_fixtures/fixtures.py", "tests/fixtures/gguf/qwen3-f32.json"):
        assert CS.select([p], areas) == ["gguf"], p
    assert "gguf" in CS.select(["src/mashbox/cli.py"], areas)            # core changes run the GGUF tests too
    # cli.main imports gguf/__init__.py and gguf/resolve.py on every run, for every model: a change to them runs every area
    for p in ("src/mashbox/gguf/__init__.py", "src/mashbox/gguf/resolve.py"):
        assert set(CS.select([p], areas)) >= {"gguf", "hy_nomas", "qwen3_8_27b"}, p
    assert "gguf" in CS.select(["src/mashbox/runtime/engine.py"], areas)


def test_t1_shards_run_every_file_of_an_area_exactly_once():
    """A sharded area (SHARDS) runs as several T1 jobs; a file dealt to no job is never tested and a file dealt twice
    wastes runner time, so the shards of every area partition its test files, and the matrix has one row per shard."""
    areas = CS.model_areas()
    for area in list(areas) + ["core"]:
        files = CS.area_files(area, areas)
        n = CS.SHARDS.get(area, 1)
        dealt = [f for i in range(1, n + 1) for f in CS.shard_files(files, f"{i}/{n}")]
        assert sorted(dealt) == sorted(files) and len(dealt) == len(set(dealt)), area
        assert all(CS.shard_files(files, f"{i}/{n}") for i in range(1, n + 1)), f"{area}: an empty shard"
        rows = [r for r in CS.jobs([area]) if r["area"] == area]
        assert [r["shard"] for r in rows] == [f"{i}/{n}" for i in range(1, n + 1)], area
    every = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests" / "cpu").rglob("test_*.py") if p.name not in CS.T0_TESTS}
    assert every == {f for a in list(areas) + ["core"] for f in CS.area_files(a, areas)}   # every T1 file has an area


def test_t1_shards_split_an_area_by_time():
    """The shards of an area share its time (SECONDS, per test file): the slowest shard is at most its longest file or
    a quarter over an even split, whichever is more, and stays under 60% of the T1 timeout, so no job runs close to it."""
    import re
    t1 = (ROOT / ".github" / "workflows" / "ci.yml").read_text().split("\n  t1:\n", 1)[1]
    limit = 60 * int(re.search(r"^    timeout-minutes: (\d+)$", t1, re.M).group(1))
    areas = CS.model_areas()
    for area, n in CS.SHARDS.items():
        files = CS.area_files(area, areas)
        sec = {f: CS.SECONDS.get(f, CS.SECONDS_DEFAULT) for f in files}
        loads = [sum(sec[f] for f in CS.shard_files(files, f"{i}/{n}")) for i in range(1, n + 1)]
        bound = max(max(sec.values()), 1.25 * sum(loads) / n)
        assert max(loads) <= bound, f"{area}: shard seconds {loads}, bound {bound:.0f}"
        assert max(loads) <= 0.6 * limit, f"{area}: slowest shard {max(loads)} s of a {limit} s T1 timeout"


def test_views_workflow_checks_and_writes_nothing(tmp_path):
    """The Views workflow renders every view of models.json into a temporary folder and fails when a committed view
    differs from its render or is missing. It holds no secret, asks for no write permission, keeps no credential in its
    checkout and commits or pushes nothing, so it runs the same on a pull request from a fork."""
    import re
    import subprocess
    wf = (ROOT / ".github" / "workflows" / "views.yml").read_text()
    code = "".join(line + "\n" for line in wf.splitlines() if not line.lstrip().startswith("#"))
    assert "secrets." not in code and "ssh-key" not in code and "pull_request_target" not in code
    assert not re.search(r"\bgit\b[^\n]*\b(add|commit|push)\b", code)
    assert not re.search(r"python3 [^\n]*hf_github_sync_check\.py[^\n]*--write", code)   # it never renders in place
    levels = re.findall(r"^\s+[a-z-]+: (read|write|none)$", code, re.M)
    assert levels and "write" not in levels, levels
    checkouts = code.count("uses: actions/checkout@")
    assert checkouts and code.count("persist-credentials: false") == checkouts
    assert re.search(r'hf_github_sync_check\.py render --repo \. --out "\$RUNNER_TEMP/views"', code)
    step = re.search(r"\n( +)- name: Fail when a committed view differs from its render\n +run: \|\n((?:(?:\1    .*)?\n)+)", code)
    assert step, "no compare step"
    script = "\n".join(line[len(step.group(1)) + 4:] for line in step.group(2).splitlines())
    assert script.rstrip().endswith('exit "$bad"'), script   # the whole step, to its last line

    def compare(case, rendered, committed):
        out, ws = tmp_path / case / "views", tmp_path / case / "ws"
        for base, files in ((out, rendered), (ws, committed)):
            for rel, text in files.items():
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                (base / rel).write_text(text)
        env = {"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path / case), "GITHUB_WORKSPACE": str(ws)}
        return subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)

    views = {"README.md": "models\n", "docs/models/a.md": "a\n", "hf/a/README.md": "card\n"}
    same = compare("same", views, dict(views, **{"docs/other.md": "not a view\n"}))
    assert same.returncode == 0, same.stdout + same.stderr
    drift = compare("drift", views, dict(views, **{"docs/models/a.md": "a, edited by hand\n"}))
    assert drift.returncode != 0 and "docs/models/a.md" in drift.stdout and "README.md" not in drift.stdout
    gone = {k: v for k, v in views.items() if k != "hf/a/README.md"}
    missing = compare("missing", views, gone)
    assert missing.returncode != 0 and "hf/a/README.md" in missing.stdout


def test_t1_gate_is_green_only_when_every_picked_leg_passed(tmp_path):
    """Branch protection requires one T1 check, the `t1` gate, not one check per leg: a change that picks no area (docs,
    licences, a registry version) skips the legs, and a required leg that never reports would hold the pull request
    forever. The gate runs always, after T0 and the legs: green when every picked leg passed or none was picked; red when
    T0 failed, a picked leg failed, was cancelled or did not run, or the pick is missing."""
    import re
    import subprocess
    wf = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = re.search(r"\n  (t1-gate):\n((?:(?:    .*)?\n)+)", wf)
    assert job, "no t1-gate job"
    body = job.group(2)
    assert re.search(r"^    name: t1$", body, re.M)
    assert re.search(r"^    needs: \[t0, t1\]$", body, re.M) and re.search(r"^    if: always\(\)$", body, re.M)
    for var, expr in (("T0", "needs.t0.result"), ("PICKED", "needs.t0.outputs.any"), ("T1", "needs.t1.result")):
        assert re.search(rf"^ +{var}: \$\{{\{{ {re.escape(expr)} \}}\}}$", body, re.M), var
    step = re.search(r"\n( +)- name: Every picked T1 leg passed\n(?: +env:\n(?: +[A-Z0-9]+: .*\n)+)? +run: \|\n((?:(?:\1    .*)?\n)+)",
                     body + "\n")
    assert step, "no gate step"
    script = "\n".join(line[len(step.group(1)) + 4:] for line in step.group(2).splitlines())

    def gate(t0, picked, t1):
        env = {"PATH": "/usr/bin:/bin", "T0": t0, "PICKED": picked, "T1": t1}
        return subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True).returncode

    assert gate("success", "false", "skipped") == 0          # T0-only change: no leg picked, none ran
    assert gate("success", "true", "success") == 0           # every picked leg passed
    for t1 in ("failure", "cancelled", "skipped", ""):       # a picked leg failed, was cancelled or never ran
        assert gate("success", "true", t1) != 0, t1
    assert gate("success", "false", "success") != 0          # legs ran although none was picked: the pick is wrong
    assert gate("failure", "", "skipped") != 0               # red T0 (no pick)
    assert gate("success", "", "skipped") != 0               # T0 passed but gave no pick
    assert gate("cancelled", "true", "cancelled") != 0


def _registry_commits(tmp_path):
    """A git repo holding models.json at five commits: the real registry, then a change to one gguf[] row, to one
    models[] row, to both, and to a top-level field only."""
    import copy
    import json
    import subprocess
    reg = json.loads((ROOT / "models.json").read_text())
    assert reg["gguf"] and reg["models"]

    def edit(rows=False, model=False, top=False):
        r = copy.deepcopy(reg)
        if rows:
            r["gguf"][0]["bytes"] = r["gguf"][0]["bytes"] + 1
            r["gguf"].append(dict(r["gguf"][0], id=r["gguf"][0]["id"] + "-x"))
        if model:
            r["models"][0]["summary"] = r["models"][0]["summary"] + " x"
        if top:
            r["version"] = r["version"] + ".1"
        return r

    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    shas = {}
    for name, r in (("base", reg), ("rows", edit(rows=True)), ("model", edit(model=True)),
                    ("both", edit(rows=True, model=True)), ("top", edit(top=True))):
        (tmp_path / "models.json").write_text(json.dumps(r, indent=1) + "\n")
        subprocess.run(git + ["add", "models.json"], check=True)
        subprocess.run(git + ["commit", "-q", "-m", name], check=True)
        shas[name] = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return shas


def test_a_change_to_gguf_rows_only_runs_the_gguf_tests_and_the_tests_that_read_the_rows(tmp_path, monkeypatch):
    """A diff that changes only gguf[] rows (models.json and its package copy, plus the views rendered from them and the
    docs-lint allow counts) runs T0, the gguf area and one leg of the tests that read the rows (the catalog, model lists,
    docs and card sync, the speculation records); it does not wait on every model's tests. Any other change keeps the
    pick it has: a models[] row runs that model and core, a code file runs every area. Before this rule a gguf-row change
    ran T0 alone (the rows were read as top-level fields) and the allow counts under tools/ ran every area."""
    areas = CS.model_areas()
    shas = _registry_commits(tmp_path)
    (tmp_path / "tests").symlink_to(ROOT / "tests")                # the rules read the real tests; models.json is the repo's
    monkeypatch.setattr(CS, "ROOT", tmp_path)
    every = sorted(set(areas) | {"core"})
    reg = ["models.json", "src/mashbox/models.json"]
    views = ["README.md", "docs/supported-models.md", "docs/gguf.md", "llms.txt", "tools/docs_lint_allow.txt"]
    pick = lambda files, head: CS.select(files, areas, shas["base"], shas[head])
    assert pick(reg, "rows") == ["gguf", "gguf_rows"]
    assert pick(reg + views, "rows") == ["gguf", "gguf_rows"]
    model = CS.model_areas()[0]                                    # models[0] is the row the "model" commit changed
    assert pick(reg, "model") == sorted({model, "core"})           # a models[] row: that model plus core, as before
    assert pick(reg + views, "model") == sorted({model, "core"})
    both = pick(reg, "both")                                       # core runs every rows reader, so no extra leg
    assert set(both) == {model, "core", "gguf"}, both
    assert pick(reg, "top") == []                                  # top-level fields feed the views only (T0)
    for code in ("src/mashbox/cli.py", "src/mashbox/catalog.py", "tools/gguf_rows.py", "tools/hf_github_sync_check.py",
                 "tests/cpu/test_catalog.py", "some/new/kind/of/file.cfg"):
        assert pick(reg + views + [code], "rows") == every, code   # rows plus any other file: the full pick
    assert CS.select(["tools/docs_lint_allow.txt"], areas) == []    # only T0's docs lint reads the allow counts


def test_the_rows_leg_holds_every_test_that_reads_the_gguf_rows():
    """The gguf_rows leg runs the T1 tests outside the gguf area that read gguf[] rows: directly, through the catalog,
    model lists, card and docs sync, the GGUF row tool, the speculation records, the `models` command or a helper that
    does; each file once, none from the gguf area (its own leg runs those), and fast enough to finish before the gguf
    leg does on a CI runner."""
    import re
    areas = CS.model_areas()
    files = CS.area_files("gguf_rows", areas)
    assert files == sorted(set(files), key=lambda f: (f.rsplit("/", 1)[-1], f))
    t1 = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests" / "cpu").rglob("test_*.py") if p.name not in CS.T0_TESTS}
    assert set(files) <= t1
    assert not [f for f in files if CS.test_area(f.rsplit("/", 1)[-1], areas) == "gguf"]
    for f in ("test_catalog.py", "test_model_lists.py", "test_docs_supported_models.py", "test_org_card.py",
              "test_ci_areas.py", "test_port_plan.py", "test_spec_page.py", "test_drafter_gate.py", "test_drafters.py"):
        assert f"tests/cpu/{f}" in files, f
    for f in t1 - set(files):                                      # a file that reads the rows directly is in the leg
        if CS.test_area(f.rsplit("/", 1)[-1], areas) != "gguf":
            assert not re.search(r"""\[["']gguf["']\]|get\(["']gguf["']""", (ROOT / f).read_text()), f
    assert sum(CS.SECONDS.get(f, CS.SECONDS_DEFAULT) for f in files) <= 300
    assert CS.jobs(["gguf", "gguf_rows"]) == [{"area": "gguf", "extra": "", "shard": "1/1"},
                                              {"area": "gguf_rows", "extra": "", "shard": "1/1"}]


def test_conftest_marks_the_rows_leg_tests():
    """The leg runs `pytest <its files> -m "gguf_rows and not t0 ..."`, so conftest gives each of its tests the
    gguf_rows mark (beside the area mark) and no other test; a file outside the leg collects nothing under that mark."""
    import subprocess
    import sys
    run = lambda f: subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                                    "-m", "gguf_rows and not t0", f], cwd=ROOT, capture_output=True, text=True)
    r = run("tests/cpu/test_ci_areas.py")
    assert r.returncode == 0 and "test_the_rows_leg_holds_every_test_that_reads_the_gguf_rows" in r.stdout, r.stdout[-800:]
    r = run("tests/cpu/test_sampler.py")
    assert "test_" not in r.stdout.split("deselected")[0].split("no tests ran")[0] or r.returncode == 5, r.stdout[-800:]
