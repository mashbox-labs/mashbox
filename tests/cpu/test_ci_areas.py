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
        files, units = CS.area_files(area, areas), CS.units(CS.area_files(area, areas))
        n = CS.SHARDS.get(area, 1)
        sec = {u: CS.unit_seconds(u, CS.SECONDS) for u in units}
        dealt = [u for i in range(1, n + 1) for u in CS.shard_files(units, f"{i}/{n}", sec)]
        assert sorted(dealt) == sorted(units) and len(dealt) == len(set(dealt)), area
        assert all(CS.shard_files(units, f"{i}/{n}", sec) for i in range(1, n + 1)), f"{area}: an empty shard"
        args = [a for i in range(1, n + 1) for a in CS.shard_args(files, f"{i}/{n}")]
        assert sorted({a for a in args if not a.startswith("--")}) == sorted(files), area       # every file in a job
        for f, m in CS.SPLIT.items():                    # every part of a split file in exactly one job
            if f in files:
                got = sorted(a.split("=", 1)[1] for a in args if a.startswith(f"--ci-part={f}@"))
                assert got == [f"{f}@{k}/{m}" for k in range(1, m + 1)] or (got == [] and args.count(f) == 1), (f, got)
        rows = [r for r in CS.jobs([area]) if r["area"] == area]
        assert [r["shard"] for r in rows] == [f"{i}/{n}" for i in range(1, n + 1)], area
    every = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests" / "cpu").rglob("test_*.py") if p.name not in CS.T0_TESTS}
    assert every == {f for a in list(areas) + ["core"] for f in CS.area_files(a, areas)}   # every T1 file has an area


def test_t1_shards_split_an_area_by_time():
    """The shards of an area share its time (SECONDS, per test file, a SPLIT file's parts each a share): the slowest
    shard is at most its longest unit or a quarter over an even split, whichever is more, and stays under 60% of the
    T1 timeout, so no job runs close to it."""
    import re
    t1 = (ROOT / ".github" / "workflows" / "ci.yml").read_text().split("\n  t1:\n", 1)[1]
    limit = 60 * int(re.search(r"^    timeout-minutes: (\d+)$", t1, re.M).group(1))
    areas = CS.model_areas()
    for area, n in CS.SHARDS.items():
        files = CS.units(CS.area_files(area, areas))
        sec = {u: CS.unit_seconds(u, CS.SECONDS) for u in files}
        loads = [sum(sec[f] for f in CS.shard_files(files, f"{i}/{n}", sec)) for i in range(1, n + 1)]
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
    both = pick(reg, "both")                                       # core runs the rows readers in its area; a rows reader
    outside = {CS.test_area(f.rsplit("/", 1)[-1], areas)           # in another model's area (one that reads the catalog)
               for f in CS.area_files(CS.ROWS, areas)} - {model, "core", "gguf"}    # keeps the rows leg, which runs it
    assert set(both) == {model, "core", "gguf"} | ({CS.ROWS} if outside else set()), (both, outside)
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
        if (ROOT / "tests" / "cpu" / f).exists():                  # a held file is not in the public tree
            assert f"tests/cpu/{f}" in files, f
    for f in t1 - set(files):                                      # a file that reads the rows directly is in the leg
        if CS.test_area(f.rsplit("/", 1)[-1], areas) != "gguf":
            assert not re.search(r"""\[["']gguf["']\]|get\(["']gguf["']""", (ROOT / f).read_text()), f
    units = CS.units(files)                                        # its jobs (SHARDS, parts of SPLIT files) each finish
    sec = {u: CS.unit_seconds(u, CS.SECONDS) for u in units}       # in 300 s by the runner's seconds per file
    n = CS.SHARDS.get(CS.ROWS, 1)
    loads = [sum(sec[u] for u in CS.shard_files(units, f"{i}/{n}", sec)) for i in range(1, n + 1)]
    assert max(loads) <= 300, loads
    assert CS.jobs(["gguf", "gguf_rows"]) == [{"area": "gguf", "extra": "", "shard": "1/1"}] + [
        {"area": "gguf_rows", "extra": "", "shard": f"{i}/{n}"} for i in range(1, n + 1)]


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


def _collect(*args):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.returncode, sorted(x for x in r.stdout.splitlines() if "::" in x)


def test_a_split_file_runs_each_test_in_exactly_one_part_and_each_script_in_one_part():
    """tests/conftest.py --ci-part: the parts of a SPLIT file partition its tests (none dropped, none twice), the
    qwen3_8_27b tests of one gate script stay in one part (the script runs once per part that needs it), and a part
    that holds no test, or a bad K/M, is a usage error (exit 4), never a green job that ran nothing."""
    f = "tests/cpu/test_qwen3_8_27b.py"
    m = CS.SPLIT[f]
    rc, every = _collect(f)
    assert rc == 0 and every
    parts = []
    for k in range(1, m + 1):
        rc, got = _collect(f, f"--ci-part={f}@{k}/{m}")
        assert rc == 0 and got, k
        parts.append(got)
    dealt = [t for p in parts for t in p]
    assert sorted(dealt) == every and len(dealt) == len(set(dealt))
    for s in ("prefix_cache_checks.py", "continuous_batching_checks.py", "engine_batch_checks.py", "determinism_checks.py"):
        assert sum(any(s in t for t in p) for p in parts) == 1, s
    assert _collect(f, f"--ci-part={f}@{m + 1}/{m}")[0] == 4
    assert _collect(f, f"--ci-part={f}@{m + 3}/{m + 3}")[0] == 4      # more parts than the file has groups


def test_timing_tests_run_in_the_sweep_and_never_gate_a_pull_request():
    """A test that passes or fails on wall-clock time (marker timing) flakes with box speed, so PR CI's T1 deselects it
    and the sweep (nightly, by hand, every v* tag) runs it with every other T1 test; both run the same shards
    (--args-of). The one timing test is the "merge": false latency gate; a new one is added here on purpose."""
    import re
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    sw = (ROOT / ".github" / "workflows" / "sweep.yml").read_text()
    pr_m = re.findall(r'-m "([^"]+)"', ci.split("\n  t1:\n", 1)[1].split("\n  t1-gate:", 1)[0])
    sw_m = re.findall(r'-m "([^"]+)"', sw)
    assert pr_m == ["$AREA and not t0 and not quarantine and not timing"], pr_m
    assert sw_m == ["$AREA and not t0 and not quarantine"], sw_m
    assert "--args-of" in ci and "--args-of" in sw and "--all --github-output" in sw
    assert "schedule:" in sw and "workflow_dispatch:" in sw and 'tags: ["v*"]' in sw and "pull_request" not in sw
    marked = sorted(q.relative_to(ROOT).as_posix() for q in (ROOT / "tests").rglob("*.py")
                    if "mark.timing" in q.read_text() and q.name != "test_ci_areas.py")
    assert marked == ["tests/cpu/test_qwen3_8_27b.py"], marked
    rc, timing = _collect(marked[0], "-m", "timing")
    assert rc == 0 and timing == ["tests/cpu/test_qwen3_8_27b.py::test_gate[engine_batch_checks.py-E33]"], timing


def test_file_seconds_sums_a_junit_file_per_test_file(tmp_path):
    """--file-seconds turns a leg's junit file into SECONDS lines (longest first) so the shard table is refreshed from
    real CI times; a missing file says so instead of failing the step."""
    x = tmp_path / "t1.xml"
    x.write_text('<testsuites><testsuite><testcase file="tests/cpu/test_a.py" time="1.5"/>'
                 '<testcase file="tests/cpu/test_a.py" time="2"/><testcase file="tests/cpu/test_b.py" time="9"/>'
                 '</testsuite></testsuites>')
    assert CS.file_seconds(str(x)).splitlines() == ['    "tests/cpu/test_b.py": 9,', '    "tests/cpu/test_a.py": 4,']
    assert "did not run" in CS.file_seconds(str(tmp_path / "none.xml"))


def test_contracts_job_runs_every_contract_on_every_pull_request():
    """The contracts job runs tests/contracts on every pull request with no area pick (no `if:`, no `needs:`), CPU
    only, no torch and no model download, inside 10 minutes, and writes contracts.json to the summary; a change to
    tests/contracts alone runs T0 and that job, not every T1 area."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = ci.split("\n  contracts:\n", 1)[1].split("\n\n  ", 1)[0]
    assert "\n    if:" not in job and "\n    needs:" not in job
    assert "python3 -m pytest tests/contracts" in job and "MASHBOX_CONTRACTS_OUT" in job and "--contracts-summary" in job
    install = [x for x in job.splitlines() if "pip install" in x]
    assert install and not any("torch" in x or "[dev]" in x for x in install), install
    assert "hf_hub_download" not in job and "HF_TOKEN" not in job and "secrets." not in job
    assert int(job.split("timeout-minutes: ", 1)[1].split()[0]) <= 10
    areas = CS.model_areas()
    assert CS.select(["tests/contracts/test_x.py"], areas) == []


def test_contracts_summary_lists_each_contract(tmp_path):
    """--contracts-summary prints one table row per contract id with its status and fingerprint, failures first."""
    import json
    f = tmp_path / "contracts.json"
    f.write_text(json.dumps({"schema": 1, "code_commit": "abc", "dirty": False, "contracts": {
        "tok/qwen2": {"status": "PASS", "fingerprint": "f" * 64, "cases": 3},
        "op/Q4_K": {"status": "FAIL", "fingerprint": "0" * 64, "cases": 2}}}))
    out = CS.contracts_summary(str(f))
    rows = [x for x in out.splitlines() if x.startswith("| ") and "/" in x.split("|")[1]]
    assert [r.split("|")[1].strip() for r in rows] == ["op/Q4_K", "tok/qwen2"], out
    assert "1 FAIL" in out and "1 PASS" in out
    assert "no contracts.json" in CS.contracts_summary(str(tmp_path / "none.json"))


def test_the_official_client_tests_run_in_their_own_leg_with_the_pinned_clients_and_tokenizer(tmp_path):
    """test_api_conformance.py and test_api_loaded_clients.py drive the server through the official openai and anthropic
    clients at the versions they pin, and skip without them; the conformance file also skips without the Qwen3.8-27B
    tokenizer and chat template (MASHBOX_QWEN_UPSTREAM). A skip there reads as green, so on CI they never ran. Their own
    T1 leg installs exactly the pinned clients and fetches the pinned upstream files (checked by sha256, a wrong byte
    stops the leg), every change that runs core runs it too, and the two files leave the core shards (each file in one
    job). The sweep runs the same leg the same way."""
    import hashlib
    import re
    areas = CS.model_areas()
    files = ["tests/cpu/test_api_conformance.py", "tests/cpu/test_api_loaded_clients.py"]
    assert CS.area_files(CS.CLIENTS, areas) == files
    assert not set(files) & set(CS.area_files("core", areas))
    for f in files:                                                # the leg installs the very versions each file pins
        oa, an = re.search(r'!= \("(\d[\w.]*)", "(\d[\w.]*)"\)', (ROOT / f).read_text()).groups()
        assert sorted(CS.extra(CS.CLIENTS).split()) == [f"anthropic=={an}", f"openai=={oa}"], f
    assert CS.CLIENTS in CS.select(["src/mashbox/runtime/server.py"], areas)
    assert CS.CLIENTS in CS.select(["pyproject.toml"], areas)
    assert CS.select(["tests/cpu/test_api_loaded_clients.py"], areas) == [CS.CLIENTS]
    assert CS.CLIENTS in {r["area"] for r in CS.jobs(CS.select(["src/mashbox/cli.py"], areas))}
    # the upstream files: the revision models.json pins for the model, the three files runtime/chat.py reads
    up = next(m["upstream"] for m in CS.json.loads((ROOT / "models.json").read_text())["models"] if m["id"] == CS.UPSTREAM_MODEL)
    assert set(CS.UPSTREAM_FILES) == {"tokenizer.json", "chat_template.jinja", "generation_config.json"}
    chat = (ROOT / "src" / "mashbox" / "runtime" / "chat.py").read_text()
    assert all(f'"{n}"' in chat for n in CS.UPSTREAM_FILES)
    good = {n: n.encode() * 3 for n in CS.UPSTREAM_FILES}
    sums = {n: hashlib.sha256(b).hexdigest() for n, b in good.items()}
    asked = []

    def fetch(url):
        asked.append(url)
        return good[url.rsplit("/", 1)[1]]
    CS.upstream_files(str(tmp_path / "ok"), fetch=fetch, sums=sums)
    assert sorted(asked) == sorted(f"https://huggingface.co/{up['hf']}/resolve/{up['revision']}/{n}" for n in CS.UPSTREAM_FILES)
    assert {p.name: p.read_bytes() for p in (tmp_path / "ok").iterdir()} == good
    bad = dict(good, **{"chat_template.jinja": b"{{ changed }}"})
    try:
        CS.upstream_files(str(tmp_path / "bad"), fetch=lambda u: bad[u.rsplit("/", 1)[1]], sums=sums)
    except SystemExit as e:
        assert "chat_template.jinja" in str(e), e
    else:
        raise AssertionError("a file whose sha256 differs from its pin was accepted")
    assert not (tmp_path / "bad" / "chat_template.jinja").exists()
    # both workflows fetch the files for this leg and hand the folder to its tests
    for wf in ("ci.yml", "sweep.yml"):
        t1 = (ROOT / ".github" / "workflows" / wf).read_text().split("\n  t1:\n", 1)[1]
        step = re.search(r"if: matrix\.area == '([a-z_]+)'\n\s+run: \|\n((?:\s{10}.*\n)+)", t1)
        assert step and step.group(1) == CS.CLIENTS, wf
        assert "tools/ci_select.py --upstream-files" in step.group(2) and "MASHBOX_QWEN_UPSTREAM=" in step.group(2), wf
        assert t1.index("--upstream-files") < t1.index("python3 -m pytest"), wf


def test_tp_batched_runs_as_parts_in_pr_ci_and_whole_in_the_sweep():
    """test_tp_batched.py's one script run outlasts a core shard, so in PR CI it runs as SPLIT parts (each part runs the
    script with TP_ONLY = its gates). The sweep runs every file whole (--whole-files: no parts), the way a local pytest
    run does, so the all-gates run of one process is still checked nightly."""
    import re
    areas = CS.model_areas()
    f = "tests/cpu/test_tp_batched.py"
    assert CS.SPLIT.get(f, 1) >= 2
    files = CS.area_files("core", areas)
    n = CS.SHARDS["core"]
    whole = [a for i in range(1, n + 1) for a in CS.shard_args(files, f"{i}/{n}", whole=True)]
    assert not [a for a in whole if a.startswith("--")] and sorted(whole) == sorted(files)     # each file once, whole
    parts = [a for i in range(1, n + 1) for a in CS.shard_args(files, f"{i}/{n}")]
    assert sorted(a for a in parts if a.startswith(f"--ci-part={f}@")) == [f"--ci-part={f}@{k}/{CS.SPLIT[f]}"
                                                                          for k in range(1, CS.SPLIT[f] + 1)]
    for wf, flag in (("ci.yml", False), ("sweep.yml", True)):
        t1 = (ROOT / ".github" / "workflows" / wf).read_text().split("\n  t1:\n", 1)[1]
        call = re.search(r"ci_select\.py --args-of [^\n]*", t1).group(0)
        assert ("--whole-files" in call) == flag, (wf, call)


def test_pr_ci_skips_the_timing_gates_runs_and_the_sweep_runs_them():
    """Deselecting a timing test does not stop its gate script from doing that gate's runs (the wall-clock gate's warm pairs
    and 8 runs at ~1000 tokens inside engine_batch_checks.py held the longest T1 leg). PR CI's T1 step sets MASHBOX_SKIP_TIMING=1, every
    script with a TIMING gate reads it, and the sweep never sets it, so the sweep still runs and checks those gates."""
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location("tq", ROOT / "tests" / "cpu" / "test_qwen3_8_27b.py")
    src = spec.loader.get_source("tq")
    timing = eval(re.search(r"^TIMING = (\{.*\})$", src, re.M).group(1))
    assert timing
    for script, _ in timing:
        assert 'os.environ.get("MASHBOX_SKIP_TIMING") == "1"' in (ROOT / "tests" / "cpu" / script).read_text(), script
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text().split("\n  t1:\n", 1)[1].split("\n  t1-gate:\n", 1)[0]
    step = ci.split("- name: CPU tests of", 1)[1].split("\n      - ", 1)[0]
    assert re.search(r'^\s+MASHBOX_SKIP_TIMING: "1"', step, re.M) and '-m "$AREA and not t0 and not quarantine and not timing"' in step
    assert "MASHBOX_SKIP_TIMING" not in (ROOT / ".github" / "workflows" / "sweep.yml").read_text()
