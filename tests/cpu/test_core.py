"""Checks of the shared core: the registry, its packaged copy, the generated views, imports and the CLI."""
import ast
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALLOWED = {"numpy", "jax", "jaxlib", "ml_dtypes", "tokenizers", "jinja2", "huggingface_hub", "mashbox"}
IMPORT_NAME = {"pyyaml": "yaml", "imageio-ffmpeg": "imageio_ffmpeg", "pillow": "PIL"}     # pip name -> import name


def extras():
    """[project.optional-dependencies] of pyproject.toml as {extra: [import names]} (each value is a JSON-style list)."""
    out, inside = {}, False
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.startswith("["):
            inside = line.strip() == "[project.optional-dependencies]"
        elif inside and "=" in line:
            k, v = line.split("=", 1)
            pkgs = [re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in json.loads(v)]
            out[k.strip()] = {IMPORT_NAME.get(p, p.replace("-", "_")) for p in pkgs}
    return out


def model_allowed(py, ex):
    """Extra imports for a file inside a model package: the pyproject extra named after the model id, and the top-level
    packages the model loads from upstream source pinned by commit and hash (upstream.json code.keep)."""
    rel = py.relative_to(ROOT / "src" / "mashbox")
    if len(rel.parts) < 3 or rel.parts[0] != "models":
        return set()
    pkg = ROOT / "src" / "mashbox" / "models" / rel.parts[1]
    allowed = set(ex.get(rel.parts[1].replace("_", "-"), ()))
    up = pkg / "upstream.json"
    if up.is_file():
        allowed |= set(json.loads(up.read_text()).get("code", {}).get("keep", ()))
    return allowed


def first_difference(a, b, path="models.json"):
    """The JSON path of the first place two parsed registries differ, or None when they are equal."""
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            d = first_difference(a.get(k), b.get(k), f"{path}.{k}") if k in a and k in b else f"{path}.{k}"
            if d:
                return d
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            label = x.get("id", i) if isinstance(x, dict) else i
            d = first_difference(x, y, f"{path}[{label}]")
            if d:
                return d
        return None if len(a) == len(b) else f"{path} (length {len(a)} vs {len(b)})"
    return None if a == b else path


def test_models_json_copy():
    """The wheel ships src/mashbox/models.json. It must be a byte copy of models.json, or the package carries a stale
    licence line and stale pins (it nearly shipped both on 2026-10-28)."""
    src, copy = ROOT / "models.json", ROOT / "src" / "mashbox" / "models.json"
    fix = "fix: cp models.json src/mashbox/models.json"
    where = first_difference(json.loads(src.read_text()), json.loads(copy.read_text()))
    assert where is None, f"src/mashbox/models.json differs from models.json first at {where}; {fix}"
    assert src.read_bytes() == copy.read_bytes(), f"src/mashbox/models.json has the same content but other formatting; {fix}"


KNOWN_WITHOUT_DOCSTRING = {"test_hy_nomas.py"}      # hy-nomas lane adds it; this set only shrinks


def test_every_test_module_says_what_it_guards():
    """Every test module opens with a docstring naming the failure it guards (CI.md: every test earns its place)."""
    missing = {p.name for p in (ROOT / "tests").rglob("test_*.py") if not ast.get_docstring(ast.parse(p.read_text()))}
    assert missing <= KNOWN_WITHOUT_DOCSTRING, f"add a module docstring naming the failure each guards: {sorted(missing - KNOWN_WITHOUT_DOCSTRING)}"


def test_every_model_has_its_folders():
    reg = json.loads((ROOT / "models.json").read_text())
    ids = [m["id"] for m in reg["models"]]
    assert len(ids) == len(set(ids))
    for m in reg["models"]:
        assert (ROOT / m["path"] / "__init__.py").is_file(), m["id"]
        assert m["path"] == "src/mashbox/models/" + m["id"].replace("-", "_"), m["id"]
        assert (ROOT / "docs" / "models" / (m["id"] + ".md")).is_file(), m["id"]
        assert any((ROOT / "licenses" / m["id"]).iterdir()), m["id"]


def test_imports_are_public():
    """Package code imports only the core dependencies, its model's declared extra and pinned upstream code."""
    bad, ex = [], extras()
    for py in (ROOT / "src" / "mashbox").rglob("*.py"):
        ok = ALLOWED | model_allowed(py, ex)
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top not in ok and top not in sys.stdlib_module_names:
                    bad.append(f"{py.relative_to(ROOT)}: {n}")
    assert not bad, bad


def test_views_match_models_json():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "hf_github_sync_check.py"), "check", "--repo", str(ROOT), "--offline"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_licence_check_catches_drift(tmp_path):
    """The licence check (report only, never edits) fails on each kind of mismatch with models.json: the package licence,
    an upstream LICENSE that is not the pinned file, and a page that says a licence is not chosen."""
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = json.loads((ROOT / "models.json").read_text())
    problems = sc.Problems()
    sc.check_licences(reg, ROOT, problems)
    assert not problems, problems
    shutil.copy(ROOT / "LICENSE", tmp_path / "LICENSE")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nlicense = "MIT"\nlicense-files = ["LICENSE"]\n')
    mid = next(m["id"] for m in reg["models"] if m["upstream"].get("license_sha256"))
    (tmp_path / "licenses" / mid).mkdir(parents=True)
    (tmp_path / "licenses" / mid / "LICENSE").write_text("edited\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("Our code: licence not chosen yet.\n")
    problems = sc.Problems()
    sc.check_licences(reg, tmp_path, problems)
    text = " ".join(problems)
    assert "pyproject.toml" in text and f"licenses/{mid}/LICENSE" in text and "docs/x.md:1" in text, text


def test_run_stops_cleanly_for_a_model_without_its_runtime(capsys):
    from mashbox import cli
    reg = json.loads((ROOT / "models.json").read_text())
    for m in reg["models"]:
        if m["id"] not in cli.RUNNABLE:
            assert cli.main(["run", m["id"]]) == 2
            assert cli.main(["download", m["id"]]) == 2
            assert "not in mashbox" in capsys.readouterr().err


KNOWN_WITHOUT_KERNEL_CHECK = {"qwen3-8-27b"}        # Qwen lane adds tests/tpu/qwen3_8_27b_kernel_check.py; this set only shrinks


def test_every_kernel_bundle_has_a_chip_check():
    """A model that ships compiled kernels for a chip needs tests/tpu/<id>_kernel_check.py and its tolerance file, or
    the chip tier (T2) never compares those kernels with plain JAX and a broken bundle ships green."""
    missing = set()
    for m in json.loads((ROOT / "models.json").read_text())["models"]:
        a = m["id"].replace("-", "_")
        if m.get("kernels") and not all((ROOT / "tests" / "tpu" / f).is_file() for f in (f"{a}_kernel_check.py", f"{a}_kernel_tolerances.json")):
            missing.add(m["id"])
    assert missing <= KNOWN_WITHOUT_KERNEL_CHECK, f"kernel bundles with no chip check: {sorted(missing - KNOWN_WITHOUT_KERNEL_CHECK)}"


def test_path_filter_runs_what_a_change_can_break():
    """tools/ci_select.py: a wrong rule silently skips the tests a change can break, so the rules that matter are pinned."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci_select", ROOT / "tools" / "ci_select.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    areas = cs.model_areas()
    every = sorted(set(areas) | ({"core"} if cs.core_has_t1_tests(areas) else set()))
    a = areas[0]
    assert cs.select(["README.md", "docs/faq.md", f"hf/{a}/README.md", "licenses/x"], areas) == []
    assert cs.select([f"src/mashbox/models/{a}/x.py"], areas) == [a]
    assert cs.select([f"tests/cpu/test_{a}_x.py"], areas) == [a]
    assert cs.select(["src/mashbox/cli.py"], areas) == every
    assert cs.select(["some/new/kind/of/file.cfg"], areas) == every
    for p in (ROOT / "tests" / "tpu").glob("*"):                       # a tests/tpu file a CPU test reads runs that test
        readers = {cs.test_area(t.name, areas) for t in (ROOT / "tests" / "cpu").glob("*.py") if p.name in t.read_text()}
        assert readers <= set(cs.select([f"tests/tpu/{p.name}"], areas)), p.name


def test_sync_online_coming_soon_model():
    """The nightly --online check passes a coming-soon model whose HF repo is private (401/403) or absent (404), and
    fails one that is public before its launch or whose state cannot be read (fixtures, no network)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = {"hf": {"org": "mashbox-labs"}, "github": {"org": "mashbox-labs", "repo": "mashbox"}, "version": "0.1.1",
           "models": [{"id": "soon", "hf_repo": "mashbox-labs/Soon-TPU", "hf_revision": "", "launch": "coming soon"}]}

    class Fetch:
        def __init__(self, repo_state):
            self.repo_state = repo_state

        def get(self, url):
            if url == "https://huggingface.co/api/models?author=mashbox-labs&limit=1000":
                return 200, "[]"
            if url == "https://huggingface.co/api/models/mashbox-labs/Soon-TPU":
                return self.repo_state
            if url == "https://api.github.com/repos/mashbox-labs/mashbox/git/ref/tags/v0.1.1":
                return 200, "{}"
            return 404, ""

    def problems(repo_state):
        p = sc.Problems()
        sc.check_online(reg, {}, Fetch(repo_state), p)
        return [str(x) for x in p]

    assert problems((401, "")) == []
    assert problems((403, "")) == []
    assert problems((404, "")) == []
    assert problems((200, json.dumps({"private": True}))) == []
    assert any("public before its launch" in x for x in problems((200, json.dumps({"private": False}))))
    assert any("cannot read the repo state" in x for x in problems((500, "")))
