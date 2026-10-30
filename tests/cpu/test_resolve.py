"""The one resolver (src/mashbox/resolve.py) picks up any package from its data, with no CLI edit.

Guards against `mashbox run` needing a code change for a new package: a model id, name or alias typed into the CLI, a
runtime imported by name, a new kind of target that falls through to a stack trace, or a package that cannot run being
started anyway. Fake packages (rows added to the catalog at test time, their runtime packages written to a temporary
folder on mashbox.models' path) must resolve and dispatch exactly as the shipped ones; every refusal is one line and
exit 2. CPU, no network (the Hugging Face pick is stubbed where a repo is named).
"""
import json
import sys
import textwrap

import pytest

import mashbox.models as MM
from mashbox import cli, hub, resolve as R
from mashbox.gguf import pick as GR

REAL = hub.registry()

OWN = '''
CALLS = []


def main(cmd, argv):
    CALLS.append((cmd, list(argv)))
    return 0
'''
ENGINE = '''
class Config:
    def __init__(self, d):
        raise RuntimeError(f"zz engine Config({d})")


def new_state(c, S, CTX, extra_groups=0):
    raise RuntimeError("zz engine new_state")


def make_forward(c, decode_kernels=None):
    raise RuntimeError("zz engine make_forward")
'''
WEIGHTS = '''
def load(path, c, log=None):
    raise RuntimeError("zz engine weights.load")
'''


def row(i, pkg, **kw):
    return dict(id=f"zz-demo-{i}", name=f"ZZ-Demo-{i.title()}", hf_repo=f"mashbox-labs/ZZ-Demo-{i.title()}-TPU",
                path=f"src/mashbox/models/{pkg}", kind="test", files=[], chips={}, **kw)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """the shipped catalog plus fake rows, and their packages on mashbox.models' path; -> (rows by id, stderr lines)"""
    pk = tmp_path / "pkgs"
    for name, files in {"zz_demo_own": {"__init__.py": OWN},
                        "zz_demo_engine": {"__init__.py": ENGINE, "weights.py": WEIGHTS},
                        "zz_demo_neither": {"__init__.py": "X = 1\n"}}.items():
        (pk / name).mkdir(parents=True)
        for f, body in files.items():
            (pk / name / f).write_text(textwrap.dedent(body))
    rows = [row("own", "zz_demo_own"), row("engine", "zz_demo_engine"), row("neither", "zz_demo_neither"),
            row("absent", "zz_demo_not_in_this_version"), dict(row("outside", "x"), path="src/mashbox/runtime")]
    reg = dict(REAL, models=list(REAL["models"]) + rows)
    monkeypatch.setattr(hub, "registry", lambda: reg)
    monkeypatch.setattr(MM, "__path__", list(MM.__path__) + [str(pk)])
    for name in ("zz_demo_own", "zz_demo_engine", "zz_demo_neither"):
        monkeypatch.delitem(sys.modules, f"mashbox.models.{name}", raising=False)
    lines = []
    monkeypatch.setattr(cli, "log", lines.append)
    yield {r["id"]: r for r in rows}, lines
    for name in ("zz_demo_own", "zz_demo_engine", "zz_demo_neither"):
        sys.modules.pop(f"mashbox.models.{name}", None)


def spellings(m):
    return [m["id"], m["id"].upper(), m["name"], m["hf_repo"], m["hf_repo"].lower(),
            "https://huggingface.co/" + m["hf_repo"], "hf.co/" + m["hf_repo"] + "/"]


def test_a_new_package_with_its_own_command_line_runs_with_no_cli_edit(fake):
    rows, _ = fake
    m = rows["zz-demo-own"]
    assert m["id"] not in cli.MODEL_CLI                 # nothing in the CLI knows it: only the catalog does
    for s in spellings(m):
        p = R.resolve(s, download=False, log=lambda *x: None)
        assert (p.format, p.row["id"], p.entry, p.module) == ("catalog", m["id"], "own", "mashbox.models.zz_demo_own"), s
        assert R.own_command_line(s) == "mashbox.models.zz_demo_own", s
        assert cli.main(["run", s, "clip.npz", "-o", "out"]) == 0
        assert cli.main(["download", s]) == 0
    calls = sys.modules["mashbox.models.zz_demo_own"].CALLS
    assert calls == [("run", ["clip.npz", "-o", "out"]), ("download", [])] * len(spellings(m)), calls


def test_a_new_engine_package_loads_its_own_model_definition(fake, monkeypatch, tmp_path):
    """cli.load builds the engine from the package the row's path names: the fake package's Config is the one called"""
    pytest.importorskip("jax")
    rows, _ = fake
    m = rows["zz-demo-engine"]
    for s in spellings(m):
        p = R.resolve(s, download=False, log=lambda *x: None)
        assert (p.format, p.row["id"], p.entry, p.module) == ("catalog", m["id"], "engine", "mashbox.models.zz_demo_engine")
        assert R.own_command_line(s) is None
    up = tmp_path / "upstream"
    up.mkdir()
    monkeypatch.setattr(hub, "upstream_dir", lambda m, log=None: str(up))
    monkeypatch.setattr(hub, "weights_file", lambda m, name=None, path=None, log=None: str(tmp_path / "w.bshj"))
    monkeypatch.setattr(hub, "compile_cache", lambda *a, **k: None)
    for s in (m["id"], m["hf_repo"]):
        with pytest.raises(RuntimeError, match=f"zz engine Config\\({up}\\)"):
            cli.main(["run", s, "--prompt", "hi", "--no-kernels", "--cache-dir", str(tmp_path / "xla")])
        a = cli.build_parser().parse_args(["run", s, "--no-kernels", "--cache-dir", str(tmp_path / "xla")])
        with pytest.raises(RuntimeError, match="zz engine Config"):
            R.load(R.resolve(s, download=False), a)


@pytest.mark.parametrize("mid, words", [
    ("zz-demo-neither", "has neither main(cmd, argv)"),
    ("zz-demo-absent", "is not in mashbox"),
    ("zz-demo-outside", "names no runtime package"),
])
def test_a_package_this_version_cannot_run_stops_in_one_line(fake, mid, words):
    _, lines = fake
    for cmd in ("run", "download"):
        del lines[:]
        assert cli.main([cmd, mid]) == 2
        assert len(lines) == 1 and words in lines[0] and "Traceback" not in lines[0], lines
    with pytest.raises(R.ResolveError, match=words.replace("(", "\\(").replace(")", "\\)")):
        R.resolve(mid, download=False)


def test_an_unknown_name_stops_in_one_line(fake):
    _, lines = fake
    assert cli.main(["run", "zz-no-such-model"]) == 2
    assert len(lines) == 1 and "unknown model 'zz-no-such-model'" in lines[0], lines


def test_every_shipped_row_resolves_from_its_own_data():
    """models[] rows by id, name and Hugging Face repo, to the runtime their path names; MODEL_CLI is that data"""
    for m in REAL["models"]:
        for s in spellings(m):
            p = R.resolve(s, download=False, log=lambda *x: None)
            assert (p.format, p.row["id"]) == ("catalog", m["id"]), s
            assert p.module == "mashbox." + m["path"].split("src/mashbox/", 1)[1].replace("/", ".")
            assert (p.entry == "own") == (m["id"] in cli.MODEL_CLI)
            assert not R.is_file_target(s)


def test_a_gguf_row_id_runs_the_file_its_row_pins(monkeypatch):
    """every shipped gguf[] row's id resolves to its own file at its own revision (the repo's head is never listed),
    and the row's `run` line names the same repo and quant tag that a changed upstream file is sent to"""
    asked = []
    monkeypatch.setattr(GR, "resolve", lambda *a, **k: pytest.fail("a gguf row id read the repo's head"))
    monkeypatch.setattr(R, "hub_sha256", lambda repo, files, revision: asked.append((repo, revision)) or
                        {f: g["sha256"] for f in files})
    for g in REAL["gguf"]:
        assert R.is_file_target(g["id"])
        p = R.resolve(g["id"].upper(), download=False, log=lambda *x: None)
        assert (p.format, p.row["id"], p.arch, p.source["repo"]) == ("gguf", g["id"], g["arch"], g["repo"])
        assert (p.source["file"], p.source["revision"], p.source["sha256"]) == (g["file"], g["revision"], g["sha256"])
        assert asked[-1] == (g["repo"], g["revision"]) and p.source["path"] is None
        assert p.why[0].startswith(f"format: catalog, models.json gguf row {g['id']}")
        assert g["run"] == f"mashbox run {g['repo']}:{g['tag']}"


def test_repo_listings_go_to_the_probe_their_files_name(monkeypatch, tmp_path):
    """a repo with no GGUF file: BSHJ weights with no row are refused in one line (and a local weights file of a row
    names its run command); config.json + safetensors is the safetensors probe's; anything else is one line"""
    listing = {}

    def no_gguf(target, log=None, download=True):
        raise GR.NoGGUF(f"mashbox: {target} has no GGUF files", listing[target])
    monkeypatch.setattr(GR, "resolve", no_gguf)
    listing["acme/zz-bshj"] = ["README.md", "transformer.bshj", "vae.bshj"]
    with pytest.raises(R.ResolveError, match="holds BSHJ weights .transformer.bshj, vae.bshj. that no models.json row"):
        R.resolve("acme/zz-bshj", download=False)
    listing["acme/zz-empty"] = ["README.md", "model.onnx"]
    with pytest.raises(R.ResolveError, match="has no GGUF, BSHJ or safetensors model files"):
        R.resolve("acme/zz-empty", download=False)
    m = next(m for m in REAL["models"] if any(f.get("kind", "weights") == "weights" and f["name"].endswith(".bshj")
                                               for f in m["files"]))
    w = tmp_path / next(f["name"] for f in m["files"] if f["name"].endswith(".bshj"))
    w.write_bytes(b"x")
    with pytest.raises(R.ResolveError, match=f"weights file of the catalog model {m['id']}: run `mashbox run {m['id']} --weights"):
        R.resolve(str(w), download=False)
    d = tmp_path / "folder"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(dict(architectures=["LlamaForCausalLM"], model_type="llama")))
    (d / "model.safetensors").write_bytes(b"")
    p = R.resolve(str(d), download=False, log=lambda *x: None)
    assert (p.format, p.arch, p.moe, p.source["path"]) == ("safetensors", "llama", False, str(d))


def test_flags_a_target_cannot_take_stop_before_anything_is_fetched(monkeypatch, fake):
    seen = []
    monkeypatch.setattr(GR, "resolve", lambda *a, **k: seen.append(a))
    _, lines = fake
    for argv, words in ((["run", "acme/zz-x", "--weights", "w.bshj"], "--weights is for models.json models"),
                        (["run", REAL["gguf"][0]["id"], "--mtp", "2"], "--mtp and --mtp-file are for models.json models"),
                        (["run", "zz-demo-engine", "--draft-model", "d.gguf"], "--speculation draft runs on a GGUF file")):
        del lines[:]
        assert cli.main(argv) == 2 and seen == [], argv
        assert len(lines) == 1 and words in lines[0], lines


def test_inspect_prints_the_plan(fake, capsys):
    rows, lines = fake
    assert cli.main(["inspect", rows["zz-demo-engine"]["hf_repo"]]) == 0
    plan = json.loads(lines[0])
    assert plan["event"] == "plan" and plan["format"] == "catalog" and len(plan["why"]) == 2, plan
    out = json.loads(capsys.readouterr().out)
    assert (out["model"], out["runtime"], out["entry"]) == ("zz-demo-engine", "mashbox.models.zz_demo_engine", "engine")
