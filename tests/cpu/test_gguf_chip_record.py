"""The first chip record of a held mixture-of-experts architecture runs on an unchanged src/ tree. The record runner
(tools/gguf_chip_record.py, outside src/) loads the file with cli._load_gguf_for_record, the loader `mashbox run` calls below
its hold (gguf.archs.held), while `run` keeps refusing the same file with its one line. One loader serves both, so a
record measures the code `run` runs and its code set v1 can equal main's. No flag and no environment variable opens the
hold. On CPU with no network, on the tiny fixtures."""
import ast
import importlib.util
import os

import pytest

import gguf  # noqa: F401,E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
pytest.importorskip("jax")

from mashbox import cli  # noqa: E402
from mashbox.gguf import archs  # noqa: E402
from mashbox.gguf import pick as GR  # noqa: E402
from mashbox.gguf.reader import GGUFError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)
RUNNER = os.path.join(ROOT, "tools", "gguf_chip_record.py")


def _runner():
    spec = importlib.util.spec_from_file_location("gguf_chip_record", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_args(p):
    """the arguments `mashbox run <file>` loads with"""
    a = cli.build_parser().parse_args(["run", p])
    a.gguf = GR.resolve(a.model, log=cli.log)
    return a


class _Stop(Exception):
    pass


def test_the_runner_loads_a_held_moe_file_that_run_refuses(tmp_path):
    p = FXT.build("qwen35moe-f32", str(tmp_path))
    line = archs.unsupported("qwen35moe", p)
    held = archs.held
    with pytest.raises(GGUFError) as e:
        cli.load_gguf(_run_args(p))
    assert str(e.value) == line
    m, eng, chat, info, t0 = _runner().load(p)
    assert eng.S >= 1 and info["device_bytes"] > 0 and info["reference_bytes"] >= 0, info
    assert archs.held is held                                     # nothing replaced at run time
    with pytest.raises(GGUFError) as e:                           # and `run` still refuses the same file afterwards
        cli.load_gguf(_run_args(p))
    assert str(e.value) == line


def test_run_and_the_runner_share_one_loader_below_the_hold(tmp_path, monkeypatch):
    seen = []

    def spy(a, f, A, c, t0):
        seen.append(f.arch)
        raise _Stop

    monkeypatch.setattr(cli, "_load_gguf_for_record", spy)
    d = FXT.build("qwen35-f32", str(tmp_path))
    p = FXT.build("qwen35moe-f32", str(tmp_path))
    with pytest.raises(_Stop):
        cli.load_gguf(_run_args(d))                               # a dense file: `run` goes on to the loader
    with pytest.raises(GGUFError):
        cli.load_gguf(_run_args(p))                               # a held MoE file: refused before the loader
    with pytest.raises(_Stop):
        _runner().load(p)                                         # the runner: the same loader, below the hold
    assert seen == ["qwen35", "qwen35moe"], seen


LOADER_NAMES = ("_load_gguf_for_record", "load_gguf_file")


def loader_references(tree):
    """-> {scope}: every place `tree` names the loader below the hold, a call or not: a name or attribute load, an import
    alias, or a string that is the name (getattr). Scope = the qualified name of the def or class holding it (K.load,
    f.load; a class body as 'class K'), '<module>' outside any; a def's or class's decorators, defaults, annotations,
    bases and type parameters belong to the scope around it. A def named after the loader is its definition, not a
    reference. Not seen: a name built from parts (any string join, code in a string)."""
    seen = set()

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                around = node.decorator_list + node.bases + [k.value for k in node.keywords]
                inner = f"class {node.name}"
            else:
                a = node.args
                params = a.posonlyargs + a.args + a.kwonlyargs + [v for v in (a.vararg, a.kwarg) if v is not None]
                around = (node.decorator_list + a.defaults + [d for d in a.kw_defaults if d is not None]
                          + [p.annotation for p in params if p.annotation is not None]
                          + ([node.returns] if node.returns is not None else []))
                inner = node.name
            for d in around + list(getattr(node, "type_params", [])):
                visit(d, scope)
            for x in node.body:
                visit(x, f"{scope}.{inner}" if scope != "<module>" else inner)
            return
        hit = ((isinstance(node, ast.Name) and node.id in LOADER_NAMES)
               or (isinstance(node, ast.Attribute) and node.attr in LOADER_NAMES)
               or (isinstance(node, ast.alias) and (node.name.rsplit(".", 1)[-1] in LOADER_NAMES
                                                   or node.asname in LOADER_NAMES))
               or (isinstance(node, ast.Constant) and node.value in LOADER_NAMES))
        if hit:
            seen.add(scope)
        for c in ast.iter_child_nodes(node):
            visit(c, scope)

    visit(tree, "<module>")
    return seen


def test_the_loader_below_the_hold_is_internal_and_only_load_gguf_and_the_runner_call_it():
    """The hold is the support gate of `run`, serving and `inspect` (`tpu up` starts `mashbox run` on the chip), not a
    same-process boundary: Python code that imports mashbox.cli can reach the loader below it. So that loader is not a
    public name of mashbox.cli (one a caller could take for a supported way to load a file), and nothing in src/ or
    tools/ names it but load_gguf, after the hold, and the record runner's load: a call, an alias, an import or a
    getattr by name, at any scope, each counts"""
    from mashbox import cli as C
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert not hasattr(C, "load_gguf_file") and callable(getattr(C, "_load_gguf_for_record", None))
    refs = set()
    for top in ("src", "tools"):
        for d, _, names in os.walk(os.path.join(root, top)):
            for n in names:
                if n.endswith(".py"):
                    path = os.path.join(d, n)
                    refs |= {(os.path.relpath(path, root), s) for s in loader_references(ast.parse(open(path).read()))}
    assert refs == {("src/mashbox/cli.py", "load_gguf"), ("tools/gguf_chip_record.py", "load")}, refs


@pytest.mark.parametrize(("src", "want"), [
    ("loader = cli._load_gguf_for_record\n", {"<module>"}),
    ("def f():\n    g = cli._load_gguf_for_record\n    return g\n", {"f"}),
    ("from mashbox.cli import _load_gguf_for_record as g\n", {"<module>"}),
    ("def f():\n    return getattr(cli, '_load_gguf_for_record')\n", {"f"}),
    ("class K:\n    load = cli._load_gguf_for_record\n", {"class K"}),
    ("def f(x: cli._load_gguf_for_record):\n    return x\n", {"<module>"}),
    ("def f() -> cli._load_gguf_for_record:\n    return 1\n", {"<module>"}),
    ("class K:\n    def load_gguf(self):\n        return cli._load_gguf_for_record(1)\n", {"class K.load_gguf"}),
    ("class load:\n    x = cli._load_gguf_for_record\n", {"class load"}),
    ("class P[T: cli._load_gguf_for_record]:\n    pass\n", {"<module>"}),
    ("def f():\n    def load():\n        return cli._load_gguf_for_record(1)\n    return load\n", {"f.load"}),
    ("def f(x=cli._load_gguf_for_record):\n    return x\n", {"<module>"}),
    ("def f():\n    return cli._load_gguf_for_record(1, 2, 3, 4, 5)\n", {"f"}),
    ("def _load_gguf_for_record(a):\n    return a\n", set()),
    ('"""cli._load_gguf_for_record is internal"""\n', set()),
], ids=["module-alias", "local-alias", "import-as", "getattr", "class-attr", "annotation", "return-annotation",
        "method-named-load_gguf", "class-named-load", "class-type-param", "nested-def-named-load", "default-arg", "call", "def", "docstring"])
def test_the_reference_scan_sees_an_alias_an_import_and_a_getattr(src, want):
    """the scan above counts a reference however it is written, so binding the loader to a new name hides no caller"""
    assert loader_references(ast.parse(src)) == want


def test_the_runner_sets_no_flag_and_replaces_nothing():
    """no environment variable written and no attribute of mashbox assigned: the runner opens the hold only by calling
    the loader below it"""
    tree = ast.parse(open(RUNNER).read())
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Attribute) or (isinstance(t, ast.Subscript) and "environ" in ast.unparse(t)):
                    bad.append(ast.unparse(t))
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ("setattr", "os.environ.setdefault", "os.putenv",
                                                               "os.environ.update"):
            bad.append(ast.unparse(n))
    assert bad == [], bad
