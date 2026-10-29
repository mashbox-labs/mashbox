"""The activation-edits engine interface (mashbox.runtime.activation_edits), checked without a chip, jax or torch:
  - the base ActivationEdits changes nothing and reads nothing;
  - the shared refusals (wrong type, the tensor-parallel layout, ed= without edits, edits with verify, act and resid
    reading the same names);
  - every model definition that lists ACTIVATION_EDIT_SITES lists only known sites and has make_forward(..., edits=None),
    and every model whose make_forward takes edits lists its sites (read from the source, no import);
  - the interface module names no model and imports nothing of a model or of jax.
"""
import ast
import pathlib

import pytest

from mashbox.runtime import activation_edits as AE

PKG = pathlib.Path(AE.__file__).resolve().parents[1]           # src/mashbox


def test_base_changes_nothing_and_reads_nothing():
    e, a, x = AE.ActivationEdits(), object(), object()
    for site in AE.WRITER_SITES:
        assert e.writer(site, 0, None, None, {}) is None
    assert e.act(0, None, None, a, {}) == (a, {})
    assert e.resid(0, None, None, x, {}) == (x, {})
    assert AE.SITES == ("writer:embed", "writer:attn", "writer:mlp", "act", "resid")


def test_refusals():
    AE.check_make_forward_args(None, True)
    AE.check_make_forward_args(AE.ActivationEdits(), False)
    with pytest.raises(TypeError, match="ActivationEdits"):
        AE.check_make_forward_args(object(), False)
    with pytest.raises(ValueError, match="one chip"):
        AE.check_make_forward_args(AE.ActivationEdits(), True)
    AE.check_forward_args(None, None, True)
    AE.check_forward_args(AE.ActivationEdits(), {"call": 1}, False)
    with pytest.raises(ValueError, match="needs make_forward"):
        AE.check_forward_args(None, {"call": None}, False)
    with pytest.raises(ValueError, match="without speculation"):
        AE.check_forward_args(AE.ActivationEdits(), None, True)
    assert AE.merge_reads({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="same names"):
        AE.merge_reads({"a": 1}, {"a": 2})
    assert AE.split_op(None) == (None, None)
    assert AE.split_op({"call": 1, "layer": 2}) == (1, 2)
    assert AE.split_op({"call": 1}) == (1, None)


def _model_files():
    """Every model definition file in the package: models/<name>/model.py and gguf/archs/<arch>.py."""
    out = sorted((PKG / "models").glob("*/model.py"))
    out += sorted(p for p in (PKG / "gguf" / "archs").glob("*.py") if p.name != "__init__.py")
    return out


def _declared(tree):
    """(ACTIVATION_EDIT_SITES value or None, make_forward's parameter names -> default or None)."""
    sites, params = None, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ACTIVATION_EDIT_SITES" for t in node.targets):
            v = node.value
            if isinstance(v, ast.Attribute) and v.attr == "SITES":
                sites = AE.SITES
            else:
                sites = tuple(ast.literal_eval(v))
        if isinstance(node, ast.FunctionDef) and node.name == "make_forward":
            a = node.args
            names = [x.arg for x in a.args + a.kwonlyargs]
            defs = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults) + list(a.kw_defaults)
            params = dict(zip(names, defs))
    return sites, params


def test_model_definitions_declare_their_sites():
    files = _model_files()
    assert any(p.parent.name == "qwen3_8_27b" for p in files), files
    implementers = []
    for p in files:
        sites, params = _declared(ast.parse(p.read_text()))
        takes = params is not None and "edits" in params
        if sites is None and not takes:
            continue
        rel = p.relative_to(PKG)
        assert sites is not None, f"{rel}: make_forward takes edits but the module lists no ACTIVATION_EDIT_SITES"
        assert takes, f"{rel}: lists ACTIVATION_EDIT_SITES but make_forward takes no edits"
        assert sites and all(s in AE.SITES for s in sites), f"{rel}: unknown sites {sites}"
        d = params["edits"]
        assert isinstance(d, ast.Constant) and d.value is None, f"{rel}: make_forward's edits must default to None"
        implementers.append(str(rel))
    assert "models/qwen3_8_27b/model.py" in implementers


def test_model_sites_reads_the_module_list():
    class M:
        __name__ = "m"
    assert AE.model_sites(M) == ()
    M.ACTIVATION_EDIT_SITES = ("act", "resid")
    assert AE.model_sites(M) == ("act", "resid")
    M.ACTIVATION_EDIT_SITES = ("act", "nope")
    with pytest.raises(ValueError, match="unknown"):
        AE.model_sites(M)


def test_interface_is_model_free():
    src = pathlib.Path(AE.__file__).read_text()
    tree = ast.parse(src)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], "the interface module imports nothing"
    names = {p.name for p in (PKG / "models").iterdir() if p.is_dir() and not p.name.startswith("_")}
    names |= {p.stem for p in (PKG / "gguf" / "archs").glob("*.py") if p.stem != "__init__"}
    low = src.lower()
    assert not [n for n in names if n.lower() in low], "the interface names no model"
