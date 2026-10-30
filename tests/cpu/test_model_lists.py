"""Every model list and count on a public page is rendered from models.json, and a page that types one fails.

The README, the org profile and the docs catalog print the same rows and the same count line, from
tools/model_lists.py, which reads mashbox/catalog.py and never changes it (catalog.py is in the code set of every GGUF
chip record). A new gguf[] row reaches all of them with render --write and no other step. Outside the rendered blocks
a page types no count of models, the org profile's prose names no model, and every printed count line adds up to
models.json's total.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from mashbox import catalog

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = _load("sync_check_lists", ROOT / "tools" / "hf_github_sync_check.py")
ML = _load("model_lists_test", ROOT / "tools" / "model_lists.py")
REG = json.loads((ROOT / "src" / "mashbox" / "models.json").read_text())

# The org profile's old intro: a hand list of two models under a table of three.
OLD_INTRO = ("Our open model ports run on Google Cloud TPU, and anyone can run them: a 27B language model on a single "
             "chip, and Hy-Nomas, a world model you steer in the browser, on a TPU v6e-4 slice.")
NEW_INTRO = ("We port open models to Google Cloud TPU, and anyone can run them. The table below lists every model in "
             "the current mashbox release and the chips it runs on.")
INSTALL = "```bash\n~/.local/bin/mashbox run qwen3-8-27b --prompt \"Write a haiku about the sea.\"\n```\n"


def profile(intro, reg=REG):
    return f"{intro}\n\n{INSTALL}\n{SC.MARK_START}\n{SC.profile_block(reg)}\n{SC.MARK_END}\n"


def block(text):
    return f"{SC.MARK_START}\n{text}\n{SC.MARK_END}\n"


def passing_row(reg):
    return next(g for g in reg["gguf"] if catalog.chips_run(g))


def with_new_row(reg):
    """reg plus one more passing GGUF file of a new model (a copy of a passing row under a new id, model and file)."""
    reg = copy.deepcopy(reg)
    g = copy.deepcopy(passing_row(reg))
    g.update(id="someone-newmodel-9b-q4-k-m", model="newmodel-9b", publisher="someone", repo="someone/NewModel-9B-GGUF",
             file="NewModel-9B-Q4_K_M.gguf", base=f"{g['maker']}/NewModel-9B", run="mashbox run someone/NewModel-9B-GGUF:Q4_K_M")
    reg["gguf"].append(g)
    return reg


def test_profile_prose_names_no_model():
    old = [m for _, m in SC.list_problems(REG, profile(OLD_INTRO), "profile", names=True)]
    assert any("Hy-Nomas" in m for m in old), old
    assert any("'27B'" in m for m in old), old          # a size as people write it names the model too
    assert SC.list_problems(REG, profile(NEW_INTRO), "profile", names=True) == []


def test_typed_count_outside_a_block_fails():
    for text in ("mashbox supports two models today.", "Run any of our 3 open model ports.", "We test 14 GGUF files."):
        assert SC.list_problems(REG, text, "page"), text
    inside = f"{SC.MARK_START}\nWe test 14 GGUF files.\n{SC.MARK_END}\n"
    assert SC.list_problems(REG, inside, "page") == []


def test_count_line_adds_up():
    for reg in (REG, with_new_row(REG)):
        c, line = ML.counts(reg), ML.count_line(reg)
        assert c["entries"] == c["ports"] + c["own"] + c["files"] == len(catalog.live(reg)) + len(reg["gguf"])
        assert sum(k for k, _ in ML.count_parts(reg)) == c["entries"]
        assert line.startswith(f"{c['entries']} entries: ")
        assert SC.list_problems(reg, block(line + "."), "page") == []
        assert SC.list_problems(reg, line + ".", "page"), "a count line outside a rendered block is a typed count"
    wrong = ML.count_line(REG).replace(f"{ML.counts(REG)['entries']} entries", "2 entries", 1)
    assert any("does not add up" in m for _, m in SC.list_problems(REG, block(wrong + "."), "page")), wrong


def test_new_row_reaches_every_list():
    reg = with_new_row(REG)
    n = ML.counts(REG)["entries"]
    views = {"README": SC.readme_block(reg), "profile": SC.profile_block(reg), "docs": ML.render_markdown(reg),
             "mashbox models": catalog.render_text(reg)}
    assert "NewModel-9B" in views.pop("mashbox models")   # the CLI lists every row; its count line comes after 0.1.6
    for name, text in views.items():
        assert "NewModel-9B" in text, name
        assert f"{n + 1} entries: " in text, name
        assert f"{n} entries: " not in text, name


def test_docs_catalog_is_one_page_in_two_tiers():
    """MASHBOX builds first (every models[] row, its measured lines), then GGUF files that run in our engine, one
    section per family with maker, architecture and quant; the README, the profile and llms.txt carry the same tiers"""
    md = ML.render_markdown(REG)
    heads = [line for line in md.splitlines() if line.startswith("## ")]
    assert heads == [f"## {ML.BUILDS}", f"## {ML.GGUF}"]
    builds, gguf = md.split(f"## {ML.GGUF}")
    for m in catalog.live(REG):
        assert f"(models/{m['id']}.md)" in builds and f"(models/{m['id']}.md)" not in gguf
        for line in ML.measured(REG, m["id"]):
            assert line in builds
    fams = [f["name"] for f in ML.families(REG) if any(r["format"] == "GGUF" for r in f["rows"])]
    assert [line[4:] for line in gguf.splitlines() if line.startswith("### ")] == fams
    for f in ML.families(REG):
        assert len({ML.family(r) for r in f["rows"]}) == 1
    for r in catalog.gguf_rows(REG):
        assert r["file_url"] in gguf and f"`{r['arch']}`" in gguf and r["maker"] in gguf
        assert r["file_url"] not in builds
    for text in (SC.readme_block(REG), SC.profile_block(REG), SC.llms_block(REG)):
        assert text.index(ML.BUILDS) < text.index(ML.GGUF)
    q = next(m for m in REG["models"] if m["id"] == "qwen3-8-27b")
    spec = next(h for h in q["headline"] if q["results"][h["results"][0]]["short"]["ours"] == h["big"] == "74.84 tok/s")
    assert spec.get("readme") and "37.77" in q["results"][spec["results"][0]]["baseline"]
    for text in (SC.readme_block(REG), md):
        assert "**74.84 tok/s**" in text and "37.77 tok/s with it off" in text
    assert ML.family({"format": "GGUF", "base": "google/gemma-3-1b-it", "family": "x"}) == "Gemma"
    assert ML.family({"format": "GGUF", "base": "mistralai/Mistral-7B-Instruct-v0.3", "family": "x"}) == "Mistral"
    assert ML.family({"format": "TPU port", "family": "Track2Rig", "name": "Track2Rig"}) == "Track2Rig"


def test_maker_is_the_base_models_org():
    assert ML.maker_problems(REG) == []
    reg = copy.deepcopy(REG)
    g = passing_row(reg)
    g.update(maker=g["publisher"], base=None)
    assert len(ML.maker_problems(reg)) >= 1
    g.update(base=g["repo"].split("/")[1], maker=g["repo"].split("/")[1])
    assert any("model's name" in p for _, p in ML.maker_problems(reg))
    for g in REG["gguf"]:
        assert g["maker"] == ML.maker_names()[g["base"].split("/")[0]]


def test_readme_on_disk_has_the_rendered_block():
    text = (ROOT / "README.md").read_text()
    p = SC.Problems()
    assert SC.pin_links(REG, ROOT, SC.readme_block(REG), "README.md", p).strip() in text and not p
    assert SC.list_problems(REG, text, "README.md") == []


# catalog.py at the tree every GGUF chip record ran (the equivalence seal): the hash of its code (syntax tree without
# import lines and docstrings, as the record guard compares it; empty fields left out, so the hash is the same on every
# Python version the package supports). Model lists live in tools/model_lists.py, so a page
# layout never voids a measured record; a change to catalog.py's code fails here first.
CATALOG_CODE_SHA256 = "6ee27370890170807d253e524911e098cbde59a633fe6e98da1dfd172774a301"


def code_only(data):
    tree = ast.parse(data)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            body[:] = [b for b in body if not isinstance(b, (ast.Import, ast.ImportFrom))]
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
    return _dump(tree)


def _dump(node):
    if isinstance(node, ast.AST):
        fields = [(f, getattr(node, f, None)) for f in node._fields]
        return f"{type(node).__name__}(" + ", ".join(f"{f}={_dump(v)}" for f, v in fields if v not in (None, [])) + ")"
    if isinstance(node, list):
        return "[" + ", ".join(_dump(x) for x in node) + "]"
    return repr(node)


def test_catalog_code_is_the_code_the_records_ran():
    now = (ROOT / "src" / "mashbox" / "catalog.py").read_bytes()
    assert hashlib.sha256(code_only(now).encode()).hexdigest() == CATALOG_CODE_SHA256
    assert code_only(now + b"\n# a comment\n") == code_only(now)          # a comment or docstring is not code
    assert code_only(now + b"\nTOP = 8\n") != code_only(now)                # a statement is


def test_model_lists_only_read_the_catalog():
    """the lists module names catalog functions it reads; it defines no level, row or speed of its own"""
    src = (ROOT / "tools" / "model_lists.py").read_text()
    for name in ("def level", "def gguf_rows", "def _speed", "def ours", "LEVELS =", "MAKER_NAMES ="):
        assert name not in src, name


def test_a_chip_with_its_own_brackets_prints_once_bracketed():
    """a result's chip that carries its own brackets ("CPU (AMD EPYC 7502P)") prints as "(CPU, AMD EPYC 7502P)" after the
    number on every list, never "(CPU (AMD EPYC 7502P))"; a plain chip keeps "(TPU v6e-1)" """
    chips = {r["chip"] for m in REG["models"] for r in m.get("results") or [] if " (" in r["chip"]}
    assert chips, "models.json has no chip with its own brackets: the case this test covers"
    views = {"README": SC.readme_block(REG), "profile": SC.profile_block(REG), "org card": SC.org_card_block(REG),
             "docs catalog": ML.render_markdown(REG)}
    for name, text in views.items():
        for c in chips:
            assert f"({c})" not in text, (name, c)
    assert "(CPU, AMD EPYC 7502P)" in views["README"]
    assert "(TPU v6e-1)" in views["README"]


def test_profile_row_carries_every_readme_headline():
    """The org profile's row for a model prints every headline number the README prints for it, not only the first."""
    rows = SC.profile_table(REG).splitlines()
    for m in REG["models"]:
        hl = [h for h in m.get("headline") or [] if h.get("readme")]
        row = next(r for r in rows if f"[{m['name']}]" in r)
        missing = [h["big"] for h in hl if f"**{h['big']}**" not in row]
        assert not missing, (m["id"], missing)
