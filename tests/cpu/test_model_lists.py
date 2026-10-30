"""Every model list and count on a public page is rendered from models.json, and a page that types one fails.

The README, llms.txt and the docs catalog print every row and the same count line, from tools/model_lists.py, which
reads mashbox/catalog.py and never changes it (catalog.py is in the code set of every GGUF chip record); `mashbox
models` prints every row from catalog.py itself. A new gguf[] row reaches all of them with render --write and no other
step. The GitHub org profile and the Hugging Face org card are overviews: no models block, no model named, no count, and
a family line that names exactly the families models.json has. Outside the rendered blocks a page types no count of
models, and every printed count line adds up to models.json's total.
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

# An intro in the style of the org profile's old one: a hand list of two models under a table of three.
OLD_INTRO = ("Our open model ports run on Google Cloud TPU, and anyone can run them: a 27B language model on a single "
             "chip, and Hy-Nomas, a world model in the browser, on a TPU v6e-4 slice.")
NEW_INTRO = ("We port open models to Google Cloud TPU, and anyone can run them. The table below lists every model in "
             "the current mashbox release and the chips it runs on.")
INSTALL = "```bash\n~/.local/bin/mashbox run qwen3-8-27b --prompt \"Write a haiku about the sea.\"\n```\n"


def overview(intro, reg=REG):
    """An org overview page (GitHub profile, Hugging Face org card): the company line, prose, the install lines."""
    return f"{reg['company_line']}\n\n{intro}\n\n{INSTALL}"


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
    old = [m for _, m in SC.list_problems(REG, overview(OLD_INTRO), "profile", names=True)]
    assert any("Hy-Nomas" in m for m in old), old
    assert any("'27B'" in m for m in old), old          # a size as people write it names the model too
    assert SC.list_problems(REG, overview(NEW_INTRO), "profile", names=True) == []


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
    views = {"README": SC.readme_block(reg), "llms.txt": SC.llms_block(reg), "docs": ML.render_markdown(reg),
             "mashbox models": catalog.render_text(reg)}
    assert "NewModel-9B" in views.pop("mashbox models")   # the CLI lists every row; its count line comes after 0.1.6
    for name, text in views.items():
        assert "NewModel-9B" in text, name
        head = "models and files run in mashbox: " if name == "README" else "entries: "   # lead line / count line
        assert f"{n + 1} {head}" in text, name
        assert f"{n} {head}" not in text, name


def test_docs_catalog_is_one_page_in_two_tiers():
    """MASHBOX builds first (every models[] row, its measured lines), then GGUF files that run in our engine, one
    section per family with maker, architecture and quant, then (when models.json has any) the models `mashbox run`
    does not start (elsewhere[]); the README and llms.txt carry the same tiers"""
    md = ML.render_markdown(REG)
    heads = [line for line in md.splitlines() if line.startswith("## ")]
    assert heads == [f"## {ML.BUILDS}", f"## {ML.GGUF}"] + ([f"## {ML.ELSEWHERE}"] if ML.elsewhere(REG) else [])
    builds, gguf = md.split(f"## {ML.GGUF}")
    gguf, _, listed = gguf.partition(f"## {ML.ELSEWHERE}")
    for r in ML.elsewhere(REG):
        assert r["how_to_run"] in listed and r["how_to_run"] not in builds + gguf
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
    for text in (SC.readme_block(REG), SC.llms_block(REG)):
        assert text.index(ML.BUILDS) < text.index(ML.GGUF)
        assert not ML.elsewhere(REG) or text.index(ML.GGUF) < text.index(ML.ELSEWHERE)
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
CATALOG_CODE_SHA256 = "a073217a76511c0716d61fe9248e8fae88c352845d49fafe1d2ce173232db167"


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
    views = {"README": SC.readme_block(REG), "llms.txt": SC.llms_block(REG), "docs catalog": ML.render_markdown(REG)}
    for name, text in views.items():
        for c in chips:
            assert f"({c})" not in text, (name, c)
    assert "(CPU, AMD EPYC 7502P)" in views["README"]
    assert "(TPU v6e-1)" in views["README"]


def overviews(reg):
    """The overview that lists the catalog: the README (the org profile and the Hugging Face org card list none)."""
    return {"README": SC.readme_block(reg)}


def many_rows(reg, n=30):
    """reg plus n more GGUF files (copies of a passing row and of a CPU-only row under new ids, files and publishers),
    so the catalog is far longer than any short list a page might print."""
    reg = copy.deepcopy(reg)
    cpu = next(g for g in reg["gguf"] if not catalog.chips_run(g))
    for i in range(n):
        g = copy.deepcopy(passing_row(reg) if i % 2 else cpu)
        g.update(id=f"pub{i % 4}-extra-{i}-q4-k-m", model=f"extra-{i}", publisher=f"pub{i % 4}",
                 repo=f"pub{i % 4}/Extra-{i}-GGUF", file=f"Extra-{i}-Q4_K_M.gguf", base=f"{g['base'].split('/')[0]}/Extra-{i}",
                 run=f"mashbox run pub{i % 4}/Extra-{i}-GGUF:Q4_K_M")
        reg["gguf"].append(g)
    return reg


def test_every_overview_leads_with_the_catalog_size_and_lists_every_row():
    """No overview shows fewer catalog rows than models.json has, or a count that is not models.json's: each one leads
    with the lead line (models and files, MASHBOX builds, GGUF files of how many models), then lists every MASHBOX build
    and every GGUF file, each GGUF file once with its level and chips, grouped by publisher then model."""
    for reg in (REG, with_new_row(REG), many_rows(REG)):
        c, rows = ML.counts(reg), catalog.gguf_rows(reg)
        lead = (f"{c['entries']} models and files run in mashbox: {c['ports'] + c['own']} MASHBOX builds and "
                f"{c['files']} GGUF files of {c['gguf_models']} models")
        for name, text in overviews(reg).items():
            assert text.lstrip().startswith(f"**{lead}.**"), (name, text[:200])
            assert SC.list_problems(reg, block(text), name) == [], name
            for m in catalog.live(reg):
                assert f"[{m['name']}]" in text or f"**{m['name']}**" in text, (name, m["id"])
            for r in rows:
                assert text.count(f"]({r['file_url']})") == 1, (name, r["id"])
                line = next(x for x in text.splitlines() if f"]({r['file_url']})" in x)
                assert f"| {r['level']}" in line and (", ".join(r["chips"]) or "n/a") in line, (name, r["id"], line)
            pubs = [p for p in catalog.publishers(reg) if p["name"] != catalog.MASHBOX]
            at = [text.index(f"<summary><b>{p['name']}</b>") for p in pubs]
            assert at == sorted(at), (name, [p["name"] for p in pubs])
            for p in pubs:
                k = sum(len(m["rows"]) for m in p["models"])
                assert f"<summary><b>{p['name']}</b> ({k} {'file' if k == 1 else 'files'}): " in text, (name, p["name"])
        assert ML.lead_line(reg) == lead
    wrong = SC.readme_block(REG).replace(ML.lead_line(REG), ML.lead_line(REG).replace(
        f"{ML.counts(REG)['files']} GGUF files", "2 GGUF files", 1), 1)
    assert any("lead line" in m for _, m in SC.list_problems(REG, block(wrong), "README")), "a wrong lead count passes"


def test_llms_txt_lists_every_row():
    """llms.txt names every gguf[] row once, publisher by publisher in the overviews' order, each with its catalog level
    and the chips it passed on (no "runs on" for a row without a chip test), then the count line; the file on disk is
    that render."""
    for reg in (REG, with_new_row(REG), many_rows(REG)):
        text, rows = SC.llms_block(reg), catalog.gguf_rows(reg)
        gguf = text.split(f"{ML.GGUF}:", 1)[1].split(f"\n{ML.ELSEWHERE}:", 1)[0]    # the next tier, when there is one
        lines = [x for x in gguf.splitlines() if x.startswith("- ") and "[Full catalog]" not in x]
        assert len(lines) == len(rows) == len(reg["gguf"]), (len(lines), len(rows))
        got = []
        for x in lines:
            hit = [r for r in rows if x.endswith(f"`{r['run']}`")]
            assert len(hit) == 1, x
            r = hit[0]
            got.append(r["id"])
            assert f": {r['level']}" in x and f"GGUF from {r['publisher']}" in x and f"by {r['maker']}" in x, x
            assert (f"runs on {', '.join(r['chips'])}." in x) if r["chips"] else ("runs on" not in x), x
        assert got == [r["id"] for _, _, rs in ML.by_publisher(reg) for r in rs]
        assert gguf.rstrip().endswith(f"{ML.count_line(reg)}.")
        assert SC.list_problems(reg, block(text), "llms.txt") == []
    disk = (ROOT / "llms.txt").read_text()
    assert disk.split(SC.MARK_START, 1)[1].split(SC.MARK_END, 1)[0].strip() == SC.llms_block(REG).strip()


def inside(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def test_every_catalog_page_matches_models_json():
    """The pages on disk that carry the catalog list every models.json row once and print models.json's counts: README
    (lead line), the docs catalog and llms.txt (count line), and `mashbox models` (every row; its count line comes with
    the release that carries it). The Hugging Face model cards are about one model each: no catalog count, no GGUF row."""
    rows, live = catalog.gguf_rows(REG), catalog.live(REG)
    pages = {"README.md": inside((ROOT / "README.md").read_text(), SC.MARK_START, SC.MARK_END),
             "docs/supported-models.md": inside((ROOT / "docs" / "supported-models.md").read_text(),
                                                "<!-- catalog:start -->", "<!-- catalog:end -->"),
             "llms.txt": inside((ROOT / "llms.txt").read_text(), SC.MARK_START, SC.MARK_END)}
    for name, text in pages.items():
        for r in rows:
            key = f"`{r['run']}`" if name == "llms.txt" else f"]({r['file_url']})"
            assert text.count(key) == 1, (name, r["id"], text.count(key))
        for m in live:
            assert m["name"] in text, (name, m["id"])
        want = f"**{ML.lead_line(REG)}.**" if name == "README.md" else f"{ML.count_line(REG)}."
        assert want in text, (name, want)
        assert SC.list_problems(REG, block(text), name) == [], name
    cli = catalog.render_text(REG)
    for r in rows:
        assert r["run"] in cli, r["id"]
    for m in live:
        assert f"[{m['id']}]" in cli, m["id"]
    for card in sorted((ROOT / "hf").glob("*/README.md")):
        text = card.read_text()
        assert not SC.LEAD_RE.search(text) and not SC.ENTRIES_RE.search(text), card
        assert SC.list_problems(REG, text, str(card)) == [], card
        assert not any(r["file_url"] in text for r in rows), card


OVERVIEW = ("We make open models run faster and cheaper on Google Cloud TPU. mashbox runs GGUF files from the community "
            "as they are. Families covered so far: {fams}. Every file is on "
            "[Supported models](https://mashbox.gg/docs/supported-models).")


def with_new_family(reg):
    """reg plus one passing GGUF file of a family models.json does not have yet (Phi)."""
    reg = copy.deepcopy(reg)
    g = copy.deepcopy(passing_row(reg))
    g.update(id="someone-phi-4-mini-q4-k-m", model="phi-4-mini", publisher="someone", repo="someone/Phi-4-mini-GGUF",
             file="Phi-4-mini-Q4_K_M.gguf", base=f"{g['base'].split('/')[0]}/Phi-4-mini",
             run="mashbox run someone/Phi-4-mini-GGUF:Q4_K_M")
    reg["gguf"].append(g)
    return reg


def test_org_overviews_carry_no_model_list():
    """The GitHub org profile and the Hugging Face org card are overviews: no models block, no model
    named, no lead line, count line or typed count. A family line names exactly models.json's GGUF families, in any
    order, so a new family shows as drift on both pages the day its first file lands."""
    fams = ML.gguf_families(REG)
    assert fams == sorted({ML.family(r) for r in catalog.gguf_rows(REG)}, key=str.casefold) and len(fams) >= 2
    line = ", ".join(fams[:-1]) + " and " + fams[-1]
    page = overview(OVERVIEW.format(fams=line))
    shuffled = overview(OVERVIEW.format(fams=", ".join(fams[:0:-1]) + " and " + fams[0]))
    for where in ("profile", "hf org card"):
        assert SC.overview_problems(REG, page, where) == [], where
        assert SC.overview_problems(REG, shuffled, where) == [], where
    bad = {"a models block": page + "\n" + block(SC.readme_block(REG)),
           "a model named": overview(OLD_INTRO + " " + OVERVIEW.format(fams=line)),
           "the lead line": page + f"\n**{ML.lead_line(REG)}.**\n",
           "the count line": page + f"\n{ML.count_line(REG)}.\n",
           "a typed count": page + "\nWe test 14 GGUF files.\n",
           "a family left out": overview(OVERVIEW.format(fams=line.replace(fams[1] + ", ", "", 1))),
           "a family models.json lacks": overview(OVERVIEW.format(fams="Phi, " + line)),
           "no company line": f"{OVERVIEW.format(fams=line)}\n\n{INSTALL}"}
    for what, text in bad.items():
        assert SC.overview_problems(REG, text, "profile"), what
    assert any("famil" in m for _, m in SC.overview_problems(with_new_family(REG), page, "profile")), \
        "a new family in models.json leaves the overview's family line as it was"
    assert SC.overview_problems(with_new_family(REG), overview(OVERVIEW.format(fams="Phi, " + line)), "profile") == []
