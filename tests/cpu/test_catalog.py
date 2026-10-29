"""The model catalog (models.json schema 2, mashbox/catalog.py): one source, many views. `mashbox models`, the docs
catalog block and the architecture list in docs/gguf.md are renders of models.json and of mashbox.gguf.archs.ARCHS, and
these tests fail when a view, a row or the code drift apart. GGUF rows are checked against the reader's own lists; the
record behind each claim is checked by the maintainer-only test at the end (the records are kept outside this
repository)."""
import ast
import copy
import fnmatch
import functools
import hashlib
import importlib.util
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import tomllib
import types
from contextlib import redirect_stdout
from pathlib import Path

import gguf
import pytest

from mashbox import catalog

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)
_spec = importlib.util.spec_from_file_location("gguf_fixtures", ROOT / "tools" / "gguf_fixtures" / "fixtures.py")
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)
REG = json.loads((ROOT / "src" / "mashbox" / "models.json").read_text())

# one GGUF row in the shape a CONFIRMED post gives (values made up for the checks; never a real record)
ROW = {"id": "tiny-qwen3-gguf-q4-k-m", "model": "tiny-qwen3", "publisher": "someone", "family": "Qwen3", "maker": "Qwen",
       "repo": "someone/Tiny-Qwen3-GGUF",
       "revision": "0" * 40, "file": "Tiny-Qwen3-Q4_K_M.gguf", "tag": "Q4_K_M", "sha256": "a" * 64, "bytes": 1234,
       "arch": "qwen3", "moe": False, "params_b": {"total": 0.5, "active": None}, "types": ["F32", "Q4_K", "Q6_K"],
       "license": "apache-2.0", "license_link": "https://huggingface.co/someone/Tiny-Qwen3-GGUF/blob/main/LICENSE",
       "base": "someone/Tiny-Qwen3", "in": "text", "out": "text", "run": "mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M",
       "experimental": False,
       "chips": {"v6e": {"status": "passing", "accelerator": "v6e-1", "ctx": 2048, "slots": 8, "kernel_path": "exact",
                         "measured": {"decode_tok_s_1_user": 50.5, "decode_tok_s_8_users": 300.25}}}}
BUNDLE = {"for": "gguf", "file": f"mashbox_engine_kernels-{REG['version']}-py3-none-any.whl",
          "url": "https://example.com/b.whl", "sha256": "b" * 64, "jax": "0.11.2", "libtpu": "0.0.48.*",
          "chips": ["v6e"], "types": ["Q4_K", "Q6_K"]}


# a real tiny GGUF file (qwen3, seven block types): the made-up records below seal its header, so the guard re-runs the
# fit on it; the row's bytes are its size, V6E its chip memory (JAX bytes_limit on v6e)
FIXTURE = ROOT / "tests" / "fixtures" / "gguf" / "qwen3-kquant.gguf"
V6E = 33_546_041_344


def _header(path):
    """a GGUF file's bytes up to its tensor data: what a record seals so the fit can be re-run without the weights"""
    return Path(path).read_bytes()[:gguf.GGUFReader(str(path), "r").data_offset]


def problems(reg):
    p = SC.Problems()
    SC.validate(reg, ROOT, p)
    return list(p)


def with_row(row=None, bundle=None):
    reg = copy.deepcopy(REG)
    reg["gguf"] = [copy.deepcopy(row or ROW)]
    reg["engine_kernels"] = [copy.deepcopy(bundle)] if bundle else []
    return reg


def test_models_json_is_schema_2_and_valid():
    assert REG["schema"] == 2 and REG["gguf"] == [] or all(isinstance(g, dict) for g in REG["gguf"])
    assert problems(REG) == []
    assert (ROOT / "models.json").read_bytes() == (ROOT / "src" / "mashbox" / "models.json").read_bytes()


def test_schema_1_still_validates():
    """old files (and old apps' copies) keep passing: schema 1 = no catalog blocks, no gguf / engine_kernels"""
    old = copy.deepcopy(REG)
    old["schema"] = 1
    for k in ("gguf", "engine_kernels"):
        old.pop(k)
    for m in old["models"]:
        m.pop("catalog")
    assert problems(old) == []
    old["gguf"] = []
    assert any("needs schema 2" in x for x in problems(old))


def test_a_confirmed_row_validates_and_every_rule_catches_its_bad_form():
    assert problems(with_row()) == []
    assert problems(with_row(dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], kernel_path="bundle")}), BUNDLE)) == []
    bad = {
        "revision": dict(ROW, revision="main"),
        "sha256": dict(ROW, sha256="abc"),
        "run": dict(ROW, run="mashbox run someone/Tiny-Qwen3-GGUF"),
        "licence": dict(ROW, license="unknown"),
        "date": dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], date="2026-10-28")}),
        "date value": dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], measured=dict(ROW["chips"]["v6e"]["measured"],
                                                                                       when="2026-10-28"))}),
        "date in text": dict(ROW, base="someone/Tiny-Qwen3 (read 2026-10-28T23:00Z)"),
        "status": dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], status="pending")}),
        "chip": dict(ROW, chips={"v7": ROW["chips"]["v6e"]}),
        "speed": dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], measured={})}),
        "moe": dict(ROW, moe="no"),
        "params": dict(ROW, params_b={"total": 0.5}),
        "dup": dict(ROW, id="qwen3-8-27b"),
        "no model id": {k: v for k, v in ROW.items() if k != "model"},
        "model id form": dict(ROW, model="Tiny Qwen3"),
        "no publisher": {k: v for k, v in ROW.items() if k != "publisher"},
        "publisher is not the repo's owner": dict(ROW, publisher="someone-else"),
    }
    for name, row in bad.items():
        assert problems(with_row(row)), name
    # kernel_path "bundle" needs every quantized type in the release's bundle for that chip
    need_bundle = dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], kernel_path="bundle")})
    assert any("no engine_kernels row" in x for x in problems(with_row(need_bundle)))
    assert any("no engine_kernels row" in x for x in problems(with_row(need_bundle, dict(BUNDLE, types=["Q4_K"]))))
    assert any("mashbox_engine_kernels" in x for x in problems(with_row(need_bundle, dict(BUNDLE, file="k-0.0.1.whl"))))
    assert any("no dates" in x for x in problems(with_row(need_bundle, dict(BUNDLE, libtpu="0.0.48 (2026-10-28)"))))
    dated = copy.deepcopy(REG)
    dated["models"][0]["catalog"]["family"] = "Hy-Nomas 2026-10-28"
    assert any("no dates" in x for x in problems(dated))
    # a word that only contains "date" is not a date
    assert problems(with_row(dict(ROW, base="someone/Tiny-Qwen3-candidate"))) == []


# a mixture-of-experts row (8 experts in the file's metadata, an active count below the total) that passes on v6e
MOE = dict(ROW, moe=True, params_b={"total": 0.5, "active": 0.1}, chips={"v6e": dict(
    ROW["chips"]["v6e"], routed_experts="kernel", measured=dict(ROW["chips"]["v6e"]["measured"], goodput_roofline_pct_1_user=25.0,
                                                                goodput_roofline_pct_8_users=40.0))})


def test_a_mixture_of_experts_row_needs_its_routed_kernel_and_goodput():
    """The rule: no mixture-of-experts row without a sealed chip record of its routed experts on the kernel and its
    goodput as % of the roofline. A row is a mixture of experts by its metadata (moe, or an active count below the
    total), never by its name."""
    assert problems(with_row(MOE)) == []
    bad = {
        "routed experts on the exact path": dict(MOE, chips={"v6e": dict(MOE["chips"]["v6e"], routed_experts="exact")}),
        "no routed path": dict(MOE, chips={"v6e": {k: v for k, v in MOE["chips"]["v6e"].items() if k != "routed_experts"}}),
        "no goodput": dict(MOE, chips={"v6e": dict(MOE["chips"]["v6e"], measured=ROW["chips"]["v6e"]["measured"])}),
        "no goodput for 8 users": dict(MOE, chips={"v6e": dict(MOE["chips"]["v6e"], measured=dict(
            MOE["chips"]["v6e"]["measured"], goodput_roofline_pct_8_users=None))}),
        "experts by the params, moe false": dict(MOE, moe=False),
    }
    for name, row in bad.items():
        assert any("mixture of experts" in x for x in problems(with_row(row))), name
    dense = dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], routed_experts="kernel")})
    assert any("no experts" in x for x in problems(with_row(dense)))
    assert problems(with_row(dict(ROW, family="Qwen3 MoE A3B"))) == []      # a name is never the test


def test_catalog_blocks_on_our_models():
    for m in REG["models"]:
        c = m["catalog"]
        assert c["source"] == "ours" and isinstance(c["experimental"], bool) and c["format"] in catalog.FORMAT_LABEL
        assert "status" not in c and "level" not in c
    bad = copy.deepcopy(REG)
    bad["models"][0]["catalog"]["number"] = {"headline": 99}
    assert any("catalog.number" in x for x in problems(bad))
    bad = copy.deepcopy(REG)
    del bad["models"][0]["catalog"]
    assert any("missing the catalog block" in x for x in problems(bad))


def test_kind_labels_match_the_views():
    assert catalog.KIND_LABEL == {k: v["label"] for k, v in SC.KINDS.items()}
    assert catalog.CHIPS == SC.CHIPS


def test_ids_unique_across_models_and_gguf():
    ids = [m["id"] for m in REG["models"]] + [g["id"] for g in REG["gguf"]]
    assert len(ids) == len(set(ids))


def test_mashbox_models_prints_the_render():
    from mashbox import cli
    for argv, want in ((["models"], catalog.render_text(REG)), (["models", "--json"], json.dumps(catalog.entries(REG), indent=1) + "\n")):
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert cli.main(argv) == 0
        assert buf.getvalue() == want


def test_views_are_the_render():
    """docs/gguf.md carries the generated architecture block, and any docs page with a catalog block carries the catalog,
    each equal to its render (render --write fixes them). The docs site's model list is the site's own page: no
    docs/models.md, which would take the same address."""
    mt = SC.ARCHS_RE.search((ROOT / "docs" / "gguf.md").read_text())
    assert mt, "docs/gguf.md has no generated block"
    assert mt.group(2).strip() == catalog.archs_block(REG).strip()
    for page in sorted((ROOT / "docs").rglob("*.md")):
        mt = SC.CATALOG_RE.search(page.read_text())
        if mt:
            assert mt.group(2).strip() == catalog.render_markdown(REG).strip(), page
    assert not (ROOT / "docs" / "models.md").exists()
    assert "(models.md)" not in (ROOT / "docs" / "SUMMARY.md").read_text()


ARCH_LINE = "GGUF architectures with a passing chip test (general.architecture):"


def test_the_architecture_line_claims_only_what_the_reader_does():
    """the architectures line says the reader takes these files, never that every file in them runs: files in listed
    architectures were refused at run for their tokenizer or rope scaling"""
    t = catalog.render_text(with_row())
    line = ARCH_LINE
    assert line in t and t.index(line) > t.index(ROW["run"])
    assert t[t.index(line) + len(line):].split() == [ROW["arch"], "(Qwen3)"]   # only the architecture with a passing row
    assert "run too" not in t and "mashbox inspect" not in t


def test_the_docs_block_says_when_no_gguf_file_is_listed_and_names_the_listed_architectures(monkeypatch):
    """with no GGUF row the docs catalog block says so in one generated line and names no architecture (a listed
    architecture needs a passing chip row); with rows it names exactly the architectures catalog.arch_rows lists, in the
    same words as `mashbox models`, so an architecture the catalog stops listing leaves the page with no hand edit"""
    empty = copy.deepcopy(REG)
    empty["gguf"] = []
    none_line = "No GGUF file is listed in this version."
    archs = "GGUF architectures with a passing chip test (`general.architecture`):"
    md = catalog.render_markdown(empty)
    assert none_line in md and archs not in md and catalog.arch_rows(empty) == []
    assert "general.architecture" not in md
    llama = dict(ROW, id="tiny-llama-gguf-q4-k-m", arch="llama", sha256="e" * 64, file="Tiny-Llama-Q4_K_M.gguf",
                 run="mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M")
    two = with_row()
    two["gguf"].append(llama)
    md = catalog.render_markdown(two)
    tail = md.split(archs)[1]
    assert none_line not in md and tail.strip() == "`llama` (Llama 3.x), `qwen3` (Qwen3)."
    assert archs.replace("`", "") in catalog.render_text(two)                       # the same words in every view
    keep = catalog.arch_rows
    monkeypatch.setattr(catalog, "arch_rows", lambda reg, src=None: [r for r in keep(reg, src) if r["arch"] != "llama"])
    tail = catalog.render_markdown(two).split(archs)[1]
    assert "`llama`" not in tail and "`qwen3`" in tail


def test_views_group_by_publisher_then_model_then_file():
    """one list: publisher -> model -> formats/files, MASHBOX Labs first as one publisher like the others, a GGUF file
    under its base model; no separate "our models" / "GGUF files" groups in any view or in the help that names them"""
    other = dict(ROW, id="tiny-qwen3-gguf-q8-0", tag="Q8_0", file="Tiny-Qwen3-Q8_0.gguf",
                 run="mashbox run someone/Tiny-Qwen3-GGUF:Q8_0", sha256="c" * 64)
    reg = with_row()
    reg["gguf"].append(other)
    assert problems(reg) == []
    pubs = catalog.entries(reg)["publishers"]
    assert [p["name"] for p in pubs] == ["MASHBOX Labs", "someone"]
    assert [m["name"] for m in pubs[0]["models"]] == [m["name"] for m in catalog.live(REG)]
    assert [(m["name"], m["base"], [r["id"] for r in m["rows"]]) for m in pubs[1]["models"]] == [
        ("Tiny-Qwen3", ROW["base"], [ROW["id"], other["id"]])]
    t = catalog.render_text(reg)
    assert t.index("MASHBOX Labs") < t.index("\nsomeone\n") < t.index("  Tiny-Qwen3  (base someone/Tiny-Qwen3)") < t.index(
        ROW["run"]) < t.index(other["run"])
    assert "tok/s" not in t.split("\nsomeone\n")[1]       # an exact-path row shows no speed (test below)
    md = catalog.render_markdown(reg)
    assert md.index("## MASHBOX Labs") < md.index("## someone") and f"/blob/{ROW['revision']}/{ROW['file']}" in md
    for view in (t, md, catalog.render_text(REG), catalog.render_markdown(REG), json.dumps(catalog.entries(reg))):
        assert "Our models" not in view and "GGUF files" not in view
    assert "Files in the catalog" in catalog.archs_block(reg) and ROW["id"] in catalog.archs_block(reg)
    from mashbox import cli
    for doc in (cli.__doc__, SC.__doc__):
        assert "by publisher, then model, then file" in doc
        assert "ours, then GGUF" not in doc and "our models, then" not in doc


def test_a_gguf_row_shows_a_speed_only_from_a_run_on_the_kernel_bundle():
    """an exact-path chip row's number is the reference path's: no catalog view (mashbox models, the catalog block, the
    JSON entries) shows it as the file's speed. A row whose passing run was on the kernel bundle shows its own number."""
    exact = with_row()
    bundle = with_row(dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], kernel_path="bundle")}))
    for reg, want in ((exact, None), (bundle, 50.5)):
        row = next(r for p in catalog.entries(reg)["publishers"] for m in p["models"] for r in m["rows"]
                   if r["id"] == ROW["id"])
        assert row["tok_s_1_user"] == want
        text = catalog.render_text(reg).split("\nsomeone\n")[1]
        md = catalog.render_markdown(reg).split("## someone")[1]
        assert ("50.5 tok/s for one request on TPU v6e-1" in text) == (want is not None)
        assert ("50.5 tok/s" in md) == (want is not None)
    assert catalog.speed(exact["gguf"][0]) == (None, None)


def _tree(reg):
    return [(p["name"], [(m["id"], m["name"], [r["id"] for r in m["rows"]]) for m in p["models"]])
            for p in catalog.entries(reg)["publishers"]]


def test_one_sorted_tree_whatever_the_order_of_the_rows():
    """publisher (MASHBOX Labs first, then by name), model (by name), then each format or file (by format, tag, file):
    the same tree and the same text for any order of models[] and gguf[]"""
    import random
    rows = [ROW,
            dict(ROW, id="tiny-qwen3-gguf-q8-0", tag="Q8_0", file="Tiny-Qwen3-Q8_0.gguf", sha256="c" * 64,
                 run="mashbox run someone/Tiny-Qwen3-GGUF:Q8_0"),
            dict(ROW, id="alpha-gguf-q4-k-m", model="alpha", base="someone/Alpha", repo="someone/Alpha-GGUF",
                 file="Alpha-Q4_K_M.gguf", sha256="d" * 64, run="mashbox run someone/Alpha-GGUF:Q4_K_M"),
            dict(ROW, id="tiny-qwen3-gguf-q4-k-m-b", publisher="Another", repo="Another/Tiny-Qwen3-GGUF",
                 sha256="e" * 64, run="mashbox run Another/Tiny-Qwen3-GGUF:Q4_K_M")]
    reg = copy.deepcopy(REG)
    reg["gguf"] = copy.deepcopy(rows)
    assert problems(reg) == []
    want = [("MASHBOX Labs", [(m["id"], m["name"], [m["id"]]) for m in sorted(
                catalog.live(REG), key=lambda m: (m["name"].casefold(), m["id"]))]),
            ("Another", [("tiny-qwen3", "Tiny-Qwen3", ["tiny-qwen3-gguf-q4-k-m-b"])]),
            ("someone", [("alpha", "Alpha", ["alpha-gguf-q4-k-m"]),
                         ("tiny-qwen3", "Tiny-Qwen3", ["tiny-qwen3-gguf-q4-k-m", "tiny-qwen3-gguf-q8-0"])])]
    assert _tree(reg) == want
    text, md = catalog.render_text(reg), catalog.render_markdown(reg)
    for seed in range(5):
        shuffled = copy.deepcopy(reg)
        random.Random(seed).shuffle(shuffled["models"])
        random.Random(seed + 100).shuffle(shuffled["gguf"])
        assert _tree(shuffled) == want, seed
        assert catalog.render_text(shuffled) == text and catalog.render_markdown(shuffled) == md, seed
    # one model id, one name
    reg["gguf"][1]["base"] = "someone/Tiny-Qwen3-Other"
    assert any("one name per model" in x for x in problems(reg))


def test_the_level_is_computed_never_stored():
    """a row stores each chip's outcome and an experimental flag; catalog.py computes the level (MASHBOX build, Tuned,
    Tested, Architecture supported) and validate() refuses a stored one. A gguf[] row is only added from a passing run:
    a failing outcome sits only beside a passing one, and Tuned needs a passing chip as well."""
    assert [r["level"] for r in catalog.ours(REG)] == ["MASHBOX build"] * len(catalog.live(REG))
    tuned = [{"piece": "kq"}]
    assert catalog.level(ROW) == "Tested" and catalog.level(dict(ROW, tuned=tuned)) == "Tuned"
    failing = dict(ROW, chips={"v6e": dict(ROW["chips"]["v6e"], status="failing")})
    assert any("no passing chip" in x for x in problems(with_row(failing)))
    assert catalog.level(dict(failing, tuned=tuned)) not in ("Tuned", "Tested")
    beside = dict(ROW, chips={"v6e": ROW["chips"]["v6e"], "v5e": dict(ROW["chips"]["v6e"], status="failing",
                                                                       accelerator="v5e-1")})
    assert problems(with_row(beside)) == [] and catalog.level(beside) == "Tested"
    for bad in (dict(ROW, status="supported"), dict(ROW, level="Tested"), dict(ROW, experimental="no")):
        assert problems(with_row(bad)), bad
    reg = copy.deepcopy(REG)
    reg["models"][0]["catalog"]["status"] = "supported"
    assert any("never stored" in x for x in problems(reg))
    assert "Experimental" in catalog.render_text(with_row(dict(ROW, experimental=True)))


def test_engine_kernels_rows_are_the_one_package_one_row_per_chip_and_jax():
    assert problems(with_row(bundle=BUNDLE)) == []
    old = dict(BUNDLE, file=f"mashbox_gguf_kernels-{REG['version']}-py3-none-any.whl")
    assert any("mashbox_engine_kernels" in x for x in problems(with_row(bundle=old)))
    reg = with_row(bundle=BUNDLE)
    reg["engine_kernels"].append(dict(BUNDLE, sha256="e" * 64))
    assert any("one row per (chip, JAX)" in x for x in problems(reg))
    assert any("one row per (chip, JAX)" in x for x in problems(with_row(bundle=dict(BUNDLE, chips=["v6e", "v5e"]))))
    reg["engine_kernels"][1]["jax"] = "0.11.3"
    assert problems(reg) == []


def test_bylines_follow_the_one_rule():
    for e, m in zip(catalog.ours(REG), catalog.live(REG)):
        own = m["catalog"]["format"] == "own-model"
        assert e["byline"] == ("Trained by MASHBOX Labs" if own else f"Made by {m['upstream']['maker']}, ported to TPU by MASHBOX Labs")


def test_numbers_come_from_backed_fields():
    """the one number per model is a headline or result already in models.json (validate() backs those by results)"""
    for e, m in zip(catalog.ours(REG), catalog.live(REG)):
        n = m["catalog"]["number"]
        if n is None:
            assert e["number"] is None
        elif "headline" in n:
            h = m["headline"][n["headline"]]
            assert e["number"] == {"big": h["big"], "label": h["label"], "chip": m["results"][h["results"][0]]["chip"]}
            assert e["number"]["chip"].startswith("TPU ")
            # the number is never printed without its chip (house rule: a comparison names the hardware)
            assert f"{h['big']} on {e['number']['chip']}, {h['label']}" in catalog.render_text(REG)


def test_no_app_add_on_words_in_any_view():
    """the catalog and every view of it name no app add-on (the add-on lives only in the Mac app)"""
    word = re.compile("st" + "eer", re.I)
    texts = [json.dumps(REG), catalog.render_text(REG), catalog.render_markdown(REG), catalog.archs_block(REG),
             json.dumps(catalog.entries(REG))]
    assert not any(word.search(t) for t in texts)


def test_archs_read_without_importing_gguf_equals_ARCHS():
    """catalog.archs() reads ARCHS from the source (the docs check has no gguf package); it must equal the module's"""
    from mashbox.gguf import archs as GA
    assert catalog.archs() == list(GA.ARCHS)
    for g in REG["gguf"]:
        assert g["arch"] in GA.ARCHS, g["id"]


def test_gguf_rows_types_are_block_types_the_reader_knows():
    from mashbox.ops import reference
    names = {v[0] for v in reference.TYPES.values()}          # the 26 block types the reference path reads
    for g in REG["gguf"] + [ROW]:
        assert set(g["types"]) <= names, (g["id"], set(g["types"]) - names)


def test_run_strings_pick_exactly_the_file():
    """each row's command parses as `mashbox run REPO:TAG [flags]`, and llama.cpp's pick rule on that repo's file list
    (the row's file among others) chooses exactly the row's file"""
    from mashbox import cli
    from mashbox.gguf import resolve as GR
    for g in REG["gguf"] + [ROW]:
        argv = g["run"].split()[1:]
        a = cli.build_parser().parse_args(argv)
        repo, tag = GR.split_repo_tag(a.model)
        assert (repo, tag) == (g["repo"], g["tag"]), g["id"]
        listing = ["README.md", f"imatrix_{g['tag']}.gguf", f"mmproj-{g['tag']}.gguf", g["file"], "other-Q8_0.gguf"]
        assert GR.pick(listing, tag) == g["file"], g["id"]


def test_app_reads_schema_2_as_schema_1():
    """the Mac app's catalog() lists the same rows from the schema 2 file as from the schema 1 form (it reads models[])"""
    from mashbox.ui import app
    old = copy.deepcopy(REG)
    old["schema"] = 1
    for k in ("gguf", "engine_kernels"):
        old.pop(k)
    for m in old["models"]:
        m.pop("catalog")
    rows = {}
    for name, reg in (("v1", old), ("v2", REG)):
        orig = (app.registry, app.active_server)
        app.registry, app.active_server = (lambda refresh=False, r=reg: (r, "test")), (lambda: None)
        try:
            rows[name] = app.catalog()["models"]
        finally:
            app.registry, app.active_server = orig
    assert rows["v1"] == rows["v2"] and rows["v2"]


EVIDENCE = Path(os.environ.get("MASHBOX_CATALOG_EVIDENCE") or ROOT / "internal" / "catalog-evidence.json")
# set by the release gate: a missing sidecar or records root then fails the record guard instead of skipping it
REQUIRE = os.environ.get("MASHBOX_REQUIRE_RECORDS") == "1"
ARCHS_SRC = "src/mashbox/gguf/archs/__init__.py"
# code set v1 of a GGUF row: the `mashbox run <gguf>` entry path by role. A row's
# record stands only when each of these files, everything they import inside src/mashbox, the arch's module and every
# ops/ file are byte-equal to the code it ran, and so are the engine_kernels rows of models.json the loader reads.
CODE_SET_VERSION = 1
CODE_SET = {
    "entry": ("src/mashbox/__main__.py", "src/mashbox/__init__.py", "src/mashbox/cli.py"),
    "resolver and loader": ("src/mashbox/gguf/resolve.py", "src/mashbox/gguf/reader.py", "src/mashbox/gguf/tokenizer.py",
                            "src/mashbox/hub.py"),
    "fit": ("src/mashbox/gguf/__init__.py",),
    "architectures": ("src/mashbox/gguf/archs/__init__.py",),
    "ops": ("src/mashbox/ops/__init__.py", "src/mashbox/ops/reference.py"),
    "engine and programs": ("src/mashbox/runtime/engine.py", "src/mashbox/runtime/programs.py",
                            "src/mashbox/runtime/compile_key.py", "src/mashbox/runtime/tenancy.py"),
    "sampler": ("src/mashbox/runtime/sampler.py",),
    "chat": ("src/mashbox/runtime/chat.py",),
    "server": ("src/mashbox/runtime/server.py",),
}
GGUF_CODE = tuple(f for files in CODE_SET.values() for f in files)
NEAR_TIE_GAP = 0.5   # the locked near-tie rule: a first divergence where llama.cpp's top-two gap is under 0.5 logit
# jax's device_kind of each chip type, as a record's "device" holds it
DEVICE_CHIP = {"TPU v5 lite": "v5e", "TPU v6 lite": "v6e", "TPU v5": "v5p", "TPU v5p": "v5p"}
SETTLED = "PRECISION"
SUMS_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _archs_map(text):
    """ARCHS (general.architecture -> module name) from the source of mashbox/gguf/archs/__init__.py, not imported"""
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARCHS" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def arch_module_files(src=None):
    """general.architecture -> the source file of its module, from ARCHS"""
    text = Path(src or ROOT / ARCHS_SRC).read_text()
    return {a: "src/" + m.replace(".", "/") + ".py" for a, m in _archs_map(text).items()}


def _module_file(repo, name):
    """mashbox.a.b -> "src/mashbox/a/b.py" or "src/mashbox/a/b/__init__.py"; None outside src/mashbox"""
    if name.split(".")[0] != "mashbox":
        return None
    base = repo / "src" / Path(*name.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p.relative_to(repo).as_posix()
    return None


def import_closure(repo, files):
    """files plus every module they import inside src/mashbox, at any depth and from any scope (an ast walk; relative
    imports are resolved against the file's package). `from pkg import name` counts pkg/name.py when that module
    exists, else pkg itself; the __init__.py of a package a module only sits in is not counted."""
    seen, todo = set(), list(files)
    while todo:
        f = todo.pop()
        if not f or f in seen or not (repo / f).is_file():
            continue
        seen.add(f)
        pkg = Path(f).with_suffix("").parts[1:-1]
        for node in ast.walk(ast.parse((repo / f).read_text())):
            if isinstance(node, ast.Import):
                todo += [_module_file(repo, a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = (list(pkg[:len(pkg) - node.level + 1]) if node.level else []) + (node.module or "").split(".")
                mod = ".".join(x for x in mod if x)
                todo += [_module_file(repo, f"{mod}.{a.name}") or _module_file(repo, mod) for a in node.names]
    return seen


def gguf_code_files(repo, arch):
    """The code a GGUF row's numbers depend on (code set v1): every CODE_SET file and the arch's module with everything
    they import inside src/mashbox, plus every ops/ file. [] when ARCHS has no such arch."""
    mod = _archs_map((repo / ARCHS_SRC).read_text()).get(arch)
    if mod is None:
        return []
    ops = [p.relative_to(repo).as_posix() for p in (repo / "src" / "mashbox" / "ops").rglob("*.py")]
    return sorted(import_closure(repo, [*GGUF_CODE, _module_file(repo, mod), *ops]))


@functools.lru_cache(maxsize=None)
def _git(git_dir, *args):
    """git's stdout as bytes, None when git fails (no such commit or path)"""
    r = subprocess.run(["git", "-C", git_dir, *args], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def code_problems(repo, git_dir, commit, arch, what, kernels=None, chip_of=None):
    """the record's code commit (in git_dir) against this tree: ARCHS maps the arch to the same module, every file of
    gguf_code_files is byte-equal, and no ops/ file the record ran is missing here"""
    if _git(git_dir, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        return [f"{what}: the code the record ran ({commit[:12]}) is not in {git_dir}"]
    out = []
    then = _archs_map((_git(git_dir, "show", f"{commit}:{ARCHS_SRC}") or b"").decode()).get(arch)
    now = _archs_map((repo / ARCHS_SRC).read_text()).get(arch)
    if then != now:
        out.append(f"{what}: ARCHS runs {arch} with {now}, the record's code ran it with {then}")
    need = gguf_code_files(repo, arch)
    if not need:
        out.append(f"{what}: no module for {arch} in ARCHS")
    out += [f"{what}: code set member {f} is not in this tree" for f in GGUF_CODE if not (repo / f).is_file()]
    for f in need:
        old = _git(git_dir, "show", f"{commit}:{f}")
        if old is None:
            out.append(f"{what}: {f} is not in the code the record ran ({commit[:12]})")
        elif old != (repo / f).read_bytes():
            out.append(f"{what}: {f} differs from the code the record ran ({commit[:12]})")
    if chip_of is not None:     # models.json's engine_kernels rows pick the kernel package the loader runs (rule 2a)
        try:
            then_k = json.loads(_git(git_dir, "show", f"{commit}:models.json") or b"{}").get("engine_kernels") or []
        except ValueError:
            then_k = None
        mine = [k for k in kernels or [] if chip_of in k.get("chips", [])]
        if then_k is None or [k for k in then_k if chip_of in k.get("chips", [])] != mine:
            out.append(f"{what}: the engine_kernels rows for {chip_of} differ from the code the record ran ({commit[:12]})")
    then_ops = (_git(git_dir, "ls-tree", "-r", "--name-only", commit, "--", "src/mashbox/ops") or b"").decode().split()
    out += [f"{what}: {f} ran in the record and is not in this tree" for f in then_ops
            if f.endswith(".py") and not (repo / f).is_file()]
    return out


def _md_cell(text, field):
    """table[COL=VALUE, ...].COLUMN: the first number in COLUMN of the first Markdown table row whose cells match"""
    mt = re.fullmatch(r"table\[(.+)\]\.(.+)", field)
    want = dict(kv.split("=", 1) for kv in mt.group(1).split(", "))
    head = None
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            head = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if head is None:
            head = cells
        elif any(c.strip("-: ") for c in cells):
            row = dict(zip(head, cells))
            if mt.group(2) in row and all(row.get(k) == v for k, v in want.items()):
                return float(re.search(r"\d[\d,]*(?:\.\d+)?", row[mt.group(2)]).group(0).replace(",", ""))
    raise KeyError(field)


def value_of(rec, field):
    """A record's value. JSON: a dotted path (weights.n_weights; goodput[users=8].pct is the pct of the one element of
    the list goodput whose users is 8), median(LIST[].A/B) (the median over LIST of A / B) or
    sum(LIST[].A)/sum(LIST[].B). Markdown (rec is its path): table[COL=VALUE, ...].COLUMN."""
    if isinstance(rec, Path):
        return _md_cell(rec.read_text(), field)
    mt = re.fullmatch(r"median\((\w+)\[\]\.(\w+)/(\w+)\)", field)
    if mt:
        return statistics.median(x[mt.group(2)] / x[mt.group(3)] for x in rec[mt.group(1)])
    mt = re.fullmatch(r"sum\((\w+)\[\]\.(\w+)\)/sum\((\w+)\[\]\.(\w+)\)", field)
    if mt:
        return sum(x[mt.group(2)] for x in rec[mt.group(1)]) / sum(x[mt.group(4)] for x in rec[mt.group(3)])
    for k in field.split("."):
        mt = re.fullmatch(r"(\w+)\[(\w+)=([^\]]+)\]", k)
        if mt:
            hit = [x for x in rec[mt.group(1)] if str(x.get(mt.group(2))) == mt.group(3)]
            if len(hit) != 1:
                raise KeyError(k)
            rec = hit[0]
        else:
            rec = rec[k]
    return rec


def compare_classes(cmp):
    """(full, near, others) recomputed from the compare's rows with the locked rule: full = no first difference; near =
    a first difference where llama.cpp's top-two gap (ref_gap) is under NEAR_TIE_GAP; every other row is an other,
    listed as (prompt, step, gap)"""
    full, near, others = 0, 0, []
    for x in cmp.get("rows") or []:
        if x.get("first_diff") is None:
            full += 1
        elif isinstance(x.get("ref_gap"), (int, float)) and x["ref_gap"] < NEAR_TIE_GAP:
            near += 1
        else:
            others.append((x.get("i"), x.get("first_diff"), x.get("ref_gap")))
    return full, near, others


def seal_problems(sums, root, what):
    """every line of a SHA256SUMS seal: "<64 hex>  <name>", the name inside the seal's folder (not absolute, no ..),
    listed once, the file there and under the records root, with that sha256"""
    out, seen, sums = [], set(), Path(sums)
    for n, line in enumerate(sums.read_text().splitlines(), 1):
        if not line.strip():
            continue
        m = SUMS_LINE.fullmatch(line.strip())
        if not m:
            out.append(f"{what}: {sums.name}:{n} is not a sha256sum line")
            continue
        name = m.group(2)
        if name.startswith("/") or ".." in Path(name).parts:
            out.append(f"{what}: {sums.name}:{n} {name} is not under the seal's folder "
                       f"({'absolute' if name.startswith('/') else '..'})")
            continue
        key = os.path.normpath(name)
        if key in seen:
            out.append(f"{what}: {sums.name}:{n} {name} is listed twice")
            continue
        seen.add(key)
        p = sums.parent / name
        if not p.is_file():
            out.append(f"{what}: {sums.name}:{n} {name} is not there")
        elif Path(root).resolve() not in p.resolve().parents:
            out.append(f"{what}: {sums.name}:{n} {name} is not under {root}")
        elif _sha(p) != m.group(1):
            out.append(f"{what}: {sums.name}:{n} {name} sha256 differs from its seal line")
    return out


def _sums(p):
    """SHA256SUMS -> {normalised absolute path: sha256}; names are relative to the file's folder (sha256sum's format)"""
    out = {}
    for line in Path(p).read_text().splitlines():
        m = SUMS_LINE.fullmatch(line.strip())
        if m:
            out[os.path.normpath(Path(p).parent / m.group(2))] = m.group(1)
    return out


def evidence_problems(reg, ev, repo, work=None):
    """The records behind every catalog claim; [] when all hold. Everything is read from the records,
    never from values typed into the sidecar: each record file exists under ev["records_root"] with its sha256; a GGUF
    chip row's record, compare, file-*.sha256 and plan-*.txt are listed in its SHA256SUMS with those sha256s; the file,
    repo and revision are the ones file-*.sha256 and the plan name; the record ran this row's file, on the code commit and
    kq kernel commit the sidecar names (its code_sha), with nothing on the exact path when the row says kernel_path
    "bundle"; every number equals the value read from its record; that code commit's GGUF files (gguf_code_files) equal
    this tree's (read with git from ev["code_repo"]); and the release bundle's manifest is built from the same kq commit
    and has every quantized type of the file. Our models' parameter counts and their one number are read from records
    the same way. A GGUF chip row's record is the zero-flag run and its slots and context are this tree's fit re-run
    on the sealed header (run_problems; the sparse file goes under work, a new temporary folder when None)."""
    out = []
    root = Path(ev["records_root"])

    def under(p):
        return root.resolve() in Path(p).resolve().parents

    def read(src, what):
        p = Path(src["path"])
        if not under(p):
            out.append(f"{what}: {p} is not under {root}")
        elif not p.is_file():
            out.append(f"{what}: {p} does not exist")
        elif _sha(p) != src["sha256"]:
            out.append(f"{what}: {p} sha256 differs")
        else:
            return json.loads(p.read_text()) if p.suffix == ".json" else p
        return None

    def got(rec, field, what):
        try:
            return value_of(rec, field)
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, AttributeError):
            out.append(f"{what}: the record has no {field}")
            return None

    def number(src, value, what):
        rec = read(src, what)
        if rec is None:
            return
        v = got(rec, src["field"], what)
        if v is not None and "round" in src:
            v = round(v, src["round"])
        if v is not None and v != value:
            out.append(f"{what}: models.json says {value}, the record's {src['field']} is {v}")

    for m in reg["models"]:
        e = ev["models"].get(m["id"])
        if e is None:
            out.append(f"{m['id']}: no evidence entry")
            continue
        tot = m["catalog"]["params_b"]["total"]
        pe = e["catalog.params_b.total"]
        if pe["value"] != tot:
            out.append(f"{m['id']}: params {tot} vs evidence {pe['value']}")
        for s in pe.get("sources", []):
            if "path" in s:
                number(s, s["value"], f"{m['id']} params")
                if round(s["value"] / 1e9, 1) != tot:
                    out.append(f"{m['id']}: {s['value']} weights is not {tot}B")
        shown, ne = catalog.number(m), e.get("catalog.number") or {}
        if shown is None:
            continue
        if "record" not in ne:
            out.append(f"{m['id']}: the number {shown['big']} has no record")
            continue
        rec = read(ne["record"], f"{m['id']} number")
        if rec is None:
            continue
        nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", shown["big"])]
        vals = [got(rec, f, f"{m['id']} number") for f in ne["fields"]]
        if vals != nums:
            out.append(f"{m['id']}: the number says {shown['big']}, the record's {ne['fields']} are {vals}")
        text = rec.read_text() if isinstance(rec, Path) else json.dumps(rec)
        if shown["chip"] not in text:
            out.append(f"{m['id']}: the number is shown on {shown['chip']}, and its record never names that chip")

    git_dir = ev.get("code_repo")
    if reg.get("gguf") and (not git_dir or not under(git_dir)):
        out.append(f"code_repo {git_dir} is not under {root}")
        git_dir = None
    manifests = {}
    for k in reg.get("engine_kernels") or []:
        ke = (ev.get("engine_kernels") or {}).get(k["file"]) or {}
        man = read(ke["manifest"], f"{k['file']} manifest") if "manifest" in ke else None
        if isinstance(man, dict):       # review R12-F01: the manifest's impls are the attested ones
            out += attest_problems(k["file"], ke, man, read)
        for ch in k["chips"]:
            manifests[ch] = man
    for g in reg.get("gguf") or []:
        for chip, r in g["chips"].items():
            what = f"{g['id']} {chip}"
            e = (ev.get("gguf") or {}).get(g["id"], {}).get(chip)
            if e is None:
                out.append(f"{what}: no evidence entry")
                continue
            have, listed = {}, {}
            for key in ("record", "compare", "sha256sums", "file", "plan", "card", "header"):
                have[key] = read(e[key], f"{what} {key}") if key in e else out.append(f"{what}: no {key} in the evidence")
            if have["sha256sums"] is not None:
                listed = _sums(have["sha256sums"])
                out += seal_problems(have["sha256sums"], root, what)
                for key in ("record", "compare", "file", "plan", "card", "header"):
                    if key in e and listed.get(os.path.normpath(e[key]["path"])) != e[key]["sha256"]:
                        out.append(f"{what}: {e[key]['path']} is not in {have['sha256sums'].name} with its sha256")
            if have["file"] is not None:
                lines = have["file"].read_text().split("\n")
                m = SUMS_LINE.fullmatch(lines[0].strip())
                size = lines[1].strip() if len(lines) > 1 else ""
                if not size:
                    out.append(f"{what}: {have['file'].name} has no size line")
                if not m or (m.group(1), Path(m.group(2)).name) != (g["sha256"], g["file"]) or (
                        size and size != str(g["bytes"])):
                    out.append(f"{what}: {have['file'].name} names another file than {g['file']} {g['sha256']}")
            if have["plan"] is not None and have["plan"].read_text().split() != [
                    g["repo"], g["revision"], g["file"], g["sha256"]]:
                out.append(f"{what}: {have['plan'].name} is not {g['repo']} @ {g['revision']} {g['file']}")
            card = have["card"]
            if card is not None and not isinstance(card, dict):
                out.append(f"{what}: the card snapshot is not JSON")
            elif card is not None:      # R12-F08: the row's card fields are the pinned card's, never typed
                cd = card.get("cardData") or {}
                base, link = cd.get("base_model"), cd.get("license_link")
                for field, ours, theirs in (("repo", g["repo"], card.get("id")), ("revision", g["revision"], card.get("sha")),
                                            ("license", g["license"], cd.get("license"))):
                    if ours != theirs:
                        out.append(f"{what}: {field} {ours!r}, the pinned card says {theirs!r}")
                if link is not None and g["license_link"] != link:
                    out.append(f"{what}: license_link {g['license_link']!r}, the pinned card says {link!r}")
                elif link is None and not str(g["license_link"]).startswith(f"https://huggingface.co/{g['repo']}/"):
                    out.append(f"{what}: license_link {g['license_link']!r}, and the pinned card names none, so it is a file "
                               f"of {g['repo']}")
                if g["base"] not in (base if isinstance(base, list) else [base]):
                    out.append(f"{what}: base {g['base']!r}, the pinned card says {base!r}")
            code, kq = e["code"]["commit"], (e.get("kernels") or {}).get("commit")
            rec = have["record"]
            if rec is not None and not isinstance(rec, dict):
                out.append(f"{what}: the record is not a JSON record")
            elif rec is not None:
                want = code[:7] + (f"+kq-{kq[:7]}" if kq else "")
                if rec.get("code_sha") != want:
                    out.append(f"{what}: the record ran {rec.get('code_sha')}, the evidence names {want}")
                w = rec.get("weights") or {}
                if w.get("file") != g["file"] or rec.get("file", g["file"]) != g["file"]:
                    out.append(f"{what}: the record ran {w.get('file')}, not {g['file']}")
                if r["kernel_path"] == "bundle" and w.get("reference_bytes") != 0:
                    out.append(f"{what}: kernel_path bundle, and the record ran {w.get('reference_bytes')} bytes on the "
                               "exact path")
                dev, n = rec.get("device"), rec.get("n_devices")
                if r["accelerator"] != f"{DEVICE_CHIP.get(dev, dev)}-{n}":
                    out.append(f"{what}: accelerator {r['accelerator']}, the record ran on {dev} x{n}")
                for key in ("ctx", "slots"):
                    if r[key] != rec.get(key):
                        out.append(f"{what}: {key} {r[key]}, the record ran {rec.get(key)}")
                if g["moe"] is True and r["status"] == "passing":     # the sealed header pins moe to the file
                    out += moe_problems(rec, chip, reg.get("engine_kernels") or [], what)
                out += run_problems(g, r, e, rec, have.get("header"), manifests.get(chip), listed, read, under,
                                    work, chip, what)
                krow = next((k for k in reg.get("engine_kernels") or [] if chip in (k.get("chips") or [])), None)
                out += pins_problems(rec, krow, repo, chip, what, exact=r["kernel_path"] == "exact")
            cmp = have["compare"]
            if r["status"] == "passing" and isinstance(cmp, dict):     # R12-F05: read from the rows, not the summary
                full, near, others = compare_classes(cmp)
                s, rows = cmp.get("summary") or {}, cmp.get("rows") or []
                said = (s.get("full_match"), s.get("near_tie_divergences"), s.get("other_divergences"))
                if len(rows) != s.get("prompts"):
                    out.append(f"{what}: the compare has {len(rows)} rows for {s.get('prompts')} prompts")
                if said != (full, near, len(others)):
                    out.append(f"{what}: the compare's rows give full {full} / near {near} / other {len(others)}, its "
                               f"summary says {said[0]} / {said[1]} / {said[2]}")
                if others and "settle" not in e:
                    out.append(f"{what}: passing, and the compare has {len(others)} other divergences and no settle")
                elif others:
                    out += settle_problems(e, others, what, read, root)
            for name, value in r["measured"].items():
                if name not in e["numbers"]:
                    out.append(f"{what}: measured.{name} has no record")
                else:
                    number(e["numbers"][name], value, f"{what} {name}")
            if git_dir:
                out += code_problems(repo, git_dir, code, g["arch"], what, reg.get("engine_kernels") or [], chip)
            if r["kernel_path"] == "bundle":
                man = manifests.get(chip)
                if man is None:
                    out.append(f"{what}: kernel_path bundle, and no manifest of this release's bundle on {chip}")
                    continue
                if not kq or man.get("commit") != kq:
                    out.append(f"{what}: the record ran kq {kq}, the release bundle is built from {man.get('commit')}")
                for t in g["types"]:
                    if t not in catalog.UNQUANTIZED and t not in (man.get("impl") or {}):
                        out.append(f"{what}: the release bundle has no {t} kernel")
    return out


# the dists every chip record names in versions{} (importlib.metadata.version in the run's process; python is
# platform.python_version()), as posted for harness v2
RUN_DISTS = ("python", "jax", "jaxlib", "libtpu", "gguf", "tokenizers", "jinja2", "numpy", "mashbox_engine_kernels")


def tree_pins(repo, extras=("tpu",)):
    """name -> the versions this tree pins exactly (name==version) for what a chip run installs (pip install
    mashbox[tpu]): pyproject.toml's dependencies and the tpu extra"""
    p = tomllib.loads((Path(repo) / "pyproject.toml").read_text())["project"]
    out = {}
    for spec in [*p.get("dependencies", []), *(s for x in extras for s in (p.get("optional-dependencies") or {}).get(x, []))]:
        m = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*(\[[^\]]*\])?\s*==\s*([^;\s]+)\s*(;.*)?", spec)
        if m:
            out.setdefault(m.group(1).lower().replace("-", "_"), set()).add(m.group(3))
    return out


def bundle_version(krow):
    """the version of an engine_kernels row's wheel (mashbox_engine_kernels-<version>-...whl), or None"""
    m = re.match(r"mashbox_engine_kernels-([^-]+)-", str(krow.get("file") or ""))
    return m.group(1) if m else None


def pins_problems(rec, krow, repo, chip, what, exact=False):
    """review R12, pinned versions: the record names every dist it ran (versions{}); each one this tree pins is the
    pinned version; and jax and libtpu are the ones the release bundle on the chip is built for. The kernel package by
    path: an exact-path record (zero flags, no kernel package installed) leaves it out or names it None, never a
    version; a bundle record names the version of the release bundle on the chip."""
    v = rec.get("versions")
    if not isinstance(v, dict):
        return [f"{what}: the record has no versions (the dists the run imported)"]
    pins, out = tree_pins(repo), []
    for n in sorted(set(RUN_DISTS) | set(pins)):
        if exact and n == KERNEL_PACKAGE:
            if v.get(n) is not None:
                out.append(f"{what}: kernel_path exact, and the record ran {n} {v[n]} (an exact-path zero-flag run has "
                           "no kernel package installed)")
        elif not isinstance(v.get(n), str) or not v[n]:
            out.append(f"{what}: the record's versions have no {n}")
        elif n == KERNEL_PACKAGE and krow is not None and v[n] != bundle_version(krow):
            out.append(f"{what}: the record ran {n} {v[n]}, the release bundle on {chip} is {bundle_version(krow)}")
    for n, want in sorted(pins.items()):
        if isinstance(v.get(n), str) and v[n] not in want:
            out.append(f"{what}: the record ran {n} {v[n]}, this tree pins {' or '.join(sorted(want))}")
    if krow is not None:
        if v.get("jax") != krow.get("jax"):
            out.append(f"{what}: the record ran jax {v.get('jax')}, the release bundle on {chip} needs {krow.get('jax')}")
        if not fnmatch.fnmatchcase(str(v.get("libtpu")), str(krow.get("libtpu"))):
            out.append(f"{what}: the record ran libtpu {v.get('libtpu')}, the release bundle on {chip} needs "
                       f"{krow.get('libtpu')}")
    return out


def attest_problems(name, ke, man, read):
    """review R12-F01, kernel attestation: the evidence seals the bundle's attestation (ATTESTATION.json), the manifest
    names it by sha256, both come from one kq commit, and each type's impl in the manifest is the attested impl"""
    if "attestation" not in ke:
        return [f"{name}: no attestation in the evidence"]
    a = read(ke["attestation"], f"{name} attestation")
    if a is None:
        return []
    if not isinstance(a, dict):
        return [f"{name}: the attestation is not JSON"]
    out, named = [], (man.get("attestation") or {}).get("sha256")
    if named != ke["attestation"]["sha256"]:
        out.append(f"{name}: the manifest names attestation {named}, the evidence seals {ke['attestation']['sha256']}")
    if a.get("kq_commit") != man.get("commit"):
        out.append(f"{name}: the manifest is built from kq {man.get('commit')}, the attestation hashed kq "
                   f"{a.get('kq_commit')}")
    ai, mi = a.get("impl") or {}, man.get("impl") or {}
    for t in sorted(set(man.get("types") or []) | set(mi)):
        if not ai.get(t) or mi.get(t) != ai.get(t):
            out.append(f"{name}: {t}'s impl in the manifest is {mi.get(t)}, the attested impl is {ai.get(t)}")
    return out


class _NoPlanes(Exception):
    pass


def _bundle_module(man):
    """the release bundle as ops.load_kernels sees it, for re-running the fit: TYPES and plane_bytes(t, (rows, cols))
    from its manifest; a (type, rows, cols) the manifest has no planes for raises _NoPlanes"""
    mod = types.ModuleType("mashbox_catalog_fit_bundle")
    tid = man.get("type_ids") or {}
    name = {int(v): k for k, v in tid.items()}
    mod.TYPES = tuple(int(tid[t]) for t in man.get("types") or [] if t in tid)

    def plane_bytes(t, shape):
        key = f"{name.get(int(t), t)}:{int(shape[0])}:{int(shape[1])}"
        if key not in (man.get("plane_bytes") or {}):
            raise _NoPlanes(key)
        return man["plane_bytes"][key]
    mod.plane_bytes = plane_bytes
    return mod


def run_problems(g, r, e, rec, header, man, listed, read, under, work, chip, what):
    """review R12-F04, a GGUF chip row: its record is the zero-flag run, and its slots and context are this tree's fit
    on the file. The record seals how it ran (invocation: the argv `mashbox` got and env, every MASHBOX_ variable set)
    and the fit's inputs (fit: the chip's bytes_limit, which is the run's memory.bytes_limit, and the kernel types it
    ran). The row's record ran `mashbox run <file>` with no flag and no MASHBOX_ variable. A forced-slot record of the
    run is listed under forced, ran with flags and backs no number; every record in the seal is one of the two. The
    fit is re-run with this tree's code (cli.gguf_slots): the sealed header (the file's bytes up to its tensor data)
    with a sparse tail to the file's size, the parser's defaults for the record's argv, the record's bytes_limit, and
    the release bundle's kernel types and plane sizes (kernel_path bundle) or none (exact). The same header gives the
    file's expert count (its own metadata, never a record field), which must be the row's moe."""
    out = []
    inv = rec.get("invocation") if isinstance(rec.get("invocation"), dict) else {}
    if not isinstance(inv.get("argv"), list) or not isinstance(inv.get("env"), dict):
        return [f"{what}: the record has no invocation (the argv and MASHBOX_ env the run got)"]
    argv, env = [str(x) for x in inv["argv"]], inv["env"]

    def zero_flag(av, en):
        return not en and len(av) == 2 and av[0] == "run" and (
            Path(av[1]).name == g["file"] or av[1] == f"{g['repo']}:{g['tag']}")
    if not zero_flag(argv, env):
        out.append(f"{what}: the record ran `mashbox {' '.join(argv)}`" + (
            f" with {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}" if env else "") +
            f"; the row's record is the zero-flag run `mashbox run {g['file']}`")
    forced = set()
    for i, src in enumerate(e.get("forced") or []):
        p, fr = os.path.normpath(src["path"]), read(src, f"{what} forced[{i}]")
        forced.add(p)
        if listed.get(p) != src["sha256"]:
            out.append(f"{what}: {src['path']} is not in the run's seal with its sha256")
        if isinstance(fr, dict):
            fi = fr.get("invocation") if isinstance(fr.get("invocation"), dict) else {}
            if zero_flag([str(x) for x in fi.get("argv") or []], fi.get("env")):
                out.append(f"{what}: a forced-slot record ran with zero flags ({Path(p).name}); only the row's record "
                           "is the zero-flag run")
            if fr.get("file") != g["file"]:
                out.append(f"{what}: the forced-slot record {Path(p).name} ran {fr.get('file')}, not {g['file']}")
    for name, src in (e.get("numbers") or {}).items():
        if os.path.normpath(src["path"]) in forced:
            out.append(f"{what}: measured.{name} is read from a forced-slot record; a row's numbers come from its "
                       "zero-flag record")
    own = forced | {os.path.normpath(e["record"]["path"])}
    for p in sorted(listed):
        if p.endswith(".json") and p not in own and under(p) and Path(p).is_file():
            try:
                d = json.loads(Path(p).read_text())
            except ValueError:
                continue
            if isinstance(d, dict) and "invocation" in d:
                out.append(f"{what}: {Path(p).name} is a record of this run that the evidence names neither as the "
                           "row's record nor as forced")
    fit = rec.get("fit") if isinstance(rec.get("fit"), dict) else {}
    lim, mem = fit.get("bytes_limit"), (rec.get("memory") or {}).get("bytes_limit")
    if not isinstance(lim, int) or isinstance(lim, bool) or lim <= 0:
        return out + [f"{what}: the record has no fit.bytes_limit (the chip's memory as JAX reported it)"]
    if lim != mem:
        out.append(f"{what}: the fit's bytes_limit {lim} is not the run's memory.bytes_limit {mem}")
    bundle = r["kernel_path"] == "bundle"
    if bundle and not isinstance(man, dict):
        return out          # refused below: no manifest of this release's bundle on the chip
    kt = sorted(man.get("types") or []) if bundle else []
    if not isinstance(fit.get("kernel_types"), list) or sorted(map(str, fit["kernel_types"])) != kt:
        out.append(f"{what}: the record's kernel types {fit.get('kernel_types')} are not the release bundle's {kt}"
                   if bundle else f"{what}: kernel_path exact, and the record ran kernel types {fit.get('kernel_types')}")
    impl, want = rec.get("impl"), {t: (man.get("impl") or {}).get(t) for t in kt}     # review R12-F01
    if not isinstance(impl, dict):
        out.append(f"{what}: the record has no impl (the kernel impl hash of each type it ran)")
    else:
        for t in sorted(set(impl) | set(want)):
            if impl.get(t) != want.get(t):
                out.append(f"{what}: the record ran {t} on impl {impl.get(t)}, the release bundle's is {want.get(t)}")
    if header is None:
        return out
    size = header.stat().st_size
    if size > g["bytes"]:
        return out + [f"{what}: the sealed header is {size} bytes, longer than the file's {g['bytes']}"]
    dst = Path(work or tempfile.mkdtemp(prefix="mashbox-catalog-fit-")) / f"{g['id']}-{chip}" / g["file"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(header.read_bytes())
        fh.truncate(g["bytes"])
    from mashbox import cli, gguf as G, ops
    saved, mod = dict(ops._KERNELS), "mashbox_catalog_fit_bundle"
    try:
        if bundle:
            sys.modules[mod] = _bundle_module(man)
        ops.load_kernels(module=mod if bundle else "off")
        f = G.File(str(dst))
        have = sorted(f.census()[0])
        if f.arch != g["arch"]:
            return out + [f"{what}: the sealed header is a {f.arch} file, the row says {g['arch']}"]
        experts = int(f.get(f"{f.arch}.expert_count") or 0)
        if (experts > 0) != (g["moe"] is True):
            out.append(f"{what}: the sealed header's metadata has {experts} experts, the row says moe {g['moe']}")
        if have != sorted(g["types"]):
            return out + [f"{what}: the sealed header's block types are {have}, the row says {sorted(g['types'])}"]
        f, _, c = G.open_model(str(dst))
        a = cli.build_parser().parse_args(argv)
        s = cli.gguf_slots(a, f, c, ops.kernel_types(), lim=lim)
        own, G._limit = G._limit, lambda: lim       # then run's own check at the record's limit (cli.load_gguf)
        try:
            G.check_fits(f, c, a.ctx, s, kernel_types=ops.kernel_types())
        except G.GGUFError as x:
            return out + [f"{what}: the candidate refuses this file at bytes_limit {lim} with {s} x {a.ctx} ({x}); the "
                          f"record ran {rec.get('slots')} x {rec.get('ctx')}"]
        finally:
            G._limit = own
    except _NoPlanes as x:
        return out + [f"{what}: the release bundle has no planes for {x.args[0]}, so the fit cannot be re-run"]
    except (G.GGUFError, SystemExit, ValueError) as x:
        return out + [f"{what}: the fit does not re-run on the sealed header ({type(x).__name__}: {x})"]
    finally:
        ops._KERNELS.clear()
        ops._KERNELS.update(saved)
        sys.modules.pop(mod, None)
    if (s, a.ctx) != (rec.get("slots"), rec.get("ctx")):
        out.append(f"{what}: the candidate's fit gives {s} slots of {a.ctx} tokens on this file with bytes_limit {lim}; "
                   f"the record ran {rec.get('slots')} x {rec.get('ctx')}")
    return out


EXPERT_KERNEL, KERNEL_PACKAGE = "moe.expert-ffn", "mashbox_engine_kernels"
GOODPUT_INPUTS = {"bandwidth", "U", "f", "Tc"}


def moe_problems(rec, chip, kernels, what):
    """a passing mixture-of-experts row (never without its experts on the kernel), in the MoE record's fields: the
    record ran the routed experts on the kernel (moe.expert_path), that kernel is moe.expert-ffn of the release package
    on this chip and not a dev build (moe.kernel), every routed-op call site is on the kernel and none on the exact path
    (moe.sites), a profiled decode step makes at least one expert-kernel call per MoE layer (moe.profile), and goodput
    holds 1 and 8 users (an int, never True), each with tok_s, roofline_tok_s and pct > 0, its inputs as numbers, and pct
    the one its tok_s / roofline_tok_s give (ARCH-FIX 2's contract)"""
    m = rec.get("moe") or {}
    if m.get("expert_path") != "kernel":
        return [f"{what}: the record ran the routed experts on {m.get('expert_path')}, not the kernel"]
    out = []
    k = m.get("kernel") if isinstance(m.get("kernel"), dict) else {}
    if (k.get("id"), k.get("package")) != (EXPERT_KERNEL, KERNEL_PACKAGE) or not k.get("version"):
        out.append(f"{what}: the expert kernel is {k.get('id')} {k.get('version')} of {k.get('package')}, not "
                   f"{EXPERT_KERNEL} of {KERNEL_PACKAGE}")
    if k.get("dev_build") is not False:
        out.append(f"{what}: the record ran a dev kernel build (dev_build {k.get('dev_build')}); a row needs the release "
                   "package")
    pkg = [x.get("sha256") for x in kernels if chip in (x.get("chips") or [])]
    if k.get("package_sha256") not in pkg:
        out.append(f"{what}: the expert kernel's package {k.get('package_sha256')} is not the release package's "
                   f"({', '.join(map(str, pkg)) or 'none on ' + chip})")
    s = m.get("sites") if isinstance(m.get("sites"), dict) else {}
    if not (isinstance(s.get("kernel"), int) and s["kernel"] >= 1 and s.get("exact") == 0 and s.get("exact_why") == []):
        out.append(f"{what}: routed-op call sites: {s.get('kernel')} on the kernel, {s.get('exact')} on the exact path "
                   f"({s.get('exact_why')}); a row needs every one on the kernel")
    p = m.get("profile") if isinstance(m.get("profile"), dict) else {}
    calls, layers = p.get("kernel_calls_per_decode_step"), p.get("moe_layers")
    if not (isinstance(calls, int) and isinstance(layers, int) and 1 <= layers <= calls):
        out.append(f"{what}: {calls} expert-kernel calls per decode step for {layers} MoE layers (a row needs one per "
                   "MoE layer at least)")
    gp = rec.get("goodput") if isinstance(rec.get("goodput"), list) else []
    for users in (1, 8):
        hit = [x for x in gp if isinstance(x, dict) and type(x.get("users")) is int and x["users"] == users]
        if len(hit) != 1:
            out.append(f"{what}: the record has no goodput for {users} users")
            continue
        x = hit[0]
        inp = x.get("inputs") if isinstance(x.get("inputs"), dict) else {}

        def num(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if not all(num(x.get(f)) for f in ("tok_s", "roofline_tok_s", "pct")) or not (
                GOODPUT_INPUTS <= set(inp) and all(num(inp[k]) for k in GOODPUT_INPUTS)):
            out.append(f"{what}: goodput at users={users} needs tok_s, roofline_tok_s, pct and its inputs "
                       f"({', '.join(sorted(GOODPUT_INPUTS))}) as numbers")
        elif min(x["tok_s"], x["roofline_tok_s"], x["pct"]) <= 0:
            out.append(f"{what}: goodput at users={users} has tok_s {x['tok_s']}, roofline_tok_s {x['roofline_tok_s']}, "
                       f"pct {x['pct']}; each must be > 0")
        elif abs(x["pct"] - 100 * x["tok_s"] / x["roofline_tok_s"]) > 0.05:
            out.append(f"{what}: goodput at users={users} is {x['pct']}%, its tok_s / roofline_tok_s give "
                       f"{100 * x['tok_s'] / x['roofline_tok_s']:.1f}%")
    return out


def settle_problems(e, others, what, read, root):
    """a passing row whose compare has other divergences (prompt, step, gap): a settle verdict, listed with its sha256 in
    its own seal (every line of which holds), reads PRECISION for the whole run and for each prompt, and its prompts are
    exactly those others, by prompt, chip step and llama.cpp gap (the rule fixed before any row)"""
    out = []
    v, sums = read(e["settle"], f"{what} settle"), read(e["settle_sums"], f"{what} settle seal") if (
        "settle_sums" in e) else out.append(f"{what}: the settle has no seal")
    if sums is not None and _sums(sums).get(os.path.normpath(e["settle"]["path"])) != e["settle"]["sha256"]:
        out.append(f"{what}: {Path(e['settle']['path']).name} is not in {sums.name} with its sha256")
    if sums is not None:
        out += seal_problems(sums, root, f"{what} settle")
    if not isinstance(v, dict):
        return out + ([f"{what}: the settle is not a JSON verdict"] if v is not None else [])
    prompts = v.get("prompts") or []
    verdicts = {v.get("verdict")} | {p.get("verdict") for p in prompts}
    if verdicts != {SETTLED}:
        out.append(f"{what}: the settle's verdict is {', '.join(sorted(map(str, verdicts - {SETTLED})))}, not {SETTLED}")
    covered = sorted(json.dumps([p.get("prompt"), p.get("chip_step"), p.get("chip_ref_gap")]) for p in prompts)
    if len(prompts) != len(others):
        out.append(f"{what}: the settle covers {len(prompts)} of {len(others)} other divergences")
    elif covered != sorted(json.dumps(list(o)) for o in others):
        out.append(f"{what}: the settle's prompts {', '.join(covered)} are not the compare's other divergences "
                   f"{', '.join(sorted(json.dumps(list(o)) for o in others))}")
    return out


def records_state(evidence, require):
    """("run", sidecar) when the sidecar and its records root are on this machine; otherwise ("skip", why), or
    ("fail", why) when the release gate requires the records (MASHBOX_REQUIRE_RECORDS=1)."""
    if not Path(evidence).exists():
        why = "the catalog's records are kept outside this repository"
    else:
        ev = json.loads(Path(evidence).read_text())
        if Path(ev["records_root"]).is_dir():
            return "run", ev
        why = f"the records root {ev['records_root']} is not on this machine"
    return ("fail", f"MASHBOX_REQUIRE_RECORDS=1 and {why}") if require else ("skip", f"maintainers only: {why}")


def test_supported_rows_have_records(tmp_path):
    """every catalog number and every GGUF chip row has its record (evidence_problems). Required at the release gate."""
    state, ev = records_state(EVIDENCE, REQUIRE)
    if state == "skip":
        pytest.skip(ev)
    if state == "fail":
        pytest.fail(ev)
    assert evidence_problems(REG, ev, ROOT, tmp_path / "fit") == []


def test_the_release_gate_turns_the_records_skip_into_a_failure(tmp_path):
    none = tmp_path / "none.json"
    assert records_state(none, False)[0] == "skip" and records_state(none, True)[0] == "fail"
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"records_root": str(tmp_path / "records")}))
    assert records_state(ev, False)[0] == "skip" and records_state(ev, True)[0] == "fail"
    (tmp_path / "records").mkdir()
    assert records_state(ev, True)[0] == "run"


def _commit(d):
    """commit everything in folder d (a new git repo on the first call); returns the commit id"""
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    git = ["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.hooksPath=/dev/null",
           "-c", "commit.gpgsign=false"]
    if not (d / ".git").exists():
        subprocess.run([*git, "init", "-q"], check=True, env=env)
    subprocess.run([*git, "add", "-A"], check=True, env=env)
    subprocess.run([*git, "commit", "-qm", "code"], check=True, env=env)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, env=env).stdout.decode().strip()


def _fake_records(base, edit=None):
    """A records root under base: one GGUF chip row's record, compare, file sha256, plan and seal, the git repo of the
    code it ran (this tree's GGUF code) and the release bundle's manifest; returns (reg, ev). edit(part, value) can
    change a part before it is written: "fixture" (the GGUF file the row is: qwen3-kquant, committed, or a fixture
    tools/gguf_fixtures builds), "code" ({path: bytes}), "rec", "cmp", "file", "plan", "seal" (lines), "manifest", then
    "reg" and "ev"."""
    edit = edit or (lambda part, v: v)
    root = base / "records"
    (root / "row").mkdir(parents=True)
    name = edit("fixture", FIXTURE.stem)
    if name != FIXTURE.stem:
        (base / "fixture").mkdir(exist_ok=True)
    path = FIXTURE if name == FIXTURE.stem else Path(FXT.build(name, str(base / "fixture")))
    from mashbox.gguf import File
    fx = File(str(path))
    files = gguf_code_files(ROOT, fx.arch) + [ARCHS_SRC]
    code = {f: (ROOT / f).read_bytes() for f in files}
    code["models.json"] = json.dumps({"engine_kernels": [dict(BUNDLE)]}).encode()
    for f, data in edit("code", code).items():
        (root / "code" / f).parent.mkdir(parents=True, exist_ok=True)
        (root / "code" / f).write_bytes(data)
    commit, kq = _commit(root / "code"), "f" * 40
    btypes = sorted(fx.census()[0])
    quant = [n for n in btypes if n not in catalog.UNQUANTIZED]
    ids = {t.type_name: t.type for t in fx.tensors.values()}
    planes = {f"{t.type_name}:{t.shape[-2]}:{t.shape[-1]}": 4 * t.nbytes for t in fx.tensors.values()
              if t.type_name in quant and len(t.shape) >= 2}       # made up: 4 x the file's blocks
    row = dict(ROW, arch=fx.arch, bytes=path.stat().st_size, types=btypes, chips={"v6e": dict(
        ROW["chips"]["v6e"], kernel_path="bundle", measured={"decode_tok_s_1_user": 51.2, "decode_tok_s_8_users": 500.0})})
    parts = {"rec": {"file": row["file"], "code_sha": f"{commit[:7]}+kq-{kq[:7]}", "device": "TPU v6 lite",
                     "n_devices": 1, "ctx": 2048, "slots": 8, "weights": {"file": row["file"], "reference_bytes": 0},
                     "invocation": {"argv": ["run", f"/models/{row['file']}"], "env": {}},
                     "fit": {"bytes_limit": V6E, "kernel_types": quant}, "memory": {"bytes_limit": V6E},
                     "versions": dict(RUN_VERSIONS), "impl": {n: "c" * 64 for n in quant},
                     "one": [{"new": 64, "decode_s": 1.0}, {"new": 64, "decode_s": 1.25}, {"new": 64, "decode_s": 1.28}],
                     "eight": [{"new": 512, "decode_s": 1.0}, {"new": 512, "decode_s": 1.048}]},
             "cmp": {"summary": {"prompts": 16, "full_match": 2, "near_tie_divergences": 14, "other_divergences": 0},
                     "rows": [{"i": i, "n": 64, "match": 64 if i < 2 else 10, "first_diff": None if i < 2 else 10,
                               "ref_gap": None if i < 2 else 0.25, "near_tie": i >= 2} for i in range(16)]},
             "card": {"id": row["repo"], "sha": row["revision"], "cardData": {
                 "license": row["license"], "license_link": row["license_link"], "base_model": row["base"]}},
             "log": "run log\n",
             "file": f"{row['sha256']}  {row['file']}\n{row['bytes']}\n",
             "plan": f"{row['repo']} {row['revision']} {row['file']} {row['sha256']}\n",
             "attestation": {"unit": "mul_mat.kquant-bf16", "unit_version": "0.1.0", "kq_commit": kq,
                             "impl": {n: "c" * 64 for n in quant}},
             "manifest": {"commit": kq, "impl": {n: "c" * 64 for n in quant}, "types": quant,
                          "type_ids": {n: ids[n] for n in quant}, "plane_bytes": planes},
             "header": _header(path)}
    names = {"rec": "rec.json", "cmp": "cmp.json", "file": "file-row.sha256", "plan": "plan-row.txt",
             "attestation": "attestation.json", "manifest": "manifest.json", "card": "card-api.json", "log": "run.log",
             "header": "header-row.gguf"}
    for k, n in names.items():
        if k == "manifest":         # the manifest names the attestation written just before it, by sha256
            parts[k] = dict(parts[k], attestation={"path": names["attestation"],
                                                   "sha256": _sha(root / "row" / names["attestation"])})
        v = edit(k, parts[k])
        if isinstance(v, bytes):
            (root / "row" / n).write_bytes(v)
        else:
            (root / "row" / n).write_text(v if isinstance(v, str) else json.dumps(v))
    forced = edit("forced", None)
    if forced is not None:
        names["forced"] = "rec-forced.json"
        (root / "row" / names["forced"]).write_text(json.dumps(forced))
    seal = edit("seal", [f"{_sha(root / 'row' / names[k])}  {names[k]}" for k in ("rec", "cmp", "file", "plan", "card",
                                                                                   "log", "header", "forced")
                         if k in names])
    seal = [x.replace("@ROW@", str(root / "row")) for x in seal]
    (root / "row" / "SHA256SUMS").write_text("\n".join(seal) + "\n")
    settle = edit("settle", None)
    if settle is not None:
        (root / "row" / "settle").mkdir()
        (root / "row" / "settle" / "verdict.json").write_text(json.dumps(settle))
        lines = edit("seal_settle", [f"{_sha(root / 'row' / 'settle' / 'verdict.json')}  verdict.json"])
        (root / "row" / "settle" / "SHA256SUMS-settle").write_text("\n".join(lines) + "\n")

    def src(n):
        return {"path": str(root / "row" / n), "sha256": _sha(root / "row" / n)}
    ev = {"records_root": str(root), "code_repo": str(root / "code"), "models": {},
          "engine_kernels": {BUNDLE["file"]: {"manifest": src("manifest.json"), "attestation": src("attestation.json")}},
          "gguf": {row["id"]: {"v6e": {
              "record": src("rec.json"), "compare": src("cmp.json"), "sha256sums": src("SHA256SUMS"),
              "file": src("file-row.sha256"), "plan": src("plan-row.txt"), "card": src("card-api.json"),
              "header": src("header-row.gguf"),
              "numbers": {"decode_tok_s_1_user": dict(src("rec.json"), field="median(one[].new/decode_s)", round=2),
                          "decode_tok_s_8_users": dict(src("rec.json"), field="sum(eight[].new)/sum(eight[].decode_s)",
                                                       round=1)},
              "code": {"commit": commit}, "kernels": {"commit": kq}}}}}
    if forced is not None:
        ev["gguf"][row["id"]]["v6e"]["forced"] = [src("rec-forced.json")]
    if settle is not None:
        ev["gguf"][row["id"]]["v6e"]["settle"] = src("settle/verdict.json")
        ev["gguf"][row["id"]]["v6e"]["settle_sums"] = src("settle/SHA256SUMS-settle")
    reg = copy.deepcopy(REG)
    reg["gguf"], reg["engine_kernels"], reg["models"] = [row], [dict(BUNDLE)], []
    return edit("reg", reg), edit("ev", ev)


# the dists a chip run records (versions{}): this tree's pins where it pins one, the release bundle's JAX and libtpu,
# and made-up values for the rest
RUN_VERSIONS = {"python": "3.12.16", "jax": "0.11.2", "jaxlib": "0.11.2", "libtpu": "0.0.48.1", "gguf": "0.19.0",
                "tokenizers": "0.22.1", "jinja2": "3.1.6", "numpy": "2.3.4", "mashbox_engine_kernels": REG["version"]}


def _set(part, fn):
    """an edit for _fake_records that changes one part with fn(value) and leaves the others"""
    def edit(p, v):
        if p == part:
            v = copy.deepcopy(v)
            new = fn(v)
            return v if new is None else new
        return v
    return edit


def _both(*edits):
    def edit(p, v):
        for e in edits:
            v = e(p, v)
        return v
    return edit


def _in_place(part, fn):
    """an edit that changes one part in place with fn, whatever fn returns"""
    return _set(part, lambda v: (fn(v), None)[1])


def _forced(argv=("run", "/models/" + ROW["file"], "--slots", "4"), declared=True, cited=False):
    """an edit: the run folder also holds a sealed forced-slot record of the same file; declared lists it under the
    evidence's forced, cited reads the row's 1-user number from it"""
    def ev(e):
        x = e["gguf"][ROW["id"]]["v6e"]
        if not declared:
            del x["forced"]
        elif cited:
            x["numbers"]["decode_tok_s_1_user"].update(x["forced"][0])
    return _both(_set("forced", lambda v: {"file": ROW["file"], "slots": 4, "ctx": 2048, "invocation": {
        "argv": list(argv), "env": {}}}), _in_place("ev", ev))


def _plane_key(name):
    """the release manifest's plane_bytes key of one fixture tensor"""
    from mashbox.gguf import File
    t = File(str(FIXTURE)).tensors[name]
    return f"{t.type_name}:{t.shape[-2]}:{t.shape[-1]}"


def _code(path, fn):
    return _set("code", lambda c: c.__setitem__(path, fn(c.get(path, b""))))


def _others(n):
    """an edit: the last n of the compare's 16 rows are other divergences (gap 1.5 at step 20 + i); the summary agrees"""
    def fn(c):
        for x in c["rows"][16 - n:]:
            x.update(match=20 + x["i"], first_diff=20 + x["i"], ref_gap=1.5, near_tie=False)
        full = sum(x["first_diff"] is None for x in c["rows"])
        c["summary"].update(full_match=full, near_tie_divergences=16 - n - full, other_divergences=n)
    return _set("cmp", fn)


def _settled(others, prompts=None, verdict="PRECISION"):
    """an edit: the compare has `others` other divergences and a sealed settle verdict covers `prompts` of them, each by
    its prompt, chip step and llama.cpp gap"""
    n = others if prompts is None else prompts
    return _both(_others(others), _set("settle", lambda v: {"verdict": verdict, "prompts": [
        {"prompt": i, "chip_step": 20 + i, "chip_ref_gap": 1.5, "verdict": verdict}
        for i in range(16 - others, 16 - others + n)]}))


# the MoE fields of a chip record, MOE's five names and nothing else (the expert count is the file's metadata, read
# from the sealed header): the routed experts ran on the expert kernel of the release package at every routed-op call
# site, one kernel call per MoE layer per profiled decode step, and goodput as % of the roofline for 1 and 8 users
# (values made up for the checks)
MOE_RECORD = {"expert_path": "kernel",
              "kernel": {"id": "moe.expert-ffn", "version": "0.1.4", "package": "mashbox_engine_kernels",
                         "package_sha256": BUNDLE["sha256"], "dev_build": False},
              "sites": {"kernel": 2, "exact": 0, "exact_why": []},
              "profile": {"kernel_calls_per_decode_step": 4, "moe_layers": 4}}
GOODPUT = [{"users": u, "tok_s": t, "roofline_tok_s": rt, "pct": p, "inputs": {"bandwidth": 1.64e12, "U": u, "f": 0.1,
                                                                                "Tc": 0.0}}
           for u, t, rt, p in ((1, 50.0, 200.0, 25.0), (8, 320.0, 800.0, 40.0))]


def _exact(package=None, drop=False):
    """an edit: the row ran on the exact reference path with zero flags and no kernel package installed (a version with
    no engine_kernels row): kernel_path exact, no kernel types or impls, its blocks on the exact path, a code sha with no
    kq part, and versions naming the kernel package as not installed (None; `package` puts a version there, `drop` leaves
    the name out)"""
    def rec(r):
        r.update(code_sha=r["code_sha"].split("+kq-")[0], impl={})
        r["fit"]["kernel_types"] = []
        r["weights"]["reference_bytes"] = 4096
        if drop:
            r["versions"].pop(KERNEL_PACKAGE)
        else:
            r["versions"][KERNEL_PACKAGE] = package

    def reg(r):
        r["engine_kernels"] = []
        r["gguf"][0]["chips"]["v6e"]["kernel_path"] = "exact"

    def ev(e):
        e["engine_kernels"] = {}
        e["gguf"][ROW["id"]]["v6e"].pop("kernels")
    return _both(_in_place("rec", rec), _in_place("reg", reg), _in_place("ev", ev),
                 _code("models.json", lambda b: json.dumps({"engine_kernels": []}).encode()))


def _moe():
    """an edit: the row is a mixture of experts (qwen35moe-pyquant, 8 experts in the file's metadata, so in its
    sealed header) whose routed experts ran on the kernel, with its goodput as % of the roofline for 1 and 8 users read
    from the record"""
    def reg(r):
        g = r["gguf"][0]
        g.update(moe=True, params_b={"total": 0.5, "active": 0.1})
        g["chips"]["v6e"].update(routed_experts="kernel")
        g["chips"]["v6e"]["measured"].update(goodput_roofline_pct_1_user=25.0, goodput_roofline_pct_8_users=40.0)

    def ev(e):
        n = e["gguf"][ROW["id"]]["v6e"]["numbers"]
        for users, name in ((1, "1_user"), (8, "8_users")):
            n[f"goodput_roofline_pct_{name}"] = dict(n["decode_tok_s_1_user"], field=f"goodput[users={users}].pct", round=1)
    return _both(_set("fixture", lambda v: "qwen35moe-pyquant"), _set("reg", reg), _set("ev", ev),
                 _set("rec", lambda x: x.update(moe=copy.deepcopy(MOE_RECORD), goodput=copy.deepcopy(GOODPUT))))


def _moe_rec(fn):
    """an edit: the mixture-of-experts row of _moe(), with fn(moe, goodput) changing its record's moe fields or goodput"""
    def rec(r):
        fn(r["moe"], r["goodput"])      # changes the record in place; returns None so _set keeps the record
    return _both(_moe(), _set("rec", rec))


MODS = arch_module_files()
# each bad form changes one thing in a backed row; the guard must name it (the text is a piece of its message)
BAD_FORMS = {
    "a number the record does not hold": (_set("reg", lambda r: r["gguf"][0]["chips"]["v6e"]["measured"].__setitem__(
        "decode_tok_s_1_user", 999.9)), "the record's median(one[].new/decode_s) is 51.2"),
    "a number with no record": (_set("reg", lambda r: r["gguf"][0]["chips"]["v6e"]["measured"].__setitem__(
        "decode_tok_s_32_users", 1.0)), "decode_tok_s_32_users has no record"),
    "a missing record": (_set("ev", lambda e: e["gguf"][ROW["id"]]["v6e"]["record"].__setitem__(
        "path", e["records_root"] + "/row/nowhere.json")), "does not exist"),
    "a changed record": (_set("ev", lambda e: e["gguf"][ROW["id"]]["v6e"]["record"].__setitem__("sha256", "0" * 64)),
                         "sha256 differs"),
    "the record ran other code": (_set("rec", lambda r: r.__setitem__("code_sha", "0000000+kq-fffffff")),
                                  "the record ran 0000000+kq-fffffff"),
    "the record ran another kq commit": (_set("rec", lambda r: r.__setitem__(
        "code_sha", r["code_sha"].split("+")[0] + "+kq-1111111")), "+kq-1111111, the evidence names"),
    "the record ran another file": (_set("rec", lambda r: r["weights"].__setitem__("file", "Other-Q4_K_M.gguf")),
                                    "the record ran Other-Q4_K_M.gguf"),
    "the record ran tensors on the exact path": (_set("rec", lambda r: r["weights"].__setitem__(
        "reference_bytes", 4096)), "4096 bytes on the exact path"),
    "a seal that lists nothing": (_set("seal", lambda s: ["unrelated"]), "rec.json is not in SHA256SUMS"),
    "a seal with another sha256 for the compare": (_set("seal", lambda s: s.__setitem__(1, "0" * 64 + "  cmp.json")),
                                                   "cmp.json is not in SHA256SUMS"),
    "file-*.sha256 names another file": (_set("file", lambda t: t.replace(ROW["file"], "Other-Q4_K_M.gguf")),
                                         "file-row.sha256 names another file"),
    "file-*.sha256 has another sha256": (_set("file", lambda t: "1" * 64 + t[64:]), "file-row.sha256 names another"),
    "file-*.sha256 has another size": (_set("file", lambda t: t.split("\n")[0] + "\n1\n"),
                                       "file-row.sha256 names another"),
    "the plan names another revision": (_set("plan", lambda t: t.replace(ROW["revision"], "1" * 40)),
                                        "plan-row.txt is not"),
    "the arch's module differs": (_code(MODS["qwen3"], lambda b: b + b"# changed\n"), "dense.py differs"),
    "a module the arch's module imports differs": (_code("src/mashbox/gguf/archs/qwen35.py", lambda b: b + b"#\n"),
                                                   "qwen35.py differs"),
    "a module two imports down differs": (_code("src/mashbox/models/qwen3_8_27b/model.py", lambda b: b + b"#\n"),
                                          "model.py differs"),
    "the reader differs": (_code("src/mashbox/gguf/reader.py", lambda b: b + b"#\n"), "reader.py differs"),
    "ARCHS ran the arch with another module": (_code(ARCHS_SRC, lambda b: b.replace(
        b'"qwen3": "mashbox.gguf.archs.dense"', b'"qwen3": "mashbox.gguf.archs.qwen35"')), "the record's code ran it"),
    "an ops file the record ran is gone": (_code("src/mashbox/ops/gone.py", lambda b: b"x = 1\n"),
                                           "gone.py ran in the record"),
    "the code commit is not in the code repo": (_both(
        _set("rec", lambda r: r.__setitem__("code_sha", "eeeeeee+kq-fffffff")),
        _set("ev", lambda e: e["gguf"][ROW["id"]]["v6e"]["code"].__setitem__("commit", "e" * 40))),
        "(eeeeeeeeeeee) is not in"),
    "the code repo is outside the records root": (_set("ev", lambda e: e.__setitem__("code_repo", str(ROOT))),
                                                  "is not under"),
    "the bundle is built from another kq commit": (_set("manifest", lambda m: m.__setitem__("commit", "1" * 40)),
                                                   "the release bundle is built from 1111"),
    "the bundle has no kernel for a type of the file": (_set("manifest", lambda m: m.__setitem__("impl", {"Q4_K": "c" * 64})),
                                                        "has no Q6_K kernel"),
    "a changed bundle manifest": (_set("ev", lambda e: e["engine_kernels"][BUNDLE["file"]]["manifest"].__setitem__(
        "sha256", "0" * 64)), "manifest: "),
    "no bundle manifest": (_set("ev", lambda e: e.__setitem__("engine_kernels", {})), "no manifest of this release"),
    "no evidence entry": (_set("ev", lambda e: e.__setitem__("gguf", {})), "no evidence entry"),
    # problem 3: the chip, ctx and slots the views show are the record's; "passing" is read from the compare or a settle
    "the row's accelerator is not the record's": (_set("reg", lambda r: r["gguf"][0]["chips"]["v6e"].__setitem__(
        "accelerator", "v6e-8")), "accelerator v6e-8, the record ran on TPU v6 lite x1"),
    "the record ran on another chip type": (_set("rec", lambda r: r.__setitem__("device", "TPU v5 lite")),
                                            "accelerator v6e-1, the record ran on TPU v5 lite x1"),
    "the row's ctx is not the record's": (_set("reg", lambda r: r["gguf"][0]["chips"]["v6e"].__setitem__("ctx", 32768)),
                                          "ctx 32768, the record ran 2048"),
    "the row's slots are not the record's": (_set("reg", lambda r: r["gguf"][0]["chips"]["v6e"].__setitem__(
        "slots", 64)), "slots 64, the record ran 8"),
    "passing with other divergences and no settle": (_others(16),
                                                     "passing, and the compare has 16 other divergences and no settle"),
    "a settle whose verdict is not PRECISION": (_settled(2, verdict="GRAPH"), "the settle's verdict is GRAPH"),
    "a settle of fewer prompts than the compare's others": (_settled(2, prompts=1), "the settle covers 1 of 2"),
    "a settle its seal does not list": (_both(_settled(2), _set("seal_settle", lambda s: ["unrelated"])),
                                        "verdict.json is not in SHA256SUMS-settle"),
    # problem 4: a code repo outside the root, written with .. so that it reads as under the root
    "the code repo is outside the records root through ..": (_set("ev", lambda e: e.__setitem__(
        "code_repo", e["records_root"] + "/" + os.path.relpath(ROOT, e["records_root"]))), "is not under"),
    # problem 5: gguf/__init__.py (open_model, fit_slots) is in the code set; the file sha256 needs its size line
    "gguf/__init__.py differs": (_code("src/mashbox/gguf/__init__.py", lambda b: b + b"#\n"), "__init__.py differs"),
    "file-*.sha256 has no size line": (_set("file", lambda t: t.split("\n")[0] + "\n"), "has no size line"),
    # The MoE rule: the expert count is the file's own metadata, read from the sealed header (never a record
    # field); the routed experts ran on the kernel
    "a file with experts on a row that says dense": (_set("fixture", lambda v: "qwen35moe-pyquant"),
                                                      "the sealed header's metadata has 8 experts, the row says moe False"),
    "a row that says mixture of experts on a file with no experts": (
        _both(_moe(), _set("fixture", lambda v: FIXTURE.stem)),
        "the sealed header's metadata has 0 experts, the row says moe True"),
    "a mixture of experts whose record has no moe fields": (_both(_moe(), _set("rec", lambda r: r.__delitem__("moe"))),
                                                            "the record ran the routed experts on None, not the kernel"),
    "a mixture of experts on the exact routed path": (_moe_rec(lambda m, g: m.update(expert_path="exact")),
                                                       "the record ran the routed experts on exact, not the kernel"),
    "a routed-op call site on the exact path": (_moe_rec(lambda m, g: m["sites"].update(exact=1, exact_why=["shape"])),
                                                "routed-op call sites: 2 on the kernel, 1 on the exact path"),
    "no routed-op call site on the kernel": (_moe_rec(lambda m, g: m["sites"].update(kernel=0)),
                                             "routed-op call sites: 0 on the kernel, 0 on the exact path"),
    "another expert kernel": (_moe_rec(lambda m, g: m["kernel"].update(id="moe.other")), "the expert kernel is moe.other"),
    "a dev kernel build": (_moe_rec(lambda m, g: m["kernel"].update(dev_build=True)), "the record ran a dev kernel build"),
    "an expert kernel from another package build": (_moe_rec(lambda m, g: m["kernel"].update(package_sha256="e" * 64)),
                                                    "is not the release package's"),
    "fewer expert-kernel calls than MoE layers": (_moe_rec(lambda m, g: m["profile"].update(
        kernel_calls_per_decode_step=3)), "3 expert-kernel calls per decode step for 4 MoE layers"),
    "no goodput for 8 users": (_moe_rec(lambda m, g: g.pop()), "the record has no goodput for 8 users"),
    "a goodput % its own numbers do not give": (_moe_rec(lambda m, g: g[0].update(roofline_tok_s=100.0)),
                                               "goodput at users=1 is 25.0%, its tok_s / roofline_tok_s give 50.0%"),
    "a goodput of no tokens": (_moe_rec(lambda m, g: g[0].update(tok_s=0.0, pct=0.0)),
                               "goodput at users=1 has tok_s 0.0, roofline_tok_s 200.0, pct 0.0; each must be > 0"),
    "a goodput with no roofline": (_moe_rec(lambda m, g: g[1].update(roofline_tok_s=0.0)),
                                   "goodput at users=8 has tok_s 320.0, roofline_tok_s 0.0, pct 40.0; each must be > 0"),
    "a goodput input that is not a number": (_moe_rec(lambda m, g: g[1]["inputs"].update(f="0.1")),
                                             "goodput at users=8 needs tok_s, roofline_tok_s, pct and its inputs"),
    "a goodput entry whose users is true, not 1": (_moe_rec(lambda m, g: g[0].update(users=True)),
                                                   "the record has no goodput for 1 users"),
    "a mixture of experts whose goodput the record does not hold": (_both(_moe(), _set("reg", lambda r: r["gguf"][0][
        "chips"]["v6e"]["measured"].__setitem__("goodput_roofline_pct_1_user", 60.0))),
        "the record's goodput[users=1].pct is 25.0"),
    # review R12, pinned versions: the record states the dists it ran; this tree's pins and the bundle's JAX and libtpu hold
    "a record with no versions": (_set("rec", lambda r: r.__delitem__("versions")), "the record has no versions"),
    "a record with no numpy version": (_in_place("rec", lambda r: r["versions"].pop("numpy")),
                                       "the record's versions have no numpy"),
    "a record that ran another jinja2 than this tree pins": (_in_place("rec", lambda r: r["versions"].update(
        jinja2="3.1.5")), "the record ran jinja2 3.1.5, this tree pins 3.1.6"),
    "a record that ran another gguf than this tree pins": (_in_place("rec", lambda r: r["versions"].update(
        gguf="0.18.0")), "the record ran gguf 0.18.0, this tree pins 0.19.0"),
    "a record that ran another jax than this tree pins": (_both(
        _in_place("rec", lambda r: r["versions"].update(jax="0.11.1")),
        _in_place("reg", lambda r: r["engine_kernels"][0].update(jax="0.11.1"))),
        "the record ran jax 0.11.1, this tree pins 0.11.2"),
    "a record whose jax is not the release bundle's": (_in_place("reg", lambda r: r["engine_kernels"][0].update(
        jax="0.11.3")), "the record ran jax 0.11.2, the release bundle on v6e needs 0.11.3"),
    "a record whose libtpu is not the release bundle's": (_in_place("rec", lambda r: r["versions"].update(
        libtpu="0.0.47.1")), "the record ran libtpu 0.0.47.1, the release bundle on v6e needs 0.0.48.*"),
    # review R12-F01, kernel attestation: each type the record ran is on the attested impl of the release bundle
    # an exact-path run names the kernel package as not installed; a run on the bundle names its version
    "an exact-path record that names a kernel package version": (_exact(package="0.1.3"),
                                                                  "kernel_path exact, and the record ran "
                                                                  "mashbox_engine_kernels 0.1.3"),
    "a bundle record that ran another kernel package version": (_in_place("rec", lambda r: r["versions"].update(
        mashbox_engine_kernels="0.0.9")), f"the record ran mashbox_engine_kernels 0.0.9, the release bundle on v6e is "
                                          f"{REG['version']}"),
    "a bundle record with no kernel package": (_in_place("rec", lambda r: r["versions"].update(
        mashbox_engine_kernels=None)), "the record's versions have no mashbox_engine_kernels"),
    "a record with no impl": (_set("rec", lambda r: r.__delitem__("impl")), "the record has no impl"),
    "a record that ran another kernel impl": (_in_place("rec", lambda r: r["impl"].update({sorted(r["impl"])[0]: "d" * 64})),
                                              "on impl " + "d" * 64),
    "no attestation in the evidence": (_in_place("ev", lambda e: e["engine_kernels"][BUNDLE["file"]].pop("attestation")),
                                       "no attestation in the evidence"),
    "a changed attestation": (_in_place("ev", lambda e: e["engine_kernels"][BUNDLE["file"]]["attestation"].update(
        sha256="0" * 64)), "attestation: "),
    "a manifest that names another attestation": (_in_place("manifest", lambda m: m["attestation"].update(
        sha256="1" * 64)), "the manifest names attestation " + "1" * 64),
    "an attestation of another kq commit": (_in_place("attestation", lambda a: a.update(kq_commit="2" * 40)),
                                            "the attestation hashed kq " + "2" * 40),
    "a manifest impl that is not the attested one": (_in_place("attestation", lambda a: a["impl"].update(
        {sorted(a["impl"])[0]: "e" * 64})), "the attested impl is " + "e" * 64),
    # R12-F04: the row's record is the zero-flag run; its slots and context are the candidate's fit on the sealed header
    "a record that ran with --slots": (_in_place("rec", lambda r: r["invocation"]["argv"].extend(["--slots", "8"])),
                                       "the record ran `mashbox run /models/Tiny-Qwen3-Q4_K_M.gguf --slots 8`"),
    "a record that ran with a MASHBOX_ variable": (_in_place("rec", lambda r: r["invocation"]["env"].update(
        MASHBOX_GGUF_KERNELS="kernel")), "with MASHBOX_GGUF_KERNELS=kernel"),
    "a record with no invocation": (_in_place("rec", lambda r: r.pop("invocation")), "the record has no invocation"),
    "review R12's scratch record: slots 1, ctx 1, --slots 1": (_in_place("rec", lambda r: (r.update(slots=1, ctx=1), r[
        "invocation"]["argv"].extend(["--slots", "1"]))), "--slots 1`; the row's record is the zero-flag run"),
    "a fit the candidate does not give": (_in_place("rec", lambda r: (r["fit"].update(bytes_limit=60_000_000), r[
        "memory"].update(bytes_limit=60_000_000))), "the candidate's fit gives 1 slots of 2048 tokens"),
    "a limit the candidate refuses the file at": (_in_place("rec", lambda r: (r["fit"].update(bytes_limit=30_000_000), r[
        "memory"].update(bytes_limit=30_000_000))), "the candidate refuses this file at bytes_limit 30000000"),
    "a fit limit that is not the run's": (_in_place("rec", lambda r: r["fit"].update(bytes_limit=16_909_334_528)),
                                          "is not the run's memory.bytes_limit 33546041344"),
    "a record with no fit": (_in_place("rec", lambda r: r.pop("fit")), "the record has no fit.bytes_limit"),
    "kernel types that are not the release bundle's": (_in_place("rec", lambda r: r["fit"].update(kernel_types=[])),
                                                        "the record's kernel types [] are not the release bundle's"),
    "a header of another architecture": (_set("header", lambda h: h.replace(
        b"\x05\x00\x00\x00\x00\x00\x00\x00qwen3", b"\x05\x00\x00\x00\x00\x00\x00\x00qwen2", 1)),
        "the sealed header is a qwen2 file, the row says qwen3"),
    "a header whose block types are not the row's": (_in_place("reg", lambda r: r["gguf"][0]["types"].remove("IQ4_NL")),
                                                    "the sealed header's block types are"),
    "a header longer than the file": (_in_place("reg", lambda r: r["gguf"][0].update(bytes=100)),
                                      "the sealed header is 7648 bytes, longer than the file's 100"),
    "a header not in the seal": (_set("seal", lambda s: [x for x in s if "header-row" not in x]),
                                 "header-row.gguf is not in SHA256SUMS"),
    "no header in the evidence": (_in_place("ev", lambda e: e["gguf"][ROW["id"]]["v6e"].pop("header")),
                                  "no header in the evidence"),
    "a manifest without the planes of a multiplied weight": (_in_place("manifest", lambda m: m["plane_bytes"].pop(
        _plane_key("blk.0.attn_q.weight"))), "the release bundle has no planes for"),
    "a forced-slot record the evidence does not name": (_forced(declared=False), "rec-forced.json is a record of this "
                                                        "run that the evidence names neither as the row's record nor as forced"),
    "a forced-slot record that ran with zero flags": (_forced(argv=("run", "/models/" + ROW["file"])),
                                                      "a forced-slot record ran with zero flags"),
    "a number read from a forced-slot record": (_forced(cited=True), "measured.decode_tok_s_1_user is read from a "
                                                                     "forced-slot record"),
    # R12-F05: full / near / other are recomputed from the compare's rows; a settle covers exactly the others
    "a summary that hides an other": (_both(_others(1), _set("cmp", lambda c: c["summary"].update(
        near_tie_divergences=14, other_divergences=0))), "the compare's rows give full 2 / near 13 / other 1"),
    "a near tie at a gap of 0.5": (_set("cmp", lambda c: c["rows"][5].update(ref_gap=0.5)),
                                   "the compare's rows give full 2 / near 13 / other 1"),
    "a compare with fewer rows than prompts": (_set("cmp", lambda c: c.__setitem__("rows", c["rows"][:15])),
                                               "the compare has 15 rows for 16 prompts"),
    "a settle of other steps than the compare's others": (_both(_settled(2), _set("settle", lambda v: v["prompts"][0].update(
        chip_step=3))), "are not the compare's other divergences"),
    # R12-F07: every line of the seal is checked against the bytes under the records root
    "a seal line whose file has another sha256": (_set("seal", lambda s: [("0" * 64 + "  run.log") if x.endswith(
        "  run.log") else x for x in s]), "run.log sha256 differs from its seal line"),
    "a seal line for a file that is not there": (_set("seal", lambda s: s + ["1" * 64 + "  gone.log"]),
                                                 "gone.log is not there"),
    "a seal line through ..": (_set("seal", lambda s: s + [s[0].replace("  rec.json", "  ../row/rec.json")]),
                               "rec.json is not under the seal's folder (..)"),
    "an absolute seal line": (_set("seal", lambda s: s + [s[0].replace("  rec.json", "  @ROW@/rec.json")]),
                              "rec.json is not under the seal's folder (absolute)"),
    "a seal that lists a file twice": (_set("seal", lambda s: s + [s[0]]), "rec.json is listed twice"),
    "a settle seal line for a file that is not there": (_both(_settled(2), _set("seal_settle", lambda s: s + [
        "1" * 64 + "  gone.json"])), "gone.json is not there"),
    # R12-F08: the row's repo, revision, licence, licence link and base are the pinned card's
    "the row's licence is not the card's": (_set("reg", lambda r: r["gguf"][0].__setitem__("license", "mit")),
                                            "license 'mit', the pinned card says 'apache-2.0'"),
    "the row's licence link is not the card's": (_set("reg", lambda r: r["gguf"][0].__setitem__(
        "license_link", ROW["license_link"] + ".txt")), "LICENSE.txt', the pinned card says"),
    "the row's base is not the card's": (_set("reg", lambda r: r["gguf"][0].__setitem__("base", "someone/Other")),
                                         "base 'someone/Other', the pinned card says 'someone/Tiny-Qwen3'"),
    "the card is another repo's": (_set("card", lambda c: c.__setitem__("id", "someone/Other-GGUF")),
                                   "repo 'someone/Tiny-Qwen3-GGUF', the pinned card says 'someone/Other-GGUF'"),
    "the card is at another revision": (_set("card", lambda c: c.__setitem__("sha", "1" * 40)),
                                        "the pinned card says '1111111111111111111111111111111111111111'"),
    "a card that states no licence": (_set("card", lambda c: c["cardData"].__delitem__("license")),
                                      "license 'apache-2.0', the pinned card says None"),
    "no card in the evidence": (_set("ev", lambda e: e["gguf"][ROW["id"]]["v6e"].__delitem__("card")), "no card in the evidence"),
    "a card the seal does not list": (_set("seal", lambda s: [x for x in s if not x.endswith("card-api.json")]),
                                      "card-api.json is not in SHA256SUMS"),
    # R12-F03: the engine_kernels rows the loader reads are part of the code set
    "the engine_kernels row is not the one the record's code had": (_code("models.json", lambda b: json.dumps(
        {"engine_kernels": []}).encode()), "the engine_kernels rows for v6e differ from the code the record ran"),
}


# code set v1 as reviewed: the entry path by role
REVIEWED_CODE_SET_V1 = (
    "src/mashbox/__main__.py", "src/mashbox/__init__.py", "src/mashbox/cli.py",
    "src/mashbox/gguf/resolve.py", "src/mashbox/gguf/reader.py", "src/mashbox/gguf/tokenizer.py", "src/mashbox/hub.py",
    "src/mashbox/gguf/__init__.py", "src/mashbox/gguf/archs/__init__.py",
    "src/mashbox/ops/__init__.py", "src/mashbox/ops/reference.py",
    "src/mashbox/runtime/engine.py", "src/mashbox/runtime/programs.py", "src/mashbox/runtime/compile_key.py",
    "src/mashbox/runtime/tenancy.py", "src/mashbox/runtime/sampler.py", "src/mashbox/runtime/chat.py",
    "src/mashbox/runtime/server.py")


@pytest.mark.parametrize("member", REVIEWED_CODE_SET_V1)
def test_record_guard_refuses_a_row_when_any_code_set_member_differs(tmp_path, member):
    """R12-F03: one mutation per member of the code set (the `mashbox run <gguf>` entry path by role): the guard names it"""
    reg, ev = _fake_records(tmp_path, _code(member, lambda b: b + b"#\n"))
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(f"{member} differs" in x for x in found), found


def test_the_code_set_is_the_reviewed_v1_and_names_only_files_of_this_tree():
    """the guard's code set is the reviewed v1, every member is a file here (one the tree lacks would drop out of the
    import walk without a word, so the guard also names it), and each arch's set holds all of them"""
    assert CODE_SET_VERSION == 1 and GGUF_CODE == REVIEWED_CODE_SET_V1
    assert [f for f in GGUF_CODE if not (ROOT / f).is_file()] == []
    assert all(set(gguf_code_files(ROOT, a)) >= set(GGUF_CODE) for a in arch_module_files())


def test_record_guard_refuses_a_record_linked_from_outside_the_records_root(tmp_path):
    """R12-F07 with the root rule: a seal line that names a link inside the row folder to bytes outside the records root"""
    reg, ev = _fake_records(tmp_path)
    e = ev["gguf"][ROW["id"]]["v6e"]
    out = tmp_path / "elsewhere" / "rec.json"
    out.parent.mkdir()
    out.write_bytes(Path(e["record"]["path"]).read_bytes())
    link = Path(e["record"]["path"]).with_name("rec-link.json")
    link.symlink_to(out)
    sums = Path(e["sha256sums"]["path"])
    sums.write_text(sums.read_text().replace("  rec.json", "  rec-link.json"))
    e["sha256sums"]["sha256"], e["record"]["path"] = _sha(sums), str(link)
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert found and all("is not under" in x for x in found), found


def test_the_records_command_fails_without_the_records(tmp_path):
    """R12-F02: tools/catalog_records_check.sh, which the private merge and release gates run, requires the records, so a
    missing sidecar fails there where plain pytest skips"""
    env = dict(os.environ, PYTHON=sys.executable, MASHBOX_CATALOG_EVIDENCE=str(tmp_path / "none.json"))
    env.pop("MASHBOX_REQUIRE_RECORDS", None)
    cmd = ROOT / "tools" / "catalog_records_check.sh"
    assert cmd.is_file()
    r = subprocess.run(["sh", str(cmd)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "MASHBOX_REQUIRE_RECORDS=1 and the catalog's records are kept outside" in r.stdout + r.stderr, r.stdout[-2000:]


def test_record_guard_holds_a_backed_row(tmp_path):
    reg, ev = _fake_records(tmp_path)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_holds_an_exact_path_row_with_no_kernel_package(tmp_path):
    """a version that installs no kernel package: the zero-flag run is on the exact path and its record names the
    package as not installed (None), which is the honest value there, never a typed one"""
    reg, ev = _fake_records(tmp_path, _exact())
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_holds_an_exact_path_row_whose_record_leaves_the_kernel_package_out(tmp_path):
    """not installed may also read as the name left out of versions{} (absent or None, never a version)"""
    reg, ev = _fake_records(tmp_path, _exact(drop=True))
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_holds_a_mixture_of_experts_row_whose_routed_experts_ran_on_the_kernel(tmp_path):
    reg, ev = _fake_records(tmp_path, _moe())
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_holds_a_row_with_a_declared_forced_slot_record(tmp_path):
    reg, ev = _fake_records(tmp_path, _forced())
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_holds_a_row_whose_other_divergences_a_sealed_settle_covers(tmp_path):
    reg, ev = _fake_records(tmp_path, _settled(3))
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


@pytest.mark.parametrize("form", list(BAD_FORMS))
def test_record_guard_names_each_bad_form(tmp_path, form):
    """each way a row can lose its backing is caught, with its own message"""
    edit, words = BAD_FORMS[form]
    reg, ev = _fake_records(tmp_path, edit)
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(words in x for x in found), found


@pytest.mark.parametrize("spelling", ["absolute", "through .."])
def test_record_guard_refuses_a_record_outside_the_records_root(tmp_path, spelling):
    """the same bytes, sealed by their path, but kept outside the records root (written as an absolute path, or as
    <root>/../elsewhere/rec.json, which reads as under the root): only the root rule catches it"""
    reg, ev = _fake_records(tmp_path)
    e = ev["gguf"][ROW["id"]]["v6e"]
    out = tmp_path / "elsewhere" / "rec.json"
    out.parent.mkdir()
    out.write_bytes(Path(e["record"]["path"]).read_bytes())
    name = str(out) if spelling == "absolute" else ev["records_root"] + "/../elsewhere/rec.json"
    sums = Path(e["sha256sums"]["path"])
    sums.write_text(sums.read_text().replace("  rec.json", f"  {name}"))
    e["sha256sums"]["sha256"], e["record"]["path"] = _sha(sums), name
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert found and all("is not under" in x for x in found), found


def test_the_one_number_is_read_from_its_record(tmp_path):
    """a model's one number (catalog.number) equals the values its Markdown record states, on the chip it names"""
    root = tmp_path / "records"
    root.mkdir()
    md = root / "RECORD.md"
    md.write_text("On ONE TPU v6e-1.\n\n| workload | conc | ours tok/s | vLLM tok/s |\n|---|---|---|---|\n"
                  "| W1 short | 1 | 38.1 | 31.4 |\n| W1 short | 8 | 1.5 | 2.5 |\n")
    m = copy.deepcopy(next(x for x in REG["models"] if x["id"] == "qwen3-8-27b"))
    m["catalog"]["params_b"] = {"total": None, "active": None}
    reg = dict(copy.deepcopy(REG), models=[m], gguf=[], engine_kernels=[])
    fields = ["table[workload=W1 short, conc=1].ours tok/s", "table[workload=W1 short, conc=1].vLLM tok/s"]
    ne = {"record": {"path": str(md), "sha256": _sha(md)}, "fields": fields}
    ev = {"records_root": str(root), "models": {m["id"]: {"catalog.params_b.total": {"value": None},
                                                          "catalog.number": ne}}}
    shown = catalog.number(m)
    assert shown["big"] == "38.1 vs 31.4 tok/s" and shown["chip"] == "TPU v6e-1"
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    bad = copy.deepcopy(ev)
    bad["models"][m["id"]]["catalog.number"]["fields"] = [fields[0], fields[0].replace("conc=1", "conc=8")]
    assert any("the record's" in x for x in evidence_problems(reg, bad, ROOT, tmp_path / "fit"))
    bad = copy.deepcopy(ev)
    del bad["models"][m["id"]]["catalog.number"]["record"]
    assert any("has no record" in x for x in evidence_problems(reg, bad, ROOT, tmp_path / "fit"))
    md.write_text(md.read_text().replace("TPU v6e-1", "TPU v5e-1"))
    ev["models"][m["id"]]["catalog.number"]["record"]["sha256"] = _sha(md)
    assert any("never names that chip" in x for x in evidence_problems(reg, ev, ROOT, tmp_path / "fit"))


def _models_literal(path):
    """a module's top-level MODELS dict literal from its source (not imported), None when it has none"""
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
            except ValueError:
                return None
            return v if isinstance(v, dict) else None
    return None


def arch_label_problems(src):
    """ruling 3: the Models label of each GGUF architecture lives in the module ARCHS maps it to, as a MODELS dict
    literal: every architecture has one non-empty label there, and no module labels an architecture ARCHS does not map
    to it"""
    by_mod = {}
    for a, mod in _archs_map(Path(src).read_text()).items():
        by_mod.setdefault(mod, []).append(a)
    out = []
    for mod, names in sorted(by_mod.items()):
        f = Path(src).parent / (mod.rsplit(".", 1)[-1] + ".py")
        models = _models_literal(f)
        if models is None:
            out.append(f"{f.name}: no MODELS dict literal")
            continue
        out += [f"{f.name}: no Models label for {a}" for a in names
                if not isinstance(models.get(a), str) or not models[a].strip()]
        out += [f"{f.name}: labels {a}, which ARCHS does not map to {mod}" for a in sorted(set(models) - set(names))]
    return out


def test_every_architecture_has_its_models_label_in_its_own_module():
    """the Models column of the architecture lists (docs/gguf.md) is each architecture module's own MODELS label, read
    without importing it; no hand list in catalog.py"""
    src = ROOT / ARCHS_SRC
    assert arch_label_problems(src) == []
    want = {a: _models_literal(ROOT / f)[a] for a, f in arch_module_files().items()}
    assert catalog.arch_models() == want and set(want) == set(catalog.archs())
    listed = [a for a in catalog.archs() if a in passing_archs(REG, want)]
    assert [r["models"] for r in catalog.arch_rows(REG)] == [want[a] for a in listed]
    assert "qwen35moe" not in listed or "qwen35moe" in passing_archs(REG, want)
    assert not hasattr(catalog, "ARCH_MODELS")


def passing_archs(reg, labels):
    """the architectures a catalog may list: each with a passing chip row of a GGUF file of it, and a mixture-of-experts
    architecture only with a passing chip row of a MoE file of it"""
    return {g["arch"] for g in reg.get("gguf") or [] if catalog.chips_run(g)
            and (g.get("moe") is True or not catalog.moe_arch(g["arch"], labels.get(g["arch"])))}


def _arch_tree(base, m1='{"a1": "A1", "a2": "A2"}', m2='{"b1": "B1"}'):
    """a planted archs folder: ARCHS maps a1 and a2 to m1, b1 to m2; m1 and m2 carry the MODELS given (None: none)"""
    d = base / "archs"
    d.mkdir()
    (d / "__init__.py").write_text('ARCHS = {"a1": "mashbox.gguf.archs.m1", "a2": "mashbox.gguf.archs.m1", '
                                   '"b1": "mashbox.gguf.archs.m2"}\n')
    for name, models in (("m1", m1), ("m2", m2)):
        (d / f"{name}.py").write_text("import os\n" + (f"MODELS = {models}\n" if models is not None else ""))
    return d / "__init__.py"


ARCH_LABEL_FORMS = {
    "an architecture with no label": (dict(m1='{"a1": "A1"}'), "m1.py: no Models label for a2"),
    "an empty label": (dict(m1='{"a1": " ", "a2": "A2"}'), "m1.py: no Models label for a1"),
    "a label that is not a string": (dict(m2='{"b1": 3}'), "m2.py: no Models label for b1"),
    "a module labelling an architecture ARCHS maps elsewhere": (dict(m2='{"b1": "B1", "a1": "X"}'),
                                                               "m2.py: labels a1, which ARCHS does not map to"),
    "a module labelling an architecture not in ARCHS": (dict(m1='{"a1": "A1", "a2": "A2", "zz": "Z"}'),
                                                       "m1.py: labels zz, which ARCHS does not map to"),
    "a module with no MODELS": (dict(m2=None), "m2.py: no MODELS dict literal"),
    "a MODELS that is not a literal": (dict(m2='dict(b1="B1")'), "m2.py: no MODELS dict literal"),
}


def test_a_planted_archs_folder_has_its_labels(tmp_path):
    src = _arch_tree(tmp_path)
    assert arch_label_problems(src) == []
    assert catalog.arch_models(str(src)) == {"a1": "A1", "a2": "A2", "b1": "B1"}
    assert catalog.arch_rows({"gguf": []}, str(src)) == []          # no passing chip row: no architecture is listed
    rows = [dict(ROW, id=f"{a}-row", arch=a) for a in ("b1", "a1")]
    assert [r["models"] for r in catalog.arch_rows({"gguf": rows}, str(src))] == ["A1", "B1"]   # in ARCHS order
    block = catalog.archs_block({"gguf": []}, str(src))
    assert block == catalog.NO_ARCHS and "|" not in block                # one sentence, never an empty table


@pytest.mark.parametrize("form", list(ARCH_LABEL_FORMS))
def test_each_bad_architecture_label_is_named(tmp_path, form):
    kw, words = ARCH_LABEL_FORMS[form]
    src = _arch_tree(tmp_path, **kw)
    found = arch_label_problems(src)
    assert any(words in x for x in found), found
    labels = catalog.arch_models(str(src))
    assert all(isinstance(v, str) and v.strip() for v in labels.values()), labels      # never a blank or odd label
    assert labels.get("a1") in ("A1", None)         # a label is only ever read from the architecture's own module


def test_the_release_check_finds_every_file_in_the_one_public_repo():
    """the release export is checked logged out: every files[] row of a released model is in that model's one public
    repo at the pinned revision (or the row's own revision of it), so a file another repo holds stops the release"""
    m = copy.deepcopy(next(x for x in REG["models"] if x["id"] == "qwen3-8-27b"))
    m["files"] = [f for f in m["files"] if not f.get("repo")] + [
        {"name": "Extra-BQ8_R.bshj", "kind": "draft", "format": "BSHJ", "gb": 0.1, "repo": "someone/Other-Repo",
         "revision": "1" * 40}]
    reg = dict(copy.deepcopy(REG), models=[m])
    rev, own = m["hf_revision"], [f["name"] for f in m["files"] if not f.get("repo")]

    class Fetch:
        def __init__(self, names):
            self.names = names

        def get(self, url):
            if url == f"https://huggingface.co/api/models/{m['hf_repo']}":
                return 200, json.dumps({"sha": rev})
            if url == f"https://huggingface.co/api/models/{m['hf_repo']}/revision/{rev}":
                return 200, json.dumps({"sha": rev, "siblings": [{"rfilename": n} for n in self.names]})
            return 404, ""

    def found(names):
        p = SC.Problems()
        SC.check_online(reg, {}, Fetch(names), p)
        return [x for x in p if "files[" in x]
    assert found(own + ["Extra-BQ8_R.bshj"]) == [
        f"hf {m['hf_repo']}: files[{len(own)}] Extra-BQ8_R.bshj is in someone/Other-Repo: every file of a released "
        f"model is in its one public repo {m['hf_repo']}"]
    assert any(f"{own[0]} is not in {m['hf_repo']} at {rev[:12]}" in x for x in found(own[1:]))


def test_no_private_repository_reaches_a_public_view():
    """models.json ships in the wheel and the export, and every view renders from it, so the generator refuses a catalog
    that names a private repository: no string in it says "private", and a files[] row has no repo of its own (every file
    of a model is in that model's one public repo, so a private repo under any name cannot be pinned there either)"""
    for rel in ("models.json", "src/mashbox/models.json"):
        assert re.findall(r'"[^"]*private[^"]*"', (ROOT / rel).read_text(), re.I) == [], rel
    i = next(n for n, x in enumerate(REG["models"]) if x["id"] == "qwen3-8-27b")
    hf_repo = REG["models"][i]["hf_repo"]
    edits = {
        "a row in a repo labelled private": ("private repository", lambda m, f: f.update(repo=f"{hf_repo}-private")),
        "a row in another repo": ("its one public repo", lambda m, f: f.update(repo="someone/Other-Repo")),
        "the label in a text field": ("private repository", lambda m, f: f.update(quality="the copy in our private repo")),
        "the label in a drafter field": ("private repository", lambda m, f: m["drafter"].update(note="Private copy")),
    }
    for name, (want, edit) in edits.items():
        reg = copy.deepcopy(REG)
        edit(reg["models"][i], reg["models"][i]["files"][0])
        assert any(want in x for x in problems(reg)), name
    assert any("private repository" in x for x in problems(with_row(dict(ROW, base="someone/Tiny-Qwen3-private"))))


def _moe_tree(base):
    """a planted archs folder with one dense architecture (d1), one MoE by name (m1moe) and one MoE by its label (m2)"""
    d = base / "archs"
    d.mkdir()
    (d / "__init__.py").write_text('ARCHS = {"d1": "mashbox.gguf.archs.k", "m1moe": "mashbox.gguf.archs.k", '
                                   '"m2": "mashbox.gguf.archs.k"}\n')
    (d / "k.py").write_text('MODELS = {"d1": "D1", "m1moe": "M1", "m2": "M2 (Mixture of Experts)"}\n')
    return d / "__init__.py"


def _moe_row(arch, moe=True, status="passing"):
    return dict(ROW, id=f"{arch}-row", arch=arch, moe=moe, chips={"v6e": dict(ROW["chips"]["v6e"], status=status)})


ARCH_LISTING_FORMS = {
    "no GGUF row of any arch": ([], []),
    "rows that are not passing": ([_moe_row("d1", moe=False, status="failing"), _moe_row("m1moe", status="failing"),
                                   _moe_row("m2", status="pending")], []),
    "a passing dense row of the dense arch": ([_moe_row("d1", moe=False)], ["d1"]),
    "a passing row of a MoE arch that is not a MoE row": ([_moe_row("m1moe", moe=False)], []),
    "a passing MoE row of another arch": ([_moe_row("d1")], ["d1"]),
    "a passing MoE row of each MoE arch, none of the dense one": ([_moe_row("m1moe"), _moe_row("m2")], ["m1moe", "m2"]),
    "a passing row of each": ([_moe_row("m2"), _moe_row("d1", moe=False), _moe_row("m1moe")], ["d1", "m1moe", "m2"]),
}


@pytest.mark.parametrize("form", list(ARCH_LISTING_FORMS))
def test_an_architecture_is_listed_only_with_a_passing_chip_row(tmp_path, form):
    """Architectures listed = architectures with a passing chip record. An architecture is in the architecture lists
    (docs/gguf.md and the views) only when models.json holds a passing chip row of a GGUF file of it, which the record
    guard holds to its sealed chip record; a mixture-of-experts architecture ("moe" in its name or "mixture of experts"
    in its Models label, the release check's rule) only with a passing chip row of a MoE file of it (experts on the
    kernel, 0 exact sites). Listing only: an unlisted dense architecture still runs on the exact path."""
    rows, want = ARCH_LISTING_FORMS[form]
    src = _moe_tree(tmp_path)
    reg = {"gguf": rows}
    assert [r["arch"] for r in catalog.arch_rows(reg, str(src))] == want
    block = catalog.archs_block(reg, str(src))
    for a in ("d1", "m1moe", "m2"):
        assert (f"`{a}`" in block) == (a in want), (a, block)
    assert (block == catalog.NO_ARCHS) == (not want)
    text = catalog.render_text(dict(REG, gguf=rows), str(src))
    assert (ARCH_LINE in text) == bool(want)


def test_the_docs_list_no_architecture_without_its_passing_chip_row():
    """the docs check on this tree: docs/gguf.md's architecture block lists an architecture only with a passing chip row
    of a GGUF file of it in models.json (a MoE architecture: of a MoE file of it), and is the one sentence when none has
    one"""
    mt = SC.ARCHS_RE.search((ROOT / "docs" / "gguf.md").read_text())
    assert mt, "docs/gguf.md has no generated block"
    listed = re.findall(r"^\| `([^`]+)` \| ([^|]*)\|", mt.group(2), re.M)
    passing = passing_archs(REG, catalog.arch_models())
    assert [a for a, _ in listed if a not in passing] == []
    assert sorted(a for a, _ in listed) == sorted(a for a in passing if a in catalog.archs())
    if not passing:
        assert mt.group(2).strip() == catalog.NO_ARCHS


def test_every_command_a_message_names_is_a_command():
    """each `mashbox <command>` named in a string of the package (messages, help, docstrings) is a command of the CLI"""
    import argparse
    from mashbox import cli
    sub = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    bad = []
    for p in sorted((ROOT / "src" / "mashbox").rglob("*.py")):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                bad += [f"{p.relative_to(ROOT)}:{node.lineno}: mashbox {w}"
                        for w in re.findall(r"`mashbox ([a-z][a-z0-9-]*)", node.value) if w not in sub.choices]
    assert bad == []
