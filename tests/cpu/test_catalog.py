"""The model catalog (models.json schema 2, mashbox/catalog.py): one source, many views. `mashbox models`, the docs
catalog block and the architecture list in docs/gguf.md are renders of models.json and of mashbox.gguf.archs.ARCHS, and
these tests fail when a view, a row or the code drift apart. GGUF rows are checked against the reader's own lists; the
record behind each claim is checked by the maintainer-only test at the end (the records are kept outside this
repository)."""
import ast
import copy
import datetime
import fnmatch
import functools
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
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
ML = SC._lists()   # the model lists every docs page renders (the docs catalog by family)
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
       "base": "Qwen/Tiny-Qwen3", "in": "text", "out": "text", "run": "mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M",
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
    """old files (and old apps' copies) keep passing: schema 1 = no catalog blocks, no gguf / engine_kernels / elsewhere"""
    old = copy.deepcopy(REG)
    old["schema"] = 1
    for k in ("gguf", "engine_kernels", "elsewhere"):
        old.pop(k, None)
    for m in old["models"]:
        m.pop("catalog")
    assert problems(old) == []
    for k in ("gguf", "elsewhere"):
        assert any(f"{k} needs schema 2" in x for x in problems(dict(old, **{k: []}))), k


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
        "date in text": dict(ROW, base="Qwen/Tiny-Qwen3 (read 2026-10-28T23:00Z)"),
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
    assert problems(with_row(dict(ROW, base="Qwen/Tiny-Qwen3-candidate"))) == []


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
    """every kind reads the same in `mashbox models` and the supported-models page (catalog.entries: KIND_LABEL, else
    the kind id) as in README, llms.txt and the cards (the kind table of tools/hf_github_sync_check.py)"""
    assert set(catalog.KIND_LABEL) <= set(SC.KINDS)
    assert {k: catalog.KIND_LABEL.get(k, k) for k in SC.KINDS} == {k: v["label"] for k, v in SC.KINDS.items()}
    assert catalog.CHIPS == SC.CHIPS


def test_a_card_names_a_base_model_only_with_the_relation_its_row_records():
    """Hugging Face reads a card's base_model with no base_model_relation as a fine-tune of that model. A card names its
    base model only with the relation its row records (quantized: the repo hosts our quantized weights of it). A row
    with none, such as a card-only port that hosts none of the model's weights, names no base model; its card text
    credits and links the maker. Our own model names none."""
    for m in REG["models"]:
        fm = SC.front_matter(REG, m).splitlines()
        base = [x for x in fm if x.startswith("base_model:")]
        rel = [x for x in fm if x.startswith("base_model_relation:")]
        if m.get("base_model_relation") and not m["upstream"].get("own"):
            assert (base, rel) == ([f"base_model: {m['upstream']['hf']}"],
                                   [f"base_model_relation: {m['base_model_relation']}"]), (m["id"], fm)
        else:
            assert (base, rel) == ([], []), (m["id"], fm)


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
        if mt:   # the docs catalog is the model lists' render (tools/model_lists.py), by family
            assert mt.group(2).strip() == ML.render_markdown(REG).strip(), page
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
    assert t.index("MASHBOX Labs") < t.index("\nsomeone\n") < t.index(f"  Tiny-Qwen3  (base {ROW['base']})") < t.index(
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
            dict(ROW, id="alpha-gguf-q4-k-m", model="alpha", base="Qwen/Alpha", repo="someone/Alpha-GGUF",
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
    reg["gguf"][1]["base"] = "Qwen/Tiny-Qwen3-Other"
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


@pytest.mark.parametrize("bad", ["/tmp/{f}", "../../outside/{f}", "sub/{f}", "sub\\{f}", ".", ".."])
def test_an_engine_kernels_file_is_a_bare_file_name(bad):
    """the loader joins the row's file under its digest directory in the cache: an absolute path, a separator or a
    parent step would put the verified wheel outside it, so the catalog refuses any file that is not a bare name"""
    f = bad.format(f=BUNDLE["file"])
    assert any("bare file name" in x for x in problems(with_row(bundle=dict(BUNDLE, file=f)))), f


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


@pytest.mark.parametrize("src", ["none", "imports", "fails", "on path"],
                         ids=["tree without src", "tree with src", "src fails to import", "src already on sys.path"])
def test_package_import_leaves_sys_path_as_it_found_it(tmp_path, src):
    # _package() imports mashbox from <tree>/src for the docs views and must not leave that folder on sys.path: the
    # numbers check calls it on a test's tmp tree, and test_b3_sigint later built its worker's PYTHONPATH from
    # sys.path[:3], which three such folders pushed the real src out of ("No module named mashbox" in the worker)
    pkg = tmp_path / "src" / "mashbox"
    if src != "none":
        (pkg / "runtime").mkdir(parents=True)
        for rel in ("__init__.py", "cli.py", "tpu.py", "catalog.py", "runtime/__init__.py", "runtime/openapi.py"):
            (pkg / rel).write_text("")
        (pkg / "late.py").write_text("WHERE = 'tmp tree'\n")
        if src == "fails":
            (pkg / "tpu.py").write_text("raise ImportError('this tree is broken')\n")
    cached = {n: m for n, m in sys.modules.items() if n == "mashbox" or n.startswith("mashbox.")}
    outer = list(sys.path)
    if src == "on path":    # the caller's own entry stays where it was
        sys.path.insert(1, str(pkg.parent.resolve()))
    before = list(sys.path)
    try:
        problems = SC.Problems()
        mods = SC._package(tmp_path, problems)
        assert sys.path == before
        if src in ("imports", "on path"):
            # the import still came from the tree asked for, and so does a submodule imported after the call returns
            # (through mashbox.__path__, not sys.path), as the docs views import theirs
            assert mods is not None and Path(mods[""].__file__).resolve().parent == pkg.resolve()
            assert importlib.import_module("mashbox.late").WHERE == "tmp tree"
        else:
            want = "cannot import the package" if src == "fails" else "mashbox imported from"
            assert mods is None and len(problems) == 1 and want in problems[0]
    finally:
        sys.path[:] = outer
        for n in [n for n in sys.modules if n == "mashbox" or n.startswith("mashbox.")]:
            del sys.modules[n]
        sys.modules.update(cached)


@pytest.mark.parametrize("call", ["docs_views", "check_commands", "check_numbers"])
@pytest.mark.parametrize("src", ["none", "stub"], ids=["tree without src", "tree with a stub src"])
def test_a_check_on_another_tree_leaves_the_loaded_mashbox_as_it_found_it(tmp_path, src, call):
    # The docs views, the command check and the numbers check import mashbox from the tree they check (_package), which
    # first drops a different loaded mashbox from sys.modules. They must put it back when they return: test_core's
    # numbers check on a tmp tree left new copies of every mashbox module behind, and test_hf_safetensors' native BF16
    # tests then patched the dense16 they had imported while the code ran the new copy (3 failed after test_core in one
    # process, 4 passed alone)
    import mashbox.ops.dense16  # noqa: F401  (a submodule a later test holds and patches)
    found = {n: m for n, m in sys.modules.items() if n == "mashbox" or n.startswith("mashbox.")}
    if src == "stub":    # imports, then stops the check part way (no parsers, no pyproject): the other way out
        pkg = tmp_path / "src" / "mashbox"
        (pkg / "runtime").mkdir(parents=True)
        for rel in ("__init__.py", "cli.py", "tpu.py", "catalog.py", "runtime/__init__.py", "runtime/openapi.py"):
            (pkg / rel).write_text("")
    (tmp_path / "docs").mkdir()
    run = {"docs_views": SC.docs_views, "check_commands": SC.check_commands, "check_numbers": SC.check_numbers}[call]
    try:
        try:
            run(REG, tmp_path, SC.Problems())
        except (AttributeError, FileNotFoundError, KeyError):
            assert src == "stub"
        now = {n: m for n, m in sys.modules.items() if n == "mashbox" or n.startswith("mashbox.")}
        assert sorted(set(now) ^ set(found)) == [] and [n for n in found if now[n] is not found[n]] == []
    finally:
        for n in [n for n in sys.modules if n == "mashbox" or n.startswith("mashbox.")]:
            del sys.modules[n]
        sys.modules.update(found)


def _mashbox_modules():
    return {n: m for n, m in sys.modules.items() if n == "mashbox" or n.startswith("mashbox.")}


def _parents_disagree():
    """the loaded mashbox modules whose parent package holds another object under the name, and the submodules a
    loaded package holds that sys.modules does not (the next import of that name would make a second copy)"""
    bad = []
    for n, m in _mashbox_modules().items():
        parent, _, child = n.rpartition(".")
        if parent and getattr(sys.modules.get(parent), child, m) is not m:
            bad.append(n)
        bad += [v.__name__ for k, v in vars(m).items()
                if isinstance(v, types.ModuleType) and v.__name__ == f"{n}.{k}" and sys.modules.get(v.__name__) is not v]
    return sorted(bad)


def _stub_tree(root):
    """a tree whose src/mashbox imports and has none of the check's functions (the check stops part way)"""
    pkg = root / "src" / "mashbox"
    (pkg / "runtime").mkdir(parents=True)
    for rel in ("__init__.py", "cli.py", "tpu.py", "catalog.py", "runtime/__init__.py", "runtime/openapi.py"):
        (pkg / rel).write_text("")
    (root / "docs").mkdir()


@pytest.mark.parametrize("call", ["docs_views", "check_commands", "check_numbers"])
def test_a_check_of_the_loaded_tree_keeps_what_it_found_and_what_it_imported(tmp_path, call):
    # When the loaded mashbox is the checked tree's own, nothing is put back: the module objects found stay, and a
    # submodule the check imports for the first time (runtime.openapi) stays loaded as its parent's attribute, as after
    # any import. Putting back on every call would drop it from sys.modules while its parent still holds it, so the
    # next import of that name would make a second copy
    import importlib
    keep = _mashbox_modules()
    _stub_tree(tmp_path)
    run = {"docs_views": SC.docs_views, "check_commands": SC.check_commands, "check_numbers": SC.check_numbers}[call]
    try:
        for n in keep:
            del sys.modules[n]
        sys.path.insert(0, str(tmp_path / "src"))
        try:
            for n in ("mashbox", "mashbox.cli", "mashbox.tpu", "mashbox.catalog", "mashbox.runtime"):
                importlib.import_module(n)
        finally:
            sys.path.remove(str(tmp_path / "src"))
        found = _mashbox_modules()
        assert "mashbox.runtime.openapi" not in found and _parents_disagree() == []
        try:
            run(REG, tmp_path, SC.Problems())
        except (AttributeError, FileNotFoundError, KeyError):
            pass
        now = _mashbox_modules()
        assert [n for n in found if now.get(n) is not found[n]] == []
        assert "mashbox.runtime.openapi" in now and _parents_disagree() == [], _parents_disagree()
    finally:
        for n in list(_mashbox_modules()):
            del sys.modules[n]
        sys.modules.update(keep)


@pytest.mark.parametrize("call", ["docs_views", "check_commands", "check_numbers"])
@pytest.mark.parametrize("src", ["none", "stub"], ids=["tree without src", "tree with a stub src"])
def test_a_check_with_no_mashbox_loaded_leaves_none_loaded(tmp_path, src, call):
    # A process with no mashbox loaded has none loaded after a check: the package the check imported is dropped on the
    # way out, as a different one would be. Putting back only when something was found would leave the checked tree's
    # modules behind for the next import of mashbox
    keep = _mashbox_modules()
    if src == "stub":
        _stub_tree(tmp_path)
    else:
        (tmp_path / "docs").mkdir()
    run = {"docs_views": SC.docs_views, "check_commands": SC.check_commands, "check_numbers": SC.check_numbers}[call]
    try:
        for n in keep:
            del sys.modules[n]
        try:
            run(REG, tmp_path, SC.Problems())
        except (AttributeError, FileNotFoundError, KeyError):
            pass
        assert sorted(_mashbox_modules()) == []
    finally:
        for n in list(_mashbox_modules()):
            del sys.modules[n]
        sys.modules.update(keep)


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
    from mashbox.gguf import pick as GR
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
# ops/ file hold the same code as the code it ran (compared as code: _code_only), and the engine_kernels rows of
# models.json the loader reads are equal.
CODE_SET_VERSION = 1
CODE_SET = {
    "entry": ("src/mashbox/__main__.py", "src/mashbox/__init__.py", "src/mashbox/cli.py"),
    "resolver and loader": ("src/mashbox/gguf/pick.py", "src/mashbox/gguf/reader.py", "src/mashbox/gguf/tokenizer.py",
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


def code_problems(repo, git_dir, commit, arch, what, kernels=None, chip_of=None, equiv=()):
    """the record's code commit (in git_dir) against this tree: ARCHS maps the arch to the same module, every file of
    gguf_code_files holds the same code (_code_only: comments, docstrings and import lines never void a record, as for
    a CPU record) or, for a file that differs, a sealed equivalence that holds (equiv, from equivalence_problems) ran
    this tree's code of it (equivalent_file), and no ops/ file the record ran is missing here"""
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
        elif _code_only(f, old) != _code_only(f, (repo / f).read_bytes()):
            out += equivalent_file(repo, git_dir, f, commit, equiv, what)
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


EQUIV_PROMPTS = 16     # the record's sealed prompts (ids-<tag>.json), every one run at 1 user and at 8 users
EQUIV_MIN_TOKENS = 64
IDS_RE = re.compile(r"ids-.+\.json")


def equivalence_problems(g, e, listed, read, root, git_dir, what):
    """(problems, held): each sealed equivalence the evidence cites (e["equivalence"]: sha256sums, ids, a, c) holds
    only when its seal lists the ids file and both run records with their sha256s; the ids file is byte for byte the one
    the record's own seal lists (the record's 16 prompts); run A ran the record's commit and run C a commit in code_repo,
    each run's code_files are its commit's files; both ran the row's file with no MASHBOX_ variable, the same N (at least
    64) on the same jax, device, host and group; and every one of the 16 prompts gave the same ids in A and C at 1 user
    and at 8 users, read from the id lists (never from a summary). held lists (C's commit, the files both runs hashed)
    for each equivalence that holds."""
    out, held = [], []
    for n, q in enumerate(e.get("equivalence") or []):
        w, bad = f"{what} equivalence[{n}]", []
        if any(k not in q for k in ("sha256sums", "ids", "a", "c")):
            out.append(f"{w}: the evidence needs sha256sums, ids, a and c")
            continue
        sums = read(q["sha256sums"], f"{w} sha256sums")
        if sums is None:
            continue
        mine = _sums(sums)
        bad += seal_problems(sums, root, w)
        for k in ("ids", "a", "c"):
            if mine.get(os.path.normpath(q[k]["path"])) != q[k]["sha256"]:
                bad.append(f"{w}: {q[k]['path']} is not in {sums.name} with its sha256")
        ids, a, c = (read(q[k], f"{w} {k}") for k in ("ids", "a", "c"))
        if not all(isinstance(x, dict) for x in (ids, a, c)):
            out += bad + [f"{w}: the ids file and both run records must be JSON objects"]
            continue
        rids = [p for p in listed if IDS_RE.fullmatch(Path(p).name)]
        if len(rids) != 1:
            bad.append(f"{w}: the record's seal lists {len(rids)} ids files, so its prompts are not pinned")
        elif listed[rids[0]] != q["ids"]["sha256"]:
            bad.append(f"{w}: the equivalence's prompt ids are not the record's ({Path(rids[0]).name})")
        if len(ids.get("ids") or []) != EQUIV_PROMPTS:
            bad.append(f"{w}: the ids file has {len(ids.get('ids') or [])} prompts, not {EQUIV_PROMPTS}")
        if a.get("commit") != e["code"]["commit"]:
            bad.append(f"{w}: run A ran {str(a.get('commit'))[:12]}, the record ran {e['code']['commit'][:12]}")
        for t, r in (("A", a), ("C", c)):
            commit, files = str(r.get("commit")), r.get("code_files")
            if _git(git_dir, "cat-file", "-e", f"{commit}^{{commit}}") is None:
                bad.append(f"{w}: run {t}'s commit {commit[:12]} is not in {git_dir}")
            if not isinstance(files, dict) or not files:
                bad.append(f"{w}: run {t} names no code_files")
                continue
            for f, h in sorted(files.items()):
                blob = _git(git_dir, "show", f"{commit}:{f}")
                if h != (None if blob is None else hashlib.sha256(blob).hexdigest()):
                    bad.append(f"{w}: run {t}'s {f} is not the code of its commit {commit[:12]}")
            if r.get("file_sha256") != g["sha256"]:
                bad.append(f"{w}: run {t} ran another file than the row's ({str(r.get('file_sha256'))[:16]}...)")
            if r.get("env") != {}:
                env = r.get("env") if isinstance(r.get("env"), dict) else {"env": r.get("env")}
                bad.append(f"{w}: run {t} ran with {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}")
        for k in ("N", "jax", "device", "host", "group"):
            if a.get(k) != c.get(k):
                bad.append(f"{w}: runs A and C differ in {k} ({a.get(k)} / {c.get(k)})")
        N = a.get("N")
        if not isinstance(N, int) or isinstance(N, bool) or N < EQUIV_MIN_TOKENS:
            bad.append(f"{w}: the runs decoded {N} tokens, fewer than {EQUIV_MIN_TOKENS}")
        for k, users in (("one", "1 user"), ("eight", "8 users")):
            for t, r in (("A", a), ("C", c)):
                rows = r.get(k)
                if not isinstance(rows, list) or len(rows) != EQUIV_PROMPTS:
                    bad.append(f"{w}: run {t} has {len(rows) if isinstance(rows, list) else 0} prompts at {users}, "
                               f"not {EQUIV_PROMPTS}")
                elif any(not isinstance(x, list) or len(x) != N for x in rows):
                    bad.append(f"{w}: run {t} has a prompt at {users} without {N} ids")
            if isinstance(a.get(k), list) and isinstance(c.get(k), list):
                diff = [i for i in range(max(len(a[k]), len(c[k])))
                        if i >= len(a[k]) or i >= len(c[k]) or a[k][i] != c[k][i]]
                if diff:
                    bad.append(f"{w}: ids differ between run A and run C at {users} for prompts {diff}")
        out += bad
        if not bad:
            held.append((c["commit"], frozenset(a["code_files"]) & frozenset(c["code_files"])))
    return out, held


def equivalent_file(repo, git_dir, f, commit, equiv, what):
    """[] when a held equivalence's run C hashed f and C's f is this tree's f as code; else the problem"""
    now = _code_only(f, (repo / f).read_bytes())
    for c, files in equiv:
        then = _git(git_dir, "show", f"{c}:{f}") if f in files else None
        if then is not None and _code_only(f, then) == now:
            return []
    tail = (" and from the code of the equivalence's run C " + ", ".join(c[:12] for c, _ in equiv)) if equiv else ""
    return [f"{what}: {f} differs from the code the record ran ({commit[:12]}){tail}"]


CODE_ID_RE = re.compile(r"code-.+\.json")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def code_id_problems(e, listed, read, git_dir, what):
    """A run's sealed code id (code-<tag>.json) binds its record to the code commit the evidence
    names. When the seal lists one, the evidence cites it (code_id, path + sha256) and the seal lists no other; the cited
    one is in the seal with its sha256 and is a JSON object whose code_sha is that commit, whose src_tree and
    pyproject_blob are that commit's src/ tree and pyproject.toml blob (read with git from code_repo), and whose
    wheel_content_sha256 is a sha256 equal to the evidence's (the wheel file's own sha256 changes with its zip
    timestamps, so it is not compared). Which code counts as the same stays code set v1 (code_problems), with or
    without a code id: src/ holds the packaged catalog, so its tree changes with every catalog row. A record from before
    code ids has none in its seal and keeps the commit path alone."""
    ids = sorted(p for p in listed if CODE_ID_RE.fullmatch(Path(p).name))
    if "code_id" not in e:
        return [f"{what}: the seal lists {Path(ids[0]).name}, and the evidence cites no code id"] if ids else []
    out = [f"{what}: the seal lists {len(ids)} code ids"] if len(ids) > 1 else []
    if listed.get(os.path.normpath(e["code_id"]["path"])) != e["code_id"]["sha256"]:
        out.append(f"{what}: {Path(e['code_id']['path']).name} is not in the seal with its sha256")
    cid = read(e["code_id"], f"{what} code id")
    if cid is None:
        return out
    if not isinstance(cid, dict):
        return out + [f"{what}: the code id is not a JSON object"]
    commit = e["code"]["commit"]
    if cid.get("code_sha") != commit:
        out.append(f"{what}: the code id names {cid.get('code_sha')}, the evidence names {commit}")
    elif git_dir:
        for key, rev in (("src_tree", f"{commit}:src"), ("pyproject_blob", f"{commit}:pyproject.toml")):
            want = (_git(git_dir, "rev-parse", "--verify", "-q", rev) or b"").decode().strip() or None
            if want is None or cid.get(key) != want:
                out.append(f"{what}: the code id's {key} is {cid.get(key)}, the code {commit[:12]} has {want}")
    w = cid.get("wheel_content_sha256")
    if not (isinstance(w, str) and SHA256_RE.fullmatch(w)):
        out.append(f"{what}: the code id has no wheel_content_sha256")
    elif w != e["code"].get("wheel_content_sha256"):
        out.append(f"{what}: the code id's wheel_content_sha256 is {w[:16]}..., the evidence names "
                   f"{str(e['code'].get('wheel_content_sha256'))[:16]}...")
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
    this tree's (read with git from ev["code_repo"]); a run's sealed code id binds the record to that code commit
    (code_id_problems); and the release bundle's manifest is built from the same kq commit and has every quantized type
    of the file. Our models' parameter counts and their one number are read from records
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
        if isinstance(man, dict):       # the manifest's impls are the attested ones
            out += attest_problems(k["file"], ke, man, read)
        for ch in k["chips"]:
            manifests[ch] = man
    for g in reg.get("gguf") or []:
        if "cpu_check" in g:
            out += cpu_record_problems(g, (ev.get("gguf") or {}).get(g["id"], {}).get("cpu_check"), read, root, repo,
                                       git_dir, f"{g['id']} cpu_check")
        if g.get("speculation"):        # gate entries `--speculation auto` reads: each from its sealed n-gram runs
            out += quick_ngram_problems(g, (ev.get("gguf") or {}).get(g["id"], {}), read, root,
                                        reg.get("engine_kernels") or [])
        for chip, r in g["chips"].items():
            if r.get("status") == "pending":      # nobody ran it there: "Not tested yet", no record claimed
                continue
            what = f"{g['id']} {chip}"
            e = (ev.get("gguf") or {}).get(g["id"], {}).get(chip)
            if e is None:
                out.append(f"{what}: no evidence entry")
                continue
            if e.get("kind") in QUICK_KINDS or "record_kind" in r:      # a quick-check or QUICK run: its own record kind
                out += quick_check_problems(g, r, e, read, root, repo, git_dir, reg.get("engine_kernels") or [], chip, what,
                                      work)
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
            out += row_file_problems(g, have, what)
            code, kq = e["code"]["commit"], (e.get("kernels") or {}).get("commit")
            out += code_id_problems(e, listed, read, git_dir, what)
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
            if r["status"] == "passing" and isinstance(cmp, dict):     # read from the rows, not the summary
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
                eq, held = equivalence_problems(g, e, listed, read, root, git_dir, what)
                out += eq + code_problems(repo, git_dir, code, g["arch"], what, reg.get("engine_kernels") or [], chip,
                                          equiv=held)
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


def row_file_problems(g, have, what):
    """A chip row's file, plan and card against the row: the file-*.sha256 names the row's file, sha256 and size, the
    plan its repo, revision, file and sha256, and the pinned card its repo, revision, license, license_link and base
    (never typed)"""
    out = []
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
    elif card is not None:      # the row's card fields are the pinned card's, never typed
        cd = card.get("cardData") or {}
        base, link = cd.get("base_model"), cd.get("license_link")
        for field, ours, theirs in (("repo", g["repo"], card.get("id")), ("revision", g["revision"], card.get("sha")),
                                    ("license", g["license"], cd.get("license"))):
            if ours != theirs:
                out.append(f"{what}: {field} {ours!r}, the pinned card says {theirs!r}")
        repo, rev = card.get("id"), card.get("sha")
        if link is not None and _card_link(g["license_link"], repo, rev) != _card_link(link, repo, rev):
            out.append(f"{what}: license_link {g['license_link']!r}, the pinned card says {link!r}")
        elif link is None and not str(g["license_link"]).startswith(f"https://huggingface.co/{g['repo']}/"):
            out.append(f"{what}: license_link {g['license_link']!r}, and the pinned card names none, so it is a file "
                       f"of {g['repo']}")
        org = str(repo or "").split("/")[0]
        theirs = [_card_repo(b, org) for b in (base if isinstance(base, list) else [base])]
        if _card_repo(g["base"], org) not in theirs:
            out.append(f"{what}: base {g['base']!r}, the pinned card says {base!r}")
    return out


def _card_link(link, repo, rev):
    """a licence link by meaning: a URL as it is; a relative path (the card's "LICENSE") is that file of the card's repo
    at the pinned revision. The licence it names is never changed or compared here."""
    s = str(link or "")
    if re.match(r"^[a-z][a-z0-9+.-]*://", s) or not s:
        return s
    return f"https://huggingface.co/{repo}/blob/{rev}/{s.lstrip('./')}"


def _card_repo(name, org):
    """a base model by meaning, as an HF repo id: "https://huggingface.co/A/B" -> "A/B"; "B" with no org -> "<org>/B",
    the card repo's own org; None stays None (the row may name none then)"""
    if name is None:
        return None
    s = re.sub(r"^https?://huggingface\.co/", "", str(name)).strip("/")
    return s if "/" in s or not org else f"{org}/{s}"


# The locked CPU bars a cpu_check row's record must carry, and the files whose code
# decides a CPU record: the import closure inside src/mashbox of tools/gguf_conform.py and the record's arch module, plus
# that tool and the llama.cpp reference sources it builds. Nothing else voids a record (measured numbers stand):
# tools/gguf_scan.py, other architectures, ops the arch does not import, comments, docstrings and import lines.
CPU_BARS = dict(identical_weights_nmse_max=1e-4, top1_min_share=15 / 16, near_tie_gap=NEAR_TIE_GAP,
                other_divergences_max=0, dequant_vs_gguf_py="bit-exact", dequant_vs_ggml_max_ulp=1)
CPU_TOOL = "tools/gguf_conform.py"
# imported only by gguf.archs.held (the MoE hold): it refuses a MoE architecture before any check and holds no dense one,
# and no MoE row is listed from a CPU check, so the catalog view never decides a CPU record
CPU_NOT_DECIDING = ("src/mashbox/catalog.py",)
CPU_REFERENCE = ("tools/gguf_fixtures/llama_greedy.cpp", "tools/gguf_fixtures/llama_logits.cpp",
                 "tools/gguf_fixtures/ggml_dequant.c")


def cpu_record_files(repo, arch):
    """the files that decide a CPU record of a file of `arch` (sorted); [] when ARCHS has no such arch"""
    mod = _archs_map((repo / ARCHS_SRC).read_text()).get(arch)
    if mod is None:
        return []
    return sorted((import_closure(repo, [CPU_TOOL, _module_file(repo, mod)]) | set(CPU_REFERENCE)) - set(CPU_NOT_DECIDING))


def _code_only(path, data):
    """what decides a file's behaviour: the bytes of a non-Python file; for Python, its syntax tree without import
    lines and docstrings (so an unused import or a comment never voids a measured record)"""
    if not path.endswith(".py"):
        return data
    tree = ast.parse(data)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            body[:] = [b for b in body if not isinstance(b, (ast.Import, ast.ImportFrom))]
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
    return ast.dump(tree)


def conform_code_id(git_dir, commit):
    """gguf_conform.code_id() of the tree at `commit`, read with git: sha256 over (path NUL bytes NUL) of every file its
    CODE_FILES patterns match there, sorted by path; None when the commit or the tool is missing"""
    src = _git(git_dir, "show", f"{commit}:{CPU_TOOL}")
    names = _git(git_dir, "ls-tree", "-r", "--name-only", commit)
    if src is None or names is None:
        return None
    pats = next((ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "CODE_FILES" for t in n.targets)), [])

    def hit(name, pat):
        a, b = name.split("/"), pat.split("/")
        return len(a) == len(b) and all(fnmatch.fnmatchcase(x, y) for x, y in zip(a, b))
    h = hashlib.sha256()
    for f in sorted(n for n in names.decode().split() if any(hit(n, q) for q in pats)):
        h.update(f.encode() + b"\0" + _git(git_dir, "show", f"{commit}:{f}") + b"\0")
    return h.hexdigest()


def cpu_check_of(rec):
    """a row's cpu_check block computed from its sealed gguf_conform record (tools/gguf_rows.py writes the same)"""
    c, tk, same = rec["c_cpu"], rec["b_tokenizer"], rec["c_cpu"]["identical_weights"]
    return dict(status="passing" if rec["verdict"] == "cpu-checked" else "failing",
                reference=f"llama.cpp {rec['versions']['llama_cpp'][:7]} CPU",
                identical_weights=dict(nmse=float(f"{same['nmse']:.2g}"), top1=f"{same['top1_agree']}/{same['positions']}"),
                block_path=dict(top1=f"{c['top1_agree']}/{c['positions']}", kl=float(f"{c['kl_mean']:.2g}")),
                greedy=dict(prompts=c["prompts"], tokens=c["n_gen"], full=c["full_match"],
                            near_tie=c["near_tie_divergences"], other=c["other_divergences"]),
                tokenizer=dict(texts=tk["texts"], equal=tk["nfc_equal"], explained=tk["nfc_explained"]))


def cpu_record_problems(g, e, read, root, repo, git_dir, what):
    """A cpu_check row against its sealed gguf_conform record ([] when it holds): the evidence names the record, its
    SHA256SUMS, card.json, header.bin and the quant record (path + sha256 under the records root; the seal lists the
    first three), the code commit, and `decides` (the files that decide it, never fewer than cpu_record_files). The
    record is a gguf-conformance record of this row's repo, revision, file, sha256, size, arch, moe and types, with the
    locked bars, a verdict matching the status, and every cpu_check value equal to the record's (cpu_check_of); the
    quant record is the one it names and passes every quantized type of the file; the pinned card names this repo,
    revision and licence. Code: the commit's gguf_conform code id equals the record's (so it is the code that ran),
    and only the files in `decides` must hold the same code in this tree (_code_only)."""
    if e is None:
        return [f"{what}: no evidence entry"]
    out = []
    have = {k: read(e[k], f"{what} {k}") if k in e else out.append(f"{what}: no {k} in the evidence")
            for k in ("record", "sha256sums", "card", "header", "quant")}
    if have["sha256sums"] is not None:
        listed = _sums(have["sha256sums"])
        out += seal_problems(have["sha256sums"], root, what)
        for k in ("record", "card", "header"):
            if k in e and listed.get(os.path.normpath(e[k]["path"])) != e[k]["sha256"]:
                out.append(f"{what}: {e[k]['path']} is not in {have['sha256sums'].name} with its sha256")
    rec, card = have["record"], have["card"]
    if isinstance(rec, dict):
        pin = rec.get("pin") or {}
        for k, want in (("kind", "gguf-conformance"), ("pin.repo", g["repo"]), ("pin.revision", g["revision"]),
                        ("pin.file", g["file"]), ("sha256", g["sha256"]), ("bytes", g["bytes"]), ("arch", g["arch"]),
                        ("moe", g["moe"]), ("types", g["types"])):
            got = pin.get(k[4:]) if k.startswith("pin.") else rec.get(k)
            if got != want:
                out.append(f"{what}: the record's {k} is {got!r}, the row says {want!r}")
        bars = {k: (rec.get("bars") or {}).get(k) for k in CPU_BARS}
        if bars != CPU_BARS:
            out.append(f"{what}: the record's bars {bars} are not the locked CPU bars {CPU_BARS}")
        try:
            want = cpu_check_of(rec)
        except (KeyError, TypeError, ValueError):
            want = None
            out.append(f"{what}: the record has no complete CPU check")
        if want is not None and want != g["cpu_check"]:
            diff = sorted(k for k in set(want) | set(g["cpu_check"]) if want.get(k) != g["cpu_check"].get(k))
            out.append(f"{what}: cpu_check {diff} differ from the record (status from verdict {rec.get('verdict')!r})")
        q = rec.get("d_quant") or {}
        if "quant" in e and q.get("record_sha256") != e["quant"]["sha256"]:
            out.append(f"{what}: the record names quant record {str(q.get('record_sha256'))[:16]}, the evidence another")
        lack = [t for t in g["types"] if t not in catalog.UNQUANTIZED and (q.get("types") or {}).get(t) is not True]
        if lack:
            out.append(f"{what}: the record's quant check does not pass {lack}")
        if git_dir and "code" in e:
            commit = e["code"].get("commit", "")
            cid = conform_code_id(git_dir, commit)
            if cid is None or cid != (rec.get("code") or {}).get("sha256"):
                out.append(f"{what}: the code id at {commit[:12]} is {str(cid)[:16]}, the record ran "
                           f"{str((rec.get('code') or {}).get('sha256'))[:16]}")
            need = cpu_record_files(repo, g["arch"])
            decides = list(e.get("decides") or [])
            if not need or set(need) - set(decides):
                out.append(f"{what}: decides must list every file that decides the record: {sorted(set(need) - set(decides))}")
            for f in sorted(set(need) | set(decides)):
                old, now = _git(git_dir, "show", f"{commit}:{f}"), repo / f
                if old is None:
                    out.append(f"{what}: {f} is not in the code the record ran ({commit[:12]})")
                elif not now.is_file() or _code_only(f, old) != _code_only(f, now.read_bytes()):
                    out.append(f"{what}: {f} differs from the code the record ran ({commit[:12]})")
        elif "code" not in e:
            out.append(f"{what}: no code in the evidence")
    if isinstance(card, dict):
        cd = card.get("cardData") or {}
        for field, ours, theirs in (("repo", g["repo"], card.get("id")), ("revision", g["revision"], card.get("sha")),
                                    ("license", g["license"], cd.get("license"))):
            if ours != theirs:
                out.append(f"{what}: {field} {ours!r}, the pinned card says {theirs!r}")
    elif card is not None:
        out.append(f"{what}: the card snapshot is not JSON")
    return out


# the dists every chip record names in versions{} (importlib.metadata.version in the run's process; python is
# platform.python_version()), as posted for harness v2
KERNELS_DIST = "mashbox_engine_kernels"   # the kernel package as a record's versions{} names it (a pin, not an MoE rule)
RUN_DISTS = ("python", "jax", "jaxlib", "libtpu", "gguf", "tokenizers", "jinja2", "numpy", KERNELS_DIST)


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
    """Pinned versions: the record names every dist it ran (versions{}); each one this tree pins is the
    pinned version; and jax and libtpu are the ones the release bundle on the chip is built for. The kernel package by
    path: an exact-path record (zero flags, no kernel package installed) leaves it out or names it None, never a
    version; a bundle record names the version of the release bundle on the chip."""
    v = rec.get("versions")
    if not isinstance(v, dict):
        return [f"{what}: the record has no versions (the dists the run imported)"]
    pins, out = tree_pins(repo), []
    for n in sorted(set(RUN_DISTS) | set(pins)):
        if exact and n == KERNELS_DIST:
            if v.get(n) is not None:
                out.append(f"{what}: kernel_path exact, and the record ran {n} {v[n]} (an exact-path zero-flag run has "
                           "no kernel package installed)")
        elif not isinstance(v.get(n), str) or not v[n]:
            out.append(f"{what}: the record's versions have no {n}")
        elif n == KERNELS_DIST and krow is not None and v[n] != bundle_version(krow):
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
    """Kernel attestation: the evidence seals the bundle's attestation (ATTESTATION.json), the manifest
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
    """A GGUF chip row: its record is the zero-flag run, and its slots and context are this tree's fit
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
    impl, want = rec.get("impl"), {t: (man.get("impl") or {}).get(t) for t in kt}
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


# A quick-check chip run (record_kind "quick-check"): four runs of `mashbox run REPO:TAG --prompt P --max-tokens 64
# --temperature 0` with zero engine flags on one chip, each compared with llama.cpp's greedy ids on the same templated
# prompt ids. A QUICK chip run (record_kind "quick", catalog.QUICK_KIND) is the same check with catalog.QUICK_TOKENS
# tokens, its compare tool sealed as cmp_q.py, and its file pin sealed from the pick (no CPU record behind it). Its seal
# folder holds these files by name, besides the file/plan/card/header the evidence cites.
QUICK_RUNS, QUICK_TOKENS = 4, 64
QUICK_KINDS = {"quick-check": (QUICK_TOKENS, "cmp_p1.py"), "quick": (32, "cmp_q.py")}     # catalog.QUICK_KIND, QUICK_TOKENS


def quick_files(kind):
    """the files a seal of this record kind holds by name"""
    return ("cmp.json", QUICK_KINDS[kind][1], "setup.log", "freeze.txt", "accelerator-type.txt", "CODE_SHA",
            *(f"{x}{i}{y}" for i in range(QUICK_RUNS) for x, y in (
                ("run", ".json"), ("run", ".rc"), ("run", ".stdout"), ("run", ".stderr"), ("prompt", ".txt"),
                ("ref", ".ids"))))


QUICK_FILES = quick_files("quick-check")
QUICK_MEASURED = ("decode_tok_s_1_user", "steady_decode_tok_s_1_user", "steady_condition")   # each optional
QUICK_DEVICE_RE = re.compile(r"^\S+ \S+ \S+ (TPU .+)$", re.M)      # setup's last check: mashbox, jax, libtpu, device kind


def quick_check_decode(runs):
    """the row's number: decode tok/s at 1 user = sum(new tokens) / sum(decode seconds) over runs 1-3 (run 0 holds the
    first compile), 2 decimals"""
    t = [r["generate_calls"][0]["timing"] for r in runs[1:]]
    return round(sum(x["new_tokens"] for x in t) / sum(x["decode_s"] for x in t), 2)


def quick_check_first_diff(ours, ref, stop, n=QUICK_TOKENS):
    """the compare tool's rule on the sealed ids: the first of the first n generated tokens where ours and llama.cpp's
    differ; when the engine stopped early with every token equal, a match only if it returned a stop id or llama.cpp's
    next pick is one, else a difference at the stop"""
    m = ours[:n]
    k = next((s for s, (x, y) in enumerate(zip(m, ref)) if x != y), None)
    if k is None and len(m) < n and not (m and m[-1] in stop) and not (len(m) < len(ref) and ref[len(m)] in stop):
        k = len(m)
    return k


def _quick_events(text):
    """event name -> its JSON lines in a run's stderr"""
    out = {}
    for line in text.splitlines():
        if line.startswith('{"event"'):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            out.setdefault(x.get("event"), []).append(x)
    return out


def _quick_header_problems(g, header, work, chip, what):
    """the sealed header (the file up to its tensor data, sparse to the file's size) is a file of the row's arch, block
    types and expert count"""
    size = header.stat().st_size
    if size > g["bytes"]:
        return [f"{what}: the sealed header is {size} bytes, longer than the file's {g['bytes']}"]
    dst = Path(work or tempfile.mkdtemp(prefix="mashbox-catalog-qc-")) / f"{g['id']}-{chip}-qc" / g["file"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(header.read_bytes())
        fh.truncate(g["bytes"])
    from mashbox import gguf as G
    try:
        f = G.File(str(dst))
        have, experts = sorted(f.census()[0]), int(f.get(f"{f.arch}.expert_count") or 0)
    except (G.GGUFError, ValueError) as x:
        return [f"{what}: the sealed header does not read ({type(x).__name__}: {x})"]
    out = []
    if f.arch != g["arch"]:
        out.append(f"{what}: the sealed header is a {f.arch} file, the row says {g['arch']}")
    if have != sorted(g["types"]):
        out.append(f"{what}: the sealed header's block types are {have}, the row says {sorted(g['types'])}")
    if (experts > 0) != (g["moe"] is True):
        out.append(f"{what}: the sealed header's metadata has {experts} experts, the row says moe {g['moe']}")
    return out


QUICK_EXACT_FALLBACK = ("did not load", "have no TPU kernel", "kq planes for")   # the lines ops and gguf.plan print when
# the kernel package, or a block type of the file, runs on the exact path instead


QUICK_BF16_RE = re.compile(r"^mashbox: BF16 \(([0-9.]+)% of the weights\) have no TPU kernel")


def _quick_bf16_served(gp, w, text):
    """BF16 on the native bf16 path (Dense16) counts as served (Dense16 reads a BF16 weight as fast as XLA's own plain
    read of the same bytes, measured on a v6e chip): True only when the run's one exact-path line names BF16 alone, no
    other fallback line, its load put 0 bytes on the exact path and more than 0 on Dense16, and the plan's share is
    Dense16's bytes over the plan's weights (to its rounding: share 0.1 points, weights_gb 0.01 GB)"""
    lines = [x for x in text.splitlines() if "have no TPU kernel" in x]
    m = QUICK_BF16_RE.match(lines[0]) if len(lines) == 1 else None
    nb, gb, share = w.get("native_bf16_bytes"), gp.get("weights_gb"), gp.get("reference_share")
    if not m or any(x in text for x in QUICK_EXACT_FALLBACK if x != "have no TPU kernel") or w.get("reference_bytes") != 0:
        return False
    if (not isinstance(nb, int) or isinstance(nb, bool) or nb <= 0 or isinstance(gb, bool) or not isinstance(gb, (int, float))
            or gb <= 0 or isinstance(share, bool) or not isinstance(share, (int, float)) or float(m.group(1)) != share):
        return False
    est = 100.0 * nb / (gb * 1e9)
    return abs(share - est) <= 0.05 + est * 0.005 / gb + 1e-9


def _host_chips(acc):
    """the chips of an accelerator's host from its name ("v6e-4" -> 4), else None"""
    t = str(acc).rsplit("-", 1)[-1]
    return int(t) if t.isdigit() else None


def _quick_bundle_run_problems(g, gp, w, text, krow, chip, run, what):
    """one run on the kernel package (kernel_path bundle): its plan ran the engine_kernels row's types with 0% of the
    weights on the exact path, its weights event put 0 bytes on the exact path and more than 0 on a kernel, and it
    printed no line of a type or the package falling back to the exact path; BF16 on Dense16 counts as served
    (_quick_bf16_served)"""
    out = []
    bf16 = _quick_bf16_served(gp, w, text)
    kt = sorted(map(str, krow.get("types") or [])) if krow else None
    if gp.get("kernel_types") != kt:
        out.append(f"{what}: kernel_path bundle, and {run} ran kernel types {gp.get('kernel_types')}, the engine_kernels "
                   f"row for {chip} has {kt}")
    if gp.get("reference_share") != 0.0 and not bf16:
        out.append(f"{what}: kernel_path bundle, and {run} ran {gp.get('reference_share')}% of the weights on the exact "
                   "path")
    if w.get("file") != g["file"] or w.get("reference_bytes") != 0:
        out.append(f"{what}: kernel_path bundle, and {run} ran {w.get('reference_bytes')} bytes on the exact path "
                   f"({w.get('file')})")
    kb = w.get("kernel_bytes")
    if not isinstance(kb, int) or isinstance(kb, bool) or kb <= 0:
        out.append(f"{what}: kernel_path bundle, and {run} ran {kb} bytes on a kernel")
    said = next((x for x in QUICK_EXACT_FALLBACK if x in text), None)
    if said and not bf16:
        out.append(f"{what}: kernel_path bundle, and {run} printed '{said}' (a block type or the kernel package on the "
                   "exact path)")
    return out


def _quick_chips_problems(r, run, bundle, name, what):
    """the chips a run used: the row's chip_count, else its accelerator's host (v6e-4: 4), is the device count the run
    recorded (its observer's len(jax.devices())); a row with chip_count, or on the kernel package, needs that count"""
    want = r["chip_count"] if "chip_count" in r else _host_chips(r.get("accelerator"))
    nd = run.get("devices")
    if nd is None:
        return [f"{what}: {name} recorded no device count (the row's chip count needs the run's own)"] if (
            "chip_count" in r or bundle) else []
    if nd != want:
        return [f"{what}: the row runs on {want} chip{'s' * (want != 1)} of a TPU {r.get('accelerator')}, and {name} "
                f"saw {nd}"]
    return []


def _quick_compare_problems(cmp, runs, files, loaded, tokens, label, what):
    """(problems, (ok, full, near, other, first) or None): the compare's rows are the runs' own ids and prompt ids,
    llama.cpp ran the same prompt ids, each first difference is recomputed from the sealed ids, and the summary is the
    rows' counts (label names the runs: "" or "ngram: ")"""
    out = []
    rows = cmp.get("rows") if isinstance(cmp, dict) else None
    if not isinstance(rows, list) or [x.get("i") for x in rows] != list(range(QUICK_RUNS)):
        return [f"{what}: {label}the compare has no row per run (0-{QUICK_RUNS - 1})"], None
    ok, full, near, other, first = 0, 0, 0, 0, []
    pre = label.replace(": ", "-") if label else ""
    for i, (x, run) in enumerate(zip(rows, runs)):
        calls = run.get("generate_calls") or [{}]
        c = calls[0] if len(calls) == 1 else {}
        pid, ours, stop = (c.get("prompt_ids") or [[]])[0], (c.get("ids") or [[]])[0], set(c.get("stop_ids") or [])
        a, _, b = files[f"ref{i}.ids"].read_text().partition("\n\n")
        rp, rg = [int(t) for t in a.split()], [int(t) for t in b.split()]
        if x.get("ids_ours") != ours or x.get("ids_prompt") != pid:
            out.append(f"{what}: {label}compare row {i} is not {pre}run{i}'s ids")
        if rp != pid:
            out.append(f"{what}: {label}prompt {i}: llama.cpp ran other prompt ids than the chip")
        k = quick_check_first_diff(ours, rg, stop, tokens)
        if x.get("first_diff") != k:
            out.append(f"{what}: {label}prompt {i}: the compare says first difference {x.get('first_diff')}, the sealed "
                       f"ids give {k}")
        gap = x.get("ref_gap")
        if k == 0:
            first.append(i)
        if k is None:
            full += 1
        elif isinstance(gap, (int, float)) and not isinstance(gap, bool) and gap < NEAR_TIE_GAP:
            near += 1
        else:
            other += 1
        ok += bool(loaded[i])
    s = cmp.get("summary") or {}
    said = (s.get("runs_ok"), s.get("full"), s.get("near"), s.get("other"))
    if said != (ok, full, near, other):
        out.append(f"{what}: {label}the compare's rows give runs {ok} / full {full} / near {near} / other {other}, its "
                   f"summary says {said[0]} / {said[1]} / {said[2]} / {said[3]}")
    return out, (ok, full, near, other, first)


def _quick_fail_reason(ok, rows, forced=None):
    """the reason class of a QUICK that did not pass (catalog.QUICK_FAIL_REASONS), from the sealed compare and forced
    runs: "run" when fewer than 4 runs (or a forced run) load and print, then "first token" (an other difference at step
    0), "divergence" (an other one later, or a forced continuation with one), else "near tie" (a token-0 near tie with
    no forced run)"""
    forced = forced or {}
    if ok != QUICK_RUNS or any(not v[0] for v in forced.values()):
        return "run"
    other = [x.get("first_diff") for x in rows if x.get("first_diff") is not None and not (
        isinstance(x.get("ref_gap"), (int, float)) and not isinstance(x.get("ref_gap"), bool)
        and x["ref_gap"] < NEAR_TIE_GAP)]
    if 0 in other:
        return "first token"
    return "divergence" if other or any(not v[1] for v in forced.values()) else "near tie"


def _quick_forced_problems(cmp, runs, files, d, listed, tokens, what):
    """(problems, {prompt: (loads and prints, 0 other divergences)}) of the sealed forced runs (the token-0 near-tie
    rule: a token-0 near tie counts as a near tie, the rest checked by one forced run): each
    tf-run<i> in the seal is of a prompt whose first difference is a token-0 near tie, ran the plain run's argv with no
    MASHBOX_ variable and llama.cpp's sealed first token appended to the plain prompt ids, and tf-cmp.json's row is [that
    token] + its ids with the first difference recomputed from llama.cpp's sealed ids"""
    out, got = [], {}
    names = sorted({Path(p).name for p in listed if Path(p).parent == Path(os.path.normpath(d))
                    and Path(p).name.startswith("tf-run") and Path(p).name.endswith(".json")})
    if not names:
        return out, got
    tcmp_p = d / "tf-cmp.json"
    if os.path.normpath(tcmp_p) not in listed or not tcmp_p.is_file():
        return [f"{what}: the seal holds forced runs and no tf-cmp.json"], got
    trows = {x.get("i"): x for x in (json.loads(tcmp_p.read_text()).get("rows") or [])}
    rows = {x.get("i"): x for x in (cmp.get("rows") or [])}
    for n in names:
        i = int(n[6:-5])
        x = rows.get(i) or {}
        gap = x.get("ref_gap")
        if x.get("first_diff") != 0 or isinstance(gap, bool) or not isinstance(gap, (int, float)) or gap >= NEAR_TIE_GAP:
            out.append(f"{what}: forced run {i}: prompt {i} has no token-0 near tie")
            continue
        parts = {e: d / f"tf-run{i}{e}" for e in (".json", ".rc", ".stdout", ".stderr")}
        if any(os.path.normpath(p) not in listed or not p.is_file() for p in parts.values()):
            out.append(f"{what}: forced run {i} is not sealed whole (tf-run{i}.json/.rc/.stdout/.stderr)")
            continue
        run, plain = json.loads(parts[".json"].read_text()), runs[i]
        rg = [int(t) for t in files[f"ref{i}.ids"].read_text().partition("\n\n")[2].split()]
        argv, pargv = [str(a) for a in run.get("argv") or []], [str(a) for a in plain.get("argv") or []]
        if argv != pargv or run.get("env"):
            out.append(f"{what}: forced run {i} ran `mashbox {' '.join(argv)}` with {run.get('env')}, the plain run{i} "
                       f"`mashbox {' '.join(pargv)}`")
        if run.get("forced") != rg[0]:
            out.append(f"{what}: forced run {i} appended {run.get('forced')}, llama.cpp's first token is {rg[0]}")
        calls = run.get("generate_calls") or []
        c = calls[0] if len(calls) == 1 else {}
        p0 = ((plain.get("generate_calls") or [{}])[0].get("prompt_ids") or [[]])[0]
        if (c.get("cli_prompt_ids") or [[]])[0] != p0 or (c.get("prompt_ids") or [[]])[0] != p0 + [rg[0]]:
            out.append(f"{what}: forced run {i} ran prompt ids {(c.get('prompt_ids') or [[]])[0]}, the plain run{i}'s + "
                       f"[{rg[0]}] are {p0 + [rg[0]]}")
        ours = ([rg[0]] + (c.get("ids") or [[]])[0])[:tokens]
        k = quick_check_first_diff(ours, rg, set(c.get("stop_ids") or []), tokens)
        t = trows.get(i) or {}
        if t.get("ids_ours") != ours or t.get("first_diff") != k:
            out.append(f"{what}: forced run {i}: tf-cmp.json says first difference {t.get('first_diff')}, the sealed ids "
                       f"give {k}")
        tg = t.get("ref_gap")
        agree = k is None or (isinstance(tg, (int, float)) and not isinstance(tg, bool) and tg < NEAR_TIE_GAP)
        rc = parts[".rc"].read_text().split()[:1]
        loads = (rc == ["0"] and run.get("rc") in (0, None) and run.get("error") is None and len(calls) == 1
                 and bool(parts[".stdout"].read_text().strip()))
        got[i] = (loads, agree)
    return out, got


def quick_check_problems(g, r, e, read, root, repo, git_dir, kernels, chip, what, work=None):
    """A chip row from a quick-check run, every value read from its seal under the records root, never typed: the
    row and evidence both name the kind; the four runs ran `mashbox run REPO:TAG --prompt <sealed prompt i>
    --max-tokens 64 --temperature 0` with no MASHBOX_ variable, exited 0 and printed one generation; each run's events
    name the row's repo, file, revision, sha256 and arch, with no kernel type and no kernel byte (kernel_path exact); the
    chip is the row's accelerator and device; the dists come from the sealed pip freeze and setup log (pins_problems,
    exact path); the compare's rows are the runs' own ids, its first differences recomputed from llama.cpp's sealed ids,
    its summary the rows' counts, and passing needs 4/4 runs and 0 other divergences; the one number is recomputed from
    runs 1-3; the file, plan, card and header hold for the row; the wheel was built from the evidence's code commit,
    whose code set equals this tree's (code_problems)."""
    out = []
    kind = r.get("record_kind")
    if kind not in QUICK_KINDS or e.get("kind") != kind:
        return [f"{what}: the evidence kind is {e.get('kind')!r} (a quick-check record), and the row's record_kind is "
                f"{r.get('record_kind')!r}"]
    tokens, tool = QUICK_KINDS[kind]
    if g["moe"] is True:
        out.append(f"{what}: a mixture of experts passes only with its routed experts on the kernel; a quick-check run "
                   "is the exact path")
    bundle = r.get("kernel_path") == "bundle"
    krow = next((k for k in kernels if chip in (k.get("chips") or [])), None)
    if r.get("kernel_path") not in ("exact", "bundle") or "ctx" in r or "slots" in r:
        out.append(f"{what}: a quick-check run records no ctx or slots and runs the exact path or the catalog's kernel "
                   "package (kernel_path exact or bundle)")
    if bundle and krow is None:
        out.append(f"{what}: kernel_path bundle, and no engine_kernels row lists {chip}")
    have = {}
    for key in ("sha256sums", "compare", "file", "plan", "card", "header"):
        have[key] = read(e[key], f"{what} {key}") if key in e else out.append(f"{what}: no {key} in the evidence")
    sums = have["sha256sums"]
    if sums is None:
        return out
    listed = _sums(sums)
    out += seal_problems(sums, root, what)
    for key in ("compare", "file", "plan", "card", "header"):
        if key in e and listed.get(os.path.normpath(e[key]["path"])) != e[key]["sha256"]:
            out.append(f"{what}: {e[key]['path']} is not in {sums.name} with its sha256")
    files = {}
    need = quick_files(kind) + (("kernel-wheel.sha256",) if bundle else ())
    for n in need:
        p = sums.parent / n
        if os.path.normpath(p) not in listed:
            out.append(f"{what}: {n} is not in {sums.name}")
        elif p.is_file() and _sha(p) == listed[os.path.normpath(p)]:
            files[n] = p
    if os.path.normpath(e["compare"]["path"]) != os.path.normpath(sums.parent / "cmp.json"):
        out.append(f"{what}: the compare is not the seal's cmp.json")
    out += row_file_problems(g, have, what)
    out += code_id_problems(e, listed, read, git_dir, what)
    if have["header"] is not None:
        out += _quick_header_problems(g, have["header"], work, chip, what)
    if len(files) != len(need):
        return out
    tag, runs, loaded = f"{g['repo']}:{g['tag']}", [], []
    for i in range(QUICK_RUNS):
        try:
            run = json.loads(files[f"run{i}.json"].read_text())
        except ValueError:
            return out + [f"{what}: run{i}.json is not JSON"]
        runs.append(run)
        argv, env = [str(x) for x in run.get("argv") or []], run.get("env")
        prompt = files[f"prompt{i}.txt"].read_text()
        if argv != ["run", tag, "--prompt", prompt, "--max-tokens", str(tokens), "--temperature", "0"]:
            out.append(f"{what}: run{i} ran `mashbox {' '.join(argv)}`; a {kind} run is `mashbox run {tag} --prompt "
                       f"<prompt{i}.txt> --max-tokens {tokens} --temperature 0`")
        if env:
            out.append(f"{what}: run{i} ran with {' '.join(f'{k}={v}' for k, v in sorted(dict(env).items()))}; a "
                       "quick-check run sets no MASHBOX_ variable")
        calls = run.get("generate_calls") or []
        ids = (calls[0].get("ids") or [[]])[0] if len(calls) == 1 else []
        rc = files[f"run{i}.rc"].read_text().split()[:1]
        if (rc != ["0"] or run.get("rc") not in (0, None) or run.get("error") is not None or len(calls) != 1 or not ids
                or not files[f"run{i}.stdout"].read_text().strip()):
            out.append(f"{what}: run{i} did not load and print (exit {rc}, rc {run.get('rc')}, error "
                       f"{run.get('error')!r}, {len(calls)} generate calls, {len(ids)} ids)")
            loaded.append(False)
        else:
            loaded.append(True)
        text = files[f"run{i}.stderr"].read_text()
        ev = _quick_events(text)
        gf, gi, gp, w = (next(iter(ev.get(k) or []), {}) for k in ("gguf_file", "gguf_identity", "gguf_plan", "weights"))
        if (gf.get("repo"), gf.get("file"), gf.get("revision")) != (g["repo"], g["file"], g["revision"]):
            out.append(f"{what}: run{i} resolved {gf.get('repo')} {gf.get('file')} @ {gf.get('revision')}, the row is "
                       f"{g['repo']} {g['file']} @ {g['revision']}")
        if gi.get("sha256") != g["sha256"]:
            out.append(f"{what}: run{i} read a file of sha256 {gi.get('sha256')}, the row's is {g['sha256']}")
        if gp.get("arch") != g["arch"]:
            out.append(f"{what}: run{i} ran it as {gp.get('arch')}, the row says {g['arch']}")
        if bundle:
            out += _quick_bundle_run_problems(g, gp, w, text, krow, chip, f"run{i}", what)
        else:
            if gp.get("kernel_types") != []:
                out.append(f"{what}: kernel_path exact, and run{i} ran kernel types {gp.get('kernel_types')}")
            if w.get("file") != g["file"] or w.get("kernel_bytes") != 0:
                out.append(f"{what}: kernel_path exact, and run{i} ran {w.get('kernel_bytes')} bytes on a kernel "
                           f"({w.get('file')})")
        out += _quick_chips_problems(r, run, bundle, f"run{i}", what)
    setup = files["setup.log"].read_text()
    acc = files["accelerator-type.txt"].read_text().strip()
    if acc != r.get("accelerator"):
        out.append(f"{what}: accelerator {r.get('accelerator')}, the chip says {acc}")
    devs = QUICK_DEVICE_RE.findall(setup)
    dev = devs[-1].strip() if devs else None
    if DEVICE_CHIP.get(dev) != chip or not str(r.get("accelerator")).startswith(f"{chip}-"):
        out.append(f"{what}: the run's device is {dev}, the row is {chip} {r.get('accelerator')}")
    v = {k.lower().replace("-", "_"): x for k, x in re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)$",
                                                             files["freeze.txt"].read_text(), re.M)}
    py = re.search(r"^Using CPython (\S+)", setup, re.M)
    if py:
        v["python"] = py.group(1)
    if bundle and "kernel-wheel.sha256" in files:
        kw = files["kernel-wheel.sha256"].read_text().split()
        name = Path(kw[1]).name if len(kw) > 1 else None
        if krow is None or kw[:1] != [krow.get("sha256")] or name != krow.get("file"):
            out.append(f"{what}: kernel_path bundle, and the kernel package the runs loaded is {name} sha256 "
                       f"{kw[0] if kw else None}, the engine_kernels row for {chip} is {(krow or {}).get('file')} sha256 "
                       f"{(krow or {}).get('sha256')}")
        elif KERNELS_DIST not in v:     # the engine fetched the catalog's package (not pip-installed): its version
            v[KERNELS_DIST] = bundle_version(krow)
    out += pins_problems({"versions": v}, krow, repo, chip, what, exact=not bundle)
    built = files["CODE_SHA"].read_text().split()[:1]
    if built != [e["code"]["commit"]]:
        out.append(f"{what}: the wheel was built from {built[0] if built else None}, the evidence names "
                   f"{e['code']['commit']}")
    got_cmp = _quick_compare_problems(have["compare"], runs, files, loaded, tokens, "", what)
    out += got_cmp[0]
    forced = {}
    if kind == catalog.QUICK_KIND and got_cmp[1] is not None:
        bad, forced = _quick_forced_problems(have["compare"], runs, files, sums.parent, listed, tokens, what)
        out += bad
    if got_cmp[1] is not None:
        ok, full, near, other, first = got_cmp[1]
        first = [i for i in first if forced.get(i) != (True, True)]
        if r.get("status") == "passing" and kind == catalog.QUICK_KIND and first:
            out.append(f"{what}: a QUICK run passes only with llama.cpp's first token on every prompt, or a near tie there "
                       "whose forced run agrees on the rest; prompt "
                       f"{', '.join(map(str, first))} differ{'s' * (len(first) == 1)} at the first token")
        if r.get("status") == "passing" and (ok != QUICK_RUNS or other):
            out.append(f"{what}: passing needs {QUICK_RUNS}/{QUICK_RUNS} runs that load and print and 0 other divergences; "
                       f"the sealed runs give {ok} and {other} other divergence{'s' * (other != 1)}")
        if r.get("status") == "failing" and kind == catalog.QUICK_KIND:
            want = _quick_fail_reason(ok, have["compare"].get("rows") or [], forced)
            if r.get("reason") != want:
                out.append(f"{what}: the failing QUICK's reason is {r.get('reason')!r}, the sealed compare gives {want}")
    measured = r.get("measured") or {}
    if set(measured) - set(QUICK_MEASURED):
        out.append(f"{what}: a QUICK row's measured, when present, holds {', '.join(QUICK_MEASURED)} only; measured has "
                   f"{sorted(measured)}")
    if "decode_tok_s_1_user" in measured:      # optional; when present, the sealed runs' own number (never shown)
        try:
            got = quick_check_decode(runs)
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            out.append(f"{what}: runs 1-3 have no decode timing")
        else:
            if measured.get("decode_tok_s_1_user") != got:
                out.append(f"{what}: models.json says {measured.get('decode_tok_s_1_user')}, the sealed runs 1-3 give "
                           f"{got}")
    if {"steady_decode_tok_s_1_user", "steady_condition"} & set(measured) or "steady" in e:
        out += steady_problems(g, r, e, root, kernels, chip, what)
    if git_dir:
        out += code_problems(repo, git_dir, e["code"]["commit"], g["arch"], what, kernels, chip)
    return out


GGUF_SPEC_KEY_RE = re.compile(r"^(v\d+[a-z]*)-(\d+)$")       # the engine's chip_key: family-devices


STEADY_FILES = ("steady.json", "steady.stderr", "steady.rc", "prompt.txt", "kernel-wheel.sha256", "CODE_SHA", "host.txt")
STEADY_ARGV_TAIL = ["--max-tokens", "8", "--temperature", "0"]
HOST_TIME = "%Y-%m-%dT%H:%M:%SZ"


def _host_shared(text, what):
    """(shared, problems) from a steady record's host.txt: one "observed-from T" (when the host's process log started),
    one "window T0 T1" (the run's start and end) and one "other T0 T1 ..." line per other process that held a chip of the
    host (the log's first sighting minus its elapsed time, to its last sighting); shared when an other line overlaps the
    window. A run the log did not watch from its start, with no overlap seen, proves neither condition (a problem)."""
    seen, window, others = None, None, []
    for line in text.splitlines():
        f = line.split()
        try:
            if not f:
                continue
            if f[0] == "observed-from" and len(f) == 2:
                seen = datetime.datetime.strptime(f[1], HOST_TIME)
            elif f[0] == "window" and len(f) == 3:
                window = tuple(datetime.datetime.strptime(x, HOST_TIME) for x in f[1:])
            elif f[0] == "other" and len(f) >= 3:
                others.append(tuple(datetime.datetime.strptime(x, HOST_TIME) for x in f[1:3]))
            else:
                return None, [f"{what}: host.txt line {line!r} is not observed-from, window or other"]
        except ValueError:
            return None, [f"{what}: host.txt line {line!r} has a time that is not {HOST_TIME}"]
    if seen is None or window is None or window[0] > window[1]:
        return None, [f"{what}: host.txt needs one observed-from and one window (start no later than end)"]
    shared = any(a <= window[1] and b >= window[0] for a, b in others)
    if not shared and seen > window[0]:
        return None, [f"{what}: the host log did not watch the steady run from its start (from {seen:%H:%M:%S}, the run "
                      f"started {window[0]:%H:%M:%S}) and saw no other job: neither condition is proven"]
    return shared, []


def steady_problems(g, r, e, root, kernels, chip, what):
    """A QUICK row's steady-decode speed (measured.steady_decode_tok_s_1_user with its steady_condition), every value read
    from its own seal (the evidence's <chip>.steady.sha256sums), never typed: one `mashbox run REPO:TAG --prompt
    <prompt.txt> --max-tokens 8 --temperature 0` process on one chip that exited 0, whose engine ran one warm request and
    then STEADY_ROUNDS timed requests of STEADY_TOKENS new tokens each at 1 user, greedy, the same ids every round; its
    events name the row's repo, file, revision, sha256 and arch with every weight on the kernel package (as a kernel-path
    QUICK run), and the package it loaded is the catalog's engine_kernels row for the chip; each round's ms per new token
    is its decode_s / (new tokens - 1); the number is 1000 / their median (2 decimals) and their spread (max / min - 1) is
    at most STEADY_SPREAD; the condition is the one the sealed host log gives (_host_shared); its code commit is the
    evidence's. The steady run's code is recorded, not held to this tree: a speed is a measurement of the version it ran
    on, never a gate."""
    out, m, se = [], r.get("measured") or {}, e.get("steady")
    if not isinstance(se, dict) or not isinstance(se.get("sha256sums"), dict):
        return [f"{what}: a steady-decode speed needs its steady record in the evidence (steady.sha256sums)"]
    sums = Path(se["sha256sums"].get("path", ""))
    if not sums.is_file() or _sha(sums) != se["sha256sums"].get("sha256"):
        return [f"{what}: the steady record {sums} is not there with its sha256"]
    out += seal_problems(sums, root, what)
    listed, files = _sums(sums), {}
    for n in STEADY_FILES:
        q = sums.parent / n
        if os.path.normpath(q) not in listed:
            out.append(f"{what}: steady {n} is not in {sums}")
        elif q.is_file() and _sha(q) == listed[os.path.normpath(q)]:
            files[n] = q
    if len(files) != len(STEADY_FILES):
        return out
    try:
        st = json.loads(files["steady.json"].read_text())
    except ValueError:
        return out + [f"{what}: steady.json is not JSON"]
    tag = f"{g['repo']}:{g['tag']}"
    if st.get("model") != tag or st.get("argv_tail") != STEADY_ARGV_TAIL:
        out.append(f"{what}: the steady run ran `mashbox run {st.get('model')} --prompt <prompt.txt> "
                   f"{' '.join(map(str, st.get('argv_tail') or []))}`; a steady run is `mashbox run {tag} --prompt "
                   f"<prompt.txt> {' '.join(STEADY_ARGV_TAIL)}`")
    if st.get("devices") != r.get("chip_count", _host_chips(r.get("accelerator"))):
        out.append(f"{what}: the steady run saw {st.get('devices')} devices, the row ran on "
                   f"{r.get('chip_count', _host_chips(r.get('accelerator')))}")
    if files["steady.rc"].read_text().split() != ["rc", "0"]:
        out.append(f"{what}: the steady run did not exit 0 ({files['steady.rc'].read_text().strip()!r})")
    rounds, ms = st.get("rounds") or [], []
    try:
        for x in rounds:
            if abs(x["ms_per_token"] - 1e3 * x["decode_s"] / (x["new_tokens"] - 1)) > 1e-3:
                out.append(f"{what}: steady round {x.get('round')}'s ms_per_token {x['ms_per_token']} is not its "
                           "decode_s / (new tokens - 1)")
            ms.append(x["ms_per_token"])
        ids = {tuple(x["ids"]) for x in rounds}
        tokens = [x["new_tokens"] for x in rounds]
    except (KeyError, TypeError, ZeroDivisionError):
        return out + [f"{what}: the steady rounds lack decode_s, new_tokens, ms_per_token or ids"]
    if tokens != [catalog.STEADY_TOKENS] * catalog.STEADY_ROUNDS:
        out.append(f"{what}: a steady run times {catalog.STEADY_ROUNDS} requests of {catalog.STEADY_TOKENS} new tokens "
                   f"after its warm one; the record's rounds have {tokens}")
    if len(ids) != 1:
        out.append(f"{what}: the steady rounds did not give the same ids every round (greedy, 1 user)")
    if ms and min(ms) > 0:
        tok, spread = round(1e3 / statistics.median(ms), 2), max(ms) / min(ms) - 1
        if m.get("steady_decode_tok_s_1_user") != tok:
            out.append(f"{what}: models.json says {m.get('steady_decode_tok_s_1_user')} steady tok/s, the sealed steady "
                       f"rounds give {tok}")
        if spread > catalog.STEADY_SPREAD:
            out.append(f"{what}: the steady rounds spread {spread:.2%} (max / min ms per token), above the "
                       f"{catalog.STEADY_SPREAD:.0%} bar: no steady number")
    text = files["steady.stderr"].read_text()
    ev = _quick_events(text)
    gf, gi, gp, w = (next(iter(ev.get(k) or []), {}) for k in ("gguf_file", "gguf_identity", "gguf_plan", "weights"))
    if (gf.get("repo"), gf.get("file"), gf.get("revision")) != (g["repo"], g["file"], g["revision"]):
        out.append(f"{what}: the steady run resolved {gf.get('repo')} {gf.get('file')} @ {gf.get('revision')}, the row is "
                   f"{g['repo']} {g['file']} @ {g['revision']}")
    if gi.get("sha256") != g["sha256"]:
        out.append(f"{what}: the steady run read a file of sha256 {gi.get('sha256')}, the row's is {g['sha256']}")
    if gp.get("arch") != g["arch"]:
        out.append(f"{what}: the steady run ran it as {gp.get('arch')}, the row says {g['arch']}")
    krow = next((k for k in kernels if chip in (k.get("chips") or [])), None)
    out += _quick_bundle_run_problems(g, gp, w, text, krow, chip, "the steady run", what)
    kw = files["kernel-wheel.sha256"].read_text().split()
    name = Path(kw[1]).name if len(kw) > 1 else None
    if krow is None or kw[:1] != [krow.get("sha256")] or name != krow.get("file"):
        out.append(f"{what}: the steady run loaded the kernel package {name} sha256 {kw[0] if kw else None}, the "
                   f"engine_kernels row for {chip} is {(krow or {}).get('file')} sha256 {(krow or {}).get('sha256')}")
    if files["CODE_SHA"].read_text().split()[:1] != [(se.get("code") or {}).get("commit")]:
        out.append(f"{what}: the steady run's code is {files['CODE_SHA'].read_text().split()[:1]}, the evidence names "
                   f"{(se.get('code') or {}).get('commit')}")
    shared, bad = _host_shared(files["host.txt"].read_text(), what)
    out += bad
    if shared is not None:
        want = catalog.STEADY_CONDITIONS["shared" if shared else "alone"]
        if m.get("steady_condition") != want:
            out.append(f"{what}: steady_condition {m.get('steady_condition')!r}, the sealed host log gives {want!r}")
    return out


def quick_ngram_problems(g, e_row, read, root, kernels):
    """A gguf[] row's speculation gate entries (speculation.ngram.<engine chip key>), each read from its seal, never
    typed: the evidence names the seal of the row's own QUICK chip record on that chip family (one folder holds the
    plain runs and the n-gram runs) and its n-gram compare; the entry's record is gguf/<row id>/ngram/<chip key> with
    that seal's SHA256SUMS sha256; the four n-gram runs ran `mashbox run REPO:TAG --prompt <sealed prompt i> --speculation
    ngram --max-tokens 32 --temperature 0` with no MASHBOX_ variable, loaded and printed, read the row's file on the
    kernel package (kernel_path bundle, as the plain runs), ran the n-gram drafter at the k_table's 1-row k, on the
    entry's chip key, component set and speculation step version; the compare's rows are their ids against llama.cpp's
    sealed ids. Its status is what the runs give: passed only with the QUICK bars (4/4 load and print, llama.cpp's first
    token on every prompt, 0 other divergences), the row's plain kernel-path QUICK on that chip passing, and the n-gram
    decode (runs 1-3) at least the plain one. The plain runs decode without speculation: a plain run whose speculation
    event says used (a catalog gate turned it on) refuses the entry, whatever its status."""
    out = []
    tag = f"{g['repo']}:{g['tag']}"
    for kind, per in (g.get("speculation") or {}).items():
        for ck, ent in (per or {}).items():
            what = f"{g['id']} {kind} {ck}"
            e = (((e_row or {}).get("speculation") or {}).get(kind) or {}).get(ck)
            if e is None:
                out.append(f"{what}: no evidence entry")
                continue
            m = GGUF_SPEC_KEY_RE.match(str(ck))
            if kind != "ngram" or not m:
                out.append(f"{what}: a GGUF row's gate entries are ngram, keyed by the engine's chip key")
                continue
            fam = m.group(1)
            r, pe = (g.get("chips") or {}).get(fam) or {}, (e_row or {}).get(fam) or {}
            if any(k not in e for k in ("sha256sums", "compare")):
                out.append(f"{what}: the evidence needs sha256sums and compare")
                continue
            sums, cmp = read(e["sha256sums"], f"{what} sha256sums"), read(e["compare"], f"{what} compare")
            if sums is None:
                continue
            listed = _sums(sums)
            out += seal_problems(sums, root, what)
            if ent.get("record_sha256") != e["sha256sums"]["sha256"]:
                out.append(f"{what}: record_sha256 {ent.get('record_sha256')}, the record's SHA256SUMS is "
                           f"{e['sha256sums']['sha256']}")
            if ent.get("record") != f"gguf/{g['id']}/{kind}/{ck}":
                out.append(f"{what}: the record is {ent.get('record')}, the entry's own is gguf/{g['id']}/{kind}/{ck}")
            same_seal = (pe.get("sha256sums") or {}).get("path") == e["sha256sums"]["path"]
            if not same_seal:
                out.append(f"{what}: the n-gram runs are not sealed with the row's {fam} chip record (one folder holds "
                           "the plain and the n-gram runs)")
            if os.path.normpath(e["compare"]["path"]) != os.path.normpath(sums.parent / "ngram-cmp.json") or \
                    listed.get(os.path.normpath(e["compare"]["path"])) != e["compare"]["sha256"]:
                out.append(f"{what}: the compare is not the seal's ngram-cmp.json with its sha256")
            files = {}
            need = ([f"ngram-run{i}{x}" for i in range(QUICK_RUNS) for x in (".json", ".rc", ".stdout", ".stderr")]
                    + [f"{x}{i}{y}" for i in range(QUICK_RUNS) for x, y in (("prompt", ".txt"), ("ref", ".ids"))])
            for n in need:
                q = sums.parent / n
                if os.path.normpath(q) not in listed:
                    out.append(f"{what}: {n} is not in {sums.name}")
                elif q.is_file() and _sha(q) == listed[os.path.normpath(q)]:
                    files[n] = q
            if len(files) != len(need) or not isinstance(cmp, dict):
                continue
            krow = next((k for k in kernels if fam in (k.get("chips") or [])), None)
            kt = ent.get("k_table") or {}
            runs, loaded, dks, steps, csss = [], [], set(), set(), set()
            for i in range(QUICK_RUNS):
                name = f"ngram-run{i}"
                try:
                    run = json.loads(files[f"{name}.json"].read_text())
                except ValueError:
                    out.append(f"{what}: {name}.json is not JSON")
                    break
                runs.append(run)
                argv, env = [str(x) for x in run.get("argv") or []], run.get("env")
                prompt = files[f"prompt{i}.txt"].read_text()
                want = ["run", tag, "--prompt", prompt, "--speculation", "ngram", "--max-tokens",
                        str(catalog.QUICK_TOKENS), "--temperature", "0"]
                if argv != want:
                    out.append(f"{what}: {name} ran `mashbox {' '.join(argv)}`; an n-gram gate run is `mashbox run {tag} "
                               f"--prompt <prompt{i}.txt> --speculation ngram --max-tokens {catalog.QUICK_TOKENS} "
                               "--temperature 0`")
                if env:
                    out.append(f"{what}: {name} ran with {' '.join(f'{k}={v}' for k, v in sorted(dict(env).items()))}; "
                               "a gate run sets no MASHBOX_ variable")
                calls = run.get("generate_calls") or []
                ids = (calls[0].get("ids") or [[]])[0] if len(calls) == 1 else []
                rc = files[f"{name}.rc"].read_text().split()[:1]
                good = not (rc != ["0"] or run.get("rc") not in (0, None) or run.get("error") is not None
                            or len(calls) != 1 or not ids or not files[f"{name}.stdout"].read_text().strip())
                if not good:
                    out.append(f"{what}: {name} did not load and print (exit {rc}, rc {run.get('rc')}, error "
                               f"{run.get('error')!r}, {len(calls)} generate calls, {len(ids)} ids)")
                loaded.append(good)
                text = files[f"{name}.stderr"].read_text()
                ev = _quick_events(text)
                gi, gp, w, sp, dp = (next(iter(ev.get(k) or []), {}) for k in (
                    "gguf_identity", "gguf_plan", "weights", "speculation", "deployment"))
                if gi.get("sha256") != g["sha256"]:
                    out.append(f"{what}: {name} read a file of sha256 {gi.get('sha256')}, the row's is {g['sha256']}")
                out += _quick_bundle_run_problems(g, gp, w, text, krow, fam, name, what)
                if (sp.get("used"), sp.get("method"), sp.get("gated")) != (True, "ngram", False):
                    out.append(f"{what}: {name} ran speculation {sp.get('method')} (used {sp.get('used')}, gated "
                               f"{sp.get('gated')}), not the forced n-gram drafter")
                dks.add(sp.get("draft_k"))
                steps.add(sp.get("spec_step_version"))
                csss.add(dp.get("component_set_sha"))
                if run.get("chip_key") != ck:
                    out.append(f"{what}: {name} ran on chip key {run.get('chip_key')}, the entry is {ck}")
            if len(runs) != QUICK_RUNS:
                continue
            if dks != {kt.get("1")}:
                out.append(f"{what}: the k_table's 1-row k {kt.get('1')} is not the runs' draft_k "
                           f"{' / '.join(map(str, sorted(dks, key=str)))}")
            if steps != {ent.get("spec_step_version")}:
                out.append(f"{what}: the entry says speculation step version {ent.get('spec_step_version')}, the runs "
                           f"measured speculation step version {' / '.join(map(str, sorted(steps, key=str)))}")
            if csss != {ent.get("component_set_sha")}:
                out.append(f"{what}: the entry names component set {ent.get('component_set_sha')}, the runs' component "
                           f"set is {' / '.join(map(str, sorted(csss, key=str)))}")
            got, counts = _quick_compare_problems(cmp, runs, files, loaded, catalog.QUICK_TOKENS, "ngram: ", what)
            out += got
            if counts is None:
                continue
            ok, full, near, other, first = counts
            why = []
            if ok != QUICK_RUNS:
                why.append(f"{ok}/{QUICK_RUNS} runs load and print")
            if other:
                why.append(f"{other} other divergence{'s' * (other != 1)}")
            if first:
                why.append(f"prompt {', '.join(map(str, first))} differ{'s' * (len(first) == 1)} at the first token")
            try:        # the plain runs sealed in the same folder (same_seal), timed as the n-gram runs are
                plain = quick_check_decode([json.loads((sums.parent / f"run{i}.json").read_text())
                                            for i in range(QUICK_RUNS)])
            except (OSError, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
                plain = None
            for i in range(QUICK_RUNS):     # the plain arm decodes without speculation (a catalog gate can turn it on)
                q = sums.parent / f"run{i}.stderr"
                sp = {}
                if q.is_file() and _sha(q) == listed.get(os.path.normpath(q)):
                    sp = next(iter(_quick_events(q.read_text()).get("speculation") or []), {})
                if sp.get("used") is True:
                    out.append(f"{what}: plain run{i} ran speculation {sp.get('method')} (used True, gated "
                               f"{sp.get('gated')}); the gate compares n-gram with a plain arm that does not speculate")
            if not (r.get("status") == "passing" and r.get("kernel_path") == "bundle" and same_seal
                    and isinstance(plain, (int, float))):
                why.append(f"the row's plain kernel-path QUICK on {fam} does not pass")
            try:
                fast = quick_check_decode(runs)
            except (KeyError, IndexError, TypeError, ZeroDivisionError):
                fast = None
                why.append("n-gram runs 1-3 have no decode timing")
            if fast is not None and isinstance(plain, (int, float)) and fast < plain:
                why.append(f"n-gram {fast} tok/s below plain {plain}")
            verdict = "not_passed" if why else "passed"
            if ent.get("status") != verdict:
                out.append(f"{what}: status {ent.get('status')}, and the sealed runs give {verdict}"
                           + (f" ({'; '.join(why)})" if why else ""))
    return out


def _moe_record():
    """the ONE MoE record schema and validator, internal/moe_record.py. None where internal/ is not shipped (the export): there no MoE row can pass."""
    p = ROOT / "internal" / "moe_record.py"
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location("mashbox_internal_moe_record", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def moe_problems(rec, chip, kernels, what):
    """a passing mixture-of-experts row (never without its experts on the kernel): the record's MoE fields hold
    under internal/moe_record.py's problems(), the one validator; this file keeps no names or rules of its own"""
    mr = _moe_record()
    if mr is None:
        return [f"{what}: this tree has no MoE record validator (internal/moe_record.py), so a mixture-of-experts row "
                "cannot pass"]
    return [f"{what}: {x}" for x in mr.problems(rec, chip, kernels)]


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
    tools/gguf_fixtures builds), "code" ({path: bytes}), "rec", "cmp", "file", "plan", "code_id" (the run's sealed
    code-<tag>.json; None leaves it out, as a record from before code ids), "seal" (lines), "manifest", then "reg" and
    "ev"."""
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
    code["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    for f, data in edit("code", code).items():
        (root / "code" / f).parent.mkdir(parents=True, exist_ok=True)
        (root / "code" / f).write_bytes(data)
    commit, kq = _commit(root / "code"), "f" * 40
    tree = {k: subprocess.run(["git", "-C", str(root / "code"), "rev-parse", "--verify", "-q", f"{commit}:{p}"],
                              capture_output=True).stdout.decode().strip() or None
            for k, p in (("src_tree", "src"), ("pyproject_blob", "pyproject.toml"))}
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
             "header": _header(path),
             "code_id": {"code_sha": commit, **tree, "wheel_file": "mashbox-0.1.3-py3-none-any.whl",
                         "wheel_sha256": "a" * 64, "wheel_content_sha256": "b" * 64}}
    names = {"rec": "rec.json", "cmp": "cmp.json", "file": "file-row.sha256", "plan": "plan-row.txt",
             "attestation": "attestation.json", "manifest": "manifest.json", "card": "card-api.json", "log": "run.log",
             "header": "header-row.gguf", "code_id": "code-row.json"}
    for k, n in list(names.items()):
        if k == "manifest":         # the manifest names the attestation written just before it, by sha256
            parts[k] = dict(parts[k], attestation={"path": names["attestation"],
                                                   "sha256": _sha(root / "row" / names["attestation"])})
        v = edit(k, parts[k])
        if v is None:               # a part left out (a record from before code ids: no code-<tag>.json)
            del names[k]
            continue
        if isinstance(v, bytes):
            (root / "row" / n).write_bytes(v)
        else:
            (root / "row" / n).write_text(v if isinstance(v, str) else json.dumps(v))
    forced = edit("forced", None)
    if forced is not None:
        names["forced"] = "rec-forced.json"
        (root / "row" / names["forced"]).write_text(json.dumps(forced))
    seal = edit("seal", [f"{_sha(root / 'row' / names[k])}  {names[k]}" for k in ("rec", "cmp", "file", "plan", "card",
                                                                                   "log", "header", "forced", "code_id")
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
              "code": {"commit": commit, "wheel_content_sha256": "b" * 64}, "kernels": {"commit": kq}}}}}
    if "code_id" in names:
        ev["gguf"][row["id"]]["v6e"]["code_id"] = src("code-row.json")
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


def _moe_fields():
    """the moe fields of a passing chip record for the release bundle: internal/moe_record.py's own example, so this
    file keeps no copy of the schema; {} where internal/ is not shipped (a mixture-of-experts row then fails closed)"""
    mr = _moe_record()
    return {} if mr is None else mr.example(BUNDLE["sha256"])["moe"]


# goodput as % of the roofline for 1 and 8 users (values made up for the checks)
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
            r["versions"].pop(KERNELS_DIST)
        else:
            r["versions"][KERNELS_DIST] = package

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
                 _set("rec", lambda x: x.update(moe=_moe_fields(), goodput=copy.deepcopy(GOODPUT))))


def _moe_rec(fn):
    """an edit: the mixture-of-experts row of _moe(), with fn(moe, goodput) changing its record's moe fields or goodput"""
    def rec(r):
        fn(r["moe"], r["goodput"])      # changes the record in place; returns None so _set keeps the record
    return _both(_moe(), _set("rec", rec))


MOE_RECORD_FORMS = {"a mixture of experts whose record has no moe fields", "no routed-op call site on the kernel",
                    "a routed-op call site on the exact path", "a dev kernel build",
                    "a mixture of experts on the exact routed path",
                    "another expert kernel",
                    "an expert kernel from another package build",
                    "fewer expert-kernel calls than MoE layers",
                    "no goodput for 8 users",
                    "a goodput % its own numbers do not give",
                    "a goodput of no tokens",
                    "a goodput with no roofline",
                    "a goodput input that is not a number",
                    "a goodput entry whose users is true, not 1"}  # the forms internal/moe_record.py's problems() refuses


def test_the_catalog_guard_gives_each_moe_record_the_one_modules_verdict():
    """moe_problems is internal/moe_record.py's problems(); every record of its corpus gets the module's
    verdict here (a count that is True, a NaN goodput, a number for the kernel version: each refused)"""
    mr = _moe_record()
    if mr is None:
        pytest.skip("maintainers only: internal/moe_record.py, the MoE record validator, is not in this tree")
    kernels = [{"sha256": "b" * 64, "chips": ["v6e"]}]
    for name, rec, passes in mr.corpus("b" * 64):
        assert (moe_problems(rec, "v6e", kernels, "row") == []) == passes, name


def test_no_moe_row_passes_without_the_moe_record_module(monkeypatch):
    """where internal/ is not shipped (the export) a passing mixture-of-experts row fails closed"""
    fields = _moe_fields()
    monkeypatch.setattr(sys.modules[__name__], "_moe_record", lambda: None)
    found = moe_problems({"moe": fields, "goodput": copy.deepcopy(GOODPUT)}, "v6e",
                         [{"sha256": "b" * 64, "chips": ["v6e"]}], "row")
    assert found == ["row: this tree has no MoE record validator (internal/moe_record.py), so a mixture-of-experts row "
                     "cannot pass"]


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
    "the arch's module differs": (_code(MODS["qwen3"], lambda b: b + b"\n_guard_probe = 1\n"), "dense.py differs"),
    "a module the arch's module imports differs": (_code("src/mashbox/gguf/archs/qwen35.py", lambda b: b + b"\n_guard_probe = 1\n"),
                                                   "qwen35.py differs"),
    "a module two imports down differs": (_code("src/mashbox/models/qwen3_8_27b/model.py", lambda b: b + b"\n_guard_probe = 1\n"),
                                          "model.py differs"),
    "the reader differs": (_code("src/mashbox/gguf/reader.py", lambda b: b + b"\n_guard_probe = 1\n"), "reader.py differs"),
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
    "gguf/__init__.py differs": (_code("src/mashbox/gguf/__init__.py", lambda b: b + b"\n_guard_probe = 1\n"), "__init__.py differs"),
    "file-*.sha256 has no size line": (_set("file", lambda t: t.split("\n")[0] + "\n"), "has no size line"),
    # The MoE rule: the expert count is the file's own metadata, read from the sealed header (never a record
    # field); the routed experts ran on the kernel
    "a file with experts on a row that says dense": (_set("fixture", lambda v: "qwen35moe-pyquant"),
                                                      "the sealed header's metadata has 8 experts, the row says moe False"),
    "a row that says mixture of experts on a file with no experts": (
        _both(_moe(), _set("fixture", lambda v: FIXTURE.stem)),
        "the sealed header's metadata has 0 experts, the row says moe True"),
    "a mixture of experts whose record has no moe fields": (_both(_moe(), _set("rec", lambda r: r.__delitem__("moe"))),
                                                            "the record's moe.expert_path is None, not 'kernel'"),
    "a mixture of experts on the exact routed path": (_moe_rec(lambda m, g: m.update(expert_path="exact")),
                                                       "the record's moe.expert_path is 'exact', not 'kernel'"),
    "a routed-op call site on the exact path": (_moe_rec(lambda m, g: m["sites"].update(exact=1, exact_why=["shape"])),
                                                "routed-op call sites: 2 on the kernel, 1 on the exact path"),
    "no routed-op call site on the kernel": (_moe_rec(lambda m, g: m["sites"].update(kernel=0)),
                                             "routed-op call sites: 0 on the kernel, 0 on the exact path"),
    "another expert kernel": (_moe_rec(lambda m, g: m["kernel"].update(id="moe.other")), "the expert kernel is 'moe.other' '0.1.4' of 'mashbox_engine_kernels', not moe.expert-ffn"),
    "a dev kernel build": (_moe_rec(lambda m, g: m["kernel"].update(dev_build=True)), "the record ran a dev kernel build"),
    "an expert kernel from another package build": (_moe_rec(lambda m, g: m["kernel"].update(package_sha256="e" * 64)),
                                                    "is not a release package for this chip"),
    "fewer expert-kernel calls than MoE layers": (_moe_rec(lambda m, g: m["profile"].update(
        kernel_calls_per_decode_step=3, moe_layers=4)), "3 expert-kernel calls per profiled decode step for 4 MoE layers"),
    "no goodput for 8 users": (_moe_rec(lambda m, g: g.pop()), "the record has 0 goodput entries for 8 users (needs exactly 1)"),
    "a goodput % its own numbers do not give": (_moe_rec(lambda m, g: g[0].update(roofline_tok_s=100.0)),
                                               "goodput at users=1 is 25.0%, its tok_s / roofline_tok_s give 50.00%"),
    "a goodput of no tokens": (_moe_rec(lambda m, g: g[0].update(tok_s=0.0, pct=0.0)),
                               "goodput at users=1 needs tok_s, roofline_tok_s, pct as finite numbers > 0"),
    "a goodput with no roofline": (_moe_rec(lambda m, g: g[1].update(roofline_tok_s=0.0)),
                                   "goodput at users=8 needs tok_s, roofline_tok_s, pct as finite numbers > 0"),
    "a goodput input that is not a number": (_moe_rec(lambda m, g: g[1]["inputs"].update(f="0.1")),
                                             "goodput at users=8 needs tok_s, roofline_tok_s, pct as finite numbers > 0 and inputs (bandwidth, U, f, Tc)"),
    "a goodput entry whose users is true, not 1": (_moe_rec(lambda m, g: g[0].update(users=True)),
                                                   "the record has 0 goodput entries for 1 users (needs exactly 1)"),
    "a mixture of experts whose goodput the record does not hold": (_both(_moe(), _set("reg", lambda r: r["gguf"][0][
        "chips"]["v6e"]["measured"].__setitem__("goodput_roofline_pct_1_user", 60.0))),
        "the record's goodput[users=1].pct is 25.0"),
    # pinned versions: the record states the dists it ran; this tree's pins and the bundle's JAX and libtpu hold
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
    # kernel attestation: each type the record ran is on the attested impl of the release bundle
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
    # the row's record is the zero-flag run; its slots and context are the candidate's fit on the sealed header
    "a record that ran with --slots": (_in_place("rec", lambda r: r["invocation"]["argv"].extend(["--slots", "8"])),
                                       "the record ran `mashbox run /models/Tiny-Qwen3-Q4_K_M.gguf --slots 8`"),
    "a record that ran with a MASHBOX_ variable": (_in_place("rec", lambda r: r["invocation"]["env"].update(
        MASHBOX_GGUF_KERNELS="kernel")), "with MASHBOX_GGUF_KERNELS=kernel"),
    "a record with no invocation": (_in_place("rec", lambda r: r.pop("invocation")), "the record has no invocation"),
    "a scratch record: slots 1, ctx 1, --slots 1": (_in_place("rec", lambda r: (r.update(slots=1, ctx=1), r[
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
    # full / near / other are recomputed from the compare's rows; a settle covers exactly the others
    "a summary that hides an other": (_both(_others(1), _set("cmp", lambda c: c["summary"].update(
        near_tie_divergences=14, other_divergences=0))), "the compare's rows give full 2 / near 13 / other 1"),
    "a near tie at a gap of 0.5": (_set("cmp", lambda c: c["rows"][5].update(ref_gap=0.5)),
                                   "the compare's rows give full 2 / near 13 / other 1"),
    "a compare with fewer rows than prompts": (_set("cmp", lambda c: c.__setitem__("rows", c["rows"][:15])),
                                               "the compare has 15 rows for 16 prompts"),
    "a settle of other steps than the compare's others": (_both(_settled(2), _set("settle", lambda v: v["prompts"][0].update(
        chip_step=3))), "are not the compare's other divergences"),
    # every line of the seal is checked against the bytes under the records root
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
    # the row's repo, revision, licence, licence link and base are the pinned card's
    "the row's licence is not the card's": (_set("reg", lambda r: r["gguf"][0].__setitem__("license", "mit")),
                                            "license 'mit', the pinned card says 'apache-2.0'"),
    "the row's licence link is not the card's": (_set("reg", lambda r: r["gguf"][0].__setitem__(
        "license_link", ROW["license_link"] + ".txt")), "LICENSE.txt', the pinned card says"),
    "the row's base is not the card's": (_set("reg", lambda r: r["gguf"][0].__setitem__("base", "someone/Other")),
                                         f"base 'someone/Other', the pinned card says '{ROW['base']}'"),
    "the card is another repo's": (_set("card", lambda c: c.__setitem__("id", "someone/Other-GGUF")),
                                   "repo 'someone/Tiny-Qwen3-GGUF', the pinned card says 'someone/Other-GGUF'"),
    "the card is at another revision": (_set("card", lambda c: c.__setitem__("sha", "1" * 40)),
                                        "the pinned card says '1111111111111111111111111111111111111111'"),
    "a card that states no licence": (_set("card", lambda c: c["cardData"].__delitem__("license")),
                                      "license 'apache-2.0', the pinned card says None"),
    "no card in the evidence": (_set("ev", lambda e: e["gguf"][ROW["id"]]["v6e"].__delitem__("card")), "no card in the evidence"),
    "a card the seal does not list": (_set("seal", lambda s: [x for x in s if not x.endswith("card-api.json")]),
                                      "card-api.json is not in SHA256SUMS"),
    # the engine_kernels rows the loader reads are part of the code set
    "the engine_kernels row is not the one the record's code had": (_code("models.json", lambda b: json.dumps(
        {"engine_kernels": []}).encode()), "the engine_kernels rows for v6e differ from the code the record ran"),
}


# code set v1 as reviewed: the entry path by role
REVIEWED_CODE_SET_V1 = (
    "src/mashbox/__main__.py", "src/mashbox/__init__.py", "src/mashbox/cli.py",
    "src/mashbox/gguf/pick.py", "src/mashbox/gguf/reader.py", "src/mashbox/gguf/tokenizer.py", "src/mashbox/hub.py",
    "src/mashbox/gguf/__init__.py", "src/mashbox/gguf/archs/__init__.py",
    "src/mashbox/ops/__init__.py", "src/mashbox/ops/reference.py",
    "src/mashbox/runtime/engine.py", "src/mashbox/runtime/programs.py", "src/mashbox/runtime/compile_key.py",
    "src/mashbox/runtime/tenancy.py", "src/mashbox/runtime/sampler.py", "src/mashbox/runtime/chat.py",
    "src/mashbox/runtime/server.py")


@pytest.mark.parametrize("member", REVIEWED_CODE_SET_V1)
def test_record_guard_refuses_a_row_when_any_code_set_member_differs(tmp_path, member):
    """One code change (a new statement) per member of the code set (the `mashbox run <gguf>` entry path by role): the
    guard names it"""
    reg, ev = _fake_records(tmp_path, _code(member, lambda b: b + b"\n_guard_probe = 1\n"))
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(f"{member} differs" in x for x in found), found


@pytest.mark.parametrize("member", REVIEWED_CODE_SET_V1)
def test_record_guard_keeps_a_row_when_a_code_set_member_differs_only_in_a_comment(tmp_path, member):
    """One guard model for every record (chip rows as CPU rows): the code set is compared as code, so a comment, a
    docstring or an import line never voids a measured record; a new statement does (the test above)"""
    reg, ev = _fake_records(tmp_path, _code(member, lambda b: b + b"\n# a comment, no code\n"))
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert not any(f"{member} differs" in x for x in found), found


# The sealed equivalence (MAIN's code-set ruling, 2026-10-29 23:2xZ part 2): a code-set file that differs as code from the
# code a chip record ran stands only when equiv.py ran the record's own 16 prompt ids on the record's code (A) and on a
# later commit (C) whose file is this tree's, same file, same machine, zero flags, and every id came out the same at
# 1 user and at 8 users.
EQUIV_FILES = ("src/mashbox/cli.py", "src/mashbox/tpu/__init__.py", "src/mashbox/catalog.py")


def _show(d, commit, f):
    r = subprocess.run(["git", "-C", str(d), "show", f"{commit}:{f}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _equivalence(base, members=EQUIV_FILES, edit=None):
    """A chip row whose record ran code A (each of members with one statement more than this tree; the record's seal
    lists its ids-row.json) and a sealed equivalence of it: equiv.py's runs of A and of C (a later commit of the same
    code repo whose members are this tree's) on the record's 16 prompt ids at 1 user and 8 users, every id the same.
    edit(part, value) changes a part before it is written: "c_code" ({path: bytes} of commit C), "ids" (the
    equivalence's ids file, bytes), "a", "c" (the run jsons), "seal" (lines), then "ev". Returns (reg, ev)."""
    edit = edit or (lambda part, v: v)
    reg, ev = _fake_records(base, _set("code", lambda c: c.update(
        {m: c.get(m, (ROOT / m).read_bytes()) + b"\n_guard_probe = 1\n" for m in members})))
    root, row = Path(ev["records_root"]), reg["gguf"][0]
    e, code = ev["gguf"][row["id"]]["v6e"], Path(ev["code_repo"])
    a_commit = e["code"]["commit"]
    for f, data in edit("c_code", {m: (ROOT / m).read_bytes() for m in members}).items():
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        (code / f).write_bytes(data)
    c_commit = _commit(code)
    prompts = [f"prompt {i}" for i in range(16)]
    ids = json.dumps({"file": row["file"], "prompts": prompts, "ids": [[1, 2, 3 + i] for i in range(16)],
                      "stop_ids": [2]}).encode()
    (root / "row" / "ids-row.json").write_bytes(ids)
    with open(root / "row" / "SHA256SUMS", "a") as fh:
        fh.write(f"{_sha(root / 'row' / 'ids-row.json')}  ids-row.json\n")
    e["sha256sums"]["sha256"] = _sha(root / "row" / "SHA256SUMS")
    out = [[(i * 131 + j * 7) % 50000 for j in range(64)] for i in range(16)]

    def run(commit, tree):
        files = {f: (lambda b: None if b is None else hashlib.sha256(b).hexdigest())(_show(code, commit, f))
                 for f in EQUIV_FILES}
        return {"src": f"/equiv/tree-{tree}", "commit": commit, "file": row["file"], "file_sha256": row["sha256"],
                "N": 64, "slots": 8, "group": 8, "jax": "0.11.2", "device": "cpu", "host": "build", "env": {},
                "ready_s": 1.0, "ids_chat_equal_sealed": True, "code_files": files, "one": copy.deepcopy(out),
                "eight": copy.deepcopy(out), "utc_end": "2026-10-30T01:00:00Z"}
    q = root / "equiv"
    (q / "in").mkdir(parents=True)
    (q / "out").mkdir()
    (q / "in" / "ids-row.json").write_bytes(edit("ids", ids))
    (q / "out" / "row-A.json").write_text(json.dumps(edit("a", run(a_commit, "A"))))
    (q / "out" / "row-C.json").write_text(json.dumps(edit("c", run(c_commit, "C"))))
    names = ("in/ids-row.json", "out/row-A.json", "out/row-C.json")
    (q / "SHA256SUMS").write_text("\n".join(edit("seal", [f"{_sha(q / n)}  {n}" for n in names])) + "\n")

    def src(n):
        return {"path": str(q / n), "sha256": _sha(q / n)}
    e["equivalence"] = [{"sha256sums": src("SHA256SUMS"), "ids": src(names[0]), "a": src(names[1]),
                         "c": src(names[2])}]
    return reg, edit("ev", ev)


def _differs(found, f):
    return [x for x in found if f"{f} differs from the code the record ran" in x]


def test_record_guard_keeps_a_row_whose_differing_code_set_files_have_a_sealed_equivalence(tmp_path):
    reg, ev = _equivalence(tmp_path)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_without_an_equivalence_every_differing_code_set_file_stays_a_problem(tmp_path):
    reg, ev = _equivalence(tmp_path, edit=_in_place("ev", lambda ev: [x["v6e"].pop("equivalence")
                                                                     for x in ev["gguf"].values()]))
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert all(_differs(found, f) for f in EQUIV_FILES) and len(found) == 3, found


def _bump(key, i, j=0):
    return lambda r: r[key][i].__setitem__(j, r[key][i][j] + 1)


EQUIV_BAD = {
    "one id differs at 1 user": (_set("c", _bump("one", 3, 10)), "ids differ between run A and run C at 1 user "
                                                               "for prompts [3]"),
    "one id differs at 8 users": (_set("c", _bump("eight", 5)), "ids differ between run A and run C at 8 users "
                                                              "for prompts [5]"),
    "15 prompts": (_both(*(_in_place(t, lambda r: [r[k].pop() for k in ("one", "eight")]) for t in ("a", "c"))),
                   "run A has 15 prompts at 1 user, not 16"),
    "ids file is not the record's": (_set("ids", lambda b: b.replace(b"[1, 2, 3]", b"[1, 2, 4]")),
                                     "prompt ids are not the record's (ids-row.json)"),
    "run A is not the record's commit": (_set("a", lambda r: r.__setitem__("commit", "e" * 40)),
                                         "run A ran eeeeeeeeeeee, the record ran"),
    "a code_files hash is not its commit's": (
        _set("a", lambda r: r["code_files"].__setitem__("src/mashbox/cli.py", "0" * 64)),
        "run A's src/mashbox/cli.py is not the code of its commit"),
    "another file": (_set("c", lambda r: r.__setitem__("file_sha256", "e" * 64)),
                     "run C ran another file than the row's"),
    "env set": (_set("c", lambda r: r.__setitem__("env", {"MASHBOX_GGUF_KERNELS": "off"})),
                "run C ran with MASHBOX_GGUF_KERNELS=off"),
    "another machine": (_set("c", lambda r: r.__setitem__("host", "other")), "runs A and C differ in host"),
    "fewer than 64 tokens": (_both(*(_set(t, lambda r: r.__setitem__("N", 32)) for t in ("a", "c"))),
                             "the runs decoded 32 tokens, fewer than 64"),
    "not in the seal": (_set("seal", lambda lines: lines[:2]), "out/row-C.json is not in SHA256SUMS with its sha256"),
}


@pytest.mark.parametrize("form", sorted(EQUIV_BAD))
def test_record_guard_names_each_bad_equivalence_and_keeps_the_differing_files(tmp_path, form):
    """a bad equivalence is named, and accepts nothing: every differing code-set file stays a problem"""
    reg, ev = _equivalence(tmp_path, edit=EQUIV_BAD[form][0])
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(EQUIV_BAD[form][1] in x for x in found), found
    assert all(_differs(found, f) for f in EQUIV_FILES), found


def test_an_equivalence_accepts_a_file_only_where_this_tree_holds_the_code_it_ran(tmp_path):
    """C's cli.py is neither A's nor this tree's: cli.py stays a problem (named with C), the other two are accepted"""
    reg, ev = _equivalence(tmp_path, edit=_set("c_code", lambda c: c.__setitem__(
        "src/mashbox/cli.py", c["src/mashbox/cli.py"] + b"\n_other = 2\n")))
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert [x for x in found if "differs" in x] == _differs(found, "src/mashbox/cli.py") and len(found) == 1, found
    assert "and from the code of the equivalence's run C" in found[0], found


def test_the_code_set_is_the_reviewed_v1_and_names_only_files_of_this_tree():
    """the guard's code set is the reviewed v1, every member is a file here (one the tree lacks would drop out of the
    import walk without a word, so the guard also names it), and each arch's set holds all of them"""
    assert CODE_SET_VERSION == 1 and GGUF_CODE == REVIEWED_CODE_SET_V1
    assert [f for f in GGUF_CODE if not (ROOT / f).is_file()] == []
    assert all(set(gguf_code_files(ROOT, a)) >= set(GGUF_CODE) for a in arch_module_files())


def test_code_link_of_a_model_newer_than_the_release_pins_to_the_release_that_ships_it():
    """A model added after the last tag is not in that tag: its code link (README, tables, HF card) names its
    card_release, and the README's pin pass keeps that newer pin; links of an older release still move to the current one."""
    reg = copy.deepcopy(REG)
    reg["version"] = "0.1.3"
    new, old = SC.live(reg)[:2]
    new["card_release"], old["card_release"] = "0.1.4", "0.1.2"
    base = f"{SC.gh_base(reg)}/tree/"
    assert SC.code_url(reg, new) == f"{base}v0.1.4/{new['path']}"
    assert SC.code_url(reg, old) == f"{base}v0.1.3/{old['path']}"
    p = SC.Problems()
    readme = SC.pin_links(reg, ROOT, SC.models_table(reg, absolute=False), "README.md", p)
    assert p == []
    assert f"({base}v0.1.4/{new['path']})" in readme and f"({base}v0.1.3/{old['path']})" in readme
    moved = SC.pin_links(reg, ROOT, f"[a]({base}v0.1.2/{old['path']}) [b]({base}v0.1.4/{new['path']})", "README.md", p)
    assert moved == f"[a]({base}v0.1.3/{old['path']}) [b]({base}v0.1.4/{new['path']})" and p == []


def test_record_guard_refuses_a_record_linked_from_outside_the_records_root(tmp_path):
    """The root rule: a seal line that names a link inside the row folder to bytes outside the records root"""
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
    """tools/catalog_records_check.sh requires the records, so a
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
    """a passing MoE row needs internal/moe_record.py; without it the guard fails closed (its own test)"""
    if _moe_record() is None:
        pytest.skip("maintainers only: internal/moe_record.py, the MoE record validator, is not in this tree")
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
    if form in MOE_RECORD_FORMS and _moe_record() is None:
        pytest.skip("maintainers only: internal/moe_record.py, the MoE record validator, is not in this tree")
    reg, ev = _fake_records(tmp_path, edit)
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(words in x for x in found), found


def _card_meaning(card_link=..., card_base=..., row=None):
    """an edit: the pinned card states license_link card_link / base_model card_base (None drops the field, ... leaves
    it), and the row's fields are updated with row"""
    def card(c):
        for k, v in (("license_link", card_link), ("base_model", card_base)):
            if v is None:
                c["cardData"].pop(k, None)
            elif v is not ...:
                c["cardData"][k] = v
    edits = [_in_place("card", card), _in_place("reg", lambda r: r["gguf"][0].update(row or {}))]
    return _both(*edits)


REV0 = "0" * 40
CARD_MEANING_GOOD = {
    "a relative licence link, resolved at the pinned revision": dict(
        card_link="LICENSE", row={"license_link": f"https://huggingface.co/someone/Tiny-Qwen3-GGUF/blob/{REV0}/LICENSE"}),
    "a base model named without its org, the card repo's own": dict(card_base="Tiny-Qwen3",
                                                                    row={"base": "someone/Tiny-Qwen3"}),
    "a base model given as its HF URL": dict(card_base="https://huggingface.co/Qwen/Tiny-Qwen3", row={"base": "Qwen/Tiny-Qwen3"}),
}
CARD_MEANING_BAD = {
    "a relative licence link at another revision": (dict(card_link="LICENSE", row={
        "license_link": "https://huggingface.co/someone/Tiny-Qwen3-GGUF/blob/main/LICENSE"}), "the pinned card says 'LICENSE'"),
    "a relative licence link to another file": (dict(card_link="LICENSE.md", row={
        "license_link": f"https://huggingface.co/someone/Tiny-Qwen3-GGUF/blob/{REV0}/LICENSE"}), "the pinned card says 'LICENSE.md'"),
    "a base model of another org": (dict(card_base="Tiny-Qwen3", row={"base": "Qwen/Tiny-Qwen3"}),
                                    "base 'Qwen/Tiny-Qwen3', the pinned card says 'Tiny-Qwen3'"),
    "a base model the card and the header do not name": (dict(card_base=None, row={"base": "Qwen/Tiny-Qwen3"}),
                                                         "base 'Qwen/Tiny-Qwen3', the pinned card says None"),
}


@pytest.mark.parametrize("form", sorted(CARD_MEANING_GOOD))
def test_record_guard_compares_card_fields_by_meaning(tmp_path, form):
    """the row's licence link and base are the pinned card's by meaning, never by spelling: a relative licence link is
    a file of the card's repo at the pinned revision, a base_model with no org is a repo of the card repo's own org, an
    HF URL is its repo id. The licence itself is compared as the card states it."""
    reg, ev = _fake_records(tmp_path, _card_meaning(**CARD_MEANING_GOOD[form]))
    assert not [x for x in evidence_problems(reg, ev, ROOT, tmp_path / "fit") if "card" in x], form


@pytest.mark.parametrize("form", sorted(CARD_MEANING_BAD))
def test_record_guard_refuses_card_fields_of_another_meaning(tmp_path, form):
    kw, words = CARD_MEANING_BAD[form]
    reg, ev = _fake_records(tmp_path, _card_meaning(**kw))
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


def _ev_code(fn):
    """an edit of the row's evidence entry (its code, code_id and the rest) with fn"""
    return _in_place("ev", lambda e: fn(e["gguf"][ROW["id"]]["v6e"]))


def _no_code_id(p, v):
    """an edit: the run has no code-<tag>.json, as a record from before code ids"""
    return None if p == "code_id" else v


# a record's sealed code-<tag>.json binds it to the code commit the evidence names; which code
# counts as the same stays code set v1 (code_problems), with or without a code id
CODE_ID_FORMS = {
    "a code id of another commit": (_in_place("code_id", lambda c: c.update(code_sha="1" * 40)),
                                    "the code id names 1111111111111111111111111111111111111111, the evidence names"),
    "a code id with another src tree": (_in_place("code_id", lambda c: c.update(src_tree="2" * 40)),
                                        "the code id's src_tree is 2222222222222222222222222222222222222222, the code"),
    "a code id with another pyproject blob": (_in_place("code_id", lambda c: c.update(pyproject_blob="3" * 40)),
                                              "the code id's pyproject_blob is 3333333333333333333333333333333333333333"),
    "a code id with no src tree": (_in_place("code_id", lambda c: c.pop("src_tree")), "the code id's src_tree is None"),
    "a code commit with no pyproject.toml": (_in_place("code", lambda c: c.pop("pyproject.toml")),
                                             "the code id's pyproject_blob is None, the code"),
    "a code id with no wheel content sha256": (_in_place("code_id", lambda c: c.pop("wheel_content_sha256")),
                                               "the code id has no wheel_content_sha256"),
    "a code id whose wheel content sha256 is not one": (_in_place("code_id", lambda c: c.update(
        wheel_content_sha256="B" * 64)), "the code id has no wheel_content_sha256"),
    "a code id whose wheel content is not the evidence's": (_ev_code(lambda x: x["code"].update(
        wheel_content_sha256="c" * 64)), "the code id's wheel_content_sha256 is bbbbbbbbbbbbbbbb..., the evidence names "
                                         "cccccccccccccccc..."),
    "a code id that is not a JSON object": (_set("code_id", lambda c: [c]), "the code id is not a JSON object"),
    "a sealed code id the evidence does not cite": (_ev_code(lambda x: x.pop("code_id")),
                                                    "the seal lists code-row.json, and the evidence cites no code id"),
    "a cited code id the seal does not list": (_set("seal", lambda s: [x for x in s if "code-row" not in x]),
                                               "code-row.json is not in the seal with its sha256"),
    "a cited code id with another sha256": (_ev_code(lambda x: x["code_id"].update(sha256="0" * 64)),
                                            "code id: "),
    "two code ids in the seal": (_set("seal", lambda s: s + [x.replace("code-row.json", "code-other.json")
                                                             for x in s if "code-row" in x]),
                                 "the seal lists 2 code ids"),
}


@pytest.mark.parametrize("form", sorted(CODE_ID_FORMS))
def test_record_guard_names_each_bad_code_id(tmp_path, form):
    """each way a record's sealed code id can fail to bind it to the evidence's code commit is caught, by its message"""
    edit, words = CODE_ID_FORMS[form]
    reg, ev = _fake_records(tmp_path, edit)
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(words in x for x in found), found


def test_record_guard_holds_a_record_from_before_code_ids_on_the_commit_path(tmp_path):
    """a record whose seal has no code-<tag>.json keeps the commit path: it holds when its code is this tree's code
    set, and a code set member that differs still fails it"""
    reg, ev = _fake_records(tmp_path, _no_code_id)
    assert "code_id" not in ev["gguf"][ROW["id"]]["v6e"]
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    reg, ev = _fake_records(tmp_path / "b", _both(_no_code_id, _code(MODS["qwen3"], lambda b: b + b"\n_guard_probe = 1\n")))
    assert any("dense.py differs" in x for x in evidence_problems(reg, ev, ROOT, tmp_path / "fit"))


def test_a_code_id_does_not_replace_the_code_set(tmp_path):
    """a sealed code id that binds the record to its commit does not stand in for the code set: the packaged catalog
    (src/mashbox/models.json) is under src/, so the src tree differs with every catalog row, and a code set member that
    differs fails the row whatever its code id says"""
    reg, ev = _fake_records(tmp_path, _code(MODS["qwen3"], lambda b: b + b"\n_guard_probe = 1\n"))
    assert "code_id" in ev["gguf"][ROW["id"]]["v6e"]
    found = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any("dense.py differs" in x for x in found) and not any("code id" in x for x in found), found


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


def test_a_card_names_a_hugging_face_tag_only_when_the_tag_points_at_the_pinned_revision():
    """uploading a card makes a new revision that no tag points at, and a tag is never moved (a move deletes it first), so
    the card names the weights by file and sha256, never by tag, and check --online refuses a card line that names a tag
    of its repo pointing at any other revision; hf_tag (the dated weights version) must point at a revision whose weights
    file is the one models.json pins, which a card-only revision after it keeps"""
    m = copy.deepcopy(next(x for x in REG["models"] if x["id"] == "track2rig"))
    reg = dict(copy.deepcopy(REG), models=[m])
    rev, old, tag = m["hf_revision"], "9" * 40, m["hf_tag"]
    w = next(f for f in m["files"] if f["kind"] == "weights" and f.get("default"))
    card = SC.render_card(reg, m, ROOT / "templates" / "hf", ROOT)
    assert tag not in card and f"`{w['name']}` has sha256 `{w['sha256']}`" in card
    base = f"https://huggingface.co/api/models/{m['hf_repo']}"

    class Fetch:
        def __init__(self, card, tag_rev, tag_weights):
            self.card, self.tag_rev, self.tag_weights = card, tag_rev, tag_weights

        def get(self, url):
            if url == base:
                return 200, json.dumps({"sha": rev})
            if url == base + "/refs":
                return 200, json.dumps({"tags": [{"name": tag, "targetCommit": "4" * 40}],
                                        "branches": [{"name": "main", "targetCommit": rev}]})
            if url.startswith(base + "/revision/"):
                r = url[len(base + "/revision/"):].split("?")[0]
                r = self.tag_rev if r == tag else r
                sha = self.tag_weights if r == self.tag_rev else w["sha256"]
                return 200, json.dumps({"sha": r, "siblings": [
                    {"rfilename": f["name"], "lfs": {"sha256": sha if f is w else f.get("sha256")}} for f in m["files"]]})
            if url == f"https://huggingface.co/{m['hf_repo']}/raw/{rev}/README.md":
                return 200, self.card
            return 404, ""

        def status(self, url):      # the card's links are checked too; not this test's subject
            return 200

    def found(card, tag_rev, tag_weights=w["sha256"]):
        p = SC.Problems()
        SC.check_online(reg, {f"hf/{m['id']}/README.md": card}, Fetch(card, tag_rev, tag_weights), p)
        return [x for x in p if x.startswith(f"hf {m['hf_repo']}: ") and ("Hugging Face tag" in x or "hf_tag" in x)]
    named = card.replace(f"`{w['name']}` has sha256", f"`{w['name']}` (weights version `{tag}`) has sha256")
    line = next(i + 1 for i, x in enumerate(named.splitlines()) if tag in x)
    assert found(named, old) == [
        f"hf {m['hf_repo']}: the card at {rev[:12]} line {line} names the Hugging Face tag {tag}, which points at "
        f"{old[:12]}, not at the pinned revision {rev[:12]}: a card names no tag, or only a tag of the revision it is on"]
    assert found(named, rev) == []                      # the tag points at the pinned revision: true, allowed
    assert found(card, old) == []                       # the card names no tag: the tag may point at an older revision
    assert found(card, old, "0" * 64) == [              # ... but only at one with the same weights file
        f"hf {m['hf_repo']}: hf_tag {tag}: {w['name']} at tag {tag} has sha256 {'0' * 12}, models.json pins "
        f"{w['sha256'][:12]}"]


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


def test_release_accepts_pending_on_a_released_row_only_with_its_not_run_reason():
    """A chip never run is not a finding, so it is never marked "not supported": under --release a released row may
    say "pending" for a chip it has not run on when the chip entry carries an explicit not_run_reason; the pages print
    "Not tested yet" for it, never "Not supported". "pending" with no not_run_reason (an empty one, or only the
    measured-failure "reason" field) still fails, and so do "failing" and a missing status. not_run_reason sits only on
    a pending chip."""
    base = next(m for m in REG["models"] if m["kind"] != "llm-gguf" and not m.get("launch"))
    chip = SC.CHIPS[0]

    def row_problems(entry, release=True):
        reg = copy.deepcopy(REG)
        row = next(r for r in reg["models"] if r["id"] == base["id"])
        if entry is None:
            row["chips"].pop(chip, None)
        else:
            row["chips"][chip] = entry
        p = SC.Problems()
        SC.validate(reg, ROOT, p, release=release)
        return [x for x in p if re.match(rf"models\[\d+\] {re.escape(base['id'])}: ", x)
                and ("every chip must be passing" in x or "not_run_reason" in x)]

    why = "CPU model: runs on the CPU, no TPU needed"
    assert row_problems({"status": "pending", "not_run_reason": why}) == []
    assert row_problems({"status": "pending", "not_run_reason": why}, release=False) == []
    for bad in ({"status": "pending"}, {"status": "pending", "not_run_reason": "  "},
                {"status": "pending", "reason": why}, {"status": "failing", "date": "2026-10-29", "reason": "x y"}, None):
        assert any("every chip must be passing" in x for x in row_problems(bad)), bad
    for st in ({"status": "passing", "date": "2026-10-29"}, {"status": "not supported", "reason": "x y"},
               {"status": "failing", "date": "2026-10-29", "reason": "x y"}):
        assert any("not_run_reason" in x for x in row_problems({**st, "not_run_reason": why}, release=False)), st
    assert any("not_run_reason" in x for x in row_problems({"status": "pending", "not_run_reason": " "}, release=False))

    m = copy.deepcopy(base)
    m["chips"][chip] = {"status": "pending", "not_run_reason": why}
    assert SC.chip_cell(m, chip) == "Not tested yet"
    assert SC.chip_text(m, chip) == "Not tested yet"
    assert [r for r in SC.chips_table(m).splitlines() if SC.CHIP_LABEL[chip] in r and "Not tested yet" in r]
    assert "Not supported" not in SC.chip_text(m, chip) + SC.chips_table(m).split(SC.CHIP_LABEL[chip], 1)[1].split("\n", 1)[0]
    m["chips"][chip]["note"] = "Runs on the host CPU"
    assert SC.chip_text(m, chip) == "Not tested yet"


def _row_problems(model_id, chip, entry, release=False, results=None, tag=None):
    """validate() problems of one model after replacing chips.<chip> (and optionally its results), filtered to tag"""
    reg = copy.deepcopy(REG)
    row = next(r for r in reg["models"] if r["id"] == model_id)
    if entry is not None:
        row["chips"][chip] = entry
    if results is not None:
        row["results"] = results
    p = SC.Problems()
    SC.validate(reg, ROOT, p, release=release)
    return [x for x in p if re.match(rf"models\[\d+\] {re.escape(model_id)}: ", x) and (tag is None or tag in x)]


def test_chip_and_result_dates_are_real_calendar_dates_never_in_the_future():
    """DATE_RE was redefined without anchors further down the module, and the later definition is the
    one validate() used, so '9999-99-99', '0000-00-00' and '2026-10-29 (never run)' passed as a run's date. A chips
    row's date and a result's date are a real YYYY-MM-DD, never after tomorrow (UTC)."""
    assert SC.DATE_RE.pattern.startswith("^") and SC.DATE_RE.match("2026-10-28 (never run)") is None
    assert SC._dated({"x": "measured 2026-10-28 on a v6e-1"})      # _dated still finds a date inside a string
    hy = next(m for m in REG["models"] if m["id"] == "hy-nomas")
    good = {**hy["chips"]["v6e-4"]}
    assert _row_problems("hy-nomas", "v6e-4", good, tag="date") == []
    for bad in ("9999-99-99", "0000-00-00", "2026-02-30", "2099-01-01", "2026-10-29 (never run)", "26-10-29", ""):
        assert _row_problems("hy-nomas", "v6e-4", {**good, "date": bad}, tag="date"), bad
        res = copy.deepcopy(hy["results"])
        res[0]["date"] = bad or "x"
        assert _row_problems("hy-nomas", "v6e-4", None, results=res, tag="date"), bad


NEVER_RUN_PARAPHRASES = ("runs on TPU v6e only (one chip or a v6e-4)", "only v6e", "needs a TPU v6e", "requires a v6e-4",
                         "use a v6e-1 instead", "v6e only", "fits on v6e", "TPU v6e required", "works on the v6e-4 slice",
                         "supported on v6e", "v6e-1 or v6e-4", "Runs on TPU v6e only: use a v6e-1")


def test_a_chip_row_speaks_for_its_own_chip_and_a_never_run_row_reads_not_tested_yet():
    """A chips row's reason and note never name another chip type (the old 'runs on TPU v6e only' on v5e, and every
    paraphrase of it, as reason or note); a reason never says the chip was not tested; a row nobody ran (not passing or
    failing, no date, no result on its chip) reads 'Not tested yet' and carries no reason, and it is stored as
    "pending" with its not_run_reason: 'not supported' + note 'Not tested yet' prints the same words but is
    reported by the status rule only. Today's rows pass, in both modes."""
    tag = lambda x: "another chip" in x or "never run" in x or "not tested" in x
    for release in (False, True):
        for m in REG["models"]:
            reg_p = _row_problems(m["id"], None, None, release=release)
            assert [x for x in reg_p if tag(x)] == [], (m["id"], release)
    nty = {"status": "not supported", "note": "Not tested yet"}
    for ok in (nty, {"status": "pending", "not_run_reason": "CPU model: runs on the CPU, no TPU needed"}):
        got = [x for x in _row_problems("hy-nomas", "v5e", ok) if tag(x)]
        assert [x for x in got if NOT_SUPPORTED_STATUS not in x] == [], ok
        assert bool(got) == (ok is nty), (ok, got)   # a chip nobody ran is pending, never "not supported"
        m = copy.deepcopy(next(r for r in REG["models"] if r["id"] == "hy-nomas"))
        m["chips"]["v5e"] = ok
        assert SC.chip_text(m, "v5e") == "Not tested yet"
    for release in (False, True):
        assert any(tag(x) for x in _row_problems("hy-nomas", "v5e", {"status": "not supported"}, release=release))
        for said in NEVER_RUN_PARAPHRASES + ("Not tested on this chip yet.",):
            for entry in ({"status": "not supported", "reason": said}, {"status": "not supported", "note": said},
                          {**nty, "reason": said}):
                assert any(tag(x) for x in _row_problems("hy-nomas", "v5e", entry, release=release)), (entry, release)
    ran = {"status": "not supported", "date": "2026-10-28"}
    assert [x for x in _row_problems("hy-nomas", "v5e", {**ran, "reason": "too little memory: 16 GB of HBM per chip"})
            if tag(x)] == []
    for said in ("needs a TPU v6e", "use a v6e-1 instead", "Not tested on this chip yet.", "never run on this chip"):
        assert any(tag(x) for x in _row_problems("hy-nomas", "v5e", {**ran, "reason": said})), said
    qv5e = next(r for r in REG["models"] if r["id"] == "qwen3-8-27b")["chips"]["v5e"]
    assert any("another chip" in x for x in _row_problems("qwen3-8-27b", "v5e", {**qv5e, "note": "faster on a v6e-1"}))


def test_runs_only_on_counts_a_never_run_not_supported_row_like_pending():
    """A gap in only_chip_claims: a chip nobody ran may be stored as 'not supported' + note 'Not tested yet',
    not as 'pending', so 'runs only on a TPU v6e-4' passed while v5e and v5p-8 were never run."""
    m = copy.deepcopy(next(r for r in REG["models"] if r["id"] == "hy-nomas"))
    m["results"] = [r for r in m["results"] if "v6e-4" in r["chip"]]
    m["chips"]["v6e"] = {"status": "not supported", "note": "Not tested yet"}
    for c in ("v5e", "v5p-8"):
        m["chips"][c] = {"status": "not supported", "note": "Not tested yet"}
    got = SC.only_chip_claims(m, "Hy-Nomas runs only on a TPU v6e-4.")
    assert got and all(c in got[0][1] for c in ("v5e", "v5p-8", "v6e ")), got
    m["chips"]["v5e"] = {"status": "not supported", "date": "2026-10-28", "reason": "too little memory per chip"}
    m["chips"]["v5p-8"] = {"status": "failing", "date": "2026-10-28", "reason": "x y"}
    m["chips"]["v6e"] = {"status": "failing", "date": "2026-10-28", "reason": "x y"}
    assert SC.only_chip_claims(m, "Hy-Nomas runs only on a TPU v6e-4.") == []


MEASURED_DOES_NOT_RUN = ("the full pipeline does not run on a 16 GB chip: out of memory at load",
                         "the compiled kernels do not run on this chip (kernel lowering error)",
                         "XLA would not run the decoder: 16 GB of HBM is too little",
                         "it will never run in 16 GB: the weights alone are 27 GB",
                         "cannot run: the weights alone are 27 GB and a chip has 16 GB")
SAYS_NOT_TESTED = ("Not tested on this chip yet.", "never run on this chip", "not yet run", "has not been run",
                   "wasn't run", "untested", "never been tested", "not run yet", "nobody ran it")


def test_a_measured_reason_may_say_the_model_does_not_run_there_but_never_that_nobody_ran_it():
    """The not-tested words caught every 'not run' and 'never run', so a dated row's measured
    finding ('does not run on a 16 GB chip: out of memory at load', 'will never run in 16 GB') failed as if nobody
    had run the chip. A reason may state that the model does not run there; it never says the chip was not run."""
    for status in ("not supported", "failing"):
        ran = {"status": status, "date": "2026-10-28"}
        for said in MEASURED_DOES_NOT_RUN:
            assert _row_problems("hy-nomas", "v5e", {**ran, "reason": said}) == [], (status, said)
        for said in SAYS_NOT_TESTED:
            assert _row_problems("hy-nomas", "v5e", {**ran, "reason": said}, tag="not tested"), (status, said)


def test_a_never_run_row_cannot_hide_behind_a_junk_date_or_carry_a_reason():
    """never_run() took any date as a run, so a 'not supported' row nobody ran passed with
    'date': 'never' or '9999-99-99' and its reason rendered as a finding ('Not supported: too little memory'). Any
    chips row's date is a real date; a row without one is never run; a never-run row carries no reason even when its
    note says 'Not tested yet'; a passing or failing row is a run whatever its date says."""
    for junk in ("never", "9999-99-99", "2026-02-30", "2026-10-28 (never run)"):
        for entry in ({"status": "not supported", "date": junk},
                      {"status": "not supported", "date": junk, "reason": "too little memory"}):
            for release in (False, True):
                assert _row_problems("hy-nomas", "v5e", entry, release=release, tag="never run"), (entry, release)
                assert _row_problems("hy-nomas", "v5e", entry, release=release, tag="chips.v5e.date"), (entry, release)
    entry = {"status": "not supported", "note": "Not tested yet", "reason": "too little memory per chip"}
    assert _row_problems("hy-nomas", "v5e", entry, tag="carries no reason"), entry
    m = copy.deepcopy(next(r for r in REG["models"] if r["id"] == "hy-nomas"))
    m["results"] = [r for r in m["results"] if "v6e-4" in r["chip"]]
    for status in ("passing", "failing"):
        m["chips"]["v5e"] = {"status": status}
        assert not SC.never_run(m, "v5e"), status
    today = datetime.datetime.now(datetime.timezone.utc).date()
    assert SC.real_date(str(today + datetime.timedelta(days=1)))
    assert not SC.real_date(str(today + datetime.timedelta(days=2)))
    for bad in ("2026-10-28\n", " 2026-10-28", "2026-10-28 "):
        assert not SC.real_date(bad), repr(bad)


def test_a_chip_row_accelerator_is_its_own_chip_type_and_chip_names_are_read_in_any_case():
    """A never-run v5e row with 'accelerator': 'v6e-4' rendered as 'TPU v5e (v6e-4)' and made
    only_chip_claims skip it, so 'runs only on a TPU v6e-4' passed again; and 'Runs on TPU V6E only' in upper case was
    not read as a chip name. A row's accelerator is a host of its own chip type; chip names are read in any case."""
    nty = {"status": "not supported", "note": "Not tested yet"}
    for release in (False, True):
        assert _row_problems("hy-nomas", "v5e", {**nty, "accelerator": "v6e-4"}, release=release, tag="another chip")
        assert _row_problems("hy-nomas", "v5e", {**nty, "accelerator": "v5e-4"}, release=release, tag="another chip") == []
        for said in ("Runs on TPU V6E only", "needs a V6E-4"):
            for key in ("reason", "note"):
                assert _row_problems("hy-nomas", "v5e", {"status": "not supported", "date": "2026-10-28", key: said},
                                     release=release, tag="another chip"), (key, said, release)
    m = copy.deepcopy(next(r for r in REG["models"] if r["id"] == "hy-nomas"))
    m["results"] = [r for r in m["results"] if "v6e-4" in r["chip"]]
    m["chips"]["v6e"] = {"status": "not supported", "note": "Not tested yet"}
    m["chips"]["v5p-8"] = {"status": "not supported", "note": "Not tested yet"}
    m["chips"]["v5e"] = {**nty, "accelerator": "v6e-4"}
    got = SC.only_chip_claims(m, "Hy-Nomas runs only on a TPU v6e-4.")
    assert got and "its v5e chips row was never run" in got[0][1], got



def _measured_run(m, chip):
    """Why a measured run of m exists on chips.<chip>, "" when none: a dated results row on that chip, or the row's own
    measured block"""
    if chip in SC.chip_rows_measured(m):
        return "a dated results row on it"
    if (m["chips"].get(chip) or {}).get("measured"):
        return "its measured block"
    return ""


def _chips_named(text):
    """The chips rows a results row's chip names: exactly (v6e-4, v5p-8) or by type (v6e-1 -> v6e)."""
    return {t if t in SC.CHIPS else t.split("-")[0] for t in SC.CHIP_TOKEN_RE.findall(str(text or "").lower())}


NOT_SUPPORTED_STATUS = 'status "not supported"'   # the validator's words for both rules below
MEASURED_PENDING = 'never reads "Not tested yet"'   # the validator's words for a pending row on a chip with a measured run


def test_no_chip_a_model_ran_on_says_not_supported_and_a_chip_nobody_ran_is_pending_with_its_reason():
    """The rule: a chip a model ran on is never marked "not supported", and a chip nobody ran reads "Not tested yet".
    The pages followed it, but the raw models.json (linked from llms.txt and shipped in the package) still
    had "status": "not supported" on a chips row with a measured run on its chip and on rows nobody ran, and no check
    read the raw field. In both copies of models.json: a chips row with a measured run on its chip (a dated results row
    on it, or its measured block) never has status "not supported" or "pending"; a chips row nobody ran is "pending"
    with a non-empty not_run_reason; a GGUF row's chip (each one has its record) is never "not supported"."""
    bad = []
    for path in (ROOT / "models.json", ROOT / "src" / "mashbox" / "models.json"):
        reg = json.loads(path.read_text())
        where = path.relative_to(ROOT).as_posix()
        for m in reg["models"]:
            for chip, c in (m.get("chips") or {}).items():
                st = (c or {}).get("status")
                if st in ("not supported", "pending") and _measured_run(m, chip):
                    bad.append(f"{where} {m['id']} chips.{chip}: {_measured_run(m, chip)}, but status {st!r}")
                if SC.never_run(m, chip) and not (st == "pending" and str(c.get("not_run_reason") or "").strip()):
                    bad.append(f"{where} {m['id']} chips.{chip}: never run, but status {st!r} and not pending with "
                               "its not_run_reason")
        for g in reg.get("gguf") or []:
            for chip, c in (g.get("chips") or {}).items():
                if (c or {}).get("status") == "not supported":
                    bad.append(f"{where} gguf {g.get('id')} chips.{chip}: {NOT_SUPPORTED_STATUS}")
    assert bad == [], bad


def test_validate_reports_not_supported_on_a_chip_with_a_measured_run_or_on_a_chip_nobody_ran():
    """The release gate reads models.json through validate(), so it reports both rules of the test above in every mode,
    for every model and chip: a chips row nobody ran that says "not supported" (with or without the note 'Not tested
    yet'), and a chips row with a measured run on its chip that says "not supported" (dated, with a reason, or not) or
    "pending" (its pages would print 'Not tested yet'). A pending row with its not_run_reason on a chip nobody ran, and a
    row with a measured finding on a chip with no measured run on it, pass."""
    nty = {"status": "not supported", "note": "Not tested yet"}
    pend = {"status": "pending", "not_run_reason": "Outside the chip types this release was tested on"}
    finding = {"status": "not supported", "date": "2026-10-28", "reason": "too little memory: 16 GB of HBM per chip"}
    checked = 0
    for release in (False, True):
        for m in REG["models"]:
            if m["kind"] == "llm-gguf" or m.get("launch"):
                continue
            for chip in SC.CHIPS:
                dated = {**(m["results"] or [{}])[0], "chip": f"TPU {chip}", "date": "2026-10-28"}
                for results in (None, copy.deepcopy(m["results"]) + [dated]):
                    probe = copy.deepcopy(m)
                    if results is not None:
                        probe["results"] = results
                    for entry in (nty, {"status": "not supported"}, finding):
                        probe["chips"][chip] = entry
                        want = bool(_measured_run(probe, chip)) or SC.never_run(probe, chip)
                        got = _row_problems(m["id"], chip, entry, release=release, results=results,
                                            tag=NOT_SUPPORTED_STATUS)
                        assert bool(got) == want, (m["id"], chip, entry, results is not None, release, got)
                        checked += 1
                    probe["chips"][chip] = pend
                    got = _row_problems(m["id"], chip, pend, release=release, results=results, tag=MEASURED_PENDING)
                    assert bool(got) == bool(_measured_run(probe, chip)), (m["id"], chip, release, got)
                    if SC.never_run(probe, chip):
                        assert _row_problems(m["id"], chip, pend, release=release, results=results,
                                             tag=NOT_SUPPORTED_STATUS) == [], (m["id"], chip, release)
    assert checked >= 2 * 3 * len(SC.CHIPS) * 2 * 3
    finding_words = "states what a run found there"
    for m in (m for m in REG["models"] if m["kind"] != "llm-gguf" and not m.get("launch")):
        for chip in SC.CHIPS:
            for entry in ({"status": "not supported", "date": "2026-10-28"},
                          {"status": "not supported", "date": "2026-10-28", "note": "Not tested yet"},
                          {"status": "not supported", "reason": "too little memory: 16 GB of HBM per chip"}):
                assert _row_problems(m["id"], chip, entry, release=True, tag=finding_words), (m["id"], chip, entry)
            assert _row_problems(m["id"], chip, finding, release=True, tag=finding_words) == [], (m["id"], chip)


def test_every_chip_a_model_ran_on_is_in_its_runs_on_line_with_its_run_note():
    """llms.txt listed Hy-Nomas as 'Runs on: TPU v6e slice (v6e-4)' while it has a dated result on one TPU v6e chip, and
    agents read llms.txt to pick a chip. Every passing chip, and every chip a live model has a measured run on that is
    not failing, is named in its llms.txt 'Runs on'. A run_note (the measured speed and its condition, in a few words)
    prints after its chip there; a passing row's run_note also follows "Supported" in the README chips cell. A
    run_note's measured numbers (a number with its unit) are values of a dated results row on its chip; run_note sits
    only on a passing or measured row and speaks for its own chip."""
    llms = (ROOT / "llms.txt").read_text().splitlines()
    readme = (ROOT / "README.md").read_text()
    for m in SC.live(REG):
        line = next(x for x in llms if x.startswith(f"- [{m['name']}]("))
        runs = line.split(" Runs on: ", 1)[1] if " Runs on: " in line else ""
        for chip in SC.CHIPS:
            c = m["chips"].get(chip) or {}
            if c.get("status") == "passing" or (_measured_run(m, chip) and c.get("status") != "failing"):
                assert SC.chip_label(m, chip) in runs, (m["id"], chip, line)
            if c.get("run_note"):
                assert f"{SC.chip_label(m, chip)}: {c['run_note']}" in runs, (m["id"], chip, line)
            if c.get("run_note") and c.get("status") == "passing":
                assert f"{SC.chip_label(m, chip)}: Supported, {c['run_note']}" in readme, (m["id"], chip)
    m0 = next(m for m in REG["models"] if m["kind"] != "llm-gguf" and not m.get("launch") and m["results"])
    chip = next(c for c in SC.CHIPS if c in SC.chip_rows_measured(m0))
    value = next(r["value"] for r in m0["results"] if chip in {t if t in SC.CHIPS else t.split("-")[0]
                                                               for t in SC.CHIP_TOKEN_RE.findall(r["chip"].lower())})
    num = re.search(r"\d+(?:\.\d+)?", value).group(0)
    ok = {"status": "passing", "date": "2026-10-28", "run_note": f"{num} s for one stream"}
    assert _row_problems(m0["id"], chip, ok, tag="run_note") == []
    for bad in ({**ok, "run_note": "123456.789 s for one stream"}, {**ok, "run_note": "  "},
                {**ok, "run_note": "999 seconds for one stream"}, {**ok, "run_note": "61 fps"},
                {**ok, "run_note": "10 hours"}, {**ok, "run_note": "5× faster"},
                {"status": "pending", "not_run_reason": "x y", "run_note": ok["run_note"]},
                {**ok, "run_note": f"{num} s, faster on a v5p-8"}):
        assert _row_problems(m0["id"], chip, bad, tag="run_note"), bad
    m = copy.deepcopy(m0)
    m["chips"][chip] = ok
    assert f"{SC.chip_label(m, chip)}: {ok['run_note']}" in SC.runs_on(m)
    assert f"{SC.chip_label(m, chip)}: Supported, {ok['run_note']}" in SC.models_table({**REG, "models": [m]}, False)


MEASURED_ROW_PAGES = ("README", "chips table", "chip text", "chips block", "llms.txt")


def test_a_measured_chip_never_reads_not_supported_and_the_package_never_offers_it():
    """'measured' says what is true for a chip a model has a dated measured run on while this version's `mashbox run`
    does not start there (a passing row would make `mashbox models`, the app and `tpu up` offer the chip). It needs a
    real date, a dated results row on its chip, a public note and a run_note. It never renders as 'Not supported' on
    any page, raw file or the package catalog; the pages print its note and llms.txt names the chip with its run_note;
    the package's own chip lists (catalog, app shell, `tpu up`) never name it."""
    from mashbox import tpu as T
    from mashbox.ui import shell as US
    m0 = next(m for m in REG["models"] if m["kind"] != "llm-gguf" and not m.get("launch") and m["results"])
    chip = next(c for c in SC.CHIPS if c in SC.chip_rows_measured(m0))
    num = re.search(r"\d+(?:\.\d+)?", next(r["value"] for r in m0["results"] if chip in _chips_named(r["chip"]))).group(0)
    row = {"status": "measured", "date": "2026-10-28", "accelerator": f"{chip}-1" if "-" not in chip else chip,
           "note": "One chip: the model fits. This version of `mashbox run` starts on another host",
           "run_note": f"{num} s per unit of output (this version of `mashbox run` starts on another host)"}
    assert _row_problems(m0["id"], chip, row, release=True) == []
    assert _row_problems(m0["id"], chip, {**row, "run_note": "fits on one chip (this version of `mashbox run` starts "
                                                             "on another host)"}, release=True)
    for key in ("date", "note", "run_note"):
        assert _row_problems(m0["id"], chip, {k: v for k, v in row.items() if k != key}, release=True), key
    no_result = [r for r in m0["results"] if chip not in _chips_named(r["chip"])]
    assert any("measured" in x for x in _row_problems(m0["id"], chip, row, results=no_result)), "needs a dated result"
    m = copy.deepcopy(m0)
    for other in SC.CHIPS:   # every other chip passing, so any "not supported" or "not tested" left comes from this row
        m["chips"][other] = {"status": "passing", "date": "2026-10-28"}
    m["chips"][chip] = row
    reg = {**REG, "models": [m]}
    seen = {"README": SC.models_table(reg, False), "chips table": SC.chips_table(m), "chip text": SC.chip_text(m, chip),
            "chips block": SC.chips_block(reg), "llms.txt": SC.llms_block(reg)}
    assert set(seen) == set(MEASURED_ROW_PAGES)
    for page, text in seen.items():
        assert row["note"].split(": ", 1)[-1] in text or row["run_note"] in text, page   # chips table: "One chip | rest"
        assert "not supported" not in text.lower() and "not tested" not in text.lower(), (page, text)
    assert SC.chip_text(m, chip) == row["note"]
    name, acc = m["name"], row["accelerator"]
    for said, flagged in ((f"{name} runs on a TPU {acc}.", True), (f"{name} works on one {chip} chip.", True),
                          (f"{name} runs on a single TPU {chip} with mashbox run.", True),
                          (f"The full pipeline of {name} fits on a single TPU {chip} chip.", False)):
        assert bool(SC.bad_chip_claims(m, said)) == flagged, said   # it fits there; `mashbox run` starts elsewhere
    assert f"{SC.chip_label(m, chip)}: {row['run_note']}" in SC.runs_on(m)
    assert f"TPU {row['accelerator']}" not in catalog.chips_run(m)
    assert f"TPU {chip}" not in US.chips(m["id"], reg)
    assert row["accelerator"] not in [k for k, _ in T.offered({**m, "cloud": m.get("cloud") or {"accelerator": "x"}})]
    for path in (ROOT / "models.json", ROOT / "src" / "mashbox" / "models.json"):
        for mm in json.loads(path.read_text())["models"]:
            for ch, c in (mm.get("chips") or {}).items():
                if c.get("status") == "measured":
                    assert _measured_run(mm, ch) and c.get("run_note") and c.get("note"), (path, mm["id"], ch)
                    assert f"TPU {c.get('accelerator') or ch}" not in catalog.chips_run(mm), (mm["id"], ch)

OUR_OTHER_CHIP_SPELLINGS = ("runs on Trillium only", "use a TPUv6e", "needs a TPU-v6e-4", "fits on TPU v6 lite x1",
                            "v​6e only", "needs a v6e⁠-4", "ｖ６ｅ only", "needs a v5p-16 slice")
SAYS_NOT_TESTED_TOO = ("hasn't been tested", "hasn’t been tested", "not yet been tested", "yet to be tested",
                       "no run yet", "TBD", "not tested", "untried", "no one has tried it")


def test_a_chip_row_reads_our_own_chip_spellings_and_not_tested_wordings_after_unicode_normalisation():
    """Our own chip spellings, read after NFKC with zero-width characters dropped, and the common not-tested
    wordings: a dated v5e row named v6e as 'Trillium', 'TPUv6e' or 'TPU v6 lite', or behind a
    zero-width or full-width character, and said 'hasn't been tested' or 'yet to be tested', and passed. A v5e row's
    own gcloud name (v5litepod-4) is its own chip."""
    for release in (False, True):
        ran = {"status": "not supported", "date": "2026-10-28"}
        for said in OUR_OTHER_CHIP_SPELLINGS:
            for key in ("reason", "note"):
                assert _row_problems("hy-nomas", "v5e", {**ran, key: said}, release=release, tag="another chip"), (key, said)
        for said in SAYS_NOT_TESTED_TOO:
            assert _row_problems("hy-nomas", "v5e", {**ran, "reason": said}, release=release, tag="not tested"), said
        for said in MEASURED_DOES_NOT_RUN:
            assert _row_problems("hy-nomas", "v5e", {**ran, "reason": said}, release=release) == [], said
        q = next(r for r in REG["models"] if r["id"] == "qwen3-8-27b")["chips"]["v5e"]
        assert _row_problems("qwen3-8-27b", "v5e", {**q, "accelerator": "v5litepod-4"}, release=release) == []
        assert _row_problems("qwen3-8-27b", "v6e", {"status": "not supported", "date": "2026-10-28",
                                                    "reason": "fits a v5litepod-4 only"}, release=release, tag="another chip")


def test_a_never_run_row_never_prints_not_supported_anywhere_note_column_included():
    """A pending row's note prints in the chips table's Note column, so pending + note 'Not
    supported' rendered '| Not tested yet | Not supported |' on the Hugging Face card with 0 problems. A row nobody ran
    prints 'Not supported' nowhere: not in its status, note, reason or not_run_reason."""
    pend = {"status": "pending", "not_run_reason": "CPU model: runs on the CPU, no TPU needed"}
    for release in (False, True):
        for entry in ({**pend, "note": "Not supported"}, {**pend, "note": "unsupported on this chip"},
                      {"status": "pending", "not_run_reason": "Not supported here"},
                      {"status": "not supported", "note": "Not tested yet. Not supported for now"}):
            assert _row_problems("track2rig", "v5e", entry, release=release, tag="Not supported"), (entry, release)
        assert _row_problems("track2rig", "v5e", pend, release=release, tag="Not supported") == []
    m = copy.deepcopy(next(r for r in REG["models"] if r["id"] == "track2rig"))
    m["chips"]["v5e"] = {**pend, "note": "Not supported"}
    assert "| Not supported |" in SC.chips_table(m)      # what the rule keeps off the pages


def test_a_row_that_reads_not_tested_yet_prints_not_supported_nowhere_and_only_a_dated_result_proves_a_run():
    """A pending row with a date, or a pending row on a chip with a results row, skipped the never-run checks, so its
    note 'Not supported' printed in the chips table next to 'Not tested yet'; an accelerator of 'not supported' printed
    in the chip label ('TPU v5e (not supported)'); and a results row with a null date counted as a run on its chip."""
    for release in (False, True):
        for entry in ({"status": "pending", "not_run_reason": "Not run on this chip yet", "date": "2026-10-28",
                       "note": "Not supported"},
                      {"status": "pending", "not_run_reason": "CPU model: runs on the CPU, no TPU needed",
                       "accelerator": "not supported"}):
            assert _row_problems("track2rig", "v5e", entry, release=release, tag="Not supported"), (entry, release)
        assert _row_problems("hy-nomas", "v6e", {"status": "pending", "not_run_reason": "the chip quota is not granted",
                                                 "note": "Not supported"}, release=release, tag="Not supported")
        hy = next(m for m in REG["models"] if m["id"] == "hy-nomas")
        res = copy.deepcopy(hy["results"]) + [{**hy["results"][0], "chip": "TPU v5e", "date": None}]
        got = _row_problems("hy-nomas", "v5e", {"status": "not supported"}, release=release, results=res)
        assert any("date is not a real" in x for x in got) and any("never run" in x for x in got), got
    m = copy.deepcopy(next(r for r in REG["models"] if r["id"] == "hy-nomas"))
    m["results"].append({**m["results"][0], "chip": "TPU v5e", "date": None})
    assert "v5e" not in SC.chip_rows_measured(m)


def test_a_not_supported_row_that_prints_not_tested_yet_prints_not_supported_nowhere_dated_or_not():
    """A chip nobody ran is stored as status 'not supported' with note 'Not tested yet', and the pages print the note.
    With a real date on that row, or a dated results row on its chip, the row no longer counted as never run and its
    cell is 'Not supported', so neither check read it: 'Not tested yet: not supported by this release' printed in the
    chips table and 'TPU v5e (v5e-4, not supported)' in the chip label, with 0 problems. Whatever a row prints as
    'Not tested yet', it prints 'Not supported' nowhere."""
    base = {"status": "not supported", "note": "Not tested yet"}
    hy = next(m for m in REG["models"] if m["id"] == "hy-nomas")
    res = copy.deepcopy(hy["results"]) + [{**hy["results"][0], "chip": "TPU v5e-4", "date": "2026-10-28"}]
    tag = "says Not supported"
    for release in (False, True):
        for entry in ({**base, "date": "2026-10-28", "note": "Not tested yet: not supported by this release"},
                      {**base, "date": "2026-10-28", "accelerator": "v5e-4, not supported"},
                      {**base, "date": "2026-10-28", "note": "NOT TESTED YET, unsupported"}):
            assert _row_problems("hy-nomas", "v5e", entry, release=release, tag=tag), (entry, release)
        for entry in ({**base, "accelerator": "not supported"}, {**base, "note": "Not tested yet: unsupported"}):
            assert _row_problems("hy-nomas", "v5e", entry, release=release, results=res, tag=tag), (entry, release)
        assert _row_problems("hy-nomas", "v5e", {**base, "date": "2026-10-28"}, release=release, tag=tag) == []
        assert _row_problems("hy-nomas", "v5e", base, release=release, results=res, tag=tag) == []
    m = copy.deepcopy(hy)
    m["chips"]["v5e"] = {**base, "date": "2026-10-28", "note": "Not tested yet: not supported by this release"}
    assert SC.chip_text(m, "v5e") == "Not tested yet: not supported by this release"   # what the rule keeps off the pages


def test_a_page_never_makes_fitting_one_chip_the_condition_while_a_model_passes_split_across_a_host():
    """README and docs/chips.md said "A model runs on v5e when its weights and working memory fit in 16 GB" while
    Qwen3.8-27B (26.94 GB) passes on a v5e-4, split across its 4 chips. A sentence that makes fitting one chip the
    condition to run on a chip type names the host of several chips or the split, for every chip type a model passes
    on that way; the public pages say so."""
    hosts = SC.multi_chip_hosts(REG)
    assert hosts.get("v5e") == ["v5e-4"], hosts
    for kind, hs in hosts.items():
        assert SC.chip_fit_problems(REG, f"A model runs on {kind} when its weights fit in one chip."), kind
        assert SC.chip_fit_problems(REG, f"TPU {kind} | One chip each. A model runs there when it fits. |"), kind
        assert SC.chip_fit_problems(REG, f"A model runs on {kind} when it fits one chip, or on a {hs[0]} otherwise.") == []
        assert SC.chip_fit_problems(REG, f"A model runs on {kind} if it fits; a larger one splits across chips.") == []
    assert SC.chip_fit_problems(REG, "A model runs on v5p when it fits in 95 GB.") == []      # no host of several v5p
    assert SC.chip_fit_problems(REG, "A TPU v5e chip has 16 GB.") == []                       # a spec, no condition
    found = [(rel, why) for rel, raw in SC.guard_pages(REG, ROOT) for line in SC._guard_text(raw).splitlines()
             for why in SC.chip_fit_problems(REG, line)]
    assert found == [], found


# ---------------------------------------------------------------- Checked on CPU

CPU = dict(status="passing", reference="llama.cpp a37cc46 CPU", identical_weights=dict(nmse=4.2e-12, top1="131/131"),
           block_path=dict(top1="128/131", kl=0.0021), greedy=dict(prompts=4, tokens=16, full=3, near_tie=1, other=0),
           tokenizer=dict(texts=8, equal=8, explained=0))


def cpu_row(**kw):
    """ROW with no chip run: its CPU check against llama.cpp passed and its chip is pending with the reason"""
    return {**ROW, "cpu_check": copy.deepcopy(CPU),
            "chips": {"v6e": dict(status="pending", not_run_reason="no chip run of this file")}, **kw}


def test_a_row_with_a_passing_cpu_check_and_no_chip_run_is_checked_on_cpu():
    row = cpu_row()
    assert problems(with_row(row)) == []
    assert catalog.level(row) == "Checked on CPU" and catalog.LEVELS.index("Checked on CPU") == 4
    text = catalog.render_text(with_row(row))
    assert ("Runs, checked against llama.cpp on CPU: same weights NMSE 4.2e-12, top-1 131/131; llama.cpp's own path "
            "top-1 128/131; greedy 4 x 16 tokens: 3 identical, 1 near tie") in text
    assert "TPU: Not tested yet" in text and "Checked on CPU" in catalog.render_markdown(with_row(row))
    assert catalog.level(dict(ROW, cpu_check=copy.deepcopy(CPU))) == "Tested"    # a chip run outranks the CPU check


def test_a_cpu_only_row_needs_a_passing_check_and_every_chip_pending_with_its_reason():
    failing = cpu_row(cpu_check=dict(CPU, status="failing"))
    assert any("no passing chip" in x for x in problems(with_row(failing))) and catalog.level(failing) == "Architecture supported"
    other = cpu_row(cpu_check=dict(CPU, greedy=dict(CPU["greedy"], full=2, other=1)))
    assert any("0 other greedy divergences" in x for x in problems(with_row(other)))
    for chips in ({"v6e": dict(status="pending")}, {"v6e": dict(status="pending", not_run_reason=" ")},
                  {"v6e": dict(status="pending", not_run_reason="x", measured={"decode_tok_s_1_user": 9.0})},
                  {"v6e": dict(status="pending", not_run_reason="x"), "v5e": dict(ROW["chips"]["v6e"], status="failing")}):
        assert problems(with_row(cpu_row(chips=chips))), chips
    assert any("cpu_check must have" in x for x in problems(with_row(cpu_row(cpu_check={"status": "passing"}))))


def test_a_cpu_checked_row_whose_chip_quick_failed_says_so():
    """a CPU-checked row whose chip QUICK ran and did not pass carries that failing QUICK (record_kind quick, the reason
    class its sealed compare gives) and reads "TPU: QUICK check did not pass": never "Not tested yet", never a speed"""
    fail = dict(QUICK_K, status="failing", reason="divergence", measured={"decode_tok_s_1_user": 9.5})
    row = cpu_row(chips={"v6e": fail})
    assert problems(with_row(row, bundle=BUNDLE)) == []
    assert catalog.level(row) == "Checked on CPU" and catalog.chips_run(row) == [] and catalog.quick_line(row) is None
    assert catalog.tpu_line(row) == "TPU: QUICK check did not pass"
    r = next(x for x in catalog.gguf_rows(with_row(row, bundle=BUNDLE)) if x["id"] == ROW["id"])
    assert (r["tpu"], r["tok_s_1_user"]) == ("TPU: QUICK check did not pass", None)
    assert catalog.QUICK_FAIL_REASONS == SC.QUICK_FAIL_REASONS and set(catalog.QUICK_FAIL_REASONS) == {
        "run", "first token", "divergence", "near tie"}
    for bad in (dict(fail, reason=None), dict(fail, reason="slow"), {k: v for k, v in fail.items() if k != "reason"},
                {k: v for k, v in fail.items() if k != "record_kind"}):
        assert problems(with_row(cpu_row(chips={"v6e": bad}), bundle=BUNDLE)), bad
    assert problems(with_row(cpu_row(cpu_check=dict(CPU, status="failing"), chips={"v6e": fail}), bundle=BUNDLE))


def test_a_mixture_of_experts_is_never_listed_from_a_cpu_check():
    moe = cpu_row(moe=True, params_b={"total": 30.0, "active": 3.0})
    assert any("mixture of experts is only listed" in x for x in problems(with_row(moe)))


def _cpu_records(base, edit=None):
    """A records root under base: one sealed gguf_conform CPU record (record.json, card.json, header.bin, SHA256SUMS,
    quant.json), the git repo of the code it ran (this tree's files) and a copy of that tree to compare; returns
    (row, evidence, tree). edit(part, value) changes a part before it is written."""
    edit = edit or (lambda part, v: v)
    root, code, tree = base / "records", base / "records" / "code", base / "tree"
    for d in (code, tree):
        for sub in ("src/mashbox", "tools"):
            shutil.copytree(ROOT / sub, d / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    commit = _commit(code)
    rec_dir = root / "rec" / "f"
    rec_dir.mkdir(parents=True)
    (rec_dir / "quant.json").write_text(json.dumps({"types": {"Q4_K": True}}))
    rec = edit("record", dict(
        kind="gguf-conformance", pin=dict(repo=ROW["repo"], revision=ROW["revision"], file=ROW["file"]),
        sha256=ROW["sha256"], bytes=ROW["bytes"], arch=ROW["arch"], moe=ROW["moe"], types=ROW["types"],
        bars=dict(CPU_BARS, verdict="..."), verdict="cpu-checked",
        code=dict(sha256=conform_code_id(str(code), commit), files=20, git_head=None),
        versions=dict(llama_cpp="a37cc46104eb50900d5ab4a9ba02a0abdac039db"),
        b_tokenizer=dict(texts=8, nfc_equal=8, nfc_explained=0),
        c_cpu=dict(identical_weights=dict(nmse=4.2e-12, top1_agree=131, positions=131), top1_agree=128, positions=131,
                   kl_mean=0.00213, prompts=4, n_gen=16, full_match=3, near_tie_divergences=1, other_divergences=0),
        d_quant=dict(record_sha256=_sha(rec_dir / "quant.json"),
                     types={t: True for t in ROW["types"] if t not in catalog.UNQUANTIZED})))
    (rec_dir / "record.json").write_text(json.dumps(rec))
    (rec_dir / "card.json").write_text(json.dumps(edit("card", {"id": ROW["repo"], "sha": ROW["revision"],
                                                                "cardData": {"license": ROW["license"]}})))
    (rec_dir / "header.bin").write_bytes(b"GGUF")
    (rec_dir / "SHA256SUMS").write_text("".join(f"{_sha(rec_dir / n)}  {n}\n" for n in ("card.json", "header.bin", "record.json")))
    ev = {k: {"path": str(rec_dir / n), "sha256": _sha(rec_dir / n)} for k, n in (
        ("record", "record.json"), ("card", "card.json"), ("header", "header.bin"), ("sha256sums", "SHA256SUMS"),
        ("quant", "quant.json"))}
    ev = edit("evidence", dict(ev, code={"commit": commit}, decides=cpu_record_files(tree, ROW["arch"])))
    row = edit("row", cpu_row(cpu_check=cpu_check_of(rec)))
    return row, dict(records_root=str(root), code_repo=str(code), models={}, gguf={row["id"]: {"cpu_check": ev}}), tree


def _cpu_guard(row, ev, tree):
    reg = copy.deepcopy(REG)
    reg["models"], reg["gguf"], reg["engine_kernels"] = [], [row], []
    return evidence_problems(reg, ev, tree)


def test_the_cpu_record_guard_holds_a_backed_row_and_reads_every_number_from_its_record(tmp_path):
    row, ev, tree = _cpu_records(tmp_path)
    assert _cpu_guard(row, ev, tree) == []
    off = copy.deepcopy(row)
    off["cpu_check"]["block_path"]["top1"] = "129/131"
    assert any("cpu_check ['block_path'] differ from the record" in x for x in _cpu_guard(off, ev, tree))
    for part, change, says in (
            ("record", lambda r: dict(r, bars=dict(r["bars"], top1_min_share=0.9)), "not the locked CPU bars"),
            ("record", lambda r: dict(r, pin=dict(r["pin"], revision="1" * 40)), "pin.revision"),
            ("record", lambda r: dict(r, code=dict(r["code"], sha256="0" * 64)), "the code id at"),
            ("record", lambda r: dict(r, d_quant=dict(r["d_quant"], types={})), "quant check does not pass"),
            ("card", lambda c: dict(c, cardData={"license": "other"}), "the pinned card says"),
            ("evidence", lambda e: dict(e, decides=e["decides"][1:]), "decides must list")):
        sub = tmp_path / f"{part}-{says[:6]}"
        sub.mkdir()
        r2, e2, t2 = _cpu_records(sub, lambda p, v, part=part, change=change: change(v) if p == part else v)
        assert any(says in x for x in _cpu_guard(r2, e2, t2)), (part, says)
    failed = tmp_path / "failed"
    failed.mkdir()
    r2, e2, t2 = _cpu_records(failed, lambda p, v: dict(v, verdict="failed") if p == "record" else v)
    r2["cpu_check"]["status"] = "passing"
    assert any("cpu_check ['status'] differ from the record (status from verdict 'failed')" in x
               for x in _cpu_guard(r2, e2, t2))


def test_the_cpu_record_guard_voids_a_record_only_when_a_file_that_decides_it_changes(tmp_path):
    """Measured numbers stand. The scan tool, another architecture, the catalog view, an import line, a comment or
    a docstring never void a CPU record; a code change in a file that decides it (its arch module, the reader, the
    check tool) does."""
    row, ev, tree = _cpu_records(tmp_path)
    arch_file = arch_module_files(tree / ARCHS_SRC)[row["arch"]]
    others = [f for a, f in arch_module_files(tree / ARCHS_SRC).items() if f not in cpu_record_files(tree, row["arch"])]
    for f, change in (("tools/gguf_scan.py", lambda t: t + "\nX = 1\n"), (others[0], lambda t: t + "\nX = 1\n"),
                      ("src/mashbox/catalog.py", lambda t: t + "\nX = 1\n"),
                      (CPU_TOOL, lambda t: t.replace("import argparse\n", "import argparse\nimport urllib.request\n", 1)),
                      (arch_file, lambda t: "# a comment\n" + t)):
        p = tree / f
        p.write_text(change(p.read_text()))
        assert _cpu_guard(row, ev, tree) == [], f
    for f in (arch_file, "src/mashbox/gguf/reader.py", CPU_TOOL):
        p = tree / f
        keep = p.read_text()
        p.write_text(keep + "\nX = 1\n")
        assert any(f"{f} differs from the code the record ran" in x for x in _cpu_guard(row, ev, tree)), f
        p.write_text(keep)


def test_the_guard_reads_the_same_code_id_as_the_check_tool(tmp_path):
    """conform_code_id (git, at a commit) is gguf_conform.code_id() of the same files"""
    spec = importlib.util.spec_from_file_location("gguf_conform", ROOT / CPU_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    code = tmp_path / "code"
    for sub in ("src/mashbox", "tools"):
        shutil.copytree(ROOT / sub, code / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert conform_code_id(str(code), _commit(code)) == tool.code_id()["sha256"]
def test_a_model_that_runs_on_the_cpu_says_so_in_every_chip_list():
    """Track2Rig's cpu row passes in models.json, but llms.txt and docs/supported-models.md said "TPU v6e" only, the
    README chips cell had no CPU line, and `mashbox tpu up track2rig` stopped without saying where it runs. Every live
    model with a passing cpu row is named CPU in each chip list, first, from that row (llms.txt Runs on, the README
    chips cell, the package catalog behind supported-models.md and `mashbox models`), and `tpu up` for one without TPU
    settings says it runs on a CPU with its install and run lines."""
    from mashbox import tpu as T
    llms = (ROOT / "llms.txt").read_text().splitlines()
    readme = (ROOT / "README.md").read_text().splitlines()
    supported = (ROOT / "docs" / "supported-models.md").read_text().splitlines()
    seen = 0
    for m in SC.live(REG):
        if (m["chips"].get("cpu") or {}).get("status") != "passing":
            continue
        seen += 1
        line = next(x for x in llms if x.startswith(f"- [{m['name']}]("))
        assert line.split(" Runs on: ", 1)[-1].startswith("CPU"), line
        row = next(x for x in readme if x.startswith(f"| **[{m['name']}]("))
        assert row.split(" | ")[2].startswith("CPU: Supported<br>"), row
        assert catalog.chips_run(m)[0] == "CPU"
        row = next(x for x in supported if x.startswith(f"| **[{m['name']}]("))
        assert row.split(" | ")[3].startswith("CPU"), row
        if "cloud" not in m:
            with pytest.raises(T.Stop) as e:
                T.model(m["id"])
            assert "runs on a CPU" in str(e.value) and m["install"] in str(e.value) and m["run"] in str(e.value), e.value
    assert seen


def test_the_chips_table_runs_on_names_every_chip_the_model_starts_on():
    """docs/chips.md and the FAQ's chips table showed only the default chip under "Runs on" (Qwen3.8-27B: "TPU v6e-1")
    while its v5e-4 and v6e-4 rows pass. That column names every chip the model starts on, the same list as
    docs/supported-models.md "Runs on" and `mashbox models` (the package catalog: CPU when its row passes, then every
    passing chips row); a chip it was only measured on keeps its note in its own column."""
    for page in ("docs/chips.md", "docs/faq.md"):
        block = (ROOT / page).read_text().split("<!-- chips:start -->", 1)[1].split("<!-- chips:end -->", 1)[0]
        for m in SC.live(REG):
            row = next(x for x in block.splitlines() if x.startswith(f"| {m['name']} | "))
            assert row.split(" | ")[1] == ", ".join(catalog.chips_run(m)), (page, row)
    assert SC.chips_block(REG) in (ROOT / "docs" / "chips.md").read_text()


def test_a_passing_chip_prints_its_measured_speed():
    """Qwen3.8-27B's v5e-4 and v6e-4 rows carry their measured decode speed in models.json (76.73 and 110.78 tokens/s
    for 1 request), but no page printed it: a chip we ran on is supported with its measured speed. A passing chips row
    with a measured block prints its decode speed for 1 request (and for 8 when measured) wherever its chip line
    prints: the chips table (docs/chips.md, the FAQ) and the model's page."""
    pages = {p: (ROOT / "docs" / p).read_text() for p in ("chips.md", "faq.md")}
    seen = 0
    for m in SC.live(REG):
        page = (ROOT / "docs" / "models" / f"{m['id']}.md").read_text()
        for chip in SC.CHIPS:
            c = m["chips"].get(chip) or {}
            one = (c.get("measured") or {}).get("decode_tok_s_1_user")
            if c.get("status") != "passing" or one is None:
                continue
            seen += 1
            said = f"{one:g} tokens/s for 1 request"
            assert said in SC.chip_text(m, chip), (m["id"], chip)
            for name, text in {**pages, m["id"]: page}.items():
                assert said in text, (name, m["id"], chip)
    assert seen >= 3


# ---------------------------------------------------------------- quick-check chip record (record_kind "quick-check")
# A row's chip says "passing" from a quick-check run: four `mashbox run REPO:TAG --prompt P --max-tokens 64
# --temperature 0` runs with zero engine flags, each compared with llama.cpp's greedy ids on the same templated prompt.
# It measures one number (decode tok/s, 1 user) and records no ctx, no slots and no 8-user number, so its row has none.

QUICK_PROMPTS = ["The capital of France is Paris. The capital of Japan is",
              'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
              "Translate to German: The weather is nice.\nGerman:", "Question: What is 17 times 23?\nAnswer:"]
QUICK_STOP = [7, 8]
# (new tokens, decode seconds) per run; run 0 holds the first compile, so the row's number is runs 1-3:
# (64 + 14 + 64) / (4.0 + 1.6 + 4.4) = 14.2
QUICK_TIMING = [(3, 100.0), (64, 4.0), (14, 1.6), (64, 4.4)]
QUICK_DECODE = 14.2
# a QUICK run (record_kind "quick") is 32 tokens: (32 + 14 + 32) / (2.0 + 1.6 + 2.4) = 13.0
QUICK_TIMING_Q = [(3, 100.0), (32, 2.0), (14, 1.6), (32, 2.4)]
QUICK_DECODE_Q = 13.0
# its `--speculation ngram` runs (same ids): (32 + 14 + 32) / (1.0 + 0.8 + 1.2) = 26.0
QUICK_TIMING_N = [(3, 90.0), (32, 1.0), (14, 0.8), (32, 1.2)]
QUICK_DECODE_N = 26.0
# the engine_kernels row a bundle QUICK ran (types: the fixture's quantized types), the n-gram runs' component set, the
# engine's speculation step version and the gate entry's k_table: k 3 (the runs' draft_k) at the 1-row bucket it
# measured, 0 at every other bucket the engine has (engine._rows)
QUICK_KROW = dict(BUNDLE, sha256="c" * 64)
QUICK_CSS = "f" * 64
QUICK_SPEC_STEP = __import__("mashbox.runtime.spec_step", fromlist=["x"]).SPEC_STEP_VERSION
QUICK_K_TABLE = {"1": 3, "4": 0, "16": 0, "64": 0, "256": 0}


def _quick_ids(i, timing=QUICK_TIMING, top=64):
    """(prompt ids, ours, llama.cpp's generated ids) of run i: run 0 and 2 stop on a stop id llama.cpp also picks; run 3
    meets llama.cpp's near tie at step 31"""
    pid = [1, 2, 3, 40 + i]
    n = timing[i][0]
    ours = [100 + i * 100 + s for s in range(n - 1)] + [QUICK_STOP[0]] if n < top else [100 + i * 100 + s for s in range(top)]
    ref = list(ours) + [9] * (top - len(ours))
    if i == 3:
        ref[31] = 5
    return pid, ours, ref


STEADY_MS = [10.0, 10.0008, 9.9992, 10.0, 10.0016]     # ms per new token of the 5 timed rounds: median 10.0
STEADY_TOK = 100.0
STEADY_HOST = "observed-from 2026-10-30T15:00:00Z\nwindow 2026-10-30T15:30:00Z 2026-10-30T15:31:00Z\n"


def _fake_quick_check(base, edit=None, kind="quick-check", path="exact", ngram=False, forced=None, steady=None):
    """A records root under base with one quick-check chip run of the qwen3-kquant fixture: the four runs (json, exit
    code, stdout, stderr events), the compare, llama.cpp's ids, the prompts, the chip's setup log, pip freeze,
    accelerator type and code commit, the file/plan/card/header of the row, one SHA256SUMS seal, and the git repo of the
    code it ran. edit(part, value) changes a part before it is written ("run0".."run3", "rc0".., "stderr0"..,
    "cmp", "setup", "freeze", "accelerator", "code_sha", "prompt0".., "ref0".., "seal" (lines), "row", "reg", "ev").
    path="bundle": the runs ran on the engine_kernels row's package (QUICK_KROW, fetched by the engine from the catalog:
    kernel-wheel.sha256 sealed), on one chip of a v6e-4 host (chip_count 1; each run records the engine's chip key and
    device count). ngram=True: the same folder also seals the four `--speculation ngram` runs ("ngram-run0".., their
    compare "ngram_cmp") and the row carries their gate entry (edit "spec"). forced={i: {"token", "ids", "gap"}}: the
    same folder also seals prompt i's forced run (the token-0 near-tie rule: the plain argv, `token` appended to the prompt ids,
    `ids` its output; edit "tfrun<i>") and their compare tf-cmp.json (gap: llama.cpp's gap at a first difference).
    steady="alone"|"shared" (path bundle): the row's number is its steady-decode speed with that condition, from its own
    sealed record in steady-row/ (edit "steady_json", "steady_stderr", "steady_rc", "steady_prompt", "steady_wheel",
    "steady_code", "host", "steady_seal"); "shared" adds an other-job line over the run's window to host.txt."""
    edit = edit or (lambda part, v: v)
    top, tool = QUICK_KINDS[kind]
    timing, decode = (QUICK_TIMING, QUICK_DECODE) if kind == "quick-check" else (QUICK_TIMING_Q, QUICK_DECODE_Q)
    bundle = path == "bundle"
    root = base / "records"
    d = root / "p1row"
    d.mkdir(parents=True)
    from mashbox.gguf import File
    fx = File(str(FIXTURE))
    code = {f: (ROOT / f).read_bytes() for f in gguf_code_files(ROOT, fx.arch) + [ARCHS_SRC]}
    krow = dict(QUICK_KROW, types=sorted(t for t in fx.census()[0] if t not in ("F32", "F16", "BF16")))
    code["models.json"] = json.dumps({"engine_kernels": [krow] if bundle else []}).encode()
    code["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    for f, data in code.items():
        (root / "code" / f).parent.mkdir(parents=True, exist_ok=True)
        (root / "code" / f).write_bytes(data)
    commit = _commit(root / "code")
    on = ({"accelerator": "v6e-4", "chip_count": 1, "kernel_path": "bundle"} if bundle else
          {"accelerator": "v6e-1", "kernel_path": "exact"})
    measured = ({"steady_decode_tok_s_1_user": STEADY_TOK, "steady_condition": catalog.STEADY_CONDITIONS[steady]}
                if steady else {"decode_tok_s_1_user": decode})
    row = edit("row", dict(ROW, arch=fx.arch, bytes=FIXTURE.stat().st_size, types=sorted(fx.census()[0]), chips={
        "v6e": {"status": "passing", **on, "record_kind": kind, "measured": measured}}))
    tag = f"{row['repo']}:{row['tag']}"
    parts, rows = {}, []
    for pre, tm, flags in (("", timing, []),) + ((("ngram-", QUICK_TIMING_N, ["--speculation", "ngram"]),) if ngram else ()):
        rows = []
        for i in range(4):
            pid, ours, ref = _quick_ids(i, timing, top)
            n, dec = tm[i]
            parts[f"prompt{i}"] = QUICK_PROMPTS[i]
            parts[f"{pre}run{i}"] = {"argv": ["run", tag, "--prompt", QUICK_PROMPTS[i], *flags, "--max-tokens", str(top),
                                              "--temperature", "0"],
                                     **({"chip_key": "v6e-1", "devices": 1} if bundle else {}),
                                     "env": {}, "rc": None, "error": None, "wall_s": dec + 12.0, "generate_calls": [{
                                         "call_start_s": 11.0, "call_end_s": 11.0 + dec, "prompt_ids": [pid], "ids": [ours],
                                         "timing": {"prefill_s": 0.5, "decode_s": dec, "decode_calls": 9, "new_tokens": n,
                                                    "sampled": False}, "max_new": top, "stop_ids": QUICK_STOP, "samp": None}]}
            parts[f"{pre}rc{i}"] = f"0 {dec + 12.5}\n"
            parts[f"{pre}stdout{i}"] = "some text\n"
            parts[f"{pre}stderr{i}"] = "\n".join(json.dumps(x) for x in (
                {"event": "gguf_file", "repo": row["repo"], "file": row["file"], "shards": 1, "gb": 0.0,
                 "revision": row["revision"], "rule": f"tag {row['tag']}"},
                {"event": "gguf_plan", "arch": row["arch"], "layers": 2,
                 "kernel_types": krow["types"] if bundle else [], "reference_share": 0.0 if bundle else 99.9,
                 "chip": None},
                {"event": "weights", "file": row["file"], "kernel_bytes": 900 if bundle else 0,
                 "reference_bytes": 0 if bundle else 1000},
                *([{"event": "speculation", "used": True, "method": "ngram", "draft_k": 3, "gated": False,
                    "reason": "--speculation ngram", "spec_step_version": QUICK_SPEC_STEP}] if flags else []),
                {"event": "gguf_identity", "sha256": row["sha256"], "shards": 1},
                {"event": "deployment", "deployment_id": "d" * 64, "component_set_sha": QUICK_CSS if flags else "e" * 64},
                {"event": "done", "new_tokens": n, "prefill_s": 0.5, "decode_s": dec})) + "\n"
            parts[f"ref{i}"] = " ".join(map(str, pid)) + "\n\n" + " ".join(map(str, ref)) + "\n"
            k = 31 if i == 3 else None
            rows.append({"i": i, "rc": None, "process_rc": 0, "wall_s": dec + 12.5, "error": None, "printed_chars": 9,
                         "generate_calls": 1, "loads_and_prints": True, "ids_prompt": pid, "ids_ours": ours,
                         "ids_ref": ref, "ref_prompt_equal": True, "compared": len(ours), "first_diff": k,
                         "ref_gap": 0.21 if k is not None else None, "near_tie": k is not None,
                         "kind": "near" if k is not None else "full"})
        parts[f"{pre.replace('-', '_')}cmp"] = {"summary": {"runs_ok": 4, "full": 3, "near": 1, "other": 0, "PASS": True},
                                                "bar": f"{top} greedy tokens vs llama.cpp", "rows": rows}
    parts["setup"] = ("Using CPython 3.12.12\nCreating virtual environment\nsetup rc 0\n"
                      "0.1.5 0.11.2 0.0.48.1 TPU v6 lite\n")
    parts["freeze"] = "".join(f"{k}=={v}\n" for k, v in (
        ("gguf", "0.19.0"), ("jax", "0.11.2"), ("jaxlib", "0.11.2"), ("jinja2", "3.1.6"), ("libtpu", "0.0.48.1"),
        ("numpy", "2.5.4"), ("tokenizers", "0.23.3"))) + "mashbox @ file:///k/wheel/mashbox-0.1.5-py3-none-any.whl\n"
    parts["accelerator"] = on["accelerator"]
    parts["kernel_wheel"] = f"{krow['sha256']}  /home/u/.cache/mashbox/kernels/{krow['sha256'][:16]}/{krow['file']}\n"
    parts["code_sha"] = commit + "\n"
    parts["cmp_tool"] = "# the compare tool\n"
    parts["card"] = {"id": row["repo"], "sha": row["revision"], "cardData": {
        "license": row["license"], "license_link": row["license_link"], "base_model": row["base"]}}
    parts["file"] = f"{row['sha256']}  {row['file']}\n{row['bytes']}\n"
    parts["plan"] = f"{row['repo']} {row['revision']} {row['file']} {row['sha256']}\n"
    parts["header"] = _header(FIXTURE)
    names = {**{f"{x}{i}": f"{y}{i}{z}" for i in range(4) for x, y, z in (
        ("run", "run", ".json"), ("rc", "run", ".rc"), ("stdout", "run", ".stdout"), ("stderr", "run", ".stderr"),
        ("prompt", "prompt", ".txt"), ("ref", "ref", ".ids"))},
             "cmp": "cmp.json", "setup": "setup.log", "freeze": "freeze.txt", "accelerator": "accelerator-type.txt",
             "code_sha": "CODE_SHA", "cmp_tool": tool, "card": "card-api.json", "file": "file-row.sha256",
             "plan": "plan-row.txt", "header": "header-row.gguf",
             **({"kernel_wheel": "kernel-wheel.sha256"} if bundle else {}),
             **({f"ngram-{x}{i}": f"ngram-{y}{i}{z}" for i in range(4) for x, y, z in (
                 ("run", "run", ".json"), ("rc", "run", ".rc"), ("stdout", "run", ".stdout"),
                 ("stderr", "run", ".stderr"))} if ngram else {}),
             **({"ngram_cmp": "ngram-cmp.json"} if ngram else {})}
    for k, n in names.items():
        v = edit(k, parts[k])
        if isinstance(v, bytes):
            (d / n).write_bytes(v)
        else:
            (d / n).write_text(v if isinstance(v, str) else json.dumps(v))
    tf_names, tf_rows = [], []
    for i, f in (forced or {}).items():
        a, _, b = (d / f"ref{i}.ids").read_text().partition("\n\n")
        rg = [int(t) for t in b.split()]
        run = json.loads((d / f"run{i}.json").read_text())
        c = run["generate_calls"][0]
        tf = edit(f"tfrun{i}", {**run, "forced": f["token"], "generate_calls": [dict(
            c, cli_prompt_ids=c["prompt_ids"], prompt_ids=[c["prompt_ids"][0] + [f["token"]]], ids=[f["ids"]])]})
        ours = ([f["token"]] + f["ids"])[:top]
        k = quick_check_first_diff(ours, rg, set(QUICK_STOP), top)
        gap = None if k is None else f.get("gap", 1.0)
        tf_rows.append({"i": i, "forced": f["token"], "ids_ours": ours, "first_diff": k, "ref_gap": gap,
                        "kind": "full" if k is None else ("near" if gap < NEAR_TIE_GAP else "other"),
                        "loads_and_prints": True})
        for n, v in ((f"tf-run{i}.json", json.dumps(tf)), (f"tf-run{i}.rc", "0 9.0\n"), (f"tf-run{i}.stdout", "some text\n"),
                     (f"tf-run{i}.stderr", (d / f"run{i}.stderr").read_text())):
            (d / n).write_text(v)
            tf_names.append(n)
    if forced:
        (d / "tf-cmp.json").write_text(json.dumps({"summary": {"runs": len(tf_rows)}, "rows": tf_rows}))
        tf_names.append("tf-cmp.json")
    seal = edit("seal", [f"{_sha(d / n)}  {n}" for n in [*names.values(), *tf_names]])
    (d / "SHA256SUMS").write_text("\n".join(seal) + "\n")

    def src(n):
        return {"path": str(d / n), "sha256": _sha(d / n)}
    ev = {"records_root": str(root), "code_repo": str(root / "code"), "models": {}, "engine_kernels": {},
          "gguf": {row["id"]: {"v6e": {
              "kind": kind, "sha256sums": src("SHA256SUMS"), "compare": src("cmp.json"),
              "file": src("file-row.sha256"), "plan": src("plan-row.txt"), "card": src("card-api.json"),
              "header": src("header-row.gguf"), "code": {"commit": commit}}}}}
    if ngram:
        row["speculation"] = {"ngram": {"v6e-1": edit("spec", {
            "status": "passed", "record": f"gguf/{row['id']}/ngram/v6e-1", "record_sha256": _sha(d / "SHA256SUMS"),
            "component_set_sha": QUICK_CSS, "spec_step_version": QUICK_SPEC_STEP, "k_table": dict(QUICK_K_TABLE)})}}
        ev["gguf"][row["id"]]["speculation"] = {"ngram": {"v6e-1": {"sha256sums": src("SHA256SUMS"),
                                                                    "compare": src("ngram-cmp.json")}}}
    if steady:
        sd = root / "steady-row"
        sd.mkdir()
        ids = [5 + (i % 50) for i in range(128)]
        rounds = [{"round": i, "wall_s": 1.3, "prefill_s": 0.01, "decode_s": round(ms * 127 / 1e3, 6), "new_tokens": 128,
                   "decode_calls": 17, "ms_per_token": ms, "ids": list(ids)} for i, ms in enumerate(STEADY_MS)]
        sparts = {
            "steady_json": {"model": tag, "rounds": rounds, "new": 128, "jax": "0.11.2", "devices": 1,
                            "argv_tail": ["--max-tokens", "8", "--temperature", "0"], "slots": 8, "kernels": "0.1.5",
                            "warm_s": 6.0, "median_ms_per_token": 10.0, "steady_decode_tok_s_1_user": STEADY_TOK,
                            "spread": 0.00024, "same_ids": True},
            "steady_stderr": parts["stderr1"],
            "steady_rc": "rc 0\n",
            "steady_prompt": QUICK_PROMPTS[0],
            "steady_wheel": parts["kernel_wheel"],
            "steady_code": commit + "\n",
            "host": STEADY_HOST + ("other 2026-10-30T15:13:29Z 2026-10-30T15:40:00Z chip 0\n" if steady == "shared" else ""),
        }
        snames = {"steady_json": "steady.json", "steady_stderr": "steady.stderr", "steady_rc": "steady.rc",
                  "steady_prompt": "prompt.txt", "steady_wheel": "kernel-wheel.sha256", "steady_code": "CODE_SHA",
                  "host": "host.txt"}
        for k, n in snames.items():
            v = edit(k, sparts[k])
            (sd / n).write_text(v if isinstance(v, str) else json.dumps(v))
        sseal = edit("steady_seal", [f"{_sha(sd / n)}  {n}" for n in snames.values()])
        (sd / "SHA256SUMS").write_text("\n".join(sseal) + "\n")
        ev["gguf"][row["id"]]["v6e"]["steady"] = {"sha256sums": {"path": str(sd / "SHA256SUMS"),
                                                                 "sha256": _sha(sd / "SHA256SUMS")},
                                                  "code": {"commit": commit}}
    reg = copy.deepcopy(REG)
    reg["gguf"], reg["engine_kernels"], reg["models"] = [row], [krow] if bundle else [], []
    return edit("reg", reg), edit("ev", ev)


def _quick_ref_first(v):
    """llama.cpp's sealed ids for prompt 1 differ from ours at the first generated token"""
    a, b = v.split("\n\n")
    ids = b.split()
    ids[0] = "6"
    return a + "\n\n" + " ".join(ids) + "\n"


def _quick_cmp_first(c):
    """the compare says prompt 1 met a near tie at step 0 (gap 0.1); the summary agrees"""
    c["rows"][1].update(first_diff=0, ref_gap=0.1, near_tie=True, kind="near")
    c["summary"].update(full=2, near=2)


def _quick_cmp_other(c):
    """the compare says prompt 2 diverged at step 5 with llama.cpp's gap 1.5 (an other divergence); the summary agrees"""
    c["rows"][2].update(first_diff=5, ref_gap=1.5, near_tie=False, kind="other")
    c["summary"].update(full=2, other=1, PASS=False)


def _quick_ref_differs(v):
    """llama.cpp's sealed ids for prompt 2 differ from ours at step 5"""
    a, b = v.split("\n\n")
    ids = b.split()
    ids[5] = "6"
    return a + "\n\n" + " ".join(ids) + "\n"


def _quick_stderr(i, fn):
    """an edit of run i's stderr: fn(event dict) on every event line"""
    def change(v):
        out = []
        for line in v.splitlines():
            x = json.loads(line)
            fn(x)
            out.append(json.dumps(x))
        return "\n".join(out) + "\n"
    return _set(f"stderr{i}", change)


def _quick_stderr_n(i, fn):
    """an edit of ngram-run i's stderr: fn(event dict) on every event line"""
    def change(v):
        return "\n".join(json.dumps((lambda x: (fn(x), x)[1])(json.loads(line))) for line in v.splitlines()) + "\n"
    return _set(f"ngram-stderr{i}", change)


def _quick_row(fn):
    return _in_place("row", lambda r: fn(r["chips"]["v6e"]))


QUICK_BAD = {
    "a flag": (_in_place("run1", lambda v: v["argv"].extend(["--ctx", "4096"])),
               "run1 ran `mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M --prompt"),
    "another prompt": (_set("prompt2", lambda v: v + " Please."), "run2 ran `mashbox run"),
    "a local file": (_in_place("run0", lambda v: v["argv"].__setitem__(1, "/models/Tiny-Qwen3-Q4_K_M.gguf")),
                     "run0 ran `mashbox run /models/"),
    "env": (_in_place("run3", lambda v: v["env"].update(MASHBOX_GGUF_KERNELS="off")),
            "run3 ran with MASHBOX_GGUF_KERNELS=off"),
    "exit code": (_set("rc2", lambda v: "1 13.0\n"), "run2 did not load and print"),
    "error": (_in_place("run1", lambda v: v.__setitem__("error", "boom")), "run1 did not load and print"),
    "nothing printed": (_set("stdout0", lambda v: "\n"), "run0 did not load and print"),
    "other divergence, said": (_both(_set("ref2", _quick_ref_differs), _in_place("cmp", _quick_cmp_other)),
                               "1 other divergence"),
    "other divergence, hidden": (_set("ref2", _quick_ref_differs), "prompt 2: the compare says first difference None, "
                                                                "the sealed ids give 5"),
    "summary": (_in_place("cmp", lambda c: c["summary"].update(full=4, near=0)), "its summary says"),
    "compare of other ids": (_in_place("cmp", lambda c: c["rows"][1]["ids_ours"].__setitem__(0, 1)),
                             "compare row 1 is not run1's ids"),
    "sha256": (_quick_stderr(1, lambda x: x.update(sha256="b" * 64) if x["event"] == "gguf_identity" else None),
               f"run1 read a file of sha256 {'b' * 64}"),
    "revision": (_quick_stderr(0, lambda x: x.update(revision="1" * 40) if x["event"] == "gguf_file" else None),
                 "run0 resolved someone/Tiny-Qwen3-GGUF Tiny-Qwen3-Q4_K_M.gguf @ 1111"),
    "kernel types": (_quick_stderr(2, lambda x: x.update(kernel_types=["Q4_K"]) if x["event"] == "gguf_plan" else None),
                     "kernel_path exact, and run2 ran kernel types ['Q4_K']"),
    "kernel bytes": (_quick_stderr(2, lambda x: x.update(kernel_bytes=5) if x["event"] == "weights" else None),
                     "kernel_path exact, and run2 ran 5 bytes on a kernel"),
    "arch": (_quick_stderr(3, lambda x: x.update(arch="llama") if x["event"] == "gguf_plan" else None),
             "run3 ran it as llama"),
    "version": (_set("freeze", lambda v: v.replace("jinja2==3.1.6", "jinja2==3.1.5")),
                "the record ran jinja2 3.1.5, this tree pins 3.1.6"),
    "no python": (_set("setup", lambda v: v.replace("Using CPython 3.12.12\n", "")),
                  "the record's versions have no python"),
    "decode number": (_quick_row(lambda r: r["measured"].update(decode_tok_s_1_user=15.0)),
                      "models.json says 15.0, the sealed runs 1-3 give 14.2"),
    "decode timing": (_in_place("run2", lambda v: v["generate_calls"][0]["timing"].update(decode_s=2.6)),
                      "models.json says 14.2, the sealed runs 1-3 give 12.91"),
    "a typed number": (_quick_row(lambda r: r["measured"].update(decode_tok_s_8_users=120.0)),
                       "measured, when present, holds"),
    "not sealed": (_set("seal", lambda v: [x for x in v if not x.endswith("setup.log")]), "setup.log is not in SHA256SUMS"),
    "seal sha256": (_set("seal", lambda v: [("0" * 64 + x[64:]) if x.endswith("freeze.txt") else x for x in v]),
                    "freeze.txt sha256 differs from its seal line"),
    "accelerator": (_set("accelerator", lambda v: "v6e-4"), "accelerator v6e-1, the chip says v6e-4"),
    "device": (_set("setup", lambda v: v.replace("TPU v6 lite", "TPU v5 lite")), "the run's device is TPU v5 lite"),
    "code": (_set("code_sha", lambda v: "c" * 40 + "\n"), f"the wheel was built from {'c' * 40}"),
    "ctx on the row": (_quick_row(lambda r: r.update(ctx=2048)), "a quick-check run records no ctx or slots"),
    "no record_kind on the row": (_quick_row(lambda r: r.pop("record_kind")), "(a quick-check record), and the row's "
                                                                           "record_kind is None"),
    "header types": (_in_place("row", lambda r: r.__setitem__("types", ["F32", "Q4_K"])),
                     "the sealed header's block types are"),
    "file": (_set("file", lambda v: v.split("\n")[0] + "\n999\n"), "file-row.sha256 names another file"),
}


def test_record_guard_holds_a_quick_check_row(tmp_path):
    reg, ev = _fake_quick_check(tmp_path)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


@pytest.mark.parametrize("form", sorted(QUICK_BAD))
def test_record_guard_names_each_bad_quick_check_form(tmp_path, form):
    edit, want = QUICK_BAD[form]
    reg, ev = _fake_quick_check(tmp_path, edit)
    got = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(want in x for x in got), got


def test_a_run3_row_still_needs_its_ctx_and_slots(tmp_path):
    """the run3 path is unchanged: a row without record_kind and without ctx is refused by the schema"""
    row = dict(ROW, chips={"v6e": {k: v for k, v in ROW["chips"]["v6e"].items() if k != "ctx"}})
    assert any("chips.v6e needs" in x for x in problems(with_row(row)))


def test_the_schema_holds_a_quick_check_row_to_what_it_measured():
    qc = {"status": "passing", "accelerator": "v6e-1", "kernel_path": "exact", "record_kind": "quick-check",
          "measured": {"decode_tok_s_1_user": 14.2}}
    assert problems(with_row(dict(ROW, chips={"v6e": qc}))) == []
    bad = {
        "ctx": (dict(qc, ctx=2048), "records no ctx or slots"),
        "slots": (dict(qc, slots=8), "records no ctx or slots"),
        "8 users": (dict(qc, measured={"decode_tok_s_1_user": 14.2, "decode_tok_s_8_users": 99.0}),
                    "measured, when present, holds"),
        "bundle without the package": (dict(qc, kernel_path="bundle"),
                                       "kernel_path bundle, but no engine_kernels row for v6e has"),
        "kind": (dict(qc, record_kind="other"), "record_kind must be one of"),
        "no status": ({k: v for k, v in qc.items() if k != "status"}, "chips.v6e needs"),
    }
    assert problems(with_row(dict(ROW, chips={"v6e": {k: v for k, v in qc.items() if k != "measured"}}))) == []
    for name, (chip, want) in bad.items():
        got = problems(with_row(dict(ROW, chips={"v6e": chip})))
        assert any(want in x for x in got), (name, got)
    rendered = catalog.tpu_line(dict(ROW, chips={"v6e": qc}))
    assert rendered == "Runs on: TPU v6e-1" and "ctx" not in rendered and "slot" not in rendered


# ---------------------------------------------------------------- QUICK chip record (record_kind "quick")
# A row's chip says "passing" from a QUICK run: four zero-flag `mashbox run REPO:TAG --prompt P --max-tokens 32
# --temperature 0` runs compared with llama.cpp's greedy ids, its file pinned by the pick (repo, revision, file, sha256,
# bytes, card, header) with no CPU record behind it. The row's level is its own, Quick check, and its words say only that.

QUICK_Q = {"status": "passing", "accelerator": "v6e-1", "kernel_path": "exact", "record_kind": "quick",
           "measured": {"decode_tok_s_1_user": 0.52}}
QUICK_Q_LINE = "Runs on TPU v6e-1: short check against llama.cpp (4 prompts x 32 tokens)"


def test_the_guard_reads_the_quick_check_the_catalog_names():
    """the row's words come from catalog.py, the guard's bar from QUICK_KINDS: they name the same check"""
    assert (catalog.QUICK_KIND, catalog.QUICK_PROMPTS, catalog.QUICK_TOKENS) == ("quick", QUICK_RUNS, 32)
    assert QUICK_KINDS[catalog.QUICK_KIND][0] == catalog.QUICK_TOKENS


def test_a_quick_row_is_quick_check_and_says_only_what_was_checked():
    row = dict(ROW, chips={"v6e": QUICK_Q})
    assert catalog.level(row) == "Quick check" and catalog.quick_line(row) == QUICK_Q_LINE
    reg = with_row(row)
    r = next(x for x in catalog.gguf_rows(reg) if x["id"] == ROW["id"])
    assert (r["level"], r["tpu"], r["chips"], r["tok_s_1_user"]) == ("Quick check", QUICK_Q_LINE, ["TPU v6e-1"], None)
    assert f"Quick check<br>{QUICK_Q_LINE}" in catalog.render_markdown(reg)
    text = catalog.render_text(reg)
    assert "Quick check" in text and QUICK_Q_LINE in text
    # the level sits below Tested: a row with another passing chip run is Tested, the QUICK clause still shown
    both = dict(ROW, chips={"v6e": QUICK_Q, "v5e": dict(ROW["chips"]["v6e"], accelerator="v5e-1")})
    assert catalog.level(both) == "Tested" and catalog.quick_line(both) == QUICK_Q_LINE
    assert catalog.LEVELS.index("Tested") < catalog.LEVELS.index("Quick check") < catalog.LEVELS.index("Checked on CPU")
    # a failing QUICK run claims nothing
    assert catalog.quick_line(dict(ROW, chips={"v6e": dict(QUICK_Q, status="failing")})) is None


def test_the_schema_holds_a_quick_row_to_what_it_measured():
    assert problems(with_row(dict(ROW, chips={"v6e": QUICK_Q}))) == []
    bad = {
        "ctx": (dict(QUICK_Q, ctx=2048), "records no ctx or slots"),
        "8 users": (dict(QUICK_Q, measured={"decode_tok_s_1_user": 0.52, "decode_tok_s_8_users": 9.0}),
                    "measured, when present, holds"),
        "bundle without the package": (dict(QUICK_Q, kernel_path="bundle"),
                                       "kernel_path bundle, but no engine_kernels row for v6e has"),
    }
    for name, (chip, want) in bad.items():
        got = problems(with_row(dict(ROW, chips={"v6e": chip})))
        assert any(want in x for x in got), (name, got)


def test_record_guard_holds_a_quick_row(tmp_path):
    reg, ev = _fake_quick_check(tmp_path, kind="quick")
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


QUICK_Q_BAD = {
    "64 tokens": (_both(*(_in_place(f"run{i}", lambda v: v["argv"].__setitem__(5, "64")) for i in range(4))),
                  "a quick run is `mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M --prompt <prompt0.txt> --max-tokens 32"),
    "the 64-token kind's tool": (_set("seal", lambda v: [x.replace("cmp_q.py", "cmp_p1.py") for x in v]),
                                 "cmp_q.py is not in SHA256SUMS"),
    "kind mismatch": (_in_place("ev", lambda e: next(iter(e["gguf"].values()))["v6e"].__setitem__("kind", "quick-check")),
                      "the evidence kind is 'quick-check'"),
    "pin sha256": (_set("file", lambda v: "c" * 64 + v[64:]), "file-row.sha256 names another file"),
    "pin revision": (_set("plan", lambda v: v.replace(ROW["revision"], "1" * 40)), "plan-row.txt"),
    "seal sha256": (_set("seal", lambda v: [("0" * 64 + x[64:]) if x.endswith("cmp.json") else x for x in v]),
                    "sha256 differs from its seal line"),
    "decode number": (_quick_row(lambda r: r["measured"].update(decode_tok_s_1_user=14.2)),
                      "models.json says 14.2, the sealed runs 1-3 give 13.0"),
    "hidden divergence": (_set("ref2", _quick_ref_differs), "prompt 2: the compare says first difference None"),
    "near tie at the first token": (_both(_set("ref1", _quick_ref_first), _in_place("cmp", _quick_cmp_first)),
                                    "prompt 1 differs at the first token"),
}


@pytest.mark.parametrize("form", sorted(QUICK_Q_BAD))
def test_record_guard_names_each_bad_quick_form(tmp_path, form):
    edit, want = QUICK_Q_BAD[form]
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick")
    got = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(want in x for x in got), got


# ---------------------------------------------------------------- QUICK on the kernel package, and its n-gram gate entry
# A QUICK run with zero flags on a TPU loads the kernel package the catalog's engine_kernels row names (the engine fetches
# it, sha256-checked): kernel_path "bundle" when every weight ran on it. Run on one chip of a larger host (one process
# per chip), the row says so: accelerator "v6e-4" and chip_count 1, rendered "one chip of a TPU v6e-4". The same seal
# can hold the file's four `--speculation ngram` runs: their gate entry (speculation.ngram.<engine chip key>) is what
# `--speculation auto` reads, passed only when those runs pass the QUICK bars and decode at least as fast as the plain runs.

QUICK_K = {"status": "passing", "accelerator": "v6e-4", "chip_count": 1, "kernel_path": "bundle", "record_kind": "quick",
           "measured": {"decode_tok_s_1_user": 26.4}}
QUICK_K_LINE = ("Runs on one chip of a TPU v6e-4 with the engine's TPU kernels: short check against llama.cpp (4 prompts x "
                "32 tokens)")
QUICK_K0 = {k: v for k, v in QUICK_K.items() if k != "measured"}     # what the seal tool writes: no number
STEADY_ALONE = {"steady_decode_tok_s_1_user": 91.01, "steady_condition": catalog.STEADY_CONDITIONS["alone"]}


def test_a_kernel_path_quick_row_on_one_chip_of_a_host_says_one_chip():
    row = dict(ROW, chips={"v6e": QUICK_K})
    assert catalog.quick_line(row) == QUICK_K_LINE
    assert catalog.chips_run(row) == ["one chip of a TPU v6e-4"]
    assert catalog.speed(row) == (None, None)      # a QUICK run's own number is its fresh process's first request
    assert catalog.level(row) == "Quick check"
    reg = with_row(row, bundle=BUNDLE)
    r = next(x for x in catalog.gguf_rows(reg) if x["id"] == ROW["id"])
    assert (r["tpu"], r["tok_s_1_user"], r["tok_s_chip"]) == (QUICK_K_LINE, None, None)
    for view in (catalog.render_markdown(reg), catalog.render_text(reg), json.dumps(catalog.entries(reg))):
        assert "26.4" not in view
    # the whole host, or a one-chip host, keeps its plain name
    assert catalog.chips_run(dict(ROW, chips={"v6e": dict(QUICK_K, chip_count=4)})) == ["TPU v6e-4"]
    assert catalog.chips_run(dict(ROW, chips={"v6e": dict(QUICK_K, accelerator="v6e-1")})) == ["TPU v6e-1"]


def test_a_quick_row_shows_only_its_steady_decode_speed():
    """a QUICK run's own decode number is the first request of a fresh process (first-call cost in it): no view shows
    it, with or without a steady number beside it. A row's speed is its steady-decode number
    (measured.steady_decode_tok_s_1_user, from its own sealed record: one warm request, then the median of 5 requests of
    128 new tokens at 1 user), shown as "N tok/s, steady decode, 1 user" with the chip it ran on; its condition
    (steady_condition) goes in the JSON entries"""
    assert (catalog.STEADY_ROUNDS, catalog.STEADY_TOKENS, catalog.STEADY_SPREAD) == (5, 128, 0.02)
    assert SC.STEADY_CONDITIONS == catalog.STEADY_CONDITIONS and set(catalog.STEADY_CONDITIONS) == {"alone", "shared"}
    for measured in (STEADY_ALONE, dict(STEADY_ALONE, decode_tok_s_1_user=26.4)):
        row = dict(ROW, chips={"v6e": dict(QUICK_K, measured=measured)})
        assert catalog.speed(row) == (91.01, "one chip of a TPU v6e-4")
        reg = with_row(row, bundle=BUNDLE)
        assert problems(reg) == []
        r = next(x for x in catalog.gguf_rows(reg) if x["id"] == ROW["id"])
        assert (r["tpu"], r["tok_s_1_user"], r["tok_s_chip"], r["tok_s_condition"]) == (
            QUICK_K_LINE, 91.01, "one chip of a TPU v6e-4", catalog.STEADY_CONDITIONS["alone"])
        said = "91.01 tok/s, steady decode, 1 user, on one chip of a TPU v6e-4"
        for view in (catalog.render_markdown(reg), catalog.render_text(reg)):
            assert said in view and "26.4" not in view
        assert "26.4" not in json.dumps(catalog.entries(reg))
    # another chip record's number keeps its own words; a failing or exact-path QUICK shows no speed
    assert catalog.speed(dict(ROW, chips={"v6e": dict(QUICK_K, status="failing", reason="run", measured=STEADY_ALONE)})) == (
        None, None)
    assert catalog.speed(dict(ROW, chips={"v6e": dict(QUICK_Q, measured=STEADY_ALONE)})) == (None, None)


def test_the_schema_holds_a_quick_rows_steady_speed_to_its_form():
    ok = dict(QUICK_K, measured=STEADY_ALONE)
    assert problems(with_row(dict(ROW, chips={"v6e": ok}), bundle=BUNDLE)) == []
    shared = dict(QUICK_K, measured=dict(STEADY_ALONE, steady_condition=catalog.STEADY_CONDITIONS["shared"]))
    assert problems(with_row(dict(ROW, chips={"v6e": shared}), bundle=BUNDLE)) == []
    assert problems(with_row(dict(ROW, chips={"v6e": QUICK_K0}), bundle=BUNDLE)) == []     # no number at all
    bad = {
        "no condition": dict(QUICK_K, measured={"steady_decode_tok_s_1_user": 91.01}),
        "a condition only": dict(QUICK_K, measured={"steady_condition": catalog.STEADY_CONDITIONS["alone"]}),
        "another condition": dict(QUICK_K, measured=dict(STEADY_ALONE, steady_condition="steady decode, 1 user")),
        "zero": dict(QUICK_K, measured=dict(STEADY_ALONE, steady_decode_tok_s_1_user=0)),
        "a string": dict(QUICK_K, measured=dict(STEADY_ALONE, steady_decode_tok_s_1_user="91.01")),
        "failing": dict(QUICK_K, status="failing", reason="run", measured=STEADY_ALONE),
        "exact path": dict(QUICK_K, kernel_path="exact", measured=STEADY_ALONE),
    }
    for name, chip in bad.items():
        got = problems(with_row(dict(ROW, chips={"v6e": chip}), bundle=BUNDLE))
        assert any("a steady-decode speed is measured.steady_decode_tok_s_1_user" in x for x in got), (name, got)
    for name, m in {"empty": {}, "8 users": dict(STEADY_ALONE, decode_tok_s_8_users=9.0)}.items():
        got = problems(with_row(dict(ROW, chips={"v6e": dict(QUICK_K, measured=m)}), bundle=BUNDLE))
        assert any("measured, when present, holds" in x for x in got), (name, got)


def test_the_schema_takes_a_kernel_path_quick_row_and_its_chip_count():
    assert problems(with_row(dict(ROW, chips={"v6e": QUICK_K}), bundle=BUNDLE)) == []
    assert problems(with_row(dict(ROW, chips={"v6e": dict(QUICK_K, accelerator="v6e-1", chip_count=1)}),
                             bundle=BUNDLE)) == []
    bad = {
        "no package": (dict(QUICK_K), None, "kernel_path bundle, but no engine_kernels row for v6e has"),
        "zero chips": (dict(QUICK_K, chip_count=0), BUNDLE, "chip_count"),
        "more chips than the host": (dict(QUICK_K, chip_count=8), BUNDLE, "chip_count"),
        "a string": (dict(QUICK_K, chip_count="1"), BUNDLE, "chip_count"),
    }
    for name, (chip, bundle, want) in bad.items():
        got = problems(with_row(dict(ROW, chips={"v6e": chip}), bundle=bundle))
        assert any(want in x for x in got), (name, got)


QUICK_SPEC = {"ngram": {"v6e-1": {"status": "passed", "record": f"gguf/{ROW['id']}/ngram/v6e-1", "record_sha256": "e" * 64,
                                  "component_set_sha": "f" * 64, "spec_step_version": 1, "k_table": QUICK_K_TABLE}}}


def test_the_schema_holds_a_gguf_rows_speculation_gate_entries_to_their_form():
    ok = dict(ROW, chips={"v6e": QUICK_K}, speculation=QUICK_SPEC)
    assert problems(with_row(ok, bundle=BUNDLE)) == []
    no = copy.deepcopy(QUICK_SPEC)
    no["ngram"]["v6e-1"]["status"] = "not_passed"
    assert problems(with_row(dict(ok, speculation=no), bundle=BUNDLE)) == []

    def spec(fn):
        x = copy.deepcopy(QUICK_SPEC)
        fn(x)
        return x
    bad = {
        "mtp": (spec(lambda x: x.__setitem__("mtp", x["ngram"])), "speculation kinds"),
        "status": (spec(lambda x: x["ngram"]["v6e-1"].update(status="ok")), "status passed or not_passed"),
        "no record": (spec(lambda x: x["ngram"]["v6e-1"].pop("record")), "needs"),
        "a path": (spec(lambda x: x["ngram"]["v6e-1"].update(record="release/engine/x")), "record gguf/"),
        "sha": (spec(lambda x: x["ngram"]["v6e-1"].update(record_sha256="e" * 63)), "record_sha256"),
        "css": (spec(lambda x: x["ngram"]["v6e-1"].update(component_set_sha=None)), "component_set_sha"),
        "chip key": (spec(lambda x: x["ngram"].__setitem__("v6e", x["ngram"].pop("v6e-1"))), "chip key"),
        "k at an unmeasured bucket": (spec(lambda x: x["ngram"]["v6e-1"]["k_table"].update({"4": 3})), "k_table"),
        "k 0 where measured": (spec(lambda x: x["ngram"]["v6e-1"]["k_table"].update({"1": 0})), "k_table"),
        "another chip family": (spec(lambda x: x["ngram"].__setitem__("v5e-1", x["ngram"].pop("v6e-1"))),
                                "a passed gate entry needs the row's passing kernel-path QUICK"),
    }
    for name, (sp, want) in bad.items():
        got = problems(with_row(dict(ok, speculation=sp), bundle=BUNDLE))
        assert any(want in x for x in got), (name, got)
    exact = dict(ROW, chips={"v6e": QUICK_Q}, speculation=QUICK_SPEC)
    assert any("a passed gate entry needs the row's passing kernel-path QUICK" in x
               for x in problems(with_row(exact, bundle=BUNDLE)))


def test_record_guard_holds_a_kernel_path_quick_row(tmp_path):
    reg, ev = _fake_quick_check(tmp_path, kind="quick", path="bundle")
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


@pytest.mark.parametrize("kind,path", [("quick-check", "exact"), ("quick", "exact"), ("quick", "bundle")])
def test_record_guard_holds_a_quick_row_with_no_number(tmp_path, kind, path):
    """measured is optional on a QUICK row: pass/fail is the record's claim"""
    reg, ev = _fake_quick_check(tmp_path, _quick_row(lambda r: r.pop("measured")), kind=kind, path=path)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_reads_the_plain_runs_of_an_ngram_gate_entry_from_the_seal(tmp_path):
    """the n-gram gate compares the n-gram runs with the plain runs sealed in the same folder, not with a number on the
    row: a row with no number keeps its passed entry"""
    reg, ev = _fake_quick_check(tmp_path, _quick_row(lambda r: r.pop("measured")), kind="quick", path="bundle", ngram=True)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []


def test_record_guard_holds_a_quick_rows_steady_speed(tmp_path):
    reg, ev = _fake_quick_check(tmp_path, kind="quick", path="bundle", steady="alone")
    assert reg["gguf"][0]["chips"]["v6e"]["measured"] == {"steady_decode_tok_s_1_user": STEADY_TOK,
                                                          "steady_condition": catalog.STEADY_CONDITIONS["alone"]}
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    reg2, ev2 = _fake_quick_check(tmp_path / "b", kind="quick", path="bundle", steady="shared", ngram=True)
    assert evidence_problems(reg2, ev2, ROOT, tmp_path / "fit2") == []


def _steady_round(i, fn):
    return _in_place("steady_json", lambda s: fn(s["rounds"][i]))


STEADY_BAD = {
    "a typed number": (_quick_row(lambda r: r["measured"].update(steady_decode_tok_s_1_user=101.0)),
                       f"models.json says 101.0 steady tok/s, the sealed steady rounds give {STEADY_TOK}"),
    "a slow round": (_steady_round(2, lambda x: x.update(decode_s=1.397, ms_per_token=11.0)), "above the 2% bar"),
    "a round's ms": (_steady_round(1, lambda x: x.update(ms_per_token=9.0)), "is not its decode_s / (new tokens - 1)"),
    "four rounds": (_in_place("steady_json", lambda s: s["rounds"].pop()), "requests of 128 new tokens"),
    "64 tokens": (_steady_round(0, lambda x: x.update(new_tokens=64, decode_s=0.63)), "requests of 128 new tokens"),
    "other ids": (_steady_round(3, lambda x: x["ids"].__setitem__(5, 1)), "the same ids every round"),
    "exit code": (_set("steady_rc", lambda v: "rc 1\n"), "the steady run did not exit 0"),
    "another model": (_in_place("steady_json", lambda s: s.update(model="someone/Tiny-Qwen3-GGUF:Q8_0")),
                      "a steady run is `mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M"),
    "another argv": (_in_place("steady_json", lambda s: s.update(argv_tail=["--max-tokens", "8"])),
                     "a steady run is `mashbox run"),
    "another file": (_set("steady_stderr", lambda v: v.replace(ROW["sha256"], "b" * 64)),
                     f"the steady run read a file of sha256 {'b' * 64}"),
    "exact path": (_set("steady_stderr", lambda v: v.replace('"reference_bytes": 0', '"reference_bytes": 7')),
                   "ran 7 bytes on the exact path"),
    "another package": (_set("steady_wheel", lambda v: "d" * 64 + v[64:]), "the steady run loaded the kernel package"),
    "another code": (_set("steady_code", lambda v: "c" * 40 + "\n"), "the steady run's code is"),
    "two chips": (_in_place("steady_json", lambda s: s.update(devices=2)), "the steady run saw 2 devices"),
    "a shared host said alone": (_set("host", lambda v: v + "other 2026-10-30T15:20:00Z 2026-10-30T15:30:30Z chip 0\n"),
                                 "the sealed host log gives"),
    "an unwatched start": (_set("host", lambda v: v.replace("observed-from 2026-10-30T15:00:00Z",
                                                            "observed-from 2026-10-30T15:30:10Z")),
                           "did not watch the steady run from its start"),
    "a host line": (_set("host", lambda v: v + "note all fine\n"), "is not observed-from, window or other"),
    "not sealed": (_set("steady_seal", lambda v: [x for x in v if not x.endswith("steady.json")]),
                   "steady steady.json is not in"),
    "seal sha256": (_set("steady_seal", lambda v: [("0" * 64 + x[64:]) if x.endswith("host.txt") else x for x in v]),
                    "host.txt sha256 differs from its seal line"),
    "no record": (_in_place("ev", lambda e: next(iter(e["gguf"].values()))["v6e"].pop("steady")),
                  "a steady-decode speed needs its steady record"),
    "a record and no number": (_quick_row(lambda r: r.pop("measured")), "models.json says None steady tok/s"),
}


@pytest.mark.parametrize("form", sorted(STEADY_BAD))
def test_record_guard_names_each_bad_steady_form(tmp_path, form):
    edit, want = STEADY_BAD[form]
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle", steady="alone")
    got = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(want in x for x in got), got


def test_record_guard_holds_a_kernel_path_quick_row_with_its_ngram_gate_entry(tmp_path):
    reg, ev = _fake_quick_check(tmp_path, kind="quick", path="bundle", ngram=True)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    # the engine reads that entry: auto picks n-gram on the chip key it was measured on, with the measured k
    from mashbox.runtime.drafters import select as SEL
    g = reg["gguf"][0]
    rec = SEL.catalog_records(reg, dict(repo=g["repo"], revision=g["revision"], file=g["file"]))
    meta = dict(format="gguf", verify=True, mtp_head=False, experts=0, mtp_layers=0, draft_given=False, trained=[],
                trained_why=None, draft_models=[])
    kind, gated, why, gcss = SEL.pick(meta, "v6e-1", records=rec, later=True)
    assert (kind, gated, gcss) == ("ngram", True, QUICK_CSS), why
    assert SEL.pick(meta, "v6e-4", records=rec, later=True)[0] is None       # another chip key: not gated there


def _ngram_slower(v):
    """the n-gram runs decode slower than the plain runs (run 2: 0.8 s -> 6.0 s): (32 + 14 + 32) / 8.2 = 9.51"""
    v["generate_calls"][0]["timing"]["decode_s"] = 6.0


QUICK_BF16_LINE = ("mashbox: BF16 (25.0% of the weights) have no TPU kernel in this install and run on the exact float32 "
                   "reference path, which is slow (it rebuilds the weights at every step)\n")


def _bf16_dense16(i, share=25.0, bf16=300_000_000, ref=0, line=QUICK_BF16_LINE):
    """run i as a file with a BF16 output weight runs on 0.1.9: its plan names BF16 (share% of weights_gb 1.2) as having
    no TPU kernel, and its load puts bf16 bytes on the native bf16 path (Dense16) and ref bytes on the exact path"""
    def ev(x):
        if x["event"] == "gguf_plan":
            x.update(reference_share=share, weights_gb=1.2)
        if x["event"] == "weights":
            x.update(native_bf16_bytes=bf16, reference_bytes=ref)
    return _both(_quick_stderr(i, ev), _set(f"stderr{i}", lambda v: v + line))


# BF16 on Dense16 counts as served (Dense16 reads a BF16 weight as fast as XLA's own plain read of the same bytes,
# measured on a v6e chip); never a BF16 byte
# rebuilt on the exact path, another type beside it, or a plan share the Dense16 bytes do not cover
QUICK_BF16_BAD = {
    "bf16 on the exact path": (_bf16_dense16(1, bf16=0, ref=300_000_000),
                               "kernel_path bundle, and run1 ran 300000000 bytes on the exact path"),
    "bf16 share not on dense16": (_bf16_dense16(2, bf16=0), "kernel_path bundle, and run2 ran 25.0% of the weights on the "
                                                            "exact path"),
    "bf16 and another type": (_bf16_dense16(0, line=QUICK_BF16_LINE.replace("BF16 (", "BF16, Q6_K (")),
                              "kernel_path bundle, and run0 printed 'have no TPU kernel'"),
    "bf16 share past its bytes": (_bf16_dense16(3, share=30.0), "kernel_path bundle, and run3 ran 30.0% of the weights on "
                                                                "the exact path"),
}


QUICK_K_BAD = {
    "reference share": (_quick_stderr(1, lambda x: x.update(reference_share=2.5) if x["event"] == "gguf_plan" else None),
                        "kernel_path bundle, and run1 ran 2.5% of the weights on the exact path"),
    "reference bytes": (_quick_stderr(2, lambda x: x.update(reference_bytes=7) if x["event"] == "weights" else None),
                        "kernel_path bundle, and run2 ran 7 bytes on the exact path"),
    "no kernel bytes": (_quick_stderr(0, lambda x: x.update(kernel_bytes=0) if x["event"] == "weights" else None),
                        "kernel_path bundle, and run0 ran 0 bytes on a kernel"),
    "other kernel types": (_quick_stderr(3, lambda x: x.update(kernel_types=["Q4_K"]) if x["event"] == "gguf_plan" else None),
                           "kernel_path bundle, and run3 ran kernel types ['Q4_K'], the engine_kernels row for v6e has"),
    "a type on the exact path": (_set("stderr1", lambda v: v + "mashbox: Q6_K (1.0% of the weights) have no TPU kernel "
                                                         "in this install and run on the exact float32 reference path\n"),
                                 "kernel_path bundle, and run1 printed 'have no TPU kernel'"),
    "the package did not load": (_set("stderr0", lambda v: v + "mashbox: kernel package mashbox_engine_kernels did not "
                                                         "load (x); every type runs on the exact path\n"),
                                 "kernel_path bundle, and run0 printed 'did not load'"),
    "another package": (_set("kernel_wheel", lambda v: "d" * 64 + v[64:]), "kernel_path bundle, and the kernel package the runs "
                                                                        "loaded is"),
    "no package sealed": (_set("seal", lambda v: [x for x in v if not x.endswith("kernel-wheel.sha256")]),
                          "kernel-wheel.sha256 is not in SHA256SUMS"),
    "more chips": (_in_place("run1", lambda v: v.update(devices=4)), "the row runs on 1 chip of a TPU v6e-4, and run1 saw 4"),
    "no chip count": (_quick_row(lambda r: r.pop("chip_count")), "the row runs on 4 chips of a TPU v6e-4, and run0 saw 1"),
    "no device count": (_in_place("run2", lambda v: v.pop("devices")), "run2 recorded no device count"),
    **QUICK_BF16_BAD,
}


@pytest.mark.parametrize("form", sorted(QUICK_K_BAD))
def test_record_guard_names_each_bad_kernel_path_quick_form(tmp_path, form):
    edit, want = QUICK_K_BAD[form]
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle")
    got = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(want in x for x in got), got


def _spec_entry(fn):
    return _in_place("spec", fn)


QUICK_N_BAD = {
    "slower than plain": (_in_place("ngram-run2", _ngram_slower),
                          "status passed, and the sealed runs give not_passed (n-gram 9.51 tok/s below plain 13.0)"),
    "a hidden divergence": (_in_place("ngram-run2", lambda v: v["generate_calls"][0]["ids"][0].__setitem__(5, 6)),
                            "ngram: prompt 2: the compare says first difference None, the sealed ids give 5"),
    "said divergence": (_both(_in_place("ngram-run2", lambda v: v["generate_calls"][0]["ids"][0].__setitem__(5, 6)),
                              _in_place("ngram_cmp", lambda c: (c["rows"][2].update(first_diff=5, ref_gap=1.5,
                                                                                     near_tie=False, kind="other",
                                                                                     ids_ours=c["rows"][2]["ids_ours"][:5] + [6] + c["rows"][2]["ids_ours"][6:]),
                                                                c["summary"].update(full=2, other=1, PASS=False)))),
                        "status passed, and the sealed runs give not_passed (1 other divergence"),
    "plain flags": (_in_place("ngram-run0", lambda v: v["argv"].__delitem__(slice(4, 6))),
                    "ngram-run0 ran `mashbox run someone/Tiny-Qwen3-GGUF:Q4_K_M --prompt"),
    "env": (_in_place("ngram-run1", lambda v: v["env"].update(MASHBOX_GGUF_KERNELS="off")),
            "ngram-run1 ran with MASHBOX_GGUF_KERNELS=off"),
    "not n-gram": (_quick_stderr_n(3, lambda x: x.update(method="draft_model") if x["event"] == "speculation" else None),
                   "ngram-run3 ran speculation"),
    "the plain arm speculated": (_set("stderr1", lambda v: v + json.dumps(
        {"event": "speculation", "used": True, "method": "ngram", "draft_k": 3, "gated": True}) + "\n"),
        "plain run1 ran speculation ngram (used True, gated True); the gate compares n-gram with a plain arm that"),
    "k": (_spec_entry(lambda e: e["k_table"].update({"1": 4})), "k_table's 1-row k 4 is not the runs' draft_k 3"),
    "css": (_spec_entry(lambda e: e.update(component_set_sha="0" * 64)), "the runs' component set is " + QUICK_CSS),
    "step": (_spec_entry(lambda e: e.update(spec_step_version=99)), "the runs measured speculation step version"),
    "chip key": (_in_place("ngram-run1", lambda v: v.update(chip_key="v6e-4")), "ngram-run1 ran on chip key v6e-4"),
    "record sha": (_spec_entry(lambda e: e.update(record_sha256="0" * 64)), "record_sha256 0000"),
    "record name": (_spec_entry(lambda e: e.update(record="gguf/x/ngram/v6e-1")), "the record is gguf/x/ngram/v6e-1"),
    "not_passed hides a pass": (_spec_entry(lambda e: e.update(status="not_passed")),
                                "status not_passed, and the sealed runs give passed"),
    "no evidence": (_in_place("ev", lambda e: e["gguf"][ROW["id"]].pop("speculation")), "ngram v6e-1: no evidence entry"),
    "not sealed": (_set("seal", lambda v: [x for x in v if not x.endswith("ngram-run2.stderr")]),
                   "ngram-run2.stderr is not in SHA256SUMS"),
}


def _seal_inputs(base, src, krow, ngram=True, tf=()):
    """the folders tools/gguf_quick_seal.py reads, cut from a fake seal folder src: runs, compare, chip kit, kit, pin,
    and the n-gram runs and compare, each with its own SHA256SUMS; the kernel wheel line and the code's models.json"""
    def folder(name, files, sums="SHA256SUMS"):
        d = base / name
        for dst, data in files.items():
            (d / dst).parent.mkdir(parents=True, exist_ok=True)
            (d / dst).write_bytes(data)
        (d / sums).write_text("".join(f"{_sha(d / n)}  {n}\n" for n in sorted(files)))
        return d
    rd = lambda n: (src / n).read_bytes()      # noqa: E731
    runs = folder("runs", {f"run{i}{x}": rd(f"run{i}{x}") for i in range(4) for x in (".json", ".rc", ".stdout", ".stderr")})
    cmp = folder("cmp", {"cmp.json": rd("cmp.json"), **{f"ref{i}.ids": rd(f"ref{i}.ids") for i in range(4)}})
    chip = folder("chip", {n: rd(n) for n in ("setup.log", "freeze.txt", "accelerator-type.txt")}, "SHA256SUMS-chipkit")
    kit = folder("kit", {"CODE_SHA": rd("CODE_SHA"), **{f"in/prompt{i}.txt": rd(f"prompt{i}.txt") for i in range(4)}},
                 "SHA256SUMS-kit")
    card = json.loads(rd("card-api.json"))
    pin = folder("pin", {"header.bin": rd("header-row.gguf"), "card.json": rd("card-api.json"), "pin.json": json.dumps(dict(
        repo=card["id"], revision=card["sha"], file=ROW["file"], tag=ROW["tag"], lfs_sha256=ROW["sha256"],
        bytes=FIXTURE.stat().st_size)).encode()})
    out = dict(runs=runs, compare=cmp, chip=chip, kit=kit, pin=pin, tool=src / "cmp_q.py")
    if ngram:
        out["ngram_runs"] = folder("nruns", {f"run{i}{x}": rd(f"ngram-run{i}{x}") for i in range(4)
                                             for x in (".json", ".rc", ".stdout", ".stderr")})
        out["ngram_compare"] = folder("ncmp", {"cmp.json": rd("ngram-cmp.json"),
                                               **{f"ref{i}.ids": rd(f"ref{i}.ids") for i in range(4)}})
    if tf:
        out["tf_runs"] = folder("truns", {f"tf-run{i}{x}": rd(f"tf-run{i}{x}") for i in tf
                                          for x in (".json", ".rc", ".stdout", ".stderr")})
        out["tf_compare"] = folder("tcmp", {"tf-cmp.json": rd("tf-cmp.json")})
    (base / "kernel-wheel.sha256").write_bytes(rd("kernel-wheel.sha256"))
    (base / "models.json").write_text(json.dumps({"engine_kernels": [krow]}))
    return out


def _run_seal(base, ins, out, row_id=None):
    spec = importlib.util.spec_from_file_location("gguf_quick_seal_t", ROOT / "tools" / "gguf_quick_seal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["--kind", "quick", "--runs", str(ins["runs"]), "--compare", str(ins["compare"]), "--chip", str(ins["chip"]),
            "--kit", str(ins["kit"]), "--tool", str(ins["tool"]), "--pin", str(ins["pin"]), "--tag", "row",
            "--out", str(out), "--kernel-wheel", str(base / "kernel-wheel.sha256"), "--models", str(base / "models.json")]
    if "ngram_runs" in ins:
        argv += ["--ngram-runs", str(ins["ngram_runs"]), "--ngram-compare", str(ins["ngram_compare"])]
    if "tf_runs" in ins:
        argv += ["--tf-runs", str(ins["tf_runs"]), "--tf-compare", str(ins["tf_compare"])]
    if row_id:
        argv += ["--row-id", row_id]
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main(argv)
    return json.loads(buf.getvalue())


def test_the_quick_seal_tool_writes_a_kernel_path_row_and_gate_entry_the_guard_holds(tmp_path):
    """tools/gguf_quick_seal.py on a bundle QUICK's plain and n-gram runs: it reads kernel_path, chip_count, the chip
    key, component set, step version and k from the runs' own events, seals both sets in one folder, and the row block,
    gate entry and evidence it prints pass the record guard as they are"""
    reg, ev = _fake_quick_check(tmp_path, kind="quick", path="bundle", ngram=True)
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0])
    got = _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])
    assert got["chips"] == {"v6e": QUICK_K0}        # pass/fail only: the runs' first-request number is never a row's
    ent = got["speculation"]["ngram"]["v6e-1"]
    assert (ent["status"], ent["component_set_sha"], ent["k_table"], ent["record"]) == (
        "passed", QUICK_CSS, QUICK_K_TABLE, f"gguf/{ROW['id']}/ngram/v6e-1")
    g = reg["gguf"][0]
    g["chips"], g["speculation"] = got["chips"], got["speculation"]
    ev["gguf"][g["id"]] = {**got["evidence"]}
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    # slower n-gram runs: the same tool seals a not_passed entry, which the guard also holds
    reg2, ev2 = _fake_quick_check(tmp_path / "b", _in_place("ngram-run2", _ngram_slower), kind="quick", path="bundle",
                                  ngram=True)
    root2 = Path(ev2["records_root"])
    got2 = _run_seal(tmp_path / "b" / "in", _seal_inputs(tmp_path / "b" / "in", root2 / "p1row", reg2["engine_kernels"][0]),
                     root2 / "seal", row_id=ROW["id"])
    assert got2["speculation"]["ngram"]["v6e-1"]["status"] == "not_passed"
    g2 = reg2["gguf"][0]
    g2["chips"], g2["speculation"] = got2["chips"], got2["speculation"]
    ev2["gguf"][g2["id"]] = {**got2["evidence"]}
    assert evidence_problems(reg2, ev2, ROOT, tmp_path / "fit2") == []


def test_the_quick_seal_tool_refuses_a_run_set_with_weights_on_the_exact_path(tmp_path):
    """MIXED (some weights on the kernel package, some on the exact path): no kernel-path row, no exact row"""
    reg, ev = _fake_quick_check(tmp_path, _quick_stderr(1, lambda x: x.update(reference_share=2.5, reference_bytes=7) if
                                                         x["event"] in ("gguf_plan", "weights") else None),
                                kind="quick", path="bundle", ngram=False)
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0], ngram=False)
    with pytest.raises(SystemExit) as x:
        _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])
    assert "mixed" in str(x.value) and not (root / "seal").exists()


def test_bf16_on_dense16_counts_as_served_in_the_seal_and_the_guard(tmp_path):
    """a file whose only exact-path type is BF16, every byte of it on the native bf16 path (Dense16) and 0 bytes on the
    exact path, in all four runs: the seal tool writes a kernel_path bundle row and the record guard holds it"""
    edit = _both(*(_bf16_dense16(i) for i in range(4)))
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle", ngram=False)
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0], ngram=False)
    got = _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])
    assert got["chips"]["v6e"]["kernel_path"] == "bundle"
    g = reg["gguf"][0]
    g["chips"] = got["chips"]
    ev["gguf"][g["id"]] = {**got["evidence"]}
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit2") == []


def _later_divergence(c):
    """an edit of the compare: prompt 3's difference at step 31 has llama.cpp's gap 0.9 (an other divergence)"""
    c["rows"][3].update(ref_gap=0.9, near_tie=False, kind="other")
    c["summary"].update(near=0, other=1, PASS=False)


def test_a_failing_quick_seal_names_its_reason_class_and_the_guard_recomputes_it(tmp_path):
    """the seal of a QUICK that did not pass writes status failing and the reason class its compare gives (run, first
    token, divergence, near tie); the record guard recomputes the class from the sealed compare, never typed"""
    reg, ev = _fake_quick_check(tmp_path, _set("cmp", _later_divergence), kind="quick", path="bundle", ngram=False)
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0], ngram=False)
    got = _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])
    assert (got["chips"]["v6e"]["status"], got["chips"]["v6e"]["reason"]) == ("failing", "divergence")
    g = reg["gguf"][0]
    g["chips"] = got["chips"]
    ev["gguf"][g["id"]] = {**got["evidence"]}
    assert evidence_problems(reg, ev, ROOT, tmp_path / "fit") == []
    g["chips"]["v6e"]["reason"] = "near tie"
    assert any("the sealed compare gives divergence" in x for x in evidence_problems(reg, ev, ROOT, tmp_path / "fit2"))


QUICK_T0 = _both(_set("ref1", _quick_ref_first), _in_place("cmp", _quick_cmp_first))   # prompt 1: a near tie at token 0
QUICK_T0_TF = {1: {"token": 6, "ids": list(range(201, 232))}}     # its forced run: llama.cpp's tokens 2..32


def _t0_seal(tmp_path, forced, edit=QUICK_T0, row_edit=None):
    """a bundle QUICK whose prompt 1 meets a near tie at the first token (gap 0.1), with the forced runs given; -> (reg,
    ev, the seal tool's output)"""
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle", forced=forced)
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0], ngram=False, tf=tuple(forced or ()))
    return reg, ev, _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])


def _guard_with(reg, ev, got, tmp_path, name):
    g = reg["gguf"][0]
    g["chips"] = copy.deepcopy(got["chips"])
    ev["gguf"][g["id"]] = {**got["evidence"]}
    return evidence_problems(reg, ev, ROOT, tmp_path / name)


def test_a_first_token_near_tie_passes_when_its_forced_run_agrees(tmp_path):
    """The token-0 near-tie rule: a first-token difference under llama.cpp's own
    near-tie gap counts as a near tie, and the rest is checked by one forced run (the chip's prompt ids + llama.cpp's
    first token, the plain argv) with 0 other divergences on tokens 2..32: the seal passes it and the guard holds it"""
    reg, ev, got = _t0_seal(tmp_path, QUICK_T0_TF)
    assert got["chips"]["v6e"]["status"] == "passing" and "reason" not in got["chips"]["v6e"]
    assert _guard_with(reg, ev, got, tmp_path, "fit") == []
    # the same seal without its forced run: failing, reason near tie; a row that says passing is refused
    reg2, ev2, got2 = _t0_seal(tmp_path / "b", None)
    assert (got2["chips"]["v6e"]["status"], got2["chips"]["v6e"]["reason"]) == ("failing", "near tie")
    assert _guard_with(reg2, ev2, got2, tmp_path / "b", "fit") == []
    got2["chips"]["v6e"].update(status="passing")
    got2["chips"]["v6e"].pop("reason")
    assert any("prompt 1 differs at the first token" in x for x in _guard_with(reg2, ev2, got2, tmp_path / "b", "fit2"))


def test_a_forced_run_that_diverges_fails_as_a_divergence(tmp_path):
    """the forced continuation meets an other divergence (llama.cpp's gap 0.9 at step 5): failing, reason divergence,
    recomputed by the guard; a row that says passing is refused"""
    ids = list(range(201, 232))
    ids[4] = 7
    reg, ev, got = _t0_seal(tmp_path, {1: {"token": 6, "ids": ids, "gap": 0.9}})
    assert (got["chips"]["v6e"]["status"], got["chips"]["v6e"]["reason"]) == ("failing", "divergence")
    assert _guard_with(reg, ev, got, tmp_path, "fit") == []
    got["chips"]["v6e"].update(status="passing")
    got["chips"]["v6e"].pop("reason")
    assert any("forced run" in x for x in _guard_with(reg, ev, got, tmp_path, "fit2"))


QUICK_T0_BAD = {
    "another forced token": ({1: {"token": 7, "ids": list(range(201, 232))}}, None,
                             "forced run 1 appended 7, llama.cpp's first token is 6"),
    "another argv": (QUICK_T0_TF, _in_place("tfrun1", lambda v: v["argv"].insert(3, "--ctx")), "forced run 1 ran `mashbox"),
    "other prompt ids": (QUICK_T0_TF, _in_place("tfrun1", lambda v: v["generate_calls"][0]["prompt_ids"][0].insert(0, 9)),
                         "forced run 1 ran prompt ids"),
}


@pytest.mark.parametrize("form", sorted(QUICK_T0_BAD))
def test_the_guard_and_the_seal_refuse_each_bad_forced_run(tmp_path, form):
    forced, extra, want = QUICK_T0_BAD[form]
    edit = _both(QUICK_T0, extra) if extra else QUICK_T0
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle", forced=forced)
    assert any(want in x for x in evidence_problems(reg, ev, ROOT, tmp_path / "fit")), want
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0], ngram=False, tf=(1,))
    with pytest.raises(SystemExit) as x:
        _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])
    assert "forced run 1" in str(x.value) and not (root / "seal").exists()


@pytest.mark.parametrize("form", sorted(QUICK_BF16_BAD))
def test_the_quick_seal_tool_refuses_each_bad_bf16_form(tmp_path, form):
    edit, _ = QUICK_BF16_BAD[form]
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle", ngram=False)
    root = Path(ev["records_root"])
    ins = _seal_inputs(tmp_path / "in", root / "p1row", reg["engine_kernels"][0], ngram=False)
    with pytest.raises(SystemExit) as x:
        _run_seal(tmp_path / "in", ins, root / "seal", row_id=ROW["id"])
    assert "mixed" in str(x.value) and not (root / "seal").exists()


@pytest.mark.parametrize("form", sorted(QUICK_N_BAD))
def test_record_guard_names_each_bad_ngram_gate_form(tmp_path, form):
    edit, want = QUICK_N_BAD[form]
    reg, ev = _fake_quick_check(tmp_path, edit, kind="quick", path="bundle", ngram=True)
    got = evidence_problems(reg, ev, ROOT, tmp_path / "fit")
    assert any(want in x for x in got), got
