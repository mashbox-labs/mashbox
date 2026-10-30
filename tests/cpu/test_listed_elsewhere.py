"""Listed, runs elsewhere: a model `mashbox run` does not start is still in the catalog (models.json elsewhere[]), with
the device its files are made for and a how-to-run link instead of a run command.

One row type for every such model (weights-only repos, a port that runs on another device such as the Apple Neural
Engine): the validator holds its fields, the release rule takes the row's own device (no TPU chip run is asked of it),
every view renders it from models.json, and the online check reads its repo, pinned revision, licence files and the
heading its link points at. The rows here are made up; models.json's own rows go through the same checks in T0
(the offline check) and online.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sync_check_elsewhere", ROOT / "tools" / "hf_github_sync_check.py")
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)
REG = json.loads((ROOT / "models.json").read_text())
ORG = REG["hf"]["org"]
REV = "a" * 40
LICENCE = "Example licence text\n"


def row(**kw):
    r = {"id": "example-model", "name": "Example Model", "summary": "Text to speech.", "device": "Apple Neural Engine",
         "how_to_run": f"https://huggingface.co/{ORG}/Example-Model-ANE#run", "hf_repo": f"{ORG}/Example-Model-ANE",
         "hf_revision": REV,
         "upstream": {"name": "Example Model", "maker": "Example Lab", "hf": "example-lab/Example-Model",
                      "revision": "b" * 40, "license": "apache-2.0",
                      "license_sha256": hashlib.sha256(LICENCE.encode()).hexdigest()}}
    r.update(kw)
    return r


def with_rows(*rows):
    reg = copy.deepcopy(REG)
    reg["elsewhere"] = list(rows)
    return reg


def problems_of(reg, release=False):
    p = SC.Problems()
    SC.validate(reg, ROOT, p, release=release)
    return [str(x) for x in p]


def test_an_elsewhere_row_is_valid_and_the_release_rule_takes_its_own_device():
    """A row with a device and a how-to-run link passes the release check with no chip run: the release rule asks a
    models[] row for a passing TPU chip, and an elsewhere[] row for its own device (DEVICES) and its pinned revision."""
    assert "Apple Neural Engine" in SC.DEVICES and "TPU" in SC.DEVICES
    assert problems_of(with_rows(row()), release=True) == []
    assert problems_of(with_rows(row(device="TPU", how_to_run=f"https://huggingface.co/{ORG}/Example-Model-ANE#load")),
                       release=True) == []


def test_an_elsewhere_row_names_its_device_its_link_and_its_pinned_revision():
    cases = {"device": row(device="a laptop"), "how_to_run": row(how_to_run="http://example.com/run"),
             "hf_revision": row(hf_revision=None), "hf_repo": row(hf_repo="someone-else/Example-Model-ANE")}
    for key, r in cases.items():
        got = problems_of(with_rows(r), release=True)
        assert any("elsewhere" in p and key in p for p in got), (key, got)
    r = row()
    del r["how_to_run"]
    assert any("missing key 'how_to_run'" in p for p in problems_of(with_rows(r)))


def test_an_elsewhere_row_copies_its_licence_as_stated_upstream():
    r = row()
    r["upstream"]["license"] = "unknown"
    assert any("licence not confirmed" in p for p in problems_of(with_rows(r)))
    r["upstream"].update(license="other")
    assert any("license_name and license_link" in p for p in problems_of(with_rows(r)))
    r["upstream"].update(license_name="example-licence", license_link="https://example.com/licence")
    assert problems_of(with_rows(r)) == []


def test_an_elsewhere_row_has_no_run_path_and_no_numbers():
    """No `mashbox run` command, no chip rows and no results: what it runs on and how fast are the linked page's to say."""
    for key, val in (("run", "mashbox run example-model"), ("chips", {}), ("results", []), ("path", "src/x")):
        got = problems_of(with_rows(row(**{key: val})))
        assert any("elsewhere" in p and repr(key) in p for p in got), (key, got)


def test_ids_are_unique_across_models_gguf_and_elsewhere():
    taken = REG["models"][0]["id"]
    assert any("duplicate id" in p for p in problems_of(with_rows(row(id=taken))))
    assert any("duplicate id" in p for p in problems_of(with_rows(row(), row(hf_repo=f"{ORG}/Other-ANE"))))
    if REG.get("gguf"):
        assert any("duplicate id" in p for p in problems_of(with_rows(row(id=REG["gguf"][0]["id"]))))


def test_every_view_lists_an_elsewhere_row_with_its_device_and_link():
    """README, llms.txt and the docs catalog list the row from models.json: name, maker, device and the how-to-run link,
    and no `mashbox run` line for it. A registry with no elsewhere[] renders as before."""
    r = row()
    reg = with_rows(r)
    views = {"README": SC.readme_block(reg), "llms.txt": SC.llms_block(reg), "docs": SC._lists().render_markdown(reg)}
    for where, text in views.items():
        assert r["name"] in text and r["how_to_run"] in text and r["device"] in text, where
        assert r["upstream"]["maker"] in text, where
        assert f"mashbox run {r['id']}" not in text, where
    plain = copy.deepcopy(REG)
    plain.pop("elsewhere", None)
    assert SC.readme_block(plain) == SC.readme_block(with_rows()), "an empty list renders as no list"
    assert SC._lists().ELSEWHERE not in SC.readme_block(plain)


class Hub:
    """The Hub as a logged-out reader sees it: the org listing, the repo at its head, the card and licence at the
    revision. Every other URL is a 404 (not this test's subject)."""

    def __init__(self, r, head=REV, card="# Example Model\n\n## Run\n\nSteps.\n", licence=LICENCE):
        repo = r["hf_repo"]
        self.pages = {f"https://huggingface.co/api/models?author={ORG}&limit=1000": json.dumps([{"id": repo}]),
                      f"https://huggingface.co/api/models/{repo}": json.dumps({"id": repo, "sha": head}),
                      f"https://huggingface.co/{repo}/raw/{REV}/README.md": card,
                      f"https://huggingface.co/{repo}/raw/{REV}/LICENSE": licence}

    def get(self, url):
        return (200, self.pages[url]) if url in self.pages else (404, "")

    def status(self, url):
        return 200


def online(reg, hub, repo=row()["hf_repo"]):
    """the problems about the org's listing and about the repo; models[] rows have their own online tests"""
    p = SC.Problems()
    reg = copy.deepcopy(reg)
    reg["models"] = []
    SC.check_online(reg, {}, hub, p)
    return [str(x) for x in p if str(x).startswith(("hf: ", f"hf {repo}"))]


def test_online_an_elsewhere_repo_is_listed_pinned_and_its_link_lands_on_a_heading():
    r = row()
    reg = with_rows(r)
    assert online(reg, Hub(r)) == []
    assert any("not in models.json" in p for p in online(with_rows(), Hub(r))), "an unlisted org repo passes"
    assert any("not the pinned revision" in p for p in online(reg, Hub(r, head="c" * 40)))
    assert any("#run" in p for p in online(reg, Hub(r, card="# Example Model\n\n## Load\n"))), "a dead anchor passes"
    assert any("LICENSE" in p for p in online(reg, Hub(r, licence="edited\n"))), "a changed licence file passes"


DOCS = "https://mashbox.gg/docs/models/example-model"
FIG = {"decode": {"text": "11.1 tokens/s", "what": "decode speed, one conversation", "device": "a test Mac",
                  "date": "2026-10-30"}}


def test_a_listed_rows_figures_fill_number_markers_and_come_from_the_private_map():
    """A listed row's measured numbers live where every row's do (figures by name, the private number map): a docs page
    shows them through number markers, so the page, the map and the card each hold one copy that is checked against the
    others, never a number typed twice."""
    reg = with_rows(row(figures=FIG))
    assert SC.figure_text(reg, "example-model", "decode") == "11.1 tokens/s"
    p = SC.Problems()
    marked = "decodes at <!-- r:example-model/decode -->9 tokens/s<!-- /r --> here"
    assert SC.fill_numbers(reg, marked, "page", p) == marked.replace("9 tokens/s", "11.1 tokens/s") and not list(p)
    assert SC.figure_text(reg, "example-model", "prefill") is None
    plain = with_rows(row())
    SC.merge_figures(plain, {"figures": {"example-model": FIG}})
    assert plain["elsewhere"][0]["figures"] == FIG


def test_a_listed_row_that_links_its_docs_page_has_the_page_and_a_sidebar_line(tmp_path):
    """how_to_run on the row's own docs page (mashbox.gg/docs/models/<id>): docs/models/<id>.md is in the tree and the
    docs sidebar lists it; a row whose how_to_run is its Hugging Face card gets no sidebar line."""
    reg = with_rows(row(how_to_run=DOCS))
    p = SC.Problems()
    SC.elsewhere_page_problems(reg, tmp_path, p)
    assert any("docs/models/example-model.md" in str(x) for x in p), list(p)
    (tmp_path / "docs" / "models").mkdir(parents=True)
    (tmp_path / "docs" / "models" / "example-model.md").write_text("# Example Model\n")
    p = SC.Problems()
    SC.elsewhere_page_problems(reg, tmp_path, p)
    assert not list(p)
    other = SC.Problems()
    SC.elsewhere_page_problems(with_rows(row(how_to_run="https://mashbox.gg/docs/models/someone-else")), tmp_path, other)
    assert any("its own page" in str(x) for x in other), list(other)
    assert "* [Example Model](models/example-model.md)" in SC.sidebar_block(reg).splitlines()
    assert "example-model" not in SC.sidebar_block(with_rows(row()))


def test_online_a_listed_rows_figures_are_on_its_card_at_the_pinned_revision():
    r = row(figures=FIG)
    reg = with_rows(r)
    on_card = "# Example Model\n\n## Run\n\nDecoding: 11.1 tokens/s.\n"
    assert online(reg, Hub(r, card=on_card)) == []
    got = online(reg, Hub(r, card="# Example Model\n\n## Run\n\nDecoding: 10.9 tokens/s.\n"))
    assert any("decode" in p and "11.1 tokens/s" in p for p in got), got
