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
