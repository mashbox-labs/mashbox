"""The Hugging Face org card and the "TPU models" collection, checked online against models.json like the profile.

The org card is the org's static README Space, an overview like the GitHub org profile: no models
block, no model named, no count, and a family line with exactly models.json's GGUF families (overview_problems). The
"TPU models" collection holds every live TPU port.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sync_check_org_card", ROOT / "tools" / "hf_github_sync_check.py")
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)
REG = json.loads((ROOT / "src" / "mashbox" / "models.json").read_text())
ORG = REG["hf"]["org"]
CARD_URL = f"https://huggingface.co/spaces/{ORG}/README/raw/main/README.md"
COLL_URL = f"https://huggingface.co/api/collections?owner={ORG}&limit=100"

# The org card as it read before 2026-10-30: a hand-typed table of one model and a "Coming soon" line naming another.
OLD_CARD = (f"**{REG['company_line']}**\n\n| Model | What it does | Chip tests | Measured |\n|---|---|---|---|\n"
            f"| [Qwen3.8-27B-TPU](https://huggingface.co/{ORG}/Qwen3.8-27B-TPU) | text in, text out | TPU v6e | "
            "**130.1 vs 1,606.7 s** first start to ready |\n\nComing soon: Hy-Nomas by Tencent Hunyuan.\n")
INSTALL = "```bash\n~/.local/bin/mashbox run qwen3-8-27b\n```\n"


def new_card(reg=REG):
    fams = SC._lists().gguf_families(reg)
    return (f"**{reg['company_line']}**\n\nOur engine runs GGUF files from the community as they are. Families covered "
            f"so far: {', '.join(fams[:-1])} and {fams[-1]}.\n\n{INSTALL}\n"
            f"{SC.REPOS_START}\n{SC.repos_text(reg)}\n{SC.REPOS_END}\n")


def listed_in(reg, title):
    """our repos models.json puts in a collection that is neither a kind's nor a family's: elsewhere[] rows by device"""
    return sorted(r["hf_repo"] for r in reg.get("elsewhere") or [] if SC.DEVICE_INFO[r["device"]]["collection"] == title)


def collections(ids, reg=REG):
    extra = [{"title": d["collection"], "items": [{"type": "model", "id": i} for i in listed_in(reg, d["collection"])]}
             for d in SC.DEVICE_INFO.values() if listed_in(reg, d["collection"])]
    return json.dumps([{"title": "TPU models", "items": [{"type": "model", "id": i} for i in ids]},
                       {"title": "Qwen on TPU", "items": [{"type": "model", "id": f"{ORG}/Qwen3.8-27B-TPU"}]}] + extra)


class Fetch:
    """Answers only the org card and the collections; every other URL is a 404 (those checks are not under test)."""
    def __init__(self, card, coll):
        self.pages = {CARD_URL: card, COLL_URL: coll}

    def get(self, url):
        return (200, self.pages[url]) if url in self.pages else (404, "")


def org_problems(card, coll):
    problems = SC.Problems()
    SC.check_online(REG, {}, Fetch(card, coll), problems)
    return [str(p) for p in problems if "org card" in str(p) or "collections" in str(p)]


def test_the_old_org_card_and_collection_fail_online():
    got = org_problems(OLD_CARD, collections([f"{ORG}/Qwen3.8-27B-TPU"]))
    assert any("Hy-Nomas" in p for p in got), got                # "Coming soon: Hy-Nomas" names a model in prose
    assert any("lacks" in p and "Hy-Nomas-TPU" in p for p in got), got


def test_the_overview_org_card_and_full_collection_pass_online():
    assert org_problems(new_card(), collections(sorted(SC.tpu_ports(REG)) + ["tencent/Hy-Nomas"])) == []


def test_a_models_block_or_a_count_on_the_card_fails():
    full = collections(sorted(SC.tpu_ports(REG)))
    with_block = new_card() + f"\n{SC.MARK_START}\n{SC.readme_block(REG)}\n{SC.MARK_END}\n"
    assert any("block" in p for p in org_problems(with_block, full)), "a models block on the overview passes"
    counted = new_card() + f"\n**{SC._lists().lead_line(REG)}.**\n"
    assert any("lead line" in p for p in org_problems(counted, full)), "a lead line on the overview passes"


def test_a_new_tpu_port_reaches_the_collection_and_leaves_the_card_as_it_is():
    reg = json.loads(json.dumps(REG))
    m = json.loads(json.dumps(next(x for x in reg["models"] if x["id"] == "qwen3-8-27b")))
    m.update(id="newmodel-9b", name="NewModel-9B", hf_repo=f"{ORG}/NewModel-9B-TPU")
    m["catalog"].pop("family_collection", None)   # another family: only the TPU collection is under test
    reg["models"].append(m)
    got = SC.collection_problems(reg, collections(sorted(SC.tpu_ports(REG))), "c")
    assert got == [("c", f"'TPU models' lacks {ORG}/NewModel-9B-TPU")], got
    assert SC.overview_problems(reg, new_card(REG), "card") == []   # the overview names no model: nothing to update


def test_each_kind_names_its_collection_and_a_new_kind_is_checked(monkeypatch):
    """KINDS names each kind's standing collection (every "-TPU" kind: "TPU models"); collection_problems loops over the
    titles, so a port of a new kind that names its own collection is checked there without another rule."""
    assert {k["collection"] for k in SC.KINDS.values() if k["suffix"] == "-TPU"} == {"TPU models"}
    assert all("collection" in k for k in SC.KINDS.values())
    monkeypatch.setitem(SC.KINDS, "new-kind", {"suffix": "-NEW", "label": "New", "collection": "New models"})
    reg = json.loads(json.dumps(REG))
    m = json.loads(json.dumps(next(x for x in reg["models"] if x["id"] == "qwen3-8-27b")))
    m.update(id="newkind-1b", name="NewKind-1B", kind="new-kind", hf_repo=f"{ORG}/NewKind-1B-NEW")
    m["catalog"].pop("family_collection", None)   # another family
    reg["models"].append(m)
    tpu = collections(sorted(SC.tpu_ports(REG)))
    assert SC.collection_problems(reg, tpu, "c") == [("c", "0 collections titled 'New models' (want 1)")]
    both = json.dumps(json.loads(tpu) + [{"title": "New models", "items": [{"type": "model", "id": m["hf_repo"]}]}])
    assert SC.collection_problems(reg, both, "c") == []


def test_a_family_collection_named_on_the_row_holds_it():
    """A row that names its family collection (catalog.family_collection) must be in that collection, and our repos in it
    must be rows that name it; a row naming none is in no family collection."""
    named = [m for m in SC.live(REG) if (m.get("catalog") or {}).get("family_collection")]
    assert named, "no live row names its family collection"
    ok = collections(sorted(SC.tpu_ports(REG)))
    assert SC.collection_problems(REG, ok, "c") == []
    gone = json.dumps([c for c in json.loads(ok) if c["title"] != "Qwen on TPU"]
                      + [{"title": "Qwen on TPU", "items": [{"type": "model", "id": "Qwen/Qwen3.8-27B"}]}])
    assert ("c", f"'Qwen on TPU' lacks {ORG}/Qwen3.8-27B-TPU") in SC.collection_problems(REG, gone, "c")
    extra = json.dumps([c for c in json.loads(ok) if c["title"] != "Qwen on TPU"]
                       + [{"title": "Qwen on TPU", "items": [{"type": "model", "id": f"{ORG}/Qwen3.8-27B-TPU"},
                                                             {"type": "model", "id": f"{ORG}/Hy-Nomas-TPU"}]}])
    assert any("holds" in p and "Hy-Nomas-TPU" in p for _, p in SC.collection_problems(REG, extra, "c"))


def test_a_collection_listing_shows_four_items_so_each_is_read_whole():
    """The org's collection listing (api/collections?owner=) shows at most 4 items of each collection: each matched
    collection is read whole (api/collections/<slug>) before it is compared, so a fifth port is never reported missing.
    (elsewhere[] rows and their collections are not under test here)"""
    reg = no_listed()
    base = next(x for x in reg["models"] if x["id"] == "qwen3-8-27b")
    for n in range(4):
        m = json.loads(json.dumps(base))
        m.update(id=f"port-{n}", name=f"Port-{n}", hf_repo=f"{ORG}/Port-{n}-TPU")
        m["catalog"].pop("family_collection", None)
        reg["models"].append(m)
    want = sorted(SC.tpu_ports(reg))
    listing = json.dumps([{"title": "TPU models", "slug": f"{ORG}/tpu-models-1",
                           "items": [{"type": "model", "id": i} for i in want[:4]]},
                          {"title": "Qwen on TPU", "slug": f"{ORG}/qwen-on-tpu-1",
                           "items": [{"type": "model", "id": f"{ORG}/Qwen3.8-27B-TPU"}]}])

    class Whole:
        def get(self, url):
            if url == f"https://huggingface.co/api/collections/{ORG}/tpu-models-1":
                return 200, json.dumps({"items": [{"type": "model", "id": i} for i in want]})
            if url == f"https://huggingface.co/api/collections/{ORG}/qwen-on-tpu-1":
                return 200, json.dumps({"items": [{"type": "model", "id": f"{ORG}/Qwen3.8-27B-TPU"}]})
            return 404, ""
    assert len(want) > 4 and SC.collection_problems(reg, listing, "c", Whole()) == []
    assert any("lacks" in p for _, p in SC.collection_problems(reg, listing, "c"))   # the listing alone looks short


def tpu_row(**kw):
    r = {"id": "weights-x", "name": "Weights X", "summary": "Text to image.", "device": "TPU",
         "how_to_run": f"https://huggingface.co/{ORG}/Weights-X-TPU#load", "hf_repo": f"{ORG}/Weights-X-TPU",
         "hf_revision": "a" * 40, "upstream": {"name": "Weights X", "maker": "Someone", "hf": "someone/x",
                                               "license": "apache-2.0"}}
    r.update(kw)
    return r


def no_listed(reg=REG):
    reg = json.loads(json.dumps(reg))
    reg.pop("elsewhere", None)
    return reg


def test_the_repos_paragraph_is_rendered_from_models_json_and_true_for_every_repo():
    """The card's one paragraph on what our repos hold is rendered from models.json between REPOS_START and REPOS_END:
    a sentence for the run tier ("TPU models": `mashbox run` starts them) and, per device of elsewhere[] rows, one for
    the repos `mashbox run` does not start, naming their device's collection. No row named, no count, no claim a repo
    of either tier does not keep (bits per weight, measured numbers and file names are each card's to give)."""
    run_only = SC.repos_text(no_listed())
    assert "TPU models" in run_only and "`mashbox` package" in run_only
    assert "does not start" not in run_only and "Weights for TPU" not in run_only
    reg = no_listed()
    reg["elsewhere"] = [tpu_row()]
    both = SC.repos_text(reg)
    assert "TPU models" in both and '"Weights for TPU"' in both and "`mashbox run` does not start" in both
    reg["elsewhere"].append(tpu_row(id="ane-x", name="ANE X", device="Apple Neural Engine",
                                    hf_repo=f"{ORG}/ANE-X-CoreML", how_to_run=f"https://huggingface.co/{ORG}/ANE-X-CoreML#run"))
    ane = SC.repos_text(reg)
    assert "the Apple Neural Engine" in ane and "`<Model>-CoreML`" in ane and SC.DEVICE_INFO["Apple Neural Engine"]["collection"] in ane
    for text in (run_only, both, ane):
        assert "\n" not in text and "Weights X" not in text and "ANE X" not in text
        assert not SC.COUNT_RE.search(text), text
        for claim in ("bits per weight", "measured", ".bshj"):
            assert claim not in text, (claim, text)


def test_the_repos_paragraph_claims_our_tpu_kernels_only_as_far_as_the_rows_ship_them():
    """The run tier's sentence claims our own TPU kernels from the rows (each row's kernels[]): for every repo when
    every row of that collection ships kernels, for "some" when only some do, and not at all when none do. A port
    that runs on plain JAX with no compiled kernels (TRELLIS.2) keeps the sentence true."""
    def tier(reg):
        return next(x for x in SC.repos_text(reg).split(". ") if '"TPU models"' in x)
    tpu = [m for m in SC.live(REG) if m.get("hf_repo") and SC.KINDS[m["kind"]].get("collection") == SC.TPU_COLLECTION]
    assert any(m.get("kernels") for m in tpu) and not all(m.get("kernels") for m in tpu), "the case this test covers"
    some = tier(no_listed())
    assert "port and tune with our own TPU kernels" not in some and "some of them also use our own TPU kernels" in some
    reg = no_listed()
    for m in reg["models"]:
        m["kernels"] = m.get("kernels") or [{"file": "kernels/k.whl", "sha256": "a" * 64}]
    every = tier(reg)
    assert "port and tune with our own TPU kernels" in every and "some of them" not in every
    for m in reg["models"]:
        m["kernels"] = []
    none = tier(reg)
    assert "kernels" not in none and "Google Cloud TPU" in none and "`mashbox` package" in none


def test_the_card_carries_the_repos_paragraph_as_rendered_or_fails_online():
    full = collections(sorted(SC.tpu_ports(REG)))
    assert org_problems(new_card(), full) == []
    old_line = new_card().replace(SC.repos_text(REG), "Repo names: `<Model>-TPU` runs on Google Cloud TPU with the "
                                                       "`mashbox` package.")
    assert any("repos" in p for p in org_problems(old_line, full)), "a hand-typed repos paragraph passes"
    gone = new_card().split(SC.REPOS_START)[0]
    assert any("repos" in p for p in org_problems(gone, full)), "a card with no repos paragraph passes"


def test_render_writes_the_org_cards_repos_paragraph_from_models_json(tmp_path):
    """render --org-card re-renders the card's repos paragraph (the rest of the card stays as it is) into the extra output
    hf-org-card/README.md; the views compared with the tree never hold it."""
    card = tmp_path / "card.md"
    card.write_text(new_card().replace(SC.repos_text(REG), "stale words"))
    p = SC.Problems()
    views = SC.render_all(REG, ROOT, None, ROOT / "templates" / "hf", p, org_card=card)
    assert "hf-org-card/README.md" not in views
    assert SC.render_all.overviews == {"hf-org-card/README.md": new_card()}


def test_an_elsewhere_row_sits_in_its_devices_collection_and_only_there():
    """elsewhere[] rows go in their device's collection (DEVICE_INFO): the "TPU models" collection holds what `mashbox
    run` starts on TPU, so a weights-only TPU repo is in "Weights for TPU", never in "TPU models"."""
    assert set(SC.DEVICE_INFO) == set(SC.DEVICES)
    reg = no_listed()
    reg["elsewhere"] = [tpu_row()]
    want = SC.collection_wants(reg)
    assert want["Weights for TPU"] == {f"{ORG}/Weights-X-TPU"}
    assert f"{ORG}/Weights-X-TPU" not in want["TPU models"]
    assert "Weights for TPU" not in SC.collection_wants(no_listed()), "a device with no row wants no collection"
    ok = collections(sorted(SC.tpu_ports(reg)), reg)
    assert SC.collection_problems(reg, ok, "c") == []
    lacking = json.dumps([c for c in json.loads(ok) if c["title"] != "Weights for TPU"]
                         + [{"title": "Weights for TPU", "items": []}])
    assert SC.collection_problems(reg, lacking, "c") == [("c", f"'Weights for TPU' lacks {ORG}/Weights-X-TPU")]
