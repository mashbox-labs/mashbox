"""The TRELLIS.2 page and Hugging Face card: the download size the pinned files add up to, the "Experimental" the
catalog views print from the row's flag, each gated component's accept line, every licence as the row records it, and
the credit and unchanged LICENSE of the code the port carries over from another project."""
import copy
import fnmatch
import hashlib
import importlib.util
import json
import re
import tomllib
from pathlib import Path

from mashbox import catalog
from mashbox.models.trellis_2 import weights as W

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "models" / "trellis-2.md"


def test_page_states_the_pinned_download_size():
    text = PAGE.read_text()
    size = f"{W.needed_bytes() / 1e9:.1f} GB"
    assert f"the first run downloads {size} into" in text, size


def test_the_catalog_views_print_the_experimental_label_from_the_flag():
    """The row stores the flag and no status or label; the catalog views print "Experimental" next to TRELLIS.2 from that
    flag, the page carries the same word in its label block, and with the flag off no view prints it."""
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = catalog.registry(str(ROOT / "src" / "mashbox" / "models.json"))
    row = next(m for m in reg["models"] if m["id"] == "trellis-2")
    assert row["catalog"]["experimental"] is True and not {"status", "level", "label"} & set(row["catalog"])
    assert "<!-- catalog:start label -->\nExperimental\n<!-- catalog:end -->" in PAGE.read_text()
    assert "**[TRELLIS.2](models/trellis-2.md)** (Experimental)<br>" in catalog.render_markdown(reg)
    assert "TRELLIS.2  [trellis-2]  MASHBOX build  Experimental\n" in catalog.render_text(reg)
    assert [r["experimental"] for p in catalog.entries(reg)["publishers"] for m in p["models"] for r in m["rows"]
            if r["id"] == "trellis-2"] == [True]
    full = json.loads((ROOT / "models.json").read_text())
    assert sc.experimental_note(next(m for m in full["models"] if m["id"] == "trellis-2"), "<br>") == "<br>Experimental"
    off = copy.deepcopy(reg)
    next(m for m in off["models"] if m["id"] == "trellis-2")["catalog"]["experimental"] = False
    assert "xperimental" not in catalog.render_text(off).split("TRELLIS.2")[1][:80]
    assert "**[TRELLIS.2](models/trellis-2.md)**<br>" in catalog.render_markdown(off)


def _gated():
    """The gated upstream components, as upstream.json pins them and as the row copies them (licence values by script)."""
    pins = W.pins()["repos"]
    reg = catalog.registry(str(ROOT / "src" / "mashbox" / "models.json"))
    row = next(m for m in reg["models"] if m["id"] == "trellis-2")
    comps = {c["hf"]: c for c in row["upstream"]["components"] if c.get("hf")}
    out = []
    for p in pins.values():
        if p.get("gated"):
            c = comps[p["hf"]]
            assert (c.get("license"), c.get("license_name"), c.get("gated")) == (p["license"], p["license_name"], True)
            out.append(c)
    assert out, "no gated component pinned"
    return out


def _accept_lines(text, c):
    return [ln for ln in text.splitlines() if "accept" in ln and f"https://huggingface.co/{c['hf']}" in ln
            and c["license_name"] in ln and c["license_link"] in ln]


def test_every_weight_comes_from_its_makers_repo():
    """mashbox run trellis-2 downloads each pinned file from its maker's own Hugging Face repo with the user's login; no
    pinned repo is ours, so nothing is re-hosted (DINOv3 stays behind Meta's gate)."""
    for p in W.pins()["repos"].values():
        assert not p["hf"].lower().startswith("mashbox"), p["hf"]


def test_page_says_in_one_line_to_accept_the_gated_licence_first():
    page = PAGE.read_text()
    for c in _gated():
        assert len(_accept_lines(page, c)) == 1, c["hf"]


def test_card_says_in_one_line_to_accept_the_gated_licence_first():
    spec = importlib.util.spec_from_file_location("hf_sync", ROOT / "tools" / "hf_github_sync_check.py")
    hs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs)
    reg = __import__("json").loads((ROOT / "models.json").read_text())
    row = copy.deepcopy(next(m for m in reg["models"] if m["id"] == "trellis-2"))
    row["hf_repo"] = row["hf_repo"] or "mashbox-labs/TRELLIS.2-TPU"
    card = hs.render_card(reg, row, ROOT / "templates" / "hf", ROOT)
    for c in _gated():
        assert len(_accept_lines(card, c)) == 1, c["hf"]
        assert f"{row['hf_repo']}/resolve" not in card or c["hf"] not in card.split(f"{row['hf_repo']}/resolve")[1][:200]


def _row():
    return next(m for m in json.loads((ROOT / "models.json").read_text())["models"] if m["id"] == "trellis-2")


def _card(row):
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    hs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs)
    reg = json.loads((ROOT / "models.json").read_text())
    return hs.render_card(reg, dict(row, hf_repo="mashbox-labs/TRELLIS.2-TPU"), ROOT / "templates" / "hf", ROOT)


def test_ported_code_ships_its_own_licence_unchanged_and_the_page_and_card_credit_it():
    """Code ported from another project (the GLB export's simplify path, from CuMesh) is a row component with a notice:
    that project's LICENSE file, byte for byte as pinned, beside the ported code and in the wheel. The page and the card
    name it with its licence as the notice itself names it."""
    row = _row()
    credits = [c for c in row["upstream"]["components"] if c.get("notice")]
    assert [c["github"] for c in credits] == ["JeffreyXiang/CuMesh", "JeffreyXiang/FlexGEMM", "uploadcare/pillow-simd"]
    card = _card(row)
    for c in credits:
        notice = ROOT / c["notice"]
        assert hashlib.sha256(notice.read_bytes()).hexdigest() == c["license_sha256"]
        assert c["license"] in notice.read_text()   # the licence named as the notice itself states it
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["mashbox"]
        assert any(fnmatch.fnmatch(Path(c["notice"]).relative_to("src/mashbox").as_posix(), g) for g in data)
        for where, text in (("page", PAGE.read_text()), ("card", card)):
            assert any(c["github"] in line and c["license"] in line and Path(c["notice"]).name in line
                       for line in text.splitlines()), f"the {where} does not credit {c['github']}"
    # read, not copied: one credit line with its licence as upstream states it, and no file
    refs = [c for c in row["upstream"]["components"] if c.get("reference")]
    assert [c["github"] for c in refs] == ["huggingface/transformers"] and not any(c.get("notice") for c in refs)
    for c in refs:
        for where, text in (("page", PAGE.read_text()), ("card", card)):
            assert any(c["github"] in line and c["license"] in line and "read, not copied" in line
                       for line in text.splitlines()), f"the {where} does not credit {c['github']}"


def test_the_card_names_every_weights_repo_and_its_licence_and_where_the_licence_file_comes_from():
    row = _row()
    card = _card(row)
    lic_line = next(line for line in card.splitlines() if line.startswith("Weights: "))
    for c in [row["upstream"]] + [c for c in row["upstream"]["components"] if c.get("hf")]:
        assert c["hf"] in lic_line and (c.get("license_name") or c["license"]).lower() in lic_line.lower(), c["hf"]
    src = next(c for c in row["upstream"]["components"] if c.get("license_sha256") == row["upstream"]["license_sha256"])
    assert f"https://github.com/{src['github']}/blob/{src['revision']}/LICENSE" in card
    assert "licence file from [microsoft/TRELLIS.2-4B]" not in card


def test_the_card_carries_the_experimental_label_from_the_flag():
    row = _row()
    assert "\n**Experimental**\n" in _card(row)
    off = copy.deepcopy(row)
    off["catalog"]["experimental"] = False
    assert "xperimental" not in _card(off)


def _quickstart(card):
    block = card.split("## Quickstart", 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]
    return block.splitlines()


def test_the_card_quickstart_signs_the_vm_in_before_the_first_run():
    """A gated component downloads with the user's own token: the card's commands sign the VM in between the install and
    the run, as the page's do, and the accept line says so; a row with nothing gated gets no sign-in line."""
    row = _row()
    card = _card(row)
    lines = _quickstart(card)
    assert lines == [row["install"], "~/.local/bin/hf auth login", row["run"]]
    assert "~/.local/bin/hf auth login" in PAGE.read_text().split("## Install", 1)[1].split("## Run", 1)[0]
    for c in _gated():
        assert any("HF_TOKEN" in ln and "hf auth login" in ln for ln in _accept_lines(card, c)), c["hf"]
    open_row = copy.deepcopy(row)
    for c in open_row["upstream"]["components"]:
        c.pop("gated", None)
    assert _quickstart(_card(open_row)) == [row["install"], row["run"]]


def test_the_page_and_card_plan_host_memory_for_the_heaviest_run_known():
    """The row's notes (built from the records) give the heaviest run of any record, an earlier build's run on another
    picture; the card prints them, and the page gives the same two numbers in its host-memory guidance."""
    notes = _row().get("notes", "")
    assert "earlier build" in notes and "plan for at least that much host memory" in notes
    assert notes in _card(_row())
    page = PAGE.read_text().split("## What you need", 1)[1].split("## Install", 1)[0]
    wall, rss = re.search(r" took (\d[^,]*?) and peaked at (\d+\.\d GB)", notes).groups()
    assert f"took {wall}" in page and f"was {rss}" in page, (wall, rss)
    assert "sealed record" not in page and "same GLB export code" not in page


def test_the_cost_names_the_region_of_the_price_it_uses():
    prices = json.loads((ROOT / "docs" / "data" / "cloud-prices.json").read_text())["prices"]
    region = next(p["region"] for p in prices if (p["hardware"], p["column"]) == ("v6e", "On Demand")).replace(", ", " and ")
    cost = next(r for r in _row()["results"] if r["metric"].startswith("Cost per picture"))
    assert f"in {region}" in cost["metric"]
    price_line = next(ln for ln in PAGE.read_text().splitlines() if "an hour at Google's on-demand list price" in ln)
    assert f"in {region}" in price_line


def test_the_licence_page_lists_every_weights_repo_with_its_licence_as_the_card_names_it():
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    hs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs)
    row = _row()
    line = next(ln for ln in (ROOT / "docs" / "licence.md").read_text().splitlines() if ln.startswith(f"| {row['name']} |"))
    for c in [row["upstream"]] + [c for c in row["upstream"]["components"] if c.get("hf")]:
        assert f"{c['hf']}@{c['revision'][:8]}" in line and hs._licence_name(c) in line, c["hf"]
    assert "../licenses/trellis-2/LICENSE" in line and "models/trellis-2.md#credits-and-licences" in line


def test_owners_and_core_changes_name_what_the_row_ships():
    row = _row()
    owners = next(ln for ln in (ROOT / "OWNERS.md").read_text().splitlines() if ln.startswith("| `trellis-2` |"))
    for c in row["upstream"]["components"]:
        if c.get("notice"):
            assert f"`{c['notice']}`" in owners, c["notice"]
    changes = (ROOT / "CORE-CHANGES.md").read_text()
    head = changes.split("### TRELLIS.2: the image-to-3d kind", 1)[0].rsplit("\n## ", 1)[1].split("\n", 1)[0]
    assert head == row["card_release"]
