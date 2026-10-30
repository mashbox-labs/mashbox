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
            "Repo names: `<Model>-TPU` runs on Google Cloud TPU with the `mashbox` package.\n")


def collections(ids):
    return json.dumps([{"title": "TPU models", "items": [{"type": "model", "id": i} for i in ids]},
                       {"title": "Qwen on TPU", "items": [{"type": "model", "id": f"{ORG}/Qwen3.8-27B-TPU"}]}])


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
    reg["models"].append(m)
    got = SC.collection_problems(reg, collections(sorted(SC.tpu_ports(REG))), "c")
    assert got == [("c", f"'TPU models' lacks {ORG}/NewModel-9B-TPU")], got
    assert SC.overview_problems(reg, new_card(REG), "card") == []   # the overview names no model: nothing to update
