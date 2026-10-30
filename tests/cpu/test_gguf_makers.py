"""A GGUF row names its base model as org/name and its maker as that model's card names it: MAKER_NAMES (tools/gguf_rows.py,
the one table) of the base's Hugging Face org, never the GGUF publisher or the model's own name. tools/gguf_rows.py writes
both from the sealed record's card; an unknown base or org is refused, never guessed."""
import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rows_tool():
    spec = importlib.util.spec_from_file_location("gguf_rows", os.path.join(ROOT, "tools", "gguf_rows.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_gguf_row_names_its_base_and_its_maker():
    names = rows_tool().MAKER_NAMES
    reg = json.load(open(os.path.join(ROOT, "src", "mashbox", "models.json")))
    assert reg["gguf"]
    for g in reg["gguf"]:
        assert re.fullmatch(r"[^/\s]+/[^/\s]+", str(g["base"])), (g["id"], g["base"])
        org = g["base"].split("/")[0]
        assert g["maker"] == names.get(org), (g["id"], g["maker"], org)
        assert g["maker"] not in (g["publisher"], g["model"], g["base"].split("/")[1]), g["id"]


def test_rows_tool_takes_base_and_maker_from_the_card():
    R = rows_tool()
    assert R.upstream("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen/Qwen2.5-0.5B-Instruct") == \
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen")
    # a bare base_model is a repo of the card's own org
    assert R.upstream("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF", "EXAONE-3.5-2.4B-Instruct") == \
        ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "LG AI Research")
    # no base_model in the card metadata: the original model its card text links
    assert R.upstream("bartowski/Mistral-7B-Instruct-v0.3-GGUF", None) == \
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral AI")
    with pytest.raises(ValueError):
        R.upstream("someone/Unknown-GGUF", None)
    with pytest.raises(ValueError):
        R.upstream("bartowski/X-GGUF", "someorg/X")
