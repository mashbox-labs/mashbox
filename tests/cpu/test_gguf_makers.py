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
        assert g["maker"] not in (g["model"], g["base"].split("/")[1]), g["id"]
        # never the GGUF publisher, unless the publisher is the base model's own org (a first-party GGUF)
        assert g["maker"] != g["publisher"] or g["publisher"] == org, g["id"]


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


def test_rows_tool_writes_every_key_a_gguf_row_needs(tmp_path, monkeypatch):
    """a row from tools/gguf_rows.py passes the catalog schema as written: every key of GGUF_KEYS, in and out "text" (a
    row is written only from a gguf_conform record, whose checks are text: tokenizer and greedy tokens), in the key
    order of the rows in models.json"""
    import sys
    import types
    spec = importlib.util.spec_from_file_location("sync_check", os.path.join(ROOT, "tools", "hf_github_sync_check.py"))
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    fake = types.ModuleType("gguf_scan")
    fake.parse = lambda data: ({"general.basename": "Qwen2.5", "general.size_label": "7B"},)
    monkeypatch.setitem(sys.modules, "gguf_scan", fake)
    (tmp_path / "header.bin").write_bytes(b"GGUF")
    (tmp_path / "card.json").write_text(json.dumps({"cardData": {"base_model": "Qwen/Qwen2.5-7B-Instruct",
                                                                 "license": "apache-2.0"}}))
    same = dict(nmse=1e-11, top1_agree=128, positions=128)
    rec = dict(pin=dict(repo="bartowski/Qwen2.5-7B-Instruct-GGUF", revision="0" * 40, file="Q.gguf", tag="Q4_K_M"),
               c_cpu=dict(identical_weights=same, top1_agree=120, positions=128, kl_mean=0.004, prompts=4, n_gen=32,
                          full_match=3, near_tie_divergences=1, other_divergences=0),
               b_tokenizer=dict(texts=8, nfc_equal=8, nfc_explained=0), types=["F32", "Q4_K", "Q6_K"],
               tpu_key="qwen2 x Q4_K+Q6_K", sha256="1" * 64, bytes=1, arch="qwen2", moe=False, verdict="cpu-checked",
               versions=dict(llama_cpp="a37cc46aa"))
    g = rows_tool().row(str(tmp_path), rec, [])[0]
    assert [k for k in sc.GGUF_KEYS if k not in g] == []
    assert (g["in"], g["out"]) == ("text", "text")
    reg = json.load(open(os.path.join(ROOT, "src", "mashbox", "models.json")))
    have = [list(x) for x in reg["gguf"] if "cpu_check" in x]
    assert have and list(g) in have, list(g)


def _sync_check():
    spec = importlib.util.spec_from_file_location("sync_check", os.path.join(ROOT, "tools", "hf_github_sync_check.py"))
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def test_rows_tool_keeps_a_smoke_from_another_row_off_the_row(tmp_path, monkeypatch):
    """a row shows a speed only when measured on that exact file: a file no chip ran reads "TPU: Not tested yet" with
    its v6e chip holding status and not_run_reason only, even when an arch x block-type smoke of another row covers
    it; that smoke comes back beside the row (the rows JSON keeps it), never inside the row or its text"""
    import sys
    import types
    fake = types.ModuleType("gguf_scan")
    fake.parse = lambda data: ({"general.basename": "Qwen2.5", "general.size_label": "7B"},)
    monkeypatch.setitem(sys.modules, "gguf_scan", fake)
    (tmp_path / "header.bin").write_bytes(b"GGUF")
    (tmp_path / "card.json").write_text(json.dumps({"cardData": {"base_model": "Qwen/Qwen2.5-7B-Instruct",
                                                                 "license": "apache-2.0"}}))
    same = dict(nmse=1e-11, top1_agree=128, positions=128)
    rec = dict(pin=dict(repo="bartowski/Qwen2.5-7B-Instruct-GGUF", revision="0" * 40, file="Q.gguf", tag="Q4_K_M"),
               c_cpu=dict(identical_weights=same, top1_agree=120, positions=128, kl_mean=0.004, prompts=4, n_gen=32,
                          full_match=3, near_tie_divergences=1, other_divergences=0),
               b_tokenizer=dict(texts=8, nfc_equal=8, nfc_explained=0), types=["F32", "Q4_K", "Q6_K"],
               tpu_key="qwen2 x Q4_K+Q6_K", sha256="1" * 64, bytes=1, arch="qwen2", moe=False, verdict="cpu-checked",
               versions=dict(llama_cpp="a37cc46aa"))
    # a passing v6e run of another qwen2 file whose quantized types cover this file's (what smokes() returns)
    smk = [("qwen2", frozenset({"Q4_K", "Q5_0", "Q6_K", "Q8_0"}), "other-qwen2-row", "v6e-1", 17.05, "2" * 64)]
    out = rows_tool().row(str(tmp_path), rec, smk)
    g, line, tpu = out[0], out[1], out[2]
    assert g["chips"] == {"v6e": {"status": "pending", "not_run_reason": "no chip run of this file"}}
    assert "smoke" not in json.dumps(g) and "other-qwen2-row" not in json.dumps(g)
    assert tpu == "TPU: Not tested yet"
    assert "tok/s" not in line + tpu and "other-qwen2-row" not in line + tpu
    assert len(out) == 4 and out[3] == dict(row="other-qwen2-row", accelerator="v6e-1", decode_tok_s_1_user=17.05)
    # its own chip run: passing, and no smoke beside it
    own = rows_tool().row(str(tmp_path), rec, [smk[0][:5] + ("1" * 64,)])
    assert own[0]["chips"]["v6e"]["status"] == "passing" and own[3] is None


def test_sync_check_refuses_a_smoke_on_a_not_run_chip():
    """a pending (not-run) chip holds status and not_run_reason only: the sync check refuses a smoke on it, even one
    from a passing run of the same architecture whose types cover the file's (the case the old rule let through)"""
    import copy
    sc = _sync_check()
    reg = json.load(open(os.path.join(ROOT, "src", "mashbox", "models.json")))

    def problems(r):
        p = sc.Problems()
        sc.validate_catalog(r, p)
        return list(p)

    src = next(g for g in reg["gguf"] if ((g.get("chips") or {}).get("v6e") or {}).get("status") == "passing")
    pend = next(g for g in reg["gguf"] if ((g.get("chips") or {}).get("v6e") or {}).get("status") == "pending")
    base = problems(copy.deepcopy(reg))
    bad = copy.deepcopy(reg)
    g = next(x for x in bad["gguf"] if x["id"] == pend["id"])
    g["arch"] = src["arch"]
    g["types"] = [t for t in src["types"] if t in sc.UNQUANTIZED] + [t for t in src["types"] if t not in sc.UNQUANTIZED][:1]
    # any number: the check refuses a smoke on a pending chip whatever it holds (a QUICK row has no first-request
    # number to copy)
    g["chips"]["v6e"]["smoke"] = dict(row=src["id"], accelerator=src["chips"]["v6e"]["accelerator"],
                                      decode_tok_s_1_user=10.0)
    covered = copy.deepcopy(bad)
    del next(x for x in covered["gguf"] if x["id"] == pend["id"])["chips"]["v6e"]["smoke"]
    # the row change alone (arch, types) adds nothing the smoke check is about
    assert [x for x in problems(covered) if "chips.v6e" in x and x not in base] == []
    added = [x for x in problems(bad) if x not in problems(covered)]
    assert len(added) == 1 and "chips.v6e: a pending chip holds its status and not_run_reason" in added[0], added
    assert "no smoke" in added[0], added


# Makers added for the next-20 files, each named as upstream writes it on its own Hugging Face org page or card
# (sources and revisions in the item notes): (org, maker, a base of that org).
NEW_MAKERS = (("naver-hyperclovax", "HyperCLOVA X", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"),
              ("janhq", "Jan", "janhq/Jan-v3-4B-base-instruct"),
              ("prism-ml", "Prism ML", "prism-ml/Ternary-Bonsai-8B-unpacked"),
              ("superwhisper", "Superwhisper", "superwhisper/s1-mini"),
              ("ornith-ai", "Ornith", "ornith-ai/Ornith-1.5-9B"),
              ("openbmb", "OpenBMB", "openbmb/MiniCPM5-2B"),
              ("defog", "Defog, Inc", "defog/sqlcoder-7b-2"),
              # a personal account: the name its cards publish under, never the account page's full name
              ("DavidAU", "DavidAU", "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP"))


@pytest.mark.parametrize("org,maker,base", NEW_MAKERS)
def test_a_new_maker_name_maps_only_from_its_own_org(org, maker, base):
    R = rows_tool()
    assert R.MAKER_NAMES[org] == maker
    assert R.upstream("someone/X-GGUF", base) == (base, maker)
    # the same model name under an org with no maker entry is refused: the name never travels with the model name
    with pytest.raises(ValueError, match="no maker name"):
        R.upstream("someone/X-GGUF", "no-such-org/" + base.split("/")[1])
    # the publisher's org never stands in for a base the card does not state
    with pytest.raises(ValueError, match="names no base model"):
        R.upstream(f"{org}/Some-Other-Model-GGUF", None)


def test_a_card_with_no_base_model_lists_only_the_original_its_own_text_states():
    R = rows_tool()
    # the card says the publisher made the model and links that original at the pinned revision
    assert R.upstream("ornith-ai/Ornith-1.5-9B-GGUF", None) == ("ornith-ai/Ornith-1.5-9B", "Ornith")
    assert R.upstream("openbmb/MiniCPM5-2B-GGUF", None) == ("openbmb/MiniCPM5-2B", "OpenBMB")
    # the repo is the original itself ("Developed by: Defog, Inc")
    assert R.upstream("defog/sqlcoder-7b-2", None) == ("defog/sqlcoder-7b-2", "Defog, Inc")
    # a card that states no maker or base, or links another org's original, still refuses
    for repo in ("ornith-ai/Ornith-1.0-9B-GGUF", "someone/Unknown-GGUF", "openbmb/Other-GGUF"):
        with pytest.raises(ValueError, match="names no base model"):
            R.upstream(repo, None)
