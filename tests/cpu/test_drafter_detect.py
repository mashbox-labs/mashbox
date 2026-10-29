"""Trained drafters and draft-model siblings are detected from metadata, never from a name (runtime/drafters/select.py,
ranked first). A trained drafter's drafter.json manifest (mbx-drafter/1, kind trained, method dflash, dflare or
eagle3) matches a target when its tokenizer sha256 equals the engine's tokenizer hash and its target shape equals the
target's. ORDER ranks trained first. A detected method whose layer math is not built never applies. A draft-model sibling
has the same tokenizer hash, is dense and is at most 1/4 of the target's bytes."""
from types import SimpleNamespace

from mashbox.runtime import drafters as DRS, spec_step as SS
from mashbox.runtime.drafters import select as SEL

TOK, OTHER = "a" * 64, "b" * 64
# Qwen3-4B's config.json shape keys
QWEN3_4B = dict(hidden_size=2560, num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                vocab_size=151936)
SHAPE = dict(hidden=2560, layers=36, heads=32, kv_heads=8, head_dim=128, vocab=151936)
CFG = SimpleNamespace(E=0)
DENSE = SimpleNamespace(SPEC_VERIFY=True)                               # verify pass, no MTP head module
DENSE_MTP = SimpleNamespace(SPEC_VERIFY=True, SPEC_MTP_HEAD=object())   # verify pass and an MTP head module


def manifest(method="dflare", tok=TOK, shape=None, hf_id="Qwen/Qwen3-4B", **kw):
    m = dict(format="mbx-drafter/1", kind="trained", method=method, k_max=16, needs_hidden=True,
             hidden_layers=[1, 9, 17, 25, 33],
             target=dict(hf_id=hf_id, revision="r1", config_sha256="c" * 64, tokenizer_sha256=tok,
                         shape=dict(SHAPE) if shape is None else shape),
             files={"model.safetensors": "d" * 64, "config.json": "e" * 64}, origin="public:org/drafter@r1",
             record=None, spec_step_version=SS.SPEC_STEP_VERSION)
    m.update(kw)
    return m


def gated_all(css):
    ok = dict(status="passed", record="r", record_sha256="f" * 64, component_set_sha=css,
              spec_step_version=SS.SPEC_STEP_VERSION, k_table={"1": 2})
    return {k: {"v6e-1": dict(ok)} for k in DRS.ORDER}


def test_hf_shape_reads_the_config_keys():
    assert SEL.hf_shape(QWEN3_4B) == SHAPE
    assert SEL.hf_shape(dict(text_config=QWEN3_4B)) == SHAPE
    no_hd = {k: v for k, v in QWEN3_4B.items() if k != "head_dim"}
    assert SEL.hf_shape(no_hd)["head_dim"] == 2560 // 32
    assert SEL.hf_shape({})["hidden"] is None


def test_trained_detected_on_tokenizer_and_shape_never_a_name():
    good = manifest(hf_id="someone/unrelated-name")            # the target's hf_id is never read
    named = manifest(tok=OTHER, hf_id="Qwen/Qwen3-4B")           # the right name, another tokenizer
    shaped = manifest(shape=dict(SHAPE, layers=28))             # same tokenizer, another target shape
    broken = manifest()
    del broken["files"]
    m = SEL.meta_of(DENSE, CFG, "bshj", target=dict(tokenizer_sha256=TOK, shape=SEL.hf_shape(QWEN3_4B),
                                                     trained=[named, shaped, broken, good]))
    assert m["trained"] == [SEL.name_free(good)] and SEL.detect_trained([good], TOK, SHAPE)[0] == [good]
    assert "origin" not in m["trained"][0] and not {"hf_id", "revision"} & set(m["trained"][0]["target"])    # no name
    assert [w.split(": ", 1)[1] for w in m["trained_why"]] == [
        "declared for another tokenizer", "declared for another target shape", "files needs the sha256 of every file"]
    none = SEL.meta_of(DENSE, CFG, "bshj", target=dict(tokenizer_sha256=None, shape=SHAPE, trained=[good]))
    assert none["trained"] == []                                # no tokenizer hash: nothing is detected
    assert SEL.meta_of(DENSE, CFG, "bshj")["trained"] == []     # no manifests: nothing is detected


def test_order_ranks_trained_first(monkeypatch):
    """with its layer math built and a passed record, a trained drafter wins over the MTP head; without its record the
    MTP head wins (the trained why line first); with no manifest the order is unchanged"""
    css = "c" * 64
    good = manifest()
    monkeypatch.setattr(SEL, "BUILT", {"dflare": "mashbox.runtime.drafters.dflare"}, raising=False)
    base = SEL.meta_of(DENSE_MTP, CFG, "bshj", 1)
    m = dict(base, trained=[good], trained_why=[])
    rec = gated_all(css)
    kind, g, why, _ = SEL.pick(m, "v6e-1", rec, css=css)
    assert (kind, g) == ("trained", True) and len(why) == 1 and why[0].startswith("trained: the dflare drafter")
    rec.pop("trained")
    kind, g, why, _ = SEL.pick(m, "v6e-1", rec, css=css)
    assert (kind, g) == ("mtp", True) and [w.split(":")[0] for w in why] == ["trained", "mtp"]
    kind, g, why, _ = SEL.pick(dict(base, trained=[], trained_why=[]), "v6e-1", gated_all(css), css=css)
    assert (kind, g) == ("mtp", True)


def test_first_matched_manifest_decides(monkeypatch):
    """two manifests match this target: the first in the catalog's order decides what applies (built: it applies; not
    built: its why line, even when a later match is built)"""
    monkeypatch.setattr(SEL, "BUILT", {"dflare": "mashbox.runtime.drafters.dflare"}, raising=False)
    base = SEL.meta_of(DENSE, CFG, "bshj")
    ok, w = SEL.applies("trained", dict(base, trained=[manifest(method="dflare"), manifest(method="eagle3")]))
    assert ok is True and w == "trained: the dflare drafter declared for this target", w
    ok, w = SEL.applies("trained", dict(base, trained=[manifest(method="eagle3"), manifest(method="dflare")]))
    assert ok is False and w.startswith("trained: a eagle3 drafter declared for this target is detected"), w


def test_detected_trained_method_not_built_never_applies(monkeypatch):
    """a detected manifest whose method has no layer math in BUILT gives its why line and never applies, gated or
    forced; auto falls through to the next kind"""
    css = "c" * 64
    monkeypatch.setattr(SEL, "BUILT", {}, raising=False)
    m = dict(SEL.meta_of(DENSE, CFG, "bshj"), trained=[manifest(method="eagle3")], trained_why=[])
    ok, w = SEL.applies("trained", m)
    assert ok is False and w == ("trained: a eagle3 drafter declared for this target is detected; its layer math is "
                                 "not built in this version"), w
    kind, g, why, _ = SEL.pick(m, "v6e-1", gated_all(css), css=css)
    assert (kind, g) == ("ngram", True) and why[0].startswith("trained: a eagle3 drafter declared")
    assert SEL.pick(m, "v6e-1", gated_all(css), forced="trained", css=css)[0] is None


def test_manifest_problems():
    assert SEL.manifest_problem(manifest()) is None
    bad = [(dict(format="mbx-drafter/0"), "not an mbx-drafter/1 manifest"),
           (dict(kind="draft_model"), "kind 'draft_model' is not trained"),
           (dict(method="medusa"), "method 'medusa' is not one of dflash, dflare, eagle3"),
           (dict(k_max=0), "k_max 0: 1 to 16"), (dict(k_max=17), "k_max 17: 1 to 16"), (dict(k_max=True), "k_max True: 1 to 16"),
           (dict(files={}), "files needs the sha256 of every file"),
           (dict(files={"model.safetensors": "d" * 63}), "files needs the sha256 of every file")]
    for kw, why in bad:
        assert SEL.manifest_problem(manifest(**kw)) == why, kw
    assert SEL.manifest_problem(manifest(tok="A" * 64)) == "target.tokenizer_sha256 is not a sha256"
    for sh in (dict(SHAPE, vocab=None), dict(SHAPE, heads=True), {k: v for k, v in SHAPE.items() if k != "kv_heads"},
               dict(SHAPE, layers=0), dict(SHAPE, hidden=-2560)):
        p = SEL.manifest_problem(manifest(shape=sh))
        assert p is not None and p.startswith("target.shape needs positive integers"), (sh, p)
    assert SEL.manifest_problem(None) == "not an mbx-drafter/1 manifest"


def test_draft_model_siblings_same_tokenizer_dense_quarter_bytes():
    """Qwen3-8B bf16 (16,381,470,720 bytes): the same-tokenizer dense rows at most 1/4 of that, largest first; a row
    with the target's own id but another tokenizer, a MoE row and a row over 1/4 are not siblings"""
    target = 16_381_470_720
    rows = [dict(id="qwen3-0.6b", tokenizer_sha256=TOK, bytes=1_503_300_328, experts=0),
            dict(id="qwen3-1.7b", tokenizer_sha256=TOK, bytes=4_063_000_000, experts=0),
            dict(id="qwen3-4b", tokenizer_sha256=TOK, bytes=8_044_000_000, experts=0),
            dict(id="qwen3-8b", tokenizer_sha256=OTHER, bytes=1_000_000_000, experts=0),
            dict(id="moe-small", tokenizer_sha256=TOK, bytes=1_000_000_000, experts=128),
            dict(id="flag", tokenizer_sha256=TOK, bytes=True, experts=0)]
    got = SEL.detect_draft_model(rows, TOK, target)
    assert [r["id"] for r in got] == ["qwen3-1.7b", "qwen3-0.6b"]
    assert SEL.detect_draft_model(rows, None, target) == [] and SEL.detect_draft_model(rows, TOK, None) == []
    llama = [dict(id="l-1b", tokenizer_sha256=OTHER, bytes=2_471_645_184, experts=0)]
    assert [r["id"] for r in SEL.detect_draft_model(llama, OTHER, 16_060_522_496)] == ["l-1b"]
    m = SEL.meta_of(DENSE, CFG, "bshj", target=dict(tokenizer_sha256=TOK, bytes=target, siblings=rows))
    assert m["draft_models"] == [dict(bytes=4_063_000_000, sha256=None), dict(bytes=1_503_300_328, sha256=None)]  # no id
    ok, w = SEL.applies("draft_model", m)
    assert ok is False and w == ("draft_model: no draft model given (--draft-model); 2 same-tokenizer catalog sibling(s) "
                                 "detected, loading one from the catalog is not in this version"), w
