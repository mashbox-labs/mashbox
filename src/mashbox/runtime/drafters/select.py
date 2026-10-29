"""Which drafter a deployment uses, from metadata and records, never from a model name.

Order (one drafter per deployment):
  1 mtp          the file has an MTP head (BSHJ meta / models.json drafter row of kind mtp, GGUF <arch>.nextn_predict_layers
                 > 0 with its tensors, HF mtp_num_hidden_layers or num_nextn_predict_layers) and its architecture module
                 declares an MTP head implementation for this format (SPEC_MTP_HEAD)
  2 draft_model  a catalog row with the same tokenizer hash, dense, at most 1/4 of the target's bytes, passing the fit
                 check (not built in this version: it never applies)
  3 ngram        any architecture with a verify pass
On a chip, the first that applies AND has a passed gate record for this chip wins (auto). A forced kind (--speculation
<kind>) runs where it applies, gated or not, and then says "not gated on this chip". Each rank gives one why line.
"""
def _modules(model):
    """the architecture module and the module its make_forward lives in (a package re-exports it)"""
    import sys
    mods = [model, sys.modules.get(getattr(getattr(model, "make_forward", None), "__module__", None) or "")]
    return [m for m in mods if m is not None]


def mtp_head(model):
    """the module the architecture declares for its MTP head (SPEC_MTP_HEAD: make_mtp, load_mtp, HEAD_ABI), or None"""
    return next((getattr(m, "SPEC_MTP_HEAD") for m in _modules(model) if getattr(m, "SPEC_MTP_HEAD", None)), None)


def meta_of(model, cfg, fmt, mtp_layers=0):
    """the metadata the choice reads: format, the architecture module's own declarations (SPEC_VERIFY: its forward has
    the verify pass and a commit; SPEC_MTP_HEAD: the module that implements its MTP head), MoE or dense (expert counts),
    MTP layers in the file. Never a model or file name."""
    verify = any(getattr(m, "SPEC_VERIFY", False) for m in _modules(model))
    head = mtp_head(model)
    # MoE: an expert count (cfg.E) or the MoE flag a MoE config sets (cfg.moe), whichever the architecture's config has
    experts = int(getattr(cfg, "E", 0) or 0) or (1 if getattr(cfg, "moe", False) else 0)
    return dict(format=fmt, verify=bool(verify), mtp_head=bool(head) and fmt != "gguf", experts=experts,
                mtp_layers=int(mtp_layers or 0))


def applies(kind, meta):
    """-> (applies, why) for one drafter kind on this file's metadata"""
    if meta.get("experts", 0) > 0:
        return False, f"{kind}: MoE verify at K+1 tokens per row is not in this version"
    if not meta.get("verify"):
        return False, f"{kind}: the architecture has no verify pass"
    if kind == "mtp":
        if meta.get("mtp_layers", 0) <= 0:
            return False, "mtp: the file has no MTP head"
        if not meta.get("mtp_head"):
            return False, "mtp: the file has an MTP head, but no MTP implementation for this format and architecture"
        return True, "mtp: the file's own MTP head"
    if kind == "draft_model":
        return False, "draft_model: not built in this version"
    if kind == "ngram":
        return True, "ngram: prompt lookup over the row's own text (no weights)"
    return False, f"{kind}: unknown"


def gated(rec, chip, css):
    """the gate record of one drafter kind ({chip: entry}) for this chip and this deployment -> (passed, why). Passed only
    when the entry says passed for this deployment's component_set_sha (which holds the drafter's key) and this engine's
    SPEC_STEP_VERSION, with k > 0 at one or more row buckets of its k_table."""
    from ..spec_step import SPEC_STEP_VERSION
    if chip is None:
        return False, "not on a TPU"
    g = (rec or {}).get(chip) or {}
    if g.get("status") != "passed":
        return False, f"not gated on this chip ({chip})"
    if css is None:
        return False, f"not gated on this chip ({chip}): the deployment's component set is not known"
    if g.get("component_set_sha") != css:
        return False, f"not gated on this chip ({chip}): the record is for another component set"
    if g.get("spec_step_version") != SPEC_STEP_VERSION:
        return False, (f"not gated on this chip ({chip}): the record measured speculation step version "
                       f"{g.get('spec_step_version')}, this engine runs {SPEC_STEP_VERSION}")
    if not any(int(k) > 0 for k in (g.get("k_table") or {}).values()):
        return False, f"not gated on this chip ({chip}): the record has k = 0 at every row bucket"
    return True, f"gated on {chip} ({g.get('record')})"


def pick(meta, chip, records=None, forced=None, css=None, later=False):
    """meta = meta_of(...); records {kind: {chip: gate entry}}; forced None/"auto", "off" or a kind; css this
    deployment's component_set_sha with that drafter on: a function of the kind, a value, or None (not known: nothing is
    gated). A forced kind is returned where it applies, never gated, and no record is read for it. later=True (the
    engine is built after the pick, as cli.load does): the record's own component_set_sha stands in, and the caller
    checks the built engine's against it (gate_css) and turns speculation off before the warm-up when they differ.
    -> (kind or None, gated, why lines, gate_css)"""
    from . import ORDER
    if forced == "off":
        return None, False, ["speculation: off (--speculation off)"], None
    why = []
    kinds = ORDER if forced in (None, "auto") else (forced,)
    for kind in kinds:
        ok, w = applies(kind, meta)
        if not ok:
            why.append(w)
            continue
        if forced not in (None, "auto"):     # a drafter the user picked never reads the auto gate: no record changes it
            why.append(f"{w}; speculation: not gated on this chip")
            return kind, False, why, None
        rec = (records or {}).get(kind)
        c = css(kind) if callable(css) else css
        stand = c is None and later and chip is not None
        if stand:
            c = ((rec or {}).get(chip) or {}).get("component_set_sha")
        g, gw = gated(rec, chip, c)
        if not g:
            why.append(f"{w}; {gw}")
            continue
        why.append(f"{w}; {gw}")
        return kind, True, why, (c if stand else None)
    return None, False, why, None
