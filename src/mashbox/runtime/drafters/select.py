"""Which drafter a deployment uses, from metadata and records, never from a model name.

Order (one drafter per deployment):
  0 trained      a trained drafter (DFlash, DFlare or EAGLE3 block/head) whose drafter.json manifest (mbx-drafter/1)
                 declares this target: its tokenizer sha256 equals the engine's and its target shape equals the target's
                 (detect_trained; the target's hf_id is never read). Detected here; its layer math is a module in BUILT,
                 and a method not in BUILT gives a why line and never applies
  1 mtp          the file has an MTP head (BSHJ meta / models.json drafter row of kind mtp, GGUF <arch>.nextn_predict_layers
                 > 0 with its tensors, HF mtp_num_hidden_layers or num_nextn_predict_layers) and its architecture module
                 declares an MTP head implementation for this format (SPEC_MTP_HEAD)
  2 draft_model  a draft model given with --draft-model (a GGUF file with the target's tokenizer whose architecture
                 module declares SPEC_DRAFT, checked by cli.gguf_draft); catalog rows with the same tokenizer hash, dense,
                 at most 1/4 of the target's bytes are detected here (detect_draft_model), but loading one from the
                 catalog is not in this version, so without --draft-model it never applies
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


HEX = frozenset("0123456789abcdef")
TRAINED_METHODS = ("dflash", "dflare", "eagle3")     # the methods a trained drafter's manifest may name (mbx-drafter/1)
BUILT = {}          # method -> the module with its layer math (runtime/drafters/<module>.py); none in this version
SHAPE_KEYS = ("hidden", "layers", "heads", "kv_heads", "head_dim", "vocab")    # the target shape a drafter is built for


def _hex64(v):
    return isinstance(v, str) and len(v) == 64 and set(v) <= HEX


def hf_shape(cfg):
    """The target's shape (SHAPE_KEYS) from a Hugging Face config.json dict (its text_config where it has one): hidden
    size, layers, attention heads, K/V heads, head dim and vocabulary size. A missing key stays None (matches nothing)."""
    t = (cfg or {}).get("text_config") or cfg or {}
    h, n = t.get("hidden_size"), t.get("num_attention_heads")
    return dict(hidden=h, layers=t.get("num_hidden_layers"), heads=n, kv_heads=t.get("num_key_value_heads", n),
                head_dim=t.get("head_dim") or (h // n if h and n else None), vocab=t.get("vocab_size"))


def manifest_problem(m):
    """None for a usable trained-drafter manifest (drafter.json: format mbx-drafter/1, kind trained, a method of
    TRAINED_METHODS, the target's tokenizer sha256 and shape, the sha256 of every file, k_max 1 to 16), else the why"""
    if not isinstance(m, dict) or m.get("format") != "mbx-drafter/1":
        return "not an mbx-drafter/1 manifest"
    if m.get("kind") != "trained":
        return f"kind {m.get('kind')!r} is not trained"
    if m.get("method") not in TRAINED_METHODS:
        return f"method {m.get('method')!r} is not one of {', '.join(TRAINED_METHODS)}"
    t = m.get("target") if isinstance(m.get("target"), dict) else {}
    if not _hex64(t.get("tokenizer_sha256")):
        return "target.tokenizer_sha256 is not a sha256"
    sh = t.get("shape") if isinstance(t.get("shape"), dict) else {}
    if any(isinstance(sh.get(k), bool) or not isinstance(sh.get(k), int) or sh[k] <= 0 for k in SHAPE_KEYS):
        return f"target.shape needs positive integers for {', '.join(SHAPE_KEYS)}"
    files = m.get("files") if isinstance(m.get("files"), dict) else {}
    if not files or not all(_hex64(v) for v in files.values()):
        return "files needs the sha256 of every file"
    k = m.get("k_max")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 16:
        return f"k_max {k!r}: 1 to 16"
    return None


def detect_trained(manifests, tok_sha, shape):
    """Auto-detect the trained drafters for this target from their manifests: the manifest's target tokenizer sha256 equals
    this engine's tokenizer hash (tok_sha) and its target shape equals the target's (shape, SHAPE_KEYS). Never a name: the
    target's hf_id and the drafter's id are not read. -> (the matching manifests in the given order, why lines)"""
    out, why = [], []
    for i, m in enumerate(manifests or []):
        p = manifest_problem(m)
        tag = f"trained[{i}]" + (f" {m.get('method')}" if isinstance(m, dict) and m.get("method") else "")
        if p:
            why.append(f"{tag}: {p}")
        elif tok_sha is None or m["target"]["tokenizer_sha256"] != tok_sha:
            why.append(f"{tag}: declared for another tokenizer")
        elif any(m["target"]["shape"][k] != (shape or {}).get(k) for k in SHAPE_KEYS):
            why.append(f"{tag}: declared for another target shape")
        else:
            out.append(m)
    return out, why


def detect_draft_model(rows, tok_sha, target_bytes):
    """Draft-model siblings for this target from catalog rows ({tokenizer_sha256, bytes, experts}): the same tokenizer
    hash, dense, at most 1/4 of the target's bytes; the largest first. Never a name."""
    if tok_sha is None or not target_bytes:
        return []
    ok = [r for r in rows or [] if r.get("tokenizer_sha256") == tok_sha and not r.get("experts")
          and isinstance(r.get("bytes"), int) and not isinstance(r.get("bytes"), bool) and 0 < r["bytes"] * 4 <= target_bytes]
    return sorted(ok, key=lambda r: -r["bytes"])


def name_free(m):
    """a matched manifest as the pick sees it: without its origin and without the target's hf_id and revision, so no
    name reaches the choice (ST-S9)"""
    out = {k: v for k, v in m.items() if k not in ("origin", "target")}
    out["target"] = {k: v for k, v in m["target"].items() if k not in ("hf_id", "revision")}
    return out


def meta_of(model, cfg, fmt, mtp_layers=0, target=None, draft_given=False):
    """the metadata the choice reads: format, the architecture module's own declarations (SPEC_VERIFY: its forward has
    the verify pass and a commit; SPEC_MTP_HEAD: the module that implements its MTP head), MoE or dense (expert counts),
    MTP layers in the file, whether a draft model was given (--draft-model). Never a model or file name."""
    verify = any(getattr(m, "SPEC_VERIFY", False) for m in _modules(model))
    head = mtp_head(model)
    # MoE: an expert count (cfg.E) or the MoE flag a MoE config sets (cfg.moe), whichever the architecture's config has
    experts = int(getattr(cfg, "E", 0) or 0) or (1 if getattr(cfg, "moe", False) else 0)
    t = target or {}        # {tokenizer_sha256, shape, bytes, trained: [manifests], siblings: [catalog rows]}
    trained, twhy = detect_trained(t.get("trained"), t.get("tokenizer_sha256"), t.get("shape"))
    return dict(format=fmt, verify=bool(verify), mtp_head=bool(head) and fmt != "gguf", experts=experts,
                mtp_layers=int(mtp_layers or 0), draft_given=bool(draft_given), trained=[name_free(m) for m in trained], trained_why=twhy,
                draft_models=[dict(bytes=r["bytes"], sha256=r.get("sha256"))      # no catalog id or repo name
                              for r in detect_draft_model(t.get("siblings"), t.get("tokenizer_sha256"), t.get("bytes"))])


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
    if kind == "trained":
        found = meta.get("trained") or []
        if not found:
            return False, "; ".join(["trained: no trained drafter declares this target (tokenizer hash and shape)"]
                                    + list(meta.get("trained_why") or []))
        m = found[0]
        if m["method"] not in BUILT:
            return False, (f"trained: a {m['method']} drafter declared for this target is detected; its layer math is "
                           f"not built in this version")
        return True, f"trained: the {m['method']} drafter declared for this target"
    if kind == "draft_model":
        if meta.get("draft_given"):
            return True, "draft_model: the draft model given with --draft-model"
        n = len(meta.get("draft_models") or [])
        return False, ("draft_model: no draft model given (--draft-model)" + (
            f"; {n} same-tokenizer catalog sibling(s) detected, loading one from the catalog is not in this version"
            if n else ""))
    if kind == "ngram":
        return True, "ngram: prompt lookup over the row's own text (no weights)"
    return False, f"{kind}: unknown"


def sealed(g):
    """A gate entry counts only when its chip record is sealed: it names the record and the record's sha256 (64 hex),
    checked for form here (this function does not read the record file). An entry without them
    never turns speculation on, whatever its status says."""
    s = (g or {}).get("record_sha256")
    return bool((g or {}).get("record")) and isinstance(s, str) and len(s) == 64 and set(s) <= HEX


def catalog_records(reg, r):
    """The speculation gate records {kind: {chip: entry}} of the catalog's gguf[] row for this GGUF file (r: the GGUF
    resolver's dict), matched on the file's identity: repo, revision and file of a Hugging Face file, or the row's sha256
    where r carries the file's sha256. Never a name. {} when no row matches (plain decoding unless a flag forces one)."""
    for g in (reg or {}).get("gguf") or []:
        same_hub = r.get("repo") and (g.get("repo"), g.get("revision"), g.get("file")) == (r.get("repo"), r.get("revision"),
                                                                                          r.get("file"))
        same_sha = r.get("sha256") and g.get("sha256") == r.get("sha256")
        if same_hub or same_sha:
            return g.get("speculation") or {}
    return {}


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
    if not sealed(g):
        return False, f"not gated on this chip ({chip}): the record is not sealed (record and record_sha256)"
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
