"""A model's drafter for speculative decoding (INTERFACES-v1 §10): its models.json declaration, the gate rule that turns
it on by default, and its part of the deployment key.

models.json, on the model row (absent = no drafter = speculation off):
    "drafter": {"kind": "mtp", "id": "...", "version": 1, "file": "<files[] name>", "k_default": 2, "k_max": 3,
                "gate": {"<chip key, e.g. v6e-1>": {"status": "passed" | "not_passed", "record": "<dated record>",
                                                    "engine_commit": "<sha>", "component_set_sha": "<hex>"}},
                "k_table": {"<chip key>": {"<row bucket>": k}}}      # optional: measured draft length per row bucket
A chip's k_table gives the draft length per decode row bucket (0 = verify without drafts) from a measured record; buckets
it does not name use the deployment's k.
The weights' sha256, repo and revision live once, on the files[] row the drafter names (kind "draft"); declared() joins
them. Only the drafter gate's record sets a status to "passed", for one engine commit and component set.
"""

HEAD_ABI = 1         # the interface a model's MTP head module implements (its SPEC_MTP_HEAD module's HEAD_ABI)
# The drafter interface (Drafter, DRAFTER_ABI) and the order of choice (ORDER) are in runtime/drafters/__init__.py, the
# pick in runtime/drafters/select.py: this file holds the declaration and the gate only.


def declared(m):
    """The model's drafter with its file row joined in (sha256, bytes, repo, revision), or None."""
    dr = m.get("drafter")
    if not dr:
        return None
    row = next((f for f in m.get("files", []) if f.get("name") == dr.get("file")), None)
    if row is None or row.get("kind") != "draft":
        raise ValueError(f"{m.get('id')}: drafter file {dr.get('file')!r} is not a files[] row of kind draft")
    out = dict(dr)
    out.update(sha256=row.get("sha256"), bytes=row.get("bytes"), repo=row.get("repo") or m.get("hf_repo"),
               revision=row.get("revision") or m.get("hf_revision"))
    return out


def chip_key():
    """This host's chip key as the gate records it ('v6e-1', 'v6e-4', 'v5e-4', 'v5p-8'), or None off TPU. v5p and v4
    accelerator names count TensorCores: two per device JAX lists."""
    import jax
    from .. import kernels as K
    fam = K.chip_family()
    if fam is None:
        return None
    return f"{fam}-{len(jax.devices()) * (2 if fam in ('v5p', 'v4') else 1)}"


def k_table(dr, chip):
    """This chip's measured draft length per row bucket ({rows: k}, k from 0 to k_max), or {} (k for every bucket).
    ValueError for a table that is not one."""
    if dr is None or chip is None:
        return {}
    kt = (dr.get("k_table") or {}).get(chip) or {}
    if not isinstance(kt, dict):
        raise ValueError(f"drafter k_table for {chip}: a {{rows: k}} map, not {type(kt).__name__}")
    out = {}
    for b, k in kt.items():
        if not str(b).isdigit() or int(b) < 1 or isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= dr["k_max"]:
            raise ValueError(f"drafter k_table for {chip}: {b!r}: {k!r} (rows >= 1, k 0 to {dr['k_max']})")
        out[int(b)] = k
    return out


def key(dr):
    """The drafter's part of component_set_sha (INTERFACES-v1 §6): kind, id@version, weights sha256, k rule and the
    speculation step version (runtime/spec_step.py SPEC_STEP_VERSION: a passed gate never covers a step it did not
    measure). None (speculation off) for no drafter."""
    if dr is None:
        return None
    from .spec_step import SPEC_STEP_VERSION
    return {"kind": dr["kind"], "id": f"{dr['id']}@{dr['version']}", "weights_sha256": dr["sha256"],
            "k": {"k_default": dr["k_default"], "k_max": dr["k_max"], "k_table": dr.get("k_table") or {}},
            "spec_step_version": SPEC_STEP_VERSION}


def gated_css(dr, chip):
    """The component_set_sha the drafter's gate on this chip passed for, or None where it has not passed (cli.load turns
    speculation on with it before the engine exists, then compares it with the built engine's component_set_sha and
    turns speculation off on a mismatch, before the warm-up)."""
    g = ((dr or {}).get("gate") or {}).get(chip) or {} if chip else {}
    return g.get("component_set_sha") if g.get("status") == "passed" else None


def gate(dr, chip, css):
    """-> (passed, reason). Passed only when the drafter's gate on this chip has status "passed" for this deployment's
    component_set_sha (css: the sha with this drafter on, or None when it is not known), and this chip's k_table, when
    it has one, has k > 0 at one or more row buckets (k = 0 at every bucket: never faster than plain, so not eligible)."""
    if dr is None:
        return False, "no drafter for this model"
    if chip is None:
        return False, "not on a TPU"
    g = (dr.get("gate") or {}).get(chip) or {}
    if g.get("status") != "passed":
        return False, f"not gated on this chip ({chip})"
    if css is None:
        return False, f"gated on {chip}, but this deployment's component_set_sha is not known"
    if g.get("component_set_sha") != css:
        return False, f"gated on {chip} for another component set"
    kt = (dr.get("k_table") or {}).get(chip)
    if isinstance(kt, dict) and kt and not any(isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in kt.values()):
        return False, f"not gated on this chip ({chip}): the k_table has k = 0 at every row bucket"
    return True, f"gated on {chip} ({g.get('record')})"
