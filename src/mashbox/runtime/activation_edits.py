"""Activation edits: the one engine interface through which a tool changes or reads what a model's layers compute, as
llama.cpp's control vectors and eval callback do for its tools.

A model definition implements it in its make_forward(config, ..., edits=None): with edits=None the forward is the program
built without the argument (no edit site at all); with an ActivationEdits the forward takes ed= (the per-call operand)
and returns out["edit_reads"]. A model module that implements it lists the sites it runs in ACTIVATION_EDIT_SITES (a
subset of SITES); the runtime and the tools read that list, never a model name.

Sites, in the order a block runs them:
  writer(site, l, call, lay, ctx) -> None or f(y32) -> y32: the f32 output of a residual writer before it is rounded:
      "embed" (l = -1, lay = None), "attn" (the attention output projection), "mlp" (the MLP down projection).
  act(l, call, lay, a, ctx) -> (a, reads): the MLP activation (the input of the down projection), (B, T, F).
  resid(l, call, lay, x, ctx) -> (x, reads): the residual stream leaving block l, (B, T, D).
SITES names them as "writer:embed", "writer:attn", "writer:mlp", "act", "resid".

op (the per-call operand, forward(..., ed=op)): {"call": any pytree, "layer": a pytree whose leaves have a leading axis
    of the model's layer count (row l belongs to layer l)}, or None. split_op gives (call, layer).
ctx: dict(pos (B, T) int32 positions, valid (B, T) bool, slots (B,), p0 (B,), lens (B,)) of this call.
l: the layer index (traced inside the layer loop), -1 for the embedding.
reads: a dict of arrays with the same names and shapes at every layer ({} = none); act's and resid's names must differ
    (merge_reads). The forward returns them stacked over the layers as out["edit_reads"][name], leading axis = layers.

Activation edits run on one chip without speculation: a model's make_forward refuses them with its tensor-parallel layout
and its forward refuses them with verify (check_forward_args). A prefix-cache point computed under edits belongs to the
edit variant of its owner key (runtime/engine.py), and compile keys carry the edits' description (runtime/compile_key.py).
"""

WRITER_SITES = ("embed", "attn", "mlp")
SITES = tuple(f"writer:{s}" for s in WRITER_SITES) + ("act", "resid")


class ActivationEdits:
    """The base edits object: it changes nothing and reads nothing, so a forward built with it gives outputs bit-identical
    to edits=None. A tool overrides the sites it needs. Every method runs inside the traced forward."""

    def writer(self, site, l, call, lay, ctx):
        return None

    def act(self, l, call, lay, a, ctx):
        return a, {}

    def resid(self, l, call, lay, x, ctx):
        return x, {}


def split_op(ed):
    """The per-call operand -> (call, layer); None -> (None, None)."""
    if ed is None:
        return None, None
    return ed.get("call"), ed.get("layer")


def merge_reads(ra, rr):
    """One layer's act reads + resid reads -> one dict; refuses names they share."""
    both = set(ra) & set(rr)
    if both:
        raise ValueError(f"activation edits: act and resid read the same names {sorted(both)}")
    return {**ra, **rr}


def check_make_forward_args(edits, tensor_parallel):
    """make_forward's refusals: edits must be an ActivationEdits, and they run on one chip."""
    if edits is None:
        return
    if not isinstance(edits, ActivationEdits):
        raise TypeError("edits must be an ActivationEdits (mashbox.runtime.activation_edits)")
    if tensor_parallel:
        raise ValueError("activation edits run on one chip (not in the tensor-parallel layout)")


def check_forward_args(edits, ed, verify):
    """forward's refusals: ed= needs edits, and edits run without speculation."""
    if edits is None and ed is not None:
        raise ValueError("ed= needs make_forward(edits=...)")
    if edits is not None and verify:
        raise ValueError("activation edits run without speculation (no verify pass)")


def model_sites(module):
    """The sites a model module implements (its ACTIVATION_EDIT_SITES), () when it implements none; refuses unknown
    names."""
    sites = tuple(getattr(module, "ACTIVATION_EDIT_SITES", ()))
    bad = [s for s in sites if s not in SITES]
    if bad:
        raise ValueError(f"{module.__name__}: unknown activation-edit sites {bad} (known: {list(SITES)})")
    return sites
