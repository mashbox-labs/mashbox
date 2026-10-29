"""Drafters for speculative decoding (runtime/spec_step.py runs them inside the batched step). One drafter per deployment.

Each kind implements Drafter below (DRAFTER_ABI 2); runtime/drafter.py holds the models.json declaration and the gate.

Order of choice (select.py): the model's own MTP head, then a draft model from the catalog with the same tokenizer, then
n-gram over the row's own text. Chosen from metadata and records, never from a model name.
"""
from ..spec_step import SPEC_STEP_VERSION

DRAFTER_ABI = 2      # the drafter interface (class Drafter below; runtime/drafters/<kind>.py implement it)
ORDER = ("mtp", "draft_model", "ngram")      # every drafter kind, in the order of choice (select.py)

MODULES = {"mtp": "mashbox.runtime.drafters.mtp", "ngram": "mashbox.runtime.drafters.ngram"}


def key_base(kind, ident, weights_sha256, params, k):
    """The part every drafter's gate key has: kind, id@version, weights sha256 (None without weights), params, k rule and
    the speculation step version."""
    return {"kind": kind, "id": ident, "weights_sha256": weights_sha256, "params": params, "k": k,
            "spec_step_version": SPEC_STEP_VERSION}


class Drafter:
    """DRAFTER_ABI 2: what a drafter gives the engine's speculation step (runtime/spec_step.py make_step). Each kind is
    one file in runtime/drafters/; select.py picks one per deployment from metadata and gate records (select.pick).
      abi              the drafter interface it implements (DRAFTER_ABI; the engine refuses another)
      kind, k_max      its kind (one of ORDER) and the most drafts per round it supports
      W                its device weights (None for none)
      extra_groups     K/V groups it keeps in the engine's slot state (an MTP head: 1)
      needs_hidden     whether prefill returns the target's hidden states for it
      verify_kw        extra keyword arguments for the architecture's verify pass
      plain_ok         whether a plain decode call may run between its calls (its state is then rebuilt from the host)
      backoff          whether a row whose drafts keep failing drops to k = 0 for a while (the row's own history)
    Drafts are token ids only: the step never reads draft probabilities, so any drafter keeps the plain output."""
    abi = DRAFTER_ABI
    kind, k_max, W = None, 0, None
    extra_groups, needs_hidden, verify_kw, plain_ok, backoff = 0, False, {}, False, False

    def key(self):
        """its block of the component set, for what was loaded: kind, id@version, weights sha256 (None without
        weights), params, k rule and the speculation step version"""
        raise NotImplementedError

    def state(self, eng):
        """its device state for every slot of eng, made before ready"""
        raise NotImplementedError

    def on_prefill(self):
        """None, or a function the engine jits and runs over each prefill chunk:
        (W, ck, cv, DW, D, toks, hid, p0, lens, slots, hrow, hidx, Tc) -> (D, ck, cv)"""
        return None

    def carry_in(self, D, X, B):
        """its per-call carry inside the step program, from its state D and its host inputs X, for rows 0..B-1"""
        return D[:B]

    def carry_out(self, D, Dc, B):
        """the carry back into its state"""
        return D.at[:B].set(Dc)

    def draft(self, W, DW, ck, Dc, tok, pos, Tc, K, sp, st):
        """-> (d (B, K) i32 drafts, found (B,) i32 drafts per row or None, ck)"""
        raise NotImplementedError

    def after_commit(self, Dc, r, emit, n, go, tok, pos, K1):
        """its carry after each row committed its n tokens (rejected drafts never enter it) -> Dc"""
        raise NotImplementedError

    def host_args(self, eng, B, live, k):
        """X: a tuple of host arrays for one call"""
        return ()

    def host_zero(self, eng, B):
        """X for the warm-up"""
        return ()

    def host_after(self, eng, r, seq, last, p2):
        """the host side of one call for one row"""

    def programs(self, eng):
        """its own host-called programs, compiled by the warm-up: [(name, function, args after its state, kwargs)]"""
        return []

    def found(self, eng, r, K):
        """how many drafts row r would get now (the call-level plain choice of a plain_ok drafter)"""
        return K
