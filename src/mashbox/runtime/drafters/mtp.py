"""The MTP drafter: a model's own multi-token-prediction head (one decoder layer in front of the shared lm_head). The head
math and its weight loader are the model's (a head module: make_mtp(cfg), load_mtp(path, cfg), HEAD_ABI of
runtime/drafter.py); this file runs it as a drafter of the engine's step.

Per round each row drafts from its carried input: the tokens it committed last round with the target's hidden states
there (a new row: its first token and the prompt's last hidden state), K drafts, K - 1 of them on the head's own output.
Sampled rows draft with the shared sampler's draw at the same answer index as the target, which raises acceptance.
"""
import numpy as np
import jax.numpy as jnp

from .. import sampler as SP
from . import Drafter

f32, bf16, i32 = jnp.float32, jnp.bfloat16, jnp.int32


def make_draft(mtp):
    """-> draft(W, Wm, ck, toks, hid, p0, lens, Tc, K, sp=None, step0=None) -> (d (B,K), dlg (B,K,V) f32, ck): the head
    over toks/hid at p0 (row b uses lens[b] positions; its last one predicts the first draft), then K - 1 steps on its
    own output."""
    def draft(W, Wm, ck, toks, hid, p0, lens, Tc, K, sp=None, step0=None):
        B = toks.shape[0]
        slots = jnp.arange(B, dtype=i32)
        h, lg, ck = mtp(W, Wm, ck, toks, hid, p0, lens, slots, Tc)
        h = h[jnp.arange(B), jnp.maximum(lens - 1, 0)]
        p = p0 + lens - 1
        ds, ls = [], []
        for k in range(K):
            lgf = lg.astype(f32)
            if sp is None:
                t = jnp.argmax(lg, -1).astype(i32)
            else:       # the shared sampler's draw at answer index step0 + k (Gumbel key (seed, index)): a draw from the
                t = SP.sample(lgf, sp, step0 + k)       # draft distribution, coupled to the target's draw at that index
            ds.append(t)
            ls.append(lgf)
            if k + 1 < K:
                p = p + 1
                hx, lg, ck = mtp(W, Wm, ck, t[:, None], h[:, None], p, jnp.ones((B,), i32), slots, Tc)
                h = hx[:, 0]
        return jnp.stack(ds, 1), jnp.stack(ls, 1), ck
    return draft


class MTP(Drafter):
    kind = "mtp"
    extra_groups = 1
    needs_hidden = True
    verify_kw = {}
    plain_ok = False        # the head's carry needs the target's hidden states of every committed token
    backoff = False

    def __init__(self, head, Wm, cfg, k_max, declared=None):
        from .. import drafter as DR
        if getattr(head, "HEAD_ABI", None) != DR.HEAD_ABI:
            raise ValueError(f"speculation: {head.__name__} implements MTP head ABI {getattr(head, 'HEAD_ABI', None)}, "
                             f"this engine {DR.HEAD_ABI}")
        self.head, self.W, self.cfg, self.k_max, self.declared = head, Wm, cfg, int(k_max), declared
        self.mfn = head.make_mtp(cfg)
        self._draft = make_draft(self.mfn)

    def key(self):
        """declared = the identity of what was loaded (drafter.key's form, bound to the bytes the engine loaded), given by
        the entry point that loaded the head; with this step's version. None when none was given."""
        if not self.declared:
            return None
        from ..spec_step import SPEC_STEP_VERSION
        return dict(self.declared, spec_step_version=SPEC_STEP_VERSION)

    # device state: each slot's carried hidden states (S, k_max + 1, D)
    def state(self, eng):
        return eng._committed(jnp.zeros((eng.S, self.k_max + 1, self.cfg.D), bf16))

    def carry_in(self, D, X, B):
        pt, pp, pn = X
        return (D[:B], pt, pp, pn)

    def carry_out(self, D, Dc, B):
        return D.at[:B].set(Dc[0])

    def draft(self, W, DW, ck, Dc, tok, pos, Tc, K, sp, st):
        h, pt, pp, pn = Dc
        d, _, ck = self._draft(W, DW, ck, pt, h, pp, pn, Tc, K, sp, st)
        return d, None, ck

    def after_commit(self, Dc, r, emit, n, go, tok, pos, K1):
        h, pt, pp, pn = Dc
        B, P = pt.shape

        def pad(x):
            return jnp.concatenate([x, jnp.zeros((B, P - K1) + x.shape[2:], x.dtype)], 1) if P > K1 else x
        h = jnp.where(go[:, None, None], pad(r["hid"].astype(h.dtype)), h)
        pt = jnp.where(go[:, None], pad(emit), pt)
        pp, pn = jnp.where(go, pos, pp), jnp.where(go, n, pn)
        return (h, pt, pp, pn)

    # host side
    def host_args(self, eng, B, live, k):
        P = self.k_max + 1
        pt, pp, pn = np.zeros((B, P), np.int32), np.full(B, eng.CTX, np.int32), np.zeros(B, np.int32)
        for s_ in live:
            r = eng.rows[s_]
            # a row's first decode step: the head reads its first token with the prompt's last hidden state (D[s_, 0])
            pd = r.get("pend") or ([int(eng.tok[s_])], int(eng.pos[s_]) - 1)
            pt[s_, :len(pd[0])], pp[s_], pn[s_] = pd[0], pd[1], len(pd[0])
        return pt, pp, pn

    def host_zero(self, eng, B):
        return np.zeros((B, self.k_max + 1), np.int32), np.full(B, eng.CTX, np.int32), np.zeros(B, np.int32)

    def host_after(self, eng, r, seq, last, p2):
        if last:
            r["pend"] = (seq[len(seq) - last:], int(p2) - last)

    def on_prefill(self):
        mfn = self.mfn

        def mtp_pf(W, ck, cv, Wm, ph, tn, hid, p0, lm, sl, hrow, hidx, Tc):
            """The MTP head over a prefill chunk (its K/V group), and the prompt's last hidden state of each row whose
            prompt ends in this chunk into its carried slot (hrow = slot, S for the others: dropped)."""
            _, _, ck = mfn(W, Wm, ck, tn, hid, p0, lm, sl, Tc)
            hl = hid[jnp.arange(hid.shape[0]), hidx].astype(ph.dtype)
            return ph.at[hrow, 0].set(hl, mode="drop"), ck, cv
        return mtp_pf
