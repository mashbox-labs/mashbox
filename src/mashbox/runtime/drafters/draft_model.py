"""The draft-model drafter: a smaller model with the target's tokenizer drafts K tokens one at a time from its own K/V
cache. Any architecture module with the dense contract works as the draft (new_state(cfg, S, CTX), make_forward(cfg)
-> forward(W, ck, cv, toks, p0, lens, slots, Tc) -> dict(ck, cv, nxt, lg)), full attention only: positions past a row's
cursor are masked, so the draft cache needs no rollback.

State: the draft model's own (ck, cv) for every slot, holding the row's prompt and committed tokens at their positions
(0..pos-1). The host writes a row in full (one prefill, in CHUNK-token pieces) before the row's first speculation call
and after any plain decode call. Inside the step, draft() runs K one-token steps on a scratch copy (the drafts' K/V are
dropped with it) and after_commit writes only the n committed tokens, so rejected drafts leave no trace.
Sampled rows draft with the shared sampler's draw at the same answer index as the target (as the MTP drafter).
"""
import numpy as np
import jax
import jax.numpy as jnp

from .. import sampler as SP
from . import Drafter, key_base

f32, i32 = jnp.float32, jnp.int32
VERSION = 1
CHUNK = 128


class DraftModel(Drafter):
    kind = "draft_model"
    extra_groups = 0
    needs_hidden = False
    verify_kw = {}
    plain_ok = True         # its state is rebuilt from the host after a plain call (hist_ok)
    backoff = True

    def __init__(self, arch, cfg, W, k_max, declared=None):
        """arch: the draft model's architecture module, cfg its config, W its device weights; declared: the identity of
        the loaded draft weights (id and weights_sha256), given by the entry point that loaded them"""
        if getattr(cfg, "E", 0) or getattr(cfg, "moe", False):
            raise ValueError("speculation: a mixture-of-experts draft model is not supported")
        self.arch, self.cfg, self.W, self.k_max, self.declared = arch, cfg, W, int(k_max), declared
        self.fwd = arch.make_forward(cfg)
        self.pf, self._round = None, None

    def key(self):
        d = self.declared or {}
        return key_base("draft_model", f"{d.get('id')}@draft_model.{VERSION}", d.get("weights_sha256"),
                        {"vocab": int(self.cfg.V)}, {"k_max": self.k_max})

    def state(self, eng):
        if int(self.cfg.V) != int(eng.c.V):
            raise ValueError(f"speculation: the draft model's vocabulary ({self.cfg.V}) is not the target's ({eng.c.V})")
        fwd = self.fwd

        def pf(D, DW, toks, p0, lens, slots, Tc):
            r = fwd(DW, D[0], D[1], toks, p0, lens, slots, Tc)
            return r["ck"], r["cv"]
        self.pf = jax.jit(pf, static_argnames=("Tc",), donate_argnums=(0,))
        return jax.tree.map(eng._committed, self.arch.new_state(self.cfg, eng.S, eng.CTX))

    def carry_in(self, D, X, B):
        return D

    def carry_out(self, D, Dc, B):
        return Dc

    def draft(self, W, DW, ck, Dc, tok, pos, Tc, K, sp, st):
        B = tok.shape[0]
        slots, one = jnp.arange(B, dtype=i32), jnp.ones((B,), i32)
        self._round = (DW, Tc)   # after_commit runs in the same round body of the same trace and reads these
        dck, dcv = Dc            # a scratch copy: the drafts' K/V are dropped with it
        t, p, ds = tok, pos, []
        for k in range(K):
            r = self.fwd(DW, dck, dcv, t[:, None], p, one, slots, Tc)
            dck, dcv = r["ck"], r["cv"]
            lg = r["lg"].reshape(B, -1).astype(f32)
            t = jnp.argmax(lg, -1).astype(i32) if sp is None else SP.sample(lg, sp, st + k)
            ds.append(t)
            p = p + 1
        return jnp.stack(ds, 1), None, ck

    def after_commit(self, Dc, r, emit, n, go, tok, pos, K1):
        """the committed tokens at pos .. pos + n - 1 (the fed token and the accepted drafts) into the draft cache"""
        DW, Tc = self._round
        B = tok.shape[0]
        seq = jnp.concatenate([tok[:, None], emit[:, :K1 - 1]], 1)
        CTX = Dc[0][0].shape[2]
        rr = self.fwd(DW, Dc[0], Dc[1], seq, jnp.where(go, pos, CTX), jnp.maximum(n, 1), jnp.arange(B, dtype=i32), Tc)
        return rr["ck"], rr["cv"]

    # host side
    def host_args(self, eng, B, live, k):
        for s_ in live:
            r = eng.rows[s_]
            if not r.get("hist_ok"):
                seq = list(r["prompt"]) + list(r["ids"][:-1]) if r["ids"] else list(r["prompt"])
                eng.dstate = self.fill(eng.dstate, s_, seq[:eng.CTX])
                r["hist_ok"] = True
        return ()

    def fill(self, D, slot, seq):
        """the draft model over seq at positions 0..len-1 of slot, in CHUNK-token pieces"""
        for p0 in range(0, len(seq), CHUNK):
            ch = seq[p0:p0 + CHUNK]
            toks = np.zeros((1, CHUNK), np.int32)
            toks[0, :len(ch)] = ch
            D = self.pf(D, self.W, toks, np.asarray([p0], np.int32), np.asarray([len(ch)], np.int32),
                        np.asarray([slot], np.int32), Tc=p0)
        return D

    def host_zero(self, eng, B):
        return ()

    def host_after(self, eng, r, seq, last, p2):
        pass

    def programs(self, eng):
        """the host prefill at every chunk start, as warm() lists calls"""
        def z(v):
            return np.asarray([v], np.int32)
        return [("spec_fill", self.pf, (self.W, np.zeros((1, CHUNK), np.int32), z(p0), z(1), z(0)), dict(Tc=p0))
                for p0 in range(0, eng.CTX, CHUNK)]

    def on_prefill(self):
        return None
