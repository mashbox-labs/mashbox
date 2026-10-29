"""Speculative decoding inside the batched step, for every model the engine runs: one program per call runs N rounds of
draft, verify, accept, commit and after_commit for every decoding row of a row bucket.

The three parts and who provides them:
  drafter   runtime/drafters/<kind>.py: token ids only (draft) and its own state (after_commit). The coupled draw never
            reads draft probabilities, so any drafter gives the plain output (greedy: the tie rule; sampled: the same draw).
  verify    the architecture's forward(verify=True) over [token, d1..dK] at positions pos..pos+K of contiguous rows:
            out["tgt"] (B, K+1) argmax, out["lg"] (B, K+1, V) f32, out["rec"] what the commit needs, out["hid"] when asked.
  commit    arch_commit(model, cfg): keeps exactly the n accepted steps of the row's state. Full attention needs nothing
            (rejected K/V sit past the row's position: masked, then written over); linear-attention layers replay n steps
            from the state before the verify pass (gated_delta_commit + conv_commit).

SPEC_STEP_VERSION is part of every drafter's gate key (drafters.key): it is bumped whenever the meaning of verify, accept,
commit or after_commit changes, so a passed chip record never covers a step it did not measure. Version 1 is the step as
the first MTP chip records measured it (one round: draft, verify, accept by greedy prefix or the coupled draw, commit, carry).
"""
import jax
import jax.numpy as jnp
from jax import lax

from . import sampler as SP

f32, i32 = jnp.float32, jnp.int32

SPEC_STEP_VERSION = 1


def arch_commit(model, cfg):
    """-> commit(cv, rec, n) for the architecture the engine runs (n (B,) accepted steps per contiguous row), from the
    module the engine got or the module that defines its forward: its own spec_commit(cfg) when it has one; else the
    hybrid linear-attention commit when it has those functions; else the identity (full attention only), which refuses
    at trace time a verify pass that returned state to commit."""
    import sys
    mods = [model, sys.modules.get(getattr(getattr(model, "make_forward", None), "__module__", None) or "")]
    for m in mods:
        own = getattr(m, "spec_commit", None)
        if own is not None:
            return own(cfg)
    for m in mods:
        if hasattr(m, "gated_delta_commit") and hasattr(m, "conv_commit"):
            rep = cfg.HV // cfg.HK

            def commit(cv, rec, n, m=m):
                R, C = cv
                R = m.gated_delta_commit(R, rec, n, rep)
                C = m.conv_commit(C, rec["x"], n)
                return R, C
            return commit

    def identity(cv, rec, n):
        if jax.tree.leaves(rec):
            raise ValueError("speculation: the architecture's verify pass returned state to commit, and it has no commit")
        return cv
    return identity


def coupled(tlg, tgt, d, cnt, seen, sp, step0, kc):
    """The coupled draw (Daliri, Musco, Suresh, arXiv 2408.07978). The target at answer
    index step0 + j is the draw the non-speculative sampled program makes there: the same penalties (counts and seen
    tokens that include the drafts before j, as plain decode would have counted them), the same transform and the same
    Gumbel key (seed, index) through runtime/sampler.py sample(); greedy rows take the same argmax. Drafts are accepted
    while they equal the targets. Every committed token is then a target, so the output equals non-speculative output for
    any k and drafter; no rejection sampling. tlg (B, K+1, V) f32 verify logits, tgt (B, K+1) their argmax, d (B, K)
    drafts, cnt (B, V) f32 / seen (B, V) bool the row's counts before this round, step0 (B,) answer index of the first new
    token, kc (B,) draft caps -> (targets (B, K+1), a (B,) accepted drafts)."""
    B, K1, _ = tlg.shape
    K = K1 - 1
    b_ = jnp.arange(B)
    plain = SP.no_penalty(sp)
    tg = []
    for j in range(K1):
        x = SP.penalise(tlg[:, j], cnt, seen, sp)
        for i in range(j):     # drafts before j as counted tokens: same elementwise penalty at their ids, with their counts
            v = d[:, i]
            c_ = cnt[b_, v] + jnp.sum((d[:, :j] == v[:, None]).astype(f32), 1)
            x = x.at[b_, v].set(SP.penalise(tlg[b_, j, v][:, None], c_[:, None], jnp.ones((B, 1), bool), sp)[:, 0])
        tg.append(SP.sample(x, sp, step0 + j, greedy_tok=jnp.where(plain, tgt[:, j], jnp.argmax(x, -1).astype(i32))))
    tg = jnp.stack(tg, 1)
    ok = (d == tg[:, :K]) & (jnp.arange(K)[None] < kc[:, None])
    return tg, jnp.sum(jnp.cumprod(ok.astype(i32), 1), 1)


def make_step(fwd, commit, drafter):
    """-> step(W, ck, cv, DW, D, X, tok, pos, kr, rem, sp, step0, cnt, seen, Tc, K, N, trace)
          -> (out (N,B,K+1), ck, cv, D, n (N,B), tok, pos, cnt, seen, tr)
    fwd = the architecture's forward (plain, not jitted), commit = arch_commit(...), drafter = the drafter.
    N rounds of draft / verify / accept / commit for rows 0..B-1 of the slot state (one engine step). DW the drafter's
    weights (None for none), D its device state for all S slots, X its per-call host inputs (a tuple, () for none). Per
    row b: tok / pos the row's last token and its position (CTX for padding: its K/V writes fall past the cache and are
    dropped); kr (B,) the row's own draft cap (0..K); rem (B,) tokens it may still emit (0 = padding); sp / step0 / cnt /
    seen the sampled program's settings, first answer index and penalty counts (None for the greedy program).
    out[i, b, :n[i, b]] are the tokens row b committed in round i. A row with kc = min(kr, rem - 1, drafts found) <= 0
    drafts nothing and commits its plain next token. Sampled and penalised rows verify by the coupled draw (coupled()):
    each committed token is the shared sampler's draw at its answer index with its penalties (same key as the
    non-speculative sampled program), so their output does not depend on speculation either. trace=True also returns,
    per committed position, the verify logits' hash and top-1 / top-2 gap (N,B,K+1) each."""

    def step(W, ck, cv, DW, D, X, tok, pos, kr, rem, sp, step0, cnt, seen, Tc, K, N, trace=False):
        from .engine import logit_hash, top2_gap
        B = tok.shape[0]
        K1 = K + 1
        slots, b_ = jnp.arange(B, dtype=i32), jnp.arange(B)

        def one(i, carry):
            ck, cv, Dc, tok, pos, rem, st, out, cn, cnt, seen, tr = carry
            d, kd, ck = drafter.draft(W, DW, ck, Dc, tok, pos, Tc, K, sp, st)
            r = fwd(W, ck, cv, jnp.concatenate([tok[:, None], d], 1), pos, jnp.full((B,), K1, i32), slots, Tc, contig=True,
                    verify=True, **drafter.verify_kw)
            kc = jnp.clip(jnp.minimum(kr, rem - 1), 0, K)
            if kd is not None:          # drafts this drafter found for the row (n-gram: 0 without a match)
                kc = jnp.minimum(kc, kd)
            if sp is None:
                tgt = r["tgt"]
                a = jnp.minimum(jnp.sum(jnp.cumprod((d == tgt[:, :K]).astype(i32), 1), 1), kc)
                bonus = jnp.take_along_axis(tgt, a[:, None], 1)[:, 0]
            else:       # the coupled draw: every committed token is the plain sampler's draw at its answer index
                tg, a = coupled(r["lg"], r["tgt"], d, cnt, seen, sp, st, kc)
                bonus = jnp.take_along_axis(tg, a[:, None], 1)[:, 0]
            n = jnp.where(rem > 0, a + 1, 0)
            emit = jnp.where(jnp.arange(K1)[None] < a[:, None], jnp.concatenate([d, d[:, :1]], 1), bonus[:, None])
            cv = commit(r["cv"], r["rec"], n)
            if sp is not None:
                for j in range(K1):
                    m = j < n
                    cnt = cnt.at[b_, emit[:, j]].add(m.astype(f32))
                    seen = seen.at[b_, emit[:, j]].set(seen[b_, emit[:, j]] | m)
            if trace:
                lg = r["lg"].reshape(B * K1, -1)
                tr = (tr[0].at[i].set(logit_hash(lg).reshape(B, K1)), tr[1].at[i].set(top2_gap(lg).reshape(B, K1)))
            go = n > 0
            Dc = drafter.after_commit(Dc, r, emit, n, go, tok, pos, K1)
            tok = jnp.where(go, bonus, tok)
            out = out.at[i].set(jnp.where(jnp.arange(K1)[None] < n[:, None], emit, -1))
            return (r["ck"], cv, Dc, tok, pos + n, rem - n, None if st is None else st + n, out, cn.at[i].set(n), cnt, seen,
                    tr)
        tr0 = (jnp.zeros((N, B, K1), jnp.uint32), jnp.zeros((N, B, K1), f32)) if trace else None
        init = (ck, cv, drafter.carry_in(D, X, B), tok, pos, rem, step0, jnp.zeros((N, B, K1), i32), jnp.zeros((N, B), i32), cnt,
                seen, tr0)
        ck, cv, Dc, tok, pos, rem, _, out, cn, cnt, seen, tr = lax.fori_loop(0, N, one, init)
        return out, ck, cv, drafter.carry_out(D, Dc, B), cn, tok, pos, cnt, seen, tr
    step.__name__ = "step"
    return step
