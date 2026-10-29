"""Speculative decoding with Qwen3.8-27B's own multi-token-prediction (MTP) head.

The MTP head (config mtp_num_hidden_layers = 1) is one gated full-attention decoder layer in front of the shared lm_head:
  h = fc([pre_fc_norm_embedding(embed(t_{j+1})), pre_fc_norm_hidden(h_j)]) -> layer at position j -> norm -> lm_head
where h_j is the target's post-norm hidden state at position j (vLLM qwen3_5_mtp.py, SGLang qwen3_next_mtp.py). The same
layer is reused for every draft step, fed its own output hidden state. Its K/V live in K/V group NG of the engine state.

One round (all rows at once, K drafts):
  verify  the target reads [t, d1..dK] at positions p..p+K (model.forward verify=True: full attention writes its K/V, the
          linear-attention layers run K+1 exact one-token steps and keep their inputs, nothing is committed)
  accept  greedy: the longest prefix of drafts equal to the target's argmax, then the target's own token (a + 1 tokens);
          sampled: standard rejection sampling (Leviathan et al. 2023, Chen et al. 2023), exact in distribution
  commit  replay the a + 1 accepted steps of every linear-attention layer from the old state (model.gated_delta_commit); the
          conv tails take the matching window. Full-attention K/V past p + a are masked by position and written over later.
  draft   the MTP head reads the accepted tokens with the target's hidden states (positions p..p+a), then K - 1 more steps on
          its own output -> the next K drafts.

Inside the batched engine (make_step, Engine._spec_step) the same round runs per row, in the order draft, verify, accept,
commit, so a row that joined at this step and a row that has decoded for a while share one program: each row carries its
carried MTP input (the tokens it committed last round, their positions and the target's hidden states there; for a new row
its first token and the prompt's last hidden state). Each row has its own draft cap kc = min(k_row, remaining - 1): rows
with speculation off (k_row = 0: penalised or constrained rows) and rows within k of max_tokens draft fewer, and a row with
nothing left (padding, a finished row) commits nothing (n = 0), so its state is left as it was.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import model as M
from ... import bshj
from ...runtime import sampler as SP

f32, bf16, i32 = jnp.float32, jnp.bfloat16, jnp.int32


# ---------------------------------------------------------------------------------------------------------------- weights
DRAFTER_ABI = 1      # runtime/drafter.py DRAFTER_ABI: the engine refuses a spec module with another value
KIND = "mtp"


def load_mtp(path, c, log=None):
    """MTP weights from a .bshj with the upstream names (mtp.*): big linears int8 per row + f32 scale, or bf16."""
    f = bshj.File(path)
    t0, nbytes, nel = time.time(), 0, 0

    def one(n):
        nonlocal nbytes, nel
        f.advise(n)
        if n + "_scale" in f.tensors:
            x = dict(q=jnp.asarray(f.array(n)), s=jnp.asarray(f.array(n + "_scale")))
            nbytes += x["q"].nbytes + x["s"].nbytes; nel += x["q"].size
        else:
            x = jnp.asarray(f.array(n))
            nbytes += x.nbytes; nel += x.size
        return x
    if "mtp.fc.weight" not in f.tensors:
        raise ValueError(f"{path}: no mtp.* tensors")
    L = "mtp.layers.0."
    Wm = dict(fc=one("mtp.fc.weight"), pe=one("mtp.pre_fc_norm_embedding.weight"), ph=one("mtp.pre_fc_norm_hidden.weight"),
              ln1=one(L + "input_layernorm.weight"), ln2=one(L + "post_attention_layernorm.weight"), norm=one("mtp.norm.weight"),
              wq=one(L + "self_attn.q_proj.weight"), wk=one(L + "self_attn.k_proj.weight"), wv=one(L + "self_attn.v_proj.weight"),
              wo=one(L + "self_attn.o_proj.weight"), qn=one(L + "self_attn.q_norm.weight"), kn=one(L + "self_attn.k_norm.weight"),
              wg=one(L + "mlp.gate_proj.weight"), wu=one(L + "mlp.up_proj.weight"), wd=one(L + "mlp.down_proj.weight"))
    jax.block_until_ready(Wm)
    draft = ((f.meta() or {}).get("bshj") or {}).get("draft") or {}     # {"method": "mtp", "default_k": K} from the maker
    info = dict(load_s=round(time.time() - t0, 2), device_bytes=int(nbytes), n_weights=int(nel),
                bpw=round(nbytes * 8 / max(1, nel), 3), file=path.rsplit("/", 1)[-1], default_k=draft.get("default_k"))
    f.close()
    if log:
        log(info)
    return Wm, info


# ---------------------------------------------------------------------------------------------------------------- MTP head
def make_mtp(c):
    """-> mtp(W, Wm, ck, toks, hid, p0, lens, slots, Tc) -> (h (B,T,D) post-norm, logits at row lens-1 (B,V) bf16, ck).
    toks (B,T) = the token after each position, hid (B,T,D) = the hidden state at each position, positions p0 + 0..T-1."""
    H, KV, HD, RD, NG, eps = c.H, c.KV, c.HD, c.RD, c.NG, c.eps
    G, scale = H // KV, HD ** -0.5
    inv = (1.0 / (np.float32(c.theta) ** (np.arange(0, RD, 2, dtype=np.int64).astype(np.float32) / np.float32(RD)))).astype(np.float32)

    def mtp(W, Wm, ck, toks, hid, p0, lens, slots, Tc):
        B, T = toks.shape
        Kc, Vc = ck
        pos = p0[:, None] + jnp.arange(T, dtype=i32)[None]
        valid = jnp.arange(T)[None] < lens[:, None]
        fr = pos.astype(f32)[..., None] * jnp.asarray(inv)
        emb = jnp.concatenate([fr, fr], -1)
        cos, sin = jnp.cos(emb).astype(bf16)[:, :, None], jnp.sin(emb).astype(bf16)[:, :, None]
        m2 = (jnp.arange(T)[None, :, None] >= jnp.arange(T)[None, None, :]) & valid[:, None, :]
        if Tc:
            m1 = jnp.broadcast_to((jnp.arange(Tc)[None, :] < p0[:, None])[:, None, :], (B, T, Tc))
            mask = jnp.concatenate([m1, m2], -1)
        else:
            mask = m2
        e = M.rms_norm(M.embed(W["embed"], toks), Wm["pe"], eps)
        x = M.linear(jnp.concatenate([e, M.rms_norm(hid.astype(bf16), Wm["ph"], eps)], -1), Wm["fc"])
        h = M.rms_norm(x, Wm["ln1"], eps)
        qg = M.linear(h, Wm["wq"]).reshape(B, T, H, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:].reshape(B, T, H * HD)
        k, v = M.linear(h, Wm["wk"]).reshape(B, T, KV, HD), M.linear(h, Wm["wv"]).reshape(B, T, KV, HD)
        q, k = M.rms_norm(q, Wm["qn"], eps), M.rms_norm(k, Wm["kn"], eps)
        q, k = M.rope_partial(q, cos, sin, RD), M.rope_partial(k, cos, sin, RD)
        o = M.attend_one(Kc, Vc, NG, q.reshape(B, T, KV, G, HD), k, v, mask, slots, Tc, scale).reshape(B, T, H * HD)
        o = o * jax.nn.sigmoid(gate.astype(f32)).astype(bf16)
        x = x + M.linear(o, Wm["wo"])
        Kc, Vc = M.kv_write_group(Kc, Vc, NG, slots, pos, k, v)
        h = M.rms_norm(x, Wm["ln2"], eps)
        a = jax.nn.silu(M.linear(h, Wm["wg"]).astype(f32)).astype(bf16) * M.linear(h, Wm["wu"])
        x = M.rms_norm(x + M.linear(a, Wm["wd"]), Wm["norm"], eps)
        lg = M.linear(x[jnp.arange(B), jnp.maximum(lens - 1, 0)], W["head"])
        return x, lg, (Kc, Vc)
    return mtp


# ---------------------------------------------------------------------------------------------------------------- sampling
def _key(seed, step, salt):
    return jax.random.fold_in(jax.random.fold_in(jax.random.wrap_key_data(seed, impl="threefry2x32"), step), salt)


def _gumbel_pick(logp, seed, step, salt):
    """one draw per row from log-probabilities logp (B, V); keyed by (seed, token index, salt) per row"""
    def one(row, sd, st):
        return jnp.argmax(row + jax.random.gumbel(_key(sd, st, salt), row.shape, f32)).astype(i32)
    return jax.vmap(one)(logp, seed, step)


def _uniform(seed, step, salt):
    return jax.vmap(lambda sd, st: jax.random.uniform(_key(sd, st, salt), (), f32))(seed, step)


def probs(lg, sp):
    """processed sampling distribution (temperature, top_k, top_p, min_p of the shared sampler) for (B, V) logits"""
    return jax.nn.softmax(SP.transform(lg, sp), -1)


def accept_sampled(tlg, dlg, d, sp, step0, kcap=None):
    """Rejection sampling. tlg (B,K+1,V) f32 target logits, dlg (B,K,V) f32 draft logits, d (B,K) drafts (drawn from the
    draft distribution), step0 (B,) index of the first new token -> (a (B,) accepted drafts, bonus (B,)).
    Greedy rows (temperature 0): accept while d == target argmax, bonus = target argmax at the first miss.
    kcap (B,) or None: row b checks only its first kcap[b] drafts; when all of those are accepted its bonus is drawn from the
    target distribution itself (as at a == K), not from the residual."""
    B, K1, V = tlg.shape
    K = K1 - 1
    greedy = sp["temperature"] <= 0.0
    rep = lambda x, n: jnp.repeat(x, n, axis=0)
    spk = {k_: rep(v_, K1) for k_, v_ in sp.items()}
    p = probs(tlg.reshape(B * K1, V), spk).reshape(B, K1, V)
    spd = {k_: rep(v_, K) for k_, v_ in sp.items()}
    q = probs(dlg.reshape(B * K, V), spd).reshape(B, K, V)
    targ = jnp.argmax(tlg, -1).astype(i32)
    pd = jnp.take_along_axis(p[:, :K], d[..., None], -1)[..., 0]
    qd = jnp.take_along_axis(q, d[..., None], -1)[..., 0]
    u = jnp.stack([_uniform(sp["seed"], step0 + k, 101) for k in range(K)], 1)
    ok = jnp.where(greedy[:, None], d == targ[:, :K], u * qd <= pd)
    if kcap is not None:
        ok = ok & (jnp.arange(K)[None] < kcap[:, None])
    a = jnp.sum(jnp.cumprod(ok.astype(i32), 1), 1)
    pa = jnp.take_along_axis(p, a[:, None, None], 1)[:, 0]
    qa = jnp.take_along_axis(jnp.concatenate([q, jnp.zeros_like(q[:, :1])], 1), a[:, None, None], 1)[:, 0]
    if kcap is not None:
        qa = jnp.where((a < kcap)[:, None], qa, 0.0)
    r = jnp.maximum(pa - qa, 0.0)                       # a == K: q row is zero, r = p
    r = jnp.where(jnp.sum(r, -1, keepdims=True) > 0, r, pa)
    bonus = _gumbel_pick(jnp.log(r), sp["seed"], step0 + a, 202)
    return a, jnp.where(greedy, jnp.take_along_axis(targ, a[:, None], 1)[:, 0], bonus)


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


# ---------------------------------------------------------------------------------------------------------------- rounds
def make_spec(c, fwd, mtp):
    """-> (draft, rounds). fwd = model.make_forward(...) (plain, not jitted), mtp = make_mtp(c).

    draft(W, Wm, ck, toks, hid, p0, lens, Tc, K, sp=None, step0=None) -> (d (B,K), dlg (B,K,V) f32, ck): the MTP head over
    toks/hid at p0 (row b uses lens[b] positions; its last one predicts the first draft), then K - 1 steps on its own output.
    rounds(W, Wm, ck, cv, tok, d, dlg, pos, Tc, K, N, sp=None, step0=None) -> (out (N,B,K+1), cnt (N,B), ck, cv, tok, d, dlg, pos):
    N rounds of verify / accept / commit / draft; round i emits out[i, b, :cnt[i, b]]."""
    rep = c.HV // c.HK

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

    def rounds(W, Wm, ck, cv, tok, d, dlg, pos, Tc, K, N, sp=None, step0=None):
        B = tok.shape[0]
        slots = jnp.arange(B, dtype=i32)
        K1 = K + 1

        def one(i, carry):
            ck, cv, tok, d, dlg, pos, st, out, cnt = carry
            vt = jnp.concatenate([tok[:, None], d], 1)
            r = fwd(W, ck, cv, vt, pos, jnp.full((B,), K1, i32), slots, Tc, contig=True, verify=True)
            if sp is None:
                tgt = r["tgt"]
                a = jnp.sum(jnp.cumprod((d == tgt[:, :K]).astype(i32), 1), 1)
                bonus = jnp.take_along_axis(tgt, a[:, None], 1)[:, 0]
            else:
                a, bonus = accept_sampled(r["lg"], dlg, d, sp, st)
            n = a + 1
            emit = jnp.where(jnp.arange(K1)[None] < a[:, None], jnp.concatenate([d, d[:, :1]], 1), bonus[:, None])
            R, C = r["cv"]
            rec = r["rec"]
            R = M.gated_delta_commit(R, rec, n, rep)
            C = M.conv_commit(C, rec["x"], n)
            nd, ndlg, ck2 = draft(W, Wm, r["ck"], emit, r["hid"], pos, n, Tc, K, sp, None if st is None else st + n)
            return (ck2, (R, C), bonus, nd, ndlg, pos + n, None if st is None else st + n,
                    out.at[i].set(emit), cnt.at[i].set(n))
        init = (ck, cv, tok, d, dlg, pos, step0, jnp.zeros((N, B, K1), i32), jnp.zeros((N, B), i32))
        ck, cv, tok, d, dlg, pos, _, out, cnt = lax.fori_loop(0, N, one, init)
        return out, cnt, ck, cv, tok, d, dlg, pos
    return draft, rounds


# ---------------------------------------------------------------------------------------------------------------- batched step
def make_step(c, fwd, mtp):
    """-> step(W, ck, cv, Wm, ph, pt, pp, pn, tok, pos, kr, rem, sp, step0, cnt, seen, Tc, K, N, trace)
          -> (out (N,B,K+1), ck, cv, ph, n (N,B), tok, pos, cnt, seen, tr)
    N rounds of draft / verify / accept / commit for rows 0..B-1 of the slot state (one engine step). Per row b:
      ph (S, P, D) bf16 carried hidden states (rows 0..B-1 used; P >= K + 1), pt (B, P) carried tokens, pp (B,) their first
      position, pn (B,) how many (0 = padding: the row commits nothing); tok / pos the row's last token and its position
      (CTX for padding: its K/V writes fall past the cache and are dropped); kr (B,) the row's own draft cap (0..K); rem (B,)
      tokens it may still emit (0 = padding); sp / step0 / cnt / seen the sampled program's settings, first answer index and
      penalty counts (None for the greedy program).
    out[i, b, :n[i, b]] are the tokens row b committed in round i. A row with kc = min(kr, rem - 1) <= 0 drafts nothing and
    commits its plain next token. Sampled and penalised rows verify by the coupled draw (coupled()): each committed token
    is the shared sampler's draw at its answer index with its penalties (same key as the non-speculative sampled
    program), so their output does not depend on speculation either. trace=True also returns, per committed
    position, the verify logits' hash and top-1 / top-2 gap (N,B,K+1) each."""
    draft, _ = make_spec(c, fwd, mtp)
    rep = c.HV // c.HK

    def step(W, ck, cv, Wm, ph, pt, pp, pn, tok, pos, kr, rem, sp, step0, cnt, seen, Tc, K, N, trace=False):
        from ...runtime.engine import logit_hash, top2_gap
        B, P = pt.shape
        K1 = K + 1
        slots, b_ = jnp.arange(B, dtype=i32), jnp.arange(B)
        pad = lambda x: jnp.concatenate([x, jnp.zeros((B, P - K1) + x.shape[2:], x.dtype)], 1) if P > K1 else x

        def one(i, carry):
            ck, cv, h, pt, pp, pn, tok, pos, rem, st, out, cn, cnt, seen, tr = carry
            d, dlg, ck = draft(W, Wm, ck, pt, h, pp, pn, Tc, K, sp, st)
            r = fwd(W, ck, cv, jnp.concatenate([tok[:, None], d], 1), pos, jnp.full((B,), K1, i32), slots, Tc, contig=True,
                    verify=True)
            kc = jnp.clip(jnp.minimum(kr, rem - 1), 0, K)
            if sp is None:
                tgt = r["tgt"]
                a = jnp.minimum(jnp.sum(jnp.cumprod((d == tgt[:, :K]).astype(i32), 1), 1), kc)
                bonus = jnp.take_along_axis(tgt, a[:, None], 1)[:, 0]
            else:       # the coupled draw: every committed token is the plain sampler's draw at its answer index
                tg, a = coupled(r["lg"], r["tgt"], d, cnt, seen, sp, st, kc)
                bonus = jnp.take_along_axis(tg, a[:, None], 1)[:, 0]
            n = jnp.where(rem > 0, a + 1, 0)
            emit = jnp.where(jnp.arange(K1)[None] < a[:, None], jnp.concatenate([d, d[:, :1]], 1), bonus[:, None])
            R, C = r["cv"]
            R = M.gated_delta_commit(R, r["rec"], n, rep)
            C = M.conv_commit(C, r["rec"]["x"], n)
            if sp is not None:
                for j in range(K1):
                    m = j < n
                    cnt = cnt.at[b_, emit[:, j]].add(m.astype(f32))
                    seen = seen.at[b_, emit[:, j]].set(seen[b_, emit[:, j]] | m)
            if trace:
                lg = r["lg"].reshape(B * K1, -1)
                tr = (tr[0].at[i].set(logit_hash(lg).reshape(B, K1)), tr[1].at[i].set(top2_gap(lg).reshape(B, K1)))
            go = n > 0
            h = jnp.where(go[:, None, None], pad(r["hid"].astype(h.dtype)), h)
            pt = jnp.where(go[:, None], pad(emit), pt)
            pp, pn = jnp.where(go, pos, pp), jnp.where(go, n, pn)
            tok = jnp.where(go, bonus, tok)
            out = out.at[i].set(jnp.where(jnp.arange(K1)[None] < n[:, None], emit, -1))
            return (r["ck"], (R, C), h, pt, pp, pn, tok, pos + n, rem - n, None if st is None else st + n, out, cn.at[i].set(n),
                    cnt, seen, tr)
        tr0 = (jnp.zeros((N, B, K1), jnp.uint32), jnp.zeros((N, B, K1), f32)) if trace else None
        init = (ck, cv, ph[:B], pt, pp, pn, tok, pos, rem, step0, jnp.zeros((N, B, K1), i32), jnp.zeros((N, B), i32), cnt, seen,
                tr0)
        ck, cv, h, pt, pp, pn, tok, pos, rem, _, out, cn, cnt, seen, tr = lax.fori_loop(0, N, one, init)
        return out, ck, cv, ph.at[:B].set(h), cn, tok, pos, cnt, seen, tr
    return step
