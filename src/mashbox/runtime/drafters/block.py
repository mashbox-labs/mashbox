"""The block drafter: one causal draft pass per round over a block of 16, from the target's own hidden states.

Not picked by `--speculation auto`. Method: the JetSpec++ paper (arXiv 2610.19792). Weights: the drafter folder it is
given (config.json + model.safetensors), loaded unchanged; the model's catalog row names it.
The draft model has 5 decoder layers, an fc over the concat of the target's tapped layer outputs, and reads the target's
embed rows and lm_head. The layer math is the one drafters/dflash.py runs (rope, rms, linear) without DFlash 2's
convolutions and candidate selector, with the paper's two changes:
  causal    inside the block a query at slot i sees block keys j <= i (DFlash 2 sees the whole block)
  inputs    block = [anchor, bs - 1 tokens drawn uniformly from the row's committed positions 0 .. pos] (noise_source
            "context"); one pass; the draft at slot i is the argmax of the target's head on slot i's output
Full attention: the context ring holds W = the engine's context positions, so no position ever leaves it (W >= CTX).
The noise draws change only what is proposed; the verify pass decides every committed token.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import Drafter
from .dflash import linear, rms, rope, rope_tables, swap_rows, forget_row

f32, bf16, i32 = jnp.float32, jnp.bfloat16, jnp.int32
NEG = -1e30
METHOD = "block"
IMPL = "block@1"


class Config:
    def __init__(self, d, window):
        c = json.load(open(os.path.join(d, "config.json")))
        dc = c.get("dflash_config", {})
        g = lambda k, default=None: dc.get(k, c.get(k, default))
        if not g("causal_head"):
            raise NotImplementedError("block drafter: the draft model's config has no causal_head (a DFlash drafter: use dflash)")
        lt = c.get("layer_types") or ["full_attention"] * c["num_hidden_layers"]
        if set(lt) != {"full_attention"}:
            raise NotImplementedError(f"block drafter: layer types {sorted(set(lt))}")
        self.L, self.D, self.F = c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]
        self.H, self.KV = c["num_attention_heads"], c["num_key_value_heads"]
        self.HD = c.get("head_dim") or self.D // self.H
        self.V, self.eps = c["vocab_size"], float(c["rms_norm_eps"])
        rp = c.get("rope_parameters") or {}
        if (rp.get("rope_type") or "default") != "default" or c.get("rope_scaling"):
            raise NotImplementedError(f"block drafter: rope {rp} {c.get('rope_scaling')}")
        self.theta = float(rp.get("rope_theta") or c.get("rope_theta"))
        self.bs, self.mask = int(g("block_size")), int(g("mask_token_id"))
        self.taps = tuple(int(x) for x in g("target_layer_ids"))
        self.W = self.SW = int(window)              # ring = whole context: full attention


def q8(w):
    """(out, in) bf16 -> {"q": int8 codes, "s": bf16 scale per output row (absmax / 127)}: about half the drafter's HBM
    (what lets it sit next to the target at 8 slots). Changes only what the drafter proposes; the verify pass decides."""
    w32 = np.asarray(w, np.float32)
    s = np.maximum(np.abs(w32).max(-1, keepdims=True), 1e-12) / 127.0
    import ml_dtypes
    return {"q": np.clip(np.rint(w32 / s), -127, 127).astype(np.int8), "s": s[..., 0].astype(ml_dtypes.bfloat16)}


def lin(x, w):
    """linear() on a bf16 (out, in) matrix, or on q8 codes: the dot in bf16 on the codes, then the row scale"""
    if isinstance(w, dict):
        y = lax.dot_general(x, w["q"].astype(x.dtype), (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=f32)
        return (y * w["s"].astype(f32)).astype(x.dtype)
    return linear(x, w)


def load(d, window, put=jnp.asarray, weights="bf16"):
    import ml_dtypes
    from ...hf.source import open_source
    c = Config(d, window)
    raw, src = {}, open_source(d)
    try:
        for name, (dt, shape, buf) in src.tensors.items():
            if dt != "BF16":
                raise ValueError(f"block drafter: tensor {name} is stored as {dt}")
            raw[name] = np.array(buf).view(ml_dtypes.bfloat16).reshape(shape)
    finally:
        src.close()
    raw = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
    if weights not in ("bf16", "int8"):
        raise ValueError(f"block drafter: drafter weights {weights!r}: bf16 or int8")
    j = lambda k: put(raw.pop(k))
    st = lambda fmt: put(np.stack([raw.pop(fmt.format(i)) for i in range(c.L)]))

    def mat(fmt):               # a stacked projection, int8 on request
        if weights == "bf16":
            return st(fmt)
        qs = [q8(raw.pop(fmt.format(i))) for i in range(c.L)]
        return {"q": put(np.stack([x["q"] for x in qs])), "s": put(np.stack([x["s"] for x in qs]))}
    P = dict(fc=j("fc.weight"), hn=j("hidden_norm.weight"), norm=j("norm.weight"),
             L=dict(wq=mat("layers.{}.self_attn.q_proj.weight"), wk=mat("layers.{}.self_attn.k_proj.weight"),
                    wv=mat("layers.{}.self_attn.v_proj.weight"), wo=mat("layers.{}.self_attn.o_proj.weight"),
                    qn=st("layers.{}.self_attn.q_norm.weight"), kn=st("layers.{}.self_attn.k_norm.weight"),
                    ln1=st("layers.{}.input_layernorm.weight"), ln2=st("layers.{}.post_attention_layernorm.weight"),
                    wg=mat("layers.{}.mlp.gate_proj.weight"), wu=mat("layers.{}.mlp.up_proj.weight"),
                    wd=mat("layers.{}.mlp.down_proj.weight")))
    if raw:
        raise ValueError(f"block drafter: unused tensors {sorted(raw)[:8]}")
    return c, P


def ring_write(P, c, R, rows, taps, p0, n):
    """dflash.ring_write (the context K/V of n[b] positions from p0[b], rows rows[b]) on bf16 or q8 projections"""
    B, T, _ = taps.shape
    h = rms(linear(taps.astype(bf16), P["fc"]), P["hn"], c.eps)
    pos = p0[:, None] + jnp.arange(T, dtype=i32)[None]
    keep = (jnp.arange(T)[None] < n[:, None]) & (jnp.arange(T)[None] >= (n[:, None] - c.W))
    slot = jnp.where(keep, pos % c.W, c.W)
    cos, sin = rope_tables(c, pos, bf16)
    ks, vs = [], []
    for l in range(c.L):
        w = jax.tree.map(lambda v: v[l], P["L"])
        ks.append(rope(rms(lin(h, w["wk"]).reshape(B, T, c.KV, c.HD), w["kn"], c.eps), cos, sin))
        vs.append(lin(h, w["wv"]).reshape(B, T, c.KV, c.HD))
    K, V = jnp.stack(ks, 2), jnp.stack(vs, 2)
    r_ = rows[:, None]
    return dict(R, k=R["k"].at[r_, :, slot].set(K, mode="drop"), v=R["v"].at[r_, :, slot].set(V, mode="drop"),
                pos=R["pos"].at[r_, slot].set(pos, mode="drop"))


def block_hidden(P, c, R, emb, start):
    """emb (B, bs, D) = embed rows of the block, start (B,) the anchor's position. Block queries see ring keys at
    positions 0 <= p < start, and block keys j <= their own slot (causal)."""
    B, T, D = emb.shape
    H, KV, HD, G = c.H, c.KV, c.HD, c.H // c.KV
    qpos = start[:, None] + jnp.arange(T, dtype=i32)[None]
    dt = emb.dtype
    cos, sin = rope_tables(c, qpos, dt)
    cp = R["pos"]
    mctx = (cp[:, None, :] >= 0) & (cp[:, None, :] < start[:, None, None])
    causal = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    mask = jnp.concatenate([jnp.broadcast_to(mctx, (B, T, cp.shape[1])), jnp.broadcast_to(causal, (B, T, T))], -1)
    x = emb
    for l in range(c.L):
        w = jax.tree.map(lambda v: v[l], P["L"])
        h = rms(x, w["ln1"], c.eps)
        q = rope(rms(lin(h, w["wq"]).reshape(B, T, H, HD), w["qn"], c.eps), cos, sin)
        k = rope(rms(lin(h, w["wk"]).reshape(B, T, KV, HD), w["kn"], c.eps), cos, sin)
        v = lin(h, w["wv"]).reshape(B, T, KV, HD)
        kk = jnp.concatenate([R["k"][:, l], k], 1)
        vv = jnp.concatenate([R["v"][:, l], v], 1)
        s = jnp.einsum("btkgd,bskd->bkgts", q.reshape(B, T, KV, G, HD), kk, preferred_element_type=f32) * (HD ** -0.5)
        p = jax.nn.softmax(jnp.where(mask[:, None, None], s, NEG), -1)
        o = jnp.einsum("bkgts,bskd->btkgd", p, vv.astype(f32), preferred_element_type=f32).astype(dt).reshape(B, T, H * HD)
        x = x + lin(o, w["wo"])
        h = rms(x, w["ln2"], c.eps)
        x = x + lin(jax.nn.silu(lin(h, w["wg"]).astype(f32)).astype(dt) * lin(h, w["wu"]), w["wd"])
    return rms(x, P["norm"], c.eps)


def new_state(c, S):
    return dict(k=jnp.zeros((S, c.L, c.W, c.KV, c.HD), bf16), v=jnp.zeros((S, c.L, c.W, c.KV, c.HD), bf16),
                pos=jnp.full((S, c.W), -1, i32), tk=jnp.zeros((S, c.W), i32),
                pt=jnp.zeros((S, c.bs, len(c.taps) * c.D), bf16), pp=jnp.zeros((S,), i32), pn=jnp.zeros((S,), i32))


def _tok_write(tk, rows, toks, p0, n, W):
    """committed tokens toks (B, T) at positions p0[b] + j, j < n[b], into the token ring rows"""
    T = toks.shape[1]
    pos = p0[:, None] + jnp.arange(T, dtype=i32)[None]
    slot = jnp.where(jnp.arange(T)[None] < n[:, None], pos % W, W)
    return tk.at[rows[:, None], slot].set(toks, mode="drop")


# Working memory the speculation programs add on top of the plain engine's (verify at width k + 1, the drafter's block
# pass and its prefill hand-off), per chip. Measured on a TPU v6e-1 (a 27B hybrid target in BQ8_R, int8 drafter,
# k 15, 1 slot x 32,768, 8-bit K/V): warm-up peak 33.333 GB with the drafter less the plain engine's 29.713 GB at the same
# setting, less the drafter's weights and ring (2.534 GB): 1.086 GB; planned as 1.1 GB.
SPEC_WORK_BYTES = 1.1e9


def plan_bytes(d, weights="bf16"):
    """What the drafter in folder d costs the context plan, from shapes only (nothing loaded, nothing allocated):
    -> dict(fixed = its weights as they will sit on the chip + SPEC_WORK_BYTES, state(slots, ctx) = its per-slot state
    for a ring of ctx positions)."""
    import math
    from ...hf.source import open_source
    c = Config(d, 2048)
    src = open_source(d)
    try:
        shapes = {(k[len("model."):] if k.startswith("model.") else k): tuple(v[1]) for k, v in src.tensors.items()}
    finally:
        src.close()
    q = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    wb = 0
    for k, sh in shapes.items():
        n = math.prod(sh)
        wb += n + 2 * sh[0] if (weights == "int8" and k.startswith("layers.") and any(x in k for x in q)) else 2 * n

    def state(slots, ctx):
        import jax
        import numpy as np
        cc = Config.__new__(Config)
        cc.__dict__.update(c.__dict__, W=int(ctx), SW=int(ctx))
        st = jax.eval_shape(lambda: new_state(cc, slots))
        return sum(math.prod(x.shape) * np.dtype(x.dtype).itemsize for x in jax.tree.leaves(st))
    return dict(fixed=int(wb + SPEC_WORK_BYTES), state=state, method=METHOD)


class Block(Drafter):
    kind = "trained"
    extra_groups = 0
    needs_hidden = True
    hidden_key = "taps"
    plain_ok = False
    backoff = False

    def __init__(self, model, cfg, d, k_max=None, weights_sha256=None, window=2048, seed=9504, P=None, dcfg=None,
                 weights="bf16"):
        if dcfg is None:
            dcfg, P = load(d, window, weights=weights)
        self.wfmt = weights
        c = dcfg
        if c.D != cfg.D or c.V != cfg.V or max(c.taps) >= cfg.L:
            raise ValueError(f"block drafter: drafter hidden {c.D} / vocab {c.V} / taps {c.taps} do not fit the target")
        import sys
        mod = sys.modules.get(getattr(getattr(model, "make_forward", None), "__module__", None) or "")
        model = mod if (mod is not None and hasattr(mod, "embed") and hasattr(mod, "linear")) else model
        self.model, self.cfg, self.dc, self.W = model, cfg, c, P
        self.k_max = int(c.bs - 1 if k_max is None else k_max)
        if not 1 <= self.k_max <= c.bs - 1:
            raise ValueError(f"block drafter: k {self.k_max} outside 1..{c.bs - 1} (block {c.bs})")
        self.verify_kw = {"taps": c.taps}
        self.sha, self.seed = weights_sha256, int(seed)

    def key(self):
        from . import key_base
        c = self.dc
        return key_base(self.kind, IMPL, self.sha, dict(method=METHOD, block=c.bs, taps=list(c.taps), window=c.W,
                                                        seed=self.seed, weights=self.wfmt), self.k_max)

    def state(self, eng):
        if eng.CTX > self.dc.W:
            raise ValueError(f"block drafter: context {eng.CTX} > the drafter's ring {self.dc.W} (full attention needs W >= CTX)")
        return eng._committed(new_state(self.dc, eng.S))

    def programs(self, eng):
        return [("block_swap", swap_rows, (np.int32(0), np.int32(0)), {}), ("block_forget", forget_row, (np.int32(0),), {})]

    def on_swap(self, eng, a, b):
        eng.dstate = swap_rows(eng.dstate, np.int32(a), np.int32(b))

    def on_restore(self, eng, s):
        eng.dstate = forget_row(eng.dstate, np.int32(s))

    def carry_in(self, D, X, B):
        return jax.tree.map(lambda a: a[:B], D)

    def carry_out(self, D, Dc, B):
        return jax.tree.map(lambda a, b: a.at[:B].set(b), D, Dc)

    def draft(self, W, DW, ck, Dc, tok, pos, Tc, K, sp, st):
        c, B = self.dc, tok.shape[0]
        rows = jnp.arange(B, dtype=i32)
        Dc = ring_write(DW, c, Dc, rows, Dc["pt"], Dc["pp"], Dc["pn"])
        # the anchor at its own position: neither the prefill hand-off (positions 1 .. len - 1 of the prompt) nor a
        # commit's emit (pos + 1 ..) holds the token the next round starts from
        Dc = dict(Dc, tk=_tok_write(Dc["tk"], rows, tok[:, None], pos, jnp.ones((B,), i32), c.W))
        key = jax.random.PRNGKey(self.seed)
        keys = jax.vmap(lambda p: jax.random.fold_in(key, p))(pos)
        idx = jax.vmap(lambda k_, p: jax.random.randint(k_, (c.bs - 1,), 0, p + 1))(keys, pos)
        noise = jnp.take_along_axis(Dc["tk"], idx % c.W, 1)
        toks = jnp.concatenate([tok[:, None], noise], 1)
        h = block_hidden(DW, c, Dc, self.model.embed(W["embed"], toks), pos)
        lg = self.model.linear(h[:, 1:K + 1], W["head"])
        return jnp.argmax(lg, -1).astype(i32), None, ck, Dc

    def after_commit(self, Dc, r, emit, n, go, tok, pos, K1):
        t = r["taps"].astype(bf16)
        t = jnp.concatenate([t, jnp.zeros((t.shape[0], self.dc.bs - K1, t.shape[2]), bf16)], 1) if K1 < self.dc.bs else t
        B = emit.shape[0]
        tk = _tok_write(Dc["tk"], jnp.arange(B, dtype=i32), emit, pos + 1, jnp.where(go, n, 0), self.dc.W)
        return dict(Dc, tk=tk, pt=jnp.where(go[:, None, None], t, Dc["pt"]), pp=jnp.where(go, pos, Dc["pp"]),
                    pn=jnp.where(go, n, 0))

    def on_prefill(self):
        c = self.dc

        def block_pf(W, ck, cv, DW, D, tn, taps, p0, lm, sl, hrow, hidx, Tc):
            """ring + token ring over a prefill chunk; tn[b, j] is the prompt token at position p0 + 1 + j (j < lm)"""
            S = D["pos"].shape[0]
            n = jnp.where(hrow < S, hidx + 1, lm)
            fresh = jnp.where(p0 == 0, sl, S)
            D = dict(D, pos=D["pos"].at[fresh].set(-1, mode="drop"), pn=D["pn"].at[sl].set(0, mode="drop"))
            D = dict(D, tk=_tok_write(D["tk"], sl, tn, p0 + 1, lm, c.W))
            return ring_write(DW, c, D, sl, taps, p0, n), ck, cv
        return block_pf
