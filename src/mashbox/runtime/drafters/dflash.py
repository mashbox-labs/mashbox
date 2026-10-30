"""The DFlash 2 drafter: a trained block drafter that proposes 7 tokens in one pass from the target's own hidden states.

Weights: the drafter folder it is given (config.json + model.safetensors), loaded unchanged; the model package whose
forward declares LAYER_TAPS names the drafter published for it, the file's sha256 and the licence its publisher states. The math is re-implemented in JAX
from the reference code github.com/z-lab/dflash @ 07ebd93db9f472af339b644bb70221ad8428328a dflash/model.py (MIT License,
Copyright (c) 2026 Z Lab; its LICENSE text ships unchanged next to this file as LICENSE-dflash): DFlashDraftModel.forward, its decoder layer and attention classes,
_grouped_dynamic_convolve, GroupedDynamicCausalConv, CandidateSelector.select, dflash_generate. CPU parity of this math vs
that reference on the published target's real hidden states: f32 relative error <= 1.9e-6, the same 7 drafts in bf16.

The drafter has no embedding and no output head: it reads the target's embed rows and lm_head (the target's own formats).

Per row, per round (greedy drafts at every temperature: the coupled draw keeps a sampled row's output exact):
  context   the target's outputs at the layers its config names (target_layer_ids) for every committed position ->
            concat -> fc -> RMSNorm -> each draft layer's k_proj / k_norm / rope and v_proj -> a ring K/V of W = 2048
            positions per layer (the sliding window: a block query at q sees context keys p with q - p < 2048). The ring
            is written from prefill (every prompt position) and after each commit (the n committed positions' taps from
            the verify pass); rejected positions never enter it.
  block     [tok, MASK x 7] at positions pos .. pos + 7, 5 layers (non-causal inside the block), the target's head on
            slots 1..7, top 16 per slot, the candidate selector's greedy walk from tok -> 7 drafts.
The target verifies [tok, d1..d7] (width 8) in the engine's step (runtime/spec_step.py) and commits the accepted prefix.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import Drafter

f32, bf16, i32 = jnp.float32, jnp.bfloat16, jnp.int32
NEG = -1e30
METHOD = "dflash"
IMPL = "dflash2@1"          # this file's math; part of the gate key


class Config:
    """The drafter's config.json. Anything this file does not implement is refused with the reason."""

    def __init__(self, d):
        c = json.load(open(os.path.join(d, "config.json")))
        dc = c.get("dflash_config", {})
        g = lambda k, default=None: dc.get(k, c.get(k, default))
        for k in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
            if g(k) is None:
                raise NotImplementedError(f"DFlash: config has no {k} (a DFlash v1 drafter; this file runs DFlash 2)")
        self.L, self.D, self.F = c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]
        self.H, self.KV = c["num_attention_heads"], c["num_key_value_heads"]
        self.HD = c.get("head_dim") or self.D // self.H
        self.V, self.eps = c["vocab_size"], float(c["rms_norm_eps"])
        rp = c.get("rope_parameters") or {}
        if rp.get("rope_type", c.get("rope_scaling") and c["rope_scaling"].get("rope_type") or "default") != "default":
            raise NotImplementedError(f"DFlash: rope {rp}")
        self.theta = float(rp.get("rope_theta") or c.get("rope_theta"))
        self.bs, self.mask = int(g("block_size")), int(g("mask_token_id"))
        self.taps = tuple(int(x) for x in g("target_layer_ids"))
        self.ck, self.cg = int(g("conv_kernel_size")), int(g("conv_group_size"))
        self.sr, self.topk = int(g("selector_rank")), int(g("selector_top_k"))
        lt = c.get("layer_types") or ["full_attention"] * self.L
        if c.get("is_causal", None) is not False or set(lt) != {"sliding_attention"}:
            raise NotImplementedError("DFlash: only is_causal false with every layer sliding_attention")
        self.SW = self.W = int(c["sliding_window"])
        if (float(g("input_embedding_scale", 1.0)) != 1.0 or float(g("output_multiplier", 1.0)) != 1.0
                or g("final_logit_softcapping")):
            raise NotImplementedError("DFlash: input_embedding_scale / output_multiplier / softcap")


def load(d, put=jnp.asarray):
    """-> (cfg, P): every tensor as stored (bf16), stacked per layer. Read with the package's own safetensors reader
    (mashbox.hf.source: the standard library and NumPy, no safetensors library, which only the dev extra installs).
    Refuses a dtype other than BF16 and unused tensors."""
    import ml_dtypes
    from ...hf.source import open_source
    c = Config(d)
    raw, src = {}, open_source(d)
    try:
        for name, (dt, shape, buf) in src.tensors.items():
            if dt != "BF16":
                raise ValueError(f"DFlash: tensor {name} is stored as {dt}; this file reads the published BF16 weights")
            raw[name] = np.array(buf).view(ml_dtypes.bfloat16).reshape(shape)      # copied out of the file map
    finally:
        src.close()
    j = lambda k: put(raw.pop(k))
    st = lambda fmt: put(np.stack([raw.pop(fmt.format(i)) for i in range(c.L)]))
    P = dict(fc=j("fc.weight"), hn=j("hidden_norm.weight"), norm=j("norm.weight"),
             sel=dict(hp=j("candidate_selector.hidden_projection.weight"), pre=j("candidate_selector.predecessor_codebook"),
                      suc=j("candidate_selector.successor_codebook")),
             L=dict(wq=st("layers.{}.self_attn.q_proj.weight"), wk=st("layers.{}.self_attn.k_proj.weight"),
                    wv=st("layers.{}.self_attn.v_proj.weight"), wo=st("layers.{}.self_attn.o_proj.weight"),
                    qn=st("layers.{}.self_attn.q_norm.weight"), kn=st("layers.{}.self_attn.k_norm.weight"),
                    ln1=st("layers.{}.input_layernorm.weight"), ln2=st("layers.{}.post_attention_layernorm.weight"),
                    wg=st("layers.{}.mlp.gate_proj.weight"), wu=st("layers.{}.mlp.up_proj.weight"),
                    wd=st("layers.{}.mlp.down_proj.weight"),
                    acb=st("layers.{}.attention_conv.base_kernel"), acp=st("layers.{}.attention_conv.kernel_projection.weight"),
                    mcb=st("layers.{}.mlp_conv.base_kernel"), mcp=st("layers.{}.mlp_conv.kernel_projection.weight")))
    if raw:
        raise ValueError(f"DFlash: unused tensors {sorted(raw)[:8]}")
    return c, P


# ------------------------------------------------------------------------------------------------------------- ops
def linear(x, w):
    """nn.Linear without bias: (.., in) x (out, in), f32 accumulation, one rounding to x's dtype"""
    return lax.dot_general(x, w, (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=f32).astype(x.dtype)


def rms(x, w, eps):
    """the reference's RMSNorm: normalise in f32, round, times the weight"""
    x32 = x.astype(f32)
    return w * (x32 * lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype)


def rope_tables(c, pos, dt=bf16):
    inv = 1.0 / (np.float32(c.theta) ** (np.arange(0, c.HD, 2, dtype=np.int64).astype(np.float32) / np.float32(c.HD)))
    fr = pos.astype(f32)[..., None] * jnp.asarray(inv.astype(np.float32))
    emb = jnp.concatenate([fr, fr], -1)
    return jnp.cos(emb).astype(dt), jnp.sin(emb).astype(dt)


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = jnp.concatenate([-x[..., h:], x[..., :h]], -1)
    return x * cos[:, :, None] + rot * sin[:, :, None]


def gconv(x, dyn, base, gs):
    """two-tap grouped dynamic causal conv over the block: out[t] = sum_o base[o] * x[t - o] + dyn[t, o] * x[t - o]"""
    B, T, D = x.shape
    G = D // gs
    blocks = x.reshape(B, T, G, gs)
    out = jnp.zeros_like(blocks)
    for o in range(base.shape[0]):
        v = blocks if o == 0 else jnp.pad(blocks[:, :-o], ((0, 0), (o, 0), (0, 0), (0, 0)))
        out = out + base[o].reshape(1, 1, G, gs).astype(x.dtype) * v
        out = (out.astype(f32) + dyn[:, :, o, :, None].astype(f32) * v.astype(f32)).astype(x.dtype)
    return out.reshape(B, T, D)


def conv_prepare(x, base, proj, c):
    dyn = linear(x, proj).reshape(x.shape[:2] + (2, c.ck, c.D // c.cg))
    return gconv(x, dyn[:, :, 0], base[0], c.cg), dyn[:, :, 1]


# ------------------------------------------------------------------------------------------------------------- ring
def new_ring(c, S, dt=bf16):
    """per slot: K/V (S, L, W, KV, HD) (K after k_norm and rope) and the position each ring entry holds (S, W), -1 = empty"""
    return dict(k=jnp.zeros((S, c.L, c.W, c.KV, c.HD), dt), v=jnp.zeros((S, c.L, c.W, c.KV, c.HD), dt),
                pos=jnp.full((S, c.W), -1, i32))


def ring_write(P, c, R, rows, taps, p0, n):
    """Write the context K/V of n[b] positions p0[b] .. p0[b] + n[b] - 1 of ring row rows[b] (taps (B, T, len(taps) * D);
    rows past n[b] and rows out of range are dropped). Only the last W positions of a write are kept (each ring entry is
    written at most once per call)."""
    B, T, _ = taps.shape
    h = rms(linear(taps.astype(bf16), P["fc"]), P["hn"], c.eps)
    pos = p0[:, None] + jnp.arange(T, dtype=i32)[None]
    keep = (jnp.arange(T)[None] < n[:, None]) & (jnp.arange(T)[None] >= (n[:, None] - c.W))
    slot = jnp.where(keep, pos % c.W, c.W)                  # c.W = out of range: dropped
    cos, sin = rope_tables(c, pos, bf16)
    ks, vs = [], []
    for l in range(c.L):
        k = linear(h, P["L"]["wk"][l]).reshape(B, T, c.KV, c.HD)
        ks.append(rope(rms(k, P["L"]["kn"][l], c.eps), cos, sin))
        vs.append(linear(h, P["L"]["wv"][l]).reshape(B, T, c.KV, c.HD))
    K, V = jnp.stack(ks, 2), jnp.stack(vs, 2)              # (B, T, L, KV, HD)
    r_ = rows[:, None]
    return dict(R, k=R["k"].at[r_, :, slot].set(K, mode="drop"), v=R["v"].at[r_, :, slot].set(V, mode="drop"),
                pos=R["pos"].at[r_, slot].set(pos, mode="drop"))


def _swap_rows(D, a, b):
    """exchange slots a and b of the drafter's state (every leaf has the slot axis first)"""
    def sw(x):
        xa, xb = lax.dynamic_index_in_dim(x, a, 0), lax.dynamic_index_in_dim(x, b, 0)
        return lax.dynamic_update_index_in_dim(lax.dynamic_update_index_in_dim(x, xb, a, 0), xa, b, 0)
    return jax.tree.map(sw, D)


def _forget_row(D, s):
    """slot s holds no ring entries and no taps waiting for the ring (its K/V rows stay; pos -1 masks them)"""
    return dict(D, pos=D["pos"].at[s].set(-1), pn=D["pn"].at[s].set(0))


swap_rows = jax.jit(_swap_rows, donate_argnums=(0,))
forget_row = jax.jit(_forget_row, donate_argnums=(0,))


def block_hidden(P, c, R, emb, start):
    """emb (B, bs, D) bf16 = embed rows of [anchor, MASK x (bs - 1)], start (B,) the anchor's position, R the rows' ring
    (leading axis B). Block queries see ring keys at positions p < start with q - p < SW, and the whole block."""
    B, T, D = emb.shape
    H, KV, HD, G = c.H, c.KV, c.HD, c.H // c.KV
    qpos = start[:, None] + jnp.arange(T, dtype=i32)[None]
    dt = emb.dtype
    cos, sin = rope_tables(c, qpos, dt)
    cp = R["pos"]
    mctx = (cp[:, None, :] >= 0) & (cp[:, None, :] < start[:, None, None]) & (qpos[:, :, None] - cp[:, None, :] < c.SW)
    mask = jnp.concatenate([mctx, jnp.ones((B, T, T), bool)], -1)
    x = emb
    for l in range(c.L):
        w = {k: v[l] for k, v in P["L"].items()}
        r = x
        h = rms(x, w["ln1"], c.eps)
        h, ak = conv_prepare(h, w["acb"], w["acp"], c)
        q = rope(rms(linear(h, w["wq"]).reshape(B, T, H, HD), w["qn"], c.eps), cos, sin)
        k = rope(rms(linear(h, w["wk"]).reshape(B, T, KV, HD), w["kn"], c.eps), cos, sin)
        v = linear(h, w["wv"]).reshape(B, T, KV, HD)
        kk = jnp.concatenate([R["k"][:, l], k], 1)
        vv = jnp.concatenate([R["v"][:, l], v], 1)
        s = jnp.einsum("btkgd,bskd->bkgts", q.reshape(B, T, KV, G, HD), kk, preferred_element_type=f32) * (HD ** -0.5)
        p = jax.nn.softmax(jnp.where(mask[:, None, None], s, NEG), -1)
        o = jnp.einsum("bkgts,bskd->btkgd", p, vv.astype(f32), preferred_element_type=f32).astype(dt).reshape(B, T, H * HD)
        o = gconv(linear(o, w["wo"]), ak, w["acb"][1], c.cg)
        x = r + o
        r = x
        h = rms(x, w["ln2"], c.eps)
        h, mk = conv_prepare(h, w["mcb"], w["mcp"], c)
        a = jax.nn.silu(linear(h, w["wg"]).astype(f32)).astype(dt) * linear(h, w["wu"])
        x = r + gconv(linear(a, w["wd"]), mk, w["mcb"][1], c.cg)
    return rms(x, P["norm"], c.eps)


def propose(P, c, hid, logits, anchor):
    """hid (B, bs-1, D) block slots 1.., logits (B, bs-1, V) the target head on them, anchor (B,) -> drafts (B, bs-1):
    top-k per slot, rescored by the selector's bigram term, one greedy walk from the anchor"""
    unary, cand = lax.top_k(logits, c.topk)
    hp = linear(hid, P["sel"]["hp"])
    prev, path = anchor, []
    for t in range(hid.shape[1]):
        a = P["sel"]["pre"][prev] * hp[:, t]
        e = jnp.einsum("br,bkr->bk", a, P["sel"]["suc"][cand[:, t]], preferred_element_type=f32).astype(hid.dtype)
        i = jnp.argmax(unary[:, t] + e, -1)
        prev = jnp.take_along_axis(cand[:, t], i[:, None], 1)[:, 0]
        path.append(prev)
    return jnp.stack(path, 1).astype(i32)


# ------------------------------------------------------------------------------------------------------------- drafter
class DFlash(Drafter):
    kind = "trained"
    extra_groups = 0            # its K/V is its own ring, not a target K/V group
    needs_hidden = True
    hidden_key = "taps"         # prefill hands on_prefill out["taps"] (the engine asks the forward for verify_kw["taps"])
    plain_ok = False            # every committed position's taps must reach the ring
    backoff = False

    def __init__(self, model, cfg, d, k_max=None, weights_sha256=None, P=None, dcfg=None):
        """model = the target's module (its embed and linear read the target's own weight formats), cfg its Config,
        d = the drafter's folder (config.json + *.safetensors), or P / dcfg given (tests)."""
        if dcfg is None:
            dcfg, P = load(d)
        c = dcfg
        if c.D != cfg.D or c.V != cfg.V or max(c.taps) >= cfg.L:
            raise ValueError(f"DFlash: drafter hidden {c.D} / vocab {c.V} / taps {c.taps} do not fit the target "
                             f"(hidden {cfg.D}, vocab {cfg.V}, {cfg.L} layers)")
        import sys
        # the module that defines the target's forward has its embed and linear (a package may only re-export make_forward)
        mod = sys.modules.get(getattr(getattr(model, "make_forward", None), "__module__", None) or "")
        model = mod if (mod is not None and hasattr(mod, "embed") and hasattr(mod, "linear")) else model
        if not (hasattr(model, "embed") and hasattr(model, "linear")):
            raise ValueError("DFlash: the target module has no embed / linear for the drafter to read its tables with")
        self.model, self.cfg, self.dc, self.W = model, cfg, c, P
        self.k_max = int(c.bs - 1 if k_max is None else k_max)
        if not 1 <= self.k_max <= c.bs - 1:
            raise ValueError(f"DFlash: k {self.k_max} outside 1..{c.bs - 1} (block {c.bs})")
        self.verify_kw = {"taps": c.taps}
        self.sha = weights_sha256

    def key(self):
        from . import key_base
        c = self.dc
        return key_base(self.kind, IMPL, self.sha, dict(method=METHOD, block=c.bs, taps=list(c.taps), window=c.W,
                                                        top_k=c.topk, rank=c.sr), self.k_max)

    # device state per slot: the ring, plus the taps of the positions the last round committed (written into the ring at
    # the start of the next draft, which has the drafter's weights; upstream projects them at the same point)
    def state(self, eng):
        c, S = self.dc, eng.S
        R = new_ring(c, S)
        R.update(pt=jnp.zeros((S, c.bs, len(c.taps) * c.D), bf16), pp=jnp.zeros((S,), i32), pn=jnp.zeros((S,), i32))
        return eng._committed(R)

    # the ring belongs to the conversation in its slot: it moves when the engine moves a slot's state, and a slot restored
    # from host memory (which holds the target's state only) starts with an empty ring, never another conversation's
    def programs(self, eng):
        return [("dflash_swap", swap_rows, (np.int32(0), np.int32(0)), {}), ("dflash_forget", forget_row, (np.int32(0),), {})]

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
        Dc = ring_write(DW, c, Dc, jnp.arange(B, dtype=i32), Dc["pt"], Dc["pp"], Dc["pn"])
        toks = jnp.concatenate([tok[:, None], jnp.full((B, c.bs - 1), c.mask, i32)], 1)
        h = block_hidden(DW, c, Dc, self.model.embed(W["embed"], toks), pos)
        lg = self.model.linear(h[:, 1:K + 1], W["head"])       # the target's head on the first K drafted slots
        return propose(DW, c, h[:, 1:K + 1], lg, tok), None, ck, Dc     # Dc: the ring with the last commit's taps written

    def after_commit(self, Dc, r, emit, n, go, tok, pos, K1):
        """the committed positions' taps wait in the carry for the next draft; the ring itself was written by this round's
        draft (the taps its last round left), so Dc here is the carry draft() returned"""
        t = r["taps"].astype(bf16)
        t = jnp.concatenate([t, jnp.zeros((t.shape[0], self.dc.bs - K1, t.shape[2]), bf16)], 1) if K1 < self.dc.bs else t
        return dict(Dc, pt=jnp.where(go[:, None, None], t, Dc["pt"]), pp=jnp.where(go, pos, Dc["pp"]),
                    pn=jnp.where(go, n, 0))

    def on_prefill(self):
        c = self.dc

        def dflash_pf(W, ck, cv, DW, D, tn, taps, p0, lm, sl, hrow, hidx, Tc):
            """The ring over a prefill chunk: every prompt position of the chunk (p0 .. p0 + lens - 1; the last chunk's
            lens = hidx + 1, the others' = lm), rows sl (S = padding: dropped). A row's first chunk (p0 == 0) clears its
            ring first."""
            S = D["pos"].shape[0]
            n = jnp.where(hrow < S, hidx + 1, lm)
            fresh = jnp.where(p0 == 0, sl, S)
            D = dict(D, pos=D["pos"].at[fresh].set(-1, mode="drop"), pn=D["pn"].at[sl].set(0, mode="drop"))
            return ring_write(DW, c, D, sl, taps, p0, n), ck, cv
        return dflash_pf
