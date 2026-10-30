"""GGUF architectures `llama`, `qwen2` and `qwen3` (dense attention + SwiGLU FFN) in plain JAX.

One module for the three, following llama.cpp's graphs (src/models/llama.cpp, qwen2.cpp, qwen3.cpp) and reading the
file's tensors as stored:

  * llama: rotary embedding on adjacent pairs (llama.cpp rope type NORM; its converter permutes the q/k rows of the
    Hugging Face weights for that), optional q/k/v/output biases, optional `rope_freqs.weight` (Llama 3.1+ frequency
    factors: each pair's angle is divided by its factor), linear rope scaling.
  * qwen2: rotary embedding on halves (NEOX), q/k/v biases.
  * qwen3: rotary embedding on halves (NEOX), per-head RMSNorm on q and k before the rotation (`attn_q_norm`,
    `attn_k_norm`), no biases.
  * exaone: as llama with the rotation on halves (NEOX); granite: as llama with `attention.scale`, the embedding x
    `embedding_scale`, each branch x `residual_scale` and the logits / `logit_scale` (llama.cpp src/models/granite.cpp).
RMSNorm weights are multiplied as stored. Big linears stay in the file's own blocks (any ggml type) and run through
mashbox.ops: the TPU kernel when the kernel package covers the type, else the exact float32 reference. Nothing is
re-quantized.

The forward has the contract of models/qwen3_8_27b/model.make_forward, so the same Engine runs it. Every layer is a full
attention layer with its own K/V group; there is no linear-attention state, so the per-slot (R, C) pair the engine moves
around is one zero value per slot. `act` is the activation dtype: bfloat16 to serve, float32 for the CPU test against
llama.cpp's CPU logits.

Rope scaling: linear, and YaRN (archs/rope.py: per-pair angle factors and a cos/sin scale, as llama.cpp's rope_yarn).

Not handled (refused at load with a message): rope scaling other than linear and YaRN, a fused qkv tensor, routed experts,
per-layer head counts, key_length != value_length.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ... import ops
from ..reader import GGUFError
from .qwen35 import rms, kv_write, _T
from . import rope as yarn_rope

f32, bf16 = jnp.float32, jnp.bfloat16
# the Models label (docs/gguf.md) of each architecture ARCHS maps to this module; catalog reads it with ast
MODELS = {"llama": "Llama 3.x", "qwen2": "Qwen2.5", "qwen3": "Qwen3", "exaone": "EXAONE 3.5", "granite": "Granite 3.3"}
SPEC_VERIFY = True       # forward(verify=True) and the identity commit (runtime/spec_step.py)
SPEC_DRAFT = True        # every layer is full attention: a file of this module can be a draft model (--draft-model)
NEG = -1e30
NEOX = ("qwen2", "qwen3", "exaone")      # rope on halves; llama, granite: adjacent pairs
# a file whose block types differ by layer (Q4_K_M, Unsloth's dynamic quants) runs its layers in one lax.scan with one
# weight rebuild per type of each role (ops.Mixed), not unrolled: unrolled, every program held one rebuild per layer, so
# it grew with the layer count, which filled host RAM and the compile cache during the serve warm-up
# (tests/cpu/test_gguf_mixed_scan.py checks that the stacked program does not grow with the layer count). False keeps
# such files unrolled (the tests run both).
MIXED_SCAN = True


class Config:
    """Shapes from the GGUF metadata (keys `<arch>.*`) and the tensor table."""

    def __init__(self, f, act=bf16, layers=None):
        A = self.arch = f.arch
        k = lambda s, d=None: f.get(f"{A}.{s}", d)
        no = lambda why: GGUFError(f"mashbox: {A} file with {why} is not supported in this version")
        self.L_all = self.L = int(k("block_count"))
        self.nextn = 0
        self.D, self.F = int(k("embedding_length")), int(k("feed_forward_length"))
        hc, kc = k("attention.head_count"), k("attention.head_count_kv", k("attention.head_count"))
        if not np.isscalar(hc) or not np.isscalar(kc):
            raise no("per-layer head counts")
        self.H, self.KV = int(hc), int(kc)
        self.HD = int(k("attention.key_length", self.D // self.H))
        if int(k("attention.value_length", self.HD)) != self.HD:
            raise no("key_length != value_length")
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.RD = int(k("rope.dimension_count", self.HD))
        self.theta = float(k("rope.freq_base", 10000.0))
        sc, self.mscale = k("rope.scaling.type"), 1.0
        if sc in (None, "none"):
            self.fscale = 1.0
        elif sc == "linear":
            self.fscale = 1.0 / float(k("rope.scaling.factor", 1.0) or 1.0)
        elif sc == "yarn":
            self.fscale, self.mscale = yarn_rope.yarn(k, self.RD, self.theta, int(k("context_length", 0) or 0))
        else:
            raise no(f"rope scaling '{sc}'")
        self.scale = float(k("attention.scale", 0.0) or 0.0) or self.HD ** -0.5
        self.emb_scale, self.res_scale = float(k("embedding_scale", 0.0) or 1.0), float(k("residual_scale", 0.0) or 1.0)
        self.logit_scale = float(k("logit_scale", 0.0) or 1.0)          # granite
        self.neox = A in NEOX
        self.qknorm = A == "qwen3"
        self.freqs = "rope_freqs.weight" in f.tensors
        names = list(f.tensors)
        if any(".attn_qkv." in n for n in names):
            raise no("a fused qkv tensor")
        if any("_exps." in n or ".ffn_gate_inp." in n for n in names):
            raise no("routed experts")
        if self.qknorm and "blk.0.attn_q_norm.weight" not in f.tensors:
            raise no("no q/k norms")
        if layers is not None:                      # tests only
            assert 0 < layers <= self.L
            self.L = layers
        # the engine's state layout: one K/V group per layer (NG), no linear-attention layers (FI = 1)
        self.NG, self.FI = self.L, 1
        self.HV = self.DK = self.DV = self.CONV = 1
        self.CK = 2
        self.types = ["full_attention"] * self.L
        self.V = int(f.tensors["token_embd.weight"].shape[0])
        self.tie = "output.weight" not in f.tensors
        self.ctx_train = int(k("context_length", 0) or 0)
        self.act = act


# ---------------------------------------------------------------------------------------------------------------- load
BIG = dict(wq="attn_q.weight", wk="attn_k.weight", wv="attn_v.weight", wo="attn_output.weight", wg="ffn_gate.weight",
           wu="ffn_up.weight", wd="ffn_down.weight")
SMALL = dict(ln1="attn_norm.weight", ln2="ffn_norm.weight")
OPTIONAL = dict(bq="attn_q.bias", bk="attn_k.bias", bv="attn_v.bias", bo="attn_output.bias", qn="attn_q_norm.weight",
                kn="attn_k_norm.weight")
MULTIPLIED = tuple(BIG.values())


def load(f, c, use_kernels=True, log=None):
    """-> (W, info). Big linears: ops handles from the file's blocks, stacked over layers when every layer has the same
    types and shapes (one lax.scan runs them). When only the block types differ by layer (MIXED_SCAN), a role whose types
    differ becomes an ops.Mixed (one stack per type; the scan body picks the layer's through lax.switch) and one lax.scan
    still runs the layers; info["mixed_roles"] names those roles. Otherwise a list of layers (unrolled). Small tensors:
    exact float32."""
    import time
    t0, stats = time.time(), dict(bytes=0, values=0, kernel_bytes=0, ref_bytes=0, ref_types={})

    def rows(names, stacked=False):
        ts = [f.tensors[n] for n in names]
        for t in ts:
            stats["bytes"] += t.nbytes
            stats["values"] += int(np.prod(t.shape))
        return ts[0], (np.stack([np.asarray(t.rows()) for t in ts]) if stacked else ts[0].rows())

    def big(*names, stacked=False):
        t, raw = rows(names, stacked)
        if t.type not in ops.reference.TYPES:
            raise GGUFError(f"mashbox: {names[0]} is ggml type {t.type}, which the GGUF reader in this version cannot read")
        h = ops.prepare(raw, t.type, t.shape, use_kernel=use_kernels)
        if isinstance(h, ops.Packed):
            stats["kernel_bytes"] += sum(int(a.nbytes) for a in jax.tree_util.tree_leaves(h.packed))
        else:
            nb = t.nbytes * len(names)
            stats["ref_bytes"] += nb
            stats["ref_types"][t.type_name] = stats["ref_types"].get(t.type_name, 0) + nb
        return h

    def small(*names, stacked=False):
        t, raw = rows(names, stacked)
        return ops.reference.dequant(jnp.asarray(raw), t.type, t.shape[-1])

    def need(name):
        if name not in f.tensors:
            raise GGUFError(f"mashbox: {c.arch} file is missing tensor {name}")
        return name

    def plan(l):
        b = f"blk.{l}."
        p = {r: _T("big", [need(b + n)]) for r, n in BIG.items()}
        p.update({r: _T("small", [need(b + n)]) for r, n in SMALL.items()})
        p.update({r: _T("small", [b + n]) for r, n in OPTIONAL.items() if b + n in f.tensors})
        return p

    def make(t):
        return big(t.names[0]) if t.kind == "big" else small(t.names[0])

    def stack(*ts):
        names = [t.names[0] for t in ts]
        return big(*names, stacked=True) if ts[0].kind == "big" else small(*names, stacked=True)

    def mixed(*ts):         # one role whose block types differ by layer: one stack per type (ops.Mixed)
        names = [t.names[0] for t in ts]
        types = [f.tensors[n].type for n in names]
        if len(set(types)) == 1:
            return stack(*ts)
        if ts[0].kind == "small":                   # small tensors are float32 here: stack the per-layer values
            return jnp.stack([small(n) for n in names])
        order = sorted(set(types), key=types.index)
        which = [order.index(t) for t in types]
        sub = [which[:l].count(which[l]) for l in range(len(which))]
        stacks = [big(*[n for n, t in zip(names, types) if t == u], stacked=True) for u in order]
        mixed_roles[names[0].split(".", 2)[2]] = [f.tensors[n].type_name for n in names]
        return ops.Mixed(stacks, jnp.asarray(which, jnp.int32), jnp.asarray(sub, jnp.int32), order,
                         f.tensors[names[0]].shape)

    is_t = lambda x: isinstance(x, _T)
    plans = [plan(l) for l in range(c.L)]
    sig = lambda p, types=True: (jax.tree_util.tree_structure(p, is_leaf=is_t),
                                 [(t.kind, [((f.tensors[n].type,) if types else ()) + (tuple(f.tensors[n].shape),)
                                            for n in t.names]) for t in jax.tree_util.tree_leaves(p, is_leaf=is_t)])
    stacked = all(sig(p) == sig(plans[0]) for p in plans)
    mixed_roles = {}
    if stacked:
        G = jax.tree_util.tree_map(stack, *plans, is_leaf=is_t)
    elif MIXED_SCAN and all(sig(p, False) == sig(plans[0], False) for p in plans):
        G = jax.tree_util.tree_map(mixed, *plans, is_leaf=is_t)
        stacked = True
    else:
        G = [jax.tree_util.tree_map(make, p, is_leaf=is_t) for p in plans]
    W = dict(g=G)                                   # a list of layers = unrolled, a dict = stacked for lax.scan
    t = f.tensors["token_embd.weight"]
    W["embed"] = ops.prepare(t.rows(), t.type, t.shape, use_kernel=False)       # rows are gathered, not multiplied
    stats["bytes"] += t.nbytes
    stats["values"] += int(np.prod(t.shape))
    W["norm"] = small("output_norm.weight")
    W["head"] = big("output.weight") if not c.tie else ops.prepare(t.rows(), t.type, t.shape, use_kernel=use_kernels)
    if c.freqs:
        W["freqs"] = small("rope_freqs.weight")
    jax.block_until_ready(jax.tree_util.tree_leaves(W))
    info = dict(load_s=round(time.time() - t0, 2), device_bytes=int(stats["bytes"]), n_weights=int(stats["values"]),
                bpw=round(stats["bytes"] * 8 / max(1, stats["values"]), 3), file=f.path.rsplit("/", 1)[-1],
                layers_stacked=stacked, kernel_bytes=stats["kernel_bytes"], reference_bytes=stats["ref_bytes"],
                reference_types=stats["ref_types"], mtp_layers=0, mixed_roles=sorted(mixed_roles))
    if log:
        log(info)
    return W, info


def tensor_roles(f, c):
    """-> (loaded, multiplied): the names `load` puts on the device and the subset multiplied through mashbox.ops"""
    loaded = {n for n in f.tensors if n.startswith("blk.") and int(n.split(".")[1]) < c.L}
    loaded |= {"token_embd.weight", "output_norm.weight"} | ({"output.weight"} if not c.tie else set())
    loaded |= {"rope_freqs.weight"} if c.freqs else set()
    mult = {n for n in loaded if n.startswith("blk.") and n.split(".", 2)[2] in MULTIPLIED}
    mult |= {"token_embd.weight" if c.tie else "output.weight"}
    return loaded, mult


def mtp_tensors(f, c):
    return {}


# ---------------------------------------------------------------------------------------------------------------- state
def new_state(c, S, CTX, extra_groups=0):
    """Zeroed per-slot state for S slots (+1 scratch slot): K/V per layer, and the engine's (R, C) linear-attention pair as
    one zero value per slot (slot on axis 2, as the engine moves it)."""
    kv = (c.NG + extra_groups, S + 1, CTX, c.KV, c.HD)
    return ((jnp.zeros(kv, c.act), jnp.zeros(kv, c.act)),
            (jnp.zeros((1, 1, S + 1, 1, 1, 1), f32), jnp.zeros((1, 1, S + 1, 1, 1), c.act)))


def rope(x, cos, sin, RD, neox):
    """llama.cpp ggml_rope: NEOX rotates (i, i + RD/2), NORM rotates (2i, 2i + 1); dims past RD pass through. cos, sin
    (B, T, 1, RD/2) float32."""
    xr = x[..., :RD].astype(f32)
    if neox:
        a, b = xr[..., :RD // 2], xr[..., RD // 2:]
        o = jnp.concatenate([a * cos - b * sin, b * cos + a * sin], -1)
    else:
        a, b = xr[..., 0::2], xr[..., 1::2]
        o = jnp.stack([a * cos - b * sin, a * sin + b * cos], -1).reshape(xr.shape)
    return jnp.concatenate([o.astype(x.dtype), x[..., RD:]], -1) if RD < x.shape[-1] else o.astype(x.dtype)


def make_forward(c, decode_kernels=None):
    """-> forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, keep=None, verify=False,
    hidden=False) -> dict(ck, cv, nxt, lg), the contract of models/qwen3_8_27b/model.make_forward. keep has nothing to
    protect here (no linear-attention state). verify (speculative decoding, runtime/spec_step.py): the T tokens of each
    contiguous row at p0..p0+T-1 in one pass -> tgt (B, T) the argmax at every position (of the head's output, as
    one-token decode takes it), lg (B, T, V) f32, rec {} (full attention only: the commit is the identity, since K/V
    written past a row's accepted tokens sit past its position, masked by the next pass and written over)."""
    D, H, KV, HD, RD, act = c.D, c.H, c.KV, c.HD, c.RD, c.act
    G, scale, eps, neox = H // KV, c.scale, c.eps, c.neox
    inv = (1.0 / (np.float32(c.theta) ** (np.arange(0, RD, 2, dtype=np.int64).astype(np.float32) / np.float32(RD)))).astype(np.float32)
    lin_ = ops.qmatmul
    rs = (lambda y: (y.astype(f32) * np.float32(c.res_scale)).astype(act)) if c.res_scale != 1.0 else (lambda y: y)
    ls = (lambda y: y.astype(f32) * np.float32(1.0 / c.logit_scale)) if c.logit_scale != 1.0 else (lambda y: y)

    def forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, keep=None, verify=False,
                hidden=False):
        B, T = toks.shape
        Kc, Vc = ck
        pos = p0[:, None] + jnp.arange(T, dtype=jnp.int32)[None]
        valid = jnp.arange(T)[None] < lens[:, None]
        fq = jnp.asarray(inv) / W["freqs"] if "freqs" in W else jnp.asarray(inv)
        fr = pos.astype(f32)[..., None] * fq * np.float32(c.fscale)
        cos, sin = jnp.cos(fr)[:, :, None], jnp.sin(fr)[:, :, None]
        if c.mscale != 1.0:                         # YaRN's cos/sin scale
            cos, sin = cos * np.float32(c.mscale), sin * np.float32(c.mscale)
        m2 = (jnp.arange(T)[None, :, None] >= jnp.arange(T)[None, None, :]) & valid[:, None, :]
        if Tc:
            m1 = jnp.broadcast_to((jnp.arange(Tc)[None, :] < p0[:, None])[:, None, :], (B, T, Tc))
            mask = jnp.concatenate([m1, m2], -1)
        else:
            mask = m2
        x = ops.take_rows(W["embed"], toks)
        x = (x.astype(f32) * np.float32(c.emb_scale) if c.emb_scale != 1.0 else x).astype(act)

        def attend(l, q, k, v):
            if Tc:
                kk, vv = jnp.concatenate([Kc[l][slots, :Tc], k], 1), jnp.concatenate([Vc[l][slots, :Tc], v], 1)
            else:
                kk, vv = k, v
            s = jnp.einsum("btkgd,bskd->bkgts", q, kk, preferred_element_type=f32) * scale
            p = jax.nn.softmax(jnp.where(mask[:, None, None], s, NEG), -1).astype(act)
            return jnp.einsum("bkgts,bskd->btkgd", p, vv, preferred_element_type=f32).astype(act)

        def proj(h, w, name, bias):
            y = lin_(h, w[name])
            return (y.astype(f32) + w[bias]).astype(act) if bias in w else y.astype(act)

        def layer(x, w, l):
            h = rms(x, w["ln1"], eps)
            q = proj(h, w, "wq", "bq").reshape(B, T, H, HD)
            k = proj(h, w, "wk", "bk").reshape(B, T, KV, HD)
            v = proj(h, w, "wv", "bv").reshape(B, T, KV, HD)
            if "qn" in w:
                q, k = rms(q, w["qn"], eps), rms(k, w["kn"], eps)
            q, k = rope(q, cos, sin, RD, neox), rope(k, cos, sin, RD, neox)
            o = attend(l, q.reshape(B, T, KV, G, HD), k, v).reshape(B, T, H * HD)
            x = x + rs(proj(o, w, "wo", "bo"))
            h = rms(x, w["ln2"], eps)
            a = (jax.nn.silu(lin_(h, w["wg"]).astype(f32)) * lin_(h, w["wu"]).astype(f32)).astype(act)
            return x + rs(lin_(a, w["wd"]).astype(act)), k, v

        if not isinstance(W["g"], list):
            held, wx = ops.scan_split(W["g"])          # kernel planes stay whole: the kernel reads layer l in place

            def body(x, xs):
                x, k, v = layer(x, ops.scan_join(held, xs["w"], xs["l"]), xs["l"])
                return x, dict(k=k, v=v)
            x, ys = lax.scan(body, x, dict(w=wx, l=jnp.arange(c.L, dtype=jnp.int32)))
        else:
            yl = []
            for l in range(c.L):
                x, k, v = layer(x, W["g"][l], l)
                yl.append(dict(k=k, v=v))
            ys = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *yl)
        Kc, Vc = kv_write(Kc, Vc, slots, pos, ys["k"], ys["v"])
        out = dict(ck=(Kc, Vc), cv=cv)
        if hidden:
            out["hid"] = rms(x, W["norm"], eps)
        if verify:
            lv = ls(lin_(rms(x, W["norm"], eps), W["head"]))     # the logits plain decode gives (its logit scale too)
            out.update(tgt=jnp.argmax(lv, -1).astype(jnp.int32), lg=lv.astype(f32), rec={})
            return out
        if all_logits:
            out["lg"] = ls(lin_(rms(x, W["norm"], eps), W["head"])).astype(f32)
            return out
        hl = rms(x[jnp.arange(B), jnp.maximum(lens - 1, 0)], W["norm"], eps)
        lg = ls(lin_(hl, W["head"]))
        out.update(nxt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg.astype(f32))
        return out

    return forward
