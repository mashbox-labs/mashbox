"""GGUF architecture `qwen35` (Qwen3.5 / Qwen3.6 / Qwen3.8 dense) in plain JAX.

The math is the same model as mashbox/models/qwen3_8_27b (Hugging Face reference), but the tensors are llama.cpp's: its
converter (convert_hf_to_gguf.py, Qwen3_5TextModel) changes some of them on the way into the file, so this file follows
llama.cpp's graph (src/models/qwen35.cpp) and reads the file's values as stored:

  * RMSNorm weights are stored as w + 1 and are multiplied directly, EXCEPT the linear-attention output norm
    (`ssm_norm`, HF linear_attn.norm.weight), which the converter does not shift and which is also used as stored.
  * `ssm_a` holds -exp(A_log) already (F32): the decay is softplus(alpha + ssm_dt.bias) * ssm_a.
  * linear-attention value heads are in tiled order: value head h reads key head h % n_key_heads (HF: h // ratio).
  * conv1d is stored squeezed, (channels, kernel).
Big linears stay in the file's own blocks (any ggml type) and run through mashbox.ops: the TPU kernel when the kernel
package covers the type, else the exact float32 reference. Nothing is re-quantized.

The forward has the same signature and per-slot state as mashbox/models/qwen3_8_27b/model.py, so the same Engine
(scheduler, prefix cache, sampler, server) runs it, including the engine's keep mask (continuous batching) and the
speculative-decoding hooks verify / hidden with the Qwen port's semantics; gated_delta_commit / conv_commit below are the
commit half of a verify pass, in this file's head order. `act` is the activation dtype: bfloat16 to serve, float32 for the
CPU test against llama.cpp's CPU logits.

Not handled (refused at load with a message): rope scaling, a fused full-attention qkv, a Hadamard-folded checkpoint.
The file's MTP layer (`nextn`) is read and reported; the speculative decoder that uses it lives in the engine.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ... import ops
from ...moe import ffn as MF
from ...models.qwen3_8_27b import model as M
from ..reader import GGUFError

f32, bf16 = jnp.float32, jnp.bfloat16
# the Models label (docs/gguf.md) of each architecture ARCHS maps to this module; catalog reads it with ast
MODELS = {"qwen35": "Qwen3.5, Qwen3.6, Qwen3.8 (dense)", "qwen35moe": "Qwen3.5, Qwen3.6 (mixture of experts)"}
SPEC_VERIFY = True       # forward(verify=True) with gated_delta_commit / conv_commit (runtime/spec_step.py)
NEG = -1e30
ARCH = "qwen35"
HI = lax.Precision.HIGHEST


class Config:
    """Shapes from the GGUF metadata (keys `qwen35.*`) and the tensor table."""

    def __init__(self, f, act=bf16, layers=None):
        A = f.arch if f.arch in ("qwen35", "qwen35moe") else ARCH
        k = lambda s, d=None: f.get(f"{A}.{s}", d)
        self.L_all = int(k("block_count"))
        self.nextn = int(k("nextn_predict_layers", 0) or 0)
        self.L = self.L_all - self.nextn
        # qwen35moe files from llama.cpp's converter carry expert_feed_forward_length and
        # expert_shared_feed_forward_length and no feed_forward_length (unsloth/Qwen3.6-35B-A3B-GGUF); FFN shapes are
        # read from the tensors, so F is informational there
        self.D = int(k("embedding_length"))
        self.F = int(k("feed_forward_length", 0) or 0) if f.arch == "qwen35moe" else int(k("feed_forward_length"))
        self.H, self.KV = int(k("attention.head_count")), int(k("attention.head_count_kv"))
        self.HD = int(k("attention.key_length", self.D // self.H))
        if int(k("attention.value_length", self.HD)) != self.HD:
            raise GGUFError("mashbox: qwen35 file with key_length != value_length is not supported")
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.RD = int(k("rope.dimension_count", self.HD))
        self.theta = float(k("rope.freq_base", 10000.0))
        sc = k("rope.scaling.type")
        if sc not in (None, "none"):
            raise GGUFError(f"mashbox: qwen35 file with rope scaling '{sc}' is not supported in this version")
        self.scale = float(k("attention.scale", 0.0) or 0.0) or self.HD ** -0.5
        self.HK, self.HV = int(k("ssm.group_count")), int(k("ssm.time_step_rank"))
        self.DK = int(k("ssm.state_size"))
        self.DV = int(k("ssm.inner_size")) // self.HV
        self.CK = int(k("ssm.conv_kernel"))
        self.CONV = 2 * self.HK * self.DK + self.HV * self.DV
        self.FI = int(k("full_attention_interval", 4))
        recr = k("attention.recurrent_layers")
        want = [(i + 1) % self.FI != 0 for i in range(self.L)]
        if recr is not None and [bool(x) for x in list(recr)[:self.L]] != want:
            raise GGUFError("mashbox: qwen35 layer pattern is not (interval - 1) linear + 1 full attention, repeated")
        if self.L % self.FI:
            raise GGUFError(f"mashbox: qwen35 with {self.L} layers is not a multiple of the attention interval {self.FI}")
        if layers is not None:                      # tests only
            assert layers % self.FI == 0 and 0 < layers <= self.L
            self.L = layers
        self.NG = self.L // self.FI
        self.types = ["linear_attention" if w else "full_attention" for w in want[:self.L]]
        self.V = int(f.tensors["token_embd.weight"].shape[0])
        self.tie = "output.weight" not in f.tensors
        self.ctx_train = int(k("context_length", 0) or 0)
        self.act = act
        if any(n.endswith(".attn_qkv.weight") and int(n.split(".")[1]) % self.FI == self.FI - 1 for n in f.tensors):
            raise GGUFError("mashbox: qwen35 file with a fused full-attention qkv tensor is not supported in this version")
        # qwen35moe (llama.cpp src/models/qwen35moe.cpp): the same layers with a routed-expert FFN (softmax router, top-k,
        # weights renormalised to sum 1, optional expert_weights_scale) plus a shared expert gated by a sigmoid
        self.moe = A == "qwen35moe"
        if self.moe:
            self.E, self.K = int(k("expert_count")), int(k("expert_used_count"))
            self.w_scale = float(k("expert_weights_scale", 0.0) or 0.0)
            if any(n.endswith(".ffn_gate_up_exps.weight") for n in f.tensors):
                raise GGUFError("mashbox: qwen35moe file with fused gate+up expert tensors is not supported in this version")
            if not all(f"blk.{i}.ffn_up_shexp.weight" in f.tensors for i in range(self.L)):
                raise GGUFError("mashbox: qwen35moe file without a shared expert in every layer is not supported in this "
                                "version")
        if any("hadamard" in kk for kk in f.keys()):
            raise GGUFError("mashbox: Hadamard-folded qwen35 files are not supported in this version")


# ---------------------------------------------------------------------------------------------------------------- load
SMALL = ("attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
         "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")


class _T:
    """a weight to load: kind 'big' (one ops handle), 'small' (one float32 tensor) or 'smalls' (float32, stacked)"""

    def __init__(self, kind, names):
        self.kind, self.names = kind, names


def load(f, c, use_kernels=True, log=None):
    """-> (W, info). Big linears: ops handles from the file's blocks (stacked over layer groups when every group has the
    same types, so one lax.scan runs them; else a list of groups, unrolled). Small tensors: exact float32."""
    import time
    t0, stats = time.time(), dict(bytes=0, values=0, kernel_bytes=0, ref_bytes=0, ref_types={})

    def rows(names, stacked=False):
        """the tensors' blocks as stored; several names (one role of every layer group) are stacked on the host, so the
        device gets one array per role and never stacks or copies the weights"""
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
        nb = t.nbytes * len(names)
        if isinstance(h, ops.Packed):
            stats["kernel_bytes"] += sum(int(a.nbytes) for a in jax.tree_util.tree_leaves(h.packed))
        else:
            stats["ref_bytes"] += nb
            stats["ref_types"][t.type_name] = stats["ref_types"].get(t.type_name, 0) + nb
        return h

    def small(*names, stacked=False):
        t, raw = rows(names, stacked)
        return ops.reference.dequant(jnp.asarray(raw), t.type, t.shape[-1])

    def need(name):
        if name not in f.tensors:
            raise GGUFError(f"mashbox: qwen35 file is missing tensor {name}")
        return name

    def plan(g):
        """the group's tree with a _T (what to load) at every leaf; this reads no tensor bytes"""
        B_, S_ = (lambda n: _T("big", [need(n)])), (lambda n: _T("small", [need(n)]))
        lin, mlp, full = [], [], None
        for i in range(c.FI):
            b = f"blk.{g * c.FI + i}."
            if c.moe:
                mlp.append(dict(gi=S_(b + "ffn_gate_inp.weight"), ge=B_(b + "ffn_gate_exps.weight"),
                                ue=B_(b + "ffn_up_exps.weight"), de=B_(b + "ffn_down_exps.weight"),
                                gs=S_(b + "ffn_gate_inp_shexp.weight"), sg=B_(b + "ffn_gate_shexp.weight"),
                                su=B_(b + "ffn_up_shexp.weight"), sd=B_(b + "ffn_down_shexp.weight")))
            else:
                mlp.append(dict(wg=B_(b + "ffn_gate.weight"), wu=B_(b + "ffn_up.weight"), wd=B_(b + "ffn_down.weight")))
            if i < c.FI - 1:
                lin.append(dict(qkv=B_(b + "attn_qkv.weight"), z=B_(b + "attn_gate.weight"), b=B_(b + "ssm_beta.weight"),
                                a=B_(b + "ssm_alpha.weight"), out=B_(b + "ssm_out.weight"), conv=S_(b + "ssm_conv1d.weight"),
                                A=S_(b + "ssm_a"), dt=S_(b + "ssm_dt.bias"), nw=S_(b + "ssm_norm.weight")))
            else:
                full = dict(wq=B_(b + "attn_q.weight"), wk=B_(b + "attn_k.weight"), wv=B_(b + "attn_v.weight"),
                            wo=B_(b + "attn_output.weight"), qn=S_(b + "attn_q_norm.weight"),
                            kn=S_(b + "attn_k_norm.weight"))
        ln = lambda s: _T("smalls", [need(f"blk.{g * c.FI + i}.{s}") for i in range(c.FI)])
        return dict(lin=lin, full=full, mlp=mlp, ln1=ln("attn_norm.weight"), ln2=ln("post_attention_norm.weight"))

    def make(t):
        if t.kind == "big":
            return big(t.names[0])
        if t.kind == "small":
            return small(t.names[0])
        return small(*t.names, stacked=True)         # the group's FI norms, stacked

    def stack(*ts):
        """one role of every layer group, stacked over groups on the host (see rows): a 29 GB file on a 31 GB chip is
        never held twice and the load runs no stacking program on the device"""
        if ts[0].kind == "big":
            return big(*[t.names[0] for t in ts], stacked=True)
        if ts[0].kind == "small":
            return small(*[t.names[0] for t in ts], stacked=True)
        t, raw = rows([n for t in ts for n in t.names], True)
        return ops.reference.dequant(jnp.asarray(raw.reshape((len(ts), len(ts[0].names)) + raw.shape[1:])), t.type,
                                     t.shape[-1])

    is_t = lambda x: isinstance(x, _T)
    plans = [plan(g) for g in range(c.NG)]
    sig = lambda p: (jax.tree_util.tree_structure(p, is_leaf=is_t),
                     [(t.kind, [(f.tensors[n].type, tuple(f.tensors[n].shape)) for n in t.names])
                      for t in jax.tree_util.tree_leaves(p, is_leaf=is_t)])
    stacked = all(sig(p) == sig(plans[0]) for p in plans)
    if stacked:
        G = jax.tree_util.tree_map(stack, *plans, is_leaf=is_t)
    else:
        G = [jax.tree_util.tree_map(make, p, is_leaf=is_t) for p in plans]
    W = dict(g=G)                                   # a list of groups = unrolled, a dict = stacked for lax.scan
    t = f.tensors["token_embd.weight"]
    W["embed"] = ops.prepare(t.rows(), t.type, t.shape, use_kernel=False)       # rows are gathered, not multiplied
    stats["bytes"] += t.nbytes
    stats["values"] += int(np.prod(t.shape))
    W["norm"] = small("output_norm.weight")
    W["head"] = big("output.weight") if not c.tie else ops.prepare(t.rows(), t.type, t.shape, use_kernel=use_kernels)
    jax.block_until_ready(jax.tree_util.tree_leaves(W))
    info = dict(load_s=round(time.time() - t0, 2), device_bytes=int(stats["bytes"]), n_weights=int(stats["values"]),
                bpw=round(stats["bytes"] * 8 / max(1, stats["values"]), 3), file=f.path.rsplit("/", 1)[-1],
                layers_stacked=stacked, kernel_bytes=stats["kernel_bytes"], reference_bytes=stats["ref_bytes"],
                reference_types=stats["ref_types"], mtp_layers=c.nextn)
    if log:
        log(info)
    return W, info


MULTIPLIED = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "attn_qkv.weight", "attn_gate.weight",
              "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
              "attn_output.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")


def tensor_roles(f, c):
    """-> (loaded, multiplied): the names `load` puts on the device (main layers, embeddings, norm, head; not the MTP
    layer) and the subset the forward multiplies through mashbox.ops (big linears + head). For the memory estimate."""
    loaded = {n for n in f.tensors if n.startswith("blk.") and int(n.split(".")[1]) < c.L}
    loaded |= {"token_embd.weight", "output_norm.weight"} | ({"output.weight"} if not c.tie else set())
    mult = {n for n in loaded if n.startswith("blk.") and n.split(".", 2)[2] in MULTIPLIED}
    mult |= {"token_embd.weight" if c.tie else "output.weight"}
    return loaded, mult


def mtp_tensors(f, c):
    """names of the file's MTP (nextn) layer, by role; {} when the file has none"""
    if not c.nextn:
        return {}
    b = f"blk.{c.L_all - c.nextn}."
    return {n[len(b):]: n for n in f.tensors if n.startswith(b)}


# ---------------------------------------------------------------------------------------------------------------- state
def new_state(c, S, CTX, extra_groups=0):
    """Zeroed per-slot state for S slots (+1 scratch slot), the same layout as models/qwen3_8_27b. extra_groups: K/V groups
    after the model's NG (speculative decoding keeps the MTP layer's K/V in group NG)."""
    kv = (c.NG + extra_groups, S + 1, CTX, c.KV, c.HD)
    return ((jnp.zeros(kv, c.act), jnp.zeros(kv, c.act)),
            (jnp.zeros((c.NG, c.FI - 1, S + 1, c.HV, c.DK, c.DV), f32),
             jnp.zeros((c.NG, c.FI - 1, S + 1, c.CK - 1, c.CONV), c.act)))


# ---------------------------------------------------------------------------------------------------------------- math
def rms(x, w, eps):
    """llama.cpp RMSNorm: normalise in f32, times the stored weight (already + 1 where the converter shifted it)"""
    x32 = x.astype(f32)
    return (x32 * lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps) * w).astype(x.dtype)


def rms_gated(x, z, w, eps):
    x32 = x.astype(f32)
    return (x32 * lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps) * w * jax.nn.silu(z.astype(f32))).astype(x.dtype)


def l2(x, eps):
    """llama.cpp build_gdn_l2_norm: x / sqrt(sum(x^2) + eps)"""
    return x * lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + eps)


def kv_write(Kc, Vc, slots, pos, ks, vs):
    """Write this call's K/V (NG,B,T,KV,HD) at (slot, position) of the model's groups; positions past the cache are dropped
    (an MTP group after them is written by its own layer)."""
    n = ks.shape[0]
    return (Kc.at[:n, slots[:, None], pos].set(ks, mode="drop"), Vc.at[:n, slots[:, None], pos].set(vs, mode="drop"))


def gated_delta_commit(R, rec, n, rep):
    """Speculative commit (the Qwen port's model.gated_delta_commit in this file's head order: value head h reads key head
    h % HK): replay the first n[b] accepted steps of every linear-attention layer from the state before the verify pass;
    steps past n[b] get g = beta = 0 and leave the state as it is (s * 1 + k * 0). R (NG, FI - 1, S+1, HV, DK, DV) f32, rows
    0..B-1; rec["k"] (NG, FI - 1, B, T, HK, DK) l2-normed keys, rec["v"] (.., HV, DV), rec["g"] / rec["b"] (NG, FI - 1, B, T, HV)."""
    NG, FI1, B, T = rec["g"].shape[:4]
    m = (jnp.arange(T)[None] < n[:, None])[..., None]

    def layer(l, R):
        g_, i_ = l // FI1, l % FI1
        s = lax.dynamic_slice(R, (g_, i_, 0, 0, 0, 0), (1, 1, B) + R.shape[3:])[0, 0]
        k = jnp.tile(rec["k"][g_, i_], (1, 1, rep, 1))
        v, g, b = rec["v"][g_, i_], jnp.where(m, rec["g"][g_, i_], 0.0), jnp.where(m, rec["b"][g_, i_], 0.0)
        q0 = jnp.zeros_like(k[:, 0])                    # the decode step's query: unused by its state update
        for t in range(T):
            _, s = M.gated_delta_step(q0, k[:, t], v[:, t], g[:, t], b[:, t], s)     # the decode step's state update
        return lax.dynamic_update_slice(R, s[None, None], (g_, i_, 0, 0, 0, 0))
    return lax.fori_loop(0, NG * FI1, layer, R)


def conv_commit(C, x, n):
    """Conv tail after n[b] accepted tokens: the last CK - 1 rows of [old tail, x[:, :n]]. C (NG, FI - 1, S+1, CK - 1, CONV) rows
    0..B-1, x (NG, FI - 1, B, T, CONV) the verify pass's conv inputs."""
    B, K1 = x.shape[2], C.shape[3]
    xe = jnp.concatenate([C[:, :, :B], x.astype(C.dtype)], 3)
    idx = (n[:, None] + jnp.arange(K1)[None])[None, None, :, :, None]
    return C.at[:, :, :B].set(jnp.take_along_axis(xe, jnp.broadcast_to(idx, xe.shape[:3] + (K1, 1)), 3))


def config_moe_spec(c):
    """the routing of a qwen35moe Config (llama.cpp src/models/qwen35moe.cpp: softmax gating, top K, weights renormalised,
    expert_weights_scale when set, SiLU experts, one sigmoid-gated shared expert); used until the Config carries a
    detected MoESpec (c.moe_spec)"""
    from types import SimpleNamespace
    return SimpleNamespace(n_expert=c.E, n_used=c.K, gating="softmax", norm_w=True, w_scale=c.w_scale or 1.0,
                           n_group=1, n_group_used=0, act="silu")


def make_forward(c, decode_kernels=None):
    """-> forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, keep=None, verify=False,
    hidden=False) -> dict(ck, cv, nxt, lg), the contract of models/qwen3_8_27b/model.make_forward. decode_kernels are not
    used (GGUF weights use mashbox.ops).

      keep (B,) bool or None: rows whose linear-attention state this call may change; the state rows of the others (padding
           rows of a decode bucket, rows paused mid-prefill) are written back as they were.
      verify: speculative verify of T tokens per row (contig rows): the linear-attention layers run T exact one-token decode
           steps and keep their inputs in out["rec"] (k, v, g, b, conv input x) without committing any state; full attention
           writes its K/V. out also has hid (post-norm hidden), lg (logits at every position) and tgt (their argmax).
      hidden: out["hid"] = the post-norm hidden state at every position (the MTP head's input)."""
    L, D, H, KV, HD, V = c.L, c.D, c.H, c.KV, c.HD, c.V
    FI, NG, RD, act = c.FI, c.NG, c.RD, c.act
    HK, HV, DK, DV, CK = c.HK, c.HV, c.DK, c.DV, c.CK
    KD, VD = HK * DK, HV * DV
    G, scale, eps = H // KV, c.scale, c.eps
    rep = HV // HK
    inv = (1.0 / (np.float32(c.theta) ** (np.arange(0, RD, 2, dtype=np.int64).astype(np.float32) / np.float32(RD)))).astype(np.float32)
    lin_ = ops.qmatmul
    mspec = (getattr(c, "moe_spec", None) or config_moe_spec(c)) if c.moe else None

    def forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, keep=None, verify=False,
                hidden=False):
        B, T = toks.shape
        assert not verify or contig, "verify runs on contig rows (slots 0..B-1)"
        Kc, Vc = ck
        R0, C0 = cv
        pos = p0[:, None] + jnp.arange(T, dtype=jnp.int32)[None]
        valid = jnp.arange(T)[None] < lens[:, None]
        fr = pos.astype(f32)[..., None] * jnp.asarray(inv)
        emb = jnp.concatenate([fr, fr], -1)
        cos, sin = jnp.cos(emb).astype(act)[:, :, None], jnp.sin(emb).astype(act)[:, :, None]
        m2 = (jnp.arange(T)[None, :, None] >= jnp.arange(T)[None, None, :]) & valid[:, None, :]
        if Tc:
            m1 = jnp.broadcast_to((jnp.arange(Tc)[None, :] < p0[:, None])[:, None, :], (B, T, Tc))
            mask = jnp.concatenate([m1, m2], -1)
        else:
            mask = m2
        fresh = p0 == 0
        x = ops.take_rows(W["embed"], toks).astype(act)

        def lin_attn(h, w, r0, c0):
            """-> (out, new state, new conv tail); with verify -> (out, record for the commit, conv input)"""
            qkv = lin_(h, w["qkv"])
            z = lin_(h, w["z"]).reshape(B, T, HV, DV)
            bb, aa = lin_(h, w["b"]), lin_(h, w["a"])
            xe = jnp.concatenate([c0, qkv], 1)
            cw = w["conv"]
            acc = xe[:, 0:T].astype(f32) * cw[:, 0]
            for t_ in range(1, CK):
                acc = acc + xe[:, t_:t_ + T].astype(f32) * cw[:, t_]
            co = jax.nn.silu(acc.astype(act).astype(f32)).astype(act)
            c1 = jnp.take_along_axis(xe, (lens[:, None] + jnp.arange(CK - 1)[None])[..., None], 1)
            qr, kr = co[..., :KD].reshape(B, T, HK, DK), co[..., KD:2 * KD].reshape(B, T, HK, DK)
            v = co[..., 2 * KD:].reshape(B, T, HV, DV).astype(f32)
            beta = jax.nn.sigmoid(bb.astype(f32)).astype(act).astype(f32)
            gg = M.softplus(aa.astype(f32) + w["dt"]) * w["A"]          # llama.cpp: softplus(alpha + dt) * ssm_a
            vm = valid[..., None]
            beta, gg = jnp.where(vm, beta, 0.0), jnp.where(vm, gg, 0.0)
            q = jnp.tile(l2(qr.astype(f32), eps), (1, 1, rep, 1))      # tiled: value head h reads key head h % HK
            k = jnp.tile(l2(kr.astype(f32), eps), (1, 1, rep, 1))
            if verify:            # T one-token decode steps, state not committed (the commit replays the accepted ones)
                mv = lambda a: jnp.moveaxis(a, 1, 0)

                def st(s, xs):
                    o_, s = M.gated_delta_step(*xs, s)
                    return s, o_
                _, o = lax.scan(st, r0, (mv(q), mv(k), mv(v), mv(gg), mv(beta)))
                o = jnp.moveaxis(o, 0, 1)
                r1, c1 = dict(k=l2(kr.astype(f32), eps), v=v, g=gg, b=beta), qkv
            elif T == 1:
                o, r1 = M.gated_delta_step(q[:, 0], k[:, 0], v[:, 0], gg[:, 0], beta[:, 0], r0)
                o = o[:, None]
            else:
                o, r1 = M.gated_delta_chunk(q, k, v, gg, beta, r0)
            o = rms_gated(o.astype(act), z, w["nw"], eps).reshape(B, T, VD)
            return lin_(o, w["out"]), r1, c1

        def attend(g, q, k, v):
            if Tc:
                kk, vv = jnp.concatenate([Kc[g][slots, :Tc], k], 1), jnp.concatenate([Vc[g][slots, :Tc], v], 1)
            else:
                kk, vv = k, v
            s = jnp.einsum("btkgd,bskd->bkgts", q, kk, preferred_element_type=f32) * scale
            p = jax.nn.softmax(jnp.where(mask[:, None, None], s, NEG), -1).astype(act)
            return jnp.einsum("bkgts,bskd->btkgd", p, vv, preferred_element_type=f32).astype(act)

        def full_attn(h, w, g):
            qg = lin_(h, w["wq"]).reshape(B, T, H, 2 * HD)
            q, gate = qg[..., :HD], qg[..., HD:].reshape(B, T, H * HD)
            k, v = lin_(h, w["wk"]).reshape(B, T, KV, HD), lin_(h, w["wv"]).reshape(B, T, KV, HD)
            q, k = rms(q, w["qn"], eps), rms(k, w["kn"], eps)
            q, k = M.rope_partial(q, cos, sin, RD), M.rope_partial(k, cos, sin, RD)
            o = attend(g, q.reshape(B, T, KV, G, HD), k, v).reshape(B, T, H * HD)
            o = (o.astype(f32) * jax.nn.sigmoid(gate.astype(f32))).astype(act)
            return lin_(o, w["wo"]), k, v

        def ffn(h, g, u, d):
            a = (jax.nn.silu(lin_(h, g).astype(f32)) * lin_(h, u).astype(f32)).astype(act)
            return lin_(a, d)

        def moe(h, w, layer=None):
            """routed experts + gated shared expert (mashbox.moe.ffn: llama.cpp build_moe_ffn order; the (row, expert)
            pairs of every row in this call are sorted by expert and only the picked experts run, each fetched once;
            rows past their length add no pairs). layer: w's routed experts are the whole stacks over layer groups and
            layer is this group's index (see the layer scan below)"""
            y = MF.moe_ffn(h.reshape(B * T, D), mspec, w, valid.reshape(B * T), layer=layer)
            return y.reshape(B, T, D).astype(act)

        def mlp(x, w, ln2, layer=None):
            h = rms(x, ln2, eps)
            if c.moe:
                return x + moe(h, w, layer)
            return x + ffn(h, w["wg"], w["wu"], w["wd"])

        def group(x, R, C, g, w, exps=None):
            gi = jnp.full((FI - 1, B), g, jnp.int32)
            ii = jnp.broadcast_to(jnp.arange(FI - 1, dtype=jnp.int32)[:, None], (FI - 1, B))
            si = jnp.broadcast_to(slots[None, :], (FI - 1, B))
            Rg0, Cg0 = R[gi, ii, si], C[gi, ii, si]
            Rg = jnp.where(fresh[None, :, None, None, None], 0.0, Rg0)
            Cg = jnp.where(fresh[None, :, None, None], jnp.zeros((), C.dtype), Cg0)
            rl, cl = [], []
            k = v = None
            for i in range(FI):
                h = rms(x, w["ln1"][i], eps)
                if i < FI - 1:
                    o, r1, c1 = lin_attn(h, w["lin"][i], Rg[i], Cg[i])
                    rl.append(r1 if verify else r1.astype(R0.dtype))
                    cl.append(c1 if verify else c1.astype(C0.dtype))
                else:
                    o, k, v = full_attn(h, w["full"], g)
                wm = w["mlp"][i] if exps is None else dict(w["mlp"][i], **exps[i])
                x = mlp(x + o, wm, w["ln2"][i], None if exps is None else g)
            if verify:
                rec = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *rl)
                rec["x"] = jnp.stack(cl)
                return x, R, C, k, v, rec
            Rn, Cn = jnp.stack(rl), jnp.stack(cl)
            if keep is not None:
                km = keep[None, :, None, None]
                Rn, Cn = jnp.where(km[..., None], Rn, Rg0), jnp.where(km, Cn, Cg0)
            R = R.at[gi, ii, si].set(Rn)
            C = C.at[gi, ii, si].set(Cn)
            return x, R, C, k, v, None

        if not isinstance(W["g"], list):
            # the routed experts stay whole stacks over the groups (groups, E, ...) and get the group index: a scan
            # slice of them would be copied into the expert kernel's call at every layer, the index reads in place
            wx, exps = W["g"], None
            if c.moe:
                exps = [{k: m[k] for k in MF.EXPERT_KEYS if k in m} for m in wx["mlp"]]
                wx = dict(wx, mlp=[{k: v for k, v in m.items() if k not in MF.EXPERT_KEYS} for m in wx["mlp"]])
            held, wx = ops.scan_split(wx)              # kernel planes stay whole: the kernel reads group g in place

            def body(carry, xg):
                x, R, C = carry
                x, R, C, k, v, rec = group(x, R, C, xg["g"], ops.scan_join(held, xg["w"], xg["g"]), exps)
                return (x, R, C), dict(k=k, v=v) if rec is None else dict(k=k, v=v, rec=rec)
            (x, R1, C1), ys = lax.scan(body, (x, R0, C0), dict(w=wx, g=jnp.arange(NG, dtype=jnp.int32)))
        else:
            R1, C1, yl = R0, C0, []
            for g in range(NG):
                x, R1, C1, k, v, rec = group(x, R1, C1, g, W["g"][g])
                yl.append(dict(k=k, v=v) if rec is None else dict(k=k, v=v, rec=rec))
            ys = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *yl)
        Kc, Vc = kv_write(Kc, Vc, slots, pos, ys["k"], ys["v"])
        out = dict(ck=(Kc, Vc), cv=(R1, C1))
        if verify:                # logits at every position, as one-token decode computes them for its argmax
            xn = rms(x, W["norm"], eps)
            lg = lin_(xn, W["head"])
            out.update(rec=ys["rec"], hid=xn, tgt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg.astype(f32))
            return out
        if hidden:                # post-norm hidden at every position (the MTP head's input)
            out["hid"] = rms(x, W["norm"], eps)
        if all_logits:
            out["lg"] = lin_(rms(x, W["norm"], eps), W["head"]).astype(f32)
            return out
        hl = rms(x[jnp.arange(B), jnp.maximum(lens - 1, 0)], W["norm"], eps)
        lg = lin_(hl, W["head"])
        out.update(nxt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg.astype(f32))
        return out

    return forward
