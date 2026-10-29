"""GGUF architecture `gemma4` (dense files: Gemma 4 12B / 31B) in plain JAX, following llama.cpp's graph
(src/models/gemma4.cpp) and reading the file's tensors as stored:

  * the token embedding is scaled by sqrt(embedding_length); the output head is the embedding when there is no
    output.weight; the logits are soft-capped (final_logit_softcapping * tanh(logits / final_logit_softcapping));
  * two layer kinds from `attention.sliding_window_pattern`: sliding-window layers (`key_length_swa`, `rope.freq_base_swa`,
    `rope.dimension_count_swa`, keys older than `attention.sliding_window` positions are masked) and full layers
    (`key_length`, `rope.freq_base`, `rope_freqs.weight` frequency factors); the K/V head count can differ per layer;
  * per-head RMSNorm on q and k with weights, on v without; a layer without attn_v uses its k projection as v;
    attention scale 1; rotary embedding on halves (NEOX);
  * post-attention and post-FFN RMSNorms, a GELU (tanh form) gated FFN, and the per-layer `layer_output_scale`.
RMSNorm weights are multiplied as stored. Big linears stay in the file's own blocks (any ggml type) and run through
mashbox.ops: the TPU kernel when the kernel package covers the type, else the exact float32 reference.

The forward has the contract of models/qwen3_8_27b/model.make_forward. The engine keeps one K/V array for every layer,
so each token's K (and V) of a layer is stored flat in a row as wide as the widest layer (Config.KV = 1, Config.HD = that
width); narrower layers use the front of the row.

Not handled (refused at load with a message): per-layer token embeddings (E2B / E4B), shared K/V layers, routed experts
(26B-A4B), a fused qkv tensor, rope scaling.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ... import ops
from ..reader import GGUFError
from .qwen35 import rms, kv_write, _T
from .dense import rope

f32, bf16 = jnp.float32, jnp.bfloat16
# the Models label (docs/gguf.md) of each architecture ARCHS maps to this module; catalog reads it with ast
MODELS = {"gemma4": "Gemma 4 (dense)"}
NEG = -1e30


class Config:
    """Shapes from the GGUF metadata (keys `gemma4.*`) and the tensor table."""

    def __init__(self, f, act=bf16, layers=None):
        A = self.arch = f.arch
        k = lambda s, d=None: f.get(f"{A}.{s}", d)
        no = lambda why: GGUFError(f"mashbox: {A} file with {why} is not supported in this version")
        self.L_all = self.L = int(k("block_count"))
        self.nextn = 0
        self.D, self.F = int(k("embedding_length")), int(k("feed_forward_length"))
        self.H = int(k("attention.head_count"))
        if int(k("embedding_length_per_layer_input", 0) or 0):
            raise no("per-layer token embeddings")
        if int(k("attention.shared_kv_layers", 0) or 0):
            raise no("shared K/V layers")
        names = list(f.tensors)
        if any(".attn_qkv." in n for n in names):
            raise no("a fused qkv tensor")
        if any("_exps." in n or ".ffn_gate_inp." in n for n in names):
            raise no("routed experts")
        if k("rope.scaling.type") not in (None, "none"):
            raise no(f"rope scaling '{k('rope.scaling.type')}'")
        swa = k("attention.sliding_window_pattern")
        swa = [bool(x) for x in (swa if swa is not None and not np.isscalar(swa) else [False] * self.L)]
        if len(swa) != self.L:
            raise no("a sliding-window pattern that does not list every layer")
        kv = k("attention.head_count_kv", self.H)
        kv = [int(x) for x in kv] if not np.isscalar(kv) else [int(kv)] * self.L
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.window = int(k("attention.sliding_window", 0) or 0)
        self.softcap = float(k("final_logit_softcapping", 0.0) or 0.0)
        hd_full, hd_swa = int(k("attention.key_length")), int(k("attention.key_length_swa", k("attention.key_length")))
        if int(k("attention.value_length", hd_full)) != hd_full or int(k("attention.value_length_swa", hd_swa)) != hd_swa:
            raise no("key_length != value_length")
        rd_full, rd_swa = int(k("rope.dimension_count", hd_full)), int(k("rope.dimension_count_swa", hd_swa))
        th_full, th_swa = float(k("rope.freq_base", 10000.0)), float(k("rope.freq_base_swa", 10000.0))
        if layers is not None:                      # tests only
            assert 0 < layers <= self.L
            self.L = layers
        self.swa = swa[:self.L]
        self.kv = kv[:self.L]
        self.hd = [hd_swa if s else hd_full for s in self.swa]
        self.rd = [rd_swa if s else rd_full for s in self.swa]
        self.theta = [th_swa if s else th_full for s in self.swa]
        self.freqs = "rope_freqs.weight" in f.tensors
        # the engine's state layout: one K/V group per layer (NG), stored flat (KV = 1, HD = the widest layer's row)
        self.NG, self.FI = self.L, 1
        self.KV, self.HD = 1, max(a * b for a, b in zip(self.kv, self.hd))
        self.HV = self.DK = self.DV = self.CONV = 1
        self.CK = 2
        self.types = ["sliding_attention" if s else "full_attention" for s in self.swa]
        self.V = int(f.tensors["token_embd.weight"].shape[0])
        self.tie = "output.weight" not in f.tensors
        self.ctx_train = int(k("context_length", 0) or 0)
        self.act = act
        self.vnorm, self.qscale, self.fscale = True, 1.0, [1.0] * self.L     # archs/gemma3.py sets its own


# ---------------------------------------------------------------------------------------------------------------- load
BIG = dict(wq="attn_q.weight", wk="attn_k.weight", wo="attn_output.weight", wg="ffn_gate.weight", wu="ffn_up.weight",
           wd="ffn_down.weight")
SMALL = dict(ln1="attn_norm.weight", qn="attn_q_norm.weight", kn="attn_k_norm.weight", pa="post_attention_norm.weight",
             ln2="ffn_norm.weight", pf="post_ffw_norm.weight")
OPTIONAL_BIG = dict(wv="attn_v.weight")
OPTIONAL_SMALL = dict(osc="layer_output_scale.weight")
MULTIPLIED = tuple(BIG.values()) + tuple(OPTIONAL_BIG.values())


def load(f, c, use_kernels=True, log=None):
    """-> (W, info). Every layer is its own entry (the two layer kinds have different shapes): big linears as ops handles
    from the file's blocks, small tensors as exact float32."""
    import time
    t0, stats = time.time(), dict(bytes=0, values=0, kernel_bytes=0, ref_bytes=0, ref_types={})

    def count(t):
        stats["bytes"] += t.nbytes
        stats["values"] += int(np.prod(t.shape))

    def big(name):
        t = f.tensors[name]
        count(t)
        if t.type not in ops.reference.TYPES:
            raise GGUFError(f"mashbox: {name} is ggml type {t.type}, which the GGUF reader in this version cannot read")
        h = ops.prepare(t.rows(), t.type, t.shape, use_kernel=use_kernels)
        if isinstance(h, ops.Packed):
            stats["kernel_bytes"] += sum(int(a.nbytes) for a in jax.tree_util.tree_leaves(h.packed))
        else:
            stats["ref_bytes"] += t.nbytes
            stats["ref_types"][t.type_name] = stats["ref_types"].get(t.type_name, 0) + t.nbytes
        return h

    def small(name):
        t = f.tensors[name]
        count(t)
        return ops.reference.dequant(jnp.asarray(t.rows()), t.type, t.shape[-1])

    def need(name):
        if name not in f.tensors:
            raise GGUFError(f"mashbox: {c.arch} file is missing tensor {name}")
        return name

    G = []
    for l in range(c.L):
        b = f"blk.{l}."
        w = {r: big(need(b + n)) for r, n in BIG.items()}
        w.update({r: small(need(b + n)) for r, n in SMALL.items()})
        w.update({r: big(b + n) for r, n in OPTIONAL_BIG.items() if b + n in f.tensors})
        w.update({r: small(b + n) for r, n in OPTIONAL_SMALL.items() if b + n in f.tensors})
        G.append(w)
    W = dict(g=G)
    t = f.tensors["token_embd.weight"]
    W["embed"] = ops.prepare(t.rows(), t.type, t.shape, use_kernel=False)       # rows are gathered, not multiplied
    count(t)
    W["norm"] = small("output_norm.weight")
    W["head"] = big("output.weight") if not c.tie else ops.prepare(t.rows(), t.type, t.shape, use_kernel=use_kernels)
    if c.freqs:
        W["freqs"] = small("rope_freqs.weight")
    jax.block_until_ready(jax.tree_util.tree_leaves(W))
    info = dict(load_s=round(time.time() - t0, 2), device_bytes=int(stats["bytes"]), n_weights=int(stats["values"]),
                bpw=round(stats["bytes"] * 8 / max(1, stats["values"]), 3), file=f.path.rsplit("/", 1)[-1],
                layers_stacked=False, kernel_bytes=stats["kernel_bytes"], reference_bytes=stats["ref_bytes"],
                reference_types=stats["ref_types"], mtp_layers=0)
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
    """Zeroed per-slot state for S slots (+1 scratch slot): K/V per layer stored flat, and the engine's (R, C)
    linear-attention pair as one zero value per slot (slot on axis 2, as the engine moves it)."""
    kv = (c.NG + extra_groups, S + 1, CTX, c.KV, c.HD)
    return ((jnp.zeros(kv, c.act), jnp.zeros(kv, c.act)),
            (jnp.zeros((1, 1, S + 1, 1, 1, 1), f32), jnp.zeros((1, 1, S + 1, 1, 1), c.act)))


def make_forward(c, decode_kernels=None):
    """-> forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, keep=None, verify=False,
    hidden=False) -> dict(ck, cv, nxt, lg), the contract of models/qwen3_8_27b/model.make_forward. verify (speculative
    decoding) is not offered for these files."""
    D, H, act, eps, WD = c.D, c.H, c.act, c.eps, c.HD
    lin_ = ops.qmatmul
    inv = [(1.0 / (np.float32(th) ** (np.arange(0, rd, 2, dtype=np.int64).astype(np.float32) / np.float32(rd))))
           .astype(np.float32) for th, rd in zip(c.theta, c.rd)]

    def forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, keep=None, verify=False,
                hidden=False):
        if verify:
            raise NotImplementedError("speculative verify is not offered for GGUF gemma4 files")
        B, T = toks.shape
        Kc, Vc = ck
        pos = p0[:, None] + jnp.arange(T, dtype=jnp.int32)[None]
        valid = jnp.arange(T)[None] < lens[:, None]
        # key positions: the cache rows 0..Tc-1 (row = position) then this step's tokens
        kpos = jnp.concatenate([jnp.broadcast_to(jnp.arange(Tc, dtype=jnp.int32)[None], (B, Tc)), pos], 1) if Tc else pos
        kok = jnp.concatenate([jnp.arange(Tc)[None] < p0[:, None], valid], 1) if Tc else valid
        dist = pos[:, :, None] - kpos[:, None, :]                        # (B, T, Tc + T)
        m_full = (dist >= 0) & kok[:, None, :]
        m_swa = m_full & (dist < c.window) if c.window else m_full
        x = (ops.take_rows(W["embed"], toks).astype(f32) * np.float32(np.sqrt(D))).astype(act)

        def rot(l):
            fq = jnp.asarray(inv[l])
            if c.freqs and not c.swa[l]:
                fq = fq / W["freqs"][:c.rd[l] // 2]
            if c.fscale[l] != 1.0:                 # gemma3: linear rope scaling on the full layers
                fq = fq * np.float32(c.fscale[l])
            fr = pos.astype(f32)[..., None] * fq
            return jnp.cos(fr)[:, :, None], jnp.sin(fr)[:, :, None]

        def layer(x, w, l):
            # the layers are unrolled (their shapes and block types differ), so without this XLA may read every layer's
            # cache rows at the start and keep all of them alive at once (31.3 GB of temporaries for 8 slots x 2048 on
            # the 12B); the barrier makes layer l's cache read wait for layer l's input
            x, sl = lax.optimization_barrier((x, slots))
            KV, HD, RD = c.kv[l], c.hd[l], c.rd[l]
            G, n = H // KV, c.kv[l] * c.hd[l]
            cos, sin = rot(l)
            h = rms(x, w["ln1"], eps)
            q = rms(lin_(h, w["wq"]).astype(act).reshape(B, T, H, HD), w["qn"], eps)
            kr = lin_(h, w["wk"]).astype(act).reshape(B, T, KV, HD)
            v = lin_(h, w["wv"]).astype(act).reshape(B, T, KV, HD) if "wv" in w else kr
            k = rms(kr, w["kn"], eps)
            v = rms(v, 1.0, eps) if c.vnorm else v
            q, k = rope(q, cos, sin, RD, True), rope(k, cos, sin, RD, True)
            if Tc:
                kk = jnp.concatenate([Kc[l][sl, :Tc, 0, :n].reshape(B, Tc, KV, HD), k], 1)
                vv = jnp.concatenate([Vc[l][sl, :Tc, 0, :n].reshape(B, Tc, KV, HD), v], 1)
            else:
                kk, vv = k, v
            s = jnp.einsum("btkgd,bskd->bkgts", q.reshape(B, T, KV, G, HD), kk, preferred_element_type=f32)
            if c.qscale != 1.0:                    # gemma3 scales q (llama.cpp: before the product)
                s = s * np.float32(c.qscale)
            m = m_swa if c.swa[l] else m_full
            p = jax.nn.softmax(jnp.where(m[:, None, None], s, NEG), -1).astype(act)
            o = jnp.einsum("bkgts,bskd->btkgd", p, vv, preferred_element_type=f32).astype(act).reshape(B, T, H * HD)
            a = x + rms(lin_(o, w["wo"]).astype(act), w["pa"], eps)
            h = rms(a, w["ln2"], eps)
            g = (jax.nn.gelu(lin_(h, w["wg"]).astype(f32), approximate=True) * lin_(h, w["wu"]).astype(f32)).astype(act)
            y = a + rms(lin_(g, w["wd"]).astype(act), w["pf"], eps)
            if "osc" in w:
                y = (y.astype(f32) * w["osc"]).astype(act)
            flat = lambda z: jnp.pad(z.reshape(B, T, 1, n), ((0, 0), (0, 0), (0, 0), (0, WD - n)))
            return y, flat(k), flat(v)

        ks, vs = [], []
        for l in range(c.L):
            x, k, v = layer(x, W["g"][l], l)
            ks.append(k)
            vs.append(v)
        Kc, Vc = kv_write(Kc, Vc, slots, pos, jnp.stack(ks), jnp.stack(vs))
        out = dict(ck=(Kc, Vc), cv=cv)

        def head(hn):
            lg = lin_(hn, W["head"]).astype(f32)
            return c.softcap * jnp.tanh(lg / c.softcap) if c.softcap else lg

        if hidden:
            out["hid"] = rms(x, W["norm"], eps)
        if all_logits:
            out["lg"] = head(rms(x, W["norm"], eps))
            return out
        lg = head(rms(x[jnp.arange(B), jnp.maximum(lens - 1, 0)], W["norm"], eps))
        out.update(nxt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg)
        return out

    return forward
