"""Tensor parallel Qwen3.8-27B over N chips of one host (--tensor-parallel-size N, as vLLM).

JAX precedent (MaxText / JetStream, vLLM-TPU tpu-inference): one Mesh axis "tp", weights placed with NamedSharding
(Megatron layout: column-parallel in-projections, row-parallel out-projections), state sharded on the head axis, and the
jit programs partitioned by GSPMD. The single-chip path (weights.load, model.new_state) is not touched: this module only
places the same tensors on a mesh and hands the Engine a model object with the same two entry points.

The full-attention K/V cache read + attention core and the cache write run inside shard_map (the vLLM-TPU pattern): each
chip reads and writes only its own KV heads. Left to GSPMD, the slot gather from the cache made every chip hold a whole
copy of the cache during decode (v5e-4, ctx 32768 x 8 slots: 20.15 GB decode temp per chip, more than the 15.75 GB chip).
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ... import bshj
from . import model as QM
from . import weights as QW

try:                                    # jax >= 0.7
    from jax import shard_map as _shard_map

    def smap(f, mesh, in_specs, out_specs):
        return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
except ImportError:                     # older jax
    from jax.experimental.shard_map import shard_map as _shard_map

    def smap(f, mesh, in_specs, out_specs):
        return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)


def mesh_for(n):
    return Mesh(np.array(jax.devices()[:n]), ("tp",))


def _place(mesh, a, spec):
    sh = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(a.shape, sh, lambda idx: a[idx])


def load(path, c, mesh, log=None):
    """Same tree as weights.load, placed on the mesh. Column-parallel: out axis on tp; row-parallel: in axis on tp."""
    f = bshj.File(path)
    Pf = QW.prefix(f.tensors)
    t0, nbytes, nel = time.time(), 0, 0

    def host(n):
        f.advise(n)
        return f.array(n)

    def pl(a, spec):
        nonlocal nbytes, nel
        x = _place(mesh, a, spec)
        nbytes += x.nbytes; nel += x.size
        return x

    def rep(nd):
        return P(*([None] * nd))

    def quantised(name):
        return name + ".weight_scale" in f.tensors

    def big(fmt, ids, lead, mode):
        """mode 'col' (out sharded), 'row' (in sharded), 'heads' (axis -2 sharded, bf16 small), 'rep'."""
        n0 = Pf + fmt.format(ids[0])
        shp = tuple(f.tensors[n0]["shape"])
        nl = len(lead)
        if not quantised(n0[:-len(".weight")]):
            a = np.stack([host(Pf + fmt.format(i)) for i in ids]).reshape(lead + shp)
            spec = P(*([None] * nl), "tp", None) if mode in ("col", "heads") else P(*([None] * nl), None, "tp") if mode == "row" else rep(a.ndim)
            return pl(a, spec)
        q = np.stack([host(Pf + fmt.format(i)) for i in ids]).reshape(lead + shp)
        s = np.stack([host(Pf + fmt.format(i) + "_scale") for i in ids]).reshape(lead + shp[:1])
        if mode == "col":
            return dict(q=pl(q, P(*([None] * nl), "tp", None)), s=pl(s, P(*([None] * nl), "tp")))
        if mode == "row":
            return dict(q=pl(q, P(*([None] * nl), None, "tp")), s=pl(s, rep(s.ndim)))
        return dict(q=pl(q, rep(q.ndim)), s=pl(s, rep(s.ndim)))

    def small(fmt, ids, lead):
        a = np.stack([host(Pf + fmt.format(i)) for i in ids])
        a = a.reshape(lead + a.shape[1:])
        return pl(a, rep(a.ndim))

    def conv(ids, lead):
        a = np.stack([host(Pf + f"layers.{i}.linear_attn.conv1d.weight")[:, 0, :] for i in ids]).reshape(lead + (c.CONV, c.CK))
        return pl(a, rep(a.ndim))

    NG, FI = c.NG, c.FI

    def per(fmt, ids, n, mode):
        return [big(fmt, ids[k::n], (NG,), mode) for k in range(n)]

    lin_ids = [l for l in range(c.L) if c.types[l] == "linear_attention"]
    full_ids = [l for l in range(c.L) if c.types[l] == "full_attention"]
    allx = list(range(c.L))
    la, fa = "layers.{}.linear_attn.", "layers.{}.self_attn."
    Gl, Gf, Gm = (NG, FI - 1), (NG,), (NG, FI)
    W = dict(g=dict(
        lin=dict(qkv=per(la + "in_proj_qkv.weight", lin_ids, FI - 1, "col"), z=per(la + "in_proj_z.weight", lin_ids, FI - 1, "col"),
                 a=big(la + "in_proj_a.weight", lin_ids, Gl, "heads"), b=big(la + "in_proj_b.weight", lin_ids, Gl, "heads"),
                 out=per(la + "out_proj.weight", lin_ids, FI - 1, "row"), conv=conv(lin_ids, Gl), A=small(la + "A_log", lin_ids, Gl),
                 dt=small(la + "dt_bias", lin_ids, Gl), nw=small(la + "norm.weight", lin_ids, Gl)),
        full=dict(wq=big(fa + "q_proj.weight", full_ids, Gf, "col"), wk=big(fa + "k_proj.weight", full_ids, Gf, "col"),
                  wv=big(fa + "v_proj.weight", full_ids, Gf, "col"), wo=big(fa + "o_proj.weight", full_ids, Gf, "row"),
                  qn=small(fa + "q_norm.weight", full_ids, Gf), kn=small(fa + "k_norm.weight", full_ids, Gf)),
        mlp=dict(ln1=small("layers.{}.input_layernorm.weight", allx, Gm), ln2=small("layers.{}.post_attention_layernorm.weight", allx, Gm),
                 wg=per("layers.{}.mlp.gate_proj.weight", allx, FI, "col"), wu=per("layers.{}.mlp.up_proj.weight", allx, FI, "col"),
                 wd=per("layers.{}.mlp.down_proj.weight", allx, FI, "row"))))

    def one(n, mode):
        if quantised(n[:-len(".weight")]):
            q, s = host(n), host(n + "_scale")
            if mode == "embed":       # (V, D): shard D so the token gather stays local; per-row scale replicated
                return dict(q=pl(q, P(None, "tp")), s=pl(s, P(None)))
            return dict(q=pl(q, P("tp", None)), s=pl(s, P("tp")))      # head: shard the vocab (out) axis
        x = host(n)
        return pl(x, P(None, "tp") if x.ndim == 2 else rep(x.ndim))

    W["embed"] = one(Pf + "embed_tokens.weight", "embed")
    W["norm"] = pl(host(Pf + "norm.weight"), P(None))
    W["head"] = W["embed"] if c.tie or "lm_head.weight" not in f.tensors else one("lm_head.weight", "head")
    jax.block_until_ready(W)
    info = dict(load_s=round(time.time() - t0, 2), device_bytes=int(nbytes), n_weights=int(nel),
                bpw=round(nbytes * 8 / max(1, nel), 3), file=path.rsplit("/", 1)[-1], tp=mesh.devices.size,
                quant=(f.meta().get("bshj") or {}).get("quant", {}).get("type"))
    f.close()
    if log:
        log(info)
    return W, info


class TPModel:
    """What the Engine reads from a model module (make_forward, new_state), for a tp mesh."""

    def __init__(self, Q, mesh):
        self.Q, self.mesh = Q, mesh
        ns = lambda *s: NamedSharding(mesh, P(*s))
        self.kv_sh = ns(None, None, None, "tp", None)              # (NG, S+1, CTX, KV, HD): KV heads on tp
        self.r_sh = ns(None, None, None, "tp", None, None)         # (NG, FI-1, S+1, HV, DK, DV): value heads on tp
        self.c_sh = ns(None, None, None, None, None)               # conv tail: small, replicated

    def make_forward(self, c, decode_kernels=None):
        mesh = self.mesh
        kvs = P(None, None, None, "tp", None)                       # cache (NG, S+1, CTX, KV, HD) and ys k/v (NG, B, T, KV, HD)
        q5, k4 = P(None, None, "tp", None, None), P(None, None, "tp", None)

        def attend(Kc, Vc, g, q, k, v, mask, slots, Tc, scale):
            f = smap(lambda Kc, Vc, g, q, k, v, mask, slots: QM.attend_one(Kc, Vc, g, q, k, v, mask, slots, Tc, scale),
                     mesh, (kvs, kvs, P(), q5, k4, k4, P(), P()), q5)
            return f(Kc, Vc, g, q, k, v, mask, slots)

        def kv_write(Kc, Vc, slots, pos, ks, vs):
            return smap(QM.kv_write_one, mesh, (kvs, kvs, P(), P(), kvs, kvs), (kvs, kvs))(Kc, Vc, slots, pos, ks, vs)

        def kv_write_g(Kc, Vc, g, slots, pos, k, v):
            return smap(QM.kv_write_group, mesh, (kvs, kvs, P(), P(), P(), k4, k4), (kvs, kvs))(Kc, Vc, g, slots, pos, k, v)

        # exported kernels are single-chip shapes: plain JAX linear-attention decode under tp. The cache rides in the layer
        # loop carry (kv_in_loop): no whole-cache copies in the decode program.
        fwd = QM.make_forward(c, None, attend=attend, kv_write=kv_write, kv_in_loop=True, kv_write_g=kv_write_g)
        wsc = jax.lax.with_sharding_constraint
        kv_sh, r_sh, c_sh = self.kv_sh, self.r_sh, self.c_sh

        # Same signature as model.make_forward's forward: the Engine jits every model's forward with the same static
        # arguments and asks for the post-norm hidden on prefill. The speculative verify pass runs on one chip only
        # (it needs the K/V outside the layer loop); `mashbox run --mtp K` with --tensor-parallel-size above 1 is refused.
        def forward(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, verify=False, hidden=False):
            if verify:
                raise NotImplementedError("speculative decoding runs on one chip (no tensor parallel)")
            out = fwd(W, ck, cv, toks, p0, lens, slots, Tc, contig=contig, all_logits=all_logits, hidden=hidden)
            K_, V_ = out["ck"]
            R_, C_ = out["cv"]
            out["ck"] = (wsc(K_, kv_sh), wsc(V_, kv_sh))
            out["cv"] = (wsc(R_, r_sh), wsc(C_, c_sh))
            return out
        return forward

    def new_state(self, c, S, CTX, extra_groups=0):
        if extra_groups:
            raise NotImplementedError("speculative decoding runs on one chip (no tensor parallel)")
        f32, bf16 = jnp.float32, jnp.bfloat16
        kv = (c.NG, S + 1, CTX, c.KV, c.HD)
        z = lambda shape, dt, sh: jax.jit(lambda: jnp.zeros(shape, dt), out_shardings=sh)()
        return ((z(kv, bf16, self.kv_sh), z(kv, bf16, self.kv_sh)),
                (z((c.NG, c.FI - 1, S + 1, c.HV, c.DK, c.DV), f32, self.r_sh),
                 z((c.NG, c.FI - 1, S + 1, c.CK - 1, c.CONV), bf16, self.c_sh)))
