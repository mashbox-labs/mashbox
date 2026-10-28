"""Weights: a .bshj file with Hugging Face tensor names -> the device tree the forward pass reads.

In the file, every big linear (and the embedding and lm_head) is stored as `<name>.weight` int8 (out, in) plus
`<name>.weight_scale` f32 (out,): weight = q * scale per output row. Norms, conv, A_log, dt_bias and the small
in_proj_a / in_proj_b keep their original dtype and shape.

Device tree: {"embed", "norm", "head", "g": {"lin", "full", "mlp"}} with a leading group axis NG for the layer scan.
Big linears are lists with one (NG, out, in) buffer per position in the layer group; small tensors are stacked
(NG, n, ..). Each buffer is allocated once on the device and filled in place, layer by layer.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from ... import bshj


def _put(buf, x, idx):
    return buf.at[tuple(idx)].set(x)


_putj = jax.jit(_put, donate_argnums=(0,))


def prefix(names):
    return "model.language_model." if any(k.startswith("model.language_model.") for k in names) else "model."


def load(path, c, log=None):
    """-> (W, info). info: load seconds, device bytes, weight count, bits per weight on the device."""
    f = bshj.File(path)
    P = prefix(f.tensors)
    t0, nbytes, nel = time.time(), 0, 0

    def host(n):
        f.advise(n)
        return f.array(n)

    def small(fmt, ids, lead):
        nonlocal nbytes, nel
        a = np.stack([host(P + fmt.format(i)) for i in ids])
        x = jnp.asarray(a.reshape(lead + a.shape[1:]))
        nbytes += x.nbytes; nel += x.size
        return x

    def conv(ids, lead):          # conv1d.weight (CONV, 1, K) -> (.., CONV, K)
        nonlocal nbytes, nel
        a = np.stack([host(P + f"layers.{i}.linear_attn.conv1d.weight")[:, 0, :] for i in ids]).reshape(lead + (c.CONV, c.CK))
        x = jnp.asarray(a)
        nbytes += x.nbytes; nel += x.size
        return x

    def quantised(name):
        return name + ".weight_scale" in f.tensors

    def big(fmt, ids, lead):
        nonlocal nbytes, nel
        n0 = P + fmt.format(ids[0])
        if not quantised(n0[:-len(".weight")]):
            a = np.stack([host(P + fmt.format(i)) for i in ids]).reshape(lead + tuple(f.tensors[n0]["shape"]))
            x = jnp.asarray(a)
            nbytes += x.nbytes; nel += x.size
            return x
        shp = tuple(f.tensors[n0]["shape"])
        qb, sb = jnp.zeros(lead + shp, jnp.int8), jnp.zeros(lead + shp[:1], jnp.float32)
        for j, i in enumerate(ids):
            n = P + fmt.format(i)
            idx = jnp.asarray(np.unravel_index(j, lead), jnp.int32)
            qb = _putj(qb, jnp.asarray(host(n)), idx)
            sb = _putj(sb, jnp.asarray(host(n + "_scale")), idx)
        nbytes += qb.nbytes + sb.nbytes; nel += qb.size
        return dict(q=qb, s=sb)

    NG, FI = c.NG, c.FI

    def per(fmt, ids, n):
        return [big(fmt, ids[k::n], (NG,)) for k in range(n)]

    lin_ids = [l for l in range(c.L) if c.types[l] == "linear_attention"]
    full_ids = [l for l in range(c.L) if c.types[l] == "full_attention"]
    allx = list(range(c.L))
    la, fa = "layers.{}.linear_attn.", "layers.{}.self_attn."
    Gl, Gf, Gm = (NG, FI - 1), (NG,), (NG, FI)
    W = dict(g=dict(
        lin=dict(qkv=per(la + "in_proj_qkv.weight", lin_ids, FI - 1), z=per(la + "in_proj_z.weight", lin_ids, FI - 1),
                 a=big(la + "in_proj_a.weight", lin_ids, Gl), b=big(la + "in_proj_b.weight", lin_ids, Gl),
                 out=per(la + "out_proj.weight", lin_ids, FI - 1), conv=conv(lin_ids, Gl), A=small(la + "A_log", lin_ids, Gl),
                 dt=small(la + "dt_bias", lin_ids, Gl), nw=small(la + "norm.weight", lin_ids, Gl)),
        full=dict(wq=big(fa + "q_proj.weight", full_ids, Gf), wk=big(fa + "k_proj.weight", full_ids, Gf),
                  wv=big(fa + "v_proj.weight", full_ids, Gf), wo=big(fa + "o_proj.weight", full_ids, Gf),
                  qn=small(fa + "q_norm.weight", full_ids, Gf), kn=small(fa + "k_norm.weight", full_ids, Gf)),
        mlp=dict(ln1=small("layers.{}.input_layernorm.weight", allx, Gm), ln2=small("layers.{}.post_attention_layernorm.weight", allx, Gm),
                 wg=per("layers.{}.mlp.gate_proj.weight", allx, FI), wu=per("layers.{}.mlp.up_proj.weight", allx, FI),
                 wd=per("layers.{}.mlp.down_proj.weight", allx, FI))))

    def one(n):
        nonlocal nbytes, nel
        if quantised(n[:-len(".weight")]):
            q, s = jnp.asarray(host(n)), jnp.asarray(host(n + "_scale"))
            nbytes += q.nbytes + s.nbytes; nel += q.size
            return dict(q=q, s=s)
        x = jnp.asarray(host(n))
        nbytes += x.nbytes; nel += x.size
        return x

    W["embed"] = one(P + "embed_tokens.weight")
    W["norm"] = one(P + "norm.weight")
    W["head"] = W["embed"] if c.tie or "lm_head.weight" not in f.tensors else one("lm_head.weight")
    jax.block_until_ready(W)
    info = dict(load_s=round(time.time() - t0, 2), device_bytes=int(nbytes), n_weights=int(nel),
                bpw=round(nbytes * 8 / max(1, nel), 3), file=path.rsplit("/", 1)[-1],
                quant=(f.meta().get("bshj") or {}).get("quant", {}).get("type"))
    f.close()
    if log:
        log(info)
    return W, info
