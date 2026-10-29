"""A stand-in model for engine tests on CPU: the real engine's state layout (K/V per slot and position, a linear-attention
state per slot) with a forward that, like the model's, starts a slot's state from zero at position 0, reads earlier
positions of its own slot masked by position, and leaves padding rows' state unchanged (keep mask). Its tokens depend on
the whole history, so a wrong cache resume changes them."""
import types

import jax
import jax.numpy as jnp
from jax import lax

V = 64


def new_state(c, S, CTX):
    kv = (1, S + 1, CTX, 2)
    return ((jnp.zeros(kv, jnp.bfloat16), jnp.zeros(kv, jnp.bfloat16)),
            (jnp.zeros((1, 1, S + 1, 1, 1, 1), jnp.float32), jnp.zeros((1, 1, S + 1, 1, 1), jnp.bfloat16)))


def make_forward(c, decode_kernels=None):
    def fwd(W, ck, cv, toks, p0, lens, slots, Tc, contig=False, all_logits=False, verify=False, hidden=False, keep=None):
        (K_, V_), (R, C) = ck, cv
        B, Tn = toks.shape
        CTX = K_.shape[2]
        live = jnp.arange(Tn)[None, :] < lens[:, None]
        r0 = jnp.where(p0 == 0, 0.0, R[0, 0, slots, 0, 0, 0])          # fresh state at position 0, as the model
        x = toks.astype(jnp.float32) * W["w"]

        def body(r, xt):
            r2 = jnp.where(xt[1], r * 0.75 + xt[0] + 1.0, r)
            return r2, r2
        r_end, rs = lax.scan(body, r0, (x.T, live.T))
        att = jnp.zeros(B, jnp.float32)
        if Tc:                      # earlier positions of the row's own slot, masked by position as attention is
            kk = K_[0, slots, :Tc, 0].astype(jnp.float32)
            att = 0.001 * jnp.sum(jnp.where(jnp.arange(Tc)[None, :] < p0[:, None], kk, 0.0), 1)
        h = jnp.take_along_axis(rs.T, jnp.maximum(lens - 1, 0)[:, None], 1)[:, 0] + att
        wpos = jnp.where(live, p0[:, None] + jnp.arange(Tn)[None, :], CTX)   # padding writes fall past the cache
        K_ = K_.at[0, slots[:, None], wpos, 0].set(rs.T.astype(K_.dtype), mode="drop")
        ok = jnp.ones(B, bool) if keep is None else keep
        R = R.at[0, 0, slots, 0, 0, 0].set(jnp.where(ok, r_end, R[0, 0, slots, 0, 0, 0]))
        lg = (jax.nn.one_hot(jnp.floor(h * 7.0).astype(jnp.int32) % c.V, c.V, dtype=jnp.float32)
              + 0.01 * jnp.arange(c.V, dtype=jnp.float32))
        out = dict(ck=(K_, V_), cv=(R, C), nxt=jnp.argmax(lg, -1).astype(jnp.int32), lg=lg)
        if all_logits:
            out["lg"] = jnp.broadcast_to(lg[:, None], (B, Tn, c.V))
        return out
    return fwd


MODEL = types.SimpleNamespace(new_state=new_state, make_forward=make_forward)


def config(V_=V, H=2):
    return types.SimpleNamespace(V=V_, H=H)


def weights():
    return {"w": jnp.ones((), jnp.float32)}
