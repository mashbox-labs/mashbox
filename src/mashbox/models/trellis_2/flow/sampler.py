"""JAX flow-matching Euler sampler with CFG, guidance interval and CFG rescale.

Same math as TRELLIS.2 FlowEulerGuidanceIntervalSampler (commit 75fbf018, MIT), written
from that code, for the dense sparse-structure latent (stage 2) and the padded sparse SLat
tokens (stage 4/stage 6/stage 7).

Model contract: model_apply(model_params, x, t, cond, **model_kwargs) -> v (same shape as x),
with t a float32 [B] vector holding 1000 * t (exactly what the official sampler feeds).

Latent layouts:
- dense: x [B, ...]; CFG-rescale std = torch .std over all non-batch dims (unbiased, ddof=1).
- sparse (token_mask given): x [B, Npad, C], token_mask [B, Npad] bool. CFG-rescale std is the
  official VarLenTensor.std: per batch item, population std over valid tokens x channels,
  computed one-pass as sqrt(E[x^2] - E[x]^2) from per-token channel means. Padded tokens never
  enter the std, and v is zeroed there so padding cannot leak NaN/inf into anything.

The whole sampler runs in one jit: each run of steps with the same mode (cond-only / CFG)
is one lax.scan, and the schedule coefficients come from schedule.py (float64 host math,
rounded once to float32 like torch scalar ops).
"""
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .schedule import MODE_CFG, MODE_COND, MODE_NEG, SamplerParams, build_segments


def _std_dense(x):
    axes = tuple(range(1, x.ndim))
    return jnp.std(x, axis=axes, ddof=1, keepdims=True)


def _std_sparse(x, mask):
    # x [B, N, C], mask [B, N]; returns [B, 1, 1]
    m = mask.astype(x.dtype)
    n = jnp.sum(m, axis=1)                              # [B]
    tok_mean = jnp.mean(x, axis=-1)                     # [B, N]
    tok_mean2 = jnp.mean(x * x, axis=-1)                # [B, N]
    mean = jnp.sum(tok_mean * m, axis=1) / n            # [B]
    mean2 = jnp.sum(tok_mean2 * m, axis=1) / n
    std = jnp.sqrt(mean2 - mean * mean)
    return std[:, None, None]


def _double(tree):
    return jax.tree_util.tree_map(lambda a: jnp.concatenate([a, a], axis=0), tree)


def _sample_impl(model_apply, p: SamplerParams, sparse: bool, batch_cfg: bool, return_traj: bool,
                 model_params, noise, cond, neg_cond, model_kwargs, token_mask):
    dtype = noise.dtype
    B = noise.shape[0]
    a = jnp.asarray(1 - p.sigma_min, dtype)                  # (1 - sigma_min), f64 -> f32 once
    g = jnp.asarray(p.guidance_strength, dtype)
    one_m_g = jnp.asarray(1 - p.guidance_strength, dtype)
    r = jnp.asarray(p.guidance_rescale, dtype)
    one_m_r = jnp.asarray(1 - p.guidance_rescale, dtype)
    std_fn = (lambda x: _std_sparse(x, token_mask)) if sparse else _std_dense
    vmask = token_mask[..., None] if sparse else None

    def call(x, tvec, c):
        v = model_apply(model_params, x, tvec, c, **model_kwargs)
        return v

    def make_body(mode):
        def body(x, coef):
            t_model, b, dt = coef
            tvec = jnp.full((B,), t_model, jnp.float32)
            if mode == MODE_COND:
                v = call(x, tvec, cond)
                pos = None
            elif mode == MODE_NEG:
                v = call(x, tvec, neg_cond)
                pos = None
            else:
                if batch_cfg:
                    both = model_apply(model_params, jnp.concatenate([x, x], 0),
                                       jnp.concatenate([tvec, tvec], 0),
                                       jnp.concatenate([cond, neg_cond], 0),
                                       **_double(model_kwargs))
                    pos, neg = both[:B], both[B:]
                else:
                    pos = call(x, tvec, cond)
                    neg = call(x, tvec, neg_cond)
                if sparse:
                    pos = jnp.where(vmask, pos, 0)
                    neg = jnp.where(vmask, neg, 0)
                v = g * pos + one_m_g * neg
                if p.guidance_rescale > 0:
                    x0_pos = a * x - b * pos
                    x0_cfg = a * x - b * v
                    std_pos = std_fn(x0_pos)
                    std_cfg = std_fn(x0_cfg)
                    x0_res = x0_cfg * (std_pos / std_cfg)
                    x0 = r * x0_res + one_m_r * x0_cfg
                    v = (a * x - x0) / b
            if sparse:
                v = jnp.where(vmask, v, 0)
            x_prev = x - dt * v
            if return_traj:
                x0_pred = a * x - b * v
                return x_prev, (x_prev, x0_pred)
            return x_prev, None
        return body

    x = noise
    traj_xt, traj_x0 = [], []
    for seg in build_segments(p):
        coefs = (jnp.asarray(seg.t_model), jnp.asarray(seg.b, dtype), jnp.asarray(seg.dt, dtype))
        x, ys = jax.lax.scan(make_body(seg.mode), x, coefs)
        if return_traj:
            traj_xt.append(ys[0])
            traj_x0.append(ys[1])
    out = {"samples": x}
    if return_traj:
        out["pred_x_t"] = jnp.concatenate(traj_xt, 0)
        out["pred_x_0"] = jnp.concatenate(traj_x0, 0)
    return out


def make_sampler(model_apply: Callable, params: SamplerParams, *, sparse: bool = False,
                 batch_cfg: bool = False, return_traj: bool = False, jit: bool = True):
    """Returns f(model_params, noise, cond, neg_cond, model_kwargs=None, token_mask=None) -> dict.

    batch_cfg=True runs cond and neg_cond as one model call on a 2B batch (one big matmul
    stream on the MXU instead of two); every array in model_kwargs is doubled on axis 0.
    """
    impl = partial(_sample_impl, model_apply, params, sparse, batch_cfg, return_traj)
    if jit:
        impl = jax.jit(impl)

    def run(model_params: Any, noise, cond, neg_cond, model_kwargs: Optional[dict] = None,
            token_mask=None):
        if sparse and token_mask is None:
            raise ValueError("sparse sampler needs token_mask [B, Npad]")
        return impl(model_params, noise, cond, neg_cond, model_kwargs or {}, token_mask)

    return run
