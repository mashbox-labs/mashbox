"""Call adapters for the unchanged TRELLIS.2 numerical port.

Only static model configuration is captured by compiled functions. Parameters,
coordinates, masks, conditioning and texture concatenation are runtime arguments.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BUCKETS = (8192, 16384)


def import_port(port=None):
    """The port's numerical modules by name (they live in this package; `port` is kept for older callers)."""
    from . import dinov3_jax, dit_trunk, fdg_mesh, glb_bake, preprocess, resample_u8, sparse_conv_jax, ss_decoder
    from . import flow as trellis_flow
    return {"dinov3_jax": dinov3_jax, "preprocess": preprocess, "resample_u8": resample_u8, "dit_trunk": dit_trunk,
            "trellis_flow": trellis_flow, "ss_decoder": ss_decoder, "sparse_conv_jax": sparse_conv_jax,
            "fdg_mesh": fdg_mesh, "glb_bake": glb_bake}

def choose_bucket(n):
    if n < 1:
        raise ValueError("empty sparse structure")
    for cap in BUCKETS:
        if n <= cap:
            return cap
    raise ValueError(f"sparse token count {n} exceeds supported bucket 16384; no truncation")


def pack(coords, feats, cap=None):
    coords, feats = np.asarray(coords, np.int32), np.asarray(feats, np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) != len(feats):
        raise ValueError("sparse coordinates [N,3] and features [N,C] must agree")
    cap = choose_bucket(len(coords)) if cap is None else int(cap)
    if len(coords) < 1 or len(coords) > cap:
        raise ValueError("sparse token count outside requested capacity")
    x = np.zeros((1, cap, feats.shape[-1]), np.float32)
    c = np.zeros((1, cap, 3), np.int32)
    m = np.zeros((1, cap), bool)
    x[0, :len(coords)], c[0, :len(coords)], m[0, :len(coords)] = feats, coords, True
    return jnp.asarray(x), jnp.asarray(c), jnp.asarray(m)


def abstract(tree):
    return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)


class Flow:
    """One outer jit contains the complete Euler/CFG sampler, including all steps."""
    def __init__(self, dt, cfg, sampler_params):
        from .flow.sampler import make_sampler
        self.cfg, self.sampler_params = cfg, sampler_params
        attn = partial(dt.attention_chunked, block_q=256, block_k=256)
        rt = dt.DiTRuntime(attn_self=attn, attn_cross=attn)
        if cfg.dense:
            def apply(weights, x, t, cond):
                return dt.ss_flow_forward(weights, x, t, cond, cfg, rt=rt)
        else:
            def apply(weights, x, t, cond, coords, mask, concat_feats):
                return dt.slat_flow_forward(weights, x, coords, mask, t, cond, cfg,
                                            concat_feats=concat_feats, rt=rt)
        # make_sampler(jit=False) leaves exactly this outer whole-loop jit.
        sampler = make_sampler(apply, sampler_params, sparse=not cfg.dense, jit=False)
        self.program = jax.jit(sampler)
        self.compiled = {}
        self.negatives = {}

    def arguments(self, params, noise, cond, coords=None, mask=None, concat=None):
        neg = self.negatives.get(tuple(cond.shape))
        if neg is None:
            neg = jnp.asarray(np.zeros(cond.shape, np.float32))
        kwargs = {} if self.cfg.dense else {"coords": coords, "mask": mask, "concat_feats": concat}
        return params, noise, cond, neg, kwargs, mask

    def compile(self, params, cond_tokens, cap=None):
        cfg = self.cfg
        if cfg.dense:
            shape = (1, cfg.in_channels, cfg.resolution, cfg.resolution, cfg.resolution)
            kwargs, mask = {}, None
        else:
            shape = (1, cap, cfg.out_channels)
            mask = jax.ShapeDtypeStruct((1, cap), jnp.bool_)
            concat_c = cfg.in_channels - cfg.out_channels
            kwargs = {
                "coords": jax.ShapeDtypeStruct((1, cap, 3), jnp.int32),
                "mask": mask,
                "concat_feats": (jax.ShapeDtypeStruct((1, cap, concat_c), jnp.float32) if concat_c else None),
            }
        cond = jax.ShapeDtypeStruct((1, cond_tokens, cfg.cond_channels), jnp.float32)
        cond_shape = (1, cond_tokens, cfg.cond_channels)
        if cond_shape not in self.negatives:
            self.negatives[cond_shape] = jax.device_put(np.zeros(cond_shape, np.float32))
        args = (abstract(params), jax.ShapeDtypeStruct(shape, jnp.float32), cond, cond, kwargs, mask)
        self.compiled[cap] = self.program.lower(*args).compile()
        return {"bucket": cap, "condition_tokens": cond_tokens, "noise_shape": list(shape),
                "weight_arguments": len(jax.tree_util.tree_leaves(params))}

    def run(self, params, noise, cond, coords=None, mask=None, concat=None):
        cap = None if self.cfg.dense else noise.shape[1]
        args = self.arguments(params, noise, cond, coords, mask, concat)
        fn = self.compiled.get(cap, self.program)
        return jax.block_until_ready(fn(*args)["samples"])


def load_flow(dt, prefix, sampler_params, *, dtype="bfloat16", wiring_blocks=None):
    import json
    cfg = dt.DiTConfig.from_trellis_json(json.loads(Path(str(prefix) + ".json").read_text()), dtype=dtype)
    if wiring_blocks is not None:
        if not 1 <= wiring_blocks <= cfg.num_blocks:
            raise ValueError("invalid reduced block count")
        cfg = replace(cfg, num_blocks=wiring_blocks)
        keys = dt.expected_state_dict_keys(cfg)
    else:
        keys = None
    sd = dt.load_safetensors(str(prefix) + ".safetensors", keys=keys)
    params = dt.params_from_state_dict(sd, cfg)
    return Flow(dt, cfg, sampler_params), params


def load_decoder(modules, prefix, pred_subdiv):
    import json
    dt, sparse = modules["dit_trunk"], modules["sparse_conv_jax"]
    cfg = json.loads(Path(str(prefix) + ".json").read_text())["args"]
    state = dt.load_safetensors(str(prefix) + ".safetensors")
    params = sparse.load_decoder_params(state, dtype=jnp.float32)
    from .decoder_tiles import TiledDecoder
    dec = TiledDecoder(sparse, params, cfg["model_channels"], cfg["num_blocks"], pred_subdiv)
    return dec


class NoiseStream:
    """Per-job vendor CPU RNG. Overrides do not shift subsequent stage draws."""
    def __init__(self, seed, overrides=None):
        import torch
        self.torch = torch
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.overrides = overrides or {}

    def normal(self, stage, shape):
        draw = self.torch.randn(shape, generator=self.generator, dtype=self.torch.float32).numpy()
        if stage not in self.overrides:
            return draw
        value = np.asarray(self.overrides[stage], np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{stage} injected noise must be finite with shape {shape}")
        return value


def host_heads(sparse, h):
    with jax.default_device(jax.devices("cpu")[0]):
        return sparse.fdg_heads(h)
