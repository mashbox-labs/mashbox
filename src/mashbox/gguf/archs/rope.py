"""Rope scaling for the GGUF dense architectures: YaRN (`<arch>.rope.scaling.type` = "yarn").

As llama.cpp runs it (ggml's rope_yarn with the context's yarn_attn_factor, beta_fast 32, beta_slow 1, ext_factor 1)
and as transformers computes it (_compute_yarn_parameters): pair i of the rotary dims turns at
theta^(-2i/RD) * f_i, where f_i = s * (1 - mix_i) + mix_i with s = 1 / factor. mix_i is 1 (the trained frequency) for the
fast pairs and 0 (interpolated by the factor) for the slow ones, falling linearly between the correction dims
floor(dim(beta_fast)) and ceil(dim(beta_slow)) of the original context, dim(r) = RD ln(n_orig / (2 pi r)) / (2 ln theta).
cos and sin are scaled by 0.1 ln(factor) + 1, times rope.scaling.attn_factor when the file sets one. With a log
multiplier m the scale is (0.1 ln(factor) + 1) / (0.1 m ln(factor) + 1). llama.cpp reads
rope.scaling.yarn_log_multiplier only in its mistral3 and deepseek2-family loaders, so m comes from the caller:
llama, qwen2 and qwen3 files run without it, as in llama.cpp, even when the file carries the key.
"""
import math

import numpy as np

BETA_FAST, BETA_SLOW = 32.0, 1.0


def _mscale(factor, m=1.0):
    return 1.0 if factor <= 1.0 else 0.1 * m * math.log(factor) + 1.0


def yarn(k, RD, theta, ctx_train, log_mul=0.0):
    """-> (f: (RD/2,) float32 per-pair angle factors, cos/sin scale). k(name, default) reads `<arch>.<name>`;
    log_mul: the yarn_log_multiplier, for the architectures whose llama.cpp loader reads it."""
    factor = float(k("rope.scaling.factor", 1.0) or 1.0)
    n_orig = int(k("rope.scaling.original_context_length", 0) or ctx_train)
    dim = lambda rot: RD * math.log(n_orig / (rot * 2 * math.pi)) / (2 * math.log(theta))
    lo = max(0.0, math.floor(dim(BETA_FAST)))
    hi = min(RD - 1.0, math.ceil(dim(BETA_SLOW)))
    y = (np.arange(RD // 2, dtype=np.float32) - np.float32(lo)) / np.float32(max(0.001, hi - lo))
    mix = np.float32(1.0) - np.clip(y, 0.0, 1.0)
    s = np.float32(1.0 / factor)
    f = (s * (np.float32(1.0) - mix) + mix).astype(np.float32)
    m = float(log_mul or 0.0)
    scale = _mscale(factor) / _mscale(factor, m) if m else _mscale(factor)
    return f, scale * float(k("rope.scaling.attn_factor", 1.0) or 1.0)
