"""Native bfloat16 matmul for BF16 weights (a Hugging Face model's own weights, or a BF16 GGUF file).

The weights stay on the device as bfloat16 (rows, cols), the values as stored, and multiply in one dot_general with
float32 accumulation, the way the matrix units take bfloat16. Without it, a BF16 weight runs on the exact path, which
rebuilds float32 weights from the bytes at every call.

FALLBACK: when the native path cannot be used (MASHBOX_DENSE16=off, or a preparing or tracing error), the weight runs on
the dequantized path (mashbox.ops' exact reference) and one log line says so. Output stays correct either way.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

f32 = jnp.float32
TYPES = frozenset({30})                 # ggml BF16
_STATE = {"warned": set()}


def enabled():
    return os.environ.get("MASHBOX_DENSE16", "on").lower() not in ("off", "0", "no", "false")


def _warn(key, msg):
    if key not in _STATE["warned"]:
        _STATE["warned"].add(key)
        print(msg, file=sys.stderr, flush=True)


@jax.tree_util.register_pytree_node_class
class Dense16:
    """BF16 weight handle: w (..., rows, cols) bfloat16, plus the ggml type and one tensor's logical (rows, cols)"""

    def __init__(self, w, t, shape):
        self.w, self.t, self.shape = w, int(t), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.w,), (self.t, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], *aux)

    def __repr__(self):
        return f"Dense16({self.shape})"


def _host_values(raw, shape):
    import ml_dtypes
    a = np.asarray(raw)
    return a.reshape(a.shape[:-2] + (shape[-2], shape[-1] * 2)).view(ml_dtypes.bfloat16)


def prepare(raw, t, shape):
    """raw (..., rows, row_bytes) uint8 BF16 blocks -> Dense16, or None (the caller uses the dequantized path)"""
    if int(t) not in TYPES or not enabled():
        return None
    try:
        if isinstance(raw, jax.Array):
            w = lax.bitcast_convert_type(raw.reshape(raw.shape[:-1] + (shape[-1], 2)), jnp.bfloat16)
        else:
            w = jax.device_put(_host_values(raw, shape), jax.devices()[0])
        return Dense16(w, t, shape)
    except Exception as e:
        _warn("prepare", f"mashbox: native bf16 matmul unavailable ({type(e).__name__}: {e}); BF16 weights run on the "
                         f"dequantized path")
        return None


def native(x, w):
    return lax.dot_general(x, w.astype(x.dtype) if x.dtype == jnp.bfloat16 else w,
                           (((x.ndim - 1,), (w.ndim - 1,)), ((), ())), preferred_element_type=f32)


def exact(x, w):
    return lax.dot_general(x.astype(f32), w.astype(f32), (((x.ndim - 1,), (w.ndim - 1,)), ((), ())),
                           precision=lax.Precision.HIGHEST, preferred_element_type=f32)


def matmul(x, h):
    """x (..., cols) @ W.T -> (..., rows) in x's dtype; the exact float32 product when the native one fails to trace"""
    try:
        return native(x, h.w).astype(x.dtype)
    except Exception as e:
        _warn("matmul", f"mashbox: native bf16 matmul failed ({type(e).__name__}: {e}); this weight runs on the "
                        f"dequantized path")
        return exact(x, h.w).astype(x.dtype)


def dequant(h):
    return h.w.astype(f32)


def take_rows(h, ids):
    return h.w[ids].astype(f32)
