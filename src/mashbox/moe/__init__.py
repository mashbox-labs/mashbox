"""Mixture of experts: detection from metadata (spec), routing and the routed expert FFN, and the fit's MoE term
(need_bytes: the routed op's scratch, which gguf.need_bytes adds; MoE keeps no fit of its own)."""
import importlib

from .spec import MoESpec, MoEError, detect_gguf, detect_hf  # noqa: F401

_FFN = ("need_bytes", "routed_fn", "prepare_ungated")


def __getattr__(name):
    """.ffn needs jax: loaded on first use, so importing mashbox.moe (runtime/openapi.py reads moe.record) works on a
    base install without the [tpu] extra."""
    if name == "ffn" or name in _FFN:
        ffn = importlib.import_module(__name__ + ".ffn")   # not `from . import ffn`: its hasattr check calls this again
        return ffn if name == "ffn" else getattr(ffn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
