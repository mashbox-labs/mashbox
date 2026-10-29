"""GGUF architecture `gemma3` (Gemma 3 text models, 270M to 27B) in plain JAX, following llama.cpp's graph
(src/models/gemma3.cpp). It runs on the gemma4 module's load and forward; only the configuration differs:

  * the layer kinds come from `attention.sliding_window`: every layer is a sliding-window layer except every
    `attention.sliding_window_pattern`-th one (6 when the file does not say), as llama.cpp's set_swa_pattern;
  * full layers rotate with `rope.freq_base` and the linear `rope.scaling.factor`; sliding-window layers with
    `rope.freq_base_swa` (10000 when the file does not say) and no scaling;
  * q is scaled by 1/sqrt(key_length) (1/sqrt(embedding_length / head_count) for the 62-layer 27B), v has no norm,
    every layer has its own attn_v, and there is no per-layer output scale;
  * the logits are soft-capped only when the file has `final_logit_softcapping`.
The token embedding is scaled by sqrt(embedding_length), the norms are applied before and after attention and FFN, the
FFN is a GELU (tanh form) gated FFN, and the output head is the embedding when there is no output.weight (all as gemma4).

Not handled (refused at load with a message): rope scaling other than linear, key_length != value_length, a fused qkv
tensor, routed experts, the dense output projections of embedding models.
"""
import numpy as np
import jax.numpy as jnp

from ..reader import GGUFError
from .gemma4 import load, tensor_roles, mtp_tensors, new_state, make_forward  # noqa: F401  the shared graph

f32, bf16 = jnp.float32, jnp.bfloat16
# the Models label (docs/gguf.md) of each architecture ARCHS maps to this module; catalog reads it with ast
MODELS = {"gemma3": "Gemma 3"}
SWA_THETA = 10000.0          # llama.cpp hparams.rope_freq_base_train_swa default
SWA_PATTERN = 6              # llama.cpp gemma3 load_swa_pattern(ml, 6)


class Config:
    """Shapes from the GGUF metadata (keys `gemma3.*`) and the tensor table, in the gemma4 module's layout."""

    def __init__(self, f, act=bf16, layers=None):
        A = self.arch = f.arch
        k = lambda s, d=None: f.get(f"{A}.{s}", d)
        no = lambda why: GGUFError(f"mashbox: {A} file with {why} is not supported in this version")
        self.L_all = self.L = int(k("block_count"))
        self.nextn = 0
        self.D, self.F = int(k("embedding_length")), int(k("feed_forward_length"))
        self.H = int(k("attention.head_count"))
        names = list(f.tensors)
        if any(".attn_qkv." in n for n in names):
            raise no("a fused qkv tensor")
        if any("_exps." in n or ".ffn_gate_inp." in n for n in names):
            raise no("routed experts")
        if any(n.startswith("dense_2.") or n.startswith("dense_3.") for n in names):
            raise no("dense output projections")
        sc = k("rope.scaling.type")
        if sc not in (None, "none", "linear"):
            raise no(f"rope scaling '{sc}'")
        factor = float(k("rope.scaling.factor", 0.0) or 0.0) if sc == "linear" else 0.0
        fscale_full = 1.0 / factor if factor else 1.0
        window = int(k("attention.sliding_window", 0) or 0)
        pat = k("attention.sliding_window_pattern")
        if not window:
            swa = [False] * self.L
        elif pat is not None and not np.isscalar(pat):
            swa = [bool(x) for x in pat]
        else:
            n = SWA_PATTERN if pat is None else int(pat)
            swa = [n == 0 or (l % n < n - 1) for l in range(self.L)]
        if len(swa) != self.L:
            raise no("a sliding-window pattern that does not list every layer")
        kv = k("attention.head_count_kv", self.H)
        kv = [int(x) for x in kv] if not np.isscalar(kv) else [int(kv)] * self.L
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.window = window
        self.softcap = float(k("final_logit_softcapping", 0.0) or 0.0)
        hd = int(k("attention.key_length", self.D // self.H))
        if int(k("attention.value_length", hd)) != hd:
            raise no("key_length != value_length")
        rd = int(k("rope.dimension_count", hd))
        th_full, th_swa = float(k("rope.freq_base", 10000.0)), float(k("rope.freq_base_swa", SWA_THETA))
        L62 = self.L == 62                          # llama.cpp keys the 27B's attention scale off its 62 layers
        self.qscale = float(1.0 / np.sqrt(np.float32(self.D // self.H) if L62 else np.float32(hd)))
        self.vnorm = False
        if layers is not None:                      # tests only
            assert 0 < layers <= self.L
            self.L = layers
        self.swa = swa[:self.L]
        self.kv = kv[:self.L]
        self.hd = [hd] * self.L
        self.rd = [rd] * self.L
        self.theta = [th_swa if s else th_full for s in self.swa]
        self.fscale = [1.0 if s else fscale_full for s in self.swa]
        self.freqs = False                          # llama.cpp's gemma3 graph passes no frequency factors
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
