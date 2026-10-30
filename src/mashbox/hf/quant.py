"""8-bit on load: a Hugging Face model whose bfloat16 weights do not fit the chip is turned into llama.cpp's Q8_0 blocks
on the host while it loads, with llama.cpp's own `gguf` package (gguf-py, MIT, github.com/ggml-org/llama.cpp:
gguf.quants.quantize, the same rounding as `convert_hf_to_gguf.py --outtype q8_0`). Only the multiplied weights change;
norms, biases and the embedding rows stay as stored. The Q8_0 blocks then run like any GGUF Q8_0 weight (the TPU kernel
when the kernel package covers Q8_0, else the dequantized path)."""
import numpy as np

from .adapter import Tensor, _f32

Q8_0, BLOCK, BLOCK_BYTES = 8, 32, 34
NAME = {0: "F32", 1: "F16", 30: "BF16"}


def quantize_q8_0(a):
    """float32 (rows, cols) -> Q8_0 blocks (rows, cols / 32 * 34) uint8, gguf-py's quantize"""
    import gguf
    return gguf.quants.quantize(np.ascontiguousarray(a, np.float32), gguf.GGMLQuantizationType.Q8_0)


def q8_0_names(f, A, c):
    """the multiplied 2-D weights that become Q8_0 (rows of whole 32-value blocks)"""
    _, mult = A.tensor_roles(f, c)
    return sorted(n for n in mult if n in f.tensors and f.tensors[n].type in NAME and len(f.tensors[n].shape) == 2
                  and f.tensors[n].shape[-1] % BLOCK == 0)


def to_q8_0(f, A, c):
    """switch f's multiplied weights to Q8_0 made on load (lazy: the blocks are made when the loader reads them)"""
    for n in q8_0_names(f, A, c):
        t = f.tensors[n]
        rows, cols = t.shape
        dt = NAME[t.type]

        def make(t=t, dt=dt):
            return quantize_q8_0(_f32(dt, t.make(), t.shape)).reshape(-1)
        f.tensors[n] = Tensor(n, Q8_0, t.shape, rows * (cols // BLOCK) * BLOCK_BYTES, make)
    f.size = sum(t.nbytes for t in f.tensors.values())
    f.form = "q8_0"
    return f

