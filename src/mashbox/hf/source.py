"""The weights of a Hugging Face model folder, read without copying: `*.safetensors` shards, or one lossless `.bshj`.

  safetensors  the Hugging Face format (github.com/huggingface/safetensors): 8 bytes little-endian header length, a JSON
               header {name: {dtype, shape, data_offsets}}, then the data. Read here with the standard library and NumPy
               memory maps (no torch, no extra dependency).
  .bshj        a lossless copy of the same tensors under the same names, read with mashbox.bshj (row-major tensors only;
               a device-form file is refused).

Every tensor comes back as (dtype name, shape, raw bytes as a read-only uint8 view). Nothing is converted here.
"""
import glob
import json
import os
import struct

import numpy as np

# safetensors dtype -> bytes per value, for the dtypes the engine reads from a Hugging Face model
ST_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}
BSHJ_DTYPES = {"bf16": "BF16", "f16": "F16", "f32": "F32"}
SKIP_FILES = ("consolidated.safetensors",)          # Mistral's single-file copy of the same weights


class HFError(Exception):
    """A model folder or repo this version cannot run. The message is for the person, exit code 2."""


class Source:
    """name -> (dtype, shape, raw uint8 view); `files` are the paths read"""

    def __init__(self):
        self.tensors, self.files, self._maps, self.kind = {}, [], [], None

    def close(self):
        for m in self._maps:
            try:
                m.close()
            except Exception:
                pass
        self._maps = []


def weight_files(d):
    """-> ("safetensors", [shards]) or ("bshj", [file]) for a model folder, or (None, [])"""
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        names = sorted(set(json.load(open(idx))["weight_map"].values()))
        return "safetensors", [os.path.join(d, n) for n in names]
    st = sorted(p for p in glob.glob(os.path.join(d, "*.safetensors")) if os.path.basename(p) not in SKIP_FILES)
    if st:
        return "safetensors", st
    bj = sorted(glob.glob(os.path.join(d, "*.bshj")))
    if bj:
        return "bshj", bj
    return None, []


def read_safetensors_header(path):
    """-> (header dict without __metadata__, data start offset)"""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            raise HFError(f"mashbox: {os.path.basename(path)} is not a safetensors file")
        n = struct.unpack("<Q", head)[0]
        if n > 100_000_000 or n + 8 > os.path.getsize(path):
            raise HFError(f"mashbox: {os.path.basename(path)} has a bad safetensors header")
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def open_source(d):
    """-> Source for the model folder d"""
    kind, files = weight_files(d)
    if kind is None:
        raise HFError(f"mashbox: {d} holds no *.safetensors or .bshj weights")
    src = Source()
    src.kind = kind
    if kind == "bshj":
        if len(files) != 1:
            raise HFError(f"mashbox: {d} holds {len(files)} .bshj files; the engine reads one per model folder")
        _open_bshj(src, files[0])
        return src
    for p in files:
        if not os.path.isfile(p):
            raise HFError(f"mashbox: shard {os.path.basename(p)} is missing from {d}")
        hdr, base = read_safetensors_header(p)
        mm = np.memmap(p, dtype=np.uint8, mode="r")
        src._maps.append(getattr(mm, "_mmap", None) or mm)
        for name, t in hdr.items():
            dt = t["dtype"]
            if dt not in ST_DTYPES:
                raise HFError(f"mashbox: tensor {name} is stored as {dt}; this version reads BF16, F16 and F32 "
                              f"safetensors (a pre-quantized repo runs from its GGUF, or from the maker's own repo)")
            b, e = t["data_offsets"]
            shape = tuple(int(s) for s in t["shape"])
            if e - b != int(np.prod(shape, dtype=np.int64)) * ST_DTYPES[dt]:
                raise HFError(f"mashbox: tensor {name} in {os.path.basename(p)} has a size that does not match its shape")
            src.tensors[name] = (dt, shape, mm[base + b:base + e])
        src.files.append(p)
    return src


def _open_bshj(src, path):
    from .. import bshj
    try:
        f = bshj.File(path)
    except ValueError as e:
        raise HFError(f"mashbox: {os.path.basename(path)}: {e}. This version reads lossless .bshj copies of a Hugging Face "
                      f"model (row-major tensors)")
    for name, t in f.tensors.items():
        lay = t.get("layout", "row@1")
        if lay != "row@1" or t["dtype"] not in BSHJ_DTYPES:
            raise HFError(f"mashbox: tensor {name} in {os.path.basename(path)} is stored as {t['dtype']} ({lay}); this "
                          f"version reads lossless .bshj copies (bf16, f16, f32, row-major)")
        a = f.array(name)
        src.tensors[name] = (BSHJ_DTYPES[t["dtype"]], tuple(int(s) for s in t["shape"]),
                             a.reshape(-1).view(np.uint8))
    src.files.append(path)
    src._maps.append(f)
