"""Read a GGUF file with llama.cpp's own `gguf` Python package (gguf-py, MIT, github.com/ggml-org/llama.cpp).

GGUF and its block formats are llama.cpp's. This module only wraps gguf.GGUFReader: metadata as Python values, and for
every tensor its name, ggml type id, logical shape and its raw bytes as stored (a read-only NumPy view of the
memory-mapped file). Nothing is dequantized or converted here. Split files (`-00001-of-0000N.gguf`) are read as one.
"""
import os
import re

import numpy as np

from ..ops import reference as R

SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class GGUFError(Exception):
    """A file this version cannot run. The message is for the person, exit code 2."""


def _gguf():
    try:
        import gguf
    except ImportError:
        raise GGUFError("mashbox: reading GGUF files needs the `gguf` package: it is a dependency of mashbox, so this install is "
                        "incomplete: pip install gguf==0.19.0")
    return gguf


class Tensor:
    __slots__ = ("name", "type", "shape", "raw", "nbytes")

    def __init__(self, name, t, shape, raw):
        self.name, self.type, self.shape, self.raw, self.nbytes = name, int(t), tuple(shape), raw, int(raw.nbytes)

    @property
    def type_name(self):
        return R.TYPES[self.type][0] if self.type in R.TYPES else f"type {self.type}"

    def rows(self):
        """raw bytes as (..., rows, row_bytes) uint8 for 2-D and larger, (row_bytes,) for 1-D"""
        rb = R.row_bytes(self.type, self.shape[-1]) if self.type in R.TYPES else self.nbytes
        return self.raw.reshape(tuple(self.shape[:-1]) + (rb,))

    def __repr__(self):
        return f"Tensor({self.name}, {self.type_name}, {self.shape})"


class File:
    """One GGUF model (all split shards). .meta: key -> value; .tensors: name -> Tensor (raw bytes, never copied)."""

    def __init__(self, path):
        gguf = _gguf()
        self.path = path
        paths = [path]
        m = SPLIT_RE.search(path)
        if m:
            n = int(m.group(2))
            paths = [path[:m.start()] + f"-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise GGUFError(f"mashbox: {os.path.basename(path)} is part {m.group(1)} of {n}; missing: "
                                + ", ".join(os.path.basename(p) for p in missing))
        self.paths = paths
        self.readers = []
        self.meta, self.tensors, self._fields = {}, {}, {}
        for p in paths:
            try:
                r = gguf.GGUFReader(p, "r")
            except ValueError as e:
                m2 = re.search(r"(\d+) is not a valid GGMLQuantizationType", str(e))
                if m2:
                    raise GGUFError(f"mashbox: {os.path.basename(p)} uses ggml block type {m2.group(1)}, newer than the GGUF reader "
                                    f"in this version (gguf {reader_version()}). It runs in llama.cpp: {llama_cpp_command(p)}")
                raise GGUFError(f"mashbox: {os.path.basename(p)} is not a GGUF file this reader can open ({e})")
            except Exception as e:
                raise GGUFError(f"mashbox: {os.path.basename(p)} is not a GGUF file this reader can open ({e})")
            if r.byte_order != "I":
                raise GGUFError(f"mashbox: {os.path.basename(p)} is a big-endian GGUF; only little-endian files run here")
            self.readers.append(r)
            for k, f in r.fields.items():
                if k not in self._fields and not k.startswith("GGUF."):
                    self._fields[k] = f
            for t in r.tensors:
                shape = tuple(int(s) for s in reversed([int(x) for x in t.shape]))
                raw = r.data[t.data_offset:t.data_offset + t.n_bytes]
                self.tensors[t.name] = Tensor(t.name, int(t.tensor_type), shape, raw)
        self.size = sum(os.path.getsize(p) for p in paths)

    # metadata is decoded on first use (the tokenizer arrays are large)
    def get(self, key, default=None):
        if key in self.meta:
            return self.meta[key]
        f = self._fields.get(key)
        if f is None:
            return default
        v = f.contents()
        self.meta[key] = v
        return v

    def keys(self):
        return list(self._fields)

    @property
    def arch(self):
        return self.get("general.architecture")

    def census(self):
        """-> {type name: dict(tensors, bytes)} over every tensor, and the total bytes"""
        out, total = {}, 0
        for t in self.tensors.values():
            e = out.setdefault(t.type_name, dict(type=t.type, tensors=0, bytes=0))
            e["tensors"] += 1
            e["bytes"] += t.nbytes
            total += t.nbytes
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["bytes"])), total

    def unreadable(self):
        """tensors whose ggml type the pinned reader cannot dequantize"""
        return [t for t in self.tensors.values() if t.type not in R.TYPES]

    def close(self):
        for r in self.readers:
            mm = getattr(r, "data", None)
            if hasattr(mm, "_mmap") and mm._mmap is not None:
                try:
                    mm._mmap.close()
                except Exception:
                    pass
        self.readers = []


def llama_cpp_command(path=None, repo=None, quant=None):
    """the llama.cpp command that runs the same file (for every 'not supported here' message)"""
    if repo:
        return f"llama-server -hf {repo}" + (f":{quant}" if quant else "")
    return f"llama-server -m {path}"


def reader_version():
    try:
        from importlib.metadata import version
        return version("gguf")
    except Exception:
        return None


def known_archs():
    """architecture names the installed gguf package knows"""
    gguf = _gguf()
    return set(gguf.constants.MODEL_ARCH_NAMES.values())


def bytes_view(t):
    return np.asarray(t.raw)
