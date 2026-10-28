"""Reader for .bshj weight files (MASHBOX's TPU weight format).

Layout (little endian):
  0   4   magic b"BSHJ"
  4   2+2 major (1), minor
  8   8   json_len: bytes of the JSON header (space padded)
  16  8   data_offset (a multiple of align)
  24  4   align (>= 4096)
  28  4   flags (bit 0: tensors packed back to back in header order)
  32  8   data_len
  40  8   header hash (xxh64 of the header bytes, 0 = not set)
  48  16  reserved, zero
  64  ... JSON header: {"tensors": {name: {"dtype", "shape", "offset", "nbytes", "layout", ...}}, other metadata}
  data_offset ... tensors; each offset is relative to data_offset

A tensor is read as a NumPy view of the memory-mapped file: no copy, no conversion. The file's sha256 is checked once,
when it is downloaded (see mashbox.hub).
"""
import json
import mmap
import struct

import numpy as np

_FIXED = struct.Struct("<4sHHQQIIQQ16s")
MAX_MINOR = 1


def _dtypes():
    import ml_dtypes
    return {"bf16": np.dtype(ml_dtypes.bfloat16), "f16": np.dtype("<f2"), "f32": np.dtype("<f4"), "f64": np.dtype("<f8"),
            "i8": np.dtype("i1"), "u8": np.dtype("u1"), "i16": np.dtype("<i2"), "u16": np.dtype("<u2"),
            "i32": np.dtype("<i4"), "u32": np.dtype("<u4"), "i64": np.dtype("<i8"), "u64": np.dtype("<u8")}


class File:
    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        fixed = self._mm[:64]
        magic, major, minor, jl, self.data_offset, align, flags, data_len, _hh, reserved = _FIXED.unpack(fixed)
        if magic != b"BSHJ" or major != 1:
            raise ValueError(f"{path}: not a BSHJ 1.x file")
        if minor > MAX_MINOR:
            raise ValueError(f"{path}: BSHJ 1.{minor} is newer than this reader (1.{MAX_MINOR})")
        if any(reserved) or self.data_offset % align or self.data_offset < 64 + jl \
                or self.data_offset + data_len > len(self._mm):
            raise ValueError(f"{path}: bad container header")
        self.header = json.loads(bytes(self._mm[64:64 + jl]).rstrip(b" \t\n\r").decode("utf-8"))
        self.tensors = self.header["tensors"]
        self._dt = _dtypes()
        for name, t in self.tensors.items():
            n = int(np.prod(t["shape"])) * self._dt[t["dtype"]].itemsize
            if t["nbytes"] != n or t["offset"] % align or t["offset"] + n > data_len:
                raise ValueError(f"{path}: bad tensor record {name}")

    def meta(self):
        return {k: v for k, v in self.header.items() if k != "tensors"}

    def array(self, name):
        t = self.tensors[name]
        dt = self._dt[t["dtype"]]
        return np.frombuffer(self._mm, dtype=dt, count=t["nbytes"] // dt.itemsize,
                             offset=self.data_offset + t["offset"]).reshape(t["shape"])

    def advise(self, name):
        """Tell the OS a tensor is about to be read (read-ahead on a cold page cache)."""
        t = self.tensors[name]
        start = (self.data_offset + t["offset"]) & ~(mmap.PAGESIZE - 1)
        try:
            self._mm.madvise(mmap.MADV_WILLNEED, start, self.data_offset + t["offset"] + t["nbytes"] - start)
        except (AttributeError, OSError, ValueError):
            pass

    def close(self):
        try:
            self._mm.close()
        except BufferError:       # a caller still holds a view; the map closes with the process
            pass
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
