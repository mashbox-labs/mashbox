"""Load the released weights (safetensors) into the parameter tree the model uses. NumPy only, no pickle.

The file comes from the model's Hugging Face repo at the revision pinned in models.json (mashbox.hub), or from a local
path; in both cases its sha256 must equal the one in models.json before anything is read."""
import hashlib
import json
import os
import struct
import sys

import numpy as np

from .model import CONFIG

MODEL_ID = "track2rig"
N_TENSORS = 55
_DTYPES = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}


def entry():
    from ... import hub
    return hub.model_entry(MODEL_ID)


def weights_row():
    return next(f for f in entry()["files"] if f.get("kind") == "weights" and f.get("default"))


WEIGHTS_NAME = "track2rig-v2.safetensors"
WEIGHTS_SHA256 = "f8a79263dc21d2af078c123fbf76c2cba85df4694100f6a6c5e8f7608f6094b6"  # == models.json files[] (test checks)


def _log(s):
    print(s, file=sys.stderr, flush=True)


def weights_path(path=None, log=_log):
    """Local path of the checked weights file: `path` (sha256 checked), else the pinned download."""
    from ... import hub
    m = entry()
    if path:
        return hub.verify(path, weights_row()["sha256"], log)
    if m.get("hf_revision") is None:
        raise SystemExit("The Track2Rig weights are not on Hugging Face at a pinned revision in this version of mashbox. "
                         f"Pass a local copy of {WEIGHTS_NAME} with --weights FILE (Python: load(\"FILE\")).")
    return hub.weights_file(m, log=log)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors(path):
    """Return (tensors: dict name -> np.ndarray, metadata: dict). Plain format reader: 8-byte header length, JSON header,
    raw little-endian tensor bytes."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError(f"{path}: not a safetensors file")
    n = struct.unpack("<Q", raw[:8])[0]
    if n > len(raw) - 8:
        raise ValueError(f"{path}: header length {n} is past the end of the file")
    hdr = json.loads(raw[8:8 + n])
    meta = hdr.pop("__metadata__", {}) or {}
    base = 8 + n
    out = {}
    for name, e in hdr.items():
        if e["dtype"] not in _DTYPES:
            raise ValueError(f"{path}: tensor {name} has unsupported dtype {e['dtype']}")
        a, b = e["data_offsets"]
        arr = np.frombuffer(raw, dtype=_DTYPES[e["dtype"]], count=(b - a) // np.dtype(_DTYPES[e["dtype"]]).itemsize, offset=base + a)
        out[name] = arr.reshape(e["shape"]).copy()
    return out, meta


def unflatten(flat):
    """{'layers.0.ln1.g': arr, ...} -> nested dict; integer path parts become list positions."""
    tree = {}
    for name in sorted(flat):
        parts = name.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = flat[name]

    def fix(node):
        if not isinstance(node, dict):
            return node
        node = {k: fix(v) for k, v in node.items()}
        if node and all(k.isdigit() for k in node):
            return [node[str(i)] for i in range(len(node))]
        return node
    return fix(tree)


def load_params(path=None, log=_log):
    """Load the weights and return the parameter tree (nested dict of float32 numpy arrays). The file's sha256 is checked
    against models.json first (see weights_path)."""
    path = weights_path(path, log)
    flat, meta = read_safetensors(path)
    cfg = json.loads(meta.get("config", "{}"))
    for k, v in CONFIG.items():
        if cfg.get(k) != v:
            raise RuntimeError(f"weights config {k}={cfg.get(k)} does not match the model ({v})")
    if len(flat) != N_TENSORS:
        raise RuntimeError(f"weights hold {len(flat)} tensors; expected {N_TENSORS}")
    return unflatten(flat)
