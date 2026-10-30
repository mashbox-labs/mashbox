"""The kernel package read from a payload bundle: a mashbox_engine_kernels-*.whl holds jax.export payloads and a
manifest.json, and no code. `Bundle(path)` gives the surface `load_kernels` reads from a kernel package. The routed-expert
(MoE) half: EXPERT_KERNEL, EXPERT_ACTS, EXPERT_TYPES, expert_ffn, expert_ffn_scratch, nb_types, nb_bytes, repack_nb_host,
nb_dequant. The dense half (the kq payloads of 'payloads'): TYPES, kq_covers, plane_bytes, repack_host, repack, matmul,
dequant. A dense weight runs on kq only when the manifest lists its exact (type, rows, cols) with a cpu repack, a dequant
and a matmul for EVERY row class on the running platform (kq_covers; ops.weight_path asks it), else the exact path.

Every call is an exact-shape lookup in the manifest. A shape the manifest does not list raises KeyError, and the callers'
existing fallbacks put that weight or call on the exact path with their warning. Each payload's bytes are checked
against the manifest's sha256 before it is deserialized. The bundle is refused (ValueError) when its JAX version is not
the running one, or when the running chip family is not one of its chips (off TPU, cpu payloads may still run).

Which wheel: the catalog's engine_kernels[] row (models.json) for this chip family and JAX. With no flag, `catalog_wheel`
gives that row's file from the cache (MASHBOX_CACHE or ~/.cache/mashbox, then kernels/<sha16>/), downloaded once from the
row's url and checked against the row's sha256. Any wheel, a path in MASHBOX_GGUF_KERNELS included, is refused
(ValueError) unless its own sha256 is a row's sha256 and the setup is one that row lists (`check_row`): the running JAX,
and on TPU the chip family in its chips and the installed libtpu matching its libtpu pattern."""
import fnmatch
import hashlib
import json
import os
import urllib.request
import zipfile

import jax
import numpy as np

from . import reference as R

from . import KERNEL_MODULE as PKG    # the one kernels package name (ops/__init__.py owns it)
from ..moe.record import KERNEL_ID as EXPERT_ID   # the routed-expert kernel's id (mashbox.moe.record owns it)
NAMES = {2: "Q4_0", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}


@jax.tree_util.register_pytree_node_class
class Planes:
    """native block planes of one weight or stack: {name: uint32 array (lead + plane shape)}, ggml type t, (N, K)"""

    def __init__(self, planes, t, N, K):
        self.planes, self.t, self.N, self.K = dict(planes), int(t), int(N), int(K)

    @property
    def lead(self):
        """the stacked dims, read from the arrays (a scan or vmap slice of a stack sees its own lead)"""
        _, a = next((k, a) for k, a in sorted(self.planes.items()) if k != "d6")
        return tuple(int(v) for v in a.shape[:-2])

    def tree_flatten(self):
        keys = tuple(sorted(self.planes))
        return tuple(self.planes[k] for k in keys), (keys, self.t, self.N, self.K)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        keys, t, N, K = aux
        return cls(dict(zip(keys, ch)), t, N, K)


def _platform():
    return jax.default_backend()


def rows():
    """the catalog's engine_kernels[] rows for GGUF (models.json)"""
    from .. import catalog
    return [r for r in catalog.registry().get("engine_kernels", []) if r.get("for", "gguf") == "gguf"]


def cache_root():
    return os.path.join(os.environ.get("MASHBOX_CACHE", os.path.expanduser("~/.cache/mashbox")), "kernels")


def cache_path(row):
    """-> <cache>/kernels/<sha16>/<file>, absolute, for an engine_kernels row; ValueError unless the row's sha256 is 64 hex
    characters and its file a bare file name, and the path and its .part both sit directly in that digest directory"""
    sha, f = str(row.get("sha256") or ""), str(row.get("file") or "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"kernel bundle row sha256 {sha[:20]!r} is not 64 lowercase hex characters")
    if f in ("", ".", "..") or "/" in f or "\\" in f or os.path.isabs(f) or f != os.path.basename(f):
        raise ValueError(f"kernel bundle row file {f!r} is not a bare file name")
    d = os.path.join(os.path.abspath(cache_root()), sha[:16])
    path = os.path.join(d, f)
    for q in (path, path + ".part"):
        q = os.path.abspath(q)
        if os.path.dirname(q) != d or os.path.commonpath([d, q]) != d:
            raise ValueError(f"kernel bundle row file {f!r} is not a bare file name (it leaves {d})")
    return path


def check_row(row, sha256):
    """ValueError unless the wheel with this sha256 is the row's file and the running setup is one the row lists: the
    same JAX, and on TPU the chip family in the row's chips and the installed libtpu matching the row's libtpu pattern
    (a row without a pattern matches no libtpu)"""
    from .. import kernels as K
    if row is None or row.get("sha256") != sha256:
        raise ValueError(f"kernel bundle sha256 {sha256[:12]} is not an engine_kernels row of this catalog")
    ver, fam = K.versions(), K.chip_family()
    if ver["jax"] != row.get("jax"):
        raise ValueError(f"kernel bundle row is for jax {row.get('jax')}, this is jax {ver['jax']}")
    if fam is not None:
        if fam not in row.get("chips", ()):
            raise ValueError(f"kernel bundle row is for {row.get('chips')}, this chip is {fam}")
        if not (ver.get("libtpu") and fnmatch.fnmatchcase(ver["libtpu"], row.get("libtpu") or "")):
            raise ValueError(f"kernel bundle row needs libtpu {row.get('libtpu')}, this is libtpu {ver.get('libtpu')}")


def catalog_wheel(log=None):
    """zero flags: -> the local path of the engine_kernels row's wheel for this chip family and JAX, or None off TPU or
    when no row lists them. The file comes from the cache; a missing one is downloaded once from the row's url to a .part
    file and moved under its name only when its sha256 is the row's (else ValueError, and nothing is cached)."""
    from .. import kernels as K
    fam = K.chip_family()
    if fam is None:
        return None
    row = next((r for r in rows() if fam in r.get("chips", ()) and r.get("jax") == jax.__version__), None)
    if row is None:
        return None
    path = cache_path(row)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h = hashlib.sha256()
        with urllib.request.urlopen(row["url"]) as src, open(path + ".part", "wb") as f:
            for b in iter(lambda: src.read(1 << 20), b""):
                h.update(b)
                f.write(b)
        if h.hexdigest() != row["sha256"]:
            raise ValueError(f"kernel bundle download {row['file']}: sha256 {h.hexdigest()[:12]} is not the row's "
                             f"{row['sha256'][:12]}")
        os.replace(path + ".part", path)
        if log:
            log(f"mashbox: kernel bundle {row['file']} fetched for {fam} (sha256 {row['sha256'][:12]} = the catalog row)")
    return path


def _wheel_version(z):
    """the Version: line of the wheel's dist-info METADATA, or None"""
    meta = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
    for line in (z.read(meta).decode("utf-8", "replace").splitlines() if meta else ()):
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


class Bundle:
    def __init__(self, path, row=None):
        """row: the catalog's engine_kernels row this wheel must be (None: the row whose sha256 is the wheel's)"""
        from .. import kernels as K
        self.path = str(path)
        with open(self.path, "rb") as f:
            data = f.read()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.row = row if row is not None else next((r for r in rows() if r.get("sha256") == self.sha256), None)
        check_row(self.row, self.sha256)
        self._z = zipfile.ZipFile(self.path)    # held open for the process: payloads are read from it on first use
        man = json.loads(self._z.read(f"{PKG}/manifest.json"))
        if man.get("jax") != jax.__version__:
            raise ValueError(f"kernel bundle is for jax {man.get('jax')}, this is jax {jax.__version__}")
        fam = K.chip_family()
        if fam is not None and fam not in man.get("chips", ()):
            raise ValueError(f"kernel bundle is for {man.get('chips')}, this chip is {fam}")
        self._man, self._exp = man, {}
        self.version = _wheel_version(self._z) or man.get("version")
        self._ffn, self._rep, self._deq = {}, {}, {}
        for name, e in man.get("moe_payloads", {}).items():
            if e["op"] == "expert_ffn":
                ty = e["types"]
                key = (e["act"], e["E"], e["D"], e["F"], e["P"], e["G"], ty.get("gate", 0), ty["up"], ty["down"], e["platform"])
                self._ffn[key] = name
            else:
                key = (e["type"], tuple(e["lead"]), e["rows"], e["cols"], e["platform"])
                (self._rep if e["op"] == "repack_nb" else self._deq)[key] = name
        acts = sorted({k[0] for k in self._ffn})
        self._kq, self._kq_pb = {}, {}           # the dense half: (op, type, rows, cols, platform[, row class]) -> name
        for name, e in man.get("payloads", {}).items():
            key = (e["op"], int(e["type"]), int(e["rows"]), int(e["cols"]), e["platform"])
            self._kq[key + ((int(e["row_class"]),) if e["op"] == "matmul" else ())] = name
        self.ROW_CLASSES = tuple(sorted(int(r) for r in man.get("row_classes", ())))
        ids = {str(k): int(v) for k, v in man.get("type_ids", {}).items()}
        for k, v in man.get("plane_bytes", {}).items():
            tn, rows_, cols_ = k.split(":")
            if tn in ids:
                self._kq_pb[(ids[tn], int(rows_), int(cols_))] = int(v)
        self.TYPES = frozenset(t for (op, t, n, k, *_rest) in self._kq if op == "repack" and self.kq_covers(t, (n, k)))
        self.EXPERT_TYPES = frozenset(int(t) for t in man.get("expert_types", ()))
        self.EXPERT_ACTS = tuple(acts)
        impl = man.get("moe_impl", {})
        self.EXPERT_KERNEL = {"id": EXPERT_ID, "version": self.version, "impl": impl.get(EXPERT_ID),
                              "package_sha256": self.sha256, "dev_build": False, "form": "exported"}

    def _payload(self, name):
        if name not in self._exp:
            e = self._man["moe_payloads"][name]
            b = self._z.read(f"{PKG}/{e['file']}")
            if hashlib.sha256(b).hexdigest() != e["sha256"]:
                raise ValueError(f"kernel bundle payload {name}: sha256 does not match the manifest")
            self._exp[name] = jax.export.deserialize(bytearray(b))
        return self._exp[name]

    def nb_types(self):
        return [int(t) for t in self._man.get("nb_types", ())]

    def nb_bytes(self, t, shape):
        """only the types in NAMES (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) have native planes; another type raises KeyError"""
        *lead, N, K = (int(v) for v in shape)
        return int(self._man["nb_bytes"][f"{NAMES[int(t)]}:{':'.join(map(str, lead))}:{N}:{K}"])

    def repack_nb_host(self, raw, t, shape):
        raw = np.asarray(raw)
        N, K = (int(v) for v in shape[-2:])
        name = self._rep[(int(t), tuple(int(v) for v in raw.shape[:-2]), N, K, "cpu")]
        exp = self._payload(name)
        with jax.default_device(jax.devices("cpu")[0]):
            out = exp.call(raw)
        return Planes({k: np.asarray(v) for k, v in out.items()}, t, N, K)

    def nb_dequant(self, p):
        name = self._deq[(p.t, p.lead, p.N, p.K, _platform())]
        return self._payload(name).call(*[p.planes[k] for k in sorted(p.planes)])

    def _ffn_name(self, P, D, F, E, act, G, types):
        return self._ffn[(act, int(E), int(D), int(F), int(P), int(G), types.get("gate", 0), types["up"], types["down"], _platform())]

    def expert_ffn(self, x_sorted, e_sorted, w_sorted, W, act="silu", n_expert=None, layer=None):
        projs = [k for k in ("gate", "up", "down") if k in W]
        up = W["up"]
        E = int(n_expert if n_expert is not None else up.lead[-1])
        G = 0 if layer is None else int(up.lead[0])
        name = self._ffn_name(x_sorted.shape[0], x_sorted.shape[1], up.N, E, act, G, {k: W[k].t for k in projs})
        args = [x_sorted, e_sorted, w_sorted] + [W[k].planes[n] for k in projs for n in sorted(W[k].planes)]
        if layer is not None:
            args.append(jax.numpy.asarray(layer, jax.numpy.int32))
        return self._payload(name).call(*args)

    def expert_ffn_scratch(self, P, D, F, E):
        """the listed call's scratch bytes (manifest), else the smallest listed P' >= P of the same (D, F, E), else the
        least a kernel that reads every pair and writes its output holds (P x 6D, the caller's own stated fallback)"""
        best = None
        for e in self._man.get("moe_payloads", {}).values():
            if (e["op"] == "expert_ffn" and (e["D"], e["F"], e["E"]) == (int(D), int(F), int(E)) and e["P"] >= int(P)
                    and (best is None or e["P"] < best[0])):
                best = (e["P"], int(e["scratch_bytes"]))
        return best[1] if best else int(P) * 6 * int(D)

    # ---------------------------------------------------------------------------------------- the dense half (kq)
    def kq_covers(self, t, shape):
        """True when the manifest lists this exact (type, rows, cols) with a cpu repack, a dequant and a matmul for every
        row class on the running platform: only then does a dense weight of that shape run on kq"""
        n, k, t, p = int(shape[-2]), int(shape[-1]), int(t), _platform()
        return (bool(self.ROW_CLASSES) and ("repack", t, n, k, "cpu") in self._kq and ("dequant", t, n, k, p) in self._kq
                and all(("matmul", t, n, k, p, rc) in self._kq for rc in self.ROW_CLASSES))

    def plane_bytes(self, t, shape):
        """device bytes of one (rows, cols) tensor's kq planes (manifest), else the file's own bytes (the exact path
        holds the blocks as stored)"""
        n, k = int(shape[-2]), int(shape[-1])
        if self.kq_covers(t, (n, k)) and (int(t), n, k) in self._kq_pb:
            return self._kq_pb[(int(t), n, k)]
        _, bs, ts = R.TYPES[int(t)]
        return n * (k // bs) * ts

    def _entry(self, key):
        name = self._kq[key]                     # KeyError: a shape the manifest does not list (callers fall back)
        return self._man["payloads"][name], self._kq_payload(name)

    def _kq_payload(self, name):
        if name not in self._exp:
            e = self._man["payloads"][name]
            b = self._z.read(f"{PKG}/{e['file']}")
            if hashlib.sha256(b).hexdigest() != e["sha256"]:
                raise ValueError(f"kernel bundle payload {name}: sha256 does not match the manifest")
            self._exp[name] = jax.export.deserialize(bytearray(b))
        return self._exp[name]

    @staticmethod
    def _outs(e, out):
        return dict(out) if isinstance(out, dict) else dict(zip(e["outs"], out if isinstance(out, (list, tuple)) else (out,)))

    def repack_host(self, t, raw, shape):
        """the file's blocks (lead..., rows, row_bytes) as a NumPy array -> kq planes, on the host's XLA:CPU through the
        cpu repack payload, one tensor of a stack at a time (the payload takes one (rows, row_bytes) tensor)"""
        raw = np.asarray(raw)
        n, k = int(shape[-2]), int(shape[-1])
        if not self.kq_covers(t, (n, k)):
            raise KeyError(f"kernel bundle has no kq payloads for {R.TYPES[int(t)][0]} ({n}, {k})")
        e, exp = self._entry(("repack", int(t), n, k, "cpu"))
        lead = raw.shape[:-2]
        flat = raw.reshape((-1,) + raw.shape[-2:])
        with jax.default_device(jax.devices("cpu")[0]):
            parts = [{a: np.asarray(v) for a, v in self._outs(e, exp.call(r)).items()} for r in flat]
        planes = {a: np.stack([q[a] for q in parts]).reshape(lead + parts[0][a].shape) for a in e["outs"]}
        return Planes(planes, t, n, k)

    def repack(self, t, raw, shape):
        """traced form of repack_host for one (rows, row_bytes) device array (the running platform's repack payload)"""
        n, k = int(shape[-2]), int(shape[-1])
        if not self.kq_covers(t, (n, k)):
            raise KeyError(f"kernel bundle has no kq payloads for {R.TYPES[int(t)][0]} ({n}, {k})")
        e, exp = self._entry(("repack", int(t), n, k, _platform()))
        return Planes({a: v for a, v in self._outs(e, exp.call(raw)).items()}, t, n, k)

    def _row_class(self, m):
        return next(r for r in self.ROW_CLASSES if r >= m)

    def matmul(self, x, p):
        """x (..., cols) @ W.T -> float32 (..., rows) through the matmul payload of each chunk's row class: the rows of x
        cast to bfloat16, cut into chunks of at most the largest row class, each zero-padded to its row class and cut
        back after the call"""
        n, k, t = p.N, p.K, p.t
        lead = tuple(x.shape[:-1])
        m = 1
        for v in lead:
            m *= int(v)
        if m == 0:                                  # no rows: no payload call (its row classes start at one row)
            return jax.numpy.zeros(lead + (n,), jax.numpy.float32)
        x2 = x.reshape(m, k).astype(jax.numpy.bfloat16)
        top, ys = self.ROW_CLASSES[-1], []
        for a in range(0, m, top):
            xm = x2[a:a + top]
            rc = self._row_class(xm.shape[0])
            e, exp = self._entry(("matmul", t, n, k, _platform(), rc))
            args = [p.planes[name] for name in e["args"][1:]]
            ys.append(exp.call(jax.numpy.pad(xm, ((0, rc - xm.shape[0]), (0, 0))), *args)[:xm.shape[0]])
        y = ys[0] if len(ys) == 1 else jax.numpy.concatenate(ys, axis=0)
        return y.reshape(lead + (n,))

    def dequant(self, p):
        """kq planes -> float32 (lead..., rows, cols), through the running platform's dequant payload"""
        e, exp = self._entry(("dequant", p.t, p.N, p.K, _platform()))
        args = [p.planes[name] for name in e["args"]]
        lead = tuple(args[0].shape[:-2])
        if not lead:
            return exp.call(*args)
        flat = [a.reshape((-1,) + a.shape[len(lead):]) for a in args]
        out = jax.numpy.stack([exp.call(*[a[i] for a in flat]) for i in range(flat[0].shape[0])])
        return out.reshape(lead + out.shape[1:])
