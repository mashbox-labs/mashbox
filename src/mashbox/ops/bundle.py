"""The kernel package read from a payload bundle: a mashbox_engine_kernels-*.whl holds jax.export payloads and a
manifest.json, and no code. `Bundle(path)` gives the surface `load_kernels` reads from a kernel package. The routed-expert
(MoE) half: EXPERT_KERNEL, EXPERT_ACTS, EXPERT_TYPES, expert_ffn, expert_ffn_scratch, nb_types, nb_bytes, repack_nb_host,
nb_dequant. The dense half (the kq payloads of 'payloads'): TYPES, kq_covers, plane_bytes, repack_host, repack, matmul,
dequant. A dense weight runs on kq when the manifest lists its exact (type, rows, cols) with a cpu repack, a dequant
and a matmul for EVERY row class on the running platform (an override), else, in a format-2 manifest (rule
"shape-class"), when a shape class of its type covers it (`Classes`: the default net under the overrides), else it takes
the exact path (kq_covers; ops.weight_path asks it; kq_route says which key won).

A format-1 manifest (no "format" key) is read as before: every call is an exact-shape lookup. A format-2 class is data
only: atoms over format and hardware constants (bs, tk, n_pad, tn, n_max, k_max) and its matmul's row classes; its
payloads are exported with symbolic rows / cols, the raw blocks are zero-padded to the class's (Np, Kp), x's columns to
Kp, and the outputs cut back. A shape neither key covers raises KeyError, and the callers' existing fallbacks put that
weight or call on the exact path with their warning. A format, rule, roles map or version this reader cannot vouch for
is refused (ValueError). Each payload's bytes are checked
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
import math
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
    """native block planes of one weight or stack: {name: uint32 array (lead + plane shape)}, ggml type t, (N, K), and
    the kq key they were packed for (route: "exact" or "class"; None = whatever kq_route says for (t, N, K))"""

    def __init__(self, planes, t, N, K, route=None):
        self.planes, self.t, self.N, self.K, self.route = dict(planes), int(t), int(N), int(K), route

    @property
    def lead(self):
        """the stacked dims, read from the arrays (a scan or vmap slice of a stack sees its own lead)"""
        _, a = next((k, a) for k, a in sorted(self.planes.items()) if k != "d6")
        return tuple(int(v) for v in a.shape[:-2])

    def tree_flatten(self):
        keys = tuple(sorted(self.planes))
        return tuple(self.planes[k] for k in keys), (keys, self.t, self.N, self.K, self.route)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        keys, t, N, K, route = aux
        return cls(dict(zip(keys, ch)), t, N, K, route)


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
    when no row lists them (on TPU, one log line). The file comes from the cache; a missing one is downloaded once from
    the row's url to a .part file and moved under its name only when its sha256 is the row's (else ValueError, and
    nothing is cached)."""
    from .. import kernels as K
    fam = K.chip_family()
    if fam is None:
        return None
    row = next((r for r in rows() if fam in r.get("chips", ()) and r.get("jax") == jax.__version__), None)
    if row is None:
        if log:
            log(f"mashbox: no engine_kernels row for {fam} and JAX {jax.__version__} in this version's catalog; every "
                f"quantized type runs on the exact path")
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


FORMATS = (1, 2)
ROLES_2 = {"kq_classes": "default", "payloads": "override", "moe_payloads": "exact-only"}


def _up(n, m):
    return -(-int(n) // int(m)) * int(m)


# The N tile pick (0.1.12). A grid step of a class matmul has a fixed cost on top of its plane reads. STEP_COLS[config]
# states it in plane columns at K tile 1280, measured once per matmul config on synthetic shapes only (N 10240 x K 5120,
# tk 1280, one x row, every menu tn dividing 10240, kq 0.1.7 payloads on one v6e chip; fit t = t_bytes + sigma x grid
# steps, STEP_COLS = sigma / (t_bytes per column and K tile)); never on a model's own shapes. STEP_ERR[config] is that
# fit's error: the largest |residual| / time over its points. Record: CORE-CHANGES 0.1.12. A bundle's matmul entry
# may state its own "step_cols" and "step_err" (measured the same way for its kernels); those win over these tables, so
# a new kernel or config brings its own step cost and the reader keeps no list of kernels.
# WIDE: the row classes whose entry states form "exact" may take a larger class tn of the config (tw) when the
# modelled cost Np' x (1 + c / tw), c = STEP_COLS[config] x 1280 / tk, Np' = Np rounded up to a multiple of both tw and
# the pick's tn, is at most WIDE_GAIN (half) of the pick's, and lower by more than STEP_ERR[config]: only a tile that the
# model says at least halves the cost, so a gain the linear model overstates (larger tiles hide steps behind their
# reads) never switches. The planes are not padded for it: every call of a weight reads the one copy the pick lays out
# (Np), and an exact-form payload whose entry states atoms "n_mult" (kq 0.1.9: 128) takes planes of any width that is a
# multiple of it, running its last N step partial. A tw whose exact-form entries state no n_mult dividing Np is never
# taken, so a bundle with no such entries keeps the 0.1.11 pick. The exact form computes each output column on its own,
# so its values do not depend on the tile; a "tile" form's can, so tile-form row classes keep the pick's tn. A config
# with no STEP_COLS, or WIDE False, keeps the 0.1.11 pick. One rule for every N: no per-shape table.
STEP_COLS = {"c2m16": 32053.0, "c4m": 10066.9, "c5m": 5780.1, "c6": 276.0, "c8g16": 479.8, "c8g32": 546.2,
             "l4g16": 42.6, "l4g32": 48.0}
STEP_ERR = {"c2m16": 0.1194, "c4m": 0.2581, "c5m": 0.2799, "c6": 0.0926, "c8g16": 0.0671, "c8g32": 0.0992,
            "l4g16": 0.0452, "l4g32": 0.0398}
WIDE = True
WIDE_GAIN = 0.5


class Classes:
    """a format-2 manifest's kq_classes as an index (no payload is read here). Entry: id, op (repack | dequant | matmul),
    type (repack / dequant) or config + types (matmul), atoms {bs (repack / dequant), tk, n_pad, n_max, k_max[, tn]},
    rows (matmul: the row classes its x takes), platform, args / outs, file, sha256, impl, kernel_version. A matmul
    class takes planes, so one config serves types of different block sizes and its atoms state none. A repack /
    dequant class whose bs is not its type's block size is never picked; two classes of one type that state different
    domains make the manifest unreadable.
    A matmul entry with "layer": true is the layer form of the plain class with the same keys: fn(x, layer, *planes)
    with one leading stack axis on every plane (at most l_max layers); it reads plane block `layer` in place, so a layer
    scan never slices (copies) the planes. It is indexed apart (mml) and never picked for a plain call."""

    def __init__(self, entries):
        self.entry, self.rep, self.deq, self.mm, self.dom, self.mml = {}, {}, {}, {}, {}, {}
        for e in entries or ():
            a, name = e["atoms"], str(e["id"])
            self.entry[name] = e
            types = [int(v) for v in e["types"]] if e["op"] == "matmul" else [int(e["type"])]
            for t in types:
                if t not in R.TYPES or (e["op"] != "matmul" and int(a["bs"]) != R.TYPES[t][1]):
                    continue
                dom = (R.TYPES[t][1], int(a["n_pad"]), int(a["n_max"]), int(a["k_max"]))
                if self.dom.setdefault(t, dom) != dom:
                    raise ValueError(f"kernel bundle kq_classes: type {t} states two domains {self.dom[t]} and {dom}")
                tk = int(a["tk"])
                if e["op"] == "repack":
                    self.rep[(t, tk, e["platform"])] = name
                elif e["op"] == "dequant":
                    self.deq[(t, tk, e["platform"])] = name
                elif e["op"] == "matmul":
                    for rc in e["rows"]:
                        (self.mml if e.get("layer") else self.mm)[(t, tk, int(a["tn"]), int(rc), e["platform"])] = name
        self._ladder = {}
        for (t, tk, tn, rc, p) in self.mm:
            self._ladder.setdefault((t, p), set()).add(rc)
        self._picks = {}

    def types(self, platform):
        return frozenset(t for t in self.dom if any(self._tns(t, tk, platform) for tk in self._tks(t, platform)))

    def rows(self, t, tk, tn, platform):
        return tuple(sorted(self._ladder.get((t, platform), ())))

    def _tks(self, t, platform):
        """K tiles with a cpu repack and a dequant on the platform, largest first"""
        return sorted({tk for (t_, tk, p) in self.rep if t_ == t and p == "cpu" and (t, tk, platform) in self.deq},
                      reverse=True)

    def _tns(self, t, tk, platform):
        """N tiles with a matmul for every row class of the type on the platform, largest first"""
        lad = self._ladder.get((t, platform), ())
        tns = {tn for (t_, tk_, tn, rc, p) in self.mm if (t_, tk_, p) == (t, tk, platform)}
        return sorted((tn for tn in tns if lad and all((t, tk, tn, rc, platform) in self.mm for rc in lad)), reverse=True)

    def config(self, t, tk, tn, platform):
        """the matmul config of the class (t, tk, tn) on the platform (every row class's entry states the same one)"""
        rc = min(self._ladder.get((int(t), platform), ()), default=None)
        name = self.mm.get((int(t), tk, tn, rc, platform))
        return None if name is None else self.entry[name].get("config")

    def atoms(self, t, tk, tn, rc, platform):
        """-> the matmul entry's atoms for (type, tk, tn, row class), {} when there is none"""
        name = self.mm.get((int(t), tk, tn, rc, platform))
        return {} if name is None else self.entry[name].get("atoms") or {}

    def form(self, t, tk, tn, rc, platform):
        """the form the matmul entry (t, tk, tn, row class rc) states ("exact", "tile"), or None"""
        name = self.mm.get((int(t), tk, tn, rc, platform))
        return None if name is None else self.entry[name].get("form")

    def tn_for(self, t, n, k, rc, platform):
        """the N tile row class rc of a (rows n, cols k) weight runs on: the wide tile for an exact-form row class when
        WIDE gave one, else the pick's tn"""
        if self.pick(t, n, k, platform) is None:
            return None
        tk, tn, Kp, Np, tw = self._picks[(int(t), int(n), int(k), platform)]
        return tw if self.form(t, tk, tn, rc, platform) == "exact" else tn

    def _widen(self, t, tk, tn, Np, n_max, platform):
        """-> (tw, Np): the exact-form row classes' N tile (see STEP_COLS), else (tn, Np); the planes stay Np wide"""
        cfg = self.config(t, tk, tn, platform)
        ex = [rc for rc in self.rows(t, tk, tn, platform) if self.form(t, tk, tn, rc, platform) == "exact"]
        if not ex:
            return tn, Np
        e = self.entry[self.mm[(int(t), tk, tn, ex[0], platform)]]
        c = e["step_cols"] if "step_cols" in e else (STEP_COLS or {}).get(cfg)
        err = e["step_err"] if "step_err" in e else (STEP_ERR or {}).get(cfg, 0.0)
        if not c:
            return tn, Np
        c = c * 1280 / tk
        now = Np * (1 + c / tn)
        best = (now, -tn, tn, Np)
        for m in self._tns(t, tk, platform):
            npm = _up(Np, math.lcm(m, tn))
            if npm <= n_max and all(self.form(t, tk, m, rc, platform) == "exact"
                                    and Np % int(self.atoms(t, tk, m, rc, platform).get("n_mult", m)) == 0 for rc in ex):
                best = min(best, (npm * (1 + c / m), -m, m, npm))
        if best[2] != tn and best[0] <= now * WIDE_GAIN and best[0] < now * (1 - float(err)):
            return best[2], Np
        return tn, Np

    def layer_ok(self, t, n, k, platform):
        """True when the class pick of this (type, rows, cols) has a layer form for every row class of its ladder"""
        got = self.pick(t, n, k, platform)
        if got is None:
            return False
        tk, tn = got[0], got[1]
        lad = self.rows(t, tk, tn, platform)
        return bool(lad) and all((int(t), tk, self.tn_for(t, n, k, rc, platform), rc, platform) in self.mml for rc in lad)

    def pick(self, t, n, k, platform):
        """-> (tk, tn, Kp, Np) of the class that covers a (rows n, cols k) weight of type t, or None: tk = the largest
        whole class tk that divides k, else the smallest whole one with k zero-padded to it (then the next); Np = n
        rounded up to n_pad; tn = the largest whole class tn that divides Np. Only a weight that has that pick is on a
        class. With WIDE, the exact-form row classes may run on a wider tile over the same Np-wide planes (tn_for,
        STEP_COLS); tn stays the tile-form row classes' tile"""
        key = (int(t), int(n), int(k), platform)
        if key not in self._picks:
            self._picks[key] = self._pick(*key)
        got = self._picks[key]
        return None if got is None else got[:4]

    def _pick(self, t, n, k, platform):
        if t not in self.dom:
            return None
        bs, n_pad, n_max, k_max = self.dom[t]
        if not (0 < n <= n_max and 0 < k <= k_max and k % bs == 0):
            return None
        Np, tks = _up(n, n_pad), self._tks(t, platform)
        for tk in [c for c in tks if k % c == 0] + sorted(c for c in tks if k % c):
            tn = next((c for c in self._tns(t, tk, platform) if Np % c == 0), None)
            if tn is not None:
                tw, Np = self._widen(t, tk, tn, Np, n_max, platform) if WIDE else (tn, Np)
                return tk, tn, _up(k, tk), Np, tw
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
        self._man, self._exp, self._cexp = man, {}, {}
        self.version = _wheel_version(self._z) or man.get("version")
        self.format = int(man.get("format", 1))
        if self.format not in FORMATS:
            raise ValueError(f"kernel bundle manifest format {self.format}: this reader reads formats {FORMATS}")
        if self.format == 2:
            if man.get("rule") != "shape-class" or man.get("roles") != ROLES_2:
                raise ValueError(f"kernel bundle format 2: rule {man.get('rule')!r} / roles {man.get('roles')} are not "
                                 f"'shape-class' / {ROLES_2}")
            if man.get("version") != self.version:
                raise ValueError(f"kernel bundle manifest version {man.get('version')} is not the wheel's {self.version}")
        self._cls = Classes(man.get("kq_classes") if self.format == 2 else ())
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
        self.TYPES = (frozenset(t for (op, t, n, k, *_rest) in self._kq if op == "repack" and self._exact(t, n, k))
                      | self._cls.types(_platform()))
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
    def _exact(self, t, n, k):
        """the manifest lists this exact (type, rows, cols) with a cpu repack, a dequant and a matmul for every row class
        on the running platform (an override)"""
        t, n, k, p = int(t), int(n), int(k), _platform()
        return (bool(self.ROW_CLASSES) and ("repack", t, n, k, "cpu") in self._kq and ("dequant", t, n, k, p) in self._kq
                and all(("matmul", t, n, k, p, rc) in self._kq for rc in self.ROW_CLASSES))

    def _pick(self, t, n, k):
        return self._cls.pick(int(t), int(n), int(k), _platform())

    def kq_route(self, t, shape):
        """-> "exact" (an override lists the shape), "class" (a shape class covers it) or None (the exact path)"""
        n, k = int(shape[-2]), int(shape[-1])
        if self._exact(t, n, k):
            return "exact"
        return "class" if self._pick(t, n, k) is not None else None

    @property
    def picks_layer(self):
        """True when this bundle has layer forms of its class matmuls (ops.kernel_picks_layer reads it): then a stack of
        one (rows, cols) weight per layer is multiplied at its layer in place (matmul's layer=), never sliced"""
        return bool(self._cls.mml)

    def _route(self, t, n, k, stacked):
        """the kq key a weight is packed for: a stack whose class has layer forms takes the class (an override has no
        layer form, so a layer scan would copy its planes at every layer); else kq_route"""
        if stacked and self._cls.layer_ok(int(t), int(n), int(k), _platform()):
            return "class"
        return self.kq_route(t, (n, k))

    def kq_covers(self, t, shape):
        """True when a dense weight of this (type, rows, cols) runs on kq: an exact entry or a shape class covers it"""
        return self.kq_route(t, shape) is not None

    def plane_bytes(self, t, shape):
        """device bytes of one (rows, cols) tensor's kq planes: an override's from the manifest, a class's from its cpu
        repack's output shapes at the padded (Np, Kp); else the file's own bytes (the exact path holds the blocks as
        stored). When the class under an override has layer forms, the larger of the two: a layer stack of this shape
        is packed for the class (_route), a lone tensor for the override, and the caller does not say which it is"""
        n, k = int(shape[-2]), int(shape[-1])
        route = self.kq_route(t, (n, k))
        if route == "exact" and (int(t), n, k) in self._kq_pb:
            over = self._kq_pb[(int(t), n, k)]
            return max(over, self._class_bytes(t, n, k)) if self._cls.layer_ok(int(t), n, k, _platform()) else over
        return self._class_bytes(t, n, k, fallback_only=route != "class")

    def _class_bytes(self, t, n, k, fallback_only=False):
        """device bytes of one (rows, cols) tensor's class planes (the file's own bytes when no class repack runs)"""
        _, bs, ts = R.TYPES[int(t)]
        if not fallback_only and self._pick(t, n, k) is not None:
            tk, tn, Kp, Np = self._pick(t, n, k)
            try:
                _, exp = self._cls_payload(self._cls.rep[(int(t), tk, "cpu")])
                outs = jax.eval_shape(exp.call, jax.ShapeDtypeStruct((Np, Kp // bs * ts), np.uint8))
                return sum(int(np.prod(o.shape)) * np.dtype(o.dtype).itemsize for o in jax.tree_util.tree_leaves(outs))
            except Exception:                   # a class the loader cannot repack loads on the exact path: its bytes
                pass
        return n * (k // bs) * ts

    def _cls_payload(self, name):
        if name not in self._cexp:
            e = self._cls.entry[name]
            b = self._z.read(f"{PKG}/{e['file']}")
            if hashlib.sha256(b).hexdigest() != e["sha256"]:
                raise ValueError(f"kernel bundle class payload {name}: sha256 does not match the manifest")
            self._cexp[name] = jax.export.deserialize(bytearray(b))
        return self._cls.entry[name], self._cexp[name]

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
        lead = raw.shape[:-2]
        route = self._route(t, n, k, bool(lead))
        e, exp, pad = self._repack_entry(t, n, k, "cpu", route)
        flat = raw.reshape((-1,) + raw.shape[-2:])
        if pad is not None:                     # a class: zero blocks to (Np, Kp); zero blocks are zero weights
            flat = np.pad(flat, ((0, 0), (0, pad[0] - n), (0, pad[1] - flat.shape[-1])))
        with jax.default_device(jax.devices("cpu")[0]):
            parts = [{a: np.asarray(v) for a, v in self._outs(e, exp.call(r)).items()} for r in flat]
        planes = {a: np.stack([q[a] for q in parts]).reshape(lead + parts[0][a].shape) for a in e["outs"]}
        return Planes(planes, t, n, k, route)

    def repack(self, t, raw, shape, stacked=False):
        """traced form of repack_host for one (rows, row_bytes) device array (the running platform's repack payload);
        stacked: the tensor is one layer of a stack (the caller maps this over the stack)"""
        n, k = int(shape[-2]), int(shape[-1])
        route = self._route(t, n, k, stacked)
        e, exp, pad = self._repack_entry(t, n, k, _platform(), route)
        if pad is not None:
            raw = jax.numpy.pad(raw, ((0, pad[0] - n), (0, pad[1] - raw.shape[-1])))
        return Planes({a: v for a, v in self._outs(e, exp.call(raw)).items()}, t, n, k, route)

    def _repack_entry(self, t, n, k, platform, route=None):
        """-> (entry, payload, None) for an override, (entry, payload, (Np, padded row bytes)) for a class; KeyError
        when neither covers the shape"""
        route = self.kq_route(t, (n, k)) if route is None else route
        if route == "exact":
            return self._entry(("repack", int(t), n, k, platform)) + (None,)
        if route == "class":
            tk, tn, Kp, Np = self._pick(t, n, k)
            _, bs, ts = R.TYPES[int(t)]
            return self._cls_payload(self._cls.rep[(int(t), tk, platform)]) + ((Np, Kp // bs * ts),)
        raise KeyError(f"kernel bundle has no kq payloads for {R.TYPES[int(t)][0]} ({n}, {k})")

    def matmul(self, x, p, layer=None):
        """x (..., cols) @ W.T -> float32 (..., rows) through the matmul payload of each chunk's row class: the rows of x
        cast to bfloat16, cut into chunks of at most the largest row class, each zero-padded to its row class and cut
        back after the call. layer (an int32 scalar, traced or not): p is a stack of one weight per layer packed for its
        class, and the class's layer form multiplies by layer `layer` in place (KeyError when there is none)"""
        n, k, t, plat = p.N, p.K, p.t, _platform()
        lead = tuple(x.shape[:-1])
        m = 1
        for v in lead:
            m *= int(v)
        if m == 0:                                  # no rows: no payload call (its row classes start at one row)
            return jax.numpy.zeros(lead + (n,), jax.numpy.float32)
        route = p.route or self.kq_route(t, (n, k))
        if route is None or (layer is not None and (route != "class" or not self._cls.layer_ok(t, n, k, plat))):
            raise KeyError(f"kernel bundle has no kq payloads for {R.TYPES[int(t)][0]} ({n}, {k})"
                           + (" with layer=" if layer is not None else ""))
        x2 = x.reshape(m, k).astype(jax.numpy.bfloat16)
        if route == "class":                        # x's columns zero-padded to Kp (the planes' padded K)
            tk, tn, Kp, Np = self._pick(t, n, k)
            ladder = self._cls.rows(t, tk, tn, plat)
            x2 = jax.numpy.pad(x2, ((0, 0), (0, Kp - k))) if Kp > k else x2
        else:
            ladder = self.ROW_CLASSES
        top, ys = ladder[-1], []
        for a in range(0, m, top):
            xm = x2[a:a + top]
            rc = next(r for r in ladder if r >= xm.shape[0])
            xp = jax.numpy.pad(xm, ((0, rc - xm.shape[0]), (0, 0)))
            tr = self._cls.tn_for(t, n, k, rc, plat) if route == "class" else None
            if layer is not None:
                e, exp = self._cls_payload(self._cls.mml[(t, tk, tr, rc, plat)])
                args = [p.planes[name] for name in e["args"][2:]]
                ys.append(exp.call(xp, jax.numpy.asarray(layer, jax.numpy.int32), *args)[:xm.shape[0], :n])
                continue
            e, exp = (self._entry(("matmul", t, n, k, plat, rc)) if route == "exact"
                      else self._cls_payload(self._cls.mm[(t, tk, tr, rc, plat)]))
            args = [p.planes[name] for name in e["args"][1:]]
            ys.append(exp.call(xp, *args)[:xm.shape[0], :n])
        y = ys[0] if len(ys) == 1 else jax.numpy.concatenate(ys, axis=0)
        return y.reshape(lead + (n,))

    def dequant(self, p):
        """kq planes -> float32 (lead..., rows, cols), through the running platform's dequant payload"""
        route = p.route or self.kq_route(p.t, (p.N, p.K))
        if route == "class":                        # the class payload gives the padded (Np, Kp): cut to (N, K)
            tk = self._pick(p.t, p.N, p.K)[0]
            e, exp = self._cls_payload(self._cls.deq[(p.t, tk, _platform())])
        else:
            e, exp = self._entry(("dequant", p.t, p.N, p.K, _platform()))
        call = lambda *a: exp.call(*a)[:p.N, :p.K]
        args = [p.planes[name] for name in e["args"]]
        lead = tuple(args[0].shape[:-2])
        if not lead:
            return call(*args)
        flat = [a.reshape((-1,) + a.shape[len(lead):]) for a in args]
        out = jax.numpy.stack([call(*[a[i] for a in flat]) for i in range(flat[0].shape[0])])
        return out.reshape(lead + out.shape[1:])
