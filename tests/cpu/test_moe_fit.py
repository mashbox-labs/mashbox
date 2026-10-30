"""The fit of a mixture of experts, from saved tensor lists (names, block types, shapes, file bytes and the file's
sha256 + Hub revision; no weights): tests/fixtures/moe/tensor-lists/. The one fit (gguf.need_bytes, check_fits) counts
every expert's planes, the K/V and linear-attention state, and the routed op's own scratch through mashbox.moe.need_bytes,
which asks the kernel package for its tiling (ops.expert_scratch_bytes).

A stub package stands in for the binary kernels. Its native planes take 1.04x the file's bytes: the most the native-plane
size gate (nb_bytes / file bytes <= 1.04 per type) lets the package take, so the fit below holds for any package that
passes that gate. No kq planes (dense weights stay file blocks on the exact path)."""
import json
import math
import os
import sys
import types

import pytest

from mashbox import gguf as G
from mashbox import moe, ops
from mashbox.gguf import archs
from mashbox.ops import reference as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TL = os.path.join(ROOT, "tests", "fixtures", "moe", "tensor-lists")
V6E = 33.546e9                     # gguf.CHIPS: one v6e chip's bytes_limit, measured
NB_BOUND = 1.04                    # the native-plane size gate's bound (nb_bytes / file bytes)
FILES = {"Qwen3.6-35B-A3B-UD-Q4_K_M": ("qwen35moe", 733, 22_123_538_944),
         "Qwen3-30B-A3B-Q4_K_M": ("qwen3moe", 579, 18_550_716_416)}


class _T:
    def __init__(self, d):
        self.name, self.type, self.shape, self.nbytes = d["name"], int(d["type"]), tuple(d["shape"]), int(d["nbytes"])
        self.raw = None

    @property
    def type_name(self):
        return ops.type_name(self.type)


class _ListFile:
    """a gguf.File's metadata and tensor table from a saved tensor list (no weights)"""

    def __init__(self, d):
        self.meta, self.path = dict(d["metadata"]), d["file"]
        self.tensors = {t["name"]: _T(t) for t in d["tensors"]}

    def get(self, key, default=None):
        return self.meta.get(key, default)

    def keys(self):
        return list(self.meta)

    @property
    def arch(self):
        return self.meta["general.architecture"]

    def census(self):
        out, total = {}, 0
        for t in self.tensors.values():
            e = out.setdefault(t.type_name, dict(type=t.type, tensors=0, bytes=0))
            e["tensors"] += 1
            e["bytes"] += t.nbytes
            total += t.nbytes
        return out, total

    def unreadable(self):
        return [t for t in self.tensors.values() if t.type not in R.TYPES]


def _load(name):
    with open(os.path.join(TL, name + ".tensors.json")) as fh:
        return json.load(fh)


def _file_bytes(t, shape):
    blk, size = R.TYPES[int(t)][1], R.TYPES[int(t)][2]
    return math.prod(int(v) for v in shape[:-1]) * (int(shape[-1]) // blk) * size


def _stub(monkeypatch, nb=(2, 8, 12, 13, 14, 39), **extra):
    """the stub kernel package: native planes for `nb` at the size gate's bound (NB_BOUND); extra attributes
    (expert_ffn, EXPERT_TYPES, expert_ffn_scratch) as a test needs them"""
    m = types.ModuleType("_moe_fit_stub")
    m.TYPES = {}
    m.nb_types = lambda: tuple(nb)
    m.nb_bytes = lambda t, shape: math.ceil(NB_BOUND * _file_bytes(t, shape))
    for k, v in extra.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, m.__name__, m)
    ops.load_kernels(module=m.__name__)
    return ops.kernel_types()


@pytest.fixture(autouse=True)
def _restore():
    yield
    ops.load_kernels(module="off")


def _cfg(f):
    return archs.module(f.arch).Config(f)


@pytest.mark.parametrize("name", sorted(FILES))
def test_tensor_list_records_the_file(name):
    """each list names its file's sha256 and Hub revision; the counts are the file's"""
    d = _load(name)
    arch, n, total = FILES[name]
    assert len(d["gguf_sha256"]) == 64 and len(d["hub_revision"]) == 40 and d["hub_repo"].startswith("unsloth/")
    assert d["metadata"]["general.architecture"] == arch and len(d["tensors"]) == n
    assert sum(t["nbytes"] for t in d["tensors"]) == total


@pytest.mark.parametrize("name", sorted(FILES))
def test_i6_fits_one_v6e_on_native_planes(name, monkeypatch):
    """weights on native planes <= 1.06x the file, and check_fits passes for v6e-1 (8 slots x 2048)"""
    f = _ListFile(_load(name))
    c = _cfg(f)
    assert c.moe and c.spec is not None
    kt = _stub(monkeypatch)
    w, st, big = G.need_bytes(f, c, 2048, 8, kt)
    _, _, total = FILES[name]
    assert w <= 1.06 * total, (w, total)
    monkeypatch.setattr(G, "_limit", lambda: V6E)
    need = G.check_fits(f, c, 2048, 8, kt)
    assert need == w + st + big and need <= V6E, (w, st, big)


def test_i6_twice_the_experts_is_refused_naming_the_chip_that_fits(monkeypatch):
    """a synthetic list with twice the experts of the 35B file (every expert stack and the router doubled) is
    refused on v6e-1, and the message names the chips it fits"""
    d = _load("Qwen3.6-35B-A3B-UD-Q4_K_M")
    a = d["metadata"]["general.architecture"]
    E = int(d["metadata"][a + ".expert_count"])
    d["metadata"][a + ".expert_count"] = 2 * E
    for t in d["tensors"]:
        if t["shape"][0] == E and ("_exps." in t["name"] or t["name"].endswith("ffn_gate_inp.weight")):
            t["shape"] = [2 * E] + t["shape"][1:]
            t["nbytes"] *= 2
    f = _ListFile(d)
    c = _cfg(f)
    assert c.spec.n_expert == 2 * E
    kt = _stub(monkeypatch)
    monkeypatch.setattr(G, "_limit", lambda: V6E)
    with pytest.raises(G.GGUFError) as e:
        G.check_fits(f, c, 2048, 8, kt)
    msg = str(e.value)
    assert "needs about" in msg and "It fits: TPU v5p (one chip)" in msg and "33.5 GB" in msg, msg


def _experts(f):
    return [(t.type, t.shape, t.nbytes) for t in f.tensors.values() if len(t.shape) > 2]


def test_moe_scratch_asks_the_kernel_package(monkeypatch):
    """the routed op's scratch on the kernel path is what the package reports for the largest call (tokens x K pairs),
    plus the pairs' activations and float32 outputs; the engine holds no tile table"""
    f = _ListFile(_load("Qwen3.6-35B-A3B-UD-Q4_K_M"))
    c = _cfg(f)
    ex = _experts(f)
    seen = []

    def scratch(P, D, F, E):
        seen.append((P, D, F, E))
        return 123_456_789
    kt = _stub(monkeypatch, expert_ffn=lambda *a, **k: None, EXPERT_TYPES=(12, 13, 14), expert_ffn_scratch=scratch)
    sp = c.spec
    P = 2048 * sp.n_used
    got = moe.need_bytes(sp, c.D, 2048, ex, 2, kt)
    assert seen == [(P, c.D, sp.expert_ff, sp.n_expert)]
    assert got == P * c.D * (2 + 4) + 123_456_789
    w, st, big = G.need_bytes(f, c, 2048, 8, kt)
    w0, st0, big0 = G.need_bytes(f, c, 2048, 8, kt, tokens=0)
    assert (w, st) == (w0, st0) and big - big0 == got - moe.need_bytes(sp, c.D, 0, ex, 2, kt)


def test_moe_scratch_fallback_without_the_package_rule(monkeypatch):
    """stated fallback: a package that runs expert_ffn but reports no scratch is counted at P rows, bf16 in + f32 out"""
    f = _ListFile(_load("Qwen3.6-35B-A3B-UD-Q4_K_M"))
    c = _cfg(f)
    kt = _stub(monkeypatch, expert_ffn=lambda *a, **k: None, EXPERT_TYPES=(12, 13, 14))
    sp = c.spec
    P = 2048 * sp.n_used
    assert moe.need_bytes(sp, c.D, 2048, _experts(f), 2, kt) == P * c.D * 6 + P * 6 * c.D


def test_moe_scratch_on_the_exact_path(monkeypatch):
    """no expert_ffn in the package (or a type it does not take): the exact path's gathered slots and rebuild"""
    f = _ListFile(_load("Qwen3.6-35B-A3B-UD-Q4_K_M"))
    c = _cfg(f)
    sp = c.spec
    P, E = 2048 * sp.n_used, sp.n_expert
    for extra in ({}, dict(expert_ffn=lambda *a, **k: None, EXPERT_TYPES=(12,))):   # 12 only: Q5_K / Q6_K stacks too
        kt = _stub(monkeypatch, **extra)
        one = max(ops.device_nbytes(t, s, n, kt, experts=True) // math.prod(s[:-2]) for t, s, n in _experts(f))
        want = P * c.D * 6 + min(P, E) * one + 4 * min(ops.EXACT_CHUNK, min(P, E) * sp.expert_ff * c.D)
        assert moe.need_bytes(sp, c.D, 2048, _experts(f), 2, kt) == want, extra


def test_dense_file_has_no_moe_term(monkeypatch):
    """a file without experts keeps the scratch it had: tokens changes nothing"""
    f = _ListFile(_load("Qwen3.6-35B-A3B-UD-Q4_K_M"))
    c = _cfg(f)
    c.moe = False
    kt = _stub(monkeypatch)
    assert G.need_bytes(f, c, 2048, 8, kt)[2] == G.need_bytes(f, c, 2048, 8, kt, tokens=0)[2]
