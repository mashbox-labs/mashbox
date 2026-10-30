"""Bundle format 2 (mashbox/ops/bundle.py): shape-class keys for the dense half. A format-2 manifest (rule "shape-class")
carries kq_classes, the default net, under the exact entries of 'payloads', which stay as speed overrides. Each class is
data only: atoms over format and hardware constants (block size bs, K tile tk, N pad, N tile tn, the domain bounds
n_max / k_max) plus the row classes its matmul takes. The reader picks a class by the atoms and never knows the recipe:
tk = the largest class tk that divides K, else the smallest with K zero-padded to it; Np = N rounded up to n_pad; tn =
the largest class tn that divides Np and has a matmul for every row class (from 0.1.12, when the matmul's config has a
measured step cost, B.STEP_COLS, the least-cost class tn with Np rounded up to it: the WIDE tests at the end; the
routing tests pin B.WIDE off). The raw blocks are zero-padded to (Np, Kp),
x's columns to Kp, and the outputs are cut back to [:M, :N] (matmul) and [:N, :K] (dequant).

These tests build small bundles on CPU from plain-JAX stand-ins exported with SYMBOLIC shapes for the cpu platform (the
real bundle carries TPU kernel payloads; nothing here measures a kernel). They check the routing only, and they are red
on the format-1 reader, which reads no kq_classes: a shape no exact entry lists stays on the exact path there.

Out of domain: cols not a multiple of the block, cols over k_max, rows over n_max, a type with no class,
a class without every row class, a payload whose bytes are not the manifest's sha256, a class payload called outside its
own symbolic constraints: each stays on the exact path (or falls back with the loader's warning). A format-1 manifest
keeps loading as before, and a format the reader does not know is refused."""
import hashlib
import json
import math
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mashbox import ops
from mashbox.ops import bundle as B
from mashbox.ops import reference as R

PKG = "mashbox_engine_kernels"
VER = "0.0.9"
Q8_0, Q4_K = 8, 12
BS, TS = 32, 34                      # Q8_0 block: 32 values, 34 bytes
TKS = (128, 64)                      # class K tiles (the real menu is 1280 / 1024 / 512)
NPAD, TNS = 16, (16, 32)             # N pad and N tiles (the real ones are 128 and multiples of 128)
ROWS = (4, 8, 16, 24)                # row classes; 16 and 24 are one payload symbolic in rows (8 * c, c = 2..3)
NMAX, KMAX = 64 * 16, 512            # the domain bounds the atoms state
EXACT = (32, 64)                     # the one shape with exact (override) entries in the override bundle
ROWS_EXACT = (16, 32)
CAT = []
f32, bf16 = jnp.float32, jnp.bfloat16
spec = jax.ShapeDtypeStruct


def _q8(raw):
    """stand-in class repack: raw (Np, Kp / 32 * 34) uint8 -> v (Kp, Np) = w.T / 2, s (1, Np) = 2, in 'outs' order"""
    np_, rb = raw.shape
    b = raw.reshape(np_, rb // TS, TS)
    u = b[:, :, 0].astype(jnp.uint16) | (b[:, :, 1].astype(jnp.uint16) << 8)     # the reference's float16 decode
    d = jax.lax.bitcast_convert_type(u, jnp.float16).astype(f32)
    q = jax.lax.bitcast_convert_type(b[:, :, 2:], jnp.int8).astype(f32)
    w = (q * d[:, :, None]).reshape(np_, rb // TS * BS)
    return 0.5 * w.T, jnp.full((1, np_), 2.0, f32)


def _deq(v, s):
    return (v * s).T


def _mm(x, v, s):
    return jnp.dot(x.astype(f32), v * s, precision=jax.lax.Precision.HIGHEST)


def _export(fn, *avals):
    return jax.export.export(jax.jit(fn), platforms=("cpu",))(*avals)


def _sym(expr, constraints=()):
    return jax.export.symbolic_shape(expr, constraints=constraints)


def _mm_layer(x, layer, v, s):
    """stand-in layer form: the plain matmul on block `layer` of stacked planes"""
    return _mm(x, v[layer], s[layer])


def _classes(drop=None, n_max=NMAX, k_max=KMAX, a_max=None, bad_sha=None, t=Q8_0, layer=False, drop_layer=None, tns=TNS,
             nmult=True):
    """-> [(exported, entry)]: repack + dequant per tk, matmul per (tk, tn) and row class; drop = {(tk, tn, rc)} left out;
    a_max: the repack's own symbolic constraint on Np / NPAD (to call it outside its constraints); layer: each matmul
    also gets its layer form (id + "_layer"), except the (tk, tn, rc) in drop_layer; nmult: the exact-form ones take
    planes of any width NPAD a and state atoms n_mult NPAD (kq 0.1.9), else tn a as kq 0.1.7"""
    out = []
    atoms = {"bs": BS, "n_pad": NPAD, "n_max": n_max, "k_max": k_max}
    lab = {"impl": "0" * 64, "kernel_version": "0.0.1"}
    for tk in TKS:
        a, b = _sym("a, b", [f"a <= {a_max}"] if a_max else [])
        rep = _export(_q8, spec((NPAD * a, tk // BS * TS * b), jnp.uint8))
        out.append((rep, dict(lab, id=f"rep.q8.tk{tk}", op="repack", type=t, atoms=dict(atoms, tk=tk), platform="cpu",
                              args=["raw"], outs=["v", "s"])))
        a, b = _sym("a, b")
        deq = _export(_deq, spec((tk * b, NPAD * a), f32), spec((1, NPAD * a), f32))
        out.append((deq, dict(lab, id=f"deq.q8.tk{tk}", op="dequant", type=t, atoms=dict(atoms, tk=tk), platform="cpu",
                              args=["v", "s"])))
        for tn in tns:
            for rows in ((4,), (8,), (16, 24)):
                if any((tk, tn, r) in (drop or ()) for r in rows):
                    continue
                if len(rows) > 1:                    # one payload for rows 16 and 24: x has 8 * c rows
                    a, b, c = _sym("a, b, c", ["c >= 2", "c <= 3"])
                    xr = 8 * c
                else:
                    (a, b), xr = _sym("a, b"), rows[0]
                w = NPAD if (nmult and rows[0] <= 8) else tn
                at = dict(atoms, tk=tk, tn=tn, **({"n_mult": NPAD} if w == NPAD and rows[0] <= 8 else {}))
                mm = _export(_mm, spec((xr, tk * b), bf16), spec((tk * b, w * a), f32), spec((1, w * a), f32))
                out.append((mm, dict(lab, id=f"mm.c8.tk{tk}.tn{tn}.r{rows[0]}", op="matmul", config="C8", types=[t],
                                     atoms=at, rows=list(rows), platform="cpu",
                                     form="exact" if rows[0] <= 8 else "tile", args=["x", "v", "s"])))
                if layer and not any((tk, tn, r) in (drop_layer or ()) for r in rows):
                    if len(rows) > 1:                # one symbolic scope per export: g with the plain symbols
                        a, b, c, g = _sym("a, b, c, g", ["c >= 2", "c <= 3", "g <= 256"])
                        xr = 8 * c
                    else:
                        (a, b, g), xr = _sym("a, b, g", ["g <= 256"]), rows[0]
                    ml = _export(_mm_layer, spec((xr, tk * b), bf16), spec((), jnp.int32), spec((g, tk * b, w * a), f32),
                                 spec((g, 1, w * a), f32))
                    out.append((ml, dict(lab, id=f"mm.c8.tk{tk}.tn{tn}.r{rows[0]}_layer", op="matmul", config="C8",
                                         types=[t], atoms=at, rows=list(rows), platform="cpu",
                                         form="exact" if rows[0] <= 8 else "tile", layer=True, l_max=256,
                                         args=["x", "layer", "v", "s"])))
    if bad_sha is not None:
        out = [(exp, dict(e, sha256="0" * 64) if e["id"] == bad_sha else e) for exp, e in out]
    return out


def _exact():
    """exact (override) entries for EXACT, in the format-1 form: planes v (K, N) = w.T / 2 and s (1, N)"""
    n, k = EXACT
    base = {"type": Q8_0, "type_name": "Q8_0", "rows": n, "cols": k, "platform": "cpu"}
    pay = {f"rep_{n}_{k}": (_export(_q8, spec((n, k // BS * TS), jnp.uint8)), dict(base, op="repack", args=["raw"], outs=["v", "s"])),
           f"deq_{n}_{k}": (_export(_deq, spec((k, n), f32), spec((1, n), f32)), dict(base, op="dequant", args=["v", "s"]))}
    for rc in ROWS_EXACT:
        pay[f"mm_{n}_{k}_b{rc}"] = (_export(_mm, spec((rc, k), bf16), spec((k, n), f32), spec((1, n), f32)),
                                    dict(base, op="matmul", args=["x", "v", "s"], row_class=rc))
    return pay


def _build(tmp, fmt=2, classes=None, exact=False, rule="shape-class", man_version=VER, roles=None):
    tmp.mkdir(parents=True, exist_ok=True)
    classes = _classes() if classes is None else classes
    man = {"package": PKG, "version": man_version, "jax": jax.__version__, "chips": ["v6e"], "payloads": {},
           "row_classes": list(ROWS_EXACT) if exact else [], "plane_bytes": {}, "type_ids": {"Q8_0": Q8_0},
           "expert_types": [], "nb_types": [], "moe_impl": {}, "nb_bytes": {}, "moe_payloads": {}, "kq_classes": []}
    if fmt is not None:
        man.update(format=fmt, rule=rule, roles=roles or {"kq_classes": "default", "payloads": "override",
                                                          "moe_payloads": "exact-only"})
    p = tmp / f"{PKG}-{VER}-py3-none-any.whl"
    with zipfile.ZipFile(p, "w") as z:
        for exp, e in classes:
            b = exp.serialize()
            name = e["id"] + ".jaxexport"
            man["kq_classes"].append(dict({"sha256": hashlib.sha256(b).hexdigest()}, **e, file=name, bytes=len(b)))
            z.writestr(f"{PKG}/{name}", b)
        for name, (exp, e) in (_exact() if exact else {}).items():
            b = exp.serialize()
            man["payloads"][name] = dict(e, file=name + ".jaxexport", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
            z.writestr(f"{PKG}/{name}.jaxexport", b)
        if exact:
            man["plane_bytes"]["Q8_0:%d:%d" % EXACT] = 7777
        z.writestr(f"{PKG}/manifest.json", json.dumps(man))
        z.writestr(f"{PKG}-{VER}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {PKG}\nVersion: {VER}\n")
    CAT.append({"for": "gguf", "file": p.name, "url": p.as_uri(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "jax": jax.__version__, "libtpu": "0.0.48.*", "chips": ["v6e"], "types": ["Q8_0"]})
    return p


def _raw(n, k, seed=0):
    """valid Q8_0 blocks: a finite float16 scale and 32 int8 values per block"""
    rng = np.random.default_rng(seed)
    nb = k // BS
    d = rng.uniform(0.001, 0.02, (n, nb)).astype(np.float16).view(np.uint8).reshape(n, nb, 2)
    q = rng.integers(-127, 128, (n, nb, 32), dtype=np.int8).view(np.uint8)
    return np.concatenate([d, q], axis=-1).reshape(n, nb * TS)


def _want(x, raw, k):
    w = R.dequant(jnp.asarray(raw), Q8_0, k)
    return jnp.dot(x.astype(f32), w.T, precision=jax.lax.Precision.HIGHEST)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    CAT.clear()
    monkeypatch.setattr(B, "WIDE", False)          # the routing tests state the 0.1.11 pick; the WIDE tests set it on
    monkeypatch.setattr(B, "rows", lambda: [dict(r) for r in CAT], raising=False)
    monkeypatch.delenv("MASHBOX_GGUF_KERNELS", raising=False)
    yield
    ops.load_kernels(module="off")


# shapes no exact entry lists: (rows, cols) -> the class pick (tk, tn, Kp, Np)
CLASS_SHAPES = {(37, 96): (64, 16, 128, 48), (64, 256): (128, 32, 256, 64), (5, 32): (64, 16, 64, 16),
                (40, 448): (64, 16, 448, 48), (1024, 512): (128, 32, 512, 1024)}


@pytest.mark.parametrize("shape", sorted(CLASS_SHAPES))
def test_a_shape_class_covers_a_shape_no_exact_entry_lists(tmp_path, shape):
    got = ops.load_kernels(module=str(_build(tmp_path)))
    assert ops._KERNELS["error"] is None and Q8_0 in got
    assert ops.weight_path(Q8_0, shape, experts=False) == "kq"
    assert ops._KERNELS["module"].kq_route(Q8_0, shape) == "class"
    h = ops.prepare(_raw(*shape), Q8_0, shape)
    assert isinstance(h, ops.Packed)


@pytest.mark.parametrize("shape", [(37, 96), (64, 256), (5, 32), (40, 448)])
@pytest.mark.parametrize("lead", [(1,), (3,), (30,), (2, 20)])
def test_the_class_matmul_pads_x_and_cuts_the_padding_back(tmp_path, shape, lead):
    ops.load_kernels(module=str(_build(tmp_path)))
    raw = _raw(*shape, seed=sum(shape))
    h = ops.prepare(raw, Q8_0, shape)
    assert isinstance(h, ops.Packed)                       # the class path, not a silent fallback to the exact one
    x = jax.random.normal(jax.random.PRNGKey(1), lead + (shape[1],)).astype(bf16)
    y = ops.qmatmul(x, h)
    want = _want(x, raw, shape[1]).astype(x.dtype)
    assert y.shape == lead + (shape[0],) and y.dtype == x.dtype
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(want, np.float32), rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("jit", [False, True])
def test_the_class_matmul_runs_inside_jit_and_on_an_empty_batch(tmp_path, jit):
    ops.load_kernels(module=str(_build(tmp_path)))
    h = ops.prepare(_raw(37, 96), Q8_0, (37, 96))
    assert isinstance(h, ops.Packed)
    f = jax.jit(ops.qmatmul) if jit else ops.qmatmul
    assert f(jnp.ones((3, 96), bf16), h).shape == (3, 37)
    assert f(jnp.zeros((2, 0, 96), bf16), h).shape == (2, 0, 37)


@pytest.mark.parametrize("shape", [(37, 96), (5, 32), (40, 448)])
def test_dequant_of_class_planes_is_the_reference_bit_for_bit(tmp_path, shape):
    ops.load_kernels(module=str(_build(tmp_path)))
    raw = _raw(*shape, seed=3)
    h = ops.prepare(raw, Q8_0, shape)
    assert isinstance(h, ops.Packed)
    ref = R.dequant(jnp.asarray(raw), Q8_0, shape[1])
    got = ops.dequant(h)
    assert got.shape == shape
    np.testing.assert_array_equal(np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32))


def test_a_stack_of_class_weights_repacks_each_matrix(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path)))
    raw = np.stack([_raw(37, 96, seed=s) for s in range(3)])
    mod = ops._KERNELS["module"]
    p = mod.repack_host(Q8_0, raw, (37, 96))
    assert p.lead == (3,) and p.planes["v"].shape == (3, 128, 48)
    got = mod.dequant(p)
    ref = jnp.stack([R.dequant(jnp.asarray(r), Q8_0, 96) for r in raw])
    np.testing.assert_array_equal(np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32))


def test_plane_bytes_of_a_class_shape_come_from_the_padded_shape(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path)))
    # (37, 96): Kp 128, Np 48 -> v (128, 48) float32 + s (1, 48) float32
    assert ops.device_nbytes(Q8_0, (37, 96), 37 * 3 * TS, experts=False) == 128 * 48 * 4 + 48 * 4
    assert ops.device_nbytes(Q8_0, (2, 37, 96), 2 * 37 * 3 * TS, experts=True) == 2 * (128 * 48 * 4 + 48 * 4)


def test_an_exact_entry_overrides_the_class_and_the_class_covers_the_rest(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path, exact=True)))
    mod = ops._KERNELS["module"]
    assert mod.kq_route(Q8_0, EXACT) == "exact" and mod.kq_route(Q8_0, (37, 96)) == "class"
    assert ops.device_nbytes(Q8_0, EXACT, 1, experts=False) == 7777          # the override's own plane_bytes
    raw = _raw(*EXACT, seed=5)
    h = ops.prepare(raw, Q8_0, EXACT)
    assert isinstance(h, ops.Packed)
    x = jax.random.normal(jax.random.PRNGKey(2), (20, EXACT[1])).astype(bf16)
    y = ops.qmatmul(x, h)
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(_want(x, raw, EXACT[1]).astype(bf16), np.float32),
                               rtol=2e-2, atol=1e-2)
    assert {"rep_32_64", "mm_32_64_b16", "mm_32_64_b32"} <= set(mod._exp)   # the preflight's 1 row: b16; 20 rows: b32
    assert mod._cexp == {}                                                   # no class payload was read


def test_a_format_1_manifest_reads_no_classes(tmp_path):
    """an old reader's bundle: no format key; any kq_classes it carries are not its key, so nothing changes for it"""
    ops.load_kernels(module=str(_build(tmp_path, fmt=None, exact=True)))
    assert ops._KERNELS["error"] is None
    assert ops.weight_path(Q8_0, EXACT, experts=False) == "kq"
    assert ops.weight_path(Q8_0, (37, 96), experts=False) == "exact"
    assert isinstance(ops.prepare(_raw(37, 96), Q8_0, (37, 96)), ops.Raw)


@pytest.mark.parametrize("kw", [{"fmt": 3}, {"rule": "exact-shape"}, {"man_version": "0.0.8"},
                                {"roles": {"kq_classes": "override", "payloads": "default", "moe_payloads": "exact-only"}}])
def test_a_manifest_this_reader_cannot_vouch_for_is_refused(tmp_path, kw):
    with pytest.raises(ValueError):
        B.Bundle(_build(tmp_path, **kw))


@pytest.mark.parametrize("shape,why", [((37, 100), "cols not a multiple of the block"), ((37, 544), "cols over k_max"),
                                       ((NMAX + 1, 96), "rows over n_max"), ((0, 96), "no rows")])
def test_out_of_domain_shapes_stay_on_the_exact_path(tmp_path, shape, why):
    ops.load_kernels(module=str(_build(tmp_path)))
    mod = ops._KERNELS["module"]
    assert not mod.kq_covers(Q8_0, shape), why
    assert mod.kq_route(Q8_0, shape) is None
    assert ops.weight_path(Q8_0, shape, experts=False) == "exact"


def test_a_type_with_no_class_stays_on_the_exact_path(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path)))
    mod = ops._KERNELS["module"]
    assert Q4_K not in mod.TYPES and not mod.kq_covers(Q4_K, (256, 256))
    assert ops.weight_path(Q4_K, (256, 256), experts=False) == "exact"


def test_a_class_without_every_row_class_is_not_picked(tmp_path):
    """tn 16 has no matmul for row class 8 at either tk: a shape whose Np only tn 16 divides (48) stays exact; a shape
    whose Np tn 32 divides (64) still runs, on tn 32. With only (tk 64, tn 16) short, K 448 (tk 64 divides it) pads to
    tk 128 instead, whose tn 16 is whole"""
    ops.load_kernels(module=str(_build(tmp_path, classes=_classes(drop={(64, 16, 8), (128, 16, 8)}))))
    mod = ops._KERNELS["module"]
    assert ops.weight_path(Q8_0, (37, 96), experts=False) == "exact"
    assert ops.weight_path(Q8_0, (37, 256), experts=False) == "exact"
    assert ops.weight_path(Q8_0, (60, 96), experts=False) == "kq" and mod._pick(Q8_0, 60, 96) == (64, 32, 128, 64)
    assert ops.weight_path(Q8_0, (64, 256), experts=False) == "kq" and mod._pick(Q8_0, 64, 256) == (128, 32, 256, 64)
    ops.load_kernels(module=str(_build(tmp_path / "one", classes=_classes(drop={(64, 16, 8)}))))
    assert ops._KERNELS["module"]._pick(Q8_0, 40, 448) == (128, 16, 512, 48)


def test_a_payload_whose_bytes_are_not_the_manifests_sha256_raises_and_the_loader_falls_back(tmp_path, capsys):
    ops.load_kernels(module=str(_build(tmp_path, classes=_classes(bad_sha="rep.q8.tk64"))))
    mod = ops._KERNELS["module"]
    with pytest.raises(ValueError, match="sha256"):
        mod.repack_host(Q8_0, _raw(37, 96), (37, 96))
    assert isinstance(ops.prepare(_raw(37, 96), Q8_0, (37, 96)), ops.Raw)
    assert "kq planes for Q8_0 (37, 96) failed" in capsys.readouterr().err
    assert isinstance(ops.prepare(_raw(37, 256), Q8_0, (37, 256)), ops.Packed)       # tk 128's own payload is fine


def test_a_class_payload_called_outside_its_constraints_raises_and_the_loader_falls_back(tmp_path, capsys):
    """the atoms say rows up to n_max, but the repack payload was exported for Np <= 4 x 16: jax.export refuses the
    call, and the weight loads on the exact path with the loader's warning"""
    ops.load_kernels(module=str(_build(tmp_path, classes=_classes(a_max=4))))
    assert isinstance(ops.prepare(_raw(64, 96), Q8_0, (64, 96)), ops.Packed)
    assert isinstance(ops.prepare(_raw(100, 96), Q8_0, (100, 96)), ops.Raw)
    assert "kq planes for Q8_0 (100, 96) failed" in capsys.readouterr().err


# the pick on the default menus (tk 1280 / 1024 / 512, tn menus per config), from a manifest's kq_classes alone: no payload is read
C8 = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40)
C4 = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)
LADDER = ((8,), (16,), (32,), (64,), (128,), (256,), (512,), (1024, 1536, 2048))


def _fake(t, bs, menu, config, nmult=True):
    atoms = {"bs": bs, "n_pad": 128, "n_max": 524288, "k_max": 65536}
    out = []
    for tk in (1280, 1024, 512):
        for op, plats in (("repack", ("cpu", "tpu")), ("dequant", ("cpu", "tpu"))):
            out += [dict(id=f"{op}.{t}.{tk}.{p}", op=op, type=t, atoms=dict(atoms, tk=tk), platform=p) for p in plats]
        out += [dict(id=f"mm.{config}.{tk}.{m}.{r[0]}", op="matmul", config=config, types=[t], rows=list(r),
                     atoms=dict(atoms, tk=tk, tn=128 * m, **({"n_mult": 128} if nmult and r == (8,) else {})),
                     platform="tpu", form="exact" if r == (8,) else "tile")
                for m in menu for r in LADDER]
    return out


@pytest.mark.parametrize("t,bs,menu,config,shape,want", [
    (Q4_K, 256, C4, "C4M", (2048, 5120), (1280, 2048, 5120, 2048)),
    (Q8_0, 32, C8, "C8/32", (17408, 5120), (1280, 1024, 5120, 17408)),
    (14, 256, C4, "C6", (151936, 1024), (1024, 128, 1024, 151936)),
    (Q8_0, 32, C8, "C8/32", (896, 2880), (512, 896, 3072, 896)),
    (Q8_0, 32, C8, "C8/32", (100, 32), (512, 128, 512, 128)),
    (Q4_K, 256, C4, "C4M", (524289, 256), None),
    (Q4_K, 256, C4, "C4M", (256, 65792), None),
])
def test_the_pick_on_the_real_menus(t, bs, menu, config, shape, want):
    idx = B.Classes(_fake(t, bs, menu, config))
    assert idx.pick(t, *shape, "tpu") == want
    if want is not None:
        assert idx.rows(t, want[0], want[1], "tpu") == (8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048)


def test_one_matmul_class_serves_types_of_different_block_sizes():
    """a matmul class takes planes, not blocks: one config serves types whose blocks differ (Q4_K 256, Q4_0 32), so its
    atoms carry no block size; the block size belongs to each type's repack / dequant class"""
    atoms = {"n_pad": 128, "n_max": 524288, "k_max": 65536}
    ents = []
    for t, bs in ((Q4_K, 256), (2, 32)):
        ents += [dict(id=f"{op}.{t}.{p}", op=op, type=t, atoms=dict(atoms, bs=bs, tk=512), platform=p)
                 for op in ("repack", "dequant") for p in ("cpu", "tpu")]
    ents += [dict(id=f"mm.c4m.{r[0]}", op="matmul", config="C4M", types=[Q4_K, 2], rows=list(r),
                  atoms=dict(atoms, tk=512, tn=256), platform="tpu") for r in LADDER]
    idx = B.Classes(ents)
    assert idx.pick(Q4_K, 300, 2560, "tpu") is None                         # Np 384: tn 256 does not divide it
    assert idx.pick(Q4_K, 200, 2560, "tpu") == (512, 256, 2560, 256)
    assert idx.pick(2, 200, 2560, "tpu") == (512, 256, 2560, 256)
    assert idx.pick(2, 200, 2592, "tpu") == (512, 256, 3072, 256)            # K = 81 Q4_0 blocks: padded to tk 512
    assert idx.pick(Q4_K, 200, 2592, "tpu") is None                         # not a whole number of Q4_K blocks


# ------------------------------------------------------------------------------------------------- layer forms (layer=)
def _stack(L, n, k, seed=0):
    return np.stack([_raw(n, k, seed=seed + i) for i in range(L)])


def _warns(monkeypatch):
    got = []
    monkeypatch.setattr(ops, "warn", got.append)
    return got


@pytest.mark.parametrize("shape", [(37, 96), (64, 256)])
def test_layer_forms_multiply_a_stack_at_its_layer_without_slicing_it(tmp_path, shape, monkeypatch):
    w = _warns(monkeypatch)
    ops.load_kernels(module=str(_build(tmp_path, classes=_classes(layer=True))))
    assert ops.kernel_picks_layer() and ops._KERNELS["module"].picks_layer
    raw = _stack(3, *shape, seed=sum(shape))
    h = ops.prepare(raw, Q8_0, shape)
    assert isinstance(h, ops.Packed) and h.packed.route == "class" and h.packed.lead == (3,)
    assert ops._KQ_LAYER[(Q8_0, shape)] is True and w == []          # the layer= preflight ran, no warning
    held, xs = ops.scan_split({"w": h})
    assert isinstance(held["w"], ops.Packed) and xs["w"] is None       # held whole: the scan never slices the planes
    x = jax.random.normal(jax.random.PRNGKey(2), (5, shape[1])).astype(bf16)

    def body(c, l):
        return c, ops.qmatmul(x, ops.scan_join(held, xs, l)["w"])
    _, ys = jax.lax.scan(body, 0, jnp.arange(3, dtype=jnp.int32))
    for l in range(3):
        sliced = ops.Packed(jax.tree_util.tree_map(lambda a: a[l], h.packed), Q8_0, shape)
        np.testing.assert_array_equal(np.asarray(ys[l], np.float32), np.asarray(ops.qmatmul(x, sliced), np.float32))
        np.testing.assert_allclose(np.asarray(ys[l], np.float32), np.asarray(_want(x, raw[l], shape[1]), np.float32),
                                   rtol=2e-2, atol=1e-2)


def test_a_stack_of_an_override_shape_takes_its_class_when_the_class_has_layer_forms(tmp_path, monkeypatch):
    _warns(monkeypatch)
    ops.load_kernels(module=str(_build(tmp_path, classes=_classes(layer=True), exact=True)))
    one = ops.prepare(_raw(*EXACT), Q8_0, EXACT)
    stack = ops.prepare(_stack(2, *EXACT), Q8_0, EXACT)
    assert one.packed.route == "exact"                  # an unstacked weight keeps its override
    assert stack.packed.route == "class"                # a stack takes the class (the override has no layer form)
    assert ops._KQ_LAYER[(Q8_0, EXACT)] is True
    bd = ops._KERNELS["module"]
    over, cls = bd._kq_pb[(Q8_0,) + EXACT], bd._class_bytes(Q8_0, *EXACT)
    assert bd.plane_bytes(Q8_0, EXACT) == max(over, cls)      # the estimate covers the stack's class planes too
    assert sum(a.nbytes for a in stack.packed.planes.values()) == 2 * cls
    x = jax.random.normal(jax.random.PRNGKey(3), (4, EXACT[1])).astype(bf16)
    y = ops.qmatmul(x, ops.At(stack.packed, jnp.int32(1), Q8_0, EXACT))
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(_want(x, _raw(*EXACT, seed=1), EXACT[1]), np.float32),
                               rtol=2e-2, atol=1e-2)
    np.testing.assert_array_equal(np.asarray(ops.dequant(stack)[1]), np.asarray(ops.dequant(
        ops.prepare(_raw(*EXACT, seed=1), Q8_0, EXACT))))       # the class planes dequantize to the override's values


def test_without_layer_forms_a_stack_is_sliced_as_before(tmp_path, monkeypatch):
    w = _warns(monkeypatch)
    ops.load_kernels(module=str(_build(tmp_path, exact=True)))
    assert not ops._KERNELS["module"].picks_layer and not ops.kernel_picks_layer()
    bd = ops._KERNELS["module"]
    assert bd.plane_bytes(Q8_0, EXACT) == bd._kq_pb[(Q8_0,) + EXACT]          # no layer forms: the override's bytes
    assert ops.prepare(_stack(2, *EXACT), Q8_0, EXACT).packed.route == "exact"
    h = ops.prepare(_stack(2, 37, 96), Q8_0, (37, 96))
    assert (Q8_0, (37, 96)) not in ops._KQ_LAYER and w == []
    assert ops.scan_split({"w": h}) == (None, {"w": h})


def test_a_class_missing_one_layer_form_slices_its_stacks_with_one_warning(tmp_path, monkeypatch):
    w = _warns(monkeypatch)
    ops.load_kernels(module=str(_build(tmp_path, classes=_classes(layer=True, drop_layer={(64, 16, 8)}))))
    shape = (37, 96)                                    # picks tk 64, tn 16: its row class 8 has no layer form
    h = ops.prepare(_stack(2, *shape), Q8_0, shape)
    assert h.packed.route == "class" and ops._KQ_LAYER[(Q8_0, shape)] is not True
    assert len(w) == 1 and "layer=" in w[0] and "sliced per layer" in w[0], w
    held, xs = ops.scan_split({"w": h})
    assert held is None and xs == {"w": h}


# 0.1.12: the exact-form row classes of a weight already on a class may take a larger class tn of its config (WIDE,
# STEP_COLS[config], STEP_ERR[config]) when their payloads state n_mult; tile-form row classes keep the pick's tn, and
# no plane is padded for the wider tile (one copy of each weight serves every call)
WIDE_N = [128 * m for m in (1, 2, 3, 7, 19, 38, 61, 76, 79, 136, 1187, 1940)] + [9728, 10000, 151936, 248320, 4864, 896]
TILE_ROWS = (16, 32, 64, 128, 256, 512, 1024)


def _both(idx, monkeypatch, t, n, k):
    monkeypatch.setattr(B, "WIDE", False); idx._picks.clear(); old = idx.pick(t, n, k, "tpu")
    monkeypatch.setattr(B, "WIDE", True); idx._picks.clear(); new = idx.pick(t, n, k, "tpu")
    return old, new


@pytest.mark.parametrize("c", [100.0, 600.0, 3000.0, 12000.0])
@pytest.mark.parametrize("t,bs,menu,config", [(Q4_K, 256, C4, "C4M"), (Q8_0, 32, C8, "C8/32"), (14, 256, C4, "C6")])
def test_the_wide_tile_keeps_the_planes_and_at_least_halves_the_modelled_cost(monkeypatch, c, t, bs, menu, config):
    """for any step cost: the pick (tk, tn, Kp, Np) stays, so the planes are the 0.1.11 ones; the exact-form row class
    takes tw >= tn, every tile-form row class tn; a wider tw models at most half the cost (Np' a multiple of both tiles,
    <= n_max); no 0.1.11 pick, no pick"""
    idx = B.Classes(_fake(t, bs, menu, config))
    monkeypatch.setattr(B, "STEP_COLS", {config: c})
    monkeypatch.setattr(B, "STEP_ERR", {})
    for n in WIDE_N:
        for k in (2560, 5120, 1024):
            old, new = _both(idx, monkeypatch, t, n, k)
            if old is None:
                assert new is None
                continue
            tw = idx.tn_for(t, n, k, 8, "tpu")
            assert new[:3] == old[:3] and all(idx.tn_for(t, n, k, r, "tpu") == old[1] for r in TILE_ROWS)
            npm = -(-old[3] // math.lcm(tw, old[1])) * math.lcm(tw, old[1])
            assert new == old and tw >= old[1] and npm <= 524288
            cc = c * 1280 / old[0]
            assert npm * (1 + cc / tw) <= old[3] * (1 + cc / old[1]) * (0.5 if tw != old[1] else 1)
            assert idx.layer_ok(t, n, k, "tpu") is False             # _fake has no layer forms


@pytest.mark.parametrize("t,bs,menu,config", [(Q4_K, 256, C4, "C4M"), (Q8_0, 32, C8, "C8/32")])
def test_a_gain_inside_the_cost_models_error_keeps_the_old_pick(monkeypatch, t, bs, menu, config):
    """STEP_ERR 0.99: no modelled gain clears the error, so nothing switches"""
    idx = B.Classes(_fake(t, bs, menu, config))
    monkeypatch.setattr(B, "STEP_COLS", {config: 600.0})
    monkeypatch.setattr(B, "STEP_ERR", {config: 0.99})
    for n in WIDE_N:
        old, new = _both(idx, monkeypatch, t, n, 2560)
        assert new == old and (old is None or idx.tn_for(t, n, 2560, 8, "tpu") == old[1])


def test_a_gain_under_half_keeps_the_old_pick(monkeypatch):
    """Q8_0 17408 x 5120 with step cost 600: tile 1024 -> 3072 (Np' 18432) models 0.80 of the cost, under the error
    margin's bar but not half: the 0.1.11 tile stays (WIDE_GAIN); 151936 rows model under half and switch; the planes
    never change"""
    idx = B.Classes(_fake(Q8_0, 32, C8, "C8/32"))
    monkeypatch.setattr(B, "STEP_COLS", {"C8/32": 600.0})
    monkeypatch.setattr(B, "STEP_ERR", {})
    old, new = _both(idx, monkeypatch, Q8_0, 17408, 5120)
    assert old == new == (1280, 1024, 5120, 17408) and idx.tn_for(Q8_0, 17408, 5120, 8, "tpu") == 1024
    monkeypatch.setattr(B, "WIDE_GAIN", 0.9)
    old, new = _both(idx, monkeypatch, Q8_0, 17408, 5120)
    assert new == old and idx.tn_for(Q8_0, 17408, 5120, 8, "tpu") > 1024
    monkeypatch.setattr(B, "WIDE_GAIN", 0.5)
    old, new = _both(idx, monkeypatch, Q8_0, 151936, 5120)
    assert old[1] == 128 and new == old and idx.tn_for(Q8_0, 151936, 5120, 8, "tpu") >= 1280


def test_the_wide_tile_on_a_prime_column_count(monkeypatch):
    """N = 151936 = 128 x 1187 (1187 is prime): the 0.1.11 pick is tn 128 (1187 N tiles); with a step cost the
    exact-form row class takes a large tile over the same 151936-column planes (its last step partial); the tile-form
    ones stay on tn 128"""
    idx = B.Classes(_fake(14, 256, C4, "C6"))
    monkeypatch.setattr(B, "STEP_COLS", {"C6": 600.0})
    monkeypatch.setattr(B, "STEP_ERR", {"C6": 0.1})
    monkeypatch.setattr(B, "WIDE", True)
    tk, tn, Kp, Np = idx.pick(14, 151936, 1024, "tpu")
    tw = idx.tn_for(14, 151936, 1024, 8, "tpu")
    assert tn == 128 and tw >= 1280 and Np == 151936 and Np % tw != 0
    assert idx.tn_for(14, 151936, 1024, 64, "tpu") == 128


@pytest.mark.parametrize("steps", [None, {}, {"C8/32": 600.0}])
def test_a_config_without_a_step_cost_keeps_the_old_pick(monkeypatch, steps):
    idx = B.Classes(_fake(Q4_K, 256, C4, "C4M"))
    monkeypatch.setattr(B, "STEP_COLS", steps)
    monkeypatch.setattr(B, "WIDE", True)
    assert idx.pick(Q4_K, 9728, 2560, "tpu") == (1280, 512, 2560, 9728)
    assert idx.tn_for(Q4_K, 9728, 2560, 8, "tpu") == 512
    assert idx.config(Q4_K, 1280, 512, "tpu") == "C4M" and idx.form(Q4_K, 1280, 512, 8, "tpu") == "exact"


def test_entries_that_state_no_form_keep_the_old_pick(monkeypatch):
    ents = [{k: v for k, v in e.items() if k != "form"} for e in _fake(Q4_K, 256, C4, "C4M")]
    idx = B.Classes(ents)
    monkeypatch.setattr(B, "STEP_COLS", {"C4M": 5000.0})
    monkeypatch.setattr(B, "STEP_ERR", {})
    old, new = _both(idx, monkeypatch, Q4_K, 9728, 2560)
    assert new == old == (1280, 512, 2560, 9728) and idx.tn_for(Q4_K, 9728, 2560, 8, "tpu") == 512


@pytest.mark.parametrize("t,bs,menu,config", [(Q4_K, 256, C4, "C4M"), (14, 256, C4, "C6")])
def test_exact_payloads_that_state_no_n_mult_keep_the_old_pick(monkeypatch, t, bs, menu, config):
    """a kq 0.1.7 bundle (0.1.11): its exact-form payloads take planes only as wide as a multiple of their own tile, so
    with no padding no wider tile fits 9728 or 151936 columns and the 0.1.11 pick stays, whatever the step cost"""
    idx = B.Classes(_fake(t, bs, menu, config, nmult=False))
    monkeypatch.setattr(B, "STEP_COLS", {config: 12000.0})
    monkeypatch.setattr(B, "STEP_ERR", {})
    for n in (9728, 151936):
        old, new = _both(idx, monkeypatch, t, n, 2560)
        assert new == old and idx.tn_for(t, n, 2560, 8, "tpu") == old[1]


def test_a_step_cost_the_bundle_states_wins_over_the_readers_table(monkeypatch):
    """a matmul entry's own step_cols / step_err (a new kernel or config) are used instead of STEP_COLS / STEP_ERR"""
    ents = [dict(e, step_cols=5000.0, step_err=0.0) if e["op"] == "matmul" else e for e in _fake(Q4_K, 256, C4, "C4M")]
    idx = B.Classes(ents)
    monkeypatch.setattr(B, "STEP_COLS", {})
    monkeypatch.setattr(B, "STEP_ERR", {})
    old, new = _both(idx, monkeypatch, Q4_K, 9728, 2560)
    assert old == new == (1280, 512, 2560, 9728)
    assert idx.tn_for(Q4_K, 9728, 2560, 8, "tpu") == 2560
    ents = [dict(e, step_err=0.99) if e["op"] == "matmul" else e for e in ents]
    idx = B.Classes(ents)
    assert _both(idx, monkeypatch, Q4_K, 9728, 2560)[1] == (1280, 512, 2560, 9728)
    ents = [dict(e, step_cols=0.0) if e["op"] == "matmul" else e for e in _fake(Q4_K, 256, C4, "C4M")]
    monkeypatch.setattr(B, "STEP_COLS", {"C4M": 5000.0})
    assert _both(B.Classes(ents), monkeypatch, Q4_K, 9728, 2560)[1] == (1280, 512, 2560, 9728)


@pytest.mark.parametrize("shape", sorted(CLASS_SHAPES))
def test_a_wide_tile_multiplies_like_the_old_pick(tmp_path, monkeypatch, shape):
    """the same values through the reader for an exact-form row class (3 rows, on tw over the unpadded planes) and a
    tile-form one (20 rows, on tn), plain and through the layer form; the planes and the memory estimate are the 0.1.11
    ones. N tiles 16, 32, 64, so a wide tile can model under half the cost"""
    got = ops.load_kernels(module=str(_build(tmp_path, classes=_classes(layer=True, tns=(16, 32, 64)))))
    assert ops._KERNELS["error"] is None and Q8_0 in got
    mod = ops._KERNELS["module"]
    rng = np.random.default_rng(4)
    raw = _raw(*shape, seed=3)
    stack = np.stack([_raw(*shape, seed=5), raw])
    xs = [jnp.asarray(rng.standard_normal((r, shape[1])), bf16) for r in (3, 20)]
    monkeypatch.setattr(B, "STEP_COLS", {"C8": 5000.0})
    monkeypatch.setattr(B, "STEP_ERR", {"C8": 0.1})
    ys, picks, pb = [], [], []
    for wide in (False, True):
        monkeypatch.setattr(B, "WIDE", wide)
        mod._cls._picks.clear()
        picks.append(mod._cls.pick(Q8_0, *shape, B._platform()) + tuple(mod._cls.tn_for(Q8_0, *shape, r, B._platform())
                                                                        for r in (4, 24)))
        pb.append(mod.plane_bytes(Q8_0, shape))
        h = ops.prepare(raw, Q8_0, shape)
        hs = ops.prepare(stack, Q8_0, shape)
        assert isinstance(h, ops.Packed) and isinstance(hs, ops.Packed)
        ys.append([np.asarray(mod.matmul(x, h.packed)) for x in xs]
                  + [np.asarray(mod.matmul(x, hs.packed, layer=jnp.int32(1))) for x in xs])
    for a, b in zip(*ys):
        assert a.shape[-1] == shape[0]
        np.testing.assert_array_equal(a, b)
    assert pb[1] == pb[0] and picks[1][:4] == picks[0][:4]
    if shape == (37, 96):                                   # Np 48 on tn 16 -> 3 rows on tn 64 (one partial step); 20 rows on tn 16
        assert picks == [(64, 16, 128, 48, 16, 16), (64, 16, 128, 48, 64, 16)]
