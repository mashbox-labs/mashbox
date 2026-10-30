"""Bundle format 2 (mashbox/ops/bundle.py): shape-class keys for the dense half. A format-2 manifest (rule "shape-class")
carries kq_classes, the default net, under the exact entries of 'payloads', which stay as speed overrides. Each class is
data only: atoms over format and hardware constants (block size bs, K tile tk, N pad, N tile tn, the domain bounds
n_max / k_max) plus the row classes its matmul takes. The reader picks a class by the atoms and never knows the recipe:
tk = the largest class tk that divides K, else the smallest with K zero-padded to it; Np = N rounded up to n_pad; tn =
the largest class tn that divides Np and has a matmul for every row class. The raw blocks are zero-padded to (Np, Kp),
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


def _classes(drop=None, n_max=NMAX, k_max=KMAX, a_max=None, bad_sha=None, t=Q8_0):
    """-> [(exported, entry)]: repack + dequant per tk, matmul per (tk, tn) and row class; drop = {(tk, tn, rc)} left out;
    a_max: the repack's own symbolic constraint on Np / NPAD (to call it outside its constraints)"""
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
        for tn in TNS:
            for rows in ((4,), (8,), (16, 24)):
                if any((tk, tn, r) in (drop or ()) for r in rows):
                    continue
                if len(rows) > 1:                    # one payload for rows 16 and 24: x has 8 * c rows
                    a, b, c = _sym("a, b, c", ["c >= 2", "c <= 3"])
                    xr = 8 * c
                else:
                    (a, b), xr = _sym("a, b"), rows[0]
                mm = _export(_mm, spec((xr, tk * b), bf16), spec((tk * b, tn * a), f32), spec((1, tn * a), f32))
                out.append((mm, dict(lab, id=f"mm.c8.tk{tk}.tn{tn}.r{rows[0]}", op="matmul", config="C8", types=[t],
                                     atoms=dict(atoms, tk=tk, tn=tn), rows=list(rows), platform="cpu",
                                     args=["x", "v", "s"])))
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


def _fake(t, bs, menu, config):
    atoms = {"bs": bs, "n_pad": 128, "n_max": 524288, "k_max": 65536}
    out = []
    for tk in (1280, 1024, 512):
        for op, plats in (("repack", ("cpu", "tpu")), ("dequant", ("cpu", "tpu"))):
            out += [dict(id=f"{op}.{t}.{tk}.{p}", op=op, type=t, atoms=dict(atoms, tk=tk), platform=p) for p in plats]
        out += [dict(id=f"mm.{config}.{tk}.{m}.{r[0]}", op="matmul", config=config, types=[t], rows=list(r),
                     atoms=dict(atoms, tk=tk, tn=128 * m), platform="tpu") for m in menu for r in LADDER]
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
