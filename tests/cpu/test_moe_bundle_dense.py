"""The dense half of the kernel package read from a payload bundle (mashbox/ops/bundle.py): the kq payloads (repack,
matmul, dequant per exact (type, rows, cols) and, for matmul, per row class). These tests build a small bundle on CPU
from plain-JAX stand-ins exported for the cpu platform (the real bundle carries TPU matmul payloads; nothing here
measures a kernel) and check the routing only: a dense weight whose (type, rows, cols) the manifest lists loads on the kq
planes and multiplies through the matmul payload of its row class (activations cast to bfloat16, zero-padded, cut back);
a shape it does not list, or lists without every row class, stays on the exact path and says so in weight_path; the
memory estimate reads the manifest's plane_bytes for a listed shape and the file's bytes for any other; the planes keep
the payload's own argument order; dequant(repack(raw)) is the reference bit for bit.

Why: a bundle read for its routed-expert half only (TYPES empty) runs every dense weight on the exact float32 reference
path, even a shape whose kq payloads the wheel carries."""
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
Q8_0 = 8
N, K = 32, 64                    # the listed dense shape (rows, cols)
RB = K // 32 * 34                # Q8_0 row bytes
ROW_CLASSES = (16, 32)           # a small top class, so a call over 32 rows is cut into chunks
ROWS = []


def _repack(raw):
    """stand-in kq repack: two planes whose product is the reference dequant (v = w / 2, s = 2), returned in the order
    the manifest's 'outs' names (v, s)"""
    w = R.dequant(raw, Q8_0, K)
    return 0.5 * w, jnp.full((N, 1), 2.0, jnp.float32)


def _dequant(v, s):
    return v * s


def _matmul(x, v, s):
    return jnp.dot(x.astype(jnp.float32), (v * s).T, precision=jax.lax.Precision.HIGHEST)


def _build(tmp, shapes=((Q8_0, N, K),), drop_rc=None, plane_bytes=9999):
    tmp.mkdir(parents=True, exist_ok=True)
    spec = jax.ShapeDtypeStruct
    pay, man = {}, {"package": PKG, "version": "0.0.2", "jax": jax.__version__, "chips": ["v6e"], "payloads": {},
                    "row_classes": list(ROW_CLASSES), "plane_bytes": {}, "type_ids": {"Q8_0": Q8_0},
                    "expert_types": [], "nb_types": [], "moe_impl": {}, "nb_bytes": {}, "moe_payloads": {}}
    for t, n, k in shapes:
        assert (n, k) == (N, K), "the stand-ins are written for one shape"
        base = {"type": t, "type_name": "Q8_0", "rows": n, "cols": k, "platform": "cpu"}
        pay[f"rep_{n}_{k}"] = (jax.export.export(jax.jit(_repack), platforms=("cpu",))(spec((n, RB), jnp.uint8)),
                               dict(base, op="repack", args=["raw"], outs=["v", "s"]))
        pay[f"deq_{n}_{k}"] = (jax.export.export(jax.jit(_dequant), platforms=("cpu",))(spec((n, k), jnp.float32),
                                                                                       spec((n, 1), jnp.float32)),
                               dict(base, op="dequant", args=["v", "s"]))
        for rc in ROW_CLASSES:
            if rc == drop_rc:
                continue
            exp = jax.export.export(jax.jit(_matmul), platforms=("cpu",))(
                spec((rc, k), jnp.bfloat16), spec((n, k), jnp.float32), spec((n, 1), jnp.float32))
            pay[f"mm_{n}_{k}_b{rc}"] = (exp, dict(base, op="matmul", args=["x", "v", "s"], row_class=rc))
        man["plane_bytes"][f"Q8_0:{n}:{k}"] = plane_bytes
    p = tmp / "mashbox_engine_kernels-0.0.2-py3-none-any.whl"
    with zipfile.ZipFile(p, "w") as z:
        for name, (exp, e) in pay.items():
            b = exp.serialize()
            man["payloads"][name] = dict(e, file=name + ".jaxexport", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
            z.writestr(f"{PKG}/{name}.jaxexport", b)
        z.writestr(f"{PKG}/manifest.json", json.dumps(man))
    ROWS.append({"for": "gguf", "file": p.name, "url": p.as_uri(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                 "jax": jax.__version__, "libtpu": "0.0.48.*", "chips": ["v6e"], "types": ["Q8_0"]})
    return p


def _raw(n=N, seed=0):
    """valid Q8_0 blocks: a finite float16 scale and 32 int8 values per block"""
    rng = np.random.default_rng(seed)
    nb = K // 32
    d = rng.uniform(0.001, 0.02, (n, nb)).astype(np.float16).view(np.uint8).reshape(n, nb, 2)
    q = rng.integers(-127, 128, (n, nb, 32), dtype=np.int8).view(np.uint8)
    return np.concatenate([d, q], axis=-1).reshape(n, nb * 34)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ROWS.clear()
    monkeypatch.setattr(B, "rows", lambda: [dict(r) for r in ROWS], raising=False)
    monkeypatch.delenv("MASHBOX_GGUF_KERNELS", raising=False)
    yield
    ops.load_kernels(module="off")


def test_a_listed_dense_shape_takes_the_kq_planes_and_an_unlisted_one_the_exact_path(tmp_path):
    got = ops.load_kernels(module=str(_build(tmp_path)))
    assert ops._KERNELS["error"] is None and Q8_0 in got
    assert ops.weight_path(Q8_0, (N, K), experts=False) == "kq"
    assert ops.weight_path(Q8_0, (2 * N, K), experts=False) == "exact"
    assert isinstance(ops.prepare(_raw(), Q8_0, (N, K)), ops.Packed)
    assert isinstance(ops.prepare(_raw(2 * N), Q8_0, (2 * N, K)), ops.Raw)


@pytest.mark.parametrize("lead", [(1,), (3,), (16,), (2, 20)])
def test_the_matmul_payload_of_the_row_class_runs_and_cuts_the_padding(tmp_path, lead):
    ops.load_kernels(module=str(_build(tmp_path)))
    raw = _raw()
    h = ops.prepare(raw, Q8_0, (N, K))
    x = jax.random.normal(jax.random.PRNGKey(1), lead + (K,)).astype(jnp.bfloat16)
    y = ops.qmatmul(x, h)
    w = R.dequant(jnp.asarray(raw), Q8_0, K)
    want = jnp.dot(x.astype(jnp.float32), w.T, precision=jax.lax.Precision.HIGHEST).astype(x.dtype)
    assert y.shape == lead + (N,) and y.dtype == x.dtype
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(want, np.float32), rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("lead", [(0,), (2, 0)])
def test_an_empty_batch_gives_an_empty_result_of_the_right_shape(tmp_path, lead, jit):
    ops.load_kernels(module=str(_build(tmp_path)))
    h = ops.prepare(_raw(), Q8_0, (N, K))
    x = jnp.zeros(lead + (K,), jnp.bfloat16)
    y = (jax.jit(ops.qmatmul) if jit else ops.qmatmul)(x, h)
    assert y.shape == lead + (N,) and y.dtype == x.dtype


def test_the_matmul_runs_inside_jit_on_the_payload_not_the_exact_path(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path)))
    h = ops.prepare(_raw(), Q8_0, (N, K))
    x = jnp.ones((3, K), jnp.bfloat16)
    text = jax.jit(ops.qmatmul).lower(x, h).as_text()
    assert "optimization_barrier" not in text          # the exact path ties x and the blocks with a barrier


def test_dequant_of_the_planes_is_the_reference_bit_for_bit(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path)))
    raw = _raw()
    h = ops.prepare(raw, Q8_0, (N, K))
    ref = R.dequant(jnp.asarray(raw), Q8_0, K)
    np.testing.assert_array_equal(np.asarray(ops.dequant(h)).view(np.uint32), np.asarray(ref).view(np.uint32))


def test_plane_bytes_come_from_the_manifest_for_a_listed_shape_and_the_file_for_any_other(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path, plane_bytes=4321)))
    assert ops.device_nbytes(Q8_0, (N, K), N * RB, experts=False) == 4321
    assert ops.device_nbytes(Q8_0, (2 * N, K), 2 * N * RB, experts=False) == 2 * N * RB


def test_a_shape_without_every_row_class_stays_on_the_exact_path(tmp_path):
    ops.load_kernels(module=str(_build(tmp_path, drop_rc=32)))
    assert ops.weight_path(Q8_0, (N, K), experts=False) == "exact"
    assert isinstance(ops.prepare(_raw(), Q8_0, (N, K)), ops.Raw)


def test_the_planes_are_named_by_the_payloads_outs_and_passed_in_its_args_order(tmp_path):
    """a dict of planes comes back from device_put in sorted key order (s, v); the matmul and dequant payloads must
    still get them in the manifest's 'args' order (v, s), or the (N, K) and (N, 1) arrays land in each other's place"""
    ops.load_kernels(module=str(_build(tmp_path)))
    h = ops.prepare(_raw(), Q8_0, (N, K))
    assert sorted(h.packed.planes) == ["s", "v"]
    assert h.packed.planes["v"].shape == (N, K) and h.packed.planes["s"].shape == (N, 1)
    assert ops.qmatmul(jnp.ones((2, K), jnp.bfloat16), h).shape == (2, N) and ops.dequant(h).shape == (N, K)
