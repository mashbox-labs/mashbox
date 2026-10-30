"""The kernel package read from a payload bundle (mashbox/ops/bundle.py): a wheel of jax.export payloads + manifest.json,
no code. These tests build a small bundle on CPU from plain-JAX stand-ins exported for the cpu platform (the real bundle
carries TPU payloads; nothing here measures a kernel) and check the routing only: ops.load_kernels takes the wheel as
its kernel package, exact-shape lookup, the exact path for a shape the manifest does not list, the sha256 and version
refusals, plane order, and the record fields (dev_build false, the wheel's own sha256).

Which wheel loads: with no flag and no MASHBOX_ variable the engine reads
the catalog's engine_kernels row for this chip and JAX, fetches the row's file once into the cache and checks it; a wheel
is refused unless its sha256 is a row's (a tampered wheel with a self-consistent manifest included) and, on TPU, the
installed libtpu matches the row's pattern; the version a record names is the one of the bundle that loaded."""
import hashlib
import importlib.util
import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mashbox import ops
from mashbox.ops import bundle as B
from mashbox.moe import record as RX   # the routed-expert kernel's id: one place

E, D, F = 4, 8, 6
PKG = "mashbox_engine_kernels"


def _planes(w):
    return {"lo": w[..., 0], "hi": w[..., 1]}          # two planes, so the order of the flattened args matters


def _w(planes):
    return planes["lo"] + 2.0 * planes["hi"]


def _ffn_gated(xs, es, ws, g_hi, g_lo, u_hi, u_lo, d_hi, d_lo):
    g, u, d = _w({"lo": g_lo, "hi": g_hi}), _w({"lo": u_lo, "hi": u_hi}), _w({"lo": d_lo, "hi": d_hi})
    e = jnp.minimum(es, E - 1)
    h = jax.nn.silu(jnp.einsum("pd,pfd->pf", xs.astype(jnp.float32), g[e])) * jnp.einsum("pd,pfd->pf", xs.astype(jnp.float32), u[e])
    y = jnp.einsum("pf,pdf->pd", h, d[e])
    return jnp.where((es < E)[:, None], y * ws[:, None], 0.0)


def _repack(raw):
    r = raw.astype(jnp.float32)
    return {"hi": jnp.ones_like(r), "lo": r}


ROWS = []                       # the catalog's engine_kernels rows these tests see (B.rows)
_FIND = importlib.util.find_spec


def _row(p, **kw):
    r = {"for": "gguf", "file": p.name, "url": p.as_uri(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
         "jax": jax.__version__, "libtpu": "0.0.48.*", "chips": ["v6e"], "types": ["Q4_K"]}
    r.update(kw)
    return r


def _build(tmp, P_list=(8,), jaxv=None, corrupt=None, row=True, tamper=False):
    """a bundle wheel at tmp/<wheel name>; row: list it in the catalog (ROWS); tamper: the dequant payload is another
    program and the manifest carries that program's sha256 (self-consistent bytes, not the row's file)"""
    tmp.mkdir(parents=True, exist_ok=True)
    pay, man = {}, {"package": PKG, "version": "0.0.1", "jax": jaxv or jax.__version__, "chips": ["v6e"],
                    "payloads": {}, "expert_types": [12], "nb_types": [12], "moe_impl": {RX.KERNEL_ID: "0" * 64},
                    "nb_bytes": {"Q4_K:4:6:8": 1234}, "moe_payloads": {}}
    spec = lambda s, dt=jnp.float32: jax.ShapeDtypeStruct(s, dt)
    for P in P_list:
        name = f"ffn_p{P}"
        args = [spec((P, D), jnp.bfloat16), spec((P,), jnp.int32), spec((P,), jnp.float32)] + [spec((E, F, D))] * 4 + [spec((E, D, F))] * 2
        pay[name] = jax.export.export(jax.jit(_ffn_gated), platforms=("cpu",))(*args).serialize()
        man["moe_payloads"][name] = {"op": "expert_ffn", "platform": "cpu", "act": "silu", "E": E, "D": D, "F": F, "P": P, "G": 0,
                                         "scratch_bytes": 100 * P, "types": {"gate": 12, "up": 12, "down": 12}}
    pay["rep"] = jax.export.export(jax.jit(_repack), platforms=("cpu",))(spec((E, F, D), jnp.uint8)).serialize()
    man["moe_payloads"]["rep"] = {"op": "repack_nb", "platform": "cpu", "type": 12, "lead": [E], "rows": F, "cols": D}
    deq = (lambda hi, lo: lo + 3.0 * hi) if tamper else (lambda hi, lo: lo + 2.0 * hi)
    pay["deq"] = jax.export.export(jax.jit(deq), platforms=("cpu",))(spec((E, F, D)), spec((E, F, D))).serialize()
    man["moe_payloads"]["deq"] = {"op": "nb_dequant", "platform": "cpu", "type": 12, "lead": [E], "rows": F, "cols": D}
    for n, b in pay.items():
        man["moe_payloads"][n].update(file=n + ".jaxexport", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
    if corrupt:
        man["moe_payloads"][corrupt]["sha256"] = "0" * 64
    p = tmp / "mashbox_engine_kernels-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(f"{PKG}/manifest.json", json.dumps(man))
        for n, b in pay.items():
            z.writestr(f"{PKG}/{n}.jaxexport", b)
    if row:
        ROWS.append(_row(p))
    return p


def _W(key):
    ks = jax.random.split(key, 3)
    mk = lambda k, s: B.Planes({"lo": jax.random.normal(k, s), "hi": 0.1 * jax.random.normal(jax.random.fold_in(k, 1), s)}, 12, s[1], s[2])
    return {"gate": mk(ks[0], (E, F, D)), "up": mk(ks[1], (E, F, D)), "down": mk(ks[2], (E, D, F))}


def _ref(xs, es, ws, W):
    pl = lambda k: [W[k].planes[n] for n in sorted(W[k].planes)]
    return _ffn_gated(xs, es, ws, *pl("gate"), *pl("up"), *pl("down"))


def _pairs(P, key=0):
    k = jax.random.PRNGKey(key)
    xs = jax.random.normal(k, (P, D)).astype(jnp.bfloat16)
    es = jnp.sort(jax.random.randint(jax.random.fold_in(k, 1), (P,), 0, E + 1)).astype(jnp.int32)
    ws = jax.random.uniform(jax.random.fold_in(k, 2), (P,))
    return xs, es, ws


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ROWS.clear()
    monkeypatch.setattr(B, "rows", lambda: [dict(r) for r in ROWS], raising=False)
    monkeypatch.delenv("MASHBOX_GGUF_KERNELS", raising=False)
    yield
    ops.load_kernels(module="off")


def _v6e(monkeypatch, tmp_path, libtpu="0.0.48.1"):
    """a v6e host with no kernel module installed: chip family and libtpu as a v6e host reports them, the cache under
    tmp_path. The payloads stay cpu ones (the bundle runs the payloads of the backend it is on)."""
    from mashbox import kernels as K
    monkeypatch.setattr(K, "chip_family", lambda: "v6e")
    monkeypatch.setattr(K, "versions", lambda: {"jax": jax.__version__, "jaxlib": None, "libtpu": libtpu})
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == ops.KERNEL_MODULE else _FIND(n, *a))
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache" / "kernels"


def test_zero_flags_load_the_catalog_rows_wheel_once_into_the_cache(tmp_path, monkeypatch):
    p = _build(tmp_path / "rel")
    cache = _v6e(monkeypatch, tmp_path)
    got = ops.load_kernels()
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert ops._KERNELS["error"] is None and 12 in got
    assert ops.kernel_package() == {"name": str(cache / sha[:16] / p.name), "source": "catalog", "version": "0.0.1",
                                    "sha256": sha, "error": None}
    assert (cache / sha[:16] / p.name).read_bytes() == p.read_bytes()
    ROWS[0]["url"] = (tmp_path / "gone.whl").as_uri()         # the next start reads the cache, no fetch
    ops.load_kernels()
    assert ops.kernel_package()["source"] == "catalog" and ops._KERNELS["error"] is None


def test_a_download_whose_sha256_is_not_the_rows_is_refused_and_not_cached(tmp_path, monkeypatch):
    p = _build(tmp_path / "rel")
    q = _build(tmp_path / "bad", tamper=True, row=False)
    ROWS[0]["url"] = q.as_uri()
    cache = _v6e(monkeypatch, tmp_path)
    assert ops.load_kernels() == frozenset()
    assert ops._KERNELS["module"] is None and "sha256" in ops._KERNELS["error"]
    assert not (cache / ROWS[0]["sha256"][:16] / p.name).exists()


def test_a_tampered_wheel_with_a_self_consistent_manifest_is_refused(tmp_path, monkeypatch):
    _build(tmp_path / "rel")
    q = _build(tmp_path / "bad", tamper=True, row=False)
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", str(q))
    assert ops.load_kernels() == frozenset()
    assert ops._KERNELS["module"] is None and "not an engine_kernels row" in ops._KERNELS["error"]
    with pytest.raises(ValueError, match="not an engine_kernels row"):
        B.Bundle(q)


@pytest.mark.parametrize("libtpu", ["0.0.47.1", None])
def test_a_libtpu_outside_the_rows_pattern_is_refused(tmp_path, monkeypatch, libtpu):
    _build(tmp_path / "rel")
    _v6e(monkeypatch, tmp_path, libtpu)
    assert ops.load_kernels() == frozenset()
    assert ops._KERNELS["module"] is None and "libtpu" in ops._KERNELS["error"]


def test_off_tpu_zero_flags_fetch_nothing_and_run_exact(tmp_path, monkeypatch):
    _build(tmp_path / "rel")
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == ops.KERNEL_MODULE else _FIND(n, *a))
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    assert ops.load_kernels() == frozenset() and ops._KERNELS["module"] is None
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("form", ["absolute", "parent", "slash", "backslash"])
def test_a_row_file_that_is_not_a_bare_name_is_refused_and_nothing_is_written_outside_the_cache(tmp_path, monkeypatch,
                                                                                                form):
    p = _build(tmp_path / "rel")
    out = tmp_path / "outside"
    f = {"absolute": str(out / p.name), "parent": f"../../../outside/{p.name}", "slash": f"sub/{p.name}",
         "backslash": f"sub\\{p.name}"}[form]
    ROWS[0]["file"] = f
    cache = _v6e(monkeypatch, tmp_path)
    assert ops.load_kernels() == frozenset()
    assert ops._KERNELS["module"] is None and "bare file name" in ops._KERNELS["error"]
    assert not out.exists()
    assert not any(x.is_file() for x in tmp_path.rglob("*.whl*") if x.parent != p.parent)


def test_a_relative_cache_root_keeps_the_wheel_inside_its_digest_directory(tmp_path, monkeypatch):
    p = _build(tmp_path / "rel")
    _v6e(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MASHBOX_CACHE", "relcache")
    ops.load_kernels()
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    want = tmp_path / "relcache" / "kernels" / sha[:16] / p.name
    assert ops._KERNELS["error"] is None and want.read_bytes() == p.read_bytes()
    assert ops.kernel_package()["name"] == str(want)
    ROWS[0]["file"] = f"../{p.name}"
    ops.load_kernels()
    assert ops._KERNELS["module"] is None and "bare file name" in ops._KERNELS["error"]
    assert not (tmp_path / "relcache" / "kernels" / p.name).exists()


def test_a_row_sha256_that_is_not_64_hex_is_refused_before_any_path_is_built(tmp_path, monkeypatch):
    p = _build(tmp_path / "rel")
    ROWS[0]["sha256"] = "../../outside" + "0" * 51           # its first 16 characters name tmp_path/outside000
    _v6e(monkeypatch, tmp_path)
    assert ops.load_kernels() == frozenset()
    assert ops._KERNELS["module"] is None and "sha256" in ops._KERNELS["error"]
    assert not (tmp_path / "outside000").exists() and p.exists()


def test_the_wheel_is_the_kernel_package_and_its_expert_ffn_runs(tmp_path):
    p = _build(tmp_path)
    got = ops.load_kernels(module=str(p))
    assert 12 in got and ops._KERNELS["error"] is None
    m = ops._KERNELS["module"]
    assert m.EXPERT_ACTS == ("silu",) and m.EXPERT_TYPES == frozenset({12}) and m.TYPES == frozenset()
    W, (xs, es, ws) = _W(jax.random.PRNGKey(3)), _pairs(8)
    y = m.expert_ffn(xs, es, ws, W, "silu", E)
    assert np.array_equal(np.asarray(y), np.asarray(_ref(xs, es, ws, W)))


def test_the_record_names_the_wheel_by_its_own_sha256_and_is_not_a_dev_build(tmp_path):
    p = _build(tmp_path)
    ops.load_kernels(module=str(p))
    k = ops._KERNELS["module"].EXPERT_KERNEL
    assert k["dev_build"] is False and k["package_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert k["id"] == RX.KERNEL_ID and ops.KERNEL_MODULE == PKG


def test_a_call_the_manifest_does_not_list_is_refused_by_name(tmp_path):
    m = B.Bundle(_build(tmp_path, P_list=(8,)))
    xs, es, ws = _pairs(16)
    with pytest.raises(KeyError):
        m.expert_ffn(xs, es, ws, _W(jax.random.PRNGKey(0)), "silu", E)
    with pytest.raises(KeyError):
        m.expert_ffn(*_pairs(8), _W(jax.random.PRNGKey(0)), "relu2", E)


def test_ops_takes_the_exact_path_with_a_warning_for_an_unlisted_call(tmp_path, capsys):
    ops.load_kernels(module=str(_build(tmp_path, P_list=(8,))))
    m = ops._KERNELS["module"]
    xs, es, ws = _pairs(16)
    W = _W(jax.random.PRNGKey(1))
    assert ops._expert_kernel_error(m, (xs, es, ws, W), "silu", E) is not None


def test_a_payload_whose_bytes_do_not_match_the_manifest_is_refused(tmp_path):
    m = B.Bundle(_build(tmp_path, corrupt="ffn_p8"))
    with pytest.raises(ValueError, match="sha256"):
        m.expert_ffn(*_pairs(8), _W(jax.random.PRNGKey(0)), "silu", E)


def test_a_bundle_for_another_jax_is_refused_and_every_type_runs_exact(tmp_path):
    got = ops.load_kernels(module=str(_build(tmp_path, jaxv="0.0.0")))
    assert got == frozenset() and "jax 0.0.0" in ops._KERNELS["error"]


def test_planes_keep_their_stack_and_the_repack_and_dequant_payloads_route(tmp_path):
    m = B.Bundle(_build(tmp_path))
    raw = np.arange(E * F * D, dtype=np.uint8).reshape(E, F, D)
    pl = m.repack_nb_host(raw, 12, (F, D))
    assert isinstance(pl, B.Planes) and pl.lead == (E,) and sorted(pl.planes) == ["hi", "lo"]
    assert np.array_equal(np.asarray(m.nb_dequant(pl)), raw.astype(np.float32) + 2.0)
    assert m.nb_types() == [12] and m.nb_bytes(12, (E, F, D)) == 1234
    with pytest.raises(KeyError):
        m.repack_nb_host(raw[:2], 12, (F, D))
    one = jax.tree_util.tree_map(lambda a: a[0], pl)
    assert isinstance(one, B.Planes) and one.lead == ()


def test_scratch_comes_from_the_manifest_else_the_stated_fallback(tmp_path):
    m = B.Bundle(_build(tmp_path, P_list=(8, 32)))
    assert m.expert_ffn_scratch(8, D, F, E) == 800 and m.expert_ffn_scratch(9, D, F, E) == 3200
    assert m.expert_ffn_scratch(64, D, F, E) == 64 * 6 * D
