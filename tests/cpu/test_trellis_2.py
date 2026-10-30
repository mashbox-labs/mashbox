"""TRELLIS.2 on CPU: the call contract of the served 512 pipeline on tiny synthetic models (no downloads, no weights).

Covers: the two sparse-token buckets and the refusal above them, padding never reaching the outputs, weights staying
arguments of the compiled programs, the vendor CPU noise stream, the tiled decoder against the unchanged port,
the host GLB export, the pins (and, for maintainers, the records they come from), the command line refusing a bad
input before any download, and `mashbox run trellis-2` end to end on tiny models in the upstream file layout.
"""
import hashlib
import json
import re
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from mashbox import cli
from mashbox.models import trellis_2 as T2
from mashbox.models.trellis_2 import adapters as A, app, weights as W
from mashbox.models.trellis_2.adapters import BUCKETS, Flow, NoiseStream, choose_bucket, import_port, pack
from mashbox.models.trellis_2.driver import EXPORT, PRECISION, Job, Trellis2Driver
from mashbox.models.trellis_2.flow.schedule import SamplerParams

M = import_port()
PKG = Path(app.__file__).resolve().parent


def tiny_cfg(**kw):
    dt = M["dit_trunk"]
    return dt.DiTConfig(**dict(dict(in_channels=8, out_channels=8, model_channels=24, cond_channels=16, num_blocks=1,
                                    num_heads=2, mlp_ratio=2, qk_rms_norm=False, qk_rms_norm_cross=False, dtype="float32"), **kw))


def tiny_params(cfg):
    rng = np.random.default_rng(7)
    d, c, f = cfg.model_channels, cfg.cond_channels, cfg.mlp_hidden
    rand = lambda shape: jnp.asarray(rng.normal(0, .05, shape).astype(np.float32))
    p = {name: rand(shape) for name, shape in {"t0_w": (256, d), "t0_b": (d,), "t2_w": (d, d), "t2_b": (d,),
         "ada_w": (d, 6 * d), "ada_b": (6 * d,), "in_w": (cfg.in_channels, d), "in_b": (d,),
         "out_w": (d, cfg.out_channels), "out_b": (cfg.out_channels,)}.items()}
    shapes = {"mod": (6 * d,), "qkv_w": (d, 3 * d), "qkv_b": (3 * d,), "sa_o_w": (d, d), "sa_o_b": (d,), "n2_w": (d,),
              "n2_b": (d,), "q_w": (d, d), "q_b": (d,), "kv_w": (c, 2 * d), "kv_b": (2 * d,), "ca_o_w": (d, d),
              "ca_o_b": (d,), "fc1_w": (d, f), "fc1_b": (f,), "fc2_w": (f, d), "fc2_b": (d,)}
    p["blocks"] = {k: rand((cfg.num_blocks, *shape)) for k, shape in shapes.items()}
    p["blocks"]["n2_w"] = jnp.ones_like(p["blocks"]["n2_w"])
    return p


def test_buckets_and_refusal_above_the_largest():
    assert BUCKETS == (8192, 16384)
    assert [choose_bucket(n) for n in (1, 8192, 8193, 16384)] == [8192, 8192, 16384, 16384]
    for n in (0, 16385, 32768):
        with pytest.raises(ValueError):
            choose_bucket(n)


def test_served_defaults():
    assert EXPORT == dict(decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0)
    assert PRECISION["flow_trunk"] == "bf16"
    assert app.make_driver("/tmp/unused-trellis-cache").pipeline == "512"


def test_driver_refuses_rgb():
    d = Trellis2Driver(cache_dir="/tmp/unused-trellis-cache", pipeline="512")
    d.loaded = True
    with pytest.raises(ValueError, match="RGBA required"):
        d.run_stages(Image.new("RGB", (16, 16)))


def test_padding_never_reaches_outputs_and_weights_stay_arguments():
    cfg = tiny_cfg()
    p = tiny_params(cfg)
    flow = Flow(M["dit_trunk"], cfg, SamplerParams(steps=2, guidance_strength=1))
    coords = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], np.int32)
    x, c, m = pack(coords, np.ones((3, 8), np.float32), cap=8)
    cond = jnp.ones((1, 9, 16), jnp.float32)
    a = flow.run(p, x, cond, c, m)
    b = flow.run(p, x.at[:, 3:].set(1234.), cond, c, m)          # poisoned padding
    np.testing.assert_array_equal(np.asarray(a[:, :3]), np.asarray(b[:, :3]))
    changed = flow.run(dict(p, out_b=p["out_b"] + .5), x, cond, c, m)
    assert not np.array_equal(np.asarray(a[:, :3]), np.asarray(changed[:, :3]))
    jaxpr = jax.make_jaxpr(flow.program)(*flow.arguments(p, x, cond, c, m))
    assert not any(getattr(v, "size", 0) > 100 for v in jaxpr.consts), "weights or per-job buffers captured as constants"
    assert len(jaxpr.jaxpr.invars) > len(jax.tree_util.tree_leaves(p))
    assert a.dtype == jnp.float32


def _reset_jax_compile_cache():
    """JAX ties its persistent compile cache, once per process, to the first non-empty jax_compilation_cache_dir set
    when something compiles. Resetting it lets a test's own directory take, and keeps later tests out of a deleted one."""
    from jax._src import compilation_cache
    compilation_cache.reset_cache()


def test_compiles_both_buckets_into_the_cache():
    cfg = tiny_cfg()
    p = tiny_params(cfg)
    flow = Flow(M["dit_trunk"], cfg, SamplerParams(steps=1, guidance_strength=1))
    before = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    _reset_jax_compile_cache()
    try:
        with tempfile.TemporaryDirectory() as cache:
            jax.config.update("jax_compilation_cache_dir", cache)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            rows = [flow.compile(p, 9, b) for b in BUCKETS]
            assert [r["bucket"] for r in rows] == list(BUCKETS)
            assert set(flow.compiled) == set(BUCKETS)
            assert list(Path(cache).rglob("*"))
    finally:                                    # later tests must not write into the deleted directory
        jax.config.update("jax_compilation_cache_dir", before[0])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before[1])
        _reset_jax_compile_cache()


def test_vendor_noise_stream():
    import torch
    shapes = [("S2", (1, 8, 2, 2, 2)), ("S4", (3, 32)), ("S7", (7, 32))]
    manual = torch.Generator(device="cpu").manual_seed(42)
    stream = NoiseStream(42)
    for stage, shape in shapes:
        np.testing.assert_array_equal(stream.normal(stage, shape),
                                      torch.randn(shape, generator=manual, dtype=torch.float32).numpy())
    override, original = NoiseStream(42, {"S2": np.zeros(shapes[0][1], np.float32)}), NoiseStream(42)
    assert np.all(override.normal(*shapes[0]) == 0)
    original.normal(*shapes[0])
    np.testing.assert_array_equal(override.normal(*shapes[1]), original.normal(*shapes[1]))


def _decoder_params(out_channels, pred_subdiv):
    rng = np.random.default_rng(106)
    p = {}
    rand = lambda shape: jnp.asarray(rng.normal(0, .04, shape).astype(np.float32))
    def linear(name, ci, co):
        p[name + ".weight"], p[name + ".bias"] = rand((co, ci)), rand((co,))
    def norm(name, c):
        p[name + ".weight"], p[name + ".bias"] = jnp.ones(c), jnp.zeros(c)
    def conv(name, ci, co):
        p[name + ".wk"], p[name + ".bias"] = rand((27, ci, co)), rand((co,))
    linear("from_latent", 8, 16)
    for level in range(2):
        name = f"blocks.{level}.0"
        conv(name + ".conv", 16, 16); norm(name + ".norm", 16)
        linear(name + ".mlp.0", 16, 64); linear(name + ".mlp.2", 64, 16)
    name = "blocks.0.1"
    norm(name + ".norm1", 16)
    conv(name + ".conv1", 16, 128); conv(name + ".conv2", 16, 16)
    if pred_subdiv:
        linear(name + ".to_subdiv", 16, 8)
        p[name + ".to_subdiv.bias"] = jnp.array([-.1, .1] * 4, jnp.float32)
    linear("output_layer", 16, out_channels)
    return p


@pytest.mark.slow
def test_tiled_decoder_equals_the_unchanged_port():
    """Slow (~1 min on 4 cores): both decoders at 8,193 input rows grow past 16,384 rows over two buckets."""
    from mashbox.models.trellis_2.decoder_tiles import CAPS, TiledDecoder
    S = M["sparse_conv_jax"]
    rng = np.random.default_rng(17)
    coords = np.stack(np.unravel_index(np.arange(8193), (32, 32, 32)), 1).astype(np.int32)
    x = rng.normal(size=(8193, 8)).astype(np.float32)
    guide = None
    for p, pred in ((_decoder_params(7, True), True), (_decoder_params(6, False), False)):
        ref_dec = S.SparseUnetDecoderJax(p, [16, 16], [1, 1], pred, quantum=256)
        dec = TiledDecoder(S, p, [16, 16], [1, 1], pred)
        assert {r["bucket"] for r in dec.warm_up()} == set(CAPS)
        before = set(dec.executables)
        ref, got = ref_dec.run(x, coords, 32, guide_subs=guide), dec.run(x, coords, 32, guide_subs=guide)
        assert set(dec.executables) == before                       # no compile after warm-up
        np.testing.assert_array_equal(got["coords"], ref["coords"])
        assert got["counts"] == ref["counts"] and got["counts"][-1] > 16384
        np.testing.assert_allclose(got["h"], ref["h"], rtol=3e-6, atol=3e-6)
        for (gc, gl), (rc, rl) in zip(got["subs"], ref["subs"]):
            np.testing.assert_array_equal(gc, rc)
            np.testing.assert_array_equal(gl > 0, rl > 0)
        if pred:
            guide = [v > 0 for _, v in ref["subs"]]
    dec.p["output_layer.bias"] = dec.p["output_layer.bias"] + jnp.float32(.25)   # weights stay arguments
    np.testing.assert_allclose(dec.run(x, coords, 32, guide_subs=guide)["h"], got["h"] + .25, rtol=3e-6, atol=3e-6)


def test_host_glb_export_of_a_box(tmp_path):
    import trimesh
    d = Trellis2Driver(cache_dir=str(tmp_path / "cache"), pipeline="512", wiring_texture_size=64)
    mesh = trimesh.creation.box(extents=(.5, .5, .5))
    xyz = np.stack(np.meshgrid(*(np.arange(8),) * 3, indexing="ij"), -1).reshape(-1, 3).astype(np.int32)
    attrs = np.tile(np.array([.6, .2, .1, 0., .7, 1.], np.float32), (len(xyz), 1))
    events = []
    res = d.export_glb({"vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces, np.int64),
                        "coords": xyz, "attr_volume": attrs, "resolution": 8, "job": Job(42, "512", "reduced", events.append)},
                       tmp_path / "box.glb")
    scene = trimesh.load(res["path"], force="scene")
    assert sum(len(g.faces) for g in scene.geometry.values()) > 0
    _assert_textured(scene)
    assert res["metadata"]["export"]["remesh"] and res["metadata"]["chip_time_s"] is None
    assert [e["phase"] for e in events] == ["start", "ran"]
    assert re.fullmatch(r"[0-9a-f]{64}", res["metadata"]["sha256"])


def test_load_holds_each_file_to_its_pin_not_to_the_callers_value(tmp_path, monkeypatch):
    """The loader compares every file with its sha256 in upstream.json itself. The caller's map only says which files
    were checked: a file altered on disk and passed with its own new sha256 is refused, so is a file with no pin, and
    the not-checked marker is taken only when MASHBOX_VERIFY=0 is set."""
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("MASHBOX_VERIFY", raising=False)
    root = Path(W.root()).resolve()
    pins = _stage_tiny_upstream(root)
    monkeypatch.setattr(W, "pins", lambda: pins)
    d = Trellis2Driver(cache_dir=str(tmp_path / "xla"), pipeline="512")
    d._resolve(root)
    f = d.pipeline_path
    pin = next(x["sha256"] for x in pins["repos"]["trellis2"]["files"] if x["name"] == "pipeline.json")
    assert d._given(root, [f], {str(f): pin})[0]["sha256"] == pin
    with pytest.raises(ValueError, match="not checked before load"):
        d._given(root, [f], {})
    with pytest.raises(ValueError, match="MASHBOX_VERIFY=0"):
        d._given(root, [f], {str(f): W.UNCHECKED})
    f.write_bytes(f.read_bytes() + b" ")                         # changed on disk after the download
    with pytest.raises(ValueError, match="not its pin"):
        d._given(root, [f], {str(f): hashlib.sha256(f.read_bytes()).hexdigest()})
    with pytest.raises(SystemExit, match="does not match"):
        d._given(root, [f], {str(f): pin})
    stray = root / "stray.json"
    stray.write_text("{}")
    with pytest.raises(ValueError, match="no pin"):
        d._given(root, [stray], {str(stray): hashlib.sha256(b"{}").hexdigest()})
    monkeypatch.setenv("MASHBOX_VERIFY", "0")
    assert d._given(root, [f], {str(f): W.UNCHECKED})[0]["sha256"] == W.UNCHECKED
    with pytest.raises(ValueError, match="sha256"):
        d.load(tmp_path)


def test_pins():
    p = W.pins()
    names = set()
    for r in p["repos"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", r["revision"])
        for f in r["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["bytes"] > 0
            names.add(f["name"])
    need = {"pipeline.json", "config.json", "model.safetensors", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"}
    for m in ("ss_flow_img_dit_1_3B_64_bf16", "slat_flow_img2shape_dit_1_3B_512_bf16",
              "slat_flow_imgshape2tex_dit_1_3B_512_bf16", "shape_dec_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16"):
        need |= {f"ckpts/{m}.json", f"ckpts/{m}.safetensors"}
    assert need <= names
    assert not any("1024" in n and n.endswith(".safetensors") for n in names)   # the 512 pipeline only
    assert p["repos"]["dinov3"]["gated"] is True


def test_gated_download_says_what_to_do(monkeypatch):
    from huggingface_hub.errors import GatedRepoError
    from mashbox import hub
    def gated(*a, **k):
        raise GatedRepoError.__new__(GatedRepoError)
    monkeypatch.setattr(hub, "fetch", gated)
    with pytest.raises(SystemExit) as e:
        W._download(W.pins()["repos"]["dinov3"], "model.safetensors", "/tmp/unused")
    assert all(w in str(e.value) for w in ("Accept its terms", "HF_TOKEN", "hf auth login"))


@pytest.mark.parametrize("mode,alpha,why", [("RGB", None, "needs an RGBA image"), ("RGBA", 255, "alpha channel must mark"),
                                            ("RGBA", 254, "alpha channel must mark"), ("RGBA", 0, "alpha channel must mark")])
def test_cli_refuses_a_bad_image_before_any_download(tmp_path, monkeypatch, capsys, mode, alpha, why):
    monkeypatch.setattr(W, "stage", lambda *a, **k: pytest.fail("download started for a bad input"))
    im = Image.new(mode, (32, 32))
    if alpha is not None:
        im.putalpha(alpha)
    im.save(tmp_path / "in.png")
    with pytest.raises(SystemExit, match=why):
        T2.main("run", ["--image", str(tmp_path / "in.png"), "--out", str(tmp_path / "out.glb")])


def test_cli_refuses_an_object_too_small_to_crop_before_any_download(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "stage", lambda *a, **k: pytest.fail("download started for a bad input"))
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[10, 12] = (200, 120, 40, 255)          # one opaque pixel: nothing to crop around
    Image.fromarray(rgba, "RGBA").save(tmp_path / "in.png")
    with pytest.raises(SystemExit, match="too small"):
        T2.main("run", ["--image", str(tmp_path / "in.png"), "--out", str(tmp_path / "out.glb")])


def test_stale_cached_revisions_never_change_what_loads(tmp_path, monkeypatch):
    """Another revision of the same repos in the cache (an older download) never replaces the pinned files."""
    import shutil
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    root = Path(W.root())
    pins = _stage_tiny_upstream(root)
    monkeypatch.setattr(W, "pins", lambda: pins)
    for r in pins["repos"].values():
        shutil.copytree(W.repo_dir(r), W.repo_dir(dict(r, revision="0" * 40)))
    d = Trellis2Driver(cache_dir=str(tmp_path / "xla"), pipeline="512")
    d._resolve(root.resolve())
    pinned = [Path(W.repo_dir(r)).resolve() for r in pins["repos"].values()]
    for path in [d.pipeline_path, d.dino_path] + list(d.model_paths.values()):
        assert any(Path(path).resolve().is_relative_to(q) for q in pinned), path


def test_a_stale_flat_cache_never_changes_what_loads(tmp_path, monkeypatch):
    """An older flat layout (<root>/pipeline.json + <root>/ckpts, <root>/dinov3) beside the pinned revision directories
    is never read: the loader takes the pinned directories only."""
    import shutil
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    root = Path(W.root())
    pins = _stage_tiny_upstream(root)
    monkeypatch.setattr(W, "pins", lambda: pins)
    shutil.copytree(W.repo_dir(pins["repos"]["trellis2"]), root, dirs_exist_ok=True)
    shutil.copytree(W.repo_dir(pins["repos"]["dinov3"]), root / "dinov3")
    assert (root / "pipeline.json").is_file() and (root / "dinov3" / "config.json").is_file()
    d = Trellis2Driver(cache_dir=str(tmp_path / "xla"), pipeline="512")
    d._resolve(root.resolve())
    pinned = [Path(W.repo_dir(r)).resolve() for r in pins["repos"].values()]
    for path in [d.pipeline_path, d.dino_path] + list(d.model_paths.values()):
        assert any(Path(path).resolve().is_relative_to(q) for q in pinned), path


def _assert_textured(scene):
    """A textured GLB: every mesh has one UV per vertex and a material with a base colour image."""
    assert scene.geometry
    for g in scene.geometry.values():
        v = g.visual
        assert getattr(v, "kind", None) == "texture" and v.uv is not None and len(v.uv) == len(g.vertices)
        tex = getattr(v.material, "baseColorTexture", None)
        assert tex is not None and min(tex.size) > 0


def test_textured_check_rejects_a_plain_mesh():
    import trimesh
    with pytest.raises(AssertionError):
        _assert_textured(trimesh.Scene(trimesh.creation.box()))


def test_cli_runs_trellis_2_from_its_catalog_row_and_the_package_parses_its_options(capsys):
    """trellis-2's models.json row names the package and is not launch-gated, so `mashbox run trellis-2` reaches the
    package's own command line through MODEL_CLI, which is read from the catalog (no model id typed into cli.py), and
    that command line parses its options."""
    root = Path(__file__).resolve().parents[2]
    row = next(m for m in json.loads((root / "models.json").read_text())["models"] if m["id"] == "trellis-2")
    assert "launch" not in row and row["path"] == "src/mashbox/models/trellis_2"
    assert "trellis-2" in cli.RUNNABLE and cli.MODEL_CLI.get("trellis-2") == "mashbox.models.trellis_2"
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "trellis-2", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--image" in out and "--out" in out


def test_package_names_no_internal_paths_or_native_library():
    """No native library, no kernel the package does not ship, no pointer to a file it does not ship, no note to self."""
    text = "\n".join(p.read_text() for p in PKG.rglob("*.py"))
    low = text.lower()
    for bad in ("ctypes", "libmeshopt", "sys.path", "pallas", "see readme", "licence-pin", "license-pin", "wiring only",
                "to be validated", "until measured"):
        assert bad not in low, bad
    assert "NOT RUN" not in text
    assert not re.search(r"\.cuh?\b", text), "names a CUDA source file"
    assert json.loads((PKG / "upstream.json").read_text())["code"]["license"] == "MIT"


def test_summary_gives_chip_memory_in_decimal_gb(tmp_path, monkeypatch):
    """device_peak_gb is bytes / 1e9, the same GB as the download size, never GiB under a GB name."""
    class Dev:
        def memory_stats(self):
            return {"peak_bytes_in_use": 14_420_000_000}

    class Drv:
        cache_dir, pipeline = str(tmp_path), "512"

        def load(self, root, verified, progress):
            return {"compile_ms": 1.0}

        def generate(self, image, out, seed, progress):
            return {"path": out, "metadata": {"sha256": "0" * 64, "bytes": 1, "glb": {"vertices": 3, "faces": 1},
                                              "backend": "cpu", "stage_wall_s": {"S0": 0.1},
                                              "device_stage_wall_s": 0.0, "host_time_s": 0.1}}

        def unload(self):
            pass
    monkeypatch.setattr(jax, "devices", lambda: [Dev()])
    out = app.generate(Drv(), str(tmp_path), {}, None, str(tmp_path / "o.glb"), 42, progress=lambda s: None)
    assert out["device_peak_gb"] == 14.42


WORDS = Path(__file__).resolve().parents[2] / "internal" / "trellis-2-words.txt"


@pytest.mark.skipif(not WORDS.is_file(), reason="maintainer check: the word list is held from the public tree")
def test_package_has_no_internal_names():
    text = "\n".join(p.read_text() for p in PKG.rglob("*.py"))
    words = [w for w in WORDS.read_text().splitlines() if w.strip() and not w.startswith("#")]
    assert len(words) >= 7
    for bad in words:
        assert bad not in text, bad


RECORD = Path(__file__).resolve().parents[2] / "internal" / "trellis-2-record.json"


def _code_only(src):
    """The source with comments and docstrings dropped, trailing spaces and blank lines removed: the same text on every
    Python version (ast.unparse is not; it writes some f-strings differently between 3.12 patch releases). Only the
    comment and docstring characters go; code that shares a line with a docstring stays."""
    import ast
    import io
    import tokenize
    lines = src.splitlines()
    cut = {}                                    # line number -> (start, end) character spans to drop
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            cut.setdefault(tok.start[0], []).append((tok.start[1], tok.end[1]))
    col = lambda n, byte_col: len(lines[n - 1].encode()[:byte_col].decode())  # noqa: E731  (ast columns are UTF-8 bytes)
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            d = body[0]
            for n in range(d.lineno, d.end_lineno + 1):
                cut.setdefault(n, []).append((col(n, d.col_offset) if n == d.lineno else 0,
                                              col(n, d.end_col_offset) if n == d.end_lineno else len(lines[n - 1])))
    kept = []
    for n, line in enumerate(lines, 1):
        keep = [True] * len(line)
        for a, b in cut.get(n, ()):
            keep[a:b] = [False] * (b - a)
        kept.append("".join(c for c, k in zip(line, keep) if k).rstrip())
    return "\n".join(line for line in kept if line)


def served_code_sha256():
    """sha256 of the package's code: each .py file without comments and docstrings (so those may change, code may
    not), and every other package file by its bytes."""
    h = hashlib.sha256()
    for f in sorted(p for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        blob = _code_only(f.read_text()).encode() if f.suffix == ".py" else f.read_bytes()
        h.update(f.relative_to(PKG).as_posix().encode() + b"\0" + blob + b"\0")
    return h.hexdigest()


def test_code_digest_sees_code_on_a_docstring_line():
    """A statement after a one-line docstring on the same line is code, so the digest must change with it (an earlier digest read
    `\"\"\"doc\"\"\"; return 7` the same as an empty body); a docstring or a comment alone still drops."""
    empty = 'def value():\n    """doc"""\n'
    code = 'def value():\n    """doc"""; return 7\n'
    ns = {}
    exec(code, ns)
    assert ns["value"]() == 7
    assert _code_only(code) != _code_only(empty)
    assert _code_only(empty) == _code_only('def value():\n    """other doc"""  # note\n') == "def value():"
    assert _code_only('def f():\n    """a\n    b"""; x = 1\n    return x\n') == "def f():\n; x = 1\n    return x"


@pytest.mark.skipif(not RECORD.is_file(), reason="maintainer check: the chip record pin is held from the public tree")
def test_package_code_is_the_code_stated_against_the_chip_record():
    """A code change after the chip record was sealed fails here until the record comparison states it and the pin
    moves with it."""
    assert served_code_sha256() == json.loads(RECORD.read_text())["package_code_sha256"]


def test_sha256_check_off_is_never_reported_as_checked(tmp_path, monkeypatch):
    """MASHBOX_VERIFY=0 skips the sha256 check; the loader then gets a marker that says so, never the pin as if checked."""
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path))
    body = b"{}"
    pin = hashlib.sha256(body).hexdigest()
    repo = {"hf": "o/m", "revision": "0" * 40, "files": [{"name": "pipeline.json", "bytes": len(body), "sha256": pin}]}
    monkeypatch.setattr(W, "pins", lambda: {"repos": {"m": repo}})
    Path(W.repo_dir(repo)).mkdir(parents=True)
    (Path(W.repo_dir(repo)) / "pipeline.json").write_bytes(body)
    assert list(W.stage(log=lambda s: None)[1].values()) == [pin]
    monkeypatch.setenv("MASHBOX_VERIFY", "0")
    lines = []
    unchecked = W.stage(log=lines.append)[1]
    assert pin not in unchecked.values()
    assert list(unchecked.values()) == [W.UNCHECKED] and any("MASHBOX_VERIFY=0" in s for s in lines)


# ---- `mashbox run trellis-2` end to end on CPU, on tiny synthetic models in the upstream file layout ----------------
TINY_DINO = dict(hidden=16, heads=2, layers=1, mlp=32, patch=16, registers=4, eps=1e-5, rope_theta=100.0)
OCC_BIAS = -1.2         # out bias of the tiny sparse-structure decoder: a few hundred occupied voxels, inside one bucket


def _r(rng, shape, std=.05):
    return rng.normal(0, std, shape).astype(np.float32)


def _ones(n):
    return np.ones(n, np.float32)


def _dit_files(rng, name, args):
    """A flow model as upstream ships it: <prefix>.json + <prefix>.safetensors with the torch parameter names."""
    d, c, i, o = args["model_channels"], args["cond_channels"], args["in_channels"], args["out_channels"]
    f = int(d * args["mlp_ratio"])
    shapes = {"t_embedder.mlp.0.weight": (d, 256), "t_embedder.mlp.0.bias": (d,), "t_embedder.mlp.2.weight": (d, d),
              "t_embedder.mlp.2.bias": (d,), "adaLN_modulation.1.weight": (6 * d, d), "adaLN_modulation.1.bias": (6 * d,),
              "input_layer.weight": (d, i), "input_layer.bias": (d,), "out_layer.weight": (o, d), "out_layer.bias": (o,)}
    block = {"modulation": (6 * d,), "self_attn.to_qkv.weight": (3 * d, d), "self_attn.to_qkv.bias": (3 * d,),
             "self_attn.to_out.weight": (d, d), "self_attn.to_out.bias": (d,), "norm2.weight": (d,), "norm2.bias": (d,),
             "cross_attn.to_q.weight": (d, d), "cross_attn.to_q.bias": (d,), "cross_attn.to_kv.weight": (2 * d, c),
             "cross_attn.to_kv.bias": (2 * d,), "cross_attn.to_out.weight": (d, d), "cross_attn.to_out.bias": (d,),
             "mlp.mlp.0.weight": (f, d), "mlp.mlp.0.bias": (f,), "mlp.mlp.2.weight": (d, f), "mlp.mlp.2.bias": (d,)}
    shapes.update({f"blocks.{b}.{k}": s for b in range(args["num_blocks"]) for k, s in block.items()})
    sd = {k: _r(rng, s) for k, s in shapes.items()}
    sd.update({f"blocks.{b}.norm2.weight": _ones(d) for b in range(args["num_blocks"])})
    return {"name": name, "args": args}, sd


def _ss_decoder_files(rng, args):
    ch = args["channels"]
    sd = {}

    def conv(p, co, ci):
        sd[p + ".weight"], sd[p + ".bias"] = _r(rng, (co, ci, 3, 3, 3)), _r(rng, (co,))

    def norm(p, c):
        sd[p + ".weight"], sd[p + ".bias"] = _ones(c), np.zeros(c, np.float32)

    def res(p, c):
        norm(p + ".norm1", c); conv(p + ".conv1", c, c); norm(p + ".norm2", c); conv(p + ".conv2", c, c)
    conv("input_layer", ch[0], args["latent_channels"])
    for i in range(args["num_res_blocks_middle"]):
        res(f"middle_block.{i}", ch[0])
    i = 0
    for level, c in enumerate(ch):
        for _ in range(args["num_res_blocks"]):
            res(f"blocks.{i}", c); i += 1
        if level < len(ch) - 1:
            conv(f"blocks.{i}.conv", ch[level + 1] * 8, c); i += 1
    norm("out_layer.0", ch[-1]); conv("out_layer.2", args["out_channels"], ch[-1])
    sd["out_layer.2.bias"] = np.array([OCC_BIAS], np.float32)
    assert M["ss_decoder"].check_keys(sd, args) == ([], [])
    return {"name": "SparseStructureDecoder", "args": args}, sd


def _sparse_decoder_files(rng, name, out_channels, pred_subdiv, latent=8, c=16):
    """Two levels, one ConvNeXt block each, one subdividing up block (sparse conv weights stored [Co, 3, 3, 3, Ci])."""
    sd = {}

    def lin(p, ci, co):
        sd[p + ".weight"], sd[p + ".bias"] = _r(rng, (co, ci), .04), _r(rng, (co,), .04)

    def conv(p, ci, co):
        sd[p + ".weight"], sd[p + ".bias"] = _r(rng, (co, 3, 3, 3, ci), .04), _r(rng, (co,), .04)

    def norm(p):
        sd[p + ".weight"], sd[p + ".bias"] = _ones(c), np.zeros(c, np.float32)
    lin("from_latent", latent, c)
    for level in range(2):
        p = f"blocks.{level}.0"
        conv(p + ".conv", c, c); norm(p + ".norm"); lin(p + ".mlp.0", c, 4 * c); lin(p + ".mlp.2", 4 * c, c)
    p = "blocks.0.1"
    norm(p + ".norm1"); conv(p + ".conv1", c, 8 * c); conv(p + ".conv2", c, c)
    if pred_subdiv:
        lin(p + ".to_subdiv", c, 8)
        sd[p + ".to_subdiv.bias"] = np.array([-.1, .1] * 4, np.float32)
    lin("output_layer", c, out_channels)
    if pred_subdiv:                          # vertex offsets, then intersection flags (> 0 marks an edge), then weight
        sd["output_layer.bias"] = np.array([0, 0, 0, .5, .5, .5, 0], np.float32)
    args = {"model_channels": [c, c], "num_blocks": [1, 1], "latent_channels": latent, "out_channels": out_channels}
    return {"name": name, "args": args}, sd


def _save_safetensors(path, tensors):
    """The safetensors layout (u64 header size, JSON header, raw little-endian data), float32 only."""
    header, blobs, offset = {}, [], 0
    for k, a in tensors.items():
        b = np.ascontiguousarray(a, "<f4").tobytes()
        header[k] = {"dtype": "F32", "shape": list(np.shape(a)), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def _stage_tiny_upstream(root):
    """Write tiny models where `mashbox download trellis-2` puts the real ones; return the matching pins."""
    rng = np.random.default_rng(29)
    flow = dict(model_channels=24, cond_channels=TINY_DINO["hidden"], num_blocks=1, num_heads=2, mlp_ratio=2,
                pe_mode="rope", share_mod=True, qk_rms_norm=False, qk_rms_norm_cross=False, dtype="bfloat16")
    sampler = lambda rescale: {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-05},  # noqa: E731
                               "params": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": rescale,
                                          "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}}
    norm8 = {"mean": [0.] * 8, "std": [1.] * 8}
    pipeline = {"name": "Trellis2ImageTo3DPipeline", "args": {
        "models": {"sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec",
                   "sparse_structure_flow_model": "ckpts/ss_flow", "shape_slat_decoder": "ckpts/shape_dec",
                   "shape_slat_flow_model_512": "ckpts/shape_flow", "tex_slat_decoder": "ckpts/tex_dec",
                   "tex_slat_flow_model_512": "ckpts/tex_flow"},
        "sparse_structure_sampler": sampler(.7), "shape_slat_sampler": sampler(.5), "tex_slat_sampler": sampler(0.),
        "shape_slat_normalization": norm8, "tex_slat_normalization": norm8}}
    models = {
        ("trellis2", "ckpts/ss_flow"): _dit_files(rng, "SparseStructureFlowModel",
                                                  dict(flow, resolution=16, in_channels=8, out_channels=8)),
        ("trellis2", "ckpts/shape_flow"): _dit_files(rng, "SLatFlowModel", dict(flow, resolution=32, in_channels=8, out_channels=8)),
        ("trellis2", "ckpts/tex_flow"): _dit_files(rng, "SLatFlowModel", dict(flow, resolution=32, in_channels=16, out_channels=8)),
        ("trellis2", "ckpts/shape_dec"): _sparse_decoder_files(rng, "FlexiDualGridVaeDecoder", 7, True),
        ("trellis2", "ckpts/tex_dec"): _sparse_decoder_files(rng, "SparseUnetVaeDecoder", 6, False),
        ("image_large", "ckpts/ss_dec"): _ss_decoder_files(rng, dict(out_channels=1, latent_channels=8, num_res_blocks=1,
                                                                    num_res_blocks_middle=1, channels=[8, 8, 8])),
    }
    D = TINY_DINO["hidden"]
    dino = {"embeddings.patch_embeddings.weight": _r(rng, (D, 3, 16, 16)), "embeddings.patch_embeddings.bias": _r(rng, (D,)),
            "embeddings.cls_token": _r(rng, (1, 1, D)), "embeddings.register_tokens": _r(rng, (1, TINY_DINO["registers"], D))}
    F = TINY_DINO["mlp"]
    for k, s in {"norm1.weight": (D,), "norm1.bias": (D,), "norm2.weight": (D,), "norm2.bias": (D,),
                 "attention.q_proj.weight": (D, D), "attention.q_proj.bias": (D,), "attention.k_proj.weight": (D, D),
                 "attention.v_proj.weight": (D, D), "attention.v_proj.bias": (D,), "attention.o_proj.weight": (D, D),
                 "attention.o_proj.bias": (D,), "layer_scale1.lambda1": (D,), "layer_scale2.lambda1": (D,),
                 "mlp.up_proj.weight": (F, D), "mlp.up_proj.bias": (F,), "mlp.down_proj.weight": (D, F),
                 "mlp.down_proj.bias": (D,)}.items():
        dino[f"layer.0.{k}"] = _r(rng, s)
    dino_cfg = {"hidden_size": D, "num_attention_heads": TINY_DINO["heads"], "num_hidden_layers": 1, "intermediate_size": F,
                "patch_size": 16, "num_register_tokens": TINY_DINO["registers"], "layer_norm_eps": 1e-5,
                "rope_theta": 100.0, "hidden_act": "gelu", "use_gated_mlp": False, "key_bias": False}
    repos = {"trellis2": {"hf": "microsoft/TRELLIS.2-4B", "revision": "a" * 40},
             "image_large": {"hf": "microsoft/TRELLIS-image-large", "revision": "b" * 40},
             "dinov3": {"hf": "facebook/dinov3-vitl16-pretrain-lvd1689m", "revision": "c" * 40, "gated": True}}
    files = {"trellis2": {"pipeline.json": json.dumps(pipeline).encode()}, "image_large": {},
             "dinov3": {"config.json": json.dumps(dino_cfg).encode()}}
    for (repo, prefix), (cfg, sd) in models.items():
        files[repo][prefix + ".json"] = json.dumps(cfg).encode()
        files[repo][prefix + ".safetensors"] = sd
    files["dinov3"]["model.safetensors"] = dino
    for key, r in repos.items():
        r["files"] = []
        for name, data in files[key].items():
            path = root / (r["hf"].replace("/", "_") + "@" + r["revision"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _save_safetensors(path, data) if isinstance(data, dict) else path.write_bytes(data)
            blob = path.read_bytes()
            r["files"].append({"name": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    return {"repos": repos}


@pytest.mark.slow
def test_mashbox_run_trellis_2_end_to_end_on_cpu(tmp_path, monkeypatch, capsys):
    """Slow (~1-3 min on 4 cores): the real command (sha256 checks, load, compile, every stage, GLB export) on tiny
    models with 1 sampler step and a 64 px texture; only the model sizes differ from the served run."""
    import trimesh
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    pins = _stage_tiny_upstream(Path(W.root()))
    monkeypatch.setattr(W, "pins", lambda: pins)
    monkeypatch.setattr(M["dinov3_jax"], "CFG", dict(TINY_DINO))
    reduced = app.make_driver
    monkeypatch.setattr(app, "make_driver", lambda cache_dir: reduced(cache_dir, wiring_steps=1, wiring_texture_size=64))
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[16:48, 20:44] = (200, 120, 40, 255)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "in.png")
    glb, report = tmp_path / "out.glb", tmp_path / "run.json"
    before = {k: getattr(jax.config, k) for k in ("jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs",
                                                  "jax_default_matmul_precision")}
    _reset_jax_compile_cache()
    try:
        rc = T2.main("run", ["--image", str(tmp_path / "in.png"), "--out", str(glb),
                       "--cache-dir", str(tmp_path / "xla"), "--report", str(report)])
    finally:
        for k, v in before.items():
            jax.config.update(k, v)
        _reset_jax_compile_cache()
    assert rc == 0
    assert list((tmp_path / "xla").iterdir()), "the run's --cache-dir took no compiled program"
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert json.loads(report.read_text()) == out
    assert out["backend"] == "cpu" and out["pipeline"] == "512" and out["seed"] == 42
    assert set(out["stage_wall_s"]) == {"S0", "S1", "S2", "S3", "S4", "S7", "S8", "S9", "S10"}
    assert out["sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest() and out["bytes"] == glb.stat().st_size
    scene = trimesh.load(str(glb), force="scene")
    assert sum(len(g.faces) for g in scene.geometry.values()) > 0
    _assert_textured(scene)
    assert len(list((tmp_path / "cache" / "verified").iterdir())) == sum(len(r["files"]) for r in pins["repos"].values())


PIN_RECORDS = Path(__file__).resolve().parents[2] / "internal" / "trellis-2-pins"


def _recorded_pins():
    """(source path, sha256) pairs from our own pin records: a checkpoint list and three checksum lists."""
    rows = []

    def walk(o):
        if isinstance(o, dict):
            if "uri" in o and "sha256" in o:
                rows.append((o["uri"], o["sha256"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads((PIN_RECORDS / "checkpoints.json").read_text()))
    for name in ("large-files.sha256", "pin-folder.sha256", "small-files.sha256"):
        for line in (PIN_RECORDS / name).read_text().splitlines():
            a, b = line.split()[:2]
            rows.append((b, a) if re.fullmatch(r"[0-9a-f]{64}", a) else (a, b))
    return rows


@pytest.mark.skipif(not PIN_RECORDS.is_dir(), reason="maintainer check: the pin records are held from the public tree")
def test_pins_equal_their_records():
    """Every pinned file's sha256 is in a record of the same file name, and a record path carries its revision."""
    rows = _recorded_pins()
    for r in W.pins()["repos"].values():
        for f in r["files"]:
            base = f["name"].rsplit("/", 1)[-1]
            hits = [p for p, s in rows if s == f["sha256"] and p.rsplit("/", 1)[-1] == base]
            assert hits, (r["hf"], f["name"], "sha256 in no record")
            assert any(r["revision"][:8] in p for p in hits), (r["hf"], f["name"], "no record at this revision")
