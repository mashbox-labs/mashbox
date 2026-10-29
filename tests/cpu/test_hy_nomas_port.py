"""Hy-Nomas runtime on CPU: tiny models with the real architecture, the window loop, padding, LoRA merge, kernel dispatch,
decoder shapes, cameras, prompts, the command line and the app server against the stand-in engine."""
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hy_nomas_tiny as TINY  # noqa: E402
from mashbox.models.hy_nomas import buckets as B, cameras as C, decoder as DEC, dit as M, ops as O, prompts as PR  # noqa: E402
from mashbox.models.hy_nomas import runtime as RT, weights as W, kernels as KB  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = M.Arch(hidden=256, heads=2, head_dim=128, ff=512, layers=1, latent_ch=32, text_dim=32, cam_heads=2, cam_head_dim=64,
              frames=2, height=8, width=8)
BUCKETS = (32, 64)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("hyn")
    sd = TINY.dit_state(ARCH, seed=3)
    TINY.write_safetensors(str(d / "dit.safetensors"), sd)
    TINY.write_safetensors(str(d / "adapter.safetensors"), TINY.lora_state(sd, ARCH.layers, rank=4, seed=4))
    (d / "adapter_config.json").write_text(json.dumps(TINY.lora_config(4, 8.0)))
    return d


def runtime(ckpt, preset="quality", ops=None, buckets=BUCKETS):
    params, merge = W.dit_params(str(ckpt / "dit.safetensors"), str(ckpt / "adapter.safetensors"), str(ckpt / "adapter_config.json"), ARCH)
    assert merge["merged"] == merge["targets"] == 17 * ARCH.layers
    return RT.DiTRuntime(jax.devices()[:1], params, ops or O.Ops(None), arch=ARCH, preset=preset, sink=1, hist=2, buckets=buckets,
                         bq=128, bkv=128, log=lambda *a: None)


TEXTS = {"seg1": (21, None), "seg2": (25, dict(valid_post_crop=25, offs=[9, 25], win=[[0, 2, 0]], action_start=9)),
         "act": (30, dict(valid_post_crop=30, offs=[9, 18, 30], win=[[0, 1, 0], [1, 2, 1]], action_start=9))}


def session(rt, n=3, order=("seg1", "seg2", "act")):
    r = np.random.RandomState(9)
    items = {}
    for k, (n_tok, info) in TEXTS.items():
        t = (r.randn(40, ARCH.text_dim) * 0.5).astype(np.float32)
        items[k] = rt.make_item(k, t, np.arange(40) < n_tok, info)
    rt.reset(5)
    out = []
    for w in range(n):
        nf = 4 * ARCH.frames - 3 if w == 0 else 4 * ARCH.frames
        e = np.tile(np.eye(4, dtype=np.float32), (nf, 1, 1))
        e[:, :3, 3] += 0.01 * w
        x, T = rt.window(rt.cam_tokens(e), rt.tables(w), items[order[w]])
        out.append(np.asarray(x))
    return out, items


# ------------------------------------------------------------------ app files
def test_world_page_has_one_source():
    """The world page lives once, in the package (the installed app serves it); no second copy to drift."""
    pages = sorted(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/")
                   for top in ("apps", "src")
                   for d, _, fs in os.walk(os.path.join(ROOT, top)) for f in fs
                   if f == "page.html" and ("hy-nomas" in d or "hy_nomas" in d))
    assert pages == ["src/mashbox/models/hy_nomas/page.html"], pages


def test_upstream_pins():
    p = W.pins()
    assert p["repos"]["tencent"]["revision"] == "4b8b4ed8ac67bf7ed82a5b80e953af548d82bd8f"
    for r in p["repos"].values():
        for f in r["files"]:
            assert len(f["sha256"]) == 64 and f["bytes"] > 0


# ------------------------------------------------------------------ ops
def test_masked_attention_matches_dense():
    r = np.random.RandomState(0)
    H, D, LQ, LK, nv = 2, 128, 256, 384, 300
    q = (r.randn(LQ, H * D) * 0.3).astype(np.float32)
    k = (r.randn(LK, H * D) * 0.3).astype(np.float32)
    v = r.randn(LK, H * D).astype(np.float32)
    gq = r.randint(-1, 2, (LQ, 1)).astype(np.int32)
    gk = r.randint(-1, 2, (1, LK)).astype(np.int32)
    gk[0, :5] = -1
    bf = lambda x: np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))
    out = np.asarray(O.masked_attention(jnp.asarray(q, jnp.bfloat16), jnp.asarray(k, jnp.bfloat16), jnp.asarray(v, jnp.bfloat16),
                                        jnp.asarray(gq), jnp.asarray(gk), H=H, nvalid=nv, bq=128, bkv=128).astype(jnp.float32))
    vis = ((gk < 0) | (gk == gq)) & (np.arange(LK) < nv)[None]
    ref = np.zeros((LQ, H * D))
    for h in range(H):
        s = (bf(q)[:, h * D:(h + 1) * D] @ bf(k)[:, h * D:(h + 1) * D].T) * np.log(2)
        s = np.where(vis, s, -np.inf)
        p = np.exp(s - s.max(1, keepdims=True))
        ref[:, h * D:(h + 1) * D] = (p @ bf(v)[:, h * D:(h + 1) * D]) / p.sum(1, keepdims=True)
    assert rel(out, ref) < 1e-2


# ------------------------------------------------------------------ main model
@pytest.mark.parametrize("preset", ["quality", "live"])
def test_window_loop(ckpt, preset):
    rt = runtime(ckpt, preset)
    assert rt.steps == (3 if preset == "quality" else 4)
    out, items = session(rt)
    for x in out:
        assert x.shape == (32, ARCH.frames, ARCH.height, ARCH.width) and np.isfinite(x).all()
    assert rt.hk["q"].shape == (ARCH.layers, 2 * ARCH.frame_tokens + 32, ARCH.heads, 128)
    assert rt.prev == (32, 30, 9)                    # the action text stays visible up to its head in the next window
    assert all(v == "plain" for v in rt.ops.used.values())


def test_padding_does_not_change_the_result(ckpt):
    """The same texts padded to 32 or to 64 tokens give the same video (padding is masked; only rounding order moves)."""
    a, _ = session(runtime(ckpt, buckets=(32, 64)))
    b, _ = session(runtime(ckpt, buckets=(64, 128)))
    for x, y in zip(a, b):
        assert rel(x, y) < 1e-2


def test_lora_merge_matches_formula(ckpt):
    ms = W.MergedState(str(ckpt / "dit.safetensors"), str(ckpt / "adapter.safetensors"), str(ckpt / "adapter_config.json"))
    ad = M.st_read(str(ckpt / "adapter.safetensors"))
    base = M.st_read(str(ckpt / "dit.safetensors"))
    k = "double_blocks.0.img_attn_q.weight"
    want = base[k] + 2.0 * (ad["double_blocks.0.img_attn_q.lora_B.weight"] @ ad["double_blocks.0.img_attn_q.lora_A.weight"])
    assert np.array_equal(ms[k], want.astype(np.float32))
    assert np.array_equal(ms["final_layer.linear.weight"], base["final_layer.linear.weight"])


def test_text_rules():
    assert RT.text_info(10, None, 2)["head"] is None
    ti = RT.text_info(30, TEXTS["act"][1], 2)
    assert ti["intra"] and ti["frame_act"] == [0, 1] and ti["head"] == 9
    with pytest.raises(ValueError):
        RT.text_info(31, TEXTS["act"][1], 2)


# ------------------------------------------------------------------ kernel bundle: load and dispatch (CPU-exported stand-ins)
def fake_bundle(shapes):
    """A bundle in the release format whose programs are the plain ops exported for CPU."""
    from jax import export
    rows, files = [], {}
    for op, key, fn, avals in shapes:
        ex = export.export(jax.jit(fn), platforms=["cpu"])(*[jax.ShapeDtypeStruct(s, d) for s, d in avals])
        blob = ex.serialize()
        name = "%s__%s.jaxexport" % (op, key)
        files[name] = bytes(blob)
        rows.append(dict(op=op, shape_key=key, file=name, sha256=KB.sha256(bytes(blob)),
                         in_avals=[[list(s), str(np.dtype(d))] for s, d in avals],
                         out_avals=[[list(o.shape), str(o.dtype)] for o in ex.out_avals]))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(KB.PKG + "/kernels.json", json.dumps(dict(model="hy-nomas", jax=jax.__version__, files=rows)))
        for n, b in files.items():
            z.writestr(KB.PKG + "/" + n, b)
    data = buf.getvalue()
    return KB.Bundle(data, KB.sha256(data))


@pytest.mark.cpu_only   # stand-in bundles are exported for CPU; on the chip, tests/tpu/hy_nomas_kernel_check.py checks the real bundle
def test_bundle_dispatch_inside_shard_map(ckpt):
    H, Li, LQ = ARCH.heads, ARCH.tokens, 256
    bf, f32 = jnp.bfloat16, np.float32
    c_ = O.q_scale(128)
    shapes = [("dit_qk_norm_rope", "q_r%d" % LQ, lambda x, w, c, s: O.qk_norm_rope(x, w, c, s, H=H, scale=c_, out_rows=LQ),
               [((Li, H * 128), bf), ((128,), f32), ((Li, 128), f32), ((Li, 128), f32)])]
    for lk in (256, 384):
        shapes.append(("dit_qk_norm_rope", "k_r%d" % lk, lambda x, w, c, s, lk=lk: O.qk_norm_rope(x, w, c, s, H=H, out_rows=lk),
                       [((Li, H * 128), bf), ((128,), f32), ((Li, 128), f32), ((Li, 128), f32)]))
    b = fake_bundle(shapes)
    rt_k = runtime(ckpt, ops=O.Ops(b))
    rt_p = runtime(ckpt)
    a, _ = session(rt_k)
    p, _ = session(rt_p)
    used = rt_k.ops.summary()
    assert "dit_qk_norm_rope q_r256" in used["kernel"] and "dit_qk_norm_rope k_r256" in used["kernel"]
    assert any(n.startswith("dit_attention_masked ") for n in used["plain"])
    for x, y in zip(a, p):
        assert rel(x, y) < 1e-2
    assert b.get("dit_v_prep", "r1") is None


# ------------------------------------------------------------------ decoder
def test_decoder_shapes():
    A = DEC.Arch(hidden=256, intermediate=512, layers=1, heads=4, kv_heads=2, head_dim=128)
    r = np.random.RandomState(1)
    import ml_dtypes
    g = lambda *s: (r.randn(*s) * 0.02).astype(ml_dtypes.bfloat16)
    top = dict(embed={"weight": g(256, 3072), "bias": g(256)}, cond_proj={"weight": g(1024, 32), "bias": g(1024)},
               final_norm={"weight": g(256) + 1}, head={"norm.weight": g(256) + 1, "linear.weight": g(3072, 256), "linear.bias": g(3072)})
    st = {"input_norm.weight": g(1, 256) + 1, "post_attn_norm.weight": g(1, 256) + 1, "attn.q_proj.weight": g(1, 512, 256),
          "attn.k_proj.weight": g(1, 256, 256), "attn.v_proj.weight": g(1, 256, 256), "attn.o_proj.weight": g(1, 256, 512),
          "attn.q_norm_t.weight": g(1, 64) + 1, "attn.q_norm_hw.weight": g(1, 64) + 1, "attn.k_norm_t.weight": g(1, 64) + 1,
          "attn.k_norm_hw.weight": g(1, 64) + 1, "mlp.gate_proj.weight": g(1, 512, 256), "mlp.up_proj.weight": g(1, 512, 256),
          "mlp.down_proj.weight": g(1, 256, 512)}
    d = DEC.Decoder((top, st), jax.devices()[:1], O.Ops(None), A, log=lambda *a: None)
    fr = np.asarray(d.call(0, np.zeros((1, 32, 2, 16, 16), np.float32), 3))
    assert fr.shape == (8, 512, 512, 3) and fr.dtype == np.uint8
    fr2 = np.asarray(d.call(0, np.zeros((1, 32, 2, 16, 16), np.float32), 3))
    assert np.array_equal(fr, fr2)                    # noise is a function of the absolute latent index


# ------------------------------------------------------------------ cameras and prompts
def test_camera_turns_about_own_axes():
    c = C.Cameras(None, gain=1.0)
    c.next_window(["w"])
    yaw, pitch = C.block_angles(c.next_window(["w", "left"]))
    assert yaw < -20 and abs(pitch) < 1
    c = C.Cameras(None, gain=2.5, pitch_max=60)
    for _ in range(3):
        c.next_window(["up"])
    assert abs(c.pitch - 24.0) < 1e-6               # 60 deg seen / gain 2.5


def test_action_prompt():
    s1, s2 = PR.world_from_form(dict(name="red fox", scene="A snowy forest", action="trots forward"))
    line = PR.action_text(s1, "the fox leaps high", True)
    assert line == "leaps high"
    seg2, span = PR.action_seg2(s2, line, 1.0, 1.5)
    acts = json.loads(seg2)["segments"][0]["contained_actions"]
    assert span == (1.0, 2.5) and len(acts) == 3 and "leaps high" in acts[1]["action_description"]


# ------------------------------------------------------------------ command line and app server
def test_cli_routes_to_model(capsys):
    from mashbox import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "hy-nomas", "--help"])
    assert e.value.code == 0
    assert "--preset" in capsys.readouterr().out


def test_app_server_against_stub():
    from mashbox.models.hy_nomas import server as S
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        pytest.skip("imageio-ffmpeg not installed")
    eng = S.StubEngine(lambda *a: None, dit_s=0.2, dec_s=0.1, encode_s=0.0)
    live = S.Live(eng, lambda *a: None, hw="STUB", stub=True, buffer=0.2)
    srv = live.start(0)                 # port 0: the system's free port, bound by the server itself
    port = srv.server_address[1]
    try:
        assert srv.server_address[0] == "127.0.0.1"
        base = "http://127.0.0.1:%d" % port
        page = urllib.request.urlopen(base + "/", timeout=5).read()
        assert b"<title>" in page
        tok = re.search(rb'<meta name="mbx-token" content="([^"]+)">', page).group(1).decode()   # POST needs this run's token
        st = json.loads(urllib.request.urlopen(base + "/api/state", timeout=5).read())
        assert st["stub"] and len(st["worlds"]) >= 1
        req = lambda p, b: json.loads(urllib.request.urlopen(urllib.request.Request(base + p, json.dumps(b).encode(),
                                      {"Content-Type": "application/json", "X-Mashbox-Token": tok}), timeout=5).read())
        assert req("/api/world", dict(mode="preset", id=st["worlds"][0]["id"]))["ok"]
        assert req("/api/settings", dict(preset="live", gain=3.0))["settings"]["preset"] == "live"
        t0 = time.time()
        while time.time() - t0 < 30 and live.frames < 48:
            time.sleep(0.2)
        assert live.frames >= 48 and live.stream.sent > 0
        assert eng.preset == "live"
    finally:
        srv.shutdown()


def test_seed_is_a_whole_number_in_range_and_zero_counts():
    from mashbox.models.hy_nomas.server import seed_of
    assert seed_of(0, 123) == 0 and seed_of(None, 123) == 123 and seed_of("", 7) == 7 and seed_of("42", 7) == 42
    assert seed_of(2 ** 32 - 1, 0) == 2 ** 32 - 1 and seed_of(5.0, 0) == 5
    for bad in (-1, 2 ** 32, 1.5, 4294967295.9, True, False, "x", "1.5", [1], {}):
        with pytest.raises(ValueError):
            seed_of(bad, 0)


def test_app_server_same_machine_only_and_bad_seeds():
    """Only pages on this machine drive the server (Host and Origin loopback, JSON POSTs); a bad seed is a 400 and generation goes on."""
    import http.client
    import urllib.error
    from mashbox.models.hy_nomas import server as S
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        pytest.skip("imageio-ffmpeg not installed")
    eng = S.StubEngine(lambda *a: None, dit_s=0.05, dec_s=0.02, encode_s=0.0)
    live = S.Live(eng, lambda *a: None, hw="STUB", stub=True, buffer=0.2)
    srv = live.start(0)
    port = srv.server_address[1]
    base = "http://127.0.0.1:%d" % port
    tok = re.search(rb'<meta name="mbx-token" content="([^"]+)">', urllib.request.urlopen(base + "/", timeout=5).read()).group(1).decode()

    def code(p, body=None, **h):
        r = urllib.request.Request(base + p, None if body is None else json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(r, timeout=5) as f:
                return f.status
        except urllib.error.HTTPError as e:
            return e.code

    def ws(origin):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/ws?token=" + tok, headers={"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
                                         "Sec-WebSocket-Key": base64.b64encode(b"the sample nonce").decode(), "Origin": origin})
        st = c.getresponse().status
        c.close()
        return st
    J = {"Content-Type": "application/json", "X-Mashbox-Token": tok}      # POST needs this run's token
    try:
        assert code("/") == 200 and code("/", Host="localhost:%d" % (port + 1)) == 200   # a tunnel on another local port
        assert code("/", Host="evil.example:%d" % port) == 403 and code("/api/state", Host="evil.example") == 403
        wid = json.loads(urllib.request.urlopen(base + "/api/state", timeout=5).read())["worlds"][0]["id"]
        assert code("/api/world", dict(mode="preset", id=wid), **J, Origin="https://evil.example") == 403
        assert code("/api/world", dict(mode="preset", id=wid), **{"Content-Type": "text/plain"}) == 415
        assert code("/api/world", dict(mode="preset", id=wid), **J, Origin=base) == 200
        assert ws("https://evil.example") == 403 and ws("null") == 403 and ws(base) == 101
        for bad in (-1, 2 ** 32, "x"):
            assert code("/api/restart", dict(seed=bad), **J) == 400
            assert code("/api/world", dict(mode="preset", id=wid, seed=bad), **J) == 400
        assert code("/api/restart", dict(seed=2 ** 32 - 1), **J) == 200
        f0, t0 = live.frames, time.time()
        while time.time() - t0 < 30 and live.frames < f0 + 48:
            time.sleep(0.2)
        assert live.frames >= f0 + 48
    finally:
        srv.shutdown()
