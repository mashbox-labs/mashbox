"""Track2Rig (models/track2rig): guards the port against wrong rigs, wrong exports and unsafe inputs. It fails when the
model's output drifts from the recorded rows, the weights file is not the pinned one, a GLB or rig.json is malformed, the
GLB is not in glTF's Y-up frame, a bad tracks file is accepted (wrong shape, NaN, camera cut, oversized archive), values
at hidden samples change the rig, an fps or output name the CLI should refuse gets through, or the video package's input
checks disagree with ours.

Weights and example clips: MASHBOX_TRACK2RIG_WEIGHTS / MASHBOX_TRACK2RIG_CLIPS (local copies), else the pinned Hugging
Face revision in models.json; tests that need them skip, with the reason, when neither is available."""
import json
import struct
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mashbox import hub  # noqa: E402
from mashbox.cli import main as cli_main  # noqa: E402
from mashbox.models import track2rig  # noqa: E402
from mashbox.models.track2rig import eval as ev  # noqa: E402
from mashbox.models.track2rig.app import TOLERANCE  # noqa: E402
from mashbox.models.track2rig.bench import evaluate, gt_rig, load_example  # noqa: E402
from mashbox.models.track2rig.model import CONFIG, predict  # noqa: E402
from mashbox.models.track2rig.rig import Rig  # noqa: E402
from mashbox.models.track2rig.ssdr import ssdr  # noqa: E402
from mashbox.models.track2rig.tracks import check_tracks, load_tracks, save_tracks  # noqa: E402
from mashbox.models.track2rig.weights import WEIGHTS_SHA256, read_safetensors, sha256_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track2rig_tracks import hinge  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRACKS = str(ROOT / "examples" / "track2rig" / "tracks")
REFERENCE = str(ROOT / "examples" / "track2rig" / "clips" / "reference.json")
ENTRY = hub.model_entry("track2rig")


# Track frame (Z up, facing -Y) -> glTF 2.0 frame (Y up, facing +Z): (x, y, z) -> (x, z, -y).
C_GLTF = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])


def _rot(axis, ang):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def standing_rig(T=24, seed=7):
    """A four-legged creature built by hand in the track frame (Z up, feet at z = 0, head toward -Y), no model weights:
    trunk (root), head, tail, and per leg upper -> lower -> paw. Each bone turns about its joint (all three axes over
    the body) on top of its parent's motion; frame 0 is the bind pose. Points near a joint blend the two bones 50/50."""
    rng = np.random.default_rng(seed)
    bones = [("trunk", -1, (0, 0, 0.55), ((-0.1, 0.1), (-0.4, 0.4), (0.45, 0.65))),
             ("head", 0, (0, -0.4, 0.62), ((-0.06, 0.06), (-0.6, -0.4), (0.55, 0.8))),
             ("tail", 0, (0, 0.4, 0.55), ((-0.03, 0.03), (0.4, 0.7), (0.52, 0.58)))]
    for sx in (-0.08, 0.08):
        for sy in (-0.3, 0.3):
            up = len(bones)
            bones += [("upper", 0, (sx, sy, 0.45), ((sx - 0.03, sx + 0.03), (sy - 0.03, sy + 0.03), (0.22, 0.45))),
                      ("lower", up, (sx, sy, 0.22), ((sx - 0.025, sx + 0.025), (sy - 0.025, sy + 0.025), (0.04, 0.22))),
                      ("paw", up + 1, (sx, sy, 0.04), ((sx - 0.03, sx + 0.03), (sy - 0.05, sy + 0.01), (0.0, 0.04)))]
    B = len(bones)
    parent = np.array([b[1] for b in bones])
    joints = np.array([b[2] for b in bones], float)
    pts, own = [], []
    for i, (_, _, _, box) in enumerate(bones):
        lo, hi = np.array(box).T
        pts.append(rng.uniform(lo, hi, (40, 3)))
        own += [i] * 40
    P = np.concatenate(pts)
    own = np.array(own)
    W = np.zeros((len(P), B))
    W[np.arange(len(P)), own] = 1.0
    for i in range(B):   # blend points of a child bone that sit close to its joint with the parent
        near = (own == i) & (np.linalg.norm(P - joints[i], axis=1) < 0.05) & (parent[i] >= 0)
        W[near, i] = 0.5
        W[near, parent[i]] = 0.5
    axes = {"trunk": (0, 0, 1), "head": (0.3, 1, 0.2), "tail": (1, 0, 1), "upper": (1, 0, 0), "lower": (1, 0.2, 0),
            "paw": (1, 0, 0.3)}
    Rt = np.zeros((T, B, 3, 4))
    for t in range(T):
        s = np.sin(2 * np.pi * t / T)
        for i, (name, pa, j, _) in enumerate(bones):
            ang = {"trunk": 0.15, "head": 0.4, "tail": 0.6, "upper": 0.45, "lower": -0.5, "paw": 0.35}[name] * s
            Rl = _rot(axes[name], ang * (1 if i % 2 else -1))
            tl = np.asarray(j) - Rl @ j        # turn about the joint (bind frame)
            if name == "trunk":
                tl = tl + np.array([0.0, 0.0, 0.02 * s])
            Rp, tp = (np.eye(3), np.zeros(3)) if pa < 0 else (Rt[t, pa, :, :3], Rt[t, pa, :, 3])
            Rt[t, i, :, :3] = Rp @ Rl
            Rt[t, i, :, 3] = Rp @ tl + tp
    cen = (W.T @ P) / W.sum(0)[:, None]
    diag = float(np.linalg.norm(np.ptp(P, 0)))
    return Rig(weights=W, transforms=Rt, bind_points=P, parent=parent, joints=joints, centroids=cen, root=0,
               model_slot=np.arange(B), quality=dict(bbox_diag=diag), flags=[])


@pytest.fixture(scope="module")
def weights_file():
    p = os.environ.get("MASHBOX_TRACK2RIG_WEIGHTS")
    if p:
        return p
    if ENTRY["hf_revision"] is None:
        pytest.skip("Track2Rig weights: set MASHBOX_TRACK2RIG_WEIGHTS (not on Hugging Face at a pinned revision yet)")
    from mashbox.models.track2rig.weights import weights_path
    return weights_path()


@pytest.fixture(scope="module")
def clips():
    d = os.environ.get("MASHBOX_TRACK2RIG_CLIPS")
    if d:
        return d
    if ENTRY["hf_revision"] is None:
        pytest.skip("Track2Rig example clips: set MASHBOX_TRACK2RIG_CLIPS (not on Hugging Face at a pinned revision yet)")
    from mashbox.models.track2rig.app import _example_clips
    return os.path.dirname(_example_clips()[0][0])


@pytest.fixture(scope="module")
def params(weights_file):
    from mashbox.models.track2rig.weights import load_params
    return load_params(weights_file)


@pytest.fixture(scope="module")
def model(weights_file):
    return track2rig.load(weights_file)


@pytest.fixture(scope="module")
def hinge_rig(model):
    X, mask, _ = hinge()
    return model.rig(X, mask), X, mask


def _glb(b):
    magic, ver, total = struct.unpack("<4sII", b[:12])
    assert magic == b"glTF" and ver == 2 and total == len(b)
    jl, jt = struct.unpack("<I4s", b[12:20])
    assert jt == b"JSON"
    g = json.loads(b[20:20 + jl])
    bl, bt = struct.unpack("<I4s", b[20 + jl:28 + jl])
    assert bt == b"BIN\x00" and bl == g["buffers"][0]["byteLength"]
    return g


def _glb_reader(b):
    """(glTF JSON, accessor -> array) for a GLB, read from the file bytes only."""
    g = _glb(b)
    jl = struct.unpack("<I", b[12:16])[0]
    bin_ = b[28 + jl:]
    dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}
    nc = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def acc(i):
        a = g["accessors"][i]
        v = g["bufferViews"][a["bufferView"]]
        n = nc[a["type"]]
        x = np.frombuffer(bin_, dt[a["componentType"]], a["count"] * n, v["byteOffset"] + a.get("byteOffset", 0))
        return x.reshape(a["count"], n).astype(np.float64)

    return g, acc


def _quat_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _glb_node_worlds(g, acc, frame=None):
    """World 4x4 matrix of every node: the rest pose from the node TRS (frame None) or the animation keys at `frame`."""
    local = []
    for nd in g["nodes"]:
        M = np.eye(4)
        M[:3, :3] = _quat_mat(np.array(nd.get("rotation", [0, 0, 0, 1.0])))
        M[:3, 3] = nd.get("translation", [0, 0, 0])
        local.append(M)
    if frame is not None:
        an = g["animations"][0]
        for ch in an["channels"]:
            sm = an["samplers"][ch["sampler"]]
            v = acc(sm["output"])[frame]
            M = local[ch["target"]["node"]]
            if ch["target"]["path"] == "translation":
                M[:3, 3] = v
            else:
                M[:3, :3] = _quat_mat(v)
    par = {c: i for i, nd in enumerate(g["nodes"]) for c in nd.get("children", [])}
    world = {}

    def get(i):
        if i not in world:
            world[i] = (get(par[i]) if i in par else np.eye(4)) @ local[i]
        return world[i]

    return [get(i) for i in range(len(g["nodes"]))]


def _glb_skinned_points(g, acc, frame):
    """glTF linear blend skinning from the file: sum_k w_k (joint world x inverse bind) v, then each point's tetrahedron
    centroid -> [N,3] in the GLB frame."""
    prim = g["meshes"][0]["primitives"][0]["attributes"]
    V, J, Wt = acc(prim["POSITION"]), acc(prim["JOINTS_0"]).astype(int), acc(prim["WEIGHTS_0"])
    sk = g["skins"][0]
    ibm = acc(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)   # column-major in the file
    worlds = _glb_node_worlds(g, acc, frame)
    S = np.stack([worlds[n] @ ibm[j] for j, n in enumerate(sk["joints"])])
    Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
    out = np.einsum("vk,vkij,vj->vi", Wt, S[J], Vh)[:, :3]
    return out.reshape(-1, 4, 3).mean(1)


def _check_glb_rest_pose(r, fps):
    """(a) The GLB's rest pose is in glTF's frame (Y up, facing +Z): the track frame turned by C_GLTF, so +Y is up."""
    g, acc = _glb_reader(r.to_glb(fps=fps))
    tol = 1e-6 * max(1.0, float(np.abs(r.bind_points).max()))
    rest = _glb_node_worlds(g, acc)
    jw = np.array([rest[b][:3, 3] for b in range(r.num_bones)])   # joint world positions from the node hierarchy
    assert np.abs(jw - r.joints @ C_GLTF.T).max() < tol
    pos = acc(g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).reshape(-1, 4, 3).mean(1)   # mesh at bind
    assert np.abs(pos - r.bind_points @ C_GLTF.T).max() < tol
    # the GLB's vertical (Y) extent is the track's up (Z) extent; its 4 lowest joints are the track's 4 lowest (the feet
    # on a walking animal), below the body centroid
    assert abs(np.ptp(pos[:, 1]) - np.ptp(r.bind_points[:, 2])) < tol
    assert abs(np.ptp(pos[:, 2]) - np.ptp(r.bind_points[:, 1])) < tol
    lowest = np.argsort(jw[:, 1])[:4]
    assert (jw[lowest, 1] < pos[:, 1].mean()).all()
    assert np.abs(np.sort(r.joints[lowest, 2]) - np.sort(r.joints[:, 2])[:4]).max() < tol


def _check_glb_skinning(r, fps):
    """(b) Skinning the GLB in glTF space (inverse bind x joint world, keys at whole frames, all read from the file)
    reproduces the rig turned by C_GLTF, at float32 precision."""
    g, acc = _glb_reader(r.to_glb(fps=fps))
    times = acc(g["animations"][0]["samplers"][0]["input"])[:, 0]
    T = r.transforms.shape[0]
    rec = r.reconstruct()
    for t in sorted({0, 1, T // 3, T // 2, T - 1}):
        assert abs(times[t] - t / fps) < 1e-6
        err = np.abs(_glb_skinned_points(g, acc, t) - rec[t] @ C_GLTF.T).max() / r.quality["bbox_diag"]
        assert err < 1e-5, (t, err)


_GLB_CHECKS = {"rest_pose": _check_glb_rest_pose, "skinning": _check_glb_skinning}


@pytest.mark.parametrize("check", sorted(_GLB_CHECKS))
def test_glb_is_y_up_synthetic(check):
    """glTF 2.0 is Y up. A hand-built Z-up creature (no model weights needed) stands on +Y in its GLB."""
    r = standing_rig()
    assert np.ptp(r.bind_points[:, 2]) > 0.7 and r.bind_points[:, 2].min() >= 0.0   # stands on z = 0, 0.8 tall
    _GLB_CHECKS[check](r, fps=24.0)


@pytest.fixture(scope="module")
def quickstart_rig(model):
    return model.rig_file(os.path.join(TRACKS, "grison_walk.npz"))


@pytest.mark.parametrize("check", sorted(_GLB_CHECKS))
def test_glb_is_y_up_quickstart(quickstart_rig, check):
    """The same checks on the real rig of the quickstart clip (needs the weights)."""
    r = quickstart_rig
    P = r.bind_points
    assert P[:, 2].min() > -0.01 and np.ptp(P[:, 2]) < np.ptp(P[:, 1])   # the clip is Z up, long on Y
    _GLB_CHECKS[check](r, fps=30.0)


def test_rig_json_stays_in_the_track_frame():
    r = standing_rig()
    d = json.loads(r.to_json())
    assert np.abs(np.array(d["joints"]) - r.joints).max() < 1e-6
    assert np.abs(np.array(d["transforms"]["data"]) - r.transforms).max() < 1e-6


def test_predict_shapes_and_skin(params):
    X, mask, _ = hinge(N=256, T=40)
    W, Rt, P = predict(params, X, mask)
    assert W.shape == (256, 32) and Rt.shape == (40, 32, 3, 4) and P.shape == (256, 3)
    assert np.allclose(W.sum(1), 1, atol=1e-5)
    assert ((W > 0).sum(1) <= 4).all()
    R = Rt[..., :3]
    assert np.allclose(np.einsum("tbij,tbkj->tbik", R, R), np.eye(3), atol=1e-5)
    assert np.allclose(np.linalg.det(R), 1, atol=1e-5)


def test_python_api_runs_the_model_on_the_cpu(model, monkeypatch):
    # The documented Python API (load + rig) must not use the TPU or GPU when a TPU/GPU JAX is installed: the CLI's
    # JAX_PLATFORMS=cpu does not apply to it, so the model call itself pins the host CPU (on a TPU VM, without the pin it
    # ran on the TPU and the rig differed). Here: record the default device the model call sees.
    import jax
    from mashbox.models.track2rig import model as net
    seen, real = [], net._forward_jit

    def spy(*a):
        seen.append(jax.config.jax_default_device)
        return real(*a)

    monkeypatch.setattr(net, "_forward_jit", spy)
    X, mask, _ = hinge(N=128, T=24)
    model.rig(X, mask)
    assert seen and all(d is not None and d.platform == "cpu" for d in seen), seen


def test_eval_jobs_fails_fast_when_a_worker_dies(tmp_path):
    # A worker that dies (a crash in a native library, a kill, out of memory) must stop `eval --jobs N` at once with an
    # error. With multiprocessing.Pool the lost task left the run waiting forever. Run in a fresh process with a time
    # limit, so a hang fails this test instead of hanging the suite.
    import subprocess
    (tmp_path / "t2r_crashjob.py").write_text("import os\n\ndef job(args):\n    os._exit(139)\n")
    code = ("import sys, time\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import t2r_crashjob\n"
            "from mashbox.models.track2rig import eval as ev\n"
            "t0 = time.time()\n"
            "try:\n"
            "    ev.run(['a.npz', 'b.npz', 'c.npz'], with_ssdr=False, jobs=2, job=t2r_crashjob.job, log=lambda s: None)\n"
            "except RuntimeError as e:\n"
            "    print('FAILED_FAST', round(time.time() - t0, 1), 'a.npz' in str(e))\n")
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[:1] == ["FAILED_FAST"] and r.stdout.split()[-1] == "True", r.stdout


def test_deterministic(params):
    X, mask, _ = hinge(N=256, T=40)
    a = predict(params, X, mask)
    b = predict(params, X, mask)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_hinge_rig(hinge_rig):
    r, X, mask = hinge_rig
    # the rig re-poses the input points closely (clean synthetic input)
    assert r.quality["recon_err_pct_diag"] < 1.0
    # points on the two segments mostly go to different dominant bones
    _, _, moving = hinge()
    dom = r.weights.argmax(1)
    assert len(set(dom[moving]) & set(dom[~moving])) < len(set(dom))
    assert (r.parent == -1).sum() == 1 and r.parent[r.root] == -1
    assert r.joints.shape == (r.num_bones, 3)


def test_max_bones_cap(model):
    X, mask, _ = hinge(N=256, T=40)
    r = model.rig(X, mask, max_bones=4)
    assert r.num_bones == 4
    assert "max_bones_capped" in r.flags
    assert np.allclose(r.weights.sum(1), 1)


def test_glb_structure(hinge_rig):
    r, X, mask = hinge_rig
    g = _glb(r.to_glb(fps=24))
    B = r.num_bones
    assert len(g["skins"][0]["joints"]) == B
    assert len(g["nodes"]) == B + 1
    assert len(g["animations"][0]["channels"]) == 2 * B
    pos = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert pos["count"] == 4 * X.shape[1]
    times = g["accessors"][g["animations"][0]["samplers"][0]["input"]]
    assert abs(times["max"][0] - (X.shape[0] - 1) / 24) < 1e-5


def test_rig_json(hinge_rig):
    r, X, mask = hinge_rig
    d = json.loads(r.to_json())
    assert d["format"] == "track2rig.rig/1"
    assert d["bones"] == r.num_bones and len(d["parent"]) == r.num_bones
    assert np.array(d["skin"]["index"]).shape == (X.shape[1], 4)
    assert np.array(d["transforms"]["data"]).shape == (X.shape[0], r.num_bones, 3, 4)
    w = np.array(d["skin"]["weight"])
    assert np.allclose(w.sum(1), 1, atol=1e-4)


def test_json_skin_reproduces_rig(hinge_rig):
    r, X, mask = hinge_rig
    d = json.loads(r.to_json())
    idx, w = np.array(d["skin"]["index"]), np.array(d["skin"]["weight"])
    W = np.zeros((X.shape[1], d["bones"]))
    np.add.at(W, (np.arange(X.shape[1])[:, None], idx), w)
    assert np.abs(W - r.weights).max() < 1e-4


def test_roundtrip(tmp_path):
    X, mask, _ = hinge(N=64, T=20)
    p = str(tmp_path / "t.npz")
    save_tracks(p, X, mask)
    X2, m2 = load_tracks(p)
    assert np.array_equal(X, X2) and np.array_equal(mask, m2)


@pytest.mark.parametrize("bad,msg", [
    (lambda X, m: (X[:, :, :2], m), "need \\[T, N, 3\\]"),
    (lambda X, m: (np.where(np.isnan(X), X, np.nan), m), "NaN"),
    (lambda X, m: (X, np.zeros_like(m)), "frame 0"),
    (lambda X, m: (np.repeat(X, 10, 0)[:200], None), "max 176"),
    (lambda X, m: (np.concatenate([X[:5], X[5:] + 100.0]), None), "camera cut"),
])
def test_bad_inputs(bad, msg):
    X, mask, _ = hinge(N=64, T=20)
    with pytest.raises(ValueError, match=msg):
        check_tracks(*bad(X, mask))


def test_hidden_samples_may_hold_any_finite_value():
    """docs/models/track2rig.md: the value at a hidden sample is ignored (any finite number). The 1e6 limit is for
    visible samples; NaN and Inf are refused everywhere."""
    X, mask, _ = hinge(N=64, T=40)
    assert (~mask).any() and mask[5, 3]
    zero, big = X.copy(), X.copy()
    zero[~mask] = 0.0
    big[~mask] = 1e7
    a, b = check_tracks(zero, mask), check_tracks(big, mask)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    vis = X.copy()
    vis[5, 3] = 1e7
    with pytest.raises(ValueError, match="exceed"):
        check_tracks(vis, mask)
    for v in (np.nan, np.inf, -np.inf):
        bad = X.copy()
        bad[~mask] = v
        with pytest.raises(ValueError, match="NaN or Inf"):
            check_tracks(bad, mask)


def test_cli_rig(tmp_path, capsys, weights_file):
    src = os.path.join(TRACKS, "grison_walk.npz")
    assert cli_main(["run", "track2rig", src, "-o", str(tmp_path / "out"), "--weights", weights_file]) == 0
    assert (tmp_path / "out" / "rig.json").stat().st_size > 1000
    assert (tmp_path / "out" / "rig.glb").stat().st_size > 1000
    assert "bones" in capsys.readouterr().out


def test_cli_info(capsys, weights_file):
    cli_main(["run", "track2rig", "info", "--weights", weights_file])
    assert '"weights_sha256_ok": true' in capsys.readouterr().out


def test_sha256_pinned(weights_file):
    assert sha256_file(weights_file) == WEIGHTS_SHA256


def test_tensors_and_config(params, weights_file):
    flat, meta = read_safetensors(weights_file)
    assert len(flat) == 55
    assert all(v.dtype == np.float32 for v in flat.values())
    assert meta["license"] == "Apache-2.0"
    assert len(params["layers"]) == CONFIG["L"]
    assert params["slots"].shape == (CONFIG["K"], CONFIG["D"])
    assert params["inp"]["w"].shape == (3 + CONFIG["NB"] * 4, CONFIG["D"])
    assert all(np.isfinite(v).all() for v in flat.values())


def test_tampered_copy_is_not_the_pinned_file(tmp_path, weights_file):
    raw = bytearray(open(weights_file, "rb").read())
    raw[-1] ^= 1
    p = tmp_path / "w.safetensors"
    p.write_bytes(bytes(raw))
    assert sha256_file(str(p)) != WEIGHTS_SHA256


def test_example_replay_is_deterministic(clips):
    p = os.path.join(clips, "grison_walk.npz")
    a, b = load_example(p), load_example(p)
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["mask"], b["mask"])
    assert a["mask"][0].all()
    hidden = 1 - a["mask"][1:].mean()
    assert 0.19 < hidden < 0.30


def test_true_rig_is_the_floor(clips):
    c = load_example(os.path.join(clips, "grison_walk.npz"))
    m = evaluate(c, *gt_rig(c))
    assert m["recon_tracked"] < 0.01


def test_matches_recorded_rows(clips, params):
    """Track2Rig on one shipped clip reproduces the row recorded for it (different CPU and build, so a small tolerance)."""
    ev._PARAMS = params
    rows = ev.run([os.path.join(clips, "grison_walk.npz")], with_ssdr=False, log=lambda *a: None)
    worst, _ = ev.compare(rows, REFERENCE)
    assert worst["recon_heldout"] <= TOLERANCE["recon_heldout"]
    assert worst["recon_tracked"] <= TOLERANCE["recon_tracked"]
    assert worst["j2j"] <= TOLERANCE["j2j"]
    assert worst["bones"] == 0


def _ref_row(clip="c", method="track2rig", **kw):
    return dict(dict(clip=clip, method=method, recon_heldout=0.5, recon_tracked=0.1, j2j=3.0, bones=32), **kw)


def _write_ref(tmp_path, rows):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(dict(source="test rows", rows=rows)))
    return str(p)


@pytest.mark.parametrize("ref_rows,ours,msg", [
    (None, [_ref_row()], "not found"),
    ([], [_ref_row()], "no recorded rows"),
    ([_ref_row(), _ref_row(j2j=3.1)], [_ref_row()], "two rows"),
    ([_ref_row(method="gt_rig")], [_ref_row()], "no recorded row"),
    ([_ref_row(clip="other")], [_ref_row()], "no recorded row"),
    ([{k: v for k, v in _ref_row().items() if k != "j2j"}], [_ref_row()], "j2j"),
    ([_ref_row(bones=None)], [_ref_row()], "bones"),
])
def test_reference_problems_are_errors(tmp_path, ref_rows, ours, msg):
    """A missing or broken reference, or a scored (clip, method) without a recorded row, is an error, never a pass."""
    path = str(tmp_path / "missing.json") if ref_rows is None else _write_ref(tmp_path, ref_rows)
    with pytest.raises(ValueError, match=msg):
        ev.compare(ours, path)


def test_reference_compare_reports_the_largest_difference(tmp_path):
    path = _write_ref(tmp_path, [_ref_row(), _ref_row(method="gt_rig")])
    worst, lines = ev.compare([_ref_row(j2j=3.25), _ref_row(method="gt_rig")], path)
    assert worst == dict(recon_heldout=0.0, recon_tracked=0.0, j2j=0.25, bones=0.0) and "test rows" in lines[0]
    worst, _ = ev.compare([_ref_row(recon_heldout=float("nan"))], path)
    assert worst["recon_heldout"] == float("inf")   # a NaN result is over any tolerance, not skipped


@pytest.mark.parametrize("ref_rows", [None, [], "other clip"])
def test_cli_eval_reference_problems_exit_1(tmp_path, capsys, clips, weights_file, ref_rows):
    clip = os.path.join(clips, "grison_walk.npz")
    if ref_rows is None:
        ref = str(tmp_path / "missing.json")
    elif ref_rows == "other clip":
        ref = _write_ref(tmp_path, [_ref_row(clip="grison_idle", method=m) for m in ("gt_rig", "track2rig")])
    else:
        ref = _write_ref(tmp_path, ref_rows)
    argv = ["run", "track2rig", "eval", clip, "--no-ssdr", "--weights", weights_file]
    assert cli_main(argv + ["--reference", ref]) == 1
    assert "REFERENCE ERROR" in capsys.readouterr().out
    assert cli_main(argv + ["--reference", REFERENCE]) == 0
    assert "all within tolerance" in capsys.readouterr().out


def test_reference_file():
    """The recorded rows name exactly the example clips listed in models.json."""
    ref = json.load(open(REFERENCE))
    names = {f["name"].rsplit("/", 1)[-1][:-4] for f in ENTRY["files"] if f.get("kind") == "example" and f["name"].endswith(".npz")}
    assert {r["clip"] for r in ref["rows"]} == names


def test_ssdr_fits_a_hinge():
    X, mask, _ = hinge(N=64, T=12)
    X = X.astype(np.float64)
    X[~mask] = 0.0
    W, Rt, P, hist = ssdr(X, mask, 2, iters=5)
    assert W.shape == (64, 2) and Rt.shape == (12, 2, 3, 4)
    assert np.allclose(W.sum(1), 1)
    assert hist[-1] < 0.02   # mean error in input units; the limb is 2 units long


def test_weights_pin_matches_models_json():
    row = next(f for f in ENTRY["files"] if f.get("kind") == "weights" and f.get("default"))
    assert row["sha256"] == WEIGHTS_SHA256 and row["name"] == "track2rig-v2.safetensors"


@pytest.mark.parametrize("far_value", [1e6, 1e7, -3e9])
def test_hidden_samples_cannot_change_the_rig(model, far_value):
    """Any finite value at a hidden sample gives the same rig (they were weighted EPS in the rigid fits), also one past
    the 1e6 limit that visible samples have."""
    X, mask, _ = hinge(N=256, T=40)
    far = X.copy()
    far[~mask] = far_value
    a, b = model.rig(X, mask), model.rig(far, mask)
    assert np.array_equal(a.weights, b.weights) and np.array_equal(a.transforms, b.transforms)
    Wa, Ra, _ = predict(model.params, X, mask)
    Wb, Rb, _ = predict(model.params, far, mask)
    assert np.array_equal(Wa, Wb) and np.array_equal(Ra, Rb)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_bad_fps_is_refused(hinge_rig, tmp_path, fps):
    r = hinge_rig[0]
    for f in (lambda: r.to_glb(fps), lambda: r.to_json(fps), lambda: r.save(str(tmp_path), fps=fps)):
        with pytest.raises(ValueError, match="fps"):
            f()


@pytest.mark.parametrize("flag", [["--fps", "0"], ["--fps", "nan"], ["--name", "../x"], ["--name", "a/b"], ["--max-bones", "0"]])
def test_cli_refuses_bad_options(flag, capsys):
    with pytest.raises(SystemExit) as e:
        cli_main(["run", "track2rig", os.path.join(TRACKS, "grison_walk.npz")] + flag)
    assert e.value.code == 2


def test_save_stays_in_its_folder(hinge_rig, tmp_path):
    with pytest.raises(ValueError, match="file name"):
        hinge_rig[0].save(str(tmp_path / "out"), name="../escaped")
    assert not (tmp_path / "escaped.json").exists()


def test_oversized_archive_is_refused_before_reading(tmp_path):
    p = tmp_path / "big.npz"
    np.savez_compressed(p, tracks=np.zeros((200, 4096, 3)))   # 19.7 MB once decompressed, a few kB on disk
    with pytest.raises(ValueError, match="uncompressed"):
        load_tracks(str(p))


def test_video_package_checks_agree():
    """packages/track2rig-video writes tracks.npz with its own copy of our checks: same verdicts, same arrays."""
    sys.path.insert(0, str(ROOT / "packages" / "track2rig-video" / "src"))
    try:
        from track2rig_video import tracksfile
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 30, 3)).astype(np.float32)
    m = np.ones((12, 30), bool)
    m[3:6, :4] = False
    cut = base.copy()
    cut[6:] += 20.0
    hole = m.copy()
    hole[0, 1] = False
    far_hidden = base.copy()
    far_hidden[~m] = 1e7
    far_visible = base.copy()
    far_visible[0, 0] = 1e7
    for X, M in [(base, m), (cut, np.ones_like(m)), (base, hole), (np.zeros_like(base), m), (base[:, :4], m[:, :4]),
                 (far_hidden, m), (far_visible, m)]:
        got = []
        for f in (check_tracks, tracksfile.check_tracks):
            try:
                got.append(f(X, M))
            except ValueError as e:
                got.append(str(e).split(" (")[0].split(";")[0])
        if isinstance(got[0], str) or isinstance(got[1], str):
            assert got[0] == got[1]
        else:
            assert all(np.array_equal(a, b) for a, b in zip(got[0], got[1]))
