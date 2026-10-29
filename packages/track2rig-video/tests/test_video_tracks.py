# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for track2rig_video: no GPU, no tracker, no network.

A synthetic four-legged 'animal' walks across a floor, seen by a tilted camera. We render
its depth, give `align` the same arrays a tracker would, and check that the result is Z up,
faces -Y, stays in place and passes the rig step's input checks."""
import json

import numpy as np
import pytest

from track2rig_video import check_tracks, load_tracks, save_tracks
from track2rig_video import align
from track2rig_video.cli import main


def _scene(T=60, N=400, seed=1, walk=(0.6, 0.0)):
    """World: Z up, floor z=0. Body points on a box (long axis along walk dir) with swinging legs."""
    rng = np.random.default_rng(seed)
    body = rng.uniform([-0.5, -0.15, 0.35], [0.5, 0.15, 0.6], (N, 3))
    leg = rng.integers(0, 4, N)
    on_leg = rng.random(N) < 0.4
    hip = np.array([[-0.4, -0.1], [-0.4, 0.1], [0.4, -0.1], [0.4, 0.1]])
    body[on_leg, :2] = hip[leg[on_leg]] + rng.normal(0, 0.02, (on_leg.sum(), 2))
    body[on_leg, 2] = rng.uniform(0.02, 0.35, on_leg.sum())
    t = np.arange(T)[:, None]
    X = np.repeat(body[None], T, 0)
    swing = 0.12 * np.sin(2 * np.pi * t / 20 + np.pi * (leg % 2))  # [T, N]
    X[:, on_leg, 0] += (swing[:, on_leg] * (0.35 - body[on_leg, 2]) / 0.35)
    ang = np.arctan2(walk[1], walk[0])  # body long axis = walk direction
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    X = X @ Rz.T
    X[..., 0] += np.linspace(0, walk[0], T)[:, None]
    X[..., 1] += np.linspace(0, walk[1], T)[:, None]
    X[..., 2] += 0.0
    return X


def _camera(eye=(0.3, -3.0, 1.2), target=(0.3, 0.0, 0.3)):
    """OpenCV world-to-camera (x right, y down, z forward) looking from eye at target, world Z up."""
    eye, target = np.array(eye, float), np.array(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: camera axes in world
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def _raw(T=60, N=400, walk=(0.6, 0.0), world_tilt=True):
    X = _scene(T, N, walk=walk)
    w2c = _camera()
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    H, W = 240, 320
    # depth of the floor plane z=0 per pixel, plus the subject mask from projected frame-0 points
    c2w = np.linalg.inv(w2c)
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    dw = d @ c2w[:3, :3].T
    o = c2w[:3, 3]
    s = -o[2] / dw[..., 2]
    depth = np.where(s > 0, s, 0.0)
    pc = X[0] @ w2c[:3, :3].T + w2c[:3, 3]
    u = (pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = (pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    mask0 = np.zeros((H, W), bool)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    for du in range(-3, 4):
        for dv in range(-3, 4):
            mask0[np.clip(v[ok] + dv, 0, H - 1), np.clip(u[ok] + du, 0, W - 1)] = True
    raw = dict(coords=X.copy(), visib=np.full(X.shape[:2], 3.0), K=np.repeat(K[None], T, 0),
               w2c=np.repeat(w2c[None], T, 0), depth0=depth, mask0=mask0)
    if world_tilt:  # trackers return an arbitrary world frame: rotate everything by a random rigid transform
        rng = np.random.default_rng(7)
        A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(A) < 0:
            A[:, 0] *= -1
        tvec = rng.normal(size=3)
        raw["coords"] = X @ A.T + tvec
        G = np.eye(4)
        G[:3, :3] = A
        G[:3, 3] = tvec
        raw["w2c"] = np.repeat((w2c @ np.linalg.inv(G))[None], T, 0)
    raw["visib"][5:15, :50] = -3.0  # some points hidden for a while
    return raw, X


@pytest.mark.parametrize("walk", [(0.6, 0.0), (0.0, 0.6), (-0.4, 0.4)])
def test_align_recovers_up_heading_in_place(walk):
    raw, X_true = _raw(walk=walk)
    X, M, info = align(raw)
    assert info["align"]["up"]["src"] == "ground RANSAC"
    assert info["align"]["heading"] == "travel direction"
    check_tracks(X, M)
    # Z up with the floor at 0: frame-0 heights match the true heights above the floor
    z, zt = X[0][:, 2], X_true[0][:, 2]
    assert abs(np.percentile(z, 1) - np.percentile(zt, 1)) < 0.01
    assert abs(z.max() - zt.max()) < 0.01
    # long axis along Y, facing -Y: the body moved toward -Y before travel removal
    ext = np.ptp(X[0], 0)
    assert ext[1] > ext[0]
    tr = np.array(info["travel_xy"])
    assert tr[-1][1] < -0.3 and abs(tr[-1][0]) < 0.05
    # in place: visible-point centroid barely moves horizontally
    c = np.array([X[t][M[t]].mean(0) for t in range(len(X))])
    assert np.abs(c[:, :2] - c[0, :2]).max() < 0.06
    # hidden samples are zero and flagged
    assert not M[5:15, :50].any() and np.all(X[5:15, :50] == 0)


def test_keep_travel_and_flip():
    raw, _ = _raw()
    X, M, info = align(raw, keep_travel=True)
    c = np.array([X[t][M[t]].mean(0) for t in range(len(X))])
    assert c[-1, 1] - c[0, 1] < -0.4  # walks toward -Y
    Xf, Mf, infof = align(raw, keep_travel=True, flip_heading=True)
    cf = np.array([Xf[t][Mf[t]].mean(0) for t in range(len(Xf))])
    assert cf[-1, 1] - cf[0, 1] > 0.4
    assert infof["align"]["heading"].endswith("FLIPPED")


def test_no_camera_falls_back():
    raw, _ = _raw(world_tilt=False)
    for k in ("w2c", "K", "depth0", "mask0"):
        raw.pop(k)
    X, M, info = align(raw)
    assert info["align"]["up"]["src"].startswith("camera-0 up")
    check_tracks(X, M)


def test_frame_cap_and_file_roundtrip(tmp_path):
    raw, _ = _raw(T=200, N=100)
    X, M, info = align(raw)
    assert X.shape == (176, 100 - info["align"]["outliers_dropped"], 3)
    p = tmp_path / "tracks.npz"
    save_tracks(p, X, M)
    with np.load(p, allow_pickle=False) as d:
        assert sorted(d.files) == ["tracks", "visibility"]
    X2, M2 = load_tracks(p)
    assert np.array_equal(X, X2) and np.array_equal(M, M2)
    assert info["align"]["T"] == 176


def test_checks_reject_what_the_rig_step_rejects():
    X = np.random.default_rng(0).normal(size=(10, 20, 3)).astype(np.float32)
    M = np.ones((10, 20), bool)
    check_tracks(X, M)
    bad = M.copy()
    bad[0, 3] = False
    with pytest.raises(ValueError, match="frame 0"):
        check_tracks(X, bad)
    cut = X.copy()
    cut[5:] += 10.0
    with pytest.raises(ValueError, match="camera cut"):
        check_tracks(cut, M)
    nan = X.copy()
    nan[3, 3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        check_tracks(nan, M)
    with pytest.raises(ValueError, match="max 176"):
        check_tracks(np.zeros((177, 20, 3), np.float32) + X[:1], np.ones((177, 20), bool))


def test_hidden_samples_may_hold_any_finite_value():
    """Any finite value at a hidden sample is accepted and written as 0; the 1e6 limit is for visible samples."""
    X = np.random.default_rng(0).normal(size=(10, 20, 3)).astype(np.float32)
    M = np.ones((10, 20), bool)
    M[4:7, :5] = False
    zero, big = X.copy(), X.copy()
    zero[~M] = 0.0
    big[~M] = 1e7
    a, b = check_tracks(zero, M), check_tracks(big, M)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    vis = X.copy()
    vis[1, 1] = 1e7
    with pytest.raises(ValueError, match="exceed"):
        check_tracks(vis, M)
    for v in (np.nan, np.inf):
        bad = X.copy()
        bad[~M] = v
        with pytest.raises(ValueError, match="NaN or Inf"):
            check_tracks(bad, M)


def test_cli_align(tmp_path, capsys):
    raw, _ = _raw()
    rp = tmp_path / "raw.npz"
    np.savez(rp, **raw)
    out = tmp_path / "tracks.npz"
    assert main(["align", str(rp), "-o", str(out)]) == 0
    assert main(["check", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    assert json.load(open(tmp_path / "tracks.json"))["align"]["up"]["src"] == "ground RANSAC"


def test_checks_match_track2rig():
    """Same verdicts as the rig package's own checks, when both are installed in one environment."""
    t2r = pytest.importorskip("mashbox.models.track2rig.tracks")
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 30, 3)).astype(np.float32)
    cases = [(base, np.ones((12, 30), bool))]
    m = np.ones((12, 30), bool)
    m[0, 1] = False
    cases.append((base, m))
    cut = base.copy()
    cut[6:] += 20.0
    cases.append((cut, np.ones((12, 30), bool)))
    cases.append((np.zeros((12, 30, 3), np.float32), np.ones((12, 30), bool)))
    cases.append((np.zeros((177, 30, 3), np.float32) + base[:1], np.ones((177, 30), bool)))
    hidden = np.ones((12, 30), bool)
    hidden[3:6, :4] = False
    far = base.copy()
    far[~hidden] = 1e7
    cases.append((far, hidden))
    far_visible = base.copy()
    far_visible[0, 0] = 1e7
    cases.append((far_visible, hidden))
    for X, M in cases:
        ours = theirs = None
        try:
            check_tracks(X, M)
        except ValueError as e:
            ours = str(e).split(" (")[0].split(";")[0]
        try:
            t2r.check_tracks(X, M)
        except ValueError as e:
            theirs = str(e).split(" (")[0].split(";")[0]
        assert (ours is None) == (theirs is None), (ours, theirs)


def test_opposite_up_axes_give_a_rotation():
    """Exactly opposite vectors: a 180-degree turn (det +1), never the reflection -I."""
    from track2rig_video.align import _rot_to
    for u in ([0, 0, -1.0], [0, 1.0, 0], [1.0, 2.0, -3.0]):
        u = np.array(u)
        R = _rot_to(u, -u)
        assert np.isclose(np.linalg.det(R), 1.0) and np.allclose(R @ R.T, np.eye(3))
        assert np.allclose(R @ (u / np.linalg.norm(u)), -u / np.linalg.norm(u))


def test_ground_fit_falls_back_with_too_few_background_pixels():
    from track2rig_video.align import _ground_up
    rng = np.random.default_rng(0)
    D = np.ones((4, 4))
    m0 = np.ones((4, 4), bool)
    m0[0, 0] = False                       # one background pixel
    raw = dict(w2c=np.eye(4)[None], K=np.array([[[2.0, 0, 2], [0, 2.0, 2], [0, 0, 1]]]), depth0=D, mask0=m0)
    log = {}
    up, floor = _ground_up(raw, rng.normal(size=(10, 3)), rng, log)
    assert floor is None and "fewer than 3" in log["up"]["src"]
    assert np.allclose(up, [0, -1, 0])


@pytest.mark.parametrize("step,frames", [(0, 10), (-1, 10), (1, 0)])
def test_read_video_refuses_bad_steps(step, frames):
    from track2rig_video.trackeverything import read_video
    with pytest.raises(ValueError, match="at least 1"):
        read_video("unused.mp4", max_frames=frames, frame_step=step)


def test_read_video_stops_decoding_at_the_last_frame_it_keeps(monkeypatch):
    """Frames are decoded one at a time and decoding stops at the last frame kept; the whole video is never
    decoded first (a long clip could fill memory before --max-frames applied)."""
    import sys
    import types
    from track2rig_video.trackeverything import read_video
    pulled = []

    def imiter(path, plugin=None):
        for i in range(10_000):              # a long video: far more frames than are kept
            pulled.append(i)
            yield np.full((2, 2, 3), i, np.uint8)

    def imread(*a, **k):
        raise AssertionError("read_video decoded the whole video")

    v3 = types.ModuleType("imageio.v3")
    v3.imiter, v3.imread = imiter, imread
    pkg = types.ModuleType("imageio")
    pkg.v3 = v3
    monkeypatch.setitem(sys.modules, "imageio", pkg)
    monkeypatch.setitem(sys.modules, "imageio.v3", v3)
    frames = read_video("long.mp4", max_frames=3, frame_step=2)
    assert [int(f[0, 0, 0]) for f in frames] == [0, 2, 4]
    assert pulled == [0, 1, 2, 3, 4]         # source frames 0..4 only: frame 4 is the last one kept


def test_cli_refuses_frame_step_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["clip.mp4", "-o", "t.npz", "--frame-step", "0"])
    assert e.value.code == 2


def test_raw_file_checks(tmp_path):
    from track2rig_video.raw import load_raw
    good = dict(coords=np.zeros((5, 9, 3)), visib=np.ones((5, 9)))
    p = tmp_path / "r.npz"
    np.savez(p, **good, w2c=np.zeros((5, 4, 3)))
    with pytest.raises(ValueError, match="w2c shape"):
        load_raw(p)
    np.savez(p, **good, K=np.full((5, 3, 3), np.nan))
    with pytest.raises(ValueError, match="K contains"):
        load_raw(p)


def _no_reads(monkeypatch):
    """Make any array read from an .npz fail loudly, to prove a loader refuses a file before reading it."""
    def boom(self, key):
        raise AssertionError(f"read {key} before the size check")
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", boom)


def test_tracks_file_member_size_is_checked_before_reading(tmp_path, monkeypatch):
    p = tmp_path / "big.npz"
    np.savez_compressed(p, tracks=np.zeros((177, 2048, 3)), visibility=np.ones((177, 2048), bool))  # 8,700,032 bytes
    _no_reads(monkeypatch)
    with pytest.raises(ValueError, match="uncompressed"):
        load_tracks(p)


def test_raw_file_member_size_is_checked_before_reading(tmp_path, monkeypatch):
    from track2rig_video import raw as rawmod
    p = tmp_path / "r.npz"
    np.savez_compressed(p, coords=np.zeros((5, 9, 3)), visib=np.ones((5, 9)), depth0=np.zeros((64, 64)))
    monkeypatch.setattr(rawmod, "MAX_MEMBER_BYTES", 20000)      # depth0 is 32,896 bytes, the others fit
    _no_reads(monkeypatch)
    with pytest.raises(ValueError, match="depth0.npy is .* bytes uncompressed"):
        rawmod.load_raw(p)


def test_raw_file_total_size_is_capped_before_reading(tmp_path, monkeypatch):
    from track2rig_video import raw as rawmod
    p = tmp_path / "r.npz"
    np.savez_compressed(p, coords=np.zeros((5, 9, 3)), visib=np.ones((5, 9)), depth0=np.zeros((40, 40)), mask0=np.zeros((40, 40), bool))
    monkeypatch.setattr(rawmod, "MAX_TOTAL_BYTES", 12000)       # every array fits on its own; together they do not
    _no_reads(monkeypatch)
    with pytest.raises(ValueError, match="in all"):
        rawmod.load_raw(p)


def test_raw_file_takes_only_documented_arrays(tmp_path, monkeypatch):
    from track2rig_video.raw import load_raw
    p = tmp_path / "r.npz"
    np.savez(p, coords=np.zeros((5, 9, 3)), visib=np.ones((5, 9)), notes=np.zeros(3))
    with pytest.raises(ValueError, match="unexpected arrays.*notes"):
        load_raw(p)


@pytest.mark.parametrize("v", [np.nan, np.inf, -np.inf])
def test_raw_visibility_must_be_finite(tmp_path, v):
    from track2rig_video.raw import load_raw
    raw, _ = _raw(T=20, N=60)
    raw["visib"][3, 4] = v
    p = tmp_path / "r.npz"
    np.savez(p, **raw)
    with pytest.raises(ValueError, match="visib contains NaN or Inf"):
        load_raw(p)
    with pytest.raises(ValueError, match="visib contains NaN or Inf"):
        align(raw)                       # the Python API checks too, for arrays that never went through a file


@pytest.mark.parametrize("thr", [float("nan"), float("inf"), -0.1, 1.5])
def test_vis_thr_must_be_finite_and_within_0_1(thr):
    raw, _ = _raw(T=20, N=60)
    with pytest.raises(ValueError, match="vis_thr"):
        align(raw, vis_thr=thr)


@pytest.mark.parametrize("thr", ["nan", "inf", "-0.1", "1.5", "x"])
def test_cli_refuses_bad_vis_thr(tmp_path, thr):
    raw, _ = _raw(T=20, N=60)
    rp = tmp_path / "raw.npz"
    np.savez(rp, **raw)
    for argv in (["align", str(rp), "-o", str(tmp_path / "t.npz")], ["clip.mp4", "-o", str(tmp_path / "t.npz")]):
        with pytest.raises(SystemExit) as e:
            main(argv + ["--vis-thr", thr])
        assert e.value.code == 2
    assert not (tmp_path / "t.npz").exists()


def test_vis_thr_edges_are_accepted():
    raw, _ = _raw(T=20, N=60)
    for thr in (0.0, 1.0):
        X, M, _ = align(raw, vis_thr=thr)
        check_tracks(X, M)


class _FakeHub:
    """Stands in for huggingface_hub (no network): records snapshot_download calls and serves a local folder."""
    def __init__(self, folder, files=("config.json", "model.safetensors"), fail=None):
        self.folder, self.calls, self.fail = folder, [], fail
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            (folder / f).write_text("x")

    def module(self):
        import types
        m = types.ModuleType("huggingface_hub")

        def snapshot_download(repo_id, **kw):
            self.calls.append(dict(repo_id=repo_id, **kw))
            if self.fail:
                raise self.fail
            return str(self.folder)
        m.snapshot_download = snapshot_download
        return m


@pytest.mark.parametrize("offline", [None, "1"])
def test_pi3_is_fetched_at_the_pinned_revision(tmp_path, monkeypatch, offline):
    import sys
    from track2rig_video import trackeverything as te
    hub = _FakeHub(tmp_path / "snap")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    else:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert te.pi3_snapshot(log=lambda *a: None) == str(tmp_path / "snap")
    (call,) = hub.calls
    assert call["repo_id"] == te.PI3_WEIGHTS and call["revision"] == te.PI3_REVISION
    assert call["local_files_only"] is bool(offline)


def test_pi3_offline_without_that_revision_stops(tmp_path, monkeypatch):
    import sys
    from track2rig_video import trackeverything as te
    hub = _FakeHub(tmp_path / "snap", fail=FileNotFoundError("not cached"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(SystemExit, match="not in the Hugging Face cache"):
        te.pi3_snapshot(log=lambda *a: None)
    hub2 = _FakeHub(tmp_path / "partial", files=("config.json",))   # a half-downloaded snapshot of that revision
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub2.module())
    with pytest.raises(SystemExit, match="model.safetensors"):
        te.pi3_snapshot(log=lambda *a: None)


def test_tracker_loads_pi3_from_the_local_snapshot(tmp_path, monkeypatch):
    """track() hands TrackEverything the local folder of the pinned revision, never the repo name (which would load
    whatever the repo holds now)."""
    import sys
    import types
    from track2rig_video import trackeverything as te
    hub = _FakeHub(tmp_path / "snap")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen = {}

    class Stop(Exception):
        pass

    class FakeTE:
        @staticmethod
        def from_pretrained(repo, **kw):
            seen.update(kw, repo=repo)
            raise Stop

    monkeypatch.setattr(te, "_import_te", lambda: FakeTE)
    with pytest.raises(Stop):
        te.track("unused.mp4", log=lambda *a: None)
    assert seen["pi3_checkpoint"] == str(tmp_path / "snap")
    assert seen["repo"] == te.TE_WEIGHTS and seen["revision"] == te.TE_WEIGHTS_REVISION
