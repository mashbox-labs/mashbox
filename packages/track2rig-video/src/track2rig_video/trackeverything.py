# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""Optional wrapper around TrackEverything (CMU and Meta), the 3D point tracker we use.

TrackEverything's code and weights are licensed CC BY-NC 4.0: NON-COMMERCIAL USE ONLY.
None of its code or weights are in this repository. packages/track2rig-video/install_trackeverything.sh
clones it from its own GitHub repo at the commit below, and its weights download from its
own Hugging Face repo at the revision below. Using it means accepting its licence.
For commercial work, get the authors' permission or bring tracks from another tracker
(see `track2rig-video align --help`).

Geometry (depth and camera per frame) comes from Pi3, which TrackEverything loads itself.
Pi3's code is BSD-3-Clause. For its weights, Pi3's GitHub README lists CC BY-NC 4.0
(non-commercial), while the Hugging Face card is tagged BSD-2-Clause and its README says
BSD for academic use and to contact the authors for commercial use: treat them as
non-commercial. The wrapper downloads Pi3's weights at a pinned revision and hands TrackEverything
that local folder, so a newer upload to Pi3's repo is never loaded.
"""
import os
import sys
import time

TE_REPO = "https://github.com/ayushjain1144/trackeverything"
TE_COMMIT = "f0c18dd9714b9f85825c45a1c3aae685da4e6222"
TE_WEIGHTS = "katefgroup/TrackEverything"
TE_WEIGHTS_REVISION = "1957810f46ef684154fec91a9b8c0e6a68cd695e"
PI3_WEIGHTS = "yyfz233/Pi3"
PI3_REVISION = "ae722e7039287d0c8fde9f11f197f804f44b510c"  # pinned Pi3 weights revision
LICENCE_NOTE = ("TrackEverything (CMU and Meta) is licensed CC BY-NC 4.0: non-commercial use only. Pi3's weights: "
                "treat as non-commercial (see README). Their code and weights are not part of this package.")


PI3_FILES = ("config.json", "model.safetensors")  # what Pi3.from_pretrained reads from a local folder


def _offline():
    """HF_HUB_OFFLINE as huggingface_hub reads it."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")


def pi3_snapshot(log=print):
    """Pi3's weights at PI3_REVISION as a local folder. TrackEverything's Pi3 loader takes a folder as well as a repo
    name, so passing this folder makes it load exactly that revision. Online: download it (or reuse the cached copy).
    Offline (HF_HUB_OFFLINE=1): the cached copy of that exact revision, or stop."""
    from huggingface_hub import snapshot_download
    offline = _offline()
    try:
        path = snapshot_download(PI3_WEIGHTS, revision=PI3_REVISION, allow_patterns=list(PI3_FILES),
                                 local_files_only=offline)
    except Exception as e:
        if offline:
            raise SystemExit(f"{PI3_WEIGHTS} at revision {PI3_REVISION} is not in the Hugging Face cache and "
                             "HF_HUB_OFFLINE is set. Run once online to download it.") from e
        raise SystemExit(f"could not download {PI3_WEIGHTS} at revision {PI3_REVISION} ({type(e).__name__}: {e})") from e
    missing = [f for f in PI3_FILES if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise SystemExit(f"{PI3_WEIGHTS} at revision {PI3_REVISION}: {', '.join(missing)} missing from {path}"
                         + (". It is not in the Hugging Face cache and HF_HUB_OFFLINE is set; run once online." if offline else ""))
    log(f"Pi3 weights: {PI3_WEIGHTS} at revision {PI3_REVISION}")
    return path


def _import_te():
    src = os.environ.get("TRACKEVERYTHING_SRC")  # the cloned repo dir, named 'trackeverything'
    if src:
        parent = os.path.dirname(os.path.abspath(src))
        if parent not in sys.path:
            sys.path.insert(0, parent)
    try:
        from trackeverything.demo_hf import TrackEverything
    except ImportError as e:
        raise SystemExit("TrackEverything is not installed. Run packages/track2rig-video/install_trackeverything.sh, then set "
                         "TRACKEVERYTHING_SRC to the cloned 'trackeverything' folder.\n" + LICENCE_NOTE) from e
    return TrackEverything


def read_video(path, max_frames=176, frame_step=1):
    """Every frame_step-th frame from the start, at most max_frames of them. Frames are decoded one at a time and
    decoding stops at the last frame needed, so a long video never sits in memory whole."""
    if int(frame_step) < 1 or int(max_frames) < 1:
        raise ValueError(f"frame_step and max_frames must be at least 1 (got {frame_step}, {max_frames})")
    import imageio.v3 as iio
    import numpy as np
    out = []
    for i, f in enumerate(iio.imiter(path, plugin="pyav")):
        if i % frame_step == 0:
            out.append(f)
            if len(out) == max_frames:
                break
    if not out:
        raise SystemExit(f"{path}: no frames decoded")
    return np.stack(out)


def track(video_path, n_points=1024, max_frames=176, frame_step=1, mask_path=None, device="cuda",
          rembg_model="birefnet-general", log=print):
    """Video -> raw tracks dict (see raw.py) for the subject visible at frame 0."""
    import numpy as np
    import torch
    from .subject import sample_queries, subject_mask

    log(LICENCE_NOTE)
    TrackEverything = _import_te()
    tm = {}
    t0 = time.time()
    pi3_dir = pi3_snapshot(log)
    tm["pi3_fetch_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    model = TrackEverything.from_pretrained(TE_WEIGHTS, revision=TE_WEIGHTS_REVISION, device=device,
                                            geometry_backend="pi3", pi3_checkpoint=pi3_dir)
    tm["load_s"] = round(time.time() - t0, 2)
    if device.startswith("cuda") and torch.cuda.get_device_capability()[0] < 8:
        g = model.geometry  # pre-Ampere GPUs have no bf16 flash attention: run the geometry model in fp16
        if hasattr(g, "autocast_dtype"):
            g.autocast_dtype = "float16"
            log("geometry autocast -> float16 (GPU older than sm_80)")
    frames = read_video(video_path, max_frames, frame_step)
    T, H, W = frames.shape[:3]
    log(f"video {T} frames {W}x{H}")
    t0 = time.time()
    depths, intr, extr, conf = model._estimate_geometry(frames)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    tm["geometry_s"] = round(time.time() - t0, 2)
    depths, intr, extr = np.asarray(depths), np.asarray(intr), np.asarray(extr)
    if depths.shape[1:3] != (H, W):
        raise SystemExit(f"geometry resolution {depths.shape[1:3]} != video {(H, W)}")
    t0 = time.time()
    mask0 = subject_mask(frames[0], mask_path, rembg_model)
    q2d = sample_queries(mask0, n_points, seed=0)
    tm["mask_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    out = model.infer(frames, depths=depths, intrinsics=intr, extrinsics=extr,
                      conf_mask=None if conf is None else np.asarray(conf), query_points_2d=q2d, mode="sparse")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    tm["track_s"] = round(time.time() - t0, 2)
    raw = dict(coords=out.query_coords[0].float().cpu().numpy(), visib=out.query_visibs[0].float().cpu().numpy(),
               depth0=out.depths[0].float().cpu().numpy(), K=out.intrinsics.float().cpu().numpy(),
               w2c=out.extrinsics.float().cpu().numpy(), mask0=mask0, q2d=q2d)
    info = dict(tracker="TrackEverything", tracker_commit=TE_COMMIT, tracker_weights=f"{TE_WEIGHTS}@{TE_WEIGHTS_REVISION}",
                geometry=f"{PI3_WEIGHTS}@{PI3_REVISION}", frames=int(T),
                width=int(W), height=int(H), frame_step=int(frame_step),
                points_requested=int(n_points), points_tracked=int(len(q2d)), timings=tm)
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return raw, info
