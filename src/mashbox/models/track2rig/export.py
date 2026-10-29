"""Rig exports: rig.json (all numbers) and rig.glb (glTF 2.0 binary: skinned point cloud + skeleton + animation)."""
import json
import struct

import numpy as np

from . import __version__
from .rig import THRESHOLDS, TOPK, decimals_for, skin_top4
from .weights import WEIGHTS_SHA256

GENERATOR = f"track2rig {__version__} (weights track2rig-v2 sha256:{WEIGHTS_SHA256[:12]})"

# Tracks (and rig.json) are Z up, facing -Y; glTF 2.0 is Y up, facing +Z. rig.glb is written in glTF's frame with the
# change of basis (x, y, z) -> (x, z, -y), as Blender's glTF exporter does: points p -> C p, rotations R -> C R C^T.
GLTF_FROM_TRACK = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)


def mat2quat(R):
    """[...,3,3] proper rotations -> [...,4] unit quaternions (x, y, z, w)."""
    sh = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    c = np.argmax(np.stack([tr, m00, m11, m22], 1), 1)
    q = np.zeros((len(R), 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = np.sqrt(np.maximum(tr + 1, 0)) * 2
        s1 = np.sqrt(np.maximum(1 + m00 - m11 - m22, 0)) * 2
        s2 = np.sqrt(np.maximum(1 + m11 - m00 - m22, 0)) * 2
        s3 = np.sqrt(np.maximum(1 + m22 - m00 - m11, 0)) * 2
        cand = [
            np.stack([(R[:, 2, 1] - R[:, 1, 2]) / s0, (R[:, 0, 2] - R[:, 2, 0]) / s0, (R[:, 1, 0] - R[:, 0, 1]) / s0, 0.25 * s0], 1),
            np.stack([0.25 * s1, (R[:, 0, 1] + R[:, 1, 0]) / s1, (R[:, 0, 2] + R[:, 2, 0]) / s1, (R[:, 2, 1] - R[:, 1, 2]) / s1], 1),
            np.stack([(R[:, 0, 1] + R[:, 1, 0]) / s2, 0.25 * s2, (R[:, 1, 2] + R[:, 2, 1]) / s2, (R[:, 0, 2] - R[:, 2, 0]) / s2], 1),
            np.stack([(R[:, 0, 2] + R[:, 2, 0]) / s3, (R[:, 1, 2] + R[:, 2, 1]) / s3, 0.25 * s3, (R[:, 1, 0] - R[:, 0, 1]) / s3], 1)]
    for i in range(4):
        q[c == i] = cand[i][c == i]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(sh + (4,))


def check_fps(fps):
    fps = float(fps)
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be a finite number above 0, got {fps}")
    return fps


def rig_json(r, fps=30.0):
    fps = check_fps(fps)
    T = r.transforms.shape[0]
    diag = r.quality["bbox_diag"]
    dp = decimals_for(diag)
    idx, val = skin_top4(r.weights)
    Rt = r.transforms
    Rq = np.concatenate([np.round(Rt[..., :3], 6), np.round(Rt[..., 3:], dp)], -1)
    doc = dict(format="track2rig.rig/1", generator=GENERATOR, units="same as input tracks", bind_frame=0,
               fps=float(fps), frames=int(T), points=int(r.bind_points.shape[0]), bones=int(r.num_bones), root=int(r.root),
               bone_model_slot=np.asarray(r.model_slot).tolist(), parent=np.asarray(r.parent).tolist(),
               joints=np.round(r.joints, dp).tolist(), bone_centroids=np.round(r.centroids, dp).tolist(),
               skin=dict(top_k=TOPK, index=idx.tolist(), weight=np.round(val, 5).tolist(),
                         note="per tracked point (input order); index into bones; unused slots are index 0 with weight 0, "
                              "so decode by summing (an index may repeat)"),
               transforms=dict(layout="[T][B][3][4] row-major [R|t], bind (frame 0) -> frame t: x_t = R x_bind + t", data=Rq.tolist()),
               quality=r.quality, flags=list(r.flags), thresholds=THRESHOLDS, notes=list(r.notes))
    return json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()


def rig_glb(r, fps=30.0):
    """One small tetrahedron per tracked point (coloured by bone, skinned with the same top-4 weights), one node per bone,
    and one animation clip of the per-frame bone transforms. The tetrahedron offsets sum to zero, so each skinned
    centroid lands exactly on the point's rig position. Everything is written in glTF's Y-up frame (GLTF_FROM_TRACK)."""
    fps = check_fps(fps)
    C = GLTF_FROM_TRACK
    W, parent = r.weights, np.asarray(r.parent)
    P, jn = r.bind_points @ C.T, r.joints @ C.T
    Rt = np.concatenate([C @ r.transforms[..., :3] @ C.T, (r.transforms[..., 3] @ C.T)[..., None]], -1)
    T, B = Rt.shape[:2]
    N = P.shape[0]
    diag = r.quality["bbox_diag"]
    idx, val = skin_top4(W)
    a = 0.004 * diag
    off = a * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float64) @ C.T
    pos = (P[:, None, :] + off[None]).reshape(-1, 3).astype(np.float32)
    jnts = np.repeat(idx, 4, 0).astype(np.uint16)
    wts = np.repeat(val, 4, 0).astype(np.float32)
    wts /= wts.sum(1, keepdims=True)
    hue = (np.arange(B) * 0.61803398875) % 1.0
    pal = np.stack([np.abs(hue * 6 - 3) - 1, 2 - np.abs(hue * 6 - 2), 2 - np.abs(hue * 6 - 4)], 1).clip(0, 1) * 0.8 + 0.15
    col = np.repeat(W @ pal, 4, 0).astype(np.float32)
    tri = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    ind = (np.arange(N)[:, None, None] * 4 + tri[None]).reshape(-1).astype(np.uint16 if 4 * N < 65536 else np.uint32)
    # bind world of bone b = translate(joint_b); world(t) = Rt_b(t) * bind; local = world_parent^-1 * world
    Rw = Rt[..., :3]
    tw = np.einsum("tbij,bj->tbi", Rw, jn) + Rt[..., 3]
    Rl = Rw.copy()
    tl = tw.copy()
    for b in range(B):
        pa = parent[b]
        if pa >= 0:
            RpT = np.swapaxes(Rw[:, pa], -1, -2)
            Rl[:, b] = RpT @ Rw[:, b]
            tl[:, b] = np.einsum("tij,tj->ti", RpT, tw[:, b] - tw[:, pa])
    ql = mat2quat(Rl)
    for t in range(1, T):   # quaternion sign continuity for interpolation
        flip = (ql[t] * ql[t - 1]).sum(-1) < 0
        ql[t, flip] *= -1
    ibm = np.tile(np.eye(4), (B, 1, 1))
    ibm[:, :3, 3] = -jn
    ibm = np.ascontiguousarray(np.swapaxes(ibm, 1, 2)).astype(np.float32)   # glTF matrices are column-major
    times = (np.arange(T) / float(fps)).astype(np.float32)

    bin_ = bytearray()
    views, accs = [], []

    def add(arr, comp, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        while len(bin_) % 4:
            bin_.append(0)
        v = dict(buffer=0, byteOffset=len(bin_), byteLength=arr.nbytes)
        if target:
            v["target"] = target
        views.append(v)
        bin_.extend(arr.tobytes())
        ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[typ]
        acc = dict(bufferView=len(views) - 1, componentType=comp, count=int(arr.size // ncomp), type=typ)
        if minmax:
            a2 = arr.reshape(-1, ncomp)
            acc["min"] = [float(x) for x in a2.min(0)]
            acc["max"] = [float(x) for x in a2.max(0)]
        accs.append(acc)
        return len(accs) - 1

    a_pos = add(pos, 5126, "VEC3", 34962, True)
    a_j = add(jnts, 5123, "VEC4", 34962)
    a_w = add(wts, 5126, "VEC4", 34962)
    a_c = add(col, 5126, "VEC3", 34962)
    a_i = add(ind, 5123 if ind.dtype == np.uint16 else 5125, "SCALAR", 34963)
    a_ibm = add(ibm, 5126, "MAT4")
    a_t = add(times, 5126, "SCALAR", None, True)
    nodes = []
    for b in range(B):
        nd = dict(name=f"bone_{b}", translation=[float(x) for x in (jn[b] - (jn[parent[b]] if parent[b] >= 0 else 0))])
        ch = [int(c) for c in range(B) if parent[c] == b]
        if ch:
            nd["children"] = ch
        nodes.append(nd)
    nodes.append(dict(name="points", mesh=0, skin=0))
    samplers, channels = [], []
    for b in range(B):
        samplers.append(dict(input=a_t, output=add(tl[:, b].astype(np.float32), 5126, "VEC3"), interpolation="LINEAR"))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=b, path="translation")))
        samplers.append(dict(input=a_t, output=add(ql[:, b].astype(np.float32), 5126, "VEC4"), interpolation="LINEAR"))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=b, path="rotation")))
    while len(bin_) % 4:
        bin_.append(0)
    g = dict(asset=dict(version="2.0", generator=GENERATOR), scene=0, scenes=[dict(nodes=[int(r.root), B])],
             nodes=nodes, meshes=[dict(name="tracked_points", primitives=[dict(mode=4, indices=a_i,
                                  attributes=dict(POSITION=a_pos, JOINTS_0=a_j, WEIGHTS_0=a_w, COLOR_0=a_c))])],
             skins=[dict(joints=list(range(B)), inverseBindMatrices=a_ibm, skeleton=int(r.root))],
             animations=[dict(name="motion", samplers=samplers, channels=channels)],
             buffers=[dict(byteLength=len(bin_))], bufferViews=views, accessors=accs)
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(bin_), b"BIN\x00") + bytes(bin_))
