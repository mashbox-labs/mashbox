"""Our to_glb: same inputs, steps and output layout as o_voxel.postprocess.to_glb at the pin (standard branch,
remesh=False), with every CUDA piece replaced by our code. Inputs are numpy arrays (no torch needed).

Step map (official -> ours):
  CuMesh fill_holes / cleanup         -> meshops.*
  cuBVH(vertices, faces)              -> closest.ClosestPointMesh (built on the hole-filled ORIGINAL mesh, as official)
  CuMesh simplify (3x, then target)   -> simplify.simplify (port of the CuMesh simplify call path)
  CuMesh unify_face_orientations      -> meshops.unify_face_orientations
  CuMesh uv_unwrap (+xatlas)          -> unwrap.uv_unwrap (our clustering + xatlas)
  GPU rasterize (100k chunks)         -> raster.rasterize_uv (JAX)
  GPU interpolate                     -> raster.interpolate
  bvh.unsigned_distance + uvw         -> ClosestPointMesh.query (closest point itself)
  flex_gemm grid_sample_3d trilinear  -> sample.grid_sample_trilinear (JAX)
  cv2.inpaint TELEA, trimesh GLB      -> same calls (OpenCV Apache-2.0, trimesh MIT)
remesh=True: CuMesh remesh_narrow_band_dc -> remesh.remesh_narrow_band_dc (our code; UDF from ClosestPointMesh, never
cubvh), then ONE simplify to the target, as official (no cleanup, no orientation pass in that branch); doubleSided
False. remesh_fallback=True still runs the standard branch instead and records that in the stats.
"""
import time
import os

import numpy as np

from . import meshops, simplify as simp, unwrap, raster, closest, sample, remesh as rm

# Closest-point query knobs. Defaults 0 / 1 = the standard path.
BAKE_KEXT = int(os.environ.get("MASHBOX_TRELLIS_BAKE_KEXT", "0"))
BAKE_WORKERS = int(os.environ.get("MASHBOX_TRELLIS_BAKE_WORKERS", "1"))


def _grid(aabb, voxel_size, grid_size):
    aabb = np.asarray(aabb, np.float32).reshape(2, 3)
    if voxel_size is not None:
        vs = np.asarray([voxel_size] * 3 if np.isscalar(voxel_size) else voxel_size, np.float32)
        gs = np.round((aabb[1] - aabb[0]) / vs).astype(np.int32)
    else:
        gs = np.asarray([grid_size] * 3 if np.isscalar(grid_size) else grid_size, np.int32)
        vs = ((aabb[1] - aabb[0]) / gs).astype(np.float32)
    return aabb, vs, gs


def bake(out_vertices, out_faces, out_uvs, orig_V, orig_F, attr_volume, coords, aabb, voxel_size, grid_size,
         texture_size, cp_index=None, stats=None):
    """The texture-baking half of to_glb, from an already simplified + unwrapped mesh. Returns (attrs [R,R,C], mask)."""
    aabb, vs, gs = _grid(aabb, voxel_size, grid_size)
    t = time.time()
    rast = raster.rasterize_uv(out_uvs, out_faces, texture_size)
    mask = rast[..., 3] > 0
    t_r = time.time() - t
    pos = raster.interpolate(out_vertices, rast, out_faces)
    valid_pos = pos[mask]
    t = time.time()
    if cp_index is None:
        cp_index = closest.ClosestPointMesh(orig_V, orig_F, k_ext=BAKE_KEXT, workers=BAKE_WORKERS)
    snapped, _, _, cst = cp_index.query(valid_pos, return_stats=True)
    t_c = time.time() - t
    t = time.time()
    q = (snapped - aabb[0]) / vs
    vals = sample.grid_sample_trilinear(np.asarray(attr_volume, np.float32), np.asarray(coords)[:, -3:], tuple(gs), q)
    t_s = time.time() - t
    attrs = np.zeros((texture_size, texture_size, vals.shape[1]), np.float32)
    attrs[mask] = vals
    if stats is not None:
        stats.update(t_raster_s=t_r, t_closest_s=t_c, t_sample_s=t_s, covered_texels=int(mask.sum()), closest=cst)
    return attrs, mask, rast


def finalize_maps(attrs, mask, attr_layout):
    """Official post-processing: clip*255 -> uint8, TELEA inpaint of uncovered texels (radius 3 colour, 1 others)."""
    import cv2

    def sl(k):
        s = attr_layout[k]
        return s if isinstance(s, slice) else slice(*s)

    base_color = np.clip(attrs[..., sl('base_color')] * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., sl('metallic')] * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., sl('roughness')] * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., sl('alpha')] * 255, 0, 255).astype(np.uint8)
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    return base_color, metallic, roughness, alpha


def make_trimesh(vertices, faces, uvs, normals, base_color, metallic, roughness, alpha, double_sided=True):
    """Same material + axis conversion as the official code (trimesh writes the GLB)."""
    import trimesh
    import trimesh.visual
    from PIL import Image
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=double_sided,
    )
    v = np.array(vertices, np.float32, copy=True)
    n = np.array(normals, np.float32, copy=True)
    uv = np.array(uvs, np.float32, copy=True)
    v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()
    n[:, 1], n[:, 2] = n[:, 2].copy(), -n[:, 1].copy()
    uv[:, 1] = 1 - uv[:, 1]
    return trimesh.Trimesh(vertices=v, faces=faces, vertex_normals=n, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))


def to_glb(vertices, faces, attr_volume, coords, attr_layout, aabb, voxel_size=None, grid_size=None,
           decimation_target=1000000, texture_size=2048, remesh=False, remesh_fallback=False, verbose=False,
           return_stats=False, remesh_band=1.0, remesh_project=0.9):
    st = {}
    st["remesh_requested"] = bool(remesh)
    T0 = time.time()
    V, F = meshops._as(vertices, faces)
    st["in"] = [int(V.shape[0]), int(F.shape[0])]
    t = time.time()
    V, F, fh = meshops.fill_holes(V, F, 3e-2, return_stats=True)
    st["fill0"] = fh
    st["t_fill_s"] = time.time() - t
    orig_V, orig_F = V, F
    t = time.time()
    cp_index = closest.ClosestPointMesh(orig_V, orig_F, k_ext=BAKE_KEXT, workers=BAKE_WORKERS)
    st["t_index_s"] = time.time() - t
    t = time.time()
    if remesh and not remesh_fallback:
        ab, _, gs = _grid(aabb, voxel_size, grid_size)
        center = ((ab[0] + ab[1]) / np.float32(2)).astype(np.float32)
        ext = float((ab[1] - ab[0]).max())
        res = int(np.max(gs))
        V, F, rst = rm.remesh_narrow_band_dc(V, F, center, (res + 3 * remesh_band) / res * ext, res, band=remesh_band,
                                             project_back=remesh_project, cp=cp_index, return_stats=True)
        st["remesh"] = rst
        V, F, s1 = simp.simplify(V, F, decimation_target, return_stats=True)
        st["simplify"] = [s1]
    else:
        V, F, s1 = simp.simplify(V, F, decimation_target * 3, return_stats=True)
        V, F = meshops.cleanup(V, F)
        V, F, s2 = simp.simplify(V, F, decimation_target, return_stats=True)
        V, F = meshops.cleanup(V, F)
        V, F = meshops.unify_face_orientations(V, F)
        st["simplify"] = [s1, s2]
    st["after_post"] = [int(V.shape[0]), int(F.shape[0])]
    st["t_simplify_clean_s"] = time.time() - t
    t = time.time()
    V, F = meshops.remove_degenerate_faces(V, F)
    vn = meshops.vertex_normals(V, F)
    out_v, out_f, out_uv, vmap, ust = unwrap.uv_unwrap(V, F, return_stats=True)
    out_n = vn[vmap]
    st["unwrap"] = ust
    st["t_unwrap_s"] = time.time() - t
    attrs, mask, _ = bake(out_v, out_f, out_uv, orig_V, orig_F, attr_volume, coords, aabb, voxel_size, grid_size,
                          texture_size, cp_index=cp_index, stats=st)
    t = time.time()
    maps = finalize_maps(attrs, mask, attr_layout)
    st["t_inpaint_s"] = time.time() - t
    tm = make_trimesh(out_v, out_f, out_uv, out_n, *maps, double_sided=not remesh)
    st["out"] = [int(out_v.shape[0]), int(out_f.shape[0])]
    st["t_total_s"] = time.time() - T0
    if verbose:
        print(st, flush=True)
    return (tm, st) if return_stats else tm
