"""TRELLIS.2 on TPU, stage 10: GLB export + texture bake with no CUDA library.

Modules (our own code, except simplify: a port of CuMesh's simplify path; remesh and meshops follow CuMesh's documented
behaviour; CuMesh's LICENSE ships unchanged beside them as LICENSE-CuMesh; helper libraries are permissive):
  meshops   hole fill, duplicate / degenerate / small-component removal, non-manifold repair, orientation, normals
  simplify  QEM simplification (port of the CuMesh simplify call path)
  unwrap    normal-cone chart clustering (ours) + xatlas (MIT) parameterize and pack: the two steps of CuMesh's uv_unwrap
  raster    UV-space rasterizer + attribute interpolation in JAX
  closest   exact closest point on a triangle mesh (JAX point-triangle kernel, host KD-tree candidates)
  sample    sparse-voxel trilinear sampler in JAX
  to_glb    the to_glb pipeline (standard branch) + GLB writing with trimesh (MIT)
"""
