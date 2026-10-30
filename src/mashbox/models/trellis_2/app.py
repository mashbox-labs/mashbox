"""`mashbox run trellis-2` and `mashbox download trellis-2`.

  run trellis-2 --image in.png --out out.glb     one RGBA image in, one textured GLB mesh out (512 pipeline)
  download trellis-2                              download and check every file only

The image encoder, the three flow models and the decoders run on the TPU. The GLB export (remesh, simplify,
UV unwrap, texture bake) runs on the host CPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def log(s):
    print(s, file=sys.stderr, flush=True)


def parser(cmd):
    p = argparse.ArgumentParser(prog="mashbox %s trellis-2" % cmd)
    if cmd == "download":
        return p
    p.add_argument("--image", required=True, help="input picture: RGBA with the background cut out (alpha = 0 around the object)")
    p.add_argument("--out", required=True, help="output .glb file")
    p.add_argument("--seed", type=int, default=42, help="noise seed (default %(default)s)")
    p.add_argument("--cache-dir", help="XLA compile cache directory (default ~/.cache/mashbox/xla/trellis-2)")
    p.add_argument("--report", help="also write the per-stage times and the output's sha256 to this JSON file")
    return p


def read_image(path):
    """The input as an RGBA PIL image, or SystemExit with one plain line (checked before any download)."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(path) as im:
            im.load()
            image = im.copy()
    except (OSError, UnidentifiedImageError) as e:
        raise SystemExit(f"{path}: not a readable image ({e})")
    if image.mode != "RGBA":
        raise SystemExit(f"{path}: TRELLIS.2 needs an RGBA image with the background cut out (this one is {image.mode})")
    a = np.asarray(image)[:, :, 3]
    if not np.any(a > 204) or np.all(a > 204):        # stage 0 crops to alpha > 0.8: it needs object and background
        raise SystemExit(f"{path}: the alpha channel must mark the object (opaque) and the background (transparent)")
    from .preprocess import preprocess_rgba
    try:
        side = min(preprocess_rgba(image).size)
    except (ValueError, SystemError):
        side = 0
    if side < 2:
        raise SystemExit(f"{path}: the opaque part is too small to crop around; the object needs a few pixels at least")
    return image


def make_driver(cache_dir, **reduced):
    from .driver import Trellis2Driver
    cache_dir = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "mashbox", "xla", "trellis-2")
    return Trellis2Driver(cache_dir=cache_dir, pipeline="512", **reduced)


def generate(driver, weights_root, verified, image, out, seed, progress=log):
    """Load, run every stage once and write the GLB. Returns the run summary (also what --report writes)."""
    import jax
    t0 = time.perf_counter()
    load = driver.load(weights_root, verified=verified,
                       progress=lambda e: progress(json.dumps(dict(event="load", **e))))
    ready_s = time.perf_counter() - t0
    progress(json.dumps(dict(event="ready", seconds=round(ready_s, 1), backend=jax.default_backend())))
    cached = _count(driver.cache_dir)
    try:
        res = driver.generate(image, out, seed=seed,
                              progress=lambda e: progress(json.dumps(dict(event="stage", **e))))
        stats = jax.devices()[0].memory_stats() or {}
    finally:
        driver.unload()
    m = res["metadata"]
    return dict(path=res["path"], sha256=m["sha256"], bytes=m["bytes"], vertices=m["glb"]["vertices"],
                faces=m["glb"]["faces"], seed=seed, pipeline=driver.pipeline, backend=m["backend"],
                ready_s=round(ready_s, 2), compile_ms=round(load["compile_ms"], 1),
                stage_wall_s={k: round(v, 3) for k, v in m["stage_wall_s"].items()},
                device_s=round(m["device_stage_wall_s"], 3), host_s=round(m["host_time_s"], 3),
                device_peak_gb=round(stats["peak_bytes_in_use"] / 1e9, 2) if "peak_bytes_in_use" in stats else None,
                compiles_after_ready=_count(driver.cache_dir) - cached)


def _count(d):
    """Entries in the compile cache: a program compiled after the start adds one."""
    return sum(1 for e in os.scandir(d) if not e.name.endswith("-atime")) if d and os.path.isdir(d) else 0


def main(cmd, argv):
    a = parser(cmd).parse_args(argv)
    from . import weights as W
    if cmd == "download":
        root, verified = W.stage(log)
        print(json.dumps(dict(root=root, files=len(verified)), indent=1))
        return 0
    image = read_image(a.image)
    if not a.out.lower().endswith(".glb"):
        raise SystemExit(f"--out {a.out}: the output is a GLB file; give it a .glb name")
    root, verified = W.stage(log)
    out = generate(make_driver(a.cache_dir), root, verified, image, a.out, a.seed)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out))
    return 0
