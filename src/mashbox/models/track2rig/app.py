"""Command line for Track2Rig, reached through `mashbox run track2rig ...` and `mashbox download track2rig`.

  mashbox run track2rig TRACKS.npz -o OUT       rig one tracks file: OUT/rig.json and OUT/rig.glb
  mashbox run track2rig eval [CLIPS.npz ...]     score Track2Rig (and SSDR) on clips with a true rig
  mashbox run track2rig info                     versions, config and the weights check
  mashbox download track2rig                     download and check the weights only
"""
import argparse
import glob
import json
import math
import os
import sys
import time

# Published tolerances for `eval` against the recorded rows (docs/models/track2rig.md). Above them, eval exits 1.
TOLERANCE = dict(recon_heldout=1e-5, recon_tracked=1e-5, j2j=1e-3, bones=0)
EXAMPLE_CLIPS = os.path.join("examples", "track2rig", "clips")


def log(s):
    print(s, file=sys.stderr, flush=True)


def _weights(a):
    from .weights import weights_path
    p = weights_path(a.weights, log)
    os.environ["MASHBOX_TRACK2RIG_WEIGHTS"] = p  # eval worker processes read it
    return p


def _rig(a):
    from . import load
    t0 = time.perf_counter()
    m = load(_weights(a))
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = m.rig_file(a.tracks, max_bones=a.max_bones)
    t_rig = time.perf_counter() - t0
    out = a.out or os.path.splitext(os.path.basename(a.tracks))[0] + "_rig"
    pj, pg = r.save(out, fps=a.fps, name=a.name)
    q = r.quality
    print(f"{q['frames']} frames, {q['points']} points -> {r.num_bones} bones, skeleton depth {q['skeleton_max_depth']}, "
          f"{q['skeleton_leaves']} leaves")
    print(f"re-posing error vs input tracks: mean {q['recon_err_pct_diag']:.3f}% of body size (input noise included)")
    if r.flags:
        print("flags:", ", ".join(r.flags))
    for n in r.notes:
        print("note:", n)
    print(f"load {t_load:.2f} s, rig {t_rig:.2f} s (model {r.timings_ms['model'] / 1e3:.2f} s incl. first-call compile)")
    print("wrote", pj)
    print("wrote", pg)
    return 0


def _example_clips():
    """Shipped example clips: examples/track2rig/clips/*.npz under the current folder (a clone with the clips copied in),
    else the files[] rows of kind "example" from the model's Hugging Face repo at the pinned revision."""
    local = sorted(glob.glob(os.path.join(EXAMPLE_CLIPS, "*.npz")))
    if local:
        return local, os.path.join(EXAMPLE_CLIPS, "reference.json")
    from ... import hub
    from .weights import entry
    m = entry()
    if m.get("hf_revision") is None:
        raise SystemExit("no clips given, none in examples/track2rig/clips/, and the example clips are not on Hugging Face "
                         "at a pinned revision in this version of mashbox; pass clip files")
    got = {f["name"]: hub.verify(hub.fetch(m["hf_repo"], f["name"], m["hf_revision"]), f["sha256"], log)
           for f in m["files"] if f.get("kind") == "example"}
    clips = [got[n] for n in sorted(got) if n.endswith(".npz")]
    ref = next((got[n] for n in got if n.endswith("reference.json")), None)
    return clips, ref


def _eval(a):
    from . import eval as ev
    _weights(a)
    if a.clips:
        paths, ref = a.clips, os.path.join(os.path.dirname(a.clips[0]), "reference.json")
    else:
        paths, ref = _example_clips()
    ref = a.reference or ref
    if not a.clean:
        try:
            ev.load_reference(ref)  # before the run: a missing or broken reference is an error, never a skipped check
        except ValueError as e:
            print(f"REFERENCE ERROR: {e}")
            return 1
    t0 = time.perf_counter()
    try:
        rows = ev.run(paths, with_ssdr=not a.no_ssdr, noisy=not a.clean, jobs=a.jobs, out=a.out)
    except RuntimeError as e:
        print(f"EVAL ERROR: {e}")
        return 1
    print()
    print(ev.table(rows))
    print(f"wall {time.perf_counter() - t0:.1f} s")
    if a.clean:
        print("not compared: the recorded rows are for noisy, partly hidden tracks (drop --clean to compare)")
        return 0
    try:
        worst, lines = ev.compare(rows, ref)
    except ValueError as e:
        print(f"REFERENCE ERROR: {e}")
        return 1
    print("\n".join(lines))
    over = [k for k, d in worst.items() if d > TOLERANCE[k]]
    if over:
        print("OVER TOLERANCE: " + ", ".join(f"{k} {worst[k]:.3g} > {TOLERANCE[k]:g}" for k in over))
        return 1
    print("all within tolerance: " + ", ".join(f"{k} {v:g}" for k, v in TOLERANCE.items()))
    return 0


def _info(a):
    import jax
    import numpy
    from . import __version__
    from .model import CONFIG
    from .weights import WEIGHTS_SHA256, sha256_file
    p = _weights(a)
    got = sha256_file(p)
    print(json.dumps(dict(track2rig=__version__, config=CONFIG, weights=p, weights_sha256=got,
                          weights_sha256_ok=got == WEIGHTS_SHA256, jax=jax.__version__, numpy=numpy.__version__,
                          devices=[str(d) for d in jax.devices()]), indent=1))
    return 0


def _fps(s):
    v = float(s)
    if not (math.isfinite(v) and v > 0):
        raise argparse.ArgumentTypeError(f"must be a finite number above 0, got {s}")
    return v


def _stem(s):
    if not s or s in (".", "..") or "/" in s or "\\" in s or os.sep in s or (os.altsep and os.altsep in s):
        raise argparse.ArgumentTypeError(f"must be a file name without a folder, got {s!r}")
    return s


def _bones(s):
    v = int(s)
    if not 1 <= v <= 32:
        raise argparse.ArgumentTypeError(f"must be 1 to 32, got {s}")
    return v


def parser(cmd="run"):
    """Options of `mashbox run track2rig` (cmd "run"), its `eval` and `info` forms, and `mashbox download track2rig`."""
    weights_help = "a local copy of track2rig-v2.safetensors instead of the download (its sha256 is still checked)"
    if cmd in ("download", "info"):
        p = argparse.ArgumentParser(prog=f"mashbox {'download' if cmd == 'download' else 'run'} track2rig{' info' if cmd == 'info' else ''}",
                                    description="Download and check the weights." if cmd == "download" else "Versions, config and the weights check.")
        p.add_argument("--weights", metavar="FILE", help=weights_help)
        return p
    if cmd == "eval":
        p = argparse.ArgumentParser(prog="mashbox run track2rig eval", description="Score Track2Rig and SSDR on clips with a true rig.")
        p.add_argument("clips", nargs="*", help="example clip .npz files (default: the shipped example clips)")
        p.add_argument("--no-ssdr", action="store_true", help="skip the SSDR baseline (about 26 s per clip on an Intel Core i9-12900K)")
        p.add_argument("--clean", action="store_true", help="no tracking noise or hidden samples")
        p.add_argument("--jobs", type=int, default=1, metavar="N", help="clips in parallel (processes)")
        p.add_argument("--out", metavar="FILE", help="append per-clip rows to this .jsonl")
        p.add_argument("--reference", metavar="FILE", help="recorded rows to compare with (default: reference.json next to the clips)")
        p.add_argument("--weights", metavar="FILE", help=weights_help)
        return p
    p = argparse.ArgumentParser(prog="mashbox run track2rig", description="3D point tracks in, rig out (CPU).",
                                epilog="Also: `mashbox run track2rig eval [CLIPS]` and `mashbox run track2rig info`.")
    p.add_argument("tracks", help=".npz with tracks float32 [T,N,3] and optional visibility bool [T,N]")
    p.add_argument("-o", "--out", metavar="DIR", help="output folder (default: <tracks name>_rig)")
    p.add_argument("--max-bones", type=_bones, default=32, metavar="N", help="keep at most this many bones (1 to 32, default 32)")
    p.add_argument("--fps", type=_fps, default=30.0, metavar="FPS", help="frame rate written into the GLB animation (default 30)")
    p.add_argument("--name", type=_stem, default="rig", metavar="STEM", help="output file name without extension (default rig)")
    p.add_argument("--weights", metavar="FILE", help=weights_help)
    return p


def main(cmd, argv):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # f32 on the CPU, also on a TPU VM (the recorded rows are CPU rows)
    argv = list(argv)
    if cmd == "download":
        a = parser("download").parse_args(argv)
        print(_weights(a))
        return 0
    if argv[:1] == ["eval"]:
        return _eval(parser("eval").parse_args(argv[1:]))
    if argv[:1] == ["info"]:
        return _info(parser("info").parse_args(argv[1:]))
    return _rig(parser().parse_args(argv))
