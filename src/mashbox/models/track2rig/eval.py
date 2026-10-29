"""Score Track2Rig and the SSDR baseline (at Track2Rig's bone count) on clips with a ground-truth rig."""
import json
import math
import os
import time

import numpy as np

from .bench import evaluate, gt_rig, load_example

_PARAMS = None


def _params():
    global _PARAMS
    if _PARAMS is None:
        from .weights import load_params
        _PARAMS = load_params(os.environ.get("MASHBOX_TRACK2RIG_WEIGHTS"))
    return _PARAMS


def clip_name(path):
    return os.path.basename(path)[:-4] if path.endswith(".npz") else os.path.basename(path)


def run_clip(path, with_ssdr=True, noisy=True):
    """Rows for one example clip: the true rig (floor), Track2Rig, and SSDR at Track2Rig's bone count."""
    from .model import predict
    from .ssdr import ssdr
    c = load_example(path, noisy=noisy)
    name = clip_name(path)
    cond = "noisy" if noisy else "clean"
    rows = []
    W, R, P = gt_rig(c)
    rows.append(dict(method="gt_rig", **evaluate(c, W, R, P)))
    t0 = time.perf_counter()
    W, Rt, P = predict(_params(), c["X"], c["mask"])
    tp = time.perf_counter() - t0
    alive = W.sum(0) > 1e-3
    W, Rt = W[:, alive], Rt[:, alive]
    rows.append(dict(method="track2rig", secs=round(tp, 3), **evaluate(c, W, Rt, P)))
    if with_ssdr:
        B = int(alive.sum())
        t0 = time.perf_counter()
        Ws, Rs, Ps, _ = ssdr(c["X"], c["mask"], B)
        rows.append(dict(method="ssdr", B_in=B, secs=round(time.perf_counter() - t0, 1), **evaluate(c, Ws, Rs, Ps)))
    return [dict(clip=name, cond=cond, **r) for r in rows]


def _job(args):
    return run_clip(*args)


def run(paths, with_ssdr=True, noisy=True, jobs=1, out=None, log=print, job=_job):
    """Score the clips; rows come back in the order of `paths`. With jobs > 1 the clips run in worker processes, and a
    worker that dies (crash, kill, out of memory) stops the whole run at once with an error instead of waiting forever."""
    jobs_args = [(p, with_ssdr, noisy) for p in paths]
    rows = []
    fo = open(out, "a") if out else None

    def take(r):
        rows.extend(r)
        log(f"{r[0]['clip']} done")
        if fo:
            fo.writelines(json.dumps(x) + "\n" for x in r)

    try:
        if jobs > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            from concurrent.futures.process import BrokenProcessPool
            ex = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"))
            futs = [ex.submit(job, a) for a in jobs_args]
            try:
                for a, f in zip(jobs_args, futs):
                    try:
                        take(f.result())
                    except BrokenProcessPool as e:
                        raise RuntimeError(f"a worker process died while scoring the clips (first clip without a "
                                           f"result: {os.path.basename(a[0])}); no rows were written for the clips still "
                                           f"running. Run again with --jobs 1 to see that clip's own error.") from e
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
        else:
            for a in jobs_args:
                take(job(a))
    finally:
        if fo:
            fo.close()
    return rows


def table(rows):
    by = {}
    for r in rows:
        by.setdefault(r["clip"], {})[r["method"]] = r
    has_ssdr = all("ssdr" in m for m in by.values())
    if has_ssdr:
        hdr = (f"{'clip':<20} {'held-out % T2R':>14} {'SSDR':>7} {'joints % T2R':>12} {'SSDR':>7} {'bones':>5} "
               f"{'secs T2R':>8} {'SSDR':>6}")
    else:
        hdr = f"{'clip':<20} {'held-out % T2R':>14} {'joints % T2R':>12} {'bones':>5} {'secs T2R':>8}"
    lines = [hdr, "-" * len(hdr)]
    for clip, m in by.items():
        a = m["track2rig"]
        if has_ssdr:
            s = m["ssdr"]
            lines.append(f"{clip:<20} {a['recon_heldout']:>14.4f} {s['recon_heldout']:>7.4f} {a['j2j']:>12.3f} {s['j2j']:>7.3f} "
                         f"{a['bones']:>5d} {a['secs']:>8.2f} {s['secs']:>6.1f}")
        else:
            lines.append(f"{clip:<20} {a['recon_heldout']:>14.4f} {a['j2j']:>12.3f} {a['bones']:>5d} {a['secs']:>8.2f}")
    lines.append("-" * len(hdr))
    t2r = [m["track2rig"]["recon_heldout"] for m in by.values()]
    if has_ssdr:
        ss = [m["ssdr"]["recon_heldout"] for m in by.values()]
        better = sum(a < b for a, b in zip(t2r, ss))
        lines.append(f"{'mean':<20} {np.mean(t2r):>14.4f} {np.mean(ss):>7.4f}")
        lines.append(f"Track2Rig lower held-out error on {better}/{len(t2r)} clips")
    else:
        lines.append(f"{'mean':<20} {np.mean(t2r):>14.4f}")
    return "\n".join(lines)


METRICS = ("recon_heldout", "recon_tracked", "j2j", "bones")


def load_reference(path):
    """The recorded rows as (source, {(clip, method): row}). ValueError when the file is missing, has no rows, has two
    rows for one (clip, method), or a row without a number for every metric: a broken reference never passes."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"reference file not found: {path} (pass --reference FILE)")
    with open(path) as f:
        ref = json.load(f)
    rows = ref.get("rows") if isinstance(ref, dict) else None
    if not rows or not isinstance(rows, list):
        raise ValueError(f"{path} has no recorded rows")
    recs = {}
    for r in rows:
        k = (r.get("clip"), r.get("method"))
        if k in recs:
            raise ValueError(f"{path} has two rows for clip {k[0]!r}, method {k[1]!r}")
        for m in METRICS:
            v = r.get(m)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{path}: the row for clip {k[0]!r}, method {k[1]!r} has no number for {m}")
        recs[k] = r
    return ref.get("source", path), recs


CHECKED = ("track2rig", "gt_rig")   # SSDR rows are printed, not checked (see compare)


def compare(rows, reference_path):
    """Compare with the recorded rows for the shipped clips. Returns (max abs diff per metric over the checked methods,
    lines). Checked: Track2Rig and the true-rig floor. SSDR rows are printed but not checked: its 30-iteration
    optimisation starts from rigid fits that some frames leave undetermined, so on another CPU it can settle in a
    different solution. Every scored (clip, method) needs its recorded row (ValueError otherwise); a NaN or Inf result
    counts as an infinite difference."""
    source, recs = load_reference(reference_path)
    by = {}
    for r in rows:
        k = (r["clip"], r["method"])
        if k not in recs:
            raise ValueError(f"no recorded row for clip {k[0]!r}, method {k[1]!r} in {reference_path}")
        w = by.setdefault(r["method"], {})
        for m in METRICS:
            d = abs(float(r[m]) - float(recs[k][m]))
            w[m] = max(w.get(m, 0.0), d if math.isfinite(d) else math.inf)
    worst = {}
    for meth in CHECKED:
        for m, d in by.get(meth, {}).items():
            worst[m] = max(worst.get(m, 0.0), d)
    lines = [f"reference: {source}"]
    for meth, w in by.items():
        note = "" if meth in CHECKED else "  (printed, not checked)"
        lines.append(f"  max |ours - recorded| {meth:<10} " + ", ".join(f"{m} {d:.3g}" for m, d in w.items()) + note)
    return worst, lines
