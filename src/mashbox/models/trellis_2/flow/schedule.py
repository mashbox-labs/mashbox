"""Host-side (numpy, float64) schedule for the TRELLIS.2 flow Euler sampler.

Matches FlowEulerSampler.sample + GuidanceIntervalSamplerMixin +
ClassifierFreeGuidanceSamplerMixin at TRELLIS.2 commit 75fbf018 (MIT):
- t_seq = linspace(1, 0, steps + 1), then rescale_t * t / (1 + (rescale_t - 1) * t), in float64;
- the guidance interval test uses that float64 t (lo <= t <= hi), outside it the strength is 1;
- strength 1 => one pass on cond, strength 0 => one pass on neg_cond, else CFG (two passes);
- every scalar coefficient is formed in float64 on the host; the sampler rounds it once to the
  latent dtype (float32 in production), which is what torch does when a Python float
  multiplies a float32 tensor. The model timestep 1000*t is always float32, as in the official code.
"""
from dataclasses import dataclass, field
import json
from typing import List, Tuple

import numpy as np

MODE_COND = "cond"   # one pass on cond (strength 1, or outside the interval)
MODE_NEG = "neg"     # one pass on neg_cond (strength 0 inside the interval)
MODE_CFG = "cfg"     # two passes + CFG mix (+ optional rescale)


@dataclass(frozen=True)
class SamplerParams:
    steps: int = 12
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.0
    guidance_interval: Tuple[float, float] = (0.0, 1.0)
    rescale_t: float = 1.0
    sigma_min: float = 1e-5


@dataclass(frozen=True)
class Segment:
    """Consecutive steps that share one mode; run as one lax.scan."""
    mode: str
    t: Tuple[float, ...]          # float64 sampler t of each step
    t_prev: Tuple[float, ...]
    t_model: np.ndarray = field(compare=False)  # f32 [S], 1000*t (model timestep input)
    b: np.ndarray = field(compare=False)        # f64 [S], sigma_min + (1 - sigma_min) * t
    dt: np.ndarray = field(compare=False)       # f64 [S], t - t_prev


def timesteps(steps: int, rescale_t: float) -> List[float]:
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return t_seq.tolist()


def step_mode(t: float, p: SamplerParams) -> str:
    lo, hi = p.guidance_interval
    g = p.guidance_strength if (lo <= t <= hi) else 1
    if g == 1:
        return MODE_COND
    if g == 0:
        return MODE_NEG
    return MODE_CFG


def build_segments(p: SamplerParams) -> Tuple[Segment, ...]:
    t_seq = timesteps(p.steps, p.rescale_t)
    pairs = [(t_seq[i], t_seq[i + 1]) for i in range(p.steps)]
    segs, cur = [], []
    for t, tp in pairs:
        m = step_mode(t, p)
        if cur and cur[-1][0] != m:
            segs.append(cur)
            cur = []
        cur.append((m, t, tp))
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        ts = tuple(x[1] for x in s)
        tps = tuple(x[2] for x in s)
        out.append(Segment(
            mode=s[0][0], t=ts, t_prev=tps,
            t_model=np.array([1000 * t for t in ts], dtype=np.float64).astype(np.float32),
            b=np.array([p.sigma_min + (1 - p.sigma_min) * t for t in ts], dtype=np.float64),
            dt=np.array([t - tp for t, tp in zip(ts, tps)], dtype=np.float64),
        ))
    return tuple(out)


def model_calls(p: SamplerParams) -> int:
    return sum((2 if s.mode == MODE_CFG else 1) * len(s.t) for s in build_segments(p))


def load_pipeline_samplers(path: str) -> dict:
    """Read the pinned pipeline.json -> {'sparse_structure': SamplerParams, 'shape_slat': ..., 'tex_slat': ...}."""
    args = json.load(open(path))["args"]
    out = {}
    for key in ("sparse_structure", "shape_slat", "tex_slat"):
        s = args[f"{key}_sampler"]
        assert s["name"] == "FlowEulerGuidanceIntervalSampler", s["name"]
        pr = s["params"]
        out[key] = SamplerParams(
            steps=int(pr["steps"]),
            guidance_strength=float(pr["guidance_strength"]),
            guidance_rescale=float(pr.get("guidance_rescale", 0.0)),
            guidance_interval=tuple(float(v) for v in pr["guidance_interval"]),
            rescale_t=float(pr["rescale_t"]),
            sigma_min=float(s["args"]["sigma_min"]),
        )
    return out
