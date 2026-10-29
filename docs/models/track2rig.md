# Track2Rig

Dense 3D point tracks of a moving creature go in. A rig comes out: bones, skin weights (top 4 per point), per-frame bone transforms and a skeleton tree with joint positions, written as `rig.json` and as an animated glTF 2.0 file (`rig.glb`).

Track2Rig is experimental. It does not work well on many clips yet. We plan to keep improving it. Newer weights will be published as dated versions, and a version never changes once published; this release uses weights version `2026-10-29`. See [Known limitations](#known-limitations).

<img alt="Track2Rig preview" width="720" src="https://huggingface.co/mashbox-labs/Track2Rig/resolve/main/assets/hero.gif">

<sub>The quickstart clip (grison_walk.npz, a grison walking, from the synthetic test set) rigged by Track2Rig 0.1.0 (weights track2rig-v2) on a CPU, in a side view and a three-quarter view. Every point is moved only by the predicted skin and bone transforms; lines are the predicted skeleton. Also as an [MP4 video](https://huggingface.co/mashbox-labs/Track2Rig/resolve/main/assets/hero.mp4) (the clip looped).</sub>

## What it is

Track2Rig is our own model, trained from scratch by MASHBOX Labs. There is no template skeleton, so the same model rigs a four-legged mammal and a legless snake. It is one forward pass of a 2.76M-parameter network (11 MB of weights), followed by a few NumPy steps for the skeleton and the exports. Plain JAX and NumPy on the CPU: no GPU, no TPU and no custom compiled kernels.

## What you need

| | Tested | Not tested |
|---|---|---|
| Rig step (`mashbox run track2rig`) | x86-64 Linux (Ubuntu 24.04) with Python 3.12.3: Intel Core i9-12900K, Intel Core i5-13500 and AMD EPYC 7502P; the AMD EPYC 9B14 host of a Google Cloud TPU VM (Ubuntu 22.04, Python 3.12 installed with uv) | macOS, Windows, Arm CPUs, Python 3.13 and later |
| Video step (`track2rig-video`, optional) | Linux with one NVIDIA RTX PRO 4000 Blackwell (24 GB) or one NVIDIA L4 (24 GB), Python 3.10 | other GPUs, macOS, Windows |
| Opening `rig.glb` in an editor | Linux (Ubuntu 24.04): Blender 4.2.9 LTS, and Unreal Engine 5.8.3 both from the GLB and from an FBX that Blender exported | Windows and macOS editors, other editor versions, other engines, retargeting |

Cost: none beyond your own machine. The rig step runs on any of the CPUs above in about a second per clip.

## Install

Track2Rig needs Python 3.12. Get the code and the example clips:

```bash
git clone https://github.com/mashbox-labs/mashbox && cd mashbox
```

If `python3 --version` prints 3.12 (Ubuntu 24.04, for example):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install ".[cpu,track2rig]"
```

If it prints an older version, as on a Google Cloud TPU VM (Python 3.10), let [uv](https://docs.astral.sh/uv/) fetch Python 3.12, as the [Qwen3.8-27B page](qwen3-8-27b.md) does:

```bash
python3 -m pip install uv
python3 -m uv venv --python 3.12 .venv
python3 -m uv pip install --python .venv/bin/python ".[cpu,track2rig]"
. .venv/bin/activate
```

`cpu` adds JAX for the CPU (pinned, as for every model in this package) and `track2rig` adds SciPy. In a fresh virtual environment with no pip cache, `pip install` took 11 s on the i9-12900K (fast network).

## Run

```bash
mashbox run track2rig examples/track2rig/tracks/grison_walk.npz -o grison_rig
```

This writes `grison_rig/rig.glb` and `grison_rig/rig.json` and prints the bone count and fit numbers. It took 1.0 s on the i9-12900K, compile included. The first run downloads `track2rig-v2.safetensors` (11 MB) from [huggingface.co/mashbox-labs/Track2Rig](https://huggingface.co/mashbox-labs/Track2Rig) at the revision pinned in `models.json` and checks its sha256; `--weights FILE` uses a local copy instead (checked the same way). `examples/track2rig/tracks/` has three ready walk clips: a grison, a pelican spider and a glass snake.

In the GLB each tracked point is a small tetrahedron coloured by its main bone and moved only by the rig, so what you see moving is the rig itself. The file has one skin, a joint per bone and inverse bind matrices, and each point's skin weights sum to 1. It opens in Blender and in Unreal Engine on Linux (see the FAQ).

### Python

```python
import numpy as np
from mashbox.models import track2rig

model = track2rig.load()          # pinned download, sha256 checked; or track2rig.load("track2rig-v2.safetensors")
d = np.load("examples/track2rig/tracks/grison_walk.npz")
rig = model.rig(d["tracks"], d["visibility"])

rig.num_bones                     # bones the model used (at most 32)
rig.weights                       # [N, B] skin weights, at most 4 non-zero per point
rig.transforms                    # [T, B, 3, 4] frame 0 -> frame t, x_t = R x_0 + t
rig.parent, rig.joints            # skeleton tree and joint positions at frame 0
rig.quality, rig.flags            # fit numbers and input warnings
rig.save("grison_rig", fps=30)    # rig.json + rig.glb
```

`model.rig(..., max_bones=12)` keeps the 12 heaviest bones, moves points whose bones were dropped to the kept bone that best explains their motion, and re-solves the transforms.

With a TPU or GPU build of JAX in the same environment, set `JAX_PLATFORMS=cpu` before your script imports JAX, for example `JAX_PLATFORMS=cpu python my_script.py`. The model runs on the host CPU either way, but without that setting JAX also starts the TPU or GPU and holds it until your process exits. `mashbox run track2rig` sets it for you.

## Options

<!-- cli:start run track2rig -->
| Option | Default | What it does |
|---|---|---|
| `TRACKS` |  | .npz with tracks float32 [T,N,3] and optional visibility bool [T,N] |
| `--out DIR` |  | output folder (default: <tracks name>_rig) |
| `--max-bones N` | `32` | keep at most this many bones (1 to 32, default 32) |
| `--fps FPS` | `30.0` | frame rate written into the GLB animation (default 30) |
| `--name STEM` | `rig` | output file name without extension (default rig) |
| `--weights FILE` |  | a local copy of track2rig-v2.safetensors instead of the download (its sha256 is still checked) |
<!-- cli:end -->

`mashbox run track2rig eval [CLIPS]` scores Track2Rig and SSDR on clips with a true rig (see Performance). Its options:

<!-- cli:start run track2rig eval -->
| Option | Default | What it does |
|---|---|---|
| `CLIPS` |  | example clip .npz files (default: the shipped example clips) |
| `--no-ssdr` | off | skip the SSDR baseline (about 26 s per clip on an Intel Core i9-12900K) |
| `--clean` | off | no tracking noise or hidden samples |
| `--jobs N` | `1` | clips in parallel (processes) |
| `--out FILE` |  | append per-clip rows to this .jsonl |
| `--reference FILE` |  | recorded rows to compare with (default: reference.json next to the clips) |
| `--weights FILE` |  | a local copy of track2rig-v2.safetensors instead of the download (its sha256 is still checked) |
<!-- cli:end -->

`mashbox run track2rig info` prints the versions, the model config and the weights check. `mashbox download track2rig` downloads and checks the weights only.

### Input

A `.npz` file (or two arrays in Python):

| Array | Type | Shape | Meaning |
|---|---|---|---|
| `tracks` | float32 | [T, N, 3] | 3D position of each tracked point in each frame, any unit |
| `visibility` | bool | [T, N] | optional; False where the tracker lost the point. The value stored there is ignored (any finite number) |

- Frame 0 is the bind pose and every point must be visible in it.
- Every value must be finite (no NaN or Inf). Visible positions must be at most 1e6 in magnitude; a hidden sample can hold any finite value: it is set to 0 before the range check and never used.
- T is at most 176 frames per call (split or resample longer clips). N is 8 to 2,048 points; the model was trained with 1,024 points sampled evenly over the body surface, so spread the points the same way.
- A frame where most points jump by more than half the body size is rejected as a camera cut.

### Output

`rig.json` (`format: track2rig.rig/1`):

| Field | Meaning |
|---|---|
| `bones`, `root`, `parent` | bone count, root bone, parent of each bone (-1 for the root) |
| `joints` | bone head positions at frame 0 (the joint with its parent; the centroid for the root) |
| `skin.index`, `skin.weight` | [N][4] per tracked point, in input order; unused slots have weight 0 |
| `transforms.data` | [T][B][3][4] row-major `[R|t]`, frame 0 to frame t |
| `quality` | re-posing error against the input tracks, motion and visibility numbers, skeleton depth and leaves |
| `flags` | warnings such as `depth_squashed`, `low_motion`, `heavy_occlusion`, `few_bones`, `high_recon_error`, `short_clip` (heuristic thresholds, listed in `thresholds`) |

`rig.glb`: a skin with one joint node per bone (parented as in `parent`), inverse bind matrices, and one animation with a translation and a rotation channel per bone (linear keys). The mesh is one small tetrahedron per tracked point, skinned with the same top-4 weights. There are no normals.

`rig.json` uses the frame of the input tracks, which Track2Rig expects to be Z up with the creature facing -Y (the frame it was trained in). `rig.glb` uses the glTF 2.0 frame, Y up with the front toward +Z: each track point (x, y, z) is written as (x, z, -y).

## From video

Track2Rig starts from 3D point tracks. To make them from a video of one animal we use [TrackEverything](https://github.com/ayushjain1144/trackeverything) (CMU and Meta) at commit `f0c18dd9714b9f85825c45a1c3aae685da4e6222`, licensed **CC BY-NC 4.0 (non-commercial use only)**, through our wrapper, the Apache-2.0 package `track2rig-video` in [packages/track2rig-video](../../packages/track2rig-video/README.md). Its geometry model, [Pi3](https://huggingface.co/yyfz233/Pi3), is pinned to one revision; treat its weights as non-commercial too (details in the package README). Neither project's code or weights are in this repo: the install script fetches them from their authors.

The tracker needs Python 3.10 and `mashbox` needs Python 3.12, so the example makes two environments. From the repository root, on Linux with an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and git:

```bash
bash examples/track2rig/video-to-rig/setup.sh
bash examples/track2rig/video-to-rig/run.sh bear-example
```

1. **Video to tracks** (GPU): `track2rig-video VIDEO -o tracks.npz` cuts out the animal at frame 0, tracks 1,024 points on it through the clip and puts them in the frame Track2Rig was trained in (Z up, facing -Y, moving in place).
2. **Tracks to rig** (CPU): `mashbox run track2rig tracks.npz -o rig` writes `rig/rig.json` and `rig/rig.glb`.

On one NVIDIA RTX PRO 4000 Blackwell SFF Edition (24 GB) with an Intel Core i5-13500, the example (a public CC BY-SA 3.0 clip of a bear walking, 60 frames at 1280x720) took 53 s end to end with weights cached and 190 s on the first run, which also downloads about 5.4 GB of tracker, geometry and mask weights. Peak GPU memory was 9.2 GB.

Tracks from any other 3D point tracker work through `track2rig-video align RAW.npz -o tracks.npz`, which runs on the CPU (input format in the package README).

## Variants

One file: `track2rig-v2.safetensors`, float32, 55 tensors, 2,764,416 parameters, 11 MB, sha256 `f8a79263dc21d2af078c123fbf76c2cba85df4694100f6a6c5e8f7608f6094b6`. There is no reduced-size variant: the whole model is 11 MB and loads in a fraction of a second.

## Performance

Score: held-out reconstruction error. The rig re-poses surface points that were never tracked (each takes the skin of its nearest tracked point), and we measure their mean distance to the true positions, in % of the creature's bounding-box diagonal. Lower is better. Joint error is the symmetric Chamfer distance between the rig's joints and the true joints at frame 0, in the same unit.

Inputs carried tracking noise (Gaussian, 0.5% of the diagonal), and at least 20% of track samples after frame 0 were hidden, in runs of 5 to 30 frames. SSDR (Le and Deng 2012) is our own implementation (`src/mashbox/models/track2rig/ssdr.py`: 30 iterations, one initialisation, heuristic sparse weights), run at the same bone count as Track2Rig on the same inputs. The synthetic set's 16 creatures are not in the training data. The artist set is 43 clips of 11 species whose artist clips were left out of training. We also used these test sets to pick which version to release, so they are not untouched estimates and the comparisons below are exploratory.

With the weights above, on an AMD EPYC 7502P CPU (Track2Rig on JAX for CPU, SSDR on NumPy and SciPy, 8 clips in parallel for the synthetic set and 6 for the artist set, on a shared machine):

| Test set | Clips | Track2Rig | SSDR | Track2Rig lower on | Joint error, Track2Rig / SSDR | Median time per clip, Track2Rig / SSDR |
|---|---|---|---|---|---|---|
| Synthetic creatures (16 unseen) | 32 | 0.784% | 0.915% | 31 of 32 | 6.24% / 7.73% | 0.18 s / 105 s |
| Artist-rigged animals | 43 | 1.32% | 1.20% | 23 of 43 | 4.25% / 5.33% | 0.17 s / 89 s |

Times are medians per clip. Track2Rig's time is the network and bone fits; the skeleton read-out and the exports are not included. SSDR here is our own single-process NumPy/SciPy implementation; an optimised SSDR such as Dem Bones will be much faster and is not in this comparison.

### Reproduce on the example clips

Six clips from the synthetic test set, with their true rigs, are in the Hugging Face repo under `examples/clips/`: walk and idle for a grison (a four-legged mustelid, 30 true bones), a pelican spider (eight legs, 58 true bones) and a glass snake (legless, 27 true bones), the first four-legged, many-legged and legless creatures in test-set order. [examples/track2rig/clips/reference.json](../../examples/track2rig/clips/reference.json) has the reference rows for them.

```bash
mashbox run track2rig eval --no-ssdr      # Track2Rig only, a few seconds
mashbox run track2rig eval --jobs 6       # plus SSDR: about 26 to 28 s per clip on the i9-12900K
```

Example output (fresh virtual environment, Intel Core i9-12900K CPU, Python 3.12.3, jax 0.11.2, the six clips in a local `clips/` folder):

```text
$ mashbox run track2rig eval --no-ssdr
clip                 held-out % T2R joints % T2R bones secs T2R
---------------------------------------------------------------
glass-snake_idle             0.6835       13.360    32     0.45
glass-snake_walk             0.6483       13.365    32     0.06
grison_idle                  0.6699        4.329    32     0.06
grison_walk                  0.7779        3.368    32     0.06
pelican-spider_idle          0.7118        6.081    32     0.06
pelican-spider_walk          1.3939        4.733    32     0.06
---------------------------------------------------------------
mean                         0.8142

$ mashbox run track2rig eval clips/grison_walk.npz
clip                 held-out % T2R    SSDR joints % T2R    SSDR bones secs T2R   SSDR
--------------------------------------------------------------------------------------
grison_walk                  0.7779  0.8692        3.368   3.987    32     0.45   28.2
```

`eval` compares every row with its recorded row and exits with an error when a difference is above 1e-5 (held-out and tracked error), 1e-3 (joint error) or 0 (bones). It checks the Track2Rig rows and the true-rig rows. SSDR rows are printed but not checked: SSDR starts from rigid fits that frames with only 1 or 2 visible points leave undetermined, so on another CPU it can settle in a different solution. It also exits with an error, before any scoring, when the reference file is missing, has no rows, has two rows for one clip and method, or has a row without all four numbers, and after scoring when a clip and method it scored has no recorded row. With `--clean` it does not compare: the recorded rows are for noisy, partly hidden tracks. In the output above the largest differences were 3.3e-7, 2.2e-7, 3.9e-5 and 0. The first clip of a run includes the one-time compile.

## Known limitations

- **Clean input.** Given perfect tracks, SSDR fits better: Track2Rig 0.147% vs SSDR 0.090% on the synthetic set and 0.770% vs 0.278% on the artist rigs, with SSDR lower on every clip. Track2Rig is ahead on the synthetic set's noisy, partly hidden tracks, and on time.
- **Artist-rigged animals.** On that set SSDR's mean held-out error is lower (1.20% vs 1.32%), while Track2Rig places joints closer (4.25 vs 5.33) and is lower on 23 of 43 clips.
- **Joint placement.** Joints are approximate: mean joint error is 6.24% of body size on the synthetic set and 4.25% on the artist rigs. The model has 32 bone slots and used all 32 on every synthetic test clip, so creatures with more true bones (the pelican spider has 58) get a coarser rig and creatures with fewer get extra bones.
- **Speckled skin.** Neighbouring points can take different main bones, most visibly on the trunk (see the grison in the preview).
- **Real video.** Real footage has no ground-truth rig, so there is no real-video accuracy number here. Tracks from one camera cover only the visible side of the body.
- **Clip length and compile.** At most 176 frames per call. Each new point count N compiles the network once (about 0.5 s on the i9-12900K).
- **Small differences between CPUs and runs.** Results differ slightly from CPU to CPU: on the CPUs listed in [What you need](#what-you-need), joint error stays within 2e-4 of the recorded rows. On one machine the result also depends on how many CPU cores the process may use. On the i9-12900K (24 threads), repeated runs pinned to the same cores (for example `taskset -c 0`) gave identical output, and different core counts gave slightly different results; with all cores free, two runs of the same clip can differ. The largest differences were about 1e-8 (relative) in the fit and 9.9e-4 in a raw bone transform, on a bone with little skin. Both are far inside the `eval` tolerances above.
- **Flags.** The thresholds behind `flags` are heuristics, not calibrated on a measured set.

## Use it with your coding agent

Point the agent at this page and the example: "Install mashbox with the cpu and track2rig extras, run `mashbox run track2rig examples/track2rig/tracks/grison_walk.npz -o grison_rig`, then load `grison_rig/rig.json`." Every command on this page runs without a GPU except the video step.

## Troubleshooting

| Error | Fix |
|---|---|
| `The Track2Rig weights are not on Hugging Face at a pinned revision in this version of mashbox` | pass a local copy with `--weights track2rig-v2.safetensors` |
| `sha256 ... does not match the pinned ...` | the file is damaged or is another file; delete it and download again |
| `tracks has T=... frames; max 176` | split or resample the clip |
| `frame 0 is the bind pose and must be fully visible` | drop the points not seen at frame 0, or start the clip at a frame where all are seen |
| `frame t->t+1: median point jump ...` | a camera cut or a lost subject; cut the clip before that frame |
| `... bytes uncompressed; the most a tracks file needs is ...` | the archive holds arrays far larger than a valid tracks file; check what wrote it |
| `--fps: must be a finite number above 0` | give the clip's real frame rate |
| `REFERENCE ERROR: ...` after `eval` | the reference file is missing or does not hold one complete row per clip and method; pass the shipped `reference.json` with `--reference`, or the clips it was recorded for |
| `OVER TOLERANCE` after `eval` | a result moved further from the recorded row than the published tolerance; report it with the output of `mashbox run track2rig info` |

## FAQ

**Can I use it in Unreal Engine?** Yes, tested on Linux. `rig.glb` is a standard glTF 2.0 skin (a joint per bone, 4 weights per vertex, linear keys, no normals). Unreal Engine 5.8.3 on Ubuntu 24.04 (editor run headless) imports the quickstart's `rig.glb` through its glTF importer with no errors and one warning (the mesh has no material). It makes a Skeletal Mesh, a Skeleton with a bone for every rig bone and an Animation Sequence with the clip's length and a key per frame. Every vertex of the mesh has bone weights, and every bone's pose changes between the first frame and mid-clip. An FBX of the same rig, exported from Blender, imports the same way with the same results. Not tested: Unreal on Windows or macOS, retargeting to another skeleton, or playing the animation in a level.

**Can I open it in Blender?** Yes, tested on Linux. Blender 4.2.9 LTS on Ubuntu 24.04 (run headless, stock glTF importer) imports `rig.glb` as an armature with a bone for every rig bone and the point mesh with an Armature modifier and a weighted vertex group per bone. The action has a key per frame and the mesh moves between frames. Set the scene frame rate to the one the rig was written with (`--fps`) before importing, so the keys land on whole frames. Blender also adds its own bone display shape (an icosphere), which is not in the file. Not tested: Blender on Windows or macOS, or other Blender versions.

**Can I use it commercially?** Track2Rig's code and weights are Apache-2.0. The optional video step's tracker (TrackEverything) and geometry weights (Pi3) are non-commercial; for commercial work, get their authors' permission or bring tracks from another tracker through `track2rig-video align`.

**Does it need a GPU or TPU?** No. On a TPU VM it runs on the host CPU.

## Changelog

- **0.1.0**: first release in `mashbox`.

## Credits and licences

- **Track2Rig** code and weights: Apache-2.0, MASHBOX Labs. Trained from scratch on one TPU v6e chip, 12,000 steps (about 2 hours), on 6,981 animated clips: 2,266 clips of 913 procedurally generated creatures of our own, 727 clips of 56 species from an artist-made animal animation library, and 3,988 clips from 3,399 animated 3D assets in public collections (3,724 of the clips from Objaverse). All training tracks were simulated from the animated meshes. The training data and training code are not in this repo.
- **Objaverse:** Matt Deitke et al., *Objaverse: A Universe of Annotated 3D Objects*, CVPR 2023.
- **SSDR:** Binh Huy Le and Zhigang Deng, *Smooth Skinning Decomposition with Rigid Bones*, ACM Transactions on Graphics 31(6), SIGGRAPH Asia 2012.
- **Skeleton from bone motion:** Binh Huy Le and Zhigang Deng, *Robust and Accurate Skeletal Rigging from Mesh Sequences*, ACM Transactions on Graphics 33(4), SIGGRAPH 2014.
- **Weighted rigid fit:** W. Kabsch, *A solution for the best rotation to relate two sets of vectors*, Acta Crystallographica A32, 1976.
- Built on [JAX](https://github.com/jax-ml/jax), [NumPy](https://numpy.org) and [SciPy](https://scipy.org).
- **Video step:** TrackEverything (CC BY-NC 4.0), Pi3 (code BSD-3-Clause, weights: see the package README), rembg and BiRefNet (MIT), and the example bear clip (CC BY-SA 3.0) are fetched from their authors when you run it, never included here. Full credits: [packages/track2rig-video/README.md](../../packages/track2rig-video/README.md).

Made by [MASHBOX Labs](https://mashbox.gg). MASHBOX Labs makes open models run faster and cheaper on TPUs and accelerated hardware, as well as exploring the future of the intersection of AI and gaming.
