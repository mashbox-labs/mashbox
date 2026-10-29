# track2rig-video

Video of one moving creature in, dense 3D point tracks of it out, written as the `tracks.npz` that `mashbox run track2rig` reads. This is the step before Track2Rig: it puts the points in space and moves them through the clip. Track2Rig then turns those tracks into bones, skin weights and an animated GLB.

Two parts:

- **The tracker**, TrackEverything (CMU and Meta), run at a pinned commit through a thin wrapper. It is **licensed CC BY-NC 4.0, non-commercial use only**, and none of its code or weights are in this repository. The install script fetches it from its authors.
- **Our glue** (Apache-2.0): the frame-0 subject mask and query points, and `align`, which puts any tracker's output into the frame Track2Rig was trained in (Z up, facing -Y, moving in place) and writes a checked `tracks.npz`.

`align` runs on a CPU and takes tracks from any 3D point tracker, so you can swap the tracker out.

## What you need

| Step | Hardware | Measured |
|---|---|---|
| `track2rig-video` (tracker) | NVIDIA GPU (tested: RTX PRO 4000 Blackwell and L4, 24 GB each) with a driver that supports CUDA 12.9, the CUDA version PyTorch is built for (tested: drivers 615.71.09 and 580.178.04), Linux | 49 s with weights cached (load 1.7 s, geometry 26 s, subject mask 11 s, tracking 8.9 s), 164 s on the first run. Peak GPU memory 9.2 GB |
| `mashbox run track2rig` | a CPU (tested: the same machine's Intel Core i5-13500) | 1.0 s, including the first-call compile |

The times above are on one NVIDIA RTX PRO 4000 Blackwell SFF Edition (24 GB, NVIDIA driver 615.71.09, PyTorch 2.8.0 built for CUDA 12.9) in a machine with an Intel Core i5-13500 and 64 GB RAM, on the example below: 60 frames at 1280x720. The whole example took 53 s end to end with weights cached and 190 s on the first run, which also downloads about 5.4 GB of tracker, geometry and mask weights. Peak RAM 8.7 GiB.

Python: the tracker pins Python 3.10 and older numpy/scipy, while `mashbox` needs Python 3.12, so the two steps live in two environments. `align` and `check` also run in the `mashbox` environment (`pip install ./packages/track2rig-video`).

## Install

From the repository root, with [uv](https://docs.astral.sh/uv/) and git:

```bash
bash examples/track2rig/video-to-rig/setup.sh
```

This makes `.venv-video` (Python 3.10: TrackEverything at commit `f0c18dd9`, its pinned requirements, rembg, this package) and `.venv-rig` (Python 3.12: `mashbox[cpu,track2rig]`). By running it you accept TrackEverything's licence.

By hand, in a Python 3.10 environment:

```bash
bash packages/track2rig-video/install_trackeverything.sh third_party_src
pip install ./packages/track2rig-video
export TRACKEVERYTHING_SRC=$PWD/third_party_src/trackeverything
```

## Run

```bash
.venv-video/bin/track2rig-video my_clip.mp4 -o tracks.npz
.venv-rig/bin/mashbox run track2rig tracks.npz -o rig --fps 30
```

`tracks.npz` holds `tracks` (float32 [T, N, 3]) and `visibility` (bool [T, N]). `tracks.json` next to it records the settings, the tracker, tracker weights and Pi3 revisions, timings and what `align` did (ground plane found or not, heading source, travel removed, points dropped).

The full example, a public CC BY-SA 3.0 clip of a walking bear to a rigged GLB:

```bash
bash examples/track2rig/video-to-rig/run.sh bear-example
```

## Example output

The example clip (a brown bear walking side-on while the camera pans to follow it; its first 5 seconds at every 2nd frame, so 60 frames, 12 per second) gave, on the RTX PRO 4000 above:

- 1,018 of 1,024 points kept, seen in 96.0% of samples. Ground plane found 8.7 degrees from the camera's up direction. The bear travelled 201% of its body size, which `align` removed.
- `mashbox run track2rig`: 32 bones, skeleton 11 levels deep. The rig re-poses the input tracks with a mean error of 1.96% of the body's size (this measures fit to the tracks, tracker noise included, not accuracy against a true rig).
- Runs are not bit-identical on a GPU. A second run on the same machine put co-visible points a median 0.25% of the body's size away from the first run's. The rig's skeleton came out just as deep, with 12 leaves instead of 11.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--points N` | 1024 | points tracked on the subject, 8 to 2048. Track2Rig was trained on 1024 |
| `--max-frames T` | 176 | frames used from the start of the clip, at most 176 (Track2Rig's window) |
| `--frame-step K` | 1 | use every K-th frame, for long or high frame rate clips (frames are decoded one at a time, up to the last one used) |
| `--mask PNG` | rembg | frame-0 subject mask, white on the subject. Without it, rembg (BiRefNet general model) cuts out the subject |
| `--vis-thr P` | 0.5 | a point counts as seen in a frame when the tracker's visibility is above P (0 to 1) |
| `--keep-travel` | off | keep whole-body travel. By default it is removed, since Track2Rig was trained on in-place motion |
| `--flip-heading` | off | turn the subject 180 degrees about the up axis, for when it ends up facing backwards |
| `--raw-out RAW.npz` | none | also save the raw tracker output, so `align` can be re-run with other settings without the GPU |
| `--device` | cuda | torch device for the tracker |

Other commands:

```bash
track2rig-video align RAW.npz -o tracks.npz   # raw tracks from any tracker, CPU only
track2rig-video check tracks.npz              # the checks `mashbox run track2rig` applies
```

## What `align` does

1. **Up axis.** Fits a plane (RANSAC, seeded) to frame-0 background pixels within 35 degrees of the camera's up direction. If more than 10% of them fit, the plane's normal becomes +Z and the floor sits at z = 0. Otherwise it uses the camera's up direction.
2. **Stray points.** Drops points whose frame-0 position is more than 4 median absolute deviations from the middle of the body (depth bleeds at silhouette edges).
3. **Heading and travel.** The heading is the direction of whole-body travel, rotated to -Y. If the subject travels less than 5% of its body size, the frame-0 long axis is used and the sign is a guess (`tracks.json` says so; use `--flip-heading`). Horizontal travel is then subtracted.
4. **Visibility.** Frame 0 is the bind pose and is always visible. Hidden samples are written as 0.

## Bring your own tracker

`align` reads an `.npz` with:

| Array | Shape | Needed |
|---|---|---|
| `coords` | float [T, N, 3], world positions | yes |
| `visib` | float [T, N], probability or logits | yes |
| `w2c` | float [T, 4, 4], world-to-camera, OpenCV axes | for the camera's up direction |
| `K` | float [T, 3, 3], intrinsics | for the ground fit |
| `depth0` | float [H, W], frame-0 depth | for the ground fit |
| `mask0` | bool [H, W], frame-0 subject mask | for the ground fit |
| `q2d` | float [N, 3], query points as [frame, x, y] in pixels | no (`--raw-out` writes it; `align` does not use it) |

Without `w2c`, the world is taken as camera 0 (y down). Without `K`, `depth0` and `mask0`, there is no ground fit.

Every value must be finite (no NaN or Inf). Before any array is read, `align` refuses a file with any other array, with one array over 256 MiB once decompressed, or with more than 512 MiB in all.

## Known limitations

- **Only the visible surface is tracked.** Track2Rig was trained on simulated tracks spread over the whole body, with 0.5% noise. Real tracker output is noisier: on one of our own renders with exact ground truth, tracked points were off by 3.5% (visible) and 7.2% (hidden) of the body's diagonal (TrackEverything with Pi3 geometry, NVIDIA V100).
- **No accuracy number on real video.** Real clips have no ground-truth rig, so the example shows the output only. Accuracy figures for Track2Rig are on simulated clips.
- **Facing is a guess** when the subject stays in one spot. Check the GLB and re-run `align` with `--flip-heading` if it walks backwards.
- **One subject, fully in view at frame 0.** The largest masked region at frame 0 is tracked. Camera cuts break the tracks; `check` stops on a jump larger than half the body size between two frames.
- **Long or large clips can run out of GPU memory** (see Troubleshooting).

## Troubleshooting

| Error | Fix |
|---|---|
| `torch.OutOfMemoryError` in `refine_endpoints` on a 24 GB GPU | a clip of a large animal runs out of memory on an L4 at 165 frames (960x540 or 640x360) and at 120 frames (640x360). Use fewer frames (`--max-frames`, `--frame-step`) or a GPU with more memory |
| `RuntimeError: No available kernel` on V100 or other pre-Ampere GPUs | the wrapper already runs the geometry model in fp16 there; if TrackEverything's own attention layers raise it, use an Ampere or newer GPU |
| `TrackEverything is not installed` | run `packages/track2rig-video/install_trackeverything.sh` and set `TRACKEVERYTHING_SRC` |
| `yyfz233/Pi3 at revision ... is not in the Hugging Face cache and HF_HUB_OFFLINE is set` | run once online (without `HF_HUB_OFFLINE`) so the pinned revision of Pi3's weights is downloaded |
| `rembg is not installed` | `pip install "rembg[gpu]==2.0.69"` or pass `--mask` |
| `frame t->t+1: median point jump ...` | a camera cut or the tracker lost the subject; trim the clip with `--max-frames` to stop before that frame |
| `frame 0 ... must be fully visible` | only from your own raw files: frame 0 has to be the bind pose with every point seen |
| `No matching distribution found for rembg==2.0.86` | rembg 2.0.70 and later need Python 3.11; the tracker's environment is Python 3.10, so we pin 2.0.69 |

## Credits and licences

- **This package** (`packages/track2rig-video/`): Apache-2.0, MASHBOX Labs.
- **TrackEverything**, Ayush Jain, Sreeharsha Paruchuri, Ishita Gupta, Fan Zhang, Tanner Schmidt, Jakob Engel, Katerina Fragkiadaki, Adam W. Harley (CMU and Meta). Code at [ayushjain1144/trackeverything](https://github.com/ayushjain1144/trackeverything) commit `f0c18dd9714b9f85825c45a1c3aae685da4e6222`, weights at [katefgroup/TrackEverything](https://huggingface.co/katefgroup/TrackEverything) revision `1957810f46ef684154fec91a9b8c0e6a68cd695e`. Licence: CC BY-NC 4.0 for both, non-commercial use only. Paper: arXiv 2609.30222.
- **Pi3** geometry, weights from [yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3) at revision `ae722e7039287d0c8fde9f11f197f804f44b510c` (the wrapper downloads that revision and gives TrackEverything the local copy, so later uploads are not used; code at [yyfz/Pi3](https://github.com/yyfz/Pi3)). The Pi3 code is BSD-3-Clause. For the weights, Pi3's GitHub README lists CC BY-NC 4.0 (non-commercial), while the Hugging Face card is tagged BSD-2-Clause and its README says BSD for academic use and to contact the authors for commercial use; treat them as non-commercial.
- **rembg** (MIT) with the **BiRefNet** general model (MIT), for the frame-0 subject mask.
- **Example clip**: "Ours marchant à Juraparc 720p", a bear walking in Juraparc, Switzerland, by Clément Bucco-Lechat (own work, 2012). [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ours_marchant_%C3%A0_Juraparc_720p.ogv). Downloaded by the example script, not stored here.

Commercial use of the video step needs permission from the TrackEverything and Pi3 authors, or tracks from another tracker through `align`. Track2Rig itself and this package's code are Apache-2.0.
