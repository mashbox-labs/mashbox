---
license: apache-2.0
tags:
- jax
- mashbox
- rigging
- animation
- gltf
---

<p align="center"><img alt="MASHBOX Labs" src="https://huggingface.co/mashbox-labs/Track2Rig/resolve/main/assets/banner-dark.png" width="100%"></p>

# Track2Rig

**Experimental**

### Read the guide: [Track2Rig docs](https://mashbox.gg/docs/models/track2rig)

Our own model: dense 3D point tracks of a moving creature in, a rig out (bones, skin weights, per-frame bone transforms and a skeleton). Runs on a CPU. It does not work well on many clips yet. Newer weights will be published as dated versions. Trained from scratch by MASHBOX Labs. Code and weights: Apache-2.0.

<img alt="Track2Rig preview" width="720" src="https://huggingface.co/mashbox-labs/Track2Rig/resolve/main/assets/hero.gif">

<sub>The quickstart clip (grison_walk.npz, a grison walking, from the synthetic test set) rigged by Track2Rig 0.1.0 (weights track2rig-v2) on a CPU, in a side view and a three-quarter view. Every point is moved only by the predicted skin and bone transforms; lines are the predicted skeleton. Also as an [MP4 video](https://huggingface.co/mashbox-labs/Track2Rig/resolve/main/assets/hero.mp4) (the clip looped).</sub>

## Quickstart

On a CPU, with Python 3.12 (tested on x86-64 Linux). If your `python3` is older, as on a Google Cloud TPU VM, use the uv steps in the [install guide](https://mashbox.gg/docs/models/track2rig#install). The first run downloads `track2rig-v2.safetensors` (11 MB) from this repo at the revision pinned in `models.json` and checks its sha256.

```bash
git clone https://github.com/mashbox-labs/mashbox && cd mashbox
python3 -m venv .venv && . .venv/bin/activate
pip install ".[cpu,track2rig]"
mashbox run track2rig examples/track2rig/tracks/grison_walk.npz -o grison_rig
```

This writes `grison_rig/rig.json` and `grison_rig/rig.glb` (glTF 2.0: one joint per bone, top-4 skin, one animation).

## From video

Track2Rig starts from 3D point tracks. To make them from a video of one animal we use [TrackEverything](https://github.com/ayushjain1144/trackeverything) (CMU and Meta) at commit `f0c18dd9`, licensed **CC BY-NC 4.0 (non-commercial use only)**, through our wrapper `track2rig-video`. It needs Linux and an NVIDIA GPU. Steps and measured times: [video to tracks to rig](https://mashbox.gg/docs/models/track2rig#from-video). Tracks from any other 3D point tracker work through `track2rig-video align`.

## Results

| Chip | Measure | Result | Baseline |
|---|---|---|---|
| CPU (AMD EPYC 7502P) | Held-out reconstruction error, synthetic test set (32 clips, noisy and partly hidden tracks), % of body size, lower is better | 0.784 | SSDR, our 30-iteration NumPy/SciPy implementation, same bone count: 0.915 |
| CPU (AMD EPYC 7502P) | Joint error, synthetic test set, % of body size, lower is better | 6.24 | SSDR, same implementation and inputs: 7.73 |

## Files

| File | Format | Size | Note |
|---|---|---|---|
| `track2rig-v2.safetensors` | safetensors | 11.1 MB | float32, 55 tensors, 2,764,416 parameters |
| `examples/clips/glass-snake_idle.npz` | npz | 2.9 MB | example |
| `examples/clips/glass-snake_walk.npz` | npz | 2.9 MB | example |
| `examples/clips/grison_idle.npz` | npz | 2.2 MB | example |
| `examples/clips/grison_walk.npz` | npz | 2.1 MB | example |
| `examples/clips/pelican-spider_idle.npz` | npz | 2.5 MB | example |
| `examples/clips/pelican-spider_walk.npz` | npz | 2.7 MB | example |
| `examples/clips/reference.json` | json | 4.2 KB | example |

Weights version `2026-10-29` (the Hugging Face tag of the revision `models.json` pins). Newer weights are published as new dated versions; a published version never changes.

## Links

|  |  |
|---|---|
| Code | [mashbox-labs/mashbox](https://github.com/mashbox-labs/mashbox/tree/v0.1.4/src/mashbox/models/track2rig) |
| Guide | [mashbox.gg/docs/models/track2rig](https://mashbox.gg/docs/models/track2rig) |
| Original model | [mashbox-labs/Track2Rig](https://huggingface.co/mashbox-labs/Track2Rig) by MASHBOX Labs |
| Setup from a new Google account | [mashbox.gg/docs/google-cloud](https://mashbox.gg/docs/google-cloud) |
| Companies | hello@mashbox.gg |

## Licence

Code and weights: Apache-2.0 (MASHBOX Labs). The video step's tracker (TrackEverything) and its geometry model (Pi3) keep their own licences and are not in this repo.
