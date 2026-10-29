---
${front_matter}
---

<p align="center"><img alt="MASHBOX Labs" src="${banner}" width="100%"></p>

# ${name}

**${maturity_label}**

### Read the guide: [${name} docs](${docs_url})

${summary} Trained from scratch by MASHBOX Labs. Code and weights: Apache-2.0.

${hero}

## Quickstart

On a CPU, with Python 3.12 (tested on x86-64 Linux). If your `python3` is older, as on a Google Cloud TPU VM, use the uv steps in the [install guide](${docs_url}#install). The first run downloads `track2rig-v2.safetensors` (11 MB) from this repo at the revision pinned in `models.json` and checks its sha256.

```bash
git clone https://github.com/mashbox-labs/mashbox && cd mashbox
python3 -m venv .venv && . .venv/bin/activate
pip install ".[cpu,track2rig]"
${run}
```

This writes `grison_rig/rig.json` and `grison_rig/rig.glb` (glTF 2.0: one joint per bone, top-4 skin, one animation).

## From video

Track2Rig starts from 3D point tracks. To make them from a video of one animal we use [TrackEverything](https://github.com/ayushjain1144/trackeverything) (CMU and Meta) at commit `f0c18dd9`, licensed **CC BY-NC 4.0 (non-commercial use only)**, through our wrapper `track2rig-video`. It needs Linux and an NVIDIA GPU. Steps and measured times: [video to tracks to rig](${docs_url}#from-video). Tracks from any other 3D point tracker work through `track2rig-video align`.

## Results

${results_table}

## Files

${files_table}

${weights_version}

## Links

${links_table}

## Licence

Code and weights: ${our_license} (MASHBOX Labs). The video step's tracker (TrackEverything) and its geometry model (Pi3) keep their own licences and are not in this repo.
