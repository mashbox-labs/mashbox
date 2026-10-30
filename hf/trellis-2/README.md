---
license: mit
pipeline_tag: image-to-3d
tags:
- tpu
- jax
- mashbox
---

<p align="center"><img alt="MASHBOX Labs" src="https://huggingface.co/mashbox-labs/TRELLIS.2-TPU/resolve/main/assets/banner-dark.png" width="100%"></p>

# TRELLIS.2 on Google Cloud TPU

**Experimental**

Runs with `mashbox run trellis-2` in the `mashbox` package, [release 0.1.10](https://pypi.org/project/mashbox/0.1.10/).

<p>
  <a href="https://github.com/mashbox-labs/mashbox"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-mashbox--labs%2Fmashbox-181717?logo=github"></a>
  <a href="https://mashbox.gg/docs/models/trellis-2"><img alt="Docs" src="https://img.shields.io/badge/Docs-mashbox.gg-ff5a1f"></a>
  <a href="https://github.com/mashbox-labs/mashbox/blob/v0.1.10/AGENTS.md"><img alt="AGENTS.md" src="https://img.shields.io/badge/For%20coding%20agents-AGENTS.md-555555"></a>
  <a href="https://x.com/mashboxgg"><img alt="X" src="https://img.shields.io/badge/X-%40mashboxgg-000000?logo=x"></a>
</p>

### Read our guide: [run TRELLIS.2 on TPU](https://mashbox.gg/docs/models/trellis-2)

Microsoft's open image-to-3D model: one RGBA picture in, a textured GLB mesh out, on a TPU v6e-1 VM (the model on the chip, the GLB export on the VM's CPU). Made by Microsoft, ported to TPU by MASHBOX Labs.

- **15 min 32.4 s to 30 min 11.7 s**: one picture in, a textured GLB out: the whole command (sha256 check, load, every stage, GLB export; the weights were already on disk, so the download is not in it), one run per picture, on TPU v6e-1
- **46.7 s to 1 min 37.5 s**: picture in to mesh out: the model's stages on the chip, with the host work between them, on TPU v6e-1
- **14 min 26.8 s to 28 min 14.6 s**: GLB export (remesh, simplify, UV unwrap, texture bake) on the VM's own CPU, on TPU v6e-1
- **6 min 28.6 s**: first run on a new chip, load to ready: loading the weights and compiling, timed from the start of the load (weights on disk; the seconds before the load starts are not in it), on TPU v6e-1
- **17 to 55**: programs compiled after ready, per run: new compile-cache entries from the chip stages and from the GLB export's JAX work on the VM's CPU together (so no run here is a warm run), on TPU v6e-1
- **14.4 GB**: chip memory peak, on TPU v6e-1
- **72.9 to 89.1 GB**: whole-command VM maximum RSS: one value per run for the whole command, all stages together (no per-stage memory is recorded), on TPU v6e-1
- **$0.70 to $1.36**: cost per picture: the whole command (weights already on disk) at the Google Cloud on-demand list price of a v6e-1 in us-east1 and us-east5, on TPU v6e-1

## Quickstart

On a Google Cloud TPU VM (the [setup guide](https://mashbox.gg/docs/google-cloud) gets you there from a new Google account; coding agents can follow [AGENTS.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.10/AGENTS.md)):

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu,trellis-2]>=0.1.10" --with-executables-from huggingface_hub --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
~/.local/bin/hf auth login
~/.local/bin/mashbox run trellis-2 --image object.png --out object.glb
```

Before the first run, open [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) on Hugging Face and accept its licence ([dinov3-license](https://ai.meta.com/resources/models-and-libraries/dinov3-license)) with your account. `mashbox run` downloads it with your own token: the `hf auth login` line above signs the VM in (or set `HF_TOKEN`), and this repo does not host it.

Host memory: the whole command's VM maximum RSS was 72.9 to 89.1 GB per run here, out of the VM's 185.7 GB. A run of an earlier build of this port, on another picture, took 42 min 26.2 s and peaked at 122.9 GB, so plan for at least that much host memory and run one picture at a time.

## Results

| Chip | Measure | Result | Baseline |
|---|---|---|---|
| TPU v6e-1 | One picture in, a textured GLB out: the whole command (sha256 check, load, every stage, GLB export; the weights were already on disk, so the download is not in it), one run per picture | 15 min 32.4 s to 30 min 11.7 s | n/a |
| TPU v6e-1 | Picture in to mesh out: the model's stages on the chip, with the host work between them | 46.7 s to 1 min 37.5 s | n/a |
| TPU v6e-1 | GLB export (remesh, simplify, UV unwrap, texture bake) on the VM's own CPU | 14 min 26.8 s to 28 min 14.6 s | n/a |
| TPU v6e-1 | First run on a new chip, load to ready: loading the weights and compiling, timed from the start of the load (weights on disk; the seconds before the load starts are not in it) | 6 min 28.6 s | n/a |
| TPU v6e-1 | Programs compiled after ready, per run: new compile-cache entries from the chip stages and from the GLB export's JAX work on the VM's CPU together (so no run here is a warm run) | 17 to 55 | n/a |
| TPU v6e-1 | Chip memory peak | 14.4 GB | n/a |
| TPU v6e-1 | Whole-command VM maximum RSS: one value per run for the whole command, all stages together (no per-stage memory is recorded) | 72.9 to 89.1 GB | n/a |
| TPU v6e-1 | Cost per picture: the whole command (weights already on disk) at the Google Cloud on-demand list price of a v6e-1 in us-east1 and us-east5 | $0.70 to $1.36 | n/a |

## Files

This port loads the original weights unchanged from their makers' repos: [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B), [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large), [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). This repo holds none of them.

## Chips

| Chip | Status | Note |
|---|---|---|
| TPU v5e | Not tested yet |  |
| TPU v6e | Supported | one chip (v6e-1) |
| TPU v5p slice (v5p-8) | Not tested yet |  |
| TPU v6e slice (v6e-4) | Not tested yet |  |

## Kernels

This model runs on plain JAX. It ships no compiled kernels.

## Links

|  |  |
|---|---|
| Code | [mashbox-labs/mashbox](https://github.com/mashbox-labs/mashbox/tree/v0.1.10/src/mashbox/models/trellis_2) |
| Guide | [mashbox.gg/docs/models/trellis-2](https://mashbox.gg/docs/models/trellis-2) |
| Original model | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) by Microsoft |
| Setup from a new Google account | [mashbox.gg/docs/google-cloud](https://mashbox.gg/docs/google-cloud) |
| Companies | hello@mashbox.gg |

## Licence

Weights: MIT licence, from [microsoft/TRELLIS.2-4B@af44b45f](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d); [microsoft/TRELLIS-image-large@25e0d31f](https://huggingface.co/microsoft/TRELLIS-image-large/tree/25e0d31ffbebe4b5a97464dd851910efc3002d96): MIT; [facebook/dinov3-vitl16-pretrain-lvd1689m@ea8dc286](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51): [dinov3-license](https://ai.meta.com/resources/models-and-libraries/dinov3-license); code: Apache-2.0 (MASHBOX Labs). [LICENSE](https://huggingface.co/mashbox-labs/TRELLIS.2-TPU/blob/main/LICENSE) in this repo is Microsoft's licence file from [microsoft/TRELLIS.2@75fbf018](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/LICENSE) on GitHub, unchanged. Credit: [JeffreyXiang/CuMesh@12289e10](https://github.com/JeffreyXiang/CuMesh/tree/12289e1062f0603f2f0d0771b02e1395d247f26f), MIT License, code (the GLB export's mesh simplification is a port of its simplify path; its remesh and mesh clean-up follow its documented behaviour, and the UV unwrap keeps the two steps of its uv_unwrap: chart clustering, then xatlas); its LICENSE ships unchanged in the package as `LICENSE-CuMesh` beside that code. Credit: [JeffreyXiang/FlexGEMM@6dd94a85](https://github.com/JeffreyXiang/FlexGEMM/tree/6dd94a859c26ee8246888502eada3dd8ad85532e), MIT License, code (the sparse 3D convolution and the GLB export's texture sampling follow its op semantics and explicit-GEMM math); its LICENSE ships unchanged in the package as `LICENSE-FlexGEMM` beside that code. Credit: [uploadcare/pillow-simd@aa38aa68](https://github.com/uploadcare/pillow-simd/tree/aa38aa687d4bcb79fa6c7712610ac3ffc1193234), MIT-CMU License, code (stage 0's image resize and alpha handling reproduce the arithmetic of its 12.1.1.post0 release); its LICENSE ships unchanged in the package as `LICENSE-Pillow-SIMD` beside that code. Credit: [huggingface/transformers@e7d351ce](https://github.com/huggingface/transformers/tree/e7d351cebad5f6dcdd169b0c034fdee0a000e6a9), Apache License, Version 2.0, reference (the image encoder's layer math was checked against its DINOv3ViTModel definition; read, not copied).
