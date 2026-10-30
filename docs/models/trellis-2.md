# TRELLIS.2 on TPU

<!-- catalog:start label -->
Experimental
<!-- catalog:end -->

Microsoft's open image-to-3D model, ported to TPU by MASHBOX Labs. You give it one picture of an object with the background cut out, and it writes a textured 3D mesh (a GLB file) you can open in Blender or a game engine.

## What it is

How it runs:

- **Port:** [src/mashbox/models/trellis_2](../../src/mashbox/models/trellis_2). The image encoder, the flow models (one each for sparse structure, shape and texture) and the decoders run on one TPU v6e chip as plain JAX; the GLB export is CPU code on the host. It needs no CUDA library, and you install nothing else.
- **GLB export:** remesh, simplify, UV unwrap and texture bake run on the VM's own CPU after the chip stages, in the same command. This is the slow part of a run.
- **Pipeline:** the 512 pipeline (Microsoft's default is the 1024 cascade, which this port does not run), with Microsoft's sampler and export settings: 12 sampling steps per flow model, and a GLB of up to 1,000,000 faces with a 4096 x 4096 texture.
- **Weights:** Microsoft's own, downloaded from [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) and [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) at pinned revisions and checked by sha256 before use. The image encoder is Meta's [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), which Meta gates: accept its licence ([dinov3-license](https://ai.meta.com/resources/models-and-libraries/dinov3-license)) on that page with your Hugging Face account first, then set `HF_TOKEN` on the VM or log in with the last line under [Install](#install) before the first run. We do not re-upload any of them.

## What you need

| Chip | Status |
|---|---|
| TPU v6e (one chip, v6e-1) | Supported: one chip (v6e-1) |
| TPU v5e | Not tested yet |
| TPU v5p slice (v5p-8) | Not tested yet |
| TPU v6e slice (v6e-4) | Not tested yet |

v6e-1 costs $2.70 an hour at Google's on-demand list price in us-east1 and us-east5, where the commands below create the VM. The chip bills from create to delete.

The input must be RGBA: the object opaque, the background transparent. A plain RGB photo is refused with a message before anything downloads.

Disk: the first run downloads 11.0 GB into `~/.cache/mashbox/trellis-2` (set `MASHBOX_CACHE` to use another disk). On a TPU v6e-1 VM, the whole-command VM maximum RSS (the peak host memory of the whole command, all stages together; no per-stage figure is recorded) was 72.9 to 89.1 GB per run, out of the VM's 185.7 GB, so run one picture at a time.

A run of an earlier build of this port, on another picture, was heavier: its whole command took 42 min 26.2 s and its whole-command VM maximum RSS was 122.9 GB, so plan for at least that much host memory.

## Install

```bash
gcloud compute tpus tpu-vm create trellis-2 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh trellis-2 --zone=us-east5-b
```

On the VM:

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu,trellis-2]>=0.1.10" --with-executables-from huggingface_hub --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
~/.local/bin/hf auth login
```

## Run

From your own machine, copy the picture to the VM:

```bash
gcloud compute tpus tpu-vm scp object.png trellis-2:~ --zone=us-east5-b
```

On the VM:

```bash
~/.local/bin/mashbox run trellis-2 --image object.png --out object.glb
```

It prints one JSON line when it is done: the output path and sha256, the mesh size, the time each stage took and the chip memory peak. `--seed` changes the noise seed (default 42), `--report run.json` writes the same summary to a file, and `~/.local/bin/mashbox download trellis-2` fetches and checks every file without running.

Back on your machine, copy the GLB off the VM, then delete the chip so it stops billing:

```bash
gcloud compute tpus tpu-vm scp trellis-2:~/object.glb . --zone=us-east5-b
gcloud compute tpus tpu-vm delete trellis-2 --zone=us-east5-b --quiet
```

## Performance

Measured on one TPU v6e chip (v6e-1) in Google Cloud zone asia-northeast1-b, with the GLB export on the same VM's CPU (44 vCPUs): one run per picture, one after another, with the weights already on disk (fetched first with `~/.local/bin/mashbox download trellis-2`), so no time here includes the download. The first run compiled on a new chip; the later runs read the programs it compiled. Every run still compiled more programs after it was ready (last column), so none of these is a warm-run time.

| Run | Whole command | Load to ready | Picture to mesh | GLB export | Programs compiled after ready |
|---|---|---|---|---|---|
| 1 | 23 min 22.6 s | 6 min 28.6 s | 55.2 s | 15 min 53.1 s | 55 |
| 2 | 15 min 32.4 s | 15.2 s | 46.7 s | 14 min 26.8 s | 17 |
| 3 | 30 min 11.7 s | 14.8 s | 1 min 37.5 s | 28 min 14.6 s | 19 |
| 4 | 16 min 32.3 s | 15.3 s | 50.0 s | 15 min 22.8 s | 22 |

- Chip memory peak: 14.4 GB.
- Cost per picture, the whole command at the on-demand list price above: $0.70 to $1.36.
- The GLB export on the VM's CPU is most of each run. The chip waits while it runs.

## Credits and licences

TRELLIS.2 is made by Microsoft. The weights are downloaded unchanged from their makers' repos: [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) (MIT), [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) (MIT), [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) ([dinov3-license](https://ai.meta.com/resources/models-and-libraries/dinov3-license)). The port follows Microsoft's code, [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd) (MIT); its LICENSE is kept unchanged in [licenses/trellis-2/LICENSE](../../licenses/trellis-2/LICENSE).

The port also draws on these projects:

- [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh/tree/12289e1062f0603f2f0d0771b02e1395d247f26f) (MIT License): the GLB export's mesh simplification is a port of its simplify path; its remesh and mesh clean-up follow its documented behaviour, and the UV unwrap keeps the two steps of its uv_unwrap (chart clustering, then xatlas); its LICENSE is kept unchanged as [LICENSE-CuMesh](../../src/mashbox/models/trellis_2/glb_bake/LICENSE-CuMesh) beside that code.
- [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM/tree/6dd94a859c26ee8246888502eada3dd8ad85532e) (MIT License): the sparse 3D convolution and the GLB export's texture sampling follow its op semantics and explicit-GEMM math; its LICENSE is kept unchanged as [LICENSE-FlexGEMM](../../src/mashbox/models/trellis_2/LICENSE-FlexGEMM) beside that code.
- [uploadcare/pillow-simd](https://github.com/uploadcare/pillow-simd/tree/aa38aa687d4bcb79fa6c7712610ac3ffc1193234) (MIT-CMU License): stage 0's image resize and alpha handling reproduce the arithmetic of its 12.1.1.post0 release; its LICENSE is kept unchanged as [LICENSE-Pillow-SIMD](../../src/mashbox/models/trellis_2/LICENSE-Pillow-SIMD) beside that code.
- [huggingface/transformers](https://github.com/huggingface/transformers/tree/e7d351cebad5f6dcdd169b0c034fdee0a000e6a9) (Apache License, Version 2.0): the image encoder's layer math was checked against its DINOv3ViTModel definition; read, not copied.

Our code in this repo: Apache-2.0 (see [LICENSE](../../LICENSE)).
