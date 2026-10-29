<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/docs/assets/banner-dark.png">
    <img alt="MASHBOX Labs" src="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/docs/assets/banner-light.png" width="100%">
  </picture>
</p>

<h3 align="center">MASHBOX Labs makes open models run faster and cheaper on TPUs and accelerated hardware, as well as exploring the future of the intersection of AI and gaming.</h3>

<p align="center">
  <a href="#quickstart"><b>Quickstart</b></a> •
  <a href="#models"><b>Models</b></a> •
  <a href="https://mashbox.gg/docs"><b>Docs</b></a> •
  <a href="https://huggingface.co/mashbox-labs"><b>Hugging Face</b></a> •
  <a href="#coding-agents"><b>Coding agents</b></a>
</p>

<p align="center">
  <a href="https://github.com/mashbox-labs/mashbox/blob/v0.1.3/LICENSE"><img alt="Licence: Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue"></a>
  <a href="#tpu-v6e-and-v5e"><img alt="TPU v6e" src="https://img.shields.io/badge/TPU-v6e-ff5a1f"></a>
  <a href="https://docs.astral.sh/uv/"><img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white"></a>
  <a href="https://huggingface.co/mashbox-labs"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-mashbox--labs-ffcc4d"></a>
  <a href="https://x.com/mashboxgg"><img alt="X" src="https://img.shields.io/badge/X-%40mashboxgg-000000?logo=x"></a>
</p>

<p align="center"><a href="https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/models/qwen3-8-27b.md"><img alt="Qwen3.8-27B on one TPU v6e chip" src="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/hf/qwen3-8-27b/assets/hero.gif" width="100%"></a></p>
<p align="center"><b>Qwen3.8-27B</b> on one TPU v6e chip: first-time install to first answer (sped up 10x).</p>

## Quickstart

From your own machine, with the Google Cloud CLI set up ([docs/google-cloud.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/google-cloud.md) does that from a new Google account). This runs Qwen3.8-27B on one TPU v6e chip:

```bash
gcloud compute tpus tpu-vm create mashbox-1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]" && ~/.local/bin/mashbox run qwen3-8-27b --prompt "Write a haiku about the sea."'
gcloud compute tpus tpu-vm delete mashbox-1 --zone=us-east5-b --quiet
```

The chip's system Python is 3.10 and JAX for TPU needs 3.12, so the second line installs [uv](https://docs.astral.sh/uv/), which brings its own Python 3.12 and puts the `mashbox` command in `~/.local/bin`. The first run downloads the 26.9 GB weights file and the compile cache; with them on disk, the server on a TPU v6e-1 is ready in 69.6 s. If the create command says the zone has no capacity, use another zone from [step 6 of the setup guide](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/google-cloud.md#6-create-one-chip) in all three lines.

The chip bills from create to delete: one TPU v6e chip is $2.70 an hour at Google's on-demand [list price](https://cloud.google.com/tpu/pricing). To serve Qwen3.8-27B with an OpenAI-compatible API instead, see [docs/models/qwen3-8-27b.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/models/qwen3-8-27b.md).

## Models

<!-- models:start -->
| Model | What it does | Chips | Measured | Weights |
|---|---|---|---|---|
| **[Hy-Nomas](https://github.com/mashbox-labs/mashbox/tree/v0.1.3/src/mashbox/models/hy_nomas)**<br>by Tencent Hunyuan | World model: text and movement keys in, video out | TPU v5e: Not tested yet<br>TPU v6e (v6e-1): One chip: the full pipeline fits on a single TPU v6e chip (32 GB), 2.46 s of compute per second of video. This version of `mashbox run` starts on a v6e-4<br>TPU v5p slice (v5p-8): Not tested yet<br>TPU v6e slice (v6e-4): Supported | Compute per second of video, quality preset, Tencent decoder, 1.00 = real time: **1.045 s** (TPU v6e-4)<br>Compute per second of video, quality preset, MASHBOX Fast decoder: **0.689 s** (TPU v6e-4) | [Hy-Nomas-TPU](https://huggingface.co/mashbox-labs/Hy-Nomas-TPU) |
| **[Qwen3.8-27B](https://github.com/mashbox-labs/mashbox/tree/v0.1.3/src/mashbox/models/qwen3_8_27b)**<br>by Qwen | LLM: text in, text out | TPU v5e (v5e-4): Supported<br>TPU v6e (v6e-1): Supported<br>TPU v5p slice (v5p-8): Not tested yet<br>TPU v6e slice (v6e-4): Supported | **38.1 vs 31.4 tok/s** vs vLLM on the same chip, 1 request at a time; **326.6 vs 1,606.7 s** first start to ready, weights on disk, empty compile caches, vs vLLM on the same chip type (TPU v6e-1). [All results](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU) | [Qwen3.8-27B-TPU](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU) |
<!-- models:end -->

Every number names the chip it was measured on; each model's guide has all its results. Options for any model: `~/.local/bin/mashbox run <model> --help`.

## Results

<!-- charts:start -->
Each chart names the hardware on both sides.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/docs/media/charts/01-qwen38-27b-time-to-ready-dark.png"><img alt="Qwen3.8-27B: time to ready on one TPU v6e-1. Hardware: one TPU v6e-1 (Google Cloud, asia-northeast1-b, on demand) vs one TPU v6e-1 (Google Cloud, us-east5-b)." src="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/docs/media/charts/01-qwen38-27b-time-to-ready-light.png" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/docs/media/charts/03-qwen3-30b-a3b-quality-vs-size-dark.png"><img alt="Qwen3-30B-A3B: quality vs file size, 20 files. Hardware: a frozen llama.cpp CPU scorer (16 threads, AMD EPYC 7502P / 7B13) vs the same scorer and the same 10,240 positions for every rival file." src="https://raw.githubusercontent.com/mashbox-labs/mashbox/v0.1.3/docs/media/charts/03-qwen3-30b-a3b-quality-vs-size-light.png" width="100%"></picture>

<sub>Qwen3-30B-A3B is not in this package: this chart compares quantized files of it, each scored the same way.</sub>
<!-- charts:end -->

## What you get

- **One command per model.** `mashbox run <model>` downloads the weights, checks every file's sha256, loads the model and serves it.
- **Fits on fewer chips.** Qwen3.8-27B runs on a single TPU v6e chip (32 GB) from a 26.94 GB weights file at 8.01 bits per weight, with 99.72% top-1 agreement against Qwen's own bf16 weights.
- **Fast starts.** Models ship their compiled TPU kernels, and Qwen3.8-27B also ships a precompiled compile cache for TPU v6e, so a start skips compiling: Qwen3.8-27B on TPU v6e-1 is ready in 69.6 s with it, 326.6 s without it.
- **An OpenAI-compatible server** for the language models, with batching: 8 requests at once share the chip.
- **Works with Claude Code, Codex and Pi.** The server also speaks the Anthropic Messages and OpenAI Responses APIs, so coding agents can use a model you serve as their LLM: [Use with coding agents](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/agents-on-tpu.md).
- **Docs for coding agents.** [AGENTS.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/AGENTS.md) gives an agent the rules and the steps to set up a chip, run a model and delete the chip; [llms.txt](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/llms.txt) lists the docs.
- **Named files.** Every weights file on Hugging Face is named by its type (for example `Qwen3.8-27B-BQ8_R.bshj`), with size, bits per weight and measured quality next to it.

## TPU v6e and v5e

| | TPU v6e | TPU v5e |
|---|---|---|
| Memory per chip | 32 GB | 16 GB |
| Memory bandwidth per chip | 1,638 GBps | 800 GiBps |
| Peak bf16 per chip | 918 TFLOPs | 197 TFLOPs |
| Peak int8 per chip | 1,836 TOPs | 393 TOPs |
| Create with | `--accelerator-type=v6e-1 --version=v2-alpha-tpuv6e` | `--accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite` |
| Zones to try first | us-east5-b, europe-west4-a | us-central1-a |

v6e is the default. A model runs on v5e when its weights and working memory fit in 16 GB; the models table says which chips each model runs on. Specs from Google's [v6e](https://cloud.google.com/tpu/docs/v6e) and [v5e](https://cloud.google.com/tpu/docs/v5e) pages. More in [docs/chips.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/chips.md).

TPU v6e (one chip and a v6e-4) against one NVIDIA H100 at Google Cloud list prices, with the memory bandwidth of each and the cost per 1M output tokens of Qwen3.8-27B (27B parameters, dense): [TPU or GPU: price and bandwidth](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/tpu-vs-gpu.md).

## Open code, compiled kernels

The source of the model ports, CLI, runtime, demo apps and docs is in this repo, under Apache-2.0. Each model ships with its own compiled TPU kernels (our Culet kernels) and, where we built one, a precompiled compile cache, only what that model needs: `mashbox run` downloads them from the model's Hugging Face repo and checks them against the sha256 in `models.json`. Models that list plain JAX as tested also run without them, slower. The compiled files are under the [MASHBOX Kernel Licence](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/LICENSE-KERNELS): free for personal use, research and evaluation, and for any use by companies under US$1M annual revenue. Larger companies in production: hello@mashbox.gg.

## Coding agents

**Use a model as your agent's LLM.** Claude Code, Codex and Pi work with every LLM that `mashbox run` serves: start the server, open the tunnel, set the agent's base URL. Settings for each agent and measured results: [Use with coding agents](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/agents-on-tpu.md).

**Let an agent set it up.** [AGENTS.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/AGENTS.md) gives an agent the rules and the steps, from a Google account to a run and a deleted chip, and tells it to ask you before anything that costs money. [llms.txt](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/llms.txt) lists the docs.

## Community and links

| | |
|---|---|
| Docs | [mashbox.gg/docs](https://mashbox.gg/docs) |
| Models and weights | [huggingface.co/mashbox-labs](https://huggingface.co/mashbox-labs) |
| News | [x.com/mashboxgg](https://x.com/mashboxgg) |
| Bugs and model requests | [GitHub issues](https://github.com/mashbox-labs/mashbox/issues), see [CONTRIBUTING.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/CONTRIBUTING.md) |
| Security reports | [SECURITY.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/SECURITY.md) |
| Companies | hello@mashbox.gg |

## Licence

Our code is under the [Apache License 2.0](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/LICENSE). The compiled kernels and compile cache are under the [MASHBOX Kernel Licence](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/LICENSE-KERNELS). Model weights keep each model's own licence, shown on its Hugging Face card and kept unchanged in [licenses/](https://github.com/mashbox-labs/mashbox/tree/v0.1.3/licenses). Details: [docs/licence.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/licence.md).

## Citation

```bibtex
@software{mashbox,
  author = {{MASHBOX Labs}},
  title = {mashbox: open models that run faster and cheaper},
  url = {https://github.com/mashbox-labs/mashbox},
  year = {2026}
}
```

## Thank you

- [Tencent Hunyuan](https://huggingface.co/tencent/Hy-Nomas) for Hy-Nomas and [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) for Qwen3.8-27B, and for releasing their weights.
- The [JAX](https://github.com/jax-ml/jax) and Pallas teams, and Google Cloud TPU.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and the ggml authors for GGUF; mashbox reads GGUF files with their `gguf` package ([GGUF files](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/docs/gguf.md)).
- [Hugging Face](https://huggingface.co) for hosting the weights and [uv](https://github.com/astral-sh/uv) for a one-line Python 3.12.
