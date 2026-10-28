---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
tags:
- tpu
- jax
- mashbox
---

<p align="center"><img alt="MASHBOX Labs" src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/banner-dark.png" width="100%"></p>

# Qwen3.8-27B on Google Cloud TPU

<p>
  <a href="https://github.com/mashbox-labs/mashbox"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-mashbox--labs%2Fmashbox-181717?logo=github"></a>
  <a href="https://mashbox.gg/docs/models/qwen3-8-27b"><img alt="Docs" src="https://img.shields.io/badge/Docs-mashbox.gg-ff5a1f"></a>
  <a href="https://github.com/mashbox-labs/mashbox/blob/v0.1.0/AGENTS.md"><img alt="AGENTS.md" src="https://img.shields.io/badge/For%20coding%20agents-AGENTS.md-555555"></a>
  <a href="https://x.com/mashboxgg"><img alt="X" src="https://img.shields.io/badge/X-%40mashboxgg-000000?logo=x"></a>
</p>

### Read our guide: [run Qwen3.8-27B on TPU](https://mashbox.gg/docs/models/qwen3-8-27b)

Qwen's 27B hybrid-attention language model on one Google Cloud TPU v6e chip, with an OpenAI-compatible server. One install and one command, from a new Google account to a running model. Made by Qwen, ported to TPU by MASHBOX Labs.

<table>
<tr>
<td width="50%"><b>38.1 vs 31.4 tok/s</b><br>vs vLLM on the same chip, 1 request at a time<br><sub>TPU v6e-1, 2026-10-28</sub></td>
<td width="50%"><b>130.1 vs 1,606.7 s</b><br>first start to ready, weights on disk, vs vLLM on the same chip<br><sub>TPU v6e-1, 2026-10-28</sub></td>
</tr>
<tr>
<td width="50%"><b>258 tok/s total</b><br>8 users at once on one chip (37.45 for 1 user)<br><sub>TPU v6e-1, 2026-10-28</sub></td>
<td width="50%"><b>9.8 s</b><br>to ready after the download, with the shipped compile cache (25.7 s without)<br><sub>TPU v6e-1, 2026-10-28</sub></td>
</tr>
</table>

<sub>vLLM comparisons ran both sides on the same Google Cloud TPU v6e-1: ours with Qwen3.8-27B-BQ8_R.bshj (26.94 GB, 8.01 bits per weight), vLLM TPU 0.32.0 with Qwen's FP8 weights (Qwen/Qwen3.8-27B-FP8), since Qwen's bf16 weights do not fit one chip. Measured 2026-10-28 (UTC). Every measurement and its method: [Results](#results).</sub>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/top/01-qwen38-27b-vs-vllm-at-a-glance-dark.png"><img alt="Qwen3.8-27B vs vLLM-TPU on the same TPU v6e-1. Measured 2026-10-28 03:55-05:47 UTC on one TPU v6e-1 (Google Cloud, us-east5-b) vs the same TPU v6e-1. MASHBOX measurement ledger CH-09." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/top/01-qwen38-27b-vs-vllm-at-a-glance-light.png" width="100%"></picture>

<img alt="Qwen3.8-27B running on Google Cloud TPU" width="720" src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/hero.gif">

<sub>A first-time user on a new TPU v6e-1 (28 October 2026): install, weights download, first answer. Sped up 10x, pauses over 2 s cut to 2 s.</sub>

## Quickstart

On a Google Cloud TPU VM (the [setup guide](https://mashbox.gg/docs/google-cloud) gets you there from a new Google account; coding agents can follow [AGENTS.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.0/AGENTS.md)):

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
~/.local/bin/mashbox run qwen3-8-27b
```

### Works with Claude Code, Codex and Pi

The `mashbox` server speaks the Anthropic Messages, OpenAI Responses and OpenAI chat completions APIs, so coding agents can use Qwen3.8-27B as their LLM. Start it with `~/.local/bin/mashbox run qwen3-8-27b --ctx 32768 --slots 1` and point the agent at it: [Use with coding agents](https://mashbox.gg/docs/agents-on-tpu).

## Results

<details>
<summary>All 10 measurements, all on TPU v6e-1, 2026-10-28</summary>

| Measure | Ours | Compared with |
|---|---|---|
| Decode speed, 1 request<sup>1</sup> | **38.6 tok/s** | n/a<sup>2</sup> |
| Decode speed, 8 requests at once<sup>3</sup> | **269 tok/s total (33.6 each)** | n/a<sup>2</sup> |
| Start to ready, first run<sup>4</sup> | **44.0 s** | n/a |
| Start to ready, later runs | **7.1 s** | n/a |
| Start to ready on a new chip, first run<sup>5</sup> | **138 s (9.8 s after the download)** | 166.6 s (25.7 s) without the compile cache |
| Server speed, 1 request at a time<sup>6</sup> | **38.1 tok/s** | vLLM: 31.4 tok/s<sup>7</sup> |
| Server speed, 1 request at a time, long prompts<sup>8</sup> | **37.0 tok/s** | vLLM: 31.0 tok/s<sup>7</sup> |
| Server speed, 1 request at a time, sampled<sup>9</sup> | **33.0 tok/s** | vLLM: 23.7 tok/s<sup>7</sup> |
| Start to ready, empty compile caches<sup>10</sup> | **130.1 s** | vLLM: 1,606.7 s<sup>7,11</sup> |
| Server speed, 8 users at once<sup>12</sup> | **258.0 tok/s total** | 1 user: 37.45 tok/s<sup>13</sup> |

<sub><sup>1</sup> 64-token prompt, 128 new tokens, greedy.</sub><br>
<sub><sup>2</sup> Qwen's bf16 weights (55.6 GB) do not fit one v6e chip (32 GB).</sub><br>
<sub><sup>3</sup> Same prompt length and output for every request.</sub><br>
<sub><sup>4</sup> Weights file already on disk; sha256 check and compiling included.</sub><br>
<sub><sup>5</sup> Includes the 26.94 GB download from Hugging Face and its sha256 check; the number in brackets counts from the end of the download and check. Same chip and day for both.</sub><br>
<sub><sup>6</sup> Short chat prompts, 256 new tokens, greedy.</sub><br>
<sub><sup>7</sup> vLLM TPU 0.32.0 with Qwen's FP8 weights (Qwen/Qwen3.8-27B-FP8) on the same TPU v6e-1.</sub><br>
<sub><sup>8</sup> 790-token prompts, 256 new tokens, greedy.</sub><br>
<sub><sup>9</sup> Temperature 0.7, top_p 0.8, top_k 20.</sub><br>
<sub><sup>10</sup> Weights already on disk.</sub><br>
<sub><sup>11</sup> vLLM starts this model only with 3 memory flags, and Qwen's bf16 weights do not fit.</sub><br>
<sub><sup>12</sup> Closed loop for 60 s through the OpenAI-compatible server, chat prompts of 29 to 33 tokens, 128 new tokens, greedy.</sub><br>
<sub><sup>13</sup> Same server and chip.</sub><br>

</details>

Each chart names its measurement date (UTC) and the hardware on both sides.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/01-qwen38-27b-time-to-ready-dark.png"><img alt="Qwen3.8-27B: time to ready on one TPU v6e-1. Measured 2026-10-28 03:53-05:39 UTC on one TPU v6e-1 (Google Cloud, us-east5-b) vs the same TPU v6e-1. MASHBOX measurement ledger CH-01." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/01-qwen38-27b-time-to-ready-light.png" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/02-qwen38-27b-one-request-speed-dark.png"><img alt="Qwen3.8-27B: one request at a time, same TPU v6e-1. Measured 2026-10-28 03:55-05:47 UTC on one TPU v6e-1 (Google Cloud, us-east5-b) vs the same TPU v6e-1. MASHBOX measurement ledger CH-02." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/02-qwen38-27b-one-request-speed-light.png" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/03-qwen38-27b-fits-one-chip-dark.png"><img alt="Qwen3.8-27B on one TPU v6e-1: weights in chip memory. Measured 2026-10-28 03:53-04:27 UTC on one TPU v6e-1, 31.24 GiB HBM vs the same TPU v6e-1 (vLLM-TPU 0.32.0). MASHBOX measurement ledger CH-03." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/03-qwen38-27b-fits-one-chip-light.png" width="100%"></picture>

## Files

| File | Format | Size (GB) | Bits per weight | Quality vs full precision | Speed | Measured |
|---|---|---|---|---|---|---|
| `Qwen3.8-27B-BQ8_R.bshj` | BSHJ | 26.94 | 8.01 | mean KL 0.00065, top-1 agreement 99.72% vs bf16 (8 prompts x 32 tokens, measured 2026-10-27) | 38.6 tokens/s for 1 request, 269 tokens/s for 8 (TPU v6e-1) | 2026-10-28 |
| `xla-cache/v6e-jax0.11.2-libtpu0.0.48.1.tar` | XLA compile cache (tar) | 0.05 | n/a | n/a (compiled programs, not weights) | first start on TPU v6e-1: ready 9.8 s after the weights download, 25.7 s without it | 2026-10-28 |

## Chips

| Chip | Status | Note |
|---|---|---|
| TPU v6e | passing, 2026-10-28 |  |
| TPU v5e | not supported | the weights alone are about 27 GB and a v5e chip has 16 GB |

## Kernels

Compiled TPU kernels for this model only. `mashbox run` picks the file that matches your chip and JAX version and checks its sha256 against `models.json`. Without a match, the model runs on plain JAX, slower.

| File | Chips | jax | jaxlib | libtpu |
|---|---|---|---|---|
| `kernels/mashbox_qwen3_8_27b_kernels-0.1.0-py3-none-any.whl` | TPU v6e | 0.11.2 | 0.11.2 | 0.0.48.* |

## Links

|  |  |
|---|---|
| Code | [mashbox-labs/mashbox](https://github.com/mashbox-labs/mashbox/tree/v0.1.0/src/mashbox/models/qwen3_8_27b) |
| Guide | [mashbox.gg/docs/models/qwen3-8-27b](https://mashbox.gg/docs/models/qwen3-8-27b) |
| Original model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen |
| Setup from a new Google account | [mashbox.gg/docs/google-cloud](https://mashbox.gg/docs/google-cloud) |
| Companies | hello@mashbox.gg |

## Licence

The weights keep the original licence: apache-2.0. MASHBOX code: Apache-2.0. The compiled kernels in `kernels/` are under their own licence: [LICENSE-KERNELS](https://github.com/mashbox-labs/mashbox/blob/v0.1.0/LICENSE-KERNELS).
