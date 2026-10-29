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
  <a href="https://github.com/mashbox-labs/mashbox/blob/v0.1.3/AGENTS.md"><img alt="AGENTS.md" src="https://img.shields.io/badge/For%20coding%20agents-AGENTS.md-555555"></a>
  <a href="https://x.com/mashboxgg"><img alt="X" src="https://img.shields.io/badge/X-%40mashboxgg-000000?logo=x"></a>
</p>

### Read our guide: [run Qwen3.8-27B on TPU](https://mashbox.gg/docs/models/qwen3-8-27b)

Qwen's 27B hybrid-attention language model on one Google Cloud TPU v6e chip, with an OpenAI-compatible server. One install and one command, from a new Google account to a running model. Made by Qwen, ported to TPU by MASHBOX Labs.

<table>
<tr>
<td width="50%"><b>38.1 vs 31.4 tok/s</b><br>vs vLLM on the same chip, 1 request at a time<br><sub>TPU v6e-1</sub></td>
<td width="50%"><b>326.6 vs 1,606.7 s</b><br>first start to ready, weights on disk, empty compile caches, vs vLLM on the same chip type<br><sub>TPU v6e-1</sub></td>
</tr>
<tr>
<td width="50%"><b>258 tok/s total</b><br>8 users at once on one chip (37.45 for 1 user)<br><sub>TPU v6e-1</sub></td>
<td width="50%"><b>69.6 s</b><br>to ready with the shipped compile cache, weights on disk (326.6 s without)<br><sub>TPU v6e-1</sub></td>
</tr>
</table>

<sub>vLLM comparisons ran both sides on Google Cloud TPU v6e-1: ours with Qwen3.8-27B-BQ8_R.bshj (26.94 GB, 8.01 bits per weight), vLLM TPU 0.32.0 with Qwen's FP8 weights (Qwen/Qwen3.8-27B-FP8), since Qwen's bf16 weights do not fit one chip. Every measurement and its method: [Results](#results).</sub>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/top/01-qwen38-27b-vs-vllm-at-a-glance-dark.png"><img alt="Qwen3.8-27B vs vLLM-TPU on the same TPU v6e-1. Hardware: one TPU v6e-1 (Google Cloud, us-east5-b) vs the same TPU v6e-1." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/top/01-qwen38-27b-vs-vllm-at-a-glance-light.png" width="100%"></picture>

<img alt="Qwen3.8-27B running on Google Cloud TPU" width="720" src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/hero.gif">

<sub>Qwen3.8-27B on a new TPU v6e-1: install, weights download, first answer. Sped up 10x, pauses over 2 s cut to 2 s.</sub>

## Quickstart

On a Google Cloud TPU VM (the [setup guide](https://mashbox.gg/docs/google-cloud) gets you there from a new Google account; coding agents can follow [AGENTS.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/AGENTS.md)):

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
~/.local/bin/mashbox run qwen3-8-27b
```

### Works with Claude Code, Codex and Pi

The `mashbox` server speaks the Anthropic Messages, OpenAI Responses and OpenAI chat completions APIs, so coding agents can use Qwen3.8-27B as their LLM. Start it with `~/.local/bin/mashbox run qwen3-8-27b --ctx 32768 --slots 1` and point the agent at it: [Use with coding agents](https://mashbox.gg/docs/agents-on-tpu).

## Results

<details>
<summary>All 12 measurements, all on TPU v6e-1</summary>

| Measure | Ours | Compared with |
|---|---|---|
| Decode speed, 1 request<sup>1</sup> | **38.6 tok/s** | n/a<sup>2</sup> |
| Decode speed, 8 requests at once<sup>3</sup> | **269 tok/s total (33.6 each)** | n/a<sup>2</sup> |
| Start to ready, first run<sup>4</sup> | **326.6 s** | n/a |
| Start to ready, shipped compile cache | **69.6 s** | 326.6 s without it |
| Start to ready, shipped compile cache, 32k context, 1 slot | **32.6 s** | n/a |
| Server speed, 1 request at a time<sup>5</sup> | **38.1 tok/s** | vLLM: 31.4 tok/s<sup>6</sup> |
| Server speed, 1 request at a time, long prompts<sup>7</sup> | **37.0 tok/s** | vLLM: 31.0 tok/s<sup>6</sup> |
| Server speed, 1 request at a time, sampled<sup>8</sup> | **33.0 tok/s** | vLLM: 23.7 tok/s<sup>6</sup> |
| Start to ready, empty compile caches<sup>9</sup> | **326.6 s** | vLLM: 1,606.7 s<sup>10,11</sup> |
| Server speed, 8 users at once<sup>12</sup> | **258.0 tok/s total** | 1 user: 37.45 tok/s<sup>13</sup> |
| Decode speed, 1 request, speculative decoding<sup>14,15</sup> | **74.84 tok/s** | off: 37.77 tok/s<sup>16</sup> |
| Decode speed, 8 requests at once, speculative decoding<sup>17</sup> | **436.87 tok/s total** | off: 244.69 tok/s total<sup>16</sup> |

<sub><sup>1</sup> 64-token prompt, 128 new tokens, greedy.</sub><br>
<sub><sup>2</sup> Qwen's bf16 weights (55.6 GB) do not fit one v6e chip (32 GB).</sub><br>
<sub><sup>3</sup> Same prompt length and output for every request.</sub><br>
<sub><sup>4</sup> Weights file already on disk, no compile cache; every prefill and decode program compiles before the server says ready.</sub><br>
<sub><sup>5</sup> Short chat prompts, 256 new tokens, greedy.</sub><br>
<sub><sup>6</sup> vLLM TPU 0.32.0 with Qwen's FP8 weights (Qwen/Qwen3.8-27B-FP8) on the same TPU v6e-1.</sub><br>
<sub><sup>7</sup> 790-token prompts, 256 new tokens, greedy.</sub><br>
<sub><sup>8</sup> Temperature 0.7, top_p 0.8, top_k 20.</sub><br>
<sub><sup>9</sup> Weights already on disk.</sub><br>
<sub><sup>10</sup> vLLM TPU 0.32.0 with Qwen's FP8 weights (Qwen/Qwen3.8-27B-FP8) on a TPU v6e-1 (the same chip type).</sub><br>
<sub><sup>11</sup> vLLM starts this model only with 3 memory flags, and Qwen's bf16 weights do not fit.</sub><br>
<sub><sup>12</sup> Closed loop for 60 s through the OpenAI-compatible server, chat prompts of 29 to 33 tokens, 128 new tokens, greedy.</sub><br>
<sub><sup>13</sup> Same server and chip.</sub><br>
<sub><sup>14</sup> The model's own MTP head drafts 2 tokens per pass. 8 chat prompts (chat, code, math, summary), 256 new tokens, greedy, ctx 2048, median of 3 runs per prompt, then the median of the 8.</sub><br>
<sub><sup>15</sup> Greedy output identical to decoding without it on 5 of 8 prompts; the others split where the two best next tokens score equal or one bf16 step apart.</sub><br>
<sub><sup>16</sup> Same engine, chip and prompts, --mtp 0.</sub><br>
<sub><sup>17</sup> The same 8 prompts sent together, 256 new tokens each, greedy.</sub><br>

</details>

Each chart names the hardware on both sides.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/01-qwen38-27b-time-to-ready-dark.png"><img alt="Qwen3.8-27B: time to ready on one TPU v6e-1. Hardware: one TPU v6e-1 (Google Cloud, asia-northeast1-b, on demand) vs one TPU v6e-1 (Google Cloud, us-east5-b)." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/01-qwen38-27b-time-to-ready-light.png" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/02-qwen38-27b-one-request-speed-dark.png"><img alt="Qwen3.8-27B: one request at a time, same TPU v6e-1. Hardware: one TPU v6e-1 (Google Cloud, us-east5-b) vs the same TPU v6e-1." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/02-qwen38-27b-one-request-speed-light.png" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/03-qwen38-27b-fits-one-chip-dark.png"><img alt="Qwen3.8-27B on one TPU v6e-1: weights in chip memory. Hardware: one TPU v6e-1, 31.24 GiB HBM vs the same TPU v6e-1 (vLLM-TPU 0.32.0)." src="https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/resolve/main/assets/charts/03-qwen38-27b-fits-one-chip-light.png" width="100%"></picture>

## Files

| File | Format | Size (GB) | Bits per weight | Quality vs full precision | Speed |
|---|---|---|---|---|---|
| `Qwen3.8-27B-BQ8_R.bshj` | BSHJ | 26.94 | 8.01 | mean KL 0.00065, top-1 agreement 99.72% vs bf16 (8 prompts x 32 tokens) | 38.6 tokens/s for 1 request, 269 tokens/s for 8 (TPU v6e-1) |
| `Qwen3.8-27B-MTP-BQ8_R.bshj` | BSHJ | 0.43 | 8.01 | draft head for speculative decoding (every token is checked by the main weights); greedy output identical to decoding without it on 5 of 8 test prompts | 74.84 tokens/s for 1 request (37.77 without), 436.87 tokens/s for 8 (244.69 without) |
| `xla-cache/v6e-jax0.11.2-libtpu0.0.48.1.tar` | XLA compile cache (tar) | 0.89 | n/a | n/a (compiled programs, not weights) | first start on TPU v6e-1, weights on disk: ready in 69.6 s with it, 326.6 s without it |

## Chips

| Chip | Status | Note |
|---|---|---|
| TPU v5e | Supported | a v5e-4 (4 chips), split across them automatically; one v5e chip is too small: the weights alone are about 27 GB and a v5e chip has 16 GB |
| TPU v6e | Supported | one chip (v6e-1) |
| TPU v5p slice (v5p-8) | Not supported |  |
| TPU v6e slice (v6e-4) | Supported | 4 chips, split across them automatically |

## Kernels

Compiled TPU kernels for this model only. `mashbox run` picks the file that matches your chip and JAX version and checks its sha256 against `models.json`. Without a match, the model runs on plain JAX, slower.

| File | Chips | jax | jaxlib | libtpu |
|---|---|---|---|---|
| `kernels/mashbox_qwen3_8_27b_kernels-0.1.0-py3-none-any.whl` | TPU v6e | 0.11.2 | 0.11.2 | 0.0.48.* |

## Links

|  |  |
|---|---|
| Code | [mashbox-labs/mashbox](https://github.com/mashbox-labs/mashbox/tree/v0.1.3/src/mashbox/models/qwen3_8_27b) |
| Guide | [mashbox.gg/docs/models/qwen3-8-27b](https://mashbox.gg/docs/models/qwen3-8-27b) |
| Original model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen |
| Setup from a new Google account | [mashbox.gg/docs/google-cloud](https://mashbox.gg/docs/google-cloud) |
| Companies | hello@mashbox.gg |

## Licence

Weights: Apache-2.0 licence, from [Qwen/Qwen3.8-27B@1d4bf0f2](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0); code: Apache-2.0 (MASHBOX Labs). Qwen's LICENSE file is in this repo, unchanged: [LICENSE](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU/blob/main/LICENSE). The compiled kernels in `kernels/` are under their own licence: [LICENSE-KERNELS](https://github.com/mashbox-labs/mashbox/blob/v0.1.3/LICENSE-KERNELS). The compile cache in `xla-cache/` is under the same licence.
