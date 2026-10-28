# Licence

Which licence covers which part of this repo and its downloads. The licence files themselves are the final word: [LICENSE](../LICENSE), [LICENSE-KERNELS](../LICENSE-KERNELS) and the upstream files in `licenses/`.

> **Our code is Apache-2.0.** The compiled kernels and the compile cache ship under the MASHBOX Kernel Licence: free for personal use, research, evaluation, and for any use by companies under US$1M annual revenue.

## At a glance

| Part | Licence |
|---|---|
| Our code: the `mashbox` package and CLI, model ports, demo apps, docs | Apache License 2.0 ([LICENSE](../LICENSE)). |
| Compiled kernel bundles (our Culet kernels, one bundle per release) and the precompiled compile cache | MASHBOX Kernel Licence 1.0 ([LICENSE-KERNELS](../LICENSE-KERNELS)): free for personal use, research and evaluation, and for any use by companies under US$1M annual revenue; larger companies in production write to hello@mashbox.gg. |
| Model weights | Each model's own licence, as its makers state it. It is shown on the model's Hugging Face card, and upstream licence files are kept unchanged in `licenses/`. |

## Details

### Per model

| Model | Made by | Licence |
|---|---|---|
| Hy-Nomas | Tencent Hunyuan | [as stated by Tencent](https://huggingface.co/tencent/Hy-Nomas/blob/main/LICENSE), kept unchanged in [licenses/hy-nomas/](../licenses/hy-nomas/LICENSE) |
| Qwen3.8-27B | Qwen | [as stated by Qwen](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE), kept unchanged in [licenses/qwen3-8-27b/](../licenses/qwen3-8-27b/LICENSE) |

### Contributions

Contributions are accepted under the licence in [LICENSE](../LICENSE), with a DCO sign-off ([Contributing](../CONTRIBUTING.md)).

## Sources

- Upstream licences: each maker's Hugging Face repo at the revision pinned in `models.json`.

## Changelog

| Date | What changed |
|---|---|
| 28 October 2026 | First version. |
