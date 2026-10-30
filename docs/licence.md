# Licence

Which licence covers which part of this repo and its downloads. The licence files themselves are the final word: [LICENSE](../LICENSE), [LICENSE-KERNELS](../LICENSE-KERNELS) and the upstream files in `licenses/`.

> **Our code is Apache-2.0.** The compiled kernels and the compile cache ship under the MASHBOX Kernel Licence: free for personal use, research, evaluation, and for any use by companies under US$1M annual revenue.

## At a glance

| Part | Licence |
|---|---|
| Our code: the `mashbox` package and CLI, model ports, demo apps, docs | Apache License 2.0 ([LICENSE](../LICENSE)). |
| Compiled kernel bundles (our Culet kernels, one bundle per release) and the precompiled compile cache | MASHBOX Kernel Licence 1.0 ([LICENSE-KERNELS](../LICENSE-KERNELS)): free for personal use, research and evaluation, and for any use by companies under US$1M annual revenue; larger companies in production write to hello@mashbox.gg. |
| Our own trained weights: the MASHBOX Fast Decoder for Hy-Nomas (`decoders/hy-nomas-fast-decoder-v1.safetensors` in [mashbox-labs/Hy-Nomas-TPU](https://huggingface.co/mashbox-labs/Hy-Nomas-TPU)) | Apache License 2.0. |
| Model weights, and our conversions of them (such as our BQ8_R file of Qwen3.8-27B) | The licence their makers publish, read at the revision we pin. It is shown on the model's Hugging Face card, and the upstream LICENSE file is kept unchanged in `licenses/` and in the model's Hugging Face repo. |

## Details

### Per model

| Model | Made by | Weights licence | From |
|---|---|---|---|
| Hy-Nomas | Tencent Hunyuan | Apache-2.0, [as stated by Tencent](https://huggingface.co/tencent/Hy-Nomas/blob/main/LICENSE), kept unchanged in [licenses/hy-nomas/](../licenses/hy-nomas/LICENSE) | tencent/Hy-Nomas@4b8b4ed8 |
| Qwen3.8-27B | Qwen | Apache-2.0, [as stated by Qwen](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE), kept unchanged in [licenses/qwen3-8-27b/](../licenses/qwen3-8-27b/LICENSE) | Qwen/Qwen3.8-27B@1d4bf0f2 |
| TRELLIS.2 | Microsoft | MIT, [as stated by Microsoft](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/LICENSE), kept unchanged in [licenses/trellis-2/](../licenses/trellis-2/LICENSE); microsoft/TRELLIS-image-large: MIT; facebook/dinov3-vitl16-pretrain-lvd1689m: [dinov3-license](https://ai.meta.com/resources/models-and-libraries/dinov3-license) (gated: you accept it and download it with your own Hugging Face login; we never re-host it). Code the port follows from other projects: [its credits](models/trellis-2.md#credits-and-licences) | [microsoft/TRELLIS.2-4B@af44b45f](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d), [microsoft/TRELLIS-image-large@25e0d31f](https://huggingface.co/microsoft/TRELLIS-image-large/tree/25e0d31ffbebe4b5a97464dd851910efc3002d96), [facebook/dinov3-vitl16-pretrain-lvd1689m@ea8dc286](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51) |

### Contributions

Contributions are accepted under the licence in [LICENSE](../LICENSE), with a DCO sign-off ([Contributing](../CONTRIBUTING.md)).

## Sources

- Upstream licences: each maker's Hugging Face repo at the revision pinned in `models.json`.
