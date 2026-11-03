---
license: apache-2.0
pipeline_tag: text-to-speech
tags:
- tpu
- jax
- mashbox
---

<p align="center"><img alt="MASHBOX Labs" src="https://huggingface.co/mashbox-labs/maya1-TPU/resolve/main/assets/banner-dark.png" width="100%"></p>

# maya1 on Google Cloud TPU

**Experimental**

Runs with `mashbox run maya1` in the `mashbox` package, [release 0.1.15](https://pypi.org/project/mashbox/0.1.15/).

<p>
  <a href="https://github.com/mashbox-labs/mashbox"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-mashbox--labs%2Fmashbox-181717?logo=github"></a>
  <a href="https://mashbox.gg/docs/models/maya1"><img alt="Docs" src="https://img.shields.io/badge/Docs-mashbox.gg-ff5a1f"></a>
  <a href="https://github.com/mashbox-labs/mashbox/blob/v0.1.15/AGENTS.md"><img alt="AGENTS.md" src="https://img.shields.io/badge/For%20coding%20agents-AGENTS.md-555555"></a>
  <a href="https://x.com/mashboxgg"><img alt="X" src="https://img.shields.io/badge/X-%40mashboxgg-000000?logo=x"></a>
</p>

### Read our guide: [run maya1 on TPU](https://mashbox.gg/docs/models/maya1)

Maya Research's open text-to-speech model: a voice description and the text to say in, 24 kHz speech out, on one TPU chip, with the SNAC 24 kHz decoder it speaks through. Emotion tags such as <laugh> and <whisper> go inline in the text. Made by Maya Research, ported to TPU by MASHBOX Labs.

- **0.266 to 0.269 s**: first audio, one request, decoded while it is generated (programs compiled), on TPU v6e-1
- **0.536 to 0.545 s**: compute per second of audio, one request at a time, whole take (programs compiled), on TPU v6e-1
- **4.95 s of audio per second**: 8 requests at once (programs compiled), on TPU v6e-1
- **11.33 GB**: chip memory peak (8 slots, 3,072 tokens each), on TPU v6e-1
- **0.450 to 0.452 s**: first audio, one request, decoded while it is generated (programs compiled), on TPU v5e-1
- **0.903 to 0.917 s**: compute per second of audio, one request at a time, whole take (programs compiled), on TPU v5e-1
- **4.65 s of audio per second**: 8 requests at once (programs compiled), on TPU v5e-1
- **11.29 GB**: chip memory peak (8 slots, 3,072 tokens each), on TPU v5e-1
- **maya1-BF16.gguf** (6.60 GB, 16.00 bits per weight): mean KL 0.00255, top-1 agreement 91.4% vs the float32 reference (38 prompts, 15,392 codes)

## Quickstart

On a Google Cloud TPU VM (the [setup guide](https://mashbox.gg/docs/google-cloud) gets you there from a new Google account; coding agents can follow [AGENTS.md](https://github.com/mashbox-labs/mashbox/blob/v0.1.15/AGENTS.md)):

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]>=0.1.15" --with-executables-from huggingface_hub
~/.local/bin/mashbox run maya1 --text "Hello there. <laugh> It works." --voice alloy --out hello.wav
```

## Results

| Chip | Measure | Result | Baseline |
|---|---|---|---|
| TPU v6e-1 | First audio, one request, decoded while it is generated (programs compiled) | 0.266 to 0.269 s | n/a |
| TPU v6e-1 | Compute per second of audio, one request at a time, whole take (programs compiled) | 0.536 to 0.545 s | n/a |
| TPU v6e-1 | 8 requests at once (programs compiled) | 4.95 s of audio per second | n/a |
| TPU v6e-1 | Chip memory peak (8 slots, 3,072 tokens each) | 11.33 GB | n/a |
| TPU v5e-1 | First audio, one request, decoded while it is generated (programs compiled) | 0.450 to 0.452 s | n/a |
| TPU v5e-1 | Compute per second of audio, one request at a time, whole take (programs compiled) | 0.903 to 0.917 s | n/a |
| TPU v5e-1 | 8 requests at once (programs compiled) | 4.65 s of audio per second | n/a |
| TPU v5e-1 | Chip memory peak (8 slots, 3,072 tokens each) | 11.29 GB | n/a |

## Files

| File | Format | Size (GB) | Bits per weight | Quality vs full precision | Speed |
|---|---|---|---|---|---|
| `maya1-BF16.gguf` | GGUF | 6.60 | 16.00 | mean KL 0.00255, top-1 agreement 91.4% vs the float32 reference (38 prompts, 15,392 codes) | first audio 0.27 s, 0.54 s of compute per second of audio for 1 request (TPU v6e-1) |
| `snac/snac_24khz_decoder.npz` | NumPy arrays (npz) | 0.05 | n/a | 109.9 dB SNR or more vs the official SNAC decoder in float32 (38 prompts) | n/a |

## Chips

| Chip | Status | Note |
|---|---|---|
| TPU v5e | Supported | one chip (v5e-1) |
| TPU v6e | Supported | one chip (v6e-1) |
| TPU v5p slice (v5p-8) | Not tested yet |  |
| TPU v6e slice (v6e-4) | Not tested yet |  |

## Kernels

This model runs on plain JAX. It ships no compiled kernels.

## Links

|  |  |
|---|---|
| Code | [mashbox-labs/mashbox](https://github.com/mashbox-labs/mashbox/tree/v0.1.15/src/mashbox/models/maya1) |
| Guide | [mashbox.gg/docs/models/maya1](https://mashbox.gg/docs/models/maya1) |
| Original model | [maya-research/maya1](https://huggingface.co/maya-research/maya1) by Maya Research |
| Setup from a new Google account | [mashbox.gg/docs/google-cloud](https://mashbox.gg/docs/google-cloud) |
| Companies | hello@mashbox.gg |

## Licence

Weights: Apache-2.0 licence, from [maya-research/maya1@b1b9b38c](https://huggingface.co/maya-research/maya1/tree/b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572); [hubertsiuzdak/snac_24khz@d73ad176](https://huggingface.co/hubertsiuzdak/snac_24khz/tree/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec): MIT; code: Apache-2.0 (MASHBOX Labs). [snac/LICENSE](https://huggingface.co/mashbox-labs/maya1-TPU/blob/main/snac/LICENSE) in this repo is the licence file of [hubertsiuzdak/snac@8f79a718](https://github.com/hubertsiuzdak/snac/blob/8f79a718f1ad71f94f79999f0071348227aff22e/LICENSE), unchanged. Credit: [hubertsiuzdak/snac@8f79a718](https://github.com/hubertsiuzdak/snac/tree/8f79a718f1ad71f94f79999f0071348227aff22e), MIT, code (the SNAC decoder in plain JAX follows its decode path); its LICENSE ships unchanged in the package as `LICENSE-SNAC` beside that code.
