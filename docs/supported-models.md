# Supported models

Every model and file this version of mashbox runs on a TPU, by publisher, then model, then file. `mashbox models` prints the same list from the package's own catalog, and `mashbox models --json` prints it as JSON.

```bash
mashbox models
```

Each row names the chips it runs on and the command that runs it. A GGUF link opens the exact file revision that ran.

<!-- catalog:start -->
## MASHBOX Labs ([Hugging Face](https://huggingface.co/mashbox-labs))

| Model | Format or file | Level | Runs on | Speed | Run |
|---|---|---|---|---|---|
| **[Hy-Nomas](models/hy-nomas.md)**<br>Made by Tencent Hunyuan, ported to TPU by MASHBOX Labs | TPU port<br>World model: text and movement keys in, video out | MASHBOX build | TPU v6e-4 |  | `~/.local/bin/mashbox run hy-nomas --ui` |
| **[Qwen3.8-27B](models/qwen3-8-27b.md)**<br>Made by Qwen, ported to TPU by MASHBOX Labs | BSHJ build: BQ8_R<br>LLM: text in, text out | MASHBOX build | TPU v5e-4, TPU v6e-1, TPU v6e-4 | 38.1 vs 31.4 tok/s on TPU v6e-1, vs vLLM on the same chip, 1 request at a time | `~/.local/bin/mashbox run qwen3-8-27b` |
| **[Track2Rig](models/track2rig.md)** (Experimental)<br>Trained by MASHBOX Labs | MASHBOX model<br>Rigging: 3D point tracks in, rig (JSON and glTF) out | MASHBOX build | TPU v6e |  | `mashbox run track2rig examples/track2rig/tracks/grison_walk.npz -o grison_rig` |

## bartowski ([Hugging Face](https://huggingface.co/bartowski))

| Model | Format or file | Level | Runs on | Speed | Run |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct**<br>base Qwen/Qwen2.5-0.5B-Instruct | [Qwen2.5-0.5B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)<br>GGUF Q4_K_M, 0.4 GB | Tested | TPU v6e-1 |  | `mashbox run bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M` |
| **Qwen2.5-1.5B-Instruct**<br>base Qwen/Qwen2.5-1.5B-Instruct | [Qwen2.5-1.5B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/blob/9eadc66189c7641e1ddd226b8267a9119b2ce2d4/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)<br>GGUF Q4_K_M, 0.99 GB | Tested | TPU v6e-1 |  | `mashbox run bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` |

## ggml-org ([Hugging Face](https://huggingface.co/ggml-org))

| Model | Format or file | Level | Runs on | Speed | Run |
|---|---|---|---|---|---|
| **Qwen3.5-0.8B**<br>base Qwen/Qwen3.5-0.8B | [Qwen3.5-0.8B-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/blob/807dfd9a6fadbf703207aac3bfb0b2b3040d077c/Qwen3.5-0.8B-Q8_0.gguf)<br>GGUF Q8_0, 0.83 GB | Tested | TPU v6e-1 |  | `mashbox run ggml-org/Qwen3.5-0.8B-GGUF:Q8_0` |

## unsloth ([Hugging Face](https://huggingface.co/unsloth))

| Model | Format or file | Level | Runs on | Speed | Run |
|---|---|---|---|---|---|
| **Qwen3-0.6B**<br>base Qwen/Qwen3-0.6B | [Qwen3-0.6B-Q8_0.gguf](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf)<br>GGUF Q8_0, 0.64 GB | Tested | TPU v6e-1 |  | `mashbox run unsloth/Qwen3-0.6B-GGUF:Q8_0` |
| **Qwen3.5-4B**<br>base Qwen/Qwen3.5-4B | [Qwen3.5-4B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf)<br>GGUF Q4_K_M, 2.74 GB | Tested | TPU v6e-1 |  | `mashbox run unsloth/Qwen3.5-4B-GGUF:Q4_K_M` |

GGUF architectures with a passing chip test (`general.architecture`): `qwen35` (Qwen3.5, Qwen3.6, Qwen3.8 (dense)), `qwen2` (Qwen2.5), `qwen3` (Qwen3).
<!-- catalog:end -->

## Levels

| Level | What the row means |
|---|---|
| MASHBOX build | One of our own TPU builds, with weights on Hugging Face and its own model page. |
| Tuned | A file that passed its chip test on the chips in its row, with a part tuned for it. |
| Tested | A file that passed its chip test on the chips in its row, with this version of mashbox. |

A GGUF file with no row here can still run: `mashbox inspect FILE` says whether it runs here, and why not, before anything loads. [GGUF files](gguf.md#what-runs) says what runs.
