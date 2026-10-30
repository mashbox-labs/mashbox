# Supported models

Every model and file in this version of mashbox's catalog, on one page in two parts. **MASHBOX builds** are the models we port or make ourselves, each with its measured number against its baseline. **GGUF files that run in our engine** are files from other publishers that mashbox runs as they are, one section per model family, with each file's maker, architecture and quant. The Level column says how far each one has been checked. `mashbox models` prints the same rows from the package's own catalog, and `mashbox models --json` prints them as JSON.

```bash
mashbox models
```

Each row names the chips it runs on and the command that runs it. A GGUF link opens the exact file revision that ran.

<!-- catalog:start -->
**26 entries: 2 MASHBOX ports, 1 MASHBOX model and 23 GGUF files of 18 models.**

## MASHBOX builds

Models we port or make ourselves. A BSHJ build is our weights file for the model, run on our TPU kernels. Measured is each one's number against its baseline, on the chip named.

| Model | Format | Level | Runs on | Measured | Run |
|---|---|---|---|---|---|
| **[Hy-Nomas](models/hy-nomas.md)**<br>Made by Tencent Hunyuan, ported to TPU by MASHBOX Labs | TPU port<br>World model: text and movement keys in, video out<br>[Hugging Face](https://huggingface.co/mashbox-labs/Hy-Nomas-TPU) | MASHBOX build | TPU v6e-4 | Compute per second of video, quality preset, Tencent decoder, 1.00 = real time: **1.045 s** (TPU v6e-4)<br>Compute per second of video, quality preset, MASHBOX Fast decoder: **0.689 s** (TPU v6e-4) | `~/.local/bin/mashbox run hy-nomas --ui` |
| **[Qwen3.8-27B](models/qwen3-8-27b.md)**<br>Made by Qwen, ported to TPU by MASHBOX Labs | BSHJ build: BQ8_R<br>LLM: text in, text out<br>[Hugging Face](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU) | MASHBOX build | TPU v5e-4, TPU v6e-1, TPU v6e-4 | **38.1 vs 31.4 tok/s** vs vLLM on the same chip, 1 request at a time (TPU v6e-1)<br>**74.84 tok/s** with speculative decoding (--mtp 2), 1 request at a time, greedy; 37.77 tok/s with it off (TPU v6e-1)<br>**326.6 vs 1,606.7 s** first start to ready, weights on disk, empty compile caches, vs vLLM on the same chip type (TPU v6e-1) | `~/.local/bin/mashbox run qwen3-8-27b` |
| **[Track2Rig](models/track2rig.md)** (Experimental)<br>Trained by MASHBOX Labs | MASHBOX model<br>Rigging: 3D point tracks in, rig (JSON and glTF) out<br>[Hugging Face](https://huggingface.co/mashbox-labs/Track2Rig) | MASHBOX build | CPU, TPU v6e | **0.784% vs 0.915%** held-out error vs SSDR at the same bone count, synthetic test set with noisy, partly hidden tracks, % of body size, lower is better (CPU, AMD EPYC 7502P)<br>**6.24% vs 7.73%** joint error vs SSDR, % of body size, lower is better (CPU, AMD EPYC 7502P) | `mashbox run track2rig examples/track2rig/tracks/grison_walk.npz -o grison_rig` |

## GGUF files that run in our engine

GGUF files from other publishers, run as they are by mashbox. Level and Runs on say how far each file has been checked; a speed shows only where a chip run measured it on our kernels.

### Qwen

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct**<br>by Qwen | [Qwen2.5-0.5B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)<br>from bartowski, 0.4 GB | `qwen2` | Q4_K_M | Tested | TPU v6e-1 | `mashbox run bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M` |
| **Qwen2.5-1.5B-Instruct**<br>by Qwen | [Qwen2.5-1.5B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/blob/9eadc66189c7641e1ddd226b8267a9119b2ce2d4/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)<br>from bartowski, 0.99 GB | `qwen2` | Q4_K_M | Tested | TPU v6e-1 | `mashbox run bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` |
| **Qwen3-0.6B**<br>by Qwen | [Qwen3-0.6B-Q8_0.gguf](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf)<br>from unsloth, 0.64 GB | `qwen3` | Q8_0 | Tested | TPU v6e-1 | `mashbox run unsloth/Qwen3-0.6B-GGUF:Q8_0` |
| **Qwen3.5-0.8B**<br>by Qwen | [Qwen3.5-0.8B-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/blob/807dfd9a6fadbf703207aac3bfb0b2b3040d077c/Qwen3.5-0.8B-Q8_0.gguf)<br>from ggml-org, 0.83 GB | `qwen35` | Q8_0 | Tested | TPU v6e-1 | `mashbox run ggml-org/Qwen3.5-0.8B-GGUF:Q8_0` |
| **Qwen3.5-4B**<br>by Qwen | [Qwen3.5-4B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf)<br>from unsloth, 2.74 GB | `qwen35` | Q4_K_M | Tested | TPU v6e-1 | `mashbox run unsloth/Qwen3.5-4B-GGUF:Q4_K_M` |
| **Qwen2.5-0.5B-Instruct**<br>by Qwen | [qwen2.5-0.5b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf)<br>from Qwen, 0.49 GB | `qwen2` | q4_k_m | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 2.6e-11, top-1 131/131; llama.cpp's own path top-1 126/131; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m` |
| **Qwen2.5-1.5B-Instruct**<br>by Qwen | [qwen2.5-1.5b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf)<br>from Qwen, 1.12 GB | `qwen2` | q4_k_m | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 1.2e-11, top-1 131/131; llama.cpp's own path top-1 125/131; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m` |
| **Qwen2.5-1.5B-Instruct**<br>by Qwen | [Qwen2.5-1.5B-Instruct-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/blob/9eadc66189c7641e1ddd226b8267a9119b2ce2d4/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf)<br>from bartowski, 1.13 GB | `qwen2` | Q5_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 1.2e-11, top-1 131/131; llama.cpp's own path top-1 129/131; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q5_K_M` |
| **Qwen2.5-7B-Instruct**<br>by Qwen | [Qwen2.5-7B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/8911e8a47f92bac19d6f5c64a2e2095bd2f7d031/Qwen2.5-7B-Instruct-Q4_K_M.gguf)<br>from bartowski, 4.68 GB | `qwen2` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 1.1e-11, top-1 131/131; llama.cpp's own path top-1 124/131; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` |
| **Qwen2.5-Coder-1.5B-Instruct**<br>by Qwen | [qwen2.5-coder-1.5b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/blob/f86cb2c1fa58255f8052cc32aeede1b7482d4361/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf)<br>from Qwen, 1.12 GB | `qwen2` | q4_k_m | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 2.1e-12, top-1 131/131; llama.cpp's own path top-1 124/131; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m` |
| **Qwen3-1.7B**<br>by Qwen | [Qwen3-1.7B-Q4_0.gguf](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/blob/d7f544eead698dbd1f15126ef60b45a1e1933222/Qwen3-1.7B-Q4_0.gguf)<br>from unsloth, 1.06 GB | `qwen3` | Q4_0 | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 6.6e-12, top-1 131/131; llama.cpp's own path top-1 124/131; greedy 4 x 16 tokens: 2 identical, 2 near tie<br>TPU: Not tested yet | n/a | `mashbox run unsloth/Qwen3-1.7B-GGUF:Q4_0` |
| **Qwen3-14B**<br>by Qwen | [Qwen3-14B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-14B-GGUF/blob/530227a7d994db8eca5ab5ced2fb692b614357fd/Qwen3-14B-Q4_K_M.gguf)<br>from Qwen, 9 GB | `qwen3` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 3.3e-12, top-1 131/131; llama.cpp's own path top-1 130/131; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run Qwen/Qwen3-14B-GGUF:Q4_K_M` |
| **Qwen3-4B**<br>by Qwen | [Qwen3-4B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf)<br>from Qwen, 2.5 GB | `qwen3` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 1.7e-11, top-1 131/131; llama.cpp's own path top-1 129/131; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run Qwen/Qwen3-4B-GGUF:Q4_K_M` |
| **Qwen3.5-9B**<br>by Qwen | [Qwen3.5-9B-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/blob/1379f25c6b505a3fc737bd7818cb09389cf807c1/Qwen3.5-9B-Q4_K_M.gguf)<br>from lmstudio-community, 5.63 GB | `qwen35` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 1.5e-12, top-1 135/135; llama.cpp's own path top-1 134/135; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run lmstudio-community/Qwen3.5-9B-GGUF:Q4_K_M` |
| **Qwen3.5-9B**<br>by Qwen | [Qwen3.5-9B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf)<br>from unsloth, 5.68 GB | `qwen35` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 2e-12, top-1 135/135; llama.cpp's own path top-1 134/135; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run unsloth/Qwen3.5-9B-GGUF:Q4_K_M` |

### EXAONE

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **EXAONE-3.5-2.4B-Instruct**<br>by LG AI Research | [EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF/blob/142acae803a41c206e8d0fa978c6102c748911bb/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf)<br>from LGAI-EXAONE, 1.64 GB | `exaone` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 8e-12, top-1 141/141; llama.cpp's own path top-1 138/141; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:Q4_K_M` |

### Falcon

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **Falcon3-1B-Instruct**<br>by Technology Innovation Institute | [Falcon3-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Falcon3-1B-Instruct-GGUF/blob/9e280a1e3b7f287edb7248042e3d14f893227b21/Falcon3-1B-Instruct-Q8_0.gguf)<br>from bartowski, 1.78 GB | `llama` | Q8_0 | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 1.3e-12, top-1 144/144; llama.cpp's own path top-1 143/144; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/Falcon3-1B-Instruct-GGUF:Q8_0` |

### Gemma

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **gemma-3-1b-it**<br>by Google DeepMind | [gemma-3-1b-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/blob/f0b45be0aac41bd6a100a4b5734cad5f67255bfb/gemma-3-1b-it-Q4_K_M.gguf)<br>from unsloth, 0.81 GB | `gemma3` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 2.4e-07, top-1 142/142; llama.cpp's own path top-1 138/142; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run unsloth/gemma-3-1b-it-GGUF:Q4_K_M` |

### Granite

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **granite-3.3-2b-instruct**<br>by IBM | [granite-3.3-2b-instruct-Q4_K_M.gguf](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/blob/7cdf86ccd1f1bb3491c9b7017b033f2e51367397/granite-3.3-2b-instruct-Q4_K_M.gguf)<br>from ibm-granite, 1.55 GB | `granite` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 3.1e-12, top-1 145/145; llama.cpp's own path top-1 144/145; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run ibm-granite/granite-3.3-2b-instruct-GGUF:Q4_K_M` |
| **granite-3.3-2b-instruct**<br>by IBM | [ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF/blob/4e06dce39d4c5fcf65ca1cc69b92962e411d38e1/ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf)<br>from bartowski, 1.55 GB | `granite` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 3.3e-12, top-1 145/145; llama.cpp's own path top-1 144/145; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF:Q4_K_M` |

### Llama

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **Llama-3.2-1B-Instruct**<br>by Meta | [Llama-3.2-1B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q4_K_M.gguf)<br>from bartowski, 0.81 GB | `llama` | Q4_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 4.2e-12, top-1 133/133; llama.cpp's own path top-1 130/133; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M` |
| **Llama-3.2-3B-Instruct**<br>by Meta | [Llama-3.2-3B-Instruct-IQ4_XS.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-IQ4_XS.gguf)<br>from bartowski, 1.83 GB | `llama` | IQ4_XS | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 3e-12, top-1 133/133; llama.cpp's own path top-1 128/133; greedy 4 x 16 tokens: 4 identical, 0 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/Llama-3.2-3B-Instruct-GGUF:IQ4_XS` |

### Mistral

| Model | GGUF file | Architecture | Quant | Level | Runs on | Run |
|---|---|---|---|---|---|---|
| **Mistral-7B-Instruct-v0.3**<br>by Mistral AI | [Mistral-7B-Instruct-v0.3-Q3_K_M.gguf](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/blob/61fd4167fff3ab01ee1cfe0da183fa27a944db48/Mistral-7B-Instruct-v0.3-Q3_K_M.gguf)<br>from bartowski, 3.52 GB | `llama` | Q3_K_M | Checked on CPU<br>Runs, checked against llama.cpp on CPU: same weights NMSE 4.4e-12, top-1 150/150; llama.cpp's own path top-1 143/150; greedy 4 x 16 tokens: 3 identical, 1 near tie<br>TPU: Not tested yet | n/a | `mashbox run bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q3_K_M` |

GGUF architectures with a passing chip test (`general.architecture`): `qwen35` (Qwen3.5, Qwen3.6, Qwen3.8 (dense)), `qwen2` (Qwen2.5), `qwen3` (Qwen3).
<!-- catalog:end -->

## Levels

| Level | What the row means |
|---|---|
| MASHBOX build | One of our own TPU builds, with weights on Hugging Face and its own model page. |
| Tuned | A file that passed its chip test on the chips in its row, with a part tuned for it. |
| Tested | A file that passed its chip test on the chips in its row, with this version of mashbox. |
| Checked on CPU | A file that passed its check against llama.cpp on CPU, with no passing chip test in this version of mashbox. The row prints the numbers from that CPU check. |

A GGUF file with no row here can still run: `mashbox inspect FILE` says whether it runs here, and why not, before anything loads. [GGUF files](gguf.md#what-runs) says what runs.
