# GGUF files

mashbox reads [llama.cpp](https://github.com/ggml-org/llama.cpp)'s GGUF files and runs them on a TPU with its engine.
Bring the file you already have, or name a Hugging Face repo, and mashbox loads the file's own blocks. Nothing is
re-quantized: the TPU kernels take the blocks repacked into their layout, value for value, and every other weight runs on
the blocks as stored. For CPU, GPU, Mac or phone, use llama.cpp.

```bash
mashbox run unsloth/Qwen3.8-27B-GGUF
mashbox run unsloth/Llama-3.2-3B-Instruct-GGUF
mashbox run ./my-model-Q4_K_M.gguf --prompt "Hi"
mashbox inspect unsloth/Qwen3.8-27B-GGUF:Q8_0
```

The first two lines pick a file with llama.cpp's default rule and serve it, the third runs a local file for one answer,
and `inspect` shows what a file holds and how it would run here, without loading it. A quant tag after the repo name (`:Q8_0`) picks that file,
as `llama-server -hf` does.

## Run a GGUF file on a TPU

You need the Google Cloud CLI signed in to a project with billing and the TPU API turned on. [Run on Google Cloud TPU](google-cloud.md) steps 1 to 5 do that from a new Google account. Then, from your own machine:

```bash
gcloud compute tpus tpu-vm create mashbox-1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]" && ~/.local/bin/mashbox run bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M --prompt "Write a haiku about the sea."'
gcloud compute tpus tpu-vm delete mashbox-1 --zone=us-east5-b --quiet
```

The second line installs `mashbox` with Python 3.12 ([Install](install.md) has the other ways), picks the `Q4_K_M` file from the repo, downloads it, prints the file's block types, their share of the weights and which of them run on the kernels, then prints the answer. At the default `--slots 8` it takes as many request slots as fit the chip's memory and says so. A file that does not fit even one request stops before loading and names the chips it would fit on, if any. The GGUF reader comes with the package, and the kernels download by themselves ([Kernels](#kernels)).

Sampling works as for every other model: `--temperature`, `--top-p`, `--top-k` and `--seed` next to `--prompt` ([Command line](cli.md#mashbox-run)). [Supported models](supported-models.md) lists the models and files with a passing chip test, and the command for each.

> [!NOTE]
> The chip bills from create to delete. Delete it when you are done, or check what is still running with `gcloud compute tpus tpu-vm list --zone=us-east5-b`.

## Serve a GGUF file

Without `--prompt`, `mashbox run` serves the file with the [Server API](server-api.md), the same routes as Qwen3.8-27B. On the chip from the section above:

```bash
gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='~/.local/bin/mashbox run bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M' -- -L 8000:127.0.0.1:8000
```

The server listens on 127.0.0.1:8000 on the chip, and the SSH tunnel brings it to port 8000 on your machine. It prints `"event": "ready"` when it takes requests. Then, from your machine:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

`GET /v1/models` names the model by its repo (`bartowski/Llama-3.2-3B-Instruct-GGUF`), or by its file name for a local file; a request may leave `model` out. Requests run together, up to `--slots` at once, and a prompt that starts like an earlier one reuses the cached start. Speculative decoding on GGUF files: [Speculative decoding](speculation.md). The server and the engine run in one process, so after a chip fault, start the command again. Stopping the server works as for Qwen3.8-27B ([Serve it](models/qwen3-8-27b.md#serve-it)).

## Which file runs

A repo name picks one file with the same rule as `llama-server -hf`:

- the tag (default: `Q4_K_M`, then `Q8_0`) must appear in the file path followed by `.` or `-`, in any case;
- `imatrix` files and side files (`mmproj`, `mtp-`, `dflash-`, `eagle3-`, `dspark-`) are never the model;
- of a split model, the first shard is picked and every shard is downloaded;
- with no tag and no `Q4_K_M` or `Q8_0` file, the first model file in the repo is used.

The chosen file is printed (`event: gguf_file`) before the download starts. Gated repos need `HF_TOKEN`.

## What runs

| | |
|---|---|
| Architectures | the ones the GGUF reader takes. `mashbox inspect FILE` says whether a file runs here, and why not, before anything loads. A file that does not run here stops with one line naming the llama.cpp command that runs it. [The list below](#architectures) names the architectures with a GGUF file that passed a chip test. |
| Chips | TPU v6e, one chip. `--tensor-parallel-size` above 1 is for the MASHBOX builds and stops a GGUF run. |
| Block types read | the 26 types the pinned `gguf` package reads: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4 |
| Tokenizer and chat template | read from the file; the sampling defaults in the file too (`general.sampling.*`) |
| Split files | yes (`-00001-of-0000N.gguf`) |
| Multi-token prediction layer | read and mapped when the file has one (Unsloth's Qwen3.8 files do); the main pass skips it, and speculative decoding does not use it |

## Kernels

On a chip the package is built for (the table below), the first start of a GGUF file fetches the engine kernel
package: one file, attached to a mashbox release on GitHub, checked against its sha256 in `models.json` and kept in `~/.cache/mashbox/kernels/` (`MASHBOX_CACHE` moves the
cache). There is nothing to install or pick. A package that does not match the chip, the JAX version or the TPU runtime
is refused with one line, and the file then runs on the exact reference path.

<!-- kernels:start -->
| Chips | JAX | 2-D weights of these types run on the kernels | These quantized types run on the exact reference path |
|---|---|---|---|
| TPU v6e | 0.11.2 | Q4_0, Q4_1, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, NVFP4 | Q5_1, IQ1_S, IQ1_M, TQ1_0, TQ2_0, MXFP4 |
<!-- kernels:end -->

| Weight | How it runs |
|---|---|
| a 2-D weight of a type on the kernels | the TPU kernels, on the file's blocks repacked into their layout |
| BF16 | the chip's own bfloat16 matrix multiply, on the values as stored |
| F32 | exact float32 math on the values as stored, with no rebuild at each step |
| F16 | the exact path's float32 math on the values as stored, with no rebuild at each step; on TPU, some outputs can round differently from the exact path's |
| every other quantized type | the exact reference path |
| the token embedding table, when the file has its own output weight | the rows of each token, read exactly |

The exact reference path is plain JAX that rebuilds the weights in float32 from the file's blocks at every step, with
the same math as llama.cpp's `gguf` package (bit for bit on CPU). It is slow. Quantized weights also take it on a chip the
kernel package does not list (TPU v5e, for example), for a weight whose shape the package does not cover, and with
`--no-kernels`, which puts BF16 and F16 weights on it too. When a kernel fails on the first weight of a type and shape, the weights of that type and shape load on the
exact path, with a warning.

Before loading, mashbox prints its plan (`"event": "gguf_plan"`): the file's block types with their share of the weights,
the types this install runs on the kernels (`kernel_types`) and the share of the weights left on the exact path
(`reference_share`), with a warning that names those types. The `"event": "loaded"` line gives the gigabytes on each path
(`kernel_gb`, `reference_gb`, `resident_f32_gb`). `mashbox inspect` prints the same plan for the machine it runs on,
without loading the file.

mashbox estimates the chip memory a file needs before it loads: the weights as the chip holds them, the per-request
state, and the float32 values the exact path rebuilds at a time for its largest weight (none when every weight runs on
the kernels). It stops when the estimate is over the chip's memory, naming the chips that would fit, and warns when it is
close. `mashbox inspect` prints the estimate, for this install and for the exact path alone; XLA's own working memory
comes on top.

## Architectures

<!-- archs:start -->
| Architecture (`general.architecture`) | Models | Files in the catalog |
|---|---|---|
| `qwen35` | Qwen3.5, Qwen3.6, Qwen3.8 (dense) | unsloth-qwen3-5-4b-q4-k-m, unsloth-qwen3-5-9b-q4-k-m, lmstudio-community-qwen3-5-9b-q4-k-m, peculiar-ragdoll-qwen3-8-27b-q4-k-m, lmstudio-community-qwen3-8-27b-q4-k-m, unsloth-qwen3-8-27b-q4-k-m, huihui-ai-qwen3-8-27b-q4-k-m, ggml-org-qwen3-8-27b-q4-k-m |
| `llama` | Llama 3.x | bartowski-falcon3-1b-instruct-q8-0, bartowski-llama-3-2-3b-instruct-iq4-xs, bartowski-mistral-7b-instruct-v0-3-gguf-q3-k-m, bartowski-llama-3-2-1b-instruct-q4-k-m |
| `qwen2` | Qwen2.5 | bartowski-qwen2-5-0-5b-instruct-q4-k-m, bartowski-qwen2-5-1-5b-instruct-q4-k-m, bartowski-qwen2-5-1-5b-instruct-q5-k-m, qwen-qwen2-5-0-5b-instruct-q4-k-m, qwen-qwen2-5-1-5b-instruct-q4-k-m, bartowski-qwen2-5-7b-instruct-q4-k-m, qwen-qwen2-5-coder-1-5b-instruct-q4-k-m |
| `qwen3` | Qwen3 | unsloth-qwen3-0-6b-q8-0, unsloth-qwen3-1-7b-q4-0, qwen-qwen3-4b-q4-k-m, qwen-qwen3-14b-q4-k-m, unsloth-qwen3-4b-instruct-2507-q4-k-m |
<!-- archs:end -->

## Tokenizer note

mashbox tokenizes the way the model's own Hugging Face tokenizer does, including its NFC Unicode normalization.
llama.cpp skips that normalization, so text with decomposed accents (for example `e` followed by U+0301) can get
different token ids in llama.cpp. On normalized text the ids are the same.

## Checks you can run

- `pytest tests/cpu -k gguf`: every block type against `gguf`, bit for bit, each architecture's graph against
  llama.cpp's CPU logits on small random files, and the engine's batching and speculative-decoding hooks (no chip, no
  network).
- `python tools/gguf_check.py FILE.gguf --hf-dir DIR --llama-cpp DIR`: a real file, on CPU: every tensor header, the
  architecture mapping, a sample of every block type, and the tokenizer against the model's Hub files and llama.cpp.

## Credits

GGUF and its block formats are llama.cpp's, by the ggml authors. Files are read with llama.cpp's
[`gguf`](https://pypi.org/project/gguf/) package (MIT), pinned to 0.19.0. See [NOTICE](../NOTICE).
