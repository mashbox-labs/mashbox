# Qwen3.8-27B on TPU

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen, running on a single Google Cloud TPU v6e chip, with an OpenAI-compatible server. The model code is plain JAX in [src/mashbox/models/qwen3_8_27b](../../src/mashbox/models/qwen3_8_27b). The weights come as one BSHJ file from [mashbox-labs/Qwen3.8-27B-TPU](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU), and decoding uses our compiled TPU kernels (Culet).

> Status: staging. On 28 October 2026 this page ran on a new TPU v6e-1 as a first-time user would follow it: install, first answer, the server through an SSH tunnel and 8 users at once. The package goes to PyPI and the weights repo goes public at release.

## What it is

![A first-time user on a new TPU v6e-1: install, weights download, first answer](../../hf/qwen3-8-27b/assets/hero.gif)

A first-time user on a new TPU v6e-1 (28 October 2026): install, weights download, first answer. Sped up 10x, pauses over 2 s cut to 2 s.

How it runs:

- **Port:** plain JAX in [src/mashbox/models/qwen3_8_27b](../../src/mashbox/models/qwen3_8_27b).
- **Kernels:** the compiled TPU kernels ship as one bundle in [mashbox-labs/Qwen3.8-27B-TPU](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU) under `kernels/`, without source; `mashbox run` downloads it and checks its sha256 against `models.json`. On a chip or JAX version the bundle was not built for, the model runs on plain JAX and prints one line saying so.
- **Weights:** one BSHJ file from [mashbox-labs/Qwen3.8-27B-TPU](https://huggingface.co/mashbox-labs/Qwen3.8-27B-TPU) (see [Variants](#variants)).

## What you need

| | TPU v6e (v6e-1) | TPU v5e |
|---|---|---|
| Status | passing (28 October 2026) | does not fit: the weights alone are about 27 GB and a v5e chip has 16 GB |
| Create with | `--accelerator-type=v6e-1 --version=v2-alpha-tpuv6e` | |

One TPU v6e chip is $2.70 an hour at Google's on-demand list price (read 19 October 2026). The chip bills from create to delete. Context is 2,048 tokens per request with 8 request slots.

You also need the Google Cloud CLI set up: [google-cloud.md](../google-cloud.md) does that from a new Google account.

## Install

On the chip, the package installs with one line; the command in [Run](#run) does it for you:

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
```

The chip's system Python is 3.10 and JAX for TPU needs 3.12, so pip installs [uv](https://docs.astral.sh/uv/), and uv installs `mashbox` with its own Python 3.12. The `mashbox` command goes to `~/.local/bin`; the commands on this page use that full path. pip and uv each print a warning that `~/.local/bin` is not on your `PATH`: that is expected. The install took 5 to 7 seconds on a new TPU v6e-1 (28 October 2026).

## Run

From your own machine, with the Google Cloud CLI set up ([google-cloud.md](../google-cloud.md) does that from a new Google account):

```bash
gcloud compute tpus tpu-vm create qwen-1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh qwen-1 --zone=us-east5-b --command='python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]" && ~/.local/bin/mashbox run qwen3-8-27b --prompt "Write a haiku about the sea."'
gcloud compute tpus tpu-vm delete qwen-1 --zone=us-east5-b --quiet
```

The first run downloads the tokenizer and config from Qwen's repo (pinned revision), the weights file and the kernel bundle, and checks each against the sha256 in [models.json](../../models.json).

### Serve it

```bash
gcloud compute tpus tpu-vm ssh qwen-1 --zone=us-east5-b --command='~/.local/bin/mashbox run qwen3-8-27b' -- -L 8000:127.0.0.1:8000
```

The server listens on 127.0.0.1:8000 on the chip, and the SSH tunnel brings it to port 8000 on your machine. It prints `"event": "ready"` when it takes requests. Then, from your machine:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

## Options

`~/.local/bin/mashbox run qwen3-8-27b [options]`

| Option | Default | What it does |
|---|---|---|
| `--prompt TEXT` | none | answer one chat turn, print it and exit; without it, start the server |
| `--thinking` | off | thinking mode for `--prompt` |
| `--max-tokens N` | 512 | most new tokens for `--prompt` |
| `--temperature T` | 0 | 0 is greedy |
| `--top-p P` | 1.0 | nucleus sampling cut |
| `--top-k K` | 0 | 0 means no top-k cut |
| `--seed N` | 0 | seed for sampled answers |
| `--host H` | 127.0.0.1 | server address; keep it on 127.0.0.1 and use the SSH tunnel |
| `--port N` | 8000 | server port |
| `--ctx N` | 2048 | tokens per request (prompt plus answer) |
| `--slots N` | 8 | requests decoded together; the kernel bundle is built for 8, other values run plain JAX |
| `--steps N` | 8 | decode steps per device call |
| `--cache-dir DIR` | `~/.cache/mashbox/xla` | compile cache (also `JAX_COMPILATION_CACHE_DIR`) |
| `--no-warm` | off | skip compiling the first request's programs before `ready` |
| `--weights FILE` | download | a local .bshj file instead of the download; its sha256 is still checked |
| `--no-kernels` | off | plain JAX only: slower decode, same outputs within rounding |

`~/.local/bin/mashbox download qwen3-8-27b` fetches and checks the files without loading them. `~/.local/bin/mashbox env` prints the versions and chip for bug reports.

### Recommended settings

The default is greedy (temperature 0): the same prompt gives the same answer. For sampled answers with thinking off, Qwen's settings are below; send them in each server request.

| Request field | Our default | Qwen's settings, thinking off |
|---|---|---|
| `temperature` | 0 (greedy) | 0.7 |
| `top_p` | 1.0 | 0.8 |
| `top_k` | 0 (no cut) | 20 |
| `presence_penalty` | 0 | 1.5 |

With these settings our server gave 33.0 tokens/s for one request at a time on TPU v6e-1 (28 October 2026, [Compared with vLLM TPU](#compared-with-vllm-tpu-on-the-same-chip)).

### Server routes and sampling

Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, with `"stream": true` for server-sent events. Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`; temperature 0 (the default) is greedy. Thinking mode: `"chat_template_kwargs": {"enable_thinking": true}`. Requests sent within 20 ms of each other share the chip in one batch of up to 8. A request that arrives while a batch is running waits for that batch to finish, then runs with the others that arrived meanwhile.

## Variants

| File | Format | Size | Bits per weight | Quality vs full precision | Speed on TPU v6e-1 |
|---|---|---|---|---|---|
| `Qwen3.8-27B-BQ8_R.bshj` | BSHJ | 26.94 GB | 8.01 | mean KL 0.00065, top-1 agreement 99.72% vs bf16 (8 prompts x 32 tokens) | 38.6 tokens/s for 1 request, 269 tokens/s for 8 |

Made from the original bf16 weights: each large weight matrix (and the embedding and output layers) is stored as int8 values with one f32 scale per output row; norms and the small linear-attention tensors keep their original dtype. The original weights stay at [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Speed measured 2026-10-28 on TPU v6e-1 (64-token prompt, 128 new tokens, greedy, default options); quality measured 2026-10-27 against Hugging Face transformers in bf16 with the same weights.

## Performance

Measured on 28 October 2026 on one TPU v6e-1 (on demand, zone europe-west4-a), after following this page on a new chip: `mashbox` 0.1.0 with its kernel bundle and JAX 0.11.2 for TPU (`~/.local/bin/mashbox env` prints the full list), through the server with the default options (greedy, `--ctx 2048 --slots 8`). Chat prompts of about 30 tokens, 128 new tokens per request.

| What | Result |
|---|---|
| 1 user | 37.5 tokens/s; first token after 52 ms |
| 8 users sending at the same moment, streamed | first token after 0.11 s for each of the 8 |
| 8 users, each sending a new request as soon as its answer came back, for 60 s | 258 tokens/s in total; a request took 4.0 s (median) |
| Install (`pip install uv` and `uv tool install`) | 5 to 7 s |
| Start to ready on a new chip: download of the 26.9 GB file from Hugging Face (about 200 MB/s there), sha256 check, load; compiled programs come from the shipped compile cache | 138 s; 9.8 s of it after the download and check |
| The same first start without the compile cache (same chip, same day) | 166.6 s; 25.7 s after the download and check |
| Start to ready, later runs | 6.6 s (median of 3) |

The engine on its own does 38.6 tokens/s for 1 request and 269 tokens/s for 8 ([Variants](#variants)). On TPU v6e with JAX 0.11.2, `mashbox run` also downloads a 47 MB compile cache from the model repo (checked by sha256 like the weights), so the first start loads its compiled programs instead of compiling them. Any other chip or JAX version compiles on first use and keeps the result. A prompt length the server has not seen before compiles its programs once (5.8 s extra on that first request in our run); the result is kept on disk in `~/.cache/mashbox/xla`, so later runs skip it. Download speed depends on the zone and the day.

Quality from the installed package on the same chip, the same day: mean KL 0.00065 and top-1 agreement 99.72% against Hugging Face transformers in bf16 on our 8 fixed prompts (367 token positions), the same as in [Variants](#variants).

### Compared with vLLM TPU on the same chip

Measured on 28 October 2026 on one TPU v6e-1 (on demand, zone us-east5-b), one server on the chip at a time, the same OpenAI chat requests to both (streamed, 256 new tokens, thinking off). Ours: `mashbox` 0.1.0 with `Qwen3.8-27B-BQ8_R.bshj` (26.94 GB, 8.01 bits per weight) and default options. vLLM: vllm-tpu 0.32.0 with Qwen's own 8-bit file [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), the closest like-for-like.

| What | mashbox | vLLM TPU 0.32.0 |
|---|---|---|
| Starts on one v6e-1 with default settings | yes | no: Qwen's bf16 weights run out of chip memory; the FP8 weights start only with `--max-num-seqs 8 --gpu-memory-utilization 0.98 --no-enable-prefix-caching` |
| Start to ready, weights already on disk, empty compile caches | 130.1 s | 1,606.7 s |
| 1 request at a time, short prompts, greedy | 38.1 tokens/s | 31.4 tokens/s |
| 1 request at a time, 790-token prompts, greedy | 37.0 tokens/s | 31.0 tokens/s |
| 1 request at a time, sampled (temperature 0.7, top_p 0.8, top_k 20) | 33.0 tokens/s | 23.7 tokens/s |

<!-- TODO source="public link to the v6e-1 run record (status/tpu-v6e.json) once tpu_ci publishes it" -->

## Use it with your coding agent

Claude Code, Codex and Pi can use Qwen3.8-27B as their LLM: start the server with `--ctx 32768 --slots 1` and point the agent at it, as in [Use with coding agents](../agents-on-tpu.md). [AGENTS.md](../../AGENTS.md) gives an agent the rules and the steps to set up the chip itself, and tells it to ask you before anything that costs money.

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| one line saying the model runs on plain JAX | the kernel bundle was not built for this chip or JAX version, or `--slots` is not 8 | use TPU v6e and the default `--slots 8`; `~/.local/bin/mashbox env` shows the chip and versions |
| out of memory | another process holds the chip, or `--ctx`/`--slots` are above the defaults | run on a TPU v6e-1 with nothing else on it, with `--ctx 2048 --slots 8` |
| `sha256 ... does not match the pinned ...` | a download was cut off or changed | delete the file and download again, as the message says |
| `ERROR: No matching distribution found for jax[tpu]==0.11.2` | plain `pip install "mashbox[tpu]"` ran on the chip's Python 3.10; JAX for TPU needs 3.12 | use the install line in [Install](#install) |
| `mashbox: command not found` | `~/.local/bin` is not on your `PATH` until you log in again | use the full path `~/.local/bin/mashbox` |
| `429 Too Many Requests` from `cdn.hf.co`, then a traceback | Hugging Face's Xet transfer is limiting the download | run the command again; `mashbox` downloads over plain HTTPS unless you set `HF_HUB_DISABLE_XET=0`, so remove that setting if you added it |
| `The TPU is already in use by another process` | one process at a time can use the chip, for example a server you started in another SSH session | stop that process (Ctrl-C in its session) and run again; if `sudo lsof -w /dev/vfio/0` shows no process, the lock file is left over: `sudo rm /tmp/libtpu_lockfile`, as the message says |
| `There is no more capacity in the zone` when you create the chip | no free TPU v6e chips in that zone right now | create it in another zone from [step 6 of the setup guide](../google-cloud.md#6-create-one-chip) and use that zone in every later command |

The last five rows are errors we hit ourselves, following this page on a new TPU v6e-1 on 28 October 2026.

## FAQ

- **Why not TPU v5e?** The weights alone are about 27 GB and a v5e chip has 16 GB.
- **Can several people use one server?** Yes. Requests sent together share the chip in one batch of up to 8; a request that arrives while a batch runs waits for it to finish. With 8 users each sending a new request as soon as the last answer came back, the server gave 258 tokens/s in total on TPU v6e-1 ([Performance](#performance)).
- **How do I turn on thinking mode?** `--thinking` for `--prompt`, or `"chat_template_kwargs": {"enable_thinking": true}` in a server request.

## Changelog

| Version | Date | What changed |
|---|---|---|
| 0.1.0 | staging, not released | First release: one TPU v6e chip, OpenAI-compatible server, compiled kernels for 8 request slots, compile cache for TPU v6e. |

## Credits and licences

Qwen3.8-27B is made by Qwen: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Model weights: Apache-2.0, Qwen's licence as published with Qwen3.8-27B (recorded 28 October 2026), kept unchanged in [licenses/qwen3-8-27b/LICENSE](../../licenses/qwen3-8-27b/LICENSE).

Our code in this repo: Apache-2.0 (see [LICENSE](../../LICENSE)). Compiled kernels: see [LICENSE-KERNELS](../../LICENSE-KERNELS).
