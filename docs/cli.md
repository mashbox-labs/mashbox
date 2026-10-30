<!-- Generated from the argparse parsers in src/mashbox/cli.py and src/mashbox/tpu/__init__.py by tools/hf_github_sync_check.py render --write. Do not edit. -->
# Command line

Every `mashbox` command and option, rendered from the package's own parsers (`mashbox <command> --help` prints the same). On a TPU VM the command is `~/.local/bin/mashbox` until `~/.local/bin` is on your PATH.

## mashbox env

Versions and chip, for bug reports.

```
mashbox env
```

No options.

## mashbox run

Download (first time), load and serve a model, or answer one prompt.

```
mashbox run MODEL [options]
```

| Option | Default | What it does |
|---|---|---|
| `MODEL` |  | model id, as in models.json (e.g. qwen3-8-27b), a .gguf file, or a Hugging Face GGUF repo with an optional quant tag (e.g. unsloth/Qwen3.8-27B-GGUF or unsloth/Qwen3.8-27B-GGUF:Q8_0) |
| `--weights FILE` |  | a local .bshj file instead of the download (its sha256 is still checked) |
| `--no-kernels` | off | plain JAX only: a catalog model runs without its TPU kernels (slower decode, same outputs within rounding) |
| `--prompt TEXT` |  | answer one chat turn, print it and exit; without it, start the server |
| `--thinking` | off | thinking mode for --prompt |
| `--max-tokens N` | `512` | most new tokens for --prompt |
| `--temperature T` | `0.0` | sampling temperature for --prompt; 0 is greedy |
| `--top-p P` | `1.0` | nucleus sampling cut for --prompt |
| `--top-k K` | `0` | top-k cut for --prompt; 0 means none |
| `--seed N` | `0` | seed for sampled --prompt answers |
| `--host H` | `127.0.0.1` | server address; keep it on 127.0.0.1 and use an SSH tunnel |
| `--port N` | `8000` | server port; 0 picks a free one, named in the "listening" log line (and "ui" with --ui) |
| `--api-key-file FILE` |  | require an API key from FILE on /v1 requests: one per line, a key or a key then its tenant's name; keys of one tenant share one prompt cache and one waiting queue, and a key without a name is its own tenant; only you may read or write FILE (chmod 600). Also MASHBOX_API_KEY=k1,k2. Default: none on 127.0.0.1, and on any other --host a key made in ~/.config/mashbox/api-key |
| `--ui` | off | also serve the MASHBOX Chat page at http://127.0.0.1:<port>/ |
| `--ctx N` | `2048` | context per request in tokens (prompt plus answer) |
| `--slots N` | `8` | requests decoded together; a catalog model's kernels are built for 8 and other values run it on plain JAX; at the default, a GGUF file takes as many as fit the chip's memory |
| `--steps N` | `8` | decode steps per device call |
| `--mtp K` |  | speculative decoding with the model's own multi-token-prediction head: K drafts per verify pass (1 to 3; 0 = off). Default: on only on chips where it passed its checks, else off. Uses the model's draft file (Qwen3.8-27B: --mtp 2); one chip only |
| `--mtp-file FILE` |  | a local draft .bshj file instead of the download (its sha256 is still checked) |
| `--speculation SPECULATION` |  | speculative decoding. auto (default): the first drafter that applies to the model file, in this order: 1 a trained drafter made for this model (not in this release); 2 the model's own MTP head; 3 a smaller draft model with the same tokenizer (only one you give with --draft-model); 4 n-gram over the request's own text (llama, qwen2, qwen3, qwen35, exaone and granite GGUF files), on only where it passed its checks on this chip (Qwen3.8-27B reads its MTP head only); off; or force one kind (then it says 'not gated on this chip'). One chip only (`auto`, `off`, `mtp`, `ngram`, `draft`) |
| `--draft-k K` |  | drafts per verify pass for a drafter you pick: ngram 1 to 8 (default 3); draft 1 to 8 (default 3); mtp 1 to the head's k_max (Qwen3.8-27B: 1 to 3, default 2) |
| `--draft-model FILE` |  | --speculation draft with this draft model: a smaller GGUF file or repo (as the model argument) with the same tokenizer, a llama, qwen2, qwen3, exaone or granite file; for a GGUF model only. Its weights and cache take chip memory next to the model's |
| `--tensor-parallel-size N` | `0` | chips to split the model across, as vLLM (1, 2 or 4); default: the chips of this host, at most 4 (1 on a TPU v6e-1, 4 on a TPU v5e-4 or v6e-4) |
| `--cache-dir DIR` |  | compile cache (default ~/.cache/mashbox/xla, or JAX_COMPILATION_CACHE_DIR) |
| `--no-warm` | off | skip compiling the first request's programs before ready |
| `--no-prefix-cache` | off | prefill every prompt in full (by default the server reuses a cached prompt start, see docs/agents-on-tpu.md) |
| `--cache-ram MIB` | `8192` | host memory for saved prompt states, as llama-server's --cache-ram: a conversation that loses its slot is kept here and restored when it comes back (default 8192, 0 = off) |
| `--ctx-checkpoints N` | `8` | prompt points kept per slot where a cached prompt can resume (default 8) |
| `--checkpoint-min-step N` | `8192` | minimum tokens between kept user-message points, as llama-server (default 8192) |

## mashbox download

Download and check a model's files without loading them.

```
mashbox download MODEL [options]
```

| Option | Default | What it does |
|---|---|---|
| `MODEL` |  | model id, as in models.json (e.g. qwen3-8-27b), a .gguf file, or a Hugging Face GGUF repo with an optional quant tag (e.g. unsloth/Qwen3.8-27B-GGUF or unsloth/Qwen3.8-27B-GGUF:Q8_0) |
| `--weights FILE` |  | a local .bshj file instead of the download (its sha256 is still checked) |
| `--no-kernels` | off | plain JAX only: a catalog model runs without its TPU kernels (slower decode, same outputs within rounding) |

## mashbox models

The models in this version's catalog, by publisher, then model, then file.

```
mashbox models [options]
```

| Option | Default | What it does |
|---|---|---|
| `--json` | off | the same list as JSON |

## mashbox inspect

What a GGUF file or repo holds (architecture, block types, tokenizer) and how it would run here, without loading it.

```
mashbox inspect MODEL [options]
```

| Option | Default | What it does |
|---|---|---|
| `MODEL` |  | a .gguf file or a Hugging Face GGUF repo[:quant tag] |
| `--no-download` | off | for a repo: only show which file llama.cpp's rule picks |

## mashbox tpu up

Create a chip for a model, install, start the server, open a tunnel.

```
mashbox tpu up MODEL [options] [-- options for mashbox run on the chip]
```

| Option | Default | What it does |
|---|---|---|
| `MODEL` |  |  |
| `--zone ZONE` |  | only this zone (default: the model's zones in order) |
| `--chip CHIP` |  | chip type, e.g. v6e-1, v5e-4, v6e-4 (default: the model's first; the model's docs page lists the chips it runs on) |
| `--spot` | off | a spot chip: cheaper, Google can stop it at any time |
| `--name NAME` |  | chip name (default mashbox-<model>) |
| `--max-hours MAX_HOURS` | `8.0` | the chip deletes itself this long after it starts (default 8; 0 = off) |
| `--queue-minutes QUEUE_MINUTES` | `60.0` | how long to wait for a queued chip (default 60; 0 = no queued requests) |
| `--yes` | off | create without asking |
| `--agents` | off | start the server with the coding-agent settings of the model's row |
| `--restart` | off | stop a running server (Ctrl-C, SIGINT) and start it again |
| `--project PROJECT` |  | Google Cloud project (default: gcloud's) |

## mashbox tpu down

Delete a mashbox chip and its queued request.

```
mashbox tpu down [NAME] [options]
```

| Option | Default | What it does |
|---|---|---|
| `NAME` |  |  |
| `--zone ZONE` |  | this zone only |
| `--all` | off | every mashbox chip and queued request in the project |
| `--project PROJECT` |  | Google Cloud project (default: gcloud's) |

## mashbox tpu status

List this project's mashbox chips and queued requests.

```
mashbox tpu status [options]
```

| Option | Default | What it does |
|---|---|---|
| `--zone ZONE` |  | this zone only |
| `--project PROJECT` |  | Google Cloud project (default: gcloud's) |

## mashbox tpu ssh

Open a shell on a running mashbox chip.

```
mashbox tpu ssh [NAME] [options]
```

| Option | Default | What it does |
|---|---|---|
| `NAME` |  |  |
| `--zone ZONE` |  |  |
| `--project PROJECT` |  | Google Cloud project (default: gcloud's) |

## mashbox tpu tunnel

Open the local tunnel to the server on a running mashbox chip again.

```
mashbox tpu tunnel [NAME] [options]
```

| Option | Default | What it does |
|---|---|---|
| `NAME` |  |  |
| `--zone ZONE` |  |  |
| `--project PROJECT` |  | Google Cloud project (default: gcloud's) |
