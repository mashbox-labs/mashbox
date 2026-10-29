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
| `MODEL` |  | model id, as in models.json (e.g. qwen3-8-27b) |
| `--weights FILE` |  | a local .bshj file instead of the download (its sha256 is still checked) |
| `--no-kernels` | off | plain JAX only (slower decode, same outputs within rounding) |
| `--prompt TEXT` |  | answer one chat turn, print it and exit; without it, start the server |
| `--thinking` | off | thinking mode for --prompt |
| `--max-tokens N` | `512` | most new tokens for --prompt |
| `--temperature T` | `0.0` | sampling temperature for --prompt; 0 is greedy |
| `--top-p P` | `1.0` | nucleus sampling cut for --prompt |
| `--top-k K` | `0` | top-k cut for --prompt; 0 means none |
| `--seed N` | `0` | seed for sampled --prompt answers |
| `--host H` | `127.0.0.1` | server address; keep it on 127.0.0.1 and use an SSH tunnel |
| `--port N` | `8000` | server port |
| `--api-key-file FILE` |  | require an API key from FILE (one per line; each key has its own prompt cache) on /v1 requests; also MASHBOX_API_KEY=k1,k2. Default: none on 127.0.0.1, and on any other --host a key made in ~/.config/mashbox/api-key |
| `--ui` | off | also serve the MASHBOX Chat page at http://127.0.0.1:<port>/ |
| `--ctx N` | `2048` | context per request in tokens (prompt plus answer) |
| `--slots N` | `8` | requests decoded together; the kernel bundle is built for 8, other values run plain JAX |
| `--steps N` | `8` | decode steps per device call |
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
| `MODEL` |  | model id, as in models.json (e.g. qwen3-8-27b) |
| `--weights FILE` |  | a local .bshj file instead of the download (its sha256 is still checked) |
| `--no-kernels` | off | plain JAX only (slower decode, same outputs within rounding) |

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
