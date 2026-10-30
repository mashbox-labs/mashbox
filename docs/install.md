# Install

<!-- install:start -->
Two lines on your own machine (Linux or macOS, x86_64 or arm64; on Windows, use WSL2). The first installs the `mashbox` command, the second gets a Google Cloud TPU chip and starts Qwen3.8-27B on it:

```bash
curl -fsSL https://mashbox.gg/install.sh | sh
mashbox tpu up qwen3-8-27b
```

Line 2 needs the Google Cloud CLI, signed in to a project with billing and the TPU API turned on; [Run on Google Cloud TPU](google-cloud.md) steps 1 to 5 do that from a new Google account. It prints the chip's price per hour and asks before it creates anything. Then it installs the same `mashbox` version on the chip and starts the server there, reached through an SSH tunnel from your machine. `mashbox tpu down mashbox-qwen3-8-27b` deletes the chip, and if you forget, the chip deletes itself at its max lifetime.

On a machine that has a TPU, such as a Cloud TPU VM you created yourself, the same line 1 installs the TPU version and `mashbox run` runs a model there:

```bash
curl -fsSL https://mashbox.gg/install.sh | sh
mashbox run qwen3-8-27b --prompt "Write a haiku about the sea."
```

If the shell does not find `mashbox` after line 1, `~/.local/bin` is not on your `PATH`: line 1 printed the `export PATH=...` line for this terminal, or call it as `~/.local/bin/mashbox`. To do the same by hand with `gcloud`, one chip and one answer: [Quickstart](quickstart.md).

What each install adds:

| Install | What it adds | Where it goes |
|---|---|---|
| `mashbox` | the `mashbox` command with `mashbox models` and `mashbox tpu`, which gets a chip and runs a model on it from your machine. It has no JAX, so models do not run on this machine. | your laptop or desktop: line 1 installs this anywhere but a TPU VM |
| `mashbox[cpu]` | JAX for CPU | a machine that runs a CPU model: Track2Rig, which adds its own extra (`mashbox[cpu,track2rig]`) |
| `mashbox[tpu]` | JAX for TPU | a TPU VM: line 1 installs this there |
| `mashbox[tpu,hy-nomas]` | JAX for TPU, plus PyTorch and the other libraries the Hy-Nomas pipeline uses | a TPU VM that runs Hy-Nomas: `mashbox tpu up hy-nomas` installs it on its chip |

Each one needs Python 3.12 or newer. Line 1 installs with [uv](https://docs.astral.sh/uv/), which brings its own Python 3.12, since a Cloud TPU VM's system Python is 3.10. On a TPU VM without line 1: `python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"`. For Hy-Nomas with line 1, name the extras:

```bash
curl -fsSL https://mashbox.gg/install.sh | MASHBOX_EXTRAS=tpu,hy-nomas sh
```

`mashbox run` fetches the compiled TPU kernels a model uses by itself, so there is nothing to install or pick. They come from two places. The engine kernel package, used for GGUF files and Hugging Face models, is one file attached to a mashbox release on GitHub; `models.json` lists it under `engine_kernels`, with the chips and the JAX version it is built for. Each MASHBOX build that ships kernels has its own kernel package in its Hugging Face repo, under `kernels/`. `mashbox run` checks every file against its sha256 in `models.json` before it uses it. When no package fits the chip and the JAX version, the model runs without one, on plain JAX, which is slower, and the log says so.
<!-- install:end -->

## What line 1 changes on your machine

- It installs nothing with sudo and never edits your shell profiles or your system Python.
- If you already have [uv](https://docs.astral.sh/uv/), it uses yours. If not, it puts uv (and uvx) in `~/.local/bin`.
- It installs `mashbox` as a uv tool, in its own folder with Python 3.12, and puts the `mashbox` command in `~/.local/bin`.
- It never overwrites a `mashbox` command that another tool put there: it stops and says why.

## What line 2 does

1. Prints the price per hour of the chip and its max lifetime, and asks before it creates anything:

   ```text
   Qwen3.8-27B on a new TPU v6e-1: $2.70 per hour while it exists (estimate at Google's on-demand list price per chip-hour, US regions; Google bills the real price).
   Max lifetime: 8 h after the chip starts, then it deletes itself (--max-hours 0 turns this off).
   Create mashbox-qwen3-8-27b now? [y/N]
   ```

2. Gets a chip. When a zone has no free chip it tries the next one, then asks Google for one with queued requests and prints how long it has waited every minute. The wait is Google's capacity: see [the FAQ](faq.md#how-long-does-it-take-to-get-a-chip).
3. Installs `mashbox` on the chip with the same install.sh and starts the server. The first start downloads the 26.94 GB weights.
4. Opens an SSH tunnel to the chip's `127.0.0.1` (nothing is open to the internet) and prints what to run next:

   ```text
   Qwen3.8-27B is running on mashbox-qwen3-8-27b (TPU v6e-1, us-east5-b).
     Local URL     http://127.0.0.1:8000/v1   (SSH tunnel to the chip's 127.0.0.1; nothing is open to the internet)
     Price         $2.70 per hour while the chip exists   estimate at Google's on-demand list price per chip-hour, US regions; Google bills the real price
     Chat          curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
                     -d '{"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
     Take it down  mashbox tpu down mashbox-qwen3-8-27b
   This terminal holds the tunnel. Ctrl-C closes the tunnel only; the chip keeps running and billing.
   ```

If port 8000 is taken on your machine, it uses the next free port and says so. Use the URL it prints.

## Chat with it

In a second terminal, while the first one holds the tunnel:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

It answers, for example, `Hello! How can I help you today?`.

## Use it from a coding agent

The first server has the default context. For Claude Code, Codex or Pi, restart it with the agents settings. This keeps the chip:

```bash
mashbox tpu up qwen3-8-27b --agents --restart
```

Then set your agent's base URL as [Use with coding agents](agents-on-tpu.md) shows.

## Run line 2 again

- A server already running on the chip is kept when it is the same `mashbox` and the same model and settings; otherwise `up` starts it again. `--agents` puts the model's coding-agent settings on the server's command line, so adding it or leaving it out starts the server again. `--restart` always starts it again.
- `up` uses and deletes only chips and queued requests that `mashbox` made (they carry the `mashbox` label). One of the same name without the label is left as it is, and `up` stops and says so.
- The local port opens only after the chip's server refuses a request sent as if from another web site, so a page open in your browser cannot use a server that lacks that check.

## See what is running, and take it down

```bash
mashbox tpu status
mashbox tpu down mashbox-qwen3-8-27b
```

`status` lists every mashbox chip in your project with its price per hour, its cost so far and when it deletes itself. `down` deletes the chip and prints what it cost. If you forget, the chip deletes itself at its max lifetime, 8 hours unless you change it.

## How long it takes

Measured on a TPU v6e-1. The wait for a chip is Google's and is counted apart.

| Step | Time |
|---|---|
| Line 1 on a MacBook (macOS 26.5.1, arm64), no uv | 6.8 s |
| Line 1 on Linux (x86_64), no uv | 1.1 s |
| Line 2 to the price and the question | 4.1 s |
| Wait for a chip (Google's capacity) | 8 to 26 min |
| Install on the chip | 35 s |
| Start to ready once the weights are on the chip, with the shipped compile cache | 69.6 s |
| First chat answer | 8.0 s |
| `down`, until the chip is gone | 99 s |

From create to delete the chip bills for about 17 minutes: about $0.76 at Google's on-demand list price.

## Environment variables

| Variable | Where it is read | What it does |
|---|---|---|
| `MASHBOX_VERSION` | line 1 (install.sh) | installs that version of `mashbox` instead of the newest |
| `MASHBOX_EXTRAS` | line 1 (install.sh) | installs these extras instead of the machine's default (`tpu` on a TPU VM, none elsewhere), for example `tpu,hy-nomas` |
| `MASHBOX_WHEEL` | your machine, by `mashbox tpu up` | installs this wheel file (or an http(s) URL of one) on the chip instead of the release from PyPI; the chip checks its sha256 before it installs |
| `MASHBOX_AOT` | the machine that runs `mashbox run` | `0` turns off the saved programs ([FAQ](faq.md#why-is-the-first-start-slow-and-later-starts-fast)) |
| `MASHBOX_STALL_S` | the machine that runs `mashbox run` | seconds a response may wait for a client that reads nothing before the server ends it (default 30) |

## Next

- Something went wrong: [Frequently asked questions](faq.md).
- Every option of `mashbox run`: [Command line](cli.md).
