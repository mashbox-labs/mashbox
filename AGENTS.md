# AGENTS.md

For coding agents that work on this repo, or that run its models for a person.

## What this repo is

`mashbox` is the MASHBOX Labs package for running open models faster and cheaper; the models in it run on Google Cloud TPU. Each model port is plain JAX in `src/mashbox/models/<id>/`. The list of models, with the chips each one runs on, is `models.json`. [OWNERS.md](OWNERS.md) says which folders are the shared core and which belong to one model.

## Rules

1. Ask the person before anything that costs money: creating a chip, picking its type, keeping it running. Tell them the hourly price first ([docs/google-cloud.md](docs/google-cloud.md), "What it costs").
2. Create one chip of the smallest type the model's row lists as `passing`: those are the types `mashbox tpu up` creates. A chips row's `note`, when it has one, is what to tell the person about that chip.
3. When the run ends or fails, delete the chip. Then show the person the output of `gcloud compute tpus tpu-vm list --zone=<zone>` without it.
4. Never write passwords, tokens or keys into files, commits or command lines. `gcloud auth login` opens a browser and the person signs in.
5. Never make anything public: no public repos, buckets or open ports. Servers bind 127.0.0.1 on the chip; reach them with an SSH tunnel (`-- -L <port>:127.0.0.1:<port>`), never by opening a firewall port.
6. Turning on billing and asking Google for TPU quota happen in the Google Cloud console. Give the person the link and wait for them.

## Run a model for a person

- **Qwen3.8-27B** runs on one TPU v6e-1 ($2.70 an hour on demand). TPU v5e status: Supported: a v5e-4 (4 chips), split across them automatically; one v5e chip is too small: the weights alone are about 27 GB and a v5e chip has 16 GB. 76.73 tokens/s for 1 request, 470.4 tokens/s for 8. Install on the chip: `python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"` (the chip's system Python is 3.10; JAX for TPU needs 3.12, which uv brings). One answer: `~/.local/bin/mashbox run qwen3-8-27b --prompt "..."`. A server: [docs/models/qwen3-8-27b.md](docs/models/qwen3-8-27b.md).
- **Hy-Nomas** (a video world model in the browser) runs on a TPU v6e-4 (a TPU v6e slice of 4 chips, $10.80 an hour on demand in US regions). One chip: the full pipeline fits on a single TPU v6e chip (32 GB), 2.46 s of compute per second of video, main model quantized. This version of `mashbox run` starts on a v6e-4. Install on the chip with the `install` line of its `models.json` row (it adds the `hy-nomas` extra and the CPU build of PyTorch), then `~/.local/bin/mashbox download hy-nomas` (the weights; the page lists the disk it needs) and `~/.local/bin/mashbox run hy-nomas --ui`. Connect with `-- -L 8790:127.0.0.1:8790` on the SSH line and give the person http://127.0.0.1:8790. Stop the app with Ctrl-C before you delete the chip. Every step: [docs/models/hy-nomas.md](docs/models/hy-nomas.md).
- Run only the models in the README's Models table.
- **Coding agents as clients:** to point Claude Code, Codex or Pi at a model you serve, follow [docs/agents-on-tpu.md](docs/agents-on-tpu.md).

1. Read `models.json`. Pick the model, note its chips and its `run` command.
2. Follow [docs/google-cloud.md](docs/google-cloud.md) steps 1 to 8 in order. Check each step's output before the next one.
3. Copy any output files to the person's machine (step 7).
4. Delete the chip (step 8) and show the empty list.

## Work on the code

- Install for development: `python3 -m pip install -e ".[dev]"`
- CPU tests: `python3 -m pytest tests/cpu`. CI runs them on every pull request.
- TPU tests live in `tests/tpu/`. Maintainers run them on real chips before each release. Say in the pull request if your change needs one.
- After editing `models.json`, copy it to `src/mashbox/models.json`, then run `python3 tools/hf_github_sync_check.py render --repo . --write` and commit the files it rewrites.
- The kernel source is not in this repo. Each model's compiled kernels come as one bundle that `mashbox run` downloads from the model's Hugging Face repo and checks against the sha256 in `models.json`. Do not patch or decompile it; report kernel problems as issues. `--no-kernels` runs without it where the model supports plain JAX.
- Sign off every commit: `git commit -s` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## When something fails

| What you see | What to do |
|---|---|
| an error that mentions quota | the person requests TPU quota for that zone in the console ([docs/google-cloud.md](docs/google-cloud.md), step 5) |
| no capacity in the zone | try the next zone in the step 6 table |
| the TPU API is not enabled | run step 4 |
| `User does not have permission to submit requests for accelerator type` | the project has no on-demand quota for that chip in that zone: use another zone from the step 6 table |
| `No matching distribution found for jax[tpu]==0.11.2` | plain pip ran on the chip's Python 3.10: install with the uv line above |
| `mashbox: command not found` | use the full path `~/.local/bin/mashbox` |
| `<model> is not in mashbox <version>` | that model is not in the installed version: pick one from the README's Models table |
| out of memory | use the chip type the model's row lists, with nothing else running on it; for Qwen3.8-27B keep the default `--ctx 2048 --slots 8` |
