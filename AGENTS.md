# AGENTS.md

For coding agents that work on this repo, or that run its models for a person.

## What this repo is

`mashbox` is the MASHBOX Labs package for running open models faster and cheaper; the models in it run on Google Cloud TPU. Each model port is plain JAX in `src/mashbox/models/<id>/`. The list of models, with the chips each one passed on, is `models.json`. [OWNERS.md](OWNERS.md) says which folders are the shared core and which belong to one model.

## Rules

1. Ask the person before anything that costs money: creating a chip, picking its type, keeping it running. Tell them the hourly price first ([docs/google-cloud.md](docs/google-cloud.md), "What it costs").
2. Create one chip of the smallest type the model's row lists.
3. When the run ends or fails, delete the chip. Then show the person the output of `gcloud compute tpus tpu-vm list --zone=<zone>` without it.
4. Never write passwords, tokens or keys into files, commits or command lines. `gcloud auth login` opens a browser and the person signs in.
5. Never make anything public: no public repos, buckets or open ports. Servers bind 127.0.0.1 on the chip; reach them with an SSH tunnel (`-- -L <port>:127.0.0.1:<port>`), never by opening a firewall port.
6. Turning on billing and asking Google for TPU quota happen in the Google Cloud console. Give the person the link and wait for them.

## Run a model for a person

- **Qwen3.8-27B** runs on one TPU v6e-1 ($2.70 an hour on demand); it does not fit on v5e. Install on the chip: `python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"` (the chip's system Python is 3.10; JAX for TPU needs 3.12, which uv brings). One answer: `~/.local/bin/mashbox run qwen3-8-27b --prompt "..."`. A server: [docs/models/qwen3-8-27b.md](docs/models/qwen3-8-27b.md).
- **Hy-Nomas** is coming soon: its weights are not public yet, so do not try to run it.
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
| `its runtime is not in mashbox <version> yet` | that model's port is not in the installed version; its page under `docs/models/` says what is ready |
| out of memory | use the chip type the model's row lists, with nothing else running on it; for Qwen3.8-27B keep the default `--ctx 2048 --slots 8` |
