# Quickstart

Run Qwen3.8-27B on one Google Cloud TPU v6e chip with three commands from your own machine: create the chip, install and run, delete the chip.

You need the Google Cloud CLI signed in to a project with billing and the TPU API turned on. [Run on Google Cloud TPU](google-cloud.md) steps 1 to 5 do that from a new Google account.

```bash
gcloud compute tpus tpu-vm create mashbox-1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]" && ~/.local/bin/mashbox run qwen3-8-27b --prompt "Write a haiku about the sea."'
gcloud compute tpus tpu-vm delete mashbox-1 --zone=us-east5-b --quiet
```

The second line installs [uv](https://docs.astral.sh/uv/) and `mashbox` with Python 3.12 (the chip's own Python is 3.10, too old for JAX on TPU), downloads the 26.94 GB weights file, checks it and prints the answer. On a new TPU v6e-1 the answer was:

```text
Salt spray kisses stone,
Blue waves roll in endless rhythm,
Sun sets on the tide.
```

> [!NOTE]
> The chip bills from create to delete: $2.70 an hour for one TPU v6e chip at Google's on-demand list price. On a new TPU v6e-1 the server's first start took 138 s, most of it the weights download; later starts took 6.6 s.

> [!TIP]
> If the create command says the zone has no capacity, use another zone in all three lines. Zones to try for a TPU v6e-1: us-east5-b and europe-west4-a. More in the [FAQ](faq.md#the-create-command-says-there-is-no-capacity-in-the-zone).

## Next

- Serve it with an OpenAI-compatible API: [Qwen3.8-27B, Serve it](models/qwen3-8-27b.md#serve-it).
- Use it from Claude Code, Codex or Pi: [Use with coding agents](agents-on-tpu.md).
- Something went wrong: [Frequently asked questions](faq.md).
