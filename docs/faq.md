<!-- Generated file: edit the FAQ source, not this page. -->
# Frequently asked questions

What we ran into while running these models on new Google Cloud TPU chips as first-time users on 28 October 2026, and what to do about each. Each model guide also has a Troubleshooting table with every error message we saw: [Qwen3.8-27B](models/qwen3-8-27b.md#troubleshooting), [Hy-Nomas](models/hy-nomas.md#troubleshooting).

## Getting a chip

### The create command says there is no capacity in the zone

**What you see:** `There is no more capacity in the zone` or `Insufficient capacity` when you create the chip.

**Why:** Google has no free chip of that type in that zone right now. It changes from hour to hour, and it is outside our control. On 28 October 2026 we saw it for TPU v6e-1 in us-east5-b, us-east5-a, us-south1-a and asia-northeast1-b, and for TPU v6e-4 in asia-northeast1-b, us-east5-a, us-east5-b, europe-west4-a and us-south1-a.

**What to do:** Run the same create command in another zone, then use that zone in every later command. Or queue a request in that zone: it costs nothing while it waits, and the chip bills from the moment Google grants it (the two lines are on the [Hy-Nomas page](models/hy-nomas.md#install)). Zones that gave us a chip on 28 October 2026: TPU v6e-1 in europe-west4-a and us-east5-b, TPU v6e-4 in asia-northeast1-b, europe-west4-a and us-east5-b (all through queued requests), TPU v5e-1 in us-south1-a.

### How long does it take to get a chip?

Usually minutes, sometimes an hour or more. The wait is Google's capacity, not something `mashbox` controls. What we measured on 28 October 2026:

| Chip | How | Wait |
|---|---|---|
| TPU v6e-1 | create command, europe-west4-a | the command returned after about 5 minutes |
| TPU v6e-1 | Google Cloud console, us-east5-b | READY after 9 minutes |
| TPU v6e-1 | create command, us-east5-b | no capacity at first, created about 14 minutes later, READY 7 minutes after that |
| TPU v6e-4 | create command | no free chip in 5 zones for about an hour |
| TPU v6e-4 | queued request | granted after about 14 minutes (asia-northeast1-b), 11.5 minutes (europe-west4-a) and 10 minutes (us-east5-b) |

Two of our queued v6e-4 requests went to PROVISIONING and then to SUSPENDING on Google's side before another zone granted one.

**What to do:** For a TPU v6e-4, queue a request in a few zones at once and delete the others once one is ACTIVE. Requests that were never granted cost us nothing.

### The create command says I do not have permission for the accelerator type

**What you see:** `User does not have permission to submit requests for accelerator type "v6e-4"` (or `"v6e-1"`).

**Why:** Your project has no on-demand TPU v6e quota in that zone. On 28 October 2026 we got this in us-east1-d and us-central1-b for both v6e-1 and v6e-4, and a quota limit of 0 in us-east5-c.

**What to do:** Use a zone where your project has quota, or ask for quota in that zone ([step 5 of the setup guide](google-cloud.md#5-check-your-tpu-quota)). A TPU v6e-4 needs 4 of "TPU v6e cores per project per zone". A new project we made on 28 October 2026 already had 16 on-demand v6e cores in us-east5-b.

## Installing

### pip install fails with No matching distribution found for jax[tpu]==0.11.2

**What you see:** `ERROR: No matching distribution found for jax[tpu]==0.11.2`, about 2 seconds after `pip install "mashbox[tpu]"` on the chip.

**Why:** The TPU VM image (Ubuntu 22.04) comes with Python 3.10, and JAX 0.11.2 for TPU needs Python 3.12. We hit this on TPU v6e-1, v6e-4 and v5e-1.

**What to do:** Install with [uv](https://docs.astral.sh/uv/), which brings its own Python 3.12 and puts `mashbox` in `~/.local/bin`:

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
```

Hy-Nomas is coming soon; its own install lines come with it.

### mashbox: command not found

**What you see:** `mashbox: command not found` right after the install, with a warning from pip and uv that `~/.local/bin` is not on your `PATH`.

**Why:** uv puts the command in `~/.local/bin`, which is not on your `PATH` in a `gcloud ... --command` session or until you log in again.

**What to do:** Use the full path `~/.local/bin/mashbox`, as every command in these docs does.

## Downloading and disk

### The download stops with 429 Too Many Requests

**What you see:** `429 Too Many Requests` from `cdn.hf.co` during the weights download, then a traceback.

**Why:** Hugging Face's Xet transfer limited the download. On 28 October 2026 it stopped two runs in a row on one TPU v6e-1, after 84 s and 135 s; another run on Xet the same morning finished at about 63 MB/s.

**What to do:** `mashbox` downloads over plain HTTPS by default (it sets `HF_HUB_DISABLE_XET=1` unless you set it yourself), which ran at 200 to 230 MB/s on a TPU v6e-1. If you set `HF_HUB_DISABLE_XET=0`, remove it, then run the command again.

### How much disk do I need?

| Model | Download | Disk on our new VM |
|---|---|---|
| Qwen3.8-27B | one 26.94 GB weights file, plus a 47 MB compile cache on TPU v6e | TPU v6e-1: 97 GB disk, plenty of room |
| Hy-Nomas | 87.4 GB: Tencent's main model 34.2 GB, LoRA adapter 6.1 GB, video decoder 31.6 GB, text encoder 15.5 GB | TPU v6e-4: default 100 GB boot disk, 3.9 GB left after the install and download |

The install itself (Python 3.12, JAX, the package) took 2.3 GB for Hy-Nomas.

**What to do:** If your disk is smaller or fuller, set `HF_HUB_CACHE` to a folder with more room before you download. On a TPU v6e-4 VM, `/dev/shm` holds 355 GB in RAM and is emptied when the VM stops.

## Starting and stopping

### Why is the first start slow, and later starts fast?

The first start downloads the weights, checks each file's sha256 and compiles the TPU programs. Compiled programs are kept in `~/.cache/mashbox/xla`, so later starts skip that. Measured on 28 October 2026:

| Model | Chip | First start | Later starts |
|---|---|---|---|
| Qwen3.8-27B | TPU v6e-1 | 138 s to ready, most of it the download; 166.6 s without the shipped compile cache | 6.6 s to ready (median of 3) |
| Hy-Nomas | TPU v6e-4 | 13 min 10 s to the first frame, about 12 minutes of it compiling | 1 min 58 s to the first frame |

On TPU v6e with JAX 0.11.2, Qwen3.8-27B also downloads a 47 MB compile cache, so its first start loads compiled programs instead of compiling them. A prompt length the server has not seen before compiles once: 5.8 s extra on that one request in our run.

### How do I stop a server, and what if the TPU will not start again?

**What you see:** `Unable to initialize backend 'tpu': INTERNAL: TPU initialization failed: ... START_SESSION failed` when `mashbox` starts, after `TPU backend initialization is taking more than 60.0 seconds`.

**Why:** We saw this twice on 28 October 2026, both on TPU v6e-4 with Hy-Nomas: once after a server was stopped with `kill` (SIGTERM), once when a new run started 0.84 s after the previous one exited. A start 2.84 s after a clean exit worked, and starts a minute or more apart always worked. A reboot did not clear it, and retries over 11 minutes all failed. The cause is likely on Google's side of the TPU; with two cases it is not proven.

**What to do:** Stop servers with Ctrl-C in the terminal where they run: the Hy-Nomas server exited 4.7 to 5.0 s after Ctrl-C on a TPU v6e-4. Wait 10 s after the prompt returns before the next start. Run servers in the foreground or in tmux, not with `&`: a background process ignores Ctrl-C. If it happens anyway, delete the chip and create a new one.

### It says the TPU is already in use

**What you see:** `The TPU is already in use by another process probably owned by another user`

**Why:** One process at a time can use the chip: a server still running in another SSH session, or a lock file left by a run that crashed.

**What to do:** Stop the other process with Ctrl-C in its session, then run again. If no process is left, the message names the lock file to remove.

### Why do some Hy-Nomas frames arrive late?

On a TPU v6e-4 the default quality preset takes 1.045 s of compute per second of video, slightly slower than real time, so 2 to 5% of frames reach the browser late (28 October 2026). The live preset costs the same. The MASHBOX Fast decoder (the Decoder switch in the app, or `--decoder fast`) took 0.689 s per second of video with no late frames in 8 windows, with a softer picture.

## Chips, models and cost

### Which chip does each model need? Can I use TPU v5e?

| Model | TPU v6e-1 | TPU v6e-4 | TPU v5e |
|---|---|---|---|
| Qwen3.8-27B | yes: the 26.94 GB file (8.01 bits per weight) fits one chip | not tested | no: the weights alone are about 27 GB and a v5e chip has 16 GB |
| Hy-Nomas | not supported yet | yes | no: its kernels run on v5e, but the model as laid out needs more than 16 GB per chip |

A TPU v6e chip has 32 GB of memory (JAX reports 31.24 GiB); a TPU v5e chip has 16 GB (15.75 GiB). [TPU v6e and v5e](chips.md) compares the two.

### What does it cost, and when does billing stop?

A chip bills from create to delete, whether anything runs on it or not. Google's on-demand list prices, read 19 October 2026: TPU v6e-1 $2.70 an hour, TPU v6e-4 $10.80 an hour, TPU v5e-1 $1.20 an hour. A queued request costs nothing while it waits. Deleting a chip took about 2.5 minutes in our runs.

**What to do:** Delete the chip when you are done and check that the list is empty ([step 8 of the setup guide](google-cloud.md#8-delete-the-chip)). If you queued the chip, delete the queued request: that deletes the chip with it.

## Before release

### Do I need a Hugging Face token?

Not after release: the mashbox-labs model repos will be public, and `mashbox` downloads without a token. Until then they are private, so early testers need a read token in `~/.cache/huggingface/token` (where `hf auth login` puts it).
