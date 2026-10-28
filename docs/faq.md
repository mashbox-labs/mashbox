# Frequently asked questions

Problems you can meet when you run these models on a new Google Cloud TPU chip, why they happen, and what to do. The [Qwen3.8-27B guide](models/qwen3-8-27b.md#troubleshooting) also has a Troubleshooting table with each error message.

## Getting a chip

### The create command says there is no capacity in the zone

**What you see:** `There is no more capacity in the zone` or `Insufficient capacity` when you create the chip.

**Why:** Google has no free chip of that type in that zone at the moment. Capacity changes from hour to hour, and `mashbox` cannot change it.

**What to do:** Run the same create command in another zone, then use that zone in every later command. Or queue a request in that zone: it costs nothing while it waits, and the chip bills from the moment Google grants it. Zones to try first are in [step 6 of the setup guide](google-cloud.md#6-create-one-chip).

### How long does it take to get a chip?

Usually minutes, sometimes an hour or more. The wait depends on Google's capacity. For example:

| Chip | How | Wait |
|---|---|---|
| TPU v6e-1 | create command | the command returned after about 5 minutes |
| TPU v6e-1 | Google Cloud console | `READY` after 9 minutes |
| TPU v6e-1 | create command, zone out of capacity at first | created about 14 minutes later, `READY` 7 minutes after that |
| TPU v6e-4 | create command | no free chip in 5 zones for about an hour |
| TPU v6e-4 | queued request | granted after 10 to 14 minutes |

A queued request can go from `PROVISIONING` to `SUSPENDING` on Google's side without being granted.

**What to do:** For a TPU v6e-4, queue a request in a few zones at once and delete the others once one is `ACTIVE`. A request that is never granted costs nothing.

### The create command says I do not have permission for the accelerator type

**What you see:** `User does not have permission to submit requests for accelerator type "v6e-4"` (or `"v6e-1"`).

**Why:** Your project has no on-demand TPU v6e quota in that zone.

**What to do:** Use a zone where your project has quota, or ask for quota in that zone ([step 5 of the setup guide](google-cloud.md#5-check-your-tpu-quota)). A TPU v6e-4 needs 4 of "TPU v6e cores per project per zone".

## Installing

### pip install fails with No matching distribution found for jax[tpu]==0.11.2

**What you see:** `ERROR: No matching distribution found for jax[tpu]==0.11.2`, about 2 seconds after `pip install "mashbox[tpu]"` on the chip.

**Why:** The TPU VM image (Ubuntu 22.04) comes with Python 3.10, and JAX 0.11.2 for TPU needs Python 3.12. It is the same on TPU v6e-1, v6e-4 and v5e-1.

**What to do:** Install with [uv](https://docs.astral.sh/uv/), which brings its own Python 3.12 and puts `mashbox` in `~/.local/bin`:

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
```

### pip install uv fails with externally-managed-environment

**What you see:** `error: externally-managed-environment` right after `python3 -m pip install uv` on your own computer.

**Why:** That line is for the TPU VM, whose image (Ubuntu 22.04) lets pip install into the system Python. Ubuntu 24.04 and newer refuse it (PEP 668).

**What to do:** Install uv with its own installer, then `mashbox` with uv. On your own computer the plain `mashbox` package is enough for `mashbox tpu`; `mashbox[tpu]` is for the chip.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv tool install --python 3.12 mashbox
```

### mashbox: command not found

**What you see:** `mashbox: command not found` right after the install, with a warning from pip and uv that `~/.local/bin` is not on your `PATH`.

**Why:** uv puts the command in `~/.local/bin`, which is not on your `PATH` in a `gcloud ... --command` session or until you log in again.

**What to do:** Use the full path `~/.local/bin/mashbox`, as every command in these docs does.

## Downloading and disk

### The download stops with 429 Too Many Requests

**What you see:** `429 Too Many Requests` from `cdn.hf.co` during the weights download, then a traceback.

**Why:** Hugging Face's Xet transfer can limit a download and stop it partway.

**What to do:** `mashbox` downloads over plain HTTPS by default (it sets `HF_HUB_DISABLE_XET=1` unless you set it yourself), which ran at 200 to 230 MB/s on a TPU v6e-1. If you set `HF_HUB_DISABLE_XET=0`, remove it, then run the command again.

### How much disk do I need?

| Model | Download | Disk on a new VM |
|---|---|---|
| Qwen3.8-27B | one 26.94 GB weights file, plus a 47 MB compile cache on TPU v6e | TPU v6e-1: 97 GB disk, plenty of room |

**What to do:** If your disk is smaller or fuller, set `HF_HUB_CACHE` to a folder with more room before you download. On a TPU v6e-4 VM, `/dev/shm` holds 355 GB in RAM and is emptied when the VM stops.

## Starting and stopping

### Why is the first start slow, and later starts fast?

The first start downloads the weights, checks each file's sha256 and compiles the TPU programs. Compiled programs are kept in `~/.cache/mashbox/xla`, so later starts skip that. On a new chip:

| Model | Chip | First start | Later starts |
|---|---|---|---|
| Qwen3.8-27B | TPU v6e-1 | 138 s to ready, most of it the download; 166.6 s without the shipped compile cache | 6.6 s to ready (median of 3) |

On TPU v6e with JAX 0.11.2, Qwen3.8-27B also downloads a 47 MB compile cache, so its first start loads compiled programs instead of compiling them. A prompt length the server has not seen before compiles once: 5.8 s extra on that one request on a TPU v6e-1.

### How do I stop a server, and what if the TPU will not start again?

**What you see:** `Unable to initialize backend 'tpu': INTERNAL: TPU initialization failed: ... START_SESSION failed` when `mashbox` starts, after `TPU backend initialization is taking more than 60.0 seconds`.

**Why:** On a TPU v6e-4 this can follow a server stopped with `kill` (SIGTERM), or a new start right after the previous run exited, and it can last through a reboot. The cause may be on Google's side of the TPU.

**What to do:** Stop servers with Ctrl-C in the terminal where they run. Wait 10 s after the prompt returns before the next start. Run servers in the foreground or in tmux, not with `&`: a background process ignores Ctrl-C. If it happens anyway, delete the chip and create a new one.

### It says the TPU is already in use

**What you see:** `The TPU is already in use by another process probably owned by another user`

**Why:** One process at a time can use the chip: a server still running in another SSH session, or a lock file left by a run that crashed.

**What to do:** Stop the other process with Ctrl-C in its session, then run again. If no process is left, the message names the lock file to remove.

## Chips, models and cost

### Which chip does each model need? Can I use TPU v5e?

| Model | TPU v6e-1 | TPU v5e |
|---|---|---|
| Qwen3.8-27B | yes: the 26.94 GB file (8.01 bits per weight) fits one chip | no: the weights alone are about 27 GB and a v5e chip has 16 GB |

A TPU v6e chip has 32 GB of memory (JAX reports 31.24 GiB); a TPU v5e chip has 16 GB (15.75 GiB). [TPU v6e and v5e](chips.md) compares the two.

### What does it cost, and when does billing stop?

A chip bills from create to delete, whether anything runs on it or not. Google's on-demand list prices: TPU v6e-1 $2.70 an hour, TPU v6e-4 $10.80 an hour, TPU v5e-1 $1.20 an hour. A queued request costs nothing while it waits. Deleting a chip takes about 2.5 minutes.

**What to do:** Delete the chip when you are done and check that the list is empty ([step 8 of the setup guide](google-cloud.md#8-delete-the-chip)). If you queued the chip, delete the queued request: that deletes the chip with it.
