# Frequently asked questions

Problems you can meet when you run these models on a new Google Cloud TPU chip, why they happen, and what to do. The [Qwen3.8-27B guide](models/qwen3-8-27b.md#troubleshooting) also has a Troubleshooting table with each error message.

## Getting a chip

### The create command says there is no capacity in the zone

**What you see:** `There is no more capacity in the zone` or `Insufficient capacity` when you create the chip.

**Why:** Google has no free chip of that type in that zone at the moment. Capacity changes from hour to hour, any zone can run out, and `mashbox` cannot change it.

**What to do:** Run the same create command in another zone, then use that zone in every later command. Or [queue a request](google-cloud.md#if-no-zone-has-a-chip-queue-a-request) in that zone with `gcloud compute tpus queued-resources create`: it costs nothing while it waits, and the chip bills from the moment Google grants it. The zones in the setup guide's table are a place to start, not a full list.

### How long does it take to get a chip?

Usually minutes; when your zones have no capacity it can take an hour or more. The wait depends on Google's capacity, not on `mashbox`.

- **TPU v6e-1:** from the create command or a queued request to a `READY` chip can take 5 to 27 minutes.
- **TPU v6e-4:** a plain create command can find no free chip in any of several zones for an hour; a queued request can be granted in 10 to 14 minutes.

A queued request can sit in `PROVISIONING` for several minutes, or go from `PROVISIONING` to `SUSPENDING` without being granted while another zone grants one.

**What to do:** When the create command finds no capacity, [queue a request](google-cloud.md#if-no-zone-has-a-chip-queue-a-request) rather than trying zones one by one. For a TPU v6e-4, queue a request in a few zones at once and delete the others once one is `ACTIVE`. A request that is never granted costs nothing.

### The create command says I do not have permission for the accelerator type

**What you see:** `User does not have permission to submit requests for accelerator type "v6e-4"` (or `"v6e-1"`).

**Why:** Your project has no on-demand TPU v6e quota in that zone. TPU quota varies by project and by zone.

**What to do:** Check your quota in the console ([step 5 of the setup guide](google-cloud.md#5-check-your-tpu-quota)), or from the terminal with your project ID in place of `PROJECT_ID`:

```bash
gcloud services enable cloudquotas.googleapis.com --project=PROJECT_ID
gcloud beta quotas info describe TPUV6EPerProjectPerZoneForTPUAPI --service=tpu.googleapis.com --project=PROJECT_ID --flatten=dimensionsInfos --format="table(dimensionsInfos.dimensions.zone,dimensionsInfos.details.value)"
```

`VALUE` is the number of on-demand TPU v6e cores your project can use in that zone. Use a zone where your project has quota, or ask for quota in that zone (step 5). A TPU v6e-4 needs 4 of "TPU v6e cores per project per zone".

## Installing

### pip install fails with No matching distribution found for jax[tpu]==0.11.2

**What you see:** `ERROR: No matching distribution found for jax[tpu]==0.11.2`, about 2 seconds after `pip install "mashbox[tpu]"` on the chip.

**Why:** The TPU VM image (Ubuntu 22.04) comes with Python 3.10, and JAX 0.11.2 for TPU needs Python 3.12. It is the same on TPU v6e-1, v6e-4 and v5e.

**What to do:** Install with [uv](https://docs.astral.sh/uv/), which brings its own Python 3.12 and puts `mashbox` in `~/.local/bin`:

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
```

### mashbox: command not found

**What you see:** `mashbox: command not found` right after the install, with a warning from uv (and sometimes from pip) that `~/.local/bin` is not on your `PATH`.

**Why:** uv puts the command in `~/.local/bin`, which is not on your `PATH` in a `gcloud ... --command` session or until you log in again.

**What to do:** Use the full path `~/.local/bin/mashbox`, as every command in these docs does.

## Downloading and disk

### The download stops with 429 Too Many Requests

**What you see:** `429 Too Many Requests` from `cdn.hf.co` during the weights download, then a traceback.

**Why:** Hugging Face's Xet transfer can limit a download and stop it partway, sometimes a minute or two in.

**What to do:** `mashbox` downloads over plain HTTPS by default (it sets `HF_HUB_DISABLE_XET=1` unless you set it yourself), at about 200 to 230 MB/s on a TPU v6e-1. If you set `HF_HUB_DISABLE_XET=0`, remove it, then run the command again.

### How much disk do I need?

| Model | Download | Disk on a new VM |
|---|---|---|
| Qwen3.8-27B | one 26.94 GB weights file, plus an 890 MB compile cache on TPU v6e | TPU v6e-1: 97 GB disk, plenty of room |

**What to do:** If your disk is smaller or fuller, set `HF_HUB_CACHE` to a folder with more room before you download. On a TPU v6e-4 VM, `/dev/shm` holds 355 GB in RAM and is emptied when the VM stops.

## Starting and stopping

### Why is the first start slow, and later starts fast?

The first start downloads the weights, checks each file's sha256 and compiles the TPU programs. Compiled programs are kept in `~/.cache/mashbox/xla`, so later starts skip that. On a new chip:

| Model | Chip | First start | Later starts |
|---|---|---|---|
| Qwen3.8-27B | TPU v6e-1 | the weights download, then 69.6 s to ready with the shipped compile cache (326.6 s without it) | 69.6 s to ready |

The first start depends mostly on your download speed: on a new TPU v6e-1 the download and checks alone can take about 279.2 s. To get the download out of the way, run `~/.local/bin/mashbox download qwen3-8-27b` on the chip first: it fetches and checks the files without loading them ([Command line](cli.md#mashbox-download)). `--no-warm` makes the server ready sooner and compiles on the first request instead. If you find a way to make starts faster, pull requests are welcome on [GitHub](https://github.com/mashbox-labs/mashbox).

On TPU v6e with JAX 0.11.2, Qwen3.8-27B also downloads an 890 MB compile cache, so its first start loads compiled programs instead of compiling them.

### How do I stop a server, and what if the TPU will not start again?

**What you see:** `Unable to initialize backend 'tpu': INTERNAL: TPU initialization failed: ... START_SESSION failed` when `mashbox` starts, after `TPU backend initialization is taking more than 60.0 seconds`.

**Why:** The chips of a host could not be linked into one TPU slice. This can follow a server stopped with `kill` (SIGTERM), or a new start less than a second after the previous run exited, and it can last through a reboot.

**What to do:** Stop servers with Ctrl-C in the terminal where they run. If the serve command runs in the background or from a script, end that command on your machine first: while it runs, gcloud retries it when the server stops and starts the server again. Then stop the server over SSH: `gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='pkill -INT -f "^/home/.*/mashbox run"'` (your chip name and zone; `-INT` is the Ctrl-C signal). Wait 10 s after the prompt returns before the next start. Run servers in the foreground or in tmux, not with `&`: a background process ignores Ctrl-C. If the TPU still does not start, delete the chip and create a new one.

### It says the TPU is already in use

**What you see:** `The TPU is already in use by another process probably owned by another user`

**Why:** One process at a time can use the chip: a server still running in another SSH session, or a lock file left by a run that crashed.

**What to do:** Stop the other process with Ctrl-C in its session, then run again. If no process is left, the message names the lock file to remove.

## Chips, models and cost

### Which chip does each model need? Can I use TPU v5e?

<!-- chips:start -->
| Model | Runs on | TPU v5e | TPU v6e | TPU v5p slice (v5p-8) | TPU v6e slice (v6e-4) |
|---|---|---|---|---|---|
| Hy-Nomas | TPU v6e-4 | Not supported: runs only on a v6e-4 (a TPU v6e slice of 4 chips) | One chip: the full pipeline fits on a single TPU v6e chip (32 GB), 2.46 s of compute per second of video (measured 26 October 2026). One-chip mode in `mashbox run` ships in the next release; this version starts on a v6e-4 | Not supported: runs only on a v6e-4 (a TPU v6e slice of 4 chips) | Supported: 4 chips |
| Qwen3.8-27B | TPU v6e-1 | Supported: a v5e-4 (4 chips), split across them automatically; one v5e chip is too small: the weights alone are about 27 GB and a v5e chip has 16 GB | Supported: one chip (v6e-1) | Not supported | Supported: 4 chips, split across them automatically |
<!-- chips:end -->

A TPU v6e chip has 32 GB of memory (JAX reports 31.24 GiB); a TPU v5e chip has 16 GB (15.75 GiB). [TPU v6e and v5e](chips.md) compares the two.

### What does it cost, and when does billing stop?

A chip bills from create to delete, whether anything runs on it or not. Google's on-demand list prices in US regions: TPU v6e-1 $2.70 an hour, TPU v6e-4 $10.80 an hour, TPU v5e-1 $1.20 an hour; the price differs by region ([TPU pricing](https://cloud.google.com/tpu/pricing)). A queued request costs nothing while it waits. Deleting a chip takes about 2.5 minutes.

**What to do:** Delete the chip when you are done and check that the list is empty ([step 8 of the setup guide](google-cloud.md#8-delete-the-chip)). If you queued the chip, delete the queued request: `gcloud compute tpus queued-resources delete REQUEST_NAME --zone=ZONE --force --quiet` deletes the chip with it. Then check that `gcloud compute tpus queued-resources list --zone=ZONE` no longer shows it.

## Coding agents

### Claude Code reads the whole prompt again on every request

**What you see:** `cached_tokens` near 0 on every Claude Code request in the server's log.

**Why:** Claude Code puts a line for Anthropic's billing at the top of its system prompt. Before Claude Code 2.1.181 that line held a token that changed on every request to a server other than Anthropic's, so no request could reuse the one before it (Claude Code's [environment variables](https://code.claude.com/docs/en/env-vars) page; [llama.cpp issue 19494](https://github.com/ggml-org/llama.cpp/issues/19494) shows the same on llama-server). On 2.1.312 the line stays the same within a session and from one new session to the next.

**What to do:** Set `CLAUDE_CODE_ATTRIBUTION_HEADER` to `"0"`, as in the settings file on the [Claude Code](agents/claude-code.md) page. Also check that no second agent uses the same chip.

### Claude Code sends a long request before each shell command

**What you see:** A request with `"tools": 0` and about 32,700 prompt tokens in the server's log, right before Claude Code runs a shell command; the first one takes about 30 s.

**Why:** Claude Code 2.1.283 and later start in auto mode, where a second model checks each shell command before it runs ([permission modes](https://code.claude.com/docs/en/permission-modes)). With the settings on the [Claude Code](agents/claude-code.md) page that model is the one you serve. A check can be about 32,705 tokens, close to the 32,767 limit: 31 s to the first token the first time, 3.1 s the next time, when 30,892 of its tokens come back from host memory.

**What to do:** Press Shift+Tab once to switch from auto mode to asking before each command.

### Codex sends a short extra request after each of my messages

It asks the model for a title for the conversation (about 4,700 tokens, no tools). Codex keeps the title only if the answer is JSON that starts with `{` ([Codex source](https://github.com/openai/codex/blob/26ad281fd1ca712e4f0c7920a6273cf18c961b8a/codex-rs/tui/src/app/thread_title.rs)). The `mashbox` server does not apply the JSON format setting on the Responses API (llama-server does not either: [issue 21922](https://github.com/ggml-org/llama.cpp/issues/21922)), so Codex asks again after your next message. The server keeps your conversation in host memory while the title request runs and brings it back afterwards, so it is not read again. `codex exec` sends no title request.

### Does the prompt cache change the answers?

No. The server only reuses work it already did for the same tokens, and it can resume a prompt only at a saved point (the end of the previous prompt, or the start of a user message), never in the middle. With temperature 0 and a fixed seed, a 17,000-token conversation that comes back from host memory gives the same text, token for token, as one that never left the chip. `--cache-ram 0` turns the host-memory cache off.

### Can two agents share one chip?

Yes, but they take turns: the server answers one request at a time. The conversation that is not on the chip waits in host memory and comes back when its agent sends the next request: with Codex and Pi on one TPU v6e-1 the first token comes 0.4 to 0.9 s after a switch, plus any wait for the other agent's request. For two agents working all the time, use one chip each.

### Can I give the agent a longer context on one v6e chip?

Not on one TPU v6e chip. `--ctx 65536 --slots 1` runs out of chip memory by 4 MB when the server compiles it, and two 32,768-token slots needed 33.65 GB where 31.24 GB is free. Keep `--ctx 32768 --slots 1` and the compaction settings on your agent's page.
