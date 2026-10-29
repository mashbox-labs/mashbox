# Use with coding agents

Claude Code, Codex and Pi can use a model that `mashbox` serves on your TPU as their LLM. Keep the agent you already have: nothing on this page reinstalls it or changes its normal settings. Claude Code reads one extra settings file that you name when you start it, Codex gets an added provider and profile, and Pi gets a profile folder of its own. Started the usual way, each agent works as before.

The `mashbox` server speaks the APIs these agents use: Anthropic Messages (Claude Code), OpenAI Responses (Codex) and OpenAI chat completions with tool calls (Pi). Every LLM that `mashbox run` serves gets them. The example on this page is [Qwen3.8-27B](models/qwen3-8-27b.md).

## What it costs

The chip bills from create to delete: $2.70 an hour for one TPU v6e chip at Google's on-demand list price. The server does not charge per token and does not check keys.

## Before you start

- Claude Code, Codex CLI or Pi, already installed on your machine. The settings below are for Claude Code 2.1.312, Codex CLI 0.165.0 and Pi 1.4.0.
- For the one-command way in step 1: `mashbox` on your machine and the Google Cloud CLI signed in, as in [Install](install.md).
- For the way by hand: a TPU v6e-1 with `mashbox` on it, from the [Quickstart](quickstart.md) or the [Qwen3.8-27B guide](models/qwen3-8-27b.md). There the chip is `mashbox-1` in zone `us-east5-b`: use your chip's name and zone. Port 8000 must be free on your machine.

## 1. Start the server and the tunnel

Both ways below start the server with the same settings. `--ctx 32768 --slots 1` gives one conversation of up to 32,768 tokens, which is what fits next to the 26.94 GB of weights on a v6e chip (two slots at this context do not fit in chip memory). Each agent below is set to compact its conversation before it gets there.

### With one command

From your machine:

```bash
mashbox tpu up qwen3-8-27b --agents --restart
```

It gets a TPU v6e-1 named `mashbox-qwen3-8-27b` (it prints the price per hour and asks before it creates one; `--yes` skips the question), or uses that chip if you already have it, for example from [Install](install.md). It installs `mashbox` on the chip and starts the server with the coding-agent settings above (`--agents`). If a server is already running on the chip, `--restart` stops it with the same signal as Ctrl-C and starts it again with these settings, so a chip that runs the default context gets the agent settings without a new chip. Then it opens an SSH tunnel from the chip's 127.0.0.1 to port 8000 on your machine, waits until the server answers through it and prints the base URLs for the agents.

This terminal holds the tunnel: leave it running. If port 8000 is taken on your machine, it uses the next free port and says which one: use that port in every address in steps 2 and 3. Ctrl-C closes the tunnel only; the server and the chip keep running and billing. `mashbox tpu tunnel mashbox-qwen3-8-27b` opens the tunnel again, and `mashbox tpu down mashbox-qwen3-8-27b` deletes the chip. If you forget, the chip deletes itself 8 hours after it starts (`--max-hours` changes that).

### By hand, on a chip you made

From your machine:

```bash
gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='~/.local/bin/mashbox run qwen3-8-27b --ctx 32768 --slots 1' -- -L 8000:127.0.0.1:8000
```

Leave it running. It takes requests once it prints a line with `"event": "ready"`. The server listens only on the chip's 127.0.0.1 and the SSH tunnel brings it to port 8000 on your machine, so nothing is open to the internet.

Closing the tunnel does not stop the server on the chip. To stop it, first end this command on your machine (Ctrl-C in its terminal, or stop its process if it runs in the background or from a script): while it runs, gcloud retries it when the server stops and starts the server again. Then stop the server on the chip:

```bash
gcloud compute tpus tpu-vm ssh mashbox-1 --zone=us-east5-b --command='pkill -INT -f "^/home/.*/mashbox run"'
```

`-INT` sends the same signal as Ctrl-C, and the `^/home/` start keeps the pattern from matching the shell that runs it.

## 2. Point your agent at it

Each agent has a page with its settings, the start command, a check, why each setting is there, how its prompts are cached and what a session looks like:

- [Claude Code](agents/claude-code.md): a settings file that you name with `--settings`.
- [Codex](agents/codex.md): a profile file that you start with `--profile mashbox`.
- [Pi](agents/pi.md): a profile folder that you name with `PI_CODING_AGENT_DIR`.

Each agent then compacts its conversation before it reaches the server's 32,768 tokens.

## 3. Check it works

From a second terminal on your machine:

```bash
curl http://127.0.0.1:8000/health
```

It prints `{"status": "ok"}`. Then run the check on your agent's page: one turn that ends with `ready`. The first request after the server starts is slow while the server compiles its programs for this context: 46 s on a TPU v6e-1. After that, a request whose prompt the server still has cached gets its first token in about 0.2 to 0.5 s, and one it has to read again takes several seconds (medians 4.5 to 9 s in the table below).

The server's terminal shows one line per request, with the agent in `ua`, the API in `api` and the part of the prompt it did not have to read again in `cached_tokens`:

```text
{"event": "request", "api": "anthropic", "ua": "claude-cli/2.1.312 (external, cli)", "prompt_tokens": 12700, "cached_tokens": 12438, "new_tokens": 143, "tools": 7, "tool_calls": 1, "thinking": false, "finish": "stop", "dropped_tools": null, "batch_rows": 1, "ttft_s": 0.269, "total_s": 4.319}
```

## Benchmark: a 9-turn coding task

Nine prompts to an agent that works in a clone of [python-slugify](https://github.com/un33k/python-slugify) (133 tests), one run per agent. Each agent runs in its terminal interface with the settings on its page, on one TPU v6e-1 (on demand) with `mashbox` 0.1.0, which keeps only the conversation in the slot (no host-memory cache). So that the agents answer without stopping to ask, Claude Code runs with `--permission-mode acceptEdits --allowedTools Bash` and Codex with `--dangerously-bypass-approvals-and-sandbox`, added to the commands above. Each agent answers every turn correctly, and the project's full test suite passes at the end.

Seconds per turn, from sending the prompt to the agent finishing the turn (approximate):

| Turn | What the agent is asked | Claude Code 2.1.312 | Codex CLI 0.165.0 | Pi 1.4.0 |
|---|---|---|---|---|
| 1 | read two files and note a codename and a reviewer for later | 44 | 27 | 21 |
| 2 | write a new module and its tests | 18 | 36 | 21 |
| 3 | run the new tests | 6 | 14 | 72 |
| 4 | fix the module for a failing test added to the project | 24 | 33 | 31 |
| 5 | add a parameter and two tests, run them | 38 | 32 | 23 |
| 6 | run the whole suite and `git status` | 18 | 23 | 10 |
| 7 | read a long CI log and name the failed step, error code, file and line | 175 | 52 | 50 |
| 8 | give the codename and reviewer from turn 1, from memory | 184 | 56 | 5 |
| 9 | commit the three new files with the reviewer in the message | 24 | 24 | 80 |
| | Requests | 35 | 36 | 34 |
| | Prompt tokens read from cache | 84% | 52% | 87% |
| | Time to first token, median, prompt cached / not cached | 0.27 s / 9.0 s | 0.33 s / 5.5 s | 0.26 s / 4.5 s |
| | Compactions (turn) | 2 (7 and 8) | 1 (8) | 1 (9) |

Where the time goes:

- **Compacting.** Claude Code's two compactions take 139 s and 173 s: it writes a summary of up to 4,096 tokens, and answers stream at about 29 tokens a second at a 20,000-token context.
- **Codex's title request.** Every turn Codex also sends a short request (about 4,700 tokens) to name the task. Without the host-memory cache it takes the place of the conversation in the chip's one slot, so the next request reads the whole conversation again (4 to 17 s to the first token). That is why Codex reads less from cache here.
- **First request.** The first request after the server starts takes about 46 s while the server compiles its programs.

## How the server reuses prompts

The three agents send the whole conversation with every request, so most of each prompt is the same as the one before. The server keeps the conversation it just answered on the chip, and a request that starts the same way only has its new part to read.

When a request takes the chip's one slot from another conversation (a second agent, or a short request that an agent sends on its own), the server first copies that conversation to host memory, and brings it back from there when it returns, on an exact match of its tokens, instead of reading it again. This follows llama-server's [host-memory prompt cache](https://github.com/ggml-org/llama.cpp/pull/16391) and its [resume points at user messages](https://github.com/ggml-org/llama.cpp/pull/24176), with the same option names. It is on by default and works the same for every client, chat included.

| Option | Default | What it does |
|---|---|---|
| `--cache-ram MIB` | `8192` | host memory for saved conversations; `0` turns it off |
| `--ctx-checkpoints N` | `8` | resume points kept per conversation |
| `--checkpoint-min-step N` | `8192` | tokens between resume points |

Qwen3.8-27B mixes attention layers with linear-attention layers, whose state cannot be cut at any token, so the server resumes a prompt only at a saved point: the end of the previous prompt, or the start of a user message (always the first one, then one at least every 8,192 tokens, up to 8). A prompt that changes before its first user message is read again in full. The answers do not change: with temperature 0 and a fixed seed, a 17,000-token conversation brought back from host memory gives the same text, token for token, as one that never left the chip, with its first token 1.0 s after the request.

The same task on one TPU v6e-1 (on demand), on a `mashbox` build without the host-memory cache and on one with it (without / with), counted by the server per request:

| | Claude Code 2.1.312 | Codex CLI 0.165.0 | Pi 1.4.0 |
|---|---|---|---|
| Prompt tokens read from cache | 78.9% / 85.4% | 51.8% / 86.3% | 88.5% / 89.5% |
| Requests that read the whole prompt | 7 of 41 / 2 of 37 | 21 of 38 / 4 of 38 | 3 of 25 / 2 of 26 |
| Times to first token, added up | 95 s / 75 s | 192 s / 104 s | 33 s / 54 s |

- **Codex** gains the most: after each title request its conversation comes back from host memory instead of being read again (188,234 tokens over the 9 turns).
- **Claude Code** gains at compaction: each of its 3 rebuilt histories resumes after the 6,126 tokens of tools and system prompt (4.0 to 4.7 s to the first token) instead of being read in full.
- **Pi** sends no extra requests, so it has little to gain. With the cache, its two requests that read the whole prompt take 6.3 s and 12.8 s to the first token, against 0.7 s and 1.2 s without it.

Each agent's page has the details: [Claude Code](agents/claude-code.md#caching), [Codex](agents/codex.md#caching), [Pi](agents/pi.md#caching).

## Subagents

Subagents work with all three agents. Each subagent is a conversation of its own, sent to the same server, so the server sees them as several users at once.

**Claude Code.** Subagents run through the `Agent` tool, which the deny list in the [Claude Code settings file](agents/claude-code.md) turns off to keep prompts short. To use them, take `"Agent",` out of the deny list in `~/.claude/mashbox.json` and add these lines to its `env` block:

```json
"ANTHROPIC_DEFAULT_FABLE_MODEL": "qwen3-8-27b",
"CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
"CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
```

The server answers whatever model name a request carries, so a subagent never fails on its name. With these lines every request names `qwen3-8-27b`: without them a subagent file with `model: fable` asks for `claude-fable-5-1`, and one that names a full Claude model id sends that id. `CLAUDE_CODE_ATTRIBUTION_HEADER` leaves out a line Claude Code puts at the top of its system prompt for Anthropic's billing, which changes from one session to the next.

**Codex.** Nothing to change. Codex starts subagents when you ask for them, for example "spawn three subagents, one per question", and runs up to 6 at once by default (`[agents] max_threads`).

**Pi.** Pi has no subagents of its own. Its package ships an example extension that runs each subagent as a separate `pi` process. Copy it into the profile folder from step 2. Its four sample agents name Claude models, which fail with any other provider (`Model "claude-haiku-4-5" is ambiguous across providers`), so these lines leave out their `model:` lines and the subagents use the session's model:

```bash
X="$(npm root -g)/@earendil-works/pi-coding-agent/examples/extensions/subagent"
mkdir -p ~/.pi/mashbox/extensions/subagent ~/.pi/mashbox/agents
cp "$X"/index.ts "$X"/agents.ts ~/.pi/mashbox/extensions/subagent/
for f in "$X"/agents/*.md; do grep -v '^model:' "$f" > ~/.pi/mashbox/agents/"$(basename "$f")"; done
```

**Several subagents on one chip.** With `--slots 1` the server answers one request at a time, so subagents started together take turns, request by request, and a request can wait several seconds for the slot. A conversation that has to wait keeps its prompt in host memory, and its next request picks up from there instead of reading the prompt again. Three subagents at once work on a TPU v6e-1 with each of the three agents: the main agent gets every subagent's answer and can use it in later turns. A second slot does not make them faster: a request that arrives while another runs waits for it to finish.

## Limits

- **One request at a time.** With one slot, agents that share a chip wait for each other's requests. With the host-memory cache the other conversation comes back on a switch: with Codex and Pi on one chip, 0.4 to 0.9 s to the first token after a switch, plus any wait for the other agent's request. Use one chip per agent if you run several.
- **Context.** Prompts of 32,768 tokens or more are refused with `prompt is too long`. The settings on each agent's page make it compact before that. A bigger context does not fit on one v6e chip: two 32,768-token slots need 33.65 GB where 31.24 GB is free, and `--ctx 65536` runs out of chip memory by 4 MB when the server compiles it.
- **Thinking is off** in the settings above. Thinking on is not supported with these settings: above about 27,000 tokens answers are cut off at the context limit, and a compaction can run for over 8 minutes.
- **Text only.** Images and files in a request are refused with an error that says so.
- **Hosted tools.** Tools that run on Anthropic's or OpenAI's servers (web search) are not available here. A Claude Code web search request is refused with an error; Codex's web search tool is left out of the list the model sees, and the server log names it under `dropped_tools`.

## Troubleshooting

| What you see | What to do |
|---|---|
| `prompt is too long: N tokens > 32767 maximum` | use the context settings in step 2 for your agent, or start a new session |
| `Connection refused` from the agent or from `curl` | the tunnel is down; with one command, run `mashbox tpu tunnel mashbox-qwen3-8-27b`; by hand, run step 1 again |
| `bind [127.0.0.1]:8000: Address already in use` in step 1 by hand | something else on your machine uses port 8000; stop it, or use `-L 8001:127.0.0.1:8000` and 8001 in every address in step 2 |
| `403 Host not allowed`, `403 Origin not allowed` or `415 Content-Type must be application/json` from the server | the server answers clients on this machine by localhost or an IP address; a web page of another site, or a request with another host name, is refused, and a POST must send `Content-Type: application/json` (agents and `curl -H 'Content-Type: application/json'` do) |
| `The TPU is already in use by process ...` when starting the server | an old server is still running on the chip; stop it with the command at the end of step 1 and wait a few seconds |
| Claude Code: `[claude-code:unrecognized_model]` | a warning only; Claude Code works |
| Codex: `Model metadata for qwen3-8-27b not found` | a warning only; Codex works |
| Codex: `--profile mashbox cannot be used while ... config.toml contains legacy ... [profiles.mashbox]` | Codex 0.165 wants the profile in its own file: move those lines to `~/.codex/mashbox.config.toml` as in step 2 |
| Codex: `wire_api = "chat" is no longer supported` | use `wire_api = "responses"` as in step 2 |

## Next

- [Qwen3.8-27B](models/qwen3-8-27b.md): the model, its options and its numbers.
- [Run on Google Cloud TPU](google-cloud.md): create and delete the chip.
