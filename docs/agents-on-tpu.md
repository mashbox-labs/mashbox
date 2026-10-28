# Use with coding agents

Claude Code, Codex and Pi can use a model that `mashbox` serves on your TPU as their LLM. Keep the agent you already have: nothing on this page reinstalls it or changes its normal settings. Claude Code reads one extra settings file that you name when you start it, Codex gets an added provider and profile, and Pi gets a profile folder of its own. Started the usual way, each agent works as before.

The `mashbox` server speaks the APIs these agents use: Anthropic Messages (Claude Code), OpenAI Responses (Codex) and OpenAI chat completions with tool calls (Pi). Every LLM that `mashbox run` serves gets them. In this release that is [Qwen3.8-27B](models/qwen3-8-27b.md), the example on this page.

## What it costs

The chip bills from create to delete: $2.70 an hour for one TPU v6e chip at Google's on-demand list price. The server does not charge per token and does not check keys.

## Before you start

- A TPU v6e-1 with `mashbox` on it, from the [Quickstart](quickstart.md) or the [Qwen3.8-27B guide](models/qwen3-8-27b.md). Below the chip is `qwen-1` in zone `us-east5-b`: use your chip's name and zone.
- Claude Code, Codex CLI or Pi, already installed on your machine. The settings below are for Claude Code 2.1.312, Codex CLI 0.165.0 and Pi 1.4.0.
- Port 8000 free on your machine.

## 1. Start the server and the tunnel

From your machine:

```bash
gcloud compute tpus tpu-vm ssh qwen-1 --zone=us-east5-b --command='~/.local/bin/mashbox run qwen3-8-27b --ctx 32768 --slots 1' -- -L 8000:127.0.0.1:8000
```

Leave it running. It takes requests once it prints a line with `"event": "ready"`. The server listens only on the chip's 127.0.0.1 and the SSH tunnel brings it to port 8000 on your machine, so nothing is open to the internet.

Closing the tunnel does not stop the server on the chip. To stop it:

```bash
gcloud compute tpus tpu-vm ssh qwen-1 --zone=us-east5-b --command='pkill -f "mashbox run"'
```

`--ctx 32768 --slots 1` gives one conversation of up to 32,768 tokens, which is what fits next to the 26.94 GB of weights on a v6e chip; two slots at this context do not fit. Each agent below is set to compact its conversation before it gets there.

## 2. Point your agent at it

### Claude Code

Save these settings in a new file:

```bash
cat > ~/.claude/mashbox.json <<'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "qwen3-8-27b",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen3-8-27b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3-8-27b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-8-27b",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3-8-27b",
    "CLAUDE_CODE_SUBAGENT_MODEL": "qwen3-8-27b",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "36000",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
    "MAX_THINKING_TOKENS": "0"
  },
  "model": "qwen3-8-27b",
  "permissions": {
    "deny": ["Agent", "CronCreate", "CronDelete", "CronList", "EnterWorktree", "ExitWorktree", "ListAgents", "Monitor", "NotebookEdit", "ReportFindings", "ScheduleWakeup", "SendMessage", "Skill", "TaskStop", "WebFetch", "WebSearch", "Workflow"]
  }
}
EOF
```

Then, in your project folder:

```bash
claude --settings ~/.claude/mashbox.json --strict-mcp-config
```

The file sends every request to the tunnel with our model in every role and turns off Claude Code's other network traffic for the session. It makes Claude Code compact at about 19,000 tokens (it compacts at `CLAUDE_CODE_MAX_CONTEXT_TOKENS` minus its output size minus about 13,000), so the summary still has room, and turns thinking off so one step cannot use up the rest of the 32,768 tokens. The deny list leaves out 17 tools that cannot work here or that a coding session does not need: the first prompt drops from about 20,000 tokens to about 8,000. `--strict-mcp-config` leaves out your MCP servers for the session, as their tools would take room too. Nothing is written to your `~/.claude/settings.json`, and `claude` without these options works as before.

### Codex CLI

Codex reads a named profile from its own file next to `config.toml`. Save this as `~/.codex/mashbox.config.toml`:

```bash
cat > ~/.codex/mashbox.config.toml <<'EOF'
model_provider = "mashbox"
model = "qwen3-8-27b"
model_context_window = 32768
model_auto_compact_token_limit = 26214
model_reasoning_effort = "none"

[model_providers.mashbox]
name = "MASHBOX TPU"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
EOF
```

Then, in your project folder:

```bash
codex --profile mashbox
```

Codex compacts at 26,214 tokens (80% of 32,768), and `model_reasoning_effort = "none"` keeps thinking off even when your own config sets an effort, so long thinking cannot fill the context. Your `~/.codex/config.toml` stays as it is, and `codex` without `--profile mashbox` works as before. Codex 0.165 refuses a `[profiles.mashbox]` table inside `config.toml` when you start it with `--profile mashbox`: use the separate file.

### Pi

Make a Pi profile folder with the provider and compaction settings:

```bash
mkdir -p ~/.pi/mashbox
cat > ~/.pi/mashbox/models.json <<'EOF'
{
  "providers": {
    "mashbox": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{"id": "qwen3-8-27b", "name": "Qwen3.8-27B on MASHBOX TPU", "contextWindow": 32768, "maxTokens": 4096, "reasoning": false}]
    }
  }
}
EOF
cat > ~/.pi/mashbox/settings.json <<'EOF'
{
  "defaultProvider": "mashbox",
  "defaultModel": "qwen3-8-27b",
  "compaction": {"enabled": true, "reserveTokens": 6144, "keepRecentTokens": 8192}
}
EOF
```

Then, in your project folder:

```bash
PI_CODING_AGENT_DIR=~/.pi/mashbox pi
```

Pi's default compaction (reserve 16,384 tokens, keep 20,000) does not fit a 32,768 context: after one large file read Pi stays stuck on `prompt is too long`. These settings compact at 26,624 tokens and keep the last 8,192. Your `~/.pi/agent` folder stays as it is, and `pi` alone works as before.

## 3. Check it works

From a second terminal on your machine:

```bash
curl http://127.0.0.1:8000/health
```

It prints `{"status": "ok"}`. Then one turn from your agent, in your project folder. Claude Code:

```bash
claude --settings ~/.claude/mashbox.json --strict-mcp-config -p "Reply with one word: ready"
```

Codex:

```bash
codex exec --profile mashbox "Reply with one word: ready"
```

Pi:

```bash
PI_CODING_AGENT_DIR=~/.pi/mashbox pi -p "Reply with one word: ready"
```

Each ends with `ready`. Before it, Claude Code prints a warning that it does not know the model name (`[claude-code:unrecognized_model]`) and Codex one about model metadata: both are harmless. The first request after the server starts is slow while the server compiles its programs for this context: 46 s on a TPU v6e-1, then about 0.2 to 0.5 s to the first token on later turns.

The server's terminal shows one line per request, with the agent in `ua`, the API in `api` and the part of the prompt it did not have to read again in `cached_tokens`:

```text
{"event": "request", "api": "anthropic", "ua": "claude-cli/2.1.312 (external, cli)", "prompt_tokens": 12700, "cached_tokens": 12438, "new_tokens": 143, "tools": 7, "tool_calls": 1, "thinking": false, "finish": "stop", "dropped_tools": null, "batch_rows": 1, "ttft_s": 0.269, "total_s": 4.319}
```

## Benchmark: a 9-turn coding task

Each agent did the same 9-turn session in a clone of [python-slugify](https://github.com/un33k/python-slugify) (133 tests). The agents ran in their terminal interfaces with the settings on this page, on one TPU v6e-1 (on demand) with `mashbox` 0.1.0 from PyPI. So that they did not stop to ask, Claude Code started with `--permission-mode acceptEdits --allowedTools Bash` and Codex with `--dangerously-bypass-approvals-and-sandbox`, added to the commands above. Each agent answered every turn correctly, and the project's full test suite passed at the end of each session.

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

- **Compacting.** Claude Code's two compactions took 139 s and 173 s: it writes a summary of up to 4,096 tokens, and answers stream at about 29 tokens a second at a 20,000-token context.
- **Codex's title request.** Every turn Codex also sends a short request (about 4,700 tokens) to name the task. With one slot it takes the place of the conversation in the cache, so the next request reads the whole conversation again (4 to 17 s to the first token). That is why Codex reads less from cache.
- **First request.** The first request after the server starts took 46 s while the server compiled its programs, and a prompt length the server has not seen before adds a few seconds once.

## Limits

- **One conversation at a time is cached.** With one slot, two agents on one chip take turns and each switch reads the whole prompt again. Use one chip per agent if you run several.
- **Context.** Prompts over 32,767 tokens are refused with `prompt is too long`. The settings above make each agent compact before that.
- **Thinking is off** in the settings above. With thinking on, Claude Code on a TPU v6e-1 ran out of room above 27,000 tokens: 12 answers were cut off at the context limit and one turn kept compacting for over 8 minutes.
- **Text only.** Images and files in a request are refused with an error that says so.
- **Hosted tools.** Tools that run on Anthropic's or OpenAI's servers (web search) are not available here. A Claude Code web search request is refused with an error; Codex's web search tool is left out of the list the model sees, and the server log names it under `dropped_tools`.

## Troubleshooting

| What you see | What to do |
|---|---|
| `prompt is too long: N tokens > 32767 maximum` | use the context settings in step 2 for your agent, or start a new session |
| `Connection refused` from the agent or from `curl` | the tunnel is down; run step 1 again |
| `bind [127.0.0.1]:8000: Address already in use` in step 1 | something else on your machine uses port 8000; stop it, or use `-L 8001:127.0.0.1:8000` and 8001 in every address in step 2 |
| `403 Host not allowed`, `403 Origin not allowed` or `415 Content-Type must be application/json` from the server | the server answers clients on this machine by localhost or an IP address; a web page of another site, or a request with another host name, is refused, and a POST must send `Content-Type: application/json` (agents and `curl -H 'Content-Type: application/json'` do) |
| `The TPU is already in use by process ...` when starting the server | an old server is still running on the chip; stop it with the command at the end of step 1 and wait a few seconds |
| Claude Code: `[claude-code:unrecognized_model]` | a warning only; Claude Code works |
| Codex: `Model metadata for qwen3-8-27b not found` | a warning only; Codex works |
| Codex: `--profile mashbox cannot be used while ... config.toml contains legacy ... [profiles.mashbox]` | Codex 0.165 wants the profile in its own file: move those lines to `~/.codex/mashbox.config.toml` as in step 2 |
| Codex: `wire_api = "chat" is no longer supported` | use `wire_api = "responses"` as in step 2 |

## Next

- [Qwen3.8-27B](models/qwen3-8-27b.md): the model, its options and its numbers.
- [Run on Google Cloud TPU](google-cloud.md): create and delete the chip.
