# Use with coding agents

Claude Code, Codex and Pi can use a model that `mashbox` serves as their LLM. The `mashbox` server speaks the three APIs these agents use: Anthropic Messages (Claude Code), OpenAI Responses (Codex) and OpenAI chat completions with tool calls (Pi). This is part of the server, not of one model, so every LLM that `mashbox run` serves gets it. Today that is [Qwen3.8-27B](models/qwen3-8-27b.md), the example on this page. For another LLM, put its id from the models table in place of `qwen3-8-27b`.

The agent keeps working the way it always does: it reads files, runs commands, edits code and goes back and forth with the model, and every request runs on your chip.

| Model | Server options for agents | Chip |
|---|---|---|
| [Qwen3.8-27B](models/qwen3-8-27b.md) | `--ctx 32768 --slots 1` | TPU v6e-1 |

## What you need

- A chip with `mashbox` installed: [Run on Google Cloud TPU](google-cloud.md) and the model's guide. Qwen3.8-27B runs on one TPU v6e chip, $2.70 an hour on demand at Google's list price (read 19 October 2026), billed from create to delete.
- Claude Code, Codex CLI or Pi on your own machine.

## 1. Start the server and the tunnel

From your machine, with your chip's name and zone (the [Qwen3.8-27B guide](models/qwen3-8-27b.md) calls the chip `qwen-1`):

<!-- test:tpu -->
```bash
gcloud compute tpus tpu-vm ssh qwen-1 --zone=us-east5-b --command='~/.local/bin/mashbox run qwen3-8-27b --ctx 32768 --slots 1' -- -L 8000:127.0.0.1:8000
```

Agents send long prompts (Claude Code sends about 18,000 tokens of instructions and tool descriptions before your first word), so the server runs a 32,768-token context with one conversation slot. That is what fits next to the 26.94 GB of weights on a v6e chip: two slots at this context ran out of chip memory in our test.

Leave it running and wait for the line `{"event": "ready", ...}`. The server listens only on the chip's 127.0.0.1 and the SSH tunnel brings it to port 8000 on your machine, so nothing is open to the internet. Check it from a second terminal: `curl http://127.0.0.1:8000/health` prints `{"status": "ok"}`.

The server does not check API keys. The agents below want one, so give them any string, for example `local`.

## 2. Point your agent at it

### Claude Code

Add an `env` block to `~/.claude/settings.json`:

```json
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
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"
  },
  "model": "qwen3-8-27b"
}
```

Then run `claude` as usual. If you already use Claude Code with your Anthropic account, put these settings in a separate user or project so your normal setup stays as it is.

### Codex CLI

In `~/.codex/config.toml`:

```toml
model_provider = "mashbox"
model = "qwen3-8-27b"
model_context_window = 32768
model_auto_compact_token_limit = 26214
check_for_update_on_startup = false

[analytics]
enabled = false

[model_providers.mashbox]
name = "MASHBOX TPU"
base_url = "http://127.0.0.1:8000/v1"
env_key = "MASHBOX_API_KEY"
wire_api = "responses"
```

Then `export MASHBOX_API_KEY=local` and run `codex`. Codex 0.165 accepts only `wire_api = "responses"`; the server has the Responses API for it.

### Pi

In `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "mashbox": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "$MASHBOX_API_KEY",
      "models": [{"id": "qwen3-8-27b", "name": "Qwen3.8-27B on MASHBOX TPU", "contextWindow": 32768, "maxTokens": 4096, "reasoning": false}]
    }
  }
}
```

And in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "mashbox",
  "defaultModel": "qwen3-8-27b"
}
```

Then `export MASHBOX_API_KEY=local` and run `pi`.

Other agents that take an OpenAI-compatible base URL can use the same values: base URL `http://127.0.0.1:8000/v1`, the model id, any key. We tested the three above.

## Check it worked

Every request is one line in the server's output, for example:

```text
{"event": "request", "api": "anthropic", "ua": "claude-cli/2.1.312 ...", "prompt_tokens": 18585, "cached_tokens": 18253, "new_tokens": 128, "tool_calls": 1, "ttft_s": 0.629, ...}
```

`api` says which API the agent used and `cached_tokens` how much of the prompt the server did not have to read again.

## What we measured

Qwen3.8-27B on one TPU v6e-1 (us-east5-b, 28 October 2026). Each agent was given the same kind of task, "run the test, find the bug, fix it, run the test again", in a small Python project with a planted bug. All three fixed it, and the server log shows every request.

| Agent | Version | API | Result |
|---|---|---|---|
| Claude Code | 2.1.312 | Anthropic Messages, streamed | ran the test, read 2 files, edited, reran: `ok`. 6 requests, 182 s |
| Codex CLI | 0.165.0 | OpenAI Responses, streamed | ran the test, read the files, patched, reran: `ok` |
| Pi | 1.4.0 | OpenAI chat completions with tools, streamed | ran the test and read the file in one turn, edited, reran: `ok` |

Agents send the whole conversation again on every turn. The server keeps the start of the last prompt on the chip and reads only what is new:

| Prompt | Time to first token, first turn | Time to first token, next turns |
|---|---|---|
| 19,467 tokens | 12.2 s | 0.70 to 1.15 s (99.6% of the prompt cached) |
| 30,179 tokens | 27.4 s | 0.77 s |

Answers stream at about 29 tokens a second at a 20,000-token context.

## Limits

- **One conversation at a time is cached.** With one slot, two agents on one chip take turns and each switch reads the whole prompt again (about 12 s for 19,000 tokens on Qwen3.8-27B). Use one chip per agent if you run several.
- **Context.** Prompts over 32,767 tokens are refused with `prompt is too long`. Set the agent's context to 32,768 so it compacts the conversation before that.
- **Thinking.** Claude Code asks for thinking, and the model then thinks before each step, which takes longer. Codex and Pi do not ask for it by default.
- **Text only.** Images and files in a request are refused with an error that says so.
- **Hosted tools.** Tools that run on Anthropic's or OpenAI's servers (web search) are not available here. A Claude Code web search request is refused with an error; Codex's web search tool is left out of the list the model sees, and the server log names it under `dropped_tools`.

## Troubleshooting

| What you see | What to do |
|---|---|
| `prompt is too long: N tokens > 32767 maximum` | lower the agent's context setting to 32,768 or start a new session |
| `Connection refused` from the agent | the tunnel is down; start step 1 again |
| `The TPU is already in use by process ...` when restarting the server | an old server is still running on the chip; stop it with Ctrl-C in its terminal and wait a few seconds |
| Codex: `wire_api = "chat" is no longer supported` | use `wire_api = "responses"` |

## Next

- [Qwen3.8-27B](models/qwen3-8-27b.md): the model, its options and its numbers.
- [Run on Google Cloud TPU](google-cloud.md): create and delete the chip.
