<!-- Generated from src/mashbox/runtime/openapi.py by tools/hf_github_sync_check.py render --write. Do not edit. -->
# Server API

The HTTP server `mashbox run <model>` starts on 127.0.0.1 (reach it from your machine with an SSH tunnel). It speaks the OpenAI Chat Completions, Completions and Responses APIs and the Anthropic Messages API, so OpenAI and Anthropic clients and coding agents work against it unchanged. On 127.0.0.1 no API key is checked and any key a client sends is ignored; with `--api-key-file` or `MASHBOX_API_KEY`, and always when `--host` is not loopback, `/v1` routes, `/metrics` and `/openapi.json` need a key (`Authorization: Bearer <key>` or `x-api-key: <key>`; 401 without). POST bodies must be `Content-Type: application/json` (415 otherwise), and a request from a web page of another origin, or naming the server by a host name it does not answer to, gets 403. Fields a route does not list are accepted and ignored.

The same description as an OpenAPI 3.1 file: [openapi.json](openapi.json) in these docs, or `GET /openapi.json` on a running server.

| Method | Path | What it does |
|---|---|---|
| GET | `/v1/health` | [Health](#health) |
| GET | `/health` | [Health (short path)](#healthshort) |
| GET | `/` | [Health, or the chat page](#root) |
| HEAD | `/v1/health` | [Ready check without a body](#healthhead) |
| GET | `/v1/models` | [List models](#listmodels) |
| GET | `/v1/models/{model}` | [Get a model](#getmodel) |
| POST | `/v1/chat/completions` | [Chat completion](#createchatcompletion) |
| POST | `/v1/completions` | [Text completion](#createcompletion) |
| POST | `/v1/messages` | [Messages (Anthropic)](#createmessage) |
| POST | `/v1/messages/count_tokens` | [Count tokens](#counttokens) |
| POST | `/v1/responses` | [Responses (OpenAI)](#createresponse) |
| GET | `/metrics` | [Metrics (Prometheus text)](#metrics) |
| GET | `/openapi.json` | [This description](#openapi) |

<a id="health"></a>

## GET /v1/health

200 when the model is loaded and the server takes requests, 503 while it loads. GET /health is the same.

**Responses:**

- 200: OK `application/json` Health
- 503: The model is still loading. `application/json` Health

<a id="healthshort"></a>

## GET /health

Same as GET /v1/health.

**Responses:**

- 200: OK `application/json` Health
- 503: The model is still loading. `application/json` Health

<a id="root"></a>

## GET /

Same as GET /v1/health. With `mashbox run <model> --ui` it is the MASHBOX Chat page instead.

**Responses:**

- 200: OK `application/json` Health
- 503: The model is still loading. `application/json` Health

<a id="healthhead"></a>

## HEAD /v1/health

200 when ready, 503 while loading, no body. The server answers HEAD the same way on every path.

**Responses:**

- 200: Ready.
- 503: Loading.

<a id="listmodels"></a>

## GET /v1/models

The one model this server runs.

**Responses:**

- 200: OK `application/json` ModelList

<a id="getmodel"></a>

## GET /v1/models/{model}

The model, when {model} is its id; 404 otherwise.

**Responses:**

- 200: OK `application/json` Model
- 404: Another model id. `application/json` Error

<a id="createchatcompletion"></a>

## POST /v1/chat/completions

OpenAI Chat Completions with function tools, thinking and streaming. Coding agents such as Pi use it.

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string |  | Accepted for compatibility; the server answers with the one model it runs. |
| `messages` (required) | array of ChatMessage |  | The conversation. |
| `tools` | array of object |  | Function tools the model may call. |
| `tool_choice` | string or object |  | auto (default), none, required, or {"type": "function", "function": {"name": ...}}. |
| `stream` | boolean | false | Stream the answer as server-sent events (see the 200 text/event-stream response). A client that reads nothing from the stream for 30 s is disconnected and its answer stops. |
| `stream_options` | object |  |  Fields: `include_usage`. |
| `chat_template_kwargs` | object |  | Chat template options. Fields: `enable_thinking`. |
| `enable_thinking` | boolean |  | Same as chat_template_kwargs.enable_thinking. |
| `merge` | boolean | true | Accepted and ignored: every request joins the running batch at the next device call, and no request field changes how other requests are scheduled. |
| `prompt_cache_key` | string |  | Tags the request's session in the server log (as the X-Session-Id header does). |
| `store` | boolean |  | Accepted and ignored. |
| `logprobs` | boolean |  | Not supported (log probabilities): a request that sets it gets 400 naming the field. |
| `top_logprobs` | integer |  | Not supported (log probabilities): a request that sets it gets 400 naming the field. |
| `response_format` | object |  | Only {"type": "text"}. Not supported: json_object and json_schema (structured output), answered with 400. |
| `max_tokens` | integer |  | Most new tokens (>= 1). |
| `max_completion_tokens` | integer |  | Most new tokens (>= 1); max_tokens is the same. Default: up to the end of the context (--ctx). |
| `temperature` | number | 0.0 | 0 = greedy (default). Up to 4. |
| `top_p` | number | 1.0 | Nucleus sampling, in (0, 1]. |
| `top_k` | integer | 0 | Keep the k most likely tokens; 0 = off. |
| `min_p` | number | 0.0 | Drop tokens below min_p times the top probability, in [0, 1). |
| `seed` | integer |  | Sampling seed; a random one when absent. |
| `stop` | string or array |  | A string or a list of non-empty strings; the text ends before the first one. |
| `stop_sequences` | array of string |  | Same as stop (Anthropic's name), used when stop is absent. |
| `repetition_penalty` | number | 1.0 | > 0; 1 = off. |
| `frequency_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `presence_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `n` | integer |  | Must be 1. |

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Name three planets."}], "max_tokens": 64}'
```

**Responses:**

- 200: OK `application/json` ChatCompletion, `text/event-stream` ChatCompletionChunk
- 400: Bad request. `application/json` Error
- 404: A model id other than the one this server serves (GET /v1/models), code model_not_found. Leaving model out is accepted. `application/json` Error
- 429: Too many requests are waiting for a slot: 32 of yours (per tenant: keys the key file names with one tenant share it, a key without a name is its own tenant; without keys, 32 in all), or the server's queue is full (256 waiting in all). Retry after the Retry-After header's seconds. `application/json` Error
- 500: The request failed on the device: its one retry failed too, or it failed after its stream had sent output. A request whose client has received nothing is sent again (once, from the start): not streamed, until its response is written; streamed, until its first output is sent. A stream that has sent output is never retried by the server, since a second run could repeat what the client already has; it ends with the route's error event instead, carrying the response id as request_id. `application/json` Error
- 503: The model is still loading. Retry after the Retry-After header's seconds. `application/json` Error

<a id="createcompletion"></a>

## POST /v1/completions

Continue a plain prompt, no chat template.

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string |  | Accepted for compatibility; the server answers with the one model it runs. |
| `prompt` (required) | string or array |  | The text to continue (a string, or a list of one string). |
| `stream` | boolean | false | Stream the answer as server-sent events (see the 200 text/event-stream response). A client that reads nothing from the stream for 30 s is disconnected and its answer stops. |
| `merge` | boolean | true | Accepted and ignored: every request joins the running batch at the next device call, and no request field changes how other requests are scheduled. |
| `prompt_cache_key` | string |  | Tags the request's session in the server log (as the X-Session-Id header does). |
| `logprobs` | integer |  | Not supported (log probabilities): a request that sets it gets 400 naming the field. |
| `max_tokens` | integer |  | Most new tokens (>= 1). |
| `max_completion_tokens` | integer |  | Same as max_tokens (OpenAI's newer name); wins when both are set. |
| `temperature` | number | 0.0 | 0 = greedy (default). Up to 4. |
| `top_p` | number | 1.0 | Nucleus sampling, in (0, 1]. |
| `top_k` | integer | 0 | Keep the k most likely tokens; 0 = off. |
| `min_p` | number | 0.0 | Drop tokens below min_p times the top probability, in [0, 1). |
| `seed` | integer |  | Sampling seed; a random one when absent. |
| `stop` | string or array |  | A string or a list of non-empty strings; the text ends before the first one. |
| `stop_sequences` | array of string |  | Same as stop (Anthropic's name), used when stop is absent. |
| `repetition_penalty` | number | 1.0 | > 0; 1 = off. |
| `frequency_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `presence_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `n` | integer |  | Must be 1. |

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The three largest planets are", "max_tokens": 32}'
```

**Responses:**

- 200: OK `application/json` Completion, `text/event-stream` Completion
- 400: Bad request. `application/json` Error
- 404: A model id other than the one this server serves (GET /v1/models), code model_not_found. Leaving model out is accepted. `application/json` Error
- 429: Too many requests are waiting for a slot: 32 of yours (per tenant: keys the key file names with one tenant share it, a key without a name is its own tenant; without keys, 32 in all), or the server's queue is full (256 waiting in all). Retry after the Retry-After header's seconds. `application/json` Error
- 500: The request failed on the device: its one retry failed too, or it failed after its stream had sent output. A request whose client has received nothing is sent again (once, from the start): not streamed, until its response is written; streamed, until its first output is sent. A stream that has sent output is never retried by the server, since a second run could repeat what the client already has; it ends with the route's error event instead, carrying the response id as request_id. `application/json` Error
- 503: The model is still loading. Retry after the Retry-After header's seconds. `application/json` Error

<a id="createmessage"></a>

## POST /v1/messages

Anthropic Messages with client tools, thinking and streaming, for Claude Code. The ?beta=true query Claude Code adds is accepted.

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string |  | Accepted for compatibility; the server answers with the one model it runs. |
| `messages` (required) | array of object |  | The conversation. |
| `system` | string or array |  | System prompt: a string or text blocks. |
| `tools` | array of object |  | Client tools the model may call. |
| `tool_choice` | object |  | {"type": "auto" \| "any" \| "tool" \| "none", "name": ...}. |
| `thinking` | object |  |  Fields: `type`, `budget_tokens`. |
| `stream` | boolean | false | Stream the answer as server-sent events (see the 200 text/event-stream response). A client that reads nothing from the stream for 30 s is disconnected and its answer stops. |
| `prompt_cache_key` | string |  | Tags the request's session in the server log (as the X-Session-Id header does). |
| `metadata` | object |  | Accepted and ignored. |
| `output_config` | object |  | effort is accepted and ignored. Not supported: format (structured output), answered with 400. |
| `max_tokens` | integer |  | Most new tokens; default: up to the end of the context (--ctx). |
| `max_completion_tokens` | integer |  | Same as max_tokens (OpenAI's newer name); wins when both are set. |
| `temperature` | number | 0.0 | 0 = greedy (default). Up to 4. |
| `top_p` | number | 1.0 | Nucleus sampling, in (0, 1]. |
| `top_k` | integer | 0 | Keep the k most likely tokens; 0 = off. |
| `min_p` | number | 0.0 | Drop tokens below min_p times the top probability, in [0, 1). |
| `seed` | integer |  | Sampling seed; a random one when absent. |
| `stop` | string or array |  | A string or a list of non-empty strings; the text ends before the first one. |
| `stop_sequences` | array of string |  | Same as stop (Anthropic's name), used when stop is absent. |
| `repetition_penalty` | number | 1.0 | > 0; 1 = off. |
| `frequency_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `presence_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `n` | integer |  | Must be 1. |

```bash
curl -s http://127.0.0.1:8000/v1/messages -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b", "max_tokens": 64, "messages": [{"role": "user", "content": "Name three planets."}]}'
```

**Responses:**

- 200: OK `application/json` Message, `text/event-stream` MessageStreamEvent
- 400: Bad request. `application/json` AnthropicError
- 429: Too many requests are waiting for a slot: 32 of yours (per tenant: keys the key file names with one tenant share it, a key without a name is its own tenant; without keys, 32 in all), or the server's queue is full (256 waiting in all). Retry after the Retry-After header's seconds. `application/json` AnthropicError
- 500: The request failed on the device: its one retry failed too, or it failed after its stream had sent output. A request whose client has received nothing is sent again (once, from the start): not streamed, until its response is written; streamed, until its first output is sent. A stream that has sent output is never retried by the server, since a second run could repeat what the client already has; it ends with the route's error event instead, carrying the response id as request_id. `application/json` AnthropicError
- 529: The model is still loading, or its worker is starting again after a fault, or the server's queue is full (256 waiting in all): overloaded_error. Retry after the Retry-After header's seconds (x-should-retry: true while not ready). `application/json` AnthropicError

<a id="counttokens"></a>

## POST /v1/messages/count_tokens

Prompt tokens a Messages request would use, after the chat template.

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string |  | Accepted for compatibility; the server answers with the one model it runs. |
| `messages` (required) | array of object |  |  |
| `system` | string or array |  | As in MessagesRequest. |
| `tools` | array of object |  |  |
| `tool_choice` | object |  | As in MessagesRequest. |
| `thinking` | object |  | As in MessagesRequest. |

```bash
curl -s http://127.0.0.1:8000/v1/messages/count_tokens -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Name three planets."}]}'
```

**Responses:**

- 200: OK `application/json` TokenCount
- 400: Bad request. `application/json` AnthropicError
- 529: The model is still loading, or its worker is starting again after a fault, or the server's queue is full (256 waiting in all): overloaded_error. Retry after the Retry-After header's seconds (x-should-retry: true while not ready). `application/json` AnthropicError

<a id="createresponse"></a>

## POST /v1/responses

OpenAI Responses with function, custom and namespace tools, reasoning and streaming, for Codex.

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string |  | Accepted for compatibility; the server answers with the one model it runs. |
| `input` (required) | string or array |  | A string, or items: message (system, developer, user, assistant), function_call, function_call_output, custom_tool_call, custom_tool_call_output, local_shell_call, local_shell_call_output, reasoning. Text parts only: image, audio and file parts are refused with 400 (text-only model). |
| `instructions` | string |  | System prompt. |
| `tools` | array of object |  |  |
| `tool_choice` | string or object |  | auto (default), none, required, or {"type": "function", "function": {"name": ...}}. |
| `parallel_tool_calls` | boolean | false | Echoed in the response. |
| `reasoning` | object |  |  Fields: `effort`. |
| `stream` | boolean | false | Stream the answer as server-sent events (see the 200 text/event-stream response). A client that reads nothing from the stream for 30 s is disconnected and its answer stops. |
| `prompt_cache_key` | string |  | Tags the request's session in the server log (as the X-Session-Id header does). |
| `store` | boolean |  | Accepted and ignored. |
| `client_metadata` | object |  | Accepted and ignored. |
| `include` | array |  | Accepted and ignored, except an entry naming logprobs: 400 (log probabilities are not supported). |
| `top_logprobs` | integer |  | Not supported (log probabilities): a request that sets it gets 400 naming the field. |
| `text` | object |  | Only format {"type": "text"}. Not supported: other formats (structured output), answered with 400. |
| `max_output_tokens` | integer |  | Most new tokens; default: up to the end of the context (--ctx). |
| `max_tokens` | integer |  | Most new tokens (>= 1). |
| `max_completion_tokens` | integer |  | Same as max_tokens (OpenAI's newer name); wins when both are set. |
| `temperature` | number | 0.0 | 0 = greedy (default). Up to 4. |
| `top_p` | number | 1.0 | Nucleus sampling, in (0, 1]. |
| `top_k` | integer | 0 | Keep the k most likely tokens; 0 = off. |
| `min_p` | number | 0.0 | Drop tokens below min_p times the top probability, in [0, 1). |
| `seed` | integer |  | Sampling seed; a random one when absent. |
| `stop` | string or array |  | A string or a list of non-empty strings; the text ends before the first one. |
| `stop_sequences` | array of string |  | Same as stop (Anthropic's name), used when stop is absent. |
| `repetition_penalty` | number | 1.0 | > 0; 1 = off. |
| `frequency_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `presence_penalty` | number | 0.0 | -2 to 2; 0 = off. |
| `n` | integer |  | Must be 1. |

```bash
curl -s http://127.0.0.1:8000/v1/responses -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b", "input": "Name three planets.", "max_output_tokens": 64}'
```

**Responses:**

- 200: OK `application/json` Response, `text/event-stream` ResponseStreamEvent
- 400: Bad request. `application/json` Error
- 404: A model id other than the one this server serves (GET /v1/models), code model_not_found. Leaving model out is accepted. `application/json` Error
- 429: Too many requests are waiting for a slot: 32 of yours (per tenant: keys the key file names with one tenant share it, a key without a name is its own tenant; without keys, 32 in all), or the server's queue is full (256 waiting in all). Retry after the Retry-After header's seconds. `application/json` Error
- 500: The request failed on the device: its one retry failed too, or it failed after its stream had sent output. A request whose client has received nothing is sent again (once, from the start): not streamed, until its response is written; streamed, until its first output is sent. A stream that has sent output is never retried by the server, since a second run could repeat what the client already has; it ends with the route's error event instead, carrying the response id as request_id. `application/json` Error
- 503: The model is still loading. Retry after the Retry-After header's seconds. `application/json` Error

<a id="metrics"></a>

## GET /metrics

What the server does, in the Prometheus text format: requests running and waiting, KV cache use, the cache layout (one slot holds one whole context: block_size = the context length, num_blocks = the slots), requests answered by finish reason, prompt and generated tokens, prefix cache lookups and hits (tokens), preemptions (never), worker restarts, programs compiled after ready, and histograms of time to first token, queue time, end-to-end time and time per output token. Every name starts with `mashbox:`. No label names a user, an API key or a session.

**Responses:**

- 200: Prometheus text exposition format. `text/plain`

<a id="openapi"></a>

## GET /openapi.json

The OpenAPI 3.1 description of this server, with the running model's id.

**Responses:**

- 200: OpenAPI 3.1 document. `application/json`

## Schemas

### Health

| Field | Type | Default | Description |
|---|---|---|---|
| `status` (required) | string |  | ok when the model is loaded, loading before (and while its worker restarts). |
| `live_rows` | integer or null |  | Requests being answered now. |
| `waiting` | integer |  | Requests waiting for a slot. |
| `cache_bytes` | integer |  | Bytes the prompt cache holds in host memory: host_cache_bytes plus snapshot_bytes, each copy counted once. |
| `host_cache_bytes` | integer |  | Bytes of the host memory prompt cache, each copy counted once. |
| `snapshot_bytes` | integer |  | Bytes of saved slot state points that host_cache_bytes does not already count. |
| `host_cache_cap_bytes` | integer |  | Most bytes the host memory prompt cache holds (0 = off). |
| `host_cache_evictions` | integer |  | Prompt cache entries removed from host memory to stay under that cap. |
| `cache_namespace` | string or null |  | Id of what cached prompts are valid for: model, weights, tokenizer, chat template and cache policy. |
| `deployment_id` | string or null |  | Id of the compiled programs this server runs (null until set). |
| `component_set_sha` | string or null |  | Id of the model's component set (null until set). |
| `certification` | string |  | Whether this server's build passed the certification gate on its chip. |
| `compiles_after_ready` | integer or null |  | Programs compiled since the server reported ready, in every worker process it has run (0 when the start-up warm-up compiled every program the requests ran; null before ready or where this process does not count them). |
| `routed_experts` | object or null |  | Mixture-of-experts models: the path the routed experts run on, read where the engine runs. path: kernel (the kernel package's routed-expert kernel at every call site), exact (the exact path) or mixed; kernel: that kernel's id, version and package (null when the package has none); sites: the routed-op call sites traced on each path; exact_why: why any site ran exact. Null for a dense model and before the first program is traced. Fields: `path`, `kernel`, `sites`, `exact_why`. |
| `worker_pid` | integer |  | Process id of the model's worker, when the engine runs in a worker process. |
| `worker_restarts` | integer |  | Times the worker was started again after a fault. |

### Model

| Field | Type | Default | Description |
|---|---|---|---|
| `id` (required) | string |  | Model id, as in models.json. |
| `object` (required) | string |  | model |
| `owned_by` (required) | string |  | mashbox-labs |
| `created` (required) | integer |  | Always 0. |
| `max_model_len` (required) | integer |  | Context length the server runs (--ctx). |
| `weights` | string |  | Weights file loaded. |
| `gb` | number |  | Weights in device memory, GB. |
| `bpw` | number |  | Bits per weight of the weights file. |
| `prefix_cache` | boolean |  | Whether the server reuses cached prompt starts. |
| `slots` | integer |  | Requests decoded together (--slots). |
| `cache_ram_mib` | integer |  | Host memory for saved prompt states (--cache-ram). |

### ModelList

| Field | Type | Default | Description |
|---|---|---|---|
| `object` (required) | string |  | list |
| `data` (required) | array of Model |  |  |

### Error

OpenAI error shape (all routes except /v1/messages*).

| Field | Type | Default | Description |
|---|---|---|---|
| `error` (required) | object |  |  Fields: `message`, `type`. |

### AnthropicError

Anthropic error shape (/v1/messages and /v1/messages/count_tokens).

| Field | Type | Default | Description |
|---|---|---|---|
| `type` (required) | string |  | error |
| `error` (required) | object |  |  Fields: `type`, `message`. |

### ChatMessage

| Field | Type | Default | Description |
|---|---|---|---|
| `role` (required) | string |  | system and developer join the system prompt. |
| `content` | string or array or null |  | Text, or a list of text parts. Text parts only: image, audio and file parts are refused with 400 (text-only model). |
| `tool_calls` | array of object |  | assistant: the calls it made. |
| `tool_call_id` | string |  | tool: the call this answers. |
| `reasoning_content` | string |  | assistant: its earlier thinking. |
| `name` | string |  | Accepted and ignored. |

### ChatCompletion

| Field | Type | Default | Description |
|---|---|---|---|
| `id` (required) | string |  | chatcmpl-... |
| `object` (required) | string |  | chat.completion |
| `created` (required) | integer |  | Unix time. |
| `model` (required) | string |  | Model id. |
| `choices` (required) | array of object |  |  |
| `usage` (required) | ChatUsage |  |  |
| `timings` | ChatTimings |  |  |

### ChatTimings

Server timings, the fields of llama-server's timings block.

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt_n` (required) | integer |  | Prompt tokens prefilled (not served from the prefix cache). |
| `prompt_ms` (required) | number |  | Milliseconds from the start of the run to the first new token. |
| `prompt_per_second` | number or null |  | prompt_n per second of prompt_ms. |
| `predicted_n` (required) | integer |  | New tokens. |
| `predicted_ms` (required) | number |  | Milliseconds from the first to the last new token. |
| `predicted_per_second` | number or null |  | Decode speed: new tokens after the first, per second of predicted_ms. |
| `cache_n` (required) | integer |  | Prompt tokens served from the prefix cache. |
| `queue_ms` (required) | number |  | Milliseconds the request waited before it ran. |

### ChatUsage

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt_tokens` (required) | integer |  | Prompt tokens. |
| `completion_tokens` (required) | integer |  | New tokens. |
| `total_tokens` (required) | integer |  | Sum. |
| `prompt_tokens_details` | object |  |  Fields: `cached_tokens`. |

### ChatCompletionChunk

One `data:` line. The stream ends with `data: [DONE]`. With stream_options.include_usage the last chunk has empty choices and usage. The chunk with the finish_reason, and the usage chunk, carry timings.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` (required) | string |  | Same id on every chunk. |
| `object` (required) | string |  | chat.completion.chunk |
| `created` (required) | integer |  | Unix time. |
| `model` (required) | string |  | Model id. |
| `choices` (required) | array of object |  |  |
| `usage` | ChatUsage |  |  |
| `timings` | ChatTimings |  |  |

### Completion

Whole answer, or one `data:` line of a stream (no usage); the stream ends with `data: [DONE]`.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` (required) | string |  | cmpl-... |
| `object` (required) | string |  | text_completion |
| `created` (required) | integer |  | Unix time. |
| `model` (required) | string |  | Model id. |
| `choices` (required) | array of object |  |  |
| `usage` | object |  |  Fields: `prompt_tokens`, `completion_tokens`, `total_tokens`. |
| `system_fingerprint` | string |  | seed-<seed used>. |

### Message

| Field | Type | Default | Description |
|---|---|---|---|
| `id` (required) | string |  | msg_... |
| `type` (required) | string |  | message |
| `role` (required) | string |  | assistant |
| `model` (required) | string |  | The model field of the request, else the model id. |
| `content` (required) | array of ContentBlock |  |  |
| `stop_reason` (required) | string or null |  | end_turn, max_tokens, stop_sequence or tool_use; null in message_start. |
| `stop_sequence` (required) | string or null |  | The stop sequence that ended the text. |
| `usage` (required) | MessageUsage |  |  |

### ContentBlock

| Field | Type | Default | Description |
|---|---|---|---|
| `type` (required) | string |  | text, thinking or tool_use. |
| `text` | string |  | text block. |
| `thinking` | string |  | thinking block. |
| `signature` | string |  | thinking block: mashbox-local. |
| `id` | string |  | tool_use: toolu_... |
| `name` | string |  | tool_use: tool name. |
| `input` | object |  | tool_use: arguments. |

### MessageUsage

| Field | Type | Default | Description |
|---|---|---|---|
| `input_tokens` | integer |  | Prompt tokens not served from the prefix cache. |
| `cache_read_input_tokens` | integer |  | Prompt tokens served from the prefix cache. |
| `cache_creation_input_tokens` | integer |  | Always 0. |
| `output_tokens` | integer |  | New tokens. |

### MessageStreamEvent

One `event:` + `data:` pair: message_start, ping, then per block content_block_start / _delta / _stop, then message_delta and message_stop.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` (required) | string |  | Same as the SSE event name. |
| `message` | Message |  |  |
| `index` | integer |  | Content block number. |
| `content_block` | ContentBlock |  |  |
| `delta` | object |  | text_delta, thinking_delta, signature_delta, input_json_delta; on message_delta: stop_reason and stop_sequence. |
| `usage` | MessageUsage |  |  |
| `error` | object |  | On error: type and message. |

### TokenCount

| Field | Type | Default | Description |
|---|---|---|---|
| `input_tokens` (required) | integer |  | Prompt tokens after the chat template. |

### Response

| Field | Type | Default | Description |
|---|---|---|---|
| `id` (required) | string |  | resp_... |
| `object` (required) | string |  | response |
| `created_at` (required) | integer |  | Unix time. |
| `status` (required) | string |  | completed, or incomplete when max_output_tokens ran out. |
| `model` (required) | string |  | The model field of the request, else the model id. |
| `output` (required) | array of ResponseItem |  |  |
| `parallel_tool_calls` | boolean |  | From the request. |
| `tool_choice` | string or object |  | From the request. |
| `tools` | array |  | From the request. |
| `usage` (required) | object or null |  | input_tokens, input_tokens_details.cached_tokens, output_tokens, output_tokens_details.reasoning_tokens, total_tokens; null while in progress. |
| `incomplete_details` | object |  |  Fields: `reason`. |

### ResponseItem

| Field | Type | Default | Description |
|---|---|---|---|
| `type` (required) | string |  | message, reasoning, function_call or custom_tool_call. |
| `id` (required) | string |  | Item id. |
| `status` | string |  | in_progress or completed. |
| `role` | string |  | message: assistant. |
| `content` | array |  | message: output_text parts; reasoning: reasoning_text parts. |
| `summary` | array |  | reasoning: empty. |
| `call_id` | string |  | Call id to answer with an *_output item. |
| `name` | string |  | Tool name. |
| `arguments` | string |  | function_call: arguments, JSON. |
| `input` | string |  | custom_tool_call: raw input. |
| `namespace` | string |  | function_call: the namespace tool it came from. |

### ResponseStreamEvent

One `event:` + `data:` pair. An error after the stream started is sent as event `error`, then `response.failed` (the response with status failed and its error).

| Field | Type | Default | Description |
|---|---|---|---|
| `type` (required) | string |  | Same as the SSE event name. |
| `sequence_number` (required) | integer |  | 1, 2, 3, ... |
| `response` | Response |  |  |
| `item` | ResponseItem |  |  |
| `item_id` | string |  | Item the event belongs to. |
| `output_index` | integer |  | Item number. |
| `content_index` | integer |  | Part number. |
| `delta` | string |  | New text or arguments. |
| `text` | string |  | Whole text (done events). |
| `part` | object |  | Content part. |
| `arguments` | string |  | Whole arguments (done event). |
| `input` | string |  | Whole input (done event). |
