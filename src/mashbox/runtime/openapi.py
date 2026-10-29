"""OpenAPI 3.1 description of the mashbox server (runtime/server.py): every route, request field and response shape.

One table: ROUTES lists what the server answers, SCHEMAS the request and response shapes. A running server returns
spec(server) at GET /openapi.json. docs/openapi.json and docs/server-api.md are generated from this module by
tools/hf_github_sync_check.py (render --write), and CI fails when they differ from it (check --offline).
tests/cpu/test_openapi.py starts the server and fails when it answers a route this table lacks, when this table lists
a route it does not answer, or when a request field it reads or a response field it sends is missing here.

    python3 -m mashbox.runtime.openapi          the spec (JSON)
    python3 -m mashbox.runtime.openapi --md     the Server API page (Markdown)
"""
import json
import re
import sys

from .server import KNOBS

GENERATED = "Generated from src/mashbox/runtime/openapi.py by tools/hf_github_sync_check.py render --write. Do not edit."


def ref(name):
    return {"$ref": "#/components/schemas/" + name}


def f(typ, desc, **kw):
    """One field. typ: a JSON type, a list of types, or a schema dict."""
    s = dict(typ) if isinstance(typ, dict) else {"type": typ}
    s["description"] = desc
    s.update(kw)
    return s


def obj(props, required=(), desc=None):
    o = {"type": "object", "properties": props}
    if required:
        o["required"] = list(required)
    if desc:
        o["description"] = desc
    return o


def arr(items, desc=None):
    a = {"type": "array", "items": items}
    if desc:
        a["description"] = desc
    return a


IGNORED = "Accepted and ignored."
TEXT_ONLY = "Text parts only: image, audio and file parts are refused with 400 (text-only model)."

# Sampling fields, read by knobs_of() in server.py on every generating route. The test checks these names equal KNOBS
# plus the three aliases below, so the spec cannot list a field the server does not read, or miss one it does.
SAMPLING = {
    "max_tokens": f("integer", "Most new tokens (>= 1).", minimum=1),
    "max_completion_tokens": f("integer", "Same as max_tokens (OpenAI's newer name); wins when both are set.", minimum=1),
    "temperature": f("number", "0 = greedy (default). Up to 4.", minimum=0, maximum=4),
    "top_p": f("number", "Nucleus sampling, in (0, 1].", maximum=1),
    "top_k": f("integer", "Keep the k most likely tokens; 0 = off.", minimum=0),
    "min_p": f("number", "Drop tokens below min_p times the top probability, in [0, 1).", minimum=0),
    "seed": f("integer", "Sampling seed; a random one when absent."),
    "stop": f(["string", "array"], "A string or a list of non-empty strings; the text ends before the first one."),
    "stop_sequences": f("array", "Same as stop (Anthropic's name), used when stop is absent.", items={"type": "string"}),
    "repetition_penalty": f("number", "> 0; 1 = off.", exclusiveMinimum=0),
    "frequency_penalty": f("number", "-2 to 2; 0 = off.", minimum=-2, maximum=2),
    "presence_penalty": f("number", "-2 to 2; 0 = off.", minimum=-2, maximum=2),
    "n": f("integer", "Must be 1.", enum=[1]),
}
for _k, _v in KNOBS.items():
    if _v is not None and _k != "max_tokens":
        SAMPLING[_k]["default"] = _v


def sampling(max_key, max_desc):
    """The sampling fields with the route's own length field first."""
    out = {max_key: f("integer", max_desc, minimum=1)} if max_key not in SAMPLING else {}
    for k, v in SAMPLING.items():
        out[k] = dict(v, description=max_desc) if k == max_key else v
    return out


TO_END = "Most new tokens; default: up to the end of the context (--ctx)."
STREAM = f("boolean", "Stream the answer as server-sent events (see the 200 text/event-stream response).", default=False)
MERGE = f("boolean", "false runs the request alone instead of sharing a batch with requests sent at the same time.",
          default=True)
CACHE_KEY = f("string", "Tags the request's session in the server log (as the X-Session-Id header does).")
MODEL = f("string", "Accepted for compatibility; the server answers with the one model it runs.")
THINK_KW = obj({"enable_thinking": f("boolean", "Think before answering (Qwen3 thinking mode).", default=False)},
               desc="Chat template options.")

FN_TOOL = obj({"type": f("string", "function", enum=["function"]),
               "function": obj({"name": f("string", "Tool name."), "description": f("string", "What the tool does."),
                                "parameters": f("object", "JSON Schema of the arguments.")}, ["name"])}, ["function"])
TOOL_CALL = obj({"id": f("string", "Call id."), "type": f("string", "function", enum=["function"]),
                 "function": obj({"name": f("string", "Tool name."),
                                  "arguments": f("string", "Arguments as a JSON string.")}, ["name", "arguments"])},
                ["id", "type", "function"])
CHOICE = f(["string", "object"], 'auto (default), none, required, or {"type": "function", "function": {"name": ...}}.')

SCHEMAS = {
    # ------------------------------------------------------------------ shared
    "Health": obj({"status": f("string", "ok when the model is loaded, loading before.", enum=["ok", "loading"])},
                  ["status"]),
    "Model": obj({"id": f("string", "Model id, as in models.json."), "object": f("string", "model", const="model"),
                  "owned_by": f("string", "mashbox-labs"), "created": f("integer", "Always 0."),
                  "max_model_len": f("integer", "Context length the server runs (--ctx)."),
                  "weights": f("string", "Weights file loaded."), "gb": f("number", "Weights in device memory, GB."),
                  "bpw": f("number", "Bits per weight of the weights file."),
                  "prefix_cache": f("boolean", "Whether the server reuses cached prompt starts."),
                  "slots": f("integer", "Requests decoded together (--slots)."),
                  "cache_ram_mib": f("integer", "Host memory for saved prompt states (--cache-ram).")},
                 ["id", "object", "owned_by", "created", "max_model_len"]),
    "ModelList": obj({"object": f("string", "list", const="list"), "data": arr(ref("Model"))}, ["object", "data"]),
    "Error": obj({"error": obj({"message": f("string", "What went wrong."),
                                "type": f("string", "invalid_request_error, unavailable or server_error.")},
                               ["message", "type"])}, ["error"], "OpenAI error shape (all routes except /v1/messages*)."),
    "AnthropicError": obj({"type": f("string", "error", const="error"),
                           "error": obj({"type": f("string", "invalid_request_error, overloaded_error or api_error."),
                                         "message": f("string", "What went wrong.")}, ["type", "message"])},
                          ["type", "error"], "Anthropic error shape (/v1/messages and /v1/messages/count_tokens)."),
    # ------------------------------------------------------------------ OpenAI chat
    "ChatMessage": obj({
        "role": f("string", "system and developer join the system prompt.",
                  enum=["system", "developer", "user", "assistant", "tool", "function"]),
        "content": f(["string", "array", "null"], "Text, or a list of text parts. " + TEXT_ONLY),
        "tool_calls": arr(TOOL_CALL, "assistant: the calls it made."),
        "tool_call_id": f("string", "tool: the call this answers."),
        "reasoning_content": f("string", "assistant: its earlier thinking."),
        "name": f("string", IGNORED)}, ["role"]),
    "ChatCompletionRequest": obj(dict({
        "model": MODEL, "messages": arr(ref("ChatMessage"), "The conversation."),
        "tools": arr(FN_TOOL, "Function tools the model may call."), "tool_choice": CHOICE,
        "stream": STREAM, "stream_options": obj({"include_usage": f("boolean", "Send a last chunk with usage.")}),
        "chat_template_kwargs": THINK_KW, "enable_thinking": f("boolean", "Same as chat_template_kwargs.enable_thinking."),
        "merge": MERGE, "prompt_cache_key": CACHE_KEY, "store": f("boolean", IGNORED)},
        **sampling("max_completion_tokens", "Most new tokens (>= 1); max_tokens is the same. Default: up to the end "
                                            "of the context (--ctx).")), ["messages"]),
    "ChatCompletion": obj({
        "id": f("string", "chatcmpl-..."), "object": f("string", "chat.completion", const="chat.completion"),
        "created": f("integer", "Unix time."), "model": f("string", "Model id."),
        "choices": arr(obj({"index": f("integer", "0."),
                            "message": obj({"role": f("string", "assistant"),
                                            "content": f(["string", "null"], "The answer (null when it only calls tools)."),
                                            "reasoning_content": f("string", "Thinking, when thinking is on."),
                                            "tool_calls": arr(TOOL_CALL)}, ["role", "content"]),
                            "finish_reason": f("string", "stop, length or tool_calls.", enum=["stop", "length", "tool_calls"])},
                           ["index", "message", "finish_reason"])),
        "usage": ref("ChatUsage"), "timings": ref("ChatTimings")}, ["id", "object", "created", "model", "choices", "usage"]),
    "ChatTimings": obj({"prompt_n": f("integer", "Prompt tokens prefilled (not served from the prefix cache)."),
                        "prompt_ms": f("number", "Milliseconds from the start of the run to the first new token."),
                        "prompt_per_second": f(["number", "null"], "prompt_n per second of prompt_ms."),
                        "predicted_n": f("integer", "New tokens."),
                        "predicted_ms": f("number", "Milliseconds from the first to the last new token."),
                        "predicted_per_second": f(["number", "null"], "Decode speed: new tokens after the first, per second "
                                                                      "of predicted_ms."),
                        "cache_n": f("integer", "Prompt tokens served from the prefix cache."),
                        "queue_ms": f("number", "Milliseconds the request waited before it ran.")},
                       ["prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n", "queue_ms"],
                       "Server timings, the fields of llama-server's timings block."),
    "ChatUsage": obj({"prompt_tokens": f("integer", "Prompt tokens."), "completion_tokens": f("integer", "New tokens."),
                      "total_tokens": f("integer", "Sum."),
                      "prompt_tokens_details": obj({"cached_tokens": f("integer", "Prompt tokens served from the prefix cache.")})},
                     ["prompt_tokens", "completion_tokens", "total_tokens"]),
    "ChatCompletionChunk": obj({
        "id": f("string", "Same id on every chunk."),
        "object": f("string", "chat.completion.chunk", const="chat.completion.chunk"),
        "created": f("integer", "Unix time."), "model": f("string", "Model id."),
        "choices": arr(obj({"index": f("integer", "0."),
                            "delta": obj({"role": f("string", "assistant, first chunk only."),
                                          "content": f("string", "New text."), "reasoning_content": f("string", "New thinking."),
                                          "tool_calls": arr(obj({"index": f("integer", "Call number."), "id": f("string", "Call id."),
                                                                 "type": f("string", "function"),
                                                                 "function": obj({"name": f("string", "Tool name."),
                                                                                  "arguments": f("string", "Whole arguments, JSON.")})}))}),
                            "finish_reason": f(["string", "null"], "null until the last chunk.")}, ["index", "delta"])),
        "usage": ref("ChatUsage"), "timings": ref("ChatTimings")}, ["id", "object", "created", "model", "choices"],
        "One `data:` line. The stream ends with `data: [DONE]`. With stream_options.include_usage the last chunk has "
        "empty choices and usage. The chunk with the finish_reason, and the usage chunk, carry timings."),
    # ------------------------------------------------------------------ OpenAI completions
    "CompletionRequest": obj(dict({
        "model": MODEL, "prompt": f(["string", "array"], "The text to continue (a string, or a list of one string)."),
        "stream": STREAM, "merge": MERGE, "prompt_cache_key": CACHE_KEY},
        **sampling("max_tokens", "Most new tokens (>= 1).")), ["prompt"]),
    "Completion": obj({
        "id": f("string", "cmpl-..."), "object": f("string", "text_completion", const="text_completion"),
        "created": f("integer", "Unix time."), "model": f("string", "Model id."),
        "choices": arr(obj({"index": f("integer", "0."), "text": f("string", "The continuation (in a stream: new text)."),
                            "finish_reason": f(["string", "null"], "stop or length; null until the last chunk.")},
                           ["index", "text", "finish_reason"])),
        "usage": obj({"prompt_tokens": f("integer", "Prompt tokens."), "completion_tokens": f("integer", "New tokens."),
                      "total_tokens": f("integer", "Sum.")}),
        "system_fingerprint": f("string", "seed-<seed used>.")}, ["id", "object", "created", "model", "choices"],
        "Whole answer, or one `data:` line of a stream (no usage); the stream ends with `data: [DONE]`."),
    # ------------------------------------------------------------------ Anthropic Messages
    "MessagesRequest": obj(dict({
        "model": MODEL, "messages": arr(obj({"role": f("string", "user or assistant (system also accepted).",
                                                         enum=["user", "assistant", "system", "developer"]),
                                             "content": f(["string", "array"], "Text, or content blocks: text, thinking, "
                                                                              "tool_use, tool_result. " + TEXT_ONLY)},
                                            ["role", "content"]), "The conversation."),
        "system": f(["string", "array"], "System prompt: a string or text blocks."),
        "tools": arr(obj({"name": f("string", "Tool name."), "description": f("string", "What the tool does."),
                          "input_schema": f("object", "JSON Schema of the input."),
                          "type": f("string", "custom or absent; server tools (web search and others) are refused.")},
                         ["name"]), "Client tools the model may call."),
        "tool_choice": f("object", '{"type": "auto" | "any" | "tool" | "none", "name": ...}.'),
        "thinking": obj({"type": f("string", "enabled or adaptive turns thinking on.", enum=["enabled", "adaptive", "disabled"]),
                         "budget_tokens": f("integer", IGNORED)}),
        "stream": STREAM, "prompt_cache_key": CACHE_KEY,
        "metadata": f("object", IGNORED), "output_config": f("object", IGNORED)},
        **sampling("max_tokens", TO_END)), ["messages"]),
    "Message": obj({
        "id": f("string", "msg_..."), "type": f("string", "message", const="message"), "role": f("string", "assistant"),
        "model": f("string", "The model field of the request, else the model id."),
        "content": arr(ref("ContentBlock")),
        "stop_reason": f(["string", "null"], "end_turn, max_tokens, stop_sequence or tool_use; null in message_start.",
                         enum=["end_turn", "max_tokens", "stop_sequence", "tool_use", None]),
        "stop_sequence": f(["string", "null"], "The stop sequence that ended the text."),
        "usage": ref("MessageUsage")}, ["id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"]),
    "ContentBlock": obj({"type": f("string", "text, thinking or tool_use.", enum=["text", "thinking", "tool_use"]),
                         "text": f("string", "text block."), "thinking": f("string", "thinking block."),
                         "signature": f("string", "thinking block: mashbox-local."),
                         "id": f("string", "tool_use: toolu_..."), "name": f("string", "tool_use: tool name."),
                         "input": f("object", "tool_use: arguments.")}, ["type"]),
    "MessageUsage": obj({"input_tokens": f("integer", "Prompt tokens not served from the prefix cache."),
                         "cache_read_input_tokens": f("integer", "Prompt tokens served from the prefix cache."),
                         "cache_creation_input_tokens": f("integer", "Always 0."),
                         "output_tokens": f("integer", "New tokens.")}),
    "MessageStreamEvent": obj({
        "type": f("string", "Same as the SSE event name.",
                  enum=["message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop",
                        "message_delta", "message_stop", "error"]),
        "message": ref("Message"), "index": f("integer", "Content block number."), "content_block": ref("ContentBlock"),
        "delta": f("object", "text_delta, thinking_delta, signature_delta, input_json_delta; on message_delta: "
                             "stop_reason and stop_sequence."),
        "usage": ref("MessageUsage"), "error": f("object", "On error: type and message.")}, ["type"],
        "One `event:` + `data:` pair: message_start, ping, then per block content_block_start / _delta / _stop, then "
        "message_delta and message_stop."),
    "CountTokensRequest": obj({
        "model": MODEL, "messages": arr(f("object", "As in MessagesRequest.")), "system": f(["string", "array"], "As in MessagesRequest."),
        "tools": arr(f("object", "As in MessagesRequest.")), "tool_choice": f("object", "As in MessagesRequest."),
        "thinking": f("object", "As in MessagesRequest.")}, ["messages"]),
    "TokenCount": obj({"input_tokens": f("integer", "Prompt tokens after the chat template.")}, ["input_tokens"]),
    # ------------------------------------------------------------------ OpenAI Responses
    "ResponsesRequest": obj(dict({
        "model": MODEL,
        "input": f(["string", "array"], "A string, or items: message (system, developer, user, assistant), function_call, "
                                        "function_call_output, custom_tool_call, custom_tool_call_output, local_shell_call, "
                                        "local_shell_call_output, reasoning. " + TEXT_ONLY),
        "instructions": f("string", "System prompt."),
        "tools": arr(f("object", "function, custom (grammar given to the model as text) or namespace tools. Hosted tools "
                                 "(web_search, file_search, code_interpreter, image_generation, computer_use_preview, mcp, "
                                 "local_shell) are dropped: the model does not see them.")),
        "tool_choice": CHOICE, "parallel_tool_calls": f("boolean", "Echoed in the response.", default=False),
        "reasoning": obj({"effort": f("string", "Any value but none or minimal turns thinking on.")}),
        "stream": STREAM, "prompt_cache_key": CACHE_KEY,
        "store": f("boolean", IGNORED), "include": f("array", IGNORED), "client_metadata": f("object", IGNORED)},
        **sampling("max_output_tokens", TO_END)), ["input"]),
    "Response": obj({
        "id": f("string", "resp_..."), "object": f("string", "response", const="response"),
        "created_at": f("integer", "Unix time."),
        "status": f("string", "completed, or incomplete when max_output_tokens ran out.",
                    enum=["in_progress", "completed", "incomplete"]),
        "model": f("string", "The model field of the request, else the model id."),
        "output": arr(ref("ResponseItem")), "parallel_tool_calls": f("boolean", "From the request."),
        "tool_choice": f(["string", "object"], "From the request."), "tools": f("array", "From the request."),
        "usage": f(["object", "null"], "input_tokens, input_tokens_details.cached_tokens, output_tokens, "
                                       "output_tokens_details.reasoning_tokens, total_tokens; null while in progress."),
        "incomplete_details": obj({"reason": f("string", "max_output_tokens")})},
        ["id", "object", "created_at", "status", "model", "output", "usage"]),
    "ResponseItem": obj({
        "type": f("string", "message, reasoning, function_call or custom_tool_call.",
                  enum=["message", "reasoning", "function_call", "custom_tool_call"]),
        "id": f("string", "Item id."), "status": f("string", "in_progress or completed."), "role": f("string", "message: assistant."),
        "content": f("array", "message: output_text parts; reasoning: reasoning_text parts."),
        "summary": f("array", "reasoning: empty."), "call_id": f("string", "Call id to answer with an *_output item."),
        "name": f("string", "Tool name."), "arguments": f("string", "function_call: arguments, JSON."),
        "input": f("string", "custom_tool_call: raw input."), "namespace": f("string", "function_call: the namespace tool it came from.")},
        ["type", "id"]),
    "ResponseStreamEvent": obj({
        "type": f("string", "Same as the SSE event name.",
                  enum=["response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
                        "response.content_part.added", "response.content_part.done", "response.output_text.delta",
                        "response.output_text.done", "response.reasoning_text.delta", "response.reasoning_text.done",
                        "response.function_call_arguments.delta", "response.function_call_arguments.done",
                        "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done",
                        "response.completed", "response.incomplete"]),
        "sequence_number": f("integer", "1, 2, 3, ..."), "response": ref("Response"), "item": ref("ResponseItem"),
        "item_id": f("string", "Item the event belongs to."), "output_index": f("integer", "Item number."),
        "content_index": f("integer", "Part number."), "delta": f("string", "New text or arguments."),
        "text": f("string", "Whole text (done events)."), "part": f("object", "Content part."),
        "arguments": f("string", "Whole arguments (done event)."), "input": f("string", "Whole input (done event).")},
        ["type", "sequence_number"],
        "One `event:` + `data:` pair. An error mid-stream is sent as `data: {\"error\": ...}` then `data: [DONE]`."),
}


def ok(schema, stream=None, desc="OK"):
    content = {"application/json": {"schema": ref(schema)}}
    if stream:
        content["text/event-stream"] = {"schema": ref(stream)}
    return {"description": desc, "content": content}


def err(desc, anth=False):
    return {"description": desc, "content": {"application/json": {"schema": ref("AnthropicError" if anth else "Error")}}}


LOADING = {"description": "The model is still loading.", "content": {"application/json": {"schema": ref("Health")}}}

# Every route the server answers (ui off). method, path, operationId, API family, summary, request, responses.
ROUTES = [
    dict(method="get", path="/v1/health", id="health", family="mashbox", summary="Health",
         desc="200 when the model is loaded and the server takes requests, 503 while it loads. GET /health is the same.",
         responses={"200": ok("Health"), "503": LOADING}),
    dict(method="get", path="/health", id="healthShort", family="mashbox", summary="Health (short path)",
         desc="Same as GET /v1/health.", responses={"200": ok("Health"), "503": LOADING}),
    dict(method="get", path="/", id="root", family="mashbox", summary="Health, or the chat page",
         desc="Same as GET /v1/health. With `mashbox run <model> --ui` it is the MASHBOX Chat page instead.",
         responses={"200": ok("Health"), "503": LOADING}),
    dict(method="head", path="/v1/health", id="healthHead", family="mashbox", summary="Ready check without a body",
         desc="200 when ready, 503 while loading, no body. The server answers HEAD the same way on every path.",
         responses={"200": {"description": "Ready."}, "503": {"description": "Loading."}}),
    dict(method="get", path="/v1/models", id="listModels", family="openai", summary="List models",
         desc="The one model this server runs.", responses={"200": ok("ModelList")}),
    dict(method="get", path="/v1/models/{model}", id="getModel", family="openai", summary="Get a model",
         desc="The model, when {model} is its id; 404 otherwise.", params=["model"],
         responses={"200": ok("Model"), "404": err("Another model id.")}),
    dict(method="post", path="/v1/chat/completions", id="createChatCompletion", family="openai", summary="Chat completion",
         desc="OpenAI Chat Completions with function tools, thinking and streaming. Coding agents such as Pi use it.",
         request="ChatCompletionRequest", responses={"200": ok("ChatCompletion", "ChatCompletionChunk"),
                                                    "400": err("Bad request."), "503": err("The model is still loading.")}),
    dict(method="post", path="/v1/completions", id="createCompletion", family="openai", summary="Text completion",
         desc="Continue a plain prompt, no chat template.", request="CompletionRequest",
         responses={"200": ok("Completion", "Completion"), "400": err("Bad request."), "503": err("The model is still loading.")}),
    dict(method="post", path="/v1/messages", id="createMessage", family="anthropic", summary="Messages (Anthropic)",
         desc="Anthropic Messages with client tools, thinking and streaming, for Claude Code. "
              "The ?beta=true query Claude Code adds is accepted.",
         request="MessagesRequest", responses={"200": ok("Message", "MessageStreamEvent"), "400": err("Bad request.", True),
                                               "503": err("The model is still loading.", True)}),
    dict(method="post", path="/v1/messages/count_tokens", id="countTokens", family="anthropic", summary="Count tokens",
         desc="Prompt tokens a Messages request would use, after the chat template.", request="CountTokensRequest",
         responses={"200": ok("TokenCount"), "400": err("Bad request.", True), "503": err("The model is still loading.", True)}),
    dict(method="post", path="/v1/responses", id="createResponse", family="openai", summary="Responses (OpenAI)",
         desc="OpenAI Responses with function, custom and namespace tools, reasoning and streaming, for Codex.",
         request="ResponsesRequest", responses={"200": ok("Response", "ResponseStreamEvent"), "400": err("Bad request."),
                                                "503": err("The model is still loading.")}),
    dict(method="get", path="/openapi.json", id="openapi", family="mashbox", summary="This description",
         desc="The OpenAPI 3.1 description of this server, with the running model's id.",
         responses={"200": {"description": "OpenAPI 3.1 document.", "content": {"application/json": {"schema": {"type": "object"}}}}}),
]


def route(method, path):
    """The ROUTES entry a request goes to, and its path parameters; (None, {}) when there is none. runtime/server.py
    finds every route through this (as FastAPI dispatches from its routes table), so a path the server answers is a
    path listed here, and test_openapi.py fails when its handlers test a path any other way."""
    for r in ROUTES:
        if r["method"] == method:
            m = re.fullmatch(re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(r["path"])), path)
            if m:
                return r, m.groupdict()
    return None, {}


# One example request per generating route: the body docs/quickstart.md sends, and the same question on each API.
EXAMPLES = {
    "createChatCompletion": {"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Name three planets."}],
                             "max_tokens": 64},
    "createCompletion": {"prompt": "The three largest planets are", "max_tokens": 32},
    "createMessage": {"model": "qwen3-8-27b", "max_tokens": 64, "messages": [{"role": "user", "content": "Name three planets."}]},
    "countTokens": {"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "Name three planets."}]},
    "createResponse": {"model": "qwen3-8-27b", "input": "Name three planets.", "max_output_tokens": 64},
}


def spec(server=None, version=None):
    """The OpenAPI 3.1 document. server: a running runtime.server.Server (its model id goes in the description)."""
    if version is None:
        from .. import __version__ as version
    paths = {}
    for r in ROUTES:
        op = {"operationId": r["id"], "summary": r["summary"], "description": r["desc"], "tags": [r["family"]],
              "responses": r["responses"]}
        if r.get("params"):
            op["parameters"] = [{"name": p, "in": "path", "required": True, "schema": {"type": "string"}} for p in r["params"]]
        if r.get("request"):
            op["requestBody"] = {"required": True, "content": {"application/json": {
                "schema": ref(r["request"]), **({"example": EXAMPLES[r["id"]]} if r["id"] in EXAMPLES else {})}}}
        paths.setdefault(r["path"], {})[r["method"]] = op
    desc = ("The HTTP server `mashbox run <model>` starts on 127.0.0.1 (reach it from your machine with an SSH tunnel). "
            "It speaks the OpenAI Chat Completions, Completions and Responses APIs and the Anthropic Messages API, so "
            "OpenAI and Anthropic clients and coding agents work against it unchanged. On 127.0.0.1 no API key is checked "
            "and any key a client sends is ignored; with `--api-key-file` or `MASHBOX_API_KEY`, and always when `--host` is "
            "not loopback, `/v1` routes and `/openapi.json` need a key (`Authorization: Bearer <key>` or "
            "`x-api-key: <key>`; 401 without). POST bodies must be `Content-Type: application/json` (415 otherwise), "
            "and a request from a web page of another origin, or naming the server by a host name it does not answer "
            "to, gets 403. Fields a route does not list are accepted and ignored.")
    if server is not None:
        desc += f" This server runs {server.model_id}."
    return {"openapi": "3.1.0",
            "info": {"title": "mashbox server", "version": version, "description": desc},
            "servers": [{"url": "http://127.0.0.1:8000", "description": "`mashbox run <model>` (default `--port 8000`)"}],
            "tags": [{"name": "openai", "description": "OpenAI-compatible routes."},
                     {"name": "anthropic", "description": "Anthropic-compatible routes."},
                     {"name": "mashbox", "description": "Health and this description."}],
            "paths": paths, "components": {"schemas": SCHEMAS}}


# ---------------------------------------------------------------------------------------------- the Markdown page
def _type(s):
    if "$ref" in s:
        return s["$ref"].rsplit("/", 1)[1]
    t = s.get("type", "any")
    if t == "array" and isinstance(s.get("items"), dict):
        return f"array of {_type(s['items'])}"
    return " or ".join(t) if isinstance(t, list) else t


def _cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def _fields(schema):
    req = set(schema.get("required", ()))
    rows = ["| Field | Type | Default | Description |", "|---|---|---|---|"]
    for name, s in schema["properties"].items():
        d = s.get("description", "")
        if "properties" in s and "$ref" not in s:
            d += " Fields: " + ", ".join(f"`{k}`" for k in s["properties"]) + "."
        dflt = json.dumps(s["default"]) if "default" in s else ""
        rows.append(f"| `{name}`{' (required)' if name in req else ''} | {_cell(_type(s))} | {_cell(dflt)} | {_cell(d)} |")
    return "\n".join(rows)


def markdown(version=None, recorded=None):
    """docs/server-api.md: one section per route, as llama.cpp's server README. `recorded` = docs/api-examples.json
    (tools/record_api_examples.py on a real chip): each route then shows the answer it gave there."""
    sp = spec(version=version)
    out = [f"<!-- {GENERATED} -->", "# Server API", "",
           sp["info"]["description"], "",
           "The same description as an OpenAPI 3.1 file: [openapi.json](openapi.json) in these docs, or "
           "`GET /openapi.json` on a running server.", "",
           "| Method | Path | What it does |", "|---|---|---|"]
    for r in ROUTES:
        out.append(f"| {r['method'].upper()} | `{r['path']}` | [{r['summary']}](#{r['id'].lower()}) |")
    for r in ROUTES:
        op = sp["paths"][r["path"]][r["method"]]
        out += ["", f'<a id="{r["id"].lower()}"></a>', "", f"## {r['method'].upper()} {r['path']}", "", op["description"]]
        if r.get("request"):
            out += ["", "**Request body** (JSON):", "", _fields(SCHEMAS[r["request"]])]
            if r["id"] in EXAMPLES:
                body = json.dumps(EXAMPLES[r["id"]])
                out += ["", "```bash", f"curl -s http://127.0.0.1:8000{r['path']} -H 'Content-Type: application/json' \\",
                        f"  -d '{body}'", "```"]
        got = (recorded or {}).get("routes", {}).get(r["id"])
        if got:
            out += ["", f"Example answer from {recorded['model']} on a {recorded['chip']}, HTTP {got['status']}:", "",
                    "```json", json.dumps(got["response"], indent=1, ensure_ascii=False), "```"]
        out += ["", "**Responses:**", ""]
        for code, resp in r["responses"].items():
            kinds = []
            for ctype, c in (resp.get("content") or {}).items():
                s = c["schema"]
                kinds.append(f"`{ctype}` {_type(s)}" if "$ref" in s else f"`{ctype}`")
            out.append(f"- {code}: {resp['description']}" + (" " + ", ".join(kinds) if kinds else ""))
    out += ["", "## Schemas", ""]
    for name, s in SCHEMAS.items():
        if name.endswith("Request"):
            continue
        out += [f"### {name}", ""] + ([s["description"], ""] if s.get("description") else []) + [_fields(s), ""]
    return "\n".join(out).rstrip() + "\n"


def spec_text(version=None):
    """docs/openapi.json, byte for byte."""
    return json.dumps(spec(version=version), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    sys.stdout.write(markdown() if "--md" in sys.argv[1:] else spec_text())
