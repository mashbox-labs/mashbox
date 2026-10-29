"""A stand-in OpenAI-compatible chat server for the chat module's tests (CPU only, no model).

It streams a fixed, clearly labelled test reply built from the request, so the tests can check what the page sent
(GET /stub/requests) and how it shows a stream: reasoning when thinking is on, Markdown with a code block, usage at
the end, max_tokens as a length stop, and a long slow reply ("long" in the message) for the stop button.
Never shown to anyone as a model's output.

    python tests/cpu/chat_stub.py --port 8911 [--model stub-model] [--ctx 4096]
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIRST_S, STEP_S = 0.15, 0.012


def reply_words(body):
    msgs = body.get("messages") or []
    last = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "") or ""
    n_user = sum(1 for m in msgs if m.get("role") == "user")
    knobs = {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                                  "frequency_penalty", "seed", "max_tokens", "stop") if k in body}
    if "long" in last.lower():
        return ["word%d " % i for i in range(600)]
    text = ("**Stub reply** to turn %d: %s\n\nSettings seen: `%s`\n\n```python\nprint(\"hello from the stub\")\n```\n\n"
            "- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n" % (n_user, last[:80], json.dumps(knobs, sort_keys=True)))
    return [w + " " for w in text.split(" ")]


def make(model="stub-model", ctx=4096):
    log = {"requests": [], "aborted": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/v1/models":
                return self.send(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "stub", "max_model_len": ctx,
                                                                   "slots": 8, "prefix_cache": True}]})
            if self.path == "/health":
                return self.send(200, {"status": "ok"})
            if self.path == "/stub/requests":
                return self.send(200, log)
            self.send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            log["requests"].append(b)
            if self.path != "/v1/chat/completions":
                return self.send(404, {"error": {"message": "not found"}})
            if b.get("temperature", 0) > 2.5:
                return self.send(400, {"error": {"message": "temperature must be 0..2.5 on this test server", "type": "invalid_request_error"}})
            words = reply_words(b)
            think = (b.get("chat_template_kwargs") or {}).get("enable_thinking")
            reasoning = ["Thinking ", "about ", "the ", "question. "] if think else []
            mx = b.get("max_tokens")
            finish = "stop"
            if mx is not None and len(reasoning) + len(words) > mx:
                words, finish = words[:max(0, mx - len(reasoning))], "length"
            prompt = sum(len(str(m.get("content", "")).split()) for m in b.get("messages", [])) + 5
            usage = {"prompt_tokens": prompt, "completion_tokens": len(reasoning) + len(words), "total_tokens": prompt + len(reasoning) + len(words),
                     "prompt_tokens_details": {"cached_tokens": prompt - 5 if len(b.get("messages", [])) > 2 else 0}}
            if not b.get("stream"):
                return self.send(200, {"id": "x", "object": "chat.completion", "model": model, "usage": usage,
                                       "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": "".join(words)}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            head = {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "model": model}

            def ev(o):
                self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
                self.wfile.flush()
            try:
                ev(dict(head, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
                time.sleep(FIRST_S)
                for w in reasoning:
                    ev(dict(head, choices=[{"index": 0, "delta": {"reasoning_content": w}, "finish_reason": None}]))
                    time.sleep(STEP_S)
                for i in range(0, len(words), 2):
                    ev(dict(head, choices=[{"index": 0, "delta": {"content": "".join(words[i:i + 2])}, "finish_reason": None}]))
                    time.sleep(STEP_S * (4 if len(words) > 500 else 1))
                ev(dict(head, choices=[{"index": 0, "delta": {}, "finish_reason": finish}]))
                if (b.get("stream_options") or {}).get("include_usage"):
                    ev(dict(head, choices=[], usage=usage))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log["aborted"] += 1

        def log_message(self, *a):
            pass
    return H, log


def serve(port, model="stub-model", ctx=4096, host="127.0.0.1"):
    H, log = make(model, ctx)
    httpd = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8911)
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args()
    httpd, _ = serve(a.port, a.model, a.ctx)
    print(f"stub on http://127.0.0.1:{a.port}", flush=True)
    threading.Event().wait()
