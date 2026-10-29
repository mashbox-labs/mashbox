"""Stop strings: matched on the detokenized text in the server (the front, when the engine runs in a worker process),
never a byte past the stop goes to the client, usage counts tokens up to the finish point, and the row ends at the
engine's next step. Checked on chat, completions and Anthropic messages, streamed and not, with the engine in this
process and in a worker process. The scripted engine answers "11 12 13 ... 50 " one token per step."""
import json
import os
import sys
import time
import urllib.request

import pytest

from mashbox.runtime import server as SV

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402

OUT = list(range(11, 51))


class IdChat:
    stop_ids = ()

    def encode(self, text):
        return [int(x) for x in str(text).split() if x.isdigit()]

    def decode(self, ids):
        return "".join(f"{int(t)} " for t in ids)

    def chat_ids(self, msgs, *a, **k):
        return self.encode(msgs[-1].get("content"))


@pytest.fixture(params=["in_process", "worker"])
def server(request, tmp_path):
    log = str(tmp_path / "engine.jsonl")
    kw = dict(S=2, step_s=0.04, outs=[OUT], log=log)
    if request.param == "in_process":
        srv = SV.Server(STUB.make(**kw), IdChat(), "stub-model", {}, log=lambda m: None)
        srv.ready = True
    else:
        from mashbox.runtime import front as F
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.abspath("src"), os.environ.get("PYTHONPATH", "")]),
                   MASHBOX_FAULT="", JAX_PLATFORMS="cpu")
        srv = F.Front(IdChat(), "stub-model", {}, "api_stub_engine:make", kw, env=env, log=lambda m: None)
    port = STUB.serve_on_free_port(srv)
    t = time.time()
    while not srv.ready and time.time() - t < 60:
        time.sleep(0.02)
    assert srv.ready, "the worker did not say ready in 60 s"
    yield port, log
    if hasattr(srv, "close"):                           # the worker process ends with the test
        srv.close()


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
    if not body.get("stream"):
        return json.loads(raw)
    ev = []
    for blk in raw.split("\n\n"):
        name = next((ln[7:] for ln in blk.split("\n") if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in blk.split("\n") if ln.startswith("data: ")), None)
        if data and data != "[DONE]":
            ev.append((name, json.loads(data)))
    return ev


def pieces(route, r, stream):
    """-> (text pieces sent, finish reason, completion tokens or None)."""
    if not stream:
        if route == "anthropic":
            return [b["text"] for b in r["content"] if b["type"] == "text"], (r["stop_reason"], r["stop_sequence"]), r["usage"]["output_tokens"]
        c = r["choices"][0]
        return [c["message"]["content"] if route == "chat" else c["text"]], c["finish_reason"], r["usage"]["completion_tokens"]
    if route == "anthropic":
        md = [e for n, e in r if n == "message_delta"][0]
        return ([e["delta"]["text"] for n, e in r if n == "content_block_delta" and e["delta"]["type"] == "text_delta"],
                (md["delta"]["stop_reason"], md["delta"]["stop_sequence"]), md["usage"]["output_tokens"])
    ch = [e["choices"][0] for _, e in r if e.get("choices")]
    fin = [c["finish_reason"] for c in ch if c.get("finish_reason")]
    us = [e["usage"] for _, e in r if e.get("usage")]
    txt = [(c.get("delta") or {}).get("content") if route == "chat" else c.get("text") for c in ch]
    return [t for t in txt if t], fin[-1] if fin else None, us[-1]["completion_tokens"] if us else None


def body(route, stream, stop, max_tokens=40):
    if route == "chat":
        return dict(model="stub-model", messages=[dict(role="user", content="1 2 3")], max_tokens=max_tokens, stop=stop,
                    stream=stream, stream_options=dict(include_usage=True) if stream else None)
    if route == "completions":
        return dict(model="stub-model", prompt="1 2 3", max_tokens=max_tokens, stop=stop, stream=stream)
    return dict(model="stub-model", messages=[dict(role="user", content="1 2 3")], max_tokens=max_tokens,
                stop_sequences=stop if isinstance(stop, list) else [stop], stream=stream)


PATH = dict(chat="/v1/chat/completions", completions="/v1/completions", anthropic="/v1/messages")


@pytest.mark.parametrize("route", ["chat", "completions", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_stop_across_tokens_sends_nothing_past_it_and_ends_the_row(server, route, stream):
    port, log = server
    r = post(port, PATH[route], body(route, stream, ["zz", "13 1"]))      # "13 1" spans tokens 13 and 14
    sent, fin, n_out = pieces(route, r, stream)
    text = "".join(sent)
    assert text == ("11 12 " if route == "completions" else "11 12"), (sent, text)   # chat output drops end spaces
    assert not any("13" in s for s in sent)                                  # no piece ever held text past the stop
    assert fin == (("stop_sequence", "13 1") if route == "anthropic" else "stop"), fin
    if n_out is not None:
        assert n_out == 4                                                    # tokens up to the finish point
    t = time.time()
    while not [e for e in STUB.read_log(log) if e["ev"] == "end"] and time.time() - t < 5:
        time.sleep(0.02)
    end = [e for e in STUB.read_log(log) if e["ev"] == "end"]
    assert end and end[-1]["fin"] == "cancel" and end[-1]["n"] <= 4 + 3, end   # retired within a few steps, not 40


@pytest.mark.parametrize("route", ["chat", "completions", "anthropic"])
def test_held_back_prefix_of_a_stop_is_sent_at_the_end(server, route):
    """The text that could start a stop string is held back while the answer runs; when the answer ends without the
    stop, it is all sent."""
    port, log = server
    r = post(port, PATH[route], body(route, True, "50 x"))
    sent, fin, _ = pieces(route, r, True)
    full = "".join(f"{t} " for t in OUT)
    assert "".join(sent) == (full if route == "completions" else full.rstrip())
    assert fin == (("max_tokens", None) if route == "anthropic" else "length")


def test_stop_in_the_first_token(server):
    port, log = server
    r = post(port, PATH["chat"], body("chat", True, "11"))
    sent, fin, n_out = pieces("chat", r, True)
    assert sent == [] and fin == "stop" and n_out == 1
