"""The official clients through a long queue: openai-python 3.40.0 and anthropic-sdk-python
2.3.0, as pinned, each with its own default timeout and retries, wait behind a request that holds the server's one row
for QUEUE_S seconds: longer than any request waited in a saved v6e-1 load run (queue_s
at most 137.2 s over 14,132 requests, 1,439 of them over 100 s; the short probe's first output at 112.7 s).

A stream's 200 goes out with its first event, so the whole wait comes before the response headers. Each call must still
get its whole answer with one attempt: no client timeout, no client retry (the engine admits each call once). A client
whose own timeout is shorter than the queue gets its timeout error, and the server forgets its request at once: it stops
counting as waiting and never runs. With the engine in this process and behind a worker process. Needs the two clients
at the pinned versions; otherwise it skips. About 150 s per mode."""
import os
import sys
import threading
import time
import urllib.request

import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
if (openai.__version__, anthropic.__version__) != ("3.40.0", "2.3.0"):
    pytest.skip(f"pinned clients openai 3.40.0 / anthropic 2.3.0, found {openai.__version__} / {anthropic.__version__}",
                allow_module_level=True)

from mashbox.runtime import server as SV  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402

QUEUE_S = 140.0                 # > 137.2 s, the longest queue_s of the saved load run
ANSWER = list(range(11, 21))
TEXT = " ".join(str(t) for t in ANSWER)         # IdChat's text of ANSWER, as the routes end it (no trailing space)
MSG = [dict(role="user", content="1 2 3")]


def _up(mode, tmp_path):
    go = tmp_path / "go"
    kw = dict(S=1, step_s=0.05, outs=[ANSWER], log=str(tmp_path / "engine.jsonl"), hold=dict(after=1, path=str(go)))
    if mode == "in_process":
        srv = SV.Server(STUB.StubEngine(**kw), STUB.IdChat(), "m", {}, log=lambda m: None)
        srv.ready = True
    else:
        from mashbox.runtime import front as F
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.abspath("src"), os.environ.get("PYTHONPATH", "")]),
                   MASHBOX_FAULT="", JAX_PLATFORMS="cpu")
        srv = F.Front(STUB.IdChat(), "m", {}, "api_stub_engine:make", kw, env=env, log=lambda m: None)
    port = STUB.serve_on_free_port(srv)
    t = time.time()
    while not srv.ready and time.time() - t < 60:
        time.sleep(0.05)
    assert srv.ready, "the worker did not say ready in 60 s"
    return srv, port, go


def _until(cond, s):
    t = time.time()
    while not cond() and time.time() - t < s:
        time.sleep(0.05)
    return cond()


@pytest.mark.parametrize("mode", ["in_process", "worker"])
def test_pinned_clients_with_their_default_timeouts_get_whole_answers_through_a_long_queue(mode, tmp_path):
    srv, port, go = _up(mode, tmp_path)
    try:
        admits = lambda: sum(e.get("ev") == "admit" for e in STUB.read_log(str(tmp_path / "engine.jsonl")))  # noqa: E731
        blocker = {}

        def hold_the_row():                  # a streamed request that holds the one row until `go` exists
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=(
                b'{"model": "m", "stream": true, "max_tokens": 4, "messages": [{"role": "user", "content": "1"}]}'),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                blocker.update(status=r.status, body=r.read().decode())
        threading.Thread(target=hold_the_row, daemon=True).start()
        assert _until(lambda: admits() == 1, 20)

        base = f"http://127.0.0.1:{port}"
        oc = openai.OpenAI(base_url=base + "/v1", api_key="none")          # the clients' own timeout and retries
        ac = anthropic.Anthropic(base_url=base, api_key="none")
        calls = {
            "chat": lambda: oc.chat.completions.create(model="m", messages=MSG, max_tokens=64).choices[0].message.content,
            "chat_stream": lambda: "".join(ch.choices[0].delta.content or "" for ch in oc.chat.completions.create(
                model="m", messages=MSG, max_tokens=64, stream=True) if ch.choices),
            "responses": lambda: oc.responses.create(model="m", input="1 2 3", max_output_tokens=64).output_text,
            "responses_stream": lambda: "".join(e.delta for e in oc.responses.create(
                model="m", input="1 2 3", max_output_tokens=64, stream=True) if e.type == "response.output_text.delta"),
            "messages": lambda: "".join(b.text for b in ac.messages.create(model="m", max_tokens=64, messages=MSG).content
                                        if b.type == "text"),
            "messages_stream": lambda: "".join(e.delta.text for e in ac.messages.create(
                model="m", max_tokens=64, messages=MSG, stream=True)
                if e.type == "content_block_delta" and e.delta.type == "text_delta"),
        }
        short = openai.OpenAI(base_url=base + "/v1", api_key="none", timeout=5.0, max_retries=0)
        calls["short_timeout"] = lambda: "".join(ch.choices[0].delta.content or "" for ch in short.chat.completions.create(
            model="m", messages=MSG, max_tokens=64, stream=True) if ch.choices)
        res = {}

        def run(name, fn):
            t0 = time.monotonic()
            try:
                res[name] = ("ok", fn(), time.monotonic() - t0)
            except Exception as e:                       # noqa: BLE001 - recorded, checked below
                res[name] = ("error", f"{type(e).__name__}: {e}", time.monotonic() - t0)
        ths = [threading.Thread(target=run, args=kv, daemon=True) for kv in calls.items()]
        for th in ths:
            th.start()
        assert _until(lambda: srv.waiting == len(calls), 30), srv.waiting      # every call waits for the row
        t_queued = time.monotonic()
        assert _until(lambda: "short_timeout" in res, 30), res
        assert res["short_timeout"][0] == "error" and "Timeout" in res["short_timeout"][1], res["short_timeout"]
        assert _until(lambda: srv.waiting == len(calls) - 1, 2), srv.waiting   # its request stops counting at once
        time.sleep(max(0.0, t_queued + QUEUE_S - time.monotonic()))
        assert srv.waiting == len(calls) - 1 and admits() == 1 and not set(res) - {"short_timeout"}, (srv.waiting, res)
        go.touch()                                       # the row frees: the queue drains, one request at a time
        for th in ths:
            th.join(120)
        for name in calls:
            if name == "short_timeout":
                continue
            kind, got, secs = res[name]
            assert kind == "ok" and got.strip() == TEXT and secs >= QUEUE_S, (name, res[name])
        assert blocker.get("status") == 200 and "[DONE]" in blocker.get("body", ""), blocker
        assert admits() == 1 + len(calls) - 1, STUB.read_log(str(tmp_path / "engine.jsonl"))  # the blocker + each call once
                                                        # (no retry); never short_timeout
    finally:
        go.touch()
        if hasattr(srv, "close"):
            srv.close()
