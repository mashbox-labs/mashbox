"""Coding-agent APIs (runtime/agents.py and the server routes) against a scripted engine: tool calls in the model's
format come back as OpenAI tool_calls, Anthropic tool_use blocks and Responses function_call items, streamed and not.
Needs the Qwen3.8-27B tokenizer and chat template: set MASHBOX_QWEN_UPSTREAM to a folder with tokenizer.json,
chat_template.jinja and generation_config.json (the model's upstream folder), otherwise the server tests skip."""
import json
import os
import socket
import threading
import time
import urllib.request

import pytest

from mashbox.runtime import agents as A

UP = os.environ.get("MASHBOX_QWEN_UPSTREAM")
TOOLS = [dict(type="function", function=dict(name="Bash", description="run", parameters=dict(
    type="object", properties=dict(command=dict(type="string"), timeout=dict(type="number")), required=["command"])))]


def test_parser_stream_split_anywhere():
    out = ("I will list the files.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n"
           "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>")
    for step in (1, 2, 3, 7, len(out)):
        p, ev = A.OutParser(TOOLS, False), []
        for i in range(0, len(out), step):
            ev += p.feed(out[i:i + step], final=i + step >= len(out))
        text = "".join(e[1] for e in ev if e[0] == "text")
        calls = [e[1:] for e in ev if e[0] == "tool"]
        assert text == "I will list the files.", (step, text)
        assert calls == [("Bash", {"command": "ls -la", "timeout": 30})], (step, calls)


def test_parser_thinking_and_multiline_string():
    out = "plan it\n</think>\n\n<tool_call>\n<function=Bash>\n<parameter=command>\necho a\necho b\n</parameter>\n</function>\n</tool_call>"
    p = A.OutParser(TOOLS, True)
    ev = []
    for ch in out:
        ev += p.feed(ch)
    ev += p.feed("", final=True)
    assert "".join(e[1] for e in ev if e[0] == "reasoning") == "plan it"
    assert [e[1:] for e in ev if e[0] == "tool"] == [("Bash", {"command": "echo a\necho b"})]
    assert not [e for e in ev if e[0] == "text"]


def test_inputs_refuse_images_and_server_tools():
    with pytest.raises(A.BadRequest, match="image"):
        A.from_anthropic(dict(messages=[dict(role="user", content=[dict(type="image", source={})])]))
    with pytest.raises(A.BadRequest, match="web_search"):
        A.from_anthropic(dict(messages=[dict(role="user", content="hi")], tools=[dict(type="web_search_20250305", name="web_search")]))


# ---------------------------------------------------------------------------------------------- server, scripted engine
class Script:
    """Engine stand-in: answers each prompt with the next scripted text (then the stop token)."""
    CTX, S, K, prefix_cache = 4096, 2, 4, True

    def __init__(self, chat, texts):
        self.chat, self.texts, self.prompts, self.cached_now = chat, list(texts), [], [0]

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None):
        self.prompts.append(prompts[0])
        self.cached_now = [0] * len(prompts)
        ids = [self.chat.encode(self.texts.pop(0))[:max_new] for _ in prompts]
        fin = ["stop" if len(i) < max_new else "length" for i in ids]
        for k in range(1, max(len(i) for i in ids) + 1, 3):
            on_step([i[:k] for i in ids], [None] * len(ids))
        on_step(ids, fin)
        return dict(ids=ids, finish=fin, cached=[0] * len(ids), timing={})


CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture()
def server():
    if not UP:
        pytest.skip("MASHBOX_QWEN_UPSTREAM not set")
    from mashbox.runtime.chat import Chat
    from mashbox.runtime.server import Server, serve
    chat = Chat(UP)
    eng = Script(chat, [])
    srv = Server(eng, chat, "qwen3-8-27b", {}, log=lambda m: None)
    srv.ready = True
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    threading.Thread(target=serve, args=(srv, "127.0.0.1", port, lambda m: None), daemon=True).start()
    time.sleep(0.3)
    return eng, f"http://127.0.0.1:{port}"


def post(url, body, stream=False):
    r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}))
    raw = r.read().decode()
    if not stream:
        return json.loads(raw)
    ev = []
    for blk in raw.split("\n\n"):
        name = next((ln[7:] for ln in blk.split("\n") if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in blk.split("\n") if ln.startswith("data: ")), None)
        if data and data != "[DONE]":
            ev.append((name, json.loads(data)))
    return ev


def test_openai_chat_tools(server):
    eng, url = server
    eng.texts += ["Checking.\n\n" + CALL, CALL, "Done: /home/u"]
    r = post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="where am I?")], tools=TOOLS))
    m = r["choices"][0]
    assert m["finish_reason"] == "tool_calls" and m["message"]["content"] == "Checking."
    assert json.loads(m["message"]["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd"}
    assert "# Tools" in eng.chat.decode(eng.prompts[-1])
    ev = post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="where?")], tools=TOOLS, stream=True,
                                                 stream_options=dict(include_usage=True)), True)
    tc = [d["choices"][0]["delta"]["tool_calls"][0] for _, d in ev if d.get("choices") and "tool_calls" in d["choices"][0]["delta"]]
    assert tc[0]["function"]["name"] == "Bash" and ev[-1][1]["usage"]["prompt_tokens"] > 0
    # the tool result goes back in and renders as a tool response
    post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="where?"), dict(role="assistant", content="", tool_calls=[
        dict(id="c1", type="function", function=dict(name="Bash", arguments='{"command": "pwd"}'))]),
        dict(role="tool", tool_call_id="c1", content="/home/u")], tools=TOOLS))
    p = eng.chat.decode(eng.prompts[-1])
    assert "<tool_response>\n/home/u\n</tool_response>" in p and "<parameter=command>\npwd\n</parameter>" in p


def test_anthropic_messages_stream(server):
    eng, url = server
    eng.texts += ["Let me look.\n\n" + CALL]
    tools = [dict(name="Bash", description="run", input_schema=TOOLS[0]["function"]["parameters"])]
    ev = post(url + "/v1/messages?beta=true", dict(model="qwen3-8-27b", max_tokens=32000, stream=True, tools=tools,
                                                   system=[dict(type="text", text="You are a coding agent.", cache_control=dict(type="ephemeral"))],
                                                   messages=[dict(role="user", content=[dict(type="text", text="where am I?")])]), True)
    names = [n for n, _ in ev]
    assert names[:2] == ["message_start", "ping"] and names[-2:] == ["message_delta", "message_stop"]
    starts = [d["content_block"] for n, d in ev if n == "content_block_start"]
    assert [b["type"] for b in starts] == ["text", "tool_use"] and starts[1]["name"] == "Bash"
    js = "".join(d["delta"]["partial_json"] for n, d in ev if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert json.loads(js) == {"command": "pwd"}
    assert ev[-2][1]["delta"]["stop_reason"] == "tool_use"
    n = post(url + "/v1/messages/count_tokens", dict(model="x", messages=[dict(role="user", content="hi")]))
    assert n["input_tokens"] > 0


def test_responses_stream(server):
    eng, url = server
    eng.texts += [CALL, "It is /home/u."]
    tools = [dict(type="function", name="Bash", description="run", parameters=TOOLS[0]["function"]["parameters"], strict=False)]
    ev = post(url + "/v1/responses", dict(model="qwen3-8-27b", instructions="You are Codex.", stream=True, tools=tools, input=[
        dict(type="message", role="developer", content=[dict(type="input_text", text="sandbox: none")]),
        dict(type="message", role="user", content=[dict(type="input_text", text="where am I?")])]), True)
    done = [d["item"] for n, d in ev if n == "response.output_item.done"]
    assert done[0]["type"] == "function_call" and json.loads(done[0]["arguments"]) == {"command": "pwd"}
    assert ev[-1][0] == "response.completed" and ev[-1][1]["response"]["usage"]["input_tokens"] > 0
    r = post(url + "/v1/responses", dict(model="qwen3-8-27b", tools=tools, input=[
        dict(type="message", role="user", content=[dict(type="input_text", text="where am I?")]),
        dict(type="function_call", name="Bash", arguments='{"command":"pwd"}', call_id="call_1"),
        dict(type="function_call_output", call_id="call_1", output="/home/u")]))
    assert r["output"][0]["content"][0]["text"] == "It is /home/u."
    assert "<tool_response>\n/home/u\n</tool_response>" in eng.chat.decode(eng.prompts[-1])
