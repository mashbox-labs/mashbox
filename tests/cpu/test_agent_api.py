"""Coding-agent APIs (runtime/agents.py and the server routes) against a scripted engine: tool calls in the model's
format come back as OpenAI tool_calls, Anthropic tool_use blocks and Responses function_call items, streamed and not.
Needs the Qwen3.8-27B tokenizer and chat template: set MASHBOX_QWEN_UPSTREAM to a folder with tokenizer.json,
chat_template.jinja and generation_config.json (the model's upstream folder), otherwise the server tests skip."""
import json
import os
import socket
import time
import urllib.request

import pytest

from mashbox.runtime import agents as A
import api_stub_engine as STUB

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
        self.marks = marks
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
    from mashbox.runtime.server import Server
    chat = Chat(UP)
    eng = Script(chat, [])
    srv = Server(eng, chat, "qwen3-8-27b", {}, log=lambda m: None)
    srv.ready = True
    port = STUB.serve_on_free_port(srv)
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


def test_cache_points_at_user_message_starts(server):
    """the server offers the prefix cache the start of every user message and of the last message block"""
    eng, url = server
    eng.texts += ["ok"]
    msgs = [dict(role="system", content="be brief"), dict(role="user", content="one"), dict(role="assistant", content="1"),
            dict(role="user", content="two")]
    post(url + "/v1/chat/completions", dict(messages=msgs))
    ids, im = eng.prompts[-1], eng.chat.encode("<|im_start|>")[0]
    starts = [j for j in range(len(ids)) if ids[j] == im]
    users = [j for j in starts if eng.chat.decode(ids[j + 1:j + 2]) == "user"]
    assert len(users) == 2 and eng.marks[0] == sorted(set(users + [starts[-1]]))


# ---------------------------------------------------------------------------------------------- review fixes (10-28)
def test_tool_results_follow_call_order():
    """the template has no call ids: results sent in another order than the calls are put back in call order"""
    def tools_of(cv):
        return [m["content"] for m in cv.messages() if m["role"] == "tool"]
    use = [dict(type="tool_use", id="A", name="Bash", input=dict(command="a")),
           dict(type="tool_use", id="B", name="Bash", input=dict(command="b"))]
    res = [dict(type="tool_result", tool_use_id="B", content="out b"), dict(type="tool_result", tool_use_id="A", content="out a")]
    cv, _, _ = A.from_anthropic(dict(messages=[dict(role="user", content="go"), dict(role="assistant", content=use),
                                               dict(role="user", content=res)]))
    assert tools_of(cv) == ["out a", "out b"]
    assert not [k for m in cv.messages() for k in m if k.startswith("_")]
    cv, _, _ = A.from_openai_chat(dict(messages=[dict(role="user", content="go"), dict(role="assistant", content="", tool_calls=[
        dict(id="A", type="function", function=dict(name="Bash", arguments='{"command": "a"}')),
        dict(id="B", type="function", function=dict(name="Bash", arguments='{"command": "b"}'))]),
        dict(role="tool", tool_call_id="B", content="out b"), dict(role="tool", tool_call_id="A", content="out a")]))
    assert tools_of(cv) == ["out a", "out b"]
    cv, _, _, _, _ = A.from_responses(dict(input=[dict(type="message", role="user", content="go"),
                                                  dict(type="function_call", name="Bash", arguments='{"command": "a"}', call_id="A"),
                                                  dict(type="function_call", name="Bash", arguments='{"command": "b"}', call_id="B"),
                                                  dict(type="function_call_output", call_id="B", output="out b"),
                                                  dict(type="function_call_output", call_id="A", output="out a")]))
    assert tools_of(cv) == ["out a", "out b"]
    # results without ids, or naming an unknown call, keep the order the client sent
    cv, _, _ = A.from_openai_chat(dict(messages=[dict(role="user", content="go"), dict(role="assistant", content="", tool_calls=[
        dict(id="A", type="function", function=dict(name="Bash", arguments="{}")),
        dict(id="B", type="function", function=dict(name="Bash", arguments="{}"))]),
        dict(role="tool", content="first"), dict(role="tool", tool_call_id="A", content="second")]))
    assert tools_of(cv) == ["first", "second"]


def test_responses_stream_added_item_starts_empty(server):
    """an aggregator that starts from output_item.added and appends the deltas (as OpenAI's SDK does) gets valid JSON"""
    eng, url = server
    eng.texts += [CALL]
    tools = [dict(type="function", name="Bash", description="run", parameters=TOOLS[0]["function"]["parameters"])]
    ev = post(url + "/v1/responses", dict(model="qwen3-8-27b", stream=True, tools=tools, input="where am I?"), True)
    added = [d["item"] for n, d in ev if n == "response.output_item.added" and d["item"]["type"] == "function_call"]
    assert added and added[0]["arguments"] == ""
    agg = added[0]["arguments"] + "".join(d["delta"] for n, d in ev if n == "response.function_call_arguments.delta")
    done = [d["arguments"] for n, d in ev if n == "response.function_call_arguments.done"]
    assert json.loads(agg) == {"command": "pwd"} and done == [agg]


class Slow(Script):
    """Engine stand-in that decodes one token per step and ends the row when on_step returns it (as Engine._cancel)."""
    ended_at = None             # the server streams its first chunk (the role) before generate() starts

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None):
        self.prompts.append(prompts[0])
        self.cached_now, self.ended_at = [0], None
        full = self.chat.encode(self.texts.pop(0))[:max_new]
        fin = ["length"]
        for k in range(1, len(full) + 1):
            time.sleep(0.005)
            fin = ["stop" if len(full) < max_new else "length"] if k == len(full) else [None]
            if (on_step([full[:k]], fin) or []) and fin[0] is None:
                self.ended_at = k
                return dict(ids=[full[:k]], finish=["cancel"], cached=[0], timing={})
        return dict(ids=[full], finish=fin, cached=[0], timing={})


@pytest.fixture()
def slow_server():
    if not UP:
        pytest.skip("MASHBOX_QWEN_UPSTREAM not set")
    from mashbox.runtime.chat import Chat
    from mashbox.runtime.server import Server
    chat = Chat(UP)
    eng = Slow(chat, [])
    srv = Server(eng, chat, "qwen3-8-27b", {}, log=lambda m: None)
    srv.ready = True
    port = STUB.serve_on_free_port(srv)
    return eng, f"http://127.0.0.1:{port}", port


def test_stop_string_ends_generation(slow_server):
    eng, url, _ = slow_server
    eng.texts += ["alpha beta STOP" + " gamma" * 300]
    r = post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="hi")], stop=["STOP"]))
    assert r["choices"][0]["message"]["content"] == "alpha beta" and r["choices"][0]["finish_reason"] == "stop"
    assert eng.ended_at is not None and eng.ended_at < 20         # not the 300 tokens after the stop string


def test_client_disconnect_ends_generation(slow_server):
    eng, url, port = slow_server
    eng.texts += [" word" * 2000, "ok"]
    body = json.dumps(dict(model="qwen3-8-27b", max_tokens=4000, stream=True, messages=[dict(role="user", content="hi")])).encode()
    s_ = socket.create_connection(("127.0.0.1", port))
    s_.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: "
               + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while b"text_delta" not in got:
        got += s_.recv(4096)
    s_.close()                                     # the client hangs up mid-answer
    t = time.time()
    while eng.ended_at is None and time.time() - t < 5:
        time.sleep(0.05)
    assert eng.ended_at is not None and eng.ended_at < 1000      # ended soon after, not after all 2000 tokens
    r = post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="hi")]))      # the server still answers
    assert r["choices"][0]["message"]["content"] == "ok"


def test_client_half_close_ends_generation(slow_server):
    """Behind an SSH tunnel a stopped browser shows up as EOF on the server's socket while writes still succeed (the tunnel
    keeps taking bytes), so a failed write never comes. The server checks the socket itself, as llama-server does."""
    eng, url, port = slow_server
    eng.texts += [" word" * 2000, "ok"]
    body = json.dumps(dict(model="qwen3-8-27b", max_tokens=4000, stream=True, messages=[dict(role="user", content="hi")])).encode()
    s_ = socket.create_connection(("127.0.0.1", port))
    s_.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: "
               + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while b"content" not in got:
        part = s_.recv(4096)
        assert part, got                           # the server closed before streaming (a refused request)
        got += part
    s_.shutdown(socket.SHUT_WR)                    # EOF to the server, the socket stays open and readable
    t = time.time()
    while eng.ended_at is None and time.time() - t < 5:
        s_.settimeout(0.05)
        try:
            s_.recv(65536)
        except OSError:
            pass
    s_.close()
    assert eng.ended_at is not None and eng.ended_at < 1000      # ended soon after, not after all 2000 tokens
    r = post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="hi")]))
    assert r["choices"][0]["message"]["content"] == "ok" and r["timings"]["predicted_n"] == 1
