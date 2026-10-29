"""The CPU standards tier, coding-agent rows. Plumbing only: the API routes, tool-call
parsing, streaming and the pinned clients' contract. These rows never test model quality.

Scripted logits: the engine is the BSHJ fixture's real Engine (admit / step / cancel, slots, K/V cache, sampler) behind
the real server, with the forward's next-token logits replaced by a scripted logits table (tiny.ScriptedLogits,
tiny.agent_engine). Every answer is the scripted text
tiny.SCRIPT_TEXT: a short sentence, then one tool call in the template's format, cut into tokens inside every tag, so a
streamed answer's chunks split the tags. Nothing here is model behaviour.

ST-A1  scripted logits: the scripted tool call comes back as OpenAI chat `tool_calls` (finish_reason "tool_calls"),
       Responses `function_call` and Anthropic `tool_use` (stop_reason "tool_use"), streamed and not, with the
       sentence before it as the text and no tag text in any text delta (K = 1: every token is its own stream
       update, so the parser sees every cut inside a tag); and the same through the pinned clients
       (openai 3.40.0, anthropic 2.3.0): their parsers and stream helpers rebuild the same call
ST-A2  scripted logits: no text at or past a stop string, on chat completions, completions and Anthropic messages,
       streamed and not. The stop "ng " starts in the answer's first token ("Checking") and ends in its second, so the
       stream's first update ends on 2 of its 3 characters: the server must hold them back. The text is "Checki", the
       finish is "stop" / "stop_sequence", and usage counts the 2 tokens up to the finish point, never the tokens the
       engine decoded past the stop inside its device call; on the K = 1 engine (every token its own update) and the
       K = 4 one (tokens past the stop decoded in the same call)
"""
import json

import pytest

import tiny

pytestmark = pytest.mark.standards

NAME, ARGS = tiny.SCRIPT_CALL
TOOLS_OAI = [dict(type="function", function=dict(name=NAME, description="run a command", parameters=dict(
    type="object", properties=dict(command=dict(type="string")), required=["command"])))]
TOOLS_ANT = [dict(name=NAME, description="run a command", input_schema=TOOLS_OAI[0]["function"]["parameters"])]
TOOLS_RESP = [dict(type="function", name=NAME, description="run a command", parameters=TOOLS_OAI[0]["function"]["parameters"],
                   strict=False)]
ASK = "Where am I?"
PINNED = ("3.40.0", "2.3.0")


_PORTS = {}


def agent_port(K):
    """scripted logits: the BSHJ fixture's Engine (K tokens per device call) with a scripted logits table behind the real
    server, one per K in this process. K = 1: every token is its own stream update, so every cut inside a tag reaches
    the parser and the stop-string hold; K = 4: the engine decodes past a stop inside one call (usage must not count it)."""
    if K not in _PORTS:
        eng, chat, _ids = tiny.agent_engine(K=K)
        srv, port = tiny.serve_http(eng, chat)
        srv.ready = True
        _PORTS[K] = port
    return _PORTS[K]


def sse(raw):
    """[(event name or None, data dict)] of a server-sent event stream, without the final [DONE]"""
    out = []
    for blk in raw.decode().split("\n\n"):
        name = next((ln[7:] for ln in blk.split("\n") if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in blk.split("\n") if ln.startswith("data: ")), None)
        if data and data != "[DONE]":
            out.append((name, json.loads(data)))
    return out


def no_tag(texts):
    bad = [t for t in texts if "<" in t or ">" in t]
    assert not bad, ("scripted logits: tag text leaked into the text deltas", bad)


def ok(code, body):
    assert code == 200, (code, body[:500])
    return body


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
def test_a1_openai_chat_tool_calls_scripted_logits(stream):
    """ST-A1, scripted logits, OpenAI chat completions: the scripted call is one tool_calls entry (function name, JSON
    arguments, an id), finish_reason "tool_calls", the sentence is the content; streamed, the content deltas join to the
    sentence with no tag text and the tool_calls deltas join to the same call"""
    port = agent_port(1)
    body = dict(model="tiny", messages=[dict(role="user", content=ASK)], tools=TOOLS_OAI, max_tokens=64, stream=stream)
    code, raw, _h = tiny.post(port, "/v1/chat/completions", body)
    ok(code, raw)
    if not stream:
        m = json.loads(raw)["choices"][0]
        tc = m["message"]["tool_calls"]
        assert m["finish_reason"] == "tool_calls", ("finish_reason", m["finish_reason"])
        assert m["message"]["content"] == tiny.SCRIPT_SAYS, ("content", m["message"]["content"])
        assert len(tc) == 1 and tc[0]["type"] == "function" and tc[0]["id"], tc
        assert (tc[0]["function"]["name"], json.loads(tc[0]["function"]["arguments"])) == (NAME, ARGS), tc
        return
    ch = [d for _e, d in sse(raw) if d.get("choices")]
    deltas = [c["choices"][0]["delta"] for c in ch]
    texts = [d["content"] for d in deltas if d.get("content")]
    no_tag(texts)
    assert "".join(texts) == tiny.SCRIPT_SAYS, ("streamed content", texts)
    calls = [t for d in deltas for t in d.get("tool_calls") or []]
    assert calls and {t["index"] for t in calls} == {0}, calls
    name = "".join(t["function"].get("name") or "" for t in calls)
    args = json.loads("".join(t["function"].get("arguments") or "" for t in calls))
    assert (name, args) == (NAME, ARGS), calls
    assert [c["choices"][0]["finish_reason"] for c in ch if c["choices"][0].get("finish_reason")] == ["tool_calls"]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
def test_a1_anthropic_tool_use_scripted_logits(stream):
    """ST-A1, scripted logits, Anthropic messages: a text block (the sentence) then one tool_use block (name, input,
    an id), stop_reason "tool_use"; streamed, the block starts are text then tool_use, the text deltas join to the
    sentence with no tag text and the input_json_delta parts join to the same input"""
    port = agent_port(1)
    body = dict(model="tiny", messages=[dict(role="user", content=ASK)], tools=TOOLS_ANT, max_tokens=64, stream=stream)
    code, raw, _h = tiny.post(port, "/v1/messages", body)
    ok(code, raw)
    if not stream:
        m = json.loads(raw)
        assert m["stop_reason"] == "tool_use", ("stop_reason", m["stop_reason"])
        assert [b["type"] for b in m["content"]] == ["text", "tool_use"], m["content"]
        assert m["content"][0]["text"] == tiny.SCRIPT_SAYS, m["content"][0]
        tu = m["content"][1]
        assert (tu["name"], tu["input"]) == (NAME, ARGS) and tu["id"], tu
        return
    ev = sse(raw)
    starts = [d["content_block"] for e, d in ev if e == "content_block_start"]
    assert [b["type"] for b in starts] == ["text", "tool_use"] and starts[1]["name"] == NAME, starts
    texts = [d["delta"]["text"] for e, d in ev if e == "content_block_delta" and d["delta"]["type"] == "text_delta"]
    no_tag(texts)
    assert "".join(texts) == tiny.SCRIPT_SAYS, ("streamed text", texts)
    parts = [d["delta"]["partial_json"] for e, d in ev if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
    assert json.loads("".join(parts)) == ARGS, parts
    assert [d["delta"]["stop_reason"] for e, d in ev if e == "message_delta"] == ["tool_use"]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
def test_a1_responses_function_call_scripted_logits(stream):
    """ST-A1, scripted logits, OpenAI Responses: the output is a message with the sentence, then one function_call item
    (name, JSON arguments, a call_id), status completed; streamed, the output_text deltas join to the sentence with no
    tag text, the function_call_arguments deltas join to the same arguments, and response.completed carries it all"""
    port = agent_port(1)
    body = dict(model="tiny", input=ASK, tools=TOOLS_RESP, max_output_tokens=64, stream=stream)
    code, raw, _h = tiny.post(port, "/v1/responses", body)
    ok(code, raw)
    if stream:
        ev = sse(raw)
        texts = [d["delta"] for e, d in ev if e == "response.output_text.delta"]
        no_tag(texts)
        assert "".join(texts) == tiny.SCRIPT_SAYS, ("streamed text", texts)
        args = "".join(d["delta"] for e, d in ev if e == "response.function_call_arguments.delta")
        assert json.loads(args) == ARGS, args
        done = [d["response"] for e, d in ev if e == "response.completed"]
        assert len(done) == 1, [e for e, _d in ev]
        r = done[0]
    else:
        r = json.loads(raw)
    assert r["status"] == "completed", r["status"]
    msg = [o for o in r["output"] if o["type"] == "message"]
    fc = [o for o in r["output"] if o["type"] == "function_call"]
    assert [c["text"] for o in msg for c in o["content"]] == [tiny.SCRIPT_SAYS], msg
    assert len(fc) == 1 and fc[0]["call_id"] and (fc[0]["name"], json.loads(fc[0]["arguments"])) == (NAME, ARGS), fc


def test_a1_pinned_clients_scripted_logits():
    """ST-A1, scripted logits, through the pinned clients (openai 3.40.0, anthropic 2.3.0): chat completions, Anthropic
    messages and Responses, each whole and through the client's stream helper, rebuild the scripted call. The clients
    must be installed at the pinned versions (a missing client is a FAIL here, never a skip)."""
    try:
        import anthropic
        import openai
    except ImportError as e:
        pytest.fail(f"ST-A1 needs the pinned clients openai {PINNED[0]} and anthropic {PINNED[1]} in the test env: {e}")
    assert (openai.__version__, anthropic.__version__) == PINNED, ("pinned clients", openai.__version__, anthropic.__version__)
    base = f"http://127.0.0.1:{agent_port(1)}"
    oc = openai.OpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=120)
    ac = anthropic.Anthropic(base_url=base, api_key="none", max_retries=0, timeout=120)
    msgs = [dict(role="user", content=ASK)]
    r = oc.chat.completions.create(model="tiny", messages=msgs, tools=TOOLS_OAI, max_tokens=64)
    with oc.chat.completions.stream(model="tiny", messages=msgs, tools=TOOLS_OAI, max_tokens=64) as s:
        f = s.get_final_completion()
    for c in (r.choices[0], f.choices[0]):
        tc = c.message.tool_calls
        assert c.finish_reason == "tool_calls" and c.message.content == tiny.SCRIPT_SAYS, c
        assert len(tc) == 1 and (tc[0].function.name, json.loads(tc[0].function.arguments)) == (NAME, ARGS), tc
    m = ac.messages.create(model="tiny", max_tokens=64, messages=msgs, tools=TOOLS_ANT)
    with ac.messages.stream(model="tiny", max_tokens=64, messages=msgs, tools=TOOLS_ANT) as s:
        fm = s.get_final_message()
    for x in (m, fm):
        assert x.stop_reason == "tool_use" and [b.type for b in x.content] == ["text", "tool_use"], x
        assert x.content[0].text == tiny.SCRIPT_SAYS and (x.content[1].name, x.content[1].input) == (NAME, ARGS), x
    rr = oc.responses.create(model="tiny", input=ASK, tools=TOOLS_RESP, max_output_tokens=64)
    with oc.responses.stream(model="tiny", input=ASK, tools=TOOLS_RESP, max_output_tokens=64) as s:
        fr = s.get_final_response()
    for x in (rr, fr):
        fc = [o for o in x.output if o.type == "function_call"]
        assert x.output_text == tiny.SCRIPT_SAYS, x.output_text
        assert len(fc) == 1 and (fc[0].name, json.loads(fc[0].arguments)) == (NAME, ARGS), fc


STOP = "ng "                                             # "Checki|ng" + " the folder.": across the first two tokens
STOP_TEXT = tiny.SCRIPT_SAYS[:tiny.SCRIPT_SAYS.find(STOP)]   # "Checki"
STOP_TOKENS = 2                                          # "Checking", " the folder.": the shortest prefix holding the stop


@pytest.mark.parametrize("K", [1, 4], ids=["K1", "K4"])
@pytest.mark.parametrize("route", ["chat", "completions", "anthropic"])
@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
def test_a2_stop_string_across_chunks_scripted_logits(route, stream, K):
    """ST-A2, scripted logits: a stop string split over two tokens (and two stream updates) cuts the text exactly: no
    character at or past it is sent, whole or streamed; finish "stop" (Anthropic: "stop_sequence" naming it); usage
    counts the tokens up to the finish point"""
    assert tiny.SCRIPT_PIECES[0].endswith(STOP[:2]) and tiny.SCRIPT_PIECES[1].startswith(STOP[2:]), "the stop must straddle"
    msgs = [dict(role="user", content=ASK)]
    if route == "chat":
        path, body = "/v1/chat/completions", dict(model="tiny", messages=msgs, max_tokens=64, stop=[STOP], stream=stream)
        if stream:
            body["stream_options"] = dict(include_usage=True)
    elif route == "completions":
        path, body = "/v1/completions", dict(model="tiny", prompt="<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                                             max_tokens=64, stop=[STOP], stream=stream)
        if stream:
            body["stream_options"] = dict(include_usage=True)
    else:
        path, body = "/v1/messages", dict(model="tiny", messages=msgs, max_tokens=64, stop_sequences=[STOP], stream=stream)
    code, raw, _h = tiny.post(agent_port(K), path, body)
    ok(code, raw)
    if not stream:
        r = json.loads(raw)
        if route == "anthropic":
            got, fin, n = "".join(b["text"] for b in r["content"]), (r["stop_reason"], r["stop_sequence"]), r["usage"]["output_tokens"]
        else:
            c = r["choices"][0]
            got, fin, n = c["message"]["content"] if route == "chat" else c["text"], c["finish_reason"], r["usage"]["completion_tokens"]
    else:
        ev = sse(raw)
        if route == "anthropic":
            deltas = [d["delta"]["text"] for e, d in ev if e == "content_block_delta" and d["delta"]["type"] == "text_delta"]
            md = [d for e, d in ev if e == "message_delta"]
            fin, n = (md[-1]["delta"]["stop_reason"], md[-1]["delta"]["stop_sequence"]), md[-1]["usage"]["output_tokens"]
        else:
            ch = [d for _e, d in ev if d.get("choices")]
            deltas = [(c["choices"][0]["delta"].get("content") if route == "chat" else c["choices"][0].get("text")) or "" for c in ch]
            fins = [c["choices"][0]["finish_reason"] for c in ch if c["choices"][0].get("finish_reason")]
            fin = fins[-1] if len(fins) == 1 else fins
            n = [d["usage"]["completion_tokens"] for _e, d in ev if d.get("usage")][-1]
        sent = ""
        for d in deltas:
            sent += d
            assert STOP_TEXT.startswith(sent), ("scripted logits: a streamed chunk went past the stop string", deltas)
        got = sent
    assert got == STOP_TEXT, ("scripted logits: text at or past the stop string was sent", route, got)
    assert fin == (("stop_sequence", STOP) if route == "anthropic" else "stop"), ("finish", fin)
    assert n == STOP_TOKENS, ("usage must count the tokens up to the finish point", n)
