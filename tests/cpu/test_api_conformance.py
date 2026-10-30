"""The official clients against this server: openai-python 3.40.0 and anthropic-sdk-python 2.3.0, as pinned. Each call
goes through the client's own request building, response parsing, stream parsing and error mapping, so a field the
client reads differently from what we send fails here. Covered: model list and lookup, chat completions and text
completions (streamed and not, usage incl. the stream's usage chunk, stop and finish reasons), function tools (the
client's stream helper rebuilds the calls), Anthropic messages (text, tools, stop sequences, max tokens, token count, the
stream helper), Responses (text, function calls, stream helper), the error codes the clients map (400, 404, 503 / 529
with Retry-After and x-should-retry, 500 and the error event mid-stream), with the engine in this process and behind a
worker process.

Needs the Qwen3.8-27B tokenizer and chat template (MASHBOX_QWEN_UPSTREAM, as test_agent_api.py) and the two clients at
the pinned versions; otherwise it skips. The engine is the scripted one in api_stub_engine.py, so answers are exact."""
import json
import os
import sys
import time

import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
if (openai.__version__, anthropic.__version__) != ("3.40.0", "2.3.0"):
    pytest.skip(f"pinned clients openai 3.40.0 / anthropic 2.3.0, found {openai.__version__} / {anthropic.__version__}",
                allow_module_level=True)
UP = os.environ.get("MASHBOX_QWEN_UPSTREAM")
if not UP:
    pytest.skip("MASHBOX_QWEN_UPSTREAM not set", allow_module_level=True)

from mashbox.runtime import server as SV  # noqa: E402
from mashbox.runtime.chat import Chat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_stub_engine as STUB  # noqa: E402

MID = "qwen3-8-27b"
VOCAB = 1 << 20                                # the stand-in engine takes any id the real tokenizer makes
CHAT = Chat(UP)
HELLO = "Hello there, friend."
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
TOOLS_OAI = [dict(type="function", function=dict(name="Bash", description="run a command", parameters=dict(
    type="object", properties=dict(command=dict(type="string")), required=["command"])))]
TOOLS_ANT = [dict(name="Bash", description="run a command", input_schema=dict(
    type="object", properties=dict(command=dict(type="string")), required=["command"]))]
TOOLS_RESP = [dict(type="function", name="Bash", description="run a command", parameters=dict(
    type="object", properties=dict(command=dict(type="string")), required=["command"]))]
MSG = [dict(role="user", content="Say hello.")]


class Failing(STUB.StubEngine):
    """Raises from its device call after a few tokens (a device fault inside this process)."""

    def step(self, waiting=False):
        if any(r is not None and len(r["ids"]) >= 3 for r in self.rows):
            raise RuntimeError("injected device fault")
        return super().step(waiting)


FRONTS = []


@pytest.fixture(autouse=True)
def _close_fronts():
    """Each test's worker processes end with it (Front.close())."""
    yield
    while FRONTS:
        FRONTS.pop().close()


def up(texts, mode="in_process", ready=True, engine=None):
    outs = [CHAT.encode(t) for t in texts]
    if mode == "in_process":
        srv = SV.Server(engine or STUB.make(S=2, outs=outs, V=VOCAB), CHAT, MID, {}, log=lambda m: None)
        srv.ready = ready
    else:
        from mashbox.runtime import front as F
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.abspath("src"), os.environ.get("PYTHONPATH", "")]),
                   MASHBOX_FAULT="", JAX_PLATFORMS="cpu")
        srv = F.Front(CHAT, MID, {}, "api_stub_engine:make", dict(S=2, outs=outs, V=VOCAB), env=env, log=lambda m: None)
        FRONTS.append(srv)
    port = STUB.serve_on_free_port(srv)
    t = time.time()
    while ready and not srv.ready and time.time() - t < 60:
        time.sleep(0.02)
    assert srv.ready or not ready, "the worker did not say ready in 60 s"
    base = f"http://127.0.0.1:{port}"
    return (openai.OpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=30),
            anthropic.Anthropic(base_url=base, api_key="none", max_retries=0, timeout=30))


MODES = ["in_process", "worker"]


@pytest.mark.parametrize("mode", MODES)
def test_models_and_chat(mode):
    oc, _ = up([HELLO, HELLO, HELLO + " More words here.", HELLO], mode)
    assert [m.id for m in oc.models.list().data] == [MID] and oc.models.retrieve(MID).id == MID
    with pytest.raises(openai.NotFoundError):
        oc.models.retrieve("no-such-model")
    r = oc.chat.completions.create(model=MID, messages=MSG, max_tokens=64)
    assert r.object == "chat.completion" and r.id.startswith("chatcmpl-") and r.model == MID
    c = r.choices[0]
    assert c.message.role == "assistant" and c.message.content == HELLO and c.finish_reason == "stop"
    n = len(CHAT.encode(HELLO))
    assert r.usage.completion_tokens == n and r.usage.total_tokens == r.usage.prompt_tokens + n and r.usage.prompt_tokens > 0
    chunks = list(oc.chat.completions.create(model=MID, messages=MSG, max_tokens=64, stream=True,
                                             stream_options=dict(include_usage=True)))
    assert "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices) == HELLO
    assert [ch.choices[0].finish_reason for ch in chunks if ch.choices and ch.choices[0].finish_reason] == ["stop"]
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == n
    r = oc.chat.completions.create(model=MID, messages=MSG, max_tokens=3)
    assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 3
    r = oc.chat.completions.create(model=MID, messages=MSG, max_tokens=64, stop=[" there"])
    assert r.choices[0].message.content == "Hello" and r.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("mode", MODES)
def test_chat_tools_and_completions(mode):
    oc, _ = up([CALL, CALL, HELLO, HELLO], mode)
    r = oc.chat.completions.create(model=MID, messages=MSG, tools=TOOLS_OAI, max_tokens=64)
    tc = r.choices[0].message.tool_calls
    assert r.choices[0].finish_reason == "tool_calls" and len(tc) == 1 and tc[0].type == "function"
    assert tc[0].function.name == "Bash" and json.loads(tc[0].function.arguments) == {"command": "pwd"} and tc[0].id
    with oc.chat.completions.stream(model=MID, messages=MSG, tools=TOOLS_OAI, max_tokens=64) as s:
        final = s.get_final_completion()
    tc = final.choices[0].message.tool_calls
    assert final.choices[0].finish_reason == "tool_calls" and tc[0].function.name == "Bash"
    assert json.loads(tc[0].function.arguments) == {"command": "pwd"}
    r = oc.completions.create(model=MID, prompt="Say hello.", max_tokens=64)
    assert r.object == "text_completion" and r.choices[0].text == HELLO and r.choices[0].finish_reason == "stop"
    assert r.usage.completion_tokens == len(CHAT.encode(HELLO))
    got = list(oc.completions.create(model=MID, prompt="Say hello.", max_tokens=64, stream=True))
    assert "".join(ch.choices[0].text for ch in got if ch.choices) == HELLO


@pytest.mark.parametrize("mode", MODES)
def test_anthropic_messages(mode):
    _, ac = up([HELLO, HELLO, CALL, HELLO, HELLO], mode)
    m = ac.messages.create(model=MID, max_tokens=64, messages=MSG)
    assert m.type == "message" and m.role == "assistant" and m.id.startswith("msg_")
    assert [b.type for b in m.content] == ["text"] and m.content[0].text == HELLO and m.stop_reason == "end_turn"
    assert m.usage.output_tokens == len(CHAT.encode(HELLO)) and m.usage.input_tokens > 0
    with ac.messages.stream(model=MID, max_tokens=64, messages=MSG) as s:
        text = "".join(s.text_stream)
        fm = s.get_final_message()
    assert text == HELLO and fm.stop_reason == "end_turn" and fm.content[0].text == HELLO
    m = ac.messages.create(model=MID, max_tokens=64, messages=MSG, tools=TOOLS_ANT)
    tu = [b for b in m.content if b.type == "tool_use"]
    assert m.stop_reason == "tool_use" and tu[0].name == "Bash" and tu[0].input == {"command": "pwd"} and tu[0].id
    m = ac.messages.create(model=MID, max_tokens=64, messages=MSG, stop_sequences=[" there"])
    assert m.content[0].text == "Hello" and m.stop_reason == "stop_sequence" and m.stop_sequence == " there"
    m = ac.messages.create(model=MID, max_tokens=2, messages=MSG)
    assert m.stop_reason == "max_tokens" and m.usage.output_tokens == 2
    assert ac.messages.count_tokens(model=MID, messages=MSG).input_tokens > 0


@pytest.mark.parametrize("mode", MODES)
def test_responses(mode):
    oc, _ = up([HELLO, CALL, HELLO], mode)
    r = oc.responses.create(model=MID, input="Say hello.", max_output_tokens=64)
    assert r.status == "completed" and r.output_text == HELLO and r.usage.output_tokens == len(CHAT.encode(HELLO))
    r = oc.responses.create(model=MID, input="Say hello.", tools=TOOLS_RESP, max_output_tokens=64)
    fc = [o for o in r.output if o.type == "function_call"]
    assert fc and fc[0].name == "Bash" and json.loads(fc[0].arguments) == {"command": "pwd"} and fc[0].call_id
    with oc.responses.stream(model=MID, input="Say hello.", max_output_tokens=64) as s:
        fr = s.get_final_response()
    assert fr.status == "completed" and fr.output_text == HELLO


def test_error_codes_the_clients_map():
    oc, ac = up([HELLO])
    with pytest.raises(openai.BadRequestError):
        oc.chat.completions.create(model=MID, messages=MSG, temperature=9)
    with pytest.raises(anthropic.BadRequestError):
        ac.messages.create(model=MID, max_tokens=8, messages=[])
    oc, ac = up([HELLO], ready=False)                  # loading: 503 / 529, Retry-After, x-should-retry
    with pytest.raises(openai.InternalServerError) as e:
        oc.chat.completions.create(model=MID, messages=MSG)
    h = e.value.response.headers
    assert e.value.status_code == 503 and h.get("retry-after") and h.get("x-should-retry") == "true"
    with pytest.raises(anthropic.APIStatusError) as e:
        ac.messages.create(model=MID, max_tokens=8, messages=MSG)
    assert e.value.status_code == 529 and e.value.response.headers.get("x-should-retry") == "true"
    assert e.value.body["error"]["type"] == "overloaded_error"


def test_device_fault_mid_stream_raises_in_the_clients():
    """A failed device call after output: each client raises on the stream's error event (it never ends quietly)."""
    for route in ("chat", "anthropic", "responses"):
        oc, ac = up([], engine=Failing(S=1, V=VOCAB, outs=[CHAT.encode(HELLO + " More and more words follow here.")]))
        got = []
        if route == "chat":
            with pytest.raises(openai.APIError) as e:
                for ch in oc.chat.completions.create(model=MID, messages=MSG, max_tokens=64, stream=True):
                    got.append(ch.choices[0].delta.content or "" if ch.choices else "")
            assert "injected device fault" in str(e.value)
        elif route == "anthropic":
            with pytest.raises(anthropic.APIStatusError) as e:
                with ac.messages.stream(model=MID, max_tokens=64, messages=MSG) as s:
                    for t in s.text_stream:
                        got.append(t)
            assert e.value.body["error"]["type"] == "api_error" and e.value.body["request_id"].startswith("msg_")
        else:
            evs = list(oc.responses.create(model=MID, input="Say hello.", max_output_tokens=64, stream=True))
            assert [ev.type for ev in evs][-2:] == ["error", "response.failed"]
            assert evs[-1].response.status == "failed" and evs[-1].response.error.message
            got = [ev.delta for ev in evs if ev.type == "response.output_text.delta"]
        full = HELLO + " More and more words follow here."
        assert "".join(got) and full.startswith("".join(got)) and "".join(got) != full, (route, got)


# ---- the real engine (runtime/engine.py, continuous batching) on a tiny random Qwen3.8 model, in this process and
# behind a worker process. Token ids are the text (api_stub_engine.IdChat), so every answer can be checked exactly.
IDS = STUB.IdChat()
PROMPT = "12 7 301 44 9 250 3"
N = 16


@pytest.fixture(scope="module")
def tiny_wd(tmp_path_factory):
    pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5TextConfig"):
        pytest.skip("transformers without qwen3_5")
    return STUB.build_tiny(str(tmp_path_factory.mktemp("tiny_qwen")))


def up_real(wd, mode, faults=""):
    if mode == "in_process":
        srv = SV.Server(STUB.tiny(wd), IDS, MID, {}, log=lambda m: None)
        srv.ready = True
    else:
        from mashbox.runtime import front as F
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.abspath("src"), os.environ.get("PYTHONPATH", "")]),
                   MASHBOX_FAULT="", MASHBOX_FAULTS=faults, JAX_PLATFORMS="cpu",
                   MASHBOX_WATCHDOG_MIN_S="600")       # first calls compile on a loaded CPU; the watchdog is tested elsewhere
        srv = F.Front(IDS, MID, {}, "api_stub_engine:tiny", dict(wd=wd), env=env, log=lambda m: None)
        FRONTS.append(srv)
    port = STUB.serve_on_free_port(srv)
    t = time.time()
    while not srv.ready and time.time() - t < 300:
        time.sleep(0.05)
    assert srv.ready, "the worker did not say ready"
    base = f"http://127.0.0.1:{port}"
    return srv, port, openai.OpenAI(base_url=base + "/v1", api_key="none", max_retries=0, timeout=300), \
        anthropic.Anthropic(base_url=base, api_key="none", max_retries=0, timeout=300)


def answers(oc, ac, stop=None):
    """PROMPT on every route, greedy, N tokens (or up to `stop`), checked for the fields the clients read."""
    msgs, out = [dict(role="user", content=PROMPT)], {}
    fin = "stop" if stop else "length"
    r = oc.chat.completions.create(model=MID, messages=msgs, max_tokens=N, stop=stop)
    assert r.model == MID and r.choices[0].finish_reason == fin and r.usage.prompt_tokens == len(IDS.encode(PROMPT))
    out["chat"], out["chat_n"] = r.choices[0].message.content, r.usage.completion_tokens
    ch = list(oc.chat.completions.create(model=MID, messages=msgs, max_tokens=N, stop=stop, stream=True,
                                         stream_options=dict(include_usage=True)))
    out["chat_stream"] = "".join(c.choices[0].delta.content or "" for c in ch if c.choices)
    assert [c.choices[0].finish_reason for c in ch if c.choices and c.choices[0].finish_reason] == [fin]
    assert ch[-1].choices == [] and ch[-1].usage.completion_tokens == out["chat_n"]
    c = oc.completions.create(model=MID, prompt=PROMPT, max_tokens=N, stop=stop)
    assert c.choices[0].finish_reason == fin
    out["completions"], out["completions_n"] = c.choices[0].text, c.usage.completion_tokens
    out["completions_stream"] = "".join(x.choices[0].text for x in oc.completions.create(
        model=MID, prompt=PROMPT, max_tokens=N, stop=stop, stream=True) if x.choices)
    m = ac.messages.create(model=MID, max_tokens=N, messages=msgs, stop_sequences=[stop] if stop else None)
    assert m.stop_reason == ("stop_sequence" if stop else "max_tokens") and m.stop_sequence == stop
    out["anthropic"], out["anthropic_n"] = m.content[0].text if m.content else "", m.usage.output_tokens
    with ac.messages.stream(model=MID, max_tokens=N, messages=msgs, stop_sequences=[stop] if stop else None) as s:
        out["anthropic_stream"] = s.get_final_text()
    return out


@pytest.fixture(scope="module")
def ref(tiny_wd):
    """The answers of the real engine in this process (the one-process server)."""
    _, _, oc, ac = up_real(tiny_wd, "in_process")
    return answers(oc, ac)


def test_real_engine_behind_the_worker_answers_as_in_process(tiny_wd, ref):
    """The worker is invisible to clients: every route gives the in-process answer, greedy, run after run; streamed text
    equals the whole answer; usage counts N output tokens."""
    whole = ref["completions"]
    assert len(IDS.encode(whole)) == N and ref["completions_n"] == ref["chat_n"] == ref["anthropic_n"] == N
    assert ref["chat"] == ref["chat_stream"] == ref["anthropic"] == ref["anthropic_stream"] == whole.rstrip()
    assert ref["completions_stream"] == whole
    _, _, oc, ac = up_real(tiny_wd, "worker")
    assert answers(oc, ac) == ref
    assert answers(oc, ac) == ref                                   # single-user greedy: the same answer run to run


def test_real_engine_stop_string_cuts_exactly(tiny_wd, ref):
    """A stop string spanning two tokens: nothing at or past it is sent on any route (streamed or not), usage counts the
    tokens up to the finish point, and the answer is the same behind the worker."""
    whole = ref["completions"]
    t = whole.split()
    stop = f"{t[5]} {t[6]}"
    cut = whole[:whole.find(stop)]
    n = SV.stop_point(IDS, IDS.encode(whole), [stop])
    for mode in ("in_process", "worker"):
        _, _, oc, ac = up_real(tiny_wd, mode)
        a = answers(oc, ac, stop=stop)
        assert a["completions"] == a["completions_stream"] == cut, (mode, a)
        assert a["chat"] == a["chat_stream"] == a["anthropic"] == a["anthropic_stream"] == cut.rstrip(), (mode, a)
        assert a["chat_n"] == a["completions_n"] == a["anthropic_n"] == n, (mode, a)


def test_real_engine_device_fault_mid_stream_then_the_same_answer(tiny_wd, ref):
    """MASHBOX_FAULTS=device_error@4 (the 4th engine call): the stream has output, then the client raises on the error
    event; the worker restarts by itself and answers exactly as before the fault; /health counts the restart."""
    import urllib.request
    srv, port, oc, ac = up_real(tiny_wd, "worker", faults="device_error@4")
    pid0, got = srv.link.proc.pid, []
    with pytest.raises(openai.APIError):
        for c in oc.chat.completions.create(model=MID, messages=[dict(role="user", content=PROMPT)], max_tokens=N,
                                            stream=True):
            got.append(c.choices[0].delta.content or "" if c.choices else "")
    text = "".join(got)
    assert text and ref["chat"].startswith(text) and text != ref["chat"], text
    t = time.time()
    while not (srv.ready and srv.link is not None and srv.link.proc.pid != pid0) and time.time() - t < 300:
        time.sleep(0.05)
    assert answers(oc, ac) == ref
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as r:
        h = json.loads(r.read())
    assert h["worker_restarts"] == 1 and h["worker_pid"] == srv.link.proc.pid != pid0, h


def test_real_engine_compiles_after_ready_are_the_workers(tiny_wd):
    """/health compiles_after_ready is counted where the engine runs. This worker skips the warm-up, so the first request
    compiles its programs after ready inside the worker: /health goes up, while this process (the front) compiles
    nothing. A front reading its own count would report no change."""
    import urllib.request
    from mashbox.runtime import programs

    def health(port):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as r:
            return json.loads(r.read())
    srv, port, oc, ac = up_real(tiny_wd, "worker")
    here0, h0 = programs.counts()["compiles_after_ready"], health(port)["compiles_after_ready"]
    oc.chat.completions.create(model=MID, messages=[dict(role="user", content=PROMPT)], max_tokens=N)
    t = time.time()
    while health(port)["compiles_after_ready"] == h0 and time.time() - t < 30:
        time.sleep(0.05)
    assert health(port)["compiles_after_ready"] > h0 and programs.counts()["compiles_after_ready"] == here0


@pytest.mark.parametrize("mode", MODES)
def test_unknown_model_id_is_404_on_the_openai_routes(mode):
    """A model id other than the served one is 404 model_not_found on chat, completions and
    responses (before the request reaches the engine); no model field is accepted. The Anthropic routes answer any model
    name and echo it (Claude Code sends its own names for the main model and its subagents)."""
    oc, ac = up([HELLO] * 4, mode)
    for call in (lambda: oc.chat.completions.create(model="no-such-model", messages=MSG, max_tokens=8),
                 lambda: oc.completions.create(model="no-such-model", prompt="Hi", max_tokens=8),
                 lambda: oc.responses.create(model="no-such-model", input="Hi", max_output_tokens=8)):
        with pytest.raises(openai.NotFoundError) as e:
            call()
        assert e.value.code == "model_not_found" and "no-such-model" in e.value.message
    m = ac.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=MSG)
    assert m.model == "claude-haiku-4-5" and m.content[0].text == HELLO


@pytest.mark.parametrize("mode", MODES)
def test_logprobs_and_structured_output_are_refused_with_400_naming_the_field(mode):
    """Log probabilities and structured output are not built yet (not in the base): a request
    that asks for either gets 400 naming the field through the official clients, never an answer that silently lacks
    it. The plain forms still answer (logprobs false, response_format text, output_config effort only)."""
    oc, ac = up([HELLO] * 16, mode)
    schema = dict(type="object", properties=dict(a=dict(type="string")), required=["a"], additionalProperties=False)
    asks = [("logprobs", lambda: oc.chat.completions.create(model=MID, messages=MSG, max_tokens=8, logprobs=True)),
            ("top_logprobs", lambda: oc.chat.completions.create(model=MID, messages=MSG, max_tokens=8, top_logprobs=2)),
            ("response_format", lambda: oc.chat.completions.create(model=MID, messages=MSG, max_tokens=8,
                                                                   response_format=dict(type="json_object"))),
            ("response_format", lambda: oc.chat.completions.create(
                model=MID, messages=MSG, max_tokens=8,
                response_format=dict(type="json_schema", json_schema=dict(name="x", schema=schema, strict=True)))),
            ("logprobs", lambda: oc.completions.create(model=MID, prompt="Hi", max_tokens=8, logprobs=1)),
            ("text.format", lambda: oc.responses.create(model=MID, input="Hi", max_output_tokens=8, text=dict(
                format=dict(type="json_schema", name="x", schema=schema, strict=True)))),
            ("top_logprobs", lambda: oc.responses.create(model=MID, input="Hi", max_output_tokens=8, top_logprobs=2)),
            ("include", lambda: oc.responses.create(model=MID, input="Hi", max_output_tokens=8,
                                                    include=["message.output_text.logprobs"]))]
    for field, call in asks:
        with pytest.raises(openai.BadRequestError) as e:
            call()
        assert e.value.status_code == 400 and f"`{field}" in e.value.message and "not supported" in e.value.message, field
    with pytest.raises(anthropic.BadRequestError) as e:
        ac.messages.create(model=MID, max_tokens=8, messages=MSG,
                           output_config=dict(format=dict(type="json_schema", schema=schema)))
    assert e.value.status_code == 400 and "`output_config.format`" in e.value.message
    assert e.value.body["error"]["type"] == "invalid_request_error"
    r = oc.chat.completions.create(model=MID, messages=MSG, max_tokens=64, logprobs=False, response_format=dict(type="text"))
    assert r.choices[0].message.content == HELLO
    m = ac.messages.create(model=MID, max_tokens=64, messages=MSG, output_config=dict(effort="low"))
    assert m.content[0].text == HELLO
