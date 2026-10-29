"""Subagents on one server: Claude Code and Codex subagents are separate conversations sent to the same endpoint, with
the parent's session id plus an agent header, and often a model id that is not ours (Claude Code sends haiku/fable ids or
full Claude ids from agent files). The server answers any model id, as llama-server does in single-model mode, and its
request log keeps the requested id and a per-subagent tag. Set MASHBOX_QWEN_UPSTREAM (tokenizer + chat template)."""
import hashlib
import json
import socket
import threading
import time
import urllib.request

import pytest

from test_agent_api import UP, Script


def tag(x):
    return hashlib.sha256(x.encode()).hexdigest()[:8]


@pytest.fixture()
def logged():
    if not UP:
        pytest.skip("MASHBOX_QWEN_UPSTREAM not set")
    from mashbox.runtime.chat import Chat
    from mashbox.runtime.server import Server, serve
    chat = Chat(UP)
    eng = Script(chat, [])
    lines = []
    srv = Server(eng, chat, "qwen3-8-27b", {}, log=lines.append)
    srv.ready = True
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    threading.Thread(target=serve, args=(srv, "127.0.0.1", port, lambda m: None), daemon=True).start()
    time.sleep(0.3)
    return eng, f"http://127.0.0.1:{port}", lines


def post(url, body, headers):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read().decode())


def requests_logged(lines):
    return [r for r in (json.loads(x) for x in lines) if r.get("event") == "request"]


def test_claude_code_subagents_any_model_id(logged):
    eng, url, lines = logged
    sid = "d6084c75-3a4a-43b6-8f33-645b62c2a878"
    eng.texts += ["parent", "sub one", "sub two", "sub three"]
    body = lambda m: dict(model=m, max_tokens=64, messages=[dict(role="user", content="find the version")])
    st, r = post(url + "/v1/messages?beta=true", body("qwen3-8-27b"), {"X-Claude-Code-Session-Id": sid})
    assert st == 200 and r["content"][0]["text"] == "parent"
    # ids Claude Code sent for subagents in a captured session: an unmapped alias and a full Claude id from an agent file
    for i, m in enumerate(["claude-fable-5-1", "claude-opus-5-5", "claude-haiku-4-5"], 1):
        st, r = post(url + "/v1/messages?beta=true", body(m), {"X-Claude-Code-Session-Id": sid, "x-claude-code-agent-id": f"a{i}"})
        assert st == 200 and r["model"] == m and r["content"][0]["text"].startswith("sub")
    log = requests_logged(lines)
    assert [x["model"] for x in log] == ["qwen3-8-27b", "claude-fable-5-1", "claude-opus-5-5", "claude-haiku-4-5"]
    assert {x["session"] for x in log} == {tag(sid)}                       # one client session ...
    assert [x["agent"] for x in log] == [None, tag("a1"), tag("a2"), tag("a3")]  # ... four conversations


def test_codex_subagent_tagged(logged):
    eng, url, lines = logged
    parent = "01a181b1-76a2-7291-9217-279576384901"
    eng.texts += ["main", "spawned"]
    # as captured from Codex 0.165: the spawned agent sends the parent's session-id and prompt_cache_key
    body = dict(model="qwen3-8-27b", input="count the tests", prompt_cache_key=parent)
    st, _ = post(url + "/v1/responses", body, {"session-id": parent})
    assert st == 200
    st, _ = post(url + "/v1/responses", body, {"session-id": parent, "x-openai-subagent": "collab_spawn",
                                               "x-codex-parent-thread-id": parent})
    assert st == 200
    log = requests_logged(lines)
    assert [x["agent"] for x in log] == [None, tag("collab_spawn")] and log[0]["session"] == log[1]["session"] == tag(parent)
    assert all(x["model"] == "qwen3-8-27b" and "queue_s" in x for x in log)


def test_no_model_field_is_fine(logged):
    eng, url, lines = logged
    eng.texts += ["ok"]
    st, r = post(url + "/v1/chat/completions", dict(messages=[dict(role="user", content="hi")]), {})
    assert st == 200 and r["choices"][0]["message"]["content"] == "ok"
    assert requests_logged(lines)[-1]["model"] is None and requests_logged(lines)[-1]["agent"] is None
