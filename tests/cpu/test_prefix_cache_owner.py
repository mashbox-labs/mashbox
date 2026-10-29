"""Prefix cache owner rule: the cache key is (owner, token prefix) and the owner
is the tenant of the API key (an HMAC of the server's cache secret and the key, runtime/tenancy.py). A coding agent's parent and its subagents (Claude Code: same session id + x-claude-code-agent-id; Codex:
same session-id + x-openai-subagent) are one owner, so they keep reusing their shared system prompt and tools; another
key is another owner; without keys everything is one local owner. Session and agent headers never split the cache."""
import json
import urllib.request

from mashbox.runtime import server as SV
import api_stub_engine as STUB


class CharChat:
    stop_ids = (0,)

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)

    def chat_ids(self, messages, enable_thinking=False, tools=None):
        return self.encode(json.dumps([messages, tools]) + "\nassistant:\n")


class Eng:
    CTX, S, K, prefix_cache = 1 << 14, 2, 4, True

    def __init__(self):
        self.scopes = []

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, **kw):
        self.scopes.append(kw.get("scopes"))
        ids = [[ord(c) for c in "ok"]] * len(prompts)
        on_step(ids, ["stop"] * len(prompts))
        return dict(ids=ids, finish=["stop"] * len(prompts), cached=[0] * len(prompts), timing={})


def start(keys=None):
    eng = Eng()
    s = SV.Server(eng, CharChat(), "qwen3-8-27b", {}, log=lambda m: None)
    s.ready = True
    port = STUB.serve_on_free_port(s, keys=keys)
    return eng, port


def post(port, path, headers, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                               {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(r, timeout=20) as x:
        return x.status


MSG = dict(model="m", max_tokens=8, messages=[dict(role="user", content="hi")])


def test_parent_and_subagents_on_one_key_share_the_cache_scope():
    eng, port = start(keys=["key-a", "key-b"])
    a = {"Authorization": "Bearer key-a"}
    assert post(port, "/v1/messages", {**a, "X-Claude-Code-Session-Id": "s1"}, MSG) == 200            # parent
    assert post(port, "/v1/messages", {**a, "X-Claude-Code-Session-Id": "s1",
                                       "x-claude-code-agent-id": "sub-1"}, MSG) == 200                 # its subagent
    assert post(port, "/v1/responses", {**a, "session_id": "c1", "x-openai-subagent": "review"},
                dict(model="qwen3-8-27b", input="hi", max_output_tokens=8)) == 200                     # a Codex subagent
    assert post(port, "/v1/messages", {"x-api-key": "key-b", "X-Claude-Code-Session-Id": "s1"}, MSG) == 200   # other owner
    from mashbox.runtime import tenancy as T
    a_id, b_id = T.current().tenant_id("key-a"), T.current().tenant_id("key-b")
    assert eng.scopes == [[a_id], [a_id], [a_id], [b_id]] and a_id != b_id


def test_no_key_is_one_local_owner():
    eng, port = start()
    assert post(port, "/v1/messages", {"X-Claude-Code-Session-Id": "s1"}, MSG) == 200
    assert post(port, "/v1/messages", {"X-Claude-Code-Session-Id": "s2", "x-claude-code-agent-id": "x"}, MSG) == 200
    assert eng.scopes == [None, None]                    # no scopes passed: the engine's single shared scope


def get(port, path, headers):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
    with urllib.request.urlopen(r, timeout=20) as x:
        return json.loads(x.read())


def test_health_load_fields_only_for_api_callers():
    """/health stays open (status only); a caller with a key (or any caller of a server without keys) also gets the
    load and cache fields checks read: live rows, waiting, cache bytes, the cache namespace, certification state."""
    _, port = start(keys=["key-a"])
    assert get(port, "/health", {}) == {"status": "ok"}
    h = get(port, "/health", {"Authorization": "Bearer key-a"})
    assert h["status"] == "ok" and h["waiting"] == 0 and h["certification"] == "not_certified"
    assert {"live_rows", "cache_bytes", "cache_namespace", "component_set_sha", "deployment_id"} <= set(h)
    _, port = start()
    assert "waiting" in get(port, "/health", {})
