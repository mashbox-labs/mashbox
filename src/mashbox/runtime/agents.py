"""Coding-agent APIs: tool calls in the model's own format, and the request and response shapes of three APIs.

  OpenAI chat completions with tools (Pi, Cursor, Aider, any OpenAI-compatible agent)
  Anthropic Messages, /v1/messages (Claude Code)
  OpenAI Responses, /v1/responses (Codex)

Every API is turned into one internal conversation (system, user, assistant with tool_calls, tool messages) that the
model's chat template renders, and the model's output is split back into reasoning, text and tool calls by OutParser.
Tool calls use the chat template's format: <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>.
Features the server cannot do (images, server-side tools such as web search) are refused with an exact error.
"""
import json
import re
import uuid


class BadRequest(Exception):
    pass


class TooLong(BadRequest):
    pass


def rid(prefix):
    return prefix + uuid.uuid4().hex[:24]


# ------------------------------------------------------------------------------------------------------------ tool calls
TC_OPEN, TC_CLOSE, TH_CLOSE = "<tool_call>", "</tool_call>", "</think>"
FN = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PR = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.S)


def _hold(buf, tag):
    """length of the longest end of buf that could be the start of tag"""
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


def _types(schema):
    t = (schema or {}).get("type")
    if t is None:
        t = [x.get("type") for x in (schema or {}).get("anyOf") or (schema or {}).get("oneOf") or [] if isinstance(x, dict)]
    return set(t if isinstance(t, list) else [t]) - {None}


class OutParser:
    """Splits the model's streamed text into ("reasoning", s), ("text", s) and ("tool", name, args) events."""

    def __init__(self, tools, thinking):
        self.props = {}
        for t in tools or []:
            f = t.get("function", t)
            self.props[f.get("name")] = ((f.get("parameters") or {}).get("properties") or {})
        self.tools = bool(tools)
        self.mode, self.buf, self.lead = ("think" if thinking else "text"), "", True

    def value(self, fn, key, v):
        ts = _types(self.props.get(fn, {}).get(key))
        if "string" in ts:
            return v
        s = v.strip()
        if not ts and not s[:1] in ("{", "["):
            return v
        try:
            return json.loads(s)
        except ValueError:
            if s in ("True", "False"):
                return s == "True"
            return v

    def parse(self, body):
        m = FN.search(body)
        if not m:
            try:
                o = json.loads(body.strip())
                a = o.get("arguments", o.get("parameters", {}))
                return o["name"], (json.loads(a) if isinstance(a, str) else a)
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
        name, args = m.group(1).strip(), {}
        for pm in PR.finditer(m.group(2)):
            v = pm.group(2)
            v = v[1:] if v.startswith("\n") else v
            v = v[:-1] if v.endswith("\n") else v
            args[pm.group(1).strip()] = self.value(name, pm.group(1).strip(), v)
        return name, args

    def feed(self, s, final=False):
        self.buf += s
        ev = []
        while True:
            if self.mode == "think":
                i = self.buf.find(TH_CLOSE)
                if i >= 0:
                    if self.buf[:i].rstrip():
                        ev.append(("reasoning", self.buf[:i].rstrip()))
                    self.buf, self.mode, self.lead = self.buf[i + len(TH_CLOSE):], "text", True
                    continue
                k = 0 if final else _hold(self.buf, TH_CLOSE)
                out = self.buf[:len(self.buf) - k] if final else self.buf[:len(self.buf) - k].rstrip()
                if out:
                    ev.append(("reasoning", out))
                self.buf = self.buf[len(out):]
                break
            if self.mode == "text":
                if self.lead:
                    st = self.buf.lstrip()
                    if not st:
                        self.buf = "" if final else self.buf
                        break
                    self.buf, self.lead = st, False
                i = self.buf.find(TC_OPEN) if self.tools else -1
                if i >= 0:
                    if self.buf[:i].strip():
                        ev.append(("text", self.buf[:i].rstrip()))
                    self.buf, self.mode = self.buf[i + len(TC_OPEN):], "tool"
                    continue
                if final:
                    out = self.buf.rstrip()
                else:
                    k = _hold(self.buf, TC_OPEN) if self.tools else 0
                    rest = self.buf[:len(self.buf) - k]
                    out = rest.rstrip()        # whitespace waits: it may sit in front of a tool call
                if out:
                    ev.append(("text", out))
                self.buf = self.buf[len(out):]
                if final:
                    self.buf = ""
                break
            # tool mode: wait for the whole call
            i = self.buf.find(TC_CLOSE)
            if i < 0 and not final:
                break
            body = self.buf[:i] if i >= 0 else self.buf
            self.buf = self.buf[i + len(TC_CLOSE):] if i >= 0 else ""
            c = self.parse(body)
            if c is not None and (i >= 0 or "</function>" in body):
                ev.append(("tool",) + c)
            elif body.strip():
                ev.append(("text", TC_OPEN + body))
            self.mode, self.lead = "text", True
            if not self.buf:
                break
        return ev


# ------------------------------------------------------------------------------------------------------------ inputs
def _text(content, where):
    """OpenAI / Anthropic / Responses content (string or list of parts) -> plain text; images are refused."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    out = []
    for p in content:
        if isinstance(p, str):
            out.append(p)
            continue
        t = p.get("type")
        if t in ("text", "input_text", "output_text", "refusal") or (t is None and "text" in p):
            out.append(p.get("text") or p.get("refusal") or "")
        elif t in ("image", "image_url", "input_image", "document", "input_file", "input_audio", "file"):
            raise BadRequest(f"{where}: content of type {t} is not supported by this server (text-only model port)")
        elif t == "tool_result":
            out.append(_text(p.get("content"), where))
        else:
            raise BadRequest(f"{where}: unknown content type {t!r}")
    return "\n".join(x for x in out if x != "") if len(out) > 1 else (out[0] if out else "")


def _args(a):
    if isinstance(a, dict):
        return a
    if a in (None, ""):
        return {}
    try:
        o = json.loads(a)
        return o if isinstance(o, dict) else {"input": o}
    except (TypeError, ValueError):
        return {"input": a}


class Conv:
    """The internal conversation: system text, then user / assistant / tool messages in template form."""

    def __init__(self):
        self.system, self.msgs = [], []

    def sys(self, text):
        if not text:
            return
        if self.msgs:       # the template allows one system message, at the start
            self.user(text)
        else:
            self.system.append(text)

    def user(self, text):
        self.msgs.append(dict(role="user", content=text))

    def assistant(self, text="", reasoning="", calls=()):
        last = self.msgs[-1] if self.msgs else None
        if last and last["role"] == "assistant" and not (text and last.get("tool_calls")):
            if text:
                last["content"] = (last["content"] + "\n\n" + text) if last["content"] else text
            if reasoning:
                last["reasoning_content"] = (last.get("reasoning_content", "") + "\n" + reasoning).strip()
            last.setdefault("tool_calls", []).extend(calls)
            if not last["tool_calls"]:
                del last["tool_calls"]
            return
        m = dict(role="assistant", content=text or "")
        if reasoning:
            m["reasoning_content"] = reasoning
        if calls:
            m["tool_calls"] = list(calls)
        self.msgs.append(m)

    def tool(self, text):
        self.msgs.append(dict(role="tool", content=text))

    def messages(self, extra_system=""):
        s = "\n\n".join(x for x in self.system + [extra_system] if x)
        return ([dict(role="system", content=s)] if s else []) + self.msgs


def call(name, args):
    return dict(type="function", function=dict(name=name, arguments=args))


def tools_openai(tools, where):
    out = []
    for t in tools or []:
        if t.get("type", "function") != "function":
            raise BadRequest(f"{where}: tool type {t.get('type')!r} is not supported (only function tools run on this server)")
        f = t.get("function") or {}
        if not f.get("name"):
            raise BadRequest(f"{where}: each tool needs function.name")
        out.append(dict(type="function", function=dict(name=f["name"], description=f.get("description", ""),
                                                       parameters=f.get("parameters") or dict(type="object", properties={}))))
    return out


def choice_text(tc, names):
    """tool_choice -> (use tools?, an extra system line or "")"""
    if tc in (None, "auto") or (isinstance(tc, dict) and tc.get("type") == "auto"):
        return True, ""
    if tc == "none" or (isinstance(tc, dict) and tc.get("type") == "none"):
        return False, ""
    if tc in ("required", "any") or (isinstance(tc, dict) and tc.get("type") in ("any", "required")):
        return True, "You must answer with a function call."
    if isinstance(tc, dict):
        n = tc.get("name") or (tc.get("function") or {}).get("name")
        if n:
            return True, f"You must answer with a call to the function {n}."
    raise BadRequest(f"tool_choice {tc!r} is not supported")


def from_openai_chat(b):
    ms = b.get("messages")
    if not isinstance(ms, list) or not ms:
        raise BadRequest("messages must be a non-empty list")
    cv = Conv()
    for m in ms:
        role = m.get("role")
        if role in ("system", "developer"):
            cv.sys(_text(m.get("content"), role))
        elif role == "user":
            cv.user(_text(m.get("content"), "user"))
        elif role == "assistant":
            calls = [call((c.get("function") or {}).get("name"), _args((c.get("function") or {}).get("arguments")))
                     for c in m.get("tool_calls") or []]
            cv.assistant(_text(m.get("content"), "assistant"), m.get("reasoning_content") or m.get("reasoning") or "", calls)
        elif role in ("tool", "function"):
            cv.tool(_text(m.get("content"), "tool"))
        else:
            raise BadRequest(f"unknown message role {role!r}")
    tools = tools_openai(b.get("tools"), "tools")
    use, extra = choice_text(b.get("tool_choice"), [t["function"]["name"] for t in tools])
    return cv, (tools if use else []), extra


def from_anthropic(b):
    ms = b.get("messages")
    if not isinstance(ms, list) or not ms:
        raise BadRequest("messages: must be a non-empty list")
    cv = Conv()
    cv.sys(_text(b.get("system"), "system"))
    for m in ms:
        role, c = m.get("role"), m.get("content")
        blocks = [dict(type="text", text=c)] if isinstance(c, str) else (c or [])
        if role in ("system", "developer"):     # Claude Code also sends system messages inside messages
            cv.sys(_text(c, "system message"))
        elif role == "user":
            text = []
            for x in blocks:
                if x.get("type") == "tool_result":
                    out = _text(x.get("content"), "tool_result")
                    cv.tool(("Error: " + out) if x.get("is_error") else out)
                else:
                    text.append(_text([x], "user"))
            if any(t for t in text) or not blocks:
                cv.user("\n".join(t for t in text if t))
        elif role == "assistant":
            text, think, calls = [], [], []
            for x in blocks:
                t = x.get("type")
                if t == "text":
                    text.append(x.get("text") or "")
                elif t == "thinking":
                    think.append(x.get("thinking") or "")
                elif t == "redacted_thinking":
                    pass
                elif t in ("tool_use", "server_tool_use"):
                    calls.append(call(x.get("name"), _args(x.get("input"))))
                else:
                    raise BadRequest(f"messages: assistant content type {t!r} is not supported")
            cv.assistant("\n\n".join(text).strip(), "\n".join(think).strip(), calls)
        else:
            raise BadRequest(f"messages: unknown role {role!r}")
    tools = []
    for t in b.get("tools") or []:
        if t.get("type") not in (None, "custom"):
            raise BadRequest(f"tools: server tool {t.get('type')!r} ({t.get('name')}) is not supported by this server")
        tools.append(dict(type="function", function=dict(name=t.get("name"), description=t.get("description", ""),
                                                         parameters=t.get("input_schema") or dict(type="object", properties={}))))
    use, extra = choice_text(b.get("tool_choice"), [t["function"]["name"] for t in tools])
    return cv, (tools if use else []), extra


HOSTED = ("web_search", "web_search_preview", "file_search", "code_interpreter", "image_generation", "computer_use_preview",
          "mcp", "local_shell")


def from_responses(b):
    cv = Conv()
    cv.sys(b.get("instructions") or "")
    items = b.get("input")
    if isinstance(items, str):
        items = [dict(type="message", role="user", content=items)]
    if not isinstance(items, list) or not items:
        raise BadRequest("input must be a string or a non-empty list")
    custom = set()
    for it in items:
        t = it.get("type", "message")
        if t == "message":
            role = it.get("role")
            text = _text(it.get("content"), f"input {role} message")
            if role in ("system", "developer"):
                cv.sys(text)
            elif role == "user":
                cv.user(text)
            elif role == "assistant":
                cv.assistant(text)
            else:
                raise BadRequest(f"input: unknown message role {role!r}")
        elif t == "function_call":
            cv.assistant(calls=[call(it.get("name"), _args(it.get("arguments")))])
        elif t == "custom_tool_call":
            custom.add(it.get("name"))
            cv.assistant(calls=[call(it.get("name"), {"input": it.get("input", "")})])
        elif t == "local_shell_call":
            cv.assistant(calls=[call("local_shell", dict((it.get("action") or {})))])
        elif t in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            o = it.get("output")
            if isinstance(o, dict):
                o = o.get("content", o.get("output", json.dumps(o)))
            cv.tool(_text(o, t) if not isinstance(o, str) else o)
        elif t == "reasoning":
            r = "\n".join(x.get("text", "") for x in (it.get("content") or []) + (it.get("summary") or []) if isinstance(x, dict))
            if r.strip():
                cv.assistant(reasoning=r.strip())
        else:
            raise BadRequest(f"input: item type {t!r} is not supported by this server")
    tools, kinds, dropped = [], {}, []

    def add(t, ns=None):
        ty = t.get("type")
        if ty == "function":
            tools.append(dict(type="function", function=dict(name=t.get("name"), description=t.get("description", ""),
                                                             parameters=t.get("parameters") or dict(type="object", properties={}))))
            kinds[t.get("name")] = ("function", ns)
        elif ty == "custom":
            fmt = t.get("format") or {}
            desc = (t.get("description") or "") + (("\n\nThe input follows this grammar (" + str(fmt.get("syntax")) + "):\n"
                                                    + str(fmt.get("definition"))) if fmt.get("definition") else "")
            tools.append(dict(type="function", function=dict(name=t.get("name"), description=desc, parameters=dict(
                type="object", properties=dict(input=dict(type="string", description="the raw tool input")), required=["input"]))))
            kinds[t.get("name")] = ("custom", ns)
        elif ty == "namespace" and ns is None:
            for x in t.get("tools") or []:
                add(x, t.get("name"))
        elif ty in HOSTED:
            dropped.append(ty)       # runs on OpenAI's servers; not available here, the model does not see it
        else:
            raise BadRequest(f"tools: tool type {ty!r} is not supported by this server (function, custom and namespace tools only)")
    for t in b.get("tools") or []:
        add(t)
    use, extra = choice_text(b.get("tool_choice"), list(kinds))
    return cv, (tools if use else []), extra, kinds, dropped
