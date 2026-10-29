"""The OpenAPI description (runtime/openapi.py) against the real server (runtime/server.py), CPU only, no model.

The server runs with a one-id-per-character stand-in tokenizer and a scripted engine, so every route goes through the
real request parsing and response building. The test fails when:
- the server answers a route the spec lacks, or the spec lists a route the server does not answer;
- the server reads a request field the spec does not list, or the sampling fields drift from server.KNOBS;
- a field a real coding agent sends is not in the spec (keys recorded from Claude Code, Codex and Pi);
- a response, stream event or error carries a field the spec does not list, or misses a required one;
- GET /openapi.json differs from openapi.spec().
"""
import ast
import json
import pathlib
import re
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from mashbox.runtime import openapi as O
from mashbox.runtime.server import KNOBS, Server, serve

RT = pathlib.Path(O.__file__).resolve().parent
MODEL = "qwen3-8-27b"
TOOLS = [dict(type="function", function=dict(name="Bash", description="run a command", parameters=dict(
    type="object", properties=dict(command=dict(type="string")), required=["command"])))]
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"

# Request keys real clients sent to the server's routes (logging probe): Claude Code 2.1.312, Codex CLI
# 0.165.0, Pi 1.4.0.
CLIENT_KEYS = {
    "/v1/messages": "max_tokens messages metadata model output_config stream system thinking tools",
    "/v1/responses": "client_metadata include input model parallel_tool_calls prompt_cache_key reasoning store stream tool_choice tools",
    "/v1/chat/completions": "max_completion_tokens messages model store stream stream_options tools",
}
# A chat completion recorded from a real run of the server (Qwen3.8-27B-TPU). It came from an earlier
# build, so it is checked for required fields and types only.
RECORDED_CHAT = {"id": "chatcmpl-35874d4cebb34fd8b0c6e04b", "object": "chat.completion", "created": 1793159712,
                 "model": "qwen3-8-27b", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                     "role": "assistant", "content": "Mercury\nVenus\nEarth"}}],
                 "usage": {"prompt_tokens": 24, "completion_tokens": 7, "total_tokens": 31},
                 "system_fingerprint": "seed-29770299736766241"}


# ---------------------------------------------------------------------------------------------- stand-ins
class CharChat:
    """Tokenizer stand-in: one id per character, id 0 ends the answer."""
    stop_ids = (0,)

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)

    def chat_ids(self, messages, enable_thinking=False, tools=None):
        return self.encode(json.dumps([messages, tools, enable_thinking]) + "\nassistant:\n")


class Script:
    """Engine stand-in: answers each call with the next scripted text, streamed a few ids at a time."""
    CTX, S, K, prefix_cache = 1 << 16, 2, 4, False

    def __init__(self):
        self.texts = []

    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None):
        ids = [[ord(c) for c in (self.texts.pop(0) if self.texts else "Hello.")][:max_new] for _ in prompts]
        fin = ["stop" if len(i) < max_new else "length" for i in ids]
        for k in range(1, max(len(i) for i in ids) + 1, 5):
            on_step([i[:k] for i in ids], [None] * len(ids))
        on_step(ids, fin)
        return dict(ids=ids, finish=fin, cached=[0] * len(ids), timing={})


INFO = dict(weights="model.bshj", gb=1.0, bpw=4.5, prefix_cache=False, slots=2, cache_ram_mib=0)


@pytest.fixture(scope="module")
def srv():
    eng = Script()
    s = Server(eng, CharChat(), MODEL, INFO, log=lambda m: None)
    s.ready = True
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    threading.Thread(target=serve, args=(s, "127.0.0.1", port, lambda m: None), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/v1/health", timeout=1)
            break
        except OSError:
            time.sleep(0.05)
    return s, eng, url


def call(url, method="GET", body=None):
    """-> (status, parsed body or list of (event name, data) for a stream, content type). A text/plain body (GET
    /metrics) comes back as its text; every other body must parse as JSON."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        r = e
    raw, ctype = r.read().decode(), r.headers.get("Content-Type", "")
    if "event-stream" in ctype:
        ev = []
        for blk in raw.split("\n\n"):
            name = next((ln[7:] for ln in blk.split("\n") if ln.startswith("event: ")), None)
            d = next((ln[6:] for ln in blk.split("\n") if ln.startswith("data: ")), None)
            if d and d != "[DONE]":
                ev.append((name, json.loads(d)))
        return r.status, ev, ctype
    if ctype.startswith("text/plain"):
        return r.status, raw, ctype
    return r.status, (json.loads(raw) if raw else None), ctype


# ---------------------------------------------------------------------------------------------- schema check (stdlib)
SPEC = O.spec()
TYPES = dict(string=str, boolean=bool, object=dict, array=list)


def resolve(s):
    while "$ref" in s:
        s = SPEC["components"]["schemas"][s["$ref"].rsplit("/", 1)[1]]
    return s


def is_type(x, t):
    if t == "null":
        return x is None
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    return isinstance(x, TYPES[t])


def check(x, s, strict=True, at="$"):
    """Raise AssertionError when x does not fit schema s. strict: an object field the schema does not list fails."""
    s = resolve(s)
    t = s.get("type")
    if t is not None:
        assert any(is_type(x, u) for u in (t if isinstance(t, list) else [t])), f"{at}: {x!r} is not {t}"
    if "enum" in s:
        assert x in s["enum"], f"{at}: {x!r} not in {s['enum']}"
    if "const" in s:
        assert x == s["const"], f"{at}: {x!r} != {s['const']!r}"
    if isinstance(x, dict) and "properties" in s:
        for r in s.get("required", ()):
            assert r in x, f"{at}: required field {r} missing"
        for k, v in x.items():
            if k in s["properties"]:
                check(v, s["properties"][k], strict, f"{at}.{k}")
            else:
                assert not strict or s.get("additionalProperties"), f"{at}.{k}: sent by the server but not in the spec"
    if isinstance(x, list) and isinstance(s.get("items"), dict):
        for i, v in enumerate(x):
            check(v, s["items"], strict, f"{at}[{i}]")


def op(method, path):
    return SPEC["paths"][path][method]


def body_schema(method, path, code, stream=False):
    return op(method, path)["responses"][code]["content"]["text/event-stream" if stream else "application/json"]["schema"]


# ---------------------------------------------------------------------------------------------- the spec itself
def test_spec_refs_resolve_and_ids_are_unique():
    text = json.dumps(SPEC)
    for name in set(x.split('"')[0] for x in text.split('"$ref": "#/components/schemas/')[1:]):
        assert name in SPEC["components"]["schemas"], name
    ids = [r["id"] for r in O.ROUTES]
    assert len(ids) == len(set(ids))
    assert SPEC["openapi"].startswith("3.1") and SPEC["info"]["version"]


def test_sampling_fields_equal_the_server_knobs():
    assert set(O.SAMPLING) == set(KNOBS) | {"max_completion_tokens", "stop_sequences", "n"}


# ---------------------------------------------------------------------------------------------- routes
def server_paths():
    """Every path literal in server.py's request handlers (do_GET / do_POST)."""
    tree = ast.parse((RT / "server.py").read_text())
    out = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in ("do_GET", "do_POST"):
            for n in ast.walk(fn):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("/") and " " not in n.value:
                    out.add(n.value)
    return out


def test_handlers_route_only_through_the_table():
    """The handlers are one do_<METHOD> per method in ROUTES (a do_PUT with no route fails), and they read the request
    path only through openapi.route (or the --ui page lookup UI.handle), never by comparing it: self.path is read once,
    into path, at the top of a handler, so a route answered by the server is always a ROUTES entry, whatever constant or
    helper holds its path. Only those two qualified calls take the path; the handler object itself goes to code of
    this module only (whose path reads this test sees), never to a helper in another module, in any wrapping (a
    container, a lambda, *args, a thread's args=, x.__dict__); code of this module that receives it is read by the same
    rule; and the handler class takes its methods from this module only."""
    tree = ast.parse((RT / "server.py").read_text())
    handlers = [fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name.startswith("do_")]
    assert {fn.name[3:].lower() for fn in handlers} == {r["method"] for r in O.ROUTES}, \
        "a request handler without routes in runtime/openapi.py ROUTES (or ROUTES with a method no handler serves)"

    def lookup(f):   # the two calls that may read the path: openapi.route(method, path) and UI.handle(path, model)
        return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in {("openapi", "route"), ("UI", "handle")}
    bad, ok = [], set()
    up = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    classes = {c.name: c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)}
    owner = {f: c for c in classes.values() for f in c.body if isinstance(f, ast.FunctionDef)}   # method -> its class
    hcls = [c for c in classes.values() if any(f in handlers for f in c.body)]
    for cls in hcls:
        if [ast.unparse(b) for b in cls.bases] != ["BaseHTTPRequestHandler"]:   # a mixin from another module adds handlers
            bad.append(f"class {cls.name} bases {[ast.unparse(b) for b in cls.bases]}")
        bad += [f"class {cls.name} line {n.lineno} assigns a do_ handler" for n in cls.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                and any(isinstance(t, ast.Name) and t.id.startswith("do_") for t in getattr(n, "targets", [getattr(n, "target", None)]))]
    # Where the request handler may go: read from (h.wfile), compared, named again (h = self), kept on an object of this
    # module (SSE: self.h = h) or handed to a function, class or method defined here, whose parameter then holds it and is
    # read by the same rule. Anything else hands it on: an argument to other code, a container, a lambda, a walrus,
    # *args or **kwargs, a thread's args=, a return. x.__dict__ is the handler too.
    held = {f: {"self"} for c in hcls for f in c.body if isinstance(f, ast.FunctionDef)}   # function -> names holding it
    kept = {c: set() for c in classes}                                                 # class -> attributes holding it

    def me(fn):   # a method's own self
        return fn.args.args[0].arg if fn in owner and fn.args.args else None

    def reached(call, fn):   # the functions of this module a call runs, with the bound-self offset
        f, out = call.func, []
        if isinstance(f, ast.Name):   # f(...), C(...) -> C.__init__
            out = [(g, 0) for g in ast.walk(tree) if isinstance(g, ast.FunctionDef) and g.name == f.id and g not in owner]
            out += [(g, 1) for g in getattr(classes.get(f.id), "body", []) if isinstance(g, ast.FunctionDef) and g.name == "__init__"]
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == me(fn):   # self.m(...)
            out = [(g, 1) for g in owner[fn].body if isinstance(g, ast.FunctionDef) and g.name == f.attr]
        return out

    def param(g, off, call, arg):   # the parameter an argument lands in; None for *args, **kwargs or a shifted position
        pos = [a.arg for a in g.args.posonlyargs + g.args.args]
        if isinstance(arg, ast.keyword):
            return arg.arg if arg.arg in pos + [a.arg for a in g.args.kwonlyargs] else None
        i = call.args.index(arg)
        return pos[i + off] if i + off < len(pos) and not any(isinstance(a, ast.Starred) for a in call.args[:i]) else None

    for _ in range(8):
        size, gone = sum(map(len, held.values())) + sum(map(len, kept.values())), set()
        for fn in set(held) | {f for c, a in kept.items() if a for f in classes[c].body if isinstance(f, ast.FunctionDef)}:
            names, own = held.setdefault(fn, set()), owner.get(fn)
            for e in ast.walk(fn):
                if isinstance(getattr(e, "ctx", None), (ast.Store, ast.Del)) or not (
                        isinstance(e, ast.Name) and e.id in names or isinstance(e, ast.Attribute) and own is not None
                        and e.attr in kept[own.name] and isinstance(e.value, ast.Name) and e.value.id == me(fn)):
                    continue
                while isinstance(up[e], ast.Attribute) and up[e].attr == "__dict__" or isinstance(up[e], ast.IfExp) and up[e].test is not e:
                    e = up[e]   # self.__dict__ and (self if c else None) are the handler too
                p, where = up[e], f"{fn.name} line {e.lineno}"
                if isinstance(p, (ast.Attribute, ast.Compare)):   # h.wfile, h.send_response(...), h is None
                    continue
                arg = e if isinstance(p, ast.Call) and e in p.args else p if isinstance(p, ast.keyword) and p.arg else None
                if arg is not None:
                    call = p if arg is e else up[p]
                    hits = [(g, param(g, off, call, arg)) for g, off in reached(call, fn)]
                    if not hits or not all(n for _, n in hits):
                        gone.add(f"{where} passes the request handler to {ast.unparse(call.func)}")
                    for g, n in hits:
                        held.setdefault(g, set()).update([n] if n else [])
                    continue
                if isinstance(p, ast.Assign) and p.value is e:
                    tgts = p.targets
                elif isinstance(p, ast.Tuple) and isinstance(up[p], ast.Assign) and up[p].value is p and all(
                        isinstance(t, ast.Tuple) and len(t.elts) == len(p.elts) for t in up[p].targets):
                    tgts = [t.elts[p.elts.index(e)] for t in up[p].targets]   # self.h, self.started = h, False
                else:
                    gone.add(f"{where} puts the request handler in {type(p).__name__} {ast.unparse(p)[:60]!r}")
                    continue
                for t in tgts:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute) and own is not None and isinstance(t.value, ast.Name) and t.value.id == me(fn):
                        kept[own.name].add(t.attr)
                    else:
                        gone.add(f"{where} stores the request handler in {ast.unparse(t)}")
        if size == sum(map(len, held.values())) + sum(map(len, kept.values())):
            break
    bad += sorted(gone)
    for fn in handlers:
        if fn.body and isinstance(fn.body[0], ast.Assign):   # path = self.path.split("?")[0]...
            ok |= {id(x) for x in ast.walk(fn.body[0].value)}
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and lookup(n.func):
                ok |= {id(a) for a in n.args}
            if isinstance(n, ast.JoinedStr):          # f-strings: error messages that quote the path
                ok |= {id(x) for x in ast.walk(n)}
            if isinstance(n, ast.Assign):
                ok |= {id(t) for t in n.targets}
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("json.dump", "json.dumps"):   # json.dump(dict(path=self.path)):
                ok |= {id(x) for d in n.args if isinstance(d, ast.Call) and ast.unparse(d.func) == "dict"   # data written out,
                       for k in d.keywords if k.arg == "path" for x in ast.walk(k.value)}                   # never handed on
        held = {"path"}   # every name that holds the path: path = q = self.path..., q = path, q = path.lower()
        routed = {id(x) for n in ast.walk(fn) if isinstance(n, ast.Call) and lookup(n.func)
                  for a in n.args for x in ast.walk(a)}   # what route() returns is a route
        for _ in range(3):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Assign, ast.NamedExpr)) and any(id(x) not in routed and (
                        (isinstance(x, ast.Name) and x.id in held) or (isinstance(x, ast.Attribute) and x.attr == "path"))
                        for x in ast.walk(n.value)):
                    held |= {x.id for tg in (n.targets if isinstance(n, ast.Assign) else [n.target])
                             for x in ast.walk(tg) if isinstance(x, ast.Name)}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in held and id(n) not in ok:
                bad.append(f"{fn.name} line {n.lineno} ({n.id})")
    for n in ast.walk(tree):   # .path anywhere else in the module (a helper, a second read, h.path in a local helper)
        if isinstance(n, ast.Attribute) and n.attr == "path" and not (isinstance(n.value, ast.Name) and n.value.id == "os") \
                and id(n) not in ok:
            bad.append(f"{ast.unparse(n)} line {n.lineno}")
        if isinstance(n, ast.Attribute) and n.attr in ("requestline", "raw_requestline"):   # the path again, another way
            bad.append(f"self.{n.attr} line {n.lineno}")
    assert not bad, "request path used outside openapi.route (add the route to runtime/openapi.py ROUTES): " + ", ".join(bad)


@pytest.mark.parametrize("call", ["contributor_route.handle(self)", "contributor_route.handle(path)", "helpers.answer(h=self)",
                                  "contributor_route.handle(self.__dict__)", "contributor_route.handle([self])",
                                  "contributor_route.handle(lambda: self)", "contributor_route.handle(**{'h': self})",
                                  "threading.Thread(target=helpers.answer, args=(self,)).start()", "_hand_on(self)",
                                  "_Keep(self).go()"])
def test_route_rule_catches_an_imported_helper(tmp_path, monkeypatch, call):
    """A helper in another module, handed the request handler or its path (directly, wrapped, or through a function or
    an object of server.py), answers a path outside ROUTES while every other check passes: the
    route rule fails on it, so the rule stays on."""
    import sys
    src = (RT / "server.py").read_text()
    mt = re.search(r"\n( +)def do_GET\(self\):\n( +)path = [^\n]*\n", src)
    ind = mt.group(2)
    hand_on = ("\n\ndef _hand_on(h):\n    from . import contributor_route\n    return contributor_route.handle(h)\n"
               "\n\nclass _Keep:\n    def __init__(self, h):\n        self.h = h\n\n    def go(self):\n"
               "        from . import contributor_route\n        return contributor_route.handle(self.h)\n")
    (tmp_path / "server.py").write_text(src[:mt.end()] + f"{ind}from . import contributor_route, helpers\n"
                                        f"{ind}if {call}:\n{ind}    return\n" + src[mt.end():] + hand_on)
    monkeypatch.setattr(sys.modules[__name__], "RT", tmp_path)
    with pytest.raises(AssertionError, match="passes the request handler|request path used outside"):
        test_handlers_route_only_through_the_table()


def test_route_finds_only_its_own_path():
    """openapi.route() reads nothing but ROUTES and its arguments (no alias map, no second table), and what it returns is
    the route whose path template the request path matches."""
    fn = next(n for n in ast.parse((RT / "openapi.py").read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "route")
    local = {a.arg for a in fn.args.args} | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - local
    assert used <= {"ROUTES", "re"}, f"route() reads {sorted(used - {'ROUTES', 're'})}: every path goes through ROUTES only"
    import re
    probes = {concrete(r["path"]) for r in O.ROUTES} | {"/v1/ping", "/ping", "/v1", "/v1/healthz", "/v1/health/", "/v1/models/a/b", "/x"}
    for method in {r["method"] for r in O.ROUTES}:
        for p in sorted(probes):
            r, _ = O.route(method, p)
            if r is not None:
                assert r["method"] == method and re.fullmatch(re.sub(r"\\\{\w+\\\}", "[^/]+", re.escape(r["path"])), p), (method, p, r["path"])


def test_ui_answers_only_pages_and_files():
    """With --ui the server also asks ui.chat.handle; that lookup answers its page paths and static files only, never
    a JSON route (a route that is not in ROUTES)."""
    from mashbox.ui import chat, shell
    for mod in (chat, shell):
        assert "json" not in " ".join(getattr(mod, "TYPES", {}).values()), mod.__name__
    fn = next(n for n in ast.parse(pathlib.Path(chat.__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "handle")
    consts = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes))]
    assert not [c for c in consts if isinstance(c, bytes) or c.startswith("/") or ("/" in c and not c.startswith("text/html"))], \
        f"ui.chat.handle answers something besides PAGE_PATHS and files: {consts}"
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert calls <= {"split", "page", "asset"}, f"ui.chat.handle calls {sorted(calls - {'split', 'page', 'asset'})}: " \
        "it answers PAGE_PATHS and files only (a JSON route goes in runtime/openapi.py ROUTES)"
    consts = {n.value for m in (chat, shell) for n in ast.walk(ast.parse(pathlib.Path(m.__file__).read_text()))
              if isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"/[\w./-]*", n.value)}
    for p in ({concrete(r["path"]) for r in O.ROUTES} | consts | {"/v1/ping", "/api/state"}) - set(chat.PAGE_PATHS):
        assert chat.handle(p, MODEL) is None, p


def concrete(path):
    return path.replace("{model}", MODEL)


def test_routes_equal_what_the_server_answers(srv):
    _, _, url = srv
    paths = {p for p in server_paths() if not p.endswith("/")} | {concrete(r["path"]) for r in O.ROUTES}
    paths |= {concrete("/v1/models/{model}")} if "/v1/models/" in server_paths() else set()
    answered = set()
    for p in sorted(paths):
        for m in ("GET", "POST"):
            st, _, _ = call(url + p, m, {} if m == "POST" else None)
            if st != 404:
                answered.add((m.lower(), p))
    listed = {(r["method"], concrete(r["path"])) for r in O.ROUTES if r["method"] != "head"}
    assert answered - listed == set(), "answered by the server, missing from runtime/openapi.py ROUTES"
    assert listed - answered == set(), "in runtime/openapi.py ROUTES, not answered by the server"
    assert call(url + "/v1/models/another-model")[0] == 404
    for r in O.ROUTES:
        if r["method"] == "head":
            assert call(url + r["path"], "HEAD")[0] == 200


def test_served_spec_equals_the_module(srv):
    s, _, url = srv
    st, got, _ = call(url + "/openapi.json")
    assert st == 200 and got == json.loads(json.dumps(O.spec(s)))
    assert MODEL in got["info"]["description"]


# ---------------------------------------------------------------------------------------------- request fields
def body_reads(path):
    """{function name: keys read as b.get("k") or b["k"]} in one source file."""
    out = {}
    for fn in ast.walk(ast.parse(path.read_text())):
        if not isinstance(fn, ast.FunctionDef):
            continue
        keys = set()
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "b" and n.args
                    and isinstance(n.args[0], ast.Constant)):
                keys.add(n.args[0].value)
            elif (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "b"
                  and isinstance(n.slice, ast.Constant)):
                keys.add(n.slice.value)
        if keys:
            out[fn.name] = keys
    return out


def request_props():
    return {r["path"]: set(O.SCHEMAS[r["request"]]["properties"]) for r in O.ROUTES if r.get("request")}


def test_every_field_the_server_reads_is_in_the_spec():
    reads = {**body_reads(RT / "agents.py"), **body_reads(RT / "server.py")}
    props = request_props()
    every = set().union(*props.values())
    missing = {(fn, k) for fn, ks in reads.items() for k in ks if k not in every}
    assert not missing, f"request fields read by the server but on no route in runtime/openapi.py: {sorted(missing)}"
    for path in ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"):   # knobs_of() runs on each
        assert reads["knobs_of"] <= props[path], (path, reads["knobs_of"] - props[path])
    assert {"merge"} <= props["/v1/chat/completions"] & props["/v1/completions"]


def test_fields_real_clients_send_are_in_the_spec():
    props = request_props()
    for path, keys in CLIENT_KEYS.items():
        assert set(keys.split()) <= props[path], (path, set(keys.split()) - props[path])


# ---------------------------------------------------------------------------------------------- responses
def run(srv, path, body, texts, stream):
    _, eng, url = srv
    eng.texts[:] = list(texts)
    st, got, ctype = call(url + path, "POST", dict(body, stream=stream) if stream else body)
    assert st == 200, got
    assert ("event-stream" in ctype) == stream
    return got


CASES = [
    ("/v1/chat/completions", dict(model=MODEL, messages=[dict(role="user", content="where?")], tools=TOOLS,
                                  stream_options=dict(include_usage=True)), ["Checking.\n\n" + CALL]),
    ("/v1/chat/completions", dict(messages=[dict(role="user", content="hi")], chat_template_kwargs=dict(enable_thinking=True),
                                  max_tokens=40), ["plan\n</think>\n\nHello there."]),
    ("/v1/completions", dict(prompt="The three largest planets are", max_tokens=8, stop=["\n"]), [" Jupiter, Saturn"]),
    ("/v1/messages", dict(model=MODEL, max_tokens=200, tools=[dict(name="Bash", input_schema=TOOLS[0]["function"]["parameters"])],
                          thinking=dict(type="enabled", budget_tokens=100), system="You are a coding agent.",
                          messages=[dict(role="user", content="where am I?")]), ["plan\n</think>\n\nLet me look.\n\n" + CALL]),
    ("/v1/messages", dict(model=MODEL, max_tokens=3, messages=[dict(role="user", content="hi")]), ["Hello there."]),
    ("/v1/responses", dict(model=MODEL, input="where am I?", instructions="Be brief.", reasoning=dict(effort="high"),
                           tools=[dict(type="function", name="Bash", parameters=TOOLS[0]["function"]["parameters"]),
                                  dict(type="custom", name="apply_patch", description="edit files"),
                                  dict(type="web_search")]), ["plan\n</think>\n\nLooking.\n\n" + CALL]),
    ("/v1/responses", dict(model=MODEL, input=[dict(type="message", role="user", content="hi")], max_output_tokens=3),
     ["Hello there."]),
]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("path,body,texts", CASES, ids=[f"{c[0]}#{i}" for i, c in enumerate(CASES)])
def test_responses_fit_the_spec(srv, path, body, texts, stream):
    got = run(srv, path, body, texts, stream)
    if not stream:
        return check(got, body_schema("post", path, "200"))
    assert got, "no events"
    sch = body_schema("post", path, "200", stream=True)
    for name, data in got:
        check(data, sch)
        if name is not None:
            assert data["type"] == name


def test_count_tokens_and_reads_fit_the_spec(srv):
    _, _, url = srv
    st, got, _ = call(url + "/v1/messages/count_tokens", "POST", dict(model=MODEL, messages=[dict(role="user", content="hi")]))
    assert st == 200
    check(got, body_schema("post", "/v1/messages/count_tokens", "200"))
    for path, schema_path in (("/v1/health", "/v1/health"), ("/health", "/health"), ("/", "/"), ("/v1/models", "/v1/models"),
                              (f"/v1/models/{MODEL}", "/v1/models/{model}")):
        st, got, _ = call(url + path)
        assert st == 200
        check(got, body_schema("get", schema_path, "200"))


def test_errors_fit_the_spec(srv):
    _, _, url = srv
    for r in O.ROUTES:
        if r["method"] == "post":
            st, got, _ = call(url + r["path"], "POST", {"messages": "not a list", "input": 3, "prompt": 3})
            assert st == 400, (r["path"], got)
            check(got, body_schema("post", r["path"], "400"))
    st, got, _ = call(url + "/v1/models/another-model")
    check(got, body_schema("get", "/v1/models/{model}", "404"))


def test_a_recorded_real_response_still_parses():
    check(RECORDED_CHAT, body_schema("post", "/v1/chat/completions", "200"), strict=False)


EXAMPLES_FILE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "api-examples.json"


@pytest.mark.skipif(not EXAMPLES_FILE.exists(), reason="no answers recorded (tools/record_api_examples.py on a chip)")
def test_recorded_answers_fit_the_spec():
    """docs/api-examples.json holds real answers from a chip (tools/record_api_examples.py); each one must fit its schema."""
    rec = json.loads(EXAMPLES_FILE.read_text())
    assert rec["chip"] and rec["zone"] and rec["recorded_utc"]
    for rid, got in rec["routes"].items():
        assert got["status"] == 200, (rid, got["status"])
        check(got["response"], body_schema(got["method"], got["path"], "200"), strict=True)
