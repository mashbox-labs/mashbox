"""MASHBOX Chat: the chat module page for any model behind an OpenAI-compatible server.

The page (index.html) sits in the MASHBOX module shell and fills only the content area: streaming replies, multi-turn
with edit, regenerate and stop, chats saved in the browser, a system prompt, every sampling setting the engine takes,
thinking on or off, Markdown with code copy, measured time to first token and tokens per second per reply, context used,
export and import, shortcuts, phone and desktop. It talks to the server it was served from (GET /v1/models,
POST /v1/chat/completions), so the same page works for every chat model.

Model details come from two places, never from the page itself:
- the running server's /v1/models (model id, context length);
- the model's models.json row: the shell's model bar (name, maker, one line, chips, links) and the optional "chat"
  object (thinking, recommended sampling for each mode and their source), put into the page's #mbx-chat block.

A server serves it with `handle(path, model_id)`: GET / (and /chat) is the page, /ui/shell/... the shell's files and
/ui/chat/... this module's script. `python -m mashbox.ui.chat --api URL` serves the page for any OpenAI-compatible server.
"""
from __future__ import annotations

import json
import re
from importlib import resources

from .. import shell

ROUTE = "ui/chat/"
PAGE = "index.html"
PAGE_PATHS = ("/", "/chat", "/index.html")   # never a path ending in "/" past the root: the page loads its files relative to it
TYPES = {".js": "text/javascript; charset=utf-8"}
CHAT_BLOCK = re.compile(rb'(<script type="application/json" id="mbx-chat">)(.*?)(</script>)', re.S)
CHAT_KEYS = {"thinking", "thinking_default", "keep_thinking", "sampling", "source"}


def _json(o) -> bytes:
    return json.dumps(o, ensure_ascii=False).replace("</", "<\\/").encode()


def chat_config(model_id: str | None, reg: dict | None = None) -> dict:
    """The model's chat settings from its models.json row ({} for a model that has none or is not in models.json)."""
    reg = reg or shell.registry()
    row = next((m for m in reg["models"] if m["id"] == model_id), None)
    c = dict((row or {}).get("chat") or {})
    bad = set(c) - CHAT_KEYS
    if bad:
        raise ValueError(f"models.json {model_id}: unknown chat keys {sorted(bad)}; allowed {sorted(CHAT_KEYS)}")
    return c


def page(model_id: str | None = None, reg: dict | None = None) -> bytes:
    """The chat page for the model a server runs, with the shell config and the chat block resolved."""
    reg = reg or shell.registry()
    raw = resources.files(__package__).joinpath(PAGE).read_bytes()
    own = shell.page_config(raw) or {}
    row = next((m for m in reg["models"] if m["id"] == model_id), None)
    if row is not None:
        own["model"] = model_id
        up = row.get("upstream") or {}
        if up.get("hf"):
            own["resources"] = [{"title": f"{up.get('name', row['name'])} model card", "by": up.get("maker"),
                                 "url": f"https://huggingface.co/{up['hf']}"}]
    elif model_id:          # a model outside models.json: the bar names it, with no links of ours
        own.pop("model", None)
        own.update(name=model_id, line="A chat model behind an OpenAI-compatible server.")
    raw = shell.BLOCK.sub(lambda m: m.group(1) + _json(own) + m.group(3), raw, count=1)
    raw = shell.render(raw, reg)
    cfg = chat_config(model_id, reg) if row is not None else {}
    return CHAT_BLOCK.sub(lambda m: m.group(1) + _json(cfg) + m.group(3), raw, count=1)


def asset(path: str) -> tuple[bytes, str] | None:
    """(bytes, content type) for a request under ui/chat/ (this module's script), or None."""
    path = path.split("?", 1)[0].lstrip("/")
    if not path.startswith(ROUTE):
        return None
    rel = path[len(ROUTE):]
    if not rel or "/" in rel or rel.startswith("."):
        return None
    ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else ""
    f = resources.files(__package__).joinpath(rel)
    if ext not in TYPES or not f.is_file():
        return None
    return f.read_bytes(), TYPES[ext]


_cache: dict = {}


def handle(path: str, model_id: str | None) -> tuple[bytes, str] | None:
    """What a server answers for a GET of `path` when it serves the chat page; None for every other path."""
    p = path.split("?", 1)[0]
    if p in PAGE_PATHS:
        if model_id not in _cache:
            _cache[model_id] = page(model_id)
        return _cache[model_id], "text/html; charset=utf-8"
    return shell.asset(p) or asset(p)
