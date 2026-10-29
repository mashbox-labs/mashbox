"""The MASHBOX module shell: one top bar, model bar, footer, set of tokens and fonts for every MASHBOX module page.

A module page (a chat, a world model, anything we ship with a browser UI) links the shell and carries one config block,
then fills only its own content area (README.md in this folder has the full contract):

    <link rel="stylesheet" href="ui/shell/shell.css">
    <script type="application/json" id="mbx-module">{"model": "qwen3-8-27b", "module": "Chat"}</script>
    <script src="ui/shell/shell.js"></script>
    ...
    <body class="mbx-ui"><header data-mbx-top></header> <main class="mbx-main">...</main> <footer data-mbx-foot></footer>

The module's server does two things:

- `render(page_bytes)` resolves the config block against models.json (model name, maker, one line, chips, links) and
  adds the shared site links and company line, so the registry stays the one source and a page names nothing twice.
- `asset(path)` answers every request under `ui/shell/` with the bundled file, so the page fetches nothing at run time.

`check_page()` and `check_links()` are the rules the CPU test (tests/cpu/test_module_shell.py) and the release gate run.
"""
from __future__ import annotations

import json
import pathlib
import re
from importlib import resources

ROUTE = "ui/shell/"
SITE_URL = "https://mashbox.gg"
DOCS = f"{SITE_URL}/docs"
COMPANY_LINE = ("MASHBOX Labs makes open models run faster and cheaper on TPUs and accelerated hardware, "
                "as well as exploring the future of the intersection of AI and gaming.")
# Pages of mashbox.gg the shell links back to (label, path). The first is home.
SITE_PAGES = (("MASHBOX Labs", "/"), ("Models", "/models"), ("Docs", "/docs"), ("Research", "/research"))
MODULE_NAMES = {"world": "World model", "llm": "Chat", "llm-tpu": "Chat"}
THEMES = ("system", "dark", "light")
STATES = ("live", "busy", "error", "off")
# Keys a page may set in its block. Everything else (site links, company line) comes from the shell only.
PAGE_KEYS = {"model", "module", "name", "maker", "line", "chips", "links", "theme", "status", "resources", "sticky", "foot_links"}
LINK_KEYS = {"docs", "card", "code", "upstream", "extra"}
TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2",
         ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml"}   # pages are never served from here
BLOCK = re.compile(rb'(<script type="application/json" id="mbx-module">)(.*?)(</script>)', re.S)


def registry() -> dict:
    """The packaged models.json."""
    return json.loads(resources.files("mashbox").joinpath("models.json").read_text())


def _row(reg: dict, model_id: str) -> dict:
    for m in reg["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"no model {model_id!r} in models.json")


def site(reg: dict | None = None) -> list[dict]:
    """The links back to the rest of the MASHBOX Labs site, in nav order: mashbox.gg pages, then GitHub and Hugging Face."""
    reg = reg or registry()
    out = [{"label": label, "url": SITE_URL + path} for label, path in SITE_PAGES]
    out.append({"label": "GitHub", "url": f"https://github.com/{reg['github']['org']}"})
    out.append({"label": "Hugging Face", "url": f"https://huggingface.co/{reg['hf']['org']}"})
    return out


def links(model_id: str, reg: dict | None = None) -> dict:
    """The links a model module shows: its docs page, our model card, our code, and the original model."""
    reg = reg or registry()
    m, gh = _row(reg, model_id), reg["github"]
    out = {
        "docs": f"{DOCS}/models/{m['id']}",
        "card": f"https://huggingface.co/{m['hf_repo']}",
        "code": f"https://github.com/{gh['org']}/{gh['repo']}/tree/v{reg['version']}/{m['path']}",
    }
    if m.get("upstream", {}).get("hf"):
        out["upstream"] = f"https://huggingface.co/{m['upstream']['hf']}"
    return out


def chips(model_id: str, reg: dict | None = None) -> list[str]:
    """Hardware labels for the chips the model passes on, from models.json (e.g. ["TPU v6e"])."""
    m = _row(reg or registry(), model_id)
    return [f"TPU {k}" for k, v in m.get("chips", {}).items() if v.get("status") == "passing"]


def config(model_id: str | None = None, reg: dict | None = None, **over) -> dict:
    """The full config of one module page: models.json defaults, then the page's own keys on top, then the shell's.

    Page keys: model, module, name, maker, line, chips (list of str), links {docs, card, code, upstream, extra:
    [{label, url}]}, theme ("system" | "dark" | "light"), status {state, text}, resources [{title, by, url}],
    sticky (bool, the model bar stays in view; default true), foot_links [{label, url}].
    Shell keys (never set by a page): site [{label, url}], company.
    """
    bad = set(over) - PAGE_KEYS
    if bad:
        raise ValueError(f"unknown config keys {sorted(bad)}; a page may set {sorted(PAGE_KEYS)}")
    reg = reg or registry()
    cfg: dict = {"theme": "system"}
    if model_id:
        m = _row(reg, model_id)
        cfg.update(model=model_id, module=MODULE_NAMES.get(m.get("kind"), m["name"]), name=m["name"],
                   maker=m.get("upstream", {}).get("maker"), line=m["summary"], chips=chips(model_id, reg),
                   links=links(model_id, reg))
    page_links = over.pop("links", None) or {}
    cfg.update({k: v for k, v in over.items() if v is not None})
    cfg["links"] = {**cfg.get("links", {}), **page_links}
    cfg["site"], cfg["company"] = site(reg), COMPANY_LINE
    if cfg["theme"] not in THEMES:
        raise ValueError(f"theme must be one of {THEMES}")
    if "status" in cfg and cfg["status"].get("state") not in STATES:
        raise ValueError(f"status.state must be one of {STATES}")
    return cfg


def page_config(page: bytes | str) -> dict | None:
    """The config block a page carries, as written (None when the page has no block)."""
    raw = page.encode() if isinstance(page, str) else page
    m = BLOCK.search(raw)
    return json.loads(m.group(2)) if m else None


def render(page: bytes | str, reg: dict | None = None) -> bytes:
    """The page with its config block resolved. A page without a block comes back unchanged."""
    raw = page.encode() if isinstance(page, str) else page
    own = page_config(raw)
    if own is None:
        return raw
    own = dict(own)
    cfg = config(own.pop("model", None), reg, **own)
    body = json.dumps(cfg, ensure_ascii=False).replace("</", "<\\/").encode()
    return BLOCK.sub(lambda m: m.group(1) + body + m.group(3), raw, count=1)


def asset(path: str) -> tuple[bytes, str] | None:
    """(bytes, content type) for a request path under ui/shell/, or None when it is not a shell file."""
    path = path.split("?", 1)[0].lstrip("/")
    if not path.startswith(ROUTE):
        return None
    rel = path[len(ROUTE):]
    parts = rel.split("/")
    if not rel or any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else ""
    if ext not in TYPES:
        return None
    f = resources.files(__package__)
    for p in parts:
        f = f.joinpath(p)
    if not f.is_file():
        return None
    return f.read_bytes(), TYPES[ext]


# ---------- the standard, as checks (the CPU test and the release gate run these) ----------

# Markup and styles that belong to the shell alone. A page that carries any of them has copied part of the shell.
COPIES = [
    # --mbx-ar, --mbx-reserve, --mbx-cols and --mbx-acc are layout and accent settings a page sets on its own elements
    (re.compile(r"--mbx-(?!(?:ar|reserve|cols|acc)\s*:)[\w-]+\s*:"), "defines a shell token (--mbx-...: ...); map your own names to var(--mbx-...) instead"),
    (re.compile(r"@font-face"), "declares a font; the shell bundles the fonts"),
    (re.compile(r"Made by MASHBOX Labs", re.I), "writes 'Made by MASHBOX Labs' itself; the shell's top bar and footer say it"),
    (re.compile(r'class="[^"]*\bmbx-(top|top-in|bar|bar-wrap|brand|made|site|foot|foot-in)\b'), "hand-builds shell markup; use <header data-mbx-top> and <footer data-mbx-foot>"),
    (re.compile(re.escape(COMPANY_LINE[:40])), "copies the company line; the shell's footer carries it"),
    (re.compile(r"https?://(fonts\.googleapis|fonts\.gstatic|cdn\.|unpkg\.com|cdnjs\.)"), "loads something from the internet at run time"),
]


def check_page(page: bytes | str) -> list[str]:
    """Problems that keep a module page from meeting the standard (empty list = it meets it)."""
    html = page.decode() if isinstance(page, bytes) else page
    errs = []
    if not re.search(r'<link[^>]+href="(\./)?ui/shell/shell\.css"', html):
        errs.append('does not link ui/shell/shell.css')
    m = re.search(r'<script[^>]+src="(\./)?ui/shell/shell\.js"[^>]*>', html)
    if not m:
        errs.append('does not load ui/shell/shell.js')
    elif re.search(r"\b(defer|async)\b", m.group(0)):
        errs.append('loads shell.js with defer or async; load it plainly in <head> so the theme is set before paint')
    blk = BLOCK.search(html.encode())
    if not blk:
        errs.append('has no <script type="application/json" id="mbx-module"> config block')
    else:
        if m and m.start() < blk.start():
            errs.append('loads shell.js before its config block')
        try:
            own = json.loads(blk.group(2))
            config(own.pop("model", None), **own)
        except (ValueError, KeyError) as e:
            errs.append(f"config block: {e}")
    for hook in ("data-mbx-top", "data-mbx-foot"):
        if hook not in html:
            errs.append(f"has no element with {hook}")
    if not re.search(r'<body[^>]*class="[^"]*\bmbx-ui\b', html):
        errs.append('<body> does not have class "mbx-ui"')
    for rx, why in COPIES:
        if rx.search(html):
            errs.append(why)
    return errs


def check_links(cfg: dict, root: pathlib.Path, reg: dict | None = None) -> list[str]:
    """Every link in a resolved config must point at a real page: a docs page in docs/, a model repo in models.json,
    a path in this repo, the original model, or a mashbox.gg page the shell knows. Offline; the release gate also
    fetches each one (tests/cpu/test_module_shell.py with MBX_CHECK_LINKS_ONLINE=1)."""
    reg = reg or registry()
    root = pathlib.Path(root)
    gh, hf = reg["github"], reg["hf"]
    repos = {m["hf_repo"] for m in reg["models"]} | {m["upstream"]["hf"] for m in reg["models"] if m.get("upstream", {}).get("hf")}
    upstream_gh = {m["upstream"]["github"] for m in reg["models"] if m.get("upstream", {}).get("github")}
    site_urls = {SITE_URL + p for _, p in SITE_PAGES}
    urls = [s["url"] for s in cfg.get("site", [])] + [s["url"] for s in cfg.get("foot_links", [])]
    for k, v in (cfg.get("links") or {}).items():
        urls += [x["url"] for x in v] if k == "extra" else [v]
    urls += [r["url"] for r in cfg.get("resources", [])]
    errs = []
    for u in urls:
        if u.startswith(DOCS):
            rest = u[len(DOCS):].strip("/")
            if not any((root / "docs" / c).is_file() for c in ([f"{rest}.md", f"{rest}/README.md"] if rest else ["README.md"])):
                errs.append(f"{u}: no docs/{rest or 'README'}.md in this repo")
        elif u.rstrip("/") in site_urls or u == SITE_URL + "/":
            continue
        elif u.startswith(f"https://github.com/{gh['org']}/{gh['repo']}/"):
            mm = re.match(rf"https://github\.com/{gh['org']}/{gh['repo']}/(tree|blob)/v{re.escape(reg['version'])}/(.+)$", u)
            if not mm or not (root / mm.group(2)).exists():
                errs.append(f"{u}: not a path of this repo at v{reg['version']}")
        elif u in (f"https://github.com/{gh['org']}", f"https://huggingface.co/{hf['org']}"):
            continue
        elif u.startswith("https://huggingface.co/"):
            if u[len("https://huggingface.co/"):].strip("/") not in repos:
                errs.append(f"{u}: not a model repo in models.json")
        elif u.startswith("https://github.com/"):
            if u[len("https://github.com/"):].strip("/") not in upstream_gh:
                errs.append(f"{u}: not an upstream repo in models.json")
        elif u.startswith("https://arxiv.org/abs/"):
            continue
        else:
            errs.append(f"{u}: not a docs, model, code or mashbox.gg page")
    return errs


def module_pages(root: pathlib.Path) -> list[pathlib.Path]:
    """Every module page in the repo: the apps' pages, the pages packaged with each model, the chat module's page, the
    shell's template and example."""
    root = pathlib.Path(root)
    pages = sorted(root.glob("apps/*/*.html")) + sorted(root.glob("src/mashbox/models/*/*.html"))
    ui = root / "src/mashbox/ui"
    return pages + [ui / "chat/index.html", ui / "shell/template.html", ui / "shell/example.html"]
