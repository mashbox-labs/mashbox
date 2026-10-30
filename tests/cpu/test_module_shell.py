"""The module shell standard: every module page loads the one shared shell, shows "Made by MASHBOX Labs", what the module
and model are, and links back; no page carries a copy of a shell part; every link resolves to a real docs, model or
code page. Offline by default. MBX_CHECK_LINKS_ONLINE=1 also fetches every link (the release gate runs it that way)."""
import os
import pathlib
import re
import tomllib
import urllib.request

import pytest

from mashbox.ui import shell

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHELL = ROOT / "src/mashbox/ui/shell"
PAGES = shell.module_pages(ROOT)


def resolved(page: pathlib.Path) -> dict:
    own = dict(shell.page_config(page.read_text()))
    return shell.config(own.pop("model", None), **own)


def test_pages_found():
    names = {p.relative_to(ROOT).as_posix() for p in PAGES}
    assert {"src/mashbox/models/hy_nomas/page.html", "src/mashbox/ui/shell/example.html", "src/mashbox/ui/shell/template.html"} <= names


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_page_loads_the_shell(page):
    assert shell.check_page(page.read_text()) == []


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_page_links_resolve(page):
    cfg = resolved(page)
    assert cfg["name"] and cfg["module"] and cfg["line"] and cfg["links"]["docs"], "the model bar needs model, module, line and docs"
    assert shell.check_links(cfg, ROOT) == []


@pytest.mark.skipif(os.environ.get("MBX_CHECK_LINKS_ONLINE") != "1", reason="online link check runs in the release gate")
def test_page_links_answer_online():
    urls = sorted({u for p in PAGES for u in _urls(resolved(p))})
    dead = []
    for u in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "mashbox-link-check"}), timeout=20) as r:
                if r.status >= 400:
                    dead.append(f"{u} {r.status}")
        except Exception as e:  # noqa: BLE001 - any failure is a dead link
            dead.append(f"{u} {e}")
    assert dead == []


def _urls(cfg):
    out = [s["url"] for s in cfg["site"]] + [s["url"] for s in cfg.get("foot_links", [])] + [r["url"] for r in cfg.get("resources", [])]
    for k, v in cfg["links"].items():
        out += [x["url"] for x in v] if k == "extra" else [v]
    return out


GOOD = (SHELL / "template.html").read_text()


@pytest.mark.parametrize("bad, why", [
    (GOOD.replace('<link rel="stylesheet" href="ui/shell/shell.css">', ""), "shell.css"),
    (GOOD.replace('<script src="ui/shell/shell.js"></script>', ""), "shell.js"),
    (GOOD.replace('<script src="ui/shell/shell.js"></script>', '<script src="ui/shell/shell.js" defer></script>'), "defer"),
    (GOOD.replace('id="mbx-module"', 'id="other"'), "config block"),
    (GOOD.replace('"model": "qwen3-8-27b"', '"model": "no-such-model"'), "no model"),
    (GOOD.replace('"module": "Chat"', '"module": "Chat", "company": "x"'), "unknown config keys"),
    (GOOD.replace("<header data-mbx-top></header>", '<header class="mbx-top">Made by MASHBOX Labs</header>'), "data-mbx-top"),
    (GOOD.replace("<footer data-mbx-foot></footer>", ""), "data-mbx-foot"),
    (GOOD.replace('<body class="mbx-ui">', "<body>"), "mbx-ui"),
    (GOOD.replace("<style>", "<style>:root { --mbx-bg: #000; }"), "shell token"),
    (GOOD.replace("<style>", '<style>@font-face { font-family: X; src: url("x.woff2"); }'), "font"),
    (GOOD.replace("</head>", '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=X"></head>'), "internet"),
    (GOOD.replace("</main>", "<p>" + shell.COMPANY_LINE + "</p></main>"), "company line"),
])
def test_check_page_fails_bad_pages(bad, why):
    errs = shell.check_page(bad)
    assert any(why in e for e in errs), errs


def test_layout_settings_are_not_copies():
    page = GOOD.replace("<style>", "<style>#x { --mbx-ar: 1.8; --mbx-reserve: 200px; --mbx-cols: 2; --mbx-acc: var(--mbx-ok); }")
    assert shell.check_page(page) == []


def test_check_links_fails_dead_links():
    cfg = shell.config("qwen3-8-27b", links={"docs": shell.DOCS + "/models/nope", "card": "https://huggingface.co/mashbox-labs/Nope",
                                             "code": "https://github.com/mashbox-labs/mashbox/tree/v0.1.0/src/nope",
                                             "extra": [{"label": "x", "url": "https://example.com/"}]})
    errs = shell.check_links(cfg, ROOT)
    for part in ("docs/models/nope.md", "Nope: not a model repo", "src/nope: not a path", "example.com"):
        assert any(part in e for e in errs), (part, errs)


def test_render_fills_the_bar_from_the_registry():
    out = shell.render((SHELL / "example.html").read_bytes()).decode()
    cfg = shell.page_config(out)
    reg = shell.registry()
    row = next(m for m in reg["models"] if m["id"] == "qwen3-8-27b")
    assert (cfg["name"], cfg["maker"], cfg["line"]) == (row["name"], row["upstream"]["maker"], row["summary"])
    assert cfg["company"] == shell.COMPANY_LINE and cfg["site"][0]["url"] == "https://mashbox.gg/"
    assert {"Docs", "Models", "GitHub", "Hugging Face"} <= {s["label"] for s in cfg["site"]}


def test_company_line_is_exact():
    assert shell.COMPANY_LINE == ("MASHBOX Labs makes open models run faster and cheaper on TPUs and accelerated hardware, "
                                  "as well as exploring the future of the intersection of AI and gaming.")


def test_shell_files_are_bundled_and_offline():
    css, js = (SHELL / "shell.css").read_text(), (SHELL / "shell.js").read_text()
    for font in re.findall(r'url\("(fonts/[^"]+)"\)', css):
        assert (SHELL / font).is_file(), font
    assert not re.search(r"url\(\s*['\"]?https?:", css)
    assert not re.search(r"\b(fetch|XMLHttpRequest|import\s*\(|WebSocket|sendBeacon)\b", js)
    assert "Made by MASHBOX Labs" in js
    for lic in ("OFL-ChakraPetch.txt", "OFL-ShareTechMono.txt", "OFL-Outfit.txt", "OFL-Geist.txt", "OFL-BigShoulders.txt"):
        assert (SHELL / "fonts" / lic).is_file()


def test_asset_route_serves_shell_files_only():
    assert shell.asset("/ui/shell/shell.css")[1].startswith("text/css")
    assert shell.asset("/ui/shell/shell.js?v=2")[1].startswith("text/javascript")
    assert shell.asset("/ui/shell/fonts/ChakraPetch-700.woff2")[1] == "font/woff2"
    for bad in ("/ui/shell/../models.json", "/ui/shell/__init__.py", "/ui/shell/example.html", "/ui/shell/", "/other/shell.css",
                "/ui/shell/.hidden.css", "/ui/shell/nope.css"):
        assert shell.asset(bad) is None, bad


def test_package_ships_every_shell_file():
    pats = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["mashbox"]
    base = ROOT / "src/mashbox"
    for f in SHELL.rglob("*"):
        if f.is_file() and f.suffix != ".py" and "__pycache__" not in f.parts:
            rel = f.relative_to(base)
            assert any(rel.match(p) for p in pats), f"{rel} is not in package-data"


def test_model_servers_serve_the_shell():
    for page in ROOT.glob("src/mashbox/models/*/page.html"):
        src = "\n".join(p.read_text() for p in page.parent.glob("*.py"))
        assert "shell.render(" in src and "shell.asset(" in src, f"{page.parent.name}: renders its page without the shell route"


def test_no_copies_of_the_shell_elsewhere():
    found = []
    for f in list(ROOT.glob("apps/**/*")) + list(ROOT.glob("src/**/*")):
        if not f.is_file() or SHELL in f.parents or f.suffix not in (".html", ".css", ".js"):
            continue
        text = f.read_text(errors="ignore")
        if f.name in ("shell.css", "shell.js") or re.search(r"--mbx-(?!(?:ar|reserve|cols|acc)\s*:)[\w-]+\s*:", text) \
                or "Made by MASHBOX Labs" in text:
            found.append(f.relative_to(ROOT).as_posix())
    assert found == []
