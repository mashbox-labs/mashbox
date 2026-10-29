"""Run a module page with the shell, for building and reviewing module UIs.

    python -m mashbox.ui.shell serve PAGE [--port 8800]   serve PAGE at http://127.0.0.1:PORT/ (shell files under ui/shell/)
    python -m mashbox.ui.shell build PAGE OUTDIR          write OUTDIR/index.html and OUTDIR/ui/shell/ as static files
    python -m mashbox.ui.shell check [PAGE ...]           the module page rules (default: every module page in this repo)
"""
from __future__ import annotations

import argparse
import http.server
import pathlib
import sys
from importlib import resources

from . import ROUTE, asset, check_links, check_page, module_pages, page_config, render, config


def _files(base, rel=""):
    for f in base.iterdir():
        if f.is_dir():
            if not f.name.startswith(("_", ".")):
                yield from _files(f, f"{rel}{f.name}/")
        elif asset(f"{ROUTE}{rel}{f.name}"):
            yield f"{rel}{f.name}", f.read_bytes()


def build(page: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_bytes(render(page.read_bytes()))
    for rel, data in _files(resources.files(__package__)):
        dst = out / ROUTE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)


def serve(page: pathlib.Path, port: int) -> None:
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(page.parent), **k)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            hit = (render(page.read_bytes()), "text/html; charset=utf-8") if path in ("/", "/index.html") else asset(path)
            if hit is None:
                return super().do_GET()
            body, ctype = hit
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"http://127.0.0.1:{port}/", flush=True)
    srv.serve_forever()


def check(pages: list[pathlib.Path], root: pathlib.Path) -> int:
    bad = 0
    for p in pages:
        html = p.read_text()
        errs = check_page(html)
        if not errs:
            own = dict(page_config(html))
            errs = check_links(config(own.pop("model", None), **own), root)
        print(f"{'FAIL' if errs else 'ok  '} {p}")
        for e in errs:
            print(f"     {e}")
        bad += bool(errs)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mashbox.ui.shell")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("page", type=pathlib.Path); s.add_argument("--port", type=int, default=8800)
    b = sub.add_parser("build"); b.add_argument("page", type=pathlib.Path); b.add_argument("out", type=pathlib.Path)
    c = sub.add_parser("check"); c.add_argument("pages", type=pathlib.Path, nargs="*"); c.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.page.resolve(), a.port)
    elif a.cmd == "build":
        build(a.page, a.out)
    else:
        return check(a.pages or module_pages(a.root), a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
