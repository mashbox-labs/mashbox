"""Serve the MASHBOX Chat page for any OpenAI-compatible server.

    python -m mashbox.ui.chat --api http://127.0.0.1:8000          # then open http://127.0.0.1:8801/

The page and its files come from this package; /v1/... and /health go to --api as they are, streamed. Binds to
127.0.0.1 by default. `mashbox run <model> --ui` serves the same page from the model server itself. Every request
passes runtime/guard.py first (Host, Origin, JSON-only POST): this relay sets the upstream Host itself, so without it a
page whose domain is re-pointed at 127.0.0.1 could read the model server's replies through it.
"""
import argparse
import http.client
import json
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ...runtime.guard import Guard
from . import handle


def model_of(api):
    try:
        u = urllib.parse.urlsplit(api)
        c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5)
        c.request("GET", "/v1/models")
        r = c.getresponse()
        return (json.loads(r.read()).get("data") or [{}])[0].get("id") if r.status == 200 else None
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mashbox.ui.chat", description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="the OpenAI-compatible server (default %(default)s)")
    ap.add_argument("--model", help="model id for the page (default: what --api's /v1/models reports)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8801)
    a = ap.parse_args(argv)
    api = urllib.parse.urlsplit(a.api.rstrip("/"))
    if api.scheme != "http" or not api.hostname:
        ap.error("--api must be http://host:port")
    state = {"model": a.model or model_of(a.api)}
    guard = Guard(hosts=[a.host])

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"      # a streamed reply ends when the connection closes

        def refused(self):
            err = guard.check(self.command, self.path, self.headers, auth=False)
            if err:
                self.send_error(err[0], err[1])
            return err is not None

        def proxy(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            c = http.client.HTTPConnection(api.hostname, api.port or 80, timeout=600)
            try:
                hdr = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept", "authorization")}
                c.request(method, self.path, body=body, headers=hdr)
                r = c.getresponse()
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in ("connection", "transfer-encoding", "keep-alive", "content-length"):
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                       # the browser stopped the reply: closing our side tells the server
            except OSError as e:
                try:
                    self.send_error(502, f"server not reachable: {e}")
                except OSError:
                    pass
            finally:
                c.close()

        def do_GET(self):
            if self.refused():
                return
            p = self.path.split("?", 1)[0]
            if p.startswith("/v1/") or p == "/health":
                if p == "/v1/models" and not state["model"]:
                    state["model"] = model_of(a.api)
                return self.proxy("GET")
            if not state["model"] and p in ("/", "/chat", "/index.html"):
                for _ in range(10):                # the server may still be starting: wait up to ~5 s for its model id
                    state["model"] = model_of(a.api)
                    if state["model"]:
                        break
                    time.sleep(0.5)
            hit = handle(p, state["model"] or "Chat model")
            if hit is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", hit[1])
            self.send_header("Content-Length", str(len(hit[0])))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(hit[0])

        def do_POST(self):
            if self.refused():
                return
            if not self.path.startswith("/v1/"):
                return self.send_error(404)
            self.proxy("POST")

        def log_message(self, *x):
            pass

    httpd = ThreadingHTTPServer((a.host, a.port), H)
    print(json.dumps(dict(event="listening", url=f"http://{a.host}:{a.port}/", api=a.api, model=state["model"])), file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
