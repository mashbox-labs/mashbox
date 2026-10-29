#!/usr/bin/env python3
"""Record one real response per route of a RUNNING mashbox server into docs/api-examples.json (stdlib only).

Run it on (or through a tunnel to) a real chip after `mashbox run <model> --serve`, naming the chip it runs on:

    python3 tools/record_api_examples.py --base http://127.0.0.1:8000 --chip "TPU v6e-1" --zone us-east5-b

Each generating route gets the request in runtime/openapi.py EXAMPLES (non-streaming); GET routes are called as they are.
docs/server-api.md then shows the recorded answer under each route (render --write), and tests/cpu/test_openapi.py checks
every recorded response against the route's schema. Never record from a stub or test server: the page says these are real.
"""
import argparse, datetime, json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mashbox.runtime import openapi  # noqa: E402


def call(base, method, path, body=None):
    req = urllib.request.Request(base.rstrip("/") + path, method=method.upper(),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="URL of the running server")
    ap.add_argument("--chip", required=True, help='the chip the server runs on, e.g. "TPU v6e-1"')
    ap.add_argument("--zone", required=True, help="its Google Cloud zone")
    ap.add_argument("--out", default=str(ROOT / "docs" / "api-examples.json"))
    a = ap.parse_args()
    st, models = call(a.base, "get", "/v1/models")
    if st != 200:
        sys.exit(f"no server at {a.base} (GET /v1/models: HTTP {st})")
    rec = {"recorded_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "chip": a.chip, "zone": a.zone, "model": models["data"][0]["id"], "routes": {}}
    for r in openapi.ROUTES:
        if r["method"] not in ("get", "post") or "{" in r["path"] or r["id"] == "openapi":
            continue
        if r["method"] == "post" and r["id"] not in openapi.EXAMPLES:
            continue
        body = openapi.EXAMPLES.get(r["id"]) if r["method"] == "post" else None
        st, resp = call(a.base, r["method"], r["path"], body)
        rec["routes"][r["id"]] = {"method": r["method"], "path": r["path"], "request": body, "status": st, "response": resp}
        print(f"{st} {r['method'].upper()} {r['path']}")
    Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
