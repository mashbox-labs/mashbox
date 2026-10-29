"""A fake gcloud for the `mashbox tpu` CPU tests. State and scenario live in the JSON file named by FAKE_GCLOUD_STATE.

Scenario keys: account, project, api (bool), plain {zone: ok|capacity|quota|no-access|vanish|accepted-auth|accepted-internal},
qr_create {zone: ok|quota|accepted-auth|accepted-internal} (accepted-*: Google made it, then the call ended in that error;
accepted-auth also leaves the sign-in expired, so every later call fails as UNAUTHENTICATED),
grant {zone: describes before ACTIVE}, list_error {zone: auth|other}, probe ("404" or "403"), armed (int), server ("running" or ""),
tunnel_fail (bool: the SSH forward cannot be set up).
Every call is appended to state["calls"]. `tpu-vm ssh ... -- -N -L SOCKET:...` serves HTTP 200 ("ok") on that Unix
socket (the tunnel's forward), until killed; with -L 127.0.0.1:L:... (mashbox before the held port) on 127.0.0.1:L, and
it exits as ssh does with ExitOnForwardFailure when L is taken.
"""
import fcntl
import json
import os
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ERR = {
    "capacity": 'ERROR: (gcloud.compute.tpus.tpu-vm.create) {"code": 8, "message": "There is no more capacity in the zone \\"{z}\\"; you can try in another zone where Cloud TPU Nodes are offered"}',
    "quota": "ERROR: (gcloud.compute.tpus.tpu-vm.create) Quota limit 'TPUV6EPerProjectPerZoneForTPUAPI' has been exceeded. Limit: 0 in zone {z}.",
    "internal": 'ERROR: (gcloud.compute.tpus.tpu-vm.create) {\n  "code": 13,\n  "message": "an internal error has occurred"\n}',
    "no-access": 'ERROR: (gcloud.compute.tpus.tpu-vm.create) User does not have permission to submit requests for accelerator type "v6e-1" in location "{z}".',
    "auth": "ERROR: (gcloud.compute.tpus.tpu-vm.create) UNAUTHENTICATED: Request had invalid authentication credentials. "
            "Expected OAuth 2 access token, login cookie or other valid authentication credential.",
}


LIST_ERR = {
    "auth": "ERROR: (gcloud.compute.tpus.tpu-vm.list) UNAUTHENTICATED: Request had invalid authentication credentials. "
            "Expected OAuth 2 access token, login cookie or other valid authentication credential.",
    "other": "ERROR: (gcloud.compute.tpus.tpu-vm.list) INTERNAL: an internal error has occurred",
}


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def opt(args, k, default=None):
    return args[args.index(k) + 1] if k in args else default


def meta(args):
    out = dict(kv.split("=", 1) for kv in (opt(args, "--metadata") or "").split(",") if "=" in kv)
    if opt(args, "--metadata-from-file"):
        out["startup-script"] = "(file)"
    return out


def labels(args):
    return dict(kv.split("=", 1) for kv in (opt(args, "--labels") or "").split(",") if "=" in kv)


def node(name, zone, args, qr=None):
    n = {"name": f"projects/p/locations/{zone}/nodes/{name}", "state": "READY", "createTime": stamp(),
         "acceleratorType": opt(args, "--accelerator-type"), "labels": labels(args), "metadata": meta(args),
         "schedulingConfig": {"spot": True} if "--spot" in args else {}}
    if qr:
        n["queuedResource"] = f"projects/p/locations/{zone}/queuedResources/{qr}"
    return n


def tunnel(args):
    spec = args[args.index("-L") + 1]

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def address_string(self):
            return "tunnel"

        def log_message(self, *a):
            pass
    if spec.startswith("/"):
        sock = spec.split(":")[0]
        if os.path.exists(sock):                         # StreamLocalBindUnlink=yes
            os.unlink(sock)

        class U(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
        U(sock, H).serve_forever()
    try:
        srv = HTTPServer(("127.0.0.1", int(spec.split(":")[1])), H)
    except OSError as e:
        print(f"bind [127.0.0.1]:{spec.split(':')[1]}: {e}\nCould not request local forwarding.", file=sys.stderr)
        print("Error: Port forwarding failed.", file=sys.stderr)
        sys.exit(255)
    srv.serve_forever()


def run(S, args):
    pos = [a for a in args if not a.startswith("--")]
    z = opt(args, "--zone")
    if args[:2] == ["auth", "list"]:
        print(S.get("account", ""))
        return 0
    if args[:3] == ["config", "get-value", "project"]:
        print(S.get("project", ""))
        return 0
    if args[:2] == ["services", "list"]:
        print("tpu.googleapis.com" if S.get("api", True) else "")
        return 0
    if args[:3] != ["compute", "tpus", "tpu-vm"] and args[:3] != ["compute", "tpus", "queued-resources"]:
        print(f"fake gcloud: unknown {args}", file=sys.stderr)
        return 2
    res, verb = args[2], args[3]
    nodes, qrs = S.setdefault("nodes", {}), S.setdefault("qrs", {})
    if S.get("expired"):                         # the sign-in expired during an earlier call (seen 10-28)
        print(LIST_ERR["auth"], file=sys.stderr)
        return 1
    if res == "tpu-vm" and verb == "create":
        name, out = pos[4], S.get("plain", {}).get(z, "capacity")
        if f"{z}/{name}" in nodes:
            print(f"ERROR: (gcloud.compute.tpus.tpu-vm.create) ALREADY_EXISTS: {name} already exists", file=sys.stderr)
            return 1
        if out.startswith("accepted-"):             # Google made it, then the call failed (a 9 min create, seen 10-28)
            nodes[f"{z}/{name}"] = dict(node(name, z, args), state="CREATING")
            S["expired"] = out == "accepted-auth"
            print(ERR[out.split("-", 1)[1]].replace("{z}", z), file=sys.stderr)
            return 1
        if out in ERR:
            print(ERR[out].replace("{z}", z), file=sys.stderr)
            return 1
        if out != "vanish":                      # created, then gone before READY (seen 10-28)
            nodes[f"{z}/{name}"] = node(name, z, args)
        return 0
    if res == "tpu-vm" and verb == "describe":
        n = nodes.get(f"{z}/{pos[4]}")
        if not n:
            print("ERROR: NOT_FOUND", file=sys.stderr)
            return 1
        print(json.dumps(n))
        return 0
    if verb == "list":
        if S.get("list_error", {}).get(z):           # e.g. an expired sign-in (seen in real use)
            print(LIST_ERR[S["list_error"][z]], file=sys.stderr)
            return 1
        rows = nodes if res == "tpu-vm" else qrs
        print(json.dumps([v for k, v in rows.items() if k.startswith(z + "/")]))
        return 0
    if res == "tpu-vm" and verb == "delete":
        n = nodes.pop(f"{z}/{pos[4]}", None)
        if not n:
            print("ERROR: NOT_FOUND: not found", file=sys.stderr)
            return 1
        return 0
    if res == "queued-resources" and verb == "create":
        q, nid, out = pos[4], opt(args, "--node-id"), S.get("qr_create", {}).get(z, "ok")
        if f"{z}/{q}" in qrs:
            print(f"ERROR: (gcloud.compute.tpus.queued-resources.create) ALREADY_EXISTS: {q} already exists", file=sys.stderr)
            return 1
        if out == "quota":
            print(ERR["quota"].replace("{z}", z), file=sys.stderr)
            return 1
        qrs[f"{z}/{q}"] = {"name": f"projects/p/locations/{z}/queuedResources/{q}", "createTime": stamp(),
                           "state": {"state": "WAITING_FOR_RESOURCES"}, "describes": 0, "args": args,
                           "tpu": {"nodeSpec": [{"nodeId": nid, "node": {"acceleratorType": opt(args, "--accelerator-type"),
                                                                       "labels": labels(args), "metadata": meta(args)}}]}}
        if out.startswith("accepted-"):
            S["expired"] = out == "accepted-auth"
            print(ERR[out.split("-", 1)[1]].replace("tpu-vm.create", "queued-resources.create"), file=sys.stderr)
            return 1
        return 0
    if res == "queued-resources" and verb == "describe":
        q = qrs.get(f"{z}/{pos[4]}")
        if not q:
            print("ERROR: NOT_FOUND", file=sys.stderr)
            return 1
        q["describes"] += 1
        g = S.get("grant", {}).get(z)
        if g is not None and q["describes"] >= g and q["state"]["state"] != "ACTIVE":
            q["state"]["state"] = "ACTIVE"
            nid = q["tpu"]["nodeSpec"][0]["nodeId"]
            nodes[f"{z}/{nid}"] = node(nid, z, q["args"], qr=pos[4])
        print(json.dumps(q))
        return 0
    if res == "queued-resources" and verb == "delete":
        q = qrs.pop(f"{z}/{pos[4]}", None)
        if not q:
            print("ERROR: NOT_FOUND: not found", file=sys.stderr)
            return 1
        if "--force" in args:
            nodes.pop(f"{z}/{q['tpu']['nodeSpec'][0]['nodeId']}", None)
        return 0
    if res == "tpu-vm" and verb == "scp":
        dest = next(i for i, a in enumerate(args) if a.endswith(":"))
        S.setdefault("copied", []).extend(os.path.basename(a) for a in args[4:dest])
        return 0
    if res == "tpu-vm" and verb == "ssh":
        if "-L" in args:
            return "tunnel-fail" if S.get("tunnel_fail") else "tunnel"
        cmd = opt(args, "--command") or ""
        if "mashbox-install.sh" in cmd:
            print("    mashbox install: done: /home/u/.local/bin/mashbox (mashbox 0.1.0)")
            print("MBX_DISK free_gb=80 have_gb=80")
            print("MBX_SERVER=" + ("running" if S.get("server") else "started"))
            S["server"] = "running"
            S["starts"] = S.get("starts", 0) + 1
        elif "MBX_PROBE_NODE" in cmd:
            print(f"MBX_PROBE_NODE={S.get('probe', '404')}")
            if "queuedResources/mashbox-permcheck" in cmd and "q=''" not in cmd:
                print(f"MBX_PROBE_QR={S.get('probe', '404')}")
            print(f"MBX_ARMED={S.get('armed', 1)}")
        elif "MBX_ALIVE" in cmd:
            print("MBX_ALIVE=1\nloading")
        return 0
    print(f"fake gcloud: unknown {args}", file=sys.stderr)
    return 2


def main():
    path = os.environ["FAKE_GCLOUD_STATE"]
    args = sys.argv[1:]
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            S = json.load(f)
        S.setdefault("calls", []).append(args)
        rc = run(S, args)
        with open(path, "w") as f:
            json.dump(S, f)
    if rc == "tunnel":
        tunnel(args)
    if rc == "tunnel-fail":                               # as ssh -o ExitOnForwardFailure=yes when it cannot bind
        print("unix_listener: cannot bind to path: Permission denied\nCould not request local forwarding.", file=sys.stderr)
        print("Error: Port forwarding failed.", file=sys.stderr)
        sys.exit(255)
    sys.exit(rc)


if __name__ == "__main__":
    main()
