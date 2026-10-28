"""A fake gcloud for the `mashbox tpu` CPU tests. State and scenario live in the JSON file named by FAKE_GCLOUD_STATE.

Scenario keys: account, project, api (bool), plain {zone: ok|capacity|quota|no-access|vanish}, qr_create {zone: ok|quota},
grant {zone: describes before ACTIVE}, probe ("404" or "403"), armed (int), server ("running" or "").
Every call is appended to state["calls"]. `tpu-vm ssh ... -- -N -L 127.0.0.1:L:...` serves HTTP 200 on 127.0.0.1:L (the
tunnel), until killed.
"""
import fcntl
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ERR = {
    "capacity": 'ERROR: (gcloud.compute.tpus.tpu-vm.create) {"code": 8, "message": "There is no more capacity in the zone \\"{z}\\"; you can try in another zone where Cloud TPU Nodes are offered"}',
    "quota": "ERROR: (gcloud.compute.tpus.tpu-vm.create) Quota limit 'TPUV6EPerProjectPerZoneForTPUAPI' has been exceeded. Limit: 0 in zone {z}.",
    "no-access": 'ERROR: (gcloud.compute.tpus.tpu-vm.create) User does not have permission to submit requests for accelerator type "v6e-1" in location "{z}".',
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
    port = int(args[args.index("-L") + 1].split(":")[1])

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()


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
    if res == "tpu-vm" and verb == "create":
        name, out = pos[4], S.get("plain", {}).get(z, "capacity")
        if f"{z}/{name}" in nodes:
            print(f"ERROR: (gcloud.compute.tpus.tpu-vm.create) ALREADY_EXISTS: {name} already exists", file=sys.stderr)
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
        q, nid = pos[4], opt(args, "--node-id")
        if S.get("qr_create", {}).get(z) == "quota":
            print(ERR["quota"].replace("{z}", z), file=sys.stderr)
            return 1
        qrs[f"{z}/{q}"] = {"name": f"projects/p/locations/{z}/queuedResources/{q}", "createTime": stamp(),
                           "state": {"state": "WAITING_FOR_RESOURCES"}, "describes": 0, "args": args,
                           "tpu": {"nodeSpec": [{"nodeId": nid, "node": {"acceleratorType": opt(args, "--accelerator-type"),
                                                                       "labels": labels(args), "metadata": meta(args)}}]}}
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
            return "tunnel"
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
    sys.exit(rc)


if __name__ == "__main__":
    main()
