"""mashbox tpu: a Cloud TPU chip from your own machine, with your own gcloud.

  mashbox tpu up <model> [--zone Z] [--spot] [--name N] [--max-hours H]   chip, install, server, tunnel to 127.0.0.1
  mashbox tpu status                                                      every mashbox chip: age, cost so far
  mashbox tpu tunnel [name]                                               reopen the tunnel
  mashbox tpu ssh [name]                                                  a shell on the chip
  mashbox tpu down [name]                                                 delete the chip and any queued request

`up` checks gcloud (signed in, project, TPU API), then creates the chip type the model's `cloud` row in models.json
names: a plain on-demand create in the row's zones in order, and when none of the first PLAIN_TRIES has a free chip,
on-demand queued requests in up to QUEUE_ZONES zones at once (the first one granted is kept, the others deleted).
It installs mashbox on the chip with this package's install.sh, starts the model's `run` command in the background
(stopped by SIGINT to its process group, the same as Ctrl-C; never SIGTERM) and holds an SSH tunnel to 127.0.0.1.

Max lifetime (--max-hours, default 8, 0 = off): life.sh goes into the chip's metadata as its startup script. At boot it
arms a timer on the chip, and at the deadline the chip deletes itself through the TPU API with its own service account,
so it works with this machine closed. `up` probes whether that service account may delete TPUs and says so when not.

The tunnel: this process binds the local port itself and keeps it for the whole session (tunnel drops included), and
relays each connection to the SSH forward, which listens on a Unix socket in a private (0700) directory. So no other
user of this machine can take the port or the forward and answer in the chip's place, and "Server ready" means the chip's
server answered through our own SSH process. Chip names, zones and projects are checked before any gcloud call.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from importlib import resources

POLL = float(os.environ.get("MASHBOX_TPU_POLL", "30"))      # seconds between checks while waiting
PLAIN_TRIES = 3                                              # zones tried with a plain create before queued requests
QUEUE_ZONES = 3                                              # queued requests made at once
READY_MIN = 45                                               # minutes to wait for the server before giving up
PRICE = {"v6e": 2.70, "v5litepod": 1.20}                     # USD per chip-hour, on demand
PRICE_NOTE = "ESTIMATE: Google's on-demand list price per chip-hour, US regions, October 2026; Google bills the real price"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
STATE = os.path.join(os.path.expanduser("~"), ".config", "mashbox", "tpu.json")
REMOTE_MB = "~/.local/bin/mashbox"
NAME = re.compile(r"[a-z]([-a-z0-9]{0,61}[a-z0-9])?")                    # a TPU node name (RFC 1035 label)
ZONE = re.compile(r"[a-z]+-[a-z]+[0-9]+-[a-z]")                           # us-east5-b, europe-west4-a, asia-northeast1-b
PROJECT = re.compile(r"([a-z][-a-z0-9.]{0,62}:)?[a-z][-a-z0-9]{4,28}[a-z0-9]")   # my-project, example.com:my-project


def checked(rx, what):
    """An argparse type: the value must match rx in full, so it can never be read as a gcloud option or another chip."""
    def check(v):
        if not rx.fullmatch(v):
            raise argparse.ArgumentTypeError(f"not a valid {what}: {v!r}")
        return v
    check.__name__ = what
    return check


class Stop(Exception):
    """A problem the person fixes; the message says how."""


def say(s=""):
    print(s, flush=True)


def now():
    return time.time()


def mins(s):
    return f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s"


# --------------------------------------------------------------------------------------------------------------- gcloud
def gcloud():
    g = os.environ.get("MASHBOX_GCLOUD") or shutil.which("gcloud")
    if not g:
        raise Stop("gcloud is not installed. Install the Google Cloud CLI (https://cloud.google.com/sdk/docs/install), "
                   "then run: gcloud auth login")
    return g


def gc(args, project=None, timeout=None):
    """Run gcloud without a terminal; returns CompletedProcess (stdout, stderr as text)."""
    if project:
        cut = args.index("--") if "--" in args else len(args)
        args = args[:cut] + ["--project", project] + args[cut:]
    try:
        return subprocess.run([gcloud()] + args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"gcloud timed out after {timeout} s")


def gj(args, project=None):
    r = gc(args + ["--format=json"], project)
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout or "[]")
    except ValueError:
        return None


def kind(err):
    """What a gcloud error means for `up`."""
    e = (err or "").lower()
    if "does not have permission to submit requests for accelerator type" in e:
        return "no-access"
    if any(s in e for s in ("no more capacity", "insufficient capacity", "stockout", "resource_pool_exhausted",
                            "does not have enough resources", "currently unavailable")):
        return "capacity"
    if "quota" in e:
        return "quota"
    if "already exists" in e:
        return "exists"
    if "tpu.googleapis.com" in e and ("has not been used" in e or "disabled" in e or "service_disabled" in e):
        return "api"
    if "billing" in e:
        return "billing"
    if any(s in e for s in ("gcloud auth login", "reauthentication", "no credentialed accounts", "active account",
                            "unauthenticated", "invalid authentication credentials")):
        return "auth"
    return "other"


def gerr(err):
    """One readable line from a gcloud error: its JSON "message" when it has one (the last line is often just "}")."""
    e = (err or "").strip()
    msgs = re.findall(r'"message":\s*"((?:[^"\\]|\\.)*)"', e)
    if msgs:
        return msgs[-1].replace('\\"', '"')[:300]
    lines = [x.strip() for x in e.splitlines() if x.strip().strip("{}[],")]
    return ([x for x in lines if x.startswith("ERROR:")] or lines or ["(no message)"])[-1][:300]


def fix(k, project, zone=None, acc=None):
    return {
        "api": f"the Cloud TPU API is off. Run: gcloud services enable tpu.googleapis.com --project {project}",
        "auth": "gcloud is not signed in (or its sign-in expired). Run: gcloud auth login",
        "billing": f"project {project} has no billing account. Link one: https://console.cloud.google.com/billing/linkedaccount?project={project}",
        "quota": f"no TPU quota for {acc} in {zone}. Ask for it: https://console.cloud.google.com/iam-admin/quotas?project={project} (filter: {acc.split('-')[0] if acc else 'tpu'}, {zone})",
        "no-access": f"project {project} may not create on-demand {acc} in {zone} (no quota there). Ask for quota: https://console.cloud.google.com/iam-admin/quotas?project={project}",
    }.get(k, "")


def preflight(project=None):
    gcloud()
    r = gc(["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"])
    token = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE")      # gcloud's own token-file mode lists no account
    if not token and (r.returncode or not r.stdout.strip()):
        raise Stop("gcloud is not signed in. Run: gcloud auth login")
    project = project or gc(["config", "get-value", "project"]).stdout.strip()
    if not project or project == "(unset)":
        raise Stop("gcloud has no project set. See yours with: gcloud projects list   then run: gcloud config set project <PROJECT_ID>")
    r = gc(["services", "list", "--enabled", "--filter=config.name=tpu.googleapis.com", "--format=value(config.name)"], project)
    if r.returncode == 0 and "tpu.googleapis.com" not in r.stdout:
        raise Stop(fix("api", project))
    if r.returncode and kind(r.stderr) in ("api", "auth", "billing"):
        raise Stop(fix(kind(r.stderr), project))
    return project


# --------------------------------------------------------------------------------------------------------------- models
def registry():
    return json.loads(resources.files("mashbox").joinpath("models.json").read_text())


def model(name):
    key = name.replace(".", "-").lower()
    for m in registry()["models"]:
        if key in (m["id"], m["name"].replace(".", "-").lower()):
            if "cloud" not in m:
                raise Stop(f"{m['id']}: no TPU settings in this version of mashbox")
            return m
    raise Stop(f"unknown model {name!r}. Models: " + ", ".join(m["id"] for m in registry()["models"] if "cloud" in m))


def all_zones():
    z = [z for m in registry()["models"] for z in m.get("cloud", {}).get("zones", [])]
    z += [v.get("zone") for v in load_state().values() if v.get("zone")]
    return list(dict.fromkeys(z))


def chips(acc):
    fam, n = acc.rsplit("-", 1)
    return fam, int(n)


def price(acc):
    fam, n = chips(acc)
    return PRICE[fam] * n if fam in PRICE else None


def money(x):
    return "unknown" if x is None else f"${x:,.2f}"


# ---------------------------------------------------------------------------------------------------------------- state
def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(name, **kv):
    s = load_state()
    if kv:
        s.setdefault(name, {}).update(kv)
    else:
        s.pop(name, None)
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + ".part", "w") as f:
        json.dump(s, f, indent=1)
    os.replace(STATE + ".part", STATE)


# -------------------------------------------------------------------------------------------------------------- listing
def short(full):
    return (full or "").rsplit("/", 1)[-1]


def created(x):
    t = (x.get("createTime") or "").split(".")[0].rstrip("Z")
    try:
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def qr_node(q):
    spec = ((q.get("tpu") or {}).get("nodeSpec") or [{}])[0]
    return spec.get("nodeId"), spec.get("node") or {}


def scan(project, zones, unchanged=True):
    """Every mashbox node and queued request in these zones: ([(zone, node)], [(zone, qr)])."""
    out, lock, errs = ([], []), threading.Lock(), []
    mine = set(load_state())

    def listed(what, z):
        for _ in range(2):                              # one retry: a single failed call must not read as "nothing up"
            r = gc(["compute", "tpus", what, "list", "--zone", z, "--format=json"], project)
            if r.returncode == 0:
                try:
                    return json.loads(r.stdout or "[]")
                except ValueError:
                    r = subprocess.CompletedProcess(r.args, 1, "", "unreadable gcloud output")
        with lock:
            errs.append((z, r.stderr))
        return None

    def one(z):
        ns, qs = listed("tpu-vm", z), listed("queued-resources", z)
        if ns is None or qs is None:
            return
        with lock:
            out[0].extend((z, n) for n in ns if (n.get("labels") or {}).get("mashbox") or short(n.get("name")) in mine)
            for q in qs:
                nid, node = qr_node(q)
                if (node.get("labels") or {}).get("mashbox") or (q.get("labels") or {}).get("mashbox") or nid in mine:
                    out[1].append((z, q))
    ts = [threading.Thread(target=one, args=(z,)) for z in zones]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:                                            # never report "no chips" (or forget one) on a failed list
        k = next((kind(e) for _, e in errs if kind(e) in ("auth", "api", "billing")), None)
        last = gerr(errs[0][1])
        raise Stop((fix(k, project) + "\n" if k else "") + "could not list chips in " + ", ".join(sorted({z for z, _ in errs}))
                   + f", so this cannot say what runs and bills (gcloud: {last}). "
                   + ("Nothing was changed. " if unchanged else "")
                   + "To look in one zone only, add --zone ZONE. "
                   + f"The console lists them too: https://console.cloud.google.com/compute/tpus?project={project}")
    return out


def prune(nodes, qrs, zones):
    """Forget local entries for chips that are gone (deleted by their max lifetime, the console or another machine)."""
    seen = {short(x.get("name")) for _, x in nodes} | {qr_node(x)[0] for _, x in qrs}
    for name, v in load_state().items():
        if v.get("zone") in zones and name not in seen:
            save_state(name)


def find(project, name, zones, unchanged=True):
    nodes, qrs = scan(project, zones, unchanged)
    n = next(((z, x) for z, x in nodes if short(x.get("name")) == name), None)
    q = next(((z, x) for z, x in qrs if qr_node(x)[0] == name), None)
    return n, q


# --------------------------------------------------------------------------------------------------------------- create
def life_script():
    return resources.files("mashbox.tpu").joinpath("life.sh").read_text()


def create_args(m, a, project, zone, name, qr, script_path):
    c = m["cloud"]
    meta = dict(mashbox_max_seconds=int(a.max_hours * 3600), mashbox_node=name, mashbox_zone=zone,
                mashbox_project=project, mashbox_qr=qr or "")
    args = ["--zone", zone, "--accelerator-type", c["accelerator"],
            "--labels", f"mashbox=tpu-up,mashbox-model={m['id']}",
            "--metadata", ",".join(f"{k.replace('_', '-')}={v}" for k, v in meta.items()),
            "--metadata-from-file", f"startup-script={script_path}", "--scopes", SCOPE]
    return args + (["--spot"] if a.spot else [])


def wait_ready_node(project, zone, name, timeout=900):
    t0, misses = now(), 0
    while now() - t0 < timeout:
        x = gj(["compute", "tpus", "tpu-vm", "describe", name, "--zone", zone], project)
        st = (x or {}).get("state")
        if st == "READY":
            return x
        misses = misses + 1 if x is None else 0          # 3 failed describes in a row: gone (one can be a blip)
        if misses >= 3 or st in ("PREEMPTED", "TERMINATED", "STOPPED", "STOPPING", "DELETING"):
            return None
        time.sleep(min(POLL, 10))
    return None


def get_chip(m, a, project, name):
    """Create the node (plain create, then queued requests). Returns (zone, qr name or None)."""
    c = m["cloud"]
    zones = [a.zone] if a.zone else list(c["zones"])
    tried = {}
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(life_script())
        script = f.name
    try:
        for z in zones[:1 if a.zone else PLAIN_TRIES]:
            t0 = now()
            say(f"  {z}: creating {c['accelerator']} (a zone with no free chip answers in about 2 min)")
            r = gc(["compute", "tpus", "tpu-vm", "create", name, "--version", c["runtime"]]
                   + create_args(m, a, project, z, name, None, script), project)
            k = "ok" if r.returncode == 0 else kind(r.stderr)
            if k == "exists":                                   # not made by this command: never waited for, used or deleted
                raise Stop(f"a TPU named {name} already exists in {z} and was not made by this command; it is left as "
                           "it is. Choose another name with --name (mashbox tpu status lists the chips mashbox made)")
            # Google may hold a node of ours in z after any answer that is not a clear refusal: seen 10-28, the
            # sign-in expired during a 9 min create, the command stopped and Google went on creating the chip.
            sent = k in ("ok", "auth", "other")
            if k == "ok":
                if wait_ready_node(project, z, name):
                    say(f"  {z}: chip created ({mins(now() - t0)})")
                    return z, None
                k = "capacity"                                  # created, then gone before READY: seen 10-28
            tried[z] = k
            say(f"  {z}: {dict(capacity='no free chip', quota='no quota', other='failed').get(k, k)} ({mins(now() - t0)})")
            if k == "other":
                say("    Google said: " + gerr(r.stderr))
            if sent and not drop_node(project, z, name):
                raise Stop(left_behind(project, z, name, k))
            if k in ("api", "auth", "billing"):
                raise Stop(fix(k, project))
            if k in ("quota", "no-access"):
                say("    " + fix(k, project, z, c["accelerator"]))
        if a.queue_minutes <= 0:
            raise Stop("no chip in the zones tried, and queued requests are off (--queue-minutes 0)")
        qz = [z for z in zones if tried.get(z) not in ("quota", "no-access")][:QUEUE_ZONES]
        if not qz:
            raise Stop("no zone with quota for " + c["accelerator"] + ". " + fix("quota", project, ", ".join(zones), c["accelerator"]))
        return queue(m, a, project, name, qz, script)
    finally:
        os.unlink(script)


def drop_node(project, zone, name):
    """Delete, by its exact name, a node that a create of this command may have made. True when it is gone or never was."""
    r = gc(["compute", "tpus", "tpu-vm", "delete", name, "--zone", zone, "--quiet"], project)
    if r.returncode == 0:
        say(f"  {zone}: {name} deleted (the create had gone through)")
        return True
    if "not found" in (r.stderr or "").lower():
        return True
    say(f"  {zone}: {name} NOT deleted: {gerr(r.stderr)}")
    return False


def left_behind(project, zone, name, k):
    return (f"the create of {name} in {zone} may have reached Google, and a chip bills until it is deleted. "
            + (fix(k, project) + ", then delete it: " if k == "auth" else "Delete it: ")
            + f"mashbox tpu down {name} (or: gcloud compute tpus tpu-vm delete {name} --zone {zone} --project {project} --quiet)")


def queue(m, a, project, name, zones, script):
    c, qr, made = m["cloud"], name + "-q", []
    say(f"No free chip right now. Asking Google for one with on-demand queued requests in {', '.join(zones)} "
        f"(up to {a.queue_minutes:g} min; Google decides when; nothing bills until a chip is granted).")
    say("If two are granted at the same moment, the second is deleted at once and bills only for those minutes.")
    t0 = now()
    try:
        for z in zones:
            r = gc(["compute", "tpus", "queued-resources", "create", qr, "--node-id", name, "--runtime-version", c["runtime"],
                    "--valid-until-duration", f"{max(1, int(a.queue_minutes))}m"]
                   + create_args(m, a, project, z, name, qr, script), project)
            k = "ok" if r.returncode == 0 else kind(r.stderr)
            if k == "ok":
                made.append(z)
                say(f"  {z}: queued request {qr} made")
            elif k in ("auth", "other"):                          # not a clear refusal: it may have been made, so it
                made.append(z)                                    # is watched like the others and deleted at the end
                say(f"  {z}: queued request {qr}: error ({gerr(r.stderr)}); it may have been made, so it counts as made "
                    "(watched, and deleted by name when this command ends)")
            elif k == "exists":                                   # not made by this command: never used or deleted
                say(f"  {z}: a queued request named {qr} already exists (not made by this command); left as it is")
            else:
                say(f"  {z}: queued request refused ({k}) " + fix(k, project, z, c["accelerator"]))
            if k in ("api", "auth", "billing"):
                raise Stop(fix(k, project))
        if not made:
            raise Stop("no queued request is waiting (messages above)")
        last = 0
        while True:
            for z in list(made):
                x = gj(["compute", "tpus", "queued-resources", "describe", qr, "--zone", z], project) or {}
                st = (x.get("state") or {}).get("state", "UNKNOWN")
                if st == "ACTIVE":
                    say(f"  {z}: granted after {mins(now() - t0)}")
                    for o in made:
                        if o != z:
                            delete_qr(project, o, qr)
                    made.clear()
                    if not wait_ready_node(project, z, name):
                        raise Stop(f"the granted chip in {z} did not reach READY; take it down: mashbox tpu down {name}")
                    return z, qr
                if st in ("FAILED", "SUSPENDED"):
                    say(f"  {z}: Google ended the request ({st})")
                    delete_qr(project, z, qr)
                    made.remove(z)
            if not made:
                raise Stop("every queued request ended without a chip. Try again later, or another zone with --zone")
            if now() - t0 > a.queue_minutes * 60:
                raise Stop(f"no chip after {a.queue_minutes:g} min of waiting. Chip waits are outside our control; "
                           "try again later, or another zone with --zone")
            if now() - last >= 60:
                say(f"  waiting for a chip: {mins(now() - t0)} so far")
                last = now()
            time.sleep(POLL)
    finally:
        if made:
            say("Deleting the queued requests this command made, so none is granted later:")
            for z in made:
                delete_qr(project, z, qr)


def delete_qr(project, zone, qr):
    r = gc(["compute", "tpus", "queued-resources", "delete", qr, "--zone", zone, "--force", "--quiet"], project)
    gone = r.returncode == 0 or "not found" in (r.stderr or "").lower()
    say(f"  {zone}: queued request {qr} " + ("deleted" if r.returncode == 0 else "not there" if gone else
        f"NOT deleted: {gerr(r.stderr)}. Delete it: gcloud compute tpus queued-resources delete {qr} --zone {zone} "
        f"--project {project} --force --quiet"))
    return gone


# ------------------------------------------------------------------------------------------------------------- the chip
LAUNCH = r"""
import os, signal, subprocess, sys
os.setsid()
open(os.path.expanduser("~/.mashbox/serve.pid"), "w").write(str(os.getpid()))
signal.signal(signal.SIGINT, signal.SIG_DFL)    # started with &, SIGINT arrives ignored; Ctrl-C must stop it
mb, model, run = sys.argv[1], sys.argv[2], sys.argv[3:]
rc = subprocess.call([mb, "download", model])
if rc:
    sys.exit(rc)
os.execv(mb, [mb] + run)
"""


def remote_setup(m, project, zone, name, extra=(), restart=False):
    """Copy install.sh (and staging files), install, check disk, start the server unless it runs. Returns dict."""
    c = m["cloud"]
    tmp = tempfile.mkdtemp(prefix="mashbox-tpu-")
    try:
        files = [os.path.join(tmp, "mashbox-install.sh")]
        with open(files[0], "w") as f:
            f.write(resources.files("mashbox.tpu").joinpath("install.sh").read_text())
        env = dict(MASHBOX_EXTRAS=c.get("extras", "tpu"))
        wheel = os.environ.get("MASHBOX_WHEEL", "")
        if wheel and os.path.isfile(wheel):                 # pre-release tests: a wheel on this machine
            files.append(os.path.abspath(wheel))
            env["MASHBOX_WHEEL"] = "$HOME/" + os.path.basename(wheel)
        elif wheel:
            env["MASHBOX_WHEEL"] = wheel
        else:
            from .. import __version__
            env["MASHBOX_VERSION"] = __version__
        if os.environ.get("MASHBOX_INDEX_URL"):
            env["MASHBOX_INDEX_URL"] = os.environ["MASHBOX_INDEX_URL"]
        tok = os.environ.get("MASHBOX_HF_TOKEN_FILE")       # pre-release tests: private Hugging Face repo
        if tok:
            files.append(os.path.join(tmp, "mashbox-hf-token"))
            shutil.copyfile(tok, files[-1])
        t0 = now()
        r = gc(["compute", "tpus", "tpu-vm", "scp"] + files + [f"{name}:", "--zone", zone, "--quiet"], project)
        if r.returncode:
            raise Stop("could not copy files to the chip: " + r.stderr.strip()[-300:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    run = shlex.split(m["run"])[1:] + list(extra)
    envs = " ".join(f"{k}={v if v.startswith('$HOME/') else shlex.quote(v)}" for k, v in env.items())
    script = f"""set -e
mkdir -p ~/.mashbox
if [ -f ~/mashbox-hf-token ]; then mkdir -p ~/.cache/huggingface; [ -s ~/.cache/huggingface/token ] || cp ~/mashbox-hf-token ~/.cache/huggingface/token; chmod 600 ~/.cache/huggingface/token; rm -f ~/mashbox-hf-token; fi
{envs} sh ~/mashbox-install.sh > ~/.mashbox/install.log 2>&1 || {{ tail -n 20 ~/.mashbox/install.log; exit 3; }}
grep '^mashbox install:' ~/.mashbox/install.log | sed 's/^/    /' || true
free=$(df -Pk ~ | awk 'NR==2{{print int($4/976562)}}')
used=$(du -sk ~/.cache/huggingface ~/.cache/mashbox 2>/dev/null | awk '{{s+=$1}} END{{print int(s/976562)}}')
echo "MBX_DISK free_gb=$free have_gb=$((free + used))"
pid=$(cat ~/.mashbox/serve.pid 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  if [ {int(restart)} = 0 ]; then echo "MBX_SERVER=running"; echo "MBX_ARGS=$(cat ~/.mashbox/serve.args 2>/dev/null)"; exit 0; fi
  kill -INT -- "-$pid"; i=0                       # Ctrl-C to its process group, never SIGTERM
  while kill -0 "$pid" 2>/dev/null && [ $i -lt 300 ]; do sleep 1; i=$((i+1)); done
  if kill -0 "$pid" 2>/dev/null; then echo MBX_SERVER=stuck; exit 0; fi
  sleep 10                                        # a start right after a stop can fail to open the TPU
fi
if [ $((free + used)) -lt {int(c.get("disk_gb", 0))} ]; then echo MBX_SERVER=disk; exit 0; fi
echo "=== start $(date -u +%FT%TZ)" >> ~/.mashbox/serve.log
echo {shlex.quote(' '.join(run))} > ~/.mashbox/serve.args
nohup python3 -c {shlex.quote(LAUNCH)} {REMOTE_MB} {shlex.quote(m['id'])} {' '.join(shlex.quote(x) for x in run)} >> ~/.mashbox/serve.log 2>&1 < /dev/null &
echo MBX_SERVER=started
"""
    r = gc(["compute", "tpus", "tpu-vm", "ssh", name, "--zone", zone, "--quiet", "--command", script], project)
    out = r.stdout + r.stderr
    for line in r.stdout.splitlines():
        if line.startswith("    mashbox install:") and "next:" not in line and "not on your PATH" not in line:
            say(line)
    if r.returncode:
        raise Stop("install on the chip failed:\n" + out.strip()[-1500:])
    res = dict(seconds=now() - t0)
    for line in out.splitlines():
        if line.startswith("MBX_SERVER="):
            res["server"] = line.split("=", 1)[1].strip()
        if line.startswith("MBX_ARGS="):
            res["args"] = line.split("=", 1)[1].strip()
        if line.startswith("MBX_DISK"):
            res.update(kv.split("=") for kv in line.split()[1:])
    if res.get("server") == "stuck":
        raise Stop(f"the old server did not stop within 5 min of Ctrl-C (SIGINT). Look: mashbox tpu ssh {name}")
    if res.get("server") == "disk":
        raise Stop(f"the chip's disk has {res.get('have_gb')} GB for this model and it needs {c.get('disk_gb')} GB")
    return res


PROBE = r"""p={p}; z={z}; q={q}
h=$HOME/.mashbox/h; umask 077
curl -fs -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
 | sed -n 's/.*"access_token" *: *"\([^"]*\)".*/Authorization: Bearer \1/p' > "$h"
api=https://tpu.googleapis.com/v2/projects/$p/locations/$z
echo MBX_PROBE_NODE=$(curl -s -o /dev/null -w '%{{http_code}}' -X DELETE -H @"$h" "$api/nodes/mashbox-permcheck-{r}")
[ -z "$q" ] || echo MBX_PROBE_QR=$(curl -s -o /dev/null -w '%{{http_code}}' -X DELETE -H @"$h" "$api/queuedResources/mashbox-permcheck-{r}")
rm -f "$h"; echo "MBX_ARMED=$(grep -c armed /var/log/mashbox-life.log 2>/dev/null || echo 0)"
tail -n 1 /var/log/mashbox-life.log 2>/dev/null | sed 's/^/MBX_LIFE /'
"""


def life_check(project, zone, name, qr):
    """Can the chip delete itself? DELETE on a random name that does not exist: 404 = allowed, 403 = not allowed."""
    script = PROBE.format(p=shlex.quote(project), z=shlex.quote(zone), q=shlex.quote(qr or ""), r=uuid.uuid4().hex[:12])
    for _ in range(4):                                  # the startup script may still be running right after boot
        r = gc(["compute", "tpus", "tpu-vm", "ssh", name, "--zone", zone, "--quiet", "--command", script], project)
        v = dict(line.split("=", 1) for line in r.stdout.splitlines() if line.startswith("MBX_") and "=" in line)
        if v.get("MBX_ARMED", "0").strip() not in ("", "0"):
            break
        time.sleep(min(POLL, 15))
    codes = [v.get("MBX_PROBE_NODE")] + ([v.get("MBX_PROBE_QR")] if qr else [])
    ok = all(c == "404" for c in codes) and v.get("MBX_ARMED", "0").strip() not in ("", "0")
    return ok, codes, v.get("MBX_ARMED", "0").strip()


def sa_fix(project):
    return (f"gcloud projects add-iam-policy-binding {project} --role=roles/tpu.admin "
            f"--member=serviceAccount:$(gcloud projects describe {project} --format='value(projectNumber)')"
            "-compute@developer.gserviceaccount.com")


# --------------------------------------------------------------------------------------------------------------- tunnel
class Port:
    """127.0.0.1:port, bound by this process from the first check to the end of the session, and relayed to the SSH
    forward's Unix socket (self.sock, in a 0700 directory of this user). Another local user can neither bind the port
    (it is never released, not even while a dropped tunnel is reopened) nor reach or replace the socket."""

    def __init__(self, port):
        self.dir = tempfile.mkdtemp(prefix="mbx-")                          # mode 0700
        self.sock = os.path.join(self.dir, "s")
        self.ls = socket.socket()
        self.ls.bind(("127.0.0.1", port))
        self.ls.listen(64)
        self.port = port
        threading.Thread(target=self.accept, daemon=True).start()

    def accept(self):
        while True:
            try:
                c, _ = self.ls.accept()
            except OSError:
                return                                                      # closed
            threading.Thread(target=self.relay, args=(c,), daemon=True).start()

    def relay(self, c):
        u = socket.socket(socket.AF_UNIX)
        try:
            u.connect(self.sock)
        except OSError:                                                     # no forward yet (starting, reopening)
            u.close()
            c.close()
            return

        def pipe(a, b):
            try:
                while True:
                    d = a.recv(65536)
                    if not d:
                        break
                    b.sendall(d)
            except OSError:
                pass
            for x in (b, a):
                try:
                    x.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        t = threading.Thread(target=pipe, args=(u, c), daemon=True)
        t.start()
        pipe(c, u)
        t.join()
        u.close()
        c.close()

    def close(self):
        try:
            self.ls.shutdown(socket.SHUT_RDWR)                              # wakes the accept() thread (Linux)
        except OSError:
            pass
        self.ls.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def free_port(p):
    """A Port on the first free local port from p (up to p + 19); it stays bound by this process."""
    for q in range(p, p + 20):
        try:
            return Port(q)
        except OSError:
            continue
    raise Stop(f"ports {p} to {p + 19} on 127.0.0.1 are all in use here")


TUNNEL_FAIL = ("forwarding failed", "cannot listen", "address already in use", "bind:", "port forwarding failed")


def open_tunnel(project, zone, name, rport, lp):
    cmd = [gcloud(), "compute", "tpus", "tpu-vm", "ssh", name, "--zone", zone, "--project", project, "--quiet", "--",
           "-N", "-L", f"{lp.sock}:127.0.0.1:{rport}", "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=yes",
           "-o", "ServerAliveInterval=30"]
    log = os.path.join(lp.dir, "tunnel.log")                                # ssh says "connect failed" until the server listens
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    p.log = log
    return p


def forward_failed(tun):
    """The SSH forward could not be set up (ExitOnForwardFailure): the lines of its log that say so, else None."""
    try:
        with open(tun.log) as f:
            bad = [x.strip() for x in f.read().splitlines() if any(w in x.lower() for w in TUNNEL_FAIL)]
    except OSError:
        return None
    return bad[-1][:300] if bad else None


def up_http(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def remote_tail(project, zone, name):
    r = gc(["compute", "tpus", "tpu-vm", "ssh", name, "--zone", zone, "--quiet", "--command",
            "pid=$(cat ~/.mashbox/serve.pid 2>/dev/null); kill -0 \"$pid\" 2>/dev/null && echo MBX_ALIVE=1 || echo MBX_ALIVE=0; "
            "tail -n 15 ~/.mashbox/serve.log"], project, timeout=60)
    alive = "MBX_ALIVE=0" not in r.stdout
    return alive, [x for x in r.stdout.splitlines() if not x.startswith("MBX_ALIVE")]


def wait_server(project, zone, name, m, lp):
    """Ready = our SSH process is running and the chip's server answered through its forward (our Port relays to our
    socket only). A forward SSH could not set up stops here with its message, never a silent retry."""
    c = m["cloud"]
    url = f"http://127.0.0.1:{lp.port}{c.get('health', '/')}"
    tun, t0, last = open_tunnel(project, zone, name, c["port"], lp), now(), now()
    say(f"Starting the server on the chip (first start downloads the model; measured times: docs/models/{m['id']}.md).")
    while not (tun.poll() is None and up_http(url)):
        if tun.poll() is not None:
            bad = forward_failed(tun)
            if bad:
                raise Stop(f"the SSH tunnel could not open its forward ({bad}). The chip is still up and billing: "
                           f"mashbox tpu down {name}")
            tun = open_tunnel(project, zone, name, c["port"], lp)
        if now() - last >= 60:
            alive, tail = remote_tail(project, zone, name)
            line = next((x for x in reversed(tail) if x.strip()), "")
            say(f"  {mins(now() - t0)} so far: {line[:150]}")
            if not alive:
                tun.terminate()
                raise Stop("the server on the chip stopped. Last lines of ~/.mashbox/serve.log:\n" + "\n".join(tail)
                           + f"\nThe chip is still up and billing: mashbox tpu down {name}")
            last = now()
        if now() - t0 > READY_MIN * 60:
            tun.terminate()
            raise Stop(f"the server was not ready after {READY_MIN} min. Log: mashbox tpu ssh {name}, then "
                       f"tail ~/.mashbox/serve.log. The chip is still up and billing: mashbox tpu down {name}")
        time.sleep(min(POLL, 5))
    say(f"Server ready ({mins(now() - t0)}).")
    return tun


def node_gone(project, zone, name):
    """True only when gcloud says the node does not exist (or is being deleted); an error of any other kind is not 'gone'."""
    r = gc(["compute", "tpus", "tpu-vm", "describe", name, "--zone", zone, "--format=value(state)"], project)
    if r.returncode:
        return "not_found" in r.stderr.lower() or "was not found" in r.stderr.lower()
    return r.stdout.strip() in ("DELETING", "PREEMPTED", "TERMINATED")


def hold(tun, project, zone, name, m, lp, deadline):
    """Keep the tunnel open until Ctrl-C; reopen it if it drops; stop when the chip is gone (down from another
    terminal, its max lifetime); at the deadline take the chip down from here too."""
    reopened = 0.0
    try:
        while True:
            if deadline and now() > deadline + 600:
                say("Max lifetime reached and the chip is still here: taking it down from this machine.")
                tun.terminate()
                try:
                    return take_down(project, [(zone, name)])
                except Stop as e:
                    raise Stop(f"{e}\n{name} ({zone}) may still be running and billing. Take it down: "
                               f"mashbox tpu down {name}") from None
            if tun.poll() is not None:                 # the tunnel dropped: chip gone, or a network blip
                if node_gone(project, zone, name):
                    say(f"\n{name} is gone (taken down, or deleted by its max lifetime). Nothing of it bills now. "
                        "Tunnel closed.")
                    save_state(name)
                    return 0
                if now() - reopened < 10:              # at most one reopen every 10 s
                    time.sleep(10 - (now() - reopened))
                tun, reopened = open_tunnel(project, zone, name, m["cloud"]["port"], lp), now()
            time.sleep(POLL)
    except KeyboardInterrupt:
        tun.terminate()
        say(f"\nTunnel closed. The chip is still running and billing. Reopen: mashbox tpu tunnel {name}   "
            f"Take it down: mashbox tpu down {name}")
        return 0


def info(m, name, zone, acc, spot, lport, deadline, life, agents=False):
    c, p = m["cloud"], price(acc)
    base = f"http://127.0.0.1:{lport}"
    say("")
    say(f"{m['name']} is running on {name} (TPU {acc}, {zone}{', spot' if spot else ''}).")
    say(f"  Local URL     {base}{c.get('url_path', '/')}   (SSH tunnel to the chip's 127.0.0.1; nothing is open to the internet)")
    say(f"  Price         {money(p)} per hour while the chip exists{' (spot: lower, and Google can stop it)' if spot else ''}"
        f"   {PRICE_NOTE}")
    if deadline:
        when = datetime.fromtimestamp(deadline, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        say(f"  Max lifetime  the chip deletes itself at about {when} (it counts from its boot, a few minutes later): {life}")
    else:
        say("  Max lifetime  off (--max-hours 0): the chip runs until you take it down")
    if m["kind"].startswith("llm"):
        say(f"  Chat          curl -s {base}/v1/chat/completions -H 'Content-Type: application/json' \\")
        say(f"                  -d '{{\"model\": \"{m['id']}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}], \"max_tokens\": 128}}'")
    else:
        say(f"  Open          {base}{c.get('url_path', '/')} in your browser")
    if c.get("agents"):
        g = c["agents"]
        say(f"  Coding agents base URL {base} (Claude Code), {base}/v1 (Codex, Pi, any OpenAI-compatible agent); "
            f"settings: {g['guide']}" + ("" if lport == c["port"] else f" (use port {lport} where it says {c['port']})"))
        if agents is False:
            say(f"                this server has the default context; for agents start it with: mashbox tpu up {m['id']} --agents --restart")
    say(f"  Shell         mashbox tpu ssh {name}   (server log: ~/.mashbox/serve.log)")
    say(f"  Take it down  mashbox tpu down {name}")
    say("This terminal holds the tunnel. Ctrl-C closes the tunnel only; the chip keeps running and billing.")


# ------------------------------------------------------------------------------------------------------------- commands
def cmd_up(a):
    m = model(a.model)
    c, acc = m["cloud"], m["cloud"]["accelerator"]
    name = a.name or f"mashbox-{m['id']}"[:60]
    if a.max_hours < 0:
        raise Stop("--max-hours must be 0 (off) or more")
    project = preflight(a.project)
    p = price(acc)
    say(f"{m['name']} needs a TPU {acc}: {money(p)} per hour while it exists ({PRICE_NOTE}).")
    say(f"Max lifetime: {'%g h after the chip starts, then it deletes itself (--max-hours 0 turns this off)' % a.max_hours if a.max_hours else 'off (--max-hours 0)'}.")
    zones = list(dict.fromkeys(([a.zone] if a.zone else []) + c["zones"] + all_zones()))
    node, qr = find(project, name, zones)
    qrname = None
    if node:
        zone, x = node
        age = now() - (created(x) or now())
        say(f"{name} already exists in {zone} ({x.get('state')}, up {mins(age)}, about {money((p or 0) * age / 3600)} so far, ESTIMATE). Using it.")
        if x.get("state") not in ("READY", "CREATING", "STARTING", "RESTARTING", "REPAIRING"):
            raise Stop(f"{name} is {x.get('state')}. Take it down (mashbox tpu down {name}) and run up again")
        if x.get("state") != "READY" and not wait_ready_node(project, zone, name):
            raise Stop(f"{name} did not reach READY. Take it down: mashbox tpu down {name}")
        qrname = short(x.get("queuedResource")) or None
        acc = x.get("acceleratorType") or acc
    elif qr:
        zone = qr[0]
        raise Stop(f"a queued request for {name} is waiting in {zone} from an earlier run. Delete it with "
                   f"mashbox tpu down {name}, then run up again")
    else:
        if not a.yes:
            if not sys.stdin.isatty():
                raise Stop("this creates a billed chip; add --yes to confirm when not running in a terminal")
            if input(f"Create {name} now? [y/N] ").strip().lower() not in ("y", "yes"):
                say("Nothing created.")
                return 0
        say(f"Getting a chip ({'zone ' + a.zone if a.zone else 'zones in order: ' + ', '.join(c['zones'])}):")
        try:
            zone, qrname = get_chip(m, a, project, name)
        except KeyboardInterrupt:
            say(f"\nStopped. A create already sent may still finish: check with mashbox tpu status, "
                f"take it down with mashbox tpu down {name}. Its max lifetime still applies.")
            return 130
        save_state(name, project=project, zone=zone, model=m["id"], qr=qrname, max_hours=a.max_hours, created=now())
    try:
        say(f"Installing mashbox on {name} with install.sh ...")
        extra = (c["agents"]["run_args"] if a.agents and c.get("agents") else []) + list(a.run_args)
        if a.agents and not c.get("agents"):
            raise Stop(f"{m['id']} has no coding-agent settings")
        res = remote_setup(m, project, zone, name, extra, a.restart)
        if res.get("server") == "running" and res.get("args"):
            say(f"  the server was already running ({res['args']}); --restart starts it again with this command's options")
        say(f"  done in {mins(res['seconds'])}; disk for the model: {res.get('have_gb')} GB (needs {c.get('disk_gb')}); "
            f"server {res.get('server')}")
        n, _ = find(project, name, [zone], unchanged=False)
        mh = int(((n[1].get("metadata") or {}).get("mashbox-max-seconds") or 0)) / 3600 if n else a.max_hours
        deadline = (created(n[1]) or now()) + mh * 3600 if n and mh else None
        life = "off"
        if deadline:
            ok, codes, armed = life_check(project, zone, name, qrname)
            if ok:
                life = "checked: the timer is armed on the chip and its service account may delete it"
            else:
                life = (f"NOT ACTIVE (timer armed: {armed != '0'}; delete probe HTTP {', '.join(map(str, codes))}, "
                        f"403 = the chip's service account may not delete TPUs). Only this terminal will delete it at "
                        f"the deadline, while it stays open. To let the chip delete itself in this project: {sa_fix(project)}")
            save_state(name, life=ok)
            say(f"  max lifetime: {life}")
        lp = free_port(c["port"])
        tun = wait_server(project, zone, name, m, lp)
    except KeyboardInterrupt:
        say(f"\nStopped. The chip is still running and billing. Take it down: mashbox tpu down {name}")
        return 130
    except Stop as e:
        raise Stop(f"{e}\n{name} ({zone}) is still running and billing. Take it down: mashbox tpu down {name}") from None
    info(m, name, zone, acc, a.spot, lp.port, deadline, life, agents=a.agents)
    return hold(tun, project, zone, name, m, lp, deadline)


def pick(project, name, zones):
    nodes, qrs = scan(project, zones)
    names = list(dict.fromkeys([short(x.get("name")) for _, x in nodes] + [qr_node(x)[0] for _, x in qrs]))
    if name:
        return [n for n in names if n == name], nodes, qrs
    return names, nodes, qrs


def take_down(project, targets, nodes=None, qrs=None):
    """targets: [(zone, name)]. Deletes the queued request (with its node) or the node; then checks the list."""
    if nodes is None:
        nodes, qrs = scan(project, list(dict.fromkeys(z for z, _ in targets)))
    rc = 0
    for zone, name in targets:
        t0 = now()
        q = [(z, x) for z, x in qrs if qr_node(x)[0] == name]
        n = [(z, x) for z, x in nodes if short(x.get("name")) == name]
        for z, x in n:
            age = now() - (created(x) or now())
            say(f"{name} ({x.get('acceleratorType')}, {z}): up {mins(age)}, about "
                f"{money((price(x.get('acceleratorType', '')) or 0) * age / 3600)} so far (ESTIMATE, list price). Deleting ...")
        for z, x in q:
            if not delete_qr(project, z, short(x.get("name"))):
                rc = 1
        for z, x in n:
            if short(x.get("queuedResource")) and q:
                continue                                         # deleted with its queued request
            r = gc(["compute", "tpus", "tpu-vm", "delete", name, "--zone", z, "--quiet"], project)
            if r.returncode and "not found" not in r.stderr.lower():
                say(f"  NOT deleted: {gerr(r.stderr)}")
                rc = 1
        try:
            left, lq = scan(project, list(dict.fromkeys([z for z, _ in n + q])), unchanged=False)
        except Stop as e:
            say(f"{name}: delete sent, but the check failed: {e}\nCheck with: mashbox tpu status")
            rc = 1
            continue
        still = [short(x.get("name")) for _, x in left if short(x.get("name")) == name] + \
                [qr_node(x)[0] for _, x in lq if qr_node(x)[0] == name]
        if still:
            say(f"{name}: STILL LISTED after delete; check: gcloud compute tpus tpu-vm list --zone {zone}")
            rc = 1
        else:
            say(f"{name}: gone ({mins(now() - t0)}). Nothing of it bills now.")
            save_state(name)
    return rc


def cmd_down(a):
    project = preflight(a.project)
    zones = [a.zone] if a.zone else all_zones()
    names, nodes, qrs = pick(project, a.name, zones)
    prune(nodes, qrs, zones)
    if not names:
        say(f"Nothing to take down: no mashbox chip or queued request{' named ' + a.name if a.name else ''} in project "
            f"{project} (zones checked: {', '.join(zones)}).")
        return 0
    if len(names) > 1 and not a.all:
        raise Stop("several mashbox chips: " + ", ".join(names) + ". Name one (mashbox tpu down <name>) or use --all")
    zone_of = {short(x.get("name")): z for z, x in nodes}
    zone_of.update({qr_node(x)[0]: z for z, x in qrs if qr_node(x)[0] not in zone_of})
    return take_down(project, [(zone_of[n], n) for n in names], nodes, qrs)


def cmd_status(a):
    project = preflight(a.project)
    zones = [a.zone] if a.zone else all_zones()
    nodes, qrs = scan(project, zones)
    prune(nodes, qrs, zones)
    if not nodes and not qrs:
        say(f"No mashbox chips or queued requests in project {project} (zones checked: {', '.join(zones)}).")
        return 0
    total, rate = 0.0, 0.0
    say(f"{'NAME':<28} {'ZONE':<18} {'TYPE':<12} {'STATE':<21} {'AGE':>8} {'$/HOUR':>8} {'SO FAR':>9}  DELETES ITSELF")
    for z, x in nodes:
        acc, age = x.get("acceleratorType", "?"), now() - (created(x) or now())
        p = price(acc) if "-" in acc else None
        spot = (x.get("schedulingConfig") or {}).get("spot") or (x.get("schedulingConfig") or {}).get("preemptible")
        ms = int((x.get("metadata") or {}).get("mashbox-max-seconds") or 0)
        life = datetime.fromtimestamp((created(x) or now()) + ms, timezone.utc).strftime("~%H:%M UTC %d %b") if ms else "no (off)"
        if ms and load_state().get(short(x.get("name")), {}).get("life") is False:
            life += " (NOT ACTIVE)"
        total += (p or 0) * age / 3600
        rate += p or 0
        say(f"{short(x.get('name')):<28} {z:<18} {acc + (' spot' if spot else ''):<12} {x.get('state', '?'):<21} "
            f"{mins(age):>8} {money(p):>8} {money((p or 0) * age / 3600):>9}  {life}")
    for z, x in qrs:
        st = (x.get("state") or {}).get("state", "?")
        if st != "ACTIVE":
            say(f"{qr_node(x)[0] + ' (queued)':<28} {z:<18} {(qr_node(x)[1].get('acceleratorType') or '?'):<12} "
                f"{st:<21} {mins(now() - (created(x) or now())):>8} {'$0.00':>8} {'$0.00':>9}  (no chip yet)")
    say(f"Running now: {money(rate)} per hour, about {money(total)} so far. {PRICE_NOTE}.")
    say("Take one down: mashbox tpu down <name>")
    return 0


def target(a):
    project = preflight(a.project)
    names, nodes, _ = pick(project, a.name, list(dict.fromkeys(([a.zone] if a.zone else []) + all_zones())))
    ready = [(z, x) for z, x in nodes if short(x.get("name")) in names]
    if not ready:
        raise Stop("no mashbox chip" + (f" named {a.name}" if a.name else "") + " is running. Start one: mashbox tpu up <model>")
    if len(ready) > 1:
        raise Stop("several mashbox chips: " + ", ".join(short(x.get("name")) for _, x in ready) + ". Name one")
    return project, ready[0][0], ready[0][1]


def cmd_ssh(a):
    project, zone, x = target(a)
    return subprocess.call([gcloud(), "compute", "tpus", "tpu-vm", "ssh", short(x.get("name")), "--zone", zone, "--project", project])


def cmd_tunnel(a):
    project, zone, x = target(a)
    name = short(x.get("name"))
    m = model((x.get("labels") or {}).get("mashbox-model") or load_state().get(name, {}).get("model") or "")
    lp = free_port(m["cloud"]["port"])
    ms = int((x.get("metadata") or {}).get("mashbox-max-seconds") or 0)
    deadline = (created(x) or now()) + ms if ms else None
    tun = wait_server(project, zone, name, m, lp)
    life = {True: "checked at up", False: "NOT ACTIVE (see mashbox tpu status)"}.get(load_state().get(name, {}).get("life"), "set at up")
    info(m, name, zone, x.get("acceleratorType", m["cloud"]["accelerator"]), False, lp.port, deadline, life, agents=None)
    return hold(tun, project, zone, name, m, lp, deadline)


def main(argv):
    ap = argparse.ArgumentParser(prog="mashbox tpu", description="Cloud TPU chips from this machine, with your own gcloud.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="create a chip for a model, install, start the server, open a tunnel",
                       usage="mashbox tpu up MODEL [options] [-- options for mashbox run on the chip]")
    u.add_argument("model")
    u.add_argument("--zone", type=checked(ZONE, "zone"), help="only this zone (default: the model's zones in order)")
    u.add_argument("--spot", action="store_true", help="a spot chip: cheaper, Google can stop it at any time")
    u.add_argument("--name", type=checked(NAME, "chip name"), help="chip name (default mashbox-<model>)")
    u.add_argument("--max-hours", type=float, default=8.0, help="the chip deletes itself this long after it starts (default 8; 0 = off)")
    u.add_argument("--queue-minutes", type=float, default=60.0, help="how long to wait for a queued chip (default 60; 0 = no queued requests)")
    u.add_argument("--yes", action="store_true", help="create without asking")
    u.add_argument("--agents", action="store_true", help="start the server with the coding-agent settings of the model's row")
    u.add_argument("--restart", action="store_true", help="stop a running server (Ctrl-C, SIGINT) and start it again")
    for n in ("down", "status", "ssh", "tunnel"):
        p = sub.add_parser(n)
        if n != "status":
            p.add_argument("name", nargs="?", type=checked(NAME, "chip name"))
        p.add_argument("--zone", type=checked(ZONE, "zone"), help="this zone only" if n in ("down", "status") else None)
        if n == "down":
            p.add_argument("--all", action="store_true", help="every mashbox chip and queued request in the project")
    for p in sub.choices.values():
        p.add_argument("--project", type=checked(PROJECT, "project id"), help="Google Cloud project (default: gcloud's)")
    argv = list(argv)
    cut = argv.index("--") if "--" in argv else len(argv)     # after --: options for `mashbox run` on the chip
    a = ap.parse_args(argv[:cut])
    a.run_args = argv[cut + 1:]
    try:
        return dict(up=cmd_up, down=cmd_down, status=cmd_status, ssh=cmd_ssh, tunnel=cmd_tunnel)[a.cmd](a)
    except Stop as e:
        print(f"mashbox tpu: {e}", file=sys.stderr, flush=True)
        return 2
