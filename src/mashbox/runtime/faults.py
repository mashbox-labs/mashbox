"""Fault injection for failure tests. Off unless the environment variable MASHBOX_FAULT names faults; a normal run never
reads anything else here.

MASHBOX_FAULT="<where>:<n>:<kind>[:<lives>],..."
  where   load (before the engine is built), warm (the compile warm-up), admit, step (one engine call each)
  n       the n-th call of that kind in this worker process, from 1
  kind    raise (a device error), oom (an out-of-memory error, as XLA raises it), kill (SIGKILL to itself, as the kernel's
          out-of-memory killer), hang (the call never returns), exit (the process exits with status 3)
  lives   in how many worker processes the fault fires (default 1), counted in MASHBOX_FAULT_STATE (a directory), so a
          restarted worker comes back healthy unless the test asks for more
MASHBOX_FAULTS="<kind>@<n>[,...]" is the short form the certification harness uses: kind device_error, oom or hang,
fired at the n-th device call (admit or step, counted together) after the worker is ready, in one worker life.
Each fault that fires logs one line {"event": "fault_injected", "kind": ..., "call": n, "where": ...} on stderr.
MASHBOX_FAULT="force:<name>" (an entry of the same variable) makes the named fast path fail so its fallback runs, for
the fallback tests: FORCE lists every name (the fallback table's force names, ARCHITECTURE-v1 section 3) and the fast
path asks forced(name). A name whose fast path does not call forced() changes nothing.
"""
import json
import os
import signal
import sys
import threading
import time

ENV, SHORT, STATE = "MASHBOX_FAULT", "MASHBOX_FAULTS", "MASHBOX_FAULT_STATE"
WHERE = ("load", "warm", "admit", "step", "device")         # device = admit and step counted together
SHORT_KINDS = {"device_error": "raise", "oom": "oom", "hang": "hang"}
KINDS = ("raise", "oom", "kill", "hang", "exit")
FORCE = ("speculation", "speculation-row", "moe-routed", "fit", "quant-kernel", "kernels-package", "model-bundle",
         "compile-cache", "pin", "safetensors-bf16", "dense16", "native-planes")


class InjectedFault(RuntimeError):
    """The error a 'raise' or 'oom' fault raises inside an engine call (a stand-in for the device runtime's error)."""


def parse_short(spec):
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, n = part.partition("@")
        if k not in SHORT_KINDS or not n.isdigit() or int(n) < 1:
            raise ValueError(f"{SHORT}: bad fault {part!r} (<kind>@<n>, kind in {tuple(SHORT_KINDS)})")
        out.append(dict(where="device", n=int(n), kind=SHORT_KINDS[k], name=k, lives=1, key="short_" + part))
    return out


def _force_names(spec):
    """the force:<name> entries of a MASHBOX_FAULT value; an unknown name is an error"""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if part.startswith("force:"):
            name = part[len("force:"):]
            if name not in FORCE:
                raise ValueError(f"{ENV}: bad force name {name!r} (force:<name>, name in {FORCE})")
            out.append(name)
    return out


def forced(name):
    """True when MASHBOX_FAULT holds force:<name>: the fast path named in FORCE takes its fallback (tests only)"""
    if name not in FORCE:
        raise ValueError(f"forced: unknown force name {name!r} (one of {FORCE})")
    return name in _force_names(os.environ.get(ENV))


def parse(spec):
    _force_names(spec)                                       # force:<name> entries are checked here, read by forced()
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or part.startswith("force:"):
            continue
        f = part.split(":")
        if len(f) not in (3, 4) or f[0] not in WHERE or f[2] not in KINDS or not f[1].isdigit() or int(f[1]) < 1:
            raise ValueError(f"{ENV}: bad fault {part!r} (where:n:kind[:lives], where in {WHERE}, kind in {KINDS})")
        out.append(dict(where=f[0], n=int(f[1]), kind=f[2], lives=int(f[3]) if len(f) == 4 else 1, key=part))
    return out


class Faults:
    def __init__(self, spec=None, state=None):
        self.faults = parse(os.environ.get(ENV) if spec is None else spec)
        if spec is None:
            self.faults += parse_short(os.environ.get(SHORT))
        self.state = os.environ.get(STATE) if state is None else state
        self.calls = dict.fromkeys(WHERE, 0)
        self.lock = threading.Lock()

    def _spend(self, f):
        """True when this fault still has a worker life to fire in (and uses one)."""
        if not self.state:
            return True
        os.makedirs(self.state, exist_ok=True)
        tag = "".join(c if c.isalnum() else "_" for c in f["key"])
        used = sum(1 for n in os.listdir(self.state) if n.startswith(tag + "."))
        if used >= f["lives"]:
            return False
        with open(os.path.join(self.state, f"{tag}.{os.getpid()}"), "w") as fh:
            fh.write(f["key"])
        return True

    def hit(self, where):
        """Called before each engine call of kind `where`; fires the fault named for this call, if any."""
        if not self.faults:
            return
        with self.lock:
            self.calls[where] += 1
            n = self.calls[where]
            if where in ("admit", "step"):
                self.calls["device"] += 1
            due = [f for f in self.faults if (f["where"] == where and f["n"] == n
                                              or f["where"] == "device" and where in ("admit", "step")
                                              and f["n"] == self.calls["device"]) and self._spend(f)]
        for f in due:
            k = f["kind"]
            print(json.dumps(dict(event="fault_injected", kind=f.get("name", k), call=f["n"], where=where)),
                  file=sys.stderr, flush=True)
            if k == "raise":
                raise InjectedFault(f"injected device fault at {where} call {n}")
            if k == "oom":
                raise InjectedFault(f"RESOURCE_EXHAUSTED: injected out of memory at {where} call {n}")
            if k == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(60)
            if k == "exit":
                os._exit(3)
            if k == "hang":
                while True:
                    time.sleep(3600)
