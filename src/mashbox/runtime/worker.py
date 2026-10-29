"""Worker process: holds one engine and runs the server's batching loop for an HTTP front (runtime/front.py), over the
socket described in runtime/protocol.py.

The worker runs the same loop as the one-process server (server.Server.loop_live, or loop_batches for engines without
admit/step); only the ends change: requests come in as SUBMIT frames and tokens leave as TOKENS frames. The front keeps
the waiting queue and its limits and sends at most one request per free row, so everything the worker holds is live.

Any failed device call (an error from the runtime, out of memory) and any call that runs past the watchdog ends this
process after one fatal ERROR frame: the device state may be half written, so the front starts a fresh worker instead of
reusing it. A front that goes away ends the worker too (its rows end with the process).

The front's /health reads this process's engine: each HEALTH frame (every second, and at once after any compile after
ready) carries `stats`, the fields server.Server.health computes here (live rows, cache bytes, cache namespace,
deployment id, component set, compiles after ready). The ready frame also carries `info`, the served weights' fields for
/v1/models (engine.serve_info, set by from_cli).

Usage (the front starts it): python -m mashbox.runtime.worker --socket PATH --factory module:function [--kwargs JSON]
[--warm]. The factory returns the engine; --warm runs engine.warm() (the compile warm-up) before the worker says ready.
"""
import argparse
import collections
import importlib
import json
import os
import queue
import statistics
import sys
import threading
import time
import uuid

from . import programs
from . import protocol as P
from .agents import BadRequest
from .faults import Faults
from .server import Server

WATCHED = ("admit", "step", "generate", "warm")


class Watched:
    """The engine, with each call timed for the watchdog and passed through the fault hooks (faults.py)."""

    def __init__(self, eng, faults):
        self._e, self._f = eng, faults
        self.busy_since, self.busy_what = None, None
        self.times = collections.defaultdict(lambda: collections.deque(maxlen=64))

    def __getattr__(self, name):
        fn = getattr(self._e, name)            # AttributeError when the engine has no such method (hasattr stays true)
        if name not in WATCHED:
            return fn
        where = "step" if name == "generate" else name

        def call(*a, **k):
            t0 = time.monotonic()
            self.busy_since, self.busy_what = t0, where
            try:
                self._f.hit(where)
                return fn(*a, **k)
            finally:
                self.times[where].append(time.monotonic() - t0)
                self.busy_since = None
        return call

    def deadline(self, min_s):
        ts = self.times.get(self.busy_what)
        return max(min_s, P.WATCHDOG_X * (statistics.median(ts) if ts else 0.0))


class _Ids:
    """The loop needs only the end-of-text ids from the tokenizer; the front sends them with each request."""
    stop_ids = ()


class Sink:
    """Stands in for a job's stream queue: turns the loop's events (ids so far, done, finish) into frames."""

    def __init__(self, w, j):
        self.w, self.j, self.n, self.seq, self.over = w, j, 0, 0, False

    def _tokens(self, ids):
        new = [int(t) for t in ids[self.n:]]
        if new:
            self.w.send(dict(t="TOKENS", req=self.j["req"], seq=self.seq, ids=new))
            self.seq += 1
            self.n += len(new)

    def put(self, ev):
        if self.over:
            return
        if ev is not None:
            return self._tokens(ev[0])
        self.over = True
        j = self.j
        self.w.jobs.pop(j["req"], None)
        if "error" in j:
            return self.w.request_error(j, j["error"])
        r = j["result"]
        self._tokens(r["ids"])
        self.w.send(dict(t="FINISH", req=j["req"], reason=r["finish"], calls=r.get("timing", {}).get("decode_calls"),
                         sampled=bool(r.get("timing", {}).get("sampled")), batch_rows=r.get("batch_rows"),
                         usage=dict(prompt=len(j["ids"]), cached=int(r.get("cached") or 0),
                                    restored=int(r.get("restored") or 0), output=len(r["ids"]))))


class Worker(Server):
    """server.Server's batching loop with frames at both ends. Server.__init__ is not called: it starts the loop at once
    and needs a tokenizer, and the worker starts its loop only once the engine is ready."""

    def __init__(self, conn, log):
        self.conn, self.log = conn, log
        self.engine, self.chat, self.model_id, self.info = None, _Ids(), None, {}
        self.q, self.waiting, self.wlock = queue.Queue(), 0, threading.Lock()
        self.wait_by = collections.defaultdict(list)
        self.ready, self.jobs, self.dying = False, {}, False
        self.admits = 0                  # admission counter ("seq" of the MASHBOX_TRACE admit line), per worker life
        self.announced = False          # the ready frame with the limits went out: heartbeats may say ready from here on

    def send(self, msg):
        try:
            self.conn.send(msg)
        except OSError:                 # the front is gone: nothing to answer
            os._exit(0)

    def _unwait(self, j):
        j["unwaited"] = True            # the front keeps the waiting limits

    def retry_after(self):
        return 1

    def fatal(self, e, code=None, table=None):
        """A device call failed or hung: say so once and end the process (its device state is not reused). table: set
        when the failed call was a program loaded from the saved executable table (Worker.table_failed); the frame
        carries it so the front starts the next worker without the table."""
        if self.dying:
            return
        self.dying = True
        msg = f"{type(e).__name__}: {e}"[:300] if isinstance(e, BaseException) else str(e)[:300]
        if "RESOURCE_EXHAUSTED" in msg or isinstance(e, MemoryError):
            code = "oom"
        code = code or "device_fault"
        frame = dict(t="ERROR", req=None, code=code, fatal=True, msg=msg)
        if table is not None:
            frame["table"] = table
        try:
            self.log(json.dumps(dict(event="worker_fault", code=code, error=msg)))
            self.conn.send(frame)
        finally:
            os._exit(70 if code != "watchdog" else 71)

    def table_failed(self, jobs, e):
        """A program loaded from the saved executable table failed in a request or in the warm-up before ready
        (programs.TableFailed; the engine has quarantined the table's folder and dropped the table). This process never warms again (a
        compile here would be a compile after ready): one fatal ERROR frame with table={"closed": whether the folder was
        renamed or refused in place}, then exit. The front owns the rest: 503 while no worker is ready, a new worker
        without the table (MASHBOX_AOT=0) that compiles before it says ready, and one more try for each request of the
        call whose client has received nothing (runtime/front.py)."""
        self.log(json.dumps(dict(event="aot_table", phase="failed", **e.info)))
        self.fatal(e, table=dict(closed=bool(e.info.get("closed", True))))

    def fail(self, jobs, e):
        if isinstance(e, BadRequest):
            for j in jobs:
                j["error"] = e
                j["stream"].put(None)
            return
        self.fatal(e)

    def request_error(self, j, e):
        if isinstance(e, BadRequest):
            self.send(dict(t="ERROR", req=j["req"], code="bad_params", fatal=False, msg=str(e)[:300]))
        elif isinstance(e, ConnectionResetError):        # the whole-batch loop: cancelled while it waited
            self.send(dict(t="FINISH", req=j["req"], reason="cancel", usage=dict(prompt=len(j["ids"]), cached=0,
                                                                                   restored=0, output=0)))
        else:
            self.fatal(e)

    def submit_frame(self, m):
        req, p = m["req"], m.get("params") or {}
        try:
            ids = P.ids_list(m["ids"])
            j = dict(req=req, ids=ids, max_new=int(p["max_tokens"]), samp=p.get("samp"), mark=p.get("mark"),
                     scope=p.get("scope"), variant=p.get("variant"), rid=p.get("rid"), ev=threading.Event())
        except (KeyError, TypeError, ValueError, P.FrameError) as e:
            return self.send(dict(t="REJECTED", req=req, code="bad_params", msg=f"bad request frame: {e}"[:300]))
        if p.get("stop_ids") is not None:
            self.chat.stop_ids = tuple(int(x) for x in p["stop_ids"])
        if not self.ready:
            return self.send(dict(t="REJECTED", req=req, code="loading", msg="the model is loading"))
        why = self.engine.check(dict(ids=ids, max_new=j["max_new"], samp=j["samp"])) if hasattr(self.engine, "check") else None
        if why is not None:
            return self.send(dict(t="REJECTED", req=req, code="bad_params", msg=str(why)[:300]))
        j["stream"] = Sink(self, j)
        self.jobs[req] = j
        self.send(dict(t="ACCEPTED", req=req))
        self.q.put(j)

    def read_loop(self):
        while True:
            try:
                m = self.conn.recv()
            except (P.FrameError, OSError) as e:
                try:
                    self.conn.send(dict(t="ERROR", req=None, code=getattr(e, "code", "bad_frame"), fatal=True, msg=str(e)))
                except Exception:
                    pass
                os._exit(0)
            if m is None:                       # the front went away: the rows end with this process
                os._exit(0)
            t = m["t"]
            if t == "SUBMIT":
                self.submit_frame(m)
            elif t in ("CANCEL", "FINISH_AT"):
                j = self.jobs.get(m.get("req"))
                if j is not None:
                    j["cancel"] = True
            elif t == "HEALTH":
                self.send(self.health())

    def health(self):
        """The HEALTH frame. Once ready it carries `stats` (server.Server.health on this process's engine; the front
        keeps the waiting queue, so `waiting` is left out). It never raises: the heartbeat thread sends it, and a worker
        that sends nothing is killed by the front."""
        m = dict(t="HEALTH", state="ready" if self.ready and self.announced else "loading", live=len(self.jobs))
        if self.ready:
            try:
                st = Server.health(self)
                st.pop("waiting", None)
                m["stats"] = st
            except Exception as e:
                m["stats_error"] = f"{type(e).__name__}: {e}"[:200]
        return m

    def compiled_after_ready(self, name, secs):
        """A compile after ready (runtime/programs.py): logged as the one-process server logs it, and sent to the front at
        once, so /health counts it even when this process is killed a moment later."""
        self.log(json.dumps(dict(event="compile_after_ready", program=name, seconds=round(secs, 3))))
        self.send(self.health())

    def beat(self, min_s, warm_s):
        """Heartbeat for the front, and the watchdog: a call running past its deadline ends the process."""
        while True:
            time.sleep(P.HEARTBEAT_S)
            e = self.engine
            if isinstance(e, Watched) and e.busy_since is not None:
                late = time.monotonic() - e.busy_since
                limit = warm_s if e.busy_what == "warm" else e.deadline(min_s)
                if late > limit:
                    self.fatal(f"{e.busy_what} call running for {late:.1f} s (watchdog {limit:.1f} s)", code="watchdog")
            self.send(self.health())


def from_cli(argv):
    """Worker factory for `mashbox run`: the engine built exactly as the one-process server builds it (cli.load, then
    cli.warm unless --no-warm), from the same command-line arguments, e.g. argv=["run", "qwen3-8-27b"]."""
    from .. import cli
    a = cli.build_parser().parse_args(list(argv))
    m, eng, chat, info, t0 = cli.load(a)
    eng.serve_info = dict(weights=info["file"], gb=round(info["device_bytes"] / 1e9, 2), bpw=info["bpw"],  # /v1/models
                          prefix_cache=eng.prefix_cache, slots=eng.S, cache_ram_mib=a.cache_ram)
    if not getattr(a, "no_warm", False):
        cli.warm(eng, chat)
    return eng


def limits(eng):
    V = getattr(eng, "V", None)
    if V is None and getattr(eng, "c", None) is not None:
        V = getattr(eng.c, "V", None)
    return dict(R=int(eng.S), vocab=None if V is None else int(V), ctx=int(eng.CTX), max_frame=P.MAX_FRAME,
                out_buffer_tokens=P.OUT_BUFFER_TOKENS, dk=int(getattr(eng, "DK", 0) or 0),
                prefix_cache=bool(getattr(eng, "prefix_cache", False)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mashbox.runtime.worker")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--factory", required=True, help="module:function that returns the engine")
    ap.add_argument("--kwargs", default="{}", help="JSON object passed to the factory")
    ap.add_argument("--warm", action="store_true", help="run engine.warm() before saying ready")
    a = ap.parse_args(argv)

    def log(m):
        print(m, file=sys.stderr, flush=True)
    faults = Faults()
    lst = P.listen(a.socket)
    lst.settimeout(120)
    sock, _ = lst.accept()
    lst.close()
    os.unlink(a.socket)
    if P.peer_uid(sock) != os.getuid():      # only a front running as this user may drive the engine
        sock.close()
        return 2
    sock.settimeout(None)
    conn = P.Conn(sock)
    try:
        hello = conn.recv()
    except P.FrameError as e:
        conn.send(dict(t="ERROR", req=None, code=e.code, fatal=True, msg=str(e)))
        return 2
    if not hello or hello.get("t") != "HELLO" or P.PROTO not in (hello.get("proto") or []):
        conn.send(dict(t="ERROR", req=None, code="proto_mismatch", fatal=True, msg=f"this worker speaks protocol {P.PROTO}"))
        return 2
    w = Worker(conn, log)
    conn.send(dict(t="HELLO_ACK", proto=P.PROTO, worker_id=uuid.uuid4().hex, models=[], state="loading", limits=None))
    threading.Thread(target=w.read_loop, daemon=True).start()
    min_s = float(os.environ.get("MASHBOX_WATCHDOG_MIN_S") or P.WATCHDOG_MIN_S)
    warm_s = float(os.environ.get("MASHBOX_WARM_TIMEOUT_S") or 7200)
    w.engine = Watched(None, faults)
    threading.Thread(target=w.beat, args=(min_s, warm_s), daemon=True).start()
    try:
        w.engine.busy_since, w.engine.busy_what = time.monotonic(), "warm"
        faults.hit("load")
        programs.install()                       # count every compile of this process (the warm-up's too)
        mod, fn = a.factory.split(":")
        eng = getattr(importlib.import_module(mod), fn)(**json.loads(a.kwargs))
        w.engine = Watched(eng, faults)
        if a.warm and hasattr(eng, "warm"):
            w.engine.warm()
        programs.mark_ready(w.compiled_after_ready)      # from here on, a compile is a compile after ready
    except programs.TableFailed as e:
        w.table_failed([], e)        # a loaded program failed in the warm-up: the table's fatal frame, then exit
    except Exception as e:
        w.fatal(e, code="load_failed")       # oom stays oom; anything else before ready is the load's own error
    w.ready = True
    w.send(dict(w.health(), state="ready", limits=limits(eng), info=getattr(eng, "serve_info", None)))
    w.announced = True                   # a heartbeat says ready only after this frame (the front takes its limits)
    log(json.dumps(dict(event="worker_ready", pid=os.getpid(), limits=limits(eng))))
    w.loop()


if __name__ == "__main__":
    sys.exit(main())
