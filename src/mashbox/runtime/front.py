"""HTTP front with the engine in a worker process it supervises (runtime/worker.py, protocol in runtime/protocol.py).

The front is the one-process server (server.Server and server.serve: the same routes, tokenizer, stop strings, tool-call
parsing, waiting limits and error shapes) with its batching loop replaced by a link to a worker. It keeps the waiting
queue and sends a request only when the worker has a free row, so a worker failure touches only the live rows.

When the worker fails (an error from a device call, out of memory, a call past the watchdog, or the process killed):
- a request whose client has received no bytes of its answer is sent once more, from the start, to the restarted worker:
  a non-streamed request until its response is written (the worker's tokens stay in the front until the request ends,
  so it may have tokens and still be sent again), a stream until a token was queued for it (Front.delivered);
- a stream that already sent output ends with the route's error event (with the response id);
- a request whose second try also fails gets 500 (or the error event once its stream has started);
- while no worker is ready, new requests get 503 (Anthropic route 529) with Retry-After and x-should-retry: true;
- the front starts a new worker at once (with a growing pause, up to 30 s, if workers keep failing before ready) and
  takes requests again when it says ready. No step by hand.
- a worker whose program loaded from the saved executable table failed, in a request or in its warm-up before ready
  (its ERROR frame names the table), is replaced at once by workers that start without the table (MASHBOX_AOT=0) and
  compile before they say ready; the first of them is timed as a first start, and a failure in the warm-up is not
  counted as a failed start.
- a worker that cannot get ready ends the server instead of answering 503 forever: START_TRIES worker
  lives in a row that end before ready (a load that fails every time: wrong path, too small a chip), or a restart slower
  than START_SLOW_X x the first start (at least START_FLOOR_S), and the front stops serving; serve_split exits non-zero
  with the worker's own error.

/health reads the worker's engine (Front.health): its last HEALTH frame's numbers, the front's waiting count, and the
compiles after ready summed over every worker life, so a restart never hides a compile. The front prints the "ready" line
once, when it first takes requests.
"""
import collections
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid

from . import protocol as P
from .agents import BadRequest, NotReady, WorkerFault
from .server import Server


class RemoteEngine:
    """What the front's server needs to know about the engine (context, rows, vocabulary), filled in from the worker.
    check() is the engine's own admission check (runtime/engine.py Engine.check), run here so a request the engine
    cannot take is refused with 400 before any 200."""

    def __init__(self):
        self.CTX, self.S, self.V, self.DK, self.prefix_cache = 1 << 30, 1, None, 0, False

    def check(self, q):
        if self.V is None:
            return None
        from .engine import Engine          # the same check as the worker's engine, never a copy of it

        class _Shape:
            pass
        me = _Shape()
        me.CTX, me.c = self.CTX, _Shape()
        me.c.V = self.V
        return Engine.check(me, q)


class Link:
    """One worker process and its socket."""

    def __init__(self, proc, conn, sock_dir):
        self.proc, self.conn, self.dir = proc, conn, sock_dir
        self.ready, self.inflight, self.last = False, {}, time.monotonic()
        self.dead, self.fatal, self.t_start = threading.Event(), None, time.monotonic()
        self.stats, self.counted = None, False      # its engine's numbers (last HEALTH frame); added to the past lives once


class Front(Server):
    """server.Server whose loop sends requests to a worker process. factory = "module:function" returning the engine
    inside the worker; kwargs = JSON-able arguments for it; warm = run the engine's warm-up before ready."""

    def __init__(self, chat, model_id, info, factory, kwargs=None, warm=False, env=None, log=None, python=None,
                 not_ready_s=None, t0=None):
        self.factory, self.kwargs, self.warm_flag = factory, dict(kwargs or {}), warm
        self.env = dict(os.environ if env is None else env)
        if (self.env.get("MASHBOX_FAULT") or self.env.get("MASHBOX_FAULTS")) and not self.env.get("MASHBOX_FAULT_STATE"):
            import tempfile          # injected faults fire in one worker life, so the restarted worker comes back
            self.env["MASHBOX_FAULT_STATE"] = tempfile.mkdtemp(prefix="mashbox-faults-")
        self.python = python or sys.executable
        self.not_ready_s = float(not_ready_s if not_ready_s is not None else os.environ.get("MASHBOX_NOT_READY_S") or 900)
        self.watchdog_s = float(self.env.get("MASHBOX_WATCHDOG_MIN_S") or P.WATCHDOG_MIN_S)   # the worker's value
        self.pending, self.link, self.lock = collections.deque(), None, threading.RLock()
        self.workers_started, self.restarts, self.cold_ready_s, self.down_since = 0, 0, None, time.monotonic()
        self.t0, self.said_ready, self.compiles_before = time.time() if t0 is None else t0, False, 0
        self.wake, self.closed = threading.Event(), threading.Event()
        self.start_fails, self.first_ready_s, self.gave_up = 0, None, None
        self.start_floor_s = float(self.env.get("MASHBOX_START_FLOOR_S") or P.START_FLOOR_S)
        super().__init__(RemoteEngine(), chat, model_id, info, log)
        self.supervisor = threading.Thread(target=self.supervise, daemon=True)   # close() waits for it to end
        self.supervisor.start()

    # ------------------------------------------------------------------------------------------- worker lifecycle
    def _spawn(self):
        d = P.socket_dir()
        path = os.path.join(d, "worker.sock")
        cmd = [self.python, "-m", "mashbox.runtime.worker", "--socket", path, "--factory", self.factory,
               "--kwargs", json.dumps(self.kwargs)] + (["--warm"] if self.warm_flag else [])
        proc = subprocess.Popen(cmd, env=self.env, stdin=subprocess.DEVNULL)
        self.workers_started += 1
        t = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"worker exited with status {proc.returncode} before it listened")
            if os.path.exists(path):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(path)
                    break
                except OSError:
                    s.close()
            if time.monotonic() - t > 120:
                proc.kill()
                raise RuntimeError("worker did not listen within 120 s")
            time.sleep(0.02)
        conn = P.Conn(s)
        conn.send(dict(t="HELLO", proto=[P.PROTO], front_id=uuid.uuid4().hex))
        ack = conn.recv()
        if not ack or ack.get("t") != "HELLO_ACK" or ack.get("proto") != P.PROTO:
            proc.kill()
            raise RuntimeError(f"worker answered {ack and ack.get('t')} {ack and ack.get('code')} to HELLO")
        return Link(proc, conn, d)

    def supervise(self):
        pause = 0.0
        while not self.closed.is_set():
            try:
                link = self._spawn()
            except Exception as e:
                self.log(json.dumps(dict(event="worker_start_failed", error=f"{type(e).__name__}: {e}"[:300])))
                if self._start_failed("start_failed", f"{type(e).__name__}: {e}"):
                    return
                pause = min(30.0, max(0.5, pause * 2))
                self.closed.wait(pause)
                continue
            with self.lock:
                if self.closed.is_set():         # close() came while this worker started: end it, start no other
                    link.conn.close()
                    self._reap(link)
                    return
                self.link = link
            threading.Thread(target=self.read_loop, args=(link,), daemon=True).start()
            slow = None
            while not link.dead.wait(0.25):
                if link.inflight and time.monotonic() - link.last > self.watchdog_s + 5 * P.HEARTBEAT_S:
                    self.log(json.dumps(dict(event="worker_silent", s=round(time.monotonic() - link.last, 1))))
                    link.proc.kill()            # no frame, not even a heartbeat: the worker is stuck
                limit = self.start_limit_s()
                if slow is None and not link.ready and limit is not None and time.monotonic() - link.t_start > limit:
                    slow = (f"the restarted worker was not ready after {limit:.0f} s ({P.START_SLOW_X} x the first "
                            f"start, {self.first_ready_s:.1f} s)")
                    self.log(json.dumps(dict(event="worker_slow_start", limit_s=round(limit, 1))))
                    link.proc.kill()
            was_ready = link.ready
            self._reap(link)
            table = (link.fatal or {}).get("table")
            self.log(json.dumps(dict(event="worker_down", pid=link.proc.pid, status=link.proc.returncode,
                                     fault=(link.fatal or {}).get("code"), was_ready=was_ready, table=table)))
            if self.closed.is_set():
                return
            if table is not None:
                # a program loaded from the saved executable table failed (worker.py Worker.table_failed): every later
                # worker starts without the table, so a folder the worker could not close is never loaded again, and
                # compiles every program before it says ready; that start is timed as a first start (no restart
                # limit), and its ready time is the new base
                with self.lock:
                    self.env["MASHBOX_AOT"], self.first_ready_s = "0", None
                self.log(json.dumps(dict(event="aot_table_off", closed=table.get("closed"))))
            if not was_ready:
                f = link.fatal or {}
                if slow is not None:
                    return self.give_up("slow_start", slow)
                # a table failure in the warm-up is not a failed start: the next worker starts without the table
                if table is None and self._start_failed(
                        f.get("code") or "worker_exit",
                        f.get("msg") or f"the worker exited with status {link.proc.returncode} before it was ready"):
                    return
            self.restarts += 1
            pause = 0.0 if was_ready or table is not None else min(30.0, max(0.5, pause * 2))
            self.closed.wait(pause)

    def start_limit_s(self):
        """How long a restarted worker may take to get ready; None before the first worker was ready."""
        if self.first_ready_s is None:
            return None
        return max(P.START_SLOW_X * self.first_ready_s, self.start_floor_s)

    def _start_failed(self, code, msg):
        """One more worker life ended before ready; True when that was the last try (the front has given up)."""
        self.start_fails += 1
        if self.start_fails < P.START_TRIES:
            return False
        self.give_up(code, f"the model's worker failed {self.start_fails} times in a row before it was ready; "
                           f"its last error ({code}): {msg}")
        return True

    def give_up(self, code, why):
        """Stop serving for good: no worker can get ready. Requests here end (503 / 529), the HTTP server stops, and
        serve_split exits non-zero with `why`. Before a server has started, close() alone ends it."""
        self.gave_up = why
        self.log(json.dumps(dict(event="worker_gave_up", code=code, error=why[:600])))
        self.close()

        def stop_http():
            for _ in range(600):                  # serve() sets .httpd just after Front is made
                h = getattr(self, "httpd", None)
                if h is not None:
                    h.shutdown()                  # serve_forever returns; then the port is closed (nothing listens)
                    return h.server_close()
                time.sleep(0.05)
        threading.Thread(target=stop_http, daemon=True).start()

    @staticmethod
    def _reap(link, timeout=10):
        """Wait for the worker to exit (it exits by itself when its socket closes); kill it after `timeout`."""
        try:
            link.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            link.proc.kill()
            link.proc.wait()

    def close(self, timeout=10):
        """Stop serving: requests still here end (503 / 529 when they have no output, the stream's error event when
        they have), the worker process exits (killed after `timeout`) and no new one starts. Requests after close get
        503 / 529 at once. close() then waits, at most `timeout` more, for the supervisor thread to end, so its last
        line (worker_down) comes before front_closed and the supervisor writes nothing after close() returns: mashbox run
        exits right after it, and a daemon thread still writing to stdout at interpreter exit aborts the process
        (CPython's 'could not acquire lock for <stdout> at interpreter shutdown', SIGABRT). Not covered: a supervisor
        still starting a worker after that wait (_spawn waits up to the worker's start time) can still write its
        start-failure lines later, and a worker closed while it started ends with no worker_down. A second close(), also
        the one after give_up(), waits the same way."""
        with self.lock:
            if self.closed.is_set():
                return self._join_supervisor(timeout)
            self.closed.set()
            link, self.link, self.ready = self.link, None, False
            jobs = list(self.pending) + (list(link.inflight.values()) if link is not None else [])
            self.pending.clear()
            if link is not None:
                link.inflight = {}
            for j in jobs:
                self._unwait(j)
                self.fail([j], WorkerFault("the server stopped mid-answer; this answer is cut off here", code="shutdown")
                          if self.delivered(j) else NotReady("the server is shutting down; try again shortly", 1))
        self.wake.set()
        if link is not None:
            link.conn.close()
            self._reap(link, timeout)
        self._join_supervisor(timeout)
        self.log(json.dumps(dict(event="front_closed", ended=len(jobs))))

    def _join_supervisor(self, timeout):
        """Wait, at most `timeout` s, for the supervisor thread to end (it reaps the worker, writes worker_down and
        returns once closed is set; its worker's reader thread has ended before that: supervise waits for link.dead,
        the reader's last step). Not from the supervisor itself (give_up() closes from there)."""
        th = getattr(self, "supervisor", None)
        if th is not None and th is not threading.current_thread():
            th.join(timeout)

    def read_loop(self, link):
        try:
            while True:
                try:
                    m = link.conn.recv()
                except (P.FrameError, OSError):
                    m = None
                if m is None:
                    break
                link.last = time.monotonic()
                self.on_frame(link, m)
        finally:
            self._down(link)
            link.conn.close()
            link.dead.set()

    def on_frame(self, link, m):
        t = m["t"]
        if t == "HEALTH":
            if m.get("stats") is not None:
                link.stats = m["stats"]
            if m.get("state") == "ready" and not link.ready and m.get("limits"):    # the ready frame, not a heartbeat
                if m.get("info"):
                    self.info = dict(m["info"])      # the served weights' fields for /v1/models
                lim = m.get("limits") or {}
                E = self.engine
                E.S, E.CTX, E.V = int(lim.get("R") or 1), int(lim.get("ctx") or E.CTX), lim.get("vocab")
                E.DK, E.prefix_cache = int(lim.get("dk") or 0), bool(lim.get("prefix_cache"))
                with self.lock:
                    link.ready = True
                    self.cold_ready_s = time.monotonic() - link.t_start
                    if self.first_ready_s is None:
                        self.first_ready_s = self.cold_ready_s
                    self.start_fails = 0
                    self.ready = True
                self.log(json.dumps(dict(event="worker_ready", pid=link.proc.pid, ready_s=round(self.cold_ready_s, 2))))
                if not self.said_ready:              # the line the docs say to wait for, once
                    self.said_ready = True
                    self.log(json.dumps(dict(event="ready", seconds_since_start=round(time.time() - self.t0, 1))))
                self.wake.set()
            return
        if t == "ERROR" and m.get("req") is None:
            link.fatal = m                      # the worker exits next; _down handles its requests
            return
        with self.lock:
            j = link.inflight.get(m.get("req"))
            if j is None:
                return
            if t == "TOKENS":
                r = j["row"]
                r["ids"].extend(m.get("ids") or [])
                if r["ids"]:
                    j["got"] = True
                    if r["t_first"] is None:
                        r["t_first"] = time.perf_counter()
                self.emit(j)
            elif t == "FINISH":
                link.inflight.pop(m["req"], None)
                r, u = j["row"], m.get("usage") or {}
                r.update(fin=m.get("reason") or "length", cached=int(u.get("cached") or 0),
                         restored=int(u.get("restored") or 0), calls=m.get("calls") or 0, sampled=bool(m.get("sampled")),
                         t_end=time.perf_counter())
                j["cached"] = r["cached"]
                self.emit(j)
                self.finish(j, r, max(j.get("batch_rows") or 1, m.get("batch_rows") or 1))
                self.wake.set()
            elif t in ("REJECTED", "ERROR"):
                link.inflight.pop(m["req"], None)
                bad = m.get("code") in ("bad_params", "too_long")
                self.fail([j], BadRequest(m.get("msg") or "bad request") if bad else
                          WorkerFault(m.get("msg") or "the request failed on the device", code=m.get("code")))
                self.wake.set()

    def _down(self, link):
        """The worker is gone: retry the requests whose client has received no bytes of the answer (Front.delivered),
        end the others (module docstring)."""
        with self.lock:
            if self.link is link:
                self.link, self.ready, self.down_since = None, False, time.monotonic()
            if not link.counted:                 # this life's compiles after ready stay in /health
                link.counted = True
                self.compiles_before += int((link.stats or {}).get("compiles_after_ready") or 0)
            jobs, link.inflight = list(link.inflight.values()), {}
            code = (link.fatal or {}).get("code") or "worker_exit"
            retry = []
            for j in jobs:
                if j.get("cancel"):
                    self.finish(j, self._row(j, "cancel"), j.get("batch_rows") or 1)
                elif not self.delivered(j) and not j.get("retried"):
                    j["retried"], j["row"] = True, None
                    retry.append(j)
                elif not self.delivered(j):
                    self.fail([j], WorkerFault("the request failed twice on the device; try again", code=code, retried=True))
                else:
                    self.fail([j], WorkerFault("the model's worker stopped mid-answer (it restarts by itself); this "
                                               "answer is cut off here", code=code))
            self.pending.extendleft(reversed(retry))
        if jobs:
            self.log(json.dumps(dict(event="worker_lost_requests", code=code, retried=len(retry), ended=len(jobs) - len(retry))))

    @staticmethod
    def delivered(j):
        """True when the request's client may have received bytes of the answer: a stream (the job's "streamed", which
        Server.submit writes before it queues the job, so it holds from the first moment this front can run the job; a
        job without it counts as one) with a token queued for its handler (Server.emit's "sent"). Every route reads
        tokens as they come, streamed or not, but a request that is not streamed writes nothing to its client before it
        ends, whatever tokens the worker made."""
        return bool(j.get("sent")) and j.get("streamed", True) is not False

    @staticmethod
    def _row(j, fin):
        now = time.perf_counter()
        r = j.get("row") or dict(ids=[], t_first=None, t_admit=now, cached=0, restored=0, calls=0, sampled=False)
        r.update(fin=fin, t_end=now)
        r.setdefault("t_admit", now)
        return r

    HEALTH_FIELDS = ("live_rows", "cache_bytes", "host_cache_bytes", "snapshot_bytes", "cache_namespace", "deployment_id",
                     "component_set_sha", "routed_experts")

    def health(self):
        """/health's load fields for API callers, read where the engine runs: the worker's last HEALTH frame (null while
        no worker is ready), the waiting requests (this front keeps the queue), and compiles_after_ready = this worker's
        count + every earlier worker's (a restart never hides a compile; the front itself never compiles)."""
        with self.lock:
            link, before = self.link, self.compiles_before
            st = dict((link.stats if link is not None else None) or {})
        out = dict({k: None for k in self.HEALTH_FIELDS}, **st)
        out.update(waiting=self.waiting, certification=st.get("certification") or "not_certified",
                   compiles_after_ready=before + int(st.get("compiles_after_ready") or 0))
        return out

    def running(self):
        """Requests the worker holds now (sent to it, not finished): /metrics' running count, exact at any moment (the
        worker's HEALTH frame comes once a heartbeat)."""
        with self.lock:
            link = self.link
            return len(link.inflight) if link is not None else 0

    def health_extra(self):
        """Fields /health adds for the worker: its process id (the one to kill in a failure test) and restarts."""
        link = self.link
        return dict(worker_pid=link.proc.pid if link is not None else None, worker_restarts=self.restarts)

    def not_ready_retry_after(self):
        est = self.cold_ready_s or 30.0
        left = est - (time.monotonic() - self.down_since)
        return int(min(60, max(1, left)))

    # ------------------------------------------------------------------------------------------- the loop
    def loop(self):
        """Replaces the in-process batching loop: queue -> worker, at most one request per free worker row, taken in
        Server.next_waiting's order (round robin across owners, first in first out within one, a request sent again
        first), so one owner's long queue never holds back another owner's request."""
        while True:
            try:
                self.pending.append(self.q.get(timeout=0.01))
                while True:
                    self.pending.append(self.q.get_nowait())
            except queue.Empty:
                pass
            self.wake.clear()
            with self.lock:
                link = self.link
                if self.closed.is_set():                # closed: nothing waits for a worker that will not come
                    while self.pending:
                        j = self.pending.popleft()
                        self._unwait(j)
                        self.fail([j], NotReady("the server is shutting down; try again shortly", 1))
                for j in [j for j in self.pending if j.get("cancel")]:        # client gone while it waited
                    self.pending.remove(j)
                    self._unwait(j)
                    j["t_run"] = time.monotonic()
                    self.finish(j, self._row(j, "cancel"), 0)
                if link is not None:
                    for req, j in list(link.inflight.items()):
                        if j.get("cancel") and not j.get("cancel_sent"):     # client gone, or a stop string matched
                            j["cancel_sent"] = True
                            self._send(link, dict(t="CANCEL", req=req))
                if link is not None and link.ready:
                    while self.pending and len(link.inflight) < max(1, self.engine.S):
                        j = self.next_waiting(self.pending)     # round robin across owners, never by content
                        self._unwait(j)
                        if not self.still_to_admit([j]):        # dropped or left while it was being taken
                            continue
                        j.setdefault("t_run", time.monotonic())
                        req = uuid.uuid4().bytes
                        now = time.perf_counter()
                        j["row"] = dict(ids=[], fin=None, t_first=None, t_admit=now, t_end=None, cached=0, restored=0,
                                        calls=0, sampled=False)
                        link.inflight[req] = j
                        j["batch_rows"] = len(link.inflight)
                        self._send(link, dict(t="SUBMIT", req=req, model=self.model_id, ids=P.ids_bytes(j["ids"]),
                                              params=dict(max_tokens=j["max_new"], samp=j["samp"], mark=j.get("mark"),
                                                          scope=j.get("scope"), variant=j.get("variant"), rid=j.get("rid"),
                                                          stop_ids=list(self.chat.stop_ids))))
                elif self.pending and time.monotonic() - self.down_since > self.not_ready_s:
                    while self.pending:                  # no worker came back in time: 503 / 529, try again later
                        j = self.pending.popleft()
                        self._unwait(j)
                        self.fail([j], NotReady("the model is restarting; try again shortly", self.not_ready_retry_after()))
            self.wake.wait(0.01)

    def _send(self, link, msg):
        try:
            link.conn.send(msg)
        except OSError:
            pass                 # the reader sees the end of the socket and _down() takes the request


def serve_split(chat, model_id, info, factory, kwargs=None, warm=False, host="127.0.0.1", port=8000, log=print, ui=False,
                keys=None, env=None, t0=None):
    """Start the worker and serve HTTP on host:port (server.serve) with the front's failure handling. Returns only when
    the front gave up (no worker could get ready): then it exits non-zero with the worker's own error."""
    from .server import serve
    front = Front(chat, model_id, info, factory, kwargs, warm=warm, env=env, log=log, t0=t0)
    serve(front, host, port, log, ui, keys)
    if front.gave_up:
        raise SystemExit(f"mashbox: {front.gave_up}")


def serve_cli(a, argv, keys, log, t0=None):
    """`mashbox run MODEL` serving (no --prompt) with the engine in a worker process. This process keeps the HTTP
    front, the tokenizer and the API keys (cli.api_keys ran here: the tenant secret lives in the front, which computes
    each request's owner) and never opens the chip. The worker runs worker.from_cli(argv): cli.load + cli.warm, exactly
    as the one-process server, with the environment this process had before it was kept off the chip."""
    env = dict(os.environ)
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "jax" in sys.modules:
        sys.modules["jax"].config.update("jax_platforms", "cpu")
    from .. import hub
    from .chat import Chat
    m = hub.model_entry(a.model)
    chat = Chat(hub.upstream_dir(m, log))
    serve_split(chat, m["id"], {}, "mashbox.runtime.worker:from_cli", dict(argv=list(argv)), host=a.host, port=a.port,
                log=log, ui=a.ui, keys=keys, env=env, t0=t0)
