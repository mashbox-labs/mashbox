"""OpenAI-compatible HTTP server on 127.0.0.1 (reach it from your machine with an SSH tunnel).

Routes: GET /health, GET /v1/models, POST /v1/chat/completions (with tools), POST /v1/completions,
POST /v1/messages and /v1/messages/count_tokens (Anthropic Messages, for Claude Code), POST /v1/responses (OpenAI
Responses, for Codex); stream true or false. See agents.py for the coding-agent APIs. Every route, request field and
response shape is in runtime/openapi.py, served at GET /openapi.json (tests/cpu/test_openapi.py keeps the two equal).
With ui=True (`mashbox run <model> --ui`) GET / is the MASHBOX Chat page (mashbox.ui.chat) and /ui/... its files.
Continuous batching: up to S requests (the engine's slot count) are live at once, and a
request that arrives while others are mid-answer takes a free slot at the next device call. Its prompt is read in chunks
next to the decoding rows (a long prompt never stalls them for more than one chunk), it gets its first token at the call
where its prefill ends, and it ends at its own finish. Requests beyond S wait for a slot and are admitted round robin
across owners, in arrival order within one (Server.next_waiting). A streamed request's 200 goes out with its first token,
so a request refused while it waits keeps its own status code. The request
field "merge" is accepted and ignored: no request field changes how other rows are scheduled. Engines without
admit/step (test stubs) run the older whole-batch loop; speculative decoding (`mashbox run --mtp K`) runs inside the
continuous path's step.

Who may call it (runtime/guard.py): a request must name the server as localhost, 127.0.0.1, [::1], another IP address
or its own --host (a DNS-rebinding page names its own domain and is refused; with API keys any name, the key is the
check; MASHBOX_ALLOW_HOSTS adds names to a server without keys); a browser request must come from the
server's own origin (the chat page); a POST must be Content-Type application/json, so a web page cannot send one without
a CORS preflight, which this server never approves. Clients that are not browsers (curl, SDKs, coding agents) pass as
before. With API keys (cli.py: --api-key-file or MASHBOX_API_KEY; made for you when --host is not loopback) every /v1 and
/openapi.json request needs one (Authorization: Bearer or x-api-key), and each tenant's prompts have a prefix cache and
a waiting queue of their own: keys the key file names with one tenant share them, and a key without a name is its own
tenant; a cached prompt is reused, and reported as cached_tokens, only for requests of the same tenant. The owner of
a request is its tenant id (runtime/tenancy.py): an HMAC of the server's cache secret and the key's tenant name (the key
file may name one after the key) or the key itself, never the key's place in the key list; without keys there is one
local owner. A request whose client hangs up is ended, waiting or running, streamed or not. Waiting limits: at most
MAX_WAITING_OWNER requests of one owner wait at once, more get 429; at most MAX_WAITING wait overall: when that is full
and this owner is under its share, the newest waiting request of the owner with the most waiting is dropped
(longest-queue drop), else 429 (Anthropic route: 529 overloaded_error). 503 means only that the model is not ready (still
loading). Every refusal carries Retry-After (the estimated wait, 1 to 60 s) and goes out before any 200, the legacy
streaming route included.
"""
import collections
import hashlib
import io
import json
import math
import os
import queue
import select
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agents as A
from . import guard as G
from . import metrics as M
from . import programs
from . import tenancy as T
from .agents import BadRequest, NotReady, TooLong, WorkerFault
from .protocol import READ_S, STALL_S, raise_open_files

GATHER_S = 0.02      # how long the first request of a batch waits for others sent at the same time
MAX_WAITING = 256         # requests waiting for a slot at once, all owners (the queue is never unbounded)
MAX_WAITING_OWNER = 32    # per owner (4x the largest measured agent fan-out of 7); more get 429
SEND_BUFFER = 1 << 20     # bytes the kernel may hold for one streamed response (POLICY 1 MiB; capped by net.core.wmem_max)
KNOBS = dict(max_tokens=512, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=None,
             repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0)


def session_of(headers, b):
    """Short, stable tag for the client session a request belongs to, so the log can tell several agents apart: Claude
    Code's X-Claude-Code-Session-Id header, Codex's prompt_cache_key, or an X-Session-Id header any client can set.
    Logged as the first 8 hex digits of its sha256 (the id itself is not kept); None when the client sends none."""
    key = (headers.get("X-Claude-Code-Session-Id") or b.get("prompt_cache_key") or headers.get("X-Session-Id")
           or headers.get("session_id"))
    return hashlib.sha256(str(key).encode()).hexdigest()[:8] if key else None


def agent_of(headers, b):
    """Short tag for the subagent a request comes from, None for the main conversation. Subagents send the parent's session
    id (session_of), so the log needs a second key: Claude Code's x-claude-code-agent-id header (one per subagent), or
    Codex's x-openai-subagent header. Codex 0.165 subagents also send the parent's session-id and prompt_cache_key and no
    id of their own, so all subagents of one Codex session share one tag. Logged as 8 hex of sha256."""
    key = headers.get("x-claude-code-agent-id") or headers.get("X-Agent-Id") or headers.get("x-openai-subagent")
    return hashlib.sha256(str(key).encode()).hexdigest()[:8] if key else None


def refuse_unbuilt(b):
    """Not supported: log probabilities and structured output. A request that asks for either gets 400 naming the
    field, never an answer that silently lacks it (OpenAI chat, completions and Responses fields, the Anthropic
    output_config.format, and the guided_* / structured_outputs extensions other servers take)."""
    def no(field, what):
        raise BadRequest(f"`{field}` is not supported by this server ({what}); send the request without it")
    if b.get("logprobs") not in (None, False):
        no("logprobs", "log probabilities")
    if b.get("top_logprobs") not in (None, 0):
        no("top_logprobs", "log probabilities")
    inc = b.get("include")
    if isinstance(inc, list) and any(isinstance(x, str) and "logprobs" in x for x in inc):
        no("include: message.output_text.logprobs", "log probabilities")
    rf = b.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type") == "text"):
        no("response_format", "structured output")
    tf = b["text"].get("format") if isinstance(b.get("text"), dict) else None
    if tf is not None and not (isinstance(tf, dict) and tf.get("type") == "text"):
        no("text.format", "structured output")
    if isinstance(b.get("output_config"), dict) and b["output_config"].get("format") is not None:
        no("output_config.format", "structured output")
    for g in ("output_format", "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs"):
        if b.get(g) is not None:
            no(g, "structured output")


def knobs_of(b, max_key=None):
    refuse_unbuilt(b)
    k = {n: b.get(n, d) for n, d in KNOBS.items()}
    if b.get("max_completion_tokens") is not None:
        k["max_tokens"] = b["max_completion_tokens"]
    if max_key is not None:       # agent APIs: their own max key; absent = up to the end of the context (set later)
        k["max_tokens"] = b.get(max_key) if b.get(max_key) is not None else (k["max_tokens"] if b.get("max_tokens") is not None
                                                                            or b.get("max_completion_tokens") is not None else 1 << 30)
    if b.get("stop_sequences") is not None and k["stop"] is None:
        k["stop"] = b["stop_sequences"] or None
    try:
        k["max_tokens"], k["temperature"] = int(k["max_tokens"]), float(k["temperature"] or 0)
        k["top_p"], k["top_k"], k["min_p"] = float(1.0 if k["top_p"] is None else k["top_p"]), int(k["top_k"] or 0), float(k["min_p"] or 0)
        for n_, d_ in (("repetition_penalty", 1.0), ("frequency_penalty", 0.0), ("presence_penalty", 0.0)):
            k[n_] = float(d_ if k[n_] is None else k[n_])
    except (TypeError, ValueError, OverflowError):      # OverflowError: int() of a JSON Infinity or 1e999
        raise BadRequest("max_tokens, temperature, top_p, top_k, min_p and the penalties must be finite numbers")
    if not all(math.isfinite(k[n_]) for n_ in ("temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty",
                                                "presence_penalty")):
        raise BadRequest("temperature, top_p, min_p and the penalties must be finite numbers")
    if not (k["max_tokens"] <= (1 << 31) - 1 and k["top_k"] <= (1 << 31) - 1):
        raise BadRequest("max_tokens and top_k must fit 32 bits")
    if not (k["max_tokens"] >= 1 and 0 <= k["temperature"] <= 4 and 0 < k["top_p"] <= 1 and k["top_k"] >= 0 and 0 <= k["min_p"] < 1
            and k["repetition_penalty"] > 0 and -2 <= k["frequency_penalty"] <= 2 and -2 <= k["presence_penalty"] <= 2):
        raise BadRequest("max_tokens >= 1, temperature 0..4, top_p (0, 1], top_k >= 0, min_p [0, 1), repetition_penalty > 0, "
                         "frequency_penalty and presence_penalty -2..2")
    if isinstance(k["stop"], str):
        k["stop"] = [k["stop"]]
    if k["stop"] is not None and not (isinstance(k["stop"], list) and all(isinstance(x, str) and x for x in k["stop"])):
        raise BadRequest("stop must be a string or a list of non-empty strings")
    if b.get("n", 1) != 1:
        raise BadRequest("n must be 1")
    if k["seed"] is None:
        k["seed"] = int.from_bytes(os.urandom(7), "big")
    else:
        try:
            k["seed"] = int(k["seed"])
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("seed must be an integer")
        if not -(1 << 63) <= k["seed"] < (1 << 64):
            raise BadRequest("seed must fit 64 bits")
    return k


def samp_of(k):
    if k["temperature"] == 0 and k["repetition_penalty"] == 1.0 and k["frequency_penalty"] == 0 and k["presence_penalty"] == 0:
        return None
    return dict(temperature=k["temperature"], top_k=k["top_k"], top_p=k["top_p"], min_p=k["min_p"], seed=k["seed"],
                repetition_penalty=k["repetition_penalty"], frequency_penalty=k["frequency_penalty"],
                presence_penalty=k["presence_penalty"])


class Busy(Exception):
    """Over this owner's waiting limit: 429 with Retry-After."""
    def __init__(self, msg, retry_after=1):
        super().__init__(msg)
        self.retry_after = int(retry_after)


class Overloaded(Busy):
    """Overall waiting limit full: 429 (Anthropic route 529 overloaded_error) with Retry-After. Never 503: that means only
    that the model is not ready."""


def stop_point(chat, ids, stops):
    """Tokens up to the finish point: the shortest prefix of ids whose text holds a stop string (len(ids) if none).
    Usage counts these, never tokens the engine decoded past the stop inside a device call."""
    if not stops or not any(x in chat.decode(ids) for x in stops):
        return len(ids)
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if any(x in chat.decode(ids[:mid]) for x in stops):
            hi = mid
        else:
            lo = mid + 1
    return lo


class Text:
    """Incremental detokenisation with stop strings: emits only text that can no longer change."""

    def __init__(self, chat, stops):
        self.chat, self.stops, self.sent, self.done = chat, stops or [], 0, False
        self.hold = max((len(s) for s in self.stops), default=1) - 1

    def update(self, ids, final=False):
        text = self.chat.decode(ids)
        cut = min([text.find(s) for s in self.stops if s in text] or [-1]) if self.stops else -1
        if cut >= 0:
            text, final, self.done = text[:cut], True, True
        end = len(text) if final else max(self.sent, len(text) - self.hold - (1 if text.endswith("�") else 0))
        delta = text[self.sent:end]
        self.sent = end
        return delta


def cache_bytes_once(host, slot_snaps):
    """-> (host_cache_bytes, snapshot_bytes) for /health: the host memory the prompt cache holds, each array counted once.
    One state point can sit in several places at once: a slot saved to the host cache keeps its points that the next
    request on that slot shares (a system prompt), and two saves of one slot both hold that point, so summing each
    entry's own byte count would count it two or three times. host_cache_bytes = the distinct arrays of the host
    entries (their K/V blocks and points); snapshot_bytes = the distinct points of the slots that no host entry holds.
    Their sum is what a soak subtracts from RSS (net of every host-RAM cache). host: a HostCache, or
    the tuple of its entries that cache_view took."""
    seen = set()

    def once(t):
        if isinstance(t, dict):
            return sum(once(v) for v in t.values())
        if isinstance(t, (list, tuple)):
            return sum(once(v) for v in t)
        if id(t) in seen or not hasattr(t, "nbytes"):
            return 0
        seen.add(id(t))
        return int(t.nbytes)
    hb = 0
    for e in (host.items if hasattr(host, "items") else host or ()):
        hb += once([b for _, b in e.get("kv", ())]) + once([(x["r"], x["c"]) for x in e.get("snaps", ())])
    snaps = once([(x["r"], x["c"]) for sn in (slot_snaps or ()) for x in sn])
    return hb, snaps


def cache_view(engine):
    """-> (host entries, slot points): one moment of the prompt cache's structure, which cache_bytes_once(*view) counts.
    The engine's own loop takes it between device calls (Server.publish_cache_view), where nothing is half changed: a host
    entry never changes once made (HostCache.add appends it; an eviction or a namespace change replaces the list), and a
    slot's points change only as a whole list. So a shallow copy of those lists is that moment's cache, and /health,
    counting it later on another thread, counts exactly the arrays of that moment: never an entry evicted since, next to
    a slot's newer points."""
    host = getattr(engine, "host", None)
    return (tuple(host.items) if host is not None else (),
            tuple(tuple(sn) for sn in (getattr(engine, "snaps", None) or ())))


class Server:
    def __init__(self, engine, chat, model_id, info, log=None):
        self.engine, self.chat, self.model_id, self.info = engine, chat, model_id, info
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        im = chat.encode("<|im_start|>")
        self.im_start = im[0] if len(im) == 1 else None
        self.user = tuple(chat.encode("user"))
        self.q = queue.Queue()
        self.waiting, self.wlock = 0, threading.Lock()
        self.wait_by = collections.defaultdict(list)     # owner -> its waiting jobs, oldest first
        self.admits = 0                                  # admission counter ("seq" of the MASHBOX_TRACE admit line)
        self.last_admit, self.n_taken = {}, 0            # owner -> when it was last taken from the queue (next_waiting)
        self.ready = False
        self.metrics = M.Metrics()                       # GET /metrics (runtime/metrics.py)
        self.exit = os._exit        # Server.table_failed: a failed warm-up ends the process
        if os.environ.get("MASHBOX_ASSERT_OWNERS") == "1":      # the owner checks always run; this says so in the log
            self.log(json.dumps(dict(event="owner_asserts", on=True)))
        threading.Thread(target=self.loop, daemon=True).start()

    def health(self):
        """What /health adds for a caller that may use the API: live rows, waiting requests, prefix-cache bytes (host
        copies + slot snapshots), the host cache's cap in bytes and how many entries it evicted to stay under it (per-tenant
        floors decide which: a tenant inside cap / tenants is never evicted for another's traffic), the cache namespace and, once the compile key exists, the deployment id and component
        set, and the XLA compiles since ready (runtime/programs.py; 0 when the warm-up covered every program, None in a
        process that does not count them). Certification
        is a state the certification gate sets; this server reports not_certified. The cache bytes are those of the loop's last call
        boundary (publish_cache_view), one moment of the cache, not a sum read while the engine changes it."""
        E = self.engine
        host = getattr(E, "host", None)
        view = getattr(self, "cache_view", None)
        if view is None:                 # no loop has run yet (its first act is to publish), so nothing changes the cache
            view = cache_view(E)
        hb, snaps = cache_bytes_once(*view)
        return dict(live_rows=E.S - E.free_slots() if hasattr(E, "free_slots") else None, waiting=self.waiting,
                    cache_bytes=hb + snaps, host_cache_bytes=hb, snapshot_bytes=snaps,
                    host_cache_cap_bytes=host.cap if host is not None else 0,
                    host_cache_evictions=host.evicted if host is not None else 0,
                    cache_namespace=getattr(E, "cache_ns", None), deployment_id=getattr(E, "deployment_id", None),
                    component_set_sha=getattr(E, "component_set_sha", None), certification="not_certified",
                    compiles_after_ready=programs.counts()["compiles_after_ready"])

    def publish_cache_view(self):
        """The engine loop, at a call boundary: keep cache_view(engine) for /health, which another thread reads (the HTTP
        handler here, the heartbeat thread in a worker). A shallow copy of the entry and point lists, no array copied."""
        self.cache_view = cache_view(self.engine)

    # -------------------------------------------------------------------------------------------------- batching
    def loop(self):
        if hasattr(self.engine, "admit") and hasattr(self.engine, "step"):
            return self.loop_live()
        return self.loop_batches()

    def loop_live(self):
        """Continuous batching: between device calls, admit waiting requests into free
        slots (round robin across owners, first come, first served within one: next_waiting), then one engine step:
        a decode call for the decoding rows and a prefill chunk for
        rows still reading their prompt. A request waits only for a free slot, never for a batch to finish; the engine
        runs 1 decode step per call while anyone waits. No request field changes how other rows are scheduled ("merge"
        is accepted and ignored)."""
        E = self.engine
        wait, live = collections.deque(), {}
        while True:
            self.publish_cache_view()                 # a call boundary: /health counts this moment of the cache
            if not live and not wait:
                wait.append(self.q.get())
            while True:
                try:
                    wait.append(self.q.get_nowait())
                except queue.Empty:
                    break
            for j in [j for j in wait if j.get("cancel")]:      # client gone before it started
                wait.remove(j)
                self._unwait(j)
                j["t_run"] = time.monotonic()
                self.finish(j, dict(slot=None, ids=[], fin="cancel", cached=0, restored=0, sampled=False, calls=0,
                                    t_admit=0.0, t_first=0.0, t_end=0.0), 0)
            take, room = [], E.free_slots()
            while wait and room:
                take.append(self.next_waiting(wait))
                room -= 1
            if take:
                t_run = time.monotonic()
                for j in take:
                    j["t_run"] = t_run
                    self._unwait(j)
                take = self.still_to_admit(take)
            if take:
                try:
                    rows = E.admit([dict(ids=j["ids"], max_new=j["max_new"], samp=j["samp"], mark=j.get("mark"),
                                         scope=j.get("scope"), variant=j.get("variant")) for j in take],
                                   stop_ids=self.chat.stop_ids)
                except Exception as e:
                    if isinstance(e, programs.TableFailed):
                        wait.extendleft(reversed(self.table_failed(take + list(live.values()), e)))
                        live = {}
                        continue
                    if type(e).__name__ == "IsolationError":
                        self.log(json.dumps(dict(event="owner_assert", error=str(e)[:200], rows=len(take) + len(live))))
                    self.fail(take + list(live.values()), e)
                    live = {}
                    continue
                for j, r in zip(take, rows):
                    j["row"], j["cached"], j["batch_rows"] = r, r["cached"], len(live) + 1
                    if r["fin"] != "error":
                        self._admitted(j, r["slot"])
                    if r["fin"] == "error":            # refused by the engine's own check: this request only
                        self.fail([j], BadRequest(r.get("error") or "bad request"))
                    elif r["fin"] is None:
                        live[r["slot"]] = j
                    else:
                        self.finish(j, r, j["batch_rows"])
            if not live:
                continue
            gone = E.cancel([j["row"] for j in live.values() if j.get("cancel")])
            for r in gone:
                j = live.pop(r["slot"])
                self.finish(j, r, j["batch_rows"])
            if not E.busy():
                continue
            n_live = len(live)
            try:
                ended = E.step(waiting=bool(wait) or not self.q.empty())
            except Exception as e:
                if isinstance(e, programs.TableFailed):
                    wait.extendleft(reversed(self.table_failed(list(live.values()), e)))
                    live = {}
                    continue
                self.fail(list(live.values()), e)
                live = {}
                continue
            self.publish_cache_view()
            for j in live.values():
                j["batch_rows"] = max(j["batch_rows"], n_live)
                if j["row"]["ids"] or j["row"]["fin"] is not None:
                    self.emit(j)
            for r in ended:
                j = live.pop(r["slot"])
                self.finish(j, r, j["batch_rows"])

    def next_waiting(self, wait):
        """Take the next request to admit out of `wait` (a deque of waiting jobs in arrival order) and return it.
        Round robin across owners (tenant = owner group): the owner taken least recently goes first (an
        owner never taken before goes before every owner that was), first in, first out within an owner; a request sent
        again after a worker fault (its "retried" mark, runtime/front.py) goes before all. So a request of an owner with
        nothing else waiting joins at the next free row, however long other owners' queues are (bounded
        progress: a late request emits before an earlier max-length row ends). The order depends only on the
        arrivals and the earlier admissions, never on prompt content or cached state (no content-dependent
        ordering). With one owner (no API keys) it is plain first come, first served. Fallback: an empty `wait` raises
        IndexError, as deque.popleft() does."""
        if not wait:
            raise IndexError("next_waiting from an empty queue")
        if getattr(self, "last_admit", None) is None:    # a subclass that skips Server.__init__ (worker.Worker)
            self.last_admit, self.n_taken = {}, 0
        best, key, seen = None, None, set()
        for i, j in enumerate(wait):
            if j.get("retried"):
                best = i
                break
            sc = j.get("scope")
            if sc in seen:                       # only an owner's oldest waiting request competes
                continue
            seen.add(sc)
            k = (self.last_admit.get(sc, -1), i)
            if key is None or k < key:
                best, key = i, k
        j = wait[best]
        del wait[best]
        self.n_taken += 1
        self.last_admit[j.get("scope")] = self.n_taken
        return j

    def still_to_admit(self, taken):
        """The jobs of `taken` (taken from the queue by next_waiting, then off the waiting lists by _unwait) that still go to
        the engine. A job stays in its owner's waiting list until _unwait, so in that moment a longest-queue drop (submit)
        can pick it: it is then answered already (its "error": 429 / 529 + Retry-After) and is not sent; a job whose client
        left in that moment (hangup) ends as a cancel, as the loop's sweep ends one. _unwait takes the waiting lock, so after
        it no drop can pick the job."""
        keep = []
        for j in taken:
            if "error" in j:
                continue
            if j.get("cancel"):
                j.setdefault("t_run", time.monotonic())
                self.finish(j, dict(slot=None, ids=[], fin="cancel", cached=0, restored=0, sampled=False, calls=0,
                                    t_admit=0.0, t_first=0.0, t_end=0.0), 0)
                continue
            keep.append(j)
        return keep

    def _unwait(self, j):
        """A job stops counting against the waiting limits once (admitted, or its client left while it waited)."""
        with self.wlock:
            if not j.get("unwaited"):
                j["unwaited"] = True
                self.waiting -= 1
                mine = self.wait_by.get(j.get("scope"))
                if mine is not None and j in mine:
                    mine.remove(j)

    def _admitted(self, j, slot):
        """MASHBOX_TRACE=1: one log line per admitted request, written when the loop admits it, in admission order:
        {"event": "admit", "rid": the response id its client receives, "slot": its slot (whole-batch loop: its row in
        the batch), "seq": admission counter, "tenant": the first 11 characters of its tenant id, null for the one local
        owner}. A request refused at admission or whose client left while it waited gets no line."""
        if os.environ.get("MASHBOX_TRACE") != "1":
            return
        self.admits += 1
        sc = j.get("scope")
        self.log(json.dumps(dict(event="admit", rid=j.get("rid"), slot=slot, seq=self.admits,
                                 tenant=sc[:11] if isinstance(sc, str) else None)))

    def hangup(self, j):
        """The client left: the engine ends the job at its next step, and a job still waiting frees its waiting count
        at once, without waiting for a free slot (hung-up requests free their count within 0.5 s)."""
        j["cancel"] = True
        if "t_run" not in j:
            self._unwait(j)

    def retry_after(self):
        """Estimated wait in seconds, 1 to 60 (estimated as 2 s per request ahead per slot)."""
        return int(min(60, max(1, math.ceil(2.0 * self.waiting / max(1, getattr(self.engine, "S", 1))))))

    def emit(self, j):
        r = j["row"]
        if j.get("stream") is not None and not j.get("cancel"):
            ids = r["ids"][:j["max_new"]]
            j["sent"] = True        # a token is queued for the request's handler (a stream's client may have it)
            j["stream"].put((list(ids), r["fin"] is not None or len(ids) >= j["max_new"], r["fin"]))

    def finish(self, j, r, batch_rows):
        ids = r["ids"][:j["max_new"]]
        te = r["t_end"] or time.perf_counter()
        tf = r["t_first"] if r.get("t_first") is not None else te
        j["result"] = dict(ids=list(ids), finish="length" if len(ids) >= j["max_new"] else (r["fin"] or "length"),
                           batch_rows=batch_rows, cached=r["cached"], restored=r["restored"],
                           timing=dict(prefill_s=tf - r["t_admit"], decode_s=max(0.0, te - tf),
                                       decode_calls=r["calls"], new_tokens=len(r["ids"]), sampled=r["sampled"]))
        j.setdefault("cached", r["cached"])
        if os.environ.get("MASHBOX_TRACE") == "1":    # token ids and per-step logit hashes (engine trace=True), for checks
            self.log(json.dumps(dict(event="trace", ids=[int(t) for t in ids], cached=int(r["cached"]),
                                     lh=[[int(h), float(g)] for h, g in (r.get("lh") or [])][:len(ids)])))
        j["ev"].set()
        if j.get("stream") is not None:
            j["stream"].put(None)

    def table_failed(self, jobs, e):
        """A program loaded from the saved executable table failed in a request (programs.TableFailed): the engine
        dropped its live rows and the table, and quarantined the table's folder. This process never warms again: a compile here would be a compile after ready, and a failed device call restarts the engine's process.
        The server says not ready, every job of the call gets the error, and the process ends (exit code 70).
        `mashbox run` serves through the HTTP front, where the engine's process is a worker (runtime/worker.py
        Worker.table_failed): the front starts a new worker without the table, which compiles before it says ready,
        and sends a request whose client has received nothing once more. -> [] (nothing runs again in this process)"""
        self.ready = False
        self.log(json.dumps(dict(event="aot_table", phase="failed", **e.info)))
        self.fail(jobs, e)
        self.exit(70)
        return []

    def fail(self, jobs, e):
        for j in jobs:
            j["error"] = e
            j["ev"].set()
            if j.get("stream") is not None:
                j["stream"].put(None)

    def loop_batches(self):
        """Whole-batch loop for engines with generate() only: requests sent together run as one batch; later ones wait."""
        hold = collections.deque()
        S = self.engine.S
        while True:
            self.publish_cache_view()                 # between batches: /health counts this moment of the cache
            first = hold.popleft() if hold else self.q.get()
            # Requests sent together reach the queue a few milliseconds apart; wait up to GATHER_S for them, so they share
            # one batch instead of the first one running alone while the rest wait for it to finish.
            end = time.monotonic() + GATHER_S
            while len(hold) < S - 1:
                try:
                    hold.append(self.q.get(timeout=max(0.0, end - time.monotonic())))
                except queue.Empty:
                    break
            while True:
                try:
                    hold.append(self.q.get_nowait())
                except queue.Empty:
                    break
            batch = [first]
            while hold and len(batch) < S:
                batch.append(hold.popleft())
            self.run(batch)

    def run(self, batch):
        for j in batch:
            self._unwait(j)
        batch = [j for j in batch if "error" not in j]       # dropped from the queue while it was gathered: answered
        for j in [j for j in batch if j.get("cancel")]:       # the client left while it waited: nothing to run
            j["error"] = ConnectionResetError("client closed the connection")
            j["ev"].set()
            if j.get("stream") is not None:
                j["stream"].put(None)
        batch = [j for j in batch if not j.get("cancel")]
        if not batch:
            return
        for i, j in enumerate(batch):
            self._admitted(j, i)
        mx = max(j["max_new"] for j in batch)
        t_run = time.monotonic()
        for j in batch:                 # time spent waiting for the slot(s), logged as queue_s
            j["t_run"] = t_run

        def tap(ids, fin):
            for i, j in enumerate(batch):
                if "cached" not in j:
                    j["cached"] = (getattr(self.engine, "cached_now", None) or [0] * len(batch))[i]
                if j.get("stream") is not None and not j.get("cancel"):
                    j["stream"].put((ids[i][:j["max_new"]], fin[i] is not None or len(ids[i]) >= j["max_new"], fin[i]))
            # rows whose client hung up or whose stop string was found: the engine ends them at this step
            return [i for i, j in enumerate(batch) if j.get("cancel")]
        try:
            kw = dict(marks=[j.get("mark") for j in batch]) if getattr(self.engine, "prefix_cache", False) else {}
            if kw and any(j.get("scope") is not None for j in batch):     # API keys: each key's cache is its own
                kw["scopes"] = [j.get("scope") for j in batch]
            r = self.engine.generate([j["ids"] for j in batch], max_new=mx, stop_ids=self.chat.stop_ids,
                                     samp=[j["samp"] for j in batch], on_step=tap, **kw)
            for i, j in enumerate(batch):
                ids = r["ids"][i]
                j["result"] = dict(ids=ids[:j["max_new"]], finish="length" if len(ids) >= j["max_new"] else (r["finish"][i] or "length"),
                                   batch_rows=len(batch), timing=r["timing"], cached=(r.get("cached") or [0] * len(batch))[i],
                                   restored=(r.get("restored") or [0] * len(batch))[i])
        except Exception as e:      # reported to every request of the batch
            for j in batch:
                j["error"] = e
        for j in batch:
            j["ev"].set()
            if j.get("stream") is not None:
                j["stream"].put(None)

    def submit(self, ids, k, merge, stream, mark=None, clamp=False, scope=None, variant=None, rid=None, streamed=None):
        """Queue a request. stream: the job gets a token queue (every route reads its tokens as they come); streamed:
        its client reads a stream (None: the same as stream, so a caller that does not say counts as a stream), written
        into the job before it is queued, since the front may run the job and lose its worker before submit returns,
        and its retry rule reads this (Front.delivered); scope: its owner (tenant id, None = the one local owner);
        variant: the hash of its activation-edit table or adapter (None = no edits), part of its prefix-cache key; rid:
        the response id its client receives (the MASHBOX_TRACE admit line names it)."""
        room = self.engine.CTX - len(ids)
        if clamp:
            if room < 1:
                raise TooLong(f"prompt is too long: {len(ids)} tokens > {self.engine.CTX - 1} maximum "
                              f"(this server runs a context of {self.engine.CTX} tokens; start it with a larger --ctx or compact)")
            k["max_tokens"] = min(k["max_tokens"], room)
        if len(ids) + k["max_tokens"] > self.engine.CTX:
            raise BadRequest(f"prompt ({len(ids)} tokens) + max_tokens ({k['max_tokens']}) is over the context of {self.engine.CTX}")
        if hasattr(self.engine, "check"):      # admission phase 1: a request the engine cannot take is refused alone
            why = self.engine.check(dict(ids=ids, max_new=k["max_tokens"], samp=samp_of(k)))
            if why is not None:
                raise BadRequest(why)
        # merge: accepted and ignored (no request field changes how other rows are scheduled)
        job = dict(ids=ids, max_new=k["max_tokens"], samp=samp_of(k), ev=threading.Event(),
                   stream=queue.Queue() if stream else None, mark=mark, scope=scope, variant=variant, rid=rid,
                   streamed=bool(stream if streamed is None else streamed))
        with self.wlock:
            mine = self.wait_by[scope]
            if len(mine) >= min(MAX_WAITING_OWNER, MAX_WAITING):
                raise Busy(f"too many requests: {len(mine)} of yours are waiting; try again shortly", self.retry_after())
            if self.waiting >= MAX_WAITING:
                owners = [o for o, js in self.wait_by.items() if js]
                big = max(owners, key=lambda o: len(self.wait_by[o]))
                if big != scope and len(mine) < MAX_WAITING / len(set(owners) | {scope}):
                    v = self.wait_by[big].pop()                 # longest-queue drop: that owner's newest waiting request
                    v["unwaited"], v["cancel"] = True, True
                    self.waiting -= 1
                    v["error"] = Overloaded("server overloaded: this request was dropped from the queue; try again",
                                            self.retry_after())
                    v["ev"].set()
                    if v.get("stream") is not None:
                        v["stream"].put(None)
                else:
                    raise Overloaded(f"server overloaded: {self.waiting} requests are waiting; try again shortly",
                                     self.retry_after())
            self.waiting += 1
            mine.append(job)
        self.q.put(job)
        return job

    def wait(self, job, gone):
        """Wait for a job that streams nothing to its client; end it if the client hangs up meanwhile."""
        while not job["ev"].wait(0.25):
            if gone is not None and gone():
                self.hangup(job)
                raise ConnectionResetError("client closed the connection")

    # -------------------------------------------------------------------------------------------------- routes
    def prompt_ids(self, b, chat):
        if chat:
            ms = b.get("messages")
            if not isinstance(ms, list) or not ms:
                raise BadRequest("messages must be a non-empty list")
            kw = b.get("chat_template_kwargs") or {}
            think = bool(kw.get("enable_thinking", b.get("enable_thinking", False)))
            try:
                return self.chat.chat_ids([dict(role=m.get("role"), content=m.get("content") or "") for m in ms], think)
            except Exception as e:
                raise BadRequest(f"chat template: {e}")
        p = b.get("prompt")
        if isinstance(p, list) and len(p) == 1:
            p = p[0]
        if not isinstance(p, str) or not p:
            raise BadRequest("prompt must be a non-empty string")
        return self.chat.encode(p)

    def models(self):
        return dict(object="list", data=[dict(id=self.model_id, object="model", owned_by="mashbox-labs", created=0,
                                              max_model_len=self.engine.CTX, **self.info)])

    def complete(self, b, chat, write_sse=None, meta=None):
        """Plain completions (and chat without the agent parser). Streamed or not, the tokens are read as they come, so a
        stop string ends the row at the engine's next step and nothing past it is sent or counted."""
        t0 = time.monotonic()
        k = knobs_of(b)
        ids = self.prompt_ids(b, chat)
        stream = write_sse is not None
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        job = self.submit(ids, k, bool(b.get("merge", True)), True, scope=(meta or {}).get("scope"), rid=rid,
                          streamed=stream)  # not streamed, the client receives nothing before the end
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        tx = Text(self.chat, k["stop"])
        if stream and chat:
            write_sse(dict(id=rid, object=obj + ".chunk", created=created, model=self.model_id,
                           choices=[dict(index=0, delta=dict(role="assistant"), finish_reason=None)]))
        finish, parts, t_first, t_last = None, [], None, None
        gone_fn, t_chk = (meta or {}).get("gone"), time.monotonic()
        while True:
            try:
                ev = job["stream"].get(timeout=0.25)
            except queue.Empty:
                ev = False
            if gone_fn is not None and time.monotonic() - t_chk >= 0.25:  # a client that hangs up before anything
                t_chk = time.monotonic()                                    # is written to it (waiting, prefill)
                if gone_fn():                                               # ends it and frees its waiting count
                    self.hangup(job)
                    raise ConnectionResetError("client closed the connection")
            if ev is False:
                continue
            if ev is None:
                break
            got, done, fin = ev
            if tx.done:
                continue
            if got:
                t_last = time.monotonic()
                t_first = t_first or t_last
            delta = tx.update(got, final=done)
            done = done or tx.done
            if tx.done:
                job["cancel"] = True             # stop string found: the engine ends this row at its next step
            if done:
                finish = "stop" if (fin == "stop" or tx.done) else "length"
            if not stream:
                parts.append(delta)
                continue
            if delta or done:
                ch = dict(index=0, finish_reason=finish if done else None)
                if chat:
                    ch["delta"] = dict(content=delta)
                else:
                    ch["text"] = delta
                try:
                    write_sse(dict(id=rid, object=obj + ".chunk" if chat else obj, created=created, model=self.model_id,
                                   choices=[ch]))
                except (BrokenPipeError, ConnectionResetError):
                    self.hangup(job)            # the client hung up: the engine ends this row at its next step
                    raise
        job["ev"].wait()
        if "error" in job:
            raise job["error"]
        r = job["result"]
        n_out = stop_point(self.chat, r["ids"], k["stop"]) if tx.done else len(r["ids"])
        usage = dict(prompt_tokens=len(ids), completion_tokens=n_out, total_tokens=len(ids) + n_out)
        self.metrics.request(finish or ("stop" if (r["finish"] == "stop" or tx.done) else "length"), len(ids),
                             r.get("cached") or 0, n_out, queue_s=job.get("t_run", t0) - t0, e2e_s=time.monotonic() - t0,
                             ttft_s=None if t_first is None else t_first - t0,
                             tpot_s=(t_last - t_first) / (n_out - 1) if n_out > 1 and t_last > t_first else None)
        if stream:
            if not chat and (b.get("stream_options") or {}).get("include_usage"):
                write_sse(dict(id=rid, object=obj, created=created, model=self.model_id, choices=[], usage=usage))
            return None
        if finish is None:                       # the row ended with no last event (it never started)
            parts.append(tx.update(r["ids"], final=True))
            finish = "stop" if (r["finish"] == "stop" or tx.done) else "length"
        choice = dict(index=0, finish_reason=finish)
        choice.update(dict(message=dict(role="assistant", content="".join(parts))) if chat else dict(text="".join(parts)))
        return dict(id=rid, object=obj, created=created, model=self.model_id, choices=[choice], usage=usage,
                    system_fingerprint=f"seed-{k['seed']}")

    # -------------------------------------------------------------------------------------------------- agent APIs
    def marks_of(self, ids):
        """Points the prefix cache may remember besides the prompt end: the start (<|im_start|>) of every user message,
        then the start of the last message. The engine keeps the last one always and the others as llama.cpp keeps its
        user-message checkpoints (the first, then at least --checkpoint-min-step apart), so a compacted or new
        conversation with the same tools and system prompt resumes after them."""
        if self.im_start is None:
            return None
        u, n = self.user, len(self.user)
        at = [j for j in range(1, len(ids)) if ids[j] == self.im_start]
        if not at:
            return None
        return sorted(set([j for j in at if tuple(ids[j + 1:j + 1 + n]) == u] + [at[-1]]))

    def run_chat(self, msgs, tools, think, k, merge=True, on_start=None, on_event=None, meta=None):
        """Render, generate, parse. on_start(prompt_tokens) runs once the engine has sent the request's first event (its
        first token, or its end), never while it waits for a row: a streamed request's 200 goes out then, so a request
        refused or dropped while it waits (longest-queue drop, a worker that is not ready) gets its own status code
        (429 / 529 / 503 + Retry-After) and never a 200 followed by an error event (every refusal goes out before
        any 200). on_event(ev) gets each ("reasoning", s) / ("text", s) / ("tool", name, args).
        -> dict(text, reasoning, calls, finish, usage)."""
        t0 = time.monotonic()
        try:
            ids = self.chat.chat_ids(msgs, think, tools or None)
        except Exception as e:
            raise BadRequest(f"chat template: {e}")
        job = self.submit(ids, k, merge, True, mark=self.marks_of(ids), clamp=True, scope=(meta or {}).get("scope"),
                          rid=(meta or {}).get("rid"),
                          streamed=(meta or {}).get("closed") is not None)     # the handler sets closed for a stream only
        gone_fn = (meta or {}).get("gone")
        par, tx = A.OutParser(tools, think), Text(self.chat, k["stop"])
        out = dict(text=[], reasoning=[], calls=[], finish="length", stop_sequence=None)
        t_first, t_last, n_out, gone = None, None, 0, None
        closed = (meta or {}).get("closed")       # streaming: () -> True once the client has closed its side
        started, seen = on_start is None, False
        try:
            t_chk = time.monotonic()
            while True:
                try:
                    ev = job["stream"].get(timeout=0.25)
                except queue.Empty:
                    ev = False
                if gone_fn is not None and time.monotonic() - t_chk >= 0.25:  # a client that hangs up mid-answer when
                    t_chk = time.monotonic()                                    # nothing is written to it (waiting,
                    if gone_fn():                                               # prefill, not streamed) ends it too
                        raise ConnectionResetError("client closed the connection")
                if ev is False:
                    continue
                if ev is None:
                    break
                if closed is not None and closed():
                    raise BrokenPipeError("client closed the connection")
                if not started:                  # the engine's first event: the row is past every waiting refusal
                    started = True
                    on_start(len(ids))
                seen = True
                got, done, fin = ev
                if tx.done:
                    continue
                n_out = len(got)
                t_last = time.monotonic()
                if t_first is None and got:
                    t_first = t_last
                delta = tx.update(got, final=done)
                end = done or tx.done
                if tx.done:
                    job["cancel"] = True         # stop string found: the engine ends this row at its next step
                    n_out = stop_point(self.chat, got, k["stop"])
                for e in par.feed(delta, final=end):
                    (out["calls"].append(e[1:]) if e[0] == "tool" else out[e[0]].append(e[1]))
                    if on_event is not None:
                        on_event(e)
                if end:
                    out["finish"] = "stop" if (fin == "stop" or tx.done) else "length"
                    if tx.done:
                        out["stop_sequence"] = next((s for s in k["stop"] or [] if s in self.chat.decode(got)), None)
        except (BrokenPipeError, ConnectionResetError) as e:
            self.hangup(job)
            gone, out["finish"] = e, ("consumer_stalled" if isinstance(e, Stalled) else "client_closed")  # the engine
                                                                             # ends this row at its next step
        job["ev"].wait()
        if "error" in job:
            raise job["error"]               # before any event: the route answers it with its own status code
        if not started and gone is None:     # it ended with no event (no token): its 200 goes out with its end
            started = True
            on_start(len(ids))
        if not seen and gone is None and "result" in job:   # no event at all: its answer and finish are its result's,
            r = job["result"]                                # as the legacy route takes them (legacy())
            got = list(r.get("ids") or [])
            for e in par.feed(tx.update(got, final=True), final=True):
                (out["calls"].append(e[1:]) if e[0] == "tool" else out[e[0]].append(e[1]))
                if on_event is not None:
                    on_event(e)
            n_out = stop_point(self.chat, got, k["stop"]) if tx.done else len(got)
            out["finish"] = "stop" if (r.get("finish") == "stop" or tx.done) else "length"
            if tx.done:
                out["stop_sequence"] = next((s for s in k["stop"] or [] if s in self.chat.decode(got)), None)
        cached = job.get("cached", job.get("result", {}).get("cached", 0))
        out["usage"] = dict(prompt=len(ids), completion=n_out, cached=cached)
        # llama-server's timings block: prompt = queue end to first token (restore + prefill of the uncached tokens),
        # predicted = first to last token; predicted_per_second = tokens after the first / that time (decode only)
        t_run = job.get("t_run", t0)
        p_ms = ((t_first or time.monotonic()) - t_run) * 1e3
        g_ms = ((t_last or 0) - (t_first or 0)) * 1e3
        out["timings"] = dict(prompt_n=len(ids) - cached, prompt_ms=round(p_ms, 1),
                              prompt_per_second=round((len(ids) - cached) / p_ms * 1e3, 2) if p_ms > 0 else None,
                              predicted_n=n_out, predicted_ms=round(g_ms, 1),
                              predicted_per_second=round((n_out - 1) / g_ms * 1e3, 2) if n_out > 1 and g_ms > 0 else None,
                              cache_n=cached, queue_ms=round((t_run - t0) * 1e3, 1))
        self.log(json.dumps(dict(event="request", t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), api=(meta or {}).get("api"),
                                 ua=(meta or {}).get("ua"), model=(meta or {}).get("model"), session=(meta or {}).get("session"),
                                 agent=(meta or {}).get("agent"), queue_s=round(job.get("t_run", t0) - t0, 3), prompt_tokens=len(ids),
                                 cached_tokens=cached, restored_tokens=job.get("result", {}).get("restored", 0), new_tokens=n_out, tools=len(tools or []), tool_calls=len(out["calls"]),
                                 thinking=bool(think), finish=out["finish"], dropped_tools=(meta or {}).get("dropped_tools"), batch_rows=job.get("result", {}).get("batch_rows"),
                                 ttft_s=round((t_first or time.monotonic()) - t0, 3), total_s=round(time.monotonic() - t0, 3),
                                 prompt_ms=out["timings"]["prompt_ms"], predicted_ms=out["timings"]["predicted_ms"],
                                 predicted_per_second=out["timings"]["predicted_per_second"])))
        self.metrics.request(out["finish"], len(ids), cached, n_out, queue_s=t_run - t0, e2e_s=time.monotonic() - t0,
                             ttft_s=None if t_first is None else t_first - t0,
                             tpot_s=g_ms / 1e3 / (n_out - 1) if n_out > 1 and g_ms > 0 else None)
        if gone is not None:
            raise gone
        return out

    def count_tokens(self, b):
        cv, tools, extra = A.from_anthropic(b)
        think = (b.get("thinking") or {}).get("type") in ("enabled", "adaptive")
        try:
            return dict(input_tokens=len(self.chat.chat_ids(cv.messages(extra), think, tools or None)))
        except Exception as e:
            raise BadRequest(f"chat template: {e}")

    def openai_chat(self, b, sse=None, meta=None):
        k = knobs_of(b, "max_completion_tokens")
        cv, tools, extra = A.from_openai_chat(b)
        kw = b.get("chat_template_kwargs") or {}
        think = bool(kw.get("enable_thinking", b.get("enable_thinking", False)))
        cid, created = A.rid("chatcmpl-"), int(time.time())
        head = dict(id=cid, object="chat.completion.chunk", created=created, model=self.model_id)
        n_calls = [0]
        if sse is not None:
            sse.rid = cid

        def start(_n):
            sse.start()
            sse.data(dict(head, choices=[dict(index=0, delta=dict(role="assistant", content=""), finish_reason=None)]))

        def event(e):
            if e[0] == "text":
                d = dict(content=e[1])
            elif e[0] == "reasoning":
                d = dict(reasoning_content=e[1])
            else:
                d = dict(tool_calls=[dict(index=n_calls[0], id=A.rid("call_"), type="function",
                                          function=dict(name=e[1], arguments=json.dumps(e[2], ensure_ascii=False)))])
                n_calls[0] += 1
            sse.data(dict(head, choices=[dict(index=0, delta=d, finish_reason=None)]))
        r = self.run_chat(cv.messages(extra), tools, think, k, bool(b.get("merge", True)),
                          start if sse else None, event if sse else None, dict(meta or {}, rid=cid))
        fin = "tool_calls" if r["calls"] else r["finish"]
        u = r["usage"]
        usage = dict(prompt_tokens=u["prompt"], completion_tokens=u["completion"], total_tokens=u["prompt"] + u["completion"],
                     prompt_tokens_details=dict(cached_tokens=u["cached"]))
        if sse:
            sse.data(dict(head, choices=[dict(index=0, delta={}, finish_reason=fin)], timings=r["timings"]))
            if (b.get("stream_options") or {}).get("include_usage"):
                sse.data(dict(head, choices=[], usage=usage, timings=r["timings"]))
            sse.done()
            return None
        msg = dict(role="assistant", content="".join(r["text"]) or (None if r["calls"] else ""))
        if r["reasoning"]:
            msg["reasoning_content"] = "".join(r["reasoning"])
        if r["calls"]:
            msg["tool_calls"] = [dict(id=A.rid("call_"), type="function", function=dict(name=n, arguments=json.dumps(a, ensure_ascii=False)))
                                 for n, a in r["calls"]]
        return dict(id=cid, object="chat.completion", created=created, model=self.model_id,
                    choices=[dict(index=0, message=msg, finish_reason=fin)], usage=usage, timings=r["timings"])

    def anthropic(self, b, sse=None, meta=None):
        k = knobs_of(b, "max_tokens")
        cv, tools, extra = A.from_anthropic(b)
        think = (b.get("thinking") or {}).get("type") in ("enabled", "adaptive")
        mid = A.rid("msg_")
        blocks, cur = [], [None]          # finished content blocks, open block type
        if sse is not None:
            sse.rid = mid

        def start(n):
            sse.start()
            sse.event("message_start", dict(type="message_start", message=dict(
                id=mid, type="message", role="assistant", model=b.get("model") or self.model_id, content=[], stop_reason=None,
                stop_sequence=None, usage=dict(input_tokens=n, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0))))
            sse.event("ping", dict(type="ping"))

        def close():
            if cur[0] is None:
                return
            if cur[0] == "thinking":
                sse.event("content_block_delta", dict(type="content_block_delta", index=len(blocks) - 1,
                                                      delta=dict(type="signature_delta", signature="mashbox-local")))
            sse.event("content_block_stop", dict(type="content_block_stop", index=len(blocks) - 1))
            cur[0] = None

        def event(e):
            kind = dict(text="text", reasoning="thinking", tool="tool_use")[e[0]]
            if kind == "tool_use" or cur[0] != kind:
                close()
                cb = (dict(type="text", text="") if kind == "text" else dict(type="thinking", thinking="") if kind == "thinking"
                      else dict(type="tool_use", id=A.rid("toolu_"), name=e[1], input={}))
                blocks.append(cb)
                sse.event("content_block_start", dict(type="content_block_start", index=len(blocks) - 1, content_block=cb))
                cur[0] = kind
            d = (dict(type="text_delta", text=e[1]) if kind == "text" else dict(type="thinking_delta", thinking=e[1]) if kind == "thinking"
                 else dict(type="input_json_delta", partial_json=json.dumps(e[2], ensure_ascii=False)))
            sse.event("content_block_delta", dict(type="content_block_delta", index=len(blocks) - 1, delta=d))
            if kind == "tool_use":
                close()
        r = self.run_chat(cv.messages(extra), tools, think, k, True, start if sse else None, event if sse else None,
                          dict(meta or {}, rid=mid))
        stop = "tool_use" if r["calls"] else ("max_tokens" if r["finish"] == "length" else
                                               "stop_sequence" if r["stop_sequence"] else "end_turn")
        u = r["usage"]
        usage = dict(input_tokens=u["prompt"] - u["cached"], cache_read_input_tokens=u["cached"], cache_creation_input_tokens=0,
                     output_tokens=u["completion"])
        if sse:
            close()
            sse.event("message_delta", dict(type="message_delta", delta=dict(stop_reason=stop, stop_sequence=r["stop_sequence"]),
                                            usage=usage))
            sse.event("message_stop", dict(type="message_stop"))
            return None
        content = []
        if r["reasoning"]:
            content.append(dict(type="thinking", thinking="".join(r["reasoning"]), signature="mashbox-local"))
        if r["text"]:
            content.append(dict(type="text", text="".join(r["text"])))
        content += [dict(type="tool_use", id=A.rid("toolu_"), name=n, input=a) for n, a in r["calls"]]
        return dict(id=mid, type="message", role="assistant", model=b.get("model") or self.model_id, content=content,
                    stop_reason=stop, stop_sequence=r["stop_sequence"], usage=usage)

    def responses(self, b, sse=None, meta=None):
        k = knobs_of(b, "max_output_tokens")
        cv, tools, extra, kinds, dropped = A.from_responses(b)
        if dropped:
            meta = dict(meta or {}, dropped_tools=dropped)
        eff = (b.get("reasoning") or {}).get("effort")
        think = eff not in (None, "none", "minimal")
        resp = dict(id=A.rid("resp_"), object="response", created_at=int(time.time()), status="in_progress",
                    model=b.get("model") or self.model_id, output=[], parallel_tool_calls=bool(b.get("parallel_tool_calls", False)),
                    tool_choice=b.get("tool_choice", "auto"), tools=b.get("tools") or [], usage=None)
        seq, items, cur = [0], [], [None]
        if sse is not None:
            sse.rid, sse.api, sse.resp, sse.seq = resp["id"], "responses", resp, seq

        def ev(name, **o):
            seq[0] += 1
            sse.event(name, dict(type=name, sequence_number=seq[0], **o))

        def start(_n):
            sse.start()
            ev("response.created", response=resp)
            ev("response.in_progress", response=resp)

        def close():
            it = cur[0]
            if it is None:
                return
            oi = len(items) - 1
            if it["type"] == "message":
                part = dict(type="output_text", text=it["_t"], annotations=[])
                ev("response.output_text.done", item_id=it["id"], output_index=oi, content_index=0, text=it["_t"])
                ev("response.content_part.done", item_id=it["id"], output_index=oi, content_index=0, part=part)
                it["content"], it["status"] = [part], "completed"
            elif it["type"] == "reasoning":
                ev("response.reasoning_text.done", item_id=it["id"], output_index=oi, content_index=0, text=it["_t"])
                it["content"] = [dict(type="reasoning_text", text=it["_t"])]
            it.pop("_t", None)
            ev("response.output_item.done", output_index=oi, item=it)
            cur[0] = None

        def event(e):
            kind = dict(text="message", reasoning="reasoning", tool="call")[e[0]]
            if kind == "call":
                close()
                n, a = e[1], e[2]
                kind_, ns = kinds.get(n, ("function", None))
                if kind_ == "custom":
                    it = dict(type="custom_tool_call", id=A.rid("ctc_"), call_id=A.rid("call_"), name=n,
                              input=a.get("input", "") if isinstance(a.get("input", ""), str) else json.dumps(a.get("input")),
                              status="completed")
                else:
                    it = dict(type="function_call", id=A.rid("fc_"), call_id=A.rid("call_"), name=n,
                              arguments=json.dumps(a, ensure_ascii=False), status="completed")
                    if ns:
                        it["namespace"] = ns
                items.append(it)
                if sse:
                    ev("response.output_item.added", output_index=len(items) - 1,      # empty: the value comes in the delta
                       item=dict(it, status="in_progress", **({"arguments": ""} if it["type"] == "function_call" else {"input": ""})))
                    if it["type"] == "function_call":
                        ev("response.function_call_arguments.delta", item_id=it["id"], output_index=len(items) - 1, delta=it["arguments"])
                        ev("response.function_call_arguments.done", item_id=it["id"], output_index=len(items) - 1,
                           arguments=it["arguments"])
                    else:
                        ev("response.custom_tool_call_input.delta", item_id=it["id"], output_index=len(items) - 1, delta=it["input"])
                        ev("response.custom_tool_call_input.done", item_id=it["id"], output_index=len(items) - 1, input=it["input"])
                    ev("response.output_item.done", output_index=len(items) - 1, item=it)
                return
            if cur[0] is None or cur[0]["type"] != kind:
                close()
                it = (dict(type="message", id=A.rid("msg_"), status="in_progress", role="assistant", content=[], _t="")
                      if kind == "message" else dict(type="reasoning", id=A.rid("rs_"), summary=[], _t=""))
                items.append(it)
                cur[0] = it
                if sse:
                    ev("response.output_item.added", output_index=len(items) - 1, item={x: y for x, y in it.items() if x != "_t"})
                    if kind == "message":
                        ev("response.content_part.added", item_id=it["id"], output_index=len(items) - 1, content_index=0,
                           part=dict(type="output_text", text="", annotations=[]))
            it = cur[0]
            it["_t"] += e[1]
            if sse:
                ev("response.output_text.delta" if kind == "message" else "response.reasoning_text.delta", item_id=it["id"],
                   output_index=len(items) - 1, content_index=0, delta=e[1])

        if sse is None:
            class _Quiet:      # non-streaming: build the same items, write nothing
                def event(self, *a):
                    pass
            sse = _Quiet()
        r = self.run_chat(cv.messages(extra), tools, think, k, True, start if hasattr(sse, "start") else None, event,
                          dict(meta or {}, rid=resp["id"]))
        close()
        u = r["usage"]
        resp.update(output=items, usage=dict(input_tokens=u["prompt"], input_tokens_details=dict(cached_tokens=u["cached"]),
                                             output_tokens=u["completion"], output_tokens_details=dict(reasoning_tokens=0),
                                             total_tokens=u["prompt"] + u["completion"]))
        if r["finish"] == "length" and not r["calls"]:
            resp.update(status="incomplete", incomplete_details=dict(reason="max_output_tokens"))
        else:
            resp["status"] = "completed"
        if hasattr(sse, "start"):
            ev("response.incomplete" if resp["status"] == "incomplete" else "response.completed", response=resp)
            return None
        return resp


def stall_s():
    """Seconds a response write may wait for a client that reads nothing (MASHBOX_STALL_S, default protocol.STALL_S)."""
    try:
        v = float(os.environ.get("MASHBOX_STALL_S") or STALL_S)
    except ValueError:
        return float(STALL_S)
    return v if v > 0 else float(STALL_S)


def read_s():
    """Seconds a connection may wait for a request's first byte, and then for the whole request (MASHBOX_READ_S, default
    protocol.READ_S)."""
    try:
        v = float(os.environ.get("MASHBOX_READ_S") or READ_S)
    except ValueError:
        return float(READ_S)
    return v if v > 0 else float(READ_S)


class RequestReader(io.RawIOBase):
    """The request side of one connection, under a deadline: while armed, each read waits only until the deadline, so a
    client that sends nothing, half a request, or a byte now and then is cut off (TimeoutError: the standard request
    handler closes the connection). The deadline is read_s() for the first byte, then read_s() again from that byte for
    the whole request. Disarmed (None) once the body is in; then reads are the socket's own."""

    def __init__(self, sock):
        self.sock, self.deadline, self.got, self.timed_out, self.span = sock, None, 0, False, None

    def arm(self, span):
        self.deadline, self.got, self.timed_out, self.span = time.monotonic() + span, 0, False, span

    def readable(self):
        return True

    def readinto(self, b):
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                self.timed_out = True
                raise TimeoutError("the request did not arrive whole in time")
            self.sock.settimeout(left)
        try:
            n = self.sock.recv_into(b)
        except TimeoutError:
            self.timed_out = True
            raise
        if n and not self.got and self.deadline is not None:
            self.deadline = time.monotonic() + self.span     # the request has started: it gets the whole span to arrive
        self.got += n
        return n


class Stalled(ConnectionResetError):
    """consumer_stalled: the client stopped reading. A ConnectionResetError, so every route ends the
    request as a hang-up: the engine ends the row at its next step."""


class StallWriter:
    """The response writer of one POST. Once the request body is read (arm), every write must finish within stall_s
    seconds: a client that reads nothing fills the connection's buffers, the write times out and raises Stalled, and later
    writes raise at once. on_stall(stall_s) runs once, for the log. Before arm() it writes as the plain writer."""

    def __init__(self, w, on_stall):
        self.w, self.on_stall, self.stall_s, self.stalled = w, on_stall, None, False

    def write(self, b):
        if not self.stalled:
            try:
                return self.w.write(b)
            except socket.timeout:
                self.stalled = True
                self.on_stall(self.stall_s)
        raise Stalled(f"consumer_stalled: the client read nothing for {self.stall_s} s")

    def __getattr__(self, name):              # flush, close, closed: the plain writer's
        return getattr(self.w, name)


class SSE:
    """Server-sent events on one response: data-only lines (OpenAI chat) or named events (Anthropic, Responses)."""

    def __init__(self, h):
        self.h, self.started = h, False
        self.rid, self.api, self.resp, self.seq = None, None, None, None

    def start(self):
        h = self.h
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        self.started = True

    def closed(self):
        """The client has closed its side: the request handler's one check (H._gone), read before each event."""
        return self.h._gone()

    def _w(self, b):
        self.h.wfile.write(b)
        self.h.wfile.flush()

    def data(self, o):
        self._w(b"data: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n")

    def event(self, name, o):
        self._w(b"event: " + name.encode() + b"\ndata: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n")

    def done(self):
        self._w(b"data: [DONE]\n\n")

    def error(self, msg, anth, typ="invalid_request_error"):
        """The last event of a stream whose request failed after its 200 went out (a status code can no longer change):
        the route's error event with the response id as request_id, then the end of the stream. Anthropic: event error;
        Responses: event error, then response.failed; chat and completions: data {"error": ...}, then [DONE]."""
        try:
            if anth:
                self.event("error", dict(type="error", error=dict(type=typ, message=msg), request_id=self.rid))
            elif self.api == "responses":
                self.seq[0] += 1
                self.event("error", dict(type="error", code=typ, message=msg, param=None, sequence_number=self.seq[0]))
                self.seq[0] += 1
                fail = dict(self.resp, status="failed", error=dict(code=typ, message=msg))
                self.event("response.failed", dict(type="response.failed", sequence_number=self.seq[0], response=fail))
            else:
                self.data(dict(error=dict(message=msg, type=typ, param=None, code=None), request_id=self.rid))
                self.done()
        except (BrokenPipeError, ConnectionResetError):
            pass



# POST routes by their runtime/openapi.py ROUTES id; the server finds a request's route only through openapi.route.
POST_API = {"createChatCompletion": "chat", "createCompletion": "completions", "createMessage": "anthropic",
            "countTokens": "count_tokens", "createResponse": "responses"}
OPENAI_API = ("chat", "completions", "responses")      # the routes that check the request's model id

DRAIN_S = 10.0       # a stop waits this long for the requests still being answered (stop_serving), POLICY

class Listener(ThreadingHTTPServer):
    """The HTTP server. Its accept queue holds request_queue_size connections (the standard library's default is 5: with
    64 users connecting at once on a v6e-1, 5 overflowed and 15 of 128 requests were reset by peer). The kernel caps it at
    net.core.somaxconn."""
    request_queue_size = 1024

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.answering, self.answer_cv = 0, threading.Condition()

    def answer_begin(self):
        with self.answer_cv:
            self.answering += 1

    def answer_end(self):
        with self.answer_cv:
            self.answering -= 1
            self.answer_cv.notify_all()

    def drain(self, timeout):
        """Wait until no POST is being answered (each counts from the top of do_POST, once its path is read, to its last
        write), at most `timeout` seconds. -> how many are still being answered. An idle keep-alive connection is not
        counted."""
        end = time.monotonic() + timeout
        with self.answer_cv:
            while self.answering and end - time.monotonic() > 0:
                self.answer_cv.wait(end - time.monotonic())
            return self.answering


def key_scope(headers, keys):
    """The tenant id (runtime/tenancy.py) of the API key a request carries (Authorization: Bearer or x-api-key), None when
    it carries none of them. Every key is compared in full, in constant time. The id is an HMAC of the server's cache
    secret and the key's tenant (or the key), so it never depends on the key's place in the list."""
    a = headers.get("Authorization") or ""
    given = [g for g in (a[7:].strip() if a[:7].lower() == "bearer " else None, headers.get("x-api-key")) if g]
    hit = None
    for k in keys:
        for g in given:
            if G._same(g, k) and hit is None:
                hit = k
    return None if hit is None else T.current().tenant_id(hit)


def serve(server, host="127.0.0.1", port=8000, log=print, ui=False, keys=None):
    """keys: API keys a request must carry on /v1 and /openapi.json (None or empty: no key). Each key is its own prefix
    cache scope. The chat page, its files and /health stay open, as llama-server's public endpoints. port 0: the system
    picks a free port when the server binds; the "ui" (with ui) and "listening" log lines and server.httpd.server_address
    name it (the default stays 8000)."""
    S = server
    lim = raise_open_files()                    # one descriptor per connection: soft limit up to the hard limit
    if lim is not None:
        log(json.dumps(dict(event="open_files_limit", **lim)))
    keys = [k for k in (keys or []) if k]
    guard = G.Guard(hosts=[host])
    if keys:            # the key guards every API route, so any name may reach the server (a VM's DNS name, a proxy)
        guard.host_ok = lambda h: True
    UI = None
    if ui:                                     # the chat page at / and its files under /ui/
        from ..ui import chat as UI

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        scope = None
        api_name = None                 # the POST route being answered, for the consumer_stalled log line

        def _gate(self, key=True, anth=False):
            """Host, Origin and Content-Type for every request, and the API key when `key`. True = answered (refused),
            in the Anthropic error shape when `anth` (the route's family in runtime/openapi.py)."""
            err = guard.check(self.command, None, self.headers, auth=False)   # no session token here: no path needed
            if err is None and keys and key:
                self.scope = key_scope(self.headers, keys)
                if self.scope is None:
                    err = (401, "missing or wrong API key: send Authorization: Bearer <key> or x-api-key: <key>")
            if err is None:
                return False
            typ = {401: "authentication_error", 403: "permission_error"}.get(err[0], "invalid_request_error")
            self._err(err[0], err[1], typ, anth=anth)
            self.close_connection = True
            return True

        def _gone(self):
            """True once the client has closed its end of the connection (a request is answered whole, so nothing else
            arrives on it while one runs). poll(), not select(): select() cannot watch a descriptor numbered 1024 or more,
            so a server holding that many connections would read every client as gone and cut its answer."""
            try:
                if not hasattr(select, "poll"):         # Windows: its select() has no limit on descriptor numbers
                    r, _, _ = select.select([self.connection], [], [], 0)
                    return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
                p = select.poll()
                p.register(self.connection, select.POLLIN | select.POLLPRI)
                ev = p.poll(0)
                if not ev:
                    return False
                if ev[0][1] & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
                    return True
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):               # ValueError: the socket is already closed (descriptor -1)
                return True

        def _send(self, code, obj, retry_after=None, should_retry=False):
            body = json.dumps(obj).encode()
            self.send_response(code)
            if retry_after is not None or code in (429, 503, 529):
                self.send_header("Retry-After", str(retry_after or 1))
            if should_retry:                # not ready: the official clients retry on it
                self.send_header("x-should-retry", "true")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg, typ="invalid_request_error", anth=False, retry_after=None, should_retry=False, ecode=None):
            if anth:     # Anthropic error shape
                typ = {"invalid_request_error": "invalid_request_error", "unavailable": "overloaded_error",
                       "server_error": "api_error"}.get(typ, typ)
                return self._send(code, dict(type="error", error=dict(type=typ, message=msg)), retry_after, should_retry)
            e = dict(message=msg, type=typ)
            if ecode:
                e["code"] = ecode
            self._send(code, dict(error=e), retry_after, should_retry)

        def _not_ready(self, msg, anth, retry_after=None):
            """Not ready (loading, or the worker restarting after a fault): the only meaning of 503 (Anthropic 529)."""
            if retry_after is None:
                retry_after = S.not_ready_retry_after() if hasattr(S, "not_ready_retry_after") else 1
            return self._err(529 if anth else 503, msg, "unavailable", anth=anth, retry_after=retry_after, should_retry=True)

        def do_HEAD(self):
            if self._gate(key=False):
                return
            self.send_response(200 if S.ready else 503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            if self._gate(key=False):                    # the chat page and /health need no key
                return
            hit = UI.handle(path, S.model_id) if UI is not None else None
            if hit is not None:
                self.send_response(200)
                self.send_header("Content-Type", hit[1])
                self.send_header("Content-Length", str(len(hit[0])))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                return self.wfile.write(hit[0])
            from . import openapi                         # one route table for the server and its spec (as FastAPI)
            route, params = openapi.route("get", path)
            op = route and route["id"]
            if op in ("health", "healthShort", "root"):
                caller = not keys or key_scope(self.headers, keys) is not None
                more = S.health() if hasattr(S, "health") and caller else {}   # load: API callers only (a server
                if hasattr(S, "health_extra") and caller:                     # without load fields: status only)
                    more.update(S.health_extra())                             # worker process id and restarts
                return self._send(200 if S.ready else 503, dict(status="ok" if S.ready else "loading", **more),
                                  should_retry=not S.ready)
            if self._gate():
                return
            if op == "metrics":                          # Prometheus text (runtime/metrics.py); behind the key
                body = S.metrics.render(S).encode()
                self.send_response(200)
                self.send_header("Content-Type", M.CONTENT_TYPE)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if op == "listModels":
                return self._send(200, S.models())
            if op == "openapi":                          # the OpenAPI 3.1 description (runtime/openapi.py)
                return self._send(200, openapi.spec(S))
            if op == "getModel":
                m = S.models()["data"][0]
                return self._send(200, m) if params["model"] == m["id"] else self._err(404, f"model {params['model']} not found")
            self._err(404, "not found")

        def setup(self):
            super().setup()
            self.reader = RequestReader(self.connection)          # every request read goes through its deadline
            self.rfile = io.BufferedReader(self.reader)

        def handle_one_request(self):
            """Each request is read under a deadline (RequestReader): read_s() for its first byte, then read_s() for the
            whole request; a connection that sends nothing or half a request is closed after it, so idle or half-sent
            connections cannot pile up. Each answer goes through a StallWriter: once do_POST has read a body and
            armed it, a client that stops reading is ended after stall_s() and its connection closed. The connection's
            own timeout and writer are restored after each request, so a keep-alive connection waits for its next
            request as before."""
            w, t = self.wfile, self.connection.gettimeout()
            self.api_name = None
            self.reader.arm(read_s())
            self.wfile = sw = StallWriter(w, lambda s: S.log(json.dumps(dict(event="consumer_stalled", api=self.api_name,
                                                                              stall_s=s))))
            try:
                super().handle_one_request()
            finally:
                self.wfile = w
                if self.reader.timed_out:
                    self.close_connection = True
                    if self.reader.got:                 # half a request: logged (an idle keep-alive end is not)
                        S.log(json.dumps(dict(event="request_read_timeout", read_s=self.reader.span,
                                              bytes=self.reader.got)))
                self.reader.deadline = None
                if sw.stalled:
                    self.close_connection = True
                try:
                    self.connection.settimeout(t)
                except OSError:
                    pass

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            self.server.answer_begin()   # counted while answered: a stop waits for its last write (stop_serving)
            try:
                from . import openapi
                route, _ = openapi.route("post", path)
                api = self.api_name = POST_API.get(route and route["id"])
                anth = api in ("anthropic", "count_tokens")
                if self._gate(anth=anth):
                    return
                if api is None:
                    return self._err(404, f"not found: POST {path}", anth=anth)
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    b = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(b, dict):
                        raise BadRequest("body must be a JSON object")
                except (ValueError, BadRequest) as e:
                    return self._err(400, f"bad JSON body: {e}", anth=anth)
                self.reader.deadline = None                  # the request is in: no read deadline while it is answered
                if isinstance(self.wfile, StallWriter):   # the body is in: from here every write must finish in stall_s
                    self.wfile.stall_s = stall_s()
                    try:
                        self.connection.settimeout(self.wfile.stall_s)
                        if b.get("stream"):          # a stream keeps at most SEND_BUFFER bytes in the kernel for its client
                            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER)
                    except OSError:
                        pass
                if api in OPENAI_API and b.get("model") not in (None, "", S.model_id):
                    # OpenAI routes serve one model id (GET /v1/models); the Anthropic routes answer any model name, because
                    # Claude Code sends its own (main model and subagents) unless every model variable is set.
                    return self._err(404, f"The model `{str(b.get('model'))[:64]}` does not exist on this server; it serves "
                                          f"`{S.model_id}` (GET /v1/models)", ecode="model_not_found")
                if not S.ready:
                    return self._not_ready("the model is loading or restarting; try again shortly", anth)
                meta = dict(api=api, ua=(self.headers.get("User-Agent") or "")[:80], session=session_of(self.headers, b),
                            agent=agent_of(self.headers, b), model=str(b.get("model") or "")[:64] or None)
                meta.update(scope=self.scope, gone=self._gone)           # not logged: run_chat logs named fields only
                if os.environ.get("MASHBOX_DUMP_DIR"):        # debugging client compatibility: keep each request body
                    try:
                        with open(os.path.join(os.environ["MASHBOX_DUMP_DIR"], f"{time.time():.3f}-{api}.json"), "w") as f:
                            json.dump(dict(path=self.path, headers={k: v for k, v in self.headers.items()
                                                                    if k.lower() not in ("authorization", "x-api-key")}, body=b), f)
                    except OSError:
                        pass
                sse = SSE(self) if b.get("stream") else None
                if sse is not None:
                    meta["closed"] = sse.closed
                try:
                    if api == "count_tokens":
                        return self._send(200, S.count_tokens(b))
                    if api == "completions":
                        return self.legacy(b, False, meta)
                    fn = dict(chat=S.openai_chat, anthropic=S.anthropic, responses=S.responses)[api]
                    r = fn(b, sse, meta)
                    if sse is None:
                        return self._send(200, r)
                except NotReady as e:
                    if sse is not None and sse.started:
                        return sse.error(str(e), anth, "overloaded_error" if anth else "server_error")
                    return self._not_ready(str(e), anth, e.retry_after)
                except WorkerFault as e:
                    S.log(json.dumps(dict(event="error", api=api, error=f"WorkerFault {e.code}: {e}"[:300])))
                    if sse is not None and sse.started:
                        return sse.error(str(e), anth, "api_error" if anth else "server_error")
                    return self._err(500, str(e), "server_error", anth=anth)
                except Busy as e:
                    over = isinstance(e, Overloaded)
                    typ = ("overloaded_error" if anth else "server_overloaded") if over else ("rate_limit_error" if anth else "rate_limit_exceeded")
                    if sse is not None and sse.started:
                        return sse.error(str(e), anth, typ)
                    return self._err(529 if (over and anth) else 429, str(e), typ, anth=anth, retry_after=e.retry_after)
                except BadRequest as e:
                    S.log(json.dumps(dict(event="rejected", api=api, ua=meta["ua"], error=str(e)[:300])))
                    if sse is not None and sse.started:
                        return sse.error(str(e), anth)
                    return self._err(400, str(e), anth=anth)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as e:
                    S.log(json.dumps(dict(event="error", api=api, error=f"{type(e).__name__}: {e}"[:300])))
                    if sse is not None and sse.started:
                        return sse.error(f"{type(e).__name__}: {e}", anth, "server_error" if not anth else "api_error")
                    return self._err(500, f"{type(e).__name__}: {e}", "server_error" if not anth else "api_error", anth=anth)
            finally:
                self.server.answer_end()

        def legacy(self, b, chat, meta=None):
            if b.get("stream"):
                sent = []

                def head():         # the 200 goes out with the first event, so a refusal before it keeps its own status
                    if not sent:
                        sent.append(1)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True

                def write(o):
                    head()
                    self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
                    self.wfile.flush()
                try:
                    S.complete(b, chat, write, meta)
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as e:
                    if not sent:
                        raise                   # do_POST answers 400 / 429 / 500 / 503 with its status code
                    plain = isinstance(e, (BadRequest, Busy, NotReady, WorkerFault))
                    try:              # after the 200: the error event, then [DONE]
                        write(dict(error=dict(message=str(e) if plain else f"{type(e).__name__}: {e}", param=None, code=None,
                                              type="invalid_request_error" if isinstance(e, (BadRequest, Busy)) else "server_error")))
                    except (BrokenPipeError, ConnectionResetError):
                        return
                head()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            return self._send(200, S.complete(b, chat, meta=meta))

        def log_message(self, *a):
            pass

    httpd = Listener((host, port), H)
    server.httpd = httpd                       # a front that gives up (runtime/front.py) stops serving through it
    port = httpd.server_address[1]             # the bound port (port 0: the one the system picked)
    if ui:
        log(json.dumps(dict(event="ui", url=f"http://{host}:{port}/")))
    log(json.dumps(dict(event="listening", url=f"http://{host}:{port}/v1")))
    got = []
    main = threading.current_thread() is threading.main_thread()   # signal handlers exist only in the main thread

    def on_signal(signum, frame):            # the first SIGINT or SIGTERM: later ones are ignored from here on
        got.append(signum)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt
    try:              # installing the handlers is inside the block, so a signal right after it still stops the server
        if main:
            signal.signal(signal.SIGTERM, on_signal)
            if signal.getsignal(signal.SIGINT) is signal.default_int_handler:   # a SIGINT the parent ignores stays so
                signal.signal(signal.SIGINT, on_signal)
        httpd.serve_forever()
    except KeyboardInterrupt:
        stop_serving(server, httpd, log, got[0] if got else signal.SIGINT, main)


def stop_serving(server, httpd, log, signum, main=True):
    """SIGINT or SIGTERM (the process group exits, rc 0 / 130, no orphan): stop taking
    connections, then close the server: a front (runtime/front.py) ends its requests (a stream with output gets its
    error event, one without gets 503 / 529) and waits for its worker process to exit, killing it after Front.close's
    timeout, so no process of the server outlives it, also when the worker is mid-call on the device. Then it waits, at
    most DRAIN_S, for every request still being answered to write its last bytes (that error event or that 503 / 529), so
    no client gets half an answer. A second signal while this runs is ignored (from the first
    signal's handler on), so it cannot leave the worker behind. Then exit 128 + the signal number. Fallback: a server
    without close() (the one-process server: its engine thread ends with the process) waits for its answers the same
    way, then exits."""
    if main:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
    log(json.dumps(dict(event="stopping", signal=signal.Signals(signum).name)))
    try:
        httpd.server_close()
    finally:
        close = getattr(server, "close", None)
        if callable(close):
            close()
        left = httpd.drain(DRAIN_S) if hasattr(httpd, "drain") else 0
        log(json.dumps(dict(event="stopped", answering=left)))
    raise SystemExit(128 + int(signum))
