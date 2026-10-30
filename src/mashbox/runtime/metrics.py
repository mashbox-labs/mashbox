"""GET /metrics: what the server does, in the Prometheus text format (version 0.0.4).

The families follow the Model Server Protocol's metric types and meaning (requests running and waiting, KV cache use,
block size and block count; the protocol's names are recommendations) plus request latency histograms, so a Prometheus
scrape or an inference gateway reads the server's load without parsing logs. One slot holds one whole context here, so
a "block" is a slot: block_size = the context length, num_blocks = the slots.

No label names a tenant, an API key, a session or an agent: a scrape shows the server as a whole. The route is
behind the API key when the server has keys, as every /v1 route.
"""
import bisect
import collections
import threading

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0)
QUEUE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
E2E_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0)
TPOT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0)


class Histogram:
    def __init__(self, buckets):
        self.le = tuple(buckets)
        self.n = [0] * (len(self.le) + 1)       # the last count is above every bound (+Inf only)
        self.sum, self.count = 0.0, 0

    def observe(self, v):
        self.n[bisect.bisect_left(self.le, v)] += 1     # bucket le=x counts v <= x
        self.sum += v
        self.count += 1


def _num(v):
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def _labels(d):
    if not d:
        return ""
    esc = {k: str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') for k, v in d.items()}
    return "{" + ",".join(f'{k}="{v}"' for k, v in esc.items()) + "}"


class Metrics:
    """Counters and histograms of the requests a server answered. Server.run_chat and Server.complete call request() once
    per answered request (the moment they write its "request" log line, or return a completion); render() adds the
    gauges read from the server at scrape time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.finished = collections.Counter()           # API-level finish reason -> requests
        self.prompt_tokens = self.generation_tokens = 0
        self.cache_queries = self.cache_hits = 0         # prompt tokens looked up in the prefix cache / found there
        self.ttft = Histogram(TTFT_BUCKETS)
        self.queue = Histogram(QUEUE_BUCKETS)
        self.e2e = Histogram(E2E_BUCKETS)
        self.tpot = Histogram(TPOT_BUCKETS)

    def request(self, finish, prompt, cached, new, queue_s, e2e_s, ttft_s=None, tpot_s=None):
        """One answered request. finish: stop, length, client_closed or consumer_stalled; prompt / cached / new: its
        prompt tokens, how many of them came from the prefix cache, the tokens it got; seconds from arrival to a slot
        (queue_s), to the first token (ttft_s, None when no token came) and to the end (e2e_s); tpot_s = the mean time
        between its tokens after the first (None with fewer than 2 tokens)."""
        with self.lock:
            self.finished[str(finish)] += 1
            self.prompt_tokens += int(prompt)
            self.generation_tokens += int(new)
            self.cache_queries += int(prompt)
            self.cache_hits += int(cached or 0)
            self.queue.observe(max(0.0, queue_s))
            self.e2e.observe(max(0.0, e2e_s))
            if ttft_s is not None:
                self.ttft.observe(max(0.0, ttft_s))
            if tpot_s is not None:
                self.tpot.observe(max(0.0, tpot_s))

    def render(self, server):
        """The exposition text. Gauges come from the server now: running (rows in use; the front counts the requests its
        worker holds), waiting, slots and context, compiles after ready, worker restarts (0 in one process)."""
        h = server.health() if hasattr(server, "health") else {}
        S = int(getattr(server.engine, "S", 1) or 1)
        CTX = int(getattr(server.engine, "CTX", 0) or 0)
        running = server.running() if hasattr(server, "running") else h.get("live_rows")
        restarts = int(getattr(server, "restarts", 0) or 0)
        out = []

        def fam(name, typ, help_, samples):
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {typ}")
            for suffix, labels, v in samples:
                out.append(f"{name}{suffix}{_labels(labels)} {_num(v)}")

        def hist(name, help_, hg):
            samples, acc = [], 0
            for le, n in zip(hg.le, hg.n):
                acc += n
                samples.append(("_bucket", {"le": _num(le)}, acc))
            samples += [("_bucket", {"le": "+Inf"}, hg.count), ("_sum", None, hg.sum), ("_count", None, hg.count)]
            fam(name, "histogram", help_, samples)

        with self.lock:
            if running is not None:
                fam("mashbox:num_requests_running", "gauge", "Requests in a slot now (prefill or decode).",
                    [("", None, int(running))])
                fam("mashbox:kv_cache_usage_perc", "gauge",
                    "Share of the KV cache in use, 0 to 1 (slots in use / slots; one slot holds one whole context).",
                    [("", None, min(1.0, int(running) / S))])
            fam("mashbox:num_requests_waiting", "gauge", "Requests waiting for a slot.",
                [("", None, int(h.get("waiting", getattr(server, "waiting", 0)) or 0))])
            fam("mashbox:cache_config_info", "gauge",
                "KV cache layout: block_size = tokens per block (one slot = one context), num_blocks = slots.",
                [("", {"block_size": CTX, "num_blocks": S}, 1)])
            reasons = dict({"stop": 0, "length": 0}, **self.finished)
            fam("mashbox:request_success_total", "counter", "Requests answered, by finish reason.",
                [("", {"finished_reason": r}, n) for r, n in sorted(reasons.items())])
            fam("mashbox:prompt_tokens_total", "counter", "Prompt tokens of the answered requests.",
                [("", None, self.prompt_tokens)])
            fam("mashbox:generation_tokens_total", "counter", "Tokens generated for the answered requests.",
                [("", None, self.generation_tokens)])
            fam("mashbox:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prefix cache.",
                [("", None, self.cache_queries)])
            fam("mashbox:prefix_cache_hits_total", "counter", "Prompt tokens found in the prefix cache.",
                [("", None, self.cache_hits)])
            fam("mashbox:num_preemptions_total", "counter",
                "Requests taken out of their slot to make room for others (this engine never does).", [("", None, 0)])
            fam("mashbox:worker_restarts_total", "counter",
                "Times the model's worker process was started again after a fault (0 when the engine runs in the "
                "server's process).", [("", None, restarts)])
            if h.get("compiles_after_ready") is not None:
                fam("mashbox:compiles_after_ready", "gauge",
                    "Programs compiled since the server reported ready (0 when the warm-up covered every program).",
                    [("", None, int(h["compiles_after_ready"]))])
            hist("mashbox:time_to_first_token_seconds", "Seconds from a request's arrival to its first token.", self.ttft)
            hist("mashbox:request_queue_time_seconds", "Seconds a request waited for a slot.", self.queue)
            hist("mashbox:e2e_request_latency_seconds", "Seconds from a request's arrival to its end.", self.e2e)
            hist("mashbox:request_time_per_output_token_seconds",
                 "Mean seconds between a request's tokens after the first.", self.tpot)
        return "\n".join(out) + "\n"
