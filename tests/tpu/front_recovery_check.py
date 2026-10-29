"""Failure rules A13 on a TPU chip: the HTTP front (runtime/front.py) with the real engine in a worker process.

  python tests/tpu/front_recovery_check.py OUT_DIR [--model qwen3-8-27b] [--n-fault 6] [-- extra `mashbox run` args]

One front, three worker lives, the same greedy request each time (chat, temperature 0):
  1. cold start of worker 1 (load + compile warm-up), the baseline answer;
  2. kill -9 of worker 1 while a stream is mid-answer: the stream ends with the error event, requests during the restart
     get 503 (Anthropic route 529) with Retry-After and x-should-retry, worker 2 comes back by itself;
  3. worker 2 starts with MASHBOX_FAULTS=device_error@N (a device error at its N-th engine call): the stream has output,
     then the error event; worker 3 comes back by itself and answers exactly the baseline;
  4. Front.close(): the worker exits and no process holds the TPU.
/health is read where the engine runs: deployment id and component set come from the worker (the same after both
restarts), compiles_after_ready never drops when a worker dies (its value is recorded; zero is CERT's gate B2), and the
front prints the "ready" line once.
Writes OUT_DIR/events.jsonl (every step with a UTC time), OUT_DIR/front.log (front and worker lines), the raw streams,
and OUT_DIR/summary.json (cold ready, each restart's ready time, pass/fail per rule). Exit 0 when every rule holds.
The front runs with JAX on CPU so it never opens the TPU; the worker gets the caller's environment.
"""
import argparse
import datetime
import json
import os
import platform
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ENV0 = dict(os.environ)                         # the worker's environment (TPU), taken before the front sets CPU
os.environ["JAX_PLATFORMS"] = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "cpu"))

from mashbox import hub                              # noqa: E402
from mashbox.runtime.chat import Chat               # noqa: E402
from mashbox.runtime import front as F              # noqa: E402
import api_stub_engine as STUB                      # noqa: E402  (serve_on_free_port only)

PROMPT = "Write three short sentences about the sea."
MODEL = None                                    # the served model id (OpenAI routes answer 404 for any other)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Rec:
    def __init__(self, out):
        self.out, self.t0 = out, time.monotonic()
        os.makedirs(out, exist_ok=True)
        self.ev = open(os.path.join(out, "events.jsonl"), "a")
        self.lines = open(os.path.join(out, "front.log"), "a")
        self.lock = threading.Lock()

    def __call__(self, event, **kw):
        row = dict(utc=now(), t=round(time.monotonic() - self.t0, 3), event=event, **kw)
        with self.lock:
            self.ev.write(json.dumps(row) + "\n")
            self.ev.flush()
        print(json.dumps(row), flush=True)
        return row

    def log(self, m):
        with self.lock:
            self.lines.write(f"{now()} {m}\n")
            self.lines.flush()

    def save(self, name, text):
        with open(os.path.join(self.out, name), "w") as f:
            f.write(text)


def post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")
    except OSError:
        return {}


def body(n, stream):
    return dict(model=MODEL, messages=[dict(role="user", content=PROMPT)], max_tokens=n, temperature=0, stream=stream)


def use_ids():
    """--idchat (a CPU dry run on the tiny model of api_stub_engine.build_tiny): token ids are the text."""
    global PROMPT
    PROMPT = "12 7 301 44 9 250 3"


def stream_text(raw):
    """(text, error events, finish reasons) of a chat SSE body; text = reasoning then answer, as sent."""
    text, errs, fins = [], [], []
    for blk in raw.split("\n\n"):
        for ln in blk.split("\n"):
            if not ln.startswith("data: ") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[6:])
            if "error" in d:
                errs.append(d)
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                text.append((dl.get("reasoning_content") or "") + (dl.get("content") or ""))
                if c.get("finish_reason"):
                    fins.append(c["finish_reason"])
    return "".join(text), errs, fins


def whole_text(raw):
    m = json.loads(raw)["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


def stream_and_cut(port, fr, n, cut, rec, name):
    """Stream a greedy answer; once `cut` pieces have arrived, call `cut_fn` (kill) if given. Returns the raw body."""
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps(body(n, True)), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw, pieces, fired = [], 0, False
    while True:
        ln = r.readline()
        if not ln:
            break
        s = ln.decode()
        raw.append(s)
        if s.startswith("data: {") and '"choices"' in s:
            pieces += 1
            if cut is not None and pieces >= cut[0] and not fired:
                fired = True
                cut[1]()
    c.close()
    out = "".join(raw)
    rec.save(name, out)
    return r.status, out


def probe_not_ready(port, fr, rec, until_s=1800):
    """While no worker is ready: one OpenAI and one Anthropic request; record codes and headers. Returns (seconds to
    ready, probes)."""
    t, probes = time.monotonic(), {}
    while time.monotonic() - t < until_s:
        if fr.ready:
            break
        for k, path, b in (("openai", "/v1/chat/completions", body(8, False)),
                           ("anthropic", "/v1/messages", dict(model=MODEL, max_tokens=8, messages=[dict(role="user", content=PROMPT)]))):
            if k not in probes and not fr.ready:
                code, raw, h = post(port, path, b, timeout=30)
                if not fr.ready:
                    probes[k] = dict(code=code, retry_after=h.get("Retry-After"), should_retry=h.get("x-should-retry"),
                                     type=(json.loads(raw).get("error") or {}).get("type") if raw.startswith("{") else None)
        time.sleep(0.2)
    return time.monotonic() - t, probes


def tpu_holders():
    """Processes with a TPU device file open (/dev/vfio/*, /dev/accel*)."""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
                if tgt.startswith("/dev/vfio/") or tgt.startswith("/dev/accel"):
                    out.append(int(pid))
                    break
        except OSError:
            continue
    return sorted(set(out))


def chip():
    """Accelerator type and zone from the TPU VM's metadata server (None off a VM)."""
    out = {}
    for k, path in (("accelerator", "instance/attributes/accelerator-type"), ("zone", "instance/zone")):
        try:
            rq = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/" + path,
                                        headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(rq, timeout=2) as r:
                out[k] = r.read().decode().rsplit("/", 1)[-1]
        except OSError:
            out[k] = None
    return out


def _stop(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)   # a start from a background shell inherits SIGINT ignored
    signal.signal(signal.SIGTERM, _stop)
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="qwen3-8-27b")
    ap.add_argument("--n-fault", type=int, default=6, help="worker 2's device call that fails")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--sha", default=os.environ.get("CODE_SHA", "unknown"))
    ap.add_argument("--factory", default="mashbox.runtime.worker:from_cli", help="worker factory (CPU dry run: "
                    "api_stub_engine:tiny with --kwargs '{\"wd\": ...}' and --idchat)")
    ap.add_argument("--kwargs", default=None, help="JSON kwargs for --factory (default: the `mashbox run` argv)")
    ap.add_argument("--idchat", action="store_true", help="token ids as text instead of the model's tokenizer")
    ap.add_argument("run_args", nargs="*", help="extra `mashbox run` arguments for the worker (after --)")
    a = ap.parse_args()
    rec = Rec(a.out)
    import jax
    rec("start", code_sha=a.sha, host=platform.node(), python=platform.python_version(), jax=jax.__version__,
        model=a.model, run_args=a.run_args, tpu_holders_before=tpu_holders(), **chip())
    if a.idchat:
        use_ids()
        m, chat = dict(id=a.model), STUB.IdChat()
    else:
        m = hub.model_entry(a.model)
        chat = Chat(hub.upstream_dir(m, rec.log))
    state = tempfile.mkdtemp(prefix="front-check-faults-")
    env = dict(ENV0, MASHBOX_FAULT_STATE=state)
    env.pop("MASHBOX_FAULTS", None)
    env.pop("MASHBOX_FAULT", None)
    global MODEL
    MODEL = m["id"]
    kw = json.loads(a.kwargs) if a.kwargs else dict(argv=["run", a.model] + a.run_args)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "cpu"), os.path.join(HERE, "..", "..", "src"),
                                         env.get("PYTHONPATH", "")])
    t_start = time.monotonic()                           # the worker starts inside Front()
    fr = F.Front(chat, m["id"], {}, a.factory, kw, env=env, log=rec.log)
    try:
        return checks(a, rec, fr, t_start)
    finally:
        fr.close()                                       # the worker never outlives this script


def checks(a, rec, fr, t_start):
    port = STUB.serve_on_free_port(fr)
    S, ok = {}, {}
    while not fr.ready and time.monotonic() - t_start < 3600:
        time.sleep(0.5)
    S["cold_ready_s"] = round(time.monotonic() - t_start, 1)
    h1 = health(port)
    rec("worker1_ready", seconds=S["cold_ready_s"], health=h1)

    code, raw, _ = post(port, "/v1/chat/completions", body(a.tokens, False))
    base = whole_text(raw) if code == 200 else ""
    rec.save("baseline.json", raw)
    code_s, raw_s = stream_and_cut(port, fr, a.tokens, None, rec, "baseline-stream.sse")
    t_s, errs_s, fins_s = stream_text(raw_s)
    ok["baseline"] = code == 200 and bool(base) and t_s == base and not errs_s and len(fins_s) == 1
    rec("baseline", ok=ok["baseline"], chars=len(base), stream_equals_whole=t_s == base, finish=fins_s)

    # 2. kill -9 mid-stream; worker 2 is armed with a device error at its n-th engine call
    pid1 = fr.link.proc.pid
    fr.env["MASHBOX_FAULTS"] = f"device_error@{a.n_fault}"
    kill = {}

    def do_kill():
        kill["t"] = time.monotonic()
        os.kill(pid1, signal.SIGKILL)
        rec("killed", pid=pid1)
    code_k, raw_k = stream_and_cut(port, fr, a.tokens, (8, do_kill), rec, "kill9-stream.sse")
    t_k, errs_k, fins_k = stream_text(raw_k)
    ok["kill9_stream"] = (code_k == 200 and bool(t_k) and base.startswith(t_k) and t_k != base and len(errs_k) == 1
                          and errs_k[0].get("request_id", "").startswith("chatcmpl-") and not fins_k)
    rec("kill9_stream", ok=ok["kill9_stream"], chars=len(t_k), error=errs_k, finish=fins_k)
    if "t" not in kill:                                  # the stream ended before 8 pieces: no kill, nothing to time
        kill["t"] = time.monotonic()
        rec("kill_not_fired", pieces_needed=8)
    s2, probes2 = probe_not_ready(port, fr, rec)
    S["restart1_ready_s"] = round(time.monotonic() - kill["t"], 1)
    ok["not_ready_codes_1"] = (probes2.get("openai", {}).get("code") == 503 and probes2.get("anthropic", {}).get("code") == 529
                               and all(p.get("should_retry") == "true" and p.get("retry_after") for p in probes2.values()))
    h2 = health(port)
    rec("worker2_ready", seconds_since_kill=S["restart1_ready_s"], probes=probes2, ok=ok["not_ready_codes_1"], health=h2)

    # 3. device error in worker 2 mid-stream (MASHBOX_FAULTS), worker 3 healthy (the fault fires in one life)
    pid2 = fr.link.proc.pid
    fr.env.pop("MASHBOX_FAULTS", None)
    code_f, raw_f = stream_and_cut(port, fr, a.tokens, None, rec, "device-error-stream.sse")
    t_f = time.monotonic()                               # the error event ends the stream when the worker fails
    t_fx, errs_f, fins_f = stream_text(raw_f)
    ok["device_error_stream"] = (code_f == 200 and bool(t_fx) and base.startswith(t_fx) and t_fx != base and len(errs_f) == 1
                                 and not fins_f)
    rec("device_error_stream", ok=ok["device_error_stream"], chars=len(t_fx), error=errs_f, finish=fins_f, pid=pid2)
    s3, probes3 = probe_not_ready(port, fr, rec)
    S["restart2_ready_s"] = round(time.monotonic() - t_f, 1)
    h3 = health(port)
    ok["not_ready_codes_2"] = (probes3.get("openai", {}).get("code") == 503 and probes3.get("anthropic", {}).get("code") == 529
                               and all(p.get("should_retry") == "true" and p.get("retry_after") for p in probes3.values()))
    rec("worker3_ready", seconds_since_error=S["restart2_ready_s"], probes=probes3, ok=ok["not_ready_codes_2"], health=h3)
    code, raw, _ = post(port, "/v1/chat/completions", body(a.tokens, False))
    after = whole_text(raw) if code == 200 else ""
    rec.save("after.json", raw)
    ok["same_answer_after_restarts"] = code == 200 and after == base
    ok["restarts_counted"] = h3.get("worker_restarts") == 2 and h3.get("worker_pid") not in (pid1, pid2, None)
    rec("after", ok=ok["same_answer_after_restarts"], restarts=h3.get("worker_restarts"), pid=h3.get("worker_pid"))
    h4 = health(port)
    ids = [(h.get("deployment_id"), h.get("component_set_sha")) for h in (h1, h2, h3, h4)]
    ok["health_from_worker"] = all(i[0] and i[1] for i in ids) and len(set(ids)) == 1
    car = [h.get("compiles_after_ready") for h in (h1, h2, h3, h4)]
    ok["compiles_never_hidden"] = all(isinstance(c, int) for c in car) and car == sorted(car)
    S["compiles_after_ready_by_step"] = car
    rec("health_fields", ok=ok["health_from_worker"] and ok["compiles_never_hidden"], ids=ids, compiles_after_ready=car,
        final=h4)
    down = []
    with open(os.path.join(a.out, "front.log")) as f:
        for ln in f:
            if '"worker_down"' in ln or '"fault_injected"' in ln or '"worker_fault"' in ln:
                down.append(ln.strip())
    rec("worker_lines", lines=down)
    with open(os.path.join(a.out, "front.log")) as f:
        ready_lines = [ln.strip() for ln in f if '"event": "ready"' in ln]
    ok["ready_line_once"] = len(ready_lines) == 1
    rec("ready_lines", lines=ready_lines)

    # 4. close: the worker exits, nothing holds the TPU
    pid3 = fr.link.proc.pid
    fr.close()
    time.sleep(2)
    hold = tpu_holders()
    ok["close_frees_tpu"] = not hold and fr.link is None
    try:
        os.kill(pid3, 0)
        alive = True
    except OSError:
        alive = False
    ok["worker_gone_after_close"] = not alive
    rec("closed", tpu_holders=hold, worker3_alive=alive)
    S.update(ok=ok, passed=all(ok.values()), code_sha=a.sha, model=a.model, n_fault=a.n_fault, tokens=a.tokens,
             utc_end=now())
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    rec("summary", **S)
    return 0 if S["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
