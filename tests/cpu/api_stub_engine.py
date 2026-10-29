"""Scripted engine with the continuous-batching interface of runtime/engine.py (check / admit / step / cancel / warm /
free_slots / busy), for the server, front and worker tests. Not a test file (no test_ prefix).

Each admitted row answers with the next scripted token list (outs, in turn), one token per step; it ends with "stop"
when its script runs out before max_new (as an end-of-text token would) and with "length" at max_new. With log=PATH every
admit, cancel and end is appended to PATH as one JSON line (the tests read what happened inside a worker process).
make(**kwargs) is the worker factory: python -m mashbox.runtime.worker --factory api_stub_engine:make. make() also takes
the fields a real engine carries for /health and /v1/models (deployment_id, component_set_sha, cache_ns, serve_info), and
compile_each_admit=True compiles one new XLA program per admitted row (logged as "compiled" with the process's count of
compiles after ready), so tests can see where /health's compile count comes from.
"""
import json
import os
import threading
import time


class StubEngine:
    K, DK = 1, 0

    def __init__(self, S=2, CTX=4096, V=1000, outs=None, step_s=0.0, prefix_cache=False, log=None, compile_each_admit=False,
                 first_free_slots_s=0.0, hold=None):
        self.S, self.CTX, self.V, self.step_s, self.prefix_cache = S, CTX, V, step_s, prefix_cache
        self.compile_each_admit = compile_each_admit
        self.first_free_slots_s = first_free_slots_s       # the first free_slots() call takes this long (a slow first read)
        self.outs = [list(o) for o in (outs or [[7] * 64])]
        self.rows, self.n_admitted, self.log_path = [None] * S, 0, log
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.gate.set()
        # hold=dict(after=n, path=P, then="go"|"raise"): a row with n tokens gets no more (each step still takes
        # step_s and returns) until the file P exists, so a test can set up what it needs first. With then="raise" the
        # first step that sees P renames it to P.fired and raises a device fault (the error faults.py's raise kind
        # raises), once over all worker processes; while P.fired exists the hold is open.
        self.hold = dict(hold) if hold else None

    def _held(self):
        """True while rows past hold['after'] tokens wait for the hold file; raises the held fault when it is due."""
        h = self.hold
        if not h:
            return False
        p = h["path"]
        if os.path.exists(p):
            if h.get("then") == "raise":
                try:
                    os.rename(p, p + ".fired")
                except FileNotFoundError:                # another worker process fired it first
                    return False
                from mashbox.runtime.faults import InjectedFault
                self._log(ev="fault", kind="raise")
                raise InjectedFault("injected device fault at the held step")
            return False
        return not os.path.exists(p + ".fired")

    def _log(self, **e):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(dict(e, pid=os.getpid(), t=time.time())) + "\n")

    def warm(self, workers=8):
        self._log(ev="warm")

    def check(self, q):
        ids = q.get("ids")
        if not ids or any((not isinstance(t, int)) or t < 0 or t >= self.V for t in ids):
            return f"token ids must be 0..{self.V - 1}"
        if len(ids) + int(q["max_new"]) > self.CTX:
            return f"prompt ({len(ids)} tokens) + max_new ({q['max_new']}) is over the context of {self.CTX}"
        return None

    def free_slots(self):
        if self.first_free_slots_s:
            s, self.first_free_slots_s = self.first_free_slots_s, 0.0
            time.sleep(s)
        return sum(r is None for r in self.rows)

    def busy(self):
        return any(r is not None for r in self.rows)

    def admit(self, reqs, stop_ids=()):
        out = []
        for q in reqs:
            now = time.perf_counter()
            why = self.check(q)
            if why is not None:
                out.append(dict(slot=None, ids=[], fin="error", error=why, cached=0, restored=0, sampled=False,
                                t_admit=now, t_first=None, t_end=now, calls=0))
                continue
            s = self.rows.index(None)
            script = self.outs[self.n_admitted % len(self.outs)]
            self.n_admitted += 1
            r = dict(slot=s, ids=[], fin=None, cached=0, restored=0, sampled=q.get("samp") is not None, t_admit=now,
                     t_first=None, t_end=None, calls=0, max_new=q["max_new"], script=script, prompt=list(q["ids"]))
            self.rows[s] = r
            self._log(ev="admit", slot=s, prompt=len(q["ids"]), scope=q.get("scope"), variant=q.get("variant"))
            if self.compile_each_admit:
                self._compile()
            out.append(r)
        return out

    def _compile(self):
        """One new XLA program (a new function, so never a cache hit), then this process's compiles after ready."""
        import jax
        import numpy as np
        from mashbox.runtime import programs
        jax.jit(lambda x: x * 2 + self.n_admitted)(np.zeros(self.n_admitted + 1, np.float32)).block_until_ready()
        self._log(ev="compiled", after=programs.counts()["compiles_after_ready"])

    def _end(self, r, fin):
        r["fin"], r["t_end"] = fin, time.perf_counter()
        self.rows[r["slot"]] = None
        self._log(ev="end", slot=r["slot"], fin=fin, n=len(r["ids"]))

    def cancel(self, rows):
        out = []
        for r in rows or ():
            if r is not None and r["fin"] is None and self.rows[r["slot"]] is r:
                self._end(r, "cancel")
                out.append(r)
        return out

    def step(self, waiting=False):
        self.gate.wait()
        if self.step_s:
            time.sleep(self.step_s)
        held = self._held()
        ended = []
        for r in [r for r in self.rows if r is not None]:
            if held and len(r["ids"]) >= self.hold["after"]:
                continue                                 # held: this row gets no token this step
            r["calls"] += 1
            if len(r["ids"]) >= len(r["script"]):
                self._end(r, "stop")
                ended.append(r)
                continue
            r["ids"].append(r["script"][len(r["ids"])])
            if r["t_first"] is None:
                r["t_first"] = time.perf_counter()
            if len(r["ids"]) >= r["max_new"]:
                self._end(r, "length")
                ended.append(r)
        return ended


def make(deployment_id=None, component_set_sha=None, cache_ns=None, serve_info=None, life_dir=None, slow_from_life=None,
         slow_s=0.0, **kw):
    """The stand-in engine. life_dir counts worker lives (one file each); from life `slow_from_life` on, the engine takes
    `slow_s` seconds to load (a restart that is slow to get ready)."""
    if life_dir:
        os.makedirs(life_dir, exist_ok=True)
        n = len(os.listdir(life_dir)) + 1
        open(os.path.join(life_dir, f"life-{n}-{os.getpid()}"), "w").close()
        if slow_from_life and n >= slow_from_life:
            time.sleep(slow_s)
    e = StubEngine(**kw)
    for k, v in dict(deployment_id=deployment_id, component_set_sha=component_set_sha, cache_ns=cache_ns,
                     serve_info=serve_info).items():
        if v is not None:
            setattr(e, k, v)
    return e


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def serve_on_free_port(srv, tries=5, wait_s=60):
    """Start server.serve(srv) on a free loopback port in a daemon thread and return the port once serve() says it is
    listening (its log line comes after the bind). A port that another process took between the pick and the bind ends
    that serve thread; then a new port is tried."""
    import socket
    from mashbox.runtime.server import serve
    errs = []
    for _ in range(tries):
        with socket.socket() as s_:
            s_.bind(("127.0.0.1", 0))
            port = s_.getsockname()[1]
        up = threading.Event()

        def run(port=port, up=up):
            try:
                serve(srv, "127.0.0.1", port, lambda m: up.set() if '"listening"' in str(m) else None)
            except Exception as e:                # noqa: BLE001 - reported below if every port fails
                errs.append(f"{type(e).__name__}: {e}")
        th = threading.Thread(target=run, daemon=True)
        th.start()
        t = time.time()
        while not up.is_set() and th.is_alive() and time.time() - t < wait_s:
            up.wait(0.05)
        if up.is_set():
            return port
    raise RuntimeError(f"serve() did not start on {tries} ports: {errs}")


# ---- the real engine on a tiny random Qwen3.8 model (the architecture of qwen3_8_27b_checks.py's tiny model) --------
class IdChat:
    """Token ids as text ('12 7 301 '): the server path without a tokenizer, for engines with a small vocabulary."""
    stop_ids = ()

    def encode(self, text):
        return [int(x) for x in str(text).split() if x.isdigit()]

    def decode(self, ids):
        return "".join(f"{int(t)} " for t in ids)

    def chat_ids(self, msgs, *a, **k):
        c = msgs[-1].get("content")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        return self.encode(c)


def build_tiny(wd, seed=0):
    """Write wd/tiny (config.json) and wd/tiny-bf16.bshj: a tiny random Qwen3.8 (vocabulary 512, 8 layers, hidden 64),
    built as qwen3_8_27b_checks.py builds its tiny model. Needs torch and transformers with qwen3_5."""
    import glob
    import struct

    import ml_dtypes
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    md = os.path.join(wd, "tiny")
    cfg = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4, rms_norm_eps=1e-6,
        rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25, mrope_section=[2, 1, 1],
                             mrope_interleaved=True),
        tie_word_embeddings=False, attn_output_gate=True, pad_token_id=0)
    torch.manual_seed(seed)
    m = Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif "dt_bias" in n:
                p.normal_(0, 1)
            elif p.ndim == 1:
                p.normal_(0, 0.3)
            elif "conv1d" in n:
                p.normal_(0, 0.4)
            else:
                p.normal_(0, 0.08)
    m.to(torch.bfloat16).eval().save_pretrained(md)
    T = {}
    for f in sorted(glob.glob(os.path.join(md, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                t = s.get_tensor(k)
                T[k] = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.float().numpy()
    align, pos, ents, bufs = 4096, 0, {}, []          # .bshj container (layout documented in mashbox/bshj.py)
    for name, a in T.items():
        a = np.ascontiguousarray(a)
        pos = -(-pos // align) * align
        ents[name] = dict(dtype={"bfloat16": "bf16", "float32": "f32"}[a.dtype.name], shape=list(a.shape), offset=pos,
                          nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    raw = json.dumps(dict(format="bshj", version="1.0", data_len=pos, tensors=ents), separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(os.path.join(wd, "tiny-bf16.bshj"), "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p, b in bufs:
            f.seek(off + p)
            f.write(b)
        f.truncate(off + pos)
    return wd


def tiny(wd, S=2, CTX=512, K=4, deployment_id=None, component_set_sha=None):
    """Worker factory: the real engine (runtime/engine.py) on the tiny model build_tiny(wd) wrote. deployment_id and
    component_set_sha stand in for the ones cli.load computes (a dry run of a check that reads them from /health)."""
    from mashbox.models import qwen3_8_27b as Q
    from mashbox.models.qwen3_8_27b import weights as QW
    from mashbox.runtime.engine import Engine
    cfg = Q.Config(os.path.join(wd, "tiny"))
    W, _ = QW.load(os.path.join(wd, "tiny-bf16.bshj"), cfg)
    e = Engine(cfg, W, Q, CTX=CTX, S=S, K=K)
    if deployment_id is not None:
        e.deployment_id, e.component_set_sha = deployment_id, component_set_sha
    return e
