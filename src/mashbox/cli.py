"""mashbox command line.

  mashbox run qwen3-8-27b                   download (first time), load, serve on 127.0.0.1:8000
  mashbox run qwen3-8-27b --prompt "Hi"     one chat turn, printed
  mashbox run qwen3-8-27b --ui              the same server plus the chat page at http://127.0.0.1:8000/
  mashbox download qwen3-8-27b              download and check the files only
  mashbox run hy-nomas --ui                 world model in the browser on 127.0.0.1:8790 (TPU v6e-4)
  mashbox env                               versions and chip, for bug reports
  mashbox tpu up qwen3-8-27b                from your machine: a chip, install, server, tunnel (mashbox tpu -h)

A model with its own options (MODEL_CLI) gets `run` and `download` handed to its package; see `run <model> --help`.
"""
import argparse
import json
import os
import sys
import time


def log(s):
    print(s, file=sys.stderr, flush=True)


# Models whose runtime is in this version of the package. `run` and `download` for any other model in models.json
# stop with a clear message (exit 2) instead of starting a download it cannot use.
RUNNABLE = ("qwen3-8-27b", "hy-nomas")
# Models whose package parses its own `run` / `download` options: model id -> module with main(cmd, argv).
MODEL_CLI = {"hy-nomas": "mashbox.models.hy_nomas"}


def setup_cache(d=None):
    import jax
    d = d or os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "mashbox", "xla")
    os.makedirs(d, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", d)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return d


def cmd_env(a):
    import platform
    from . import kernels as K, __version__
    out = dict(mashbox=__version__, python=platform.python_version(), **K.versions())
    try:
        import jax
        d = jax.devices()[0]
        out.update(platform=d.platform, device_kind=d.device_kind, devices=len(jax.devices()), chip=K.chip_family())
    except Exception as e:
        out["devices"] = f"error: {e}"
    print(json.dumps(out, indent=1))


def files(a):
    from . import hub
    m = hub.model_entry(a.model)
    d = hub.upstream_dir(m, log)
    w = hub.weights_file(m, path=a.weights, log=log)
    return m, d, w


def cmd_download(a):
    m, d, w = files(a)
    from . import hub
    b, _ = hub.kernel_bundle(m, log) if not a.no_kernels else (None, None)
    dr = None
    if hub.draft_row(m):
        try:
            dr = hub.draft_file(m, log=log)
        except SystemExit:
            raise
        except Exception as e:          # `run --mtp K` then stops with this reason (see speculation())
            log(json.dumps(dict(event="draft", fetched=False, reason=f"{type(e).__name__}: {e}"[:300])))
    print(json.dumps(dict(model=m["id"], upstream=d, weights=w, kernels=b, draft=dr), indent=1))


def tensor_parallel(n):
    """Chips to split the model across. n > 0 is used as given; 0 uses the chips of this host, at most 4 (the model's 4
    KV heads), as MaxText / JetStream shard over every device of the host: 1 on a one-chip host (the single-chip path,
    with kernels), 4 on a v5e-4, v6e-4 or v5p-8. On a v6e-4, 4 chips decode faster than 1 with kernels and hold long
    contexts at 8 slots that one chip cannot."""
    import jax
    devs = jax.devices()
    if n:
        if n not in (1, 2, 4) or n > len(devs):
            raise SystemExit(f"--tensor-parallel-size {n}: use 1, 2 or 4, at most the {len(devs)} chips this host has")
        return n
    return next(k for k in (4, 2, 1) if k <= len(devs))


def speculation(a, m, cfg, tp):
    """Engine arguments for speculative decoding with the model's MTP head. Off by default (and with --mtp 0); --mtp K (1 to 3) turns
    it on with K drafts per verify pass, from the draft file models.json lists for the model (or --mtp-file)."""
    from . import hub
    k = getattr(a, "mtp", None)
    if k is not None and not 0 <= k <= 3:
        raise SystemExit(f"--mtp {k}: use 0 (off) to 3")
    mfile = getattr(a, "mtp_file", None)
    if not k:
        if mfile and k is None:
            raise SystemExit("--mtp-file: speculative decoding is off by default; add --mtp K (1 to 3) to use the file")
        log(json.dumps(dict(event="speculation", used=False, reason="--mtp 0" if k == 0 else "off by default")))
        return {}
    if hub.draft_row(m) is None and not mfile:
        raise SystemExit(f"--mtp {k}: {m['id']} has no draft file in models.json; give one with --mtp-file")
    if tp > 1:
        raise SystemExit(f"--mtp {k}: speculative decoding runs on one chip (no tensor parallel); use --mtp 0")
    from .models.qwen3_8_27b import spec as QSP
    try:
        path = hub.draft_file(m, path=mfile, log=log)
    except SystemExit:                  # a sha256 mismatch: never run on a file that is not the pinned one
        raise
    except Exception as e:              # not downloadable (not in the repo at this revision, offline)
        raise SystemExit(f"--mtp {k}: the draft file could not be fetched: {type(e).__name__}: {e}"[:400])
    Wm, _ = QSP.load_mtp(path, cfg, log=lambda i: log(json.dumps(dict(event="mtp_weights", **i))))
    log(json.dumps(dict(event="speculation", used=True, method="mtp", draft_k=k)))
    return dict(mtp=(QSP, Wm), draft_k=k)


def fit_check(a, m, d, tp):
    """Stop before any weights are downloaded, checked or loaded when the model cannot fit the chips it would run on.

    Per chip it needs at least its share of the weights (the size models.json gives for the file, or the size of the
    --weights file) plus its share of the context state the engine allocates for --ctx x --slots (the engine's own
    new_state, traced with jax.eval_shape: shapes only, nothing allocated), against the memory JAX reports for each chip
    (memory_stats bytes_limit). Nothing is added for working memory, so a setting that can fit is never refused. Without
    a memory figure (CPU) nothing is checked. -> the bytes per chip it needs, or None when not checked."""
    import math
    import types
    import jax
    import numpy as np
    from . import hub, kernels as K, tpu as T
    from .models import qwen3_8_27b as Q
    devs = jax.devices()
    try:
        have = min(int(x.memory_stats()["bytes_limit"]) for x in devs)
    except Exception:
        return None
    wb = os.path.getsize(a.weights) if a.weights else int(hub.weights_row(m)["bytes"])
    try:
        from .runtime.engine import Engine
        cfg = Q.Config(d)
        # the engine's slot state (Engine.__init__: one scratch slot for padding rows above 2 slots), shapes only
        st = jax.eval_shape(lambda: Q.new_state(cfg, a.slots - 1 if a.slots <= 2 else a.slots, a.ctx))
        sb = sum(math.prod(x.shape) * np.dtype(x.dtype).itemsize for x in jax.tree.leaves(st))
    except Exception:           # weights alone are still a lower bound
        sb = 0
    need = lambda k, s=sb: (wb + s) / k
    gb = lambda b: f"{b / 1e9:.2f} GB"
    log(json.dumps(dict(event="fit", tp=tp, need_gb_per_chip=round(need(tp) / 1e9, 2), chip_gb=round(have / 1e9, 2),
                        weights_gb=round(wb / 1e9, 2), context_gb=round(sb / 1e9, 2))))
    if need(tp) <= have:
        return need(tp)
    fam = K.chip_family() or devs[0].device_kind
    rows = [c for c, _ in T.offered(m)]
    chip = next((c for c in rows if c.startswith(f"{fam}-")), rows[0] if rows else None)
    up = f"mashbox tpu up {m['id']} --chip {chip} (from your own machine)" if chip else ""
    more = [k for k in (4, 2, 1) if tp < k <= len(devs) and need(k) <= have]
    if more:
        k = tensor_parallel(0) if tensor_parallel(0) in more else more[-1]
        use = f"leave out --tensor-parallel-size (it then uses {k})" if k == tensor_parallel(0) else f"run with --tensor-parallel-size {k}"
        fix = f"This host has {len(devs)} chips: {use}" + (f", or on a new chip: {up}" if up else "")
    elif need(max(k for k in (4, 2, 1) if k <= len(devs)), 0) <= have:
        fix = "Lower --ctx or --slots" + (f", or use a chip with more memory: {up}" if up else "")
    else:
        fix = f"It runs on a {chip}: {up}" if up else "No chip type in models.json fits it"
    raise SystemExit(f"{m['id']} does not fit on {tp} TPU {fam} chip{'s' if tp > 1 else ''}: it needs at least "
                     f"{gb(need(tp))} per chip (weights {gb(wb)} + context {gb(sb)} for --ctx {a.ctx} x --slots {a.slots}, "
                     f"split over {tp}) and each chip here has {gb(have)}. {fix}.")


def load(a):
    t0 = time.time()
    from . import hub
    m = hub.model_entry(a.model)
    d = hub.upstream_dir(m, log)                         # config and tokenizer only (small)
    cache = setup_cache(a.cache_dir)
    tp = tensor_parallel(getattr(a, "tensor_parallel_size", 0))
    fit_check(a, m, d, tp)                               # before the weights download, check or load
    w = hub.weights_file(m, path=a.weights, log=log)
    from . import kernels as K
    from .models import qwen3_8_27b as Q
    from .models.qwen3_8_27b import weights as QW
    from .runtime.engine import Engine
    from .runtime.chat import Chat
    hub.compile_cache(m, cache, log)
    t_files = time.time()
    dk = None
    if not a.no_kernels:
        bpath, row = hub.kernel_bundle(m, log)
        if bpath:
            for name, blob in K.cache_entries(bpath).items():
                p = os.path.join(cache, name)
                if not os.path.exists(p):
                    with open(p + ".part", "wb") as f:
                        f.write(blob)
                    os.replace(p + ".part", p)
            dk = K.decode_kernels(bpath, row, log)
    if dk is not None and a.slots != 8:
        log(json.dumps(dict(event="kernels", used=False, reason="the kernel bundle is built for --slots 8")))
        dk = None
    if dk is not None and tp > 1:
        log(json.dumps(dict(event="kernels", used=False, reason="the kernel bundle is built for one chip; tensor parallel runs plain JAX")))
        dk = None
    cfg = Q.Config(d)
    model = Q
    if tp > 1:
        from .models.qwen3_8_27b import shard as QS
        mesh = QS.mesh_for(tp)
        W, info = QS.load(w, cfg, mesh, log=lambda i: log(json.dumps(dict(event="weights", **i))))
        model = QS.TPModel(Q, mesh)
    else:
        W, info = QW.load(w, cfg, log=lambda i: log(json.dumps(dict(event="weights", **i))))
    spec = speculation(a, m, cfg, tp)
    eng = Engine(cfg, W, model, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=dk, **spec,
                 prefix_cache=a.prompt is None and not a.no_prefix_cache,
                 cache_ram=max(0, a.cache_ram) << 20, ckpts=a.ctx_checkpoints, ckpt_step=a.checkpoint_min_step,
                 attn_bytes=2e9 if a.ctx <= 8192 else 5e8)       # long context: smaller prefill chunks, less HBM
    chat = Chat(d)
    log(json.dumps(dict(event="loaded", files_s=round(t_files - t0, 1), weights_s=info["load_s"], gb=round(info["device_bytes"] / 1e9, 2),
                        bpw=info["bpw"], kernels=eng.kernels)))
    return m, eng, chat, info, t0


def warm(eng, chat):
    """Compile every program a request can reach before ready (served from the compile cache when it has them): two short
    requests for the small helper programs, then each prefill and decode program (Engine.warm)."""
    t = time.perf_counter()
    ids = chat.chat_ids([dict(role="user", content="Hello")])
    eng.generate([ids], max_new=eng.K + 1, stop_ids=())
    eng.generate([ids], max_new=eng.K + 1, stop_ids=(), samp=[dict(temperature=0.7, top_p=0.9, seed=1)])
    w = eng.warm(workers=min(8, os.cpu_count() or 1))
    log(json.dumps(dict(event="warm", prefill_programs=w["prefill"], decode_programs=w["decode"], compile_s=w["compile_s"],
                        run_s=w["run_s"], compile_failed=w["compile_failed"], seconds=round(time.perf_counter() - t, 1))))


def cmd_run(a):
    keys = api_keys(a) if a.prompt is None else []       # before the load: a bad key file fails at once
    m, eng, chat, info, t0 = load(a)
    if a.prompt is not None:
        ids = chat.chat_ids([dict(role="user", content=a.prompt)], enable_thinking=a.thinking)
        samp = None if a.temperature == 0 else [dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed)]
        r = eng.generate([ids], max_new=a.max_tokens, stop_ids=chat.stop_ids, samp=samp)
        print(chat.decode(r["ids"][0]))
        t = r["timing"]
        log(json.dumps(dict(event="done", new_tokens=t["new_tokens"], prefill_s=round(t["prefill_s"], 3), decode_s=round(t["decode_s"], 3))))
        return
    from .runtime.server import Server, serve
    srv = Server(eng, chat, m["id"], dict(weights=info["file"], gb=round(info["device_bytes"] / 1e9, 2), bpw=info["bpw"],
                                          prefix_cache=eng.prefix_cache, slots=eng.S, cache_ram_mib=a.cache_ram), log=log)
    if not a.no_warm:
        warm(eng, chat)
    srv.ready = True
    log(json.dumps(dict(event="ready", seconds_since_start=round(time.time() - t0, 1))))
    if a.ui:
        log(json.dumps(dict(event="ui", url=f"http://{a.host}:{a.port}/")))
    serve(srv, a.host, a.port, log, ui=a.ui, keys=keys)


KEY_FILE = os.path.join(os.path.expanduser("~"), ".config", "mashbox", "api-key")


def api_keys(a):
    """The server's API keys: --api-key-file (one per line, as llama-server), else MASHBOX_API_KEY (comma-separated),
    else none on a loopback --host; on any other --host one is made (once) in ~/.config/mashbox/api-key, mode 0600, and
    its path is printed, never the key."""
    from .runtime import guard
    if a.api_key_file:
        with open(a.api_key_file) as f:
            keys = [x.strip() for x in f.read().splitlines() if x.strip()]
        if not keys:
            raise SystemExit(f"mashbox: {a.api_key_file} holds no key")
        src = a.api_key_file
    elif os.environ.get("MASHBOX_API_KEY", "").strip():
        keys, src = [x.strip() for x in os.environ["MASHBOX_API_KEY"].split(",") if x.strip()], "MASHBOX_API_KEY"
    elif guard.loopback(a.host):
        return []
    else:
        if not os.path.exists(KEY_FILE):
            os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
            fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(guard.new_token() + "\n")
        with open(KEY_FILE) as f:
            keys, src = [x.strip() for x in f.read().splitlines() if x.strip()], KEY_FILE
    if not keys:                                                        # never start open because a key source is empty
        raise SystemExit(f"mashbox: {src} holds no key"
                         + (" (delete the file and start again: a new key is made)" if src == KEY_FILE else ""))
    log(json.dumps(dict(event="api_key", keys=len(keys), source=src,
                        use="send Authorization: Bearer <key> (or x-api-key: <key>) on /v1 requests")))
    return keys


def build_parser():
    """The `mashbox` parser (env, tpu, run, download). docs/cli.md and the model pages' option tables are generated from it
    by tools/hf_github_sync_check.py, so the help text here is the documentation."""
    ap = argparse.ArgumentParser(prog="mashbox", description="mashbox by MASHBOX Labs: open models that run faster and cheaper.")
    from . import __version__ as v
    ap.add_argument("--version", action="version", version=f"mashbox {v}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="versions and chip, for bug reports")
    # `mashbox tpu ...` is parsed by mashbox.tpu (main() hands it over before this parser runs); listed here for --help
    sub.add_parser("tpu", add_help=False, help="Cloud TPU chips from this machine, with your own gcloud: "
                   "up, down, status, ssh, tunnel (mashbox tpu --help)")
    for name in ("run", "download"):
        p = sub.add_parser(name, help="download (first time), load and serve a model, or answer one prompt" if name == "run"
                           else "download and check a model's files without loading them")
        p.add_argument("model", help="model id, as in models.json (e.g. qwen3-8-27b)")
        p.add_argument("--weights", metavar="FILE", help="a local .bshj file instead of the download (its sha256 is still checked)")
        p.add_argument("--no-kernels", action="store_true", help="plain JAX only (slower decode, same outputs within rounding)")
        if name == "run":
            p.add_argument("--prompt", metavar="TEXT", help="answer one chat turn, print it and exit; without it, start the server")
            p.add_argument("--thinking", action="store_true", help="thinking mode for --prompt")
            p.add_argument("--max-tokens", type=int, default=512, metavar="N", help="most new tokens for --prompt")
            p.add_argument("--temperature", type=float, default=0.0, metavar="T", help="sampling temperature for --prompt; 0 is greedy")
            p.add_argument("--top-p", type=float, default=1.0, metavar="P", help="nucleus sampling cut for --prompt")
            p.add_argument("--top-k", type=int, default=0, metavar="K", help="top-k cut for --prompt; 0 means none")
            p.add_argument("--seed", type=int, default=0, metavar="N", help="seed for sampled --prompt answers")
            p.add_argument("--host", default="127.0.0.1", metavar="H", help="server address; keep it on 127.0.0.1 and use an SSH tunnel")
            p.add_argument("--port", type=int, default=8000, metavar="N", help="server port")
            p.add_argument("--api-key-file", metavar="FILE",
                           help="require an API key from FILE (one per line; each key has its own prompt cache) on /v1 "
                                "requests; also MASHBOX_API_KEY=k1,k2. Default: none on 127.0.0.1, and on any other --host "
                                "a key made in ~/.config/mashbox/api-key")
            p.add_argument("--ui", action="store_true", help="also serve the MASHBOX Chat page at http://127.0.0.1:<port>/")
            p.add_argument("--ctx", type=int, default=2048, metavar="N", help="context per request in tokens (prompt plus answer)")
            p.add_argument("--slots", type=int, default=8, metavar="N",
                           help="requests decoded together; the kernel bundle is built for 8, other values run plain JAX")
            p.add_argument("--steps", type=int, default=8, metavar="N", help="decode steps per device call")
            p.add_argument("--mtp", type=int, metavar="K",
                           help=argparse.SUPPRESS)
            p.add_argument("--mtp-file", metavar="FILE",
                           help=argparse.SUPPRESS)
            p.add_argument("--tensor-parallel-size", "-tp", type=int, default=0, metavar="N",
                           help="chips to split the model across, as vLLM (1, 2 or 4); default: the chips of this host, at most 4 "
                                "(1 on a TPU v6e-1, 4 on a TPU v5e-4 or v6e-4)")
            p.add_argument("--cache-dir", metavar="DIR", help="compile cache (default ~/.cache/mashbox/xla, or JAX_COMPILATION_CACHE_DIR)")
            p.add_argument("--no-warm", action="store_true", help="skip compiling the first request's programs before ready")
            p.add_argument("--no-prefix-cache", action="store_true",
                           help="prefill every prompt in full (by default the server reuses a cached prompt start, see docs/agents-on-tpu.md)")
            p.add_argument("--cache-ram", type=int, default=8192, metavar="MIB",
                           help="host memory for saved prompt states, as llama-server's --cache-ram: a conversation that loses its "
                                "slot is kept here and restored when it comes back (default 8192, 0 = off)")
            p.add_argument("--ctx-checkpoints", type=int, default=8, metavar="N",
                           help="prompt points kept per slot where a cached prompt can resume (default 8)")
            p.add_argument("--checkpoint-min-step", type=int, default=8192, metavar="N",
                           help="minimum tokens between kept user-message points, as llama-server (default 8192)")
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["tpu"]:
        from . import tpu
        return tpu.main(argv[1:])
    if len(argv) >= 2 and argv[0] in ("run", "download") and argv[1].replace(".", "-").lower() in MODEL_CLI:
        import importlib
        return importlib.import_module(MODEL_CLI[argv[1].replace(".", "-").lower()]).main(argv[0], argv[2:])
    a = build_parser().parse_args(argv)
    if a.cmd == "env":
        return cmd_env(a)
    from . import hub, __version__
    mid = hub.model_entry(a.model)["id"]
    if mid not in RUNNABLE:
        log(f"{mid} is not in mashbox {__version__}. Models in this version: https://mashbox.gg/docs/models")
        return 2
    if a.cmd == "download":
        return cmd_download(a)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        log("note: serving on a non-loopback address exposes the server to your network (an API key is required); "
            "prefer an SSH tunnel")
    return cmd_run(a)
