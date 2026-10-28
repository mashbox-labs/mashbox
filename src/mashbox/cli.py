"""mashbox command line.

  python3 -m mashbox run qwen3-8-27b                   download (first time), load, serve on 127.0.0.1:8000
  python3 -m mashbox run qwen3-8-27b --prompt "Hi"     one chat turn, printed
  python3 -m mashbox download qwen3-8-27b              download and check the files only
  python3 -m mashbox run hy-nomas --ui                 world model in the browser on 127.0.0.1:8790 (TPU v6e-4)
  python3 -m mashbox env                               versions and chip, for bug reports
  mashbox tpu up qwen3-8-27b                           from your machine: a chip, install, server, tunnel (mashbox tpu -h)

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
    print(json.dumps(dict(model=m["id"], upstream=d, weights=w, kernels=b), indent=1))


def load(a):
    t0 = time.time()
    m, d, w = files(a)
    cache = setup_cache(a.cache_dir)
    from . import hub, kernels as K
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
    cfg = Q.Config(d)
    W, info = QW.load(w, cfg, log=lambda i: log(json.dumps(dict(event="weights", **i))))
    eng = Engine(cfg, W, Q, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=dk,
                 prefix_cache=a.prompt is None and not a.no_prefix_cache,
                 attn_bytes=2e9 if a.ctx <= 8192 else 5e8)       # long context: smaller prefill chunks, less HBM
    chat = Chat(d)
    log(json.dumps(dict(event="loaded", files_s=round(t_files - t0, 1), weights_s=info["load_s"], gb=round(info["device_bytes"] / 1e9, 2),
                        bpw=info["bpw"], kernels=eng.kernels)))
    return m, eng, chat, info, t0


def warm(eng, chat):
    """Compile the programs a first request needs (served from the compile cache when it has them)."""
    ids = chat.chat_ids([dict(role="user", content="Hello")])
    eng.generate([ids], max_new=eng.K + 1, stop_ids=())
    eng.generate([ids], max_new=eng.K + 1, stop_ids=(), samp=[dict(temperature=0.7, top_p=0.9, seed=1)])


def cmd_run(a):
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
                                          prefix_cache=eng.prefix_cache, slots=eng.S), log=log)
    if not a.no_warm:
        warm(eng, chat)
    srv.ready = True
    log(json.dumps(dict(event="ready", seconds_since_start=round(time.time() - t0, 1))))
    serve(srv, a.host, a.port, log)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["tpu"]:
        from . import tpu
        return tpu.main(argv[1:])
    if len(argv) >= 2 and argv[0] in ("run", "download") and argv[1].replace(".", "-").lower() in MODEL_CLI:
        import importlib
        return importlib.import_module(MODEL_CLI[argv[1].replace(".", "-").lower()]).main(argv[0], argv[2:])
    ap = argparse.ArgumentParser(prog="mashbox", description="mashbox by MASHBOX Labs: open models that run faster and cheaper.")
    from . import __version__ as v
    ap.add_argument("--version", action="version", version=f"mashbox {v}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    for name in ("run", "download"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--weights", help="a local .bshj file instead of the download (its sha256 is still checked)")
        p.add_argument("--no-kernels", action="store_true", help="plain JAX only (slower decode, same outputs within rounding)")
        if name == "run":
            p.add_argument("--prompt")
            p.add_argument("--thinking", action="store_true")
            p.add_argument("--max-tokens", type=int, default=512)
            p.add_argument("--temperature", type=float, default=0.0)
            p.add_argument("--top-p", type=float, default=1.0)
            p.add_argument("--top-k", type=int, default=0)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--ctx", type=int, default=2048)
            p.add_argument("--slots", type=int, default=8)
            p.add_argument("--steps", type=int, default=8, help="decode steps per device call")
            p.add_argument("--cache-dir")
            p.add_argument("--no-warm", action="store_true")
            p.add_argument("--no-prefix-cache", action="store_true",
                           help="prefill every prompt in full (by default the server reuses a cached prompt start, see docs/agents-on-tpu.md)")
    a = ap.parse_args(argv)
    if a.cmd == "env":
        return cmd_env(a)
    from . import hub, __version__
    mid = hub.model_entry(a.model)["id"]
    if mid not in RUNNABLE:
        log(f"{mid}: its runtime is not in mashbox {__version__} yet. Status and steps: https://mashbox.gg/docs/models/{mid}")
        return 2
    if a.cmd == "download":
        return cmd_download(a)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        log("note: serving on a non-loopback address exposes the server to your network; prefer an SSH tunnel")
    return cmd_run(a)
