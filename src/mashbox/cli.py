"""mashbox command line.

  mashbox run qwen3-8-27b                   download (first time), load, serve on 127.0.0.1:8000
  mashbox run unsloth/Qwen3.8-27B-GGUF      a GGUF repo (llama.cpp's file pick: Q4_K_M, then Q8_0) or a .gguf path
  mashbox inspect unsloth/Qwen3.8-27B-GGUF  what a GGUF file holds and how it would run, without loading it
  mashbox run qwen3-8-27b --prompt "Hi"     one chat turn, printed
  mashbox run qwen3-8-27b --ui              the same server plus the chat page at http://127.0.0.1:8000/
  mashbox download qwen3-8-27b              download and check the files only
  mashbox models                            the models this version runs, by publisher, then model, then file
  mashbox run hy-nomas --ui                 world model in the browser on 127.0.0.1:8790 (TPU v6e-4)
  mashbox run track2rig TRACKS.npz -o OUT   rig from 3D point tracks: OUT/rig.json and OUT/rig.glb (CPU)
  mashbox env                               versions and chip, for bug reports
  mashbox tpu up qwen3-8-27b                from your machine: a chip, install, server, tunnel (mashbox tpu -h)

A model with its own options (MODEL_CLI) gets `run` and `download` handed to its package; see `run <model> --help`.
"""
import argparse
import json
import os
import sys
import time

from . import hub


def log(s):
    print(s, file=sys.stderr, flush=True)


def catalog_runtimes(reg):
    """(RUNNABLE, MODEL_CLI) read from the catalog, so no model id is typed into this file. A models.json row is runnable
    when the package its path names (src/mashbox/models/<package>) ships in this version of mashbox; that package parses
    its own `run` / `download` options when its __init__ defines main(cmd, argv) (read from the source, nothing is
    imported)."""
    import ast
    import importlib.util
    import pathlib
    runnable, own = [], {}
    for m in reg.get("models", []):
        parts = pathlib.PurePosixPath(m.get("path") or "").parts
        parts = parts[1:] if parts[:1] == ("src",) else parts
        if len(parts) != 3 or parts[:2] != (__package__, "models"):
            continue
        mod = ".".join(parts)
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            continue
        runnable.append(m["id"])
        body = ast.parse(pathlib.Path(spec.origin).read_text(encoding="utf-8")).body
        if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in body):
            own[m["id"]] = mod
    return tuple(runnable), own


# Models whose runtime is in this version of the package. `run` and `download` for any other model in models.json
# stop with a clear message (exit 2) instead of starting a download it cannot use. MODEL_CLI: the models whose package
# parses its own `run` / `download` options, model id -> module with main(cmd, argv).
RUNNABLE, MODEL_CLI = catalog_runtimes(hub.registry())


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
    from .runtime import drafter as DR
    dr, dl = None, DR.declared(m)
    if dl and any((g or {}).get("status") == "passed" for g in (dl.get("gate") or {}).values()):
        try:                            # only a drafter that passed its gate on some chip turns on without a flag
            dr = hub.draft_file(m, log=log)
        except SystemExit:
            raise
        except Exception as e:          # `run` then serves without it, with this reason (see speculation())
            log(json.dumps(dict(event="draft", fetched=False, reason=f"{type(e).__name__}: {e}"[:300])))
    elif hub.draft_row(m):
        log(json.dumps(dict(event="draft", fetched=False, reason="speculation is not gated on any chip; `run --mtp K` fetches it")))
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


def speculation(a, m, cfg, tp, component_set_sha=None, later=False):
    """Engine arguments for speculative decoding with the model's drafter (models.json `drafter`, INTERFACES-v1 §10).

    No flag: on only where the drafter passed its gate on this chip for this deployment (gate[chip] "passed" with the
    component_set_sha this deployment has with the drafter on; component_set_sha(drafter key) computes it, None = not
    known), with k_default drafts; the draft file is fetched only then. Otherwise off, with one log line saying why.
    --mtp 0: off. --mtp K (1 to k_max): on, from the draft file models.json lists (or --mtp-file); where the gate has not
    passed it says "speculation: not gated on this chip", and such a deployment is never a certified one. Draft length
    per row bucket comes from the drafter's measured k_table for this chip (runtime/drafter.k_table), else k.

    later=True (cli.load, where the deployment's component_set_sha is known only once the engine is built): with no flag
    the gate's own recorded component_set_sha stands in, and the result carries it as gate_css; deployment_keys() then
    turns speculation off before the warm-up when the built engine's component_set_sha is another one."""
    from . import hub
    from .runtime import drafter as DR
    dr = DR.declared(m)
    kmax = dr["k_max"] if dr else 3
    k = getattr(a, "mtp", None)
    if k is not None and not 0 <= k <= kmax:
        raise SystemExit(f"--mtp {k}: use 0 (off) to {kmax}")
    mfile = getattr(a, "mtp_file", None)

    def off(why):
        log(json.dumps(dict(event="speculation", used=False, reason=why)))
        return {}
    if k == 0:
        return off("--mtp 0")
    if k is None and mfile:
        raise SystemExit(f"--mtp-file: give --mtp K (1 to {kmax}) with it")
    chip = DR.chip_key()
    css = component_set_sha(DR.key(dr)) if (component_set_sha and dr) else None
    gcss = DR.gated_css(dr, chip) if (later and k is None and css is None) else None
    css = gcss if gcss is not None else css
    gated, why = DR.gate(dr, chip, css)
    from .models.qwen3_8_27b import spec as QSP
    try:
        kt = DR.k_table(dr, chip)
    except ValueError as e:
        if k is None:
            return off(str(e))
        raise SystemExit(f"--mtp {k}: {e}")
    if k is None:
        if not gated:
            return off(why)
        if tp > 1:
            return off("speculation runs on one chip (no tensor parallel)")
        k = dr["k_default"]
        try:
            path = hub.draft_file(m, log=log)
        except (SystemExit, Exception) as e:     # the base model serves as without a drafter
            return off(f"the draft file could not be fetched or checked: {type(e).__name__}: {e}"[:300])
    else:
        if hub.draft_row(m) is None and not mfile:
            raise SystemExit(f"--mtp {k}: {m['id']} has no draft file in models.json; give one with --mtp-file")
        if tp > 1:
            raise SystemExit(f"--mtp {k}: speculative decoding runs on one chip (no tensor parallel); use --mtp 0")
        try:
            path = hub.draft_file(m, path=mfile, log=log)
        except SystemExit:                  # a sha256 mismatch: never run on a file that is not the pinned one
            raise
        except Exception as e:              # not downloadable (not in the repo at this revision, offline)
            raise SystemExit(f"--mtp {k}: the draft file could not be fetched: {type(e).__name__}: {e}"[:400])
        if not gated:
            log(f"speculation: {why}" if why.startswith("not gated on this chip") else f"speculation: not gated on this chip ({why})")
    Wm, _ = QSP.load_mtp(path, cfg, log=lambda i: log(json.dumps(dict(event="mtp_weights", **i))))
    log(json.dumps(dict(event="speculation", used=True, method="mtp", draft_k=k, k_table=kt, gated=gated, reason=why)))
    return dict(mtp=(QSP, Wm), draft_k=k, k_table=kt, drafter=drafter_identity(m, dr, path, k, kmax, mfile),
                **({"gate_css": gcss} if gcss is not None else {}))


def drafter_identity(m, dr, path, k, kmax, mfile=None):
    """The drafter's part of the deployment's component set (runtime/drafter.key: kind, id@version, weights_sha256, k
    rule; INTERFACES-v1 section 6) for the draft file this start loaded (path), bound to its bytes: weights_sha256 is the
    pinned sha256 hub.draft_file checked the file against (every download; a --mtp-file unless MASHBOX_VERIFY=0), else
    the sha256 of the file's own bytes. A drafter models.json does not declare keys as kind "mtp", the file's name and
    this k."""
    from . import hub
    from .runtime import drafter as DR, tenancy
    checked = (hub.draft_row(m) or {}).get("sha256") if not mfile or os.environ.get("MASHBOX_VERIFY", "1") != "0" else None
    base = dr or dict(kind="mtp", id=os.path.basename(path), version=0, k_default=k, k_max=kmax)
    return DR.key(dict(base, sha256=checked or tenancy.file_sha256(path)))


def deployment_keys(eng, m, config_sha, layout_sha, kernel_bundle_sha=None, tp=1, gate_css=None, mesh=None, drafter=None):
    """Set eng.deployment_id and eng.component_set_sha (runtime/compile_key.py: with the chip's topology, tensor parallel
    and its mesh). While speculation is on, the drafter's identity is part of the component set: `drafter` is the one
    speculation() returned for the file it loaded (drafter_identity; None = the model's declared drafter, drafter.key).
    When speculation came from the drafter's gate (gate_css, speculation(later=True)) and this deployment's
    component_set_sha is another one, speculation is turned off before the warm-up (one log line) and the keys are made
    again without the drafter. -> the compile key."""
    from .runtime import compile_key as CK, drafter as DR

    def keys():
        dk = (drafter if drafter is not None else DR.key(DR.declared(m))) if getattr(eng, "DK", 0) else None
        return CK.for_engine(eng, m["id"], config_sha, layout_sha, kernel_bundle_sha=kernel_bundle_sha, tp=tp, drafter=dk,
                             mesh=mesh)
    key, eng.deployment_id, eng.component_set_sha = keys()
    if gate_css is not None and getattr(eng, "DK", 0) and eng.component_set_sha != gate_css:
        log(json.dumps(dict(event="speculation", used=False, reason=f"gated on {DR.chip_key()} for another component set")))
        eng.drop_speculation()
        key, eng.deployment_id, eng.component_set_sha = keys()
    return key


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
    dk = bpath = mesh = None
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
    spec = speculation(a, m, cfg, tp, later=True)
    gate_css, drafter = spec.pop("gate_css", None), spec.pop("drafter", None)
    eng = Engine(cfg, W, model, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=dk, **spec,
                 prefix_cache=a.prompt is None and not a.no_prefix_cache,
                 cache_ram=max(0, a.cache_ram) << 20, ckpts=a.ctx_checkpoints, ckpt_step=a.checkpoint_min_step,
                 attn_bytes=2e9 if a.ctx <= 8192 else 5e8,       # long context: smaller prefill chunks, less HBM
                 trace=os.environ.get("MASHBOX_TRACE") == "1")   # per-token logit hashes for checks (log event "trace")
    chat = Chat(d)
    from .runtime import tenancy, compile_key as CK
    # the deployment's compile key and component set (runtime/compile_key.py): /health, the startup log, the namespace
    key = deployment_keys(eng, m, tenancy.file_sha256(os.path.join(d, "config.json")), CK.weights_layout(W),
                          kernel_bundle_sha=tenancy.file_sha256(bpath) if dk is not None else None, tp=tp, gate_css=gate_css,
                          mesh=mesh, drafter=drafter)
    log(json.dumps(dict(event="deployment", deployment_id=eng.deployment_id, component_set_sha=eng.component_set_sha)))
    eng.aot_dir = aot_dir(cache, eng)
    eng.set_namespace(tenancy.namespace(deployment=eng.deployment_id, model=dict(id=m["id"], weights=info["file"], config=tenancy.file_sha256(os.path.join(d, "config.json")),
                                                   component_set_sha=eng.component_set_sha),
                                        tokenizer=tenancy.file_sha256(os.path.join(d, "tokenizer.json")),
                                        chat_template=tenancy.file_sha256(os.path.join(d, "chat_template.jinja")),
                                        checkpoint_policy=eng.checkpoint_policy()))
    log(json.dumps(dict(event="loaded", files_s=round(t_files - t0, 1), weights_s=info["load_s"], gb=round(info["device_bytes"] / 1e9, 2),
                        bpw=info["bpw"], kernels=eng.kernels)))
    return m, eng, chat, info, t0


def aot_dir(cache, eng):
    """The executable table's folder (runtime/programs.py): one per deployment under the compile cache folder, local to
    this machine and user. None (no table) with MASHBOX_AOT=0, and with speculative decoding on: the warm-up compiles its
    programs through jit, outside the table, so a table would not hold every program this deployment runs."""
    why = ("MASHBOX_AOT=0" if os.environ.get("MASHBOX_AOT") == "0" else
           "speculative decoding is on: the table does not hold its programs" if getattr(eng, "DK", 0) else None)
    if why:
        log(json.dumps(dict(event="aot_table", used=False, reason=why)))
        return None
    return os.path.join(cache, "aot", eng.deployment_id)


def gguf_slots(a, f, c, kt, lim=None):
    """the slots `run` uses for a GGUF file: --slots as given, or at its default (8) the most that fit this chip
    (gguf.fit_slots; lim = the chip's bytes_limit, JAX's when None). The catalog's record guard re-runs this on a
    record's sealed file header."""
    from . import gguf as G
    return G.fit_slots(f, c, a.ctx, a.slots, kernel_types=kt, lim=lim) if a.slots == 8 else a.slots


def load_gguf(a):
    """`run` for a GGUF file or repo: the file's own blocks on TPU (mashbox.ops), its tokenizer and chat template"""
    t0 = time.time()
    from . import gguf as G, ops
    from .gguf import tokenizer as GT
    from .runtime.engine import Engine
    from .runtime.chat import Chat
    r = a.gguf
    f, A, c = G.open_model(r["path"], repo=r["repo"], quant=r["tag"])
    held = G.archs.held(f.arch, r["path"], r["repo"], r["tag"])     # an unlisted MoE architecture: one line, nothing loaded
    if held:
        raise G.GGUFError(held)
    if getattr(a, "tensor_parallel_size", 0) > 1:
        raise G.GGUFError("mashbox: GGUF files run on one chip in this version; --tensor-parallel-size is for models.json models")
    if not a.no_kernels:
        ops.load_kernels(log)
    line, warn = G.plan(f, c, kernel_types=frozenset() if a.no_kernels else None)
    kt = frozenset() if a.no_kernels else ops.kernel_types()
    s = gguf_slots(a, f, c, kt)
    if s != a.slots:
        log(json.dumps(dict(event="gguf_slots", slots=s, asked=a.slots, ctx=a.ctx)))
        ops.warn(f"mashbox: {a.slots} slots of {a.ctx} tokens do not fit this chip with this file; running --slots {s} "
                 + ("(one request at a time)" if s == 1 else f"({s} requests at once)") + ". --slots N sets it")
        a.slots = s
    G.check_fits(f, c, a.ctx, a.slots, kernel_types=kt, warn=ops.warn)
    log(json.dumps(dict(event="gguf_plan", **line)))
    if warn:
        ops.warn(warn)
    chat = Chat(GT.model_dir(f, r["path"]))
    setup_cache(a.cache_dir)
    W, info = A.load(f, c, use_kernels=not a.no_kernels, log=lambda i: log(json.dumps(dict(event="weights", **i))))
    info["file"] = r["file"]
    eng = Engine(c, W, A, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=None,
                 prefix_cache=a.prompt is None and not a.no_prefix_cache,
                 cache_ram=max(0, a.cache_ram) << 20, ckpts=a.ctx_checkpoints, ckpt_step=a.checkpoint_min_step,
                 attn_bytes=2e9 if a.ctx <= 8192 else 5e8,
                 trace=os.environ.get("MASHBOX_TRACE") == "1")   # per-token logit hashes for checks, as load()
    gguf_deployment(eng, f, r, W)
    log(json.dumps(dict(event="loaded", weights_s=info["load_s"], gb=round(info["device_bytes"] / 1e9, 2), bpw=info["bpw"],
                        kernel_gb=round(info["kernel_bytes"] / 1e9, 2), reference_gb=round(info["reference_bytes"] / 1e9, 2))))
    return dict(id=r["repo"] or r["file"]), eng, chat, info, t0


def file_identity(path):
    """-> (sha256 of a weights file, how it was found). A Hugging Face cache blob is named by the Hub's sha256 of its
    bytes (blobs/<sha256>): that name is used; any other file is read once."""
    from .runtime import tenancy
    real = os.path.realpath(path)
    name = os.path.basename(real)
    if os.path.basename(os.path.dirname(real)) == "blobs" and len(name) == 64 and all(ch in "0123456789abcdef" for ch in name):
        return name, "hub_blob_name"
    return tenancy.file_sha256(path), "read"


def gguf_deployment(eng, f, r, W):
    """The GGUF file's deployment, as load() makes one for a models.json model (runtime/compile_key.py): deployment id
    and component set (/health, the startup log), and the prefix cache namespace. The file's sha256 (every shard of a
    split file) stands for its weights, tokenizer and chat template, which it all holds."""
    import hashlib
    from .gguf import resolve as GR
    from .runtime import tenancy, compile_key as CK
    t = time.time()
    shards = GR.shards(os.path.basename(r["path"]))
    paths = [os.path.join(os.path.dirname(r["path"]), s) for s in shards] if len(shards) > 1 else [r["path"]]
    ids = [file_identity(p) for p in paths]
    fsha = ids[0][0] if len(ids) == 1 else hashlib.sha256(" ".join(i[0] for i in ids).encode()).hexdigest()
    log(json.dumps(dict(event="gguf_identity", sha256=fsha, shards=len(ids), how=sorted({i[1] for i in ids}),
                        seconds=round(time.time() - t, 1))))
    mid = r["repo"] or r["file"]
    _, eng.deployment_id, eng.component_set_sha = CK.for_engine(eng, mid, fsha, CK.weights_layout(W), kernel_bundle_sha=None, tp=1)
    log(json.dumps(dict(event="deployment", deployment_id=eng.deployment_id, component_set_sha=eng.component_set_sha)))
    tmpl = f.get("tokenizer.chat_template")
    eng.set_namespace(tenancy.namespace(deployment=eng.deployment_id,
                                        model=dict(id=mid, weights=r["file"], config=fsha, component_set_sha=eng.component_set_sha),
                                        tokenizer=fsha, chat_template=hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None,
                                        checkpoint_policy=eng.checkpoint_policy()))


def cmd_models(a):
    """`mashbox models`: the catalog in models.json, as mashbox/catalog.py renders it (`--json`: the same entries as data)"""
    from . import catalog
    reg = catalog.registry()
    if a.json:
        print(json.dumps(catalog.entries(reg), indent=1))
    else:
        sys.stdout.write(catalog.render_text(reg))
    return 0


def cmd_inspect(a):
    from . import gguf as G
    from .gguf import resolve as GR
    r = GR.resolve(a.model, log=log, download=not a.no_download)
    if r["path"] is None:
        return 0
    print(json.dumps(G.inspect(r["path"], repo=r["repo"], quant=r["tag"]), indent=1))
    return 0


def warm(eng, chat):
    """Compile the serving path's programs before ready (served from the compile cache when it has them): two short
    requests (the request path end to end), then each prefill, decode, speculation and helper program (Engine.warm).
    Counts every compile from here on (runtime/programs.py). A table folder the load refused, or the warm-up dropped for
    another program list, is moved aside (programs.set_aside) so the save writes a new table there; after a save,
    programs.prune removes the tables no start needs. A program loaded from the table that fails in the two requests or
    in Engine.warm raises programs.TableFailed out of here (the table quarantined and dropped; info["closed"] says
    whether its folder could be closed): the caller ends this process, which never warms a second time (A13), and the
    next start compiles without the table (worker.main sends the table's fatal frame to the front; cmd_run exits 70)."""
    from .runtime import programs
    programs.install()
    t = time.perf_counter()
    d, nw = getattr(eng, "aot_dir", None), min(8, os.cpu_count() or 1)
    meta = ld = None
    if d:       # the saved executables first (checked before use): the warm-up then compiles only what they do not hold
        try:    # the table is an optimisation: any problem with it means compile, never a failed start
            meta = programs.table_meta(getattr(eng, "deployment_id", None))
            ld = eng.aot.load(d, meta, workers=nw)
        except Exception as e:
            eng.aot.discard()
            ld = dict(loaded=0, failed=0, refused=f"{type(e).__name__}: {e}"[:300])
        log(json.dumps(dict(event="aot_table", phase="load", **ld)))
        if ld.get("refused") and ld["refused"] != "no table":
            aside(d)
    ids = chat.chat_ids([dict(role="user", content="Hello")])
    for samp in (None, [dict(temperature=0.7, top_p=0.9, seed=1)]):
        eng.generate([ids], max_new=eng.K + 1, stop_ids=(), samp=samp)
    w = eng.warm(workers=nw)
    if ld and ld.get("loaded") and ld.get("program_list_sha") != w["list_sha"]:
        # built for another program list: drop it before ready and compile every program
        log(json.dumps(dict(event="aot_table", phase="check", refused=f"built for program list {str(ld.get('program_list_sha'))[:12]}, "
                            f"this warm-up's is {w['list_sha'][:12]}: dropped, every program compiles")))
        eng.aot.discard()
        aside(d)
        w = eng.warm(workers=nw)
    if d:
        try:
            s = eng.aot.save(d, meta) if meta is not None else dict(saved=0, refused="no table meta")
        except Exception as e:      # save never raises; the server has its programs in memory either way
            s = dict(saved=0, refused=f"not saved: {type(e).__name__}: {e}"[:300])
        if s.get("saved"):      # a new table: the ones no start needs go (never raises)
            s["pruned"] = programs.prune(os.path.dirname(os.path.abspath(d)), os.path.basename(os.path.abspath(d)))
        log(json.dumps(dict(event="aot_table", phase="save", **s)))
    log(json.dumps(dict(event="warm", prefill_programs=w["prefill"], decode_programs=w["decode"],
                        helper_programs=w["helpers"], programs=w["programs"], program_list_sha=w["list_sha"],
                        compiles=programs.counts()["compiles"], compile_s=w["compile_s"], run_s=w["run_s"],
                        compile_failed=w["compile_failed"], table=w.get("table"), run_programs=w.get("run_programs"),
                        skipped=w.get("skipped"), run_why=w.get("run_why"), fallback=w.get("fallback"),
                        seconds=round(time.perf_counter() - t, 1))))


def aside(d):
    """Move the table folder d that this start will not use aside (programs.set_aside), and log where it went."""
    from .runtime import programs
    r = programs.set_aside(d)
    if r is not None:
        log(json.dumps(dict(event="aot_table", phase="aside", to=r[0], closed=r[1])))


def watch_compiles():
    """From here on every compile is a compile after ready: one compile_after_ready log line each, counted on /health
    (runtime/programs.py). Called where the engine's process says ready: cmd_run's one-process server, and the worker
    process that holds the engine behind an HTTP front (runtime/worker.py from_cli)."""
    from .runtime import programs
    programs.mark_ready(lambda name, secs: log(json.dumps(dict(event="compile_after_ready", program=name,
                                                               seconds=round(secs, 3)))))


def cmd_run(a, argv=None):
    keys = api_keys(a) if a.prompt is None else []       # before the load: a bad key file fails at once
    if a.prompt is None and not getattr(a, "gguf", None):
        # serving a models.json model: this process is the HTTP front; the engine (load, warm-up, chip) runs in a
        # worker process that the front restarts by itself after a device fault, out of memory, a hung call or a kill
        # (runtime/front.py). A GGUF file is still served in this one process (below).
        from .runtime.front import serve_cli          # serve() logs the "ui" url once the port is bound
        return serve_cli(a, sys.argv[1:] if argv is None else list(argv), keys, log, t0=time.time())
    m, eng, chat, info, t0 = load_gguf(a) if getattr(a, "gguf", None) else load(a)
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
        from .runtime import programs
        try:
            warm(eng, chat)
        except programs.TableFailed as e:       # quarantined and dropped: this process never warms a second time (A13)
            log(json.dumps(dict(event="aot_table", phase="failed", **e.info)))
            log("mashbox: a program loaded from the saved program table failed while the server started; start it again"
                + ("" if e.info.get("closed", True) else " with MASHBOX_AOT=0 (the table's folder could not be closed: "
                   + str(e.info.get("quarantined"))[:300] + ")"))
            sys.exit(70)
    watch_compiles()
    srv.ready = True
    log(json.dumps(dict(event="ready", seconds_since_start=round(time.time() - t0, 1))))
    serve(srv, a.host, a.port, log, ui=a.ui, keys=keys)       # it logs the "ui" url once the port is bound


KEY_FILE = os.path.join(os.path.expanduser("~"), ".config", "mashbox", "api-key")


def read_key_file(path):
    """An API key file's text, read only from a regular file (no link followed) that belongs to this user and that no one
    else can read or write: a key others can read lets them act as its tenant. Anything else ends the start with one
    plain error that names the file, never a key."""
    import stat
    fix = f"; an API key file must be a regular file of this user that only it can read and write (chmod 600 {path})"
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                     | getattr(os, "O_CLOEXEC", 0))
    except OSError as e:
        if os.path.islink(path):
            raise SystemExit(f"mashbox: {path} is a symlink" + fix)
        raise SystemExit(f"mashbox: cannot read {path}: {e.strerror}")
    with os.fdopen(fd) as f:
        st = os.fstat(f.fileno())
        why = ("is not a regular file" if not stat.S_ISREG(st.st_mode) else
               "belongs to another user" if hasattr(os, "getuid") and st.st_uid != os.getuid() else
               f"can be read or written by group or others (mode {stat.S_IMODE(st.st_mode):o})"
               if hasattr(os, "getuid") and st.st_mode & 0o077 else None)
        if why:
            raise SystemExit(f"mashbox: {path} {why}" + fix)
        return f.read()


def api_keys(a):
    """The server's API keys: --api-key-file (one per line, as llama-server; a line may name the key's tenant after it,
    "<key> <tenant>", so keys of one tenant share its prefix cache), else MASHBOX_API_KEY (comma-separated), else none on
    a loopback --host; on any other --host one is made (once) in ~/.config/mashbox/api-key, mode 0600, and its path is
    printed, never the key. A key file is read only when it passes read_key_file's owner and mode check. Sets the
    process's tenancy (runtime/tenancy.py): a fresh cache secret, and each key's tenant (a key without one is its own
    tenant)."""
    from .runtime import guard, tenancy
    tenants = {}
    if a.api_key_file:
        keys, tenants = tenancy.parse_key_lines(read_key_file(a.api_key_file).splitlines())
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
        keys, src = [x.strip() for x in read_key_file(KEY_FILE).splitlines() if x.strip()], KEY_FILE
    if not keys:                                                        # never start open because a key source is empty
        raise SystemExit(f"mashbox: {src} holds no key"
                         + (" (delete the file and start again: a new key is made)" if src == KEY_FILE else ""))
    tenancy.configure(tenants)
    log(json.dumps(dict(event="api_key", keys=len(keys), source=src, tenants=len(set(tenants.values())),
                        own_tenant=sum(1 for k in keys if k not in tenants),
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
        p.add_argument("model", help="model id, as in models.json (e.g. qwen3-8-27b), a .gguf file, or a Hugging Face GGUF "
                       "repo with an optional quant tag (e.g. unsloth/Qwen3.8-27B-GGUF or unsloth/Qwen3.8-27B-GGUF:Q8_0)")
        p.add_argument("--weights", metavar="FILE", help="a local .bshj file instead of the download (its sha256 is still checked)")
        p.add_argument("--no-kernels", action="store_true", help="plain JAX only: a catalog model runs without its TPU kernels "
                       "(slower decode, same outputs within rounding)")
        if name == "run":
            p.add_argument("--prompt", metavar="TEXT", help="answer one chat turn, print it and exit; without it, start the server")
            p.add_argument("--thinking", action="store_true", help="thinking mode for --prompt")
            p.add_argument("--max-tokens", type=int, default=512, metavar="N", help="most new tokens for --prompt")
            p.add_argument("--temperature", type=float, default=0.0, metavar="T", help="sampling temperature for --prompt; 0 is greedy")
            p.add_argument("--top-p", type=float, default=1.0, metavar="P", help="nucleus sampling cut for --prompt")
            p.add_argument("--top-k", type=int, default=0, metavar="K", help="top-k cut for --prompt; 0 means none")
            p.add_argument("--seed", type=int, default=0, metavar="N", help="seed for sampled --prompt answers")
            p.add_argument("--host", default="127.0.0.1", metavar="H", help="server address; keep it on 127.0.0.1 and use an SSH tunnel")
            p.add_argument("--port", type=int, default=8000, metavar="N",
                           help="server port; 0 picks a free one, named in the \"listening\" log line (and \"ui\" with --ui)")
            p.add_argument("--api-key-file", metavar="FILE",
                           help="require an API key from FILE on /v1 requests: one per line, a key or a key then its "
                                "tenant's name; keys of one tenant share one prompt cache and one waiting queue, and a key "
                                "without a name is its own tenant; only you may read or write FILE (chmod 600). Also "
                                "MASHBOX_API_KEY=k1,k2. Default: none on 127.0.0.1, and on any other --host a key made in "
                                "~/.config/mashbox/api-key")
            p.add_argument("--ui", action="store_true", help="also serve the MASHBOX Chat page at http://127.0.0.1:<port>/")
            p.add_argument("--ctx", type=int, default=2048, metavar="N", help="context per request in tokens (prompt plus answer)")
            p.add_argument("--slots", type=int, default=8, metavar="N",
                           help="requests decoded together; a catalog model's kernels are built for 8 and other values run it on "
                                "plain JAX; at the default, a GGUF file takes as many as fit the chip's memory")
            p.add_argument("--steps", type=int, default=8, metavar="N", help="decode steps per device call")
            p.add_argument("--mtp", type=int, metavar="K",
                           help="speculative decoding with the model's own multi-token-prediction head: K drafts per verify pass "
                                "(1 to 3; 0 = off). Default: on only on chips where it passed its checks, else off. Uses the "
                                "model's draft file (Qwen3.8-27B: --mtp 2); one chip only")
            p.add_argument("--mtp-file", metavar="FILE",
                           help="a local draft .bshj file instead of the download (its sha256 is still checked)")
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
    p = sub.add_parser("models", help="the models in this version's catalog, by publisher, then model, then file")
    p.add_argument("--json", action="store_true", help="the same list as JSON")
    p = sub.add_parser("inspect", help="what a GGUF file or repo holds (architecture, block types, tokenizer) and how it "
                       "would run here, without loading it")
    p.add_argument("model", help="a .gguf file or a Hugging Face GGUF repo[:quant tag]")
    p.add_argument("--no-download", action="store_true", help="for a repo: only show which file llama.cpp's rule picks")
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
    if a.cmd == "models":
        return cmd_models(a)
    from .gguf import resolve as GR
    if a.cmd == "inspect" or GR.is_gguf_request(a.model):
        return cmd_gguf(a)
    from . import __version__
    mid = hub.model_entry(a.model)["id"]
    if mid not in RUNNABLE:
        log(f"{mid} is not in mashbox {__version__}. Models in this version: https://mashbox.gg/docs/models")
        return 2
    if a.cmd == "download":
        return cmd_download(a)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        log("note: serving on a non-loopback address exposes the server to your network (an API key is required); "
            "prefer an SSH tunnel")
    return cmd_run(a, argv)


def cmd_gguf(a):
    """GGUF files and repos: every refusal is one clear line and exit 2, never a stack trace"""
    from .gguf import GGUFError, resolve as GR
    try:
        if a.cmd == "inspect":
            return cmd_inspect(a)
        if getattr(a, "weights", None):
            raise GGUFError("mashbox: --weights is for models.json models; pass the .gguf file as the model")
        if getattr(a, "mtp", None) or getattr(a, "mtp_file", None):
            raise GGUFError("mashbox: --mtp and --mtp-file are for models.json models; GGUF files run without speculative decoding")
        a.gguf = GR.resolve(a.model, log=log)
        if a.cmd == "download":
            log(json.dumps(dict(event="downloaded", path=a.gguf["path"])))
            return 0
        if a.host not in ("127.0.0.1", "localhost", "::1"):
            log("note: serving on a non-loopback address exposes the server to your network (an API key is required); "
                "prefer an SSH tunnel")
        return cmd_run(a)
    except GGUFError as e:
        log(str(e))
        return 2
