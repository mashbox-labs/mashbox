"""Hugging Face in: run a model's own safetensors (the maker's repo as published, or a local folder) with the engine.

The weights go through the same architecture modules as GGUF files (mashbox/gguf/archs): adapter.File presents the
folder as the GGUF File they read. The form is picked from the chip's memory, with no flag:

  bf16   the weights as published, when they fit at the default slots and context (native bfloat16 matmul)
  q8_0   llama.cpp's 8-bit blocks made on load (quant.py), when bf16 does not fit and Q8_0 does
  else   a clear refusal naming the chips that fit

  claims(ref, files=None)    -> True when ref has this path's shape (a folder, or a repo with config.json and
                                safetensors and no GGUF files); resolve then reads config.json
  resolve(ref, log, download) -> dict(path, repo, revision, files, bytes): the folder, downloaded at the repo's revision
                                once its config.json names an architecture this path reads (else one-line refusal)
  open_model(path, act, form) -> (File, arch module, Config), the contract of mashbox.gguf.open_model
  pick_form(f, A, c, ...)    -> (form, why)
  run_refusal(a)             -> the one-line refusal for run flags this path does not take, else None
  load(a)                    -> (model, engine, chat, info, t0), the contract of cli.load_gguf
  inspect(ref)               -> what the model is and how it would run here, without loading it
MASHBOX_HF_LOAD=bf16|q8_0 forces the form (tests).
"""
import contextlib
import hashlib
import json
import os
import time

from .source import HFError, open_source, weight_files  # noqa: F401
from .adapter import ARCHITECTURES, File, text_config  # noqa: F401
from . import quant

FORMS = ("bf16", "q8_0")
ONE_CHIP = ("mashbox: Hugging Face models run on one chip in this version; --tensor-parallel-size is for models.json "
            "models")
ALLOW = ["*.safetensors", "*.json", "*.jinja", "tokenizer*"]
DENY = ["consolidated.safetensors", "original/*", "*.pth", "*.bin", "*.gguf", "*.pt"]


def _err(e):
    from ..gguf import GGUFError
    return isinstance(e, (HFError, GGUFError))


@contextlib.contextmanager
def hub(ref):
    """Hugging Face Hub errors as one-line refusals: a gated repo, no such repo or revision, no config.json or safetensors"""
    from huggingface_hub import errors as E
    missing = tuple(getattr(E, n) for n in ("EntryNotFoundError", "NotASafetensorsRepoError") if hasattr(E, n))
    try:
        yield
    except E.GatedRepoError:
        raise HFError(f"mashbox: {ref} is gated: accept its terms on huggingface.co and log in (hf auth login)") from None
    except (E.RepositoryNotFoundError, E.RevisionNotFoundError):
        raise HFError(f"mashbox: {ref}: no such model on Hugging Face (or it is gated and you are not logged in)") from None
    except missing:
        raise HFError(f"mashbox: {ref} has no config.json with safetensors weights") from None


def is_model_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")) and weight_files(path)[0] is not None


def claims(ref, files=None):
    """a local model folder, or a Hub repo (owner/name, no quant tag) whose listing has config.json and safetensors
    weights and no .gguf file. files: the repo's paths when the caller already listed it."""
    if os.path.isdir(ref):
        return is_model_dir(ref)
    if ref.lower().endswith((".gguf", ".bshj")) or ":" in ref or ref.count("/") != 1 or os.path.exists(ref):
        return False
    if files is None:
        try:
            from huggingface_hub import HfApi
            files = [s.rfilename for s in HfApi().model_info(ref).siblings or []]
        except Exception:
            return False
    files = list(files)
    return ("config.json" in files and any(p.endswith(".safetensors") for p in files)
            and not any(p.endswith(".gguf") for p in files))


def resolve(ref, log=print, download=True):
    """-> dict(path, repo, revision, files, bytes). A folder passes through; a repo is listed at its current revision and
    downloaded (config, tokenizer, chat template, safetensors; never .bin, original/ or GGUF files)."""
    if os.path.isdir(ref):
        kind, ws = weight_files(ref)
        return dict(path=os.path.abspath(ref), repo=None, revision=None, files=[os.path.basename(p) for p in ws],
                    bytes=sum(os.path.getsize(p) for p in ws))
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    with hub(ref):
        info = HfApi().model_info(ref, files_metadata=True)
    sib = {s.rfilename: (s.size or 0) for s in info.siblings or []}
    if not claims(ref, list(sib)):
        raise HFError(f"mashbox: {ref} has no config.json with safetensors weights")
    with hub(ref):
        cfg = json.load(open(hf_hub_download(ref, "config.json", revision=info.sha)))
    text_config(cfg)                                        # an unmapped architecture stops here, before the weights
    st = [p for p in sib if p.endswith(".safetensors") and "/" not in p and p not in DENY]
    total = sum(sib[p] for p in st)
    log(json.dumps(dict(event="hf_repo", repo=ref, revision=info.sha, shards=len(st), gb=round(total / 1e9, 2))))
    if not download:
        return dict(path=None, repo=ref, revision=info.sha, files=st, bytes=total)
    with hub(ref):
        path = snapshot_download(ref, revision=info.sha, allow_patterns=ALLOW, ignore_patterns=DENY)
    return dict(path=path, repo=ref, revision=info.sha, files=st, bytes=total)


def read_config(path):
    p = os.path.join(path, "config.json")
    if not os.path.isfile(p):
        raise HFError(f"mashbox: {path} has no config.json")
    return json.load(open(p))


def open_model(path, act=None, form=None):
    """-> (File, arch module, Config) for a model folder; form "q8_0" turns the multiplied weights into Q8_0 on load"""
    from ..gguf import archs
    cfg = read_config(path)
    text_config(cfg)                                        # refuse before opening the weights
    f = File(path, cfg, open_source(path))
    f.form = "bf16"
    A = archs.module(f.arch, path)
    c = A.Config(f) if act is None else A.Config(f, act=act)
    if form == "q8_0":
        quant.to_q8_0(f, A, c)
    return f, A, c


def chip_limit():
    try:
        import jax
        return (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
    except Exception:
        return None


def pick_form(f, A, c, ctx, slots, kernel_types=frozenset(), lim=None):
    """-> (form, why). bf16 when its estimate is within the gguf HEADROOM share of the chip, else q8_0 when that fits,
    else the refusal that names the chips that fit. No chip (CPU): bf16."""
    from .. import gguf as G
    forced = os.environ.get("MASHBOX_HF_LOAD")
    if forced:
        if forced not in FORMS:
            raise HFError(f"mashbox: MASHBOX_HF_LOAD={forced!r}; use one of {', '.join(FORMS)}")
        return forced, "MASHBOX_HF_LOAD"
    lim = chip_limit() if lim is None else lim
    if not lim:
        return "bf16", "no chip memory limit (CPU)"
    need = sum(G.need_bytes(f, c, ctx, slots, kernel_types))
    if need <= G.HEADROOM * lim:
        return "bf16", f"bf16 needs about {need / 1e9:.1f} GB of {lim / 1e9:.1f} GB"
    q = q8_0_need(f, A, c, ctx, slots, kernel_types)
    if q <= G.HEADROOM * lim:
        return "q8_0", f"bf16 needs about {need / 1e9:.1f} GB of {lim / 1e9:.1f} GB; Q8_0 made on load needs {q / 1e9:.1f} GB"
    fit = [n for n, b in G.CHIPS if b * G.HEADROOM >= q]
    raise HFError(f"mashbox: this model needs about {need / 1e9:.1f} GB in bf16 and {q / 1e9:.1f} GB in Q8_0 on the chip "
                  f"with {slots} slots of {ctx} tokens, and {G.HEADROOM * lim / 1e9:.1f} GB of this chip's "
                  f"{lim / 1e9:.1f} GB is usable. " + (f"It fits: {', '.join(fit)}. " if fit else "") +
                  "A GGUF of the same model at a smaller quant, a shorter --ctx or fewer --slots also helps.")


def fit_form(f, A, c, ctx, slots, kernel_types=frozenset(), lim=None, step_down=True):
    """-> (form, slots, why). pick_form at slots; when step_down (--slots left at its default) and nothing fits, the same
    at slots/2, ... 1 before refusing, as gguf.fit_slots does for a GGUF file (one v6e chip: Qwen3.5-27B Q8_0 made on
    load needs 32.4 GB with 8 slots, over the usable share, and fits with 2). The refusal is the one at the fewest slots
    tried."""
    s = slots
    while True:
        try:
            form, why = pick_form(f, A, c, ctx, s, kernel_types, lim)
            return form, s, why
        except HFError:
            if not step_down or s <= 1:
                raise
            s //= 2


def q8_0_need(f, A, c, ctx, slots, kernel_types=frozenset()):
    from .. import gguf as G
    saved = dict(f.tensors)
    try:
        quant.to_q8_0(f, A, c)
        return sum(G.need_bytes(f, c, ctx, slots, kernel_types))
    finally:
        f.tensors = saved
        f.size = sum(t.nbytes for t in saved.values())
        f.form = "bf16"


def plan(f, c, form, kernel_types=frozenset()):
    census, total = f.census()
    from .. import ops
    return dict(arch=f.arch, layers=c.L, form=form, weights_gb=round(total / 1e9, 2),
                types=", ".join(f"{k} {100.0 * v['bytes'] / max(1, total):.1f}%" for k, v in census.items()),
                bf16_matmul="native" if ops.dense16.enabled() else "dequantized",
                q8_0_kernel=quant.Q8_0 in kernel_types if form == "q8_0" else None)


# ------------------------------------------------------------------------------------------------------- chat
def chat_files(path):
    """-> {name: text} for tokenizer.json, chat_template.jinja, generation_config.json: the folder's own files, the chat
    template from tokenizer_config.json when the folder has no chat_template.jinja, and stop ids from config.json when it
    has no generation_config.json: its eos ids plus the id of tokenizer_config.json's eos_token (the token that ends an
    assistant turn, <|im_end|> in the Qwen3.5 repos, whose config.json names <|endoftext|>)"""
    j = lambda n: os.path.join(path, n)
    if not os.path.isfile(j("tokenizer.json")):
        raise HFError(f"mashbox: {path} has no tokenizer.json (this version reads the tokenizers library's format)")
    out = {"tokenizer.json": open(j("tokenizer.json"), encoding="utf-8").read()}
    tc = json.load(open(j("tokenizer_config.json"), encoding="utf-8")) if os.path.isfile(j("tokenizer_config.json")) else {}
    if os.path.isfile(j("chat_template.jinja")):
        out["chat_template.jinja"] = open(j("chat_template.jinja"), encoding="utf-8").read()
    else:
        t = tc.get("chat_template")
        if isinstance(t, list):
            t = {e.get("name"): e.get("template") for e in t}.get("default")
        if not t:
            raise HFError(f"mashbox: {path} has no chat template (chat_template.jinja or tokenizer_config.json)")
        out["chat_template.jinja"] = t
    if os.path.isfile(j("generation_config.json")):
        out["generation_config.json"] = open(j("generation_config.json"), encoding="utf-8").read()
    else:
        cfg = read_config(path)
        e = cfg.get("eos_token_id", (cfg.get("text_config") or {}).get("eos_token_id"))
        ids = [] if e is None else list(e) if isinstance(e, list) else [e]
        t = tc.get("eos_token")
        t = t.get("content") if isinstance(t, dict) else t
        if isinstance(t, str):
            tj = json.loads(out["tokenizer.json"])
            vocab = (tj.get("model") or {}).get("vocab")
            i = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}.get(t, vocab.get(t) if isinstance(vocab, dict) else None)
            if isinstance(i, int) and i not in ids:
                ids.append(i)
        if not ids:
            raise HFError(f"mashbox: {path} has no generation_config.json, and neither config.json nor tokenizer_config.json "
                          "names an end-of-sequence token")
        out["generation_config.json"] = json.dumps(dict(eos_token_id=ids if len(ids) > 1 else ids[0]))
    return out


def model_dir(path, root=None):
    """the three files Chat(model_dir) reads, written once to ~/.cache/mashbox/hf/<sha16 of their text>/"""
    fs = chat_files(path)
    h = hashlib.sha256()
    for k in sorted(fs):
        h.update(k.encode() + b"\0" + fs[k].encode() + b"\0")
    root = root or os.path.join(os.environ.get("MASHBOX_CACHE", os.path.expanduser("~/.cache/mashbox")), "hf")
    d = os.path.join(root, h.hexdigest()[:16])
    if not os.path.exists(os.path.join(d, "generation_config.json")):
        os.makedirs(d, exist_ok=True)
        for k in sorted(fs, key=lambda n: n == "generation_config.json"):          # the marker file last
            with open(os.path.join(d, k + ".part"), "w", encoding="utf-8") as o:
                o.write(fs[k])
            os.replace(os.path.join(d, k + ".part"), os.path.join(d, k))
    return d


# ------------------------------------------------------------------------------------------------------- run
def run_refusal(a):
    """the run flags this path does not take, as one line, checked before any download; else None"""
    if getattr(a, "tensor_parallel_size", 0) > 1:
        return ONE_CHIP
    if getattr(a, "weights", None):
        return "mashbox: --weights is for models.json models; a Hugging Face model reads its own safetensors"
    if getattr(a, "mtp", None) or getattr(a, "mtp_file", None):
        return "mashbox: --mtp and --mtp-file are for models.json models"
    if getattr(a, "speculation", None) not in (None, "auto", "off") or getattr(a, "draft_model", None):
        return ("mashbox: --speculation and --draft-model are for models.json models and GGUF files; a Hugging Face "
                "model runs without speculation in this version")
    return None


def load(a, log=None):
    """`run` for a Hugging Face model (a.hf = resolve(...)): the same engine, scheduler and server as a GGUF file"""
    from .. import cli, gguf as G, ops
    from ..runtime.engine import Engine
    from ..runtime.chat import Chat
    log = log or cli.log
    t0 = time.time()
    r = a.hf
    if getattr(a, "tensor_parallel_size", 0) > 1:
        raise HFError(ONE_CHIP)
    if not a.no_kernels:
        ops.load_kernels(log)
    kt = frozenset() if a.no_kernels else ops.kernel_types()
    f, A, c = open_model(r["path"])
    from ..gguf import archs
    held = archs.held(f.arch, r["path"], moe=archs.moe_file(f))     # as a GGUF file: a MoE file with no passing row of its own, one line
    if held:
        raise HFError(held)
    form, s, why = fit_form(f, A, c, a.ctx, a.slots, kt, step_down=a.slots == 8)   # 8 = the default; an explicit --slots is kept
    if s != a.slots:
        log(json.dumps(dict(event="hf_slots", slots=s, asked=a.slots, ctx=a.ctx)))
        ops.warn(f"mashbox: {a.slots} slots of {a.ctx} tokens do not fit this chip with this model; running --slots {s} "
                 + ("(one request at a time)" if s == 1 else f"({s} requests at once)") + ". --slots N sets it")
        a.slots = s
    if form == "q8_0":
        quant.to_q8_0(f, A, c)
    G.check_fits(f, c, a.ctx, a.slots, kernel_types=kt, warn=ops.warn)
    log(json.dumps(dict(event="hf_plan", repo=r["repo"], revision=r["revision"], why=why, **plan(f, c, form, kt))))
    chat = Chat(model_dir(r["path"]))
    cli.setup_cache(a.cache_dir)
    W, info = A.load(f, c, use_kernels=not a.no_kernels)
    import jax
    # the path each multiplied weight runs on, from the handles load made: Dense16 = native bf16, Raw = the exact
    # (dequantized) path. Dense32 (an F32 weight held as its values) and the router are resident_f32, counted by load
    # itself. W["embed"] is left out: its rows are gathered, not multiplied (a tied head is its own handle, W["head"])
    kinds = (ops.Raw, ops.dense16.Dense16)
    mult = {k: v for k, v in W.items() if k != "embed"}
    hs = [x for x in jax.tree_util.tree_leaves(mult, is_leaf=lambda x: isinstance(x, kinds)) if isinstance(x, kinds)]
    ref = {}
    for x in hs:
        if isinstance(x, ops.Raw):
            n = int(x.raw.nbytes)
            ref[ops.reference.TYPES[x.t][0]] = ref.get(ops.reference.TYPES[x.t][0], 0) + n
    info.update(file=r["repo"] or os.path.basename(r["path"]), form=form, reference_bytes=sum(ref.values()),
                reference_types=ref, native_bf16_bytes=sum(int(x.w.nbytes) for x in hs if isinstance(x, ops.dense16.Dense16)))
    log(json.dumps(dict(event="weights", **info)))
    eng = Engine(c, W, A, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=None,
                 prefix_cache=a.prompt is None and not a.no_prefix_cache,
                 cache_ram=max(0, a.cache_ram) << 20, ckpts=a.ctx_checkpoints, ckpt_step=a.checkpoint_min_step,
                 attn_bytes=2e9 if a.ctx <= 8192 else 5e8)
    log(json.dumps(dict(event="loaded", weights_s=info["load_s"], gb=round(info["device_bytes"] / 1e9, 2), bpw=info["bpw"],
                        form=form, kernel_gb=round(info["kernel_bytes"] / 1e9, 2),
                        native_bf16_gb=round(info["native_bf16_bytes"] / 1e9, 2),
                        resident_f32_gb=round(info["resident_f32_bytes"] / 1e9, 2),
                        reference_gb=round(info["reference_bytes"] / 1e9, 2))))
    return dict(id=r["repo"] or r["path"]), eng, chat, info, t0


def inspect(ref, log=print):
    """what the model is and which form would run on each chip, from config.json and the safetensors headers only"""
    from .. import gguf as G
    from .source import Source, ST_DTYPES
    out = dict(model=ref)
    try:
        if os.path.isdir(ref):
            path, rev = ref, None
            cfg, src = read_config(ref), open_source(ref)
        else:
            from huggingface_hub import HfApi, hf_hub_download
            with hub(ref):
                info = HfApi().model_info(ref)
                cfg = json.load(open(hf_hub_download(ref, "config.json", revision=info.sha)))
            path, rev = None, info.sha
            text_config(cfg)
            with hub(ref):
                meta = HfApi().get_safetensors_metadata(ref, revision=rev)
            src = Source()
            for fm in meta.files_metadata.values():
                for n, t in fm.tensors.items():
                    if t.dtype not in ST_DTYPES:
                        raise HFError(f"mashbox: tensor {n} is stored as {t.dtype}; this version reads BF16, F16 and F32")
                    src.tensors[n] = (t.dtype, tuple(t.shape), None)
        from ..gguf import archs
        f = File(path or ref, cfg, src)
        A = archs.module(f.arch, ref)
        c = A.Config(f)
        census, total = f.census()
        out.update(revision=rev, architecture=cfg.get("architectures", [None])[0], arch=f.arch, layers=c.L,
                   weights_gb_bf16=round(total / 1e9, 2), types={k: round(v["bytes"] / 1e9, 3) for k, v in census.items()})
        est = {}
        for name, lim in G.CHIPS:
            try:
                form, why = pick_form(f, A, c, 2048, 8, frozenset(), lim=lim)
                est[name] = dict(form=form, why=why)
            except HFError as e:
                est[name] = dict(form=None, why=str(e))
        out.update(runs_here=True, chips_ctx2048_slots8=est)
    except Exception as e:
        if not _err(e):
            raise
        out.update(runs_here=False, why=str(e))
    return out
