"""The one resolver: every `mashbox run | download | inspect <target>` becomes a Plan
read from the target's own data, and load(plan, overrides) runs it. No model name or id is written here: a new package
needs its models.json row (or files in a format below), never an edit of this file or of the CLI.

  target                                               probe        runs as
  a models.json id, name or Hugging Face repo          catalog      the runtime package its row's path names
                                                                    (src/mashbox/models/<package>): the package's own
                                                                    command line when it defines main(cmd, argv), else the
                                                                    engine with the package's model definition
  a models.json gguf[] row's id                        catalog      the GGUF file the row pins (revision + sha256)
  a .gguf file, or a repo with GGUF files              gguf         llama.cpp's -hf file pick (gguf/resolve.py)
  a .bshj file, or a repo of BSHJ files with no row    bshj         refused in one line: BSHJ weights run as a catalog model
  a model folder, or a repo with config.json and       safetensors  mashbox.hf (the same architectures and engine)
  safetensors weights

The first probe that claims the target wins. A target no probe claims, and a package this version cannot run, stop with
one line saying why (ResolveError; the CLI prints it and exits 2). Nothing here imports JAX or opens a chip: the serving
front resolves without the chip; the kernel path and the fit are decided at load, on the chip.
"""
import dataclasses
import os


class ResolveError(Exception):
    """A target this version cannot run. The message is for the person, exit code 2."""


@dataclasses.dataclass
class Plan:
    target: str                     # what was asked for, as typed
    format: str = None              # "catalog" | "gguf" | "bshj" | "safetensors"
    source: dict = None             # a file or folder: path (None until fetched), repo, tag, file, revision, files, bytes
    arch: str = None                # architecture, when the target's data names it before load
    moe: bool = None                # routed experts, when known before load
    types: tuple = None             # quant types, when known before load (a catalog gguf[] row lists them)
    kernel_path: str = None         # decided at load, on the chip (ops.load_kernels; a catalog model's kernels[])
    bundle: dict = None
    fit: dict = None                # decided at load (cli.fit_check, gguf.fit_slots, hf.fit_form)
    defaults: dict = dataclasses.field(default_factory=dict)    # run's flags hold the defaults in this version
    row: dict = None                # the models.json row the target names (models[] or gguf[])
    why: list = dataclasses.field(default_factory=list)          # one line per decision, for inspect and the log
    entry: str = None               # a models[] row: "engine" (cli.load) or "own" (the package's main(cmd, argv))
    module: str = None              # a models[] row: its runtime package


ENGINE_NAMES = ("Config", "new_state", "make_forward")      # the model definition the engine runs


def runtime(m):
    """-> (module, entry) of a models.json models[] row, read from its package's source (nothing is imported): module =
    the package its path names (src/mashbox/models/<package>); entry "own" when the package's __init__ defines
    main(cmd, argv) (it parses its own run and download options), else "engine" when it holds a model definition the
    engine runs (Config, new_state and make_forward, and a weights module). Anything else raises ResolveError with the
    one line saying what is missing."""
    import ast
    import importlib.util
    import pathlib
    from . import __version__
    parts = pathlib.PurePosixPath(m.get("path") or "").parts
    parts = parts[1:] if parts[:1] == ("src",) else parts
    if len(parts) != 3 or parts[:2] != (__package__, "models"):
        raise ResolveError(f"mashbox: {m['id']}: its models.json row names no runtime package (path {m.get('path')!r}; "
                           f"a runtime package is src/{__package__}/models/<package>)")
    mod = ".".join(parts)
    spec = importlib.util.find_spec(mod)
    if spec is None or not spec.origin:
        raise ResolveError(f"{m['id']} is not in mashbox {__version__}. Models in this version: https://mashbox.gg/docs/models")
    body = ast.parse(pathlib.Path(spec.origin).read_text(encoding="utf-8")).body
    if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in body):
        return mod, "own"
    names = set()
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(x.asname or x.name).split(".")[0] for x in n.names}
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    missing = [x for x in ENGINE_NAMES if x not in names]
    if not (pathlib.Path(spec.origin).parent / "weights.py").is_file():
        missing.append("weights.py")
    if missing:
        raise ResolveError(f"mashbox: {m['id']}: its package {mod} has neither main(cmd, argv) (its own command line) nor "
                           f"the model definition the engine runs (missing {', '.join(missing)})")
    return mod, "engine"


def own_command_line(target):
    """-> the runtime package of the models.json row `target` names when that package parses its own `run` / `download`
    options (entry "own"), else None. The catalog only (no network, nothing imported); a row this version cannot run is
    left to resolve(), which says why."""
    from . import hub
    if os.path.exists(target):
        return None
    m, _ = hub.find_row(target)
    if m is None:
        return None
    try:
        mod, entry = runtime(m)
    except ResolveError:
        return None
    return mod if entry == "own" else None


def _key(target):
    """a target as the catalog spells it: no Hugging Face host in front, no slash at either end"""
    from . import hub
    key = str(target).strip()
    for h in hub.HF_HOSTS:
        if key.lower().startswith(h):
            key = key[len(h):]
    return key.strip("/")


def _gguf_row(key, reg):
    return next((g for g in reg.get("gguf") or [] if g.get("id") == key.lower()), None)


def is_file_target(target):
    """True when a target names files (a path on disk, a GGUF row's id, or anything shaped like a Hugging Face repo),
    so the flags of a GGUF file apply to it, False for a models.json model or any other name. Local: no network."""
    from . import hub
    from .gguf import resolve as GR
    if os.path.exists(target):
        return True
    if hub.find_row(target)[0] is not None:
        return False
    return _gguf_row(_key(target), hub.registry()) is not None or GR.is_gguf_request(target)


def catalog_probe(target, ctx):
    """a models.json models[] row by id, name or Hugging Face repo, or a gguf[] row by id"""
    from . import hub
    if os.path.exists(target):
        return None                                 # a file or folder on disk is never a catalog name
    reg = hub.registry()
    key = _key(target)
    repo, _, tag = key.partition(":")
    m, how = hub.find_row(repo, reg)
    if m is not None:
        if tag:
            raise ResolveError(f"mashbox: {repo} is the catalog model {m['id']}; a quant tag picks a GGUF file: "
                               f"run `mashbox run {m['id']}`")
        mod, entry = runtime(m)
        run = ("its own command line (the package's main)" if entry == "own" else
               "the engine with the package's model definition; kernels, fit and speculation from the row, decided on "
               "the chip at load")
        return Plan(target=target, format="catalog", row=m, entry=entry, module=mod, arch=None, moe=m.get("moe"),
                    source=dict(module=mod, entry=entry),
                    why=[f"format: catalog, models.json row {m['id']} (matched by its {how})", f"runtime: {mod}, {run}"])
    g = _gguf_row(key, reg)
    if g is None:
        return None
    return pinned(target, g, ctx)


class PinGone(Exception):
    """the repo, or the revision a gguf[] row pins, is gone from the Hub (the message says which)"""


def hub_sha256(repo, files, revision):
    """-> {path: the Hub's sha256 of the file's bytes (its LFS sha256; None for a file stored without LFS)} for those of
    `files` that exist in `repo` at `revision`. Raises PinGone when the repo or the revision is gone, and one of
    unreachable() when the Hub cannot be reached."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
    from huggingface_hub.hf_api import RepoFile
    try:
        infos = HfApi().get_paths_info(repo, list(files), revision=revision)
    except RevisionNotFoundError:
        raise PinGone(f"its pinned revision {revision[:12]} is gone from {repo}")
    except RepositoryNotFoundError:
        raise PinGone(f"{repo} is gone from Hugging Face (or it is gated and HF_TOKEN is not set)")
    return {i.path: (i.lfs.sha256 if getattr(i, "lfs", None) else None) for i in infos if isinstance(i, RepoFile)}


def unreachable():
    """the exception types of a Hub that cannot be reached (no network, offline mode, a timeout, a server error)"""
    import importlib
    out = [OSError]
    for mod, name in (("httpx", "TransportError"), ("requests", "RequestException")):
        try:
            out.append(getattr(importlib.import_module(mod), name))
        except (ImportError, AttributeError):
            pass
    return tuple(out)


def pinned(target, g, ctx):
    """a models.json gguf[] row's Plan: the file the row names (its shards), at the row's revision, whose sha256 must be
    the row's. The Hub's sha256 of each shard at that revision is read now, and the bytes are checked again after
    download (fetch). A revision or file gone upstream, a sha256 that is not the row's, or MASHBOX_FAULT=force:pin
    refuses the pinned run in one line naming the unpinned run (`mashbox run repo:tag`, the repo's current file). When
    the Hub cannot be reached the check waits for the bytes."""
    import json
    from .gguf import resolve as GR
    from .runtime import faults
    repo, tag, rev, file, sha = (g.get(k) for k in ("repo", "tag", "revision", "file", "sha256"))
    if not (repo and tag and rev and file and sha):
        raise ResolveError(f"mashbox: {g['id']}: its models.json gguf row does not pin a repo, quant tag, revision, file "
                           f"and sha256")
    unpinned = f"`mashbox run {repo}:{tag}`"

    def refuse(why):
        return ResolveError(f"mashbox: {g['id']}: {why}, so the pinned run stops; the repo's current file for {tag} runs "
                            f"only when asked unpinned: {unpinned}")
    files = GR.shards(file)
    if faults.forced("pin"):                        # the fallback test's switch (MASHBOX_FAULT=force:pin)
        raise refuse(f"{file} at {rev[:12]} is not the file the row pins (forced)")
    try:
        got = hub_sha256(repo, files, rev)
    except PinGone as e:
        raise refuse(str(e))
    except unreachable() as e:
        got, offline = None, f"{type(e).__name__}: {e}"[:160]
    if got is not None:
        gone = [f for f in files if f not in got]
        if gone:
            raise refuse(f"{', '.join(gone)} is gone from {repo} at {rev[:12]}")
        have = GR.joined_identity(got[f] for f in files)
        if have is not None and have != sha:
            raise refuse(f"{file} at {rev[:12]} has sha256 {have[:12]} upstream, not the row's {sha[:12]}")
        check = "pin: the Hub's sha256 at that revision is the row's" if have is not None else \
            "pin: the Hub states no sha256 for it; the bytes are checked after download"
    else:
        check = f"pin: the Hub could not be reached ({offline}); the bytes are checked after download"
    ctx["log"](json.dumps(dict(event="gguf_file", repo=repo, file=file, shards=len(files),
                               gb=round((g.get("bytes") or 0) / 1e9, 2), revision=rev, rule=f"models.json gguf row {g['id']}")))
    plan = Plan(target=target, format="gguf", row=g, arch=g.get("arch"), moe=g.get("moe"),
                types=tuple(g.get("types") or ()) or None,
                source=dict(path=None, repo=repo, tag=tag, file=file, revision=rev, files=files, bytes=g.get("bytes"),
                            sha256=sha),
                why=[f"format: catalog, models.json gguf row {g['id']}: {file} from {repo}, pinned at {rev[:12]} "
                     f"(sha256 {sha[:12]})", check,
                     "architecture, MoE hold, kernel path and fit: from the file's header at load (gguf_plan)"])
    return fetch(plan, ctx["log"]) if ctx["download"] else plan


def gguf_probe(target, ctx):
    """a .gguf file (any file that is not a .bshj), or a Hugging Face repo with GGUF files (owner/name[:quant tag])"""
    from .gguf import resolve as GR
    if os.path.isdir(target) or target.lower().endswith(".bshj") or not GR.is_gguf_request(target):
        return None
    try:
        r = GR.resolve(target, log=ctx["log"], download=ctx["download"])
    except GR.NoGGUF as e:
        ctx["files"] = e.files                      # the repo's listing, for the probes after this one
        return None
    why = f"format: GGUF, {r['file']}" + (f" from {r['repo']}" + (f" (tag {r['tag']})" if r["tag"] else
                                                                   " (llama.cpp's default pick)") if r["repo"] else "")
    return Plan(target=target, format="gguf", source=r,
                why=[why, "architecture, MoE hold, kernel path and fit: from the file's header at load (gguf_plan)"])


def bshj_probe(target, ctx):
    """a .bshj file, or a repo of .bshj files with no GGUF file (listed by the GGUF probe) that no catalog row names:
    BSHJ weights run only as a catalog model (their row names the runtime package), so this probe claims and refuses"""
    from . import __version__, hub
    local = os.path.isfile(target)
    if local:
        names = [os.path.basename(target)] if target.lower().endswith(".bshj") else []
    else:
        names = [p for p in ctx.get("files") or [] if p.lower().endswith(".bshj")]
    if not names:
        return None
    if local:                                       # a catalog model's own weights file: run that model with it
        m = next((m for m in hub.registry().get("models") or [] if any(f.get("kind", "weights") == "weights"
                  and f.get("name") == names[0] for f in m.get("files") or [])), None)
        if m is not None:
            raise ResolveError(f"mashbox: {target} is the weights file of the catalog model {m['id']}: run "
                               f"`mashbox run {m['id']} --weights {target}`")
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    raise ResolveError(f"mashbox: {target} holds BSHJ weights ({shown}) that no models.json row of mashbox {__version__} "
                       f"names: BSHJ weights run as a catalog model (`mashbox models`); their tensors open with "
                       f"mashbox.bshj.File")


def safetensors_probe(target, ctx):
    """a model folder, or a Hugging Face repo with config.json and safetensors weights and no GGUF file (mashbox.hf). A
    folder's config.json is read here; a repo's is read by hf.resolve when the plan is fetched, before the weights (an
    architecture that path does not read is refused there), so download=False lists the repo only"""
    from . import hf as H
    from .moe import spec as MS
    if not H.claims(target, files=ctx.get("files")):
        return None
    form = "form (bf16 or Q8_0 made on load) and fit: from config.json and the chip at load"
    if not os.path.isdir(target):
        plan = Plan(target=target, format="safetensors", source=dict(path=None, repo=target, revision=None),
                    why=[f"format: safetensors, {target} (config.json and safetensors weights, no GGUF file)",
                         f"architecture: from config.json, read before the weights download; {form}"])
        return fetch(plan, ctx["log"]) if ctx["download"] else plan
    r = H.resolve(target, log=ctx["log"], download=False)
    cfg = H.read_config(r["path"])
    arch, _ = H.text_config(cfg)                    # an architecture this path does not read is refused here
    return Plan(target=target, format="safetensors", source=r, arch=arch, moe=MS.detect_hf(cfg) is not None,
                why=[f"format: safetensors, model folder {r['path']}", f"architecture: {arch}; {form}"])


PROBES = [catalog_probe, gguf_probe, bshj_probe, safetensors_probe]


def resolve(target, *, chip=None, download=True, log=print):
    """-> the Plan of the first probe that claims `target`. download=False reads listings and small files only (the
    weights come with fetch(plan) or load). A target no probe claims raises ResolveError with one line."""
    target = str(target)
    ctx = dict(chip=chip, download=download, log=log)
    for probe in PROBES:
        plan = probe(target, ctx)
        if plan is not None:
            return plan
    if os.path.isdir(target):
        raise ResolveError(f"mashbox: {target}: no config.json with safetensors weights here. mashbox runs a model folder, "
                           f"a Hugging Face repo with safetensors or GGUF files, a .gguf file and the models in "
                           f"`mashbox models`")
    if ctx.get("files") is not None:
        raise ResolveError(f"mashbox: {target} has no GGUF, BSHJ or safetensors model files. mashbox runs the models in "
                           f"`mashbox models`, GGUF files and repos, and model folders or Hugging Face repos with "
                           f"config.json and safetensors weights")
    raise ResolveError(f"mashbox: unknown model {target!r}: not an id, name or Hugging Face repo in `mashbox models`, a "
                       f"GGUF file or repo (owner/name[:quant tag]), or a model folder")


def fetch(plan, log=print):
    """Download the weights a Plan made with download=False names (the files its probe picked, at the revision it
    read; a safetensors repo is listed and checked again by hf.resolve, at its current revision). A catalog model's
    files come at load (hub.py, pinned by sha256). -> plan"""
    s = plan.source
    if plan.format not in ("gguf", "safetensors") or not isinstance(s, dict) or s.get("path") is not None:
        return plan
    if plan.format == "gguf":
        from huggingface_hub import hf_hub_download
        local = [hf_hub_download(s["repo"], p, revision=s["revision"]) for p in s["files"]]
        if s.get("sha256"):                         # a pinned row: the bytes must be the file the row names
            from .gguf import resolve as GR
            have = GR.identity(local[0])[0]
            if have != s["sha256"]:
                g = plan.row or {}
                raise ResolveError(f"mashbox: {g.get('id', s['file'])}: the downloaded {s['file']} has sha256 "
                                   f"{str(have)[:12]}, not the row's {s['sha256'][:12]}, so the pinned run stops; the "
                                   f"repo's current file for {s['tag']} runs only when asked unpinned: "
                                   f"`mashbox run {s['repo']}:{s['tag']}`")
        s["path"] = local[0]
    else:                                           # the repo's current revision, config.json first, then the weights
        from . import hf as H
        plan.source = H.resolve(s["repo"], log=log, download=True)
    return plan


def given(a):
    """-> the Plan of a run whose caller picked the file itself (a.gguf: a GGUF pick, a.hf: an hf.resolve result), else
    None: the loader takes it as given, with nothing resolved again"""
    for fmt, key in (("gguf", "gguf"), ("safetensors", "hf")):
        if getattr(a, key, None):
            return Plan(target=getattr(a, "model", None), format=fmt, source=getattr(a, key))
    return None


def refusals():
    """the exception types whose message is the one line a refused target prints (exit 2)"""
    from .gguf import GGUFError
    from .hf import HFError
    from .moe import MoEError
    return ResolveError, GGUFError, HFError, MoEError


def load(plan, overrides=None):
    """Load what a Plan names, as `run` does, with `overrides` = run's parsed command line (None: run's defaults).
    -> (model, engine, chat, info, t0). A models[] row whose package has its own command line is not loaded here
    (`mashbox run <id>` hands that package the command line)."""
    from . import cli
    a = overrides if overrides is not None else cli.build_parser().parse_args(["run", plan.target])
    if plan.format == "catalog":
        if plan.entry != "engine":
            raise ResolveError(f"mashbox: {plan.row['id']} runs through its own command line: "
                               f"`mashbox run {plan.row['id']} --help`")
        a.model = plan.row["id"]
        return cli.load(a)
    fetch(plan, cli.log)
    if plan.format == "gguf":
        a.gguf = plan.source
        return cli.load_gguf(a)
    if plan.format == "safetensors":
        from . import hf as H
        a.hf = plan.source
        return H.load(a)
    raise ResolveError(f"mashbox: {plan.target}: no loader for the {plan.format} format in this version")
