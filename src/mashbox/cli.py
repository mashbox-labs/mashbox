"""mashbox command line.

  mashbox run MODEL                         download (first time), load, serve on 127.0.0.1:8000
  mashbox run MODEL --prompt "Hi"           one chat turn, printed
  mashbox run MODEL --ui                    the same server plus the chat page at http://127.0.0.1:8000/
  mashbox download MODEL                    download and check the files only
  mashbox inspect MODEL                     how MODEL resolves and would run, without loading it
  mashbox models                            the models this version runs, by publisher, then model, then file
  mashbox env                               versions and chip, for bug reports
  mashbox tpu up MODEL                      from your machine: a chip, install, server, tunnel (mashbox tpu -h)

MODEL is resolved by the one resolver (resolve.py), from data only: a models.json id, name or Hugging Face repo (the
runtime package its row names), a GGUF row's id, a .gguf file or GGUF repo[:quant tag] (llama.cpp's file pick), or a
model folder or repo with config.json and safetensors weights. A catalog package with its own options (main(cmd, argv))
gets `run` and `download` handed to it; see `mashbox run <model> --help`. What no probe can run stops with one line.
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
    """(RUNNABLE, MODEL_CLI) read from the catalog, so no model id is typed into this file: the models.json rows whose
    runtime package (resolve.runtime: the package their path names, src/mashbox/models/<package>) ships in this version
    of mashbox and runs, and of those the ones whose package parses its own `run` / `download` options (main(cmd, argv),
    read from the source; nothing is imported). `run` and `download` dispatch through resolve.py; these lists are for
    the docs check (tools/hf_github_sync_check.py)."""
    from . import resolve as R
    runnable, own = [], {}
    for m in reg.get("models", []):
        try:
            mod, entry = R.runtime(m)
        except R.ResolveError:
            continue
        runnable.append(m["id"])
        if entry == "own":
            own[m["id"]] = mod
    return tuple(runnable), own


# Models whose runtime is in this version of the package (MODEL_CLI: those whose package parses its own `run` /
# `download` options, model id -> module with main(cmd, argv)). Data for the docs check; `run` reads the catalog at call
# time through resolve.py, so a row added to models.json runs with no edit here.
RUNNABLE, MODEL_CLI = catalog_runtimes(hub.registry())


def engine_package(m):
    """-> (package, its weights module, its shard module or None) of a models.json row that runs on the engine
    (resolve.runtime entry "engine"): the model definition (Config, new_state, make_forward), the weights loader, and
    the tensor-parallel split when the package has one. Imported by the name the row's path gives, never typed here."""
    import importlib
    import importlib.util
    from . import resolve as R
    try:
        mod, entry = R.runtime(m)
    except R.ResolveError as e:
        raise SystemExit(str(e))
    if entry != "engine":
        raise SystemExit(f"mashbox: {m['id']} runs through its own command line: `mashbox run {m['id']} --help`")
    shard = importlib.import_module(mod + ".shard") if importlib.util.find_spec(mod + ".shard") else None
    return importlib.import_module(mod), importlib.import_module(mod + ".weights"), shard


def setup_cache(d=None):
    import jax
    d = d or os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "mashbox", "xla")
    os.makedirs(d, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", d)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return d


# What this install has. install.sh installs `mashbox` with no extras on any machine that is not a Cloud TPU VM: the client
# for `mashbox tpu`, with no JAX. A command that needs JAX, or a model's extras, says so in one line with the line that
# installs them, before any device check or download.
INSTALL_SH = "curl -fsSL https://mashbox.gg/install.sh"


def has_jax():
    """JAX can be imported here (checked without importing it)"""
    import importlib.util
    if "jax" in sys.modules:
        return sys.modules["jax"] is not None
    return importlib.util.find_spec("jax") is not None


def uv_tool():
    """this Python is a uv tool's own environment (install.sh's, or the catalog's `uv tool install` lines): uv writes
    uv-receipt.toml in the tool's prefix"""
    return os.path.isfile(os.path.join(sys.prefix, "uv-receipt.toml"))


def install_line(extras, m=None):
    """the one line that installs mashbox with `extras` (e.g. "cpu,track2rig") in place of this install: install.sh again
    with MASHBOX_EXTRAS in a uv tool (uv replaces the tool's environment when its extras change), else the models.json
    model's own install line, else pip for this Python"""
    if uv_tool():
        return f"{INSTALL_SH} | MASHBOX_EXTRAS={extras} sh"
    if m is not None and m.get("install"):
        return m["install"]
    import shlex
    return f'{shlex.quote(sys.executable)} -m pip install "mashbox[{extras}]"'


def model_extras(m):
    """the extras of a models.json model's install line (its own `install`, else the catalog's), e.g. "tpu,hy-nomas" """
    import re
    got = re.search(r"mashbox\[([^\]]*)\]", m.get("install") or hub.registry().get("install", ""))
    return got.group(1) if got else ""


def requirements():
    """mashbox's requirement lines as its installed metadata lists them (e.g. 'torch>=2.5; extra == "hy-nomas"'); run
    from a source tree that is not installed, its pyproject.toml's in the same form; [] when neither is there"""
    import importlib.metadata as md
    try:
        return list(md.requires(__package__) or [])
    except md.PackageNotFoundError:
        pass
    import pathlib
    import tomllib
    try:
        proj = tomllib.loads((pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"]
    except (OSError, ValueError, KeyError):
        return []
    if proj.get("name") != __package__:
        return []
    return list(proj.get("dependencies", [])) + [f'{r}; extra == "{x}"' for x, rs in
                                                 proj.get("optional-dependencies", {}).items() for r in rs]


def installed(dist):
    """a distribution of that name is installed in this Python"""
    import importlib.metadata as md
    try:
        md.distribution(dist)
        return True
    except md.PackageNotFoundError:
        return False


def lacking(extras):
    """the distributions mashbox's `extras` (e.g. "tpu,hy-nomas") add that this Python does not have, in requirements()
    order"""
    import re
    norm = lambda s: re.sub(r"[-_.]+", "-", s.strip()).lower()      # noqa: E731
    want, out = {norm(x) for x in extras.split(",") if x.strip()}, []
    for r in requirements():
        spec, _, marker = r.partition(";")
        x = re.search(r"""extra\s*==\s*["']([^"']+)["']""", marker)
        name = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
        if x and name and norm(x.group(1)) in want and name.group(1) not in out and not installed(name.group(1)):
            out.append(name.group(1))
    return out


def extras_refusal(model, cmd="run"):
    """None, or the one line `mashbox <cmd> <model>` stops with, before its chip check and any download, when this install
    lacks part of the extras of the model's install line: no JAX -> no_jax's line; JAX but not the rest (Hy-Nomas on a
    Cloud TPU VM installed with mashbox[tpu] only: no torch) -> those distributions and the line that installs them. A
    model calls it where its run starts needing them (Hy-Nomas: `run`, not --stub or download)."""
    m = hub.model_entry(model)
    if not has_jax():
        return no_jax(cmd, m["id"])
    extras = model_extras(m)
    lack = [d for d in lacking(extras) if d != "jax"]               # JAX: has_jax, which imports nothing
    if not lack:
        return None
    names = lack[0] if len(lack) == 1 else ", ".join(lack[:-1]) + f" and {lack[-1]}"
    return (f"mashbox: {cmd} {m['id']} needs {names} (mashbox[{extras}]), which this install does not have. "
            f"Install {'it' if len(lack) == 1 else 'them'}, then run the same command again: {install_line(extras, m)}")


def not_in_version(mid):
    from . import __version__
    return f"{mid} is not in mashbox {__version__}. Models in this version: https://mashbox.gg/docs/models"


def no_jax(cmd, model):
    """the one line `mashbox <cmd> <model>` stops with on an install without JAX: a models.json model that runs on a TPU
    -> `mashbox tpu up` and its Cloud TPU VM install line; another models.json model -> the line that installs it here; a
    GGUF file, a Hugging Face repo or folder, or any other name -> the CPU install"""
    head = f"mashbox: {cmd} {model} needs JAX, which this install does not have (it is the client for mashbox tpu)."
    mid = model.replace(".", "-").lower()
    m = None if cmd == "inspect" else next((r for r in hub.registry()["models"] if r["id"] == mid), None)
    if m is None:
        return (f"{head} For GGUF files and Hugging Face models on this machine, install it with: {install_line('cpu')}. "
                "This version's models: mashbox models")
    if mid not in RUNNABLE:
        return not_in_version(mid)
    if m.get("cloud"):
        return (f"{head} From this machine: mashbox tpu up {mid}. On a Cloud TPU VM: "
                f"{m.get('install') or hub.registry()['install']}")
    return f"{head} Install it with: {install_line(model_extras(m), m)}"


def cmd_env(a):
    """versions and chip, for bug reports; never a traceback (an install without JAX prints jax null and why)"""
    import platform
    from . import __version__
    out = dict(mashbox=__version__, python=platform.python_version())
    if not has_jax():
        out.update(jax=None, jaxlib=None, libtpu=None, devices="none: this install has no JAX (it is the client for mashbox tpu)")
        print(json.dumps(out, indent=1))
        return
    try:
        from . import kernels as K
        out.update(K.versions())
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
    if dl and any(DR.gate(dl, chip, (g or {}).get("component_set_sha"))[0] for chip, g in (dl.get("gate") or {}).items()):
        try:                            # only a drafter that run's no-flag gate can turn on: DR.gate (passed, sealed, k > 0)
                                        # with the record's own component set standing in, as cli.load does
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


def forced_plain():
    """The test-only speculation-off force (runtime/faults.py): MASHBOX_FAULT=force:speculation, or the older
    MASHBOX_FORCE_FAIL=speculation, turns speculation off at load, before the engine and its warm-up exist, whatever the
    flags ask. -> the why line, or None."""
    from .runtime import faults
    if faults.forced("speculation"):
        return "forced failure (MASHBOX_FAULT=force:speculation): plain decoding"
    if os.environ.get("MASHBOX_FORCE_FAIL") == "speculation":
        return "forced failure (MASHBOX_FORCE_FAIL=speculation): plain decoding"
    return None


def speculation(a, m, cfg, tp, component_set_sha=None, later=False, model=None):
    """Engine arguments for speculative decoding with the model's drafter (models.json `drafter`).

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
    from .runtime.spec_step import SPEC_STEP_VERSION
    dr = DR.declared(m)
    kmax = dr["k_max"] if dr else 3
    k = getattr(a, "mtp", None)
    sv, dk = getattr(a, "speculation", None), getattr(a, "draft_k", None)
    if sv is not None and k is not None:
        raise SystemExit("--mtp K is the short form of --speculation mtp --draft-k K: give one of them")
    forced_why = forced_plain()
    if forced_why:
        a.spec_why = forced_why
        log(json.dumps(dict(event="speculation", used=False, reason=forced_why)))
        return {}
    if sv == "off":
        k = 0
    elif sv == "mtp":
        if dr is None:
            raise SystemExit(f"--speculation mtp: {m['id']} has no MTP head in models.json")
        k = dk if dk is not None else dr["k_default"]
    elif sv in ("ngram", "draft"):
        return forced_drafter(sv, dk, tp, model, cfg)
    elif sv == "dflash":
        return dflash_drafter(a, dk, tp, model, cfg)
    if k is not None and not (1 if sv == "mtp" else 0) <= k <= kmax:      # --draft-k 0 is never off
        raise SystemExit(f"--draft-k {k}: use 1 to {kmax}" if sv == "mtp" else f"--mtp {k}: use 0 (off) to {kmax}")
    mfile = getattr(a, "mtp_file", None)

    def off(why):
        log(json.dumps(dict(event="speculation", used=False, reason=why)))
        a.spec_why = why            # the reason Engine.spec_health gives while decoding is plain (Engine spec_why)
        return {}
    if k == 0:
        return off("--speculation off" if sv == "off" else "--mtp 0")
    if k is None and mfile:
        raise SystemExit(f"--mtp-file: give --mtp K (1 to {kmax}) with it")
    chip = DR.chip_key()
    css = component_set_sha(DR.key(dr)) if (component_set_sha and dr) else None
    gcss = DR.gated_css(dr, chip) if (later and k is None and css is None) else None
    css = gcss if gcss is not None else css
    gated, why = DR.gate(dr, chip, css)
    from .runtime.drafters import select as SEL
    head = SEL.mtp_head(model)        # the MTP head module the model's architecture declares (SPEC_MTP_HEAD), or None
    try:
        kt = DR.k_table(dr, chip)
    except ValueError as e:
        if k is None:
            return off(str(e))
        raise SystemExit(f"--mtp {k}: {e}")
    why0 = f"the MTP drafter needs every committed token, so it runs no row bucket at k = 0 (k_table {kt})"
    if k is None:
        if not gated:
            return off(why)
        if 0 in kt.values():
            return off(why0 + ": plain decoding")
        if tp > 1:
            return off("speculation runs on one chip (no tensor parallel)")
        if head is None:
            return off("the model's architecture declares no MTP head implementation (SPEC_MTP_HEAD): plain decoding")
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
        if head is None:
            raise SystemExit(f"--mtp {k}: the model's architecture declares no MTP head implementation (SPEC_MTP_HEAD)")
        try:
            path = hub.draft_file(m, path=mfile, log=log)
        except SystemExit:                  # a sha256 mismatch: never run on a file that is not the pinned one
            raise
        except Exception as e:              # not downloadable (not in the repo at this revision, offline)
            raise SystemExit(f"--mtp {k}: the draft file could not be fetched: {type(e).__name__}: {e}"[:400])
        if not gated:
            log(f"speculation: {why}" if why.startswith("not gated on this chip") else f"speculation: not gated on this chip ({why})")
        if 0 in kt.values():
            log(f"speculation: {why0}: k = {k} at every row bucket")
            kt = {}
    import importlib
    QSP = importlib.import_module(head)
    Wm, _ = QSP.load_mtp(path, cfg, log=lambda i: log(json.dumps(dict(event="mtp_weights", **i))))
    log(json.dumps(dict(event="speculation", used=True, method="mtp", draft_k=k, k_table=kt, gated=gated, reason=why,
                    spec_step_version=SPEC_STEP_VERSION)))
    return dict(mtp=(QSP, Wm), draft_k=k, k_table=kt, drafter_key=drafter_identity(m, dr, path, k, kmax, mfile),
                **({"gate_css": gcss} if gcss is not None else {}))


def drafter_identity(m, dr, path, k, kmax, mfile=None):
    """The drafter's part of the deployment's component set (runtime/drafter.key: kind, id@version, weights_sha256, k
    rule) for the draft file this start loaded (path), bound to its bytes: weights_sha256 is the
    pinned sha256 hub.draft_file checked the file against (every download; a --mtp-file unless MASHBOX_VERIFY=0), else
    the sha256 of the file's own bytes. A drafter models.json does not declare keys as kind "mtp", the file's name and
    this k."""
    from . import hub
    from .runtime import drafter as DR, tenancy
    checked = (hub.draft_row(m) or {}).get("sha256") if not mfile or os.environ.get("MASHBOX_VERIFY", "1") != "0" else None
    base = dr or dict(kind="mtp", id=os.path.basename(path), version=0, k_default=k, k_max=kmax)
    return DR.key(dict(base, sha256=checked or tenancy.file_sha256(path)))


DRAFT_GGUF_ONLY = "--speculation draft runs on a GGUF file in this version; for a models.json model use ngram or mtp"


def draft_refusal(a, gguf=None):
    """--speculation draft / --draft-model FILE: a draft model drafts for a GGUF target only, given with --draft-model
    (--draft-model alone means --speculation draft). Where that does not hold every run path stops with this line before
    any file is fetched or loaded or the server starts (main before the model row, cmd_run before the API keys, load()
    and cmd_gguf/load_gguf first); else None. gguf: whether the target is a GGUF file (default: a.gguf is set).
    --speculation dflash / --drafter-dir DIR stop here the same way (dflash_refusal)."""
    why = dflash_refusal(a, gguf)
    if why:
        return why
    sv, dm = getattr(a, "speculation", None), getattr(a, "draft_model", None)
    if dm and sv not in (None, "auto", "draft"):
        return f"--draft-model is the draft model of --speculation draft, not of --speculation {sv}"
    if sv != "draft" and not dm:
        return None
    if not (bool(getattr(a, "gguf", None)) if gguf is None else gguf):
        return DRAFT_GGUF_ONLY
    if not dm:
        return "--speculation draft needs --draft-model FILE: a smaller GGUF file or repo with the same tokenizer"
    return None


DFLASH_GGUF = ("--speculation dflash runs on a model from the catalog whose forward gives the drafter its layer outputs, "
               "not on a GGUF file")


def dflash_refusal(a, gguf=None):
    """--speculation dflash / --drafter-dir DIR: the DFlash drafter runs from the folder --drafter-dir gives, on a model
    from the catalog only (a GGUF forward gives no layer outputs). The line every run path stops with (draft_refusal
    calls this first), else None. gguf: whether the target is a GGUF file (default: a.gguf is set)."""
    sv, dd = getattr(a, "speculation", None), getattr(a, "drafter_dir", None)
    if dd and sv != "dflash":
        return f"--drafter-dir is the drafter of --speculation dflash, not of --speculation {sv or 'auto'}"
    if sv != "dflash":
        return None
    if bool(getattr(a, "gguf", None)) if gguf is None else gguf:
        return DFLASH_GGUF
    if not dd:
        return ("--speculation dflash needs --drafter-dir DIR: the DFlash drafter's folder (config.json and "
                "model.safetensors); docs/speculation.md names the drafter to download for each model")
    return None


def dflash_drafter(a, k, tp, model, cfg):
    """--speculation dflash --drafter-dir DIR on a model from the catalog: the DFlash block drafter
    (runtime/drafters/dflash.py) from that folder, where the module that defines the model's forward declares
    LAYER_TAPS (the forward returns layer outputs for a drafter), never by the model's name.
    --draft-k 1 to the drafter's block less one (default: all of them). One chip only. auto never picks it and it has no
    gate record in this version, so it always says "not gated on this chip" and such a deployment is never certified."""
    from .runtime import tenancy
    from .runtime.drafters import dflash as DF
    from .runtime.spec_step import SPEC_STEP_VERSION
    why = dflash_refusal(a, gguf=False)
    if why:
        raise SystemExit(why)
    if tp > 1:
        raise SystemExit("--speculation dflash: speculative decoding runs on one chip (no tensor parallel)")
    fwd = sys.modules.get(getattr(getattr(model, "make_forward", None), "__module__", None) or "")
    if not (getattr(model, "LAYER_TAPS", False) or getattr(fwd, "LAYER_TAPS", False)):
        raise SystemExit("--speculation dflash: this model's forward does not give a drafter its layer outputs "
                         "(its architecture declares no LAYER_TAPS)")
    d = a.drafter_dir
    if not os.path.isfile(os.path.join(d, "config.json")):
        raise SystemExit(f"--drafter-dir {d}: no config.json there; give the drafter's folder (config.json and model.safetensors)")
    try:
        bs = DF.Config(d).bs
    except (NotImplementedError, KeyError, ValueError) as e:
        raise SystemExit(f"--drafter-dir {d}: not a DFlash 2 drafter this version runs: {e}")
    k = bs - 1 if k is None else k
    if not 1 <= k <= bs - 1:
        raise SystemExit(f"--draft-k {k}: use 1 to {bs - 1} (the drafter's block of {bs} less its first token)")
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    sha = tenancy.file_sha256(os.path.join(d, files[0])) if len(files) == 1 else None    # the deployment's drafter key
    try:
        dr = DF.DFlash(model, cfg, d, k, weights_sha256=sha)
    except (NotImplementedError, KeyError, ValueError) as e:
        raise SystemExit(f"--speculation dflash: {e}")
    log("speculation: not gated on this chip")
    log(json.dumps(dict(event="speculation", used=True, method="dflash", draft_k=k, gated=False, reason="--speculation dflash",
                        spec_step_version=SPEC_STEP_VERSION)))
    return dict(drafter=dr, draft_k=k)


def dflash_slots(a):
    """--speculation dflash at --slots' default (8) runs 1 slot: the drafter's weights and its K/V ring per slot come on
    top of the model's, and on a TPU v6e-1 the 8-bit Qwen3.8-27B with the drafter at 8 slots ran out of memory where 1
    slot fits. --slots N runs as given."""
    if getattr(a, "speculation", None) == "dflash" and a.slots == 8:
        log("mashbox: --speculation dflash runs one request at a time (--slots 1): with the drafter, 8 slots did not fit "
            "a TPU v6e-1's memory. --slots N sets it")
        log(json.dumps(dict(event="dflash_slots", slots=1, asked=8)))
        a.slots = 1


def forced_drafter(kind, k, tp, model, cfg):
    """--speculation ngram | draft on a models.json model: the drafter where it applies to the model's metadata (select.py:
    a verify pass, not MoE), not gated on this chip, so never certified"""
    from .runtime.drafters import select as SEL
    from .runtime.spec_step import SPEC_STEP_VERSION
    if kind == "draft":
        raise SystemExit(DRAFT_GGUF_ONLY)
    k = 3 if k is None else k
    if not 1 <= k <= 8:
        raise SystemExit(f"--draft-k {k}: use 1 to 8")
    if tp > 1:
        raise SystemExit(f"--speculation {kind}: speculative decoding runs on one chip (no tensor parallel)")
    ok, why = SEL.applies("ngram", SEL.meta_of(model, cfg, "bshj"))
    if not ok:
        raise SystemExit(f"--speculation {kind}: {why}")
    from .runtime.drafters.ngram import NGram
    log("speculation: not gated on this chip")
    log(json.dumps(dict(event="speculation", used=True, method=kind, draft_k=k, gated=False, reason="--speculation " + kind,
                        spec_step_version=SPEC_STEP_VERSION)))
    return dict(drafter=NGram(k), draft_k=k)


def gguf_speculation(a, c, A, records=None, f=None):
    """Engine arguments for speculative decoding on a GGUF file, picked from its metadata (runtime/drafters/select.py):
    auto (no flag) runs only a drafter with a passed gate record on this chip (records {kind: {chip: entry}}), else plain
    decoding with the why lines; --speculation <kind> forces one where it applies (--draft-model FILE: the draft model,
    loaded by gguf_draft against the target file f, which stops with its line when the file cannot draft for it); --speculation off and
    the speculation-off force (forced_plain) decode plain. A gated pick carries the record's component_set_sha as gate_css:
    load_gguf checks the built engine's against it before the warm-up (gguf_spec_gate)."""
    from .gguf import GGUFError
    from .runtime import drafter as DR
    from .runtime.drafters import select as SEL
    from .runtime.spec_step import SPEC_STEP_VERSION
    sv, k = getattr(a, "speculation", None), getattr(a, "draft_k", None)
    if sv == "dflash":
        raise GGUFError(f"mashbox: {DFLASH_GGUF}")
    given = getattr(a, "draft_gguf", None)
    forced = "draft_model" if sv == "draft" or (given and sv in (None, "auto")) else sv

    def off(why):
        log(json.dumps(dict(event="speculation", used=False, reason=why)))
        a.spec_why = why            # the reason Engine.spec_health gives while decoding is plain (Engine spec_why)
        return {}
    forced_why = forced_plain()
    if forced_why:
        return off(forced_why)
    chip = DR.chip_key()
    kind, gated, why, gcss = SEL.pick(SEL.meta_of(A, c, "gguf", getattr(c, "nextn", 0), draft_given=bool(given)), chip,
                                      records=records, forced=forced, later=True)
    if kind is None:
        if forced not in (None, "auto", "off"):
            raise GGUFError(f"mashbox: --speculation {sv}: {why[-1] if why else 'does not apply'}")
        return off("; ".join(why))
    kt = {}
    if gated:       # the record's measured draft length per row bucket reaches the engine (k = 0: that bucket runs plain)
        ent = ((records or {}).get(kind) or {}).get(chip) or {}
        try:
            kt = DR.k_table(dict(k_max=8, k_table={chip: ent.get("k_table") or {}}), chip)
        except ValueError as e:
            return off(f"the gate record's {e}")
        k = max(kt.values()) if (k is None and kt) else k
    k = 3 if k is None else k
    if not 1 <= k <= 8:
        raise GGUFError(f"mashbox: --draft-k {k}: use 1 to 8")
    if kind == "draft_model":       # a draft model you gave stops with its line when it cannot draft for this file
        dr = gguf_draft(a, f, c, k)
    else:
        try:
            from .runtime.drafters.ngram import NGram
            dr = NGram(k)
        except Exception as e:      # the base model serves as without a drafter
            return off(f"the drafter could not be built: {type(e).__name__}: {e}"[:300])
    if not gated:
        log("speculation: not gated on this chip")
    log(json.dumps(dict(event="speculation", used=True, method=kind, draft_k=k, k_table=kt, gated=gated, reason=why[-1],
                        spec_step_version=SPEC_STEP_VERSION)))
    return dict(drafter=dr, draft_k=k, k_table=kt, **({"gate_css": gcss} if gcss is not None else {}))


def gguf_draft(a, f, c, k):
    """The draft-model drafter (runtime/drafters/draft_model.py) for --draft-model on a GGUF target (f, c): the draft file
    a.draft_gguf opened as the target is, on its own architecture module, which must declare SPEC_DRAFT (full attention
    in every layer, so its cache needs no rollback), with the target's tokenizer (the same tokenizer.ggml.tokens) and no
    experts; its weights loaded on this chip. Its identity in the deployment's component set is the draft file's sha256
    (every shard), as gguf_deployment keys the target. Anything else stops with one line (GGUFError)."""
    from . import gguf as G
    from .runtime.drafters.draft_model import DraftModel
    d = a.draft_gguf
    f2, A2, c2 = G.open_model(d["path"], repo=d["repo"], quant=d["tag"])
    if not getattr(A2, "SPEC_DRAFT", False):
        raise G.GGUFError(f"mashbox: --draft-model: a {f2.arch} file cannot draft: a draft model needs full attention in "
                          f"every layer (GGUF {', '.join(sorted(G.archs.drafting()))})")
    if f is None or list(f2.get("tokenizer.ggml.tokens") or []) != list(f.get("tokenizer.ggml.tokens") or []) \
            or int(c2.V) != int(c.V):
        raise G.GGUFError("mashbox: --draft-model: its tokenizer is not the target's (a draft model needs the same tokens)")
    if getattr(c2, "E", 0) or getattr(c2, "moe", False):
        raise G.GGUFError("mashbox: --draft-model: a mixture-of-experts draft model is not supported")
    line, warn = G.plan(f2, c2, kernel_types=frozenset() if a.no_kernels else None)
    log(json.dumps(dict(event="draft_plan", **line)))
    if warn:
        from . import ops
        ops.warn(warn.replace("mashbox: ", "mashbox: --draft-model: ", 1))
    W2, info = A2.load(f2, c2, use_kernels=not a.no_kernels, log=lambda i: log(json.dumps(dict(event="draft_weights", **i))))
    sha, _ = gguf_file_sha(d)
    ident = f"{d['repo']}/{d['file']}" if d.get("repo") else d["file"]
    log(json.dumps(dict(event="draft_model", file=ident, sha256=sha, arch=f2.arch, layers=c2.L,
                        gb=round(info["device_bytes"] / 1e9, 2))))
    return DraftModel(A2, c2, W2, k, declared=dict(id=ident, weights_sha256=sha))


def gguf_spec_gate(eng, gate_css, keys):
    """Speculation picked by a gate record (gguf_speculation's gate_css) stays on only when the built engine's own
    component_set_sha, which holds the drafter's key, is the record's: else it is turned off before the warm-up (plain
    decoding, one why line) and keys() makes the deployment keys again."""
    if gate_css is None or not getattr(eng, "DK", 0) or eng.component_set_sha == gate_css:
        return
    why = "the gate record is for another component set than this deployment's: plain decoding"
    eng.drop_speculation()
    eng.spec_why = why                  # the reason Engine.spec_health gives
    log(json.dumps(dict(event="speculation", used=False, reason=why)))
    keys()


def deployment_keys(eng, m, config_sha, layout_sha, kernel_bundle_sha=None, tp=1, gate_css=None, mesh=None, drafter=None):
    """Set eng.deployment_id and eng.component_set_sha (runtime/compile_key.py: with the chip's topology, tensor parallel
    and its mesh). While speculation is on, the drafter's identity is part of the component set: `drafter` is the
    drafter_key speculation() returned for the file it loaded (drafter_identity); None = the engine's drafter's own key
    (a drafter that keys itself, e.g. n-gram), else the model's declared drafter (drafter.key).
    When speculation came from the drafter's gate (gate_css, speculation(later=True)) and this deployment's
    component_set_sha is another one, speculation is turned off before the warm-up (one log line) and the keys are made
    again without the drafter. -> the compile key."""
    from .runtime import compile_key as CK, drafter as DR

    def keys():
        own = getattr(getattr(eng, "drafter", None), "key", lambda: None)     # the engine's drafter keys itself (n-gram)
        dk = (drafter if drafter is not None else (own() or DR.key(DR.declared(m)))) if getattr(eng, "DK", 0) else None
        return CK.for_engine(eng, m["id"], config_sha, layout_sha, kernel_bundle_sha=kernel_bundle_sha, tp=tp, drafter=dk,
                             mesh=mesh)
    key, eng.deployment_id, eng.component_set_sha = keys()
    if gate_css is not None and getattr(eng, "DK", 0) and eng.component_set_sha != gate_css:
        log(json.dumps(dict(event="speculation", used=False, reason=f"gated on {DR.chip_key()} for another component set")))
        eng.drop_speculation()
        eng.spec_why = f"gated on {DR.chip_key()} for another component set: plain decoding"     # Engine.spec_health's reason
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
    devs = jax.devices()
    try:
        have = min(int(x.memory_stats()["bytes_limit"]) for x in devs)
    except Exception:
        return None
    wb = os.path.getsize(a.weights) if a.weights else int(hub.weights_row(m)["bytes"])
    Q = engine_package(m)[0]                  # the model definition the row's runtime package holds
    try:
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
    if draft_refusal(a):
        raise SystemExit(draft_refusal(a))              # before the config, the weights or any cache work
    from . import hub
    m = hub.model_entry(a.model)
    Q, QW, QS = engine_package(m)                        # the model definition the row's runtime package holds
    d = hub.upstream_dir(m, log)                         # config and tokenizer only (small)
    cache = setup_cache(a.cache_dir)
    tp = tensor_parallel(getattr(a, "tensor_parallel_size", 0))
    if tp > 1 and QS is None:                            # a package with no tensor-parallel split runs on one chip
        if getattr(a, "tensor_parallel_size", 0):
            raise SystemExit(f"mashbox: {m['id']} runs on one chip in this version (its package has no shard module); "
                             f"use --tensor-parallel-size 1")
        log(json.dumps(dict(event="tensor_parallel", tp=1, reason="the package has no shard module: one chip")))
        tp = 1
    dflash_slots(a)                                      # before the fit check and the kernels (built for 8 slots)
    fit_check(a, m, d, tp)                               # before the weights download, check or load
    w = hub.weights_file(m, path=a.weights, log=log)
    from . import kernels as K
    from .runtime.engine import Engine
    from .runtime.chat import Chat
    hub.compile_cache(m, cache, log, tp=tp)            # the cache built for this accelerator and tensor-parallel size
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
        mesh = QS.mesh_for(tp)
        W, info = QS.load(w, cfg, mesh, log=lambda i: log(json.dumps(dict(event="weights", **i))))
        model = QS.TPModel(Q, mesh)
    else:
        W, info = QW.load(w, cfg, log=lambda i: log(json.dumps(dict(event="weights", **i))))
    spec = speculation(a, m, cfg, tp, later=True, model=model)
    gate_css, drafter = spec.pop("gate_css", None), spec.pop("drafter_key", None)
    eng = Engine(cfg, W, model, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=dk, spec_why=getattr(a, "spec_why", None), **spec,
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
    this machine and user. None (no table) with MASHBOX_AOT=0, with speculative decoding on: the warm-up compiles its
    programs through jit, outside the table, so a table would not hold every program this deployment runs; and with the
    record-only top-k hook on (eng.topk, MASHBOX_RECORD_TOPK): its traced programs also return top-k, which the compile
    key does not name, so a table saved without the hook must not be loaded for them."""
    why = ("MASHBOX_AOT=0" if os.environ.get("MASHBOX_AOT") == "0" else
           "speculative decoding is on: the table does not hold its programs" if getattr(eng, "DK", 0) else
           "the record-only top-k hook is on: the compile key does not name top-k" if getattr(eng, "topk", 0) else None)
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
    """`run` for a GGUF file or repo: the file's own blocks on TPU (mashbox.ops), its tokenizer and chat template. An
    MoE file with no passing chip row of its own (gguf.archs.held) is refused with one line before anything is loaded"""
    t0 = time.time()
    from . import gguf as G
    if draft_refusal(a):
        raise G.GGUFError(f"mashbox: {draft_refusal(a)}")
    r = a.gguf
    f, A, c = G.open_model(r["path"], repo=r["repo"], quant=r["tag"])
    held = G.archs.held(f.arch, r["path"], r["repo"], r["tag"], moe=G.archs.moe_file(f))   # a MoE file with no passing row of its own: one line, nothing loaded
    if held:
        raise G.GGUFError(held)
    from .gguf import tokenizer as GT
    if GT.no_tokenizer(f, r["path"]):                   # no tokenizer in the file: one line, nothing loaded
        raise G.GGUFError(GT.no_tokenizer(f, r["path"]))
    return _load_gguf_for_record(a, f, A, c, t0)


def _load_gguf_for_record(a, f, A, c, t0):
    """load_gguf below the hold: the opened file f (its arch module A, config c) loaded as `run` loads it. `run` reaches
    it only through load_gguf; the chip record runner (tools/gguf_chip_record.py) calls it directly, because a held file
    needs a chip record before it can have a passing catalog row. Internal, not a supported way to load a file: the hold
    is the support gate of `run`, serving and `inspect` (`tpu up` takes catalog models only and starts `mashbox run` on
    the chip), not a same-process boundary (any Python code that imports this module can call this function)"""
    from . import gguf as G, ops
    from .gguf import tokenizer as GT
    from .runtime.engine import Engine
    from .runtime.chat import Chat
    r = a.gguf
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
    from . import hub
    from .runtime.drafters import select as SEL
    spec = gguf_speculation(a, c, A, records=SEL.catalog_records(hub.registry(), r), f=f)   # the catalog's sealed chip records
    gate_css = spec.pop("gate_css", None)
    eng = Engine(c, W, A, CTX=a.ctx, S=a.slots, K=a.steps, decode_kernels=None,
                 prefix_cache=a.prompt is None and not a.no_prefix_cache,
                 cache_ram=max(0, a.cache_ram) << 20, ckpts=a.ctx_checkpoints, ckpt_step=a.checkpoint_min_step,
                 attn_bytes=2e9 if a.ctx <= 8192 else 5e8,
                 trace=os.environ.get("MASHBOX_TRACE") == "1", spec_why=getattr(a, "spec_why", None), **spec)   # trace: per-token logit hashes for checks, as load()
    gguf_deployment(eng, f, r, W)
    gguf_spec_gate(eng, gate_css, lambda: gguf_deployment(eng, f, r, W))
    log(json.dumps(dict(event="loaded", weights_s=info["load_s"], gb=round(info["device_bytes"] / 1e9, 2), bpw=info["bpw"],
                        kernel_gb=round(info["kernel_bytes"] / 1e9, 2), reference_gb=round(info["reference_bytes"] / 1e9, 2),
                        resident_f32_gb=round(info["resident_f32_bytes"] / 1e9, 2))))
    return dict(id=r["repo"] or r["file"]), eng, chat, info, t0


def file_identity(path):
    """-> (sha256 of a weights file, how it was found): gguf.file_identity (the file pick's, gguf/resolve.py)"""
    from . import gguf as G
    return G.file_identity(path)


def gguf_file_sha(r):
    """-> (the sha256 of a resolved GGUF file r: of its bytes, or for a split file of its shards' sha256s, [(sha256, how
    it was found) per shard])"""
    import hashlib
    from . import gguf as G
    shards = G.shards(os.path.basename(r["path"]))
    paths = [os.path.join(os.path.dirname(r["path"]), s) for s in shards] if len(shards) > 1 else [r["path"]]
    ids = [file_identity(p) for p in paths]
    return (ids[0][0] if len(ids) == 1 else hashlib.sha256(" ".join(i[0] for i in ids).encode()).hexdigest()), ids


def gguf_deployment(eng, f, r, W):
    """The GGUF file's deployment, as load() makes one for a models.json model (runtime/compile_key.py): deployment id
    and component set (/health, the startup log), and the prefix cache namespace. The file's sha256 (every shard of a
    split file) stands for its weights, tokenizer and chat template, which it all holds."""
    import hashlib
    from . import gguf as G
    from .runtime import tenancy, compile_key as CK
    t = time.time()
    fsha, n, how = G.identity(r["path"])
    log(json.dumps(dict(event="gguf_identity", sha256=fsha, shards=n, how=how, seconds=round(time.time() - t, 1))))
    mid = r["repo"] or r["file"]
    _, eng.deployment_id, eng.component_set_sha = CK.for_engine(eng, mid, fsha, CK.weights_layout(W), kernel_bundle_sha=None, tp=1,
                                                              drafter=eng.drafter.key() if getattr(eng, "DK", 0) else None)
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
    """`mashbox inspect MODEL`: the resolver's plan (one log line, its why lines), then what the target holds: a GGUF
    file's header (gguf.inspect), a safetensors model's config and headers (hf.inspect), or a catalog row's runtime and
    files. Nothing is loaded; a repo's GGUF file is downloaded unless --no-download."""
    from . import resolve as R
    plan = R.resolve(a.model, download=False, log=log)
    log(json.dumps(dict(event="plan", format=plan.format, why=plan.why)))
    if plan.format == "gguf":
        if a.no_download and plan.source["path"] is None:
            return 0
        from . import gguf as G
        s = R.fetch(plan, log).source
        print(json.dumps(G.inspect(s["path"], repo=s["repo"], quant=s["tag"]), indent=1))
    elif plan.format == "safetensors":
        from . import hf as H
        print(json.dumps(H.inspect(a.model, log=log), indent=1))
    else:
        m = plan.row
        print(json.dumps(dict(model=m["id"], name=m.get("name"), hf_repo=m.get("hf_repo"), runtime=plan.module,
                              entry=plan.entry, files=[f["name"] for f in m.get("files") or []],
                              chips={c: r.get("status") for c, r in (m.get("chips") or {}).items()}), indent=1))
    return 0


def warm(eng, chat):
    """Compile the serving path's programs before ready (served from the compile cache when it has them): two short
    requests (the request path end to end), then each prefill, decode, speculation and helper program (Engine.warm).
    Counts every compile from here on (runtime/programs.py). A table folder the load refused, or the warm-up dropped for
    another program list, is moved aside (programs.set_aside) so the save writes a new table there; after a save,
    programs.prune removes the tables no start needs. A program loaded from the table that fails in the two requests or
    in Engine.warm raises programs.TableFailed out of here (the table quarantined and dropped; info["closed"] says
    whether its folder could be closed): the caller ends this process, which never warms a second time, and the
    next start compiles without the table (worker.main sends the table's fatal frame to the front; cmd_run exits 70).
    Any other failure while speculation is on leaves the plain engine (drop_speculation), which warms again."""
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

    def run():
        for samp in (None, [dict(temperature=0.7, top_p=0.9, seed=1)]):
            eng.generate([ids], max_new=eng.K + 1, stop_ids=(), samp=samp)
        return eng.warm(workers=nw)
    try:
        w = run()
    except Exception as e:      # fallback: speculation that fails before ready leaves the plain engine, which warms again
        if isinstance(e, programs.TableFailed) or not getattr(eng, "DK", 0):
            raise               # a table program's failure ends this process, which never warms a second time
        eng.drop_speculation(failed=True)
        eng.spec_why = f"a speculation program failed before ready: {type(e).__name__}: {e}"[:300] + "; plain decoding"
        log(json.dumps(dict(event="speculation", used=False,
                            reason=f"a speculation program failed before ready: {type(e).__name__}: {e}"[:300] + "; plain decoding")))
        w = run()
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
    if draft_refusal(a):
        log(f"mashbox: {draft_refusal(a)}")
        return 2                                         # before an API key is made or the server starts
    keys = api_keys(a) if a.prompt is None else []       # before the load: a bad key file fails at once
    from . import resolve as R
    plan = getattr(a, "plan", None) or R.given(a) or R.resolve(a.model, download=False, log=log)   # main() resolved it
    if a.prompt is None and plan.format == "catalog":
        # serving a models.json model: this process is the HTTP front; the engine (load, warm-up, chip) runs in a
        # worker process that the front restarts by itself after a device fault, out of memory, a hung call or a kill
        # (runtime/front.py). A GGUF file or safetensors model is still served in this one process (below).
        from .runtime.front import serve_cli          # serve() logs the "ui" url once the port is bound
        return serve_cli(a, sys.argv[1:] if argv is None else list(argv), keys, log, t0=time.time())
    m, eng, chat, info, t0 = R.load(plan, a)
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
        except programs.TableFailed as e:       # quarantined and dropped: this process never warms a second time
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
        p.add_argument("model", help="a model in `mashbox models` by its id, name or Hugging Face repo (e.g. qwen3-8-27b or "
                       "mashbox-labs/Qwen3.8-27B-TPU), a .gguf file, a Hugging Face GGUF repo with an optional quant tag "
                       "(e.g. unsloth/Qwen3.8-27B-GGUF or unsloth/Qwen3.8-27B-GGUF:Q8_0), or a model folder or Hugging "
                       "Face repo with config.json and safetensors weights (e.g. Qwen/Qwen3-4B)")
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
            p.add_argument("--speculation", choices=("auto", "off", "mtp", "ngram", "draft", "dflash"),
                           help="speculative decoding. auto (default): the first drafter that applies to the model file, in this "
                                "order: 1 a trained drafter made for this model (auto picks none in this release); 2 the model's own MTP "
                                "head; 3 a smaller draft model with the same tokenizer (only one you give with --draft-model); "
                                "4 n-gram over the request's own text (llama, qwen2, qwen3, qwen35, exaone and granite GGUF files), on only where it passed its checks on this chip (Qwen3.8-27B reads its MTP head "
                                "only); off; or force one kind (then it says 'not gated on this chip'). dflash: the DFlash block drafter "
                                "from --drafter-dir (Qwen3.8-27B), never picked by auto. One chip only")
            p.add_argument("--draft-k", type=int, metavar="K",
                           help="drafts per verify pass for a drafter you pick: ngram 1 to 8 (default 3); draft 1 to 8 (default 3); "
                                "mtp 1 to the head's k_max (Qwen3.8-27B: 1 to 3, default 2); dflash 1 to 7 (default 7)")
            p.add_argument("--draft-model", metavar="FILE",
                           help="--speculation draft with this draft model: a smaller GGUF file or repo (as the model argument) "
                                "with the same tokenizer, a llama, qwen2, qwen3, exaone or granite file; for a GGUF model "
                                "only. Its weights and cache take chip memory next to the model's")
            p.add_argument("--drafter-dir", metavar="DIR",
                           help="--speculation dflash with the DFlash drafter in DIR: its config.json and model.safetensors "
                                "(for Qwen3.8-27B, the Hugging Face repo z-lab/Qwen3.8-27B-DFlash2). It runs one request at "
                                "a time unless --slots N is given")
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
    p = sub.add_parser("inspect", help="how a model resolves (one line per decision) and what it holds: a GGUF file's "
                       "architecture, block types and tokenizer, a safetensors model's config, or a catalog model's "
                       "runtime and files; nothing is loaded")
    p.add_argument("model", help="what `run` takes: a model in `mashbox models`, a .gguf file or GGUF repo[:quant tag], or "
                   "a model folder or safetensors repo")
    p.add_argument("--no-download", action="store_true", help="for a GGUF repo: only show which file llama.cpp's rule picks")
    return ap


def main(argv=None):
    """`mashbox <command>`. A download that fails ends the command with hub.DownloadError's one line and exit 1."""
    try:
        return dispatch(sys.argv[1:] if argv is None else list(argv))
    except hub.DownloadError as e:
        log(str(e))
        return 1


def dispatch(argv):
    if argv[:1] == ["tpu"]:
        from . import tpu
        return tpu.main(argv[1:])
    if len(argv) >= 2 and argv[0] in ("run", "download"):
        from . import resolve as R
        own = R.own_command_line(argv[1])     # a catalog package that parses its own options (its main(cmd, argv))
        if own:
            import importlib
            try:
                return importlib.import_module(own).main(argv[0], argv[2:])
            except ModuleNotFoundError as e:    # an install without JAX: the model's package imports it before any download
                if e.name not in ("jax", "jaxlib") or has_jax():
                    raise
                log(no_jax(argv[0], argv[1].replace(".", "-").lower()))
                return 2
    a = build_parser().parse_args(argv)
    if a.cmd == "env":
        return cmd_env(a)
    if a.cmd == "models":
        return cmd_models(a)
    if not has_jax():                                    # run, download and inspect read every model with JAX
        log(no_jax(a.cmd, a.model))
        return 2
    return cmd_target(a, argv)


def host_note(a):
    if getattr(a, "host", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1"):
        log("note: serving on a non-loopback address exposes the server to your network (an API key is required); "
            "prefer an SSH tunnel")


def cmd_target(a, argv=None):
    """run, download and inspect: the target through the one resolver (resolve.py) into its loader. The flags each kind
    of target refuses are checked before any file is fetched; every refusal is one clear line and exit 2, never a stack
    trace. A models.json model's own problems (fit, sha256, drafter) still stop as before (SystemExit)."""
    from . import resolve as R
    try:
        if a.cmd == "inspect":
            return cmd_inspect(a)
        if R.is_file_target(a.model):                # a file, folder, repo or GGUF row: the file flags apply
            return cmd_gguf(a, argv)
        if a.cmd != "download" and draft_refusal(a):
            log(f"mashbox: {draft_refusal(a)}")
            return 2                                 # before the model row, an API key or the server
        a.plan = R.resolve(a.model, download=False, log=log)       # a models.json model: nothing is fetched here
        if a.plan.entry == "own":                    # its package parses its own options
            import importlib
            return importlib.import_module(a.plan.module).main(a.cmd, list(argv or [a.cmd, a.model])[2:])
        if a.cmd == "download":
            return cmd_download(a)
        host_note(a)
        return cmd_run(a, argv)
    except R.refusals() as e:
        log(str(e))
        return 2


def cmd_gguf(a, argv=None):
    """files and repos (a GGUF row's id, a GGUF file or repo, a model folder or safetensors repo) through the resolver:
    the flags such a target cannot take stop it before anything is fetched (a folder's and a safetensors repo's are
    hf.run_refusal's); then the plan is fetched and run"""
    from . import hf as HF
    from . import resolve as R
    try:
        folder = os.path.isdir(a.model)              # a model folder: the safetensors path's own refusals, below
        if not folder and getattr(a, "weights", None):
            raise R.ResolveError("mashbox: --weights is for models.json models; pass the .gguf file as the model")
        if not folder and (getattr(a, "mtp", None) or getattr(a, "mtp_file", None)):
            raise R.ResolveError("mashbox: --mtp and --mtp-file are for models.json models; for a GGUF file use --speculation")
        if not folder and a.cmd != "download" and draft_refusal(a, gguf=True):
            raise R.ResolveError(f"mashbox: {draft_refusal(a, gguf=True)}")     # before the file is fetched
        a.plan = R.resolve(a.model, download=False, log=log)
        if a.plan.format == "safetensors" and a.cmd != "download" and HF.run_refusal(a):
            raise R.ResolveError(HF.run_refusal(a))                          # before the weights download
        if a.plan.format != "gguf" and getattr(a, "draft_model", None):
            raise R.ResolveError(f"mashbox: --draft-model drafts for a GGUF file; {a.model} is a {a.plan.format} model")
        R.fetch(a.plan, log)
        if a.plan.format == "gguf":
            a.gguf = a.plan.source
        elif a.plan.format == "safetensors":
            a.hf = a.plan.source
        if getattr(a, "draft_model", None):
            d = R.resolve(a.draft_model, download=True, log=log)
            if d.format != "gguf":
                raise R.ResolveError(f"mashbox: --draft-model {a.draft_model}: a draft model is a GGUF file or repo, "
                                     f"not a {d.format} model")
            a.draft_gguf = d.source
        if a.cmd == "download":
            log(json.dumps(dict(event="downloaded", path=a.plan.source["path"])))
            return 0
        host_note(a)
        return cmd_run(a, argv)
    except R.refusals() as e:
        log(str(e))
        return 2
