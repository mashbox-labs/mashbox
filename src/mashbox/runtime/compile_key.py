"""Compile key and component set of a deployment (which compiled programs and caches may be reused).

Each key is sha256 over canonical JSON (keys sorted, no whitespace, UTF-8).
component_set = {"v": 1, "model", "components", "activation_edits", "flags", "drafter"}: what the model runs (its config
and weight layout, the kernels and sampler, the context and batch shapes, the activation-edit capability (whether the
programs take a table of vectors applied to hidden activations at chosen layers, its shape and dtype), the
program-changing compiler flags, the speculative drafter or null when speculation is off). component_set_sha names it.
compile_key = {"v": 1, "chip", "sw", "flags", "model", "components", "activation_edits", "component_set_sha"}: the
component set plus the chip (with its topology: devices per chip, tensor parallel, mesh shape, the devices it runs on)
and the software that compiles it (engine_commit = "source-sha256:" + the sha256 of the engine's own Python source, so a
changed function body is a new deployment even at the same signatures and version). Its sha is the deployment id. Weight
values and the values of activation-edit vectors are in neither: they change no program. A shipped compile cache carries
the component_set_sha it was built for and is refused when it differs (check_cache).
"""
import functools
import hashlib
import json
import os

import numpy as np

V = 1
FORWARD_ABI = "qwen3_8_27b.forward.contig.v1"       # the forward the engine calls today (paged pool: a new value)
STEP_PROGRAM_VERSION = "engine.step.v1"
SAMPLER = "mashbox.sampler@1"
ATTENTION = "plain-jax.contig@1"
# compiler settings that change programs (environment entries are read by name prefix)
FLAG_ENV = ("XLA_FLAGS", "LIBTPU_INIT_ARGS")


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def flags(env=None):
    """Program-changing compiler settings: XLA_FLAGS / LIBTPU_INIT_ARGS entries (sorted) and the JAX settings that
    change numerics."""
    env = os.environ if env is None else env
    out = {k: sorted(env.get(k, "").split()) for k in FLAG_ENV}
    try:
        import jax
        for k in ("jax_threefry_partitionable", "jax_default_matmul_precision"):
            out[k] = getattr(jax.config, k, None)
    except Exception:
        pass
    out["xla_allow_excess_precision"] = next((f for f in out["XLA_FLAGS"] if "allow_excess_precision" in f), None)
    return out


def weights_layout(W):
    """Names, shapes and dtypes of the weight tree (not values)."""
    import jax
    leaves = jax.tree_util.tree_leaves_with_path(W)
    return sha([[jax.tree_util.keystr(p), list(getattr(x, "shape", ())), str(getattr(x, "dtype", type(x).__name__))]
                for p, x in leaves])


def component_set(model, components, activation_edits=None, flags_=None, drafter=None):
    return {"v": V, "model": model, "components": components,
            "activation_edits": activation_edits or {"enabled": False},
            "flags": flags() if flags_ is None else flags_, "drafter": drafter}


def chip(tp=1, mesh=None):
    """The chip and its topology: family, accelerator type, devices on the host, devices per chip, tensor parallel, mesh
    shape, and the devices the deployment runs on (id, coordinates, core on chip): the mesh's, else the first device."""
    import jax
    d = jax.devices()
    used = list(getattr(mesh, "devices", np.empty(0)).flat) if mesh is not None else d[:1]
    cores = {getattr(x, "core_on_chip", 0) for x in d}
    return {"family": d[0].platform, "accelerator_type": getattr(d[0], "device_kind", None), "devices": len(d),
            "devices_per_chip": len(cores), "tp": tp,
            "mesh_shape": {str(k): int(v) for k, v in dict(mesh.shape).items()} if mesh is not None else None,
            "topology": [[int(getattr(x, "id", -1)), list(getattr(x, "coords", ()) or ()), int(getattr(x, "core_on_chip", 0))]
                         for x in used]}


def source_sha(root=None):
    """sha256 over the engine's Python source: every .py file of the mashbox package (relative path and the sha256 of its
    bytes, in path order). An installed wheel has no git commit; this names the exact code it runs. root = another
    package folder (tests); the installed package's value is computed once per process."""
    return _source_sha(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if root is None else _hash_tree(root)


@functools.lru_cache(maxsize=1)
def _source_sha(root):
    return _hash_tree(root)


def _hash_tree(root):
    paths = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x != "__pycache__"]
        paths += [os.path.join(dp, f) for f in fn if f.endswith(".py")]
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: os.path.relpath(p, root).replace(os.sep, "/")):
        with open(p, "rb") as fh:
            h.update(os.path.relpath(p, root).replace(os.sep, "/").encode() + b"\0")
            h.update(hashlib.sha256(fh.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def sw(engine_commit=None):
    import jax
    import jaxlib
    try:
        from importlib.metadata import version
        lt = version("libtpu")
    except Exception:
        lt = None
    try:
        from importlib.metadata import version
        pkg = version("mashbox")
    except Exception:
        pkg = None
    return {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "libtpu": lt, "mashbox": pkg,
            "engine_commit": engine_commit or "source-sha256:" + source_sha(), "FORWARD_ABI": FORWARD_ABI, "step_program_version": STEP_PROGRAM_VERSION}


def for_engine(eng, model_id, config_sha, weights_layout_sha, kernel_bundle_sha=None, tp=1, drafter=None,
               chip_=None, sw_=None, flags_=None, activation_edits=None, mesh=None):
    """-> (compile_key dict, deployment_id, component_set_sha) for an Engine (runtime/engine.py)."""
    comps = {"kernel_bundle_sha": kernel_bundle_sha, "bindings": {"decode": list(getattr(eng, "kernels", []) or [])},
             "sampler": SAMPLER, "attention": ATTENTION, "kv_format": "contig", "max_ctx": eng.CTX, "slots": eng.S,
             "steps_per_call": eng.K, "chunk": eng.chunk, "tokens_per_call": eng.tokens_per_call,
             "attn_bytes": eng.attn_bytes, "budget_busy": eng.budget_busy, "budget_idle": eng.budget_idle,
             "prefix_cache": bool(eng.prefix_cache), "host_cache": getattr(eng, "host", None) is not None,
             "kv_block": eng.kv_block, "trace": bool(eng.trace), "tp": tp,
             "spec_method": None if not getattr(eng, "DK", 0) else f"mtp@{eng.DK}x{getattr(eng, 'NR', 1)}"}
    model = {"model_id": model_id, "config_sha": config_sha, "weights_layout_sha": weights_layout_sha}
    fl = flags() if flags_ is None else flags_
    cs = component_set(model, comps, activation_edits, fl, drafter)
    cs_sha = sha(cs)
    key = {"v": V, "chip": chip(tp, mesh) if chip_ is None else chip_, "sw": sw() if sw_ is None else sw_, "flags": fl,
           "model": model, "components": comps, "activation_edits": cs["activation_edits"], "component_set_sha": cs_sha}
    return key, sha(key), cs_sha


def check_cache(provenance, component_set_sha):
    """A shipped compile cache's provenance (dict with component_set_sha) against this deployment. -> None when it
    matches, else the reason it is refused."""
    have = (provenance or {}).get("component_set_sha")
    if have != component_set_sha:
        return (f"the compile cache was built for component set {str(have)[:12]}, this deployment is "
                f"{component_set_sha[:12]}: not used (programs compile at start instead)")
    return None
