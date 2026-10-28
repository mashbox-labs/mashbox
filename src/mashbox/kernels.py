"""Compiled kernel bundles: one wheel per model and JAX version, stored in the model's Hugging Face repo under kernels/.

The wheel is a zip that holds compiled TPU programs (jax.export payloads) and a manifest.json; no source code. It is
read straight from the downloaded, sha256-checked file: nothing is installed. If the bundle does not match the running
chip or JAX version, the caller gets None and the model runs on plain JAX.
"""
import json
import zipfile

import jax


def chip_family():
    """'v6e', 'v5e', ... from the first device, or None off TPU."""
    d = jax.devices()[0]
    if d.platform != "tpu":
        return None
    kind = d.device_kind.lower()
    for name, fam in (("v6 lite", "v6e"), ("v6e", "v6e"), ("v5 lite", "v5e"), ("v5e", "v5e"), ("v5p", "v5p"), ("v4", "v4")):
        if name in kind:
            return fam
    return kind


def versions():
    import importlib.metadata as md
    out = {"jax": jax.__version__}
    for pkg in ("jaxlib", "libtpu"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def open_bundle(path, row, log=print):
    """-> (manifest, {payload name: bytes}) or None when this bundle does not fit the running setup."""
    fam, ver = chip_family(), versions()
    if fam not in row.get("chips", []):
        log(json.dumps(dict(event="kernels", used=False, reason=f"chip {fam} not in {row.get('chips')}")))
        return None
    if ver["jax"] != row.get("jax"):
        log(json.dumps(dict(event="kernels", used=False, reason=f"jax {ver['jax']} != {row.get('jax')}")))
        return None
    with zipfile.ZipFile(path) as z:
        pkg = next(n.split("/", 1)[0] for n in z.namelist() if n.endswith("/manifest.json") and ".dist-info" not in n)
        man = json.loads(z.read(f"{pkg}/manifest.json"))
        blobs = {name: z.read(f"{pkg}/{p['file']}") for name, p in man["payloads"].items()}
    return man, blobs


def decode_kernels(path, row, log=print):
    """{batch rows B: callable(q, k, v, g, beta, pool, slots, layer) -> (o, pool)} for the linear-attention decode step."""
    got = open_bundle(path, row, log)
    if got is None:
        return None
    man, blobs = got
    out = {}
    try:
        for name, p in man["payloads"].items():
            if p.get("op") != "linear_attention_decode":
                continue
            exp = jax.export.deserialize(bytearray(blobs[name]))
            out[int(p["batch"])] = exp.call
    except Exception as e:  # e.g. the flatbuffers package missing: plain JAX, never a crash
        log(json.dumps(dict(event="kernels", used=False, reason=f"{type(e).__name__}: {e}"[:300])))
        return None
    log(json.dumps(dict(event="kernels", used=True, batches=sorted(out), bundle=path.rsplit("/", 1)[-1])))
    return out or None


def cache_entries(path):
    """Compile-cache files shipped in the bundle (for this exact chip and JAX version), as {name: bytes}."""
    with zipfile.ZipFile(path) as z:
        pkg = next(n.split("/", 1)[0] for n in z.namelist() if n.endswith("/manifest.json") and ".dist-info" not in n)
        man = json.loads(z.read(f"{pkg}/manifest.json"))
        return {n.rsplit("/", 1)[-1]: z.read(n) for n in z.namelist()
                if n.startswith(f"{pkg}/{man.get('cache_dir', 'xla_cache')}/") and not n.endswith("/")}
