"""Downloads, pinned by revision and checked by sha256.

Upstream files (config, tokenizer, chat template, licence) come from the original model repo at the revision in
models.json. Our files (the .bshj weights and the kernel bundle) come from the model's mashbox-labs repo. Each big file's
sha256 is checked once after download; a small stamp under ~/.cache/mashbox/verified records the result.
"""
import hashlib
import json
import os
import time

# Downloads use plain HTTPS unless you set HF_HUB_DISABLE_XET yourself (0 = use Xet). On new TPU VMs, Xet transfers of the
# 26.9 GB weights file can stop with HTTP 429 or stall, where plain HTTPS finishes. Set before huggingface_hub is imported,
# which reads it once.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

UPSTREAM_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE")


def cache_home():
    return os.environ.get("MASHBOX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "mashbox"))


def registry():
    from importlib import resources
    return json.loads(resources.files("mashbox").joinpath("models.json").read_text())


def model_entry(model_id):
    want = model_id.replace(".", "-").lower()
    for m in registry()["models"]:
        if m["id"] == want:
            return m
    raise SystemExit(f"unknown model {model_id!r}; known: {[m['id'] for m in registry()['models']]}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(path, sha256, log=print):
    """sha256 of a file, checked once per (path, size, mtime) and remembered."""
    st = os.stat(path)
    key = hashlib.sha256(f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:32]
    stamp = os.path.join(cache_home(), "verified", key + ".json")
    try:
        if json.load(open(stamp)).get("sha256") == sha256:
            return path
    except (OSError, ValueError):
        pass
    t0 = time.time()
    got = _sha256(path)
    if got != sha256:
        raise SystemExit(f"{path}: sha256 {got} does not match the pinned {sha256}; delete the file and download again")
    os.makedirs(os.path.dirname(stamp), exist_ok=True)
    json.dump(dict(path=os.path.realpath(path), sha256=sha256, checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              open(stamp, "w"))
    log(json.dumps(dict(event="verified", file=os.path.basename(path), seconds=round(time.time() - t0, 1))))
    return path


def fetch(repo, filename, revision, local_dir=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=local_dir)


def upstream_dir(m, log=print):
    """Directory holding the upstream config / tokenizer files at the pinned revision."""
    up = m["upstream"]
    d = os.path.join(cache_home(), "upstream", up["hf"].replace("/", "--"), up["revision"])
    for f in UPSTREAM_FILES:
        if not os.path.exists(os.path.join(d, f)):
            fetch(up["hf"], f, up["revision"], local_dir=d)
    return d


def weights_row(m, name=None):
    """The models.json row of the model's weights file: the default one, or the one called `name`."""
    rows = [f for f in m["files"] if f.get("kind", "weights") == "weights"]
    rows = ([f for f in rows if f.get("default")] or rows) if name is None else [f for f in rows if f["name"] == name]
    if not rows:
        raise SystemExit(f"no weights file {name!r} in models.json for {m['id']}")
    return rows[0]


def weights_file(m, name=None, path=None, log=print):
    """Local path of the model's weights file (downloaded and checked unless `path` is given)."""
    row = weights_row(m, name)
    if path:
        return verify(path, row["sha256"], log) if row.get("sha256") and os.environ.get("MASHBOX_VERIFY", "1") != "0" else path
    p = fetch(m["hf_repo"], row["name"], m["hf_revision"])
    return verify(p, row["sha256"], log)


def draft_row(m):
    """The models.json files[] row of the model's speculative-decoding draft weights (kind "draft"), or None."""
    return next((f for f in m.get("files", []) if f.get("kind") == "draft"), None)


def draft_file(m, path=None, log=print):
    """Local path of the draft weights (downloaded and checked unless `path` is given). A row may pin its own revision of
    the model repo (a draft file added after the weights' release commit), else the model's hf_revision, and may name its
    own repo, else the model's hf_repo."""
    row = draft_row(m)
    if path:
        ok = row and row.get("sha256") and os.environ.get("MASHBOX_VERIFY", "1") != "0"
        return verify(path, row["sha256"], log) if ok else path
    if row is None:
        raise SystemExit(f"no draft file in models.json for {m['id']}")
    p = fetch(row.get("repo") or m["hf_repo"], row["name"], row.get("revision") or m["hf_revision"])
    return verify(p, row["sha256"], log)


def compile_cache(m, cache_dir, log=print):
    """Unpack the model's XLA compile cache (models.json files[] row with kind "xla-cache") into cache_dir when its chip, jax,
    jaxlib and libtpu match this setup, so the first run loads its programs instead of compiling them. -> local path or None.
    No match, or any error, = one log line and compiling at first use, as before. That includes a tar whose sha256 does not
    match the row: verify() stops the run with SystemExit for weights, so this path catches SystemExit too."""
    import shutil
    import tarfile
    from . import kernels as K
    try:
        fam, ver = K.chip_family(), K.versions()
        for row in m.get("files", []):
            if row.get("kind") != "xla-cache" or fam not in row.get("chips", []) \
                    or any(ver.get(k) != row.get(k) for k in ("jax", "jaxlib", "libtpu")):
                continue
            p = verify(fetch(m["hf_repo"], row["name"], m["hf_revision"]), row["sha256"], log)
            n = 0
            with tarfile.open(p) as t:
                for mem in t.getmembers():
                    name = mem.name
                    if not mem.isfile() or "/" in name or name.startswith("."):
                        continue
                    dst = os.path.join(cache_dir, name)
                    if os.path.exists(dst):
                        continue
                    with t.extractfile(mem) as src, open(dst + ".part", "wb") as out:
                        shutil.copyfileobj(src, out, 64 << 20)
                    os.replace(dst + ".part", dst)
                    n += 1
            log(json.dumps(dict(event="compile_cache", used=True, file=row["name"].rsplit("/", 1)[-1], new_entries=n)))
            return p
        log(json.dumps(dict(event="compile_cache", used=False, reason=f"no cache for chip {fam}, jax {ver['jax']}, libtpu {ver.get('libtpu')}")))
    except (Exception, SystemExit) as e:  # a cache is an optimisation: never fail the run over it
        log(json.dumps(dict(event="compile_cache", used=False, reason=f"{type(e).__name__}: {e}"[:300])))
    return None


def kernel_bundle(m, log=print):
    """(local path, models.json row) of the bundle that matches this setup, or (None, None)."""
    from . import kernels as K
    fam, ver = K.chip_family(), K.versions()
    for row in m.get("kernels", []):
        if fam in row["chips"] and ver["jax"] == row["jax"]:
            p = fetch(m["hf_repo"], row["file"], m["hf_revision"])
            return verify(p, row["sha256"], log), row
    log(json.dumps(dict(event="kernels", used=False, reason=f"no bundle for chip {fam}, jax {ver['jax']}")))
    return None, None
