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


HF_HOSTS = ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/")


def find_row(target, reg=None):
    """-> (the models.json models[] row a run target names, the field it matched) or (None, None). A target names a row
    by its id (any case, '.' read as '-': Qwen3.8-27B), its name, or its Hugging Face repo (owner/name, with or without
    https://huggingface.co/ or hf.co/). The catalog only: no network, nothing imported, no id written here."""
    key = str(target).strip()
    for h in HF_HOSTS:
        if key.lower().startswith(h):
            key = key[len(h):]
    key = key.strip("/")
    low = key.lower()
    for m in (registry() if reg is None else reg).get("models") or []:
        if m["id"] == low.replace(".", "-"):
            return m, "id"
        if low and low == (m.get("name") or "").lower():
            return m, "name"
        if low and low == (m.get("hf_repo") or "").lower():
            return m, "hf_repo"
    return None, None


def model_entry(model_id):
    m, _ = find_row(model_id)
    if m is not None:
        return m
    raise SystemExit(f"unknown model {model_id!r}; known: {[m['id'] for m in registry().get('models') or []]}")


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


class DownloadError(Exception):
    """A download from the Hub that failed: refused (HTTP 401, 403, 404), unreachable, offline, or the disk full. str() is
    the one line to show. Running the same command again is safe: files already downloaded are kept."""


def download_errors():
    """What hf_hub_download raises when a download is refused or cannot finish, never a programming error: OSError
    (huggingface_hub 0.x's requests errors, its LocalEntryNotFoundError, disk errors) and, from 1.0, its HTTP library's
    HTTPError (a connection, DNS or read failure is not an OSError there) and XetDownloadError."""
    try:
        from huggingface_hub import errors as E
    except ImportError:
        return (OSError,)
    more = (getattr(E, n, None) for n in ("HfHubHTTPError", "HTTPError", "XetDownloadError"))
    return (OSError,) + tuple(c for c in more if isinstance(c, type) and issubclass(c, Exception))


def download_error(filename, repo, revision, e, hint=""):
    """the one line for a failed download: the file, its repo and revision, why, and that running again is safe"""
    import re
    code = getattr(getattr(e, "response", None), "status_code", None)
    first = (str(e).strip().splitlines() or [""])[0]
    first = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)[0].rstrip(".")[:200]     # its first sentence, whole
    why = f"Hugging Face answered HTTP {code}" if code else type(e).__name__ + (f": {first}" if first else "")
    return (f"mashbox: could not download {filename} from https://huggingface.co/{repo} (revision {(revision or 'main')[:8]}): "
            f"{why}. Check the network and run the same command again (files already downloaded are kept){hint}.")


def fetch(repo, filename, revision, local_dir=None, hint=""):
    """hf_hub_download of one file at a pinned revision. A download that fails raises DownloadError, one line (hint: what
    else the person can do, appended to it)."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=local_dir)
    except download_errors() as e:
        raise DownloadError(download_error(filename, repo, revision, e, hint)) from e


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


def compile_cache(m, cache_dir, log=print, tp=1):
    """Unpack the model's XLA compile cache (models.json files[] row with kind "xla-cache") into cache_dir when its
    deployment and jax, jaxlib and libtpu match this setup, so the first run loads its programs instead of compiling them.
    -> local path or None. Deployment = the row's accelerator (chip family + chips, kernels.accelerator(): v6e-1, v6e-4)
    and the tensor-parallel size its programs were built for (row "tp", 1 when absent; tp = this run's). The chip family
    alone does not say which programs a cache holds (a v6e-4 tensor-parallel start compiled every program from the
    one-chip v6e cache), so a row that names no accelerator is never unpacked.
    No match, or any error, = one log line and compiling at first use, as before. That includes a tar whose sha256 does not
    match the row: verify() stops the run with SystemExit for weights, so this path catches SystemExit too."""
    import shutil
    import tarfile
    from . import kernels as K
    try:
        acc, ver = K.accelerator(), K.versions()
        for row in m.get("files", []):
            if row.get("kind") != "xla-cache" or acc is None or row.get("accelerator") != acc or row.get("tp", 1) != tp \
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
        log(json.dumps(dict(event="compile_cache", used=False, reason=f"no cache for {acc} tp {tp}, jax {ver['jax']}, libtpu {ver.get('libtpu')}")))
    except (Exception, SystemExit) as e:  # a cache is an optimisation: never fail the run over it
        log(json.dumps(dict(event="compile_cache", used=False, reason=f"{type(e).__name__}: {e}"[:300])))
    return None


def kernel_bundle(m, log=print):
    """(local path, models.json row) of the bundle that matches this setup, or (None, None)."""
    from . import kernels as K
    fam, ver = K.chip_family(), K.versions()
    for row in m.get("kernels", []):
        if fam in row["chips"] and ver["jax"] == row["jax"]:
            p = fetch(m["hf_repo"], row["file"], m["hf_revision"],
                      hint=", or start with --no-kernels to run without it, in plain JAX (slower)")
            return verify(p, row["sha256"], log), row
    log(json.dumps(dict(event="kernels", used=False, reason=f"no bundle for chip {fam}, jax {ver['jax']}")))
    return None, None
