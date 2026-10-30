"""Which GGUF file `mashbox run <model>` loads: a local path, or a Hugging Face repo picked with llama.cpp's `-hf` rule.

  mashbox run ./model.gguf
  mashbox run unsloth/Qwen3.8-27B-GGUF            llama.cpp's default: the first Q4_K_M file, else the first Q8_0 file
  mashbox run unsloth/Qwen3.8-27B-GGUF:Q8_0       that quant tag

The rule follows llama.cpp's common/download.cpp (find_best_model): a tag matches when the regex `<tag>[.-]` is found in
the file's path (any case); imatrix files and sidecars (mmproj, mtp-, dflash-, eagle3-, dspark-) are never the model;
of a split model only the first shard is picked and every shard is downloaded; with no tag and no Q4_K_M or Q8_0 file,
the first model file is used. Files are listed in the Hub's tree order, as llama.cpp lists them.
"""
import json
import os
import re

from .reader import GGUFError


class NoGGUF(GGUFError):
    """a Hub repo with no .gguf file at all; files = its listing, so the caller can try the Hugging Face safetensors path
    without listing the repo again"""

    def __init__(self, msg, files):
        super().__init__(msg)
        self.files = files

SIDECARS = ("mmproj", "mtp-", "dflash-", "eagle3-", "dspark-")
DEFAULT_TAGS = ("Q4_K_M", "Q8_0")
SPLIT = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.I)
HOSTS = ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/")


def is_gguf_request(model):
    """a .gguf path, an existing file, or something shaped like a Hub repo (models.json ids have no slash)"""
    return model.lower().endswith(".gguf") or os.path.isfile(model) or "/" in model


def split_repo_tag(s):
    for h in HOSTS:
        if s.startswith(h):
            s = s[len(h):]
    repo, _, tag = s.partition(":")
    return repo.strip("/"), tag or None


def is_model_file(path):
    if not path.endswith(".gguf"):
        return False
    name = path.rsplit("/", 1)[-1]
    return "imatrix" not in name and not any(k in name for k in SIDECARS)


def split_info(path):
    """-> (index, count, prefix) for <prefix>-00001-of-00003.gguf, else (1, 1, prefix)"""
    prefix = path[:-5] if path.endswith(".gguf") else path
    m = SPLIT.match(prefix)
    return (int(m.group(2)), int(m.group(3)), m.group(1)) if m else (1, 1, prefix)


def pick(files, tag=None):
    """llama.cpp's find_best_model over `files` (paths in tree order) -> the primary path or None"""
    tags = [tag] if tag else list(DEFAULT_TAGS)
    for t in tags:
        try:
            pat = re.compile(t + "[.-]", re.I)
        except re.error:
            pat = re.compile(re.escape(t) + "[.-]", re.I)
        for p in files:
            if is_model_file(p) and pat.search(p):
                i, n, _ = split_info(p)
                if n > 1 and i != 1:
                    continue
                return p
    if not tag:
        for p in files:
            if is_model_file(p):
                i, n, _ = split_info(p)
                if n > 1 and i != 1:
                    continue
                return p
    return None


def shards(primary):
    i, n, prefix = split_info(primary)
    return [primary] if n == 1 else [f"{prefix}-{k:05d}-of-{n:05d}.gguf" for k in range(1, n + 1)]


def file_identity(path):
    """-> (sha256 of a weights file, how it was found), sha256 None when it does not exist. A Hugging Face cache blob is
    named by the Hub's sha256 of its bytes (blobs/<sha256>): that name is used; any other file is read once."""
    from ..runtime import tenancy
    real = os.path.realpath(path)
    name = os.path.basename(real)
    if os.path.basename(os.path.dirname(real)) == "blobs" and len(name) == 64 and all(ch in "0123456789abcdef" for ch in name):
        return name, "hub_blob_name"
    return tenancy.file_sha256(path), "read"


def identity(path):
    """-> (sha256, shards, how): a GGUF file's identity, the one its deployment logs (gguf_identity) and a catalog row
    names: the file's sha256, or for a split file the sha256 of its shards' sha256s joined by spaces (the shards beside
    the first, by its name); sha256 None when a shard does not exist"""
    import hashlib
    sh = shards(os.path.basename(path))
    paths = [os.path.join(os.path.dirname(path), s) for s in sh] if len(sh) > 1 else [path]
    ids = [file_identity(p) for p in paths]
    if any(i[0] is None for i in ids):
        sha = None
    else:
        sha = ids[0][0] if len(ids) == 1 else hashlib.sha256(" ".join(i[0] for i in ids).encode()).hexdigest()
    return sha, len(ids), sorted({i[1] for i in ids})


def resolve(model, log=print, download=True):
    """-> dict(path, repo, tag, file, revision, files, bytes). Local paths pass through; repos are listed and downloaded
    (Hugging Face cache, HF_TOKEN if set)."""
    if os.path.isfile(model):
        return dict(path=os.path.abspath(model), repo=None, tag=None, file=os.path.basename(model), revision=None,
                    files=[model], bytes=os.path.getsize(model))
    if model.lower().endswith(".gguf") or model.startswith(("/", ".", "~")):
        raise GGUFError(f"mashbox: {model}: no such file")
    repo, tag = split_repo_tag(model)
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError, RevisionNotFoundError
    api = HfApi()
    try:
        info = api.model_info(repo)
        tree = [e for e in api.list_repo_tree(repo, revision=info.sha, recursive=True) if isinstance(e, RepoFile)]
    except HFValidationError:
        raise GGUFError(f"mashbox: {model}: not a file, and not a Hugging Face repo name (owner/name[:quant tag])")
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise GGUFError(f"mashbox: {repo}: no such model on Hugging Face (or it is gated and HF_TOKEN is not set)")
    paths = [e.path for e in tree]
    size = {e.path: e.size for e in tree}
    if any(p == "preset.ini" for p in paths):
        raise GGUFError(f"mashbox: {repo} holds a llama.cpp preset.ini; name the file instead: {repo}:<quant tag>")
    primary = pick(paths, tag)
    ggufs = [p for p in paths if p.endswith(".gguf")]
    if primary is None:
        if not ggufs:
            raise NoGGUF(f"mashbox: {repo} has no GGUF files and no config.json with safetensors weights. mashbox runs GGUF "
                         f"repos, .gguf files, Hugging Face safetensors repos and the models in `mashbox models`", paths)
        raise GGUFError(f"mashbox: {repo} has no GGUF file for tag {tag!r}. Files: " + ", ".join(ggufs))
    want = shards(primary)
    missing = [p for p in want if p not in size]
    if missing:
        raise GGUFError(f"mashbox: {repo}: shard(s) missing from the repo: {', '.join(missing)}")
    rule = f"tag {tag}" if tag else "llama.cpp's default (first Q4_K_M, then Q8_0)"
    total = sum(size[p] for p in want)
    log(json.dumps(dict(event="gguf_file", repo=repo, file=primary, shards=len(want), gb=round(total / 1e9, 2),
                        revision=info.sha, rule=rule)))
    if not download:
        return dict(path=None, repo=repo, tag=tag, file=primary, revision=info.sha, files=want, bytes=total)
    local = [hf_hub_download(repo, p, revision=info.sha) for p in want]
    return dict(path=local[0], repo=repo, tag=tag, file=primary, revision=info.sha, files=want, bytes=total)
