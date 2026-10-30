"""Upstream weights for TRELLIS.2: downloaded from Hugging Face at pinned revisions, checked by sha256, never re-uploaded.

upstream.json next to this file lists every file with its size and sha256:
  * microsoft/TRELLIS.2-4B: the pipeline file, the three flow models and the shape and texture decoders of the 512 pipeline;
  * microsoft/TRELLIS-image-large: the sparse-structure decoder the pipeline file points to;
  * facebook/dinov3-vitl16-pretrain-lvd1689m: the image encoder. Meta gates it on Hugging Face, so it downloads with your
    own login after you accept its terms on the model page.
Files go to <cache>/trellis-2/<owner>_<name>@<revision>/, the layout the loader reads.
"""
import json
import os
import shutil
from importlib import resources

GATE = ("TRELLIS.2 reads Meta's DINOv3 image encoder, which is gated on Hugging Face. Accept its terms at "
        "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m with your account, give this machine your "
        "Hugging Face access token (set HF_TOKEN, or run `hf auth login`), then run the command again.")
UNCHECKED = "not checked (MASHBOX_VERIFY=0)"     # what the loader gets for a file whose sha256 check was skipped


def pins():
    return json.loads(resources.files(__package__).joinpath("upstream.json").read_text())


def root():
    from ... import hub
    return os.path.join(hub.cache_home(), "trellis-2")


def repo_dir(repo):
    return os.path.join(root(), repo["hf"].replace("/", "_") + "@" + repo["revision"])


def needed_bytes():
    return sum(f["bytes"] for r in pins()["repos"].values() for f in r["files"])


def missing_bytes():
    n = 0
    for r in pins()["repos"].values():
        for f in r["files"]:
            p = os.path.join(repo_dir(r), f["name"])
            if not (os.path.isfile(p) and os.path.getsize(p) == f["bytes"]):
                n += f["bytes"]
    return n


def check_disk(log=print):
    """Stop early, with numbers, when the cache disk cannot hold the files still missing."""
    os.makedirs(root(), exist_ok=True)
    free, need = shutil.disk_usage(root()).free, missing_bytes()
    if free < need:
        raise SystemExit(f"{root()}: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB still to download "
                         f"({needed_bytes() / 1e9:.1f} GB in all). Set MASHBOX_CACHE to a bigger disk.")


def _download(repo, name, local_dir):
    from ... import hub
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:                                   # older huggingface_hub
        GatedRepoError = HfHubHTTPError = OSError
    try:
        return hub.fetch(repo["hf"], name, repo["revision"], local_dir=local_dir)
    except GatedRepoError:
        raise SystemExit(GATE)
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if repo.get("gated") and code in (401, 403):
            raise SystemExit(GATE)
        raise SystemExit(f"download of {repo['hf']}/{name} failed ({code or type(e).__name__}); check the network and try again")


def stage(log=print):
    """Download what is missing, check every file's sha256, and return (root, {path: sha256}) for the loader. With
    MASHBOX_VERIFY=0 the check is skipped and each file maps to UNCHECKED instead of its pin."""
    from ... import hub
    check_disk(log)
    check = os.environ.get("MASHBOX_VERIFY", "1") != "0"
    if not check:
        log(json.dumps(dict(event="sha256 check off", why="MASHBOX_VERIFY=0")))
    verified = {}
    for r in pins()["repos"].values():
        d = repo_dir(r)
        for f in r["files"]:
            p = os.path.join(d, f["name"])
            if not os.path.isfile(p):
                log(json.dumps(dict(event="download", repo=r["hf"], file=f["name"], gb=round(f["bytes"] / 1e9, 2))))
                p = _download(r, f["name"], d)
            if check:
                hub.verify(p, f["sha256"], log)
            verified[p] = f["sha256"] if check else UNCHECKED
    return root(), verified
