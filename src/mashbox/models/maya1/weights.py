"""maya1's files: downloaded from Hugging Face at pinned revisions and checked by sha256 before use.

  * the model's own repo (models.json files[] of the maya1 row, at its hf_revision): maya1 as one BF16 GGUF file, and the
    SNAC 24 kHz decoder as NumPy arrays;
  * maya-research/maya1 (upstream.json next to this file): tokenizer.json, as published (the GGUF file holds no
    tokenizer).
Files go to <cache>/maya1/<owner>_<name>@<revision>/.
"""
import json
import os
import shutil
from importlib import resources

UNCHECKED = "not checked (MASHBOX_VERIFY=0)"


def pins():
    return json.loads(resources.files(__package__).joinpath("upstream.json").read_text())


def repos():
    """-> [{hf, revision, files: [{name, bytes, sha256}]}]: the model's own repo, then upstream.json's"""
    from ... import hub
    m = hub.model_entry("maya1")
    own = dict(hf=m["hf_repo"], revision=m["hf_revision"],
               files=[dict(name=f["name"], bytes=f["bytes"], sha256=f["sha256"]) for f in m["files"]])
    return [own] + list(pins()["repos"].values())


def root():
    from ... import hub
    return os.path.join(hub.cache_home(), "maya1")


def repo_dir(repo):
    return os.path.join(root(), repo["hf"].replace("/", "_") + "@" + repo["revision"])


def missing_bytes():
    n = 0
    for r in repos():
        for f in r["files"]:
            p = os.path.join(repo_dir(r), f["name"])
            if not (os.path.isfile(p) and os.path.getsize(p) == f["bytes"]):
                n += f["bytes"]
    return n


def check_disk():
    os.makedirs(root(), exist_ok=True)
    free, need = shutil.disk_usage(root()).free, missing_bytes()
    if free < need:
        raise SystemExit(f"{root()}: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB still to download. "
                         f"Set MASHBOX_CACHE to a bigger disk.")


def stage(log=print):
    """Download what is missing, check every file's sha256, and return {role: path} with roles gguf, snac and
    tokenizer. With MASHBOX_VERIFY=0 the check is skipped."""
    from ... import hub
    check_disk()
    check = os.environ.get("MASHBOX_VERIFY", "1") != "0"
    if not check:
        log(json.dumps(dict(event="sha256 check off", why="MASHBOX_VERIFY=0")))
    out = {}
    for r in repos():
        d = repo_dir(r)
        for f in r["files"]:
            p = os.path.join(d, f["name"])
            if not os.path.isfile(p):
                log(json.dumps(dict(event="download", repo=r["hf"], file=f["name"], gb=round(f["bytes"] / 1e9, 3))))
                p = hub.fetch(r["hf"], f["name"], r["revision"], local_dir=d)
            if check:
                hub.verify(p, f["sha256"], log)
            role = {"maya1-BF16.gguf": "gguf", "snac/snac_24khz_decoder.npz": "snac", "tokenizer.json": "tokenizer"}[f["name"]]
            out[role] = p
    return out
