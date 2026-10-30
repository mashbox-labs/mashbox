"""TRELLIS.2 chip record, for maintainers: the seal's own sha256, its file count and the code identity are checked
together from the sealed files, so a record can never be cited for code it does not name.

A record made with the current chip tooling seals rec/build.json (commit, tree, and the sha256 of the source archive,
the commit object and the wheel it seals beside it). The commit and tree are recomputed here from those sealed bytes,
never taken as written: the tree from the archive's files, the commit from the commit object, which must name that
tree, and the commit the archive's own git header names. A record that seals only the wheel's sha256 says so in its
pin (code_identity.in_seal false). Skipped where the record or the pin is not on this machine, unless
MASHBOX_REQUIRE_RECORDS=1 asks for them. Stdlib only (git for the fixture repo), so it runs on the machine that holds the record.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

RECORD = Path(__file__).resolve().parents[2] / "internal" / "trellis-2-record.json"
REQUIRE = os.environ.get("MASHBOX_REQUIRE_RECORDS") == "1"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git_hash(kind, body):
    return hashlib.sha1(b"%s %d\0" % (kind, len(body)) + body).hexdigest()


def _archive_tree(tar_path):
    """(tree, commit): the git tree id of a `git archive <commit>` tar recomputed from its bytes, and the commit id
    its pax global header names."""
    root = {}
    with tarfile.open(tar_path) as t:
        for m in t:
            *dirs, name = [p for p in m.name.split("/") if p]
            node = root
            for p in dirs:
                node = node.setdefault(p, {})
            if m.isdir():
                node.setdefault(name, {})
            elif m.issym():
                node[name] = ("120000", _git_hash(b"blob", m.linkname.encode()))
            else:
                assert m.isreg(), f"{m.name}: git archive writes files, directories and symlinks only"
                node[name] = ("100755" if m.mode & 0o100 else "100644", _git_hash(b"blob", t.extractfile(m).read()))
        commit = t.pax_headers.get("comment")

    def tree(node):
        ents = []
        for n, v in node.items():
            if isinstance(v, dict):
                ents.append(((n + "/").encode(), b"40000 %s\0" % n.encode() + bytes.fromhex(tree(v))))
            else:
                ents.append((n.encode(), b"%s %s\0" % (v[0].encode(), n.encode()) + bytes.fromhex(v[1])))
        return _git_hash(b"tree", b"".join(e for _, e in sorted(ents)))

    return tree(root), commit


def _check_build_identity(d, listed, rec):
    """A record that seals rec/build.json: the archive, commit object and wheel it names are sealed at those sha256
    values, and its commit and tree are the ones their bytes give, the same commit as the pin's code_commit."""
    assert _sha(d / "rec/build.json") == listed["rec/build.json"]
    build = json.loads((d / "rec/build.json").read_text())
    for part in ("source", "wheel", "commit_object"):
        assert listed[build[part]["path"]] == build[part]["sha256"] == _sha(d / build[part]["path"])
    tree, header_commit = _archive_tree(d / build["source"]["path"])
    obj = (d / build["commit_object"]["path"]).read_bytes()
    obj_tree = re.match(rb"tree ([0-9a-f]{40})\n", obj)
    assert obj_tree and build["tree"] == tree == obj_tree.group(1).decode()
    assert build["commit"] == _git_hash(b"commit", obj) == header_commit == rec["code_commit"]
    assert rec["code_identity"]["in_seal"] is True


def _record():
    if not RECORD.is_file():
        pytest.skip("maintainer check: the chip record pin is held from the public tree")
    rec = json.loads(RECORD.read_text())["record"]
    if not Path(rec["dir"]).is_dir():
        why = f"the chip record {rec['dir']} is not on this machine"
        pytest.fail(f"MASHBOX_REQUIRE_RECORDS=1 and {why}") if REQUIRE else pytest.skip(why)
    return rec


def test_chip_record_seal_count_and_code_identity():
    rec = _record()
    d = Path(rec["dir"])
    assert _sha(d / "SHA256SUMS") == rec["sha256sums_sha256"]
    listed = {}
    for line in (d / "SHA256SUMS").read_text().splitlines():
        h, name = re.fullmatch(r"([0-9a-f]{64})  (.+)", line).groups()
        listed[name] = h
    assert len(listed) == rec["files"]
    ident = rec["code_identity"]
    if "rec/build.json" in listed:
        _check_build_identity(d, listed, rec)
    else:
        assert ident["in_seal"] is False and ident["commit_joined_after_run"]
        assert _sha(d / "rec/chain.log") == listed["rec/chain.log"]
        wheels = re.findall(r"^([0-9a-f]{64})  dist/mashbox-\S+\.whl$", (d / "rec/chain.log").read_text(), re.M)
        assert wheels == [ident["wheel_sha256_in_seal"]]


def _git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="fixture", GIT_AUTHOR_EMAIL="fixture@example.com",
               GIT_COMMITTER_NAME="fixture", GIT_COMMITTER_EMAIL="fixture@example.com",
               GIT_AUTHOR_DATE="2026-10-29T06:00:00Z", GIT_COMMITTER_DATE="2026-10-29T06:00:00Z",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env).stdout


def _sealed_build_record(root, commit, tree, tar, obj, code_commit=None):
    """A minimal sealed record with rec/build.json (archive, commit object, wheel), its listing and its pin."""
    d = root / "record"
    (d / "build").mkdir(parents=True)
    (d / "rec").mkdir()
    (d / "build/tree.tar").write_bytes(tar)
    (d / "build/commit.obj").write_bytes(obj)
    (d / "build/mashbox-0.0.0-py3-none-any.whl").write_bytes(b"wheel")
    parts = {"source": "build/tree.tar", "commit_object": "build/commit.obj",
             "wheel": "build/mashbox-0.0.0-py3-none-any.whl"}
    build = {"commit": commit, "tree": tree, **{k: {"path": p, "sha256": _sha(d / p)} for k, p in parts.items()}}
    (d / "rec/build.json").write_text(json.dumps(build))
    listed = {n: _sha(d / n) for n in [*parts.values(), "rec/build.json"]}
    return d, listed, {"code_commit": code_commit or commit, "code_identity": {"in_seal": True}}


def test_a_build_record_binds_its_commit_and_tree_to_the_sealed_archive(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    repo = tmp_path / "src"
    (repo / "pkg").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "pkg.txt").write_text("sorts before pkg/\n")
    (repo / "run.sh").write_text("#!/bin/sh\n")
    (repo / "run.sh").chmod(0o755)
    (repo / "link").symlink_to("pkg/a.py")
    ids = []
    for n in (1, 2):
        (repo / "pkg/a.py").write_text(f"x = {n}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"c{n}")
        c = _git(repo, "rev-parse", "HEAD").decode().strip()
        ids.append((c, _git(repo, "rev-parse", "HEAD^{tree}").decode().strip(),
                    _git(repo, "archive", "--format=tar", c), _git(repo, "cat-file", "commit", c)))
    (c1, t1, tar1, obj1), (c2, t2, _, obj2) = ids
    _check_build_identity(*_sealed_build_record(tmp_path / "ok", c1, t1, tar1, obj1))
    forged = [(c2, t1, tar1, obj1),          # another real commit, declared and pinned, beside this archive
              (c1, t2, tar1, obj1),          # another real tree
              (c1, "0" * 40, tar1, obj1),    # a well-formed, made-up tree
              (c2, t2, tar1, obj2)]          # the other commit's ids and commit object beside this archive
    for i, f in enumerate(forged):
        with pytest.raises(AssertionError):
            _check_build_identity(*_sealed_build_record(tmp_path / f"forged{i}", *f))
    with pytest.raises(AssertionError):      # the pin names another commit than the record
        _check_build_identity(*_sealed_build_record(tmp_path / "pin", c1, t1, tar1, obj1, code_commit=c2))
