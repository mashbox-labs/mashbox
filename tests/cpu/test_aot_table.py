"""Executable table (runtime/programs.py Table, Dispatch): a start that finds the table saved by an earlier start of the
same deployment compiles none of the engine's programs the table holds (it loads them; no trace, no lowering) and answers
exactly as the start that built it (on CPU the client cannot serialize some programs, e.g. a sort comparator: those, and
only those, compile at each start, as the save reports); a table built for another deployment or software is refused; a bad entry is skipped and compiled;
a call the table does not hold runs through jit and is a counted compile.

Trust: nothing in a table folder is unpickled before the folder, its parent and every file pass the owner
checks (owned by this user, not writable by group or others, no symlink, no other kind of file) and every file's bytes
match manifest.json (sha256, size, the call key it holds); an extra, missing, changed or swapped file refuses the whole
table. Every read or write error falls back to compiling, never a crash. The deployment id binds the engine source, the
chip topology and the drafter; a table whose program list differs from the warm-up's is dropped before ready."""
import json
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox.runtime import programs  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import engine_standin as M  # noqa: E402


def start(table, out, **env_):
    env = dict(os.environ, JAX_PLATFORMS="cpu", **env_)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)          # no persistent compile cache: only the table can spare a compile
    env.pop("MASHBOX_WARM_RUN_ALL", None) if "MASHBOX_WARM_RUN_ALL" not in env_ else None
    r = subprocess.run([sys.executable, str(HERE / "aot_table_run.py"), str(table), str(out)], capture_output=True,
                       text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def test_second_start_loads_every_program_and_answers_the_same(tmp_path):
    a = start(tmp_path / "table", tmp_path / "a.json")
    b = start(tmp_path / "table", tmp_path / "b.json")
    print(json.dumps({k: {f: r[f] for f in ("load", "save", "compiles", "traces", "warm")} for k, r in (("a", a), ("b", b))}))
    un = (a["save"] or {}).get("unserializable", {})                    # programs this backend cannot write
    assert {"fwd", "dec"} <= set(a["compiles"]["start"]) and not {"fwd", "dec"} & set(un)   # the first start compiles
    assert set(b["compiles"]["start"]) <= set(un), (b["compiles"]["start"], un)   # the second compiles only those
    assert set(b["traces"]["start"]) <= set(un), (b["traces"]["start"], un)       # and traces only those
    assert a["save"]["saved"] > 0
    assert a["compiles"]["after"] == {} and b["compiles"]["after"] == {}  # neither compiles after its warm-up
    assert b["load"]["loaded"] == a["save"]["saved"] and b["load"]["failed"] == 0
    assert b["save"]["saved"] == 0 and b["save"]["unserializable"] == un
    assert b["answers"] == a["answers"]
    # warm-up run pass: the first start runs every program once; the second runs one representative per prefill row
    # bucket and per decoder function (dec, decs) and proves the rest without running them (loaded from the table, the call's own argument
    # checks pass, memory fits); the answers above then ran the skipped programs for the first time: 0 compiles
    wa, wb = a["warm"], b["warm"]
    assert wa["skipped"] == 0 and wa["run_programs"] == wa["prefill"] + wa["decode"] and wa["fallback"] is None
    buckets = len({B for B, T, Tc in b["prefill_shapes"]})
    assert wb["run_programs"] == buckets + 2 and wb["run_why"] == {} and wb["fallback"] is None, wb
    assert wb["skipped"] == wb["prefill"] + wb["decode"] - wb["run_programs"] > 0, wb
    for r in (a, b):        # the programs the warm-up really executed (each run counted at the executable) == its count
        assert r["ran_in_warm"] == r["warm"]["run_programs"] and r["ran_recorded"], (r["ran_in_warm"], r["warm"])
    c = start(tmp_path / "table", tmp_path / "c.json", MASHBOX_WARM_RUN_ALL="1")   # the switch: run every program
    assert c["warm"]["skipped"] == 0 and c["warm"]["run_programs"] == c["warm"]["prefill"] + c["warm"]["decode"]
    assert c["ran_in_warm"] == c["warm"]["run_programs"] and c["ran_recorded"], (c["ran_in_warm"], c["warm"])
    assert c["compiles"]["after"] == {} and c["answers"] == a["answers"]


def engine():
    return Engine(types.SimpleNamespace(V=M.V, H=2), {"w": jnp.ones((), jnp.float32)}, M.MODEL, CTX=256, S=2, K=4)


def test_table_of_another_build_is_refused_and_bad_entries_are_compiled(tmp_path):
    e = engine()
    e.warm(workers=2)
    meta = programs.table_meta("dep-a")
    s = e.aot.save(tmp_path, meta)
    assert s["saved"] == s["entries"] - sum(s["unserializable"].values()) > 0 and s["refused"] is None
    other = engine()
    r = other.aot.load(tmp_path, programs.table_meta("dep-b"))
    assert r["loaded"] == 0 and "deployment_id" in r["refused"] and not other.aot.ex
    r = other.aot.load(tmp_path, dict(meta, jax="0.0.0"))
    assert r["loaded"] == 0 and "jax" in r["refused"]
    assert other.aot.save(tmp_path, programs.table_meta("dep-b"))["refused"]      # never writes into another build's
    files = sorted(p for p in os.listdir(tmp_path) if p.endswith(".exe"))
    (tmp_path / files[0]).write_bytes(b"not an executable")
    third = engine()
    r = third.aot.load(tmp_path, meta)
    assert r["loaded"] == 0 and r["refused"] and not third.aot.ex       # a changed file: the whole table is refused
    w = third.warm(workers=2)
    assert w["compile_failed"] == 0 and w["table"] == s["entries"]       # and every program is compiled again


def test_a_call_the_table_does_not_hold_runs_through_jit_and_counts(tmp_path):
    e = engine()
    e.warm(workers=2)
    ck, cv = e.new_state()
    toks, pos = jnp.zeros(1, jnp.int32), jnp.full(1, e.CTX, jnp.int32)
    n = {"c": 0}

    def listen(ev, secs, **k):
        if ev == programs.EVENT and k.get("fun_name") == "jit(dec)":
            n["c"] += 1
    jax.monitoring.register_event_duration_secs_listener(listen)
    e.dec(e.W, ck, cv, toks, pos, Tc=128, K=1, trace=False)              # held: no compile
    assert n["c"] == 0
    ck, cv = e.new_state()
    e.dec(e.W, ck, cv, toks, pos, Tc=128, K=3, trace=False)              # K=3 is in no program of the list
    assert n["c"] == 1


def test_each_program_keeps_the_jit_interface():
    """A program behind the table stands in for its jit function wherever a caller uses one (main's TP9 check reads the
    decode and prefill jaxprs with .trace): every public method of a jax.jit function is there and gives the jit's own
    result."""
    e = engine()
    public = {n for n in dir(jax.jit(lambda x: x)) if not n.startswith("_")}
    progs = sorted(n for n, v in vars(e).items() if isinstance(v, programs.Dispatch))
    assert {"fwd", "dec"} <= set(progs)
    missing = {n: sorted(public - set(dir(getattr(e, n)))) for n in progs}
    assert missing == {n: [] for n in progs}
    ck, cv = e.new_state()
    a, kw = (e.W, ck, cv, jnp.zeros(1, jnp.int32), jnp.full(1, e.CTX, jnp.int32)), dict(Tc=128, K=1, trace=False)
    assert str(e.dec.trace(*a, **kw).jaxpr) == str(e.dec.jitted.trace(*a, **kw).jaxpr)
    assert e.dec.lower(*a, **kw).as_text() == e.dec.jitted.lower(*a, **kw).as_text()
    assert e.dec.eval_shape(*a, **kw) == e.dec.jitted.eval_shape(*a, **kw)
    e.dec(*a, **kw)                                                      # no table entries: runs through jit
    assert e.dec._cache_size() == 1
    e.dec.clear_cache()
    assert e.dec._cache_size() == 0


# ------------------------------------------------------------------------------------------- trust and fallback
def built(tmp_path, name="table"):
    """A table saved by a warmed stand-in engine. -> (folder, meta, the save's result, sorted .exe names)"""
    e = engine()
    e.warm(workers=2)
    d, meta = tmp_path / name, programs.table_meta("dep-a")
    s = e.aot.save(d, meta)
    assert s["refused"] is None and s["saved"] > 0, s
    return d, meta, s, sorted(p for p in os.listdir(d) if p.endswith(".exe"))


def refused_whole(d, meta):
    e = engine()
    r = e.aot.load(d, meta)                     # never raises
    assert r["refused"] and r["loaded"] == 0 and not e.aot.ex, r
    return e, r


class _Payload:                                 # unpickling it makes the folder named `path` (a harmless sentinel)
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (os.mkdir, (self.path,))


def payload(path):
    import pickle
    return pickle.dumps(_Payload(path))


def test_the_payload_runs_when_unpickled(tmp_path):     # so the two tests below are not vacuous
    import pickle
    pickle.loads(payload(tmp_path / "ran"))
    assert (tmp_path / "ran").is_dir()


@pytest.mark.parametrize("how", ["extra file", "replaces an entry"])
def test_a_malicious_pickle_is_refused_before_it_is_unpickled(tmp_path, how):
    d, meta, s, files = built(tmp_path)
    sentinel = tmp_path / "sentinel"
    name = "0" * 40 + ".exe" if how == "extra file" else files[0]
    (d / name).write_bytes(payload(sentinel))
    refused_whole(d, meta)
    assert not sentinel.exists()


def test_a_valid_executable_under_another_calls_key_is_refused(tmp_path):
    import pickle
    d, meta, s, files = built(tmp_path)
    a, b = (pickle.loads((d / f).read_bytes()) for f in files[:2])
    a["key"] = b["key"]                         # a's program, claiming b's call
    (d / files[1]).write_bytes(pickle.dumps(a))
    refused_whole(d, meta)


def test_swapped_files_are_refused(tmp_path):
    d, meta, s, files = built(tmp_path)
    x, y = (d / files[0]).read_bytes(), (d / files[1]).read_bytes()
    (d / files[0]).write_bytes(y)
    (d / files[1]).write_bytes(x)
    refused_whole(d, meta)


def test_an_extra_copy_of_a_valid_entry_is_refused(tmp_path):
    d, meta, s, files = built(tmp_path)
    (d / ("f" * 40 + ".exe")).write_bytes((d / files[0]).read_bytes())
    refused_whole(d, meta)


def test_a_file_the_manifest_lists_that_is_missing_is_refused_before_any_is_read(tmp_path):
    """What a writer of an older build (it takes no lock on its part folder) can rename into place after the part
    cleanup removed some of its files: a manifest that lists a file the folder no longer has. The whole table is
    refused on the file set, before any table file is read, and nothing is loaded (the start compiles)."""
    d, meta, s, files = built(tmp_path)
    (d / files[0]).unlink()
    e, r = refused_whole(d, meta)
    assert "files the manifest lists are missing" in r["refused"] and files[0][:20] in r["refused"], r


def test_a_symlinked_entry_is_refused(tmp_path):
    d, meta, s, files = built(tmp_path)
    keep = tmp_path / "elsewhere.exe"
    keep.write_bytes((d / files[0]).read_bytes())          # the same valid bytes, behind a link
    (d / files[0]).unlink()
    (d / files[0]).symlink_to(keep)
    refused_whole(d, meta)


def test_one_changed_byte_is_refused(tmp_path):
    d, meta, s, files = built(tmp_path)
    b = bytearray((d / files[0]).read_bytes())
    b[len(b) // 2] ^= 1
    (d / files[0]).write_bytes(bytes(b))
    refused_whole(d, meta)


def test_the_table_folder_is_owner_only_and_a_loose_one_is_refused(tmp_path):
    d, meta, s, files = built(tmp_path)
    assert (os.stat(d).st_mode & 0o777) == 0o700
    os.chmod(d, 0o777)
    try:
        refused_whole(d, meta)
    finally:
        os.chmod(d, 0o700)
    os.chmod(d / files[0], 0o666)               # a file others can write
    refused_whole(d, meta)


@pytest.mark.parametrize("bad", [b"", b"{\"format\": 1, \"me", b"[]", b"\"x\"", b"{\"format\": 1}"])
def test_corrupt_metadata_is_refused_not_a_crash(tmp_path, bad):
    d, meta, s, files = built(tmp_path)
    for j in d.glob("*.json"):
        j.write_bytes(bad)
    refused_whole(d, meta)


def test_unreadable_files_and_folders_are_refused_not_a_crash(tmp_path):
    d, meta, s, files = built(tmp_path)
    os.chmod(d / files[0], 0)
    try:
        refused_whole(d, meta)
    finally:
        os.chmod(d / files[0], 0o600)
    os.chmod(d, 0o300)                          # can enter, cannot list
    try:
        refused_whole(d, meta)
    finally:
        os.chmod(d, 0o700)


def test_an_entry_the_manifest_vouches_for_that_does_not_load_is_skipped_and_compiled(tmp_path):
    import hashlib
    import pickle
    d, meta, s, files = built(tmp_path)
    e0 = pickle.loads((d / files[0]).read_bytes())
    e0["blob"] = b"not an executable"
    raw = pickle.dumps(e0)
    (d / files[0]).write_bytes(raw)
    for j in d.glob("*.json"):                  # the folder's own record agrees with the new bytes
        m = json.loads(j.read_text())
        if isinstance(m, dict) and files[0] in m.get("files", {}):
            m["files"][files[0]].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            j.write_text(json.dumps(m))
    e = engine()
    r = e.aot.load(d, meta)
    assert r["refused"] is None and r["failed"] == 1 and r["loaded"] == len(files) - 1, r
    w = e.warm(workers=2)
    assert w["compile_failed"] == 0 and w["table"] == s["entries"]


def test_a_failed_save_leaves_the_engine_serving(tmp_path, monkeypatch):
    import errno
    e = engine()
    e.warm(workers=2)
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        r = e.aot.save(ro / "table", programs.table_meta("dep-a"))         # never raises
        assert r["refused"] and r["saved"] == 0
    finally:
        os.chmod(ro, 0o700)

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(os, "replace", full)
    monkeypatch.setattr(os, "rename", full)
    r = e.aot.save(tmp_path / "t2", programs.table_meta("dep-a"))
    assert r["refused"] and r["saved"] == 0
    monkeypatch.undo()
    assert not (tmp_path / "t2").exists() or not os.listdir(tmp_path / "t2")
    assert e.generate([[5, 6, 7]], max_new=5)["ids"][0]


class _Chat:
    def chat_ids(self, msgs):
        return [5, 6, 7]


def cli_start(d, capsys):
    """One start as `mashbox run` warms (cli.warm), table in folder d. -> (engine, its aot_table log events)"""
    from mashbox import cli
    e = engine()
    e.deployment_id, e.aot_dir = "dep-a", str(d)
    cli.warm(e, _Chat())
    ev = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    return e, [x for x in ev if x.get("event") == "aot_table"]


@pytest.mark.parametrize("bad", ["corrupt", "read-only parent"])
def test_the_cli_start_compiles_and_serves_when_the_table_cannot_be_read_or_written(tmp_path, capsys, bad):
    if bad == "corrupt":
        d = tmp_path / "table"
        d.mkdir(mode=0o700)
        for n in ("meta.json", "manifest.json"):
            (d / n).write_text("{not json")
    else:
        ro = tmp_path / "ro"
        ro.mkdir()
        d = ro / "table"
        os.chmod(ro, 0o500)
    try:
        e, ev = cli_start(d, capsys)
    finally:
        os.chmod(d.parent, 0o700)
    assert e.generate([[5, 6, 7]], max_new=5)["ids"][0]
    assert [x["phase"] for x in ev][0] == "load" and ev[0]["refused"] and ev[-1]["phase"] == "save"
    if bad == "corrupt":        # moved aside (kept as evidence) and a new table saved in its place (test_aot_table_prune.py)
        assert [x["phase"] for x in ev][1] == "aside" and ev[-1]["saved"] > 0 and ev[-1]["refused"] is None, ev
    else:                       # the parent folder cannot be written: nothing saved
        assert ev[-1]["refused"], ev


def test_a_table_built_for_another_program_list_is_dropped_before_ready(tmp_path, capsys):
    d = tmp_path / "table"
    a, ev = cli_start(d, capsys)
    assert ev[-1]["phase"] == "save" and ev[-1]["saved"] > 0
    for j in d.glob("*.json"):                  # the folder's own record says another program list
        m = json.loads(j.read_text())
        m["program_list_sha"] = "0" * 64
        j.write_text(json.dumps(m))
    b, ev = cli_start(d, capsys)
    load = next(x for x in ev if x["phase"] == "load")
    assert load["refused"] is None and load["loaded"] > 0                 # it loaded and verified ...
    chk = [x for x in ev if x["phase"] == "check"]
    assert chk and "program list" in chk[0]["refused"]                   # ... then the warm-up's list disagreed
    assert not b.aot.loaded and b.aot.ex                                 # dropped; every program compiled
    aside = [x for x in ev if x["phase"] == "aside"]                     # moved aside as evidence, a new table saved
    assert aside and aside[0]["closed"] and ev[-1]["phase"] == "save" and ev[-1]["saved"] > 0, ev
    moved = [q for q in tmp_path.iterdir() if q.name.startswith(".refused-table-")]
    assert len(moved) == 1 and json.loads((moved[0] / "manifest.json").read_text())["program_list_sha"] == "0" * 64
    assert json.loads((d / "manifest.json").read_text())["program_list_sha"] == b.aot.list_sha
    assert b.generate([[5, 6, 7]], max_new=5)["ids"] == a.generate([[5, 6, 7]], max_new=5)["ids"]


# ---------------------------------------------------------------- what the deployment id binds (real CLI)
def test_production_software_key_carries_the_engine_source(tmp_path):
    import shutil
    from mashbox.runtime import compile_key as CK
    sw = CK.sw()
    assert sw["engine_commit"] and sw["engine_commit"] == "source-sha256:" + CK.source_sha()
    root = pathlib.Path(CK.__file__).parent.parent
    cp = tmp_path / "mashbox"
    shutil.copytree(root, cp, ignore=shutil.ignore_patterns("__pycache__"))
    assert CK.source_sha(str(cp)) == CK.source_sha(str(root))
    f = cp / "runtime" / "engine.py"
    f.write_bytes(f.read_bytes() + b"\n# one changed line\n")              # a body change, same signatures
    assert CK.source_sha(str(cp)) != CK.source_sha(str(root))


def _dep(tmp_path, monkeypatch, dk=0, **over):
    """The deployment id cli.load makes (cli.deployment_keys); dk = the engine's draft length (0 = speculation off)."""
    from mashbox import cli
    from mashbox.runtime import compile_key as CK
    e = types.SimpleNamespace(CTX=256, S=2, K=4, chunk=128, tokens_per_call=256, attn_bytes=2e9, budget_busy=128,
                              budget_idle=256, prefix_cache=True, host=None, kv_block=256, trace=False, kernels=[], DK=dk)
    a = dict(config_sha="c" * 64, layout_sha=CK.weights_layout({"w": jnp.ones((2,), jnp.float32)}), kernel_bundle_sha=None,
             tp=1, mesh=None, drafter=None)
    a.update(over)
    cli.deployment_keys(e, {"id": "qwen3-8-27b"}, **a)
    return e.deployment_id


DRAFT = {"kind": "mtp", "id": "qwen38-mtp@1", "weights_sha256": "a" * 64, "k": {"k_default": 2, "k_max": 3, "k_table": {}}}


def test_the_cli_deployment_id_binds_source_topology_and_drafter(tmp_path, monkeypatch):
    from mashbox.runtime import compile_key as CK
    base = _dep(tmp_path, monkeypatch)
    assert base == _dep(tmp_path, monkeypatch)
    assert _dep(tmp_path, monkeypatch, drafter=DRAFT) == base           # speculation off: no drafter in the key
    mesh = types.SimpleNamespace(shape={"tp": 2}, devices=np.array(jax.devices()[:1] * 2))
    ids = {"mesh": _dep(tmp_path, monkeypatch, tp=2, mesh=mesh),
           "mesh shape": _dep(tmp_path, monkeypatch, tp=2, mesh=types.SimpleNamespace(shape={"dp": 2}, devices=mesh.devices)),
           "drafter on": _dep(tmp_path, monkeypatch, dk=2, drafter=DRAFT),
           "drafter weights": _dep(tmp_path, monkeypatch, dk=2, drafter=dict(DRAFT, weights_sha256="b" * 64)),
           "drafter kind": _dep(tmp_path, monkeypatch, dk=2, drafter=dict(DRAFT, kind="ngram", weights_sha256=None)),
           "k rule": _dep(tmp_path, monkeypatch, dk=2, drafter=dict(DRAFT, k=dict(DRAFT["k"], k_default=3))),
           "draft length": _dep(tmp_path, monkeypatch, dk=3, drafter=DRAFT)}
    monkeypatch.setattr(CK, "source_sha", lambda root=None: "1" * 64)
    ids["source"] = _dep(tmp_path, monkeypatch)
    assert base not in ids.values() and len(set(ids.values())) == len(ids), ids


def test_the_drafter_in_the_key_is_the_file_speculation_loaded(tmp_path, monkeypatch):
    """cli.speculation returns the drafter's identity for the file it loaded: the declaration's kind, id and k rule with
    the pinned sha256 the file was checked against, else (a --mtp-file with MASHBOX_VERIFY=0) the sha256 of the file's
    own bytes, so the deployment never names pinned weights it did not load; a drafter models.json does not declare
    keys by its file's name and bytes. cli.load passes it to deployment_keys."""
    import argparse
    import hashlib
    from mashbox import cli
    from mashbox.models.qwen3_8_27b import spec as QSP
    from mashbox.runtime import drafter as DR
    monkeypatch.setattr(DR, "chip_key", lambda: None)
    monkeypatch.setattr(QSP, "load_mtp", lambda path, cfg, log=None: ({}, {}))
    monkeypatch.setattr(cli, "log", lambda s: None)
    f = tmp_path / "Local-MTP.bshj"
    f.write_bytes(b"draft weights")
    own = hashlib.sha256(b"draft weights").hexdigest()
    row = {"name": "D.bshj", "kind": "draft", "sha256": own}
    m = {"id": "t", "files": [row], "drafter": {"kind": "mtp", "id": "t-mtp", "version": 1, "file": "D.bshj",
                                                 "k_default": 2, "k_max": 3}}

    def run(mm):
        return cli.speculation(argparse.Namespace(mtp=2, mtp_file=str(f)), mm, None, 1)["drafter"]
    assert run(m) == DR.key(DR.declared(m))                                # checked against the pin: the declared one
    pinned = dict(m, files=[dict(row, sha256="a" * 64)])
    monkeypatch.setenv("MASHBOX_VERIFY", "0")
    got = run(pinned)                                                      # not checked: its own bytes, not the pin
    assert got == dict(DR.key(DR.declared(pinned)), weights_sha256=own)
    assert _dep(tmp_path, monkeypatch, dk=2, drafter=got) != _dep(tmp_path, monkeypatch, dk=2, drafter=DR.key(DR.declared(pinned)))
    monkeypatch.delenv("MASHBOX_VERIFY")
    bare = {"id": "t", "files": [dict(row, sha256=None)]}                  # no declaration, no pin
    assert run(bare) == {"kind": "mtp", "id": "Local-MTP.bshj@0", "weights_sha256": own,
                         "k": {"k_default": 2, "k_max": 3, "k_table": {}}}


def test_no_table_with_speculative_decoding(tmp_path):
    from mashbox import cli
    e = types.SimpleNamespace(DK=0, deployment_id="dep-a")
    assert cli.aot_dir(str(tmp_path), e) == os.path.join(str(tmp_path), "aot", "dep-a")
    e.DK = 2                                    # its programs are not in the warm list: no table, no claim
    assert cli.aot_dir(str(tmp_path), e) is None


# ------------------------------------------------------------------ warm-up: table programs proven without running them
def loaded_engine(tmp_path):
    """A fresh stand-in engine with the table of a warmed one loaded (every serializable program). -> (engine, load)"""
    d, meta, s, files = built(tmp_path)
    e = engine()
    r = e.aot.load(d, meta)
    assert r["loaded"] == s["saved"] > 0 and r["refused"] is None, r
    return e, r


def test_call_check_agrees_with_the_call_itself(tmp_path):
    """programs.call_check makes the checks jax makes when an executable is called (argument pytree, shapes and dtypes,
    committed shardings and layouts) without running it: it passes exactly when the call would run the executable."""
    e, _ = loaded_engine(tmp_path)
    ck, cv = e.new_state()
    toks, pos = jnp.zeros(1, jnp.int32), jnp.full(1, e.CTX, jnp.int32)
    kw = dict(Tc=128, K=1, trace=False)
    exe = e.aot.ex[e.dec.key((e.W, ck, cv, toks, pos), kw)]
    assert programs.call_check(exe, (e.W, ck, cv, toks, pos)) is None
    exe(e.W, ck, cv, toks, pos)                                          # runs
    ck, cv = e.new_state()
    for bad in ((e.W, ck, cv, toks.astype(jnp.int16), pos), (e.W, ck, cv, toks), (e.W, ck, cv, toks, pos, pos),
                (e.W, ck, cv, jnp.zeros(2, jnp.int32), pos)):
        why = programs.call_check(exe, bad)
        assert why, bad
        with pytest.raises((TypeError, ValueError)):
            exe(*bad)


def test_a_program_that_fails_its_check_runs_once(tmp_path, monkeypatch):
    """A loaded program whose check fails is run in the warm-up as before (its reason counted); the rest are skipped."""
    e, _ = loaded_engine(tmp_path)
    real = programs.call_check
    hit = {"n": 0}

    def check(exe, args):
        if args[3].shape == (2,) and exe is e.aot.ex.get(e.dec.key(args, dict(Tc=256, K=1, trace=False))):
            hit["n"] += 1
            return "test: refused"
        return real(exe, args)
    monkeypatch.setattr(programs, "call_check", check)
    w = e.warm(workers=2)
    assert hit["n"] == 1 and w["run_why"] == {"test: refused": 1} and w["fallback"] is None, (hit, w)
    assert w["skipped"] == w["prefill"] + w["decode"] - w["run_programs"] and w["run_programs"] == len(
        {B for B, T, Tc in e.prefill_shapes()}) + 2 + 1


def test_memory_that_does_not_fit_runs_the_program(tmp_path, monkeypatch):
    """With less device memory free than a loaded program needs (temp + outputs - aliased inputs), it runs once."""
    e, _ = loaded_engine(tmp_path)
    monkeypatch.setattr(programs, "device_free", lambda: 0)
    w = e.warm(workers=2)
    assert w["skipped"] == 0 and w["run_programs"] == w["prefill"] + w["decode"], w
    assert w["run_why"] and all(k.startswith("needs ") for k in w["run_why"]), w["run_why"]


def test_a_loaded_program_that_fails_in_the_warm_up_is_a_table_failure(tmp_path):
    """A canary (a loaded program the warm-up runs) that raises: Engine.warm raises programs.TableFailed with the table
    quarantined and dropped, and never warms a second time in this process (A13: the process ends, and the next start
    compiles without the table). Old code: the warm-up dropped the table and warmed again here."""
    e, _ = loaded_engine(tmp_path)
    B, T, Tc = e.prefill_shapes()[0]
    a = jnp.asarray
    args = (a(np.zeros((B, T), np.int32)), a(np.zeros(B, np.int32)), a(np.ones(B, np.int32)), a(np.arange(B, dtype=np.int32)))
    ck, cv = e.new_state()
    k = e.fwd.key((e.W, ck, cv, *args), dict(Tc=Tc, hidden=False))
    assert k in e.aot.ex

    def broken(*x):
        raise RuntimeError("device fault in a loaded program")
    e.aot.ex[k] = broken
    warms, real = [], e.warm
    e.warm = lambda *x, **kw: (warms.append(1), real(*x, **kw))[1]
    with pytest.raises(programs.TableFailed) as ex:
        e.warm(workers=2)
    info = ex.value.info
    assert "device fault" in info["error"] and info["closed"] is True and info["quarantined"], info
    assert warms == [1] and not e.aot.loaded and not e.aot.ex and e.state is None
    assert not (tmp_path / "table").exists() and [n for n in os.listdir(tmp_path) if n.startswith(".refused-table-")]


def test_a_table_failure_in_the_start_requests_leaves_cli_warm(tmp_path, capsys):
    """cli.warm's two requests reach a loaded program that fails: programs.TableFailed leaves cli.warm with the table
    quarantined and dropped, and nothing runs again in this process: no second request, no Engine.warm, no save (A13).
    Old code: cli.warm logged it, ran the request again and warmed on."""
    from mashbox import cli
    d = tmp_path / "table"
    cli_start(d, capsys)                                      # an earlier start saved the table
    e = engine()
    e.deployment_id, e.aot_dir = "dep-a", str(d)
    real_load, runs, warms = e.aot.load, [], []

    def load(*x, **kw):
        out = real_load(*x, **kw)

        def broken(*y):
            runs.append(1)
            raise RuntimeError("device fault at launch")
        for key in list(e.aot.loaded):
            e.aot.ex[key] = broken
        return out
    e.aot.load = load
    real_warm = e.warm
    e.warm = lambda *x, **kw: (warms.append(1), real_warm(*x, **kw))[1]
    with pytest.raises(programs.TableFailed) as ex:
        cli.warm(e, _Chat())
    assert ex.value.info["closed"] is True and "at launch" in ex.value.info["error"], ex.value.info
    assert runs == [1] and warms == [] and not e.aot.loaded and not e.aot.ex
    ev = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    assert [x.get("phase") for x in ev if x.get("event") == "aot_table"] == ["load"], ev      # no save, no prune
    assert not d.exists() and [n for n in os.listdir(tmp_path) if n.startswith(".refused-table-")]


@pytest.mark.parametrize("closed", [True, False])
def test_a_one_process_server_whose_table_fails_while_it_starts_exits_70(monkeypatch, capsys, closed):
    """`mashbox run` of a GGUF file serves in one process: a table failure in its warm-up ends the process (exit 70)
    before it says ready or serves, and when the table's folder could not be closed the message says to start again
    with MASHBOX_AOT=0 (the next start would load the same folder)."""
    from mashbox import cli
    from mashbox.runtime import server as SV
    from test_front_failures import IdChat
    e = engine()
    info = dict(file="x.gguf", device_bytes=1, bpw=4.0)
    monkeypatch.setattr(cli, "load_gguf", lambda a: (dict(id="x"), e, IdChat(), info, 0.0))
    monkeypatch.setattr(cli, "api_keys", lambda a: [])
    fail = programs.TableFailed(dict(program="fwd", key="0" * 16, error="RuntimeError: device fault (test)",
                                     quarantined="not renamed (test)", closed=closed))
    monkeypatch.setattr(cli, "warm", lambda eng, chat: (_ for _ in ()).throw(fail))
    served = []
    monkeypatch.setattr(SV, "serve", lambda *x, **k: served.append(1))
    a = types.SimpleNamespace(prompt=None, gguf="x.gguf", no_warm=False, cache_ram=0, ui=False, host="127.0.0.1", port=0)
    with pytest.raises(SystemExit) as ex:
        cli.cmd_run(a, [])
    err = capsys.readouterr().err
    assert ex.value.code == 70 and served == [] and '"ready"' not in err, err
    assert '"phase": "failed"' in err and ("MASHBOX_AOT=0" in err) is (not closed), err


def test_only_programs_that_ran_are_saved(tmp_path):
    """Every table entry ran in the start that saved it: a program built into the table but never run is not saved
    (so a later start that skips running a loaded program skips one that ran on this deployment)."""
    e = engine()
    e.warm(workers=2)
    ck, cv = e.new_state()
    toks, pos = jnp.zeros(1, jnp.int32), jnp.full(1, e.CTX, jnp.int32)
    assert e.dec.build(e.W, ck, cv, toks, pos, Tc=128, K=3, trace=False)    # in the table, never run
    k = e.dec.key((e.W, ck, cv, toks, pos), dict(Tc=128, K=3, trace=False))
    s = e.aot.save(tmp_path / "t", programs.table_meta("dep-a"))
    assert s["saved"] == s["entries"] - sum(s["unserializable"].values()) - 1 and s["not_run"] == 1, s
    assert programs.Table._file(k) not in os.listdir(tmp_path / "t")
