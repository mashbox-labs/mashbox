"""A program loaded from the saved executable table that the warm-up did not run (review R11, 2026-10-28).

Engine.warm runs one representative per prefill row bucket and one per decoder function (dec, decs) of a loaded table;
every other loaded program (other attention windows, step widths, row buckets) first runs in a request. If it fails
there, that is a table failure (programs.TableFailed): the table folder is quarantined (renamed beside it, never loaded
again), the table is dropped, and the call never runs through jit in its place (that would be a compile after ready).
The request gets an error and the engine's process ends: it never warms again where it said ready (review R12-F02);
behind the HTTP front a new worker starts without the table and the front sends a request that got no output once more
(test_aot_table_restart.py). A failure that a representative reports only at the warm-up's final barrier (a device
error raised asynchronously) drops the table as a synchronous one does."""
import os
import threading

import pytest

jax = pytest.importorskip("jax")

from mashbox.runtime import programs  # noqa: E402
from mashbox.runtime import server as SV  # noqa: E402
from test_aot_table import M, engine, loaded_engine  # noqa: E402

PROMPTS = (3, 17, 64, 100, 129, 200)


def prompt(L):
    return [(7 * j + L) % (M.V - 1) + 1 for j in range(L)]


class Broken:
    """Stands in for a loaded executable that fails when it is called."""

    def __init__(self, exc, hits):
        self.exc, self.hits = exc, hits

    def __call__(self, *a):
        self.hits.append(1)
        raise self.exc


def skipped_after_warm(tmp_path):
    """A stand-in engine that loaded a saved table and warmed with it. -> (engine, keys of loaded prefill / decode
    programs the warm-up did not run, the table folder)"""
    e, _ = loaded_engine(tmp_path)
    w = e.warm(workers=2)
    assert w["fallback"] is None and w["skipped"] > 0, w
    skipped = [k for k in e.aot.loaded if k not in e.aot.ran and k[0] in ("fwd", "dec", "decs")]
    assert len(skipped) == w["skipped"], (len(skipped), w)
    return e, skipped, tmp_path / "table"


def compiles(flag=None):
    """Count compiles of the engine's prefill / decode programs from now on (with flag(): its value at each one)."""
    n = {"c": 0, "while": []}

    def listen(ev, secs, **k):
        if ev == programs.EVENT and str(k.get("fun_name")) in ("jit(fwd)", "jit(dec)", "jit(decs)"):
            n["c"] += 1
            if flag is not None:
                n["while"].append(flag())
    jax.monitoring.register_event_duration_secs_listener(listen)
    return n


def first_failure(e):
    """Run the fixed prompts until one raises. -> (the exception or None, its prompt length)"""
    for L in PROMPTS:
        try:
            e.generate([prompt(L)], max_new=12)
        except Exception as x:
            return x, L
    return None, None


def assert_quarantined(tmp_path, d):
    assert not e_loaded_any(d), "the failed table's folder still loads"
    q = [p for p in os.listdir(tmp_path) if p.startswith(".refused-table-")]
    assert len(q) == 1, os.listdir(tmp_path)        # kept beside it as evidence, not deleted


def e_loaded_any(d):
    if not os.path.exists(d):
        return False
    r = engine().aot.load(d, programs.table_meta("dep-a"))
    return bool(r.get("loaded"))


XRE = getattr(jax.errors, "JaxRuntimeError", RuntimeError)
LAUNCH = [("device fault at launch", RuntimeError),
          ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.00GiB", XRE),       # OOM at launch
          ("INVALID_ARGUMENT: Invalid buffer passed: buffer has been deleted or donated", XRE)]      # donation / aliasing


@pytest.mark.parametrize("msg,cls", LAUNCH, ids=["device", "oom", "donated"])
def test_a_skipped_program_that_fails_at_first_use_quarantines_and_drops_the_table(tmp_path, msg, cls):
    e, skipped, d = skipped_after_warm(tmp_path)
    hits = []
    for k in skipped:
        e.aot.ex[k] = Broken(cls(msg), hits)
    err, L = first_failure(e)
    assert hits and err is not None, "no request reached a skipped program"
    assert not e.aot.loaded and not any(isinstance(x, Broken) for x in e.aot.ex.values()), "the table is still in use"
    assert_quarantined(tmp_path, d)
    assert isinstance(err, programs.TableFailed) and msg in str(err) and isinstance(err.__cause__, cls), repr(err)
    assert err.info["program"] in ("fwd", "dec", "decs") and err.info["quarantined"].startswith(str(tmp_path)), err.info
    n = compiles()
    e.warm(workers=2)                               # the server's recovery: every program again, without the table
    e.generate([prompt(L)], max_new=12)
    assert n["c"] > 0 and len(hits) == 1            # compiled in the warm-up; the broken program never ran again
    n["c"] = 0
    e.generate([prompt(L)], max_new=12)
    assert n["c"] == 0


@pytest.mark.parametrize("exc", [TypeError, ValueError])
def test_a_skipped_program_refused_after_its_check_never_runs_through_jit(tmp_path, exc):
    """The executable's own argument check refuses the call after programs.proven passed it: before this fix the call ran
    through jit instead (a compile after ready, and the table stayed in use)."""
    e, skipped, d = skipped_after_warm(tmp_path)
    hits = []
    for k in skipped:
        e.aot.ex[k] = Broken(exc("argument refused by the executable"), hits)
    n = compiles()
    err, L = first_failure(e)
    assert hits, "no request reached a skipped program"
    assert err is not None and n["c"] == 0, ("ran through jit", repr(err), n)
    assert not e.aot.loaded
    assert_quarantined(tmp_path, d)
    assert isinstance(err, programs.TableFailed), repr(err)


class Chat:
    stop_ids = ()

    def encode(self, text):
        return [ord(c) % 60 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def serve_until_failure(srv):
    for L in PROMPTS:
        j = srv.submit(prompt(L), SV.knobs_of(dict(max_tokens=12)), False, False)
        assert j["ev"].wait(120)
        if j.get("error") is not None:
            return j, L
    return None, None


def one_process(tmp_path, only_decode=False):
    """A ready one-process server over an engine that loaded a table, whose skipped programs fail when called; its
    exit is recorded instead of ending pytest. -> (server, engine, logs, exit codes, the exit event, warm calls, hits,
    the table folder)"""
    e, skipped, d = skipped_after_warm(tmp_path)
    logs, codes, warms, hits, ended = [], [], [], [], threading.Event()
    srv = SV.Server(e, Chat(), "m", {}, log=logs.append)
    srv.ready = True
    srv.exit = lambda c: (codes.append(c), ended.set())
    real = e.warm
    e.warm = lambda *a, **k: (warms.append(1), real(*a, **k))[1]
    for k in skipped:
        if not only_decode or k[0] in ("dec", "decs"):     # decode: the first token has gone out when it fails
            e.aot.ex[k] = Broken(RuntimeError("device fault at launch"), hits)
    return srv, e, logs, codes, ended, warms, hits, d


def test_a_table_failure_ends_the_one_process_server_without_warming_again(tmp_path):
    """review R12-F02: the server used to warm every program again in its ready process (each compile a compile after
    ready) and serve on. Now the jobs of that call get the error and the process ends (exit code 70) with no warm-up;
    behind the HTTP front the worker ends instead and the front restarts it (test_aot_table_restart.py)."""
    srv, e, logs, codes, ended, warms, hits, d = one_process(tmp_path)
    n = compiles()
    j, L = serve_until_failure(srv)
    assert hits == [1] and j is not None and isinstance(j["error"], programs.TableFailed), repr(j and j.get("error"))
    assert ended.wait(60) and codes == [70] and not srv.ready and not warms and n["c"] == 0, (codes, warms, n)
    assert not j.get("retried")
    assert_quarantined(tmp_path, d)
    assert sum('"aot_table"' in x and '"failed"' in x for x in logs) == 1 and not any('"rewarm"' in x for x in logs)


def test_a_streamed_request_that_sent_tokens_gets_the_error_and_the_process_ends(tmp_path):
    srv, e, logs, codes, ended, warms, hits, d = one_process(tmp_path, only_decode=True)
    j = srv.submit(prompt(3), SV.knobs_of(dict(max_tokens=12)), False, True)
    got = []
    while (x := j["stream"].get(timeout=300)) is not None:
        got.append(x)
    assert hits and got and got[0][0], (hits, got)      # a token was sent before the failure
    assert isinstance(j.get("error"), programs.TableFailed) and not j.get("retried"), repr(j.get("error"))
    assert ended.wait(60) and codes == [70] and not srv.ready and not warms, (codes, warms)
    assert_quarantined(tmp_path, d)


def test_a_representative_failure_reported_after_it_returned_is_a_table_failure_in_the_warm_up(tmp_path, monkeypatch):
    """A device error raised asynchronously surfaces at jax.block_until_ready, after the call returned."""
    import numpy as np
    e, _ = loaded_engine(tmp_path)
    B, T, Tc = e.prefill_shapes()[0]
    a = jax.numpy.asarray
    args = (a(np.zeros((B, T), np.int32)), a(np.zeros(B, np.int32)), a(np.ones(B, np.int32)), a(np.arange(B, dtype=np.int32)))
    ck, cv = e.new_state()
    k = e.fwd.key((e.W, ck, cv, *args), dict(Tc=Tc, hidden=False))
    real, pending = e.aot.ex[k], []

    def late(*x):
        out = real(*x)
        pending.append(1)
        return out
    e.aot.ex[k] = late
    barrier = jax.block_until_ready

    def bur(x):
        if pending:
            pending.clear()
            raise RuntimeError("device fault reported at the barrier")
        return barrier(x)
    monkeypatch.setattr(jax, "block_until_ready", bur)
    with pytest.raises(programs.TableFailed) as ex:          # the process ends (A13); old code warmed again here
        e.warm(workers=2)
    assert "reported at the barrier" in ex.value.info["error"] and ex.value.info["closed"] is True, ex.value.info
    assert not e.aot.loaded and not e.aot.ex
    assert_quarantined(tmp_path, tmp_path / "table")


def test_a_skipped_program_whose_fault_is_reported_after_it_returned_quarantines_and_drops_the_table(tmp_path, monkeypatch):
    """review R12-F01: JAX runs asynchronously, so a loaded program can return and its device fault be reported at the
    first synchronization after it (here a stand-in: the real executable runs, then the next jax.block_until_ready
    raises). Its first use in this process waits for it inside the table boundary, so the fault is a table failure
    (quarantine, drop, no jit in its place), not a request error that leaves the table in use."""
    e, skipped, d = skipped_after_warm(tmp_path)
    pending, hits = [], []
    for k in skipped:
        def late(*x, real=e.aot.ex[k]):
            out = real(*x)
            hits.append(1)
            pending.append(1)
            return out
        e.aot.ex[k] = late
    barrier = jax.block_until_ready

    def bur(x):
        if pending:
            pending.clear()
            raise RuntimeError("device fault reported after the call returned")
        return barrier(x)
    monkeypatch.setattr(jax, "block_until_ready", bur)
    n = compiles()
    err, L = first_failure(e)
    assert hits, "no request reached a skipped program"
    assert isinstance(err, programs.TableFailed) and "after the call returned" in str(err), repr(err)
    assert n["c"] == 0, "ran through jit"
    assert not e.aot.loaded and not e.aot.ex, "the table is still in use"
    assert_quarantined(tmp_path, d)


def _no_rename(a, b):
    raise PermissionError(13, "Permission denied (test)", a)


def test_a_table_that_cannot_be_renamed_is_refused_in_place(tmp_path, monkeypatch):
    """review R12-F04: a failed rename used to leave the folder loadable by the next start."""
    e, skipped, d = skipped_after_warm(tmp_path)
    hits = []
    for k in skipped:
        e.aot.ex[k] = Broken(RuntimeError("device fault at launch"), hits)
    monkeypatch.setattr(os, "rename", _no_rename)
    err, L = first_failure(e)
    monkeypatch.undo()
    assert hits and isinstance(err, programs.TableFailed), repr(err)
    assert err.info.get("closed") is True and "refused in place" in str(err.info["quarantined"]), err.info
    assert os.path.isdir(d) and not e_loaded_any(d), "the failed table's folder still loads"


def test_a_table_that_can_be_neither_renamed_nor_refused_ends_the_process_before_it_serves(tmp_path, monkeypatch):
    e, skipped, d = skipped_after_warm(tmp_path)
    srv = SV.Server(e, Chat(), "m", {}, log=lambda m: None)
    srv.ready = True
    ended, codes, warms = threading.Event(), [], []
    srv.exit = lambda c: (codes.append(c), ended.set())
    real_warm, real_open = e.warm, os.open
    e.warm = lambda *a, **k: (warms.append(1), real_warm(*a, **k))[1]
    hits = []
    for k in skipped:
        e.aot.ex[k] = Broken(RuntimeError("device fault at launch"), hits)

    def no_refused(path, *a, **k):
        if str(path).endswith("REFUSED"):
            raise PermissionError(13, "Permission denied (test)", path)
        return real_open(path, *a, **k)
    monkeypatch.setattr(os, "rename", _no_rename)
    monkeypatch.setattr(os, "open", no_refused)
    j, L = serve_until_failure(srv)
    assert hits and j is not None and isinstance(j["error"], programs.TableFailed), repr(j and j.get("error"))
    assert ended.wait(60) and codes == [70] and not srv.ready and not warms, (codes, warms)
    assert j["error"].info.get("closed") is False
