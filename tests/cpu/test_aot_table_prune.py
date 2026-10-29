"""Executable tables on disk do not pile up, and a refused table does not refuse every later start (runtime/programs.py
prune, set_aside, Table.hold; cli.warm).

After a start saves its table, prune() keeps the current deployment's table and the newest other one used in the last
14 days (a start of the previous version still finds its table), keeps the newest two refused tables from the last 14
days as evidence (dated by the refusal), removes what an interrupted save left (any part folder no live save
holds; a save also removes them before it writes), and removes the rest. A table another process holds
(every process that loads or saves a table holds a shared lock on its lock file while it uses it, and one
that cannot take the lock does not use the table) is never removed, and
only this user's own table folders are touched: other names, links, folders another user owns or others can write, and
folders holding anything a table does not write stay as they are. A start that refuses its table folder, or drops it
for another program list, moves it aside (kept as evidence) and saves a new table in its place."""
import fcntl
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mashbox.runtime import programs  # noqa: E402
from test_aot_table import cli_start, engine  # noqa: E402

DAY = 86400
A, B, C, D, E = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
EXE = "0" * 40 + ".exe"
# The holders end on the SIGINT the tests send, so each puts SIGINT back to Python's default first: a pytest
# started in the background by a non-interactive shell (nohup ... &) hands its children SIGINT ignored, and the
# holder then slept through it and h.wait(30) raised TimeoutExpired (cert_lib.py and front_recovery_check.py do
# the same for their servers).
HOLD = ("import fcntl, os, signal, sys, time\n"
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
        "fcntl.flock(fd, fcntl.LOCK_SH)\n"
        "print('held', flush=True)\n"
        "time.sleep(120)\n")


def aot(tmp_path):
    p = tmp_path / "aot"
    p.mkdir(mode=0o700)
    return p


def table(parent, name, age_days=0.0, files=("manifest.json", EXE), mode=0o700):
    """A folder as a save writes it (owner-only files), last changed age_days ago."""
    p = parent / name
    p.mkdir(mode=0o700)
    for f in files:
        fd = os.open(p / f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, b"x")
        os.close(fd)
    os.chmod(p, mode)
    t = time.time() - age_days * DAY
    os.utime(p, (t, t))
    return p


def listed(parent):
    return sorted(n for n in os.listdir(parent) if not n.endswith(".lock"))


def free(lock):
    """True when nothing holds the lock file (an exclusive lock can be taken now)."""
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def test_the_current_table_and_the_newest_other_stay_and_older_ones_go(tmp_path):
    p = aot(tmp_path)
    table(p, A, 3)
    table(p, B, 2)
    table(p, C, 1)
    table(p, D, 0)
    r = programs.prune(str(p), D)
    assert sorted(r["removed"]) == [A, B] and listed(p) == [C, D], r
    assert not (p / f".{A}.lock").exists() and not (p / f".{B}.lock").exists()
    table(p, E, 0)                      # the next deployment: the one before it stays, C goes
    r = programs.prune(str(p), E)
    assert r["removed"] == [C] and listed(p) == [D, E], r


def test_a_table_unused_for_14_days_goes_but_never_the_current_one(tmp_path):
    p = aot(tmp_path)
    table(p, A, 20)
    table(p, B, 30)
    r = programs.prune(str(p), B)
    assert r["removed"] == [A] and listed(p) == [B], r


def test_the_last_start_that_used_a_table_counts_not_only_its_save(tmp_path):
    p = aot(tmp_path)
    table(p, A, 20)                     # saved 20 days ago, used by a start an hour ago
    table(p, B, 1)
    table(p, C, 0)
    lk = p / f".{A}.lock"
    lk.touch(mode=0o600)
    t = time.time() - 3600
    os.utime(lk, (t, t))
    r = programs.prune(str(p), C)
    assert r["removed"] == [B] and listed(p) == [A, C], r


def test_a_table_another_process_holds_is_never_removed(tmp_path):
    p = aot(tmp_path)
    table(p, A, 30)
    table(p, B, 0)
    h = subprocess.Popen([sys.executable, "-c", HOLD, str(p / f".{A}.lock")], stdout=subprocess.PIPE, text=True)
    try:
        assert h.stdout.readline().strip() == "held"
        old = time.time() - 30 * DAY                            # a process that has used A for 30 days: A is old
        os.utime(p / f".{A}.lock", (old, old))
        table(p, C, 1)                                           # and not the newest other table
        r = programs.prune(str(p), B)
        assert r["removed"] == [] and "in use" in r["left"][A] and (p / A / EXE).is_file(), r
    finally:
        h.send_signal(signal.SIGINT)
        h.wait(30)
        h.stdout.close()
    r = programs.prune(str(p), B)
    assert r["removed"] == [A] and listed(p) == [B, C], r


def test_a_table_that_loads_or_saves_holds_its_lock_while_it_uses_it(tmp_path):
    p = aot(tmp_path)
    e = engine()
    e.warm(workers=2)
    meta = programs.table_meta("dep-a")
    assert e.aot.save(p / A, meta)["saved"] > 0
    assert not free(p / f".{A}.lock")                       # the saving process holds it
    old = time.time() - 30 * DAY
    os.utime(p / A, (old, old))
    os.utime(p / f".{A}.lock", (old, old))
    table(p, B, 0)
    r = programs.prune(str(p), B)
    assert "in use" in r["left"][A] and (p / A).is_dir(), r
    e3 = engine()
    assert e3.aot.load(p / A, meta)["loaded"] > 0
    e.aot.hold(p / D)                                        # e now uses another folder: A is held by e3 only
    assert not free(p / f".{A}.lock")
    e3.aot.hold(p / D)
    assert free(p / f".{A}.lock")


def test_refusal_evidence_stays_newest_two_from_the_last_14_days(tmp_path):
    p = aot(tmp_path)
    table(p, D, 0)
    new = table(p, f".refused-{A}-20261028T120000Z-11", 0.1)
    inplace = table(p, B, 0.5, files=("manifest.json", EXE, "REFUSED"))     # refused in place: evidence, not a table
    third = table(p, f".refused-{C}-20261027T120000Z-12", 1)
    old = table(p, f".refused-{E}-20261008T120000Z-13", 20)
    r = programs.prune(str(p), D)
    assert sorted(r["removed"]) == sorted([third.name, old.name]), r
    assert new.is_dir() and inplace.is_dir() and not third.exists() and not old.exists()
    os.utime(new, (time.time() - 20 * DAY,) * 2)              # an old refused folder goes even when it is one of two
    r = programs.prune(str(p), D)
    assert r["removed"] == [new.name] and inplace.is_dir(), r


HOLD_DIR = ("import fcntl, os, signal, sys, time\n"
            "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
            "fd = os.open(sys.argv[1], os.O_RDONLY)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n")


def test_what_an_interrupted_save_left_goes_at_once_unless_a_save_writes_it(tmp_path):
    """A save holds an exclusive lock on its part folder while it writes it: a part folder whose lock can be taken is
    debris, however young, and goes; one a live save holds stays (so part folders are never more than the saves running).
    Old code: a part folder younger than a day stayed, however many a crash loop left."""
    p = aot(tmp_path)
    table(p, D, 0)
    old = table(p, f".{A}.part-abc12345", 2, files=(EXE,))
    new = table(p, f".{B}.part-x_9y8z7w", 0.0, files=(EXE,))
    held = table(p, f".{C}.part-q1w2e3r4", 3, files=(EXE,))
    h = subprocess.Popen([sys.executable, "-c", HOLD_DIR, str(held)], stdout=subprocess.PIPE, text=True)
    try:
        assert h.stdout.readline().strip() == "held"           # a save of C writes this folder
        r = programs.prune(str(p), D)
    finally:
        h.send_signal(signal.SIGINT)
        h.wait(30)
        h.stdout.close()
    assert sorted(r["removed"]) == sorted([old.name, new.name]) and "another process" in r["left"][held.name], r
    assert held.is_dir() and listed(p) == [held.name, D]


def test_a_save_first_removes_what_interrupted_saves_left_and_holds_its_own_part_folder(tmp_path, monkeypatch):
    """A crash loop of saves that die mid-write (each start holds its deployment's lock, as after its load): the next
    save removes their part folders before it writes, without a prune (old code: only a prune after a save that worked
    removed them, and only after a day, so they could fill the disk and stop every save). A prune that runs while this
    save writes leaves its part folder."""
    import jax.experimental.serialize_executable as SE
    p = aot(tmp_path)
    e = engine()
    e.warm(workers=2)
    e.aot.hold(p / C)                                        # this start's deployment lock, as its load took it
    debris = [table(p, f".{C}.part-crash{i:03d}", 0.0, files=(EXE,)) for i in range(5)]
    mid, real = [], SE.serialize

    def serialize(exe):
        if not mid:
            mid.append(programs.prune(str(p), D))            # another start's prune while this save writes
            mid.append(sorted(n for n in os.listdir(p) if ".part-" in n))
        return real(exe)
    monkeypatch.setattr(SE, "serialize", serialize)
    s = e.aot.save(p / C, programs.table_meta("dep-a"))
    assert s["saved"] > 0 and s["refused"] is None, s
    assert not [d for d in debris if d.exists()] and len(mid[1]) == 1, mid    # only this save's own part folder
    assert mid[0]["removed"] == [] and mid[1][0] in mid[0]["left"], mid
    assert listed(p) == [C]


def test_fresh_refusal_evidence_is_dated_by_its_refusal_not_by_the_table_save(tmp_path):
    """A table saved 30 days ago and refused now: its quarantined folder is evidence from today and stays. Old code: the
    rename kept the table's own time, and the next prune removed it as older than 14 days."""
    p = aot(tmp_path)
    table(p, D, 0)
    t = table(p, A, 30)
    q, closed = programs.quarantine(t)
    assert closed and os.path.basename(q).startswith(f".refused-{A}-"), q
    r = programs.prune(str(p), D)
    assert r["removed"] == [] and os.path.isdir(q), r


def test_a_table_whose_lock_cannot_be_taken_is_not_used(tmp_path, monkeypatch):
    """A process that cannot take a table's shared lock does not use the table: its load refuses it (every program
    compiles) and its save writes nothing, so a prune that then takes the lock removes no table a live process uses.
    Old code: the lock failure was ignored and the table loaded (or the save wrote) with no lock held."""
    p = aot(tmp_path)
    e = engine()
    e.warm(workers=2)
    meta = programs.table_meta("dep-a")
    assert e.aot.save(p / A, meta)["saved"] > 0
    e.aot.hold(p / D)                                        # e lets A go
    real = programs._flock

    def no_shared(path, how):
        if how == fcntl.LOCK_SH:
            raise OSError(37, "No locks available (test)")
        return real(path, how)
    monkeypatch.setattr(programs, "_flock", no_shared)
    e2 = engine()
    r = e2.aot.load(p / A, meta)
    assert r["loaded"] == 0 and "lock could not be taken" in (r["refused"] or "") and not e2.aot.ex, r
    assert e2.aot.held is None and e2.aot.loaded_from is None
    s = e.aot.save(p / B, meta)
    assert s["saved"] == 0 and "lock could not be taken" in (s["refused"] or ""), s
    assert not (p / B).exists() and not [n for n in os.listdir(p) if ".part-" in n]
    monkeypatch.undo()
    old = time.time() - 30 * DAY
    os.utime(p / A, (old, old))
    os.utime(p / f".{A}.lock", (old, old))
    table(p, C, 0)
    r = programs.prune(str(p), C)                            # nothing holds A: no process uses it
    assert A in r["removed"] and listed(p) == [C], r


def test_links_foreign_names_and_folders_a_table_does_not_write_are_never_touched(tmp_path):
    p = aot(tmp_path)
    table(p, D, 0)
    outside = table(tmp_path, "outside", 30)
    os.symlink(outside, p / A)                                # a link named like a table
    os.symlink(outside, p / f".refused-{A}-20261001T000000Z-1")
    foreign = table(p, "notes", 30)                           # another name
    extra = table(p, B, 30, files=("manifest.json", "notes.txt"))   # a file a table does not write
    loose = table(p, C, 30, mode=0o770)                       # others can write it
    inner = table(p, E, 30, files=("manifest.json",))
    os.symlink(outside / EXE, inner / EXE)                    # a link inside
    t = time.time() - 30 * DAY
    os.utime(inner, (t, t))
    sub = table(p, f".{A}.part-abc12345", 30, files=())
    (sub / "deeper").mkdir()                                   # a folder inside
    os.utime(sub, (t, t))
    before = {n: sorted(os.listdir(p / n)) for n in os.listdir(p) if (p / n).is_dir() and not (p / n).is_symlink()}
    r = programs.prune(str(p), D)
    os.chmod(loose, 0o700)
    assert r["removed"] == [], r
    assert {n: sorted(os.listdir(p / n)) for n in before} == before
    assert (p / A).is_symlink() and sorted(os.listdir(outside)) == sorted(["manifest.json", EXE]) and foreign.is_dir()
    assert {B, C, E, sub.name} <= set(r["left"]) and extra.is_dir()


def test_a_start_that_saves_removes_old_tables_and_keeps_the_previous_one(tmp_path, capsys):
    p = aot(tmp_path)
    table(p, A, 3)
    table(p, B, 1)
    e, ev = cli_start(p / C, capsys)
    s = ev[-1]
    assert s["phase"] == "save" and s["saved"] > 0 and s["pruned"]["removed"] == [A], s
    assert listed(p) == [B, C]
    assert e.generate([[5, 6, 7]], max_new=5)["ids"][0]


def test_a_refused_table_is_moved_aside_and_the_start_saves_a_new_one(tmp_path, capsys):
    p = aot(tmp_path)
    d = table(p, C, 0, files=("manifest.json",))              # "x": not a manifest
    a, ev = cli_start(d, capsys)
    ph = [x["phase"] for x in ev]
    assert ph[:2] == ["load", "aside"] and ev[0]["refused"] and ev[1]["closed"], ev
    assert ev[-1]["phase"] == "save" and ev[-1]["saved"] > 0 and ev[-1]["refused"] is None, ev
    moved = [n for n in os.listdir(p) if n.startswith(f".refused-{C}-")]
    assert len(moved) == 1 and (p / moved[0] / "manifest.json").read_bytes() == b"x"
    b, ev = cli_start(d, capsys)                               # the next start loads the new table
    assert ev[0]["phase"] == "load" and ev[0]["refused"] is None and ev[0]["loaded"] > 0, ev
    assert "aside" not in [x["phase"] for x in ev]
    assert b.generate([[5, 6, 7]], max_new=5)["ids"] == a.generate([[5, 6, 7]], max_new=5)["ids"]


def test_no_table_is_not_moved_aside_and_an_untrusted_parent_is_left_alone(tmp_path, capsys):
    p = aot(tmp_path)
    _, ev = cli_start(p / C, capsys)                           # no folder yet: nothing to move
    assert "aside" not in [x["phase"] for x in ev] and ev[-1]["saved"] > 0
    q = tmp_path / "loose"
    q.mkdir(mode=0o700)
    d = table(q, C, 0, files=("manifest.json",))
    os.chmod(q, 0o770)
    try:
        _, ev = cli_start(d, capsys)
    finally:
        os.chmod(q, 0o700)
    assert ev[0]["refused"] and "aside" not in [x["phase"] for x in ev] and ev[-1]["refused"], ev
    assert sorted(os.listdir(q)) == [C] and (d / "manifest.json").read_bytes() == b"x"
