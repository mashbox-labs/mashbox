"""The programs a server compiles, and a watch on every compile after it reports ready.

Engine.warm() builds the program list: one entry per program it compiles and runs before ready (each prefill and decode
program, and the small helper programs a request runs: logit rows by slot, the first sampled token, slot state take / put
/ swap, host-cache copies). Each entry is a signature: the function name, its static arguments and the shape, dtype and
weak type of every argument. list_sha() hashes the sorted signatures, so two builds that would compile the same program set
print the same value (a compile cache made for one list is checked against it).

The watch counts every XLA backend compile in this process (jit programs and eager ops alike, by function name). After
mark_ready() each compile is a compile after ready: it is counted and handed to the callback (the server logs it), and
/health reports the count (null in a process that has not marked ready, never a 0 it did not count). Zero compiles after ready is the gate (spec E14 on CPU, the certified base on a chip).
"""
import hashlib
import json
import re
import threading

import numpy as np

EVENT = "/jax/core/compile/backend_compile_duration"
_lock = threading.Lock()
_st = dict(total=0, after=0, ready=False, names={}, cb=None, on=False)


def install():
    """Start counting compiles in this process (once; later calls do nothing)."""
    with _lock:
        if _st["on"]:
            return
        _st["on"] = True
    from jax import monitoring
    monitoring.register_event_duration_secs_listener(_listen)


def _listen(ev, secs, **k):
    if ev != EVENT:
        return
    with _lock:
        _st["total"] += 1
        if not _st["ready"]:
            return
        name = str(k.get("fun_name"))
        _st["after"] += 1
        _st["names"][name] = _st["names"].get(name, 0) + 1
        cb = _st["cb"]
    if cb is not None:
        try:
            cb(name, secs)
        except Exception:
            pass


def mark_ready(on_compile=None):
    """From now on every compile is a compile after ready; on_compile(name, seconds) is called for each."""
    install()
    with _lock:
        _st["ready"], _st["cb"] = True, on_compile


def counts():
    """-> dict(compiles, compiles_after_ready, after_ready_by_name). A count this process does not keep is None, never 0:
    compiles before install(), compiles_after_ready before mark_ready() (a server before ready, or a process whose engine
    runs in another process, such as the HTTP front of runtime/front.py, has no count of its own)."""
    with _lock:
        return dict(compiles=_st["total"] if _st["on"] else None, compiles_after_ready=_st["after"] if _st["ready"] else None,
                    after_ready_by_name=dict(_st["names"]))


def _leaf(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return [list(np.shape(x)), str(x.dtype), bool(getattr(x, "weak_type", False))]
    if isinstance(x, bool):
        return ["bool", x]
    if isinstance(x, int):
        return ["int", "weak"]
    if isinstance(x, float):
        return ["float", "weak"]
    return [type(x).__name__]


def signature(name, args, statics=None):
    """One program's signature: name, static arguments, and (shape, dtype, weak type) of each argument leaf."""
    import jax
    return json.dumps([name, sorted((statics or {}).items()), [_leaf(x) for x in jax.tree.leaves(args)]],
                      sort_keys=True, default=str)


def list_sha(sigs):
    """sha256 of the sorted, de-duplicated signatures."""
    h = hashlib.sha256()
    for s in sorted(set(sigs)):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ executable table
# A start served from the compile cache still traces and lowers every program (JAX needs the lowered program to compute
# the cache key), then loads it, then traces it again on its first call: on a TPU v6e-1 (Qwen3.8-27B, ctx 2048 x 8) that
# is most of the 46 s warm-up. The table keeps each program as the executable itself, keyed by its call: a start loads the
# executables (no trace, no lowering) and every call goes straight to its executable. A call whose key is not in the table
# runs through jit as before (after ready it is a compile after ready, counted as always).
#
# Trust: a serialized executable is code (JAX's own format is a pickle, and unpickling can run code). The table is local
# to this machine and user: a start that finds none compiles and saves one for the next start. Nothing in a table folder
# is unpickled before all of this holds: the folder and its parent are real folders (not links) owned by this user and not
# writable by group or others; the folder holds manifest.json and exactly the files it lists, each a regular file (not a
# link) owned by this user, not writable by group or others; each file's size and sha256 equal the manifest's; the
# manifest's meta equals this start's (format, jax, jaxlib, backend and platform version, device kind and count, and the
# deployment id, which binds the engine source, chip topology, model, components and drafter: runtime/compile_key.py).
# Anything else refuses the whole table and every program compiles. After unpickling a verified file, the call key it
# holds must be the one the manifest lists for it (else the whole table is refused); an executable that then fails to
# load is skipped and compiled. A save writes the whole table into a new owner-only folder and renames it into place
# (an existing folder is never written into), so a table folder is complete or absent. Every read or write error is a
# refusal (the start compiles), never a crash.
FORMAT = 2
MANIFEST = "manifest.json"
_NAME = re.compile(r"[0-9a-f]{40}\.exe")
_HEX = re.compile(r"[0-9a-f]{64}")


def _key_leaf(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return (tuple(np.shape(x)), str(x.dtype), bool(getattr(x, "weak_type", False)))
    if isinstance(x, bool):
        return ("bool", x)
    return (type(x).__name__,)


def table_meta(deployment_id=None):
    """What a saved table must match to be used: format, jax / jaxlib, backend and its platform version (libtpu), device
    kind and count, and the deployment (runtime/compile_key.py)."""
    import jax
    import jaxlib
    d = jax.devices()
    return dict(format=FORMAT, jax=jax.__version__, jaxlib=jaxlib.__version__, backend=jax.default_backend(),
                platform_version=str(getattr(d[0].client, "platform_version", "")), device_kind=d[0].device_kind,
                devices=len(d), deployment_id=deployment_id)


class Refused(Exception):
    """A table that is not used: the reason."""


def key_sha(k):
    return hashlib.sha256(json.dumps(k, default=str).encode()).hexdigest()


def _owner_checks(st, what, kind):
    import os
    import stat
    uid = os.getuid() if hasattr(os, "getuid") else None
    if uid is None:
        raise Refused("no file owner check on this system")
    if stat.S_ISLNK(st.st_mode):
        raise Refused(f"{what} is a symlink")
    if not (stat.S_ISDIR if kind == "dir" else stat.S_ISREG)(st.st_mode):
        raise Refused(f"{what} is not a {'folder' if kind == 'dir' else 'regular file'}")
    if st.st_uid != uid:
        raise Refused(f"{what} belongs to another user")
    if st.st_mode & 0o022:
        raise Refused(f"{what} is writable by group or others")


def _read(path, what, cap):
    """The bytes of a regular file this user owns and only it can write (no link followed, nothing else opened)."""
    import os
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        _owner_checks(st, what, "file")
        if st.st_size > cap:
            raise Refused(f"{what} is larger than {cap} bytes")
        return f.read(cap + 1)


def _manifest(raw, meta):
    try:
        m = json.loads(raw)
    except ValueError as e:
        raise Refused(f"{MANIFEST} is not JSON ({e})"[:200])
    if not isinstance(m, dict) or not isinstance(m.get("meta"), dict) or not isinstance(m.get("files"), dict):
        raise Refused(f"{MANIFEST} is not a table manifest")
    if m["meta"] != meta:
        have = m["meta"]
        diff = sorted(k for k in set(have) | set(meta) if have.get(k) != meta.get(k))
        raise Refused("built for another " + ", ".join(diff))
    pl = m.get("program_list_sha")
    if pl is not None and not (isinstance(pl, str) and _HEX.fullmatch(pl)):
        raise Refused(f"{MANIFEST}: bad program_list_sha")
    for n, r in m["files"].items():
        if not (_NAME.fullmatch(n) and isinstance(r, dict) and isinstance(r.get("sha256"), str)
                and _HEX.fullmatch(r["sha256"]) and isinstance(r.get("key_sha256"), str) and _HEX.fullmatch(r["key_sha256"])
                and r["key_sha256"][:40] + ".exe" == n and type(r.get("bytes")) is int and r["bytes"] >= 0):
            raise Refused(f"{MANIFEST}: bad entry {str(n)[:60]}")
    return m


class TableFailed(RuntimeError):
    """A program loaded from a saved executable table raised when it was called (at launch, or in the executable's own
    argument check after programs.proven passed it). Table.fail has quarantined the table's folder and dropped
    the table: the call never runs through jit in its place (that would be a compile after ready). The caller ends the
    rows of that call and the engine's process ends, never warming again where it said ready (Server.table_failed,
    Worker.table_failed: behind the HTTP front a new worker starts without the table).
    info = dict(program, key, error, quarantined)."""

    def __init__(self, info):
        super().__init__(f"executable table: {info['program']} failed: {info['error']}")
        self.info = info


def quarantine(d):
    """Make table folder d never load again, keeping it as evidence. First one atomic rename in the same parent to
    .refused-<its name>-<time>-<pid>, its time set to now (a start loads only d, and saves a new table there). If the
    rename fails, a file
    REFUSED written into d with the reason: a start refuses a table folder holding any file its manifest does not list,
    before reading any table bytes. -> (where it went or why not, closed): closed False when neither
    worked, and then nothing stops the next start from loading d (the caller must not serve again)."""
    import os
    import time
    d = os.path.abspath(d)
    q = os.path.join(os.path.dirname(d), f".refused-{os.path.basename(d)}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                                         f"-{os.getpid()}")
    try:
        os.rename(d, q)
    except OSError as e:
        why = f"not renamed: {type(e).__name__}: {e}"[:300]
    else:
        try:
            os.utime(q)     # its time is the refusal's (prune ages refusal evidence by it), not the table's save
        except OSError:
            pass
        return q, True
    p = os.path.join(d, "REFUSED")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(why + "\n")
        return f"{why}; refused in place: {p}", True
    except FileExistsError:
        return f"{why}; refused in place already: {p}", True
    except OSError as e:
        return f"{why}; not refused in place: {type(e).__name__}: {e}"[:600], False


def set_aside(d):
    """A table folder this start refused, or dropped for another program list: moved aside with quarantine() (kept as
    evidence until prune() removes it), so this start saves a new table in its place instead of every later start
    refusing the same folder. -> quarantine's (where it went or why not, closed), or None when there is nothing to
    move or the parent folder fails the owner checks (every load refuses a table there, and nothing is moved in a
    folder others can write)."""
    import os
    d = os.path.abspath(str(d))
    try:
        if not os.path.lexists(d):
            return None
        _owner_checks(os.lstat(os.path.dirname(d)), "the table's parent folder", "dir")
    except (Refused, OSError):
        return None
    return quarantine(d)


KEEP_TABLES, KEEP_S = 2, 14 * 86400
_ID = re.compile(r"[0-9a-f]{64}")                                   # a deployment id (compile_key.sha)
_PART = re.compile(r"\.([0-9a-f]{64})\.part-[a-z0-9_]{1,16}")       # Table._save's folder while it writes
_REFUSED = re.compile(r"\.refused-([0-9a-f]{64})-\d{8}T\d{6}Z-\d+")  # quarantine()'s name
_LOCK = re.compile(r"\.([0-9a-f]{64})\.lock")


def lock_path(d):
    """The lock file of table folder d: <parent>/.<name>.lock."""
    import os
    d = os.path.abspath(str(d))
    return os.path.join(os.path.dirname(d), f".{os.path.basename(d)}.lock")


def _flock(path, how):
    """Open (create 0600, no link followed, not inherited by a program this process starts) and flock the file at path,
    the open file being the one now at path (a lock on a file prune() unlinked meanwhile would hold nothing).
    -> (fd, created), or None when a non-blocking lock is taken elsewhere."""
    import fcntl
    import os
    for _ in range(5):
        created = not os.path.lexists(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            fcntl.flock(fd, how)
            st = os.fstat(fd)
            try:
                now = os.lstat(path)
            except FileNotFoundError:
                now = None
            if now is not None and (st.st_dev, st.st_ino) == (now.st_dev, now.st_ino):
                return fd, created
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
    raise OSError(f"{path} keeps changing")


def _own_folder(p, kind):
    """Check that p is a folder this user owns and only it can write, holding only regular files of this user with the
    names a table writes (manifest.json, <40 hex>.exe, REFUSED). -> (its file names, last change time); Refused if not."""
    import os
    st = os.lstat(p)
    _owner_checks(st, kind, "dir")
    names = os.listdir(p)
    for n in names:
        if not (n in (MANIFEST, "REFUSED") or _NAME.fullmatch(n)):
            raise Refused(f"{kind} holds {n[:60]}, which a table does not write")
        _owner_checks(os.lstat(os.path.join(p, n)), n, "file")
    return names, st.st_mtime


def prune(parent, keep, now=None):
    """After a start saved its table `keep` in folder parent (<cache>/aot), remove the executable tables no start needs
    (never raises). Under an exclusive lock on <parent>/.lock (one prune at a time):
    - deployment tables (folders named by a deployment id): `keep` stays; of the others, the newest KEEP_TABLES - 1 used
      in the last KEEP_S seconds stay (a start of the previous version finds its table), the rest go. A table's time is
      its save or the last start that used it (that start's lock file time), whichever is later;
    - refused tables (quarantine()'s .refused-<id>-..., or a table folder holding REFUSED) are evidence: the newest
      KEEP_TABLES from the last KEEP_S seconds stay, the rest go;
    - what an interrupted save left (.<id>.part-*) goes (_clean_parts: every part folder no live save holds);
    - a lock file no folder of its deployment uses any more goes.
    A table or save folder whose lock file another process holds (it loaded, saved or is saving that table) is never
    removed. Only this user's own table folders are touched: an entry with any other name is ignored, and a link, a
    folder another user owns or others can write, or a folder holding anything a table does not write is left as it is
    (named in `left`). Files are removed one by one, then the empty folder.
    -> dict(removed=[names], left={name: why}, error=None or why the prune did not run)"""
    import fcntl
    import os
    import time
    out = dict(removed=[], left={}, error=None)
    now = time.time() if now is None else now
    try:
        parent = os.path.abspath(str(parent))
        _owner_checks(os.lstat(parent), "the table's parent folder", "dir")
        g = _flock(os.path.join(parent, ".lock"), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if g is None:
            out["error"] = "another prune runs"
            return out
        try:
            _prune(parent, keep, now, out)
        finally:
            os.close(g[0])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def _prune(parent, keep, now, out):
    import fcntl
    import os
    tables, refused, ids = [], [], set()
    out["removed"], left = _clean_parts(parent)
    out["left"].update(left)
    for n in sorted(os.listdir(parent)):
        if n == keep:
            continue
        m = _ID.fullmatch(n) or _REFUSED.fullmatch(n)
        if not m:
            continue                                # not a name a table uses: never touched
        p = os.path.join(parent, n)
        try:
            names, t = _own_folder(p, "the folder")
        except (Refused, OSError) as e:
            out["left"][n] = str(e)[:200]
            continue
        if m.re is _REFUSED or "REFUSED" in names:
            refused.append((t, n, None))
        else:
            lk = lock_path(p)
            if os.path.lexists(lk):
                t = max(t, os.lstat(lk).st_mtime)
            tables.append((t, n, n))
    go = []
    for i, (t, n, i_d) in enumerate(sorted(tables, reverse=True)):
        if i >= KEEP_TABLES - 1 or now - t >= KEEP_S:
            go.append((n, i_d))
        else:
            ids.add(n)
    for i, (t, n, i_d) in enumerate(sorted(refused, reverse=True)):
        if i >= KEEP_TABLES or now - t >= KEEP_S:
            go.append((n, None))
    for n, i_d in go:
        p, held = os.path.join(parent, n), None
        try:
            if i_d is not None:                     # a table or save another process holds stays
                held = _flock(lock_path(os.path.join(parent, i_d)), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if held is None:
                    out["left"][n] = "in use by another process"
                    ids.add(i_d)
                    continue
            names, _ = _own_folder(p, "the folder")     # again, under the lock
            for f in names:
                os.unlink(os.path.join(p, f))
            os.rmdir(p)
            out["removed"].append(n)
        except (Refused, OSError) as e:
            out["left"][n] = str(e)[:200]
            if i_d is not None:
                ids.add(i_d)
                if held is not None and held[1]:    # the lock file this prune made: its time would read as a use
                    os.unlink(lock_path(os.path.join(parent, i_d)))
        finally:
            if held is not None:
                os.close(held[0])
    for n in sorted(os.listdir(parent)):           # lock files of deployments with no folder left
        m = _LOCK.fullmatch(n)
        if not m or m.group(1) in ids or m.group(1) == keep or os.path.lexists(os.path.join(parent, m.group(1))):
            continue
        lk = os.path.join(parent, n)
        try:
            held = _flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if held is not None:
                try:
                    os.unlink(lk)
                finally:
                    os.close(held[0])
        except OSError:
            pass


def _clean_parts(parent):
    """Remove what interrupted saves left in folder parent: every part folder (.<id>.part-*) that no live save holds. A
    save holds an exclusive lock on its part folder while it writes it (Table._save), so a part folder whose lock can be
    taken is one whose writer is gone, however young; one another save writes stays, so the part folders are never more
    than the saves running. Only this user's own folders holding only what a table writes are touched (_own_folder).
    Called before a save writes (a start after a crash loop of interrupted saves is not stopped by their debris) and by
    prune. A writer that takes no lock on its part folder (an older build of this code) looks like debris here: its
    save fails (that process keeps its programs in memory), or it may rename a folder with files missing into place,
    which the next start refuses before it reads any table bytes (the manifest's file set, sizes and sha256 are checked
    first), so that start compiles every program as with no table: the cost is one start's warm skip, never a program
    that was not checked. -> (removed names, {name: why it was left})"""
    import fcntl
    import os
    removed, left = [], {}
    for n in sorted(os.listdir(parent)):
        if not _PART.fullmatch(n):
            continue
        p, fd = os.path.join(parent, n), None
        try:
            _own_folder(p, "the folder")
            fd = os.open(p, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_CLOEXEC", 0))
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                left[n] = "being written by another process"
                continue
            names, _ = _own_folder(p, "the folder")     # again, under the lock
            for f in names:
                os.unlink(os.path.join(p, f))
            os.rmdir(p)
            removed.append(n)
        except (Refused, OSError) as e:
            left[n] = str(e)[:200]
        finally:
            if fd is not None:
                os.close(fd)
    return removed, left


class Table:
    """Executables by call key (see Dispatch.key), shared by one engine's Dispatch wrappers. list_sha = the program list
    of the engine's warm-up (Engine.warm sets it; a save records it). loaded_from = the folder this table was loaded from
    or saved to (None when it holds no table that passed the checks). loaded = keys of executables loaded from a saved
    table; ran = keys of executables that ran in this process (only those are saved: every entry of a saved table ran
    in the start that saved it)."""

    MANIFEST_CAP, ENTRY_CAP = 64 << 20, 16 << 30

    def __init__(self):
        self.ex, self.names, self.saved, self.unser, self.lock = {}, {}, set(), {}, threading.Lock()
        self.list_sha, self.loaded_from = None, None
        self.loaded, self.ran, self.failures = set(), set(), []
        self.held = None        # (folder, fd): the shared lock this table holds on the folder it uses (hold)

    def hold(self, d):
        """Hold a shared lock on table folder d's lock file (lock_path) while this table uses d, from its load or save
        until the process ends or the table uses another folder (the lock of the one before is let go): prune() never
        removes a table a live process holds, since a process that cannot take the lock does not use the folder (load
        refuses it and every program compiles; save writes nothing). The lock file's time is set to now (prune reads it
        as the last use). -> None when the lock is held, else why not (never raises)."""
        import fcntl
        import os
        d = os.path.abspath(str(d))
        if self.held and self.held[0] == d:
            return None
        old, self.held = self.held, None
        if old:
            os.close(old[1])
        try:       # only in a parent folder that passes the owner checks (no lock file where others can write)
            _owner_checks(os.lstat(os.path.dirname(d)), "the table's parent folder", "dir")
            fd = _flock(lock_path(d), fcntl.LOCK_SH)[0]
        except Exception as e:
            return f"the table's lock could not be taken: {type(e).__name__}: {e}"[:300]
        self.held = (d, fd)
        try:
            os.utime(fd)
        except OSError:
            pass
        return None

    def put(self, k, exe, name):
        with self.lock:
            self.ex[k], self.names[k] = exe, name

    def discard(self):
        """Drop every executable (they compile again through warm or jit)."""
        with self.lock:
            self.ex, self.names, self.saved, self.loaded_from = {}, {}, set(), None
            self.loaded.clear()
            self.ran.clear()

    def fail(self, k, name, e):
        """A program of this table loaded from a saved one raised (k = its call key, None when the failure is not one
        call's, e.g. at the warm-up's final barrier): quarantine the folder it was loaded from and drop the table.
        -> TableFailed (the caller raises it); its info["closed"] is False when the folder could be neither renamed nor
        refused in place."""
        with self.lock:
            src = self.loaded_from if self.loaded else None
        q, closed = quarantine(src) if src else (None, True)
        info = dict(program=name, key=key_sha(k)[:16] if k is not None else None, error=f"{type(e).__name__}: {e}"[:300],
                    quarantined=q, closed=closed)
        self.failures.append(info)
        self.discard()
        return TableFailed(info)

    @staticmethod
    def _file(k):
        return key_sha(k)[:40] + ".exe"

    def _unser_by_name(self):
        un = {}
        for k in self.unser:
            un[self.names.get(k)] = un.get(self.names.get(k), 0) + 1
        return un

    def save(self, d, meta):
        """Write the table into folder d (never raises). The folder is written whole in a new owner-only folder next to
        it, then renamed to d; an existing d is left as it is (refused), unless it is the table this one was loaded from
        or saved to (nothing to write). An executable the backend cannot serialize (the CPU client refuses some, e.g. a
        sort comparator: "`LessThan` is not serializable") is not written: its program compiles at each start as without a
        table. A program that never ran in this process is not written either (not_run): a later start may skip running
        a loaded program in its warm-up (Engine.warm), so each one it loads ran once on this deployment.
        -> dict(saved, entries, unserializable {name: count}, not_run, bytes, seconds, refused)."""
        import time
        t0 = time.perf_counter()
        try:
            n, nb, why = self._save(str(d), meta)
        except Exception as e:      # a cache: a failed write never stops the server (it has the programs in memory)
            n, nb, why = 0, 0, f"not saved: {type(e).__name__}: {e}"[:300]
        return dict(saved=n, entries=len(self.ex), unserializable=self._unser_by_name(),
                    not_run=sum(1 for k in self.ex if k not in self.ran and k not in self.saved), bytes=nb,
                    seconds=round(time.perf_counter() - t0, 2), refused=why)

    def _save(self, d, meta):
        import fcntl
        import os
        import pickle
        import shutil
        import tempfile
        from jax.experimental import serialize_executable as SE
        d = os.path.abspath(d)
        if self.loaded_from == d:      # the table is there: only say which programs it cannot hold
            for k, exe in list(self.ex.items()):
                if k not in self.saved and k not in self.unser:
                    try:
                        SE.serialize(exe)
                    except Exception as e:
                        self.unser[k] = f"{type(e).__name__}: {e}"[:200]
            return 0, 0, None
        if os.path.lexists(d) and not (os.path.isdir(d) and not os.path.islink(d) and not os.listdir(d)):
            return 0, 0, f"{d} is there and this start did not use it: left as it is (remove it to save a new table)"
        parent = os.path.dirname(d)
        os.makedirs(parent, mode=0o700, exist_ok=True)
        _owner_checks(os.lstat(parent), "the table's parent folder", "dir")
        why = self.hold(d)      # before the folder exists: a prune by another start never takes it while it is written
        if why is not None:
            return 0, 0, f"not saved: {why}"
        _clean_parts(parent)    # what interrupted saves left, before this one writes
        tmp = tempfile.mkdtemp(prefix="." + os.path.basename(d) + ".part-", dir=parent)     # mode 0700
        pfd = None
        try:
            # held while it is written: _clean_parts removes only part folders no live save holds
            pfd = os.open(tmp, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
            fcntl.flock(pfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            files, n, nb, wrote = {}, 0, 0, []
            for k, exe in list(self.ex.items()):
                if k in self.unser or k not in self.ran:
                    continue
                try:
                    blob, it, ot = SE.serialize(exe)
                except Exception as e:
                    self.unser[k] = f"{type(e).__name__}: {e}"[:200]
                    continue
                raw = pickle.dumps(dict(key=k, name=self.names.get(k), blob=blob, in_tree=it, out_tree=ot))
                name = self._file(k)
                fd = os.open(os.path.join(tmp, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), key_sha256=key_sha(k), bytes=len(raw),
                                   name=self.names.get(k))
                n, nb = n + 1, nb + len(raw)
                wrote.append(k)
            man = dict(format=FORMAT, meta=meta, program_list_sha=self.list_sha, files=files)
            fd = os.open(os.path.join(tmp, MANIFEST), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(man, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, d)       # onto an absent or empty folder only; a folder filled meanwhile makes this fail
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)      # this call's own partial folder
            raise
        finally:
            if pfd is not None:
                os.close(pfd)
        with self.lock:
            self.saved.update(wrote)        # the entries written: a program that never ran stays unsaved (not_run)
            self.loaded_from = d
        return n, nb, None

    def load(self, d, meta, workers=8):
        """Load the table in folder d when it passes every check above (never raises). A table that fails any check is
        refused whole (nothing loaded); an entry whose bytes match the manifest but whose executable fails to load is
        skipped and counted (its program compiles in the warm-up instead).
        -> dict(loaded, failed, seconds, refused, program_list_sha)."""
        import os
        import time
        t0 = time.perf_counter()
        d = os.path.abspath(str(d))
        out = dict(loaded=0, failed=0, seconds=0.0, refused=None, program_list_sha=None)
        why = None if not os.path.lexists(d) else self.hold(d)    # before any read: a prune waits, or skips this table
        try:
            if why is not None:     # not held: not used (a prune could remove it while it is read)
                raise Refused(why)
            got = self._load(d, meta, workers)
        except Refused as e:
            out["refused"] = str(e)[:300]
        except Exception as e:      # any read error: compile instead
            out["refused"] = f"{type(e).__name__}: {e}"[:300]
        else:
            ok, failed, pl = got
            for k, exe, name in ok:
                self.put(k, exe, name)
            with self.lock:
                self.saved.update(k for k, _, _ in ok)
                self.loaded.update(k for k, _, _ in ok)
                self.loaded_from = d
            out.update(loaded=len(ok), failed=failed, program_list_sha=pl)
        out["seconds"] = round(time.perf_counter() - t0, 2)
        return out

    def _load(self, d, meta, workers):
        import os
        import pickle
        from concurrent.futures import ThreadPoolExecutor
        from jax.experimental import serialize_executable as SE
        if not os.path.lexists(d):
            raise Refused("no table")
        _owner_checks(os.lstat(os.path.dirname(d)), "the table's parent folder", "dir")
        _owner_checks(os.lstat(d), "the table folder", "dir")
        m = _manifest(_read(os.path.join(d, MANIFEST), MANIFEST, self.MANIFEST_CAP), meta)
        names = set(os.listdir(d))
        extra, missing = sorted(names - set(m["files"]) - {MANIFEST}), sorted(set(m["files"]) - names)
        if extra:
            raise Refused(f"files the manifest does not list: {', '.join(extra)[:120]}")
        if missing:
            raise Refused(f"files the manifest lists are missing: {', '.join(missing)[:120]}")
        for n in m["files"]:            # say what is wrong before opening anything (a link, a pipe, a folder)
            _owner_checks(os.lstat(os.path.join(d, n)), n, "file")

        def verified(n):
            r = m["files"][n]
            raw = _read(os.path.join(d, n), n, min(self.ENTRY_CAP, r["bytes"]))
            if len(raw) != r["bytes"] or hashlib.sha256(raw).hexdigest() != r["sha256"]:
                raise Refused(f"{n} does not match the manifest")
            return n, raw
        order = sorted(m["files"])
        w = max(1, min(workers, len(order)))
        with ThreadPoolExecutor(w) as ex:
            raws = dict(ex.map(verified, order))           # every byte checked before anything is unpickled

        def one(n):
            e = pickle.loads(raws.pop(n))
            k = e["key"] if isinstance(e, dict) else None
            if not isinstance(k, tuple) or key_sha(k) != m["files"][n]["key_sha256"]:
                raise Refused(f"{n} holds another call's program than the manifest lists")
            try:
                return k, SE.deserialize_and_load(e["blob"], e["in_tree"], e["out_tree"]), e.get("name")
            except Exception:
                return None
        with ThreadPoolExecutor(w) as ex:
            got = list(ex.map(one, order))
        ok = [g for g in got if g is not None]
        return ok, len(got) - len(ok), m.get("program_list_sha")


class Dispatch:
    """A jitted function behind the table: same call (statics as keywords), and every public method of the jit function
    (.lower, .trace, .eval_shape, .clear_cache) is the jit's own, so a caller that inspects or lowers the program gets
    what the jit gives. skip = leading arguments that never change for the engine (the weights) and are left out of the
    key; for an executable built in this process, a changed weights tree is caught by its own input check and the call
    runs through jit (a counted compile). An executable loaded from a saved table that raises, in its input check or at
    launch, is a table failure (TableFailed: Table.fail quarantines and drops the table), never a jit call in its place."""

    def __init__(self, name, jitted, table, skip=0):
        self.__name__, self.jitted, self.table, self.skip = name, jitted, table, skip

    def key(self, args, kw):
        import jax
        leaves, tree = jax.tree.flatten(args[self.skip:])
        return (self.__name__, tuple(sorted((k, str(v)) for k, v in kw.items())), str(tree),
                tuple(_key_leaf(x) for x in leaves))

    def __call__(self, *args, **kw):
        if self.table.ex:
            k = self.key(args, kw)
            exe = self.table.ex.get(k)
            if exe is not None:
                try:
                    out = exe(*args)
                    if k in self.table.loaded and k not in self.table.ran:
                        # first run in this process of a program loaded from the saved table (one the warm-up did not
                        # run): wait for it here, so a device fault reported after the call returned (JAX runs
                        # asynchronously) is a table failure too, not a request error that leaves the table in use
                        import jax
                        jax.block_until_ready(out)
                except Exception as e:
                    if k in self.table.loaded:      # from the saved table: quarantine and drop it, never jit in its place
                        raise self.table.fail(k, self.__name__, e) from e
                    if not isinstance(e, (TypeError, ValueError)):
                        raise
                    # built here and its input check refused the arguments (checked before it runs): jit below
                else:
                    if k not in self.table.ran:
                        self.table.ran.add(k)
                    return out
        return self.jitted(*args, **kw)

    def lower(self, *args, **kw):
        return self.jitted.lower(*args, **kw)

    def trace(self, *args, **kw):
        return self.jitted.trace(*args, **kw)

    def eval_shape(self, *args, **kw):
        return self.jitted.eval_shape(*args, **kw)

    def clear_cache(self):      # the jit's compiled programs; the table's entries stay
        return self.jitted.clear_cache()

    def _cache_size(self):      # programs compiled through jit (calls the table did not hold), as jax's own counter
        return self.jitted._cache_size()

    def build(self, *args, **kw):
        """Compile this call's program into the table (once). -> True if it was compiled now."""
        k = self.key(args, kw)
        if k in self.table.ex:
            return False
        self.table.put(k, self.jitted.lower(*args, **kw).compile(), self.__name__)
        return True


# ------------------------------------------------------------- a loaded program proven without running it (Engine.warm)
def call_check(exe, args):
    """The checks jax makes when the compiled executable exe is called with args, before it runs: the argument pytree
    (stages.Compiled.call), each argument's shape and dtype, and a committed array's sharding and layout
    (pxla.MeshExecutable.call), made here without running it. -> None when that call would run exe; else why not. Any
    error reading exe's fields is a why-not too (the warm-up then runs the program as before)."""
    try:
        from jax._src import core, tree_util
        from jax._src.interpreters import pxla
        p, m = exe._params, exe._executable
        if p.is_high or getattr(m, "_mut", None):
            return "not checked: high-level types or mutable arrays"
        flat, tree = tree_util.tracing_registry.flatten((tuple(args), {}))
        if tree != p.in_tree:
            return "argument pytree differs"
        full = [*p.const_args, *flat]
        kept = tuple(a for i, a in enumerate(full) if i in m._kept_var_idx)
        if m._all_args_info is not None:
            pxla.check_arg_avals_for_call(m._all_args_info.in_avals, [core.shaped_abstractify(a) for a in full],
                                          m._all_args_info.debug_info)
        else:
            pxla.check_arg_avals_for_call(m.in_avals, [core.shaped_abstractify(a) for a in kept],
                                          core.DebugInfo("MeshExecutable", "<unknown>", ("",) * len(kept), None))
        pxla.check_array_xla_sharding_layout_match(kept, m._in_shardings, m._xla_in_layouts, ("",) * len(kept))
        return None
    except (TypeError, ValueError) as e:
        return f"{type(e).__name__}: {e}"[:200]
    except Exception as e:
        return f"not checked: {type(e).__name__}: {e}"[:200]


def memory_need(exe):
    """Device bytes a run of exe adds while it runs: temporaries + outputs - outputs that reuse donated inputs (XLA's
    memory analysis of the compiled program, per device). None when the backend does not say."""
    try:
        m = exe.memory_analysis()
        return int(m.temp_size_in_bytes) + int(m.output_size_in_bytes) - int(m.alias_size_in_bytes)
    except Exception:
        return None


def device_free():
    """The fewest bytes free on any local device (bytes_limit - bytes_in_use), or None when the backend keeps no device
    memory count (the CPU client: host memory)."""
    import jax
    free = []
    for d in jax.local_devices():
        try:
            st = d.memory_stats()
        except Exception:
            return None
        if not st or "bytes_limit" not in st:
            return None
        free.append(int(st["bytes_limit"]) - int(st.get("bytes_in_use", 0)))
    return min(free) if free else None


def proven(table, f, args, kw, free):
    """None when the call f(*args, **kw) need not run once in the warm-up: its program was loaded from a saved table
    (each entry of which ran in the start that saved it, on this deployment), the call would run that executable
    (call_check) and the device has room for it (free: device_free(), None = no device count). Else why it runs."""
    if not hasattr(f, "key"):
        return "not a table program"
    k = f.key(args, kw)
    exe = table.ex.get(k)
    if exe is None or k not in table.loaded:
        return "compiled in this start"
    why = call_check(exe, args)
    if why:
        return why
    if free is not None:
        need = memory_need(exe)
        if need is None:
            return "memory use unknown"
        if need > free:
            return f"needs {need} B, {free} B free"
    return None
