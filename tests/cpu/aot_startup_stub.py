"""Worker factory for test_aot_table_startup.py (not a test file): the stand-in engine of test_aot_table.py built and
warmed as `mashbox run` builds and warms it in its worker (cli.aot_dir, then the real cli.warm: two short requests, then
Engine.warm), with the table saved by an earlier start at <cache>/aot/dep-a.

A life that loads the table (MASHBOX_AOT not "0") fails in its warm-up, before ready, where `where` says:
- "short": every program loaded from the table raises when called (a device fault at launch), so the first of cli.warm's
  two requests fails on a loaded program (programs.Dispatch -> Table.fail -> TableFailed);
- "barrier": the loaded programs run, and Engine.warm's final jax.block_until_ready reports a device fault (one no single
  table call raised).
closed=False forces the quarantine's two steps to fail (the rename of the table folder and the REFUSED file in it raise
PermissionError), so the folder stays loadable on disk. A life without the table (MASHBOX_AOT=0) is left as it is.
Each life logs what it saw (the table used or not, MASHBOX_AOT, its compiles, the table failure) to `log`."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _log(path, **e):
    with open(path, "a") as f:
        f.write(json.dumps(dict(e, pid=os.getpid())) + "\n")


class Chat:
    def chat_ids(self, msgs, *a, **k):
        return [5, 6, 7]


def make(cache, where="short", closed=True, log=None):
    import jax
    from mashbox import cli
    from mashbox.runtime import programs
    import test_aot_table as T
    e = T.engine()
    e.deployment_id = "dep-a"
    e.aot_dir = cli.aot_dir(cache, e)
    table = os.path.join(cache, "aot", "dep-a")
    if e.aot_dir and not closed:
        real_rename, real_open = os.rename, os.open

        def rename(a, b, *x, **k):
            if os.path.abspath(a) == table:
                raise PermissionError(13, "rename refused (test)", a)
            return real_rename(a, b, *x, **k)

        def open_(p, flags, *x, **k):
            if os.path.basename(str(p)) == "REFUSED":
                raise PermissionError(13, "write refused (test)", p)
            return real_open(p, flags, *x, **k)
        os.rename, os.open = rename, open_
    if e.aot_dir and where == "short":
        real_load = e.aot.load

        def load(*a, **k):
            out = real_load(*a, **k)

            def broken(*x):
                raise RuntimeError("device fault at launch (test)")
            for key in list(e.aot.loaded):
                e.aot.ex[key] = broken
            return out
        e.aot.load = load
    if e.aot_dir and where == "barrier":
        real_bur, fired = jax.block_until_ready, []

        def bur(x):
            f = sys._getframe(1)
            if (not fired and e.aot.loaded and f.f_code.co_name == "warm"
                    and f.f_globals.get("__name__") == "mashbox.runtime.engine"):
                fired.append(1)
                raise RuntimeError("device fault reported at the final barrier (test)")
            return real_bur(x)
        jax.block_until_ready = bur
    try:
        cli.warm(e, Chat())
    except programs.TableFailed as ex:
        _log(log, ev="table_failed", aot=os.environ.get("MASHBOX_AOT"), **ex.info)
        raise
    _log(log, ev="warm", aot=os.environ.get("MASHBOX_AOT"), used=e.aot_dir is not None,
         compiles=programs.counts()["compiles"])
    return e
