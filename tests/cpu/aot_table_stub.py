"""Worker factory for test_aot_table_restart.py (not a test file): api_stub_engine's scripted engine with a stand-in for a
saved executable table, so the front + worker recovery from a table failure runs with real processes.

A worker life "loads" the table when MASHBOX_AOT is not "0" and the table folder is present and holds no REFUSED file
(as programs.Table.load refuses a folder holding a file its manifest does not list): its warm-up compiles nothing (the
programs came from the table). Its loaded program fails in a request at the step chosen by `fail` ("first": the first
step, before any token; "token": once a row has a token): the folder is quarantined with programs.quarantine (unless
closed=False: a folder that could be neither renamed nor refused) and programs.TableFailed is raised, as Table.fail does.
A life without the table compiles in its warm-up (one real XLA compile, before ready) and then takes slow_s seconds
more (a start that compiles every program). Each warm-up logs what that life saw: the table loaded or not, MASHBOX_AOT,
this process's compiles."""
import os
import time

import api_stub_engine as S
from mashbox.runtime import programs


def loadable(table):
    return os.environ.get("MASHBOX_AOT") != "0" and os.path.isdir(table) and not os.path.exists(
        os.path.join(table, "REFUSED"))


class TableEngine(S.StubEngine):
    def __init__(self, table, fail="first", closed=True, slow_s=0.0, **kw):
        super().__init__(**kw)
        self.table, self.fail, self.closed, self.slow_s = table, fail, closed, slow_s
        self.loaded, self.failed = loadable(table), False

    def warm(self, workers=8):
        if not self.loaded:
            import jax
            import numpy as np
            jax.jit(lambda x: x * 3 + 1)(np.zeros(5, np.float32)).block_until_ready()    # compiled before ready
            time.sleep(self.slow_s)
        self._log(ev="warm", loaded=self.loaded, aot=os.environ.get("MASHBOX_AOT"),
                  compiles=programs.counts()["compiles"])

    def step(self, waiting=False):
        rows = [r for r in self.rows if r is not None]
        due = self.fail == "first" or any(r["ids"] for r in rows)
        if self.loaded and not self.failed and rows and due:
            self.failed = True
            q, closed = programs.quarantine(self.table) if self.closed else ("not renamed, not refused (test)", False)
            self._log(ev="table_failed", quarantined=q, closed=closed)
            raise programs.TableFailed(dict(program="decode", key="0" * 16, error="RuntimeError: device fault (test)",
                                            quarantined=q, closed=closed))
        return super().step(waiting)


def make(table, fail="first", closed=True, slow_s=0.0, **kw):
    return TableEngine(table, fail=fail, closed=closed, slow_s=slow_s, **kw)
