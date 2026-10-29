"""Quitting the MASHBOX app (the window sends SIGTERM and waits 12 s): every `mashbox tpu` job gets Ctrl-C at once,
anything still running after the time is stopped, and no new job starts while it closes."""
import json
import signal
import subprocess
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

from mashbox.ui import app


def test_quit_closes_every_job_within_the_time():
    app.JOBS.clear()
    app.CLOSING.clear()
    polite = app.Job("up", "a", ["sleep", "30"])                                         # ends on Ctrl-C
    stuck = app.Job("tunnel", "b", [sys.executable, "-u", "-c",
                                    "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('ready'); time.sleep(30)"])
    app.JOBS.update(a=polite, b=stuck)
    try:
        t0 = time.time()
        while "ready" not in stuck.lines and time.time() - t0 < 20:
            time.sleep(0.05)
        t0 = time.time()
        app.close_jobs(1.0)
        took = time.time() - t0
        polite.proc.wait(5)
        stuck.proc.wait(5)
        assert polite.proc.returncode == -signal.SIGINT and stuck.proc.returncode == -signal.SIGKILL
        assert took < 2.5 and app.CLOSING.is_set()
    finally:
        for j in (polite, stuck):
            if j.running():
                j.proc.kill()
        app.JOBS.clear()
        app.CLOSING.clear()


def test_no_new_job_while_closing(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    app.CLOSING.set()
    try:
        import re
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()
        req = urllib.request.Request(url + "/api/tpu/up", data=json.dumps({"model": "qwen3-8-27b"}).encode(),
                                     headers={"Content-Type": "application/json", "X-Mashbox-Token": tok})
        try:
            urllib.request.urlopen(req, timeout=10)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        assert code == 503 and not app.JOBS
    finally:
        app.CLOSING.clear()
        srv.shutdown()
        wsrv.shutdown()


def test_run_quick_output_and_a_command_that_outlives_the_app(tmp_path):
    r = app.run_quick([sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err')"], 30)
    assert r["code"] == 0 and "out" in r["out"] and "err" in r["out"]
    marker = tmp_path / "deleted"                         # like `tpu down`: prints, then does the work
    started = tmp_path / "started"
    child = (f"import os, time\np = os.getppid(); open({str(started)!r}, 'w').close(); t0 = time.time()\n"
             "while os.getppid() == p and time.time() - t0 < 20: time.sleep(0.01)\n"     # until the app has quit
             f"print('Deleting ...', flush=True); open({str(marker)!r}, 'w').write('ok')")
    parent = (f"import os, sys, threading, time; sys.path.insert(0, {str(Path(app.__file__).parents[3])!r})\n"
              "from mashbox.ui import app\n"
              f"threading.Thread(target=lambda: app.run_quick([sys.executable, '-c', {child!r}], 60), daemon=True).start()\n"
              f"t0 = time.time()\nwhile not os.path.exists({str(started)!r}) and time.time() - t0 < 20: time.sleep(0.01)\n"
              "os._exit(0)\n")
    subprocess.run([sys.executable, "-c", parent], timeout=60)
    assert started.exists()                               # the app quit while the command was running
    t0 = time.time()
    while not marker.exists() and time.time() - t0 < 15:
        time.sleep(0.1)
    assert marker.exists() and marker.read_text() == "ok"
