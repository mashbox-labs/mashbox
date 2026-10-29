"""Ctrl-C stops the Hy-Nomas app, however it was started, and the app lets go of everything it holds.

A shell starts a background command (`cmd &` or `nohup cmd &` in a script, which is how agents and scripts start the app)
with SIGINT ignored. Python then never raises KeyboardInterrupt, so before the fix `kill -INT <pid>` and Ctrl-C did
nothing and the app kept the TPU until it was killed. These tests start the stand-in app (`--ui --stub`) as a real process
and stop it the way people do: `kill -INT <pid>` (the app only), Ctrl-C in a terminal (SIGINT to the whole foreground
process group: the app and its ffmpeg child at once, while frames flow) and SIGTERM."""
import os
import signal
import socket
import subprocess
import sys
import json
import time
import urllib.request

import pytest

pytest.importorskip("jax")


def held_port():
    """A port kept bound here (SO_REUSEADDR, never listening) until closed: no other process can take it before the app
    binds it, and the app (an http.server server, SO_REUSEADDR) still can (api_stub_engine.held_port)."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    return s


def children(pid):
    out = set()
    for t in os.listdir("/proc/%d/task" % pid):
        try:
            with open("/proc/%d/task/%s/children" % (pid, t)) as f:
                out |= {int(x) for x in f.read().split()}
        except FileNotFoundError:   # that thread ended after listdir; the kernel moves its children to another thread
            continue
    return out


def gone(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"   # a zombie has exited
    except FileNotFoundError:
        return True


def start_app(ignore_sigint, terminal=False):
    pytest.importorskip("imageio_ffmpeg")
    import mashbox
    if not os.path.isdir("/proc/self/task"):
        pytest.skip("needs Linux /proc")
    src = os.path.dirname(os.path.dirname(os.path.abspath(mashbox.__file__)))
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""), JAX_PLATFORMS="cpu")
    hold = held_port()                  # closed with this test process; it never listens, so a stopped app's port refuses
    port = hold.getsockname()[1]

    def pre():
        if ignore_sigint:   # what `cmd &` in a script (and nohup from a script) hands the command
            signal.signal(signal.SIGINT, signal.SIG_IGN)
    p = subprocess.Popen([sys.executable, "-m", "mashbox", "run", "hy-nomas", "--ui", "--stub", "--port", str(port)],
                         env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         preexec_fn=pre, start_new_session=terminal)   # terminal: its own process group, as a shell job
    t0 = time.time()
    while time.time() - t0 < 60:
        if p.poll() is not None:
            raise AssertionError("app ended at start: %s" % p.stderr.read().decode()[-2000:])
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=2).read()
            break
        except OSError:
            time.sleep(0.3)
    else:
        p.kill()
        raise AssertionError("app did not answer in 60 s")
    return p, port


def wait_for_frames(port, n=48, timeout=60):
    """Until the app has sent n frames (so ffmpeg is busy when the signal comes)."""
    t0, frames = time.time(), 0
    while time.time() - t0 < timeout:
        try:
            frames = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/api/state" % port, timeout=2).read())["frames"]
            if frames > n:
                return frames
        except OSError:
            pass
        time.sleep(0.3)
    raise AssertionError("only %d frames in %d s" % (frames, timeout))


def stop_and_check(p, port, sig, group=False):
    kids = children(p.pid)
    assert kids, "the app should run its video encoder (ffmpeg) as a child"
    t0 = time.time()
    if group:                                              # Ctrl-C in a terminal: the app and ffmpeg get it together
        os.killpg(p.pid, sig)
    else:
        os.kill(p.pid, sig)
    try:
        rc = p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise AssertionError("the app did not stop within 20 s of %s" % signal.Signals(sig).name)
    err = p.stderr.read().decode()
    assert rc == 0, "exit code %s after %s:\n%s" % (rc, signal.Signals(sig).name, err[-2000:])
    assert '"event": "stopped"' in err, err[-2000:]
    assert "Traceback" not in err and "Broken pipe" not in err, "not a clean stop:\n" + err[-2000:]
    t1 = time.time()
    while time.time() - t1 < 10 and not all(gone(k) for k in kids):
        time.sleep(0.1)
    assert all(gone(k) for k in kids), "the video encoder outlived the app"
    with pytest.raises(OSError):
        urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=2)
    return time.time() - t0


@pytest.mark.parametrize("ignore_sigint", [True, False], ids=["started-from-a-script", "sigint-not-ignored"])
def test_kill_int_by_pid_stops_the_app(ignore_sigint):
    p, port = start_app(ignore_sigint)
    stop_and_check(p, port, signal.SIGINT)


@pytest.mark.parametrize("ignore_sigint", [False, True], ids=["terminal", "terminal-nohup"])
def test_ctrl_c_in_a_terminal_stops_the_app(ignore_sigint):
    p, port = start_app(ignore_sigint, terminal=True)
    wait_for_frames(port)
    stop_and_check(p, port, signal.SIGINT, group=True)


def test_sigterm_stops_the_app_cleanly():
    p, port = start_app(False)
    stop_and_check(p, port, signal.SIGTERM)
