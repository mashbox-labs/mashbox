"""Front <-> worker protocol v1 (runtime/protocol.py, runtime/worker.py), driven frame by frame against a real worker
process with the scripted engine (api_stub_engine.py):
- frames: length + msgpack map round trip; a frame over 4 MiB, a non-map or an untyped map is refused, never a crash;
- the worker's socket is private (directory 0700, socket 0600) and it reports its peer's uid;
- HELLO with no common version gets ERROR proto_mismatch (fatal) and the worker exits cleanly;
- per request: ACCEPTED, TOKENS with seq 0, 1, 2, ... and no gaps, exactly one terminal frame, nothing after it;
- REJECTED leaves nothing in the worker; CANCEL ends a live row at its next step;
- the worker exits when the front goes away.
"""
import os
import socket
import stat
import subprocess
import sys
import time

import pytest

from mashbox.runtime import protocol as P

HERE = os.path.dirname(os.path.abspath(__file__))


def test_frames_round_trip_and_refusals():
    a, b = socket.socketpair()
    ca, cb = P.Conn(a), P.Conn(b)
    msg = dict(t="SUBMIT", req=b"\x01" * 16, ids=P.ids_bytes([1, 2, 70000]), params=dict(max_tokens=5, samp=None))
    ca.send(msg)
    got = cb.recv()
    assert got == msg and P.ids_list(got["ids"]) == [1, 2, 70000]
    with pytest.raises(P.FrameError) as e:
        P.pack(dict(t="SUBMIT", ids=b"\0" * (P.MAX_FRAME + 1)))
    assert e.value.code == "frame_too_large"
    a.sendall((P.MAX_FRAME + 1).to_bytes(4, "big"))          # a peer that announces too much is refused before reading
    with pytest.raises(P.FrameError) as e:
        cb.recv()
    assert e.value.code == "frame_too_large"
    a2, b2 = socket.socketpair()
    for bad in (P.packb([1, 2]), P.packb(dict(no_type=1)), b"\xc1"):
        a2.sendall(len(bad).to_bytes(4, "big") + bad)
        with pytest.raises(P.FrameError) as e:
            P.Conn(b2).recv()
        assert e.value.code == "bad_frame"
    a.close()
    assert P.Conn(b).recv() is None                           # end of stream is None, not an error


def test_wire_format_is_msgpack():
    """The standard-library encoder writes what the msgpack library reads, and reads what it writes."""
    msgpack = pytest.importorskip("msgpack")
    for o in (None, True, False, 0, 127, 128, -1, -32, -33, 255, 256, 65535, 65536, (1 << 32) + 5, (1 << 64) - 1,
              -(1 << 63), 1.5, -0.0, "", "x" * 31, "y" * 32, "z" * 300, "é" * 70000, b"", b"\0" * 300, b"\1" * 70000,
              [], list(range(20)), list(range(70000)), {}, {f"k{i}": i for i in range(20)},
              dict(t="SUBMIT", req=bytes(16), ids=P.ids_bytes(list(range(1000))), params=dict(max_tokens=5, samp=None,
                   stop_ids=[1, 2], scope=None, mark=[3, 4]))):
        assert msgpack.unpackb(P.packb(o), raw=False) == o, repr(o)[:60]
        assert P.unpackb(msgpack.packb(o, use_bin_type=True)) == o, repr(o)[:60]
        assert P.unpackb(P.packb(o)) == o


def spawn(tmp_path, S=2, step_s=0.0, outs=None):
    d = P.socket_dir()
    path = os.path.join(d, "worker.sock")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.abspath("src"), os.environ.get("PYTHONPATH", "")]),
               MASHBOX_FAULT="", JAX_PLATFORMS="cpu")
    import json
    kw = dict(S=S, step_s=step_s, outs=outs or [[5, 6, 7, 8]], log=str(tmp_path / "engine.jsonl"))
    proc = subprocess.Popen([sys.executable, "-m", "mashbox.runtime.worker", "--socket", path, "--factory",
                             "api_stub_engine:make", "--kwargs", json.dumps(kw)], env=env)
    t = time.time()
    while not os.path.exists(path):
        assert proc.poll() is None and time.time() - t < 20
        time.sleep(0.01)
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700 and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    assert P.peer_uid(s) == os.getuid()
    return proc, P.Conn(s)


def hello(c):
    c.send(dict(t="HELLO", proto=[1], front_id="test"))
    ack = c.recv()
    assert ack["t"] == "HELLO_ACK" and ack["proto"] == 1
    while True:
        m = c.recv()
        if m["t"] == "HEALTH" and m["state"] == "ready":
            return m


def frames_until(c, done, s=10):
    out, t = [], time.time()
    while not done(out) and time.time() - t < s:
        m = c.recv()
        if m is None:
            break
        if m["t"] != "HEALTH":
            out.append(m)
    return out


def test_wrong_version_is_refused_and_worker_exits(tmp_path):
    proc, c = spawn(tmp_path)
    c.send(dict(t="HELLO", proto=[99], front_id="test"))
    m = c.recv()
    assert m["t"] == "ERROR" and m["code"] == "proto_mismatch" and m["fatal"] is True
    assert proc.wait(10) == 2


def test_sequences_terminal_frames_reject_and_cancel(tmp_path):
    proc, c = spawn(tmp_path, S=2, step_s=0.01, outs=[[5, 6, 7, 8, 9, 10, 11, 12]])
    ready = hello(c)
    assert ready["limits"]["R"] == 2 and ready["limits"]["vocab"] == 1000 and ready["limits"]["ctx"] == 4096
    reqs = [bytes([i]) * 16 for i in range(3)]
    for r in reqs[:2]:
        c.send(dict(t="SUBMIT", req=r, model="m", ids=P.ids_bytes([1, 2, 3]), params=dict(max_tokens=6, stop_ids=[])))
    c.send(dict(t="SUBMIT", req=reqs[2], model="m", ids=P.ids_bytes([1, 5000]), params=dict(max_tokens=6)))
    out = frames_until(c, lambda o: sum(m["t"] in ("FINISH", "ERROR", "REJECTED") for m in o) >= 3)
    rej = [m for m in out if m["req"] == reqs[2]]
    assert [m["t"] for m in rej] == ["REJECTED"] and rej[0]["code"] == "bad_params"
    for r in reqs[:2]:
        mine = [m for m in out if m["req"] == r]
        assert mine[0]["t"] == "ACCEPTED" and mine[-1]["t"] == "FINISH"
        assert sum(m["t"] in ("FINISH", "ERROR") for m in mine) == 1
        toks = [m for m in mine if m["t"] == "TOKENS"]
        assert [m["seq"] for m in toks] == list(range(len(toks)))
        assert sum((m["ids"] for m in toks), []) == [5, 6, 7, 8, 9, 10]
        assert mine[-1]["reason"] == "length" and mine[-1]["usage"]["output"] == 6 and mine[-1]["usage"]["prompt"] == 3
    r = reqs[0]
    c.send(dict(t="SUBMIT", req=r, model="m", ids=P.ids_bytes([1]), params=dict(max_tokens=4000)))
    c.send(dict(t="CANCEL", req=r))
    out = frames_until(c, lambda o: any(m["t"] == "FINISH" for m in o))
    fin = [m for m in out if m["t"] == "FINISH"]
    assert fin and fin[0]["reason"] == "cancel" and fin[0]["usage"]["output"] < 8
    c.close()                                                   # the front goes away: the worker ends
    assert proc.wait(10) == 0
