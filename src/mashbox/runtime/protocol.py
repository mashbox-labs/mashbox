"""Front <-> worker protocol, version 1: how the HTTP front talks to the process that holds the engine.

The front (runtime/front.py) answers HTTP, tokenizes, detokenizes and matches stop strings; the worker (runtime/worker.py)
holds the model on the chip and runs the batching loop. They talk over one Unix stream socket in a directory of mode
0700 (the socket itself 0600); before it reads a frame the worker checks that the peer runs as its own user. If the
worker dies (a device fault, out of memory, a hung call, kill -9), the front stays up: it ends every stream that already
sent output with an error event, sends requests whose client has received nothing once more to the restarted worker, and answers 503
(Anthropic route 529) while no worker is ready.

A frame is a 4-byte big-endian length and one msgpack map (the subset below, written with the standard library); every map has "t", the message type:
  HELLO        front -> worker   proto [1], front_id
  HELLO_ACK    worker -> front   proto, worker_id, models, limits {R, vocab, ctx, max_frame, out_buffer_tokens}, state
  SUBMIT       front -> worker   req (16 bytes), model, ids (int32 little-endian bytes), params {max_tokens, samp,
                                 stop_ids, mark, scope, variant, rid (the response id its client
                                 receives: the MASHBOX_TRACE admit line names it; optional)}
  ACCEPTED     worker -> front   req (the worker took it: its checks passed and it waits for a free row)
  REJECTED     worker -> front   req, code, msg (nothing of the request stays in the worker)
  TOKENS       worker -> front   req, seq (0, 1, 2, ... per request), ids
  CANCEL       front -> worker   req (client gone, or the front found a stop string: the row ends at the next step)
  FINISH       worker -> front   req, reason ("stop", "length", "cancel"), usage {prompt, cached, restored, output},
                                 calls, sampled
  ERROR        worker -> front   req or None, code, fatal, msg (fatal: the worker exits next and the front restarts it)
  HEALTH       both ways         state ("loading", "ready"), live, sent every second by the worker
Exactly one terminal frame (FINISH or ERROR) follows each ACCEPTED, and nothing for that request after it.
"""
import os
import socket
import struct
import tempfile
import threading

PROTO = 1
MAX_FRAME = 4 << 20          # bytes; a 262,144-token prompt as int32 ids is 1 MiB
OUT_BUFFER_TOKENS = 1024
STALL_S = 30                 # a client that reads nothing for this long is ended, as a stalled write
READ_S = 30                  # a connection may wait this long for a request's first byte, and a request then has this
                             # long to arrive whole (line, headers, body); a half-sent request is closed after it
WATCHDOG_MIN_S = 30          # a device call slower than max(this, WATCHDOG_X x its usual time) counts as hung
WATCHDOG_X = 10
HEARTBEAT_S = 1.0
START_TRIES = 2              # worker lives in a row that end before ready, then the server exits non-zero
START_SLOW_X = 2             # a restart slower than this x the first start (and START_FLOOR_S) exits non-zero
START_FLOOR_S = 60

_LEN = struct.Struct(">I")


def packb(o):
    """msgpack bytes for None, bool, int (64-bit), float, str, bytes, list / tuple and dict."""
    out = []

    def w(o):
        if o is None:
            out.append(b"\xc0")
        elif o is True or o is False:
            out.append(b"\xc3" if o else b"\xc2")
        elif isinstance(o, int):
            if 0 <= o < 128:
                out.append(struct.pack("B", o))
            elif -32 <= o < 0:
                out.append(struct.pack("b", o))
            elif 0 <= o < 1 << 64:
                out.append(next(struct.pack(">B" + f, t, o) for lim, t, f in ((1 << 8, 0xcc, "B"), (1 << 16, 0xcd, "H"),
                           (1 << 32, 0xce, "I"), (1 << 64, 0xcf, "Q")) if o < lim))
            elif -(1 << 63) <= o < 0:
                out.append(next(struct.pack(">B" + f, t, o) for lim, t, f in ((1 << 7, 0xd0, "b"), (1 << 15, 0xd1, "h"),
                           (1 << 31, 0xd2, "i"), (1 << 63, 0xd3, "q")) if o >= -lim))
            else:
                raise FrameError("bad_frame", "integer out of 64-bit range")
        elif isinstance(o, float):
            out.append(struct.pack(">Bd", 0xcb, o))
        elif isinstance(o, str):
            b = o.encode()
            n = len(b)
            out.append((struct.pack("B", 0xa0 | n) if n < 32 else struct.pack(">BB", 0xd9, n) if n < 1 << 8
                        else struct.pack(">BH", 0xda, n) if n < 1 << 16 else struct.pack(">BI", 0xdb, n)) + b)
        elif isinstance(o, (bytes, bytearray)):
            n = len(o)
            out.append((struct.pack(">BB", 0xc4, n) if n < 1 << 8 else struct.pack(">BH", 0xc5, n) if n < 1 << 16
                        else struct.pack(">BI", 0xc6, n)) + bytes(o))
        elif isinstance(o, (list, tuple)):
            n = len(o)
            out.append(struct.pack("B", 0x90 | n) if n < 16 else struct.pack(">BH", 0xdc, n) if n < 1 << 16
                       else struct.pack(">BI", 0xdd, n))
            for x in o:
                w(x)
        elif isinstance(o, dict):
            n = len(o)
            out.append(struct.pack("B", 0x80 | n) if n < 16 else struct.pack(">BH", 0xde, n) if n < 1 << 16
                       else struct.pack(">BI", 0xdf, n))
            for k, v in o.items():
                w(k)
                w(v)
        else:
            raise FrameError("bad_frame", f"cannot send a {type(o).__name__}")
    w(o)
    return b"".join(out)


_FIXED = {0xcc: ">B", 0xcd: ">H", 0xce: ">I", 0xcf: ">Q", 0xd0: ">b", 0xd1: ">h", 0xd2: ">i", 0xd3: ">q", 0xca: ">f",
          0xcb: ">d"}


def unpackb(b):
    """The value of one msgpack document (the subset packb writes, plus float32); raises ValueError on anything else."""
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(b):
            raise ValueError("truncated")
        pos += n
        return b[pos - n:pos]

    def num(f):
        return struct.unpack(f, take(struct.calcsize(f)))[0]

    def r():
        t = take(1)[0]
        if t < 0x80:
            return t
        if t >= 0xe0:
            return t - 0x100
        if 0x80 <= t <= 0x8f or t in (0xde, 0xdf):
            n = t & 0x0f if t <= 0x8f else num(">H" if t == 0xde else ">I")
            d = {}
            for _ in range(n):
                k = r()
                if isinstance(k, (dict, list)):
                    raise ValueError("unhashable key")
                d[k] = r()
            return d
        if 0x90 <= t <= 0x9f or t in (0xdc, 0xdd):
            n = t & 0x0f if t <= 0x9f else num(">H" if t == 0xdc else ">I")
            return [r() for _ in range(n)]
        if 0xa0 <= t <= 0xbf or t in (0xd9, 0xda, 0xdb):
            n = t & 0x1f if t <= 0xbf else num({0xd9: ">B", 0xda: ">H", 0xdb: ">I"}[t])
            return take(n).decode()
        if t in (0xc4, 0xc5, 0xc6):
            return bytes(take(num({0xc4: ">B", 0xc5: ">H", 0xc6: ">I"}[t])))
        if t in _FIXED:
            return num(_FIXED[t])
        if t in (0xc0, 0xc2, 0xc3):
            return {0xc0: None, 0xc2: False, 0xc3: True}[t]
        raise ValueError(f"msgpack type 0x{t:02x} is not used by this protocol")
    v = r()
    if pos != len(b):
        raise ValueError("trailing bytes")
    return v


class FrameError(Exception):
    """A frame the reader refuses (too large, not a map, no type). The connection is closed after it."""

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def pack(msg):
    body = packb(msg)
    if len(body) > MAX_FRAME:
        raise FrameError("frame_too_large", f"frame of {len(body)} bytes is over the {MAX_FRAME}-byte limit")
    return _LEN.pack(len(body)) + body


class Conn:
    """One end of the socket: send() from any thread (one lock), recv() from one reader thread."""

    def __init__(self, sock):
        self.sock, self.lock, self.buf = sock, threading.Lock(), b""

    def send(self, msg):
        data = pack(msg)
        with self.lock:
            self.sock.sendall(data)

    def _read(self, n):
        while len(self.buf) < n:
            part = self.sock.recv(max(65536, n - len(self.buf)))
            if not part:
                return None
            self.buf += part
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        """The next message, or None at end of stream. Raises FrameError for a frame over the limit or not a typed map."""
        head = self._read(4)
        if head is None:
            return None
        n = _LEN.unpack(head)[0]
        if n > MAX_FRAME:
            raise FrameError("frame_too_large", f"frame of {n} bytes is over the {MAX_FRAME}-byte limit")
        body = self._read(n)
        if body is None:
            return None
        try:
            msg = unpackb(body)
        except Exception as e:
            raise FrameError("bad_frame", f"frame is not msgpack: {type(e).__name__}")
        if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
            raise FrameError("bad_frame", "frame is not a map with a type")
        return msg

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def ids_bytes(ids):
    return struct.pack(f"<{len(ids)}i", *ids)


def ids_list(b):
    if len(b) % 4:
        raise FrameError("bad_frame", "ids must be int32 little-endian")
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def raise_open_files():
    """Raise this process's soft limit on open files to its hard limit. A server holds one descriptor per connection, and
    the usual soft limit (1,024 on Ubuntu) would refuse connections past about a thousand (accept fails with EMFILE)
    although the hard limit allows far more. -> dict(before, soft, hard) for the server's log line; None where the
    platform has no such limit (Windows). A hard limit of "unlimited" is not a value the kernel takes for the soft one,
    so the largest of a few usual ceilings that it accepts is used then."""
    try:
        import resource
    except ImportError:
        return None
    before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    inf = resource.RLIM_INFINITY
    for want in ([hard] if hard != inf else [1 << 20, 1 << 16, 10240]):
        if before != inf and want <= before:
            break
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            break
        except (ValueError, OSError):
            continue
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    return dict(before=before, soft=soft, hard=hard)


def socket_dir():
    """A new private directory (mode 0700) for one worker's socket."""
    return tempfile.mkdtemp(prefix="mbx-engine-")


def listen(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        s.bind(path)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    s.listen(1)
    return s


def peer_uid(sock):
    """The user id of the process on the other end of a Unix socket (SO_PEERCRED)."""
    cred = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", cred)[1]
