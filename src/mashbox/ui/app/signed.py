"""Signature check for the model catalog the app reads at run time (minisign format, Ed25519, stdlib only).

The app accepts the public models.json only when its signature, made with the MASHBOX Labs catalog key, matches the exact
bytes it read. A signature file is the minisign text format:

    untrusted comment: <anything>
    base64( "ED" | key id (8 bytes) | Ed25519 signature of BLAKE2b-512(file) (64 bytes) )   ("Ed": of the file itself)
    trusted comment: <text, covered by the second signature>
    base64( Ed25519 signature of (first signature | trusted comment) )

The Ed25519 verify below is the reference algorithm of RFC 8032 section 6 (pure Python, a few ms per check).
"""
from __future__ import annotations

import base64
import hashlib

# The MASHBOX Labs catalog key (minisign public key). The private half never leaves our signing machines.
CATALOG_KEY = "RWRTHTeJTA9viNg4q6VuA82X2SIqkFcpgg25M/SJVvE1Xre6wYtkWuo4"  # gitleaks:allow (a public key, meant to be published)

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _add(a, b):
    A = (a[1] - a[0]) * (b[1] - b[0]) % _P
    B = (a[1] + a[0]) * (b[1] + b[0]) % _P
    C = 2 * a[3] * b[3] * _D % _P
    D = 2 * a[2] * b[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s, pt):
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = _add(q, pt)
        pt = _add(pt, pt)
        s >>= 1
    return q


def _same(a, b):
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _x(y, sign):
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


def _point(b):
    if len(b) != 32:
        return None
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


_GY = 4 * pow(5, _P - 2, _P) % _P
_G = (_x(_GY, 0), _GY, 1, _x(_GY, 0) * _GY % _P)


def _small(pt):
    """A point of small order (8 * P is the neutral element): never a real public key."""
    q = _mul(8, pt)
    return q[0] % _P == 0 and (q[1] - q[2]) % _P == 0


def ed25519_verify(public: bytes, msg: bytes, sig: bytes) -> bool:
    """RFC 8032 Ed25519 verification: True only for a valid signature of msg by public (small-order keys refused)."""
    if len(public) != 32 or len(sig) != 64:
        return False
    a, r = _point(public), _point(sig[:32])
    if a is None or r is None or _small(a):
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(sig[:32] + public + msg).digest(), "little") % _L
    return _same(_mul(s, _G), _add(r, _mul(h, a)))


def _b64(text):
    try:
        return base64.b64decode(text.strip(), validate=True)
    except ValueError:
        return b""


def public_key(text: str) -> tuple[bytes, bytes] | None:
    """(key id, Ed25519 key) from a minisign public key: the base64 line, optionally after its comment line."""
    lines = [x for x in text.strip().splitlines() if x.strip() and not x.startswith("untrusted comment:")]
    raw = _b64(lines[-1]) if lines else b""
    return (raw[2:10], raw[10:]) if len(raw) == 42 and raw[:2] == b"Ed" else None


def verify(data: bytes, sig_text: str, key: str | None = None) -> str | None:
    """The trusted comment when sig_text is a valid signature of data by key, else None. A signature file that is
    itself base64 (as Tauri's signer writes it) is read too. key: a minisign public key, default CATALOG_KEY."""
    pk = public_key(key or CATALOG_KEY)
    if pk is None or not sig_text:
        return None
    if "trusted comment:" not in sig_text:
        sig_text = _b64(sig_text).decode("utf-8", "replace")
    lines = [x.rstrip("\r") for x in sig_text.splitlines() if x.strip()]
    if len(lines) < 4 or not lines[0].startswith("untrusted comment:") or not lines[2].startswith("trusted comment: "):
        return None
    raw, glob = _b64(lines[1]), _b64(lines[3])
    if len(raw) != 74 or len(glob) != 64 or raw[2:10] != pk[0] or raw[:2] not in (b"ED", b"Ed"):
        return None
    msg = hashlib.blake2b(data, digest_size=64).digest() if raw[:2] == b"ED" else data
    trusted = lines[2][len("trusted comment: "):]
    if not ed25519_verify(pk[1], msg, raw[10:]) or not ed25519_verify(pk[1], raw[10:] + trusted.encode(), glob):
        return None
    return trusted


def fields(trusted: str | None) -> dict:
    """key=value words of a trusted comment (ours: mashbox-catalog utc=... sha256=... source=...)."""
    return dict(w.split("=", 1) for w in (trusted or "").split() if "=" in w)
