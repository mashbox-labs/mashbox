"""Who owns a prefix-cache entry: the tenant id and the cache namespace.

Tenant id: HMAC-SHA256(server cache secret, tenant name or API key), versioned ("v1." + 32 hex digits). A tenant is an
owner group the operator names in the key file (a line "<key> <tenant>"); a key without a name is its own tenant. The id
never depends on where a key sits in the key file, and the key itself is never kept in it. The secret is made at start
(32 random bytes; the cache lives in this process's memory, so a restart empties it anyway) and can be rotated: every id
changes, so nothing cached before the rotation is ever served again. Without API keys (a loopback server) there is one
local owner, None, and everything shares.

Namespace: sha256 over canonical JSON of what makes a cached state valid: the deployment, the model, its tokenizer and
chat template, and the engine's checkpoint policy. A state cached under one namespace is never served under another.
"""
import hashlib
import hmac
import json
import secrets
import threading

VERSION = 1


def canon(obj):
    """Canonical JSON: keys sorted, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_json(obj):
    return hashlib.sha256(canon(obj)).hexdigest()


def file_sha256(path):
    """sha256 of a file's bytes, or None when it does not exist."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def namespace(deployment=None, model=None, tokenizer=None, chat_template=None, checkpoint_policy=None):
    return "ns1." + sha256_json(dict(v=1, deployment=deployment, model=model, tokenizer=tokenizer,
                                     chat_template=chat_template, checkpoint_policy=checkpoint_policy))


class Tenancy:
    """API keys -> tenant ids. tenants: {key: tenant name} for keys the operator grouped; other keys are their own tenant."""

    def __init__(self, tenants=None, secret=None):
        self._lock = threading.Lock()
        self.tenants = {str(k): str(t) for k, t in (tenants or {}).items()}
        self._secret = bytes(secret) if secret is not None else secrets.token_bytes(32)
        self.generation = 1

    def rotate(self, secret=None):
        """A new secret: every tenant id changes, so no entry cached before this call matches again."""
        with self._lock:
            self._secret = bytes(secret) if secret is not None else secrets.token_bytes(32)
            self.generation += 1

    def tenant_id(self, key):
        """The tenant id of an accepted API key (None for None: the one local owner of a server without keys)."""
        if key is None:
            return None
        key = str(key)
        name = self.tenants.get(key)
        msg = (b"tenant\x00" + name.encode()) if name is not None else (b"key\x00" + key.encode())
        with self._lock:
            d = hmac.new(self._secret, msg, hashlib.sha256).hexdigest()
        return f"v{VERSION}.{d[:32]}"


_current = Tenancy()


def current():
    return _current


def configure(tenants=None, secret=None):
    """Set the process's tenancy (cli.py at start; tests). -> the new Tenancy."""
    global _current
    _current = Tenancy(tenants, secret)
    return _current


def parse_key_lines(lines):
    """Key file lines -> (keys, {key: tenant}). A line is "<key>" or "<key> <tenant>"; blank lines are skipped."""
    keys, tenants = [], {}
    for x in lines:
        parts = x.split()
        if not parts:
            continue
        keys.append(parts[0])
        if len(parts) > 1:
            tenants[parts[0]] = " ".join(parts[1:])
    return keys, tenants
