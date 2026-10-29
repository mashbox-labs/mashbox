"""The MASHBOX app's signed model catalog (src/mashbox/ui/app/signed.py + registry()): the default catalog is used only
with a valid MASHBOX Labs signature that is not older than the saved one; a bad one falls back to the saved copy or the
package's; a catalog address the user typed may be unsigned. Plus the Hugging Face add-by-id seam (501 until GGUF)."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from mashbox.ui import app, shell
from mashbox.ui.app import signed

# A test key (seed = sha256 of a fixed phrase; never a real key) and signatures made with the `cryptography` package.
PUB = "RWQBAgMEBQYHCDwzhjaBTqAr9lOsSH3SeuXYD1vwgIC+I4zD7XgohxnY"
PAYLOAD = b'{"models": [{"id": "m1", "kind": "llm"}]}'
SIG = ("untrusted comment: test\nRUQBAgMEBQYHCErUq9fOEc5rgAFad+zBJsXAaJd+M479mWw2iRiW84Cg72jFizYz9Nbvez0A3oFW8QE7D3qBq4TW0mYNtXD4pwE=\n"
       "trusted comment: mashbox-catalog utc=2026-10-28T13:00:00Z sha256=e49164ac22d7dda8e6d431fff0744efde1e1c7a39ce2a6defb1c0605ac110455 source=test version=0\n"
       "HJjqme2+OplWL5NerXYiLSP4FFAmRyNQCEtpZV7urMqZbWZF62DrtlL3L0TXlxDziBJWgDxIjhEPG0IyBtvkAw==\n")
OLDER = ("untrusted comment: test\nRUQBAgMEBQYHCErUq9fOEc5rgAFad+zBJsXAaJd+M479mWw2iRiW84Cg72jFizYz9Nbvez0A3oFW8QE7D3qBq4TW0mYNtXD4pwE=\n"
         "trusted comment: mashbox-catalog utc=2026-10-28T12:00:00Z sha256=e49164ac22d7dda8e6d431fff0744efde1e1c7a39ce2a6defb1c0605ac110455 source=test version=0\n"
         "DUtSrfP8+6QerZlxA0vtP7+REgmsoOF7DPw4KuXiBM7jnucy8a60+k0QeaAiPRQ0NXbuYAwtaPhuRiM4eWydBQ==\n")


def test_ed25519_rfc8032_vector_1():
    pk = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    sig = bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
    assert signed.ed25519_verify(pk, b"", sig)
    assert not signed.ed25519_verify(pk, b"\x00", sig)
    bad = bytearray(sig)
    bad[40] ^= 1
    assert not signed.ed25519_verify(pk, b"", bytes(bad))
    s_plus_l = int.from_bytes(sig[32:], "little") + signed._L        # same point, non-canonical s: refused
    assert not signed.ed25519_verify(pk, b"", sig[:32] + s_plus_l.to_bytes(32, "little"))
    ident = (1).to_bytes(32, "little")                                  # the neutral point as a key (small order): refused
    assert not signed.ed25519_verify(ident, b"", (1).to_bytes(32, "little") + bytes(32))   # R = neutral, s = 0
    assert signed._small(signed._point(ident)) and not signed._small(signed._point(pk))


def test_minisign_verify_and_tampering():
    assert signed.fields(signed.verify(PAYLOAD, SIG, PUB))["utc"] == "2026-10-28T13:00:00Z"
    assert signed.verify(PAYLOAD + b" ", SIG, PUB) is None                       # file changed
    assert signed.verify(PAYLOAD, SIG.replace("version=0", "version=9"), PUB) is None   # trusted comment changed
    assert signed.verify(PAYLOAD, SIG, signed.CATALOG_KEY) is None              # another key (key id differs)
    assert signed.verify(PAYLOAD, "", PUB) is None and signed.verify(PAYLOAD, "junk", PUB) is None
    import base64
    assert signed.verify(PAYLOAD, base64.b64encode(SIG.encode()).decode(), PUB)  # Tauri's signer wraps it in base64
    assert signed.public_key(signed.CATALOG_KEY) is not None                     # the shipped key parses


@pytest.fixture
def net(tmp_path, monkeypatch):
    """registry() against a fake network: {url: bytes or an exception}."""
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    monkeypatch.setattr(signed, "CATALOG_KEY", PUB)
    web: dict = {}

    class R:
        def __init__(self, b):
            self.b = b

        def read(self):
            return self.b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(url, *a, **k):
        v = web.get(getattr(url, "full_url", url), urllib.error.HTTPError(url, 404, "Not Found", {}, None))
        if isinstance(v, Exception):
            raise v
        return R(v)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    app._cat.clear()
    yield web, tmp_path
    app._cat.clear()


def test_default_catalog_needs_our_signature(net):
    web, tmp = net
    web[app.CATALOG_URL] = PAYLOAD
    reg, src = app.registry(refresh=True)                     # no signature published: not used
    assert src["from"] == "package" and "signature" in src["error"]
    web[app.CATALOG_SIG_URL] = SIG.encode()
    reg, src = app.registry(refresh=True)
    assert src["from"] == "url" and src["signed"] and [m["id"] for m in reg["models"]] == ["m1"]
    web[app.CATALOG_URL] = PAYLOAD.replace(b"m1", b"m2")       # changed on the way: the saved signed copy stays
    reg, src = app.registry(refresh=True)
    assert src["from"] == "cache" and src["signed"] and "does not match" in src["error"] and reg["models"][0]["id"] == "m1"
    web[app.CATALOG_URL], web[app.CATALOG_SIG_URL] = PAYLOAD, OLDER.encode()   # a validly signed but OLDER catalog
    reg, src = app.registry(refresh=True)
    assert src["from"] == "cache" and "older" in src["error"]
    cache = json.loads((tmp / "catalog-cache.json").read_text())
    assert cache["signed"] and cache["utc"] == "2026-10-28T13:00:00Z"
    (tmp / "catalog-cache.json").write_text(json.dumps(dict(cache, signed=False)))   # an unsigned saved copy is not used
    web[app.CATALOG_URL] = urllib.error.URLError("offline")
    reg, src = app.registry(refresh=True)
    assert src["from"] == "package" and reg["models"] == shell.registry()["models"]


def test_own_catalog_address_may_be_unsigned_but_never_badly_signed(net):
    web, tmp = net
    own = "https://example.test/my/models.json"
    (tmp / "state.json").write_text(json.dumps({"catalog_url": own}))
    web[own] = PAYLOAD
    reg, src = app.registry(refresh=True)
    assert src["from"] == "url" and src["signed"] is False
    web[own + ".minisig"] = SIG.encode()
    reg, src = app.registry(refresh=True)
    assert src["signed"] is True
    web[own] = PAYLOAD.replace(b"m1", b"m3")
    reg, src = app.registry(refresh=True)
    assert src["from"] == "cache" and "does not match" in src["error"]


def test_add_hf_model_seam(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_APP_DATA", str(tmp_path))
    app._cat.clear()
    srv, wsrv = app.start("127.0.0.1", 0)
    for s in (srv, wsrv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        import re
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            tok = re.search(rb'<meta name="mbx-token" content="([^"]+)"', r.read()).group(1).decode()

        def post(body):
            req = urllib.request.Request(url + "/api/models/add", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-Mashbox-Token": tok})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        assert post({"hf": "unsloth/Qwen3.8-27B-GGUF:Q8_0"}) == 501
        assert post({"hf": "../etc/passwd"}) == 400 and post({"hf": "a/b c"}) == 400 and post({}) == 400
        req = urllib.request.Request(url + "/api/models/add", data=b'{"hf": "a/b"}', headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:     # no token: the guard refuses it as every /api call
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 401
    finally:
        srv.shutdown()
        wsrv.shutdown()


def test_a_signature_without_its_signing_time_is_refused(net, monkeypatch):
    """The age check needs the utc= field of the trusted comment, so a valid signature without one is not used."""
    web, tmp = net
    web[app.CATALOG_URL], web[app.CATALOG_SIG_URL] = PAYLOAD, SIG.encode()
    monkeypatch.setattr(app.signed, "verify", lambda raw, sig, *a: "mashbox-catalog sha256=x source=test")
    reg, src = app.registry(refresh=True)
    assert src["from"] == "package" and "signing time" in src["error"]
    monkeypatch.setattr(app.signed, "verify", lambda raw, sig, *a: "mashbox-catalog utc=yesterday source=test")
    reg, src = app.registry(refresh=True)
    assert src["from"] == "package" and "signing time" in src["error"]
