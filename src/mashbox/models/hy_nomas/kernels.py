"""Load this model's compiled TPU kernel bundle (one wheel, downloaded from the model's Hugging Face repo).

The bundle holds jax.export files, one per kernel and shape, plus kernels.json (op name, shape key, sha256, input and
output shapes). Nothing here contains kernel logic: it checks hashes and versions, then deserializes.
If anything does not match (file hash, jax version, chip, input shapes), ``get`` returns None and the caller runs the
plain JAX path instead. It never raises on a mismatch and never runs an unverified file.
"""
import hashlib
import io
import json
import logging
import zipfile

log = logging.getLogger("mashbox.hy_nomas.kernels")
PKG = "mashbox_hy_nomas_kernels"


def sha256(b):
    return hashlib.sha256(b).hexdigest()


class Bundle:
    def __init__(self, wheel_bytes, expect_sha256):
        if sha256(wheel_bytes) != expect_sha256:
            raise ValueError("kernel bundle sha256 does not match models.json")
        self._z = zipfile.ZipFile(io.BytesIO(wheel_bytes))
        self.index = json.loads(self._z.read(PKG + "/kernels.json"))
        self._rows = {(f["op"], f["shape_key"]): f for f in self.index["files"]}
        self._cache = {}

    def usable(self):
        import jax
        if jax.__version__ != self.index["jax"]:
            log.info("kernel bundle is for jax %s, running %s: plain JAX path", self.index["jax"], jax.__version__)
            return False
        kind = jax.devices()[0].device_kind
        if not kind.startswith("TPU"):
            log.info("no TPU (%s): plain JAX path", kind)
            return False
        return True

    def get(self, op, shape_key):
        """Callable for (op, shape_key), or None when the bundle has no such file."""
        key = (op, shape_key)
        if key in self._cache:
            return self._cache[key]
        row = self._rows.get(key)
        fn = None
        if row is not None:
            blob = self._z.read(PKG + "/" + row["file"])
            if sha256(blob) != row["sha256"]:
                raise ValueError("kernel file %s does not match its sha256" % row["file"])
            import jax
            exp = jax.export.deserialize(blob)

            def fn(*args, _exp=exp, _row=row):
                args = [a.astype(dt) if str(a.dtype) != dt else a for a, (_, dt) in zip(args, _row["in_avals"])]
                for a, (shape, _) in zip(args, _row["in_avals"]):
                    if list(a.shape) != shape:
                        raise ValueError("%s %s: input shape %s, compiled for %s" % (op, shape_key, list(a.shape), shape))
                return _exp.call(*args)
        self._cache[key] = fn
        return fn


def open_bundle(path, expect_sha256):
    with open(path, "rb") as f:
        return Bundle(f.read(), expect_sha256)
