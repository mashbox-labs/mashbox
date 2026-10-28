"""Name of the chip this process runs on, read from JAX and the VM's metadata (never a fixed string)."""
import urllib.request

KINDS = {"TPU v6 lite": "v6e", "TPU v5 lite": "v5e", "TPU v5": "v5p", "TPU v4": "v4"}


def zone(timeout=0.5):
    """Google Cloud zone of this VM, or None off Google Cloud."""
    req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/zone",
                                 headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode().rsplit("/", 1)[-1] or None
    except Exception:
        return None


def chip_label(devices=None, zone_name=None):
    """e.g. 'TPU v6e-4 (4 chips), asia-northeast1-b'."""
    if devices is None:
        import jax
        devices = jax.devices()
    kind = devices[0].device_kind
    n = len(devices)
    short = KINDS.get(kind)
    name = ("TPU %s-%d (%d chip%s)" % (short, n, n, "" if n == 1 else "s")) if short else ("%s x%d" % (kind, n))
    z = zone_name if zone_name is not None else zone()
    return name + (", " + z if z else "")
