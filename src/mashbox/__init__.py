"""mashbox: open models that run faster and cheaper, from MASHBOX Labs."""
import json
import os

# The release version is the "version" field of the packaged models.json; pyproject.toml reads it from here.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json"), encoding="utf-8") as _f:
    __version__ = json.load(_f)["version"]
del _f

# Strict f32 semantics in XLA (no extra-precision shortcuts): the measured outputs of these ports assume it.
if "xla_allow_excess_precision" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false").strip()
