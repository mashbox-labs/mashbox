"""mashbox by MASHBOX Labs: open models that run faster and cheaper."""
import os

__version__ = "0.1.1"

# Strict f32 semantics in XLA (no extra-precision shortcuts): the measured outputs of these ports assume it.
if "xla_allow_excess_precision" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false").strip()
