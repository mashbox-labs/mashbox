"""tools/gguf_check.py's independent GGUF header parser (struct reads from the format spec, no gguf-py) against gguf-py's
tensor table on the fixtures: same names, ggml types, shapes and absolute data offsets."""
import importlib.util
import os

import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
pytest.importorskip("jax")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CHECK = _load("gguf_check", os.path.join(ROOT, "tools", "gguf_check.py"))
FXT = _load("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))


@pytest.mark.parametrize("name", ["qwen35-kquant", "qwen35-pyquant", "qwen35-mtp"])
def test_independent_header_parse_equals_gguf_py(name, tmp_path):
    p = os.path.join(FX, f"{name}.gguf") if name == "qwen35-kquant" else FXT.build(name, str(tmp_path))
    ver, align, start, ind = CHECK.parse_header(p)
    r = gguf.GGUFReader(p)
    want = {t.name: (int(t.tensor_type), tuple(reversed([int(x) for x in t.shape])), int(t.data_offset)) for t in r.tensors}
    assert ver == 3 and align == 32 and ind == want
