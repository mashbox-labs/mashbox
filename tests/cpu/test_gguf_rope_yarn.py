"""YaRN rope scaling in the GGUF dense architectures (gguf/archs/rope.py): our logits vs llama.cpp's CPU logits on tiny
random fixtures (tests/fixtures/gguf/<name>.json says how each was made). No chip, no network, no secrets.

  qwen3-yarn-f32  qwen3 (rope on halves, q/k norms), factor 4 on an original context of 64
  llama-yarn-f32  llama (rope on adjacent pairs, rope_freqs factors), factor 8 on an original context of 128
Both contexts are short enough that the correction range falls inside the rotary dims, so the fast, mixed and
interpolated pairs are all used. Bound NMSE <= 1e-4 (llama.cpp's cross-backend bar), prefill and decode.
"""
import importlib.util
import os

import numpy as np
import pytest

import gguf  # noqa: F401,E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
pytest.importorskip("jax")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("test_gguf_dense_helpers", os.path.join(ROOT, "tests", "cpu", "test_gguf_dense.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


@pytest.mark.parametrize("name", ["qwen3-yarn-f32", "llama-yarn-f32"])
@pytest.mark.parametrize("split", [None, 8], ids=["prefill", "decode"])
def test_yarn_logits_match_llama_cpp(name, split, tmp_path):
    p, meta, ref = H.fixture(name, tmp_path)
    ours, c, _ = H.run(p, meta["tokens"], split=split)
    assert c.mscale > 1.0 and 0.0 < float(np.min(c.fscale)) < 1.0 == float(np.max(c.fscale)), (c.mscale, c.fscale)
    assert H.nmse(ours, ref) <= H.NMSE_MAX, (name, split, H.nmse(ours, ref))


def test_yarn_factors_follow_the_correction_range():
    from mashbox.gguf.archs import rope
    k = lambda s, d=None: {"rope.scaling.factor": 4.0, "rope.scaling.original_context_length": 32768}.get(s, d)
    f, scale = rope.yarn(k, 128, 1e6, 131072)
    # dim(32) = 128 ln(32768 / (64 pi)) / (2 ln 1e6) = 23.6 -> 23; dim(1) = 39.6 -> 40
    assert f[:24].tolist() == [1.0] * 24 and np.allclose(f[40:], 0.25) and 0.25 < f[39] < 1.0 and np.all(np.diff(f) <= 0)
    assert abs(scale - (1 + 0.1 * np.log(4.0))) < 1e-12


def test_dense_files_run_yarn_without_the_log_multiplier(tmp_path):
    """llama.cpp reads rope.scaling.yarn_log_multiplier only in its mistral3 and deepseek2-family loaders, so a qwen3
    file that carries it runs as the same file without it: the qwen3-yarn-f32 weights plus the key, against that
    fixture's llama.cpp logits."""
    _, meta, ref = H.fixture("qwen3-yarn-f32", tmp_path)
    fx = H.FXT.FIXTURES["qwen3-yarn-f32"]
    p = str(tmp_path / "logmul.gguf")
    H.FXT.write_dense(p, fx["arch"], seed=fx["seed"], layers=fx["layers"], quant=None,
                      rope=dict(fx["rope"], yarn_log_multiplier=0.1))
    ours, c, _ = H.run(p, meta["tokens"])
    assert abs(c.mscale - (1 + 0.1 * np.log(4.0))) < 1e-6, c.mscale
    assert H.nmse(ours, ref) <= H.NMSE_MAX, H.nmse(ours, ref)
