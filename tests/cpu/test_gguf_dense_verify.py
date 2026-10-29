"""The dense GGUF verify pass (gguf/archs/dense.py forward(verify=True), runtime/spec_step.py's contract): over T = K + 1
tokens of a contiguous row it gives the logits one-token decode gives at each position, the argmax of the head's output,
and nothing to commit (full attention only). Tiny f32 fixtures llama / qwen2 / qwen3 from tools/gguf_fixtures; CPU."""
import importlib.util
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import gguf  # noqa: E402,F401  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_fx = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_fx)
_fx.loader.exec_module(FXT)
DENSE = ("llama-f32", "qwen2-f32", "qwen3-f32")


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    from mashbox.gguf import open_model
    out = {}
    wd = tmp_path_factory.mktemp("dense_verify")
    for name in DENSE:
        f, A, c = open_model(FXT.build(name, str(wd)), act=jnp.float32)
        W, _ = A.load(f, c, use_kernels=False)
        out[name] = (c, W, A)
    return out


def loop_prompts(V, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ph = rng.integers(1, V, int(rng.integers(3, 7))).tolist()
        out.append((rng.integers(1, V, int(rng.integers(2, 6))).tolist() + ph * int(rng.integers(2, 4)))[:40])
    return out


@pytest.mark.parametrize("name,scale", [(n, 1.0) for n in DENSE] + [("llama-f32", 4.0)])
def test_dense_verify_equals_one_token_steps(dense, name, scale):
    """the dense verify pass over T = K + 1 tokens gives the logits one-token decode gives at each position (red on a tree
    whose dense forward refuses verify), and its commit is the identity (rec is empty); with a file's logit scale
    (granite's logit_scale) too, so the verify logits are the scaled ones plain decode gives"""
    import copy
    c, W, A = dense[name]
    if scale != 1.0:
        c = copy.copy(c)
        c.logit_scale = scale
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits", "verify", "hidden"))
    i32 = jnp.int32
    p = loop_prompts(c.V, 1, 3)[0][:12]
    ck, cv = A.new_state(c, 1, 128)
    r = fwd(W, ck, cv, jnp.asarray([p], i32), jnp.zeros(1, i32), jnp.asarray([len(p)], i32), jnp.zeros(1, i32), 0)
    nxt = [int(r["nxt"][0])]
    ck0, cv0 = r["ck"], r["cv"]
    ck, cv, one = ck0, cv0, []
    for j in range(4):          # one-token decode steps over [nxt, d1, d2, d3] (drafts = the greedy tokens)
        r1 = fwd(W, ck, cv, jnp.asarray([[nxt[j]]], i32), jnp.asarray([len(p) + j], i32), jnp.ones(1, i32), jnp.zeros(1, i32), 128)
        ck, cv = r1["ck"], r1["cv"]
        one.append(np.asarray(r1["lg"][0]))
        nxt.append(int(r1["nxt"][0]))
    rv = fwd(W, ck0, cv0, jnp.asarray([nxt[:4]], i32), jnp.asarray([len(p)], i32), jnp.full((1,), 4, i32),
             jnp.zeros(1, i32), 128, contig=True, verify=True)
    assert rv["lg"].shape == (1, 4, c.V) and rv["lg"].dtype == jnp.float32 and jax.tree.leaves(rv["rec"]) == []
    np.testing.assert_allclose(np.asarray(rv["lg"][0]), np.stack(one), rtol=2e-4, atol=2e-4)
    assert np.asarray(rv["tgt"][0]).tolist() == nxt[1:5]


def test_module_declares_verify():
    from mashbox.gguf.archs import dense as A
    assert A.SPEC_VERIFY is True
