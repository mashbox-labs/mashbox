"""GGUF qwen35 forward: the engine's per-call hooks, with the Qwen port's semantics (models/qwen3_8_27b/model.py).

  keep    (B,) bool: rows whose linear-attention state the call may change; the others are written back as they were
          (continuous batching: padding rows of a decode bucket, rows paused mid-prefill)
  hidden  out["hid"] = post-norm hidden state at every position (the MTP head's input)
  verify  speculative verify of T tokens: T exact one-token decode steps in the linear-attention layers, inputs kept in
          out["rec"], no state committed; gated_delta_commit / conv_commit then commit the first n[b] of them

Bounds, fixed before the first run (f32 activations, CPU):
  keep: state rows of keep=False rows equal their old values bit for bit; keep=True rows equal a call without keep, bit for bit
  commit masking: steps past n[b] leave the state bit for bit (commit over T recorded steps == commit over the first n)
  verify vs one-token decode steps (different matmul shapes, so rounding may differ): logits NMSE <= 1e-9 and the same
          argmax at every position; committed linear state, conv tail and K/V after n[b] tokens within 1e-5 of the decode
          path's, relative to the largest magnitude
  hidden: head(hid at the last position) within 1e-5 (relative) of the call's own logits
Both layer paths run: the scan (uniform layer groups) and the unrolled one (groups with different block types).
"""
import os

import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FXT)

from mashbox import ops  # noqa: E402
from mashbox.gguf.reader import File  # noqa: E402
from mashbox.gguf.archs import qwen35 as A  # noqa: E402

i32, f32 = jnp.int32, jnp.float32
CTX, TC = 64, 32
STATIC = ("Tc", "contig", "all_logits", "verify", "hidden")          # what the engine branches jit with


@pytest.fixture(scope="module", params=["scan", "unrolled"])
def model(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(request.param)
    if request.param == "scan":
        p = FXT.build("qwen35-f32", str(d))
    else:
        p = FXT.write_qwen35(str(d / "mixed.gguf"), seed=4, layers=8, quant=["Q8_0", "Q4_0"])
    f = File(p)
    c = A.Config(f, act=f32)
    W, info = A.load(f, c, use_kernels=False)
    assert info["layers_stacked"] is (request.param == "scan")
    fwd = jax.jit(A.make_forward(c), static_argnames=STATIC)
    return c, W, fwd


def prefill(c, W, fwd, B=2):
    rng = np.random.default_rng(7)
    lens = np.array([8, 6][:B], np.int32)
    toks = rng.integers(3, c.V - 3, (B, 8)).astype(np.int32)
    ck, cv = A.new_state(c, B, CTX)
    r = fwd(W, ck, cv, jnp.asarray(toks), jnp.zeros((B,), i32), jnp.asarray(lens), jnp.arange(B, dtype=i32), 0)
    return r["ck"], r["cv"], lens, rng


def bits(a):
    a = np.asarray(a)
    return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint16)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(1e-30, np.max(np.abs(b))))


def nmse(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def test_keep_mask_leaves_other_rows_untouched(model):
    c, W, fwd = model
    ck, cv, lens, rng = prefill(c, W, fwd)
    R0, C0 = (np.asarray(x) for x in cv)
    tok = jnp.asarray(rng.integers(3, c.V - 3, (2, 1)).astype(np.int32))
    args = (tok, jnp.asarray(lens), jnp.ones((2,), i32), jnp.arange(2, dtype=i32), TC)
    a = fwd(W, *jax.tree_util.tree_map(jnp.array, (ck, cv)), *args, contig=True)
    b = fwd(W, *jax.tree_util.tree_map(jnp.array, (ck, cv)), *args, contig=True, keep=jnp.asarray([True, False]))
    Ra, Ca = (np.asarray(x) for x in a["cv"])
    Rb, Cb = (np.asarray(x) for x in b["cv"])
    assert (bits(Rb[:, :, 1]) == bits(R0[:, :, 1])).all() and (bits(Cb[:, :, 1]) == bits(C0[:, :, 1])).all()
    assert (bits(Rb[:, :, 0]) == bits(Ra[:, :, 0])).all() and (bits(Cb[:, :, 0]) == bits(Ca[:, :, 0])).all()
    assert not (bits(Ra[:, :, 1]) == bits(R0[:, :, 1])).all()          # without keep, row 1's state does change
    assert (bits(a["lg"]) == bits(b["lg"])).all()


def test_hidden_is_the_post_norm_state(model):
    c, W, fwd = model
    ck, cv, lens, rng = prefill(c, W, fwd)
    tok = jnp.asarray(rng.integers(3, c.V - 3, (2, 4)).astype(np.int32))
    r = fwd(W, ck, cv, tok, jnp.asarray(lens), jnp.full((2,), 4, i32), jnp.arange(2, dtype=i32), TC, contig=True,
            hidden=True)
    assert r["hid"].shape == (2, 4, c.D)
    lg = ops.qmatmul(r["hid"][:, -1], W["head"])
    e = rel(lg, r["lg"])
    print(f"head(hid[last]) vs lg: rel {e:.2e}")
    assert e <= 1e-5, e


def test_verify_then_commit_matches_decode_steps(model):
    c, W, fwd = model
    K1 = 4
    ck, cv, lens, rng = prefill(c, W, fwd)
    d = rng.integers(3, c.V - 3, (2, K1)).astype(np.int32)
    keep_ck, keep_cv = jax.tree_util.tree_map(jnp.array, (ck, cv))
    # one-token decode steps, state snapshot after each
    snaps, lgs = [], []
    for j in range(K1):
        r = fwd(W, ck, cv, jnp.asarray(d[:, j:j + 1]), jnp.asarray(lens + j), jnp.ones((2,), i32), jnp.arange(2, dtype=i32),
                TC, contig=True)
        ck, cv = r["ck"], r["cv"]
        lgs.append(np.asarray(r["lg"]))
        snaps.append([np.asarray(x) for x in (*ck, *cv)])
    # verify all K1 tokens at once, commit n = (K1, 2)
    n = np.array([K1, 2], np.int32)
    v = fwd(W, keep_ck, keep_cv, jnp.asarray(d), jnp.asarray(lens), jnp.full((2,), K1, i32), jnp.arange(2, dtype=i32), TC,
            contig=True, verify=True)
    Rv, Cv = v["cv"]
    assert (bits(Rv) == bits(keep_cv[0])).all() and (bits(Cv) == bits(keep_cv[1])).all(), "verify must not commit"
    lv = np.asarray(v["lg"])
    dl = np.stack(lgs, 1)
    e = nmse(lv, dl)
    same = int((lv.argmax(-1) == dl.argmax(-1)).sum())
    print(f"verify logits vs decode: NMSE {e:.3e}, argmax {same}/{dl.shape[0] * dl.shape[1]}, "
          f"bit-identical {bool((bits(lv) == bits(dl)).all())}")
    assert e <= 1e-9 and same == dl.shape[0] * dl.shape[1], (e, same)
    assert (np.asarray(v["tgt"]) == lv.argmax(-1)).all()
    R = A.gated_delta_commit(Rv, v["rec"], jnp.asarray(n), c.HV // c.HK)
    C = A.conv_commit(Cv, v["rec"]["x"], jnp.asarray(n))
    Kv, Vv = (np.asarray(x) for x in v["ck"])
    R, C = np.asarray(R), np.asarray(C)
    for b in range(2):
        Kd, Vd, Rd, Cd = snaps[n[b] - 1]
        P = lens[b] + n[b]
        er = [rel(R[:, :, b], Rd[:, :, b]), rel(C[:, :, b], Cd[:, :, b]),
              rel(Kv[:, b, :P], Kd[:, b, :P]), rel(Vv[:, b, :P], Vd[:, b, :P])]
        exact = [bool((bits(x) == bits(y)).all()) for x, y in ((R[:, :, b], Rd[:, :, b]), (C[:, :, b], Cd[:, :, b]))]
        print(f"row {b} n={n[b]}: rel R {er[0]:.2e} C {er[1]:.2e} K {er[2]:.2e} V {er[3]:.2e}; R, C bit-identical {exact}")
        assert max(er) <= 1e-5, er


def test_commit_masks_steps_past_n_bit_for_bit(model):
    c, W, fwd = model
    ck, cv, lens, rng = prefill(c, W, fwd)
    d = rng.integers(3, c.V - 3, (2, 4)).astype(np.int32)
    v = fwd(W, ck, cv, jnp.asarray(d), jnp.asarray(lens), jnp.full((2,), 4, i32), jnp.arange(2, dtype=i32), TC,
            contig=True, verify=True)
    Rv, Cv = v["cv"]
    n = jnp.asarray([2, 2], i32)
    cut = jax.tree_util.tree_map(lambda a: a[:, :, :, :2], v["rec"])
    rep = c.HV // c.HK
    assert (bits(A.gated_delta_commit(Rv, v["rec"], n, rep)) == bits(A.gated_delta_commit(Rv, cut, n, rep))).all()
    assert (bits(A.conv_commit(Cv, v["rec"]["x"], n)) == bits(A.conv_commit(Cv, cut["x"], n))).all()


def test_new_state_takes_extra_groups(model):
    c, W, fwd = model
    (K, V), _ = A.new_state(c, 2, CTX, extra_groups=1)
    assert K.shape[0] == c.NG + 1 and V.shape[0] == c.NG + 1
