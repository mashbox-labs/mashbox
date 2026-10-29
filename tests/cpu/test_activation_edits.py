"""Activation edits on the Qwen3.8-27B forward (models/qwen3_8_27b/model.py make_forward(edits=...)): the hook a tool
uses to change or read what a layer computes, on a tiny random model of the same architecture (no torch needed).

  - the base ActivationEdits (no change, no read) gives the same outputs, bit for bit, as edits=None (prefill and decode);
  - a residual add on one row leaves every other row bit-identical, and the residual it reads is the one it changed;
  - an MLP activation multiply equals the same edit made in the weights (a down-projection column set to zero);
  - the writer sites (embedding, attention output, MLP output) equal the same edit made in the weights;
  - reads come back stacked over the layers; the combinations the hook does not run on are refused.
"""
import json
import struct

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import ml_dtypes  # noqa: E402

from mashbox.models.qwen3_8_27b import model as M  # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW  # noqa: E402

CFG = dict(model_type="qwen3_5_text", vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
           num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=6,
           linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
           rms_norm_eps=1e-6, tie_word_embeddings=False, attn_output_gate=True,
           rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25))
S, CTX = 2, 64


def write_bshj(path, tensors):
    """The .bshj container (layout as in mashbox/bshj.py), as tests/cpu/qwen3_8_27b_checks.py writes it."""
    align, pos, ents, bufs = 4096, 0, {}, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {"bfloat16": "bf16", "float32": "f32"}[a.dtype.name]
        pos = -(-pos // align) * align
        ents[name] = dict(dtype=dt, shape=list(a.shape), offset=pos, nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    raw = json.dumps(dict(format="bshj", version="1.0", data_len=pos, tensors=ents), separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p, b in bufs:
            f.seek(off + p)
            f.write(b)
        f.truncate(off + pos)


def tiny_tensors(seed=0):
    c = CFG
    D, F, V = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    H, KV, HD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    HK, HV, DK, DV, CK = (c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                          c["linear_value_head_dim"], c["linear_conv_kernel_dim"])
    CONV = 2 * HK * DK + HV * DV
    rng = np.random.default_rng(seed)
    bf = lambda a: np.asarray(a, np.float32).astype(ml_dtypes.bfloat16)  # noqa: E731
    w = lambda *s: bf(rng.normal(0, 0.08, s))  # noqa: E731
    n1 = lambda n: bf(rng.normal(0, 0.3, n))  # noqa: E731
    T = {"model.embed_tokens.weight": w(V, D), "model.norm.weight": n1(D), "lm_head.weight": w(V, D)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        T.update({p + "input_layernorm.weight": n1(D), p + "post_attention_layernorm.weight": n1(D),
                  p + "mlp.gate_proj.weight": w(F, D), p + "mlp.up_proj.weight": w(F, D), p + "mlp.down_proj.weight": w(D, F)})
        if (i + 1) % c["full_attention_interval"]:
            la = p + "linear_attn."
            T.update({la + "in_proj_qkv.weight": w(CONV, D), la + "in_proj_z.weight": w(HV * DV, D),
                      la + "in_proj_a.weight": w(HV, D), la + "in_proj_b.weight": w(HV, D),
                      la + "conv1d.weight": bf(rng.normal(0, 0.4, (CONV, 1, CK))),
                      la + "A_log": bf(np.log(rng.uniform(1, 16, HV))), la + "dt_bias": bf(rng.normal(0, 1, HV)),
                      la + "norm.weight": n1(DV), la + "out_proj.weight": w(D, HV * DV)})
        else:
            fa = p + "self_attn."
            T.update({fa + "q_proj.weight": w(2 * H * HD, D), fa + "k_proj.weight": w(KV * HD, D),
                      fa + "v_proj.weight": w(KV * HD, D), fa + "o_proj.weight": w(D, H * HD),
                      fa + "q_norm.weight": n1(HD), fa + "k_norm.weight": n1(HD)})
    return T


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("act_edits")
    (d / "config.json").write_text(json.dumps(CFG))
    write_bshj(str(d / "tiny.bshj"), tiny_tensors())
    c = M.Config(str(d))
    W, _ = QW.load(str(d / "tiny.bshj"), c)
    return c, W


PRE = dict(toks=np.array([[5, 17, 300, 2, 99, 410, 7, 64], [11, 3, 250, 480, 1, 42, 0, 0]], np.int32),
           p0=np.zeros(2, np.int32), lens=np.array([8, 6], np.int32), slots=np.arange(2, dtype=np.int32))


def run(c, W, edits=None, ed=None, Wx=None):
    """A prefill of two rows, then one decode step from the state it left -> (prefill out, decode out)."""
    fwd = jax.jit(M.make_forward(c, edits=edits), static_argnames=("Tc", "contig", "all_logits"))
    W = W if Wx is None else Wx
    ck, cv = M.new_state(c, S, CTX)
    kw = {} if edits is None else dict(ed=ed)
    a = fwd(W, ck, cv, PRE["toks"], PRE["p0"], PRE["lens"], PRE["slots"], Tc=0, all_logits=True, **kw)
    a = jax.tree_util.tree_map(np.asarray, a)
    tok = np.array([[a["lg"][0, 7].argmax()], [a["lg"][1, 5].argmax()]], np.int32)
    b = fwd(W, a["ck"], a["cv"], tok, PRE["lens"], np.ones(2, np.int32), PRE["slots"], Tc=16, contig=True, **kw)
    return a, jax.tree_util.tree_map(np.asarray, b)


def same(x, y):
    xl, yl = jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)
    return len(xl) == len(yl) and all(np.array_equal(np.asarray(p).view(np.uint8), np.asarray(q).view(np.uint8))
                                      for p, q in zip(xl, yl))


def core(o):
    return {k: o[k] for k in ("ck", "cv", "lg") + (("nxt",) if "nxt" in o else ())}


def test_base_hook_is_bit_identical_to_no_edits(tiny):
    c, W = tiny
    a0, b0 = run(c, W)
    a1, b1 = run(c, W, M.ActivationEdits())
    assert same(core(a0), core(a1)) and same(core(b0), core(b1))
    assert a1["edit_reads"] == {} and b1["edit_reads"] == {}
    assert "edit_reads" not in a0 and "edit_reads" not in b0


class AddAt(M.ActivationEdits):
    """Adds call["v"][row] to the residual leaving the layers whose lay["on"] is 1; reads it before and after."""

    def resid(self, l, call, lay, x, ctx):
        y = jnp.where(lay["on"] > 0, (x.astype(jnp.float32) + call["v"][:, None]).astype(x.dtype), x)
        return y, {"before": x, "after": y}


def test_residual_add_reads_and_isolates_rows(tiny):
    c, W = tiny
    lay = 5
    v = np.zeros((2, c.D), np.float32)
    v[0] = np.random.default_rng(3).normal(0, 2.0, c.D)
    ed = {"call": {"v": jnp.asarray(v)}, "layer": {"on": jnp.asarray((np.arange(c.L) == lay).astype(np.float32))}}
    a0, b0 = run(c, W)
    a1, b1 = run(c, W, AddAt(), ed)
    r = a1["edit_reads"]
    assert r["before"].shape == (c.L, 2, 8, c.D) and r["after"].shape == (c.L, 2, 8, c.D)
    want = (r["before"][lay, 0].astype(np.float32) + v[0]).astype(ml_dtypes.bfloat16)
    assert np.array_equal(r["after"][lay, 0].view(np.uint16), want.view(np.uint16))
    others = [l for l in range(c.L) if l != lay]
    assert np.array_equal(r["after"][others].view(np.uint16), r["before"][others].view(np.uint16))
    assert not np.array_equal(a1["lg"][0], a0["lg"][0])                      # the edit reaches the logits
    assert np.array_equal(a1["lg"][1], a0["lg"][1]) and np.array_equal(b1["lg"][1], b0["lg"][1])   # row 1 untouched
    for o0, o1 in ((a0, a1), (b0, b1)):          # slot 1: K/V (NG, S+1, ..) slot axis 1; linear state (NG, FI-1, S+1, ..) axis 2
        for x0, x1, ax in ((o0["ck"][0], o1["ck"][0], 1), (o0["ck"][1], o1["ck"][1], 1),
                           (o0["cv"][0], o1["cv"][0], 2), (o0["cv"][1], o1["cv"][1], 2)):
            assert np.array_equal(np.take(x0, 1, axis=ax).view(np.uint8), np.take(x1, 1, axis=ax).view(np.uint8))


def _zero_wd_col(W, c, layer, j):
    g, i = divmod(layer, c.FI)
    wd = [np.asarray(x) for x in W["g"]["mlp"]["wd"]]
    wd[i] = wd[i].copy()
    wd[i][g, :, j] = 0
    Wx = dict(W, g=dict(W["g"], mlp=dict(W["g"]["mlp"], wd=[jnp.asarray(x) for x in wd])))
    return Wx


class MulAct(M.ActivationEdits):
    def act(self, l, call, lay, a, ctx):
        return (a.astype(jnp.float32) * lay["m"]).astype(a.dtype), {}


def test_activation_multiply_equals_the_weight_edit(tiny):
    c, W = tiny
    layer, j = 6, 17
    m = np.ones((c.L, c.F), np.float32)
    m[layer, j] = 0.0
    a1, b1 = run(c, W, MulAct(), {"call": None, "layer": {"m": jnp.asarray(m)}})
    a2, b2 = run(c, W, Wx=_zero_wd_col(W, c, layer, j))
    assert same(core(a1), core(a2)) and same(core(b1), core(b2))


class ZeroWriter(M.ActivationEdits):
    def __init__(self, site):
        self.site = site

    def writer(self, site, l, call, lay, ctx):
        if site != self.site:
            return None
        on = jnp.float32(1.0) if lay is None else lay["z"]
        return lambda y: jnp.where(on > 0, jnp.zeros_like(y), y)


def _zero(W, path, g):
    """W with the weight at path (a list of per-group-position arrays or a stacked array) zero for group g."""
    def put(t, keys):
        k = keys[0]
        if len(keys) == 1:
            a = np.asarray(t[k]).copy()
            a[g] = 0
            return dict(t, **{k: jnp.asarray(a)})
        if isinstance(t[k], list):
            i = keys[1]
            lst = list(t[k])
            a = np.asarray(lst[i]).copy()
            a[g] = 0
            lst[i] = jnp.asarray(a)
            return dict(t, **{k: lst})
        return dict(t, **{k: put(t[k], keys[1:])})
    return put(W, path)


@pytest.mark.parametrize("site,layer", [("mlp", 2), ("attn", 1), ("attn", 3)])
def test_writer_sites_equal_the_weight_edit(tiny, site, layer):
    c, W = tiny
    g, i = divmod(layer, c.FI)
    z = (np.arange(c.L) == layer).astype(np.float32)
    a1, b1 = run(c, W, ZeroWriter(site), {"call": None, "layer": {"z": jnp.asarray(z)}})
    if site == "mlp":
        Wx = _zero(W, ["g", "mlp", "wd", i], g)
    elif i < c.FI - 1:
        Wx = _zero(W, ["g", "lin", "out", i], g)
    else:
        Wx = _zero(W, ["g", "full", "wo"], g)
    a2, b2 = run(c, W, Wx=Wx)
    assert same(core(a1), core(a2)) and same(core(b1), core(b2))


def test_embed_writer_equals_the_weight_edit(tiny):
    c, W = tiny
    a1, b1 = run(c, W, ZeroWriter("embed"), None)
    Wx = dict(W, embed=jnp.zeros_like(W["embed"]))
    a2, b2 = run(c, W, Wx=Wx)
    assert same(core(a1), core(a2)) and same(core(b1), core(b2))


def test_refused_combinations(tiny):
    c, W = tiny
    with pytest.raises(ValueError, match="one chip"):
        M.make_forward(c, edits=M.ActivationEdits(), kv_in_loop=True)
    ck, cv = M.new_state(c, S, CTX)
    with pytest.raises(ValueError, match="needs make_forward"):
        M.make_forward(c)(W, ck, cv, PRE["toks"], PRE["p0"], PRE["lens"], PRE["slots"], 0, ed={"call": None})
    tok = np.ones((2, 1), np.int32)
    with pytest.raises(ValueError, match="without speculation"):
        M.make_forward(c, edits=M.ActivationEdits())(W, ck, cv, tok, PRE["lens"], np.ones(2, np.int32), PRE["slots"], 16,
                                                     contig=True, verify=True)
