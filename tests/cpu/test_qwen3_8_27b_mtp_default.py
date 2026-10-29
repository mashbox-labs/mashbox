"""Speculative decoding is off by default and opt-in: the draft file ships with the weights (models.json lists it), and
`mashbox run --mtp K` (1 to 3) turns speculation on with K drafts per pass. Guards: without --mtp the run decodes without it,
even when the model has a draft file, and fetches nothing; `--mtp K` sets K; `--mtp 0` is off; --mtp-file needs --mtp K;
asking for it on a model without a draft file, under tensor parallel, or when the draft file cannot be fetched is an error;
K outside 0..3 is refused. Tiny MTP head on CPU, no download."""
import argparse
import hashlib
import json
import struct

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
pytest.importorskip("jax")

from mashbox import cli  # noqa: E402
from mashbox.models import qwen3_8_27b as Q  # noqa: E402

D, F, H, KV, HD = 64, 96, 4, 2, 32


def write_bshj(path, tensors, meta):
    align, pos, ents, bufs = 4096, 0, {}, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {"bfloat16": "bf16", "float32": "f32", "int8": "i8"}[a.dtype.name]
        pos = -(-pos // align) * align
        ents[name] = dict(dtype=dt, shape=list(a.shape), offset=pos, nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    raw = json.dumps(dict(format="bshj", version="1.0", data_len=pos, tensors=ents, **meta), separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p, b in bufs:
            f.seek(off + p)
            f.write(b)
        f.truncate(off + pos)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("mtp")
    json.dump(dict(model_type="qwen3_5_text", vocab_size=512, hidden_size=D, intermediate_size=F, num_hidden_layers=4,
                   num_attention_heads=H, num_key_value_heads=KV, head_dim=HD, linear_num_key_heads=2, linear_num_value_heads=6,
                   linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
                   rms_norm_eps=1e-6, rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25),
                   tie_word_embeddings=False, attn_output_gate=True, mtp_num_hidden_layers=1), open(d / "config.json", "w"))
    rng = np.random.default_rng(0)
    r = lambda *s: rng.normal(0, 0.1, s).astype(ml_dtypes.bfloat16)
    m = "mtp.layers.0."
    T = {"mtp.fc.weight": r(D, 2 * D), "mtp.pre_fc_norm_embedding.weight": r(D), "mtp.pre_fc_norm_hidden.weight": r(D),
         "mtp.norm.weight": r(D), m + "input_layernorm.weight": r(D), m + "post_attention_layernorm.weight": r(D),
         m + "self_attn.q_proj.weight": r(2 * H * HD, D), m + "self_attn.k_proj.weight": r(KV * HD, D),
         m + "self_attn.v_proj.weight": r(KV * HD, D), m + "self_attn.o_proj.weight": r(D, H * HD), m + "self_attn.q_norm.weight": r(HD),
         m + "self_attn.k_norm.weight": r(HD), m + "mlp.gate_proj.weight": r(F, D), m + "mlp.up_proj.weight": r(F, D),
         m + "mlp.down_proj.weight": r(D, F)}
    p = d / "draft.bshj"
    write_bshj(p, T, {"bshj": {"draft": {"method": "mtp", "default_k": 2}}})
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    row = {"name": "draft.bshj", "kind": "draft", "format": "BSHJ", "gb": 0.0, "sha256": sha, "draft": {"method": "mtp", "default_k": 2}}
    return Q.Config(str(d)), str(p), row


def args(**kw):
    return argparse.Namespace(**{"mtp": None, "mtp_file": None, **kw})


def test_default_off_and_mtp_2_on(tiny, monkeypatch):
    cfg, path, row = tiny
    from mashbox import hub

    def no_fetch(*a, **k):
        raise AssertionError("the default run must not fetch the draft file")
    monkeypatch.setattr(hub, "draft_file", no_fetch)
    m = {"id": "t", "files": [row]}
    assert cli.speculation(args(), m, cfg, 1) == {}
    assert cli.speculation(args(mtp=0), m, cfg, 1) == {}
    monkeypatch.undo()
    s = cli.speculation(args(mtp=2, mtp_file=path), m, cfg, 1)
    assert s["draft_k"] == 2 and s["mtp"][0].__name__.endswith("spec")


def test_mtp_file_needs_mtp_k(tiny):
    cfg, path, row = tiny
    m = {"id": "t", "files": [row]}
    with pytest.raises(SystemExit):
        cli.speculation(args(mtp_file=path), m, cfg, 1)


def test_mtp_k_and_off(tiny):
    cfg, path, row = tiny
    m = {"id": "t", "files": [row]}
    assert cli.speculation(args(mtp=1, mtp_file=path), m, cfg, 1)["draft_k"] == 1
    assert cli.speculation(args(mtp=0, mtp_file=path), m, cfg, 1) == {}


def test_no_draft_file(tiny):
    cfg, path, row = tiny
    m = {"id": "t", "files": []}
    assert cli.speculation(args(), m, cfg, 1) == {}
    with pytest.raises(SystemExit):
        cli.speculation(args(mtp=2), m, cfg, 1)


def test_tensor_parallel_and_range(tiny):
    cfg, path, row = tiny
    m = {"id": "t", "files": [row]}
    assert cli.speculation(args(), m, cfg, 2) == {}
    with pytest.raises(SystemExit):
        cli.speculation(args(mtp=2, mtp_file=path), m, cfg, 2)
    with pytest.raises(SystemExit):
        cli.speculation(args(mtp=4, mtp_file=path), m, cfg, 1)


def test_draft_not_fetchable(tiny, monkeypatch):
    cfg, path, row = tiny
    from mashbox import hub
    calls = []

    def gone(*a, **k):
        calls.append(a)
        raise FileNotFoundError("not in the repo at this revision")
    monkeypatch.setattr(hub, "fetch", gone)
    m = {"id": "t", "files": [row], "hf_repo": "x/y", "hf_revision": "0" * 40}
    assert cli.speculation(args(), m, cfg, 1) == {} and calls == []
    with pytest.raises(SystemExit):
        cli.speculation(args(mtp=2), m, cfg, 1)
    assert calls


def test_parser_default_is_unset():
    p = cli.build_parser()
    assert p.parse_args(["run", "qwen3-8-27b"]).mtp is None
    assert p.parse_args(["run", "qwen3-8-27b", "--mtp", "0"]).mtp == 0
