"""Tiny GGUF fixtures for the gemma3, exaone and granite CPU tests, and llama.cpp's reference logits and token ids for
them. Same method and output layout as fixtures.py (which this file loads and reuses):

  python tools/gguf_fixtures/more.py make --llama-cpp DIR [--out tests/fixtures/gguf] [--only NAME ...]
  python tools/gguf_fixtures/more.py tok  --llama-cpp DIR [--out tests/fixtures/gguf]

`make` writes <name>.json and <name>.logits.f32 per fixture (plus <name>.f32ref.logits.f32 for block-type fixtures);
`tok` writes gemma3-spm.tokens.json: llama.cpp's llama-tokenize ids for TEXTS on the gemma3 fixture, whose vocabulary
is a small SentencePiece one (tokenizer.ggml.model "llama", with scores and byte tokens), and spm-prefix.tokens.json:
the same on that vocabulary with a space prefix (tokenizer.ggml.add_space_prefix true, and left out). DIR is a llama.cpp
checkout at fixtures.LLAMA_CPP_COMMIT with a CPU build in DIR/build.

Random weights, small shapes, seeded. Nothing here is a trained model.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_s = importlib.util.spec_from_file_location("gguf_fixtures_base",
                                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.py"))
base = importlib.util.module_from_spec(_s)
_s.loader.exec_module(base)
sha256, TOKENS, LLAMA_CPP_COMMIT = base.sha256, base.TOKENS, base.LLAMA_CPP_COMMIT

SPACE = "▁"
GEMMA_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n"
                  "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
TEXTS = ["the hat", " the in a hat", "thethe in", "hello\nworld", "in the  hat", "aaaa", "über naïve", "東京 🙂",
         "  leading", "trailing  ", "a\n\nb", "<start_of_turn>user\nhi<end_of_turn>\n", "x" * 12, "tab\there",
         SPACE + " literal", "sat on the mat."]


def spm_vocab():
    """A small SentencePiece vocabulary in llama.cpp's GGUF layout: 4 special tokens, the 256 byte tokens, pieces with
    scores (longer merges score lower, as SentencePiece orders them), two turn markers. 286 tokens."""
    toks = ["<pad>", "<eos>", "<bos>", "<unk>"]
    types = [3, 3, 3, 2]
    scores = [0.0] * 4
    for b in range(256):
        toks.append(f"<0x{b:02X}>")
        types.append(6)
        scores.append(0.0)
    pieces = [SPACE, "t", "h", "e", "a", "i", "n", "s", "o", "\n", "he", SPACE + "t", "in", "at", SPACE + "a",
              SPACE + "the", SPACE + "in", SPACE + "s", "th", SPACE + "h", SPACE + "hat", SPACE + SPACE, "the", "aa"]
    for j, p in enumerate(pieces):
        toks.append(p)
        types.append(1)
        scores.append(-1.0 - (len(p) > 1) * (j + 1))
    for t in ("<start_of_turn>", "<end_of_turn>"):
        toks.append(t)
        types.append(3)
        scores.append(0.0)
    return toks, types, scores


GEMMA3 = dict(D=256, F=384, H=4, KV=2, HD=32, window=4, theta=1e6, factor=8.0, eps=1e-6, ctx=4096)
DENSE2 = dict(exaone=dict(D=256, F=384, H=8, KV=2, HD=32, theta=1e6, eps=1e-5, ctx=4096, freqs=True, tie=False),
              granite=dict(D=256, F=384, H=8, KV=2, HD=32, theta=1e7, eps=1e-5, ctx=4096, tie=True,
                           scales=dict(attention_scale=0.0625, embedding_scale=12.0, residual_scale=0.22,
                                       logit_scale=8.0)))


def _put(w, rng, quant, override):
    import gguf
    qcycle, k = list(quant or []), [0]

    def put(name, a, big=False):
        if override is not None and name in override:
            w.add_tensor(name, np.ascontiguousarray(override[name], dtype=np.float32))
            return
        if big and qcycle:
            t = gguf.GGMLQuantizationType[qcycle[k[0] % len(qcycle)]]
            k[0] += 1
            w.add_tensor(name, gguf.quants.quantize(a, t), raw_dtype=t)
        else:
            w.add_tensor(name, a)
    lin = lambda rows, cols: (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)
    norm = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return put, lin, norm


def _close(w):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_gemma3(path, seed=0, layers=6, quant=None, override=None, prefix=False):
    """A tiny random gemma3 GGUF in llama.cpp's conventions: 6 layers = 5 sliding-window layers (window 4, so the 16 test
    tokens cross it) and 1 full layer (the default pattern of 6), linear rope scaling 8 on the full layer, the default
    sliding-window rope base, a head size that is not embedding_length / head_count, q/k norms, post norms, tied head,
    and the small SentencePiece vocabulary of spm_vocab(). prefix: tokenizer.ggml.add_space_prefix (None leaves the key out,
    which llama.cpp reads as true)."""
    import gguf
    c, A = GEMMA3, "gemma3"
    rng = np.random.default_rng(seed)
    D, F, H, KV, HD = c["D"], c["F"], c["H"], c["KV"], c["HD"]
    toks, types, scores = spm_vocab()
    V = len(toks)
    w = gguf.GGUFWriter(path, A)
    w.add_name("mashbox-tiny-gemma3")
    w.add_context_length(c["ctx"])
    w.add_embedding_length(D)
    w.add_block_count(layers)
    w.add_feed_forward_length(F)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_rope_freq_base(c["theta"])
    w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    w.add_rope_scaling_factor(c["factor"])
    w.add_uint32(f"{A}.attention.sliding_window", c["window"])
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_scores(scores)
    w.add_bos_token_id(2)
    w.add_eos_token_id(1)
    w.add_unk_token_id(3)
    w.add_pad_token_id(0)
    w.add_add_bos_token(True)
    if prefix is not None:
        w.add_add_space_prefix(prefix)
    w.add_chat_template(GEMMA_TEMPLATE)
    put, lin, norm = _put(w, rng, quant, override)
    put("token_embd.weight", (0.5 * rng.standard_normal((V, D))).astype(np.float32))
    put("output_norm.weight", norm(D))
    for l in range(layers):
        b = f"blk.{l}."
        put(b + "attn_norm.weight", norm(D))
        put(b + "attn_q.weight", lin(H * HD, D), big=True)
        put(b + "attn_k.weight", lin(KV * HD, D), big=True)
        put(b + "attn_v.weight", lin(KV * HD, D), big=True)
        put(b + "attn_output.weight", lin(D, H * HD), big=True)
        put(b + "attn_q_norm.weight", norm(HD))
        put(b + "attn_k_norm.weight", norm(HD))
        put(b + "post_attention_norm.weight", norm(D))
        put(b + "ffn_norm.weight", norm(D))
        put(b + "ffn_gate.weight", lin(F, D), big=True)
        put(b + "ffn_up.weight", lin(F, D), big=True)
        put(b + "ffn_down.weight", lin(D, F), big=True)
        put(b + "post_ffw_norm.weight", norm(D))
    _close(w)
    return path


def write_dense2(path, arch, seed=0, layers=4, quant=None, override=None, pre="llama-bpe"):
    """A tiny random exaone or granite GGUF in llama.cpp's conventions (the llama tensor names): exaone with rope
    frequency factors and its own output head, granite with its four scalars and a tied head. The vocabulary is the
    shared tiny gpt2 one (llama.cpp reads the vocabulary independently of the architecture; the test feeds ids). pre: the
    pre-tokenizer name written (the real granite and EXAONE files carry "refact" and "exaone")."""
    import gguf
    c = DENSE2[arch]
    rng = np.random.default_rng(seed)
    D, F, H, KV, HD = c["D"], c["F"], c["H"], c["KV"], c["HD"]
    toks, types, merges = base.vocab()
    V = len(toks)
    w = gguf.GGUFWriter(path, arch)
    w.add_name(f"mashbox-tiny-{arch}")
    w.add_context_length(c["ctx"])
    w.add_embedding_length(D)
    w.add_block_count(layers)
    w.add_feed_forward_length(F)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_rope_dimension_count(HD)
    w.add_rope_freq_base(c["theta"])
    for key, v in c.get("scales", {}).items():
        w.add_float32(f"{arch}.{key.replace('attention_scale', 'attention.scale')}", v)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(V - 1)
    w.add_bos_token_id(V - 3)
    w.add_add_bos_token(False)
    w.add_chat_template(base.CHAT_TEMPLATE)
    put, lin, norm = _put(w, rng, quant, override)
    put("token_embd.weight", (0.5 * rng.standard_normal((V, D))).astype(np.float32))
    put("output_norm.weight", norm(D))
    if not c["tie"]:
        put("output.weight", lin(V, D), big=True)
    if c.get("freqs"):
        put("rope_freqs.weight", rng.uniform(1.0, 8.0, HD // 2).astype(np.float32))
    for l in range(layers):
        b = f"blk.{l}."
        put(b + "attn_norm.weight", norm(D))
        put(b + "attn_q.weight", lin(H * HD, D), big=True)
        put(b + "attn_k.weight", lin(KV * HD, D), big=True)
        put(b + "attn_v.weight", lin(KV * HD, D), big=True)
        put(b + "attn_output.weight", lin(D, H * HD), big=True)
        put(b + "ffn_norm.weight", norm(D))
        put(b + "ffn_gate.weight", lin(F, D), big=True)
        put(b + "ffn_up.weight", lin(F, D), big=True)
        put(b + "ffn_down.weight", lin(D, F), big=True)
    _close(w)
    return path


def write(name, path, override=None, quant=True):
    """fixture `name`'s file at path (fixtures.write's contract)"""
    fx = FIXTURES[name]
    q = fx.get("quant") if quant else None
    if fx["arch"] == "gemma3":
        return write_gemma3(path, seed=fx["seed"], layers=fx["layers"], quant=q, override=override)
    return write_dense2(path, fx["arch"], seed=fx["seed"], layers=fx["layers"], quant=q, override=override)


FIXTURES = {
    "gemma3-f32": dict(arch="gemma3", seed=30, layers=6, make="gguf-py", quant=None),
    "gemma3-pyquant": dict(arch="gemma3", seed=31, layers=6, make="gguf-py", quant=["Q8_0", "Q4_0", "Q5_0", "Q4_1", "Q5_1"]),
    "exaone-f32": dict(arch="exaone", seed=32, layers=4, make="gguf-py", quant=None),
    "exaone-pyquant": dict(arch="exaone", seed=35, layers=4, make="gguf-py", quant=["Q8_0", "Q4_0", "Q5_0"]),
    "granite-f32": dict(arch="granite", seed=33, layers=4, make="gguf-py", quant=None),
    "granite-pyquant": dict(arch="granite", seed=34, layers=4, make="gguf-py", quant=["Q8_0", "Q4_0", "Q5_0"]),
}
# fixtures.make / build / dequant_copy read FIXTURES and write from their own module: point this private copy at ours
base.FIXTURES, base.write = FIXTURES, write
build, dequant_copy, make = base.build, base.dequant_copy, base.make


PREFIX = {"true": True, "absent": None}       # tokenizer.ggml.add_space_prefix of the two prefix variants (Llama 2, Mistral 7B)


def tok(args):
    """llama-tokenize ids (no BOS, special tokens parsed) for TEXTS on the gemma3 fixture -> gemma3-spm.tokens.json, and
    on the same vocabulary with a space prefix (the key true, and the key left out) -> spm-prefix.tokens.json"""
    exe = os.path.join(os.path.abspath(args.llama_cpp), "build", "bin", "llama-tokenize")
    command = "llama-tokenize -m <fixture> -p <text> --ids --no-bos --log-disable"

    def ids(p):
        rows = []
        for t in TEXTS:
            r = subprocess.run([exe, "-m", p, "-p", t, "--ids", "--no-bos", "--log-disable"], capture_output=True,
                               text=True, check=True)
            rows.append(dict(text=t, ids=json.loads(r.stdout.strip().splitlines()[-1].replace("'", '"'))))
        return rows
    with tempfile.TemporaryDirectory() as td:
        p = build("gemma3-f32", td)
        meta = dict(fixture="gemma3-f32", gguf_sha256=sha256(p), llama_cpp_commit=LLAMA_CPP_COMMIT, command=command,
                    rows=ids(p))
        variants = {}
        for name, v in PREFIX.items():
            q = write_gemma3(os.path.join(td, f"prefix-{name}.gguf"), seed=FIXTURES["gemma3-f32"]["seed"], layers=1, prefix=v)
            variants[name] = dict(gguf_sha256=sha256(q), rows=ids(q))
        pmeta = dict(fixture="gemma3 writer, seed of gemma3-f32, 1 layer, add_space_prefix true / left out",
                     llama_cpp_commit=LLAMA_CPP_COMMIT, command=command, variants=variants)
    for fn, m in (("gemma3-spm.tokens.json", meta), ("spm-prefix.tokens.json", pmeta)):
        with open(os.path.join(args.out, fn), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(fn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("make", "tok"):
        m = sub.add_parser(n)
        m.add_argument("--llama-cpp", required=True, metavar="DIR")
        m.add_argument("--out", default=os.path.join("tests", "fixtures", "gguf"))
        m.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    return make(a) if a.cmd == "make" else tok(a)


if __name__ == "__main__":
    sys.exit(main())
