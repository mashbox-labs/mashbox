"""Tiny GGUF fixtures for the CPU architecture tests, and llama.cpp's reference logits for them.

  python tools/gguf_fixtures/fixtures.py make --arch qwen35 --llama-cpp DIR [--out tests/fixtures/gguf]

writes, per fixture, `<name>.json` (file sha256, the token ids, llama.cpp commit, the logits' sha256) and
`<name>.logits.f32` (n_tokens x n_vocab float32 from llama.cpp's CPU backend). Fixtures that gguf-py can write by itself
(float weights, and the block types gguf-py can quantize) are NOT committed: the test rebuilds them from the seed with
`write(...)` and checks the sha256. Fixtures with K-quant and IQ blocks are made with llama.cpp's own llama-quantize and
committed as `.gguf` (small). DIR is a llama.cpp checkout at the pinned commit with a CPU build in DIR/build (only
needed to make fixtures; the tests need neither llama.cpp nor a chip).

Random weights, small shapes, seeded. Nothing here is a trained model.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

LLAMA_CPP_COMMIT = "a37cc46"           # upstream ggml-org/llama.cpp commit the committed logits come from
TOKENS = [262, 72, 101, 256, 258, 32, 107, 263, 10, 262, 97, 259, 260, 33, 46, 120]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def byte_chars():
    """GPT-2 byte-level alphabet: byte value -> printable unicode character"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def vocab():
    """264 tokens: the 256 byte tokens, 5 merged tokens, 3 special tokens (Qwen names)"""
    bc = byte_chars()
    toks = [bc[b] for b in range(256)]
    merges = ["Ġ t", "h e", "Ġt he", "i n", "Ġ a"]
    toks += ["Ġt", "he", "Ġthe", "in", "Ġa"]
    types = [1] * len(toks)
    for s in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        toks.append(s)
        types.append(3)
    return toks, types, merges


CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

QWEN35 = dict(D=256, F=256, H=4, KV=2, HD=64, RD=16, sections=[3, 3, 2, 0], HK=2, HV=4, DK=64, CK=4, FI=4, theta=1e7,
              eps=1e-6, ctx=4096)


def write_qwen35(path, seed=0, layers=4, quant=None, nextn=False, override=None, moe=None):
    """Write a tiny random qwen35 GGUF in llama.cpp's conventions (norms stored + 1 except ssm_norm, ssm_a = -exp(A_log),
    conv squeezed). quant: None (all F32) or a list of gguf-py-quantizable type names cycled over the big linears.
    override: {tensor name: float32 array} written instead of the random values (same metadata, same order).
    moe: None, or dict(E=experts, K=experts used, FE=expert FFN size, FS=shared expert FFN size) -> architecture qwen35moe
    (llama.cpp's routed experts + sigmoid-gated shared expert in place of the dense FFN)."""
    import gguf
    c = QWEN35
    rng = np.random.default_rng(seed)
    D, F, H, KV, HD, HK, HV, DK = c["D"], c["F"], c["H"], c["KV"], c["HD"], c["HK"], c["HV"], c["DK"]
    DV = DK
    CONV = 2 * HK * DK + HV * DV
    toks, types, merges = vocab()
    V = len(toks)
    L_all = layers + (1 if nextn else 0)
    w = gguf.GGUFWriter(path, "qwen35moe" if moe else "qwen35")
    w.add_name("mashbox-tiny-qwen35")
    w.add_context_length(c["ctx"])
    w.add_embedding_length(D)
    w.add_block_count(L_all)
    w.add_feed_forward_length(F)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_rope_dimension_count(c["RD"])
    w.add_rope_freq_base(c["theta"])
    w.add_rope_dimension_sections(c["sections"])
    w.add_ssm_conv_kernel(c["CK"])
    w.add_ssm_inner_size(HV * DV)
    w.add_ssm_state_size(DK)
    w.add_ssm_time_step_rank(HV)
    w.add_ssm_group_count(HK)
    w.add_full_attention_interval(c["FI"])
    if moe:
        w.add_expert_count(moe["E"])
        w.add_expert_used_count(moe["K"])
        w.add_expert_feed_forward_length(moe["FE"])
        w.add_expert_shared_feed_forward_length(moe["FS"])
    if nextn:
        w.add_nextn_predict_layers(1)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(V - 1)
    w.add_pad_token_id(V - 3)
    w.add_bos_token_id(V - 3)
    w.add_add_bos_token(False)
    w.add_chat_template(CHAT_TEMPLATE)
    qcycle = list(quant or [])
    k = [0]

    def lin(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def norm(n, shift=True):
        return ((1.0 if shift else 1.0) + 0.1 * rng.standard_normal(n)).astype(np.float32)

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

    put("token_embd.weight", (0.5 * rng.standard_normal((V, D))).astype(np.float32))
    put("output_norm.weight", norm(D))
    put("output.weight", lin(V, D), big=True)
    for l in range(L_all):
        b = f"blk.{l}."
        put(b + "attn_norm.weight", norm(D))
        put(b + "post_attention_norm.weight", norm(D))
        if moe:
            E, FE, FS = moe["E"], moe["FE"], moe["FS"]
            put(b + "ffn_gate_inp.weight", lin(E, D))
            put(b + "ffn_gate_exps.weight", np.stack([lin(FE, D) for _ in range(E)]), big=True)
            put(b + "ffn_up_exps.weight", np.stack([lin(FE, D) for _ in range(E)]), big=True)
            put(b + "ffn_down_exps.weight", np.stack([lin(D, FE) for _ in range(E)]), big=True)
            put(b + "ffn_gate_inp_shexp.weight", (rng.standard_normal(D) / np.sqrt(D)).astype(np.float32))
            put(b + "ffn_gate_shexp.weight", lin(FS, D), big=True)
            put(b + "ffn_up_shexp.weight", lin(FS, D), big=True)
            put(b + "ffn_down_shexp.weight", lin(D, FS), big=True)
        else:
            put(b + "ffn_gate.weight", lin(F, D), big=True)
            put(b + "ffn_up.weight", lin(F, D), big=True)
            put(b + "ffn_down.weight", lin(D, F), big=True)
        if l < layers and (l + 1) % c["FI"]:
            put(b + "attn_qkv.weight", lin(CONV, D), big=True)
            put(b + "attn_gate.weight", lin(HV * DV, D), big=True)
            put(b + "ssm_beta.weight", lin(HV, D))
            put(b + "ssm_alpha.weight", lin(HV, D))
            put(b + "ssm_conv1d.weight", (0.3 * rng.standard_normal((CONV, c["CK"]))).astype(np.float32))
            put(b + "ssm_dt.bias", (0.5 * rng.standard_normal(HV)).astype(np.float32))
            put(b + "ssm_a", (-np.exp(rng.uniform(0.0, 1.5, HV))).astype(np.float32))
            put(b + "ssm_norm.weight", norm(DV, shift=False))
            put(b + "ssm_out.weight", lin(D, HV * DV), big=True)
        else:
            put(b + "attn_q.weight", lin(2 * H * HD, D), big=True)
            put(b + "attn_k.weight", lin(KV * HD, D), big=True)
            put(b + "attn_v.weight", lin(KV * HD, D), big=True)
            put(b + "attn_output.weight", lin(D, H * HD), big=True)
            put(b + "attn_q_norm.weight", norm(HD))
            put(b + "attn_k_norm.weight", norm(HD))
            if l >= layers:            # the MTP (nextn) layer
                put(b + "nextn.eh_proj.weight", lin(D, 2 * D), big=True)
                put(b + "nextn.enorm.weight", norm(D))
                put(b + "nextn.hnorm.weight", norm(D))
                put(b + "nextn.shared_head_norm.weight", norm(D))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


DENSE = dict(llama=dict(D=256, F=384, H=8, KV=2, HD=32, theta=5e5, eps=1e-5, ctx=4096, pre="llama-bpe", freqs=True,
                        bias=("attn_q", "attn_k", "attn_v", "attn_output")),
             qwen2=dict(D=256, F=384, H=8, KV=2, HD=32, theta=1e6, eps=1e-6, ctx=4096, pre="qwen2", bias=("attn_q", "attn_k", "attn_v")),
             qwen3=dict(D=256, F=512, H=8, KV=4, HD=64, theta=1e6, eps=1e-6, ctx=4096, pre="qwen2", qknorm=True))


def write_dense(path, arch="llama", seed=0, layers=4, quant=None, override=None, rope=None):
    """Write a tiny random dense-attention GGUF (llama / qwen2 / qwen3) in llama.cpp's conventions: llama with a
    rope_freqs tensor (Llama 3.1 style frequency factors) and q/k/v/output biases, qwen2 with q/k/v biases, qwen3 with
    per-head q/k norms and a head size that is not embedding_length / head_count. quant, override: as write_qwen35."""
    import gguf
    c = DENSE[arch]
    rng = np.random.default_rng(seed)
    D, F, H, KV, HD = c["D"], c["F"], c["H"], c["KV"], c["HD"]
    toks, types, merges = vocab()
    V = len(toks)
    w = gguf.GGUFWriter(path, arch)
    w.add_name(f"mashbox-tiny-{arch}")
    w.add_context_length(c["ctx"])
    w.add_embedding_length(D)
    w.add_block_count(layers)
    w.add_feed_forward_length(F)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    if HD != D // H:
        w.add_key_length(HD)
        w.add_value_length(HD)
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_rope_dimension_count(HD)
    w.add_rope_freq_base(c["theta"])
    for key, v in (rope or {}).items():          # rope scaling keys (the YaRN fixtures)
        (w.add_string if isinstance(v, str) else w.add_float32 if isinstance(v, float) else w.add_uint32)(f"{arch}.rope.scaling.{key}", v)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(c["pre"])
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(V - 1)
    w.add_bos_token_id(V - 3)
    w.add_add_bos_token(arch == "llama")
    w.add_chat_template(CHAT_TEMPLATE)
    qcycle = list(quant or [])
    k = [0]

    def lin(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

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

    put("token_embd.weight", (0.5 * rng.standard_normal((V, D))).astype(np.float32))
    put("output_norm.weight", norm(D))
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
        for t in c.get("bias", ()):
            n = D if t == "attn_output" else (H * HD if t == "attn_q" else KV * HD)
            put(b + t + ".bias", (0.2 * rng.standard_normal(n)).astype(np.float32))
        if c.get("qknorm"):
            put(b + "attn_q_norm.weight", norm(HD))
            put(b + "attn_k_norm.weight", norm(HD))
        put(b + "ffn_norm.weight", norm(D))
        put(b + "ffn_gate.weight", lin(F, D), big=True)
        put(b + "ffn_up.weight", lin(F, D), big=True)
        put(b + "ffn_down.weight", lin(D, F), big=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


GEMMA4 = dict(D=256, F=384, H=4, swa=[True, True, False, True, True, False], kv=[2, 2, 1, 2, 2, 1], hd_swa=32, hd=64,
              window=4, theta=1e6, theta_swa=1e4, eps=1e-6, softcap=30.0, ctx=4096)


def write_gemma4(path, seed=0, layers=6, quant=None, override=None):
    """Write a tiny random gemma4 GGUF (dense) in llama.cpp's conventions: sliding-window layers (2 K/V heads of 32,
    window 4, so the 16 test tokens cross it) and full layers (1 K/V head of 64, no attn_v: v is the k projection,
    rope_freqs factors), q/k norms, post-attention / post-FFN norms, layer_output_scale, tied output head, logit
    soft-cap. The tokenizer block is the shared tiny gpt2 vocabulary (llama.cpp reads the vocabulary independently of
    the architecture, and the test feeds token ids). quant, override: as write_qwen35."""
    import gguf
    c = GEMMA4
    rng = np.random.default_rng(seed)
    D, F, H = c["D"], c["F"], c["H"]
    swa = (c["swa"] * layers)[:layers]
    kvs = (c["kv"] * layers)[:layers]
    toks, types, merges = vocab()
    V = len(toks)
    A = "gemma4"
    w = gguf.GGUFWriter(path, A)
    w.add_name("mashbox-tiny-gemma4")
    w.add_context_length(c["ctx"])
    w.add_embedding_length(D)
    w.add_block_count(layers)
    w.add_feed_forward_length(F)
    w.add_head_count(H)
    w.add_head_count_kv(kvs)
    w.add_key_length(c["hd"])
    w.add_value_length(c["hd"])
    w.add_uint32(f"{A}.attention.key_length_swa", c["hd_swa"])
    w.add_uint32(f"{A}.attention.value_length_swa", c["hd_swa"])
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_rope_dimension_count(c["hd"])
    w.add_uint32(f"{A}.rope.dimension_count_swa", c["hd_swa"])
    w.add_rope_freq_base(c["theta"])
    w.add_float32(f"{A}.rope.freq_base_swa", c["theta_swa"])
    w.add_uint32(f"{A}.attention.sliding_window", c["window"])
    w.add_array(f"{A}.attention.sliding_window_pattern", swa)
    w.add_uint32(f"{A}.attention.shared_kv_layers", 0)
    w.add_uint32(f"{A}.embedding_length_per_layer_input", 0)
    w.add_float32(f"{A}.final_logit_softcapping", c["softcap"])
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(V - 1)
    w.add_bos_token_id(V - 3)
    w.add_add_bos_token(False)
    w.add_chat_template(CHAT_TEMPLATE)
    qcycle = list(quant or [])
    k = [0]

    def lin(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

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

    put("token_embd.weight", (0.5 * rng.standard_normal((V, D))).astype(np.float32))
    put("output_norm.weight", norm(D))
    put("rope_freqs.weight", rng.uniform(1.0, 8.0, c["hd"] // 2).astype(np.float32))
    for l in range(layers):
        b = f"blk.{l}."
        HD, KV = (c["hd_swa"] if swa[l] else c["hd"]), kvs[l]
        put(b + "attn_norm.weight", norm(D))
        put(b + "attn_q.weight", lin(H * HD, D), big=True)
        put(b + "attn_k.weight", lin(KV * HD, D), big=True)
        if swa[l]:
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
        put(b + "layer_output_scale.weight", rng.uniform(0.5, 1.5, 1).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def write(name, path, override=None, quant=True):
    """fixture `name`'s file at path: its seed, layers and (unless quant=False) block types; override as write_qwen35"""
    fx = FIXTURES[name]
    q = fx.get("quant") if quant else None
    if fx["arch"] == "gemma4":
        return write_gemma4(path, seed=fx["seed"], layers=fx["layers"], quant=q, override=override)
    if fx["arch"] in DENSE:
        return write_dense(path, fx["arch"], seed=fx["seed"], layers=fx["layers"], quant=q, override=override, rope=fx.get("rope"))
    return write_qwen35(path, seed=fx["seed"], layers=fx["layers"], quant=q, nextn=fx.get("nextn", False),
                        override=override, moe=fx.get("moe"))


# name -> how it is made. "gguf-py": rebuilt by the test from the seed. "llama-quantize": committed .gguf.
FIXTURES = {
    "qwen35-f32": dict(arch="qwen35", seed=0, layers=8, make="gguf-py", quant=None),
    "qwen35-pyquant": dict(arch="qwen35", seed=1, layers=4, make="gguf-py",
                           quant=["Q8_0", "Q4_0", "Q5_1", "Q4_1", "Q5_0", "BF16", "MXFP4", "Q8_0"]),
    "qwen35-kquant": dict(arch="qwen35", seed=2, layers=4, make="llama-quantize", base="Q4_K_M",
                          tensor_types=["ffn_down=q5_k", "attn_gate=iq4_xs", "ssm_out=iq4_nl", "attn_qkv=q6_k",
                                        "ffn_up=q3_k", "attn_output=q2_k", "attn_q=q8_0", "output=q6_k"]),
    "qwen35-mtp": dict(arch="qwen35", seed=3, layers=4, make="gguf-py", quant=None, nextn=True),
    "qwen35moe-f32": dict(arch="qwen35moe", seed=6, layers=8, make="gguf-py", quant=None,
                          moe=dict(E=8, K=2, FE=16, FS=24)),
    "qwen35moe-pyquant": dict(arch="qwen35moe", seed=7, layers=4, make="gguf-py", quant=["Q8_0", "Q4_0", "Q5_0", "Q8_0"],
                              moe=dict(E=8, K=2, FE=32, FS=32)),
    "llama-f32": dict(arch="llama", seed=10, layers=4, make="gguf-py", quant=None),
    "qwen2-f32": dict(arch="qwen2", seed=11, layers=4, make="gguf-py", quant=None),
    "qwen3-f32": dict(arch="qwen3", seed=12, layers=4, make="gguf-py", quant=None),
    "llama-pyquant": dict(arch="llama", seed=13, layers=4, make="gguf-py", quant=["Q8_0", "Q4_0", "Q5_0", "Q4_1", "Q5_1"]),
    "qwen3-kquant": dict(arch="qwen3", seed=14, layers=4, make="llama-quantize", base="Q4_K_M",
                         tensor_types=["ffn_down=q6_k", "ffn_up=q5_k", "attn_v=iq4_nl", "attn_output=iq4_xs",
                                       "attn_q=q8_0", "output=q6_k"]),
    "qwen3-yarn-f32": dict(arch="qwen3", seed=15, layers=4, make="gguf-py", quant=None,
                           rope=dict(type="yarn", factor=4.0, original_context_length=64)),
    "llama-yarn-f32": dict(arch="llama", seed=16, layers=4, make="gguf-py", quant=None,
                           rope=dict(type="yarn", factor=8.0, original_context_length=128)),
    "gemma4-f32": dict(arch="gemma4", seed=20, layers=6, make="gguf-py", quant=None),
    "gemma4-kquant": dict(arch="gemma4", seed=21, layers=6, make="llama-quantize", base="Q4_K_M",
                          tensor_types=["ffn_down=q6_k", "attn_v=q6_k", "attn_output=q5_k", "attn_q=q8_0"]),
}


def dequant_copy(name, src, dst):
    """an all-F32 copy of fixture `name` holding exactly gguf-py's dequantization of every tensor in `src`"""
    import gguf
    fx = FIXTURES[name]
    r = gguf.GGUFReader(src)
    ov = {t.name: gguf.quants.dequantize(t.data, t.tensor_type).reshape(tuple(reversed([int(x) for x in t.shape])))
          for t in r.tensors}
    return write(name, dst, override=ov, quant=False)


def build(name, outdir):
    """the fixture's .gguf: rebuilt (gguf-py kinds) into outdir, or the committed file"""
    fx = FIXTURES[name]
    p = os.path.join(outdir, f"{name}.gguf")
    if fx["make"] == "gguf-py":
        write(name, p)
    return p


def make(args):
    os.makedirs(args.out, exist_ok=True)
    root = os.path.abspath(args.llama_cpp)
    bindir = os.path.join(root, "build", "bin")
    exe = os.path.join(args.out, ".llama_logits")
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_logits.cpp")
    subprocess.run(["g++", "-O2", "-std=c++17", src, f"-I{root}/include", f"-I{root}/ggml/include", f"-L{bindir}",
                    "-lllama", f"-Wl,-rpath,{bindir}", "-o", exe], check=True)
    env = dict(os.environ, LD_LIBRARY_PATH=bindir)
    with tempfile.TemporaryDirectory() as td:
        for name, fx in FIXTURES.items():
            if args.only and name not in args.only:
                continue
            if fx["make"] == "gguf-py":
                p = build(name, td)
            else:
                f32 = write(name, os.path.join(td, f"{name}-src.gguf"), quant=False)
                p = os.path.join(args.out, f"{name}.gguf")
                cmd = [os.path.join(bindir, "llama-quantize")]
                for tt in fx["tensor_types"]:
                    cmd += ["--tensor-type", tt]
                subprocess.run(cmd + [f32, p, fx["base"]], check=True, env=env, stdout=subprocess.DEVNULL)
            out = os.path.join(args.out, f"{name}.logits.f32")
            r = subprocess.run([exe, p, out] + [str(t) for t in TOKENS], check=True, env=env, capture_output=True, text=True)
            n, nv = map(int, r.stdout.split()[-2:])
            extra = {}
            if fx.get("quant") or fx["make"] == "llama-quantize":
                # llama.cpp on an F32 file with exactly the dequantized weights: same weight values on both sides, so the
                # comparison is not blurred by llama.cpp's 8-bit activation rounding in its block-type dot products
                fq = dequant_copy(name, p, os.path.join(td, f"{name}-dequant-f32.gguf"))
                out2 = os.path.join(args.out, f"{name}.f32ref.logits.f32")
                subprocess.run([exe, fq, out2] + [str(t) for t in TOKENS], check=True, env=env, capture_output=True)
                extra = dict(f32ref_logits_sha256=sha256(out2), f32ref_gguf_sha256=sha256(fq),
                             f32ref="llama.cpp CPU on an all-F32 copy holding gguf-py's dequantization of every tensor")
            meta = dict(name=name, arch=fx["arch"], made_by=fx["make"], gguf_sha256=sha256(p), gguf_bytes=os.path.getsize(p),
                        tokens=TOKENS, n_vocab=nv, logits="f32 row-major (n_tokens, n_vocab)", logits_sha256=sha256(out),
                        llama_cpp_commit=LLAMA_CPP_COMMIT, llama_cpp_backend="CPU, float32 K/V cache, flash attention off",
                        fixture=fx, **extra)
            with open(os.path.join(args.out, f"{name}.json"), "w") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            print(name, meta["gguf_sha256"][:16], n, nv)
    os.remove(exe)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--arch", default="qwen35")
    m.add_argument("--llama-cpp", required=True, metavar="DIR")
    m.add_argument("--out", default=os.path.join("tests", "fixtures", "gguf"))
    m.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        return make(a)


if __name__ == "__main__":
    sys.exit(main())
