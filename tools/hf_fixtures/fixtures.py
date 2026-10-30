"""Tiny random Hugging Face models and transformers' own float32 logits for them: the fixtures of tests/cpu/test_hf_*.py.

  python tools/hf_fixtures/fixtures.py make [names...]    needs torch + transformers (+ gguf for the Q8_0 references)
  python tools/hf_fixtures/fixtures.py chat               the chat-a2 template fixture (downloads one tokenizer)

Each fixture is tests/fixtures/hf/<name>/: config.json + model.safetensors (float32, written by save_pretrained) +
logits.f32 (transformers' logits for meta.json's 16 tokens, one forward) + for Q8_0_REF models logits.q8_0.f32
(transformers' logits after every weight is rounded to bfloat16 and every multiplied weight is replaced by gguf-py's
dequantize(quantize(., Q8_0)) of it, in the GGUF tensor's layout: the same values the Q8_0-on-load path multiplies) +
meta.json (versions, seed, sha256 of every
file). The weights are random (seeded) at sizes where every block contributes to the logits. Also here, with NumPy
only (the tests use them): write_safetensors and write_bshj (a lossless container in mashbox.bshj's documented layout).
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "hf")
SEED, T = 1234, 16

BASE = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
            rms_norm_eps=1e-6, max_position_embeddings=256)
CONFIGS = {
    # llama: rope on adjacent pairs after the q/k permute, Llama 3 rope factors, q/k/v/o biases, untied head
    "llama": ("LlamaForCausalLM", dict(BASE, num_hidden_layers=3, attention_bias=True, tie_word_embeddings=False,
                                       rope_theta=500000.0, rope_scaling=dict(rope_type="llama3", factor=8.0,
                                       low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=32))),
    # mistral: the llama graph, head size != hidden / heads, no biases, tied head
    "mistral": ("MistralForCausalLM", dict(BASE, num_hidden_layers=2, head_dim=32, sliding_window=None,
                                           tie_word_embeddings=True, rope_theta=1000000.0)),
    "qwen2": ("Qwen2ForCausalLM", dict(BASE, num_hidden_layers=3, tie_word_embeddings=True, rope_theta=1000000.0,
                                       use_sliding_window=False)),
    "qwen3": ("Qwen3ForCausalLM", dict(BASE, num_hidden_layers=3, head_dim=32, tie_word_embeddings=False,
                                       rope_theta=1000000.0)),
    # qwen35: 3 linear-attention layers + 1 full, 4 value heads over 2 key heads (the tiled reorder), partial rope
    "qwen35": ("Qwen3_5ForCausalLM", dict(BASE, num_hidden_layers=4, head_dim=32, tie_word_embeddings=False,
                                          rope_parameters=dict(rope_type="default", rope_theta=1000000.0,
                                                               partial_rotary_factor=0.25),
                                          full_attention_interval=4, linear_conv_kernel_dim=4, linear_key_head_dim=16,
                                          linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                                          layer_types=["linear_attention"] * 3 + ["full_attention"])),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_safetensors(path, tensors, dtypes=None):
    """tensors: name -> NumPy array (float32 / float16) or (dtype name, shape, raw uint8)"""
    hdr, blobs, off = {}, [], 0
    for n, a in tensors.items():
        if isinstance(a, tuple):
            dt, shape, raw = a
            raw = np.asarray(raw, np.uint8).tobytes()
        else:
            dt, shape, raw = {np.dtype("float32"): "F32", np.dtype("float16"): "F16"}[a.dtype], a.shape, a.tobytes()
        hdr[n] = dict(dtype=dt, shape=list(shape), data_offsets=[off, off + len(raw)])
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def write_bshj(path, tensors, align=4096):
    """tensors: name -> (dtype name "BF16"/"F16"/"F32", shape, raw uint8): a BSHJ 1.0 container, row-major, header order"""
    names = {"BF16": "bf16", "F16": "f16", "F32": "f32"}
    meta, off, blobs = {}, 0, []
    for n, (dt, shape, raw) in tensors.items():
        raw = np.asarray(raw, np.uint8).tobytes()
        meta[n] = dict(dtype=names[dt], shape=list(shape), offset=off, nbytes=len(raw), layout="row@1")
        pad = -len(raw) % align
        blobs.append(raw + b"\0" * pad)
        off += len(raw) + pad
    js = json.dumps(dict(tensors=meta, format="lossless copy")).encode()
    data_offset = 64 + len(js) + (-(64 + len(js)) % align)
    js += b" " * (data_offset - 64 - len(js))
    fixed = struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(js), data_offset, align, 1, off, 0, b"\0" * 16)
    with open(path, "wb") as f:
        f.write(fixed + js + b"".join(blobs))


Q8_0_REF = ("qwen3", "qwen35")
MULTIPLIED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
              "in_proj_b", "in_proj_a", "out_proj", "lm_head")


def q8_0_values(n, w, cfg):
    """dequantize(quantize(w, Q8_0)) in the GGUF layout: Qwen3.5's out_proj columns are value heads in tiled order there
    (mashbox/hf/adapter.py), so they are put in that order for the blocks and back after"""
    import gguf
    q = lambda a: gguf.quants.dequantize(gguf.quants.quantize(np.ascontiguousarray(a, np.float32),
                                                               gguf.GGMLQuantizationType.Q8_0), gguf.GGMLQuantizationType.Q8_0)
    if n.endswith("linear_attn.out_proj.weight"):
        HK, HV, DV = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
        perm = np.asarray([(j % HK) * (HV // HK) + j // HK for j in range(HV)])
        g = w.reshape(w.shape[0], HV, DV)[:, perm].reshape(w.shape)
        back = np.argsort(perm)
        return q(g).reshape(w.shape[0], HV, DV)[:, back].reshape(w.shape)
    return q(w)                       # row permutations (llama q/k, value-head rows) keep each row's blocks whole


def make(name):
    import torch
    import transformers
    cls, cfg = CONFIGS[name]
    torch.manual_seed(SEED)
    C = getattr(transformers, cls).config_class
    model = getattr(transformers, cls)(C(**cfg)).float().eval()
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("A_log"):
                continue
            if p.ndim >= 2:
                fan = p.shape[1] * (p.shape[2] if p.ndim > 2 else 1)
                p.copy_(torch.randn(p.shape, generator=g) / fan ** 0.5)
            else:      # norm weights near their usual value (1, or 0 for Qwen3.5's 1 + w norms), biases near 0
                one = "linear_attn.norm" in n or ("norm" in n and name != "qwen35")
                p.copy_(float(one) + 0.2 * torch.randn(p.shape, generator=g))
    d = os.path.join(FX, name)
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    toks = torch.randint(0, cfg["vocab_size"], (1, T), generator=g)
    with torch.no_grad():
        lg = model(toks).logits[0].float().numpy()
    lg.astype(np.float32).tofile(os.path.join(d, "logits.f32"))
    if name in Q8_0_REF:
        with torch.no_grad():
            for n, p in model.named_parameters():
                w = p.to(torch.bfloat16).float()
                if p.ndim == 2 and n.split(".")[-2] in MULTIPLIED and p.shape[1] % 32 == 0:
                    w = torch.from_numpy(q8_0_values(n, w.numpy(), cfg))
                p.copy_(w)
            model(toks).logits[0].float().numpy().astype(np.float32).tofile(os.path.join(d, "logits.q8_0.f32"))
    files = sorted(f for f in os.listdir(d) if f != "meta.json")
    meta = dict(name=name, architecture=cls, seed=SEED, tokens=toks[0].tolist(), n_vocab=cfg["vocab_size"],
                transformers=transformers.__version__, torch=torch.__version__,
                sha256={f: sha256(os.path.join(d, f)) for f in files})
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(name, meta["sha256"].get("model.safetensors"), lg.shape)


def make_chat(repo="Qwen/Qwen3-4B-Instruct-2507", rev="cdbee75f17c01a7cc42f958dc650907174af0554"):
    """chat-a2: a repo whose chat template is only in tokenizer_config.json, and transformers' renders of it"""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    d = os.path.join(FX, "chat-a2")
    os.makedirs(d, exist_ok=True)
    tc = json.load(open(hf_hub_download(repo, "tokenizer_config.json", revision=rev)))
    json.dump(dict(chat_template=tc["chat_template"]), open(os.path.join(d, "tokenizer_config.json"), "w"))
    convs = [[dict(role="user", content="Hello")],
             [dict(role="system", content="Be brief."), dict(role="user", content="What is 2+2?")],
             [dict(role="user", content="Hi"), dict(role="assistant", content="Hello! How can I help?"),
              dict(role="user", content="Name a prime.")]]
    renders = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    json.dump(dict(repo=repo, revision=rev, conversations=convs, renders=renders), open(os.path.join(d, "renders.json"), "w"),
              indent=1)
    import transformers
    json.dump(dict(name="chat-a2", transformers=transformers.__version__,
                   sha256={f: sha256(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f != "meta.json"}),
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    print("chat-a2", len(renders))


if __name__ == "__main__":
    if sys.argv[1:2] == ["chat"]:
        make_chat()
    elif sys.argv[1:2] == ["make"]:
        for n in sys.argv[2:] or list(CONFIGS):
            make(n)
