#!/usr/bin/env python3
"""Check a real GGUF file without a chip: the file's sha256, every tensor header, the architecture mapping, the
tokenizer and chat template, and a sample of every block type dequantized by mashbox.ops.reference against gguf-py, bit
for bit.

  python tools/gguf_check.py FILE.gguf [--rows 64] [--tensors-per-type 24] [--hf-dir DIR] [--llama-cpp DIR]
                             [--expect-sha256 HEX] [--out R.json]

Headers are checked two ways: mashbox's reader (which uses gguf-py's GGUFReader) against gguf-py's own tensor table, and
both against an independent parse of the GGUF header written here from the format spec (struct reads, no gguf-py), plus
each tensor's byte size from its type's block size, alignment, no overlaps and every tensor inside the file.

--hf-dir: the model's Hub files (tokenizer.json, chat_template.jinja, generation_config.json) to compare the tokenizer
built from the GGUF against. --llama-cpp: a llama.cpp checkout with a CPU build, to compare token ids with llama.cpp's
own tokenizer on the same file (vocab only, fast).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

TEXT = ("Hello world! It's 2026-10-28, the price is $1,234.56 (about 12%).\n\n"
        "def f(x):\n    return x ** 2  # square\n\tTabs and   spaces.\n"
        "Unicode: naïve café, Ünïcödé, 東京は晴れ, 你好世界, Привет мир, مرحبا, emoji 🚀🔥 and é (combining).\n"
        "I'll we've they're DON'T it'S; numbers 3.14159 1e-6 0x1F 1000000.\n<|im_start|>user\nhi<|im_end|>\n")


VT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def parse_header(path):
    """Independent GGUF v2/v3 header parse (llama.cpp gguf.md): -> (version, alignment, data_start, {name: (type, shape
    row-major, absolute offset)}). Values of the metadata are skipped except general.alignment."""
    import mmap
    import struct
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, m, pos)[0]
        pos += struct.calcsize(fmt)
        return v

    def rstr():
        nonlocal pos
        n = rd("<Q")
        b = bytes(m[pos:pos + n])
        pos += n
        return b.decode("utf-8")

    def skip(vt):
        nonlocal pos
        if vt == 8:
            n = rd("<Q")                        # (not `pos += rd(...)`: that would read pos before rd moves it)
            pos += n
        elif vt == 9:
            it, n = rd("<I"), rd("<Q")
            if it in VT:
                pos += n * struct.calcsize(VT[it])
            else:
                for _ in range(n):
                    skip(it)
        else:
            pos += struct.calcsize(VT[vt])
    assert m[:4] == b"GGUF", "not a GGUF file"
    pos = 4
    ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    align = 32
    for _ in range(nkv):
        k, vt = rstr(), rd("<I")
        if k == "general.alignment":
            align = rd(VT[vt])
        else:
            skip(vt)
    infos = []
    for _ in range(nt):
        name, nd = rstr(), rd("<I")
        dims = [rd("<Q") for _ in range(nd)]
        infos.append((name, rd("<I"), tuple(reversed(dims)), rd("<Q")))
    start = -(-pos // align) * align
    m.close()
    return ver, align, start, {n: (t, sh, start + off) for n, t, sh, off in infos}


def first_diff(a, b):
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if k is None and len(a) != len(b):
        k = min(len(a), len(b))
    return None if k is None else dict(at=k, a=a[k:k + 6], b=b[k:k + 6])


CONVS = [
    [dict(role="user", content="Hi")],
    [dict(role="system", content="You are terse."), dict(role="user", content="2+2?")],
    [dict(role="user", content="Name a colour."), dict(role="assistant", content="Blue."), dict(role="user", content="Another.")],
    [dict(role="system", content="S"), dict(role="user", content="a"), dict(role="assistant", content="b"),
     dict(role="user", content="café 東京 🚀")],
]


def render_check(ours, hub):
    """the file's chat template vs the Hub's, rendered the way transformers does (sandboxed jinja2, trim/lstrip blocks)
    on 4 conversations x add_generation_prompt on/off: -> dict(renders, identical, first_difference)"""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, **k: json.dumps(x, indent=indent, ensure_ascii=False)
    to, th = env.from_string(ours), env.from_string(hub)
    same, n, diff = 0, 0, None
    for conv in CONVS:
        for agp in (True, False):
            x, y = to.render(messages=conv, add_generation_prompt=agp), th.render(messages=conv, add_generation_prompt=agp)
            n += 1
            same += x == y
            if x != y and diff is None:
                diff = first_diff(list(x), list(y))
    return dict(renders=n, identical=same, first_difference=diff)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--tensors-per-type", type=int, default=24)
    ap.add_argument("--hf-dir")
    ap.add_argument("--llama-cpp")
    ap.add_argument("--expect-sha256", help="the Hub's LFS oid for this file (recorded and compared)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import gguf
    import jax
    import jax.numpy as jnp
    from mashbox.gguf import File, inspect
    from mashbox.gguf.archs import qwen35 as A
    from mashbox.gguf import tokenizer as GT
    from mashbox.ops import reference as R
    import importlib.metadata
    t0 = time.time()
    rec = dict(file=os.path.abspath(a.file), bytes=os.path.getsize(a.file), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               gguf=importlib.metadata.version("gguf"), jax=jax.__version__, backend=jax.default_backend())
    h = hashlib.sha256()
    with open(a.file, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    rec["sha256"] = h.hexdigest()
    if a.expect_sha256:
        rec["expect_sha256"] = a.expect_sha256
        rec["sha256_equal"] = rec["sha256"] == a.expect_sha256.lower()
    f = File(a.file)
    g = gguf.GGUFReader(a.file)
    size = os.path.getsize(a.file)
    # 1. headers: ours == gguf-py's, every tensor's bytes == rows x row bytes of its type, inside the file
    bad = []
    names = {t.name: t for t in g.tensors}
    for n, t in f.tensors.items():
        r = names.get(n)
        shape = tuple(reversed([int(x) for x in r.shape]))
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        want = rows * R.row_bytes(int(r.tensor_type), shape[-1])
        if (int(t.type), tuple(t.shape), t.nbytes) != (int(r.tensor_type), shape, int(r.n_bytes)) or t.nbytes != want \
                or r.data_offset + r.n_bytes > size:
            bad.append(n)
    # independent parse: same names, types, shapes and offsets as gguf-py; sizes from the block table; aligned, in the
    # file, no overlaps
    ver, align, start, ind = parse_header(a.file)
    ibad = sorted(set(ind) ^ set(f.tensors))
    spans = []
    for n, (ty, sh, off) in ind.items():
        if n not in f.tensors or n not in names:
            continue
        t, r = f.tensors[n], names[n]
        rows = int(np.prod(sh[:-1])) if len(sh) > 1 else 1
        nb = rows * R.row_bytes(ty, sh[-1])
        if (ty, sh) != (int(t.type), tuple(t.shape)) or off != int(r.data_offset) or nb != t.nbytes or off % align \
                or off + nb > size:
            ibad.append(n)
        spans.append((off, off + nb, n))
    spans.sort()
    overlaps = [b[2] for a_, b in zip(spans, spans[1:]) if b[0] < a_[1]]
    rec["headers"] = dict(tensors=len(f.tensors), gguf_py_tensors=len(g.tensors), mismatches=bad[:20], n_bad=len(bad),
                          metadata_keys=len(f.keys()),
                          independent_parse=dict(gguf_version=ver, alignment=align, data_start=start, tensors=len(ind),
                                                 n_bad=len(ibad), mismatches=ibad[:20], overlaps=overlaps[:20]))
    census, total = f.census()
    rec["census"] = {k: dict(tensors=v["tensors"], gb=round(v["bytes"] / 1e9, 3)) for k, v in census.items()}
    # 2. architecture mapping: every tensor the arch reads exists, no tensor is left unread
    c = A.Config(f)
    used = set()
    for l in range(c.L_all):
        b = f"blk.{l}."
        used |= {b + x for x in ("attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                                 "ffn_down.weight")}
        if l < c.L and c.types[l] == "linear_attention":
            used |= {b + x for x in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
                                     "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ssm_out.weight")}
        else:
            used |= {b + x for x in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                     "attn_q_norm.weight", "attn_k_norm.weight")}
    used |= {"token_embd.weight", "output_norm.weight"} | (set() if c.tie else {"output.weight"})
    mtp = A.mtp_tensors(f, c) if c.nextn else {}
    have = set(f.tensors)
    rec["arch"] = dict(arch=f.arch, layers=c.L, nextn=c.nextn, L_all=c.L_all, D=c.D, F=c.F, H=c.H, KV=c.KV, HD=c.HD,
                       HK=c.HK, HV=c.HV, DK=c.DK, DV=c.DV, FI=c.FI, V=c.V, tie=c.tie, eps=c.eps, theta=c.theta,
                       missing=sorted(used - have)[:20],
                       unread=sorted(have - used - {f"blk.{c.L + i}.{k}" for i in range(c.nextn) for k in mtp})[:20],
                       mtp_roles=sorted(mtp))
    # 3. dequant sample: per type, up to N tensors, `rows` random rows each (small tensors whole), bit for bit
    rng = np.random.default_rng(0)
    by_type = {}
    for n, t in f.tensors.items():
        by_type.setdefault(int(t.type), []).append(n)
    dq = {}
    fn = {}
    for ty, ns in sorted(by_type.items()):
        pick = [ns[i] for i in sorted(rng.choice(len(ns), min(len(ns), a.tensors_per_type), replace=False))]
        mism = vals = 0
        for n in pick:
            r = names[n]
            shape = tuple(reversed([int(x) for x in r.shape]))
            cols = shape[-1]
            raw = np.asarray(f.tensors[n].raw).view(np.uint8).reshape(-1, R.row_bytes(ty, cols))
            idx = np.sort(rng.choice(raw.shape[0], min(raw.shape[0], a.rows), replace=False))
            blk = np.ascontiguousarray(raw[idx])
            want = gguf.quants.dequantize(blk, gguf.GGMLQuantizationType(ty)).reshape(len(idx), cols)
            if (ty, cols) not in fn:
                fn[(ty, cols)] = jax.jit(lambda x, ty=ty, cols=cols: R.dequant(x, ty, cols))
            got = np.asarray(fn[(ty, cols)](jnp.asarray(blk)))
            nan = np.isnan(got) | np.isnan(want)
            same = np.where(nan, np.isnan(got) & np.isnan(want),
                            got.view(np.uint32) == np.asarray(want, np.float32).view(np.uint32))
            mism += int((~same).sum())
            vals += same.size
        dq[R.TYPES[ty][0]] = dict(tensors_checked=len(pick), of=len(ns), values=vals, mismatches=mism)
    rec["dequant_vs_gguf_py"] = dq
    # 4. tokenizer, chat template, stop ids from the file vs the Hub files and llama.cpp
    fs = GT.files_dict(f, a.file)
    from tokenizers import Tokenizer
    ours = Tokenizer.from_str(fs["tokenizer.json"])
    ids = ours.encode(TEXT, add_special_tokens=False).ids
    import unicodedata
    nfc = unicodedata.normalize("NFC", TEXT)
    tk = dict(vocab=ours.get_vocab_size(), ids=len(ids), text_is_nfc=TEXT == nfc,
              roundtrip=ours.decode(ids, skip_special_tokens=False) == TEXT,
              roundtrip_nfc_text=ours.decode(ours.encode(nfc, add_special_tokens=False).ids, skip_special_tokens=False) == nfc,
              roundtrip_note="the tokenizer normalizes to NFC (as the Hub's), so a non-NFC text cannot round-trip",
              stop_ids=json.loads(fs["generation_config.json"]).get("eos_token_id"))
    if a.hf_dir:
        up = Tokenizer.from_file(os.path.join(a.hf_dir, "tokenizer.json"))
        uids = up.encode(TEXT, add_special_tokens=False).ids
        tk["hub_tokenizer_same_ids"] = uids == ids
        tk["hub_first_difference"] = first_diff(ids, uids)
        tk["hub_vocab"] = up.get_vocab_size()
        tk["chat_template_same_as_hub"] = open(os.path.join(a.hf_dir, "chat_template.jinja")).read().strip() == fs["chat_template.jinja"].strip()
        tk["hub_stop_ids"] = json.load(open(os.path.join(a.hf_dir, "generation_config.json"))).get("eos_token_id")
        tk["hub_roundtrip"] = up.decode(uids, skip_special_tokens=False) == TEXT
        tk["chat_template_renders"] = render_check(fs["chat_template.jinja"],
                                                   open(os.path.join(a.hf_dir, "chat_template.jinja")).read())
    if a.llama_cpp:
        root = os.path.abspath(a.llama_cpp)
        bindir = os.path.join(root, "build", "bin")
        with tempfile.TemporaryDirectory() as td:
            exe = os.path.join(td, "ll")
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_fixtures", "llama_logits.cpp")
            subprocess.run(["g++", "-O2", "-std=c++17", src, f"-I{root}/include", f"-I{root}/ggml/include", f"-L{bindir}",
                            "-lllama", f"-Wl,-rpath,{bindir}", "-o", exe], check=True)
            tf = os.path.join(td, "text.txt")
            import unicodedata
            for tag, text in (("", TEXT), ("_nfc_text", unicodedata.normalize("NFC", TEXT))):
                open(tf, "w", encoding="utf-8").write(text)
                out = subprocess.run([exe, "--tokenize", a.file, tf], check=True, capture_output=True, text=True).stdout
                lids = [int(x) for x in out.split()]
                oids = ours.encode(text, add_special_tokens=False).ids
                tk["llama_cpp_same_ids" + tag] = lids == oids
                tk["llama_cpp_first_difference" + tag] = first_diff(oids, lids)
    rec["tokenizer"] = tk
    rec["inspect"] = {k: v for k, v in inspect(a.file).items() if k in ("runs_here", "plan", "warning", "mtp_layers",
                                                                        "chip_gb_estimate", "chips_that_fit",
                                                                        "chip_estimate_ctx2048_slots8", "why")}
    rec["seconds"] = round(time.time() - t0, 1)
    ok = (rec["headers"]["n_bad"] == 0 and rec["headers"]["independent_parse"]["n_bad"] == 0
          and not rec["headers"]["independent_parse"]["overlaps"] and not rec["arch"]["missing"] and not rec["arch"]["unread"]
          and all(v["mismatches"] == 0 for v in dq.values()) and rec.get("sha256_equal", True))
    rec["ok"] = ok
    s = json.dumps(rec, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
