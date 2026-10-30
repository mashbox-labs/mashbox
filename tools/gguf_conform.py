#!/usr/bin/env python3
"""One-command GGUF conformance: does a GGUF file run on this engine, and does it compute what llama.cpp computes?
No chip. Modeled on llama.cpp's own tests (test-tokenizer-0, test-quantize-fns, logits checks).

  python tools/gguf_conform.py REPO[:TAG] --llama-cpp DIR --out RECORDS_DIR [--dl DIR] [--n-gen 16] [--keep]
  python tools/gguf_conform.py FILE.gguf  --llama-cpp DIR --out RECORDS_DIR
  python tools/gguf_conform.py --quant all --llama-cpp DIR --out RECORDS_DIR      # (d) once per block type

Per file (one sealed record folder: RECORDS_DIR/<repo>__<file>/):
  (a) probe       the engine's own loader on the file: open_model, the MoE hold and the tokenizer build. Supported, or the
                  exact refusal line, and the run stops there.
  (b) tokenizer   our token ids vs llama.cpp's (llama_logits --tokenize, the pinned build) on TEXTS, raw and NFC.
  (c) cpu         llama.cpp's greedy continuation of PROMPTS on CPU (llama_greedy) and its logits at every position;
                  our engine's forward on CPU (float32 activations, every block type on the exact reference path), the
                  prompt as one prefill and the last DECODE_TAIL tokens as one-token decode steps on the K/V cache,
                  teacher-forced on llama.cpp's sequence: top-1 agreement at every position, mean KL(llama.cpp || ours),
                  NMSE, and the greedy first difference (near tie = llama.cpp's top-two gap < NEAR_TIE_GAP logit).
  (d) quant       the record names the per-type quant records (--quant): ggml's own quantize + to_float on random
                  rows vs our dequant (mashbox.ops.reference) and gguf-py, so a block type is proven once, not per file.
  (e) tpu         never here: the record carries tpu_key = arch x quant family; ONE chip smoke per key gives the speed.
The bars are fixed before any run (BARS) and copied from tests/cpu/test_gguf_dense.py (KL_MAX, TOP1_MIN of 16) and the
catalog's NEAR_TIE_GAP. Every record folder ends with SHA256SUMS over its files; the record names the file's sha256, the
code id (sha256 of CODE_FILES) and the llama.cpp commit.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "src")]

BARS = dict(identical_weights_nmse_max=1e-4, top1_min_share=15 / 16, near_tie_gap=0.5, other_divergences_max=0,
            tokenizer="every NFC text equal, except differences explained by llama.cpp's control-looking token override "
                      "(the ids llama.cpp itself warns about when it loads the file); those are counted and listed",
            block_path_kl="recorded, not gated (llama.cpp's CPU block path quantizes activations to Q8 for its dot "
                          "products; KL against it measures that, not this engine)",
            near_tie="at the first greedy difference, llama.cpp's logit of its own pick minus its logit of our pick is "
                     "< near_tie_gap (0.5 logit, the catalog's NEAR_TIE_GAP); any other difference is 'other'",
            dequant_vs_gguf_py="bit-exact", dequant_vs_ggml_max_ulp=1,
            verdict="cpu-checked = probe supported, tokenizer gate, identical-weights NMSE <= 1e-4, block-path top-1 >= "
                    "15/16 of positions, 0 other greedy divergences, every quantized type's quant record ok",
            locked="2026-10-29T22:04Z, MAIN approved; dev file unsloth/Qwen3-0.6B-GGUF Q8_0 (seen before the lock); "
                   "every later file is held out: no bar or definition changes after seeing it")
TEXTS = [
    "Hello world! It's 2026-10-28, the price is $1,234.56 (about 12%).",
    "def f(x):\n    return x ** 2  # square\n\tTabs and   spaces.\n\n\n",
    "Unicode: naïve café, Ünïcödé, 東京は晴れ, 你好世界, Привет мир, مرحبا, नमस्ते, 안녕하세요, emoji 🚀🔥👍🏽",
    "I'll we've they're DON'T it'S; numbers 3.14159 1e-6 0x1F 1000000 123456789 ٣٤٥",
    "   leading spaces, trailing spaces   \n \n\t\n",
    "<|im_start|>user\nhi<|im_end|>\n<start_of_turn>user\nhi<end_of_turn>\n[INST] hi [/INST]</s><|eot_id|>",
    "The quick brown fox jumps over the lazy dog. " * 6,
    "é (decomposed accent) and é (composed); Å vs Å; ﬁ ligature; full-width ＡＢＣ",
]
PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Translate to German: The weather is nice today and we are going to the park.\nGerman:",
    "Question: What is 17 multiplied by 23? Think step by step.\nAnswer:",
]
N_GEN, DECODE_TAIL = 16, 4
CODE_FILES = ["src/mashbox/gguf/*.py", "src/mashbox/gguf/archs/*.py", "src/mashbox/ops/*.py", "tools/gguf_conform.py",
              "tools/gguf_scan.py", "tools/gguf_fixtures/llama_greedy.cpp", "tools/gguf_fixtures/llama_logits.cpp",
              "tools/gguf_fixtures/ggml_dequant.c"]
UNQ = ("F32", "F16", "BF16")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_id():
    files = sorted({f for g in CODE_FILES for f in glob.glob(os.path.join(ROOT, g))})
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, ROOT).encode() + b"\0" + open(f, "rb").read() + b"\0")
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    return dict(sha256=h.hexdigest(), files=len(files), git_head=head or None)


def tools(llama, work):
    """build the three reference tools against the pinned llama.cpp build (cached by source sha256)"""
    bindir = os.path.join(llama, "build", "bin")
    out = {}
    os.makedirs(work, exist_ok=True)
    for name, cc in (("llama_greedy.cpp", "g++"), ("llama_logits.cpp", "g++"), ("ggml_dequant.c", "gcc")):
        src = os.path.join(HERE, "gguf_fixtures", name)
        exe = os.path.join(work, name.split(".")[0] + "-" + sha_file(src)[:12])
        if not os.path.exists(exe):
            std = ["-std=c++17"] if cc == "g++" else []
            libs = ["-lllama"] if cc == "g++" else ["-lggml", "-lggml-base", "-lggml-cpu"]
            subprocess.run([cc, "-O2", *std, src, f"-I{llama}/include", f"-I{llama}/ggml/include", f"-L{bindir}", *libs,
                            f"-Wl,-rpath,{bindir}", "-o", exe], check=True)
        out[name.split(".")[0]] = exe
    commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return out, commit


def fetch(spec, dl):
    """REPO[:TAG] -> (local primary path, dict of the pin: repo, revision, file, lfs sha256, bytes, card, seconds)"""
    import urllib.request
    from huggingface_hub import hf_hub_download
    from mashbox.gguf import resolve as GR
    import gguf_scan as S
    repo, tag = (spec.split(":", 1) + [None])[:2] if ":" in spec else (spec, None)
    info = json.loads(S.get(f"{S.HF}/api/models/{repo}?blobs=true"))
    sib = {s["rfilename"]: s for s in info.get("siblings", [])}
    path = GR.pick([p for p in sib if p.endswith(".gguf")], tag)
    if not path:
        raise SystemExit(f"{spec}: no GGUF model file for tag {tag}")
    t0 = time.time()
    local = None
    for p in GR.shards(path):
        lp = hf_hub_download(repo, p, revision=info["sha"], local_dir=os.path.join(dl, repo.replace("/", "__")))
        local = local or lp
    pin = dict(repo=repo, revision=info["sha"], file=path, tag=tag, lfs_sha256=(sib[path].get("lfs") or {}).get("sha256"),
               bytes=sib[path].get("size"), shards=len(GR.shards(path)), gated=info.get("gated"),
               card={"id": info.get("id"), "sha": info.get("sha"), "cardData": info.get("cardData") or {}},
               download_s=round(time.time() - t0, 1))
    return local, pin


def probe(path):
    import gguf_scan as S
    from gguf import GGUFReader
    off = int(GGUFReader(path, "r").data_offset)
    with open(path, "rb") as fh:
        hdr = fh.read(off)
    return S.probe(hdr, os.path.getsize(path), path), hdr


def first_diff(a, b):
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if k is None and len(a) != len(b):
        k = min(len(a), len(b))
    return None if k is None else dict(at=k, ours=a[k:k + 5], llama_cpp=b[k:k + 5])


def tokenizer_check(path, f, exe, work):
    from tokenizers import Tokenizer
    from mashbox.gguf import tokenizer as GT
    ours = Tokenizer.from_str(GT.files_dict(f, path)["tokenizer.json"])
    rows, override = [], {}
    for i, text in enumerate(TEXTS):
        for form, t in (("raw", text), ("nfc", unicodedata.normalize("NFC", text))):
            tf = os.path.join(work, f"text{i}-{form}.txt")
            open(tf, "w", encoding="utf-8").write(t)
            pr = subprocess.run([exe, "--tokenize", path, tf], check=True, capture_output=True, text=True)
            lids = [int(x) for x in pr.stdout.split()]
            for m in re.finditer(r"control-looking token:\s+(\d+) '([^']*)'", pr.stderr):
                override[int(m.group(1))] = m.group(2)
            oids = ours.encode(t, add_special_tokens=False).ids
            fd = first_diff(oids, lids)
            rows.append(dict(text=i, form=form, n=len(lids), equal=oids == lids, first_difference=fd,
                             explained=bool(fd) and any(x in override for x in lids[fd["at"]:fd["at"] + 5])))
    nfc = [r for r in rows if r["form"] == "nfc"]
    return dict(texts=len(TEXTS), nfc_equal=sum(r["equal"] for r in nfc), raw_equal=sum(r["equal"] for r in rows if r["form"] == "raw"),
                nfc_explained=sum(r["explained"] for r in nfc), llama_cpp_override={str(k): v for k, v in override.items()},
                ok=all(r["equal"] or r["explained"] for r in nfc), rows=rows,
                note="the engine tokenizes with the model's NFC normalization (docs/gguf.md); llama.cpp does not, so the "
                     "gate is the NFC form and the raw form is recorded")


def kl_rows(ref, ours):
    p = np.exp(ref - ref.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    lq = ours - ours.max(-1, keepdims=True)
    lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    return np.sum(p * (np.log(np.maximum(p, 1e-300)) - lq), -1)


def f32_copy(src, dst):
    """an all-F32 GGUF holding exactly gguf-py's dequantization of every tensor of src, with src's metadata (the
    identical-weights reference of tests/cpu/test_gguf_dense.py, for a real file): llama.cpp on it multiplies float32
    weights by float32 activations, so it takes llama.cpp's activation quantization out of the comparison"""
    import gguf
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, arch=r.fields["general.architecture"].contents())
    for fd in r.fields.values():
        if fd.name == "general.architecture" or fd.name.startswith("GGUF."):
            continue
        vt = fd.types[0]
        w.add_key_value(fd.name, fd.contents(), vt, sub_type=fd.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    for t in r.tensors:
        shape = tuple(reversed([int(x) for x in t.shape]))
        w.add_tensor(t.name, np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(shape))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def cpu_check(path, exe, work, n_gen, exe_logits=None, f32_max_gb=60.0):
    import jax
    import jax.numpy as jnp
    from mashbox.gguf import open_model
    t0 = time.time()
    tfs = []
    for i, p in enumerate(PROMPTS):
        tf = os.path.join(work, f"prompt{i}.txt")
        open(tf, "w", encoding="utf-8").write(p)
        tfs.append(tf)
    pre = os.path.join(work, "llama-")
    subprocess.run([exe, path, str(n_gen), pre, *tfs], check=True, capture_output=True)
    seqs, refs, nps = [], [], []
    for i in range(len(PROMPTS)):
        a, b = open(f"{pre}{i}.ids").read().split("\n\n")
        pi, gi = [int(x) for x in a.split()], [int(x) for x in b.split()]
        seqs.append(pi + gi)
        nps.append(len(pi))
        refs.append(np.fromfile(f"{pre}{i}.f32", np.float32))
    t_llama = time.time() - t0
    t1 = time.time()
    f, A, c = open_model(path, act=jnp.float32)
    W, info = A.load(f, c, use_kernels=False)
    V = refs[0].size // (len(seqs[0]) - 1)
    fwd = jax.jit(A.make_forward(c), static_argnames=("Tc", "contig", "all_logits"))
    S = len(seqs)
    L = [len(s) for s in seqs]
    tpre = [n - DECODE_TAIL for n in L]
    CTX = -(-(max(L) + 8) // 32) * 32
    ck, cv = A.new_state(c, S, CTX)
    i32 = jnp.int32
    toks = np.zeros((S, max(tpre)), np.int32)
    for i, s in enumerate(seqs):
        toks[i, :tpre[i]] = s[:tpre[i]]
    slots = jnp.arange(S, dtype=i32)
    r = fwd(W, ck, cv, jnp.asarray(toks), jnp.zeros((S,), i32), jnp.asarray(tpre, i32), slots, 0, all_logits=True)
    lg = np.asarray(r["lg"], np.float32)
    ours = [[lg[i, :tpre[i]]] for i in range(S)]
    ck, cv = r["ck"], r["cv"]
    for j in range(DECODE_TAIL):
        t = jnp.asarray([[s[tpre[i] + j]] for i, s in enumerate(seqs)], i32)
        r = fwd(W, ck, cv, t, jnp.asarray([p + j for p in tpre], i32), jnp.ones((S,), i32), slots, CTX, all_logits=True)
        ck, cv = r["ck"], r["cv"]
        d = np.asarray(r["lg"], np.float32)
        for i in range(S):
            ours[i].append(d[i, :1])
    t_ours = time.time() - t1
    rows, kl_all, top1, npos, num, den = [], [], 0, 0, 0.0, 0.0
    for i, s in enumerate(seqs):
        ref = refs[i].reshape(len(s) - 1, V)
        o = np.concatenate(ours[i], 0)[:len(s) - 1, :V]
        kl = kl_rows(ref.astype(np.float64), o.astype(np.float64))
        kl_all.append(kl)
        agree = ref.argmax(-1) == o.argmax(-1)
        top1 += int(agree.sum())
        npos += len(agree)
        num += float(np.sum((o.astype(np.float64) - ref) ** 2))
        den += float(np.sum(ref.astype(np.float64) ** 2))
        g0 = nps[i] - 1
        gen = s[nps[i]:]
        mine = o[g0:g0 + len(gen)].argmax(-1).tolist()
        k = next((j for j, (x, y) in enumerate(zip(mine, gen)) if x != y), None)
        gap = None if k is None else float(ref[g0 + k, gen[k]] - ref[g0 + k, mine[k]])
        rows.append(dict(i=i, n_prompt=nps[i], n=len(gen), match=len(gen) if k is None else k, first_diff=k, ref_gap=gap,
                         near_tie=bool(k is not None and gap < BARS["near_tie_gap"]), top1=int(agree.sum()),
                         positions=len(agree), kl_mean=float(kl.mean()), kl_max=float(kl.max())))
    same = None
    params = sum(int(np.prod(t.shape)) for t in f.tensors.values())
    if exe_logits and params * 4 / 1e9 <= f32_max_gb:
        tf = time.time()
        fp = os.path.join(work, "f32-copy.gguf")
        f32_copy(path, fp)
        sn, sd, st, sp = 0.0, 0.0, 0, 0
        per = []
        for i, s in enumerate(seqs):
            out = os.path.join(work, f"f32ref{i}.f32")
            subprocess.run([exe_logits, fp, out, *map(str, s)], check=True, capture_output=True)
            ref = np.fromfile(out, np.float32).reshape(len(s), -1)[:len(s) - 1].astype(np.float64)
            o = np.concatenate(ours[i], 0)[:len(s) - 1, :ref.shape[1]].astype(np.float64)
            n_, d_ = float(np.sum((o - ref) ** 2)), float(np.sum(ref ** 2))
            sn, sd = sn + n_, sd + d_
            a_ = int((o.argmax(-1) == ref.argmax(-1)).sum())
            st, sp = st + a_, sp + len(ref)
            per.append(dict(i=i, nmse=n_ / d_, top1=a_, positions=len(ref), kl_mean=float(kl_rows(ref, o).mean())))
            os.remove(out)                          # (our own scratch, written above)
        os.remove(fp)
        same = dict(method="llama.cpp CPU on an all-F32 copy holding gguf-py's dequantization of every tensor (the "
                           "identical-weights reference of tests/cpu/test_gguf_dense.py), same sequences",
                    nmse=sn / sd, nmse_max=BARS["identical_weights_nmse_max"], top1_agree=st, positions=sp, rows=per,
                    ok=sn / sd <= BARS["identical_weights_nmse_max"],
                    seconds=round(time.time() - tf, 1))
    full = sum(r["first_diff"] is None for r in rows)
    near = sum(r["near_tie"] for r in rows)
    kl_mean = float(np.concatenate(kl_all).mean())
    out = dict(method="llama.cpp greedy on CPU; ours teacher-forced on llama.cpp's sequence: prefill + "
                      f"{DECODE_TAIL} decode steps on the K/V cache, float32 activations, exact reference path",
               prompts=len(rows), n_gen=n_gen, positions=npos, top1_agree=top1, top1_share=round(top1 / npos, 4),
               kl_mean=kl_mean, nmse=num / den, full_match=full, near_tie_divergences=near,
               other_divergences=len(rows) - full - near, rows=rows, llama_s=round(t_llama, 1), ours_s=round(t_ours, 1),
               reference_path_types=sorted({t.type_name for t in f.tensors.values()} - set(UNQ)), layers=c.L,
               identical_weights=same)
    out["ok"] = bool(same and same["ok"] and top1 / npos >= BARS["top1_min_share"]
                     and out["other_divergences"] <= BARS["other_divergences_max"])
    if same is None:
        out["not_checked"] = "identical-weights reference not run: the F32 copy would be over f32_max_gb"
    return out, f


def quant_check(types, exe, out):
    """(d): per block type, ggml's quantize + to_float on 64 random rows x 4096 vs our dequant and gguf-py's"""
    import gguf
    import jax
    import jax.numpy as jnp
    from mashbox.ops import reference as R
    res = {}
    for ty, (name, *_r) in sorted(R.TYPES.items()):
        if types != ["all"] and name not in types:
            continue
        pre = os.path.join(out, f"q-{name}")
        p = subprocess.run([exe, str(ty), "64", "4096", "20261029", pre], capture_output=True, text=True)
        if p.returncode:
            res[name] = dict(ggml="cannot quantize this type: " + p.stderr.strip()[:200])
            continue
        rb = int(p.stdout.split()[1])
        q = np.fromfile(pre + ".q", np.uint8).reshape(64, rb)
        want = np.fromfile(pre + ".f32", np.float32).reshape(64, 4096)
        got = np.asarray(jax.jit(lambda x: R.dequant(x, ty, 4096))(jnp.asarray(q)), np.float32)
        gp = np.asarray(gguf.quants.dequantize(q, gguf.GGMLQuantizationType(ty)), np.float32).reshape(64, 4096)
        ulp = np.abs(got.view(np.int32).astype(np.int64) - want.view(np.int32).astype(np.int64))
        res[name] = dict(values=int(got.size), vs_ggml_bit_equal=int((got.view(np.uint32) == want.view(np.uint32)).sum()),
                         vs_ggml_max_ulp=int(ulp.max()), vs_ggml_max_abs=float(np.abs(got - want).max()),
                         vs_gguf_py_bit_equal=int((got.view(np.uint32) == gp.view(np.uint32)).sum()),
                         ok=bool(ulp.max() <= BARS["dequant_vs_ggml_max_ulp"]
                                 and (got.view(np.uint32) == gp.view(np.uint32)).all()))
        for x in (".q", ".f32"):
            os.remove(pre + x)          # (our own scratch, written above)
    return res


def seal(d):
    lines = [f"{sha_file(os.path.join(d, n))}  {n}" for n in sorted(os.listdir(d)) if n != "SHA256SUMS"]
    open(os.path.join(d, "SHA256SUMS"), "w").write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dl", default=None)
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--quant")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded file (default: delete it after the seal)")
    a = ap.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", False)
    t0 = time.time()
    exes, lcommit = tools(os.path.abspath(a.llama_cpp), os.path.join(a.out, ".bin"))
    cid = code_id()
    import importlib.metadata as md
    versions = dict(gguf=md.version("gguf"), jax=jax.__version__, tokenizers=md.version("tokenizers"), llama_cpp=lcommit)
    if a.quant:
        d = os.path.join(a.out, "quant")
        os.makedirs(d, exist_ok=True)
        res = quant_check(a.quant.split(","), exes["ggml_dequant"], d)
        rec = dict(kind="gguf-quant", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code=cid, versions=versions,
                   bars=BARS, rows="64 x 4096, seed 20261029", types=res, seconds=round(time.time() - t0, 1))
        json.dump(rec, open(os.path.join(d, "quant.json"), "w"), indent=1)
        seal(d)
        print(json.dumps({k: v.get("ok") for k, v in res.items()}))
        return 0 if all(v.get("ok") for v in res.values()) else 1
    dl = a.dl or os.path.join(a.out, ".dl")
    downloaded = not os.path.exists(a.spec)
    if downloaded:
        path, pin = fetch(a.spec, dl)
    else:
        path, pin = a.spec, dict(file=os.path.basename(a.spec), download_s=0)
    name = (pin.get("repo", "local").replace("/", "__") + "__" + os.path.basename(pin["file"]))
    d = os.path.join(a.out, name)
    os.makedirs(d, exist_ok=True)
    work = os.path.join(a.out, ".work", name)
    os.makedirs(work, exist_ok=True)
    rec = dict(kind="gguf-conformance", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pin={k: v for k, v in pin.items() if k != "card"},
               code=cid, versions=versions, bars=BARS, host=os.uname().nodename)
    ts = time.time()
    rec["sha256"] = sha_file(path)
    rec["bytes"] = os.path.getsize(path)
    rec["sha256_equal_hub"] = rec["sha256"] == pin.get("lfs_sha256") if pin.get("lfs_sha256") else None
    rec["sha256_s"] = round(time.time() - ts, 1)
    if "card" in pin:
        json.dump(pin["card"], open(os.path.join(d, "card.json"), "w"), indent=1)
    tp = time.time()
    pr, hdr = probe(path)
    open(os.path.join(d, "header.bin"), "wb").write(hdr)
    from mashbox.gguf import File
    f = File(path)
    census, total = f.census()
    quants = sorted(k for k in census if k not in UNQ)
    arch = f.arch
    moe = int(f.get(f"{arch}.expert_count", 0) or 0) > 0
    rec["a_probe"] = dict(pr, seconds=round(time.time() - tp, 1))
    rec.update(arch=arch, moe=moe, types=sorted(census), tpu_key=f"{arch} x {'+'.join(quants) or 'float'}",
               ctx_train=f.get(f"{arch}.context_length"), tokenizer=dict(model=f.get("tokenizer.ggml.model"),
                                                                        pre=f.get("tokenizer.ggml.pre")))
    if pr.get("supported"):
        tt = time.time()
        rec["b_tokenizer"] = dict(tokenizer_check(path, f, exes["llama_logits"], work), seconds=0)
        rec["b_tokenizer"]["seconds"] = round(time.time() - tt, 1)
        tc = time.time()
        try:
            cpu, _ = cpu_check(path, exes["llama_greedy"], work, a.n_gen, exe_logits=exes["llama_logits"])
        except Exception as e:      # a crash in our forward is a failure of this file, recorded, never a pass
            cpu = dict(ok=False, error=f"{type(e).__name__}: {e}"[:600])
        cpu["seconds"] = round(time.time() - tc, 1)
        rec["c_cpu"] = cpu
        qrec = os.path.join(a.out, "quant", "quant.json")
        q = json.load(open(qrec)) if os.path.exists(qrec) else {}
        rec["d_quant"] = dict(record="quant/quant.json" if q else None, record_sha256=sha_file(qrec) if q else None,
                              types={t: (q.get("types", {}).get(t) or {}).get("ok") for t in quants})
        rec["e_tpu"] = dict(key=rec["tpu_key"], run=False, note="ONE chip smoke per arch x quant family, never per file")
        rec["verdict"] = ("cpu-checked" if rec["b_tokenizer"]["ok"] and cpu.get("ok") and all(rec["d_quant"]["types"].values())
                          else "failed")
    else:
        rec["verdict"] = "blocked"
        rec["blocked_by"] = pr.get("why")
    rec["seconds_total"] = round(time.time() - t0, 1)
    json.dump(rec, open(os.path.join(d, "record.json"), "w"), indent=1, ensure_ascii=False)
    seal(d)
    if downloaded and not a.keep:       # only the file this run downloaded, after its record is sealed
        from mashbox.gguf import resolve as GR
        base = os.path.dirname(path)
        for p in GR.shards(os.path.basename(pin["file"])):
            q = os.path.join(base, p)
            if os.path.exists(q):
                os.remove(q)
    print(json.dumps({k: rec.get(k) for k in ("verdict", "arch", "tpu_key", "seconds_total")}
                     | dict(tok=rec.get("b_tokenizer", {}).get("nfc_equal"),
                            top1=rec.get("c_cpu", {}).get("top1_share"), kl=rec.get("c_cpu", {}).get("kl_mean"),
                            same_nmse=(rec.get("c_cpu", {}).get("identical_weights") or {}).get("nmse"),
                            greedy=[rec.get("c_cpu", {}).get(k) for k in ("full_match", "near_tie_divergences", "other_divergences")])))
    return 0 if rec["verdict"] == "cpu-checked" else 1


if __name__ == "__main__":
    sys.exit(main())
