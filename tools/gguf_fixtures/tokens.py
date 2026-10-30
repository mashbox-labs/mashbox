"""Tiny vocabulary-only GGUF files for the pre-tokenizer test, and llama.cpp's token ids for them.

  python tools/gguf_fixtures/tokens.py make --llama-cpp DIR [--out tests/fixtures/gguf/pretok.json]

One file per case (a tokenizer.ggml.pre name and a vocabulary layout), written by gguf-py from the case alone: the test
rebuilds it with write() and checks its sha256. The ids are llama.cpp's tokenizer on that file for each text, through
llama_tokenize.cpp (as llama-tokenize --ids --no-bos: no BOS, special tokens parsed). DIR is a llama.cpp checkout at fixtures.LLAMA_CPP_COMMIT with a CPU build in DIR/build
(only needed to make the json; the test needs neither llama.cpp nor a chip).

The vocabulary: the 256 byte tokens, merged tokens for a few words and digit runs, two whole words with no merges that
spell them (only a whole-word lookup finds them), and 3 control tokens placed after the normal tokens ("after"),
before them ("first") or among them ("mixed").
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import LLAMA_CPP_COMMIT, byte_chars, sha256  # noqa: E402

SPECIAL = ["<s>", "</s>", "<|im_end|>"]
MERGES = ["Ġ t", "h e", "Ġt he", "i n", "Ġ a", "1 2", "12 3", "o r", "l d", "or ld", "Ġ W", "ĠW orld", "C a", "s e", "Ca se",
          "Ġ 1", "Ġ1 2", "' s", "e r"]
WHOLE = ["Ġhello", "Server"]          # in the vocabulary, not reachable by the merges
CASES = [("deepseek-r1-qwen", "after"), ("falcon3", "first"), ("dbrx", "mixed"), ("tekken", "first"), ("smollm", "first"),
         ("qwen2", "after"), ("llama-bpe", "after"), ("tekken", "mixed"), ("refact", "first"), ("exaone", "first"),
         ("qwen35", "after"), ("gemma4", "gemma4"), ("llama-bpe", "digits")]
# layout "digits": the "after" vocabulary plus merges that build whole digit runs, so a split that cuts digit runs
# differently (1-3 digits, one digit, any length) gives different ids (llama.cpp's LLAMA3 split cuts every 3)
DIGIT_MERGES = ["123 4", "1234 5", "4 5", "45 6", "456 7", "5 6", "56 7", "3 4", "3 .", "3. 1", "3.1 4"]
# layout "gemma4": a tokenizer.ggml.model "gemma4" vocabulary (gguf/tokenizer.py _gemma4): byte tokens <0x00>..<0xFF>, the
# space written as U+2581, the merges above in that form, the same two whole words, control tokens first
G4_SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>", "<|im_end|>"]
BYTE = 6
TEXTS = ["the cat in a hat", " hello World", "hello", "HTTPServer camelCaseWord", "ÉCOLE école Ünïcödé", "123 12345 3.14",
         " 12 1234567", "it's IT'S they'RE", "a  b\n\nc\n", "x/y\r\nz", "the<|im_end|>in", "<s>hi</s> a", "東京 🙂 naïve",
         "  leading", "trailing  ", "\t\ttabs and\n\n\n"]


def tokens(layout):
    """-> (tokens, types, merges) for a layout"""
    bc = byte_chars()
    normal = [bc[b] for b in range(256)]
    for m in MERGES:
        t = m.replace(" ", "")
        if t not in normal:
            normal.append(t)
    normal += WHOLE
    if layout == "after":
        toks = normal + SPECIAL
    elif layout == "digits":
        for m in DIGIT_MERGES:
            t = m.replace(" ", "")
            if t not in normal:
                normal.append(t)
        merges = MERGES + DIGIT_MERGES
        return normal + SPECIAL, [3 if t in SPECIAL else 1 for t in normal + SPECIAL], merges
    elif layout == "first":
        toks = SPECIAL + normal
    else:                                   # mixed: one control token among the bytes, two after the merged tokens
        toks = normal[:100] + SPECIAL[:1] + normal[100:] + SPECIAL[1:]
    types = [3 if t in SPECIAL else 1 for t in toks]
    return toks, types, MERGES


def g4_tokens():
    """-> (tokens, types, merges) of the gemma4 layout"""
    sp = "\u2581"
    toks, types = list(G4_SPECIAL), [3] * len(G4_SPECIAL)
    toks += [f"<0x{b:02X}>" for b in range(256)]
    types += [BYTE] * 256
    normal = [sp] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    normal += [str(d) for d in range(10)] + list(".,'/!?-") + ["\n", "\t"]
    merges = [m.replace("\u0120", sp) for m in MERGES] + [f"{sp} {sp}"]
    for m in merges:
        t = m.replace(" ", "")
        if t not in normal:
            normal.append(t)
    normal += [w.replace("\u0120", sp) for w in WHOLE]
    return toks + normal, types + [1] * len(normal), merges


def write(path, pre, layout):
    """the case's vocabulary-only GGUF (llama hyperparameters so llama.cpp opens it; no tensors)"""
    import gguf
    toks, types, merges = g4_tokens() if layout == "gemma4" else tokens(layout)
    w = gguf.GGUFWriter(path, "llama")
    w.add_name(f"mashbox-vocab-{pre}-{layout}")
    w.add_context_length(256)
    w.add_embedding_length(64)
    w.add_block_count(1)
    w.add_feed_forward_length(128)
    w.add_head_count(4)
    w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(16)
    w.add_tokenizer_model("gemma4" if layout == "gemma4" else "gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(toks.index("<bos>" if layout == "gemma4" else "<s>"))
    w.add_eos_token_id(toks.index("<eos>" if layout == "gemma4" else "</s>"))
    w.add_add_bos_token(False)
    w.add_chat_template("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def compile_tool(llama_cpp, out):
    """llama_tokenize.cpp built against DIR's CPU build -> (exe, env)"""
    root = os.path.abspath(llama_cpp)
    bindir = os.path.join(root, "build", "bin")
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_tokenize.cpp")
    subprocess.run(["g++", "-O2", "-std=c++17", src, f"-I{root}/include", f"-I{root}/ggml/include", f"-L{bindir}", "-lllama",
                    f"-Wl,-rpath,{bindir}", "-o", out], check=True)
    return out, dict(os.environ, LD_LIBRARY_PATH=bindir)


def llama_ids(exe, env, path, texts, td):
    """llama.cpp's ids for each text on the file at path"""
    tf = os.path.join(td, "texts.bin")
    with open(tf, "wb") as f:
        for t in texts:
            f.write(t.encode("utf-8") + b"\0")
    r = subprocess.run([exe, path, tf], capture_output=True, text=True, check=True, env=env)
    ids = [json.loads(line) for line in r.stdout.splitlines()]
    assert len(ids) == len(texts), (len(ids), len(texts))
    return ids


def make(args):
    out = []
    with tempfile.TemporaryDirectory() as td:
        exe, env = compile_tool(args.llama_cpp, os.path.join(td, "llama_tokenize"))
        for pre, layout in CASES:
            p = write(os.path.join(td, f"{pre}-{layout}.gguf"), pre, layout)
            out.append(dict(pre=pre, layout=layout, gguf_sha256=sha256(p), ids=llama_ids(exe, env, p, TEXTS, td)))
            print(pre, layout, out[-1]["gguf_sha256"][:16], [len(x) for x in out[-1]["ids"]])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(llama_cpp_commit=LLAMA_CPP_COMMIT, tool="tools/gguf_fixtures/llama_tokenize.cpp (as llama-tokenize "
                       "--ids --no-bos: no BOS, special tokens parsed)", texts=TEXTS, cases=out), f, ensure_ascii=False, indent=1)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--llama-cpp", required=True, metavar="DIR")
    m.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                 "tests", "fixtures", "gguf", "pretok.json"))
    args = ap.parse_args(argv)
    make(args)


if __name__ == "__main__":
    main()
