"""llama.cpp's greedy decoding on the tiny GGUF fixtures, for the architecture contracts (tests/contracts/test_arch.py).

  python tools/gguf_fixtures/greedy.py make --llama-cpp DIR [--out tests/fixtures/gguf] [--only NAME ...]

writes, per fixture of fixtures.py and more.py, `<name>.greedy.json` (the file it ran on and its sha256, the prompt ids,
llama.cpp's greedy ids and top-two gaps, the logits' sha256, llama.cpp commit) and `<name>.greedy.logits.f32` (GREEDY_NEW
x n_vocab float32: row s = the logits llama.cpp picks step s from). Float fixtures run as they are. A block-type fixture
runs as its F32 copy holding gguf-py's dequantization of every tensor (fixtures.dequant_copy, the `.f32ref` of the logits
tests): the engine's exact path computes in float32 on those same values, so a greedy difference is about the graph,
not llama.cpp's 8-bit activation rounding in its block-type dot products. DIR is a llama.cpp checkout at
fixtures.LLAMA_CPP_COMMIT with a CPU build in DIR/build (only needed to make references; the tests need neither).

Random weights, small shapes, seeded. Nothing here is a trained model.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GREEDY_NEW = 16


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


base = _load("gguf_fixtures_greedy_base", "fixtures.py")
more = _load("gguf_fixtures_greedy_more", "more.py")
TOKENS, LLAMA_CPP_COMMIT, sha256 = base.TOKENS, base.LLAMA_CPP_COMMIT, base.sha256


def owner(name):
    """the module whose build / dequant_copy writes fixture `name` (more.py's private copy of fixtures.py for its own)"""
    if name in base.FIXTURES:
        return base, base.FIXTURES[name]
    return more.base, more.FIXTURES[name]


def blocks(fx):
    return bool(fx.get("quant")) or fx["make"] == "llama-quantize"


def greedy_file(name, workdir, fixtures_dir):
    """-> (path greedy runs on, "file" | "f32ref"): the fixture as written, or the F32 copy of a block-type fixture"""
    mod, fx = owner(name)
    p = mod.build(name, workdir) if fx["make"] == "gguf-py" else os.path.join(fixtures_dir, f"{name}.gguf")
    if not blocks(fx):
        return p, "file"
    return mod.dequant_copy(name, p, os.path.join(workdir, f"{name}-dequant-f32.gguf")), "f32ref"


def make(args):
    os.makedirs(args.out, exist_ok=True)
    root = os.path.abspath(args.llama_cpp)
    bindir = os.path.join(root, "build", "bin")
    exe = os.path.join(args.out, ".llama_greedy_ids")
    subprocess.run(["g++", "-O2", "-std=c++17", os.path.join(HERE, "llama_greedy_ids.cpp"), f"-I{root}/include",
                    f"-I{root}/ggml/include", f"-L{bindir}", "-lllama", f"-Wl,-rpath,{bindir}", "-o", exe], check=True)
    env = dict(os.environ, LD_LIBRARY_PATH=bindir)
    names = list(base.FIXTURES) + list(more.FIXTURES)
    try:
        with tempfile.TemporaryDirectory() as td:
            for name in names:
                if args.only and name not in args.only:
                    continue
                p, ran_on = greedy_file(name, td, args.out)
                out = os.path.join(args.out, f"{name}.greedy.logits.f32")
                r = subprocess.run([exe, p, out, str(GREEDY_NEW)] + [str(t) for t in TOKENS], check=True, env=env,
                                   capture_output=True, text=True)
                rows = [line.split() for line in r.stdout.strip().splitlines()]
                assert [int(s) for s, _, _ in rows] == list(range(GREEDY_NEW)), r.stdout
                meta = dict(name=name, arch=owner(name)[1]["arch"], ran_on=ran_on, gguf_sha256=sha256(p),
                            prompt=TOKENS, n_new=GREEDY_NEW, ids=[int(i) for _, i, _ in rows],
                            gaps=[float(g) for _, _, g in rows], logits="f32 row-major (n_new, n_vocab)",
                            logits_sha256=sha256(out), llama_cpp_commit=LLAMA_CPP_COMMIT,
                            llama_cpp_backend="CPU, float32 K/V cache, flash attention off, one-token decodes")
                with open(os.path.join(args.out, f"{name}.greedy.json"), "w") as f:
                    json.dump(meta, f, indent=1)
                    f.write("\n")
                print(name, ran_on, meta["ids"], f"min gap {min(meta['gaps']):.3g}")
    finally:
        os.remove(exe)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--llama-cpp", required=True, metavar="DIR")
    m.add_argument("--out", default=os.path.join("tests", "fixtures", "gguf"))
    m.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    return make(a)


if __name__ == "__main__":
    sys.exit(main())
