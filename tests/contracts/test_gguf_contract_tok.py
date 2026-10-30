"""Tokenizer contracts tok/<tokenizer> (tools/contracts.py): our ids vs llama.cpp a37cc46's on tiny vocabulary files.

Bar, written before the first run: every fixture text gives exactly llama.cpp's ids (llama-tokenize --ids --no-bos:
no BOS, special tokens parsed), and the BOS id and add-BOS flag are llama.cpp's for the file. No tolerance: ids are
integers. Coverage: every tokenizer this version reads (tools/contracts.py tokenizer_ids) has at least one case here,
so a pre-tokenizer added to gguf/tokenizer.py PRE without a llama.cpp case is red.

Cases: tok/<pre> and tok/gemma4 (layout "gemma4") from tests/fixtures/gguf/pretok.json (tools/gguf_fixtures/tokens.py), tok/spm from
gemma3-spm.tokens.json and spm-prefix.tokens.json (tools/gguf_fixtures/more.py). No chip, no network, no secrets.
"""
import importlib.util
import json
import os

import pytest

import gguf  # noqa: F401  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
import tokenizers  # noqa: E402  hard dependency (pyproject): a missing install fails, never skips

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(ROOT, "tests", "fixtures", "gguf")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TK = _load("gguf_tokens", "tools/gguf_fixtures/tokens.py")
MORE = _load("gguf_fixtures_more", "tools/gguf_fixtures/more.py")
C = _load("mashbox_contracts", "tools/contracts.py")
PRETOK = json.load(open(os.path.join(FX, "pretok.json"), encoding="utf-8"))
SPM_ROWS = json.load(open(os.path.join(FX, "gemma3-spm.tokens.json"), encoding="utf-8"))
SPM_PREFIX = json.load(open(os.path.join(FX, "spm-prefix.tokens.json"), encoding="utf-8"))

from mashbox.gguf import pretok, tokenizer as GT  # noqa: E402
from mashbox.gguf.reader import File  # noqa: E402

CASES = ([(C.tok_id("gemma4" if c["layout"] == "gemma4" else "gpt2", c["pre"]), f"{c['pre']}-{c['layout']}")
          for c in PRETOK["cases"]]
         + [(C.SPM, "gemma3-spm")] + [(C.SPM, f"spm-prefix-{v}") for v in sorted(SPM_PREFIX["variants"])])


def _ours(tok, texts):
    return [tok.encode(t, add_special_tokens=False).ids for t in texts]


def _check(c, p, texts, want):
    f = File(p)
    tok = tokenizers.Tokenizer.from_str(GT.files_dict(f, p)["tokenizer.json"])
    got = _ours(tok, texts)
    c.record(json.dumps([list(map(int, x)) for x in got]))
    c.record(json.dumps(pretok.bos(f)))
    bad = [(t, g, w) for t, g, w in zip(texts, got, want) if g != w]
    assert len(got) == len(want) and not bad, bad


@pytest.mark.contract
@pytest.mark.parametrize("cid,case", CASES, ids=[f"{c}:{k}" for c, k in CASES])
def test_tokenizer_ids_match_llama_cpp(cid, case, contract, tmp_path):
    c = contract(cid)
    if case.startswith("spm-prefix-"):
        v = case[len("spm-prefix-"):]
        want = SPM_PREFIX["variants"][v]
        p = MORE.write_gemma3(str(tmp_path / f"prefix-{v}.gguf"), seed=MORE.FIXTURES["gemma3-f32"]["seed"], layers=1,
                              prefix=MORE.PREFIX[v])
        rows = want["rows"]
    elif case == "gemma3-spm":
        want, p, rows = SPM_ROWS, MORE.build("gemma3-f32", str(tmp_path)), SPM_ROWS["rows"]
    else:
        want = next(x for x in PRETOK["cases"] if f"{x['pre']}-{x['layout']}" == case)
        p = TK.write(str(tmp_path / "v.gguf"), want["pre"], want["layout"])
        rows = [dict(text=t, ids=i) for t, i in zip(PRETOK["texts"], want["ids"])]
    assert MORE.sha256(p) == want["gguf_sha256"], f"{case}: fixture bytes changed; rerun its tools/gguf_fixtures maker"
    _check(c, p, [r["text"] for r in rows], [r["ids"] for r in rows])


def test_every_tokenizer_the_engine_reads_has_a_contract_case():
    have = {cid for cid, _ in CASES}
    missing = [t for t in C.tokenizer_ids() if t not in have]
    assert not missing, f"tokenizers read with no llama.cpp case: {missing}"


def test_tok_id():
    assert C.tok_id("gpt2", "qwen2") == "tok/qwen2" and C.tok_id("llama", None) == C.SPM
    assert C.tok_id("gemma4", "x") == C.GEMMA4 and C.tok_id("gpt2", None) == "tok/default"
