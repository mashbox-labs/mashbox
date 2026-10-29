"""GGUF pre-tokenizer names and vocabulary layouts (gguf/pretok.py): our token ids vs llama.cpp's llama-tokenize on tiny
vocabulary-only files (tests/fixtures/gguf/pretok.json says how they were made). No chip, no network, no secrets.

Cases: deepseek-r1-qwen, falcon3, dbrx, tekken, smollm, refact, exaone (the names pretok.py adds) and qwen2, llama-bpe (the names
tokenizer.py had), with the control tokens after the normal tokens, before them, or among them. Bar: every text gives
the same ids as llama.cpp.
"""
import json
import os

import pytest

import gguf  # noqa: F401,E402  hard dependency (pyproject gguf==0.19.0): a missing install fails, never skips
import tokenizers  # noqa: E402  hard dependency (pyproject): a missing install fails, never skips

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "gguf", "pretok.json"), encoding="utf-8"))

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gguf_tokens", os.path.join(ROOT, "tools", "gguf_fixtures", "tokens.py"))
TK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TK)

from mashbox.gguf import GGUFError, tokenizer as GT  # noqa: E402
from mashbox.gguf.reader import File  # noqa: E402


@pytest.mark.parametrize("case", FX["cases"], ids=lambda c: f"{c['pre']}-{c['layout']}")
def test_ids_match_llama_tokenize(case, tmp_path):
    p = TK.write(str(tmp_path / "v.gguf"), case["pre"], case["layout"])
    assert TK.sha256(p) == case["gguf_sha256"], "fixture bytes changed; rerun tools/gguf_fixtures/tokens.py make"
    d = GT.model_dir(File(p), p, root=str(tmp_path / "tok"))          # also checks every added token keeps its id
    tok = tokenizers.Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    bad = [(t, tok.encode(t, add_special_tokens=False).ids, want) for t, want in zip(FX["texts"], case["ids"])
           if tok.encode(t, add_special_tokens=False).ids != want]
    assert not bad, bad


def test_layouts_keep_file_ids(tmp_path):
    for layout in ("after", "first", "mixed"):
        toks, _, _ = TK.tokens(layout)
        p = TK.write(str(tmp_path / f"{layout}.gguf"), "qwen2", layout)    # a name every version reads: only the layout is new
        tj = json.loads(GT.files_dict(File(p), p)["tokenizer.json"])
        added = {a["content"]: a["id"] for a in tj["added_tokens"]}
        assert added == {s: toks.index(s) for s in TK.SPECIAL}, layout
        if layout != "after":
            assert tj["model"]["vocab"] == {t: i for i, t in enumerate(toks)}, layout


def test_repeated_token_in_mixed_layout_is_one_line(tmp_path):
    from mashbox.gguf import pretok
    toks, types, _ = TK.tokens("first")
    with pytest.raises(GGUFError) as e:
        pretok.vocab_layout(toks + ["Ġt"], types + [1], "m.gguf")
    assert "repeats token" in str(e.value) and "\n" not in str(e.value)
    _, _, kept = pretok.vocab_layout(toks, types, "m.gguf", ["h e", "x \ufffd", "q z"])
    assert kept == ["h e"]                        # "\ufffd" and "qz" are not in the vocabulary


def test_bos_defaults_follow_llama_cpp():
    """falcon3 and tekken files turn BOS on (id 11 when the file names none) unless the file says; others keep the keys"""
    from mashbox.gguf import pretok

    class F(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)
    assert pretok.bos(F({"tokenizer.ggml.pre": "falcon3"})) == (11, True)
    assert pretok.bos(F({"tokenizer.ggml.pre": "tekken", "tokenizer.ggml.bos_token_id": 1})) == (1, True)
    assert pretok.bos(F({"tokenizer.ggml.pre": "tekken", "tokenizer.ggml.add_bos_token": False})) == (11, False)
    assert pretok.bos(F({"tokenizer.ggml.pre": "dbrx", "tokenizer.ggml.bos_token_id": 7})) == (7, None)
    assert pretok.bos(F({"tokenizer.ggml.pre": "qwen2"})) == (None, None)


def test_chat_template_can_print_the_date(tmp_path):
    """Mistral's templates call strftime_now (transformers and llama.cpp give templates that function)"""
    import datetime
    from mashbox.runtime.chat import Chat
    d = tmp_path / "m"
    d.mkdir()
    p = TK.write(str(tmp_path / "v.gguf"), "qwen2", "after")             # a file every version reads: only the date call is new
    for k, v in GT.files_dict(File(p), p).items():
        (d / k).write_text(v if k != "chat_template.jinja" else "{{ strftime_now('%Y-%m-%d') }}|{{ messages[0]['content'] }}")
    day = lambda: datetime.datetime.now().strftime("%Y-%m-%d")
    before = day()
    text = Chat(str(d)).render([dict(role="user", content="hi")])
    assert text in (before + "|hi", day() + "|hi"), text          # either side of a midnight
