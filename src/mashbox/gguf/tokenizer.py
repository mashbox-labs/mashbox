"""The tokenizer, chat template and sampling defaults inside a GGUF file, as the three files mashbox.runtime.chat.Chat
reads (tokenizer.json, chat_template.jinja, generation_config.json), written once per file to
~/.cache/mashbox/gguf/<sha16 of the metadata>/.

GGUF stores a byte-level BPE vocabulary (`tokenizer.ggml.model` = "gpt2") with its merges, token types and a
pre-tokenizer name (`tokenizer.ggml.pre`); llama.cpp maps the name to a split regex (src/llama-vocab.cpp). The names
below map to the same regex in Hugging Face `tokenizers` form; pretok.py adds more names and the vocabulary layout with
control tokens first. A name not listed stops with a clear message.

Gemma 4 files store `tokenizer.ggml.model` = "gemma4": BPE merges over raw UTF-8 (no byte-level remap), every space
written as U+2581 first, the text split only into runs of newlines and runs of anything else, and byte-fallback tokens
<0xXX> for what the merges cannot spell (llama.cpp's LLAMA_VOCAB_PRE_TYPE_GEMMA4).
"""
import hashlib
import json
import os

from . import pretok
from .reader import GGUFError, llama_cpp_command

# tokenizer.ggml.pre -> (normalizer, split regex), as the model's own tokenizer.json on the Hub
QWEN35_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
             r"\s*[\r\n]+|\s+(?!\S)|\s+")
QWEN2_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
            r"\s+(?!\S)|\s+")
LLAMA3_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
             r"\s+(?!\S)|\s+")
# name -> (normalizer or None, split regex, ignore_merges: a word that is one vocabulary entry is taken whole, as
# llama.cpp's LLAMA3 pre-tokenizer type and Llama 3's own tokenizer.json do)
PRE = {"qwen35": (None, QWEN35_RE, False), "qwen2": ("NFC", QWEN2_RE, False), "llama-bpe": (None, LLAMA3_RE, True)}
PRE.update(pretok.table(QWEN2_RE, LLAMA3_RE))
CONTROL, USER_DEFINED, UNUSED = 3, 4, 5
EOG_NAMES = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<eos>", "<turn|>",
             "<|tool_response>")   # as llama.cpp's EOG scan


def _bytes_level():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    return bs


def no_tokenizer(f, path=None):
    """None, or the one line for a model file with no tokenizer: no tokenizer.ggml.model key (maya1's GGUF has no
    tokenizer.* key at all; llama.cpp a37cc46 requires that key and does not load the file: "error loading model
    vocabulary: key not found in model: tokenizer.ggml.model"), or "none" / "no_vocab" (a file with no vocabulary, which
    takes no text)"""
    kind = f.get("tokenizer.ggml.model")
    if kind not in (None, "none", "no_vocab"):
        return None
    name = path.rsplit("/", 1)[-1] if path else "this file"
    what = "no tokenizer.ggml.model key" if kind is None else f'tokenizer.ggml.model is "{kind}"'
    return (f"mashbox: {name} has no tokenizer ({what}), so mashbox cannot turn text into its tokens and does not run it."
            + (' llama.cpp does not load it either ("key not found in model: tokenizer.ggml.model").' if kind is None else ""))


def files_dict(f, path=None):
    """-> {file name: text} for tokenizer.json, chat_template.jinja, generation_config.json"""
    if no_tokenizer(f, path):
        raise GGUFError(no_tokenizer(f, path))
    kind = f.get("tokenizer.ggml.model")
    pre = f.get("tokenizer.ggml.pre", "default")
    if kind == "gemma4":
        tj, tokens = _gemma4(f, path)
        return _finish(f, tj, tokens)
    if kind == "llama":                            # SentencePiece vocabularies (gguf/spm.py)
        from .spm import tokenizer_json
        return _finish(f, *tokenizer_json(f, path))
    if kind != "gpt2" or pre not in PRE:
        raise GGUFError(f'mashbox: the tokenizer in this file (model "{kind}", pre-tokenizer "{pre}") is not read by this version. '
                        f"llama.cpp runs it: {llama_cpp_command(path)}")
    tokens = list(f.get("tokenizer.ggml.tokens"))
    types = list(f.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    merges = list(f.get("tokenizer.ggml.merges", []))
    norm, regex, ignore_merges = PRE[pre]
    # Hugging Face `tokenizers` numbers added tokens after the BPE vocabulary, so the vocabulary must be ids 0..N-1 and
    # every control, user-defined and unused (padding) token an added token from N on; a file with control tokens first
    # or among the normal ones keeps every token at its file id (pretok.vocab_layout)
    added, vocab = [], {}
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if ty in (CONTROL, USER_DEFINED, UNUSED):
            added.append(dict(id=i, content=t, single_word=False, lstrip=False, rstrip=False, normalized=False,
                              special=ty != USER_DEFINED))
        elif added:
            added, vocab, merges = pretok.vocab_layout(tokens, types, path, merges)
            break
        else:
            vocab[t] = i
    bl = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=False)
    tj = dict(version="1.0", truncation=None, padding=None, added_tokens=added,
              normalizer=dict(type=norm) if norm else None,
              pre_tokenizer=dict(type="Sequence", pretokenizers=[
                  *[dict(type="Split", pattern=dict(Regex=r), behavior="Isolated", invert=False) for r in pretok.splits(regex)],
                  bl]),
              post_processor=bl, decoder=bl,
              model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="",
                         fuse_unk=False, byte_fallback=False, ignore_merges=ignore_merges, vocab=vocab, merges=merges))
    return _finish(f, tj, tokens)


def _gemma4(f, path):
    """-> (tokenizer.json dict, tokens) for tokenizer.ggml.model "gemma4". Every token keeps its id in the BPE vocabulary;
    control, user-defined and unused tokens are also added tokens (matched before the split, as llama.cpp partitions
    on them)."""
    tokens = list(f.get("tokenizer.ggml.tokens"))
    types = list(f.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    vocab, added = {}, []
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if t in vocab:
            raise GGUFError(f"mashbox: token {i} ({t!r}) repeats token {vocab[t]} in this file's vocabulary; this tokenizer "
                            f"layout is not read by this version. llama.cpp runs it: {llama_cpp_command(path)}")
        vocab[t] = i
        if ty in (CONTROL, USER_DEFINED, UNUSED):
            added.append(dict(id=i, content=t, single_word=False, lstrip=False, rstrip=False, normalized=False,
                              special=ty != USER_DEFINED))
    merges = []
    for m in f.get("tokenizer.ggml.merges", []):
        j = m.find(" ", 1)                         # llama.cpp: the first space after the first byte separates the pair
        if j > 0:
            merges.append([m[:j], m[j + 1:]])
    tj = dict(version="1.0", truncation=None, padding=None, added_tokens=added,
              normalizer=dict(type="Replace", pattern=dict(String=" "), content="\u2581"),
              pre_tokenizer=dict(type="Split", pattern=dict(Regex="[^\n]+|[\n]+"), behavior="Isolated", invert=False),
              post_processor=None,
              decoder=dict(type="Sequence", decoders=[dict(type="Replace", pattern=dict(String="\u2581"), content=" "),
                                                      dict(type="ByteFallback"), dict(type="Fuse")]),
              model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="",
                         fuse_unk=False, byte_fallback=True, ignore_merges=False, vocab=vocab, merges=merges))
    return tj, tokens


def _finish(f, tj, tokens):
    """-> the three files from a tokenizer.json dict: stop ids, sampling defaults, chat template with BOS/EOS names"""
    types = list(f.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    eog = []
    for k in ("eos_token_id", "eot_token_id", "eom_token_id"):
        v = f.get(f"tokenizer.ggml.{k}")
        if v is not None and int(v) not in eog:
            eog.append(int(v))
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if t in EOG_NAMES and i not in eog:          # any token type: llama.cpp marks these EOG and makes them control
            eog.append(i)
    gc = dict(eos_token_id=eog)
    for k in ("bos_token_id", "pad_token_id"):
        v = f.get(f"tokenizer.ggml.{k}")
        if v is not None:
            gc[k] = int(v)
    for ours, key in (("temperature", "general.sampling.temp"), ("top_k", "general.sampling.top_k"),
                      ("top_p", "general.sampling.top_p")):
        v = f.get(key)
        if v is not None:
            gc[ours] = float(f"{v:.6g}") if isinstance(v, float) else v      # stored as float32: 0.95, not 0.949999988
    tmpl = f.get("tokenizer.chat_template")
    if not tmpl:
        raise GGUFError("mashbox: this file has no chat template (tokenizer.chat_template); add one with llama.cpp's "
                        "gguf-py (gguf_new_metadata.py --chat-template-file) or use a file that has one")
    tmpl = bos_template(f, tokens, tmpl)
    return {"tokenizer.json": json.dumps(tj, ensure_ascii=False), "chat_template.jinja": tmpl,
            "generation_config.json": json.dumps(gc, indent=1)}


def bos_template(f, tokens, tmpl):
    """llama.cpp renders a chat template with the file's bos_token / eos_token text, and when the file says
    add_bos_token it strips one leading BOS from the result and adds BOS when tokenizing: one BOS at the start either way.
    Chat renders without those names, so they are set at the top of the template here, and with add_bos_token a template
    that never writes bos_token gets the BOS text in front. Files without a BOS id (Qwen) are unchanged."""
    head, (bos_id, add_bos) = "", pretok.bos(f)
    for name in ("bos", "eos"):
        i = bos_id if name == "bos" else f.get("tokenizer.ggml.eos_token_id")
        if i is not None and 0 <= int(i) < len(tokens):
            head += "{%- set " + name + "_token = " + json.dumps(tokens[int(i)]) + " %}"
    if not head:
        return tmpl
    if tmpl[:1] == "\n":
        head += "\n"                   # trim_blocks removes the first newline after a block tag: keep the template's own
    if add_bos and bos_id is not None and "bos_token" not in tmpl:
        head += "{{ bos_token }}"
    return head + tmpl


def model_dir(f, path=None, root=None):
    """write the three files once (keyed by their content) and return the directory Chat(model_dir) reads"""
    fs = files_dict(f, path)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(fs["tokenizer.json"])
    for a in json.loads(fs["tokenizer.json"])["added_tokens"]:
        if tok.token_to_id(a["content"]) != a["id"]:
            raise GGUFError(f"mashbox: token {a['id']} ({a['content']!r}) cannot keep its id in the tokenizer built from this "
                            f"file. It runs in llama.cpp: {llama_cpp_command(path)}")
    h = hashlib.sha256()
    for k in sorted(fs):
        h.update(k.encode() + b"\0" + fs[k].encode() + b"\0")
    root = root or os.path.join(os.environ.get("MASHBOX_CACHE", os.path.expanduser("~/.cache/mashbox")), "gguf")
    d = os.path.join(root, h.hexdigest()[:16])
    if not os.path.exists(os.path.join(d, "generation_config.json")):
        os.makedirs(d, exist_ok=True)
        for k, v in fs.items():
            with open(os.path.join(d, k + ".part"), "w", encoding="utf-8") as o:
                o.write(v)
            os.replace(os.path.join(d, k + ".part"), os.path.join(d, k))
    return d
