"""More GGUF pre-tokenizer names, the vocabulary layout with control tokens first or mixed in, and llama.cpp's BOS
defaults for those names. tokenizer.py registers the names next to its own (qwen35, qwen2, llama-bpe).

Each name maps to the split llama.cpp gives it (src/llama-vocab.cpp: the name picks a pre type, the type picks its
regexes), written in Hugging Face `tokenizers` form:

  deepseek-r1-qwen  llama.cpp's QWEN2 type: the qwen2 split with NFC, as the DeepSeek-R1-Distill-Qwen tokenizer.json.
  falcon3           llama.cpp's LLAMA3 group: the llama-bpe split, a word that is one vocabulary entry taken whole,
                    BOS on when the file does not say.
  dbrx              llama.cpp's DBRX type: the llama3 split without the whole-word lookup (phi-4's files).
  tekken            Mistral's Tekken split as Mistral's tokenizer.json writes it (llama.cpp's TEKKEN type is the same
                    split, except that llama.cpp treats non-ASCII letters as caseless; ids measured equal on batch C's
                    files), whole-word lookup, BOS on when the file does not say.
  smollm            llama.cpp's SMOLLM type: two splits in order, every digit alone, then llama.cpp's GPT2 split.
  refact, exaone    llama.cpp's REFACT and EXAONE types: the same two splits as smollm (granite and EXAONE 3.5 files).
  minicpm5          llama.cpp's MINICPM5 type: two splits in order, digits in runs of up to 3, then the qwen2 split with
                    digit runs of any length; a word that is one vocabulary entry taken whole (MiniCPM5 files).
  granite-docling   llama.cpp's GRANITE_DOCLING type: llama.cpp's GPT2 split alone (granite-4.2 files).

Vocabulary layout: llama.cpp keeps every token at its file id. Hugging Face `tokenizers` numbers added tokens after the
BPE vocabulary, so tokenizer.py reads a file whose control, user-defined and unused tokens all come after the normal
ones as before (vocabulary 0..N-1, those tokens added from N on). A file with control tokens first (Mistral's Tekken
files, SmolLM2, Falcon3) or mixed in (phi-4) goes through vocab_layout below: every token keeps its file id in the
vocabulary and the control, user-defined and unused ones are also added tokens with the same ids, the layout of those
models' own tokenizer.json.
"""
from .reader import GGUFError, llama_cpp_command

CONTROL, USER_DEFINED, UNUSED = 3, 4, 5
TEKKEN_RE = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
             r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|"
             r"\s*[\r\n]+|\s+(?!\S)|\s+")
GPT2_RE = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
SMOLLM_RES = [r"\p{N}", GPT2_RE]
MINICPM5_RES = [r"\p{N}{1,3}", r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|"
                                r"\s*[\r\n]+|\s+(?!\S)|\s+"]
ADD_BOS = ("falcon3", "tekken")       # llama.cpp turns BOS on for these pre types unless tokenizer.ggml.add_bos_token says
BPE_BOS_ID = 11                       # llama.cpp's BOS id for a BPE vocabulary whose file names none


def table(qwen2_re, llama3_re):
    """-> {name: (normalizer or None, split regex or a list of regexes applied in order, ignore_merges)}"""
    return {"deepseek-r1-qwen": ("NFC", qwen2_re, False), "falcon3": (None, llama3_re, True),
            "dbrx": (None, llama3_re, False), "tekken": (None, TEKKEN_RE, True), "smollm": (None, SMOLLM_RES, False),
            "refact": (None, SMOLLM_RES, False), "exaone": (None, SMOLLM_RES, False),
            "minicpm5": (None, MINICPM5_RES, True), "granite-docling": (None, GPT2_RE, False)}


def splits(regex):
    """-> the regexes of one PRE entry, in order"""
    return [regex] if isinstance(regex, str) else list(regex)


def vocab_layout(tokens, types, path=None, merges=()):
    """-> (added tokens, vocab, merges) for a file whose control tokens come first or among the normal ones: every token
    at its file id, the control, user-defined and unused ones also added tokens (matched before the split, as llama.cpp
    partitions the text on them). A merge whose parts or result are not in the vocabulary is dropped: Hugging Face
    `tokenizers` refuses it, and llama.cpp never uses it (phi-4's files store id 5809 as a control token \ufffd, so the
    merges that spell its byte form only ever meet text that was already split out as that control token)."""
    vocab, added = {}, []
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if t in vocab:
            raise GGUFError(f"mashbox: token {i} ({t!r}) repeats token {vocab[t]} in this file's vocabulary; this tokenizer "
                            f"layout is not read by this version. llama.cpp runs it: {llama_cpp_command(path)}")
        vocab[t] = i
        if ty in (CONTROL, USER_DEFINED, UNUSED):
            added.append(dict(id=i, content=t, single_word=False, lstrip=False, rstrip=False, normalized=False,
                              special=ty != USER_DEFINED))
    pair = lambda m: m.split(" ", 1) if " " in m else (m, None)
    keep = [m for m in merges if pair(m)[1] is not None and pair(m)[0] in vocab and pair(m)[1] in vocab
            and "".join(pair(m)) in vocab]
    return added, vocab, keep


def bos(f):
    """-> (BOS id or None, add BOS) as llama.cpp reads a file: its own keys, and for the names in ADD_BOS, BOS on when
    the file does not say and id BPE_BOS_ID when it names none. Other names: the file's keys only."""
    i, add = f.get("tokenizer.ggml.bos_token_id"), f.get("tokenizer.ggml.add_bos_token")
    if f.get("tokenizer.ggml.pre") in ADD_BOS:
        add = True if add is None else add
        i = BPE_BOS_ID if i is None else i
    return i, add
