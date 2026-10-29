"""SentencePiece vocabularies in GGUF (`tokenizer.ggml.model` = "llama": Gemma 3, Llama 2, Mistral 7B, Phi-3 mini and
others) as a Hugging Face `tokenizers` BPE that encodes the way llama.cpp's SPM tokenizer does
(src/llama-vocab.cpp, llm_tokenizer_spm_session):

  * every space becomes U+2581, with one in front of each piece of text when `tokenizer.ggml.add_space_prefix` is true
    (llama.cpp's default for these files);
  * the text is not split first: adjacent pieces merge, highest score of the merged piece first;
  * characters no piece covers fall back to the <0xXX> byte tokens.
The merges are listed from the scores the way transformers' SentencePieceExtractor lists them: every split of a piece
into two pieces of the vocabulary, the piece with the higher score first. Control, user-defined, unused and unknown
tokens are also added tokens (matched before the merges, as llama.cpp partitions the text on them); every token keeps
its id.
"""
from .reader import GGUFError, llama_cpp_command

UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 2, 3, 4, 5, 6
SPACE = "▁"


def merges_from_scores(tokens, scores):
    """-> [[left, right], ...]: for each piece, every split whose halves are pieces (ordered by the halves' ids), then
    all of them by the piece's score, highest first (a stable sort keeps the id order among equal scores)"""
    vocab = {t: i for i, t in enumerate(tokens)}
    out = []
    for t, s in zip(tokens, scores):
        local = [(vocab[t[:j]], vocab[t[j:]]) for j in range(1, len(t)) if t[:j] in vocab and t[j:] in vocab]
        out.extend((s, a, b) for a, b in sorted(local))
    out.sort(key=lambda m: -m[0])
    return [[tokens[a], tokens[b]] for _, a, b in out]


def tokenizer_json(f, path=None):
    """-> (tokenizer.json dict, tokens) for tokenizer.ggml.model "llama"."""
    tokens = list(f.get("tokenizer.ggml.tokens"))
    scores = f.get("tokenizer.ggml.scores")
    if scores is None or len(scores) != len(tokens):
        raise GGUFError(f"mashbox: this SentencePiece vocabulary has no scores for its tokens; this tokenizer layout is not "
                        f"read by this version. llama.cpp runs it: {llama_cpp_command(path)}")
    scores = [float(s) for s in scores]
    types = list(f.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    vocab, added = {}, []
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if t in vocab:
            raise GGUFError(f"mashbox: token {i} ({t!r}) repeats token {vocab[t]} in this file's vocabulary; this tokenizer "
                            f"layout is not read by this version. llama.cpp runs it: {llama_cpp_command(path)}")
        vocab[t] = i
        if ty in (CONTROL, USER_DEFINED, UNUSED, UNKNOWN):
            added.append(dict(id=i, content=t, single_word=False, lstrip=False, rstrip=False, normalized=False,
                              special=ty != USER_DEFINED))
    prefix = f.get("tokenizer.ggml.add_space_prefix")
    prefix = True if prefix is None else bool(prefix)
    unk = f.get("tokenizer.ggml.unknown_token_id")
    unk = tokens[int(unk)] if unk is not None and 0 <= int(unk) < len(tokens) else None
    norm = [dict(type="Replace", pattern=dict(String=" "), content=SPACE)]
    dec = [dict(type="Replace", pattern=dict(String=SPACE), content=" "), dict(type="ByteFallback"), dict(type="Fuse")]
    if prefix:
        norm.insert(0, dict(type="Prepend", prepend=SPACE))
        dec.append(dict(type="Strip", content=" ", start=1, stop=0))
    tj = dict(version="1.0", truncation=None, padding=None, added_tokens=added,
              normalizer=dict(type="Sequence", normalizers=norm), pre_tokenizer=None, post_processor=None,
              decoder=dict(type="Sequence", decoders=dec),
              model=dict(type="BPE", dropout=None, unk_token=unk, continuing_subword_prefix=None, end_of_word_suffix=None,
                         fuse_unk=True, byte_fallback=True, ignore_merges=False, vocab=vocab,
                         merges=merges_from_scores(tokens, scores)))
    return tj, tokens
