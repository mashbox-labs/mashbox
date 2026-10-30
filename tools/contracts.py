#!/usr/bin/env python3
"""Which contracts a GGUF row stands on, and whether it still stands (tests/contracts/conftest.py writes the results).

A GGUF file runs here when its architecture, its tensor types and its tokenizer do, as llama.cpp supports files: by
type, never by file. So a row's record names the contracts of its kinds, not the bytes of the engine files it ran:

  arch/<general.architecture>     a tiny random-weight file of that architecture vs llama.cpp's logits
  op/<ggml type>                  every tensor type in the file: dequantization and matmul vs a NumPy reference
  tok/<tokenizer>                 the file's tokenizer: tok/<tokenizer.ggml.pre> for "gpt2" vocabularies, tok/spm for
                                  SentencePiece ("llama"), tok/gemma4 for gemma4 vocabularies; ids vs llama.cpp's
  path/run                        `mashbox run` through the CLI, resolver, loader, engine, sampler, chat and server
"""
SPM, GEMMA4 = "tok/spm", "tok/gemma4"


def tok_id(model, pre):
    """the tokenizer contract of a file's tokenizer.ggml.model and tokenizer.ggml.pre"""
    if model == "llama":
        return SPM
    if model == "gemma4":
        return GEMMA4
    return f"tok/{pre or 'default'}"


def tokenizer_ids():
    """every tokenizer contract this version reads a file with (gguf/tokenizer.py PRE, SentencePiece, gemma4)"""
    from mashbox.gguf import tokenizer
    return sorted({tok_id("gpt2", p) for p in tokenizer.PRE} | {SPM, GEMMA4})
