# Speculative decoding

Speculative decoding makes `mashbox run` generate text faster with the same model. A cheap drafter guesses the next few tokens, and the model checks all of the guesses in one pass. It keeps every guess it agrees with, plus one token of its own, so one pass can give several tokens. You meet it as the `--speculation` option of `mashbox run`, and in the `speculation` line the server prints when it starts.

## What it does for you

- More tokens per second from the same chip and model, most of all when the answer repeats text that is already in the conversation: code edits, quoting a document, agents re-reading files.
- The answer follows the same rules as without it. A sampled request draws each token as it would without speculation (same settings, same seed). A greedy request keeps the model's top token at every position; where the two best tokens score the same, or one bf16 step apart, the text can differ from a run without speculation.
- It runs inside the batch. Every request in a batch can speculate, and a request whose guesses keep missing goes back to plain decoding on its own.

## Options

| Option | What it does |
|---|---|
| `--speculation auto` | The default. Picks a drafter from the model file (see [How auto picks a drafter](#how-auto-picks-a-drafter)) and turns it on only where that drafter passed its checks for this model on this chip. Everywhere else the model decodes without speculation. |
| `--speculation off` | Never speculate. |
| `--speculation mtp` | The model's own multi-token-prediction (MTP) head. |
| `--speculation ngram` | Copy from the conversation: the drafter looks up the last few tokens earlier in the request and proposes the tokens that followed them there. |
| `--speculation draft` | A smaller model with the same tokenizer. This release does not run draft models: the option stops with an error. |
| `--draft-k K` | Guesses per pass for a drafter you pick. `ngram`: 1 to 8, default 3. `mtp`: 1 to the most the model's head supports, default the model's own (Qwen3.8-27B: 1 to 3, default 2). |
| `--mtp K` | Short for `--speculation mtp --draft-k K`. `--mtp 0` turns speculation off. |
| `--mtp-file FILE` | A local MTP draft file instead of the download (its sha256 is still checked). |

A drafter you pick with `--speculation mtp`, `ngram` or `--mtp K` runs wherever it applies to the model, checked on this chip or not. The server then logs `speculation: not gated on this chip`.

Speculation runs on one chip. With `--tensor-parallel-size` above 1, `auto` decodes without it and a drafter you pick stops with an error.

## How auto picks a drafter

`auto` reads the model file, never its name, and takes the first drafter in this list that applies to the model and has passed its checks on this chip:

1. **The model's own MTP head**, when the file has one and `mashbox` implements that head for the file's format. Qwen3.8-27B ships one with its `.bshj` weights.
2. **A smaller model with the same tokenizer.** This release does not run draft models, so `auto` moves on to the next one.
3. **Copy from the conversation** (n-gram prompt lookup), on GGUF files of the `llama`, `qwen2`, `qwen3` and `qwen35` architectures.

On Qwen3.8-27B, `auto` looks at the MTP head only. Copy from the conversation runs there when you pick it with `--speculation ngram`.

When nothing in the list has passed its checks on this chip, the model decodes without speculation and the start-up line says why. Turn it on yourself with `--speculation ngram` or `--mtp K`.

## Check it is on

When the model loads, `mashbox run` prints one JSON line to stderr:

```
{"event": "speculation", "used": true, "method": "ngram", "draft_k": 3, "gated": false, "reason": "--speculation ngram", ...}
```

| Field | Meaning |
|---|---|
| `used` | `true` when speculation is on |
| `method` | the drafter: `mtp` or `ngram` |
| `draft_k` | guesses per pass |
| `gated` | `true` when the drafter passed its checks for this model on this chip |
| `reason` | why it is on, or why the model decodes without it |

## Limits

- Mixture-of-experts files (`qwen35moe`) and Gemma files (`gemma4`, `gemma3`) decode without speculation, and a drafter you pick stops with an error.
- GGUF files run copy from the conversation only. An MTP layer inside a GGUF file is not used.
- If a speculation program fails to compile while the server warms up, the server starts without speculation and the start-up line says so.

## Related

- [Command line](cli.md): every option of `mashbox run`
- [Qwen3.8-27B](models/qwen3-8-27b.md#speculative-decoding): its MTP head, and how the output compares with decoding without it
- [GGUF files](gguf.md): the architectures `mashbox` runs from a GGUF file
