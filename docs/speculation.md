# Speculative decoding

Speculative decoding lets one pass of the model give several tokens. A cheap drafter guesses the next few tokens, and the model checks all of the guesses in one pass. It keeps every guess it agrees with, plus one token of its own. You meet it as the `--speculation` option of `mashbox run`, and in the `speculation` line the server prints when it starts.

## What it does for you

- Measured speed: Qwen3.8-27B with its own MTP head (`--mtp 2`) decodes 74.84 tokens/s against 37.77 tokens/s without speculation, 1 request at a time on one TPU v6e-1 (8 chat prompts, 256 new tokens, greedy; see [Qwen3.8-27B](models/qwen3-8-27b.md#speculative-decoding)). No other drafter has a measured speed.
- The answer follows the same rules as without it. A greedy request keeps the model's top token at every position, and a sampled request draws each token with the same settings and seed. Where the two best tokens score the same, or one bf16 step apart, the text can differ from a run without speculation.
- It runs inside the batch: every request in a batch can speculate. With copy from the conversation, a request whose guesses keep missing stops guessing for a while: after 4 passes in a row in which the model keeps none of its guesses, it decodes without guesses for 16 passes, then guesses again, and each time it misses again the wait doubles, up to 256 passes. The MTP head guesses on every pass.
- No drafter has passed its checks on a chip in this release, so `auto` decodes without speculation. Turn it on with `--mtp K` or `--speculation ngram`.

## Where it runs

| Drafter | Tested | Not tested |
|---|---|---|
| The model's own MTP head | Qwen3.8-27B | |
| Copy from the conversation | GGUF files of the `llama`, `qwen2`, `qwen3` and `qwen35` architectures; Qwen3.8-27B with `--speculation ngram` | GGUF files of the `exaone` and `granite` architectures: they run it |

Tested means the drafter runs in `mashbox`'s own tests on small files of that kind. Only the MTP head has a speed measured on a chip.

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
