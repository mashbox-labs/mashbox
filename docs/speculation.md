# Speculative decoding

Speculative decoding lets one pass of the model give several tokens. A cheap drafter guesses the next few tokens, and the model checks all of the guesses in one pass. It keeps the guesses up to the first one it disagrees with, plus one token of its own. You meet it as the `--speculation` option of `mashbox run`, and in the `speculation` line the server prints when it starts.

## What it does for you

- Measured speed: Qwen3.8-27B with its own MTP head (`--mtp 2`) decodes 74.84 tokens/s against 37.77 tokens/s without speculation, 1 request at a time on one TPU v6e-1 (8 chat prompts, 256 new tokens, greedy; see [Qwen3.8-27B](models/qwen3-8-27b.md#speculative-decoding)). No other drafter has a measured speed.
- The answer follows the same rules as without it. A greedy request keeps the model's top token at every position, and a sampled request draws each token with the same settings and seed. Where the two best tokens score the same, or one bf16 step apart, the text can differ from a run without speculation.
- It runs inside the batch: every request in a batch can speculate. With copy from the conversation or a draft model, a request whose guesses keep missing stops guessing for a while: the engine runs several passes in one device call and decides once per device call whether it speculates, so a call in which no request has a guess runs without speculation, and its passes do not count toward the misses, only toward the wait of a request that is already waiting, while in a call that speculates, every pass counts for each request that may guess, whether or not it has a guess in that pass; after 4 passes in a row that count and in which the model keeps none of its guesses, the request waits for 16 passes, in which a device call that starts during the wait makes no guesses for it (the rest of the call in which the wait began still guesses, and those passes count toward the 16 until the wait ends, whether or not the model keeps a guess), then guesses again; each further 4 such passes in a row double the wait, up to 256 passes, and a pass that keeps a guess sets it back to 16 unless the request is waiting. The MTP head guesses on every pass.
- No drafter has passed its checks on a chip in this release, so `auto` decodes without speculation. Turn it on with `--mtp K` or `--speculation ngram`.

## Where it runs

| Drafter | Tested | Not tested |
|---|---|---|
| The model's own MTP head | Qwen3.8-27B | |
| Copy from the conversation | GGUF files of the `llama`, `qwen2`, `qwen3` and `qwen35` architectures; Qwen3.8-27B with `--speculation ngram` | GGUF files of the `exaone` and `granite` architectures: they run it |
| A smaller model with the same tokenizer | A `llama` GGUF model with a `llama` or `qwen3` draft file | GGUF models of the `qwen2`, `qwen3`, `qwen35`, `exaone` and `granite` architectures: they run it; `qwen2`, `exaone` and `granite` draft files: they run as one |

Tested means the drafter runs in `mashbox`'s own tests on small files of that kind. Only the MTP head has a speed measured on a chip.

## Options

| Option | What it does |
|---|---|
| `--speculation auto` | The default. Picks a drafter from the model file (see [How auto picks a drafter](#how-auto-picks-a-drafter)) and turns it on only where that drafter passed its checks for this model on this chip. Everywhere else the model decodes without speculation. |
| `--speculation off` | Never speculate. |
| `--speculation mtp` | The model's own multi-token-prediction (MTP) head. |
| `--speculation ngram` | Copy from the conversation: the drafter looks up the last few tokens earlier in the request and proposes the tokens that followed them there. |
| `--speculation draft` | A smaller model with the same tokenizer drafts for the model. Give it with `--draft-model FILE`; without one the option stops with an error. GGUF models only. |
| `--draft-model FILE` | The draft model: a GGUF file or repo, given as you give the model. It needs the model's tokenizer (the same tokens) and a `llama`, `qwen2`, `qwen3`, `exaone` or `granite` file. On its own it means `--speculation draft`. Its weights and cache take chip memory next to the model's: lower `--slots` or `--ctx` if they do not fit. |
| `--draft-k K` | Guesses per pass for a drafter you pick. `ngram` and `draft`: 1 to 8, default 3. `mtp`: 1 to the most the model's head supports, default the model's own (Qwen3.8-27B: 1 to 3, default 2). |
| `--mtp K` | Short for `--speculation mtp --draft-k K`. `--mtp 0` turns speculation off. |
| `--mtp-file FILE` | A local MTP draft file instead of the download (its sha256 is still checked). |

A drafter you pick with `--speculation mtp`, `ngram`, `draft` or `--mtp K` runs wherever it applies to the model, checked on this chip or not. When it has not passed its checks on this chip, the server logs `speculation: not gated on this chip`.

Speculation runs on one chip. A model from the catalog on more than one chip (`--tensor-parallel-size` above 1; left out, it is 4 on a v5e-4, v6e-4 or v5p-8) decodes without it under `auto`, and a drafter you pick stops with an error. A GGUF file runs on one chip only: `--tensor-parallel-size` above 1 stops the run.

## How auto picks a drafter

`auto` never goes by a name: it reads what the model declares (a GGUF file's metadata, or the catalog entry of a model from the catalog) and takes the first drafter in this list that applies to the model and has passed its checks on this chip:

1. **A trained drafter made for this model** (DFlash, DFlare or EAGLE3), matched by the model's tokenizer and shape. This release does not run trained drafters, so `auto` moves on to the next one.
2. **The model's own MTP head**, when the file has one and `mashbox` implements that head for the file's format. Qwen3.8-27B ships one with its `.bshj` weights.
3. **A smaller model with the same tokenizer**. `auto` does not look one up in this release, so it moves on to the next one. One you give with `--draft-model` runs as `--speculation draft` does: your pick, not gated on this chip.
4. **Copy from the conversation** (n-gram prompt lookup), on the GGUF files listed for it in [Where it runs](#where-it-runs).

On Qwen3.8-27B, `auto` looks at the MTP head only. Copy from the conversation runs there when you pick it with `--speculation ngram`.

When nothing in the list has passed its checks on this chip, the model decodes without speculation and the start-up line says why. Turn it on yourself with `--speculation ngram` or `--mtp K`.

## Check it is on

When the model loads, `mashbox run` prints one JSON line to stderr. For a model from the catalog with `--speculation ngram` it reads:

```
{"event": "speculation", "used": true, "method": "ngram", "draft_k": 3, "gated": false, "reason": "--speculation ngram", ...}
```

| Field | Meaning |
|---|---|
| `used` | `true` when speculation is on |
| `method` | the drafter: `mtp`, `draft_model` or `ngram` |
| `draft_k` | guesses per pass |
| `gated` | `true` when the drafter passed its checks for this model on this chip |
| `reason` | why it is on, or why the model decodes without it |

## Limits

- Mixture-of-experts files (`qwen35moe`, `qwen3moe`, `qwen2moe`) and Gemma files (`gemma4`, `gemma3`) decode without speculation, and a drafter you pick stops with an error.
- GGUF files run copy from the conversation, or a draft model you give. An MTP layer inside a GGUF file is not used.
- A draft model runs on a GGUF model only, and only `llama`, `qwen2`, `qwen3`, `exaone` and `granite` files can be one (every layer full attention).
- If a speculation program fails to compile while the server warms up, the server starts without speculation and the start-up line says so.

## Related

- [Command line](cli.md): every option of `mashbox run`
- [Qwen3.8-27B](models/qwen3-8-27b.md#speculative-decoding): its MTP head, and how the output compares with decoding without it
- [GGUF files](gguf.md): the architectures `mashbox` runs from a GGUF file
