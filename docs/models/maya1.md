# maya1 on TPU

<!-- catalog:start label -->
Experimental
<!-- catalog:end -->

Maya Research's open text-to-speech model, on one Google Cloud TPU chip. You give it the text to say and a voice (one of 13 names, or a description of your own), and it writes 24 kHz speech to a WAV file. Emotion tags such as `<laugh>`, `<sigh>` and `<whisper>` go inline in the text, where you want them.

## What it is

maya1 is a 3B Llama-style model that writes SNAC audio codes, 7 codes for every 85.3 ms of audio. The SNAC 24 kHz decoder then turns the codes into sound.

How it runs:

- **Model:** maya1 runs on the `mashbox` engine as a Llama model, read from one BF16 GGUF file, with the engine's continuous batching: up to 8 takes decode together, and a new take joins as soon as a slot is free.
- **Decoder:** the SNAC decoder is plain JAX in [src/mashbox/models/maya1/snac.py](../../src/mashbox/models/maya1/snac.py), on the same chip. Its noise comes from the seed, so the same request with the same seed gives the same audio.
- **Kernels:** none of its own. maya1's BF16 weights run as plain JAX bfloat16 matrix products. On a v6e chip the first run still fetches the engine's compiled kernel package (sha256 checked, kept in `~/.cache/mashbox/kernels`), but maya1 uses none of its kernels: the `weights` line it prints shows `"kernel_bytes": 0`.
- **Weights:** maya1 as one BF16 GGUF file and the SNAC decoder as NumPy arrays, from [mashbox-labs/maya1-TPU](https://huggingface.co/mashbox-labs/maya1-TPU). The GGUF file holds no tokenizer, so `tokenizer.json` comes unchanged from [maya-research/maya1](https://huggingface.co/maya-research/maya1). Every file is downloaded at a pinned revision and checked by sha256 before use.

## What you need

| Chip | Status |
|---|---|
| TPU v6e (one chip, v6e-1) | Supported: one chip (v6e-1) |
| TPU v5e (one chip, v5e-1) | Supported: one chip (v5e-1) |
| TPU v5p slice (v5p-8) | Not tested yet |
| TPU v6e slice (v6e-4) | Not tested yet |

At Google's on-demand list price, v6e-1 costs $2.70 an hour in us-east5 and v5e-1 costs $1.20 an hour in us-south1 ([TPU pricing](https://cloud.google.com/tpu/pricing)). The chip bills from create to delete.

You also need the Google Cloud CLI set up and TPU quota in your zone: [google-cloud.md](../google-cloud.md) does that from a new Google account.

Disk: the first run downloads 6.68 GB into `~/.cache/mashbox/maya1` and, on v6e, the engine's kernel package into `~/.cache/mashbox/kernels` (set `MASHBOX_CACHE` to use another disk).

## Install

From your own machine:

```bash
gcloud compute tpus tpu-vm create maya1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh maya1 --zone=us-east5-b
```

On a v5e chip, create it with `--accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite` in a zone with v5e (us-south1-a, for example); [chips.md](../chips.md) lists the settings per chip.

On the VM:

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]>=0.1.15" --with-executables-from huggingface_hub
~/.local/bin/mashbox download maya1
```

`download` fetches the three files (6.68 GB), checks each sha256, and is safe to run again. `run` also downloads what is missing.

## Run

On the VM:

```bash
~/.local/bin/mashbox run maya1 --text "Hello there. <laugh> It works." --voice alloy --out hello.wav
```

It prints one JSON line when it is done: the output file, the seconds of audio, and each take's token count and frame count. The first run on a new chip compiles its programs and saves them in `~/.cache/mashbox/xla/maya1`, so later runs start faster ([Performance](#performance) has both times).

Pick the voice in one of three ways:

```bash
# one of the 13 voice names
~/.local/bin/mashbox run maya1 --text "We are live." --voice onyx --out live.wav
# a voice name, with how to say it
~/.local/bin/mashbox run maya1 --text "We are live." --voice nova --instructions "Very excited, speaking fast." --out live.wav
# your own description, in the words maya1 was trained on
~/.local/bin/mashbox run maya1 --text "We are live." --description "Realistic male voice in the 40s age with british accent. Low pitch, warm timbre, measured pacing." --out live.wav
```

`~/.local/bin/mashbox run maya1 --voices` lists the voice names and the description each one sends. Maya Research's [prompt guide](https://huggingface.co/maya-research/maya1/blob/b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572/prompt.txt) lists the words a description can use (age, accent, pitch, timbre, pacing, tone, intensity).

Emotion tags are the 20 single-token tags of maya1's tokenizer: `<angry>`, `<appalled>`, `<chuckle>`, `<cry>`, `<curious>`, `<disappointed>`, `<excited>`, `<exhale>`, `<gasp>`, `<giggle>`, `<gulp>`, `<laugh>`, `<laugh_harder>`, `<mischievous>`, `<sarcastic>`, `<scream>`, `<sigh>`, `<sing>`, `<snort>` and `<whisper>`. A tag reaches the model only where you type it in `--text`; none is ever added for you, and a tag in a description stays plain text.

Many requests at once: put one JSON object per line in a file and pass it with `--batch`. Each line takes `text`, `out`, and optionally `voice`, `description`, `instructions`, `seed` and the sampling settings below.

```bash
cat > requests.jsonl <<'EOF'
{"text": "Welcome back.", "voice": "ash", "out": "a.wav"}
{"text": "Your order is on its way. <chuckle> Finally.", "voice": "coral", "out": "b.wav"}
EOF
~/.local/bin/mashbox run maya1 --batch requests.jsonl
```

When you are done, delete the chip from your own machine so it stops billing:

```bash
gcloud compute tpus tpu-vm scp maya1:~/hello.wav . --zone=us-east5-b
gcloud compute tpus tpu-vm delete maya1 --zone=us-east5-b --quiet
```

## Options

| Option | Default | What it does |
|---|---|---|
| `--text` | | The text to say. Text longer than 240 characters is cut into takes of up to 240 characters each, at a sentence end where one fits (else at a comma, semicolon, colon, dash or space), and the takes are joined in order. Only the spaces at a cut are dropped; nothing is added. At most 4096 characters. |
| `--voice` | `alloy` | One of the 13 voice names (`--voices` lists them). |
| `--description` | | Your own voice description, instead of `--voice`. Up to 1000 characters with `--instructions`. |
| `--instructions` | | How to say it (emotion, style), added after the voice description as one more sentence. |
| `--out` | | The output file: `.wav` (16-bit PCM, mono, 24 kHz) or `.pcm` (the same samples, raw little-endian 16-bit). |
| `--batch` | | A JSON Lines file of requests, one per line, decoded together. |
| `--out-dir` | `.` | With `--batch`: where a request without its own `out` goes. |
| `--seed` | `0` | Seed for sampling and for the decoder's noise. Take `k` of a request uses seed + `k`. |
| `--temperature` | `0.4` | Sampling temperature; `0` picks the most likely code every step. |
| `--top-p` | `0.9` | Nucleus sampling. |
| `--top-k` | `50` | Top-k sampling. |
| `--repetition-penalty` | `1.1` | Penalty on every token already in the prompt or the take. |
| `--max-tokens` | `2048` | New tokens per take, at most 2048 (about 25 s of audio). |
| `--slots` | `8` | Takes decoded at once. |
| `--ctx` | `3072` | Tokens per slot: the prompt plus the take's new tokens. |
| `--codec` | `f32` | SNAC decoder precision: `f32`, or `bf16` (bfloat16 convolutions, float32 sums). |
| `--no-kernels` | off | For checks only. Skips the kernel package and the bfloat16 path, so every weight runs on the engine's exact reference path. It is much slower and the audio differs from a default run. |
| `--cache-dir` | `~/.cache/mashbox/xla/maya1` | Where compiled programs are saved. |
| `--report` | | Also write what ran (takes, tokens, timings) to this JSON file. |
| `--voices` | | Print the voice names and their descriptions, then exit. |

The sampling defaults are those of Maya Research's example (temperature 0.4, top-p 0.9, repetition penalty 1.1, at most 2048 new tokens), with top-k 50, the default of the Transformers library that example runs on.

## Python

```python
from mashbox.models.maya1 import speech, weights

m = speech.Maya1.load(weights.stage())
res = m.speak([dict(text="Hello there. <laugh> It works.", voice="alloy", seed=0)])
speech.write_wav("hello.wav", res[0]["audio"])
```

`speak` takes a list of requests and decodes their takes together. With `on_audio=callback`, each take is decoded while it is generated and `callback(request index, take index, samples)` gets each piece of audio as soon as it is ready. Joined in order, the pieces have the same length as the whole-take decode and match it to within 1e-6 per sample. With `on_audio`, the result's `audio` is `None`: the pieces are the audio, so keep them.

## Performance

Measured with the `mashbox` engine, BF16 weights, the SNAC decoder in float32 and 8 slots of 3,072 tokens, with the programs already compiled unless a row says otherwise.

| What | TPU v6e-1 | TPU v5e-1 |
|---|---|---|
| First audio, one request, decoded while it is generated (5 requests) | 0.266 to 0.269 s | 0.450 to 0.452 s |
| Compute per second of audio, one request at a time, whole take (3 requests) | 0.536 to 0.545 s | 0.903 to 0.917 s |
| Compute per second of audio, one request at a time, decoded while it is generated (5 requests) | 0.54 to 0.59 s | 0.91 to 1.00 s |
| 8 requests at once | 40.9 s of audio in 8.25 s | 42.0 s of audio in 9.03 s |
| Command line, new chip, empty compile cache (whole command, one short take) | 48.3 s | 38.6 s |
| The same command again (compiled programs on disk) | 9.5 s | 10.1 s |
| Chip memory peak | 11.33 GB | 11.29 GB |

Two more requests run one at a time took 1.10 and 4.05 s (v6e-1) and 1.28 and 3.21 s (v5e-1) per second of audio: their take lengths compiled new programs while they ran. Each new take length compiles once; the programs are kept on disk.

## Quality

The reference is Maya Research's model in the Hugging Face Transformers library in float32, on the CPU, greedy with repetition penalty 1.1, on 38 prompts of up to 512 new tokens each.

- **Same codes as the reference.** With float32 activations and full-precision matrix products, the `mashbox` engine on the chip writes the same audio codes as the reference: 38 of 38 prompts, all 15,392 codes, identical.
- **The BF16 file, as `mashbox run maya1` runs it.** Read along the reference's own codes, its next-code distribution differs from the reference's by a mean KL divergence of 0.00255, and it picks the same top code at 91.4% of positions, including every position where the reference's top two codes are 0.5 logit or more apart. In 53% of positions the top two codes are closer than that, so a greedy BF16 take soon picks a different code and goes its own way.
- **SNAC decoder.** On the reference codes of all 38 prompts, with the same noise, the decoder here against the official `snac` package's decoder: 109.9 dB signal-to-noise ratio or more in float32 (the default), 34.2 dB or more with `--codec bf16`.

Same seed, same request, run alone: the same audio every time. A take decoded next to other takes in a batch can come out different from the same take run alone, because the chip's sums come out slightly differently with a different batch.

## Use it with your coding agent

Give your agent this: "Read https://github.com/mashbox-labs/mashbox/blob/main/AGENTS.md, create a v6e-1 TPU VM, install mashbox, and run `mashbox run maya1 --text "..." --voice alloy --out out.wav`; copy out.wav back and delete the VM." [AGENTS.md](../../AGENTS.md) covers the Google Cloud steps; the agent stops and asks you before anything that costs money.

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `x.mp3: the output is a .wav or .pcm file` | Only WAV and raw PCM are written. | Write `.wav` and convert it with any audio tool. |
| `voice: unknown voice 'name'` | The name is not one of the 13. | `mashbox run maya1 --voices`, or pass `--description`. |
| `max_tokens: max_tokens must be 1..2048` | One take holds at most 2048 new tokens. | Use a smaller value; longer text is split into takes on its own. |
| A take stops in the middle of a sentence | It reached `--max-tokens` before maya1 ended it. | Leave `--max-tokens` at 2048, or shorten the sentence. |
| `sha256 ... does not match the pinned ...` | A download was cut short or the file changed on disk. | Delete that file under `~/.cache/mashbox/maya1` and run again. |

## FAQ

- **Does it stream?** The command line writes whole files. From Python, `speak(..., on_audio=callback)` hands over audio while it is generated ([Performance](#performance) has the time to the first piece).
- **Can it speak other languages?** maya1 is trained for English; see [maya-research/maya1](https://huggingface.co/maya-research/maya1).
- **mp3 or opus?** Not in this version. Convert the WAV with any audio tool.
- **Is there an HTTP route?** Not in this version: maya1 runs from the command line and from Python.

## Credits and licences

maya1 is made by Maya Research: [maya-research/maya1](https://huggingface.co/maya-research/maya1/tree/b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572), Apache-2.0 as stated in its model card. The GGUF file in [mashbox-labs/maya1-TPU](https://huggingface.co/mashbox-labs/maya1-TPU) holds its weights, and the tokenizer is downloaded unchanged from maya-research/maya1.

The SNAC 24 kHz decoder is by Hubert Siuzdak: weights from [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz/tree/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec) (MIT), code from [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac/tree/8f79a718f1ad71f94f79999f0071348227aff22e) (MIT). The decoder in [snac.py](../../src/mashbox/models/maya1/snac.py) follows that code's decode path; its LICENSE is kept unchanged as [LICENSE-SNAC](../../src/mashbox/models/maya1/LICENSE-SNAC) beside it.

Our code in this repo: Apache-2.0 (see [LICENSE](../../LICENSE)).
