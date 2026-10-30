# LFM3-1.7B on the Apple Neural Engine

This page covers Liquid AI's [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) as a Core ML model that runs on the
Apple Neural Engine (What you need lists the devices). It is one of our ports. It runs on Apple hardware, not on a TPU, so
`mashbox run` does not start it. You run it with Core ML from Swift, as shown below.

## What it is

The weights are Liquid AI's own quantization-aware release, `LFM3-1.7B-QAD-Q4_0.gguf` in
[LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF). The layers use 4-bit weights and the shared
embedding and output layer use 6-bit weights. They are carried over exactly, so the model computes with Liquid's
weights in float16.

We rebuilt the model as one stateful Core ML program with three functions that share one cache:

- `prefill`: 8 tokens per call.
- `decode`: 1 token per call.
- `verify`: 8 tokens per call, for speculative decoding.

The files are on Hugging Face at
[mashbox-labs/LFM3-1.7B-CoreML](https://huggingface.co/mashbox-labs/LFM3-1.7B-CoreML).

## Measured

On a Mac with an Apple M5 (macOS 27.2):

| | |
|---|---|
| Size | 1,059.6 MB, 4.9 bits per weight |
| Neural Engine | 91.9 % of the compute ops in `decode`. The rest look up the input token's embedding on the CPU |
| Same output as Liquid's weights | 98.22 % next-token agreement on the Neural Engine with Liquid's weights in full precision, over 2,811 positions of 44 prompts. The same weights in float16 PyTorch agree 99.89 %. On the CPU, the same Core ML model agrees on all 400 positions of the first 6 prompts |
| Greedy text | 25 of 44 prompts give word-for-word the same reply as the full-precision reference |
| Speed | 11.1 tokens/s decoding and 47.7 tokens/s for the prompt, one conversation on an idle Mac |
| First load | 158 s the first time on a device, while macOS compiles the model for the Neural Engine. Later loads take about 2 s |

## What you need

| | Tested | Not tested |
|---|---|---|
| Device | Mac with Apple M5, macOS 27.2, Xcode with Swift 6 | iPhone, iPad, other Macs, older macOS |
| Prompts | English | the other languages LFM3-1.7B supports |

## Run

```bash
python3 -m venv hf-env && source hf-env/bin/activate
pip install -U huggingface_hub
hf download mashbox-labs/LFM3-1.7B-CoreML --local-dir LFM3-1.7B-CoreML
cd LFM3-1.7B-CoreML/example
swift build -c release
.build/release/lfm3-generate .. "Write two sentences about the ocean."
```

`lfm3-generate` loads the model with `computeUnits = .cpuAndNeuralEngine` and formats the message with the model's chat
template (thinking off). It then decodes greedily and prints the reply as it is generated.

It also takes a system prompt and a cap on the reply length, which is 256 tokens unless you set it:

```text
lfm3-generate MODEL_DIR "message" [--system TEXT] [--max-new N]
```

The first load on a device is slow because macOS compiles the model for the Neural Engine. Later loads use the cached
result.

## In your own app

The model card lists every input and output of the three functions. `example/Sources/lfm3-generate/main.swift` builds
each one. The context is 2048 tokens. Keep `enable_thinking=false` in the chat template.

## Licence

LFM3-1.7B is Liquid AI's model under the LFM Open License v1.0 (`lfm1.0`), as Liquid AI states it. The licence file is
in the Hugging Face repo, unchanged.
