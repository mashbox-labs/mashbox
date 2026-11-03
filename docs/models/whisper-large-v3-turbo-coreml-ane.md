# Whisper large-v3-turbo on the Apple Neural Engine

This page covers OpenAI's [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) as a Core ML
model whose encoder and decoder run on the Apple Neural Engine. It is one of our ports. It runs on Apple hardware, not on
a TPU, so `mashbox run` does not start it. You run it from Swift with our
[whisper-ane](https://github.com/mashbox-labs/whisper-ane) package, as shown below.

## What it is

Speech in, text out, in any of the 100 language tokens of Whisper large-v3-turbo (listed in `manifest.json`). The
encoder and the decoder are each one Core ML program that takes a 10 s or a 30 s window, so the Neural Engine holds a
single copy of the weights.

The files are on Hugging Face at
[mashbox-labs/whisper-large-v3-turbo-coreml-ane](https://huggingface.co/mashbox-labs/whisper-large-v3-turbo-coreml-ane).
It is the model our Free Flow dictation app uses for languages such as Thai, Korean, Japanese and Chinese.

## Measured

On a MacBook Air (M5), against full-precision [whisper.cpp](https://github.com/ggml-org/whisper.cpp) v1.9.5 with its
Core ML encoder, on the same 500 FLEURS test clips (100 per language). Time is end to end, from audio to text, with
the model already loaded.

| Median ms per clip | This build | whisper.cpp + Core ML |
|---|---|---|
| English | 176 | 519 |
| Japanese | 605 | 578 |
| Korean | 578 | 568 |
| Chinese (Mandarin) | 574 | 569 |
| Thai | 767 | 766 |

English is about 2.9x faster. Japanese, Korean and Chinese are
about 1 to 5% slower, and Thai is even.

| | |
|---|---|
| Neural Engine memory | 833 MB with both windows loaded |
| Whole process | at most 1,305 MB, against at least 1,946 MB for whisper.cpp with Core ML |

The two memory figures come from different tools. The one used for whisper.cpp does not count memory the Neural Engine
holds, so its real figure is at least the one shown. The model card has the error rates and how each figure was read.

## What you need

| | |
|---|---|
| Device | iPhone or iPad with iOS 18 or iPadOS 18, or a Mac with macOS 15, or later |
| Measured on | MacBook Air (M5) |

The first load on a device prepares the model for its Neural Engine. Later loads come from the system cache.

## In your own app

Add the Swift package. In Xcode: File > Add Package Dependencies, then `https://github.com/mashbox-labs/whisper-ane`.
Or in `Package.swift`:

```swift
.package(url: "https://github.com/mashbox-labs/whisper-ane", from: "0.1.0")
```

Then:

```swift
import WhisperANE

let whisper = try await WhisperANE.load()
let text = try await whisper.transcribe(URL(fileURLWithPath: "clip.m4a"))
```

The package downloads the files from Hugging Face at a pinned revision, checks each one against its SHA-256 and runs
them on the Neural Engine. Its [README](https://github.com/mashbox-labs/whisper-ane) covers language choice, raw
samples and shipping the model inside your app.

## Licence

Whisper large-v3-turbo is OpenAI's model under the MIT licence, as OpenAI states it. The licence file is in the Hugging
Face repo, unchanged. The whisper-ane package is under Apache-2.0. Compared against whisper.cpp by ggml-org; evaluation
clips from FLEURS by Google.
