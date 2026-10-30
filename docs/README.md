# MASHBOX docs

Docs for `mashbox`, the MASHBOX Labs package that runs open models faster and cheaper. The models in it run on Google Cloud TPU. The same pages are on [mashbox.gg/docs](https://mashbox.gg/docs).

- [Install](install.md): two lines on your own machine, from nothing to a running model
- [Quickstart](quickstart.md): one chip, one model, one answer, by hand with `gcloud`
- [Run on Google Cloud TPU](google-cloud.md): from a new Google account to a running model and a deleted chip
- [TPU or GPU](tpu-vs-gpu.md): when a TPU is the better pick
- [Supported models](supported-models.md): every model and file this version lists, and the command for each
- Models: [Qwen3.8-27B](models/qwen3-8-27b.md), [Hy-Nomas](models/hy-nomas.md), [Track2Rig](models/track2rig.md)
- [GGUF files](gguf.md): run a llama.cpp GGUF file on a TPU; [Unsloth fine-tunes](unsloth.md): run the GGUF you trained
- [Use with coding agents](agents-on-tpu.md): Claude Code, Codex and Pi on a model you serve
- [Command line](cli.md): every `mashbox` command and option
- [Speculative decoding](speculation.md): several tokens from one pass of the model, where it runs, and the `--speculation` options
- [Server API](server-api.md): every route, request field and response
- [TPU v6e and v5e](chips.md): which chip to pick
- [FAQ](faq.md): problems you can meet, why, and what to do
- [Glossary](glossary.md): the words these docs use
- [Licence](licence.md): which licence covers what
