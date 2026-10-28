# hy_nomas

Port of Tencent Hunyuan's Hy-Nomas world model. How to run it, chips and numbers: [docs/models/hy-nomas.md](../../../../docs/models/hy-nomas.md). The browser app is in [apps/hy-nomas](../../../../apps/hy-nomas); `page.html` here is a copy of it, so the installed package can serve it.

| File | What it does |
|---|---|
| `app.py` | `python3 -m mashbox run hy-nomas` and `download hy-nomas`: options, chip check, loading, browser app, `--out` clips, `--stub` |
| `dit.py` | the main model (double-stream blocks and the camera action module) in JAX, split over the chips by attention heads |
| `runtime.py` | the window loop: noise schedule per preset, history kept as int8, text items, one compiled program per shape |
| `ops.py` | the steps that have a compiled kernel in the bundle, each with its plain JAX version, and the kernel-or-plain dispatch |
| `kernels.py` | loads this model's compiled kernel bundle after checking hashes, JAX version and chip; plain JAX otherwise |
| `buckets.py` | fixed text lengths and the attention groups that mask padding, so new text never compiles |
| `decoder.py` | the pixel video decoder in JAX, one copy per decoder chip, decoding overlapping chunks |
| `text_encoder.py` | the text encoder (26 layers of Qwen2.5-VL-7B's language model) in JAX, layer groups over the chips |
| `upstream.py` | Tencent's own code at a pinned commit: prompt template and crop, timed-action offsets, camera normalization |
| `weights.py` | downloads at pinned revisions with sha256 checks; LoRA adapter merged into the main model at load |
| `cameras.py` | keys held per video frame to the camera path of each window |
| `prompts.py` | worlds and timed actions in Tencent's structured prompt format |
| `server.py` | the app server on 127.0.0.1: pacing, decoder workers, H.264 stream over a WebSocket, HTTP API, CPU stand-in engine |
| `chip.py` | the chip name shown in the app, read from JAX and the VM |
| `defaults.json` | chip layout, presets, decoder overlap, camera rates, text lengths, port |
| `worlds.json` | the preset worlds |
| `upstream.json` | pinned upstream files: Tencent's code commit and tree hash, Hugging Face revisions, sizes and sha256 |
| `page.html` | the browser app (copy of `apps/hy-nomas/page.html`; a test checks they match) |
