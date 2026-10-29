# Supported models

Every model and file this version of mashbox runs on a TPU, by publisher, then model, then file. `mashbox models` prints the same list from the package's own catalog, and `mashbox models --json` prints it as JSON.

```bash
mashbox models
```

Each row names the chips it runs on and the command that runs it. A GGUF link opens the exact file revision that ran.

<!-- catalog:start -->
## MASHBOX Labs ([Hugging Face](https://huggingface.co/mashbox-labs))

| Model | Format or file | Level | Runs on | Speed | Run |
|---|---|---|---|---|---|
| **[Hy-Nomas](models/hy-nomas.md)**<br>Made by Tencent Hunyuan, ported to TPU by MASHBOX Labs | TPU port<br>World model: text and movement keys in, video out | MASHBOX build | TPU v6e-4 |  | `~/.local/bin/mashbox run hy-nomas --ui` |
| **[Qwen3.8-27B](models/qwen3-8-27b.md)**<br>Made by Qwen, ported to TPU by MASHBOX Labs | BSHJ build: BQ8_R<br>LLM: text in, text out | MASHBOX build | TPU v5e-4, TPU v6e-1, TPU v6e-4 | 38.1 vs 31.4 tok/s on TPU v6e-1, vs vLLM on the same chip, 1 request at a time | `~/.local/bin/mashbox run qwen3-8-27b` |
| **[Track2Rig](models/track2rig.md)** (Experimental)<br>Trained by MASHBOX Labs | MASHBOX model<br>Rigging: 3D point tracks in, rig (JSON and glTF) out | MASHBOX build | TPU v6e |  | `mashbox run track2rig examples/track2rig/tracks/grison_walk.npz -o grison_rig` |

No GGUF file is listed in this version.
<!-- catalog:end -->

## Levels

| Level | What the row means |
|---|---|
| MASHBOX build | One of our own TPU builds, with weights on Hugging Face and its own model page. |
| Tuned | A file that passed its chip test on the chips in its row, with a part tuned for it. |
| Tested | A file that passed its chip test on the chips in its row, with this version of mashbox. |

A GGUF file with no row here can still run: `mashbox inspect FILE` says whether it runs here, and why not, before anything loads. [GGUF files](gguf.md#what-runs) says what runs.
