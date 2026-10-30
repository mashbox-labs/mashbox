# Owners

This repo holds one shared core and one set of folders per model. A change to a model's folders touches only that model. A change to the core touches every model, so it gets an entry in [CORE-CHANGES.md](CORE-CHANGES.md) and keeps existing models working.

## Shared core

| Path | What it is |
|---|---|
| `src/mashbox/__init__.py`, `src/mashbox/__main__.py`, `src/mashbox/cli.py` | package entry and the `mashbox` command |
| `src/mashbox/hub.py` | downloads pinned by revision, sha256 checks |
| `src/mashbox/bshj.py` | reader for .bshj weight files |
| `src/mashbox/kernels.py` | loader for compiled kernel bundles (chip and JAX version checks, plain JAX otherwise) |
| `src/mashbox/runtime/` | LLM engine, sampler, chat template, OpenAI-compatible server |
| `src/mashbox/models/__init__.py` | the models folder |
| `models.json`, `src/mashbox/models.json` | the one registry (the second is the packaged copy, always equal to the first); each row belongs to its model |
| `pyproject.toml` | the `mashbox` package |
| `README.md`, `AGENTS.md`, `llms.txt`, `CONTRIBUTING.md`, `SECURITY.md`, `OWNERS.md`, `CORE-CHANGES.md` | repo pages |
| `LICENSE`, `LICENSE-KERNELS` | our licences |
| `docs/README.md`, `docs/SUMMARY.md`, `docs/google-cloud.md`, `docs/chips.md`, `docs/assets/` | shared docs |
| `templates/hf/` | Hugging Face card templates, one per model kind |
| `tools/` | `hf_github_sync_check.py` (keeps the views in sync with `models.json`), `tpu_status.py` (TPU badges) |
| `src/mashbox/ui/` | the MASHBOX module shell: the top bar, model bar, footer, tokens and fonts every module page loads (`ui/shell/README.md`); MASHBOX Chat, the chat module every chat model serves with `--ui` (`ui/chat/`) |
| `tests/cpu/test_core.py` | core checks |
| `tests/cpu/test_module_shell.py` | module page rules: every page loads the shell, copies none of it, and links to real pages |
| `.github/` | CI, issue forms, pull request template |

## Per model

| Model id | Port | App | Docs page | Upstream licence | Tests | Hugging Face card (generated) |
|---|---|---|---|---|---|---|
| `hy-nomas` | `src/mashbox/models/hy_nomas/` | `apps/hy-nomas/` | `docs/models/hy-nomas.md` | `licenses/hy-nomas/` | `tests/cpu/test_hy_nomas.py`, `tests/cpu/test_hy_nomas_port.py`, `tests/cpu/hy_nomas_tiny.py`, `tests/cpu/test_hy_nomas_fast_decoder.py`, `tests/cpu/test_hy_nomas_engine_handover.py`, `tests/cpu/test_hy_nomas_render.py`, `tests/tpu/hy_nomas_kernel_check.py`, `tests/tpu/hy_nomas_kernel_tolerances.json` | `hf/hy-nomas/README.md` |
| `track2rig` | `src/mashbox/models/track2rig/` | none; video step: `packages/track2rig-video/`, `examples/track2rig/` | `docs/models/track2rig.md` | `licenses/track2rig/` (our own model: Apache-2.0) | `tests/cpu/test_track2rig.py`, `packages/track2rig-video/tests/` | `hf/track2rig/README.md` |
| `qwen3-8-27b` | `src/mashbox/models/qwen3_8_27b/` | none | `docs/models/qwen3-8-27b.md` | `licenses/qwen3-8-27b/` | `tests/cpu/test_qwen3_8_27b.py`, `tests/cpu/qwen3_8_27b_checks.py`, `tests/tpu/qwen3_8_27b_smoke.py` | `hf/qwen3-8-27b/README.md` |
| `trellis-2` | `src/mashbox/models/trellis_2/` | none | `docs/models/trellis-2.md` | `licenses/trellis-2/` (Microsoft's LICENSE); `src/mashbox/models/trellis_2/glb_bake/LICENSE-CuMesh`, `src/mashbox/models/trellis_2/LICENSE-FlexGEMM` and `src/mashbox/models/trellis_2/LICENSE-Pillow-SIMD` (each project's own LICENSE, unchanged, beside the code that follows it) | `tests/cpu/test_trellis_2.py`, `tests/cpu/test_trellis_2_record.py`, `tests/cpu/test_trellis_2_page.py` | `templates/hf/card-image-to-3d.md` |

Each model's compiled kernel bundle is not in this repo. It lives in that model's Hugging Face repo under `kernels/`, pinned by sha256 in the model's `models.json` row.

## Rules

1. A model change edits only that model's folders and its own row in `models.json` (then copy `models.json` to `src/mashbox/models.json`).
2. Generated views are never edited by hand: the models table in `README.md` and the model list in `llms.txt` (between the `models:start` and `models:end` markers), and every `hf/<id>/README.md`. Re-render them with `python3 tools/hf_github_sync_check.py render --repo . --write`.
3. A core change gets an entry in [CORE-CHANGES.md](CORE-CHANGES.md): what changed, why, and how existing models keep working. Run `python3 -m pytest tests/cpu` before sending it.
4. Upstream licence files in `licenses/` stay exactly as their authors wrote them.
5. A new model adds `src/mashbox/models/<id with underscores>/`, `docs/models/<id>.md`, `licenses/<id>/`, tests whose names carry the id, and its row in `models.json`. Adding its id to `RUNNABLE` in `cli.py` once its runtime is in is a core change.
