# Contributing

## What we take

- **Bug reports** with the exact command, the chip, the versions (`~/.local/bin/mashbox env` prints them) and the full error.
- **Fixes to the open code:** model ports, the CLI, demo apps, docs and tests.
- **Model requests:** open a "Model request" issue with the model's Hugging Face link.
- **Small docs fixes** need no issue first.

## What we can't take

- **Kernel changes.** The kernel source is not in this repo; each model's kernels ship as a compiled bundle. Report kernel bugs and slowdowns as issues with a way to reproduce them.
- **Edits to upstream licence files** in `licenses/`. They stay exactly as their authors wrote them.
- **New dependencies** without a reason in the pull request.

## Pull requests

1. One change per pull request. For anything bigger than a bug fix, open an issue first.
2. Run what CI runs. `python3 -m pytest tests/contracts` is the check every pull request gets: each supported architecture, tensor type and tokenizer against its reference on a CPU, in a few minutes, with no model download. Then `python3 tools/ci_select.py --base origin/main` prints the `tests/cpu` areas your change selects; run those ([CI.md](CI.md) has the commands). The whole `tests/cpu` suite is the nightly sweep, not a pull request check.
3. If you change `models.json`, copy it to `src/mashbox/models.json` and regenerate the files made from it: `python3 tools/hf_github_sync_check.py render --repo . --write`.
4. Sign off every commit with `git commit -s`. This is the [Developer Certificate of Origin](https://developercertificate.org): you confirm you have the right to send the code. CI checks it.
5. TPU time costs money, so maintainers run `tests/tpu` on real chips before each release, not on every pull request. Say in the pull request if your change needs a chip run.

## Add a GGUF architecture or a GGUF file

Support is by type, not by file: a GGUF file runs when its architecture, every tensor type in it and its tokenizer pass their contracts in `tests/contracts`. So there are two kinds of contribution.

**A new architecture** (a file whose `general.architecture` mashbox does not read): follow [Add a GGUF architecture](docs/add-an-architecture.md). In short: one model file under `src/mashbox/gguf/archs/`, a tiny random fixture with llama.cpp's reference logits and greedy ids, and a case for it in `CONTRACTS` in `tests/contracts/test_arch.py`. `python3 -m pytest tests/contracts` is the gate: it fails while an architecture has no case or a fixture is in no contract. No chip is needed.

**A file of an architecture mashbox already reads:** open a "Model request" issue with the Hugging Face repo, the revision, the file name and its sha256, and the output of `mashbox inspect <repo>:<tag>` if you have the package installed. Catalog rows are generated from sealed check records, never typed into `models.json`, and a maintainer does not merge a pull request that adds a row by hand. A maintainer runs the file's short check on a TPU (that costs money, so it is not part of pull request CI), seals the record, and the row appears on every surface from `models.json` at the next render.

Kernels for a tensor type the engine runs on plain JAX only are the maintainers' work. Ask for one in an issue with the file and the speed you measured.

## AI tools

You can use them. You are responsible for every line you send and must be able to explain it in review.

## Licence of contributions

Contributions are accepted under the licence in [LICENSE](LICENSE).

## Conduct

Be direct and civil. Spam, harassment and off-topic promotion get removed.
