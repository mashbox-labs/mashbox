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
2. Run the CPU tests: `python3 -m pytest tests/cpu`. CI runs them on every pull request.
3. If you change `models.json`, copy it to `src/mashbox/models.json` and regenerate the files made from it: `python3 tools/hf_github_sync_check.py render --repo . --write`.
4. Sign off every commit with `git commit -s`. This is the [Developer Certificate of Origin](https://developercertificate.org): you confirm you have the right to send the code. CI checks it.
5. TPU time costs money, so maintainers run `tests/tpu` on real chips before each release, not on every pull request. Say in the pull request if your change needs a chip run.

## AI tools

You can use them. You are responsible for every line you send and must be able to explain it in review.

## Licence of contributions

Contributions are accepted under the licence in [LICENSE](LICENSE).

## Conduct

Be direct and civil. Spam, harassment and off-topic promotion get removed.
