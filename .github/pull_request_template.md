## What and why

## How you tested

- [ ] `python3 -m pytest tests/contracts`, and the `tests/cpu` areas `python3 tools/ci_select.py --base origin/main` selected
- [ ] Ran on a TPU (which chip, what command), or say that it needs a maintainer chip run

## Checklist

- [ ] Every commit is signed off (`git commit -s`)
- [ ] If `models.json` changed: copied it to `src/mashbox/models.json` and ran `python3 tools/hf_github_sync_check.py render --repo . --write`
- [ ] Docs updated if behaviour changed
