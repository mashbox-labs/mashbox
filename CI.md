# CI

Three tiers. Each change runs only what it can break, and every test has to earn its place.

| Tier | What it checks | Where | Runs on |
|---|---|---|---|
| **T0** | registry and its packaged copy, generated views, imports, test docstrings, syntax errors and undefined names, sign-off | GitHub, pytest and ruff only, about 10 s | every push and pull request |
| **T1** | CPU tests against references (NumPy, transformers, plain JAX), only for the areas the change touches | GitHub, one job per area, CPU torch | when code in an area changes |
| **T2** | the same code on a real TPU: kernels vs plain JAX, version pins, the CPU suite again on the chip | a fresh on-demand TPU v6e-1 that is deleted after the run | started by a maintainer for a commit |

## What runs on which change

`tools/ci_select.py` picks the T1 areas from the changed files. Areas come from `models.json`, so a new model needs no CI edit. One check, `t1`, stands for all of T1: it passes when every picked area passed, or when no area was picked.

| Change | T0 | T1 |
|---|---|---|
| docs, `*.md`, `licenses/`, `hf/`, `templates/`, issue forms, the docs-lint allow counts (`tools/docs_lint_allow.txt`) | yes | none |
| a `tests/tpu/` file | yes | the area of any CPU test that reads it (the kernel check's CPU stand-in test), else none |
| `models.json` top-level fields (views only) | yes | none |
| a model's `models.json` row | yes | that model, plus core tests (the CLI and `mashbox tpu` tests read the rows) |
| `gguf` rows in `models.json` | yes | `gguf`, plus the `gguf_rows` job: the CPU tests outside `gguf` that read the rows (catalog, model lists, card and docs sync, speculation records) |
| `src/mashbox/models/<id>/`, `apps/<id>/`, `tests/cpu/test_<id>*.py` | yes | that model |
| core: `src/mashbox/*.py`, `src/mashbox/runtime/`, `pyproject.toml`, `tests/conftest.py`, `.github/workflows/`, `tools/` | yes | every model, plus core tests |
| any path not listed here | yes | every model (a new kind of file is never silently untested) |

Run the same thing locally:

```bash
PYTHONPATH=src python3 -m pytest -m t0 tests/cpu/test_core.py          # T0, needs only pytest
python3 tools/ci_select.py --base origin/main                           # which areas your change selects
python3 -m pytest tests/cpu -m "hy_nomas and not t0" --durations=15     # T1 for one area
```

Markers come from file names in `tests/conftest.py` (no test file has to move): `<model id with underscores>` or `core`, and `t0` for `tests/cpu/test_core.py`.

## Shards and parts

Each T1 area runs as `SHARDS[area]` jobs (core 12, qwen3_8_27b 4), split by the seconds per test file in `SECONDS`, so no job runs much past a few minutes. A file whose one shared fixture outlasts a shard is in `SPLIT` and runs as parts: `tools/ci_select.py --args-of AREA --shard i/n` passes `pytest --ci-part=FILE@K/M`, and `tests/conftest.py` keeps part K's tests. A module can group its tests with `ci_part(item)` (the Qwen3.8-27B checks group by gate script, so each script runs once, in one part). The parts of a file hold every test exactly once; a part with no test is an error. Every leg prints its seconds per file at the end of its summary: copy them into `SECONDS` when a shard runs long.

## Contracts

The `contracts` job runs `tests/contracts` on every pull request, the same for every change: each supported kind (`op/<ggml type>`, `tok/<tokenizer>`, `arch/<architecture>`, `path/run`) against its reference on CPU, with no model download and no chip. It writes each contract's status and fingerprint to `contracts.json` and to the job summary. A new contract joins the job when its test lands in `tests/contracts`. A change to `tests/contracts` alone runs T0 and this job.

## The sweep

`.github/workflows/sweep.yml` runs every T1 leg of every area nightly, by hand, and on every `v*` tag, with the tests marked `timing` included. A `timing` test passes or fails on wall-clock time, so it depends on the runner's speed and never gates a pull request. The one such check is the `"merge": false` latency gate in `tests/cpu/engine_batch_checks.py`: B's first token arrives before A ends, and B's p95 first-token time stays under double its time without the field. Its other condition, no compile in the measured runs, is the class of the serve-time compile gates in the same script, and those stay in pull request CI. A red sweep is fixed forward on main and holds the release it ran for. Chip runs and speed records stay outside GitHub (T2 and the release records).

## Can the CPU tiers stand in for the chip?

No. A green CPU run never proves a green chip. T0 and T1 check the logic against references: shapes, masks, padding, LoRA math, the sampler, the server, the registry, the kernel loader. Only a chip catches TPU compiler limits, version skew between kernels and the TPU runtime, TPU number formats, chip memory at real sizes, and speed. That is why T2 exists and why it runs on a real chip, not an emulator.

## T2: the chip tier

One run, on one fresh on-demand v6e-1:
1. **versions:** jax, jaxlib and libtpu must equal the kernel pins in `models.json`.
2. **kernels:** every `tests/tpu/<model>_kernel_check.py` compares each compiled kernel with the plain JAX version of the same op on the chip. `tools/ci_tpu_gate.py` applies the gates from `tests/tpu/<model>_kernel_tolerances.json`: every kernel checked, no non-finite values, and per op bit-exact or a relative L2 error at or under its written tolerance. A kernel with no rule fails.
3. **tiny:** the CPU suite again on the chip, one pytest process per area (`JAX_PLATFORMS=tpu,cpu`: the TPU is the default device, f32 matmuls at full precision). Tests that need CPU-only stand-ins carry `@pytest.mark.cpu_only`.

The chip gets the commit and the kernel bundles, never a token. The run is posted on the commit as the status `tpu/v6e-1`. The chip is deleted when the run ends, on any error, and at a hard deadline.

**How to trigger it:** say in the pull request that the change needs a chip run; a maintainer starts T2 for the commit.

**Cost:** on-demand v6e-1 list price is about $2.70 an hour. A T2 run bills about 10.1 minutes, about $0.46 at that price: 2.6 min to create the chip, 13 s to install, 229 s for the 28 Hy-Nomas kernels, 46 s for the CPU suite on the chip, 2.2 min to delete. Getting a chip can take longer than the run: on-demand v6e-1 can have no capacity in five zones for about 30 minutes, and a failed create bills nothing.

## Every test earns its place

1. Each test module opens with a docstring that names the failure it guards. T0 fails a module without one.
2. The owner is the area in the file name (`test_<model id>...` belongs to that model, any other name to core), so the owners table in [OWNERS.md](OWNERS.md) applies.
3. A test asserts. A script that only prints is a tool until it has gates.
4. A T1 test takes at most 10 s on the CI runner. Longer needs `@pytest.mark.slow` and the reason in its docstring. Every run prints the 15 slowest tests.
5. Expensive setup is shared through a fixture, and loops over a JAX function go through `jax.jit`.
6. A check that passes or fails on wall-clock time gets `@pytest.mark.timing`: it runs in the sweep, not in pull request CI.
7. No retries. A flaky test gets `@pytest.mark.quarantine(reason="...", until="YYYY-MM-DD")` at most 14 days out. CI deselects it; after that date collection fails until it is fixed or deleted.
8. No duplicates: when two tests guard the same failure, keep the cheaper or more precise one.
9. Removing code removes its tests in the same change. A quiet test is not deleted for being quiet.
10. A bug fix brings a regression test at the cheapest tier that catches it (T0, then T1, then T2).
11. Every numeric comparison names its reference and its tolerance. Loosening a tolerance needs the measured cause written next to it.

## Adding a test

- Put it in `tests/cpu/test_<model id with underscores>_<what>.py` for a model, or a name without a model id for core.
- Open the module with a docstring: the failure it guards, and the reference it compares against.
- Pick the cheapest tier that catches the failure: a registry or file rule goes in `tests/cpu/test_core.py` (T0), logic goes in T1, chip-only behaviour goes in a `tests/tpu` check with a tolerance file.
- Run it with `--durations=15` and keep it under 10 s, or mark it `slow` with the reason.
- If you change `models.json`, copy it: `cp models.json src/mashbox/models.json` (T0 checks they are equal and names the first field that differs).
