# Add a GGUF architecture

Adding an architecture is one model file and one CPU test. The test compares mashbox's logits with llama.cpp's CPU
logits on a small random GGUF file. It needs no chip and no secrets.

1. **Model file.** `src/mashbox/gguf/archs/<arch>.py`, named after the file's `general.architecture`. Follow
   llama.cpp's graph (`src/models/<arch>.cpp`) and its converter (`conversion/`) for anything the converter changes on
   the way in (for `qwen35`: norms stored as `w + 1` except `ssm_norm`, `ssm_a` stored as `-exp(A_log)`). It provides
   `Config(file)`, `load(file, config)`, `tensor_roles(file, config)`, `new_state(config, slots, ctx, extra_groups=0)`
   and `make_forward(config)`, with the same contract as `archs/qwen35.py` (`archs/dense.py` is the shorter example: one
   module for `llama`, `qwen2` and `qwen3`): the forward takes the engine's `keep`,
   `verify` and `hidden` arguments (see `tests/cpu/test_gguf_engine_hooks.py`). Weights go through `mashbox.ops` (`prepare`, `qmatmul`, `take_rows`), never through
   your own dequantization, so every block type and every TPU kernel works without more code.
2. **Register it** in `src/mashbox/gguf/archs/__init__.py` (`ARCHS`).
3. **Fixture.** Add a writer and an entry in `tools/gguf_fixtures/fixtures.py`: small shapes, seeded random weights,
   every metadata key llama.cpp reads for the architecture, written with `gguf`'s `GGUFWriter`.
4. **Reference logits and greedy ids.** With a llama.cpp checkout at the commit in `fixtures.py` and a CPU build in
   `build/`: `python tools/gguf_fixtures/fixtures.py make --llama-cpp ../llama.cpp --only <name>`, then
   `python tools/gguf_fixtures/greedy.py make --llama-cpp ../llama.cpp --only <name>`. Commit the `.json`,
   `.logits.f32`, `.greedy.json` and `.greedy.logits.f32` files (a few KB). Files `gguf` can write are rebuilt by the
   test from the seed and checked by sha256.
5. **Contract case.** Add the architecture to `CONTRACTS` in `tests/contracts/test_arch.py`: a float case at the
   architecture's own eps, an eps-sensitive case, and a block case when a fixture is quantized, each naming the
   features its fixture carries. `python3 -m pytest tests/contracts/test_arch.py` compares logits and greedy ids with
   llama.cpp's for a prefill and for decode steps, and fails while an architecture in `ARCHS` has no case or a fixture
   is in no contract. A `tests/cpu/test_gguf_<arch>.py` is only for what the contract does not cover, and its
   docstring says what that is.
6. **A real file**, if you have one: `python tools/gguf_check.py FILE.gguf`.

Open the pull request with the test output. Say in it which llama.cpp converter steps the file follows.
