"""The draft-model drafter (runtime/drafters/draft_model.py): a second dense model drafts for the target inside the
batched step. Tiny f32 GGUF fixtures (tools/gguf_fixtures write_dense), CPU, no network.

  self    the target as its own draft: output equals the plain engine's and almost every draft is accepted (a draft cache
          that misses the prompt, a committed token or a position cannot do that)
  other   a draft model with other weights (seed 7): greedy and sampled rows equal the plain engine's, and after the run
          the draft cache of the row's slot holds exactly the row's prompt and committed tokens (a fresh prefill of them)
  key     the gate key: kind draft_model, the draft weights' sha256, the speculation step version
  vocab   a draft model with another vocabulary is refused
"""
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import gguf  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402

from mashbox.runtime.engine import Engine  # noqa: E402
from mashbox.runtime import spec_step as SS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_fx = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_fx)
_fx.loader.exec_module(FXT)
K = 3


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    from mashbox.gguf import open_model
    wd = tmp_path_factory.mktemp("draft_model")
    out = {}
    for tag, seed in (("target", 0), ("other", 7)):
        path = str(wd / f"llama-{tag}.gguf")
        FXT.write_dense(path, arch="llama", seed=seed)
        f, A, c = open_model(path, act=jnp.float32)
        W, _ = A.load(f, c, use_kernels=False)
        out[tag] = (c, W, A)
    return out


def DM():
    from mashbox.runtime.drafters import draft_model
    return draft_model


def ids_of(res):
    """generate() -> dict(ids=[one list of token ids per prompt], ...): the token ids, each a non-empty list of ints"""
    ids = [[int(t) for t in r] for r in res["ids"]]
    assert ids and all(ids), ids
    return ids


def prompts(V, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(1, V, int(rng.integers(5, 40))).tolist() for _ in range(n)]


def pair(models, draft, **kw):
    c, W, A = models["target"]
    dc, dW, dA = models[draft]
    kw = dict(dict(CTX=256, S=4, K=4), **kw)
    dr = DM().DraftModel(dA, dc, dW, K, declared=dict(id="llama-" + draft, weights_sha256="ab" * 32))
    return Engine(c, W, A, **kw), Engine(c, W, A, drafter=dr, draft_k=K, spec_rounds=3, **kw)


def test_self_draft_equals_plain_and_accepts_almost_every_draft(models):
    plain, spec = pair(models, "target")
    ps = prompts(models["target"][0].V, 4, 1)
    want, got = ids_of(plain.generate(ps, max_new=40)), ids_of(spec.generate(ps, max_new=40))
    assert got == want
    st = spec.spec_stats
    assert not st.get("faults") and st["rounds"] > 0
    per_round = st["tokens"] / st["rounds"]
    assert per_round >= 0.9 * (K + 1), st       # K + 1 when every draft is accepted (the last round of a row may be short)


@pytest.mark.parametrize("samp", [None, dict(temperature=0.8, top_k=20, seed=5)])
def test_other_draft_equals_plain(models, samp):
    plain, spec = pair(models, "other")
    ps = prompts(models["target"][0].V, 4, 2)
    sm = None if samp is None else [samp] * len(ps)
    assert ids_of(spec.generate(ps, max_new=30, samp=sm)) == ids_of(plain.generate(ps, max_new=30, samp=sm))
    assert not spec.spec_stats.get("faults") and spec.spec_stats["rounds"] > 0


def test_draft_cache_holds_only_the_rows_tokens(models):
    """one row on one slot: after every speculation call, the draft cache at positions 0..pos-1 equals a fresh prefill of
    the row's prompt and committed tokens (the last committed token is the one fed next); rejected drafts left nothing
    there. After a plain call the row's cache is stale by design (hist_ok False: rebuilt before its next speculation
    call), so only calls that leave hist_ok True are checked."""
    _, spec = pair(models, "other", S=1)
    p = prompts(models["target"][0].V, 1, 3)[0]
    dc, dW, dA = models["other"]
    checked = []

    def check(ids, finish):
        r = spec.rows[0]
        if r is None or not r.get("hist_ok") or not r["ids"]:
            return None
        seq = list(r["prompt"]) + list(r["ids"][:-1])
        fresh = spec.drafter.fill(jax.tree.map(jnp.asarray, dA.new_state(dc, 1, spec.CTX)), 0, seq)
        n = len(seq)
        for got, want in zip(spec.dstate[0], fresh[0]):         # K, then V
            np.testing.assert_allclose(np.asarray(got[:, 0, :n]), np.asarray(want[:, 0, :n]), rtol=1e-5, atol=1e-5)
        checked.append(n)
        return None
    ids_of(spec.generate([p], max_new=30, on_step=check))
    assert spec.spec_stats["rounds"] > 0 and len(checked) >= 3, (checked, spec.spec_stats)


def test_key_names_the_draft_weights_and_the_step_version(models):
    dc, dW, dA = models["other"]
    k = DM().DraftModel(dA, dc, dW, K, declared=dict(id="d", weights_sha256="cd" * 32)).key()
    assert k["kind"] == "draft_model" and k["weights_sha256"] == "cd" * 32
    assert k["spec_step_version"] == SS.SPEC_STEP_VERSION and k["k"] == {"k_max": K}


def test_another_vocabulary_is_refused(models):
    dc, dW, dA = models["other"]
    dr = DM().DraftModel(dA, dc, dW, K)
    eng = SimpleNamespace(c=SimpleNamespace(V=int(dc.V) + 1), S=1, CTX=128, _committed=lambda x: x)
    with pytest.raises(ValueError, match="vocabulary"):
        dr.state(eng)
