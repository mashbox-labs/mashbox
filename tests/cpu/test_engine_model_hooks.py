"""The engine asks the model for its own sampler and placement when the model has them (tensor parallel: shard.py's
sampler works on vocabulary-split logits and commit() places host arrays on the mesh). A model without them runs the
shared sampler and the engine's own placement, as before; with a sampler that is the shared one, tokens are unchanged."""
import types

import pytest

pytest.importorskip("jax")

from engine_standin import MODEL, config, weights  # noqa: E402
from mashbox.runtime import sampler as SP  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402

SAMPLED = dict(temperature=0.7, top_p=0.9, top_k=20, seed=3)


def counting_model():
    calls = dict(sample=0, penalise=0, no_penalty=0, commit=0)

    def wrap(name):
        def f(*a, **k):
            calls[name] += 1            # counted while tracing: the engine's programs were built with this sampler
            return getattr(SP, name)(*a, **k)
        return f
    sampler = types.SimpleNamespace(**{n: wrap(n) for n in ("sample", "penalise", "no_penalty")})

    def commit(x):
        calls["commit"] += 1
        return x
    return types.SimpleNamespace(new_state=MODEL.new_state, make_forward=MODEL.make_forward, sampler=sampler,
                                 commit=commit), calls


def run(model):
    e = Engine(config(), weights(), model, CTX=256, S=4, K=4)
    return (e.generate([[5, 6, 7], [9, 10]], max_new=6)["ids"],
            e.generate([[5, 6, 7], [9, 10]], max_new=6, samp=[SAMPLED, dict(SAMPLED, seed=4)])["ids"])


def test_model_sampler_and_commit_are_used():
    m, calls = counting_model()
    got = run(m)
    assert calls["sample"] > 0 and calls["penalise"] > 0 and calls["no_penalty"] > 0, calls
    assert calls["commit"] > 0, calls
    assert got == run(MODEL)            # the same sampler underneath: same tokens, greedy and sampled
