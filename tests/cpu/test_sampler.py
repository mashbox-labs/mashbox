"""The shared sampler keeps exactly the tokens top_k, top_p and min_p allow, no fewer and no more.

Guards an off-by-one at a cut: keeping k - 1 tokens passed every other test on 2026-10-28 (the Qwen T7 check only asks
that drawn tokens fall inside the allowed set). Reference: the allowed sets worked out by hand from the logits below;
draws are seeded, so the drawn sets are fixed.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mashbox.runtime import sampler as SP

V, STEPS = 64, 300
# five close logits, the rest far below; softmax of the five: 0.242 0.219 0.198 0.180 0.162
TOP5 = np.array([5.0, 4.9, 4.8, 4.7, 4.6], np.float32)


def drawn(rows, logits):
    x = jnp.asarray(np.stack(logits))
    sp = SP.to_device(SP.pack(rows, len(rows)))
    draw = jax.jit(jax.vmap(lambda st: SP.sample(x, sp, jnp.full((len(rows),), st, jnp.int32))))
    out = np.asarray(draw(jnp.arange(STEPS, dtype=jnp.int32)))
    return [set(out[:, i].tolist()) for i in range(len(rows))]


def test_cuts_keep_exactly_the_allowed_tokens():
    near = np.full(V, -30.0, np.float32)
    near[:5] = TOP5
    tied = np.full(V, -30.0, np.float32)
    tied[:4] = 5.0
    rows = [dict(temperature=1.0, top_k=3, seed=1),       # the 3 largest
            dict(temperature=1.0, top_p=0.5, seed=2),     # smallest top set with mass >= 0.5: 0.242 + 0.219 + 0.198
            dict(temperature=1.0, min_p=0.8, seed=3),     # p >= 0.8 x 0.242 = 0.194: tokens 0, 1, 2
            dict(temperature=1.0, top_k=2, seed=4)]       # ties at the cut are kept (sampler.py docstring)
    got = drawn(rows, [near, near, near, tied])
    assert got == [{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2, 3}], got
