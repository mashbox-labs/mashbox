"""Compile key and component set (runtime/compile_key.py; INTERFACES-v1 section 6): every listed field changes the key,
weight values and dict order do not, and a shipped cache built for another component set is refused."""
import copy
import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np                      # noqa: E402
import pytest                           # noqa: E402

from mashbox.runtime import compile_key as CK      # noqa: E402

FL = {"XLA_FLAGS": [], "LIBTPU_INIT_ARGS": [], "jax_threefry_partitionable": True, "jax_default_matmul_precision": None,
      "xla_allow_excess_precision": None}
CHIP = {"family": "tpu", "accelerator_type": "TPU v6 lite", "devices": 1}
SW = {"jax": "0.11.2", "jaxlib": "0.11.2", "libtpu": None, "mashbox": "0.1.2", "engine_commit": "x",
      "FORWARD_ABI": CK.FORWARD_ABI, "step_program_version": CK.STEP_PROGRAM_VERSION}
DRAFTER = {"kind": "mtp", "id": "qwen3-8-27b-mtp@1", "weights_sha256": "a" * 64, "k": 2}


def eng(**over):
    e = types.SimpleNamespace(CTX=2048, S=8, K=8, chunk=512, tokens_per_call=2048, attn_bytes=2e9, budget_busy=128,
                              budget_idle=1024, prefix_cache=True, host=object(), kv_block=2048, trace=False,
                              kernels=["decode@0.1.0"], DK=0)
    for k, v in over.items():
        setattr(e, k, v)
    return e


def key(e=None, **kw):
    a = dict(model_id="qwen3-8-27b", config_sha="c" * 64, weights_layout_sha="l" * 64, kernel_bundle_sha="k" * 64,
             chip_=CHIP, sw_=SW, flags_=FL, drafter=None)
    a.update(kw)
    return CK.for_engine(e or eng(), **a)


BASE = key()


@pytest.mark.parametrize("name,k", [
    ("drafter on", lambda: key(drafter=DRAFTER)),
    ("drafter weights", lambda: key(drafter=dict(DRAFTER, weights_sha256="b" * 64))),
    ("k rule", lambda: key(drafter=dict(DRAFTER, k=3))),
    ("kernel bundle", lambda: key(kernel_bundle_sha="m" * 64)),
    ("bindings", lambda: key(eng(kernels=[]))),
    ("context (page policy)", lambda: key(eng(CTX=4096))),
    ("slots", lambda: key(eng(S=4))),
    ("steps per call", lambda: key(eng(K=4))),
    ("prefix cache off", lambda: key(eng(prefix_cache=False))),
    ("activation-edit capability",
     lambda: key(activation_edits={"enabled": True, "table_shape": [8, 4, 5120], "dtype": "bfloat16"})),
    ("program-changing flag", lambda: key(flags_=dict(FL, XLA_FLAGS=["--xla_tpu_enable_latency_hiding_scheduler=true"]))),
    ("config", lambda: key(config_sha="d" * 64)),
    ("weights layout", lambda: key(weights_layout_sha="n" * 64)),
])
def test_each_component_changes_the_component_set_and_the_deployment_id(name, k):
    _, dep, cs = k()
    assert cs != BASE[2] and dep != BASE[1], name


@pytest.mark.parametrize("name,k", [("chip", lambda: key(chip_=dict(CHIP, accelerator_type="TPU v5 lite"))),
                                    ("jax", lambda: key(sw_=dict(SW, jax="0.11.3"))),
                                    ("engine commit", lambda: key(sw_=dict(SW, engine_commit="y")))])
def test_chip_and_software_change_the_deployment_id_only(name, k):
    _, dep, cs = k()
    assert dep != BASE[1] and cs == BASE[2], name


def test_dict_order_and_weight_values_do_not_change_it():
    fl = dict(reversed(list(FL.items())))
    assert key(flags_=fl)[1:] == BASE[1:]
    W1 = {"a": np.zeros((2, 3), np.float32), "b": np.ones(4, np.int8)}
    W2 = {"b": np.full(4, 7, np.int8), "a": np.full((2, 3), 3.5, np.float32)}
    assert CK.weights_layout(W1) == CK.weights_layout(W2)
    assert CK.weights_layout(W1) != CK.weights_layout({"a": np.zeros((2, 4), np.float32), "b": np.ones(4, np.int8)})


def test_component_set_sha_is_inside_the_compile_key():
    k, dep, cs = BASE
    assert k["component_set_sha"] == cs and dep == CK.sha(k) and len(dep) == len(cs) == 64
    assert copy.deepcopy(k) == k


def test_a_cache_for_another_component_set_is_refused():
    assert CK.check_cache({"component_set_sha": BASE[2]}, BASE[2]) is None
    why = CK.check_cache({"component_set_sha": key(eng(CTX=4096))[2]}, BASE[2])
    assert why and "not used" in why
    assert CK.check_cache({}, BASE[2])


COUNT_PROBE = r"""
import json, types
from mashbox.runtime import programs
from mashbox.runtime.server import Server
front = Server.health(types.SimpleNamespace(engine=types.SimpleNamespace(S=1), waiting=0))
before = programs.counts()
programs.mark_ready()
import jax.numpy as jnp
jnp.arange(7).sum().block_until_ready()          # one compile after ready
after = programs.counts()
print(json.dumps(dict(front=front["compiles_after_ready"], before=before, after=after)))
"""


def test_a_process_that_does_not_count_compiles_reports_none_not_zero():
    """/health's compiles_after_ready is a gate witness (B2): a process that never marked ready (a front whose engine runs
    in a worker process, or a server before ready) has no count, so it reports null, never a 0 that reads as a pass."""
    import json
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=os.path.abspath(src))
    r = subprocess.run([sys.executable, "-c", COUNT_PROBE], capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["front"] is None
    assert got["before"]["compiles"] is None and got["before"]["compiles_after_ready"] is None
    assert got["after"]["compiles_after_ready"] >= 1 and got["after"]["compiles"] >= got["after"]["compiles_after_ready"]
