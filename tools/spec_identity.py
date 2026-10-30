"""Are the MTP speculation programs of this tree the programs the chip records measured?

Builds the tiny Qwen3.8 MTP engine of tests/cpu/test_spec_in_step.py (random weights, no chip, no network), lists every
program warm() compiles with speculation on (prefill with hidden states, plain decode, the drafter's prefill program and
the step program at every window, row bucket and round count), lowers each one for the given platforms with
jit(...).trace(...).lower(lowering_platforms=...) and hashes its StableHLO with the debug information left out (no
locations; argument and result names are dropped too, since they are only names). Run it on two trees and compare:

    python tools/spec_identity.py --out a.json          # on tree A (PYTHONPATH=A/src)
    python tools/spec_identity.py --out b.json          # on tree B
    python tools/spec_identity.py --compare a.json b.json

Equal hashes for every program = the same programs, so the chip records made on tree A cover tree B for these programs.
The shapes are the tiny model's, not the served model's: the program text is made by the same Python code for any shape.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile


def strip(txt):
    txt = re.sub(r"\s*loc\([^()]*(\([^()]*\)[^()]*)*\)", "", txt)
    txt = "\n".join(ln for ln in txt.splitlines() if not ln.startswith("#loc"))
    txt = re.sub(r'jax\.(arg|result)_info = "[^"]*",?\s*', "", txt)
    txt = re.sub(r"\{\s*\}", "", txt)
    return txt


def programs(platforms):
    import numpy as np
    import jax
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "tests", "cpu"))
    import test_spec_in_step as T
    from mashbox.runtime.engine import Engine
    from mashbox.runtime import programs as PG
    wd = tempfile.mkdtemp(prefix="spec-identity-")
    json.dump(dict(model_type="qwen3_5_text", vocab_size=T.V, hidden_size=T.D, intermediate_size=T.F,
                   num_hidden_layers=T.L, num_attention_heads=T.H, num_key_value_heads=T.KV, head_dim=T.HD,
                   linear_num_key_heads=T.HK, linear_num_value_heads=T.HV, linear_key_head_dim=T.DK,
                   linear_value_head_dim=T.DV, linear_conv_kernel_dim=T.CK, full_attention_interval=4, rms_norm_eps=1e-6,
                   rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25),
                   tie_word_embeddings=False, attn_output_gate=True, mtp_num_hidden_layers=1),
              open(os.path.join(wd, "config.json"), "w"))
    cfg = T.Q.Config(wd)
    path = os.path.join(wd, "w.bshj")
    T.write_bshj(path, T.tensors(11, 1.0))
    W, _ = T.QW.load(path, cfg)
    Wm, _ = T.QS.load_mtp(path, cfg)
    eng = Engine(cfg, W, T.Q, CTX=256, S=4, K=4, mtp=(T.QS, Wm), draft_k=2, spec_rounds=2)
    ck, cv = eng.new_state()
    a = jax.numpy.asarray
    i32 = np.int32
    calls = []
    for B, Tn, Tc in eng.prefill_shapes():
        calls.append(("prefill", eng.fwd, (a(np.zeros((B, Tn), i32)), a(np.zeros(B, i32)), a(np.ones(B, i32)),
                                           a(np.arange(B, dtype=i32))), dict(Tc=Tc, hidden=True)))
    calls += eng.spec_programs(ck, cv)
    out = {}
    for kind, f, args, kw in calls:
        sig = PG.signature(f.__name__ if hasattr(f, "__name__") else kind, args, kw)
        tr = f.trace(eng.W, ck, cv, *args, **kw)
        for plat in platforms:
            txt = tr.lower(lowering_platforms=(plat,)).as_text(debug_info=False)
            out[f"{plat} {kind} {sig}"] = hashlib.sha256(strip(txt).encode()).hexdigest()
    return out, dict(jax=jax.__version__, programs=len(calls), platforms=list(platforms))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--platforms", default="cpu,tpu")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args(argv)
    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        hx, hy = x["hashes"], y["hashes"]
        same = sorted(k for k in hx if hy.get(k) == hx[k])
        diff = sorted(k for k in set(hx) | set(hy) if hx.get(k) != hy.get(k))
        print(json.dumps(dict(equal=len(same), different=len(diff), only_a=len(set(hx) - set(hy)),
                              only_b=len(set(hy) - set(hx)), first_different=diff[:10]), indent=1))
        return 0 if not diff else 1
    h, info = programs([p for p in a.platforms.split(",") if p])
    rec = dict(info, hashes=h)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(dict(info, n=len(h))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
