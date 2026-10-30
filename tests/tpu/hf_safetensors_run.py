"""One Hugging Face model on one TPU chip the way `mashbox run <org/model>` loads it (mashbox.hf.load + cli.warm, zero
flags), then: greedy ids for the preregistered prompts one user at a time and 8 at once, decode speed, peak device
memory and every compile after ready. Writes one JSON record.

  python tests/tpu/hf_safetensors_run.py <org/model or folder> IDS.json OUT.json [N]
IDS.json: {"prompts": [...], "ids": [...]} with the ids made off the chip from the same folder's chat template.
"""
import json
import os
import platform
import sys
import time

import jax
import jax.monitoring

from mashbox import cli, hf, __version__

ref, idsf, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 64
comp = dict(phase="load", n=0, after_ready=[])


def on_ev(ev, dur, **kw):
    if ev == "/jax/core/compile/backend_compile_duration":
        comp["n"] += 1
        if comp["phase"] == "ready":
            comp["after_ready"].append(round(dur, 3))


jax.monitoring.register_event_duration_secs_listener(on_ev)
P = json.load(open(idsf))
a = cli.build_parser().parse_args(["run", ref])
a.hf = hf.resolve(ref, log=cli.log)
assert a.hf["revision"] == P["revision"], f"Hub revision {a.hf['revision']} is not {idsf}'s pinned {P['revision']}"
t0 = time.time()
m, eng, chat, info, _ = hf.load(a)
t_load = time.time() - t0
cli.warm(eng, chat)
t_ready = time.time() - t0
comp["at_ready"], comp["phase"] = comp["n"], "ready"
ids_chip = [chat.chat_ids([dict(role="user", content=p)]) for p in P["prompts"]]
assert ids_chip == P["ids"], "token ids on the chip differ from the off-chip ids"
rec = dict(model=ref, revision=a.hf["revision"], mashbox=__version__, code_sha=os.environ.get("CODE_SHA"), jax=jax.__version__,
           device=str(jax.devices()[0].device_kind), n_devices=len(jax.devices()), host=platform.node(),
           zone=os.environ.get("ZONE"), N=N, slots=eng.S, ctx=eng.CTX, kernels=os.environ.get("MASHBOX_GGUF_KERNELS"),
           load_s=round(t_load, 1), ready_s=round(t_ready, 1),
           weights={k: v for k, v in info.items() if isinstance(v, (int, float, str))})
one = []
for i, ids in enumerate(P["ids"]):
    r = eng.generate([ids], max_new=N, stop_ids=())
    t = r["timing"]
    one.append(dict(i=i, ids=list(map(int, r["ids"][0])), prefill_s=t["prefill_s"], decode_s=t["decode_s"], new=t["new_tokens"]))
    print(json.dumps(dict(event="one", i=i, prefill_s=round(t["prefill_s"], 3), decode_s=round(t["decode_s"], 3))), flush=True)
rec["one"] = one
eight = []
for g in range(0, len(P["ids"]), min(8, eng.S)):
    r = eng.generate(P["ids"][g:g + min(8, eng.S)], max_new=N, stop_ids=())
    t = r["timing"]
    eight.append(dict(first=g, ids=[list(map(int, x)) for x in r["ids"]], prefill_s=t["prefill_s"], decode_s=t["decode_s"],
                      new=t["new_tokens"]))
rec["eight"] = eight
r = eng.generate([P["ids"][0]], max_new=256, stop_ids=())
rec["one_256"] = dict(prefill_s=r["timing"]["prefill_s"], decode_s=r["timing"]["decode_s"], new=r["timing"]["new_tokens"])
ms = jax.devices()[0].memory_stats() or {}
rec["memory"] = {k: ms.get(k) for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit", "largest_alloc_size")}
rec["compiles"] = dict(before_ready=comp["at_ready"], after_ready=len(comp["after_ready"]), after_ready_s=comp["after_ready"])
rec["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(rec, open(out, "w"))
print(json.dumps(dict(event="record", out=out, tok_s_1=round(256 / max(1e-9, rec["one_256"]["decode_s"]), 1),
                      peak_gb=round((ms.get("peak_bytes_in_use") or 0) / 1e9, 2))), flush=True)
