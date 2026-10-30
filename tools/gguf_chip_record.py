"""One GGUF file on one TPU chip loaded the way `mashbox run <file>` loads it (zero flags), then its chip record: greedy
ids for the preregistered prompts one user at a time and 8 at once, first token, decode tok/s, peak device memory and
every XLA backend compile after ready, written as one JSON record.

It loads with cli._load_gguf_for_record, the internal loader `run` calls below its hold (gguf.archs.held). So the first
record of a held mixture-of-experts file runs on an unchanged src/ tree. `run`, serving and `inspect` keep refusing a
held file (`tpu up` takes catalog models only and starts `mashbox run` on the chip): no flag and no environment variable
opens the hold (a support gate on those commands, not a same-process boundary). The record written here is a raw chip record: it does not yet carry every field the catalog's
record guard needs for a passing row (invocation, fit, impl, versions, the MoE route proof, 1/8-user goodput), so on its
own it lists nothing.

usage: [SLOTS=n] [CODE_SHA=sha] [ZONE=zone] python tools/gguf_chip_record.py FILE.gguf IDS.json OUT.json [N]
(SLOTS unset = zero flags; CODE_SHA and ZONE are only written into the record)"""
import argparse
import json
import logging
import os
import platform
import sys
import time


def load(path, slots=None):
    """(model, engine, chat, info, t0) for a GGUF file, as cli.load_gguf returns them, from the loader below its hold"""
    from mashbox import cli, gguf as G
    a = cli.build_parser().parse_args(["run", path] + (["--slots", str(slots)] if slots else []))
    a = argparse.Namespace(**{**vars(a), "gguf": G.fetch_file(a.model, log=cli.log)})   # run's args and resolve, one call
    t0 = time.time()
    r = a.gguf
    f, A, c = G.open_model(r["path"], repo=r["repo"], quant=r["tag"])
    return cli._load_gguf_for_record(a, f, A, c, t0)


def main(argv):
    import jax
    import jax.monitoring
    from mashbox import cli, __version__
    path, idsf, out = argv[0], argv[1], argv[2]
    N = int(argv[3]) if len(argv) > 3 else 64
    comp = dict(phase="load", n=0, after_ready=[])

    def on_ev(ev, dur, **kw):
        if ev == "/jax/core/compile/backend_compile_duration":
            comp["n"] += 1
            if comp["phase"] == "ready":
                comp["after_ready"].append(round(dur, 3))
    jax.monitoring.register_event_duration_secs_listener(on_ev)
    m, eng, chat, info, t0 = load(path, os.environ.get("SLOTS"))   # t0: after resolve, as before the file is opened
    t_load = time.time() - t0
    cli.warm(eng, chat)
    t_ready = time.time() - t0
    comp["at_ready"] = comp["n"]
    comp["phase"] = "ready"

    class _Names(logging.Handler):
        """names every program compiled after ready: jax logs "Compiling <name> ..." at WARNING"""
        def emit(self, r):
            msg = r.getMessage()
            if "ompil" in msg:
                comp.setdefault("after_ready_log", []).append(msg[:300])
    for _n in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler", "jax"):
        logging.getLogger(_n).addHandler(_Names())
    jax.config.update("jax_log_compiles", True)
    P = json.load(open(idsf))
    ids_chip = [chat.chat_ids([dict(role="user", content=p)]) for p in P["prompts"]]
    assert ids_chip == P["ids"], "token ids on the chip differ from the off-chip ids"
    rec = dict(file=os.path.basename(path), mashbox=__version__, code_sha=os.environ.get("CODE_SHA"), jax=jax.__version__,
               device=str(jax.devices()[0].device_kind), n_devices=len(jax.devices()), host=platform.node(),
               zone=os.environ.get("ZONE"), N=N, slots=eng.S, ctx=eng.CTX, kernels=os.environ.get("MASHBOX_GGUF_KERNELS"),
               load_s=round(t_load, 1), ready_s=round(t_ready, 1), weights=dict((k, info[k]) for k in info if isinstance(info[k], (int, float, str))))
    one = []
    for i, ids in enumerate(P["ids"]):
        r = eng.generate([ids], max_new=N, stop_ids=())
        t = r["timing"]
        one.append(dict(i=i, ids=list(map(int, r["ids"][0])), prefill_s=t["prefill_s"], decode_s=t["decode_s"], new=t["new_tokens"]))
        print(json.dumps(dict(event="one", i=i, prefill_s=round(t["prefill_s"], 3), decode_s=round(t["decode_s"], 3), new=t["new_tokens"])), flush=True)
    rec["one"] = one
    eight = []
    for g in range(0, len(P["ids"]), min(8, eng.S)):   # 8 users at once, or as many as the engine's slots (--slots)
        r = eng.generate(P["ids"][g:g + min(8, eng.S)], max_new=N, stop_ids=())
        t = r["timing"]
        eight.append(dict(first=g, ids=[list(map(int, x)) for x in r["ids"]], prefill_s=t["prefill_s"], decode_s=t["decode_s"], new=t["new_tokens"]))
        print(json.dumps(dict(event="eight", first=g, prefill_s=round(t["prefill_s"], 3), decode_s=round(t["decode_s"], 3), new=t["new_tokens"])), flush=True)
    rec["eight"] = eight
    r = eng.generate([P["ids"][0]], max_new=256, stop_ids=())
    rec["one_256"] = dict(prefill_s=r["timing"]["prefill_s"], decode_s=r["timing"]["decode_s"], new=r["timing"]["new_tokens"])
    ms = jax.devices()[0].memory_stats() or {}
    rec["memory"] = dict((k, ms.get(k)) for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit", "largest_alloc_size"))
    rec["compiles"] = dict(before_ready=comp["at_ready"], after_ready=len(comp["after_ready"]), after_ready_s=comp["after_ready"], after_ready_log=comp.get("after_ready_log", []))
    rec["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(out, "w"))
    print(json.dumps(dict(event="record", out=out, compiles_after_ready=len(comp["after_ready"]), peak_gb=round((ms.get("peak_bytes_in_use") or 0) / 1e9, 2))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
