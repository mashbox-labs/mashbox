"""On a TPU v6e-1 after installing this package with the tpu extra: load the real model, run one greedy chat turn, print JSON.

  python3 tests/tpu/qwen3_8_27b_smoke.py [<local .bshj>]
"""
import argparse
import json
import sys
import time

from mashbox import cli

a = argparse.Namespace(model="qwen3-8-27b", weights=sys.argv[1] if len(sys.argv) > 1 else None, no_kernels=False, cache_dir=None,
                       ctx=2048, slots=8, steps=8)
m, eng, chat, info, t0 = cli.load(a)
ids = chat.chat_ids([dict(role="user", content="Name three planets of the solar system, one per line.")])
t = time.time()
r = eng.generate([ids], max_new=64, stop_ids=chat.stop_ids)
print(json.dumps(dict(load_s=round(t - t0, 1), gen_s=round(time.time() - t, 2), kernels=eng.kernels, text=chat.decode(r["ids"][0]),
                      timing=r["timing"]), indent=1, default=str))
