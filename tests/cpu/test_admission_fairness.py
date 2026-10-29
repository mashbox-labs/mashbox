"""Admission fairness: a request sent a moment after a full batch of others joins at the next device call; it never
waits for them to finish (CHIPS-FINISH, v5e-4, before continuous batching: in 2 of 5 rounds of 8 users, one request sent
with the other 7 got its first token at 2.93 s vs 0.62 s, after they had all finished; the whole-batch loop gathers
requests for 20 ms and a request that misses the window waits for the whole batch).

The server runs the real engine (continuous batching, warmed) on the CPU stand-in model: 7 users send long answers at
once, the 8th user sends 0.2 s later; each engine step takes at least 5 ms, as a device call does. Gate: the 8th's first token comes while the 7 are still answering, within a fixed
bound. The whole-batch loop (Server.loop_batches: the server's path for an engine without admit / step; Server.loop
says which engines take it) fails the same gate and stays visible as a strict expected failure."""
import queue
import threading
import time

import pytest

from mashbox.runtime import server as SV

pytest.importorskip("jax")

from engine_standin import MODEL, config, weights  # noqa: E402
from mashbox.runtime.engine import Engine  # noqa: E402


class Chat:
    stop_ids = ()

    def encode(self, text):
        return [ord(c) % 60 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


class Timed(Engine):
    """Each engine step takes at least STEP_S, as a device call does (the stand-in's own call takes ~1 ms on CPU)."""
    STEP_S = 0.005

    def _step(self, *a, **k):
        time.sleep(self.STEP_S)
        return super()._step(*a, **k)


class WholeBatch:
    """The engine without admit / step: the server runs its whole-batch loop (Server.loop_batches)."""

    def __init__(self, e):
        self.e, self.CTX, self.S, self.K, self.prefix_cache = e, e.CTX, e.S, e.K, e.prefix_cache

    def check(self, q):
        return self.e.check(q)

    def generate(self, *a, **k):
        r = self.e.generate(*a, **k)
        self.cached_now = self.e.cached_now
        return r


def first_token_times(whole_batch):
    e = Timed(config(), weights(), MODEL, CTX=512, S=8, K=4, prefix_cache=True)
    e.generate([[5, 6, 7]], max_new=5)
    e.warm(workers=2)
    srv_engine = WholeBatch(e) if whole_batch else e
    srv = SV.Server(srv_engine, Chat(), "m", {}, log=lambda m: None)
    srv.ready = True
    t_first, t_end, t_sent = {}, {}, {}

    def user(i, delay, max_new):
        time.sleep(delay)
        t_sent[i] = time.perf_counter()
        j = srv.submit([(i * 7 + k) % 60 + 1 for k in range(20)], SV.knobs_of(dict(max_tokens=max_new)), True, True)
        while True:
            try:
                x = j["stream"].get(timeout=60)
            except queue.Empty:
                break
            if x is None:
                break
            if x[0] and i not in t_first:
                t_first[i] = time.perf_counter()
        t_end[i] = time.perf_counter()
    th = [threading.Thread(target=user, args=(i, 0.0, 400)) for i in range(7)]
    th.append(threading.Thread(target=user, args=(7, 0.2, 8)))
    for t in th:
        t.start()
    for t in th:
        t.join(120)
    return t_sent, t_first, t_end


@pytest.mark.parametrize("whole_batch", [
    False,
    pytest.param(True, marks=pytest.mark.xfail(strict=True, reason="the whole-batch loop (the server's path for an "
                                                                     "engine without admit / step) admits only between "
                                                                     "batches"))])
def test_late_request_joins_at_the_next_call(whole_batch):
    t_sent, t_first, t_end = first_token_times(whole_batch)
    assert sorted(t_end) == list(range(8)) and 7 in t_first, (t_first, t_end)
    late_ttft = t_first[7] - t_sent[7]
    others_end = min(t_end[i] for i in range(7))
    others_run = others_end - min(t_sent[i] for i in range(7))
    print(dict(whole_batch=whole_batch, late_ttft_s=round(late_ttft, 4), others_run_s=round(others_run, 4)))
    assert t_first[7] < others_end, dict(late_ttft=late_ttft, others_run=others_run)
    assert late_ttft < 0.25 * others_run, dict(late_ttft=late_ttft, others_run=others_run)
