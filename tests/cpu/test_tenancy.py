"""Owner isolation of the prefix cache (skill mashbox-engine-batching A9/A10, certified-base gate B1).

The owner of a cached point is (cache namespace, tenant id, edit variant). The tenant id is an HMAC of the server's cache
secret and the key's tenant (or the key), never the key's place in the key list; rotating the secret or changing the
namespace leaves nothing that matches (E49). Another tenant's prompt never hits (B1 canary, E13), a request never copies
another tenant's slot state on its own admission (E31), the host cache keeps per-tenant floors (E32), a row with
activation edits never resumes another variant's state (E38), and every resume checks the owner key (IsolationError).

The engine runs a stand-in model with the real state layout (K/V per slot and position, a linear-attention state per slot
that starts from zero at position 0, as the model's forward does), so these run with JAX on CPU in seconds.
"""
import time

import numpy as np
import pytest

from mashbox.runtime import server as SV
from mashbox.runtime import tenancy as T

jax = pytest.importorskip("jax")

from engine_standin import MODEL, V, config, weights  # noqa: E402
from mashbox.runtime.engine import Engine, HostCache, IsolationError  # noqa: E402


def engine(S=2, cache_ram=1 << 30, ns="ns-a"):
    e = Engine(config(), weights(), MODEL, CTX=256, S=S, K=4,
               prefix_cache=True, cache_ram=cache_ram)
    e.set_namespace(ns)
    return e


def run(e, prompt, scope=None, variant=None, max_new=6):
    row = e.admit([dict(ids=list(prompt), max_new=max_new, scope=scope, variant=variant)])[0]
    while e.busy():
        e.step()
    return row


def prompts(seed, n=40):
    return np.random.default_rng(seed).integers(1, V, n).tolist()


TEN = T.Tenancy(secret=b"s" * 32)
A, B = TEN.tenant_id("key-a"), TEN.tenant_id("key-b")


# ------------------------------------------------------------------------------------------------- tenant id (A10)
def test_tenant_id_is_an_hmac_never_the_key_position():
    T.configure()
    h = {"Authorization": "Bearer key-a"}
    a1, a2 = SV.key_scope(h, ["key-a", "key-b"]), SV.key_scope(h, ["key-b", "key-a"])
    assert a1 == a2 == T.current().tenant_id("key-a")              # the same owner wherever the key sits in the file
    assert isinstance(a1, str) and a1.startswith("v1.") and len(a1) == 35 and "key-a" not in a1
    assert SV.key_scope({"x-api-key": "key-b"}, ["key-a", "key-b"]) != a1
    assert SV.key_scope({"x-api-key": "nope"}, ["key-a", "key-b"]) is None
    assert T.Tenancy(secret=b"x" * 32).tenant_id("key-a") != T.Tenancy(secret=b"y" * 32).tenant_id("key-a")


def test_key_file_tenants_group_keys():
    keys, ten = T.parse_key_lines(["k1 acme", "", "k2 acme", "k3"])
    assert keys == ["k1", "k2", "k3"] and ten == {"k1": "acme", "k2": "acme"}
    t = T.Tenancy(ten, secret=b"z" * 32)
    assert t.tenant_id("k1") == t.tenant_id("k2") != t.tenant_id("k3")


# ------------------------------------------------------------------------------------------------- B1 canary, E13
def test_canary_another_tenant_never_hits_and_gets_fresh_tokens():
    """Tenant A caches a conversation; tenant B sends A's exact next turn: B gets 0 cached tokens and the tokens B gets
    on a fresh engine, and A's own next turn still hits."""
    p = prompts(1)
    e = engine()
    ra = run(e, p, A)
    turn2 = p + ra["ids"] + [3, 4, 5]
    rb = run(e, turn2, B)
    assert rb["cached"] == 0 and rb["restored"] == 0
    assert rb["ids"] == run(engine(), turn2, B)["ids"]
    assert run(e, turn2, A)["cached"] == len(p)
    assert e.stats["other_copy_bytes"] == 0


# ------------------------------------------------------------------------------------------------- E31
def test_admission_never_copies_another_tenants_slot():
    """On one slot, B's request takes the slot that holds A's conversation: nothing of A's is copied to host memory on
    B's path (A's state is dropped). A request of A's own tenant on that slot still saves it, and A comes back restored."""
    e = engine(S=1)
    added = []
    orig = e.host.add

    def spy(x):
        added.append(x["scope"])
        orig(x)
    e.host.add = spy
    p, q = prompts(2), prompts(3)
    run(e, p, A)
    added.clear()
    run(e, q, B)
    assert added == [] and e.stats["dropped_other"] == 1 and e.stats["other_copy_bytes"] == 0
    ra = run(e, p, A)                                       # A's state was dropped: prefilled again, nothing cached
    assert ra["cached"] == 0
    added.clear()
    run(e, q, A)                                            # A's own other conversation: A's state is saved first
    assert len(added) == 1 and added[0][1] == A
    assert run(e, p + ra["ids"] + [7], A)["restored"] == len(p)


# ------------------------------------------------------------------------------------------------- E32 floors
def test_host_cache_floors_keep_a_quiet_tenant():
    """Cap 100 bytes, two tenants: A saves two 20-byte entries (inside its floor of 50), then B saves five. B's own
    oldest entries go; A keeps both."""
    h = HostCache(100)
    h.ns = "ns"

    def ent(t, k):
        return dict(kv=[], snaps=[dict(ids=(k,), r=np.zeros(1), c=np.zeros(1))], scope=("ns", t, None), bytes=20)
    for k in (1, 2):
        h.add(ent(A, k))
    for k in range(10, 15):
        h.add(ent(B, k))
    held = {}
    for x in h.items:
        held.setdefault(x["scope"][1], []).append(x["snaps"][0]["ids"][0])
    assert held[A] == [1, 2] and len(held[B]) == 3 and h.nbytes() <= 100
    assert h.evicted == 2


def test_health_states_the_host_cache_cap_and_evictions():
    """/health carries the host cache cap and its eviction count, so a live gate can size cache pressure from the server
    and see that evictions happened (CERT B1 floor arm)."""
    e = engine(S=1, cache_ram=12345)
    h = SV.Server(e, IdChat(), "standin", {}, log=lambda m: None).health()
    assert h["host_cache_cap_bytes"] == 12345 and h["host_cache_evictions"] == 0
    e.host.evicted = 3
    assert SV.Server(e, IdChat(), "standin", {}, log=lambda m: None).health()["host_cache_evictions"] == 3


# ------------------------------------------------------------------------------------------------- E49
def test_secret_rotation_and_namespace_change_give_zero_hits():
    t = T.Tenancy(secret=b"r" * 32)
    p = prompts(4)
    e = engine()
    a1 = t.tenant_id("key-a")
    turn2 = p + run(e, p, a1)["ids"] + [9]
    t.rotate()
    a2 = t.tenant_id("key-a")
    assert a2 != a1
    assert run(e, turn2, a2)["cached"] == 0                 # the same key after a rotation: nothing matches
    turn3 = turn2 + run(e, turn2, a2)["ids"] + [9]
    assert run(e, turn3, a2)["cached"] == len(turn2)         # (it caches again under the new id)
    e.set_namespace("ns-b")                                  # new weights, tokenizer, template or policy
    assert run(e, turn3, a2)["cached"] == 0
    assert all(x["scope"][0] == "ns-b" for x in e.host.items)   # entries of the old namespace are forgotten
    assert all(x["scope"][0] == "ns-b" for sn in e.snaps for x in sn)


def test_namespace_covers_each_field():
    base = dict(deployment="d", model="m", tokenizer="t", chat_template="c", checkpoint_policy=dict(ckpts=8))
    n0 = T.namespace(**base)
    for k in base:
        assert T.namespace(**dict(base, **{k: "other"})) != n0, k
    assert T.namespace(**dict(reversed(list(base.items())))) == n0


# ------------------------------------------------------------------------------------------------- E38 (CPU part)
def test_edit_variant_is_in_the_cache_key():
    p = prompts(5)
    e = engine()
    turn2 = p + run(e, p, A, variant="v1")["ids"] + [2]
    assert run(e, turn2, A, variant=None)["cached"] == 0      # no edits never resumes an edited state
    assert run(e, turn2, A, variant="v2")["cached"] == 0      # nor one edited differently
    assert run(e, turn2, A, variant="v1")["cached"] == len(p)


# ------------------------------------------------------------------------------------------------- ownership check
def test_a_resume_of_another_owner_key_fails_closed():
    e = engine(S=1)
    p = prompts(6)
    run(e, p, A)
    run(e, prompts(7), A)                                    # A's conversation goes to host memory
    forged = e.host.items[0]
    e.host.best = lambda prompt, own=None: (len(p), forged, forged["snaps"][-1])
    with pytest.raises(IsolationError):
        e.admit([dict(ids=p + [1, 2], max_new=2, scope=B)])
    assert e.stats["owner_checks"] >= 1


# ------------------------------------------------------------------------------------------------- A9 admission order
class IdChat:
    """Token ids as text: the server path without a tokenizer."""
    stop_ids = ()

    def encode(self, text):
        return [int(x) for x in text.split() if x.isdigit()]


C = TEN.tenant_id("key-c")


def admitted(warm_c, arrivals):
    """Server + real engine (stand-in model, 2 slots, prefix cache). Tenant C's cache is warmed with warm_c on the engine
    before the server starts. arrivals: (engine step, tenant, tag, prompt, max_new); the step-0 request is queued alone,
    every later one is submitted from inside the loop right before that engine step (a replayable arrival schedule).
    -> [(tenant, tag)] in the order the engine admitted them, [slot] in that order, the engine's stats, and per admit call
    the longest point of C's cache (slots + host memory) that the request tagged "c1" could resume at that moment."""
    e = engine(S=2)
    for p in warm_c:
        run(e, p, C)
    tags, order, slots, jobs, n, offer = {}, [], [], [], [0], []
    c1 = next(tuple(p) for _, _, tag, p, _ in arrivals if tag == "c1")

    def c_offer():
        pts = [tuple(x["ids"]) for sn in e.snaps for x in sn if (x.get("scope") or (None, None))[1] == C]
        pts += [tuple(sn["ids"]) for h in e.host.items if (h.get("scope") or (None, None))[1] == C for sn in h["snaps"]]
        return max((len(q) for q in pts if len(q) < len(c1) and c1[:len(q)] == q), default=0)
    srv = SV.Server(e, IdChat(), "standin", {}, log=lambda m: None)
    o_admit, o_step = e.admit, e.step

    def submit(t, tag, p, mx):
        tags[tuple(p)] = tag
        jobs.append(srv.submit(list(p), SV.knobs_of(dict(max_tokens=mx, temperature=0)), False, False, scope=t))

    def admit(reqs, stop_ids=()):
        offer.append(c_offer())
        rows = o_admit(reqs, stop_ids)
        for q, r in zip(reqs, rows):
            order.append((q.get("scope"), tags[tuple(q["ids"])]))
            slots.append(r["slot"])
        return rows

    def step(waiting=False):
        n[0] += 1
        for k, t, tag, p, mx in arrivals:
            if k == n[0]:
                submit(t, tag, p, mx)
        return o_step(waiting)
    e.admit, e.step = admit, step
    first = [a for a in arrivals if a[0] == 0]
    assert len(first) == 1
    submit(*first[0][1:])
    end = time.monotonic() + 300
    while time.monotonic() < end and not (len(jobs) == len(arrivals) and all(j["ev"].is_set() for j in jobs)):
        time.sleep(0.02)
    assert len(jobs) == len(arrivals) and all(j["ev"].is_set() for j in jobs)
    return order, slots, dict(e.stats), offer


def test_admission_order_does_not_depend_on_another_tenants_cache():
    """A9 (no content-dependent ordering): the same arrival schedule twice, where the ONLY change is what tenant C has in
    its cache (prompts x in run 1, other prompts of the same lengths in run 2). In run 1 C's queued request continues x (a
    cache hit is on offer when the engine admits), in run 2 it hits nothing. The engine admits the requests in arrival
    order (first come, first served) and on the same slots in both runs. Recorded at each admission: when b1 is admitted
    ahead of it, C's cache offers c1 a resume of all of x in run 1 and nothing in run 2 (the difference is there at the
    decision)."""
    x, y = prompts(20), prompts(21)
    pa, pa2, pb, pb2, pa3 = prompts(22), prompts(23), prompts(24), prompts(25), prompts(26)
    arrivals = [(0, A, "a1", pa, 6), (1, B, "b1", pb, 6), (1, C, "c1", x + [5, 6, 7], 6), (1, A, "a2", pa2, 6),
                (3, B, "b2", pb2, 4), (4, A, "a3", pa3, 4)]
    o1, s1, st1, f1 = admitted([x], arrivals)
    o2, s2, st2, f2 = admitted([y], arrivals)
    fifo = [(t, tag) for _, t, tag, _, _ in arrivals]
    assert o1 == o2 == fifo, (o1, o2)
    assert s1 == s2, (s1, s2)
    assert f1[1] == len(x) and f2[1] == 0, (f1, f2)                  # admit call 2 = b1's (a1 is call 1)
    assert st1["other_copy_bytes"] == st2["other_copy_bytes"] == 0
