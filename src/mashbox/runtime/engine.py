"""Batched generation on one chip: S slots, chunked prefill, K decode steps per device call.

Greedy rows run the greedy decode program on the smallest batch bucket (1, 4 or S) that holds the active rows.
If any row samples or uses a penalty, the batch runs the sampled program on all S rows, so a row's draws never depend
on the batch shape.

Continuous batching (admit / step, used by the server): rows live in their slots between
device calls. admit() checks each request on the host first (check(): a bad request is refused alone and changes no shared
state), then places it in a free slot with its prefix-cache work; it does not prefill. step() is one scheduler step: one
decode call for every decoding row and, in the same step, at most one prefill chunk per row that is still reading its
prompt, inside a token budget (budget_busy while any row decodes, so a long prompt never stalls the others for more than
one chunk; budget_idle when nothing decodes). A chunk cut by the budget ends on a multiple of 64 in absolute position (the
gated-delta chunk grid). A row gets its first token at the step where its prefill ends and decodes from the next step. A
decode call holds K steps only while no row is prefilling and nobody waits (step(waiting=True) -> 1 step), so a new request
joins within one decode step. Rows end at their own finish (EOS, stop token, max_new, cancel); tokens a K-step call made past
a row's finish are dropped. generate() is admit + step until its rows end.

Padding rows (free or still-prefilling slots inside a decode bucket) never write live state: their K/V writes go past the
cache and are dropped, and their linear-attention state rows are written back unchanged (keep mask), so a row paused
mid-prefill keeps its state. trace=True records, per generated token, a hash of the row's f32 logits and its top-1 / top-2
gap (the E25 identity harness compares these between a lone run and a batched run).

Prefix cache (prefix_cache=True, used by the server): the slot state lives on between calls. Each slot remembers the
token ids its state holds at a few points (the end of the prompt, the start of its last message and, as llama.cpp's
context checkpoints, user-message starts spaced at least `ckpt_step` tokens apart), with a host copy of the slot's
linear-attention state at each point (the full-attention K/V stay in the slot). The linear-attention state cannot be
rolled back, so a cached prompt can only be resumed at one of these points. A new prompt that starts with one of those
token lists goes to that slot and prefills only the tokens after it. Agents resend the whole conversation every turn,
so each turn prefills only what is new; after a compaction the start of the first user message still matches.

Host-memory prompt cache (cache_ram bytes, 0 = off), the same design as llama.cpp's `--cache-ram` (llama-server, PR
16391): when a request takes a slot whose conversation it does not continue, the slot's K/V up to its longest point and
its snapshots are copied to host memory first. A later prompt that starts with one of those points is restored into a
slot, which puts back exactly the state the slot held, instead of being prefilled again. So two agents (or an agent and
its side requests) on one slot stop evicting each other.

Owners (generate(scopes=...), the server's tenants; runtime/tenancy.py): every saved point, in a slot or in host memory,
belongs to the owner key of the request that made it = (cache namespace, tenant id, edit variant), and a prompt resumes
only from points with its own owner key, so one tenant's prompts are never served from (or reported as cached by)
another's, a row with activation edits never resumes a state computed without them (or with other edits), and a
namespace change (set_namespace) or a secret rotation leaves nothing that matches. Every resume checks the owner key
again before it touches the state (IsolationError if it ever differs). A request never copies another tenant's slot:
when it takes a slot that holds another tenant's conversation, that state is dropped, not copied to host memory on this
request's path (a slot of its own tenant is saved first, as before). The host cache evicts with per-tenant floors: while
over its cap, the least recently saved entry of a tenant above its floor (cap / tenants present) goes first, so another
tenant's traffic never evicts a tenant that stays inside its floor. Without scopes everything is one owner, as before.
"""
import os
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import programs as PG
from . import sampler as SP


def _tenant(own):
    """(namespace, tenant) of an owner key (a bare scope from older callers: no namespace)."""
    return tuple(own[:2]) if isinstance(own, tuple) else (None, own)


class IsolationError(RuntimeError):
    """A cache entry of one owner key reached a request of another: never expected; the request fails, nothing is served."""

i32 = np.int32
I32_MAX = (1 << 31) - 1
ALIGN = 64          # prefill chunk boundaries cut by the token budget land on this grid (the gated-delta chunk, model.CHUNK)


def logit_hash(lg):
    """(B, V) f32 -> (B,) uint32: sum over v of bits(lg[v]) * (2v + 1) mod 2^32. Integer sums do not depend on reduction
    order, and any single changed bit changes the hash."""
    u = lax.bitcast_convert_type(lg.astype(jnp.float32), jnp.uint32)
    return jnp.sum(u * (jnp.arange(lg.shape[-1], dtype=jnp.uint32) * jnp.uint32(2) + jnp.uint32(1)), -1, dtype=jnp.uint32)


def top2_gap(lg):
    t = lax.top_k(lg.astype(jnp.float32), 2)[0]
    return t[:, 0] - t[:, 1]


def bucket(n, lo):
    b = lo
    while b < n:
        b *= 2
    return b


def _nbytes(t):
    return sum(int(x.nbytes) for x in jax.tree.leaves(t))


class HostCache:
    """Saved slot states in host memory, matched on exact token ids, capped by bytes, least recently saved out first."""

    def __init__(self, cap):
        self.cap, self.items, self.clock, self.ns = int(cap), [], 0, None
        self.evicted = 0            # entries evicted to stay under the cap (/health host_cache_evictions)

    def nbytes(self):
        return sum(e["bytes"] for e in self.items)

    def tenant_bytes(self):
        """{(namespace, tenant): bytes held}"""
        use = {}
        for e in self.items:
            t = _tenant(e.get("scope"))
            use[t] = use.get(t, 0) + e["bytes"]
        return use

    def best(self, p, own=None):
        """-> (length, entry, snapshot) of the longest saved point of this owner key that is a strict prefix of prompt p
        (0, None, None)"""
        best = (0, None, None)
        for e in self.items:
            if e.get("scope") != own:
                continue
            for sn in e["snaps"]:
                L = len(sn["ids"])
                if best[0] < L < len(p) and sn["ids"] == tuple(p[:L]):
                    best = (L, e, sn)
        return best

    def add(self, e):
        top = e["snaps"][-1]["ids"]
        # an entry is superseded when all its points are prefixes of the new entry's longest point (same owner key only)
        self.items = [x for x in self.items if x.get("scope") != e.get("scope")
                      or not all(top[:len(sn["ids"])] == sn["ids"] for sn in x["snaps"])]
        self.clock += 1
        e["t"] = self.clock
        self.items.append(e)
        while self.nbytes() > self.cap:
            self.take(self.victim())
            self.evicted += 1

    def victim(self):
        """The entry to evict: one of another namespace first (it can never match again), else the least recently saved
        entry of a tenant above its floor (cap / tenants present; while over the cap, at least one tenant is above it)."""
        stale = [x for x in self.items if _tenant(x.get("scope"))[0] != self.ns]
        if stale:
            return min(stale, key=lambda x: x["t"])
        use = self.tenant_bytes()
        floor = self.cap / max(1, len(use))
        over = [x for x in self.items if use[_tenant(x.get("scope"))] > floor]
        return min(over or self.items, key=lambda x: x["t"])

    def take(self, e):
        self.items = [x for x in self.items if x is not e]     # by identity: entries hold arrays, == would compare them


class Engine:
    # speculation is off unless __init__ turns it on (an Engine made with __new__ in a test reads these too)
    drafter = dpf = dstate = sstep = spec_why = None
    spec_off = False
    _dhid = False

    def __init__(self, cfg, W, model, CTX=2048, S=8, K=8, decode_kernels=None, chunk=512, tokens_per_call=2048,
                 prefix_cache=False, attn_bytes=2e9, cache_ram=8 << 30, ckpts=8, ckpt_step=8192, budget_busy=128,
                 budget_idle=1024, trace=False, mtp=None, draft_k=0, spec_rounds=None, k_table=None, drafter=None,
                 spec_why=None):
        self.c, self.W, self.CTX, self.S, self.K = cfg, W, CTX, S, K
        self.chunk, self.tokens_per_call = chunk, tokens_per_call
        self.prefix_cache, self.attn_bytes = prefix_cache, attn_bytes
        self.state, self.snaps, self.used, self.clock = None, [[] for _ in range(S)], [0] * S, 0
        self.ckpts, self.ckpt_step = max(2, int(ckpts)), int(ckpt_step)
        self.host = HostCache(cache_ram) if (prefix_cache and cache_ram > 0) else None
        self.kv_block = min(2048, CTX)      # host copies move K/V in blocks of this many positions (small device temp)
        self.stats = dict(requests=0, prompt_tokens=0, cached_tokens=0, restored_tokens=0, saves=0, dropped_other=0,
                          other_copy_bytes=0, owner_checks=0)
        self.cache_ns = None        # the prefix-cache namespace (set_namespace; runtime/tenancy.py)
        # prefill tokens per step: while any row decodes (spec 2.3 stall-free budget; ~128 on v6e-1 8-bit, ESTIMATE) and idle
        self.budget_busy, self.budget_idle, self.trace = max(ALIGN, int(budget_busy)), max(ALIGN, int(budget_idle)), bool(trace)
        # sampled rows on row buckets (spec 7, G13): OFF until the batch-invariance probe passes for the sampler and the decode
        # ops at every bucket on the chip type; off, any sampled row runs the sampled program on all S rows
        self.sampled_buckets = False
        self.order = []             # live rows in admission order (prefill chunks are handed out first come, first served)
        self.model = model
        # continuous batching: the live row of each slot (None = free), its next token / position, and the sampled
        # program's per-slot settings and counts (made at the first sampled row)
        self.rows = [None] * S
        # pos CTX = not decoding: such a row's K/V writes fall past the cache and are dropped (padding never writes live state)
        self.tok, self.pos, self.act = np.zeros(S, i32), np.full(S, CTX, i32), np.zeros(S, bool)
        self.samp_rows, self.sp, self.cnt, self.seen = [None] * S, None, None, None
        fwd = model.make_forward(cfg, decode_kernels)
        # the model's own sampler when it has one (tensor parallel: logits split on the vocabulary axis, shard.py)
        SPm = getattr(model, "sampler", None) or SP
        # Speculative decoding (runtime/spec_step.py, runtime/drafters/): drafter = the drafter object (mtp = (head module, MTP
        # weights) is the MTP drafter's short form); draft_k = k_max drafts per round; spec_rounds rounds per device call
        # when nobody waits, else 1; k_table {row bucket: k} = the measured per-bucket draft length, k_max where absent.
        # It runs inside the batched step (_spec_step): every decoding row of a step is in one draft / verify / commit
        # program; requests with draft=False (such as constrained rows) run with k = 0 in that same program, sampled and
        # penalised rows verify by the coupled draw, and each row caps its own k by its remaining max_tokens.
        if drafter is None and mtp is not None and draft_k:
            from .drafters.mtp import MTP
            drafter = MTP(mtp[0], mtp[1], cfg, draft_k)
        self.drafter = drafter if draft_k else None
        self.DK = int(draft_k) if self.drafter is not None else 0
        self.k_table = {int(b): max(0, min(self.DK, int(k))) for b, k in (k_table or {}).items()}
        # a row bucket at k = 0 decodes with the plain program (_spec_call). A drafter that allows no plain call between
        # its calls (plain_ok False: the MTP head carries every committed token) is not turned on with such a table:
        # plain decoding, and spec_why says why
        self.spec_why = None if self.DK else spec_why      # why decoding is plain (the load's reason from cli), Engine.spec_health
        if self.DK and not getattr(self.drafter, "plain_ok", False) and 0 in self.k_table.values():
            self.spec_why = (f"speculation off: the {getattr(self.drafter, 'kind', None)} drafter needs every committed "
                             f"token, so it cannot run a row bucket at k = 0 (k_table {self.k_table}): plain decoding")
            self.drafter, self.DK, self.k_table = None, 0, {}
        self.spec_stats = dict(rounds=0, tokens=0, hist=[0] * (self.DK + 2), plain_calls=0, backoffs=0)
        self.hlast = {}
        self.dpf = self.sstep = None
        self._dhid = bool(self.DK) and self.drafter.needs_hidden
        if self.DK:
            from . import spec_step as SS
            from . import drafters as DRS
            if getattr(self.drafter, "kind", None) not in DRS.ORDER:
                raise ValueError(f"speculation: unknown drafter kind {getattr(self.drafter, 'kind', None)!r}")
            if getattr(self.drafter, "abi", None) != DRS.DRAFTER_ABI:
                raise ValueError(f"speculation: the {self.drafter.kind} drafter implements drafter ABI "
                                 f"{getattr(self.drafter, 'abi', None)}, this engine {DRS.DRAFTER_ABI}")
            self.NR = int(spec_rounds or max(1, K // 2))
            pf = self.drafter.on_prefill()
            self.dpf = jax.jit(pf, static_argnames=("Tc",), donate_argnums=(1, 2)) if pf is not None else None
            self.sstep = jax.jit(SS.make_step(fwd, SS.arch_commit(model, cfg), self.drafter),
                                 static_argnames=("Tc", "K", "N", "trace"), donate_argnums=(1, 2))
        self.kernels = sorted(decode_kernels) if decode_kernels else []
        self.shapes = None          # set to a list to record the program shapes one call runs

        CTX_ = CTX

        def tr0(K, B):
            return (jnp.zeros((K, B), jnp.uint32), jnp.zeros((K, B), jnp.float32))

        def tr1(t, i, lg):
            return (t[0].at[i].set(logit_hash(lg)), t[1].at[i].set(top2_gap(lg)))

        def dec(W, ck, cv, tok, pos, Tc, K, trace=False):
            B = tok.shape[0]
            slots, ones = jnp.arange(B, dtype=jnp.int32), jnp.ones((B,), jnp.int32)

            def step(i, c_):
                ck, cv, tok, pos, out, tr = c_
                r = fwd(W, ck, cv, tok[:, None], pos, ones, slots, Tc, contig=True, keep=pos < CTX_)
                tr = tr1(tr, i, r["lg"]) if trace else tr
                return r["ck"], r["cv"], r["nxt"], pos + 1, out.at[i].set(r["nxt"]), tr
            ck, cv, tok, pos, out, tr = lax.fori_loop(0, K, step, (ck, cv, tok, pos, jnp.zeros((K, B), jnp.int32), tr0(K, B)))
            return out, ck, cv, tok, pos, tr

        def decs(W, ck, cv, tok, pos, sp, step0, cnt, seen, Tc, K, trace=False):
            B = tok.shape[0]
            slots, ones = jnp.arange(B, dtype=jnp.int32), jnp.ones((B,), jnp.int32)
            plain = SPm.no_penalty(sp)

            def step(i, c_):
                ck, cv, tok, pos, out, cnt, seen, tr = c_
                keep = pos < CTX_
                r = fwd(W, ck, cv, tok[:, None], pos, ones, slots, Tc, contig=True, keep=keep)
                tr = tr1(tr, i, r["lg"]) if trace else tr
                x = SPm.penalise(r["lg"], cnt, seen, sp)
                nt = SPm.sample(x, sp, step0 + i, greedy_tok=jnp.where(plain, r["nxt"], jnp.argmax(x, -1).astype(jnp.int32)))
                b_ = jnp.arange(B)
                # counts of rows that are not decoding (padding, prefilling) stay as they are
                return (r["ck"], r["cv"], nt, pos + 1, out.at[i].set(nt), cnt.at[b_, nt].add(keep.astype(jnp.float32)),
                        seen.at[b_, nt].set(seen[b_, nt] | keep), tr)
            init = (ck, cv, tok, pos, jnp.zeros((K, B), jnp.int32), cnt, seen, tr0(K, B))
            ck, cv, tok, pos, out, cnt, seen, tr = lax.fori_loop(0, K, step, init)
            return out, ck, cv, tok, pos, cnt, seen, tr

        def first(lg, nxt, sp, cnt, seen):
            x = SPm.penalise(lg, cnt, seen, sp)
            tok = SPm.sample(x, sp, jnp.zeros(lg.shape[0], jnp.int32),
                            greedy_tok=jnp.where(SPm.no_penalty(sp), nxt, jnp.argmax(x, -1).astype(jnp.int32)))
            b_ = jnp.arange(lg.shape[0])
            return tok, cnt.at[b_, tok].add(1.0), seen.at[b_, tok].set(True)

        def first_rows(lg, nxt, sp, cnt, seen, rows, seen_new):
            """First token of new rows only. rows (R,): slot of each row of lg, S for padding (dropped). Resets each new
            slot's counts and seen set (seen_new: its prompt tokens), then penalises and samples as `first`."""
            S_ = cnt.shape[0]
            cnt = cnt.at[rows].set(0.0, mode="drop")
            seen = seen.at[rows].set(seen_new, mode="drop")
            g = lambda a: jnp.take(a, jnp.minimum(rows, S_ - 1), axis=0)
            spr = jax.tree.map(g, sp)
            x = SPm.penalise(lg, g(cnt), g(seen), spr)
            tok = SPm.sample(x, spr, jnp.zeros(lg.shape[0], jnp.int32),
                            greedy_tok=jnp.where(SPm.no_penalty(spr), nxt, jnp.argmax(x, -1).astype(jnp.int32)))
            return tok, cnt.at[rows, tok].add(1.0, mode="drop"), seen.at[rows, tok].set(True, mode="drop")

        self.fwd = jax.jit(fwd, static_argnames=("Tc", "contig", "all_logits", "verify", "hidden"), donate_argnums=(1, 2))
        self.first_rows = jax.jit(first_rows, donate_argnums=(3, 4))
        self.dec = jax.jit(dec, static_argnames=("Tc", "K", "trace"), donate_argnums=(1, 2))
        self.decs = jax.jit(decs, static_argnames=("Tc", "K", "trace"), donate_argnums=(1, 2))
        self.lhash = jax.jit(lambda lg: (logit_hash(lg), top2_gap(lg)))
        self.first = jax.jit(first)

        def put(cv, s, r, c):
            R, C = cv
            return (lax.dynamic_update_slice(R, r[:, :, None].astype(R.dtype), (0, 0, s, 0, 0, 0)),
                    lax.dynamic_update_slice(C, c[:, :, None].astype(C.dtype), (0, 0, s, 0, 0)))

        def take(cv, s):
            R, C = cv
            return lax.dynamic_index_in_dim(R, s, 2, keepdims=False), lax.dynamic_index_in_dim(C, s, 2, keepdims=False)
        self.put = jax.jit(put, donate_argnums=(0,))
        self.take = jax.jit(take)

        def kv_take(ck, s, p, n):        # K/V of slot s, positions p..p+n-1 (axes: group, slot, position, ...)
            return jax.tree.map(lambda a: lax.dynamic_slice(a, (0, s, p) + (0,) * (a.ndim - 3), (a.shape[0], 1, n) + a.shape[3:]), ck)

        def kv_put(ck, s, p, blk):
            return jax.tree.map(lambda a, b: lax.dynamic_update_slice(a, b.astype(a.dtype), (0, s, p) + (0,) * (a.ndim - 3)), ck, blk)
        self.kv_take = jax.jit(kv_take, static_argnames=("n",))

        def swap(ck, cv, a, b):
            """Exchange the whole state of slots a and b (K/V: slot axis 1; linear state R, C: slot axis 2)."""
            def sw(x, ax):
                xa, xb = lax.dynamic_index_in_dim(x, a, ax), lax.dynamic_index_in_dim(x, b, ax)
                return lax.dynamic_update_index_in_dim(lax.dynamic_update_index_in_dim(x, xb, a, ax), xa, b, ax)
            return jax.tree.map(lambda x: sw(x, 1), ck), jax.tree.map(lambda x: sw(x, 2), cv)
        self.swap = jax.jit(swap, donate_argnums=(0, 1))
        self.kv_put = jax.jit(kv_put, donate_argnums=(0,))

        def gather_last(buf, lg, src, dst):
            """buf (S, V) f32 logits by slot; row src[k] of one prefill call's logits lg goes to slot dst[k] (S = none).
            One program per prefill row bucket, so a first token never runs an eager op whose shape is the row count."""
            return buf.at[dst].set(jnp.take(lg, src, axis=0).astype(jnp.float32), mode="drop")
        self.gather_last = jax.jit(gather_last)
        # Weights on one device: the slot state is made there as committed arrays (shard.py does the same on a mesh). JAX keys
        # a compiled program on whether each argument is committed, and every program's outputs from a committed state are
        # committed; a fresh uncommitted state made the first prefill and decode of each new state programs of their own,
        # which the warm-up never ran (measured: a 5.4 s compile inside the first reply on a TPU v6e-1, ctx 2048 x 8 slots).
        devs = {d for a in jax.tree.leaves(W) if hasattr(a, "devices") for d in a.devices()}
        self.dev = jax.sharding.SingleDeviceSharding(devs.pop()) if len(devs) == 1 else None
        # with speculation a drafter may keep K/V in extra groups (the MTP head: one)
        xg = dict(extra_groups=self.drafter.extra_groups) if (self.DK and self.drafter.extra_groups) else {}
        mk = lambda: model.new_state(cfg, S - 1 if S <= 2 else S, CTX, **xg)
        self._new_state = jax.jit(mk, out_shardings=self.dev) if self.dev is not None else mk
        S_, V_ = S, cfg.V
        # (S, V) logits by slot: the zeros and every gather_last output in one placement, so a first-token call that starts
        # from the zeros and one that follows another prefill call are one program per row bucket (tensor parallel: the
        # model's own placement for (S, V) rows, its vocabulary split; one chip: self.dev, as before)
        zsh = self.dev if self.dev is not None or not hasattr(model, "placement") else model.placement((S_, V_))
        self._zl = jax.jit(lambda: jnp.zeros((S_, V_), jnp.float32), out_shardings=zsh)
        if self.dev is None and zsh is not None:
            self.gather_last = jax.jit(gather_last, out_shardings=zsh)
        # every program a request runs goes through the executable table (runtime/programs.py Table, Dispatch): warm()
        # builds or loads each one there, and a call runs its executable with no trace; a call not in it runs through jit
        self.aot = PG.Table()
        for n_, skip in (("fwd", 1), ("dec", 1), ("decs", 1), ("first_rows", 0), ("first", 0), ("lhash", 0), ("put", 0),
                         ("take", 0), ("kv_take", 0), ("kv_put", 0), ("swap", 0), ("gather_last", 0), ("_zl", 0)):
            setattr(self, n_, PG.Dispatch(n_, getattr(self, n_), self.aot, skip))
        self.program_list = []      # signatures of the programs warm() compiled (runtime/programs.py)
        # speculation: the drafter's device state for every slot (made here, before ready, so no request compiles it)
        self.dstate = self.drafter.state(self) if self.DK else None

    def new_state(self):
        # The model adds one scratch slot for padding rows. With S <= 2 no call ever has a padding row (prefill and decode
        # use 1 or S rows), so the scratch slot is left out: at a long context that is one slot of K/V less on the chip.
        return self._new_state()

    def checkpoint_policy(self):
        """What decides which cached points exist (part of the cache namespace)."""
        return dict(ckpts=self.ckpts, ckpt_step=self.ckpt_step, ctx=self.CTX, align=ALIGN)

    def set_namespace(self, ns):
        """The prefix-cache namespace (tenancy.namespace(...)): points cached under another namespace never match again, so
        they are forgotten here (slot points and host entries)."""
        if ns != self.cache_ns:
            self.snaps = [[x for x in sn if _tenant(x.get("scope"))[0] == ns] for sn in self.snaps]
        self.cache_ns = ns
        if self.host is not None:
            self.host.ns = ns
            self.host.items = [x for x in self.host.items if _tenant(x.get("scope"))[0] == ns]

    def _own(self, q):
        """The owner key of request q: (cache namespace, tenant id or None, edit variant or None)."""
        return (self.cache_ns, q.get("scope"), q.get("variant"))

    def _check_own(self, x, own):
        """Every resume of a saved point or host entry passes here: its owner key must be the request's."""
        self.stats["owner_checks"] += 1
        if x.get("scope") != own:
            raise IsolationError("a cache entry of another owner key reached this request")

    def _committed(self, x):
        commit = getattr(self.model, "commit", None)      # tensor parallel: the model places it on its mesh
        if commit is not None:
            return commit(x)
        return jax.device_put(x, self.dev) if self.dev is not None else x

    def _rows(self, n, cap):
        for b in (1, 4, 16, 64, 256):
            if n <= b <= cap:
                return b
        return cap

    def _note(self, *shape):
        if self.shapes is not None:
            self.shapes.append(shape)

    # ------------------------------------------------------------------------------------------------------- prefill
    def prefill(self, state, prompts, slots=None, starts=None, ends=None, lazy=False):
        """Run each prompt into its slot from starts[i] to ends[i] (defaults: slot i, 0, the whole prompt).
        -> state, greedy next token per slot, last logits per slot (only for rows that reached the end of their prompt;
        lazy=True: (the call's logits, row) per slot, read later by _last_rows, so no device op runs per row)."""
        ck, cv = state
        n = len(prompts)
        slots = list(range(n)) if slots is None else slots
        starts = [0] * n if starts is None else starts
        ends = [len(p) for p in prompts] if ends is None else ends
        work = [[slots[i], np.asarray(p, i32), starts[i], ends[i]] for i, p in enumerate(prompts) if ends[i] > starts[i]]
        nxt, last = {}, {}
        while work:
            T = min(self.chunk, bucket(max(w[3] - w[2] for w in work), 16))
            cap = max(1, min(self.S, self.tokens_per_call // T))
            p0m = max(w[2] for w in work[:cap])
            Tc = 0 if p0m == 0 else min(self.CTX, bucket(p0m, 128))
            if Tc:      # keep the attention scores of one call (rows x heads x T x (Tc + T), f32 + bf16) under attn_bytes
                per = 10 * self.c.H
                while T > 16 and T * (Tc + T) * per > self.attn_bytes:
                    T //= 2
                cap = max(1, min(cap, int(self.attn_bytes // (T * (Tc + T) * per))))
            grp = work[:cap]
            B = self._rows(len(grp), cap)
            toks, p0 = np.zeros((B, T), i32), np.zeros(B, i32)
            lens, sl = np.ones(B, i32), np.full(B, self.S, i32)
            for b, (slot, ids, s0, e) in enumerate(grp):
                ch = ids[s0:min(e, s0 + T)]
                toks[b, :len(ch)], p0[b], lens[b], sl[b] = ch, s0, len(ch), slot
            Tc = 0 if p0.max() == 0 else min(self.CTX, bucket(int(p0.max()), 128))
            r = self.fwd(self.W, ck, cv, jnp.asarray(toks), jnp.asarray(p0), jnp.asarray(lens), jnp.asarray(sl), Tc=Tc,
                         hidden=self._dhid)
            ck, cv = r["ck"], r["cv"]
            self._note("prefill", B, T, Tc)
            done = [b for b, w in enumerate(grp) if w[2] + int(lens[b]) >= len(w[1])]
            if self.dpf is not None:    # the drafter over this chunk (MTP: position j reads the token at j + 1 and the hidden state at j)
                tn, lm, hrow, hidx = np.zeros((B, T), i32), np.zeros(B, i32), np.full(B, self.S, i32), np.zeros(B, i32)
                for b, (slot, ids, s0, e) in enumerate(grp):
                    m_ = max(0, min(int(lens[b]), len(ids) - 1 - s0))
                    tn[b, :m_], lm[b] = ids[s0 + 1:s0 + 1 + m_], m_
                for b in done:
                    hrow[b], hidx[b] = grp[b][0], int(lens[b]) - 1
                self.dstate, ck, cv = self.dpf(self.W, ck, cv, self.drafter.W, self.dstate, jnp.asarray(tn), r["hid"], jnp.asarray(p0),
                                              jnp.asarray(lm), jnp.asarray(sl), jnp.asarray(hrow), jnp.asarray(hidx), Tc=Tc)
            if done:
                nx, lg = np.asarray(r["nxt"]), r["lg"]
                for b in done:
                    nxt[grp[b][0]], last[grp[b][0]] = int(nx[b]), ((lg, b) if lazy else lg[b])
            for b, w in enumerate(grp):
                w[2] += int(lens[b])
            work = [w for w in work if w[2] < w[3]]
        return (ck, cv), nxt, last

    # ------------------------------------------------------------------------------------------------------- prefix cache
    def _place(self, prompts, owns=None):
        """-> slot per prompt, cached snapshot (or None) per prompt, host cache entry to restore (or None) per prompt.
        owns: the owner key of each prompt (_own). Longest cached prefix of the prompt's own owner key first (in a slot, or
        in host memory when that is longer), then free slots: while rows are live, those that keep the decode batch
        bucket as small as it is; then empty slots and slots of the prompt's own tenant before another tenant's (whose
        state would be dropped); then least recently used, lowest index first (one user decodes on the smallest batch).
        Slots with a live row are never taken."""
        n, S = len(prompts), self.S
        busy = {s_ for s_, r in enumerate(getattr(self, "rows", None) or [None] * S) if r is not None}
        sc = list(owns or [self._own({})] * n)
        slot, snap, ent = [None] * n, [None] * n, [None] * n
        if self.prefix_cache and self.state is not None:
            hits = []
            for i, p in enumerate(prompts):
                for s_ in range(S):
                    if s_ in busy:
                        continue
                    for sn in self.snaps[s_]:
                        L = len(sn["ids"])
                        if L < len(p) and sn.get("scope") == sc[i] and sn["ids"] == tuple(p[:L]):
                            hits.append((L, i, s_, sn))
            for L, i, s_, sn in sorted(hits, key=lambda h: -h[0]):
                if slot[i] is None and s_ not in slot:
                    slot[i], snap[i] = s_, sn
        if self.host is not None:
            for i, p in enumerate(prompts):
                L, e, sn = self.host.best(p, sc[i])
                if e is not None and L > (len(snap[i]["ids"]) if snap[i] else 0) and all(e is not x for x in ent):
                    slot[i], snap[i], ent[i] = None, sn, e
        cur = self._rows(max(busy) + 1, S) if busy else 0
        free = [s_ for s_ in range(S) if s_ not in slot and s_ not in busy]

        def other(s_, i):           # the slot holds a conversation of another namespace or tenant
            return bool(self.snaps[s_]) and _tenant(self.snaps[s_][-1].get("scope")) != _tenant(sc[i])
        for i in range(n):
            if slot[i] is None:
                slot[i] = min(free, key=lambda s_: (max(cur, self._rows(s_ + 1, S)) if busy else 0, other(s_, i),
                                                    self.used[s_], s_))
                free.remove(slot[i])
        return slot, snap, ent

    def _save(self, state, s_, by):
        """Copy slot s_ (its K/V up to its longest point and its snapshots) to the host cache, on the path of a request
        with owner key `by` (only ever a request of the slot's own tenant: _admit drops another tenant's state)."""
        sn = self.snaps[s_]
        own = sn[-1].get("scope")
        if any(x.get("scope") != own for x in sn):
            raise IsolationError("a slot holds points of two owner keys")
        L, n = len(sn[-1]["ids"]), self.kv_block
        per = sum(a.dtype.itemsize * a.shape[0] * int(np.prod(a.shape[3:])) for a in jax.tree.leaves(state[0]))
        if -(-L // n) * n * per + sum(x["r"].nbytes + x["c"].nbytes for x in sn) > self.host.cap:
            return                          # larger than the whole cache: not worth the copy
        kv = [(min(p, self.CTX - n), jax.tree.map(np.asarray, self.kv_take(state[0], s_, min(p, self.CTX - n), n=n)))
              for p in range(0, L, n)]
        e = dict(kv=kv, snaps=list(sn), scope=own,
                 bytes=_nbytes([b for _, b in kv]) + sum(x["r"].nbytes + x["c"].nbytes for x in sn))
        if _tenant(own) != _tenant(by):
            self.stats["other_copy_bytes"] += e["bytes"]
        self.host.add(e)
        self.stats["saves"] += 1

    def _restore(self, state, s_, e):
        ck = state[0]
        for p, blk in e["kv"]:
            ck = self.kv_put(ck, s_, p, blk)
        return (ck, state[1])

    def _trim(self, sn, new):
        """llama.cpp's checkpoint limit (server-context.cpp create_checkpoint): over the limit, drop older points that sit
        within ckpt_step of an earlier kept one (never this request's own), then the oldest."""
        sn = sorted(sn, key=lambda x: len(x["ids"]))
        out, last = [], -1
        for k, x in enumerate(sn):
            if (len(out) + len(sn) - k > self.ckpts and id(x) not in new and last >= 0
                    and len(x["ids"]) <= last + self.ckpt_step):
                continue
            out.append(x)
            last = len(x["ids"])
        return out[-self.ckpts:]

    def _snap(self, state, s, ids, own=None):
        r, c = self.take(state[1], s)
        return dict(ids=tuple(int(t) for t in ids), r=np.asarray(r), c=np.asarray(c), scope=own)

    def reset_cache(self):
        """Forget the slot states (the host cache holds separate copies and stays)."""
        self.state, self.snaps = None, [[] for _ in range(self.S)]

    def windows(self, lo):
        """Attention windows a call can use: lo, 2 lo, ... up to the first >= CTX, capped at CTX (bucket(p, lo) for p < CTX)."""
        out, t = [], lo
        while True:
            out.append(min(self.CTX, t))
            if t >= self.CTX:
                return out
            t *= 2

    def prefill_shapes(self):
        """Every (rows, chunk, attention window) program prefill() can run, by its own rules: chunk = min(chunk, bucket(n, 16)),
        rows per call capped by tokens_per_call, window 0 (prompt start) or bucket(start, 128), chunk and rows cut to keep
        the attention scores under attn_bytes (cut for the group's first estimate of the window, then run at the window
        of the rows actually taken, which can be smaller)."""
        per, Ts, t = 10 * self.c.H, [], 16
        while t < self.chunk:
            Ts.append(t)
            t *= 2
        Ts.append(self.chunk)
        Tcs = [0] + self.windows(128)
        out = set()
        for T0 in Ts:
            cap0 = max(1, min(self.S, self.tokens_per_call // T0))
            for Tc in Tcs:
                T, cap = T0, cap0
                if Tc:
                    while T > 16 and T * (Tc + T) * per > self.attn_bytes:
                        T //= 2
                    cap = max(1, min(cap, int(self.attn_bytes // (T * (Tc + T) * per))))
                for m in range(1, cap + 1):
                    for Tc_ in Tcs:
                        if Tc_ <= Tc:
                            out.add((self._rows(m, cap), T, Tc_))
        return sorted(out)

    def warm(self, workers=8):
        """Compile, then run once, every program a request can reach, so no request stops to compile one (as vLLM's TPU
        runner precompiles its buckets before serving): each prefill program (prefill_shapes), each decode program
        (every attention window 128 ... CTX; greedy at each row bucket, sampled at S rows; K steps per call and 1 step
        per call, which the scheduler runs while a row prefills or a request waits), with speculative decoding on
        (--mtp K) its programs (spec_programs()), and the helper programs a request runs (logit rows by slot for each
        prefill row bucket, the sampled first token, logit hashes, slot state take / put / swap, host-cache copies when
        the host cache is on). The CPU gate E14 counts every compile, eager ops included, over 40 mixed requests after
        this. Arguments are made as a request makes them (committed slot state and sampler counts, host-made token
        arrays, numpy slot indexes and host copies, the same static arguments), since JAX keys programs on that too. The
        prefill, decode and speculation programs are compiled first in `workers` threads (XLA compiles on the host): the
        prefill and decode programs into the executable table (self.aot; a program already loaded there from a saved
        table is not compiled again), the speculation programs through jit (the table does not hold them); then each
        runs once; compiles are served from the compile cache when it holds them. A program loaded from a saved table is
        not run again when programs.proven says so (the table's entries all ran in the start that saved them, on this
        deployment; the call's own argument checks pass without running it; the device has room for it). Only
        representatives run: the first (smallest) program of each prefill row bucket (its logits feed the helper
        programs) and the first of each decoder function (dec, decs); every other loaded program (other attention
        windows, step widths, row buckets) first runs in a request. Any failure here while a loaded table is in use,
        including one a representative reports only at the final barrier, quarantines and drops the table and raises
        programs.TableFailed, as a loaded program that fails in a request does: the engine's process ends, and the next
        one starts without the table and compiles before it says ready (A13: a failed device call restarts the process;
        this one never warms a second time). MASHBOX_WARM_RUN_ALL=1 runs every program. Call before the first request, with no row live: the slot
        states are cleared after. self.program_list = the signature of each program (runtime/programs.py).
        -> dict(prefill=programs, decode=programs, helpers=programs, programs=total, list_sha, compile_s, run_s,
        compile_failed, run_programs, skipped, run_why {reason: count}, fallback=None: a table failure raises)."""
        if self.busy():
            raise RuntimeError("warm() while rows are live")
        t0, S, CTX, V, a = time.perf_counter(), self.S, self.CTX, self.c.V, jnp.asarray
        ck, cv = self.state if self.state is not None else self.new_state()
        self.state = None
        calls = []
        for B, T, Tc in self.prefill_shapes():
            calls.append(("prefill", self.fwd, (a(np.zeros((B, T), i32)), a(np.zeros(B, i32)), a(np.ones(B, i32)),
                                                  a(np.arange(B, dtype=i32))), dict(Tc=Tc, hidden=self._dhid)))
        n_pre = len(calls)
        sp = SP.to_device(SP.pack([dict(temperature=0.7, top_p=0.9, top_k=20, presence_penalty=1.0, seed=1)] * S, S))
        cnt, seen = self._committed(jnp.zeros((S, V), jnp.float32)), self._committed(jnp.zeros((S, V), bool))
        tr = self.trace
        for Tc in self.windows(128):
            for K in sorted({1, self.K}):
                for B in sorted({self._rows(n, S) for n in range(1, S + 1)}):
                    calls.append(("decode", self.dec, (a(np.zeros(B, i32)), a(np.full(B, CTX, i32))), dict(Tc=Tc, K=K, trace=tr)))
                calls.append(("decode", self.decs, (a(np.zeros(S, i32)), a(np.full(S, CTX, i32)), sp, a(np.zeros(S, i32)),
                                                    cnt, seen), dict(Tc=Tc, K=K, trace=tr)))
        calls += self.spec_programs(ck, cv)         # speculation (G-D3): [] when off

        def comp(c):
            try:        # into the executable table (a program already there, loaded from a saved table, is skipped)
                if hasattr(c[1], "build"):
                    c[1].build(self.W, ck, cv, *c[2], **c[3])
                else:
                    c[1].lower(self.W, ck, cv, *c[2], **c[3]).compile()
                return 0
            except Exception:       # the run below compiles it then
                return 1
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as ex:
                failed = sum(ex.map(comp, calls))
        else:
            failed = sum(map(comp, calls))
        sigs = [PG.signature(f.__name__ if hasattr(f, "__name__") else kind, args, kw) for kind, f, args, kw in calls]
        t1 = time.perf_counter()
        lgs = {}
        skip = bool(self.aot.loaded) and os.environ.get("MASHBOX_WARM_RUN_ALL") != "1"
        canary, tags = set(), set()
        for i, (kind, f, args, kw) in enumerate(calls):    # the first (smallest) call per prefill row bucket / decoder function
            tag = (kind, args[0].shape[0]) if kind == "prefill" else (kind, getattr(f, "__name__", i))
            if tag not in tags:
                tags.add(tag)
                canary.add(i)
        free = PG.device_free() if skip else None
        ran, skipped, why_run = 0, 0, {}
        # helper programs, called exactly as a request calls them (_last_rows, _first, _admit, _snap, _save, _restore,
        # the rows held by decode kernels in _step)
        src, dst, helpers = np.zeros(S, i32), np.full(S, S, i32), []

        def h(name, f, *args, **kw):
            helpers.append(PG.signature(name, args, kw))
            if hasattr(f, "build"):
                f.build(*args, **kw)
            return f(*args, **kw)
        try:
            for i, (kind, f, args, kw) in enumerate(calls):
                if skip and i not in canary:
                    why = PG.proven(self.aot, f, (self.W, ck, cv, *args), kw, free)
                    if why is None:
                        skipped += 1
                        continue
                    why_run[why] = why_run.get(why, 0) + 1
                r = f(self.W, ck, cv, *args, **kw)
                ran += 1
                ck, cv = (r["ck"], r["cv"]) if kind == "prefill" else (r[1], r[2])
                if kind == "prefill":
                    lgs.setdefault(r["lg"].shape[0], r["lg"])
            buf = h("zeros_by_slot", self._zl)
            for B_ in sorted(lgs):
                buf = h("gather_last", self.gather_last, buf, lgs[B_], src, dst)
            h("lhash", self.lhash, buf)
            _, cnt, seen = h("first_rows", self.first_rows, buf, np.zeros(S, i32), sp, cnt, seen, np.full(S, S, i32),
                             np.zeros((S, V), bool))
            r_, c_ = h("take", self.take, cv, 0)
            cv = h("put", self.put, cv, 0, r_, c_)
            cv = h("put", self.put, cv, 0, np.asarray(r_), np.asarray(c_))
            ck, cv = h("swap", self.swap, ck, cv, np.int32(0), np.int32(0))
            if self.DK:         # the drafter's own host-called programs (n-gram: its row put)
                for name, f, args, kw in self.drafter.programs(self):
                    self.dstate = h(name, f, self.dstate, *args, **kw)
            if self.host is not None:
                blk = jax.tree.map(np.asarray, h("kv_take", self.kv_take, ck, 0, 0, n=self.kv_block))
                ck = h("kv_put", self.kv_put, ck, 0, 0, blk)
            jax.block_until_ready((ck, cv, buf))     # a device error a representative raised asynchronously shows here
        except Exception as e:
            traceback.clear_frames(e.__traceback__)     # free the device state the failed call still holds
            self.state = None
            if isinstance(e, PG.TableFailed) or not (skip or self.aot.loaded):
                raise       # a loaded program's own failure (Dispatch quarantined and dropped the table), or no table
            # a failure no single table call raised (e.g. at the final barrier) while the table is in use: the same
            # table failure (quarantined and dropped, with whether its folder could be closed)
            raise self.aot.fail(None, "warm", e) from e
        ck = cv = buf = None    # drop the warm-up state (rebound, not deleted: comp above closes over these names)
        lgs = {}
        self.reset_cache()
        self.program_list = sorted(set(sigs + helpers))
        self.aot.list_sha = PG.list_sha(self.program_list)     # a saved table records the list it was built for
        t2 = time.perf_counter()
        return dict(prefill=n_pre, decode=len(calls) - n_pre, helpers=len(set(helpers)), programs=len(self.program_list),
                    list_sha=PG.list_sha(self.program_list), compile_s=round(t1 - t0, 1), run_s=round(t2 - t1, 1),
                    compile_failed=failed, table=len(self.aot.ex), run_programs=ran, skipped=skipped, run_why=why_run,
                    fallback=None)

    # ------------------------------------------------------------------------------------------------------- generate
    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, scopes=None):
        try:
            return self._generate(prompts, max_new, stop_ids, samp, on_step, marks, scopes)
        except BaseException as e:
            self._drop_live()
            traceback.clear_frames(e.__traceback__)     # free the device state the failed call still holds
            raise

    def _drop_live(self):
        """After a failed device call: forget the live rows; the slot state may be half written or donated, so with the
        prefix cache the slot states are dropped too (the host cache holds separate copies and stays)."""
        self.rows, self.order = [None] * self.S, []
        self.hlast = {}
        self.act[:], self.tok[:], self.pos[:] = False, 0, self.CTX
        self.sp, self.cnt, self.seen = None, None, None
        if self.prefix_cache:
            self.reset_cache()
        else:
            self.state = None

    def busy(self):
        """True while any row is live (between admit and its end)."""
        return any(r is not None for r in self.rows)

    def free_slots(self):
        return sum(r is None for r in self.rows)

    def prefilling(self):
        """Live rows still reading their prompt, in admission order."""
        return [r for r in self.order if r["cur"] < r["prompt_len"]]

    def _commit_snaps(self, r):
        """Publish the prompt points row r's prefill has passed as its slot's cache points."""
        if self.prefix_cache and r.get("new_snaps") is not None:
            self.clock += 1
            self.snaps[r["slot"]], self.used[r["slot"]] = self._trim(r["new_snaps"], r["made"]), self.clock
            r["new_snaps"] = None

    def _end(self, r, fin):
        """End live row r and free its slot. The prompt points its prefill passed stay in the prefix cache (a row ended
        mid-prefill keeps only the complete points it reached)."""
        s_ = r["slot"]
        r["fin"], r["t_end"] = fin, time.perf_counter()
        self._commit_snaps(r)
        self.rows[s_], self.act[s_], self.tok[s_], self.pos[s_] = None, False, 0, self.CTX
        self.order = [x for x in self.order if x is not r]
        if not self.busy() and not self.prefix_cache:
            self.state = None           # no cache to keep: free the slot state, as a finished generate() does

    def cancel(self, rows):
        """End live rows now (client gone, stop string found), decoding or still prefilling. -> the rows ended."""
        out = []
        for r in rows or ():
            if r is not None and r["fin"] is None and r.get("slot") is not None and self.rows[r["slot"]] is r:
                self._end(r, "cancel")
                out.append(r)
        return out

    def check(self, q):
        """Admission phase 1, on the host, touching no shared state: None when request q (dict(ids, max_new, samp=None))
        fits this engine's types, else the reason. A refused request is answered alone (400); other rows never see it."""
        V, CTX = self.c.V, self.CTX
        try:
            ids, mx = np.asarray(q["ids"]), q["max_new"]
        except Exception:
            return "ids and max_new are required"
        if ids.ndim != 1 or ids.size < 1 or ids.dtype.kind not in "iu":
            return "the prompt must be 1 or more integer token ids"
        if int(ids.min()) < 0 or int(ids.max()) >= V:
            return f"token ids must be 0..{V - 1}"
        if isinstance(mx, bool) or not isinstance(mx, (int, np.integer)) or not 1 <= int(mx) <= I32_MAX:
            return f"max_new must be an integer 1..{I32_MAX}"
        if ids.size + int(mx) > CTX:
            return f"prompt ({ids.size} tokens) + max_new ({int(mx)}) is over the context of {CTX}"
        return SP.check(q.get("samp"), V)

    def admit(self, reqs, stop_ids=()):
        """Admission phase 2: start new rows while others may be live. reqs: list of dict(ids, max_new, samp=None,
        mark=None, scope=None, variant=None), at most free_slots(). A request that fails check() gets a row with fin "error" (reason in
        "error") and changes no shared state. Each other request gets a free slot and its prefix-cache work (cache hit,
        host restore); its prefill runs in step(), one chunk per step, and its first token comes at the step where the
        prefill ends. -> one row dict per request, in order: slot, ids (generated tokens, grows in step()), fin (None while
        live; "stop", "length", "cancel", "error"), cached, restored, sampled, t_admit / t_first (None until the first
        token) / t_end (perf_counter), calls (device calls it was in), lh (trace=True: (hash, gap) per token)."""
        try:
            return self._admit(reqs, stop_ids)
        except BaseException as e:
            self._drop_live()
            traceback.clear_frames(e.__traceback__)
            raise

    def step(self, waiting=False):
        """One scheduler step: a decode call for every decoding row (K steps only when no row prefills and waiting is
        False, else 1) and at most one prefill chunk per prefilling row inside the token budget. -> rows that ended."""
        try:
            return self._step(waiting)
        except BaseException as e:
            self._drop_live()
            traceback.clear_frames(e.__traceback__)
            raise

    def _admit(self, reqs, stop_ids=()):
        n_all, S = len(reqs), self.S
        if not 0 < n_all <= self.free_slots():
            raise ValueError(f"1..{self.free_slots()} new rows (free slots) per admit")
        t0 = time.perf_counter()
        why = [self.check(q) for q in reqs]
        out_all = [None] * n_all
        for i, w in enumerate(why):
            if w is not None:
                out_all[i] = dict(slot=None, ids=[], fin="error", error=w, cached=0, restored=0, sampled=False, t_admit=t0,
                                  t_first=None, t_end=t0, calls=0, prompt_len=len(reqs[i].get("ids") or ()), lh=[])
        good = [i for i in range(n_all) if why[i] is None]
        if not good:
            return out_all
        reqs = [reqs[i] for i in good]
        n = len(reqs)
        prompts = [[int(t) for t in q["ids"]] for q in reqs]
        samp = [q.get("samp") for q in reqs]
        marks = [q.get("mark") for q in reqs]
        sc = [self._own(q) for q in reqs]
        is_s = [bool(r and (float(r.get("temperature") or 0) > 0 or float(r.get("repetition_penalty") or 1.0) != 1.0
                            or float(r.get("frequency_penalty") or 0) != 0 or float(r.get("presence_penalty") or 0) != 0))
                for r in samp]
        pc = self.prefix_cache
        slot, snap, ent = self._place(prompts, sc)
        if self.state is None:
            self.state = self.new_state()
        state = self.state
        self.state = None            # donated below; put back at the end
        # Decode runs on the smallest batch bucket (1, 4, S) that holds the highest live slot, so a new row goes to the
        # lowest free slot: the slot _place picked (cache hit or least recently used) trades its whole state with it on the
        # device (a copy of two slots, ~1 ms here), so the cached conversations it held stay cached, one slot over.
        busy = {s_ for s_ in range(S) if self.rows[s_] is not None}
        for i in range(n):
            cur = self._rows(max(busy) + 1, S) if busy else 0
            t = min((s_ for s_ in range(S) if s_ not in busy and s_ not in slot[i + 1:] and s_ not in slot[:i]), default=None)
            if t is not None and t < slot[i] and max(cur, self._rows(t + 1, S)) < max(cur, self._rows(slot[i] + 1, S)):
                a_ = slot[i]
                state = self.swap(state[0], state[1], np.int32(a_), np.int32(t))
                self.snaps[a_], self.snaps[t] = self.snaps[t], self.snaps[a_]
                self.used[a_], self.used[t] = self.used[t], self.used[a_]
                slot[i] = t
            busy.add(slot[i])
        start = [len(sn["ids"]) if sn else 0 for sn in snap]
        keep_snaps, pts = [[] for _ in range(n)], [[] for _ in range(n)]
        if pc:
            for e in ent:
                if e is not None:    # restored below: it lives in its slot again (and is saved again if evicted)
                    self.host.take(e)
            for i in range(n):
                s_, old = slot[i], self.snaps[slot[i]]
                if old and (ent[i] is not None or snap[i] is not old[-1]):   # it does not continue the slot's conversation
                    if _tenant(old[-1].get("scope")) != _tenant(sc[i]):
                        self.stats["dropped_other"] += 1   # another tenant's state: dropped, never copied on this path
                    elif self.host is not None:
                        self._save(state, s_, sc[i])
            for i in range(n):
                s_ = slot[i]
                if ent[i] is not None:
                    self._check_own(ent[i], sc[i])
                if snap[i] is not None:
                    self._check_own(snap[i], sc[i])
                src = ent[i]["snaps"] if ent[i] is not None else self.snaps[s_]
                keep_snaps[i] = [x for x in src if snap[i] is not None and len(x["ids"]) <= start[i]
                                 and x.get("scope") == sc[i] and x["ids"] == tuple(prompts[i][:len(x["ids"])])]
                if ent[i] is not None:
                    state = self._restore(state, s_, ent[i])
                if snap[i] is not None:
                    state = (state[0], self.put(state[1], s_, snap[i]["r"], snap[i]["c"]))
            for i, m in enumerate(marks):
                c = sorted(set(x for x in ([] if m is None else [m] if isinstance(m, int) else m) if start[i] < x < len(prompts[i])))
                last, take = (len(keep_snaps[i][-1]["ids"]) if keep_snaps[i] else -1), []
                for j, x in enumerate(c):
                    if j == len(c) - 1 or last < 0 or x > last + self.ckpt_step:
                        take.append(x)
                        last = x
                pts[i] = take
        for i in range(n):
            self.samp_rows[slot[i]] = samp[i] if is_s[i] else None
        if any(is_s) or self.sp is not None:      # the sampled program's settings follow every slot that changed
            self.sp = SP.to_device(SP.pack(self.samp_rows, S))
        if any(is_s) and self.cnt is None:
            # committed, as warm() makes them (JAX keys a program on that too)
            self.cnt, self.seen = self._committed(jnp.zeros((S, self.c.V), jnp.float32)), self._committed(jnp.zeros((S, self.c.V), bool))
        self.state = state
        stop = set(stop_ids)
        for i in range(n):
            s_ = slot[i]
            r = dict(slot=s_, ids=[], fin=None, max_new=int(reqs[i]["max_new"]), stop=stop, sampled=is_s[i],
                     cached=len(snap[i]["ids"]) if snap[i] else 0, restored=(len(snap[i]["ids"]) if ent[i] is not None else 0),
                     t_admit=t0, t_first=None, t_end=None, calls=0, prompt_len=len(prompts[i]), lh=[], draft=reqs[i].get("draft", True),
                     prompt=prompts[i], cur=start[i], pts=pts[i], scope=sc[i][1], own=sc[i], made=set(),
                     new_snaps=keep_snaps[i] if pc else None)
            out_all[good[i]] = r
            self.rows[s_] = r
            self.order.append(r)
            self.act[s_], self.pos[s_] = False, self.CTX
        self.cached_now = [r["cached"] for r in out_all]
        self.stats["requests"] += n
        self.stats["prompt_tokens"] += sum(len(p) for p in prompts)
        self.stats["cached_tokens"] += sum(out_all[g]["cached"] for g in good)
        self.stats["restored_tokens"] += sum(out_all[g]["restored"] for g in good)
        return out_all

    def _plan(self, pre, decoding):
        """Prefill work for this step: (row, start, end), at most one chunk per row, first come first served, inside the
        token budget. A chunk stops at the row's next cache point, at its prompt end, or where the budget runs out (then on
        a multiple of ALIGN in absolute position when the chunk keeps at least one grid step)."""
        budget = self.budget_busy if decoding else self.budget_idle
        work, used = [], 0
        for r in pre:
            room = min(budget - used, self.chunk)
            if room <= 0:
                break
            cur, L = r["cur"], r["prompt_len"]
            stop_at = r["pts"][0] if r["pts"] else L
            end = min(stop_at, cur + room)
            if end < stop_at and (end // ALIGN) * ALIGN > cur:
                end = (end // ALIGN) * ALIGN
            work.append((r, cur, end))
            used += end - cur
        return work

    def _step(self, waiting=False, hold=False):
        """hold: prefill only; decoding rows wait (generate() with speculative decoding: every row's prefill first)."""
        S, CTX = self.S, self.CTX
        if not self.busy():
            return []
        pre = self.prefilling()
        act = self.act.copy()
        live = [s_ for s_ in range(S) if act[s_]] if not hold else []
        K = 1 if (waiting or pre) else self.K
        ck, cv = self.state
        self.state = None
        dec_out = None
        if live and self.DK and not self.spec_off and self._spec_call(live, K):     # speculation: every row in one program
            try:
                dec_out, ck, cv, K = self._spec_step(ck, cv, live, K)
            except Exception as e:      # the call replays with plain decoding while its state is unchanged, else A13
                if not self._spec_fault(e, ck, cv):
                    raise
        if live and dec_out is None:
            Tc = min(CTX, bucket(int(self.pos[act].max()) + K + 1, 128))
            sampled = any(self.rows[s_]["sampled"] for s_ in live)
            B = S if (sampled and not self.sampled_buckets) else self._rows(max(live) + 1, S)
            # exported decode kernels update the linear state of every row of the bucket: put back the state of rows that
            # are still prefilling inside it (plain JAX decode keeps them through the keep mask)
            held = [(r["slot"], self.take(cv, r["slot"])) for r in pre if self.kernels and r["slot"] < B]
            if sampled:
                step0 = jnp.asarray(np.asarray([len(self.rows[s_]["ids"]) if act[s_] else 0 for s_ in range(B)], i32))
                cut = (lambda a: a) if B == S else (lambda a: a[:B])
                out, ck, cv, t2, p2, cnt, seen, tr = self.decs(
                    self.W, ck, cv, jnp.asarray(self.tok[:B]), jnp.asarray(self.pos[:B]), jax.tree.map(cut, self.sp), step0,
                    cut(self.cnt), cut(self.seen), Tc=Tc, K=K, trace=self.trace)
                if B == S:
                    self.cnt, self.seen = cnt, seen
                else:
                    self.cnt, self.seen = self.cnt.at[:B].set(cnt), self.seen.at[:B].set(seen)
                self._note("decode_sampled", B, K, Tc)
            else:
                out, ck, cv, t2, p2, tr = self.dec(self.W, ck, cv, jnp.asarray(self.tok[:B]), jnp.asarray(self.pos[:B]),
                                                   Tc=Tc, K=K, trace=self.trace)
                self._note("decode", B, K, Tc)
            for s_, (r_, c_) in held:
                cv = self.put(cv, s_, r_, c_)
            dec_out = (out, t2, p2, tr)
            if self.DK:         # a plain call while speculation is on: each row's drafter state is rebuilt before its next one
                self.spec_stats["plain_calls"] += 1
                for s_ in live:
                    self.rows[s_]["hist_ok"] = False
        work = self._plan(pre, bool(live))
        nxt, last = {}, {}
        if work:
            (ck, cv), nxt, last = self.prefill((ck, cv), [w[0]["prompt"] for w in work], [w[0]["slot"] for w in work],
                                               [w[1] for w in work], [w[2] for w in work], lazy=True)
        self.state = (ck, cv)
        ended = []
        if dec_out is not None:
            out, t2, p2, tr = dec_out
            toks, t2, p2 = np.asarray(out), np.asarray(t2), np.asarray(p2)
            if self.trace:
                th, tg = np.asarray(tr[0]), np.asarray(tr[1])
            for s_ in live:
                r = self.rows[s_]
                r["calls"] += 1
                fin = None
                for k in range(K):
                    t = int(toks[k, s_])
                    if t < 0:           # past the tokens this row committed (speculative decoding)
                        break
                    if t in r["stop"]:
                        fin = "stop"
                        break
                    r["ids"].append(t)
                    if self.trace:
                        r["lh"].append((int(th[k, s_]), float(tg[k, s_])))
                    if len(r["ids"]) >= r["max_new"]:
                        fin = "length"
                        break
                if fin:
                    self._end(r, fin)
                    ended.append(r)
                else:
                    self.tok[s_], self.pos[s_] = t2[s_], p2[s_]
        done = []
        for r, a, b in work:
            r["calls"] += 1
            r["cur"] = b
            if r["pts"] and b == r["pts"][0]:
                r["pts"].pop(0)
                x = self._snap(self.state, r["slot"], r["prompt"][:b], r["own"])
                r["new_snaps"].append(x)
                r["made"].add(id(x))
            if b == r["prompt_len"]:
                if self.prefix_cache:
                    x = self._snap(self.state, r["slot"], r["prompt"], r["own"])
                    r["new_snaps"].append(x)
                    r["made"].add(id(x))
                self._commit_snaps(r)
                done.append(r)
        if done:
            ended += self._first(done, nxt, last)
        return ended

    def _last_rows(self, slots, last):
        """(S, V) f32 last logits by slot for `slots`, from prefill(lazy=True)'s (logits, row) pairs: one gather_last call
        per prefill call (its shape is the call's row bucket, all warmed), the other rows zero."""
        S, buf, by = self.S, self._zl(), {}
        for s_ in slots:
            lg, b = last[s_]
            by.setdefault(id(lg), (lg, []))[1].append((s_, b))
        for lg, sb in by.values():
            src, dst = np.zeros(S, i32), np.full(S, S, i32)
            for k, (s_, b) in enumerate(sb):
                src[k], dst[k] = b, s_
            buf = self.gather_last(buf, lg, src, dst)
        return buf

    def _first(self, done, nxt, last):
        """First token of rows whose prefill ended in this step (sampled rows through the sampled first-token program).
        Rows sit at their slot's index in every program here, so each program has one shape whatever the row count; a
        row's draw depends on its seed and token index only (sampler.py), never on its place in the call."""
        S, V = self.S, self.c.V
        t_first = {r["slot"]: nxt[r["slot"]] for r in done}
        new = [r for r in done if r["sampled"]]
        buf = self._last_rows([r["slot"] for r in (done if self.trace else new)], last) if (new or self.trace) else None
        if new:
            rows_np, seen_np, nx = np.full(S, S, i32), np.zeros((S, V), bool), np.zeros(S, i32)
            for r in new:
                s_ = r["slot"]
                rows_np[s_], nx[s_] = s_, nxt[s_]
                seen_np[s_, np.asarray(r["prompt"], np.int64)] = True
            tk, self.cnt, self.seen = self.first_rows(buf, nx, self.sp, self.cnt, self.seen, rows_np, seen_np)
            tk = np.asarray(tk)
            for r in new:
                t_first[r["slot"]] = int(tk[r["slot"]])
        if self.trace:
            h, g = self.lhash(buf)
            h, g = np.asarray(h), np.asarray(g)
        now, ended = time.perf_counter(), []
        for b, r in enumerate(done):
            s_ = r["slot"]
            r["t_first"] = now
            t = int(t_first[s_])
            if t in r["stop"]:
                self._end(r, "stop")
                ended.append(r)
                continue
            r["ids"].append(t)
            if self.trace:
                r["lh"].append((int(h[s_]), float(g[s_])))
            if len(r["ids"]) >= r["max_new"]:
                self._end(r, "length")
                ended.append(r)
                continue
            self.tok[s_], self.pos[s_], self.act[s_] = t, r["prompt_len"], True
        return ended

    def _generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, scopes=None):
        """prompts: up to S lists of token ids. samp[i]: None (greedy) or dict(temperature, top_k, top_p, min_p, seed,
        repetition_penalty, frequency_penalty, presence_penalty). on_step(ids, finish) streams after every device call and may
        return the indexes of prompts to end now (finish "cancel").
        marks[i]: with the prefix cache, points to remember inside prompt i: an int or a sorted list of token positions
        (message starts). The last one is always kept (the start of the last message); the others only when the slot has
        no point yet or they are more than ckpt_step tokens past its latest point (llama.cpp's user-message checkpoints).
        scopes[i]: with the prefix cache, the cache scope of prompt i (any hashable; None = the shared scope).
        Runs on the continuous-batching path (admit, then step until its rows end); no other row may be live.
        -> dict(ids, finish, timing, cached (prompt tokens served from the prefix cache, per prompt))."""
        n, S = len(prompts), self.S
        if not 0 < n <= S:
            raise ValueError(f"1..{S} prompts per call")
        if self.busy():
            raise RuntimeError("generate() while rows are live (the server's continuous batching owns the slots)")
        if any(len(p) + max_new > self.CTX or not len(p) for p in prompts):
            raise ValueError(f"each prompt needs 1 token or more and prompt + max_new <= {self.CTX}")
        samp = list(samp or [None] * n)
        marks = list(marks or [None] * n)
        sc = list(scopes or [None] * n)
        t0 = time.perf_counter()
        reqs = [dict(ids=p, max_new=max_new, samp=samp[i], mark=marks[i], scope=sc[i]) for i, p in enumerate(prompts)]
        bad = [w for w in (self.check(q) for q in reqs) if w is not None]
        if bad:
            raise ValueError(bad[0])
        rows = self._admit(reqs, stop_ids)
        ids = [r["ids"] for r in rows]
        fins = lambda: [r["fin"] for r in rows]
        calls = 0
        while self.busy():
            self._step()
            calls += 1
            if on_step is not None:
                self.cancel([rows[i] for i in on_step(ids, fins()) or ()])
        t2_ = time.perf_counter()
        t1 = max([r["t_first"] for r in rows if r["t_first"] is not None] or [t2_])
        res = dict(ids=ids, finish=fins(), cached=[r["cached"] for r in rows], restored=[r["restored"] for r in rows],
                   slots=[r["slot"] for r in rows],
                   timing=dict(prefill_s=t1 - t0, decode_s=t2_ - t1, decode_calls=calls,
                               new_tokens=sum(len(x) for x in ids), sampled=any(r["sampled"] for r in rows)))
        if self.trace:
            res["lh"] = [r["lh"] for r in rows]
        return res

    # per-row backoff (POLICY, the row's own history only): after BO_MISS rounds in a row with 0 accepted drafts the row
    # drafts nothing for BO_WAIT rounds, then tries again; each further failure doubles the wait up to BO_MAX; any
    # accepted draft resets it. Only for drafters that ask for it (n-gram).
    BO_MISS, BO_WAIT, BO_MAX = 4, 16, 256

    def _k_row(self, r):
        """This row's own draft cap: 0 for rows that do not speculate (draft=False, e.g. constrained rows) and rows in
        backoff, else k_max. Sampled and penalised rows speculate: they verify by the coupled draw (spec_step.coupled),
        with penalty counts at each position of the verify pass."""
        if not r.get("draft", True):
            return 0
        bo = r.get("bo")
        return 0 if (bo and bo["wait"] > 0) else self.DK

    def _backoff(self, r, n_, kr):
        """one round of row r: n_ committed tokens with draft cap kr"""
        if not self.drafter.backoff:
            return
        bo = r.setdefault("bo", dict(miss=0, wait=0, span=self.BO_WAIT))
        if bo["wait"] > 0:
            bo["wait"] -= 1
            return
        if kr <= 0:
            return
        if n_ > 1:
            bo["miss"], bo["span"] = 0, self.BO_WAIT
        elif n_ == 1:
            bo["miss"] += 1
            if bo["miss"] >= self.BO_MISS:
                bo["miss"], bo["wait"], bo["span"] = 0, bo["span"], min(self.BO_MAX, bo["span"] * 2)
                self.spec_stats["backoffs"] += 1

    def _spec_rows(self, live):
        """the row bucket of a decode call over these live rows (the speculation and the plain program use the same)"""
        sampled = any(self.rows[s_]["sampled"] for s_ in live)
        return self.S if (sampled and not self.sampled_buckets) else self._rows(max(live) + 1, self.S)

    def _spec_call(self, live, K):
        """Whether this step's decode runs the speculation program. Drafters that allow plain calls between theirs
        (n-gram) take the plain decode program when the call's row bucket has k = 0 in the k_table (measured slower than
        plain there), and when no live row would draft anything (each row's own state decides: its cap and the host
        reference of its drafts), so a call with nothing to verify costs no verify width."""
        if not self.drafter.plain_ok:
            return True
        if self.k_table.get(self._spec_rows(live), self.DK) > 0:
            for s_ in live:
                r = self.rows[s_]
                if self._k_row(r) > 0 and r["max_new"] - len(r["ids"]) > 1 and self.drafter.found(self, r, self.DK) > 0:
                    return True
        for s_ in live:         # a row in backoff counts its plain rounds too
            bo = self.rows[s_].get("bo")
            if bo and bo["wait"] > 0:
                bo["wait"] = max(0, bo["wait"] - (self.K if K > 1 else 1))
        return False

    def _spec_step(self, ck, cv, live, K):
        """The decode part of one step with speculation on: one draft / verify / commit program for every decoding row
        (self.NR rounds when the plain path would run K steps, else 1). Each row drafts from its drafter's state, the
        target verifies [token, drafts], each row commits only its own accepted tokens (n = a + 1, its k capped by
        _k_row, its remaining max_tokens and the drafts found), and the architecture's commit keeps exactly those steps;
        rejected drafts leave nothing (their K/V sit past the row's position, masked and written over later).
        -> ((out (N * (k + 1), S) committed tokens per row in order, -1 after them; next token; next position; trace), ck,
        cv, N * (k + 1)) in the form _step reads for the plain decode call."""
        S, CTX = self.S, self.CTX
        N = self.NR if K > 1 else 1
        sampled = any(self.rows[s_]["sampled"] for s_ in live)
        B = self._spec_rows(live)
        k = self.k_table.get(B, self.DK)
        self._spec_ran = False
        X = self.drafter.host_args(self, B, live, k)
        tok, pos, kr, rem, st = np.zeros(B, i32), np.full(B, CTX, i32), np.zeros(B, i32), np.zeros(B, i32), np.zeros(B, i32)
        for s_ in live:
            r = self.rows[s_]
            tok[s_], pos[s_], kr[s_] = self.tok[s_], self.pos[s_], min(k, self._k_row(r))
            rem[s_], st[s_] = r["max_new"] - len(r["ids"]), len(r["ids"])
        Tc = min(CTX, bucket(int(pos[live].max()) + N * (k + 1) + k + 1, 128))
        a = jnp.asarray
        if sampled:
            cut = (lambda x: x) if B == S else (lambda x: x[:B])
            sp, cnt, seen = jax.tree.map(cut, self.sp), cut(self.cnt), cut(self.seen)
        else:
            sp = cnt = seen = None
        out, ck, cv, self.dstate, cn, t2, p2, cnt, seen, tr = self.sstep(
            self.W, ck, cv, self.drafter.W, self.dstate, tuple(a(x) for x in X), a(tok), a(pos), a(kr), a(rem), sp,
            None if sp is None else a(st), cnt, seen, Tc=Tc, K=k, N=N, trace=self.trace)
        self._spec_ran = True       # dispatched: the donated slot state now belongs to this call
        if sampled:
            if B == S:
                self.cnt, self.seen = cnt, seen
            else:
                self.cnt, self.seen = self.cnt.at[:B].set(cnt), self.seen.at[:B].set(seen)
        self._note("spec", B, k, N, Tc)
        out, cn, t2, p2 = np.asarray(out), np.asarray(cn), np.asarray(t2), np.asarray(p2)
        L = N * (k + 1)
        toks, t2s, p2s = np.full((L, S), -1, i32), np.zeros(S, i32), np.full(S, CTX, i32)
        th, tg = (np.zeros((L, S), np.uint32), np.zeros((L, S), np.float32)) if self.trace else (None, None)
        trn = (np.asarray(tr[0]), np.asarray(tr[1])) if self.trace else None
        for s_ in live:
            seq, last = [], None
            r = self.rows[s_]
            for i in range(N):
                n_ = int(cn[i, s_])
                if n_ == 0:
                    continue
                if self.trace:
                    th[len(seq):len(seq) + n_, s_], tg[len(seq):len(seq) + n_, s_] = trn[0][i, s_, :n_], trn[1][i, s_, :n_]
                seq += [int(x) for x in out[i, s_, :n_]]
                last = n_
                self.spec_stats["rounds"] += 1
                self.spec_stats["tokens"] += n_
                self.spec_stats["hist"][n_] += 1
                self._backoff(r, n_, int(kr[s_]))
            toks[:len(seq), s_] = seq
            t2s[s_], p2s[s_] = t2[s_], p2[s_]
            self.drafter.host_after(self, r, seq, last, p2[s_])
            r["drafted"] = r.get("drafted", 0) + len(seq)
        return (toks, t2s, p2s, (th, tg)), ck, cv, L

    def _spec_fault(self, e, ck, cv):
        """A speculation call raised (fallback F-spec-row). When it failed before the device took the slot state (its
        donated K/V and linear state are all still there: tracing, compiling, the drafter's host side), the state is
        unchanged: this call replays with the plain decode program (warmed), and speculation stays off for this engine's
        life (spec_off, spec_why; spec_stats faults). -> True. Else False: the state is gone, so the A13 failure rules
        apply (the call's streams end with the error and the worker restarts), never a silent re-run."""
        if getattr(self, "_spec_ran", False) or any(getattr(x, "is_deleted", lambda: False)() for x in jax.tree.leaves((ck, cv))):
            return False
        self.spec_off = True
        self.spec_why = f"speculation off after a failed speculation call (plain decoding): {type(e).__name__}: {e}"[:400]
        self.spec_stats["faults"] = self.spec_stats.get("faults", 0) + 1
        import json
        import sys
        print(json.dumps(dict(event="speculation", used=False, reason=self.spec_why, faults=self.spec_stats["faults"])),
              file=sys.stderr, flush=True)       # the startup log's speculation line, for a fault mid-run
        return True

    def spec_health(self):
        """The engine's speculation state: on (a drafter is built and no speculation call failed), its method, draft length
        and measured k_table, why decoding is plain when it is (spec_why: the load's reason, the k = 0 rule, a drop before
        the warm-up or a failed call), a copy of spec_stats (rounds, tokens, faults, plain calls, back-offs, the accepted
        length histogram) and the spec_step version. A server that fell back to plain mid-run shows it here."""
        from .spec_step import SPEC_STEP_VERSION
        st = {k: (list(v) if isinstance(v, list) else v) for k, v in self.spec_stats.items()}
        st.setdefault("faults", 0)
        on = bool(self.DK) and not self.spec_off
        return dict(on=on, method=getattr(self.drafter, "kind", None) if self.DK else None, draft_k=self.DK,
                    k_table={str(b): k for b, k in sorted(self.k_table.items())},
                    reason=None if on else (self.spec_why or "no drafter: plain decoding"), stats=st,
                    spec_step_version=SPEC_STEP_VERSION)

    def spec_programs(self, ck, cv):
        """Every speculation program a request can reach, as warm() lists its calls (kind, jitted function, arguments after
        (W, ck, cv), static arguments): the drafter's prefill program over each prefill shape (MTP), and the step program
        at every attention window, greedy at each row bucket (with its k) and sampled at S rows, for 1 round and NR
        rounds. [] when off. The drafter's host-called programs (n-gram: its row put) are warm()'s helpers."""
        if not self.DK:
            return []
        S, CTX, D, a = self.S, self.CTX, self.c.D, jnp.asarray
        dr = self.drafter
        calls = []
        if self.dpf is not None:
            for B, T, Tc in self.prefill_shapes():
                hid = self._committed(jnp.zeros((B, T, D), jnp.bfloat16))
                calls.append(("spec", self.dpf, (dr.W, self.dstate, a(np.zeros((B, T), i32)), hid, a(np.zeros(B, i32)),
                                                 a(np.zeros(B, i32)), a(np.arange(B, dtype=i32)), a(np.full(B, S, i32)),
                                                 a(np.zeros(B, i32))), dict(Tc=Tc)))
        sp = SP.to_device(SP.pack([dict(temperature=0.7, top_p=0.9, top_k=20, presence_penalty=1.0, seed=1)] * S, S))
        cnt, seen = self._committed(jnp.zeros((S, self.c.V), jnp.float32)), self._committed(jnp.zeros((S, self.c.V), bool))

        def args(B, smp):
            z = lambda: a(np.zeros(B, i32))
            return (dr.W, self.dstate, tuple(a(x) for x in dr.host_zero(self, B)), z(), a(np.full(B, CTX, i32)), z(),
                    z(), sp if smp else None, z() if smp else None, cnt if smp else None, seen if smp else None)
        for Tc in self.windows(128):
            for N in sorted({1, self.NR}):
                for B in sorted({self._rows(n, S) for n in range(1, S + 1)}):
                    if self.k_table.get(B, self.DK) > 0:        # a bucket at k = 0 runs the plain decode program
                        calls.append(("spec", self.sstep, args(B, False), dict(Tc=Tc, K=self.k_table.get(B, self.DK), N=N,
                                                                                trace=self.trace)))
                if self.k_table.get(S, self.DK) > 0:
                    calls.append(("spec", self.sstep, args(S, True), dict(Tc=Tc, K=self.k_table.get(S, self.DK), N=N,
                                                                          trace=self.trace)))
        return calls

    def drop_speculation(self, failed=False):
        """Turn speculation off before the first request and the warm-up (cli.load does this when the drafter's gate passed
        for another component set than this deployment's). The engine is then the plain engine: no speculation program,
        the plain slot state (no extra K/V group), the drafter's weights released. RuntimeError once a speculation program
        ran, except failed=True (cli.warm: a speculation program failed to compile or run before ready), which is allowed
        while no row is live: the slot states, their snapshots and the host cache, which may hold the drafter's K/V, are
        made again without it. spec_why says why; cli.load sets its own reason after the call."""
        if not self.DK:
            return
        if failed and any(self.rows):       # the failed call's rows end (their streams get the error)
            self._drop_live()
        if any(self.rows) or (not failed and (self.state is not None or (self.dpf is not None and self.dpf._cache_size())
                                              or self.sstep._cache_size())):
            raise RuntimeError("drop_speculation: only before the first request and the warm-up")
        if failed:
            self.reset_cache()
            if self.host is not None:
                h, h.ns = HostCache(self.host.cap), self.host.ns
                self.host = h
        self.spec_why = ("speculation off: a speculation program failed before ready" if failed
                         else "speculation off before the warm-up")
        cfg, S, CTX, model = self.c, self.S, self.CTX, self.model
        self.DK, self.NR, self.k_table, self.drafter, self.dstate, self.dpf, self.sstep = 0, 1, {}, None, None, None, None
        self._dhid = False
        self.spec_stats, self.hlast = dict(rounds=0, tokens=0, hist=[0, 0], plain_calls=0, backoffs=0), {}
        mk = lambda: model.new_state(cfg, S - 1 if S <= 2 else S, CTX)        # as __init__ makes it without speculation
        self._new_state = jax.jit(mk, out_shardings=self.dev) if self.dev is not None else mk

    def logits(self, seq):
        """Teacher-forced f32 logits at every position of one sequence (tests)."""
        ck, cv = self.new_state()
        s, res, p = np.asarray(seq, i32), [], 0
        while p < len(s):
            n = min(self.chunk, len(s) - p)
            T = bucket(n, 16)
            toks = np.zeros((1, T), i32)
            toks[0, :n] = s[p:p + n]
            Tc = 0 if p == 0 else min(self.CTX, bucket(p, 128))
            r = self.fwd(self.W, ck, cv, jnp.asarray(toks), jnp.asarray([p], i32), jnp.asarray([n], i32),
                         jnp.zeros(1, i32), Tc=Tc, all_logits=True)
            ck, cv = r["ck"], r["cv"]
            res.append(np.asarray(r["lg"][0, :n]))
            p += n
        return np.concatenate(res)
