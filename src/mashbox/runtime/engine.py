"""Batched generation on one chip: S slots, chunked prefill, K decode steps per device call.

Greedy rows run the greedy decode program on the smallest batch bucket (1, 4 or S) that holds the active rows.
If any row samples or uses a penalty, the batch runs the sampled program on all S rows, so a row's draws never depend
on the batch shape.

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

Scopes (generate(scopes=...), the server's API keys): every saved point, in a slot or in host memory, belongs to the
scope of the request that made it, and a prompt resumes only from points of its own scope, so one key's prompts are never
served from (or reported as cached by) another key's. Without scopes everything is one scope, as before.
"""
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import sampler as SP

i32 = np.int32


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
        self.cap, self.items, self.clock = int(cap), [], 0

    def nbytes(self):
        return sum(e["bytes"] for e in self.items)

    def best(self, p, scope=None):
        """-> (length, entry, snapshot) of the longest saved point of this scope that is a strict prefix of prompt p
        (0, None, None)"""
        best = (0, None, None)
        for e in self.items:
            if e.get("scope") != scope:
                continue
            for sn in e["snaps"]:
                L = len(sn["ids"])
                if best[0] < L < len(p) and sn["ids"] == tuple(p[:L]):
                    best = (L, e, sn)
        return best

    def add(self, e):
        top = e["snaps"][-1]["ids"]
        # an entry is superseded when all its points are prefixes of the new entry's longest point (same scope only)
        self.items = [x for x in self.items if x.get("scope") != e.get("scope")
                      or not all(top[:len(sn["ids"])] == sn["ids"] for sn in x["snaps"])]
        self.clock += 1
        e["t"] = self.clock
        self.items.append(e)
        while self.nbytes() > self.cap:
            self.take(min(self.items, key=lambda x: x["t"]))

    def take(self, e):
        self.items = [x for x in self.items if x is not e]     # by identity: entries hold arrays, == would compare them


class Engine:
    def __init__(self, cfg, W, model, CTX=2048, S=8, K=8, decode_kernels=None, chunk=512, tokens_per_call=2048,
                 prefix_cache=False, attn_bytes=2e9, cache_ram=8 << 30, ckpts=8, ckpt_step=8192, mtp=None, draft_k=0,
                 spec_rounds=None):
        self.c, self.W, self.CTX, self.S, self.K = cfg, W, CTX, S, K
        self.chunk, self.tokens_per_call = chunk, tokens_per_call
        self.prefix_cache, self.attn_bytes = prefix_cache, attn_bytes
        self.state, self.snaps, self.used, self.clock = None, [[] for _ in range(S)], [0] * S, 0
        self.ckpts, self.ckpt_step = max(2, int(ckpts)), int(ckpt_step)
        self.host = HostCache(cache_ram) if (prefix_cache and cache_ram > 0) else None
        self.kv_block = min(2048, CTX)      # host copies move K/V in blocks of this many positions (small device temp)
        self.stats = dict(requests=0, prompt_tokens=0, cached_tokens=0, restored_tokens=0, saves=0)
        self.model = model
        fwd = model.make_forward(cfg, decode_kernels)
        # Speculative decoding (mtp = (spec module, MTP weights), draft_k drafts per round, spec_rounds rounds per device call):
        # greedy and sampled rows without penalties; a batch with a penalty runs the plain sampled program.
        self.DK = int(draft_k) if mtp is not None else 0
        self.spec_stats = dict(rounds=0, tokens=0, hist=[0] * (self.DK + 2))
        self.hlast = {}
        if self.DK:
            smod, self.Wm = mtp
            mfn = smod.make_mtp(cfg)
            sdraft, srounds = smod.make_spec(cfg, fwd, mfn)
            self.NR = int(spec_rounds or max(1, K // 2))
            self.mtp_pf = jax.jit(mfn, static_argnames=("Tc",), donate_argnums=(2,))
            self.sdraft = jax.jit(sdraft, static_argnames=("Tc", "K"), donate_argnums=(2,))
            self.srounds = jax.jit(srounds, static_argnames=("Tc", "K", "N"), donate_argnums=(2, 3))
        self.kernels = sorted(decode_kernels) if decode_kernels else []
        self.shapes = None          # set to a list to record the program shapes one call runs

        def dec(W, ck, cv, tok, pos, Tc, K):
            B = tok.shape[0]
            slots, ones = jnp.arange(B, dtype=jnp.int32), jnp.ones((B,), jnp.int32)

            def step(i, c_):
                ck, cv, tok, pos, out = c_
                r = fwd(W, ck, cv, tok[:, None], pos, ones, slots, Tc, contig=True)
                return r["ck"], r["cv"], r["nxt"], pos + 1, out.at[i].set(r["nxt"])
            ck, cv, tok, pos, out = lax.fori_loop(0, K, step, (ck, cv, tok, pos, jnp.zeros((K, B), jnp.int32)))
            return out, ck, cv, tok, pos

        def decs(W, ck, cv, tok, pos, sp, step0, cnt, seen, Tc, K):
            B = tok.shape[0]
            slots, ones = jnp.arange(B, dtype=jnp.int32), jnp.ones((B,), jnp.int32)
            plain = SP.no_penalty(sp)

            def step(i, c_):
                ck, cv, tok, pos, out, cnt, seen = c_
                r = fwd(W, ck, cv, tok[:, None], pos, ones, slots, Tc, contig=True)
                x = SP.penalise(r["lg"], cnt, seen, sp)
                nt = SP.sample(x, sp, step0 + i, greedy_tok=jnp.where(plain, r["nxt"], jnp.argmax(x, -1).astype(jnp.int32)))
                b_ = jnp.arange(B)
                return r["ck"], r["cv"], nt, pos + 1, out.at[i].set(nt), cnt.at[b_, nt].add(1.0), seen.at[b_, nt].set(True)
            init = (ck, cv, tok, pos, jnp.zeros((K, B), jnp.int32), cnt, seen)
            ck, cv, tok, pos, out, cnt, seen = lax.fori_loop(0, K, step, init)
            return out, ck, cv, tok, pos, cnt, seen

        def first(lg, nxt, sp, cnt, seen):
            x = SP.penalise(lg, cnt, seen, sp)
            tok = SP.sample(x, sp, jnp.zeros(lg.shape[0], jnp.int32),
                            greedy_tok=jnp.where(SP.no_penalty(sp), nxt, jnp.argmax(x, -1).astype(jnp.int32)))
            b_ = jnp.arange(lg.shape[0])
            return tok, cnt.at[b_, tok].add(1.0), seen.at[b_, tok].set(True)

        self.fwd = jax.jit(fwd, static_argnames=("Tc", "contig", "all_logits", "verify", "hidden"), donate_argnums=(1, 2))
        self.dec = jax.jit(dec, static_argnames=("Tc", "K"), donate_argnums=(1, 2))
        self.decs = jax.jit(decs, static_argnames=("Tc", "K"), donate_argnums=(1, 2))
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
        self.kv_put = jax.jit(kv_put, donate_argnums=(0,))
        # Weights on one device: the slot state is made there as committed arrays (shard.py does the same on a mesh). JAX keys
        # a compiled program on whether each argument is committed, and every program's outputs from a committed state are
        # committed; a fresh uncommitted state made the first prefill and decode of each new state programs of their own,
        # which the warm-up never ran (measured: a 5.4 s compile inside the first reply on a TPU v6e-1, ctx 2048 x 8 slots).
        devs = {d for a in jax.tree.leaves(W) if hasattr(a, "devices") for d in a.devices()}
        self.dev = jax.sharding.SingleDeviceSharding(devs.pop()) if len(devs) == 1 else None
        # with speculation the MTP head keeps its K/V in one extra group
        xg = dict(extra_groups=1) if self.DK else {}
        mk = lambda: model.new_state(cfg, S - 1 if S <= 2 else S, CTX, **xg)
        self._new_state = jax.jit(mk, out_shardings=self.dev) if self.dev is not None else mk

    def new_state(self):
        # The model adds one scratch slot for padding rows. With S <= 2 no call ever has a padding row (prefill and decode
        # use 1 or S rows), so the scratch slot is left out: at a long context that is one slot of K/V less on the chip.
        return self._new_state()

    def _committed(self, x):
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
    def prefill(self, state, prompts, slots=None, starts=None, ends=None):
        """Run each prompt into its slot from starts[i] to ends[i] (defaults: slot i, 0, the whole prompt).
        -> state, greedy next token per slot, last logits per slot (only for rows that reached the end of their prompt)."""
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
                         hidden=bool(self.DK))
            ck, cv = r["ck"], r["cv"]
            self._note("prefill", B, T, Tc)
            done = [b for b, w in enumerate(grp) if w[2] + int(lens[b]) >= len(w[1])]
            if self.DK:     # the MTP head over this chunk: position j reads the token at j + 1 and the hidden state at j
                tn, lm = np.zeros((B, T), i32), np.zeros(B, i32)
                for b, (slot, ids, s0, e) in enumerate(grp):
                    m_ = max(0, min(int(lens[b]), len(ids) - 1 - s0))
                    tn[b, :m_], lm[b] = ids[s0 + 1:s0 + 1 + m_], m_
                _, _, ck = self.mtp_pf(self.W, self.Wm, ck, jnp.asarray(tn), r["hid"], jnp.asarray(p0), jnp.asarray(lm),
                                       jnp.asarray(sl), Tc=Tc)
                for b in done:
                    self.hlast[grp[b][0]] = r["hid"][b, int(lens[b]) - 1]
            if done:
                nx, lg = np.asarray(r["nxt"]), r["lg"]
                for b in done:
                    nxt[grp[b][0]], last[grp[b][0]] = int(nx[b]), lg[b]
            for b, w in enumerate(grp):
                w[2] += int(lens[b])
            work = [w for w in work if w[2] < w[3]]
        return (ck, cv), nxt, last

    # ------------------------------------------------------------------------------------------------------- prefix cache
    def _place(self, prompts, scopes=None):
        """-> slot per prompt, cached snapshot (or None) per prompt, host cache entry to restore (or None) per prompt.
        Longest cached prefix first (in a slot, or in host memory when that is longer), then empty or least recently used
        slots (lowest index first, so one user decodes on the smallest batch)."""
        n, S = len(prompts), self.S
        sc = list(scopes or [None] * n)
        slot, snap, ent = [None] * n, [None] * n, [None] * n
        if self.prefix_cache and self.state is not None:
            hits = []
            for i, p in enumerate(prompts):
                for s_ in range(S):
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
        free = sorted((s_ for s_ in range(S) if s_ not in slot), key=lambda s_: (self.used[s_], s_))
        for i in range(n):
            if slot[i] is None:
                slot[i] = free.pop(0)
        return slot, snap, ent

    def _save(self, state, s_):
        """Copy slot s_ (its K/V up to its longest point and its snapshots) to the host cache."""
        sn = self.snaps[s_]
        L, n = len(sn[-1]["ids"]), self.kv_block
        per = sum(a.dtype.itemsize * a.shape[0] * int(np.prod(a.shape[3:])) for a in jax.tree.leaves(state[0]))
        if -(-L // n) * n * per + sum(x["r"].nbytes + x["c"].nbytes for x in sn) > self.host.cap:
            return                          # larger than the whole cache: not worth the copy
        kv = [(min(p, self.CTX - n), jax.tree.map(np.asarray, self.kv_take(state[0], s_, min(p, self.CTX - n), n=n)))
              for p in range(0, L, n)]
        self.host.add(dict(kv=kv, snaps=list(sn), scope=sn[-1].get("scope"),
                           bytes=_nbytes([b for _, b in kv]) + sum(x["r"].nbytes + x["c"].nbytes for x in sn)))
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

    def _snap(self, state, s, ids, scope=None):
        r, c = self.take(state[1], s)
        return dict(ids=tuple(int(t) for t in ids), r=np.asarray(r), c=np.asarray(c), scope=scope)

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
        runner precompiles its buckets before serving): each prefill program (prefill_shapes) and each decode program (every
        attention window 128 ... CTX; greedy at each row bucket, sampled at S rows). Arguments are made as a request makes
        them (committed slot state and sampler counts, host-made token arrays), since JAX keys programs on that too.
        The programs are compiled first in `workers` threads (XLA compiles on the host), then each runs once, which loads it;
        both are served from the compile cache when it holds them. Call before the first request: the slot states are
        cleared after. -> dict(prefill=programs, decode=programs, compile_s, run_s, compile_failed)."""
        t0, S, CTX, V, a = time.perf_counter(), self.S, self.CTX, self.c.V, jnp.asarray
        ck, cv = self.state if self.state is not None else self.new_state()
        self.state = None
        calls = []
        for B, T, Tc in self.prefill_shapes():
            calls.append(("prefill", self.fwd, (a(np.zeros((B, T), i32)), a(np.zeros(B, i32)), a(np.ones(B, i32)),
                                                  a(np.arange(B, dtype=i32))), dict(Tc=Tc, hidden=bool(self.DK))))
        n_pre = len(calls)
        sp = SP.to_device(SP.pack([dict(temperature=0.7, top_p=0.9, top_k=20, presence_penalty=1.0, seed=1)] * S, S))
        cnt, seen = self._committed(jnp.zeros((S, V), jnp.float32)), self._committed(jnp.zeros((S, V), bool))
        for Tc in self.windows(128):
            for B in sorted({self._rows(n, S) for n in range(1, S + 1)}):
                calls.append(("decode", self.dec, (a(np.zeros(B, i32)), a(np.full(B, CTX - 1, i32))), dict(Tc=Tc, K=self.K)))
            calls.append(("decode", self.decs, (a(np.zeros(S, i32)), a(np.full(S, CTX - 1, i32)), sp, a(np.zeros(S, i32)),
                                                cnt, seen), dict(Tc=Tc, K=self.K)))
        failed = 0
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            def comp(c):
                try:
                    c[1].lower(self.W, ck, cv, *c[2], **c[3]).compile()
                    return 0
                except Exception:       # the run below compiles it then
                    return 1
            with ThreadPoolExecutor(workers) as ex:
                failed = sum(ex.map(comp, calls))
        t1 = time.perf_counter()
        for kind, f, args, kw in calls:
            r = f(self.W, ck, cv, *args, **kw)
            ck, cv = (r["ck"], r["cv"]) if kind == "prefill" else (r[1], r[2])
        jax.block_until_ready((ck, cv))
        ck = cv = None          # drop the warm-up state (rebound, not deleted: comp above closes over these names)
        self.reset_cache()
        t2 = time.perf_counter()
        return dict(prefill=n_pre, decode=len(calls) - n_pre, compile_s=round(t1 - t0, 1), run_s=round(t2 - t1, 1),
                    compile_failed=failed)

    # ------------------------------------------------------------------------------------------------------- generate
    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, scopes=None):
        try:
            return self._generate(prompts, max_new, stop_ids, samp, on_step, marks, scopes)
        except BaseException as e:
            if self.prefix_cache:      # the slot state may be half written or donated: start clean
                self.reset_cache()
            traceback.clear_frames(e.__traceback__)     # free the device state the failed call still holds
            raise

    def _cancel(self, rows, fin, act, slot, tok, pos):
        """End the rows on_step returned (the caller no longer wants them: client gone, stop string found) as a padding row
        would be. Their prompt points were saved before decoding, so the prefix cache keeps them."""
        for i in rows or ():
            if fin[i] is None:
                fin[i], act[slot[i]], tok[slot[i]], pos[slot[i]] = "cancel", False, 0, self.CTX - 1

    def _generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, scopes=None):
        """prompts: up to S lists of token ids. samp[i]: None (greedy) or dict(temperature, top_k, top_p, min_p, seed,
        repetition_penalty, frequency_penalty, presence_penalty). on_step(ids, finish) streams after every device call and may
        return the indexes of prompts to end now (finish "cancel").
        marks[i]: with the prefix cache, points to remember inside prompt i: an int or a sorted list of token positions
        (message starts). The last one is always kept (the start of the last message); the others only when the slot has
        no point yet or they are more than ckpt_step tokens past its latest point (llama.cpp's user-message checkpoints).
        scopes[i]: with the prefix cache, the cache scope of prompt i (any hashable; None = the shared scope).
        -> dict(ids, finish, timing, cached (prompt tokens served from the prefix cache, per prompt))."""
        n, S, CTX, V = len(prompts), self.S, self.CTX, self.c.V
        if not 0 < n <= S:
            raise ValueError(f"1..{S} prompts per call")
        if any(len(p) + max_new > CTX or not len(p) for p in prompts):
            raise ValueError(f"each prompt needs 1 token or more and prompt + max_new <= {CTX}")
        samp = list(samp or [None] * n)
        sampled = any(r and (float(r.get("temperature") or 0) > 0 or float(r.get("repetition_penalty") or 1.0) != 1.0
                             or float(r.get("frequency_penalty") or 0) != 0 or float(r.get("presence_penalty") or 0) != 0)
                      for r in samp)
        t0 = time.perf_counter()
        pc = self.prefix_cache
        sc = list(scopes or [None] * n)
        slot, snap, ent = self._place(prompts, sc)
        if pc and self.state is None:
            self.state = self.new_state()
        state = self.state if pc else self.new_state()
        self.state = None            # donated below; put back at the end
        start = [len(sn["ids"]) if sn else 0 for sn in snap]
        new_snaps, made = {}, set()
        if pc:
            for e in ent:
                if e is not None:    # restored below: it lives in its slot again (and is saved again if evicted)
                    self.host.take(e)
            for i in range(n):
                s_, old = slot[i], self.snaps[slot[i]]
                if self.host is not None and old and (ent[i] is not None or snap[i] is not old[-1]):
                    self._save(state, s_)            # this request does not continue the slot's conversation
            for i in range(n):
                s_ = slot[i]
                src = ent[i]["snaps"] if ent[i] is not None else self.snaps[s_]
                keep = [x for x in src if snap[i] is not None and len(x["ids"]) <= start[i] and x.get("scope") == sc[i]
                        and x["ids"] == tuple(prompts[i][:len(x["ids"])])]
                new_snaps[s_] = keep
                if ent[i] is not None:
                    state = self._restore(state, s_, ent[i])
                if snap[i] is not None:
                    state = (state[0], self.put(state[1], s_, jnp.asarray(snap[i]["r"]), jnp.asarray(snap[i]["c"])))
            pts = []
            for i, m in enumerate(marks or [None] * n):
                c = sorted(set(x for x in ([] if m is None else [m] if isinstance(m, int) else m) if start[i] < x < len(prompts[i])))
                last, take = (len(new_snaps[slot[i]][-1]["ids"]) if new_snaps[slot[i]] else -1), []
                for j, x in enumerate(c):
                    if j == len(c) - 1 or last < 0 or x > last + self.ckpt_step:
                        take.append(x)
                        last = x
                pts.append(take)
            while any(pts):
                idx = [i for i in range(n) if pts[i]]
                ends = [pts[i].pop(0) for i in idx]
                state, _, _ = self.prefill(state, [prompts[i] for i in idx], [slot[i] for i in idx], [start[i] for i in idx], ends)
                for i, e_ in zip(idx, ends):
                    x = self._snap(state, slot[i], prompts[i][:e_], sc[i])
                    new_snaps[slot[i]].append(x)
                    made.add(id(x))
                    start[i] = e_
        state, nxt, last = self.prefill(state, prompts, slot, start)
        if pc:
            for i in range(n):
                x = self._snap(state, slot[i], prompts[i], sc[i])
                new_snaps[slot[i]].append(x)
                made.add(id(x))
        stop = set(stop_ids)
        ids, fin = [[] for _ in range(n)], [None] * n
        row = {slot[i]: i for i in range(n)}
        tok, pos, act = np.zeros(S, i32), np.full(S, CTX - 1, i32), np.zeros(S, bool)
        if sampled:
            sp = SP.to_device(SP.pack([samp[row[s_]] if s_ in row else None for s_ in range(S)], S))
            seen_np = np.zeros((S, V), bool)
            for i, p in enumerate(prompts):
                seen_np[slot[i], np.asarray(p, np.int64)] = True
            cnt, seen = jnp.zeros((S, V), jnp.float32), jnp.asarray(seen_np)
            lg = jnp.stack([last[s_] if s_ in row else jnp.zeros((V,), jnp.float32) for s_ in range(S)])
            nx = jnp.asarray(np.asarray([nxt.get(s_, 0) for s_ in range(S)], i32))
            t_first, cnt, seen = self.first(lg, nx, sp, cnt, seen)
            t_first = np.asarray(t_first)
        else:
            t_first = np.zeros(S, i32)
            for i in range(n):
                t_first[slot[i]] = nxt[slot[i]]
        jax.block_until_ready(state)
        t1 = time.perf_counter()
        self.cached_now = [len(sn["ids"]) if sn else 0 for sn in snap]
        for i in range(n):
            t = int(t_first[slot[i]])
            if t in stop:
                fin[i] = "stop"
                continue
            ids[i].append(t)
            if len(ids[i]) >= max_new:
                fin[i] = "length"
                continue
            tok[slot[i]], pos[slot[i]], act[slot[i]] = t, len(prompts[i]), True
        if on_step is not None:
            self._cancel(on_step(ids, fin), fin, act, slot, tok, pos)
        calls = 0
        ck, cv = state
        pen = sampled and any(r and (float(r.get("repetition_penalty") or 1.0) != 1.0 or float(r.get("frequency_penalty") or 0) != 0
                                     or float(r.get("presence_penalty") or 0) != 0) for r in samp)
        if self.DK and not pen and act.any():
            ck, cv, calls = self._spec_decode(ck, cv, prompts, ids, fin, act, slot, row, tok, pos, sp if sampled else None,
                                              stop, max_new, on_step)
        while act.any():
            Tc = min(CTX, bucket(int(pos[act].max()) + self.K + 1, 128))
            if sampled:
                B = S
                step0 = jnp.asarray(np.asarray([len(ids[row[s_]]) if s_ in row else 0 for s_ in range(S)], i32))
                out, ck, cv, t2, p2, cnt, seen = self.decs(self.W, ck, cv, jnp.asarray(tok), jnp.asarray(pos), sp, step0,
                                                           cnt, seen, Tc=Tc, K=self.K)
                self._note("decode_sampled", B, self.K, Tc)
            else:
                B = self._rows(int(np.nonzero(act)[0].max()) + 1, S)
                out, ck, cv, t2, p2 = self.dec(self.W, ck, cv, jnp.asarray(tok[:B]), jnp.asarray(pos[:B]), Tc=Tc, K=self.K)
                self._note("decode", B, self.K, Tc)
            toks, t2, p2 = np.asarray(out), np.asarray(t2), np.asarray(p2)
            calls += 1
            for i in range(n):
                s_ = slot[i]
                if not act[s_]:
                    continue
                for k in range(self.K):
                    t = int(toks[k, s_])
                    if t in stop:
                        fin[i] = "stop"
                        break
                    ids[i].append(t)
                    if len(ids[i]) >= max_new:
                        fin[i] = "length"
                        break
                if fin[i]:
                    act[s_] = False
                else:
                    tok[s_], pos[s_] = t2[s_], p2[s_]
            tok, pos = np.where(act, tok, 0).astype(i32), np.where(act, pos, CTX - 1).astype(i32)
            if on_step is not None:
                self._cancel(on_step(ids, fin), fin, act, slot, tok, pos)
        if pc:
            self.state = (ck, cv)
            self.clock += 1
            for s_, keep in new_snaps.items():
                self.snaps[s_], self.used[s_] = self._trim(keep, made), self.clock      # host copies, about 154 MB each here
        t2_ = time.perf_counter()
        cached = [len(sn["ids"]) if sn else 0 for sn in snap]
        restored = [c if e is not None else 0 for c, e in zip(cached, ent)]
        self.stats["requests"] += n
        self.stats["prompt_tokens"] += sum(len(p) for p in prompts)
        self.stats["cached_tokens"] += sum(cached)
        self.stats["restored_tokens"] += sum(restored)
        return dict(ids=ids, finish=fin, cached=cached, restored=restored, slots=slot,
                    timing=dict(prefill_s=t1 - t0, decode_s=t2_ - t1, decode_calls=calls,
                                new_tokens=sum(len(x) for x in ids), sampled=sampled))

    def _spec_decode(self, ck, cv, prompts, ids, fin, act, slot, row, tok, pos, sp, stop, max_new, on_step):
        """Speculative decode loop (see models/<model>/spec.py): rounds of verify / accept / commit / draft, NR per call."""
        S, CTX, K, N, D = self.S, self.CTX, self.DK, self.NR, self.c.D
        B = S if sp is not None else self._rows(int(np.nonzero(act)[0].max()) + 1, S)
        hid = jnp.stack([self.hlast[s_] if (act[s_] and s_ in self.hlast) else jnp.zeros((D,), jnp.bfloat16) for s_ in range(B)])
        p_init = np.where(act[:B], pos[:B] - 1, CTX - 1).astype(i32)
        Tc = min(CTX, bucket(int(p_init.max()) + 1, 128))
        step = lambda: jnp.asarray(np.asarray([len(ids[row[s_]]) if s_ in row else 0 for s_ in range(B)], i32))
        spB = None if sp is None else {k_: v_[:B] for k_, v_ in sp.items()}
        d, dlg, ck = self.sdraft(self.W, self.Wm, ck, jnp.asarray(tok[:B])[:, None], hid[:, None], jnp.asarray(p_init),
                                 jnp.ones((B,), i32), Tc=Tc, K=K, sp=spB, step0=None if sp is None else step())
        self._note("spec_draft", B, K, Tc)
        calls = 0
        while act.any():
            Bn = S if sp is not None else self._rows(int(np.nonzero(act)[0].max()) + 1, S)
            if Bn < B:
                B, d, dlg = Bn, d[:Bn], dlg[:Bn]
                spB = None if sp is None else {k_: v_[:B] for k_, v_ in sp.items()}
            Tc = min(CTX, bucket(int(pos[act].max()) + N * (K + 1) + K + 1, 128))
            out, cnt, ck, cv, t2, d, dlg, p2 = self.srounds(self.W, self.Wm, ck, cv, jnp.asarray(tok[:B]), d, dlg,
                                                            jnp.asarray(pos[:B]), Tc=Tc, K=K, N=N, sp=spB,
                                                            step0=None if sp is None else step())
            self._note("spec", B, K, N, Tc)
            out, cnt, t2, p2 = np.asarray(out), np.asarray(cnt), np.asarray(t2), np.asarray(p2)
            calls += 1
            for i in range(len(prompts)):
                s_ = slot[i]
                if not act[s_]:
                    continue
                for r_ in range(N):
                    n_ = int(cnt[r_, s_])
                    self.spec_stats["rounds"] += 1
                    self.spec_stats["tokens"] += n_
                    self.spec_stats["hist"][n_] += 1
                    for k in range(n_):
                        t = int(out[r_, s_, k])
                        if t in stop:
                            fin[i] = "stop"
                            break
                        ids[i].append(t)
                        if len(ids[i]) >= max_new:
                            fin[i] = "length"
                            break
                    if fin[i]:
                        break
                if fin[i]:
                    act[s_] = False
                else:
                    tok[s_], pos[s_] = t2[s_], p2[s_]
            tok[:], pos[:] = np.where(act, tok, 0).astype(i32), np.where(act, pos, CTX - 1).astype(i32)
            if on_step is not None:
                self._cancel(on_step(ids, fin), fin, act, slot, tok, pos)
        return ck, cv, calls

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
