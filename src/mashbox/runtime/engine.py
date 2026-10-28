"""Batched generation on one chip: S slots, chunked prefill, K decode steps per device call.

Greedy rows run the greedy decode program on the smallest batch bucket (1, 4 or S) that holds the active rows.
If any row samples or uses a penalty, the batch runs the sampled program on all S rows, so a row's draws never depend
on the batch shape.

Prefix cache (prefix_cache=True, used by the server): the slot state lives on between calls. Each slot remembers the
token ids its state holds at up to two points (`mark`, usually the start of the last message, and the end of the
prompt), with a host copy of the slot's linear-attention state at each point (the full-attention K/V stay in the slot).
A new prompt that starts with one of those token lists goes to that slot and prefills only the tokens after it.
Agents resend the whole conversation every turn, so each turn prefills only what is new.
Scopes (generate(scopes=...), the server's API keys): every saved point belongs to the scope of the request that made
it, and a prompt resumes only from points of its own scope. Without scopes everything is one scope, as before.
"""
import time

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


class Engine:
    def __init__(self, cfg, W, model, CTX=2048, S=8, K=8, decode_kernels=None, chunk=512, tokens_per_call=2048,
                 prefix_cache=False, attn_bytes=2e9):
        self.c, self.W, self.CTX, self.S, self.K = cfg, W, CTX, S, K
        self.chunk, self.tokens_per_call = chunk, tokens_per_call
        self.prefix_cache, self.attn_bytes = prefix_cache, attn_bytes
        self.state, self.snaps, self.used, self.clock = None, [[] for _ in range(S)], [0] * S, 0
        self.stats = dict(requests=0, prompt_tokens=0, cached_tokens=0)
        self.model = model
        fwd = model.make_forward(cfg, decode_kernels)
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

        self.fwd = jax.jit(fwd, static_argnames=("Tc", "contig", "all_logits"), donate_argnums=(1, 2))
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

    def new_state(self):
        # The model adds one scratch slot for padding rows. With S <= 2 no call ever has a padding row (prefill and decode
        # use 1 or S rows), so the scratch slot is left out: at a long context that is one slot of K/V less on the chip.
        return self.model.new_state(self.c, self.S - 1 if self.S <= 2 else self.S, self.CTX)

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
            r = self.fwd(self.W, ck, cv, jnp.asarray(toks), jnp.asarray(p0), jnp.asarray(lens), jnp.asarray(sl), Tc=Tc)
            ck, cv = r["ck"], r["cv"]
            self._note("prefill", B, T, Tc)
            done = [b for b, w in enumerate(grp) if w[2] + int(lens[b]) >= len(w[1])]
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
        """-> slot per prompt, cached snapshot (or None) per prompt. Longest cached prefix of the prompt's own scope first,
        then empty or least recently used slots (lowest index first, so one user decodes on the smallest batch)."""
        n, S = len(prompts), self.S
        sc = list(scopes or [None] * n)
        slot, snap = [None] * n, [None] * n
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
        free = sorted((s_ for s_ in range(S) if s_ not in slot), key=lambda s_: (self.used[s_], s_))
        for i in range(n):
            if slot[i] is None:
                slot[i] = free.pop(0)
        return slot, snap

    def _snap(self, state, s, ids, scope=None):
        r, c = self.take(state[1], s)
        return dict(ids=tuple(int(t) for t in ids), r=np.asarray(r), c=np.asarray(c), scope=scope)

    def reset_cache(self):
        self.state, self.snaps = None, [[] for _ in range(self.S)]

    # ------------------------------------------------------------------------------------------------------- generate
    def generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, scopes=None):
        try:
            return self._generate(prompts, max_new, stop_ids, samp, on_step, marks, scopes)
        except BaseException:
            if self.prefix_cache:      # the slot state may be half written or donated: start clean
                self.reset_cache()
            raise

    def _generate(self, prompts, max_new=128, stop_ids=(), samp=None, on_step=None, marks=None, scopes=None):
        """prompts: up to S lists of token ids. samp[i]: None (greedy) or dict(temperature, top_k, top_p, min_p, seed,
        repetition_penalty, frequency_penalty, presence_penalty). on_step(ids, finish) streams after every device call.
        marks[i]: with the prefix cache, a second point to remember inside prompt i (the start of its last message).
        scopes[i]: with the prefix cache, the cache scope of prompt i (any hashable; None = the shared scope). A prompt
        resumes only from points of its own scope, so one API key's prompts are never served from, or reported as cached
        by, another key's.
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
        slot, snap = self._place(prompts, sc)
        if pc and self.state is None:
            self.state = self.new_state()
        state = self.state if pc else self.new_state()
        self.state = None            # donated below; put back at the end
        start = [len(sn["ids"]) if sn else 0 for sn in snap]
        new_snaps = {}
        if pc:
            for i in range(n):
                s_ = slot[i]
                keep = [x for x in self.snaps[s_] if snap[i] is not None and len(x["ids"]) <= start[i] and x.get("scope") == sc[i]
                        and x["ids"] == tuple(prompts[i][:len(x["ids"])])]
                new_snaps[s_] = keep
                if snap[i] is not None:
                    state = (state[0], self.put(state[1], s_, jnp.asarray(snap[i]["r"]), jnp.asarray(snap[i]["c"])))
            mk = [m if (m is not None and start[i] < m < len(prompts[i])) else None for i, m in enumerate(marks or [None] * n)]
            if any(m is not None for m in mk):
                idx = [i for i in range(n) if mk[i] is not None]
                state, _, _ = self.prefill(state, [prompts[i] for i in idx], [slot[i] for i in idx], [start[i] for i in idx],
                                           [mk[i] for i in idx])
                for i in idx:
                    new_snaps[slot[i]].append(self._snap(state, slot[i], prompts[i][:mk[i]], sc[i]))
                    start[i] = mk[i]
        state, nxt, last = self.prefill(state, prompts, slot, start)
        if pc:
            for i in range(n):
                new_snaps[slot[i]].append(self._snap(state, slot[i], prompts[i], sc[i]))
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
            on_step(ids, fin)
        calls = 0
        ck, cv = state
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
                on_step(ids, fin)
        if pc:
            self.state = (ck, cv)
            self.clock += 1
            for s_, keep in new_snaps.items():
                self.snaps[s_], self.used[s_] = keep[-4:], self.clock      # host copies, about 154 MB each for this model
        t2_ = time.perf_counter()
        cached = [len(sn["ids"]) if sn else 0 for sn in snap]
        self.stats["requests"] += n
        self.stats["prompt_tokens"] += sum(len(p) for p in prompts)
        self.stats["cached_tokens"] += sum(cached)
        return dict(ids=ids, finish=fin, cached=cached, slots=slot,
                    timing=dict(prefill_s=t1 - t0, decode_s=t2_ - t1, decode_calls=calls,
                                new_tokens=sum(len(x) for x in ids), sampled=sampled))

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
