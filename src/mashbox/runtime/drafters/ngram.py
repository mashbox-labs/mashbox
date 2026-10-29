"""The n-gram drafter (prompt lookup) for any model whose architecture has a verify pass: no weights, no extra programs
at prefill. It runs on the device inside the step, so one call keeps its N rounds.

State: hist (S, CTX) i32, each slot's own tokens: its prompt and the tokens it committed, at their positions (-1 where
none). It never holds another row's or another tenant's tokens. The host writes a row in full (one fixed-shape put)
before the row's first speculation call and after any plain decode call; inside the step after_commit writes only
committed tokens, so rejected drafts leave no trace.

Match (POLICY, from the start values; changed only from the dev replay, with the replay file's sha256 recorded): for n
from N_MAX down to N_MIN, the last n tokens of the row (ending at the token being fed, at pos) against every earlier
window; the most recent window whose continuation starts at or before pos wins; up to K tokens of that continuation that
lie at or before pos are the drafts (found = how many; 0 without a match). match() is the host reference of the device's
draft(), and they agree token for token (tests/cpu/test_drafters.py); find() is the same rule with NumPy windows, which
the host's call-level plain choice runs for each live row before a call.
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import Drafter, key_base

i32 = jnp.int32
N_MAX, N_MIN = 5, 3
VERSION = 1


def match(seq, K, n_max=N_MAX, n_min=N_MIN):
    """Host reference. seq: the row's tokens 0..pos (the last one is the token being fed) -> (drafts list, length <= K)."""
    s = np.asarray(seq, np.int64)
    pos = len(s) - 1
    for n in range(n_max, n_min - 1, -1):
        if pos - n + 1 < 0:
            continue
        suf = s[pos - n + 1:]
        best = -1
        for i in range(pos - n, -1, -1):        # windows i..i+n-1 with continuation i+n <= pos, most recent first
            if np.array_equal(s[i:i + n], suf):
                best = i
                break
        if best >= 0:
            c = best + n
            return [int(x) for x in s[c:min(pos + 1, c + K)]]
    return []


def find(seq, K, n_max=N_MAX, n_min=N_MIN):
    """match() with NumPy windows (equal to it token for token): the host runs it for each live row before a call."""
    s = np.asarray(seq, np.int64)
    pos = len(s) - 1
    for n in range(n_max, n_min - 1, -1):
        if pos - n < 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(s[:pos], n)    # window i = s[i:i+n], its continuation i+n <= pos
        hit = np.flatnonzero((win == s[pos - n + 1:]).all(1))
        if hit.size:
            c = int(hit[-1]) + n
            return s[c:min(pos + 1, c + K)].tolist()
    return []


class NGram(Drafter):
    kind = "ngram"
    extra_groups = 0
    needs_hidden = False
    verify_kw = {}
    plain_ok = True
    backoff = True
    W = None

    def __init__(self, k_max, n_max=N_MAX, n_min=N_MIN):
        self.k_max, self.n_max, self.n_min = int(k_max), int(n_max), int(n_min)
        self.put = None

    def key(self):
        return key_base("ngram", f"ngram@{VERSION}", None, {"n_max": self.n_max, "n_min": self.n_min},
                        {"k_max": self.k_max})

    def state(self, eng):
        CTX = eng.CTX
        self.put = jax.jit(lambda h, s, row: h.at[s].set(row), donate_argnums=(0,))
        return eng._committed(jnp.full((eng.S, CTX), -1, i32))

    def carry_in(self, D, X, B):
        return D[:B]

    def carry_out(self, D, Dc, B):
        return D.at[:B].set(Dc)

    def draft(self, W, DW, ck, Dc, tok, pos, Tc, K, sp, st):
        hist = Dc
        B, C = hist.shape
        b_ = jnp.arange(B)
        full = hist.at[b_, pos].set(tok, mode="drop")     # the row's tokens 0..pos (padding rows: pos = CTX, dropped)
        idx = jnp.arange(C, dtype=i32)
        found, start = jnp.zeros((B,), bool), jnp.zeros((B,), i32)
        padv = jnp.full((B, self.n_max), -2, i32)
        ext = jnp.concatenate([full, padv], 1)
        for n in range(self.n_max, self.n_min - 1, -1):
            eq = jnp.ones((B, C), bool)
            for j in range(n):
                sj = jnp.take_along_axis(full, jnp.clip(pos - n + 1 + j, 0, C - 1)[:, None], 1)
                eq = eq & (ext[:, j:j + C] == sj)
            ok = eq & (idx[None] + n <= pos[:, None]) & (pos[:, None] - n + 1 >= 0)
            last = jnp.max(jnp.where(ok, idx[None], -1), 1)
            hit = (last >= 0) & ~found
            start = jnp.where(hit, last + n, start)
            found = found | hit
        avail = jnp.where(found, jnp.clip(pos + 1 - start, 0, K), 0).astype(i32)
        ci = jnp.clip(start[:, None] + jnp.arange(K, dtype=i32)[None], 0, C - 1)
        d = jnp.take_along_axis(full, ci, 1)
        d = jnp.where(jnp.arange(K)[None] < avail[:, None], jnp.maximum(d, 0), 0).astype(i32)
        return d, avail, ck

    def after_commit(self, Dc, r, emit, n, go, tok, pos, K1):
        B, C = Dc.shape
        seq = jnp.concatenate([tok[:, None], emit[:, :K1 - 1]], 1)        # the tokens at pos .. pos + K
        p = pos[:, None] + jnp.arange(K1, dtype=i32)[None]
        p = jnp.where(jnp.arange(K1)[None] < n[:, None], p, C)            # committed ones only; C = dropped
        return Dc.at[jnp.arange(B)[:, None], p].set(seq, mode="drop")

    # host side
    def host_args(self, eng, B, live, k):
        for s_ in live:
            r = eng.rows[s_]
            if not r.get("hist_ok"):
                seq = list(r["prompt"]) + list(r["ids"][:-1]) if r["ids"] else list(r["prompt"])
                row = np.full(eng.CTX, -1, np.int32)
                row[:len(seq)] = seq[:eng.CTX]
                eng.dstate = self.put(eng.dstate, np.int32(s_), row)
                r["hist_ok"] = True
        return ()

    def host_zero(self, eng, B):
        return ()

    def host_after(self, eng, r, seq, last, p2):
        pass

    def programs(self, eng):
        """the host put (one program), as warm() lists calls: (kind, function, args after the state, static kwargs)"""
        return [("spec_put", self.put, (np.int32(0), np.full(eng.CTX, -1, np.int32)), {})]

    def found(self, eng, r, K):
        """the call-level plain choice: drafts this row would get now (find() over its prompt and committed tokens)"""
        seq = np.asarray(r["prompt"], np.int64)
        if r["ids"]:
            seq = np.concatenate([seq, np.asarray(r["ids"], np.int64)])
        return len(find(seq, K, self.n_max, self.n_min))

    def on_prefill(self):
        return None
