"""How much context a model gets on this machine: the one place the engine decides it, for every model it runs (models.json
builds and GGUF files alike).

The engine keeps each slot's K/V for the whole context on the chip (runtime/kvcache.py), so memory per chip is

    weights + the slots' state for (slots, ctx) + working memory

The slot state is what the model's own new_state allocates (K/V grows with ctx; a recurrent state does not), so it is
measured from the model's shapes, never from a table. Working memory is the attention scratch of one prefill call plus
XLA's own buffers: WORK_BYTES, a measured figure (see below), plus a FLOOR of headroom.

plan() answers, for this chip:
  - the longest context ONE request can have (slots 1), capped at the model's trained context;
  - for each slot count 1, 2, 4, 8, the longest context per slot;
  - what to run when the person asks for nothing (`--ctx auto`, the default): the most slots (at most 8) whose context
    per slot is at least min(AGENT_CTX, trained context), at the longest context that fits them; one slot at the
    longest that fits when even one slot cannot reach AGENT_CTX.
`--ctx max` = the longest one request can have (one slot). `--ctx N` = N tokens, with as many slots (at most the
asked --slots) as fit N.

The rule is fixed (not tuned on any test set). AGENT_CTX = 32,768 is the documented agent context (room for a coding
agent's ~20k-token start, docs/agents-on-tpu.md)."""
import math

AGENT_CTX = 32768        # the context every slot gets before the engine adds slots
STEP = 1024              # a planned context is a multiple of this
SLOT_COUNTS = (8, 4, 2, 1)
# Working memory per chip on top of weights and slot state, by the prefill attention budget the engine runs with
# (Engine attn_bytes: 2e9 at a context up to 8,192, else 5e8, cli.py). Measured on a TPU v6e-1 for Qwen3.8-27B BQ8_R
# (peak minus the weights file minus the slot state, over the warm-up and passkey requests): 1.21-1.40 GB at 5e8
# (16k-64k, dense and blocked prefill) and 2.57 GB at 2e9 (4k); each figure here is at least 0.35 GB above that.
WORK_BYTES = {5e8: 1.75e9, 2e9: 3.5e9}
FLOOR = 0.5e9            # headroom never planned into


def attn_bytes(ctx):
    """the prefill attention budget the engine runs a context with (cli.py passes it to Engine)"""
    return 2e9 if ctx <= 8192 else 5e8


def work_bytes(ctx):
    return WORK_BYTES[attn_bytes(ctx)] + FLOOR


def parse(v):
    """--ctx: 'auto' (default), 'max', or a token count >= 1 -> 'auto' | 'max' | int"""
    s = str(v).strip().lower()
    if s in ("auto", "max"):
        return s
    try:
        n = int(s.replace("_", "").replace(",", ""))
    except ValueError:
        n = 0
        if s[-1:] == "k" and s[:-1].isdigit():
            n = int(s[:-1]) * 1024
    if n < 1:
        raise ValueError(f"--ctx takes a number of tokens (e.g. 32768 or 32k), auto or max; got {v!r}")
    return n


class Fit:
    """state(slots, ctx) -> bytes of slot state per chip; fixed = weights and other bytes per chip that do not depend on
    ctx or slots; limit = bytes per chip (JAX bytes_limit); trained = the model's trained context (0 = unknown)."""

    def __init__(self, state, fixed, limit, trained=0):
        self.state, self.fixed, self.limit, self.trained = state, int(fixed), int(limit), int(trained or 0)
        self._lin = {}

    def need(self, slots, ctx):
        return self.fixed + self.state(slots, ctx) + work_bytes(ctx)

    def _line(self, slots):
        """state(slots, ctx) = a + b * ctx (K/V is linear in ctx), from two points"""
        if slots not in self._lin:
            a1, a2 = self.state(slots, STEP), self.state(slots, 2 * STEP)
            b = (a2 - a1) / STEP
            self._lin[slots] = (a1 - b * STEP, b)
        return self._lin[slots]

    def longest(self, slots):
        """the longest context (a multiple of STEP, capped at the trained context) that fits `slots` slots, or 0"""
        a, b = self._line(slots)
        best = 0
        for wb in sorted(set(WORK_BYTES.values()), reverse=True):     # the larger budget first: it is for short contexts
            room = self.limit - self.fixed - a - wb - FLOOR
            if room <= 0 or b <= 0:
                continue
            n = int(room // b) // STEP * STEP
            n = min(n, self.trained) if self.trained else n
            n = n // STEP * STEP if n >= STEP else n
            while n >= STEP and self.need(slots, n) > self.limit:      # the budget a context runs with decides
                n -= STEP
            best = max(best, n if n >= STEP else 0)
        return best

    def table(self):
        return {s: self.longest(s) for s in sorted(SLOT_COUNTS)}

    def choose(self, ctx="auto", slots=None):
        """-> (ctx, slots, why). slots=None: the engine picks (at most 8). Raises ValueError when nothing fits."""
        want = min(AGENT_CTX, self.trained) if self.trained else AGENT_CTX
        top = 8 if slots is None else int(slots)
        counts = [s for s in SLOT_COUNTS if s <= top]
        if ctx == "max":
            n = self.longest(1)
            if n < STEP:
                raise ValueError("no context fits this chip next to the weights")
            return n, 1, "max: the longest context one request can have"
        if ctx == "auto":
            if slots is not None:
                n = self.longest(top)
                if n < STEP:
                    raise ValueError(f"--slots {top}: no context fits this chip next to the weights")
                return n, top, f"auto: the longest context that fits {top} slot{'s' if top > 1 else ''}"
            for s in counts:
                n = self.longest(s)
                if n >= want:
                    return n, s, f"auto: the most slots with at least {want} tokens each, at the longest context that fits"
            n = self.longest(1)
            if n < STEP:
                raise ValueError("no context fits this chip next to the weights")
            return n, 1, f"auto: one slot (no slot count fits {want} tokens each)"
        n = int(ctx)
        for s in counts:
            if self.need(s, n) <= self.limit:
                return n, s, f"--ctx {n}: the most slots (at most {top}) that fit it"
        raise ValueError(f"--ctx {n} does not fit this chip next to the weights; the longest that fits is "
                         f"{self.longest(1)} tokens for one request (--ctx max)")


def line(fit, ctx, slots, why, model=""):
    """the start-up line a person reads: what this chip holds and what runs"""
    t = fit.table()
    per = ", ".join(f"{s} x {t[s]:,}" for s in sorted(t) if t[s])
    tr = f"; trained context {fit.trained:,}" if fit.trained else ""
    return (f"mashbox: context{(' for ' + model) if model else ''}: this chip holds up to {t.get(1, 0):,} tokens for one "
            f"request (slots x tokens each: {per}){tr}. Running {slots} slot{'s' if slots > 1 else ''} x {ctx:,} tokens "
            f"({why}). --ctx N|max|auto and --slots N set it.")


def event(fit, ctx, slots, why):
    return dict(event="context", ctx=ctx, slots=slots, why=why, trained=fit.trained, limit_bytes=fit.limit,
                fixed_bytes=fit.fixed, per_slots={str(s): n for s, n in fit.table().items()},
                work_bytes=int(work_bytes(ctx)), need_bytes=int(fit.need(slots, ctx)))


def int8_take_bytes(new_state, slots, ctx):
    """bytes of the device temporary the engine's K/V block copy (Engine.kv_take, the host tier's program) holds with the
    8-bit K/V store: the whole largest 8-bit K/V leaf, every slot (measured on a TPU v6e-1: temp_size of
    kv_take = 16 x S x CTX x 1152 bytes at (1, 16384), (1, 32768) and (2, 16384), the codes-and-scales leaf; the bf16
    store's copy holds no temporary). 0 with the bf16 store."""
    from . import kvcache as KVC
    if KVC.STORE != "int8":
        return 0
    import jax
    import numpy as np
    st = jax.eval_shape(lambda: new_state(slots - 1 if slots <= 2 else slots, ctx))
    return max([math.prod(x.shape) for x in jax.tree.leaves(st) if np.dtype(x.dtype) == np.int8 and len(x.shape) == 4]
               or [0])


def eval_state(new_state, slots, ctx):
    """bytes of the engine's slot state for (slots, ctx), from shapes only (jax.eval_shape: nothing allocated). The engine
    adds one scratch slot above 2 slots (Engine.new_state)."""
    import jax
    import numpy as np
    st = jax.eval_shape(lambda: new_state(slots - 1 if slots <= 2 else slots, ctx))
    return sum(math.prod(x.shape) * np.dtype(x.dtype).itemsize for x in jax.tree.leaves(st))
