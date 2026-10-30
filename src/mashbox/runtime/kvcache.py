"""The K/V cache of the full-attention layers, for every model the engine runs (GGUF and Hugging Face files through
gguf/archs, the Qwen3.8-27B build on one chip and under tensor parallel): its layout, and the only code that reads or
writes it.

Layout: one array for K and one for V, each (groups, slots + 1, positions, KV heads x head size), in the model's
activation dtype. The heads are flat in the last axis, so the two axes a TPU tiles are positions x (KV x HD): with
bfloat16, positions fill the sublanes and KV x HD the lanes, with no padding for any head count or head size that makes a
multiple of 128 (every catalog file). The old layout (..., positions, KV, HD) tiled (KV, HD): one K/V head (gemma4 keeps
each token's K/V flat in one head) padded 1 -> 2, and a 64-wide head padded 64 -> 128 lanes.

Read: the rows of one group, positions < Tc, held in the cache's own layout (with_layout_constraint) and then shaped
(B, Tc, KV, HD). Without the constraint XLA on TPU gives the attention dot's preferred layout to the whole cache, so it
copied the whole cache at every decode step.

Write: one dynamic_update_slice per row, at (group, slot, position) with the position clamped into the cache; a position
past the cache end writes back the values already there, so nothing past the cache is written (the old code's dropping
scatter) and a write never sits under a lax.cond. XLA then updates the donated cache in place through the decode loop:
the cond and the dropping scatter each gave the while loop a layout of their own, and XLA copied the whole cache into it
before the loop, back after it, and inside it at every step (tools/kv_hlo_check.py reads this from a TPU compile).

Ring (sliding-window layers, archs/gemma4.py): the same layout with `size` = min(window, context) positions per slot.
Position p lives at entry p mod size, so a slot holds its last `size` positions. ring_pos() says which position each
entry holds before a call (the call's attention masks by that real position); ring_write() writes a call's positions
there in place, at most two dynamic_update_slices per row (the run up to the ring's end, then the part that wraps to
its start). A ring cannot be rolled back to an earlier position: the engine keeps it with the linear-attention state
(snapshots at the prompt points, put back on a resume), and a token is written only when it is real: inside the row's
length, before the context end (decode rows that are not live sit at the context end) and in a row the call may
change (keep)."""
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.layout import Layout, with_layout_constraint


def shape(groups, rows, ctx, kv, hd):
    """the K (and V) array's shape for `groups` K/V groups, `rows` slots (scratch included), ctx positions"""
    return (groups, rows, ctx, kv * hd)


def zeros(groups, rows, ctx, kv, hd, dtype):
    s = shape(groups, rows, ctx, kv, hd)
    return jnp.zeros(s, dtype), jnp.zeros(s, dtype)


def _rm(x):
    return with_layout_constraint(x, Layout(major_to_minor=tuple(range(x.ndim))))


def read(Kc, Vc, g, slots, Tc, contig, width=None):
    """Group g's cached K/V of this call's rows, positions < Tc -> (B, Tc, width) each, in the cache's own layout.
    contig: the rows are slots 0..B-1 (one slice); else each row's slot is gathered. width: the front of each row that
    this layer uses (gemma4's narrower layers), default the whole row. g may be a traced index (a layer scan)."""
    B, F = slots.shape[0], Kc.shape[-1]
    n = F if width is None else width
    if contig:
        cut = lambda C: lax.dynamic_slice(C, (g, 0, 0, 0), (1, B, Tc, n))[0]
    else:
        cut = lambda C: jnp.take(lax.dynamic_slice(C, (g, 0, 0, 0), (1, C.shape[1], Tc, n))[0], slots, axis=0)
    return _rm(cut(Kc)), _rm(cut(Vc))


def read_heads(Kc, Vc, g, slots, Tc, contig, kv, hd):
    """read() shaped (B, Tc, kv, hd) for the attention dot"""
    k, v = read(Kc, Vc, g, slots, Tc, contig, kv * hd)
    B = slots.shape[0]
    return k.reshape(B, Tc, kv, hd), v.reshape(B, Tc, kv, hd)


def write(Kc, Vc, slots, pos, ks, vs, g=0):
    """Write this call's K/V into groups g .. g+n-1 at each row's (slot, positions); positions past the cache are dropped.
    ks, vs (n, B, T, ...) (the trailing axes flatten to at most the cache row; a narrower one fills its front); slots (B,);
    pos (B, T) = each row's first position + 0..T-1. g may be traced (the tensor-parallel layer loop writes its group).

    Per row: one dynamic_update_slice of a (n, 1, T, F) window starting at min(p0, CTX - T) (T at most CTX: a longer
    call keeps its first CTX tokens, the only ones whose positions can be in the cache). When the row's positions run
    past the cache end the window starts d = p0 - start earlier, so its first d positions take the values already there
    and the rest take the row's first T - d tokens; positions past the cache are never written. Duplicate slots (padding
    rows sent to the scratch slot) are written in row order."""
    n, B, T = ks.shape[:3]
    CTX = Kc.shape[2]
    if T > CTX:                                         # a call longer than the cache (a warm-up chunk on a short
        T = CTX                                         # context): positions p0 + t < CTX need t < CTX, so only the
        ks, vs, pos = ks[:, :, :T], vs[:, :, :T], pos[:, :T]    # first CTX tokens of a row can land in the cache
    ks = ks.reshape(n, B, T, -1).astype(Kc.dtype)
    vs = vs.reshape(n, B, T, -1).astype(Vc.dtype)
    w = ks.shape[-1]
    p0 = pos[:, 0].astype(jnp.int32)
    st = jnp.clip(p0, 0, CTX - T)
    d = p0 - st                                         # > 0 only for a row whose positions pass the cache end
    j = jnp.arange(T, dtype=jnp.int32)

    def put(C, new, b):
        at = (jnp.asarray(g, jnp.int32), slots[b].astype(jnp.int32), st[b], jnp.int32(0))
        old = lax.dynamic_slice(C, at, (n, 1, T, w))
        x = new[:, b:b + 1]
        if T > 1:                                       # token j - d goes to window position j
            x = jnp.take(x, jnp.clip(j - d[b], 0, T - 1), axis=2)
        x = jnp.where((j >= d[b])[None, None, :, None], x, old)
        return lax.dynamic_update_slice(C, x, at)

    # the cache stays in its own (row-major) layout through the writes: a one-position update has size-1 axes, and
    # without the constraint XLA gave the dynamic_update_slice (and so the whole cache) a layout that tiles the group or
    # slot axis instead, copying the whole cache into it and back
    Kc, Vc = _rm(Kc), _rm(Vc)
    for b in range(B):
        Kc, Vc = put(Kc, ks, b), put(Vc, vs, b)
    return _rm(Kc), _rm(Vc)


def ring_pos(p0, n, size):
    """The position held by each of the first n entries of a ring of `size` entries, before a call whose rows start at
    p0 (B,) and that wrote every position before p0 -> (B, n) int32; < 0: the entry is empty."""
    q = p0.astype(jnp.int32)[:, None] - 1
    return q - jnp.mod(q - jnp.arange(n, dtype=jnp.int32)[None], size)


def ring_write(Ks, Vs, slots, pos, lens, ks, vs, ctx, keep=None, g=0):
    """Write this call's K/V into groups g .. g+n-1 of a ring cache at entry position mod size. ks, vs (n, B, T, ...),
    slots (B,), pos (B, T) as write(); lens (B,) the real tokens of each row; ctx the context length (positions from ctx
    on are not written, as write() drops them); keep (B,) bool or None: rows this call may change.

    Per row only its first c = min(lens, ctx - p0) tokens are real (0 when keep is False). With more than `size` of them
    only the last `size` matter: the window moves to start at c - size. Then the run that fits before the ring's end goes
    in one dynamic_update_slice at entry a = start mod size (clamped to size - T, as write() clamps), and the rest wraps
    to entry 0 in a second one; every entry of either window that takes no real token keeps the value already there."""
    n, B, T = ks.shape[:3]
    size = Ks.shape[2]
    ks = ks.reshape(n, B, T, -1).astype(Ks.dtype)
    vs = vs.reshape(n, B, T, -1).astype(Vs.dtype)
    w = ks.shape[-1]
    i32 = jnp.int32
    p0 = pos[:, 0].astype(i32)
    c = jnp.clip(jnp.minimum(lens.astype(i32), ctx - p0), 0, T)
    if keep is not None:
        c = jnp.where(keep, c, 0)
    Tw = min(T, size)
    off = jnp.clip(c - size, 0, T - size) if T > size else jnp.zeros((B,), i32)
    c = c - off
    a = jnp.mod(p0 + off, size)
    st = jnp.minimum(a, size - Tw)
    j = jnp.arange(Tw, dtype=i32)

    def put(C, new, b):
        x = new[:, b:b + 1]
        if T > size:
            x = lax.dynamic_slice_in_dim(x, off[b], size, axis=2)
        gi, si = jnp.asarray(g, i32), slots[b].astype(i32)
        at = (gi, si, st[b], i32(0))
        i1 = j - (a[b] - st[b])                         # token index for each entry of the first window
        y = jnp.take(x, jnp.clip(i1, 0, Tw - 1), axis=2) if Tw > 1 else x
        C = lax.dynamic_update_slice(C, jnp.where(((i1 >= 0) & (i1 < c[b]))[None, None, :, None], y,
                                                  lax.dynamic_slice(C, at, (n, 1, Tw, w))), at)
        if Tw > 1:                                      # the tokens past the ring's end, from entry 0
            at = (gi, si, i32(0), i32(0))
            i2 = j + size - a[b]
            y = jnp.take(x, jnp.clip(i2, 0, Tw - 1), axis=2)
            C = lax.dynamic_update_slice(C, jnp.where((i2 < c[b])[None, None, :, None], y,
                                                      lax.dynamic_slice(C, at, (n, 1, Tw, w))), at)
        return C

    Ks, Vs = _rm(Ks), _rm(Vs)
    for b in range(B):
        Ks, Vs = put(Ks, ks, b), put(Vs, vs, b)
    return _rm(Ks), _rm(Vs)
