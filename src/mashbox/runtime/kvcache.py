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
import jax.numpy as jnp
from jax import lax
from jax.experimental.layout import Layout, with_layout_constraint

# 8-bit K/V (opt-in, MASHBOX_KV_CACHE=int8 for `mashbox run`): each position's K row holds the KV x HD K values as int8
# codes, then one float32 scale per KV head for K and then for V (absmax / 127 of that head's HD values, 4 bytes each), in
# QPAD extra bytes at the end of the K row (a TPU pads the last axis to 128 lanes, so the scales cost one 128-byte tile per
# position); the V row holds V's codes only. Per position: 1,152 + 1,024 bytes for KV x HD = 1,024 instead of 2 x 2,048 in
# bfloat16. The cache stays one array per K and V with the same (groups, slots, positions, row) axes, so every slice,
# swap and host copy of the engine is unchanged. Attention reads the codes and scales of one key block at a time and
# applies the scales after the dot (runtime/attention.py), so the window is never dequantized whole.
QPAD = 128
STORE = "bf16"           # "bf16" (default: the model's activation dtype) or "int8"; set once per process (set_store)


def set_store(name):
    global STORE
    if name not in ("bf16", "int8"):
        raise ValueError(f"{name}: bf16 or int8")
    STORE = name


def quantized(C):
    return C.dtype == jnp.int8


def rows_kv(kv, hd):
    """(K row, V row) widths in the cache's store"""
    if STORE == "int8":
        if 8 * kv > QPAD:
            raise ValueError(f"MASHBOX_KV_CACHE=int8 holds at most {QPAD // 8} KV heads per row ({kv})")
        return kv * hd + QPAD, kv * hd
    return kv * hd, kv * hd


def shape(groups, rows, ctx, kv, hd):
    """the K (and V) array's shape for `groups` K/V groups, `rows` slots (scratch included), ctx positions (bf16 store)"""
    return (groups, rows, ctx, kv * hd)


def zeros(groups, rows, ctx, kv, hd, dtype):
    fk, fv = rows_kv(kv, hd)
    if STORE == "int8" and ctx % TILE:
        raise ValueError(f"MASHBOX_KV_CACHE=int8 needs a context that is a multiple of {TILE} ({ctx})")
    dt = jnp.int8 if STORE == "int8" else dtype
    return jnp.zeros((groups, rows, ctx, fk), dt), jnp.zeros((groups, rows, ctx, fv), dt)


def _q(x, kv, hd):
    x = x.reshape(x.shape[:-2] + (kv, hd)).astype(jnp.float32)
    s = jnp.max(jnp.abs(x), -1) / 127.0
    c = jnp.clip(jnp.round(x / jnp.where(s > 0, s, 1.0)[..., None]), -127, 127).astype(jnp.int8)
    return c.reshape(s.shape[:-1] + (kv * hd,)), lax.bitcast_convert_type(s, jnp.int8).reshape(s.shape[:-1] + (4 * kv,))


def pack(k, v, kv, hd):
    """(..., kv, hd) K and V values -> K rows (..., kv x hd + QPAD) and V rows (..., kv x hd), int8: codes round(x / s)
    with s = absmax / 127 per head; the K row ends with K's kv float32 scales, then V's, as bytes, then zeros"""
    ck, sk = _q(k, kv, hd)
    cv, sv = _q(v, kv, hd)
    pad = jnp.zeros(sk.shape[:-1] + (QPAD - 8 * kv,), jnp.int8)
    return jnp.concatenate([ck, sk, sv, pad], -1), cv


def unpack(kr, vr, kv, hd):
    """pack's rows -> (K codes (..., kv, hd) int8, K scales (..., kv) float32), (V codes, V scales)"""
    n = kv * hd
    sc = lambda b: lax.bitcast_convert_type(b.reshape(b.shape[:-1] + (kv, 4)), jnp.float32)
    ck = kr[..., :n].reshape(kr.shape[:-1] + (kv, hd))
    cv = vr[..., :n].reshape(vr.shape[:-1] + (kv, hd))
    return (ck, sc(kr[..., n:n + 4 * kv])), (cv, sc(kr[..., n + 4 * kv:n + 8 * kv]))


def dequant(c, s, dtype):
    return (c.astype(jnp.float32) * s[..., None]).astype(dtype)


def _rm(x):
    return with_layout_constraint(x, Layout(major_to_minor=tuple(range(x.ndim))))


def read(Kc, Vc, g, slots, Tc, contig, width=None, start=0, mult=1):
    """Group g's cached K/V of this call's rows, positions start .. start + Tc - 1 -> (B, Tc, width) each, in the cache's
    own layout. contig: the rows are slots 0..B-1 (one slice); else each row's slot is gathered. width: the front of each
    row that this layer uses (gemma4's narrower layers), default the whole row. g and start may be traced (a layer scan,
    a key block of runtime/attention.attend_cache)."""
    B = slots.shape[0]
    w = lambda C: C.shape[-1] if width is None else width       # an int8 cache's K row is wider than its V row
    if quantized(Kc):
        return _read_tiles(Kc, g, slots, Tc, contig, start, mult), _read_tiles(Vc, g, slots, Tc, contig, start, mult)
    if contig:
        cut = lambda C: lax.dynamic_slice(C, (g, 0, start, 0), (1, B, Tc, w(C)))[0]
    else:
        cut = lambda C: jnp.take(lax.dynamic_slice(C, (g, 0, start, 0), (1, C.shape[1], Tc, w(C)))[0], slots, axis=0)
    return _rm(cut(Kc)), _rm(cut(Vc))


TILE = 32     # an int8 array's TPU tile is 32 positions x 128 bytes


def _tiles(C):
    """an int8 cache seen as (groups, slots, positions / TILE, TILE, row): the same bytes (a reshape the TPU does for
    free), so a read or write that covers whole tiles is a plain slice of it. A one-position dynamic_update_slice of
    the 4-axis int8 cache made XLA move the whole cache into an 8-position tiling and back at every decode step, and
    the block reads did the same inside the attention loop (HLO of Engine.dec on a v6e-1)"""
    G, S, CTX, F = C.shape
    if CTX % TILE:
        raise ValueError(f"MASHBOX_KV_CACHE=int8 needs a context that is a multiple of {TILE} ({CTX})")
    return _rm(C.reshape(G, S, CTX // TILE, TILE, F))


def _read_tiles(C, g, slots, n, contig, start, mult=1):
    """read() of an int8 cache. mult: a static number every start this call site can pass is a multiple of
    (runtime/attention.attend_cache: gcd of its window and block). With mult and n multiples of TILE the read is whole
    tiles; otherwise it reads the tiles that cover [start, start + n) and cuts the window out of that block."""
    B, F, nT = slots.shape[0], C.shape[-1], C.shape[2] // TILE
    T5 = _tiles(C)
    rows = B if contig else C.shape[1]
    t0 = jnp.asarray(start, jnp.int32) // TILE
    whole = n % TILE == 0 and mult % TILE == 0
    nt = n // TILE if whole else min(nT, -(-n // TILE) + 1)
    if not whole:
        t0 = jnp.minimum(t0, nT - nt)
    x = lax.dynamic_slice(T5, (g, 0, t0, 0, 0), (1, rows, nt, TILE, F))[0]
    if not contig:
        x = jnp.take(x, slots, axis=0)
    x = x.reshape(B, nt * TILE, F)
    if not whole:
        x = lax.dynamic_slice_in_dim(x, jnp.asarray(start, jnp.int32) - t0 * TILE, n, 1)
    return x


def reader(Kc, Vc, g, slots, contig, kv, hd, width=None):
    """-> read(start, n): (B, n, kv, hd) K and V of group g, positions start .. start + n - 1 (runtime/attention.attend_cache
    reads the cache through it, block by block, with no copy of the whole window). An int8 cache gives (codes (B, n, kv,
    hd) int8, scales (B, n, kv) float32) for each of K and V instead."""
    if quantized(Kc):
        if width is not None and width != kv * hd:
            raise NotImplementedError("MASHBOX_KV_CACHE=int8 reads whole rows")

        def rdq(start, n, mult=1):
            k, v = read(Kc, Vc, g, slots, n, contig, None, start, mult)
            return unpack(k, v, kv, hd)
        return rdq

    def rd(start, n, mult=1):
        k, v = read(Kc, Vc, g, slots, n, contig, kv * hd if width is None else width, start)
        B = slots.shape[0]
        return k.reshape(B, n, kv, hd), v.reshape(B, n, kv, hd)
    return rd


def read_heads(Kc, Vc, g, slots, Tc, contig, kv, hd):
    """read() shaped (B, Tc, kv, hd) for the attention dot (an int8 cache: dequantized, bfloat16)"""
    if quantized(Kc):
        (ck, sk), (cv, sv) = unpack(*read(Kc, Vc, g, slots, Tc, contig), kv, hd)
        return dequant(ck, sk, jnp.bfloat16), dequant(cv, sv, jnp.bfloat16)
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
    if quantized(Kc):                                   # (n, B, T, KV, HD) -> packed int8 rows of the full width
        if ks.ndim != 5 or ks.shape[3] * ks.shape[4] + QPAD != Kc.shape[-1]:
            raise NotImplementedError("MASHBOX_KV_CACHE=int8 writes whole rows of (KV, HD) heads")
        ks, vs = pack(ks, vs, *ks.shape[3:])
    ks = ks.reshape(n, B, T, -1).astype(Kc.dtype)
    vs = vs.reshape(n, B, T, -1).astype(Vc.dtype)
    p0 = pos[:, 0].astype(jnp.int32)
    st = jnp.clip(p0, 0, CTX - T)
    d = p0 - st                                         # > 0 only for a row whose positions pass the cache end
    j = jnp.arange(T, dtype=jnp.int32)

    def put(C, new, b):
        at = (jnp.asarray(g, jnp.int32), slots[b].astype(jnp.int32), st[b], jnp.int32(0))
        old = lax.dynamic_slice(C, at, (n, 1, T, new.shape[-1]))
        x = new[:, b:b + 1]
        if T > 1:                                       # token j - d goes to window position j
            x = jnp.take(x, jnp.clip(j - d[b], 0, T - 1), axis=2)
        x = jnp.where((j >= d[b])[None, None, :, None], x, old)
        return lax.dynamic_update_slice(C, x, at)

    if quantized(Kc):
        Kc, Vc = _write_tiles(Kc, ks, slots, st, d, g), _write_tiles(Vc, vs, slots, st, d, g)
        return Kc, Vc

    # the cache stays in its own (row-major) layout through the writes: a one-position update has size-1 axes, and
    # without the constraint XLA gave the dynamic_update_slice (and so the whole cache) a layout that tiles the group or
    # slot axis instead, copying the whole cache into it and back
    Kc, Vc = _rm(Kc), _rm(Vc)
    for b in range(B):
        Kc, Vc = put(Kc, ks, b), put(Vc, vs, b)
    return _rm(Kc), _rm(Vc)


def _write_tiles(C, new, slots, st, d, g):
    """write()'s per-row update on an int8 cache, on whole TILE-position tiles: the window [st, st + T) widened to the
    tiles that cover it (Tw positions from a0, a0 a multiple of TILE); positions outside [st + d, st + T) keep their
    values. Same result as write() on a bfloat16 cache."""
    n, B, T, F = new.shape
    G, S, CTX, _ = C.shape
    Tw = min(CTX, -(-(T + TILE - 1) // TILE) * TILE)
    a0 = jnp.minimum(st // TILE * TILE, CTX - Tw)
    off = st - a0
    u = jnp.arange(Tw, dtype=jnp.int32)
    T5 = _tiles(C)
    for b in range(B):
        at = (jnp.asarray(g, jnp.int32), slots[b].astype(jnp.int32), a0[b] // TILE, jnp.int32(0), jnp.int32(0))
        old = lax.dynamic_slice(T5, at, (n, 1, Tw // TILE, TILE, F)).reshape(n, 1, Tw, F)
        r = u - off[b]
        x = jnp.take(new[:, b:b + 1], jnp.clip(r - d[b], 0, T - 1), axis=2)
        x = jnp.where(((r >= d[b]) & (r < T))[None, None, :, None], x, old)
        T5 = _rm(lax.dynamic_update_slice(T5, x.reshape(n, 1, Tw // TILE, TILE, F), at))
    return _rm(T5.reshape(G, S, CTX, F))


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
    if quantized(Ks):
        raise NotImplementedError("MASHBOX_KV_CACHE=int8 is not offered for sliding-window rings")
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
