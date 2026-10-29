"""GGUF in: run a llama.cpp GGUF file with the mashbox engine.

GGUF is llama.cpp's file format (github.com/ggml-org/llama.cpp); files are read with llama.cpp's own `gguf` package.
For CPU, GPU, Mac or phone, use llama.cpp. This package reads the file's architecture, tokenizer, chat template,
sampling defaults and every tensor's block type, and hands the blocks to mashbox.ops untouched (never re-quantized).

  open_model(path, act)  -> (File, arch module, Config)     with the clear refusals for what does not run here
  plan(f, cfg, ...)      -> one line: what runs, on what path, how much memory
  check_fits(f, cfg, ...) -> a clear refusal when the weights and state do not fit this chip, naming chips that fit
  fit_slots(f, cfg, ...)  -> the most slots (8, 4, 2, 1) whose estimate fits this chip: `run` uses it when --slots is the default
  inspect(path)          -> what the file holds and how it would run here, without loading it
"""
from .reader import File, GGUFError, llama_cpp_command, reader_version  # noqa: F401
from . import archs

# device memory per chip: v5e and v6e = JAX memory_stats()["bytes_limit"] MEASURED on our chips (16,909,334,528 and
# 33,546,041,344); v5p = the 95 GB HBM of Google's spec (not measured here)
CHIPS = (("TPU v5e (v5litepod-1)", 16.909e9), ("TPU v6e (v6e-1)", 33.546e9), ("TPU v5p (one chip)", 95e9))
HEADROOM = 0.92        # share of the limit the estimate may use: XLA's own scratch (activations, logits) comes on top


def open_model(path, act=None, repo=None, quant=None):
    """-> (File, arch module, Config). Raises GGUFError with the person-facing message for anything that cannot run."""
    f = File(path)
    arch = f.arch
    A = archs.module(arch, path, repo, quant)
    bad = f.unreadable()
    if bad:
        names = sorted({t.type_name for t in bad})
        raise GGUFError(f"mashbox: {len(bad)} tensors use block types the GGUF reader in this version (gguf {reader_version()}) "
                        f"cannot read: {', '.join(names)}. It runs in llama.cpp: {llama_cpp_command(path, repo, quant)}")
    kw = {} if act is None else dict(act=act)
    c = A.Config(f, **kw)
    return f, A, c


def plan(f, c, kernel_types=None, chip=None, mem=None):
    """One plan line: architecture, layers, block types with byte shares, which run on kernels and which on the exact
    reference (slow), the MTP layer, and the bytes the weights take."""
    from .. import ops
    kernel_types = ops.kernel_types() if kernel_types is None else kernel_types
    census, total = f.census()
    parts, ref, ref_types = [], 0, []
    for name, e in census.items():
        share = 100.0 * e["bytes"] / max(1, total)
        on_k = e["type"] in kernel_types
        parts.append(f"{name} {share:.1f}%")
        if not on_k and name not in ("F32",):
            ref += e["bytes"]
            ref_types.append(name)
    line = dict(arch=f.arch, layers=c.L, mtp_layers=getattr(c, "nextn", 0), weights_gb=round(total / 1e9, 2),
                types=", ".join(parts), kernel_types=sorted(ops.type_name(t) for t in kernel_types),
                reference_share=round(100.0 * ref / max(1, total), 1), chip=chip)
    warn = None
    if ref_types:
        warn = (f"mashbox: {', '.join(ref_types)} ({line['reference_share']}% of the weights) have no TPU kernel in this install "
                f"and run on the exact float32 reference path, which is slow (it rebuilds the weights at every step)")
    return line, warn


def need_bytes(f, c, ctx, slots, kernel_types=frozenset(), roles=None):
    """-> (weights, state, scratch) device bytes, an estimate (XLA's own scratch comes on top):
      weights  the blocks `load` puts on the device, as stored in the file; a multiplied weight whose type runs on the
               kernel package counts the package's planes when it reports them (ops.device_nbytes; the MTP layer is not
               loaded)
      state    K/V and linear-attention state for slots + 1 rows of ctx tokens
      scratch  the float32 values the exact reference path rebuilds at a time: the largest chunk of rows
               (ops.exact_chunk_rows) of a multiplied weight on that path (0 when every multiplied weight has a kernel in
               kernel_types)"""
    import numpy as np
    from . import archs
    loaded, mult = roles if roles is not None else archs.module(f.arch).tensor_roles(f, c)
    from .. import ops
    w = sum(ops.device_nbytes(f.tensors[n].type, f.tensors[n].shape, f.tensors[n].nbytes,
                              kernel_types if n in mult else frozenset()) for n in loaded)
    act = np.dtype(c.act).itemsize
    kv = 2 * c.NG * (slots + 1) * ctx * c.KV * c.HD * act
    rec = c.NG * (c.FI - 1) * (slots + 1) * (c.HV * c.DK * c.DV * 4 + (c.CK - 1) * c.CONV * act)
    sh = [tuple(int(v) for v in f.tensors[n].shape) for n in mult if f.tensors[n].type not in kernel_types]
    big = max([ops.exact_chunk_rows(s[-2], s[-1]) * s[-1] * 4 for s in sh if len(s) >= 2], default=0)
    return w, kv + rec, big


def _limit():
    """this chip's device memory in bytes (JAX memory_stats bytes_limit), or None when JAX does not report one (CPU)"""
    try:
        import jax
        return (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
    except Exception:
        return None


def slot_need(f, c, ctx, slots, kernel_types=frozenset()):
    """-> (weights + float32 scratch + the slot state, the same + the scratch that moving the state takes: the state and one
    more row, since jit_swap asked for 5.00 GiB = 10 rows at 8 slots = 9 rows of Qwen3-32B Q4_K_M on one v6e chip)"""
    w, st, big = need_bytes(f, c, ctx, slots, kernel_types)
    return w + big + st, w + big + st + st // (slots + 1) * (slots + 2)


def fit_slots(f, c, ctx, slots, kernel_types=frozenset(), lim=None):
    """-> the slot count `run` uses when --slots is left at its default: slots itself when slot_need's first value is inside
    HEADROOM of this chip and its second inside the limit (or the chip reports no limit); else the most of slots/2, ... 1
    that fits; else 1 when both are inside the limit; else slots, so check_fits refuses. llama.cpp fits its defaults to
    the device the same way. Why: on one v6e chip, Qwen3.8-27B Q8_0 was estimated at 32.8 GB with 8 slots (over
    HEADROOM, under the limit), warned, then ran out of memory at load; it ran with 4."""
    lim = _limit() if lim is None else lim
    if not lim:
        return slots
    def fits(s, room):
        one, two = slot_need(f, c, ctx, s, kernel_types)
        return one <= room and two <= lim
    s = slots
    while s >= 1:
        if fits(s, HEADROOM * lim):
            return s
        s //= 2
    return 1 if fits(1, lim) else slots


def check_fits(f, c, ctx, slots, kernel_types=frozenset(), warn=None):
    """Refuses when the estimate is over this chip's memory; warns when it is over HEADROOM of it (it may still load)."""
    lim = _limit()
    if not lim:
        return None
    w, st, big = need_bytes(f, c, ctx, slots, kernel_types)
    need = w + st + big
    if need <= lim and need > HEADROOM * lim and warn:
        warn(f"mashbox: this file needs about {need / 1e9:.1f} GB of this chip's {lim / 1e9:.1f} GB (estimate), close to "
             f"the limit; if it does not load, use fewer --slots or a shorter --ctx")
    if need > lim:
        fit = [n for n, b in CHIPS if b * HEADROOM >= need]
        raise GGUFError(f"mashbox: this file needs about {need / 1e9:.1f} GB on the chip (weights {w / 1e9:.1f} GB, state for "
                        f"{slots} x {ctx} tokens {st / 1e9:.2f} GB, largest float32 weight {big / 1e9:.2f} GB) and this chip "
                        f"has {lim / 1e9:.1f} GB. " + (f"It fits: {', '.join(fit)}. " if fit else "") +
                        "A smaller quant of the same model, a shorter --ctx or fewer --slots also helps.")
    return need


def inspect(path, repo=None, quant=None):
    f = File(path)
    census, total = f.census()
    out = dict(file=path, arch=f.arch, name=f.get("general.name"), weights_gb=round(total / 1e9, 3),
               tensors=len(f.tensors), types={k: dict(tensors=v["tensors"], gb=round(v["bytes"] / 1e9, 3),
                                                       share=round(100.0 * v["bytes"] / max(1, total), 2))
                                              for k, v in census.items()},
               tokenizer=dict(model=f.get("tokenizer.ggml.model"), pre=f.get("tokenizer.ggml.pre"),
                              vocab=len(f.get("tokenizer.ggml.tokens", []) or []),
                              chat_template=bool(f.get("tokenizer.chat_template"))),
               gguf_reader=reader_version(), llama_cpp=llama_cpp_command(path, repo, quant))
    try:
        _, A, c = open_model(path, repo=repo, quant=quant)
        held = archs.held(f.arch, path, repo, quant)                # as `run`: an unlisted MoE architecture does not run here
        if held:
            raise GGUFError(held)
        from .. import ops
        ops.load_kernels()                          # which types this install runs on a kernel (none without the package)
        line, warn = plan(f, c)
        est = {}
        for which, kt in (("this_install", ops.kernel_types()), ("reference_path", frozenset())):
            w, st, big = need_bytes(f, c, 2048, 8, kt)
            est[which] = dict(gb=round((w + st + big) / 1e9, 2), weights_gb=round(w / 1e9, 2), state_gb=round(st / 1e9, 2),
                             f32_scratch_gb=round(big / 1e9, 2),
                             chips_that_fit=[n for n, b in CHIPS if b * HEADROOM >= w + st + big])
        out.update(runs_here=True, plan=line, warning=warn, mtp_layers=c.nextn, chip_estimate_ctx2048_slots8=est,
                   chip_gb_estimate=est["this_install"]["gb"], chips_that_fit=est["this_install"]["chips_that_fit"])
    except GGUFError as e:
        out.update(runs_here=False, why=str(e))
    return out
