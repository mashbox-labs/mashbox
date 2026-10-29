"""Weight ops on GGUF tensors: the one boundary between the public engine and the TPU kernel package.

A weight reaches the device as the file's own blocks, untouched: raw uint8 bytes, the ggml type id and the logical shape
(rows, cols). `prepare` turns that into a handle and `qmatmul` multiplies activations by it.

  prepare(raw, t, shape, name) -> handle      never dequantizes or re-quantizes for the fast path
  qmatmul(x, handle) -> x @ W.T                f32 accumulation, result in x's dtype
  take_rows(handle, ids) -> W[ids] as float32  embedding lookup (always the exact path)
  dequant(handle) -> W as float32              exact, for tests

Two implementations sit behind the same names:
  * the exact reference (public, plain JAX, mashbox/ops/reference.py): dequantizes the blocks to float32 with gguf-py's
    own math at every call and multiplies at HIGHEST precision. It runs every type gguf-py reads, on any chip and on CPU,
    and it is slow, so `plan` prints a warning naming the types and the share of bytes that use it.
  * the TPU kernel package (licensed binary, optional): a Python module that provides, per supported type,
      repack(t, raw, shape) -> packed      lossless byte permutation into the kernel's layout (a pytree of arrays)
      matmul(x, packed) -> y               the kernel
                                           optional keyword layer=: packed stacked on one leading axis (the layer
                                           groups of a lax.scan) and the layer's index, read in place by the kernel
      dequant(packed) -> float32           for the contract test: must equal the reference bit for bit
    and a `TYPES` collection of the ggml type ids it covers. Optional:
      repack_host(t, raw, shape) -> packed the same planes from a host NumPy array, on the host's cores (used for every
                                           host array instead of repack; leading axes = a stack of tensors)
      plane_bytes(t, shape) -> int         device bytes of one (rows, cols) tensor's planes (the memory estimate)
    and, optional, native block planes at the file's own size (one plane family for dense weights and stacked expert
    weights (experts, rows, cols); the package's mul_mat.nblock unit):
      nb_types() -> ggml type ids          the types with native planes
      nb_bytes(t, shape) -> int            device bytes of the planes of a (..., rows, cols) weight, every matrix counted
      repack_nb_host(raw, t, shape) -> planes   NumPy, once at load (raw (..., rows, row_bytes); shape = (rows, cols))
      matmul_nb(x, planes) -> y            dense planes (no stacked dims)
      nb_dequant(planes) -> float32        for the contract test: must equal the reference bit for bit
    The planes are a pytree whose `lead` attribute is the stacked dims read from its arrays, so a lax.scan or vmap
    slice of a stack (one expert or one layer group) is a dense set of planes inside the body.
    `load_kernels` imports it; nothing here depends on its layout.

Which path a weight takes (weight_path): stacked expert weights use the native planes when the package has them for
the type; a dense weight uses kq when the package's kq covers its type, else the exact path (DENSE_NATIVE = True moves
dense weights of a native type to the native planes; it is off by default). Every fast path falls back: if building the
native planes fails, a dense weight loads on kq (when kq covers the type) or the exact path, and a stacked expert weight
loads on the exact path (the file's own bytes: kq's planes are larger than the native planes the memory estimate
counted), with a warning.
On kq, the repack and the kernel's matmul must both work before a weight is used: the first weight of each (type, shape)
runs the kernel once, compiled, on one row (the preflight). If the repack or the preflight raises, at trace, lowering,
compile or run, weights of that type and shape load on the exact path, with a warning.
"""
import importlib
import inspect
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..runtime import faults
from . import reference as R
from . import dense16

f32 = jnp.float32
HI = lax.Precision.HIGHEST
KERNEL_MODULE = "mashbox_engine_kernels"     # the licensed kernel wheel's module (release item `engine`)
_KERNELS = {"module": None, "types": frozenset(), "nb": frozenset(), "name": None, "error": None}
_KQ_RAN = {}             # kq preflight per (type, shape) since load_kernels: True = the kernel ran, else why it failed
_KQ_LAYER = {}           # kq layer= preflight per (type, shape) on a layer stack: True = it ran, else why it failed
PREFLIGHT_DTYPE = jnp.bfloat16   # the preflight's input dtype: the forward's activation dtype (the archs' Config act)
DENSE_NATIVE = False     # dense weights stay on kq or the exact path; experts go native


def type_name(t):
    return R.TYPES[t][0] if t in R.TYPES else f"type {t}"


@jax.tree_util.register_pytree_node_class
class Raw:
    """Reference handle: the file's blocks as stored, (..., rows, row_bytes) uint8 or as one byte run per tensor
    (..., rows * row_bytes) (see _flat), plus type and logical shape."""

    def __init__(self, raw, t, shape):
        self.raw, self.t, self.shape = raw, int(t), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.raw,), (self.t, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], *aux)

    def __repr__(self):
        return f"Raw({type_name(self.t)}, {self.shape})"


@jax.tree_util.register_pytree_node_class
class NB:
    """Native-plane handle: the kernel package's repack_nb_host planes (opaque here), plus type and logical shape"""

    def __init__(self, planes, t, shape):
        self.planes, self.t, self.shape = planes, int(t), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.planes,), (self.t, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], *aux)

    def __repr__(self):
        return f"NB({type_name(self.t)}, {self.shape})"


@jax.tree_util.register_pytree_node_class
class Packed:
    """Kernel handle: whatever the kernel package's repack returned (opaque here), plus type and logical shape."""

    def __init__(self, packed, t, shape):
        self.packed, self.t, self.shape = packed, int(t), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.packed,), (self.t, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], *aux)

    def __repr__(self):
        return f"Packed({type_name(self.t)}, {self.shape})"


@jax.tree_util.register_pytree_node_class
class At:
    """One layer of a Packed handle stacked over layer groups, for a kernel that picks the layer itself (matmul's
    layer=): the whole stack and the layer index. A lax.scan body multiplies by it; the stack is never sliced (on TPU
    a slice of the planes is a copy: one more read and write of the layer's planes at every call)."""

    def __init__(self, packed, layer, t, shape):
        self.packed, self.layer, self.t, self.shape = packed, layer, int(t), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.packed, self.layer), (self.t, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], ch[1], *aux)

    def __repr__(self):
        return f"At({type_name(self.t)}, {self.shape})"


@jax.tree_util.register_pytree_node_class
class Mixed:
    """One weight role stacked over layers whose block types differ by layer (a Q4_K_M file keeps some layers' attn_v and
    ffn_down in Q6_K): one stacked handle per type, holding that type's layers in order, and per layer `which` stack and
    the layer's index `sub` in it ((layers,) int32 each). A lax.scan holds the stacks whole and scans which / sub
    (scan_split / scan_join); its body multiplies through a Pick, one lax.switch branch per type, so a program holds one
    weight rebuild per type of the role instead of one per layer (unrolled layers made each program ~30x larger)."""

    def __init__(self, stacks, which, sub, types, shape):
        self.stacks, self.which, self.sub = tuple(stacks), which, sub
        self.types, self.shape = tuple(int(t) for t in types), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.stacks, self.which, self.sub), (self.types, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], ch[1], ch[2], *aux)

    def __repr__(self):
        return f"Mixed({', '.join(type_name(t) for t in self.types)}, {self.shape})"


@jax.tree_util.register_pytree_node_class
class Pick:
    """One layer of a Mixed handle inside the scan body: the whole stacks, and the layer's stack (which) and index in it
    (sub) as traced scalars"""

    def __init__(self, stacks, which, sub, types, shape):
        self.stacks, self.which, self.sub = tuple(stacks), which, sub
        self.types, self.shape = tuple(int(t) for t in types), tuple(int(s) for s in shape)

    def tree_flatten(self):
        return (self.stacks, self.which, self.sub), (self.types, self.shape)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(ch[0], ch[1], ch[2], *aux)

    def __repr__(self):
        return f"Pick({', '.join(type_name(t) for t in self.types)}, {self.shape})"


def _pick_one(s, sub):
    """layer `sub` of one type stack of a Pick: At(stack, sub) when the kernel reads the layer in place (the layer=
    preflight passed for this type and shape), else the stack's leaves indexed at sub (what a lax.scan slice gives)"""
    if _held(s) and kernel_picks_layer():
        return At(s.packed, sub, s.t, s.shape)
    return jax.tree_util.tree_map(lambda a: lax.dynamic_index_in_dim(a, sub, 0, keepdims=False), s)


def _switch(h, fn, *args):
    """fn(handle of the picked layer, *args) through one lax.switch branch per type stack of Pick h"""
    if len(h.stacks) == 1:
        return fn(_pick_one(h.stacks[0], h.sub), *args)
    return lax.switch(h.which, [lambda *a, s=s: fn(_pick_one(s, h.sub), *a) for s in h.stacks], *args)


def load_kernels(log=None, module=None):
    """Import the TPU kernel package if it is installed (env MASHBOX_GGUF_KERNELS names another module, `off` disables).
    -> the ggml type ids it runs. Without it every type uses the exact reference."""
    module = module or os.environ.get("MASHBOX_GGUF_KERNELS", KERNEL_MODULE)
    _KERNELS.update(module=None, types=frozenset(), nb=frozenset(), name=None, error=None)
    _EXPERT_CHECKED.clear()
    _KQ_RAN.clear()
    _KQ_LAYER.clear()
    if module in ("", "off", "none"):
        return _KERNELS["types"]
    try:
        m = importlib.import_module(module)
        nb = frozenset(int(t) for t in m.nb_types()) if hasattr(m, "nb_types") else frozenset()
        _KERNELS.update(module=m, types=frozenset(int(t) for t in getattr(m, "TYPES", ())), nb=nb, name=module)
    except ImportError as e:
        _KERNELS["error"] = str(e)
    except Exception as e:                      # a broken install must not stop the exact path
        _KERNELS["error"] = f"{type(e).__name__}: {e}"
        if log:
            log(f"mashbox: kernel package {module} did not load ({_KERNELS['error']}); every type runs on the exact path")
    return _KERNELS["types"] | _KERNELS["nb"]


def kernel_types():
    """every ggml type id some kernel path runs (kq planes or native planes) for SOME weight shape. Membership does not
    mean that a given tensor of that type runs on a kernel: a native type that kq lacks runs native for stacked experts
    and on the exact path for a dense weight (unless DENSE_NATIVE). Ask weight_path(t, shape, kernel_types=...) per
    tensor for the path, the float32 scratch and the warning."""
    return _KERNELS["types"] | _KERNELS["nb"]


def native_types():
    return _KERNELS["nb"]


def weight_path(t, shape=(), experts=None, kernel_types=None):
    """-> "native" | "kq" | "exact": the path a weight of ggml type t and GGUF shape takes. experts: the caller says
    whether the tensor is a stack of experts (arch modules know it from the tensor's role, e.g. ffn_*_exps); None
    infers it from the dims (more than 2 = experts), which holds for the archs loaded today but not for every arch
    (deepseek2's MLA attn_k_b / attn_v_b are 3-D), so such an arch passes experts=False. kernel_types limits the kernel
    paths to those types (frozenset() = the exact path for everything)."""
    t = int(t)
    allowed = (_KERNELS["types"] | _KERNELS["nb"]) if kernel_types is None else frozenset(int(v) for v in kernel_types)
    if t not in allowed:
        return "exact"
    experts = len(tuple(shape)) > 2 if experts is None else bool(experts)
    kq = t in _KERNELS["types"]
    if t in _KERNELS["nb"] and (experts or DENSE_NATIVE):
        return "native"
    return "kq" if kq else "exact"


def device_nbytes(t, shape, nbytes, kernel_types=None, experts=None):
    """device bytes of one tensor of GGUF shape (..., rows, cols) whose file blocks take nbytes, on the path weight_path
    picks: the native planes (nb_bytes, every matrix of a stack counted), the kq planes (plane_bytes, once per matrix of
    a stacked weight), else the blocks as stored"""
    mod = _KERNELS["module"]
    path = weight_path(t, shape, experts, kernel_types)
    if path == "native" and mod is not None and hasattr(mod, "nb_bytes"):
        return int(mod.nb_bytes(int(t), tuple(int(v) for v in shape)))
    if path == "kq" and mod is not None and hasattr(mod, "plane_bytes"):
        lead = 1
        for v in shape[:-2]:            # stacked weights (experts, rows, cols): every matrix has its planes
            lead *= int(v)
        return lead * int(mod.plane_bytes(int(t), tuple(int(v) for v in shape[-2:])))
    return int(nbytes)


HOST_REPACK = None      # None: repack host arrays on the host CPU when the default device is not the CPU; True: always


def _host_cpu():
    """the host's XLA:CPU device, or None when JAX runs without it (JAX_PLATFORMS without cpu)"""
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


def _flat(raw):
    """(..., rows, row_bytes) -> (..., rows * row_bytes): the exact path keeps a weight's blocks as one byte run. On TPU a
    2-D uint8 parameter gets a column-major tiled layout, and a loop that reads it (the engine's K-step decode program)
    carries it in the row-major layout its body wants, so XLA copied every raw weight once before the loop and kept all
    the copies alive (27B UD-Q4_K_M: 34.35 GB of temporaries in the decode program on one v6e vs 2.45 GB at one step).
    A 1-D run has one layout; the rows are formed after the barrier in qmatmul, one weight or chunk at a time."""
    return raw.reshape(raw.shape[:-2] + (raw.shape[-2] * raw.shape[-1],))


def _mat(raw, t, shape):
    """a Raw handle's blocks as (..., rows, row_bytes) uint8, from either form"""
    rows, cols = shape[-2], shape[-1]
    _, bs, ts = R.TYPES[t]
    rb = cols // bs * ts
    if raw.ndim >= 2 and tuple(raw.shape[-2:]) == (rows, rb):
        return raw
    return raw.reshape(raw.shape[:-1] + (rows, rb))


def prepare(raw, t, shape, use_kernel=True, experts=None):
    """raw: host (NumPy) or device uint8 array (rows, row_bytes), or with leading axes (a stack of same-shape tensors, e.g.
    one role of every layer group) -> Raw or Packed handle; shape is one tensor's (rows, cols). A stack is repacked in one
    program (the kernel's repack mapped over the leading axes), never one tensor at a time and then stacked.

    A kernel's repack is a byte permutation: a host array is repacked on the host's CPU (XLA:CPU compiles it in a moment;
    on a TPU every role's repack was a separate slow compile and device scratch next to the weights) and the packed
    planes are sent to the default device once. When the kernel package has repack_host, a host array goes through it
    instead: XLA:CPU runs the repack of a large tensor on one thread for many minutes, the package's NumPy repack splits
    the rows over the host's cores.

    Native planes (weight_path "native"): the package's repack_nb_host runs once on the host and the planes go to the
    default device; leading axes of raw are kept as the planes' stacked dims. If it fails, a dense weight falls back to
    kq (when kq covers the type) or to the exact path, a stacked expert weight to the exact path (the file's bytes, never
    more than the native planes the memory estimate counted), and a warning names the type, the shape and the error.

    kq planes (weight_path "kq"): the first weight of each (type, shape) also runs the kernel's matmul (_kq_preflight).
    If the repack or the preflight raises, this weight and every later one of that type and shape load on the exact
    path, and one warning names the type, the shape and the error."""
    t = int(t)
    if t not in R.TYPES:
        raise NotImplementedError(f"ggml type {t} has no reader in this version")
    path = weight_path(t, shape, experts, None if use_kernel else frozenset())
    if path == "native":
        try:
            planes = _KERNELS["module"].repack_nb_host(np.asarray(raw), t, tuple(int(v) for v in shape[-2:]))
            return NB(jax.device_put(planes, jax.devices()[0]), t, shape)
        except Exception as e:                  # a fast path must never stop the load: fall back, loudly
            stack = len(tuple(shape)) > 2 if experts is None else bool(experts)
            path = "kq" if t in _KERNELS["types"] and not stack else "exact"
            warn(f"mashbox: native planes for {type_name(t)} {tuple(shape)} failed ({type(e).__name__}: {e}); "
                 f"this weight runs on the {'kq planes' if path == 'kq' else 'exact path'}")
    key = (t, tuple(int(s) for s in shape))
    if path == "kq" and _KQ_RAN.get(key, True) is True:
        try:
            h = _kq_planes(raw, t, shape)
            if key not in _KQ_RAN:
                _kq_preflight(h, max(0, raw.ndim - 2))
                _KQ_RAN[key] = True
            if raw.ndim == 3 and len(key[1]) == 2 and key not in _KQ_LAYER and kernel_picks_layer():
                _kq_layer_preflight(h, key)
            return h
        except Exception as e:                  # a fast path must never stop the load: fall back, loudly
            _KQ_RAN[key] = f"{type(e).__name__}: {e}"
            warn(f"mashbox: kq planes for {type_name(t)} {key[1]} failed ({_KQ_RAN[key]}); weights of this type and "
                 f"shape run on the exact path")
    if use_kernel and t in dense16.TYPES:              # BF16: native bfloat16 matmul, else the dequantized path below
        h = dense16.prepare(raw, t, shape)
        if h is not None:
            return h
    return Raw(jnp.asarray(_flat(raw)), t, shape)


def _kq_planes(raw, t, shape):
    """raw -> Packed handle on the kernel package's kq planes (see prepare)"""
    mod = _KERNELS["module"]
    if not isinstance(raw, jax.Array) and hasattr(mod, "repack_host"):
        return Packed(jax.device_put(mod.repack_host(t, np.asarray(raw), tuple(shape)), jax.devices()[0]), t, shape)
    rep = lambda r: mod.repack(t, r, tuple(shape))
    for _ in range(max(0, raw.ndim - 2)):
        rep = jax.vmap(rep)
    cpu = _host_cpu()
    dev = jax.devices()[0]
    if cpu is not None and not isinstance(raw, jax.Array) and (HOST_REPACK or dev.platform != "cpu"):
        packed = jax.jit(rep)(jax.device_put(raw, cpu))
        return Packed(jax.device_put(packed, dev), t, shape)
    raw = jnp.asarray(raw)
    return Packed(jax.jit(rep)(raw) if raw.ndim > 2 else rep(raw), t, shape)


def _kq_preflight(h, lead):
    """The kernel's matmul, compiled and run once on one tensor of h (index 0 of each of its `lead` stacked axes) with one
    row of input in the forward's dtype; it waits for the result, so the repack's planes are on the device by then.
    Raises what the kernel raises at trace, lowering, compile or run. It shows the kernel runs, not its values: those are
    the kernel package's own tests and each model's confirmation."""
    planes = jax.tree_util.tree_map(lambda a: a[(0,) * lead], h.packed) if lead else h.packed
    x = jnp.ones((1, h.shape[-1]), PREFLIGHT_DTYPE)
    jax.block_until_ready(jax.jit(_KERNELS["module"].matmul)(x, planes))


def _kq_layer_preflight(h, key):
    """The kernel's matmul with layer= on a whole layer stack (h: one (rows, cols) weight per layer), compiled and run
    once with a traced layer index, as the layer scan calls it (scan_split). If it raises, stacks of this type and shape
    are sliced per layer before the kernel instead (the plain call, which _kq_preflight ran), with one warning."""
    try:
        x = jnp.ones((1, h.shape[-1]), PREFLIGHT_DTYPE)
        mm = jax.jit(lambda x_, p_, l_: _KERNELS["module"].matmul(x_, p_, layer=l_))
        jax.block_until_ready(mm(x, h.packed, jnp.int32(0)))
        _KQ_LAYER[key] = True
    except Exception as e:                      # never stops the load: the sliced path still runs this weight
        _KQ_LAYER[key] = f"{type(e).__name__}: {e}"
        warn(f"mashbox: kq layer= for {type_name(key[0])} {key[1]} failed ({_KQ_LAYER[key]}); stacks of this type and "
             f"shape are sliced per layer before the kernel")


def dequant(h):
    """handle -> float32 weights (..., rows, cols), exact"""
    if isinstance(h, Pick):
        return _switch(h, dequant)
    if isinstance(h, NB):
        return _KERNELS["module"].nb_dequant(h.planes)
    if isinstance(h, Packed):
        return _KERNELS["module"].dequant(h.packed)
    if isinstance(h, dense16.Dense16):
        return dense16.dequant(h)
    return R.dequant(_mat(h.raw, h.t, h.shape), h.t, h.shape[-1])


def take_rows(h, ids):
    """rows `ids` of the weights as float32 (embedding lookup), exact"""
    if isinstance(h, NB):
        return _KERNELS["module"].nb_dequant(h.planes)[ids]
    if isinstance(h, Packed):
        return _KERNELS["module"].dequant(h.packed)[ids]
    if isinstance(h, dense16.Dense16):
        return dense16.take_rows(h, ids)
    rows, cols = h.shape[-2], h.shape[-1]
    _, bs, ts = R.TYPES[h.t]
    rb = cols // bs * ts
    if h.raw.ndim == 1:               # gather the rows' bytes from the run (no row-major copy of the whole table)
        rb_ids = jnp.asarray(ids)[..., None] * rb + jnp.arange(rb, dtype=jnp.int32)
        return R.dequant(h.raw[rb_ids], h.t, cols)
    return R.dequant(_mat(h.raw, h.t, h.shape)[ids], h.t, cols)


# float32 weight values the exact path rebuilds per step. A weight with more rows is rebuilt and multiplied a chunk of rows
# at a time (lax.map over a divisor of its rows), so the exact path's scratch is one chunk on any backend. On TPU the
# dequant's small minor axes (a scale per 16 or 32 values) are tiled to 128 lanes, which makes a whole-tensor rebuild many
# times the tensor's float32 size: the 248,320 x 5,120 output head of a 27B file needed more scratch than one v6e holds.
EXACT_CHUNK = 1 << 24


def exact_chunk_rows(rows, cols, limit=None):
    """-> rows per step of the exact path for a (rows, cols) weight: the largest divisor of rows with rows * cols <= limit
    (all rows when the weight is that small; 1 row when one row is already over the limit)"""
    limit = EXACT_CHUNK if limit is None else limit
    if rows * cols <= limit:
        return rows
    best, d = 1, 1
    while d * d <= rows:
        if rows % d == 0:
            for c in (d, rows // d):
                if c * cols <= limit and c > best:
                    best = c
        d += 1
    return best


def _exact(x, raw, t, cols):
    w = R.dequant(raw, t, cols)
    return lax.dot_general(x.astype(f32), w, (((x.ndim - 1,), (1,)), ((), ())), precision=HI, preferred_element_type=f32)


def qmatmul(x, h):
    """x (..., cols) @ W.T -> (..., rows), f32 accumulation, rounded to x.dtype.

    Exact path: the float32 weights are rebuilt from the blocks inside the call. The barrier ties the blocks to x, so XLA
    cannot rebuild a weight before its input exists: without it, a forward whose layers are unrolled (files that mix block
    types per layer) has every layer's float32 weights computed up front and live at once. A weight over EXACT_CHUNK values
    is rebuilt and multiplied a chunk of rows at a time (each output value is a dot over all cols either way; its float32
    rounding can differ from the whole-weight dot by the summation order).

    Native planes: one matrix's planes. A stack is multiplied by the routed expert op, never here; a lax.scan or vmap
    slice of a stack (one expert, one layer group) has no stacked dims left in its planes and multiplies as a dense
    weight (the handle's static shape still names the stack it came from)."""
    if isinstance(h, Pick):
        return _switch(h, lambda one, x_: qmatmul(x_, one), x)
    if isinstance(h, NB):
        lead = tuple(getattr(h.planes, "lead", ()))
        if lead:
            raise ValueError(f"{h}: planes stacked {lead} are multiplied by the routed expert op, not qmatmul")
        return _KERNELS["module"].matmul_nb(x, h.planes).astype(x.dtype)
    if isinstance(h, Packed):
        return _KERNELS["module"].matmul(x, h.packed).astype(x.dtype)
    if isinstance(h, At):
        return _KERNELS["module"].matmul(x, h.packed, layer=h.layer).astype(x.dtype)
    if isinstance(h, dense16.Dense16):
        return dense16.matmul(x, h)
    x, raw = lax.optimization_barrier((x, h.raw))
    rows, cols = h.shape[-2], h.shape[-1]
    rc = exact_chunk_rows(rows, cols)
    raw = _mat(raw, h.t, h.shape)
    if rc == rows or raw.ndim != 2:
        return _exact(x, raw, h.t, cols).astype(x.dtype)
    chunks = raw.reshape(rows // rc, rc, raw.shape[-1])
    ys = lax.map(lambda r: _exact(x, r, h.t, cols), chunks)                  # (rows // rc, ..., rc)
    return jnp.moveaxis(ys, 0, -2).reshape(x.shape[:-1] + (rows,)).astype(x.dtype)


def kernel_picks_layer():
    """True when the loaded kernel package's matmul takes layer= (see the module docstring)"""
    mm = getattr(_KERNELS["module"], "matmul", None)          # a package of expert kernels only has no dense matmul
    try:
        return mm is not None and "layer" in inspect.signature(mm).parameters
    except (TypeError, ValueError):
        return False


def _is_handle(v):
    return isinstance(v, (Raw, Packed, NB, At, Mixed, Pick, dense16.Dense16))


def _held(v):
    """a Packed stack of one (rows, cols) weight per layer whose (type, shape) passed the layer= preflight"""
    return isinstance(v, Packed) and len(v.shape) == 2 and _KQ_LAYER.get((v.t, v.shape)) is True


def scan_split(ws):
    """ws: a tree of weights stacked over layer groups, about to be the xs of a lax.scan -> (held, xs). When the kernel
    package picks a layer itself, each Packed handle of a (rows, cols) weight is held out of the scan (held keeps the
    handle, xs has None there) so the body multiplies by the whole stack at its layer (scan_join), and nothing slices
    the planes. A Mixed handle (block types differ by layer) is always held, and xs has its (which, sub) per layer.
    Everything else is scanned as before. With neither, held is None and xs is ws."""
    picks = kernel_picks_layer()
    leaves = jax.tree_util.tree_leaves(ws, is_leaf=_is_handle)
    if not any(isinstance(v, Mixed) or (picks and _held(v)) for v in leaves):
        return None, ws

    def hold(v):
        return v if isinstance(v, Mixed) or (picks and _held(v)) else None

    def scan(v):        # a Mixed handle scans its (which, sub) per layer; a held stack scans nothing
        return (v.which, v.sub) if isinstance(v, Mixed) else None if hold(v) is not None else v
    return (jax.tree_util.tree_map(hold, ws, is_leaf=_is_handle),
            jax.tree_util.tree_map(scan, ws, is_leaf=_is_handle))


def scan_join(held, w, layer):
    """inside the scan body: the layer's weights = w (the scanned slice) with each held handle as At(stack, layer), and
    each held Mixed handle as Pick(stacks, which, sub) from its scanned (which, sub)"""
    if held is None:
        return w

    def join(h, v):
        if h is None:
            return v
        if isinstance(h, Mixed):
            return Pick(h.stacks, v[0], v[1], h.types, h.shape)
        return At(h.packed, layer, h.t, h.shape)
    return jax.tree_util.tree_map(join, held, w, is_leaf=lambda v: v is None or _is_handle(v))


def contract_check(t, raw, shape):
    """The READER <-> kernel contract for one tensor: dequant(repack(raw)) == reference dequant(raw), bit for bit.
    -> (ok, mismatching values). Runs only when the kernel package covers type t."""
    mod = _KERNELS["module"]
    if t in _KERNELS["nb"]:
        ref = R.dequant(jnp.asarray(raw), t, shape[-1])
        got = mod.nb_dequant(jax.device_put(mod.repack_nb_host(np.asarray(raw), t, tuple(shape[-2:]))))
        bad = int(jnp.sum(lax.bitcast_convert_type(ref, jnp.uint32) != lax.bitcast_convert_type(got.astype(f32), jnp.uint32)))
        return bad == 0, bad
    if t not in _KERNELS["types"]:
        return None
    ref = R.dequant(jnp.asarray(raw), t, shape[-1])
    got = _KERNELS["module"].dequant(_KERNELS["module"].repack(t, jnp.asarray(raw), tuple(shape)))
    bad = int(jnp.sum(lax.bitcast_convert_type(ref, jnp.uint32) != lax.bitcast_convert_type(got.astype(f32), jnp.uint32)))
    return bad == 0, bad


def warn(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------- routed experts
# The routed-expert op (mashbox.moe.ffn calls it). The kernel package provides it when it has `expert_ffn` and its
# `EXPERT_TYPES` cover every handle's type (a package without EXPERT_TYPES runs no routed experts: its TYPES are the dense
# matmul's); kq-plane (Packed) and native-plane (NB) handles reach it as their planes. Before the kernel is taken for a
# call shape it is lowered and compiled once on its own (cached per shape), so a failure while tracing, lowering (an op
# with no TPU lowering) or compiling (a kernel compile error, VMEM overflow) sends that shape to the exact reference
# below, with a warning naming the cause. With `layer` (the arch's lax.scan group index) every expert handle and bias
# is the whole stack over layer groups (groups, E, ...): the kernel reads that group's experts in place through its
# index_map and the reference gathers only that group's picked experts (a scan slice of the stack would instead be
# copied into the kernel's call at every layer). Reference: the picked experts' blocks only (file blocks, kq planes, or
# native planes plane by plane) are gathered into U = min(P, E) slots and rebuilt, then one grouped matmul over the
# slots: on TPU / GPU lax.ragged_dot (bfloat16 operands, float32 accumulation; float32 at HIGHEST is wrong there,
# measured); on CPU ragged_dot is a dense masked dot (G x the FLOPs, measured 16.25x at G = 16), so each pair multiplies
# its own slot's weights (gather, float32 at HIGHEST), in pair chunks of at most EXACT_CHUNK weight values. A projection
# whose slots hold more than EXACT_CHUNK weight values is rebuilt a chunk of slots at a time (ragged), or each pair
# rebuilds its own expert's blocks a chunk of pairs at a time (gather), so the exact path's rebuild scratch is at most
# EXACT_CHUNK values per projection on any backend.
EXPERT_IMPL = None          # None: by backend ("ragged" on TPU/GPU, "gather" on CPU); tests set "ragged" or "gather"
SWIGLU_OAI = (1.702, 7.0)   # llama.cpp LLM_FFN_SWIGLU_OAI_MOE: alpha, limit
_EXPERT_CHECKED = {}        # (module, act, E, argument shapes) -> None when the kernel compiles, else the error text


def _expert_impl():
    if EXPERT_IMPL is not None:
        return EXPERT_IMPL
    return "gather" if jax.default_backend() == "cpu" else "ragged"


def expert_act(g, u, act):
    """float32 gate / up projections -> activation (llama.cpp swiglu / geglu / swiglu_oai)"""
    if act == "silu":
        return jax.nn.silu(g) * u
    if act == "gelu":
        return jax.nn.gelu(g, approximate=True) * u
    if act == "swiglu_oai":
        alpha, limit = SWIGLU_OAI
        x, y = jnp.minimum(g, limit), jnp.clip(u, -limit, limit)
        return x / (1.0 + jnp.exp(-alpha * x)) * (y + 1.0)
    raise ValueError(f"mashbox: expert activation '{act}' is not supported")


def expert_slots(e_sorted, E):
    """-> (slot_expert (U,), slot_of_row (P,), U): the distinct experts of the sorted pairs in order (U = min(P, E) static
    slots; unused slots name expert 0 and get no rows) and each row's slot (U for the sentinel rows)"""
    P = e_sorted.shape[0]
    ok = e_sorted < E
    if P >= E:
        return jnp.arange(E, dtype=jnp.int32), jnp.where(ok, e_sorted, E), E
    U = P
    first = ok & jnp.concatenate([jnp.ones((1,), jnp.bool_), e_sorted[1:] != e_sorted[:-1]])
    sid = (jnp.cumsum(first.astype(jnp.int32)) - 1).astype(jnp.int32)
    slot_e = jnp.zeros((U,), jnp.int32).at[jnp.where(first, sid, U)].set(e_sorted, mode="drop")
    return slot_e, jnp.where(ok, sid, U).astype(jnp.int32), U


def _pick(a, layer, ids):
    """one array of a stacked expert handle (leading axes (E, ...), or (groups, E, ...) when layer is given) -> layer
    group `layer`'s experts `ids` (None: all)"""
    if layer is not None:
        return a[layer] if ids is None else a[layer, ids]
    return a if ids is None else a[ids]


def expert_weights(h, ids, x, dtype=f32, layer=None):
    """experts `ids` (None: all) of a stacked (E, rows, cols) handle, rebuilt: (len(ids), rows, cols) in dtype. layer:
    the handle is the whole stack over layer groups (groups, E, rows, cols) and layer picks the group. Only the picked
    experts' blocks are gathered: file blocks (Raw), kq planes (Packed, rebuilt per expert by the package's dequant) or
    native planes (NB, gathered plane by plane and rebuilt by the package's nb_dequant, which takes any leading dims).
    The barrier ties the blocks to the activations x (see qmatmul) so an unrolled forward never rebuilds every layer up
    front. -> (weights, x)"""
    sel = lambda a: _pick(a, layer, ids)
    if isinstance(h, NB):
        sub = jax.tree_util.tree_map(sel, h.planes)
        x, sub = lax.optimization_barrier((x, sub))
        return _KERNELS["module"].nb_dequant(sub).astype(dtype), x
    if isinstance(h, Packed):
        sub = jax.tree_util.tree_map(sel, h.packed)
        x, sub = lax.optimization_barrier((x, sub))
        return jax.vmap(_KERNELS["module"].dequant)(sub).astype(dtype), x
    raw = sel(h.raw)
    x, raw = lax.optimization_barrier((x, raw))
    return R.dequant(_mat(raw, h.t, h.shape), h.t, h.shape[-1]).astype(dtype), x


def _ragged(x, w, gsz):
    """x (P, cols) rows sorted by group, w (G, rows, cols), gsz (G,) -> (P, rows) float32; rows past sum(gsz) give 0"""
    dn = lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(([1], [2]), ([], [])), lhs_ragged_dimensions=[0],
                                       rhs_group_dimensions=[0])
    y = lax.ragged_dot_general(x.astype(w.dtype), w, gsz, dn, preferred_element_type=f32)
    return jnp.where((jnp.arange(x.shape[0]) < jnp.sum(gsz))[:, None], y, 0.0)


def expert_proj(x, h, slot_e, slot_of_row, gsz, e_rows, dtype, all_experts, impl, layer=None):
    """one projection of the sorted pairs: x (P, cols) -> (P, rows) float32 through each pair's expert (rows of no slot
    give 0). At most EXACT_CHUNK weight values are rebuilt at a time. layer: as expert_weights."""
    U, P = slot_e.shape[0], x.shape[0]
    rows, cols = h.shape[-2], h.shape[-1]
    S = exact_chunk_rows(U, rows * cols)                     # slots rebuilt per step
    ids = None if all_experts else slot_e
    if impl == "ragged":
        if S == U:
            w, x = expert_weights(h, ids, x, dtype, layer)
            return _ragged(x, w, gsz)
        start = (jnp.cumsum(gsz) - gsz).astype(jnp.int32)      # first row of each slot

        def step(c, acc):                                       # slots c*S .. c*S+S-1: their rows rolled to the front
            g = lax.dynamic_slice_in_dim(gsz, c * S, S)
            w, xs = expert_weights(h, lax.dynamic_slice_in_dim(slot_e, c * S, S), x, dtype, layer)
            off = start[c * S]
            return acc + jnp.roll(_ragged(jnp.roll(xs, -off, 0), w, g), off, 0)
        return lax.fori_loop(0, U // S, step, jnp.zeros((P, rows), f32))
    one = lambda xc, wc: jnp.einsum("pc,prc->pr", xc.astype(f32), wc, precision=HI, preferred_element_type=f32)
    if S == U:
        w, x = expert_weights(h, ids, x, dtype, layer)
        pick, idx = (lambda xc, sc: one(xc, w[sc])), jnp.minimum(slot_of_row, U - 1)
    else:                                                       # each pair rebuilds its own expert's blocks
        def pick(xc, ec):
            wc, xc = expert_weights(h, ec, xc, dtype, layer)
            return one(xc, wc)
        idx = e_rows
    pc = exact_chunk_rows(P, rows * cols)
    if pc == P:
        return pick(x, idx)
    ys = lax.map(lambda a: pick(*a), (x.reshape(P // pc, pc, cols), idx.reshape(P // pc, pc)))
    return ys.reshape(P, rows)


def _expert_ref(x_sorted, e_sorted, w_sorted, W, act, E, layer=None):
    impl = _expert_impl()
    dt = jnp.bfloat16 if impl == "ragged" and jax.default_backend() == "tpu" else f32
    ad = x_sorted.dtype
    slot_e, slot_row, U = expert_slots(e_sorted, E)
    full = x_sorted.shape[0] >= E
    gsz = jnp.zeros((U,), jnp.int32).at[slot_row].add(1, mode="drop")
    eb = jnp.minimum(e_sorted, E - 1)

    def proj(x, key):
        y = expert_proj(x, W[key], slot_e, slot_row, gsz, eb, dt, full, impl, layer)
        b = W.get(key + "_b")
        if b is not None:
            y = y + (b if layer is None else b[layer]).astype(f32)[eb]
        return y.astype(ad).astype(f32)                       # rounded to the activation dtype, as qmatmul returns it

    x = x_sorted
    if "gate_up" in W:
        gu = proj(x, "gate_up")
        F = gu.shape[-1] // 2
        g, u = gu[:, :F], gu[:, F:]
    else:
        g, u = proj(x, "gate"), proj(x, "up")
    a = expert_act(g, u, act).astype(ad)
    y = proj(a, "down")
    return jnp.where((e_sorted < E)[:, None], y * w_sorted[:, None], 0.0)


def _kernel_call(mod, args, act, E):
    """the package's expert_ffn(xs, es, ws, W, act, E), with layer= when args carry the layer-group index"""
    if len(args) == 4:
        return mod.expert_ffn(*args, act, E)
    return mod.expert_ffn(*args[:4], act, E, layer=args[4])


def _expert_kernel_error(mod, args, act, E):
    """lower and compile the kernel's expert_ffn on its own for these argument shapes, once per shape: -> None when it
    compiles, else the error text (the forward's own lower / compile happens later, outside any fallback)"""
    spec = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), args)
    leaves, tree = jax.tree_util.tree_flatten(spec)
    key = (mod, act, E, tree, tuple((a.shape, str(a.dtype)) for a in leaves))
    if key not in _EXPERT_CHECKED:
        try:
            jax.jit(lambda *a: _kernel_call(mod, a, act, E)).lower(*spec).compile()
            _EXPERT_CHECKED[key] = None
        except Exception as e:
            _EXPERT_CHECKED[key] = f"{type(e).__name__}: {e}"
    return _EXPERT_CHECKED[key]


def expert_ffn(x_sorted, e_sorted, w_sorted, W, act="silu", n_expert=None, layer=None):
    """Routed experts on pairs sorted by expert. x_sorted (P, D) activations (sentinel rows zero), e_sorted (P,) int32
    expert per row (n_expert = sentinel: not computed), w_sorted (P,) float32 routing weights. W: handles (Raw, Packed
    or NB) stacked (E, rows, cols) under gate / up / down (or gate_up, fused), optional gate_b / up_b / down_b /
    gate_up_b (E, rows) float32. layer: None, or a layer-group index (int32 scalar, traced is fine) when every W entry is
    the whole stack over layer groups (groups, E, ...): that group's experts are used, read in place. -> (P, D) float32:
    each row's expert output times its weight (0 for sentinel rows)."""
    E = int(n_expert if n_expert is not None else next(h.shape[0] for k, h in W.items() if not k.endswith("_b")))
    mod = _KERNELS["module"]
    hs = [h for k, h in W.items() if not k.endswith("_b")]
    et = getattr(mod, "EXPERT_TYPES", ()) if mod is not None else ()
    if (mod is not None and hasattr(mod, "expert_ffn") and all(isinstance(h, (Packed, NB)) for h in hs)
            and all(h.t in et for h in hs)):
        Wp = {k: (h.packed if isinstance(h, Packed) else h.planes if isinstance(h, NB) else h) for k, h in W.items()}
        args = (x_sorted, e_sorted, w_sorted, Wp) + (() if layer is None else (jnp.asarray(layer, jnp.int32),))
        if faults.forced("moe-routed"):                     # the fallback test's switch (MASHBOX_FAULT=force:moe-routed)
            err = "forced by MASHBOX_FAULT=force:moe-routed"
        else:
            err = _expert_kernel_error(mod, args, act, E)
        if err is None:
            try:
                return _kernel_call(mod, args, act, E).astype(f32)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        # the fast path falls back to the exact routed path, never silently
        warn(f"mashbox: kernel expert_ffn failed ({err}); routed experts run on the exact path")
    return _expert_ref(x_sorted, e_sorted, w_sorted, W, act, E, layer)
