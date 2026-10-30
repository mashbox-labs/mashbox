"""Fixed-row serving adapter for SparseUnetDecoderJax.

The port's linear, normalization and C2S formulas are unchanged. Host indexing
gathers the same 27 neighbours into fixed tiles; the convolution accumulation
keeps the port's k=0..26 f32 dot/add order. Coordinates and expanded voxel arrays
stay on the host between calls. No port module is patched.
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

CAPS = (8192, 16384)


def _shape(x):
    return jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), x)


class TiledDecoder:
    def __init__(self, sparse, params, model_channels, num_blocks, pred_subdiv):
        self.s, self.p = sparse, params
        self.mc, self.nb, self.pred_subdiv = model_channels, num_blocks, pred_subdiv
        self.executables = {}
        # Only the numerical MODULE and static subdivision flag are captured.
        # Every parameter leaf and per-job buffer is a compiled argument.
        def conv(patch, wk, bias):
            def body(k, acc):
                return acc + jnp.dot(patch[:, k], wk[k], precision=jax.lax.Precision.HIGHEST,
                                     preferred_element_type=jnp.float32)
            y = jax.lax.fori_loop(0, 27, body, jnp.zeros((patch.shape[0], wk.shape[-1]), jnp.float32))
            return y + bias.astype(jnp.float32)

        def convnext(p, x, patch, valid):
            h = conv(patch, p["conv.wk"], p["conv.bias"])
            h = sparse.layer_norm(h, p["norm.weight"], p["norm.bias"], 1e-6)
            h = jax.nn.silu(sparse.linear(h, p["mlp.0.weight"], p["mlp.0.bias"]))
            return sparse.linear(h, p["mlp.2.weight"], p["mlp.2.bias"]) + x

        def pre(p, x, patch, valid):
            logits = sparse.linear(x, p["to_subdiv.weight"], p["to_subdiv.bias"]) if pred_subdiv else None
            h = jax.nn.silu(sparse.layer_norm(patch, p["norm1.weight"], p["norm1.bias"], 1e-6))
            # The port appends a zero sentinel AFTER normalization.
            h = jnp.where(valid[..., None], h, 0.)
            return conv(h, p["conv1.wk"], p["conv1.bias"]), logits

        def post(p, residual, patch, valid):
            h = jax.nn.silu(sparse.layer_norm(patch, None, None, 1e-6))
            h = jnp.where(valid[..., None], h, 0.)
            h = conv(h, p["conv2.wk"], p["conv2.bias"])
            return h + jnp.repeat(residual, h.shape[-1] // residual.shape[-1], axis=1)

        def head(p, x):
            return sparse.linear(sparse.layer_norm(x, None, None, 1e-5), p["weight"], p["bias"])

        self.programs = {"convnext": jax.jit(convnext), "pre": jax.jit(pre), "post": jax.jit(post),
                         "linear": jax.jit(lambda p, x: sparse.linear(x, p["weight"], p["bias"])),
                         "head": jax.jit(head)}
        self.blocks = []
        for level, n in enumerate(num_blocks):
            for i in range(n + int(level < len(num_blocks) - 1)):
                name = f"blocks.{level}.{i}"
                kind = "convnext" if i < n else "pre"
                self.blocks.append((level, name, kind, sparse.block_params(params, name)))

    def _projection(self, name):
        return {"weight": self.p[name + ".weight"], "bias": self.p[name + ".bias"]}

    def warm_up(self):
        records = []
        def compile_one(name, kind, p, channels, patch_channels=None):
            for cap in CAPS:
                start = time.perf_counter()
                x = jax.ShapeDtypeStruct((cap, channels), jnp.float32)
                args = [_shape(p), x]
                if patch_channels is not None:
                    args += [jax.ShapeDtypeStruct((cap, 27, patch_channels), jnp.float32),
                             jax.ShapeDtypeStruct((cap, 27), jnp.bool_)]
                self.executables[(name, cap)] = self.programs[kind].lower(*args).compile()
                records.append({"block": name, "bucket": cap, "compile_s": time.perf_counter() - start})
        compile_one("from_latent", "linear", self._projection("from_latent"), self.p["from_latent.weight"].shape[1])
        for level, name, kind, p in self.blocks:
            compile_one(name, kind, p, self.mc[level], self.mc[level])
            if kind == "pre":
                compile_one(name + ".post", "post", p, self.mc[level] // 8, self.mc[level + 1])
        compile_one("output_layer", "head", self._projection("output_layer"), self.mc[-1])
        return records

    def _neighbors(self, coords, grid):
        n = len(coords)
        if not n:
            return np.empty((0, 27), np.int32)
        def key(c):
            return (c[:, 0].astype(np.int64) * grid + c[:, 1]) * grid + c[:, 2]
        keys = key(coords)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        out = np.empty((n, 27), np.int32)
        for k, off in enumerate(self.s.OFFSETS):
            q = coords + off
            valid = ((q >= 0) & (q < grid)).all(1)
            qkey = key(q)
            at = np.searchsorted(sorted_keys, qkey)
            safe = np.minimum(at, n - 1)
            found = valid & (at < n) & (sorted_keys[safe] == qkey)
            out[:, k] = np.where(found, order[safe], -1)
        return out

    @staticmethod
    def _patch(x, neighbors, cap):
        n, c = len(neighbors), x.shape[1]
        patch = np.zeros((cap, 27, c), np.float32)
        valid = np.zeros((cap, 27), bool)
        for k in range(27):
            idx = neighbors[:, k]
            good = idx >= 0
            valid[:n, k] = good
            patch[:n, k][good] = x[idx[good]]
        return patch, valid

    def _apply(self, name, kind, p, x, stats, *, neighbor_x=None, neighbors=None, out_channels=None):
        n = len(x)
        ys, ls = [], []
        for start in range(0, n, CAPS[-1]):
            count = min(n - start, CAPS[-1])
            cap = CAPS[0] if count <= CAPS[0] else CAPS[1]
            tile = np.zeros((cap, x.shape[1]), np.float32)
            tile[:count] = x[start:start + count]
            args = [p, tile]
            if neighbors is not None:
                patch, valid = self._patch(neighbor_x, neighbors[start:start + count], cap)
                args += [patch, valid]
            compiled = self.executables.get((name, cap))
            if compiled is None:
                raise RuntimeError(f"decoder block {name} bucket {cap} was not warmed at load")
            begin = time.perf_counter()
            value = jax.block_until_ready(compiled(*args))
            stats["device_s"] += time.perf_counter() - begin
            if kind == "pre":
                y, logits = value
                if logits is not None:
                    ls.append(np.asarray(logits)[:count])
            else:
                y = value
            ys.append(np.asarray(y)[:count])
            stats["tiles"].append({"block": name, "tokens": count, "bucket": cap,
                                   "padding_tokens": cap - count, "waste_fraction": (cap - count) / cap})
        if not n:
            width = out_channels
            return (np.empty((0, width), np.float32),
                    np.empty((0, 8), np.float32) if self.pred_subdiv else None) if kind == "pre" else np.empty((0, width), np.float32)
        out = np.concatenate(ys)
        return (out, np.concatenate(ls) if ls else None) if kind == "pre" else out

    def run(self, feats, coords, grid, guide_subs=None, upsample_times=None):
        begin = time.perf_counter()
        stats = {"device_s": 0., "tiles": []}
        x, c = np.asarray(feats, np.float32), np.asarray(coords, np.int32)
        if x.ndim != 2 or c.shape != (len(x), 3):
            raise ValueError("decoder needs features [N,C] and coordinates [N,3]")
        h = self._apply("from_latent", "linear", self._projection("from_latent"), x, stats,
                        out_channels=self.mc[0])
        counts, subs = [len(c)], []
        neighbors = self._neighbors(c, grid)
        for level, name, kind, p in self.blocks:
            if upsample_times is not None and level == upsample_times:
                return self._result(None, c, subs, counts, grid, stats, begin)
            if kind == "convnext":
                h = self._apply(name, kind, p, h, stats, neighbor_x=h, neighbors=neighbors,
                                out_channels=self.mc[level])
                continue
            hh, logits = self._apply(name, kind, p, h, stats, neighbor_x=h, neighbors=neighbors,
                                     out_channels=8*self.mc[level + 1])
            if self.pred_subdiv:
                subs.append((c.copy(), logits))
                mask = logits > 0
            else:
                if guide_subs is None or np.shape(guide_subs[level]) != (len(c), 8):
                    raise ValueError("guided subdivision mask does not match parent coordinates")
                mask = np.asarray(guide_subs[level], bool)
            f = np.flatnonzero(mask.reshape(-1))
            parent, child = f // 8, f % 8
            bits = np.stack((child & 1, (child >> 1) & 1, (child >> 2) & 1), 1)
            c = (c[parent] * 2 + bits).astype(np.int32)
            hs = hh.reshape(-1, self.mc[level + 1])[f]
            residual = h.reshape(-1, self.mc[level] // 8)[f]
            grid *= 2
            neighbors = self._neighbors(c, grid)
            h = self._apply(name + ".post", "post", p, residual, stats,
                            neighbor_x=hs, neighbors=neighbors, out_channels=self.mc[level + 1])
            counts.append(len(c))
        # A terminal level can have zero blocks (the real shape checkpoint does).
        # The port checks the upsample stop before entering that level.
        if upsample_times == len(self.nb) - 1:
            return self._result(None, c, subs, counts, grid, stats, begin)
        h = self._apply("output_layer", "head", self._projection("output_layer"), h, stats,
                        out_channels=self.p["output_layer.weight"].shape[0])
        return self._result(h, c, subs, counts, grid, stats, begin)

    @staticmethod
    def _result(h, c, subs, counts, grid, stats, begin):
        wall = time.perf_counter() - begin
        return {"h": h, "coords": c, "subs": subs, "counts": counts, "grid": grid,
                "timing": {"device_s": stats["device_s"], "host_s": max(0., wall - stats["device_s"])},
                "padding": stats["tiles"]}
