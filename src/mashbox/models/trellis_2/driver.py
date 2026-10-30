"""TRELLIS.2 S0--S10 host driver for the unchanged numerical ports.

mashbox run trellis-2 runs the 512 pipeline (the package pins the 512 weights only), with the
vendor 12-step samplers and bf16 DiT blocks with the port's f32 exceptions. Background removal and the choice of chip are
outside this driver. Reduced steps/blocks/texture (tests) run on CPU only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .adapters import BUCKETS, Flow, NoiseStream, abstract, choose_bucket, host_heads, import_port, load_decoder, load_flow, pack

PRECISION = {"flow_trunk": "bf16", "flow_top_norm_accumulation": "f32",
             "dino": "f32", "ss_decoder": "f32", "shape_decoder": "f32",
             "texture_decoder": "f32", "sampler_latents": "f32"}
EXPORT = {"decimation_target": 1_000_000, "texture_size": 4096,
          "remesh": True, "remesh_band": 1, "remesh_project": 0}
MODELS = {"S2": "sparse_structure_flow_model", "S4": "shape_slat_flow_model_512",
          "S6": "shape_slat_flow_model_1024", "S7": "tex_slat_flow_model_1024"}


def _finite(value, name):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def describe(x):
    if isinstance(x, (np.ndarray, jax.Array)):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {k: describe(v) for k, v in x.items() if k not in ("params",)}
    if isinstance(x, (tuple, list)):
        return [describe(v) for v in x]
    if isinstance(x, Image.Image):
        return {"mode": x.mode, "size": list(x.size)}
    return x if isinstance(x, (str, int, float, bool, type(None))) else type(x).__name__


class Job:
    def __init__(self, seed, pipeline, label, progress=None, cancelled=None):
        self.progress, self.cancelled = progress, cancelled
        self.metadata = {"seed": seed, "pipeline": pipeline, "precision": dict(PRECISION),
                         "label": label, "backend": jax.default_backend(), "stages": [],
                         "padding": [], "host_overlap": "run_stages/export_glb boundary"}

    def check(self):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("generation cancelled")

    def emit(self, event):
        if self.progress is not None:
            self.progress({"label": self.metadata["label"], **event})

    def stage(self, name, location, fn):
        self.check()
        self.emit({"stage": name, "phase": "start"})
        start = time.perf_counter()
        row = {"stage": name, "location": location}
        try:
            result = fn()
            jax.block_until_ready(result) if isinstance(result, (jax.Array, tuple, list, dict)) else None
            row["output"] = describe(result)
            if isinstance(result, dict) and "timing" in result:
                row["device_s"] = result["timing"]["device_s"]
            row["status"] = "ran"
            return result
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row["wall_s"] = time.perf_counter() - start
            row.setdefault("device_s", row["wall_s"] if location == "device" else 0.)
            row["host_s"] = max(0., row["wall_s"] - row["device_s"])
            self.metadata["stages"].append(row)
            self.emit({"stage": name, "phase": row.get("status", "error"), "wall_s": row["wall_s"]})

    def totals(self):
        host = sum(r["host_s"] for r in self.metadata["stages"])
        device = sum(r["device_s"] for r in self.metadata["stages"])
        stages = {}
        for r in self.metadata["stages"]:
            key = r["stage"].split(".")[0]
            stages[key] = stages.get(key, 0.) + r["wall_s"]
        self.metadata.update(host_time_s=host, device_stage_wall_s=device, stage_wall_s=stages,
                             chip_time_s=device if self.metadata["backend"] != "cpu" else None,
                             timing_note="Synchronized stage wall times; device stages include host orchestration. "
                                         "CPU runs do not measure chip time.")
        return self.metadata


class Trellis2Driver:
    def __init__(self, *, port_dir=None, cache_dir=None, pipeline="1024_cascade",
                 wiring_steps=None, wiring_blocks=None, wiring_texture_size=None):
        if pipeline not in ("512", "1024_cascade"):
            raise ValueError("pipeline must be 512 or 1024_cascade")
        self.port_dir = Path(port_dir or Path(__file__).resolve().parent.parent)
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.cache_dir = self.cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR")
        self.pipeline = pipeline
        self.wiring_steps, self.wiring_blocks = wiring_steps, wiring_blocks
        self.wiring_texture_size = wiring_texture_size
        self.wiring = any(v is not None for v in (wiring_steps, wiring_blocks, wiring_texture_size))
        self.label = "reduced" if self.wiring else "full"
        self.loaded = False
        self.flows, self.params = {}, {}
        self.last_job = None

    def _resolve(self, root):
        """The files at the pinned revisions only (the weights.repo_dir layout): another cached revision or an older flat
        layout (<root>/pipeline.json, <root>/dinov3) never loads."""
        from .weights import pins
        pinned = {r["hf"]: root / (r["hf"].replace("/", "_") + "@" + r["revision"]) for r in pins()["repos"].values()}
        self.model_root = pinned.get("microsoft/TRELLIS.2-4B")
        if self.model_root is None or not (self.model_root / "pipeline.json").is_file():
            raise FileNotFoundError("pipeline.json missing from staged TRELLIS.2 weights")
        self.pipeline_path = self.model_root / "pipeline.json"
        args = json.loads(self.pipeline_path.read_text())["args"]
        self.model_paths = {}
        for key, rel in args["models"].items():
            p = self.model_root / rel
            if not Path(str(p) + ".json").is_file():
                large = pinned.get("microsoft/TRELLIS-image-large")
                p = large / "ckpts" / Path(rel).name if large is not None else None
                if p is None or not Path(str(p) + ".json").is_file():
                    raise FileNotFoundError(f"missing model {key}: {rel}")
            self.model_paths[key] = p
        self.dino_path = pinned.get("facebook/dinov3-vitl16-pretrain-lvd1689m")
        if self.dino_path is None or not (self.dino_path / "config.json").is_file():
            raise FileNotFoundError("dinov3 staged directory missing")
        return args

    def _given(self, root, required, verified):
        """Every required file is held here to its own sha256 in upstream.json (weights.pins), whatever the caller passes.
        The map given to load() (path -> sha256, from weights.stage) must name each file with exactly that pin, or with
        weights.UNCHECKED when MASHBOX_VERIFY=0 skipped the check, which the run metadata then says."""
        from ... import hub
        from .weights import UNCHECKED, pins
        pinned = {str((root / (r["hf"].replace("/", "_") + "@" + r["revision"]) / f["name"]).resolve()): f["sha256"]
                  for r in pins()["repos"].values() for f in r["files"]}
        known = {str(Path(p).resolve()): h for p, h in verified.items()}
        skipped = os.environ.get("MASHBOX_VERIFY", "1") == "0"
        out = []
        for path in required:
            pin, h = pinned.get(str(path.resolve())), known.get(str(path.resolve()))
            if pin is None:
                raise ValueError(f"no pin in upstream.json for {path}")
            if h is None:
                raise ValueError(f"weight not checked before load: {path}")
            if h == UNCHECKED:
                if not skipped:
                    raise ValueError(f"{path} is marked not checked, but MASHBOX_VERIFY=0 is not set")
            elif h != pin:
                raise ValueError(f"{path}: the sha256 given to load() is not its pin in upstream.json")
            else:
                hub.verify(str(path), pin, log=lambda s: None)
            out.append({"file": str(path.relative_to(root)), "sha256": h})
        return out

    def load(self, directory, progress=None, verified=None):
        if self.loaded:
            raise RuntimeError("unload before loading again")
        if verified is None:
            raise ValueError("load() needs the map of files checked by sha256 (weights.stage gives it)")
        if self.wiring and jax.default_backend() != "cpu":
            raise ValueError("reduced steps/blocks/texture are CPU-only")
        if not self.cache_dir:
            raise ValueError("persistent compile cache required: cache_dir or JAX_COMPILATION_CACHE_DIR")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(Path(self.cache_dir).resolve()))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_default_matmul_precision", "highest")
        start = time.perf_counter()
        root = Path(directory).resolve()
        self.args = self._resolve(root)
        keys = dict(MODELS)
        if self.pipeline == "512":
            keys.pop("S6")
            keys["S7"] = "tex_slat_flow_model_512"
        required = [self.pipeline_path, self.dino_path / "config.json", self.dino_path / "model.safetensors"]
        for key in set(keys.values()) | {"sparse_structure_decoder", "shape_slat_decoder", "tex_slat_decoder"}:
            required += [Path(str(self.model_paths[key]) + suffix) for suffix in (".json", ".safetensors")]
        self.verified = self._given(root, required, verified)
        verified_ms = (time.perf_counter() - start) * 1000
        self.modules = m = import_port(self.port_dir)
        if m["preprocess"].RESIZE != "ours":
            raise ValueError("serving requires default resample_u8 (TRELLIS_S0_RESIZE=ours)")
        from .flow.schedule import load_pipeline_samplers
        from .flow.glue import load_normalization
        samplers = load_pipeline_samplers(str(self.pipeline_path))
        if self.wiring_steps is not None:
            if not 1 <= self.wiring_steps <= 12:
                raise ValueError("reduced steps must be 1..12")
            samplers = {k: replace(v, steps=self.wiring_steps) for k, v in samplers.items()}
        self.normalization = load_normalization(str(self.pipeline_path))
        start = time.perf_counter()
        self.dino = m["dinov3_jax"].load_params(str(self.dino_path))
        if self.wiring_blocks is not None:
            self.dino["layers"] = jax.tree_util.tree_map(lambda a: a[:self.wiring_blocks], self.dino["layers"])
        for stage, key in keys.items():
            sampler = samplers["sparse_structure" if stage == "S2" else "tex_slat" if stage == "S7" else "shape_slat"]
            flow, params = load_flow(m["dit_trunk"], self.model_paths[key], sampler,
                                     wiring_blocks=self.wiring_blocks)
            self.flows[stage], self.params[stage] = flow, params
            if progress:
                progress({"label": self.label, "stage": "load", "model": key})
        self.ss, self.ss_cfg = m["ss_decoder"].load_params(str(self.model_paths["sparse_structure_decoder"]))
        self.ss_program = jax.jit(lambda p, z: m["ss_decoder"].decode_logits(p, z))
        self.occupancy_program = jax.jit(lambda logits: m["ss_decoder"].occupancy(logits, 32))
        self.shape = load_decoder(m, self.model_paths["shape_slat_decoder"], True)
        self.texture = load_decoder(m, self.model_paths["tex_slat_decoder"], False)
        jax.block_until_ready(self.arrays())
        read_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        self.warmup = self.warm_up(progress)
        compile_ms = (time.perf_counter() - start) * 1000
        self.loaded = True
        self.load_metadata = {"prep_ms": verified_ms, "read_ms": read_ms, "compile_ms": compile_ms,
                              "read_note": "Checkpoint conversion and device placement included.",
                              "warmup": self.warmup, "verified": self.verified, "label": self.label}
        return self.load_metadata

    def arrays(self):
        return [getattr(self, "dino", {}), self.params, getattr(self, "ss", {}),
                [list(f.negatives.values()) for f in self.flows.values()],
                getattr(getattr(self, "shape", None), "p", {}), getattr(getattr(self, "texture", None), "p", {})]

    def warm_up(self, progress=None):
        records = []
        self.dino_programs = {}
        for size in ((512,) if self.pipeline == "512" else (512, 1024)):
            start = time.perf_counter()
            self.dino_programs[size] = self.modules["dinov3_jax"].encode.lower(
                abstract(self.dino), jax.ShapeDtypeStruct((1, 3, size, size), jnp.float32)).compile()
            records.append({"stage": "S1", "image_size": size, "compile_s": time.perf_counter() - start})
        cfg = self.flows["S2"].cfg
        start = time.perf_counter()
        self.ss_compiled = self.ss_program.lower(abstract(self.ss),
            jax.ShapeDtypeStruct((1, cfg.out_channels, cfg.resolution, cfg.resolution, cfg.resolution), jnp.float32)).compile()
        side = cfg.resolution * 2 ** (len(self.ss_cfg["channels"]) - 1)
        self.occupancy_compiled = self.occupancy_program.lower(
            jax.ShapeDtypeStruct((1, 1, side, side, side), jnp.float32)).compile()
        records.append({"stage": "S3", "compile_s": time.perf_counter() - start})
        for stage, flow in self.flows.items():
            cond_tokens = 4101 if stage in ("S6", "S7") and self.pipeline != "512" else 1029
            for bucket in ((None,) if flow.cfg.dense else BUCKETS):
                start = time.perf_counter()
                rec = flow.compile(self.params[stage], cond_tokens, bucket)
                rec.update(stage=stage, compile_s=time.perf_counter() - start)
                records.append(rec)
                if progress:
                    progress({"label": self.label, "phase": "compiled", **rec})
        for stage, decoder in (("S5/S8", self.shape), ("S9", self.texture)):
            for row in decoder.warm_up():
                records.append({"stage": stage, **row})
            if progress:
                progress({"label": self.label, "phase": "compiled", "stage": stage,
                          "buckets": list(BUCKETS), "programs": len(decoder.executables)})
        return records

    def run_stages(self, image, *, seed=42, progress=None, cancelled=None, noise=None):
        if not self.loaded:
            raise RuntimeError("driver not loaded")
        if not isinstance(image, Image.Image) or image.mode != "RGBA":
            raise ValueError("RGBA required: supply an RGBA image with a usable foreground alpha channel")
        if type(seed) is not int or not -(1 << 63) <= seed < (1 << 64):
            raise ValueError("seed must be a torch.manual_seed-compatible integer")
        # The vendor draws CPU torch.randn in stage 2/stage 4/stage 6/stage 7 order after manual_seed.
        # A per-job generator preserves that stream without cross-request global state.
        rng = NoiseStream(seed, noise)
        job = self.last_job = Job(seed, self.pipeline, self.label, progress, cancelled)
        job.metadata["reductions"] = {"steps": self.wiring_steps, "blocks": self.wiring_blocks,
                                      "texture_size": self.wiring_texture_size}
        job.metadata["rng"] = "torch CPU Generator.manual_seed; draws before padding in vendor stage order"
        job.metadata["samplers"] = {k: vars(v.sampler_params) for k, v in self.flows.items()}
        job.metadata["compile_cache"] = str(self.cache_dir)
        m = self.modules

        normal = rng.normal

        def preprocess():
            rgba = np.asarray(image)
            if not np.any(rgba[:, :, 3] > 204) or np.all(rgba[:, :, 3] == 255):
                raise ValueError("RGBA required with a non-opaque, usable foreground alpha channel")
            pre = m["preprocess"].preprocess_rgba(image)
            sizes = (512,) if self.pipeline == "512" else (512, 1024)
            return {str(s): m["preprocess"].to_encoder_input([pre], s) for s in sizes}
        inputs = job.stage("S0", "host", preprocess)
        cond = job.stage("S1", "device", lambda: {
            s: self.dino_programs[int(s)](self.dino, jnp.asarray(x)) for s, x in inputs.items()})
        flow = self.flows["S2"]
        cfg = flow.cfg
        z = job.stage("S2", "device", lambda: flow.run(self.params["S2"],
            jnp.asarray(normal("S2", (1, cfg.in_channels, cfg.resolution, cfg.resolution, cfg.resolution))), cond["512"]))
        logits = job.stage("S3.decode", "device", lambda: self.ss_compiled(self.ss, z))
        occ = job.stage("S3.occupancy", "device", lambda: self.occupancy_compiled(logits))
        coords = job.stage("S3.coordinates", "host", lambda: np.argwhere(
            np.asarray(occ))[:, 1:].astype(np.int32))

        def sample(stage, xyz, condition, concat=None):
            n = len(xyz)
            cap = choose_bucket(n)
            f = self.flows[stage]
            nz = normal(stage, (n, f.cfg.out_channels))
            x, c, mask = pack(xyz, nz, cap)
            extra = pack(xyz, concat, cap)[0] if concat is not None else None
            y = np.asarray(f.run(self.params[stage], x, condition, c, mask, extra))[0, :n]
            job.metadata["padding"].append({"stage": stage, "tokens": n, "bucket": cap,
                "padding_tokens": cap - n, "waste_fraction": (cap - n) / cap})
            return _finite(y, stage)
        from .flow import glue
        shape = job.stage("S4", "device", lambda: glue.slat_denormalize(
            sample("S4", coords, cond["512"]), self.normalization["shape_slat"]))
        resolution = 512
        if self.pipeline == "1024_cascade":
            up = job.stage("S5.upsample", "device", lambda: self.shape.run(shape, coords, 32, upsample_times=4))
            coords_b = np.column_stack((np.zeros(len(up["coords"]), np.int32), up["coords"]))
            q, resolution = job.stage("S5.quantize", "host", lambda: glue.cascade_quantize(coords_b, 512, 1024))
            coords = q[:, 1:]
            shape = job.stage("S6", "device", lambda: glue.slat_denormalize(
                sample("S6", coords, cond["1024"]), self.normalization["shape_slat"]))
        else:
            job.metadata["omitted_stages"] = {"S5": "vendor 512 pipeline has no cascade", "S6": "vendor 512 pipeline"}
        tex_cond = cond["512" if self.pipeline == "512" else "1024"]
        tex = job.stage("S7", "device", lambda: glue.slat_denormalize(sample(
            "S7", coords, tex_cond, glue.slat_normalize(shape, self.normalization["shape_slat"])),
            self.normalization["tex_slat"]))
        decoded = job.stage("S8.decode", "device", lambda: self.shape.run(shape, coords, resolution // 16))

        v_head, inter, weight = job.stage("S8.head", "host",
            lambda: host_heads(m["sparse_conv_jax"], decoded["h"]))
        def mesh():
            v = v_head
            v, f = m["fdg_mesh"].fdg_to_mesh_np(decoded["coords"], v, inter, weight,
                np.array([[-.5] * 3, [.5] * 3], np.float32), grid_size=resolution, prod_dtype="float32")
            if len(v) == 0 or len(f) == 0:
                raise ValueError("shape decoder produced an empty mesh")
            from .glb_bake.meshops import fill_holes
            return fill_holes(_finite(v, "vertices"), f, 3e-2)
        vertices, faces = job.stage("S8.mesh", "host", mesh)
        guide = [logits > 0 for _, logits in decoded["subs"]]
        texture = job.stage("S9", "device", lambda: self.texture.run(tex, coords, resolution // 16, guide_subs=guide))
        if not np.array_equal(texture["coords"], decoded["coords"]):
            raise ValueError("shape and guided texture decoder coordinates differ")
        attrs = _finite(texture["h"] * np.float32(.5) + np.float32(.5), "PBR attrs")
        job.metadata["decoder_padding"] = {
            "S8": decoded["padding"], "S9": texture["padding"]}
        if self.pipeline == "1024_cascade":
            job.metadata["decoder_padding"]["S5"] = up["padding"]
        job.metadata["decoder_counts"] = {"shape": decoded["counts"], "texture": texture["counts"]}
        return {"vertices": vertices, "faces": faces, "attr_volume": attrs, "coords": texture["coords"],
                "resolution": resolution, "job": job}

    def export_glb(self, bundle, output_path):
        job = bundle["job"]
        from .glb_bake.to_glb import to_glb
        kwargs = dict(EXPORT)
        if self.wiring_texture_size is not None:
            kwargs["texture_size"] = self.wiring_texture_size
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def bake():
            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                tm, stats = to_glb(vertices=bundle["vertices"], faces=bundle["faces"],
                    attr_volume=bundle["attr_volume"], coords=bundle["coords"],
                    attr_layout={"base_color": (0, 3), "metallic": (3, 4), "roughness": (4, 5), "alpha": (5, 6)},
                    aabb=[[-.5] * 3, [.5] * 3], voxel_size=1 / bundle["resolution"],
                    return_stats=True, **kwargs)
                tm.export(str(path), file_type="glb")
            return {"vertices": len(tm.vertices), "faces": len(tm.faces), "stats": stats}
        info = job.stage("S10", "host", bake)
        metadata = job.totals()
        metadata.update(glb=info, export=kwargs, bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return {"path": str(path), "metadata": metadata}

    def generate(self, image, output_path, *, seed=42, progress=None, cancelled=None, noise=None,
                 on_device_done=None):
        bundle = self.run_stages(image, seed=seed, progress=progress, cancelled=cancelled, noise=noise)
        if on_device_done is not None:
            on_device_done()
        return self.export_glb(bundle, output_path)

    def unload(self):
        for a in jax.tree_util.tree_leaves(self.arrays()):
            if isinstance(a, jax.Array):
                a.delete()
        self.flows.clear()
        self.params.clear()
        for name in ("dino", "ss", "shape", "texture", "ss_program", "ss_compiled", "occupancy_program", "occupancy_compiled", "dino_programs", "last_job"):
            self.__dict__.pop(name, None)
        self.loaded = False
        gc.collect()
