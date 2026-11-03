"""maya1 text to speech on the mashbox engine.

maya1 is a Llama 3.2 3B shaped model: the GGUF file runs through the engine's own llama architecture
(mashbox.gguf.archs.dense) on the engine's continuous-batching scheduler (mashbox.runtime.engine.Engine), so many takes
decode together in one batch. The engine is given token ids built here with maya1's own tokenizer (prompt.py), and
returns the SNAC code tokens maya1 writes; the SNAC decoder (snac.py) turns them into 24 kHz audio.

  Maya1.load(files, ...)        files = weights.stage(): {"gguf", "snac", "tokenizer"} paths
  m.speak(requests)             -> one result per request: 24 kHz float32 audio and what ran
  write_wav(path, audio)        16-bit PCM WAV, mono 24 kHz; to_pcm(audio) = the raw s16le bytes

A request: text, plus voice (a name from voices.json) or description (any voice description), and optional
instructions (appended to the description), seed and sampling settings. Text longer than one take (prompt.SEG_CHARS) is
cut at sentence ends into takes; take k of a request uses seed + k. With on_audio, every take is decoded while it is
generated (SNAC streaming: the same length as the whole-take decode, within 1e-6 per sample) and on_audio(request index, take index, chunk) gets
each chunk as soon as it is ready.
"""
import json
import struct
import time
from importlib import resources

import numpy as np

from . import prompt as P
from . import snac as S

SR = S.SR
# maya1's generation settings: Maya Research's example (temperature 0.4, top_p 0.9, repetition_penalty 1.1, at most 2048
# new tokens, stop at 128258) with transformers' default top_k 50, which that example runs with
DEFAULTS = dict(temperature=0.4, top_p=0.9, top_k=50, repetition_penalty=1.1, max_tokens=2048)
MAX_TOKENS = 2048
STOP = (P.EOS_SPEECH,)


def voices():
    return json.loads(resources.files(__package__).joinpath("voices.json").read_text())


def description_of(voice=None, description=None, instructions=None):
    """-> the voice description maya1 gets: description as given, else the named voice's (the default voice when neither
    is given), with instructions appended. An unknown voice name raises InputError naming the known ones."""
    if description is not None and voice is not None:
        raise P.InputError("voice", "give a voice name or a description, not both")
    if description is None:
        v = voices()
        name = voice or v["default"]
        if name not in v["voices"]:
            raise P.InputError("voice", f"unknown voice {name!r}; voices: {', '.join(v['voices'])} (or give a description)")
        description = v["voices"][name]
    return P.compose_description(description, instructions)


def sampling(seed=0, **kw):
    """-> the engine's sampling settings for one take: DEFAULTS with kw on top. temperature 0 = greedy."""
    s = dict(DEFAULTS)
    s.update({k: v for k, v in kw.items() if v is not None})
    mx = int(s.pop("max_tokens"))
    if not 1 <= mx <= MAX_TOKENS:
        raise P.InputError("max_tokens", f"max_tokens must be 1..{MAX_TOKENS}")
    return dict(temperature=float(s["temperature"]), top_p=float(s["top_p"]), top_k=int(s["top_k"]), min_p=0.0,
                repetition_penalty=float(s["repetition_penalty"]), seed=int(seed) & 0x7FFFFFFFFFFFFFFF), mx


class Maya1:
    def __init__(self, eng, tok, snac_params, codec="f32", info=None):
        self.eng, self.tok, self.p, self.codec, self.info = eng, tok, snac_params, codec, info or {}

    @classmethod
    def load(cls, files, slots=8, ctx=3072, steps=8, codec="f32", act=None, use_kernels=True, check_tokenizer=True,
             log=print):
        """files: {"gguf", "snac", "tokenizer"} paths. slots: takes decoded at once. ctx: tokens per slot (prompt + the
        take's new tokens). steps: decode steps per device call."""
        import jax.numpy as jnp
        from ... import gguf as G, ops
        from ...runtime.engine import Engine
        from . import weights as W
        tok_pin = next(f["sha256"] for f in W.pins()["repos"]["tokenizer"]["files"]) if check_tokenizer else None
        tok = P.Tokenizer(files["tokenizer"], pin=tok_pin)
        f, A, c = G.open_model(files["gguf"], act=act or jnp.bfloat16)
        if f.arch != "llama" or c.V != P.VOCAB:
            raise SystemExit(f"{files['gguf']}: a {f.arch} file with {c.V} tokens; maya1 is llama with {P.VOCAB}")
        if use_kernels:
            ops.load_kernels(log)
        t0 = time.perf_counter()
        Wt, info = A.load(f, c, use_kernels=use_kernels, log=lambda i: log(json.dumps(dict(event="weights", **i))))
        eng = Engine(c, Wt, A, CTX=ctx, S=slots, K=steps)
        params = S.load_params(files["snac"], codec)
        info = dict(load_s=round(time.perf_counter() - t0, 2), slots=slots, ctx=ctx, steps=steps, codec=codec,
                    device_gb=round(info.get("device_bytes", 0) / 1e9, 2))
        log(json.dumps(dict(event="loaded", **info)))
        return cls(eng, tok, params, codec, info)

    # ------------------------------------------------------------------------------------------------------------ plan
    def plan(self, req):
        """One request -> its takes: [(text, prompt ids, sampling, max_new)]."""
        desc = description_of(req.get("voice"), req.get("description"), req.get("instructions"))
        text = req.get("text")
        if not isinstance(text, str) or not text.strip():
            raise P.InputError("input", "input must be a non-empty string")
        if len(text) > P.MAX_INPUT_CHARS:
            raise P.InputError("input", f"input is longer than {P.MAX_INPUT_CHARS} characters")
        seed = int(req.get("seed") or 0)
        kw = {k: req.get(k) for k in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")}
        out = []
        for k, seg in enumerate(P.segment_text(text)):
            ids = self.tok.build(desc, seg)
            sp, mx = sampling(seed + k, **kw)
            if len(ids) + mx > self.eng.CTX:
                raise P.InputError("input", f"a take needs {len(ids) + mx} tokens; this load holds {self.eng.CTX} per take")
            out.append((seg, ids, sp, mx))
        return out

    # ----------------------------------------------------------------------------------------------------------- speak
    def speak(self, requests, on_audio=None, first_chunk_frames=2, chunk_frames=8):
        """requests: list of request dicts. -> one dict per request: audio (float32, 24 kHz; None with on_audio, whose
        pieces are the audio), seconds, takes, and the timings. Every take of every request goes through one engine: each joins as soon as a slot is free."""
        plans = [self.plan(r) for r in requests]
        jobs = [(ri, k, *t) for ri, pl in enumerate(plans) for k, t in enumerate(pl)]
        t0 = time.perf_counter()
        eng = self.eng
        live, queue, done = {}, list(range(len(jobs))), {}
        first_audio = {}
        while queue or live:
            n = min(eng.free_slots(), len(queue))
            if n:
                take = [queue.pop(0) for _ in range(n)]
                rows = eng.admit([dict(ids=jobs[j][3], max_new=jobs[j][5], samp=jobs[j][4]) for j in take], STOP)
                for j, r in zip(take, rows):
                    if r["fin"] == "error":
                        raise P.InputError("input", r.get("error") or "the engine refused this take")
                    st = None
                    if on_audio is not None:
                        st = S.SnacStream(self.p, S.KeyNoise(jobs[j][4]["seed"]), self.codec, chunk_frames=chunk_frames,
                                          first_chunk_frames=first_chunk_frames)
                    live[j] = dict(row=r, seen=0, stream=st)
            eng.step()
            for j in list(live):
                L = live[j]
                r = L["row"]
                new = r["ids"][L["seen"]:]
                L["seen"] = len(r["ids"])
                if L["stream"] is not None and new:
                    for c in L["stream"].push(new):
                        first_audio.setdefault(jobs[j][0], time.perf_counter() - t0)
                        on_audio(jobs[j][0], jobs[j][1], c)
                if r["fin"] is not None:
                    if L["stream"] is not None:
                        for c in L["stream"].finish():
                            first_audio.setdefault(jobs[j][0], time.perf_counter() - t0)
                            on_audio(jobs[j][0], jobs[j][1], c)
                    done[j] = dict(ids=list(r["ids"]), fin=r["fin"], t_first=r["t_first"], t_end=r.get("t_end"),
                                   t_admit=r["t_admit"])
                    del live[j]
        t_lm = time.perf_counter() - t0
        out = []
        for ri, req in enumerate(requests):
            takes, audio = [], []
            for j in [j for j in range(len(jobs)) if jobs[j][0] == ri]:
                _, k, seg, ids, sp, mx = jobs[j]
                d = done[j]
                q = S.frames_from_tokens(d["ids"])
                if on_audio is None:
                    y = S.decode_frames(self.p, q, S.KeyNoise(sp["seed"]), self.codec)
                    audio.append(y)
                takes.append(dict(chars=len(seg), prompt_tokens=len(ids), new_tokens=len(d["ids"]), frames=int(q.shape[0]),
                                  finish=d["fin"], seed=sp["seed"], ids=d["ids"]))
            y = np.concatenate(audio) if audio else np.zeros(0, np.float32)
            n = sum(t["frames"] for t in takes) * S.FRAME
            out.append(dict(audio=y if on_audio is None else None, seconds=n / SR, takes=takes,
                            first_audio_s=first_audio.get(ri), lm_s=t_lm, wall_s=time.perf_counter() - t0))
        return out


def to_pcm(y):
    """float audio -> s16le bytes: clip to [-1, 1], then round(x * 32767)."""
    y = np.clip(np.asarray(y, np.float32).reshape(-1), -1.0, 1.0)
    return np.rint(y * 32767.0).astype("<i2").tobytes()


def wav_bytes(y):
    pcm = to_pcm(y)
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm)


def write_wav(path, y):
    with open(path, "wb") as f:
        f.write(wav_bytes(y))
