"""`mashbox run maya1` and `mashbox download maya1`.

  run maya1 --text "Hello there." --out hello.wav             one take (or several, for long text) to a WAV file
  run maya1 --text "..." --voice onyx --instructions "Speak slowly, with a smile."
  run maya1 --text "..." --description "Realistic male voice in the 40s age with british accent. ..."
  run maya1 --batch requests.jsonl --out-dir out/              many requests at once, one WAV each
  run maya1 --voices                                           the voice names and their descriptions
  download maya1                                               download and check every file only

maya1 writes SNAC audio codes on the TPU through the mashbox engine; the SNAC decoder makes 24 kHz audio from them.
"""
import argparse
import json
import os
import sys

from . import prompt as P


def log(s):
    print(s, file=sys.stderr, flush=True)


def parser(cmd):
    p = argparse.ArgumentParser(prog=f"mashbox {cmd} maya1")
    if cmd == "download":
        return p
    p.add_argument("--text", help="the text to say; emotion tags such as <laugh> or <whisper> go inline where you want them")
    p.add_argument("--voice", help="a voice name (see --voices; default alloy)")
    p.add_argument("--description", help="your own voice description instead of a voice name")
    p.add_argument("--instructions", help="how to say it (emotion, style), added to the voice description")
    p.add_argument("--out", help="output file (.wav, or .pcm for raw 16-bit 24 kHz mono)")
    p.add_argument("--batch", help="a JSON Lines file, one request per line: text, out, and optionally voice, description, "
                                   "instructions, seed and the sampling settings below")
    p.add_argument("--out-dir", help="with --batch: where outputs without a path of their own go")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling and the decoder's noise (default %(default)s)")
    p.add_argument("--temperature", type=float, help="default 0.4; 0 = greedy")
    p.add_argument("--top-p", type=float, help="default 0.9")
    p.add_argument("--top-k", type=int, help="default 50")
    p.add_argument("--repetition-penalty", type=float, help="default 1.1")
    p.add_argument("--max-tokens", type=int, help="new tokens per take, at most 2048 (about 25 s of audio; default 2048)")
    p.add_argument("--slots", type=int, default=8, help="takes decoded at once (default %(default)s)")
    p.add_argument("--ctx", type=int, default=3072, help="tokens per slot: prompt plus new tokens (default %(default)s)")
    p.add_argument("--codec", choices=("f32", "bf16"), default="f32", help="SNAC decoder precision (default %(default)s)")
    p.add_argument("--no-kernels", action="store_true", help="for checks only: skip the kernel package and the bfloat16 path (every weight on the exact reference path; much slower, different audio)")
    p.add_argument("--cache-dir", help="XLA compile cache directory (default ~/.cache/mashbox/xla/maya1)")
    p.add_argument("--report", help="also write what ran (takes, tokens, timings) to this JSON file")
    p.add_argument("--voices", action="store_true", help="list the voice names and their descriptions, then exit")
    return p


def requests_of(a):
    """-> [(request dict, output path)] from --text/--out or --batch, checked before anything is downloaded."""
    knobs = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, repetition_penalty=a.repetition_penalty,
                 max_tokens=a.max_tokens)
    if a.batch:
        if a.text:
            raise SystemExit("give --text or --batch, not both")
        out = []
        for i, line in enumerate(open(a.batch, encoding="utf-8")):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{a.batch} line {i + 1}: not JSON ({e})")
            if not isinstance(r, dict):
                raise SystemExit(f"{a.batch} line {i + 1}: a request is a JSON object")
            path = r.pop("out", None) or os.path.join(a.out_dir or ".", f"maya1-{len(out):04d}.wav")
            req = {k: v for k, v in knobs.items() if v is not None}
            req["seed"] = a.seed
            req.update(r)
            out.append((req, path))
    else:
        if not a.text:
            raise SystemExit("--text is required (or --batch)")
        if not a.out:
            raise SystemExit("--out is required: the .wav (or .pcm) file to write")
        out = [(dict(text=a.text, voice=a.voice, description=a.description, instructions=a.instructions, seed=a.seed,
                     **knobs), a.out)]
    from . import speech as SP
    for req, path in out:
        if not path.lower().endswith((".wav", ".pcm")):
            raise SystemExit(f"{path}: the output is a .wav or .pcm file")
        try:
            SP.description_of(req.get("voice"), req.get("description"), req.get("instructions"))
            SP.sampling(req.get("seed") or 0, **{k: req.get(k) for k in knobs})
            if not isinstance(req.get("text"), str) or not req["text"].strip():
                raise P.InputError("input", "text must be a non-empty string")
            if len(req["text"]) > P.MAX_INPUT_CHARS:
                raise P.InputError("input", f"text is longer than {P.MAX_INPUT_CHARS} characters")
        except P.InputError as e:
            raise SystemExit(f"{e.param}: {e}")
    return out


def main(cmd, argv):
    a = parser(cmd).parse_args(argv)
    from . import speech as SP, weights as W
    if cmd == "download":
        files = W.stage(log)
        print(json.dumps(files, indent=1))
        return 0
    if a.voices:
        print(json.dumps(SP.voices()["voices"], indent=1))
        return 0
    reqs = requests_of(a)
    files = W.stage(log)
    from ... import cli
    cli.setup_cache(a.cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "mashbox", "xla", "maya1"))
    m = SP.Maya1.load(files, slots=a.slots, ctx=a.ctx, codec=a.codec, use_kernels=not a.no_kernels, log=log)
    try:
        res = m.speak([r for r, _ in reqs])
    except P.InputError as e:
        raise SystemExit(f"{e.param}: {e}")
    summary = []
    for (req, path), r in zip(reqs, res):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(SP.to_pcm(r["audio"]) if path.lower().endswith(".pcm") else SP.wav_bytes(r["audio"]))
        summary.append(dict(out=path, seconds=round(r["seconds"], 3), takes=[{k: v for k, v in t.items() if k != "ids"}
                                                                            for t in r["takes"]]))
    rep = dict(outputs=summary, wall_s=round(res[0]["wall_s"], 3) if res else 0, load=m.info)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=1)
    print(json.dumps(rep))
    return 0
