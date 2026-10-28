"""`python3 -m mashbox run hy-nomas` and `python3 -m mashbox download hy-nomas`.

  run hy-nomas --ui                 load on the TPU v6e-4 and serve the browser app on 127.0.0.1:8790
  run hy-nomas --out out.mp4        render a clip without the browser (--world, --windows, --keys)
  run hy-nomas --ui --stub          the browser app on CPU against a stand-in engine (no model, no downloads)
  download hy-nomas                 download and check every file only
"""
import argparse
import json
import os
import subprocess
import sys
import time
from importlib import resources

import numpy as np

DEFAULTS = json.loads(resources.files(__package__).joinpath('defaults.json').read_text())


def log(s):
    print(s, file=sys.stderr, flush=True)


def parser(cmd):
    p = argparse.ArgumentParser(prog='mashbox %s hy-nomas' % cmd)
    if cmd == 'download':
        return p
    d = DEFAULTS
    p.add_argument('--ui', action='store_true', help='serve the browser app (the default when --out is not given)')
    p.add_argument('--out', help='render a clip to this .mp4 instead of serving the app')
    p.add_argument('--world', default='eagle', help='preset world id for --out or to start with --ui (see worlds.json)')
    p.add_argument('--windows', type=int, default=4, help='--out: number of 5 s windows')
    p.add_argument('--keys', default='w', help="--out: keys held per window, ';' between windows, ',' between keys (e.g. 'w;w,left;w')")
    p.add_argument('--preset', choices=sorted(d['presets']), default=d['preset'],
                   help='quality: 3 steps plus a history pass (default); live: 4 steps, history from the last step')
    p.add_argument('--seed', type=int, default=2701, help='noise seed of the first world')
    p.add_argument('--port', type=int, default=d['server']['port'], help='browser app port on 127.0.0.1 (default %(default)s)')
    p.add_argument('--decoder', choices=sorted(d['decoders']), default=d['decoder_default'],
                   help="full: Tencent's pixel decoder (default); fast: the MASHBOX Fast Decoder (ours, much faster, softer); switch live in the app")
    p.add_argument('--no-kernels', action='store_true', help='plain JAX only (slower, same outputs within bf16 rounding)')
    p.add_argument('--cache-dir', help='XLA compile cache directory (default ~/.cache/mashbox/xla)')
    p.add_argument('--stub', action='store_true', help='CPU stand-in engine: tests the app, controls and streaming without the model')
    p.add_argument('--no-precompile', action='store_true', help='compile each program on first use instead of at start')
    return p


def main(cmd, argv):
    a = parser(cmd).parse_args(argv)
    if cmd == 'download':
        from . import weights as W, upstream as U
        W.check_disk(log)
        out = W.download_all(log)
        out['tencent_code'] = U.source_dir(log)
        print(json.dumps(out, indent=1))
        return 0
    if a.stub:
        return run_stub(a)
    return run(a)


# ------------------------------------------------------------------ model
class Model:
    """Everything on the chips: text encoder (chips 0-3), main model (chips 0-3), decoder (chips 1-3)."""

    def __init__(self, a):
        import jax
        from ... import hub
        from ...cli import setup_cache
        from ... import kernels as K
        from . import chip, decoder as DEC, fast_decoder as FD, kernels as KB, ops as O, runtime as RT, text_encoder as TE, upstream as U, weights as W
        t0 = time.time()
        self.cache = setup_cache(a.cache_dir)
        devs = jax.devices()
        lay = DEFAULTS['layout']
        need = max(lay['dit_chips'] + lay['decoder_chips'] + lay['text_encoder_chips']) + 1
        fam = K.chip_family()
        if fam is None or len(devs) < need:
            raise SystemExit('hy-nomas needs a TPU v6e-4 (4 chips); found %d %s device(s). Use --stub to try the app on CPU.'
                             % (len(devs), devs[0].platform))
        self.hw = chip.chip_label(devs, chip.zone())
        log(json.dumps(dict(event='chip', label=self.hw, jax=jax.__version__)))
        m = hub.model_entry('hy-nomas')
        hub.compile_cache(m, self.cache, log)              # our compiled programs for this chip and jax, when they match
        bundle = None
        if not a.no_kernels:
            path, row = hub.kernel_bundle(m, log)
            if path is not None:
                b = KB.open_bundle(path, row['sha256'])
                bundle = b if (b.usable() and fam in row['chips']) else None
        self.ops = O.Ops(bundle)
        W.check_disk(log)
        src = U.source_dir(log)
        dit = W.fetch('tencent', 'DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors', log)
        lora = W.fetch('tencent', 'DiT/lora_weights/cinematic/adapter.safetensors', log)
        lcfg = W.fetch('tencent', 'DiT/lora_weights/cinematic/adapter_config.json', log)
        dec = W.fetch('tencent', 'VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors', log)
        enc_dir = W.text_encoder_dir(log)
        log(json.dumps(dict(event='files_ready', s=round(time.time() - t0, 1))))
        self.tencent_cam = U.camera_module(src)
        self.text = TE.TextEncoder(U.TencentText(src, enc_dir), enc_dir, [devs[i] for i in lay['text_encoder_chips']], log=log)
        t1 = time.time()
        params, merge = W.dit_params(dit, lora, lcfg)
        log(json.dumps(dict(event='dit_weights', lora_merged=merge, s=round(time.time() - t1, 1))))
        self.dit = RT.DiTRuntime([devs[i] for i in lay['dit_chips']], params, self.ops, preset=a.preset, log=log)
        del params
        self.decoder = DEC.Decoder(DEC.Weights(dec).load(), [devs[i] for i in lay['decoder_chips']], self.ops, log=log)
        self.decoder_kind, self.fast_decoder = a.decoder, None
        try:                                               # optional: the app works without it (full decoder only)
            self.fast_decoder = FD.FastDecoder(W.fetch('mashbox', 'decoders/hy-nomas-fast-decoder-v1.safetensors', log),
                                               [devs[i] for i in lay['decoder_chips']], log=log)
        except Exception as e:
            log(json.dumps(dict(event='fast_decoder_unavailable', err=repr(e)[:300])))
            if a.decoder == 'fast':
                raise
        dd = DEFAULTS['decoder']
        self.split, self.overlap, self.crossfade = tuple(dd['split_latent_frames']), dd['overlap_latent_frames'], dd['crossfade_frames']
        if not a.no_precompile:
            t2 = time.time()
            got = self.dit.compile_all(log)
            for ci in range(len(self.split)):
                for n in sorted(set(self.split) | {v + self.overlap for v in self.split}):
                    self.decoder.call(ci, np.zeros((1, 32, n, 22, 40), np.float32), 0).block_until_ready()
                if self.fast_decoder is not None:          # first part of a session, and every other part (warm context in front)
                    for n, wm in ((self.split[0], 0), (self.split[0] + self.overlap + FD.WARM, FD.WARM)):
                        self.fast_decoder.call(ci, np.zeros((1, 32, n, 22, 40), np.float32), 0, wm).block_until_ready()
            log(json.dumps(dict(event='compiled', dit_shapes=got, s=round(time.time() - t2, 1))))
        log(json.dumps(dict(event='ready', kernels=self.ops.summary(), s=round(time.time() - t0, 1))))


def run(a):
    from .server import Engine, Live
    model = Model(a)
    eng = Engine(model, log)
    if a.out:
        return render(eng, model, a)
    live = Live(eng, log, hw=model.hw, seed=a.seed, preset=a.preset, camera=DEFAULTS['camera'], tencent_cam=model.tencent_cam)
    live.start(a.port, autostart=a.world if a.world in live.worlds else None)
    log('open http://127.0.0.1:%d/ through your SSH tunnel (see docs/models/hy-nomas.md); Ctrl-C stops the server' % a.port)
    _wait()
    return 0


def run_stub(a):
    from .server import StubEngine, Live
    eng = StubEngine(log, split=tuple(DEFAULTS['decoder']['split_latent_frames']), overlap=DEFAULTS['decoder']['overlap_latent_frames'])
    if a.out:
        raise SystemExit('--stub serves the app only; --out needs the model on a TPU v6e-4')
    live = Live(eng, log, hw='STUB on CPU (no model)', seed=a.seed, preset=a.preset, camera=DEFAULTS['camera'], stub=True)
    live.start(a.port, autostart=a.world if a.world in live.worlds else None)
    log('stub: open http://127.0.0.1:%d/' % a.port)
    _wait()
    return 0


def _wait():
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return


def render(eng, model, a):
    """Generate --windows windows of --world with --keys and write them to --out (H.264 mp4, 24 fps)."""
    from .cameras import Cameras
    from .server import Texts, crossfade, ffmpeg_exe, worlds
    w = worlds()[a.world]
    tx = Texts(eng)
    items = []
    for k in ('seg1', 'seg2'):
        t, m, info, _ = tx.get(w[k])
        items.append(eng.make_item(k, t, m, info))
    eng.set_preset(a.preset)
    eng.reset(a.seed)
    cams = Cameras(model.tencent_cam, rate=DEFAULTS['camera']['turn_deg_per_s'], pitch_max=DEFAULTS['camera']['pitch_limit_deg'],
                   gain=DEFAULTS['camera']['rotation_scale'])
    keys = [k.split(',') if k else [] for k in a.keys.split(';')]
    ff = subprocess.Popen([ffmpeg_exe(), '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x704', '-r', '24',
                           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', a.out], stdin=subprocess.PIPE)
    parts = {}

    def on_part(sess, wi, j, frames, info):
        parts[(wi, j)] = frames
    hold, n_frames, t0 = None, 0, time.time()
    for wi in range(a.windows):
        blk = cams.next_window(keys[min(wi, len(keys) - 1)])
        T = eng.window(0, wi, blk, items[0 if wi == 0 else 1], on_part)
        eng.barrier(0, wi)
        for j in range(eng.nch):
            fr, hold = crossfade(parts.pop((wi, j)), hold, 4 * eng.ov)
            ff.stdin.write(np.ascontiguousarray(fr).tobytes())
            n_frames += fr.shape[0]
        log(json.dumps(dict(event='window', w=wi + 1, text=T['text'], dit_s=T['dit_s'], total_s=round(time.time() - t0, 1))))
    if hold is not None:
        ff.stdin.write(np.ascontiguousarray(hold).tobytes())
        n_frames += hold.shape[0]
    ff.stdin.close()
    ff.wait()
    log(json.dumps(dict(event='done', out=os.path.abspath(a.out), frames=n_frames, seconds=round(time.time() - t0, 1))))
    print(os.path.abspath(a.out))
    return 0
