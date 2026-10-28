"""On-chip check of the Hy-Nomas kernel bundle: every compiled kernel against the plain JAX version of the same op.

    python3 tests/tpu/hy_nomas_kernel_check.py [--bundle FILE.whl] [--out result.json]

Without --bundle it downloads the bundle pinned in models.json and checks its sha256. Each kernel runs on one chip with
inputs of the shape it was compiled for, built the way the model builds them (zero padded rows, group ids for the text
masks). For each kernel the report gives: bit-exact or not, max abs error, relative L2 error vs plain JAX, and the median
time of both. The main model's attention is three programs that only run together (value prep, a check that picks the
fast or the exact program, and the attention itself), so it is compared as a whole, once through each attention program.
Runs on any TPU that loads the bundle (v6e, v5e); it does not need the model weights.
"""
import argparse
import json
import re
import statistics
import sys
import time

import numpy as np


def stats(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    nb = float(np.linalg.norm(b.ravel()))
    return dict(bit_exact=bool(np.array_equal(a, b)), max_abs=float(d.max()), rel_l2=float(np.linalg.norm((a - b).ravel()) / max(nb, 1e-30)),
                nonfinite=int((~np.isfinite(a)).sum()), ref_max_abs=float(np.abs(b).max()))


def timed(f, *args, reps=5):
    import jax
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    return out, round(statistics.median(ts), 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', help='local kernel bundle .whl (default: download the one pinned in models.json)')
    ap.add_argument('--out', help='write the report here as JSON')
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    from mashbox import hub, kernels as K
    from mashbox.models.hy_nomas import buckets as B, decoder as DEC, kernels as KB, ops as O

    jax.config.update('jax_enable_compilation_cache', False)
    m = hub.model_entry('hy-nomas')
    ver = K.versions()
    row = next((r for r in m['kernels'] if r['jax'] == ver['jax']), None)
    if row is None:
        raise SystemExit('no kernel bundle in models.json for jax %s' % ver['jax'])
    path = a.bundle or hub.fetch(m['hf_repo'], row['file'], m['hf_revision'])
    bundle = KB.open_bundle(path, row['sha256'])
    dev = jax.devices()[0]
    rep = dict(utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), device_kind=dev.device_kind, devices=len(jax.devices()),
               chip=K.chip_family(), bundle=path.rsplit('/', 1)[-1], bundle_sha256=row['sha256'], bundle_chips=row['chips'],
               usable=bundle.usable(), kernels=len(bundle.index['files']), **ver)
    try:
        st = dev.memory_stats() or {}
        rep['hbm_gib'] = round(st.get('bytes_limit', 0) / 2**30, 2)
    except Exception:
        pass
    print(json.dumps(rep), flush=True)
    rng = np.random.default_rng(0)
    BF, F32 = jnp.bfloat16, jnp.float32
    put = lambda x: jax.device_put(x, dev)
    results, done = [], set()

    def record(name, keys, s, k_ms, p_ms, extra=None):
        r = dict(name=name, kernels=keys, kernel_ms=k_ms, plain_ms=p_ms, **s, **(extra or {}))
        results.append(r)
        done.update(keys)
        print(json.dumps(r), flush=True)

    def rope_tables(L, D=128):
        ang = rng.uniform(0, 2 * np.pi, (L, D // 2)).astype(np.float32)
        return np.repeat(np.cos(ang), 2, -1), np.repeat(np.sin(ang), 2, -1)

    files = bundle.index['files']
    # ---- main model q/k norm + rotary
    H, D, N_IMG = 4, 128, 26400
    for f in [f for f in files if f['op'] == 'dit_qk_norm_rope']:
        which, rows = re.match(r'([qk])_r(\d+)', f['shape_key']).groups()
        L = f['in_avals'][0][0][0]
        x = put(jnp.asarray(rng.standard_normal((L, H * D), np.float32) * 2.0, BF))
        w = put(jnp.asarray(1.0 + 0.1 * rng.standard_normal(D), F32))
        c, s_ = (put(jnp.asarray(t)) for t in rope_tables(L))
        fn = bundle.get(f['op'], f['shape_key'])
        scale = O.q_scale(D) if which == 'q' else None
        kern = jax.jit(fn)
        plain = jax.jit(lambda x, w, c, s_, scale=scale, rows=int(rows): O.qk_norm_rope(x, w, c, s_, H=H, scale=scale, out_rows=rows))
        ok, k_ms = timed(kern, x, w, c, s_)
        op_, p_ms = timed(plain, x, w, c, s_)
        # rows L.. are left for the caller, which overwrites every one of them (text, history, zeros): compare rows :L only
        record('dit_qk_norm_rope ' + f['shape_key'], ['dit_qk_norm_rope ' + f['shape_key']], stats(np.asarray(ok)[:L], np.asarray(op_)[:L]),
               k_ms, p_ms, dict(rows_compared=L, rows_left_to_caller=int(rows) - L))

    # ---- main model attention (value prep + check + fast / exact program), per compiled shape
    HIST_IMG, FRAME_TOK = 15840, 880
    for f in [f for f in files if f['op'] == 'dit_attention_masked']:
        LQ, LK, nvalid = map(int, re.match(r'q(\d+)_k(\d+)_n(\d+)', f['shape_key']).groups())
        rest = nvalid - N_IMG
        if rest in B.BUCKETS:
            tb, hist, pb = rest, 0, 0
        else:
            rest -= HIST_IMG
            tb, pb = {512: (256, 256), 768: (512, 256), 1024: (512, 512)}[rest]
            hist = HIST_IMG
        n_text, offs = min(200, tb), [120, 160, 200]
        frame_act = [-1] * 10 + [0] * 10 + [1] * 10
        n_prev = min(180, pb) if hist else 0
        gq, gk = B.groups(N_IMG, FRAME_TOK, frame_act, n_text, offs, tb, LQ, LK, hist_img=hist, n_prev=n_prev,
                          prev_keep=min(100, n_prev), prev_bucket=pb)
        real_q = np.zeros(LQ, bool)
        real_q[:N_IMG + n_text] = True
        real_k = np.zeros(LK, bool)
        real_k[:N_IMG + n_text] = True
        if hist:
            real_k[N_IMG + tb:N_IMG + tb + hist + n_prev] = True
        q = rng.standard_normal((LQ, H * D), np.float32) * O.q_scale(D) * real_q[:, None]
        k = rng.standard_normal((LK, H * D), np.float32) * real_k[:, None]
        v = rng.standard_normal((LK, H * D), np.float32) * real_k[:, None]
        q, k, v = (put(jnp.asarray(t, BF)) for t in (q, k, v))
        gq, gk = put(jnp.asarray(gq)), put(jnp.asarray(gk))
        prep = bundle.get('dit_v_prep', 'r%d' % LK)
        check = bundle.get('dit_attention_masked_pad', f['shape_key'])
        fast = bundle.get('dit_attention_masked', f['shape_key'])
        long_ = bundle.get('dit_attention_masked_long', f['shape_key'])
        plain = jax.jit(lambda q, k, v, gq, gk, nvalid=nvalid: O.masked_attention(q, k, v, gq, gk, H=H, nvalid=nvalid, bq=1024, bkv=7168))
        ref, p_ms = timed(plain, q, k, v, gq, gk)
        _, g = jax.jit(check)(q, k)
        g = float(g)
        kf = jax.jit(lambda q, k, v, gq, gk: (lambda v1: fast(q, k, v1, check(q, k)[0], gq, gk))(prep(v)))
        kl = jax.jit(lambda q, k, v, gq, gk: long_(q, k, prep(v), gq, gk))
        of, f_ms = timed(kf, q, k, v, gq, gk)
        ol, l_ms = timed(kl, q, k, v, gq, gk)
        names = ['dit_v_prep r%d' % LK, 'dit_attention_masked_pad ' + f['shape_key']]
        real = np.asarray(real_q)
        sf = stats(np.asarray(of)[real], np.asarray(ref)[real])
        sl = stats(np.asarray(ol)[real], np.asarray(ref)[real])
        ex = dict(text_bucket=tb, hist_img=hist, prev_bucket=pb, check_value=round(g, 3), check_limit=O.CHECK_LIMIT,
                  dispatch=('exact' if g > O.CHECK_LIMIT else 'fast'), rows_compared=int(real.sum()))
        record('dit_attention fast ' + f['shape_key'], names + ['dit_attention_masked ' + f['shape_key']], sf, f_ms, p_ms, ex)
        record('dit_attention exact ' + f['shape_key'], names + ['dit_attention_masked_long ' + f['shape_key']], sl, l_ms, p_ms, ex)

    # ---- pixel decoder: q/k norm + rotary, local attention (frames 10 and 11)
    arch = DEC.ARCH
    lays = {}
    for f in [f for f in files if f['op'].startswith('dec_')]:
        t = int(re.search(r't(\d+)', f['shape_key']).group(1))
        if t not in lays:
            lays[t] = DEC.layout((t, 44, 80), arch)
        lay = lays[t]
        n = lay['n']
        fn = bundle.get(f['op'], f['shape_key'])
        if f['op'] == 'dec_qk_norm_rope':
            hh = arch.heads if f['shape_key'].startswith('q') else arch.kv_heads
            x = put(jnp.asarray(rng.standard_normal((n, hh * 128), np.float32) * 3.0))
            wt = put(jnp.asarray(1.0 + 0.1 * rng.standard_normal(64), F32))
            wh = put(jnp.asarray(1.0 + 0.1 * rng.standard_normal(64), F32))
            c = put(jnp.asarray(lay['cos'].reshape(n, 128)))
            s_ = put(jnp.asarray(lay['sin'].reshape(n, 128)))
            kern = jax.jit(fn)
            plain = jax.jit(lambda x, wt, wh, c, s_, hh=hh: O.dec_qk_norm_rope(x, wt, wh, c, s_, H=hh))
            ok, k_ms = timed(kern, x, wt, wh, c, s_)
            op_, p_ms = timed(plain, x, wt, wh, c, s_)
        else:
            q = put(jnp.asarray(rng.standard_normal((n, arch.heads, 128), np.float32), BF))
            k = put(jnp.asarray(rng.standard_normal((n, arch.kv_heads, 128), np.float32), BF))
            v = put(jnp.asarray(rng.standard_normal((n, arch.kv_heads, 128), np.float32), BF))
            ld = {kk: put(jnp.asarray(lay[kk])) for kk in ('qi', 'qg', 'ki', 'kv')}
            kern = jax.jit(fn)
            plain = jax.jit(lambda q, k, v, ld: O.dec_local_attention(q, k, v, ld, heads=arch.heads, kv_heads=arch.kv_heads,
                                                                      head_dim=arch.head_dim))
            ok, k_ms = timed(kern, q, k, v)
            op_, p_ms = timed(plain, q, k, v, ld)
        record(f['op'] + ' ' + f['shape_key'], [f['op'] + ' ' + f['shape_key']], stats(ok, op_), k_ms, p_ms)

    every = {f['op'] + ' ' + f['shape_key'] for f in files}
    rep.update(results=results, kernels_checked=len(done & every), kernels_not_checked=sorted(every - done),
               utc_end=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(rep, fh, indent=1)
    print(json.dumps(dict(kernels=rep['kernels'], checked=rep['kernels_checked'], not_checked=rep['kernels_not_checked'],
                          bit_exact=sum(r['bit_exact'] for r in results), compared=len(results),
                          worst_rel_l2=max(r['rel_l2'] for r in results))), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
