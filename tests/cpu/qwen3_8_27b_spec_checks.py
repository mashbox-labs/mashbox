"""CPU checks of speculative decoding (models/qwen3_8_27b/spec.py) on tiny random models of the Qwen3.8-27B architecture.

  JAX_PLATFORMS=cpu python3 tests/cpu/qwen3_8_27b_spec_checks.py <workdir>

  S1 greedy with speculation, K = 1, 2, 3, batch of 4 and each prompt alone, on a random model (drafts mostly rejected)
     and a near-bigram model (drafts mostly accepted, so commits of 1..K+1 tokens all run): the tie-aware replay (the
     drafter gate's rule): every committed token, teacher-forced through the plain forward, is its argmax or within
     one bf16 step of its top logit; near-tie counts recorded. Equality with greedy without speculation (equal_batch,
     equal_alone) is recorded as information, not gated (a bf16 tie can round differently at the verify width)
  S2 rejection sampling keeps the target distribution: 200,000 rows, V = 32, K = 2, temperature 0.8 and top_k 12;
     total variation distance of the first new token's histogram to the target distribution < 0.01
  S3 a sampled batch with speculation runs and its seeded rows repeat
Needs: jax (CPU), numpy, ml_dtypes. Exit 0 = all pass.
"""
import json
import os
import struct
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                     # noqa: E402
import jax                             # noqa: E402
import jax.numpy as jnp                # noqa: E402
import ml_dtypes                       # noqa: E402

from mashbox.models import qwen3_8_27b as Q                      # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW, spec as QS  # noqa: E402
from mashbox.runtime.engine import Engine                       # noqa: E402
from mashbox.runtime import sampler as SP                       # noqa: E402

WD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mashbox-spec-test"
os.makedirs(WD, exist_ok=True)
bf = ml_dtypes.bfloat16
D, F, V, L, H, KV, HD, HK, HV, DK, DV, CK = 64, 96, 512, 8, 4, 2, 32, 2, 6, 16, 16, 4


def write_bshj(path, tensors):
    align, pos, ents, bufs = 4096, 0, {}, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {"bfloat16": "bf16", "float32": "f32", "int8": "i8"}[a.dtype.name]
        pos = -(-pos // align) * align
        ents[name] = dict(dtype=dt, shape=list(a.shape), offset=pos, nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    raw = json.dumps(dict(format="bshj", version="1.0", data_len=pos, tensors=ents), separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p, b in bufs:
            f.seek(off + p)
            f.write(b)
        f.truncate(off + pos)


def model(seed, layer_out):
    """random weights; layer_out scales every residual write (o, out, down): small = close to a bigram model"""
    rng = np.random.default_rng(seed)
    r = lambda *s, sd=0.08: (rng.normal(0, sd, s)).astype(bf)
    T = {"model.embed_tokens.weight": r(V, D, sd=1.0), "model.norm.weight": r(D, sd=0.1), "lm_head.weight": r(V, D, sd=0.3)}
    for i in range(L):
        p = f"model.layers.{i}."
        T[p + "input_layernorm.weight"], T[p + "post_attention_layernorm.weight"] = r(D, sd=0.3), r(D, sd=0.3)
        T[p + "mlp.gate_proj.weight"], T[p + "mlp.up_proj.weight"] = r(F, D), r(F, D)
        T[p + "mlp.down_proj.weight"] = r(D, F, sd=0.08 * layer_out)
        if (i + 1) % 4:
            a = p + "linear_attn."
            T[a + "in_proj_qkv.weight"], T[a + "in_proj_z.weight"] = r(2 * HK * DK + HV * DV, D), r(HV * DV, D)
            T[a + "in_proj_a.weight"], T[a + "in_proj_b.weight"] = r(HV, D), r(HV, D)
            T[a + "out_proj.weight"] = r(D, HV * DV, sd=0.08 * layer_out)
            T[a + "conv1d.weight"] = r(2 * HK * DK + HV * DV, 1, CK, sd=0.4)
            T[a + "A_log"] = np.log(rng.uniform(1, 16, HV)).astype(np.float32)
            T[a + "dt_bias"] = r(HV, sd=1.0)
            T[a + "norm.weight"] = r(DV, sd=0.3)
        else:
            a = p + "self_attn."
            T[a + "q_proj.weight"], T[a + "k_proj.weight"], T[a + "v_proj.weight"] = r(2 * H * HD, D), r(KV * HD, D), r(KV * HD, D)
            T[a + "o_proj.weight"] = r(D, H * HD, sd=0.08 * layer_out)
            T[a + "q_norm.weight"], T[a + "k_norm.weight"] = r(HD, sd=0.3), r(HD, sd=0.3)
    m = "mtp.layers.0."
    fc = r(D, 2 * D)
    if layer_out < 1:          # near-bigram: the MTP head passes the next token's embedding through
        fc = np.concatenate([np.eye(D), np.zeros((D, D))], 1).astype(bf)
    T.update({"mtp.fc.weight": fc, "mtp.pre_fc_norm_embedding.weight": r(D, sd=0.1), "mtp.pre_fc_norm_hidden.weight": r(D, sd=0.1),
              "mtp.norm.weight": T["model.norm.weight"], m + "input_layernorm.weight": r(D, sd=0.3),
              m + "post_attention_layernorm.weight": r(D, sd=0.3), m + "self_attn.q_proj.weight": r(2 * H * HD, D),
              m + "self_attn.k_proj.weight": r(KV * HD, D), m + "self_attn.v_proj.weight": r(KV * HD, D),
              m + "self_attn.o_proj.weight": r(D, H * HD, sd=0.08 * layer_out), m + "self_attn.q_norm.weight": r(HD, sd=0.3),
              m + "self_attn.k_norm.weight": r(HD, sd=0.3), m + "mlp.gate_proj.weight": r(F, D), m + "mlp.up_proj.weight": r(F, D),
              m + "mlp.down_proj.weight": r(D, F, sd=0.08 * layer_out)})
    return T


cfgd = os.path.join(WD, "cfg")
os.makedirs(cfgd, exist_ok=True)
json.dump(dict(model_type="qwen3_5_text", vocab_size=V, hidden_size=D, intermediate_size=F, num_hidden_layers=L, num_attention_heads=H,
               num_key_value_heads=KV, head_dim=HD, linear_num_key_heads=HK, linear_num_value_heads=HV, linear_key_head_dim=DK,
               linear_value_head_dim=DV, linear_conv_kernel_dim=CK, full_attention_interval=4, rms_norm_eps=1e-6,
               rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25), tie_word_embeddings=False,
               attn_output_gate=True, mtp_num_hidden_layers=1), open(os.path.join(cfgd, "config.json"), "w"))
cfg = Q.Config(cfgd)
R, ok = {}, True
rng = np.random.default_rng(5)
prompts = [rng.integers(1, V, n).tolist() for n in (9, 30, 17, 70)]

for tag, lo in (("random", 1.0), ("bigram", 0.05)):
    T = model(11, lo)
    path = os.path.join(WD, f"{tag}.bshj")
    write_bshj(path, T)
    W, _ = QW.load(path, cfg)
    Wm, _ = QS.load_mtp(path, cfg)
    base = Engine(cfg, W, Q, CTX=256, S=4, K=4)
    want = base.generate(prompts, max_new=40)["ids"]
    alone = [base.generate([p], max_new=40)["ids"][0] for p in prompts]
    R[f"S1_{tag}_batch_eq_alone"] = want == alone
    for k in (1, 2, 3):
        e = Engine(cfg, W, Q, CTX=256, S=4, K=4, mtp=(QS, Wm), draft_k=k, spec_rounds=3)
        got = e.generate(prompts, max_new=40)["ids"]
        got1 = [e.generate([p], max_new=40)["ids"][0] for p in prompts]
        h = e.spec_stats["hist"]
        near = over = 0
        for p, y in list(zip(prompts, got)) + list(zip(prompts, got1)):
            lg = np.asarray(base.logits(p + y)[len(p) - 1:len(p) - 1 + len(y)], np.float32)
            top, mine = lg.max(-1), lg[np.arange(len(y)), y]
            near += int(np.sum(lg.argmax(-1) != np.asarray(y)))
            over += int(np.sum(top - mine > 2.0 ** (np.floor(np.log2(np.maximum(np.abs(top), 1e-30))) - 7)))
        lens_ok = all(len(y) == 40 for y in got + got1)
        R[f"S1_{tag}_K{k}"] = dict(replay_near_tie=near, replay_beyond_one_bf16_step=over, lens_ok=lens_ok,
                                   equal_batch=got == want, equal_alone=got1 == want,
                                   tokens_per_round=round(e.spec_stats["tokens"] / max(1, e.spec_stats["rounds"]), 3), hist=h)
        ok &= over == 0 and lens_ok
        if got != want:
            for i, (a, b) in enumerate(zip(want, got)):
                if a != b:
                    fd = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    print(tag, k, "prompt", i, "first diff at", fd, a[fd:fd + 5], b[fd:fd + 5])
    if tag == "bigram":
        ok &= R["S1_bigram_K3"]["tokens_per_round"] > 2.0     # the near-bigram head must actually get drafts accepted

# S2: rejection sampling keeps the target distribution
Vs, K, N = 32, 2, 200000
k1, k2 = jax.random.split(jax.random.PRNGKey(3))
tlg = jax.random.normal(k1, (1, K + 1, Vs)) * 2.0
dlg = tlg[:, :K] + jax.random.normal(k2, (1, K, Vs)) * 1.5          # a draft that is close, not equal
tl, dl = jnp.broadcast_to(tlg, (N, K + 1, Vs)), jnp.broadcast_to(dlg, (N, K, Vs))
sp = SP.to_device(SP.pack([dict(temperature=0.8, top_k=12, seed=i + 1) for i in range(N)], N))
step0 = jnp.zeros((N,), jnp.int32)
d = jnp.stack([QS._gumbel_pick(jnp.log(QS.probs(dl[:, k], sp)), sp["seed"], step0 + k, 7 + k) for k in range(K)], 1)
a, bonus = jax.jit(QS.accept_sampled)(tl, dl, d, sp, step0)
first = np.asarray(jnp.where(a > 0, d[:, 0], bonus))
p_true = np.asarray(QS.probs(tlg[:, 0], {k_: v_[:1] for k_, v_ in sp.items()}))[0]
hist = np.bincount(first, minlength=Vs) / N
tv = 0.5 * float(np.abs(hist - p_true).sum())
R["S2"] = dict(tv=round(tv, 5), mean_accepted=float(np.asarray(a).mean()))
ok &= tv < 0.01

# S3: sampled batch with speculation, seeded rows repeat
e = Engine(cfg, W, Q, CTX=256, S=4, K=4, mtp=(QS, Wm), draft_k=2, spec_rounds=3)
samp = [dict(temperature=0.8, top_p=0.9, seed=7), dict(temperature=0.7, top_k=20, seed=8), None, dict(temperature=1.0, seed=9)]
r1 = e.generate(prompts, max_new=30, samp=samp)["ids"]
r2 = e.generate(prompts, max_new=30, samp=samp)["ids"]
R["S3"] = dict(repeat=r1 == r2, greedy_row_equal=r1[2] == want[2][:30], lens=[len(x) for x in r1])
ok &= r1 == r2 and r1[2] == want[2][:30]
R["pass"] = bool(ok)
print(json.dumps(R, indent=1))
sys.exit(0 if ok else 1)
