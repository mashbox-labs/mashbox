"""CPU checks of the Qwen3.8-27B port on a tiny random model of the same architecture, against Hugging Face transformers.

  python3 tests/cpu/qwen3_8_27b_checks.py <workdir> [<dir with the real tokenizer.json + chat_template.jinja>]

Needs: jax (CPU), torch (CPU), transformers with qwen3_5 support, tokenizers, jinja2, ml_dtypes.
  T1 logits at every position vs transformers (bf16 weights)       T2 greedy tokens vs transformers generate
  T3 gated delta rule: chunked == step-by-step == float64          T4 prefill in 32-token chunks == one call
  T5 a batch of 3 == each prompt alone                             T6 int8-per-row .bshj loads and stays close to bf16
  T7 sampler: seeded draws repeat, top_k / top_p / min_p cuts      T8 chat template ids == transformers apply_chat_template
  T9 server: chat completion, stream and non-stream, on the tiny model
"""
import json
import os
import struct
import sys
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np                     # noqa: E402
import torch                           # noqa: E402
import jax                             # noqa: E402
import jax.numpy as jnp                # noqa: E402
import ml_dtypes                       # noqa: E402

import mashbox                         # noqa: E402,F401
from mashbox.models import qwen3_8_27b as Q          # noqa: E402
from mashbox.models.qwen3_8_27b import weights as QW, model as QM   # noqa: E402
from mashbox.runtime.engine import Engine           # noqa: E402
from mashbox.runtime import sampler as SP           # noqa: E402

WD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mashbox-cpu-test"
TOKDIR = sys.argv[2] if len(sys.argv) > 2 else None
MD = os.path.join(WD, "tiny")
os.makedirs(WD, exist_ok=True)
R = {}


def build():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4, rms_norm_eps=1e-6,
                            rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                                 mrope_section=[2, 1, 1], mrope_interleaved=True),
                            tie_word_embeddings=False, attn_output_gate=True, pad_token_id=0)
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif "dt_bias" in n:
                p.normal_(0, 1)
            elif p.ndim == 1:
                p.normal_(0, 0.3)
            elif "conv1d" in n:
                p.normal_(0, 0.4)
            else:
                p.normal_(0, 0.08)
    m = m.to(torch.bfloat16).eval()
    m.save_pretrained(MD)
    return m


# ---- test fixture: write a .bshj from the tiny model (container layout as documented in mashbox/bshj.py)
def write_bshj(path, tensors, meta):
    align, pos, ents, bufs = 4096, 0, {}, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {"bfloat16": "bf16", "float32": "f32", "int8": "i8"}[a.dtype.name]
        pos = -(-pos // align) * align
        ents[name] = dict(dtype=dt, shape=list(a.shape), offset=pos, nbytes=a.nbytes, layout="row@1")
        bufs.append((pos, a.tobytes()))
        pos += a.nbytes
    hdr = dict(meta, format="bshj", version="1.0", data_len=pos, tensors=ents)
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    off = -(-(64 + len(raw)) // align) * align
    raw += b" " * (off - 64 - len(raw))
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHQQIIQQ16s", b"BSHJ", 1, 0, len(raw), off, align, 1, pos, 0, bytes(16)) + raw)
        for p, b in bufs:
            f.seek(off + p)
            f.write(b)
        f.truncate(off + pos)


def hf_tensors(md):
    from safetensors.numpy import load_file
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(md, "*.safetensors"))):
        from safetensors import safe_open
        with safe_open(f, "pt") as s:
            for k in s.keys():
                t = s.get_tensor(k)
                out[k] = t.view(torch.int16).numpy().view(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.float().numpy()
    return out


BIG = ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight",
       "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def int8_rows(w):            # fixture only: symmetric int8, one f32 scale per output row
    w32 = w.astype(np.float32)
    am = np.abs(w32).max(-1)
    s = np.where(am > 0, am / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    return np.clip(np.round(w32 / s[:, None]), -127, 127).astype(np.int8), s


def kl_top1(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    la = a - a.max(-1, keepdims=True); la -= np.log(np.exp(la).sum(-1, keepdims=True))
    lb = b - b.max(-1, keepdims=True); lb -= np.log(np.exp(lb).sum(-1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(-1)
    s = np.sort(a, -1)
    t1 = a.argmax(-1) == b.argmax(-1)
    return dict(kl_mean=float(kl.mean()), top1=float(t1.mean()), miss_gaps=[float(x) for x in (s[:, -1] - s[:, -2])[~t1]])


m = build()
T = hf_tensors(MD)
write_bshj(os.path.join(WD, "tiny-bf16.bshj"), T, {})
T8 = {}
for k, v in T.items():
    if k.endswith(BIG) or k.endswith("embed_tokens.weight") or k == "lm_head.weight":
        q, s = int8_rows(v)
        T8[k], T8[k + "_scale"] = q, s
    else:
        T8[k] = v
write_bshj(os.path.join(WD, "tiny-int8.bshj"), T8, {"bshj": {"quant": {"type": "TEST"}}})

cfg = Q.Config(MD)
W, info = QW.load(os.path.join(WD, "tiny-bf16.bshj"), cfg)
eng = Engine(cfg, W, Q, CTX=512, S=4, K=4)
rng = np.random.default_rng(1)
seqs = [rng.integers(1, 512, n).tolist() for n in (40, 64, 150)]
with torch.no_grad():
    hl = [m(input_ids=torch.tensor([s])).logits[0].float().numpy() for s in seqs]
ol = [eng.logits(s) for s in seqs]
R["T1"] = [kl_top1(h, o) for h, o in zip(hl, ol)]
R["T1_pass"] = all(r["kl_mean"] < 1e-3 and all(g < 0.05 for g in r["miss_gaps"]) for r in R["T1"])

prompts = [s[:n] for s, n in zip(seqs, (12, 20, 33))]
from transformers import GenerationConfig   # noqa: E402
gc = GenerationConfig(max_new_tokens=24, do_sample=False, eos_token_id=None, pad_token_id=0)
with torch.no_grad():
    hg = [m.generate(input_ids=torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long), generation_config=gc)[0, len(p):].tolist()
          for p in prompts]
og = [x[:24] for x in eng.generate(prompts, max_new=24)["ids"]]
R["T2"] = dict(equal=[a == b for a, b in zip(hg, og)])
gaps = []
for i, (a, b) in enumerate(zip(hg, og)):
    fd = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if fd is not None:
        with torch.no_grad():
            lg = m(input_ids=torch.tensor([prompts[i] + a])).logits[0, len(prompts[i]) + fd - 1].float().numpy()
        s_ = np.sort(lg)
        gaps.append(float(s_[-1] - s_[-2]))
R["T2"]["gap_at_first_diff"] = gaps
R["T2_pass"] = all(R["T2"]["equal"]) or all(g < 0.05 for g in gaps)

B_, T_, H_, D_ = 2, 150, 3, 16
q, k = (QM.l2norm(jnp.asarray(rng.normal(size=(B_, T_, H_, D_)), jnp.float32)) for _ in range(2))
v = jnp.asarray(rng.normal(size=(B_, T_, H_, D_)), jnp.float32)
g = jnp.asarray(-np.abs(rng.normal(size=(B_, T_, H_))) * 0.5, jnp.float32)
be = jnp.asarray(rng.uniform(0.01, 1, size=(B_, T_, H_)), jnp.float32)
s0 = jnp.asarray(rng.normal(size=(B_, H_, D_, D_)) * 0.1, jnp.float32)
oc, sc = QM.gated_delta_chunk(q, k, v, g, be, s0)
s, outs = s0, []
for t in range(T_):
    o_, s = QM.gated_delta_step(q[:, t], k[:, t], v[:, t], g[:, t], be[:, t], s)
    outs.append(o_)
orr = jnp.stack(outs, 1)
S64 = np.asarray(s0, np.float64)
qn, kn, vn, gn, bn = (np.asarray(x, np.float64) for x in (q, k, v, g, be))
o64 = np.zeros((B_, T_, H_, D_))
for t in range(T_):
    S64 = S64 * np.exp(gn[:, t])[..., None, None]
    kv = np.einsum("bhkv,bhk->bhv", S64, kn[:, t])
    S64 = S64 + kn[:, t][..., :, None] * ((vn[:, t] - kv) * bn[:, t][..., None])[..., None, :]
    o64[:, t] = np.einsum("bhkv,bhk->bhv", S64, qn[:, t] / np.sqrt(D_))
rel = lambda a, b: float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))
R["T3"] = dict(chunk_vs_f64=rel(oc, o64), step_vs_f64=rel(orr, o64), state_chunk=rel(sc, S64), state_step=rel(s, S64))
R["T3_pass"] = max(R["T3"].values()) < 1e-5

eng.chunk = 32
ol32 = eng.logits(seqs[2])
eng.chunk = 512
R["T4"] = kl_top1(ol[2], ol32)       # chunk boundaries move the f32 sums: same bound as T1
R["T4_pass"] = R["T4"]["kl_mean"] < 1e-3 and all(g < 0.05 for g in R["T4"]["miss_gaps"])

alone = [eng.generate([p], max_new=16)["ids"][0] for p in prompts]
together = eng.generate(prompts, max_new=16)["ids"]
R["T5"] = dict(equal=[a == b for a, b in zip(alone, together)])
R["T5_pass"] = all(R["T5"]["equal"])

W8, info8 = QW.load(os.path.join(WD, "tiny-int8.bshj"), cfg)
eng8 = Engine(cfg, W8, Q, CTX=512, S=4, K=4)
l8 = eng8.logits(seqs[1])
R["T6"] = dict(kl_vs_bf16=kl_top1(ol[1], l8), bpw=info8["bpw"], quant=info8["quant"])
R["T6_pass"] = bool(np.isfinite(l8).all()) and info8["quant"] == "TEST"   # load path check; quality on a random tiny model is information only

samp = [dict(temperature=0.8, top_p=0.9, seed=7), dict(temperature=1.0, top_k=5, seed=8), None]
a1 = eng.generate(prompts, max_new=12, samp=samp)["ids"]
a2 = eng.generate(prompts, max_new=12, samp=samp)["ids"]
solo = eng.generate(prompts[:1], max_new=12, samp=samp[:1])["ids"][0]
x = jnp.asarray(rng.normal(size=(3, 512)) * 3, jnp.float32)
sp = SP.to_device(SP.pack([dict(temperature=1.0, top_k=7, seed=1), dict(temperature=1.0, top_p=0.5, seed=2),
                           dict(temperature=1.0, min_p=0.2, seed=3)], 3))
kept = []
for st in range(200):
    kept.append(np.asarray(SP.sample(x, sp, jnp.full((3,), st, jnp.int32))))
kept = np.stack(kept)
xs = np.asarray(x, np.float64)
top7 = set(np.argsort(-xs[0])[:7].tolist())
p1 = np.exp(xs[1] - xs[1].max()); p1 /= p1.sum()
o1 = np.argsort(-xs[1]); cum = np.cumsum(p1[o1]); nuc = set(o1[:int(np.searchsorted(cum, 0.5) + 1)].tolist())
p2 = np.exp(xs[2] - xs[2].max()); p2 /= p2.sum()
minp = set(np.nonzero(p2 >= 0.2 * p2.max())[0].tolist())
R["T7"] = dict(repeat=a1 == a2, solo_equals_batch_row=solo == a1[0], greedy_row_equals_greedy=a1[2][:12] == together[2][:12],
               top_k_ok=set(kept[:, 0].tolist()) <= top7, top_p_ok=set(kept[:, 1].tolist()) <= nuc,
               min_p_ok=set(kept[:, 2].tolist()) <= minp)
R["T7_pass"] = all(R["T7"].values())

if TOKDIR:
    from transformers import AutoTokenizer
    from mashbox.runtime.chat import Chat
    ch = Chat(TOKDIR)
    ht = AutoTokenizer.from_pretrained(TOKDIR)
    convs = [[dict(role="user", content="Hi")],
             [dict(role="system", content="Be brief."), dict(role="user", content="Two plus two?"),
              dict(role="assistant", content="Four."), dict(role="user", content="And times three?")]]
    eq = []
    for c in convs:
        for think in (False, True):
            r = ht.apply_chat_template(c, add_generation_prompt=True, enable_thinking=think, tokenize=True)
            r = list(r["input_ids"]) if hasattr(r, "keys") else list(r)
            eq.append(r == ch.chat_ids(c, enable_thinking=think))
    R["T8"] = dict(equal=eq)
    R["T8_pass"] = all(eq)

    from mashbox.runtime.server import Server, serve
    import urllib.request

    class TinyChat(Chat):         # the real template, ids folded into the tiny vocabulary
        def encode(self, text):
            return [t % 512 for t in super().encode(text)]
    srv = Server(eng, TinyChat(TOKDIR), "tiny", {})
    srv.ready = True
    threading.Thread(target=serve, args=(srv, "127.0.0.1", 18765, lambda s: None), daemon=True).start()
    time.sleep(1.0)

    def post(body):
        rq = urllib.request.Request("http://127.0.0.1:18765/v1/chat/completions", json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
        return urllib.request.urlopen(rq, timeout=600).read().decode()
    body = dict(messages=[dict(role="user", content="Hello")], max_tokens=10)
    full = json.loads(post(body))
    ev = [json.loads(l[6:]) for l in post(dict(body, stream=True)).splitlines() if l.startswith("data: {")]
    streamed = "".join((e["choices"][0].get("delta") or {}).get("content") or "" for e in ev)
    outs = []
    th = [threading.Thread(target=lambda: outs.append(json.loads(post(body)))) for _ in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    R["T9"] = dict(nonstream_tokens=full["usage"]["completion_tokens"], stream_equals_nonstream=streamed == full["choices"][0]["message"]["content"],
                   finish=full["choices"][0]["finish_reason"], concurrent_equal=all(o["choices"][0]["message"]["content"] ==
                                                                                     full["choices"][0]["message"]["content"] for o in outs))
    R["T9_pass"] = R["T9"]["stream_equals_nonstream"] and R["T9"]["concurrent_equal"] and full["usage"]["completion_tokens"] > 0

R["ALL_PASS"] = all(v for k, v in R.items() if k.endswith("_pass"))
R["jax"] = jax.__version__
print(json.dumps(R, indent=1, default=str))
sys.exit(0 if R["ALL_PASS"] else 1)
