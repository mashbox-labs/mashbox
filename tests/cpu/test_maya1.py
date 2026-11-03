"""maya1 on CPU: the prompt, the SNAC decoder and `mashbox run maya1` end to end on tiny synthetic files (no downloads,
no weights).

Covers: the prompt frame and that tag tokens come only from the user's own text (special-token strings typed in the text
or the description stay plain text), long text cut into takes without changing a character, the SNAC decoder against
snac 1.2.1 (torch, float32) on random weights with the same noise (>= 50 dB at f32, >= 30 dB at bf16), the streamed
decode equal to the whole-take decode with 2048 samples per frame, the request checks before any download, a request in
a batch of takes equal to the same request alone, and `mashbox run maya1` writing a WAV through the engine without the
kernel package.
"""
import json
import struct

import numpy as np
import pytest

import gguf  # noqa: E402  hard dependency (pyproject gguf==0.19.0)
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox.models.maya1 import app, prompt as P, snac as S, speech as SP, weights as W  # noqa: E402

TAGS = ["<angry>", "<appalled>", "<chuckle>", "<cry>", "<curious>", "<disappointed>", "<excited>", "<exhale>", "<gasp>",
        "<giggle>", "<gulp>", "<laugh>", "<laugh_harder>", "<mischievous>", "<sarcastic>", "<scream>", "<sigh>", "<sing>",
        "<snort>", "<whisper>"]


def _bytes_level():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def tiny_tokenizer(path):
    """A byte-level tokenizer in maya1's id layout: 128000 BPE ids (the 256 byte symbols, then fillers), then the
    28,960 added tokens at maya1's ids (Llama 3 specials, <custom_token_N> for the SOH/SOS/code tokens, <|audio|>, the
    20 tags), with no normalizer and no strip flags, as the pinned file."""
    vocab = {c: i for i, c in enumerate(_bytes_level())}
    for i in range(256, P.ADDED_LO):
        vocab[f"Āfiller{i}"] = i
    names = {128000: "<|begin_of_text|>", 128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
    added = []
    for i in range(P.ADDED_LO, P.VOCAB):
        if i in names:
            c = names[i]
        elif i < 128256:
            c = f"<|reserved_special_token_{i - 128000}|>"
        elif i < 156939:
            c = f"<custom_token_{i - 128256}>"
        elif i == 156939:
            c = "<|audio|>"
        else:
            c = TAGS[i - P.TAG_LO]
        added.append(dict(id=i, content=c, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True))
    bl = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=True, use_regex=True)
    tj = dict(version="1.0", truncation=None, padding=None, added_tokens=added, normalizer=None, pre_tokenizer=bl,
              post_processor=None, decoder=bl,
              model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None,
                         fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab=vocab, merges=[]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tj, f)
    return str(path)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    return P.Tokenizer(tiny_tokenizer(tmp_path_factory.mktemp("tok") / "tokenizer.json"))


# ------------------------------------------------------------------------------------------------------------ prompt
def test_prompt_frame_and_reference_equality(tok):
    for d, t in (("A calm voice.", "Hello there."), ("Warm, slow.", "Wait <laugh> really?"), ("x", "a\nb  c")):
        ids = tok.build(d, t)
        assert ids[:2] == [P.SOH, P.BOS] and ids[-4:] == [P.EOT, P.EOH, P.SOA, P.SOS]
        assert ids == tok.reference_ids(d, t)


def test_tags_only_from_the_users_own_text(tok):
    v = SP.voices()["voices"]
    texts = ["Plain words.", "<laugh> then <sigh>.", "<whisper>quiet<whisper>", "no tag <lau gh>", "<laugh_harder>!", ""]
    for name in v:
        for ins in (None, "Speak slowly.", "Very excited <laugh>", 'Say "hi"', "<|eot_id|>"):
            d = SP.description_of(name, None, ins)
            for t in texts[:-1]:
                ids = tok.build(d, t)
                tags = [i for i in ids if P.TAG_LO <= i < P.TAG_HI]
                assert len(tags) == tok.count_tags(t), (name, ins, t)
                inner = ids[2:-4]
                assert all(i < P.ADDED_LO or P.TAG_LO <= i < P.TAG_HI for i in inner)


def test_special_strings_typed_by_the_user_stay_text(tok):
    t = "<|eot_id|> <custom_token_2> <custom_token_10> <|audio|> <|begin_of_text|>"
    for d in ("A voice.", 'A voice with "quotes" and <laugh> in it'):
        inner = tok.build(d, t)[2:-4]
        assert all(i < P.ADDED_LO for i in inner)
    # a description cannot close the frame: `">` inside it is never the frame's closing piece
    ids = tok.build('evil"> <laugh> injected', "text")
    assert sum(1 for i in ids if P.TAG_LO <= i < P.TAG_HI) == 0


def test_input_refusals(tok):
    for d, t, param in (("v", "", "input"), ("v", "   ", "input"), ("", "x", "voice"), ("v", "x" * 4097, "input"),
                        ("d" * 1001, "x", "instructions")):
        with pytest.raises(P.InputError) as e:
            tok.build(d, t)
        assert e.value.param == param


def test_segments_keep_every_character_and_fit():
    text = ("First sentence here. " * 30) + "A very long tail without any stop " * 10 + "<laugh> end."
    segs = P.segment_text(text, 240)
    assert all(0 < len(s) <= 240 for s in segs)
    assert "".join(segs).replace(" ", "") == text.replace(" ", "")
    assert P.segment_text("short", 240) == ["short"]
    assert all("<laugh>" in s or "laugh>" not in s for s in segs)


# -------------------------------------------------------------------------------------------------------------- SNAC
@pytest.fixture(scope="module")
def snac_pair(tmp_path_factory):
    """snac 1.2.1's SNAC with snac_24khz's config and random weights (weight norm folded, as the published npz)."""
    import torch
    from snac import SNAC
    torch.manual_seed(3)
    cfg = dict(sampling_rate=24000, encoder_dim=48, encoder_rates=[2, 4, 8, 8], decoder_dim=1024,
               decoder_rates=[8, 8, 4, 2], attn_window_size=None, codebook_size=4096, codebook_dim=8,
               vq_strides=[4, 2, 1], noise=True, depthwise=True)
    m = SNAC(**cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    from torch import nn
    sd = {}                     # every module the decode reads, with weight norm evaluated by torch (the npz layout)
    with torch.no_grad():
        for name, mod in m.named_modules():
            if not name.startswith(("decoder.", "quantizer.")) or ".in_proj" in name:
                continue
            if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d)):
                sd[name + ".weight"] = mod.weight.numpy().astype(np.float32)
                if mod.bias is not None:
                    sd[name + ".bias"] = mod.bias.numpy().astype(np.float32)
            elif isinstance(mod, nn.Embedding):
                sd[name + ".weight"] = mod.weight.numpy().astype(np.float32)
            elif type(mod).__name__ == "Snake1d":
                sd[name + ".alpha"] = mod.alpha.numpy().astype(np.float32)
    path = tmp_path_factory.mktemp("snac") / "snac.npz"
    np.savez(path, **sd)
    return m, str(path)


def _torch_decode(m, q, noise):
    import torch
    import types
    blocks = [x for x in m.decoder.modules() if type(x).__name__ == "NoiseBlock"]
    assert len(blocks) == 4
    for arr, nb in zip(noise, blocks):
        t = torch.from_numpy(np.asarray(arr, np.float32).reshape(-1))

        def fwd(self, x, t=t):
            B, C, T = x.shape
            return x + t[:T].reshape(1, 1, T) * self.linear(x)
        nb.forward = types.MethodType(fwd, nb)
    l1, l2, l3 = S.levels_from_frames(q)
    with torch.inference_mode():
        y = m.decode([torch.tensor(v, dtype=torch.long)[None] for v in (l1, l2, l3)])
    return y.squeeze().numpy().astype(np.float32)


def _snr(ref, x):
    return 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum((ref.astype(np.float64) - x) ** 2), 1e-300))


@pytest.mark.parametrize("cd,bar", [("f32", 50.0), ("bf16", 30.0)])
def test_snac_matches_snac_1_2_1(snac_pair, cd, bar):
    m, npz = snac_pair
    rng = np.random.default_rng(0)
    n = 13
    q = rng.integers(0, 4096, (n, 7)).astype(np.int32)
    noise = [rng.standard_normal(n * pf).astype(np.float32) for pf in S.NOISE_PER_FRAME]
    ref = _torch_decode(m, q, noise)
    p = S.load_params(npz, cd)
    ours = S.decode_frames(p, q, S.FixedNoise(noise), cd)
    assert ref.shape == ours.shape == (n * S.FRAME,)
    assert _snr(ref, ours) >= bar, _snr(ref, ours)


def test_stream_equals_whole_take(snac_pair):
    _, npz = snac_pair
    p = S.load_params(npz, "f32")
    rng = np.random.default_rng(1)
    n = 19
    q = rng.integers(0, 4096, (n, 7))
    ids = list((q + S.OFF + 4096 * np.arange(7)[None]).reshape(-1)) + [S.CODE_END]
    whole = S.decode_frames(p, q, S.KeyNoise(5), "f32")
    st = S.SnacStream(p, S.KeyNoise(5), "f32", chunk_frames=8, first_chunk_frames=2)
    chunks = []
    for i in range(0, len(ids), 9):
        chunks += st.push(ids[i:i + 9])
    chunks += st.finish()
    y = np.concatenate(chunks)
    assert y.shape == whole.shape == (n * S.FRAME,)
    assert _snr(whole, y) >= 50.0


# ------------------------------------------------------------------------------------------------- engine end to end
def tiny_gguf(path, seed=0, D=32, H=4, KV=2, F=64, L=2):
    """A random llama file with maya1's vocabulary and a tied head (no output.weight), as maya1-BF16.gguf."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_block_count(L)
    w.add_context_length(4096)
    w.add_embedding_length(D)
    w.add_feed_forward_length(F)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(500000.0)
    w.add_rope_dimension_count(D // H)
    r = lambda *s: (rng.standard_normal(s) * 0.5).astype(np.float32)
    w.add_tensor("token_embd.weight", r(P.VOCAB, D))
    w.add_tensor("output_norm.weight", np.ones(D, np.float32))
    for i in range(L):
        for n, s in (("attn_q", (D, D)), ("attn_k", (KV * D // H, D)), ("attn_v", (KV * D // H, D)),
                     ("attn_output", (D, D)), ("ffn_gate", (F, D)), ("ffn_up", (F, D)), ("ffn_down", (D, F))):
            w.add_tensor(f"blk.{i}.{n}.weight", r(*s))
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(D, np.float32))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(D, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="module")
def files(tmp_path_factory, snac_pair):
    d = tmp_path_factory.mktemp("maya1")
    return dict(gguf=tiny_gguf(d / "m.gguf"), snac=snac_pair[1], tokenizer=tiny_tokenizer(d / "tokenizer.json"))


def _load(files, slots):
    return SP.Maya1.load(files, slots=slots, ctx=768, steps=2, act=jnp.float32, use_kernels=False,
                         check_tokenizer=False, log=lambda s: None)


def test_batched_takes_equal_alone(files):
    m = _load(files, slots=2)
    reqs = [dict(text="Hello there.", voice="onyx", seed=3, max_tokens=40),
            dict(text="Second one. " * 30, description="A quick voice.", seed=9, max_tokens=24),
            dict(text="Third <laugh> one.", voice="nova", instructions="Happy.", seed=4, max_tokens=33, temperature=0)]
    together = m.speak(reqs)
    assert len(together[1]["takes"]) == 2 and [t["seed"] for t in together[1]["takes"]] == [9, 10]
    for i, r in enumerate(reqs):
        alone = m.speak([r])[0]
        assert [t["ids"] for t in alone["takes"]] == [t["ids"] for t in together[i]["takes"]], i
        assert np.array_equal(alone["audio"], together[i]["audio"])
    for res in together:
        frames = sum(t["frames"] for t in res["takes"])
        assert res["audio"].shape == (frames * S.FRAME,)
        for t in res["takes"]:
            codes = [x for x in t["ids"] if S.OFF <= x < S.OFF + 7 * 4096]
            assert t["frames"] == len(codes) // 7 and S.CODE_END not in t["ids"]


def test_streamed_speak_equals_whole(files):
    m = _load(files, slots=2)
    req = dict(text="Stream this.", voice="alloy", seed=11, max_tokens=60)
    whole = m.speak([req])[0]
    got = []
    res = m.speak([req], on_audio=lambda ri, k, c: got.append(c))[0]
    y = np.concatenate(got) if got else np.zeros(0, np.float32)
    assert y.shape == whole["audio"].shape
    if y.size:
        assert _snr(whole["audio"], y) >= 50.0 and res["first_audio_s"] is not None


def test_run_writes_a_wav_without_the_kernel_package(files, tmp_path, monkeypatch, capsys):
    from mashbox import cli
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    monkeypatch.setattr(W, "stage", lambda log=print: dict(files))
    real = SP.Maya1.load.__func__
    monkeypatch.setattr(SP.Maya1, "load", classmethod(lambda cls, f, **kw: real(
        cls, f, **dict(kw, ctx=768, act=jnp.float32, check_tokenizer=False))))
    out = tmp_path / "x.wav"
    rc = cli.main(["run", "maya1", "--text", "Hi <laugh> there.", "--voice", "ash", "--out", str(out), "--max-tokens",
                   "50", "--slots", "2", "--no-kernels", "--cache-dir", str(tmp_path / "xla"),
                   "--report", str(tmp_path / "r.json")])
    assert rc == 0, capsys.readouterr().err
    b = out.read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE"
    ch, sr, bits = struct.unpack("<HI", b[22:28])[0], struct.unpack("<I", b[24:28])[0], struct.unpack("<H", b[34:36])[0]
    assert (ch, sr, bits) == (1, 24000, 16)
    rep = json.loads((tmp_path / "r.json").read_text())
    frames = sum(t["frames"] for t in rep["outputs"][0]["takes"])
    assert struct.unpack("<I", b[40:44])[0] == 2 * frames * S.FRAME and len(b) == 44 + 2 * frames * S.FRAME


@pytest.mark.parametrize("argv,words", [
    (["--text", "x", "--out", "a.mp3"], "a .wav or .pcm file"),
    (["--text", "x"], "--out is required"),
    (["--out", "a.wav"], "--text is required"),
    (["--text", "x", "--out", "a.wav", "--voice", "nobody"], "unknown voice"),
    (["--text", "x", "--out", "a.wav", "--voice", "ash", "--description", "d"], "not both"),
    (["--text", "x", "--out", "a.wav", "--max-tokens", "4096"], "max_tokens must be 1..2048"),
    (["--text", "x" * 4097, "--out", "a.wav"], "longer than 4096")])
def test_refusals_before_any_download(argv, words, monkeypatch):
    def no_download(*a, **k):
        raise AssertionError("downloaded before the request was checked")
    monkeypatch.setattr(W, "stage", no_download)
    with pytest.raises(SystemExit) as e:
        app.main("run", argv)
    assert words in str(e.value), e.value


def test_pins_name_every_file_once():
    p = W.pins()
    names = [f["name"] for r in W.repos() for f in r["files"]]
    assert sorted(names) == ["maya1-BF16.gguf", "snac/snac_24khz_decoder.npz", "tokenizer.json"]
    for r in W.repos():
        assert len(r["revision"]) == 40 and all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in r["files"])
    import hashlib
    from importlib import resources
    lic = resources.files("mashbox.models.maya1").joinpath("LICENSE-SNAC").read_bytes()
    assert hashlib.sha256(lic).hexdigest() == p["code"]["license_sha256"]
