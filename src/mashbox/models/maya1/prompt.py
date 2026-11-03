"""maya1's prompt: its tokenizer, the prompt layout, input checks and the SNAC code unpack.

The prompt is the layout of Maya Research's example (maya-research/maya1 README):

  [SOH 128259] [BOS 128000]  tokens('<description="DESCRIPTION"> TEXT')  [EOT 128009] [EOH 128260] [SOA 128261] [SOS 128257]

and the model answers with SNAC code tokens (7 per audio frame) until 128258.

Emotion tags (`<laugh>`, `<whisper>` and the rest of the 20 single-token tags) reach the model only where the user's own
text holds that tag literally. Nothing is ever inserted. Every other special-token string in user input
(`<|eot_id|>`, `<custom_token_N>`, `<|audio|>`) is tokenized as plain text, so typed input can never forge the prompt
frame or a SNAC code. Two tokenizers are built from the pinned tokenizer.json for that:

  tags   the pinned tokenizer with only the 20 tag tokens kept as added tokens. Hugging Face `tokenizers` splits on
         added tokens first and runs the pre-tokenizer and BPE on the text between them, so for input with no other
         special-token string the ids are the same as the pinned tokenizer's.
  plain  no added tokens at all: a description holding any of  "  <  >  is tokenized piece by piece with it, each `"`
         alone, so the closing `">` of the frame can never form inside a description and a tag typed there stays text.
"""
import hashlib
import json
import os
import re

SOH, EOH, SOA, SOS, EOS_SPEECH = 128259, 128260, 128261, 128257, 128258
BOS, EOT, PAD = 128000, 128009, 128263
CODE0, NCODE = 128266, 7 * 4096          # SNAC code tokens [128266, 156938)
TAG_LO, TAG_HI = 156940, 156960          # the 20 single-token tags
ADDED_LO, VOCAB = 128000, 156960
MAX_INPUT_CHARS = 4096                    # the OpenAI speech API's input limit; longer input is refused, never cut
MAX_DESC_CHARS = 1000                     # voice description + instructions
_GUARD = re.compile(r'["<>]')


class InputError(ValueError):
    """A request the model cannot take: param names the field."""
    def __init__(self, param, msg):
        super().__init__(msg)
        self.param = param


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Tokenizer:
    """maya1's tokenizer from its tokenizer.json (path or directory). pin: the file's sha256 to check, or None."""

    def __init__(self, path, pin=None):
        from tokenizers import Tokenizer as T
        path = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
        if pin is not None:
            got = sha256(path)
            if got != pin:
                raise RuntimeError(f"{path}: sha256 {got}, expected {pin}")
        raw = open(path, encoding="utf-8").read()
        j = json.loads(raw)
        added = j["added_tokens"]
        self.tags = {a["content"]: a["id"] for a in added if TAG_LO <= a["id"] < TAG_HI}
        if len(self.tags) != 20 or sorted(self.tags.values()) != list(range(TAG_LO, TAG_HI)):
            raise RuntimeError(f"{path}: the tag tokens are not ids {TAG_LO}..{TAG_HI - 1}; this is not maya1's tokenizer")
        for a in added:          # the equivalence above needs no normalizer and no strip or single_word flags
            if a["lstrip"] or a["rstrip"] or a["single_word"]:
                raise RuntimeError(f"{path}: added token {a['content']!r} has strip flags; this is not maya1's tokenizer")
        if j.get("normalizer") is not None:
            raise RuntimeError(f"{path}: has a normalizer; this is not maya1's tokenizer")
        self.full = T.from_str(raw)

        def variant(keep):
            v = dict(j)
            v["added_tokens"] = [a for a in added if a["id"] in keep]
            v["post_processor"] = None
            return T.from_str(json.dumps(v))
        self.t_tags = variant(set(self.tags.values()))
        self.t_plain = variant(set())
        # `tokenizers` numbers added tokens missing from the BPE vocabulary from the vocabulary size on, in load order,
        # so the tags variant's tag ids are mapped back to the pinned ids; the BPE ids are the same in all three
        self.tag_map = {self.t_tags.token_to_id(c): i for c, i in self.tags.items()}
        self.tag_re = re.compile("|".join(re.escape(t) for t in sorted(self.tags, key=len, reverse=True)))

    def _enc(self, t, s):
        if not s:
            return []
        ids = t.encode(s, add_special_tokens=False).ids
        if t is self.t_tags:
            ids = [self.tag_map.get(i, i) if i >= ADDED_LO else i for i in ids]
        bad = [i for i in ids if not (i < ADDED_LO or TAG_LO <= i < TAG_HI)]
        if bad:
            raise RuntimeError(f"tokenizer gave special ids {bad[:4]} for user text")
        return list(ids)

    def count_tags(self, text):
        return len(self.tag_re.findall(text))

    def build(self, description, text):
        """-> the prompt's token ids for one take."""
        if not isinstance(text, str) or not text.strip():
            raise InputError("input", "input must be a non-empty string")
        if len(text) > MAX_INPUT_CHARS:
            raise InputError("input", f"input is longer than {MAX_INPUT_CHARS} characters")
        if not isinstance(description, str) or not description.strip():
            raise InputError("voice", "the voice description must be a non-empty string")
        if len(description) > MAX_DESC_CHARS:
            raise InputError("instructions", f"voice description and instructions are longer than {MAX_DESC_CHARS} characters")
        head, tail = [SOH, BOS], [EOT, EOH, SOA, SOS]
        if not _GUARD.search(description):
            return head + self._enc(self.t_tags, f'<description="{description}"> ' + text) + tail
        body = self._enc(self.t_plain, '<description="')
        for piece in re.split(r'(")', description):
            body += self._enc(self.t_plain, piece)
        body += self._enc(self.t_tags, '"> ' + text)
        return head + body + tail

    def reference_ids(self, description, text):
        """The same prompt through the pinned tokenizer as published (Maya Research's example builds it this way). Equal to
        build() for any description without  "  <  >  and any text without a special-token string other than a tag."""
        dec = lambda i: self.full.decode([i], skip_special_tokens=False)
        p = dec(SOH) + dec(BOS) + f'<description="{description}"> ' + text + dec(EOT) + dec(EOH) + dec(SOA) + dec(SOS)
        return list(self.full.encode(p, add_special_tokens=False).ids)


def compose_description(voice_description, instructions=None):
    """The voice's description, with instructions (emotion, style) appended as a sentence. No tag is ever added."""
    d = voice_description.strip()
    ins = (instructions or "").strip()
    if not ins:
        return d
    if d and d[-1] not in ".!?":
        d += "."
    return f"{d} {ins}"


# ---------------------------------------------------------------------------------------------- long input -> takes
SEG_CHARS = 240          # longest single take; 2048 generated tokens hold 24.97 s of audio
_B1 = re.compile(r'[.!?…]["\'”’)\]]*(\s+)|(\n\s*)')
_B2 = re.compile(r'[,;:—–](\s+)')
_B3 = re.compile(r'(\s+)')
_TAGLIT = re.compile(r"<[a-z_]{2,16}>")


def _last_cut(rx, s, lo, hi):
    best = None
    for m in rx.finditer(s, lo, hi + 1):
        a = m.start(m.lastindex)
        if lo < a <= hi:
            best = (a, m.end(m.lastindex))
    return best


def segment_text(text, n=SEG_CHARS):
    """Text longer than n characters -> takes of at most n characters, cut at the last sentence end (or newline) that
    fits, else the last , ; : or dash, else the last space, else a hard cut that never splits a <tag>. Only whitespace at
    a cut is dropped; every other character stays, in order, and nothing is inserted."""
    if len(text) <= n:
        return [text]
    out, start, L = [], 0, len(text)
    while True:
        while start < L and text[start].isspace():
            start += 1
        if L - start <= n:
            if start < L:
                out.append(text[start:].rstrip())
            return out
        hi = start + n
        cut = None
        for rx in (_B1, _B2, _B3):
            cut = _last_cut(rx, text, start, hi)
            if cut:
                break
        if cut is None:
            a = hi
            for m in _TAGLIT.finditer(text, max(start, hi - 20), min(L, hi + 20)):
                if m.start() < a < m.end():
                    a = m.start()
            if a <= start:
                a = hi
            cut = (a, a)
        seg = text[start:cut[0]].rstrip()
        if seg:
            out.append(seg)
        start = cut[1]
