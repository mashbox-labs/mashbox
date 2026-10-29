"""Tokenizer and chat template: the model's own tokenizer.json and chat_template.jinja, rendered the way Hugging Face
apply_chat_template does (sandboxed jinja2, add_generation_prompt=True). Stop ids come from generation_config.json.

Control tokens come only from the template. Every string a client sends (message and system text, tool results, tool
call names and arguments, tool definitions) is data: a literal "<|im_end|>" or "</tool_response>" in it is tokenized as
plain text, never as the control token the template uses for a turn boundary, so a file or web page an agent reads cannot
forge a user or system turn (Hugging Face tokenizers' split_special_tokens, applied to the client's parts only). A chat
whose strings hold no control-token text is tokenized exactly as before, in one pass over the rendered prompt.
"""
import json
import os
import re
import secrets


class Chat:
    def __init__(self, model_dir):
        from tokenizers import Tokenizer
        import jinja2
        import jinja2.sandbox
        spec = json.load(open(os.path.join(model_dir, "tokenizer.json")))
        self.tok = Tokenizer.from_str(json.dumps(spec))
        self.plain = Tokenizer.from_str(json.dumps(dict(spec, added_tokens=[])))     # every string is plain text to it
        lits = sorted({t["content"] for t in spec.get("added_tokens") or []}, key=len, reverse=True)
        self.control = re.compile("|".join(map(re.escape, lits))) if lits else None   # longest first, as tokenizers matches
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(m):
            raise jinja2.exceptions.TemplateError(m)
        env.globals["raise_exception"] = raise_exception
        import datetime
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)   # as transformers and llama.cpp
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self.template = env.from_string(open(os.path.join(model_dir, "chat_template.jinja")).read())
        gc = json.load(open(os.path.join(model_dir, "generation_config.json")))
        e = gc.get("eos_token_id")
        self.stop_ids = tuple(e if isinstance(e, list) else [e])
        self.defaults = {k: gc[k] for k in ("temperature", "top_k", "top_p") if k in gc}

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def render(self, messages, enable_thinking=False, tools=None):
        kw = {"tools": tools} if tools else {}
        return self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking, **kw)

    def chat_ids(self, messages, enable_thinking=False, tools=None):
        if self.control is None:
            return self.encode(self.render(messages, enable_thinking, tools))
        # Each control-token text in a client string becomes a marker the client cannot know (a fresh nonce in a Unicode
        # private-use plane), the template is rendered, and then: the template's own control tokens are cut out of the
        # prompt and kept as tokens, and the text between them, with the client's literals put back, is plain text.
        lits, nonce = [], secrets.token_hex(8)

        def hide(m):
            lits.append(m.group(0))
            return f"\U00100000{nonce}{len(lits) - 1}\U00100001"

        def walk(x):
            if isinstance(x, str):
                return self.control.sub(hide, x)
            if isinstance(x, dict):
                return {walk(k): walk(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [walk(v) for v in x]
            return x
        text = self.render(walk(messages), enable_thinking, walk(tools) if tools else tools)
        if not lits:
            return self.encode(text)                          # the usual case: one pass, the same ids as always
        back = re.compile(f"\U00100000{nonce}(\\d+)\U00100001")
        ids, at = [], 0
        for m in [*self.control.finditer(text), None]:
            piece = text[at:m.start() if m else len(text)]
            if piece:
                ids += self.plain.encode(back.sub(lambda k: lits[int(k.group(1))], piece), add_special_tokens=False).ids
            if m:
                ids.append(self.tok.token_to_id(m.group(0)))
                at = m.end()
        return ids
