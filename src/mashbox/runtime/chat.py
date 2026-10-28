"""Tokenizer and chat template: the model's own tokenizer.json and chat_template.jinja, rendered the way Hugging Face
apply_chat_template does (sandboxed jinja2, add_generation_prompt=True). Stop ids come from generation_config.json."""
import json
import os


class Chat:
    def __init__(self, model_dir):
        from tokenizers import Tokenizer
        import jinja2
        import jinja2.sandbox
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(m):
            raise jinja2.exceptions.TemplateError(m)
        env.globals["raise_exception"] = raise_exception
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
        return self.encode(self.render(messages, enable_thinking, tools))
