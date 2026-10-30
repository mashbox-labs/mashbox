"""GGUF architectures this version runs on TPU: `general.architecture` -> the module with Config, load, tensor_roles,
new_state and make_forward. Adding one = one file here + one CPU test against llama.cpp's logits (docs/add-an-architecture.md)."""
import importlib

from ..reader import GGUFError, llama_cpp_command, known_archs, reader_version

ARCHS = {
    "qwen35": "mashbox.gguf.archs.qwen35",
    "qwen35moe": "mashbox.gguf.archs.qwen35",
    "llama": "mashbox.gguf.archs.dense",
    "qwen2": "mashbox.gguf.archs.dense",
    "qwen3": "mashbox.gguf.archs.dense",
    "gemma4": "mashbox.gguf.archs.gemma4",
    "gemma3": "mashbox.gguf.archs.gemma3",
    "exaone": "mashbox.gguf.archs.dense",
    "granite": "mashbox.gguf.archs.dense",
    "qwen3moe": "mashbox.gguf.archs.dense",          # routed experts: mashbox.moe reads them from the metadata
    "qwen2moe": "mashbox.gguf.archs.dense",
}
REQUEST = "https://github.com/mashbox-labs/mashbox/issues"
# architectures llama.cpp runs whose routed experts mashbox.moe reads, but whose other layers are not ported yet
MISSING = {"olmoe": "q/k RMSNorm over all heads", "deepseek2": "multi-head latent attention (MLA)",
           "gpt-oss": "attention sinks and sliding-window layers", "llama4": "interleaved chunked attention"}


def unsupported(arch, path=None, repo=None, quant=None):
    from ... import __version__
    cmd = llama_cpp_command(path, repo, quant)
    if arch and arch not in known_archs():
        return (f'mashbox: "{arch}" is newer than the GGUF reader in mashbox {__version__} (gguf {reader_version()}), and it '
                f"does not run on TPU here. llama.cpp runs it: {cmd}  (https://github.com/ggml-org/llama.cpp). "
                f"Ask for it on TPU: {REQUEST}")
    why = f" (its routed experts are read; missing here: {MISSING[arch]})" if arch in MISSING else ""
    return (f'mashbox: "{arch}" does not run on TPU in mashbox {__version__}{why}. llama.cpp runs it: {cmd}  '
            f"(https://github.com/ggml-org/llama.cpp). Ask for it on TPU: {REQUEST}")



def unsupported_moe(arch, path=None, repo=None, quant=None):
    """the line for a file with routed experts under an architecture whose files without them run here (llama +
    llama.expert_count, as llama.cpp writes Mixtral): it names the mixture-of-experts path, never the architecture"""
    from ... import __version__
    cmd = llama_cpp_command(path, repo, quant)
    return (f'mashbox: a "{arch}" file with routed experts (a mixture of experts) does not run on TPU in mashbox '
            f'{__version__}; "{arch}" files without routed experts do. llama.cpp runs it: {cmd}  '
            f"(https://github.com/ggml-org/llama.cpp). Ask for it on TPU: {REQUEST}")

def unproven_moe(arch, path=None, repo=None, quant=None, ids=()):
    """the line for a MoE file of an architecture whose proven MoE files run (ids: their catalog rows): this file is not
    one of them"""
    from ... import __version__
    cmd = llama_cpp_command(path, repo, quant)
    return (f'mashbox: this "{arch}" file with routed experts (a mixture of experts) does not run on TPU in mashbox '
            f'{__version__}; the routed experts run on TPU only in the "{arch}" files the catalog lists ({", ".join(ids)}). '
            f"llama.cpp runs it: {cmd}  (https://github.com/ggml-org/llama.cpp). Ask for it on TPU: {REQUEST}")


def proven(arch):
    """the catalog's passing chip rows of MoE files of arch: each proves its own file (sha256), and no other"""
    from ... import catalog
    return [g for g in catalog.registry().get("gguf") or []
            if g["arch"] == arch and g.get("moe") is True and catalog.chips_run(g)]


def held(arch, path=None, repo=None, quant=None, moe=False):
    """None, or the one line `run` refuses with. A mixture-of-experts file runs only when the catalog has a passing chip
    row of that file: a MoE row of its architecture whose sha256 is the file's (resolve.identity: the bytes the record
    guard holds the row's sealed chip record to, which also fix its repo, revision and name), so a MoE file is never
    served on the exact reference path in place of its routed-expert kernel, whose payloads are keyed to the proven
    file's shapes. A passing row lists the architecture (catalog.arch_rows) and releases no other file of it. A file
    with no name (path None) proves nothing. moe: the file has routed experts (moe_file); every file of
    a mixture-of-experts architecture (catalog.moe_arch) is held this way, and a MoE file of an architecture whose name
    says dense (llama + llama.expert_count, as llama.cpp writes Mixtral) the same way: a dense row of it releases dense
    files only, and a dense file is never held. The line: with no passing MoE row of the architecture, the
    unsupported-architecture line, or for a MoE file of an architecture whose dense files run unsupported_moe's line,
    which names the mixture-of-experts path and never the architecture as a whole; with one, unproven_moe's line, which
    names the files that run. The file is read for its sha256 only when its architecture has a passing MoE row (a Hugging
    Face cache blob is named by it, so a Hub file is not read)."""
    from ... import catalog
    named = catalog.moe_arch(arch, catalog.arch_models().get(arch))
    if not named and not moe:
        return None
    rows = proven(arch)
    if not rows:
        return unsupported(arch, path, repo, quant) if named else unsupported_moe(arch, path, repo, quant)
    if path:
        from ..resolve import identity
        sha = identity(path)[0]
        if sha is not None and any(g.get("sha256") == sha for g in rows):
            return None
    return unproven_moe(arch, path, repo, quant, [g["id"] for g in rows])


def drafting():
    """the architectures whose module declares SPEC_DRAFT (full attention in every layer): what a draft model
    (--draft-model) can be. A mixture-of-experts architecture (catalog.moe_arch, the rule held() uses) is left out: its
    module may declare SPEC_DRAFT for the dense files it also loads, and a draft model with routed experts is refused."""
    from ... import catalog
    labels = catalog.arch_models()
    return [a for a, m in ARCHS.items() if getattr(importlib.import_module(m), "SPEC_DRAFT", False)
            and not catalog.moe_arch(a, labels.get(a))]


def moe_file(f):
    """True when a GGUF file (reader.File) has routed experts: an ffn_*_exps tensor, as llama.cpp names them"""
    return any("_exps." in n for n in f.tensors)


def module(arch, path=None, repo=None, quant=None):
    if arch not in ARCHS:
        raise GGUFError(unsupported(arch, path, repo, quant))
    return importlib.import_module(ARCHS[arch])
