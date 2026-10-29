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
}
REQUEST = "https://github.com/mashbox-labs/mashbox/issues"


def unsupported(arch, path=None, repo=None, quant=None):
    from ... import __version__
    cmd = llama_cpp_command(path, repo, quant)
    if arch and arch not in known_archs():
        return (f'mashbox: "{arch}" is newer than the GGUF reader in mashbox {__version__} (gguf {reader_version()}), and it '
                f"does not run on TPU here. llama.cpp runs it: {cmd}  (https://github.com/ggml-org/llama.cpp). "
                f"Ask for it on TPU: {REQUEST}")
    return (f'mashbox: "{arch}" does not run on TPU in mashbox {__version__}. llama.cpp runs it: {cmd}  '
            f"(https://github.com/ggml-org/llama.cpp). Ask for it on TPU: {REQUEST}")


def held(arch, path=None, repo=None, quant=None):
    """None, or the one line `run` refuses with for an architecture the catalog does not list. A mixture-of-experts
    architecture (catalog.moe_arch) runs only once the catalog lists it (catalog.arch_rows: a passing chip row of a MoE
    file of it, which the record guard holds to its sealed chip record), so a MoE file is never served on the exact
    reference path in place of its routed-expert kernel. Listed and runnable read the one rule; a dense architecture is
    never held. The line is the unsupported-architecture line."""
    from ... import catalog
    if not catalog.moe_arch(arch, catalog.arch_models().get(arch)):
        return None
    if any(r["arch"] == arch for r in catalog.arch_rows(catalog.registry())):
        return None
    return unsupported(arch, path, repo, quant)


def module(arch, path=None, repo=None, quant=None):
    if arch not in ARCHS:
        raise GGUFError(unsupported(arch, path, repo, quant))
    return importlib.import_module(ARCHS[arch])
