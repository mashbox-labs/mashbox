"""Hy-Nomas (Tencent Hunyuan) on TPU v6e-4. See docs/models/hy-nomas.md.

Command line: ``python3 -m mashbox run hy-nomas`` (app.py). The modules import JAX only when used.
"""


def main(cmd, argv):
    from .app import main as _main
    return _main(cmd, argv)
