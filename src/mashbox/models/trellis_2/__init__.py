"""TRELLIS.2 (Microsoft) image to 3D on one TPU v6e chip. See docs/models/trellis-2.md.

Command line: ``mashbox run trellis-2 --image in.png --out out.glb`` (app.py). The modules import JAX only when used.
"""


def main(cmd, argv):
    from .app import main as _main
    return _main(cmd, argv)
