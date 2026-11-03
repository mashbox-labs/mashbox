"""maya1 (Maya Research) text to speech on a TPU, with the SNAC 24 kHz decoder. See docs/models/maya1.md.

Command line: ``mashbox run maya1 --text "Hello." --out hello.wav`` (app.py). Python: speech.Maya1. The modules import
JAX only when used.
"""


def main(cmd, argv):
    from .app import main as _main
    return _main(cmd, argv)
