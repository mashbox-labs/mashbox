"""Track2Rig (MASHBOX Labs): dense 3D point tracks of a moving creature in, a rig out (bones, top-4 skin, per-frame bone
transforms, skeleton). Plain JAX and NumPy on the CPU. See docs/models/track2rig.md.

Command line: ``mashbox run track2rig TRACKS.npz`` (app.py). Python: ``load()`` returns a ``Track2Rig``; JAX is imported
only when a model is loaded.
"""
from ... import __version__   # the mashbox package version, one source (the generator string in rig.json and rig.glb)


class Track2Rig:
    """Loaded model. `Track2Rig().rig(tracks, visibility)` returns a `Rig`."""

    def __init__(self, weights=None):
        from .weights import load_params
        self.params = load_params(weights)

    def rig(self, tracks, visibility=None, max_bones=32):
        from .rig import solve
        from .tracks import check_tracks
        X, mask = check_tracks(tracks, visibility)
        return solve(self.params, X, mask, max_bones=max_bones)

    def rig_file(self, path, max_bones=32):
        from .rig import solve
        from .tracks import load_tracks
        X, mask = load_tracks(path)
        return solve(self.params, X, mask, max_bones=max_bones)


def load(weights=None):
    """The model with the released weights: downloaded from Hugging Face at the pinned revision, or `weights` = a local
    copy of the same file. Either way its sha256 is checked against models.json before it is read."""
    return Track2Rig(weights)


def main(cmd, argv):
    from .app import main as _main
    return _main(cmd, argv)


__all__ = ["Track2Rig", "load", "main", "__version__"]
