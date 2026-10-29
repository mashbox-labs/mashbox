# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""Video -> 3D point tracks in the format Track2Rig reads (tracks.npz).

    from track2rig_video import align, load_raw, save_tracks
    tracks, visibility, info = align(load_raw("raw.npz"))   # raw: coords [T,N,3], visib [T,N], ...
    save_tracks("tracks.npz", tracks, visibility)

The video tracker (track2rig_video.trackeverything) is optional and has its own,
non-commercial licence; see README.md.
"""
__version__ = "0.1.0"

from .align import align
from .raw import load_raw
from .tracksfile import check_tracks, load_tracks, save_tracks

__all__ = ["align", "load_raw", "check_tracks", "load_tracks", "save_tracks"]
