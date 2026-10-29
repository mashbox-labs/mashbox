#!/usr/bin/env bash
# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
#
# Two Python environments, because the tracker pins Python 3.10 and older numpy/scipy
# than mashbox (Python 3.12) does:
#   .venv-video  Python 3.10, track2rig-video + TrackEverything (GPU, CC BY-NC 4.0 tracker)
#   .venv-rig    Python 3.12, mashbox[cpu,track2rig] (CPU)
# Run from the repository root. Needs uv (https://docs.astral.sh/uv/) and git.
set -euo pipefail
uv venv -q -p 3.10 .venv-video
uv venv -q -p 3.12 .venv-rig
VIRTUAL_ENV=$PWD/.venv-video uv pip install -q pip
(. .venv-video/bin/activate && bash packages/track2rig-video/install_trackeverything.sh third_party_src && pip install -q ./packages/track2rig-video)
VIRTUAL_ENV=$PWD/.venv-rig uv pip install -q ".[cpu,track2rig]"
echo "ready: .venv-video/bin/track2rig-video and .venv-rig/bin/mashbox"
