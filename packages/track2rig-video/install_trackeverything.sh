#!/usr/bin/env bash
# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
#
# Installs TrackEverything (CMU and Meta), the optional 3D point tracker used by
# `track2rig-video VIDEO`, from its own repository at a pinned commit.
#
# TrackEverything's code and weights are licensed CC BY-NC 4.0: NON-COMMERCIAL USE ONLY.
# Nothing from it is part of this repository. By running this script you fetch it from
# its authors and accept its licence: https://creativecommons.org/licenses/by-nc/4.0/
#
# Needs: Linux, an NVIDIA GPU with a CUDA 12.9-capable driver, Python 3.10, git.
# Usage: bash packages/track2rig-video/install_trackeverything.sh [DIR]     (default DIR: ./third_party_src)
# Then:  export TRACKEVERYTHING_SRC=DIR/trackeverything
set -euo pipefail
DIR=${1:-third_party_src}
COMMIT=f0c18dd9714b9f85825c45a1c3aae685da4e6222
mkdir -p "$DIR"
if [ ! -d "$DIR/trackeverything/.git" ]; then
  git clone -q https://github.com/ayushjain1144/trackeverything.git "$DIR/trackeverything"
fi
git -C "$DIR/trackeverything" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$DIR/trackeverything" checkout -q "$COMMIT"
echo "TrackEverything at $(git -C "$DIR/trackeverything" rev-parse HEAD) (licence: CC BY-NC 4.0, non-commercial)"
# The upstream README's install steps (torch 2.8.0 + CUDA 12.9 wheels, torch_scatter, then its pinned requirements).
python -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -q "torch_scatter==2.1.2+pt28cu129" -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
python -m pip install -q -r "$DIR/trackeverything/requirements.txt"
# Our side: subject mask (rembg, MIT). 2.0.69 is the last rembg release for Python 3.10.
python -m pip install -q "rembg[gpu]==2.0.69"
python -c "import torch, torch_scatter; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "export TRACKEVERYTHING_SRC=$(cd "$DIR/trackeverything" && pwd)"
