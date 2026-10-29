#!/usr/bin/env bash
# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
#
# Video in, rigged GLB out. Run examples/track2rig/video-to-rig/setup.sh first, from the repository root.
# Clip: "Ours marchant à Juraparc 720p" (a brown bear walking in Juraparc, Switzerland), by Clément Bucco-Lechat,
# CC BY-SA 3.0, via Wikimedia Commons. We use its first 5 seconds, every 2nd frame: 60 frames at 12 frames per second.
set -euo pipefail
OUT=${1:-bear-example}
export TRACKEVERYTHING_SRC=${TRACKEVERYTHING_SRC:-$PWD/third_party_src/trackeverything}
mkdir -p "$OUT"
URL="https://upload.wikimedia.org/wikipedia/commons/1/16/Ours_marchant_%C3%A0_Juraparc_720p.ogv"
SHA1=956ccb9197f6c2b24aa71d7e8b4c07d256da7d0b
[ -f "$OUT/bear.ogv" ] || curl -sSfL -A "track2rig-example/0.1" -o "$OUT/bear.ogv" "$URL"
echo "$SHA1  $OUT/bear.ogv" | sha1sum -c --quiet -
# 1. video -> 3D point tracks (GPU)
.venv-video/bin/track2rig-video "$OUT/bear.ogv" -o "$OUT/tracks.npz" --max-frames 60 --frame-step 2
# 2. tracks -> rig.json + rig.glb (CPU)
.venv-rig/bin/mashbox run track2rig "$OUT/tracks.npz" -o "$OUT/rig" --fps 12 ${TRACK2RIG_WEIGHTS:+--weights "$TRACK2RIG_WEIGHTS"}
ls -l "$OUT/rig"
