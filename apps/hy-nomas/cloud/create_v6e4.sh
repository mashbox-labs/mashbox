#!/bin/bash
# Create a TPU v6e-4 VM (4 chips, on-demand) in your own Google Cloud project, then open an SSH session with the UI port
# forwarded to your laptop. Nothing is exposed to the internet: the server binds 127.0.0.1 on the VM.
# Usage: PROJECT=<your project> ZONE=asia-northeast1-b bash cloud/create_v6e4.sh
set -eu
: "${PROJECT:?set PROJECT to your Google Cloud project id}"; ZONE=${ZONE:-asia-northeast1-b}; NAME=${NAME:-hy-nomas}
gcloud compute tpus tpu-vm create "$NAME" --project "$PROJECT" --zone "$ZONE" --accelerator-type v6e-4 --version v2-alpha-tpuv6e
echo "Created. Connect with:  gcloud compute tpus tpu-vm ssh $NAME --project $PROJECT --zone $ZONE -- -L 8790:127.0.0.1:8790"
