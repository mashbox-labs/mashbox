#!/bin/bash
# Delete the VM when you are done. On-demand TPU time is billed until the VM is deleted.
set -eu
: "${PROJECT:?set PROJECT}"; ZONE=${ZONE:-asia-northeast1-b}; NAME=${NAME:-hy-nomas}
gcloud compute tpus tpu-vm delete "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet
gcloud compute tpus tpu-vm list --project "$PROJECT" --zone "$ZONE"
