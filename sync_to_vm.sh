#!/usr/bin/env bash
# Sync the local project directory to the GCP VM.
#
# Mirrors the source tree to ~/work/gqe-torch on the VM via rsync. The
# .venv directories, __pycache__, .git, wandb, and results are excluded so
# only source code / configs / docs are pushed.
#
# Usage:
#   ./sync_to_vm.sh             # sync (preserves remote files not in the exclude list)
#   ./sync_to_vm.sh --delete    # also delete remote files that no longer exist locally

set -euo pipefail

# ── VM coordinates (matches: gcloud compute ssh --zone "us-central1-f" \
#    "instance-20260428-114640" --project "aymantarig-project") ──────────────
PROJECT_ID="aymantarig-project"
ZONE="us-central1-f"
INSTANCE="instance-20260428-114640"
VM_USER="aymantarig_aims_ac_za"
REMOTE_DIR="~/work/gqe-torch"

LOCAL_PROJECT_DIR="$HOME/Downloads/Quantum/project/gqe-torch"

DELETE_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --delete) DELETE_FLAG="--delete" ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

REMOTE="${VM_USER}@${INSTANCE}.${ZONE}.${PROJECT_ID}:${REMOTE_DIR}/"

# The synthetic hostname above is provided by gcloud's ssh config; refresh it
# so rsync can resolve and connect without prompting.
echo "Refreshing gcloud SSH config..."
gcloud compute config-ssh --project "$PROJECT_ID" >/dev/null

echo "Ensuring remote directory exists..."
gcloud compute ssh --zone "$ZONE" "$INSTANCE" --project "$PROJECT_ID" -- \
  "mkdir -p $REMOTE_DIR"

echo "Syncing ${LOCAL_PROJECT_DIR}/ → ${REMOTE}"
cd "$LOCAL_PROJECT_DIR"

rsync -avz --human-readable --partial --progress $DELETE_FLAG \
  --exclude='.venv' \
  --exclude='.venv-torch' \
  --exclude='.venv-jax' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='wandb' \
  --exclude='results' \
  ./ "$REMOTE"

echo "Done."
