#!/usr/bin/env bash
# Pull new result files from the GCP VM down to the local repo.
#
# Mirrors only the two output trees (results/runs and results/benchmarks).
# rsync transfers only files that are missing or newer on the VM, so calling
# this repeatedly after a benchmark / training run is cheap.
#
# Usage:
#   ./pull_results.sh                 # pull both runs/ and benchmarks/
#   ./pull_results.sh runs            # pull only results/runs/
#   ./pull_results.sh benchmarks      # pull only results/benchmarks/
#   ./pull_results.sh --delete        # also delete local files no longer on VM
#
# Env overrides:
#   VM_HOST   default: aymantarig_aims_ac_za@instance-20260429-120033.us-central1-f.aymantarig-project
#   VM_PATH   default: ~/work/gqe-torch
#   LOCAL     default: directory of this script

set -euo pipefail

VM_HOST="${VM_HOST:-aymantarig_aims_ac_za@instance-20260429-120033.us-central1-f.aymantarig-project}"
VM_PATH="${VM_PATH:-~/work/gqe-torch}"
LOCAL="${LOCAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

DELETE_FLAG=""
SUBDIRS=()
for arg in "$@"; do
  case "$arg" in
    --delete) DELETE_FLAG="--delete" ;;
    runs|benchmarks) SUBDIRS+=("$arg") ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ ${#SUBDIRS[@]} -eq 0 ]; then
  SUBDIRS=(runs benchmarks)
fi

for sub in "${SUBDIRS[@]}"; do
  src="${VM_HOST}:${VM_PATH}/results/${sub}/"
  dst="${LOCAL}/results/${sub}/"
  mkdir -p "$dst"
  echo "── pulling ${sub} ──"
  echo "  from: $src"
  echo "  to:   $dst"
  rsync -avz --human-readable --partial --progress $DELETE_FLAG \
    "$src" "$dst"
done

echo "Done."
