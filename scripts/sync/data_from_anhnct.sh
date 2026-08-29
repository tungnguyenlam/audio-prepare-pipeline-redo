#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${SYNC_ANHNCT_HOST:-anhnct@10.148.21.113}"
REMOTE_REPO="${SYNC_ANHNCT_REPO:-Documents/tts-data-pipeline/audio-prepare-pipeline-redo}"
EXCLUDES="$REPO_ROOT/scripts/sync/data_excludes.txt"

cd "$REPO_ROOT"
mkdir -p .data
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p .data && if [ -d src/notebooks/.data ]; then rsync -a src/notebooks/.data/ .data/; fi"

rsync -avzP \
  --safe-links \
  --prune-empty-dirs \
  --exclude-from="$EXCLUDES" \
  "${REMOTE_HOST}:${REMOTE_REPO}/.data/" .data/
