#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${SYNC_SERVER_HOST:-vsf@10.148.21.12}"
REMOTE_REPO="${SYNC_SERVER_REPO:-Documents/tts-data-pipeline/audio-prepare-pipeline-redo}"
EXCLUDES="$REPO_ROOT/scripts/sync/data_excludes.txt"

EXTRA_ARGS=()
if [ "${SYNC_DELETE:-0}" = "1" ] || [ "${SYNC_DELETE:-0}" = "true" ]; then
  EXTRA_ARGS+=(--delete)
fi

cd "$REPO_ROOT"
mkdir -p .data
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p .data && if [ -d src/notebooks/.data ]; then rsync -a src/notebooks/.data/ .data/; fi"

rsync -avzP \
  --safe-links \
  --prune-empty-dirs \
  --exclude-from="$EXCLUDES" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  "${REMOTE_HOST}:${REMOTE_REPO}/.data/" .data/
