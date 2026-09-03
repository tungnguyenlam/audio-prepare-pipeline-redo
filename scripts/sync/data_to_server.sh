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
if [ -d src/notebooks/.data ]; then
  rsync -a src/notebooks/.data/ .data/
fi
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/.data'"

rsync -avzP \
  --safe-links \
  --prune-empty-dirs \
  --exclude-from="$EXCLUDES" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  .data/ "${REMOTE_HOST}:${REMOTE_REPO}/.data/"
