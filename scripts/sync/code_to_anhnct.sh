#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${SYNC_ANHNCT_HOST:-anhnct@10.148.21.113}"
REMOTE_REPO="${SYNC_ANHNCT_REPO:-Documents/tts-data-pipeline/audio-prepare-pipeline-redo}"

cd "$REPO_ROOT"

rsync -avzP \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='.venv-*/' \
  --exclude='.data/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.ipynb_checkpoints/' \
  ./ "${REMOTE_HOST}:${REMOTE_REPO}/"
