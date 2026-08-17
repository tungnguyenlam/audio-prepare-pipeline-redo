#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="vsf@10.148.21.12:~/Documents/tts-data-pipeline/audio-prepare-pipeline-redo"

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
  "${REMOTE}/" ./
