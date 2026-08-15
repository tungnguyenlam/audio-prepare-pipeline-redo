#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="vsf@10.148.21.12:~/Documents/tts-data-pipeline/audio-prepare-pipeline-redo"

cd "$REPO_ROOT"
mkdir -p .data

rsync -avzP \
  .data/ "${REMOTE}/.data/"
