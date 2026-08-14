#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="vsf@10.148.21.12:~/Documents/tts-data-pipeline/audio-prepare-pipeline-redo"

cd "$REPO_ROOT"

mkdir -p data .data benchmarks temp

rsync -avzP data/ "${REMOTE}/data/"
rsync -avzP .data/ "${REMOTE}/.data/"
rsync -avzP benchmarks/ "${REMOTE}/benchmarks/"
rsync -avzP temp/ "${REMOTE}/temp/"
