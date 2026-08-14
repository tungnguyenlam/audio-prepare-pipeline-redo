#!/usr/bin/env bash
set -e

REMOTE="vsf@10.148.21.12:~/Documents/tts-data-pipeline/audio-prepare-pipeline-redo"

rsync -avzP data/ "${REMOTE}/data/"
rsync -avzP .data/ "${REMOTE}/.data/"
