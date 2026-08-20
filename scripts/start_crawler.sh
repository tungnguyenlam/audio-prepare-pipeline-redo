#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate virtual environment if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# HuggingFace Token for high-speed download & Pyannote diarization
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Default host and port
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8567}"

echo "========================================================"
echo " 🚀 SonicCrawl • YouTube Audio Ingestion Web Application"
echo "========================================================"
echo " 📡 Server URL: http://${HOST}:${PORT}"
echo " 📂 Audio Storage Directory: ${PROJECT_ROOT}/audio_crawl"
echo "========================================================"

# Run uvicorn server
python3 -m uvicorn src.web.app:app --host "${HOST}" --port "${PORT}" --reload
