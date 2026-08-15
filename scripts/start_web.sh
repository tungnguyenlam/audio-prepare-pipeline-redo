#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Default mode is pipeline (large-scale engine)
MODE="pipeline"
PORT=""
HOST="127.0.0.1"

# Parse arguments: start_web.sh [studio|pipeline] [port] [host]
if [ "$1" == "studio" ] || [ "$1" == "pipeline" ]; then
    MODE="$1"
    PORT="${2}"
    HOST="${3:-127.0.0.1}"
elif [ "$1" == "--mode" ]; then
    MODE="${2:-pipeline}"
    PORT="${3}"
    HOST="${4:-127.0.0.1}"
else
    # First argument is port if numeric, else keep default
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        PORT="$1"
        HOST="${2:-127.0.0.1}"
    fi
fi

if [ "$MODE" == "studio" ]; then
    PORT="${PORT:-8080}"
    echo "🎙️ Launching SonicStudio (Interactive Exploration Studio) on http://${HOST}:${PORT}..."
    exec bash "$REPO_ROOT/scripts/start_studio.sh" "$PORT" "$HOST"
else
    PORT="${PORT:-8081}"
    echo "⚡ Launching SonicPipeline (Large-Scale Batch Engine) on http://${HOST}:${PORT}..."
    exec bash "$REPO_ROOT/scripts/start_pipeline.sh" "$PORT" "$HOST"
fi
