#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8080}"
HOST="${2:-127.0.0.1}"

# Kill any existing process occupying the target port
if command -v lsof >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(lsof -ti :"${PORT}" 2>/dev/null || true)
    if [ -n "$OCCUPIED_PIDS" ]; then
        echo "⚠️  Port ${PORT} is in use by PID(s): ${OCCUPIED_PIDS}. Terminating existing process..."
        for pid in $OCCUPIED_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 0.5
    fi
elif command -v fuser >/dev/null 2>&1; then
    echo "⚠️  Releasing port ${PORT} with fuser..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 0.5
fi

echo "🚀 Starting SonicStudio on http://${HOST}:${PORT}..."
exec uv run python scripts/start_web.py --host "$HOST" --port "$PORT"
