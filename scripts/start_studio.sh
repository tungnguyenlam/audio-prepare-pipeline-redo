#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8765}"
HOST="${2:-127.0.0.1}"

# Kill any existing process occupying the target port
OCCUPIED_PIDS=""
if command -v lsof >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(lsof -ti :"${PORT}" 2>/dev/null || true)
elif command -v ss >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(ss -lptn "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)
elif command -v fuser >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(fuser "${PORT}/tcp" 2>/dev/null || true)
fi

if [ -n "$OCCUPIED_PIDS" ]; then
    echo "⚠️  Port ${PORT} is in use by PID(s): ${OCCUPIED_PIDS}. Terminating existing process..."
    for pid in $OCCUPIED_PIDS; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 0.5
fi

echo "🚀 Starting SonicStudio (Interactive Exploration) on http://${HOST}:${PORT}..."
exec uv run python scripts/start_studio.py --host "$HOST" --port "$PORT"
