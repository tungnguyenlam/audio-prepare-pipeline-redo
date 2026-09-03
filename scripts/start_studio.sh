#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8765}"
HOST="${2:-127.0.0.1}"

exec "${REPO_ROOT}/scripts/start_web.sh" "$PORT" "$HOST"
