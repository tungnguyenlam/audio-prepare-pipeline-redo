#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${SEPARATION_PYTHON:-$repo_root/.venv-separation/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing separation environment: $python_bin. See scripts/COMMANDS.md for setup." >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/separate/mel_roformer.py" "$@"
