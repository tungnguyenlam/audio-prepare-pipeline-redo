#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${VERIFIER_PYTHON:-$repo_root/.venv-verify/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing verifier environment: $python_bin. See scripts/COMMANDS.md for setup." >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/verify/unsloth.py" "$@"
