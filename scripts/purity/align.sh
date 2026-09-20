#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${ALIGNMENT_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/align/bin/python" ]]; then
        python_bin="$repo_root/.venvs/align/bin/python"
    elif [[ -x "$repo_root/.venv-align/bin/python" ]]; then
        python_bin="$repo_root/.venv-align/bin/python"
    elif [[ -x "$repo_root/.venvs/main/bin/python" ]]; then
        python_bin="$repo_root/.venvs/main/bin/python"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="$repo_root/.venvs/align/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing alignment environment: $python_bin. Run: ./envs/setup_worker_envs.sh align" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/purity/align.py" "$@"
