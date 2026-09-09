#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DIARIZATION_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/diarizen/bin/python" ]]; then
        python_bin="$repo_root/.venvs/diarizen/bin/python"
    elif [[ -x "$repo_root/.venv-diarizen/bin/python" ]]; then
        python_bin="$repo_root/.venv-diarizen/bin/python"
    else
        python_bin="$repo_root/.venvs/diarizen/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing model environment: $python_bin. Run: ./scripts/setup_worker_envs.sh diarizen" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/diarize/diarizen.py" "$@"
