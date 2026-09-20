#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DIARIZATION_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/pyannote/bin/python" ]]; then
        python_bin="$repo_root/.venvs/pyannote/bin/python"
    elif [[ -x "$repo_root/.venv-pyannote/bin/python" ]]; then
        python_bin="$repo_root/.venv-pyannote/bin/python"
    elif [[ -x "$repo_root/.venvs/main/bin/python" ]]; then
        python_bin="$repo_root/.venvs/main/bin/python"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="$repo_root/.venvs/pyannote/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing diarization environment: $python_bin. Run: ./envs/setup_worker_envs.sh pyannote" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/speaker/score.py" "$@"
