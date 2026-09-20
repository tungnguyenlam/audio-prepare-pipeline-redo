#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${VERIFIER_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/minicpmo/bin/python" ]]; then
        python_bin="$repo_root/.venvs/minicpmo/bin/python"
    elif [[ -x "$repo_root/.venv-minicpmo/bin/python" ]]; then
        python_bin="$repo_root/.venv-minicpmo/bin/python"
    else
        python_bin="$repo_root/.venvs/minicpmo/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing verifier environment: $python_bin. Run: ./envs/setup_worker_envs.sh minicpmo" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/s4-agent/verifier/minicpm.py" "$@"
