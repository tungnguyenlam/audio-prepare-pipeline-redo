#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${VERIFIER_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/kimi/bin/python" ]]; then
        python_bin="$repo_root/.venvs/kimi/bin/python"
    else
        python_bin="$repo_root/.venvs/kimi/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing verifier environment: $python_bin. Run: ./envs/setup_worker_envs.sh kimi" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/s4-agent/verifier/kimi.py" "$@"
