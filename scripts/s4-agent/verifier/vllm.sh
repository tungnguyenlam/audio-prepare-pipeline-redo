#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${VLLM_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/vllm/bin/python" ]]; then
        python_bin="$repo_root/.venvs/vllm/bin/python"
    elif [[ -x "$repo_root/.venv-vllm/bin/python" ]]; then
        python_bin="$repo_root/.venv-vllm/bin/python"
    elif [[ -x "$repo_root/.venvs/verify/bin/python" ]]; then
        python_bin="$repo_root/.venvs/verify/bin/python"
    elif [[ -x "$repo_root/.venvs/main/bin/python" ]]; then
        python_bin="$repo_root/.venvs/main/bin/python"
    else
        python_bin="$repo_root/.venvs/vllm/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    # Fallback to general verification python or root interpreter if .venv-vllm is absent
    python_bin="${VERIFIER_PYTHON:-python3}"
fi
exec "$python_bin" "$repo_root/scripts/s4-agent/verifier/vllm.py" "$@"
