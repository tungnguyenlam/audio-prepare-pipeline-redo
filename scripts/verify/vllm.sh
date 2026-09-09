#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${VLLM_PYTHON:-$repo_root/.venv-vllm/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    # Fallback to general verification python or root interpreter if .venv-vllm is absent
    python_bin="${VERIFIER_PYTHON:-python3}"
fi
exec "$python_bin" "$repo_root/scripts/verify/vllm.py" "$@"
