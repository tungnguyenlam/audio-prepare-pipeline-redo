#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${VERIFIER_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/verify/bin/python" ]]; then
        python_bin="$repo_root/.venvs/verify/bin/python"
    else
        python_bin="python3"
    fi
fi
exec "$python_bin" "$repo_root/scripts/vicut_bench/evaluate_composite_probe_gate.py" "$@"
