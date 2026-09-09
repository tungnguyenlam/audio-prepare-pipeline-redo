#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -z "${GEMINI_API_KEY:-}" ]] && [[ -f "$repo_root/.env" ]]; then
    env_key="$(grep -E '^[[:space:]]*GEMINI_API_KEY=' "$repo_root/.env" | head -n1 | cut -d'=' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["'\'']//' -e 's/["'\'']$//' || true)"
    if [[ -n "$env_key" ]]; then
        export GEMINI_API_KEY="$env_key"
    fi
fi

python_bin="${VERIFIER_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venv-verify/bin/python" ]]; then
        python_bin="$repo_root/.venv-verify/bin/python"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="$repo_root/.venv-verify/bin/python"
    fi
fi

if [[ ! -x "$python_bin" ]]; then
    echo "Missing verifier environment: $python_bin. See scripts/COMMANDS.md for setup." >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/verify/freeform/gemini.py" "$@"
