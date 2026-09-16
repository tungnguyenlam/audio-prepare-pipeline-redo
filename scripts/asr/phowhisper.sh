#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Ensure ROCm binaries are on PATH if running on an AMD GPU system
if [[ -d "/opt/rocm/bin" ]] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
    export PATH="/opt/rocm/bin:$PATH"
fi

python_bin="${ASR_PYTHON:-${PHOWHISPER_PYTHON:-${WHISPER_PYTHON:-}}}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/vibevoice/bin/python" ]]; then
        python_bin="$repo_root/.venvs/vibevoice/bin/python"
    elif [[ -x "$repo_root/.venvs/main/bin/python" ]]; then
        python_bin="$repo_root/.venvs/main/bin/python"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="python3"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing Python environment for PhoWhisper: $python_bin." >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/asr/phowhisper.py" "$@"
