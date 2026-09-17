#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${SILERO_PYTHON:-${ASR_PYTHON:-}}"
if [[ -z "$python_bin" ]]; then
    for candidate in .venvs/vibevoice .venv-vibevoice .venvs/align .venv-align; do
        if [[ -x "$repo_root/$candidate/bin/python" ]]; then
            python_bin="$repo_root/$candidate/bin/python"
            break
        fi
    done
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "Set SILERO_PYTHON to an existing Python environment with torch, torchaudio, numpy, and soundfile." >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/evaluate/silero_jit.py" "$@"
