#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${SILERO_PYTHON:-${ASR_PYTHON:-}}"
if [[ -z "$python_bin" ]]; then
    for candidate in .venvs/vibevoice .venv-vibevoice .venvs/align .venv-align \
                     .venvs/sortformer .venv-sortformer .venvs/3dspeaker .venv-3dspeaker \
                     .venvs/diarizen .venv-diarizen .venvs/main .venv; do
        if [[ -x "$repo_root/$candidate/bin/python" ]]; then
            python_bin="$repo_root/$candidate/bin/python"
            break
        fi
    done
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "Missing Silero VAD environment with torch/torchaudio. Run: ./envs/setup_worker_envs.sh vibevoice" >&2
    echo "Or set SILERO_PYTHON to an executable Python path." >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/cleanup/vad_gate_silero.py" "$@"
