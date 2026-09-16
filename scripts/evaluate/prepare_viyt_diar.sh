#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/evaluate/prepare_viyt_diar.py"

candidates=()
if [[ -n "${AUDIO_PYTHON:-}" ]]; then
    candidates+=("${AUDIO_PYTHON}")
fi
for env_dir in .venvs/3dspeaker .venv-3dspeaker .venvs/audio .venv-audio .venvs/main .venv; do
    if [[ -x "$repo_root/$env_dir/bin/python" ]]; then
        candidates+=("$repo_root/$env_dir/bin/python")
    fi
done

python_bin=""
for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import datasets, soundfile, librosa' >/dev/null 2>&1; then
        python_bin="$candidate"
        break
    fi
done
if [[ -z "$python_bin" ]]; then
    echo "Missing an environment with datasets, soundfile, and librosa. Provision .venvs/3dspeaker or set AUDIO_PYTHON." >&2
    exit 2
fi
exec "$python_bin" "$script" "$@"
