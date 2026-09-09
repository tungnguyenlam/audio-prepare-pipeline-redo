#!/usr/bin/env bash
# Shared interpreter selection for lightweight standalone commands.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$1"
shift
python_bin="${AUDIO_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in .venvs/audio .venv-audio .venvs/main .venv; do
        if [[ -x "$repo_root/$candidate/bin/python" ]]; then
            python_bin="$repo_root/$candidate/bin/python"
            break
        fi
    done
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "Missing audio environment. Run bash \"$repo_root/envs/setup_worker_envs.sh\" audio or set AUDIO_PYTHON to an executable Python path." >&2
    exit 2
fi
exec "$python_bin" "$script" "$@"
