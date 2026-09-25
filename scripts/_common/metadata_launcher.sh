#!/usr/bin/env bash
# Select an existing interpreter for metadata-only commands; never provision packages.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
script="$1"
shift
python_bin="${AUDIO_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in .venvs/audio .venvs/main; do
        if [[ -x "$repo_root/$candidate/bin/python" ]]; then
            python_bin="$repo_root/$candidate/bin/python"
            break
        fi
    done
fi
if [[ -z "$python_bin" ]]; then
    python_bin="$(command -v python3 || true)"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo 'Python is unavailable. Set AUDIO_PYTHON to an existing Python executable; no audio environment is required.' >&2
    exit 2
fi
exec "$python_bin" "$script" "$@"
