#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DIARIZATION_PYTHON:-$repo_root/.venv-diarizen/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing model environment: $python_bin. Run: ./scripts/setup_worker_envs.sh diarizen" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/diarize/diarizen.py" "$@"
