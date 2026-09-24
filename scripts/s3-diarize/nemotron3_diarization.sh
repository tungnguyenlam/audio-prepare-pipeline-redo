#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DIARIZATION_PYTHON:-$repo_root/.venvs/nemotron3/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing model environment: $python_bin. Run: ./envs/setup_worker_envs.sh nemotron3" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/s3-diarize/nemotron3_diarization.py" "$@"
