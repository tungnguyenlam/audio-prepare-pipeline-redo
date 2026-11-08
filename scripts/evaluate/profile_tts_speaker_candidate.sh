#!/usr/bin/env bash
set -euo pipefail

# scripts/evaluate/profile_tts_speaker_candidate.sh
# Wrapper script for profile_tts_speaker_candidate.py using main virtualenv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAIN_PYTHON="${REPO_ROOT}/.venvs/main/bin/python"
TARGET_SCRIPT="${REPO_ROOT}/scripts/evaluate/profile_tts_speaker_candidate.py"

if [[ ! -f "${MAIN_PYTHON}" ]]; then
    echo "ERROR: main virtualenv python not found at ${MAIN_PYTHON}" >&2
    exit 1
fi

exec "${MAIN_PYTHON}" "${TARGET_SCRIPT}" "$@"
