#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../_common/isolated_launcher.sh" \
    "$script_dir/restore_voicefixer.py" VOICEFIXER_PYTHON voicefixer voicefixer "$@"
