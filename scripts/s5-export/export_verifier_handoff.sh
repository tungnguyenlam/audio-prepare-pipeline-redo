#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../_common/metadata_launcher.sh" "$script_dir/export_verifier_handoff.py" "$@"
