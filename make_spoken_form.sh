#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/scripts/_common/metadata_launcher.sh" "$repo_root/make_spoken_form.py" "$@"
