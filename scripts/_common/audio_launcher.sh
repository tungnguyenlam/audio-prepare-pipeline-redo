#!/usr/bin/env bash
# Shared interpreter selection for lightweight standalone commands.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    setup_script="$repo_root/envs/setup_worker_envs.sh"
    auto_provision="${AUTO_PROVISION:-0}"

    if [[ "$auto_provision" != "1" && -t 0 && -t 1 ]]; then
        echo "⚠️  Audio environment (.venvs/audio) is missing." >&2
        read -r -p "Would you like to provision it now? [Y/n] " response </dev/tty || response="n"
        case "${response:-y}" in
            [yY][eE][sS]|[yY]|"") auto_provision=1 ;;
            *) auto_provision=0 ;;
        esac
    fi

    if [[ "$auto_provision" == "1" ]]; then
        echo "🚀 Provisioning audio environment via: $setup_script audio..." >&2
        bash "$setup_script" audio
        for candidate in .venvs/audio .venvs/main; do
            if [[ -x "$repo_root/$candidate/bin/python" ]]; then
                python_bin="$repo_root/$candidate/bin/python"
                break
            fi
        done
    fi
fi

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "❌ Missing audio environment (.venvs/audio)." >&2
    echo "To set it up, run:" >&2
    echo "  ./envs/setup_worker_envs.sh audio" >&2
    echo "Or set AUDIO_PYTHON to an executable Python path." >&2
    exit 2
fi

exec "$python_bin" "$script" "$@"
