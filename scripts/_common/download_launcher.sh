#!/usr/bin/env bash
# Shared interpreter and JavaScript-runtime selection for YouTube download commands.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$1"
shift

python_bin="${DOWNLOAD_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/download/bin/python" ]]; then
        python_bin="$repo_root/.venvs/download/bin/python"
    fi
fi

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    setup_script="$repo_root/envs/setup_worker_envs.sh"
    auto_provision="${AUTO_PROVISION:-0}"

    if [[ "$auto_provision" != "1" && -t 0 && -t 1 ]]; then
        echo "⚠️  Download environment (.venvs/download) is missing." >&2
        read -r -p "Would you like to provision it now? [Y/n] " response </dev/tty || response="n"
        case "${response:-y}" in
            [yY][eE][sS]|[yY]|"") auto_provision=1 ;;
            *) auto_provision=0 ;;
        esac
    fi

    if [[ "$auto_provision" == "1" ]]; then
        echo "🚀 Provisioning download environment via: $setup_script download" >&2
        bash "$setup_script" download
        if [[ -x "$repo_root/.venvs/download/bin/python" ]]; then
            python_bin="$repo_root/.venvs/download/bin/python"
        fi
    fi
fi

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "❌ Missing download environment (.venvs/download)." >&2
    echo "To set it up, run:" >&2
    echo "  ./envs/setup_worker_envs.sh download" >&2
    echo "Or set DOWNLOAD_PYTHON to an executable Python path." >&2
    exit 2
fi

# The download environment keeps Deno, Node/npm, Bun, and QuickJS in its own
# bin directory. DENO_DIR is runtime cache, so keep it under the gitignored
# artifact root rather than the user's global cache.
download_root="$(cd "$(dirname "$python_bin")/.." && pwd)"
export PATH="$download_root/bin:$PATH"
export DENO_DIR="${DENO_DIR:-$repo_root/.data/s1-download/deno}"

exec "$python_bin" "$script" "$@"
