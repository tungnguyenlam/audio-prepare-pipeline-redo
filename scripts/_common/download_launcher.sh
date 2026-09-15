#!/usr/bin/env bash
# Shared interpreter and JavaScript-runtime selection for YouTube download commands.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$1"
shift

python_bin="${DOWNLOAD_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in .venvs/download .venv-download; do
        if [[ -x "$repo_root/$candidate/bin/python" ]]; then
            python_bin="$repo_root/$candidate/bin/python"
            break
        fi
    done
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "Missing download environment. Run bash \"$repo_root/envs/setup_worker_envs.sh\" download or set DOWNLOAD_PYTHON to an executable Python path." >&2
    exit 2
fi

# The download environment keeps Deno, Node/npm, Bun, and QuickJS in its own
# bin directory. DENO_DIR is runtime cache, so keep it under the gitignored
# artifact root rather than the user's global cache.
download_root="$(cd "$(dirname "$python_bin")/.." && pwd)"
export PATH="$download_root/bin:$PATH"
export DENO_DIR="${DENO_DIR:-$repo_root/.data/s1-download/deno}"

exec "$python_bin" "$script" "$@"
