#!/usr/bin/env bash
# Shared interpreter selection for an isolated model virtualenv.
# Usage: bash isolated_launcher.sh <script.py> <ENV_VAR> <venv_name> <setup_target> [args...]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$1"
env_var="$2"
venv_name="$3"
setup_target="$4"
shift 4

if [[ -d "/opt/rocm/bin" ]] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
    export PATH="/opt/rocm/bin:$PATH"
fi

python_bin=""
if [[ -n "${!env_var:-}" ]]; then
    python_bin="${!env_var}"
fi
if [[ -z "$python_bin" ]]; then
    for candidate in ".venvs/${venv_name}" ".venv-${venv_name}"; do
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
        echo "⚠️  Isolated environment (.venvs/${venv_name}) is missing." >&2
        read -r -p "Would you like to provision it now via ./envs/setup_worker_envs.sh ${setup_target}? [Y/n] " response </dev/tty || response="n"
        case "${response:-y}" in
            [yY][eE][sS]|[yY]|"") auto_provision=1 ;;
            *) auto_provision=0 ;;
        esac
    fi

    if [[ "$auto_provision" == "1" ]]; then
        echo "🚀 Provisioning ${venv_name} environment via: $setup_script ${setup_target}..." >&2
        bash "$setup_script" "$setup_target"
        for candidate in ".venvs/${venv_name}" ".venv-${venv_name}"; do
            if [[ -x "$repo_root/$candidate/bin/python" ]]; then
                python_bin="$repo_root/$candidate/bin/python"
                break
            fi
        done
    fi
fi

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "❌ Missing isolated environment (.venvs/${venv_name})." >&2
    echo "To set it up, run:" >&2
    echo "  ./envs/setup_worker_envs.sh ${setup_target}" >&2
    echo "Or set ${env_var} to an executable Python path." >&2
    exit 2
fi

exec "$python_bin" "$script" "$@"
