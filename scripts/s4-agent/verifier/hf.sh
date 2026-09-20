#!/usr/bin/env bash
set -euo pipefail

# Add ROCm binaries to PATH if available on AMD systems
if [[ -d "/opt/rocm/bin" ]] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
    export PATH="/opt/rocm/bin:$PATH"
fi

# Ensure stable SDPA attention kernels on AMD ROCm (prevents hipErrorInvalidValue on RDNA 4 / gfx1200)
if [[ "${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-}" == "1" ]]; then
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
fi

# Sync HIP / ROCR device visibility with CUDA if specified
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${VERIFIER_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venvs/verify/bin/python" ]]; then
        python_bin="$repo_root/.venvs/verify/bin/python"
    elif [[ -x "$repo_root/.venv-verify/bin/python" ]]; then
        python_bin="$repo_root/.venv-verify/bin/python"
    elif [[ -x "$repo_root/.venvs/main/bin/python" ]]; then
        python_bin="$repo_root/.venvs/main/bin/python"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="$repo_root/.venvs/verify/bin/python"
    fi
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Missing verifier environment: $python_bin. Run: ./envs/setup_worker_envs.sh verify" >&2
    exit 2
fi
exec "$python_bin" "$repo_root/scripts/s4-agent/verifier/hf.py" "$@"
