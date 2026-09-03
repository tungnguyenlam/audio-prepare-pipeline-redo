#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8765}"
HOST="${2:-127.0.0.1}"

if [ ! -x .venv/bin/python ]; then
    echo "Creating the project environment..."
    uv sync
fi

# PyTorch exposes ROCm devices through its torch.cuda API. PyPI's default Linux
# wheel is CUDA-only, though, so uv must replace it with an AMD build on ROCm
# hosts. Keep this launcher-specific so the same repository can still be synced
# to NVIDIA and CPU machines.
AMD_GPU_PRESENT=0
for vendor_file in /sys/class/drm/card*/device/vendor; do
    if [ -r "$vendor_file" ] && [ "$(tr '[:upper:]' '[:lower:]' < "$vendor_file")" = "0x1002" ]; then
        AMD_GPU_PRESENT=1
        break
    fi
done

# Some containers and restricted shells expose PCI metadata but not DRM nodes.
# Use the PCI vendor files as a fallback so an AMD host does not silently keep
# a CUDA-only PyTorch wheel.
if [ "$AMD_GPU_PRESENT" -eq 0 ]; then
    for vendor_file in /sys/bus/pci/devices/*/vendor; do
        device_dir="${vendor_file%/vendor}"
        if [ -r "$vendor_file" ] \
            && [ -r "${device_dir}/class" ] \
            && [ "$(tr '[:upper:]' '[:lower:]' < "$vendor_file")" = "0x1002" ] \
            && [ "$(cut -c1-4 "${device_dir}/class")" = "0x03" ]; then
            AMD_GPU_PRESENT=1
            break
        fi
    done
fi

if [ "$AMD_GPU_PRESENT" -eq 1 ]; then
    ROCM_VERSION="${SONIC_ROCM_VERSION:-10.0.0}"
    AMD_GPU_TARGET="${SONIC_AMD_GPU_TARGET:-gfx1200}"
    TORCH_VERSION="${SONIC_TORCH_VERSION:-2.13.0}"
    TORCHAUDIO_VERSION="${SONIC_TORCHAUDIO_VERSION:-2.11.0.2}"
    if ! .venv/bin/python -c 'import sys, torch; expected = "+rocm" + sys.argv[1]; sys.exit(0 if torch.version.hip and expected in torch.__version__ else 1)' "$ROCM_VERSION" 2>/dev/null; then
        echo "AMD GPU detected; installing PyTorch ${TORCH_VERSION} for ROCm ${ROCM_VERSION} (${AMD_GPU_TARGET})..."
        UV_CACHE_DIR="${REPO_ROOT}/.data/uv-cache" uv pip install \
            --python .venv/bin/python \
            --reinstall \
            --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
            --index-strategy unsafe-best-match \
            "torch[device-${AMD_GPU_TARGET}]==${TORCH_VERSION}+rocm${ROCM_VERSION}" \
            "torchaudio==${TORCHAUDIO_VERSION}+rocm${ROCM_VERSION}" \
            'fsspec[http]<=2026.6.0'
    fi

    if ! .venv/bin/python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        if [ "${SONIC_ALLOW_CPU_FALLBACK:-0}" != "1" ]; then
            cat >&2 <<'EOF'
ERROR: A ROCm PyTorch build is installed, but the AMD GPU is not accessible.
Check that your user can access /dev/kfd and /dev/dri (normally via the render
and video groups), then log out and back in. Set SONIC_ALLOW_CPU_FALLBACK=1 to
start on CPU intentionally.
EOF
            exit 1
        fi
        echo "WARNING: ROCm GPU unavailable; starting with CPU fallback enabled."
    fi
fi

echo "Starting the shared Sonic backend on http://${HOST}:${PORT}..."
exec uv run --no-sync python scripts/start_web.py --host "$HOST" --port "$PORT"
