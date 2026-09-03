#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8765}"
HOST="${2:-127.0.0.1}"

# Kill any existing process occupying the target port
OCCUPIED_PIDS=""
if command -v lsof >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(lsof -ti :"${PORT}" 2>/dev/null || true)
elif command -v ss >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(ss -lptn "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)
elif command -v fuser >/dev/null 2>&1; then
    OCCUPIED_PIDS=$(fuser "${PORT}/tcp" 2>/dev/null || true)
fi

if [ -n "$OCCUPIED_PIDS" ]; then
    echo "⚠️  Port ${PORT} is in use by PID(s): ${OCCUPIED_PIDS}. Terminating existing process..."
    for pid in $OCCUPIED_PIDS; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 0.5
fi

# Ensure python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Virtual environment .venv not found. Creating with uv..."
    uv venv
fi

# Detect hardware accelerators: AMD GPU vs NVIDIA GPU vs Apple Silicon vs CPU
HAS_AMD_GPU=0
HAS_NVIDIA_GPU=0

if command -v lspci >/dev/null 2>&1; then
    if lspci -nn | grep -E "VGA|3D|Display" | grep -iqE "AMD|Radeon|Advanced Micro Devices"; then
        HAS_AMD_GPU=1
    fi
    if lspci -nn | grep -E "VGA|3D|Display" | grep -iqE "NVIDIA"; then
        HAS_NVIDIA_GPU=1
    fi
fi

if [ "$HAS_AMD_GPU" -eq 0 ] && [ -c "/dev/kfd" ]; then
    HAS_AMD_GPU=1
fi

if [ "$HAS_AMD_GPU" -eq 0 ] && (command -v rocm-smi >/dev/null 2>&1 || [ -x "/opt/rocm/bin/rocm-smi" ]); then
    HAS_AMD_GPU=1
fi

if [ "$HAS_NVIDIA_GPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA_GPU=1
fi

# Hardware-specific runtime setup
if [ "$HAS_AMD_GPU" -eq 1 ]; then
    echo "🔍 Detected AMD GPU hardware."
    
    # Ensure ROCm tools in PATH if present
    if [ -d "/opt/rocm/bin" ] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
        export PATH="/opt/rocm/bin:$PATH"
    fi
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

    # Check if installed torch supports ROCm/HIP
    NEEDS_ROCM_TORCH=1
    if [ -x ".venv/bin/python" ]; then
        if .venv/bin/python -c "import torch; exit(0 if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
            NEEDS_ROCM_TORCH=0
        fi
    fi

    if [ "$NEEDS_ROCM_TORCH" -eq 1 ]; then
        echo "⚡ Configuring AMD ROCm PyTorch wheels (torch 2.13.0+rocm10, torchaudio, triton)..."
        uv pip install --no-deps \
          https://stable.repo.amd.com/rocm/pytorch/whl-next/torch/torch-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
          https://stable.repo.amd.com/rocm/pytorch/whl-next/torchaudio/torchaudio-2.11.0.2%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
          https://stable.repo.amd.com/rocm/pytorch/whl-next/triton/triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp313-cp313-linux_x86_64.whl
        
        # Ensure device kpack for gfx1200 (Navi 44 / RX 9060 XT) is present
        .venv/bin/python -c "import importlib.util; exit(0 if importlib.util.find_spec('amd_torch_device_gfx1200') else 1)" 2>/dev/null || \
          uv pip install amd-torch-device-gfx1200 2>/dev/null || true
    fi

    # Ensure torchcodec (CUDA-only) is removed so it doesn't fail on libnvrtc.so.13
    if .venv/bin/python -c "import importlib.util; exit(0 if importlib.util.find_spec('torchcodec') else 1)" 2>/dev/null; then
        uv pip uninstall torchcodec >/dev/null 2>&1 || true
    fi

elif [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
    echo "🔍 Detected NVIDIA GPU hardware."
    # Check if installed torch supports CUDA
    NEEDS_CUDA_TORCH=1
    if [ -x ".venv/bin/python" ]; then
        if .venv/bin/python -c "import torch; exit(0 if torch.cuda.is_available() and not getattr(torch.version, 'hip', None) else 1)" >/dev/null 2>&1; then
            NEEDS_CUDA_TORCH=0
        fi
    fi
    if [ "$NEEDS_CUDA_TORCH" -eq 1 ]; then
        echo "⚡ Configuring NVIDIA CUDA PyTorch stack..."
        uv pip install --reinstall "torch>=2.13.0" "torchaudio>=2.11.0"
    fi
fi

echo "🚀 Starting the shared Sonic backend on http://${HOST}:${PORT}..."
exec uv run --no-sync python scripts/start_web.py --host "$HOST" --port "$PORT"
