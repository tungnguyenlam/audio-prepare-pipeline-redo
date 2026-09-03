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

# Hardware reconciliation helpers for primary and worker environments
reconcile_venv_rocm() {
    local target_venv="$1"
    local py_bin="${target_venv}/bin/python"
    [ -x "$py_bin" ] || return 0

    local needs_rocm=1
    if "$py_bin" -c "import torch; exit(0 if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        needs_rocm=0
    fi

    if [ "$needs_rocm" -eq 1 ]; then
        echo "⚡ Configuring AMD ROCm PyTorch wheels for ${target_venv}..."
        uv pip install --python "$py_bin" --no-deps \
          https://stable.repo.amd.com/rocm/pytorch/whl-next/torch/torch-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
          https://stable.repo.amd.com/rocm/pytorch/whl-next/torchaudio/torchaudio-2.11.0.2%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
          https://stable.repo.amd.com/rocm/pytorch/whl-next/triton/triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp313-cp313-linux_x86_64.whl 2>/dev/null || true
        
        # Ensure device kpack for gfx1200 (Navi 44 / RX 9060 XT) is present
        "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('amd_torch_device_gfx1200') else 1)" 2>/dev/null || \
          uv pip install --python "$py_bin" amd-torch-device-gfx1200 2>/dev/null || true
    fi

    # Ensure torchcodec (CUDA-only) is removed so it doesn't fail on libnvrtc.so.13
    if "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('torchcodec') else 1)" 2>/dev/null; then
        echo "🧹 Removing torchcodec from ${target_venv} (incompatible with ROCm)..."
        uv pip uninstall --python "$py_bin" torchcodec >/dev/null 2>&1 || true
    fi
}

reconcile_venv_cuda() {
    local target_venv="$1"
    local py_bin="${target_venv}/bin/python"
    [ -x "$py_bin" ] || return 0

    local needs_cuda=1
    if "$py_bin" -c "import torch; exit(0 if torch.cuda.is_available() and not getattr(torch.version, 'hip', None) else 1)" >/dev/null 2>&1; then
        needs_cuda=0
    fi

    if [ "$needs_cuda" -eq 1 ]; then
        echo "⚡ Configuring NVIDIA CUDA PyTorch stack for ${target_venv}..."
        uv pip install --python "$py_bin" --reinstall "torch>=2.13.0" "torchaudio>=2.11.0"
    fi
}

reconcile_diarizen_hardware() {
    local target_venv="$1"
    local py_bin="${target_venv}/bin/python"
    [ -x "$py_bin" ] || return 0

    if [ "$HAS_AMD_GPU" -eq 1 ]; then
        # On AMD (especially RDNA 4 / gfx1200), PyTorch 2.1 lacks HIP gfx1200 binaries.
        # Ensure CUDA-pinned torch is replaced with CPU fallback to prevent libnvrtc/libcuda errors.
        local has_cuda_torch=0
        if "$py_bin" -c "import torch; exit(0 if torch.cuda.is_available() and not getattr(torch.version, 'hip', None) else 1)" >/dev/null 2>&1; then
            has_cuda_torch=1
        fi
        if [ "$has_cuda_torch" -eq 1 ]; then
            echo "⚡ Reconciling ${target_venv} for AMD (swapping CUDA PyTorch for CPU/ROCm fallback)..."
            uv pip install --python "$py_bin" --reinstall \
                torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
                --index-url https://download.pytorch.org/whl/cpu
        fi
        if "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('torchcodec') else 1)" 2>/dev/null; then
            uv pip uninstall --python "$py_bin" torchcodec >/dev/null 2>&1 || true
        fi
    elif [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
        local needs_cuda=1
        if "$py_bin" -c "import torch; exit(0 if torch.cuda.is_available() and not getattr(torch.version, 'hip', None) else 1)" >/dev/null 2>&1; then
            needs_cuda=0
        fi
        if [ "$needs_cuda" -eq 1 ]; then
            echo "⚡ Reconciling ${target_venv} for NVIDIA CUDA..."
            uv pip install --python "$py_bin" --reinstall \
                torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
                --index-url https://download.pytorch.org/whl/cu121
        fi
    fi
}

# Hardware-specific runtime setup
if [ "$HAS_AMD_GPU" -eq 1 ]; then
    echo "🔍 Detected AMD GPU hardware."
    
    # Ensure ROCm tools in PATH if present
    if [ -d "/opt/rocm/bin" ] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
        export PATH="/opt/rocm/bin:$PATH"
    fi
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

    # Reconcile primary .venv
    reconcile_venv_rocm ".venv"

    # Reconcile any existing isolated worker environments
    for venv_path in ".venv-sortformer" ".venv-3dspeaker" ".venv-vibevoice"; do
        if [ -d "$venv_path" ]; then
            reconcile_venv_rocm "$venv_path"
        fi
    done

    if [ -d ".venv-diarizen" ]; then
        reconcile_diarizen_hardware ".venv-diarizen"
    fi

elif [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
    echo "🔍 Detected NVIDIA GPU hardware."

    # Reconcile primary .venv
    reconcile_venv_cuda ".venv"

    # Reconcile any existing isolated worker environments
    for venv_path in ".venv-sortformer" ".venv-3dspeaker" ".venv-vibevoice"; do
        if [ -d "$venv_path" ]; then
            reconcile_venv_cuda "$venv_path"
        fi
    done

    if [ -d ".venv-diarizen" ]; then
        reconcile_diarizen_hardware ".venv-diarizen"
    fi
fi

echo "🚀 Starting the shared Sonic backend on http://${HOST}:${PORT}..."
exec uv run --no-sync python scripts/start_web.py --host "$HOST" --port "$PORT"
