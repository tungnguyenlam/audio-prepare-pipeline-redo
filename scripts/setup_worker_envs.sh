#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

print_usage() {
    echo "Usage: $0 [all|sortformer|3dspeaker|vibevoice|diarizen|status] [--force]"
    echo ""
    echo "Device-agnostic environment provisioner for Sonic pipeline isolated models."
    echo ""
    echo "Targets:"
    echo "  all          Provision/reconcile all isolated worker environments"
    echo "  sortformer   NeMo Sortformer & Clustering (.venv-sortformer, Python 3.13)"
    echo "  3dspeaker    ModelScope 3D-Speaker (.venv-3dspeaker, Python 3.13)"
    echo "  vibevoice    VibeVoice-ASR purity verifier (.venv-vibevoice, Python 3.13)"
    echo "  diarizen     DiariZen WavLM (.venv-diarizen, Python 3.10)"
    echo "  status       Display environment and hardware acceleration status"
    echo ""
    echo "Options:"
    echo "  --force      Recreate target environment from scratch if it already exists"
    echo "  -h, --help   Show this help message"
}

TARGET="${1:-all}"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help) print_usage; exit 0 ;;
    esac
done

# Ensure primary .venv exists first
if [ ! -d ".venv" ]; then
    echo "📦 Primary virtual environment .venv not found. Bootstrapping with uv..."
    uv venv
    uv sync
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

if [ "$HAS_AMD_GPU" -eq 1 ]; then
    HW_DESC="AMD ROCm GPU"
    if [ -d "/opt/rocm/bin" ] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
        export PATH="/opt/rocm/bin:$PATH"
    fi
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
elif [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
    HW_DESC="NVIDIA CUDA GPU"
else
    HW_DESC="CPU / Apple Silicon"
fi

echo "🔍 Detected compute hardware: ${HW_DESC}"

reconcile_py313_hardware() {
    local target_venv="$1"
    local py_bin="${target_venv}/bin/python"
    [ -x "$py_bin" ] || return 0

    if [ "$HAS_AMD_GPU" -eq 1 ]; then
        local needs_rocm=1
        if "$py_bin" -c "import torch; exit(0 if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
            needs_rocm=0
        fi

        if [ "$needs_rocm" -eq 1 ]; then
            echo "⚡ Installing AMD ROCm PyTorch wheels into ${target_venv}..."
            uv pip install --python "$py_bin" --no-deps \
              https://stable.repo.amd.com/rocm/pytorch/whl-next/torch/torch-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
              https://stable.repo.amd.com/rocm/pytorch/whl-next/torchaudio/torchaudio-2.11.0.2%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
              https://stable.repo.amd.com/rocm/pytorch/whl-next/triton/triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp313-cp313-linux_x86_64.whl 2>/dev/null || true
            
            "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('amd_torch_device_gfx1200') else 1)" 2>/dev/null || \
              uv pip install --python "$py_bin" amd-torch-device-gfx1200 2>/dev/null || true
        fi

        if "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('torchcodec') else 1)" 2>/dev/null; then
            echo "🧹 Removing torchcodec from ${target_venv} (incompatible with ROCm)..."
            uv pip uninstall --python "$py_bin" torchcodec >/dev/null 2>&1 || true
        fi

    elif [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
        local needs_cuda=1
        if "$py_bin" -c "import torch; exit(0 if torch.cuda.is_available() and not getattr(torch.version, 'hip', None) else 1)" >/dev/null 2>&1; then
            needs_cuda=0
        fi
        if [ "$needs_cuda" -eq 1 ]; then
            echo "⚡ Configuring NVIDIA CUDA PyTorch stack for ${target_venv}..."
            uv pip install --python "$py_bin" --reinstall "torch>=2.13.0" "torchaudio>=2.11.0"
        fi
    fi
}

setup_sortformer() {
    local venv_dir=".venv-sortformer"
    echo ""
    echo "========================================================"
    echo "  Setting up Sortformer & Clustering worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating virtual environment ${venv_dir} from .venv..."
        uv venv --python .venv/bin/python "$venv_dir"
        echo "📥 Syncing project dependencies..."
        UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --frozen --no-dev
    fi

    echo "📦 Installing Sortformer requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r requirements-sortformer.txt

    echo "⚙️ Reconciling hardware stack..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying Sortformer installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP' if getattr(torch.version, 'hip', None) else ('CUDA' if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import nemo.collections.asr.models as nemo_asr
print('   -> NeMo ASR collection: successfully loaded')
"
    echo "🎉 ${venv_dir} ready!"
}

setup_3dspeaker() {
    local venv_dir=".venv-3dspeaker"
    echo ""
    echo "========================================================"
    echo "  Setting up 3D-Speaker worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating virtual environment ${venv_dir} from .venv..."
        uv venv --python .venv/bin/python "$venv_dir"
        echo "📥 Syncing project dependencies..."
        UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --frozen --no-dev
    fi

    echo "📦 Installing 3D-Speaker requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r requirements-3dspeaker.txt

    echo "⚙️ Reconciling hardware stack..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying 3D-Speaker installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP' if getattr(torch.version, 'hip', None) else ('CUDA' if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import modelscope
print(f'   -> ModelScope: {modelscope.__version__} successfully loaded')
"
    echo "🎉 ${venv_dir} ready!"
}

setup_vibevoice() {
    local venv_dir=".venv-vibevoice"
    echo ""
    echo "========================================================"
    echo "  Setting up VibeVoice-ASR worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating virtual environment ${venv_dir} from .venv..."
        uv venv --python .venv/bin/python "$venv_dir"
        echo "📥 Syncing project dependencies..."
        UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --frozen --no-dev
    fi

    echo "📦 Installing VibeVoice requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r requirements-vibevoice.txt

    echo "⚙️ Reconciling hardware stack..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying VibeVoice installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP' if getattr(torch.version, 'hip', None) else ('CUDA' if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import transformers
print(f'   -> Transformers: {transformers.__version__} successfully loaded')
"
    echo "🎉 ${venv_dir} ready!"
}

setup_diarizen() {
    local venv_dir=".venv-diarizen"
    echo ""
    echo "========================================================"
    echo "  Setting up DiariZen worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Ensuring Python 3.10 is installed via uv..."
        uv python install 3.10
        echo "📦 Creating Python 3.10 virtual environment ${venv_dir}..."
        uv venv --python 3.10 "$venv_dir"
    fi

    local py_bin="${venv_dir}/bin/python"

    if [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
        echo "⚡ Installing NVIDIA CUDA PyTorch 2.1.1 stack into ${venv_dir}..."
        uv pip install --python "$py_bin" \
            torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
            --index-url https://download.pytorch.org/whl/cu121
    else
        echo "⚡ Installing CPU/compatible PyTorch 2.1.1 stack into ${venv_dir}..."
        uv pip install --python "$py_bin" \
            torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
            --index-url https://download.pytorch.org/whl/cpu
    fi

    echo "📦 Installing DiariZen requirements..."
    uv pip install --python "$py_bin" -r requirements-diarizen.txt

    if [ "$HAS_AMD_GPU" -eq 1 ]; then
        if "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('torchcodec') else 1)" 2>/dev/null; then
            echo "🧹 Removing torchcodec from ${venv_dir}..."
            uv pip uninstall --python "$py_bin" torchcodec >/dev/null 2>&1 || true
        fi
    fi

    echo "✅ Verifying DiariZen installation..."
    "${venv_dir}/bin/python" -c "
import torch, psutil, accelerate
from diarizen.pipelines.inference import DiariZenPipeline
dev_type = 'ROCm/HIP' if getattr(torch.version, 'hip', None) else ('CUDA' if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
print('   -> DiariZenPipeline: successfully loaded')
"
    echo "🎉 ${venv_dir} ready!"
}

status_report() {
    echo ""
    echo "========================================================"
    echo "  Virtual Environments Status Report (${HW_DESC})"
    echo "========================================================"
    local venvs=(".venv" ".venv-sortformer" ".venv-3dspeaker" ".venv-vibevoice" ".venv-diarizen")
    for v in "${venvs[@]}"; do
        if [ -x "${v}/bin/python" ]; then
            local info
            info=$("${v}/bin/python" -c "
import sys, importlib.util
try:
    import torch
    hip = getattr(torch.version, 'hip', None)
    cuda = torch.cuda.is_available()
    dev = f'ROCm {hip}' if hip else (f'CUDA {torch.version.cuda}' if cuda else 'CPU')
    tver = torch.__version__
except Exception as e:
    dev = 'Torch Error'
    tver = 'N/A'
codec = 'torchcodec' if importlib.util.find_spec('torchcodec') else 'no-codec'
pyver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
print(f'Python {pyver} | Torch {tver} ({dev}) | {codec}')
" 2>/dev/null || echo "Corrupt or uninitialized")
            printf "  %-18s -> ✅ %s\n" "$v" "$info"
        else
            printf "  %-18s -> ❌ Not created\n" "$v"
        fi
    done
    echo "========================================================"
}

case "$TARGET" in
    sortformer|clustering|nemo)
        setup_sortformer
        ;;
    3dspeaker|threed|threed-speaker)
        setup_3dspeaker
        ;;
    vibevoice|vibe)
        setup_vibevoice
        ;;
    diarizen)
        setup_diarizen
        ;;
    status|check)
        status_report
        ;;
    all)
        setup_sortformer
        setup_3dspeaker
        setup_vibevoice
        setup_diarizen
        status_report
        ;;
    *)
        echo "❌ Unknown target: $TARGET"
        print_usage
        exit 1
        ;;
esac
