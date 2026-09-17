#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

print_usage() {
    echo "Usage: $0 [all|core|workers|main|download|audio|separation|pyannote|verify|align|sortformer|3dspeaker|vibevoice|diarizen|minicpmo|kimi|status] [--force]"
    echo ""
    echo "Device-agnostic environment provisioner for audio processing models."
    echo ""
    echo "Core Pipeline Targets:"
    echo "  core         Provision all core environments (download, audio, separation, pyannote, verify, align)"
    echo "  main         Reconcile hardware acceleration for main environment (.venvs/main)"
    echo "  separation   Demucs, BS-RoFormer, Mel-RoFormer (.venvs/separation, Python 3.13)"
    echo "  pyannote     Pyannote 3.1 & Community-1 diarization/scoring (.venvs/pyannote, Python 3.13)"
    echo "  verify       HF, Whisper, Gemma verifiers (.venvs/verify, Python 3.13)"
    echo "  align        Whisper-timestamped alignment (.venvs/align, Python 3.13)"
    echo "  download     YouTube downloader and JavaScript runtimes (.venvs/download, Python 3.13)"
    echo "  audio        Lightweight audio utilities (.venvs/audio, Python 3.13)"
    echo ""
    echo "Isolated Worker Targets:"
    echo "  workers      Provision all isolated worker environments"
    echo "  sortformer   NeMo Sortformer & Clustering (.venvs/sortformer, Python 3.13)"
    echo "  3dspeaker    ModelScope 3D-Speaker (.venvs/3dspeaker, Python 3.13)"
    echo "  vibevoice    VibeVoice-ASR purity verifier (.venvs/vibevoice, Python 3.13)"
    echo "  diarizen     DiariZen WavLM (.venvs/diarizen, Python 3.10)"
    echo "  minicpmo     MiniCPM-o 4.5 / 2.6 verifier (.venvs/minicpmo, Python 3.11)"
    echo "  kimi         Kimi-Audio verifier (.venvs/kimi, Python 3.11)"
    echo ""
    echo "General Targets:"
    echo "  all          Provision/reconcile core + worker environments"
    echo "  status       Display environment and hardware acceleration status"
    echo ""
    echo "Options:"
    echo "  --force      Recreate target environment from scratch if it already exists"
    echo "  -h, --help   Show this help message"
}

TARGET="${1:-status}"
FORCE=0

SILERO_MODEL_URL="https://raw.githubusercontent.com/snakers4/silero-vad/41f03a954b841327835dea1ddb7bb28ae23ddc2c/src/silero_vad/data/silero_vad.jit"
SILERO_MODEL_SHA256="e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

ensure_silero_model() {
    local cache_root="${HOME}/.cache"
    local model_path="${cache_root}/silero-vad/silero_vad.jit"
    local model_dir
    model_dir="$(dirname "$model_path")"
    mkdir -p "$model_dir"

    if [ -f "$model_path" ] && [ "$(sha256_file "$model_path")" = "$SILERO_MODEL_SHA256" ]; then
        echo "✅ Silero JIT cached at ${model_path}"
        return 0
    fi

    local temp_path
    temp_path="$(mktemp "${model_path}.tmp.XXXXXX")"
    echo "⬇️  Downloading Silero JIT model to ${model_path}..."
    if command -v curl >/dev/null 2>&1; then
        if ! curl --fail --location --retry 3 --connect-timeout 20 --output "$temp_path" "$SILERO_MODEL_URL"; then
            rm -f "$temp_path"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget --no-verbose --tries=3 --timeout=20 --output-document "$temp_path" "$SILERO_MODEL_URL"; then
            rm -f "$temp_path"
            return 1
        fi
    else
        rm -f "$temp_path"
        echo "Missing curl or wget; cannot download the Silero JIT model." >&2
        return 1
    fi

    if [ "$(sha256_file "$temp_path")" != "$SILERO_MODEL_SHA256" ]; then
        rm -f "$temp_path"
        echo "Silero JIT model checksum mismatch; refusing to install it." >&2
        return 1
    fi
    mv -f "$temp_path" "$model_path"
    chmod 0644 "$model_path"
    echo "✅ Silero JIT model ready at ${model_path}"
}

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help) print_usage; exit 0 ;;
    esac
done

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

get_cuda_wheel_index() {
    if [ -n "${CUDA_INDEX_URL:-}" ]; then
        echo "$CUDA_INDEX_URL"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "https://download.pytorch.org/whl/cu128"
        return
    fi
    local cuda_ver
    cuda_ver=$(nvidia-smi | sed -n "s/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p" | head -n 1)
    case "$cuda_ver" in
        13.*) echo "" ;; # Default PyPI hosts CUDA 13.0
        12.8*) echo "https://download.pytorch.org/whl/cu128" ;;
        12.7*|12.6*) echo "https://download.pytorch.org/whl/cu126" ;;
        12.5*|12.4*) echo "https://download.pytorch.org/whl/cu124" ;;
        12.1*|12.2*|12.3*) echo "https://download.pytorch.org/whl/cu121" ;;
        11.8*) echo "https://download.pytorch.org/whl/cu118" ;;
        *) echo "https://download.pytorch.org/whl/cu128" ;;
    esac
}

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
            uv pip install --python "$py_bin" \
              --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/ \
              --extra-index-url https://stable.repo.amd.com/rocm/pytorch/whl-next/ \
              --index-strategy unsafe-best-match \
              "torch==2.13.0+rocm10.0.0" \
              "torchaudio==2.11.0.2+rocm10.0.0" \
              "triton==3.8.0+git4cff872c.rocm10.0.0" \
              "rocm==10.0.0" \
              "rocm-sdk-core==10.0.0" \
              "rocm-sdk-libraries==10.0.0" \
              "rocm-sdk-device-gfx1200==10.0.0" \
              "amd-torch-device-gfx1200==2.13.0+rocm10.0.0"
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
            local index_url
            index_url=$(get_cuda_wheel_index)
            if [ -n "$index_url" ]; then
                echo "⚡ Installing NVIDIA CUDA PyTorch stack (${index_url}) into ${target_venv}..."
                uv pip install --python "$py_bin" \
                    --index-url "$index_url" \
                    --upgrade "torch>=2.4.0" "torchaudio>=2.4.0"
            else
                echo "⚡ Configuring NVIDIA CUDA PyTorch stack for ${target_venv}..."
                uv pip install --python "$py_bin" --upgrade "torch>=2.4.0" "torchaudio>=2.4.0"
            fi
        fi

        if "$py_bin" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
            local gpu_name
            gpu_name=$("$py_bin" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "CUDA GPU")
            echo "   -> CUDA acceleration verified: ${gpu_name}"
        fi
    fi
}

setup_main() {
    local venv_dir=".venvs/main"
    echo ""
    echo "========================================================"
    echo "  Reconciling Main environment (${venv_dir})"
    echo "========================================================"

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating Python 3.13 virtual environment ${venv_dir} via uv..."
        uv venv --python 3.13 "$venv_dir"
        uv sync
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"
    ln -sfn "$venv_dir" ".venv"

    echo "✅ Verifying Main installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
"
    echo "🎉 ${venv_dir} reconciled!"
}

setup_audio() {
    local venv_dir=".venvs/audio"
    echo ""
    echo "========================================================"
    echo "  Setting up Core Audio environment (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating Python 3.13 virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "📦 Installing audio requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-audio.txt"
    ensure_silero_model
    ln -sfn "$venv_dir" ".venv-audio"
    echo "🎉 ${venv_dir} ready!"
}

setup_download() {
    local setup_args=()
    if [ "$FORCE" -eq 1 ]; then
        setup_args+=(--force)
    fi
    bash "$REPO_ROOT/envs/setup_download_env.sh" "${setup_args[@]}"
}

setup_separation() {
    local venv_dir=".venvs/separation"
    echo ""
    echo "========================================================"
    echo "  Setting up Separation environment (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating Python 3.13 virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing Separation requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-separation.txt"

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying Separation installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import demucs, mel_band_roformer, bs_roformer
print('   -> Demucs, MelBandRoformer, BSRoformer: successfully loaded')
"
    ln -sfn "$venv_dir" ".venv-separation"
    echo "🎉 ${venv_dir} ready!"
}

setup_pyannote() {
    local venv_dir=".venvs/pyannote"
    echo ""
    echo "========================================================"
    echo "  Setting up Pyannote environment (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating Python 3.13 virtual environment ${venv_dir}......"
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing Pyannote requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-pyannote.txt"

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying Pyannote installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import pyannote.audio
print('   -> Pyannote Audio: successfully loaded')
"
    ln -sfn "$venv_dir" ".venv-pyannote"
    echo "🎉 ${venv_dir} ready!"
}

setup_verify() {
    local venv_dir=".venvs/verify"
    echo ""
    echo "========================================================"
    echo "  Setting up Verifier environment (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating Python 3.13 virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing verifier requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-verify.txt"
    if [ "$HAS_AMD_GPU" -eq 1 ]; then
        echo "⚡ Installing CPU torchvision for Gemma multimodal processor on ROCm..."
        uv pip install --python "${venv_dir}/bin/python" "torchvision==0.28.0+cpu" --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1 || true
    fi

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying Verifier installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import transformers, peft, accelerate
print('   -> Transformers, PEFT, Accelerate: successfully loaded')
"
    ln -sfn "$venv_dir" ".venv-verify"
    echo "🎉 ${venv_dir} ready!"
}

setup_align() {
    local venv_dir=".venvs/align"
    echo ""
    echo "========================================================"
    echo "  Setting up Alignment environment (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating Python 3.13 virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing alignment requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-align.txt"

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying Alignment installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import whisper_timestamped
print('   -> Whisper-timestamped: successfully loaded')
"
    ensure_silero_model
    ln -sfn "$venv_dir" ".venv-align"
    echo "🎉 ${venv_dir} ready!"
}

setup_sortformer() {
    local venv_dir=".venvs/sortformer"
    echo ""
    echo "========================================================"
    echo "  Setting up Sortformer & Clustering worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing Sortformer requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-sortformer.txt"

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying Sortformer installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import nemo.collections.asr.models as nemo_asr
print('   -> NeMo ASR collection: successfully loaded')
"
    ln -sfn "$venv_dir" ".venv-sortformer"
    echo "🎉 ${venv_dir} ready!"
}

setup_3dspeaker() {
    local venv_dir=".venvs/3dspeaker"
    echo ""
    echo "========================================================"
    echo "  Setting up 3D-Speaker worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing 3D-Speaker requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-3dspeaker.txt"

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying 3D-Speaker installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import modelscope
print(f'   -> ModelScope: {modelscope.__version__} successfully loaded')
"
    ln -sfn "$venv_dir" ".venv-3dspeaker"
    echo "🎉 ${venv_dir} ready!"
}

setup_vibevoice() {
    local venv_dir=".venvs/vibevoice"
    echo ""
    echo "========================================================"
    echo "  Setting up VibeVoice-ASR worker (${venv_dir})"
    echo "========================================================"

    if [ "$FORCE" -eq 1 ] && [ -d "$venv_dir" ]; then
        echo "🗑️  Removing existing ${venv_dir} (--force)..."
        rm -rf "$venv_dir"
    fi

    if [ ! -d "$venv_dir" ]; then
        echo "📦 Creating virtual environment ${venv_dir}..."
        uv venv --python 3.13 "$venv_dir"
    fi

    echo "⚙️ Configuring hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "📦 Installing VibeVoice requirements..."
    uv pip install --python "${venv_dir}/bin/python" -r "$REPO_ROOT/envs/requirements-vibevoice.txt"

    echo "⚙️ Re-verifying hardware acceleration..."
    reconcile_py313_hardware "$venv_dir"

    echo "✅ Verifying VibeVoice installation..."
    "${venv_dir}/bin/python" -c "
import torch
dev_type = 'ROCm/HIP: ' + torch.cuda.get_device_name(0) if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else ('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import transformers
print(f'   -> Transformers: {transformers.__version__} successfully loaded')
import whisper
print(f'   -> Whisper: {whisper.__version__} successfully loaded')
import whisper_timestamped
print(f'   -> whisper-timestamped: {whisper_timestamped.__version__} successfully loaded')
import librosa
print(f'   -> Librosa: {librosa.__version__} successfully loaded')
"
    ensure_silero_model
    ln -sfn "$venv_dir" ".venv-vibevoice"
    echo "🎉 ${venv_dir} ready!"
}

setup_diarizen() {
    local venv_dir=".venvs/diarizen"
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

    if [ "$HAS_AMD_GPU" -eq 1 ]; then
        echo "⚡ Installing AMD ROCm PyTorch wheels into ${venv_dir}..."
        uv pip install --python "$py_bin" \
          --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/ \
          --extra-index-url https://stable.repo.amd.com/rocm/pytorch/whl-next/ \
          --index-strategy unsafe-best-match \
          "torch==2.13.0+rocm10.0.0" \
          "torchaudio==2.11.0.2+rocm10.0.0" \
          "triton==3.8.0+git4cff872c.rocm10.0.0" \
          "rocm==10.0.0" \
          "rocm-sdk-core==10.0.0" \
          "rocm-sdk-libraries==10.0.0" \
          "rocm-sdk-device-gfx1200==10.0.0" \
          "amd-torch-device-gfx1200==2.13.0+rocm10.0.0"

        echo "📦 Installing DiariZen requirements..."
        uv pip install --python "$py_bin" -r "$REPO_ROOT/envs/requirements-diarizen.txt"

        if "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('torchcodec') else 1)" 2>/dev/null; then
            echo "🧹 Removing torchcodec from ${venv_dir}... (incompatible with ROCm)"
            uv pip uninstall --python "$py_bin" torchcodec >/dev/null 2>&1 || true
        fi
        if "$py_bin" -c "import importlib.util; exit(0 if importlib.util.find_spec('torchvision') else 1)" 2>/dev/null; then
            uv pip uninstall --python "$py_bin" torchvision >/dev/null 2>&1 || true
        fi
    elif [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
        local index_url
        index_url=$(get_cuda_wheel_index)
        [ -z "$index_url" ] && index_url="https://download.pytorch.org/whl/cu121"
        echo "⚡ Installing NVIDIA CUDA PyTorch stack into ${venv_dir}..."
        uv pip install --python "$py_bin" \
            "torch>=2.1.1,<2.5.0" "torchvision" "torchaudio" \
            --index-url "$index_url"

        echo "📦 Installing DiariZen requirements..."
        uv pip install --python "$py_bin" --extra-index-url "$index_url" -r "$REPO_ROOT/envs/requirements-diarizen.txt"
    else
        echo "⚡ Installing CPU PyTorch stack into ${venv_dir}..."
        uv pip install --python "$py_bin" \
            "torch>=2.1.1" "torchvision" "torchaudio" \
            --index-url https://download.pytorch.org/whl/cpu

        echo "📦 Installing DiariZen requirements..."
        uv pip install --python "$py_bin" --extra-index-url https://download.pytorch.org/whl/cpu -r "$REPO_ROOT/envs/requirements-diarizen.txt"
    fi

    echo "✅ Verifying DiariZen installation..."
    "$py_bin" -c "
import torchaudio
if not hasattr(torchaudio, 'AudioMetaData'):
    class AudioMetaData:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int, bits_per_sample: int, encoding: str):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    torchaudio.AudioMetaData = AudioMetaData
import torch, psutil, accelerate
from diarizen.pipelines.inference import DiariZenPipeline
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
dev_type = f'ROCm/HIP: {dev_name}' if getattr(torch.version, 'hip', None) and torch.cuda.is_available() else (f'CUDA: {dev_name}' if torch.cuda.is_available() else 'CPU')
print(f'   -> Torch: {torch.__version__} ({dev_type})')
print('   -> DiariZenPipeline: successfully loaded')
"
    ln -sfn "$venv_dir" ".venv-diarizen"
    echo "🎉 ${venv_dir} ready!"
}

setup_minicpmo() {
    bash "$REPO_ROOT/envs/setup_minicpmo_env.sh" "$@"
    ln -sfn ".venvs/minicpmo" ".venv-minicpmo"
}

setup_kimi() {
    bash "$REPO_ROOT/envs/setup_kimi_env.sh" "$@"
    ln -sfn ".venvs/kimi" ".venv-kimi"
}

setup_core() {
    setup_download
    setup_audio
    setup_separation
    setup_pyannote
    setup_verify
    setup_align
}

setup_workers() {
    setup_sortformer
    setup_3dspeaker
    setup_vibevoice
    setup_diarizen
    setup_minicpmo
}

status_report() {
    echo ""
    echo "========================================================"
    echo "  Virtual Environments Status Report (${HW_DESC})"
    echo "========================================================"
    local venvs=(
        ".venvs/main"
        ".venvs/download"
        ".venvs/audio"
        ".venvs/separation"
        ".venvs/pyannote"
        ".venvs/verify"
        ".venvs/align"
        ".venvs/sortformer"
        ".venvs/3dspeaker"
        ".venvs/vibevoice"
        ".venvs/diarizen"
        ".venvs/minicpmo"
        ".venvs/kimi"
    )
    for v in "${venvs[@]}"; do
        if [ -x "${v}/bin/python" ]; then
            local info
            info=$("${v}/bin/python" -c "
import sys, importlib.util
try:
    import torch
    hip = getattr(torch.version, 'hip', None)
    cuda = torch.cuda.is_available()
    dev_name = f' ({torch.cuda.get_device_name(0)})' if cuda else ''
    dev = f'ROCm {hip}' if hip else (f'CUDA {torch.version.cuda}{dev_name}' if cuda else 'CPU')
    tver = torch.__version__
except Exception as e:
    dev = 'No Torch'
    tver = '-'
codec = 'torchcodec' if importlib.util.find_spec('torchcodec') else 'no-codec'
pyver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
print(f'Python {pyver} | Torch {tver} [{dev}] | {codec}')
" 2>/dev/null || echo "Corrupt or uninitialized")
            printf "  %-18s -> ✅ %s\n" "$v" "$info"
        else
            printf "  %-18s -> ❌ Not created\n" "$v"
        fi
    done
    echo "========================================================"
}

case "$TARGET" in
    main)
        setup_main
        ;;
    audio)
        setup_audio
        ;;
    download|youtube|yt-dlp)
        setup_download
        ;;
    separation|separate|demucs|roformer)
        setup_separation
        ;;
    pyannote|diarize)
        setup_pyannote
        ;;
    verify|verifier)
        setup_verify
        ;;
    align|alignment)
        setup_align
        ;;
    core)
        setup_core
        status_report
        ;;
    workers)
        setup_workers
        status_report
        ;;
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
    minicpmo|minicpm)
        setup_minicpmo
        ;;
    kimi)
        setup_kimi
        ;;
    status|check)
        status_report
        ;;
    all)
        setup_core
        setup_workers
        status_report
        ;;
    *)
        echo "❌ Unknown target: $TARGET"
        print_usage
        exit 1
        ;;
esac
