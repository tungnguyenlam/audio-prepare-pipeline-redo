#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "  Setting up isolated MiniCPM-o environment (.venv-minicpmo)"
echo "========================================================"

if [ "${1:-}" = "--clean" ] || [ "${1:-}" = "--recreate" ]; then
    echo "Cleaning existing .venv-minicpmo..."
    rm -rf .venv-minicpmo
fi

uv python install 3.11
uv venv .venv-minicpmo --python 3.11

PY=".venv-minicpmo/bin/python"

# If NVIDIA GPU present, select PyTorch wheel index matching driver CUDA version
if command -v nvidia-smi >/dev/null 2>&1; then
    cuda_ver=$(nvidia-smi | sed -n "s/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p" | head -n 1)
    case "$cuda_ver" in
        13.*) cu_url="" ;;
        12.8*) cu_url="https://download.pytorch.org/whl/cu128" ;;
        12.7*|12.6*) cu_url="https://download.pytorch.org/whl/cu126" ;;
        12.5*|12.4*) cu_url="https://download.pytorch.org/whl/cu124" ;;
        *) cu_url="https://download.pytorch.org/whl/cu128" ;;
    esac
    if [ -n "$cu_url" ]; then
        echo "⚡ Pre-installing torch for CUDA $cuda_ver from $cu_url..."
        uv pip install --python "$PY" --index-url "$cu_url" "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0"
    fi
fi

echo "Installing OpenBMB MiniCPM-o dependencies..."
uv pip install --python "$PY" -r requirements-minicpmo.txt

echo "Verifying environment..."
"$PY" -c "
import torch
dev_type = f'CUDA ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else 'CPU'
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import transformers
print(f'   -> Transformers: {transformers.__version__}')
"

echo "🎉 .venv-minicpmo is ready!"
