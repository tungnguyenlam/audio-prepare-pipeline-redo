#!/usr/bin/env bash
# ==============================================================================
# One-Click Runner: Gemma 4 E2B LoRA Distillation Training (RTX 4090)
#
# Usage:
#   chmod +x run_train_4090.sh
#   ./run_train_4090.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===================================================================="
echo "   Gemma 4 E2B LoRA Distillation Trainer (RTX 4090 / 24GB VRAM)     "
echo "===================================================================="

# 1. GPU Detection
echo ""
echo "[1/5] Checking GPU hardware..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    echo "CUDA GPU detected successfully."
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers and CUDA are installed."
fi

# 2. Environment & Tokens (.env)
echo ""
echo "[2/5] Checking environment credentials (.env)..."
if [ ! -f ".env" ]; then
    touch .env
fi

# Source existing .env if present
set -a
[ -f .env ] && . .env
set +a

# Prompt for HF_TOKEN if not set
if [ -z "${HF_TOKEN:-}" ]; then
    echo "--------------------------------------------------------------------"
    echo "Hugging Face token is required to access google/gemma-4-E2B-it"
    read -rp "Enter Hugging Face Token (hf_...): " INPUT_HF_TOKEN
    if [ -n "$INPUT_HF_TOKEN" ]; then
        echo "HF_TOKEN=$INPUT_HF_TOKEN" >> .env
        export HF_TOKEN="$INPUT_HF_TOKEN"
        echo "Saved HF_TOKEN to .env."
    else
        echo "WARNING: No HF_TOKEN provided. Base model download might fail if gated."
    fi
else
    echo "HF_TOKEN is configured."
fi

# Optional WANDB_API_KEY
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is not set. Setting WANDB_MODE=offline (metrics saved locally)."
    export WANDB_MODE="offline"
else
    echo "WANDB_API_KEY is configured."
fi

# 3. Python Environment Setup (using uv or fallback to python3 venv)
echo ""
echo "[3/5] Setting up Python virtual environment..."

# Check PATH for uv
if ! command -v uv &>/dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

# If uv is still not found, offer/install uv for fast installation
if ! command -v uv &>/dev/null; then
    echo "Installing uv (fast Python package manager)..."
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || true
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

if command -v uv &>/dev/null; then
    echo "Using uv for environment management."
    if [ ! -d ".venv" ]; then
        echo "Creating .venv with uv..."
        uv venv .venv
    fi
    echo "Syncing / installing dependencies via uv..."
    uv pip install \
        "torch>=2.4.0" \
        "torchaudio>=2.4.0" \
        "transformers>=4.48.0" \
        "accelerate>=1.0.0" \
        "peft>=0.14.0" \
        "bitsandbytes>=0.45.0" \
        "librosa>=0.10.0" \
        "soundfile>=0.12.0" \
        "python-dotenv>=1.0.0" \
        "wandb>=0.18.0" \
        "huggingface_hub>=0.28.0"
    
    PYTHON_CMD="uv run python"
else
    echo "uv not found, falling back to python3 -m venv..."
    if [ ! -d ".venv_train" ]; then
        python3 -m venv .venv_train
    fi
    # shellcheck disable=SC1091
    source .venv_train/bin/activate
    echo "Installing dependencies with pip..."
    pip install --upgrade pip
    pip install \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install \
        "transformers>=4.48.0" \
        "accelerate>=1.0.0" \
        "peft>=0.14.0" \
        "bitsandbytes>=0.45.0" \
        "librosa>=0.10.0" \
        "soundfile>=0.12.0" \
        "python-dotenv>=1.0.0" \
        "wandb>=0.18.0" \
        "huggingface_hub>=0.28.0"
    
    PYTHON_CMD="python"
fi

# 4. Verify Dataset
echo ""
echo "[4/5] Checking acoustic dataset..."
DATA_AUDIO_DIR=".data/distillation_e2b/audio"
if [ -d "$DATA_AUDIO_DIR" ] && [ "$(find "$DATA_AUDIO_DIR" -name "*.wav" 2>/dev/null | wc -l)" -gt 50 ]; then
    NUM_FILES=$(find "$DATA_AUDIO_DIR" -name "*.wav" | wc -l)
    echo "Found local audio dataset with $NUM_FILES audio samples."
else
    echo "Audio dataset archive will be downloaded directly from Hugging Face Hub by the trainer."
fi

# 5. Launch Training
echo ""
echo "[5/5] Launching Gemma 4 E2B LoRA Distillation Training..."
echo "Command: $PYTHON_CMD scripts/train_gemma4_e2b_lora.py"
echo "--------------------------------------------------------------------"

$PYTHON_CMD scripts/train_gemma4_e2b_lora.py

echo ""
echo "===================================================================="
echo "Training finished successfully!"
echo "Checkpoints saved locally to: .data/distillation/checkpoints_e2b/best_adapter"
echo "Hub destination: https://huggingface.co/tungnguyenlam/gemma-4-e2b-acoustic-verifier"
echo "===================================================================="
