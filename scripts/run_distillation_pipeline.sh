#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "================================================================"
echo "Starting Gemma 4 E4B LoRA Distillation Pipeline"
echo "================================================================"

# 1. Wait for base model download if currently running
if pgrep -f "download_base_model.py" > /dev/null; then
    echo "[1/3] Waiting for base model download to complete..."
    while pgrep -f "download_base_model.py" > /dev/null; do
        sleep 10
    done
    echo "[1/3] Base model download complete!"
else
    echo "[1/3] Base model already downloaded."
fi

# 2. Train Gemma 4 E4B LoRA with W&B tracking and HF push
echo "[2/3] Starting LoRA Fine-Tuning on ROCm GPU..."
"${PYTHON_BIN}" scripts/train_gemma4_e4b_lora.py

# 3. Evaluate on Khanh Vy benchmark
echo "[3/3] Running evaluation and updating CSV..."
"${PYTHON_BIN}" scripts/evaluate_finetuned_verifier.py

echo "================================================================"
echo "Distillation Pipeline Completed Successfully!"
echo "Adapter on Hugging Face: https://huggingface.co/tungnguyenlam/gemma-4-e4b-acoustic-verifier"
echo "Results exported to: .data/experiment_khanhvy/results_finetuned.csv"
echo "================================================================"
