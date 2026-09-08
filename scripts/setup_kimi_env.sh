#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "  Setting up isolated Kimi-Audio environment (.venv-kimi)"
echo "========================================================"

# Kimi-Audio runs best in an isolated environment with submodules cloned.
if [ "$1" = "--clean" ] || [ "$1" = "--recreate" ]; then
    echo "Cleaning existing .venv-kimi..."
    rm -rf .venv-kimi
fi

uv python install 3.11
uv venv .venv-kimi --python 3.11

PY="$REPO_ROOT/.venv-kimi/bin/python"
KIMI_DIR="$REPO_ROOT/.data/models/Kimi-Audio"

export UV_LINK_MODE=copy

echo "Checking / cloning Kimi-Audio inference submodule repository..."
mkdir -p .data/models
if [ ! -d "$KIMI_DIR/.git" ]; then
    echo "Cloning MoonshotAI/Kimi-Audio..."
    git clone https://github.com/MoonshotAI/Kimi-Audio.git "$KIMI_DIR"
fi

cd "$KIMI_DIR"
echo "Updating submodules (GLM-4 / audio tokenizers)..."
git submodule update --init --recursive
cd "$REPO_ROOT"

echo "1/4. Installing build and runtime foundation (torch==2.6.0, torchaudio==2.6.0)..."
uv pip install --python "$PY" \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    packaging ninja setuptools wheel

echo "2/4. Installing flash-attn 2.7.4.post1 (--no-build-isolation)..."
MAX_JOBS="${MAX_JOBS:-4}" uv pip install --python "$PY" \
    "flash-attn==2.7.4.post1" \
    --no-build-isolation

echo "3/4. Installing official Kimi-Audio dependencies from requirements.txt..."
uv pip install --python "$PY" \
    -r "$KIMI_DIR/requirements.txt" \
    --no-build-isolation

echo "Installing verifier pipeline auxiliary dependencies..."
uv pip install --python "$PY" \
    python-dotenv peft

echo "4/4. Installing Kimi-Audio package into .venv-kimi (--no-deps)..."
uv pip install --python "$PY" \
    -e "$KIMI_DIR" \
    --no-deps

echo "Verifying Kimi-Audio environment..."
"$PY" - <<'PY'
import torch
import torchaudio

try:
    import flash_attn
    flash_status = flash_attn.__version__
except Exception as e:
    flash_status = f"Failed ({e})"

dev_type = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
print(f"   -> Torch: {torch.__version__} ({dev_type})")
print(f"   -> Torchaudio: {torchaudio.__version__}")
print(f"   -> FlashAttention: {flash_status}")

import sys
sys.path.insert(0, ".data/models/Kimi-Audio")
try:
    from kimia_infer.api.kimia import KimiAudio
    print("   -> KimiAudio API: successfully loaded!")
except Exception as e:
    print(f"   -> KimiAudio check: {e}")
PY

echo "🎉 .venv-kimi is ready!"
echo ""
echo "To evaluate Kimi-Audio-7B-Instruct, run:"
echo ".venv-kimi/bin/python scripts/evaluate_verifier.py \\"
echo "  --backend hf_local \\"
echo "  --model moonshotai/Kimi-Audio-7B-Instruct \\"
echo "  --input .data/experiment_khanhvy/results.json \\"
echo "  --output-json .data/distillation/reports/kimi_audio_7b_eval.json \\"
echo "  --output-report .data/distillation/reports/kimi_audio_7b_vs_gemini38.md \\"
echo "  --export-csv .data/distillation/reports/kimi_audio_7b_vs_gemini38.csv"
