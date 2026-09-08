#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "  Setting up isolated Kimi-Audio environment (.venv-kimi)"
echo "========================================================"

# Kimi-Audio runs best in an isolated environment with submodules cloned.
uv python install 3.11
uv venv .venv-kimi --python 3.11

echo "Installing Kimi-Audio base dependencies..."
uv pip install --python .venv-kimi/bin/python -r requirements-kimi.txt

echo "Checking / cloning Kimi-Audio inference submodule repository..."
mkdir -p .data/models
if [ ! -d ".data/models/Kimi-Audio/.git" ]; then
    echo "Cloning MoonshotAI/Kimi-Audio..."
    git clone https://github.com/MoonshotAI/Kimi-Audio.git .data/models/Kimi-Audio
fi

cd .data/models/Kimi-Audio
echo "Updating submodules (GLM-4 / audio tokenizers)..."
git submodule update --init --recursive
cd "$REPO_ROOT"

echo "Installing Kimi-Audio package into .venv-kimi..."
uv pip install --python .venv-kimi/bin/python -e .data/models/Kimi-Audio

echo "Verifying Kimi-Audio environment..."
.venv-kimi/bin/python -c "
import torch
dev_type = 'CUDA' if torch.cuda.is_available() else 'CPU'
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import transformers
print(f'   -> Transformers: {transformers.__version__}')
import sys
sys.path.insert(0, '.data/models/Kimi-Audio')
try:
    from kimia_infer.api.kimia import KimiAudio
    print('   -> KimiAudio API: successfully loaded!')
except Exception as e:
    print(f'   -> KimiAudio check: {e}')
"

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
