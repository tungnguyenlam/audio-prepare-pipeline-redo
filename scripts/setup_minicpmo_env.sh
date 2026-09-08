#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "  Setting up isolated MiniCPM-o environment (.venv-minicpmo)"
echo "========================================================"

# MiniCPM-o requires transformers==4.51.0 and torch <= 2.8.0.
# We create a dedicated isolated environment so the main pipeline is untouched.
uv python install 3.11
uv venv .venv-minicpmo --python 3.11

echo "Installing OpenBMB MiniCPM-o dependencies..."
uv pip install --python .venv-minicpmo/bin/python -r requirements-minicpmo.txt

echo "Verifying environment..."
.venv-minicpmo/bin/python -c "
import torch
dev_type = 'CUDA' if torch.cuda.is_available() else 'CPU'
print(f'   -> Torch: {torch.__version__} ({dev_type})')
import transformers
print(f'   -> Transformers: {transformers.__version__}')
"

echo "🎉 .venv-minicpmo is ready!"
echo ""
echo "To evaluate MiniCPM-o 4.5, run:"
echo ".venv-minicpmo/bin/python scripts/evaluate_verifier.py \\"
echo "  --backend hf_local \\"
echo "  --model openbmb/MiniCPM-o-4_5 \\"
echo "  --output-json .data/tts_strategy/gold_benchmark_20260908/reports/minicpm_o45.json \\"
echo "  --output-report .data/tts_strategy/gold_benchmark_20260908/reports/minicpm_o45.md \\"
echo "  --export-csv .data/tts_strategy/gold_benchmark_20260908/reports/minicpm_o45.csv"
