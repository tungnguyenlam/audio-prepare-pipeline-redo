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

echo "1/4. Installing build and runtime foundation (numpy, torch==2.6.0, torchaudio==2.6.0)..."
uv pip install --python "$PY" numpy
uv pip install --python "$PY" \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    packaging ninja setuptools wheel

echo "Verifying PyTorch environment compatibility for prebuilt FlashAttention wheel..."
"$PY" - <<'PY'
import sys
import torch

torch_version = torch.__version__
cuda_version = torch.version.cuda
abi_cxx11 = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)

print(f"   -> torch.__version__: {torch_version}")
print(f"   -> torch.version.cuda: {cuda_version}")
print(f"   -> torch._C._GLIBCXX_USE_CXX11_ABI: {abi_cxx11}")

errors = []
if not torch_version.startswith("2.6."):
    errors.append(f"Expected torch 2.6.x, got '{torch_version}'")
if not (cuda_version and cuda_version.startswith("12.4")):
    errors.append(f"Expected CUDA 12.4, got '{cuda_version}'")
if abi_cxx11 is not False and abi_cxx11 != 0:
    errors.append(f"Expected _GLIBCXX_USE_CXX11_ABI to be False, got '{abi_cxx11}'")

if errors:
    print("\n[ERROR] PyTorch environment does not match prebuilt FlashAttention wheel requirements:", file=sys.stderr)
    for err in errors:
        print(f"   - {err}", file=sys.stderr)
    sys.exit(1)

print("   -> Environment verified successfully (Torch 2.6, CUDA 12.4, ABI False).")
PY

echo "2/4. Installing FlashAttention 2.7.4.post1 from prebuilt official wheel..."
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FLASH_ATTN_WHEEL_NAME="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
WHEEL_CACHE_DIR="$REPO_ROOT/.data/wheels"
DEFAULT_WHEEL_PATH="$WHEEL_CACHE_DIR/$FLASH_ATTN_WHEEL_NAME"

WHEEL_SRC=""
if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "$FLASH_ATTN_WHEEL" ]; then
    WHEEL_SRC="$FLASH_ATTN_WHEEL"
elif [ -f "$DEFAULT_WHEEL_PATH" ]; then
    WHEEL_SRC="$DEFAULT_WHEEL_PATH"
elif [ -f "$REPO_ROOT/$FLASH_ATTN_WHEEL_NAME" ]; then
    WHEEL_SRC="$REPO_ROOT/$FLASH_ATTN_WHEEL_NAME"
elif [ -f "$KIMI_DIR/$FLASH_ATTN_WHEEL_NAME" ]; then
    WHEEL_SRC="$KIMI_DIR/$FLASH_ATTN_WHEEL_NAME"
fi

if [ -n "$WHEEL_SRC" ]; then
    echo "Found local FlashAttention prebuilt wheel: $WHEEL_SRC"
else
    echo "Prebuilt wheel not found locally. Downloading from GitHub release assets..."
    mkdir -p "$WHEEL_CACHE_DIR"
    WHEEL_TMP="$DEFAULT_WHEEL_PATH.tmp.$$"
    DOWNLOAD_SUCCESS=0

    if command -v curl >/dev/null 2>&1; then
        echo "Downloading via curl: $FLASH_ATTN_URL"
        if curl -fL --connect-timeout 20 --retry 2 -o "$WHEEL_TMP" "$FLASH_ATTN_URL"; then
            mv "$WHEEL_TMP" "$DEFAULT_WHEEL_PATH"
            DOWNLOAD_SUCCESS=1
            WHEEL_SRC="$DEFAULT_WHEEL_PATH"
        fi
    elif command -v wget >/dev/null 2>&1; then
        echo "Downloading via wget: $FLASH_ATTN_URL"
        if wget -O "$WHEEL_TMP" --timeout=20 --tries=2 "$FLASH_ATTN_URL"; then
            mv "$WHEEL_TMP" "$DEFAULT_WHEEL_PATH"
            DOWNLOAD_SUCCESS=1
            WHEEL_SRC="$DEFAULT_WHEEL_PATH"
        fi
    fi

    rm -f "$WHEEL_TMP"

    if [ "$DOWNLOAD_SUCCESS" -ne 1 ]; then
        echo "" >&2
        echo "========================================================================" >&2
        echo "ERROR: GitHub release assets could not be reached." >&2
        echo "Failed to download prebuilt FlashAttention wheel from:" >&2
        echo "  $FLASH_ATTN_URL" >&2
        echo "" >&2
        echo "The wheel can be downloaded elsewhere and copied onto this server at:" >&2
        echo "  $DEFAULT_WHEEL_PATH" >&2
        echo "or:" >&2
        echo "  $REPO_ROOT/$FLASH_ATTN_WHEEL_NAME" >&2
        echo "or specified using the FLASH_ATTN_WHEEL environment variable:" >&2
        echo "  export FLASH_ATTN_WHEEL=/path/to/$FLASH_ATTN_WHEEL_NAME" >&2
        echo "" >&2
        echo "Then re-run $0 to complete setup." >&2
        echo "========================================================================" >&2
        exit 1
    fi
fi

echo "Installing FlashAttention wheel with uv into .venv-kimi (--no-build)..."
uv pip install --python "$PY" \
    --no-build \
    "$WHEEL_SRC"

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
