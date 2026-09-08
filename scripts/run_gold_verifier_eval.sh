#!/usr/bin/env bash
# Portable gold-benchmark verifier run for another machine.
#
# Sync first (repo + runtime audio under .data/), then:
#   export GEMINI_API_KEY=...
#   ./scripts/run_gold_verifier_eval.sh [model] [reasoning_effort] [concurrency]
#
# Defaults: gemini-3.5-flash-lite medium 8
# Local LoRA example:
#   BACKEND=hf_local MODEL=google/gemma-4-E2B-it ADAPTER=.data/distillation/checkpoints_e2b_v3/best_adapter \
#     DEVICE=cuda:0 ./scripts/run_gold_verifier_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

BACKEND="${BACKEND:-gemini}"
MODEL="${MODEL:-${1:-gemini-3.5-flash-lite}}"
EFFORT="${REASONING_EFFORT:-${2:-medium}}"
CONCURRENCY="${CONCURRENCY:-${3:-8}}"
DEVICE="${DEVICE:-auto}"
ADAPTER="${ADAPTER:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x .venv-sortformer/bin/python ]] && [[ "$BACKEND" == "hf_local" ]]; then
    PYTHON_BIN=".venv-sortformer/bin/python"
  elif [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

GOLD_DIR=".data/tts_strategy/gold_benchmark_20260908"
INPUT="${INPUT:-$GOLD_DIR/eval_input.jsonl}"
AUDIO_DIR="${MATERIALIZE_AUDIO:-$GOLD_DIR/audio}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_MODEL="$(echo "$MODEL" | tr '/:' '__')"
OUT_DIR="${OUT_DIR:-$GOLD_DIR/reports}"
REPORT="$OUT_DIR/${SAFE_MODEL}_${EFFORT}_${STAMP}.md"
JSON_OUT="$OUT_DIR/${SAFE_MODEL}_${EFFORT}_${STAMP}.json"
CSV_OUT="$OUT_DIR/${SAFE_MODEL}_${EFFORT}_${STAMP}.csv"
RESUME_JSON="${RESUME_JSON:-}"

mkdir -p "$OUT_DIR"

echo "== Gold verifier eval =="
echo " python:      $PYTHON_BIN"
echo " backend:     $BACKEND"
echo " model:       $MODEL"
echo " effort:      $EFFORT"
echo " concurrency: $CONCURRENCY"
echo " input:       $INPUT"
echo " materialize: $AUDIO_DIR"
echo " outputs:     $REPORT"

CHECK_ARGS=(
  scripts/evaluate_verifier.py
  --backend "$BACKEND"
  --model "$MODEL"
  --input "$INPUT"
  --materialize-audio "$AUDIO_DIR"
  --check-audio-only
)
"$PYTHON_BIN" "${CHECK_ARGS[@]}"

RUN_ARGS=(
  scripts/evaluate_verifier.py
  --backend "$BACKEND"
  --model "$MODEL"
  --input "$INPUT"
  --materialize-audio "$AUDIO_DIR"
  --output-report "$REPORT"
  --output-json "$JSON_OUT"
  --export-csv "$CSV_OUT"
  --concurrency "$CONCURRENCY"
  --device "$DEVICE"
)
if [[ "$BACKEND" == "gemini" ]]; then
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "GEMINI_API_KEY is required for backend=gemini" >&2
    exit 1
  fi
  RUN_ARGS+=(--reasoning-effort "$EFFORT")
fi
if [[ -n "$ADAPTER" ]]; then
  RUN_ARGS+=(--adapter-path "$ADAPTER")
fi
if [[ -n "$RESUME_JSON" ]]; then
  RUN_ARGS+=(--resume-json "$RESUME_JSON")
fi
if [[ -n "${LIMIT:-}" ]]; then
  RUN_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${OFFSET:-}" ]]; then
  RUN_ARGS+=(--offset "$OFFSET")
fi

"$PYTHON_BIN" "${RUN_ARGS[@]}"
echo "Done. Report: $REPORT"
echo "JSON:   $JSON_OUT"
