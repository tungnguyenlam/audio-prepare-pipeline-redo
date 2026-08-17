"""ViYT-Diar offline diarization benchmark package."""

from __future__ import annotations

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_ROOT.parent
CACHE_DIR = BENCHMARK_ROOT / "cache" / "viyt_diar"
RESULTS_DIR = BENCHMARK_ROOT / "results"
FIGURES_DIR = BENCHMARK_ROOT / "figures"

VIYT_DIAR_DATASET_ID = "tuanduy1612/ViYT-Diar"
VIYT_DIAR_SPLIT = "test"
DEFAULT_COLLAR_S = 0.25
