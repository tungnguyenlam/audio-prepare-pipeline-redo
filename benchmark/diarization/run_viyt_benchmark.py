"""Run ViYT-Diar diarization benchmarks and export comparison figures.

Baseline (default): Pyannote Community-1 in the primary ``.venv``.

Example (model server)::

    uv run python -m benchmark.diarization.run_viyt_benchmark --prepare-only
    uv run python -m benchmark.diarization.run_viyt_benchmark --systems pyannote_community
    uv run python -m benchmark.diarization.run_viyt_benchmark \\
        --systems pyannote_community,pyannote_31,sortformer --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark.diarization import (
    CACHE_DIR,
    DEFAULT_COLLAR_S,
    FIGURES_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    VIYT_DIAR_DATASET_ID,
)
from benchmark.diarization.dataset_viyt import prepare_viyt_diar
from benchmark.diarization.metrics import (
    FileMetrics,
    metrics_to_dict,
    score_file,
    summarize_system,
)
from benchmark.diarization.plot import export_baseline_figures
from benchmark.diarization.systems import (
    BASELINE_SYSTEM,
    SYSTEM_REGISTRY,
    SystemSpec,
    build_diarizer,
    resolve_systems,
)
from src.utils.AudioClass import Audio

logger = logging.getLogger("benchmark.diarization")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_systems(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return [BASELINE_SYSTEM]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(REPO_ROOT / ".data" / "huggingface")


def _diarize_sample(
    diarizer: Any,
    sample_audio: Audio,
    *,
    oracle_speakers: int | None,
) -> Any:
    """Call ``diarize`` with optional oracle speaker count when supported."""
    if oracle_speakers is None:
        return diarizer.diarize(sample_audio)

    try:
        return diarizer.diarize(sample_audio, num_speakers=oracle_speakers)
    except TypeError:
        return diarizer.diarize(sample_audio)


def run_system(
    spec: SystemSpec,
    samples: list[Any],
    *,
    device: str,
    token: str | None,
    collar_s: float,
    oracle_speakers: bool,
) -> tuple[list[FileMetrics], dict[str, Any]]:
    """Evaluate one system on all samples.

    Returns:
        Per-file metrics and a small timing/meta payload.
    """
    logger.info("Loading system %s (%s) on %s…", spec.key, spec.env_hint, device)
    diarizer = build_diarizer(spec, device=device, token=token)
    file_metrics: list[FileMetrics] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    try:
        if hasattr(diarizer, "load"):
            diarizer.load()

        for index, sample in enumerate(samples, start=1):
            audio = Audio.from_file(
                sample.audio_path,
                source_id=sample.audio_id,
                title=sample.audio_id,
                native_sample_rate=sample.sample_rate,
            )
            oracle = sample.num_speakers if oracle_speakers else None
            logger.info(
                "[%s] %d/%d %s (%.1fs, %d spk)",
                spec.key,
                index,
                len(samples),
                sample.audio_id,
                sample.duration_s,
                sample.num_speakers,
            )
            try:
                result = _diarize_sample(diarizer, audio, oracle_speakers=oracle)
                metrics = score_file(sample, result, collar_s=collar_s)
                file_metrics.append(metrics)
                logger.info(
                    "[%s] %s DER=%.3f JER=%.3f |spk|=%d",
                    spec.key,
                    sample.audio_id,
                    metrics.der,
                    metrics.jer,
                    metrics.speaker_count_abs_error,
                )
            except Exception as exc:
                logger.exception("[%s] failed on %s", spec.key, sample.audio_id)
                failures.append({"audio_id": sample.audio_id, "error": str(exc)})
    finally:
        close = getattr(diarizer, "close", None) or getattr(diarizer, "unload", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning("Failed to unload %s", spec.key, exc_info=True)

    elapsed_s = time.perf_counter() - started
    meta = {
        "system": spec.key,
        "label": spec.label,
        "env_hint": spec.env_hint,
        "elapsed_s": elapsed_s,
        "failures": failures,
    }
    return file_metrics, meta


def build_arg_parser() -> argparse.ArgumentParser:
    known = ", ".join(SYSTEM_REGISTRY)
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark diarization systems on ViYT-Diar and export figures "
            "under benchmark/figures/."
        )
    )
    parser.add_argument(
        "--systems",
        default=BASELINE_SYSTEM,
        help=(
            f"Comma-separated system keys (default: {BASELINE_SYSTEM}). "
            f"Known: {known}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of clips (smoke test).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device (auto / cuda / cpu).",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=DEFAULT_COLLAR_S,
        help=f"DER forgiveness collar in seconds (default: {DEFAULT_COLLAR_S}).",
    )
    parser.add_argument(
        "--oracle-speakers",
        action="store_true",
        help="Pass reference speaker count to backends that support it.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download/cache ViYT-Diar only; skip model inference.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download and overwrite the cached dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Dataset cache directory (default: {CACHE_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"JSON results directory (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help=f"Image export directory (default: {FIGURES_DIR}).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id prefix for result/figure filenames.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)
    _load_env()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        systems = resolve_systems(_parse_systems(args.systems))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    samples = prepare_viyt_diar(
        cache_dir=args.cache_dir,
        limit=args.limit,
        force=args.force_download,
    )
    logger.info(
        "Prepared %d clips from %s under %s",
        len(samples),
        VIYT_DIAR_DATASET_ID,
        args.cache_dir,
    )

    if args.prepare_only:
        logger.info("Prepare-only mode complete.")
        return 0

    token = os.getenv("HF_TOKEN")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    per_system_files: dict[str, list[FileMetrics]] = {}
    summaries = []
    system_meta: list[dict[str, Any]] = []

    for spec in systems:
        file_metrics, meta = run_system(
            spec,
            samples,
            device=args.device,
            token=token,
            collar_s=args.collar,
            oracle_speakers=args.oracle_speakers,
        )
        per_system_files[spec.key] = file_metrics
        summary = summarize_system(spec.key, file_metrics)
        summaries.append(summary)
        system_meta.append({**meta, "summary": metrics_to_dict(summary)})
        logger.info(
            "Summary %s: mean DER=%.3f median DER=%.3f mean JER=%.3f (n=%d)",
            spec.key,
            summary.mean_der,
            summary.median_der,
            summary.mean_jer,
            summary.num_files,
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{run_id}_viyt_diar.json"
    payload = {
        "run_id": run_id,
        "dataset_id": VIYT_DIAR_DATASET_ID,
        "collar_s": args.collar,
        "device": args.device,
        "oracle_speakers": args.oracle_speakers,
        "limit": args.limit,
        "num_samples": len(samples),
        "systems": system_meta,
        "per_file": {
            system: [metrics_to_dict(m) for m in files]
            for system, files in per_system_files.items()
        },
    }
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote results → %s", result_path)

    figure_paths = export_baseline_figures(
        summaries=summaries,
        per_system_files=per_system_files,
        output_dir=args.figures_dir,
        run_id=run_id,
    )
    for path in figure_paths:
        logger.info("Wrote figure → %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
