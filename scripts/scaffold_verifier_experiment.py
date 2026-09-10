#!/usr/bin/env python3
"""Create an empty local E2B experiment workspace; no model or API execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ThreadPoolExecutor
from _common.files import LoggingArgumentParser, positive_int, progress


def _scaffold_experiment(name: str, repo_root: Path, parser: LoggingArgumentParser) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        parser.error(f"name must contain only letters, numbers, underscores and hyphens: {name}")
    root = repo_root / ".data" / "verifier_experiments" / name
    if root.exists() or root.is_symlink():
        parser.error(f"workspace already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    for directory in (
        "manifests", "audio/seeds", "audio/synthetic", "reports/baseline",
        "reports/development", "reports/benchmark", "checkpoints/run_a",
        "checkpoints/run_b",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for manifest_name in ("sources", "seeds", "examples", "train", "dev", "uncertain"):
        (root / "manifests" / f"{manifest_name}.jsonl").touch(exist_ok=False)
    config = {
        "schema_version": 1,
        "status": "scaffold_only",
        "plan": "docs/E2B_VERIFIER_EXPERIMENT.md",
        "benchmark": {
            "input": ".data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl",
            "expected_clips": 288,
            "policy": "frozen_final_benchmark_only",
            "reserve_all_source_recordings": True,
            "historical_exposure": True,
        },
        "reason_required": False,
        "synthetic_targets_include_reason": False,
        "run_specifications_not_executable_config": {
            "run_a": {"base_model": "google/gemma-4-E2B-it", "lora_rank": 16,
                      "decoder_lora": True, "train_audio_projection": True,
                      "audio_encoder_lora": False},
            "run_b": {"base_model": "google/gemma-4-E2B-it", "lora_rank": 16,
                      "decoder_lora": True, "train_audio_projection": True,
                      "audio_encoder_lora": True},
        },
    }
    templates = {
        "notice": "Incomplete templates only; never load as training examples.",
        "source": {"recording_id": None, "speaker_ids": [], "split": None},
        "example": {
            "id": None, "audio_path": None, "audio_sha256": None,
            "recording_id": None, "split": None, "parent_ids": [],
            "donor_ids": [], "source_recording_ids": [], "sample_rate": None,
            "source_start_sample": None, "source_end_sample": None,
            "operation": {"type": None, "generator_version": None,
                          "random_seed": None, "parameters": {}},
            "labels": dict.fromkeys((
                "clipped_word_start", "clipped_word_end", "secondary_speaker",
                "overlapping_speech", "music_bleed", "sound_effects",
                "excessive_noise", "reverberation", "distorted",
            ), "uncertain"),
            "label_provenance": {"origin": None, "rubric_version": None,
                                 "evidence": None},
        },
        "trainer_export": {"audio_path": None, "prompt": None,
                           "target_json": {"labels": {}, "decision": None}},
    }
    for file_name, value in (("experiment.json", config), ("templates.json", templates)):
        with (root / file_name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    progress('SCAFFOLD_DONE', f'Created {root.relative_to(repo_root)}')
    print(f"Created {root.relative_to(repo_root)} (scaffold only)")
    return root


def main() -> None:
    """Create fresh manifests and planning templates below the runtime data root.

    Raises:
        OSError: If the workspace cannot be created or written.
    """
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument("--name", nargs="+", action="extend", required=True, help="New experiment directory name (supports multiple names for batch scaffolding)")
    parser.add_argument("--concurrency", type=positive_int, default=1, help="Number of worker threads for parallel workspace creation (default: 1)")
    parser.add_argument("--batch-size", type=positive_int, default=1, help="Batch size for parallel workspace creation (default: 1)")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.concurrency > 1 and len(args.name) > 1:
        batches = [args.name[i:i + args.batch_size] for i in range(0, len(args.name), args.batch_size)]
        def process_batch(chunk: list[str]) -> list[Path]:
            return [_scaffold_experiment(n, repo_root, parser) for n in chunk]
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as ex:
            list(ex.map(process_batch, batches))
    else:
        for n in args.name:
            _scaffold_experiment(n, repo_root, parser)

    print("Next: follow docs/E2B_VERIFIER_EXPERIMENT.md; benchmark is reserved.")


if __name__ == "__main__":
    main()
