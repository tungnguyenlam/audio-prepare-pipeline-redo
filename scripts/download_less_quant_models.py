#!/usr/bin/env python3
"""Download less-quantized (Q8_0 / high precision) checkpoints for Gemma 4 E4B and 12B."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

MODELS = {
    "e4b_q8": {
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "files": ["gemma-4-E4B-it-Q8_0.gguf", "mmproj-F16.gguf"],
        "description": "Gemma 4 E4B Q8_0 (8-bit, ~8.2 GB)",
    },
    "12b_q6": {
        "repo_id": "unsloth/gemma-4-12b-it-GGUF",
        "files": ["gemma-4-12b-it-UD-Q6_K_XL.gguf", "mmproj-F16.gguf"],
        "description": "Gemma 4 12B UD-Q6_K_XL (6-bit, ~10.7 GB)",
    },
    "12b_q8": {
        "repo_id": "unsloth/gemma-4-12b-it-GGUF",
        "files": ["gemma-4-12b-it-Q8_0.gguf", "mmproj-F16.gguf"],
        "description": "Gemma 4 12B Q8_0 (8-bit, ~12.7 GB)",
    },
}


def download_target(target: str) -> None:
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN")
    cfg = MODELS[target]
    repo_id = cfg["repo_id"]
    logger.info("Starting download for %s from %s...", cfg["description"], repo_id)

    for fname in cfg["files"]:
        logger.info("Fetching %s/%s...", repo_id, fname)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            token=token,
        )
        logger.info("Downloaded %s to %s", fname, local_path)

    logger.info("Download completed successfully for %s", target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download less-quantized Gemma 4 GGUFs.")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="e4b_q8",
        help="Model target to download (e4b_q8, 12b_q6, 12b_q8, or all).",
    )
    args = parser.parse_args()

    targets = ["e4b_q8", "12b_q8"] if args.model == "all" else [args.model]
    for t in targets:
        download_target(t)


if __name__ == "__main__":
    main()
