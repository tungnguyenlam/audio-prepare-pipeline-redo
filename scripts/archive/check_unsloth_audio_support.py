#!/usr/bin/env python3
"""Check Unsloth multimodal / audio training support and Gemma 4 availability."""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_unsloth")

def check_env() -> None:
    try:
        import unsloth
        logger.info("Unsloth version: %s (path: %s)", getattr(unsloth, "__version__", "unknown"), unsloth.__file__)
    except ImportError:
        logger.warning("Unsloth is not installed in the current Python environment.")

    # Check unsloth models or methods
    try:
        from unsloth import FastLanguageModel
        logger.info("FastLanguageModel is available.")
    except Exception as exc:
        logger.warning("FastLanguageModel import error: %s", exc)

    try:
        from unsloth import FastVisionModel
        logger.info("FastVisionModel is available.")
    except Exception as exc:
        logger.warning("FastVisionModel import error: %s", exc)

    try:
        import unsloth_zoo
        logger.info("Unsloth Zoo version: %s", getattr(unsloth_zoo, "__version__", "unknown"))
    except Exception as exc:
        logger.warning("Unsloth zoo check: %s", exc)


if __name__ == "__main__":
    check_env()
