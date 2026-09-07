#!/usr/bin/env python3
"""Test Mel-Band RoFormer vocal separation on a test slice."""

from __future__ import annotations
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.separation.MelRoFormer import MelRoFormer
from src.utils.AudioClass import Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_mel_roformer")

TEST_INPUT = Path(".data/experiment_khanhvy/cuts/turn_002_spk_00_2.19-5.28.wav")

def main():
    if not TEST_INPUT.is_file():
        logger.error("Test input missing: %s", TEST_INPUT)
        return

    logger.info("Initializing MelRoFormer (device='cuda')...")
    separator = MelRoFormer(device="cuda", output_dir=".data/test_mel_roformer/out", work_dir=".data/test_mel_roformer/work")
    separator.load()
    logger.info("Loaded MelRoFormer! Running separation on %s...", TEST_INPUT)
    aud = Audio.from_file(TEST_INPUT)
    res = separator.separate(aud)
    logger.info("Separation complete! Output: %s (duration: %.2fs)", res.path, res.duration_s)
    separator.unload()
    logger.info("Test passed successfully!")

if __name__ == "__main__":
    main()
