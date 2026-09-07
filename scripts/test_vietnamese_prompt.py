#!/usr/bin/env python3
"""Test Vietnamese acoustic boundary verification prompt on Gemini 3.8 Flash and local Gemma 4."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vi_prompt_test")

VI_PROMPT = """Bạn là chuyên gia thẩm âm ranh giới âm thanh cho dữ liệu huấn luyện Text-to-Speech (TTS) tiếng Việt chất lượng cao.
Hãy lắng nghe kỹ đoạn âm thanh này và đánh giá dựa trên 2 tiêu chuẩn âm học nghiêm ngặt:

1. Độ thuần người nói (Speaker Purity):
- Chỉ có duy nhất MỘT người nói xuyên suốt đoạn âm thanh.
- Không có giọng người khác nói đè (overlap), tiếng nói nền, tiếng cười đùa chen ngang hoặc tiếng thì thầm.
- Đặc biệt chú ý 500ms cuối: không được dính đuôi giọng hoặc âm thanh của người thứ hai.

2. Độ trọn vẹn của từ (Acoustic Word Completeness - Tránh lẹm chữ):
- Ranh giới đầu và cuối phải là ranh giới từ tự nhiên, nguyên vẹn về âm vị học.
- ĐẦU ĐOẠN: Không bị lẹm hoặc mất phụ âm đầu của từ mở đầu.
- CUỐI ĐOẠN: Từ cuối cùng phải phát âm trọn vẹn cả thanh điệu và âm cuối (coda), tắt dần tự nhiên vào khoảng lặng. Tuyệt đối không bị cắt cụt (lẹm chữ) khi người nói đang phát âm dở.
- Chú ý: Không đánh giá hỏng chỉ vì câu chưa hoàn chỉnh ngữ pháp; chỉ kiểm tra xem từ ngữ thực tế có bị cắt đứt âm thanh hay không.

Xuất ra duy nhất định dạng JSON chuẩn (không dùng markdown backticks, không kèm văn bản giải thích bên ngoài):
{
  "speaker_purity": "pure" | "impure" | "uncertain",
  "word_completeness": "complete" | "incomplete" | "uncertain",
  "boundary_issue": "none" | "clipped_start" | "clipped_end" | "clipped_both" | "uncertain",
  "failure_codes": ["clipped_word_end"],
  "reason": "Giải thích âm học ngắn gọn bằng tiếng Việt"
}
Quy tắc:
- Nếu impure, failure_codes phải chứa ít nhất một trong: "overlapping_speech", "secondary_speaker", "tail_speaker_intrusion".
- Nếu incomplete, boundary_issue không được là "none", và failure_codes phải chứa ít nhất một trong: "clipped_word_start", "clipped_word_end", "unintelligible_boundary".
- Nếu pure và complete, failure_codes phải là [] và boundary_issue là "none"."""


def test_gemini(audio_path: Path) -> dict:
    from src.diarization.OverlapVerifier import GeminiOverlapVerifier
    verifier = GeminiOverlapVerifier(
        model="gemini-3.8-flash",
        prompt=VI_PROMPT,
        thinking_level="LOW",
        max_output_tokens=2048,
    )
    from src.utils.AudioClass import Audio
    audio = Audio.from_file(audio_path)
    t0 = time.time()
    res = verifier.verify(audio)
    res["latency_s"] = round(time.time() - t0, 2)
    return res


def test_gemma12b(audio_path: Path) -> dict:
    from src.utils.AudioClass import Audio
    from src.diarization.OverlapVerifier import _read_audio, _normalize_result
    audio = Audio.from_file(audio_path)
    audio_bytes, _, _ = _read_audio(audio)
    payload = {
        "model": "unsloth/gemma-4-12B-it-qat-GGUF",
        "messages": [{"role": "user", "content": VI_PROMPT}],
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        "http://localhost:8889/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer sk-unsloth-d5e3632a095b7e04619fd36a650a9162",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        res = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    msg = res["choices"][0]["message"]
    content = msg.get("content", "").strip()
    parsed = _normalize_result(content, backend="Gemma-VI")
    parsed["latency_s"] = round(elapsed, 2)
    return parsed


def main() -> None:
    test_files = [
        Path(".data/experiment_khanhvy/cuts/turn_001_spk_00_0.10-1.85.wav"),
        Path(".data/experiment_khanhvy/cuts/turn_005_spk_00_10.83-19.84.wav"),
        Path(".data/experiment_khanhvy/cuts/turn_008_spk_00_28.16-36.02.wav"),
    ]

    for tf in test_files:
        if not tf.exists():
            continue
        logger.info("=== Testing on %s ===", tf.name)
        
        # Test Gemini 3.8 Flash
        try:
            gem_res = test_gemini(tf)
            logger.info("Gemini 3.8 Flash (VI prompt): %s in %.2fs | Reason: %s", gem_res["decision"], gem_res["latency_s"], gem_res["reason"])
        except Exception as e:
            logger.error("Gemini failed: %s", e)

        # Test Gemma 4 12B
        try:
            g12_res = test_gemma12b(tf)
            logger.info("Gemma 4 12B      (VI prompt): %s in %.2fs | Reason: %s", g12_res["decision"], g12_res["latency_s"], g12_res["reason"])
        except Exception as e:
            logger.error("Gemma 12B failed: %s", e)


if __name__ == "__main__":
    main()
