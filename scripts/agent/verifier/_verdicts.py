"""Shared verdict schema checks for offline verifier analysis and comparison."""
from __future__ import annotations

from typing import Any

from _common.files import ROOT

FAILURE_CODES = (
    "clipped_word_start",
    "clipped_word_end",
    "secondary_speaker",
    "overlapping_speech",
    "music_bleed",
    "noisy_reverberant",
    "distorted",
)


def _known_prompts() -> dict[str, str]:
    result = {}
    for name in ("acoustic_defect", "speaker_purity", "word_boundary"):
        path = ROOT / "prompts" / f"{name}.txt"
        if path.is_file():
            result[path.read_text(encoding="utf-8").strip()] = f"{name}_v1"
    return result


def _validate_verdict(
    verdict: Any, prompt: Any, backend: str, known_prompts: dict[str, str]
) -> tuple[str, str | None]:
    if not isinstance(verdict, dict):
        return "unknown", "verdict_not_object"
    decision = verdict.get("decision")
    if decision not in {"pass", "reject"}:
        return "unknown", "invalid_decision"

    profile = known_prompts.get(prompt.strip(), "custom") if isinstance(prompt, str) else "custom"
    if not isinstance(prompt, str) and backend == "vibevoice":
        profile = "vibevoice_v1"

    if profile == "acoustic_defect_v1":
        speaker = verdict.get("speaker_purity")
        boundary = verdict.get("word_completeness")
        quality = verdict.get("audio_quality")
        codes = verdict.get("failure_codes")
        if speaker not in {"pure", "secondary_speaker", "overlapping_speech"}:
            return profile, "invalid_speaker_purity"
        if boundary not in {"complete", "clipped_word_start", "clipped_word_end"}:
            return profile, "invalid_word_completeness"
        if quality not in {"studio_clean", "music_bleed", "noisy_reverberant", "distorted"}:
            return profile, "invalid_audio_quality"
        if not isinstance(codes, list) or any(code not in FAILURE_CODES for code in codes):
            return profile, "invalid_failure_codes"
        expected_codes = {
            value
            for value in (speaker, boundary, quality)
            if value not in {"pure", "complete", "studio_clean"}
        }
        if len(codes) != len(set(codes)) or set(codes) != expected_codes:
            return profile, "inconsistent_failure_codes"
        expected_decision = "pass" if not expected_codes else "reject"
        if decision != expected_decision:
            return profile, "inconsistent_decision"
    elif profile == "speaker_purity_v1":
        speaker = verdict.get("speaker_purity")
        if speaker not in {"pure", "secondary_speaker", "overlapping_speech"}:
            return profile, "invalid_speaker_purity"
        expected_decision = "pass" if speaker == "pure" else "reject"
        if decision != expected_decision:
            return profile, "inconsistent_decision"
    elif profile == "word_boundary_v1":
        start = verdict.get("boundary_start")
        end = verdict.get("boundary_end")
        if start not in {"clean", "clipped"} or end not in {"clean", "clipped"}:
            return profile, "invalid_word_boundary"
        expected_decision = "pass" if start == end == "clean" else "reject"
        if decision != expected_decision:
            return profile, "inconsistent_decision"
    return profile, None
