"""Shared verdict schema checks for runtime writing, analysis, and comparison."""
from __future__ import annotations

from typing import Any

from _common.files import ROOT

ACOUSTIC_FAILURE_CODES = (
    "clipped_word_start",
    "clipped_word_end",
    "secondary_speaker",
    "overlapping_speech",
    "music_bleed",
    "noisy_reverberant",
    "distorted",
)
ELIGIBILITY_FAILURE_CODES = ("unsupported_language", "singing")
FAILURE_CODES = (*ACOUSTIC_FAILURE_CODES, *ELIGIBILITY_FAILURE_CODES)


def _known_prompts() -> dict[str, str]:
    result = {}
    for filename, profile in (
        ("full-tags-prompt.md", "acoustic_defect_v3"),
        ("acoustic_defect-3.txt", "acoustic_defect_v3"),
        ("speaker_purity.txt", "speaker_purity_v1"),
        ("word_boundary.txt", "word_boundary_v1"),
        ("archive/acoustic_defect-3.txt", "acoustic_defect_v3"),
        ("archive/speaker_purity.txt", "speaker_purity_v1"),
        ("archive/word_boundary.txt", "word_boundary_v1"),
    ):
        path = ROOT / "prompts" / filename
        if path.is_file():
            result[path.read_text(encoding="utf-8").strip()] = profile
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
    if profile == "custom" and isinstance(prompt, str) and (
        prompt.strip().startswith(
            "Listen to the supplied audio directly. Perform both acoustic verification and transcription"
        )
        or prompt.strip().startswith("# SYSTEM PROMPT — ACOUSTIC QC")
    ):
        profile = "acoustic_defect_v3"
    if not isinstance(prompt, str) and backend == "vibevoice":
        profile = "vibevoice_v1"

    if profile == "acoustic_defect_v3":
        speaker = verdict.get("speaker_purity")
        boundary = verdict.get("word_completeness")
        quality = verdict.get("audio_quality")
        codes = verdict.get("failure_codes")
        emotion = verdict.get("emotion")
        if speaker not in {"pure", "secondary_speaker", "overlapping_speech"}:
            return profile, "invalid_speaker_purity"
        if boundary not in {"complete", "clipped_word_start", "clipped_word_end"}:
            return profile, "invalid_word_completeness"
        if quality not in {"studio_clean", "music_bleed", "noisy_reverberant", "distorted"}:
            return profile, "invalid_audio_quality"
        if emotion is not None and (not isinstance(emotion, str) or not emotion.strip()):
            return profile, "invalid_emotion"
        if not isinstance(codes, list) or any(code not in FAILURE_CODES for code in codes):
            return profile, "invalid_failure_codes"
        expected_acoustic_codes = {
            value
            for value in (speaker, boundary, quality)
            if value not in {"pure", "complete", "studio_clean"}
        }
        actual_codes = set(codes)
        eligibility_codes = actual_codes.intersection(ELIGIBILITY_FAILURE_CODES)
        expected_codes = expected_acoustic_codes | eligibility_codes
        if len(codes) != len(actual_codes) or actual_codes != expected_codes:
            return profile, "inconsistent_failure_codes"
        expected_decision = "pass" if not expected_codes else "reject"
        if decision != expected_decision:
            return profile, "inconsistent_decision"
        if decision == "pass":
            transcript = verdict.get("transcript")
            if not isinstance(transcript, str) or not transcript.strip():
                return profile, "missing_transcript"
            if isinstance(prompt, str) and "EMOTION / SPEAKING STYLE" in prompt:
                if emotion is None or not isinstance(emotion, str) or not emotion.strip():
                    return profile, "missing_emotion"
            public_fields = [key for key in verdict if not key.startswith("_")]
            if not public_fields or public_fields[-1] != "transcript":
                return profile, "transcript_not_last"
        elif "transcript" in verdict:
            return profile, "unexpected_transcript"
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
