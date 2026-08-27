"""Backend-independent evaluation for speaker diarization turns."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable

from src.diarization.schemas import SpeakerTurn


def _normalized_turns(turns: Iterable[SpeakerTurn | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for turn in turns:
        if isinstance(turn, SpeakerTurn):
            speaker_id = turn.speaker_id
            start_s = turn.start_s
            end_s = turn.end_s
        elif isinstance(turn, dict):
            speaker_id = str(turn.get("speaker_id") or "").strip()
            try:
                start_s = float(turn.get("start_s"))
                end_s = float(turn.get("end_s"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Every turn requires numeric start_s and end_s") from exc
        else:
            raise TypeError("turns must contain SpeakerTurn values or objects")
        if not speaker_id:
            raise ValueError("Every turn requires a non-empty speaker_id")
        if not isfinite(start_s) or not isfinite(end_s) or start_s < 0 or end_s <= start_s:
            raise ValueError("Every turn must have finite timestamps with 0 <= start_s < end_s")
        normalized.append(
            {"speaker_id": speaker_id, "start_s": start_s, "end_s": end_s}
        )
    return normalized


def _maximum_weight_assignment(
    row_ids: list[str],
    column_ids: list[str],
    weights: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Return a maximum-weight one-to-one row-to-column assignment."""
    size = max(len(row_ids), len(column_ids))
    if size == 0:
        return {}
    max_weight = max(weights.values(), default=0.0)
    cost = [
        [
            max_weight
            - (
                weights.get((row_ids[row], column_ids[column]), 0.0)
                if row < len(row_ids) and column < len(column_ids)
                else 0.0
            )
            for column in range(size)
        ]
        for row in range(size)
    ]

    # Hungarian algorithm for a square cost matrix, using one-based work arrays.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = {}
    for column in range(1, size + 1):
        row = matched_row[column] - 1
        column_index = column - 1
        if row < len(row_ids) and column_index < len(column_ids):
            assignment[row_ids[row]] = column_ids[column_index]
    return assignment


def evaluate_diarization(
    reference_turns: Iterable[SpeakerTurn | dict[str, Any]],
    hypothesis_turns: Iterable[SpeakerTurn | dict[str, Any]],
    *,
    duration_s: float,
    collar_s: float = 0.0,
    skip_overlap: bool = False,
) -> dict[str, Any]:
    """Evaluate hypothesis turns against a manually annotated reference.

    The calculation uses exact interval boundaries instead of time sampling.
    Hypothesis speakers are mapped one-to-one to reference speakers by maximum
    scored temporal overlap before missed speech, false alarm, confusion, DER,
    and JER are computed.

    Args:
        reference_turns: Ground-truth speaker turns.
        hypothesis_turns: Model-produced speaker turns.
        duration_s: Duration of the shared source audio.
        collar_s: Forgiveness excluded on each side of every reference boundary.
        skip_overlap: Exclude regions with multiple active reference speakers.

    Returns:
        JSON-compatible evaluation metrics and speaker mapping.

    Raises:
        TypeError: If numeric settings have invalid types.
        ValueError: If settings or turns are invalid.
    """
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise TypeError("duration_s must be a number")
    if isinstance(collar_s, bool) or not isinstance(collar_s, (int, float)):
        raise TypeError("collar_s must be a number")
    duration_s = float(duration_s)
    collar_s = float(collar_s)
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and greater than zero")
    if not isfinite(collar_s) or collar_s < 0:
        raise ValueError("collar_s must be finite and non-negative")

    reference = _normalized_turns(reference_turns)
    hypothesis = _normalized_turns(hypothesis_turns)
    if not reference:
        raise ValueError("reference_turns must contain at least one annotated turn")
    if any(turn["end_s"] > duration_s + 0.05 for turn in reference + hypothesis):
        raise ValueError("A turn exceeds the shared audio duration")

    reference_ids = sorted({turn["speaker_id"] for turn in reference})
    hypothesis_ids = sorted({turn["speaker_id"] for turn in hypothesis})
    boundaries = {0.0, duration_s}
    excluded_collars: list[tuple[float, float]] = []
    for turn in reference:
        boundaries.update((max(0.0, turn["start_s"]), min(duration_s, turn["end_s"])))
        if collar_s:
            for boundary in (turn["start_s"], turn["end_s"]):
                start = max(0.0, boundary - collar_s)
                end = min(duration_s, boundary + collar_s)
                excluded_collars.append((start, end))
                boundaries.update((start, end))
    for turn in hypothesis:
        boundaries.update((max(0.0, turn["start_s"]), min(duration_s, turn["end_s"])))

    intervals: list[tuple[float, set[str], set[str]]] = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        if any(left <= midpoint < right for left, right in excluded_collars):
            continue
        active_reference = {
            turn["speaker_id"]
            for turn in reference
            if turn["start_s"] <= midpoint < turn["end_s"]
        }
        if skip_overlap and len(active_reference) > 1:
            continue
        active_hypothesis = {
            turn["speaker_id"]
            for turn in hypothesis
            if turn["start_s"] <= midpoint < turn["end_s"]
        }
        intervals.append((end - start, active_reference, active_hypothesis))

    overlap_weights = {
        (hypothesis_id, reference_id): sum(
            length
            for length, active_reference, active_hypothesis in intervals
            if reference_id in active_reference and hypothesis_id in active_hypothesis
        )
        for hypothesis_id in hypothesis_ids
        for reference_id in reference_ids
    }
    hypothesis_to_reference = _maximum_weight_assignment(
        hypothesis_ids, reference_ids, overlap_weights
    )
    hypothesis_to_reference = {
        hypothesis_id: reference_id
        for hypothesis_id, reference_id in hypothesis_to_reference.items()
        if overlap_weights.get((hypothesis_id, reference_id), 0.0) > 0
    }

    scored_audio_s = sum(length for length, _, _ in intervals)
    reference_speaker_s = 0.0
    hypothesis_speaker_s = 0.0
    missed_s = 0.0
    false_alarm_s = 0.0
    confusion_s = 0.0
    correct_s = 0.0
    reference_duration = {speaker_id: 0.0 for speaker_id in reference_ids}
    hypothesis_duration = {speaker_id: 0.0 for speaker_id in hypothesis_ids}
    speaker_intersection = {speaker_id: 0.0 for speaker_id in reference_ids}

    for length, active_reference, active_hypothesis in intervals:
        reference_count = len(active_reference)
        hypothesis_count = len(active_hypothesis)
        mapped_hypothesis = {
            hypothesis_to_reference[hypothesis_id]
            for hypothesis_id in active_hypothesis
            if hypothesis_id in hypothesis_to_reference
        }
        correct_count = len(active_reference & mapped_hypothesis)
        reference_speaker_s += reference_count * length
        hypothesis_speaker_s += hypothesis_count * length
        missed_s += max(0, reference_count - hypothesis_count) * length
        false_alarm_s += max(0, hypothesis_count - reference_count) * length
        confusion_s += (min(reference_count, hypothesis_count) - correct_count) * length
        correct_s += correct_count * length
        for speaker_id in active_reference:
            reference_duration[speaker_id] += length
            if speaker_id in mapped_hypothesis:
                speaker_intersection[speaker_id] += length
        for speaker_id in active_hypothesis:
            hypothesis_duration[speaker_id] += length

    reference_to_hypothesis = {
        reference_id: hypothesis_id
        for hypothesis_id, reference_id in hypothesis_to_reference.items()
    }
    per_speaker = []
    speaker_error_rates = []
    for reference_id in reference_ids:
        hypothesis_id = reference_to_hypothesis.get(reference_id)
        ref_s = reference_duration[reference_id]
        hyp_s = hypothesis_duration.get(hypothesis_id, 0.0) if hypothesis_id else 0.0
        intersection_s = speaker_intersection[reference_id]
        union_s = ref_s + hyp_s - intersection_s
        error_rate = 1.0 - (intersection_s / union_s) if union_s else 0.0
        speaker_error_rates.append(error_rate)
        per_speaker.append(
            {
                "reference_speaker_id": reference_id,
                "hypothesis_speaker_id": hypothesis_id,
                "reference_s": round(ref_s, 6),
                "hypothesis_s": round(hyp_s, 6),
                "intersection_s": round(intersection_s, 6),
                "coverage_pct": round((intersection_s / ref_s * 100) if ref_s else 0.0, 4),
                "jer_pct": round(error_rate * 100, 4),
            }
        )

    denominator = reference_speaker_s
    diarization_error_s = missed_s + false_alarm_s + confusion_s
    return {
        "der_pct": round((diarization_error_s / denominator * 100) if denominator else 0.0, 4),
        "jer_pct": round(
            (sum(speaker_error_rates) / len(speaker_error_rates) * 100)
            if speaker_error_rates
            else 0.0,
            4,
        ),
        "missed_speech_s": round(missed_s, 6),
        "false_alarm_s": round(false_alarm_s, 6),
        "speaker_confusion_s": round(confusion_s, 6),
        "correct_speaker_s": round(correct_s, 6),
        "reference_speaker_s": round(reference_speaker_s, 6),
        "hypothesis_speaker_s": round(hypothesis_speaker_s, 6),
        "scored_audio_s": round(scored_audio_s, 6),
        "collar_s": collar_s,
        "skip_overlap": bool(skip_overlap),
        "speaker_mapping": [
            {
                "hypothesis_speaker_id": hypothesis_id,
                "reference_speaker_id": reference_id,
                "overlap_s": round(overlap_weights.get((hypothesis_id, reference_id), 0.0), 6),
            }
            for hypothesis_id, reference_id in sorted(hypothesis_to_reference.items())
            if overlap_weights.get((hypothesis_id, reference_id), 0.0) > 0
        ],
        "unmapped_hypothesis_speakers": sorted(
            set(hypothesis_ids) - set(hypothesis_to_reference)
        ),
        "per_speaker": per_speaker,
    }
