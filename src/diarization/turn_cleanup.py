"""Optional cleanup for diarization turns used in preview and extraction."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Iterable, Sequence

from src.diarization.schemas import SpeakerTurn


DEFAULT_MIN_TURN_DURATION_S = 0.5
DEFAULT_MERGE_SAME_SPEAKER_GAP_S = 1.0
DEFAULT_BOUNDARY_COLLAR_S = 0.04
DEFAULT_JITTER_MAX_DURATION_S = 3.0


def clean_speaker_turns(
    turns: Sequence[SpeakerTurn],
    *,
    min_turn_duration_s: float = DEFAULT_MIN_TURN_DURATION_S,
    merge_same_speaker_gap_s: float = DEFAULT_MERGE_SAME_SPEAKER_GAP_S,
    boundary_collar_s: float = DEFAULT_BOUNDARY_COLLAR_S,
    jitter_max_duration_s: float = DEFAULT_JITTER_MAX_DURATION_S,
) -> list[SpeakerTurn]:
    """Return a cleaned copy of speaker turns for high-precision output.

    The canonical diarization result is not mutated. Cleanup corrects short
    ``A-B-A`` label jitter, trims close boundaries between different speakers,
    merges adjacent turns from the same speaker, and drops short residual turns.
    Existing overlap evidence is preserved so cleanup is never presented as
    proof that overlapping speech was removed.

    Args:
        turns: Backend-independent diarization turns.
        min_turn_duration_s: Drop cleaned turns shorter than this duration.
        merge_same_speaker_gap_s: Merge adjacent same-speaker turns separated by
            no more than this gap.
        boundary_collar_s: Audio trimmed from each side of a close speaker
            boundary. The resulting total collar is twice this value.
        jitter_max_duration_s: Maximum middle-turn duration eligible for an
            ``A-B-A`` relabel. Set to zero to disable jitter correction.

    Returns:
        New turns ordered by start time. Input objects are not modified.

    Raises:
        TypeError: If ``turns`` contains a value other than ``SpeakerTurn``.
        ValueError: If a cleanup setting is invalid.
    """
    settings = {
        "min_turn_duration_s": min_turn_duration_s,
        "merge_same_speaker_gap_s": merge_same_speaker_gap_s,
        "boundary_collar_s": boundary_collar_s,
        "jitter_max_duration_s": jitter_max_duration_s,
    }
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if not all(isinstance(turn, SpeakerTurn) for turn in turns):
        raise TypeError("turns must contain only SpeakerTurn values")
    if not turns:
        return []

    cleaned = sorted(
        (replace(turn) for turn in turns),
        key=lambda turn: (turn.start_s, turn.end_s, turn.speaker_id),
    )

    # Only relabel a bounded, non-overlapping middle turn. Requiring both
    # neighboring gaps to be small avoids bridging independent conversations.
    if jitter_max_duration_s > 0:
        labels = [turn.speaker_id for turn in cleaned]
        for index in range(1, len(cleaned) - 1):
            previous = cleaned[index - 1]
            current = cleaned[index]
            following = cleaned[index + 1]
            if (
                labels[index - 1] == labels[index + 1]
                and labels[index] != labels[index - 1]
                and current.duration_s <= jitter_max_duration_s
                and not current.overlaps_other_speaker
                and previous.end_s <= current.start_s
                and current.end_s <= following.start_s
                and current.start_s - previous.end_s <= merge_same_speaker_gap_s
                and following.start_s - current.end_s <= merge_same_speaker_gap_s
            ):
                labels[index] = labels[index - 1]
        cleaned = [
            replace(turn, speaker_id=labels[index])
            for index, turn in enumerate(cleaned)
        ]

    # Split overlapping boundaries at their midpoint, then leave the requested
    # collar on both sides. Small non-overlapping boundaries are trimmed too.
    starts = [turn.start_s for turn in cleaned]
    ends = [turn.end_s for turn in cleaned]
    for index in range(len(cleaned) - 1):
        left = cleaned[index]
        right = cleaned[index + 1]
        if left.speaker_id == right.speaker_id:
            continue
        if ends[index] > starts[index + 1]:
            midpoint = (ends[index] + starts[index + 1]) / 2
            ends[index] = midpoint - boundary_collar_s
            starts[index + 1] = midpoint + boundary_collar_s
        elif starts[index + 1] - ends[index] < boundary_collar_s * 2:
            ends[index] -= boundary_collar_s
            starts[index + 1] += boundary_collar_s

    collared = [
        replace(turn, start_s=max(0.0, starts[index]), end_s=ends[index])
        for index, turn in enumerate(cleaned)
        if ends[index] > max(0.0, starts[index])
    ]

    merged: list[SpeakerTurn] = []
    for turn in collared:
        if (
            merged
            and merged[-1].speaker_id == turn.speaker_id
            and turn.start_s - merged[-1].end_s <= merge_same_speaker_gap_s
        ):
            previous = merged[-1]
            confidence = (
                min(previous.confidence, turn.confidence)
                if previous.confidence is not None and turn.confidence is not None
                else None
            )
            merged[-1] = replace(
                previous,
                end_s=max(previous.end_s, turn.end_s),
                confidence=confidence,
                overlaps_other_speaker=(
                    previous.overlaps_other_speaker
                    or turn.overlaps_other_speaker
                ),
            )
        else:
            merged.append(turn)

    return [
        turn for turn in merged if turn.duration_s >= min_turn_duration_s
    ]


def _clamp_pad_away_from_blockers(
    start_s: float,
    end_s: float,
    padded_start: float,
    padded_end: float,
    blockers: list[tuple[float, float]],
) -> tuple[float, float]:
    """Keep extra-before/after out of other-speaker windows.

    Extra may fill a gap up to the neighboring foreign turn. It does not trim
    overlap that is already inside the labeled ``[start_s, end_s)`` window.
    """
    for other_start, other_end in blockers:
        if other_end <= other_start:
            continue
        if other_start < padded_end and other_end > end_s:
            limit = other_start if other_start >= end_s else end_s
            padded_end = min(padded_end, limit)
        if other_end > padded_start and other_start < start_s:
            limit = other_end if other_end <= start_s else start_s
            padded_start = max(padded_start, limit)
    return padded_start, padded_end


def pad_and_merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    pre_roll_s: float = 0.0,
    post_roll_s: float = 0.0,
    start_bound_s: float = 0.0,
    end_bound_s: float | None = None,
    blocker_intervals: Iterable[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Expand time windows for extraction, then merge any that overlap or touch.

    Canonical diarization timestamps are not rewritten. Callers use this only
    when cutting or concatenating audio.

    Args:
        intervals: Inclusive-start, exclusive-end windows in seconds.
        pre_roll_s: Seconds added before each window.
        post_roll_s: Seconds added after each window.
        start_bound_s: Inclusive lower clamp, typically ``0``.
        end_bound_s: Optional inclusive upper clamp, typically source duration.
        blocker_intervals: Optional other-speaker windows. When set, extra
            before/after stops at those bounds instead of leaking foreign
            speech into the cut.

    Returns:
        Merged windows ordered by start time. Empty or inverted inputs are
        dropped.

    Raises:
        TypeError: If a bound is not a number.
        ValueError: If a bound is non-finite or a roll is negative.
    """
    settings = {
        "pre_roll_s": pre_roll_s,
        "post_roll_s": post_roll_s,
        "start_bound_s": start_bound_s,
    }
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
    if pre_roll_s < 0 or post_roll_s < 0:
        raise ValueError("pre_roll_s and post_roll_s must be non-negative")
    if end_bound_s is not None:
        if isinstance(end_bound_s, bool) or not isinstance(end_bound_s, (int, float)):
            raise TypeError("end_bound_s must be a number")
        if not isfinite(end_bound_s):
            raise ValueError("end_bound_s must be finite")

    blockers: list[tuple[float, float]] = []
    if blocker_intervals is not None:
        for item in blocker_intervals:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                raise TypeError("blocker_intervals entries must be start/end pairs")
            other_start, other_end = item[0], item[1]
            if isinstance(other_start, bool) or isinstance(other_end, bool):
                raise TypeError("blocker_intervals bounds must be numbers")
            if not isinstance(other_start, (int, float)) or not isinstance(
                other_end, (int, float)
            ):
                raise TypeError("blocker_intervals bounds must be numbers")
            if not isfinite(other_start) or not isfinite(other_end):
                raise ValueError("blocker_intervals bounds must be finite")
            if other_end > other_start:
                blockers.append((float(other_start), float(other_end)))

    padded: list[tuple[float, float]] = []
    for start_s, end_s in intervals:
        if end_s <= start_s:
            continue
        padded_start = max(start_bound_s, start_s - pre_roll_s)
        padded_end = end_s + post_roll_s
        if end_bound_s is not None:
            padded_end = min(end_bound_s, padded_end)
        if blockers:
            padded_start, padded_end = _clamp_pad_away_from_blockers(
                start_s, end_s, padded_start, padded_end, blockers
            )
        if padded_end > padded_start:
            padded.append((padded_start, padded_end))

    padded.sort(key=lambda window: window[0])
    merged: list[tuple[float, float]] = []
    for start_s, end_s in padded:
        if merged and start_s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
        else:
            merged.append((start_s, end_s))
    return merged
