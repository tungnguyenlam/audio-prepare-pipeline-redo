"""Optional cleanup for diarization turns used in preview and extraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING, Iterable, Sequence

from src.diarization.schemas import SpeakerTurn

if TYPE_CHECKING:
    from src.diarization.schemas import DiarizationResult


DEFAULT_MIN_TURN_DURATION_S = 0.5
DEFAULT_MERGE_SAME_SPEAKER_GAP_S = 1.0
DEFAULT_BOUNDARY_COLLAR_S = 0.04
DEFAULT_JITTER_MAX_DURATION_S = 3.0


@dataclass(frozen=True)
class DiarizationFilter:
    """Configurable filter and cleanup pipeline for diarization turns and results.

    Provides fine-grained controls matching SonicStudio's Diarization tab:
    speaker inclusion/exclusion, minimum/maximum turn duration, overlap
    exclusion or isolation, confidence thresholding, time-range bounds,
    custom predicates, and optional turn cleanup (A-B-A jitter correction,
    boundary collar, same-speaker gap merging, and short-turn removal).
    """

    speakers: str | Sequence[str] | None = None
    exclude_speakers: str | Sequence[str] | None = None
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    exclude_overlap: bool = False
    only_overlap: bool = False
    min_confidence: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    predicate: Callable[[SpeakerTurn], bool] | None = None

    # Cleanup settings (optional)
    clean_turns: bool = False
    clean_first: bool = True
    min_turn_duration_s: float = DEFAULT_MIN_TURN_DURATION_S
    merge_same_speaker_gap_s: float = DEFAULT_MERGE_SAME_SPEAKER_GAP_S
    boundary_collar_s: float = DEFAULT_BOUNDARY_COLLAR_S
    jitter_max_duration_s: float = DEFAULT_JITTER_MAX_DURATION_S

    def __post_init__(self) -> None:
        if self.exclude_overlap and self.only_overlap:
            raise ValueError("exclude_overlap and only_overlap cannot both be True")
        if self.min_duration_s is not None:
            if isinstance(self.min_duration_s, bool) or not isinstance(
                self.min_duration_s, (int, float)
            ):
                raise TypeError("min_duration_s must be a number")
            if not isfinite(self.min_duration_s) or self.min_duration_s < 0:
                raise ValueError("min_duration_s must be finite and non-negative")
        if self.max_duration_s is not None:
            if isinstance(self.max_duration_s, bool) or not isinstance(
                self.max_duration_s, (int, float)
            ):
                raise TypeError("max_duration_s must be a number")
            if not isfinite(self.max_duration_s) or self.max_duration_s < 0:
                raise ValueError("max_duration_s must be finite and non-negative")
        if (
            self.min_duration_s is not None
            and self.max_duration_s is not None
            and self.max_duration_s < self.min_duration_s
        ):
            raise ValueError("max_duration_s cannot be less than min_duration_s")

        if self.min_confidence is not None:
            if isinstance(self.min_confidence, bool) or not isinstance(
                self.min_confidence, (int, float)
            ):
                raise TypeError("min_confidence must be a number")
            if not isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
                raise ValueError("min_confidence must be between 0 and 1")

        if self.start_s is not None:
            if isinstance(self.start_s, bool) or not isinstance(
                self.start_s, (int, float)
            ):
                raise TypeError("start_s must be a number")
            if not isfinite(self.start_s) or self.start_s < 0:
                raise ValueError("start_s must be finite and non-negative")
        if self.end_s is not None:
            if isinstance(self.end_s, bool) or not isinstance(
                self.end_s, (int, float)
            ):
                raise TypeError("end_s must be a number")
            if not isfinite(self.end_s) or self.end_s < 0:
                raise ValueError("end_s must be finite and non-negative")
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s <= self.start_s
        ):
            raise ValueError("end_s must be greater than start_s")

        for name, value in {
            "min_turn_duration_s": self.min_turn_duration_s,
            "merge_same_speaker_gap_s": self.merge_same_speaker_gap_s,
            "boundary_collar_s": self.boundary_collar_s,
            "jitter_max_duration_s": self.jitter_max_duration_s,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def filter_turns(self, turns: Sequence[SpeakerTurn]) -> list[SpeakerTurn]:
        """Apply filtering criteria and optional cleanup to a sequence of turns."""
        if not all(isinstance(turn, SpeakerTurn) for turn in turns):
            raise TypeError("turns must contain only SpeakerTurn values")

        def _do_clean(items: Sequence[SpeakerTurn]) -> list[SpeakerTurn]:
            return clean_speaker_turns(
                items,
                min_turn_duration_s=self.min_turn_duration_s,
                merge_same_speaker_gap_s=self.merge_same_speaker_gap_s,
                boundary_collar_s=self.boundary_collar_s,
                jitter_max_duration_s=self.jitter_max_duration_s,
            )

        current = list(turns)
        if self.clean_turns and self.clean_first:
            current = _do_clean(current)

        allowed_spks = (
            {self.speakers}
            if isinstance(self.speakers, str)
            else set(self.speakers)
            if self.speakers is not None
            else None
        )
        excluded_spks = (
            {self.exclude_speakers}
            if isinstance(self.exclude_speakers, str)
            else set(self.exclude_speakers)
            if self.exclude_speakers is not None
            else None
        )

        filtered: list[SpeakerTurn] = []
        for turn in current:
            if allowed_spks is not None and turn.speaker_id not in allowed_spks:
                continue
            if excluded_spks is not None and turn.speaker_id in excluded_spks:
                continue
            if self.min_duration_s is not None and turn.duration_s < self.min_duration_s:
                continue
            if self.max_duration_s is not None and turn.duration_s > self.max_duration_s:
                continue
            if self.exclude_overlap and turn.overlaps_other_speaker:
                continue
            if self.only_overlap and not turn.overlaps_other_speaker:
                continue
            if self.min_confidence is not None:
                if turn.confidence is None or turn.confidence < self.min_confidence:
                    continue
            if self.start_s is not None and turn.start_s < self.start_s:
                continue
            if self.end_s is not None and turn.end_s > self.end_s:
                continue
            if self.predicate is not None and not self.predicate(turn):
                continue
            filtered.append(replace(turn))

        if self.clean_turns and not self.clean_first:
            filtered = _do_clean(filtered)

        return filtered

    def apply(self, result: DiarizationResult) -> DiarizationResult:
        """Apply filtering criteria and optional cleanup to a DiarizationResult."""
        from src.diarization.schemas import DiarizationResult as DiarResultCls

        if not isinstance(result, DiarResultCls):
            raise TypeError(
                f"result must be a DiarizationResult, got {type(result).__name__}"
            )
        filtered_turns = self.filter_turns(result.turns)
        return result.with_turns(filtered_turns)

    def __call__(
        self,
        target: DiarizationResult | Sequence[SpeakerTurn],
    ) -> DiarizationResult | list[SpeakerTurn]:
        """Apply this filter to either a DiarizationResult or turn sequence."""
        from src.diarization.schemas import DiarizationResult as DiarResultCls

        if isinstance(target, DiarResultCls):
            return self.apply(target)
        return self.filter_turns(target)



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
