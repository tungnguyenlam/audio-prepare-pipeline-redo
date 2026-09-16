"""Independent manifest transformation extracted from the existing purity algorithms."""
from __future__ import annotations

import logging
import math
from math import isfinite
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import soundfile as sf
from _manifest import arguments, load, save
from _common.files import identity, probe, read_json
logger = logging.getLogger(__name__)

DEFAULT_MIN_TURN_DURATION_S = 0.5
DEFAULT_MERGE_SAME_SPEAKER_GAP_S = 1.0
DEFAULT_BOUNDARY_COLLAR_S = 0.04
DEFAULT_JITTER_MAX_DURATION_S = 3.0

def clean_speaker_turns(turns: Sequence[dict], *, min_turn_duration_s: float=DEFAULT_MIN_TURN_DURATION_S, merge_same_speaker_gap_s: float=DEFAULT_MERGE_SAME_SPEAKER_GAP_S, boundary_collar_s: float=DEFAULT_BOUNDARY_COLLAR_S, jitter_max_duration_s: float=DEFAULT_JITTER_MAX_DURATION_S) -> list[dict]:
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
        TypeError: If ``turns`` contains a value other than ``JSON turn``.
        ValueError: If a cleanup setting is invalid.
    """
    settings = {'min_turn_duration_s': min_turn_duration_s, 'merge_same_speaker_gap_s': merge_same_speaker_gap_s, 'boundary_collar_s': boundary_collar_s, 'jitter_max_duration_s': jitter_max_duration_s}
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'{name} must be a number')
        if not isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    if not all((isinstance(turn, dict) for turn in turns)):
        raise TypeError('turns must contain only JSON turn values')
    if not turns:
        return []
    cleaned = sorted(({**turn} for turn in turns), key=lambda turn: (turn['start_s'], turn['end_s'], turn['speaker_id']))
    if jitter_max_duration_s > 0:
        labels = [turn['speaker_id'] for turn in cleaned]
        for index in range(1, len(cleaned) - 1):
            previous = cleaned[index - 1]
            current = cleaned[index]
            following = cleaned[index + 1]
            if labels[index - 1] == labels[index + 1] and labels[index] != labels[index - 1] and (current['end_s'] - current['start_s'] <= jitter_max_duration_s) and (not current['overlap']) and (previous['end_s'] <= current['start_s']) and (current['end_s'] <= following['start_s']) and (current['start_s'] - previous['end_s'] <= merge_same_speaker_gap_s) and (following['start_s'] - current['end_s'] <= merge_same_speaker_gap_s):
                labels[index] = labels[index - 1]
        cleaned = [{**turn, 'speaker_id': labels[index]} for index, turn in enumerate(cleaned)]
    starts = [turn['start_s'] for turn in cleaned]
    ends = [turn['end_s'] for turn in cleaned]
    for index in range(len(cleaned) - 1):
        left = cleaned[index]
        right = cleaned[index + 1]
        if left['speaker_id'] == right['speaker_id']:
            continue
        if ends[index] > starts[index + 1]:
            midpoint = (ends[index] + starts[index + 1]) / 2
            ends[index] = midpoint - boundary_collar_s
            starts[index + 1] = midpoint + boundary_collar_s
        elif starts[index + 1] - ends[index] < boundary_collar_s * 2:
            ends[index] -= boundary_collar_s
            starts[index + 1] += boundary_collar_s
    collared = [{**turn, 'start_s': max(0.0, starts[index]), 'end_s': ends[index]} for index, turn in enumerate(cleaned) if ends[index] > max(0.0, starts[index])]
    merged: list[dict] = []
    for turn in collared:
        if merged and merged[-1]['speaker_id'] == turn['speaker_id'] and (turn['start_s'] - merged[-1]['end_s'] <= merge_same_speaker_gap_s):
            previous = merged[-1]
            confidence = min(previous['confidence'], turn['confidence']) if previous['confidence'] is not None and turn['confidence'] is not None else None
            merged[-1] = {**previous, 'end_s': max(previous['end_s'], turn['end_s']), 'confidence': confidence, 'overlap': previous['overlap'] or turn['overlap']}
        else:
            merged.append(turn)
    return [turn for turn in merged if turn['end_s'] - turn['start_s'] >= min_turn_duration_s]


def main() -> int:
    p = arguments(__doc__)
    p.add_argument('-min', '--min-turn-duration-s', type=float, default=DEFAULT_MIN_TURN_DURATION_S, help='Minimum duration in seconds for surviving cleaned turns')
    p.add_argument('--merge-same-speaker-gap-s', type=float, default=DEFAULT_MERGE_SAME_SPEAKER_GAP_S, help='Maximum gap in seconds between same-speaker turns to merge')
    p.add_argument('--boundary-collar-s', type=float, default=DEFAULT_BOUNDARY_COLLAR_S, help='Collar margin in seconds shaved from each side of close speaker boundaries')
    p.add_argument('--jitter-max-duration-s', type=float, default=DEFAULT_JITTER_MAX_DURATION_S, help='Maximum turn duration in seconds for A-B-A jitter relabeling (0 to disable)')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')
    manifest, source = load(args)
    values = {key: getattr(args, key) for key in ('min_turn_duration_s', 'merge_same_speaker_gap_s', 'boundary_collar_s', 'jitter_max_duration_s')}
    turns = clean_speaker_turns(manifest['turns'], **values)
    save(args, manifest, source, turns, 'cleanup', values)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
