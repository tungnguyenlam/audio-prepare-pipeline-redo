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

DEFAULT_COLLAR_EROSION_S = 0.35
DEFAULT_MIN_TURN_DURATION_S = 0.8
DEFAULT_TRANSITION_EXCLUSION_S = 0.5
DEFAULT_HANDOFF_RISK_DISTANCE_S = 0.8
DEFAULT_SILENCE_TAIL_BUFFER_S = 0.027

def erode_turn_boundaries(turns: Sequence[dict], collar_s: float=DEFAULT_COLLAR_EROSION_S, min_duration_s: float=DEFAULT_MIN_TURN_DURATION_S, transition_exclusion_s: float=DEFAULT_TRANSITION_EXCLUSION_S, competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> list[dict]:
    """Shave inward margins from turn boundaries and excavate speaker transitions.

    Args:
        turns: Speaker turns sorted by start time.
        collar_s: Inward margin shaved from start and end of every turn.
        min_duration_s: Minimum surviving duration required.
        transition_exclusion_s: When different speakers change with gap smaller
            than this threshold, extra safety padding is excavated.
        competitor_intervals_by_speaker: Pre-extracted competitor intervals per speaker
            preserving raw evidence from primary and secondary diarizers.

    Returns:
        Eroded, boundary-safe single-speaker turns.
    """
    if not turns:
        return []
    sorted_turns = sorted(turns, key=lambda t: (t['start_s'], t['end_s']))
    eroded: list[dict] = []
    for index, turn in enumerate(sorted_turns):
        start = turn['start_s'] + collar_s
        end = turn['end_s'] - collar_s
        spk = turn['speaker_id']
        comp_intervals = competitor_intervals_by_speaker.get(spk, []) if competitor_intervals_by_speaker is not None else None
        closest_prev_gap = float('inf')
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn['start_s'] < c_e:
                    closest_prev_gap = min(closest_prev_gap, 0.0)
                elif c_e <= turn['start_s']:
                    closest_prev_gap = min(closest_prev_gap, turn['start_s'] - c_e)
        if index > 0:
            prev_turn = sorted_turns[index - 1]
            if prev_turn['speaker_id'] != turn['speaker_id']:
                gap = max(0.0, turn['start_s'] - prev_turn['end_s'])
                closest_prev_gap = min(closest_prev_gap, gap)
        if closest_prev_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_prev_gap) / 2.0)
            start += extra
        closest_next_gap = float('inf')
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn['end_s'] < c_e:
                    closest_next_gap = min(closest_next_gap, 0.0)
                elif c_s >= turn['end_s']:
                    closest_next_gap = min(closest_next_gap, c_s - turn['end_s'])
        if index + 1 < len(sorted_turns):
            next_turn = sorted_turns[index + 1]
            if next_turn['speaker_id'] != turn['speaker_id']:
                gap = max(0.0, next_turn['start_s'] - turn['end_s'])
                closest_next_gap = min(closest_next_gap, gap)
        if closest_next_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_next_gap) / 2.0)
            end -= extra
        if end - start >= min_duration_s:
            final_start = round(start, 4)
            final_end = round(end, 4)
            t = dict(speaker_id=turn['speaker_id'], start_s=final_start, end_s=final_end, confidence=turn['confidence'])
            t['_original_start_s'] = turn.get('_original_start_s', turn['start_s'])
            t['_original_end_s'] = turn.get('_original_end_s', turn['end_s'])
            t['_raw_start_s'] = final_start
            t['_raw_end_s'] = final_end
            t['_delta_start_ms'] = 0.0
            t['_delta_end_ms'] = 0.0
            t['_boundary_policy'] = 'standard'
            t['_tail_rescued'] = False
            if '_consensus_start_s' in turn:
                t['_consensus_start_s'] = turn['_consensus_start_s']
            if '_consensus_end_s' in turn:
                t['_consensus_end_s'] = turn['_consensus_end_s']
            eroded.append(t)
    return eroded

def apply_context_aware_collar(turns: Sequence[dict], *, collar_s: float=DEFAULT_COLLAR_EROSION_S, handoff_risk_s: float=DEFAULT_HANDOFF_RISK_DISTANCE_S, silence_tail_s: float=DEFAULT_SILENCE_TAIL_BUFFER_S, min_duration_s: float=DEFAULT_MIN_TURN_DURATION_S, transition_exclusion_s: float=DEFAULT_TRANSITION_EXCLUSION_S, audio_duration_s: float | None=None, competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> tuple[list[dict], list[dict[str, Any]]]:
    """Apply asymmetric context-aware collar erosion to preserve word and syllable endings.

    If a turn is adjacent to another speaker within handoff_risk_s, it is aggressively
    eroded inward to eliminate speaker bleed. If the turn transitions into natural silence
    (or speech ceases with no rival speaker nearby), the trailing coda/tone is preserved
    and gently extended by silence_tail_s.

    Crucially, competitor proximity is checked against competitor_intervals_by_speaker
    (unifying primary ∪ secondary diarizer detections) to ensure consensus filtering
    does not mask true competitor handoffs.
    """
    if not turns:
        return ([], [])
    sorted_turns = sorted(turns, key=lambda t: (t['start_s'], t['end_s']))
    refined: list[dict] = []
    audits: list[dict[str, Any]] = []
    for index, turn in enumerate(sorted_turns):
        start = turn['start_s']
        end = turn['end_s']
        start_shaved = False
        end_shaved = False
        spk = turn['speaker_id']
        comp_intervals = competitor_intervals_by_speaker.get(spk, []) if competitor_intervals_by_speaker is not None else None
        blunt_start = turn['start_s'] + collar_s
        blunt_end = turn['end_s'] - collar_s
        closest_prev_gap = float('inf')
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn['start_s'] < c_e:
                    closest_prev_gap = min(closest_prev_gap, 0.0)
                elif c_e <= turn['start_s']:
                    closest_prev_gap = min(closest_prev_gap, turn['start_s'] - c_e)
        if index > 0:
            prev_turn = sorted_turns[index - 1]
            if prev_turn['speaker_id'] != turn['speaker_id']:
                gap = max(0.0, turn['start_s'] - prev_turn['end_s'])
                closest_prev_gap = min(closest_prev_gap, gap)
        if closest_prev_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_prev_gap) / 2.0)
            blunt_start += extra
        if closest_prev_gap < handoff_risk_s:
            extra = max(0.0, (transition_exclusion_s - closest_prev_gap) / 2.0) if closest_prev_gap < transition_exclusion_s else 0.0
            start = turn['start_s'] + collar_s + extra
            start_shaved = True
        closest_next_gap = float('inf')
        closest_next_start = float('inf')
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn['end_s'] < c_e:
                    closest_next_gap = min(closest_next_gap, 0.0)
                    closest_next_start = min(closest_next_start, c_s)
                elif c_s >= turn['end_s']:
                    gap = c_s - turn['end_s']
                    closest_next_gap = min(closest_next_gap, gap)
                    closest_next_start = min(closest_next_start, c_s)
        next_turn = sorted_turns[index + 1] if index + 1 < len(sorted_turns) else None
        if next_turn is not None and next_turn['speaker_id'] != turn['speaker_id']:
            gap = max(0.0, next_turn['start_s'] - turn['end_s'])
            closest_next_gap = min(closest_next_gap, gap)
            closest_next_start = min(closest_next_start, next_turn['start_s'])
        if closest_next_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_next_gap) / 2.0)
            blunt_end -= extra
        if closest_next_gap < handoff_risk_s:
            extra = max(0.0, (transition_exclusion_s - closest_next_gap) / 2.0) if closest_next_gap < transition_exclusion_s else 0.0
            end = turn['end_s'] - (collar_s + extra)
            end_shaved = True
        else:
            max_limit = audio_duration_s if audio_duration_s else turn['end_s'] + silence_tail_s + 1.0
            if next_turn is not None:
                max_limit = min(max_limit, next_turn['start_s'] - 0.05)
            if closest_next_start < float('inf'):
                max_limit = min(max_limit, closest_next_start - 0.05)
            end = min(turn['end_s'] + silence_tail_s, max_limit)
        if blunt_end <= blunt_start + 0.05:
            blunt_start = turn['start_s']
            blunt_end = turn['end_s']
        else:
            blunt_start = round(blunt_start, 4)
            blunt_end = round(blunt_end, 4)
        if end - start >= min_duration_s:
            final_start = round(start, 4)
            final_end = round(end, 4)
            orig_start = turn.get('_original_start_s', turn['start_s'])
            orig_end = turn.get('_original_end_s', turn['end_s'])
            delta_start = round((final_start - blunt_start) * 1000.0, 1)
            delta_end = round((final_end - orig_end) * 1000.0, 1)
            refined_turn = dict(speaker_id=turn['speaker_id'], start_s=final_start, end_s=final_end, confidence=turn['confidence'])
            refined_turn['_original_start_s'] = orig_start
            refined_turn['_original_end_s'] = orig_end
            refined_turn['_raw_start_s'] = blunt_start
            refined_turn['_raw_end_s'] = blunt_end
            refined_turn['_delta_start_ms'] = delta_start
            refined_turn['_delta_end_ms'] = delta_end
            refined_turn['_boundary_policy'] = 'context_aware_collar'
            refined_turn['_tail_rescued'] = not end_shaved
            if '_consensus_start_s' in turn:
                refined_turn['_consensus_start_s'] = turn['_consensus_start_s']
            if '_consensus_end_s' in turn:
                refined_turn['_consensus_end_s'] = turn['_consensus_end_s']
            refined.append(refined_turn)
            audits.append({'raw_start_s': blunt_start, 'raw_end_s': blunt_end, 'original_start_s': orig_start, 'original_end_s': orig_end, 'start_s': final_start, 'end_s': final_end, 'delta_start_ms': delta_start, 'delta_end_ms': delta_end, 'start_shaved': start_shaved, 'end_shaved': end_shaved, 'tail_rescued': not end_shaved, 'policy': 'context_aware_collar'})
    return (refined, audits)

def main() -> int:
    p = arguments(__doc__)
    p.add_argument('--context-aware', action='store_true', help='Use asymmetric handoff-aware collar adjustment')
    p.add_argument('--collar-s', type=float, default=DEFAULT_COLLAR_EROSION_S, help='Inward margin shaved from start and end of every turn in seconds')
    p.add_argument('--min-duration-s', type=float, default=DEFAULT_MIN_TURN_DURATION_S, help='Minimum surviving turn duration in seconds required')
    p.add_argument('--transition-exclusion-s', type=float, default=DEFAULT_TRANSITION_EXCLUSION_S, help='Speaker transition gap exclusion threshold in seconds')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')
    values = {key: getattr(args, key) for key in ('collar_s', 'min_duration_s', 'transition_exclusion_s')}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        p.error('Collar settings must be finite and nonnegative')
    manifest, source = load(args)
    if args.context_aware:
        turns, audits = apply_context_aware_collar(manifest['turns'], audio_duration_s=probe(source)['duration_s'], **values)
    else:
        turns, audits = erode_turn_boundaries(manifest['turns'], **values), []
    save(args, manifest, source, turns, 'collar', {**values, 'context_aware': args.context_aware}, audits=audits)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
