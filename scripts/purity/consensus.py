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

def _maximum_weight_assignment(row_ids: list[str], column_ids: list[str], weights: dict[tuple[str, str], float]) -> dict[str, str]:
    """Return a maximum-weight one-to-one row-to-column assignment."""
    size = max(len(row_ids), len(column_ids))
    if size == 0:
        return {}
    max_weight = max(weights.values(), default=0.0)
    cost = [[max_weight - (weights.get((row_ids[row], column_ids[column]), 0.0) if row < len(row_ids) and column < len(column_ids) else 0.0) for column in range(size)] for row in range(size)]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [float('inf')] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = float('inf')
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

def compute_consensus_turns(primary_turns: Sequence[dict], secondary_turns: Sequence[dict], audio_duration_s: float) -> tuple[list[dict], dict[str, str]]:
    """Intersect two diarization outputs using Hungarian speaker alignment.

    An interval [t1, t2] is preserved for Speaker S if and only if:
    1. Primary model asserts Speaker S is speaking AND no other speaker is active.
    2. Secondary model asserts the corresponding aligned speaker is speaking AND
       no other speaker is active.

    Returns:
        Consensus single-speaker turns and the speaker mapping dict.
    """
    if not primary_turns or not secondary_turns or audio_duration_s <= 0:
        return ([], {})
    prim_speakers = sorted({t['speaker_id'] for t in primary_turns})
    sec_speakers = sorted({t['speaker_id'] for t in secondary_turns})
    weights: dict[tuple[str, str], float] = {}
    for pt in primary_turns:
        for st in secondary_turns:
            overlap = max(0.0, min(pt['end_s'], st['end_s']) - max(pt['start_s'], st['start_s']))
            if overlap > 0:
                key = (pt['speaker_id'], st['speaker_id'])
                weights[key] = weights.get(key, 0.0) + overlap
    prim_to_sec = _maximum_weight_assignment(prim_speakers, sec_speakers, weights)
    if not prim_to_sec:
        return ([], {})
    boundaries = {0.0, float(audio_duration_s)}
    for t in list(primary_turns) + list(secondary_turns):
        boundaries.add(max(0.0, min(float(audio_duration_s), t['start_s'])))
        boundaries.add(max(0.0, min(float(audio_duration_s), t['end_s'])))
    ordered = sorted(boundaries)
    consensus_slices: list[tuple[float, float, str]] = []
    for start_s, end_s in zip(ordered, ordered[1:]):
        if end_s <= start_s + 0.0001:
            continue
        mid = (start_s + end_s) / 2.0
        active_prim = {t['speaker_id'] for t in primary_turns if t['start_s'] <= mid < t['end_s']}
        active_sec = {t['speaker_id'] for t in secondary_turns if t['start_s'] <= mid < t['end_s']}
        if len(active_prim) == 1 and len(active_sec) == 1:
            p_spk = next(iter(active_prim))
            s_spk = next(iter(active_sec))
            if prim_to_sec.get(p_spk) == s_spk:
                consensus_slices.append((start_s, end_s, p_spk))
    if not consensus_slices:
        return ([], prim_to_sec)
    merged_turns: list[dict] = []
    current_start, current_end, current_spk = consensus_slices[0]
    for s_start, s_end, s_spk in consensus_slices[1:]:
        if s_spk == current_spk and abs(s_start - current_end) < 0.0001:
            current_end = s_end
        else:
            t = dict(speaker_id=current_spk, start_s=round(current_start, 4), end_s=round(current_end, 4), confidence=1.0)
            t['_consensus_start_s'] = t['start_s']
            t['_consensus_end_s'] = t['end_s']
            merged_turns.append(t)
            current_start, current_end, current_spk = (s_start, s_end, s_spk)
    last_t = dict(speaker_id=current_spk, start_s=round(current_start, 4), end_s=round(current_end, 4), confidence=1.0)
    last_t['_consensus_start_s'] = last_t['start_s']
    last_t['_consensus_end_s'] = last_t['end_s']
    merged_turns.append(last_t)
    return (merged_turns, prim_to_sec)

def main() -> int:
    p = arguments(__doc__)
    p.add_argument('--secondary-manifest', type=Path, required=True)
    args = p.parse_args()
    manifest, source = load(args)
    secondary = read_json(args.secondary_manifest)
    from _common.segments import source_path
    other_source = source_path(secondary, args.secondary_manifest)
    if identity(source)['sha256'] != identity(other_source)['sha256']:
        p.error('Consensus manifests must describe the same audio and timestamp origin')
    turns, mapping = compute_consensus_turns(manifest['turns'], secondary['turns'], probe(source)['duration_s'])
    save(args, manifest, source, turns, 'consensus', {'secondary_manifest': identity(args.secondary_manifest)}, speaker_mapping=mapping)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
