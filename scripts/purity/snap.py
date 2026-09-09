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

DEFAULT_ENERGY_SEARCH_WINDOW_S = 0.15
DEFAULT_ENERGY_VALLEY_FLOOR_DB = -30.0
DEFAULT_ENERGY_FRAME_LEN_MS = 2.0
DEFAULT_ENERGY_HOP_LEN_MS = 0.5

def _find_local_valley(waveform: np.ndarray, center_sample: int, *, search_samples: int, frame_samples: int, hop_samples: int) -> int:
    """Find local RMS energy valley and zero-crossing in waveform around center_sample."""
    left = max(0, center_sample - search_samples)
    right = min(len(waveform), center_sample + search_samples)
    if right - left < frame_samples:
        return center_sample
    segment = waveform[left:right]
    num_frames = (len(segment) - frame_samples) // hop_samples + 1
    if num_frames <= 0:
        return center_sample
    energies = []
    for f in range(num_frames):
        f_start = f * hop_samples
        frame = segment[f_start:f_start + frame_samples]
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        energies.append(rms)
    energies = np.array(energies)
    center_frame = (center_sample - left) / hop_samples
    frame_indices = np.arange(len(energies))
    dist_penalty = (frame_indices - center_frame) ** 2 * 0.05
    cost = energies + dist_penalty * np.median(energies) * 0.1
    best_frame = int(np.argmin(cost))
    best_sample = left + best_frame * hop_samples + frame_samples // 2
    zc_window = waveform[max(0, best_sample - 20):min(len(waveform), best_sample + 20)]
    zc_indices = np.where(np.diff(np.signbit(zc_window)))[0]
    if len(zc_indices) > 0:
        best_sample = max(0, best_sample - 20) + zc_indices[0]
    return best_sample

def snap_boundaries_to_acoustic_valleys(audio: Path, turns: Sequence[dict], *, search_window_s: float=DEFAULT_ENERGY_SEARCH_WINDOW_S, energy_floor_db: float=DEFAULT_ENERGY_VALLEY_FLOOR_DB, frame_len_ms: float=DEFAULT_ENERGY_FRAME_LEN_MS, hop_len_ms: float=DEFAULT_ENERGY_HOP_LEN_MS, competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> tuple[list[dict], list[dict[str, Any]]]:
    """Snap turn boundaries to local short-time energy (RMS) silence valleys.

    Prevents slicing through voiced phonemes, vowels, or coda consonants by walking
    the boundary to the nearest local silence minimum / zero-crossing in the micro-waveform.
    Constrained so boundaries cannot drift into consensus-excluded or competitor speech.
    """
    if not turns:
        return ([], [])
    waveform, sr = sf.read(str(audio), dtype='float32', always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    frame_samples = max(1, int(round(frame_len_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_len_ms * sr / 1000.0)))
    search_samples = int(round(search_window_s * sr))
    snapped: list[dict] = []
    audits: list[dict[str, Any]] = []

    def find_local_valley(center_sample: int) -> int:
        return _find_local_valley(waveform, center_sample, search_samples=search_samples, frame_samples=frame_samples, hop_samples=hop_samples)
    for turn in turns:
        start_samp = int(round(turn['start_s'] * sr))
        end_samp = int(round(turn['end_s'] * sr))
        new_start_samp = find_local_valley(start_samp)
        new_end_samp = find_local_valley(end_samp)
        new_start_s = round(new_start_samp / sr, 4)
        new_end_s = round(new_end_samp / sr, 4)
        if '_consensus_start_s' in turn:
            new_start_s = max(new_start_s, turn['_consensus_start_s'])
        if '_consensus_end_s' in turn:
            new_end_s = min(new_end_s, turn['_consensus_end_s'])
        if competitor_intervals_by_speaker:
            comp_ivs = competitor_intervals_by_speaker.get(turn['speaker_id'], [])
            for c_s, c_e in comp_ivs:
                if c_e <= turn['start_s'] + 0.0001:
                    new_start_s = max(new_start_s, c_e)
                if c_s >= turn['end_s'] - 0.0001:
                    new_end_s = min(new_end_s, c_s - 0.05)
        if new_start_s >= new_end_s:
            new_start_s = turn['start_s']
            new_end_s = turn['end_s']
        orig_start = turn.get('_original_start_s', turn['start_s'])
        orig_end = turn.get('_original_end_s', turn['end_s'])
        raw_start = turn.get('_raw_start_s', turn['start_s'])
        raw_end = turn.get('_raw_end_s', turn['end_s'])
        if new_end_s - new_start_s >= 0.3:
            delta_start = round((new_start_s - raw_start) * 1000.0, 1)
            delta_end = round((new_end_s - orig_end) * 1000.0, 1)
            tail_rescued = new_end_s > orig_end
            refined_turn = dict(speaker_id=turn['speaker_id'], start_s=new_start_s, end_s=new_end_s, confidence=turn['confidence'])
            refined_turn['_original_start_s'] = orig_start
            refined_turn['_original_end_s'] = orig_end
            refined_turn['_raw_start_s'] = raw_start
            refined_turn['_raw_end_s'] = raw_end
            refined_turn['_delta_start_ms'] = delta_start
            refined_turn['_delta_end_ms'] = delta_end
            refined_turn['_boundary_policy'] = 'acoustic_energy_valley'
            refined_turn['_tail_rescued'] = tail_rescued
            refined_turn['_transcript'] = turn.get('_transcript', None)
            refined_turn['_words'] = turn.get('_words', None)
            if '_consensus_start_s' in turn:
                refined_turn['_consensus_start_s'] = turn['_consensus_start_s']
            if '_consensus_end_s' in turn:
                refined_turn['_consensus_end_s'] = turn['_consensus_end_s']
            snapped.append(refined_turn)
            audits.append({'raw_start_s': raw_start, 'raw_end_s': raw_end, 'original_start_s': orig_start, 'original_end_s': orig_end, 'start_s': new_start_s, 'end_s': new_end_s, 'delta_start_ms': delta_start, 'delta_end_ms': delta_end, 'policy': 'acoustic_energy_valley', 'tail_rescued': tail_rescued, 'transcript': turn.get('_transcript', None)})
        else:
            snapped.append(turn)
            audits.append({'raw_start_s': raw_start, 'raw_end_s': raw_end, 'original_start_s': orig_start, 'original_end_s': orig_end, 'start_s': turn['start_s'], 'end_s': turn['end_s'], 'delta_start_ms': turn.get('_delta_start_ms', 0.0), 'delta_end_ms': turn.get('_delta_end_ms', 0.0), 'policy': turn.get('_boundary_policy', 'standard'), 'tail_rescued': turn.get('_tail_rescued', False), 'transcript': turn.get('_transcript', None)})
    return (snapped, audits)

def main() -> int:
    p = arguments(__doc__)
    p.add_argument('--search-window-s', type=float, default=DEFAULT_ENERGY_SEARCH_WINDOW_S)
    p.add_argument('--energy-floor-db', type=float, default=DEFAULT_ENERGY_VALLEY_FLOOR_DB)
    p.add_argument('--frame-len-ms', type=float, default=DEFAULT_ENERGY_FRAME_LEN_MS)
    p.add_argument('--hop-len-ms', type=float, default=DEFAULT_ENERGY_HOP_LEN_MS)
    args = p.parse_args()
    values = {key: getattr(args, key) for key in ('search_window_s', 'energy_floor_db', 'frame_len_ms', 'hop_len_ms')}
    if any(not math.isfinite(v) or (k != 'energy_floor_db' and v <= 0) for k, v in values.items()):
        p.error('Require finite settings and positive search/frame/hop sizes')
    manifest, source = load(args)
    turns, audits = snap_boundaries_to_acoustic_valleys(source, manifest['turns'], **values)
    save(args, manifest, source, turns, 'snap', values, audits=audits)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
