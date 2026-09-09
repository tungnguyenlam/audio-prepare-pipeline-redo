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
DEFAULT_ENERGY_FRAME_LEN_MS = 2.0
DEFAULT_ENERGY_HOP_LEN_MS = 0.5
DEFAULT_TARGET_MAX_DURATION_S = 10.0
DEFAULT_TARGET_MIN_DURATION_S = 3.0
DEFAULT_MIN_SPLIT_PAUSE_S = 0.2

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

def _copy_turn_meta(src: dict, dst: dict, policy: str='smart_segmentation') -> None:
    """Copy provenance metadata attributes from src turn to dst turn."""
    for attr in ('_original_start_s', '_original_end_s', '_raw_start_s', '_raw_end_s', '_delta_start_ms', '_delta_end_ms', '_tail_rescued', '_consensus_start_s', '_consensus_end_s'):
        if attr in src:
            dst[attr] = src[attr]
    dst['_boundary_policy'] = policy

def smart_segment_speaker_turns(audio: Path, turns: Sequence[dict], *, max_duration_s: float=DEFAULT_TARGET_MAX_DURATION_S, min_duration_s: float=DEFAULT_TARGET_MIN_DURATION_S, min_pause_s: float=DEFAULT_MIN_SPLIT_PAUSE_S, words: list[dict[str, Any]] | None=None, frame_len_ms: float=DEFAULT_ENERGY_FRAME_LEN_MS, hop_len_ms: float=DEFAULT_ENERGY_HOP_LEN_MS, search_window_s: float=DEFAULT_ENERGY_SEARCH_WINDOW_S) -> tuple[list[dict], list[dict[str, Any]]]:
    """Split at supported word gaps; reject remainders without a safe split.

    ASR timing is evidence, not proof of acoustic completeness. Newly introduced
    cuts stay inside a pause between nonoverlapping recognized words. Punctuation
    alone and unrestricted waveform minima cannot authorize a split.

    Args:
        audio: File-backed source audio.
        turns: Candidate turns whose outer boundaries have already been refined.
        max_duration_s: Maximum output duration, including trailing remainders.
        min_duration_s: Minimum output duration; shorter candidates are rejected.
        min_pause_s: Required gap between recognized words, in seconds.
        words: Source-relative word timestamps, or word metadata on input turns.
        frame_len_ms: RMS analysis frame length.
        hop_len_ms: RMS analysis hop.
        search_window_s: Acoustic search radius, constrained inside the word gap.

    Returns:
        Accepted children and audits for splits and rejected remainders.

    Raises:
        ValueError: If duration or acoustic search parameters are invalid.
    """
    parameters = (min_duration_s, max_duration_s, min_pause_s, frame_len_ms, hop_len_ms, search_window_s)
    if not all((math.isfinite(v) and v > 0 for v in parameters)) or min_duration_s > max_duration_s:
        raise ValueError('Expected finite positive parameters and min_duration_s <= max_duration_s.')
    if not turns:
        return ([], [])
    if words is None:
        words = [w for turn in turns for w in (turn.get('_words') or [])]
    valid_words = []
    for word in words:
        start, end = (float(word['start']), float(word['end']))
        if math.isfinite(start) and math.isfinite(end) and (0 <= start < end):
            valid_words.append(word)
    valid_words.sort(key=lambda w: (float(w['start']), float(w['end'])))
    waveform, sr = sf.read(str(audio), dtype='float32', always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    frame_samples = max(1, int(round(frame_len_ms * sr / 1000)))
    hop_samples = max(1, int(round(hop_len_ms * sr / 1000)))
    accepted: list[dict] = []
    audits: list[dict[str, Any]] = []
    for turn in turns:
        current = turn['start_s']
        turn_words = [w for w in valid_words if float(w['end']) > turn['start_s'] and float(w['start']) < turn['end_s']]
        while current < turn['end_s']:
            remaining = turn['end_s'] - current
            reason = None
            method = 'tail_remainder'
            cut = turn['end_s']
            if remaining < min_duration_s:
                reason = 'below_min_duration'
            elif remaining > max_duration_s:
                candidates = []
                covered_end = current
                for left, right in zip(turn_words, turn_words[1:]):
                    covered_end = max(covered_end, float(left['end']))
                    gap_start, gap_end = (covered_end, float(right['start']))
                    if gap_end - gap_start < min_pause_s:
                        continue
                    lower = max(gap_start + 0.02, current + min_duration_s)
                    upper = min(gap_end - 0.02, current + max_duration_s, turn['end_s'] - min_duration_s)
                    if lower > upper:
                        continue
                    midpoint = (lower + upper) / 2
                    radius = min(search_window_s, (upper - lower) / 2)
                    sample = _find_local_valley(waveform, int(round(midpoint * sr)), search_samples=int(radius * sr), frame_samples=frame_samples, hop_samples=hop_samples)
                    candidate = sample / sr
                    if not lower <= candidate <= upper:
                        candidate = midpoint
                    punctuation = str(left.get('text', '')).rstrip().endswith(('.', '!', '?', '…'))
                    candidates.append((punctuation, candidate, gap_start, gap_end))
                if not candidates:
                    reason = 'no_supported_word_gap'
                else:
                    _, cut, gap_start, gap_end = max(candidates)
                    method = 'supported_word_gap'
            if reason:
                audits.append({'action': 'reject', 'reason': reason, 'speaker_id': turn['speaker_id'], 'parent_start_s': turn['start_s'], 'parent_end_s': turn['end_s'], 'rejected_start_s': current, 'rejected_end_s': turn['end_s']})
                break
            child = dict(speaker_id=turn['speaker_id'], start_s=current, end_s=cut, confidence=turn['confidence'])
            _copy_turn_meta(turn, child, policy='smart_segmentation')
            child['_words'] = [w for w in turn_words if float(w['start']) >= current and float(w['end']) <= cut]
            child['_transcript'] = ' '.join((str(w.get('text', '')).strip() for w in child['_words'])) or None
            accepted.append(child)
            audit = {'action': 'split' if method == 'supported_word_gap' else 'split_tail', 'method': method, 'speaker_id': turn['speaker_id'], 'parent_start_s': turn['start_s'], 'parent_end_s': turn['end_s'], 'child_start_s': current, 'child_end_s': cut, 'child_duration_s': child['end_s'] - child['start_s'], 'transcript': child['_transcript']}
            if method == 'supported_word_gap':
                audit.update(word_gap_start_s=gap_start, word_gap_end_s=gap_end)
            audits.append(audit)
            current = cut
    return (accepted, audits)

def main() -> int:
    p = arguments(__doc__)
    p.add_argument('--words-file', type=Path, help='JSON object with a words array; no ASR runs implicitly')
    p.add_argument('--max-duration-s', type=float, default=DEFAULT_TARGET_MAX_DURATION_S)
    p.add_argument('--min-duration-s', type=float, default=DEFAULT_TARGET_MIN_DURATION_S)
    p.add_argument('--min-pause-s', type=float, default=DEFAULT_MIN_SPLIT_PAUSE_S)
    args = p.parse_args()
    manifest, source = load(args)
    values = {key: getattr(args, key) for key in ('max_duration_s', 'min_duration_s', 'min_pause_s')}
    words = read_json(args.words_file)['words'] if args.words_file else None
    turns, audits = smart_segment_speaker_turns(source, manifest['turns'], words=words, **values)
    save(args, manifest, source, turns, 'segment', {**values, 'words_file': identity(args.words_file) if args.words_file else None}, audits=audits)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
