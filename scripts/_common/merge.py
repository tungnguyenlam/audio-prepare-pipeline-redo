"""Silence-aware turn merging shared by diarization and the purity merge command."""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def add_merge_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument('--max-gap-s', type=float, default=1.0,
                   help='Maximum silence gap to bridge in seconds (default: 1.0)')
    p.add_argument('--silence-threshold-dbfs', type=float, default=-40.0,
                   help='Maximum per-channel frame RMS in dBFS (default: -40; calibrate for your audio)')
    p.add_argument('--frame-ms', type=float, default=20.0,
                   help='Silence analysis frame size in milliseconds (default: 20)')


def merge_parameters(args: argparse.Namespace, p: argparse.ArgumentParser) -> dict:
    if not math.isfinite(args.max_gap_s) or args.max_gap_s < 0:
        p.error('--max-gap-s must be finite and non-negative')
    if not math.isfinite(args.silence_threshold_dbfs) or args.silence_threshold_dbfs > 0:
        p.error('--silence-threshold-dbfs must be finite and <= 0')
    if not math.isfinite(args.frame_ms) or args.frame_ms <= 0:
        p.error('--frame-ms must be finite and positive')
    return {key: getattr(args, key) for key in ('max_gap_s', 'silence_threshold_dbfs', 'frame_ms')}


def merge_turns(source: Path, turns: list[dict], *, max_gap_s: float,
                silence_threshold_dbfs: float, frame_ms: float) -> tuple[list[dict], list[dict]]:
    """Consume normalized turns in time order; preserve gaps and speaker labels.

    Silence requires every nonoverlapping frame (including the final partial
    frame), in every channel, to have RMS at or below the configured threshold.
    Touching turns have no gap to measure. Overlapping turns are not merged.
    """
    import numpy as np
    import soundfile as sf

    merged: list[dict] = []
    audit: list[dict] = []
    threshold = 10.0 ** (silence_threshold_dbfs / 20.0)
    with sf.SoundFile(source) as audio:
        frame_samples = max(1, round(frame_ms * audio.samplerate / 1000))
        for index, turn in enumerate(turns):
            candidate = {**turn, 'merge_source_indices': [index]}
            if not merged or merged[-1]['speaker_id'] != turn['speaker_id']:
                merged.append(candidate)
                continue
            previous = merged[-1]
            gap_samples = turn['start_sample'] - previous['end_sample']
            gap_s = gap_samples / audio.samplerate
            decision = {'left_source_indices': list(previous['merge_source_indices']),
                        'right_source_index': index, 'gap_s': gap_s,
                        'max_frame_rms_dbfs': None}
            if gap_samples < 0:
                reason = 'overlapping_turns'
            elif gap_s > max_gap_s:
                reason = 'gap_too_long'
            elif any(other['speaker_id'] != turn['speaker_id'] and
                     other['start_sample'] < turn['end_sample'] and
                     other['end_sample'] > previous['start_sample'] for other in turns):
                reason = 'competing_speaker'
            else:
                audio.seek(previous['end_sample'])
                remaining = gap_samples
                max_rms = 0.0
                reason = 'merged'
                while remaining:
                    frame = audio.read(min(frame_samples, remaining), dtype='float64', always_2d=True)
                    if not len(frame):
                        raise ValueError('Source audio ended inside a merge gap')
                    if not np.isfinite(frame).all():
                        reason = 'nonfinite_audio'
                        break
                    max_rms = max(max_rms, float(np.sqrt(np.mean(frame * frame, axis=0)).max()))
                    remaining -= len(frame)
                if max_rms > 0:
                    decision['max_frame_rms_dbfs'] = 20.0 * math.log10(max_rms)
                if reason == 'merged' and max_rms > threshold:
                    reason = 'gap_not_silent'
            decision['reason'] = reason
            audit.append(decision)
            if reason != 'merged':
                merged.append(candidate)
                continue
            confidences = (previous.get('confidence'), turn.get('confidence'))
            # Scores/transcripts tied to an old clip are not valid for the union.
            merged[-1] = {'speaker_id': turn['speaker_id'],
                          'start_s': previous['start_s'], 'end_s': turn['end_s'],
                          'start_sample': previous['start_sample'], 'end_sample': turn['end_sample'],
                          'confidence': min(confidences) if all(c is not None for c in confidences) else None,
                          'merge_source_indices': [*previous['merge_source_indices'], index]}
    return merged, audit
