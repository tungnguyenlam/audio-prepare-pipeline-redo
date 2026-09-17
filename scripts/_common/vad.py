"""Shared validation and planning helpers for cached Silero VAD cuts."""
from __future__ import annotations

import bisect
import math
from pathlib import Path

from _common.files import FileContractError, identity, probe, read_json


def load_vad_report(source: Path, report_path: Path, vad_device: str) -> tuple[list[tuple[int, float, int, float]], dict]:
    """Validate a completed Silero report and return source-sample candidates."""
    source = source.resolve()
    report_path = report_path.resolve()
    source_info = probe(source)
    report = read_json(report_path)
    if report.get('operation') != 'evaluate_silero_jit' or not report.get('complete'):
        raise FileContractError('VAD report must be a completed evaluate/silero_jit report')
    if report.get('source', {}).get('sha256') != identity(source)['sha256']:
        raise FileContractError('VAD report and audio must identify the same source bytes')
    audio = report.get('audio', {})
    if (audio.get('source_sample_rate') != source_info['sample_rate'] or
            audio.get('source_frames') != source_info['frames']):
        raise FileContractError('VAD report source geometry does not match the input audio')
    parameters = report.get('parameters', {})
    analysis_rate = parameters.get('sample_rate')
    frame_samples = parameters.get('frame_samples')
    analysis_samples = audio.get('analysis_samples')
    if not all(isinstance(value, int) and value > 0
               for value in (analysis_rate, frame_samples, analysis_samples)):
        raise FileContractError('VAD report has invalid analysis geometry')
    track = report.get('devices', {}).get(vad_device, {})
    if track.get('status') != 'ok':
        raise FileContractError(f'Requested VAD track did not complete successfully: {vad_device}')
    probabilities = track.get('probabilities')
    if (not isinstance(probabilities, list) or len(probabilities) != audio.get('frame_count') or
            not probabilities or
            not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                    for value in probabilities)):
        raise FileContractError('Requested VAD track has invalid probabilities')

    candidates = []
    for index, probability in enumerate(probabilities):
        frame_start = index * frame_samples
        frame_end = frame_start + frame_samples
        # The evaluator zero-pads its final partial frame. It cannot create a
        # valid boundary beyond the real source duration.
        if frame_end > analysis_samples:
            break
        center_s = ((frame_start + frame_end) / 2) / analysis_rate
        candidates.append((round(center_s * source_info['sample_rate']),
                           float(probability), index, center_s))
    return candidates, report


def plan_segments(source_frames: int, max_samples: int,
                  candidates: list[tuple[int, float, int, float]], *,
                  start_sample: int = 0, end_sample: int | None = None,
                  vad_cut_threshold: float | None = None) -> tuple[list[tuple[int, int, int, int | None]], list[dict]]:
    """Recursively split one interval at eligible low VAD probabilities.

    When the lowest available probability is not below ``vad_cut_threshold``,
    the interval is retained as an overlong leaf and an explicit rejection
    event is returned. Exporters can then apply their normal duration filter.
    """
    if max_samples <= 0:
        raise ValueError('max_samples must be positive')
    if (vad_cut_threshold is not None and
            (not math.isfinite(vad_cut_threshold) or not 0 <= vad_cut_threshold <= 1)):
        raise ValueError('vad_cut_threshold must be finite and between 0 and 1')
    end_sample = source_frames if end_sample is None else end_sample
    if not 0 <= start_sample < end_sample <= source_frames:
        raise ValueError('Expected an interval within the source timeline')

    candidate_samples = [candidate[0] for candidate in candidates]
    pending = [(start_sample, end_sample, 0, None)]
    leaves, cuts = [], []
    next_cut_id = 0
    while pending:
        start, end, depth, parent_cut_id = pending.pop()
        if end - start <= max_samples:
            leaves.append((start, end, depth, parent_cut_id))
            continue
        midpoint = (start + end) / 2
        first = bisect.bisect_right(candidate_samples, start)
        stop = bisect.bisect_left(candidate_samples, end)
        if first == stop:
            raise FileContractError(
                f'No interior VAD frame can split oversized interval {start}:{end}'
            )
        minimum_probability = min(candidates[index][1] for index in range(first, stop))
        if (vad_cut_threshold is not None and
                minimum_probability >= vad_cut_threshold):
            leaves.append((start, end, depth, parent_cut_id))
            cuts.append({
                'action': 'reject',
                'reason': 'no_vad_cut_below_threshold',
                'parent_cut_id': parent_cut_id,
                'depth': depth,
                'interval_start_sample': start,
                'interval_end_sample': end,
                'interval_duration_samples': end - start,
                'minimum_speech_probability': minimum_probability,
                'vad_cut_threshold': vad_cut_threshold,
            })
            continue
        selected_index = min(
            range(first, stop),
            key=lambda index: (
                candidates[index][1], abs(candidates[index][0] - midpoint),
                candidates[index][0], candidates[index][2]
            ),
        )
        selected = candidates[selected_index]
        cut_sample, probability, frame_index, frame_center_s = selected
        cut_id = next_cut_id
        next_cut_id += 1
        cuts.append({
            'cut_id': cut_id,
            'parent_cut_id': parent_cut_id,
            'depth': depth,
            'interval_start_sample': start,
            'interval_end_sample': end,
            'interval_duration_samples': end - start,
            'cut_sample': cut_sample,
            'vad_frame_index': frame_index,
            'candidate_index': selected_index,
            'vad_frame_center_s': frame_center_s,
            'speech_probability': probability,
            'tie_break': 'lowest_probability_then_nearest_midpoint_then_earliest',
        })
        pending.append((cut_sample, end, depth + 1, cut_id))
        pending.append((start, cut_sample, depth + 1, cut_id))
    return sorted(leaves), cuts
