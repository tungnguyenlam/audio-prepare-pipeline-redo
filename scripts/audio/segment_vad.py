"""Plan recursive VAD-only cuts at the lowest speech probability in each oversized interval.

Consumes a completed evaluate/silero_jit report and writes an inspectable segment
manifest. It does not run VAD inference or render clips; use audio/export_segments
for rendering.
"""
from __future__ import annotations

import bisect
from decimal import Decimal
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (  # noqa: E402
    FileContractError, LoggingArgumentParser, ROOT, identity, probe, progress,
    read_json, write_json,
)
from _common.segments import normalize_turns  # noqa: E402


def vad_candidates(probabilities, analysis_samples, frame_samples, analysis_rate, source_rate):
    """Map valid Silero frame centers to source-sample cut candidates."""
    candidates = []
    for index, probability in enumerate(probabilities):
        frame_start = index * frame_samples
        frame_end = frame_start + frame_samples
        # The evaluator zero-pads its final partial frame. Do not let artificial
        # padding create a preferred low-activity boundary near the source end.
        if frame_end > analysis_samples:
            break
        center_s = ((frame_start + frame_end) / 2) / analysis_rate
        candidates.append((round(center_s * source_rate), float(probability), index, center_s))
    return candidates


def plan_segments(source_frames, max_samples, candidates):
    """Iteratively split oversized intervals; return ordered leaves and cut audit."""
    candidate_samples = [candidate[0] for candidate in candidates]
    pending = [(0, source_frames, 0, None)]
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


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-i', '-if', '--input-file', type=Path, required=True)
    p.add_argument('--vad-report', type=Path, required=True,
                   help='Completed evaluate/silero_jit JSON for the same source bytes')
    p.add_argument('--vad-device', default='cpu',
                   help='Probability track in the report; cuda:0 also denotes ROCm')
    p.add_argument('--max-duration-s', type=float, default=15.0,
                   help='Recursively split intervals longer than this duration')
    p.add_argument('--output-file', type=Path,
                   default=ROOT / '.data/audio/segment_vad/segments.json')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true')
    args = p.parse_args()

    if not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0:
        p.error('--max-duration-s must be finite and positive')
    destination = args.output_file.resolve()
    source = args.input_file.resolve()
    report_path = args.vad_report.resolve()
    if destination.suffix.lower() != '.json':
        p.error('--output-file must end in .json')
    if destination in {source, report_path}:
        p.error('Output cannot replace an input')
    if destination.exists() and not args.overwrite:
        p.error('Output exists; choose another path or use --overwrite')

    source_identity = identity(source)
    info = probe(source)
    report = read_json(report_path)
    if report.get('operation') != 'evaluate_silero_jit' or not report.get('complete'):
        p.error('VAD report must be a completed evaluate/silero_jit report')
    if report.get('source', {}).get('sha256') != source_identity['sha256']:
        p.error('VAD report and audio must identify the same source bytes')
    audio = report.get('audio', {})
    if (audio.get('source_sample_rate') != info['sample_rate'] or
            audio.get('source_frames') != info['frames']):
        p.error('VAD report source geometry does not match the input audio')
    parameters = report.get('parameters', {})
    analysis_rate = parameters.get('sample_rate')
    frame_samples = parameters.get('frame_samples')
    analysis_samples = audio.get('analysis_samples')
    if not all(isinstance(value, int) and value > 0
               for value in (analysis_rate, frame_samples, analysis_samples)):
        p.error('VAD report has invalid analysis geometry')
    track = report.get('devices', {}).get(args.vad_device, {})
    if track.get('status') != 'ok':
        p.error('Requested VAD track did not complete successfully')
    probabilities = track.get('probabilities')
    if (not isinstance(probabilities, list) or len(probabilities) != audio.get('frame_count') or
            not probabilities or
            not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                    for value in probabilities)):
        p.error('Requested VAD track has invalid probabilities')

    max_samples = math.floor(Decimal(str(args.max_duration_s)) * info['sample_rate'])
    candidates = vad_candidates(
        probabilities, analysis_samples, frame_samples, analysis_rate, info['sample_rate']
    )
    leaves, cuts = plan_segments(info['frames'], max_samples, candidates)
    turns = normalize_turns([
        {
            'speaker_id': 'unknown',
            'start_s': start / info['sample_rate'],
            'end_s': end / info['sample_rate'],
            'confidence': None,
            'lineage': {'depth': depth, 'parent_cut_id': parent_cut_id},
            'quality': {'status': 'candidate', 'human_verified': False},
        }
        for start, end, depth, parent_cut_id in leaves
    ], source)
    output = {
        'schema_version': 1,
        'operation': 'segment_vad',
        'model': 'silero_vad_jit',
        'source': source_identity,
        'parameters': {
            'max_duration_s': args.max_duration_s,
            'max_duration_samples': max_samples,
            'vad_device': args.vad_device,
            'vad_report': identity(report_path),
            'vad_model': report.get('model'),
            'algorithm_version': 'recursive-global-min-v1',
            'implementation': identity(Path(__file__)),
        },
        'timestamp_origin': 'source_audio',
        'source_sample_rate': info['sample_rate'],
        'speaker_ids': ['unknown'],
        'turns': turns,
        'audit': {'cuts': cuts},
        'clips_valid': False,
        'complete': True,
        'limitations': [
            'VAD selects cut boundaries only; it does not identify speakers or remove nonspeech.',
            'The minimum speech probability can still represent speech when no true silence exists.',
            'No ASR, word alignment, sentence-boundary, or phoneme-completeness evidence is used.',
        ],
    }
    write_json(destination, output)
    durations = [(turn['end_sample'] - turn['start_sample']) / info['sample_rate'] for turn in turns]
    progress(
        'VAD_SEGMENT',
        f'{info["duration_s"]:.3f}s -> {len(turns)} segments; {len(cuts)} cuts; '
        f'max={max(durations, default=0):.3f}s',
    )
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
