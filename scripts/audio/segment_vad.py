"""Plan recursive VAD-only cuts at the lowest speech probability in each oversized interval.

Consumes a completed evaluate/silero_jit report and writes an inspectable segment
manifest. It does not run VAD inference or render clips; use audio/export_segments
for rendering.
"""
from __future__ import annotations

from decimal import Decimal
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (  # noqa: E402
    FileContractError, LoggingArgumentParser, ROOT, identity, probe, progress,
    write_json,
)
from _common.segments import normalize_turns  # noqa: E402
from _common.vad import load_vad_report, plan_segments  # noqa: E402


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
    try:
        candidates, report = load_vad_report(source, report_path, args.vad_device)
    except (FileContractError, OSError, ValueError) as exc:
        p.error(str(exc))

    max_samples = math.floor(Decimal(str(args.max_duration_s)) * info['sample_rate'])
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
