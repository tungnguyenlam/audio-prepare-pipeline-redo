"""Separate overlapping talkers with ClearVoice MossFormer2_SS_16K.

Writes a stem directory with ``spk00.wav``, ``spk01.wav``, sibling sidecars, and
``stems.json``. Optional ``--start``/``--end`` limit inference to one interval so
callers can cut a diarized overlap region first. This does not assign separated
streams back onto diarization speaker IDs (permutation is left to the caller).
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.audio_utils import pin_device_visibility, working_directory  # noqa: E402
from _common.files import (  # noqa: E402
    FileContractError, batch, convert, digest, identity, parser, persist_path,
    positive_int, probe, publish, read_json, request, write_json,
)

MODEL_NAME = 'MossFormer2_SS_16K'
MODEL_SAMPLE_RATE = 16000
SPEAKER_IDS = ('spk00', 'spk01')


def main() -> int:
    p = parser(__doc__, 'cleanup', 'separate_overlap_clearvoice', segments=True)
    p.add_argument('-m', '--model', choices=(MODEL_NAME,), default=MODEL_NAME,
                   help='ClearVoice speech-separation checkpoint')
    p.add_argument('-d', '--device', default='auto',
                   help='Compute device (auto, cpu, cuda:0). Isolated venv is CPU on AMD ROCm hosts.')
    p.add_argument('-s', '--start', type=float, default=None,
                   help='Optional start seconds relative to input (default: full file)')
    p.add_argument('-e', '--end', type=float, default=None,
                   help='Optional end seconds relative to input (default: full file)')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo)')
    args = p.parse_args()
    if (args.start is None) != (args.end is None):
        p.error('Supply both --start and --end, or neither')
    if args.start is not None:
        if not math.isfinite(args.start) or not math.isfinite(args.end) or not 0 <= args.start < args.end:
            p.error('Require finite 0 <= start < end')
    if args.concurrency > 1:
        p.error('ClearVoice overlap separation keeps one session in-process; use --concurrency 1')

    pin_device_visibility(args.device)
    from clearvoice import ClearVoice

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with working_directory(args.work_dir / 'clearvoice_runtime'):
        session = ClearVoice(task='speech_separation', model_names=[args.model])

    from _common.files import manifest_destinations
    pairs = manifest_destinations(args, 'stems.json')

    def process(src: Path, dest: Path) -> None:
        info = probe(src)
        if args.end is not None and args.end > info['duration_s']:
            raise ValueError('--end exceeds input duration')
        rate = args.sample_rate or info['sample_rate']
        parameters = {
            'sample_rate': rate,
            'channels': args.channels,
            'device': args.device,
            'model_sample_rate': MODEL_SAMPLE_RATE,
            'task': 'speech_separation',
            'start_s': args.start,
            'end_s': args.end,
        }
        wanted = request(identity(src), 'separate_overlap', parameters, args.model)
        wavs = [dest.parent / f'{speaker}.wav' for speaker in SPEAKER_IDS]
        if _stems_completed(dest, wavs, wanted, args.overwrite):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_json(dest, {**wanted, 'stems': [], 'complete': False})
        with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
            work = Path(directory)
            prepared = work / 'input.wav'
            convert(src, prepared, MODEL_SAMPLE_RATE, args.channels, floating=True,
                    start=args.start, end=args.end)
            stem_dir = work / 'stems'
            stem_dir.mkdir()
            with working_directory(work):
                session(input_path=str(prepared), online_write=True, output_path=str(stem_dir))
            found = sorted(
                path for path in stem_dir.rglob('*')
                if path.is_file() and path.suffix.lower() == '.wav' and path.resolve() != prepared.resolve()
            )
            if len(found) < 2:
                raise ValueError(f'Expected two separated stems; found {len(found)} in {stem_dir}')
            stem_records = []
            for speaker, src_stem, wav in zip(SPEAKER_IDS, found[:2], wavs):
                staged = work / f'{speaker}.wav'
                convert(src_stem, staged, rate, args.channels)
                stem_meta = request(
                    identity(src), 'separate_overlap', {**parameters, 'speaker_id': speaker}, args.model,
                )
                publish(staged, wav, stem_meta)
                stem_records.append({
                    'speaker_id': speaker,
                    'path': persist_path(wav),
                    **probe(wav),
                    'sha256': digest(wav),
                })
            write_json(dest, {**wanted, 'stems': stem_records, 'complete': True})

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


def _stems_completed(stems_path: Path, wavs: list[Path], wanted: dict, overwrite: bool) -> bool:
    if not stems_path.exists() and not any(path.exists() for path in wavs):
        return False
    try:
        old = read_json(stems_path)
    except (OSError, ValueError, FileContractError):
        old = {}
    matches = all(old.get(key) == value for key, value in wanted.items())
    if matches and old.get('complete') and all(path.is_file() for path in wavs):
        try:
            for path in wavs:
                sidecar = path.with_suffix('.json')
                recorded = read_json(sidecar).get('output', {}).get('sha256')
                if recorded != digest(path):
                    return False
            return True
        except (OSError, ValueError, FileContractError):
            return False
    if not overwrite:
        raise FileContractError(f'Conflicting or unrecognized destination: {stems_path}; use --overwrite')
    return False


if __name__ == '__main__':
    raise SystemExit(main())
