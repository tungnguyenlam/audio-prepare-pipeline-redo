"""Download and cache the ViYT-Diar evaluation set as WAVs plus reference manifests."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (LoggingArgumentParser, identity, persist_path, positive_int,
                           progress, read_json, write_json)

VIYT_DIAR_DATASET_ID = 'tuanduy1612/ViYT-Diar'
VIYT_DIAR_SPLIT = 'test'
TARGET_SAMPLE_RATE = 16_000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / '.data' / 'evaluate' / 'viyt-diar'


def _speaker_list(row: dict) -> list[str]:
    raw = row.get('speaker')
    if raw is None:
        raw = row.get('speakers')
    if raw is None:
        raise KeyError('Row is missing speaker / speakers field')
    return [str(item) for item in raw]


def _write_wav_from_hf_audio(audio_info: dict, wav_path: Path, *, target_sr: int = TARGET_SAMPLE_RATE) -> None:
    """Decode HF Audio(decode=False) path/bytes via soundfile (not torchcodec)."""
    import numpy as np
    import soundfile as sf
    import librosa

    path = audio_info.get('path')
    raw = audio_info.get('bytes')
    array = None
    sr = None
    if raw is not None:
        try:
            array, sr = sf.read(io.BytesIO(raw), always_2d=False)
        except (RuntimeError, OSError, ValueError, sf.LibsndfileError):
            suffix = Path(str(path)).suffix if path else '.audio'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                array, sr = librosa.load(tmp_path, sr=None, mono=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    elif path:
        try:
            array, sr = sf.read(str(path), always_2d=False)
        except (RuntimeError, OSError, ValueError, sf.LibsndfileError):
            array, sr = librosa.load(str(path), sr=None, mono=True)
    else:
        raise ValueError('audio sample has neither path nor bytes')

    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f'empty audio for {wav_path.name}')
    if array.ndim > 1:
        array = np.mean(array, axis=-1)
    sr = int(sr)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), array.astype(np.float32), sr)


def _load_cached_samples(out_dir: Path, *, limit: int | None) -> list[dict]:
    manifest_file = out_dir / 'manifest.json'
    if not manifest_file.is_file():
        return []
    payload = read_json(manifest_file)
    samples = []
    for row in payload.get('samples', []):
        audio_path = Path(row['audio_path'])
        if not audio_path.is_absolute():
            audio_path = (Path(__file__).resolve().parents[2] / audio_path).resolve()
        reference_path = Path(row['reference_path'])
        if not reference_path.is_absolute():
            reference_path = (Path(__file__).resolve().parents[2] / reference_path).resolve()
        if not audio_path.is_file() or not reference_path.is_file():
            progress('VIYT_PREPARE_SKIP', f'Missing cached files for {row.get("audio_id")}')
            continue
        samples.append({
            'audio_id': str(row['audio_id']),
            'stem': str(row.get('stem') or Path(audio_path).stem),
            'audio_path': audio_path,
            'reference_path': reference_path,
            'sample_rate': int(row['sample_rate']),
            'duration_s': float(row['duration_s']),
            'speakers': [str(s) for s in row['speakers']],
            'turns': list(row['turns']),
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


def prepare_viyt_diar(*, output_dir: Path, limit: int | None = None, overwrite: bool = False) -> list[dict]:
    """Download ViYT-Diar and write WAVs plus reference segments.json files."""
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to prepare ViYT-Diar. "
            'Use an environment that already has it (for example .venvs/3dspeaker).'
        ) from exc
    import soundfile as sf

    out_dir = output_dir.resolve()
    audio_dir = out_dir / 'audio'
    reference_dir = out_dir / 'reference'
    manifest_file = out_dir / 'manifest.json'
    if manifest_file.is_file() and not overwrite:
        cached = _load_cached_samples(out_dir, limit=limit)
        payload = read_json(manifest_file)
        cached_limit = payload.get('limit')
        have_full = cached_limit is None and len(cached) >= 100
        have_enough = limit is not None and len(cached) >= limit
        if cached and (have_full or have_enough or (limit is None and cached_limit is None and cached)):
            progress('VIYT_PREPARE_CACHE', f'Using cached ViYT-Diar manifest ({len(cached)} clips)')
            return cached

    progress('VIYT_PREPARE', f'Loading {VIYT_DIAR_DATASET_ID} split={VIYT_DIAR_SPLIT}')
    ds = load_dataset(VIYT_DIAR_DATASET_ID, split=VIYT_DIAR_SPLIT)
    ds = ds.cast_column('audio', Audio(decode=False))
    if overwrite:
        import shutil
        shutil.rmtree(audio_dir, ignore_errors=True)
        shutil.rmtree(reference_dir, ignore_errors=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for index, row in enumerate(ds):
        if limit is not None and index >= limit:
            break
        audio_id = str(row.get('audio_id') or f'viyt_{index:03d}')
        stem = f'viyt_{index:03d}'
        wav_path = audio_dir / f'{stem}.wav'
        if overwrite or not wav_path.is_file():
            _write_wav_from_hf_audio(row['audio'], wav_path)
        starts = [float(x) for x in row['timestamps_start']]
        ends = [float(x) for x in row['timestamps_end']]
        speakers = _speaker_list(row)
        if not (len(starts) == len(ends) == len(speakers)):
            raise ValueError(
                f'{audio_id}: mismatched speaker/timestamp list lengths '
                f'({len(speakers)}, {len(starts)}, {len(ends)})'
            )
        turns = [
            {'speaker_id': spk, 'start_s': start, 'end_s': end}
            for spk, start, end in zip(speakers, starts, ends, strict=True)
            if end > start
        ]
        unique_speakers = sorted({turn['speaker_id'] for turn in turns})
        info = sf.info(str(wav_path))
        duration_s = float(info.duration)
        if turns:
            duration_s = max(duration_s, max(turn['end_s'] for turn in turns))
        sidecar = {
            'schema_version': 1,
            'operation': 'prepare_viyt_diar',
            'source': {
                'dataset_id': VIYT_DIAR_DATASET_ID,
                'split': VIYT_DIAR_SPLIT,
                'audio_id': audio_id,
                'stem': stem,
            },
            'output': {'path': persist_path(wav_path), 'sha256': identity(wav_path)['sha256']},
        }
        write_json(wav_path.with_suffix('.json'), sidecar)
        reference_path = reference_dir / stem / 'segments.json'
        write_json(reference_path, {
            'schema_version': 1,
            'operation': 'viyt_diar_reference',
            'dataset_id': VIYT_DIAR_DATASET_ID,
            'source': identity(wav_path),
            'audio_id': audio_id,
            'stem': stem,
            'sample_rate': int(info.samplerate),
            'duration_s': duration_s,
            'speaker_ids': unique_speakers,
            'turns': turns,
            'complete': True,
        })
        samples.append({
            'audio_id': audio_id,
            'stem': stem,
            'audio_path': wav_path.resolve(),
            'reference_path': reference_path.resolve(),
            'sample_rate': int(info.samplerate),
            'duration_s': duration_s,
            'speakers': unique_speakers,
            'turns': turns,
        })
        progress('VIYT_PREPARE_CLIP', f'{stem} {audio_id}', current=index + 1, total=limit or len(ds))

    if not samples:
        raise RuntimeError(f'No samples loaded from {VIYT_DIAR_DATASET_ID} ({VIYT_DIAR_SPLIT})')

    write_json(manifest_file, {
        'schema_version': 1,
        'operation': 'prepare_viyt_diar',
        'dataset_id': VIYT_DIAR_DATASET_ID,
        'split': VIYT_DIAR_SPLIT,
        'limit': limit,
        'num_samples': len(samples),
        'samples': [
            {
                'audio_id': sample['audio_id'],
                'stem': sample['stem'],
                'audio_path': persist_path(sample['audio_path']),
                'reference_path': persist_path(sample['reference_path']),
                'sample_rate': sample['sample_rate'],
                'duration_s': sample['duration_s'],
                'speakers': sample['speakers'],
                'turns': sample['turns'],
            }
            for sample in samples
        ],
    })
    progress('VIYT_PREPARE_DONE', f'Cached {len(samples)} clips under {out_dir}')
    return samples


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-od', '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                   help='Cache root for WAVs, reference manifests, and manifest.json')
    p.add_argument('-n', '-l', '--limit', type=positive_int, default=None,
                   help='Optional max number of clips (smoke test)')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true',
                   help='Re-download and overwrite cached files')
    args = p.parse_args()
    samples = prepare_viyt_diar(output_dir=args.output_dir, limit=args.limit, overwrite=args.overwrite)
    print(args.output_dir.resolve() / 'manifest.json')
    progress('VIYT_PREPARE_COUNT', f'{len(samples)} clips')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
