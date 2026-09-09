"""Plain JSON segment manifests and sample-accurate clip export."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from _common.files import FileContractError, convert, digest, probe, read_json, safe_name, write_json


def source_path(manifest: dict, manifest_path: Path, override: Path | None = None) -> Path:
    if override:
        return override.resolve()
    path = Path(manifest['source']['path'])
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def normalize_turns(turns: list[dict], source: Path) -> list[dict]:
    info = probe(source)
    rate, frames = info['sample_rate'], info['frames']
    result = []
    for turn in sorted(turns, key=lambda t: (t['start_s'], t['end_s'], t['speaker_id'])):
        start, end = float(turn['start_s']), float(turn['end_s'])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise FileContractError('Turns require finite start_s < end_s')
        a, b = max(0, min(frames, round(start * rate))), max(0, min(frames, round(end * rate)))
        if b <= a:
            continue
        clean = {k: v for k, v in turn.items() if k not in {'clip', 'clip_sha256', 'overlap', 'overlap_with', 'start_sample', 'end_sample'}}
        result.append({**clean, 'start_sample': a, 'end_sample': b, 'start_s': a / rate, 'end_s': b / rate})
    for i, turn in enumerate(result):
        overlaps = [j for j, other in enumerate(result) if j != i and
                    max(turn['start_sample'], other['start_sample']) < min(turn['end_sample'], other['end_sample'])]
        turn['overlap'] = bool(overlaps)
        turn['overlap_with'] = overlaps
    return result


def manifest_complete(path: Path, wanted: dict, overwrite: bool) -> bool:
    if not path.exists():
        if path.parent.exists() and any(path.parent.iterdir()) and not overwrite:
            raise FileContractError(f'Unrecognized output directory: {path.parent}; use --overwrite')
        return False
    try:
        old = read_json(path)
    except (ValueError, OSError):
        old = {}
    if all(old.get(k) == v for k, v in wanted.items()):
        return old.get('complete', False) and all(
            t.get('clip') and (path.parent / t['clip']).is_file()
            and digest(path.parent / t['clip']) == t.get('clip_sha256') for t in old.get('turns', []))
    if not overwrite:
        raise FileContractError(f'Conflicting manifest: {path}; use --overwrite')
    return False


def export(manifest: dict, source: Path, destination: Path, work_dir: Path, sample_rate: int | None = None, channels: int = 1) -> None:
    import soundfile as sf
    info = probe(source)
    turns = normalize_turns(manifest['turns'], source)
    output = {**manifest, 'timestamp_origin': 'diarized_input', 'source_sample_rate': info['sample_rate'],
              'sample_rate': sample_rate or info['sample_rate'], 'channels': channels, 'turns': turns, 'complete': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted export is recognizable and can be retried.
    write_json(destination, output)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as directory, sf.SoundFile(source) as audio:
        work = Path(directory)
        for i, turn in enumerate(turns):
            audio.seek(turn['start_sample'])
            data = audio.read(turn['end_sample'] - turn['start_sample'], dtype='float32', always_2d=True)
            raw, staged = work / 'raw.wav', work / 'clip.wav'
            sf.write(raw, data, info['sample_rate'], subtype='FLOAT')
            convert(raw, staged, output['sample_rate'], channels)
            name = (f"{safe_name(source.stem)}_{safe_name(str(manifest.get('model') or 'segments'))}_"
                    f"{safe_name(turn['speaker_id'])}_{round(turn['start_s'] * 1000):09d}-"
                    f"{round(turn['end_s'] * 1000):09d}_{i + 1:04d}.wav")
            clip = destination.parent / name
            if clip.resolve() == source.resolve():
                raise FileContractError('Clip would overwrite source')
            # Same-filesystem atomic publication, even when work_dir is elsewhere.
            import shutil
            fd, temporary = tempfile.mkstemp(dir=clip.parent, suffix='.wav')
            os.close(fd)
            try:
                shutil.copyfile(staged, temporary)
                os.replace(temporary, clip)
            finally:
                Path(temporary).unlink(missing_ok=True)
            turn.update(clip=name, clip_sha256=digest(clip), clip_frames=probe(clip)['frames'])
    output['complete'] = True
    write_json(destination, output)
