"""Plain JSON segment manifests and sample-accurate clip export."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from _common.files import FileContractError, convert, digest, probe, progress, read_json, safe_name, write_json


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
        clean = {k: v for k, v in turn.items() if k not in {'clip', 'clip_sha256', 'clip_frames', 'overlap', 'overlap_with', 'start_sample', 'end_sample'}}
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
        if wanted.get('operation') == 'diarize':
            raw_path = path.with_name('segments.raw.json')
            if not raw_path.is_file() or digest(raw_path) != old.get('raw_manifest_sha256'):
                return False
        return old.get('complete', False) and all(
            t.get('clip') and (path.parent / t['clip']).is_file()
            and digest(path.parent / t['clip']) == t.get('clip_sha256') for t in old.get('turns', []))
    if not overwrite:
        raise FileContractError(f'Conflicting manifest: {path}; use --overwrite')
    return False


def export(manifest: dict, source: Path, destination: Path, work_dir: Path, sample_rate: int | None = None, channels: int = 1,
           min_duration_s: float | None = None, max_duration_s: float | None = None,
           *, concurrency: int = 1, batch_size: int = 1) -> None:
    import concurrent.futures
    import shutil
    import soundfile as sf
    import threading

    info = probe(source)
    turns = normalize_turns(manifest['turns'], source)
    raw_turns = turns
    params = manifest.get('parameters', {})
    if min_duration_s is None:
        min_duration_s = params.get('min_duration_s', 2.0)
    if max_duration_s is None:
        max_duration_s = params.get('max_duration_s', 15.0)
    if min_duration_s is not None:
        turns = [t for t in turns if (t['end_s'] - t['start_s']) >= min_duration_s]
    if max_duration_s is not None:
        turns = [t for t in turns if (t['end_s'] - t['start_s']) <= max_duration_s]
    # overlap_with indices must refer to the surviving output turns.
    turns = normalize_turns(turns, source)
    turns_count = len(turns)
    progress('EXPORT', f'Exporting {turns_count} clip(s) from {source.name} (concurrency={concurrency}, batch_size={batch_size})')
    output = {**manifest, 'timestamp_origin': 'diarized_input', 'source_sample_rate': info['sample_rate'],
              'sample_rate': sample_rate or info['sample_rate'], 'channels': channels, 'turns': turns,
              'speaker_ids': sorted({t['speaker_id'] for t in turns}), 'complete': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted export is recognizable and can be retried.
    write_json(destination, output)
    if manifest.get('operation') == 'diarize':
        raw_path = destination.with_name('segments.raw.json')
        write_json(raw_path, {**manifest, 'turns': raw_turns,
                   'speaker_ids': sorted({t['speaker_id'] for t in raw_turns}),
                   'timestamp_origin': 'diarized_input', 'source_sample_rate': info['sample_rate'],
                   'duration_filter_applied': False, 'clips_valid': False, 'complete': True})
        output['raw_manifest_sha256'] = digest(raw_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, turns_count // 10)
    lock = threading.Lock()

    with tempfile.TemporaryDirectory(dir=work_dir) as directory, sf.SoundFile(source) as audio:
        work = Path(directory)
        # Read slice data from soundfile under audio lock
        turn_data = []
        for i, turn in enumerate(turns, 1):
            audio.seek(turn['start_sample'])
            data = audio.read(turn['end_sample'] - turn['start_sample'], dtype='float32', always_2d=True)
            turn_data.append((i, turn, data))

        def render_clip(i: int, turn: dict, data) -> None:
            raw, staged = work / f'raw_{i}.wav', work / f'clip_{i}.wav'
            sf.write(raw, data, info['sample_rate'], subtype='FLOAT')
            convert(raw, staged, output['sample_rate'], channels)
            name = (f"{safe_name(source.stem)}_{safe_name(str(manifest.get('model') or 'segments'))}_"
                    f"{safe_name(turn['speaker_id'])}_{round(turn['start_s'] * 1000):09d}-"
                    f"{round(turn['end_s'] * 1000):09d}_{i:04d}.wav")
            clip = destination.parent / name
            if clip.resolve() == source.resolve():
                raise FileContractError('Clip would overwrite source')
            # Same-filesystem atomic publication, even when work_dir is elsewhere.
            fd, temporary = tempfile.mkstemp(dir=clip.parent, suffix='.wav')
            os.close(fd)
            try:
                shutil.copyfile(staged, temporary)
                os.replace(temporary, clip)
            finally:
                Path(temporary).unlink(missing_ok=True)
            clip_frames = probe(clip)['frames']
            clip_digest = digest(clip)
            with lock:
                turn.update(clip=name, clip_sha256=clip_digest, clip_frames=clip_frames)
                if i == 1 or i == turns_count or i % step == 0:
                    progress('EXPORT_CLIP', f'{name}', current=i, total=turns_count)

        batches = [turn_data[k:k + max(1, batch_size)] for k in range(0, len(turn_data), max(1, batch_size))]
        if concurrency <= 1:
            for batch_chunk in batches:
                for i, turn, data in batch_chunk:
                    render_clip(i, turn, data)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for batch_chunk in batches:
                    futures = [pool.submit(render_clip, i, turn, data) for i, turn, data in batch_chunk]
                    for fut in concurrent.futures.as_completed(futures):
                        fut.result()

    output['complete'] = True
    write_json(destination, output)
    progress('EXPORT_COMPLETE', f'Exported {turns_count} clips to {destination.parent.name}')


def ensure_plots(manifest_path: Path, *, overwrite: bool = False) -> None:
    """Write timeline, duration, and cutoff plots next to a segments.json."""
    from _common.diarize_plots import plot_segment_outputs, sibling_plot_paths
    output = manifest_path.parent / 'timeline.png'
    if not overwrite and all(path.exists() for path in sibling_plot_paths(output)):
        return
    progress('PLOT_START', f'Rendering diarization plots: {manifest_path.name}')
    plot_segment_outputs(manifest_path, overwrite=overwrite)
