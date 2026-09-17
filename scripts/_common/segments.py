"""Plain JSON segment manifests and sample-accurate clip export."""
from __future__ import annotations

from decimal import Decimal
import math
import os
from pathlib import Path
import tempfile
from _common.files import (
    FileContractError,
    convert,
    digest,
    identity,
    probe,
    progress,
    read_json,
    resolve_stored_path,
    safe_name,
    write_json,
)
from _common.merge import (MEAN_ADJUST_MAX_ATTEMPTS, MEAN_ADJUST_STEP_S,
                            MEAN_DURATION_MAX_S, MEAN_DURATION_MIN_S, merge_turns)
from _common.vad import load_vad_report, plan_segments


def source_path(manifest: dict, manifest_path: Path, override: Path | None = None) -> Path:
    if override is not None:
        return override.resolve()
    path = resolve_stored_path(manifest['source']['path'], base=manifest_path.parent)
    expected = manifest['source'].get('sha256')
    if not expected:
        raise FileContractError('Manifest has no source SHA-256; supply --input-file explicitly on the same timeline')
    if digest(path) != expected:
        raise FileContractError(f'Source audio differs from the manifest: {path}; restore the original or supply --input-file explicitly on the same timeline')
    return path


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
    if overwrite:
        return False
    if not path.exists():
        if path.parent.exists() and any(path.parent.iterdir()):
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
            merged_path = path.with_name('segments.merged.json')
            if not merged_path.is_file() or digest(merged_path) != old.get('merged_manifest_sha256'):
                return False
        return old.get('complete', False) and all(
            t.get('clip') and (path.parent / t['clip']).is_file()
            and digest(path.parent / t['clip']) == t.get('clip_sha256') for t in old.get('turns', []))
    raise FileContractError(f'Conflicting manifest: {path}; use --overwrite')


def _filter_duration_turns(turns: list[dict], min_samples: int | None,
                           max_samples: int | None) -> list[dict]:
    if min_samples is not None:
        turns = [t for t in turns if t['end_sample'] - t['start_sample'] >= min_samples]
    if max_samples is not None:
        turns = [t for t in turns if t['end_sample'] - t['start_sample'] <= max_samples]
    return turns


def add_long_segment_arguments(parser) -> None:
    """Add the shared policy for turns above the export duration limit."""
    parser.add_argument('--long-segment-strategy', '--overlong-strategy',
                        dest='long_segment_strategy', choices=('vad', 'drop'), default='vad',
                        help='How to handle turns longer than --max-duration-s: vad recursively cuts them; drop discards them')
    parser.add_argument('--vad-report', type=Path,
                        help='Completed evaluate/silero_jit report for a single input file (required when VAD cuts an oversized turn)')
    parser.add_argument('--vad-report-dir', type=Path,
                        help='Directory of Silero reports named <audio-stem>.json for --input-dir VAD cuts')
    parser.add_argument('--vad-device', default='cpu',
                        help='Probability track in the VAD report; cuda:0 also denotes ROCm')
    parser.add_argument('--vad-cut-threshold', '--vad-threshold',
                        dest='vad_cut_threshold', type=float, default=0.1,
                        help='Only cut at VAD probabilities strictly below this value (default: 0.1)')


def validate_long_segment_arguments(args, parser) -> None:
    """Validate report selection without requiring a report for short inputs."""
    if args.vad_report is not None and args.vad_report_dir is not None:
        parser.error('Use only one of --vad-report and --vad-report-dir')
    if args.vad_report is not None and args.input_dir is not None:
        parser.error('--vad-report is for --input-file; use --vad-report-dir with --input-dir')
    if args.vad_report_dir is not None and args.input_file is not None:
        parser.error('--vad-report-dir is for --input-dir; use --vad-report with --input-file')
    if args.long_segment_strategy == 'drop' and (args.vad_report is not None or args.vad_report_dir is not None):
        parser.error('--vad-report and --vad-report-dir require --long-segment-strategy vad')
    for path in (args.vad_report, args.vad_report_dir):
        if path is not None and not path.exists():
            parser.error(f'VAD report path does not exist: {path}')
    if args.vad_report_dir is not None and not args.vad_report_dir.is_dir():
        parser.error(f'--vad-report-dir is not a directory: {args.vad_report_dir}')
    if (not math.isfinite(args.vad_cut_threshold) or
            not 0 <= args.vad_cut_threshold <= 1):
        parser.error('--vad-cut-threshold must be finite and between 0 and 1')


def resolve_vad_report(args, source: Path, parser) -> Path | None:
    """Resolve the report belonging to one source in a single or directory run."""
    if args.long_segment_strategy != 'vad':
        return None
    if args.vad_report is not None:
        return args.vad_report.resolve()
    if args.vad_report_dir is None:
        return None
    report = args.vad_report_dir.resolve() / f'{source.stem}.json'
    if not report.is_file():
        parser.error(f'Missing VAD report for {source.name}: {report}')
    return report


def long_segment_parameters(args, report: Path | None) -> dict:
    """Return cache-safe request parameters for the selected long-turn policy."""
    return {
        'long_segment_strategy': args.long_segment_strategy,
        'vad_device': args.vad_device,
        'vad_cut_threshold': args.vad_cut_threshold,
        'vad_report': identity(report) if report is not None else None,
    }


def _split_long_turns_vad(turns: list[dict], source: Path, max_samples: int,
                          candidates: list[tuple[int, float, int, float]],
                          vad_cut_threshold: float) -> tuple[list[dict], list[dict]]:
    """Split only oversized turns while preserving speaker labels and lineage."""
    info = probe(source)
    split_turns, audit = [], []
    for turn_index, turn in enumerate(turns):
        start_sample, end_sample = turn['start_sample'], turn['end_sample']
        if end_sample - start_sample <= max_samples:
            split_turns.append(turn)
            continue
        leaves, cuts = plan_segments(
            info['frames'], max_samples, candidates,
            start_sample=start_sample, end_sample=end_sample,
            vad_cut_threshold=vad_cut_threshold,
        )
        for cut in cuts:
            audit.append({**cut, 'turn_index': turn_index,
                          'speaker_id': turn.get('speaker_id'),
                          'parent_start_sample': start_sample,
                          'parent_end_sample': end_sample})
        for child_start, child_end, depth, parent_cut_id in leaves:
            child = {key: value for key, value in turn.items()
                     if key not in {'clip', 'clip_sha256', 'clip_frames',
                                    'start_sample', 'end_sample'}}
            lineage = dict(child.get('lineage') or {})
            lineage['long_segment'] = {
                'strategy': 'vad',
                'parent_start_sample': start_sample,
                'parent_end_sample': end_sample,
                'depth': depth,
                'parent_cut_id': parent_cut_id,
            }
            child['lineage'] = lineage
            child['start_sample'] = child_start
            child['end_sample'] = child_end
            child['start_s'] = child_start / info['sample_rate']
            child['end_s'] = child_end / info['sample_rate']
            split_turns.append(child)
    return split_turns, audit


def _mean_duration_s(turns: list[dict], sample_rate: int) -> float | None:
    if not turns:
        return None
    return sum(t['end_sample'] - t['start_sample'] for t in turns) / len(turns) / sample_rate


def _mean_distance_s(mean_duration_s: float | None) -> float:
    if mean_duration_s is None:
        return math.inf
    if mean_duration_s < MEAN_DURATION_MIN_S:
        return MEAN_DURATION_MIN_S - mean_duration_s
    if mean_duration_s > MEAN_DURATION_MAX_S:
        return mean_duration_s - MEAN_DURATION_MAX_S
    return 0.0


def _largest_same_speaker_gap_s(turns: list[dict], sample_rate: int) -> float:
    gaps = [
        (turn['start_sample'] - previous['end_sample']) / sample_rate
        for previous, turn in zip(turns, turns[1:])
        if previous['speaker_id'] == turn['speaker_id']
        and turn['start_sample'] >= previous['end_sample']
    ]
    return max(gaps, default=0.0)


def _merge_detail(mean_duration_s: float | None, merged_turn_count: int,
                  audit: list[dict], clip_count: int) -> str:
    mean = 'n/a' if mean_duration_s is None else f'{mean_duration_s:.2f}s'
    merged_gaps = sum(item['reason'] == 'merged' for item in audit)
    duration_rejections = sum(item['reason'] == 'max_duration_exceeded' for item in audit)
    filtered_out = merged_turn_count - clip_count
    return (f'merged_turns={merged_turn_count}, merged_gaps={merged_gaps}, '
            f'max_duration_rejections={duration_rejections}, filtered_out={filtered_out}, '
            f'valid_clips={clip_count}, mean_duration={mean}')


def export(manifest: dict, source: Path, destination: Path, work_dir: Path, sample_rate: int | None = None, channels: int = 1,
           min_duration_s: float | None = None, max_duration_s: float | None = None,
           *, concurrency: int = 1, batch_size: int = 1,
           long_segment_strategy: str = 'vad', vad_report: Path | None = None,
           vad_device: str = 'cpu', vad_cut_threshold: float = 0.1) -> None:
    import concurrent.futures
    import shutil
    import soundfile as sf
    import threading

    info = probe(source)
    turns = normalize_turns(manifest['turns'], source)
    raw_turns = turns
    merged_turns = raw_turns
    params = manifest.get('parameters', {})
    if min_duration_s is None:
        min_duration_s = params.get('min_duration_s', 1.5)
    if max_duration_s is None:
        max_duration_s = params.get('max_duration_s', 15.0)
    max_samples = None
    if max_duration_s is not None:
        max_samples = math.floor(Decimal(str(max_duration_s)) * info['sample_rate'])
    min_samples = (math.ceil(Decimal(str(min_duration_s)) * info['sample_rate'])
                   if min_duration_s is not None else None)
    if long_segment_strategy not in {'vad', 'drop'}:
        raise ValueError(f'Unsupported long segment strategy: {long_segment_strategy}')
    if (not math.isfinite(vad_cut_threshold) or
            not 0 <= vad_cut_threshold <= 1):
        raise ValueError('vad_cut_threshold must be finite and between 0 and 1')
    vad_candidates = None

    def apply_long_segment_strategy(candidate_turns: list[dict]) -> tuple[list[dict], list[dict]]:
        nonlocal vad_candidates
        if (long_segment_strategy != 'vad' or max_samples is None or
                not any(t['end_sample'] - t['start_sample'] > max_samples for t in candidate_turns)):
            return candidate_turns, []
        if vad_report is None:
            raise FileContractError(
                'VAD is the default long-segment strategy; provide --vad-report '
                '(or --vad-report-dir for directory runs), or select '
                '--long-segment-strategy drop'
            )
        if vad_candidates is None:
            vad_candidates, _ = load_vad_report(source, vad_report, vad_device)
        return _split_long_turns_vad(
            candidate_turns, source, max_samples, vad_candidates, vad_cut_threshold
        )

    merge_details = {}
    long_segment_audit = []
    if manifest.get('operation') == 'diarize' and params.get('merge'):
        merge_config = {key: value for key, value in params['merge'].items() if key != 'adjust_mean'}
        merge_config['max_duration_samples'] = max_samples
        initial_gap_s = merge_config['max_gap_s']
        adjust_mean = params['merge'].get('adjust_mean', True)
        max_duration_detail = 'none' if max_duration_s is None else f'{max_duration_s:.2f}s'
        progress('MERGE_START',
                 f'{source.name}: raw_turns={len(raw_turns)}, initial_max_gap={initial_gap_s:.2f}s, '
                 f'max_duration={max_duration_detail}, target_mean={MEAN_DURATION_MIN_S:.2f}-{MEAN_DURATION_MAX_S:.2f}s, '
                 f'adjust_mean={adjust_mean}, step={MEAN_ADJUST_STEP_S:.2f}s, '
                 f'max_attempts={MEAN_ADJUST_MAX_ATTEMPTS}')

        def run_merge(gap_s: float) -> tuple[list[dict], list[dict]]:
            return merge_turns(source, raw_turns, **{**merge_config, 'max_gap_s': gap_s})

        turns, audit = run_merge(initial_gap_s)
        turns, segment_audit = apply_long_segment_strategy(turns)
        filtered_turns = _filter_duration_turns(turns, min_samples, max_samples)
        mean_duration_s = _mean_duration_s(filtered_turns, info['sample_rate'])
        merged_gaps = sum(item['reason'] == 'merged' for item in audit)
        attempt_records = [{'attempt': 0, 'max_gap_s': initial_gap_s,
                            'merged_turns': len(turns), 'merged_gaps': merged_gaps,
                            'max_duration_rejections': sum(item['reason'] == 'max_duration_exceeded' for item in audit),
                            'valid_clips': len(filtered_turns),
                            'mean_duration_s': mean_duration_s}]
        progress('MERGE_MEAN', f'{source.name}: attempt=0, max_gap={initial_gap_s:.2f}s, '
                 f'{_merge_detail(mean_duration_s, len(turns), audit, len(filtered_turns))}')

        current_gap_s = initial_gap_s
        best_state = (turns, audit, filtered_turns, current_gap_s, mean_duration_s,
                      merged_gaps, segment_audit)
        best_distance_s = _mean_distance_s(mean_duration_s)
        seen_gaps = {round(initial_gap_s, 10)}
        if not adjust_mean:
            stop_reason = 'disabled'
        elif mean_duration_s is None:
            stop_reason = 'no_valid_clips'
        elif MEAN_DURATION_MIN_S <= mean_duration_s <= MEAN_DURATION_MAX_S:
            stop_reason = 'target_reached'
        else:
            largest_gap_s = _largest_same_speaker_gap_s(raw_turns, info['sample_rate'])
            max_adjustable_gap_s = max(initial_gap_s, largest_gap_s)
            stop_reason = 'retry_limit_reached'
            for attempt in range(1, MEAN_ADJUST_MAX_ATTEMPTS + 1):
                if mean_duration_s < MEAN_DURATION_MIN_S:
                    if current_gap_s >= max_adjustable_gap_s:
                        stop_reason = 'gap_upper_bound_reached'
                        break
                    next_gap_s = min(current_gap_s + MEAN_ADJUST_STEP_S, max_adjustable_gap_s)
                    direction = 'increase'
                else:
                    if current_gap_s <= 0:
                        stop_reason = 'gap_lower_bound_reached'
                        break
                    next_gap_s = max(0.0, current_gap_s - MEAN_ADJUST_STEP_S)
                    direction = 'decrease'
                next_gap_s = round(next_gap_s, 10)
                if next_gap_s in seen_gaps:
                    stop_reason = 'gap_adjustment_oscillated'
                    break
                seen_gaps.add(next_gap_s)
                progress('MERGE_RETRY',
                         f'{source.name}: attempt={attempt}/{MEAN_ADJUST_MAX_ATTEMPTS}, '
                         f'mean={mean_duration_s:.2f}s outside target, {direction} max_gap '
                         f'{current_gap_s:.2f}s -> {next_gap_s:.2f}s')
                turns, audit = run_merge(next_gap_s)
                turns, segment_audit = apply_long_segment_strategy(turns)
                filtered_turns = _filter_duration_turns(turns, min_samples, max_samples)
                mean_duration_s = _mean_duration_s(filtered_turns, info['sample_rate'])
                current_gap_s = next_gap_s
                merged_gaps = sum(item['reason'] == 'merged' for item in audit)
                attempt_records.append({'attempt': attempt, 'max_gap_s': current_gap_s,
                                        'merged_turns': len(turns), 'merged_gaps': merged_gaps,
                                        'max_duration_rejections': sum(item['reason'] == 'max_duration_exceeded' for item in audit),
                                        'valid_clips': len(filtered_turns),
                                        'mean_duration_s': mean_duration_s})
                progress('MERGE_MEAN', f'{source.name}: attempt={attempt}, max_gap={current_gap_s:.2f}s, '
                         f'{_merge_detail(mean_duration_s, len(turns), audit, len(filtered_turns))}')
                distance_s = _mean_distance_s(mean_duration_s)
                if distance_s < best_distance_s:
                    best_state = (turns, audit, filtered_turns, current_gap_s, mean_duration_s,
                                  merged_gaps, segment_audit)
                    best_distance_s = distance_s
                if mean_duration_s is None:
                    stop_reason = 'no_valid_clips'
                    break
                if MEAN_DURATION_MIN_S <= mean_duration_s <= MEAN_DURATION_MAX_S:
                    stop_reason = 'target_reached'
                    break
            else:
                stop_reason = 'retry_limit_reached'

        turns, audit, filtered_turns, current_gap_s, mean_duration_s, merged_gaps, long_segment_audit = best_state
        # Statistics describe the final merge before duration filtering.
        merge_statistics = {'input_turns': len(raw_turns), 'output_turns': len(turns),
                            'merged_gaps': merged_gaps,
                            'max_duration_rejections': sum(item['reason'] == 'max_duration_exceeded' for item in audit),
                            'max_gap_s': current_gap_s}
        merge_mean_adjustment = {
            'enabled': adjust_mean,
            'target_min_duration_s': MEAN_DURATION_MIN_S,
            'target_max_duration_s': MEAN_DURATION_MAX_S,
            'gap_step_s': MEAN_ADJUST_STEP_S,
            'initial_max_gap_s': initial_gap_s,
            'final_max_gap_s': current_gap_s,
            'final_mean_duration_s': mean_duration_s,
            'final_valid_clips': len(filtered_turns),
            'stop_reason': stop_reason,
            'attempts': attempt_records,
        }
        progress('MERGE_DONE',
                 f'{source.name}: final_max_gap={current_gap_s:.2f}s, '
                 f'{_merge_detail(mean_duration_s, len(turns), audit, len(filtered_turns))}, '
                 f'stop_reason={stop_reason}')
        merged_turns = turns
        turns = filtered_turns
        merge_details = {'merge_applied': True, 'merge_statistics': merge_statistics,
                         'merge_audit': audit, 'merge_mean_adjustment': merge_mean_adjustment}
        progress('MERGE_COMPLETE', f'{source.name}: {len(raw_turns)} raw turns -> {len(turns)} valid clips; exporting')
    else:
        merged_turns, long_segment_audit = apply_long_segment_strategy(turns)
    # Keep the unfiltered merged stage available for plotting and provenance.
    merged_turns = normalize_turns(merged_turns, source)
    turns = _filter_duration_turns(merged_turns, min_samples, max_samples)
    # overlap_with indices must refer to the surviving output turns.
    turns = normalize_turns(turns, source)
    turns_count = len(turns)
    progress('EXPORT', f'Exporting {turns_count} clip(s) from {source.name} (concurrency={concurrency}, batch_size={batch_size})')
    output = {**manifest, **merge_details, 'long_segment_audit': long_segment_audit,
              'timestamp_origin': 'diarized_input', 'source_sample_rate': info['sample_rate'],
              'sample_rate': sample_rate or info['sample_rate'], 'channels': channels, 'turns': turns,
              'speaker_ids': sorted({t['speaker_id'] for t in turns}), 'complete': False}
    for i, turn in enumerate(turns, 1):
        turn['clip'] = (f"{safe_name(source.stem)}_{safe_name(str(manifest.get('model') or 'segments'))}_"
                        f"{safe_name(turn['speaker_id'])}_{round(turn['start_s'] * 1000):09d}-"
                        f"{round(turn['end_s'] * 1000):09d}_{i:04d}.wav")
        if (destination.parent / turn['clip']).resolve() == source.resolve():
            raise FileContractError('Clip would overwrite source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = read_json(destination)
    except (ValueError, OSError):
        old = {}
    # Remove only clips owned by the previous export, while its inventory is
    # still on disk. Planned names also make interrupted exports cleanable.
    current_clips = {t['clip'] for t in turns}
    if old.get('operation') in {'diarize', 'export_segments'}:
        for turn in old.get('turns', []):
            name = turn.get('clip')
            if not isinstance(name, str) or name in current_clips:
                continue
            if Path(name).name != name or Path(name).suffix != '.wav':
                continue
            clip = destination.parent / name
            if clip.resolve() == source.resolve():
                raise FileContractError('Cannot remove source audio during clip cleanup')
            clip.unlink(missing_ok=True)
    # A retry recognizes the request and every clip this export may publish.
    write_json(destination, output)
    if manifest.get('operation') == 'diarize':
        raw_path = destination.with_name('segments.raw.json')
        write_json(raw_path, {**manifest, 'turns': raw_turns,
                   'speaker_ids': sorted({t['speaker_id'] for t in raw_turns}),
                   'timestamp_origin': 'diarized_input', 'source_sample_rate': info['sample_rate'],
                   'merge_applied': False, 'duration_filter_applied': False,
                   'clips_valid': False, 'complete': True})
        output['raw_manifest_sha256'] = digest(raw_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, turns_count // 10)
    lock = threading.Lock()

    with tempfile.TemporaryDirectory(dir=work_dir) as directory, sf.SoundFile(source) as audio:
        work = Path(directory)

        def render_clip(i: int, turn: dict) -> None:
            with lock:
                audio.seek(turn['start_sample'])
                data = audio.read(turn['end_sample'] - turn['start_sample'], dtype='float32', always_2d=True)
            raw, staged = work / f'raw_{i}.wav', work / f'clip_{i}.wav'
            sf.write(raw, data, info['sample_rate'], subtype='FLOAT')
            del data
            convert(raw, staged, output['sample_rate'], channels)
            name = turn['clip']
            clip = destination.parent / name
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

        if concurrency <= 1:
            for i, turn in enumerate(turns, 1):
                render_clip(i, turn)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for offset in range(0, turns_count, max(1, batch_size)):
                    futures = [pool.submit(render_clip, i, turn) for i, turn in
                               enumerate(turns[offset:offset + max(1, batch_size)], offset + 1)]
                    for fut in concurrent.futures.as_completed(futures):
                        fut.result()

    output['complete'] = True
    if manifest.get('operation') == 'diarize':
        merged_path = destination.with_name('segments.merged.json')
        merged_output = {
            **manifest,
            **merge_details,
            'long_segment_audit': long_segment_audit,
            'timestamp_origin': 'diarized_input',
            'source_sample_rate': info['sample_rate'],
            'merge_applied': bool(params.get('merge')),
            'duration_filter_applied': False,
            'clips_valid': False,
            'turns': merged_turns,
            'speaker_ids': sorted({t['speaker_id'] for t in merged_turns}),
            'complete': True,
        }
        write_json(merged_path, merged_output)
        output['merged_manifest_sha256'] = digest(merged_path)
    write_json(destination, output)
    progress('EXPORT_COMPLETE', f'Exported {turns_count} clips to {destination.parent.name}')


def ensure_plots(manifest_path: Path, *, overwrite: bool = False) -> None:
    """Write stage plots and final timeline, duration, and cutoff plots."""
    from _common.diarize_plots import (manifest_plot_file, plot_segment_outputs,
                                       sibling_plot_paths, stage_plot_file, write_stage_plots)

    stage_manifests = (
        ('before_merge', manifest_path.with_name('segments.raw.json')),
        ('after_merge', manifest_path.with_name('segments.merged.json')),
    )
    for stage, stage_manifest in stage_manifests:
        if not stage_manifest.is_file():
            continue
        output = stage_plot_file(manifest_path, stage)
        if not overwrite and all(path.exists() for path in sibling_plot_paths(output)):
            continue
        progress('PLOT_START', f'Rendering diarization plots: {stage} ({stage_manifest.name})')
        write_stage_plots(stage_manifest, stage, overwrite=True)

    output = manifest_plot_file(manifest_path)
    if not overwrite and all(path.exists() for path in sibling_plot_paths(output)):
        return
    progress('PLOT_START', f'Rendering diarization plots: {manifest_path.name}')
    plot_segment_outputs(manifest_path, overwrite=overwrite)

def ensure_family_plots(
    pairs: list[tuple[Path, Path]],
    args,
    *,
    overwrite: bool = True,
    concurrency: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Render aggregate stage plots after a directory diarization run.

    Groups completed manifests by their enclosing output folder
    (`<collection>/<stem>/segments.json`), not inferred video IDs.
    """
    if getattr(args, 'input_dir', None) is None or getattr(args, 'input_file', None) is not None:
        return

    from _common.diarize_plots import FAMILY_PLOT_DIR, ensure_family_plot_link, write_family_plots

    groups: dict[Path, list[Path]] = {}
    for _, destination in pairs:
        # destination is <collection>/<clip_stem>/segments.json
        groups.setdefault(destination.parent.parent, []).append(destination)

    for family_root, destinations in groups.items():
        manifests = [path for path in destinations if path.is_file()]
        if not manifests:
            continue
        progress('FAMILY_PLOT_START', f'Rendering aggregate diarization plots for {family_root.name} ({len(manifests)} manifest(s))')
        paths = write_family_plots(
            manifests,
            family_root / FAMILY_PLOT_DIR,
            root_dir=family_root,
            overwrite=overwrite,
            concurrency=concurrency or getattr(args, 'concurrency', 1),
            batch_size=batch_size or getattr(args, 'batch_size', 1),
        )
        ensure_family_plot_link(family_root)
        if paths:
            print(paths[0], flush=True)
