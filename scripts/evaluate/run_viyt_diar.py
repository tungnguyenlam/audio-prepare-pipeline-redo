"""Run current diarization launchers on a prepared ViYT-Diar cache and score DER/JER."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (LoggingArgumentParser, persist_path, positive_int, progress,
                           read_json, ROOT, write_json)
from diarization import evaluate_diarization
from prepare_viyt_diar import DEFAULT_OUTPUT_DIR, VIYT_DIAR_DATASET_ID, _load_cached_samples

DEFAULT_COLLAR_S = 0.25
CLIP_SKIP_DURATION_S = 86400.0

SYSTEMS: dict[str, dict[str, str | None]] = {
    'pyannote_community1': {
        'label': 'Pyannote Community-1',
        'launcher': 'scripts/s3-diarize/pyannote_community1.sh',
        'oracle_flag': '--num-speakers',
        'python': '.venvs/sortformer/bin/python',
    },
    'pyannote_31': {
        'label': 'Pyannote 3.1',
        'launcher': 'scripts/s3-diarize/pyannote_31.sh',
        'oracle_flag': '--num-speakers',
        'python': '.venvs/sortformer/bin/python',
    },
    'sortformer': {
        'label': 'NeMo Sortformer',
        'launcher': 'scripts/s3-diarize/sortformer.sh',
        'oracle_flag': None,
        'python': None,
    },
    'clustering': {
        'label': 'NeMo Clustering',
        'launcher': 'scripts/s3-diarize/clustering.sh',
        'oracle_flag': '--num-speakers',
        'python': None,
    },
    'threed_speaker': {
        'label': '3D-Speaker',
        'launcher': 'scripts/s3-diarize/threed_speaker.sh',
        'oracle_flag': '--num-speakers',
        'python': None,
    },
    'diarizen': {
        'label': 'DiariZen Large s80-v2',
        'launcher': 'scripts/s3-diarize/diarizen.sh',
        'oracle_flag': '--num-speakers',
        'python': None,
    },
}
ALL_SYSTEM_KEYS = tuple(SYSTEMS)


def _load_dotenv(root: Path) -> None:
    """Load repo-root .env into empty os.environ keys; never logs values."""
    path = root / '.env'
    if not path.is_file():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_systems(raw: str | None, *, run_all: bool) -> list[str]:
    if run_all:
        return list(ALL_SYSTEM_KEYS)
    if raw is None or not raw.strip():
        return ['pyannote_community1']
    resolved: list[str] = []
    seen: set[str] = set()
    for part in raw.split(','):
        key = part.strip().lower().replace('-', '_')
        if not key or key in seen:
            continue
        if key == 'all':
            for registry_key in ALL_SYSTEM_KEYS:
                if registry_key not in seen:
                    resolved.append(registry_key)
                    seen.add(registry_key)
            continue
        if key == '3d_speaker':
            key = 'threed_speaker'
        if key == 'pyannote_community':
            key = 'pyannote_community1'
        if key not in SYSTEMS:
            known = ', '.join((*ALL_SYSTEM_KEYS, 'all'))
            raise ValueError(f'Unknown system {part!r}. Choose from: {known}')
        resolved.append(key)
        seen.add(key)
    return resolved or ['pyannote_community1']


def _launcher_cmd(spec: dict[str, str | None], extra: list[str], *, device: str,
                  overwrite: bool) -> list[str]:
    cmd = [
        'bash', str(ROOT / str(spec['launcher'])),
        '--merge', 'false',
        '--min-duration-s', str(CLIP_SKIP_DURATION_S),
        '--max-duration-s', str(CLIP_SKIP_DURATION_S),
        '--device', device,
        *extra,
    ]
    if overwrite:
        cmd.append('--overwrite')
    return cmd


def _run_launcher(cmd: list[str], *, python: str | None = None) -> int:
    progress('VIYT_DIARIZE', ' '.join(cmd))
    env = os.environ.copy()
    if python:
        python_path = Path(python)
        if not python_path.is_absolute():
            python_path = ROOT / python_path
        if python_path.is_file():
            env['DIARIZATION_PYTHON'] = str(python_path)
            progress('VIYT_DIARIZE_ENV', f'DIARIZATION_PYTHON={python_path}')
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(completed.returncode)


def _subset_audio_dir(samples: list[dict], work_dir: Path) -> Path:
    import shutil
    subset = work_dir / 'audio'
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        dest = subset / Path(sample['audio_path']).name
        dest.symlink_to(Path(sample['audio_path']).resolve())
    return subset


def _score_sample(sample: dict, hypothesis_path: Path, *, collar_s: float) -> dict:
    reference = read_json(sample['reference_path'])
    hypothesis = read_json(hypothesis_path)
    ref_turns = list(reference.get('turns') or [])
    hyp_turns = list(hypothesis.get('turns') or [])
    duration_s = float(sample['duration_s'])
    ends = [float(turn['end_s']) for turn in ref_turns + hyp_turns if turn.get('end_s') is not None]
    if ends:
        duration_s = max(duration_s, max(ends))
    metrics = evaluate_diarization(ref_turns, hyp_turns, duration_s=duration_s, collar_s=collar_s)
    hyp_speakers = sorted({str(turn.get('speaker_id') or '') for turn in hyp_turns if turn.get('speaker_id')})
    ref_speakers = list(sample['speakers'])
    reference_s = float(metrics['reference_speaker_s'])
    return {
        'audio_id': sample['audio_id'],
        'stem': sample.get('stem') or Path(sample['audio_path']).stem,
        'duration_s': sample['duration_s'],
        'der_pct': metrics['der_pct'],
        'jer_pct': metrics['jer_pct'],
        'false_alarm_s': metrics['false_alarm_s'],
        'missed_speech_s': metrics['missed_speech_s'],
        'speaker_confusion_s': metrics['speaker_confusion_s'],
        'reference_speaker_s': reference_s,
        'hypothesis_speaker_s': metrics['hypothesis_speaker_s'],
        'scored_audio_s': metrics['scored_audio_s'],
        'false_alarm_pct': round((metrics['false_alarm_s'] / reference_s * 100) if reference_s else 0.0, 4),
        'missed_detection_pct': round((metrics['missed_speech_s'] / reference_s * 100) if reference_s else 0.0, 4),
        'confusion_pct': round((metrics['speaker_confusion_s'] / reference_s * 100) if reference_s else 0.0, 4),
        'ref_speakers': len(ref_speakers),
        'hyp_speakers': len(hyp_speakers),
        'speaker_count_abs_error': abs(len(hyp_speakers) - len(ref_speakers)),
        'hypothesis_path': persist_path(hypothesis_path),
    }


def _summarize(system: str, label: str, file_metrics: list[dict], *, elapsed_s: float,
               failures: list[dict], returncode: int) -> dict:
    n = len(file_metrics)
    ders = [row['der_pct'] for row in file_metrics]
    jers = [row['jer_pct'] for row in file_metrics]
    error_s = sum(row['der_pct'] / 100.0 * row['reference_speaker_s'] for row in file_metrics)
    reference_s = sum(row['reference_speaker_s'] for row in file_metrics)
    return {
        'system': system,
        'label': label,
        'num_files': n,
        'mean_der_pct': round(statistics.fmean(ders), 4) if ders else None,
        'median_der_pct': round(statistics.median(ders), 4) if ders else None,
        'weighted_der_pct': round((error_s / reference_s * 100) if reference_s else 0.0, 4) if n else None,
        'mean_jer_pct': round(statistics.fmean(jers), 4) if jers else None,
        'mean_false_alarm_pct': round(statistics.fmean([row['false_alarm_pct'] for row in file_metrics]), 4) if n else None,
        'mean_missed_detection_pct': round(statistics.fmean([row['missed_detection_pct'] for row in file_metrics]), 4) if n else None,
        'mean_confusion_pct': round(statistics.fmean([row['confusion_pct'] for row in file_metrics]), 4) if n else None,
        'mean_speaker_count_abs_error': round(statistics.fmean([row['speaker_count_abs_error'] for row in file_metrics]), 4) if n else None,
        'total_duration_s': round(sum(row['duration_s'] for row in file_metrics), 3),
        'elapsed_s': round(elapsed_s, 3),
        'failures': failures,
        'returncode': returncode,
        'aborted': returncode != 0 and not file_metrics,
    }


def _run_system(spec_key: str, samples: list[dict], *, audio_dir: Path, output_root: Path,
                device: str, overwrite: bool, oracle_speakers: bool) -> dict:
    spec = SYSTEMS[spec_key]
    hyp_dir = output_root / 'hypotheses' / spec_key
    work_dir = output_root / 'work' / spec_key
    hyp_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    returncode = 0
    if oracle_speakers and spec['oracle_flag']:
        for sample in samples:
            extra = [
                '--input-file', str(sample['audio_path']),
                '--output-dir', str(hyp_dir),
                '--work-dir', str(work_dir),
                str(spec['oracle_flag']), str(len(sample['speakers'])),
            ]
            code = _run_launcher(
                _launcher_cmd(spec, extra, device=device, overwrite=overwrite),
                python=spec.get('python'),
            )
            if code != 0:
                returncode = code
    else:
        extra = [
            '--input-dir', str(audio_dir),
            '--output-dir', str(hyp_dir),
            '--work-dir', str(work_dir),
        ]
        returncode = _run_launcher(
            _launcher_cmd(spec, extra, device=device, overwrite=overwrite),
            python=spec.get('python'),
        )
        if oracle_speakers and spec['oracle_flag'] is None:
            progress('VIYT_ORACLE_SKIP', f'{spec_key} has no --num-speakers flag; scored estimated speakers')
    return {
        'system': spec_key,
        'label': spec['label'],
        'launcher': spec['launcher'],
        'elapsed_s': time.perf_counter() - started,
        'returncode': returncode,
        'hypothesis_dir': persist_path(hyp_dir),
    }


def _score_system(samples: list[dict], hyp_dir: Path, *, collar_s: float) -> tuple[list[dict], list[dict]]:
    file_metrics: list[dict] = []
    failures: list[dict] = []
    for sample in samples:
        hypothesis_path = hyp_dir / (sample.get('stem') or Path(sample['audio_path']).stem) / 'segments.raw.json'
        if not hypothesis_path.is_file():
            failures.append({'audio_id': sample['audio_id'], 'error': f'missing {hypothesis_path.name}'})
            continue
        try:
            file_metrics.append(_score_sample(sample, hypothesis_path, collar_s=collar_s))
        except (TypeError, ValueError, OSError) as exc:
            failures.append({'audio_id': sample['audio_id'], 'error': str(exc)})
    return file_metrics, failures


def _export_figures(summaries: list[dict], per_system_files: dict[str, list[dict]],
                    output_dir: Path, run_id: str) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if not summaries:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    labels = [row['label'] or row['system'] for row in summaries]
    palette = ('#2f6fed', '#c45c26', '#2a9d8f', '#7b2cbf', '#e9c46a', '#264653', '#e76f51')
    colors = [palette[i % len(palette)] for i in range(len(summaries))]
    multi = len(summaries) > 1

    def _save(fig, name: str) -> None:
        path = output_dir / f'{run_id}_{name}.png'
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    ders = [row['mean_der_pct'] or 0.0 for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    bars = ax.bar(labels, ders, color=colors, edgecolor='none')
    ax.set_ylabel('Mean DER (%)')
    ax.set_title('ViYT-Diar — Mean Diarization Error Rate')
    ax.set_ylim(0, max(ders + [1.0]) * 1.25)
    for bar, value in zip(bars, ders, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.1f}',
                ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', labelrotation=20)
    _save(fig, 'mean_der')

    jers = [row['mean_jer_pct'] or 0.0 for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    bars = ax.bar(labels, jers, color=colors, edgecolor='none')
    ax.set_ylabel('Mean JER (%)')
    ax.set_title('ViYT-Diar — Mean Jaccard Error Rate')
    ax.set_ylim(0, max(jers + [1.0]) * 1.25)
    for bar, value in zip(bars, jers, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.1f}',
                ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', labelrotation=20)
    _save(fig, 'mean_jer')

    fig, ax = plt.subplots(figsize=(10, 5.0))
    box_data = [
        [row['der_pct'] for row in per_system_files.get(item['system'], [])] or [float('nan')]
        for item in summaries
    ]
    ax.boxplot(box_data, tick_labels=labels, showmeans=True)
    ax.set_ylabel('DER (%)')
    ax.set_title('ViYT-Diar — Per-file DER distribution')
    ax.tick_params(axis='x', labelrotation=20)
    _save(fig, 'der_boxplot')

    fig, ax = plt.subplots(figsize=(10, 5.0))
    sc_errs = [row['mean_speaker_count_abs_error'] or 0.0 for row in summaries]
    ax.bar(labels, sc_errs, color='#c45c26', edgecolor='none')
    ax.set_ylabel('Mean |#spk_hyp − #spk_ref|')
    ax.set_title('ViYT-Diar — Speaker count absolute error')
    ax.tick_params(axis='x', labelrotation=20)
    _save(fig, 'speaker_count_error')

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(summaries))
    width = 0.25
    fa = [row['mean_false_alarm_pct'] or 0.0 for row in summaries]
    miss = [row['mean_missed_detection_pct'] or 0.0 for row in summaries]
    conf = [row['mean_confusion_pct'] or 0.0 for row in summaries]
    ax.bar(x - width, fa, width, label='False alarm', color='#e9c46a')
    ax.bar(x, miss, width, label='Missed detection', color='#e76f51')
    ax.bar(x + width, conf, width, label='Confusion', color='#264653')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel('Mean component rate (%)')
    ax.set_title('ViYT-Diar — DER components')
    ax.legend(frameon=False)
    _save(fig, 'der_components')

    if not multi:
        files = list(per_system_files.get(summaries[0]['system'], []))
        if files:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(range(len(files)), [row['der_pct'] for row in files], color=colors[0], edgecolor='none')
            ax.set_xlabel('File index')
            ax.set_ylabel('DER (%)')
            ax.set_title(f'ViYT-Diar — Per-file DER ({labels[0]})')
            _save(fig, 'per_file_der')
    else:
        id_lists = [[row['audio_id'] for row in per_system_files.get(item['system'], [])] for item in summaries]
        if id_lists and all(id_lists):
            common_ids = list(dict.fromkeys(id_lists[0]))
            for ids in id_lists[1:]:
                common_ids = [audio_id for audio_id in common_ids if audio_id in set(ids)]
            if common_ids:
                fig, ax = plt.subplots(figsize=(12, 5.2))
                xs = np.arange(len(common_ids))
                for summary, color, label in zip(summaries, colors, labels, strict=True):
                    by_id = {row['audio_id']: row['der_pct'] for row in per_system_files.get(summary['system'], [])}
                    ax.plot(xs, [by_id[audio_id] for audio_id in common_ids], marker='o', markersize=3,
                            linewidth=1.2, label=label, color=color)
                ax.set_xlabel('File index (shared clips)')
                ax.set_ylabel('DER (%)')
                ax.set_title('ViYT-Diar — Per-file DER by model')
                ax.legend(frameon=False, fontsize=8, loc='upper right')
                _save(fig, 'per_file_der_compare')
    return [persist_path(path) for path in written]


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        '# ViYT-Diar benchmark',
        '',
        f'- Dataset: `{payload["dataset_id"]}`',
        f'- Collar: {payload["collar_s"]} s',
        f'- Device: `{payload["device"]}`',
        f'- Oracle speakers: {payload["oracle_speakers"]}',
        f'- Clips: {payload["num_samples"]}',
        f'- Run id: `{payload["run_id"]}`',
        '',
        '| System | Files | Mean DER % | Weighted DER % | Median DER % | Mean JER % | FA % | Miss % | Conf % | |spk| err | Seconds |',
        '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for item in payload['systems']:
        summary = item['summary']
        def fmt(value):
            return '—' if value is None else f'{value:.2f}'
        lines.append(
            f'| {summary["label"]} | {summary["num_files"]} | {fmt(summary["mean_der_pct"])} | '
            f'{fmt(summary["weighted_der_pct"])} | {fmt(summary["median_der_pct"])} | '
            f'{fmt(summary["mean_jer_pct"])} | {fmt(summary["mean_false_alarm_pct"])} | '
            f'{fmt(summary["mean_missed_detection_pct"])} | {fmt(summary["mean_confusion_pct"])} | '
            f'{fmt(summary["mean_speaker_count_abs_error"])} | {summary["elapsed_s"]:.1f} |'
        )
    failures = [item for item in payload['systems'] if item['summary'].get('failures')]
    if failures:
        lines.extend(['', '## Failures', ''])
        for item in failures:
            lines.append(f'- **{item["summary"]["label"]}**: {len(item["summary"]["failures"])} clip(s)')
            for failure in item['summary']['failures'][:10]:
                lines.append(f'  - `{failure["audio_id"]}`: {failure["error"]}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    _load_dotenv(ROOT)
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-id', '--input-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                   help='Prepared ViYT-Diar cache root (from prepare_viyt_diar)')
    p.add_argument('-od', '--output-dir', type=Path, default=None,
                   help='Hypothesis/results/figures root (default: same as --input-dir)')
    p.add_argument('--systems', default=None,
                   help=f'Comma-separated system keys (default: pyannote_community1). Known: {", ".join((*ALL_SYSTEM_KEYS, "all"))}')
    p.add_argument('--all', action='store_true', help='Run every registered system sequentially')
    p.add_argument('-n', '-l', '--limit', type=positive_int, default=None, help='Optional max number of clips')
    p.add_argument('-d', '--device', default='auto', help='Device forwarded to diarizer launchers')
    p.add_argument('--collar', type=float, default=DEFAULT_COLLAR_S, help='DER forgiveness collar in seconds')
    p.add_argument('--oracle-speakers', action='store_true',
                   help='Pass reference speaker count via --num-speakers when the launcher supports it')
    p.add_argument('--run-id', default=None, help='Optional run id prefix for result/figure filenames')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', help='Forward --overwrite to diarizer launchers')
    args = p.parse_args()
    if args.all and args.systems:
        p.error('Pass either --all or --systems, not both')
    if not isfinite(args.collar) or args.collar < 0:
        p.error('--collar must be finite and non-negative')
    try:
        system_keys = _parse_systems(args.systems, run_all=args.all)
    except ValueError as exc:
        p.error(str(exc))

    cache_dir = args.input_dir.resolve()
    output_root = (args.output_dir or args.input_dir).resolve()
    samples = _load_cached_samples(cache_dir, limit=args.limit)
    if not samples:
        p.error(f'No prepared ViYT-Diar clips in {cache_dir}. Run scripts/evaluate/prepare_viyt_diar.sh first.')

    audio_dir = cache_dir / 'audio'
    if args.limit is not None:
        audio_dir = _subset_audio_dir(samples, output_root / 'work' / 'subset')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    results_dir = output_root / 'results'
    figures_dir = output_root / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)

    progress('VIYT_RUN', f'{len(system_keys)} system(s) × {len(samples)} clip(s): {", ".join(system_keys)}')
    per_system_files: dict[str, list[dict]] = {}
    system_rows: list[dict] = []
    for index, key in enumerate(system_keys, start=1):
        progress('VIYT_SYSTEM', f'{SYSTEMS[key]["label"]} ({key})', current=index, total=len(system_keys))
        try:
            meta = _run_system(
                key, samples, audio_dir=audio_dir, output_root=output_root,
                device=args.device, overwrite=args.overwrite, oracle_speakers=args.oracle_speakers,
            )
            file_metrics, failures = _score_system(
                samples, output_root / 'hypotheses' / key, collar_s=args.collar,
            )
            meta['failures'] = failures
        except Exception as exc:
            progress('VIYT_SYSTEM_FAIL', f'{key}: {exc}')
            file_metrics = []
            meta = {
                'system': key,
                'label': SYSTEMS[key]['label'],
                'launcher': SYSTEMS[key]['launcher'],
                'elapsed_s': 0.0,
                'returncode': 1,
                'failures': [{'audio_id': '*', 'error': str(exc)}],
                'hypothesis_dir': persist_path(output_root / 'hypotheses' / key),
            }
        summary = _summarize(key, str(SYSTEMS[key]['label']), file_metrics,
                             elapsed_s=float(meta.get('elapsed_s', 0.0)),
                             failures=list(meta.get('failures') or []),
                             returncode=int(meta.get('returncode') or 0))
        per_system_files[key] = file_metrics
        row = {**meta, 'summary': summary, 'per_file': file_metrics}
        system_rows.append(row)
        checkpoint = results_dir / f'{run_id}_{key}.json'
        write_json(checkpoint, {
            'schema_version': 1,
            'operation': 'viyt_diar_system',
            'run_id': run_id,
            'dataset_id': VIYT_DIAR_DATASET_ID,
            'collar_s': args.collar,
            'device': args.device,
            'oracle_speakers': args.oracle_speakers,
            'limit': args.limit,
            'num_samples': len(samples),
            'system': row,
        })
        progress('VIYT_SYSTEM_DONE',
                 f'{key} mean DER={summary["mean_der_pct"]} weighted={summary["weighted_der_pct"]} '
                 f'n={summary["num_files"]} ({summary["elapsed_s"]:.1f}s) -> {checkpoint.name}')

    payload = {
        'schema_version': 1,
        'operation': 'viyt_diar_benchmark',
        'run_id': run_id,
        'dataset_id': VIYT_DIAR_DATASET_ID,
        'collar_s': args.collar,
        'device': args.device,
        'oracle_speakers': args.oracle_speakers,
        'limit': args.limit,
        'num_samples': len(samples),
        'systems': system_rows,
        'per_file': per_system_files,
    }
    result_path = results_dir / f'{run_id}_viyt_diar.json'
    write_json(result_path, payload)
    figure_paths = _export_figures(
        [row['summary'] for row in system_rows], per_system_files, figures_dir, run_id,
    )
    payload['figures'] = figure_paths
    write_json(result_path, payload)
    report_path = results_dir / f'{run_id}_report.md'
    _write_report(report_path, payload)
    progress('VIYT_RUN_DONE', f'{result_path}')
    print(result_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
