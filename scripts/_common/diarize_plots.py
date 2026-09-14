"""Diarization timeline, duration histogram, and min-duration cutoff plots."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import statistics
import subprocess

from _common.files import ROOT, progress, read_json


def sibling_plot_paths(gantt_file: Path) -> tuple[Path, Path, Path]:
    gantt = gantt_file.resolve()
    return gantt, gantt.with_name(f'{gantt.stem}_duration{gantt.suffix}'), gantt.with_name(f'{gantt.stem}_cutoff{gantt.suffix}')


def plot_segment_outputs(manifest_path: Path, *, overwrite: bool = False, bin_width: float = 0.25) -> list[Path]:
    """Write timeline, duration histogram, and cutoff plots next to a segments.json."""
    output_file = Path(manifest_path).resolve().parent / 'timeline.png'
    paths = list(sibling_plot_paths(output_file))
    if not overwrite and all(path.exists() for path in paths):
        return paths
    try:
        return write_plots(manifest_path, output_file, overwrite=True, bin_width=bin_width)
    except ImportError:
        return _write_plots_via_audio_python(manifest_path, output_file, bin_width=bin_width)


def write_plots(
    manifest_path: Path,
    output_file: Path,
    *,
    reference_manifest: Path | None = None,
    title: str | None = None,
    overwrite: bool = False,
    concurrency: int = 1,
    batch_size: int = 1,
    bin_width: float = 0.25,
) -> list[Path]:
    dest = Path(output_file).resolve()
    gantt, duration_path, cutoff_path = sibling_plot_paths(dest)
    existing = [path for path in (gantt, duration_path, cutoff_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Destination exists: {existing[0]}; use --overwrite')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if reference_manifest is not None:
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_hyp = ex.submit(read_json, manifest_path)
                fut_ref = ex.submit(read_json, reference_manifest)
                data = fut_hyp.result()
                ref_data = fut_ref.result()
        else:
            data = read_json(manifest_path)
            ref_data = read_json(reference_manifest)
        turns = data.get('turns', [])
        ref_turns = ref_data.get('turns', [])
        fig, (ax_ref, ax_hyp) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        _plot_turns(ax_ref, ref_turns, 'Reference Diarization', batch_size=batch_size)
        _plot_turns(ax_hyp, turns, 'Hypothesis Diarization', batch_size=batch_size)
        if title:
            fig.suptitle(title, fontsize=14)
    else:
        data = read_json(manifest_path)
        turns = data.get('turns', [])
        fig, ax = plt.subplots(figsize=(14, max(3.0, 1.2 + 0.5 * len({t['speaker_id'] for t in turns}))))
        source = data.get('source', {}).get('path', Path(manifest_path).stem)
        _plot_turns(ax, turns, title or f'Diarization: {source}', batch_size=batch_size)

    dest.parent.mkdir(parents=True, exist_ok=True)
    _save(plt, fig, gantt)
    progress('PLOT_DONE', f'Saved plot to {gantt.name}')
    _write_duration_histogram(plt, turns, duration_path, bin_width=bin_width)
    progress('PLOT_DONE', f'Saved plot to {duration_path.name}')
    _write_cutoff_bars(plt, turns, cutoff_path, bin_width=bin_width)
    progress('PLOT_DONE', f'Saved plot to {cutoff_path.name}')
    return [gantt, duration_path, cutoff_path]


def _plot_turns(ax, turns: list[dict], title: str, batch_size: int = 1):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    speakers = sorted({t['speaker_id'] for t in turns})
    if not speakers:
        speakers = ['SPEAKER_00']
    spk_to_y = {spk: idx for idx, spk in enumerate(reversed(speakers))}
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(len(speakers))]

    batch_size = max(1, batch_size)
    for i in range(0, len(turns), batch_size):
        chunk = turns[i:i + batch_size]
        for t in chunk:
            dur = t['end_s'] - t['start_s']
            spk = t['speaker_id']
            y = spk_to_y[spk]
            color = colors[speakers.index(spk) % len(colors)]
            overlap = t.get('overlap', False) or t.get('overlaps_other_speaker', False)
            edgecolor = '#ef4444' if overlap else 'none'
            linewidth = 1.5 if overlap else 0
            ax.barh(y=y, width=dur, left=t['start_s'], height=0.6, color=color,
                    edgecolor=edgecolor, linewidth=linewidth, alpha=0.85)

    has_overlap = any(t.get('overlap', False) or t.get('overlaps_other_speaker', False) for t in turns)
    if has_overlap:
        patch = mpatches.Patch(facecolor='#fca5a5', edgecolor='#ef4444', linewidth=1.5, label='Overlap')
        ax.legend(handles=[patch], loc='upper right', framealpha=0.85)

    ax.set_yticks([spk_to_y[s] for s in speakers])
    ax.set_yticklabels(speakers)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)


def _write_duration_histogram(plt, turns: list[dict], dest: Path, *, bin_width: float = 0.25) -> None:
    durations = _turn_durations(turns)
    fig, ax = plt.subplots(figsize=(10, 4))
    if not durations:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No turns to plot', ha='center', va='center')
        ax.set_title('Segment duration')
        _save(plt, fig, dest)
        return
    step = bin_width if (math.isfinite(bin_width) and bin_width > 0) else 0.25
    start = math.floor(min(durations) / step) * step
    end = math.ceil(max(durations) / step) * step
    if end <= start:
        end = start + step
    num_bins = max(1, round((end - start) / step))
    bins = [round(start + i * step, 4) for i in range(num_bins + 1)]
    ax.hist(durations, bins=bins, color='#3b82f6', edgecolor='white')

    mean_val = statistics.fmean(durations)
    median_val = float(statistics.median(durations))
    ax.axvline(mean_val, color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}s')
    ax.axvline(median_val, color='#10b981', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}s')
    ax.legend(loc='upper right', framealpha=0.85)

    ax.set_xlabel('Segment duration (s)')
    ax.set_ylabel('Count')
    ax.set_title(
        f'Segment duration (n={len(durations)}, mean={mean_val:.2f}s, '
        f'median={median_val:.2f}s)'
    )
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    _save(plt, fig, dest)


def _write_cutoff_bars(plt, turns: list[dict], dest: Path, *, bin_width: float = 0.25) -> None:
    durations = _turn_durations(turns)
    if not durations:
        fig, (ax_count, ax_dur) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax in (ax_count, ax_dur):
            ax.axis('off')
            ax.text(0.5, 0.5, 'No turns to plot', ha='center', va='center')
        ax_count.set_title('Remaining if dropping segments shorter than T')
        _save(plt, fig, dest)
        return

    step = bin_width if (math.isfinite(bin_width) and bin_width > 0) else 0.25
    n = len(durations)
    total_s = sum(durations)
    num_steps = math.floor(max(durations) / step)
    thresholds = [round(i * step, 4) for i in range(num_steps + 1)]
    remain_n = [sum(duration >= threshold for duration in durations) for threshold in thresholds]
    remain_s = [sum(duration for duration in durations if duration >= threshold) for threshold in thresholds]

    fig_w = min(24.0, max(12.0, len(thresholds) * 0.3))
    fig, (ax_count, ax_dur) = plt.subplots(2, 1, figsize=(fig_w, 7), sharex=True)

    bar_w = 0.8 * step
    rot = 90 if len(thresholds) > 15 else 0
    fsize = 6 if len(thresholds) > 30 else (7 if len(thresholds) > 15 else 8)
    headroom = 1.35 if rot == 90 else 1.22
    step_factor = math.ceil(len(thresholds) / 50) if len(thresholds) > 60 else 1

    count_labels = [f'{count} ({_pct(count, n)})' if (i % step_factor == 0) else '' for i, count in enumerate(remain_n)]
    count_bars = ax_count.bar(thresholds, remain_n, color='#3b82f6', width=bar_w)
    ax_count.bar_label(count_bars, labels=count_labels, fontsize=fsize, padding=3, rotation=rot)
    ax_count.set_ylabel('Remaining segments')
    ax_count.set_title('Remaining if dropping segments shorter than T')
    ax_count.set_ylim(0, max(remain_n) * headroom if remain_n else 1)
    ax_count.grid(True, axis='y', linestyle='--', alpha=0.4)

    dur_labels = [f'{seconds:.1f}s ({_pct(seconds, total_s)})' if (i % step_factor == 0) else '' for i, seconds in enumerate(remain_s)]
    duration_bars = ax_dur.bar(thresholds, remain_s, color='#22c55e', width=bar_w)
    ax_dur.bar_label(
        duration_bars,
        labels=dur_labels,
        fontsize=fsize,
        padding=3,
        rotation=rot,
    )
    ax_dur.set_ylabel('Remaining audio (s)')
    ax_dur.set_xlabel('Keep segments ≥ T seconds')
    ax_dur.set_ylim(0, max(remain_s) * headroom if remain_s else 1)

    tick_indices = list(range(0, len(thresholds), step_factor))
    ax_dur.set_xticks([thresholds[i] for i in tick_indices])
    if len(thresholds) > 15:
        ax_dur.tick_params(axis='x', rotation=45 if len(tick_indices) <= 35 else 90, labelsize=fsize)
    ax_dur.grid(True, axis='y', linestyle='--', alpha=0.4)
    _save(plt, fig, dest)


def _turn_durations(turns: list[dict]) -> list[float]:
    durations = []
    for turn in turns:
        duration = float(turn['end_s']) - float(turn['start_s'])
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    return durations


def _pct(part: float, whole: float) -> str:
    if whole <= 0:
        return '0%'
    value = 100.0 * part / whole
    if abs(value - round(value)) < 0.05:
        return f'{round(value)}%'
    return f'{value:.1f}%'


def _save(plt, fig, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    plt.close(fig)


def _audio_python() -> Path:
    env = os.environ.get('AUDIO_PYTHON')
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    for candidate in ('.venvs/audio', '.venv-audio', '.venvs/main', '.venv'):
        path = ROOT / candidate / 'bin/python'
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        'Missing audio environment. Run bash "envs/setup_worker_envs.sh" audio or set AUDIO_PYTHON.'
    )


def _write_plots_via_audio_python(manifest_path: Path, output_file: Path, *, bin_width: float = 0.25) -> list[Path]:
    script = Path(__file__).resolve().parents[1] / 'evaluate' / 'plot_diarization.py'
    cmd = [str(_audio_python()), str(script), '--input-manifest', str(Path(manifest_path).resolve()),
           '--output-file', str(Path(output_file).resolve()), '--bin-width', str(bin_width), '--overwrite']
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return list(sibling_plot_paths(output_file))
