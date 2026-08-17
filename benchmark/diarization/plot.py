"""Plot and export ViYT-Diar benchmark comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from benchmark.diarization import FIGURES_DIR
from benchmark.diarization.metrics import FileMetrics, SystemSummary


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_baseline_figures(
    *,
    summaries: Sequence[SystemSummary],
    per_system_files: Mapping[str, Sequence[FileMetrics]],
    output_dir: Path | None = None,
    run_id: str,
) -> list[Path]:
    """Write comparison plots under ``benchmark/figures``.

    Args:
        summaries: One summary per evaluated system.
        per_system_files: Mapping of system key → per-file metrics.
        output_dir: Figure directory (defaults to ``benchmark/figures``).
        run_id: Filename prefix shared by this run.

    Returns:
        Paths of written image files.
    """
    import matplotlib.pyplot as plt

    out = _ensure_dir(Path(output_dir) if output_dir is not None else FIGURES_DIR)
    written: list[Path] = []

    # --- Mean DER bar chart ---
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [s.system for s in summaries]
    ders = [100.0 * s.mean_der for s in summaries]
    colors = ["#2f6fed" if i == 0 else "#5b6b7c" for i in range(len(summaries))]
    bars = ax.bar(labels, ders, color=colors, edgecolor="none")
    ax.set_ylabel("Mean DER (%)")
    ax.set_title("ViYT-Diar — Mean Diarization Error Rate")
    ax.set_ylim(0, max(ders + [1.0]) * 1.25)
    for bar, value in zip(bars, ders, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path = out / f"{run_id}_mean_der.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    # --- Per-file DER box plot ---
    fig, ax = plt.subplots(figsize=(9, 4.8))
    box_data = [
        [100.0 * m.der for m in per_system_files.get(s.system, [])] for s in summaries
    ]
    ax.boxplot(box_data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("DER (%)")
    ax.set_title("ViYT-Diar — Per-file DER distribution")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path = out / f"{run_id}_der_boxplot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    # --- Speaker-count absolute error ---
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sc_errs = [s.mean_speaker_count_abs_error for s in summaries]
    ax.bar(labels, sc_errs, color="#c45c26", edgecolor="none")
    ax.set_ylabel("Mean |#spk_hyp − #spk_ref|")
    ax.set_title("ViYT-Diar — Speaker count absolute error")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path = out / f"{run_id}_speaker_count_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    # --- DER components for single-system / baseline focus ---
    if len(summaries) == 1:
        system = summaries[0].system
        files = list(per_system_files.get(system, []))
        if files:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            xs = list(range(len(files)))
            ax.bar(xs, [100.0 * m.der for m in files], color="#2f6fed", edgecolor="none")
            ax.set_xlabel("File index")
            ax.set_ylabel("DER (%)")
            ax.set_title(f"ViYT-Diar — Per-file DER ({system})")
            fig.tight_layout()
            path = out / f"{run_id}_per_file_der.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            written.append(path)

    return written
