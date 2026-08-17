"""Plot and export ViYT-Diar benchmark comparison figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from benchmark.diarization import FIGURES_DIR
from benchmark.diarization.metrics import FileMetrics, SystemSummary

_PALETTE = (
    "#2f6fed",
    "#c45c26",
    "#2a9d8f",
    "#7b2cbf",
    "#e9c46a",
    "#264653",
    "#e76f51",
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display_labels(summaries: Sequence[SystemSummary]) -> list[str]:
    return [s.label or s.system for s in summaries]


def export_comparison_figures(
    *,
    summaries: Sequence[SystemSummary],
    per_system_files: Mapping[str, Sequence[FileMetrics]],
    output_dir: Path | None = None,
    run_id: str,
) -> list[Path]:
    """Write comparison plots under ``benchmark/figures``.

    Args:
        summaries: One summary per evaluated system (plot order).
        per_system_files: Mapping of system key → per-file metrics.
        output_dir: Figure directory (defaults to ``benchmark/figures``).
        run_id: Filename prefix shared by this run.

    Returns:
        Paths of written image files.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not summaries:
        return []

    out = _ensure_dir(Path(output_dir) if output_dir is not None else FIGURES_DIR)
    written: list[Path] = []
    labels = _display_labels(summaries)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(summaries))]
    multi = len(summaries) > 1

    # --- Mean DER bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ders = [100.0 * s.mean_der for s in summaries]
    bars = ax.bar(labels, ders, color=colors, edgecolor="none")
    ax.set_ylabel("Mean DER (%)")
    title = "ViYT-Diar — Mean Diarization Error Rate"
    if multi:
        title += " (model comparison)"
    ax.set_title(title)
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

    # --- Mean JER bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5.0))
    jers = [100.0 * s.mean_jer for s in summaries]
    bars = ax.bar(labels, jers, color=colors, edgecolor="none")
    ax.set_ylabel("Mean JER (%)")
    ax.set_title("ViYT-Diar — Mean Jaccard Error Rate")
    ax.set_ylim(0, max(jers + [1.0]) * 1.25)
    for bar, value in zip(bars, jers, strict=True):
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
    path = out / f"{run_id}_mean_jer.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    # --- Per-file DER box plot ---
    fig, ax = plt.subplots(figsize=(10, 5.0))
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
    fig, ax = plt.subplots(figsize=(10, 5.0))
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

    # --- DER components (FA / Miss / Confusion) ---
    if all(
        not (
            math.isnan(s.mean_false_alarm)
            or math.isnan(s.mean_missed_detection)
            or math.isnan(s.mean_confusion)
        )
        for s in summaries
    ):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        x = np.arange(len(summaries))
        width = 0.25
        fa = [100.0 * s.mean_false_alarm for s in summaries]
        miss = [100.0 * s.mean_missed_detection for s in summaries]
        conf = [100.0 * s.mean_confusion for s in summaries]
        ax.bar(x - width, fa, width, label="False alarm", color="#e9c46a")
        ax.bar(x, miss, width, label="Missed detection", color="#e76f51")
        ax.bar(x + width, conf, width, label="Confusion", color="#264653")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("Mean component rate (%)")
        ax.set_title("ViYT-Diar — DER components")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out / f"{run_id}_der_components.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    # --- Per-file DER: single system bars, or multi-system overlay ---
    if not multi:
        system = summaries[0].system
        files = list(per_system_files.get(system, []))
        if files:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            xs = list(range(len(files)))
            ax.bar(xs, [100.0 * m.der for m in files], color=colors[0], edgecolor="none")
            ax.set_xlabel("File index")
            ax.set_ylabel("DER (%)")
            ax.set_title(f"ViYT-Diar — Per-file DER ({labels[0]})")
            fig.tight_layout()
            path = out / f"{run_id}_per_file_der.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            written.append(path)
    else:
        # Align on audio_id intersection so models stay comparable.
        id_lists = [
            [m.audio_id for m in per_system_files.get(s.system, [])] for s in summaries
        ]
        if id_lists and all(id_lists):
            common_ids = list(dict.fromkeys(id_lists[0]))
            for ids in id_lists[1:]:
                common_ids = [aid for aid in common_ids if aid in set(ids)]
            if common_ids:
                fig, ax = plt.subplots(figsize=(12, 5.2))
                xs = np.arange(len(common_ids))
                for summary, color, label in zip(summaries, colors, labels, strict=True):
                    by_id = {
                        m.audio_id: m.der
                        for m in per_system_files.get(summary.system, [])
                    }
                    ys = [100.0 * by_id[aid] for aid in common_ids]
                    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label, color=color)
                ax.set_xlabel("File index (shared clips)")
                ax.set_ylabel("DER (%)")
                ax.set_title("ViYT-Diar — Per-file DER by model")
                ax.legend(frameon=False, fontsize=8, loc="upper right")
                fig.tight_layout()
                path = out / f"{run_id}_per_file_der_compare.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                written.append(path)

    return written


def export_baseline_figures(
    *,
    summaries: Sequence[SystemSummary],
    per_system_files: Mapping[str, Sequence[FileMetrics]],
    output_dir: Path | None = None,
    run_id: str,
) -> list[Path]:
    """Backward-compatible alias for :func:`export_comparison_figures`."""
    return export_comparison_figures(
        summaries=summaries,
        per_system_files=per_system_files,
        output_dir=output_dir,
        run_id=run_id,
    )
