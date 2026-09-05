"""Notebook visualization and interactive inspection for DiarizationResult."""

from __future__ import annotations

from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data_paths import DATA_DIR
from src.utils.AudioClass import Audio
from src.utils.AudioCutter import AudioCutter

if TYPE_CHECKING:
    from src.diarization.schemas import DiarizationResult, SpeakerTurn


def plot_diarization_result(
    result: DiarizationResult,
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Any:
    """Plot a timeline / Gantt chart of speaker turns using matplotlib.

    Args:
        result: DiarizationResult to visualize.
        title: Title of plot. Defaults to audio ID and turn statistics.
        figsize: Figure size. Defaults to scaled height based on speaker count.
        show: If True, calls plt.show(), closes figure, and returns None.

    Returns:
        None if show is True, otherwise the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    speakers = sorted({turn.speaker_id for turn in result.turns})
    if not speakers:
        speakers = [s.speaker_id for s in result.speakers] or ["SPEAKER_00"]

    spk_to_y = {spk: idx for idx, spk in enumerate(reversed(speakers))}
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(len(speakers))]

    height = max(2.5, 1.0 + 0.6 * len(speakers))
    fig, ax = plt.subplots(figsize=figsize or (12, height))

    duration_s = (
        result.source_audio.duration_s
        if result.source_audio and result.source_audio.duration_s
        else max((t.end_s for t in result.turns), default=10.0)
    )

    for turn in result.turns:
        y = spk_to_y[turn.speaker_id]
        color = colors[speakers.index(turn.speaker_id) % len(colors)]
        edgecolor = "#ef4444" if turn.overlaps_other_speaker else "none"
        linewidth = 1.5 if turn.overlaps_other_speaker else 0
        ax.barh(
            y=y,
            width=turn.duration_s,
            left=turn.start_s,
            height=0.55,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.85,
        )

    has_overlap = any(turn.overlaps_other_speaker for turn in result.turns)
    if has_overlap:
        import matplotlib.patches as mpatches

        legend_patch = mpatches.Patch(
            facecolor="#fca5a5", edgecolor="#ef4444", linewidth=1.5, label="Overlap"
        )
        ax.legend(handles=[legend_patch], loc="upper right", framealpha=0.85)

    if not result.turns:
        ax.text(
            0.5,
            0.5,
            "No speaker turns detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#64748b",
            fontsize=11,
        )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(list(reversed(speakers)))
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, max(duration_s, 1.0))
    ax.set_ylim(-0.5, len(speakers) - 0.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plot_title = (
        title
        or f"Diarization: {result.audio_id} ({len(result.turns)} turns, {len(speakers)} speakers)"
    )
    ax.set_title(plot_title)
    fig.tight_layout()

    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_turn_waveform(
    result: DiarizationResult,
    turn_or_index: int | SpeakerTurn,
    *,
    context_padding_s: float = 0.20,
    figsize: tuple[float, float] = (12, 2.4),
    show: bool = True,
) -> Any:
    """Plot the audio waveform context around a specific turn with boundaries.

    Args:
        result: DiarizationResult containing source_audio and turns.
        turn_or_index: Turn index or SpeakerTurn instance.
        context_padding_s: Extra seconds displayed before and after turn boundaries.
        figsize: Matplotlib figure dimensions.
        show: If True, calls plt.show(), closes figure, and returns None.

    Returns:
        None if show is True, otherwise the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    from src.diarization.schemas import SpeakerTurn as SpeakerTurnCls

    if isinstance(turn_or_index, int):
        if not result.turns:
            raise IndexError("Result has no turns to plot.")
        if not 0 <= turn_or_index < len(result.turns):
            raise IndexError(
                f"turn index {turn_or_index} out of range (0..{len(result.turns) - 1})"
            )
        turn = result.turns[turn_or_index]
    elif isinstance(turn_or_index, SpeakerTurnCls):
        turn = turn_or_index
    else:
        raise TypeError(
            f"turn_or_index must be int or SpeakerTurn, got {type(turn_or_index).__name__}"
        )

    if result.source_audio is None:
        raise ValueError("The diarization result has no source_audio.")
    audio_path = Path(result.source_audio.path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_s = turn.start_s
    end_s = turn.end_s
    raw_start_s = float(getattr(turn, "_raw_start_s", start_s))
    raw_end_s = float(getattr(turn, "_raw_end_s", end_s))

    with sf.SoundFile(str(audio_path)) as source:
        sample_rate = int(source.samplerate)
        file_duration = source.frames / float(sample_rate) if sample_rate > 0 else (end_s + 1.0)
        context_start = max(0.0, min(start_s, raw_start_s) - context_padding_s)
        context_end = min(file_duration, max(end_s, raw_end_s) + context_padding_s)
        if context_end <= context_start:
            context_end = min(file_duration, context_start + max(0.1, context_padding_s * 2))

        seek_frame = max(0, min(int(context_start * sample_rate), max(0, source.frames - 1)))
        source.seek(seek_frame)
        frames_to_read = max(1, min(int((context_end - context_start) * sample_rate), source.frames - seek_frame))
        waveform = source.read(frames_to_read, dtype="float32", always_2d=True)

    mono = waveform.mean(axis=1) if len(waveform) else np.zeros(1, dtype=np.float32)
    stride = max(1, len(mono) // 5000)
    sub_mono = mono[::stride]
    times = context_start + np.arange(0, len(sub_mono)) * stride / sample_rate

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(times[: len(sub_mono)], sub_mono[: len(times)], color="#64748b", linewidth=0.7)
    if raw_start_s != start_s or raw_end_s != end_s:
        axis.axvspan(raw_start_s, raw_end_s, color="#f59e0b", alpha=0.18, label="Blunt/raw")
    axis.axvspan(start_s, end_s, color="#10b981", alpha=0.20, label=f"Refined ({turn.speaker_id})")
    axis.set(xlabel="Time (s)", ylabel="Amplitude", xlim=(context_start, max(context_end, context_start + 0.05)))
    axis.legend(loc="upper right")
    figure.tight_layout()

    if show:
        plt.show()
        plt.close(figure)
        return None
    return figure


class DiarizationResultNotebookViewer:
    """Interactive Jupyter viewer for a file-backed ``DiarizationResult``.

    Turn clips are created lazily under ``.data/notebook/diarization_viewer``.
    Provides interactive speaker and overlap filtering, search, boundary
    inspection, waveform context around turns, prev/next turn controls,
    and lazy per-turn audio playback.
    """

    def __init__(
        self,
        result: DiarizationResult,
        output_dir: str | Path | None = None,
    ) -> None:
        import ipywidgets as widgets

        self.result = result
        self.audio = result.source_audio
        self.has_audio = (
            self.audio is not None and Path(self.audio.path).is_file()
        )
        base_dir = (
            Path(output_dir)
            if output_dir is not None
            else DATA_DIR / "notebook" / "diarization_viewer"
        )
        self.output_dir = base_dir / result.result_id
        self.cutter = AudioCutter(output_dir=self.output_dir) if self.has_audio else None
        self._filtered_indices: list[int] = []

        speakers = ["All speakers", *sorted({t.speaker_id for t in result.turns})]
        self.speaker = widgets.Dropdown(
            options=speakers, description="Speaker:", layout=widgets.Layout(width="220px")
        )
        self.overlap = widgets.Dropdown(
            options=["All turns", "Clean only", "Overlapping only"],
            description="Overlap:",
            layout=widgets.Layout(width="220px"),
        )
        self.search = widgets.Text(
            description="Search:", placeholder="transcript or policy", layout=widgets.Layout(flex="1 1 auto")
        )
        self.prev_btn = widgets.Button(
            description="◀ Prev", tooltip="Previous turn", layout=widgets.Layout(width="70px")
        )
        self.next_btn = widgets.Button(
            description="Next ▶", tooltip="Next turn", layout=widgets.Layout(width="70px")
        )
        self.turn = widgets.Dropdown(
            options=[], description="Turn:", layout=widgets.Layout(flex="1 1 auto")
        )
        self.summary = widgets.HTML()
        self.detail = widgets.Output()

        self.speaker.observe(self._filters_changed, names="value")
        self.overlap.observe(self._filters_changed, names="value")
        self.search.observe(self._filters_changed, names="value")
        self.turn.observe(self._turn_changed, names="value")
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)

        self._refresh_filters()

    @staticmethod
    def _transcript(turn: Any) -> str:
        return str(getattr(turn, "_transcript", getattr(turn, "transcript", "")) or "")

    @staticmethod
    def _policy(turn: Any) -> str:
        return str(getattr(turn, "_boundary_policy", "standard"))

    def _filters_changed(self, _change: Any = None) -> None:
        self._refresh_filters()

    def _on_prev(self, _b: Any) -> None:
        if not self._filtered_indices or self.turn.value is None:
            return
        try:
            curr_pos = self._filtered_indices.index(int(self.turn.value))
            if curr_pos > 0:
                self.turn.value = self._filtered_indices[curr_pos - 1]
        except ValueError:
            if self._filtered_indices:
                self.turn.value = self._filtered_indices[0]

    def _on_next(self, _b: Any) -> None:
        if not self._filtered_indices or self.turn.value is None:
            return
        try:
            curr_pos = self._filtered_indices.index(int(self.turn.value))
            if curr_pos < len(self._filtered_indices) - 1:
                self.turn.value = self._filtered_indices[curr_pos + 1]
        except ValueError:
            if self._filtered_indices:
                self.turn.value = self._filtered_indices[0]

    def _update_nav_buttons(self, current_index: int | None) -> None:
        if current_index is None or not self._filtered_indices:
            self.prev_btn.disabled = True
            self.next_btn.disabled = True
            return
        try:
            pos = self._filtered_indices.index(current_index)
            self.prev_btn.disabled = pos <= 0
            self.next_btn.disabled = pos >= len(self._filtered_indices) - 1
        except ValueError:
            self.prev_btn.disabled = True
            self.next_btn.disabled = True

    def _refresh_filters(self) -> None:
        from IPython.display import HTML, clear_output, display

        wanted_speaker = self.speaker.value
        wanted_overlap = self.overlap.value
        query = self.search.value.strip().lower()
        matches = []
        for index, turn in enumerate(self.result.turns):
            if wanted_speaker != "All speakers" and turn.speaker_id != wanted_speaker:
                continue
            if wanted_overlap == "Clean only" and turn.overlaps_other_speaker:
                continue
            if wanted_overlap == "Overlapping only" and not turn.overlaps_other_speaker:
                continue
            haystack = f"{turn.speaker_id} {self._transcript(turn)} {self._policy(turn)}".lower()
            if query and query not in haystack:
                continue
            matches.append(index)

        self._filtered_indices = matches
        total_matched = len(matches)
        num_overlaps = sum(
            1 for idx in matches if self.result.turns[idx].overlaps_other_speaker
        )
        clean_count = total_matched - num_overlaps
        duration = sum(self.result.turns[index].duration_s for index in matches)
        average = duration / total_matched if total_matched else 0.0

        overlap_badge = (
            f" <span style='color: #ef4444;'>({num_overlaps} overlapping)</span>"
            if num_overlaps > 0
            else ""
        )
        self.summary.value = (
            f"<div style='margin: 4px 0 8px 0; font-size: 13px;'>"
            f"<b>{total_matched}</b> turns ({clean_count} clean{overlap_badge}) &nbsp;•&nbsp; "
            f"<b>{duration:.1f}s</b> speech &nbsp;•&nbsp; average <b>{average:.1f}s</b>"
            f"</div>"
        )

        options = [
            (
                f"#{index + 1} · {self.result.turns[index].speaker_id} · "
                f"{self.result.turns[index].start_s:.2f}–{self.result.turns[index].end_s:.2f}s"
                f"{' [OVERLAP]' if self.result.turns[index].overlaps_other_speaker else ''}",
                index,
            )
            for index in matches
        ]

        if not options:
            self.turn.options = []
            self.turn.value = None
            self._update_nav_buttons(None)
            with self.detail:
                clear_output(wait=True)
                display(HTML("<i>No turns match the active filters.</i>"))
            return

        old_val = self.turn.value
        self.turn.options = options
        if old_val in matches:
            self.turn.value = old_val
            self._render_turn(int(old_val))
        else:
            self.turn.value = matches[0]

    def _turn_changed(self, change: dict[str, Any]) -> None:
        if change.get("new") is not None:
            self._render_turn(int(change["new"]))

    def _clip(self, index: int, start_s: float, end_s: float, kind: str) -> Audio:
        if self.cutter is None or self.audio is None:
            raise RuntimeError("Cannot cut audio: source_audio is missing or not accessible.")
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        path = self.output_dir / f"turn_{index:06d}_{start_ms}_{end_ms}_{kind}.wav"
        if path.is_file():
            return Audio.from_file(path)
        return self.cutter.cut(self.audio, start_s, end_s, output_path=path)

    def _render_turn(self, index: int) -> None:
        import matplotlib.pyplot as plt
        from IPython.display import Audio as IPythonAudio, HTML, clear_output, display

        self._update_nav_buttons(index)
        turn = self.result.turns[index]
        raw_start = float(getattr(turn, "_raw_start_s", turn.start_s))
        raw_end = float(getattr(turn, "_raw_end_s", turn.end_s))
        delta_end = float(getattr(turn, "_delta_end_ms", 0.0))
        transcript = self._transcript(turn) or "—"
        confidence_str = (
            f"{turn.confidence * 100:.1f}%"
            if turn.confidence is not None
            else "—"
        )
        overlap_str = (
            "<span style='color: #ef4444; font-weight: bold;'>⚠️ Yes (Overlapping)</span>"
            if turn.overlaps_other_speaker
            else "<span style='color: #10b981;'>✓ Clean</span>"
        )
        has_refinement = (raw_start != turn.start_s or raw_end != turn.end_s)

        metadata = (
            "<table style='width:100%; text-align:left; border-collapse: collapse; margin-bottom: 8px; font-size: 13px;'>"
            f"<tr><th style='padding: 3px 6px;'>Speaker</th><td style='padding: 3px 6px;'><b>{html_escape(turn.speaker_id)}</b></td>"
            f"<th style='padding: 3px 6px;'>Duration</th><td style='padding: 3px 6px;'>{turn.duration_s:.2f}s</td></tr>"
            f"<tr><th style='padding: 3px 6px;'>Time</th><td style='padding: 3px 6px;'>{turn.start_s:.3f}–{turn.end_s:.3f}s</td>"
            f"<th style='padding: 3px 6px;'>Confidence</th><td style='padding: 3px 6px;'>{confidence_str}</td></tr>"
            f"<tr><th style='padding: 3px 6px;'>Overlap</th><td style='padding: 3px 6px;'>{overlap_str}</td>"
            f"<th style='padding: 3px 6px;'>Policy</th><td style='padding: 3px 6px;'>{html_escape(self._policy(turn))}</td></tr>"
        )
        if has_refinement:
            metadata += (
                f"<tr><th style='padding: 3px 6px;'>Raw/blunt</th><td style='padding: 3px 6px;'>{raw_start:.3f}–{raw_end:.3f}s</td>"
                f"<th style='padding: 3px 6px;'>End delta</th><td style='padding: 3px 6px;'>{delta_end:+.0f}ms</td></tr>"
            )
        metadata += f"<tr><th style='padding: 3px 6px;'>Transcript</th><td colspan='3' style='padding: 3px 6px;'>{html_escape(transcript)}</td></tr></table>"

        with self.detail:
            clear_output(wait=True)
            display(HTML(metadata))

            if not self.has_audio:
                display(
                    HTML(
                        "<div style='color: #f59e0b; padding: 6px; background: #fffbeb; "
                        "border: 1px solid #fde68a; border-radius: 4px; margin-top: 4px; font-size: 12px;'>"
                        "Audio file not available. Waveform and audio playback are disabled.</div>"
                    )
                )
                return

            try:
                figure = plot_turn_waveform(self.result, turn, show=False)
                if figure is not None:
                    display(figure)
                    plt.close(figure)
            except Exception as exc:
                display(
                    HTML(
                        f"<div style='color: #ef4444; font-size: 12px;'>"
                        f"Could not plot waveform: {html_escape(str(exc))}</div>"
                    )
                )

            try:
                refined = self._clip(index, turn.start_s, turn.end_s, "refined")
                display(HTML("<div style='margin-top: 4px;'><b>Refined boundary</b></div>"))
                display(IPythonAudio(filename=str(refined.path)))

                if has_refinement:
                    raw = self._clip(index, raw_start, raw_end, "raw")
                    display(HTML("<div style='margin-top: 4px;'><b>Raw/blunt boundary</b></div>"))
                    display(IPythonAudio(filename=str(raw.path)))
            except Exception as exc:
                display(
                    HTML(
                        f"<div style='color: #ef4444; font-size: 12px;'>"
                        f"Could not generate audio preview: {html_escape(str(exc))}</div>"
                    )
                )

    def display(self) -> None:
        """Render the interactive viewer once and return ``None``."""
        import ipywidgets as widgets
        from IPython.display import display

        filter_row = widgets.HBox(
            [self.speaker, self.overlap, self.search],
            layout=widgets.Layout(width="100%", margin="0 0 6px 0"),
        )
        nav_row = widgets.HBox(
            [self.prev_btn, self.next_btn, self.turn],
            layout=widgets.Layout(width="100%", align_items="center", margin="0 0 6px 0"),
        )
        container = widgets.VBox([filter_row, nav_row, self.summary, self.detail])
        display(container)
