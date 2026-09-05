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
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(speakers))))

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

    total_dur = result.source_audio.duration_s or (end_s + 1.0)
    context_start = max(0.0, min(start_s, raw_start_s) - context_padding_s)
    context_end = min(total_dur, max(end_s, raw_end_s) + context_padding_s)

    with sf.SoundFile(str(audio_path)) as source:
        sample_rate = int(source.samplerate)
        source.seek(int(context_start * sample_rate))
        frames_to_read = max(1, int((context_end - context_start) * sample_rate))
        waveform = source.read(frames_to_read, dtype="float32", always_2d=True)

    mono = waveform.mean(axis=1) if len(waveform) else np.zeros(1, dtype=np.float32)
    stride = max(1, len(mono) // 5000)
    times = context_start + np.arange(0, len(mono), stride) / sample_rate

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(times, mono[::stride], color="#64748b", linewidth=0.7)
    if raw_start_s != start_s or raw_end_s != end_s:
        axis.axvspan(raw_start_s, raw_end_s, color="#f59e0b", alpha=0.18, label="Blunt/raw")
    axis.axvspan(start_s, end_s, color="#10b981", alpha=0.20, label=f"Refined ({turn.speaker_id})")
    axis.set(xlabel="Time (s)", ylabel="Amplitude", xlim=(context_start, context_end))
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
    The controls mirror SonicStudio's result viewer: filter by speaker or
    text, inspect boundary metadata, compare blunt/refined audio, and view
    the waveform around the selected turn.
    """

    def __init__(
        self,
        result: DiarizationResult,
        output_dir: str | Path | None = None,
    ) -> None:
        import ipywidgets as widgets
        from IPython.display import HTML

        if result.source_audio is None:
            raise ValueError("The diarization result has no source_audio.")
        if not Path(result.source_audio.path).is_file():
            raise FileNotFoundError(result.source_audio.path)

        self.result = result
        self.audio = result.source_audio
        base_dir = (
            Path(output_dir)
            if output_dir is not None
            else DATA_DIR / "notebook" / "diarization_viewer"
        )
        self.output_dir = base_dir / result.result_id
        self.cutter = AudioCutter(output_dir=self.output_dir)
        self._filtered_indices: list[int] = []

        speakers = ["All speakers", *sorted({t.speaker_id for t in result.turns})]
        self.speaker = widgets.Dropdown(options=speakers, description="Speaker:")
        self.search = widgets.Text(
            description="Search:", placeholder="transcript or policy"
        )
        self.turn = widgets.Dropdown(
            options=[], description="Turn:", layout=widgets.Layout(width="100%")
        )
        self.summary = widgets.HTML()
        self.detail = widgets.Output()

        self.speaker.observe(self._filters_changed, names="value")
        self.search.observe(self._filters_changed, names="value")
        self.turn.observe(self._turn_changed, names="value")
        self._refresh_filters()

    @staticmethod
    def _transcript(turn: Any) -> str:
        return str(getattr(turn, "_transcript", getattr(turn, "transcript", "")) or "")

    @staticmethod
    def _policy(turn: Any) -> str:
        return str(getattr(turn, "_boundary_policy", "standard"))

    def _filters_changed(self, _change: Any = None) -> None:
        self._refresh_filters()

    def _refresh_filters(self) -> None:
        from IPython.display import HTML, clear_output

        wanted_speaker = self.speaker.value
        query = self.search.value.strip().lower()
        matches = []
        for index, turn in enumerate(self.result.turns):
            if wanted_speaker != "All speakers" and turn.speaker_id != wanted_speaker:
                continue
            haystack = f"{turn.speaker_id} {self._transcript(turn)} {self._policy(turn)}".lower()
            if query and query not in haystack:
                continue
            matches.append(index)
        self._filtered_indices = matches
        duration = sum(self.result.turns[index].duration_s for index in matches)
        average = duration / len(matches) if matches else 0.0
        self.summary.value = (
            f"<b>{len(matches)}</b> clean turns &nbsp;•&nbsp; "
            f"<b>{duration:.1f}s</b> speech &nbsp;•&nbsp; average <b>{average:.1f}s</b>"
        )
        options = [
            (
                f"#{index + 1} · {self.result.turns[index].speaker_id} · "
                f"{self.result.turns[index].start_s:.2f}–{self.result.turns[index].end_s:.2f}s",
                index,
            )
            for index in matches
        ]
        self.turn.options = options
        if not options:
            with self.detail:
                clear_output(wait=True)
                from IPython.display import display

                display(HTML("<i>No turns match the active filters.</i>"))

    def _turn_changed(self, change: dict[str, Any]) -> None:
        if change.get("new") is not None:
            self._render_turn(int(change["new"]))

    def _clip(self, index: int, start_s: float, end_s: float, kind: str) -> Audio:
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        path = self.output_dir / f"turn_{index:06d}_{start_ms}_{end_ms}_{kind}.wav"
        if path.is_file():
            return Audio.from_file(path)
        return self.cutter.cut(self.audio, start_s, end_s, output_path=path)

    def _render_turn(self, index: int) -> None:
        import matplotlib.pyplot as plt
        from IPython.display import Audio as IPythonAudio, HTML, clear_output, display

        turn = self.result.turns[index]
        raw_start = float(getattr(turn, "_raw_start_s", turn.start_s))
        raw_end = float(getattr(turn, "_raw_end_s", turn.end_s))
        delta_end = float(getattr(turn, "_delta_end_ms", 0.0))
        transcript = self._transcript(turn) or "—"
        metadata = (
            "<table style='width:100%;text-align:left'>"
            f"<tr><th>Speaker</th><td>{html_escape(turn.speaker_id)}</td><th>Duration</th><td>{turn.duration_s:.2f}s</td></tr>"
            f"<tr><th>Refined</th><td>{turn.start_s:.3f}–{turn.end_s:.3f}s</td><th>Raw/blunt</th><td>{raw_start:.3f}–{raw_end:.3f}s</td></tr>"
            f"<tr><th>Policy</th><td>{html_escape(self._policy(turn))}</td><th>End delta</th><td>{delta_end:+.0f}ms</td></tr>"
            f"<tr><th>Transcript</th><td colspan='3'>{html_escape(transcript)}</td></tr></table>"
        )
        refined = self._clip(index, turn.start_s, turn.end_s, "refined")
        raw = self._clip(index, raw_start, raw_end, "raw")
        with self.detail:
            clear_output(wait=True)
            display(HTML(metadata))
            figure = plot_turn_waveform(self.result, turn, show=False)
            if figure is not None:
                display(figure)
                plt.close(figure)
            display(HTML("<b>Refined boundary</b>"))
            display(IPythonAudio(filename=str(refined.path)))
            if raw_start != turn.start_s or raw_end != turn.end_s:
                display(HTML("<b>Raw/blunt boundary</b>"))
                display(IPythonAudio(filename=str(raw.path)))

    def display(self) -> None:
        """Render the interactive viewer once and return ``None``."""
        import ipywidgets as widgets
        from IPython.display import display

        controls = widgets.HBox([self.speaker, self.search])
        display(widgets.VBox([controls, self.summary, self.turn, self.detail]))
        if self.turn.value is not None:
            self._render_turn(int(self.turn.value))
