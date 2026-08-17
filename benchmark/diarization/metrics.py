"""Diarization metrics helpers built on pyannote.metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from benchmark.diarization import DEFAULT_COLLAR_S
from benchmark.diarization.dataset_viyt import ReferenceTurn, ViYTSample
from src.diarization.schemas import DiarizationResult, SpeakerTurn


@dataclass(frozen=True)
class FileMetrics:
    """Per-file diarization scores."""

    audio_id: str
    der: float
    jer: float
    false_alarm: float
    missed_detection: float
    confusion: float
    ref_speakers: int
    hyp_speakers: int
    speaker_count_abs_error: int
    duration_s: float


@dataclass(frozen=True)
class SystemSummary:
    """Aggregate scores for one system on one run."""

    system: str
    num_files: int
    mean_der: float
    median_der: float
    mean_jer: float
    mean_speaker_count_abs_error: float
    total_duration_s: float
    label: str = ""
    mean_false_alarm: float = float("nan")
    mean_missed_detection: float = float("nan")
    mean_confusion: float = float("nan")


def turns_to_annotation(
    turns: Iterable[ReferenceTurn | SpeakerTurn | Mapping[str, Any]],
) -> Annotation:
    """Convert turn records into a pyannote ``Annotation``."""
    annotation = Annotation()
    for turn in turns:
        if isinstance(turn, Mapping):
            speaker_id = str(turn["speaker_id"])
            start_s = float(turn["start_s"])
            end_s = float(turn["end_s"])
        else:
            speaker_id = str(turn.speaker_id)
            start_s = float(turn.start_s)
            end_s = float(turn.end_s)
        if end_s <= start_s:
            continue
        annotation[Segment(start_s, end_s)] = speaker_id
    return annotation


def result_to_annotation(result: DiarizationResult) -> Annotation:
    """Convert a pipeline ``DiarizationResult`` into an ``Annotation``."""
    return turns_to_annotation(result.turns)


def reference_annotation(sample: ViYTSample) -> Annotation:
    """Build the ground-truth annotation for a ViYT sample."""
    return turns_to_annotation(sample.turns)


def score_file(
    sample: ViYTSample,
    hypothesis: Annotation | DiarizationResult,
    *,
    collar_s: float = DEFAULT_COLLAR_S,
    skip_overlap: bool = False,
) -> FileMetrics:
    """Score one hypothesis against the ViYT reference.

    Args:
        sample: Cached reference clip.
        hypothesis: Predicted annotation or ``DiarizationResult``.
        collar_s: Forgiveness collar in seconds (default 0.25).
        skip_overlap: When True, ignore overlapping reference regions.

    Returns:
        Per-file DER / JER / speaker-count metrics.
    """
    reference = reference_annotation(sample)
    if isinstance(hypothesis, DiarizationResult):
        hyp = result_to_annotation(hypothesis)
        hyp_speakers = len(hypothesis.speakers)
    else:
        hyp = hypothesis
        hyp_speakers = len(hyp.labels())

    der_metric = DiarizationErrorRate(collar=collar_s, skip_overlap=skip_overlap)
    jer_metric = JaccardErrorRate(collar=collar_s, skip_overlap=skip_overlap)

    der_components = der_metric(reference, hyp, detailed=True)
    der = float(der_components["diarization error rate"])
    jer = float(jer_metric(reference, hyp))
    total = float(der_components.get("total", 0.0))
    denom = total if total > 0 else 1.0

    return FileMetrics(
        audio_id=sample.audio_id,
        der=der,
        jer=jer,
        false_alarm=float(der_components.get("false alarm", 0.0)) / denom,
        missed_detection=float(der_components.get("missed detection", 0.0)) / denom,
        confusion=float(der_components.get("confusion", 0.0)) / denom,
        ref_speakers=sample.num_speakers,
        hyp_speakers=hyp_speakers,
        speaker_count_abs_error=abs(hyp_speakers - sample.num_speakers),
        duration_s=sample.duration_s,
    )


def summarize_system(
    system: str,
    file_metrics: Sequence[FileMetrics],
    *,
    label: str | None = None,
) -> SystemSummary:
    """Aggregate per-file metrics into a system-level summary."""
    display = label or system
    if not file_metrics:
        return SystemSummary(
            system=system,
            label=display,
            num_files=0,
            mean_der=float("nan"),
            median_der=float("nan"),
            mean_jer=float("nan"),
            mean_speaker_count_abs_error=float("nan"),
            total_duration_s=0.0,
        )

    ders = sorted(m.der for m in file_metrics)
    n = len(ders)
    mid = n // 2
    median = ders[mid] if n % 2 == 1 else 0.5 * (ders[mid - 1] + ders[mid])
    return SystemSummary(
        system=system,
        label=display,
        num_files=n,
        mean_der=sum(ders) / n,
        median_der=median,
        mean_jer=sum(m.jer for m in file_metrics) / n,
        mean_speaker_count_abs_error=(
            sum(m.speaker_count_abs_error for m in file_metrics) / n
        ),
        total_duration_s=sum(m.duration_s for m in file_metrics),
        mean_false_alarm=sum(m.false_alarm for m in file_metrics) / n,
        mean_missed_detection=sum(m.missed_detection for m in file_metrics) / n,
        mean_confusion=sum(m.confusion for m in file_metrics) / n,
    )


def metrics_to_dict(metrics: FileMetrics | SystemSummary) -> dict[str, Any]:
    """Serialize metric dataclasses to plain JSON-friendly dicts."""
    return asdict(metrics)
