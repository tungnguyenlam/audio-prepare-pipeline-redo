"""Base interfaces and data structures for Speaker Diarization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class SpeakerTurn:
    """Represents a single speaker speech turn/segment."""
    start_s: float
    end_s: float
    speaker_id: str
    duration_s: float = 0.0
    is_overlap: bool = False
    confidence: float = 1.0
    turn_filename: Optional[str] = None
    clip_url: Optional[str] = None

    def __post_init__(self):
        if self.duration_s == 0.0:
            self.duration_s = round(max(0.0, self.end_s - self.start_s), 2)
        self.start_s = round(self.start_s, 2)
        self.end_s = round(self.end_s, 2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpeakerStats:
    """Summary statistics for a specific detected speaker."""
    speaker_id: str
    total_time_s: float
    percentage: float
    turn_count: int
    sample_audio_url: Optional[str] = None
    master_audio_url: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiarizationResult:
    """Overall result of speaker diarization execution."""
    run_id: str
    input_file: Path
    engine: str
    total_duration_s: float
    num_speakers: int
    speakers: List[SpeakerStats]
    turns: List[SpeakerTurn]
    output_dir: Path
    overlap_filtered: bool = True
    metadata_file: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "input_file": str(self.input_file),
            "input_filename": self.input_file.name,
            "engine": self.engine,
            "total_duration_s": round(self.total_duration_s, 2),
            "num_speakers": self.num_speakers,
            "speakers": [s.to_dict() for s in self.speakers],
            "turns": [t.to_dict() for t in self.turns],
            "overlap_filtered": self.overlap_filtered,
        }


def refine_and_merge_turns(
    turns: List[SpeakerTurn],
    max_merge_gap_s: float = 1.0,
    boundary_collar_s: float = 0.08,
    min_duration_s: float = 0.5,
) -> List[SpeakerTurn]:
    """Refines and merges speaker turns:
    1. Trims boundaries between adjacent turns of DIFFERENT speakers (boundary collar) to prevent voice bleed.
    2. Merges consecutive turns of the SAME speaker if the silence gap <= max_merge_gap_s (default 1.0s).
    3. Removes short residual turns < min_duration_s.
    """
    if not turns:
        return []

    # Sort turns chronologically
    sorted_turns = sorted(turns, key=lambda x: x.start_s)

    # Step 1: Boundary Collar Trimming between different speakers to prevent voice bleed
    cleaned_turns: List[SpeakerTurn] = []
    for i, t in enumerate(sorted_turns):
        t_start = t.start_s
        t_end = t.end_s

        # If previous turn belongs to a DIFFERENT speaker and is close/overlapping
        if i > 0:
            prev = sorted_turns[i - 1]
            if prev.speaker_id != t.speaker_id:
                if t_start < prev.end_s:
                    mid = (t_start + prev.end_s) / 2.0
                    t_start = mid + (boundary_collar_s / 2.0)
                elif (t_start - prev.end_s) < boundary_collar_s:
                    t_start = t_start + (boundary_collar_s / 2.0)

        # If next turn belongs to a DIFFERENT speaker and is close/overlapping
        if i < len(sorted_turns) - 1:
            nxt = sorted_turns[i + 1]
            if nxt.speaker_id != t.speaker_id:
                if t_end > nxt.start_s:
                    mid = (t_end + nxt.start_s) / 2.0
                    t_end = mid - (boundary_collar_s / 2.0)
                elif (nxt.start_s - t_end) < boundary_collar_s:
                    t_end = t_end - (boundary_collar_s / 2.0)

        dur = t_end - t_start
        if dur >= 0.25:
            cleaned_turns.append(SpeakerTurn(
                start_s=round(t_start, 2),
                end_s=round(t_end, 2),
                speaker_id=t.speaker_id,
                duration_s=round(dur, 2),
                is_overlap=t.is_overlap,
                confidence=t.confidence,
            ))

    if not cleaned_turns:
        return []

    # Step 2: Merge consecutive turns of the SAME speaker if silence gap <= max_merge_gap_s (1.0s)
    merged_turns: List[SpeakerTurn] = []
    for t in cleaned_turns:
        if (
            merged_turns
            and merged_turns[-1].speaker_id == t.speaker_id
            and (t.start_s - merged_turns[-1].end_s) <= max_merge_gap_s
        ):
            # Same speaker with silence gap <= 1.0s -> merge into one coherent turn!
            merged_turns[-1].end_s = max(merged_turns[-1].end_s, t.end_s)
            merged_turns[-1].duration_s = round(merged_turns[-1].end_s - merged_turns[-1].start_s, 2)
        else:
            merged_turns.append(t)

    # Step 3: Filter turns below min_duration_s
    final_turns = [t for t in merged_turns if t.duration_s >= min_duration_s]
    final_turns.sort(key=lambda x: x.start_s)
    return final_turns if final_turns else merged_turns


class BaseDiarizer(ABC):
    """Abstract base class for all diarization implementations."""

    @abstractmethod
    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        filter_overlap: bool = True,
        min_duration_s: float = 0.5,
    ) -> List[SpeakerTurn]:
        """Perform speaker diarization on given audio file."""
        pass
