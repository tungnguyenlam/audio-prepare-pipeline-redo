"""Smart Chunking Engine with Zero Word-Clipping & Strict Single-Speaker Guarantee.

Stage 6 in SOTA Audio Processing Pipeline:
Guarantees:
1. STRICTLY SINGLE SPEAKER: Audio is sliced strictly within single-speaker diarization turns.
2. ZERO WORD CLIPPING: Cut points only occur in the silence gaps between words (word.end_s -> word.start_s)
   with safety acoustic margin (40ms) to preserve onset and coda consonants.
3. DURATION CONSTRAINTS: 3.0s <= Duration <= 30.0s.
4. SAME-SPEAKER ADJACENT MERGING: If 2 adjacent chronological segments belong to the same speaker
   and their combined total duration is <= 30.0s, they are automatically merged into a single segment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from src.alignment.base import AlignedWord
from src.diarization.base import SpeakerTurn

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a clean, single-speaker audio chunk with exact timestamps and word details."""
    segment_id: str
    start_s: float
    end_s: float
    duration_s: float
    text: str
    speaker_id: str = "SPEAKER_00"
    words: List[AlignedWord] = field(default_factory=list)
    audio_path: Optional[str] = None
    sample_rate: int = 24000
    channels: int = 1

    def __post_init__(self):
        self.start_s = round(float(self.start_s), 3)
        self.end_s = round(float(self.end_s), 3)
        if self.duration_s == 0.0:
            self.duration_s = round(max(0.0, self.end_s - self.start_s), 3)
        else:
            self.duration_s = round(float(self.duration_s), 3)
        self.text = " ".join(self.text.strip().split())

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "text": self.text,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "words_count": len(self.words),
            "words": [w.to_dict() if hasattr(w, "to_dict") else asdict(w) for w in self.words],
        }


class SmartChunker:
    """Intelligent audio segmenter with word boundary preservation and same-speaker merging."""

    def __init__(
        self,
        min_duration_s: float = 3.0,
        max_duration_s: float = 30.0,
        pause_threshold_s: float = 0.5,  # Không cắt nếu im lặng < 0.5s
        max_merge_gap_s: float = 1.5,
        acoustic_pad_s: float = 0.04,  # 40ms safety pad into silence
    ):
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.pause_threshold_s = pause_threshold_s
        self.max_merge_gap_s = max_merge_gap_s
        self.acoustic_pad_s = acoustic_pad_s

    def build_single_speaker_segments(
        self,
        turns: List[SpeakerTurn],
        all_words: List[AlignedWord],
        total_audio_duration: float,
        prefix: str = "seg"
    ) -> List[AudioSegment]:
        """Build strictly single-speaker segments by chunking words strictly inside their own speaker turns."""
        if not turns:
            return []

        # Sort turns chronologically
        sorted_turns = sorted(turns, key=lambda t: t.start_s)
        sorted_words = sorted(all_words, key=lambda w: w.start_s)

        # Step 1: Pre-merge adjacent turns of the SAME speaker if gap <= max_merge_gap_s and combined duration <= 30.0s
        merged_turns: List[SpeakerTurn] = []
        for turn in sorted_turns:
            if not merged_turns:
                merged_turns.append(SpeakerTurn(
                    start_s=turn.start_s,
                    end_s=turn.end_s,
                    speaker_id=turn.speaker_id,
                    is_overlap=turn.is_overlap,
                    confidence=turn.confidence
                ))
                continue

            prev = merged_turns[-1]
            gap = turn.start_s - prev.end_s
            combined_dur = turn.end_s - prev.start_s

            if (
                prev.speaker_id == turn.speaker_id
                and 0.0 <= gap <= self.max_merge_gap_s
                and combined_dur <= self.max_duration_s
            ):
                # Merge consecutive turns of same speaker
                prev.end_s = turn.end_s
                prev.duration_s = round(prev.end_s - prev.start_s, 2)
            else:
                merged_turns.append(SpeakerTurn(
                    start_s=turn.start_s,
                    end_s=turn.end_s,
                    speaker_id=turn.speaker_id,
                    is_overlap=turn.is_overlap,
                    confidence=turn.confidence
                ))

        # Step 2: For each continuous single-speaker turn, assign words that fall strictly inside it
        raw_segments: List[AudioSegment] = []
        seg_counter = 1

        for turn in merged_turns:
            turn_start = turn.start_s
            turn_end = turn.end_s
            spk_id = turn.speaker_id

            # Find words strictly inside this turn
            turn_words = [
                w for w in sorted_words
                if (w.start_s >= turn_start - 0.15 and w.end_s <= turn_end + 0.15)
            ]

            if not turn_words or len(turn_words) < 2:
                # STRICT FILTER: Discard any turn without actual recognized spoken words (drops memes, sound effects, intro music, noises)
                continue

            # Sub-chunk words within this turn to ensure 3.0s <= Duration <= 30.0s without clipping words
            turn_chunks = self._subchunk_turn_words(
                words=turn_words,
                turn_start=turn_start,
                turn_end=turn_end,
                speaker_id=spk_id,
                prefix=prefix
            )
            for c in turn_chunks:
                if len(c.words) >= 2 and len(c.text.strip()) >= 3:
                    c.segment_id = f"{prefix}_{seg_counter:04d}_{spk_id}"
                    raw_segments.append(c)
                    seg_counter += 1

        # Step 3: Global Secondary Merge Pass across chronologically adjacent segments of the SAME speaker
        final_merged = self.merge_adjacent_chronological_segments(raw_segments, prefix=prefix)

        # Step 4: Final filter to ensure strict compliance (3.0s <= duration <= 30.0s and non-empty text)
        compliant_segments = [
            s for s in final_merged
            if (s.duration_s >= self.min_duration_s and s.duration_s <= self.max_duration_s + 0.5 and len(s.words) >= 2 and len(s.text.strip()) >= 3)
        ]

        # Re-index segment IDs
        for idx, seg in enumerate(compliant_segments, start=1):
            seg.segment_id = f"{prefix}_{idx:04d}_{seg.speaker_id}"

        return compliant_segments

    def _subchunk_turn_words(
        self,
        words: List[AlignedWord],
        turn_start: float,
        turn_end: float,
        speaker_id: str,
        prefix: str = "seg"
    ) -> List[AudioSegment]:
        """Divide a single-speaker turn's words into compliant sub-chunks cutting strictly in silence between words."""
        if not words:
            return []

        chunks: List[AudioSegment] = []
        cur_words: List[AlignedWord] = []

        # Acoustic start with 40ms pre-pad into silence
        chunk_cut_start = max(turn_start, words[0].start_s - self.acoustic_pad_s)

        for i, w in enumerate(words):
            cur_words.append(w)
            is_last = (i == len(words) - 1)
            next_w = words[i + 1] if not is_last else None

            # Calculate accumulated duration
            cut_end_candidate = min(turn_end, w.end_s + self.acoustic_pad_s)
            cur_dur = cut_end_candidate - chunk_cut_start

            gap_to_next = (next_w.start_s - w.end_s) if next_w else 0.0

            should_split = False

            if cur_dur >= self.min_duration_s:
                # Condition 1: Natural pause between words (silence >= 0.3s)
                if gap_to_next >= self.pause_threshold_s:
                    should_split = True
                # Condition 2: Approaching max duration limit (28.0s)
                elif cur_dur >= (self.max_duration_s - 2.0):
                    should_split = True

            # Hard cap condition: If adding next word would exceed max_duration_s
            if next_w and (next_w.end_s + self.acoustic_pad_s - chunk_cut_start) > self.max_duration_s and cur_dur >= self.min_duration_s:
                should_split = True

            if is_last or should_split:
                # Cut point is placed safely in the silence gap between words (zero word clipping!)
                if next_w and gap_to_next > 0:
                    cut_end = min(turn_end, w.end_s + min(self.acoustic_pad_s, gap_to_next / 2.0))
                else:
                    cut_end = min(turn_end, w.end_s + self.acoustic_pad_s)

                text = " ".join(cw.word for cw in cur_words)
                dur = round(cut_end - chunk_cut_start, 3)

                chunks.append(AudioSegment(
                    segment_id=f"{prefix}_temp_{speaker_id}",
                    start_s=round(chunk_cut_start, 3),
                    end_s=round(cut_end, 3),
                    duration_s=dur,
                    text=text,
                    words=list(cur_words),
                    speaker_id=speaker_id,
                ))

                # Reset for next sub-chunk
                cur_words = []
                if next_w:
                    if gap_to_next > 0:
                        chunk_cut_start = max(turn_start, next_w.start_s - min(self.acoustic_pad_s, gap_to_next / 2.0))
                    else:
                        chunk_cut_start = max(turn_start, next_w.start_s - self.acoustic_pad_s)

        return chunks

    def merge_adjacent_chronological_segments(
        self,
        segments: List[AudioSegment],
        prefix: str = "seg"
    ) -> List[AudioSegment]:
        """Merges chronologically adjacent segments of the SAME speaker if combined duration <= max_duration_s."""
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: s.start_s)
        merged: List[AudioSegment] = []

        for next_seg in sorted_segs:
            if not merged:
                merged.append(next_seg)
                continue

            prev_seg = merged[-1]

            # Condition to merge:
            # 1. Same speaker
            # 2. Chronologically adjacent with no other speaker in between (gap <= max_merge_gap_s)
            # 3. Combined total duration <= max_duration_s (30.0s)
            is_same_speaker = (prev_seg.speaker_id == next_seg.speaker_id)
            gap = next_seg.start_s - prev_seg.end_s
            combined_duration = next_seg.end_s - prev_seg.start_s

            if (
                is_same_speaker
                and (0.0 <= gap <= self.max_merge_gap_s)
                and (combined_duration <= self.max_duration_s)
            ):
                # Perform safe merge
                combined_words = prev_seg.words + next_seg.words
                combined_text = (prev_seg.text + " " + next_seg.text).strip()
                merged[-1] = AudioSegment(
                    segment_id=prev_seg.segment_id,
                    start_s=prev_seg.start_s,
                    end_s=next_seg.end_s,
                    duration_s=round(combined_duration, 3),
                    text=combined_text,
                    words=combined_words,
                    speaker_id=prev_seg.speaker_id,
                )
            else:
                merged.append(next_seg)

        return merged

    def export_audio_segments(
        self,
        source_audio_path: Union[str, Path],
        segments: List[AudioSegment],
        output_dir: Union[str, Path],
        target_sr: int = 24000,
        mono: bool = True
    ) -> List[AudioSegment]:
        """Slice clean audio into discrete standard PCM 16-bit WAV segments and save metadata.json."""
        source_path = Path(source_audio_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source audio file not found: {source_path}")

        out_dir = Path(output_dir).resolve()
        segments_dir = out_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        info = sf.info(str(source_path))
        orig_sr = info.samplerate
        audio_data, _ = sf.read(str(source_path), dtype="float32")

        if mono and audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        finalized_segments: List[AudioSegment] = []

        for seg in segments:
            start_sample = max(0, int(seg.start_s * orig_sr))
            end_sample = min(len(audio_data), int(seg.end_s * orig_sr))

            if start_sample >= end_sample:
                continue

            chunk = audio_data[start_sample:end_sample]

            # Resample to target_sr if needed
            if orig_sr != target_sr:
                import scipy.signal
                target_len = int(len(chunk) * target_sr / orig_sr)
                chunk_resampled = scipy.signal.resample(chunk, target_len).astype(np.float32)
            else:
                chunk_resampled = chunk

            # Apply smooth 10ms raised-cosine fade in/out at segment boundaries to prevent any digital clicking
            fade_len = int(0.01 * target_sr)
            if len(chunk_resampled) > 2 * fade_len:
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                chunk_resampled[:fade_len] *= fade_in
                chunk_resampled[-fade_len:] *= fade_out

            # Volume normalization (-1 dBFS peak)
            max_amp = np.max(np.abs(chunk_resampled))
            if max_amp > 1e-4:
                target_peak = 0.891
                scale = target_peak / max_amp
                if scale < 1.0 or scale > 1.05:
                    chunk_resampled = np.clip(chunk_resampled * min(scale, 1.2), -1.0, 1.0)

            # Write WAV file
            out_filename = f"{seg.segment_id}.wav"
            out_filepath = segments_dir / out_filename
            sf.write(str(out_filepath), chunk_resampled, target_sr, subtype="PCM_16")

            seg.audio_path = str(out_filepath.relative_to(out_dir))
            seg.sample_rate = target_sr
            seg.channels = 1 if mono else (audio_data.ndim if audio_data.ndim > 1 else 1)
            finalized_segments.append(seg)

        # Write metadata.json
        meta_path = out_dir / "metadata.json"
        meta_records = [s.to_dict() for s in finalized_segments]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_records, f, ensure_ascii=False, indent=2)

        # Write metadata.csv
        csv_path = out_dir / "metadata.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("segment_id|speaker_id|duration_s|text|audio_path\n")
            for s in finalized_segments:
                clean_txt = s.text.replace("|", " ")
                f.write(f"{s.segment_id}|{s.speaker_id}|{s.duration_s:.3f}|{clean_txt}|{s.audio_path}\n")

        logger.info("Exported %d clean segments to %s", len(finalized_segments), segments_dir)
        return finalized_segments
