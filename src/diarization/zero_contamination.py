"""Zero-contamination high-precision speaker diarization pipeline.

Designed for regimes where false negatives (missed detection) have zero penalty
and the sole objective is absolute single-speaker purity (zero 2-speaker segments).

This module provides modular, reusable components:
1. Asymmetric detection thresholds & competitor tripwires
2. Dual-engine consensus (e.g. Sortformer ∩ DiariZen) via Hungarian alignment
3. Aggressive boundary collar erosion and transition excavation
4. Dense sliding-window WeSpeaker embedding homogeneity checks
5. In-loop foundation model verification (Gemma 4 direct audio & VibeVoice-ASR)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf
import torch

from src.base.model import ManagedModel
from src.diarization.evaluation import _maximum_weight_assignment
from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    Speaker,
    SpeakerTurn,
)
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

DEFAULT_COLLAR_EROSION_S = 0.35
DEFAULT_MIN_TURN_DURATION_S = 0.80
DEFAULT_TRANSITION_EXCLUSION_S = 0.50
DEFAULT_TARGET_ONSET = 0.80
DEFAULT_TARGET_OFFSET = 0.65
DEFAULT_COMPETITOR_ONSET = 0.20
DEFAULT_HOMOGENEITY_WINDOW_S = 1.00
DEFAULT_HOMOGENEITY_HOP_S = 0.25
DEFAULT_MIN_HOMOGENEITY_SIMILARITY = 0.75


DEFAULT_HANDOFF_RISK_DISTANCE_S = 0.80
DEFAULT_SILENCE_TAIL_BUFFER_S = 0.027
DEFAULT_ENERGY_SEARCH_WINDOW_S = 0.15
DEFAULT_ENERGY_VALLEY_FLOOR_DB = -30.0
DEFAULT_ENERGY_FRAME_LEN_MS = 2.0
DEFAULT_ENERGY_HOP_LEN_MS = 0.5

DEFAULT_TARGET_MAX_DURATION_S = 10.0
DEFAULT_TARGET_MIN_DURATION_S = 3.0
DEFAULT_MIN_SPLIT_PAUSE_S = 0.20


def _ensure_torch_hub_trusted() -> None:
    """Ensure torch.hub trusts known repositories non-interactively (e.g. Silero VAD)."""
    try:
        import torch.hub

        # Allow snakers4 in repo owners tuple
        if hasattr(torch.hub, "_TRUSTED_REPO_OWNERS") and isinstance(torch.hub._TRUSTED_REPO_OWNERS, tuple):
            if "snakers4" not in torch.hub._TRUSTED_REPO_OWNERS:
                torch.hub._TRUSTED_REPO_OWNERS = (*torch.hub._TRUSTED_REPO_OWNERS, "snakers4")

        # Persist to hub trusted_list cache file
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        trusted_file = os.path.join(hub_dir, "trusted_list")
        existing: set[str] = set()
        if os.path.exists(trusted_file):
            with open(trusted_file, "r", encoding="utf-8") as f:
                existing = {line.strip() for line in f if line.strip()}
        needed = {"snakers4_silero-vad", "snakers4_silero-vad_master", "snakers4/silero-vad"}
        to_add = [name for name in needed if name not in existing]
        if to_add:
            with open(trusted_file, "a", encoding="utf-8") as f:
                for name in to_add:
                    f.write(f"{name}\n")

        # Monkeypatch torch.hub.load to default trust_repo=True if not explicitly provided
        if not getattr(torch.hub, "_trust_repo_patched", False):
            orig_load = torch.hub.load

            def _patched_hub_load(repo_or_dir, model, *args, **kwargs):
                if kwargs.get("trust_repo") in (None, "check"):
                    kwargs["trust_repo"] = True
                return orig_load(repo_or_dir, model, *args, **kwargs)

            torch.hub.load = _patched_hub_load
            torch.hub._trust_repo_patched = True
    except Exception as exc:
        logger.debug("Failed to pre-trust torch.hub repositories: %s", exc)


_ensure_torch_hub_trusted()


def _json_compatible(value: Any) -> Any:
    """Convert NumPy scalars nested in a result to native JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


@dataclass
class ZeroContaminationConfig:
    """Settings controlling each stage of the zero-contamination pipeline."""

    # Stage 1: Primary Diarizer
    primary_backend: str = "sortformer"  # "sortformer", "diarizen", "pyannote"
    primary_device: str | None = None  # e.g. "cuda:0", "cpu", or None for general device
    target_onset: float = DEFAULT_TARGET_ONSET
    target_offset: float = DEFAULT_TARGET_OFFSET
    competitor_onset: float = DEFAULT_COMPETITOR_ONSET

    # Stage 2: Dual-Engine Consensus
    enable_consensus: bool = True
    secondary_backend: str = "diarizen"  # "diarizen", "pyannote", "sortformer"
    secondary_device: str | None = "same"  # e.g. "same", "cuda:0", "cuda:1", "cpu"

    # Stage 3: Boundary & Syllable Integrity Gate
    enable_collar_erosion: bool = True
    boundary_collar_s: float = DEFAULT_COLLAR_EROSION_S
    min_turn_duration_s: float = DEFAULT_MIN_TURN_DURATION_S
    transition_exclusion_s: float = DEFAULT_TRANSITION_EXCLUSION_S
    allow_gap_merge: bool = False

    # Stage 3a: Option A - Context-Aware Collar & Handoff Guard
    enable_context_collar: bool = True
    handoff_risk_distance_s: float = DEFAULT_HANDOFF_RISK_DISTANCE_S
    silence_tail_buffer_s: float = DEFAULT_SILENCE_TAIL_BUFFER_S

    # Stage 3b: Option B - Syllable / Word Forced Alignment Lock (High-Compute)
    enable_syllable_alignment: bool = False
    aligner_engine: str = "whisper_timestamped"  # "whisper_timestamped", "mms_fa", "remote_whisper"
    aligner_model: str = "vinai/PhoWhisper-small"  # Vietnamese fine-tuned model or standard Whisper
    aligner_language: str = "vi"  # Target language (e.g. "vi" for Vietnamese)
    aligner_endpoint: str | None = None
    aligner_device: str | None = "cpu"  # CPU recommended to prevent GPU VRAM exhaustion

    # Stage 3c: Option C - Micro-Acoustic Energy & RMS Silence Valley Snapping
    enable_energy_snapping: bool = False
    energy_search_window_s: float = DEFAULT_ENERGY_SEARCH_WINDOW_S
    energy_valley_floor_db: float = DEFAULT_ENERGY_VALLEY_FLOOR_DB
    energy_frame_len_ms: float = DEFAULT_ENERGY_FRAME_LEN_MS
    energy_hop_len_ms: float = DEFAULT_ENERGY_HOP_LEN_MS

    # Stage 3d: Option D - Intelligent ASR & Pause-Guided Turn Segmentation (TTS Sentence Sizing)
    enable_smart_segmentation: bool = False
    target_max_duration_s: float = DEFAULT_TARGET_MAX_DURATION_S
    target_min_duration_s: float = DEFAULT_TARGET_MIN_DURATION_S
    min_split_pause_s: float = DEFAULT_MIN_SPLIT_PAUSE_S

    # Stage 4: Dense Sliding-Window Embedding Homogeneity
    enable_homogeneity: bool = False
    homogeneity_device: str | None = "same"  # e.g. "same", "cuda:0", "cpu"
    homogeneity_window_s: float = DEFAULT_HOMOGENEITY_WINDOW_S
    homogeneity_hop_s: float = DEFAULT_HOMOGENEITY_HOP_S
    min_homogeneity_similarity: float = DEFAULT_MIN_HOMOGENEITY_SIMILARITY

    # Stage 5a: In-Loop VibeVoice-ASR Speaker Count Verifier (Dedicated GPU or Remote Host)
    enable_vibevoice: bool = False
    vibevoice_model_id: str = "Dubedo/VibeVoice-ASR-HF-INT8"
    vibevoice_device: str | None = None  # e.g. "cuda:1" to run on a dedicated secondary GPU, or "cpu"
    vibevoice_endpoint: str | None = None  # optional remote HTTP endpoint if hosted on another server
    max_secondary_speech_s: float = 0.0

    # Stage 5b: Direct-audio speaker-purity and word-completeness verifier
    enable_gemma: bool = False
    gemma_backend: str = "gemini"  # "gemini" or "gemma4"
    gemma_endpoint: str | None = None
    gemma_model: str | None = "gemini-3.8-flash"
    gemma_prompt: str | None = None
    gemma_api_key: str | None = None
    gemma_timeout_s: float = 120.0
    gemma_max_output_tokens: int = 1024

    # General compute settings
    device: str = "auto"
    token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnAuditRecord:
    """Traceability record for an individual turn through the pipeline."""

    turn_id: str
    speaker_id: str
    original_start_s: float
    original_end_s: float
    start_s: float
    end_s: float
    duration_s: float
    status: str  # "passed", "rejected_consensus", "rejected_collar", "rejected_homogeneity", "rejected_gemma", "rejected_vibevoice"
    rejection_reason: str = ""
    min_similarity: float | None = None
    gemma_decision: dict[str, Any] | None = None
    vibevoice_decision: dict[str, Any] | None = None
    # Syllable & Boundary Integrity A/B audition metadata
    raw_start_s: float | None = None
    raw_end_s: float | None = None
    delta_start_ms: float = 0.0
    delta_end_ms: float = 0.0
    boundary_policy: str = "standard"  # "context_aware", "acoustic_valley", "word_locked", "standard"
    transcript: str | None = None
    tail_rescued: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


@dataclass
class ZeroContaminationResult:
    """Comprehensive result containing clean turns and funnel statistics."""

    diarization: DiarizationResult
    audit_records: list[TurnAuditRecord]
    funnel_stats: dict[str, Any]
    stage_log: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    foundation_audits: list[dict[str, Any]] = field(default_factory=list)
    boundary_audits: list[dict[str, Any]] = field(default_factory=list)
    segment_audits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        diar_dict = self.diarization.to_dict()
        enriched_turns = []
        for i, turn in enumerate(diar_dict.get("turns", [])):
            rec = self.audit_records[i] if i < len(self.audit_records) else None
            t_data = dict(turn)
            if rec:
                t_data["raw_start_s"] = rec.raw_start_s if rec.raw_start_s is not None else rec.original_start_s
                t_data["raw_end_s"] = rec.raw_end_s if rec.raw_end_s is not None else rec.original_end_s
                t_data["delta_start_ms"] = rec.delta_start_ms
                t_data["delta_end_ms"] = rec.delta_end_ms
                t_data["boundary_policy"] = rec.boundary_policy
                t_data["tail_rescued"] = rec.tail_rescued
                t_data["transcript"] = rec.transcript
            enriched_turns.append(t_data)
        diar_dict["turns"] = enriched_turns

        return _json_compatible({
            "diarization": diar_dict,
            "audit_records": [rec.to_dict() for rec in self.audit_records],
            "funnel_stats": self.funnel_stats,
            "stage_log": self.stage_log,
            "config": self.config,
            "foundation_audits": self.foundation_audits,
            "boundary_audits": self.boundary_audits,
            "segment_audits": self.segment_audits,
        })


def compute_consensus_turns(
    primary_turns: Sequence[SpeakerTurn],
    secondary_turns: Sequence[SpeakerTurn],
    audio_duration_s: float,
) -> tuple[list[SpeakerTurn], dict[str, str]]:
    """Intersect two diarization outputs using Hungarian speaker alignment.

    An interval [t1, t2] is preserved for Speaker S if and only if:
    1. Primary model asserts Speaker S is speaking AND no other speaker is active.
    2. Secondary model asserts the corresponding aligned speaker is speaking AND
       no other speaker is active.

    Returns:
        Consensus single-speaker turns and the speaker mapping dict.
    """
    if not primary_turns or not secondary_turns or audio_duration_s <= 0:
        return [], {}

    prim_speakers = sorted({t.speaker_id for t in primary_turns})
    sec_speakers = sorted({t.speaker_id for t in secondary_turns})

    # 1. Compute overlap weights between primary and secondary speakers
    weights: dict[tuple[str, str], float] = {}
    for pt in primary_turns:
        for st in secondary_turns:
            overlap = max(0.0, min(pt.end_s, st.end_s) - max(pt.start_s, st.start_s))
            if overlap > 0:
                key = (pt.speaker_id, st.speaker_id)
                weights[key] = weights.get(key, 0.0) + overlap

    # 2. Hungarian 1-to-1 speaker alignment
    prim_to_sec = _maximum_weight_assignment(prim_speakers, sec_speakers, weights)
    if not prim_to_sec:
        return [], {}

    # 3. Discretize into unique event boundaries
    boundaries = {0.0, float(audio_duration_s)}
    for t in list(primary_turns) + list(secondary_turns):
        boundaries.add(max(0.0, min(float(audio_duration_s), t.start_s)))
        boundaries.add(max(0.0, min(float(audio_duration_s), t.end_s)))
    ordered = sorted(boundaries)

    consensus_slices: list[tuple[float, float, str]] = []
    for start_s, end_s in zip(ordered, ordered[1:]):
        if end_s <= start_s + 1e-4:
            continue
        mid = (start_s + end_s) / 2.0

        active_prim = {
            t.speaker_id for t in primary_turns if t.start_s <= mid < t.end_s
        }
        active_sec = {
            t.speaker_id for t in secondary_turns if t.start_s <= mid < t.end_s
        }

        # Mutual single-speaker agreement check
        if len(active_prim) == 1 and len(active_sec) == 1:
            p_spk = next(iter(active_prim))
            s_spk = next(iter(active_sec))
            if prim_to_sec.get(p_spk) == s_spk:
                consensus_slices.append((start_s, end_s, p_spk))

    if not consensus_slices:
        return [], prim_to_sec

    # 4. Merge adjacent consecutive consensus slices with the same speaker
    merged_turns: list[SpeakerTurn] = []
    current_start, current_end, current_spk = consensus_slices[0]
    for s_start, s_end, s_spk in consensus_slices[1:]:
        if s_spk == current_spk and abs(s_start - current_end) < 1e-4:
            current_end = s_end
        else:
            t = SpeakerTurn(
                speaker_id=current_spk,
                start_s=round(current_start, 4),
                end_s=round(current_end, 4),
                confidence=1.0,
            )
            t._consensus_start_s = t.start_s
            t._consensus_end_s = t.end_s
            merged_turns.append(t)
            current_start, current_end, current_spk = s_start, s_end, s_spk

    last_t = SpeakerTurn(
        speaker_id=current_spk,
        start_s=round(current_start, 4),
        end_s=round(current_end, 4),
        confidence=1.0,
    )
    last_t._consensus_start_s = last_t.start_s
    last_t._consensus_end_s = last_t.end_s
    merged_turns.append(last_t)

    return merged_turns, prim_to_sec


def _extract_competitor_intervals_by_speaker(
    primary_turns: Sequence[SpeakerTurn],
    secondary_turns: Sequence[SpeakerTurn] | None = None,
    spk_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Extract competitor speech intervals for each primary speaker.

    For speaker S (in primary speaker namespace), any speech from other speakers
    detected by Primary OR Secondary is preserved as a competitor interval.
    This guarantees that consensus does not destroy competitor evidence near boundaries.
    """
    all_speakers = sorted({t.speaker_id for t in primary_turns})
    sec_to_prim = {sec: prim for prim, sec in (spk_map or {}).items()}

    intervals_by_spk: dict[str, list[tuple[float, float]]] = {}

    for s in all_speakers:
        raw_intervals: list[tuple[float, float]] = []

        # 1. Primary competitor turns (speaker != s)
        for pt in primary_turns:
            if pt.speaker_id != s and pt.end_s > pt.start_s:
                raw_intervals.append((pt.start_s, pt.end_s))

        # 2. Secondary competitor turns (mapped speaker != s or unmapped)
        if secondary_turns:
            for st in secondary_turns:
                if st.end_s <= st.start_s:
                    continue
                mapped = sec_to_prim.get(st.speaker_id)
                if mapped != s:
                    raw_intervals.append((st.start_s, st.end_s))

        if not raw_intervals:
            intervals_by_spk[s] = []
            continue

        raw_intervals.sort(key=lambda x: x[0])
        merged: list[tuple[float, float]] = [raw_intervals[0]]
        for cur_s, cur_e in raw_intervals[1:]:
            prev_s, prev_e = merged[-1]
            if cur_s <= prev_e:
                merged[-1] = (prev_s, max(prev_e, cur_e))
            else:
                merged.append((cur_s, cur_e))

        intervals_by_spk[s] = merged

    return intervals_by_spk


def erode_turn_boundaries(
    turns: Sequence[SpeakerTurn],
    collar_s: float = DEFAULT_COLLAR_EROSION_S,
    min_duration_s: float = DEFAULT_MIN_TURN_DURATION_S,
    transition_exclusion_s: float = DEFAULT_TRANSITION_EXCLUSION_S,
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> list[SpeakerTurn]:
    """Shave inward margins from turn boundaries and excavate speaker transitions.

    Args:
        turns: Speaker turns sorted by start time.
        collar_s: Inward margin shaved from start and end of every turn.
        min_duration_s: Minimum surviving duration required.
        transition_exclusion_s: When different speakers change with gap smaller
            than this threshold, extra safety padding is excavated.
        competitor_intervals_by_speaker: Pre-extracted competitor intervals per speaker
            preserving raw evidence from primary and secondary diarizers.

    Returns:
        Eroded, boundary-safe single-speaker turns.
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: (t.start_s, t.end_s))
    eroded: list[SpeakerTurn] = []

    for index, turn in enumerate(sorted_turns):
        start = turn.start_s + collar_s
        end = turn.end_s - collar_s
        spk = turn.speaker_id
        comp_intervals = (
            competitor_intervals_by_speaker.get(spk, [])
            if competitor_intervals_by_speaker is not None
            else None
        )

        # Transition guard: check distance to preceding competitor
        closest_prev_gap = float("inf")
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn.start_s < c_e:
                    closest_prev_gap = min(closest_prev_gap, 0.0)
                elif c_e <= turn.start_s:
                    closest_prev_gap = min(closest_prev_gap, turn.start_s - c_e)
        if index > 0:
            prev_turn = sorted_turns[index - 1]
            if prev_turn.speaker_id != turn.speaker_id:
                gap = max(0.0, turn.start_s - prev_turn.end_s)
                closest_prev_gap = min(closest_prev_gap, gap)

        if closest_prev_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_prev_gap) / 2.0)
            start += extra

        # Transition guard: check distance to following competitor
        closest_next_gap = float("inf")
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn.end_s < c_e:
                    closest_next_gap = min(closest_next_gap, 0.0)
                elif c_s >= turn.end_s:
                    closest_next_gap = min(closest_next_gap, c_s - turn.end_s)
        if index + 1 < len(sorted_turns):
            next_turn = sorted_turns[index + 1]
            if next_turn.speaker_id != turn.speaker_id:
                gap = max(0.0, next_turn.start_s - turn.end_s)
                closest_next_gap = min(closest_next_gap, gap)

        if closest_next_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_next_gap) / 2.0)
            end -= extra

        if end - start >= min_duration_s:
            final_start = round(start, 4)
            final_end = round(end, 4)
            t = SpeakerTurn(
                speaker_id=turn.speaker_id,
                start_s=final_start,
                end_s=final_end,
                confidence=turn.confidence,
            )
            t._original_start_s = getattr(turn, "_original_start_s", turn.start_s)
            t._original_end_s = getattr(turn, "_original_end_s", turn.end_s)
            t._raw_start_s = final_start
            t._raw_end_s = final_end
            t._delta_start_ms = 0.0
            t._delta_end_ms = 0.0
            t._boundary_policy = "standard"
            t._tail_rescued = False
            if hasattr(turn, "_consensus_start_s"):
                t._consensus_start_s = turn._consensus_start_s
            if hasattr(turn, "_consensus_end_s"):
                t._consensus_end_s = turn._consensus_end_s
            eroded.append(t)

    return eroded


def apply_context_aware_collar(
    turns: Sequence[SpeakerTurn],
    *,
    collar_s: float = DEFAULT_COLLAR_EROSION_S,
    handoff_risk_s: float = DEFAULT_HANDOFF_RISK_DISTANCE_S,
    silence_tail_s: float = DEFAULT_SILENCE_TAIL_BUFFER_S,
    min_duration_s: float = DEFAULT_MIN_TURN_DURATION_S,
    transition_exclusion_s: float = DEFAULT_TRANSITION_EXCLUSION_S,
    audio_duration_s: float | None = None,
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Apply asymmetric context-aware collar erosion to preserve word and syllable endings.

    If a turn is adjacent to another speaker within handoff_risk_s, it is aggressively
    eroded inward to eliminate speaker bleed. If the turn transitions into natural silence
    (or speech ceases with no rival speaker nearby), the trailing coda/tone is preserved
    and gently extended by silence_tail_s.

    Crucially, competitor proximity is checked against competitor_intervals_by_speaker
    (unifying primary ∪ secondary diarizer detections) to ensure consensus filtering
    does not mask true competitor handoffs.
    """
    if not turns:
        return [], []

    sorted_turns = sorted(turns, key=lambda t: (t.start_s, t.end_s))
    refined: list[SpeakerTurn] = []
    audits: list[dict[str, Any]] = []

    for index, turn in enumerate(sorted_turns):
        start = turn.start_s
        end = turn.end_s
        start_shaved = False
        end_shaved = False
        spk = turn.speaker_id
        comp_intervals = (
            competitor_intervals_by_speaker.get(spk, [])
            if competitor_intervals_by_speaker is not None
            else None
        )

        # Calculate standard blunt collar baseline for A/B auditioning
        blunt_start = turn.start_s + collar_s
        blunt_end = turn.end_s - collar_s

        # 1. Start boundary check: check distance to preceding competitor
        closest_prev_gap = float("inf")
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn.start_s < c_e:
                    closest_prev_gap = min(closest_prev_gap, 0.0)
                elif c_e <= turn.start_s:
                    closest_prev_gap = min(closest_prev_gap, turn.start_s - c_e)
        if index > 0:
            prev_turn = sorted_turns[index - 1]
            if prev_turn.speaker_id != turn.speaker_id:
                gap = max(0.0, turn.start_s - prev_turn.end_s)
                closest_prev_gap = min(closest_prev_gap, gap)

        if closest_prev_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_prev_gap) / 2.0)
            blunt_start += extra
        if closest_prev_gap < handoff_risk_s:
            extra = (
                max(0.0, (transition_exclusion_s - closest_prev_gap) / 2.0)
                if closest_prev_gap < transition_exclusion_s
                else 0.0
            )
            start = turn.start_s + collar_s + extra
            start_shaved = True

        # 2. End boundary check: check distance to following competitor
        closest_next_gap = float("inf")
        closest_next_start = float("inf")
        if comp_intervals is not None:
            for c_s, c_e in comp_intervals:
                if c_s < turn.end_s < c_e:
                    closest_next_gap = min(closest_next_gap, 0.0)
                    closest_next_start = min(closest_next_start, c_s)
                elif c_s >= turn.end_s:
                    gap = c_s - turn.end_s
                    closest_next_gap = min(closest_next_gap, gap)
                    closest_next_start = min(closest_next_start, c_s)
        next_turn = sorted_turns[index + 1] if index + 1 < len(sorted_turns) else None
        if next_turn is not None and next_turn.speaker_id != turn.speaker_id:
            gap = max(0.0, next_turn.start_s - turn.end_s)
            closest_next_gap = min(closest_next_gap, gap)
            closest_next_start = min(closest_next_start, next_turn.start_s)

        if closest_next_gap < transition_exclusion_s:
            extra = max(0.0, (transition_exclusion_s - closest_next_gap) / 2.0)
            blunt_end -= extra

        if closest_next_gap < handoff_risk_s:
            extra = (
                max(0.0, (transition_exclusion_s - closest_next_gap) / 2.0)
                if closest_next_gap < transition_exclusion_s
                else 0.0
            )
            end = turn.end_s - (collar_s + extra)
            end_shaved = True
        else:
            # Natural silence or monologue pause: safe to extend with silence_tail_s
            max_limit = (
                audio_duration_s
                if audio_duration_s
                else (turn.end_s + silence_tail_s + 1.0)
            )
            if next_turn is not None:
                max_limit = min(max_limit, next_turn.start_s - 0.05)
            if closest_next_start < float("inf"):
                max_limit = min(max_limit, closest_next_start - 0.05)
            end = min(turn.end_s + silence_tail_s, max_limit)

        if blunt_end <= blunt_start + 0.05:
            blunt_start = turn.start_s
            blunt_end = turn.end_s
        else:
            blunt_start = round(blunt_start, 4)
            blunt_end = round(blunt_end, 4)

        if end - start >= min_duration_s:
            final_start = round(start, 4)
            final_end = round(end, 4)
            orig_start = getattr(turn, "_original_start_s", turn.start_s)
            orig_end = getattr(turn, "_original_end_s", turn.end_s)
            delta_start = round((final_start - blunt_start) * 1000.0, 1)
            delta_end = round((final_end - orig_end) * 1000.0, 1)

            refined_turn = SpeakerTurn(
                speaker_id=turn.speaker_id,
                start_s=final_start,
                end_s=final_end,
                confidence=turn.confidence,
            )
            refined_turn._original_start_s = orig_start
            refined_turn._original_end_s = orig_end
            refined_turn._raw_start_s = blunt_start
            refined_turn._raw_end_s = blunt_end
            refined_turn._delta_start_ms = delta_start
            refined_turn._delta_end_ms = delta_end
            refined_turn._boundary_policy = "context_aware_collar"
            refined_turn._tail_rescued = not end_shaved
            if hasattr(turn, "_consensus_start_s"):
                refined_turn._consensus_start_s = turn._consensus_start_s
            if hasattr(turn, "_consensus_end_s"):
                refined_turn._consensus_end_s = turn._consensus_end_s

            refined.append(refined_turn)
            audits.append({
                "raw_start_s": blunt_start,
                "raw_end_s": blunt_end,
                "original_start_s": orig_start,
                "original_end_s": orig_end,
                "start_s": final_start,
                "end_s": final_end,
                "delta_start_ms": delta_start,
                "delta_end_ms": delta_end,
                "start_shaved": start_shaved,
                "end_shaved": end_shaved,
                "tail_rescued": not end_shaved,
                "policy": "context_aware_collar",
            })

    return refined, audits


def _find_local_valley(
    waveform: np.ndarray,
    center_sample: int,
    *,
    search_samples: int,
    frame_samples: int,
    hop_samples: int,
) -> int:
    """Find local RMS energy valley and zero-crossing in waveform around center_sample."""
    left = max(0, center_sample - search_samples)
    right = min(len(waveform), center_sample + search_samples)
    if right - left < frame_samples:
        return center_sample

    segment = waveform[left:right]
    num_frames = (len(segment) - frame_samples) // hop_samples + 1
    if num_frames <= 0:
        return center_sample

    energies = []
    for f in range(num_frames):
        f_start = f * hop_samples
        frame = segment[f_start : f_start + frame_samples]
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        energies.append(rms)

    energies = np.array(energies)
    center_frame = (center_sample - left) / hop_samples
    frame_indices = np.arange(len(energies))
    dist_penalty = (frame_indices - center_frame) ** 2 * 0.05
    cost = energies + (dist_penalty * np.median(energies) * 0.1)

    best_frame = int(np.argmin(cost))
    best_sample = left + best_frame * hop_samples + (frame_samples // 2)

    # Zero-crossing alignment to prevent audio clicks
    zc_window = waveform[max(0, best_sample - 20) : min(len(waveform), best_sample + 20)]
    zc_indices = np.where(np.diff(np.signbit(zc_window)))[0]
    if len(zc_indices) > 0:
        best_sample = max(0, best_sample - 20) + zc_indices[0]

    return best_sample


def snap_boundaries_to_acoustic_valleys(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    search_window_s: float = DEFAULT_ENERGY_SEARCH_WINDOW_S,
    energy_floor_db: float = DEFAULT_ENERGY_VALLEY_FLOOR_DB,
    frame_len_ms: float = DEFAULT_ENERGY_FRAME_LEN_MS,
    hop_len_ms: float = DEFAULT_ENERGY_HOP_LEN_MS,
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Snap turn boundaries to local short-time energy (RMS) silence valleys.

    Prevents slicing through voiced phonemes, vowels, or coda consonants by walking
    the boundary to the nearest local silence minimum / zero-crossing in the micro-waveform.
    Constrained so boundaries cannot drift into consensus-excluded or competitor speech.
    """
    if not turns:
        return [], []

    waveform, sr = sf.read(str(audio.path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    frame_samples = max(1, int(round(frame_len_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_len_ms * sr / 1000.0)))
    search_samples = int(round(search_window_s * sr))

    snapped: list[SpeakerTurn] = []
    audits: list[dict[str, Any]] = []

    def find_local_valley(center_sample: int) -> int:
        return _find_local_valley(
            waveform,
            center_sample,
            search_samples=search_samples,
            frame_samples=frame_samples,
            hop_samples=hop_samples,
        )

    for turn in turns:
        start_samp = int(round(turn.start_s * sr))
        end_samp = int(round(turn.end_s * sr))

        new_start_samp = find_local_valley(start_samp)
        new_end_samp = find_local_valley(end_samp)

        new_start_s = round(new_start_samp / sr, 4)
        new_end_s = round(new_end_samp / sr, 4)

        # Clamp against consensus bounds if present
        if hasattr(turn, "_consensus_start_s"):
            new_start_s = max(new_start_s, turn._consensus_start_s)
        if hasattr(turn, "_consensus_end_s"):
            new_end_s = min(new_end_s, turn._consensus_end_s)

        # Clamp away from competitor intervals
        if competitor_intervals_by_speaker:
            comp_ivs = competitor_intervals_by_speaker.get(turn.speaker_id, [])
            for c_s, c_e in comp_ivs:
                if c_e <= turn.start_s + 1e-4:
                    new_start_s = max(new_start_s, c_e)
                if c_s >= turn.end_s - 1e-4:
                    new_end_s = min(new_end_s, c_s - 0.05)

        if new_start_s >= new_end_s:
            new_start_s = turn.start_s
            new_end_s = turn.end_s

        orig_start = getattr(turn, "_original_start_s", turn.start_s)
        orig_end = getattr(turn, "_original_end_s", turn.end_s)
        raw_start = getattr(turn, "_raw_start_s", turn.start_s)
        raw_end = getattr(turn, "_raw_end_s", turn.end_s)

        if new_end_s - new_start_s >= 0.30:
            delta_start = round((new_start_s - raw_start) * 1000.0, 1)
            delta_end = round((new_end_s - orig_end) * 1000.0, 1)
            tail_rescued = new_end_s > orig_end

            refined_turn = SpeakerTurn(
                speaker_id=turn.speaker_id,
                start_s=new_start_s,
                end_s=new_end_s,
                confidence=turn.confidence,
            )
            refined_turn._original_start_s = orig_start
            refined_turn._original_end_s = orig_end
            refined_turn._raw_start_s = raw_start
            refined_turn._raw_end_s = raw_end
            refined_turn._delta_start_ms = delta_start
            refined_turn._delta_end_ms = delta_end
            refined_turn._boundary_policy = "acoustic_energy_valley"
            refined_turn._tail_rescued = tail_rescued
            refined_turn._transcript = getattr(turn, "_transcript", None)
            refined_turn._words = getattr(turn, "_words", None)
            if hasattr(turn, "_consensus_start_s"):
                refined_turn._consensus_start_s = turn._consensus_start_s
            if hasattr(turn, "_consensus_end_s"):
                refined_turn._consensus_end_s = turn._consensus_end_s

            snapped.append(refined_turn)
            audits.append({
                "raw_start_s": raw_start,
                "raw_end_s": raw_end,
                "original_start_s": orig_start,
                "original_end_s": orig_end,
                "start_s": new_start_s,
                "end_s": new_end_s,
                "delta_start_ms": delta_start,
                "delta_end_ms": delta_end,
                "policy": "acoustic_energy_valley",
                "tail_rescued": tail_rescued,
                "transcript": getattr(turn, "_transcript", None),
            })
        else:
            snapped.append(turn)
            audits.append({
                "raw_start_s": raw_start,
                "raw_end_s": raw_end,
                "original_start_s": orig_start,
                "original_end_s": orig_end,
                "start_s": turn.start_s,
                "end_s": turn.end_s,
                "delta_start_ms": getattr(turn, "_delta_start_ms", 0.0),
                "delta_end_ms": getattr(turn, "_delta_end_ms", 0.0),
                "policy": getattr(turn, "_boundary_policy", "standard"),
                "tail_rescued": getattr(turn, "_tail_rescued", False),
                "transcript": getattr(turn, "_transcript", None),
            })

    return snapped, audits


def _lock_turns_with_words(
    turns: Sequence[SpeakerTurn],
    words: list[dict[str, Any]],
    policy: str = "whisper_word_lock",
    audio_duration_s: float | None = None,
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Snap speaker turn boundaries to complete words to prevent syllable clipping.

    Constrains word expansions to safe regions:
    1. Does not overlap preceding or succeeding turns (or competitor speakers).
    2. Does not exceed audio duration [0.0, audio_duration_s].
    3. Does not expand outside consensus bounds into disputed speech.
    4. Does not encroach into raw competitor intervals from primary or secondary diarizers.
    """
    if not turns:
        return [], []

    aligned_turns: list[SpeakerTurn] = []
    audits: list[dict[str, Any]] = []
    n = len(turns)

    # Compute safe bounds for each turn index
    sorted_order = sorted(range(n), key=lambda idx: (turns[idx].start_s, turns[idx].end_s))
    safe_bounds: dict[int, tuple[float, float]] = {}
    for rank, orig_idx in enumerate(sorted_order):
        t = turns[orig_idx]
        prev_t = turns[sorted_order[rank - 1]] if rank > 0 else None
        next_t = turns[sorted_order[rank + 1]] if rank < n - 1 else None

        min_s = prev_t.end_s if prev_t is not None else 0.0
        max_e = next_t.start_s if next_t is not None else (audio_duration_s if audio_duration_s is not None else float("inf"))
        if audio_duration_s is not None:
            max_e = min(max_e, audio_duration_s)
            min_s = max(0.0, min_s)

        # Never expand outside the consensus safe zone into disputed speech
        if hasattr(t, "_consensus_start_s"):
            min_s = max(min_s, t._consensus_start_s)
        if hasattr(t, "_consensus_end_s"):
            max_e = min(max_e, t._consensus_end_s)

        # Never expand into competitor speech
        if competitor_intervals_by_speaker:
            comp_ivs = competitor_intervals_by_speaker.get(t.speaker_id, [])
            for c_s, c_e in comp_ivs:
                if c_e <= t.start_s + 1e-4:
                    min_s = max(min_s, c_e)
                if c_s >= t.end_s - 1e-4:
                    max_e = min(max_e, c_s)

        safe_bounds[orig_idx] = (min_s, max_e)

    for i, turn in enumerate(turns):
        orig_start = getattr(turn, "_original_start_s", turn.start_s)
        orig_end = getattr(turn, "_original_end_s", turn.end_s)
        raw_start = getattr(turn, "_raw_start_s", turn.start_s)
        raw_end = getattr(turn, "_raw_end_s", turn.end_s)

        safe_min, safe_max = safe_bounds.get(i, (0.0, audio_duration_s or float("inf")))

        new_start = turn.start_s
        new_end = turn.end_s

        # 1. End boundary protection (Syllable Coda / Trailing Word Guard)
        for w in words:
            w_start = float(w["start"])
            w_end = float(w["end"])
            # Mid-word cut: turn ends during active word
            if w_start <= new_end < w_end:
                new_end = max(new_end, w_end + 0.05)
            # Immediate trailing word: word ended within 150ms after turn end
            elif 0.0 < (w_end - new_end) <= 0.15 and w_start < new_end:
                new_end = max(new_end, w_end + 0.05)

        # 2. Start boundary protection (Leading Word Guard)
        for w in words:
            w_start = float(w["start"])
            w_end = float(w["end"])
            # Turn starts inside a word
            if w_start < new_start <= w_end:
                new_start = min(new_start, max(0.0, w_start - 0.05))

        # Clamp strictly to safe bounds
        new_start = max(safe_min, new_start)
        new_end = min(safe_max, new_end)
        if new_start >= new_end:
            # Revert to turn boundaries if clamping caused inversion
            new_start = turn.start_s
            new_end = turn.end_s

        new_start_s = round(new_start, 4)
        new_end_s = round(new_end, 4)

        # Collect words for turn transcript
        raw_words_in_turn = [
            w
            for w in words
            if (float(w["start"]) >= new_start_s - 0.15 and float(w["end"]) <= new_end_s + 0.15)
        ]
        words_in_turn = [
            str(w.get("text", "")).strip()
            for w in raw_words_in_turn
            if w.get("text")
        ]
        transcript = " ".join([wt for wt in words_in_turn if wt]) if words_in_turn else None

        delta_start = round((new_start_s - raw_start) * 1000.0, 1)
        delta_end = round((new_end_s - orig_end) * 1000.0, 1)
        tail_rescued = new_end_s > orig_end

        # Only label with word-locked policy if words actually matched or adjusted boundaries
        has_lock = bool(words_in_turn and (new_start_s != turn.start_s or new_end_s != turn.end_s or transcript))
        applied_policy = policy if has_lock else getattr(turn, "_boundary_policy", "standard")

        refined_turn = SpeakerTurn(
            speaker_id=turn.speaker_id,
            start_s=new_start_s,
            end_s=new_end_s,
            confidence=turn.confidence,
        )
        refined_turn._original_start_s = orig_start
        refined_turn._original_end_s = orig_end
        refined_turn._raw_start_s = raw_start
        refined_turn._raw_end_s = raw_end
        refined_turn._delta_start_ms = delta_start
        refined_turn._delta_end_ms = delta_end
        refined_turn._boundary_policy = applied_policy
        refined_turn._tail_rescued = tail_rescued
        refined_turn._transcript = transcript
        refined_turn._words = raw_words_in_turn
        if hasattr(turn, "_consensus_start_s"):
            refined_turn._consensus_start_s = turn._consensus_start_s
        if hasattr(turn, "_consensus_end_s"):
            refined_turn._consensus_end_s = turn._consensus_end_s

        aligned_turns.append(refined_turn)
        audits.append({
            "raw_start_s": raw_start,
            "raw_end_s": raw_end,
            "original_start_s": orig_start,
            "original_end_s": orig_end,
            "start_s": new_start_s,
            "end_s": new_end_s,
            "delta_start_ms": delta_start,
            "delta_end_ms": delta_end,
            "policy": applied_policy,
            "tail_rescued": tail_rescued,
            "transcript": transcript,
        })

    return aligned_turns, audits


def _extract_words_from_asr_result(res_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract flat list of word dictionaries from Whisper transcription output."""
    extracted = []
    for seg in res_obj.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("text") or w.get("word") or ""
            start_val = w.get("start")
            end_val = w.get("end")
            conf = w.get("confidence", 1.0)
            if text and start_val is not None and end_val is not None:
                extracted.append({
                    "text": str(text).strip(),
                    "start": float(start_val),
                    "end": float(end_val),
                    "confidence": float(conf),
                })
    return extracted


def _transcribe_words_with_whisper(
    audio: Audio,
    *,
    model_name: str = "vinai/PhoWhisper-small",
    language: str = "vi",
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Transcribe audio using whisper-timestamped (or openai-whisper) and return word timestamps.

    Supports CPU inference and automatically frees VRAM with CPU fallback upon CUDA OOM.
    """
    import torch

    device_str = "cpu"
    if device and device != "cpu" and device != "same":
        if device.startswith("cuda:"):
            device_str = device if torch.cuda.is_available() else "cpu"
        elif device in {"auto", "cuda"}:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"

    try:
        import whisper_timestamped as whisperts
    except ImportError:
        whisperts = None

    whisper_mod = None
    if whisperts is None:
        try:
            import whisper as whisper_mod
        except ImportError:
            raise RuntimeError(
                "Package 'whisper-timestamped' (or 'openai-whisper') is required for Whisper alignment. "
                "Please run: uv pip install whisper-timestamped"
            )

    model = None
    words: list[dict[str, Any]] = []

    try:
        _ensure_torch_hub_trusted()
        logger.info(
            "Loading Whisper alignment model '%s' on %s (lang=%s)...",
            model_name,
            device_str,
            language,
        )
        if whisperts is not None:
            model = whisperts.load_model(model_name, device=device_str)
            transcribe_opts = {
                "language": language or "vi",
                "vad": True,
                "detect_disfluencies": True,
            }
            try:
                res = whisperts.transcribe(model, str(audio.path), **transcribe_opts)
            except Exception as vad_exc:
                err_msg = str(vad_exc).lower()
                if "silero" in err_msg or "vad" in err_msg or "untrusted" in err_msg:
                    logger.warning(
                        "Silero VAD pre-segmentation failed (%s). Retrying Whisper alignment with vad=False...",
                        vad_exc,
                    )
                    transcribe_opts["vad"] = False
                    res = whisperts.transcribe(model, str(audio.path), **transcribe_opts)
                else:
                    raise
        else:
            model = whisper_mod.load_model(model_name, device=device_str)
            res = model.transcribe(str(audio.path), language=language or "vi", word_timestamps=True)

        words = _extract_words_from_asr_result(res)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if ("out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)) and device_str != "cpu":
            logger.warning(
                "CUDA OOM on device %s during Whisper alignment (%s). Freeing GPU VRAM and retrying on CPU.",
                device_str,
                exc,
            )
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            device_str = "cpu"
            logger.info("Executing Whisper alignment on CPU fallback...")
            if whisperts is not None:
                cpu_model = whisperts.load_model(model_name, device="cpu")
                try:
                    res = whisperts.transcribe(
                        cpu_model,
                        str(audio.path),
                        language=language or "vi",
                        vad=True,
                        detect_disfluencies=True,
                    )
                except Exception as vad_exc:
                    err_msg = str(vad_exc).lower()
                    if "silero" in err_msg or "vad" in err_msg or "untrusted" in err_msg:
                        logger.warning(
                            "Silero VAD pre-segmentation failed on CPU fallback (%s). Retrying with vad=False...",
                            vad_exc,
                        )
                        res = whisperts.transcribe(
                            cpu_model,
                            str(audio.path),
                            language=language or "vi",
                            vad=False,
                            detect_disfluencies=True,
                        )
                    else:
                        raise
                del cpu_model
            else:
                cpu_model = whisper_mod.load_model(model_name, device="cpu")
                res = cpu_model.transcribe(str(audio.path), language=language or "vi", word_timestamps=True)
                del cpu_model

            words = _extract_words_from_asr_result(res)
        else:
            raise

    finally:
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return words


def _run_whisper_timestamped_alignment(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    model_name: str = "vinai/PhoWhisper-small",
    language: str = "vi",
    device: str = "cpu",
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Align turns using whisper-timestamped with Vietnamese or multilingual models."""
    words = _transcribe_words_with_whisper(
        audio,
        model_name=model_name,
        language=language,
        device=device,
    )
    return _lock_turns_with_words(
        turns,
        words,
        policy=f"whisper_lock_{Path(model_name).name}",
        audio_duration_s=audio.duration_s,
        competitor_intervals_by_speaker=competitor_intervals_by_speaker,
    )


def _run_remote_whisper_alignment(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    endpoint: str,
    language: str = "vi",
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Query a remote OpenAI-compatible Whisper transcription server for word timestamps."""
    import json
    import urllib.parse
    import urllib.request
    import uuid

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    def add_field(name: str, value: str):
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    add_field("response_format", "verbose_json")
    add_field("timestamp_granularities[]", "word")
    if language:
        add_field("language", language)

    with open(str(audio.path), "rb") as f:
        file_bytes = f.read()

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{Path(audio.path).name}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    words: list[dict[str, Any]] = []
    for w in data.get("words", []):
        text = w.get("word") or w.get("text") or ""
        s = w.get("start")
        e = w.get("end")
        if text and s is not None and e is not None:
            words.append({
                "text": str(text).strip(),
                "start": float(s),
                "end": float(e),
                "confidence": float(w.get("confidence", 1.0)),
            })

    return _lock_turns_with_words(
        turns,
        words,
        policy="remote_whisper_lock",
        audio_duration_s=audio.duration_s,
        competitor_intervals_by_speaker=competitor_intervals_by_speaker,
    )


def _run_mms_fa_alignment(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    device: str = "cpu",
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Run PyTorch MMS forced alignment with CPU support and CUDA OOM fallback."""
    import torch
    import torchaudio

    device_str = "cpu"
    if device and device != "cpu" and device != "same":
        if device.startswith("cuda:"):
            device_str = device if torch.cuda.is_available() else "cpu"
        elif device in {"auto", "cuda"}:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    bundle = torchaudio.pipelines.MMS_FA
    model = None

    audio_data, sr = sf.read(str(audio.path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, bundle.sample_rate)
        waveform = resampler(waveform)
        sr = bundle.sample_rate

    emission = None
    try:
        model = bundle.get_model().to(device_str)
        model.eval()
        with torch.inference_mode():
            emission, _ = model(waveform.to(device_str))
            emission = torch.log_softmax(emission, dim=-1)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if ("out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)) and device_str != "cpu":
            logger.warning("CUDA OOM during MMS-FA on %s (%s). Falling back to CPU.", device_str, exc)
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            device_str = "cpu"
            model = bundle.get_model().to("cpu")
            model.eval()
            with torch.inference_mode():
                emission, _ = model(waveform.to("cpu"))
                emission = torch.log_softmax(emission, dim=-1)
        else:
            raise
    finally:
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    aligned_turns: list[SpeakerTurn] = []
    audits: list[dict[str, Any]] = []

    num_frames = emission.shape[1]
    total_duration = waveform.shape[1] / sr
    frame_to_sec = total_duration / num_frames

    blank_prob = torch.exp(emission[0, :, 0]).cpu().numpy()
    speech_prob = 1.0 - blank_prob

    n = len(turns)
    sorted_order = sorted(range(n), key=lambda idx: (turns[idx].start_s, turns[idx].end_s))
    safe_bounds: dict[int, tuple[float, float]] = {}
    for rank, orig_idx in enumerate(sorted_order):
        t = turns[orig_idx]
        prev_t = turns[sorted_order[rank - 1]] if rank > 0 else None
        next_t = turns[sorted_order[rank + 1]] if rank < n - 1 else None

        min_s = prev_t.end_s if prev_t is not None else 0.0
        max_e = next_t.start_s if next_t is not None else total_duration
        max_e = min(max_e, total_duration)
        min_s = max(0.0, min_s)

        if hasattr(t, "_consensus_start_s"):
            min_s = max(min_s, t._consensus_start_s)
        if hasattr(t, "_consensus_end_s"):
            max_e = min(max_e, t._consensus_end_s)

        # Never expand into competitor speech
        if competitor_intervals_by_speaker:
            comp_ivs = competitor_intervals_by_speaker.get(t.speaker_id, [])
            for c_s, c_e in comp_ivs:
                if c_e <= t.start_s + 1e-4:
                    min_s = max(min_s, c_e)
                if c_s >= t.end_s - 1e-4:
                    max_e = min(max_e, c_s)

        safe_bounds[orig_idx] = (min_s, max_e)

    for i, turn in enumerate(turns):
        orig_start = getattr(turn, "_original_start_s", turn.start_s)
        orig_end = getattr(turn, "_original_end_s", turn.end_s)
        raw_start = getattr(turn, "_raw_start_s", turn.start_s)
        raw_end = getattr(turn, "_raw_end_s", turn.end_s)

        safe_min, safe_max = safe_bounds.get(i, (0.0, total_duration))

        start_f = max(0, int(round(turn.start_s / frame_to_sec)))
        end_f = min(num_frames - 1, int(round(turn.end_s / frame_to_sec)))

        search_f = int(round(0.30 / frame_to_sec))
        new_end_f = end_f

        if speech_prob[end_f] > 0.3:
            for f in range(end_f, min(num_frames - 1, end_f + search_f)):
                if speech_prob[f] < 0.2:
                    new_end_f = f
                    break

        new_end_s = round(new_end_f * frame_to_sec, 4)
        new_start_s = round(start_f * frame_to_sec, 4)

        new_start_s = max(safe_min, new_start_s)
        new_end_s = min(safe_max, new_end_s)
        if new_start_s >= new_end_s:
            new_start_s = turn.start_s
            new_end_s = turn.end_s

        delta_start = round((new_start_s - raw_start) * 1000.0, 1)
        delta_end = round((new_end_s - orig_end) * 1000.0, 1)
        tail_rescued = new_end_s > orig_end

        has_lock = (new_end_f != end_f or tail_rescued)
        applied_policy = "syllable_word_lock" if has_lock else getattr(turn, "_boundary_policy", "standard")

        refined_turn = SpeakerTurn(
            speaker_id=turn.speaker_id,
            start_s=new_start_s,
            end_s=new_end_s,
            confidence=turn.confidence,
        )
        refined_turn._original_start_s = orig_start
        refined_turn._original_end_s = orig_end
        refined_turn._raw_start_s = raw_start
        refined_turn._raw_end_s = raw_end
        refined_turn._delta_start_ms = delta_start
        refined_turn._delta_end_ms = delta_end
        refined_turn._boundary_policy = applied_policy
        refined_turn._tail_rescued = tail_rescued
        if hasattr(turn, "_consensus_start_s"):
            refined_turn._consensus_start_s = turn._consensus_start_s
        if hasattr(turn, "_consensus_end_s"):
            refined_turn._consensus_end_s = turn._consensus_end_s

        aligned_turns.append(refined_turn)
        audits.append({
            "raw_start_s": raw_start,
            "raw_end_s": raw_end,
            "original_start_s": orig_start,
            "original_end_s": orig_end,
            "start_s": new_start_s,
            "end_s": new_end_s,
            "delta_start_ms": delta_start,
            "delta_end_ms": delta_end,
            "policy": applied_policy,
            "tail_rescued": tail_rescued,
        })

    return aligned_turns, audits


def align_and_lock_syllable_boundaries(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    aligner_engine: str = "whisper_timestamped",
    aligner_model: str = "vinai/PhoWhisper-small",
    aligner_language: str = "vi",
    aligner_endpoint: str | None = None,
    aligner_device: str = "cpu",
    token: str | None = None,
    competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Lock boundaries to complete syllable/word timestamps using Whisper or MMS-FA.

    Strictly forbids slicing through the interior of an active syllable.
    Supports CPU as a real inference device to eliminate GPU VRAM exhaustion.
    """
    if not turns:
        return [], []

    eng = (aligner_engine or "whisper_timestamped").lower().strip()
    try:
        if eng in {"whisper_timestamped", "whisper", "whisperts"}:
            return _run_whisper_timestamped_alignment(
                audio,
                turns,
                model_name=aligner_model or "vinai/PhoWhisper-small",
                language=aligner_language or "vi",
                device=aligner_device or "cpu",
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
        elif eng in {"remote_whisper", "remote_asr"}:
            if not aligner_endpoint:
                raise ValueError("Remote Whisper endpoint URL required for remote_whisper engine.")
            return _run_remote_whisper_alignment(
                audio,
                turns,
                endpoint=aligner_endpoint,
                language=aligner_language or "vi",
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
        elif eng in {"mms_fa", "mms"}:
            return _run_mms_fa_alignment(
                audio,
                turns,
                device=aligner_device or "cpu",
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
        else:
            raise ValueError(f"Unsupported aligner engine: {aligner_engine}")

    except Exception as exc:
        logger.warning("Forced alignment syllable lock failed (%s), keeping candidate turns: %s", aligner_engine, exc)
        fallback_audits = [
            {
                "raw_start_s": getattr(t, "_raw_start_s", t.start_s),
                "raw_end_s": getattr(t, "_raw_end_s", t.end_s),
                "original_start_s": getattr(t, "_original_start_s", t.start_s),
                "original_end_s": getattr(t, "_original_end_s", t.end_s),
                "start_s": t.start_s,
                "end_s": t.end_s,
                "delta_start_ms": getattr(t, "_delta_start_ms", 0.0),
                "delta_end_ms": getattr(t, "_delta_end_ms", 0.0),
                "policy": getattr(t, "_boundary_policy", "standard"),
                "tail_rescued": getattr(t, "_tail_rescued", False),
                "error": str(exc),
            }
            for t in turns
        ]
        return list(turns), fallback_audits


def _copy_turn_meta(src: SpeakerTurn, dst: SpeakerTurn, policy: str = "smart_segmentation") -> None:
    """Copy provenance metadata attributes from src turn to dst turn."""
    for attr in (
        "_original_start_s",
        "_original_end_s",
        "_raw_start_s",
        "_raw_end_s",
        "_delta_start_ms",
        "_delta_end_ms",
        "_tail_rescued",
        "_consensus_start_s",
        "_consensus_end_s",
    ):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    dst._boundary_policy = policy


def smart_segment_speaker_turns(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    max_duration_s: float = DEFAULT_TARGET_MAX_DURATION_S,
    min_duration_s: float = DEFAULT_TARGET_MIN_DURATION_S,
    min_pause_s: float = DEFAULT_MIN_SPLIT_PAUSE_S,
    words: list[dict[str, Any]] | None = None,
    frame_len_ms: float = DEFAULT_ENERGY_FRAME_LEN_MS,
    hop_len_ms: float = DEFAULT_ENERGY_HOP_LEN_MS,
    search_window_s: float = DEFAULT_ENERGY_SEARCH_WINDOW_S,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    """Segment long speaker turns into TTS-optimal sentence-length chunks.

    Uses ASR word timestamps, terminal/clause punctuation, and natural breathing
    pauses to avoid cutting mid-syllable or mid-word. Snaps the final cut point
    to the nearest micro-acoustic energy valley and zero-crossing to prevent clicks.

    Args:
        audio: Target Audio instance.
        turns: Candidate speaker turns to segment.
        max_duration_s: Target maximum turn length in seconds (default 10.0s).
        min_duration_s: Target minimum turn length in seconds (default 3.0s).
        min_pause_s: Minimum silence gap between words to consider a split (default 0.20s).
        words: Optional word timestamps from Whisper ASR.
        frame_len_ms: RMS frame length for acoustic valley snapping.
        hop_len_ms: RMS hop step for acoustic valley snapping.
        search_window_s: Window radius for acoustic valley snapping.

    Returns:
        (segmented_turns, segmentation_audits)
    """
    if not turns:
        return [], []

    any_long = any(t.duration_s > max_duration_s for t in turns)
    if not any_long:
        return list(turns), []

    waveform, sr = sf.read(str(audio.path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    frame_samples = max(1, int(round(frame_len_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_len_ms * sr / 1000.0)))
    search_samples = int(round(search_window_s * sr))

    if words is None:
        words = []
        for t in turns:
            if hasattr(t, "_words") and t._words:
                words.extend(t._words)

    TERMINAL_PUNCT = {".", "!", "?", "...", "…"}
    CLAUSE_PUNCT = {",", ";", ":", "—", "-", "–"}

    segmented_turns: list[SpeakerTurn] = []
    audits: list[dict[str, Any]] = []

    for turn in turns:
        if turn.duration_s <= max_duration_s:
            segmented_turns.append(turn)
            continue

        # Gather words inside this turn
        t_words = [
            w for w in (words or [])
            if float(w["start"]) >= turn.start_s - 0.15 and float(w["end"]) <= turn.end_s + 0.15
        ]
        t_words.sort(key=lambda w: (float(w["start"]), float(w["end"])))

        curr_start = turn.start_s
        turn_end = turn.end_s

        while (turn_end - curr_start) > max_duration_s:
            remaining = turn_end - curr_start

            # If remaining speech is within 2 * max_duration_s, balance the two halves
            if remaining <= 2.0 * max_duration_s:
                ideal_split = curr_start + (remaining / 2.0)
                search_min = max(curr_start + min_duration_s, ideal_split - 2.0)
                search_max = min(turn_end - min_duration_s, ideal_split + 2.0)
                if search_max <= search_min:
                    search_min = curr_start + min_duration_s
                    search_max = curr_start + max_duration_s
            else:
                search_min = curr_start + min_duration_s
                search_max = curr_start + max_duration_s

            search_min = min(search_min, turn_end - 0.5)
            search_max = min(search_max, turn_end)

            best_cut: float | None = None
            best_score = -1.0
            split_method = "acoustic_rms_valley"

            # Search word boundaries
            if len(t_words) >= 2:
                for idx_w in range(len(t_words) - 1):
                    w1 = t_words[idx_w]
                    w2 = t_words[idx_w + 1]
                    w1_end = float(w1["end"])
                    w2_start = float(w2["start"])

                    cand_time = (w1_end + w2_start) / 2.0 if w2_start > w1_end else w1_end
                    if search_min <= cand_time <= search_max:
                        text1 = str(w1.get("text", "")).strip()
                        pause = max(0.0, w2_start - w1_end)

                        has_term = any(text1.endswith(p) for p in TERMINAL_PUNCT)
                        has_clause = any(text1.endswith(p) for p in CLAUSE_PUNCT)

                        if has_term and pause >= 0.12:
                            score = 100.0 + min(pause, 1.0) * 10.0
                            method = "terminal_punctuation_pause"
                        elif has_term:
                            score = 80.0 + min(pause, 1.0) * 10.0
                            method = "terminal_punctuation"
                        elif has_clause and pause >= 0.12:
                            score = 60.0 + min(pause, 1.0) * 10.0
                            method = "clause_punctuation_pause"
                        elif has_clause:
                            score = 40.0 + min(pause, 1.0) * 10.0
                            method = "clause_punctuation"
                        elif pause >= min_pause_s:
                            score = 25.0 + min(pause, 1.0) * 10.0
                            method = "inter_word_pause"
                        else:
                            midpoint = (search_min + search_max) / 2.0
                            dist_norm = 1.0 - (abs(cand_time - midpoint) / max(0.1, (search_max - search_min) / 2.0))
                            score = 10.0 + max(0.0, dist_norm) * 5.0
                            method = "inter_word_gap"

                        if score > best_score:
                            best_score = score
                            best_cut = cand_time
                            split_method = method

            # Fallback to acoustic energy valley if no ASR word candidate in window
            if best_cut is None:
                mid_samp = int(round(((search_min + search_max) / 2.0) * sr))
                half_win_samp = max(frame_samples, int(round((search_max - search_min) / 2.0 * sr)))
                valley_samp = _find_local_valley(
                    waveform,
                    mid_samp,
                    search_samples=half_win_samp,
                    frame_samples=frame_samples,
                    hop_samples=hop_samples,
                )
                best_cut = valley_samp / sr
                split_method = "acoustic_rms_valley"
            else:
                # Snap the chosen word gap to local acoustic zero-crossing
                cut_samp = int(round(best_cut * sr))
                snap_radius = min(search_samples, int(round(0.06 * sr)))
                valley_samp = _find_local_valley(
                    waveform,
                    cut_samp,
                    search_samples=snap_radius,
                    frame_samples=frame_samples,
                    hop_samples=hop_samples,
                )
                best_cut = valley_samp / sr

            best_cut = round(float(best_cut), 4)
            if best_cut <= curr_start + 0.5:
                best_cut = round(curr_start + min_duration_s, 4)

            # Create child turn
            child = SpeakerTurn(
                speaker_id=turn.speaker_id,
                start_s=round(curr_start, 4),
                end_s=best_cut,
                confidence=turn.confidence,
            )
            _copy_turn_meta(turn, child, policy="smart_segmentation")
            child_words = [
                w for w in t_words
                if float(w["start"]) >= child.start_s - 0.1 and float(w["end"]) <= child.end_s + 0.1
            ]
            child._words = child_words
            child._transcript = " ".join([str(w.get("text", "")).strip() for w in child_words if w.get("text")]) or getattr(turn, "_transcript", None)

            segmented_turns.append(child)
            audits.append({
                "action": "split",
                "method": split_method,
                "speaker_id": turn.speaker_id,
                "parent_start_s": turn.start_s,
                "parent_end_s": turn.end_s,
                "child_start_s": child.start_s,
                "child_end_s": child.end_s,
                "child_duration_s": round(child.duration_s, 3),
                "transcript": child._transcript,
            })

            curr_start = best_cut

        # Trailing segment
        if turn_end > curr_start:
            tail_dur = turn_end - curr_start
            if tail_dur >= min(1.0, min_duration_s):
                child = SpeakerTurn(
                    speaker_id=turn.speaker_id,
                    start_s=round(curr_start, 4),
                    end_s=round(turn_end, 4),
                    confidence=turn.confidence,
                )
                _copy_turn_meta(turn, child, policy="smart_segmentation")
                child_words = [
                    w for w in t_words
                    if float(w["start"]) >= child.start_s - 0.1 and float(w["end"]) <= child.end_s + 0.1
                ]
                child._words = child_words
                child._transcript = " ".join([str(w.get("text", "")).strip() for w in child_words if w.get("text")]) or getattr(turn, "_transcript", None)
                segmented_turns.append(child)
                audits.append({
                    "action": "split_tail",
                    "method": "tail_remainder",
                    "speaker_id": turn.speaker_id,
                    "parent_start_s": turn.start_s,
                    "parent_end_s": turn.end_s,
                    "child_start_s": child.start_s,
                    "child_end_s": child.end_s,
                    "child_duration_s": round(child.duration_s, 3),
                    "transcript": child._transcript,
                })
            elif segmented_turns:
                # Merge tiny micro-tail (<1s) into preceding child turn
                last_child = segmented_turns[-1]
                updated_child = replace(last_child, end_s=round(turn_end, 4))
                _copy_turn_meta(last_child, updated_child, policy="smart_segmentation")
                updated_words = [
                    w for w in t_words
                    if float(w["start"]) >= updated_child.start_s - 0.1 and float(w["end"]) <= turn_end + 0.1
                ]
                updated_child._words = updated_words
                updated_child._transcript = " ".join([str(w.get("text", "")).strip() for w in updated_words if w.get("text")]) or getattr(last_child, "_transcript", None)
                segmented_turns[-1] = updated_child

    return segmented_turns, audits


def filter_by_embedding_homogeneity(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    *,
    window_s: float = DEFAULT_HOMOGENEITY_WINDOW_S,
    hop_s: float = DEFAULT_HOMOGENEITY_HOP_S,
    min_similarity: float = DEFAULT_MIN_HOMOGENEITY_SIMILARITY,
    device: str = "auto",
    token: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[SpeakerTurn], list[tuple[SpeakerTurn, bool, float | None, str]]]:
    """Reject candidate turns whose sliding-window embeddings show foreign voice intrusion.

    Returns:
        (passed_turns, audit_records_for_homogeneity)
    """
    if not turns:
        return [], []

    from src.diarization.SpeakerVerifier import (
        DEFAULT_EMBEDDING_MODEL_ID,
        SpeakerVerifier,
    )

    verifier = SpeakerVerifier(
        model_id=DEFAULT_EMBEDDING_MODEL_ID,
        device=device,
        token=token,
    )

    passed: list[SpeakerTurn] = []
    audits: list[tuple[SpeakerTurn, bool, float | None, str]] = []

    with verifier:
        for turn in turns:
            if cancel_check and cancel_check():
                raise InterruptedError("Pipeline execution cancelled during homogeneity verification")

            dur = turn.end_s - turn.start_s
            if dur < window_s:
                # Sub-second turn: keep with warning or accept
                passed.append(turn)
                audits.append((turn, True, 1.0, "Turn shorter than homogeneity window"))
                continue

            windows = SpeakerVerifier._candidate_windows(
                turn.start_s,
                turn.end_s,
                window_duration_s=window_s,
                window_hop_s=hop_s,
            )
            if len(windows) < 2:
                passed.append(turn)
                audits.append((turn, True, 1.0, "Single window extracted"))
                continue

            vectors: list[np.ndarray] = []
            errors: list[str] = []
            for win_start, win_end in windows:
                try:
                    emb = verifier.extract_embedding(audio, start_s=win_start, end_s=win_end)
                    v = np.asarray(emb, dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(v))
                    if norm > 0 and np.isfinite(v).all():
                        vectors.append(v / norm)
                except Exception as exc:
                    errors.append(str(exc))

            if len(vectors) < 2:
                if errors:
                    turn._min_similarity = 0.0
                    audits.append(
                        (turn, False, 0.0, f"Homogeneity embedding extraction failed: {errors[0]}")
                    )
                else:
                    passed.append(turn)
                    audits.append((turn, True, 1.0, "Single window extracted"))
                continue

            centroid = np.mean(vectors, axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 0:
                centroid /= centroid_norm

            sims = [float(np.dot(vec, centroid)) for vec in vectors]
            min_sim = min(sims)

            turn._min_similarity = min_sim
            if min_sim >= min_similarity:
                passed.append(turn)
                audits.append(
                    (turn, True, min_sim, f"Homogeneity passed (min_cos={min_sim:.3f})")
                )
            else:
                audits.append(
                    (
                        turn,
                        False,
                        min_sim,
                        f"Foreign voice / timbre drift detected (min_cos={min_sim:.3f} < {min_similarity:.2f})",
                    )
                )

    return passed, audits


def filter_by_foundation_models(
    audio: Audio,
    turns: Sequence[SpeakerTurn],
    config: ZeroContaminationConfig,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[SpeakerTurn], list[tuple[SpeakerTurn, bool, str, dict[str, Any]]]]:
    """Audit surviving turns with a direct-audio LLM or VibeVoice-ASR.

    Returns:
        (passed_turns, audit_records_for_foundation_model)
    """
    if not turns or (not config.enable_gemma and not config.enable_vibevoice):
        return list(turns), []

    passed: list[SpeakerTurn] = []
    audits: list[tuple[SpeakerTurn, bool, str, dict[str, Any]]] = []

    waveform, sr = sf.read(str(audio.path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # 1. Initialize the selected direct-audio verifier.
    gemma_verifier = None
    if config.enable_gemma:
        from src.diarization.OverlapVerifier import (
            DEFAULT_GEMMA4_MODEL_ID,
            OVERLAP_PROMPT,
            OverlapVerifierError,
            create_overlap_verifier,
        )

        verifier_config: dict[str, Any] = {
            "backend": config.gemma_backend,
            "model": config.gemma_model or (
                DEFAULT_GEMMA4_MODEL_ID
                if config.gemma_backend == "gemma4"
                else None
            ),
            "prompt": config.gemma_prompt or OVERLAP_PROMPT,
            "api_key": config.gemma_api_key,
            "timeout_s": config.gemma_timeout_s,
            "max_output_tokens": config.gemma_max_output_tokens,
        }
        if config.gemma_backend == "gemma4":
            verifier_config["endpoint"] = config.gemma_endpoint
        verifier_config = {
            key: value for key, value in verifier_config.items() if value is not None
        }
        gemma_verifier = create_overlap_verifier(verifier_config)
        readiness = gemma_verifier.check_ready()
        if not readiness.get("ready"):
            raise OverlapVerifierError(
                str(readiness.get("message") or "Direct-audio verifier is not ready"),
                readiness=True,
            )

    # 2. Initialize VibeVoice if requested (Dedicated Secondary GPU or Remote Host)
    vibevoice_verifier = None
    if config.enable_vibevoice and not config.vibevoice_endpoint:
        from src.diarization.VibeVoicePurityWorkerVerifier import (
            VibeVoicePurityWorkerVerifier,
        )

        vv_device = config.vibevoice_device if (config.vibevoice_device and config.vibevoice_device != "same") else config.device
        vibevoice_verifier = VibeVoicePurityWorkerVerifier(
            model_id=config.vibevoice_model_id,
            device=vv_device,
            min_secondary_speech_s=config.max_secondary_speech_s,
        )
        vibevoice_verifier.load()

    try:
        with tempfile.TemporaryDirectory(prefix="foundation-audit-") as tmpdir:
            total = len(turns)
            records: list[dict[str, Any]] = []

            for idx, turn in enumerate(turns):
                if cancel_check and cancel_check():
                    raise InterruptedError(
                        "Pipeline execution cancelled during foundation model verification"
                    )

                if progress_callback:
                    p = 0.70 + 0.14 * (idx / max(1, total))
                    progress_callback(
                        p,
                        f"Foundation audit (VibeVoice): turn {idx+1}/{total} ({turn.duration_s:.1f}s)...",
                    )

                start_samp = max(0, int(round(turn.start_s * sr)))
                end_samp = min(len(waveform), int(round(turn.end_s * sr)))
                clip = waveform[start_samp:end_samp]
                clip_path = Path(tmpdir) / f"turn_{idx:04d}.wav"
                sf.write(clip_path, clip, sr, subtype="PCM_16")
                clip_audio = Audio.from_file(clip_path)

                audit_meta: dict[str, Any] = {}
                is_pure = True
                rejection_reason = ""

                # 1. Fast speaker gate: VibeVoice-ASR multi-speaker check (Local Worker or Remote Endpoint)
                if config.enable_vibevoice:
                    if config.vibevoice_endpoint:
                        try:
                            import base64
                            import json
                            import urllib.request

                            with open(clip_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                            req_data = json.dumps({
                                "audio_base64": b64,
                                "model_id": config.vibevoice_model_id,
                                "min_secondary_speech_s": config.max_secondary_speech_s,
                            }).encode("utf-8")
                            req = urllib.request.Request(
                                config.vibevoice_endpoint,
                                data=req_data,
                                headers={"Content-Type": "application/json"},
                            )
                            with urllib.request.urlopen(req, timeout=120) as resp:
                                resp_data = json.loads(resp.read().decode("utf-8"))
                                if not isinstance(resp_data, dict) or "num_speakers" not in resp_data:
                                    raise ValueError(f"Malformed VibeVoice response: {resp_data}")
                                num_spk = resp_data.get("num_speakers", 1)
                                sec_dur = resp_data.get("secondary_speech_s", 0.0)
                                audit_meta["vibevoice"] = resp_data
                                if num_spk > 1 or sec_dur > config.max_secondary_speech_s:
                                    is_pure = False
                                    rejection_reason = (
                                        f"Remote VibeVoice detected {num_spk} speakers "
                                        f"({sec_dur:.2f}s secondary speech)"
                                    )
                        except Exception as exc:
                            logger.warning("Remote VibeVoice check failed on turn %s: %s", idx, exc)
                            audit_meta["vibevoice_error"] = str(exc)
                            is_pure = False
                            rejection_reason = f"Remote VibeVoice verifier failed closed: {exc}"
                    elif vibevoice_verifier is not None:
                        try:
                            vv_res = vibevoice_verifier.verify(clip_audio)
                            audit_meta["vibevoice"] = vv_res.to_dict()
                            if vv_res.num_speakers > 1 or vv_res.secondary_speech_s > config.max_secondary_speech_s:
                                is_pure = False
                                rejection_reason = (
                                    f"VibeVoice detected {vv_res.num_speakers} speakers "
                                    f"({vv_res.secondary_speech_s:.2f}s secondary speech)"
                                )
                        except Exception as exc:
                            logger.warning("VibeVoice check failed on turn %s: %s", idx, exc)
                            audit_meta["vibevoice_error"] = str(exc)
                            is_pure = False
                            rejection_reason = f"VibeVoice verifier failed closed: {exc}"

                records.append({
                    "idx": idx,
                    "turn": turn,
                    "clip_audio": clip_audio,
                    "audit_meta": audit_meta,
                    "is_pure": is_pure,
                    "rejection_reason": rejection_reason,
                })

            # 2. Semantic & completeness auditor: Direct-audio verifier (Gemini / Gemma-4)
            # Only invoked if turn passed VibeVoice speaker purity check (saves API tokens and compute).
            if gemma_verifier is not None:
                gemma_candidates = [r for r in records if r["is_pure"]]
                concurrency = getattr(
                    gemma_verifier,
                    "concurrency",
                    10 if "gemini" in config.gemma_backend.lower() else 1,
                )
                concurrency = max(1, int(concurrency))
                verifier_label = getattr(
                    gemma_verifier, "model", config.gemma_backend
                )
                gemma_total = len(gemma_candidates)
                gemma_done = 0

                def verify_single_gemma(rec: dict[str, Any]) -> None:
                    nonlocal gemma_done
                    if cancel_check and cancel_check():
                        raise InterruptedError(
                            "Pipeline execution cancelled during foundation model verification"
                        )
                    try:
                        gemma_res = gemma_verifier.verify(rec["clip_audio"])
                        rec["audit_meta"]["gemma"] = gemma_res
                        if gemma_res.get("decision") != "pass":
                            rec["is_pure"] = False
                            rec["rejection_reason"] = (
                                f"{verifier_label} {gemma_res.get('decision')}: "
                                f"{gemma_res.get('reason', '')}"
                            )
                    except Exception as exc:
                        logger.warning(
                            "Direct-audio check failed on turn %s: %s", rec["idx"], exc
                        )
                        rec["audit_meta"]["gemma_error"] = str(exc)
                        rec["is_pure"] = False
                        rec["rejection_reason"] = f"Direct-audio verifier failed closed: {exc}"
                    gemma_done += 1
                    if progress_callback:
                        p = 0.84 + 0.14 * (gemma_done / max(1, gemma_total))
                        progress_callback(
                            p,
                            f"Foundation audit ({verifier_label}): turn {gemma_done}/{gemma_total}...",
                        )

                if concurrency > 1 and len(gemma_candidates) > 1:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(
                        max_workers=min(concurrency, len(gemma_candidates))
                    ) as executor:
                        list(executor.map(verify_single_gemma, gemma_candidates))
                else:
                    for rec in gemma_candidates:
                        verify_single_gemma(rec)

            for rec in records:
                turn = rec["turn"]
                audit_meta = rec["audit_meta"]
                if rec["is_pure"]:
                    if "gemma" in audit_meta:
                        turn._gemma_decision = audit_meta["gemma"]
                    if "vibevoice" in audit_meta:
                        turn._vibevoice_decision = audit_meta["vibevoice"]
                    passed.append(turn)
                    audits.append((turn, True, "Passed foundation audit", audit_meta))
                else:
                    audits.append((turn, False, rec["rejection_reason"], audit_meta))

    finally:
        if vibevoice_verifier is not None:
            vibevoice_verifier.close()

    return passed, audits


def run_zero_contamination_pipeline(
    audio: Audio,
    config: ZeroContaminationConfig,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ZeroContaminationResult:
    """Execute the full zero-contamination diarization pipeline.

    Stages:
    1. Primary diarization inference (with sensitive tripwire thresholds)
    2. Optional secondary diarization & mutual Hungarian consensus
    3. Boundary collar erosion & transition exclusion
    4. Dense sliding-window WeSpeaker embedding homogeneity filter
    5. In-loop foundation model verification (Gemma-4 and/or VibeVoice-ASR)
    """
    t0 = time.time()
    stage_logs: list[str] = []
    funnel: dict[str, Any] = {
        "audio_duration_s": round(audio.duration_s, 2),
    }

    def log_stage(msg: str) -> None:
        elapsed = time.time() - t0
        formatted = f"[{elapsed:.1f}s] {msg}"
        stage_logs.append(formatted)
        logger.info(formatted)

    log_stage("Starting Zero-Contamination Diarization Pipeline")

    # ==================== STAGE 1: Primary Diarizer ====================
    if cancel_check and cancel_check():
        raise InterruptedError("Pipeline execution cancelled before Stage 1")

    primary_dev = config.primary_device if (config.primary_device and config.primary_device != "same") else config.device
    if progress_callback:
        progress_callback(0.05, f"Running primary diarizer ({config.primary_backend} on {primary_dev})...")
    log_stage(f"Stage 1: Running primary backend '{config.primary_backend}' on device '{primary_dev}'")

    primary_result = _run_backend(
        config.primary_backend,
        audio,
        device=primary_dev,
        token=config.token,
        onset=config.target_onset,
        offset=config.target_offset,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    initial_turns = sorted(primary_result.turns, key=lambda t: t.start_s)
    funnel["initial_turns_count"] = len(initial_turns)
    funnel["initial_speech_duration_s"] = round(
        sum(t.duration_s for t in initial_turns), 2
    )
    log_stage(
        f"Primary produced {len(initial_turns)} turns "
        f"({funnel['initial_speech_duration_s']:.1f}s speech)"
    )

    current_turns = initial_turns
    audit_map: dict[str, TurnAuditRecord] = {
        f"turn_{i:04d}": TurnAuditRecord(
            turn_id=f"turn_{i:04d}",
            speaker_id=t.speaker_id,
            original_start_s=t.start_s,
            original_end_s=t.end_s,
            start_s=t.start_s,
            end_s=t.end_s,
            duration_s=t.duration_s,
            status="active",
        )
        for i, t in enumerate(initial_turns)
    }

    # ==================== STAGE 2: Dual-Engine Consensus ====================
    secondary_turns: list[SpeakerTurn] = []
    spk_map: dict[str, str] = {}
    if config.enable_consensus:
        if cancel_check and cancel_check():
            raise InterruptedError("Pipeline execution cancelled before Stage 2")

        secondary_dev = config.secondary_device if (config.secondary_device and config.secondary_device != "same") else config.device
        if progress_callback:
            progress_callback(0.25, f"Running secondary diarizer ({config.secondary_backend} on {secondary_dev})...")
        log_stage(f"Stage 2: Running secondary backend '{config.secondary_backend}' on device '{secondary_dev}' for consensus")

        secondary_result = _run_backend(
            config.secondary_backend,
            audio,
            device=secondary_dev,
            token=config.token,
            onset=config.target_onset,
            offset=config.target_offset,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        secondary_turns = sorted(secondary_result.turns, key=lambda t: t.start_s)

        if progress_callback:
            progress_callback(0.45, "Computing Hungarian mutual consensus...")
        consensus_turns, spk_map = compute_consensus_turns(
            current_turns, secondary_result.turns, audio.duration_s
        )

        funnel["consensus_turns_count"] = len(consensus_turns)
        funnel["consensus_speech_duration_s"] = round(
            sum(t.duration_s for t in consensus_turns), 2
        )
        funnel["speaker_mapping"] = spk_map
        log_stage(
            f"Consensus retained {len(consensus_turns)} turns "
            f"({funnel['consensus_speech_duration_s']:.1f}s), "
            f"discarding {funnel['initial_speech_duration_s'] - funnel['consensus_speech_duration_s']:.1f}s "
            f"of ambiguous/disputed speech"
        )
        current_turns = consensus_turns

    # Preserve raw competitor evidence across Primary and Secondary engines
    # so consensus filtering cannot mask nearby speaker handoffs from Stage 3.
    competitor_intervals_by_speaker = _extract_competitor_intervals_by_speaker(
        initial_turns,
        secondary_turns if config.enable_consensus else None,
        spk_map if config.enable_consensus else None,
    )

    # ==================== STAGE 3: Boundary & Syllable Integrity Gate ====================
    for t in current_turns:
        if not hasattr(t, "_original_start_s"):
            t._original_start_s = t.start_s
            t._original_end_s = t.end_s
            t._raw_start_s = t.start_s
            t._raw_end_s = t.end_s
            t._delta_start_ms = 0.0
            t._delta_end_ms = 0.0
            t._boundary_policy = "standard"
            t._tail_rescued = False

    boundary_audits: list[dict[str, Any]] = []

    if config.enable_collar_erosion:
        if cancel_check and cancel_check():
            raise InterruptedError("Pipeline execution cancelled before Stage 3")

        # 3a. Option A: Context-Aware Collar Erosion (Silence vs Handoff Guard)
        if config.enable_context_collar:
            if progress_callback:
                progress_callback(0.52, "Applying context-aware handoff collar guard...")
            log_stage(
                f"Stage 3a: Applying context-aware collar guard "
                f"(handoff_risk={config.handoff_risk_distance_s:.2f}s, silence_tail={config.silence_tail_buffer_s:.2f}s)"
            )
            ctx_turns, ctx_audits = apply_context_aware_collar(
                current_turns,
                collar_s=config.boundary_collar_s,
                handoff_risk_s=config.handoff_risk_distance_s,
                silence_tail_s=config.silence_tail_buffer_s,
                min_duration_s=config.min_turn_duration_s,
                transition_exclusion_s=config.transition_exclusion_s,
                audio_duration_s=audio.duration_s,
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
            current_turns = ctx_turns
            boundary_audits = ctx_audits
        else:
            if progress_callback:
                progress_callback(0.55, "Applying aggressive blunt collar erosion...")
            log_stage(
                f"Stage 3: Eroding boundaries by {config.boundary_collar_s:.2f}s "
                f"(min_dur={config.min_turn_duration_s:.2f}s, transition_excl={config.transition_exclusion_s:.2f}s)"
            )
            eroded_turns = erode_turn_boundaries(
                current_turns,
                collar_s=config.boundary_collar_s,
                min_duration_s=config.min_turn_duration_s,
                transition_exclusion_s=config.transition_exclusion_s,
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
            current_turns = eroded_turns

        # 3b. Option C: Micro-Acoustic Energy & RMS Silence Valley Snapping (Acoustic Refinement)
        if config.enable_energy_snapping:
            if cancel_check and cancel_check():
                raise InterruptedError("Pipeline execution cancelled before Stage 3b")

            if progress_callback:
                progress_callback(0.56, "Snapping boundaries to micro-energy RMS valleys...")
            log_stage(
                f"Stage 3b: Snapping boundaries to micro-energy valleys "
                f"(±{config.energy_search_window_s*1000:.1f}ms window, "
                f"frame={config.energy_frame_len_ms:.1f}ms, hop={config.energy_hop_len_ms:.1f}ms, "
                f"floor={config.energy_valley_floor_db:.0f}dB)"
            )
            snapped_turns, snap_audits = snap_boundaries_to_acoustic_valleys(
                audio,
                current_turns,
                search_window_s=config.energy_search_window_s,
                energy_floor_db=config.energy_valley_floor_db,
                frame_len_ms=config.energy_frame_len_ms,
                hop_len_ms=config.energy_hop_len_ms,
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
            current_turns = snapped_turns
            if snap_audits:
                boundary_audits = snap_audits

        # 3c. Option B: Syllable / Word Forced Alignment Lock (FINAL BOUNDARY AUTHORITY)
        if config.enable_syllable_alignment:
            if cancel_check and cancel_check():
                raise InterruptedError("Pipeline execution cancelled before Stage 3c")

            aligner_dev = config.aligner_device if (config.aligner_device and config.aligner_device != "same") else config.device
            if progress_callback:
                progress_callback(0.60, f"Running syllable & word lock ({config.aligner_engine} on {aligner_dev})...")
            log_stage(f"Stage 3c: Locking syllables with {config.aligner_engine} on {aligner_dev} (model={config.aligner_model}, lang={config.aligner_language}) [final boundary authority]")
            aligned_turns, align_audits = align_and_lock_syllable_boundaries(
                audio,
                current_turns,
                aligner_engine=config.aligner_engine,
                aligner_model=config.aligner_model,
                aligner_language=config.aligner_language,
                aligner_endpoint=config.aligner_endpoint,
                aligner_device=aligner_dev,
                token=config.token,
                competitor_intervals_by_speaker=competitor_intervals_by_speaker,
            )
            current_turns = aligned_turns
            if align_audits:
                boundary_audits = align_audits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        funnel["eroded_turns_count"] = len(current_turns)
        funnel["eroded_speech_duration_s"] = round(
            sum(t.duration_s for t in current_turns), 2
        )
        log_stage(
            f"Boundary & syllable integrity produced {len(current_turns)} pure turns "
            f"({funnel['eroded_speech_duration_s']:.1f}s speech)"
        )
    else:
        log_stage("Stage 3: Boundary & Syllable Integrity Gate disabled; emitting raw diarizer timestamps")
        funnel["eroded_turns_count"] = len(current_turns)
        funnel["eroded_speech_duration_s"] = round(
            sum(t.duration_s for t in current_turns), 2
        )

    # 3d. Option D: Intelligent ASR & Pause-Guided Turn Segmentation (TTS Sentence Sizing)
    segment_audits: list[dict[str, Any]] = []
    if config.enable_smart_segmentation:
        if cancel_check and cancel_check():
            raise InterruptedError("Pipeline execution cancelled before Stage 3d")

        if progress_callback:
            progress_callback(0.63, "Segmenting long turns into TTS-optimal sentences...")
        log_stage(
            f"Stage 3d: Running smart turn segmentation "
            f"(target_max={config.target_max_duration_s:.1f}s, target_min={config.target_min_duration_s:.1f}s, "
            f"min_pause={config.min_split_pause_s:.2f}s)"
        )

        all_words: list[dict[str, Any]] = []
        for t in current_turns:
            if hasattr(t, "_words") and t._words:
                all_words.extend(t._words)

        any_long = any(t.duration_s > config.target_max_duration_s for t in current_turns)
        if not all_words and any_long:
            try:
                aligner_dev = config.aligner_device if (config.aligner_device and config.aligner_device != "same") else "cpu"
                log_stage(f"Stage 3d: Transcribing with {config.aligner_engine} ({config.aligner_model}) on {aligner_dev} to guide sentence cuts...")
                all_words = _transcribe_words_with_whisper(
                    audio,
                    model_name=config.aligner_model,
                    language=config.aligner_language,
                    device=aligner_dev,
                )
            except Exception as asr_exc:
                logger.warning("ASR word extraction for segmentation failed (%s); falling back to acoustic energy valleys.", asr_exc)
                all_words = []

        segmented_turns, segment_audits = smart_segment_speaker_turns(
            audio,
            current_turns,
            max_duration_s=config.target_max_duration_s,
            min_duration_s=config.target_min_duration_s,
            min_pause_s=config.min_split_pause_s,
            words=all_words or None,
            search_window_s=config.energy_search_window_s,
            frame_len_ms=config.energy_frame_len_ms,
            hop_len_ms=config.energy_hop_len_ms,
        )
        current_turns = segmented_turns
        funnel["segmented_turns_count"] = len(current_turns)
        funnel["segmented_speech_duration_s"] = round(
            sum(t.duration_s for t in current_turns), 2
        )
        log_stage(
            f"Smart segmentation produced {len(current_turns)} sentence-level turns "
            f"({funnel['segmented_speech_duration_s']:.1f}s speech)"
        )

    # ==================== STAGE 4: Embedding Homogeneity ====================
    if config.enable_homogeneity:
        if cancel_check and cancel_check():
            raise InterruptedError("Pipeline execution cancelled before Stage 4")

        homo_dev = config.homogeneity_device if (config.homogeneity_device and config.homogeneity_device != "same") else config.device
        if progress_callback:
            progress_callback(0.65, f"Verifying sliding-window WeSpeaker embedding homogeneity on {homo_dev}...")
        log_stage(
            f"Stage 4: Running sliding WeSpeaker homogeneity filter on {homo_dev} "
            f"(window={config.homogeneity_window_s:.2f}s, hop={config.homogeneity_hop_s:.2f}s, min_sim={config.min_homogeneity_similarity:.2f})"
        )
        homo_turns, homo_audits = filter_by_embedding_homogeneity(
            audio,
            current_turns,
            window_s=config.homogeneity_window_s,
            hop_s=config.homogeneity_hop_s,
            min_similarity=config.min_homogeneity_similarity,
            device=homo_dev,
            token=config.token,
            cancel_check=cancel_check,
        )
        funnel["homogeneity_turns_count"] = len(homo_turns)
        funnel["homogeneity_speech_duration_s"] = round(
            sum(t.duration_s for t in homo_turns), 2
        )
        log_stage(
            f"Homogeneity check kept {len(homo_turns)} turns "
            f"(rejected {len(current_turns) - len(homo_turns)} turns with foreign timbre/drift)"
        )
        current_turns = homo_turns
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    fm_audits: list[tuple[SpeakerTurn, bool, str, dict[str, Any]]] = []

    # ==================== STAGE 5: Foundation Model Audits ====================
    if config.enable_gemma or config.enable_vibevoice:
        if cancel_check and cancel_check():
            raise InterruptedError("Pipeline execution cancelled before Stage 5")

        log_stage("Stage 5: Executing in-loop foundation model verification")
        fm_turns, fm_audits = filter_by_foundation_models(
            audio,
            current_turns,
            config,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        funnel["foundation_turns_count"] = len(fm_turns)
        funnel["foundation_speech_duration_s"] = round(
            sum(t.duration_s for t in fm_turns), 2
        )
        log_stage(
            f"Foundation models kept {len(fm_turns)} turns "
            f"(rejected {len(current_turns) - len(fm_turns)} impure, incomplete, "
            "uncertain, or failed turns)"
        )
        current_turns = fm_turns
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Final Assembly
    elapsed_total = time.time() - t0
    funnel["final_pure_turns_count"] = len(current_turns)
    funnel["final_pure_speech_duration_s"] = round(
        sum(t.duration_s for t in current_turns), 2
    )
    funnel["total_elapsed_s"] = round(elapsed_total, 2)

    enabled_gates: list[str] = [f"Primary ({config.primary_backend})"]
    if config.enable_consensus:
        enabled_gates.append(f"Consensus ({config.secondary_backend})")
    if config.enable_collar_erosion:
        collar_parts = ["Collar"]
        if config.enable_context_collar:
            collar_parts.append("ContextHandoff")
        if config.enable_energy_snapping:
            collar_parts.append("EnergySnap")
        if config.enable_syllable_alignment:
            collar_parts.append(f"AlignLock:{config.aligner_engine}")
        enabled_gates.append("+".join(collar_parts))
    if config.enable_smart_segmentation:
        enabled_gates.append(f"SmartSegmentation:{config.target_max_duration_s:.1f}s")
    if config.enable_homogeneity:
        enabled_gates.append("Homogeneity")
    if config.enable_vibevoice:
        enabled_gates.append("VibeVoice")
    if config.enable_gemma:
        enabled_gates.append(f"DirectAudio:{config.gemma_backend}")

    funnel["contamination_risk_rating"] = f"Passed {len(enabled_gates)} active validation gates: {', '.join(enabled_gates)}"
    funnel["enabled_gates"] = enabled_gates

    foundation_audits: list[dict[str, Any]] = []
    total_usage = {
        "requests": 0,
        "prompt_tokens": 0,
        "audio_input_tokens": 0,
        "text_input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
    }
    total_cost_usd = 0.0
    priced_requests = 0
    for turn, passed_audit, reason, metadata in fm_audits:
        direct = metadata.get("gemma")
        if isinstance(direct, dict):
            usage = direct.get("usage")
            if isinstance(usage, dict):
                total_usage["requests"] += 1
                for key in (
                    "prompt_tokens",
                    "audio_input_tokens",
                    "text_input_tokens",
                    "output_tokens",
                    "thinking_tokens",
                    "total_tokens",
                ):
                    total_usage[key] += int(usage.get(key, 0) or 0)
            cost = direct.get("cost")
            if isinstance(cost, dict) and cost.get("total_usd") is not None:
                total_cost_usd += float(cost["total_usd"])
                priced_requests += 1
        foundation_audits.append({
            "speaker_id": turn.speaker_id,
            "start_s": turn.start_s,
            "end_s": turn.end_s,
            "duration_s": turn.duration_s,
            "passed": passed_audit,
            "reason": reason,
            "direct_audio": direct,
            "direct_audio_error": metadata.get("gemma_error"),
            "vibevoice": metadata.get("vibevoice"),
            "vibevoice_error": metadata.get("vibevoice_error"),
        })
    if total_usage["requests"]:
        funnel["direct_audio_usage"] = total_usage
        funnel["direct_audio_cost"] = {
            "total_usd": round(total_cost_usd, 9),
            "priced_requests": priced_requests,
            "currency": "USD",
            "pricing_tier": "paid_standard",
            "estimated": True,
        }

    active_speakers = sorted({t.speaker_id for t in current_turns})
    final_speakers = [Speaker(speaker_id=spk) for spk in active_speakers]

    final_result = DiarizationResult(
        schema_version="2.0",
        audio_id=audio.source_id,
        speakers=final_speakers,
        turns=current_turns,
        source_audio=audio,
        model=DiarizationModelInfo(
            backend="zero-contamination-pipeline",
            model_id=f"primary:{config.primary_backend}+consensus:{config.enable_consensus}",
        ),
        channel_id=audio.channel_id,
        channel_name=audio.channel_name,
        channel_url=audio.channel_url,
    )

    rescued_count = 0
    total_delta_ms = 0.0
    final_audits = []

    for i, t in enumerate(current_turns):
        raw_start = getattr(t, "_raw_start_s", t.start_s)
        raw_end = getattr(t, "_raw_end_s", t.end_s)
        orig_start = getattr(t, "_original_start_s", raw_start)
        orig_end = getattr(t, "_original_end_s", raw_end)
        delta_start = getattr(t, "_delta_start_ms", round((t.start_s - raw_start) * 1000.0, 1))
        delta_end = getattr(t, "_delta_end_ms", round((t.end_s - orig_end) * 1000.0, 1))
        policy = getattr(t, "_boundary_policy", "standard")
        tail_rescued = getattr(t, "_tail_rescued", (delta_end > 0.0))
        transcript = getattr(t, "_transcript", None)

        if tail_rescued:
            rescued_count += 1
            total_delta_ms += max(0.0, delta_end)

        final_audits.append(
            TurnAuditRecord(
                turn_id=f"pure_turn_{i:04d}",
                speaker_id=t.speaker_id,
                original_start_s=orig_start,
                original_end_s=orig_end,
                start_s=t.start_s,
                end_s=t.end_s,
                duration_s=t.duration_s,
                status="passed",
                rejection_reason="Passed all enabled validation gates",
                min_similarity=getattr(t, "_min_similarity", None),
                gemma_decision=getattr(t, "_gemma_decision", None),
                vibevoice_decision=getattr(t, "_vibevoice_decision", None),
                raw_start_s=raw_start,
                raw_end_s=raw_end,
                delta_start_ms=delta_start,
                delta_end_ms=delta_end,
                boundary_policy=policy,
                tail_rescued=tail_rescued,
                transcript=transcript,
            )
        )

    funnel["syllables_rescued_count"] = rescued_count
    funnel["avg_tail_preservation_ms"] = (
        round(total_delta_ms / max(1, rescued_count), 1) if rescued_count > 0 else 0.0
    )

    log_stage(
        f"Pipeline complete in {elapsed_total:.2f}s! Produced {len(current_turns)} candidate turns "
        f"({funnel['final_pure_speech_duration_s']:.1f}s, rescued {rescued_count} syllable tails)."
    )

    return ZeroContaminationResult(
        diarization=final_result,
        audit_records=final_audits,
        funnel_stats=funnel,
        stage_log=stage_logs,
        config=config.to_dict(),
        foundation_audits=foundation_audits,
        boundary_audits=boundary_audits,
        segment_audits=segment_audits,
    )


def _run_backend(
    backend: str,
    audio: Audio,
    *,
    device: str = "auto",
    token: str | None = None,
    onset: float = DEFAULT_TARGET_ONSET,
    offset: float = DEFAULT_TARGET_OFFSET,
) -> DiarizationResult:
    """Run an isolated or standard diarizer backend and return DiarizationResult."""
    b = backend.lower().strip()
    if b in {"sortformer", "nemo-sortformer"}:
        from src.diarization.SortformerWorkerDiarizer import SortformerWorkerDiarizer

        diarizer = SortformerWorkerDiarizer(
            device=device,
            token=token,
            onset=onset,
            offset=offset,
        )
        with diarizer:
            return diarizer.diarize(audio)

    elif b in {"diarizen", "diarizen_large_s80_v2"}:
        from src.diarization.DiariZenWorkerDiarizer import DiariZenWorkerDiarizer

        diarizer = DiariZenWorkerDiarizer(
            device=device,
            token=token,
        )
        with diarizer:
            return diarizer.diarize(audio)

    elif b in {"pyannote", "pyannote_community"}:
        from src.diarization.PyannoteDiarizer import PyannoteDiarizer

        diarizer = PyannoteDiarizer(
            model_id="pyannote/speaker-diarization-community-1",
            device=device,
            token=token,
        )
        with diarizer:
            return diarizer.diarize(audio)

    elif b in {"pyannote_31", "pyannote_3.1"}:
        from src.diarization.PyannoteDiarizer import PyannoteDiarizer

        diarizer = PyannoteDiarizer(
            model_id="pyannote/speaker-diarization-3.1",
            device=device,
            token=token,
        )
        with diarizer:
            return diarizer.diarize(audio)

    else:
        raise ValueError(f"Unsupported diarizer backend: {backend}")
