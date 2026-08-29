"""Multimodal speaker-entity enrollment and audiovisual purity gates.

Callers compose this with ``FaceAnalyzer``, ``LightASD``, an existing
``SpeakerVerifier``, and a ``DiarizationResult``. This module does not crawl,
separate, or diarize.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_paths import DATA_DIR
from src.diarization.SpeakerVerifier import (
    DEFAULT_MAX_OVERLAP_DURATION_S,
    DEFAULT_PURITY_WINDOW_DURATION_S,
    DEFAULT_PURITY_WINDOW_HOP_S,
    SpeakerProfile,
    SpeakerVerifier,
)
from src.diarization.schemas import (
    AV_SCHEMA_VERSION,
    DiarizationModelInfo,
    DiarizationResult,
    SpeakerPurityResult,
    SpeakerTurn,
)
from src.utils.AudioClass import Audio, _sanitize_filename_component
from src.visual.FaceAnalyzer import FaceAnalyzer
from src.visual.Video import Video
from src.visual.schemas import (
    ASDResult,
    AV_SCHEMA_VERSION as _AV_SCHEMA_VERSION,
    AVSegmentDecision,
    AVVerificationResult,
    ENTITY_SCHEMA_VERSION,
    FaceTrack,
    FaceTrackSet,
    SPEAKER_ENTITY_KIND,
    SpeakerEntity,
    VisualStatus,
)

# SpeakerPurityResult lives in diarization.schemas; AV schema version is independent.
del AV_SCHEMA_VERSION  # imported by mistake below if any - wait I imported AV_SCHEMA_VERSION from diarization which doesn't have it

DEFAULT_FACE_SIMILARITY_THRESHOLD = 0.50
DEFAULT_ASD_PURITY_MIN = 0.95
DEFAULT_TRANSITION_MARGIN_S = 0.08
DEFAULT_EXPAND_FACE_THRESHOLD = 0.70
DEFAULT_EXPAND_VOICE_THRESHOLD = 0.75
