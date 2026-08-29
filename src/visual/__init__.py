"""Audiovisual speaker identity: face tracks, active-speaker detection, and late fusion."""

from src.visual.AVVerifier import (
    DEFAULT_ASD_PURITY_MIN,
    DEFAULT_EXPAND_FACE_THRESHOLD,
    DEFAULT_EXPAND_VOICE_THRESHOLD,
    DEFAULT_FACE_SIMILARITY_THRESHOLD,
    DEFAULT_TRANSITION_MARGIN_S,
    AVVerifier,
    AVVerifierError,
)
from src.visual.FaceAnalyzer import FaceAnalyzer, FaceAnalyzerError
from src.visual.LightASD import LightASD, LightASDError
from src.visual.Video import Video, VideoError, probe_video
from src.visual.schemas import (
    ASDFrameScore,
    ASDResult,
    AVSegmentDecision,
    AVVerificationResult,
    FaceObservation,
    FaceTrack,
    FaceTrackSet,
    SpeakerEntity,
)

__all__ = [
    "ASDFrameScore",
    "ASDResult",
    "AVSegmentDecision",
    "AVVerifier",
    "AVVerifierError",
    "AVVerificationResult",
    "DEFAULT_ASD_PURITY_MIN",
    "DEFAULT_EXPAND_FACE_THRESHOLD",
    "DEFAULT_EXPAND_VOICE_THRESHOLD",
    "DEFAULT_FACE_SIMILARITY_THRESHOLD",
    "DEFAULT_TRANSITION_MARGIN_S",
    "FaceAnalyzer",
    "FaceAnalyzerError",
    "FaceObservation",
    "FaceTrack",
    "FaceTrackSet",
    "LightASD",
    "LightASDError",
    "SpeakerEntity",
    "Video",
    "VideoError",
    "probe_video",
]
