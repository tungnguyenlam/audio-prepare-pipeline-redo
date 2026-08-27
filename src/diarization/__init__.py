"""Speaker diarization interfaces, schemas, and backends."""

from src.diarization.BaseDiarizer import BaseDiarizer
from src.diarization.ClusteringDiarizer import ClusteringDiarizer
from src.diarization.ClusteringWorkerDiarizer import ClusteringWorkerDiarizer
from src.diarization.DiariZenDiarizer import (
    DEFAULT_DIARIZEN_MODEL_ID,
    DiariZenDiarizer,
)
from src.diarization.DiariZenWorkerDiarizer import DiariZenWorkerDiarizer
from src.diarization.OverlapVerifier import (
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_GEMMA4_MODEL_ID,
    DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
    DEFAULT_UNSLOTH_ENDPOINT,
    DEFAULT_UNSLOTH_HOST,
    DEFAULT_UNSLOTH_PORT,
    BaseOverlapVerifier,
    Gemma4OverlapVerifier,
    GeminiOverlapVerifier,
    OverlapVerificationResult,
    OverlapVerifierError,
    OVERLAP_PROMPT,
    create_overlap_verifier,
    is_overlap_readiness_error,
)
from src.diarization.PyannoteDiarizer import (
    DEFAULT_PYANNOTE_MODEL_ID,
    PyannoteDiarizer,
)
from src.diarization.SortformerDiarizer import SortformerDiarizer
from src.diarization.SortformerWorkerDiarizer import SortformerWorkerDiarizer
from src.diarization.SpeakerVerifier import (
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_MAX_OVERLAP_DURATION_S,
    DEFAULT_PURITY_WINDOW_DURATION_S,
    DEFAULT_PURITY_WINDOW_HOP_S,
    SpeakerProfile,
    SpeakerVerifier,
    SpeakerVerifierError,
)
from src.diarization.ThreeDSpeakerDiarizer import ThreeDSpeakerDiarizer
from src.diarization.ThreeDSpeakerWorkerDiarizer import ThreeDSpeakerWorkerDiarizer
from src.diarization.VibeVoicePurityVerifier import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_SECONDARY_SPEECH_S,
    DEFAULT_VIBEVOICE_MODEL_ID,
    VibeVoicePurityError,
    VibeVoicePurityVerifier,
    classify_vibevoice_segments,
)
from src.diarization.VibeVoicePurityWorkerVerifier import VibeVoicePurityWorkerVerifier
from src.diarization.turn_cleanup import (
    DEFAULT_BOUNDARY_COLLAR_S,
    DEFAULT_JITTER_MAX_DURATION_S,
    DEFAULT_MERGE_SAME_SPEAKER_GAP_S,
    DEFAULT_MIN_TURN_DURATION_S,
    clean_speaker_turns,
    pad_and_merge_intervals,
)
from src.diarization.evaluation import evaluate_diarization

from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    ScoredSegment,
    Speaker,
    SpeakerPurityResult,
    SpeakerSimilarityWindow,
    SpeakerTurn,
    TargetSpeakerResult,
    VibeVoicePurityResult,
    VibeVoiceSpeakerTurn,
)

__all__ = [
    "BaseDiarizer",
    "BaseOverlapVerifier",
    "ClusteringDiarizer",
    "ClusteringWorkerDiarizer",
    "DEFAULT_DIARIZEN_MODEL_ID",
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DEFAULT_GEMINI_MODEL_ID",
    "DEFAULT_GEMMA4_MODEL_ID",
    "DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS",
    "DEFAULT_MAX_OVERLAP_DURATION_S",
    "DEFAULT_PYANNOTE_MODEL_ID",
    "DEFAULT_PURITY_WINDOW_DURATION_S",
    "DEFAULT_PURITY_WINDOW_HOP_S",
    "DEFAULT_UNSLOTH_ENDPOINT",
    "DEFAULT_UNSLOTH_HOST",
    "DEFAULT_UNSLOTH_PORT",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MIN_SECONDARY_SPEECH_S",
    "DEFAULT_VIBEVOICE_MODEL_ID",
    "DiarizationModelInfo",
    "DiarizationResult",
    "DiariZenDiarizer",
    "DiariZenWorkerDiarizer",
    "GeminiOverlapVerifier",
    "Gemma4OverlapVerifier",
    "OverlapVerificationResult",
    "OverlapVerifierError",
    "OVERLAP_PROMPT",
    "PyannoteDiarizer",
    "ScoredSegment",
    "Speaker",
    "SpeakerProfile",
    "SpeakerPurityResult",
    "SpeakerSimilarityWindow",
    "SpeakerTurn",
    "SpeakerVerifier",
    "SpeakerVerifierError",
    "SortformerDiarizer",
    "SortformerWorkerDiarizer",
    "TargetSpeakerResult",
    "ThreeDSpeakerDiarizer",
    "ThreeDSpeakerWorkerDiarizer",
    "VibeVoicePurityError",
    "VibeVoicePurityResult",
    "VibeVoicePurityVerifier",
    "VibeVoicePurityWorkerVerifier",
    "VibeVoiceSpeakerTurn",
    "classify_vibevoice_segments",
    "DEFAULT_BOUNDARY_COLLAR_S",
    "DEFAULT_JITTER_MAX_DURATION_S",
    "DEFAULT_MERGE_SAME_SPEAKER_GAP_S",
    "DEFAULT_MIN_TURN_DURATION_S",
    "clean_speaker_turns",
    "create_overlap_verifier",
    "evaluate_diarization",
    "is_overlap_readiness_error",
    "pad_and_merge_intervals",
]
