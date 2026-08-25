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
    DEFAULT_UNSLOTH_ENDPOINT,
    DEFAULT_UNSLOTH_PORT,
    BaseOverlapVerifier,
    Gemma4OverlapVerifier,
    GeminiOverlapVerifier,
    OverlapVerificationResult,
    OverlapVerifierError,
    create_overlap_verifier,
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

from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    ScoredSegment,
    Speaker,
    SpeakerPurityResult,
    SpeakerSimilarityWindow,
    SpeakerTurn,
    TargetSpeakerResult,
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
    "DEFAULT_MAX_OVERLAP_DURATION_S",
    "DEFAULT_PYANNOTE_MODEL_ID",
    "DEFAULT_PURITY_WINDOW_DURATION_S",
    "DEFAULT_PURITY_WINDOW_HOP_S",
    "DEFAULT_UNSLOTH_ENDPOINT",
    "DEFAULT_UNSLOTH_PORT",
    "DiarizationModelInfo",
    "DiarizationResult",
    "DiariZenDiarizer",
    "DiariZenWorkerDiarizer",
    "GeminiOverlapVerifier",
    "Gemma4OverlapVerifier",
    "OverlapVerificationResult",
    "OverlapVerifierError",
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
    "create_overlap_verifier",
]
