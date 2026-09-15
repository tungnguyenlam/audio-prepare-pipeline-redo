"""ModelScope 3D-Speaker (speakerlab) diarization backend.

Cascaded audio-only pipeline from
https://github.com/modelscope/3D-Speaker : FSMN VAD, CAM++ speaker embeddings,
then spectral clustering, with optional pyannote overlap refinement. Requires
the isolated environment pinned in ``envs/requirements-3dspeaker.txt``. The
``speakerlab`` sources are shallow-cloned into ``.data/3d-speaker`` on first
load when missing (``speakerlab`` is not published as a package).
"""

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


import argparse
import contextlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, convert, identity, inputs, manifest_destinations, parser, positive_int, probe, request, safe_name
from _common.merge import add_diarization_merge_arguments, merge_parameters
from _common.segments import ensure_plots, export, manifest_complete

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
DEFAULT_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_MODEL_ID = f"{DEFAULT_VAD_MODEL_ID}+{DEFAULT_EMBEDDING_MODEL_ID}"
DEFAULT_CHUNK_DURATION_S = 1.5
DEFAULT_CHUNK_STEP_S = 0.75
SAMPLE_RATE = 16000
THREEDSPEAKER_GIT_URL = "https://github.com/modelscope/3D-Speaker.git"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class _ThreeDSpeaker:
    """Diarize audio with ModelScope 3D-Speaker's ``Diarization3Dspeaker``.

    Speaker labels from clustering are converted to result-local ``spk_NN``
    identifiers. Those identifiers are meaningful only within one result.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        num_speakers: int | None = None,
        include_overlap: bool = False,
        batch_size: int = 64,
        chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S,
        chunk_step_s: float = DEFAULT_CHUNK_STEP_S,
        token: str | None = None,
        model_cache_dir: str | Path | None = None,
        speakerlab_root: str | Path | None = None,
        work_dir: Path,
    ) -> None:
        """Initialize the 3D-Speaker diarizer.

        Args:
            device: Compute device (``"auto"``, ``"cuda"``, ``"cpu"``, etc.).
            num_speakers: Exact speaker count when known in advance.
            include_overlap: Enable pyannote segmentation-based overlap
                refinement. Requires a Hugging Face token.
            batch_size: Speaker-embedding and overlap-segmentation batch size.
            chunk_duration_s: Embedding subsegment window length in seconds.
                Upstream default is ``1.5``.
            chunk_step_s: Hop between consecutive subsegments in seconds.
                Upstream default is ``0.75`` (50% overlap at the default
                duration). Must be ``> 0`` and ``<= chunk_duration_s``.
            token: Hugging Face token used when ``include_overlap`` is True.
                Falls back to ``HF_TOKEN`` when unset.
            model_cache_dir: Directory for ModelScope pretrained downloads.
                Defaults to ``.data/modelscope`` under the repository root.
            speakerlab_root: Path to a cloned
                https://github.com/modelscope/3D-Speaker checkout. Defaults to
                ``THREEDSPEAKER_ROOT`` or ``.data/3d-speaker``. Missing checkouts
                are shallow-cloned on first ``load()``.
            work_dir: Directory for intermediate audio and model manifests.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be an integer of at least 1")
        if num_speakers is not None and num_speakers < 1:
            raise ValueError("num_speakers must be at least 1")
        duration_s, step_s = self._validate_chunk_settings(
            chunk_duration_s,
            chunk_step_s,
        )
        if include_overlap:
            resolved_token = token if token is not None else os.getenv("HF_TOKEN")
            if not resolved_token:
                raise ValueError(
                    "include_overlap=True requires token=... or HF_TOKEN in the "
                    "environment"
                )
            token = resolved_token

        self.device = str(device)
        self.num_speakers = num_speakers
        self.include_overlap = bool(include_overlap)
        self.batch_size = batch_size
        self.chunk_duration_s = duration_s
        self.chunk_step_s = step_s
        self.token = token
        self.model_cache_dir = (
            Path(model_cache_dir).expanduser()
            if model_cache_dir is not None
            else _REPO_ROOT / ".data" / "modelscope"
        )
        configured_root = (
            speakerlab_root
            if speakerlab_root is not None
            else os.getenv("THREEDSPEAKER_ROOT")
        )
        self.speakerlab_root = (
            Path(configured_root).expanduser()
            if configured_root is not None
            else _REPO_ROOT / ".data" / "3d-speaker"
        )
        self.work_dir = work_dir

        self._pipeline: Any | None = None
        self._target_device: Any | None = None

    @staticmethod
    def _validate_chunk_settings(
        chunk_duration_s: float,
        chunk_step_s: float,
    ) -> tuple[float, float]:
        """Validate embedding subsegment window and hop."""
        if isinstance(chunk_duration_s, bool) or not isinstance(
            chunk_duration_s, (int, float)
        ):
            raise TypeError("chunk_duration_s must be a number")
        if isinstance(chunk_step_s, bool) or not isinstance(
            chunk_step_s, (int, float)
        ):
            raise TypeError("chunk_step_s must be a number")
        duration_s = float(chunk_duration_s)
        step_s = float(chunk_step_s)
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("chunk_duration_s must be finite and > 0")
        if not math.isfinite(step_s) or step_s <= 0:
            raise ValueError("chunk_step_s must be finite and > 0")
        if step_s > duration_s:
            raise ValueError(
                "chunk_step_s must be <= chunk_duration_s "
                f"(got step={step_s}, duration={duration_s})"
            )
        return duration_s, step_s

    def _apply_chunk_settings(self) -> None:
        """Override upstream ``chunk(dur=1.5, step=0.75)`` with configured values."""
        pipeline = self._pipeline
        if pipeline is None:
            return

        duration_s = self.chunk_duration_s
        step_s = self.chunk_step_s

        def chunk(st: float, ed: float, dur: float = duration_s, step: float = step_s):
            chunks: list[list[float]] = []
            subseg_st = float(st)
            end = float(ed)
            while subseg_st + dur < end + step:
                subseg_ed = min(subseg_st + dur, end)
                chunks.append([subseg_st, subseg_ed])
                subseg_st += step
            return chunks

        pipeline.chunk = chunk


    def _ensure_speakerlab_path(self) -> Path:
        root = self.speakerlab_root.expanduser()
        if not (root / "speakerlab").is_dir():
            self._clone_speakerlab(root)
        root = root.resolve()
        if not (root / "speakerlab").is_dir():
            raise RuntimeError(
                f"3D-Speaker checkout is incomplete at {root}. Expected a "
                "speakerlab/ package directory."
            )
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return root

    @staticmethod
    def _clone_speakerlab(root: Path) -> None:
        """Shallow-clone modelscope/3D-Speaker into ``root`` when missing."""
        if shutil.which("git") is None:
            raise RuntimeError(
                f"3D-Speaker checkout not found at {root} and git is not "
                f"available to clone {THREEDSPEAKER_GIT_URL}. Install git or "
                "set THREEDSPEAKER_ROOT to an existing checkout."
            )

        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"3D-Speaker checkout not found at {root}: the directory "
                "exists but does not contain speakerlab/. Remove it or set "
                "THREEDSPEAKER_ROOT to a valid checkout."
            )

        logger.info("Cloning 3D-Speaker into %s", root)
        staging = root.with_name(f"{root.name}.partial")
        if staging.exists():
            shutil.rmtree(staging)
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                THREEDSPEAKER_GIT_URL,
                str(staging),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone 3D-Speaker into {root}: "
                f"{detail[:2000] or 'no output'}"
            )
        if root.exists():
            root.rmdir()
        staging.rename(root)

    def _resolve_device(self, torch: Any) -> Any:
        if self.device != "auto":
            target = torch.device(self.device)
            if target.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if target.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but PyTorch cannot initialize it"
                )
            return target

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self) -> None:
        """Load CAM++ embeddings, FSMN VAD, and optional overlap segmentation."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "3D-Speaker diarizer dependencies are unavailable. Install the "
                "pinned envs/requirements-3dspeaker.txt dependencies in an isolated "
                "environment."
            ) from exc

        self._ensure_speakerlab_path()
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(self.model_cache_dir))

        try:
            from speakerlab.bin.infer_diarization import Diarization3Dspeaker
        except ImportError as exc:
            raise RuntimeError(
                "speakerlab is unavailable. Clone "
                "https://github.com/modelscope/3D-Speaker into .data/3d-speaker "
                "or set THREEDSPEAKER_ROOT, and install "
                "envs/requirements-3dspeaker.txt."
            ) from exc

        target_device = self._resolve_device(torch)
        pipeline = Diarization3Dspeaker(
            device=target_device,
            include_overlap=self.include_overlap,
            hf_access_token=self.token,
            speaker_num=self.num_speakers,
            model_cache_dir=str(self.model_cache_dir),
        )
        self._target_device = target_device
        self._pipeline = pipeline
        pipeline.batchsize = self.batch_size
        if self.include_overlap and hasattr(pipeline, "segmentation_model"):
            pipeline.segmentation_model.batch_size = self.batch_size
        self._apply_chunk_settings()

    def _unload(self) -> None:
        """Release VAD, embedding, clustering, and accelerator caches."""
        self._pipeline = None
        self._target_device = None
        gc.collect()

        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def diarize(
        self,
        audio: Path,
        *,
        num_speakers: int | None = None,
    ) -> list[dict]:
        """Diarize ``audio`` and return ordinary turn dictionaries.

        Args:
            audio: File-backed audio item to diarize.
            num_speakers: Optional per-call oracle speaker count. When omitted,
                uses the value configured at construction time.

        Returns:
            Speaker identities and turns with local ``spk_NN`` labels.

        Raises:
            RuntimeError: If the model is not loaded or inference fails.
            FileNotFoundError: If ``audio`` does not exist.
            ValueError: If the normalized audio duration is empty.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "3D-Speaker checkpoint is unavailable. Initialize inference before "
                "diarize(), Initialize inference before processing files."
            )

        source_path = Path(audio)
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source_path}")

        oracle_speakers = (
            num_speakers if num_speakers is not None else self.num_speakers
        )
        if oracle_speakers is not None and oracle_speakers < 1:
            raise ValueError("num_speakers must be at least 1")

        with tempfile.TemporaryDirectory(prefix="3d-speaker-", dir=self.work_dir) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            normalized_path = temp_dir / "normalized.wav"
            convert(
                source_path,
                normalized_path,
                sample_rate=SAMPLE_RATE,
                channels=1,
            )
            if probe(normalized_path)['duration_s'] <= 0:
                raise ValueError(f"Audio source is empty: {source_path}")

            try:
                segments = self._pipeline(
                    str(normalized_path),
                    wav_fs=SAMPLE_RATE,
                    speaker_num=oracle_speakers,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"3D-Speaker diarization failed: {exc}"
                ) from exc

        turns, speakers = self._turns_from_segments(
            segments,
            duration_s=probe(audio)['duration_s'],
        )
        return turns

    @staticmethod
    def _turns_from_segments(
        segments: Any,
        duration_s: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Convert ``[[start, end, speaker_id], ...]`` into schema turns.

        3D-Speaker timestamps are quantized to model frames, so the final
        segment can extend slightly past the source file. Clamp generated
        turns to the known source duration before schema validation.
        """
        if segments is None:
            return [], []

        raw_turns: list[tuple[str, float, float]] = []
        for item in segments:
            try:
                start_s = float(item[0])
                end_s = float(item[1])
                label = str(item[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                continue
            if duration_s is not None:
                start_s = min(max(0.0, start_s), duration_s)
                end_s = min(max(0.0, end_s), duration_s)
            else:
                start_s = max(0.0, start_s)
            if end_s <= start_s:
                continue
            raw_turns.append((label, start_s, end_s))

        raw_turns.sort(key=lambda item: (item[1], item[2], item[0]))
        label_to_speaker_id: dict[str, str] = {}
        speakers: list[dict] = []
        turns: list[dict] = []
        for label, start_s, end_s in raw_turns:
            speaker_id = label_to_speaker_id.get(label)
            if speaker_id is None:
                speaker_id = f"spk_{len(label_to_speaker_id):02d}"
                label_to_speaker_id[label] = speaker_id
                speakers.append(dict(speaker_id=speaker_id))
            turns.append(
                dict(
                    speaker_id=speaker_id,
                    start_s=float(start_s),
                    end_s=float(end_s),
                    confidence=None,
                )
            )
        return turns, speakers


def main() -> int:
    p = parser('Standalone threed_speaker diarization.', 'diarize', 'threed_speaker', segments=True)
    p.add_argument('--device', default='auto',
                   help='Compute device for model inference (e.g. auto, cpu, cuda) (default: auto)')
    p.add_argument('--num-speakers', type=positive_int, default=None,
                   help='Exact known number of speakers (default: None)')
    p.add_argument('--batch-size', type=positive_int, default=64,
                   help='Embedding and overlap segmentation batch size (default: 64)')
    p.add_argument('--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz for exported turn clips (default: preserve source)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout for clips (1=mono, 2=stereo) (default: 1)')
    p.add_argument('--chunk-duration-s', type=float, default=DEFAULT_CHUNK_DURATION_S,
                   help='Audio chunk window length in seconds (default: 1.5)')
    p.add_argument('--chunk-step-s', type=float, default=DEFAULT_CHUNK_STEP_S,
                   help='Audio chunk step hop in seconds (default: 0.75)')
    p.add_argument('--model-cache-dir', type=Path, default=None,
                   help='Directory to cache downloaded 3D-Speaker model checkpoints (default: None)')
    p.add_argument('--speakerlab-root', type=Path, default=None,
                   help='Root directory for 3D-Speaker SpeakerLab submodule (default: None)')
    p.add_argument('--include-overlap', action=argparse.BooleanOptionalAction, default=False,
                   help='Include overlapping speaker segments (default: False)')
    p.add_argument('--min-duration-s', type=float, default=1.0, help='Minimum turn duration in seconds to keep and export (default: 1.0)')
    p.add_argument('--max-duration-s', type=float, default=15.0, help='Maximum turn duration in seconds to keep and export (default: 15.0)')
    add_diarization_merge_arguments(p)
    args = p.parse_args()
    merge_options = {**merge_parameters(args, p), 'adjust_mean': args.adjust_mean}
    if args.min_duration_s is not None and (not math.isfinite(args.min_duration_s) or args.min_duration_s < 0):
        p.error('--min-duration-s must be finite and non-negative')
    if args.max_duration_s is not None and (not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0):
        p.error('--max-duration-s must be finite and positive')
    if args.min_duration_s is not None and args.max_duration_s is not None and args.min_duration_s > args.max_duration_s:
        p.error('--min-duration-s cannot exceed --max-duration-s')
    pairs = manifest_destinations(args)
    parameters = {key: getattr(args, key) for key in ('device', 'num_speakers', 'batch_size', 'chunk_duration_s', 'chunk_step_s', 'model_cache_dir', 'speakerlab_root', 'include_overlap')}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        model = _ThreeDSpeaker(**parameters, work_dir=args.work_dir)
        model._load()
    parameters['device'] = str(model._target_device)
    parameters['checkpoint'] = DEFAULT_MODEL_ID
    parameters['speakerlab_root'] = str(model.speakerlab_root.resolve())
    parameters['model_cache_dir'] = str(model.model_cache_dir.resolve())
    parameters = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in parameters.items()}
    def process(src, dest):
        rate = args.sample_rate or probe(src)['sample_rate']
        wanted = request(identity(src), 'diarize', {**({'merge': merge_options} if args.merge else {}),
                         **parameters, 'sample_rate': rate, 'channels': args.channels,
                         'min_duration_s': args.min_duration_s, 'max_duration_s': args.max_duration_s}, 'threed_speaker')
        if manifest_complete(dest, wanted, args.overwrite):
            ensure_plots(dest, overwrite=False)
            return
        turns = model.diarize(src)
        export({**wanted, 'speaker_ids': sorted({t['speaker_id'] for t in turns}), 'turns': turns},
               src, dest, args.work_dir, rate, args.channels, args.min_duration_s, args.max_duration_s,
               concurrency=args.concurrency, batch_size=args.batch_size)
        ensure_plots(dest, overwrite=True)
    try:
        return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            model._unload()


if __name__ == '__main__':
    raise SystemExit(main())
