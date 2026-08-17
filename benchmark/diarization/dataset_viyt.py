"""Load and cache the ViYT-Diar Hugging Face evaluation set."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf

from benchmark.diarization import (
    CACHE_DIR,
    VIYT_DIAR_DATASET_ID,
    VIYT_DIAR_SPLIT,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ReferenceTurn:
    """One ground-truth speaker interval."""

    speaker_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class ViYTSample:
    """One cached ViYT-Diar clip with reference turns."""

    audio_id: str
    audio_path: Path
    sample_rate: int
    duration_s: float
    speakers: tuple[str, ...]
    turns: tuple[ReferenceTurn, ...]

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)


def _speaker_list(row: dict[str, Any]) -> list[str]:
    raw = row.get("speaker")
    if raw is None:
        raw = row.get("speakers")
    if raw is None:
        raise KeyError("Row is missing speaker / speakers field")
    return [str(item) for item in raw]


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _write_wav_from_hf_audio(
    audio_info: dict[str, Any],
    wav_path: Path,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> None:
    """Decode HF ``Audio(decode=False)`` path/bytes via soundfile (not torchcodec).

    Datasets 4+/5+ decode Audio columns with torchcodec+FFmpeg by default. That
    stack is brittle on the model server, so we keep HF decoding disabled and
    materialize mono 16 kHz WAVs ourselves.

    Args:
        audio_info: Dict with ``path`` and/or ``bytes`` from the audio column.
        wav_path: Destination WAV path.
        target_sr: Output sample rate.

    Raises:
        ValueError: If neither path nor bytes is available, or audio is empty.
    """
    import tempfile

    import librosa

    path = audio_info.get("path")
    raw = audio_info.get("bytes")

    array: np.ndarray | None = None
    sr: int | None = None
    if raw is not None:
        try:
            array, sr = sf.read(io.BytesIO(raw), always_2d=False)
        except (RuntimeError, OSError, ValueError, sf.LibsndfileError):
            suffix = Path(str(path)).suffix if path else ".audio"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                array, sr = librosa.load(tmp_path, sr=None, mono=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    elif path:
        try:
            array, sr = sf.read(str(path), always_2d=False)
        except (RuntimeError, OSError, ValueError, sf.LibsndfileError):
            array, sr = librosa.load(str(path), sr=None, mono=True)
    else:
        raise ValueError("audio sample has neither path nor bytes")

    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"empty audio for {wav_path.name}")
    if array.ndim > 1:
        array = np.mean(array, axis=-1)

    sr = int(sr)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    sf.write(str(wav_path), array.astype(np.float32), sr)


def prepare_viyt_diar(
    *,
    cache_dir: Path | None = None,
    dataset_id: str = VIYT_DIAR_DATASET_ID,
    split: str = VIYT_DIAR_SPLIT,
    limit: int | None = None,
    force: bool = False,
) -> list[ViYTSample]:
    """Download ViYT-Diar and materialize WAV + reference turns under ``cache_dir``.

    Args:
        cache_dir: Destination for WAVs and ``manifest.json``.
        dataset_id: Hugging Face dataset id.
        split: Dataset split (ViYT-Diar uses ``test``).
        limit: Optional max number of clips (for smoke tests).
        force: Re-download and overwrite cached files when True.

    Returns:
        Ordered list of cached samples.

    Raises:
        ImportError: If ``datasets`` is not installed.
        RuntimeError: If the dataset has no rows after filtering.
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with: uv add datasets"
        ) from exc

    out_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = _manifest_path(out_dir)
    if manifest_file.is_file() and not force:
        samples = load_cached_samples(cache_dir=out_dir, limit=limit)
        if samples:
            logger.info("Using cached ViYT-Diar manifest (%d clips)", len(samples))
            return samples

    logger.info("Loading %s split=%s from Hugging Face…", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    # Keep path/bytes only — datasets 4+/5+ would otherwise decode via torchcodec.
    ds = ds.cast_column("audio", Audio(decode=False))

    samples: list[ViYTSample] = []
    for index, row in enumerate(ds):
        if limit is not None and index >= limit:
            break

        audio_id = str(row.get("audio_id") or f"viyt_{index:03d}")
        audio_info = row["audio"]
        wav_path = audio_dir / f"{audio_id}.wav"

        if force or not wav_path.is_file():
            _write_wav_from_hf_audio(audio_info, wav_path)

        starts = [float(x) for x in row["timestamps_start"]]
        ends = [float(x) for x in row["timestamps_end"]]
        speakers = _speaker_list(row)
        if not (len(starts) == len(ends) == len(speakers)):
            raise ValueError(
                f"{audio_id}: mismatched speaker/timestamp list lengths "
                f"({len(speakers)}, {len(starts)}, {len(ends)})"
            )

        turns = tuple(
            ReferenceTurn(speaker_id=spk, start_s=start, end_s=end)
            for spk, start, end in zip(speakers, starts, ends, strict=True)
            if end > start
        )
        unique_speakers = tuple(sorted({turn.speaker_id for turn in turns}))
        info = sf.info(str(wav_path))
        samples.append(
            ViYTSample(
                audio_id=audio_id,
                audio_path=wav_path.resolve(),
                sample_rate=int(info.samplerate),
                duration_s=float(info.duration),
                speakers=unique_speakers,
                turns=turns,
            )
        )

    if not samples:
        raise RuntimeError(f"No samples loaded from {dataset_id} ({split})")

    payload = {
        "dataset_id": dataset_id,
        "split": split,
        "num_samples": len(samples),
        "samples": [
            {
                "audio_id": s.audio_id,
                "audio_path": str(s.audio_path),
                "sample_rate": s.sample_rate,
                "duration_s": s.duration_s,
                "speakers": list(s.speakers),
                "turns": [asdict(t) for t in s.turns],
            }
            for s in samples
        ],
    }
    manifest_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Cached %d ViYT-Diar clips under %s", len(samples), out_dir)
    return samples


def load_cached_samples(
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> list[ViYTSample]:
    """Load samples from an existing ``manifest.json`` without hitting Hugging Face."""
    out_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    manifest_file = _manifest_path(out_dir)
    if not manifest_file.is_file():
        return []

    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    samples: list[ViYTSample] = []
    for row in payload.get("samples", []):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            logger.warning("Missing cached audio for %s: %s", row["audio_id"], audio_path)
            continue
        turns = tuple(
            ReferenceTurn(
                speaker_id=str(t["speaker_id"]),
                start_s=float(t["start_s"]),
                end_s=float(t["end_s"]),
            )
            for t in row["turns"]
        )
        samples.append(
            ViYTSample(
                audio_id=str(row["audio_id"]),
                audio_path=audio_path,
                sample_rate=int(row["sample_rate"]),
                duration_s=float(row["duration_s"]),
                speakers=tuple(str(s) for s in row["speakers"]),
                turns=turns,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def iter_samples(samples: list[ViYTSample]) -> Iterator[ViYTSample]:
    """Yield samples in stable order."""
    yield from samples
