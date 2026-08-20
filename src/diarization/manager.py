"""Diarization Manager orchestrating pipeline execution, audio chopping, and export."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from src.diarization.base import DiarizationResult, SpeakerStats, SpeakerTurn
from src.diarization.clustering_diarizer import OfflineClusteringDiarizer
from src.diarization.pyannote_diarizer import PyannoteDiarizer

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_filesize(size_bytes: int) -> str:
    """Format bytes into readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class DiarizationManager:
    """Orchestrates Diarization workflows, file chopping, and export."""

    def __init__(self, base_diarized_dir: Path, device: str = "cuda"):
        self.base_diarized_dir = Path(base_diarized_dir).resolve()
        self.base_diarized_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.pyannote_diarizer = PyannoteDiarizer(device=device)
        self.offline_diarizer = OfflineClusteringDiarizer(device=device)

    def get_engines_status(self) -> dict:
        """Check status of available diarization engines."""
        pyannote_status = self.pyannote_diarizer.check_status()
        return {
            "pyannote": pyannote_status,
            "offline_clustering": {
                "available": True,
                "message": "Sẵn sàng (Offline ECAPA-TDNN & Agglomerative Clustering)",
                "model": "speechbrain/spkrec-ecapa-voxceleb",
            }
        }

    def run_diarization(
        self,
        input_audio_path: Path,
        engine: str = "offline_clustering",
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        filter_overlap: bool = True,
        min_duration_s: float = 0.5,
    ) -> DiarizationResult:
        """Run diarization, segment audio, export speaker clips, and generate stats."""
        input_audio_path = Path(input_audio_path).resolve()
        if not input_audio_path.is_file():
            raise FileNotFoundError(f"File audio đầu vào không tồn tại: {input_audio_path}")

        run_id = uuid.uuid4().hex[:12]
        output_dir = self.base_diarized_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        samples_dir = output_dir / "speaker_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        speakers_dir = output_dir / "speakers"
        speakers_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        logger.info("Starting Diarization run %s with engine '%s' on %s...", run_id, engine, input_audio_path.name)

        # 1. Execute Diarizer
        if engine == "pyannote":
            turns = self.pyannote_diarizer.diarize(
                audio_path=input_audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                filter_overlap=filter_overlap,
                min_duration_s=min_duration_s,
                auth_token=hf_token,
            )
        else:
            turns = self.offline_diarizer.diarize(
                audio_path=input_audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                filter_overlap=filter_overlap,
                min_duration_s=min_duration_s,
            )

        # Read original audio to slice speaker stems and samples
        info = sf.info(str(input_audio_path))
        orig_sr = info.samplerate
        total_audio_duration = info.duration
        audio_data, _ = sf.read(str(input_audio_path), dtype="float32")

        # 2. Group turns by speaker
        speaker_turns_map: Dict[str, List[SpeakerTurn]] = {}
        for turn in turns:
            speaker_turns_map.setdefault(turn.speaker_id, []).append(turn)

        # 3. Export Audio Clips and build stats
        speaker_stats_list: List[SpeakerStats] = []
        speaker_master_files: Dict[str, Path] = {}
        speaker_sample_files: Dict[str, Path] = {}

        # Colors for speakers UI
        speaker_ids = sorted(speaker_turns_map.keys())

        for spk_id in speaker_ids:
            spk_turns = speaker_turns_map[spk_id]
            spk_dir = speakers_dir / spk_id
            spk_dir.mkdir(parents=True, exist_ok=True)

            total_spk_time = sum(t.duration_s for t in spk_turns)
            percentage = min(100.0, round((total_spk_time / total_audio_duration * 100.0) if total_audio_duration > 0 else 0.0, 1))

            # Collect audio chunks to stitch master file and find a clean preview sample
            concatenated_chunks = []
            preview_chunk = None
            max_chunk_len = 0

            for idx, turn in enumerate(spk_turns, 1):
                start_frame = max(0, min(len(audio_data), int(turn.start_s * orig_sr)))
                end_frame = max(0, min(len(audio_data), int(turn.end_s * orig_sr)))
                chunk = audio_data[start_frame:end_frame]

                if len(chunk) > 0:
                    concatenated_chunks.append(chunk)

                    # Export individual turn
                    turn_filename = f"turn_{idx:03d}_{turn.start_s:06.2f}-{turn.end_s:06.2f}.wav"
                    turn_path = spk_dir / turn_filename
                    sf.write(str(turn_path), chunk, orig_sr, subtype="PCM_16")

                    turn.turn_filename = turn_filename
                    turn.clip_url = f"/api/diarized/{run_id}/speakers/{spk_id}/{turn_filename}"

                    # Find best representative sample (e.g. 3-6s continuous chunk)
                    dur = turn.end_s - turn.start_s
                    if dur >= 3.0 and dur > max_chunk_len:
                        max_chunk_len = dur
                        preview_chunk = chunk[:int(min(dur, 6.0) * orig_sr)]
                    elif preview_chunk is None and dur >= 1.0:
                        preview_chunk = chunk

            # Write master speaker audio (concatenated voice of this speaker)
            master_filename = f"{spk_id}_full.wav"
            master_path = output_dir / master_filename
            if concatenated_chunks:
                master_audio = np.concatenate(concatenated_chunks, axis=0)
                sf.write(str(master_path), master_audio, orig_sr, subtype="PCM_16")
                speaker_master_files[spk_id] = master_path

            # Write speaker preview sample
            sample_filename = f"{spk_id}_sample.wav"
            sample_path = samples_dir / sample_filename
            if preview_chunk is None and concatenated_chunks:
                preview_chunk = concatenated_chunks[0][:int(min(len(concatenated_chunks[0]) / orig_sr, 5.0) * orig_sr)]
            if preview_chunk is not None:
                sf.write(str(sample_path), preview_chunk, orig_sr, subtype="PCM_16")
                speaker_sample_files[spk_id] = sample_path

            speaker_stats_list.append(SpeakerStats(
                speaker_id=spk_id,
                total_time_s=round(total_spk_time, 2),
                percentage=percentage,
                turn_count=len(spk_turns),
                sample_audio_url=f"/api/diarized/{run_id}/speaker_samples/{sample_filename}" if spk_id in speaker_sample_files else None,
                master_audio_url=f"/api/diarized/{run_id}/{master_filename}" if spk_id in speaker_master_files else None,
            ))

        # Sort speaker stats by speak time descending
        speaker_stats_list.sort(key=lambda s: s.total_time_s, reverse=True)

        elapsed = time.time() - t0
        logger.info("Diarization finished in %.2fs: detected %d speakers, %d turns.", elapsed, len(speaker_stats_list), len(turns))

        # 4. Save metadata sidecar
        metadata_record = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "input_filename": input_audio_path.name,
            "input_path": str(input_audio_path),
            "engine": engine,
            "total_duration_s": round(total_audio_duration, 2),
            "total_duration_formatted": format_duration(total_audio_duration),
            "num_speakers": len(speaker_stats_list),
            "overlap_filtered": filter_overlap,
            "elapsed_seconds": round(elapsed, 2),
            "speakers": [s.to_dict() for s in speaker_stats_list],
            "turns": [t.to_dict() for t in turns],
        }

        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_record, f, ensure_ascii=False, indent=2)

        return DiarizationResult(
            run_id=run_id,
            input_file=input_audio_path,
            engine=engine,
            total_duration_s=total_audio_duration,
            num_speakers=len(speaker_stats_list),
            speakers=speaker_stats_list,
            turns=turns,
            output_dir=output_dir,
            overlap_filtered=filter_overlap,
            metadata_file=meta_path,
        )

    def get_history(self) -> List[dict]:
        """Fetch all previous diarization runs sorted by newest first."""
        history = []
        if not self.base_diarized_dir.is_dir():
            return []

        for run_dir in self.base_diarized_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            meta_file = run_dir / "metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["mtime"] = run_dir.stat().st_mtime
                    history.append(meta)
                except Exception:
                    pass

        history.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return history

    def delete_run(self, run_id: str) -> bool:
        """Delete a diarization run directory."""
        run_dir = self.base_diarized_dir / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
            return True
        return False
