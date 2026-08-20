"""Separation Manager: orchestrates HT Demucs, Mel-RoFormer, and Smart Cached Cascade 2-Pass executions."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.separation.base import BaseSeparator, SeparationError, SeparationResult
from src.separation.htdemucs import HTDemucsSeparator
from src.separation.mel_roformer import MelRoFormerSeparator
from src.separation.deepfilternet import DeepFilterNetSeparator

logger = logging.getLogger(__name__)

VALID_MODELS = {"htdemucs", "mel_roformer", "deepfilternet", "ht_then_mel", "mel_then_ht"}


def format_filesize(size_bytes: int) -> str:
    """Format bytes into readable string (KB, MB, GB)."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f} {units[i]}"


class SeparationManager:
    """Manager for Audio Source Separation models (Single, Denoising and Smart Cached Cascade 2-Pass)."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.htdemucs = HTDemucsSeparator(device=device)
        self.mel_roformer = MelRoFormerSeparator(device=device)
        self.deepfilternet = DeepFilterNetSeparator(device=device)
        self.separators: Dict[str, BaseSeparator] = {
            "htdemucs": self.htdemucs,
            "mel_roformer": self.mel_roformer,
            "deepfilternet": self.deepfilternet,
        }

    def get_models_status(self) -> Dict[str, dict]:
        """Return readiness status for all single, denoising, and cascade models."""
        ht_status = self.htdemucs.check_status()
        mel_status = self.mel_roformer.check_status()
        df_status = self.deepfilternet.check_status()
        both_ready = ht_status.get("available", False) and mel_status.get("available", False)

        return {
            "htdemucs": ht_status,
            "mel_roformer": mel_status,
            "deepfilternet": df_status,
            "ht_then_mel": {
                "available": both_ready,
                "message": "Sẵn sàng (Cascade: HT Demucs ➔ Mel-Band RoFormer)" if both_ready else "Cần cả 2 model sẵn sàng",
                "model": "ht_then_mel"
            },
            "mel_then_ht": {
                "available": both_ready,
                "message": "Sẵn sàng (Cascade: Mel-Band RoFormer ➔ HT Demucs)" if both_ready else "Cần cả 2 model sẵn sàng",
                "model": "mel_then_ht"
            }
        }

    def _find_latest_vocal_for_input(
        self,
        base_output_dir: Path,
        model_name: str,
        input_filename: str
    ) -> Optional[Tuple[Path, str]]:
        """Find the latest pre-separated vocal stem for a given input audio if already processed in Tab 2.
        
        Returns:
            Tuple of (vocal_path, run_id) or None if not found.
        """
        model_dir = base_output_dir / model_name
        if not model_dir.is_dir():
            return None

        candidates = []
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            meta_file = run_dir / "metadata.json"
            vocal_file = run_dir / "vocals.wav"
            if meta_file.is_file() and vocal_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("input_filename") == input_filename:
                        candidates.append((run_dir.stat().st_mtime, vocal_file, run_dir.name))
                except Exception:
                    pass

        if candidates:
            # Sort by newest modification time
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1], candidates[0][2]

        return None

    def run_separation(
        self,
        input_path: Path,
        models: List[str],
        base_output_dir: Path,
        deepfilternet_atten_lim_db: Optional[float] = None,
        deepfilternet_post_filter: Optional[bool] = None,
    ) -> tuple[List[dict], List[dict]]:
        """Run selected separation models (Single or Smart-Cached Cascade 2-Pass).

        Args:
            input_path: Path to input audio
            models: List of model names: ['htdemucs', 'mel_roformer', 'deepfilternet', 'ht_then_mel', 'mel_then_ht']
            base_output_dir: Root output directory (e.g. PROCESSED_DIR)
            deepfilternet_atten_lim_db: Noise attenuation limit in dB for DeepFilterNet
            deepfilternet_post_filter: Enable post-filter for DeepFilterNet

        Returns:
            Tuple of (successful_results, errors)
        """
        input_path = Path(input_path).resolve()
        base_output_dir = Path(base_output_dir).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        successful_results: List[dict] = []
        errors: List[dict] = []

        for model_name in dict.fromkeys(models):  # Deduplicate while preserving order
            if model_name not in VALID_MODELS:
                errors.append({
                    "model": model_name,
                    "message": f"Mô hình không được hỗ trợ: {model_name}"
                })
                continue

            run_id = uuid.uuid4().hex[:12]
            output_dir = base_output_dir / model_name / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            cached_pass1_run_id: Optional[str] = None

            try:
                if model_name == "htdemucs":
                    result = self.htdemucs.separate(input_path, output_dir)
                    vocal_path = result.stems.get("vocals", output_dir / "vocals.wav")

                elif model_name == "mel_roformer":
                    result = self.mel_roformer.separate(input_path, output_dir)
                    vocal_path = result.stems.get("vocals", output_dir / "vocals.wav")

                elif model_name == "deepfilternet":
                    result = self.deepfilternet.separate(
                        input_path,
                        output_dir,
                        atten_lim_db=deepfilternet_atten_lim_db,
                        post_filter=deepfilternet_post_filter,
                    )
                    vocal_path = result.stems.get("vocals", output_dir / "vocals.wav")

                elif model_name == "ht_then_mel":
                    logger.info("Executing Cascade HT Demucs ➔ Mel-Band RoFormer for %s", input_path.name)
                    # Check if HT Demucs was already run for this input audio
                    existing_ht = self._find_latest_vocal_for_input(base_output_dir, "htdemucs", input_path.name)

                    if existing_ht:
                        pass1_vocal_path, cached_run_id = existing_ht
                        cached_pass1_run_id = cached_run_id
                        logger.info("⚡ Tái sử dụng kết quả HT Demucs có sẵn (Run: %s) làm Pass 1 cho Mel-RoFormer", cached_run_id)
                        # Only run Mel-RoFormer Pass 2 directly on the existing Demucs vocal!
                        self.mel_roformer.separate(pass1_vocal_path, output_dir)
                        vocal_path = output_dir / "vocals.wav"
                    else:
                        logger.info("Chưa có bản HT Demucs trước đó. Chạy đầy đủ Pass 1 (HT Demucs) ➔ Pass 2 (Mel-RoFormer)...")
                        temp_stage1_dir = output_dir / "_temp_htdemucs"
                        temp_stage1_dir.mkdir(parents=True, exist_ok=True)
                        res1 = self.htdemucs.separate(input_path, temp_stage1_dir)
                        inter_vocal = res1.stems.get("vocals", temp_stage1_dir / "vocals.wav")
                        
                        if not inter_vocal.is_file():
                            raise SeparationError("HT Demucs Pass 1 không tạo ra file vocals.wav.")

                        self.mel_roformer.separate(inter_vocal, output_dir)
                        vocal_path = output_dir / "vocals.wav"
                        shutil.rmtree(temp_stage1_dir, ignore_errors=True)

                elif model_name == "mel_then_ht":
                    logger.info("Executing Cascade Mel-Band RoFormer ➔ HT Demucs for %s", input_path.name)
                    # Check if Mel-RoFormer was already run for this input audio
                    existing_mel = self._find_latest_vocal_for_input(base_output_dir, "mel_roformer", input_path.name)

                    if existing_mel:
                        pass1_vocal_path, cached_run_id = existing_mel
                        cached_pass1_run_id = cached_run_id
                        logger.info("⚡ Tái sử dụng kết quả Mel-RoFormer có sẵn (Run: %s) làm Pass 1 cho HT Demucs", cached_run_id)
                        # Only run HT Demucs Pass 2 directly on the existing Mel-RoFormer vocal!
                        self.htdemucs.separate(pass1_vocal_path, output_dir)
                        vocal_path = output_dir / "vocals.wav"
                    else:
                        logger.info("Chưa có bản Mel-RoFormer trước đó. Chạy đầy đủ Pass 1 (Mel-RoFormer) ➔ Pass 2 (HT Demucs)...")
                        temp_stage1_dir = output_dir / "_temp_melroformer"
                        temp_stage1_dir.mkdir(parents=True, exist_ok=True)
                        res1 = self.mel_roformer.separate(input_path, temp_stage1_dir)
                        inter_vocal = res1.stems.get("vocals", temp_stage1_dir / "vocals.wav")
                        
                        if not inter_vocal.is_file():
                            raise SeparationError("Mel-Band RoFormer Pass 1 không tạo ra file vocals.wav.")

                        self.htdemucs.separate(inter_vocal, output_dir)
                        vocal_path = output_dir / "vocals.wav"
                        shutil.rmtree(temp_stage1_dir, ignore_errors=True)

                # Ensure only vocals.wav exists in final output dir
                if not vocal_path.is_file():
                    raise SeparationError(f"Không tìm thấy file vocal đầu ra cuối cùng tại {vocal_path}")

                file_size = vocal_path.stat().st_size
                rel_path = vocal_path.relative_to(output_dir).as_posix()

                stems_info = [{
                    "name": "vocals",
                    "filename": vocal_path.name,
                    "relative_path": rel_path,
                    "url": f"/api/processed/{model_name}/{run_id}/{rel_path}",
                    "download_url": f"/api/processed/{model_name}/{run_id}/{rel_path}",
                    "filesize": file_size,
                    "filesize_formatted": format_filesize(file_size),
                }]

                # Write metadata sidecar
                metadata_record = {
                    "model": model_name,
                    "run_id": run_id,
                    "input_filename": input_path.name,
                    "input_path": str(input_path),
                    "created_at": datetime.now().isoformat(),
                    "cached_pass1_run_id": cached_pass1_run_id,
                    "atten_lim_db": deepfilternet_atten_lim_db if model_name == "deepfilternet" else None,
                    "post_filter": deepfilternet_post_filter if model_name == "deepfilternet" else None,
                    "stems": stems_info,
                }
                with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata_record, f, ensure_ascii=False, indent=2)

                successful_results.append({
                    "model": model_name,
                    "run_id": run_id,
                    "created_at": metadata_record["created_at"],
                    "cached_pass1_run_id": cached_pass1_run_id,
                    "stems": stems_info,
                })
            except (SeparationError, FileNotFoundError, ValueError, Exception) as exc:
                logger.exception("Error running separation for %s", model_name)
                errors.append({
                    "model": model_name,
                    "message": str(exc),
                })

        return successful_results, errors

    def get_history(self, base_output_dir: Path) -> List[dict]:
        """List all past separation runs sorted by creation time."""
        base_output_dir = Path(base_output_dir).resolve()
        history: List[dict] = []

        for model_dir in base_output_dir.glob("*"):
            if not model_dir.is_dir() or model_dir.name not in VALID_MODELS:
                continue
            model_name = model_dir.name
            for run_dir in model_dir.glob("*"):
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                run_id = run_dir.name
                metadata_file = run_dir / "metadata.json"
                if metadata_file.is_file():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        history.append(meta)
                        continue
                    except Exception:
                        pass

                # Fallback: scan WAV stems directly
                vocal_file = run_dir / "vocals.wav"
                if vocal_file.is_file():
                    stems_info = [{
                        "name": "vocals",
                        "filename": "vocals.wav",
                        "relative_path": "vocals.wav",
                        "url": f"/api/processed/{model_name}/{run_id}/vocals.wav",
                        "download_url": f"/api/processed/{model_name}/{run_id}/vocals.wav",
                        "filesize": vocal_file.stat().st_size,
                        "filesize_formatted": format_filesize(vocal_file.stat().st_size),
                    }]
                    history.append({
                        "model": model_name,
                        "run_id": run_id,
                        "input_filename": "Unknown",
                        "created_at": datetime.fromtimestamp(vocal_file.stat().st_mtime).isoformat(),
                        "stems": stems_info,
                    })

        # Sort newest first
        history.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return history

    def delete_run(self, model: str, run_id: str, base_output_dir: Path) -> bool:
        """Delete a specific separation run output directory."""
        target_dir = Path(base_output_dir) / model / run_id
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
            return True
        return False
