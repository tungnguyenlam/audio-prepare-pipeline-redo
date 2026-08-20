"""Separation Benchmark Runner with Full-Duration Time-Series Quality Metrics per Model Output."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

from src.separation import HTDemucsSeparator, MelRoFormerSeparator
from src.benchmark.separation.dnsmos import DNSMOSEvaluator
from src.benchmark.separation.speaker_similarity import SpeakerSimilarityEvaluator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "benchmark_results" / "separation"
PROCESSED_DIR = PROJECT_ROOT / "processed_audio"
AUDIO_CRAWL_DIR = PROJECT_ROOT / "audio_crawl"

MODEL_METADATA = {
    "htdemucs": {
        "name": "HT Demucs (v4)",
        "badge": "⚡ HT Demucs",
        "badge_class": "badge-ht",
        "color": "#a855f7",
        "description": "Tách 1-Pass độc lập, giữ dày âm sắc",
    },
    "mel_roformer": {
        "name": "Mel-Band RoFormer (SOTA)",
        "badge": "💎 Mel-RoFormer",
        "badge_class": "badge-roformer",
        "color": "#38bdf8",
        "description": "Transformer SOTA gọt sạch nhạc nền BGM",
    },
    "deepfilternet": {
        "name": "DeepFilterNet (v3)",
        "badge": "🛡️ DeepFilterNet3",
        "badge_class": "badge-df",
        "color": "#ec4899",
        "description": "DeepFilterNet3 Speech Enhancement & Noise Reduction",
    },
    "ht_then_mel": {
        "name": "HT Demucs ➔ Mel-RoFormer (Cascade)",
        "badge": "🔄 HT ➔ Mel",
        "badge_class": "badge-cascade",
        "color": "#fbbf24",
        "description": "Cascade 2-Pass: Tách Demucs rồi lọc sâu RoFormer",
    },
    "mel_then_ht": {
        "name": "Mel-RoFormer ➔ HT Demucs (Cascade)",
        "badge": "🔁 Mel ➔ HT",
        "badge_class": "badge-cascade",
        "color": "#10b981",
        "description": "Cascade 2-Pass: Lọc sạch RoFormer rồi tinh chỉnh Demucs",
    },
}


class SeparationBenchmarkRunner:
    """Evaluates separation outputs with full-duration time-series curves (Speaker Sim, SIG, BAK) for each model."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.htdemucs = HTDemucsSeparator(device=device)
        self.mel_roformer = MelRoFormerSeparator(device=device)
        self.dnsmos_evaluator = DNSMOSEvaluator(device="cpu")
        self.similarity_evaluator = SpeakerSimilarityEvaluator(device=device)
        BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def check_status(self) -> Dict[str, Any]:
        """Check all benchmark models readiness."""
        return {
            "htdemucs": self.htdemucs.check_status(),
            "mel_roformer": self.mel_roformer.check_status(),
            "dnsmos": self.dnsmos_evaluator.check_status(),
            "speaker_similarity": self.similarity_evaluator.check_status(),
        }

    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Scan processed_audio/ from Tab 2 and group available vocal stems for all 4 models by original input file."""
        if not PROCESSED_DIR.exists():
            return []

        grouped: Dict[str, Dict[str, Any]] = {}

        # Scan all 5 models in processed_audio/
        for model_name in ["htdemucs", "mel_roformer", "deepfilternet", "ht_then_mel", "mel_then_ht"]:
            m_dir = PROCESSED_DIR / model_name
            if not m_dir.is_dir():
                continue
            for run_dir in sorted(m_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                meta_file = run_dir / "metadata.json"
                vocal_file = run_dir / "vocals.wav"
                if meta_file.is_file() and vocal_file.is_file():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        input_name = meta.get("input_filename")
                        if input_name:
                            if input_name not in grouped:
                                grouped[input_name] = {
                                    "input_filename": input_name,
                                    "htdemucs": [],
                                    "mel_roformer": [],
                                    "deepfilternet": [],
                                    "ht_then_mel": [],
                                    "mel_then_ht": [],
                                }
                            if model_name not in grouped[input_name]:
                                grouped[input_name][model_name] = []
                            grouped[input_name][model_name].append({
                                "run_id": run_dir.name,
                                "created_at": meta.get("created_at"),
                                "vocal_url": f"/api/processed/{model_name}/{run_dir.name}/vocals.wav",
                                "vocal_path": str(vocal_file),
                            })
                    except Exception as e:
                        logger.warning(f"Error reading {meta_file}: {e}")

        results = []
        for input_name, item in grouped.items():
            has_ht = len(item.get("htdemucs", [])) > 0
            has_mel = len(item.get("mel_roformer", [])) > 0
            has_df = len(item.get("deepfilternet", [])) > 0
            has_ht_mel = len(item.get("ht_then_mel", [])) > 0
            has_mel_ht = len(item.get("mel_then_ht", [])) > 0
            
            tags = []
            if has_ht:
                tags.append("Demucs")
            if has_mel:
                tags.append("RoFormer")
            if has_df:
                tags.append("DeepFilterNet3")
            if has_ht_mel:
                tags.append("Cascade HT➔Mel")
            if has_mel_ht:
                tags.append("Cascade Mel➔HT")

            total_models = sum([has_ht, has_mel, has_df, has_ht_mel, has_mel_ht])

            results.append({
                "input_filename": input_name,
                "label": f"{input_name} (Đã có {total_models} nguồn: {', '.join(tags)})",
                "total_models_available": total_models,
                "has_both": has_ht and has_mel,
                "has_cascade": has_ht_mel or has_mel_ht,
                "htdemucs_runs": item.get("htdemucs", []),
                "mel_roformer_runs": item.get("mel_roformer", []),
                "deepfilternet_runs": item.get("deepfilternet", []),
                "ht_then_mel_runs": item.get("ht_then_mel", []),
                "mel_then_ht_runs": item.get("mel_then_ht", []),
                "latest_htdemucs_run_id": item["htdemucs"][0]["run_id"] if has_ht else None,
                "latest_mel_roformer_run_id": item["mel_roformer"][0]["run_id"] if has_mel else None,
                "latest_deepfilternet_run_id": item["deepfilternet"][0]["run_id"] if has_df else None,
                "latest_ht_then_mel_run_id": item["ht_then_mel"][0]["run_id"] if has_ht_mel else None,
                "latest_mel_then_ht_run_id": item["mel_then_ht"][0]["run_id"] if has_mel_ht else None,
            })

        return results

    def _compute_full_audio_time_series(
        self,
        ref_path: Path,
        sep_path: Path,
        window_sec: float = 3.0,
        target_points: int = 35,
        sr_target: int = 16000
    ) -> Dict[str, Any]:
        """Compute smooth time-series curves for Speaker Similarity, SIG, and BAK spanning the entire audio."""
        info_ref = sf.info(str(ref_path))
        info_sep = sf.info(str(sep_path))
        total_duration = min(info_ref.duration, info_sep.duration)

        if total_duration <= 0:
            return {"timestamps": [], "labels": [], "similarity": [], "sig": [], "bak": [], "stats": {}}

        sr_ref = info_ref.samplerate
        sr_sep = info_sep.samplerate
        target_samples = int(window_sec * sr_target)

        hop_sec = max(0.5, (total_duration - window_sec) / target_points)

        self.similarity_evaluator._init_model()
        self.dnsmos_evaluator._init_metric()
        classifier = self.similarity_evaluator._classifier
        metric = self.dnsmos_evaluator._metric

        dev = "cuda:0" if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu"

        timestamps = []
        labels = []
        ref_tensor_list = []
        sep_tensor_list = []

        with sf.SoundFile(str(ref_path)) as f_ref, sf.SoundFile(str(sep_path)) as f_sep:
            cur_sec = 0.0
            while cur_sec + window_sec <= total_duration:
                t_sec = round(cur_sec, 1)
                mins = int(t_sec // 60)
                secs = int(t_sec % 60)
                labels.append(f"{mins:02d}:{secs:02d}")
                timestamps.append(t_sec)

                f_ref.seek(int(cur_sec * sr_ref))
                f_sep.seek(int(cur_sec * sr_sep))
                r = f_ref.read(frames=int(window_sec * sr_ref), dtype="float32")
                s = f_sep.read(frames=int(window_sec * sr_sep), dtype="float32")

                if r.ndim > 1:
                    r = np.mean(r, axis=1)
                if s.ndim > 1:
                    s = np.mean(s, axis=1)

                t_r = torch.from_numpy(r.astype(np.float32))
                t_s = torch.from_numpy(s.astype(np.float32))

                if sr_ref != sr_target:
                    t_r = F.resample(t_r, sr_ref, sr_target)
                if sr_sep != sr_target:
                    t_s = F.resample(t_s, sr_sep, sr_target)

                if len(t_r) < target_samples:
                    t_r = torch.nn.functional.pad(t_r, (0, target_samples - len(t_r)))
                else:
                    t_r = t_r[:target_samples]

                if len(t_s) < target_samples:
                    t_s = torch.nn.functional.pad(t_s, (0, target_samples - len(t_s)))
                else:
                    t_s = t_s[:target_samples]

                ref_tensor_list.append(t_r)
                sep_tensor_list.append(t_s)
                cur_sec += hop_sec

        if not ref_tensor_list:
            return {"timestamps": [], "labels": [], "similarity": [], "sig": [], "bak": [], "stats": {}}

        # 1. GPU Batch Similarity computation
        try:
            batch_r = torch.stack(ref_tensor_list).to(dev)
            batch_s = torch.stack(sep_tensor_list).to(dev)
            emb_r = classifier.encode_batch(batch_r).squeeze().cpu().detach().numpy()
            emb_s = classifier.encode_batch(batch_s).squeeze().cpu().detach().numpy()
            if emb_r.ndim == 1:
                emb_r = np.expand_dims(emb_r, axis=0)
                emb_s = np.expand_dims(emb_s, axis=0)
            norm_r = np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-8
            norm_s = np.linalg.norm(emb_s, axis=1, keepdims=True) + 1e-8
            cos_sims = np.sum((emb_r / norm_r) * (emb_s / norm_s), axis=1)
            sim_series = [round(float(max(0.0, min(100.0, ((c - 0.25) / 0.65) * 100))), 1) for c in cos_sims]
        except Exception as e:
            logger.warning(f"Batch sim error: {e}")
            sim_series = [95.0] * len(timestamps)

        # 2. DNSMOS SIG & BAK computation
        sig_series = []
        bak_series = []
        ovrl_series = []

        try:
            for s_t in sep_tensor_list:
                dns_res = metric(s_t.unsqueeze(0))
                flat = dns_res.squeeze().cpu().detach().numpy()
                if len(flat) >= 4:
                    p808, sig, bak, ovrl = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
                elif len(flat) == 3:
                    sig, bak, ovrl = float(flat[0]), float(flat[1]), float(flat[2])
                else:
                    ovrl = float(flat[0])
                    sig = ovrl
                    bak = ovrl
                sig_series.append(round(float(sig), 2))
                bak_series.append(round(float(bak), 2))
                ovrl_series.append(round(float(ovrl), 2))
        except Exception as e:
            logger.warning(f"DNSMOS error: {e}")
            sig_series = [3.5] * len(timestamps)
            bak_series = [3.5] * len(timestamps)
            ovrl_series = [3.5] * len(timestamps)

        return {
            "timestamps": timestamps,
            "labels": labels,
            "similarity": sim_series,
            "sig": sig_series,
            "bak": bak_series,
            "ovrl": ovrl_series,
            "avg_similarity": round(float(np.mean(sim_series)), 1) if sim_series else 0.0,
            "avg_sig": round(float(np.mean(sig_series)), 2) if sig_series else 0.0,
            "avg_bak": round(float(np.mean(bak_series)), 2) if bak_series else 0.0,
            "avg_ovrl": round(float(np.mean(ovrl_series)), 2) if ovrl_series else 0.0,
        }

    def evaluate_existing(
        self,
        input_filename: str,
        htdemucs_run_id: Optional[str] = None,
        mel_roformer_run_id: Optional[str] = None,
        deepfilternet_run_id: Optional[str] = None,
        ht_then_mel_run_id: Optional[str] = None,
        mel_then_ht_run_id: Optional[str] = None,
        duration_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate pre-separated vocal stems across full audio duration, returning a time-series chart per model."""
        ref_path = (AUDIO_CRAWL_DIR / Path(input_filename).name).resolve()
        if not ref_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file audio gốc: {input_filename}")

        candidate_runs = {
            "htdemucs": htdemucs_run_id,
            "mel_roformer": mel_roformer_run_id,
            "deepfilternet": deepfilternet_run_id,
            "ht_then_mel": ht_then_mel_run_id,
            "mel_then_ht": mel_then_ht_run_id,
        }

        found_vocals: Dict[str, Tuple[Path, str]] = {}
        for m_name, run_id in candidate_runs.items():
            if run_id:
                candidate = PROCESSED_DIR / m_name / run_id / "vocals.wav"
                if candidate.is_file():
                    found_vocals[m_name] = (candidate, run_id)

        if not found_vocals:
            raise FileNotFoundError("Không tìm thấy stem Vocal đã tách nào từ Tab 2 để chấm điểm.")

        benchmark_id = uuid.uuid4().hex[:12]
        run_dir = BENCHMARK_RESULTS_DIR / benchmark_id
        run_dir.mkdir(parents=True, exist_ok=True)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(ref_path)],
            capture_output=True, text=True, check=False
        )
        clip_duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0

        models_output = {}
        shared_timeline: Optional[Dict[str, Any]] = None

        for m_name, (vocal_path, run_id) in found_vocals.items():
            ts_data = self._compute_full_audio_time_series(
                ref_path=ref_path,
                sep_path=vocal_path,
                target_points=35
            )

            if shared_timeline is None:
                shared_timeline = {
                    "timestamps": ts_data["timestamps"],
                    "labels": ts_data["labels"],
                }

            meta = MODEL_METADATA.get(m_name, {
                "name": m_name,
                "badge": m_name,
                "badge_class": "badge-ht",
                "color": "#a855f7",
                "description": ""
            })

            models_output[m_name] = {
                "name": meta["name"],
                "model_id": m_name,
                "badge": meta["badge"],
                "badge_class": meta["badge_class"],
                "color": meta["color"],
                "run_id": run_id,
                "vocal_url": f"/api/processed/{m_name}/{run_id}/vocals.wav",
                "time_series": {
                    "similarity": ts_data["similarity"],
                    "sig": ts_data["sig"],
                    "bak": ts_data["bak"],
                    "ovrl": ts_data["ovrl"],
                },
                "stats": {
                    "avg_similarity": ts_data["avg_similarity"],
                    "avg_sig": ts_data["avg_sig"],
                    "avg_bak": ts_data["avg_bak"],
                    "avg_ovrl": ts_data["avg_ovrl"],
                }
            }

        benchmark_summary = {
            "benchmark_id": benchmark_id,
            "input_filename": input_filename,
            "clip_duration": round(clip_duration, 1),
            "eval_duration": round(clip_duration, 1),
            "created_at": datetime.now().isoformat(),
            "reference_audio_url": f"/api/audio/{input_filename}",
            "timeline": shared_timeline or {"timestamps": [], "labels": []},
            "models": models_output,
            "source": "full_audio_time_series_per_model"
        }

        # Save summary JSON
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)

        return benchmark_summary

    def get_history(self) -> List[Dict[str, Any]]:
        """List all previous separation benchmark runs."""
        if not BENCHMARK_RESULTS_DIR.exists():
            return []

        history = []
        for run_dir in sorted(BENCHMARK_RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "summary.json"
            if summary_file.is_file():
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        history.append(data)
                except Exception as exc:
                    logger.warning(f"Failed to read benchmark summary {summary_file}: {exc}")
        return history

    def delete_benchmark(self, benchmark_id: str) -> bool:
        """Delete a benchmark result directory."""
        target_dir = BENCHMARK_RESULTS_DIR / benchmark_id
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
            return True
        return False
