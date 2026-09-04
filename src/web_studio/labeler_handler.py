"""Dedicated REST route handlers and dataset exporter for the Sample Labeler tab.

Supports loading DiarizationResult objects, previewing turn audio snippets,
assigning multi-class / multi-label quality tags (Accept, Background Noise,
Multi-Speaker, Chopped Word), autosaving label drafts, and exporting physically
self-contained audio datasets with train/val/test splits.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from math import isfinite
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Dict, List, Optional
import zipfile

from aiohttp import web
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn

from src.data_paths import DATA_DIR, REPO_ROOT, portable_data_path, resolve_data_path
from src.diarization.schemas import DiarizationResult, SpeakerTurn
from src.utils.AudioClass import Audio, _sanitize_filename_component
from src.utils.AudioCutter import AudioCutter

logger = logging.getLogger(__name__)

DIARIZATION_RESULTS_DIR = DATA_DIR / "diarization" / "results"
DIARIZATION_PREVIEW_DIR = DATA_DIR / "diarization" / "preview"
LABELS_DIR = DATA_DIR / "diarization" / "labels"
LABELED_DATASETS_DIR = DATA_DIR / "labeled_datasets"
MODELS_DIR = DATA_DIR / "models" / "quality_classifier"

DIARIZATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATION_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = {
    "accept": "Accept",
    "noise": "Contain background noise",
    "multi_speaker": "Contain more than 1 speaker",
    "chopped": "Word being chopped off",
}


def _sanitize_name(name: str) -> str:
    """Sanitize directory or file names to prevent path traversal."""
    cleaned = re.sub(r"[^\w\-.]", "_", name.strip())
    return cleaned.strip("._") or "dataset"


def _source_audio_path(result: DiarizationResult) -> Path | None:
    """Return the resolved source audio path when the snapshot exists."""
    if result.source_audio is None:
        return None
    path = Path(result.source_audio.path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        resolved = path.resolve()
        return resolved if resolved.is_file() else None
    except OSError:
        return None


def _load_draft_labels(result_id: str) -> dict[str, Any]:
    """Load existing label drafts for a given result ID."""
    draft_file = LABELS_DIR / f"{_sanitize_name(result_id)}.json"
    if not draft_file.is_file():
        return {}
    try:
        payload = json.loads(draft_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load draft labels for %s: %s", result_id, exc)
        return {}


def _save_draft_labels(result_id: str, data: dict[str, Any]) -> None:
    """Atomically save label draft for a given result ID."""
    draft_file = LABELS_DIR / f"{_sanitize_name(result_id)}.json"
    temp_file = draft_file.with_suffix(".tmp")
    payload = {
        "result_id": result_id,
        "updated_at": time.time(),
        "labels": data.get("labels", {}),
        "notes": data.get("notes", ""),
    }
    temp_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_file.replace(draft_file)


def _resolve_device(req_device: str | None) -> str:
    """Select requested accelerator device or fallback to best available."""
    if req_device and req_device != "auto":
        return req_device
    if torch.cuda.is_available():
        best_idx = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.get_device_properties(i).total_memory,
        )
        return f"cuda:{best_idx}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _InProcessDataset(torch.utils.data.Dataset):
    """Audio dataset for in-process background training."""

    def __init__(self, df: Any, dataset_dir: Path, target_sr: int = 16000, max_duration_s: float = 12.0):
        self.df = df
        self.dataset_dir = dataset_dir
        self.target_sr = target_sr
        self.max_frames = int(max_duration_s * target_sr)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        wav_path = self.dataset_dir / str(row["audio_path"])
        waveform, sr = sf.read(str(wav_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != self.target_sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

        if len(waveform) > self.max_frames:
            waveform = waveform[:self.max_frames]
        else:
            pad_len = self.max_frames - len(waveform)
            waveform = np.pad(waveform, (0, pad_len))

        targets = torch.tensor([
            float(row.get("has_noise", 0)),
            float(row.get("has_multi_speaker", 0)),
            float(row.get("is_chopped", 0)),
        ], dtype=torch.float32)

        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "targets": targets,
            "sample_id": str(row.get("sample_id", f"sample_{idx}")),
        }


def _train_quality_classifier_worker(
    dataset_dir: Path,
    output_model_dir: Path,
    backbone_id: str,
    finetune_mode: str,
    lr_backbone: float,
    lr_head: float,
    epochs: int,
    batch_size: int,
    device_str: str,
    progress_callback: Any,
    cancel_check: Any,
) -> dict[str, Any]:
    """Execute end-to-end training loop with boundary-aware multi-scale pooling."""
    import pandas as pd
    from transformers import AutoModel

    device = torch.device(device_str)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Missing train.csv in {dataset_dir}")

    train_df = pd.read_csv(train_csv)
    if val_csv.is_file() and val_csv.stat().st_size > 50:
        val_df = pd.read_csv(val_csv)
    else:
        n = len(train_df)
        if n >= 4:
            n_tr = int(round(n * 0.8))
            val_df = train_df.iloc[n_tr:].reset_index(drop=True)
            train_df = train_df.iloc[:n_tr].reset_index(drop=True)
        else:
            val_df = train_df.copy()

    train_ds = _InProcessDataset(train_df, dataset_dir)
    val_ds = _InProcessDataset(val_df, dataset_dir)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    progress_callback(0.05, f"Loading backbone model: {backbone_id}...", {})

    backbone = AutoModel.from_pretrained(backbone_id)
    emb_dim = backbone.config.hidden_size

    if finetune_mode == "frozen":
        for p in backbone.parameters():
            p.requires_grad = False
    elif finetune_mode == "top_layers":
        for p in backbone.parameters():
            p.requires_grad = False
        encoder = getattr(backbone, "encoder", None)
        if encoder and hasattr(encoder, "layers"):
            for layer in encoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
    else:
        for p in backbone.parameters():
            p.requires_grad = True

    backbone = backbone.to(device)

    boundary_frames = 15
    input_dim = emb_dim * 3
    classifier = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 256),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(128, 3),
    ).to(device)

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in classifier.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-2})
    param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": 1e-4})

    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, Any]] = []
    best_val_score = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        if cancel_check():
            raise InterruptedError("Training cancelled by user.")

        backbone.train(finetune_mode != "frozen")
        classifier.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            if cancel_check():
                raise InterruptedError("Training cancelled by user.")
            wavs = batch["waveform"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = backbone(wavs)
            hidden = outputs.last_hidden_state
            k = min(boundary_frames, hidden.shape[1] // 3)
            h_start = hidden[:, :k, :].mean(dim=1)
            h_global = hidden.mean(dim=1)
            h_end = hidden[:, -k:, :].mean(dim=1)
            fused = torch.cat([h_start, h_global, h_end], dim=-1)
            logits = classifier(fused)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone_params) + list(head_params), 1.0)
            optimizer.step()

            train_loss += float(loss.item())
            n_train_batches += 1

        avg_train_loss = train_loss / max(1, n_train_batches)

        backbone.eval()
        classifier.eval()
        val_loss = 0.0
        n_val_batches = 0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                wavs = batch["waveform"].to(device)
                targets = batch["targets"].to(device)
                outputs = backbone(wavs)
                hidden = outputs.last_hidden_state
                k = min(boundary_frames, hidden.shape[1] // 3)
                h_start = hidden[:, :k, :].mean(dim=1)
                h_global = hidden.mean(dim=1)
                h_end = hidden[:, -k:, :].mean(dim=1)
                fused = torch.cat([h_start, h_global, h_end], dim=-1)
                logits = classifier(fused)
                loss = criterion(logits, targets)
                val_loss += float(loss.item())
                n_val_batches += 1

                probs = torch.sigmoid(logits)
                all_preds.append((probs >= 0.5).float().cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / max(1, n_val_batches)
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 3))
        targets_arr = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 3))

        defect_metrics: dict[str, dict[str, float]] = {}
        for idx, col in enumerate(["noise", "multi_speaker", "chopped"]):
            tp = int(np.sum((preds_arr[:, idx] == 1) & (targets_arr[:, idx] == 1)))
            fp = int(np.sum((preds_arr[:, idx] == 1) & (targets_arr[:, idx] == 0)))
            fn = int(np.sum((preds_arr[:, idx] == 0) & (targets_arr[:, idx] == 1)))
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            defect_metrics[col] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

        pred_accept = np.all(preds_arr == 0, axis=1) if len(preds_arr) else np.array([])
        true_accept = np.all(targets_arr == 0, axis=1) if len(targets_arr) else np.array([])
        accept_acc = float(np.mean(pred_accept == true_accept)) if len(pred_accept) else 0.0

        mean_f1 = (defect_metrics["noise"]["f1"] + defect_metrics["multi_speaker"]["f1"] + defect_metrics["chopped"]["f1"]) / 3.0
        val_score = (mean_f1 + accept_acc) / 2.0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "clean_accept_acc": round(accept_acc, 4),
            "noise_f1": defect_metrics["noise"]["f1"],
            "multi_f1": defect_metrics["multi_speaker"]["f1"],
            "chopped_f1": defect_metrics["chopped"]["f1"],
            "mean_f1": round(mean_f1, 4),
        }
        history.append(epoch_record)

        if val_score > best_val_score or epoch == 1:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(classifier.state_dict(), output_model_dir / "best_head.pt")
            if finetune_mode != "frozen":
                torch.save(backbone.state_dict(), output_model_dir / "best_backbone.pt")
            (output_model_dir / "metrics.json").write_text(json.dumps(epoch_record, indent=2) + "\n")
            config_payload = {
                "backbone": backbone_id,
                "finetune_mode": finetune_mode,
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
                "epochs": epochs,
                "batch_size": batch_size,
                "boundary_frames": boundary_frames,
                "best_epoch": best_epoch,
                "best_score": round(best_val_score, 4),
            }
            (output_model_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

        pct = epoch / epochs
        msg = f"Epoch {epoch:02d}/{epochs:02d} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accept Acc: {accept_acc:.1%} | Mean F1: {mean_f1:.2f}"
        progress_callback(pct, msg, {
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "clean_accept_acc": round(accept_acc, 4),
            "metrics": defect_metrics,
            "history": history,
            "best_epoch": best_epoch,
        })

    return {
        "output_dir": str(output_model_dir.resolve()),
        "best_epoch": best_epoch,
        "best_score": round(best_val_score, 4),
        "history": history,
    }


class LabelerRouteHandler:
    """Encapsulates Sample Labeler routes with injected task manager and audio registry."""

    def __init__(self, task_manager: Any, registry: Any) -> None:
        self.task_manager = task_manager
        self.registry = registry

    async def handle_list_results(self, request: web.Request) -> web.Response:
        """List diarization results with labeling progress statistics."""
        del request
        items = []
        for path in DIARIZATION_RESULTS_DIR.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                result_id = str(raw.get("result_id") or path.stem)
                audio_id = str(raw.get("audio_id") or "")
                created_at = float(raw.get("created_at") or path.stat().st_mtime)
                turns = raw.get("turns", [])
                speakers = raw.get("speakers", [])
                
                # Check source file existence
                source_audio = raw.get("source_audio")
                source_available = False
                source_title = None
                source_duration_s = None
                if isinstance(source_audio, dict) and source_audio.get("path"):
                    resolved = resolve_data_path(source_audio["path"])
                    source_available = resolved.is_file()
                    source_title = source_audio.get("title") or audio_id
                    source_duration_s = source_audio.get("duration_s")

                # Check existing labels
                draft = _load_draft_labels(result_id)
                labels_map = draft.get("labels", {})
                labeled_count = len(labels_map)
                
                counts = {"accept": 0, "noise": 0, "multi_speaker": 0, "chopped": 0}
                for entry in labels_map.values():
                    lbl = entry.get("label") if isinstance(entry, dict) else entry
                    if lbl in counts:
                        counts[lbl] += 1

                items.append({
                    "result_id": result_id,
                    "audio_id": audio_id,
                    "title": source_title or audio_id,
                    "created_at": created_at,
                    "turn_count": len(turns),
                    "speaker_count": len(speakers),
                    "duration_s": source_duration_s,
                    "source_available": source_available,
                    "labeled_count": labeled_count,
                    "label_counts": counts,
                    "has_draft": bool(labels_map),
                })
            except Exception as exc:
                logger.warning("Skipping invalid diarization result %s: %s", path, exc)

        items.sort(key=lambda item: item.get("created_at", 0), reverse=True)
        return web.json_response({"results": items, "total": len(items)})

    async def handle_get_session(self, request: web.Request) -> web.Response:
        """Load full DiarizationResult data and associated hand labels."""
        result_id = request.match_info["result_id"]
        result_path = DIARIZATION_RESULTS_DIR / f"{_sanitize_name(result_id)}.json"
        if not result_path.is_file():
            return web.json_response({"error": f"Diarization result not found: {result_id}"}, status=404)

        try:
            result = DiarizationResult.load(result_path)
        except Exception as exc:
            logger.exception("Failed loading DiarizationResult %s", result_id)
            return web.json_response({"error": f"Could not load DiarizationResult: {exc}"}, status=400)

        source_path = _source_audio_path(result)
        source_available = source_path is not None and source_path.is_file()

        turns_payload = []
        for idx, turn in enumerate(result.turns):
            turns_payload.append({
                "index": idx,
                "speaker_id": turn.speaker_id,
                "start_s": turn.start_s,
                "end_s": turn.end_s,
                "duration_s": round(turn.duration_s, 4),
                "overlaps_other_speaker": bool(turn.overlaps_other_speaker),
                "confidence": turn.confidence,
            })

        draft = _load_draft_labels(result_id)

        source_info = None
        if result.source_audio:
            source_info = {
                "source_id": result.source_audio.source_id,
                "title": result.source_audio.title or result.source_audio.source_id,
                "duration_s": result.source_audio.duration_s,
                "sample_rate": result.source_audio.sample_rate,
                "channels": result.source_audio.channels,
                "path": portable_data_path(result.source_audio.path),
                "exists": source_available,
            }

        return web.json_response({
            "result_id": result.result_id,
            "audio_id": result.audio_id,
            "schema_version": result.schema_version,
            "created_at": result.created_at,
            "speakers": [s.speaker_id for s in result.speakers],
            "turn_count": len(result.turns),
            "turns": turns_payload,
            "source_audio": source_info,
            "source_available": source_available,
            "labels": draft.get("labels", {}),
            "draft_notes": draft.get("notes", ""),
            "updated_at": draft.get("updated_at"),
            "valid_labels": VALID_LABELS,
        })

    async def handle_save_session_labels(self, request: web.Request) -> web.Response:
        """Save in-progress label draft for a DiarizationResult."""
        result_id = request.match_info["result_id"]
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        labels = payload.get("labels", {})
        notes = payload.get("notes", "")

        _save_draft_labels(result_id, {"labels": labels, "notes": notes})
        labeled_count = len([k for k, v in labels.items() if v])
        return web.json_response({
            "status": "saved",
            "result_id": result_id,
            "labeled_count": labeled_count,
            "saved_at": time.time(),
        })

    async def handle_preview_turn_audio(self, request: web.Request) -> web.StreamResponse:
        """Cut and stream audio for one turn on-demand."""
        result_id = request.match_info["result_id"]
        try:
            turn_index = int(request.match_info["turn_index"])
        except ValueError:
            return web.Response(text="Invalid turn_index", status=400)

        result_path = DIARIZATION_RESULTS_DIR / f"{_sanitize_name(result_id)}.json"
        if not result_path.is_file():
            return web.Response(text=f"Diarization result not found: {result_id}", status=404)

        try:
            result = DiarizationResult.load(result_path)
            if turn_index < 0 or turn_index >= len(result.turns):
                return web.Response(text="turn_index out of bounds", status=400)
            turn = result.turns[turn_index]
        except Exception as exc:
            return web.Response(text=f"Failed loading turn: {exc}", status=400)

        source_path = _source_audio_path(result)
        if source_path is None or not source_path.is_file():
            return web.Response(text="Source audio file is not available on disk", status=404)

        # Cache preview cut under DIARIZATION_PREVIEW_DIR
        preview_dir = DIARIZATION_PREVIEW_DIR / _sanitize_name(result.result_id)
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_file = preview_dir / f"turn_{turn_index:06d}.wav"

        if not preview_file.is_file():
            cutter = AudioCutter(output_dir=preview_dir)
            try:
                await asyncio.to_thread(
                    cutter.cut,
                    result.source_audio,
                    turn.start_s,
                    turn.end_s,
                    output_path=preview_file,
                )
            except Exception as exc:
                logger.exception("Failed cutting turn audio preview")
                return web.Response(text=f"Could not extract audio turn: {exc}", status=500)

        return web.FileResponse(preview_file)

    async def handle_export_dataset(self, request: web.Request) -> web.Response:
        """Export physically independent audio samples and train/val/test splits."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        result_id = data.get("result_id")
        result_ids = data.get("result_ids") or ([result_id] if result_id else [])
        if not result_ids:
            return web.json_response({"error": "result_id or result_ids required"}, status=400)

        raw_dataset_name = data.get("dataset_name", "").strip() or f"quality_dataset_{int(time.time())}"
        dataset_name = _sanitize_name(raw_dataset_name)

        # Split settings
        split_ratios = data.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
        train_pct = float(split_ratios.get("train", 0.8))
        val_pct = float(split_ratios.get("val", 0.1))
        test_pct = float(split_ratios.get("test", 0.1))
        total_pct = train_pct + val_pct + test_pct
        if total_pct <= 0:
            return web.json_response({"error": "Split ratios must sum to > 0"}, status=400)
        train_ratio = train_pct / total_pct
        val_ratio = val_pct / total_pct
        test_ratio = test_pct / total_pct

        split_strategy = data.get("split_strategy", "grouped_by_source")  # grouped_by_source, stratified, random
        target_sample_rate = data.get("target_sample_rate")  # None = keep native
        export_unlabeled = bool(data.get("include_unlabeled", False))
        labels_override = data.get("labels_override")  # optional in-memory labels for current result

        dest_root = LABELED_DATASETS_DIR / dataset_name
        dest_root.mkdir(parents=True, exist_ok=True)
        audio_dir = dest_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        samples_to_export: list[dict[str, Any]] = []

        for rid in result_ids:
            res_path = DIARIZATION_RESULTS_DIR / f"{_sanitize_name(rid)}.json"
            if not res_path.is_file():
                logger.warning("Result file not found: %s", rid)
                continue
            try:
                res = DiarizationResult.load(res_path)
            except Exception as exc:
                logger.warning("Could not load result %s: %s", rid, exc)
                continue

            source_path = _source_audio_path(res)
            if source_path is None or not source_path.is_file():
                logger.warning("Source audio missing for result %s: %s", rid, res.source_audio)
                continue

            # Resolve labels
            if labels_override and rid == result_id:
                labels_map = labels_override
            else:
                draft = _load_draft_labels(rid)
                labels_map = draft.get("labels", {})

            for turn_idx, turn in enumerate(res.turns):
                turn_key = str(turn_idx)
                label_data = labels_map.get(turn_key)
                
                primary_label: str | None = None
                tags: list[str] = []
                notes: str = ""

                if isinstance(label_data, dict):
                    primary_label = label_data.get("label")
                    tags = label_data.get("tags") or ([primary_label] if primary_label else [])
                    notes = label_data.get("notes", "")
                elif isinstance(label_data, str) and label_data in VALID_LABELS:
                    primary_label = label_data
                    tags = [primary_label]

                if not primary_label and not export_unlabeled:
                    continue

                sample_id = f"{_sanitize_name(res.audio_id)}_{turn.speaker_id}_{int(round(turn.start_s * 1000)):07d}_{int(round(turn.end_s * 1000)):07d}"

                samples_to_export.append({
                    "sample_id": sample_id,
                    "result_id": res.result_id,
                    "audio_id": res.audio_id,
                    "source_path": source_path,
                    "source_audio": res.source_audio,
                    "turn_index": turn_idx,
                    "speaker_id": turn.speaker_id,
                    "start_s": turn.start_s,
                    "end_s": turn.end_s,
                    "duration_s": round(turn.duration_s, 4),
                    "overlaps_other_speaker": bool(turn.overlaps_other_speaker),
                    "primary_label": primary_label or "unlabeled",
                    "tags": tags,
                    "notes": notes,
                })

        if not samples_to_export:
            return web.json_response({
                "error": "No valid labeled samples found to export. Please label turns or check 'Include Unlabeled'."
            }, status=400)

        # Apply splitting strategy
        # Default: grouped_by_source (groups turns by source audio_id so same audio doesn't leak into train & test!)
        rng = random.Random(42)  # Deterministic seed for reproducible splits

        if split_strategy == "grouped_by_source":
            audio_ids = sorted(list({s["audio_id"] for s in samples_to_export}))
            rng.shuffle(audio_ids)
            n_audios = len(audio_ids)
            
            n_train = max(1, int(round(n_audios * train_ratio)))
            n_val = int(round(n_audios * val_ratio))
            if n_train + n_val >= n_audios and n_audios > 1:
                n_train = max(1, n_audios - 2)
                n_val = 1
            
            train_audios = set(audio_ids[:n_train])
            val_audios = set(audio_ids[n_train:n_train + n_val])
            test_audios = set(audio_ids[n_train + n_val:])
            if not test_audios and n_audios >= 3:
                test_audios.add(train_audios.pop())

            for s in samples_to_export:
                if s["audio_id"] in train_audios:
                    s["split"] = "train"
                elif s["audio_id"] in val_audios:
                    s["split"] = "val"
                else:
                    s["split"] = "test"

        elif split_strategy == "stratified":
            # Group by primary label
            by_label: dict[str, list[dict[str, Any]]] = {}
            for s in samples_to_export:
                by_label.setdefault(s["primary_label"], []).append(s)
            
            for lbl, items in by_label.items():
                rng.shuffle(items)
                n = len(items)
                n_tr = int(round(n * train_ratio))
                n_va = int(round(n * val_ratio))
                for i, item in enumerate(items):
                    if i < n_tr:
                        item["split"] = "train"
                    elif i < n_tr + n_va:
                        item["split"] = "val"
                    else:
                        item["split"] = "test"
        else:
            # Random split
            rng.shuffle(samples_to_export)
            n = len(samples_to_export)
            n_tr = int(round(n * train_ratio))
            n_va = int(round(n * val_ratio))
            for i, item in enumerate(samples_to_export):
                if i < n_tr:
                    item["split"] = "train"
                elif i < n_tr + n_va:
                    item["split"] = "val"
                else:
                    item["split"] = "test"

        # Physically cut each audio file to dest_root / "audio" / <sample_id>.wav
        cutter = AudioCutter(output_dir=audio_dir)
        exported_records = []
        errors = []

        for s in samples_to_export:
            sample_id = s["sample_id"]
            wav_filename = f"{sample_id}.wav"
            out_wav = audio_dir / wav_filename
            
            if not out_wav.is_file():
                try:
                    cutter.cut(
                        s["source_audio"],
                        s["start_s"],
                        s["end_s"],
                        output_path=out_wav,
                    )
                except Exception as exc:
                    logger.warning("Failed extracting audio for sample %s: %s", sample_id, exc)
                    errors.append(f"{sample_id}: {exc}")
                    continue

            # If resample requested
            if target_sample_rate:
                try:
                    data, sr = sf.read(str(out_wav))
                    if sr != target_sample_rate:
                        import librosa
                        if data.ndim > 1:
                            data = data.mean(axis=1)  # downmix mono
                        resampled = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
                        sf.write(str(out_wav), resampled, target_sample_rate, subtype="PCM_16")
                except Exception as exc:
                    logger.warning("Could not resample %s: %s", sample_id, exc)

            record = {
                "sample_id": sample_id,
                "audio_path": f"audio/{wav_filename}",
                "absolute_path": str(out_wav.resolve()),
                "primary_label": s["primary_label"],
                "tags": s["tags"],
                "has_noise": "noise" in s["tags"],
                "has_multi_speaker": "multi_speaker" in s["tags"],
                "is_chopped": "chopped" in s["tags"],
                "is_clean_accept": s["primary_label"] == "accept",
                "split": s["split"],
                "duration_s": s["duration_s"],
                "speaker_id": s["speaker_id"],
                "source_audio_id": s["audio_id"],
                "start_s": s["start_s"],
                "end_s": s["end_s"],
                "overlaps_other_speaker": s["overlaps_other_speaker"],
                "notes": s["notes"],
            }
            exported_records.append(record)

        # Write manifest.json
        split_counts: dict[str, dict[str, int]] = {
            "train": {"total": 0, "accept": 0, "noise": 0, "multi_speaker": 0, "chopped": 0, "unlabeled": 0},
            "val": {"total": 0, "accept": 0, "noise": 0, "multi_speaker": 0, "chopped": 0, "unlabeled": 0},
            "test": {"total": 0, "accept": 0, "noise": 0, "multi_speaker": 0, "chopped": 0, "unlabeled": 0},
        }
        for rec in exported_records:
            sp = rec["split"]
            lbl = rec["primary_label"]
            if sp in split_counts:
                split_counts[sp]["total"] += 1
                if lbl in split_counts[sp]:
                    split_counts[sp][lbl] += 1

        manifest = {
            "dataset_name": dataset_name,
            "created_at": time.time(),
            "total_samples": len(exported_records),
            "split_strategy": split_strategy,
            "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "target_sample_rate": target_sample_rate,
            "classes": list(VALID_LABELS.keys()),
            "class_labels": VALID_LABELS,
            "split_summary": split_counts,
            "samples": exported_records,
        }
        (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        # Write dataset.jsonl
        with open(dest_root / "dataset.jsonl", "w", encoding="utf-8") as f_jsonl:
            for rec in exported_records:
                f_jsonl.write(json.dumps(rec) + "\n")

        # Write train.csv, val.csv, test.csv
        csv_headers = [
            "sample_id",
            "audio_path",
            "primary_label",
            "tags",
            "has_noise",
            "has_multi_speaker",
            "is_chopped",
            "is_clean_accept",
            "split",
            "duration_s",
            "speaker_id",
            "source_audio_id",
            "start_s",
            "end_s",
        ]
        for sp in ("train", "val", "test"):
            sp_records = [r for r in exported_records if r["split"] == sp]
            csv_path = dest_root / f"{sp}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
                writer.writeheader()
                for r in sp_records:
                    writer.writerow({
                        "sample_id": r["sample_id"],
                        "audio_path": r["audio_path"],
                        "primary_label": r["primary_label"],
                        "tags": "|".join(r["tags"]),
                        "has_noise": int(r["has_noise"]),
                        "has_multi_speaker": int(r["has_multi_speaker"]),
                        "is_chopped": int(r["is_chopped"]),
                        "is_clean_accept": int(r["is_clean_accept"]),
                        "split": r["split"],
                        "duration_s": r["duration_s"],
                        "speaker_id": r["speaker_id"],
                        "source_audio_id": r["source_audio_id"],
                        "start_s": r["start_s"],
                        "end_s": r["end_s"],
                    })

        # Generate train_classifier.py template in destination directory
        _write_training_script_template(dest_root)

        # Generate README.md
        _write_readme(dest_root, dataset_name, split_counts)

        return web.json_response({
            "status": "success",
            "dataset_name": dataset_name,
            "dataset_path": str(dest_root.resolve()),
            "total_exported": len(exported_records),
            "split_summary": split_counts,
            "errors": errors,
        })

    async def handle_list_datasets(self, request: web.Request) -> web.Response:
        """List previously exported datasets with manifests."""
        del request
        items = []
        for p in LABELED_DATASETS_DIR.iterdir():
            if not p.is_dir():
                continue
            manifest_file = p / "manifest.json"
            if manifest_file.is_file():
                try:
                    man = json.loads(manifest_file.read_text(encoding="utf-8"))
                    items.append({
                        "dataset_name": p.name,
                        "created_at": man.get("created_at"),
                        "total_samples": man.get("total_samples", 0),
                        "split_summary": man.get("split_summary", {}),
                        "path": str(p.resolve()),
                    })
                except Exception:
                    pass
        items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
        return web.json_response({"datasets": items})

    async def handle_download_dataset_zip(self, request: web.Request) -> web.StreamResponse:
        """Package and stream an exported dataset as a zip archive."""
        dataset_name = _sanitize_name(request.match_info["name"])
        dataset_dir = LABELED_DATASETS_DIR / dataset_name
        if not dataset_dir.is_dir():
            return web.Response(text=f"Dataset {dataset_name} not found", status=404)

        zip_path = LABELED_DATASETS_DIR / f"{dataset_name}.zip"
        
        # Build zip if missing or if folder was modified
        if not zip_path.is_file() or zip_path.stat().st_mtime < dataset_dir.stat().st_mtime:
            def _build_zip():
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(dataset_dir):
                        for file in files:
                            full_path = Path(root) / file
                            arcname = full_path.relative_to(dataset_dir.parent)
                            zf.write(full_path, arcname)
            await asyncio.to_thread(_build_zip)

        return web.FileResponse(
            zip_path,
            headers={
                "Content-Disposition": f'attachment; filename="{dataset_name}.zip"',
                "Content-Type": "application/zip",
            },
        )

    async def handle_train_classifier(self, request: web.Request) -> web.Response:
        """Enqueue and launch an end-to-end quality classifier training job."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        raw_dataset_name = data.get("dataset_name", "").strip() or "__current__"
        dataset_name = _sanitize_name(raw_dataset_name) if raw_dataset_name != "__current__" else "__current__"
        result_id = data.get("result_id")

        dataset_dir = LABELED_DATASETS_DIR / dataset_name
        if dataset_name == "__current__" or not dataset_dir.is_dir():
            # Check if we can auto-export from current session
            if not result_id:
                return web.json_response({
                    "error": f"Dataset '{dataset_name}' not found on disk. Please export the dataset first or select a valid session."
                }, status=400)
            
            # Auto-export current result
            auto_dataset_name = f"auto_{_sanitize_name(result_id)}_{int(time.time())}"
            export_payload = {
                "result_id": result_id,
                "dataset_name": auto_dataset_name,
                "split_strategy": "grouped_by_source",
                "split_ratios": {"train": 0.8, "val": 0.2, "test": 0.0},
                "labels_override": data.get("labels_override"),
            }
            # Execute export inline
            mock_req = request.clone(method="POST")
            mock_req._read_bytes = json.dumps(export_payload).encode("utf-8")
            export_resp = await self.handle_export_dataset(mock_req)
            if export_resp.status >= 400:
                return export_resp
            dataset_name = auto_dataset_name
            dataset_dir = LABELED_DATASETS_DIR / dataset_name

        train_csv = dataset_dir / "train.csv"
        if not train_csv.is_file():
            return web.json_response({"error": f"Missing train.csv in {dataset_dir}"}, status=400)

        backbone = str(data.get("backbone", "microsoft/wavlm-base")).strip()
        finetune_mode = str(data.get("finetune_mode", "full")).strip()
        lr_backbone = float(data.get("lr_backbone", 1e-5))
        lr_head = float(data.get("lr_head", 5e-4))
        epochs = max(1, min(100, int(data.get("epochs", 15))))
        batch_size = max(1, min(64, int(data.get("batch_size", 8))))
        device_req = data.get("device", "auto")
        device_str = _resolve_device(device_req)

        run_id = f"run_{int(time.time())}"
        output_model_dir = MODELS_DIR / f"{dataset_name}_{run_id}"

        task_id = self.task_manager.create_task(
            "quality_classifier_train",
            {
                "title": f"Train Classifier: {dataset_name} ({backbone.split('/')[-1]})",
                "dataset_name": dataset_name,
                "backbone": backbone,
                "finetune_mode": finetune_mode,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
                "device": device_str,
                "queue_device": device_str,
                "output_dir": str(output_model_dir),
            },
        )

        loop = asyncio.get_running_loop()
        cancelled_event = threading.Event()
        self.task_manager.set_cancel_callback(task_id, cancelled_event.set)
        logs_buffer: list[str] = []

        def log_msg(msg: str) -> None:
            ts = time.strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            logs_buffer.append(line)
            if len(logs_buffer) > 200:
                logs_buffer.pop(0)

        async def run_training_task() -> None:
            def progress_cb(progress: float, message: str, meta: dict[str, Any]) -> None:
                log_msg(message)
                self.task_manager.update_task(
                    task_id,
                    progress=round(progress, 3),
                    progress_known=True,
                    message=message,
                    data={
                        "epoch": meta.get("epoch", 0),
                        "total_epochs": epochs,
                        "train_loss": meta.get("train_loss"),
                        "val_loss": meta.get("val_loss"),
                        "clean_accept_acc": meta.get("clean_accept_acc"),
                        "metrics": meta.get("metrics"),
                        "history": meta.get("history", []),
                        "best_epoch": meta.get("best_epoch", 0),
                        "logs": list(logs_buffer),
                    },
                )

            def execute() -> dict[str, Any]:
                log_msg(f"Starting training on {device_str} with {backbone} (mode: {finetune_mode})")
                return _train_quality_classifier_worker(
                    dataset_dir=dataset_dir,
                    output_model_dir=output_model_dir,
                    backbone_id=backbone,
                    finetune_mode=finetune_mode,
                    lr_backbone=lr_backbone,
                    lr_head=lr_head,
                    epochs=epochs,
                    batch_size=batch_size,
                    device_str=device_str,
                    progress_callback=progress_cb,
                    cancel_check=cancelled_event.is_set,
                )

            try:
                res_payload = await loop.run_in_executor(None, execute)
                if cancelled_event.is_set():
                    self.task_manager.update_task(
                        task_id,
                        status="cancelled",
                        message="Training cancelled by user.",
                    )
                    return
                log_msg(f"Training successfully completed! Best checkpoint: {output_model_dir}")
                self.task_manager.update_task(
                    task_id,
                    status="completed",
                    progress=1.0,
                    progress_known=True,
                    message="Training Complete! Best model saved.",
                    result=res_payload,
                    data={"logs": list(logs_buffer), **self.task_manager.get_task(task_id).get("data", {})},
                )
            except (asyncio.CancelledError, InterruptedError):
                self.task_manager.update_task(
                    task_id,
                    status="cancelled",
                    message="Training cancelled by user.",
                )
            except Exception as exc:
                logger.exception("Quality classifier training failed")
                log_msg(f"ERROR: {exc}")
                self.task_manager.update_task(
                    task_id,
                    status="failed",
                    error=str(exc),
                    message=f"Training failed: {exc}",
                    data={"logs": list(logs_buffer)},
                )
            finally:
                self.task_manager.set_cancel_callback(task_id, None)

        self.task_manager.enqueue(task_id, run_training_task)
        return web.json_response(
            {"task_id": task_id, "task": self.task_manager.get_task(task_id)},
            status=202,
        )

    async def handle_get_train_status(self, request: web.Request) -> web.Response:
        """Poll the current live status and training metrics for a training task."""
        task_id = request.match_info["task_id"]
        task = self.task_manager.get_task(task_id)
        if not task:
            return web.json_response({"error": f"Task not found: {task_id}"}, status=404)
        return web.json_response({"task": task})

    async def handle_cancel_train(self, request: web.Request) -> web.Response:
        """Cancel an active or queued training task."""
        task_id = request.match_info["task_id"]
        success = self.task_manager.cancel_task(task_id)
        return web.json_response({"task_id": task_id, "cancelled": success})

    async def handle_list_models(self, request: web.Request) -> web.Response:
        """List trained quality classifier checkpoints."""
        del request
        items = []
        for p in MODELS_DIR.iterdir():
            if not p.is_dir():
                continue
            config_file = p / "config.json"
            metrics_file = p / "metrics.json"
            best_head = p / "best_head.pt"
            if best_head.is_file():
                cfg = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
                met = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else {}
                items.append({
                    "model_name": p.name,
                    "path": str(p.resolve()),
                    "created_at": p.stat().st_mtime,
                    "config": cfg,
                    "metrics": met,
                })
        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return web.json_response({"models": items})


def _write_training_script_template(dest_root: Path) -> None:
    """Write an executable train_classifier.py template to the exported dataset folder."""
    script_content = '''#!/usr/bin/env python3
"""Quality Classifier Training Script for TTS Diarization Segments.

Architectural Design:
1. Multi-Label Classification: Targets are [has_noise_or_sfx, has_multi_speaker, is_chopped].
   If all three probabilities are below threshold (e.g. 0.5), sample is deemed 'Accept'.
2. Boundary-Aware Multi-Scale Pooling: Concatenates onset boundary, global average, and offset boundary
   embeddings [H_start, H_global, H_end] so edge truncation (chopped words) is not diluted.
3. End-to-End Fine-Tuning: Supports differential learning rates (e.g., 1e-5 for backbone, 1e-3 for MLP head)
   and optional LoRA adapter mode for small backbone models (WavLM-Base, HuBERT-Base).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoFeatureExtractor

DEFECT_COLUMNS = ["has_noise", "has_multi_speaker", "is_chopped"]


class AudioQualityDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        base_dir: Path,
        target_sr: int = 16000,
        max_duration_s: float = 12.0,
    ):
        self.df = pd.read_csv(csv_file)
        self.base_dir = Path(base_dir)
        self.target_sr = target_sr
        self.max_frames = int(max_duration_s * target_sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = self.base_dir / row["audio_path"]

        waveform, sr = sf.read(str(wav_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # downmix mono

        if sr != self.target_sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

        # Pad / crop
        if len(waveform) > self.max_frames:
            waveform = waveform[: self.max_frames]
        else:
            pad_len = self.max_frames - len(waveform)
            waveform = np.pad(waveform, (0, pad_len))

        # Multi-label binary targets: [has_noise_or_sfx, has_multi_speaker, is_chopped]
        targets = torch.tensor([
            float(row.get("has_noise", 0)),
            float(row.get("has_multi_speaker", 0)),
            float(row.get("is_chopped", 0)),
        ], dtype=torch.float32)

        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "targets": targets,
            "sample_id": str(row["sample_id"]),
        }


class BoundaryAwareAudioClassifier(nn.Module):
    """Audio classifier with boundary-aware multi-scale pooling and MLP head."""

    def __init__(
        self,
        backbone_id: str = "microsoft/wavlm-base",
        boundary_frames: int = 15,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.25,
        finetune_mode: str = "full",  # 'full', 'frozen', or 'lora'
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_id)
        self.boundary_frames = boundary_frames
        emb_dim = self.backbone.config.hidden_size

        if finetune_mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif finetune_mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, peft_config)
            except ImportError:
                print("peft library not installed. Falling back to unfreezing top 2 layers.")
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for p in self.backbone.encoder.layers[-2:].parameters():
                    p.requires_grad = True

        # Tri-pooled input: [H_onset, H_global, H_offset] = 3 * emb_dim
        input_dim = emb_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, wavs):
        # wavs: [Batch, Time]
        outputs = self.backbone(wavs)
        hidden = outputs.last_hidden_state  # [Batch, Frames, Hidden]

        # Multi-scale boundary pooling
        k = min(self.boundary_frames, hidden.shape[1] // 3)
        h_start = hidden[:, :k, :].mean(dim=1)
        h_global = hidden.mean(dim=1)
        h_end = hidden[:, -k:, :].mean(dim=1)

        fused = torch.cat([h_start, h_global, h_end], dim=-1)
        logits = self.classifier(fused)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        wavs = batch["waveform"].to(device)
        targets = batch["targets"].to(device)

        logits = model(wavs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_preds = []
    all_targets = []
    for batch in loader:
        wavs = batch["waveform"].to(device)
        logits = model(wavs)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["targets"].numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Per-class accuracy / recall
    stats = {}
    for i, col in enumerate(DEFECT_COLUMNS):
        tp = np.sum((preds_arr[:, i] == 1) & (targets_arr[:, i] == 1))
        fp = np.sum((preds_arr[:, i] == 1) & (targets_arr[:, i] == 0))
        fn = np.sum((preds_arr[:, i] == 0) & (targets_arr[:, i] == 1))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        stats[col] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # Accept metric: no defect predicted AND no defect in ground truth
    pred_accept = np.all(preds_arr == 0, axis=1)
    true_accept = np.all(targets_arr == 0, axis=1)
    accept_acc = np.mean(pred_accept == true_accept)
    stats["clean_accept"] = {"accuracy": round(float(accept_acc), 4)}
    return stats


def main():
    parser = argparse.ArgumentParser(description="End-to-End Audio Quality Classifier")
    parser.add_argument("--backbone", type=str, default="microsoft/wavlm-base", help="Pretrained HF model ID")
    parser.add_argument("--finetune-mode", type=str, default="full", choices=["full", "lora", "frozen"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Classifier head learning rate")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Backbone: {args.backbone} | Mode: {args.finetune_mode}")

    train_ds = AudioQualityDataset(root / "train.csv", root)
    val_ds = AudioQualityDataset(root / "val.csv", root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Initialize model
    model = BoundaryAwareAudioClassifier(
        backbone_id=args.backbone,
        finetune_mode=args.finetune_mode,
    ).to(device)

    # Differential parameter groups
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer_grouped_params = [
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 1e-2},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Loaded {len(train_ds)} train and {len(val_ds)} val samples.")
    print("Beginning training...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} - Loss: {loss:.4f} | Val: {val_stats}")


if __name__ == "__main__":
    main()
'''
    (dest_root / "train_classifier.py").write_text(script_content, encoding="utf-8")
    try:
        (dest_root / "train_classifier.py").chmod(0o755)
    except Exception:
        pass


def _write_readme(dest_root: Path, dataset_name: str, split_counts: dict[str, Any]) -> None:
    """Write documentation README.md inside exported dataset folder."""
    readme_text = f"""# {dataset_name} — Speech Quality & Diarization Defect Dataset

Physically decoupled dataset curated with SonicStudio Sample Quality Labeler.
All audio files inside `audio/` are independent from source audio files and DiarizationResult objects.

## Label Semantics

| Label Key | Full Name | Semantic Definition |
|---|---|---|
| `accept` | **Accept (Clean Speech)** | High quality single-speaker turn. No noticeable noise, complete words, suitable for TTS. |
| `noise` | **Contain background noise** | Background music, hiss, hum, room noise, or ambient interference. |
| `multi_speaker` | **Contain more than 1 speaker** | Overlap, crosstalk, secondary speaker bleed, or transition cut error. |
| `chopped` | **Word being chopped off** | VAD cut too early or started late; initial or final syllables truncated. |

## Dataset Splits & Summary

- **Split Strategy**: Grouped by Source Audio (Prevents channel/speaker leakage across train and test).
- **Train Total**: {split_counts.get("train", {}).get("total", 0)}
- **Validation Total**: {split_counts.get("val", {}).get("total", 0)}
- **Test Total**: {split_counts.get("test", {}).get("total", 0)}

## Files

- `manifest.json`: Full structured metadata, split distributions, and sample list.
- `dataset.jsonl`: Line-by-line JSON format compatible with HuggingFace `datasets`.
- `train.csv`, `val.csv`, `test.csv`: Standard tables with audio paths and binary indicators (`has_noise`, `has_multi_speaker`, `is_chopped`).
- `audio/`: Dedicated folder of 16-bit PCM WAV segments.
- `train_classifier.py`: Ready-to-run PyTorch training script with frozen backbone + MLP head.
"""
    (dest_root / "README.md").write_text(readme_text, encoding="utf-8")


def register_labeler_routes(
    app: web.Application,
    task_manager: Any,
    registry: Any,
) -> None:
    """Mount all Sample Labeler endpoints to the studio application."""
    handler = LabelerRouteHandler(task_manager, registry)
    app.router.add_get("/api/labeler/results", handler.handle_list_results)
    app.router.add_get("/api/labeler/session/{result_id}", handler.handle_get_session)
    app.router.add_post("/api/labeler/session/{result_id}/labels", handler.handle_save_session_labels)
    app.router.add_get("/api/labeler/results/{result_id}/turns/{turn_index}/audio", handler.handle_preview_turn_audio)
    app.router.add_post("/api/labeler/export-dataset", handler.handle_export_dataset)
    app.router.add_get("/api/labeler/datasets", handler.handle_list_datasets)
    app.router.add_get("/api/labeler/datasets/{name}/download", handler.handle_download_dataset_zip)
    app.router.add_post("/api/labeler/train", handler.handle_train_classifier)
    app.router.add_get("/api/labeler/train/status/{task_id}", handler.handle_get_train_status)
    app.router.add_post("/api/labeler/train/cancel/{task_id}", handler.handle_cancel_train)
    app.router.add_get("/api/labeler/models", handler.handle_list_models)
    logger.info("Mounted dedicated Sample Labeler routes at /api/labeler/*")
